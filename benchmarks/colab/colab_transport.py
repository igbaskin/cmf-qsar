#!/usr/bin/env python
"""
Run boxes of the transport benchmark (plan item BT9) on another machine, for example Google Colab.

What this does
--------------
``benchmarks/run_diffusion.py --stage simulate`` runs one box after the other and keeps every
finished box in a cache folder. On a machine whose files disappear when the session ends, that
is not enough. This script runs the same boxes with the same code and settings, and after every
box it copies the finished box to a **store** folder (on Colab: a folder on Google Drive). A box
that is in the store is never run again, so a new session continues where the last one stopped.

* A box is copied under a temporary name and renamed when the copy is complete. A box that was
  cut off while it was being copied is therefore never taken for a finished one.
* No new box is started if it is unlikely to finish within ``--max-hours``. The time of the
  longest box so far, plus a fifth, is the estimate. The first box has no estimate.
* After every box one line is added to ``colab_log.jsonl`` in the store: the graphics card, the
  time in all, the production time and the speed in ns/day. The speed is what decides how many
  compute units a box costs.
* The order is reversed by default. The boxes on the author's desktop are run from the front of
  the list, so a run from the back of the list does not meet them.

Usage
-----
    python benchmarks/colab/colab_transport.py --store /content/drive/MyDrive/cmf_colab/transport
    python benchmarks/colab/colab_transport.py --store STORE --runs form_EMC100,form_DMC100
    python benchmarks/colab/colab_transport.py --store STORE --max-boxes 1     # a timing run
    python benchmarks/colab/colab_transport.py --store STORE --dry-run         # only the plan

``--runs`` is ``transfer`` (the boxes of other solvents, ``FORMULATION_NAMES`` of
``run_diffusion.py``) or run names separated by commas. The result is read by copying the box
folders from the store into ``benchmarks/cache/transport`` and running
``python benchmarks/run_diffusion.py --stage analyse``.

The simulation needs OpenMM with a GPU platform, packmol and AmberTools. The notebook next to
this file installs them. This script itself uses only the standard library, and imports
``run_diffusion`` when it needs the list of runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

LOG_NAME = "colab_log.jsonl"
#: the share of the longest box that is added to its time when the next box is judged
SAFETY = 1.2


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def box_dirname(name: str) -> str:
    """The folder name of one run, the same in the cache and in the store."""
    import run_diffusion as R
    return os.path.basename(R.run_dir(name, ""))


def is_done(directory: str) -> bool:
    """A box is finished when its record is there."""
    return os.path.isfile(os.path.join(directory, "run.json"))


def names_for(spec: str) -> List[str]:
    """The run names for `spec`: ``transfer`` or names separated by commas."""
    import run_diffusion as R
    if spec == "transfer":
        return list(R.FORMULATION_NAMES)
    names = [n for n in spec.split(",") if n]
    unknown = [n for n in names if n not in R.RUNS]
    if unknown:
        raise SystemExit(f"unknown run name(s): {', '.join(unknown)}. Use `transfer` or names "
                         f"from RUNS in benchmarks/run_diffusion.py, separated by commas")
    return names


def plan(names: Sequence[str], store: str, reverse: bool = True) -> Tuple[List[str], List[str]]:
    """(to run, already in the store), both in the order in which they are taken."""
    ordered = list(reversed(names)) if reverse else list(names)
    todo, done = [], []
    for n in ordered:
        (done if is_done(os.path.join(store, box_dirname(n))) else todo).append(n)
    return todo, done


def copy_box(src: str, dst: str) -> None:
    """Copy a finished box to `dst` so that `dst` is either complete or not there."""
    part = dst + ".partial"
    if os.path.isdir(part):
        shutil.rmtree(part)
    shutil.copytree(src, part)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.replace(part, dst)


def gpu_name() -> str:
    """The name of the graphics card, or an empty text when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout else ""
    except (OSError, subprocess.SubprocessError, IndexError):
        return ""


def run_boxes(names: Sequence[str], cache: str, store: str, max_hours: float = 0.0,
              max_boxes: int = 0, reverse: bool = True,
              simulate: Optional[Callable[[str], None]] = None,
              clock: Callable[[], float] = time.time, gpu: Optional[str] = None
              ) -> Tuple[List[dict], List[str]]:
    """Run the boxes that are not in the store; returns (what was run, what is left).

    `simulate` takes a run name and makes its box in `cache`; the default is
    ``run_diffusion.stage_simulate``. `max_hours` and `max_boxes` of 0 mean no limit.
    """
    import run_diffusion as R
    if simulate is None:
        def simulate(name: str) -> None:
            R.stage_simulate([name], cache)
    os.makedirs(store, exist_ok=True)
    os.makedirs(cache, exist_ok=True)
    todo, done = plan(names, store, reverse)
    for n in done:
        log(f"{n}: already in the store, left out")
    card = gpu_name() if gpu is None else gpu
    start = clock()
    longest = 0.0
    results: List[dict] = []
    for name in todo:
        if max_boxes and len(results) >= max_boxes:
            log(f"stopped: {max_boxes} box(es) were asked for and are done")
            break
        elapsed = clock() - start
        if max_hours and elapsed + SAFETY * longest > max_hours * 3600.0:
            log(f"stopped: another box would not finish within {max_hours:g} hours "
                f"({elapsed / 3600.0:.2f} hours used, the longest box took "
                f"{longest / 3600.0:.2f})")
            break
        t0 = clock()
        simulate(name)
        wall = clock() - t0
        longest = max(longest, wall)
        src = R.run_dir(name, cache)
        with open(os.path.join(src, "run.json"), encoding="utf-8") as fh:
            rec = json.load(fh)
        copy_box(src, os.path.join(store, box_dirname(name)))
        entry = {"run": name, "gpu": card, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "wall_s": round(wall, 1), "production_s": rec.get("seconds"),
                 "equilibration_s": rec.get("equilibration_seconds"),
                 "ns_per_day": rec.get("ns_per_day"), "atoms": rec.get("atoms"),
                 "pairs": rec.get("pairs"), "production_ps": rec.get("production_ps")}
        with open(os.path.join(store, LOG_NAME), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        results.append(entry)
        log(f"{name}: in the store after {wall / 60.0:.1f} minutes "
            f"({rec.get('ns_per_day', float('nan')):.0f} ns/day)")
    ran = {r["run"] for r in results}
    return results, [n for n in todo if n not in ran]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--runs", default="transfer",
                    help="`transfer` or run names separated by commas (default: transfer)")
    ap.add_argument("--store", required=True,
                    help="the folder that keeps the finished boxes, for example on Google Drive")
    ap.add_argument("--cache", default=os.path.join(ROOT, "benchmarks", "cache", "transport"),
                    help="where boxes are made (a local disk is faster than Google Drive)")
    ap.add_argument("--max-hours", type=float, default=0.0,
                    help="start no box that is unlikely to finish within this time (0: no limit)")
    ap.add_argument("--max-boxes", type=int, default=0,
                    help="run at most this many boxes (0: no limit)")
    ap.add_argument("--order", choices=("reverse", "forward"), default="reverse",
                    help="the order of the list of runs (default: reverse)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print which boxes would be run and stop")
    args = ap.parse_args(argv)
    names = names_for(args.runs)
    reverse = args.order == "reverse"
    if args.dry_run:
        todo, done = plan(names, args.store, reverse)
        log(f"{len(names)} runs: {len(done)} already in the store, {len(todo)} to run")
        for n in todo:
            log(f"  to run: {n}")
        return 0
    results, left = run_boxes(names, args.cache, args.store, args.max_hours, args.max_boxes,
                              reverse)
    hours = sum(r["wall_s"] for r in results) / 3600.0
    log(f"done: {len(results)} box(es) in {hours:.2f} hours; {len(left)} left in the list")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
