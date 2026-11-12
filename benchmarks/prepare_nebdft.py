#!/usr/bin/env python
"""
Prepare the nebDFT2k hop barriers, the DFT part of the external check of plan item BT7.

nebDFT2k is one part of the LiTraj collection of Dembitskiy et al., npj Computational
Materials 11, 131 (2025), published under the MIT licence at
https://github.com/AIRI-Institute/LiTraj.  It holds 1,681 hops of one lithium ion into a
vacancy, in 867 crystal structures of the Materials Project.  For every hop the migration
path was relaxed by the nudged elastic band method with density functional theory (DFT, the
PBE functional), and the barrier of that path is the label.  The index of the set also gives
the barrier of the same hop from the bond-valence site-energy method, which is the cheap
empirical estimate that `prepare_litraj.py` and item BT7 have used so far.

So this is the first set on which the hop models of the program can be fitted on barriers
from a quantum calculation.

What the script does
--------------------
It reads the index of the archive, fetches the structures from the OPTIMADE interface of the
Materials Project (which needs no key), and writes every hop with its two barriers and the
part of the published split it belongs to.  A hop is named by two atom indices in the
Materials Project structure.  That reading of the index is checked for every hop against the
trajectory file of the hop: the distance that the moving ion covers in the trajectory has to
be the length of the hop in the structure.  A hop that fails the check is left out and
counted.

The archive is 69 MB (0.7 GB unpacked).  It is downloaded into ``benchmarks/cache/litraj``,
which is not in the repository, and its members are read from the archive without unpacking
it.  The prepared file is small and is written into the repository, as the licence allows, so
the benchmark runs without the network.

Usage
-----
    python benchmarks/prepare_nebdft.py --fetch      # download, fetch structures, write
    python benchmarks/prepare_nebdft.py              # reuse what is already downloaded
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.periodic import oxidation_states  # noqa: E402
import prepare_litraj as LITRAJ  # noqa: E402

ARCHIVE = "https://bs3u.obs.ru-moscow-1.hc.sbercloud.ru/litraj/nebDFT2k.zip"
SHA256 = "b7a99d89337902e9e1da319f57547170fdb132bf15bf6ffef03a0140e2207d7f"
ZIP = os.path.join(LITRAJ.CACHE, "nebDFT2k_download", "nebDFT2k.zip")
STRUCTURES = os.path.join(LITRAJ.CACHE, "mp_structures_nebdft.json")
OUT = os.path.join(HERE, "results", "nebdft", "prepared.json")
INDEX_MEMBER = "nebDFT2k/nebDFT2k_index.csv"
#: the hop of the structure and the path of the trajectory may differ by this much, in A: the
#: trajectory is in a cell that was relaxed again
LENGTH_TOLERANCE = 0.5


def download(path: str = ZIP) -> str:
    """Download the archive when it is not in the cache; returns its path."""
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {ARCHIVE} (69 MB, MIT licence)", flush=True)
    req = urllib.request.Request(ARCHIVE, headers={"User-Agent": "cmfqsar-benchmark"})
    with urllib.request.urlopen(req, timeout=900) as r, open(path + ".part", "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(path + ".part", path)
    return path


def open_archive(path: str = ZIP, want: str = SHA256) -> zipfile.ZipFile:
    """The archive, after its SHA-256 sum was compared with `want` (not compared when empty)."""
    if want:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            raise SystemExit(f"{path} has the SHA-256 sum {h.hexdigest()}; the script was "
                             f"written for {want}.  The set has changed.")
    return zipfile.ZipFile(path)


def read_index(z: zipfile.ZipFile, member: str = INDEX_MEMBER) -> List[dict]:
    """The rows of the index: one hop each, with its two barriers as numbers."""
    rows = list(csv.DictReader(io.StringIO(z.read(member).decode("utf-8"))))
    for r in rows:
        r["em_dft"], r["em_bvse"] = float(r["em_dft"]), float(r["em_bvse"])
    return rows


def path_length(text: str) -> Optional[float]:
    """The distance between the first and the last place of the moving ion in a trajectory
    file (extended XYZ with several images), or None when the file cannot be read.

    The moving ion is the atom that is furthest from its first place in the last image.  The
    images of one path are in one cell and the ion is not wrapped between them, so the plain
    distance is the length of the path from end to end.
    """
    lines = text.splitlines()
    try:
        n = int(lines[0])
        frames = len(lines) // (n + 2)
        if frames < 2:
            return None

        def places(k: int) -> np.ndarray:
            block = lines[k * (n + 2) + 2:(k + 1) * (n + 2)]
            return np.array([[float(v) for v in row.split()[1:4]] for row in block])

        moved = np.linalg.norm(places(frames - 1) - places(0), axis=1)
    except (ValueError, IndexError):
        return None
    return float(moved.max())


def hop_of(edge_id: str, symbols: List[str], cell: np.ndarray,
           positions: np.ndarray) -> Optional[Tuple[int, int, Tuple[int, int, int], float]]:
    """The two atoms, the cell offset and the length of a hop in a structure, or None when
    the indices of the hop do not name two lithium atoms of the structure."""
    a, b, off = LITRAJ.parse_edge(edge_id)
    if max(a, b) >= len(symbols) or symbols[a] != "Li" or symbols[b] != "Li":
        return None
    vec = positions[b] + np.asarray(off, float) @ cell - positions[a]
    return a, b, off, float(np.linalg.norm(vec))


def prepare(rows: List[dict], have: Dict[str, dict], lengths: Dict[str, Optional[float]]
            ) -> Tuple[List[dict], List[list], Dict[str, int]]:
    """The structures with their hops, the structures left out, and how many hops were left
    out for which reason."""
    by_id: Dict[str, List[dict]] = {}
    for r in rows:
        by_id.setdefault(r["material_id"], []).append(r)
    out, skipped = [], []
    dropped = {"not two lithium atoms": 0, "length differs from the trajectory": 0,
               "structure left out": 0}
    for mp_id in sorted(by_id):
        row = have.get(mp_id)
        reason = None
        if row is None:
            reason = "not in the Materials Project answer"
        else:
            symbols = list(row["symbols"])
            ox = oxidation_states(symbols)
            if ox is None or ox.get("Li", 0) != 1:
                reason = "no oxidation states"
        if reason:
            skipped.append([mp_id, reason])
            dropped["structure left out"] += len(by_id[mp_id])
            continue
        cell = np.asarray(row["cell"], float)
        pos = np.asarray(row["positions"], float)
        hops = []
        for r in by_id[mp_id]:
            found = hop_of(r["edge_id"], symbols, cell, pos)
            if found is None:
                dropped["not two lithium atoms"] += 1
                continue
            a, b, off, length = found
            path = lengths.get(r["edge_id"])
            if path is not None and abs(path - length) > LENGTH_TOLERANCE:
                dropped["length differs from the trajectory"] += 1
                continue
            hops.append({"edge_id": r["edge_id"], "a": a, "b": b, "offset": list(off),
                         "length": round(length, 4), "y": r["em_dft"],
                         "bvse": r["em_bvse"], "split": str(r["_split"])})
        if not hops:
            skipped.append([mp_id, "no usable hop"])
            continue
        out.append({"id": mp_id, "formula": row["formula"], "nsites": row["nsites"],
                    "cell": row["cell"], "positions": row["positions"], "symbols": symbols,
                    "oxidation": ox, "hops": hops})
    return out, skipped, dropped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the archive when missing")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    if args.fetch:
        download()
    if not os.path.isfile(ZIP):
        raise SystemExit(f"no archive at {ZIP}; run with --fetch to download it (69 MB)")
    z = open_archive()
    rows = read_index(z)
    y = np.array([r["em_dft"] for r in rows])
    ids = sorted({r["material_id"] for r in rows})
    print(f"{len(rows)} hops of {len(ids)} structures and "
          f"{len({r['chemsys'] for r in rows})} chemical systems; DFT barriers mean "
          f"{y.mean():.3f}, standard deviation {y.std(ddof=1):.3f}, range {y.min():.3f} to "
          f"{y.max():.3f} eV", flush=True)
    lengths = {}
    for r in rows:
        try:
            text = z.read(f"nebDFT2k/{r['edge_id']}_init.xyz").decode("utf-8")
        except KeyError:
            lengths[r["edge_id"]] = None
            continue
        lengths[r["edge_id"]] = path_length(text)
    print(f"trajectories read for {sum(v is not None for v in lengths.values())} hops",
          flush=True)
    have = LITRAJ.fetch_structures(ids, cache=STRUCTURES)
    out, skipped, dropped = prepare(rows, have, lengths)
    kept = [h for s in out for h in s["hops"]]
    yk = np.array([h["y"] for h in kept])
    print(f"{len(out)} structures and {len(kept)} hops kept; {len(skipped)} structures left "
          f"out; hops left out: {dropped}", flush=True)
    doc = {"source": LITRAJ.SOURCE, "dataset": "nebDFT2k", "archive": ARCHIVE,
           "reference": "Dembitskiy et al., npj Comput. Mater. 11, 131 (2025)",
           "licence": "MIT", "structures_source": LITRAJ.BASE,
           "prepared": time.strftime("%Y-%m-%d"),
           "n_index_hops": len(rows), "n_index_structures": len(ids),
           "n_index_chemsys": len({r["chemsys"] for r in rows}),
           "index_y_mean": float(y.mean()), "index_y_sd": float(y.std(ddof=1)),
           "kept_y_mean": float(yk.mean()), "kept_y_sd": float(yk.std(ddof=1)),
           "length_tolerance": LENGTH_TOLERANCE,
           "structures": out, "skipped": skipped, "dropped_hops": dropped}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(doc, fh, indent=1)
        fh.write("\n")
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
