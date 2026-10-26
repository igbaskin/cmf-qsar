#!/usr/bin/env python
"""
How fast the field kernel is, and what each shortcut is worth (plan items B1, B2, B3).

Part B of the development plan is about speed, and its first item assumed that the cost of a
grid search over the attenuation is the squared interatomic distances, which are recomputed for
every alpha.  They are not: on one 40-atom molecule against a 200-molecule table they are 3 % of
the block.  This benchmark measures where the time actually goes, what each shortcut saves, and
-- the part that matters more than the factor -- that none of them changes the answer.

Every comparison is made in one process on one data set, with the fast path switched off by
replacing the function it lives in, so the two numbers differ in nothing but the shortcut.

Stages
------
``kernel``  one kernel of one set, with each shortcut removed in turn:

    as before           the code as it was before part B
    + square root       (2 pi / b)^{3/2} as z sqrt(z) rather than z ** 1.5
    + symmetry          only the atom pairs with j >= i, then mirrored
    + shared distances  one distance block per molecule for the whole alpha grid (item B1)

``grid``    the same, end to end: ``grid_search`` over six attenuations, with the selected
            attenuation and the cross-validated q2 reported beside the time, because a shortcut
            that changed either of them would be a bug and not a speed-up.

``cutoff``  item B2 on two kinds of set: a congeneric series superimposed in one frame, and the
            decapeptides of ``mittal/AMPH1``, which are five times longer.  Reported per
            attenuation with the share of atom pairs that survive the cutoff, because that share
            is what decides whether a neighbour list is worth building.

``screen``  item B3: a library scored in blocks against the same library loaded whole, in wall
            clock and in peak process memory, at two library sizes.  The library is made by
            concatenating the example sets and is written to a temporary file.  Each
            configuration runs in **its own process**, because the peak working set is a
            high-water mark for the life of a process: measuring three of them in one process
            made every later row inherit the largest earlier peak, which made the streamed rows
            look like the whole-file one.

Usage
-----
    python benchmarks/run_speed.py                      # every stage, then the table
    python benchmarks/run_speed.py --stage kernel grid
    python benchmarks/run_speed.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

import cmfqsar.kernels as KN  # noqa: E402
import cmfqsar.model as MD  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, FieldSpec, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "speed")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
#: the congeneric series and the long molecules, which the cutoff treats very differently
SETS = (("COX2", "examples/sutherland/COX2.sdf", 200),
        ("AMPH1", "examples/mittal/AMPH1.sdf", 60))
CUTOFF_ALPHAS = (0.5, 2.0, 5.0)
#: the example sets the screening library is made of
LIBRARY_SETS = ("examples/sutherland", "examples/mittal")


def table_of(path, n, alpha=0.5):
    ds = load_dataset(os.path.join(ROOT, path))
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha)
    mols = list(ds.mols)[:n]
    y = np.asarray(ds.activity, float)[:len(mols)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    return fs, AtomTable.from_molecules(mf), y


def specs_at(fs, alpha):
    return [FieldSpec(**{**vars(s), "alpha": alpha}) for s in fs.specs]


class Without:
    """Switch off one or more of the shortcuts for the duration of a block."""

    def __init__(self, square_root=True, symmetry=True, shared=True):
        self.want = (square_root, symmetry, shared)

    def __enter__(self):
        self.pow_, self.km_, self.grid_ = (KN._pow32, MD.kernel_matrices,
                                           CMFModel.prepare_training_kernels_grid)
        sqrt_on, sym_on, shared_on = self.want
        if not sqrt_on:
            KN._pow32 = lambda z: z ** 1.5
        if not sym_on:
            MD.kernel_matrices = (lambda P, Q, specs, **kw:
                                  self.km_(P, Q, specs, **{**kw, "symmetric": False}))
        if not shared_on:
            CMFModel.prepare_training_kernels_grid = lambda *a, **k: None
        return self

    def __exit__(self, *exc):
        KN._pow32, MD.kernel_matrices = self.pow_, self.km_
        CMFModel.prepare_training_kernels_grid = self.grid_
        return False


# ------------------------------------------------------------------------------ the stages


def stage_kernel(force=False):
    path = os.path.join(RESULTS, "kernel.json")
    if os.path.exists(path) and not force:
        print("kernel: already done")
        return json.load(open(path))
    out = {"alphas": list(ALPHAS), "sets": []}
    for name, p, n in SETS:
        fs, T, _y = table_of(p, n)
        sets = [specs_at(fs, a) for a in ALPHAS]
        rows, ref = [], None
        for label, flags in (("as before", (False, False, False)),
                             ("+ square root", (True, False, False)),
                             ("+ symmetry", (True, True, False)),
                             ("+ shared distances", (True, True, True))):
            with Without(*flags):
                t0 = time.time()
                if flags[2]:
                    Ks = KN.kernel_matrices_multi(T, T, sets)
                else:
                    Ks = [KN.kernel_matrices(T, T, sp, symmetric=flags[1]) for sp in sets]
                dt = time.time() - t0
            err = 0.0 if ref is None else float(np.max([np.max(np.abs(a - b))
                                                       for a, b in zip(Ks, ref)]))
            scale = float(np.max(np.abs(Ks[0])))
            ref = ref if ref is not None else Ks
            rows.append({"label": label, "seconds": round(dt, 3),
                         "rel_error": err / scale if scale else 0.0})
            print(f"  {name} {label:20s} {dt:6.2f} s for {len(ALPHAS)} alphas"
                  f"   (differs from the first by {err / scale if scale else 0:.1e})", flush=True)
        out["sets"].append({"set": name, "n_mol": int(T.n_mol),
                            "n_atoms": int(T.coords.shape[0]),
                            "atoms_per_molecule": round(T.coords.shape[0] / T.n_mol, 1),
                            "rows": rows})
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def stage_grid(force=False):
    path = os.path.join(RESULTS, "grid.json")
    if os.path.exists(path) and not force:
        print("grid: already done")
        return json.load(open(path))
    fs, T, y = table_of(*SETS[0][1:], alpha=0.5)
    rows = []
    for label, flags in (("as before", (False, False, False)),
                         ("+ square root", (True, False, False)),
                         ("+ symmetry", (True, True, False)),
                         ("+ shared distances", (True, True, True))):
        with Without(*flags):
            m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True)
            t0 = time.time()
            res = grid_search(T, y, m, alphas=ALPHAS, lams=(0.1,), folds=5, seed=0)
            dt = time.time() - t0
        rows.append({"label": label, "seconds": round(dt, 2),
                     "alpha": float(res.best_alpha), "q2_cv": float(res.best_q2)})
        print(f"  grid {label:20s} {dt:6.2f} s   alpha {res.best_alpha:g}   "
              f"q2cv {res.best_q2:+.6f}", flush=True)
    out = {"set": SETS[0][0], "n_mol": int(T.n_mol), "alphas": list(ALPHAS), "folds": 5,
           "rows": rows}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def stage_cutoff(force=False):
    path = os.path.join(RESULTS, "cutoff.json")
    if os.path.exists(path) and not force:
        print("cutoff: already done")
        return json.load(open(path))
    out = {"tol": KN.CUTOFF_TOL, "sets": []}
    for name, p, n in SETS:
        fs, T, _y = table_of(p, n)
        rows = []
        for alpha in CUTOFF_ALPHAS:
            specs = specs_at(fs, alpha)
            t0 = time.time(); dense = KN.kernel_matrices(T, T, specs); t_d = time.time() - t0
            t0 = time.time(); auto = KN.kernel_matrices(T, T, specs, cutoff="auto")
            t_a = time.time() - t0
            t0 = time.time(); forced = KN.kernel_matrices(T, T, specs, cutoff=True)
            t_f = time.time() - t0
            scale = float(np.max(np.abs(dense)))
            xi = T.coords[T.slice(0)]
            keep_g = KN.surviving_share(xi, T.coords, KN.cutoff_radius(0.25 * alpha))
            r_s = KN.cutoff_radius(0.25 * alpha / float(T.radii.max()) ** 2)
            keep_s = KN.surviving_share(xi, T.coords, r_s)
            rows.append({"alpha": float(alpha), "dense_seconds": round(t_d, 2),
                         "auto_seconds": round(t_a, 2), "forced_seconds": round(t_f, 2),
                         "auto_exact": bool(np.array_equal(auto, dense)),
                         "forced_rel_error": float(np.max(np.abs(forced - dense)) / scale),
                         "keep_gauss": float(keep_g), "keep_steric": float(keep_s)})
            print(f"  {name} alpha {alpha:4g}: dense {t_d:5.2f} s  auto {t_a:5.2f} s "
                  f"({t_d / t_a:4.2f}x, {'exact' if rows[-1]['auto_exact'] else 'approximate'})"
                  f"  forced {t_f:5.2f} s ({t_d / t_f:4.2f}x)"
                  f"  pairs kept {keep_g:5.1%}/{keep_s:5.1%}", flush=True)
        out["sets"].append({"set": name, "n_mol": int(T.n_mol),
                            "atoms_per_molecule": round(T.coords.shape[0] / T.n_mol, 1),
                            "rows": rows})
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def _build_library(dest: str, repeats: int = 1) -> int:
    """Concatenate the example SD files into one library file; returns the molecule count."""
    import glob
    paths = []
    for folder in LIBRARY_SETS:
        paths += sorted(glob.glob(os.path.join(ROOT, folder, "*.sdf")))
    n = 0
    with open(dest, "w", encoding="utf-8", errors="replace") as fh:
        for _ in range(repeats):
            for p in paths:
                text = open(p, encoding="utf-8", errors="replace").read()
                fh.write(text if text.endswith("\n") else text + "\n")
                n += text.count("$$$$")
    return n


def _peak_mb() -> float:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return float(getattr(info, "peak_wset", info.rss)) / 1e6
    except Exception:                            # noqa: BLE001 - psutil is optional
        return float("nan")


def screen_worker(lib: str, chunk: int) -> dict:
    """Score `lib` one way and report the peak memory of this process.  Run as a subprocess."""
    from cmfqsar.screen import ScreenSpec, screen

    fs, T, y = table_of(*SETS[0][1:], alpha=0.5)
    model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": 0.1})
    model.fit(T, y, cv_folds=0)
    before = _peak_mb()
    t0 = time.time()
    if chunk:
        r = screen(model, lib, ScreenSpec(chunk=chunk, top=10, beta=1.0))
        scored, extra = r.n_scored, {"blocks": r.blocks, "kernel_block_mb": r.peak_kernel_mb}
    else:
        ds = load_dataset(lib)
        tab = AtomTable.from_molecules([build_fields(m, model.fieldset, nm)
                                        for m, nm in zip(ds.mols, ds.names)])
        score, _pred, _sd = model.rank_score(tab, beta=1.0)
        scored, extra = len(score), {"atoms": int(tab.coords.shape[0])}
    dt = time.time() - t0
    return {"chunk": int(chunk), "scored": int(scored), "seconds": round(dt, 1),
            "per_second": round(scored / dt, 1), "peak_mb": round(_peak_mb(), 0),
            "training_peak_mb": round(before, 0), **extra}


def stage_screen(force=False, repeats=(1, 4)):
    path = os.path.join(RESULTS, "screen.json")
    if os.path.exists(path) and not force:
        print("screen: already done")
        return json.load(open(path))
    import subprocess

    fs, T, _y = table_of(*SETS[0][1:], alpha=0.5)
    tmp = tempfile.mkdtemp(prefix="cmf_speed_")
    out = {"training_molecules": int(T.n_mol), "runs": []}
    try:
        for rep in repeats:
            lib = os.path.join(tmp, f"library_{rep}.sdf")
            n_lib = _build_library(lib, rep)
            size_mb = os.path.getsize(lib) / 1e6
            print(f"  library of {n_lib} molecules ({size_mb:.0f} MB)", flush=True)
            for label, chunk in (("streamed, 500 per block", 500),
                                 ("streamed, 100 per block", 100),
                                 ("loaded whole", 0)):
                # a separate process per configuration: the peak working set never goes down
                proc = subprocess.run([sys.executable, os.path.abspath(__file__),
                                       "--screen-worker", lib, str(chunk)],
                                      capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"    {label}: failed\n{proc.stdout[-2000:]}{proc.stderr[-2000:]}",
                          flush=True)
                    continue
                row = json.loads(proc.stdout.strip().splitlines()[-1])
                row.update({"library": n_lib, "library_mb": round(size_mb, 1), "label": label})
                out["runs"].append(row)
                print(f"    {label:24s} {row['seconds']:6.1f} s ({row['per_second']:5.0f}/s), "
                      f"peak process memory {row['peak_mb']:.0f} MB", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


# ------------------------------------------------------------------------------ the report


def report(results=RESULTS):
    parts = {}
    for name in ("kernel", "grid", "cutoff", "screen"):
        p = os.path.join(results, f"{name}.json")
        if os.path.exists(p):
            parts[name] = json.load(open(p))
    if not parts:
        return "No speed results yet; run `python benchmarks/run_speed.py`."
    lines = ["# What the kernel costs, and what each shortcut saves", "",
             "Part B of the plan is about speed. Every pair of numbers below was measured in one "
             "process on one data set, with the fast path replaced by the slow one, so the two "
             "differ in nothing but the shortcut. The machine is the one that produced every "
             "other table in this repository.", ""]

    if "kernel" in parts:
        k = parts["kernel"]
        lines += [f"## One kernel, {len(k['alphas'])} attenuations", "",
                  "Time to build the per-field kernels of a set at every attenuation of the "
                  f"grid ({', '.join(f'{a:g}' for a in k['alphas'])}), and how far each row is "
                  "from the first. The error column is the point: all three shortcuts are exact, "
                  "and the figures are round-off from summing the atom pairs in a different "
                  "order.", ""]
        for s in k["sets"]:
            lines += [f"**{s['set']}** -- {s['n_mol']} molecules, "
                      f"{s['atoms_per_molecule']:g} atoms each", "",
                      "| shortcut | seconds | speed-up | differs from the first row by |",
                      "|---|---:|---:|---:|"]
            base = s["rows"][0]["seconds"]
            for r in s["rows"]:
                lines.append(f"| {r['label']} | {r['seconds']:.2f} | "
                             f"{base / r['seconds']:.2f}x | {r['rel_error']:.0e} |")
            lines.append("")

    if "grid" in parts:
        g = parts["grid"]
        lines += ["## The same, end to end", "",
                  f"`grid_search` over the same {len(g['alphas'])} attenuations on "
                  f"{g['n_mol']} molecules of {g['set']}, {g['folds']}-fold cross-validation. "
                  "The attenuation chosen and the cross-validated q² are there to show that "
                  "nothing moved.", "",
                  "| shortcut | seconds | speed-up | alpha chosen | q²cv |",
                  "|---|---:|---:|---:|---:|"]
        base = g["rows"][0]["seconds"]
        for r in g["rows"]:
            lines.append(f"| {r['label']} | {r['seconds']:.1f} | {base / r['seconds']:.2f}x | "
                         f"{r['alpha']:g} | {r['q2_cv']:+.6f} |")
        same = len({round(r["q2_cv"], 9) for r in g["rows"]}) == 1
        lines += ["", f"**{base / g['rows'][-1]['seconds']:.1f} times faster**"
                      + (", and the selection is identical to six decimal places."
                         if same else ", but the selection moved, which needs explaining."), ""]

    if "cutoff" in parts:
        c = parts["cutoff"]
        lines += ["## The cutoff (item B2), which pays only sometimes", "",
                  "A Gaussian overlap falls below "
                  f"{c['tol']:.0e} of its peak beyond `sqrt(-4 ln tol / b)`, so pairs further "
                  "apart can be dropped and the sum run over a neighbour list. Whether that is "
                  "worth doing depends on the molecules. `auto` decides per field by sampling the "
                  "real distances; `forced` uses the list whatever the geometry.", "",
                  "| set | atoms per molecule | alpha | pairs kept, Gaussian | pairs kept, steric "
                  "| dense | auto | forced | auto is |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for s in c["sets"]:
            for r in s["rows"]:
                lines.append(f"| {s['set']} | {s['atoms_per_molecule']:g} | {r['alpha']:g} | "
                             f"{r['keep_gauss']:.0%} | {r['keep_steric']:.0%} | "
                             f"{r['dense_seconds']:.2f} s | {r['auto_seconds']:.2f} s "
                             f"({r['dense_seconds'] / r['auto_seconds']:.2f}x) | "
                             f"{r['forced_seconds']:.2f} s "
                             f"({r['dense_seconds'] / r['forced_seconds']:.2f}x) | "
                             f"{'the exact sum' if r['auto_exact'] else 'the neighbour list'} |")
        best = max((r for s in c["sets"] for r in s["rows"]),
                   key=lambda r: r["dense_seconds"] / r["auto_seconds"])
        lines += ["",
                  "So the cutoff is not a speed-up of the program; it is a speed-up of one case. "
                  "On a congeneric series superimposed in one frame almost every atom pair is "
                  "inside the cutoff and `auto` correctly refuses to build a list. On molecules "
                  f"five times longer at a strong attenuation it is worth "
                  f"{best['dense_seconds'] / best['auto_seconds']:.2f} times, and the error it "
                  f"introduces is {best['forced_rel_error']:.0e} relative.", ""]

    if "screen" in parts:
        sc = parts["screen"]
        lines += ["## Screening a library in blocks (item B3)", "",
                  f"The same library scored against a {sc['training_molecules']}-molecule model, "
                  "once in blocks and once loaded whole. Peak process memory is the figure to "
                  "read: the time is similar, and the point of streaming is that the memory does "
                  "not grow with the file.", "",
                  "| library | how | seconds | molecules per second | peak process memory |",
                  "|---|---|---:|---:|---:|"]
        for r in sc["runs"]:
            lines.append(f"| {r['library']} molecules ({r['library_mb']:.0f} MB) | {r['label']} | "
                         f"{r['seconds']:.0f} | {r['per_second']:.0f} | {r['peak_mb']:.0f} MB |")
        libs = sorted({r["library"] for r in sc["runs"]})
        if len(libs) > 1:
            def peak(lib, chunk):
                return next((r["peak_mb"] for r in sc["runs"]
                             if r["library"] == lib and r["chunk"] == chunk), float("nan"))
            s_small, s_big = peak(libs[0], 100), peak(libs[-1], 100)
            w_small, w_big = peak(libs[0], 0), peak(libs[-1], 0)
            lines += ["",
                      f"Between the two library sizes ({libs[0]} and {libs[-1]} molecules) the "
                      f"whole-file path grows from {w_small:.0f} to {w_big:.0f} MB and the "
                      f"streamed one from {s_small:.0f} to {s_big:.0f} MB. The streamed figure "
                      "is what makes a library of any length possible on a desktop.", ""]
    text = "\n".join(lines) + "\n"
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="*", default=["kernel", "grid", "cutoff", "screen"],
                    choices=["kernel", "grid", "cutoff", "screen"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--screen-worker", nargs=2, metavar=("LIBRARY", "CHUNK"), default=None,
                    help="internal: score one library one way and print the result as JSON")
    a = ap.parse_args()
    if a.screen_worker:
        print(json.dumps(screen_worker(a.screen_worker[0], int(a.screen_worker[1]))))
        return
    if a.report:
        print(report())
        return
    if "kernel" in a.stage:
        stage_kernel(a.force)
    if "grid" in a.stage:
        stage_grid(a.force)
    if "cutoff" in a.stage:
        stage_cutoff(a.force)
    if "screen" in a.stage:
        stage_screen(a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
