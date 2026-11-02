#!/usr/bin/env python
"""
Three-body channels and handedness for crystals (plan item M10).

Every descriptor this program has for a crystal is a **two-point** one: it counts how much field
weight sits at each distance from how much other field weight. Two things follow, and both are
limitations of the description and not of the data.

* Two structures with the same multiset of interatomic distances give exactly the same channels.
  Such structures are called **homometric**, and a defect series on one lattice is full of
  arrangements that come close to it.
* A mirror image has the same distances as the structure it reflects, so no two-point descriptor
  can tell a left-handed crystal from a right-handed one.

`cmfqsar/threebody.py` closes both: the three-body block resolves the triangle around every atom
of the cell (two radii and the distance between the two neighbours), and the handedness block is
the signed volume spanned by three field-weighted vectors about each centre. This script measures
what they are worth.

What is measured
----------------
**The two blind spots, exactly** (`--stage concept`). A pair of homometric structures built for
the purpose: the two-point channels give them a kernel of 1.000000 and the three-body channels
tell them apart. A cell and its mirror image: the three-body channels are identical to the last
digit and the handedness channels have the opposite sign. These are properties of the descriptor,
so they are reported as numbers and not as accuracy.

**What it adds to a property model** (`--stage fit`). The three Matbench tasks of
`run_reciprocal.py`, the same subsets and the same five published folds: the two-point descriptor
alone, the three-body channels alone, and the two kernels mixed as
`K = (1 - t) K_two_point + t K_three_body` with the mixing weight chosen inside every training
fold. These cells are small and mostly simple, so this stage says what the channels cost and
whether they pay for themselves where the arrangement is not the whole of the signal.

**The set where the arrangement is the signal** (`--stage defects`). The 2DMD defect series of
`run_2dmd.py` (plan item M14): one host lattice, one supercell, and a different arrangement of
one to three point defects in every cell. The two-point autocorrelation of the whole cells is
measured there already, and it is the right thing to compare against, because it differs from the
three-body channels in exactly one respect. Two splits are reported: random folds, and the
extrapolation split that trains on the cells with one and two defects and predicts the cells with
three, where the two-point description fails outright.

No new data is fetched: the Matbench files and the 2DMD archive are the ones those two scripts
already use.

Usage
-----
    python benchmarks/run_threebody.py                       # every stage, then the table
    python benchmarks/run_threebody.py --stage concept       # the two blind spots and the figure
    python benchmarks/run_threebody.py --stage fit --limit 600
    python benchmarks/run_threebody.py --stage defects --materials MoS2
    python benchmarks/run_threebody.py --probe               # what one cell costs, before a run
    python benchmarks/run_threebody.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_2dmd as D                                                     # noqa: E402
import run_reciprocal as M6                                              # noqa: E402

from cmfqsar import periodic, threebody                                  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel, channel_scale      # noqa: E402
from cmfqsar.model import median_gamma_scale                             # noqa: E402

FEATURES = os.path.join(HERE, "cache", "threebody")
RESULTS = os.path.join(HERE, "results", "threebody")

#: the three-body settings tried on the Matbench tasks: (radial reach, grid spacing)
TB: Tuple[Tuple[float, float], ...] = ((4.0, 0.5), (5.0, 0.75))
#: the two-point settings it is measured against, from run_reciprocal.py
REAL: Tuple[Tuple[str, float, float], ...] = (("4 A", 4.0, 0.35), ("6 A", 6.0, 0.35),
                                              ("6 A", 6.0, 0.5))
#: the weights the two kernels are mixed with
MIX = (0.25, 0.5, 0.75)
#: the width of the window on every grid point, in angstrom
SIGMA = 0.35
LIMIT = 600
#: the three-body settings tried on the defect series: (radial reach, grid spacing)
DEFECT_TB: Tuple[Tuple[float, float], ...] = ((4.0, 0.75), (5.0, 0.75))
DEFECT_LIMIT = 2000
#: the widths of the two-point descriptor whose kernels the mixed row draws on
DEFECT_AC_SIGMA = 0.5
#: the kinds of model the defect stage fits
DEFECT_KINDS = ("threebody", "both")


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=M6._default)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def spec_of(r_max: float, step: float, chirality: bool = False) -> threebody.ThreeBodySpec:
    return threebody.ThreeBodySpec(r_max=float(r_max), step=float(step), sigma=SIGMA,
                                   chirality=bool(chirality))


# --------------------------------------------------------------------------- #
# What one cell costs, before anything large is started
# --------------------------------------------------------------------------- #
def probe(n_cells: int = 3) -> dict:
    """The time and the memory one cell of each kind asks for, printed before a bulk run."""
    out = {}
    table, _, _, _ = M6.load_set("log_kvrh", 20)
    cells = table.molecules[:n_cells]
    for r_max, step in TB:
        spec = spec_of(r_max, step)
        t0 = time.time()
        rows = [threebody.threebody_features(c, spec) for c in cells]
        dt = (time.time() - t0) / len(cells)
        out[f"matbench r{r_max:g} step{step:g}"] = {
            "seconds_per_cell": dt, "channels": int(len(rows[0])),
            "atoms": float(np.mean([c.n_atoms for c in cells])),
            "gigabytes_per_1000": float(1000 * len(rows[0]) * 8 / 1e9)}
    try:
        prep = D.prepare("MoS2")
        series = D.Series(prep, D.pick(prep, 5))
        cells = series.whole().molecules[:n_cells]
    except Exception as exc:                                  # the archive may not be there yet
        log(f"probe: the 2DMD cells are not available ({exc})")
        cells = []
    if cells:
        for r_max, step in DEFECT_TB:
            spec = spec_of(r_max, step)
            t0 = time.time()
            rows = [threebody.threebody_features(c, spec) for c in cells]
            dt = (time.time() - t0) / len(cells)
            out[f"2dmd r{r_max:g} step{step:g}"] = {
                "seconds_per_cell": dt, "channels": int(len(rows[0])),
                "atoms": float(np.mean([c.n_atoms for c in cells])),
                "gigabytes_per_1000": float(1000 * len(rows[0]) * 4 / 1e9)}
    for name, rec in out.items():
        log(f"probe {name}: {rec['atoms']:.0f} atoms, {rec['channels']} channels, "
            f"{rec['seconds_per_cell']:.2f} s per cell, "
            f"{rec['gigabytes_per_1000']:.2f} GB per 1,000 cells")
    jdump(out, result("probe.json"))
    return out


# --------------------------------------------------------------------------- #
# Stage A: the two blind spots, exactly
# --------------------------------------------------------------------------- #
def homometric_sets(n: int = 12, k: int = 5) -> Tuple[List[int], List[int]]:
    """Two sets of sites on a ring of n with the same multiset of distances between them, and
    not related by a shift or a reflection of the ring.

    The two chains built from them have the same pair histogram, exactly, at every width and
    every cutoff: they are a test of the descriptor and not of the arithmetic.
    """
    def distances(s):
        return tuple(sorted(min((a - b) % n, (b - a) % n) for a in s for b in s if a != b))

    def orbit(s):
        return {tuple(sorted((sign * x + t) % n for x in s))
                for t in range(n) for sign in (1, -1)}

    by: Dict[tuple, List[tuple]] = {}
    for s in combinations(range(n), k):
        by.setdefault(distances(s), []).append(s)
    for sets in by.values():
        for a in sets:
            for b in sets:
                if b not in orbit(a):
                    return list(a), list(b)
    raise SystemExit("no homometric pair on this ring")


def _chain(sites: Sequence[int], fs, n: int = 12, step: float = 1.9, name: str = ""):
    from ase import Atoms

    pos = np.zeros((len(sites), 3))
    pos[:, 0] = np.asarray(sites, float) * step
    atoms = Atoms(["Na"] * len(sites), positions=pos,
                  cell=[[n * step, 0, 0], [0, 14.0, 0], [0, 0, 14.0]], pbc=(True, False, False))
    return periodic.crystal_fields(atoms, fs, name)


def _generic(fs, mirror: bool = False, seed: int = 3, name: str = ""):
    """Four atoms of four species with no symmetry: a cell that has a handedness."""
    from ase import Atoms

    rng = np.random.RandomState(seed)
    atoms = Atoms(["Na", "Cl", "K", "O"], scaled_positions=rng.rand(4, 3),
                  cell=np.diag([6.0, 6.2, 5.8]), pbc=True)
    if mirror:
        p = atoms.get_positions()
        p[:, 0] *= -1.0
        atoms.set_positions(p)
        atoms.wrap()
    return periodic.crystal_fields(atoms, fs, name)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


def concept(force: bool = False) -> dict:
    """The two structures a two-point descriptor cannot tell apart, and what these channels say."""
    path = result("concept.json")
    if os.path.isfile(path) and not force:
        log(f"concept: {path} is there")
        rec = jload(path)
        concept_figure(rec)
        return rec
    a, b = homometric_sets()
    fs = periodic.inorganic_fieldset(["Na"], charge_model="electronegativity", alpha=1.0)
    ca, cb = _chain(a, fs, name="A"), _chain(b, fs, name="B")
    out: dict = {"homometric": {"sets": [a, b], "ring": 12, "spacing": 1.9}}
    ps = periodic.PeriodicSpec(r_max=12.0, step=0.25, sigma=0.25, taper=0.0, background=True)
    pa, pb = (periodic.periodic_autocorrelation(c, ps) for c in (ca, cb))
    out["homometric"]["two_point_cosine"] = _cos(pa, pb)
    out["homometric"]["two_point_profile"] = [pa.tolist(), pb.tolist()]
    out["homometric"]["distances"] = ps.centres.tolist()
    tb = spec_of(6.0, 0.5)
    ta, tbv = (threebody.threebody_autocorrelation(c, tb) for c in (ca, cb))
    out["homometric"]["three_body_cosine"] = _cos(ta, tbv)
    out["homometric"]["three_body_profile"] = [ta.tolist(), tbv.tolist()]
    # the second blind spot: a mirror image
    fs4 = periodic.inorganic_fieldset(["Na", "Cl", "K", "O"], charge_model="electronegativity",
                                      alpha=1.0)
    spec = spec_of(4.5, 0.5, chirality=True)
    left, right = _generic(fs4, False, name="L"), _generic(fs4, True, name="R")
    tl, tr = (threebody.threebody_autocorrelation(c, spec) for c in (left, right))
    hl, hr = (threebody.periodic_chirality(c, spec) for c in (left, right))
    nacl = periodic.crystal_fields(__import__("ase.build", fromlist=["bulk"]).bulk(
        "NaCl", "rocksalt", a=5.64, cubic=True), fs4, "NaCl")
    out["mirror"] = {
        "three_body_cosine": _cos(tl, tr),
        "three_body_worst": float(np.max(np.abs(tl - tr)) / max(np.max(np.abs(tl)), 1e-300)),
        "handedness_cosine": _cos(hl, hr),
        "handedness_largest": float(np.max(np.abs(hl))),
        "handedness_sum_worst": float(np.max(np.abs(hl + hr)) / max(np.max(np.abs(hl)), 1e-300)),
        "handedness_left": hl.tolist(), "handedness_right": hr.tolist(),
        "centrosymmetric_largest": float(np.max(np.abs(
            threebody.periodic_chirality(nacl, spec)))),
    }
    jdump(out, path)
    log(f"concept: the homometric pair has a two-point kernel of "
        f"{out['homometric']['two_point_cosine']:.8f} and a three-body kernel of "
        f"{out['homometric']['three_body_cosine']:.4f}")
    log(f"concept: a mirror image has a three-body kernel of "
        f"{out['mirror']['three_body_cosine']:.8f} and a handedness kernel of "
        f"{out['mirror']['handedness_cosine']:+.6f}")
    concept_figure(out)
    return out


def concept_figure(rec: dict) -> None:
    """Four panels: the two chains, and every channel of one structure against the other's."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h, m = rec["homometric"], rec["mirror"]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.2))
    ax, bx, cx, dx = axes
    step, n = h["spacing"], h["ring"]
    for row, (sites, colour, name) in enumerate(zip(h["sets"], ("#2f6fad", "#c2562a"),
                                                    ("structure A", "structure B"))):
        ax.plot(np.arange(n) * step, np.full(n, -row), ".", color="0.85", ms=7)
        ax.plot(np.asarray(sites, float) * step, np.full(len(sites), -row), "o",
                color=colour, ms=12, label=name)
    ax.set_ylim(-1.7, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("position along the chain (A)")
    ax.set_title("(a) Two cells with the same distances:\nevery distance occurs as often in both")
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")

    def against(axis, a, b, title, colour, line=+1.0):
        a, b = np.asarray(a, float), np.asarray(b, float)
        lim = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30) * 1.08
        axis.plot([-lim, lim], [-line * lim, line * lim], "-", color="0.75", lw=1.0, zorder=1)
        axis.plot(a, b, ".", color=colour, ms=4.5, alpha=0.65, zorder=2)
        axis.set_xlim(-lim, lim)
        axis.set_ylim(-lim, lim)
        axis.set_title(title)

    against(bx, *h["two_point_profile"],
            f"(b) The two-point channels:\nkernel {h['two_point_cosine']:.8f}", "#2f6fad")
    bx.set_xlabel("channel of structure A")
    bx.set_ylabel("the same channel of structure B")
    bx.text(0.04, 0.94, "every point on the line:\nfor this descriptor the two\n"
                        "structures are one", transform=bx.transAxes, fontsize=8, va="top")

    against(cx, *h["three_body_profile"],
            f"(c) The three-body channels:\nkernel {h['three_body_cosine']:.4f}", "#c2562a")
    cx.set_xlabel("channel of structure A")
    cx.set_ylabel("the same channel of structure B")
    cx.text(0.04, 0.94, "off the line: the arrangement\nis resolved, not only the\ndistances",
            transform=cx.transAxes, fontsize=8, va="top")

    against(dx, m["handedness_left"], m["handedness_right"],
            f"(d) The handedness channels of a cell\nand its mirror image: kernel "
            f"{m['handedness_cosine']:+.4f}", "#4a7c59", line=-1.0)
    dx.set_xlabel("channel of the cell")
    dx.set_ylabel("the same channel of its mirror image")
    dx.text(0.04, 0.94, f"same size, opposite sign;\nthe three-body channels of the\n"
                        f"same pair agree to "
                        f"{m['three_body_worst']:.0e}", transform=dx.transAxes, fontsize=8,
            va="top")
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(result("concept.png"), dpi=150)
    plt.close(fig)
    log(f"figure: {result('concept.png')}")


# --------------------------------------------------------------------------- #
# Stage B: what the channels add to a property model
# --------------------------------------------------------------------------- #
def threebody_features_of(task: str, table, limit: int, r_max: float, step: float):
    """The three-body channels of a Matbench subset, computed once and kept on disk."""
    n_fields = table.weights.shape[1]

    def build():
        spec = spec_of(r_max, step)
        F = threebody.threebody_feature_matrix(table.molecules, spec, n_fields=n_fields,
                                               dtype=np.float32)
        return F, threebody.threebody_block_sizes(spec, n_fields)

    name = f"{task}_{limit}_tb_r{r_max:g}_s{step:g}.npz"
    return M6._cached(os.path.join(FEATURES, name), build)


def base_kernels(task: str, table, limit: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, r_max, sigma in REAL:
        F, blocks = M6.real_features(task, table, limit, r_max, sigma)
        out[f"two-point {name} sigma {sigma:g}"] = M6.kernel_of(F, blocks)
    for r_max, step in TB:
        F, blocks = threebody_features_of(task, table, limit, r_max, step)
        out[f"three-body r{r_max:g} step {step:g}"] = M6.kernel_of(np.asarray(F, float), blocks)
    return out


def settings_of(names: Sequence[str]) -> Dict[str, List[dict]]:
    """The families compared: each descriptor alone, and the two mixed."""
    two = [k for k in names if k.startswith("two-point")]
    three = [k for k in names if k.startswith("three-body")]
    out = {"the two-point descriptor": [{"keys": [k], "t": 0.0} for k in two],
           "the three-body channels": [{"keys": [k], "t": 0.0} for k in three],
           "both": [{"keys": [a, b], "t": t} for a in two for b in three for t in MIX]}
    return out


def fit(tasks: Sequence[str] = tuple(M6.TASKS), limit: int = LIMIT,
        force: bool = False) -> dict:
    path = result("fit.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    if out.get("limit") != int(limit):
        out = {"limit": int(limit)}
    for task in tasks:
        log(f"{task}: {M6.TASKS[task]['label']} ({M6.TASKS[task]['unit']})")
        table, y, fs, _ = M6.load_set(task, limit)
        Ks = base_kernels(task, table, limit)
        rec = out.setdefault(task, {})
        rec["n"] = int(len(y))
        rec["atoms_per_cell"] = float(np.mean([m.n_atoms for m in table.molecules]))
        rec.setdefault("descriptors", {})
        for name, setting in settings_of(list(Ks)).items():
            if name in rec["descriptors"] and not force:
                log(f"  {name}: there")
                continue
            log(f"  {name}")
            rec["descriptors"][name] = M6.evaluate(Ks, setting, y)
            jdump(out, path)
        rec["difference"] = {}
        base = rec["descriptors"].get("the two-point descriptor")
        for name in ("the three-body channels", "both"):
            other = rec["descriptors"].get(name)
            if base is None or other is None:
                continue
            d, lo, hi = M6.bootstrap_difference(y, np.asarray(other["predictions"]),
                                                np.asarray(base["predictions"]))
            rec["difference"][name] = {"d_r2": d, "low": lo, "high": hi}
            log(f"  {name} against the two-point descriptor: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        jdump(out, path)
        del Ks
    return out


# --------------------------------------------------------------------------- #
# Stage C: the set where the arrangement is the signal
# --------------------------------------------------------------------------- #
def threebody_kernels(series, material: str, r_max: float, step: float,
                      force: bool = False) -> Dict[str, np.ndarray]:
    """The three-body kernels of a defect series, with the same Gaussian layers as the
    two-point autocorrelation of `run_2dmd.py`, so that the two rows differ in one respect."""
    n = len(series.y)
    names = ["linear"] + [f"gauss {g:g}" for g in D.GAMMAS]
    paths = {s: D.kernel_path(material, n, f"threebody_r{r_max:g}_s{step:g}", s) for s in names}
    if all(os.path.isfile(p) for p in paths.values()) and not force:
        return {s: np.load(p) for s, p in paths.items()}
    table = series.whole()
    n_fields = table.weights.shape[1]
    spec = spec_of(r_max, step)
    t0 = time.time()
    F = threebody.threebody_feature_matrix(
        table.molecules, spec, n_fields=n_fields, dtype=np.float32,
        progress=lambda msg, frac: log(f"    {msg}") if frac == 0 else None)
    blocks = threebody.threebody_block_sizes(spec, n_fields)
    log(f"  three-body features {F.shape[0]} x {F.shape[1]} in {time.time() - t0:.0f} s "
        f"({F.nbytes / 1e9:.2f} GB)")
    F = np.asarray(F, float)
    sc = channel_scale(F, int(blocks[0]), blocks=blocks)
    K = autocorrelation_kernel(F, scale=sc, n_bins=int(blocks[0]), blocks=blocks)
    del F
    out = {"linear": K}
    scale = median_gamma_scale(K)
    for g in D.GAMMAS:
        out[f"gauss {g:g}"] = D.gauss_layer(K, g, scale)
    for s, M in out.items():
        np.save(paths[s], M)
    return out


def defect_kernels(series, material: str, force: bool = False) -> Dict[str, np.ndarray]:
    """Every three-body kernel of a defect series: both settings, with their Gaussian layers."""
    out: Dict[str, np.ndarray] = {}
    for r_max, step in DEFECT_TB:
        for name, K in threebody_kernels(series, material, r_max, step, force=force).items():
            out[f"r{r_max:g} step {step:g}, {name}"] = K
    return out


def mixed_kernels(series, material: str, force: bool = False) -> Dict[str, np.ndarray]:
    """The two-point kernel of `run_2dmd.py` and a three-body kernel added to it.

    The same layer is put on both before they are mixed, so that the mixture is of two
    descriptions of the same kind and the weight is the only new setting.  One width of each is
    used, chosen in advance and not by the test: the long two-point description at
    `DEFECT_AC_SIGMA` and the first of `DEFECT_TB`.
    """
    two = D.kernels_for("autocorrelation", series, material)
    r_max, step = DEFECT_TB[0]
    three = threebody_kernels(series, material, r_max, step, force=force)
    out: Dict[str, np.ndarray] = {}
    for layer in ["linear"] + [f"gauss {g:g}" for g in D.GAMMAS]:
        a = two.get(f"sigma {DEFECT_AC_SIGMA:g}, {layer}")
        b = three.get(layer)
        if a is None or b is None:
            continue
        a, b = D.cosine(a), D.cosine(b)
        for t in MIX:
            out[f"{layer}, t {t:g}"] = (1.0 - t) * a + t * b
    del two, three
    return out


def defects(materials: Sequence[str] = tuple(D.MATERIALS), limit: int = DEFECT_LIMIT,
            kinds: Sequence[str] = DEFECT_KINDS, force: bool = False) -> dict:
    """The three-body channels on the 2DMD defect series, against the two-point ones."""
    path = result("defects.json")
    out = jload(path) if os.path.isfile(path) and not force else {}
    for material in materials:
        prep = None
        for kind in kinds:
            key = f"{material} | {kind} | {limit}"
            if key in out and not force:
                log(f"{key}: there")
                continue
            if prep is None:
                prep = D.prepare(material)
                series = D.Series(prep, D.pick(prep, limit))
                n = len(series.y)
                few = series.n_defects <= 2
                log(f"=== {material}: {n} cells, {int(few.sum())} with one or two defects, "
                    f"{series.whole().molecules[0].n_atoms} atoms per cell")
            log(f"--- {key}")
            t0 = time.time()
            built = (defect_kernels(series, material, force=force) if kind == "threebody"
                     else mixed_kernels(series, material, force=force))
            Ks = D.unit_diagonal(built)
            del built
            rec = D.evaluate(Ks, series.y, series.n_defects, series.classes, D.folds(n))
            log(f"  random folds: R2 {rec['r2']:+.4f}, {1000 * rec['mae_site']:.1f} meV per site")
            log("  one and two defects -> three:")
            extra = D.evaluate(Ks, series.y, series.n_defects, series.classes,
                               [(np.where(few)[0], np.where(~few)[0])])
            rec.update({"material": material, "kernel": kind, "n": int(n),
                        "settings": list(Ks), "reach": [list(s) for s in DEFECT_TB],
                        "seconds": time.time() - t0, "extrapolation": extra})
            out[key] = rec
            jdump(out, path)
            del Ks
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report() -> str:
    lines = ["# Three-body channels and handedness for crystals (plan item M10)", ""]
    if os.path.isfile(result("concept.json")):
        rec = jload(result("concept.json"))
        h, m = rec["homometric"], rec["mirror"]
        lines += ["## The two blind spots of a two-point descriptor", "",
                  f"Two chains of five atoms on a ring of {h['ring']} sites, "
                  f"{h['sets'][0]} and {h['sets'][1]}, which have the same multiset of distances "
                  f"between their atoms. The two-point channels give them a kernel of "
                  f"{h['two_point_cosine']:.8f}: for that descriptor they are one structure. The "
                  f"three-body channels give {h['three_body_cosine']:.4f}.", "",
                  f"A cell of four species with no symmetry, against its mirror image: the "
                  f"three-body channels differ by "
                  f"{m['three_body_worst']:.1e} of the largest channel, so that block is blind to "
                  f"a reflection as every pair descriptor is; the handedness channels have the "
                  f"same size and the opposite sign (kernel {m['handedness_cosine']:+.6f}, the "
                  f"two sets summing to {m['handedness_sum_worst']:.1e} of the largest). "
                  f"Rocksalt, which is centrosymmetric, gives "
                  f"{m['centrosymmetric_largest']:.1e} in every handedness channel.", "",
                  "![The two blind spots](concept.png)", ""]
    if os.path.isfile(result("fit.json")):
        rec = jload(result("fit.json"))
        tasks = [t for t in M6.TASKS if t in rec]
        if tasks:
            lines += ["## What the channels add on three Matbench tasks", "",
                      f"Nested cross-validation over the five published folds of "
                      f"{rec['limit']} structures of each task, the protocol of "
                      f"`run_reciprocal.py`. \"Both\" mixes the two kernels as "
                      f"`K = (1 - t) K_two_point + t K_three_body`, with `t` chosen inside every "
                      f"training fold.", ""]
            head = "| Descriptor |"
            rule = "|---|"
            for t in tasks:
                head += f" {M6.TASKS[t]['label']}: R2 | MAE |"
                rule += "---:|---:|"
            lines += [head, rule]
            for name in ("the two-point descriptor", "the three-body channels", "both"):
                row = f"| {name} |"
                for t in tasks:
                    d = rec[t]["descriptors"].get(name)
                    row += (f" {d['r2']:+.3f} | {d['mae']:.4f} |" if d else " | |")
                lines.append(row)
            lines += ["", "**The difference against the two-point descriptor, with a 95 % "
                          "interval from 2,000 bootstrap resamples of the structures.**", "",
                      "| Descriptor |" + "".join(f" {M6.TASKS[t]['label']} |" for t in tasks),
                      "|---|" + "---:|" * len(tasks)]
            for name in ("the three-body channels", "both"):
                row = f"| {name} |"
                for t in tasks:
                    d = rec[t].get("difference", {}).get(name)
                    row += (f" {d['d_r2']:+.3f} [{d['low']:+.3f}, {d['high']:+.3f}] |"
                            if d else " |")
                lines.append(row)
            lines.append("")
    if os.path.isfile(result("defects.json")):
        mine = jload(result("defects.json"))
        theirs = (jload(os.path.join(D.RESULTS, "fit.json"))
                  if os.path.isfile(os.path.join(D.RESULTS, "fit.json")) else {})
        lines += ["## The set where the arrangement is the signal", "",
                  "The 2DMD defect series (plan item M14): one host lattice, one supercell, and "
                  "a different arrangement of one to three point defects in every cell. The "
                  "three-body channels of the whole cells against the two-point channels of the "
                  "same cells, with the same folds, the same ridge grid and the same Gaussian "
                  "layers. **Random folds** holds out a fifth of the cells; **one and two "
                  "defects to three** trains on the 131 cells with one or two defects and "
                  "predicts the rest, which is where a description that cannot see the "
                  "arrangement fails.", "",
                  "| The set | Descriptor | random folds: R2 | meV per site | "
                  "one and two defects to three: R2 | meV per site |",
                  "|---|---|---:|---:|---:|---:|"]
        for material in D.MATERIALS:
            rows = [("two-point, 10 A",
                     theirs.get(f"{material} | autocorrelation | {DEFECT_LIMIT}")),
                    ("arrangement (M14)",
                     theirs.get(f"{material} | arrangement | {DEFECT_LIMIT}")),
                    ("three-body, 4 to 5 A", mine.get(f"{material} | threebody | "
                                                      f"{DEFECT_LIMIT}")),
                    ("both", mine.get(f"{material} | both | {DEFECT_LIMIT}"))]
            for label, rec in rows:
                if not rec:
                    continue
                e = rec.get("extrapolation", {})
                lines.append(
                    f"| {material} | {label} | {rec['r2']:+.4f} | "
                    f"{1000 * rec['mae_site']:.1f} | {e.get('r2', float('nan')):+.3f} | "
                    f"{1000 * e.get('mae_site', float('nan')):.0f} |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    with open(result("table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    log(f"table: {result('table.md')}")
    return text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["concept", "fit", "defects"], default=None)
    p.add_argument("--kinds", nargs="*", default=list(DEFECT_KINDS))
    p.add_argument("--tasks", nargs="*", default=list(M6.TASKS))
    p.add_argument("--materials", nargs="*", default=list(D.MATERIALS))
    p.add_argument("--limit", type=int, default=LIMIT)
    p.add_argument("--defect-limit", type=int, default=DEFECT_LIMIT)
    p.add_argument("--probe", action="store_true", help="what one cell costs, then stop")
    p.add_argument("--force", action="store_true")
    p.add_argument("--report", action="store_true", help="write the table from what is there")
    a = p.parse_args()
    if a.probe:
        probe()
        return
    if a.report:
        print(report())
        return
    if a.stage in (None, "concept"):
        concept(a.force)
    if a.stage in (None, "fit"):
        fit(a.tasks, a.limit, a.force)
    if a.stage in (None, "defects"):
        defects(a.materials, a.defect_limit, a.kinds, force=a.force)
    report()


if __name__ == "__main__":
    main()
