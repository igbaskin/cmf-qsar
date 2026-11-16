#!/usr/bin/env python
"""
Item M1 measured on a set of nanotubes (plan item NT4).

A nanotube is periodic along its axis only.  Item M1 gave the periodic autocorrelation the
background term of a chain (the uniform density spread along the axis) and the neighbour list of
one periodic direction; item NT2 made the list of a tube with no lattice vector across its axis
linear in the number of atoms.  The plan says two things about tubes that this benchmark tests:
the handedness of a chiral tube is invisible to a pair kernel (item M10), and "the diameter and
the metallic or semiconducting character are visible in the distances".

The set
-------
Every carbon nanotube (n, m) with 3 <= n <= 20, 0 <= m <= n, a diameter of at least 0.4 nm and
at most 1,000 atoms in its translational cell, built by ``ase.build.nanotube`` (bond 1.42 A, no
lattice vector across the axis).  Each gets a band gap from a pi-orbital tight-binding model:
one orbital per carbon atom, hopping between bonded atoms t = -2.7 eV exp(-(d - 1.42 A) /
0.184 A), so that the shorter bonds of a curved sheet hop more strongly.  The lattice is
bipartite, so the spectrum at every wave number k along the axis is plus and minus the singular
values of the block that joins the two sublattices, and the gap is twice the smallest singular
value over k (61 points from 0 to pi per cell length, then a one-dimensional refinement).  In this
model a tube is metallic when (n - m) is a multiple of 3, apart from the small gap that curvature
opens in a tube that is not an armchair (n = m); the benchmark checks that the gaps fall into
these two groups.  The target is computed, not measured: it is a test of what the distances can
see, not a prediction of experiments.

Stage checks
------------
* the channels of a tube from one cell and from three cells of it;
* the channels of a chiral tube and of its mirror image, and the handedness block of item M10
  for the same pair;
* the channels of a tube given as a chain and the same tube in a 20 A box given as a crystal;
* a handedness reading that takes three radial shells of one field (a prototype in this file,
  ``shell_handedness``), because the handedness block of item M10 needs three different fields.

Stage model
-----------
Three targets: whether the tube is metallic ((n - m) a multiple of 3), the band gap, and the band
gap of the semiconducting tubes alone.  Compared
on 5-fold cross-validation repeated with 5 seeds:

* kernel ridge regression on the periodic autocorrelation (item M1, cutoff 8 A and 12 A);
* the same on the three-body channels (item M10, radial grid to 4 A);
* references from the indices of the tube: the diameter alone, and the diameter and the chiral
  angle, each in a Gaussian kernel; the most frequent class (or the mean);
* for the gap of the semiconducting tubes, a linear regression on 1/d, the textbook law.

A class is the sign of the regression on 0 and 1 against 0.5.  The ridge parameter and the width
of the Gaussian kernels are chosen by an inner 5-fold cross-validation in every training part.

Usage
-----
    python benchmarks/run_nanotubes.py --stage build      # about 15 minutes, cached
    python benchmarks/run_nanotubes.py --stage checks     # under a minute
    python benchmarks/run_nanotubes.py --stage model      # about 10 minutes
    python benchmarks/run_nanotubes.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import deque
from typing import Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

RESULTS = os.path.join(HERE, "results", "nanotubes")
CACHE = os.path.join(HERE, "cache", "nanotubes")
MAX_ATOMS = 1000
MIN_DIAMETER = 4.0                       # angstrom
BOND = 1.42
HOP, DECAY = -2.7, 0.184                 # eV, angstrom
N_K = 61
METAL_GAP = 0.3                          # eV: the two groups of gaps are split here
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
WIDTHS = (0.03, 0.1, 0.3, 1.0, 3.0)      # Gaussian widths in units of the reference's spread
SEEDS = (0, 1, 2, 3, 4)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The set and its tight-binding gaps
# --------------------------------------------------------------------------- #
def diameter(n: int, m: int) -> float:
    return BOND * np.sqrt(3.0) * np.sqrt(n * n + n * m + m * m) / np.pi


def chiral_angle(n: int, m: int) -> float:
    """Degrees: 0 for a zigzag tube (m = 0), 30 for an armchair one (n = m)."""
    return float(np.degrees(np.arctan2(np.sqrt(3.0) * m, 2.0 * n + m)))


def tube_list() -> List[Tuple[int, int, int]]:
    from ase.build import nanotube
    out = []
    for n in range(3, 21):
        for m in range(0, n + 1):
            if diameter(n, m) < MIN_DIAMETER:
                continue
            atoms = len(nanotube(n, m, length=1))
            if atoms <= MAX_ATOMS:
                out.append((n, m, atoms))
    return out


def tube(n: int, m: int, length: int = 1):
    from ase.build import nanotube
    t = nanotube(n, m, length=length, bond=BOND)
    t.info["name"] = f"({n},{m})"
    return t


def _bonds(t, cut: float = 1.7):
    from scipy.spatial import cKDTree
    X = t.positions
    L = float(t.cell[2, 2])
    N = len(X)
    pts = np.vstack([X + [0.0, 0.0, s * L] for s in (-1, 0, 1)])
    f = cKDTree(X).sparse_distance_matrix(cKDTree(pts), cut, output_type="ndarray")
    i, j, d = f["i"], f["j"], f["v"]
    keep = d > 0.1
    return i[keep], j[keep] % N, j[keep] // N - 1, d[keep], N


def _sublattices(i, j, N) -> np.ndarray:
    nb: List[List[int]] = [[] for _ in range(N)]
    for a, b in zip(i, j):
        nb[a].append(b)
    side = -np.ones(N, int)
    side[0] = 0
    queue = deque([0])
    while queue:
        a = queue.popleft()
        for b in nb[a]:
            if side[b] < 0:
                side[b] = 1 - side[a]
                queue.append(b)
            elif side[b] == side[a]:
                raise ValueError("the bond graph is not bipartite")
    if (side < 0).any():
        raise ValueError("the bond graph is not connected")
    return side


def tight_binding_gap(t) -> float:
    """The band gap of the pi-orbital model, in eV (see the module text)."""
    from scipy.linalg import svdvals
    from scipy.optimize import minimize_scalar
    i, j, s, d, N = _bonds(t)
    side = _sublattices(i, j, N)
    A, B = np.flatnonzero(side == 0), np.flatnonzero(side == 1)
    ia = {v: k for k, v in enumerate(A)}
    ib = {v: k for k, v in enumerate(B)}
    m = side[i] == 0
    r = np.array([ia[v] for v in i[m]])
    c = np.array([ib[v] for v in j[m]])
    shift = s[m]
    hop = HOP * np.exp(-(d[m] - BOND) / DECAY)

    def smallest(k: float) -> float:
        block = np.zeros((len(A), len(B)), complex)
        np.add.at(block, (r, c), hop * np.exp(1j * k * shift))
        return float(svdvals(block).min())
    ks = np.linspace(0.0, np.pi, N_K)
    vals = np.array([smallest(k) for k in ks])
    b = int(np.argmin(vals))
    res = minimize_scalar(smallest, bounds=(ks[max(b - 1, 0)], ks[min(b + 1, N_K - 1)]),
                          method="bounded", options={"xatol": 1e-8})
    return 2.0 * min(float(res.fun), float(vals[b]))


def stage_build() -> None:
    warnings.filterwarnings("ignore")
    path = os.path.join(CACHE, "gaps.json")
    cache = jload(path) if os.path.isfile(path) else {}
    for n, m, atoms in tube_list():
        key = f"{n},{m}"
        if key in cache:
            continue
        t0 = time.perf_counter()
        cache[key] = {"gap": tight_binding_gap(tube(n, m)), "atoms": atoms,
                      "seconds": time.perf_counter() - t0}
        print(key, cache[key], flush=True)
        jdump(cache, path)
    jdump(cache, path)


def load_set():
    """(tubes, n, m, gaps) of every tube of the list, from the cache of the build stage."""
    gaps = jload(os.path.join(CACHE, "gaps.json"))
    rows = [(n, m) for n, m, _ in tube_list()]
    missing = [f"({n},{m})" for n, m in rows if f"{n},{m}" not in gaps]
    if missing:
        raise SystemExit(f"run --stage build first; {len(missing)} tubes have no gap")
    n = np.array([r[0] for r in rows])
    m = np.array([r[1] for r in rows])
    g = np.array([gaps[f"{a},{b}"]["gap"] for a, b in rows])
    return [tube(a, b) for a, b in rows], n, m, g


# --------------------------------------------------------------------------- #
# Stage checks
# --------------------------------------------------------------------------- #
def _fields():
    from cmfqsar.periodic import inorganic_fieldset
    return inorganic_fieldset([], physchem=("steric",))


def _rows(structs, spec) -> np.ndarray:
    from cmfqsar.periodic import crystal_table, periodic_feature_matrix
    table = crystal_table(structs, _fields(), [str(k) for k in range(len(structs))])
    return periodic_feature_matrix(table.molecules, spec)


def _rel(a, b) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


def shell_handedness(rec, spec) -> np.ndarray:
    """A handedness reading for a structure of one field (a prototype, not part of item M10).

    Item M10's handedness block is the triple product of the field-weighted vector sums of three
    different fields within one radial shell, so a structure with one field has none.  This one
    takes the three vectors from three different shells of the first field instead: for every
    centre, A_s = sum over neighbours of w G_s(r) x, and the channels are A_a . (A_b x A_c) for
    the shells a < b < c, averaged over the centres.  A mirror image changes their sign.
    """
    import itertools

    from cmfqsar.threebody import _windows, neighbourhoods
    shells = spec.shell_centres
    trips = list(itertools.combinations(range(len(shells)), 3))
    out = np.zeros(len(trips))
    count = 0
    for X, Wc in neighbourhoods(rec, spec.cutoff):
        G = _windows(np.linalg.norm(X, axis=1), shells, float(spec.chirality_sigma))
        A = np.einsum("a,as,ad->sd", Wc[:, 0], G, X)
        out += np.array([A[a] @ np.cross(A[b], A[c]) for a, b, c in trips])
        count += 1
    return out / max(count, 1)


def mirror(t):
    out = t.copy()
    out.positions[:, 0] *= -1.0
    return out


def stage_checks() -> None:
    warnings.filterwarnings("ignore")
    from cmfqsar.periodic import PeriodicSpec, crystal_table
    from cmfqsar.threebody import ThreeBodySpec, periodic_chirality
    spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, taper=2.0)
    out = {}
    one, three = tube(8, 4), tube(8, 4, length=3)
    F = _rows([one, three], spec)
    out["supercell"] = {"tube": "(8,4)", "atoms": [len(one), len(three)],
                        "difference": _rel(F[0], F[1])}
    rows = []
    tb = ThreeBodySpec(chirality=True)
    fs = _fields()
    for n, m in ((8, 4), (7, 5), (10, 3), (12, 7), (10, 0), (6, 6)):
        t = tube(n, m)
        F = _rows([t, mirror(t)], spec)
        recs = crystal_table([t, mirror(t)], fs, ["a", "b"]).molecules
        block = periodic_chirality(recs[0], tb)
        across = [shell_handedness(r, tb) for r in recs]
        rows.append({"tube": f"({n},{m})", "atoms": len(t), "pair_difference": _rel(F[0], F[1]),
                     "m10_block_length": int(len(block)),
                     "across_shells": [float(v) for v in across[0]],
                     "across_shells_mirror_sum": float(np.max(np.abs(across[0] + across[1])))})
    out["mirror"] = rows
    t = tube(10, 0)
    boxed = t.copy()
    boxed.cell[0, 0] = boxed.cell[1, 1] = 20.0
    boxed.center(axis=(0, 1))
    chain = boxed.copy()
    boxed.pbc = (True, True, True)
    chain.pbc = (False, False, True)
    F = _rows([chain, boxed], spec)
    out["box"] = {"tube": "(10,0)", "box": 20.0, "difference": _rel(F[0], F[1])}
    print(json.dumps(out, indent=1))
    jdump(out, os.path.join(RESULTS, "checks.json"))


# --------------------------------------------------------------------------- #
# Stage model
# --------------------------------------------------------------------------- #
def _gauss(X: np.ndarray, width: float) -> np.ndarray:
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * width ** 2))


def _krr(K: np.ndarray, y: np.ndarray, tr, te, seed, kernels=None) -> np.ndarray:
    """Kernel ridge regression with the ridge (and, for a list of kernels, the kernel) chosen by
    an inner 5-fold cross-validation on the training part."""
    from cmfqsar.model import _Learner, cv_predict
    options = kernels if kernels is not None else [K]
    best, pick = np.inf, (options[0], LAMBDAS[0])
    for Kc in options:
        Ktr = Kc[np.ix_(tr, tr)]
        for lam in LAMBDAS:
            p = cv_predict(Ktr, y[tr], "krr", {"lam": lam}, folds=5, seed=seed)
            err = float(np.mean((p - y[tr]) ** 2))
            if err < best:
                best, pick = err, (Kc, lam)
    Kc, lam = pick
    learner = _Learner("krr", {"lam": lam}).fit(Kc[np.ix_(tr, tr)], y[tr])
    return learner.predict(Kc[np.ix_(te, tr)])


def _model_kernel(structs, **kw) -> np.ndarray:
    from cmfqsar.model import CMFModel
    from cmfqsar.periodic import crystal_table
    fs = _fields()
    table = crystal_table(structs, fs, [t.info["name"] for t in structs])
    return CMFModel(fieldset=fs, method="krr", normalize=True, **kw).training_kernel(table)


def stage_model() -> None:
    warnings.filterwarnings("ignore")
    from cmfqsar.model import _cv_splits
    structs, n, m, gaps = load_set()
    metal = ((n - m) % 3 == 0).astype(float)
    split_ok = bool(np.all((gaps < METAL_GAP) == (metal > 0.5)))
    d = np.array([diameter(a, b) for a, b in zip(n, m)])
    theta = np.array([chiral_angle(a, b) for a, b in zip(n, m)])
    t0 = time.perf_counter()
    kernels = {
        "periodic autocorrelation, cutoff 8 A": _model_kernel(
            structs, kernel_source="periodic", ac_d_max=8.0, ac_sigma=0.5, pbc_taper=2.0),
        "periodic autocorrelation, cutoff 12 A": _model_kernel(
            structs, kernel_source="periodic", ac_d_max=12.0, ac_sigma=0.5, pbc_taper=2.0),
        "three-body channels, radial grid to 4 A": _model_kernel(
            structs, kernel_source="threebody", tb_r_max=4.0),
    }
    t_kernels = time.perf_counter() - t0
    refs = {
        "Gaussian kernel on the diameter": [_gauss(d[:, None], w) for w in WIDTHS],
        "Gaussian kernel on the diameter and the chiral angle":
            [_gauss(np.stack([d, theta], 1), w) for w in WIDTHS],
    }
    semi = np.flatnonzero(metal < 0.5)
    out = {"n": len(n), "metallic": int(metal.sum()), "atoms": [int(min(len(t) for t in structs)),
                                                               int(max(len(t) for t in structs))],
           "diameter_nm": [float(d.min() / 10), float(d.max() / 10)],
           "gap_range": [float(gaps.min()), float(gaps.max())],
           "metallic_gaps": [float(gaps[metal > 0.5].min()), float(gaps[metal > 0.5].max())],
           "semiconducting_gaps": [float(gaps[semi].min()), float(gaps[semi].max())],
           "groups_split": split_ok, "kernel_seconds": t_kernels, "targets": {}}
    methods = {**{k: ("kernel", v) for k, v in kernels.items()},
               **{k: ("refs", v) for k, v in refs.items()}}
    for target in ("metallic", "gap", "gap of the semiconducting tubes"):
        idx = semi if target == "gap of the semiconducting tubes" else np.arange(len(n))
        y = (metal if target == "metallic" else gaps)[idx]
        res = {}
        all_methods = dict(methods)
        all_methods["the mean of the training part"] = ("mean", None)
        if target == "gap of the semiconducting tubes":
            all_methods["linear regression on 1/d"] = ("inverse", None)
        for name, (kind, K) in all_methods.items():
            scores = []
            for seed in SEEDS:
                p = np.full(len(y), np.nan)
                for te, tr in _cv_splits(y, 5, seed):
                    if kind == "kernel":
                        p[te] = _krr(K[np.ix_(idx, idx)], y, tr, te, seed)
                    elif kind == "refs":
                        p[te] = _krr(None, y, tr, te, seed,
                                     kernels=[Kc[np.ix_(idx, idx)] for Kc in K])
                    elif kind == "mean":
                        p[te] = y[tr].mean()
                    else:
                        x = 1.0 / d[idx]
                        coef = np.polyfit(x[tr], y[tr], 1)
                        p[te] = np.polyval(coef, x[te])
                if target == "metallic":
                    cls = (p > 0.5).astype(float)
                    pos, neg = y > 0.5, y < 0.5
                    scores.append({"accuracy": float(np.mean(cls == y)),
                                   "balanced": float(0.5 * (np.mean(cls[pos] == 1)
                                                            + np.mean(cls[neg] == 0)))})
                else:
                    scores.append({"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                                   "q2": 1.0 - float(np.sum((p - y) ** 2))
                                   / float(np.sum((y - y.mean()) ** 2))})
            res[name] = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
            res[name].update({k + "_sd": float(np.std([s[k] for s in scores]))
                              for k in scores[0]})
            print(target, name, res[name], flush=True)
        out["targets"][target] = {"n": int(len(idx)), "results": res}
    jdump(out, os.path.join(RESULTS, "model.json"))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report() -> str:
    lines = ["# Item M1 measured on a set of nanotubes (plan item NT4)", ""]
    cp = os.path.join(RESULTS, "checks.json")
    if os.path.isfile(cp):
        C = jload(cp)
        s, b = C["supercell"], C["box"]
        lines += ["## Checks of the chain kernel on tubes", "",
                  "Differences are the largest difference of the periodic autocorrelation rows "
                  "(cutoff 8 A) divided by the largest entry.", "",
                  f"* One cell of the tube {s['tube']} ({s['atoms'][0]} atoms) against three "
                  f"cells ({s['atoms'][1]} atoms): {s['difference']:.1e}.",
                  f"* The tube {b['tube']} given as a chain against the same tube in a "
                  f"{b['box']:g} A box given as a crystal: {b['difference']:.2f}. A tube read "
                  "from a file that states three periodic directions needs `--pbc c`.", "",
                  "A tube against its mirror image. 'Pair channels' is the difference of the "
                  "periodic autocorrelation rows. The handedness block of item M10 is a triple "
                  "product of three different fields, so a tube of one element (one field) has "
                  "none: its length is given. 'Across shells' is a prototype that takes the "
                  "three vectors from three radial shells of the one field instead (0, 1.3, "
                  "2.7 and 4 A; the four channels a < b < c), with the largest difference of "
                  "the tube's channels from minus its mirror image's.", "",
                  "| Tube | Atoms | Pair channels | Length of the M10 handedness block | Across "
                  "shells, the four channels | Tube plus its mirror image |",
                  "|---|---:|---:|---:|---|---:|"]
        for r in C["mirror"]:
            lines.append(f"| {r['tube']} | {r['atoms']} | {r['pair_difference']:.1e} | "
                         f"{r['m10_block_length']} | "
                         f"{', '.join(f'{v:.2e}' for v in r['across_shells'])} | "
                         f"{r['across_shells_mirror_sum']:.1e} |")
        lines.append("")
    mp = os.path.join(RESULTS, "model.json")
    if os.path.isfile(mp):
        M = jload(mp)
        lines += ["## What the distances see", "",
                  f"{M['n']} carbon nanotubes (n, m), {M['atoms'][0]} to {M['atoms'][1]} atoms "
                  f"in the cell, diameters {M['diameter_nm'][0]:.2f} to "
                  f"{M['diameter_nm'][1]:.2f} nm; {M['metallic']} have n - m divisible by 3. "
                  "Their tight-binding gaps are "
                  f"{M['metallic_gaps'][0]:.3f} to {M['metallic_gaps'][1]:.3f} eV (from "
                  "curvature) and those of the others "
                  f"{M['semiconducting_gaps'][0]:.2f} to {M['semiconducting_gaps'][1]:.2f} eV"
                  + (". " if M["groups_split"] else
                     ": the two groups overlap at their edges, because curvature opens a "
                     "larger gap in the thinnest 'metallic' tubes than the widest "
                     "semiconducting tubes have, so the class is the rule on the indices and "
                     "not a cut through the gaps. ")
                  + "The 1,000-atom limit keeps more of the tubes with n - m divisible by 3, "
                  "whose translational cell is three times shorter. 5-fold cross-validation "
                  "repeated with "
                  f"{len(SEEDS)} seeds (mean, and the standard deviation over the seeds). "
                  "References: Gaussian kernels on the diameter, and on the diameter and the "
                  "chiral angle (both from the indices); their width is chosen with the ridge.",
                  ""]
        T = M["targets"]["metallic"]
        lines += ["### Is the tube metallic?", "",
                  "| Model | Accuracy | Balanced accuracy | Standard deviation of the "
                  "balanced accuracy over seeds |", "|---|---:|---:|---:|"]
        for name, s in T["results"].items():
            lines.append(f"| {name} | {s['accuracy']:.3f} | {s['balanced']:.3f} | "
                         f"{s['balanced_sd']:.3f} |")
        lines += ["", "Balanced accuracy is the mean of the fractions right in each class: 0.5 "
                  "is no better than chance whatever the share of each class. The rule "
                  "'n - m divisible by 3' gives 1.000 from the indices.", ""]
        for target in ("gap", "gap of the semiconducting tubes"):
            T = M["targets"][target]
            lines += [f"### The {target} ({T['n']} tubes), eV", "",
                      "| Model | RMSE, eV | Standard deviation over seeds, eV | q2 |",
                      "|---|---:|---:|---:|"]
            for name, s in T["results"].items():
                lines.append(f"| {name} | {s['rmse']:.3f} | {s['rmse_sd']:.3f} | "
                             f"{s['q2']:.3f} |")
            lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("build", "checks", "model"), action="append",
                    help="stage(s) to run; the report always follows")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    for st in args.stage or []:
        {"build": stage_build, "checks": stage_checks, "model": stage_model}[st]()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
