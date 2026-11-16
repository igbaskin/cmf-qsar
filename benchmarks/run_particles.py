#!/usr/bin/env python
"""
Particles of thousands of atoms (plan item NT2).

A particle has no lattice, so the periodic autocorrelation of `cmfqsar/periodic.py` describes it
with open boundaries: a sum over the pairs of atoms closer than the cutoff, with no background
term.  The number of such pairs grows linearly with the number of atoms, so the descriptor should
too.  The overlap kernel compares every atom of one structure with every atom of the other, so
its cost grows with the square.  This benchmark measures both, and how a model on the
autocorrelation does on a property of metal particles.

Stage scaling
-------------
Copper icosahedra of 55 to 8,217 atoms built by ASE with no cell, and carbon nanotubes (6, 6) of
96 to 4,800 atoms, periodic along the axis with no lattice vector across it (as
``ase.build.nanotube`` writes them).  For each structure:

* the time of ASE's neighbour list (the route before this item; up to 5,083 atoms for the
  clusters and 3,600 for the tubes, because it is quadratic) and the peak memory it allocates;
* the time of the k-d tree route of `periodic._tree_pairs`, which every structure with an open
  direction and no lattice vector along it now takes, and whether the two give the same pairs;
* the time of the whole periodic autocorrelation by the new route;
* for the clusters up to 2,869 atoms, the time of the overlap kernel between the structure and a
  copy with every atom moved by up to 0.05 A, with open boundaries.

Stage property
--------------
150 particles of six metals (Cu, Ag, Au, Ni, Pd, Pt) in three shapes (icosahedra, decahedra and
truncated octahedra), 13 to about 3,000 atoms, built by ASE on the bulk lattice with no
relaxation.  Target: the energy per atom from ASE's effective medium theory potential (EMT), in
eV.  Compared on the same folds:

* the element mean -- the mean energy per atom of the training particles of the same metal;
* the size law -- a linear regression on n^(-1/3) (n the number of atoms) with its own intercept
  and slope for each metal, 12 coefficients; n^(-1/3) is the fraction of atoms at the surface of
  a compact particle, up to a constant;
* the coordination law -- a linear regression on the mean coordination number (the number of
  nearest neighbours of an atom, averaged over the particle) with its own intercept and slope
  for each metal, 12 coefficients; an EMT energy is close to a function of the neighbour counts,
  so this is the count that the fields could be measuring;
* the autocorrelation model -- kernel ridge regression on the cosine-normalised periodic
  autocorrelation (steric and electronegativity fields, cutoff 8 A), with the ridge parameter
  chosen by an inner 5-fold cross-validation inside every training part.

Two splits: 5-fold cross-validation repeated with 5 seeds, and extrapolation in size (train on the
particles of at most 1,000 atoms, predict the larger ones).

Stage counts (added 16 November 2026)
-------------------------------------
An EMT energy is a sum over atoms of a function of the neighbours of each atom, and the
autocorrelation reads the same pair distances. So the references that a field model has to beat
are counts: a linear regression on the number of neighbour pairs per atom in each of the first
four shells of the bulk lattice, and one on the fraction of atoms with each coordination number,
each with its own coefficients for each metal. Same folds and inner choice of the ridge; paired
bootstrap intervals over the particles.

Usage
-----
    python benchmarks/run_particles.py --stage scaling     # about 2 minutes
    python benchmarks/run_particles.py --stage property    # about 2 minutes
    python benchmarks/run_particles.py --stage counts      # about 2 minutes
    python benchmarks/run_particles.py --report
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import tracemalloc
import warnings
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.model import CMFModel, cv_predict  # noqa: E402
from cmfqsar.periodic import (OVERLAP_ATOM_LIMIT, PeriodicSpec, _tree_pairs,  # noqa: E402
                              crystal_table, inorganic_fieldset, periodic_autocorrelation,
                              periodic_feature_matrix)

RESULTS = os.path.join(HERE, "results", "particles")
CACHE = os.path.join(HERE, "cache", "particles")
SPEC = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, taper=2.0)
PHYSCHEM = ("steric", "prop:electronegativity")
METALS = ("Cu", "Ag", "Au", "Ni", "Pd", "Pt")
#: the largest structures the old (quadratic) neighbour list is run on
ASE_MAX = {"cluster": 5083, "long tube": 3600, "wide tube": 3600}
OVERLAP_MAX = 2869
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SEEDS = (0, 1, 2, 3, 4)
SPLIT_ATOMS = 1000


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Stage scaling
# --------------------------------------------------------------------------- #
def _open_cluster(k: int):
    from ase.cluster import Icosahedron
    a = Icosahedron("Cu", k)
    a.cell[:] = 0.0
    a.pbc = False
    return a


def _tube(n: int, length: int):
    from ase.build import nanotube
    return nanotube(n, n, length=length)          # zero vectors across the axis


def _same_pairs(c, r: float) -> Dict:
    """ASE's pairs against the tree's: counts and the largest distance difference."""
    from ase.neighborlist import neighbor_list
    i0, j0, d0 = neighbor_list("ijd", c.to_atoms(), r)
    i1, j1, d1 = _tree_pairs(c, r)
    o0 = np.lexsort((d0, j0, i0))
    o1 = np.lexsort((d1, j1, i1))
    same = (len(i0) == len(i1) and np.array_equal(i0[o0], i1[o1])
            and np.array_equal(j0[o0], j1[o1]))
    diff = float(np.max(np.abs(d0[o0] - d1[o1]))) if same and len(d0) else None
    return {"ase_pairs": int(len(i0)), "tree_pairs": int(len(i1)), "same_pairs": bool(same),
            "max_distance_difference": diff}


def _time_ase(c, r: float):
    from ase.neighborlist import neighbor_list
    atoms = c.to_atoms()
    tracemalloc.start()
    t0 = time.perf_counter()
    neighbor_list("ijd", atoms, r)
    t = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return t, peak / 2 ** 20


def _peak_mb(fn) -> float:
    """The peak memory that `fn` allocates through Python and NumPy, in MB."""
    tracemalloc.start()
    fn()
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return peak / 2 ** 20


def _best(fn, repeats: int = 3) -> float:
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def stage_scaling() -> None:
    warnings.filterwarnings("ignore")
    rows = []
    cases = [("cluster", k, _open_cluster(k)) for k in (3, 4, 5, 6, 7, 8, 9, 10, 12, 14)]
    cases += [("long tube", L, _tube(6, L)) for L in (4, 8, 16, 30, 50, 75, 100, 150, 250)]
    cases += [("wide tube", k, _tube(k, 10)) for k in (6, 12, 20, 30, 45, 60, 90)]
    for kind, size, s in cases:
        fs = inorganic_fieldset(sorted(set(s.get_chemical_symbols())), physchem=PHYSCHEM)
        table = crystal_table([s], fs, [f"{kind} {size}"])
        c = table.molecules[0]
        n = len(s)
        row = {"kind": kind, "size": size, "atoms": n}
        row["tree_seconds"] = _best(lambda: _tree_pairs(c, SPEC.r_max))
        row["autocorrelation_seconds"] = _best(lambda: periodic_autocorrelation(c, SPEC))
        row["autocorrelation_peak_mb"] = _peak_mb(lambda: periodic_autocorrelation(c, SPEC))
        if kind != "cluster":
            # the pairs alone: a chain's background term pairs every atom with every other
            flat = dataclasses.replace(SPEC, background=False)
            row["pairs_only_seconds"] = _best(lambda: periodic_autocorrelation(c, flat))
        if n <= ASE_MAX[kind]:
            row["ase_seconds"], row["ase_peak_mb"] = _time_ase(c, SPEC.r_max)
            row.update(_same_pairs(c, SPEC.r_max))
        if kind == "cluster" and n <= OVERLAP_MAX:
            b = s.copy()
            b.rattle(0.05, seed=1)
            pair = crystal_table([s, b], fs, ["a", "b"])
            m = CMFModel(fieldset=fs, method="krr", normalize=True, max_overlap_atoms=None)
            P, Q = pair.subset([0]), pair.subset([1])
            row["overlap_pair_seconds"] = _best(lambda: m._pair_kernels(P, Q, fs.specs), 1)
        print(row, flush=True)
        rows.append(row)
    jdump({"spec": {"r_max": SPEC.r_max, "step": SPEC.step, "sigma": SPEC.sigma},
           "fields": list(PHYSCHEM), "rows": rows}, os.path.join(RESULTS, "scaling.json"))


# --------------------------------------------------------------------------- #
# Stage property
# --------------------------------------------------------------------------- #
def particle_set() -> list:
    from ase.cluster import Decahedron, Icosahedron, Octahedron
    out = []
    for el in METALS:
        shapes = [(f"ico{k}", Icosahedron(el, k)) for k in (2, 3, 4, 5, 6, 7, 8, 9, 10)]
        shapes += [(f"deca{p}", Decahedron(el, p, p, 0)) for p in (2, 3, 4, 5, 6, 7, 8, 9)]
        shapes += [(f"octa{L}", Octahedron(el, L, cutoff=(L - 1) // 3))
                   for L in (4, 6, 8, 10, 12, 14, 16, 18)]
        for tag, a in shapes:
            a.info["name"] = f"{el}_{tag}"
            a.center(vacuum=8.0)          # a box for the EMT neighbour list; pbc stays False
            a.pbc = False
            out.append(a)
    return out


def energies(structs: list) -> np.ndarray:
    """EMT energy per atom of every particle, cached by name."""
    from ase.calculators.emt import EMT
    path = os.path.join(CACHE, "emt_energies.json")
    cache = jload(path) if os.path.isfile(path) else {}
    for a in structs:
        name = a.info["name"]
        if name not in cache:
            b = a.copy()
            b.calc = EMT()
            cache[name] = float(b.get_potential_energy()) / len(b)
    jdump(cache, path)
    return np.array([cache[a.info["name"]] for a in structs])


def _cosine(F: np.ndarray) -> np.ndarray:
    K = F @ F.T
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / d[:, None] / d[None, :]


def _krr_inner(K: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               seed: int) -> np.ndarray:
    """Predictions for `te` with the ridge parameter chosen by inner 5-fold CV on `tr`."""
    from cmfqsar.model import _Learner
    Ktr = K[np.ix_(tr, tr)]
    best, best_lam = np.inf, LAMBDAS[0]
    for lam in LAMBDAS:
        p = cv_predict(Ktr, y[tr], "krr", {"lam": lam}, folds=5, seed=seed)
        err = float(np.mean((p - y[tr]) ** 2))
        if err < best:
            best, best_lam = err, lam
    learner = _Learner("krr", {"lam": best_lam}).fit(Ktr, y[tr])
    return learner.predict(K[np.ix_(te, tr)])


def _element_mean(el: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    out = np.empty(len(te))
    for k, i in enumerate(te):
        same = tr[el[tr] == el[i]]
        out[k] = y[same].mean() if len(same) else y[tr].mean()
    return out


def mean_coordination(structs: list) -> np.ndarray:
    """The mean number of nearest neighbours of an atom, for every particle.

    A neighbour is an atom closer than halfway between the first and the second neighbour
    shell of the bulk metal (fcc: a / sqrt 2 and a), that is 0.85 a.
    """
    from ase.data import atomic_numbers, reference_states
    out = []
    for a in structs:
        el = a.get_chemical_symbols()[0]
        lattice = reference_states[atomic_numbers[el]]["a"]
        rec = crystal_table([a], inorganic_fieldset([el], physchem=("steric",)),
                            ["x"]).molecules[0]
        i, _, _ = _tree_pairs(rec, 0.5 * (lattice / np.sqrt(2.0) + lattice))
        out.append(len(i) / len(a))
    return np.array(out)


def _per_metal_line(el: np.ndarray, x: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    """Linear regression on `x` with an intercept and a slope for each metal."""
    metals = sorted(set(el))

    def design(idx):
        X = np.zeros((len(idx), 2 * len(metals)))
        for k, i in enumerate(idx):
            m = metals.index(el[i])
            X[k, 2 * m] = 1.0
            X[k, 2 * m + 1] = x[i]
        return X
    coef = np.linalg.lstsq(design(tr), y[tr], rcond=None)[0]
    return design(te) @ coef


def _size_law(el: np.ndarray, n: np.ndarray, y: np.ndarray, tr, te) -> np.ndarray:
    """Linear regression on n^(-1/3) with an intercept and a slope for each metal."""
    return _per_metal_line(el, n ** (-1.0 / 3.0), y, tr, te)


def _scores(y: np.ndarray, p: np.ndarray, ref_mean: float) -> Dict:
    rmse = float(np.sqrt(np.mean((p - y) ** 2)))
    q2 = 1.0 - float(np.sum((p - y) ** 2)) / float(np.sum((y - ref_mean) ** 2))
    return {"rmse_mev": 1000 * rmse, "q2": q2}


def stage_property() -> None:
    warnings.filterwarnings("ignore")
    structs = particle_set()
    y = energies(structs)
    names = [a.info["name"] for a in structs]
    el = np.array([a.get_chemical_symbols()[0] for a in structs])
    n = np.array([len(a) for a in structs], float)
    cn = mean_coordination(structs)
    fs = inorganic_fieldset(list(METALS), physchem=PHYSCHEM)
    t0 = time.perf_counter()
    table = crystal_table(structs, fs, names)
    t_fields = time.perf_counter() - t0
    t0 = time.perf_counter()
    F = periodic_feature_matrix(table.molecules, SPEC)
    t_ac = time.perf_counter() - t0
    K = _cosine(F)
    methods = ("element mean", "size law", "coordination law", "autocorrelation")
    preds = {m: [] for m in methods}
    from cmfqsar.model import _cv_splits
    for seed in SEEDS:
        P = {m: np.full(len(y), np.nan) for m in methods}
        for te, tr in _cv_splits(y, 5, seed):
            P["element mean"][te] = _element_mean(el, y, tr, te)
            P["size law"][te] = _size_law(el, n, y, tr, te)
            P["coordination law"][te] = _per_metal_line(el, cn, y, tr, te)
            P["autocorrelation"][te] = _krr_inner(K, y, tr, te, seed)
        for m in methods:
            preds[m].append(_scores(y, P[m], float(y.mean())))
    cv = {m: {"rmse_mev": float(np.mean([s["rmse_mev"] for s in preds[m]])),
              "rmse_mev_sd": float(np.std([s["rmse_mev"] for s in preds[m]])),
              "q2": float(np.mean([s["q2"] for s in preds[m]]))} for m in methods}
    tr = np.flatnonzero(n <= SPLIT_ATOMS)
    te = np.flatnonzero(n > SPLIT_ATOMS)
    ext_pred = {"element mean": _element_mean(el, y, tr, te),
                "size law": _size_law(el, n, y, tr, te),
                "coordination law": _per_metal_line(el, cn, y, tr, te),
                "autocorrelation": _krr_inner(K, y, tr, te, 0)}
    ext = {m: _scores(y[te], p, float(y[tr].mean())) for m, p in ext_pred.items()}
    worst = {m: float(1000 * np.max(np.abs(p - y[te]))) for m, p in ext_pred.items()}
    for m in methods:
        ext[m]["max_error_mev"] = worst[m]
    out = {"n": len(y), "atoms": [int(n.min()), int(n.max())], "atoms_total": int(n.sum()),
           "target_range_ev": [float(y.min()), float(y.max())],
           "mean_coordination_range": [float(cn.min()), float(cn.max())],
           "fields_seconds": t_fields, "autocorrelation_seconds": t_ac,
           "cv": cv, "cv_seeds": list(SEEDS), "extrapolation": ext,
           "extrapolation_split": {"train": int(len(tr)), "test": int(len(te)),
                                   "atoms": SPLIT_ATOMS},
           "lambdas": list(LAMBDAS)}
    print(json.dumps(out, indent=1))
    jdump(out, os.path.join(RESULTS, "property.json"))


# --------------------------------------------------------------------------- #
# Stage counts: references that count neighbours instead of reading fields
# --------------------------------------------------------------------------- #
#: the number of neighbour shells of the bulk fcc lattice counted by the shell reference; fixed
#: before the first run, from the cutoff of ASE's EMT, which lies between the third and the
#: fourth shell
SHELLS = 4
BOOT = 2000


def shell_counts(structs: list, shells: int = SHELLS) -> np.ndarray:
    """The number of neighbour pairs per atom in each of the first `shells` shells of the bulk
    lattice, for every particle.

    The shells of an fcc metal lie at a sqrt(k / 2), k = 1, 2, ...; a pair is counted in the
    shell whose radius is nearest, with the borders halfway between two shells.
    """
    from ase.data import atomic_numbers, reference_states
    out = []
    for a in structs:
        el = a.get_chemical_symbols()[0]
        lattice = reference_states[atomic_numbers[el]]["a"]
        radii = lattice * np.sqrt(np.arange(1, shells + 2) / 2.0)
        edges = np.r_[0.0, 0.5 * (radii[:-1] + radii[1:])]
        rec = crystal_table([a], inorganic_fieldset([el], physchem=("steric",)),
                            ["x"]).molecules[0]
        _i, _j, d = _tree_pairs(rec, edges[-1])
        out.append(np.histogram(d, bins=edges)[0] / len(a))
    return np.asarray(out, float)


def coordination_histogram(structs: list) -> np.ndarray:
    """The fraction of the atoms of a particle with each coordination number, 0 to 12.

    If the energy of an atom depended on its own number of nearest neighbours only, the energy
    per atom would be linear in these fractions.
    """
    from ase.data import atomic_numbers, reference_states
    out = []
    for a in structs:
        el = a.get_chemical_symbols()[0]
        lattice = reference_states[atomic_numbers[el]]["a"]
        rec = crystal_table([a], inorganic_fieldset([el], physchem=("steric",)),
                            ["x"]).molecules[0]
        i, _j, _d = _tree_pairs(rec, 0.5 * (lattice / np.sqrt(2.0) + lattice))
        cn = np.bincount(i, minlength=len(a))
        out.append(np.bincount(np.minimum(cn, 12), minlength=13) / len(a))
    return np.asarray(out, float)


def per_metal_kernel(el: np.ndarray, X: np.ndarray) -> np.ndarray:
    """A linear regression on the columns of X with its own coefficients and intercept for each
    metal, as a kernel: (1 + x_i . x_j) for two particles of the same metal, 0 otherwise."""
    Z = np.asarray(X, float) - np.asarray(X, float).mean(axis=0)
    same = (np.asarray(el)[:, None] == np.asarray(el)[None, :]).astype(float)
    return same * (1.0 + Z @ Z.T)


def _boot_rmse_difference(y, pa, pb, seed=0):
    """RMSE of a minus RMSE of b, meV per atom, with a 95 % bootstrap interval over particles."""
    rng = np.random.RandomState(seed)
    ea, eb = (np.asarray(pa) - y) ** 2, (np.asarray(pb) - y) ** 2
    idx = rng.randint(0, len(y), (BOOT, len(y)))
    d = 1000 * (np.sqrt(ea[idx].mean(axis=1)) - np.sqrt(eb[idx].mean(axis=1)))
    return {"mean": float(1000 * (np.sqrt(ea.mean()) - np.sqrt(eb.mean()))),
            "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5))}


def stage_counts() -> None:
    """The autocorrelation against two count references, on the folds of stage property."""
    warnings.filterwarnings("ignore")
    structs = particle_set()
    y = energies(structs)
    names = [a.info["name"] for a in structs]
    el = np.array([a.get_chemical_symbols()[0] for a in structs])
    n = np.array([len(a) for a in structs], float)
    fs = inorganic_fieldset(list(METALS), physchem=PHYSCHEM)
    K = _cosine(periodic_feature_matrix(crystal_table(structs, fs, names).molecules, SPEC))
    t0 = time.perf_counter()
    S = shell_counts(structs)
    H = coordination_histogram(structs)
    t_counts = time.perf_counter() - t0
    kernels = {"autocorrelation": K, "shell counts": per_metal_kernel(el, S),
               "coordination histogram": per_metal_kernel(el, H)}
    from cmfqsar.model import _cv_splits
    cv = {m: [] for m in kernels}
    pred = {m: np.zeros(len(y)) for m in kernels}
    for seed in SEEDS:
        P = {m: np.full(len(y), np.nan) for m in kernels}
        for te, tr in _cv_splits(y, 5, seed):
            for m, Km in kernels.items():
                P[m][te] = _krr_inner(Km, y, tr, te, seed)
        for m in kernels:
            cv[m].append(_scores(y, P[m], float(y.mean())))
            pred[m] += P[m] / len(SEEDS)
    tr = np.flatnonzero(n <= SPLIT_ATOMS)
    te = np.flatnonzero(n > SPLIT_ATOMS)
    ext_pred = {m: _krr_inner(Km, y, tr, te, 0) for m, Km in kernels.items()}
    out = {"n": len(y), "shells": SHELLS, "counts_seconds": t_counts,
           "cv": {m: {"rmse_mev": float(np.mean([s["rmse_mev"] for s in v])),
                      "rmse_mev_sd": float(np.std([s["rmse_mev"] for s in v])),
                      "q2": float(np.mean([s["q2"] for s in v]))} for m, v in cv.items()},
           "extrapolation": {m: {**_scores(y[te], p, float(y[tr].mean())),
                                 "max_error_mev": float(1000 * np.max(np.abs(p - y[te])))}
                             for m, p in ext_pred.items()},
           "differences": {
               "cv": {m: _boot_rmse_difference(y, pred[m], pred["autocorrelation"])
                      for m in kernels if m != "autocorrelation"},
               "extrapolation": {m: _boot_rmse_difference(y[te], ext_pred[m],
                                                          ext_pred["autocorrelation"])
                                 for m in kernels if m != "autocorrelation"}},
           "boot": BOOT}
    print(json.dumps(out, indent=1))
    jdump(out, os.path.join(RESULTS, "counts.json"))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _slope(rows: List[Dict], key: str, kind: str, n_min: int = 500) -> float:
    pts = [(r["atoms"], r[key]) for r in rows
           if r["kind"] == kind and key in r and r["atoms"] >= n_min and r[key] > 0]
    if len(pts) < 2:
        return float("nan")
    x, t = np.log([p[0] for p in pts]), np.log([p[1] for p in pts])
    return float(np.polyfit(x, t, 1)[0])


KIND_LABELS = {"cluster": ("copper icosahedron", "copper icosahedra"),
               "long tube": ("long tube (6, 6)", "long tubes (6, 6)"),
               "wide tube": ("wide tube (n, n)", "wide tubes (n, n)")}


def _t(x) -> str:
    return "--" if x is None else (f"{x:.3f}" if x < 10 else f"{x:.1f}")


def report() -> str:
    lines = ["# Particles of thousands of atoms (plan item NT2)", ""]
    sp = os.path.join(RESULTS, "scaling.json")
    if os.path.isfile(sp):
        S = jload(sp)
        rows = S["rows"]
        lines += ["## How the cost grows with the number of atoms", "",
                  "Copper icosahedra with no cell; carbon nanotubes (6, 6) of growing length "
                  "('long tube'); and carbon nanotubes (n, n) of ten repeat units and growing "
                  "diameter ('wide tube'). A tube is periodic along its axis and has no "
                  "lattice vector across it, as ASE writes it. Cutoff "
                  f"{S['spec']['r_max']:g} A, fields {', '.join(S['fields'])}. Times are the "
                  "best of three runs (one for the overlap kernel), in seconds. 'ASE' is the "
                  "neighbour list used before this item; 'k-d tree' is the route such "
                  "structures take now. 'Same pairs' says whether both give the same atom "
                  "pairs; the distances then agree to the 'Largest distance difference'. The "
                  "overlap kernel is that of the structure with a copy of itself whose atoms "
                  "are moved by up to 0.05 A, open boundaries. 'Pairs alone' is the "
                  "autocorrelation of a tube without its background term, the uniform density "
                  "spread along the axis. Peak memory is what Python and NumPy allocate. The "
                  "old route was not run beyond 5,083 atoms, because its memory grows with the "
                  "square of the number of atoms.", "",
                  "| Structure | Atoms | ASE neighbour list, s | ASE peak memory, MB | k-d tree, "
                  "s | Same pairs | Largest distance difference, A | Periodic autocorrelation "
                  "by the new route, s | Its peak memory, MB | Pairs alone, s | Overlap kernel "
                  "of two structures, s |",
                  "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            same = ("--" if "same_pairs" not in r else
                    "yes" if r["same_pairs"] else "no")
            diff = r.get("max_distance_difference")
            peak = "--" if "ase_peak_mb" not in r else f"{r['ase_peak_mb']:.0f}"
            label = KIND_LABELS[r["kind"]][0]
            lines.append(
                f"| {label} | {r['atoms']:,} | {_t(r.get('ase_seconds'))} | {peak} | "
                f"{_t(r['tree_seconds'])} | {same} | "
                f"{'--' if diff is None else f'{diff:.0e}'} | "
                f"{_t(r['autocorrelation_seconds'])} | {r['autocorrelation_peak_mb']:.0f} | "
                f"{_t(r.get('pairs_only_seconds'))} | {_t(r.get('overlap_pair_seconds'))} |")
        lines += ["", "Exponent of the growth with the number of atoms (slope of the "
                  "logarithm of the time against the logarithm of the atom count, structures "
                  "of 500 atoms or more): 1 is linear, 2 quadratic.", "",
                  "| Structure | ASE neighbour list | ASE peak memory | k-d tree | Periodic "
                  "autocorrelation | Its peak memory | Pairs alone | Overlap kernel |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for kind, (_, label) in KIND_LABELS.items():
            vals = [_slope(rows, k, kind) for k in
                    ("ase_seconds", "ase_peak_mb", "tree_seconds", "autocorrelation_seconds",
                     "autocorrelation_peak_mb", "pairs_only_seconds", "overlap_pair_seconds")]
            lines.append(f"| {label} | " + " | ".join(
                "--" if not np.isfinite(v) else f"{v:.2f}" for v in vals) + " |")
        big = [r for r in rows if r["kind"] == "cluster" and "overlap_pair_seconds" in r]
        if big:
            r = big[-1]
            n_set = 100
            t_over = n_set * (n_set + 1) / 2 * r["overlap_pair_seconds"]
            t_ac = n_set * r["autocorrelation_seconds"]
            lines += ["", f"For a set of {n_set} particles of {r['atoms']:,} atoms, the "
                      "measured times give a training kernel of "
                      f"{t_over / 60:.0f} minutes for the overlap kernel at one field width "
                      f"({n_set * (n_set + 1) // 2:,} pairs of structures) and "
                      f"{t_ac:.0f} s for the periodic autocorrelation of all "
                      f"{n_set}. The model therefore refuses the overlap kernel for a structure "
                      f"of more than {OVERLAP_ATOM_LIMIT:,} atoms (`max_overlap_atoms`; None "
                      "turns the guard off). Molecules are never refused."]
        lines.append("")
    pp = os.path.join(RESULTS, "property.json")
    if os.path.isfile(pp):
        R = jload(pp)
        lines += ["## A property of metal particles", "",
                  f"{R['n']} particles of Cu, Ag, Au, Ni, Pd and Pt (icosahedra, decahedra and "
                  f"truncated octahedra, {R['atoms'][0]:,} to {R['atoms'][1]:,} atoms, "
                  f"{R['atoms_total']:,} atoms in all), unrelaxed. Target: the energy per atom "
                  "from ASE's effective medium theory potential (EMT), "
                  f"{R['target_range_ev'][0]:.2f} to {R['target_range_ev'][1]:.2f} eV. The "
                  "periodic autocorrelation of all particles took "
                  f"{R['autocorrelation_seconds']:.1f} s. 'Size law' is a linear regression on "
                  "n^(-1/3) (n the number of atoms) with its own intercept and slope for each "
                  "metal; 'coordination law' the same on the mean number of nearest "
                  "neighbours of an atom ("
                  f"{R['mean_coordination_range'][0]:.2f} to "
                  f"{R['mean_coordination_range'][1]:.2f} here, 12 in the bulk); 'element "
                  "mean' is the mean of the training particles of the same metal. The "
                  "autocorrelation model is kernel ridge regression on the "
                  "cosine-normalised autocorrelation, its ridge parameter chosen by an inner "
                  "5-fold cross-validation in each training part.", "",
                  f"5-fold cross-validation repeated with {len(R['cv_seeds'])} seeds (mean, "
                  "and the standard deviation over the seeds):", "",
                  "| Model | RMSE, meV per atom | Standard deviation over seeds, meV | q2 |",
                  "|---|---:|---:|---:|"]
        for m, s in R["cv"].items():
            lines.append(f"| {m} | {s['rmse_mev']:.1f} | {s['rmse_mev_sd']:.1f} | "
                         f"{s['q2']:.4f} |")
        e = R["extrapolation_split"]
        lines += ["", f"Extrapolation in size: trained on the {e['train']} particles of at most "
                  f"{e['atoms']:,} atoms, tested on the {e['test']} larger ones (q2 against the "
                  "mean of the training part):", "",
                  "| Model | RMSE, meV per atom | Largest error, meV per atom | q2 |",
                  "|---|---:|---:|---:|"]
        for m, s in R["extrapolation"].items():
            lines.append(f"| {m} | {s['rmse_mev']:.1f} | {s['max_error_mev']:.1f} | "
                         f"{s['q2']:.4f} |")
        lines.append("")
    cp = os.path.join(RESULTS, "counts.json")
    if os.path.isfile(cp):
        C = jload(cp)
        lines += ["### References that count neighbours", "",
                  "The same particles, folds and inner choice of the ridge. 'Shell counts' is a "
                  "linear regression on the number of neighbour pairs per atom in each of the "
                  f"first {C['shells']} shells of the bulk lattice, with its own coefficients for "
                  "each metal (a linear kernel within a metal, zero between metals). "
                  "'Coordination histogram' is the same on the fraction of atoms with each "
                  "coordination number from 0 to 12. Both need no field. 'Difference' is the "
                  "RMSE of the reference minus that of the autocorrelation, with a 95 % "
                  "bootstrap interval over the particles; negative means the reference is "
                  "better. In the cross-validation the predictions are averaged over the seeds "
                  "first.", "",
                  "| Model | RMSE, meV per atom (cross-validation) | Difference | "
                  "RMSE, meV per atom (extrapolation) | Largest error, meV | Difference |",
                  "|---|---:|---:|---:|---:|---:|"]
        for m in C["cv"]:
            dc = C["differences"]["cv"].get(m)
            de = C["differences"]["extrapolation"].get(m)
            fmt = (lambda d: "" if d is None else
                   f"{d['mean']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]")
            lines.append(f"| {m} | {C['cv'][m]['rmse_mev']:.1f} | {fmt(dc)} | "
                         f"{C['extrapolation'][m]['rmse_mev']:.1f} | "
                         f"{C['extrapolation'][m]['max_error_mev']:.1f} | {fmt(de)} |")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("scaling", "property", "counts"), action="append",
                    help="stage(s) to run; the report always follows")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    for st in args.stage or []:
        if st == "scaling":
            stage_scaling()
        elif st == "property":
            stage_property()
        elif st == "counts":
            stage_counts()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
