#!/usr/bin/env python
"""
Size and shape as descriptors (plan item NT3).

The periodic autocorrelation describes a structure per atom of its cell, and the kernel divides
every row by its length, so the size of a particle never reaches the model.  Item NT3 gives it
back through the descriptor block: six computed descriptors of a structure (``calc:atoms``,
``calc:diameter``, ``calc:gyration_radius``, ``calc:aspect_ratio``, ``calc:surface_fraction``,
``calc:surface_area``) and the measured values of a table.  Before this item the invariant kernels
ignored the descriptor block without a word; they now add it as the overlap kernel does.

Stage geometry
--------------
The descriptors against answers known without them:

* surface atoms of copper icosahedra of 2 to 12 shells against the atom count of the outer shell,
  10 (k - 1)^2 + 2 for k shells;
* surface atoms of truncated octahedra and decahedra against the atoms with fewer than twelve
  nearest neighbours;
* the diameter of 39 carbon nanotubes (n, m) against a sqrt(n^2 + n m + m^2) / pi with the
  lattice constant a = 1.42 sqrt 3 A that ASE builds them with, and the gyration radius against
  half of it;
* the aspect ratio of blocks of fcc copper of 1 x 1 x L cubic cells against L;
* the surface fraction of a bulk crystal (0) and of a graphene sheet (1);
* the time of the surface descriptors against the atom count.

Stage model
-----------
The 150 metal particles of item NT2 (`benchmarks/run_particles.py`: Cu, Ag, Au, Ni, Pd, Pt,
13 to 3,564 atoms, unrelaxed), with two targets from ASE's effective medium theory potential
(EMT): the energy per atom, which is intensive, and the energy of the whole particle, which grows
with its size.  Compared on the same folds:

* kernel ridge regression on the periodic autocorrelation alone (the model as it was);
* the same with the five size and shape descriptors beside it (``calc:atoms``, ``calc:diameter``,
  ``calc:gyration_radius``, ``calc:aspect_ratio``, ``calc:surface_fraction``);
* a linear regression on the same five descriptors, with an intercept for each metal;
* for the energy per atom, linear regressions with an intercept and a slope for each metal on
  n^(-1/3) (the size law) and on the surface fraction;
* for the whole energy, a linear regression on n and n^(2/3) for each metal with no intercept,
  the volume and surface terms of a liquid drop.

The ridge parameter is chosen by an inner 5-fold cross-validation inside every training part,
and the descriptor statistics come from the training part alone.  Three splits: 5-fold
cross-validation repeated with 5 seeds; the same with folds that hold out each of the 25
geometries (a shape and a size) in all six metals at once, because the same geometry in another
metal is a near copy; and extrapolation in size (train on the particles of at most 1,000 atoms,
predict the larger ones).

Usage
-----
    python benchmarks/run_shape.py --stage geometry     # about 1 minute
    python benchmarks/run_shape.py --stage model        # about 2 minutes
    python benchmarks/run_shape.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.descriptors import read_values, surface_access  # noqa: E402
from cmfqsar.model import CMFModel, _cv_splits, _Learner, cv_predict  # noqa: E402
from cmfqsar.periodic import crystal_table, inorganic_fieldset  # noqa: E402

RESULTS = os.path.join(HERE, "results", "shape")
PHYSCHEM = ("steric", "prop:electronegativity")
SHAPE5 = ("calc:atoms", "calc:diameter", "calc:gyration_radius", "calc:aspect_ratio",
          "calc:surface_fraction")
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SEEDS = (0, 1, 2, 3, 4)
SPLIT_ATOMS = 1000
#: the kernel of item NT2's benchmark: cutoff 8 A, channels every 0.5 A, 2 A taper
KERNEL = dict(kernel_source="periodic", ac_d_max=8.0, ac_sigma=0.5, pbc_taper=2.0)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Stage geometry
# --------------------------------------------------------------------------- #
def _open(a):
    a.cell[:] = 0.0
    a.pbc = False
    return a


def _neighbour_counts(a) -> np.ndarray:
    from ase.data import atomic_numbers, reference_states
    from ase.neighborlist import neighbor_list
    lattice = reference_states[atomic_numbers[a.get_chemical_symbols()[0]]]["a"]
    i, _, _ = neighbor_list("ijd", a, 0.5 * (lattice / np.sqrt(2.0) + lattice))
    return np.bincount(i, minlength=len(a))


def stage_geometry() -> None:
    warnings.filterwarnings("ignore")
    from ase.build import bulk, graphene, nanotube
    from ase.cluster import Decahedron, Icosahedron, Octahedron
    out: Dict[str, list] = {"icosahedra": [], "fcc": [], "tubes": [], "blocks": [], "edge": []}
    for k in range(2, 13):
        a = _open(Icosahedron("Cu", k))
        t0 = time.perf_counter()
        frac = read_values(a, ["calc:surface_fraction"])[0]
        t = time.perf_counter() - t0
        out["icosahedra"].append({"shells": k, "atoms": len(a),
                                  "surface_atoms": int(round(frac * len(a))),
                                  "expected": 10 * (k - 1) ** 2 + 2, "seconds": t})
    shapes = [("truncated octahedron", L, _open(Octahedron("Cu", L, cutoff=(L - 1) // 3)))
              for L in (5, 9, 13, 17)]
    shapes += [("decahedron", p, _open(Decahedron("Cu", p, p, 0))) for p in (3, 5, 7)]
    for shape, size, a in shapes:
        access = surface_access(a)
        cn = _neighbour_counts(a)
        found, low = access > 0, cn < 12
        out["fcc"].append({"shape": shape, "size": size, "atoms": len(a),
                           "surface_atoms": int(found.sum()), "fewer_than_12": int(low.sum()),
                           "disagree": int(np.sum(found != low))})
    for n in range(3, 16):
        for m in range(0, n + 1):
            if (n, m) in ((3, 0), (3, 1), (3, 2), (3, 3), (4, 0)):
                continue                       # too thin for ASE's builder to be meaningful
            if len(out["tubes"]) >= 39:
                break
            t = nanotube(n, m, length=1)
            d_formula = 1.42 * np.sqrt(3.0) * np.sqrt(n * n + n * m + m * m) / np.pi
            d, rg = read_values(t, ["calc:diameter", "calc:gyration_radius"])
            out["tubes"].append({"n": n, "m": m, "atoms": len(t), "diameter": float(d),
                                 "formula": float(d_formula), "gyration_radius": float(rg)})
    for L in (1, 2, 3, 4, 6):
        a = _open(bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3 * L)))
        out["blocks"].append({"cells": [3, 3, 3 * L], "atoms": len(a),
                              "aspect_ratio": float(read_values(a, ["calc:aspect_ratio"])[0]),
                              "side_ratio": float(L)})
    crystal = bulk("Cu", "fcc", a=3.61, cubic=True).repeat(2)
    sheet = graphene(size=(4, 4, 1), vacuum=6.0)
    sheet.pbc = (True, True, False)
    for label, s in (("bulk copper (crystal)", crystal), ("graphene sheet (layer)", sheet)):
        out["edge"].append({"structure": label, "atoms": len(s),
                            "surface_fraction": float(
                                read_values(s, ["calc:surface_fraction"])[0]),
                            "diameter": float(read_values(s, ["calc:diameter"])[0])})
    for kind in ("icosahedra", "fcc", "tubes", "blocks", "edge"):
        print(kind, out[kind][-1])
    jdump(out, os.path.join(RESULTS, "geometry.json"))


# --------------------------------------------------------------------------- #
# Stage model
# --------------------------------------------------------------------------- #
def _per_metal(el, columns, y, tr, te, intercept=True):
    """Linear regression on `columns` (n, k) with an intercept and slopes for each metal."""
    metals = sorted(set(el))
    k = columns.shape[1]

    def design(idx):
        X = np.zeros((len(idx), len(metals) * (k + int(intercept))))
        for r, i in enumerate(idx):
            m = metals.index(el[i]) * (k + int(intercept))
            if intercept:
                X[r, m] = 1.0
            X[r, m + int(intercept):m + int(intercept) + k] = columns[i]
        return X
    coef = np.linalg.lstsq(design(tr), y[tr], rcond=None)[0]
    return design(te) @ coef


def _shared_slopes(el, columns, y, tr, te):
    """Linear regression on `columns` with an intercept for each metal and shared slopes."""
    metals = sorted(set(el))
    mu, sd = columns[tr].mean(0), columns[tr].std(0) + 1e-12

    def design(idx):
        X = np.zeros((len(idx), len(metals) + columns.shape[1]))
        for r, i in enumerate(idx):
            X[r, metals.index(el[i])] = 1.0
        X[:, len(metals):] = (columns[idx] - mu) / sd
        return X
    coef = np.linalg.lstsq(design(tr), y[tr], rcond=None)[0]
    return design(te) @ coef


class _Kernels:
    """Kernel ridge regression on the model's own kernel, with the channel features computed
    once for all particles (the model's feature function is replaced by a lookup)."""

    def __init__(self, table, fs_plain, fs_desc):
        self.table = table
        self.pos = {id(m): k for k, m in enumerate(table.molecules)}
        probe = CMFModel(fieldset=fs_plain, method="krr", normalize=True, **KERNEL)
        self.F = probe._ac_features(table)
        self.fs = {"plain": fs_plain, "desc": fs_desc}

    def predict(self, which: str, y, tr, te, seed, groups=None) -> np.ndarray:
        m = CMFModel(fieldset=self.fs[which], method="krr", normalize=True, **KERNEL)
        m._ac_features = lambda t: self.F[[self.pos[id(x)] for x in t.molecules]]
        T_tr, T_te = self.table.subset(list(tr)), self.table.subset(list(te))
        # subset() builds new records from the same molecules, so the lookup holds
        K = m.training_kernel(T_tr)
        best, lam = np.inf, LAMBDAS[0]
        for value in LAMBDAS:
            if groups is None:
                p = cv_predict(K, y[tr], "krr", {"lam": value}, folds=5, seed=seed)
            else:                                # the inner folds hold out geometries too
                p = np.full(len(tr), np.nan)
                for ite, itr in _group_splits(np.asarray(groups)[tr], 5, seed):
                    inner = _Learner("krr", {"lam": value}).fit(K[np.ix_(itr, itr)], y[tr][itr])
                    p[ite] = inner.predict(K[np.ix_(ite, itr)])
            err = float(np.mean((p - y[tr]) ** 2))
            if err < best:
                best, lam = err, value
        learner = _Learner("krr", {"lam": lam}).fit(K, y[tr])
        return learner.predict(m.test_kernel(T_te))


def _group_splits(groups, folds: int, seed: int):
    """(left-out, training) indices of folds that keep every group whole."""
    labels = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(labels))
    fold_of = {labels[p]: k % folds for k, p in enumerate(order)}
    g = np.array([fold_of[x] for x in groups])
    return [(np.flatnonzero(g == k), np.flatnonzero(g != k)) for k in range(folds)]


def _scores(y, p, ref) -> Dict:
    return {"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "q2": 1.0 - float(np.sum((p - y) ** 2)) / float(np.sum((y - ref) ** 2))}


def stage_model() -> None:
    warnings.filterwarnings("ignore")
    import run_particles as RP
    structs = RP.particle_set()
    e_atom = RP.energies(structs)
    n = np.array([len(a) for a in structs], float)
    e_total = e_atom * n
    el = np.array([a.get_chemical_symbols()[0] for a in structs])
    names = [a.info["name"] for a in structs]
    # the geometry of a particle (shape and size) is built in all six metals: a group
    geometry = np.array([nm.split("_", 1)[1] for nm in names])
    fs_plain = inorganic_fieldset(list(RP.METALS), physchem=PHYSCHEM)
    fs_desc = inorganic_fieldset(list(RP.METALS), physchem=PHYSCHEM)
    fs_desc.descriptors = SHAPE5
    t0 = time.perf_counter()
    table = crystal_table(structs, fs_desc, names)
    t_desc = time.perf_counter() - t0
    D = table.descriptors
    kern = _Kernels(table, fs_plain, fs_desc)
    # the descriptors of the plain field set are not read; its kernel ignores them
    sf = D[:, list(SHAPE5).index("calc:surface_fraction")][:, None]
    targets = {
        "energy per atom": (e_atom, 1000.0, "meV per atom", {
            "size law: linear regression on n^(-1/3) for each metal":
                lambda tr, te: _per_metal(el, (n ** (-1 / 3))[:, None], e_atom, tr, te),
            "linear regression on the surface fraction for each metal":
                lambda tr, te: _per_metal(el, sf, e_atom, tr, te)}),
        "energy of the particle": (e_total, 1.0, "eV", {
            "liquid drop: linear regression on n and n^(2/3) for each metal":
                lambda tr, te: _per_metal(el, np.stack([n, n ** (2 / 3)], 1), e_total, tr, te,
                                          intercept=False)}),
    }
    out = {"n": len(n), "atoms": [int(n.min()), int(n.max())], "descriptors": list(SHAPE5),
           "descriptor_seconds": t_desc, "targets": {}}
    for tname, (y, unit_scale, unit, refs) in targets.items():
        methods = {
            "periodic autocorrelation alone":
                lambda tr, te, s, g=None: kern.predict("plain", y, tr, te, s, g),
            "periodic autocorrelation and the five size and shape descriptors":
                lambda tr, te, s, g=None: kern.predict("desc", y, tr, te, s, g),
            "linear regression on the five descriptors, an intercept for each metal":
                lambda tr, te, s, g=None: _shared_slopes(el, D, y, tr, te)}
        methods.update({k: (lambda f: (lambda tr, te, s, g=None: f(tr, te)))(f)
                        for k, f in refs.items()})
        schemes = {"cv": lambda seed: _cv_splits(y, 5, seed),
                   "cv_geometry": lambda seed: _group_splits(geometry, 5, seed)}
        results = {}
        for scheme, splits in schemes.items():
            cv = {}
            for mname, fn in methods.items():
                runs, folds = [], []
                for seed in SEEDS:
                    p = np.full(len(y), np.nan)
                    for te, tr in splits(seed):
                        p[te] = fn(tr, te, seed, geometry if scheme == "cv_geometry" else None)
                        folds.append({"seed": seed, "held_out": sorted(set(geometry[te])),
                                      "rmse": unit_scale * float(
                                          np.sqrt(np.mean((p[te] - y[te]) ** 2)))})
                    runs.append(_scores(y, p, float(y.mean())))
                cv[mname] = {"rmse": unit_scale * float(np.mean([r["rmse"] for r in runs])),
                             "rmse_sd": unit_scale * float(np.std([r["rmse"] for r in runs])),
                             "q2": float(np.mean([r["q2"] for r in runs])),
                             "folds": folds}
                print(tname, scheme, mname, cv[mname], flush=True)
            results[scheme] = cv
        tr = np.flatnonzero(n <= SPLIT_ATOMS)
        te = np.flatnonzero(n > SPLIT_ATOMS)
        ext = {}
        for mname, fn in methods.items():
            p = fn(tr, te, 0)
            s = _scores(y[te], p, float(y[tr].mean()))
            ext[mname] = {"rmse": unit_scale * s["rmse"], "q2": s["q2"],
                          "max_error": unit_scale * float(np.max(np.abs(p - y[te])))}
        out["targets"][tname] = {"unit": unit, "range": [float(unit_scale * y.min()),
                                                          float(unit_scale * y.max())],
                                 "cv": results["cv"],
                                 "cv_geometry": results["cv_geometry"], "extrapolation": ext,
                                 "split": {"train": int(len(tr)), "test": int(len(te)),
                                           "atoms": SPLIT_ATOMS}}
    jdump(out, os.path.join(RESULTS, "model.json"))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
#: the reference each target's folds are counted against: the physical law of its size
REFERENCE = {"energy per atom": "size law", "energy of the particle": "liquid drop"}


def _ref_key(T: Dict, tname: str) -> str:
    return next(k for k in T["cv_geometry"] if k.startswith(REFERENCE[tname]))


def report() -> str:
    lines = ["# Size and shape as descriptors (plan item NT3)", ""]
    gp = os.path.join(RESULTS, "geometry.json")
    if os.path.isfile(gp):
        G = jload(gp)
        ico = G["icosahedra"]
        right = sum(r["surface_atoms"] == r["expected"] for r in ico)
        lines += ["## The descriptors against answers known without them", "",
                  "Surface atoms are the atoms a probe sphere of 1.4 A can touch (the "
                  "Shrake-Rupley method with 200 points per atom, atoms with their covalent "
                  "radii).", "",
                  f"Copper icosahedra of 2 to 12 shells: the surface atoms equal the atoms of "
                  f"the outer shell, 10 (k - 1)^2 + 2, in {right} of {len(ico)}.", "",
                  "| Shells | Atoms | Surface atoms found | Atoms of the outer shell | Time, s |",
                  "|---:|---:|---:|---:|---:|"]
        for r in ico:
            lines.append(f"| {r['shells']} | {r['atoms']:,} | {r['surface_atoms']:,} | "
                         f"{r['expected']:,} | {r['seconds']:.2f} |")
        lines += ["", "Other fcc particles: the surface atoms against the atoms with fewer than "
                  "twelve nearest neighbours.", "",
                  "| Shape | Atoms | Surface atoms found | Atoms with fewer than 12 neighbours "
                  "| Atoms classed differently |", "|---|---:|---:|---:|---:|"]
        for r in G["fcc"]:
            lines.append(f"| {r['shape']} | {r['atoms']:,} | {r['surface_atoms']:,} | "
                         f"{r['fewer_than_12']:,} | {r['disagree']} |")
        tubes = G["tubes"]
        dd = np.array([t["diameter"] - t["formula"] for t in tubes])
        dg = np.array([2 * t["gyration_radius"] - t["formula"] for t in tubes])
        lines += ["", f"{len(tubes)} carbon nanotubes (n, m), {min(t['n'] for t in tubes)} <= n "
                  f"<= {max(t['n'] for t in tubes)}: the diameter (the largest distance between "
                  "two atom centres across the axis) differs from a sqrt(n^2 + n m + m^2) / pi "
                  f"by at most {np.max(np.abs(dd)):.1e} A, and twice the gyration radius by at "
                  f"most {np.max(np.abs(dg)):.1e} A.", "",
                  "Blocks of fcc copper of 3 x 3 x 3L cubic cells: aspect ratio against the "
                  "ratio of the sides.", "", "| Cells | Atoms | Aspect ratio | Ratio of the "
                  "sides |", "|---|---:|---:|---:|"]
        for r in G["blocks"]:
            lines.append(f"| {' x '.join(str(c) for c in r['cells'])} | {r['atoms']:,} | "
                         f"{r['aspect_ratio']:.3f} | {r['side_ratio']:g} |")
        lines += ["", "| Structure | Atoms | Surface fraction | Diameter, A |",
                  "|---|---:|---:|---:|"]
        for r in G["edge"]:
            d = r["diameter"]
            lines.append(f"| {r['structure']} | {r['atoms']} | {r['surface_fraction']:.2f} | "
                         f"{'none (NaN)' if not np.isfinite(d) else f'{d:.2f}'} |")
        lines.append("")
    mp = os.path.join(RESULTS, "model.json")
    if os.path.isfile(mp):
        M = jload(mp)
        lines += ["## Size and shape beside the periodic autocorrelation", "",
                  f"{M['n']} unrelaxed particles of Cu, Ag, Au, Ni, Pd and Pt ({M['atoms'][0]:,} "
                  f"to {M['atoms'][1]:,} atoms; the set of item NT2). The size and shape "
                  f"descriptors of all of them took {M['descriptor_seconds']:.1f} s. Kernel "
                  "ridge regression on the model's own kernel, cutoff 8 A, the ridge parameter "
                  "chosen by an inner 5-fold cross-validation and the descriptor statistics "
                  "taken from each training part alone.", ""]
        for tname, T in M["targets"].items():
            u = T["unit"]
            lines += [f"### Target: the {tname} (EMT), {T['range'][0]:.4g} to "
                      f"{T['range'][1]:.4g} {u}", "",
                      f"5-fold cross-validation repeated with {len(SEEDS)} seeds (mean, and "
                      "the standard deviation over the seeds):", "",
                      f"| Model | RMSE, {u} | Standard deviation over seeds, {u} | q2 |",
                      "|---|---:|---:|---:|"]
            for m, s in T["cv"].items():
                lines.append(f"| {m} | {s['rmse']:.4g} | {s['rmse_sd']:.2g} | {s['q2']:.4f} |")
            if "cv_geometry" in T:
                lines += ["", "5-fold cross-validation that holds out each of the 25 geometries "
                          "(shape and size) in all six metals at once, repeated with "
                          f"{len(SEEDS)} seeds; the inner folds that choose the ridge parameter "
                          "hold out geometries too:", "",
                          f"| Model | RMSE, {u} | Standard deviation over seeds, {u} | q2 |",
                          "|---|---:|---:|---:|"]
                for m, s in T["cv_geometry"].items():
                    lines.append(f"| {m} | {s['rmse']:.4g} | {s['rmse_sd']:.2g} | "
                                 f"{s['q2']:.4f} |")
                lines += ["", "The same folds one by one (25 folds: 5 seeds of 5 folds). "
                          f"'Folds won' counts the folds in which the model's error is below "
                          f"that of the {REFERENCE[tname]}.", "",
                          f"| Model | Median error of a fold, {u} | Largest error of a fold, {u} "
                          "| Held out in that fold | Folds won |", "|---|---:|---:|---|---:|"]
                ref = [f["rmse"] for f in T["cv_geometry"][_ref_key(T, tname)]["folds"]]
                for m, s in T["cv_geometry"].items():
                    errs = [f["rmse"] for f in s["folds"]]
                    worst = s["folds"][int(np.argmax(errs))]
                    won = sum(e < r for e, r in zip(errs, ref))
                    lines.append(f"| {m} | {np.median(errs):.4g} | {max(errs):.4g} | "
                                 f"{', '.join(worst['held_out'])} | ")
                    count = "--" if m == _ref_key(T, tname) else f"{won} of {len(errs)}"
                    lines[-1] = lines[-1] + f"{count} |"
            sp = T["split"]
            lines += ["", f"Extrapolation in size: trained on the {sp['train']} particles of at "
                      f"most {sp['atoms']:,} atoms, tested on the {sp['test']} larger ones (q2 "
                      "against the mean of the training part):", "",
                      f"| Model | RMSE, {u} | Largest error, {u} | q2 |", "|---|---:|---:|---:|"]
            for m, s in T["extrapolation"].items():
                lines.append(f"| {m} | {s['rmse']:.4g} | {s['max_error']:.4g} | {s['q2']:.4f} |")
            lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("geometry", "model"), action="append",
                    help="stage(s) to run; the report always follows")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    for st in args.stage or []:
        if st == "geometry":
            stage_geometry()
        elif st == "model":
            stage_model()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
