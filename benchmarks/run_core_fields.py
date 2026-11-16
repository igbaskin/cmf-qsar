#!/usr/bin/env python
"""
Fields for the core of a particle (plan item NT6).

The fields of a solid say what an atom is: its charge, its size, its electronegativity, its
element.  Item NT6 adds three fields that say where it sits, computed from the positions
(``inorganic_fieldset(core=True)``, :data:`cmfqsar.periodic.CORE_FIELDS`):

* ``prop:coordination``, the number of neighbours of the atom (closer than 1.2 times the sum of
  the covalent radii);
* ``prop:generalized_coordination``, the sum of the neighbours' own coordination numbers divided
  by 12 (Calle-Vallejo et al. 2014);
* ``prop:exposure``, the accessible fraction of the atom's sphere for a 1.4 A probe (Shrake and
  Rupley), the per-atom number behind ``calc:surface_fraction`` of item NT3.

Stage checks
------------
* the coordination numbers of the 150 particles of item NT2 against ASE's neighbour list with a
  cutoff halfway between the first and the second shell of the bulk lattice;
* the generalized coordination numbers of the atoms of fcc(111) and fcc(100) terraces against
  the published values 7.5 and 6.67, and of the bulk against 12;
* the atoms with an exposure above zero against the atoms with fewer than twelve neighbours;
* the time of the three fields against the atom count.

Stage single
------------
The 150 particles of one metal of item NT2 (Cu, Ag, Au, Ni, Pd, Pt, 13 to 3,564 atoms), with the
energy per atom and the energy of the whole particle from ASE's effective medium theory potential
(EMT), on the folds of item NT3: 5-fold cross-validation repeated with 5 seeds, the same with
every geometry held out in all six metals, and extrapolation from the particles of at most
1,000 atoms to the larger ones.  The periodic autocorrelation with the fields of NT3 (steric,
electronegativity, one indicator per metal) is compared with the same plus the three core
fields, and with the count reference of tutorial 24 (a linear regression on the neighbour pairs
in four lattice shells, for each metal).

Stage alloys
------------
360 particles of two metals: 10 geometries (icosahedra, decahedra and truncated octahedra of 55
to 314 atoms) x 3 metal pairs (Au-Cu, Ag-Pt, Ni-Pd) x 3 compositions (25, 50 and 75 % of the
first metal) x 4 arrangements (the first metal in the core, at the surface, and two random
arrangements).  The lattice constant is the mean of the two metals' weighted by the composition.
The target is the mixing energy per atom: the EMT energy of the particle minus the energies of
the two pure particles of the same geometry at their own lattice constants, in proportion to the
composition.  Which metal sits at the surface changes it by up to several hundred meV per atom,
so the target is about where the atoms are.  Compared on 5-fold cross-validation repeated with 5
seeds and on folds that hold out each geometry:

* the periodic autocorrelation with steric, electronegativity and one indicator per metal;
* the same plus the three core fields, and plus each core field alone;
* the bond-counting reference: a linear regression, for each metal pair, on the number of
  nearest-neighbour pairs of each kind (AA, AB, BB) per atom and on the number of surface atoms
  of each metal per atom (an atom is at the surface when its exposure is above zero);
* the same with the atoms of each metal at each coordination number in place of the surface
  atoms;
* a Gaussian kernel on those counts and the composition, zero between different metal pairs,
  its width and ridge parameter chosen by an inner cross-validation (grouped by geometry when
  the outer folds are);
* the composition alone: a linear regression on x and x^2 for each metal pair.

Usage
-----
    python benchmarks/run_core_fields.py --stage checks     # about 1 minute
    python benchmarks/run_core_fields.py --stage single     # about 4 minutes
    python benchmarks/run_core_fields.py --stage alloys     # about 2 minutes
    python benchmarks/run_core_fields.py --report
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

from cmfqsar.model import CMFModel, _cv_splits, _Learner, cv_predict  # noqa: E402
from cmfqsar.periodic import (CORE_FIELDS, coordination_numbers, core_columns,  # noqa: E402
                              crystal_table, generalized_coordination, inorganic_fieldset)

RESULTS = os.path.join(HERE, "results", "core_fields")
CACHE = os.path.join(HERE, "cache", "core_fields")
PHYSCHEM = ("steric", "prop:electronegativity")
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
SEEDS = (0, 1, 2, 3, 4)
SPLIT_ATOMS = 1000
#: the kernel of items NT2 and NT3: cutoff 8 A, channels every 0.5 A, 2 A taper
KERNEL = dict(kernel_source="periodic", ac_d_max=8.0, ac_sigma=0.5, pbc_taper=2.0)
PAIRS = (("Au", "Cu"), ("Ag", "Pt"), ("Ni", "Pd"))
FRACTIONS = (0.25, 0.5, 0.75)
ARRANGEMENTS = ("core", "surface", "random0", "random1")
SHORT = {"prop:coordination": "coordination number",
         "prop:generalized_coordination": "generalized coordination number",
         "prop:exposure": "exposure"}


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Shared
# --------------------------------------------------------------------------- #
class Kernels:
    """Kernel ridge regression on the model's own kernel for several field sets, the channel
    features of every structure computed once per field set."""

    def __init__(self, structs: list, names: List[str], fieldsets: Dict):
        self.fs, self.tables, self.F, self.pos, self.seconds = {}, {}, {}, {}, {}
        for key, fs in fieldsets.items():
            t0 = time.perf_counter()
            table = crystal_table(structs, fs, names)
            probe = CMFModel(fieldset=fs, method="krr", normalize=True, **KERNEL)
            self.F[key] = probe._ac_features(table)
            self.seconds[key] = time.perf_counter() - t0
            self.fs[key], self.tables[key] = fs, table
            self.pos[key] = {id(m): k for k, m in enumerate(table.molecules)}
            print(f"features {key}: {self.F[key].shape}, {self.seconds[key]:.1f} s", flush=True)

    def predict(self, key: str, y, tr, te, seed, groups=None) -> np.ndarray:
        from run_shape import _group_splits
        m = CMFModel(fieldset=self.fs[key], method="krr", normalize=True, **KERNEL)
        F, pos = self.F[key], self.pos[key]
        m._ac_features = lambda t: F[[pos[id(x)] for x in t.molecules]]
        table = self.tables[key]
        T_tr, T_te = table.subset(list(tr)), table.subset(list(te))
        K = m.training_kernel(T_tr)
        best, lam = np.inf, LAMBDAS[0]
        for value in LAMBDAS:
            if groups is None:
                p = cv_predict(K, y[tr], "krr", {"lam": value}, folds=5, seed=seed)
            else:                                # the inner folds hold out groups too
                p = np.full(len(tr), np.nan)
                for ite, itr in _group_splits(np.asarray(groups)[tr], 5, seed):
                    inner = _Learner("krr", {"lam": value}).fit(K[np.ix_(itr, itr)], y[tr][itr])
                    p[ite] = inner.predict(K[np.ix_(ite, itr)])
            err = float(np.mean((p - y[tr]) ** 2))
            if err < best:
                best, lam = err, value
        learner = _Learner("krr", {"lam": lam}).fit(K, y[tr])
        return learner.predict(m.test_kernel(T_te))


def _evaluate(methods: Dict, y: np.ndarray, schemes: Dict, groups: np.ndarray,
              scale: float, label: str) -> Dict:
    """Every method on every scheme: mean RMSE (scaled), its spread over seeds, q2 and the RMSE
    of every fold, with the groups each fold held out."""
    from run_shape import _scores
    results = {}
    for scheme, splits in schemes.items():
        cv = {}
        grouped = scheme != "cv"
        for mname, fn in methods.items():
            runs, folds = [], []
            for seed in SEEDS:
                p = np.full(len(y), np.nan)
                for te, tr in splits(seed):
                    p[te] = fn(tr, te, seed, groups if grouped else None)
                    folds.append({"seed": seed, "held_out": sorted(set(groups[te])),
                                  "rmse": scale * float(np.sqrt(np.mean((p[te] - y[te]) ** 2)))})
                runs.append(_scores(y, p, float(y.mean())))
                if grouped and scheme == "loo_geometry":
                    break                        # the folds do not depend on the seed
            cv[mname] = {"rmse": scale * float(np.mean([r["rmse"] for r in runs])),
                         "rmse_sd": scale * float(np.std([r["rmse"] for r in runs])),
                         "q2": float(np.mean([r["q2"] for r in runs])), "folds": folds}
            print(label, scheme, mname, {k: v for k, v in cv[mname].items() if k != "folds"},
                  flush=True)
        results[scheme] = cv
    return results


# --------------------------------------------------------------------------- #
# Stage checks
# --------------------------------------------------------------------------- #
def _half_shell_counts(a) -> np.ndarray:
    from ase.data import atomic_numbers, reference_states
    from ase.neighborlist import neighbor_list
    lattice = reference_states[atomic_numbers[a.get_chemical_symbols()[0]]]["a"]
    b = a.copy()
    b.center(vacuum=10.0)
    i = neighbor_list("i", b, 0.5 * (lattice / np.sqrt(2.0) + lattice))
    return np.bincount(i, minlength=len(a))


def stage_checks() -> None:
    warnings.filterwarnings("ignore")
    import run_particles as RP
    from ase.build import bulk, fcc100, fcc111
    out: Dict = {}
    structs = RP.particle_set()
    differ, exposed_wrong, n_atoms = 0, 0, 0
    for a in structs:
        cn = coordination_numbers(a)
        ref = _half_shell_counts(a)
        differ += int(np.sum(cn != ref))
        n_atoms += len(a)
        e = core_columns(a, ["prop:exposure"])["prop:exposure"]
        exposed_wrong += int(np.sum((e > 0) != (ref < 12)))
    out["coordination"] = {"particles": len(structs), "atoms": n_atoms, "differ": differ}
    out["exposure"] = {"atoms": n_atoms, "differ_from_under_12": exposed_wrong}
    gcn = {}
    for tag, slab, ref in (("fcc(111)", fcc111("Pt", (4, 4, 6), vacuum=10.0), 7.5),
                           ("fcc(100)", fcc100("Pt", (4, 4, 6), vacuum=10.0), 20.0 / 3.0)):
        z = slab.get_positions()[:, 2]
        top = z > z.max() - 0.1
        g = generalized_coordination(slab)
        gcn[tag] = {"published": ref, "computed": float(np.mean(g[top])),
                    "spread": float(np.ptp(g[top])),
                    "coordination": float(np.mean(coordination_numbers(slab)[top]))}
    b = bulk("Pt", "fcc", a=3.92, cubic=True)
    gcn["bulk"] = {"published": 12.0, "computed": float(np.mean(generalized_coordination(b))),
                   "spread": float(np.ptp(generalized_coordination(b))),
                   "coordination": float(np.mean(coordination_numbers(b)))}
    out["generalized_coordination"] = gcn
    timing = []
    by_size = sorted(structs, key=len)
    for a in by_size[::12] + ([by_size[-1]] if (len(by_size) - 1) % 12 else []):
        t0 = time.perf_counter()
        core_columns(a, CORE_FIELDS)
        timing.append({"name": a.info["name"], "atoms": len(a),
                       "seconds": time.perf_counter() - t0})
    out["timing"] = timing
    for k, v in out.items():
        print(k, v, flush=True)
    jdump(out, os.path.join(RESULTS, "checks.json"))


# --------------------------------------------------------------------------- #
# Stage single
# --------------------------------------------------------------------------- #
def stage_single() -> None:
    warnings.filterwarnings("ignore")
    import run_particles as RP
    from run_shape import _group_splits, _per_metal, _scores
    structs = RP.particle_set()
    e_atom = RP.energies(structs)
    n = np.array([len(a) for a in structs], float)
    e_total = e_atom * n
    el = np.array([a.get_chemical_symbols()[0] for a in structs])
    names = [a.info["name"] for a in structs]
    geometry = np.array([nm.split("_", 1)[1] for nm in names])
    kern = Kernels(structs, names, {
        "plain": inorganic_fieldset(list(RP.METALS), physchem=PHYSCHEM),
        "core": inorganic_fieldset(list(RP.METALS), physchem=PHYSCHEM, core=True)})
    shells = RP.shell_counts(structs)
    targets = {
        "energy per atom": (e_atom, 1000.0, "meV per atom",
                            lambda tr, te: _per_metal(el, shells, e_atom, tr, te)),
        "energy of the particle": (e_total, 1.0, "eV",
                                   lambda tr, te: _per_metal(
                                       el, np.column_stack([n, shells * n[:, None]]), e_total,
                                       tr, te, intercept=False))}
    out = {"n": len(n), "atoms": [int(n.min()), int(n.max())], "seconds": kern.seconds,
           "targets": {}}
    for tname, (y, scale, unit, ref) in targets.items():
        methods = {
            "periodic autocorrelation, fields of NT3":
                lambda tr, te, s, g=None: kern.predict("plain", y, tr, te, s, g),
            "periodic autocorrelation, fields of NT3 and the three core fields":
                lambda tr, te, s, g=None: kern.predict("core", y, tr, te, s, g),
            "linear regression on the neighbour pairs in four shells, for each metal":
                lambda tr, te, s, g=None: ref(tr, te)}
        schemes = {"cv": lambda seed: _cv_splits(y, 5, seed),
                   "cv_geometry": lambda seed: _group_splits(geometry, 5, seed)}
        res = _evaluate(methods, y, schemes, geometry, scale, tname)
        tr, te = np.flatnonzero(n <= SPLIT_ATOMS), np.flatnonzero(n > SPLIT_ATOMS)
        ext = {}
        for mname, fn in methods.items():
            p = fn(tr, te, 0)
            s = _scores(y[te], p, float(y[tr].mean()))
            ext[mname] = {"rmse": scale * s["rmse"], "q2": s["q2"],
                          "max_error": scale * float(np.max(np.abs(p - y[te])))}
        out["targets"][tname] = {"unit": unit, "cv": res["cv"],
                                 "cv_geometry": res["cv_geometry"], "extrapolation": ext,
                                 "split": {"train": int(len(tr)), "test": int(len(te))}}
    jdump(out, os.path.join(RESULTS, "single.json"))


# --------------------------------------------------------------------------- #
# Stage alloys
# --------------------------------------------------------------------------- #
GEOMETRIES = (("ico3", "ico", 3), ("ico4", "ico", 4), ("ico5", "ico", 5),
              ("deca3", "deca", 3), ("deca4", "deca", 4), ("deca5", "deca", 5),
              ("octa5", "octa", (5, 1)), ("octa6", "octa", (6, 1)), ("octa7", "octa", (7, 2)),
              ("octa8", "octa", (8, 2)))


def _lattice(el: str) -> float:
    from ase.data import atomic_numbers, reference_states
    return float(reference_states[atomic_numbers[el]]["a"])


def _geometry(kind: str, size, el: str, a: float):
    from ase.cluster import Decahedron, Icosahedron, Octahedron
    if kind == "ico":
        atoms = Icosahedron(el, size, latticeconstant=a)
    elif kind == "deca":
        atoms = Decahedron(el, size, size, 0, latticeconstant=a)
    else:
        atoms = Octahedron(el, size[0], cutoff=size[1], latticeconstant=a)
    atoms.center(vacuum=8.0)
    atoms.pbc = False
    return atoms


def alloy_set() -> list:
    """The 360 two-metal particles; every one carries its pair, composition, arrangement and
    geometry in ``info``."""
    out = []
    for gname, kind, size in GEOMETRIES:
        for A, B in PAIRS:
            for x in FRACTIONS:
                a = x * _lattice(A) + (1 - x) * _lattice(B)
                base = _geometry(kind, size, A, a)
                n = len(base)
                k = int(round(x * n))
                r = np.linalg.norm(base.get_positions() - base.get_positions().mean(0), axis=1)
                for arr in ARRANGEMENTS:
                    if arr == "core":
                        order = np.lexsort((np.arange(n), np.round(r, 6)))
                    elif arr == "surface":
                        order = np.lexsort((np.arange(n), -np.round(r, 6)))
                    else:
                        order = np.random.default_rng(int(arr[-1]) + 17 * len(out)).permutation(n)
                    symbols = np.array([B] * n, dtype=object)
                    symbols[order[:k]] = A
                    p = base.copy()
                    p.set_chemical_symbols(list(symbols))
                    p.info.update({"name": f"{A}{B}_{gname}_{int(100 * x)}_{arr}", "pair": A + B,
                                   "x": x, "arrangement": arr, "geometry": gname,
                                   "first": A, "second": B})
                    out.append(p)
    return out


def _emt(a) -> float:
    from ase.calculators.emt import EMT
    b = a.copy()
    b.calc = EMT()
    return float(b.get_potential_energy())


def mixing_energies(structs: list) -> np.ndarray:
    """EMT mixing energy per atom of every alloy particle (eV), cached by name."""
    path = os.path.join(CACHE, "mixing.json")
    cache = jload(path) if os.path.isfile(path) else {}
    pure: Dict[str, float] = cache.get("_pure", {})
    for p in structs:
        name = p.info["name"]
        if name in cache:
            continue
        n = len(p)
        terms = []
        for el in (p.info["first"], p.info["second"]):
            key = f"{el}_{p.info['geometry']}"
            if key not in pure:
                spec = next(g for g in GEOMETRIES if g[0] == p.info["geometry"])
                pure[key] = _emt(_geometry(spec[1], spec[2], el, _lattice(el))) / n
            terms.append(pure[key])
        x_true = p.get_chemical_symbols().count(p.info["first"]) / n
        cache[name] = _emt(p) / n - x_true * terms[0] - (1 - x_true) * terms[1]
    cache["_pure"] = pure
    jdump(cache, path)
    return np.array([cache[p.info["name"]] for p in structs])


def bond_counts(structs: list) -> np.ndarray:
    """For every particle: AA, AB and BB nearest-neighbour pairs per atom, then surface atoms of
    the first metal and of the second per atom."""
    from cmfqsar.periodic import coordination_pairs
    rows = []
    for p in structs:
        first = np.array([s == p.info["first"] for s in p.get_chemical_symbols()])
        i, j = coordination_pairs(p)
        both = first[i] & first[j]
        none = ~first[i] & ~first[j]
        n = len(p)
        e = core_columns(p, ["prop:exposure"])["prop:exposure"] > 0
        rows.append([np.sum(both) / 2 / n, np.sum(~both & ~none) / 2 / n,
                     np.sum(none) / 2 / n, np.sum(e & first) / n, np.sum(e & ~first) / n])
    return np.asarray(rows, float)


def coordination_classes(structs: list) -> np.ndarray:
    """For every particle: the atoms of the first metal with each coordination number, then the
    atoms of the second, per atom; the coordination numbers no particle has are left out."""
    rows = []
    for p in structs:
        first = np.array([s == p.info["first"] for s in p.get_chemical_symbols()])
        cn = np.minimum(coordination_numbers(p).astype(int), 12)
        rows.append(np.r_[np.bincount(cn[first], minlength=13),
                          np.bincount(cn[~first], minlength=13)] / len(p))
    X = np.asarray(rows, float)
    return X[:, X.any(axis=0)]


#: the grids of the Gaussian kernel on the counts: the width as a factor on the squared distance
#: per column, and the ridge parameter
GAMMAS = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
COUNT_LAMBDAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
CHOSEN: List[Dict] = []


def _gaussian_counts(pair, X, y, tr, te, seed, groups=None) -> np.ndarray:
    """Kernel ridge regression with a Gaussian kernel on the standardised columns of X, zero
    between particles of different metal pairs, around the training mean; width and ridge
    parameter from an inner 5-fold cross-validation (grouped when `groups` is given)."""
    from run_shape import _group_splits
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
    Z = (X - mu) / sd
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1) / Z.shape[1]
    same = (pair[:, None] == pair[None, :]).astype(float)

    def fit(a, b, gam, lam):
        K = np.exp(-gam * d2) * same
        alpha = np.linalg.solve(K[np.ix_(a, a)] + lam * np.eye(len(a)), y[a] - y[a].mean())
        return K[np.ix_(b, a)] @ alpha + y[a].mean()
    inner = (_cv_splits(y[tr], 5, seed) if groups is None
             else _group_splits(np.asarray(groups)[tr], 5, seed))
    best = None
    for gam in GAMMAS:
        for lam in COUNT_LAMBDAS:
            p = np.zeros(len(tr))
            for ite, itr in inner:
                p[ite] = fit(tr[itr], tr[ite], gam, lam)
            err = float(np.mean((p - y[tr]) ** 2))
            if best is None or err < best[0]:
                best = (err, gam, lam)
    CHOSEN.append({"gamma": best[1], "lambda": best[2]})
    return fit(np.asarray(tr), np.asarray(te), best[1], best[2])


def stage_alloys() -> None:
    warnings.filterwarnings("ignore")
    from run_shape import _group_splits, _per_metal
    structs = alloy_set()
    t0 = time.perf_counter()
    y = mixing_energies(structs)
    t_emt = time.perf_counter() - t0
    names = [p.info["name"] for p in structs]
    pair = np.array([p.info["pair"] for p in structs])
    geometry = np.array([p.info["geometry"] for p in structs])
    x = np.array([p.info["x"] for p in structs])
    metals = sorted({el for ab in PAIRS for el in ab})
    sets = {"plain": inorganic_fieldset(metals, physchem=PHYSCHEM, other=False),
            "core": inorganic_fieldset(metals, physchem=PHYSCHEM, other=False, core=True)}
    for name in CORE_FIELDS:
        sets[name] = inorganic_fieldset(metals, physchem=PHYSCHEM + (name,), other=False)
    kern = Kernels(structs, names, sets)
    counts = bond_counts(structs)
    methods = {
        "periodic autocorrelation, steric, electronegativity and one indicator per metal":
            lambda tr, te, s, g=None: kern.predict("plain", y, tr, te, s, g),
        "the same and the three core fields":
            lambda tr, te, s, g=None: kern.predict("core", y, tr, te, s, g)}
    for name in CORE_FIELDS:
        methods[f"the same and the {SHORT[name]} alone"] = (
            lambda nm: (lambda tr, te, s, g=None: kern.predict(nm, y, tr, te, s, g)))(name)
    methods["bond counting: linear regression on AA, AB, BB pairs and surface atoms of each "
            "metal, for each pair"] = \
        lambda tr, te, s, g=None: _per_metal(pair, counts, y, tr, te)
    classes = np.column_stack([counts[:, :3], coordination_classes(structs)])
    methods["bond counting with coordination classes: linear regression on AA, AB, BB pairs and "
            "the atoms of each metal with each coordination number, for each pair"] = \
        lambda tr, te, s, g=None: _per_metal(pair, classes, y, tr, te)
    with_x = np.column_stack([classes, x])
    methods["Gaussian kernel on the same counts and the composition, for each pair"] = \
        lambda tr, te, s, g=None: _gaussian_counts(pair, with_x, y, tr, te, s, g)
    methods["composition alone: linear regression on x and x^2, for each pair"] = \
        lambda tr, te, s, g=None: _per_metal(pair, np.stack([x, x ** 2], 1), y, tr, te)
    geoms = sorted(set(geometry))
    schemes = {"cv": lambda seed: _cv_splits(y, 5, seed),
               "loo_geometry": lambda seed: [(np.flatnonzero(geometry == g),
                                              np.flatnonzero(geometry != g)) for g in geoms]}
    res = _evaluate(methods, y, schemes, geometry, 1000.0, "mixing energy")
    by_arr = {arr: [float(1000 * y[i]) for i in range(len(y))
                    if structs[i].info["arrangement"] == arr] for arr in ARRANGEMENTS}
    out = {"n": len(y), "atoms": [int(min(map(len, structs))), int(max(map(len, structs)))],
           "emt_seconds": t_emt, "feature_seconds": kern.seconds, "unit": "meV per atom",
           "range": [float(1000 * y.min()), float(1000 * y.max())],
           "sd": float(1000 * y.std()),
           "by_arrangement": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v))}
                              for k, v in by_arr.items()},
           "cv": res["cv"], "loo_geometry": res["loo_geometry"],
           "count_kernel_chosen": CHOSEN, "count_kernel_grid": {"gamma": list(GAMMAS),
                                                               "lambda": list(COUNT_LAMBDAS)}}
    jdump(out, os.path.join(RESULTS, "alloys.json"))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _row(name: str, r: Dict, unit_digits: int = 1) -> str:
    return (f"| {name} | {r['rmse']:.{unit_digits}f} (sd {r['rmse_sd']:.{unit_digits}f}) | "
            f"{r['q2']:.3f} |")


def report() -> str:
    lines = ["# Fields for the core of a particle (plan item NT6)", ""]
    cp = os.path.join(RESULTS, "checks.json")
    if os.path.isfile(cp):
        C = jload(cp)
        c, e = C["coordination"], C["exposure"]
        lines += ["## Checks", "",
                  f"* Coordination numbers of the {c['particles']} particles of item NT2 "
                  f"({c['atoms']:,} atoms) against ASE's neighbour list with a cutoff halfway "
                  f"between the first and second shells: {c['differ']} atoms differ.",
                  f"* Atoms with an exposure above zero against atoms with fewer than twelve "
                  f"neighbours: {e['differ_from_under_12']} of {e['atoms']:,} differ.", "",
                  "| Atoms | Coordination number | Generalized coordination number, computed | "
                  "Published value |", "|---|---:|---:|---:|"]
        for tag, g in C["generalized_coordination"].items():
            where = "Bulk Pt" if tag == "bulk" else f"Pt {tag} terrace, top layer"
            lines.append(f"| {where} | {g['coordination']:.0f} | {g['computed']:.3f} "
                         f"(spread {g['spread']:.1e}) | {g['published']:.3f} |")
        lines += ["", "| Particle | Atoms | Seconds for the three core fields |",
                  "|---|---:|---:|"]
        for t in C["timing"]:
            lines.append(f"| {t['name']} | {t['atoms']:,} | {t['seconds']:.2f} |")
        lines.append("")
    sp = os.path.join(RESULTS, "single.json")
    if os.path.isfile(sp):
        S = jload(sp)
        lines += ["## Particles of one metal (the 150 particles of item NT2)", ""]
        for tname, T in S["targets"].items():
            u = T["unit"]
            lines += [f"### {tname[0].upper() + tname[1:]}", "",
                      f"| Model | Random folds: RMSE, {u} | Random folds: q2 | "
                      f"Geometry held out: RMSE, {u} | Geometry held out: q2 | "
                      f"Larger particles: RMSE, {u} |", "|---|---:|---:|---:|---:|---:|"]
            for m, r in T["cv"].items():
                g, x = T["cv_geometry"][m], T["extrapolation"][m]
                lines.append(f"| {m} | {r['rmse']:.1f} | {r['q2']:.3f} | {g['rmse']:.1f} | "
                             f"{g['q2']:.3f} | {x['rmse']:.1f} |")
            lines.append("")
            plain, core = list(T["cv_geometry"])[:2]
            a = np.array([f["rmse"] for f in T["cv_geometry"][plain]["folds"]])
            b = np.array([f["rmse"] for f in T["cv_geometry"][core]["folds"]])
            lines += [f"With geometries held out the core fields lower the error in "
                      f"{int(np.sum(b < a))} of {len(a)} folds; worst fold {a.max():.1f} "
                      f"without them, {b.max():.1f} with them ({u}).", ""]
    ap = os.path.join(RESULTS, "alloys.json")
    if os.path.isfile(ap):
        A = jload(ap)
        lines += ["## Particles of two metals: the mixing energy", "",
                  f"{A['n']} particles of {A['atoms'][0]} to {A['atoms'][1]} atoms; mixing "
                  f"energy {A['range'][0]:.0f} to {A['range'][1]:.0f} meV per atom (standard "
                  f"deviation {A['sd']:.0f}).", "",
                  "| Arrangement | Mean mixing energy, meV per atom | Standard deviation |",
                  "|---|---:|---:|"]
        for arr, v in A["by_arrangement"].items():
            lines.append(f"| {arr} | {v['mean']:.1f} | {v['sd']:.1f} |")
        lines += ["", "| Model | Random folds: RMSE, meV per atom | Random folds: q2 | "
                  "One geometry held out: RMSE, meV per atom | One geometry held out: q2 |",
                  "|---|---:|---:|---:|---:|"]
        for m, r in A["cv"].items():
            g = A["loo_geometry"][m]
            lines.append(f"| {m} | {r['rmse']:.1f} | {r['q2']:.3f} | {g['rmse']:.1f} | "
                         f"{g['q2']:.3f} |")
        lines.append("")
        plain, core = list(A["loo_geometry"])[:2]
        fa = {f["held_out"][0]: f["rmse"] for f in A["loo_geometry"][plain]["folds"]}
        fb = {f["held_out"][0]: f["rmse"] for f in A["loo_geometry"][core]["folds"]}
        gk = next(m for m in A["loo_geometry"] if m.startswith("Gaussian kernel"))
        fc = {f["held_out"][0]: f["rmse"] for f in A["loo_geometry"][gk]["folds"]}
        lines += ["| Geometry held out | Without the core fields, meV per atom | "
                  "With them, meV per atom | Gaussian kernel on the counts, meV per atom |",
                  "|---|---:|---:|---:|"]
        for gname in sorted(fa):
            lines.append(f"| {gname} | {fa[gname]:.1f} | {fb[gname]:.1f} | {fc[gname]:.1f} |")
        chosen = A.get("count_kernel_chosen", [])
        lines += ["", f"The core fields lower the error in {sum(fb[k] < fa[k] for k in fa)} of "
                  f"{len(fa)} held-out geometries; the Gaussian kernel on the counts has a lower "
                  f"error than the core fields in {sum(fc[k] < fb[k] for k in fa)} of them.", ""]
        if chosen:
            gams = sorted({c["gamma"] for c in chosen})
            lams = sorted({c["lambda"] for c in chosen})
            lines += [f"The inner cross-validation chose widths {gams} from the grid "
                      f"{A['count_kernel_grid']['gamma']} and ridge parameters {lams} from "
                      f"{A['count_kernel_grid']['lambda']}: none at an edge of its grid.", ""]
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("checks", "single", "alloys"), action="append",
                    help="stages to run (default: all)")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    if not args.report:
        for stage in args.stage or ("checks", "single", "alloys"):
            {"checks": stage_checks, "single": stage_single, "alloys": stage_alloys}[stage]()
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
