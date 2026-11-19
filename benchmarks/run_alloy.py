#!/usr/bin/env python
"""
Hydrogen and hydroxyl adsorption on bimetallic alloy surfaces: field kernels against the names
and the properties of the elements (plan item EC12).

The data are the slabs of Mamun, Winther, Boes and Bligaard (Scientific Data 6, 2019): a 2 x 2 cell
of three metal layers, 40 metals, with the adsorption energy of H (the descriptor of the hydrogen
evolution reaction, HER) and of OH (the first descriptor of the oxygen reduction reaction, ORR; the
set has no OOH).  ``examples/make_alloy_examples.py`` builds ``examples/alloy`` from the Quantum
ESPRESSO logs.  Every adsorption energy is relative to the bare slab and to H2 (and H2O).

Two questions, two kinds of sample
----------------------------------
slab   The sample is the bare slab and the target is the lowest adsorption energy over its sites.
       A site is not given.  This is what a catalyst search wants: rank slabs.
site   The sample is one site of one slab: the point above the adsorbate, in the frame of the
       lattice, at a fixed height over the top layer (1.0 A for H, 1.4 A for OH).  The target is
       the adsorption energy found for that site.  The relaxed height of the adsorbate is not
       used, because it would carry part of the answer.

Why the element names are the reference
--------------------------------------
In the single-atom pilot (EC8) the element names explained most of the binding energy and the
fields did not pass their gate.  An alloy slab is the same case in a harder form: 40 elements and
two or three of them in each slab.  A field model can win only if it knows something the names do
not: how the elements are arranged (the layer, the neighbours of the site) and how different two
elements are.  So the references are

* element fractions of the slab (one column for each element), linear and Gaussian kernels;
* tabulated properties of the elements (electronegativity, d-electron count, ionisation energy,
  radius, ...) averaged over the slab, or over the top layer and the other layers separately;
* for a site, the same two kinds taken over the top-layer atoms that bind the adsorbate, with the
  kind of site (top, bridge, fcc or hcp hollow);

and the field methods are the periodic autocorrelation of the slab (rotation and translation
invariant), the rotation-invariant autocorrelation of the point environment of a site, and the
overlap kernel of that environment in the frame of the lattice.

Folds
-----
random      slabs (or sites) dealt at random into 10 folds
slab-out    for sites: every site of a slab is in one fold
alloy-out   all slabs with the same elements are in one fold (Pt-Cu and Cu-Pt are one alloy):
            a pair of elements the model has never seen together, each element seen before

Every method chooses its kernel and its ridge in inner folds of the same kind as the outer ones.
The differences between two methods are paired bootstrap intervals of the mean absolute error
(2.5 % to 97.5 %); a negative difference means that the first method has the smaller error.

Usage
-----
    python benchmarks/run_alloy.py                 # slab and site tasks, then the table
    python benchmarks/run_alloy.py --task slab
    python benchmarks/run_alloy.py --report        # rebuild the table from the stored results
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import run_sac as R  # noqa: E402

from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.hops import point_environment  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel  # noqa: E402
from cmfqsar.localenv import local_features, local_kernel  # noqa: E402
from cmfqsar.model import krr_fit, r2_score, rmse  # noqa: E402
from cmfqsar.periodic import (PeriodicSpec, crystal_fields, inorganic_fieldset,  # noqa: E402
                              periodic_feature_matrix)

EXAMPLES = os.path.join(ROOT, "examples", "alloy")
RESULTS = os.path.join(HERE, "results", "alloy")
#: height of the point of a site over the mean height of the top layer, in angstrom
HEIGHT = {"H": 1.0, "OH": 1.4}
LAYER_TOLERANCE = 0.6
#: ridge values, folds, seeds
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
INNER = 5
SEED = 0
RUNS = 2000
RHOS = (0.1, 0.2)
ALPHAS = (0.5, 1.0)
R_MAX = 6.0
PERIODIC_SPEC = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)
PROPS = ("Z", "group", "period", "electronegativity", "d_count", "valence", "ionisation",
         "affinity", "radius")
SITE_KINDS = ("top", "bridge", "fcc", "hcp")
#: the largest number of sites on which the overlap kernel is built
MAX_OVERLAP_SITES = 2500
VARIANTS = {"all": "every geometry", "converged": "geometries that did not stop at the step limit"}


# --------------------------------------------------------------------------- #
# The data
# --------------------------------------------------------------------------- #
class AlloySet:
    """The bare slabs and the kept adsorption geometries of ``examples/alloy``."""

    def __init__(self, path: str = EXAMPLES):
        from ase.io import read
        frames = read(os.path.join(path, "slabs.extxyz"), index=":")
        self.slabs = {int(a.info["slab"]): a for a in frames}
        self.rows: List[dict] = []
        with open(os.path.join(path, "adsorption.csv"), encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                if r["duplicate_of"] or r["site"] == "subsurface":
                    continue
                self.rows.append({"log": r["log"], "slab": int(r["slab"]), "ads": r["adsorbate"],
                                  "e": float(r["e_ads"]), "x": float(r["x"]), "y": float(r["y"]),
                                  "z": float(r["z"]), "site": r["site"],
                                  "bind": r["bind_elements"].split("-"),
                                  "limit": r["step_limit"] == "True"})
        self.elements = sorted({s for a in self.slabs.values() for s in a.get_chemical_symbols()})

    def alloy(self, slab: int) -> str:
        return "-".join(sorted(set(self.slabs[slab].get_chemical_symbols())))

    def layers(self, slab: int) -> Tuple[np.ndarray, np.ndarray]:
        """(top-layer atoms, the other atoms) of a slab."""
        z = self.slabs[slab].positions[:, 2]
        top = z > z.max() - LAYER_TOLERANCE
        return np.where(top)[0], np.where(~top)[0]


def select(data: AlloySet, ads: str, variant: str) -> List[dict]:
    return [r for r in data.rows if r["ads"] == ads and (variant == "all" or not r["limit"])]


def slab_task(data: AlloySet, ads: str, variant: str) -> Tuple[List[int], np.ndarray]:
    """(slab ids, lowest adsorption energy of each slab)."""
    best: Dict[int, float] = {}
    for r in select(data, ads, variant):
        best[r["slab"]] = min(best.get(r["slab"], np.inf), r["e"])
    ids = sorted(best)
    return ids, np.array([best[i] for i in ids])


# --------------------------------------------------------------------------- #
# Reference kernels
# --------------------------------------------------------------------------- #
def standard_properties(elements: Sequence[str]) -> Dict[str, np.ndarray]:
    """Tabulated properties of each element, standardised over the elements of the set."""
    X = np.array([[R.element_properties(e)[k] for k in PROPS] for e in elements], float)
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    return {e: X[i] for i, e in enumerate(elements)}


def gauss_family(X: np.ndarray, prefix: str, factors=(0.25, 1.0, 4.0)) -> Dict[str, np.ndarray]:
    D2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    med = float(np.median(D2[D2 > 0]))
    return {f"{prefix} gauss x{f:g}": np.exp(-D2 / (f * med)) for f in factors}


def slab_references(data: AlloySet, ids: Sequence[int]) -> Dict[str, Dict[str, np.ndarray]]:
    col = {e: i for i, e in enumerate(data.elements)}
    props = standard_properties(data.elements)
    F = np.zeros((len(ids), len(col)))          # element fractions of the slab
    Ft = np.zeros_like(F)                       # of the top layer
    Fr = np.zeros_like(F)                       # of the other layers
    Pw = np.zeros((len(ids), len(PROPS)))       # mean properties, whole slab
    Pt = np.zeros_like(Pw)                      # top layer
    Pr = np.zeros_like(Pw)                      # other layers
    for k, s in enumerate(ids):
        sym = data.slabs[s].get_chemical_symbols()
        top, rest = data.layers(s)
        for i in range(len(sym)):
            F[k, col[sym[i]]] += 1.0 / len(sym)
            Pw[k] += props[sym[i]] / len(sym)
        for i in top:
            Ft[k, col[sym[i]]] += 1.0 / len(top)
            Pt[k] += props[sym[i]] / len(top)
        for i in rest:
            Fr[k, col[sym[i]]] += 1.0 / len(rest)
            Pr[k] += props[sym[i]] / len(rest)
    fam: Dict[str, Dict[str, np.ndarray]] = {}
    fam["element fractions"] = {"linear": F @ F.T, **gauss_family(F, "fractions")}
    L = np.hstack([Ft, Fr])
    fam["element fractions by layer"] = {"linear": L @ L.T, **gauss_family(L, "by layer")}
    fam["element properties, whole slab"] = gauss_family(Pw, "whole slab")
    fam["element properties, top layer and the rest"] = gauss_family(np.hstack([Pt, Pr]),
                                                                     "top + rest")
    return fam


def site_reference_rows(data: AlloySet, rows: Sequence[dict]):
    """Features of a site from the elements that bind the adsorbate: element counts, mean
    properties, the kind of site, and the composition of the slab."""
    col = {e: i for i, e in enumerate(data.elements)}
    props = standard_properties(data.elements)
    B = np.zeros((len(rows), len(col)))
    P = np.zeros((len(rows), len(PROPS)))
    S = np.zeros((len(rows), len(SITE_KINDS)))
    C = np.zeros((len(rows), len(col)))
    for k, r in enumerate(rows):
        for e in r["bind"]:
            B[k, col[e]] += 1.0 / len(r["bind"])
            P[k] += props[e] / len(r["bind"])
        S[k, SITE_KINDS.index(r["site"])] = 1.0
        sym = data.slabs[r["slab"]].get_chemical_symbols()
        for e in sym:
            C[k, col[e]] += 1.0 / len(sym)
    return B, P, S, C


def site_references(data: AlloySet, rows: Sequence[dict]) -> Dict[str, Dict[str, np.ndarray]]:
    B, P, S, C = site_reference_rows(data, rows)
    fam: Dict[str, Dict[str, np.ndarray]] = {}
    X = np.hstack([B, S])
    fam["binding elements and site kind"] = {"linear": X @ X.T, **gauss_family(X, "binding")}
    Xc = np.hstack([B, S, C])
    fam["binding elements, site kind and slab elements"] = {"linear": Xc @ Xc.T,
                                                           **gauss_family(Xc, "binding+slab")}
    Xp = np.hstack([P, S])
    fam["properties of the binding elements and site kind"] = gauss_family(Xp, "properties")
    Xq = np.hstack([P, S, C])
    fam["properties of the binding elements, site kind and slab elements"] = gauss_family(
        Xq, "properties+slab")
    return fam


# --------------------------------------------------------------------------- #
# Field kernels
# --------------------------------------------------------------------------- #
def fieldset(kind: str, elements: Sequence[str], alpha: float = 1.0):
    if kind == "classic":
        return inorganic_fieldset((), physchem=R.PHYSCHEM, charge_model="electronegativity",
                                  other=False, alpha=alpha)
    if kind == "extended":
        return inorganic_fieldset((), physchem=R.EXTENDED, charge_model="electronegativity",
                                  other=False, alpha=alpha)
    if kind == "indicator":
        return inorganic_fieldset(sorted(elements), physchem=R.PHYSCHEM,
                                  charge_model="electronegativity", other=False, alpha=alpha)
    raise ValueError(kind)


def slab_field_kernels(data: AlloySet, ids: Sequence[int]) -> Dict[str, Dict[str, np.ndarray]]:
    """The periodic autocorrelation of every slab, for the three field sets."""
    fam = {}
    for kind in ("classic", "extended", "indicator"):
        fs = fieldset(kind, data.elements)
        crystals = [crystal_fields(R.with_property_arrays(data.slabs[s]), fs, name=str(s))
                    for s in ids]
        F = periodic_feature_matrix(crystals, PERIODIC_SPEC, n_fields=fs.n_fields)
        fam[f"periodic autocorrelation, {kind} fields"] = {"cosine": autocorrelation_kernel(F)}
    return fam


def site_environments(data: AlloySet, rows: Sequence[dict], kind: str, rho: float, cache: dict):
    """The point environment of every site: origin at the point over the site, axes of the lattice."""
    fs = fieldset(kind, data.elements)
    envs = []
    for k, r in enumerate(rows):
        key = (r["slab"], kind)
        if key not in cache:
            cache[key] = crystal_fields(R.with_property_arrays(data.slabs[r["slab"]]), fs,
                                        name=str(r["slab"]))
        top, _ = data.layers(r["slab"])
        z_top = float(data.slabs[r["slab"]].positions[top, 2].mean())
        centre = np.array([r["x"], r["y"], z_top + HEIGHT[r["ads"]]])
        envs.append(point_environment(cache[key], centre, rho=rho, r_max=R_MAX,
                                      name=f"{r['slab']}_{k}"))
    return envs


def site_field_kernels(data: AlloySet, rows: Sequence[dict], kinds=("classic", "extended"),
                       overlap: bool = True, log=print) -> Dict[str, Dict[str, np.ndarray]]:
    cache: dict = {}
    fam: Dict[str, Dict[str, np.ndarray]] = {}
    for kind in kinds:
        inv, ov = {}, {}
        for rho in RHOS:
            t0 = time.time()
            envs = site_environments(data, rows, kind, rho, cache)
            inv[f"rho={rho:g}"] = local_kernel(local_features(envs))
            if overlap:
                table = AtomTable.from_molecules(envs)
                for alpha in ALPHAS:
                    fs_a = fieldset(kind, data.elements, alpha=alpha)
                    for normalize in (False, True):
                        key = f"rho={rho:g} alpha={alpha:g} {'cosine' if normalize else 'raw'}"
                        ov[key] = R.overlap_kernel(table, fs_a, normalize)
            log(f"    {kind} rho {rho:g}: {np.mean([e.n_atoms for e in envs]):.0f} atoms per "
                f"environment, {time.time() - t0:.0f} s")
        fam[f"site environment, invariant, {kind} fields"] = inv
        if overlap:
            fam[f"site environment, overlap in the lattice frame, {kind} fields"] = ov
    return fam


# --------------------------------------------------------------------------- #
# Folds and cross-validation
# --------------------------------------------------------------------------- #
def folds_for(design: str, n: int, groups: Dict[str, Sequence[str]], seed: int, k: int):
    if design == "random":
        return R.random_folds(n, k, seed)
    return R.group_folds(list(groups[design]), k=k, seed=seed)


def cross_validate(name: str, family: Dict[str, np.ndarray], y: np.ndarray, design: str,
                   groups: Dict[str, Sequence[str]], seed: int = SEED) -> dict:
    """Out-of-fold predictions; the kernel and the ridge are chosen in inner folds."""
    t0 = time.time()
    n = len(y)
    pred = np.full(n, np.nan)
    chosen = collections.Counter()
    for te in folds_for(design, n, groups, seed, FOLDS):
        tr = np.setdiff1d(np.arange(n), te)
        sub = {d: [g[i] for i in tr] for d, g in groups.items()}
        inner = folds_for(design, len(tr), sub, seed, INNER)
        best = None
        for key, K in family.items():
            scores = R._cv_q2(K[np.ix_(tr, tr)], y[tr], inner, LAMS)
            j = int(np.argmax(scores))
            if best is None or scores[j] > best[0]:
                best = (scores[j], key, LAMS[j])
        _, key, lam = best
        K = family[key]
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], float(lam))
        pred[te] = K[np.ix_(te, tr)] @ a + b
        chosen[f"{key} lam={lam:g}"] += 1
    ok = np.isfinite(pred)
    return {"name": name, "design": design, "q2": r2_score(y[ok], pred[ok]),
            "rmse": rmse(y[ok], pred[ok]), "mae": float(np.mean(np.abs(y[ok] - pred[ok]))),
            "chosen": dict(chosen), "seconds": time.time() - t0,
            "predicted": [None if not np.isfinite(v) else float(v) for v in pred]}


def mean_baseline(y: np.ndarray, design: str, groups, seed: int = SEED) -> dict:
    """The training mean of every fold as the prediction: what any model must beat."""
    pred = np.full(len(y), np.nan)
    for te in folds_for(design, len(y), groups, seed, FOLDS):
        tr = np.setdiff1d(np.arange(len(y)), te)
        pred[te] = float(np.mean(y[tr]))
    return {"name": "training mean", "design": design, "q2": r2_score(y, pred),
            "rmse": rmse(y, pred), "mae": float(np.mean(np.abs(y - pred))), "chosen": {},
            "seconds": 0.0, "predicted": [float(v) for v in pred]}


def bootstrap_mae(y, a, b, runs: int = RUNS, seed: int = SEED, groups=None) -> Tuple[float, float, float]:
    """Mean absolute error of `a` minus that of `b`, with a paired bootstrap interval.

    With `groups` the resampling is over groups (the slabs of a site task), not over rows."""
    ea, eb = np.abs(y - a), np.abs(y - b)
    d = float(np.mean(ea) - np.mean(eb))
    rng = np.random.RandomState(seed)
    if groups is None:
        n = len(y)
        draws = [np.mean(ea[i] - eb[i]) for i in (rng.randint(0, n, n) for _ in range(runs))]
    else:
        labels = sorted(set(groups))
        g = np.asarray(groups)
        members = {lab: np.where(g == lab)[0] for lab in labels}
        draws = []
        for _ in range(runs):
            pick = rng.randint(0, len(labels), len(labels))
            idx = np.concatenate([members[labels[p]] for p in pick])
            draws.append(np.mean(ea[idx] - eb[idx]))
    return d, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def run_task(name: str, y: np.ndarray, families: Dict[str, Dict[str, np.ndarray]],
             refs: Sequence[str], designs: Sequence[str], groups, pair_groups,
             log=print) -> dict:
    out = {"task": name, "n": int(len(y)), "mean": float(np.mean(y)), "sd": float(np.std(y)),
           "models": [], "pairs": []}
    for design in designs:
        res = {"training mean": mean_baseline(y, design, groups)}
        for fam_name, fam in families.items():
            res[fam_name] = cross_validate(fam_name, fam, y, design, groups)
            log(f"  {name:28s} {design:9s} {fam_name:62s} q2 {res[fam_name]['q2']:+.3f} "
                f"mae {res[fam_name]['mae']:.3f} eV ({res[fam_name]['seconds']:.0f} s)")
        for r in res.values():
            out["models"].append(dict(r, task=name))
        ours = [m for m in families if m not in refs]
        for a in ours:
            for b in list(refs) + ["training mean"]:
                pa = np.array([np.nan if v is None else v for v in res[a]["predicted"]])
                pb = np.array([np.nan if v is None else v for v in res[b]["predicted"]])
                d, lo, hi = bootstrap_mae(y, pa, pb, groups=pair_groups)
                out["pairs"].append({"task": name, "design": design, "method": a, "against": b,
                                     "delta_mae": d, "low": lo, "high": hi})
    return out


def run(args, log=print) -> None:
    warnings.filterwarnings("ignore")
    os.makedirs(RESULTS, exist_ok=True)
    data = AlloySet()
    log(f"{len(data.slabs)} slabs, {len(data.elements)} elements, {len(data.rows)} kept geometries")
    tasks = []
    if args.task in ("all", "slab"):
        for ads in ("H", "OH"):
            for variant in args.variants:
                ids, y = slab_task(data, ads, variant)
                alloys = [data.alloy(s) for s in ids]
                groups = {"alloy-out": alloys}
                log(f"slab {ads} [{variant}]: {len(ids)} slabs, mean {y.mean():+.2f} eV, "
                    f"sd {y.std():.2f} eV, {len(set(alloys))} alloys")
                fams = {**slab_references(data, ids), **slab_field_kernels(data, ids)}
                refs = [f for f in fams if not f.startswith("periodic")]
                res = run_task(f"slab {ads} [{variant}]", y, fams, refs,
                               ("random", "alloy-out"), groups, None, log)
                res.update(kind="slab", adsorbate=ads, variant=variant, ids=ids,
                           y=[float(v) for v in y], n_alloys=len(set(alloys)))
                tasks.append(res)
    if args.task in ("all", "site"):
        rng = np.random.RandomState(SEED)
        for ads in ("H", "OH"):
            for variant in args.variants:
                rows = select(data, ads, variant)
                slabs = sorted({r["slab"] for r in rows})
                if len(rows) > args.max_sites:
                    rng.shuffle(slabs)
                    keep, count = set(), 0
                    for s in slabs:
                        keep.add(s)
                        count += sum(1 for r in rows if r["slab"] == s)
                        if count >= args.max_sites:
                            break
                    rows = [r for r in rows if r["slab"] in keep]
                y = np.array([r["e"] for r in rows])
                groups = {"slab-out": [str(r["slab"]) for r in rows],
                          "alloy-out": [data.alloy(r["slab"]) for r in rows]}
                log(f"site {ads} [{variant}]: {len(rows)} sites on {len({r['slab'] for r in rows})} "
                    f"slabs, mean {y.mean():+.2f} eV, sd {y.std():.2f} eV")
                fams = {**site_references(data, rows),
                        **site_field_kernels(data, rows, overlap=not args.no_overlap, log=log)}
                refs = [f for f in fams if not f.startswith("site environment")]
                res = run_task(f"site {ads} [{variant}]", y, fams, refs,
                               ("random", "slab-out", "alloy-out"), groups,
                               groups["slab-out"], log)
                res.update(kind="site", adsorbate=ads, variant=variant,
                           y=[float(v) for v in y], rows=[r["log"] for r in rows])
                tasks.append(res)
    path = os.path.join(RESULTS, "data.json" if args.task == "all" else f"data_{args.task}.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"tasks": tasks, "seed": SEED, "folds": FOLDS, "inner": INNER}, fh, default=float)
        fh.write("\n")
    log(f"wrote {path}")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _load_results() -> List[dict]:
    tasks: List[dict] = []
    for name in ("data_slab.json", "data_site.json"):
        path = os.path.join(RESULTS, name)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                tasks += json.load(fh)["tasks"]
    return tasks


def write_table(path: Optional[str] = None) -> str:
    """The table of ``benchmarks/results/alloy/table.md`` from the stored results."""
    tasks = _load_results()
    if not tasks:
        raise SystemExit("no stored results: run the benchmark first")
    lines = ["# Alloy surfaces: adsorption of H and OH, fields against element names and properties",
             "",
             "Data: the slabs of Mamun et al. (Scientific Data 2019), BEEF-vdW, built by "
             "`examples/make_alloy_examples.py`. Energies are in eV and relative to the bare slab "
             "and to H2 (and H2O).",
             "",
             "How to read the tables:",
             "",
             "- **q2** is the out-of-fold coefficient of determination. The training mean has "
             "q2 near 0.",
             "- **MAE** is the mean absolute error in eV.",
             "- **MAE against the best reference** is the MAE of the method minus the MAE of the "
             "best reference method of the same task and folds, with a paired bootstrap interval "
             "(2.5 % to 97.5 %). A negative number means that the method has the smaller error. "
             "An interval that contains zero means that the two methods do not differ clearly.",
             "- A *reference* is a model that uses only element names or tabulated element "
             "properties. A *field method* is a model on the structure.",
             "- **slab** tasks: the sample is the bare slab and the target is the lowest "
             "adsorption energy over its sites. **site** tasks: the sample is one site of one slab "
             "and the target is the energy at that site.",
             "- Folds: **random** (10 folds), **slab-out** (all sites of a slab in one fold) and "
             "**alloy-out** (all slabs with the same elements in one fold).",
             ""]
    for t in tasks:
        n_field = lambda m: m.startswith(("periodic", "site environment"))   # noqa: E731
        lines += [f"## {t['task']}: {t['n']} samples", "",
                  f"Mean {t['mean']:+.2f} eV, standard deviation {t['sd']:.2f} eV.", ""]
        designs = []
        for m in t["models"]:
            if m["design"] not in designs:
                designs.append(m["design"])
        for design in designs:
            ms = [m for m in t["models"] if m["design"] == design]
            refs = [m for m in ms if not n_field(m["name"]) and m["name"] != "training mean"]
            best = min(refs, key=lambda m: m["mae"])
            lines += [f"### Folds: {design}", "",
                      f"The best reference is *{best['name']}* (MAE {best['mae']:.3f} eV).", "",
                      "| method | kind | q2 | MAE (eV) | MAE against the best reference (eV) |",
                      "| --- | --- | --- | --- | --- |"]
            for m in ms:
                kind = ("field method" if n_field(m["name"]) else
                        "baseline" if m["name"] == "training mean" else "reference")
                diff = ""
                if kind == "field method":
                    for p in t["pairs"]:
                        if (p["design"] == design and p["method"] == m["name"]
                                and p["against"] == best["name"]):
                            diff = f"{p['delta_mae']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}]"
                lines.append(f"| {m['name']} | {kind} | {m['q2']:+.3f} | {m['mae']:.3f} | {diff} |")
            lines.append("")
    path = path or os.path.join(RESULTS, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=("all", "slab", "site"), default="all")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--max-sites", type=int, default=MAX_OVERLAP_SITES)
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print("wrote", write_table())
        return
    run(args)
    print("wrote", write_table())


if __name__ == "__main__":
    main()
