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
* for a site, the *neighbour counts*: how many atoms of each element lie in four shells (up to
  2.3, 3.3, 4.5 and 6.0 A) around the same point that the field methods use, and the sums of the
  element properties in those shells. A field of the environment is a smooth version of such
  counts, so this is the fair test of whether the smooth field adds anything. They were added
  after the first full run (``--task counts``), on the same sites and folds;

and the field methods are the periodic autocorrelation of the slab (rotation and translation
invariant), the rotation-invariant autocorrelation of the point environment of a site, and the
overlap kernel of that environment in the frame of the lattice.

The shells around a site (task ``shells``, added 19 November 2026)
-----------------------------------------------------------------
The site references above know the binding atoms, the kind of site and the composition of the
slab, but not what lies around the site: the second neighbours, or the atom under a hollow. A
field of the site environment sees those, so a reference that sees them too is needed before a
lead of the field is read as a property of the field. The task adds two such references, with no
field and no frame:

* *shell properties and site kind*: for each of four distance shells around the point of the
  site (up to 2.5, 3.5, 4.5 and 6 A, the periodic images of the slab included), the number of
  atoms and the sum of their standardised element properties; with the kind of site;
* *shell elements and site kind*: for each shell, the number of atoms of each element.

Each is a linear kernel and Gaussian kernels of five widths, on the stored rows, folds and inner
choice of the site task. The task reads the stored predictions of the site task and compares every
field method with the better of the two, by the bootstrap over slabs.

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
    python benchmarks/run_alloy.py --task shells    # after the site task: the shell references
    python benchmarks/run_alloy.py --report        # rebuild the table from the stored results
    python benchmarks/run_alloy.py --task counts   # add the neighbour-count references to the
                                                   # stored site results (needs a site run first)
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


def site_rows(data: AlloySet, ads: str, variant: str, max_sites: int,
              rng: Optional[np.random.RandomState] = None) -> List[dict]:
    """The rows of a site task: whole slabs drawn at random until `max_sites` rows are reached.

    `run` draws all its site tasks from one generator, in the order H all, H converged, OH all,
    OH converged; a task alone gets a new generator, which is the same for the first task."""
    rows = select(data, ads, variant)
    slabs = sorted({r["slab"] for r in rows})
    if len(rows) > max_sites:
        (rng if rng is not None else np.random.RandomState(SEED)).shuffle(slabs)
        keep, count = set(), 0
        for s in slabs:
            keep.add(s)
            count += sum(1 for r in rows if r["slab"] == s)
            if count >= max_sites:
                break
        rows = [r for r in rows if r["slab"] in keep]
    return rows


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


#: outer edges of the four shells around the point of a site, in angstrom
SHELLS = (2.3, 3.3, 4.5, 6.0)


def neighbour_counts(data: AlloySet, rows: Sequence[dict]) -> Tuple[np.ndarray, np.ndarray]:
    """What a field sees around a site, written as counts and without any smoothing.

    The point is the same as for the field methods (the position of the adsorbate in the plane, at
    a fixed height over the top layer). For each of the four shells around it the function counts
    the atoms of every element, and sums the standardised properties of the elements. The cell is
    repeated in the plane, so that the shells are not cut by the edge of the 2 x 2 cell.

    Returns (counts, property sums); the columns run over shells first, then elements (or
    properties)."""
    col = {e: i for i, e in enumerate(data.elements)}
    props = standard_properties(data.elements)
    n_shell = len(SHELLS)
    N = np.zeros((len(rows), n_shell, len(col)))
    Q = np.zeros((len(rows), n_shell, len(PROPS)))
    images: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    shifts = [(i, j) for i in range(-2, 3) for j in range(-2, 3)]
    for k, r in enumerate(rows):
        s = r["slab"]
        if s not in images:
            atoms = data.slabs[s]
            cell = np.asarray(atoms.cell)
            pos = np.concatenate([atoms.positions + i * cell[0] + j * cell[1] for i, j in shifts])
            idx = np.tile(np.arange(len(atoms)), len(shifts))
            top, _ = data.layers(s)
            images[s] = (pos, idx, np.array([float(atoms.positions[top, 2].mean())]))
        pos, idx, z_top = images[s]
        sym = data.slabs[s].get_chemical_symbols()
        centre = np.array([r["x"], r["y"], z_top[0] + HEIGHT[r["ads"]]])
        d = np.linalg.norm(pos - centre, axis=1)
        shell = np.searchsorted(SHELLS, d, side="left")
        for a in np.where(shell < n_shell)[0]:
            e = sym[idx[a]]
            N[k, shell[a], col[e]] += 1.0
            Q[k, shell[a]] += props[e]
    return N.reshape(len(rows), -1), Q.reshape(len(rows), -1)


#: the widths of gauss_family; the count references use gauss_light, which gives the same kernels
#: without the (n, n, d) array (about 7.4 GB for 2,500 sites and 148 count columns)
COUNT_FACTORS = (0.25, 1.0, 4.0)


def count_references(data: AlloySet, rows: Sequence[dict]) -> Dict[str, Dict[str, np.ndarray]]:
    """References that count the neighbours of a site in four shells (the fair reference for a
    field, which is also a smooth count of the neighbours)."""
    N, Q = neighbour_counts(data, rows)
    B, P, S, C = site_reference_rows(data, rows)
    fam: Dict[str, Dict[str, np.ndarray]] = {}
    fam["neighbour counts of each element in four shells"] = {
        "linear": N @ N.T, **gauss_light(N, "counts", factors=COUNT_FACTORS)}
    Xs = np.hstack([N, S])
    fam["neighbour counts of each element in four shells, and site kind"] = {
        "linear": Xs @ Xs.T, **gauss_light(Xs, "counts+site", factors=COUNT_FACTORS)}
    Xq = np.hstack([Q, S])
    fam["neighbour sums of element properties in four shells, and site kind"] = gauss_light(
        Xq, "property sums+site", factors=COUNT_FACTORS)
    Xa = np.hstack([N, B, S, C])
    fam["neighbour counts, binding elements, site kind and slab elements"] = {
        "linear": Xa @ Xa.T, **gauss_light(Xa, "counts+binding+slab", factors=COUNT_FACTORS)}
    return fam


def add_counts(log=print) -> None:
    """Add the neighbour-count references to the stored site results.

    The rows, the folds and the seeds are those of the stored run, so the stored predictions of
    every other method stay valid; only the new references are cross-validated, and every
    difference against the best reference is computed again."""
    warnings.filterwarnings("ignore")
    path = os.path.join(RESULTS, "data_site.json")
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)
    data = AlloySet()
    by_log = collections.defaultdict(list)
    for r in data.rows:
        by_log[(r["log"], r["ads"])].append(r)
    for t in stored["tasks"]:
        rows = []
        for name in t["rows"]:
            hit = by_log[(name, t["adsorbate"])]
            if len(hit) != 1:
                raise SystemExit(f"the log {name} is not one row of the data: {len(hit)}")
            rows.append(hit[0])
        y = np.array(t["y"])
        if not np.allclose(y, [r["e"] for r in rows]):
            raise SystemExit("the stored targets differ from the data: rebuild the results")
        groups = {"slab-out": [str(r["slab"]) for r in rows],
                  "alloy-out": [data.alloy(r["slab"]) for r in rows]}
        fams = count_references(data, rows)
        old = {m["name"] for m in t["models"]}
        new = [m for m in fams if m not in old]
        log(f"{t['task']}: {len(new)} new references on {len(rows)} sites")
        for design in ("random", "slab-out", "alloy-out"):
            for name in new:
                res = cross_validate(name, fams[name], y, design, groups)
                t["models"].append(dict(res, task=t["task"]))
                log(f"  {t['task']:16s} {design:9s} {name:66s} q2 {res['q2']:+.3f} "
                    f"mae {res['mae']:.3f} eV ({res['seconds']:.0f} s)")
        is_field = lambda m: m.startswith(("periodic", "site environment"))   # noqa: E731
        t["models"].sort(key=lambda m: (["random", "slab-out", "alloy-out"].index(m["design"]),
                                        m["name"] != "training mean", is_field(m["name"])))
        t["pairs"] = []
        for design in ("random", "slab-out", "alloy-out"):
            res = {m["name"]: m for m in t["models"] if m["design"] == design}
            refs = [m for m in res if not is_field(m) and m != "training mean"]
            for a in [m for m in res if is_field(m)]:
                for b in refs + ["training mean"]:
                    pa = np.array([np.nan if v is None else v for v in res[a]["predicted"]])
                    pb = np.array([np.nan if v is None else v for v in res[b]["predicted"]])
                    d, lo, hi = bootstrap_mae(y, pa, pb, groups=groups["slab-out"])
                    t["pairs"].append({"task": t["task"], "design": design, "method": a,
                                       "against": b, "delta_mae": d, "low": lo, "high": hi})
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(stored, fh, default=float)
        fh.write("\n")
    log(f"wrote {path}")


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
                rows = site_rows(data, ads, variant, args.max_sites, rng)
                y = np.array([r["e"] for r in rows])
                groups = {"slab-out": [str(r["slab"]) for r in rows],
                          "alloy-out": [data.alloy(r["slab"]) for r in rows]}
                log(f"site {ads} [{variant}]: {len(rows)} sites on {len({r['slab'] for r in rows})} "
                    f"slabs, mean {y.mean():+.2f} eV, sd {y.std():.2f} eV")
                fams = {**site_references(data, rows), **count_references(data, rows),
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
# The shells around a site (task shells)
# --------------------------------------------------------------------------- #
SHELL_EDGES = (2.5, 3.5, 4.5, 6.0)
#: the widths of the Gaussian kernels, as factors of the median squared distance; wider than
#: gauss_family's, because a first measurement chose the widest of 0.25, 1 and 4 in most folds
SHELL_FACTORS = (0.25, 1.0, 4.0, 16.0, 64.0)
SHELL_REFERENCES = ("shell properties and site kind", "shell elements and site kind")


def shell_columns(data: AlloySet, rows: Sequence[dict], ads: str):
    """For every site: per shell around its point, the number of atoms and the sum of the
    standardised element properties (`P`), and the number of atoms of each element (`E`); and the
    kind of site (`S`). The periodic images of the slab in its two in-plane directions count: two
    on each side, because a 2 x 2 cell is about 5.5 A long and the outer shell reaches 6 A."""
    props = standard_properties(data.elements)
    col = {e: i for i, e in enumerate(data.elements)}
    n_p, n_e, n_s = len(PROPS), len(col), len(SHELL_EDGES)
    P = np.zeros((len(rows), n_s * (1 + n_p)))
    E = np.zeros((len(rows), n_s * n_e))
    S = np.zeros((len(rows), len(SITE_KINDS)))
    for k, r in enumerate(rows):
        slab = data.slabs[r["slab"]]
        top, _ = data.layers(r["slab"])
        centre = np.array([r["x"], r["y"], float(slab.positions[top, 2].mean()) + HEIGHT[ads]])
        cell = slab.cell.array
        sym = slab.get_chemical_symbols()
        for i in range(-2, 3):
            for j in range(-2, 3):
                d = np.linalg.norm(slab.positions + i * cell[0] + j * cell[1] - centre, axis=1)
                low = 0.0
                for s, edge in enumerate(SHELL_EDGES):
                    for a in np.flatnonzero((d > low) & (d <= edge)):
                        P[k, s * (1 + n_p)] += 1.0
                        P[k, s * (1 + n_p) + 1:(s + 1) * (1 + n_p)] += props[sym[a]]
                        E[k, s * n_e + col[sym[a]]] += 1.0
                    low = edge
        S[k, SITE_KINDS.index(r["site"])] = 1.0
    return P, E, S


def gauss_light(X: np.ndarray, prefix: str, factors=SHELL_FACTORS) -> Dict[str, np.ndarray]:
    """The kernels of gauss_family, computed from |x|^2 + |y|^2 - 2 x.y without the (n, n, d)
    array that gauss_family builds (about 7.6 GB for 2,500 rows of 150 columns)."""
    sq = np.sum(X * X, axis=1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (X @ X.T), 0.0)
    med = float(np.median(D2[D2 > 1e-12]))
    return {f"{prefix} gauss x{f:g}": np.exp(-D2 / (f * med)) for f in factors}


def _standardised(X: np.ndarray) -> np.ndarray:
    keep = X.std(axis=0) > 0
    X = X[:, keep]
    return (X - X.mean(axis=0)) / X.std(axis=0)


def shell_references(data: AlloySet, rows: Sequence[dict], ads: str) -> Dict[str, Dict[str, np.ndarray]]:
    P, E, S = shell_columns(data, rows, ads)
    fam = {}
    for name, M, prefix in ((SHELL_REFERENCES[0], P, "shell properties"),
                            (SHELL_REFERENCES[1], E, "shell elements")):
        X = np.hstack([_standardised(M), S])
        fam[name] = {"linear": X @ X.T / X.shape[1], **gauss_light(X, prefix)}
    return fam


def run_shells(args, log=print) -> None:
    """The shell references on the rows and folds of every stored site task."""
    warnings.filterwarnings("ignore")
    path = os.path.join(RESULTS, "data_site.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: run the site task first")
    with open(path, encoding="utf-8") as fh:
        stored = json.load(fh)["tasks"]
    data = AlloySet()
    out = []
    # the generator of the stored run: one for all site tasks, in the order of the stored tasks
    rng = np.random.RandomState(SEED)
    for task in stored:
        ads, variant = task["adsorbate"], task["variant"]
        rows = site_rows(data, ads, variant, args.max_sites, rng)
        if [r["log"] for r in rows] != task["rows"]:
            raise SystemExit(f"{task['task']}: the rows differ from the stored run")
        y = np.array([r["e"] for r in rows])
        groups = {"slab-out": [str(r["slab"]) for r in rows],
                  "alloy-out": [data.alloy(r["slab"]) for r in rows]}
        fams = shell_references(data, rows, ads)
        res = {"task": task["task"], "adsorbate": ads, "variant": variant, "n": len(rows),
               "models": [], "pairs": []}
        designs = []
        for m in task["models"]:
            if m["design"] not in designs:
                designs.append(m["design"])
        for design in designs:
            sm = {m["name"]: m for m in task["models"] if m["design"] == design}
            old_refs = [n for n in sm if not n.startswith("site environment") and n != "training mean"]
            best_old = min(old_refs, key=lambda n: sm[n]["mae"])
            new = {name: cross_validate(name, fam, y, design, groups) for name, fam in fams.items()}
            best_new = min(new, key=lambda n: new[n]["mae"])
            pn = np.array([np.nan if v is None else v for v in new[best_new]["predicted"]])
            po = np.array(sm[best_old]["predicted"], float)
            for name, r in new.items():
                res["models"].append({k: v for k, v in r.items() if k != "predicted"})
                log(f"  {task['task']:14s} {design:9s} {name:32s} q2 {r['q2']:+.3f} mae {r['mae']:.3f}")
            d = bootstrap_mae(y, pn, po, groups=groups["slab-out"])
            res["pairs"].append({"design": design, "method": best_new, "against": best_old,
                                 "delta_mae": d[0], "low": d[1], "high": d[2]})
            for name in sm:
                if name.startswith("site environment"):
                    pf = np.array([np.nan if v is None else v for v in sm[name]["predicted"]])
                    d = bootstrap_mae(y, pf, pn, groups=groups["slab-out"])
                    res["pairs"].append({"design": design, "method": name, "against": best_new,
                                         "delta_mae": d[0], "low": d[1], "high": d[2]})
        out.append(res)
    path = os.path.join(RESULTS, "shells.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"tasks": out, "edges": list(SHELL_EDGES), "factors": list(SHELL_FACTORS)}, fh,
                  indent=1, default=float)
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
             "- A *reference* is a model that uses only element names, tabulated element "
             "properties or (for sites) counts of the atoms of each element in four shells around "
             "the site. A *field method* is a model on a smooth field of the structure.",
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
    shells = os.path.join(RESULTS, "shells.json")
    if os.path.exists(shells):
        with open(shells, encoding="utf-8") as fh:
            sh = json.load(fh)
        lines += ["## The shells around a site: references added on 19 November 2026", "",
                  "Two more references for the site tasks, with no field and no frame: for each "
                  "distance shell around the point of the site (up to "
                  + ", ".join(f"{e:g}" for e in sh["edges"]) + " A), the number of atoms and the "
                  "sum of their standardised element properties (*shell properties*), or the "
                  "number of atoms of each element (*shell elements*), each with the kind of "
                  "site. Same rows, folds and inner choice as the site tasks above. The "
                  "differences are in MAE (eV); a negative number means that the first method "
                  "has the smaller error.", ""]
        for t in sh["tasks"]:
            lines += [f"### {t['task']}", "",
                      "| folds | reference | q2 | MAE (eV) | chosen kernels |",
                      "| --- | --- | --- | --- | --- |"]
            for m in t["models"]:
                chosen = ", ".join(f"{k} ({v})" for k, v in m["chosen"].items())
                lines.append(f"| {m['design']} | {m['name']} | {m['q2']:+.3f} | {m['mae']:.3f} | "
                             f"{chosen} |")
            lines += ["", "| folds | method | against | MAE difference (eV) |",
                      "| --- | --- | --- | --- |"]
            for p in t["pairs"]:
                lines.append(f"| {p['design']} | {p['method']} | {p['against']} | "
                             f"{p['delta_mae']:+.3f} [{p['low']:+.3f}, {p['high']:+.3f}] |")
            lines.append("")
    path = path or os.path.join(RESULTS, "table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--task", choices=("all", "slab", "site", "counts", "shells"), default="all",
                    help="counts: add the neighbour-count references to the stored site results; "
                         "shells: the shell references on the stored site tasks")
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--max-sites", type=int, default=MAX_OVERLAP_SITES)
    ap.add_argument("--no-overlap", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        print("wrote", write_table())
        return
    if args.task == "counts":
        add_counts()
        print("wrote", write_table())
        return
    if args.task == "shells":
        run_shells(args)
        print("wrote", write_table())
        return
    run(args)
    print("wrote", write_table())


if __name__ == "__main__":
    main()
