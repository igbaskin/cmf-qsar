#!/usr/bin/env python
"""
The surface kernel on the surface of a particle (plan item NT5).

`cmfqsar.invariant.ParticleSurfaceSpec` measures the channels of the surface kernel on the
surface of a particle's core: the height of an atom is its distance from the core's surface, and
the in-plane distance between two atoms is the distance along that surface between their anchor
points (their nearest points on it).  The surface is a plane, a sphere or a cylinder fitted to
the outermost core atoms.

Stage checks
------------
* A flat core: a molecule on Au(111) and Au(100) slabs whose top layer is at z = 0, against the
  flat surface kernel.
* Invariance: a molecule on a sphere and on a cylinder, the whole system turned and moved.
* Curvature: one flat patch of ligands, wrapped onto spheres of growing radius (each point at
  the same height above the point of the sphere that the patch's in-plane coordinates map to),
  against the flat kernel of the patch.
* Cost: the time of one shell, against its number of atoms.

Stage model
-----------
Ligand shells generated on gold icosahedra of 55, 147 and 309 atoms (radius 0.5 to 0.9 nm):
chains of 2, 4 or 6 carbon atoms on a sulfur atom 2.4 A above a surface gold atom, ending in O, N
or C, at 20, 35, 50 or 65 % of the surface atoms, each chain along the radial direction tilted by
up to 25 degrees in a random direction (a chain that would come within 2.5 A of another is tried
again, up to 20 times, then left out); two random shells of every recipe, 216 in all.  Target:
the accessibility of the end groups -- the mean fraction of each end atom's sphere (covalent
radius plus a 1.4 A probe) that no other atom of the core or the shell covers (the Shrake-Rupley
method of item NT3).  It is a property of how the shell is arranged around the curved core.
Compared on 5-fold cross-validation repeated with 5 seeds, the ridge parameter chosen by an inner
5-fold cross-validation:

* the particle surface kernel (sphere), heights 2 to 12 A every 2 A, distances along the
  surface to 12 A every 1 A, and the same on a finer grid (heights every 1 A, distances every
  0.5 A);
* the autocorrelation of the shell alone (the molecular invariant kernel, to 12 A), which does
  not know where the core is;
* the periodic autocorrelation of the core and the shell together, with open boundaries (cutoff
  12 A), which sees the distances from the ligand atoms to the gold atoms;
* the recipe: a linear regression, and a Gaussian kernel, on 1/R, the coverage, the chain
  length and the end group, the numbers each shell was built from.

Usage
-----
    python benchmarks/run_ligand_shells.py --stage checks     # about 1 minute
    python benchmarks/run_ligand_shells.py --stage model      # about 15 minutes
    python benchmarks/run_ligand_shells.py --report
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

from cmfqsar.invariant import (ParticleSurfaceSpec, SurfaceSpec,  # noqa: E402
                               particle_surface_autocorrelation, surface_autocorrelation,
                               with_core)
from cmfqsar.periodic import crystal_table, inorganic_fieldset  # noqa: E402

RESULTS = os.path.join(HERE, "results", "ligand_shells")
CACHE = os.path.join(HERE, "cache", "ligand_shells")
PHYSCHEM = ("steric", "prop:electronegativity")
GRID = dict(d_max=12.0, step=1.0, sigma=1.0, z_min=2.0, z_max=12.0, z_step=2.0, z_sigma=2.0)
CORES = (3, 4, 5)                       # icosahedron shells: 55, 147 and 309 gold atoms
COVERAGES = (0.20, 0.35, 0.50, 0.65)
CHAINS = (2, 4, 6)
ENDS = ("O", "N", "C")
REPEATS = 2
LAMBDAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
WIDTHS = (0.3, 1.0, 3.0)
SEEDS = (0, 1, 2, 3, 4)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _fields():
    return inorganic_fieldset([], physchem=PHYSCHEM)


def _record(atoms, name="x"):
    return crystal_table([atoms], _fields(), [name]).molecules[0]


def _rel(a, b) -> float:
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(a)), 1e-300))


# --------------------------------------------------------------------------- #
# Stage checks
# --------------------------------------------------------------------------- #
def _sphere_points(n: int, radius: float) -> np.ndarray:
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    theta = np.pi * (1.0 + 5.0 ** 0.5) * k
    return radius * np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi),
                              np.cos(phi)], 1)


def _wrap_on_sphere(patch: np.ndarray, radius: float) -> np.ndarray:
    """Each point (x, y, z) of a flat patch moved to height z above the point of a sphere at
    great-circle distance sqrt(x^2 + y^2) from the north pole, in the direction atan2(y, x)."""
    out = []
    for x, y, z in patch:
        rho, az = np.hypot(x, y), np.arctan2(y, x)
        a = rho / radius
        d = np.array([np.sin(a) * np.cos(az), np.sin(a) * np.sin(az), np.cos(a)])
        out.append((radius + z) * d)
    return np.array(out)


def stage_checks() -> None:
    warnings.filterwarnings("ignore")
    from ase.build import fcc100, fcc111, molecule
    from ase.cluster import Icosahedron
    from scipy.spatial.transform import Rotation
    out: Dict[str, object] = {}
    flat = []
    for build, label in ((fcc111, "Au(111)"), (fcc100, "Au(100)")):
        slab = build("Au", size=(6, 6, 3), vacuum=0.0)
        slab.positions[:, 2] -= slab.positions[:, 2].max()
        for mol_name in ("CH3CH2OH", "C6H6", "CH3COOH"):
            m = molecule(mol_name)
            m.positions += [6.0, 5.0, 2.5 - m.positions[:, 2].min()]
            rec = _record(m)
            a = surface_autocorrelation(rec, SurfaceSpec(**GRID))
            b = particle_surface_autocorrelation(rec, ParticleSurfaceSpec(shape="plane", **GRID),
                                                 core=slab.positions)
            flat.append({"core": label, "molecule": mol_name, "difference": _rel(a, b)})
    out["flat"] = flat
    rot = Rotation.from_euler("xyz", [0.4, 1.1, -0.7]).as_matrix()
    shift = np.array([3.0, -2.0, 1.0])
    inv = []
    sphere = Icosahedron("Au", 5).positions
    sphere -= sphere.mean(axis=0)
    from ase.build import bulk
    rod = bulk("Au", "fcc", a=4.08, cubic=True).repeat((3, 3, 8)).positions
    rod -= rod.mean(axis=0)
    rod = rod[np.hypot(rod[:, 0], rod[:, 1]) <= 5.0]          # a wire along z, 1 nm across
    for shape, core, place in (("sphere", sphere, [0.0, 0.0, 13.0]),
                               ("cylinder", rod, [9.0, 0.0, 2.0])):
        m = molecule("CH3CH2OH")
        m.positions += np.array(place) - m.positions.mean(axis=0)
        spec = ParticleSurfaceSpec(shape=shape, **GRID)
        x = particle_surface_autocorrelation(_record(m), spec, core=core)
        m2 = m.copy()
        m2.positions = m.positions @ rot.T + shift
        y = particle_surface_autocorrelation(_record(m2), spec, core=core @ rot.T + shift)
        flat_x = surface_autocorrelation(_record(m), SurfaceSpec(**GRID))
        flat_y = surface_autocorrelation(_record(m2), SurfaceSpec(**GRID))
        inv.append({"shape": shape, "difference": _rel(x, y),
                    "flat_kernel_difference": _rel(flat_x, flat_y)})
    out["invariance"] = inv
    rng = np.random.default_rng(0)
    patch = np.column_stack([rng.uniform(-6, 6, 40), rng.uniform(-6, 6, 40),
                             rng.uniform(2.5, 10.0, 40)])
    from ase import Atoms
    symbols = ["S"] * 10 + ["C"] * 20 + ["O"] * 10
    flat_rec = _record(Atoms(symbols, positions=patch))
    ref = surface_autocorrelation(flat_rec, SurfaceSpec(**GRID))
    curve = []
    for radius in (10.0, 20.0, 50.0, 100.0, 200.0, 500.0):
        wrapped = _wrap_on_sphere(patch, radius)
        core = _sphere_points(4000, radius)
        rec = _record(Atoms(symbols, positions=wrapped))
        row = particle_surface_autocorrelation(rec, ParticleSurfaceSpec(shape="sphere", **GRID),
                                               core=core)
        curve.append({"radius": radius, "difference": _rel(ref, row)})
    out["curvature"] = curve
    cost = []
    for k in (3, 4, 5):
        for coverage in (0.2, 0.65):
            shell, core, _ = coated(k, coverage, 6, "O", 0)
            rec = _record(shell)
            t0 = time.perf_counter()
            particle_surface_autocorrelation(rec, ParticleSurfaceSpec(**GRID), core=core)
            cost.append({"core_atoms": len(core), "shell_atoms": len(shell),
                         "seconds": time.perf_counter() - t0})
    out["cost"] = cost
    print(json.dumps(out, indent=1))
    jdump(out, os.path.join(RESULTS, "checks.json"))


# --------------------------------------------------------------------------- #
# The shells
# --------------------------------------------------------------------------- #
def coated(k: int, coverage: float, chain: int, end: str, seed: int):
    """(shell Atoms, core positions, accessibility of the end groups) of one generated shell."""
    from ase import Atoms
    from ase.cluster import Icosahedron
    from scipy.spatial import cKDTree

    from cmfqsar.descriptors import surface_access
    core = Icosahedron("Au", k)
    core.positions -= core.positions.mean(axis=0)
    outer = np.flatnonzero(surface_access(core) > 0)
    rng = np.random.default_rng(1000 * k + 100 * chain + 10 * ENDS.index(end)
                                + int(100 * coverage) * 7 + seed)
    anchors = rng.choice(outer, size=max(1, int(round(coverage * len(outer)))), replace=False)
    steps = [2.4] + [1.82] + [1.27] * (chain - 1) + [1.43]   # Au-S, S-C, C-C, C-end
    kinds = ["S"] + ["C"] * chain + [end]
    placed: List[np.ndarray] = []
    symbols: List[str] = []
    for a in anchors:
        base = core.positions[a]
        radial = base / np.linalg.norm(base)
        for _ in range(20):
            tilt = np.radians(rng.uniform(0.0, 25.0))
            e1 = np.cross(radial, [0.3, 0.5, 0.8])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(radial, e1)
            az = rng.uniform(0.0, 2.0 * np.pi)
            d = np.cos(tilt) * radial + np.sin(tilt) * (np.cos(az) * e1 + np.sin(az) * e2)
            pos = base + np.cumsum(steps)[:, None] * d[None, :]
            if placed:
                if cKDTree(np.vstack(placed)).query(pos)[0].min() < 2.5:
                    continue
            placed.append(pos)
            symbols += kinds
            break
    shell = Atoms(symbols, positions=np.vstack(placed))
    whole = Atoms(list(core.get_chemical_symbols()) + symbols,
                  positions=np.vstack([core.positions, shell.positions]))
    access = surface_access(whole)[len(core):]
    ends = np.array([s == end for s in symbols])
    # the end atom is the last of every chain; for a methyl end the chain carbons are not ends
    last = np.zeros(len(symbols), bool)
    last[np.cumsum([len(kinds)] * len(placed)) - 1] = True
    return shell, core.positions, float(np.mean(access[last & ends]))


def shells() -> List[Dict]:
    """The recipes of the model stage, with their accessibility, cached."""
    path = os.path.join(CACHE, "shells.json")
    if os.path.isfile(path):
        return jload(path)
    rows = []
    for k in CORES:
        for coverage in COVERAGES:
            for chain in CHAINS:
                for end in ENDS:
                    for seed in range(REPEATS):
                        shell, core, y = coated(k, coverage, chain, end, seed)
                        rows.append({"k": k, "coverage": coverage, "chain": chain, "end": end,
                                     "seed": seed, "ligands": int(len(shell) // (chain + 2)),
                                     "shell_atoms": len(shell), "core_atoms": len(core),
                                     "accessibility": y})
    jdump(rows, path)
    return rows


# --------------------------------------------------------------------------- #
# Stage model
# --------------------------------------------------------------------------- #
def _krr(K: np.ndarray, y: np.ndarray, tr, te, seed, options=None) -> np.ndarray:
    from cmfqsar.model import _Learner, cv_predict
    options = options if options is not None else [K]
    best, pick = np.inf, (options[0], LAMBDAS[0])
    for Kc in options:
        Ktr = Kc[np.ix_(tr, tr)]
        for lam in LAMBDAS:
            p = cv_predict(Ktr, y[tr], "krr", {"lam": lam}, folds=5, seed=seed)
            err = float(np.mean((p - y[tr]) ** 2))
            if err < best:
                best, pick = err, (Kc, lam)
    Kc, lam = pick
    return _Learner("krr", {"lam": lam}).fit(Kc[np.ix_(tr, tr)], y[tr]).predict(
        Kc[np.ix_(te, tr)])


def _gauss(X: np.ndarray, width: float) -> np.ndarray:
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * width ** 2))


def stage_model() -> None:
    warnings.filterwarnings("ignore")
    from cmfqsar.model import CMFModel, _cv_splits
    rows = shells()
    y = np.array([r["accessibility"] for r in rows])
    recs, wholes = [], []
    t_build = time.perf_counter()
    for r in rows:
        shell, core, _ = coated(r["k"], r["coverage"], r["chain"], r["end"], r["seed"])
        recs.append(with_core(_record(shell, f"{r['k']}-{r['coverage']}-{r['chain']}-"
                                             f"{r['end']}-{r['seed']}"), core))
        from ase import Atoms
        wholes.append(Atoms(["Au"] * len(core) + list(shell.get_chemical_symbols()),
                            positions=np.vstack([core, shell.positions])))
    t_build = time.perf_counter() - t_build
    from cmfqsar.fields import AtomTable
    fs = _fields()
    shell_table = AtomTable.from_molecules(recs)
    whole_table = crystal_table(wholes, fs, [str(i) for i in range(len(wholes))])
    kernels, seconds = {}, {}
    for name, table, kw in (
            ("particle surface kernel (sphere)", shell_table,
             dict(kernel_source="particle", ac_d_max=12.0, ac_sigma=1.0, surf_z_min=2.0,
                  surf_z_max=12.0, surf_z_step=2.0, particle_shape="sphere")),
            ("particle surface kernel (sphere), heights every 1 A, distances every 0.5 A",
             shell_table,
             dict(kernel_source="particle", ac_d_max=12.0, ac_sigma=0.5, surf_z_min=2.0,
                  surf_z_max=12.0, surf_z_step=1.0, particle_shape="sphere")),
            ("autocorrelation of the shell alone", shell_table,
             dict(kernel_source="autocorrelation", ac_d_max=12.0, ac_sigma=1.0)),
            ("autocorrelation of the core and the shell", whole_table,
             dict(kernel_source="periodic", ac_d_max=12.0, ac_sigma=1.0, pbc_taper=2.0))):
        t0 = time.perf_counter()
        kernels[name] = CMFModel(fieldset=fs, method="krr", normalize=True,
                                 **kw).training_kernel(table)
        seconds[name] = time.perf_counter() - t0
        print(name, f"{seconds[name]:.1f} s", flush=True)
    radius = {k: float(np.mean(np.linalg.norm(rec.core, axis=1).max())) for k, rec in
              zip([r["k"] for r in rows], recs)}
    recipe = np.array([[1.0 / radius[r["k"]], r["coverage"], r["chain"]]
                       + [float(r["end"] == e) for e in ENDS] for r in rows])
    refs = [_gauss(recipe, w) for w in WIDTHS]
    methods = {**{k: ("kernel", v) for k, v in kernels.items()},
               "Gaussian kernel on the recipe": ("refs", refs),
               "linear regression on the recipe": ("linear", None),
               "the mean of the training part": ("mean", None)}
    res = {}
    for name, (kind, K) in methods.items():
        runs = []
        for seed in SEEDS:
            p = np.full(len(y), np.nan)
            for te, tr in _cv_splits(y, 5, seed):
                if kind == "kernel":
                    p[te] = _krr(K, y, tr, te, seed)
                elif kind == "refs":
                    p[te] = _krr(None, y, tr, te, seed, options=K)
                elif kind == "linear":
                    X = np.column_stack([np.ones(len(y)), recipe])
                    coef = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
                    p[te] = X[te] @ coef
                else:
                    p[te] = y[tr].mean()
            runs.append({"rmse": float(np.sqrt(np.mean((p - y) ** 2))),
                         "q2": 1.0 - float(np.sum((p - y) ** 2) / np.sum((y - y.mean()) ** 2))})
        res[name] = {"rmse": float(np.mean([r["rmse"] for r in runs])),
                     "rmse_sd": float(np.std([r["rmse"] for r in runs])),
                     "q2": float(np.mean([r["q2"] for r in runs])),
                     "q2_sd": float(np.std([r["q2"] for r in runs]))}
        print(name, res[name], flush=True)
    out = {"n": len(rows), "target_range": [float(y.min()), float(y.max())],
           "shell_atoms": [int(min(r["shell_atoms"] for r in rows)),
                           int(max(r["shell_atoms"] for r in rows))],
           "core_atoms": sorted({int(r["core_atoms"]) for r in rows}),
           "core_radius": sorted({round(v, 2) for v in radius.values()}),
           "ligands": [int(min(r["ligands"] for r in rows)), int(max(r["ligands"] for r in rows))],
           "build_seconds": t_build, "kernel_seconds": seconds, "results": res}
    jdump(out, os.path.join(RESULTS, "model.json"))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report() -> str:
    lines = ["# The surface kernel on the surface of a particle (plan item NT5)", ""]
    cp = os.path.join(RESULTS, "checks.json")
    if os.path.isfile(cp):
        C = jload(cp)
        lines += ["## Checks", "",
                  "Differences are the largest difference of two rows of channels divided by "
                  "the largest entry. Heights 2 to 12 A every 2 A, distances along the surface "
                  "to 12 A every 1 A.", "",
                  "A flat core: a molecule on a gold slab whose top layer is at z = 0, the "
                  "particle kernel with a plane against the flat surface kernel.", "",
                  "| Core | Molecule | Difference |", "|---|---|---:|"]
        for r in C["flat"]:
            lines.append(f"| {r['core']} | {r['molecule']} | {r['difference']:.1e} |")
        lines += ["", "Invariance: ethanol on a gold particle (sphere) and on a gold wire "
                  "(cylinder), the whole system turned and moved. The flat kernel, which takes "
                  "z as the height, is shown for the same pair.", "",
                  "| Core | Particle kernel | Flat surface kernel |", "|---|---:|---:|"]
        for r in C["invariance"]:
            lines.append(f"| {r['shape']} | {r['difference']:.1e} | "
                         f"{r['flat_kernel_difference']:.1e} |")
        lines += ["", "Curvature: one flat patch of 40 atoms (heights 2.5 to 10 A, 12 x 12 A "
                  "across) wrapped onto spheres of growing radius, against the flat kernel of "
                  "the patch.", "", "| Radius of the sphere, A | Difference |", "|---:|---:|"]
        for r in C["curvature"]:
            lines.append(f"| {r['radius']:g} | {r['difference']:.1e} |")
        lines += ["", "| Core atoms | Shell atoms | Seconds for one shell |", "|---:|---:|---:|"]
        for r in C["cost"]:
            lines.append(f"| {r['core_atoms']} | {r['shell_atoms']} | {r['seconds']:.2f} |")
        lines.append("")
    mp = os.path.join(RESULTS, "model.json")
    if os.path.isfile(mp):
        M = jload(mp)
        lines += ["## The accessibility of the end groups of a ligand shell", "",
                  f"{M['n']} generated shells on gold icosahedra of "
                  f"{', '.join(str(a) for a in M['core_atoms'])} atoms, "
                  f"{M['ligands'][0]} to {M['ligands'][1]} ligands, {M['shell_atoms'][0]} to "
                  f"{M['shell_atoms'][1]} shell atoms. Target: the mean accessibility of the "
                  f"end atoms to a 1.4 A probe, {M['target_range'][0]:.3f} to "
                  f"{M['target_range'][1]:.3f}. 5-fold cross-validation repeated with "
                  f"{len(SEEDS)} seeds (mean, and the standard deviation over the seeds). 'The "
                  "recipe' is 1/R, the coverage, the chain length and the end group, the "
                  "numbers each shell was built from.", "",
                  "| Model | RMSE | Standard deviation over seeds | q2 | Seconds for the kernel |",
                  "|---|---:|---:|---:|---:|"]
        for name, s in M["results"].items():
            sec = M["kernel_seconds"].get(name)
            lines.append(f"| {name} | {s['rmse']:.4f} | {s['rmse_sd']:.4f} | {s['q2']:.3f} | "
                         f"{'--' if sec is None else f'{sec:.0f}'} |")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("checks", "model"), action="append",
                    help="stage(s) to run; the report always follows")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    for st in args.stage or []:
        {"checks": stage_checks, "model": stage_model}[st]()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
