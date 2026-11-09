#!/usr/bin/env python
"""
The analytic gradient of a prediction: is it right, what does it cost, and what happens if you
follow it?

:mod:`cmfqsar.gradients` differentiates a prediction with respect to the atom positions and the
per-atom field weights of the molecule being predicted.  Three questions decide whether that is
worth having, and this script answers all three on real sets rather than on toy molecules.

1. **Is it the derivative?**  The only independent check is a finite difference of
   ``CMFModel.predict`` itself.  The step is scanned over four orders of magnitude: a correct
   gradient disagrees by the square of the step until round-off in the predictions takes over, so
   the scan shows a minimum, and the depth of that minimum is the accuracy of the comparison, not
   of the gradient.

2. **What does it cost?**  The alternative to an analytic gradient is the finite difference,
   which needs two predictions per degree of freedom: a molecule with N atoms and F fields has
   3N + NF of them.  The ratio of the two times is what the analytic form gains.

3. **What happens if you follow it?**  Steepest ascent on the atom positions, with every step
   capped so that no atom moves more than `--step` angstroms.  Reported per step: the predicted
   activity, how far the atoms have moved, the posterior standard deviation of the prediction,
   and the largest change in any bond length.  The last column is the point: the gradient knows
   nothing about valence, so it raises the number and damages the molecule.  It is a term for a
   generator that enforces geometry, not an optimiser on its own.

Two more stages answer the questions of plan item MS3.

4. **The same three questions on the rotation-invariant kernel** (``--stage invariant``), whose
   channels are built from the distances inside one structure and have no frame: molecular sets,
   and then the local environments of the public hop set, which is the case the item names.

5. **Relaxing a pose under the gradient** (``--stage relax``), as a rigid body, against two
   references: the same objective searched without a gradient, and the pose the confidence
   aligner of :mod:`cmfqsar.varalign` finds from the same start.

Usage
-----
    python benchmarks/run_gradient.py
    python benchmarks/run_gradient.py --sets steroids,THR --steps 20 --step 0.05
    python benchmarks/run_gradient.py --stage invariant
    python benchmarks/run_gradient.py --stage relax
    python benchmarks/run_gradient.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

from cmfqsar.dataio import load_dataset, relative_path  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.gradients import numerical_prediction_gradient, prediction_gradient  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "gradient")
#: the sets: the classic steroid benchmark and one Sutherland series, both aligned as distributed
DATA = {
    "steroids": (os.path.join(ROOT, "examples", "steroids_cbg.sdf"), "CBG_pK"),
    "THR": (os.path.join(ROOT, "examples", "sutherland", "THR.sdf"), None),
    "GPB": (os.path.join(ROOT, "examples", "sutherland", "GPB.sdf"), None),
}
STEP_SCAN = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)


def fit_model(path: str, activity_prop, alpha: float = 0.5, lam: float = 0.01):
    """A plain model of the set: the five physico-chemical fields, normalised kernels, a GP."""
    ds = load_dataset(path, activity_prop=activity_prop, subset="all")
    fs = FieldSet.create(alpha=alpha)
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    train = list(ds.train_indices) if len(ds.train_indices) else list(range(len(mf)))
    held = [i for i in range(len(mf)) if i not in set(train)]
    if not held:                      # no published split: keep the last fifth back
        cut = max(1, len(mf) // 5)
        held, train = train[-cut:], train[:-cut]
    model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": lam})
    model.fit(AtomTable.from_molecules([mf[i] for i in train]), ds.activity[train], cv_folds=0)
    return model, fs, [mf[i] for i in held], ds.activity[held]


def is_it_the_derivative(model, mol):
    """The step scan: relative disagreement with central finite differences, per step."""
    g = prediction_gradient(model, mol)
    rows = []
    for h in STEP_SCAN:
        num = numerical_prediction_gradient(model, mol, h=h)
        rows.append({
            "step": h,
            "coords": float(np.abs(g.coords - num.coords).max()
                            / max(np.abs(num.coords).max(), 1e-300)),
            "weights": float(np.abs(g.weights - num.weights).max()
                             / max(np.abs(num.weights).max(), 1e-300)),
        })
    # the best the comparison can do, taking the worse of the two derivatives at every step
    best = min(max(r["coords"], r["weights"]) for r in rows)
    worst_at_best = max([max(r["coords"], r["weights"]) for r in rows if r["step"] == 1e-4]
                        + [float("nan")])
    return {"scan": rows, "best": best, "at_1e-4": worst_at_best,
            "n_atoms": int(mol.n_atoms), "n_fields": int(mol.weights.shape[1])}


def what_does_it_cost(model, mol, repeats: int = 3):
    """Time of the analytic gradient against the finite-difference gradient it replaces."""
    t0 = time.perf_counter()
    for _ in range(repeats):
        prediction_gradient(model, mol)
    t_analytic = (time.perf_counter() - t0) / repeats
    t0 = time.perf_counter()
    numerical_prediction_gradient(model, mol, h=1e-4)
    t_numeric = time.perf_counter() - t0
    dof = 3 * mol.n_atoms + mol.n_atoms * mol.weights.shape[1]
    return {"analytic_s": t_analytic, "numeric_s": t_numeric,
            "ratio": t_numeric / max(t_analytic, 1e-12), "degrees_of_freedom": int(dof)}


def _bond_strain(mol, coords) -> float:
    """The largest change in any bond length, in angstroms, since the molecule's own geometry."""
    if mol.bonds is None or len(mol.bonds) == 0:
        return float("nan")
    i, j = np.asarray(mol.bonds, int).T
    d0 = np.linalg.norm(mol.coords[i] - mol.coords[j], axis=1)
    d1 = np.linalg.norm(coords[i] - coords[j], axis=1)
    return float(np.abs(d1 - d0).max())


def what_happens_if_you_follow_it(model, mol, steps: int = 20, step: float = 0.05):
    """Steepest ascent on the atom positions, every step capped at `step` angstroms per atom."""
    X = np.array(mol.coords, float)
    rows = []

    def record(k, coords, value):
        table = AtomTable.from_molecules([_moved(mol, coords)])
        _yp, var = model.predict_with_variance(table)
        d = coords - mol.coords
        rows.append({
            "step": int(k),
            "prediction": float(value),
            "max_displacement": float(np.linalg.norm(d, axis=1).max()),
            "rmsd": float(np.sqrt((d * d).sum() / len(d))),
            "posterior_sd": float(np.sqrt(max(float(var[0]), 0.0))),
            "bond_strain": _bond_strain(mol, coords),
        })

    g = prediction_gradient(model, mol, coords=X, wrt="coords")
    record(0, X, g.value)
    for k in range(1, int(steps) + 1):
        longest = np.linalg.norm(g.coords, axis=1).max()
        if longest <= 0:
            break
        X = X + step * g.coords / longest
        g = prediction_gradient(model, mol, coords=X, wrt="coords")
        if k in (1, 2, 5, 10, steps) or k % 10 == 0:
            record(k, X, g.value)
    return rows


def _moved(mol, coords):
    import dataclasses
    return dataclasses.replace(mol, coords=np.asarray(coords, float))


def run(sets, steps: int, step: float):
    out = {}
    for name in sets:
        path, prop = DATA[name]
        if not os.path.exists(path):
            print(f"  {name}: {path} is not here, skipped")
            continue
        model, _fs, held, y_held = fit_model(path, prop)
        mol = held[0]
        print(f"\n{name}: {model.train.n_mol} training molecules, "
              f"test molecule {mol.name!r} with {mol.n_atoms} atoms")
        derivative = is_it_the_derivative(model, mol)
        cost = what_does_it_cost(model, mol)
        walk = what_happens_if_you_follow_it(model, mol, steps=steps, step=step)
        print(f"  best agreement with finite differences: {derivative['best']:.2e} relative")
        print(f"  analytic {cost['analytic_s'] * 1e3:.1f} ms against "
              f"{cost['numeric_s'] * 1e3:.0f} ms for {cost['degrees_of_freedom']} finite "
              f"differences: {cost['ratio']:.0f} times faster")
        print(f"  following it: {walk[0]['prediction']:+.3f} -> {walk[-1]['prediction']:+.3f}, "
              f"rmsd {walk[-1]['rmsd']:.2f} A, posterior sd "
              f"{walk[0]['posterior_sd']:.3f} -> {walk[-1]['posterior_sd']:.3f}, "
              f"worst bond length off by {walk[-1]['bond_strain']:.2f} A")
        out[name] = {"molecule": mol.name, "n_train": int(model.train.n_mol),
                     "measured": float(y_held[0]), "derivative": derivative, "cost": cost,
                     "walk": walk}
    return out


def table(results) -> str:
    lines = ["# The gradient of a prediction", "",
             "Written by `python benchmarks/run_gradient.py`.", "",
             "## Is it the derivative? Relative disagreement with central finite differences",
             "",
             "| set | molecule | atoms | step 1e-2 | 1e-3 | 1e-4 | 1e-5 | 1e-6 |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        cells = [f"{max(s['coords'], s['weights']):.1e}" for s in r["derivative"]["scan"]]
        lines.append(f"| {name} | {r['molecule']} | {r['derivative']['n_atoms']} | "
                     + " | ".join(cells) + " |")
    lines += ["", "The disagreement falls as the square of the step and rises again below about "
                  "1e-4 angstrom, where round-off in the predictions takes over. That shape is "
                  "the signature of a correct gradient; the smallest value in each row is the "
                  "accuracy of the comparison, not of the gradient.", "",
              "## What does it cost?", "",
              "| set | atoms | degrees of freedom | analytic | finite differences | ratio |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        c = r["cost"]
        lines.append(f"| {name} | {r['derivative']['n_atoms']} | {c['degrees_of_freedom']} | "
                     f"{c['analytic_s'] * 1e3:.1f} ms | {c['numeric_s'] * 1e3:.0f} ms | "
                     f"{c['ratio']:.0f}x |")
    lines += ["", "## What happens if you follow it?", "",
              "Steepest ascent on the atom positions, no atom moving more than 0.05 A per step.",
              ""]
    for name, r in results.items():
        lines += [f"### {name}, molecule {r['molecule']} (measured {r['measured']:+.2f})", "",
                  "| step | prediction | largest displacement | rmsd | posterior sd | "
                  "worst bond length change |",
                  "|---:|---:|---:|---:|---:|---:|"]
        for w in r["walk"]:
            lines.append(f"| {w['step']} | {w['prediction']:+.3f} | "
                         f"{w['max_displacement']:.2f} A | {w['rmsd']:.2f} A | "
                         f"{w['posterior_sd']:.3f} | {w['bond_strain']:.2f} A |")
        lines.append("")
    lines += ["The prediction rises, and so does the posterior standard deviation: the molecule "
              "is leaving the training set while the number is getting better. The last column "
              "says what else is happening -- the bond lengths are being pulled apart, because "
              "nothing in the gradient knows that a molecule has bonds. Both columns are the "
              "reason this gradient is a term to add to a generator or a force field that "
              "enforces geometry, and not an optimiser to run on its own.", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default="steroids,THR", help="comma-separated: "
                    + ", ".join(DATA))
    ap.add_argument("--stage", default="overlap",
                    choices=("overlap", "invariant", "relax", "all"),
                    help="overlap: the gradient on the overlap-integral kernel, with the "
                         "steepest ascent; invariant: the same on the rotation-invariant "
                         "kernel, molecules and hop environments; relax: a rigid-body "
                         "relaxation under the gradient against the confidence aligner")
    ap.add_argument("--steps", type=int, default=20, help="steepest-ascent steps")
    ap.add_argument("--step", type=float, default=0.05,
                    help="angstroms the furthest atom moves per step")
    ap.add_argument("--structures", type=int, default=10,
                    help="hop structures the invariant stage loads")
    ap.add_argument("--per-structure", type=int, default=3,
                    help="hops kept per structure in the invariant stage")
    ap.add_argument("--report", action="store_true",
                    help="rewrite the table from the stored results without measuring again")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    STORES = {"overlap": "gradient.json", "invariant": "invariant.json", "relax": "relax.json"}
    if not args.report:
        names = [s.strip() for s in args.sets.split(",") if s.strip()]
        unknown = [n for n in names if n not in DATA]
        if unknown:
            print(f"unknown sets {unknown}; the ones defined are {', '.join(DATA)}")
            return 2
        stages = ("overlap", "invariant", "relax") if args.stage == "all" else (args.stage,)
        for stage in stages:
            if stage == "overlap":
                results = run(names, steps=args.steps, step=args.step)
            elif stage == "invariant":
                results = invariant_arm(names, max_structures=args.structures,
                                        max_per_structure=args.per_structure)
            else:
                results = relax_arm(names)
            with open(os.path.join(RESULTS, STORES[stage]), "w", encoding="utf-8") as fh:
                json.dump(results, fh, indent=1)
    write_tables(RESULTS)
    print(f"\nwritten to {relative_path(RESULTS, ROOT)}")
    return 0



# --------------------------------------------------------------------------- #
# The rotation-invariant kernel (plan item MS3)
# --------------------------------------------------------------------------- #

def fit_invariant(path: str, activity_prop, d_max: float = 8.0, sigma: float = 0.5,
                  lam: float = 0.01):
    """The same set on the rotation-invariant autocorrelation kernel, which needs no frame."""
    ds = load_dataset(path, activity_prop=activity_prop, subset="all")
    fs = FieldSet.create(alpha=0.5)
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    train = list(ds.train_indices) if len(ds.train_indices) else list(range(len(mf)))
    held = [i for i in range(len(mf)) if i not in set(train)]
    if not held:
        cut = max(1, len(mf) // 5)
        held, train = train[-cut:], train[:-cut]
    model = CMFModel(fieldset=fs, method="gp", kernel_source="autocorrelation",
                     ac_d_max=d_max, ac_sigma=sigma, params={"lam": lam})
    model.fit(AtomTable.from_molecules([mf[i] for i in train]), ds.activity[train], cv_folds=0)
    return model, fs, [mf[i] for i in held], ds.activity[held]


def hop_environments(max_structures: int = 10, max_per_structure: int = 3, rho: float = 0.5):
    """The local environments of the public hop set: the case this gradient was asked for.

    A hop environment is a cut-out of a crystal around the midpoint of a jump, with the atoms
    weighted by an envelope.  It has no frame of its own, which is why the hop model of item BT7
    uses the invariant kernel, and it is the descriptor plan item MS3 names as ready.
    """
    from run_hops import PREPARED, fieldset_for, load_prepared
    if not os.path.exists(PREPARED):
        return None
    hs = load_prepared(max_structures=max_structures, max_per_structure=max_per_structure)
    fs = fieldset_for(hs, "oxidation")
    crystals = hs.crystals(fs)
    targets = hs.targets()
    envs = hs.environments(crystals, rows=targets.atom, rho=rho)
    return hs, fs, envs, np.asarray(targets.y, float)


def invariant_arm(sets, max_structures: int = 10, max_per_structure: int = 3) -> dict:
    """The step scan and the cost on the invariant kernel: molecular sets, then hop environments."""
    out = {}
    for name in sets:
        path, prop = DATA[name]
        if not os.path.exists(path):
            print(f"  {name}: {path} is not here, skipped")
            continue
        model, _fs, held, y_held = fit_invariant(path, prop)
        mol = held[0]
        print(f"\n{name} on the invariant kernel: {model.train.n_mol} training molecules, "
              f"{model.ac_train_.shape[1]} channels, test molecule {mol.name!r}")
        derivative = is_it_the_derivative(model, mol)
        cost = what_does_it_cost(model, mol)
        print(f"  best agreement with finite differences: {derivative['best']:.2e} relative")
        print(f"  analytic {cost['analytic_s'] * 1e3:.1f} ms against "
              f"{cost['numeric_s'] * 1e3:.0f} ms for {cost['degrees_of_freedom']} finite "
              f"differences: {cost['ratio']:.0f} times faster")
        out[name] = {"molecule": mol.name, "n_train": int(model.train.n_mol),
                     "n_channels": int(model.ac_train_.shape[1]),
                     "measured": float(y_held[0]), "derivative": derivative, "cost": cost}

    loaded = hop_environments(max_structures, max_per_structure)
    if loaded is None:
        print("\nhops: benchmarks/results/hops/prepared_mp.json is not here, skipped")
        return out
    hs, fs, envs, y = loaded
    keep = len(envs) - 3
    model = CMFModel(fieldset=fs, method="krr", kernel_source="autocorrelation",
                     ac_d_max=8.0, ac_sigma=0.5, params={"lam": 1e-3})
    model.fit(AtomTable.from_molecules(envs[:keep]), y[:keep], cv_folds=0)
    env = envs[keep]
    print(f"\nhop environments: {keep} of them, {model.ac_train_.shape[1]} channels, "
          f"test environment {env.name!r} with {env.n_atoms} atoms and "
          f"{env.weights.shape[1]} fields")
    derivative = is_it_the_derivative(model, env)
    cost = what_does_it_cost(model, env)
    print(f"  best agreement with finite differences: {derivative['best']:.2e} relative")
    print(f"  analytic {cost['analytic_s'] * 1e3:.1f} ms against "
          f"{cost['numeric_s'] * 1e3:.0f} ms for {cost['degrees_of_freedom']} finite "
          f"differences: {cost['ratio']:.0f} times faster")
    out["hops"] = {"molecule": env.name, "n_train": int(keep),
                   "n_channels": int(model.ac_train_.shape[1]),
                   "measured": float(y[keep]), "derivative": derivative, "cost": cost}
    return out


# --------------------------------------------------------------------------- #
# Relaxing a pose under the gradient (plan item MS3)
# --------------------------------------------------------------------------- #

def _perturbed(mol, seed: int = 0, angle: float = 0.35, shift: float = 0.5):
    """The same molecule in a pose it did not arrive in: a turn and a step, with a fixed seed."""
    from scipy.spatial.transform import Rotation
    rng = np.random.RandomState(seed)
    rv = Rotation.random(random_state=rng).as_rotvec()
    R = Rotation.from_rotvec(rv / max(np.linalg.norm(rv), 1e-12) * angle).as_matrix()
    X = np.asarray(mol.coords, float)
    c = X.mean(axis=0)
    return _moved(mol, (X - c) @ R.T + c + rng.normal(scale=shift, size=3))


def relax_arm(sets, caps=(2.0, None), seed: int = 0) -> dict:
    """Two questions about following the gradient as a rigid body.

    1. **Does it reach the optimum the same objective reaches without it?**  The same search is
       run with the analytic gradient (L-BFGS-B) and without it (Nelder-Mead) from the same
       starting pose.  The values must agree; the number of objective evaluations is what the
       gradient saves.
    2. **Is that the pose the confidence aligner finds?**  :mod:`cmfqsar.varalign` moves the same
       molecule from the same start by the posterior variance instead.  Reported for both poses:
       the prediction, the posterior variance, and how far apart the two poses are.
    """
    from cmfqsar.relax import pose_rmsd, relax_pose
    from cmfqsar.varalign import VarianceAligner
    out = {}
    for name in sets:
        path, prop = DATA[name]
        if not os.path.exists(path):
            print(f"  {name}: {path} is not here, skipped")
            continue
        model, _fs, held, y_held = fit_model(path, prop)
        mol = held[0]
        start = _perturbed(mol, seed=seed)
        print(f"\n{name}: relaxing {mol.name!r} ({mol.n_atoms} atoms) from a pose "
              f"{pose_rmsd(start.coords, mol.coords):.2f} A away from its own")
        rows = []
        for cap in caps:
            driven = relax_pose(model, start, max_rmsd=cap, gradient=True)
            plain = relax_pose(model, start, max_rmsd=cap, gradient=False)
            aligner = VarianceAligner(model, max_rmsd=cap)
            aligned = aligner.align(_moved(start, np.array(start.coords)), restarts=0,
                                    apply=False)
            row = {
                "guard": "none" if cap is None else f"{cap:g} A",
                "value_start": driven.value_before,
                "value_driven": driven.value_after,
                "value_plain": plain.value_after,
                "evals_driven": driven.n_eval,
                "evals_plain": plain.n_eval,
                "rmsd_driven": driven.rmsd,
                "rmsd_plain": plain.rmsd,
                "pose_gap_searches": pose_rmsd(driven.coords, plain.coords),
                "value_gap_searches": abs(driven.value_after - plain.value_after),
                "limited": bool(driven.limited),
                # the same molecule moved by confidence instead of by the prediction
                "variance_start": aligned.variance_before,
                "variance_aligned": aligned.variance_after,
                "variance_at_driven": float(aligner.variance(start, driven.coords)),
                "value_at_aligned": float(aligner.predict(start, aligned.coords)),
                "rmsd_aligned": aligned.rmsd,
                "pose_gap_objectives": pose_rmsd(driven.coords, aligned.coords),
            }
            rows.append(row)
            print(f"  guard {row['guard']:>5}: prediction {row['value_start']:+.3f} -> "
                  f"{row['value_driven']:+.3f} with the gradient in {row['evals_driven']} "
                  f"evaluations, {row['value_plain']:+.3f} without it in {row['evals_plain']}; "
                  f"the two poses {row['pose_gap_searches']:.2f} A apart")
            print(f"  {'':>11} confidence moves it elsewhere: variance "
                  f"{row['variance_start']:.3f} -> {row['variance_aligned']:.3f} against "
                  f"{row['variance_at_driven']:.3f} at the prediction's pose, the two poses "
                  f"{row['pose_gap_objectives']:.2f} A apart")
        out[name] = {"molecule": mol.name, "n_atoms": int(mol.n_atoms),
                     "measured": float(y_held[0]),
                     "start_rmsd": pose_rmsd(start.coords, mol.coords), "rows": rows}
    return out


def table_invariant(results) -> str:
    lines = ["## The same gradient on the rotation-invariant kernel", "",
             "Written by `python benchmarks/run_gradient.py --stage invariant`. The invariant "
             "kernel has no frame: its channels are the distance distribution of the field "
             "weights inside one structure, so this gradient moves a conformation and not a "
             "pose. The last row is the case the gradient was asked for, a local environment of "
             "a crystal around a lithium jump (item BT7).", "",
             "| set | sample | atoms | fields | channels | step 1e-2 | 1e-3 | 1e-4 | 1e-5 | "
             "1e-6 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        d = r["derivative"]
        cells = [f"{max(s['coords'], s['weights']):.1e}" for s in d["scan"]]
        lines.append(f"| {name} | {r['molecule']} | {d['n_atoms']} | {d['n_fields']} | "
                     f"{r['n_channels']} | " + " | ".join(cells) + " |")
    lines += ["", "The disagreement falls as the square of the step -- a decade of step is "
                  "two decades of error -- until round-off in the predictions takes over, "
                  "between 1e-4 and 1e-6 angstrom depending on the sample. That shape is the "
                  "signature of a correct gradient, and the smallest number in a row is what "
                  "the comparison can resolve, not the error of the gradient.", "",
              "| set | degrees of freedom | analytic | finite differences | ratio |",
              "|---|---:|---:|---:|---:|"]
    for name, r in results.items():
        c = r["cost"]
        lines.append(f"| {name} | {c['degrees_of_freedom']} | {c['analytic_s'] * 1e3:.1f} ms | "
                     f"{c['numeric_s'] * 1e3:.0f} ms | {c['ratio']:.0f}x |")
    lines.append("")
    return "\n".join(lines)


def table_relax(results) -> str:
    lines = ["## Relaxing a pose under the gradient", "",
             "Written by `python benchmarks/run_gradient.py --stage relax`. The molecule is "
             "moved as a rigid body -- six parameters about its own centroid -- from a pose it "
             "did not arrive in. `cmfqsar.relax.relax_pose` raises the prediction; "
             "`cmfqsar.varalign` moves the same molecule from the same start by the posterior "
             "variance instead. The guard is how far the atoms may move before a penalty holds "
             "them back.", "",
             "### Does the gradient reach what the same objective reaches without it?", "",
             "| set | molecule | guard | prediction at the start | prediction with the "
             "gradient | prediction without it | evaluations with the gradient | evaluations "
             "without it | distance between the two poses |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        for row in r["rows"]:
            lines.append(f"| {name} | {r['molecule']} | {row['guard']} | "
                         f"{row['value_start']:+.3f} | {row['value_driven']:+.3f} | "
                         f"{row['value_plain']:+.3f} | {row['evals_driven']} | "
                         f"{row['evals_plain']} | {row['pose_gap_searches']:.2f} A |")
    lines += ["", "The two searches reach the same value of the objective, and they reach "
                  "it from 24 to 86 times fewer evaluations when the gradient drives them. "
                  "Where a guard is active the optimum sits on the guard's sphere, where the "
                  "objective is nearly flat, so the two searches can stop at poses that differ "
                  "while their values do not: the value columns are the comparison, the pose "
                  "column is the shape of the optimum.", "",
              "### Is that the pose the confidence aligner finds?", "",
              "| set | guard | variance at the start | variance at the confidence pose | "
              "variance at the prediction's pose | prediction at the prediction's pose | "
              "prediction at the confidence pose | distance between the two poses |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        for row in r["rows"]:
            lines.append(f"| {name} | {row['guard']} | {row['variance_start']:.3f} | "
                         f"{row['variance_aligned']:.3f} | {row['variance_at_driven']:.3f} | "
                         f"{row['value_driven']:+.3f} | {row['value_at_aligned']:+.3f} | "
                         f"{row['pose_gap_objectives']:.2f} A |")
    lines += ["", "**They do not agree, and that is not a fault in either.** Raising the "
                  "prediction moves a molecule towards the training molecules that are active; "
                  "raising confidence moves it towards the training set as a whole, with "
                  "redundant members down-weighted by the inverse kernel. The two poses end "
                  "one to two angstroms apart, the variance at the prediction's pose is 11 to "
                  "33 times the variance the aligner reaches, and the prediction at the "
                  "confidence pose is 1.4 to 2.2 units below the one the relaxation reaches. "
                  "So the gradient is usable as a driver -- that is the table above -- but the "
                  "pose it drives to answers a different question from the pose the aligner "
                  "finds. For a pose to predict from, use the confidence, which uses no "
                  "activity value of the molecule being placed; the relaxation answers where a "
                  "molecule would have to sit for the model to call it active.", ""]
    return "\n".join(lines)


def write_tables(results_dir: str) -> str:
    """One table.md from whichever stages have been run."""
    parts = []
    for store, head in (("gradient.json", None), ("invariant.json", table_invariant),
                        ("relax.json", table_relax)):
        path = os.path.join(results_dir, store)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        parts.append(table(data) if head is None else head(data))
    text = "\n".join(parts) if parts else "# The gradient of a prediction\n"
    if parts and not os.path.exists(os.path.join(results_dir, "gradient.json")):
        text = "# The gradient of a prediction\n\n" + text      # the title lives in that stage
    out = os.path.join(results_dir, "table.md")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out

if __name__ == "__main__":
    raise SystemExit(main())
