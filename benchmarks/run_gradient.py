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

Usage
-----
    python benchmarks/run_gradient.py
    python benchmarks/run_gradient.py --sets steroids,THR --steps 20 --step 0.05
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
    ap.add_argument("--steps", type=int, default=20, help="steepest-ascent steps")
    ap.add_argument("--step", type=float, default=0.05,
                    help="angstroms the furthest atom moves per step")
    ap.add_argument("--report", action="store_true",
                    help="rewrite the table from the stored results without measuring again")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    store = os.path.join(RESULTS, "gradient.json")
    if args.report:
        with open(store, encoding="utf-8") as fh:
            results = json.load(fh)
    else:
        names = [s.strip() for s in args.sets.split(",") if s.strip()]
        unknown = [n for n in names if n not in DATA]
        if unknown:
            print(f"unknown sets {unknown}; the ones defined are {', '.join(DATA)}")
            return 2
        results = run(names, steps=args.steps, step=args.step)
        with open(store, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(table(results))
    print(f"\nwritten to {relative_path(RESULTS, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
