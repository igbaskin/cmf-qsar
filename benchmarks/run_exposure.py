#!/usr/bin/env python
"""
Does weighting the fields by solvent exposure help?  The surface claim, measured.

A CMF kernel integrates over all of space, so it weights an atom buried inside a molecule as
heavily as one on the outside, although a binding partner can only touch the second.  That is the
claim the surface methods are founded on, and ``cmfqsar/exposure.py`` tests it cheaply: each
atomic weight is multiplied by its atom's relative solvent accessibility, which keeps the kernel
in closed form and keeps the co-field.  Nothing else changes -- the same fields, the same alpha and
lambda grids, the same 10-fold cross-validation, the same molecules.

Two families, chosen because their frames differ in origin and not in quality:

    sutherland  the eight sets of the 2013 paper, aligned by their authors
    openff      sixteen FEP sets whose ligands are co-crystallised, so the frame is a crystal one

Four weightings are compared with the plain fields: exposure unchanged, softened
(``power 0.5``), sharpened (``power 2``), and with a floor of 0.2 so that a shielded atom keeps a
fifth of its weight.  The exposure is computed **by group**: a heavy atom and its hydrogens count
as one, because measured atom by atom a methyl carbon is occluded by its own hydrogens and the
weighting would be counting hydrogens rather than burial.

The geometry says in advance what to expect, and the table reports it: a drug-sized ligand has a
median group exposure near 0.4 and **no buried group at all**, so this is a smooth reweighting of
the surface and not the removal of an interior.  There is no interior in a molecule of thirty heavy
atoms.

Usage
-----
    python benchmarks/run_exposure.py --family openff
    python benchmarks/run_exposure.py --family sutherland
    python benchmarks/run_exposure.py --report
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

from rdkit import Chem, RDLogger  # noqa: E402

import run_topomer as T  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.exposure import atom_exposure, exposed_set  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "exposure")
SUTHERLAND = os.path.join(ROOT, "examples", "sutherland")
OPENFF = os.path.join(ROOT, "examples", "openff")

#: (tag, kwargs for atom_exposure)
WEIGHTINGS = [
    ("plain", None),
    ("exposure", {}),
    ("exposure^0.5", {"power": 0.5}),
    ("exposure^2", {"power": 2.0}),
    ("exposure+floor", {"floor": 0.2}),
]


def load_family(family):
    """[(set name, molecules, names, activities)] for one family."""
    out = []
    if family == "sutherland":
        for name in T.SETS:
            ds = load_dataset(os.path.join(SUTHERLAND, f"{name}.sdf"))
            y = np.asarray(ds.activity, float)
            ok = np.isfinite(y)
            out.append((name, [m for m, g in zip(ds.mols, ok) if g],
                        [n for n, g in zip(ds.names, ok) if g], y[ok]))
    else:
        import run_poses as P
        for name in P.SETS:
            mols, names, y = P.load(name)
            out.append((name, mols, names, y))
    return out


def best_q2(fields_by_alpha, y, fs_by_alpha, lams=T.LAMS, folds=T.FOLDS, seed=T.SEED):
    """Best (q2, RMSEcv, alpha, lambda) over the grid, scored by the CV it selects on."""
    best = None
    for alpha, mf in fields_by_alpha.items():
        model = CMFModel(fieldset=fs_by_alpha[alpha], method="krr", normalize=True)
        K = model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(mf)))
        for lam in lams:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=folds, seed=seed)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y, pred), float(alpha), float(lam))
    return best


def run_set(name, mols, names, y, charge_model="gasteiger", weightings=WEIGHTINGS):
    t0 = time.time()
    exposures = [atom_exposure(m) for m in mols]
    heavy = [np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()]) for m in mols]
    median_e = float(np.mean([np.median(e[h]) for e, h in zip(exposures, heavy)]))
    buried = float(np.mean([np.mean(e[h] < 0.05) for e, h in zip(exposures, heavy)]))
    print(f"\n=== {name}: {len(mols)} molecules, median group exposure {median_e:.2f}, "
          f"groups below 0.05: {buried:.1%}", flush=True)
    res = {"set": name, "n": len(mols), "median_exposure": median_e, "buried_fraction": buried,
           "charge_model": charge_model, "fields": T.FIELDS, "alphas": list(T.ALPHAS),
           "lams": list(T.LAMS), "folds": T.FOLDS, "models": {}}
    fs_by_alpha = {a: FieldSet.create(physchem=T.FIELDS, alpha=a, charge_model=charge_model,
                                      param_scheme="rdkit") for a in T.ALPHAS}
    for tag, kw in weightings:
        if kw is None:
            by_alpha = {a: [build_fields(m, fs_by_alpha[a], n) for m, n in zip(mols, names)]
                        for a in T.ALPHAS}
        else:
            by_alpha = {a: exposed_set(mols, fs_by_alpha[a], names, **kw) for a in T.ALPHAS}
        q2, err, alpha, lam = best_q2(by_alpha, y, fs_by_alpha)
        res["models"][tag] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam}
        print(f"    {tag:16s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (alpha {alpha:g}, lam {lam:g})",
              flush=True)
    res["seconds"] = time.time() - t0
    return res


def run_family(family, results=RESULTS, force=False):
    out_path = os.path.join(results, f"{family}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    charge_model = "file" if family == "sutherland" else "gasteiger"
    rows = [run_set(name, mols, names, y, charge_model=charge_model)
            for name, mols, names, y in load_family(family)]
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({"family": family, "sets": rows}, fh, indent=1)
    print(f"\n    written to {out_path}", flush=True)
    return {"family": family, "sets": rows}


TAGS = [t for t, _ in WEIGHTINGS]


def report(results=RESULTS, out=None):
    families = []
    for fn in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        if fn.endswith(".json"):
            families.append(json.load(open(os.path.join(results, fn))))
    if not families:
        raise SystemExit(f"no results in {results}")
    lines = ["# Fields weighted by solvent exposure, against the plain fields", "",
             "Each atomic weight multiplied by its atom's relative solvent accessibility,",
             "computed by group (a heavy atom and its hydrogens count as one). Same fields,",
             "same alpha and lambda grids, same 10-fold CV, same molecules.", ""]
    for fam in families:
        rows = sorted(fam["sets"], key=lambda r: -r["n"])
        lines += [f"## {fam['family']}", "",
                  "| set | n | median exposure | buried | " + " | ".join(TAGS) + " |",
                  "|---|---:|---:|---:|" + "---:|" * len(TAGS)]
        for r in rows:
            cells = [f"{r['models'][t]['q2']:+.3f}" if t in r["models"] else "-" for t in TAGS]
            lines.append(f"| {r['set']} | {r['n']} | {r['median_exposure']:.2f} | "
                         f"{r['buried_fraction']:.0%} | " + " | ".join(cells) + " |")
        means = [np.mean([r["models"][t]["q2"] for r in rows if t in r["models"]]) for t in TAGS]
        lines.append("| **mean** | | | | " + " | ".join(f"**{m:+.3f}**" for m in means) + " |")
        best = [np.mean([r["models"][t]["q2"] - r["models"]["plain"]["q2"] for r in rows
                         if t in r["models"]]) for t in TAGS]
        lines.append("| **against plain** | | | | "
                     + " | ".join("-" if t == "plain" else f"{d:+.3f}"
                                  for t, d in zip(TAGS, best)) + " |")
        lines.append("")
    text = "\n".join(lines) + "\n"
    path = out or os.path.join(results, "table.md")
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written {path}")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["sutherland", "openff", "both"], default="both")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        return report()
    fams = ["sutherland", "openff"] if args.family == "both" else [args.family]
    for fam in fams:
        run_family(fam, force=args.force)
    report()


if __name__ == "__main__":
    main()
