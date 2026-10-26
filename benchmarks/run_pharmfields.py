#!/usr/bin/env python
"""
Six pharmacophore features against twenty-odd Sybyl types, as indicator fields (plan item D2).

An indicator field puts a Gaussian of weight one on every atom of one kind.  Keyed by Sybyl atom
type -- the 2015 paper's continuous indicator fields, `cif:C.ar` and the rest -- they help on sets
above about forty molecules and not below, and the reason is arithmetic: a congeneric series has
twenty-odd types, each is rare, and a field whose weights are zero on most molecules contributes
a kernel column of mostly zeros.

`cmfqsar/pharmtypes.py` offers six instead: donor, acceptor, aromatic, hydrophobe, cation, anion,
from RDKit's own feature definitions.  Each is carried by most molecules of a series, and the
vocabulary is the one a pharmacophore is written in, so the co-field of `pharm:donor` says where a
donor is wanted without a translation step.

Measured on the eight Sutherland sets, which is where the Sybyl indicators were measured, with the
sets' own alignment, the same alpha and lambda grids and the same ten-fold cross-validation:

    five            the five physico-chemical fields, the reference
    five + pharm    with the six pharmacophore indicators added
    pharm           the indicators alone, no physico-chemical field at all
    five + cif      with the Sybyl indicators added, which is what the item compares against
    cif             the Sybyl indicators alone

and the coverage of both vocabularies -- what share of the molecules carries each kind -- because
that is what the argument is about.

The item asks for the result split by set size, and the Sutherland sets cannot give it: the
smallest of the eight has 66 molecules, so all of them are on the large side of the forty the item
names.  ``--mittal`` runs the same measurement on ``examples/mittal`` instead, 38 sets of which 15
are below forty molecules, which is where the question actually lives.  Those sets are aligned
automatically rather than by hand, so their absolute numbers are lower than the Sutherland ones
and only the columns are comparable -- which is all this measurement needs.

Usage
-----
    python benchmarks/run_pharmfields.py                 # the eight Sutherland sets
    python benchmarks/run_pharmfields.py --mittal         # 38 sets, 15 of them small
    python benchmarks/run_pharmfields.py --set THR
    python benchmarks/run_pharmfields.py --report
"""
from __future__ import annotations

import argparse
import glob
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

import kernel_cache as KC  # noqa: E402
import run_topomer as T  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields, cif_types_in  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402
from cmfqsar.pharmtypes import (PHARM_FEATURES, coverage, definitions_used,  # noqa: E402
                                features_in)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "pharmfields")
MITTAL_RESULTS = RESULTS + "_mittal"
#: above this many molecules the Sybyl indicators were found to help, below it not
SIZE_SPLIT = 40


def kernels_for(mols, names, fs):
    """The joint kernel of one field set at every attenuation of the grid."""
    out = {}
    for alpha in T.ALPHAS:
        fs2 = FieldSet.create(
            physchem=[sp.name for sp in fs.specs if not (sp.is_cif or sp.is_pharm)],
            cif_types=[sp.cif_type for sp in fs.specs if sp.is_cif],
            pharm_features=[sp.pharm_feature for sp in fs.specs if sp.is_pharm],
            alpha=alpha, charge_model=fs.charge_model, param_scheme=fs.param_scheme,
            cif_scheme=fs.cif_scheme)
        mf = [build_fields(m, fs2, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs2, method="krr", normalize=True)
        out[alpha] = KC.joint_training_kernel(model, AtomTable.from_molecules(mf))
    return out


def score(kernels, y):
    """Best (q2, RMSEcv, alpha, lambda) over the attenuation and ridge grids."""
    best = None
    for alpha, K in kernels.items():
        for lam in T.LAMS:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=T.FOLDS, seed=T.SEED)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y, pred), float(alpha), float(lam))
    return best


def run_set(name, results=RESULTS, force=False, data=None):
    path = os.path.join(results, f"{name}.json")
    if os.path.exists(path) and not force:
        print(f"{name}: already done")
        return json.load(open(path))
    t0 = time.time()
    ds = load_dataset(os.path.join(data or T.DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    mols, names = list(ds.mols), list(ds.names)
    cif = cif_types_in(mols, "sybyl", 5)
    pharm = features_in(mols, 5)
    cov = coverage(mols)
    from cmfqsar.fields import atom_type_counts
    cif_counts = atom_type_counts(mols, "sybyl")
    print(f"=== {name}: {len(mols)} molecules, {len(cif)} Sybyl types, "
          f"{len(pharm)} of the six features", flush=True)

    sets = {
        "five": FieldSet.create(physchem=T.FIELDS, charge_model="file"),
        "five + pharm": FieldSet.create(physchem=T.FIELDS, pharm_features=pharm,
                                        charge_model="file"),
        "pharm": FieldSet.create(physchem=[], pharm_features=pharm, charge_model="file"),
        "five + cif": FieldSet.create(physchem=T.FIELDS, cif_types=cif, charge_model="file"),
        "cif": FieldSet.create(physchem=[], cif_types=cif, charge_model="file"),
    }
    out = {"set": name, "n": len(mols), "n_cif_types": len(cif), "cif_types": cif,
           "n_cif_types_all": len(cif_counts),
           "pharm_features": pharm, "definitions": definitions_used(),
           "coverage": {k: v for k, v in cov.items()},
           # the kept types and every type: cif_types_in already drops the ones in fewer than
           # five molecules, so comparing the six features with the kept types alone would be
           # comparing against a vocabulary that has had its rare half removed
           "cif_coverage": {t: cif_counts.get(t, 0) / len(mols) for t in cif},
           "cif_coverage_all": {t: c / len(mols) for t, c in cif_counts.items()},
           "models": {}}
    for label, fs in sets.items():
        q2, err, alpha, lam = score(kernels_for(mols, names, fs), y)
        out["models"][label] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam,
                                "n_fields": len(fs.specs)}
        print(f"    {label:14s} {len(fs.specs):3d} fields   q2 {q2:+.3f}   RMSEcv {err:.3f}"
              f"   (alpha {alpha:g}, lam {lam:g})", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(results, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


ORDER = ["five", "five + pharm", "pharm", "five + cif", "cif"]


def report(results=RESULTS):
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    data = [json.load(open(f)) for f in files]
    data = [d for d in data if "models" in d]
    if not data:
        return "No pharmacophore-field results yet; run `python benchmarks/run_pharmfields.py`."
    data.sort(key=lambda d: d["n"])
    lines = ["# Six pharmacophore features against the Sybyl indicators", "",
             "Indicator fields put a Gaussian of weight one on every atom of one kind. Keyed by "
             "Sybyl atom type they help above about forty molecules and not below, because a "
             "congeneric series has twenty-odd types and each is rare. These six -- donor, "
             "acceptor, aromatic, hydrophobe, cation, anion, from "
             f"{data[0]['definitions']} -- are each carried by most molecules of a series. "
             + (f"The {len(data)} Mittal sets, whose alignments are automatic, so the absolute "
                "numbers are lower than on the hand-aligned sets and only the columns compare."
                if len(data) > 10 else
                "Eight Sutherland sets, their own alignment.")
             + " Same attenuation and ridge grids and the same ten-fold cross-validation as "
               "`run_alignment.py`.", "",
             "| set | n | " + " | ".join(ORDER) + " | best |",
             "|---|---:|" + "---:|" * len(ORDER) + "---|"]
    for d in data:
        m = d["models"]
        best = max(ORDER, key=lambda k: m[k]["q2"] if k in m else -9)
        cells = [f"**{m[k]['q2']:+.3f}**" if k == best else f"{m[k]['q2']:+.3f}"
                 for k in ORDER if k in m]
        lines.append(f"| {d['set']} | {d['n']} | " + " | ".join(cells) + f" | {best} |")
    means = {k: float(np.mean([d["models"][k]["q2"] for d in data if k in d["models"]]))
             for k in ORDER}
    lines.append("| **mean** | | " + " | ".join(f"{means[k]:+.3f}" for k in ORDER) + " | |")

    small = [d for d in data if d["n"] < SIZE_SPLIT]
    large = [d for d in data if d["n"] >= SIZE_SPLIT]
    lines += ["", f"Split at {SIZE_SPLIT} molecules, which is where the Sybyl indicators were "
                  f"found to start helping"
                  + ("" if small else " -- and every set here is above it, which is why the "
                                     "`--mittal` run exists") + ":", "",
              "| sets | n | " + " | ".join(ORDER) + " |",
              "|---|---:|" + "---:|" * len(ORDER)]
    for label, group in (("below", small), ("at or above", large)):
        if not group:
            continue
        row = [f"{float(np.mean([d['models'][k]['q2'] for d in group])):+.3f}" for k in ORDER]
        lines.append(f"| {label} {SIZE_SPLIT} | {len(group)} | " + " | ".join(row) + " |")

    d_pharm = float(np.mean([d["models"]["five + pharm"]["q2"] - d["models"]["five"]["q2"]
                             for d in data]))
    d_cif = float(np.mean([d["models"]["five + cif"]["q2"] - d["models"]["five"]["q2"]
                           for d in data]))
    win_p = sum(1 for d in data
                if d["models"]["five + pharm"]["q2"] > d["models"]["five"]["q2"])
    win_c = sum(1 for d in data if d["models"]["five + cif"]["q2"] > d["models"]["five"]["q2"])
    lines += ["",
              f"**Added to the five fields, the six features are worth {d_pharm:+.3f} q² on "
              f"average ({win_p} of {len(data)} sets better) and the Sybyl indicators "
              f"{d_cif:+.3f} ({win_c} of {len(data)}).** Alone, the six reach "
              f"{means['pharm']:+.3f} against {means['cif']:+.3f} for the Sybyl ones and "
              f"{means['five']:+.3f} for the five physico-chemical fields.", ""]

    lines += ["## Why: what each vocabulary covers", "",
              "The share of the molecules of a set that carries each kind. This is the whole "
              "argument of the item, and it is visible before any model is fitted.", "",
              "| set | " + " | ".join(PHARM_FEATURES) + " | Sybyl types, all | median "
              "coverage | types kept | median coverage of those |",
              "|---|" + "---:|" * len(PHARM_FEATURES) + "---:|---:|---:|---:|"]
    for d in data:
        cov = d["coverage"]
        cells = [f"{cov[p]['molecules']:.0%}" if p in cov else "—" for p in PHARM_FEATURES]
        allc = list(d.get("cif_coverage_all", {}).values())
        keptc = list(d["cif_coverage"].values())
        lines.append(f"| {d['set']} | " + " | ".join(cells)
                     + f" | {d.get('n_cif_types_all', len(allc))} | "
                     + (f"{float(np.median(allc)):.0%}" if allc else "—")
                     + f" | {d['n_cif_types']} | "
                     + (f"{float(np.median(keptc)):.0%}" if keptc else "—") + " |")
    med_all = float(np.median([v for d in data for v in d.get("cif_coverage_all", {}).values()]))
    med_kept = float(np.median([v for d in data for v in d["cif_coverage"].values()]))
    mean_cov = float(np.mean([d["coverage"][p]["molecules"] for d in data
                              for p in PHARM_FEATURES if p in d["coverage"]]))
    n_all = float(np.mean([d.get("n_cif_types_all", 0) for d in data]))
    n_kept = float(np.mean([d["n_cif_types"] for d in data]))
    lines += ["",
              f"A pharmacophore feature is carried by {mean_cov:.0%} of the molecules of a set on "
              f"average, and the median Sybyl type by {med_all:.0%} -- which is the difference the "
              f"item was built on. But the Sybyl vocabulary the program actually uses is already "
              f"pruned: `cif_types_in` keeps only the types in five molecules or more, which on "
              f"these sets is {n_kept:.0f} of {n_all:.0f} types, and the median coverage of what "
              f"is kept is {med_kept:.0%}. **The rarity the item blames was already being "
              f"filtered out**, so whatever the six features gain is not that.", ""]
    text = "\n".join(lines) + "\n"
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=None)
    ap.add_argument("--mittal", action="store_true",
                    help="run on examples/mittal, 38 sets of which 15 are below forty molecules, "
                         "which is the size the item is about")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    results = MITTAL_RESULTS if a.mittal else RESULTS
    data = T.MITTAL if a.mittal else T.DATA
    if a.report:
        print(report(results))
        return
    names = a.set
    if names is None:
        names = ([os.path.splitext(os.path.basename(p))[0]
                  for p in sorted(glob.glob(os.path.join(T.MITTAL, "*.sdf")))]
                 if a.mittal else list(T.SETS))
    for name in names:
        run_set(name, results=results, force=a.force, data=data)
    print()
    print(report(results))


if __name__ == "__main__":
    main()
