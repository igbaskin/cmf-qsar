#!/usr/bin/env python
"""
What an alignment is worth, measured by replacing a good one with an automatic one.

Nine targets occur twice in ``examples/``:

  * ``examples/sutherland/<SET>.sdf`` and ``examples/steroids_cbg.sdf`` carry the conformations
    and the superposition of Sutherland, O'Brien & Weaver (J. Med. Chem. 47 (2004) 5541) -- hand
    curated, MMFF94s-minimised, as distributed;
  * ``examples/mittal/<SET>.sdf`` carries the same compounds from the 2D supplementary data of
    Mittal, McKinnon & Sorich (J. Chem. Inf. Model. 49 (2009) 1810), with one RDKit ETKDG
    conformer each and a maximum-common-substructure superposition onto a reference molecule
    chosen by rule.  The SDF records keep ``align_mcs_atoms`` and ``align_rmsd``, so how well
    each molecule actually fitted is on record.

Same targets, same activities, same fields and grids: the difference between the two is the
alignment and the conformer behind it.  Three models per file separate "the frame is worse" from
"the geometry is worse":

  aligned      the ordinary CMF model, which uses the frame
  invariant    the rotation-invariant field autocorrelation, which uses the conformer but no frame
  counts       atom-type counts, which use neither

Usage
-----
    python benchmarks/run_alignment.py [--sets ACE,THR] [--force]
    python benchmarks/run_alignment.py --report benchmarks/results/alignment/table.md
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

from rdkit import RDLogger  # noqa: E402

from cmfqsar.chemotypes import count_kernel  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               feature_matrix)
from cmfqsar.model import AtomTable, CMFModel, cv_predict, r2_score, rmse  # noqa: E402

sys.path.insert(0, HERE)
import kernel_cache as KC  # noqa: E402

RESULTS = os.path.join(HERE, "results", "alignment")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
SEED = 0

#: target: (published file, Mittal file).  The names differ in three places.
PAIRS = {
    "ACE": ("sutherland/ACE.sdf", "mittal/ACE.sdf"),
    "AChE": ("sutherland/AChE.sdf", "mittal/ACHE.sdf"),
    "BZR": ("sutherland/BZR.sdf", "mittal/BZR.sdf"),
    "COX2": ("sutherland/COX2.sdf", "mittal/COX2.sdf"),
    "DHFR": ("sutherland/DHFR.sdf", "mittal/DHFR.sdf"),
    "GPB": ("sutherland/GPB.sdf", "mittal/GPB.sdf"),
    "THER": ("sutherland/THER.sdf", "mittal/THERM.sdf"),
    "THR": ("sutherland/THR.sdf", "mittal/THR.sdf"),
    "STEROIDS": ("steroids_cbg.sdf", "mittal/STEROIDS.sdf"),
}


def best_cv(kernels, y, params):
    """Best (q2, RMSEcv, key, lambda) over a dict of kernels and the lambda grid."""
    best = None
    for key, K in kernels.items():
        for lam in LAMS:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=FOLDS, seed=SEED)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (q2, rmse(y, pred), key, float(lam))
    return best


def measure(path, tag):
    """The three models on one file."""
    ds = load_dataset(os.path.join(ROOT, "examples", path))
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    rmsds = [float(m.GetProp("align_rmsd")) for m in mols
             if m.HasProp("align_rmsd") and m.GetProp("align_rmsd") not in ("", "nan")]
    out = {"file": path, "n": len(mols),
           "align_rmsd_median": float(np.median(rmsds)) if rmsds else None}
    aligned, invariant = {}, {}
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        aligned[alpha] = KC.joint_training_kernel(model, AtomTable.from_molecules(mf))
        spec = AutocorrelationSpec()
        invariant[alpha] = autocorrelation_kernel(feature_matrix(mf, spec), n_bins=spec.n_bins)
        print(f"      alpha {alpha:g} done", flush=True)
    for name, kernels in (("aligned", aligned), ("invariant", invariant)):
        q2, err, alpha, lam = best_cv(kernels, y, None)
        out[name] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam}
        print(f"    {tag:9s} {name:9s} q2 {q2:+.3f}  RMSEcv {err:.3f}  "
              f"(alpha {alpha:g}, lam {lam:g})", flush=True)
    q2, err, _, lam = best_cv({"counts": count_kernel(mols, names)}, y, None)
    out["counts"] = {"q2": q2, "rmse_cv": err, "lam": lam}
    print(f"    {tag:9s} {'counts':9s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (lam {lam:g})", flush=True)
    return out


def run_set(name, force=False):
    out = os.path.join(RESULTS, f"{name}.json")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return json.load(open(out))
    t0 = time.time()
    published, automatic = PAIRS[name]
    print(f"\n=== {name}", flush=True)
    res = {"set": name, "published": measure(published, "published"),
           "automatic": measure(automatic, "automatic"),
           "fields": FIELDS, "alphas": list(ALPHAS), "lams": list(LAMS), "folds": FOLDS}
    res["seconds"] = time.time() - t0
    os.makedirs(RESULTS, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out}  ({res['seconds']:.0f} s)", flush=True)
    return res


def report(path):
    rows = [json.load(open(os.path.join(RESULTS, f"{n}.json"))) for n in PAIRS
            if os.path.exists(os.path.join(RESULTS, f"{n}.json"))]
    if not rows:
        raise SystemExit("no results yet")
    out = ["# A curated alignment against an automatic one", "",
           "The same nine targets twice: the conformations and superposition distributed with",
           "Sutherland et al. (2004), and the same compounds built from 2D by RDKit ETKDG and",
           "superimposed on a reference molecule by maximum common substructure for the Mittal",
           f"et al. (2009) collection. Same fields ({', '.join(rows[0]['fields'])}), same alpha",
           f"grid, same lambda grid, same {rows[0]['folds']}-fold CV.", "",
           "| target | n | q2 published | q2 automatic | median fit RMSD | gain from geometry "
           "(published) | gain from geometry (automatic) |",
           "|---|---|---|---|---|---|---|"]
    for r in rows:
        p, a = r["published"], r["automatic"]
        out.append(f"| {r['set']} | {p['n']}/{a['n']} | {p['aligned']['q2']:+.3f} | "
                   f"{a['aligned']['q2']:+.3f} | "
                   f"{'%.2f A' % a['align_rmsd_median'] if a['align_rmsd_median'] else '-'} | "
                   f"{p['aligned']['q2'] - p['counts']['q2']:+.3f} | "
                   f"{a['aligned']['q2'] - a['counts']['q2']:+.3f} |")
    out += ["", "Mean q2: published %+.3f, automatic %+.3f (difference %+.3f)." % (
        np.mean([r["published"]["aligned"]["q2"] for r in rows]),
        np.mean([r["automatic"]["aligned"]["q2"] for r in rows]),
        np.mean([r["published"]["aligned"]["q2"] - r["automatic"]["aligned"]["q2"]
                 for r in rows])), "",
        "The frame against the conformer: the invariant column uses the same coordinates without",
        "any superposition, so where it keeps its q2 and the aligned column does not, the frame is",
        "what went wrong rather than the geometry.", "",
        "| target | aligned pub. | invariant pub. | counts pub. | aligned aut. | invariant aut. "
        "| counts aut. |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        p, a = r["published"], r["automatic"]
        out.append(f"| {r['set']} | {p['aligned']['q2']:+.3f} | {p['invariant']['q2']:+.3f} | "
                   f"{p['counts']['q2']:+.3f} | {a['aligned']['q2']:+.3f} | "
                   f"{a['invariant']['q2']:+.3f} | {a['counts']['q2']:+.3f} |")
    text = "\n".join(out) + "\n"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(PAIRS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", default=None, metavar="PATH")
    args = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if args.report:
        report(args.report)
        return
    for name in [s for s in args.sets.split(",") if s]:
        run_set(name, force=args.force)


if __name__ == "__main__":
    main()
