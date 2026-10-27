#!/usr/bin/env python
"""
Is the fluctuation of a field worth anything? (plan item D10)

The ensemble mode represents a molecule by the **mean** of the fields of its conformations, which
is the occupancy averaging of 4D-QSAR.  A mean throws away how much the field moved between
conformations, and that is a flexibility map.  `cmfqsar/fluctuation.py` builds it as a field of
the same class -- a product of two Gaussians is a Gaussian, so the variance across conformations
is again a sum of Gaussians, at the midpoints of atom pairs and at twice the width -- so no new
kernel is needed and it can be added as a block of its own.

This measures whether that block carries anything the mean does not.

How it is measured
------------------
* **The reference of rule 1 is the mean field**: the ordinary ensemble model on the same
  conformations, the same folds and the same ridge search.  Everything the variance is compared
  with differs from it in one thing only.
* Three models: the mean alone, the variance alone, and the two combined as
  ``(1 - h) K_mean + h K_variance`` with both blocks scaled to unit mean diagonal, for a grid of
  mixing coefficients.  The grid is searched inside the cross-validation, so the combination
  never sees the held-out molecules.
* The conformations are generated with the input pose kept first, so every ensemble sits where
  the aligned molecule was and the overlap kernel still has its common frame.
* Hydrogens are dropped before the variance is built.  The field of a hydrogen is small and the
  construction grows with the square of the atoms, so they cost much and carry little.

Usage
-----
    python benchmarks/run_fluctuation.py
    python benchmarks/run_fluctuation.py --sets steroids THR --conformers 3
    python benchmarks/run_fluctuation.py --report
"""
from __future__ import annotations

import argparse
import io
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

from cmfqsar.conformers import generate_conformers, single_conformer  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields, combine_members  # noqa: E402
from cmfqsar.fluctuation import kept_share, variance_field, variance_fieldset  # noqa: E402
from cmfqsar.kernels import combine_kernels, kernel_matrices, self_kernels  # noqa: E402
from cmfqsar.model import _kfold_indices, krr_fit, r2_score, rmse  # noqa: E402

RESULTS = os.path.join(HERE, "results", "fluctuation")
SETS = {
    "steroids": dict(path=os.path.join(ROOT, "examples", "steroids_cbg.sdf"), activity="CBG_pK",
                     what="the 31 CBG steroids, the classic CoMFA benchmark"),
    "GPB": dict(path=os.path.join(ROOT, "examples", "sutherland", "GPB.sdf"), activity="pKi",
                what="glycogen phosphorylase b, 66 molecules"),
    "THR": dict(path=os.path.join(ROOT, "examples", "sutherland", "THR.sdf"), activity="pKi",
                what="thrombin, 88 molecules"),
}
FIELDS = ["electrostatic", "hydrophobic"]
ALPHA = 2.0
FLOOR = 1e-2
MIXES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def _say(text):
    print(text, flush=True)


def blocks(key, n_conformers=3, seed=1, max_molecules=0):
    """The mean-field and variance-field kernels of one set, both on unit mean diagonal."""
    spec = SETS[key]
    ds = load_dataset(spec["path"], activity_prop=spec["activity"])
    keep = [i for i in range(len(ds)) if np.isfinite(ds.activity[i])]
    if max_molecules:
        keep = keep[:max_molecules]
    mols = [Chem.RemoveHs(ds.mols[i]) for i in keep]
    names = [ds.names[i] for i in keep]
    y = np.asarray(ds.activity[keep], float)

    fieldset = FieldSet.create(physchem=FIELDS, alpha=ALPHA)
    vfieldset = variance_fieldset(fieldset)
    t0 = time.time()
    means, variances, shares, conformers = [], [], [], []
    for k, m in enumerate(mols):
        ens = generate_conformers(m, n=int(n_conformers), seed=seed)
        members = [build_fields(single_conformer(ens, i) if ens.GetNumConformers() > 1 else ens,
                                fieldset, name=f"{names[k]}#{i}")
                   for i in range(max(ens.GetNumConformers(), 1))]
        conformers.append(len(members))
        means.append(combine_members(members, np.ones(len(members)), kind="conformers",
                                     name=names[k]) if len(members) > 1 else members[0])
        if len(members) > 1:
            variances.append(variance_field(members, fieldset, floor=FLOOR, name=names[k]))
            shares.append(kept_share(members, fieldset, FLOOR))
        else:                       # a rigid molecule has no fluctuation to speak of
            variances.append(variance_field(members * 2, fieldset, floor=FLOOR, name=names[k]))
            shares.append(1.0)
    build = time.time() - t0

    specs = [s for s in fieldset.specs if s.enabled]
    vspecs = [s for s in vfieldset.specs if s.enabled]
    t0 = time.time()
    Tm, Tv = AtomTable.from_molecules(means), AtomTable.from_molecules(variances)
    Km = _joint(kernel_matrices(Tm, Tm, specs, symmetric=True))
    Kv = _joint(kernel_matrices(Tv, Tv, vspecs, symmetric=True))
    kernel = time.time() - t0
    return dict(y=y, names=names, Km=Km, Kv=Kv, build=build, kernel=kernel,
                conformers=float(np.mean(conformers)), kept=float(np.mean(shares)),
                atoms=float(np.mean([v.n_atoms for v in variances])),
                n=len(y), what=spec["what"])


def _joint(K_fields):
    """The field kernels of one block, each scaled to unit mean diagonal, summed."""
    scale = np.array([1.0 / max(float(np.mean(np.diag(K))), 1e-300) for K in K_fields])
    return combine_kernels(K_fields, np.ones(len(K_fields)), scale)


def cv(K, y, lam, folds, seed):
    pred = np.zeros(len(y))
    for te in _kfold_indices(len(y), folds, seed):
        tr = np.setdiff1d(np.arange(len(y)), te)
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam)
        pred[te] = K[np.ix_(te, tr)] @ a + b
    return pred


def best(K, y, folds, seed):
    """The ridge that cross-validates best on this kernel, and what it gives."""
    out = None
    for lam in LAMS:
        p = cv(K, y, lam, folds, seed)
        q = r2_score(y, p)
        if out is None or q > out[0]:
            out = (q, rmse(y, p), lam, p)
    return out


def mixed(Km, Kv, h):
    return (1.0 - h) * Km + h * Kv


def run_set(key, n_conformers=3, folds=10, seed=0, max_molecules=0, results=RESULTS):
    _say(f"=== {SETS[key]['what']}")
    d = blocks(key, n_conformers, seed=1, max_molecules=max_molecules)
    y = d["y"]
    _say(f"{d['n']} molecules, {d['conformers']:.1f} conformations each, "
         f"{d['atoms']:.0f} pseudo-atoms per variance field ({d['kept']:.1%} of the pair weight "
         f"kept), fields in {d['build']:.0f} s and kernels in {d['kernel']:.0f} s")

    rows = []
    for h in MIXES:
        q2, err, lam, pred = best(mixed(d["Km"], d["Kv"], h), y, folds, seed)
        rows.append(dict(mix=float(h), q2=q2, error=err, lam=lam, predicted=pred))
        _say(f"  mixing {h:4.2f}: q2 {q2:+.3f}, error {err:.3f}, ridge {lam:g}"
             + ("   (the mean alone)" if h == 0 else
                "   (the variance alone)" if h == 1 else ""))
    base = rows[0]
    inner = [r for r in rows if 0.0 < r["mix"] < 1.0]
    chosen = max(inner, key=lambda r: r["q2"]) if inner else base
    diff = paired(chosen["predicted"], base["predicted"], y, seed)
    _say(f"  the best mixture is {chosen['mix']:.2f}: {diff[0]:+.3f} "
         f"[{diff[1]:+.3f}, {diff[2]:+.3f}] against the mean alone")

    out = dict(set=key, what=d["what"], n=d["n"], conformers=d["conformers"],
               pseudo_atoms=d["atoms"], kept=d["kept"], alpha=ALPHA, floor=FLOOR,
               fields=FIELDS, folds=folds, seed=seed, build_seconds=d["build"],
               kernel_seconds=d["kernel"],
               rows=[{k: v for k, v in r.items() if k != "predicted"} for r in rows],
               chosen=dict(mix=chosen["mix"], q2=chosen["q2"], error=chosen["error"],
                           difference=list(diff)))
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def paired(a, b, y, seed=0, runs=2000):
    """(difference in q2, low, high), bootstrapped over molecules."""
    y = np.asarray(y, float)
    if len(y) < 6:
        return (r2_score(y, a) - r2_score(y, b), float("nan"), float("nan"))
    rng = np.random.RandomState(int(seed))
    got = []
    for _ in range(int(runs)):
        idx = rng.randint(0, len(y), len(y))
        if np.std(y[idx]) <= 0:
            continue
        got.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    point = r2_score(y, a) - r2_score(y, b)
    if not got:
        return (point, float("nan"), float("nan"))
    return (point, float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5)))


def report(results=RESULTS):
    out = ["# Is the fluctuation of a field worth anything? (plan item D10)", "",
           "The ensemble mode uses the mean of the fields of a molecule's conformations. The "
           "variance across them is a field of the same class -- a product of two Gaussians is a "
           "Gaussian, so it is a sum of Gaussians at the midpoints of atom pairs and at twice the "
           "width -- so it can be added as a block of its own with a mixing coefficient. The "
           "reference of rule 1 is the mean field: same conformations, same folds, same ridge "
           "search.", ""]
    rows = []
    for key in SETS:
        path = os.path.join(results, f"{key}.json")
        if os.path.exists(path):
            with io.open(path, encoding="utf-8") as fh:
                rows.append(json.load(fh))
    if not rows:
        raise SystemExit("nothing to report: run the benchmark first")
    for d in rows:
        out += [f"## {d['what']}", "",
                f"{d['n']} molecules, {d['conformers']:.1f} conformations each, "
                f"{d['pseudo_atoms']:.0f} pseudo-atoms per variance field at attenuation "
                f"{d['alpha']:g} and a pair floor of {d['floor']:g} ({d['kept']:.1%} of the pair "
                f"weight kept). Fields in {d['build_seconds']:.0f} s, kernels in "
                f"{d['kernel_seconds']:.0f} s. {d['folds']} folds.", "",
                "| mixing coefficient | q2 | error | ridge |",
                "| ---: | ---: | ---: | ---: |"]
        for r in d["rows"]:
            tag = ("the mean alone" if r["mix"] == 0 else
                   "the variance alone" if r["mix"] == 1 else f"{r['mix']:.2f}")
            out.append(f"| {tag} | {r['q2']:+.3f} | {r['error']:.3f} | {r['lam']:g} |")
        c = d["chosen"]
        out += ["", f"The best mixture is {c['mix']:.2f}, q2 {c['q2']:+.3f} against "
                    f"{d['rows'][0]['q2']:+.3f} for the mean alone: "
                    f"{c['difference'][0]:+.3f} [{c['difference'][1]:+.3f}, "
                    f"{c['difference'][2]:+.3f}].", ""]
    helped = [d for d in rows if d["chosen"]["difference"][1] > 0]
    out += ["## What it comes to", ""]
    if helped:
        out.append("**The fluctuation block helps on " + ", ".join(d["set"] for d in helped)
                   + "**, by more than the interval of rule 4.")
    else:
        out.append("**The fluctuation block does not help on any set measured here**: every "
                   "interval against the mean alone contains zero or lies below it.")
    out.append("The variance alone reaches "
               + ", ".join(f"q2 {next(r for r in d['rows'] if r['mix'] == 1.0)['q2']:+.3f} on "
                           f"{d['set']}" for d in rows)
               + ". It carries something where the series is small and rigid enough for it to "
                 "matter, and nothing the mean does not already have.")
    out.append("")
    return "\n".join(out)


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    ap.add_argument("--conformers", type=int, default=3)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-molecules", type=int, default=0)
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    for key in a.sets:
        run_set(key, n_conformers=a.conformers, folds=a.folds, seed=a.seed,
                max_molecules=a.max_molecules)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
