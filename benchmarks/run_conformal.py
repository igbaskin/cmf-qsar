#!/usr/bin/env python
"""
Do the conformal intervals keep the coverage they promise?  Measured, not assumed.

Split conformal has an exact finite-sample guarantee and the proof needs no assumption about the
model, only that the calibration molecules and the new one are exchangeable.  The variant the
program actually offers is the cheap one: calibrate on the out-of-fold errors of the
cross-validation that is being run anyway, then predict with the model refitted on everything.
That breaks exchangeability -- the errors come from models fitted on n(1 - 1/k) molecules and the
predictions from a model fitted on n -- so the guarantee becomes an approximation, and the size of
the gap is an empirical question about sets of this size.

This script asks it.  For every Sutherland set, repeatedly:

1. split the molecules 70/30;
2. choose alpha and lambda on the training part by 10-fold cross-validation;
3. calibrate four rules on that part --

       cv          out-of-fold errors, the model then refitted on all of it
       cv-norm     the same, with each error divided by (beta + distance from the training set)
       split       a third of the training part held out, the model fitted on the other two
       split-norm  the same, normalised

4. predict the held-out 30 % and record whether the truth fell inside, and how wide;
5. beside them, the Gaussian process posterior interval at the same nominal level, which carries
   no guarantee at all and is what a user would otherwise use.

The numbers to read are the **coverage** against the nominal level, and the **width**, in the
units of the activity.  A rule that covers is useless if it covers by being as wide as the data,
so the width is reported beside the standard deviation of the activity.

Usage
-----
    python benchmarks/run_conformal.py
    python benchmarks/run_conformal.py --sets ACE DHFR --repeats 10
    python benchmarks/run_conformal.py --report
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

from cmfqsar.conformal import calibrate, coverage, difficulty_from_kernel  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, krr_fit  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "conformal")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0)
LEVELS = (0.8, 0.9, 0.95)
BETA = 0.1                      # the fixed part of a normalised width, in units of the error
RULES = ("cv", "cv-norm", "split", "split-norm", "gp")


def kernels(mols, names, alphas=ALPHAS):
    """One joint training kernel per attenuation, over every molecule of the set."""
    out = []
    for alpha in alphas:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="file")
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
        out.append((float(alpha), model.joint_kernel(model.prepare_training_kernels(table))))
    return out


def _choose(K, y, tr, folds=10, seed=0):
    """(lambda, q2) chosen inside the training part alone."""
    best = None
    Ktr = K[np.ix_(tr, tr)]
    for lam in LAMS:
        p = cv_predict(Ktr, y[tr], "krr", {"lam": float(lam)}, folds=folds, seed=seed)
        ss = float(np.sum((y[tr] - np.mean(y[tr])) ** 2))
        q2 = 1.0 - float(np.sum((y[tr] - p) ** 2)) / max(ss, 1e-12)
        if best is None or q2 > best[1]:
            best = (float(lam), q2)
    return best


def _predict(K, y, tr, te, lam):
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
    return K[np.ix_(te, tr)] @ a + b


def _gp_interval(K, y, tr, te, lam, level):
    """The posterior interval of a Gaussian process with the same kernel and ridge.

    Not a conformal rule and not a guarantee: the spread of a model that believes its own kernel.
    """
    from scipy.stats import norm

    Ktr = K[np.ix_(tr, tr)] + lam * np.eye(len(tr))
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
    mu = K[np.ix_(te, tr)] @ a + b
    v = np.linalg.solve(Ktr, K[np.ix_(tr, te)])
    var = np.diag(K[np.ix_(te, te)]) - np.sum(K[np.ix_(te, tr)].T * v, axis=0)
    sd = np.sqrt(np.maximum(var, 0.0) * float(np.var(y[tr])) + lam * float(np.var(y[tr])))
    z = float(norm.ppf(0.5 + 0.5 * level))
    return mu - z * sd, mu + z * sd


def run_set(name, repeats=10, force=False, data=DATA, results=RESULTS, beta=BETA):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    n = len(y)
    print(f"=== {name}: {n} molecules, activity sd {np.std(y):.2f}", flush=True)
    Ks = kernels(mols, names)

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(rep)
        order = rng.permutation(n)
        n_te = max(5, int(round(0.3 * n)))
        te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])
        # the attenuation is chosen inside the training part, like everything else here
        best = None
        for alpha, K in Ks:
            lam, q2 = _choose(K, y, tr)
            if best is None or q2 > best[2]:
                best = (K, lam, q2, alpha)
        K, lam, q2_tr, alpha = best

        yp_te = _predict(K, y, tr, te, lam)
        # the out-of-fold errors of the training part
        yp_oof = cv_predict(K[np.ix_(tr, tr)], y[tr], "krr", {"lam": lam}, folds=10, seed=rep)
        d_tr = difficulty_from_kernel(K[np.ix_(tr, tr)], np.diag(K)[tr], K[np.ix_(tr, tr)])
        d_te = difficulty_from_kernel(K[np.ix_(te, tr)], np.diag(K)[te], K[np.ix_(tr, tr)])
        # and a genuine held-out calibration third of it
        cut = max(5, int(round(len(tr) / 3)))
        cal, fit = tr[:cut], tr[cut:]
        yp_cal = _predict(K, y, fit, cal, lam)
        yp_te_split = _predict(K, y, fit, te, lam)
        d_cal = difficulty_from_kernel(K[np.ix_(cal, fit)], np.diag(K)[cal], K[np.ix_(fit, fit)])
        d_te_split = difficulty_from_kernel(K[np.ix_(te, fit)], np.diag(K)[te], K[np.ix_(fit, fit)])

        for level in LEVELS:
            rules = {
                "cv": (calibrate(y[tr], yp_oof, level), yp_te, None),
                "cv-norm": (calibrate(y[tr], yp_oof, level, difficulty=d_tr, beta=beta),
                            yp_te, d_te),
                "split": (calibrate(y[cal], yp_cal, level, kind="split"), yp_te_split, None),
                "split-norm": (calibrate(y[cal], yp_cal, level, difficulty=d_cal, beta=beta,
                                         kind="split"), yp_te_split, d_te_split),
            }
            for rule, (c, yp, d) in rules.items():
                lo, hi = c.interval(yp, d)
                got = coverage(y[te], lo, hi)
                rows.append({"repeat": rep, "level": level, "rule": rule,
                             "n_cal": c.n_calibration, "usable": bool(c.usable), **got})
            lo, hi = _gp_interval(K, y, tr, te, lam, level)
            got = coverage(y[te], lo, hi)
            rows.append({"repeat": rep, "level": level, "rule": "gp", "n_cal": len(tr),
                         "usable": True, **got})
        print(f"  repeat {rep}: alpha {alpha:g}, lambda {lam:g}, q2(train) {q2_tr:+.3f}",
              flush=True)

    out = {"set": name, "n": int(n), "activity_sd": float(np.std(y)), "beta": float(beta),
           "rows": rows,
           "repeats": int(repeats), "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"  {name}: {out['seconds']:.0f} s")
    return out


def report(results=RESULTS):
    """Coverage and width per rule and level, over every set."""
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No conformal results yet; run `python benchmarks/run_conformal.py`."
    data = [json.load(open(f)) for f in files]
    lines = ["# Conformal intervals on the Sutherland sets", "",
             f"{len(data)} sets, {data[0]['repeats']} random 70/30 splits each. Coverage is the "
             "share of held-out molecules inside the interval; width is in units of the activity, "
             "with the activity's own standard deviation beside it for scale.", ""]
    sd = float(np.mean([d["activity_sd"] for d in data]))
    lines += ["| level | rule | coverage | width | width / sd |", "|---|---|---:|---:|---:|"]
    for level in LEVELS:
        for rule in RULES:
            cov, wid = [], []
            for d in data:
                for r in d["rows"]:
                    if r["level"] == level and r["rule"] == rule and r["n"]:
                        cov.append(r["coverage"])
                        wid.append(r["mean_width"])
            if not cov:
                continue
            c, w = float(np.mean(cov)), float(np.mean(wid))
            flag = "" if c >= level - 0.02 else " **low**"
            lines.append(f"| {level:.0%} | `{rule}` | {c:.1%}{flag} | {w:.2f} | {w / sd:.2f} |")
    lines += ["", f"The activity's standard deviation averages {sd:.2f} over the sets.", "",
              "## By set, at 90 %", "",
              "| set | n | cv | cv-norm | split | split-norm | gp |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in data:
        cells = []
        for rule in RULES:
            cov = [r["coverage"] for r in d["rows"] if r["level"] == 0.9 and r["rule"] == rule]
            cells.append(f"{np.mean(cov):.0%}" if cov else "--")
        lines.append(f"| {d['set']} | {d['n']} | " + " | ".join(cells) + " |")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--beta", type=float, default=BETA,
                    help="the fixed part of a normalised width (default %(default)s)")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()
    if a.report:
        print(report(a.results))
        return
    for name in a.sets:
        run_set(name, repeats=a.repeats, force=a.force, results=a.results, beta=a.beta)
    print()
    print(report(a.results))


if __name__ == "__main__":
    main()
