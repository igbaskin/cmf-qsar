#!/usr/bin/env python
"""
Does the predictive standard deviation mean what it says, and what does a tolerance rule do? (MS1)

A multiscale scheme uses the model as its fast level and needs, per structure, a standard
deviation in the units of the activity and a rule that sends the doubtful structures back to the
fine level.  ``cmfqsar/surrogate.py`` gives both.  This measures them on the eight Sutherland
sets, five random 70/30 splits each, with alpha and lambda chosen by the evidence on the training
part, as ``run_errorpred.py`` chooses them.

Three questions, per split.

1. **Is the deviation calibrated on the folds?**  The root mean square of the standardised
   out-of-fold residual of the training part, which should be one.  This is what a user sees
   before any test set exists (``cmf build --calibration``).

2. **Does it stay calibrated on structures the model never saw?**  The same statistic on the
   held-out 30 %, as it stands and after multiplication by the factor the folds gave.  A factor
   that transfers from the folds to new structures is one that can be applied in advance.

3. **What does the tolerance rule do?**  The tolerance is set to the cross-validated RMSE of the
   training part, which is the error the model claims for a typical structure.  The share of
   the held-out structures the model answers, the RMSE of that share and the RMSE of the share
   it defers are reported; the rule works if the answered share's error is at or under the
   tolerance and the deferred share's error is above it.

Usage
-----
    python benchmarks/run_surrogate.py
    python benchmarks/run_surrogate.py --sets ACE GPB --repeats 3
    python benchmarks/run_surrogate.py --report
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import r2_score  # noqa: E402
from cmfqsar.surrogate import Calibration, Handoff  # noqa: E402
from run_errorpred import DATA, FIELDS, SETS, _choose, _kernels, _model  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "surrogate")
BUDGET = 0.25           # the share of structures a fine level is assumed to afford


def run_set(name, repeats=5, force=False, data=DATA, results=RESULTS):
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
    print(f"=== {name}: {n} molecules", flush=True)
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="file")
    mf = [build_fields(m, fs, nm) for m, nm in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    cache = _kernels(table)

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(rep)
        order = rng.permutation(n)
        n_te = max(5, int(round(0.3 * n)))
        te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])
        alpha, lam = _choose(cache, y[tr], tr)

        model = _model(alpha)
        model.params["lam"] = lam
        train = AtomTable.from_molecules([mf[i] for i in tr])
        Kf, sd = cache[alpha]
        model.set_training_kernels(train, Kf[:, tr][:, :, tr],
                                   None if sd is None else sd[:, tr])
        model.fit(train, y[tr], cv_folds=10, cv_seed=rep, reuse_kernels=True)
        folds = model.calibration(folds=10, seed=rep)
        rmse_cv = float(model.stats.rmse_cv)

        test = AtomTable.from_molecules([mf[i] for i in te])
        pred, sd_te = model.predict_with_sd(test)
        held = Calibration.of(y[te], pred, sd_te)
        corrected = Calibration.of(y[te], pred, sd_te * folds.factor)
        rule = Handoff.of(sd_te, rmse_cv, y=y[te], mean=pred)
        from cmfqsar.surrogate import tolerance_for_budget
        budget = Handoff.of(sd_te, tolerance_for_budget(sd_te, BUDGET), y=y[te], mean=pred)
        row = {"repeat": rep, "alpha": alpha, "lam": lam, "n_train": int(len(tr)),
               "n_test": int(len(te)),
               "r2": float(r2_score(y[te], pred)), "rmse_cv": rmse_cv,
               "noise_sd": float(model.surrogate().noise_sd),
               "folds": dict(vars(folds)), "test": dict(vars(held)),
               "corrected": dict(vars(corrected)),
               "rule": dict(vars(rule)), "budget": dict(vars(budget))}
        rows.append(row)
        print(f"  split {rep}: alpha {alpha:g} lambda {lam:g}  test RMSE {held.rmse:.3f}  "
              f"mean sd {held.mean_sd:.3f}  z rms folds {folds.z_rms:.2f} test {held.z_rms:.2f} "
              f"corrected {corrected.z_rms:.2f}  2 sd {held.coverage_2sd:.0%}  "
              f"rule: answers {rule.fraction_answered:.0%} at {rule.rmse_answered:.3f}, "
              f"defers at {rule.rmse_deferred:.3f}", flush=True)
    out = {"set": name, "n": n, "repeats": repeats, "budget": BUDGET, "rows": rows,
           "seconds": time.time() - t0}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"  {time.time() - t0:.0f} s")
    return out


def _mean(rows, *keys):
    vals = []
    for r in rows:
        v = r
        for k in keys:
            v = v[k]
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No surrogate results yet; run `python benchmarks/run_surrogate.py`."
    data = [json.load(open(f)) for f in files]
    data.sort(key=lambda d: -d["n"])
    lines = ["# The predictive standard deviation as a fast level sees it", "",
             f"{len(data)} Sutherland sets, {data[0]['repeats']} random 70/30 splits each; alpha "
             "and lambda by the evidence on the training part. z rms is the root mean square of "
             "(measured - predicted) / sd, one for a calibrated deviation and above one for a "
             "model surer than it should be; 'corrected' multiplies the test deviations by the "
             "factor the folds of the training part gave.", "",
             "| set | n | test RMSE | mean sd | z rms, folds | z rms, test | z rms, corrected | "
             "inside 2 sd | sd vs error (rho) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    acc = {k: [] for k in ("rmse", "sd", "zf", "zt", "zc", "c2", "rho")}
    for d in data:
        r = d["rows"]
        vals = {"rmse": _mean(r, "test", "rmse"), "sd": _mean(r, "test", "mean_sd"),
                "zf": _mean(r, "folds", "z_rms"), "zt": _mean(r, "test", "z_rms"),
                "zc": _mean(r, "corrected", "z_rms"), "c2": _mean(r, "test", "coverage_2sd"),
                "rho": _mean(r, "test", "spearman")}
        for k, v in vals.items():
            acc[k].append(v)
        lines.append(f"| {d['set']} | {d['n']} | {vals['rmse']:.3f} | {vals['sd']:.3f} | "
                     f"{vals['zf']:.2f} | {vals['zt']:.2f} | {vals['zc']:.2f} | "
                     f"{vals['c2']:.0%} | {vals['rho']:+.2f} |")
    m = {k: float(np.nanmean(v)) for k, v in acc.items()}
    lines.append(f"| **mean** |  | {m['rmse']:.3f} | {m['sd']:.3f} | **{m['zf']:.2f}** | "
                 f"**{m['zt']:.2f}** | **{m['zc']:.2f}** | {m['c2']:.0%} | {m['rho']:+.2f} |")
    lines += ["", "The tolerance rule, with the tolerance set to the cross-validated RMSE of the "
              "training part (what the model claims as its typical error): the share of the "
              "held-out structures the model answers and the RMSE on each side.  The last "
              f"columns send a fixed {data[0]['budget']:.0%} of the structures to the fine level "
              "instead.", "",
              "| set | tolerance | answered | RMSE answered | RMSE deferred | "
              f"RMSE answered at {data[0]['budget']:.0%} deferred | RMSE of that {data[0]['budget']:.0%} |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    acc2 = {k: [] for k in ("tol", "ans", "ra", "rd", "ba", "bd")}
    for d in data:
        r = d["rows"]
        vals = {"tol": _mean(r, "rmse_cv"),
                "ans": float(np.mean([x["rule"]["n"] and (x["rule"]["n"] - x["rule"]["n_deferred"]) / x["rule"]["n"] for x in r])),
                "ra": _mean(r, "rule", "rmse_answered"), "rd": _mean(r, "rule", "rmse_deferred"),
                "ba": _mean(r, "budget", "rmse_answered"), "bd": _mean(r, "budget", "rmse_deferred")}
        for k, v in vals.items():
            acc2[k].append(v)
        lines.append(f"| {d['set']} | {vals['tol']:.3f} | {vals['ans']:.0%} | {vals['ra']:.3f} | "
                     f"{vals['rd']:.3f} | {vals['ba']:.3f} | {vals['bd']:.3f} |")
    m2 = {k: float(np.nanmean(v)) for k, v in acc2.items()}
    lines.append(f"| **mean** | {m2['tol']:.3f} | {m2['ans']:.0%} | **{m2['ra']:.3f}** | "
                 f"**{m2['rd']:.3f}** | **{m2['ba']:.3f}** | **{m2['bd']:.3f}** |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if not args.report:
        for name in args.sets:
            run_set(name, repeats=args.repeats, force=args.force)
    text = report()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
