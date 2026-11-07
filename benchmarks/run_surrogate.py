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

    files = sorted(f for f in glob.glob(os.path.join(results, "*.json"))
                   if not os.path.basename(f).startswith("litraj"))
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


HOPS_RHO = 0.25         # the envelope width the LiTraj sample does best at
HOPS_R_MAX = 6.0
HOPS_FOLDS = 10
HOPS_INNER = 5
HOPS_GAMMAS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def run_hops(max_structures: int = 0, seed: int = 0, force: bool = False,
             results: str = RESULTS) -> dict:
    """The same three questions on the LiTraj hop sample, with folds that hold out structures.

    This is the multiscale case the module is for: the sample is a hop of a lithium ion in a
    crystal, the label is a barrier from a bond-valence calculation (the fine level here), and
    the model is asked which hops it may answer and which must go back to that calculation.
    The description is the one the LiTraj benchmark found best among the linear kernels, the
    rotation-invariant local kernel of the environment within 6 A of the hop midpoint, and its
    Gaussian form.  Inside every outer fold the ridge (and the width) are chosen on an inner split
    that also holds out structures, as the nested arms of ``run_litraj.py`` choose them, and the
    calibration on the folds is measured on that inner split, so what it says is what a user who
    had only the training part would have seen.
    """
    from cmfqsar.localenv import LAMS, cross_validate_lams, local_features, local_kernel
    from cmfqsar.surrogate import Surrogate, cross_validate, tolerance_for_budget
    from run_litraj import PREPARED, fieldset_for, gaussian_kernel, load_prepared

    os.makedirs(results, exist_ok=True)
    tag = f"_{max_structures}" if max_structures else ""
    out_path = os.path.join(results, f"litraj{tag}.json")
    if os.path.exists(out_path) and not force:
        print("litraj: already done")
        return json.load(open(out_path))
    t0 = time.time()
    hs, doc = load_prepared(PREPARED, max_structures)
    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    t = hs.targets()
    y = np.asarray(t.y, float)
    print(f"=== LiTraj sample: {len(t)} hops in {t.n_measured_molecules} structures, barriers "
          f"mean {y.mean():.3f}, sd {y.std(ddof=1):.3f} eV", flush=True)
    envs = hs.environments(crystals, rows=t.atom, rho=HOPS_RHO, r_max=HOPS_R_MAX)
    K_lin = local_kernel(local_features(envs))
    print(f"  kernel built in {time.time() - t0:.0f} s", flush=True)
    arms = {"linear": {"linear": K_lin},
            "Gaussian": {f"gamma {g:g}": gaussian_kernel(K_lin, g) for g in HOPS_GAMMAS}}
    parts = t.folds(HOPS_FOLDS, seed)
    n = len(y)
    out = {"set": "litraj", "n": n, "n_structures": int(t.n_measured_molecules),
           "y_sd": float(y.std(ddof=1)), "rho": HOPS_RHO, "r_max": HOPS_R_MAX,
           "folds": HOPS_FOLDS, "inner": HOPS_INNER, "budget": BUDGET, "arms": {}}
    for label, ks in arms.items():
        pred = np.full(n, np.nan)
        sd_te = np.full(n, np.nan)
        sd_corr = np.full(n, np.nan)
        claimed = np.full(n, np.nan)
        fold_rows = []
        for k, te in enumerate(parts):
            te = np.asarray(te, int)
            tr = np.setdiff1d(np.arange(n), te)
            inside = t.subset(tr)
            best = None
            for key, K in ks.items():
                for r in cross_validate_lams(K[np.ix_(tr, tr)], inside, LAMS, folds=HOPS_INNER,
                                             seed=seed):
                    if best is None or r.q2 > best[0].q2:
                        best = (r, key)
            r, key = best
            K = ks[key]
            inner_parts = inside.folds(HOPS_INNER, seed)
            m_in, s_in = cross_validate(K[np.ix_(tr, tr)], y[tr], float(r.lam), parts=inner_parts)
            cal_in = Calibration.of(y[tr], m_in, s_in)
            s = Surrogate.fit(K[np.ix_(tr, tr)], y[tr], float(r.lam))
            pred[te], sd_te[te] = s.predict(K[np.ix_(te, tr)], np.diag(K)[te])
            sd_corr[te] = sd_te[te] * cal_in.factor
            claimed[te] = cal_in.rmse
            held = Calibration.of(y[te], pred[te], sd_te[te])
            fold_rows.append({"fold": k, "setting": key, "lam": float(r.lam),
                              "n_test": int(len(te)), "inner": dict(vars(cal_in)),
                              "test": dict(vars(held)), "noise_sd": s.noise_sd})
            print(f"  {label} fold {k}: {key}, lambda {r.lam:g}; inner RMSE {cal_in.rmse:.3f} "
                  f"z rms {cal_in.z_rms:.2f}; held-out RMSE {held.rmse:.3f} mean sd "
                  f"{held.mean_sd:.3f} z rms {held.z_rms:.2f}", flush=True)
        held = Calibration.of(y, pred, sd_te)
        corrected = Calibration.of(y, pred, sd_corr)
        rule = Handoff.of(sd_te, float(np.nanmean(claimed)), y=y, mean=pred)
        budget = Handoff.of(sd_te, tolerance_for_budget(sd_te, BUDGET), y=y, mean=pred)
        # the rule once more with the claimed error of each fold's own training part
        own = sd_te > claimed
        rmse = lambda m: float(np.sqrt(np.mean((y[m] - pred[m]) ** 2))) if m.any() else float("nan")
        out["arms"][label] = {
            "folds": fold_rows, "test": dict(vars(held)), "corrected": dict(vars(corrected)),
            "q2": float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
            "rule": dict(vars(rule)), "budget": dict(vars(budget)),
            "rule_per_fold": {"answered": float(np.mean(~own)), "rmse_answered": rmse(~own),
                              "rmse_deferred": rmse(own)}}
        print(f"  {label}: q2 {out['arms'][label]['q2']:+.3f}, {held.describe()}", flush=True)
        print(f"  {label}: {rule.describe()}", flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"  {time.time() - t0:.0f} s")
    return out


def report_hops(results: str = RESULTS) -> str:
    path = os.path.join(results, "litraj.json")
    if not os.path.exists(path):
        return ""
    d = json.load(open(path))
    lines = ["", "## The hop sample of LiTraj: a barrier as the activity, a structure as the "
             "fold", "",
             f"{d['n']} hops in {d['n_structures']} structures of the nebBVSE122k sample "
             f"(`benchmarks/run_litraj.py`), barriers with a standard deviation of "
             f"{d['y_sd']:.2f} eV; {d['folds']} folds that hold out whole structures, the ridge "
             f"and the width chosen on an inner split of {d['inner']} folds of the same kind. "
             "The fine level is the bond-valence calculation that labelled the hops.", "",
             "| kernel | q2 | held-out RMSE (eV) | mean sd (eV) | z rms, inner folds | "
             "z rms, held out | z rms, corrected | inside 2 sd | sd vs error (rho) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, a in d["arms"].items():
        zf = float(np.mean([f["inner"]["z_rms"] for f in a["folds"]]))
        lines.append(f"| {label} | {a['q2']:+.3f} | {a['test']['rmse']:.3f} | "
                     f"{a['test']['mean_sd']:.3f} | {zf:.2f} | {a['test']['z_rms']:.2f} | "
                     f"{a['corrected']['z_rms']:.2f} | {a['test']['coverage_2sd']:.0%} | "
                     f"{a['test']['spearman']:+.2f} |")
    lines += ["", "The tolerance rule with the tolerance at the inner-fold RMSE of the training "
              "part (the error the model claims), and with a fixed "
              f"{d['budget']:.0%} of the hops sent back to the calculation.", "",
              "| kernel | tolerance (eV) | answered | RMSE answered | RMSE deferred | "
              f"RMSE answered at {d['budget']:.0%} deferred | RMSE of that {d['budget']:.0%} |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, a in d["arms"].items():
        r, b = a["rule"], a["budget"]
        ans = (r["n"] - r["n_deferred"]) / r["n"]
        lines.append(f"| {label} | {r['tolerance']:.3f} | {ans:.0%} | {r['rmse_answered']:.3f} | "
                     f"{r['rmse_deferred']:.3f} | {b['rmse_answered']:.3f} | "
                     f"{b['rmse_deferred']:.3f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--hops", action="store_true",
                    help="the LiTraj hop sample instead of the Sutherland sets")
    ap.add_argument("--max-structures", type=int, default=0,
                    help="with --hops, a quick look on this many structures, under its own file")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.hops and not args.report:
        run_hops(max_structures=args.max_structures, force=args.force)
    elif not args.report:
        for name in args.sets:
            run_set(name, repeats=args.repeats, force=args.force)
    text = report() + "\n" + report_hops()
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
