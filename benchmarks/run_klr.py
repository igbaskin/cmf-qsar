#!/usr/bin/env python
"""
Kernel logistic regression against the other classifiers (plan item E9).

Kernel logistic regression (``--method klr``) is the probabilistic classifier that is linear in
the field: logistic loss on the latent function plus a ridge on its norm, minimised by Newton's
method.  That is the same problem the Laplace step of `cmfqsar/gpc.py` already solves, so the two
differ in exactly two things and in nothing else:

* the Gaussian process picks its amplitude from the **marginal likelihood**, which holds nothing
  out; this learner picks the equivalent ridge on the **folds**;
* the Gaussian process averages its probability over the **posterior variance**, which pulls a
  molecule unlike anything in the training set towards one half; this learner reports the
  likelihood at the mode, which does not.

So the comparison is a controlled one, and it answers a question the Gaussian-process measurement
could not: of the two things the Laplace classifier does, which one its probabilities owe their
quality to.

Protocol
--------
Exactly the protocol of plan item E8, imported from `run_kfda.py` rather than written again: the
two LIT-PCBA classification sets of plan item E4, one kernel per set (five physico-chemical
fields, attenuation 0.5, cosine-normalised), one 5-fold split stratified by class, and each
learner choosing its own setting inside every outer fold on a 5-fold cross-validation of that
fold's training molecules, scored by AUC.  Four learners: kernel logistic regression, the Laplace
Gaussian-process classifier, C-SVC and the kernel Fisher discriminant of E8.

The gate names **the Brier score and AUC**.  Balanced accuracy is reported too, under the common
threshold rule of E8, because E8 found that where a classifier leaves its decision boundary
dominates its balanced accuracy on sets with one active in twenty, and a comparison at each
learner's own boundary would measure that instead of the learner.

Every difference carries a 95 % paired bootstrap interval over the molecules.  Beside the Brier
score the report gives a reliability table -- predicted probability against the observed share of
actives, in bins -- because a Brier score on an imbalanced set is dominated by the base rate and
says little on its own; the share of actives predicted for every molecule is printed as the
reference both must beat.

Usage
-----
    python benchmarks/run_klr.py
    python benchmarks/run_klr.py --sets TP53 --force
    python benchmarks/run_klr.py --report
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

import run_kfda as RK  # noqa: E402
from cmfqsar.gpc import brier_score, calibration_table  # noqa: E402
from cmfqsar.model import classification_scores  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "klr")
SETS = RK.GATE_SETS
LAMS = RK.LAMS
CS = RK.CS
#: The learners and the settings each chooses from inside every fold.  The Gaussian-process
#: classifier is run with the **logistic** likelihood here, not with its own probit default,
#: because the plan entry's comparison rests on the two sharing the likelihood: with probit they
#: would differ in three things at once and nothing could be attributed to any of them.  It is run
#: twice, once choosing its amplitude on the folds and once from the marginal likelihood, so that
#: the two differences from kernel logistic regression are separated: averaging over the posterior
#: variance, and choosing the hyper-parameter without holding anything out.
LEARNERS = {
    "klr": ("kernel logistic regression (ridge by CV)", [{"lam": l} for l in LAMS]),
    "gpc": ("GP classifier, logistic likelihood (amplitude by CV)",
            [{"amplitude": c, "likelihood": "logit"} for c in CS]),
    "gpc-evidence": ("GP classifier, logistic likelihood (amplitude by the marginal likelihood)",
                     None),
    "svc": ("C-SVC (C by CV)", [{"C": c} for c in CS]),
    "kfda": ("kernel Fisher discriminant (ridge by CV)",
             [{"lam": l, "kfda_threshold": "cv"} for l in LAMS]),
}
ORDER = ("klr", "gpc", "gpc-evidence", "svc", "kfda")
PROBABILISTIC = ("klr", "gpc", "gpc-evidence", "kfda")
BINS = 5
#: the amplitudes the evidence chooses among, the same grid the cross-validated one sees
EVIDENCE_AMPLITUDES = CS


def cross_validate_evidence(K, y, amplitudes=EVIDENCE_AMPLITUDES, likelihood="logit",
                            folds=None, seed=None, inner=None):
    """The Gaussian-process classifier with its amplitude chosen by the marginal likelihood.

    Everything else is the protocol of `run_kfda.py`: the same outer folds, the same common
    threshold rule, the same probabilities.  Only the selection differs, and that is the point --
    the evidence holds nothing out, which is the property the plan entry says this learner has
    and kernel logistic regression does not.
    """
    from cmfqsar.gpc import select_by_evidence

    folds = RK.FOLDS if folds is None else folds
    seed = RK.SEED if seed is None else seed
    inner = RK.INNER_FOLDS if inner is None else inner
    y = np.asarray(y, float)
    n = len(y)
    decision = np.full(n, np.nan)
    tuned = np.full(n, np.nan)
    proba = np.full(n, np.nan)
    chosen = []
    for te in RK.stratified_folds(y, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        Ktr = K[np.ix_(tr, tr)]
        _, amplitude, _ = select_by_evidence({0: Ktr}, y[tr], amplitudes=amplitudes,
                                             likelihood=likelihood)
        params = {"amplitude": float(amplitude), "likelihood": likelihood}
        chosen.append({"amplitude": float(amplitude)})
        lr = RK._Learner("gpc", params).fit(Ktr, y[tr])
        Kte = K[np.ix_(te, tr)]
        decision[te] = lr.predict(Kte)
        raw_tr, raw_te = Ktr @ lr.alpha_, Kte @ lr.alpha_
        tuned[te] = raw_te - RK._common_cut(Ktr, y[tr], "gpc", params, raw_tr, inner, seed)
        proba[te] = RK.probabilities(lr, Kte, raw_te, np.diag(K)[te])
    has_proba = bool(np.isfinite(proba).all())
    return decision, tuned, (proba if has_proba else None), chosen


def measure_cost(K, y, amplitude=10.0, lam=0.1, reps=3):
    """What the plan entry's word "cheaper" is worth, on this kernel, in milliseconds.

    Two things are timed: one fit, where the saving is the factorisation the posterior variance
    is read from and the determinant the marginal likelihood needs, and one prediction of every
    molecule, where the saving is the triangular solve the variance costs per molecule.  Newton's
    method takes the same number of steps either way, so the fit can only gain a constant.
    """
    from cmfqsar.gpc import LaplaceGPC, klr_fit

    def timed(fn):
        fn()                                      # once, so the measurement is not the warm-up
        t0 = time.time()
        for _ in range(int(reps)):
            out = fn()
        return (time.time() - t0) / float(reps), out

    t_fit_klr, (a, _, fit) = timed(lambda: klr_fit(K, y, lam))
    t_fit_gpc, g = timed(lambda: LaplaceGPC(likelihood="logit", amplitude=amplitude).fit(K, y))
    diag = np.diag(K)
    t_pred_klr, _ = timed(lambda: fit.predict_proba(K))
    t_pred_gpc, _ = timed(lambda: g.predict_proba(K, diag))
    return {"n": int(len(y)), "lam": float(lam), "amplitude": float(amplitude),
            "fit_klr_ms": 1000.0 * t_fit_klr, "fit_gpc_ms": 1000.0 * t_fit_gpc,
            "fit_ratio": t_fit_gpc / max(t_fit_klr, 1e-12),
            "predict_klr_ms": 1000.0 * t_pred_klr, "predict_gpc_ms": 1000.0 * t_pred_gpc,
            "predict_ratio": t_pred_gpc / max(t_pred_klr, 1e-12),
            "iterations": int(fit.n_iter)}


def run_set(name, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    K, y, model, table = RK._litpcba_kernel(name)
    if K is None:
        return None
    n_a = int(np.sum(y > 0))
    print(f"=== {name}: {n_a} measured actives, {len(y) - n_a} measured inactives, "
          f"kernel {K.shape} in {time.time() - t0:.0f} s", flush=True)

    preds, tuned_preds, rows = {}, {}, {}
    for method in ORDER:
        label, grid = LEARNERS[method]
        t1 = time.time()
        if method == "gpc-evidence":
            d, t, p, chosen = cross_validate_evidence(K, y)
        else:
            d, t, p, chosen = RK.cross_validate(K, y, method, grid)
        preds[method] = (d, p)
        tuned_preds[method] = (t, p)
        sc, sct = classification_scores(y, d), classification_scores(y, t)
        rows[method] = {
            "auc": sc["auc"],
            "brier": brier_score(y, p) if p is not None else float("nan"),
            "balanced_accuracy": sc["balanced_accuracy"],
            "balanced_accuracy_common": sct["balanced_accuracy"],
            "sensitivity_common": sct["sensitivity"],
            "specificity_common": sct["specificity"],
            "calibration": ([list(r) for r in calibration_table(y, p, bins=BINS)]
                            if p is not None else []),
            "chosen": [{k: v for k, v in c.items()
                        if k not in ("kfda_threshold", "likelihood")} for c in chosen],
            "seconds": round(time.time() - t1, 1),
        }
        print(f"    {method:5s} AUC {sc['auc']:.3f}  Brier {rows[method]['brier']:.4f}  "
              f"balanced {sct['balanced_accuracy']:.3f} (common rule)  "
              f"({time.time() - t1:.0f} s)", flush=True)

    diffs = {}
    for other in ORDER[1:]:
        for metric in ("auc", "brier", "balanced_accuracy_common"):
            if metric == "brier" and other not in PROBABILISTIC:
                continue
            pair = tuned_preds if metric.endswith("_common") else preds
            v, lo, hi = RK.paired_interval(metric, y, pair["klr"], pair[other])
            diffs[f"klr-{other}/{metric}"] = [v, lo, hi]
            print(f"    klr - {other:4s} {metric:24s} {v:+.4f}  [{lo:+.4f}, {hi:+.4f}]",
                  flush=True)

    cost = measure_cost(K, y)
    print(f"    cost on this kernel: one fit {cost['fit_klr_ms']:.0f} ms against "
          f"{cost['fit_gpc_ms']:.0f} ms ({cost['fit_ratio']:.2f} times), one prediction of all "
          f"{len(y)} molecules {cost['predict_klr_ms']:.1f} ms against "
          f"{cost['predict_gpc_ms']:.1f} ms ({cost['predict_ratio']:.0f} times)", flush=True)

    share = n_a / float(len(y))
    base_brier = float(share * (1.0 - share))
    print(f"    the share of actives alone scores Brier {base_brier:.4f} "
          f"(share {share:.3f})", flush=True)

    out = {"set": name, "n": int(len(y)), "n_actives": n_a, "alpha": RK.ALPHA,
           "folds": RK.FOLDS, "inner_folds": RK.INNER_FOLDS, "seed": RK.SEED,
           "fields": RK.FIELDS, "bins": BINS, "methods": rows, "differences": diffs,
           "base_rate_brier": base_brier, "active_share": share, "cost": cost,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

def _load(results=RESULTS):
    out = {}
    for name in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        if name.endswith(".json"):
            d = json.load(open(os.path.join(results, name)))
            out[d["set"]] = d
    return out


def _pooled(data, key, metric):
    """Mean difference over the sets, with the mean of the per-set interval ends.

    Two sets are too few to resample as a population, so this says how large the per-set
    uncertainty is, not how well the two sets agree.
    """
    vals = [d["differences"].get(f"{key}/{metric}") for d in data.values()]
    vals = [v for v in vals if v and np.isfinite(v[0])]
    if not vals:
        return None
    return [float(np.mean([v[i] for v in vals])) for i in range(3)]


def report(path=None, results=RESULTS):
    data = _load(results)
    if not data:
        return "No results yet; run `python benchmarks/run_klr.py`.\n"
    first = next(iter(data.values()))
    L = ["# Kernel logistic regression against the other classifiers (plan item E9)", "",
         f"The protocol of plan item E8, imported from `run_kfda.py`: the two LIT-PCBA "
         f"classification sets of plan item E4, one kernel per set (five fields, attenuation "
         f"{first['alpha']:g}, cosine-normalised), one {first['folds']}-fold split stratified by "
         f"class, every learner choosing its own setting inside each fold on a "
         f"{first['inner_folds']}-fold cross-validation scored by AUC. Balanced accuracy is given "
         f"under E8's common threshold rule, because where a learner leaves its own boundary "
         f"dominates that statistic on these sets.", "",
         "| set | molecules | actives | learner | AUC | Brier score | balanced accuracy | "
         "sensitivity | specificity |",
         "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for name in SETS:
        d = data.get(name)
        if d is None:
            continue
        head = True
        for m in ORDER:
            r = d["methods"].get(m)
            if r is None:
                continue
            L.append(f"| {name if head else ''} | {d['n'] if head else ''} | "
                     f"{d['n_actives'] if head else ''} | {LEARNERS[m][0]} | "
                     f"{RK._f(r['auc'])} | {RK._f(r['brier'], 4)} | "
                     f"{RK._f(r['balanced_accuracy_common'])} | "
                     f"{RK._f(r['sensitivity_common'])} | {RK._f(r['specificity_common'])} |")
            head = False
        L.append(f"|  |  |  | *the share of actives, predicted for every molecule* | 0.500 | "
                 f"{RK._f(d.get('base_rate_brier'), 4)} | 0.500 | - | - |")

    keys = ("auc", "brier", "balanced_accuracy_common", "sensitivity_common",
            "specificity_common")
    L += ["", "Averaged over the sets:", "",
          "| learner | AUC | Brier score | balanced accuracy | sensitivity | specificity |",
          "|---|---:|---:|---:|---:|---:|"]
    for m in ORDER:
        vals = [data[n]["methods"][m] for n in SETS if n in data and m in data[n]["methods"]]
        if not vals:
            continue
        cells = []
        for k in keys:
            col = [float(v.get(k, np.nan)) for v in vals]
            col = [c for c in col if np.isfinite(c)]
            cells.append(RK._f(np.mean(col), 4 if k == "brier" else 3) if col else "-")
        L.append(f"| {LEARNERS[m][0]} | " + " | ".join(cells) + " |")
    base = [data[n].get("base_rate_brier") for n in SETS
            if n in data and data[n].get("base_rate_brier") is not None]
    if base:
        L.append(f"| *the share of actives, predicted for every molecule* | 0.500 | "
                 f"{RK._f(np.mean(base), 4)} | 0.500 | - | - |")

    L += ["", "### The differences, with 95 % paired bootstrap intervals", "",
          "The molecules are resampled once per draw and both learners scored on the same "
          "resample. A difference whose interval contains zero is not a difference. For the "
          "Brier score a negative difference is the better one, because it is an error.", "",
          "| comparison | metric | " + " | ".join(SETS) + " | mean |",
          "|---|---|" + "---|" * (len(SETS) + 1)]
    for other in ORDER[1:]:
        for metric, mlabel in (("auc", "AUC"), ("brier", "Brier score"),
                               ("balanced_accuracy_common", "balanced accuracy, common rule")):
            if metric == "brier" and other not in PROBABILISTIC:
                continue
            key = f"klr-{other}"
            nd = 4 if metric == "brier" else 3
            cells = [RK._interval(data[n]["differences"].get(f"{key}/{metric}"), nd)
                     if n in data else "-" for n in SETS]
            L.append(f"| klr - {other} | {mlabel} | " + " | ".join(cells)
                     + f" | {RK._interval(_pooled(data, key, metric), nd)} |")

    costs = [d["cost"] for d in data.values() if d.get("cost")]
    if costs:
        L += ["", "### What \"cheaper\" is worth, in milliseconds", "",
              "The plan entry says this learner is cheaper than the Gaussian process by the "
              "variance it does not compute. Measured on these kernels: Newton's method takes "
              "the same number of steps either way, so the fit saves only the factorisation the "
              "variance is read from and the determinant the marginal likelihood needs, while a "
              "prediction saves a triangular solve for every molecule scored.", "",
              "| set | molecules | one fit, klr | one fit, GP | ratio | all molecules scored, "
              "klr | all molecules scored, GP | ratio |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name in SETS:
            d = data.get(name)
            if d is None or not d.get("cost"):
                continue
            c = d["cost"]
            L.append(f"| {name} | {c['n']} | {c['fit_klr_ms']:.0f} ms | "
                     f"{c['fit_gpc_ms']:.0f} ms | {c['fit_ratio']:.2f} | "
                     f"{c['predict_klr_ms']:.1f} ms | {c['predict_gpc_ms']:.1f} ms | "
                     f"{c['predict_ratio']:.0f} |")
        L += ["", "So the saving is real but it is at prediction time, not at fit time: the fit "
                  "gains a few tens of per cent and scoring a library gains an order of "
                  "magnitude. A screen is where it matters.", ""]

    L += ["", "### Are the probabilities any good: predicted against observed", "",
          "Each row is one learner on one set: the share of molecules that turned out active, "
          "among those it gave a probability in that range. A calibrated classifier reproduces "
          "the bin. Empty cells are bins no molecule fell in, which on a set with one active in "
          "twenty is most of the upper range.", "",
          "| set | learner | " + " | ".join(f"{i / BINS:.1f}-{(i + 1) / BINS:.1f}"
                                            for i in range(BINS)) + " |",
          "|---|---|" + "---:|" * BINS]
    for name in SETS:
        d = data.get(name)
        if d is None:
            continue
        for m in ORDER:
            rowsc = (d["methods"].get(m) or {}).get("calibration") or []
            if not rowsc:
                continue
            by_bin = {int(min(BINS - 1, np.floor(r[0] * BINS))): r for r in rowsc}
            cells = []
            for i in range(BINS):
                r = by_bin.get(i)
                cells.append(f"{r[2]:.2f} ({int(r[3])})" if r else "-")
            L.append(f"| {name} | {LEARNERS[m][0]} | " + " | ".join(cells) + " |")
    L.append("")
    text = "\n".join(L) + "\n"
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=None, help="comma-separated data sets (default: both)")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None,
                    help="only write the Markdown table (default: results/klr/table.md)")
    args = ap.parse_args(argv)
    if args.report is None:
        asked = [s.strip() for s in args.sets.split(",")] if args.sets else None
        t0 = time.time()
        for name in SETS:
            if asked is None or name in asked:
                run_set(name, force=args.force)
        print(f"\nAll runs finished in {(time.time() - t0) / 60:.1f} min")
        args.report = os.path.join(RESULTS, "table.md")
    print(report(args.report))


if __name__ == "__main__":
    main()
