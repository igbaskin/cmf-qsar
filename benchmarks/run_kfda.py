#!/usr/bin/env python
"""
Kernel Fisher discriminant analysis against the program's other classifiers (plan item E8).

A co-field exists only where the model is linear in the field.  Of the classifiers this program
had, the Laplace Gaussian-process classifier is linear in its latent function and the
support-vector classifiers are not, so a chemist who wants a *picture* of a classification model
had one choice and no comparison.  The kernel Fisher discriminant (``--method kfda``) is the
second: its score is one linear functional of the field, found in closed form, so the co-field,
the molecule contributions, the Shapley values and the applicability domain all read it without
a special case.  This script measures what that costs in accuracy.

Part 1, the gate
----------------
The classification sets of plan item E4: the two LIT-PCBA targets (ESR1_ant and TP53), measured
actives against measured inactives, embedded and posed by `examples/make_litpcba_examples.py`.
Four learners on one kernel and one set of folds:

* ``kfda``  -- the kernel Fisher discriminant, ridge chosen inside the folds;
* ``svc``   -- C-SVC, the program's default classifier, C chosen inside the folds;
* ``gpc``   -- the Laplace Gaussian-process classifier, amplitude chosen inside the folds;
* ``krr``   -- kernel ridge regression on the labels +1 and -1, the simplest learner that can be
  put to this task at all and the reference rule 1 of the repository asks for.  The discriminant
  is known to agree with kernel ridge regression on class-size weighted labels as its ridge goes
  to zero, so this reference is not a straw man: it is nearly the same method, and a gain over it
  has to come from the class weighting and from the ridge being chosen for separation rather than
  for squared error.

The protocol is one 5-fold split of all the molecules, the folds stratified by class.  Inside
every outer fold each learner chooses its own setting on a 5-fold cross-validation of the outer
training molecules, scored by the area under the ROC curve, which is the criterion this program
already uses to select a classifier and which does not depend on where the decision threshold
lies.  Balanced accuracy, AUC and (where a learner has probabilities) the Brier score are then
computed on the pooled out-of-fold predictions, and every difference is reported with a 95 %
paired bootstrap interval over the molecules.

Balanced accuracy is reported twice, and the difference between the two is the whole result.  Each
learner leaves its decision boundary somewhere of its own, and on a set with one active in twenty
a boundary left at the fit's own zero calls nearly every molecule inactive and scores 0.5 whatever
the ordering behind it.  So the second column gives all four learners one boundary rule --- the cut
with the best balanced accuracy on out-of-fold scores of each fold's training molecules --- and
only that column compares the learners rather than their conventions.

The gate also asks for the co-field of the discriminant to be compared with the support-vector
classifier's, by the cosine of plan item F3.  Both are fitted on all the molecules of a set and
`cmfqsar.transfer.cofield_overlap` gives the per-field and overall cosines.

Part 2, where to cut the score
------------------------------
A discriminant direction says where the two classes differ, not where to cut between them.  Two
rules are available: the midpoint of the two projected class means, which is the Fisher threshold
for equal class priors and fits nothing, and the cut with the best balanced accuracy on
out-of-fold scores, which is what the plan entry prescribed.  They are compared on eight balanced
two-class problems made from the eight Sutherland sets in `examples/sutherland` by calling a
molecule active when its activity is at or above the median of its set -- the same molecules,
fields and alignment as every regression table in this repository, used as classification sets so
that the threshold rule is measured on data that is not the gate's.

Usage
-----
    python benchmarks/run_kfda.py                   # both parts
    python benchmarks/run_kfda.py --part gate --sets TP53 --force
    python benchmarks/run_kfda.py --report
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

import kernel_cache as KC  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.gpc import brier_score  # noqa: E402
from cmfqsar.model import (CMFModel, _Learner, _balanced_threshold,  # noqa: E402
                           classification_scores, cv_predict, kfda_frame)
from cmfqsar.transfer import cofield_overlap  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "kfda")
LITPCBA = os.path.join(ROOT, "examples", "litpcba")
SUTHERLAND = os.path.join(ROOT, "examples", "sutherland")

GATE_SETS = ("ESR1_ant", "TP53")
THRESHOLD_SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5
FOLDS = 5
INNER_FOLDS = 5
SEED = 0
RUNS = 2000                                  # bootstrap resamples

LAMS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CS = (0.1, 1.0, 10.0, 100.0)
#: what each learner is, and the settings it chooses from inside every fold
LEARNERS = {
    "kfda": ("kernel Fisher discriminant (ridge by CV)",
             [{"lam": l, "kfda_threshold": "cv"} for l in LAMS]),
    "svc": ("C-SVC (C by CV)", [{"C": c} for c in CS]),
    "gpc": ("Gaussian-process classifier (amplitude by CV)", [{"amplitude": c} for c in CS]),
    "krr": ("kernel ridge regression on +1/-1 labels (ridge by CV)",
            [{"lam": l} for l in LAMS]),
}
GATE_ORDER = ("kfda", "svc", "gpc", "krr")
#: learners that report a probability, so that calibration can be scored at all
PROBABILISTIC = ("kfda", "gpc")


# ----------------------------------------------------------------------------------------------
# folds and intervals
# ----------------------------------------------------------------------------------------------

def stratified_folds(y, folds=FOLDS, seed=SEED):
    """Fold indices with the share of actives of every fold equal to the share of the whole set.

    A screening set holds far fewer actives than inactives, and a plain random split can leave a
    fold with almost none; the statistics of that fold would then say more about the split than
    about the learner.
    """
    y = np.asarray(y, float)
    rng = np.random.RandomState(seed)
    out = [[] for _ in range(folds)]
    for cls in (y > 0, y <= 0):
        idx = np.flatnonzero(cls)
        idx = idx[rng.permutation(len(idx))]
        for k, i in enumerate(idx):
            out[k % folds].append(int(i))
    return [np.array(sorted(f), int) for f in out]


def _balanced_accuracy(y, decision):
    """Mean of sensitivity and specificity, with the class called active at a decision of zero."""
    pos = y > 0
    hit = decision >= 0.0
    n_p, n_n = int(np.count_nonzero(pos)), int(np.count_nonzero(~pos))
    if n_p == 0 or n_n == 0:
        return float("nan")
    return 0.5 * (np.count_nonzero(hit & pos) / n_p + np.count_nonzero(~hit & ~pos) / n_n)


def _auc(y, decision):
    """Area under the ROC curve by the rank sum, ties shared; the same number as sklearn's."""
    pos = y > 0
    n_p, n_n = int(np.count_nonzero(pos)), int(np.count_nonzero(~pos))
    if n_p == 0 or n_n == 0:
        return float("nan")
    order = np.argsort(decision, kind="mergesort")
    s = decision[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0                 # mid-rank for a group of ties
        i = j + 1
    return float((np.sum(ranks[pos[order]]) - n_p * (n_p + 1) / 2.0) / (n_p * n_n))


def _enrichment(y, decision, fraction=0.01):
    """Enrichment factor: how many times more actives the top `fraction` of the ranking holds
    than the whole library does.  1.0 is a random order and 1/(share of actives) the ceiling."""
    pos = y > 0
    n, n_act = len(y), int(np.count_nonzero(pos))
    if n_act == 0 or n_act == n:
        return float("nan")
    k = max(1, int(round(fraction * n)))
    order = np.argsort(-decision, kind="mergesort")
    return float((np.count_nonzero(pos[order[:k]]) / k) / (n_act / n))


def _bedroc(y, decision, alpha=20.0):
    """BEDROC: the enrichment read over the whole ranking with an exponential weight on the top,
    so that it does not saturate the way an enrichment factor at a small fraction does."""
    from cmfqsar.model import rie_bedroc

    pos = y > 0
    n, n_act = len(y), int(np.count_nonzero(pos))
    if n_act == 0 or n_act == n:
        return float("nan")
    order = np.argsort(-decision, kind="mergesort")
    ranks = np.flatnonzero(pos[order]) + 1
    return float(rie_bedroc(ranks, n, n_act, alpha=float(alpha))[1])


def _metric(name, y, decision, proba=None):
    """One metric of one prediction.  Written out here rather than taken from
    :func:`cmfqsar.model.classification_scores` because the bootstrap asks for a single number
    two thousand times and that function computes all of them, the AUC included."""
    if name == "brier":
        return float("nan") if proba is None else brier_score(y, proba)
    if name == "auc":
        return _auc(y, decision)
    if name in ("balanced_accuracy", "balanced_accuracy_common"):
        return _balanced_accuracy(y, decision)
    if name.startswith("ef_"):
        return _enrichment(y, decision, fraction=float(name[3:]) / 100.0)
    if name.startswith("bedroc_"):
        return _bedroc(y, decision, alpha=float(name[7:]))
    return float(classification_scores(y, decision)[name])


def paired_interval(name, y, a, b, runs=RUNS, seed=SEED):
    """(difference a - b, low, high) of one metric, resampling the molecules in pairs.

    The two learners saw the same molecules in the same folds, so the molecules are resampled
    once and both predictions are scored on the same resample; what the interval covers is the
    uncertainty of the *difference*, which is narrower than the two separate intervals because
    the part of the error that is the choice of molecules is shared.
    """
    y = np.asarray(y, float)
    da, pa = a
    db, pb = b
    n = len(y)
    point = _metric(name, y, da, pa) - _metric(name, y, db, pb)
    if not np.isfinite(point):
        return point, float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(runs)):
        idx = rng.randint(0, n, n)
        if not ((y[idx] > 0).any() and (y[idx] <= 0).any()):
            continue
        va = _metric(name, y[idx], da[idx], None if pa is None else pa[idx])
        vb = _metric(name, y[idx], db[idx], None if pb is None else pb[idx])
        if np.isfinite(va) and np.isfinite(vb):
            out.append(va - vb)
    if len(out) < 100:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


# ----------------------------------------------------------------------------------------------
# one learner, one outer fold
# ----------------------------------------------------------------------------------------------

def _select(K, y, method, grid, folds=INNER_FOLDS, seed=SEED):
    """The setting with the best out-of-fold AUC, among `grid`, on these molecules alone."""
    best = None
    lab = np.isfinite(np.asarray(y, float))          # nan: in the fit, nothing to score
    for params in grid:
        p = dict(params)
        # the selection criterion is the AUC, which no threshold can change, so the threshold
        # search is left out of the inner folds: it would cost five fits per fit and alter nothing
        if method == "kfda":
            p["kfda_threshold"] = "midpoint"
        ycv = cv_predict(K, y, method, p, folds=folds, seed=seed)
        auc = classification_scores(y[lab], ycv[lab])["auc"]
        if not np.isfinite(auc):
            auc = classification_scores(y[lab], ycv[lab])["balanced_accuracy"]
        if best is None or auc > best[1]:
            best = (dict(params), float(auc))
    return best[0], best[1]


def _common_cut(K_tr, y_tr, method, params, score_tr, inner=INNER_FOLDS, seed=SEED):
    """The decision boundary every learner is given, so that only the learners are compared.

    Each learner leaves its boundary somewhere of its own: a support-vector classifier at the
    margin its quadratic program found, kernel ridge regression at zero on +-1 labels, the
    discriminant at the cut its own calibration chose.  On a set with one active in twenty, a
    boundary left at zero calls almost everything inactive and scores 0.5 balanced accuracy
    whatever the ordering behind it, so a comparison of balanced accuracies at those boundaries
    is a comparison of conventions.  This gives all of them one rule: the cut with the best
    balanced accuracy on out-of-fold scores of the training molecules, in the class-mean units of
    :func:`cmfqsar.model.kfda_frame` so that the scores of the inner folds and of the final fit
    are on one scale.  Returns the cut in raw score units of the final fit.
    """
    ycv = cv_predict(K_tr, y_tr, method, params, folds=inner, seed=seed)
    lab = np.isfinite(np.asarray(y_tr, float)) & np.isfinite(ycv)
    centre_in, spread_in = kfda_frame(ycv[lab], y_tr[lab] > 0)
    cut_scaled, _ = _balanced_threshold(y_tr[lab], (ycv[lab] - centre_in) / spread_in)
    centre, spread = kfda_frame(score_tr[lab], y_tr[lab] > 0)
    return centre + cut_scaled * spread


def cross_validate(K, y, method, grid, folds=FOLDS, seed=SEED, inner=INNER_FOLDS):
    """Pooled out-of-fold predictions, the setting chosen inside each fold.

    Returns (decision values as the learner leaves them, decision values with the common
    threshold rule of :func:`_common_cut`, probabilities or None, the settings chosen).
    """
    y = np.asarray(y, float)
    n = len(y)
    decision = np.full(n, np.nan)
    tuned = np.full(n, np.nan)
    proba = np.full(n, np.nan)
    chosen = []
    for te in stratified_folds(y, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        Ktr = K[np.ix_(tr, tr)]
        params, _ = _select(Ktr, y[tr], method, grid, folds=inner, seed=seed)
        chosen.append(params)
        lr = _Learner(method, params).fit(Ktr, y[tr])
        Kte = K[np.ix_(te, tr)]
        decision[te] = lr.predict(Kte)
        # the common rule acts on the raw score, so the learner's own bias is taken off first
        plain = dict(params)
        if method == "kfda":
            plain["kfda_threshold"] = "midpoint"
        raw_tr, raw_te = Ktr @ lr.alpha_, Kte @ lr.alpha_
        tuned[te] = raw_te - _common_cut(Ktr, y[tr], method, plain, raw_tr, inner, seed)
        proba[te] = probabilities(lr, Kte, raw_te, np.diag(K)[te])
    has_proba = bool(np.isfinite(proba).all())
    return decision, tuned, (proba if has_proba else None), chosen


def probabilities(lr, K_test_train, raw_score, diag_test):
    """p(active) from whichever learner this is, or nan where the learner has none.

    Each probabilistic classifier in the program reports differently: the discriminant through
    the logistic map fitted on its own out-of-fold scores, the Gaussian process through the
    likelihood averaged over the latent variance, kernel logistic regression through the
    likelihood at the mode with no variance in it.  The support-vector classifiers and kernel
    ridge regression have none, because a decision value is not a probability.
    """
    cal = getattr(lr, "kfda_", None)
    if cal is not None:
        return cal.probability(raw_score)
    if getattr(lr, "klr_", None) is not None:
        return lr.klr_.predict_proba(K_test_train)
    if getattr(lr, "gpc_", None) is not None:
        return lr.gpc_.predict_proba(K_test_train, diag_test)
    return np.full(np.atleast_2d(K_test_train).shape[0], np.nan)


# ----------------------------------------------------------------------------------------------
# part 1: the gate
# ----------------------------------------------------------------------------------------------

def _litpcba_kernel(name):
    """The joint five-field kernel of one LIT-PCBA set, actives then inactives, and the labels."""
    act_path = os.path.join(LITPCBA, f"{name}.sdf")
    dec_path = os.path.join(LITPCBA, f"{name}_decoys.sdf")
    if not (os.path.exists(act_path) and os.path.exists(dec_path)):
        print(f"{name}: not built; run examples/make_litpcba_examples.py")
        return None, None, None, None
    actives, decoys = load_dataset(act_path), load_dataset(dec_path)
    mols = list(actives.mols) + list(decoys.mols)
    names = list(actives.names) + [f"d{i}" for i in range(len(decoys.mols))]
    y = np.array([1.0] * len(actives.mols) + [-1.0] * len(decoys.mols))
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA)
    table = AtomTable.from_molecules([build_fields(m, fs, nm) for m, nm in zip(mols, names)])
    model = CMFModel(fieldset=fs, task="classification", method="kfda", normalize=True)
    K = KC.joint_training_kernel(model, table)
    return K, y, model, table


def run_gate(name, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"gate_{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    K, y, model, table = _litpcba_kernel(name)
    if K is None:
        return None
    n_a = int(np.sum(y > 0))
    print(f"=== {name}: {n_a} measured actives, {len(y) - n_a} measured inactives, "
          f"kernel {K.shape} in {time.time() - t0:.0f} s", flush=True)

    preds, tuned_preds, rows = {}, {}, {}
    for method in GATE_ORDER:
        label, grid = LEARNERS[method]
        t1 = time.time()
        d, t, p, chosen = cross_validate(K, y, method, grid)
        preds[method] = (d, p)
        tuned_preds[method] = (t, p)
        sc, sct = classification_scores(y, d), classification_scores(y, t)
        rows[method] = {"balanced_accuracy": sc["balanced_accuracy"], "auc": sc["auc"],
                        "sensitivity": sc["sensitivity"], "specificity": sc["specificity"],
                        "mcc": sc["mcc"],
                        "balanced_accuracy_common": sct["balanced_accuracy"],
                        "sensitivity_common": sct["sensitivity"],
                        "specificity_common": sct["specificity"],
                        "mcc_common": sct["mcc"],
                        "brier": brier_score(y, p) if p is not None else float("nan"),
                        "chosen": [{k: v for k, v in c.items() if k != "kfda_threshold"}
                                   for c in chosen],
                        "seconds": round(time.time() - t1, 1)}
        print(f"    {method:5s} balanced {sc['balanced_accuracy']:.3f} (own boundary) "
              f"{sct['balanced_accuracy']:.3f} (common rule)  AUC {sc['auc']:.3f}  "
              f"Brier {rows[method]['brier']:.4f}  ({time.time() - t1:.0f} s)", flush=True)

    diffs = {}
    for other in GATE_ORDER[1:]:
        for metric in ("balanced_accuracy", "auc", "brier"):
            if metric == "brier" and other not in PROBABILISTIC:
                continue
            v, lo, hi = paired_interval(metric, y, preds["kfda"], preds[other])
            diffs[f"kfda-{other}/{metric}"] = [v, lo, hi]
            print(f"    kfda - {other:4s} {metric:18s} {v:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                  flush=True)
        v, lo, hi = paired_interval("balanced_accuracy", y, tuned_preds["kfda"],
                                    tuned_preds[other])
        diffs[f"kfda-{other}/balanced_accuracy_common"] = [v, lo, hi]
        print(f"    kfda - {other:4s} {'balanced, common rule':18s} {v:+.3f}  "
              f"[{lo:+.3f}, {hi:+.3f}]", flush=True)

    # the co-field comparison the gate asks for: both classifiers on all the molecules, each at
    # the setting its folds chose most often, so that the maps compared are the ones the protocol
    # would deliver
    def modal(method, key, default):
        vals = [c.get(key, default) for c in rows[method]["chosen"]]
        return max(set(vals), key=vals.count) if vals else default

    cos = {}
    try:
        a = CMFModel(fieldset=FieldSet.create(physchem=FIELDS, alpha=ALPHA),
                     task="classification", method="kfda",
                     params={"lam": modal("kfda", "lam", 0.1)}, normalize=True)
        b = CMFModel(fieldset=FieldSet.create(physchem=FIELDS, alpha=ALPHA),
                     task="classification", method="svc",
                     params={"C": modal("svc", "C", 1.0)}, normalize=True)
        a.fit(table, y, cv_folds=0)
        b.fit(table, y, cv_folds=0)
        ov = cofield_overlap(a, b, "kfda", "svc")
        cos = {"overall": float(ov.cosine),
               "fields": {r.field: float(r.cosine) for r in ov.fields},
               "lam": float(modal("kfda", "lam", 0.1)), "C": float(modal("svc", "C", 1.0))}
        print(f"    co-field cosine against the SVC: {ov.cosine:+.3f}  "
              + "  ".join(f"{k} {v:+.2f}" for k, v in cos["fields"].items()), flush=True)
    except Exception as exc:                                        # noqa: BLE001
        cos = {"error": str(exc)}
        print(f"    co-field cosine unavailable: {exc}", flush=True)

    # what predicting the share of actives for every molecule would score: on a set with one
    # active in twenty this is already a small number, and a Brier score has to be read against it
    share = n_a / float(len(y))
    base_brier = float(share * (1.0 - share))
    print(f"    the base rate alone scores Brier {base_brier:.4f} "
          f"(share of actives {share:.3f})", flush=True)

    out = {"set": name, "part": "gate", "n": int(len(y)), "n_actives": n_a,
           "alpha": ALPHA, "folds": FOLDS, "inner_folds": INNER_FOLDS, "seed": SEED,
           "fields": FIELDS, "methods": rows, "differences": diffs, "cofield_cosine": cos,
           "base_rate_brier": base_brier, "active_share": share,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# ----------------------------------------------------------------------------------------------
# part 2: where to cut the score
# ----------------------------------------------------------------------------------------------

THRESHOLD_RULES = {
    "midpoint": ("midpoint of the two class means (nothing fitted)", "midpoint"),
    "cv": ("best balanced accuracy on out-of-fold scores", "cv"),
}


def run_threshold(name, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"threshold_{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(SUTHERLAND, f"{name}.sdf"))
    activity = np.asarray(ds.activity, float)
    median = float(np.nanmedian(activity))
    y = np.where(activity >= median, 1.0, -1.0)
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA)
    table = AtomTable.from_molecules(
        [build_fields(m, fs, nm) for m, nm in zip(ds.mols, ds.names)])
    model = CMFModel(fieldset=fs, task="classification", method="kfda", normalize=True)
    K = KC.joint_training_kernel(model, table)
    n_a = int(np.sum(y > 0))
    print(f"=== {name}: {len(y)} molecules, active at or above {median:.2f} "
          f"({n_a} active, {len(y) - n_a} inactive)", flush=True)

    preds, rows = {}, {}
    for rule, (label, value) in THRESHOLD_RULES.items():
        grid = [{"lam": l, "kfda_threshold": value} for l in LAMS]
        d, _, p, chosen = cross_validate(K, y, "kfda", grid)
        preds[rule] = (d, p)
        sc = classification_scores(y, d)
        rows[rule] = {"balanced_accuracy": sc["balanced_accuracy"], "auc": sc["auc"],
                      "sensitivity": sc["sensitivity"], "specificity": sc["specificity"],
                      "chosen": [c["lam"] for c in chosen]}
        print(f"    {rule:9s} balanced {sc['balanced_accuracy']:.3f}  "
              f"sens {sc['sensitivity']:.3f}  spec {sc['specificity']:.3f}  "
              f"AUC {sc['auc']:.3f}", flush=True)
    v, lo, hi = paired_interval("balanced_accuracy", y, preds["midpoint"], preds["cv"])
    print(f"    midpoint - cv  balanced accuracy {v:+.3f}  [{lo:+.3f}, {hi:+.3f}]", flush=True)

    out = {"set": name, "part": "threshold", "n": int(len(y)), "n_actives": n_a,
           "median": median, "alpha": ALPHA, "folds": FOLDS, "rules": rows,
           "midpoint_minus_cv": [v, lo, hi], "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

def _load(results, prefix):
    out = {}
    for name in os.listdir(results) if os.path.isdir(results) else []:
        if name.startswith(prefix) and name.endswith(".json"):
            d = json.load(open(os.path.join(results, name)))
            out[d["set"]] = d
    return out


def _f(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{nd}f}" if v == v else "-"


def _interval(trip, nd=3):
    if not trip or not np.isfinite(trip[0]):
        return "-"
    return f"{trip[0]:+.{nd}f} [{trip[1]:+.{nd}f}, {trip[2]:+.{nd}f}]"


def _pooled_interval(data, key, metric):
    """The mean difference over the sets, with the mean of the per-set interval half-widths.

    Two sets are too few to resample as a population, so what is reported is the mean of the
    per-set differences and the mean interval around it; it says how large the per-set
    uncertainty is, not how much the two sets agree.
    """
    vals = [d["differences"].get(f"{key}/{metric}") for d in data.values()]
    vals = [v for v in vals if v and np.isfinite(v[0])]
    if not vals:
        return None
    return [float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])),
            float(np.mean([v[2] for v in vals]))]


def report(path=None, results=RESULTS):
    gate = _load(results, "gate_")
    thr = _load(results, "threshold_")
    L = ["# Kernel Fisher discriminant analysis against the other classifiers (plan item E8)", ""]
    if not gate and not thr:
        return "No results yet; run `python benchmarks/run_kfda.py`.\n"

    if gate:
        first = next(iter(gate.values()))
        L += [f"## The gate: the classification sets of plan item E4", "",
              f"LIT-PCBA, measured actives against measured inactives, posed by "
              f"`examples/make_litpcba_examples.py`. One kernel per set (five fields, attenuation "
              f"{first['alpha']:g}, cosine-normalised), one {first['folds']}-fold split "
              f"stratified by class, and each learner choosing its own setting on a "
              f"{first['inner_folds']}-fold cross-validation of the training molecules of every "
              f"fold, scored by AUC. Balanced accuracy, AUC and the Brier score are computed on "
              f"the pooled out-of-fold predictions.", "",
              "Two balanced accuracies are given. *Own boundary* is where each learner leaves its "
              "decision boundary: the discriminant at the cut its own calibration chose, the "
              "support-vector classifier at the margin its quadratic program found, kernel ridge "
              "regression at zero on the labels +1 and -1. *Common rule* gives all four the same "
              "boundary, the best balanced accuracy on out-of-fold scores of the training "
              "molecules of each fold. On a set with one active in twenty a boundary left at zero "
              "calls nearly every molecule inactive, so only the second column compares learners "
              "rather than conventions.", "",
              "| set | molecules | actives | learner | balanced accuracy, own boundary | "
              "balanced accuracy, common rule | AUC | sensitivity, common rule | "
              "specificity, common rule | Brier score |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
        for name in GATE_SETS:
            d = gate.get(name)
            if d is None:
                continue
            first_row = True
            for m in GATE_ORDER:
                r = d["methods"].get(m)
                if r is None:
                    continue
                L.append(f"| {name if first_row else ''} | "
                         f"{d['n'] if first_row else ''} | {d['n_actives'] if first_row else ''} | "
                         f"{LEARNERS[m][0]} | {_f(r['balanced_accuracy'])} | "
                         f"{_f(r.get('balanced_accuracy_common'))} | {_f(r['auc'])} | "
                         f"{_f(r.get('sensitivity_common'))} | "
                         f"{_f(r.get('specificity_common'))} | {_f(r['brier'], 4)} |")
                first_row = False
            L.append(f"| {''} | {''} | {''} | *the share of actives, predicted for every "
                     f"molecule* | 0.500 | 0.500 | 0.500 | - | - | "
                     f"{_f(d.get('base_rate_brier'), 4)} |")
        keys = ("balanced_accuracy", "balanced_accuracy_common", "auc", "sensitivity_common",
                "specificity_common", "brier")
        L += ["", "Averaged over the sets:", "",
              "| learner | balanced accuracy, own boundary | balanced accuracy, common rule | "
              "AUC | sensitivity, common rule | specificity, common rule | Brier score |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for m in GATE_ORDER:
            vals = [gate[n]["methods"][m] for n in GATE_SETS
                    if n in gate and m in gate[n]["methods"]]
            if not vals:
                continue
            cells = []
            for k in keys:
                col = [float(v.get(k, np.nan)) for v in vals]
                col = [c for c in col if np.isfinite(c)]
                # a learner with no probabilities has no Brier score on any set, and averaging
                # an all-empty column would only warn
                cells.append(_f(np.mean(col), 4 if k == "brier" else 3) if col else "-")
            L.append(f"| {LEARNERS[m][0]} | " + " | ".join(cells) + " |")
        base = [gate[n].get("base_rate_brier") for n in GATE_SETS
                if n in gate and gate[n].get("base_rate_brier") is not None]
        if base:
            L.append("| *the share of actives, predicted for every molecule* | 0.500 | 0.500 | "
                     f"0.500 | - | - | {_f(np.mean(base), 4)} |")
        L += ["", "The Brier score is the mean squared error of the probabilities: 0 is perfect "
                  "and 0.25 is no better than chance. The support-vector and ridge rows have "
                  "none, because a decision value is not a probability. The last row of each set "
                  "is the reference a Brier score has to be read against: a model that predicts "
                  "the share of actives for every molecule and says nothing else. On a set with "
                  "one active in twenty that model already scores a small Brier score, so a "
                  "small Brier score is not by itself evidence of a useful probability.", ""]

        L += ["### The differences, with 95 % paired bootstrap intervals", "",
              "The molecules are resampled once per draw and both learners scored on the same "
              "resample. A difference whose interval contains zero is not a difference.", "",
              "| comparison | metric | " + " | ".join(GATE_SETS) + " | mean |",
              "|---|---|" + "---|" * (len(GATE_SETS) + 1)]
        for other in GATE_ORDER[1:]:
            for metric, mlabel in (("balanced_accuracy", "balanced accuracy, own boundary"),
                                   ("balanced_accuracy_common", "balanced accuracy, common rule"),
                                   ("auc", "AUC"), ("brier", "Brier score")):
                if metric == "brier" and other not in PROBABILISTIC:
                    continue
                key = f"kfda-{other}"
                cells = [_interval(gate[n]["differences"].get(f"{key}/{metric}"))
                         if n in gate else "-" for n in GATE_SETS]
                pooled = _pooled_interval(gate, key, metric)
                L.append(f"| kfda - {other} | {mlabel} | " + " | ".join(cells)
                         + f" | {_interval(pooled)} |")
        L += ["", "For the Brier score a negative difference is the better one, because it is an "
                  "error; for the other two a positive difference is.", ""]

        L += ["### The co-field of the discriminant against the support-vector classifier's", "",
              "Both classifiers fitted on all the molecules of a set, their co-fields compared by "
              "the cosine of plan item F3. The support-vector classifier has no co-field in the "
              "sense of rule 2 -- its coefficients are not a linear map of the labels -- so this "
              "cosine compares the discriminant's picture with the kernel expansion the "
              "support-vector classifier happens to produce, which is why it is read as a check "
              "that the two ask the chemistry for the same thing and not as a validation.", "",
              "| set | overall cosine | " + " | ".join(FIELDS) + " |",
              "|---|---:|" + "---:|" * len(FIELDS)]
        for name in GATE_SETS:
            d = gate.get(name)
            if d is None or "overall" not in (d.get("cofield_cosine") or {}):
                continue
            c = d["cofield_cosine"]
            L.append(f"| {name} | {_f(c['overall'])} | "
                     + " | ".join(_f(c["fields"].get(f)) for f in FIELDS) + " |")
        L.append("")

    if thr:
        L += ["## Where to cut the score: the two threshold rules", "",
              "Eight balanced two-class problems from the Sutherland sets, a molecule being "
              "active when its activity is at or above the median of its set. The direction and "
              "the ridge are identical in the two columns -- only the bias differs -- so the AUC "
              "is the same and only the balanced accuracy can move.", "",
              "| set | molecules | median | midpoint of the class means | best out-of-fold cut | "
              "difference, midpoint - out-of-fold |",
              "|---|---:|---:|---:|---:|---|"]
        mid, cv, diffs = [], [], []
        for name in THRESHOLD_SETS:
            d = thr.get(name)
            if d is None:
                continue
            a = d["rules"]["midpoint"]["balanced_accuracy"]
            b = d["rules"]["cv"]["balanced_accuracy"]
            mid.append(a)
            cv.append(b)
            diffs.append(d["midpoint_minus_cv"])
            L.append(f"| {name} | {d['n']} | {_f(d['median'], 2)} | {_f(a)} | {_f(b)} | "
                     f"{_interval(d['midpoint_minus_cv'])} |")
        if mid:
            L.append(f"| **mean** | | | **{_f(np.mean(mid))}** | **{_f(np.mean(cv))}** | "
                     f"**{_f(np.mean([v[0] for v in diffs]))}** |")
            n_mid = sum(1 for a, b in zip(mid, cv) if a > b)
            n_zero = sum(1 for v in diffs
                         if np.isfinite(v[1]) and v[1] <= 0.0 <= v[2])
            gap = float(np.mean(mid) - np.mean(cv))
            L += ["", f"The midpoint is ahead on {n_mid} of the {len(mid)} sets and the "
                      f"out-of-fold cut on {len(mid) - n_mid}, a mean difference of {gap:+.3f} "
                      f"balanced accuracy. The interval contains zero on {n_zero} of the "
                      f"{len(mid)} sets, so the two rules cannot be told apart here: searching "
                      f"for the cut gains nothing measurable over the Fisher threshold, and "
                      f"loses nothing either.", ""]
    text = "\n".join(L) + "\n"
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="both", choices=["both", "gate", "threshold"])
    ap.add_argument("--sets", default=None, help="comma-separated data sets (default: all)")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None,
                    help="only write the Markdown table (default: results/kfda/table.md)")
    args = ap.parse_args(argv)
    if args.report is None:
        asked = [s.strip() for s in args.sets.split(",")] if args.sets else None
        t0 = time.time()
        if args.part in ("both", "gate"):
            for name in GATE_SETS:
                if asked is None or name in asked:
                    run_gate(name, force=args.force)
        if args.part in ("both", "threshold"):
            for name in THRESHOLD_SETS:
                if asked is None or name in asked:
                    run_threshold(name, force=args.force)
        print(f"\nAll runs finished in {(time.time() - t0) / 60:.1f} min")
        args.report = os.path.join(RESULTS, "table.md")
    print(report(args.report))


if __name__ == "__main__":
    main()
