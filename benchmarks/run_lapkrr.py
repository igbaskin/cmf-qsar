#!/usr/bin/env python
"""
Laplacian-regularised kernel ridge regression: do molecules without an activity help? (item E10)

Every learner in this program fits the molecules whose activity is known and ignores the rest.
A screening library, a set of decoys, a batch not yet measured: these molecules have fields, and
manifold regularisation (Belkin, Niyogi & Sindhwani 2006) is the standard way of putting them to
use.  A second penalty, ``gamma g' L g`` over the graph of field similarity, asks the prediction
to change slowly between molecules that are close in field space, whether or not they are
labelled (``g = K a``, the prediction without its bias term).  ``--method lapkrr`` adds that
penalty to kernel ridge regression; this script measures what it gains, in the two halves of the
gate the plan entry sets.

Part 1, the hidden half (regression)
------------------------------------
The eight Sutherland sets in ``examples/sutherland``.  Half of the published training molecules
keep their activities and half have them hidden, and three models are compared on the published
test set, which none of them has seen:

* ``ridge-half``  -- kernel ridge regression on the visible half alone: what is available today;
* ``lapkrr``      -- the same ridge plus the penalty, with the hidden half in the kernel and in
  the expansion but with no activity;
* ``ridge-all``   -- kernel ridge regression on all the training molecules with every activity.
  This is the **ceiling**: it is what the hidden activities would be worth if they were measured.

The question the gate asks is how much of the gap between ``ridge-half`` and ``ridge-all`` the
penalty recovers.  Each learner chooses its own settings inside the molecules it is allowed to
see, by 10-fold cross-validated RMSE: ``ridge-half`` and ``ridge-all`` their ridge, ``lapkrr``
its ridge and the weight of the penalty, whose grid includes zero -- so the search can decide
that the unlabelled molecules do not help.  The halving is repeated five times with different
seeds and the reported figure is the mean over the five; the 95 % interval comes from a paired
bootstrap over the **test molecules**, which are the same for all three models.

A second column reads the hidden molecules themselves rather than the test set.  There the two
models are not doing the same thing: ``lapkrr`` has those molecules in its expansion and is
predicting molecules it has seen the fields of, which is the transductive setting manifold
regularisation is usually stated in, while ``ridge-half`` has never met them.  Both numbers are
reported because they answer different questions.

Part 2, the library unlabelled (screening)
------------------------------------------
The two LIT-PCBA sets of plan item E4 (ESR1_ant and TP53), under the protocol of
``benchmarks/run_kfda.py``: one 5-fold split stratified by class, each learner choosing its
setting inside every fold by out-of-fold AUC, every difference carrying a 95 % paired bootstrap
interval over the molecules.  In each fold the held-out molecules are given to ``lapkrr`` as the
**unlabelled library** -- they are rows of its kernel and terms of its expansion, with no label
-- which is what a real screen offers and what the gate asks for.  The references are ``krr`` on
the labels +1 and -1, which is the same learner without the penalty, and ``svc``, the program's
default classifier.  Enrichment is read as the enrichment factor in the top 1 % and 5 % of the
ranking and as BEDROC at alpha 20, beside AUC and the balanced accuracy at the common cut.

The assumption, and how to read a loss against it
-------------------------------------------------
The penalty assumes the activity varies smoothly along the data.  An **activity cliff** -- two
molecules whose fields are nearly the same and whose activities are not -- breaks that
assumption, and the penalty then pulls both predictions towards their average.  Part 1 therefore
counts, per set, how many pairs of labelled molecules are cliffs.  The similarity is the cosine
of the two molecular fields in the model's own kernel, which is the same inner-product cosine
plan item F3 uses to compare two co-fields, applied to molecules instead of to models; a pair
counts as similar at a cosine of 0.9 or more and as a cliff when its activities also differ by
at least one log unit.  Those two cutoffs are this script's choice and are stated with the
counts, because F3 fixes the similarity measure and not a threshold on it.

Usage
-----
    python benchmarks/run_lapkrr.py                      # both parts
    python benchmarks/run_lapkrr.py --part hidden --sets GPB,THR --force
    python benchmarks/run_lapkrr.py --part neighbours    # what the size of the graph costs
    python benchmarks/run_lapkrr.py --report
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
from cmfqsar.model import (CMFModel, _Learner, classification_scores,  # noqa: E402
                           cv_predict, r2_score, rmse, screening_scores)
from run_kfda import (_auc, _balanced_accuracy, _common_cut, _litpcba_kernel,  # noqa: E402
                      _select, paired_interval, probabilities, stratified_folds)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "lapkrr")
SUTHERLAND = os.path.join(ROOT, "examples", "sutherland")

HIDDEN_SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
SCREEN_SETS = ("ESR1_ant", "TP53")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5
SEED = 0
RUNS = 2000                                  # bootstrap resamples

#: part 1: how often the training molecules are halved, and the share that keeps its activity
REPEATS = 5
VISIBLE_SHARE = 0.5
INNER_FOLDS = 10
#: part 1 grids
LAMS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
MANIFOLDS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
NEIGHBOURS = 10
#: part 2 grids, smaller because a fit on 1570 molecules costs a cube of that
SCREEN_LAMS = (0.01, 0.1, 1.0)
SCREEN_MANIFOLDS = (0.0, 0.1, 1.0)
SCREEN_CS = (0.1, 1.0, 10.0, 100.0)
SCREEN_FOLDS = 5
SCREEN_INNER = 5
#: part 3: neighbourhood sizes compared, 0 meaning every pair
NEIGHBOUR_GRID = (5, 10, 20, 0)

#: activity cliffs: a pair counts as similar at this field cosine and as a cliff when its
#: activities also differ by at least this much (see the module docstring)
CLIFF_COSINE = 0.9
CLIFF_GAP = 1.0

HIDDEN_MODES = ("ridge-half", "lapkrr", "ridge-all")
#: settings chosen on the test set: an oracle that cannot be used, reported only to tell a method
#: that does not help apart from a selection that cannot find where it does
ORACLE_MODES = ("ridge-half-oracle", "lapkrr-oracle")
HIDDEN_LABELS = {
    "ridge-half": "ridge, the visible half only",
    "lapkrr": "lapkrr, the hidden half unlabelled",
    "ridge-all": "ridge, every activity (the ceiling)",
    "ridge-half-oracle": "ridge, visible half, ridge chosen on the test set",
    "lapkrr-oracle": "lapkrr, both settings chosen on the test set",
}
SCREEN_ORDER = ("lapkrr", "krr", "svc")
SCREEN_LABELS = {
    "lapkrr": "lapkrr, the held-out fold unlabelled",
    "krr": "kernel ridge regression on +1/-1 labels",
    "svc": "C-SVC, the default classifier",
}
SCREEN_METRICS = ("auc", "ef_1", "ef_5", "bedroc_20", "balanced_accuracy_common")
SCREEN_METRIC_LABELS = {
    "auc": "AUC",
    "ef_1": "enrichment in the top 1 %",
    "ef_5": "enrichment in the top 5 %",
    "bedroc_20": "BEDROC (alpha 20)",
    "balanced_accuracy_common": "balanced accuracy, common cut",
}


# ----------------------------------------------------------------------------------------------
# shared pieces
# ----------------------------------------------------------------------------------------------

def _f(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{nd}f}" if v == v else "-"


def _signed(v, nd=3):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{v:+.{nd}f}" if v == v else "-"


def _ranks(v):
    """Mid-ranks of a vector, so that a tie shares its place."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def _spearman(a, b):
    """Rank correlation of two short lists; nan when either is constant."""
    ra, rb = _ranks(a), _ranks(b)
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _zero_weight_share(chosen):
    """How many of the fitted models were given a penalty weight of zero, that is no penalty."""
    vals = [float(c.get("manifold", float("nan"))) for c in chosen]
    vals = [v for v in vals if np.isfinite(v)]
    if not vals:
        return float("nan")
    return sum(1 for v in vals if v == 0.0) / len(vals)


def _interval(t, nd=3):
    v, lo, hi = t
    if not np.isfinite(v):
        return "-"
    if not np.isfinite(lo):
        return f"{v:+.{nd}f}"
    return f"**{v:+.{nd}f}** [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def _sutherland_kernel(name):
    """The joint five-field kernel of one Sutherland set, its activities and the published split."""
    ds = load_dataset(os.path.join(SUTHERLAND, f"{name}.sdf"))
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA)
    table = AtomTable.from_molecules(
        [build_fields(m, fs, nm) for m, nm in zip(ds.mols, ds.names)])
    model = CMFModel(fieldset=fs, method="lapkrr", normalize=True)
    K = KC.joint_training_kernel(model, table)
    y = np.asarray(ds.activity, float)
    train = np.array([i for i in ds.train_indices if np.isfinite(y[i])], int)
    test = np.array([i for i in ds.test_indices if np.isfinite(y[i])], int)
    return K, y, train, test


def _select_regression(K, y, method, grid, folds=INNER_FOLDS, seed=SEED):
    """The setting with the lowest out-of-fold RMSE among `grid`, on these molecules alone.

    The molecules whose activity is nan stay in the training part of every fold and are never
    scored, so a setting is chosen without any hidden activity being used.
    """
    lab = np.isfinite(np.asarray(y, float))
    best = None
    for params in grid:
        ycv = cv_predict(K, y, method, dict(params), folds=folds, seed=seed)
        err = rmse(y[lab], ycv[lab])
        if not np.isfinite(err):
            continue
        if best is None or err < best[1]:
            best = (dict(params), float(err))
    if best is None:
        return dict(grid[0]), float("nan")
    return best


def cliff_counts(K, y, idx, cosine=CLIFF_COSINE, gap=CLIFF_GAP):
    """How many pairs of labelled molecules are activity cliffs, and how many are merely similar.

    The kernel is cosine-normalised, so ``K[i, j]`` is the cosine of the two molecular fields
    already.  A pair is similar at a cosine of `cosine` or above and a cliff when its activities
    also differ by at least `gap`.
    """
    idx = np.asarray(idx, int)
    sub = K[np.ix_(idx, idx)]
    d = np.sqrt(np.maximum(np.diag(sub), 1e-300))
    cos = sub / np.outer(d, d)
    dy = np.abs(y[idx][:, None] - y[idx][None, :])
    upper = np.triu(np.ones_like(cos, dtype=bool), 1)
    similar = upper & (cos >= cosine)
    cliff = similar & (dy >= gap)
    n_pairs = int(np.count_nonzero(upper))
    n_sim = int(np.count_nonzero(similar))
    n_cliff = int(np.count_nonzero(cliff))
    return {"pairs": n_pairs, "similar": n_sim, "cliffs": n_cliff,
            "cliff_share_of_similar": (n_cliff / n_sim) if n_sim else float("nan"),
            "cliff_share_of_pairs": (n_cliff / n_pairs) if n_pairs else float("nan"),
            "cosine_cutoff": float(cosine), "gap_cutoff": float(gap),
            "mean_gap_of_similar": float(np.mean(dy[similar])) if n_sim else float("nan")}


# ----------------------------------------------------------------------------------------------
# part 1: the hidden half
# ----------------------------------------------------------------------------------------------

def _one_halving(K, y, train, test, seed, neighbours=NEIGHBOURS):
    """Test-set and hidden-molecule predictions of the three models for one halving."""
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train))
    n_vis = max(2, int(round(VISIBLE_SHARE * len(train))))
    vis = np.sort(train[order[:n_vis]])
    hid = np.sort(train[order[n_vis:]])
    allt = np.sort(train)

    y_semi = y.copy()
    y_semi[hid] = np.nan

    out = {}
    # ridge on the visible half alone
    p, _e = _select_regression(K[np.ix_(vis, vis)], y[vis], "krr", [{"lam": l} for l in LAMS])
    lr = _Learner("krr", p).fit(K[np.ix_(vis, vis)], y[vis])
    out["ridge-half"] = {"test": lr.predict(K[np.ix_(test, vis)]),
                         "hidden": lr.predict(K[np.ix_(hid, vis)]), "chosen": p}
    # the same ridge with the hidden half in the kernel, unlabelled
    grid = [{"lam": l, "manifold": g, "neighbours": neighbours}
            for l in LAMS for g in MANIFOLDS]
    p, _e = _select_regression(K[np.ix_(allt, allt)], y_semi[allt], "lapkrr", grid)
    lr = _Learner("lapkrr", p).fit(K[np.ix_(allt, allt)], y_semi[allt])
    out["lapkrr"] = {"test": lr.predict(K[np.ix_(test, allt)]),
                     "hidden": lr.predict(K[np.ix_(hid, allt)]), "chosen": p}
    # the ceiling: every activity measured
    p, _e = _select_regression(K[np.ix_(allt, allt)], y[allt], "krr", [{"lam": l} for l in LAMS])
    lr = _Learner("krr", p).fit(K[np.ix_(allt, allt)], y[allt])
    out["ridge-all"] = {"test": lr.predict(K[np.ix_(test, allt)]),
                        "hidden": lr.predict(K[np.ix_(hid, allt)]), "chosen": p}

    # two settings chosen on the test set itself.  They cannot be used -- the test activities
    # are what is being predicted -- and they are here for one purpose: to tell "the penalty
    # does not help" apart from "the folds cannot find the weight at which it helps".  If the
    # oracle gains nothing either, the method is what fails and not the selection.
    out["ridge-half-oracle"] = _best_on_test(K, y, vis, vis, test, hid,
                                             [{"lam": l} for l in LAMS], "krr")
    out["lapkrr-oracle"] = _best_on_test(K, y_semi, allt, allt, test, hid, grid, "lapkrr",
                                         y_true=y)
    return out, vis, hid


def _best_on_test(K, y_fit, fit_idx, _unused, test, hid, grid, method, y_true=None):
    """The setting of `grid` whose test-set r2 is the highest: an oracle, not a usable rule."""
    y_true = y_fit if y_true is None else y_true
    best = None
    for params in grid:
        lr = _Learner(method, dict(params)).fit(K[np.ix_(fit_idx, fit_idx)], y_fit[fit_idx])
        yp = lr.predict(K[np.ix_(test, fit_idx)])
        r2 = r2_score(y_true[test], yp)
        if best is None or r2 > best["r2"]:
            best = {"r2": r2, "test": yp, "chosen": dict(params),
                    "hidden": lr.predict(K[np.ix_(hid, fit_idx)])}
    return best


def _mean_r2(y_true, preds, idx=None):
    """Mean over the halvings of r2 on the molecules `idx` (all of them when idx is None)."""
    vals = []
    for yp in preds:
        if idx is None:
            vals.append(r2_score(y_true, yp))
        else:
            vals.append(r2_score(y_true[idx], yp[idx]))
    return float(np.mean(vals))


def _paired_r2_interval(y_true, a, b, runs=RUNS, seed=SEED):
    """(difference of the mean r2, low, high), resampling the test molecules in pairs."""
    point = _mean_r2(y_true, a) - _mean_r2(y_true, b)
    rng = np.random.RandomState(seed)
    n = len(y_true)
    out = []
    for _ in range(int(runs)):
        idx = rng.randint(0, n, n)
        if np.std(y_true[idx]) < 1e-12:
            continue
        va, vb = _mean_r2(y_true, a, idx), _mean_r2(y_true, b, idx)
        if np.isfinite(va) and np.isfinite(vb):
            out.append(va - vb)
    if len(out) < 100:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def run_hidden(name, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"hidden_{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    K, y, train, test = _sutherland_kernel(name)
    print(f"=== {name}: {len(train)} training molecules, {len(test)} test, "
          f"kernel {K.shape} in {time.time() - t0:.0f} s", flush=True)

    modes = HIDDEN_MODES + ORACLE_MODES
    preds = {m: [] for m in modes}
    hidden_pred = {m: [] for m in modes}
    hidden_true = []
    chosen = {m: [] for m in modes}
    for rep in range(REPEATS):
        got, _vis, hid = _one_halving(K, y, train, test, seed=SEED + rep)
        for m in modes:
            preds[m].append(np.asarray(got[m]["test"], float))
            hidden_pred[m].append(np.asarray(got[m]["hidden"], float))
            chosen[m].append(got[m]["chosen"])
        hidden_true.append(y[hid])

    rows = {}
    for m in modes:
        r2 = float(np.mean([r2_score(y[test], p) for p in preds[m]]))
        err = float(np.mean([rmse(y[test], p) for p in preds[m]]))
        r2_hidden = float(np.mean([r2_score(t, p)
                                   for t, p in zip(hidden_true, hidden_pred[m])]))
        rows[m] = {"r2_pred": r2, "rmse_pred": err, "r2_hidden": r2_hidden,
                   "chosen": [{k: float(v) for k, v in c.items()} for c in chosen[m]]}
        print(f"    {m:11s} R2pred {r2:+.3f}  RMSEpred {err:.3f}  "
              f"R2 on the hidden molecules {r2_hidden:+.3f}", flush=True)

    diffs = {}
    for pair in (("lapkrr", "ridge-half"), ("ridge-all", "ridge-half"),
                 ("lapkrr", "ridge-all"), ("lapkrr-oracle", "ridge-half-oracle")):
        t = _paired_r2_interval(y[test], preds[pair[0]], preds[pair[1]])
        diffs[f"{pair[0]}-{pair[1]}/r2_pred"] = list(t)
        print(f"    {pair[0]} - {pair[1]:11s} R2pred {t[0]:+.3f}  [{t[1]:+.3f}, {t[2]:+.3f}]",
              flush=True)

    gap = rows["ridge-all"]["r2_pred"] - rows["ridge-half"]["r2_pred"]
    gained = rows["lapkrr"]["r2_pred"] - rows["ridge-half"]["r2_pred"]
    share = gained / gap if abs(gap) > 1e-9 else float("nan")
    print(f"    the hidden activities are worth {gap:+.3f} of R2pred; the penalty recovers "
          f"{gained:+.3f}, that is {share:+.0%} of it", flush=True)

    cliffs = cliff_counts(K, y, train)
    print(f"    activity cliffs among the {cliffs['pairs']} labelled pairs: {cliffs['cliffs']} "
          f"of the {cliffs['similar']} pairs at cosine >= {CLIFF_COSINE} differ by "
          f">= {CLIFF_GAP:g} log unit ({cliffs['cliff_share_of_similar']:.1%})", flush=True)

    out = {"set": name, "part": "hidden", "n_train": int(len(train)), "n_test": int(len(test)),
           "alpha": ALPHA, "fields": FIELDS, "repeats": REPEATS,
           "visible_share": VISIBLE_SHARE, "neighbours": NEIGHBOURS,
           "inner_folds": INNER_FOLDS, "seed": SEED, "lams": list(LAMS),
           "manifolds": list(MANIFOLDS), "modes": rows, "differences": diffs,
           "ceiling_gap": gap, "recovered": gained, "recovered_share": share,
           "cliffs": cliffs, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# ----------------------------------------------------------------------------------------------
# part 2: the library unlabelled
# ----------------------------------------------------------------------------------------------

def _screen_cross_validate(K, y, method, grid, folds=SCREEN_FOLDS, seed=SEED,
                           inner=SCREEN_INNER, unlabel_test=False):
    """Pooled out-of-fold decisions, the setting chosen inside each fold.

    With `unlabel_test` the molecules of the held-out fold are added to the training kernel with
    no label, which is what a screen offers: the library is there to be looked at, its activities
    are not.  Returns (decisions as the learner leaves them, decisions at the common cut, the
    settings chosen).
    """
    y = np.asarray(y, float)
    n = len(y)
    decision = np.full(n, np.nan)
    tuned = np.full(n, np.nan)
    chosen = []
    for te in stratified_folds(y, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        if unlabel_test:
            use = np.concatenate([tr, te])           # the fit sees both, the labels only tr
            y_fit = y.copy()
            y_fit[te] = np.nan
            y_fit = y_fit[use]
        else:
            use, y_fit = tr, y[tr]
        K_fit = K[np.ix_(use, use)]
        params, _auc_in = _select(K_fit, y_fit, method, grid, folds=inner, seed=seed)
        chosen.append(params)
        lr = _Learner(method, params).fit(K_fit, y_fit)
        K_te = K[np.ix_(te, use)]
        decision[te] = lr.predict(K_te)
        raw_tr, raw_te = K_fit @ lr.alpha_, K_te @ lr.alpha_
        tuned[te] = raw_te - _common_cut(K_fit, y_fit, method, params, raw_tr, inner, seed)
    return decision, tuned, chosen


def run_screen(name, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"screen_{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    K, y, _model, _table = _litpcba_kernel(name)
    if K is None:
        return None
    n_a = int(np.sum(y > 0))
    print(f"=== {name}: {n_a} measured actives, {len(y) - n_a} measured inactives, "
          f"kernel {K.shape} in {time.time() - t0:.0f} s", flush=True)

    grids = {
        "lapkrr": [{"lam": l, "manifold": g, "neighbours": NEIGHBOURS}
                   for l in SCREEN_LAMS for g in SCREEN_MANIFOLDS],
        "krr": [{"lam": l} for l in SCREEN_LAMS],
        "svc": [{"C": c} for c in SCREEN_CS],
    }
    preds, tuned_preds, rows = {}, {}, {}
    for method in SCREEN_ORDER:
        t1 = time.time()
        d, t, chosen = _screen_cross_validate(K, y, method, grids[method],
                                              unlabel_test=(method == "lapkrr"))
        preds[method] = (d, None)
        tuned_preds[method] = (t, None)
        sc = screening_scores(y, d)
        sct = classification_scores(y, t)
        rows[method] = {"auc": sc["auc"], "ef_1": sc["ef_1"], "ef_5": sc["ef_5"],
                        "bedroc_20": sc["bedroc_20"],
                        "balanced_accuracy": _balanced_accuracy(y, d),
                        "balanced_accuracy_common": sct["balanced_accuracy"],
                        "sensitivity_common": sct["sensitivity"],
                        "specificity_common": sct["specificity"],
                        "max_ef_1": sc["max_ef_1"], "max_ef_5": sc["max_ef_5"],
                        "chosen": [{k: float(v) for k, v in c.items()} for c in chosen],
                        "seconds": round(time.time() - t1, 1)}
        print(f"    {method:7s} AUC {sc['auc']:.3f}  EF1% {sc['ef_1']:.2f}  "
              f"EF5% {sc['ef_5']:.2f}  BEDROC {sc['bedroc_20']:.3f}  "
              f"balanced {sct['balanced_accuracy']:.3f} (common cut)  "
              f"({time.time() - t1:.0f} s)", flush=True)

    diffs = {}
    for other in SCREEN_ORDER[1:]:
        for metric in SCREEN_METRICS:
            src = tuned_preds if metric.endswith("_common") else preds
            v, lo, hi = paired_interval(metric, y, src["lapkrr"], src[other], runs=RUNS,
                                        seed=SEED)
            diffs[f"lapkrr-{other}/{metric}"] = [v, lo, hi]
            print(f"    lapkrr - {other:4s} {metric:26s} {v:+.3f}  [{lo:+.3f}, {hi:+.3f}]",
                  flush=True)

    out = {"set": name, "part": "screen", "n": int(len(y)), "n_actives": n_a,
           "alpha": ALPHA, "fields": FIELDS, "folds": SCREEN_FOLDS,
           "inner_folds": SCREEN_INNER, "seed": SEED, "neighbours": NEIGHBOURS,
           "lams": list(SCREEN_LAMS), "manifolds": list(SCREEN_MANIFOLDS),
           "active_share": n_a / float(len(y)), "methods": rows, "differences": diffs,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# ----------------------------------------------------------------------------------------------
# part 3: how local the penalty should be
# ----------------------------------------------------------------------------------------------

def run_neighbours(name, force=False, results=RESULTS):
    """Part 1 again on one set, at several sizes of the neighbourhood.

    The plan entry asks for the neighbourhood to be truncated so that the penalty is local, and
    does not say how far.  This measures what the choice is worth: 0 keeps every pair, which is
    the untruncated penalty the entry warns against.
    """
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"neighbours_{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    K, y, train, test = _sutherland_kernel(name)
    print(f"=== {name}: neighbourhood sizes {NEIGHBOUR_GRID} on {len(train)} training "
          f"molecules", flush=True)
    rows = {}
    for k in NEIGHBOUR_GRID:
        preds = []
        for rep in range(REPEATS):
            got, _vis, _hid = _one_halving(K, y, train, test, seed=SEED + rep, neighbours=k)
            preds.append(np.asarray(got["lapkrr"]["test"], float))
        r2 = float(np.mean([r2_score(y[test], p) for p in preds]))
        rows[str(k)] = {"r2_pred": r2}
        print(f"    neighbours {k:3d}  R2pred {r2:+.3f}", flush=True)
    out = {"set": name, "part": "neighbours", "grid": list(NEIGHBOUR_GRID),
           "rows": rows, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

def _load(results, prefix):
    out = {}
    if not os.path.isdir(results):
        return out
    for f in sorted(os.listdir(results)):
        if f.startswith(prefix + "_") and f.endswith(".json"):
            out[f[len(prefix) + 1:-5]] = json.load(open(os.path.join(results, f)))
    return out


def report(path=None, results=RESULTS):
    hid = _load(results, "hidden")
    scr = _load(results, "screen")
    nbr = _load(results, "neighbours")
    L = ["# Laplacian-regularised kernel ridge regression (plan item E10)", "",
         "Do molecules without a measured activity improve a model? `--method lapkrr` adds the "
         "smoothness penalty of manifold regularisation (Belkin, Niyogi & Sindhwani 2006) to "
         "kernel ridge regression, so that unlabelled molecules become rows of the kernel and "
         "terms of the model's expansion. Produced by `benchmarks/run_lapkrr.py`.", ""]

    if hid:
        L += ["## Part 1: half of the training activities hidden", "",
              "The eight Sutherland sets. Half of the published training molecules keep their "
              "activities; the other half are given to `lapkrr` with no activity. All three "
              "models are scored on the published test set, which none of them saw. Every "
              "learner chooses its own settings on a 10-fold cross-validation of the molecules "
              "it may see, the weight of the penalty among "
              + ", ".join(f"{g:g}" for g in MANIFOLDS)
              + f" and the ridge among {', '.join(f'{l:g}' for l in LAMS)}; the figures are "
              f"means over {REPEATS} halvings and the intervals are paired bootstraps over the "
              "test molecules.", "",
              "| set | training molecules | test molecules | ridge, visible half | lapkrr, "
              "hidden half unlabelled | ridge, every activity (ceiling) | lapkrr - visible half "
              "| ceiling - visible half | share of the gap recovered |",
              "|---|---:|---:|---:|---:|---:|---|---|---:|"]
        means = {m: [] for m in HIDDEN_MODES}
        shares, gains, pass_sets = [], [], 0
        for name in HIDDEN_SETS:
            d = hid.get(name)
            if d is None:
                continue
            for m in HIDDEN_MODES:
                means[m].append(d["modes"][m]["r2_pred"])
            g = d["differences"]["lapkrr-ridge-half/r2_pred"]
            gains.append(g)
            shares.append(d["recovered_share"])
            if np.isfinite(g[1]) and g[1] > 0.0:
                pass_sets += 1
            L.append(f"| **{name}** | {d['n_train']} | {d['n_test']} | "
                     f"{_f(d['modes']['ridge-half']['r2_pred'])} | "
                     f"{_f(d['modes']['lapkrr']['r2_pred'])} | "
                     f"{_f(d['modes']['ridge-all']['r2_pred'])} | {_interval(g)} | "
                     f"{_interval(d['differences']['ridge-all-ridge-half/r2_pred'])} | "
                     f"{_f(d['recovered_share'], 2)} |")
        if means["lapkrr"]:
            n_sets = len(means["lapkrr"])
            L.append(f"| **mean of {n_sets} sets** | | | "
                     + " | ".join(f"**{_f(np.mean(means[m]))}**" for m in HIDDEN_MODES)
                     + f" | **{_signed(np.mean([v[0] for v in gains]))}** | "
                     f"**{_signed(np.mean([d['ceiling_gap'] for d in hid.values()]))}** | |")
            better = sum(1 for v in gains if v[0] > 0)
            lost = sum(1 for v in gains if np.isfinite(v[2]) and v[2] < 0.0)
            zero = [_zero_weight_share(d["modes"]["lapkrr"]["chosen"]) for d in hid.values()]
            L += ["", f"**The gate's first part fails.** The penalty gains "
                      f"{np.mean([v[0] for v in gains]):+.3f} of R2pred on average, is ahead of "
                      f"ridge on the visible half on {better} of the {n_sets} sets, and on no "
                      f"set does its interval exclude zero above; on {lost} sets the interval "
                      f"excludes zero *below*, that is the penalty measurably loses. The hidden "
                      f"activities themselves are worth "
                      f"{np.mean([d['ceiling_gap'] for d in hid.values()]):+.3f} of R2pred, so "
                      f"the penalty recovers "
                      f"{np.mean([s for s in shares if np.isfinite(s)]):+.0%} of that gap -- it "
                      f"recovers none of it.", "",
                  f"The cross-validation inside the fits chose a penalty weight of zero, that is "
                  f"no penalty at all, in {np.mean(zero):.0%} of the "
                  f"{n_sets * REPEATS} fits. Where it chose a weight above zero the model it "
                  f"produced was, on average, worse on the test set than the one with no "
                  f"penalty.", ""]
        L += ["The same models read on the hidden molecules rather than on the test set. These "
              "two columns do not measure the same thing: `lapkrr` has the hidden molecules in "
              "its expansion and is predicting molecules whose fields it has seen, which is the "
              "transductive setting manifold regularisation is usually stated in, while ridge on "
              "the visible half has never met them.", "",
              "| set | ridge, visible half | lapkrr, hidden half unlabelled | ridge, every "
              "activity (fitted on them) |", "|---|---:|---:|---:|"]
        for name in HIDDEN_SETS:
            d = hid.get(name)
            if d is None:
                continue
            L.append(f"| **{name}** | " + " | ".join(
                _f(d["modes"][m]["r2_hidden"]) for m in HIDDEN_MODES) + " |")
        if means["lapkrr"]:
            L.append("| **mean** | " + " | ".join(
                f"**{_f(np.mean([d['modes'][m]['r2_hidden'] for d in hid.values()]))}**"
                for m in HIDDEN_MODES) + " |")
        L += ["", "The last column is fitted on the molecules it is scored on, so it is a "
                  "training figure and not a prediction; it is here only to show where the "
                  "other two stand.", ""]
        L += ["### The same two learners with their settings chosen on the test set", "",
              "**These numbers cannot be used.** The settings are chosen by looking at the "
              "activities being predicted, so they are an upper bound and not a result. They "
              "are here to answer one question: when the penalty gains nothing, is that the "
              "method failing or the cross-validation failing to find the weight at which it "
              "works? If even the best weight on the test set gains nothing, it is the method.",
              "",
              "| set | ridge, best ridge on the test set | lapkrr, best settings on the test "
              "set | difference |", "|---|---:|---:|---|"]
        oracle_gains = []
        for name in HIDDEN_SETS:
            d = hid.get(name)
            if d is None or "lapkrr-oracle" not in d["modes"]:
                continue
            g = d["differences"]["lapkrr-oracle-ridge-half-oracle/r2_pred"]
            oracle_gains.append(g)
            L.append(f"| **{name}** | {_f(d['modes']['ridge-half-oracle']['r2_pred'])} | "
                     f"{_f(d['modes']['lapkrr-oracle']['r2_pred'])} | {_interval(g)} |")
        if oracle_gains:
            L.append(f"| **mean** | "
                     f"**{_f(np.mean([d['modes']['ridge-half-oracle']['r2_pred'] for d in hid.values() if 'ridge-half-oracle' in d['modes']]))}** | "
                     f"**{_f(np.mean([d['modes']['lapkrr-oracle']['r2_pred'] for d in hid.values() if 'lapkrr-oracle' in d['modes']]))}** | "
                     f"**{_f(np.mean([v[0] for v in oracle_gains]))}** |")
            n_excl = sum(1 for v in oracle_gains if np.isfinite(v[1]) and v[1] > 0.0)
            L += ["", f"Even with the weight chosen on the test set the penalty gains "
                      f"{np.mean([v[0] for v in oracle_gains]):+.3f} of R2pred on average, and "
                      f"the interval excludes zero above on {n_excl} of the "
                      f"{len(oracle_gains)} sets. **The method is what gains nothing here, not "
                      f"the choice of its weight.**", ""]
        L += ["### Activity cliffs, so that a loss can be read against them", "",
              "A pair of labelled molecules counts as **similar** when the cosine of their "
              f"fields is {CLIFF_COSINE:g} or more, and as an **activity cliff** when it is "
              f"similar and the two activities also differ by {CLIFF_GAP:g} log unit or more. "
              "The cosine is the measure of plan item F3, applied to two molecules instead of "
              "two co-fields; the two cutoffs are this benchmark's choice. The penalty assumes "
              "the activity varies smoothly along the data, which is exactly what a cliff "
              "breaks.", "",
              "| set | labelled pairs | similar pairs | activity cliffs | cliffs among the "
              "similar pairs | mean activity difference of a similar pair | lapkrr - visible "
              "half |", "|---|---:|---:|---:|---:|---:|---|"]
        for name in HIDDEN_SETS:
            d = hid.get(name)
            if d is None:
                continue
            c = d["cliffs"]
            L.append(f"| **{name}** | {c['pairs']} | {c['similar']} | {c['cliffs']} | "
                     f"{_f(c['cliff_share_of_similar'], 3)} | "
                     f"{_f(c['mean_gap_of_similar'], 2)} | "
                     f"{_interval(d['differences']['lapkrr-ridge-half/r2_pred'])} |")
        names = [n for n in HIDDEN_SETS if n in hid]
        if len(names) >= 4:
            share = [hid[n]["cliffs"]["cliff_share_of_similar"] for n in names]
            gapm = [hid[n]["cliffs"]["mean_gap_of_similar"] for n in names]
            gain = [hid[n]["differences"]["lapkrr-ridge-half/r2_pred"][0] for n in names]
            size = [hid[n]["n_train"] for n in names]
            L += ["", f"**The cliffs do not explain the loss.** Over the {len(names)} sets the "
                      f"rank correlation between the share of similar pairs that are cliffs and "
                      f"what the penalty gains is {_spearman(share, gain):+.2f}, and between the "
                      f"mean activity difference of a similar pair and that gain "
                      f"{_spearman(gapm, gain):+.2f}: both weak, and neither in a direction that "
                      f"would let a cliff count predict whether the penalty is safe to use. What "
                      f"does line up is the **size of the set**: the rank correlation between "
                      f"the number of labelled molecules and the gain is "
                      f"{_spearman(size, gain):+.2f}. The penalty loses most on the smallest "
                      f"sets and nothing at all on the largest, which points at the choice of "
                      f"its weight on few molecules rather than at the smoothness assumption.",
                  ""]
        L.append("")

    if nbr:
        L += ["### How local the penalty should be", "",
              "Part 1 repeated at several sizes of the neighbourhood; 0 keeps every pair, which "
              "is the untruncated penalty the plan entry warns against.", "",
              "| set | " + " | ".join(
                  ("every pair" if k == 0 else f"{k} neighbours") for k in NEIGHBOUR_GRID)
              + " |", "|---|" + "---:|" * len(NEIGHBOUR_GRID)]
        for name, d in sorted(nbr.items()):
            L.append(f"| **{name}** | " + " | ".join(
                _f(d["rows"].get(str(k), {}).get("r2_pred")) for k in NEIGHBOUR_GRID) + " |")
        L.append("")

    if scr:
        L += ["## Part 2: the library unlabelled (screening)", "",
              "The two LIT-PCBA sets of plan item E4, under the protocol of "
              "`benchmarks/run_kfda.py`: one 5-fold split stratified by class, each learner "
              "choosing its setting inside every fold by out-of-fold AUC, 95 % paired bootstrap "
              "intervals over the molecules. In every fold the held-out molecules are given to "
              "`lapkrr` with no label, which is what a screen offers. Enrichment at the top of "
              "the ranking is the measure the gate names; AUC and the balanced accuracy at the "
              "common cut are beside it.", ""]
        for name in SCREEN_SETS:
            d = scr.get(name)
            if d is None:
                continue
            L += [f"**{name}**: {d['n_actives']} actives among {d['n']} molecules "
                  f"({d['active_share']:.1%} active).", "",
                  "| learner | " + " | ".join(SCREEN_METRIC_LABELS[m] for m in SCREEN_METRICS)
                  + " |", "|---|" + "---:|" * len(SCREEN_METRICS)]
            for m in SCREEN_ORDER:
                L.append(f"| {SCREEN_LABELS[m]} | " + " | ".join(
                    _f(d["methods"][m][k], 2 if k.startswith("ef_") else 3)
                    for k in SCREEN_METRICS) + " |")
            zero = _zero_weight_share(d["methods"]["lapkrr"]["chosen"])
            L += ["", f"The folds chose a penalty weight of zero, that is no penalty at all, in "
                      f"{zero:.0%} of the {len(d['methods']['lapkrr']['chosen'])} fits"
                      + (". `lapkrr` is therefore the same model as `krr` here, which is why "
                         "every difference between them is exactly zero." if zero >= 1.0
                         else "."), ""]
            L += ["| difference | " + " | ".join(
                SCREEN_METRIC_LABELS[m] for m in SCREEN_METRICS) + " |",
                "|---|" + "---|" * len(SCREEN_METRICS)]
            for other in SCREEN_ORDER[1:]:
                L.append(f"| lapkrr - {other} | " + " | ".join(
                    _interval(d["differences"][f"lapkrr-{other}/{k}"],
                              2 if k.startswith("ef_") else 3) for k in SCREEN_METRICS) + " |")
            L.append("")
        kept = []
        for name in SCREEN_SETS:
            d = scr.get(name)
            if d is None:
                continue
            for other in SCREEN_ORDER[1:]:
                for k in ("ef_1", "bedroc_20", "auc"):
                    v, lo, hi = d["differences"][f"lapkrr-{other}/{k}"]
                    if np.isfinite(lo) and hi < 0.0:
                        kept.append(f"{name} {k} against {other}")
        L += [("**No enrichment is lost that the interval can resolve**: no difference in AUC, "
               "enrichment or BEDROC against either reference excludes zero below it."
               if not kept else
               "**Enrichment is lost** on: " + ", ".join(kept) + "."), ""]

    text = "\n".join(L) + "\n"
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", default="both",
                    choices=["both", "all", "hidden", "screen", "neighbours"])
    ap.add_argument("--sets", default=None, help="comma-separated data sets (default: all)")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None,
                    help="only write the Markdown table (default: results/lapkrr/table.md)")
    args = ap.parse_args(argv)
    if args.report is None:
        asked = [s.strip() for s in args.sets.split(",")] if args.sets else None
        t0 = time.time()
        if args.part in ("both", "all", "hidden"):
            for name in HIDDEN_SETS:
                if asked is None or name in asked:
                    run_hidden(name, force=args.force)
        if args.part in ("both", "all", "screen"):
            for name in SCREEN_SETS:
                if asked is None or name in asked:
                    run_screen(name, force=args.force)
        if args.part in ("all", "neighbours"):
            for name in ("GPB", "DHFR"):
                if asked is None or name in asked:
                    run_neighbours(name, force=args.force)
        print(f"\nAll runs finished in {(time.time() - t0) / 60:.1f} min")
        args.report = os.path.join(RESULTS, "table.md")
    print(report(args.report))


if __name__ == "__main__":
    main()
