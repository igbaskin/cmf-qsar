#!/usr/bin/env python
"""
Which hops need the DFT calculation?  The tolerance rule of plan item MS1 on nebDFT2k.

``run_nebdft.py`` fits the hop models on barriers from density functional theory (DFT).  Their
mean absolute error, 0.22 to 0.29 eV, is too large for a number to quote.  In a hierarchy of
methods a model does not have to answer every hop: it has to say which hops it may answer and
which have to go to the calculation.  This script measures how well the models of
``run_nebdft.py`` can say that.

The three levels
----------------
- The *coarse* level is the bond-valence barrier of the hop, which costs nothing and is in the
  index of the set.
- The *model* reads the fields of the rigid structure and, in two of its forms, the
  bond-valence barrier too.
- The *fine* level is the DFT calculation of the migration path.  Here its result is known for
  every hop, so every decision can be judged.

The standard deviation of a prediction
--------------------------------------
For kernel ridge it is the one of `cmfqsar/surrogate.py`: the Gaussian process that matches
the ridge fit, with its scale in eV.  For a random forest it is the spread of the predictions
of its trees.  Each is checked on the hops that the fit has not seen: the errors divided by
the standard deviations should have a root mean square of one.  The factor that makes it one
is measured inside the training part, on folds that hold out whole structures, and applied.

The rule
--------
A hop whose standard deviation is above a tolerance goes to the fine level.  Two tolerances
are used: the error that the model shows inside its training part, and the tolerance that
sends a fixed share of the hops to the calculation.  The standard deviation is compared, as
the signal that orders the hops, with three other signals that cost nothing: the predicted
barrier itself, the disagreement of the model with the model on the bond-valence barrier
alone, and a random order.  An order by the true error is the best that any signal could do.

Usage
-----
    python benchmarks/run_nebdft.py           # first: it builds the kernels (37 minutes)
    python benchmarks/run_nebdft_rule.py      # this benchmark, then the table
    python benchmarks/run_nebdft_rule.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.surrogate import (Calibration, Handoff, Surrogate, cross_validate,  # noqa: E402
                               tolerance_for_budget)
import run_nebdft as N  # noqa: E402

RESULTS = N.RESULTS
FOLDS = 10
INNER = 4
FOREST_INNER = 3
SEED = 0
BUDGETS = (0.1, 0.2, 0.3, 0.5)             # the share of the hops sent to the calculation
ARMS = ("bvse", "fields", "fields+bvse", "forest")
ARM_NAMES = {
    "bvse": "kernel ridge on the bond-valence barrier alone",
    "fields": "kernel ridge, fields in the frame of the hop",
    "fields+bvse": "kernel ridge, fields in the frame of the hop, with the bond-valence "
                   "barrier",
    "forest": "random forest, invariant field about the midpoint, with the bond-valence "
              "barrier"}
SIGNALS = ("own standard deviation", "predicted barrier",
           "disagreement with the model on the bond-valence barrier", "random order",
           "true error (the best possible)")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The models with a standard deviation
# --------------------------------------------------------------------------- #
def absolute_ridge(K_tr: np.ndarray, relative: float) -> float:
    """A ridge of :data:`run_nebdft.RIDGES`, which is a multiple of the mean diagonal of the
    centred training kernel, as the number that is added to the diagonal."""
    mu = K_tr.mean(axis=0)
    centred_trace = float(np.trace(K_tr) - 2.0 * mu.sum() + K_tr.mean() * len(K_tr))
    return float(relative) * max(centred_trace / len(K_tr), 1e-300)


def chosen(pool: Dict[str, Callable[[], np.ndarray]], y: np.ndarray, groups: np.ndarray,
           tr: np.ndarray, inner: Sequence[np.ndarray]) -> Tuple[str, float]:
    """The kernel of `pool` and the ridge with the smallest error inside the rows `tr`;
    `inner` holds folds as positions in `tr`."""
    best = None
    for name, make in pool.items():
        K = make()
        err = np.zeros(len(N.RIDGES))
        for f in inner:
            te_in = tr[f]
            tr_in = np.setdiff1d(tr, te_in)
            P = N.ridge_path(K, y, tr_in, te_in)
            err += np.sum((P - y[te_in][None, :]) ** 2, axis=1)
        k = int(np.argmin(err))
        if best is None or err[k] < best[0]:
            best = (float(err[k]), name, N.RIDGES[k])
    return best[1], best[2]


def kernel_fold(pool: Dict[str, Callable[[], np.ndarray]], y: np.ndarray, groups: np.ndarray,
                tr: np.ndarray, te: np.ndarray, rng) -> dict:
    """One fold of a kernel model: the prediction and its standard deviation for the rows
    `te`, the factor that calibrates the deviation, and the error inside the training
    part."""
    inner = N.group_folds(groups[tr], INNER, rng)
    name, relative = chosen(pool, y, groups, tr, inner)
    K = pool[name]()
    K_tr = K[np.ix_(tr, tr)]
    lam = absolute_ridge(K_tr, relative)
    m_in, s_in = cross_validate(K_tr, y[tr], lam, parts=inner)
    inside = Calibration.of(y[tr], m_in, s_in)
    s = Surrogate.fit(K_tr, y[tr], lam)
    mean, sd = s.predict(K[np.ix_(te, tr)], np.diag(K)[te])
    return {"mean": mean, "sd": sd, "factor": inside.factor, "claimed": inside.rmse,
            "inner_z": inside.z_rms, "setting": name}


def forest_spread(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                  seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """The prediction of a random forest for the rows `te` and the standard deviation of the
    predictions of its trees."""
    from sklearn.ensemble import RandomForestRegressor

    m = RandomForestRegressor(n_estimators=N.FOREST_TREES, n_jobs=-1, random_state=seed)
    m.fit(X[tr], y[tr])
    trees = np.array([t.predict(X[te]) for t in m.estimators_])
    return trees.mean(axis=0), np.maximum(trees.std(axis=0), 1e-6)


def forest_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, tr: np.ndarray,
                te: np.ndarray, rng) -> dict:
    """One fold of the forest, as :func:`kernel_fold` returns it."""
    m_in, s_in = np.full(len(tr), np.nan), np.full(len(tr), np.nan)
    for f in N.group_folds(groups[tr], FOREST_INNER, rng):
        rest = np.setdiff1d(np.arange(len(tr)), f)
        m_in[f], s_in[f] = forest_spread(X, y, tr[rest], tr[f])
    inside = Calibration.of(y[tr], m_in, s_in)
    mean, sd = forest_spread(X, y, tr, te)
    return {"mean": mean, "sd": sd, "factor": inside.factor, "claimed": inside.rmse,
            "inner_z": inside.z_rms, "setting": "forest"}


# --------------------------------------------------------------------------- #
# Reading a rule
# --------------------------------------------------------------------------- #
def answered_error(error: np.ndarray, signal: np.ndarray, share: float) -> dict:
    """The errors of the hops that are answered when the `share` of the hops with the largest
    `signal` goes to the calculation."""
    n = len(error)
    k = int(round(share * n))
    order = np.argsort(-signal, kind="stable")
    out, kept = order[:k], order[k:]
    return {"mae_answered": float(np.mean(np.abs(error[kept]))),
            "rmse_answered": float(np.sqrt(np.mean(error[kept] ** 2))),
            "mae_deferred": float(np.mean(np.abs(error[out]))) if k else float("nan")}


def signals(error: np.ndarray, sd: np.ndarray, pred: np.ndarray,
            coarse: Optional[np.ndarray], rng) -> Dict[str, np.ndarray]:
    """The orders in which the hops can be sent to the calculation (:data:`SIGNALS`)."""
    out = {SIGNALS[0]: sd, SIGNALS[1]: pred, SIGNALS[3]: rng.random(len(error)),
           SIGNALS[4]: np.abs(error)}
    if coarse is not None:
        out[SIGNALS[2]] = np.abs(pred - coarse)
    return out


def needed_share(error: np.ndarray, signal: np.ndarray, target: float) -> float:
    """The smallest share of the hops, in steps of one per cent, that has to go to the
    calculation so that the mean absolute error of the answered hops is at most `target`;
    1.0 when no share reaches it."""
    order = np.argsort(-signal, kind="stable")
    e = np.abs(error[order])
    n = len(e)
    for pct in range(0, 100):
        k = int(round(pct / 100.0 * n))
        if e[k:].mean() <= target:
            return pct / 100.0
    return 1.0


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def run(results: str = RESULTS, max_structures: int = 0, jobs: int = 8, folds: int = FOLDS,
        tag: str = "", cache: str = N.CACHE) -> dict:
    t0 = time.time()
    hs, doc, bvse, split = N.load(N.PREPARED, max_structures)
    t = hs.targets()
    y = np.asarray(t.y, float)
    groups = np.asarray(t.molecule, int)
    bvse = bvse[np.asarray(t.atom, int)]
    kernels, rows = N.descriptions(hs, jobs, cache, tag)
    z = (bvse - bvse.mean()) / max(float(bvse.std()), 1e-12)
    pools = {"bvse": N.variants(N.unit_scale(np.outer(z, z) + 1.0)),
             "fields": N.variants(N.unit_scale(kernels["overlap"])),
             "fields+bvse": N.with_bvse(kernels["overlap"], bvse)}
    X = np.hstack([rows["invariant"], bvse[:, None]])
    rng = np.random.default_rng(SEED)
    parts = N.group_folds(groups, folds, rng)
    n = len(y)
    pred = {a: np.full(n, np.nan) for a in ARMS}
    sd = {a: np.full(n, np.nan) for a in ARMS}
    corrected = {a: np.full(n, np.nan) for a in ARMS}
    notes: Dict[str, List[dict]] = {a: [] for a in ARMS}
    for k, te in enumerate(parts):
        tr = np.setdiff1d(np.arange(n), te)
        for arm in ARMS:
            f = forest_fold(X, y, groups, tr, te, rng) if arm == "forest" \
                else kernel_fold(pools[arm], y, groups, tr, te, rng)
            pred[arm][te], sd[arm][te] = f["mean"], f["sd"]
            corrected[arm][te] = f["sd"] * f["factor"]
            notes[arm].append({"claimed": float(f["claimed"]), "factor": float(f["factor"]),
                               "inner_z": float(f["inner_z"]), "setting": f["setting"]})
        log(f"  fold {k + 1} of {len(parts)}, {time.time() - t0:.0f} s")
    out = {"hops": n, "structures": int(len(np.unique(groups))), "folds": len(parts),
           "y_sd": float(y.std(ddof=1)), "budgets": list(BUDGETS), "arms": {}}
    for arm in ARMS:
        error = pred[arm] - y
        raw = Calibration.of(y, pred[arm], sd[arm])
        cal = Calibration.of(y, pred[arm], corrected[arm])
        claimed = float(np.mean([f["claimed"] for f in notes[arm]]))
        rule = Handoff.of(corrected[arm], claimed, y=y, mean=pred[arm])
        coarse = None if arm == "bvse" else pred["bvse"]
        orders = signals(error, corrected[arm], pred[arm], coarse, np.random.default_rng(SEED))
        target = 0.5 * float(np.mean(np.abs(error)))
        out["arms"][arm] = {
            "mae": float(np.mean(np.abs(error))), "rmse": raw.rmse,
            "mean_sd": raw.mean_sd, "z_rms_raw": raw.z_rms,
            "factor": float(np.mean([f["factor"] for f in notes[arm]])),
            "mean_sd_corrected": cal.mean_sd, "z_rms": cal.z_rms,
            "inside_1sd": cal.coverage_1sd, "inside_2sd": cal.coverage_2sd,
            "spearman": cal.spearman, "claimed": claimed,
            "rule": {"tolerance": rule.tolerance, "answered": rule.fraction_answered,
                     "rmse_answered": rule.rmse_answered, "rmse_deferred": rule.rmse_deferred,
                     "mae_answered": float(np.mean(np.abs(error[corrected[arm] <= claimed])))
                     if np.any(corrected[arm] <= claimed) else float("nan")},
            "budgets": {
                f"{b:g}": dict(answered_error(error, corrected[arm], b),
                               tolerance=tolerance_for_budget(corrected[arm], b))
                for b in BUDGETS},
            "signals": {name: {f"{b:g}": answered_error(error, s, b)["mae_answered"]
                               for b in BUDGETS} for name, s in orders.items()},
            "share_for_half_the_error": {name: needed_share(error, s, target)
                                         for name, s in orders.items()},
            "settings": sorted({f["setting"] for f in notes[arm]})}
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, f"rule{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report(results: str = RESULTS, tag: str = "") -> str:
    """The table of the benchmark as text; it is also written to ``rule.md``."""
    path = os.path.join(results, f"rule{tag}.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    arms = d["arms"]
    lines = ["# Which hops need the DFT calculation? The tolerance rule on nebDFT2k",
             "",
             f"{d['hops']} hops in {d['structures']} structures, {d['folds']} folds that "
             "hold out whole structures. The fine level is the DFT barrier, with a standard "
             f"deviation of {d['y_sd']:.3f} eV over the hops. All errors are in eV.",
             "",
             "## Does the standard deviation mean what it says?",
             "",
             "The root mean square of error / standard deviation should be 1, and 95 % of the "
             "barriers should be inside two standard deviations. The factor is measured "
             "inside the training part and applied.",
             "",
             "| Model | Mean absolute error | RMSE | Root mean square of error / sd, as "
             "computed | Factor applied | After the factor | Inside 2 sd | Rank correlation "
             "of sd and absolute error |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        a = arms[arm]
        lines.append(f"| {ARM_NAMES[arm]} | {a['mae']:.3f} | {a['rmse']:.3f} | "
                     f"{a['z_rms_raw']:.2f} | {a['factor']:.2f} | {a['z_rms']:.2f} | "
                     f"{100 * a['inside_2sd']:.0f} % | {a['spearman']:+.2f} |")
    lines += ["",
              "## The rule with the tolerance at the error that the model shows inside its "
              "training part",
              "",
              "| Model | Tolerance | Hops answered | Mean absolute error of the answered "
              "hops | RMSE of the answered hops | RMSE of the hops sent to the calculation |",
              "|---|---|---|---|---|---|"]
    for arm in ARMS:
        r = arms[arm]["rule"]
        lines.append(f"| {ARM_NAMES[arm]} | {r['tolerance']:.3f} | {100 * r['answered']:.0f} % "
                     f"| {r['mae_answered']:.3f} | {r['rmse_answered']:.3f} | "
                     f"{r['rmse_deferred']:.3f} |")
    budgets = [f"{b:g}" for b in d["budgets"]]
    lines += ["",
              "## The mean absolute error of the answered hops, by the share sent to the "
              "calculation",
              "",
              "The hops with the largest standard deviation go first.",
              "",
              "| Model | None sent | " + " | ".join(f"{100 * float(b):.0f} % sent"
                                                    for b in budgets) + " |",
              "|---|---|" + "---|" * len(budgets)]
    for arm in ARMS:
        a = arms[arm]
        lines.append(f"| {ARM_NAMES[arm]} | {a['mae']:.3f} | "
                     + " | ".join(f"{a['budgets'][b]['mae_answered']:.3f}" for b in budgets)
                     + " |")
    lines += ["",
              "## What orders the hops best",
              "",
              "The mean absolute error of the answered hops when 20 % and 50 % of the hops "
              "are sent to the calculation in the order of a signal, and the share that has "
              "to be sent so that the error of the answered hops is half of the error of all "
              "hops.",
              ""]
    for arm in ("fields+bvse", "forest"):
        a = arms[arm]
        lines += [f"{ARM_NAMES[arm]} (mean absolute error of all hops {a['mae']:.3f}):",
                  "",
                  "| Signal that orders the hops | 20 % sent | 50 % sent | Share to halve the "
                  "error |",
                  "|---|---|---|---|"]
        for name in SIGNALS:
            if name in a["signals"]:
                share = a["share_for_half_the_error"][name]
                lines.append(f"| {name} | {a['signals'][name]['0.2']:.3f} | "
                             f"{a['signals'][name]['0.5']:.3f} | "
                             + ("not reached" if share >= 1.0 else f"{100 * share:.0f} %")
                             + " |")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(results, "rule.md" if not tag else f"rule{tag}.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--max-structures", type=int, default=0,
                    help="read only the first N structures (a quick look)")
    ap.add_argument("--jobs", type=int, default=8, help="processes for the overlap kernel")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)
    tag = f"_{a.max_structures}" if a.max_structures else ""
    if not a.report:
        run(a.results, max_structures=a.max_structures, jobs=a.jobs, tag=tag)
    print(report(a.results, tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
