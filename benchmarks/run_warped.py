#!/usr/bin/env python
"""
Does a monotone transform of the activity, learned with the GP, give better intervals? (ND11)

A Gaussian process assumes the activity, less what the kernel explains, is Gaussian noise on one
scale; a pIC50 series often is not. ``cmfqsar/warp.py`` learns a monotone transform of the
response together with the process (Snelson, Ghahramani and Rasmussen, NIPS 2003), models the
transformed activity, and carries every prediction and interval back through the inverse. This
measures what that is worth, on the eight Sutherland sets and on the 38 sets of the Mittal
collection, five random 70/30 splits each, with alpha and lambda chosen by the evidence on the
training part as ``run_errorpred.py`` chooses them.

Three things are compared on the held-out molecules, at one interval level, between the warped
process, the plain process and the conformal interval of the plain fit:

1. **q2** of the point prediction (the median, for the warped process);
2. **coverage** of the central interval -- the share of held-out activities it holds, against
   the level it promises;
3. **width** of that interval, as a mean over the held-out molecules.

With them the skewness of the training activities and the evidence the warp gained over the
identity transform, so that the gain can be read against what the series looked like. The gate
of the item is read on the differences over the collection, with a paired bootstrap over the
sets as the interval of rule 4: a series whose activities are already close to normal must lose
nothing, and the warp becomes the default only if coverage improves on more series than it
worsens.

Usage
-----
    python benchmarks/run_warped.py                 # the Sutherland sets
    python benchmarks/run_warped.py --mittal        # the 38 sets of the collection
    python benchmarks/run_warped.py --report        # the table from the stored results
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

from cmfqsar.conformal import calibrate, coverage  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import cv_predict, r2_score  # noqa: E402
from cmfqsar.surrogate import Surrogate  # noqa: E402
from cmfqsar.warp import cross_validate_warped, fit_warp  # noqa: E402
from run_errorpred import DATA, FIELDS, SETS, _choose, _kernels  # noqa: E402
from run_topomer import MITTAL  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "warped")
LEVEL = 0.9             # the central interval every path is asked for
TERMS = 2               # tanh terms of the warp
REPEATS = 5
SEED = 0


def _skewness(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    s = float(np.std(y, ddof=1))
    return float(np.mean(((y - y.mean()) / s) ** 3)) if s > 0 else 0.0


def _interval_row(y_true, pred, lo, hi) -> dict:
    cov = coverage(y_true, lo, hi)
    return {"q2": float(r2_score(y_true, pred)), "coverage": float(cov["coverage"]),
            "width": float(cov["mean_width"]), "n": int(len(y_true))}


def _say(s: str) -> None:
    """Print and flush, so that a run whose output goes to a file shows where it is."""
    print(s, flush=True)


def run_set(name: str, data: str, results: str, repeats: int = REPEATS, force: bool = False,
            log=_say) -> dict:
    """One set: the three paths on every split, in the original units of the activity."""
    from scipy.stats import norm

    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        log(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    n = len(y)
    if n < 20:
        log(f"{name}: {n} molecules, too few for a 70/30 split with intervals; skipped")
        return {}
    log(f"=== {name}: {n} molecules, skewness {_skewness(y):+.2f}")
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="file")
    mf = [build_fields(m, fs, nm) for m, nm in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    cache = _kernels(table)
    k = float(norm.ppf(0.5 + 0.5 * LEVEL))
    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(rep)
        order = rng.permutation(n)
        n_te = max(5, int(round(0.3 * n)))
        te, tr = np.sort(order[:n_te]), np.sort(order[n_te:])
        try:
            alpha, lam = _choose(cache, y[tr], tr)
        except TypeError:
            # the evidence was finite for no attenuation and ridge: a degenerate kernel on this
            # split, which the exact arm of run_errorpred never met on the Sutherland sets
            log(f"  split {rep}: the evidence chose nothing (no finite value on any attenuation "
                f"and ridge); the set is left out")
            return {}
        Kf, _sd = cache[alpha]
        Ktr = Kf[:, tr][:, :, tr]
        # the model's own rule: a field that is zero for every training molecule gets a scale
        # of zero, not a division by zero
        scale = np.array([1.0 / m if m > 1e-300 else 0.0
                          for m in (np.mean(np.diag(Ktr[f])) for f in range(Ktr.shape[0]))])
        K = np.tensordot(scale, Kf, axes=(0, 0))          # the joint kernel over the whole set
        K_tr, K_te = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
        diag_te = np.diag(K)[te]
        # the plain process: the GP interval and the conformal interval, calibrated on its folds
        s0 = Surrogate.fit(K_tr, y[tr], lam)
        m0, sd0 = s0.predict(K_te, diag_te)
        plain = _interval_row(y[te], m0, m0 - k * sd0, m0 + k * sd0)
        ycv0 = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=10, seed=rep)
        conf0 = calibrate(y[tr], ycv0, level=LEVEL)
        lo_c, hi_c = conf0.interval(m0)
        conformal = _interval_row(y[te], m0, lo_c, hi_c)
        # the warped process: the warp fitted on the training part, the GP on the warped
        # activities, the interval carried back through the inverse
        t1 = time.time()
        w = fit_warp(K_tr, y[tr], lam, n_terms=TERMS, restarts=2, seed=rep)
        s1 = Surrogate.fit(K_tr, w.forward(y[tr]), lam)
        m1, sd1 = s1.predict(K_te, diag_te)
        med, lo_w, hi_w = w.interval(m1, sd1, LEVEL)
        warped = _interval_row(y[te], med, lo_w, hi_w)
        # and its conformal interval, from the warp refitted inside every fold
        ycv1, _mz, _sz, _zt = cross_validate_warped(K_tr, y[tr], lam, n_terms=TERMS, folds=10,
                                                    seed=rep, restarts=1)
        conf1 = calibrate(y[tr], ycv1, level=LEVEL)
        lo_wc, hi_wc = conf1.interval(med)
        warped_conformal = _interval_row(y[te], med, lo_wc, hi_wc)
        rows.append({"repeat": rep, "alpha": alpha, "lam": lam, "n_train": int(len(tr)),
                     "n_test": int(len(te)), "skewness_train": _skewness(y[tr]),
                     "gain": float(w.gain), "identity": bool(w.is_identity),
                     "warp": w.to_dict(), "seconds_warp": time.time() - t1,
                     "plain": plain, "conformal": conformal, "warped": warped,
                     "warped_conformal": warped_conformal})
        log(f"  split {rep}: alpha {alpha:g} lambda {lam:g}; warp gain {w.gain:+.2f}"
            f"{' (identity)' if w.is_identity else ''}; q2 plain {plain['q2']:+.3f} warped "
            f"{warped['q2']:+.3f}; coverage at {LEVEL:.0%}: GP {plain['coverage']:.0%} "
            f"(width {plain['width']:.2f}), conformal {conformal['coverage']:.0%} "
            f"({conformal['width']:.2f}), warped {warped['coverage']:.0%} "
            f"({warped['width']:.2f})")
    out = {"set": name, "n": n, "skewness": _skewness(y), "repeats": repeats, "level": LEVEL,
           "terms": TERMS, "rows": rows, "seconds": time.time() - t0}
    json.dump(out, open(out_path, "w"), indent=1)
    log(f"  {time.time() - t0:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _load_all(results: str):
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    data = [json.load(open(f)) for f in files]
    return [d for d in data if d.get("rows")]


def _mean(rows, path, key):
    return float(np.mean([r[path][key] for r in rows]))


def _paired_bootstrap(per_set: np.ndarray, runs: int = 2000, seed: int = SEED):
    """The mean of a per-set difference and its 95 % interval, resampling the sets."""
    rng = np.random.RandomState(seed)
    x = np.asarray(per_set, float)
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), float("nan"), float("nan")
    draws = np.array([np.mean(x[rng.randint(0, len(x), len(x))]) for _ in range(runs)])
    return float(np.mean(x)), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def report_collection(results: str, title: str) -> str:
    data = _load_all(results)
    if not data:
        return ""
    data.sort(key=lambda d: -d["n"])
    level = data[0]["level"]
    lines = [f"## {title}", "",
             f"{len(data)} sets, {data[0]['repeats']} random 70/30 splits each; alpha and lambda "
             f"by the evidence on the training part. Central intervals at {level:.0%}: the plain "
             "GP's, the conformal interval calibrated on the plain fit's folds, and the warped "
             "GP's, carried back through the inverse of a transform with "
             f"{data[0]['terms']} tanh terms fitted on the training part. *gain* is the log "
             "evidence the warp gained over the identity, in nats; *skew* the skewness of the "
             "activities.", "",
             "| set | n | skew | gain | q2 plain | q2 warped | coverage GP | coverage conformal | "
             "coverage warped | width GP | width conformal | width warped |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    acc = {k: [] for k in ("dq2", "dcov", "dcov_conf", "dwidth", "dwidth_conf", "gain")}
    better = worse = same = 0
    for d in data:
        r = d["rows"]
        v = {"q2p": _mean(r, "plain", "q2"), "q2w": _mean(r, "warped", "q2"),
             "cp": _mean(r, "plain", "coverage"), "cc": _mean(r, "conformal", "coverage"),
             "cw": _mean(r, "warped", "coverage"), "wp": _mean(r, "plain", "width"),
             "wc": _mean(r, "conformal", "width"), "ww": _mean(r, "warped", "width"),
             "gain": float(np.mean([x["gain"] for x in r]))}
        acc["dq2"].append(v["q2w"] - v["q2p"])
        acc["dcov"].append(v["cw"] - v["cp"])
        acc["dcov_conf"].append(v["cw"] - v["cc"])
        acc["dwidth"].append(v["ww"] - v["wp"])
        acc["dwidth_conf"].append(v["ww"] - v["wc"])
        acc["gain"].append(v["gain"])
        # closer to the promised coverage than the plain GP, or not
        gap_w, gap_p = abs(v["cw"] - level), abs(v["cp"] - level)
        if gap_w < gap_p - 1e-9:
            better += 1
        elif gap_w > gap_p + 1e-9:
            worse += 1
        else:
            same += 1
        lines.append(f"| {d['set']} | {d['n']} | {d['skewness']:+.2f} | {v['gain']:+.1f} | "
                     f"{v['q2p']:+.3f} | {v['q2w']:+.3f} | {v['cp']:.0%} | {v['cc']:.0%} | "
                     f"{v['cw']:.0%} | {v['wp']:.2f} | {v['wc']:.2f} | {v['ww']:.2f} |")
    m = {k: _paired_bootstrap(np.array(val)) for k, val in acc.items()}
    lines += ["",
              f"Over the {len(data)} sets, the warped process against the plain one (mean "
              "difference and its 95 % interval from a paired bootstrap over the sets): q2 "
              f"{m['dq2'][0]:+.3f} [{m['dq2'][1]:+.3f}, {m['dq2'][2]:+.3f}]; coverage "
              f"{m['dcov'][0]:+.1%} [{m['dcov'][1]:+.1%}, {m['dcov'][2]:+.1%}]; width "
              f"{m['dwidth'][0]:+.3f} [{m['dwidth'][1]:+.3f}, {m['dwidth'][2]:+.3f}]. Against "
              f"the conformal interval: coverage {m['dcov_conf'][0]:+.1%} "
              f"[{m['dcov_conf'][1]:+.1%}, {m['dcov_conf'][2]:+.1%}], width "
              f"{m['dwidth_conf'][0]:+.3f} [{m['dwidth_conf'][1]:+.3f}, "
              f"{m['dwidth_conf'][2]:+.3f}]. The warped interval is closer to the promised "
              f"{level:.0%} than the plain GP's on {better} sets, further on {worse}"
              + (f", the same on {same}" if same else "") + ". The warp gained "
              f"{m['gain'][0]:+.1f} nats of evidence over the identity on average."]
    return "\n".join(lines)


def report(results: str = RESULTS) -> str:
    """The table of one results folder: the Sutherland sets, or the Mittal collection when the
    folder's name ends in ``_mittal``.  One folder, one command, one table."""
    mittal = os.path.basename(os.path.normpath(results)).endswith("_mittal")
    title = ("The 38 sets of the Mittal collection, 15 of them under forty molecules" if mittal
             else "The eight Sutherland sets")
    head = ["# A warped Gaussian process: a monotone transform of the activity, learned with "
            "the GP (plan item ND11)", ""]
    body = report_collection(results, title)
    if not body:
        return "No results yet; run `python benchmarks/run_warped.py`" + (" --mittal" if mittal
                                                                             else "") + "."
    return "\n".join(head) + body


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=None)
    ap.add_argument("--mittal", action="store_true",
                    help="the 38 sets of examples/mittal instead of the Sutherland sets")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    results = RESULTS + ("_mittal" if args.mittal else "")
    data = MITTAL if args.mittal else DATA
    if not args.report:
        names = args.sets
        if names is None:
            names = ([os.path.splitext(os.path.basename(p))[0]
                      for p in sorted(glob.glob(os.path.join(MITTAL, "*.sdf")))]
                     if args.mittal else list(SETS))
        for name in names:
            run_set(name, data, results, repeats=args.repeats, force=args.force)
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
