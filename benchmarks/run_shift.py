#!/usr/bin/env python
"""
Covariate shift on the Sutherland sets: do density-ratio weights repair the intervals and the fit?

A screening library is rarely drawn from the same population as the training set.  The conformal
intervals of the program then lose their guarantee, because the calibration errors no longer
resemble the errors on the library.  :mod:`cmfqsar.shift` offers weighted conformal intervals and
weighted fitting.  This script measures both against splits whose shift is known exactly.

**The shifted splits.**  Every molecule goes to the test part independently, with probability

    pi(x) = 1 / (1 + exp(-(c z(x) + b))),

where z is a standardised property of the structure alone and b makes the mean of pi equal 0.3.
Because the choice depends on the structure only, the activity of a given molecule follows the
same rule in both parts: this is covariate shift by construction.  The true density ratio of the
test part to the training part is pi(x) / (1 - pi(x)) up to a constant, so every split carries
its own *oracle* weights, and the estimated weights can be compared with them.

Scenarios (c is the strength):

    random      c = 0, the control: no shift, and the weights should change nothing
    pc1-1, -2   z = the first principal component of the field kernel, c = 1 and 2
    size-1, -2  z = the number of heavy atoms, c = 1 and 2
    neighbours  the 30 % of molecules nearest in field distance to one random molecule; a shift
                with no overlap at the edge, so no oracle, and the case where weights must fail

**What is compared, on every split.**  The model is kernel ridge regression with alpha and lambda
chosen by cross-validation on the training part, as in `run_conformal.py`.  Intervals at 80, 90
and 95 %:

    cv, cv-norm            the program's present rules (out-of-fold errors, plain and normalised)
    cv-ulsif               the same errors weighted by the ratio estimated by uLSIF
    cv-klr, cv-norm-klr    weighted by the ratio from kernel logistic regression
    cv-oracle              weighted by the true ratio
    split, split-ulsif,    a held-out calibration third: the version with an exact guarantee
    split-klr, split-oracle (split-oracle must cover; this checks the implementation)

An infinite interval counts as covering and is reported as a share; widths are means over the
finite intervals.  The fit is compared by the RMSE on the test part: plain KRR against KRR
weighted by kernel mean matching, by uLSIF, by the classifier and by the oracle, each with its lambda chosen by
importance-weighted cross-validation.

Usage
-----
    python benchmarks/run_shift.py
    python benchmarks/run_shift.py --sets ACE --repeats 5
    python benchmarks/run_shift.py --report
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
from scipy.stats import spearmanr  # noqa: E402

from cmfqsar.conformal import calibrate, difficulty_from_kernel  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.mmd import mmd_test  # noqa: E402
from cmfqsar.model import cv_predict, krr_fit  # noqa: E402
from cmfqsar.shift import (classifier_ratio, distance2, effective_size, gaussian_layer, kmm,  # noqa: E402
                           median_gamma, ulsif_crossfit, weighted_calibrate, weighted_cv_predict,
                           weighted_krr_fit, weighted_mmd2)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "shift")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0)
LEVELS = (0.8, 0.9, 0.95)
BETA = 0.1
TEST_SHARE = 0.3
SCENARIOS = {"random": ("none", 0.0), "pc1-1": ("pc1", 1.0), "pc1-2": ("pc1", 2.0),
             "size-1": ("size", 1.0), "size-2": ("size", 2.0), "neighbours": ("neighbours", 0.0)}
RULES = ("cv", "cv-norm", "cv-ulsif", "cv-klr", "cv-norm-klr", "cv-oracle",
         "split", "split-ulsif", "split-klr", "split-oracle")
FITS = ("plain", "kmm", "ulsif", "klr", "oracle")


# --------------------------------------------------------------------------- the splits

def _offset(z, c, share=TEST_SHARE):
    """b such that the mean of sigmoid(c z + b) is `share`, by bisection."""
    lo, hi = -30.0, 30.0
    for _ in range(100):
        b = 0.5 * (lo + hi)
        if np.mean(1.0 / (1.0 + np.exp(-(c * z + b)))) > share:
            hi = b
        else:
            lo = b
    return 0.5 * (lo + hi)


def selection_probability(z, c):
    z = (z - z.mean()) / max(float(z.std()), 1e-12)
    b = _offset(z, c)
    return 1.0 / (1.0 + np.exp(-(c * z + b)))


def first_component(K):
    """Scores of every molecule on the first kernel principal component of the centred kernel."""
    n = K.shape[0]
    J = np.eye(n) - 1.0 / n
    Kc = J @ K @ J
    vals, vecs = np.linalg.eigh(0.5 * (Kc + Kc.T))
    return vecs[:, -1] * np.sqrt(max(vals[-1], 0.0))


def draw_split(kind, c, z, D2, rng, min_side=8):
    """(test mask, oracle ratio or None) for one repeat."""
    n = len(z)
    if kind == "neighbours":
        centre = rng.randint(n)
        n_te = int(round(TEST_SHARE * n))
        te = np.zeros(n, bool)
        te[np.argsort(D2[centre], kind="mergesort")[:n_te]] = True
        return te, None
    pi = np.full(n, TEST_SHARE) if kind == "none" else selection_probability(z, c)
    for _ in range(1000):
        te = rng.uniform(size=n) < pi
        if min_side <= te.sum() <= n - min_side:
            return te, pi / (1.0 - pi)
    raise RuntimeError("could not draw a split with enough molecules on both sides")


# --------------------------------------------------------------------------- the model

def _choose(K, y, tr, folds=10, seed=0):
    best = None
    Ktr = K[np.ix_(tr, tr)]
    ss = float(np.sum((y[tr] - np.mean(y[tr])) ** 2))
    for lam in LAMS:
        p = cv_predict(Ktr, y[tr], "krr", {"lam": float(lam)}, folds=folds, seed=seed)
        q2 = 1.0 - float(np.sum((y[tr] - p) ** 2)) / max(ss, 1e-12)
        if best is None or q2 > best[1]:
            best = (float(lam), q2)
    return best


def _choose_weighted(K, y, tr, w, folds=10, seed=0):
    """lambda by importance-weighted cross-validation: the error expected on the test part."""
    best = None
    for lam in LAMS:
        p = weighted_cv_predict(K[np.ix_(tr, tr)], y[tr], w, lam, folds=folds, seed=seed)
        ok = np.isfinite(p)
        err = float(np.sum(w[ok] * (y[tr][ok] - p[ok]) ** 2) / max(np.sum(w[ok]), 1e-12))
        if best is None or err < best[1]:
            best = (float(lam), err)
    return best[0]


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _cover(y, lo, hi):
    inside = (y >= lo) & (y <= hi)
    width = hi - lo
    fin = np.isfinite(width)
    return {"coverage": float(np.mean(inside)), "infinite": float(np.mean(~fin)),
            "finite_coverage": float(np.mean(inside[fin])) if fin.any() else float("nan"),
            "mean_width": float(np.mean(width[fin])) if fin.any() else float("nan")}


# --------------------------------------------------------------------------- one set

def kernels(mols, names):
    from run_conformal import kernels as conformal_kernels
    return conformal_kernels(mols, names, ALPHAS)


def run_set(name, repeats=20, force=False, data=DATA, results=RESULTS, scenarios=None):
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
    names = [nm for nm, k in zip(ds.names, keep) if k]
    y = y[keep]
    n = len(y)
    print(f"=== {name}: {n} molecules, activity sd {np.std(y):.2f}", flush=True)
    Ks = kernels(mols, names)
    K_shift = dict(Ks)[0.3]                 # the kernel that defines the pc1 shift and neighbours
    D2_shift = distance2(K_shift, np.diag(K_shift), np.diag(K_shift))
    props = {"pc1": first_component(K_shift),
             "size": np.array([m.GetNumHeavyAtoms() for m in mols], float),
             "none": np.zeros(n), "neighbours": np.zeros(n)}

    rows, fits, diags = [], [], []
    for scen in (scenarios or list(SCENARIOS)):
        kind, c = SCENARIOS[scen]
        for rep in range(repeats):
            rng = np.random.RandomState(1000 * rep + 7)
            te_mask, oracle = draw_split(kind, c, props[kind], D2_shift, rng)
            te, tr = np.flatnonzero(te_mask), np.flatnonzero(~te_mask)

            best = None
            for alpha, K in Ks:
                lam, q2 = _choose(K, y, tr, seed=rep)
                if best is None or q2 > best[2]:
                    best = (K, lam, q2, alpha)
            K, lam, q2_tr, alpha = best
            d = np.diag(K)
            D2 = distance2(K, d, d)

            # ---- the weights: estimated (uLSIF at any molecule, KMM at training ones) and true
            # cross-fitted, so that no molecule's weight contains its own kernel function
            u_ratio, w_tr, w_te = ulsif_crossfit(D2[np.ix_(tr, te)], D2[np.ix_(te, te)],
                                                 seed=rep)
            order = np.concatenate([tr, te])
            cr = classifier_ratio(D2[np.ix_(order, order)],
                                  np.r_[np.zeros(len(tr), bool), np.ones(len(te), bool)],
                                  seed=rep)
            k_tr, k_te = cr.w_train, cr.w_library
            G = gaussian_layer(D2, median_gamma(D2[np.ix_(tr, te)]))
            km = kmm(G[np.ix_(tr, tr)], G[np.ix_(tr, te)], G[np.ix_(te, te)])
            o_tr = None if oracle is None else oracle[tr]
            o_te = None if oracle is None else oracle[te]
            mt = mmd_test(G, te_mask, permutations=99, seed=rep)
            diags.append({
                "scenario": scen, "repeat": rep, "alpha": alpha, "lam": lam, "q2_train": q2_tr,
                "n_train": int(len(tr)), "n_test": int(len(te)),
                "mmd_z": mt.z, "mmd_p": mt.p_value,
                "mmd2_before": km.mmd2_before, "mmd2_kmm": km.mmd2_after,
                "mmd2_ulsif": weighted_mmd2(G[np.ix_(tr, tr)], G[np.ix_(tr, te)],
                                            G[np.ix_(te, te)], np.maximum(w_tr, 0) + 1e-300),
                "ess_ulsif": effective_size(w_tr) / len(tr), "ess_kmm": km.ess / len(tr),
                "ess_klr": effective_size(k_tr) / len(tr),
                "ulsif_gamma_factor": float(u_ratio.gamma / median_gamma(D2[np.ix_(tr, te)])),
                "klr_gamma_factor": cr.gamma_factor, "klr_lam": cr.lam,
                "klr_edge": cr.at_edge, "klr_log_loss": cr.log_loss,
                "rho_klr": None if o_tr is None else float(spearmanr(k_tr, o_tr)[0]),
                "ess_oracle": None if o_tr is None else effective_size(o_tr) / len(tr),
                "rho_ulsif": None if o_tr is None else float(spearmanr(w_tr, o_tr)[0]),
                "rho_kmm": None if o_tr is None else float(spearmanr(km.weights, o_tr)[0]),
            })

            # ---- the fit
            a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
            yp_te = K[np.ix_(te, tr)] @ a + b
            fit_row = {"scenario": scen, "repeat": rep, "plain": _rmse(y[te], yp_te)}
            for label, w in (("kmm", km.weights), ("ulsif", w_tr), ("klr", k_tr),
                             ("oracle", o_tr)):
                if w is None or not np.any(w > 0):
                    fit_row[label] = None
                    continue
                lam_w = _choose_weighted(K, y, tr, w, seed=rep)
                aw, bw = weighted_krr_fit(K[np.ix_(tr, tr)], y[tr], w, lam_w)
                fit_row[label] = _rmse(y[te], K[np.ix_(te, tr)] @ aw + bw)
            fits.append(fit_row)

            # ---- the intervals, all around the plain model's predictions
            yp_oof = cv_predict(K[np.ix_(tr, tr)], y[tr], "krr", {"lam": lam}, folds=10, seed=rep)
            d_tr = difficulty_from_kernel(K[np.ix_(tr, tr)], d[tr], K[np.ix_(tr, tr)])
            d_te = difficulty_from_kernel(K[np.ix_(te, tr)], d[te], K[np.ix_(tr, tr)])
            perm = rng.permutation(len(tr))
            cut = max(10, int(round(len(tr) / 3)))
            cal, fit = tr[np.sort(perm[:cut])], tr[np.sort(perm[cut:])]
            a2, b2 = krr_fit(K[np.ix_(fit, fit)], y[fit], lam, True)
            yp_cal = K[np.ix_(cal, fit)] @ a2 + b2
            yp_te2 = K[np.ix_(te, fit)] @ a2 + b2
            w_cal = w_tr[np.sort(perm[:cut])]
            k_cal = k_tr[np.sort(perm[:cut])]
            o_cal = None if oracle is None else oracle[cal]
            for level in LEVELS:
                cand = {
                    "cv": (calibrate(y[tr], yp_oof, level), yp_te, None, None),
                    "cv-norm": (calibrate(y[tr], yp_oof, level, difficulty=d_tr, beta=BETA),
                                yp_te, None, d_te),
                    "cv-ulsif": (weighted_calibrate(y[tr], yp_oof, w_tr, level), yp_te, w_te,
                                 None),
                    "cv-klr": (weighted_calibrate(y[tr], yp_oof, k_tr, level), yp_te, k_te,
                               None),
                    "cv-norm-klr": (weighted_calibrate(y[tr], yp_oof, k_tr, level,
                                                       difficulty=d_tr, beta=BETA),
                                    yp_te, k_te, d_te),
                    "split": (calibrate(y[cal], yp_cal, level, kind="split"), yp_te2, None, None),
                    "split-ulsif": (weighted_calibrate(y[cal], yp_cal, w_cal, level,
                                                       kind="split"), yp_te2, w_te, None),
                    "split-klr": (weighted_calibrate(y[cal], yp_cal, k_cal, level,
                                                     kind="split"), yp_te2, k_te, None),
                }
                if oracle is not None:
                    cand["cv-oracle"] = (weighted_calibrate(y[tr], yp_oof, o_tr, level),
                                         yp_te, o_te, None)
                    cand["split-oracle"] = (weighted_calibrate(y[cal], yp_cal, o_cal, level,
                                                               kind="split"), yp_te2, o_te, None)
                for rule, (cr, yp, wn, dn) in cand.items():
                    lo, hi = cr.interval(yp, dn) if wn is None else cr.interval(yp, wn, dn)
                    lo = np.broadcast_to(lo, yp.shape)
                    hi = np.broadcast_to(hi, yp.shape)
                    rows.append({"scenario": scen, "repeat": rep, "level": level, "rule": rule,
                                 **_cover(y[te], lo, hi)})
        print(f"  {scen}: done ({time.time() - t0:.0f} s)", flush=True)

    out = {"set": name, "n": int(n), "activity_sd": float(np.std(y)), "beta": BETA,
           "repeats": int(repeats), "rows": rows, "fits": fits, "diagnostics": diags,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"  {name}: {out['seconds']:.0f} s")
    return out


# --------------------------------------------------------------------------- the report

def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No shift results yet; run `python benchmarks/run_shift.py`."
    data = [json.load(open(f)) for f in files]
    sd = float(np.mean([d["activity_sd"] for d in data]))
    scen = [s for s in SCENARIOS if any(r["scenario"] == s for d in data for r in d["rows"])]
    reps = data[0]["repeats"]
    L = ["# Covariate shift on the Sutherland sets", "",
         f"{len(data)} sets, {reps} splits per set and scenario. Each molecule goes to the test "
         "part with a probability that depends on its structure only, so the activity follows "
         "the same rule in both parts and the true density ratio is known (the *oracle*). "
         "`neighbours` holds out the 30 % of molecules nearest to one molecule and has no oracle.",
         "",
         "Coverage counts an infinite interval as covering; `inf` is the share of infinite "
         "intervals; widths are means over finite intervals, divided by the standard deviation "
         f"of the activity (mean {sd:.2f}).", ""]

    L += ["## How strong is each shift", "",
          "| scenario | MMD z | MMD p < 0.05 | ESS true | ESS uLSIF | ESS KMM | ESS KLR | "
          "rank corr. uLSIF with true | KMM with true | KLR with true | MMD^2 left after KMM | "
          "after uLSIF |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in scen:
        g = [x for d in data for x in d["diagnostics"] if x["scenario"] == s]
        rel = lambda key: _mean([x[key] / x["mmd2_before"] for x in g  # noqa: E731
                                 if x["mmd2_before"] and x["mmd2_before"] > 0])
        L.append(f"| {s} | {_mean([x['mmd_z'] for x in g]):.1f} | "
                 f"{np.mean([x['mmd_p'] < 0.05 for x in g]):.0%} | "
                 f"{_mean([x['ess_oracle'] for x in g]):.0%} | {_mean([x['ess_ulsif'] for x in g]):.0%} | "
                 f"{_mean([x['ess_kmm'] for x in g]):.0%} | "
                 f"{_mean([x['ess_klr'] for x in g]):.0%} | "
                 f"{_mean([x['rho_ulsif'] for x in g]):+.2f} | "
                 f"{_mean([x['rho_kmm'] for x in g]):+.2f} | "
                 f"{_mean([x['rho_klr'] for x in g]):+.2f} | {rel('mmd2_kmm'):.0%} | "
                 f"{rel('mmd2_ulsif'):.0%} |")
    L += ["", "ESS is the effective sample size of the weights on the training part, as a share "
          "of its size. A rank correlation is Spearman's, between estimated and true weights. "
          "KLR is kernel logistic regression.", ""]
    g = [x for d in data for x in d["diagnostics"]]
    uf = [x["ulsif_gamma_factor"] for x in g]
    L += [f"Settings chosen by cross-validation: the uLSIF width factor had median "
          f"{np.median(uf):g} (grid 0.5 to 32, largest chosen in "
          f"{np.mean(np.isclose(uf, 32)):.0%} of splits); the classifier's setting was on the "
          f"edge of its grid in {np.mean([x['klr_edge'] for x in g]):.0%} of splits.", ""]

    for level in LEVELS:
        L += [f"## Coverage at {level:.0%}", "",
              "| scenario | " + " | ".join(f"`{r}`" for r in RULES) + " |",
              "|---|" + "---:|" * len(RULES)]
        for s in scen:
            cells = []
            for rule in RULES:
                g = [r for d in data for r in d["rows"]
                     if r["scenario"] == s and r["level"] == level and r["rule"] == rule]
                if not g:
                    cells.append("--")
                    continue
                c = _mean([r["coverage"] for r in g])
                inf = _mean([r["infinite"] for r in g])
                flag = " **low**" if c < level - 0.02 else ""
                cells.append(f"{c:.1%}{flag}" + (f" ({inf:.0%} inf)" if inf >= 0.005 else ""))
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")

    L += ["## Coverage at 90 % among the finite intervals only", "",
          "An infinite interval covers by construction. This table leaves those molecules out, "
          "so it shows how the weighted rules do on the molecules they do answer for. No "
          "guarantee applies to this subset.", "",
          "| scenario | " + " | ".join(f"`{r}`" for r in RULES) + " |",
          "|---|" + "---:|" * len(RULES)]
    for s in scen:
        cells = []
        for rule in RULES:
            g = [r.get("finite_coverage") for d in data for r in d["rows"]
                 if r["scenario"] == s and r["level"] == 0.9 and r["rule"] == rule]
            cells.append(f"{_mean(g):.1%}" if g else "--")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## Width at 90 %, divided by the activity's standard deviation", "",
          "| scenario | " + " | ".join(f"`{r}`" for r in RULES) + " |",
          "|---|" + "---:|" * len(RULES)]
    for s in scen:
        cells = []
        for rule in RULES:
            g = [r["mean_width"] / d["activity_sd"] for d in data for r in d["rows"]
                 if r["scenario"] == s and r["level"] == 0.9 and r["rule"] == rule]
            cells.append(f"{_mean(g):.2f}" if g else "--")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L.append("")

    L += ["## The fit: RMSE on the test part", "",
          "Mean over sets of the per-set mean RMSE. The difference columns are weighted minus "
          "plain (negative is better), with the number of sets, out of those with a value, where "
          "the weighted fit was better on average.", "",
          "| scenario | plain | KMM | uLSIF | KLR | oracle | KMM - plain | uLSIF - plain | "
          "KLR - plain | oracle - plain |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in scen:
        per = {f: [] for f in FITS}
        for d in data:
            g = [x for x in d["fits"] if x["scenario"] == s]
            for f in FITS:
                v = [x[f] for x in g if x[f] is not None]
                per[f].append(float(np.mean(v)) if v else None)
        cells = [f"{_mean(per[f]):.3f}" if _mean(per[f]) == _mean(per[f]) else "--" for f in FITS]
        diffs = []
        for f in FITS[1:]:
            pairs = [(a, b) for a, b in zip(per[f], per["plain"]) if a is not None]
            if not pairs:
                diffs.append("--")
                continue
            dd = [a - b for a, b in pairs]
            diffs.append(f"{np.mean(dd):+.3f} ({sum(x < 0 for x in dd)} of {len(dd)})")
        L.append(f"| {s} | " + " | ".join(cells + diffs) + " |")
    L.append("")

    L += ["## By set: coverage at 90 % on the strongest structural shift (`pc1-2`)", "",
          "| set | n | `cv` | `cv-norm` | `cv-ulsif` | `cv-klr` | `cv-norm-klr` | `cv-oracle` |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in data:
        cells = []
        for rule in ("cv", "cv-norm", "cv-ulsif", "cv-klr", "cv-norm-klr", "cv-oracle"):
            g = [r["coverage"] for r in d["rows"]
                 if r["scenario"] == "pc1-2" and r["level"] == 0.9 and r["rule"] == rule]
            cells.append(f"{np.mean(g):.0%}" if g else "--")
        L.append(f"| {d['set']} | {d['n']} | " + " | ".join(cells) + " |")
    text = "\n".join(L) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--scenarios", nargs="*", default=None, choices=list(SCENARIOS))
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()
    if a.report:
        print(report(a.results))
        return
    for name in a.sets:
        run_set(name, repeats=a.repeats, force=a.force, results=a.results,
                scenarios=a.scenarios)
    print()
    print(report(a.results))


if __name__ == "__main__":
    main()
