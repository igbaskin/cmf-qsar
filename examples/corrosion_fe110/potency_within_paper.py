"""Do the fields rank the analogues within a series?  Within-paper centred potencies, leave-paper-out.

Every (compound, paper) pair in 1 M HCl with at least three of the node's compounds in the
paper gives one observation: the paper-wise median MWILA minus the paper's mean over its
compounds.  So the target is "how much better than the paper's average is this compound in
this paper", which is what a series measured under one protocol can say, and the laboratory
offsets are gone by construction.

Cross-validation holds out one paper at a time.  Lenient: the paper's rows are held out and
everything else trains (a compound measured elsewhere stays in the training set under its other
papers' centred values).  Strict: the held-out paper's compounds are removed from the training
rows as well, so the model ranks analogues it has never seen.  The inner selection of alpha and
lambda is a grouped 5-fold cross-validation over the training rows (rows of one compound stay
together).  KRR on the five physicochemical fields, MCS frame and adsorbed-pose frame.

Reported: the pooled q2 over all held-out rows (the fraction of the within-paper variance
predicted), the Spearman rank correlation within each paper of five or more compounds, and the
same for E_ads centred within the paper, as the physics-only baseline.

    python examples/corrosion_fe110/potency_within_paper.py

Writes potency_within_paper.csv next to this script and prints the tables.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import compare_alignments as ca  # noqa: E402
import eads_as_second_target as est  # noqa: E402
import potency_series as ps  # noqa: E402
from cmfqsar.model import krr_fit, r2_score  # noqa: E402

MIN_COMPOUNDS = 3
MIN_RANK = 5


def observations(h: pd.DataFrame) -> pd.DataFrame:
    pm = h.groupby(["Refcode", "db_id"]).MWILA.median().reset_index()
    n = pm.groupby("Refcode").db_id.transform("size")
    pm = pm[n >= MIN_COMPOUNDS].copy()
    pm["y"] = pm.MWILA - pm.groupby("Refcode").MWILA.transform("mean")
    return pm.reset_index(drop=True)


def grouped_folds(groups: np.ndarray, k: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    uniq = rng.permutation(np.unique(groups))
    out = []
    for part in np.array_split(uniq, k):
        out.append(np.flatnonzero(np.isin(groups, part)))
    return out


def select_and_fit(Ks: dict, pos: np.ndarray, y: np.ndarray, groups: np.ndarray, lams, seed=0):
    """alpha, lambda by grouped 5-fold CV over the training rows; returns (alpha, lam, coef, bias)."""
    folds = grouped_folds(groups, 5, seed)
    best = None
    for a, K in Ks.items():
        Kt = K[np.ix_(pos, pos)]
        for lam in lams:
            pred = np.zeros(len(y))
            for te in folds:
                tr = np.setdiff1d(np.arange(len(y)), te)
                coef, b = krr_fit(Kt[np.ix_(tr, tr)], y[tr], lam)
                pred[te] = Kt[np.ix_(te, tr)] @ coef + b
            err = float(np.sqrt(np.mean((pred - y) ** 2)))
            if best is None or err < best[0]:
                best = (err, a, lam)
    _, a, lam = best
    coef, b = krr_fit(Ks[a][np.ix_(pos, pos)], y, lam)
    return a, lam, coef, b


def leave_paper_out(obs: pd.DataFrame, Ks: dict, id_to_pos: dict, strict: bool, lams, seed=0):
    pred = np.full(len(obs), np.nan)
    for paper in obs.Refcode.unique():
        test = np.flatnonzero(obs.Refcode == paper)
        train = np.flatnonzero(obs.Refcode != paper)
        if strict:
            held = set(obs.db_id[test])
            train = train[~np.isin(obs.db_id.values[train], list(held))]
        if len(train) < 10:
            continue
        pos_tr = np.array([id_to_pos[i] for i in obs.db_id.values[train]])
        pos_te = np.array([id_to_pos[i] for i in obs.db_id.values[test]])
        a, lam, coef, b = select_and_fit(Ks, pos_tr, obs.y.values[train], obs.db_id.values[train], lams, seed)
        K_te = Ks[a][np.ix_(pos_te, pos_tr)]
        p = K_te @ coef + b
        pred[test] = p - p.mean()                   # the target is centred within the paper: so is the prediction
    return pred


def summarise(obs: pd.DataFrame, pred: np.ndarray, label: str, rows: list, extra: dict | None = None):
    ok = np.isfinite(pred)
    q2 = r2_score(obs.y.values[ok], pred[ok])
    rhos = []
    for paper, g in obs[ok].groupby("Refcode"):
        if len(g) >= MIN_RANK:
            rhos.append((paper, len(g), spearmanr(g.y, pred[g.index]).correlation))
    row = {"model": label, "n_rows": int(ok.sum()), "n_papers": obs[ok].Refcode.nunique(),
           "q2_within": round(float(q2), 3),
           "spearman_mean": round(float(np.mean([r for _, _, r in rhos])), 3),
           "spearman_by_paper": "; ".join(f"{p}({n}):{r:.2f}" for p, n, r in rhos)}
    if extra:
        row.update(extra)
    rows.append(row)
    print(f"{label:34s} rows={row['n_rows']:3d} papers={row['n_papers']:2d} q2_within = {q2:6.3f}  "
          f"mean Spearman = {row['spearman_mean']:.2f}   [{row['spearman_by_paper']}]", flush=True)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--node", default="157", choices=sorted(ps.NODES))
    a = ap.parse_args(argv)
    suffix = ps.configure(a.node)
    h = ps.rows_1m_hcl()
    obs = observations(h)
    print(f"{len(obs)} (compound, paper) observations from {obs.Refcode.nunique()} papers with >= {MIN_COMPOUNDS} compounds; "
          f"{obs.db_id.nunique()} compounds; sd of the centred potency {obs.y.std():.2f}")
    fr = ps.frames_by_db_id()
    rows = []
    # the physics-only baseline: E_ads centred within the paper
    mols, names, ids, e_ads, gas_ok = fr["pose+mcs"]
    e_by_id = {i: (e if g else np.nan) for i, e, g in zip(ids, e_ads, gas_ok)}
    e = np.array([e_by_id.get(i, np.nan) for i in obs.db_id])
    ec = pd.Series(e, index=obs.index)
    ec = ec - ec.groupby(obs.Refcode).transform("mean")
    okE = np.isfinite(ec.values)
    slope = np.polyfit(ec.values[okE], obs.y.values[okE], 1)
    summarise(obs[okE].reset_index(drop=True), (np.polyval(slope, ec.values[okE])), "E_ads centred (one slope, no CV)", rows)
    for frame in ("mcs", "pose+mcs"):
        mols, names, ids, _, _ = fr[frame]
        id_to_pos = {i: k for k, i in enumerate(ids)}
        Ks = est.combined_kernels(mols, names, ca.ALPHAS)
        for strict in (False, True):
            pred = leave_paper_out(obs, Ks, id_to_pos, strict, ca.LAMS)
            summarise(obs, pred, f"fields, {frame}, {'strict' if strict else 'lenient'}", rows,
                      {"frame": frame, "strict": int(strict)})
    out = os.path.join(HERE, f"potency_within_paper{suffix}.csv")
    keys = ["model", "frame", "strict", "n_rows", "n_papers", "q2_within", "spearman_mean", "spearman_by_paper"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print("->", out)


if __name__ == "__main__":
    main()
