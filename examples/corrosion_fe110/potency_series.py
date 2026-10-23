"""The potency's reproducibility, single-laboratory series in both frames, and a
concentration-corrected target, from the per-experiment rows of the benzimidazole node.

Inputs: node157_experiments.csv (one row per experiment, exported from the database), the
node CSV and the UMA SDF.  Steps:

1. **Noise ceiling.**  Within-paper spread of a compound's MWILA over concentrations and methods,
   between-paper spread of the same compound, the split-half reliability across papers, and the
   q2 that a perfect model of the per-compound medians could reach.
2. **Single-laboratory series.**  Papers (or laboratories) with several of the node's compounds
   in 1 M HCl: the series' own medians as the target, leave-one-out outer / 5-fold inner
   cross-validation, the MCS frame against the adsorbed-pose frame, with the correlation of
   E_ads to the series potency beside.
3. **A concentration-corrected potency.**  Within a paper, MWILA falls with ln C (it should not,
   under Langmuir); a common slope is fitted over all (compound, paper) rows with a molar
   concentration, and every compound's value is taken at 1 mM.  The whole-node frame comparison
   is repeated with that target.

    python examples/corrosion_fe110/potency_series.py

Writes potency_series.csv next to this script and prints the tables.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import compare_alignments as ca  # noqa: E402
from cmfqsar.fields import FieldSet  # noqa: E402

EXPERIMENTS = os.path.join(HERE, "node157_experiments.csv")
SERIES = {
    "hu2011": ["hu2011-2402"],
    "raicheva1993": ["raicheva1993-343"],
    "popova lab": ["popova2004-1333", "popova1996-benzimidazoles-ei", "popova2003-756", "popova2003-33"],
    "dutta lab": ["dutta2017-256", "dutta2015-541", "dutta2015-921"],
    "gutierrez2016": ["gutierrez2016-23"],
}


def rows_1m_hcl():
    d = pd.read_csv(EXPERIMENTS)
    return d[(d.medium == "HCl (1 M)") & d.MWILA.notna()].copy()


def noise_ceiling(h: pd.DataFrame) -> dict:
    med = h.groupby("db_id").MWILA.median()
    var_between = med.var(ddof=1)
    within = h.groupby(["db_id", "Refcode"]).MWILA.agg(["var", "size"])
    within = within[within["size"] >= 2]["var"]
    pm = h.groupby(["db_id", "Refcode"]).MWILA.median().reset_index()
    cnt = pm.groupby("db_id").size()
    mp = pm[pm.db_id.isin(cnt[cnt >= 2].index)]
    between = mp.groupby("db_id").MWILA.var(ddof=1)
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(500):
        a, b = [], []
        for _, g in mp.groupby("db_id"):
            v = rng.permutation(g.MWILA.values)
            a.append(v[0]); b.append(v[1])
        rs.append(np.corrcoef(a, b)[0, 1])
    rows = h.groupby("db_id").size()
    papers = h.groupby("db_id").Refcode.nunique()
    err = (between.mean() / papers + within.mean() / rows).reindex(med.index)
    return {"n_compounds": len(med), "sd_between_compounds": float(np.sqrt(var_between)),
            "sd_within_paper": float(np.sqrt(within.mean())), "n_pairs_within": int(len(within)),
            "sd_between_papers": float(np.sqrt(between.mean())), "n_multi_paper": int(len(between)),
            "split_half_r": float(np.mean(rs)), "ceiling_q2": float(1 - err.mean() / var_between)}


def concentration_corrected(h: pd.DataFrame, c_ref: float = 1e-3):
    """Compound values at `c_ref` mol/L from MWILA = a_(compound, paper) + b ln(C / c_ref), one
    slope b for all; a compound's value is the median of its (compound, paper) intercepts."""
    g = h.dropna(subset=["Conc_M"]).copy()
    g["lnc"] = np.log(g.Conc_M / c_ref)
    keys = list(g.groupby(["db_id", "Refcode"]).groups)
    index = {k: i for i, k in enumerate(keys)}
    X = np.zeros((len(g), len(keys) + 1))
    for r, (cid, ref) in enumerate(zip(g.db_id, g.Refcode)):
        X[r, index[(cid, ref)]] = 1.0
    X[:, -1] = g.lnc.values
    coef = np.linalg.lstsq(X, g.MWILA.values, rcond=None)[0]
    b = float(coef[-1])
    g["intercept"] = [coef[index[(cid, ref)]] for cid, ref in zip(g.db_id, g.Refcode)]
    per_pair = g.groupby(["db_id", "Refcode"]).intercept.first().reset_index()
    y = per_pair.groupby("db_id").intercept.median()
    resid = g.MWILA.values - X @ coef
    return y, b, float(np.std(resid))


def frames_by_db_id():
    fr = ca.frames()
    node = pd.read_csv(os.path.join(ROOT, "examples", "corrosion", "157_c1ccc2_nH_cnc2c1.csv"))
    name_to_id = dict(zip(node["name"], node["db_id"]))
    out = {}
    for frame in ("mcs", "pose+mcs"):
        mols, names, _, e_ads, gas_ok = fr[frame]
        out[frame] = (mols, names, np.array([name_to_id[n] for n in names]), e_ads, gas_ok)
    return out


def evaluate_target(fr, y_by_id: pd.Series, label: str, fs, outer=None, repeats=10, rows=None, min_n=6):
    for frame in ("mcs", "pose+mcs"):
        mols, names, ids, e_ads, gas_ok = fr[frame]
        y = np.array([y_by_id.get(i, np.nan) for i in ids])
        keep = np.isfinite(y)
        idx = np.flatnonzero(keep)
        if len(idx) < min_n:
            continue
        n_out = len(idx) if outer is None else outer
        reps = 1 if outer is None else repeats
        res, dt = ca.evaluate([mols[i] for i in idx], [names[i] for i in idx], y[idx], fs, n_out, reps, 0)
        ok = np.isfinite(e_ads[idx]) & gas_ok[idx]
        r_e = float(np.corrcoef(e_ads[idx][ok], y[idx][ok])[0, 1]) if ok.sum() > 3 else np.nan
        row = {"target": label, "frame": frame, "n": len(idx), "q2": round(res.q2_ex, 3),
               "q2_sd": round(res.q2_ex_sd, 3), "rmse": round(res.rmse_ex, 3), "sd_y": round(float(np.std(y[idx])), 3),
               "r_E_ads": round(r_e, 2), "cv": "LOO" if outer is None else f"{outer}x{repeats}"}
        rows.append(row)
        print(f"{label:28s} {frame:9s} n={len(idx):3d} q2 = {res.q2_ex:6.3f} +- {res.q2_ex_sd:.3f} rmse {res.rmse_ex:.3f} "
              f"(sd y {np.std(y[idx]):.3f})  r(E_ads, y) = {r_e:5.2f}", flush=True)


def main():
    h = rows_1m_hcl()
    nc = noise_ceiling(h)
    print("noise ceiling:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in nc.items()})
    fs = FieldSet.create()
    fr = frames_by_db_id()
    rows = []
    for label, refs in SERIES.items():
        sub = h[h.Refcode.isin(refs)]
        y = sub.groupby("db_id").MWILA.median()
        print(f"series {label}: {len(y)} compounds, {len(sub)} rows, papers {refs}")
        evaluate_target(fr, y, f"series {label}", fs, outer=None, rows=rows)
    y_c, slope, resid_sd = concentration_corrected(h)
    print(f"concentration correction: common slope {slope:.3f} per ln C, residual sd {resid_sd:.2f}, {len(y_c)} compounds")
    evaluate_target(fr, h.groupby("db_id").MWILA.median(), "node median (as before)", fs, outer=5, rows=rows)
    evaluate_target(fr, y_c, "node at 1 mM (corrected)", fs, outer=5, rows=rows)
    out = os.path.join(HERE, "potency_series.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(HERE, "potency_noise.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(nc.keys()))
        w.writeheader(); w.writerow(nc)
    print("->", out)


if __name__ == "__main__":
    main()
