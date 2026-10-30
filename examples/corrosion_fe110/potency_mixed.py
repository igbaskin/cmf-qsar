"""A pooled potency without the laboratory offsets: compound + paper mixed model on the raw rows.

    MWILA_r = mu + a_compound(r) + b ln(C_r / 1 mM) + u_paper(r) + e_r,   u_paper ~ N(0, s_p^2)

fitted by REML on the 1 M HCl rows of the benzimidazole node (statsmodels MixedLM: compound
effects fixed, one random intercept per paper, one concentration slope for all).  A compound's
pooled potency is mu + a_i, its value at 1 mM in an average laboratory.  Papers linked to others
through shared compounds get an identifiable offset; papers that share no compound with any other
(20 of the 48) are shrunk towards zero offset, so their compounds keep their own means.
Concentrations given in ppm are converted with the molecular weight; the few rows in weight
per cent get no concentration term.

Then the frame comparison of compare_alignments.py is repeated with this target (and on the
39 compounds of the linked component alone), beside the raw medians.

    python examples/corrosion_fe110/potency_mixed.py

Writes potency_mixed.csv (the per-compound values) and potency_mixed_models.csv next to this
script and prints the tables.
"""
from __future__ import annotations

import csv
import os
import sys
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import potency_series as ps  # noqa: E402
from cmfqsar.fields import FieldSet  # noqa: E402


def fit_mixed(h: pd.DataFrame, c_ref: float = 1e-3):
    import statsmodels.formula.api as smf
    g = h.copy()
    conc = g.Conc_M.copy()
    ppm = g.Conc_M.isna() & g.Conc_ppm.notna() & g.MW.notna()
    conc[ppm] = g.Conc_ppm[ppm] * 1e-3 / g.MW[ppm]
    g["lnc"] = np.where(conc.notna(), np.log(conc / c_ref), 0.0)
    g["has_c"] = conc.notna()
    g["cid"] = g.db_id.astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm("MWILA ~ C(cid) + lnc", g, groups=g["Refcode"])
        res = model.fit(reml=True, method=["lbfgs"], maxiter=500)
    params = res.params
    mu = float(params["Intercept"])
    ids = sorted(g.db_id.unique())
    ref = str(ids[0])
    effects = {}
    for cid in ids:
        key = f"C(cid)[T.{cid}]"
        effects[cid] = mu + (float(params[key]) if key in params else 0.0)
    paper = res.random_effects
    info = {"slope_lnC": float(params["lnc"]), "sd_paper": float(np.sqrt(res.cov_re.iloc[0, 0])),
            "sd_resid": float(np.sqrt(res.scale)), "n_rows": len(g), "n_rows_with_conc": int(g.has_c.sum()),
            "n_papers": g.Refcode.nunique(), "converged": bool(res.converged)}
    return pd.Series(effects), {k: float(v.iloc[0]) for k, v in paper.items()}, info


def linked_component(h: pd.DataFrame):
    import networkx as nx
    G = nx.Graph()
    for cid, ref in h[["db_id", "Refcode"]].drop_duplicates().itertuples(index=False):
        G.add_edge(("c", cid), ("p", ref))
    big = max(nx.connected_components(G), key=len)
    return sorted(n[1] for n in big if n[0] == "c"), sorted(n[1] for n in big if n[0] == "p")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split(chr(10) * 2)[0])
    ap.add_argument("--node", default="157", choices=sorted(ps.NODES))
    a = ap.parse_args(argv)
    suffix = ps.configure(a.node)
    h = ps.rows_1m_hcl()
    y_mixed, paper_eff, info = fit_mixed(h)
    print("mixed model:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
    raw = h.groupby("db_id").MWILA.median()
    linked_c, linked_p = linked_component(h)
    print(f"linked component: {len(linked_c)} compounds, {len(linked_p)} papers")
    pe = pd.Series(paper_eff)
    print("paper offsets: sd %.2f over %d papers; linked papers sd %.2f, range %.2f..%.2f" % (
        pe.std(), len(pe), pe[linked_p].std(), pe[linked_p].min(), pe[linked_p].max()))
    both = pd.concat([raw.rename("raw_median"), y_mixed.rename("mixed")], axis=1)
    both["linked"] = both.index.isin(linked_c)
    print("raw median vs mixed: r %.3f, sd raw %.2f, sd mixed %.2f, max |diff| %.2f" % (
        np.corrcoef(both.raw_median, both.mixed)[0, 1], both.raw_median.std(), both.mixed.std(),
        (both.raw_median - both.mixed).abs().max()))
    node = pd.read_csv(ps.node_paths()[0])
    both = both.join(node.set_index("db_id")[["name", "smiles"]])
    both.reset_index().to_csv(os.path.join(HERE, f"potency_mixed{suffix}.csv"), index=False)

    fs = FieldSet.create()
    fr = ps.frames_by_db_id()
    rows = []
    ps.evaluate_target(fr, raw, "raw median, 73", fs, outer=5, rows=rows)
    ps.evaluate_target(fr, y_mixed, "mixed model, 73", fs, outer=5, rows=rows)
    ps.evaluate_target(fr, raw[raw.index.isin(linked_c)], "raw median, linked 39", fs, outer=5, rows=rows)
    ps.evaluate_target(fr, y_mixed[y_mixed.index.isin(linked_c)], "mixed model, linked 39", fs, outer=5, rows=rows)
    with open(os.path.join(HERE, f"potency_mixed_models{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("->", os.path.join(HERE, f"potency_mixed_models{suffix}.csv"))


if __name__ == "__main__":
    main()
