"""How reproducible is the literature potency of a scaffold node?  Data-only part of the potency
analysis (no fields, no poses), so it runs on any per-experiment export.

For each node: the 1 M HCl rows with a MWILA, the within-paper spread of a compound over
concentrations and methods, the between-paper spread of the same compound, the split-half
reliability across papers, the ceiling q2 of a perfect model of the compound medians, the common
slope of MWILA against ln C within a (compound, paper) series, the compounds shared by two or
more papers, and the papers with three or more of the node's compounds.

    python examples/corrosion_fe110/potency_reliability.py 157 158 161 4 7
    python examples/corrosion_fe110/potency_reliability.py examples/corrosion_fe110/shared_hcl_experiments.csv

An argument that ends in .csv is read as an export with the same columns (the second call is
the cross-chemotype export of the compounds measured in 1 M HCl by three or more papers).
Writes potency_reliability.csv and potency_reliability_pairs.csv next to this script (the
file names get the stem of a CSV argument as suffix) and prints the tables.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import potency_series as ps  # noqa: E402

NAMES = {"157": "benzimidazole", "158": "quinoline", "161": "benzothiazole",
         "4": "1,2,4-triazole", "7": "1,3,4-thiadiazole", "2": "imidazole"}


def load(node: str) -> pd.DataFrame:
    """A node id reads node<id>_experiments.csv next to this script; a path to a CSV with the
    same columns (for example the cross-chemotype export shared_hcl_experiments.csv) is read as is."""
    path = node if node.endswith(".csv") else os.path.join(HERE, f"node{node}_experiments.csv")
    return pd.read_csv(path)


def describe(node: str) -> dict:
    d = load(node)
    h = d[(d.medium == "HCl (1 M)") & d.MWILA.notna()].copy()
    nc = ps.noise_ceiling(h)
    y_c, slope, resid_sd = ps.concentration_corrected(h)
    papers_per_cpd = h.groupby("db_id").Refcode.nunique()
    cpds_per_paper = h.groupby("Refcode").db_id.nunique()
    multi = cpds_per_paper[cpds_per_paper >= 3].sort_values(ascending=False)
    out = {"node": node, "name": NAMES.get(node, os.path.basename(node) if node.endswith(".csv") else ""), "rows_all": len(d), "rows_1m_hcl": len(h),
           "compounds": nc["n_compounds"], "papers": h.Refcode.nunique(),
           "sd_medians": round(nc["sd_between_compounds"], 2),
           "sd_within_paper": round(nc["sd_within_paper"], 2),
           "sd_between_papers": round(nc["sd_between_papers"], 2) if nc["n_multi_paper"] >= 2 else np.nan,
           "shared_compounds": int((papers_per_cpd >= 2).sum()),
           "max_papers_per_compound": int(papers_per_cpd.max()),
           "split_half_r": round(nc["split_half_r"], 2) if nc["n_multi_paper"] >= 5 else np.nan,
           "ceiling_q2": round(nc["ceiling_q2"], 2) if nc["n_multi_paper"] >= 5 else np.nan,
           "slope_lnC": round(slope, 2), "resid_sd_series": round(resid_sd, 2),
           "papers_3plus": len(multi), "largest_series": int(multi.iloc[0]) if len(multi) else 0,
           "series_list": "; ".join(f"{k} ({v})" for k, v in multi.items())}
    return out


def repeat_pairs(nodes) -> pd.DataFrame:
    """Every pair of papers that measured the same compound in 1 M HCl: the difference of their
    per-paper medians, and whether the two papers have the same first author (the prefix of the
    Refcode).  Pairs from one laboratory agree better than pairs from two; only the latter
    measure reproducibility between laboratories."""
    import itertools
    rows = []
    for node in nodes:
        d = load(node)
        h = d[(d.medium == "HCl (1 M)") & d.MWILA.notna()]
        sd_node = h.groupby("db_id").MWILA.median().std()
        pm = h.groupby(["db_id", "Refcode"]).MWILA.median().reset_index()
        pm["lab"] = pm.Refcode.str.extract(r"^([a-z]+)")[0]
        for cid, g in pm.groupby("db_id"):
            for a, b in itertools.combinations(g.itertuples(index=False), 2):
                rows.append({"node": node, "db_id": cid, "same_author": a.lab == b.lab,
                             "diff": a.MWILA - b.MWILA, "sd_node": sd_node})
    return pd.DataFrame(rows)


def summarise_pairs(p: pd.DataFrame, label: str) -> str:
    if len(p) == 0:
        return f"{label}: no pairs"
    sd_pair = np.sqrt((p["diff"] ** 2).mean() / 2)  # between-paper sd of one paper's value
    sd_node = np.sqrt((p.sd_node ** 2).mean())
    return (f"{label}: {len(p)} pairs over {p.db_id.nunique()} compounds, median |difference| "
            f"{p['diff'].abs().median():.2f}, between-paper sd {sd_pair:.2f}, node sd {sd_node:.2f}, "
            f"implied reliability of one paper's value {1 - sd_pair ** 2 / sd_node ** 2:.2f}")


def main(argv=None):
    nodes = (argv if argv is not None else sys.argv[1:]) or ["157", "158", "161", "4", "7"]
    rows = [describe(n) for n in nodes]
    p = repeat_pairs(nodes)
    print("repeat measurements of one compound by two papers:")
    print(summarise_pairs(p[p.same_author], "  same first author"))
    print(summarise_pairs(p[~p.same_author], "  different first authors"))
    for node, q in p[~p.same_author].groupby("node"):
        print(summarise_pairs(q, f"    node {node}, different first authors"))
    if "157" in nodes and len(nodes) > 1:
        print(summarise_pairs(p[~p.same_author & (p.node != "157")], "  different first authors, nodes other than 157"))
    suffix = "_" + os.path.splitext(os.path.basename(nodes[0]))[0] if len(nodes) == 1 and nodes[0].endswith(".csv") else ""
    p.to_csv(os.path.join(HERE, f"potency_reliability_pairs{suffix}.csv"), index=False)
    t = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t.drop(columns=["series_list"]).set_index("node").T)
    for r in rows:
        print(f"node {r['node']} ({r['name']}): papers with 3+ compounds: {r['series_list'] or 'none'}")
    out = os.path.join(HERE, f"potency_reliability{suffix}.csv")
    t.to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
