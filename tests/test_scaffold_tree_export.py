"""Tests for examples/scaffold_tree_export.py on a small synthetic tree in the artifact's format."""
import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

import scaffold_tree_export as ste  # noqa: E402

BENZIMIDAZOLE = "c1ccc2[nH]cnc2c1"


def _comp(cid, name, smiles, path, hcl=None, h2so4=None, pooled=None, mw=150.0, dg=-30.0):
    """A compound record as the artifact stores it (the adjusted value is raw - 1 here)."""
    adj = (lambda v: None if v is None else round(v - 1.0, 2))
    thermo = [dg] + [None] * 8
    return [cid, name, mw, 3, 2, pooled, adj(pooled), hcl, adj(hcl), path, smiles, thermo,
            h2so4, adj(h2so4), None]


def _node(s, r, p, n):
    return {"s": s, "r": r, "p": p, "n": n, "nd": 0, "hn": 0}


@pytest.fixture(scope="module")
def tree():
    """Three nodes (benzene root, benzimidazole below it, pyridine root) and eight compounds."""
    nodes = [_node("c1ccccc1", 1, -1, 6), _node(BENZIMIDAZOLE, 2, 0, 5), _node("c1ccncc1", 1, -1, 2)]
    comps = [
        _comp(1, "Benzimidazole", BENZIMIDAZOLE, [0, 1], hcl=5.6, h2so4=5.0, pooled=5.5),
        _comp(2, "2‑Methyl‑benzimidazole", "Cc1nc2ccccc2[nH]1", [0, 1], hcl=6.7, pooled=6.2),
        _comp(3, "5-Nitrobenzimidazole", "O=[N+]([O-])c1ccc2nc[nH]c2c1", [0, 1], hcl=6.6, pooled=6.6, dg=None),
        _comp(4, "2-Aminobenzimidazole", "Nc1nc2ccccc2[nH]1", [0, 1], hcl=8.1, pooled=8.1),
        _comp(5, "2-Phenylbenzimidazole, H2SO4 only", "c1ccc(-c2nc3ccccc3[nH]2)cc1", [0, 1], h2so4=9.0, pooled=9.0),
        _comp(6, "Aniline", "Nc1ccccc1", [0], hcl=4.0, pooled=4.0),
        _comp(7, "Pyridine", "c1ccncc1", [2], hcl=3.9, pooled=3.9),
        _comp(8, "水杨", "Cc1ccncc1", [2], hcl=4.4, pooled=4.4),     # a name with no ASCII in it
    ]
    return {"date": "2026-10-16", "fit": {"intercept": 0.0, "slope": 0.0}, "nodes": nodes,
            "svgs": [""] * 3, "comps": comps, "acyc": []}


def test_load_artifact_from_html_and_json(tree, tmp_path):
    html = tmp_path / "page.html"
    html.write_text('<html><body><p>x</p>\n<script type="application/json" id="data">'
                    + json.dumps(tree) + '</script>\n<script>var a=1;</script></body></html>', encoding="utf-8")
    js = tmp_path / "page.json"
    js.write_text(json.dumps(tree), encoding="utf-8")
    for p in (html, js):
        d = ste.load_artifact(str(p))
        assert len(d["nodes"]) == 3 and len(d["comps"]) == 8
    bad = tmp_path / "bad.html"
    bad.write_text("<html>no data here</html>", encoding="utf-8")
    with pytest.raises(ValueError):
        ste.load_artifact(str(bad))


def test_rank_nodes_counts_measured_compounds(tree):
    # 1 M HCl: benzene sees the four measured benzimidazoles and aniline, pyridine two
    ranked = ste.rank_nodes(tree, "hcl", False, min_compounds=1)
    assert [(s.index, s.n) for s in ranked] == [(0, 5), (1, 4), (2, 2)]
    assert [(s.index, s.n) for s in ste.rank_nodes(tree, "hcl", False, min_compounds=3)] == [(0, 5), (1, 4)]
    assert [s.index for s in ste.rank_nodes(tree, "hcl", False, min_compounds=1, max_depth=1)] == [0, 2]
    # H2SO4: only two compounds have a value, both under benzimidazole
    h2 = ste.rank_nodes(tree, "h2so4", True, min_compounds=1)
    assert [(s.index, s.n) for s in h2] == [(0, 2), (1, 2)]
    assert np.allclose(h2[1].values, [4.0, 8.0])        # the adjusted values
    # explicit nodes are not filtered, just ordered
    assert [s.index for s in ste.rank_nodes(tree, "hcl", False, min_compounds=100, nodes=[2, 1])] == [1, 2]
    with pytest.raises(ValueError):
        ste.rank_nodes(tree, "hno3")


def test_helpers():
    assert ste.safe_scaffold(BENZIMIDAZOLE) == "c1ccc2_nH_cnc2c1"
    assert ste.safe_scaffold("C(=Nc1ccccc1)c1ccccc1") == "C_Nc1ccccc1_c1ccccc1"
    assert ste.clean_name("2‑Methyl‑benzimidazole", "x") == "2-Methyl-benzimidazole"
    assert ste.clean_name("N‑(4,6‑dimethyl)\r\nbenzene", "x") == "N-(4,6-dimethyl) benzene"
    assert ste.clean_name("水杨", "cpd_8") == "cpd_8"
    s = ste.node_series({"nodes": [_node("c1ccccc1", 1, -1, 1)],
                         "comps": [_comp(1, "a", "c1ccccc1", [0], hcl=v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]}, 0)
    assert s.quartiles() == {"min": 1.0, "q1": 2.0, "median": 3.0, "q3": 4.0, "max": 5.0}


def test_write_node_csv_columns_and_nulls(tree, tmp_path):
    series = ste.rank_nodes(tree, "hcl", False, min_compounds=1)
    paths = ste.export(tree, str(tmp_path), series, "hcl", False)
    assert os.path.basename(paths[1]) == "1_c1ccc2_nH_cnc2c1.csv"
    with open(paths[1], newline="") as fh:
        rows = list(csv.DictReader(fh))
    # the compound without a 1 M HCl value is skipped; the names are ASCII
    assert [r["name"] for r in rows] == ["Benzimidazole", "2-Methyl-benzimidazole", "5-Nitrobenzimidazole",
                                         "2-Aminobenzimidazole"]
    header = list(rows[0].keys())
    assert header[:8] == ["smiles", "name", "activity", "mw", "n_papers", "dG_ads", "scaffold", "depth"]
    # the other adjustment of the chosen medium, the H2SO4 pair (one compound has it) and the pooled pair
    assert header[8:13] == ["mwila_hcl_adj", "mwila_h2so4_raw", "mwila_h2so4_adj", "mwila_pooled_raw",
                            "mwila_pooled_adj"]
    assert header[13:] == ["leaf_scaffold", "n_rows", "db_id"]
    assert rows[0]["activity"] == "5.6" and rows[0]["mwila_hcl_adj"] == "4.6" and rows[0]["mwila_h2so4_raw"] == "5"
    assert rows[1]["mwila_h2so4_raw"] == "" and rows[2]["dG_ads"] == ""          # nulls are empty cells
    assert {r["scaffold"] for r in rows} == {BENZIMIDAZOLE} and {r["depth"] for r in rows} == {"2"}
    assert [r["db_id"] for r in rows] == ["1", "2", "3", "4"]
    # the pyridine node has no H2SO4 data at all, so those columns are absent
    with open(paths[2], newline="") as fh:
        header2 = next(csv.reader(fh))
    assert "mwila_h2so4_raw" not in header2 and "mwila_hcl_adj" in header2
    with open(paths[2], newline="") as fh:
        names = [r["name"] for r in csv.DictReader(fh)]
    assert names == ["Pyridine", "cpd_8"]
    table = ste.summary_table(series)
    assert table.splitlines()[1].split()[:3] == ["0", "1", "5"] and BENZIMIDAZOLE in table


def test_csv_loads_in_the_app_and_aligns_on_the_scaffold(tree, tmp_path):
    from cmfqsar.dataio import load_dataset
    series = ste.rank_nodes(tree, "hcl", False, min_compounds=4)
    paths = ste.export(tree, str(tmp_path), series, "hcl", False)
    ds = load_dataset(paths[1], activity_prop="activity")
    assert len(ds) == 4 and ds.names[0] == "Benzimidazole"
    assert np.allclose(ds.activity, [5.6, 6.7, 6.6, 8.1])
    assert {"mw", "dG_ads", "mwila_hcl_adj", "mwila_pooled_raw", "scaffold"} <= set(ds.properties)
    ds.set_activity_from_property("mwila_hcl_adj")                   # what the property picker does
    assert np.allclose(ds.activity, [4.6, 5.7, 5.6, 7.1])
    res = ste.check_csv(paths[1], BENZIMIDAZOLE, n_align=3)
    assert res.ok and not res.mcs_short
    assert res.n_loaded == 4 and res.scaffold_atoms == 9 and res.matched == [9, 9, 9]
    assert max(res.rmsd) < 0.5
    # a scaffold the compounds do not contain in full is reported, not raised
    res = ste.check_csv(paths[1], "c1ccc2ncccc2c1", n_align=2)        # quinoline vs benzimidazoles
    assert res.ok and res.mcs_short and all(m < 10 for m in res.matched)


def test_main_runs_end_to_end(tree, tmp_path, capsys):
    page = tmp_path / "tree.json"
    page.write_text(json.dumps(tree), encoding="utf-8")
    out = tmp_path / "out"
    rc = ste.main([str(page), "--outdir", str(out), "--min-compounds", "4", "--check-molecules", "2"])
    assert rc == 0
    assert sorted(p.name for p in out.iterdir()) == ["0_c1ccccc1.csv", "1_c1ccc2_nH_cnc2c1.csv"]
    text = capsys.readouterr().out
    assert "2 CSV files written" in text and "2 of 2 files pass" in text
    assert "MCS 9,9 of 9 scaffold atoms" in text
    rc = ste.main([str(page), "--outdir", str(out), "--medium", "h2so4", "--min-compounds", "50", "--no-check"])
    assert rc == 1


def test_protonator_caps_added_protons():
    pytest.importorskip("dimorphite_dl")
    one, two, free = ste.Protonator(0.0, 1), ste.Protonator(0.0, 2), ste.Protonator(0.0, None)
    # one basic site: the same at any cap
    assert one(BENZIMIDAZOLE) == free(BENZIMIDAZOLE) == "c1ccc2[nH+]c[nH]c2c1"
    # 2-aminobenzothiazole: the rules charge both nitrogens, the cap keeps one of them
    assert ste.net_charge(free("Nc1nc2ccccc2s1")) == 2 and ste.net_charge(one("Nc1nc2ccccc2s1")) == 1
    # the most basic site gets the proton: the alkylamine, not the pyridine
    assert one("NCCc1ccncc1") == "[NH3+]CCc1ccncc1" and two("NCCc1ccncc1") == "[NH3+]CCc1cc[nH+]cc1"
    # equivalent sites change together and exceed the cap: as many as the cap allows are kept
    assert ste.net_charge(one("NCCN")) == 1 and ste.net_charge(two("NCCN")) == 2
    assert ste.net_charge(one("c1ccc(-c2n[nH]c(-c3ccccc3)n2)cc1")) == 1
    # a permanent charge does not count against the cap; an acid stays neutral at pH 0
    assert one("C[n+]1ccn(CCN)c1") == "C[n+]1ccn(CC[NH3+])c1"
    assert ste.net_charge(one("OC(=O)c1ccccc1")) == 0
    # phenols that lose their proton at a high pH must not hide two amines gaining one
    assert one("Oc1ccccc1CNCCNCc1ccccc1O").count("+") == 1
    assert one("OC(=O)CCN") == "[NH3+]CCC(=O)O"
    # at pH 7 the benzimidazole (rule pKa below 7) stays neutral
    assert ste.net_charge(ste.Protonator(7.0, 1)(BENZIMIDAZOLE)) == 0
    with pytest.raises(ValueError):
        one("not a smiles")


def test_protonated_export(tree, tmp_path):
    pytest.importorskip("dimorphite_dl")
    from cmfqsar.dataio import load_dataset
    series = ste.rank_nodes(tree, "hcl", False, min_compounds=4)
    paths = ste.export(tree, str(tmp_path), series, "hcl", False, ste.Protonator(0.0, 1))
    with open(paths[1], newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0].keys())[-2:] == ["smiles_neutral", "charge"]
    assert [r["smiles_neutral"] for r in rows] == [BENZIMIDAZOLE, "Cc1nc2ccccc2[nH]1",
                                                   "O=[N+]([O-])c1ccc2nc[nH]c2c1", "Nc1nc2ccccc2[nH]1"]
    assert all(r["charge"] == "1" and "[nH+]" in r["smiles"] for r in rows)
    ds = load_dataset(paths[1], activity_prop="activity")
    assert len(ds) == 4 and "charge" in ds.properties
    assert all(sum(a.GetFormalCharge() for a in m.GetAtoms()) == 1 for m in ds.mols)
    # the neutral scaffold still aligns the protonated molecules in full
    res = ste.check_csv(paths[1], BENZIMIDAZOLE, n_align=2)
    assert res.ok and res.matched == [9, 9]
    # the end-to-end run reports the charges
    page = tmp_path / "tree.json"
    page.write_text(json.dumps(tree), encoding="utf-8")
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = ste.main([str(page), "--outdir", str(tmp_path / "p"), "--min-compounds", "4", "--protonate",
                       "--no-check"])
    assert rc == 0 and "protonated at pH 0" in buf.getvalue() and "+1: 9" in buf.getvalue()
