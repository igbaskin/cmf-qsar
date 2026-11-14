"""The collected activation energies of charge transfer (plan item BT12): the table and its report.

No test needs a download or a cell model.
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import run_charge_transfer as CT  # noqa: E402


def _row(**kw):
    base = {"id": "X-01", "source": "S1", "electrode_class": "graphite",
            "solvent_class": "EC-based carbonate", "film": "formed", "additive": "none",
            "solvent": "EC:DMC", "salt": "LiPF6", "state": "", "read_from": "primary",
            "ea": 60.0, "is_range": False}
    base.update(kw)
    return base


def test_an_energy_in_another_unit_is_converted_and_an_unknown_unit_is_refused():
    assert CT.to_kj(57.7, "kJ/mol") == 57.7
    assert CT.to_kj(0.60, "eV") == pytest.approx(57.891)
    assert CT.to_kj(15.1, "kcal/mol") == pytest.approx(63.178, abs=1e-3)
    with pytest.raises(ValueError):
        CT.to_kj(14.0, "cal/mol")


def test_the_table_loads_and_every_row_names_a_paper_that_has_a_record():
    rows, sources = CT.read_labels(), CT.read_sources()
    assert len(rows) == 90
    assert len({r["id"] for r in rows}) == len(rows)
    assert {r["source"] for r in rows} <= set(sources)
    assert all(r["location"] or r["cited_in"] for r in rows)       # the place it was read


def test_a_primary_value_comes_from_a_paper_that_was_read_and_a_secondary_one_says_where():
    rows, sources = CT.read_labels(), CT.read_sources()
    for r in rows:
        status = sources[r["source"]]["status"]
        if r["read_from"] == "primary":
            assert status == "read", r["id"]
            assert not r["cited_in"], r["id"]
        else:
            assert status.startswith("cited only"), r["id"]
            assert r["cited_in"], r["id"]
            # the paper it was read in is one that was read
            assert any(sources[k]["status"] == "read" for k in sources if k in r["cited_in"])


def test_a_range_gives_its_middle_and_says_that_it_is_a_range():
    rows = {r["id"]: r for r in CT.read_labels()}
    r = rows["A04-01"]                                 # 53 to 59 kJ/mol
    assert r["is_range"] and r["ea"] == pytest.approx(56.0)
    assert (r["ea_low"], r["ea_high"]) == (53.0, 59.0)
    k = rows["K19-01"]                                 # 0.58 to 0.75 eV
    assert k["ea_low"] == pytest.approx(0.58 * 96.485)
    single = rows["J18-01"]
    assert not single["is_range"] and single["ea"] == 57.7 and single["ea_sd"] is None
    assert rows["K20-01"]["ea_sd"] == pytest.approx(0.02 * 96.485)


def test_the_reference_system_leaves_out_additives_cycled_films_and_other_electrodes():
    assert CT.is_reference(_row())
    assert not CT.is_reference(_row(additive="2 % VC"))
    assert not CT.is_reference(_row(film="cycled"))
    assert not CT.is_reference(_row(electrode_class="layered oxide"))
    assert not CT.is_reference(_row(solvent_class="linear carbonate only"))


def test_the_repeatability_takes_one_value_per_study():
    rows = [_row(source="A", ea=50.0), _row(source="A", ea=54.0), _row(source="B", ea=60.0),
            _row(source="C", ea=70.0, additive="2 % VC")]
    rep = CT.repeatability(rows)
    assert rep["studies"] == {"A": 52.0, "B": 60.0}
    assert rep["n"] == 2 and rep["mean"] == pytest.approx(56.0)
    assert rep["sd"] == pytest.approx(5.657, abs=1e-3)


def test_the_range_inside_a_study_names_what_differs_and_leaves_out_ranges():
    rows = [_row(source="A", solvent="PC", ea=64.0), _row(source="A", solvent="DMC", ea=37.0),
            _row(source="A", solvent="EC:DMC", ea=90.0, is_range=True),
            _row(source="B", ea=50.0)]
    out = CT.within_study(rows)
    assert len(out) == 1                               # study B has one value
    w = out[0]
    assert (w["source"], w["n"], w["low"], w["high"], w["range"]) == ("A", 2, 37.0, 64.0, 27.0)
    assert w["varied"] == ["solvent"]


def test_a_solvent_has_one_name_in_every_study():
    assert CT.solvent_name("EC:DMC 1:1 by volume") == "EC:DMC 1:1"
    assert CT.solvent_name("EC:DMC 50:50 by weight") == "EC:DMC 1:1"
    assert CT.solvent_name("DMC") == "DMC"


def _series_rows():
    return [_row(source="A", solvent="DMC", ea=40.0, solvent_series="A: graphite"),
            _row(source="A", solvent="EC:DMC 1:1 by volume", ea=58.0, solvent_series="A: graphite"),
            _row(source="A", solvent="PC", ea=70.0, is_range=True, solvent_series="A: graphite"),
            _row(source="B", solvent="EC:DMC 50:50 by weight", ea=52.0, solvent_series="B: titanate"),
            _row(source="B", solvent="DMC", ea=51.0, solvent_series="B: titanate"),
            _row(source="B", solvent="THF", ea=40.0, solvent_series="B: titanate"),
            _row(source="C", solvent="THF", ea=42.0, solvent_series=""),
            _row(source="D", solvent="THF", ea=44.0)]


def test_a_series_holds_the_rows_that_name_it_in_the_order_of_their_values():
    series = CT.solvent_series(_series_rows())
    assert [q["series"] for q in series] == ["A: graphite", "B: titanate"]
    a, b = series
    assert a["solvents"] == ["DMC", "EC:DMC 1:1"] and a["range"] == 18.0     # the range row is left out
    assert b["solvents"] == ["THF", "DMC", "EC:DMC 1:1"]
    assert b["one_solvent"] == ["DMC", "THF"]
    count = CT.solvent_count(series)
    assert (count["n_series"], count["n_rows"], count["largest"]) == (2, 5, 3)
    assert count["one_solvent"] == ["DMC", "THF"]
    assert count["one_solvent_in_two_series"] == ["DMC"]


def test_a_pair_of_solvents_is_compared_between_the_series_that_measured_both():
    series = CT.solvent_series(_series_rows())
    contrasts = CT.shared_contrasts(series)
    assert len(contrasts) == 1                         # THF is in one series only
    c = contrasts[0]
    assert (c["first"], c["second"]) == ("DMC", "EC:DMC 1:1")
    assert [d["difference"] for d in c["differences"]] == [-18.0, -1.0]
    shared = CT.shared_solvents(series)
    assert [q["solvent"] for q in shared] == ["DMC"]
    assert [x["value"] for x in shared[0]["values"]] == [40.0, 51.0] and shared[0]["range"] == 11.0


def test_the_collected_values_and_what_they_say():
    """The numbers the plan entry quotes."""
    rows, sources = CT.read_labels(), CT.read_sources()
    res = CT.summarise(rows, sources)
    assert (res["n_primary"], res["n_secondary"], res["n_studies"]) == (88, 2, 16)
    rep = res["repeatability"]
    assert rep["n"] == 5
    assert rep["mean"] == pytest.approx(59.7, abs=0.05) and rep["sd"] == pytest.approx(5.0, abs=0.05)
    assert rep["studies"]["Yamada2009"] == 58.0        # the paper, not the 55 of the review
    neg, pos = res["means"]["negative"], res["means"]["positive"]
    assert neg["mean"] == pytest.approx(61.7, abs=0.05) and neg["shift"] == pytest.approx(26.7, abs=0.05)
    assert pos["mean"] == pytest.approx(68.0, abs=0.05) and pos["shift"] == pytest.approx(50.2, abs=0.05)
    assert res["to_read"] == ["KyotoRepository", "YamadaSagane2009"]


def test_the_series_of_the_collected_values_and_the_one_pair_they_share():
    """The numbers behind the verdict on part (c) of the gate."""
    res = CT.summarise(CT.read_labels(), CT.read_sources())
    ranges = {q["series"]: q["range"] for q in res["series"]}
    assert ranges["Ishihara2014: lithium titanate, no film"] == pytest.approx(27.0)
    assert ranges["Abe2005: solid electrolyte, no film"] == pytest.approx(38.4)
    assert ranges["Sawayama2025: graphite, film formed in each electrolyte with ethylene "
                  "sulfite"] == pytest.approx(29.1)
    assert ranges["Xu2010: lithium titanate, no film"] == pytest.approx(14.86)
    count = res["solvent_count"]
    assert (count["n_series"], count["n_rows"], count["largest"]) == (10, 36, 8)
    assert len(count["one_solvent"]) == 13
    assert count["one_solvent_in_two_series"] == ["DMC", "PC", "THF", "acetonitrile", "water"]
    # one pair of solvents was measured in more than one series, and the series do not agree
    assert len(res["contrasts"]) == 1
    c = res["contrasts"][0]
    assert (c["first"], c["second"]) == ("DMC", "EC:DMC 1:1")
    d = {x["series"].split(":")[0]: x["difference"] for x in c["differences"]
         if "pentafluorostyrene" not in x["series"]}
    assert d["Ishihara2014"] == pytest.approx(-17.0) and d["Xu2010"] == pytest.approx(-1.41)
    assert d["Yamada2009"] == pytest.approx(-18.0)


def test_the_table_is_written_without_the_cell_model_results():
    rows, sources = CT.read_labels(), CT.read_sources()
    res = dict(CT.summarise(rows, sources), cold=[])
    text = CT.table(res, sources)
    assert "| negative | 18 | 61.7 |" in text
    assert "What that difference costs a cold cell" not in text
    assert "| YamadaSagane2009 |" in text                # a paper that is still to be read
    assert "| DMC | EC:DMC 1:1 | Xu2010: lithium titanate, no film | -1.4 |" in text


def test_a_row_without_a_record_of_its_paper_is_refused():
    with pytest.raises(ValueError):
        CT.summarise([_row(source="Nobody1999")], {"S1": {"status": "read"}})
