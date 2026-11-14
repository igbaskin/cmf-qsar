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


# ---------------------------------------------------------------- the reference regression


def _points(slope=0.5, noise=None):
    """Two series with their own level and one slope: y = level + slope * x."""
    x = {"A": 100.0, "B": 120.0, "C": 150.0, "D": 200.0, "E": 230.0}
    series = {"first": (10.0, "ABCD"), "second": (40.0, "BCDE")}
    points = []
    for name, (level, solvents) in series.items():
        for k, s in enumerate(solvents):
            y = level + slope * x[s] + (0.0 if noise is None else noise[len(points)])
            points.append({"id": f"{name}-{k}", "series": name, "solvent": s, "parts": (s,),
                           "x": x[s], "y": y})
    return points


def test_the_solvents_of_an_electrolyte_come_from_its_name():
    assert CT.components("EC:DMC 1:9") == ("EC", "DMC")
    assert CT.components("EC:DMC") == ("EC", "DMC")
    assert CT.components("DMC") == ("DMC",)
    assert CT.components("poly(ethylene glycol) of molar mass 222") == (
        "poly(ethylene glycol) of molar mass 222",)
    assert CT.components("2,2,2-trifluoroethyl acetate") == ("2,2,2-trifluoroethyl acetate",)


def test_the_points_take_single_solvents_and_a_mixture_takes_its_strongest_solvent():
    rows = [_row(id="a", solvent="DMC", ea=40.0, solvent_series="S"),
            _row(id="b", solvent="EC", ea=60.0, solvent_series="S"),
            _row(id="c", solvent="EC:DMC 1:1 by volume", ea=58.0, solvent_series="S"),
            _row(id="d", solvent="PC", ea=64.0, is_range=True, solvent_series="S"),
            _row(id="e", solvent="THF", ea=30.0, solvent_series="T"),       # alone in its series
            _row(id="f", solvent="DMC", ea=33.0, solvent_series="")]
    binding = {"DMC": 190.0, "EC": 210.0, "THF": 180.0, "PC": 215.0}
    single = CT.regression_points(rows, binding)
    assert [p["id"] for p in single] == ["a", "b"]
    mixed = CT.regression_points(rows, binding, mixtures=True)
    assert [p["id"] for p in mixed] == ["a", "b", "c"]
    assert mixed[2]["parts"] == ("EC", "DMC") and mixed[2]["x"] == 210.0
    # a solvent that is left out takes its mixtures with it, and the series is then too small
    assert CT.regression_points(rows, binding, mixtures=True, leave_out=("EC",)) == []
    with pytest.raises(ValueError, match="no binding energy"):
        CT.regression_points(rows, {"DMC": 190.0})


def test_the_fit_has_one_slope_and_one_level_per_series():
    slope, means = CT.fit_slope(_points(slope=0.5))
    assert slope == pytest.approx(0.5)
    assert means["first"] == pytest.approx((142.5, 10.0 + 0.5 * 142.5))
    # no series with two different binding energies: the slope is 0 and not a division by zero
    flat = [dict(p, x=100.0) for p in _points()]
    assert CT.fit_slope(flat)[0] == 0.0


def test_a_held_out_solvent_is_predicted_from_the_other_solvents_of_its_series():
    predictions = CT.hold_out_solvents(_points(slope=0.5))
    assert len(predictions) == 8                         # every point once
    for q in predictions:
        assert q["regression"] == pytest.approx(q["y"])  # the data are exactly linear
        assert q["held_out"] == q["solvent"]
    a = [q for q in predictions if q["id"] == "first-0"][0]
    # the mean of the three points of its series that stay in: solvents B, C and D
    assert a["mean"] == pytest.approx(10.0 + 0.5 * (120.0 + 150.0 + 200.0) / 3.0)
    errors = CT._errors(predictions)
    assert errors["n"] == 8 and errors["regression"]["mae"] == pytest.approx(0.0, abs=1e-9)
    assert errors["mean"]["mae"] > 10.0


def test_a_point_whose_series_keeps_no_point_is_not_predicted():
    points = [p for p in _points() if p["series"] == "first"][:2]
    points.append({"id": "m", "series": "third", "solvent": "A:B", "parts": ("A", "B"),
                   "x": 120.0, "y": 50.0})
    points.append({"id": "n", "series": "third", "solvent": "A", "parts": ("A",),
                   "x": 100.0, "y": 45.0})
    predictions = CT.hold_out_solvents(points)
    # with A held out the third series keeps nothing; with B held out it keeps its point of A
    assert [(q["held_out"], q["id"]) for q in predictions if q["series"] == "third"] == [("B", "m")]


def test_the_gain_has_an_interval_over_the_held_out_solvents():
    predictions = CT.hold_out_solvents(_points(slope=0.5))
    g = CT.gain_interval(predictions, draws=200, seed=1)
    assert g["groups"] == 5 and g["gain"] > 10.0
    assert 0.0 < g["low"] <= g["gain"] <= g["high"]
    assert g == CT.gain_interval(predictions, draws=200, seed=1)


def test_an_exchange_of_the_binding_energies_seldom_does_as_well_on_linear_data():
    rng = [0.3, -0.2, 0.1, -0.4, 0.2, -0.1, 0.4, -0.3]
    points = _points(slope=0.5, noise=rng)
    test = CT.permutation_test(points, draws=300, seed=2)
    assert test["solvents"] == 5 and test["draws"] == 300
    # of the 120 orders of five values only the true one gives so small an error
    assert test["share_not_above"] < 0.05
    # when the values of a series are all equal, every exchange gives the same error
    flat = CT.permutation_test(_points(slope=0.0), draws=50, seed=2)
    assert flat["share_not_above"] == 1.0


def test_the_errors_of_one_solvent_in_two_series_bound_a_model_of_the_solvent_alone():
    def q(solvent, series, error):
        return {"held_out": solvent, "solvent": solvent, "series": series, "y": 50.0,
                "regression": 50.0 + error, "mean": 50.0}
    predictions = [q("DMC", "one", -10.0), q("DMC", "two", +12.0),          # median 1: 11 and 11
                   q("PC", "one", -6.0), q("PC", "two", -2.0), q("PC", "three", 7.0),
                   q("THF", "one", 30.0)]                                    # in one series only
    d = CT.series_disagreement(predictions)
    assert d["n"] == 5 and d["electrolytes"] == ["DMC", "PC"]
    assert d["regression_mae"] == pytest.approx((10 + 12 + 6 + 2 + 7) / 5.0)
    # the distances from the medians: 11, 11 and 4, 0, 9
    assert d["floor_mae"] == pytest.approx((11 + 11 + 4 + 0 + 9) / 5.0)
    assert CT.series_disagreement([q("THF", "one", 3.0)])["n"] == 0


def test_the_computed_binding_energies_are_checked_against_the_published_ones():
    rows = [{"solvent": name, "binding": value + 10.0, "binding_xtb": 2.0 * value}
            for name, value, _how, _source in CT.PUBLISHED_BINDING if name != "PC"]
    rows.append({"solvent": "PC", "binding": 225.0, "binding_xtb": 430.0})
    check = CT.check_binding(rows)
    assert check["n"] == len(CT.PUBLISHED_BINDING)       # PC has two published values
    assert check["orca"]["correlation"] > 0.99
    assert check["orca"]["mean_difference"] == pytest.approx(10.0, abs=0.5)
    assert check["xtb"]["mean_difference"] > 150.0


def test_the_sets_of_points_of_the_regression_on_the_collected_values():
    """The sizes do not depend on the binding energies, so stand-in values are enough."""
    binding = {"rows": [{"solvent": name, "binding": 150.0 + 7.0 * k}
                        for k, (name, _key, _smiles, _wraps) in enumerate(CT.SOLVENTS)]}
    reg = CT.regression_stage(binding, draws=20)
    sizes = {key: (v["n_series"], v["n_points"], v["n_solvents"])
             for key, v in reg["variants"].items()}
    assert sizes["single"] == (7, 20, 13)
    assert sizes["single_no_chelate"] == (7, 18, 11)
    assert sizes["mixtures"] == (10, 36, 13)
    assert sizes["mixtures_no_chelate"] == (10, 34, 11)
    assert sizes["published"] == (4, 12, 8)
    assert reg["repeatability"] == pytest.approx(5.0, abs=0.05)
    # every held-out value of the first set is predicted once
    assert reg["variants"]["single"]["held_out"]["n"] == 20


def _stored(name):
    import json
    with open(os.path.join(CT.RESULTS, name), encoding="utf-8") as fh:
        return json.load(fh)


def test_the_stored_binding_energies_and_their_check_against_the_papers():
    """The numbers the plan entry quotes. The file is the result of ``--stage binding``."""
    b = _stored("binding.json")
    by = {r["solvent"]: r for r in b["rows"]}
    assert len(by) == 14 and sum(r["in_a_series"] for r in b["rows"]) == 13
    assert by["water"]["binding"] == pytest.approx(155.5, abs=0.05)
    assert by["DMC"]["binding"] == pytest.approx(197.5, abs=0.05)
    assert by["DMSO"]["binding"] == pytest.approx(250.2, abs=0.05)
    # the two solvents that wrap the ion bind far more strongly than every other one
    assert by["diglyme"]["donors_at_ion"] == 3 and by["diglyme"]["binding"] > 350.0
    peg = by["poly(ethylene glycol) of molar mass 222"]
    assert peg["donors_at_ion"] == 5 and peg["binding"] > 480.0
    assert {r["solvent"] for r in b["rows"] if r["geometry"] == "GFN2-xTB"} == {
        "diglyme", "poly(ethylene glycol) of molar mass 222", "hexamethylphosphoramide"}
    # the fluorinated ester has two ways of binding, 30 kJ/mol apart
    ester = by["2,2,2-trifluoroethyl acetate"]
    assert ester["ways_of_binding"] == pytest.approx([177.3, 207.5], abs=0.05)
    check = b["check"]
    assert check["n"] == 11 and check["two_ways"] == ["2,2,2-trifluoroethyl acetate"]
    assert check["orca"]["correlation"] == pytest.approx(0.936, abs=0.001)
    assert check["xtb"]["correlation"] == pytest.approx(0.803, abs=0.001)
    assert check["orca_one_way"]["correlation"] == pytest.approx(0.986, abs=0.001)
    assert check["orca_one_way"]["sd_difference"] == pytest.approx(6.4, abs=0.05)
    shift = b["xtb_geometry_shift"]
    assert shift["n"] == 11 and shift["mean"] == pytest.approx(-1.9, abs=0.05)
    assert shift["largest"] == pytest.approx(5.8, abs=0.05)


def test_the_stored_regression_is_the_regression_of_the_stored_binding_energies():
    """``--stage regression`` again, with fewer draws: the fit and the held-out errors do not
    depend on the draws."""
    stored = _stored("regression.json")
    again = CT.regression_stage(_stored("binding.json"), draws=20)
    for key, v in stored["variants"].items():
        w = again["variants"][key]
        assert w["slope"] == pytest.approx(v["slope"])
        for model in ("mean", "regression"):
            assert w["held_out"][model]["mae"] == pytest.approx(v["held_out"][model]["mae"])
        assert w["between_series"]["floor_mae"] == pytest.approx(v["between_series"]["floor_mae"])


def test_what_the_regression_on_the_binding_energy_says():
    """The numbers the plan entry quotes."""
    v = _stored("regression.json")["variants"]
    # with every single solvent the regression is no better than the mean of the series
    assert v["single"]["gain"]["gain"] == pytest.approx(-0.5, abs=0.05)
    assert v["single"]["gain"]["low"] < 0.0 < v["single"]["gain"]["high"]
    assert v["single"]["permutation"]["share_not_above"] > 0.2
    # without the two solvents that wrap the ion it is better, and not by chance
    w = v["single_no_chelate"]
    assert w["slope"] == pytest.approx(0.559, abs=0.001)
    assert w["correlation_inside_series"] == pytest.approx(0.87, abs=0.005)
    assert w["held_out"]["mean"]["mae"] == pytest.approx(17.9, abs=0.05)
    assert w["held_out"]["regression"]["mae"] == pytest.approx(10.3, abs=0.05)
    assert (w["gain"]["gain"], w["gain"]["low"], w["gain"]["high"]) == pytest.approx(
        (7.6, 0.2, 13.2), abs=0.05)
    assert w["permutation"]["share_not_above"] < 0.001
    m = v["mixtures_no_chelate"]
    assert m["held_out"]["regression"]["mae"] == pytest.approx(7.1, abs=0.05)
    assert (m["gain"]["gain"], m["gain"]["low"], m["gain"]["high"]) == pytest.approx(
        (6.1, 3.8, 9.1), abs=0.05)
    # the binding energies of the papers give the same answer on the eight solvents they cover
    p = v["published"]
    assert p["gain"]["gain"] == pytest.approx(4.7, abs=0.05) and p["gain"]["low"] > 0.0
    # on the solvents measured in several series, a model of the solvent alone can remove
    # about 1 kJ/mol of the error of the regression and no more
    assert w["between_series"]["n"] == 12
    assert w["between_series"]["regression_mae"] == pytest.approx(11.3, abs=0.05)
    assert w["between_series"]["floor_mae"] == pytest.approx(10.4, abs=0.05)
    assert m["between_series"]["regression_mae"] == pytest.approx(8.3, abs=0.05)
    assert m["between_series"]["floor_mae"] == pytest.approx(7.5, abs=0.05)


def test_the_table_holds_the_reference_when_its_results_are_given():
    rows, sources = CT.read_labels(), CT.read_sources()
    res = dict(CT.summarise(rows, sources), cold=[])
    text = CT.table(res, sources, _stored("binding.json"), _stored("regression.json"))
    assert "## The reference: a regression on the binding energy of Li+ with the solvent" in text
    assert "| water | 155.5 | ORCA | 1 |" in text
    assert "| single solvents, without the two that wrap the ion | 7 | 18 | 11 | +0.559 |" in text
    assert "### What a model of the solvent alone cannot remove" in text
    # without the results the table has no such section
    assert "## The reference" not in CT.table(res, sources)


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
