"""The benchmark of plan item BT9, ``benchmarks/run_diffusion.py``, without a simulation.

A run is replaced by stored centres of random walkers, and the measured reference by fixed
numbers, so that the path from a stored run to the table is tested on any machine.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import run_diffusion as R  # noqa: E402

MEASURED = {"conductivity": 0.9, "diffusivity": 2.7e-10, "transference": 0.2,
            "thermodynamic_factor": 2.4, "source": "made up"}
SECOND = {"conductivity": 0.95, "diffusivity": 1.8e-10, "inside": True, "source": "made up"}
LIMITS = [{"rate": 1.0, "low": 0.32, "high": 20.0, "high_open": True},
          {"rate": 1.5, "low": 0.49, "high": 20.0, "high_open": True},
          {"rate": 2.0, "low": 0.67, "high": 20.0, "high_open": True},
          {"rate": 2.5, "low": 0.83, "high": 1.5, "high_open": False}]


def stored_run(cache, name, frames=4001, pairs=12, solvent=30, seed=3, speed=1.0):
    """Write a made-up run into `cache` under the directory the benchmark expects."""
    folder = R.run_dir(name, str(cache))
    os.makedirs(folder)
    rng = np.random.default_rng(seed)
    names = ["EC"] * 10 + ["EMC"] * (solvent - 10) + ["ANI"] * pairs + ["LI"] * pairs
    steps = speed * np.array([0.4] * solvent + [0.3] * pairs + [0.2] * pairs)
    centres = np.cumsum(rng.normal(0.0, 1.0, (frames, len(names), 3)) * steps[None, :, None],
                        axis=0)
    np.savez_compressed(os.path.join(folder, "centres.npz"),
                        centres=centres.astype(np.float32), names=np.array(names),
                        masses=np.array([88.0] * 10 + [104.0] * (solvent - 10)
                                        + [145.0] * pairs + [6.9] * pairs),
                        charges=np.array([0.0] * solvent + [-1.0] * pairs + [1.0] * pairs),
                        dt_ps=1.0, side=30.0)
    record = {"centres": "/a/path/of/another/machine/centres.npz", "ion_residue": "LI",
              "anion_residue": "ANI", "temperature": 298.15, "volume": 27000.0, "side": 30.0,
              "atoms": 400, "pairs": pairs, "density": 1.19, "realised_molality": 1.1,
              "mean_kinetic_temperature": 298.0, "ns_per_day": 100.0, "seconds": 3456.0,
              "settings": {"ion_charge_scale": 1.0}}
    with open(os.path.join(folder, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh)
    return folder


@pytest.fixture
def fixed_reference(monkeypatch):
    monkeypatch.setattr(R, "measured_reference", lambda molarity, temperature: dict(MEASURED))
    monkeypatch.setattr(R, "calisol_conductivity", lambda: 0.88)
    monkeypatch.setattr(R, "second_reference", lambda molarity, temperature: dict(SECOND))


def test_the_runs_are_named_and_an_unknown_one_is_refused():
    assert R.settings_of("full").ion_charge_scale == 1.0
    assert R.settings_of("scaled").ion_charge_scale == 0.8
    with pytest.raises(SystemExit, match="no run is called"):
        R.settings_of("polarisable")


def test_a_run_that_was_not_simulated_says_how_to_make_it(tmp_path):
    with pytest.raises(SystemExit, match="--stage simulate --runs full"):
        R.read_run("full", str(tmp_path))


def test_the_centres_are_read_from_this_machine_and_not_from_the_record(tmp_path):
    folder = stored_run(tmp_path, "full", frames=300)
    run = R.read_run("full", str(tmp_path))
    assert run["centres"] == os.path.join(folder, "centres.npz")


def test_a_stored_run_becomes_a_result(tmp_path, fixed_reference):
    stored_run(tmp_path, "full")
    res = R.analyse_run("full", str(tmp_path))
    assert res["run"] == "full" and res["pairs"] == 12
    assert res["salt_with_factor"] == pytest.approx(2.4 * res["salt_ideal"])
    assert res["ratio_with_factor"] == pytest.approx(res["salt_with_factor"] / 2.7e-10)
    assert res["ratio_conductivity"] == pytest.approx(
        res["whole"]["conductivity"]["sigma"] / 0.9)
    assert res["calisol_conductivity"] == 0.88
    # the run is 4 ns long, so it is read at all four lengths, and a longer reading has the
    # later window
    assert [round(p["length_ps"]) for p in res["lengths"]] == [500, 1000, 2000, 4000]
    json.dumps(res)                                    # everything in it can be written


def test_a_short_run_is_read_only_at_the_lengths_it_has(tmp_path, fixed_reference):
    stored_run(tmp_path, "full", frames=1201)
    res = R.analyse_run("full", str(tmp_path))
    assert [round(p["length_ps"]) for p in res["lengths"]] == [500, 1000]


def test_the_rates_a_coefficient_serves():
    assert R.served_rates(0.60, LIMITS) == [1.0, 1.5]
    assert R.served_rates(0.20, LIMITS) == []
    assert R.served_rates(1.20, LIMITS) == [1.0, 1.5, 2.0, 2.5]
    assert R.served_rates(3.00, LIMITS) == [1.0, 1.5, 2.0]     # above a closed upper end
    assert R.tolerance_limits(os.path.join(ROOT, "no", "such", "folder")) == []


def test_the_table_names_what_was_compared(tmp_path, fixed_reference):
    stored_run(tmp_path / "cache", "full")
    stored_run(tmp_path / "cache", "scaled", seed=4, speed=1.6)
    results = str(tmp_path / "results")
    R.stage_analyse(["full", "scaled"], results, str(tmp_path / "cache"))
    rows = R.read_results(["full", "scaled"], results)
    assert [r["run"] for r in rows] == ["full", "scaled"]
    text = "\n".join(R.report_lines(rows, LIMITS))
    assert "| full | whole charges on the ions |" in text
    assert "| scaled | charges of the ions multiplied by 0.8 |" in text
    assert "salt diffusion coefficient 2.70e-10 m2/s" in text
    assert "CALiSol-23 gives a conductivity of 0.880 S/m" in text
    assert "| EC | Exponent | EMC | Exponent |" in text
    assert "0.32 of the measured value at 1C" in text
    assert "None of these coefficients is corrected for the size of the box" in text
    assert text.count("| full | 0.5 |") == 1 and text.count("| scaled | 4.0 |") == 1
    # without the tolerance stage the column says so instead of being empty
    assert "the tolerance stage has not been run" in "\n".join(R.report_lines(rows, []))


def _sized(run, side, pairs, shift):
    """A made-up result of one box: every coefficient is its limit less `shift`."""
    def rec(limit):
        return {"d": limit - shift, "se": 0.02 * limit, "exponent": 0.9}
    return {"run": run, "side": side, "pairs": pairs, "temperature": 298.15,
            "measured": dict(MEASURED), "ratio_with_factor": 0.9, "ratio_ideal": 0.4,
            "whole": {"cation": rec(1.0e-10), "anion": rec(1.5e-10),
                      "solvents": {"EC": rec(3.0e-10), "EMC": rec(2.5e-10)}}}


def test_the_size_study_needs_two_boxes_and_gives_the_infinite_one():
    from cmfqsar import transport as T
    eta = 2.5e-3
    rows = [_sized(name, side, pairs, T.yeh_hummer_correction(side, eta, 298.15))
            for name, side, pairs in (("scaled_small", 27.0, 14), ("scaled", 34.0, 27),
                                      ("scaled_large", 43.0, 55))]
    assert R.size_study(rows[:1]) is None
    assert R.size_study([dict(rows[0], run="full")]) is None
    study = R.size_study(rows)
    assert study["runs"] == ["scaled_small", "scaled", "scaled_large"]
    assert study["fit"]["viscosity"] == pytest.approx(eta, rel=1e-6)
    assert study["salt_ideal"] == pytest.approx(T.salt_diffusivity(1.0e-10, 1.5e-10), rel=1e-6)
    assert study["ratio_with_factor"] == pytest.approx(
        study["salt_ideal"] * 2.4 / 2.7e-10)
    text = "\n".join(R.size_lines(study, LIMITS))
    assert "| infinite box | | | 1.00e-10 | 1.50e-10 | 3.00e-10 | 2.50e-10 | 0 |" in text
    assert "a viscosity of 2.50 mPa s" in text
    assert "which the cell tolerates at 1C, 1.5C, 2C, 2.5C" in text
    assert "| scaled_small | 14 | 27.00 |" in text


def _repeat(run, values, block):
    """A made-up result of one repeat: the four species get `values`, each with the block error
    `block` times its value."""
    names = ("cation", "anion", "EC", "EMC")
    recs = {n: {"d": v, "se": block * v, "exponent": 0.9} for n, v in zip(names, values)}
    return {"run": run, "side": 34.0, "pairs": 27, "temperature": 298.15,
            "measured": dict(MEASURED), "ratio_with_factor": values[0] / 1.0e-10,
            "whole": {"cation": recs["cation"], "anion": recs["anion"],
                      "solvents": {"EC": recs["EC"], "EMC": recs["EMC"]},
                      "conductivity": {"sigma": 3.0e9 * values[0], "se": 0.04},
                      "ionicity": 0.5}}


def test_repeats_say_whether_one_run_sees_its_whole_error():
    base = np.array([1.0e-10, 1.5e-10, 3.0e-10, 2.5e-10])
    rows = [_repeat("scaled", base * 0.9, 0.02), _repeat("scaled_seed2", base * 1.0, 0.02),
            _repeat("scaled_seed3", base * 1.1, 0.02)]
    assert R.repeat_study(rows[:1]) is None
    study = R.repeat_study(rows)
    assert study["runs"] == ["scaled", "scaled_seed2", "scaled_seed3"]
    # the repeats differ by 10 % and each block error is 2 %, so the ratio is five
    assert study["species"]["cation"]["mean"] == pytest.approx(1.0e-10)
    assert study["species"]["cation"]["ratio"] == pytest.approx(5.0, rel=1e-6)
    assert study["error_factor"] == pytest.approx(5.0, rel=1e-6)
    assert study["ratio_with_factor"]["mean"] == pytest.approx(1.0)
    text = "\n".join(R.repeat_lines(study))
    assert "The median ratio is 5.0" in text and "too small by about that factor" in text
    assert "| Cation | 9.00e-11 | 1.00e-10 | 1.10e-10 | 1.00e-10 |" in text
    # repeats that agree inside their block errors leave the errors as they are
    calm = [_repeat("scaled", base * 0.99, 0.02), _repeat("scaled_seed2", base * 1.01, 0.02)]
    study = R.repeat_study(calm)
    assert study["error_factor"] == 1.0
    assert "see the whole error" in "\n".join(R.repeat_lines(study))


def test_the_size_study_uses_the_repeats_and_can_leave_a_box_out():
    from cmfqsar import transport as T
    eta = 2.5e-3
    rows = [_sized(name, side, pairs, T.yeh_hummer_correction(side, eta, 298.15))
            for name, side, pairs in (("scaled_small", 27.0, 14), ("scaled", 34.0, 27),
                                      ("scaled_large", 43.0, 55), ("scaled_huge", 54.0, 110))]
    for row in rows:
        row["ratio_with_factor"] = 1.0
        row["whole"]["conductivity"] = {"sigma": 0.3, "se": 0.04}
        row["whole"]["ionicity"] = 0.5
    twin = dict(rows[1], run="scaled_seed2")
    study = R.size_study(rows + [twin])
    assert study["repeated"] == ["scaled"] and study["excluded"] == []
    assert study["fit"]["viscosity"] == pytest.approx(eta, rel=1e-6)
    rest = R.size_study(rows + [twin], exclude=("scaled_small",))
    assert rest["runs"] == ["scaled", "scaled_large", "scaled_huge"]
    assert rest["excluded"] == ["scaled_small"]
    text = "\n".join(R.size_lines(rest, LIMITS))
    assert "Left out of this fit: `scaled_small`." in text
    assert "enters with the mean of its repeats" in text


def test_a_size_fit_with_the_wrong_slope_is_not_extrapolated():
    """Coefficients that fall as the box grows determine no correction, and the section says
    so and gives the largest box as it is."""
    rows = []
    for name, side, pairs, shift in (("scaled", 34.0, 27, 0.0), ("scaled_large", 43.0, 55, 5e-12)):
        row = _sized(name, side, pairs, shift)          # the larger box is the slower one
        row.update(ratio_with_factor=0.70, ratio_ideal=0.30)
        row["whole"]["conductivity"] = {"sigma": 0.3, "se": 0.04}
        row["whole"]["ionicity"] = 0.5
        rows.append(row)
    study = R.size_study(rows)
    assert study["determined"] is False and "salt_ideal" not in study
    assert study["largest"]["run"] == "scaled_large"
    assert all(v < 0 for v in study["change"].values())
    text = "\n".join(R.size_lines(study, LIMITS))
    assert "These boxes determine no correction" in text
    assert "| infinite box |" not in text and "not determined" in text
    assert "0.70 of the measured value, which the cell tolerates at 1C, 1.5C, 2C" in text
    assert "none mPa" not in text


def _charged(run, scale, speed, sigma):
    row = _repeat(run, np.array([1.0e-10, 1.5e-10, 3.0e-10, 2.5e-10]) * speed, 0.02)
    row["settings"] = {"ion_charge_scale": scale}
    row["whole"]["conductivity"] = {"sigma": sigma, "se": 0.04}
    row["whole"]["ionicity"] = sigma
    return row


def test_the_charges_of_the_ions_are_read_factor_by_factor():
    rows = [_charged("full", 1.0, 0.5, 0.1), _charged("scaled", 0.8, 0.9, 0.3),
            _charged("scaled_seed2", 0.8, 1.1, 0.5), _charged("scaled_07", 0.7, 1.5, 0.6)]
    big = dict(_charged("scaled_large", 0.8, 9.0, 0.3))
    big["settings"] = {"ion_charge_scale": 0.8, "n_solvent": 500}      # not the base box
    study = R.charge_study(rows + [big])
    assert [e["scale"] for e in study] == [1.0, 0.8, 0.7]
    assert study[1]["runs"] == ["scaled", "scaled_seed2"]
    assert study[1]["cation"][0] == pytest.approx(1.0e-10)             # the mean of two repeats
    assert study[1]["conductivity"][0] == pytest.approx(0.4)
    assert np.isnan(study[0]["ratio_with_factor"][1])                  # one run: no error
    assert R.charge_study(rows[:1]) is None
    text = "\n".join(R.charge_lines(study, LIMITS))
    assert "| 0.8 | 2 | 1.00e-10 +/- " in text
    assert "the cation rises 3.0 times and the conductivity 6.0 times" in text
    assert "a calibration of one parameter on one number" in text


def test_two_size_fits_that_disagree_are_read_as_a_range():
    from cmfqsar import transport as T
    rows = []
    # the three larger boxes follow a viscous liquid; the smallest box is far too slow
    for name, side, pairs, extra in (("scaled_small", 27.0, 14, 3.0e-11), ("scaled", 34.0, 27, 0.0),
                                     ("scaled_large", 43.0, 55, 0.0),
                                     ("scaled_huge", 54.0, 110, 0.0)):
        row = _sized(name, side, pairs, T.yeh_hummer_correction(side, 1.0e-2, 298.15) + extra)
        row["whole"]["conductivity"] = {"sigma": 0.3, "se": 0.04}
        row["whole"]["ionicity"] = 0.5
        rows.append(row)
    whole, rest = R.size_study(rows), R.size_study(rows, exclude=("scaled_small",))
    assert whole["fit"]["viscosity"] < rest["fit"]["viscosity"]
    text = "\n".join(R.bracket_lines(whole, rest, LIMITS))
    assert "Neither fit can be preferred" in text
    lo, hi = sorted([whole["ratio_with_factor"], rest["ratio_with_factor"]])
    assert f"is between {lo:.2f} and {hi:.2f} of the measured value" in text
    # a fit that determines nothing gives no range
    flat = dict(rest, determined=False)
    assert R.bracket_lines(whole, flat, LIMITS) == []


def _point(run, molarity, ratio, sigma=0.6, measured_d=2.7e-10):
    """A made-up result of one box of the concentration series."""
    def rec(d):
        return {"d": d, "se": 0.05 * d, "exponent": 0.85}
    measured = dict(MEASURED, diffusivity=measured_d)
    salt = ratio * measured_d
    return {"run": run, "label": "made up", "molarity": molarity, "pairs": 100, "atoms": 14000,
            "side": 54.0, "density": 1.19, "ns_per_day": 110.0, "temperature": 298.15,
            "measured": measured, "second_measured": dict(SECOND),
            "salt_ideal": salt / 2.4, "salt_with_factor": salt,
            "ratio_ideal": ratio / 2.4, "ratio_with_factor": ratio,
            "whole": {"length_ps": 4000.0, "cation": rec(1.0e-10), "anion": rec(1.4e-10),
                      "solvents": {"EC": rec(3.0e-10), "EMC": rec(2.5e-10)},
                      "conductivity": {"sigma": sigma, "se": 0.1}, "nernst_einstein": 1.0,
                      "ionicity": 0.6, "transport_number": 0.42}}


def test_the_series_has_boxes_of_its_own():
    # the boxes of the first part keep their directories
    assert R.run_dir("scaled", "cache").endswith("LiPF6_EC30_EMC70_m1p110_scaled")
    assert R.run_dir("conc_c3p0", "cache").endswith("LiPF6_EC30_EMC70_m3p320_conc_c3p0")
    assert R.molality_of("conc_c0p5_seed2") == 0.47 and R.molality_of("full") == R.MOLALITY
    one, two = R.settings_of("conc_c2p0"), R.settings_of("conc_c2p0_seed2")
    assert one.ion_charge_scale == two.ion_charge_scale == 0.7
    assert one.n_solvent == two.n_solvent == 1000
    assert (one.seed, two.seed) == (1, 2)
    assert len(R.CONCENTRATION_RUNS) == 6
    assert all(n in R.RUNS for names in R.CONCENTRATION_RUNS.values() for n in names)


def test_the_series_is_read_against_the_measured_value_at_each_concentration():
    rows = [_point("conc_c1p1", 1.13, 0.90, sigma=0.5), _point("conc_c1p1_seed2", 1.13, 1.02,
                                                               sigma=0.7),
            _point("conc_c0p5", 0.49, 1.30, sigma=0.9, measured_d=4.0e-10)]
    assert R.concentration_study([_point("scaled", 1.1, 1.0)]) is None
    study = R.concentration_study(rows)
    low, mid = study["points"]                         # in the order of the concentration
    assert (low["key"], mid["key"]) == ("c0p5", "c1p1")
    assert mid["starts"] == 2 and low["starts"] == 1
    # two starts: the mean, and half their difference as its error
    assert mid["ratio_with_factor"] == pytest.approx([0.96, 0.06])
    assert mid["conductivity"] == pytest.approx([0.6, 0.1])
    assert mid["inside"] is True and mid["inside_with_error"] is True
    # one start has no error of its ratio, so it cannot be said to reach the band
    assert np.isnan(low["ratio_with_factor"][1])
    assert low["inside"] is False and low["inside_with_error"] is None
    assert low["conductivity"][1] == 0.1               # the block error of the one run
    assert study["passes"] is False and study["decided"] is False
    assert study["inside"] == ["c1p1"]
    assert study["largest_conductivity"]["simulated"] == pytest.approx(0.49)
    text = "\n".join(R.concentration_lines(study, LIMITS, [0.08, 0.35]))
    assert "at 1 of the 2 concentrations (1.13 mol/L)" in text
    assert "it runs from 0.96 to 1.30. The gate is not met. With a band" in text
    assert "not every concentration was run" in text
    assert "| 1.13 | 100 | 2 | 1.00e-10 +/- 0.00e+00 |" in text
    assert "between +8 % and +35 % for this size of box" in text
    assert "| 1.13 | 0.96 | 1.04 | 1.30 |" in text
    assert "| 0.49 | 4.00e-10 | 1.80e-10 | 0.45 | yes | 1.30 |" in text
    # without the size runs the correction is left out and nothing is invented
    assert "this size of box" not in "\n".join(R.concentration_lines(study, LIMITS))


def test_a_complete_series_decides_the_gate():
    rows = []
    for j, names in enumerate(R.CONCENTRATION_RUNS.values()):
        rows += [_point(names[0], 0.1 + 0.5 * j, 0.95), _point(names[1], 0.1 + 0.5 * j, 1.05)]
    study = R.concentration_study(rows)
    assert study["complete"] and study["decided"] and study["passes"]
    text = "\n".join(R.concentration_lines(study, LIMITS))
    assert "at 6 of the 6 concentrations." in text
    assert "The gate is met at every concentration that was run." in text
    assert "The series is not complete" not in text
    # one start missing: the gate is read and marked as not decided
    part = R.concentration_study(rows[:-1])
    assert part["complete"] and not part["decided"]
    assert "not every concentration has its second start" in "\n".join(
        R.concentration_lines(part, LIMITS))
    # a ratio outside the band whose error reaches it is named as such
    wide = R.concentration_study([_point("conc_c1p1", 1.1, 0.80), _point("conc_c1p1_seed2", 1.1,
                                                                         0.92)])
    assert wide["points"][0]["inside"] is False
    assert wide["points"][0]["inside_with_error"] is True
    assert "no, but its error reaches it" in "\n".join(R.concentration_lines(wide, LIMITS))


def test_the_series_stays_out_of_the_other_tables(tmp_path, fixed_reference):
    stored_run(tmp_path / "cache", "full")
    stored_run(tmp_path / "cache", "conc_c0p5", seed=5)
    results = str(tmp_path / "results")
    R.stage_analyse(["full", "conc_c0p5"], results, str(tmp_path / "cache"))
    rows = R.read_results(list(R.RUNS), results)
    assert [r["run"] for r in rows] == ["full", "conc_c0p5"]
    assert rows[1]["second_measured"] == SECOND
    text = "\n".join(R.report_lines(rows, LIMITS))
    head, tail = text.split("## The boxes of the concentration series")
    assert "conc_c0p5" not in head and "| conc_c0p5 | 0.47 mol/kg" in tail
    assert "## The range of concentrations" in tail
    assert "A second measurement of the same electrolyte" in tail
    # the series alone still makes a table
    alone = "\n".join(R.report_lines(rows[1:], LIMITS))
    assert alone.startswith("# Item BT9") and "## The range of concentrations" in alone


def test_the_size_runs_give_the_range_of_the_correction():
    from cmfqsar import transport as T
    rows = []
    for name, side, pairs, extra in (("scaled_small", 27.0, 14, 3.0e-11), ("scaled", 34.0, 27, 0.0),
                                     ("scaled_large", 43.0, 55, 0.0),
                                     ("scaled_huge", 54.0, 110, 0.0)):
        row = _sized(name, side, pairs, T.yeh_hummer_correction(side, 1.0e-2, 298.15) + extra)
        row["whole"]["conductivity"] = {"sigma": 0.3, "se": 0.04}
        row["whole"]["ionicity"] = 0.5
        rows.append(row)
    lo, hi = R.size_range(rows)
    whole, rest = R.size_study(rows), R.size_study(rows, exclude=("scaled_small",))
    largest = whole["largest"]["ratio_with_factor"]
    assert sorted([lo, hi]) == [lo, hi]
    assert {round(lo, 6), round(hi, 6)} == {round(whole["ratio_with_factor"] / largest - 1, 6),
                                             round(rest["ratio_with_factor"] / largest - 1, 6)}
    assert R.size_range(rows[:1]) is None              # one box determines nothing


def test_the_default_analysis_leaves_out_what_was_not_simulated(tmp_path, fixed_reference,
                                                               capsys):
    stored_run(tmp_path / "cache", "full")
    results = str(tmp_path / "results")
    assert R.main(["--cache", str(tmp_path / "cache"), "--results", results]) == 0
    said = capsys.readouterr().out
    assert "scaled: not simulated yet, left out" in said
    assert os.listdir(results) == ["run_full.json", "table.md"] or \
        sorted(os.listdir(results)) == ["run_full.json", "table.md"]
    # a run that was asked for by name and is missing is still an error
    with pytest.raises(SystemExit, match="--stage simulate --runs scaled"):
        R.main(["--cache", str(tmp_path / "cache"), "--results", results, "--runs", "scaled"])


def test_a_long_run_says_how_much_a_short_one_overstates():
    assert R.settings_of("conc_c3p0_long").production_ps == 20000.0
    assert R.settings_of("conc_c3p0_long").sample_ps == 2.0
    assert R.run_dir("conc_c3p0_long", "c").endswith("m3p320_conc_c3p0_long")
    short = [_point("conc_c3p0", 2.97, 1.70), _point("conc_c3p0_seed2", 2.97, 1.90)]
    long = _point("conc_c3p0_long", 2.97, 1.20)
    for r in short:
        r["whole"]["cation"]["window_ps"] = [100.0, 400.0]
    long["whole"].update(length_ps=20000.0)
    long["whole"]["cation"] = {"d": 0.8e-10, "se": 2.0e-12, "exponent": 0.95,
                               "window_ps": [500.0, 2000.0]}
    assert R.long_study(short) is None and R.long_study([long]) is None
    (entry,) = R.long_study(short + [long])
    assert entry["short_runs"] == ["conc_c3p0", "conc_c3p0_seed2"]
    assert entry["overstated"] == pytest.approx(1.80 / 1.20)
    assert entry["species"]["cation"]["short_over_long"] == pytest.approx(1.25)
    text = "\n".join(R.long_lines([entry]))
    assert "In a run of 4 ns that window is 100 to 400 ps" in text
    assert "in a run of 20 ns it is 500 to 2000 ps" in text
    assert "| 2.97 | Cation | 1.00e-10 | 0.85 | 8.00e-11 +/- 2.0e-12 | 0.95 | 1.25 |" in text
    assert "| 2.97 | 1.80 | 1.20 | 1.50 |" in text
    # in the whole table the long run is listed with the series and not with the first part
    whole = "\n".join(R.report_lines(short + [long], LIMITS))
    assert "## How much a short run overstates" in whole


def test_the_ratio_of_the_series_becomes_a_function_for_the_cell():
    pybamm = pytest.importorskip("pybamm")
    f = R.ratio_function(lambda c, T: pybamm.Scalar(2.0e-10) * (T / T), [0.5, 1.0, 2.0],
                         [0.8, 1.0, 1.4])

    def at(c):
        return float(np.asarray(f(pybamm.Scalar(c), pybamm.Scalar(298.15)).evaluate()).ravel()[0])
    assert at(1000.0) == pytest.approx(2.0e-10)
    assert at(750.0) == pytest.approx(0.9 * 2.0e-10)          # linear between two points
    assert at(1500.0) == pytest.approx(1.2 * 2.0e-10)
    assert at(100.0) == pytest.approx(0.8 * 2.0e-10)          # constant below the lowest
    assert at(3000.0) == pytest.approx(1.4 * 2.0e-10)         # and above the highest
    with pytest.raises(ValueError, match="rising"):
        R.ratio_function(None, [1.0, 0.5], [1.0, 1.0])


def test_the_cell_is_run_with_the_series_as_simulated_and_with_the_size_range():
    study = R.concentration_study([_point("conc_c0p5", 0.5, 0.8), _point("conc_c1p1", 1.1, 1.2)])
    assert R.cell_versions(study, None) == {"as simulated": [0.8, 1.2]}
    three = R.cell_versions(study, [0.1, 0.5])
    assert list(three) == ["as simulated", "with the smaller correction for the box",
                           "with the larger correction for the box"]
    assert three["with the larger correction for the box"] == pytest.approx([1.2, 1.8])
    cell = {"share": 0.05, "versions": three, "rates": [
        {"rate": 1.0, "reference": 4.9, "reference_concentration": [700.0, 1400.0],
         "runs": {n: {"usable": True, "change": c, "ended": "", "inside": abs(c) <= 0.05}
                  for n, c in zip(three, (-0.004, 0.001, 0.002))}},
        {"rate": 2.5, "reference": 3.1, "reference_concentration": [20.0, 2600.0],
         "runs": {n: {"usable": u, "change": c, "ended": "event: Maximum voltage [V]",
                      "inside": u and abs(c) <= 0.05}
                  for n, c, u in zip(three, (-0.12, 0.03, float("nan")), (True, True, False))}}]}
    text = "\n".join(R.cell_lines(cell))
    assert "| 1C | 4.900 | 0.70 to 1.40 | -0.4 % | +0.1 % | +0.2 % |" in text
    assert "| 2.5C | 3.100 | 0.02 to 2.60 | -12.0 % | +3.0 % | no usable discharge " in text
    assert "as simulated, at 1C; with the smaller correction for the box, at 1C, 2.5C" in text


def test_two_sizes_at_the_factor_of_the_series_give_a_viscosity():
    from cmfqsar import transport as T
    eta = 3.0e-3
    step = (T.yeh_hummer_correction(34.0, eta, 298.15)
            - T.yeh_hummer_correction(54.0, eta, 298.15))
    small = _point("scaled_07", 1.11, 1.0)
    small.update(side=34.0, pairs=27)
    whole = small["whole"]
    for rec in [whole["cation"], whole["anion"], *whole["solvents"].values()]:
        rec["d"] -= step                               # the smaller box is the slower one
    large = [_point("conc_c1p1", 1.13, 1.1), _point("conc_c1p1_seed2", 1.13, 1.2)]
    assert R.size_pair(large) is None and R.size_pair([small]) is None
    pair = R.size_pair([small] + large)
    assert pair["determined"] and pair["large"] == ["conc_c1p1", "conc_c1p1_seed2"]
    assert pair["difference"] == pytest.approx(step)
    assert pair["viscosity"] == pytest.approx(eta)
    left = T.yeh_hummer_correction(54.0, eta, 298.15)
    assert pair["remaining"] == pytest.approx(left)
    assert pair["share"]["cation"] == pytest.approx(left / 1.0e-10)
    assert pair["share"]["anion"] == pytest.approx(left / 1.4e-10)
    assert pair["share"]["cation"] > pair["share"]["salt"] > pair["share"]["anion"]
    text = "\n".join(R.pair_lines(pair, [0.05, 0.60]))
    assert "one run of a box with 27 ion pairs (`scaled_07`, a side of 34.0 A)" in text
    assert "a viscosity of 3.0 mPa s" in text and "this estimate is inside it" in text
    assert "this estimate is outside it" in "\n".join(R.pair_lines(pair, [0.01, 0.02]))
    # a larger box that is not faster gives no correction
    for rec in [whole["cation"], whole["anion"], *whole["solvents"].values()]:
        rec["d"] += 2.0 * step
    slow = R.size_pair([small] + large)
    assert slow["determined"] is False and "viscosity" not in slow
    assert "give no correction" in "\n".join(R.pair_lines(slow))


def test_the_gate_is_also_read_with_a_band_twice_as_wide():
    rows = [_point("conc_c0p5", 0.49, 0.75), _point("conc_c1p1", 1.13, 1.14),
            _point("conc_c2p0", 1.98, 1.24)]
    text = "\n".join(R.concentration_lines(R.concentration_study(rows), LIMITS))
    assert ("The gate is not met. With a band twice as wide, 20 %, the ratio is inside at 1 of "
            "the 3 concentrations.") in text


def test_the_series_is_corrected_for_the_length_of_the_run():
    rows = []
    for run, molarity, ratio, exponent in (("conc_c0p5", 0.49, 0.75, 0.95),
                                           ("conc_c1p1", 1.13, 1.20, 0.85),
                                           ("conc_c3p0", 2.97, 1.60, 0.55)):
        row = _point(run, molarity, ratio)
        row["whole"]["cation"]["exponent"] = exponent
        rows.append(row)
    study = R.concentration_study(rows)
    lengths = [{"key": "c1p1", "overstated": 1.2,
                "species": {"cation": {"short_exponent": 0.85, "exponent": 0.97}}},
               {"key": "c3p0", "overstated": 1.6,
                "species": {"cation": {"short_exponent": 0.55, "exponent": 0.70}}}]
    factors = R.length_factors(study, lengths)
    low, mid, high = factors["points"]
    # between a measured point and the point (exponent 1, factor 1)
    assert low["factor"] == pytest.approx(1.0 + 0.2 * 0.05 / 0.15) and not low["measured"]
    assert mid["factor"] == pytest.approx(1.2) and mid["measured"]
    assert high["corrected"] == pytest.approx(1.0) and high["inside"]
    assert low["inside"] is False
    text = "\n".join(R.length_lines(factors))
    assert "| 2.97 | 0.55 | 1.60 | the long run (its own exponent: 0.70) | 1.60 | 1.00 | yes |" \
        in text
    assert "| 0.49 | 0.95 | 1.07 | estimated from the exponent | 0.75 | 0.70 | no |" in text
    assert "at 2 of the 3 concentrations (1.13 mol/L, 2.97 mol/L)" in text
    assert "With both corrections" not in text
    # the correction for the size of the box acts in the opposite direction
    both = "\n".join(R.length_lines(factors, [0.05, 0.50]))
    assert "| 1.13 | 1.00 | 1.05 | 1.50 |" in both
    assert ("at 2 of the 3 concentrations with the smaller correction for the box, and at 1 "
            "with the larger one") in both
    # a long run that found an exponent of 1 in the short runs gives no factor
    assert R.length_factors(study, [{"key": "c1p1", "overstated": 1.0, "species": {
        "cation": {"short_exponent": 1.0, "exponent": 1.0}}}]) is None
    # the cell is then also run with the corrected ratios
    six = R.cell_versions(study, [0.1, 0.5], factors)
    assert len(six) == 6
    assert six["corrected for the length of the run"] == pytest.approx(
        [0.75 / low["factor"], 1.0, 1.0])
    assert six["corrected for the length, with the larger correction for the box"] == \
        pytest.approx([1.5 * 0.75 / low["factor"], 1.5, 1.5])
    assert list(R.cell_versions(study, None, factors)) == [
        "as simulated", "corrected for the length of the run"]


def test_the_report_needs_a_result(tmp_path):
    with pytest.raises(SystemExit, match="--stage analyse"):
        R.report(str(tmp_path))
