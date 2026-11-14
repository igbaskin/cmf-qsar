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


def test_the_report_needs_a_result(tmp_path):
    with pytest.raises(SystemExit, match="--stage analyse"):
        R.report(str(tmp_path))
