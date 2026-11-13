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
            "measured": dict(MEASURED),
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


def test_the_report_needs_a_result(tmp_path):
    with pytest.raises(SystemExit, match="--stage analyse"):
        R.report(str(tmp_path))
