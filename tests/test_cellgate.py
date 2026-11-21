"""Tests for benchmarks/run_cellgate.py: gate (c) of plan item BT9.

The reading of the gate is the part that must not move once the cell has been run, so it is tested
on made-up capacities: an interval holds the truth only if the truth is between its ends, a
capacity that is not a number holds nothing, and usefulness is a width against the published
capacity.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import run_cellgate as CG                                                    # noqa: E402


def test_an_interval_holds_the_truth_only_between_its_ends():
    r = CG.read_interval(4.80, 4.85, 4.90, truth=4.86, published=4.90)
    assert r["holds"] and r["low"] == 4.80 and r["high"] == 4.90
    assert not CG.read_interval(4.80, 4.85, 4.90, truth=4.91, published=4.90)["holds"]
    assert not CG.read_interval(4.80, 4.85, 4.90, truth=4.79, published=4.90)["holds"]


def test_the_ends_may_come_in_either_order():
    r = CG.read_interval(4.90, 4.85, 4.80, truth=4.86, published=4.90)
    assert r["holds"] and r["low"] == 4.80


def test_usefulness_is_a_width_against_five_percent_of_the_published_capacity():
    narrow = CG.read_interval(4.90, 4.91, 4.92, truth=4.91, published=5.0)
    assert narrow["useful"] and narrow["width_share"] == pytest.approx(0.004)
    wide = CG.read_interval(4.5, 4.7, 4.9, truth=4.7, published=5.0)
    assert not wide["useful"] and wide["holds"]


def test_a_capacity_that_is_not_a_number_holds_nothing():
    r = CG.read_interval(float("nan"), 4.8, 4.9, truth=4.85, published=4.9)
    assert not r["holds"] and not r["useful"]
    assert not CG.read_interval(4.8, 4.85, 4.9, truth=float("nan"), published=4.9)["holds"]


def test_the_median_distance_is_a_share_of_the_published_capacity():
    r = CG.read_interval(4.0, 4.5, 5.0, truth=4.4, published=5.0)
    assert r["median_minus_truth_share"] == pytest.approx(0.02)


def test_the_molality_is_read_between_the_boxes_of_the_series():
    series = [{"molarity": 0.1, "realised_molality": 0.09},
              {"molarity": 1.0, "realised_molality": 1.0},
              {"molarity": 3.0, "realised_molality": 3.3}]
    assert CG.molality_at(1.0, series) == pytest.approx(1.0)
    assert CG.molality_at(2.0, series) == pytest.approx(2.15)
    assert CG.molality_at(5.0, series) == pytest.approx(3.3)


def test_the_label_deviation_is_the_spread_of_the_logarithm_of_the_ratio():
    names = [n for v in CG.RD.CONCENTRATION_RUNS.values() for n in v]
    ratios = [0.35, 0.35, 0.75, 0.75, 1.14, 1.14, 1.15, 1.15, 1.24, 1.24, 1.61, 1.61]
    series = [{"run": n, "ratio_with_factor": r} for n, r in zip(names, ratios)]
    want = float(np.std(np.log([0.35, 0.75, 1.14, 1.15, 1.24, 1.61]), ddof=1))
    assert CG.label_deviation(series) == pytest.approx(want)


def test_the_scaled_function_multiplies_by_the_interpolated_ratio():
    pybamm = pytest.importorskip("pybamm")
    f = CG._scaled(lambda c, T: 2.0, np.array([100.0, 1000.0]), np.array([1.0, 3.0]))
    v = f(pybamm.Scalar(550.0), pybamm.Scalar(298.15)).evaluate()
    assert float(np.ravel(v)[0]) == pytest.approx(2.0 * 2.0)
    far = f(pybamm.Scalar(5000.0), pybamm.Scalar(298.15)).evaluate()
    assert float(np.ravel(far)[0]) == pytest.approx(6.0)


def test_the_test_mixture_is_the_cells_electrolyte():
    assert CG.TEST_SOLVENTS == {"EC": 0.3, "EMC": 0.7}
    assert CG.Z_POINTS == (-1.96, 0.0, 1.96)
