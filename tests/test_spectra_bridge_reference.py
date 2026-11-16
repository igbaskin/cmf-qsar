"""The references a spectroscopist would use first (plan item SP3, tutorial 30): binned
spectra, the peak-table search, the choice of a setting on the calibration acids, the paired
difference, and the components of each view alone."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import run_spectra_bridge as BR  # noqa: E402
from cmfqsar import spectrum as SP  # noqa: E402


def sticks(x, y):
    return SP.Spectrum(np.asarray(x, float), np.asarray(y, float), "sticks", "wavenumber")


def test_binned_sums_lines_of_one_bin_and_puts_an_edge_line_in_the_upper_bin():
    s = sticks([1003.0, 1007.0, 1010.0], [1.0, 4.0, 9.0])
    F = BR.binned([s], 10.0, 1.0, x_range=(1000.0, 1030.0))
    assert F.shape == (1, 3)
    assert np.allclose(F[0], [5.0, 9.0, 0.0])
    assert np.allclose(BR.binned([s], 10.0, 0.5, x_range=(1000.0, 1030.0))[0], [3.0, 3.0, 0.0])


def test_a_small_shift_across_a_bin_edge_loses_the_binned_match_but_not_the_kernel():
    a, b = sticks([1009.0], [1.0]), sticks([1011.0], [1.0])
    F = BR.binned([a, b], 10.0, x_range=(1000.0, 1100.0))
    assert float(BR.unit_rows(F)[0] @ BR.unit_rows(F)[1]) == 0.0
    grid = SP.Grid.for_width(1000.0, 1100.0, 10.0)
    G = BR.unit_rows(SP.feature_matrix([a, b], grid, 10.0, 1.0))
    assert float(G[0] @ G[1]) > 0.9


def test_peak_table_keeps_lines_above_the_share_of_the_strongest():
    s = sticks([500.0, 1000.0, 1700.0], [0.04, 0.5, 1.0])
    assert np.allclose(BR.peak_table(s, 0.05), [1000.0, 1700.0])
    assert np.allclose(BR.peak_table(s, 0.01), [500.0, 1000.0, 1700.0])


def test_peak_similarity_counts_partners_within_the_tolerance_on_both_sides():
    lib = [np.array([1000.0, 1700.0]), np.array([1300.0])]
    qry = [np.array([1005.0, 1700.0, 3000.0])]
    dist = BR.peak_distances(lib, qry)
    S = BR.peak_similarity(dist, 10.0)
    assert np.isclose(S[0, 0], (2 + 2) / (3 + 2))
    assert S[0, 1] == 0.0
    assert np.isclose(BR.peak_similarity(dist, 1.0)[0, 0], (1 + 1) / (3 + 2))


def test_choice_on_calibration_uses_the_calibration_rows_only():
    n = 6
    good, bad = np.eye(n), np.ones((n, n)) - np.eye(n)
    # a setting that is right on the calibration rows and wrong on the test rows
    trap = np.eye(n)
    trap[3:] = bad[3:]
    splits = [(np.arange(3), np.arange(3, 6))]
    rows = BR.choose_on_calibration({"bad": bad, "trap": trap}, splits, {}, None)
    assert rows[0]["setting"] == "trap"
    assert rows[0]["top1"] == 0.0
    rows = BR.choose_on_calibration({"bad": bad, "good": lambda cal: good}, splits, {}, None)
    assert rows[0]["setting"] == "good"
    assert rows[0]["top1"] == 1.0 and rows[0]["first"] == [1.0, 1.0, 1.0]


def test_paired_difference_is_the_mean_over_all_test_queries():
    table = {"substituent": ["a", "a", "b", "b", "c", "c"]}
    a = [{"first": [1.0, 1.0, 0.0], "test": [0, 1, 2]}, {"first": [1.0, 1.0], "test": [4, 5]}]
    b = [{"first": [0.0, 1.0, 0.0], "test": [0, 1, 2]}, {"first": [0.0, 0.0], "test": [4, 5]}]
    d = BR.paired_difference(a, b, table, repeats=200)
    assert np.isclose(d["mean"], 3 / 5)
    assert d["low"] <= d["mean"] <= d["high"]
    assert (d["ahead"], d["level"], d["behind"]) == (2, 0, 0)


def test_held_out_components_follow_a_planted_direction():
    rng = np.random.default_rng(0)
    t = rng.normal(size=60)
    X = np.outer(t, rng.normal(size=20)) * 3.0 + rng.normal(size=(60, 20)) * 0.3
    folds = [np.arange(k, 60, 5) for k in range(5)]
    P = BR.held_out_components(X, folds, k=2)
    assert abs(np.corrcoef(P[:, 0], t)[0, 1]) > 0.95


def test_kernel_components_of_a_linear_kernel_match_the_principal_components():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4)) * np.array([5.0, 2.0, 1.0, 0.5])
    V = BR.kernel_components(X @ X.T, k=2)
    Xc = X - X.mean(axis=0)
    U = np.linalg.svd(Xc, full_matrices=False)[0]
    assert abs(np.corrcoef(V[:, 0], U[:, 0])[0, 1]) > 0.999
    assert abs(np.corrcoef(V[:, 1], U[:, 1])[0, 1]) > 0.999
