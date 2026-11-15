"""Tests of cmfqsar.hsic: the statistic, the stratified permutation test and its calibration."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.hsic import _permuted_order, centre_within, hsic, hsic_test, strata_of  # noqa: E402


def _gauss(x, gamma=0.5):
    x = np.asarray(x, float).reshape(len(x), -1)
    d2 = ((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


def test_statistic_matches_the_formula():
    rng = np.random.RandomState(0)
    x, y = rng.randn(20, 2), rng.randn(20)
    K, L = _gauss(x), np.outer(y, y)
    H = np.eye(20) - 1.0 / 20
    assert hsic(K, L) == pytest.approx(np.trace(H @ K @ H @ L) / 400)
    s = np.repeat([0, 1, 2, 3], 5)
    P = np.zeros((20, 20))
    for g in range(4):
        P[np.ix_(s == g, s == g)] = 1 / 5
    HS = np.eye(20) - P
    assert np.allclose(centre_within(K, s), HS @ K @ HS)
    assert hsic(K, L, s) == pytest.approx(np.trace(HS @ K @ HS @ HS @ L @ HS) / 400)


def test_linear_test_statistic_and_alignment():
    rng = np.random.RandomState(1)
    y = rng.randn(30)
    r = hsic_test(np.outer(y, y), y, permutations=99)
    assert r.alignment == pytest.approx(1.0)
    assert r.statistic == pytest.approx(hsic(np.outer(y, y), np.outer(y, y)))
    assert r.p_value == pytest.approx(0.01)


def test_permutations_stay_inside_the_strata():
    rng = np.random.RandomState(2)
    s = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3, 3])
    for _ in range(20):
        o = _permuted_order(s, rng)
        assert np.array_equal(s[o], s)
        assert sorted(o) == list(range(10))


@pytest.mark.parametrize("kernel_y", ["linear", "gauss"])
def test_calibrated_under_independence(kernel_y):
    rng = np.random.RandomState(3)
    rejected = 0
    trials = 200
    for t in range(trials):
        x, y = rng.randn(25, 2), rng.randn(25)
        r = hsic_test(_gauss(x), y, kernel_y=kernel_y, permutations=99, seed=t)
        rejected += r.p_value <= 0.05
    # binomial(200, 0.05): mean 10, 99.9 % below 22
    assert rejected <= 22


def test_detects_dependence():
    rng = np.random.RandomState(4)
    x = rng.randn(40)
    y = np.sin(2 * x) + 0.3 * rng.randn(40)
    for kernel_y in ("linear", "gauss"):
        assert hsic_test(_gauss(x), y, kernel_y=kernel_y, permutations=199).p_value <= 0.01


def test_conditional_test_is_calibrated_where_the_plain_one_is_not():
    """y depends on z only, the field sees z: the plain test rejects, the stratified one does not."""
    rng = np.random.RandomState(5)
    plain = cond = 0
    trials = 150
    for t in range(trials):
        z = rng.randint(0, 5, 40)
        x = np.c_[z + 0.3 * rng.randn(40), rng.randn(40)]      # the field sees z, and more
        y = (z - 2.0) ** 2 + rng.randn(40)
        K = _gauss(x, gamma=0.2)
        s, exact = strata_of(z)
        assert exact
        plain += hsic_test(K, y, permutations=99, seed=t).p_value <= 0.05
        cond += hsic_test(K, y, strata=s, permutations=99, seed=t).p_value <= 0.05
    assert plain >= 0.8 * trials
    assert cond <= 19          # binomial(150, 0.05): mean 7.5, 99.9 % below 19


def test_conditional_test_finds_information_beyond_the_reference():
    rng = np.random.RandomState(6)
    z = rng.randint(0, 4, 60)
    w = rng.randn(60)                                          # what the field adds to z
    x = np.c_[z, w]
    y = z + 1.5 * w + 0.5 * rng.randn(60)
    s, _ = strata_of(z)
    r = hsic_test(_gauss(x, 0.3), y, strata=s, permutations=199)
    assert r.p_value <= 0.01 and r.strata == 4 and r.informative == 60


def test_strata_of():
    s, exact = strata_of([3, 1, 3, 2, 1])
    assert exact and list(s) == [2, 0, 2, 1, 0]
    s, exact = strata_of(np.linspace(0, 1, 100), bins=4)
    assert not exact and np.bincount(s).tolist() == [25, 25, 25, 25]
    s, exact = strata_of(np.array([[1, 0], [1, 0], [0, 2]]))
    assert exact and s[0] == s[1] != s[2]
    with pytest.raises(ValueError, match="bins"):
        strata_of(np.arange(50), max_levels=10)


def test_singletons_carry_no_information():
    rng = np.random.RandomState(7)
    y = rng.randn(12)
    s = np.arange(12)                                          # every molecule alone
    r = hsic_test(_gauss(rng.randn(12)), y, strata=s, permutations=49)
    assert r.informative == 0 and r.statistic == pytest.approx(0.0, abs=1e-12)
    assert r.p_value == 1.0


def test_input_checks():
    with pytest.raises(ValueError, match="missing"):
        hsic_test(np.eye(3), [1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="kernel_y"):
        hsic_test(np.eye(3), [1.0, 0.0, 2.0], kernel_y="cubic")


# --------------------------------------------------------------------------- the command line

GPB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "examples", "sutherland", "GPB.sdf")


def test_dependence_command(capsys):
    from cmfqsar.cli import main

    main(["dependence", GPB, "--charges", "file", "--permutations", "99"])
    text = capsys.readouterr().out
    assert "depend on the fields at all" in text and "beyond" not in text
    main(["dependence", GPB, "--charges", "file", "--permutations", "99",
          "--given", "calc:HeavyAtomCount"])
    text = capsys.readouterr().out
    assert "beyond calc:HeavyAtomCount" in text and "(exact)" in text
    main(["dependence", GPB, "--charges", "file", "--permutations", "99",
          "--given", "calc:clogp", "--bins", "4"])
    text = capsys.readouterr().out
    assert "within 4 strata" in text and "approximate" in text and "cut into 4 bins" in text
    assert "straight line" in text


def test_binned_reference_leaks_and_the_within_bin_line_repairs_it():
    """A continuous reference, a curved activity of it, a field that sees it: binning alone
    rejects too often; removing a line in z within every bin does not."""
    rng = np.random.RandomState(8)
    binned = partial = 0
    trials = 150
    for t in range(trials):
        z = rng.randn(60)
        x = np.c_[z + 0.2 * rng.randn(60), rng.randn(60)]
        y = 2.0 * (z + 0.5 * z * z) + rng.randn(60)
        K = _gauss(x, gamma=0.3)
        s, exact = strata_of(z, bins=4)
        assert not exact
        binned += hsic_test(K, y, strata=s, exact=exact, permutations=99, seed=t).p_value <= 0.05
        r = hsic_test(K, y, strata=s, exact=exact, permutations=99, seed=t, covariate=z)
        assert r.partial and not r.exact
        partial += r.p_value <= 0.05
    assert binned >= 3 * 0.05 * trials
    assert partial <= 19        # binomial(150, 0.05): 99.9 % below 19


def test_within_bin_line_keeps_power():
    rng = np.random.RandomState(9)
    z, w = rng.randn(80), rng.randn(80)
    y = z + 0.5 * z * z + w + 0.5 * rng.randn(80)
    s, _ = strata_of(z, bins=5)
    r = hsic_test(_gauss(np.c_[z, w], 0.3), y, strata=s, exact=False, covariate=z,
                  permutations=199)
    assert r.p_value <= 0.01


def test_a_covariate_constant_in_its_strata_changes_nothing():
    rng = np.random.RandomState(10)
    z = rng.randint(0, 5, 50)
    x, y = rng.randn(50, 2), rng.randn(50) + z
    s, _ = strata_of(z)
    a = hsic_test(_gauss(x), y, strata=s, permutations=99, seed=3)
    b = hsic_test(_gauss(x), y, strata=s, permutations=99, seed=3, covariate=z.astype(float))
    assert a.p_value == b.p_value and a.statistic == pytest.approx(b.statistic)
    with pytest.raises(ValueError, match="strata"):
        hsic_test(_gauss(x), y, covariate=z.astype(float))
