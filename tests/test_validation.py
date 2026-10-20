"""The three checks a model has to pass besides cross-validation."""
import numpy as np
import pytest

from cmfqsar.validation import (BootstrapResult, ScrambleResult, bootstrap_ci, bootstrap_refit,
                                domain, domain_threshold, scramble_kernel, y_scramble)


def _toy(n=40, seed=0):
    """A kernel with real signal in it: points on a line, plus noise."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, 3)
    K = X @ X.T + 1e-6 * np.eye(n)
    y = 2.0 * X[:, 0] + 0.3 * rng.randn(n)
    return X, K, y


def test_scramble_separates_signal_from_noise():
    _, K, y = _toy()
    good = scramble_kernel(K, y, params={"lam": 0.1}, folds=5, runs=20)
    assert good.observed > 0.8
    assert good.mean < 0.2                      # shuffled activities cannot be learned
    assert good.p_value <= 1.0 / 21 + 1e-12     # none of the shuffles did as well
    assert good.z_score > 3

    noise = np.random.RandomState(1).randn(len(y))
    bad = scramble_kernel(K, noise, params={"lam": 0.1}, folds=5, runs=20)
    assert bad.p_value > 0.05, "a model of pure noise must not look significant"


def test_scramble_p_value_has_the_permutation_floor():
    _, K, y = _toy()
    res = scramble_kernel(K, y, params={"lam": 0.1}, folds=5, runs=9)
    assert res.p_value >= 1.0 / 10               # 9 shuffles cannot support a smaller p
    assert res.runs == 9
    assert len(res.scrambled) == 9


def test_scramble_takes_any_scoring_function():
    calls = []

    def score(y):
        calls.append(np.array(y, copy=True))
        return float(np.mean(y))

    y = np.arange(10.0)
    res = y_scramble(score, y, runs=4, seed=0, statistic="mean")
    assert len(calls) == 5                       # the real activities plus four shuffles
    assert np.allclose(calls[0], y)              # the real ones come first, unshuffled
    for perm in calls[1:]:
        assert sorted(perm) == sorted(y)         # a shuffle, not a resample
    assert res.statistic == "mean"


def test_domain_puts_the_training_set_inside_and_a_stranger_outside():
    X, K, _ = _toy(n=30)
    inside = domain(K[:5], np.diag(K)[:5], K)
    assert inside.inside.all()
    assert np.allclose(inside.similarity, 1.0, atol=1e-9)   # each is its own nearest neighbour
    assert np.allclose(inside.distance, 0.0, atol=1e-6)

    stranger = np.array([[0.0, 0.0, 1.0]]) * 1e-3 + np.array([[0.0, 1e-3, 0.0]])
    far = domain(stranger @ X.T, [float((stranger @ stranger.T)[0, 0])], K)
    assert far.distance[0] > inside.distance.max()


def test_domain_threshold_follows_the_training_set_spread():
    rng = np.random.RandomState(0)
    tight = rng.randn(30, 3) * 0.01 + np.array([1.0, 0.0, 0.0])
    loose = rng.randn(30, 3)
    t_tight = domain_threshold(tight @ tight.T)
    t_loose = domain_threshold(loose @ loose.T)
    assert t_tight > t_loose, "a series of close analogues deserves a tighter domain"


def test_bootstrap_interval_brackets_the_statistic():
    rng = np.random.RandomState(0)
    y = rng.randn(60)
    yp = y + 0.4 * rng.randn(60)
    res = bootstrap_ci(y, yp, runs=400, seed=0)
    assert res.lo < res.value < res.hi
    assert res.runs == 400
    wide = bootstrap_ci(y[:12], yp[:12], runs=400, seed=0)
    assert (wide.hi - wide.lo) > (res.hi - res.lo), "fewer molecules, wider interval"


def test_bootstrap_rmse_statistic():
    rng = np.random.RandomState(0)
    y = rng.randn(50)
    res = bootstrap_ci(y, y + 0.5, statistic="rmse", runs=200, seed=0)
    assert 0.4 < res.value < 0.6
    assert res.lo <= res.value <= res.hi


def test_bootstrap_refit_scores_out_of_bag():
    _, K, y = _toy(n=50)
    res = bootstrap_refit(K, y, params={"lam": 0.1}, runs=25, seed=0)
    assert res.refit and res.runs > 0
    assert res.lo <= res.value <= res.hi
    assert res.value > 0.5                       # the signal survives being resampled


def test_results_describe_themselves_without_a_run():
    assert "not run" in ScrambleResult().describe()
    assert "not run" in BootstrapResult().describe()


@pytest.mark.parametrize("statistic", ["q2", "rmse"])
def test_bootstrap_refuses_too_few_points(statistic):
    res = bootstrap_ci([1.0, 2.0], [1.1, 2.1], statistic=statistic, runs=10)
    assert res.runs == 0


def test_caution_does_not_cry_chance_when_there_were_too_few_shuffles():
    _, K, y = _toy()
    few = scramble_kernel(K, y, params={"lam": 0.1}, folds=5, runs=10)
    note = few.caution()
    assert note is not None and note.startswith("NOTE")       # not a warning: too few shuffles
    assert "19" in note

    enough = scramble_kernel(K, y, params={"lam": 0.1}, folds=5, runs=25)
    assert enough.caution() is None                           # a clean bill of health

    noise = np.random.RandomState(3).randn(len(y))
    bad = scramble_kernel(K, noise, params={"lam": 0.1}, folds=5, runs=25)
    assert bad.caution().startswith("WARNING")


def test_scramble_with_the_search_redone_raises_the_scrambled_baseline():
    """Letting the search pick the best of several kernels flatters noise, which is the point."""
    from cmfqsar.validation import scramble_selection
    rng = np.random.RandomState(0)
    X = rng.randn(40, 3)
    kernels = [X @ X.T + c * np.eye(40) for c in (0.01, 0.1, 1.0)]
    y = 2.0 * X[:, 0] + 0.3 * rng.randn(40)
    strict = scramble_selection(kernels, y, param_grid=[{"lam": 0.01}, {"lam": 0.1}, {"lam": 1.0}],
                                folds=5, runs=15, seed=0)
    fixed = scramble_kernel(kernels[1], y, params={"lam": 0.1}, folds=5, runs=15, seed=0)
    assert strict.reselected and not fixed.reselected
    assert "redone" in strict.describe()
    assert strict.mean >= fixed.mean - 1e-9, "a search over a grid cannot do worse on noise"
    assert strict.observed >= fixed.observed - 1e-9
