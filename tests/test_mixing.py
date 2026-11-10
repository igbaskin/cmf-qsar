"""A posterior over the field weights (plan item E6).

The tests are on synthetic kernels, because what has to be right is the inference: a kernel that
explains the activities should get the weight, one that does not should be reduced to zero, and the
interval should say which case it is.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.mixing import (PRIOR_SD, _log_posterior, _slice_sample, _softmax, _start,  # noqa: E402
                            drop_one, evidence_mixing, pool_priors, sample_mixing)
from cmfqsar.model import align_mixing  # noqa: E402


def _world(n=60, d=6, noise=0.2, seed=0):
    """Two kernels: the first explains the activities, the second is unrelated noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Z = rng.normal(size=(n, d))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    y = X @ rng.normal(size=d) + rng.normal(scale=noise, size=n)
    return np.stack([X @ X.T, Z @ Z.T]), y


def test_the_kernel_that_explains_the_activities_takes_the_weight():
    K, y = _world()
    h = evidence_mixing(K, y)
    assert h[0] > 3.0 * h[1]


def test_the_weights_keep_the_joint_kernel_the_size_it_was():
    """The lambda grid has to stay comparable, which is the convention align_mixing keeps."""
    K, y = _world()
    h = evidence_mixing(K, y)
    ref = sum(np.mean(np.diag(K[f])) for f in range(len(K)))
    cur = sum(h[f] * np.mean(np.diag(K[f])) for f in range(len(K)))
    assert cur == pytest.approx(ref, rel=1e-9)


def test_it_beats_equal_weights_at_the_thing_it_optimises():
    """Guards against a bug: Nelder-Mead builds its simplex from a starting point of all zeros."""
    K, y = _world()
    scale = np.ones(len(K))
    lam0 = _start(K, y, scale, True)
    h = evidence_mixing(K, y)
    u = np.log(h / h.sum())
    u = u - u.mean()                      # the representative the prior prefers
    grid = np.log([1e-3, 1e-2, 0.1, 0.3, 1.0, 3.0, 10.0])
    best = max(_log_posterior(u, ll, K, y, scale, lam0, PRIOR_SD, True) for ll in grid)
    flat = max(_log_posterior(np.zeros(len(K)), ll, K, y, scale, lam0, PRIOR_SD, True)
               for ll in grid)
    assert best > flat + 0.5


def test_a_field_that_is_switched_off_gets_no_weight():
    K, y = _world()
    h = evidence_mixing(K, y, enabled=[True, False])
    assert h[1] == 0.0 and h[0] > 0.0
    assert evidence_mixing(K, y, enabled=[False, False]).tolist() == [0.0, 0.0]


def test_the_alignment_entry_point_delegates():
    K, y = _world()
    assert np.allclose(align_mixing(K, y, mode="evidence"), evidence_mixing(K, y))
    with pytest.raises(ValueError, match="unknown alignment mode"):
        align_mixing(K, y, mode="hunch")


def test_the_posterior_is_a_distribution_over_shares():
    K, y = _world()
    post = sample_mixing(K, y, ["real", "noise"], draws=120, burn=40, seed=0)
    assert post.samples.shape == (120, 2)
    assert np.allclose(post.samples.sum(axis=1), 1.0)
    iv = post.interval(0.9)
    assert iv.shape == (2, 2)
    for f in range(2):
        assert iv[f, 0] <= post.mean[f] <= iv[f, 1]
    assert np.all(post.lam > 0)
    assert "field" in post.describe()


def test_the_posterior_says_which_field_is_not_needed():
    K, y = _world()
    post = sample_mixing(K, y, ["real", "noise"], draws=200, burn=60, seed=0)
    assert post.mean[0] > post.mean[1]
    assert "noise" in post.carries_less_than_its_share(level=0.9)
    assert "real" not in post.carries_less_than_its_share(level=0.9)


def test_the_same_seed_gives_the_same_chain():
    K, y = _world()
    a = sample_mixing(K, y, draws=40, burn=10, seed=3)
    b = sample_mixing(K, y, draws=40, burn=10, seed=3)
    assert np.array_equal(a.samples, b.samples)


def test_one_field_has_nothing_to_share():
    K, y = _world()
    post = sample_mixing(K[:1], y, ["only"], draws=10, burn=2, seed=0)
    assert post.samples.shape == (1, 1) and post.samples[0, 0] == 1.0
    assert "nothing to share" in post.note


def test_dropping_the_kernel_that_matters_costs_the_evidence_dearly():
    K, y = _world()
    loss = drop_one(K, y, ["real", "noise"])
    assert loss["real"] > 5.0
    assert loss["noise"] < 1.0
    assert loss["real"] > loss["noise"]


def test_the_slice_sampler_samples_what_it_is_given():
    """A standard normal, which it has no way of knowing about in advance."""
    rng = np.random.default_rng(0)
    logp = lambda x: -0.5 * float(np.sum(x ** 2))      # noqa: E731
    x = np.zeros(1)
    draws = []
    for _ in range(3000):
        x, _cur, _ev = _slice_sample(logp, x, width=1.0, rng=rng)
        draws.append(float(x[0]))
    d = np.asarray(draws)
    assert abs(d.mean()) < 0.1
    assert abs(d.std(ddof=1) - 1.0) < 0.1


def test_the_shares_are_a_softmax_of_the_log_shares():
    u = np.array([0.0, 1.0, -1.0])
    p = _softmax(u)
    assert p.sum() == pytest.approx(1.0)
    assert np.allclose(_softmax(u + 5.0), p)           # the level does not matter, the spread does


# --- the prior pooled across series ---------------------------------------------------------


def _shares_of(u):
    p = np.exp(u - u.mean(axis=1, keepdims=True))
    return p / p.sum(axis=1, keepdims=True)


def _population(mu, tau, n_series, draws, post_sd, seed):
    """Synthetic posteriors: the centre of every series' likelihood scatters around the
    population mean by tau and is measured with standard deviation post_sd; the draws are from
    the posterior that the fixed prior gives such a likelihood."""
    rng = np.random.default_rng(seed)
    F = len(mu)
    var = 1.0 / (1.0 / post_sd ** 2 + 1.0 / PRIOR_SD ** 2)
    out = []
    for _ in range(n_series):
        u = mu + rng.normal(scale=tau, size=F)
        c = u + rng.normal(scale=post_sd, size=F)      # what this series' data say
        m = var * c / post_sd ** 2
        out.append(_shares_of(m + rng.normal(scale=np.sqrt(var), size=(draws, F))))
    return out


def test_a_prior_mean_holds_the_weights_when_the_data_do_not_care():
    rng = np.random.default_rng(1)
    K, _ = _world()
    y = rng.normal(size=K.shape[1])            # unrelated to either kernel
    h = evidence_mixing(K, y, prior_mean=[1.0, -1.0], prior_sd=0.5)
    p = h / h.sum()
    want = np.exp([1.0, -1.0]) / np.exp([1.0, -1.0]).sum()
    assert abs(p[0] - want[0]) < 0.15


def test_the_pooled_prior_shrinks_towards_the_other_series_and_no_further():
    """The posterior mode is a balance: between the series' own optimum and what the other
    series say, and with a prior of ordinary width a clear answer survives."""
    K, y = _world()
    free = evidence_mixing(K, y)
    pooled = evidence_mixing(K, y, prior_mean=[-1.0, 1.0], prior_sd=1.0)
    prior_h = 2.0 * np.exp([-1.0, 1.0]) / np.exp([-1.0, 1.0]).sum()
    assert prior_h[0] < pooled[0] < free[0]
    assert pooled[0] > pooled[1]


@pytest.mark.parametrize("method", ["gaussian", "importance"])
def test_pool_priors_recovers_the_population(method):
    mu = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    sets = _population(mu, tau=0.4, n_series=40, draws=300, post_sd=0.5, seed=0)
    prior = pool_priors(sets, method=method)
    assert np.max(np.abs(prior.mean - mu)) < 0.25
    assert 0.2 < prior.sd < 0.7
    assert prior.shares.argmax() == 0 and prior.shares.argmin() == 4


def test_pool_priors_says_when_the_series_know_nothing():
    rng = np.random.default_rng(2)
    sets = [_shares_of(rng.normal(scale=PRIOR_SD, size=(1000, 5))) for _ in range(8)]
    prior = pool_priors(sets)
    assert "no series carries information" in prior.note
    assert prior.sd == PRIOR_SD


def test_the_two_pooling_estimators_agree_where_both_are_valid():
    mu = np.array([0.8, 0.0, -0.8])
    sets = _population(mu, tau=0.5, n_series=30, draws=400, post_sd=0.3, seed=3)
    a = pool_priors(sets, method="gaussian")
    b = pool_priors(sets, method="importance")
    assert np.max(np.abs(a.mean - b.mean)) < 0.2
    assert abs(a.sd - b.sd) < 0.2


# --------------------------------------------------------------------------- #
# The ridge at which an evidence is taken
# --------------------------------------------------------------------------- #
def _exact_world(n=80, seed=0):
    """A kernel that explains the activities almost without noise, so that the best ridge is
    far below 0.001, and a kernel of noise."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.001 * rng.normal(size=n)
    Z = rng.normal(size=(n, 30))
    return np.stack([X @ X.T / 3.0, Z @ Z.T / 30.0]), y


def test_the_best_ridge_does_not_depend_on_the_scale_of_the_kernel():
    from cmfqsar.evidence import log_evidence
    from cmfqsar.mixing import LAM_GRID, best_ridge_evidence

    K, y = _exact_world()
    value, ridge = best_ridge_evidence(K[0], y)
    scaled, scaled_ridge = best_ridge_evidence(1000.0 * K[0], y)
    assert scaled == pytest.approx(value, abs=1e-5)
    assert scaled_ridge == pytest.approx(1000.0 * ridge, rel=1e-2)
    # the best ridge is below the fixed list, and no ridge of the list reaches the value
    assert ridge < min(LAM_GRID)
    assert value > max(log_evidence(K[0], y, lam) for lam in LAM_GRID) + 10.0
    # it is a maximum: the neighbours on both sides are lower
    for factor in (0.3, 3.0):
        assert log_evidence(K[0], y, factor * ridge) < value
    # a kernel that is not positive has no evidence
    assert best_ridge_evidence(-K[0], y)[0] == float("-inf")


def test_a_kernel_of_noise_is_not_credited_for_the_scale_of_the_sum():
    """With five fixed ridges the sum of two kernels, which has twice the scale, was nearer
    to the best ridge than the one kernel, and the kernel of noise got a factor of +25."""
    from cmfqsar.evidence import log_evidence
    from cmfqsar.mixing import LAM_GRID

    K, y = _exact_world()

    def on_the_fixed_ridges(on):
        h = evidence_mixing(K, y, None, on)
        joint = np.tensordot(h, K, axes=(0, 0))
        return max(log_evidence(joint, y, lam) for lam in LAM_GRID)

    before = on_the_fixed_ridges([True, True]) - on_the_fixed_ridges([True, False])
    assert before > 10.0                                   # the fault
    loss = drop_one(K, y, ["real", "noise"])
    assert loss["noise"] < 1.0                             # the repair
    assert loss["real"] > 100.0
    # a common factor on every kernel moves the numbers little
    scaled = drop_one(37.0 * K, y, ["real", "noise"])
    assert scaled["noise"] == pytest.approx(loss["noise"], abs=2.0)
    assert scaled["real"] == pytest.approx(loss["real"], rel=0.01)
