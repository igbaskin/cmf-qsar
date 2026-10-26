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
                            drop_one, evidence_mixing, sample_mixing)
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
