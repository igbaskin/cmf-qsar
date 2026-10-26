"""The marginal likelihood of a kernel ridge model, and what it is allowed to claim."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.evidence import evidence_grid, log_evidence, profile_amplitude  # noqa: E402


def _world(n=30, d=4, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] * 2.0 + rng.normal(scale=noise, size=n)
    return X, X @ X.T, y


def test_it_is_the_textbook_formula_when_nothing_is_marginalised():
    _X, K, y = _world()
    n, lam = len(y), 0.2
    A = K + lam * np.eye(n)
    want = (-0.5 * y @ np.linalg.solve(A, y) - 0.5 * np.linalg.slogdet(A)[1]
            - 0.5 * n * math.log(2 * math.pi))
    assert log_evidence(K, y, lam, fit_bias=False, profile=False) == pytest.approx(want)


def test_the_constant_mean_is_marginalised_not_fitted():
    """The Gaussian process matching a KRR with a bias has an unknown constant mean."""
    _X, K, y = _world()
    n, lam = len(y), 0.2
    A = K + lam * np.eye(n)
    Ai = np.linalg.inv(A)
    one = np.ones(n)
    denom = float(one @ Ai @ one)
    resid = y - (one @ Ai @ y) / denom
    want = (-0.5 * resid @ Ai @ resid - 0.5 * np.linalg.slogdet(A)[1] - 0.5 * math.log(denom)
            - 0.5 * (n - 1) * math.log(2 * math.pi))
    assert log_evidence(K, y, lam, fit_bias=True, profile=False) == pytest.approx(want)


def test_the_units_of_the_activity_move_every_candidate_by_the_same_amount():
    """A density is not scale-free; the ranking that selects the hyper-parameters is."""
    X, K, y = _world()
    ks = [(a, np.exp(-a * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))) for a in (0.1, 0.3, 1.0)]
    lams = (1e-3, 1e-2, 0.1, 1.0)
    first = evidence_grid(ks, y, lams)
    for c in (7.0, 0.01):
        other = evidence_grid(ks, c * y, lams)
        assert (other[0], other[1]) == (first[0], first[1])
        shifts = {k: other[2][k] - first[2][k] for k in first[2]}
        assert max(shifts.values()) - min(shifts.values()) < 1e-6


def test_the_amplitude_is_profiled_rather_than_searched():
    """Scaling kernel and ridge together leaves the model unchanged and changes the evidence."""
    _X, K, y = _world()
    lam = 0.2
    base = log_evidence(K, y, lam)
    for s in (0.1, 3.0, 50.0):
        assert log_evidence(s * K, y, s * lam) == pytest.approx(base, abs=1e-8)
    assert profile_amplitude(12.0, 4) == pytest.approx(3.0)


def test_it_prefers_the_noise_it_was_given():
    """More noise in the data, more ridge in the choice: the criterion has to track that."""
    chosen = []
    for noise in (0.05, 1.5):
        X, _K, y = _world(n=50, noise=noise, seed=2)
        d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        ks = [(0.3, np.exp(-0.3 * d2))]
        _a, lam, _surface = evidence_grid(ks, y, (1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0))
        chosen.append(lam)
    assert chosen[1] > chosen[0]


def test_a_kernel_that_is_barely_positive_still_gives_a_number():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20, 2))
    K = X @ X.T                                   # rank 2, singular without the ridge
    y = rng.normal(size=20)
    assert np.isfinite(log_evidence(K, y, 1e-10))


def test_too_few_molecules_is_nan_rather_than_a_number():
    assert np.isnan(log_evidence(np.eye(2), [1.0, 2.0], 0.1))


def test_the_grid_reports_the_whole_surface_it_searched():
    X, _K, y = _world()
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    ks = [(a, np.exp(-a * d2)) for a in (0.1, 1.0)]
    lams = (0.01, 0.1, 1.0)
    alpha, lam, surface = evidence_grid(ks, y, lams)
    assert len(surface) == 6 and (alpha, lam) in surface
    assert surface[(alpha, lam)] == max(surface.values())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
