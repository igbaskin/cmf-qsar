"""Several endpoints at once: the evidence, a free-form covariance, and task descriptors (E1).

``tests/test_multitask.py`` covers the two-endpoint behaviour. This file covers what the more
general forms add: an objective that holds nothing out, a covariance with more than one parameter,
and an endpoint that was never measured.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.evidence import log_evidence as single_evidence  # noqa: E402
from cmfqsar.multitask import (CovarianceFit, Observations, descriptor_covariance,  # noqa: E402
                               evidence_rho, fit, fit_covariance, log_evidence,
                               low_rank_covariance, pair_kernel, standardise, task_covariance,
                               task_kernel)


def _kernel(n=40, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X @ X.T


def _world(n=50, blocks=((0, 1, 2), (3, 4)), seed=1, missing=0.4):
    """Five endpoints in two groups: inside a group the functions are almost the same."""
    rng = np.random.default_rng(seed)
    K = _kernel(n, 8, seed)
    L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
    n_tasks = sum(len(b) for b in blocks)
    Y = np.zeros((n, n_tasks))
    for b in blocks:
        shared = L @ rng.normal(size=n)
        for t in b:
            Y[:, t] = shared + 0.15 * (L @ rng.normal(size=n))
    Y += 0.05 * rng.normal(size=Y.shape)
    gaps = rng.random(Y.shape) < missing
    gaps[:8, :] = False                      # every endpoint keeps a core of measurements
    Y[gaps] = np.nan
    return K, Y, blocks


# --------------------------------------------------------------------------------- the evidence


def test_the_evidence_matches_a_dense_computation():
    """The Cholesky route is used for speed, so it must agree with the direct computation."""
    K, Y, _ = _world()
    obs = Observations.from_columns(Y)
    T = task_covariance(obs.n_tasks, 0.4)
    lam = 0.17
    A = pair_kernel(K, obs, obs, T) + lam * np.eye(len(obs))
    Ai = np.linalg.inv(A)
    B, y = obs.design(), obs.y
    M = B.T @ Ai @ B
    w = np.linalg.solve(M, B.T @ Ai @ y)
    r = y - B @ w
    quad = float(r @ Ai @ r)
    dof = len(y) - B.shape[1]
    ref = (-0.5 * dof * (math.log(quad / dof) + 1.0)
           - 0.5 * float(np.linalg.slogdet(A)[1])
           - 0.5 * float(np.linalg.slogdet(M)[1])
           - 0.5 * dof * math.log(2.0 * math.pi))
    assert log_evidence(K, obs, T, lam) == pytest.approx(ref, abs=1e-6)


def test_with_one_task_it_is_the_single_task_evidence():
    """One endpoint, one covariance entry, one constant mean: the two formulas must agree."""
    K = _kernel(30)
    rng = np.random.default_rng(3)
    y = rng.normal(size=30)
    obs = Observations.from_columns(y[:, None])
    for lam in (0.01, 0.3, 3.0):
        assert log_evidence(K, obs, np.ones((1, 1)), lam) == pytest.approx(
            single_evidence(K, y, lam), abs=1e-6)


def test_a_covariance_that_cannot_be_factorised_is_refused_rather_than_crashing():
    K, Y, _ = _world(n=30)
    obs = Observations.from_columns(Y)
    bad = -np.ones((obs.n_tasks, obs.n_tasks))      # indefinite
    assert log_evidence(K, obs, bad, 1e-9) == float("-inf")


# ------------------------------------------------------------------- the free-form covariance


def test_the_low_rank_form_is_positive_definite_whatever_the_parameters():
    rng = np.random.default_rng(5)
    for rank in (1, 2, 3):
        for _ in range(10):
            theta = rng.normal(scale=3.0, size=5 * (rank + 1))
            T = low_rank_covariance(theta, 5, rank)
            assert np.all(np.linalg.eigvalsh(T) > 0)
            assert float(np.mean(np.diag(T))) == pytest.approx(1.0)


def test_the_rank_one_start_reproduces_one_shared_correlation():
    """The free-form family must contain the one-correlation model, or no comparison is possible."""
    from cmfqsar.multitask import _start_vector
    for rho in (0.1, 0.5, 0.9):
        T = low_rank_covariance(_start_vector(4, 1, rho), 4, 1)
        assert T == pytest.approx(task_covariance(4, rho), abs=1e-6)


def test_it_refuses_a_parameter_vector_of_the_wrong_length():
    with pytest.raises(ValueError, match="theta must hold"):
        low_rank_covariance(np.zeros(3), 4, 1)


def test_the_free_form_covariance_finds_the_blocks():
    K, Y, blocks = _world()
    obs = Observations.from_columns(Y)
    got = fit_covariance(K, obs, rank=2, rhos=(0.3, 0.8))
    C = got.correlations
    within = [C[i, j] for b in blocks for i in b for j in b if i < j]
    across = [C[i, j] for i in blocks[0] for j in blocks[1]]
    assert min(within) > max(across)
    assert "free" in got.describe(["a", "b", "c", "d", "e"])


def test_the_free_form_covariance_is_at_least_as_probable_as_one_correlation():
    K, Y, _ = _world()
    obs = Observations.from_columns(Y)
    one = evidence_rho(K, obs)
    free = fit_covariance(K, obs, rank=2, rhos=(0.3, 0.8))
    assert free.evidence > one.evidence
    assert "rho" in one.note and one.rank == 1


def test_it_refuses_a_single_task():
    K = _kernel(20)
    obs = Observations.from_columns(np.random.default_rng(0).normal(size=(20, 1)))
    with pytest.raises(ValueError, match="at least two tasks"):
        fit_covariance(K, obs)


# ------------------------------------------------------------------------- task descriptors


def test_the_task_kernel_falls_off_with_descriptor_distance():
    Z = np.array([[0.0], [1.0], [5.0]])
    T = task_kernel(Z, length_scale=1.0, nugget=0.0)
    assert np.allclose(np.diag(T), 1.0)
    assert T[0, 1] > T[0, 2]
    assert np.all(np.linalg.eigvalsh(T) > -1e-12)


def test_the_nugget_keeps_identical_endpoints_from_being_the_same_endpoint():
    Z = np.array([[0.0], [0.0]])                 # two endpoints with the same descriptors
    assert task_kernel(Z, 1.0, 0.0)[0, 1] == pytest.approx(1.0)
    assert np.linalg.det(task_kernel(Z, 1.0, 0.0)) == pytest.approx(0.0, abs=1e-12)
    T = task_kernel(Z, 1.0, 0.25)
    assert T[0, 1] < 1.0 and np.linalg.det(T) > 0.1


def test_the_linear_task_kernel_can_be_negative():
    Z = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert task_kernel(Z, kind="linear", nugget=0.0)[0, 1] == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="unknown task kernel"):
        task_kernel(Z, kind="cosine-ish")


def test_a_descriptor_covariance_reproduces_its_own_rows():
    K, Y, _ = _world()
    obs = Observations.from_columns(Y)
    Z = np.array([[0.0], [0.1], [0.2], [3.0], [3.1]])
    dc = descriptor_covariance(K, obs, Z, length_scales=(0.5, 1.0), nuggets=(0.1,), lams=(0.1,))
    for t in range(obs.n_tasks):
        row = dc.row_for(Z[t])
        off = [j for j in range(obs.n_tasks) if j != t]
        assert row[off] == pytest.approx(dc.T[t, off], abs=1e-9)
    assert dc.as_fit().kind.startswith("descriptor")


def test_standardise_leaves_a_constant_column_alone():
    Z = np.array([[1.0, 5.0], [3.0, 5.0]])
    out = standardise(Z)
    assert np.allclose(out[:, 1], 0.0)
    assert out[:, 0].std() == pytest.approx(1.0)


def test_a_descriptor_covariance_wants_one_row_per_task():
    K, Y, _ = _world()
    obs = Observations.from_columns(Y)
    with pytest.raises(ValueError, match="one row per task"):
        descriptor_covariance(K, obs, np.zeros((3, 1)))


# ---------------------------------------------------------------- the mean, and a new endpoint


def test_task_features_of_the_identity_reproduce_the_free_biases():
    """The indicator design is a special case, so the general form must reproduce it."""
    K, Y, _ = _world()
    plain = Observations.from_columns(Y)
    typed = Observations.from_columns(Y, task_features=np.eye(Y.shape[1]))
    a = fit(K, plain, rho=0.4, lam=0.2)
    b = fit(K, typed, rho=0.4, lam=0.2)
    assert a.bias == pytest.approx(b.bias, abs=1e-8)
    assert a.coef == pytest.approx(b.coef, abs=1e-8)


def test_a_descriptor_mean_is_linear_in_the_descriptors():
    K, Y, _ = _world()
    Z = np.c_[np.ones(Y.shape[1]), np.arange(Y.shape[1], dtype=float)]
    obs = Observations.from_columns(Y, task_features=Z)
    f = fit(K, obs, rho=0.4, lam=0.2)
    assert f.mean_coef.shape == (2,)
    assert f.bias == pytest.approx(Z @ f.mean_coef, abs=1e-10)


def test_the_new_task_route_reproduces_a_trained_endpoint_exactly():
    """Given a fitted endpoint's own covariance row and descriptors, it must reproduce it."""
    K, Y, _ = _world()
    Z = np.c_[np.ones(Y.shape[1]), np.arange(Y.shape[1], dtype=float)]
    obs = Observations.from_columns(Y, task_features=Z)
    f = fit(K, obs, rho=0.5, lam=0.2)
    for t in range(obs.n_tasks):
        assert f.predict_new_task(K, f.T[t], features=Z[t]) == pytest.approx(
            f.predict(K, t), abs=1e-9)


def test_without_descriptors_the_new_endpoint_gets_a_weighted_bias_and_says_so():
    K, Y, _ = _world()
    obs = Observations.from_columns(Y)
    f = fit(K, obs, rho=0.5, lam=0.2)
    row = np.array([0.9, 0.2, 0.2, 0.2, 0.2])
    out = f.predict_new_task(K, row)
    assert np.all(np.isfinite(out))
    assert min(f.bias) - 1e-9 <= float(np.mean(out) - np.mean(K @ f.new_task_coefficients(row))) \
        <= max(f.bias) + 1e-9
    with pytest.raises(ValueError, match="one entry per trained task"):
        f.predict_new_task(K, row[:3])


def test_an_endpoint_with_no_measurements_is_predicted_from_its_neighbours():
    """The claim being tested: remove a whole endpoint and predict it from the others."""
    K, Y, _ = _world(n=60, seed=7, missing=0.2)
    Z = np.array([[0.0], [0.1], [0.2], [3.0], [3.1]])
    held = 2                                       # inside the first block, so it has neighbours
    keep = [t for t in range(Y.shape[1]) if t != held]
    Zk = np.c_[np.ones(len(keep)), Z[keep, 0]]
    obs = Observations.from_columns(Y[:, keep], task_features=Zk)
    dc = descriptor_covariance(K, obs, Z[keep], length_scales=(0.5, 1.0, 2.0),
                               nuggets=(0.1,), lams=(0.1,))
    f = fit(K, obs, lam=dc.lam, T=dc.T)
    pred = f.predict_new_task(K, dc.row_for(Z[held]), features=[1.0, float(Z[held, 0])])
    truth = Y[:, held]
    ok = np.isfinite(truth)
    r = float(np.corrcoef(pred[ok], truth[ok])[0, 1])
    # the reference value that needs no model: the mean of the endpoints that were measured
    base = np.nanmean(Y[:, keep], axis=1)
    rb = float(np.corrcoef(base[ok], truth[ok])[0, 1])
    assert r > 0.5
    assert r > rb - 0.15            # it need not exceed the mean, but must be close to it


def test_the_subset_keeps_the_task_features():
    K, Y, _ = _world()
    Z = np.eye(Y.shape[1])
    obs = Observations.from_columns(Y, task_features=Z)
    sub = obs.subset(np.arange(0, len(obs), 2))
    assert sub.task_features is not None
    assert sub.design().shape == (len(sub), Y.shape[1])
    assert fit(K, sub, rho=0.2, lam=0.5).bias.shape == (Y.shape[1],)


def test_from_columns_checks_the_feature_shape():
    with pytest.raises(ValueError, match="one row per task"):
        Observations.from_columns(np.zeros((5, 3)), task_features=np.zeros((2, 1)))


def test_a_covariance_fit_reports_its_correlations():
    T = np.diag([4.0, 1.0]) + 0.5
    got = CovarianceFit(T=T, lam=0.1, evidence=-1.0, rank=1)
    C = got.correlations
    assert np.allclose(np.diag(C), 1.0)
    assert C[0, 1] == pytest.approx(0.5 / math.sqrt(4.5 * 1.5))
