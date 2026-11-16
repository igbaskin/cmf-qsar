"""Tests of cmfqsar.cme: the expected field at an activity (conditional mean embedding)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cme import (Standardiser, activity_kernel, centre_weights, cme_weights,  # noqa: E402
                         distance_to, expected_fields, field_loss, fit_cme, pattern)

GPB = os.path.join(os.path.dirname(__file__), "..", "examples", "sutherland", "GPB.sdf")


def _features(n=60, seed=0, how="line"):
    """Explicit feature vectors whose linear kernel plays the field kernel."""
    rng = np.random.RandomState(seed)
    y = rng.randn(n)
    noise = rng.randn(n, 6)
    if how == "line":
        signal = y[:, None] * np.array([[1.0, 0.5]])
    elif how == "square":
        signal = (y ** 2 - 1.0)[:, None] * np.array([[1.0, 0.5]])
    else:
        signal = np.zeros((n, 2))
    X = np.hstack([signal, 0.5 * noise])
    return X, y


def test_centred_weights_sum_to_one():
    beta = np.random.RandomState(0).rand(7, 3) * 0.1
    assert np.allclose(centre_weights(beta).sum(axis=0), 1.0)


def test_linear_weights_closed_form():
    y = np.random.RandomState(1).randn(25)
    u = Standardiser.fit(y)(y)
    lam = 0.05
    w = cme_weights(u, [0.7, -1.2], "linear", lam=lam)
    n = len(u)
    expect = 1.0 / n + np.outer(u, [0.7, -1.2]) / (n * (1.0 + lam))
    assert np.allclose(w, expect)


def test_linear_expected_field_is_the_straight_line_of_the_field_on_the_activity():
    X, y = _features()
    u = Standardiser.fit(y)(y)
    w = cme_weights(u, [1.5], "linear", lam=1e-7)[:, 0]
    A = np.c_[np.ones(len(u)), u]
    coef = np.linalg.lstsq(A, X, rcond=None)[0]
    assert np.allclose(w @ X, coef[0] + 1.5 * coef[1])
    # its slope is the pattern: the covariance of the field with the standardised activity
    assert np.allclose(coef[1], pattern(y, X))


def test_field_loss_and_distance_match_explicit_features():
    X, y = _features(n=30)
    K = X @ X.T
    tr, te = np.arange(20), np.arange(20, 30)
    st = Standardiser.fit(y[tr])
    beta = cme_weights(st(y[tr]), st(y[te]), "gauss", gamma=0.5, lam=0.01)
    explicit = np.sum((X[te] - (beta.T @ X[tr])) ** 2, axis=1)
    assert np.allclose(field_loss(K, te, tr, beta), explicit)
    b = beta[:, 0]
    d = distance_to(b, K[np.ix_(tr, tr)], K[np.ix_(te, tr)], np.diag(K)[te])
    assert np.allclose(d, np.sum((X[te] - b @ X[tr]) ** 2, axis=1))


def test_activity_kernel_shapes_and_errors():
    assert activity_kernel([0.0, 1.0], [0.0, 1.0, 2.0], "gauss", 1.0).shape == (2, 3)
    with pytest.raises(ValueError):
        activity_kernel([0.0], [0.0], "cubic")
    with pytest.raises(ValueError):
        fit_cme(np.eye(3), [1.0, 2.0, 3.0])


def test_explained_share_finds_the_dependence_and_its_shape():
    X, y = _features(n=120, how="line")
    lin = fit_cme(X @ X.T, y, "linear")
    gau = fit_cme(X @ X.T, y, "gauss")
    assert lin.explained > 0.2 and gau.explained > 0.2
    assert abs(lin.explained - gau.explained) < 0.05
    X, y = _features(n=120, how="square")
    lin = fit_cme(X @ X.T, y, "linear")
    gau = fit_cme(X @ X.T, y, "gauss")
    # a field that depends on the square of the activity is invisible to a straight line
    assert lin.explained < 0.02 and gau.explained > 0.2
    X, y = _features(n=120, how="none")
    assert fit_cme(X @ X.T, y, "gauss").explained < 0.02


def test_outside_the_range_the_gaussian_field_falls_back_to_the_mean():
    X, y = _features(n=80, how="line")
    fit = fit_cme(X @ X.T, y, "gauss")
    far, near = expected_fields(fit, [y.max() + 200 * y.std(), np.median(y)])
    assert far.from_mean > 0.99 and near.from_mean < 0.5
    assert np.allclose(far.on(X), X.mean(axis=0), atol=1e-3)
    assert np.isclose(far.weights.sum(), 1.0) and np.isclose(near.weights.sum(), 1.0)
    assert 1.0 <= near.effective_size <= len(y)
    names = [f"m{i}" for i in range(len(y))]
    top = near.leading(names, 3)
    assert len(top) == 3 and top[0][1] >= top[1][1] >= top[2][1]


def test_expected_command(capsys, tmp_path):
    from cmfqsar.cli import main

    prefix = str(tmp_path / "gpb")
    weights = str(tmp_path / "w.csv")
    main(["expected", GPB, "--charges", "file", "--fields", "steric", "--spacing", "1.0",
          "--at", "6", "--at", "20", "--out-prefix", prefix, "--weights-out", weights])
    text = capsys.readouterr().out
    assert "explains" in text and "Expected field at activity 6" in text
    assert "20 lies outside the training activities" in text
    for name in ("pattern_steric", "expected_6_steric", "departure_6_steric",
                 "expected_20_steric"):
        assert os.path.exists(f"{prefix}_{name}.cube")
    rows = open(weights, encoding="utf-8").read().splitlines()
    assert rows[0] == "name,activity,weight at 6,weight at 20" and len(rows) == 67


def test_cofield_tends_to_the_pattern_as_the_ridge_grows():
    """Kernel ridge regression with a huge ridge has dual coefficients (y - mean y) / lam, so its
    map, the co-field, points along the pattern; with a small ridge it does not."""
    from cmfqsar.model import krr_fit

    X, y = _features(n=50, how="line")
    K = X @ X.T
    p = pattern(y, X)

    def cos(a, b):
        return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

    a_big, _ = krr_fit(K, y, 1e8, True)
    a_small, _ = krr_fit(K, y, 1e-6, True)
    assert cos(a_big @ X, p) > 0.9999
    assert cos(a_small @ X, p) < 0.99
