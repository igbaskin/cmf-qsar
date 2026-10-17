"""Tests for Gaussian-process classification by the Laplace approximation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.gpc import (LaplaceGPC, LIKELIHOODS, brier_score, calibration_table,  # noqa: E402
                         evidence, select_by_evidence, squash)
from cmfqsar.model import _Learner, classification_scores, cv_predict  # noqa: E402


def _problem(n=60, sep=1.0, seed=0, width=1.0):
    """A separable-ish two-class problem and its RBF kernel."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = np.where(X[:, 0] + 0.3 * X[:, 1] > 0, 1.0, -1.0)
    X[:, 0] += sep * 0.5 * y
    D = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * D / width ** 2), y, X


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_the_newton_iteration_reaches_the_mode(likelihood):
    """At the mode f = K grad, which is the stationarity condition itself."""
    K, y, _ = _problem()
    g = LaplaceGPC(likelihood=likelihood, amplitude=4.0).fit(K, y)
    assert g.converged and g.n_iter < 40
    assert np.allclose(g.f_, (g.amplitude * K + g.jitter * np.eye(len(y))) @ g.grad_, atol=1e-6)


@pytest.mark.parametrize("likelihood", LIKELIHOODS)
def test_the_mode_maximises_the_objective_it_claims_to(likelihood):
    """Nothing in a neighbourhood of the mode beats it on log p(y|f) - f K^-1 f / 2."""
    K, y, _ = _problem(n=30)
    g = LaplaceGPC(likelihood=likelihood, amplitude=2.0).fit(K, y)
    Ks = g.amplitude * K + g.jitter * np.eye(len(y))
    Ki = np.linalg.inv(Ks)

    def psi(f):
        if likelihood == "probit":
            from scipy.special import log_ndtr
            ll = float(np.sum(log_ndtr(y * f)))
        else:
            ll = float(-np.sum(np.logaddexp(0.0, -y * f)))
        return ll - 0.5 * float(f @ Ki @ f)

    here = psi(g.f_)
    rng = np.random.default_rng(3)
    for _ in range(25):
        assert psi(g.f_ + rng.normal(scale=0.05, size=len(y))) <= here + 1e-8


def test_the_marginal_likelihood_matches_its_definition():
    """log q(y) = log p(y|f) - f K^-1 f / 2 - log|I + K W| / 2, computed the slow way."""
    K, y, _ = _problem(n=40)
    g = LaplaceGPC(amplitude=3.0).fit(K, y)
    Ks = g.amplitude * K + g.jitter * np.eye(len(y))
    from scipy.special import log_ndtr
    z = y * g.f_
    phi_over_Phi = np.exp(-0.5 * z * z - 0.5 * np.log(2 * np.pi) - log_ndtr(z))
    W = phi_over_Phi ** 2 + z * phi_over_Phi
    want = (float(np.sum(log_ndtr(z)))
            - 0.5 * float(g.f_ @ np.linalg.solve(Ks, g.f_))
            - 0.5 * float(np.linalg.slogdet(np.eye(len(y)) + Ks * W[None, :])[1]))
    assert g.log_marginal_likelihood == pytest.approx(want, abs=1e-6)


def test_the_latent_mean_is_a_kernel_expansion():
    """Which is why the co-field and the atom contributions survive this classifier."""
    K, y, _ = _problem()
    g = LaplaceGPC(amplitude=2.0).fit(K, y)
    assert np.allclose(g.latent(K), K @ g.coef)
    # and the learner used by the rest of the program agrees with it
    fit = _Learner("gpc", {"amplitude": 2.0}).fit(K, y)
    assert np.allclose(fit.predict(K), g.latent(K), atol=1e-10)
    assert fit.b_ == 0.0


def test_probabilities_are_bounded_calibrated_and_softened_by_the_variance():
    K, y, _ = _problem(n=80, sep=2.0)
    g = LaplaceGPC(amplitude=5.0).fit(K, y)
    p = g.predict_proba(K, np.ones(len(y)))
    assert np.all((p > 0.0) & (p < 1.0))
    assert p[y > 0].mean() > 0.7 and p[y < 0].mean() < 0.3
    assert brier_score(y, p) < 0.15
    # the posterior variance pulls a probability towards one half, never away from it
    mean = g.latent(K)
    sharp = squash(mean, np.zeros_like(mean))
    soft = squash(mean, np.full_like(mean, 4.0))
    assert np.all(np.abs(soft - 0.5) <= np.abs(sharp - 0.5) + 1e-12)
    rows = calibration_table(y, p, bins=4)
    assert rows and all(0.0 <= r[2] <= 1.0 for r in rows)


def test_the_variance_is_smaller_where_the_molecules_are():
    """A point among the training data is known better than one far away from all of it."""
    K, y, X = _problem(n=60)
    g = LaplaceGPC(amplitude=2.0).fit(K, y)
    far = np.array([[12.0, 12.0]])
    k_far = np.exp(-0.5 * ((far[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    v_here = g.latent_variance(K[:1], np.ones(1))
    v_far = g.latent_variance(k_far, np.ones(1))
    assert v_far[0] > v_here[0]
    assert v_far[0] == pytest.approx(g.amplitude, abs=1e-6)      # back to the prior
    assert g.predict_proba(k_far, np.ones(1))[0] == pytest.approx(0.5, abs=0.05)


def test_the_evidence_prefers_the_kernel_that_knows_the_answer():
    """Model selection without a single fold held out."""
    K, y, X = _problem(n=70, sep=1.5)
    rng = np.random.default_rng(11)
    Z = rng.normal(size=(len(y), 2))
    noise = np.exp(-0.5 * ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1))
    key, amp, lml = select_by_evidence({"informative": K, "noise": noise}, y,
                                       amplitudes=(0.3, 1.0, 3.0, 10.0))
    assert key == "informative"
    assert lml > evidence(noise, y, amplitude=amp)
    # and the choice agrees with what cross-validation would have said
    auc_good = classification_scores(y, cv_predict(K, y, "gpc", {"amplitude": amp}))["auc"]
    auc_noise = classification_scores(y, cv_predict(noise, y, "gpc", {"amplitude": amp}))["auc"]
    assert auc_good > auc_noise


def test_one_class_is_refused_and_the_likelihood_is_checked():
    K, y, _ = _problem(n=20)
    with pytest.raises(ValueError):
        LaplaceGPC().fit(K, np.ones_like(y))
    with pytest.raises(ValueError):
        LaplaceGPC(likelihood="hinge").fit(K, y)


def test_it_classifies_as_well_as_the_support_vector_machine_on_a_clean_problem():
    K, y, _ = _problem(n=100, sep=1.5, seed=5)
    gp = classification_scores(y, cv_predict(K, y, "gpc", {"amplitude": 5.0}))
    svc = classification_scores(y, cv_predict(K, y, "svc", {"C": 5.0}))
    assert gp["mcc"] > 0.5
    assert gp["mcc"] >= svc["mcc"] - 0.15
