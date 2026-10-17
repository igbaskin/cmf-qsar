"""Tests for the one-class descriptions other than the enclosing ball of the 1-SVM."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.model import _Learner  # noqa: E402
from cmfqsar.oneclass import (LABELS, ONECLASS_METHODS, OneClassScorer,  # noqa: E402
                              enrichment_factor, make)

KINDS = ["ocmean", "ocknn", "ocgp", "ockpca"]


def _kernel(A, B):
    return np.exp(-0.5 * ((np.asarray(A)[:, None, :] - np.asarray(B)[None, :, :]) ** 2).sum(-1))


@pytest.fixture(scope="module")
def world():
    """A cloud of 'actives' on a line, and outsiders far away from it."""
    rng = np.random.default_rng(0)
    t = rng.uniform(-2, 2, size=70)
    X = np.stack([t, 0.15 * rng.normal(size=70)], axis=1)
    far = rng.normal(scale=0.4, size=(25, 2)) + np.array([7.0, 7.0])
    return X, far


@pytest.mark.parametrize("kind", KINDS)
def test_nu_is_the_fraction_left_outside(world, kind):
    """nu means the same thing for every description, which is what makes them comparable."""
    X, _ = world
    K = _kernel(X, X)
    for nu in (0.05, 0.2, 0.5):
        sc = make(kind, nu=nu).fit(K)
        inside = sc.decision(K, np.diag(K), training=True)
        assert abs(np.mean(inside < 0) - nu) <= 0.05


@pytest.mark.parametrize("kind", KINDS)
def test_the_outsiders_land_outside(world, kind):
    X, far = world
    K = _kernel(X, X)
    sc = make(kind, nu=0.1).fit(K)
    out = sc.decision(_kernel(far, X), np.ones(len(far)))
    assert np.all(out < 0)
    inside = sc.decision(K, np.diag(K), training=True)
    assert inside.mean() > out.mean()


@pytest.mark.parametrize("kind", KINDS)
def test_the_score_ranks_the_near_above_the_far(world, kind):
    """Monotonicity along a ray leaving the training cloud."""
    X, _ = world
    K = _kernel(X, X)
    sc = make(kind, nu=0.1).fit(K)
    ray = np.stack([np.zeros(6), np.linspace(0.0, 5.0, 6)], axis=1)
    s = sc.decision(_kernel(ray, X), np.ones(len(ray)))
    assert np.all(np.diff(s) < 1e-9)


def test_the_subspace_description_beats_the_ball_on_a_flat_set(world):
    """The actives lie on a line; a ball has to swallow space the subspace does not."""
    X, _ = world
    K = _kernel(X, X)
    off = np.stack([np.zeros(40), np.linspace(0.6, 2.0, 40)], axis=1)   # beside the line
    Ko = _kernel(off, X)
    kpca = make("ockpca", nu=0.1, components=6).fit(K)
    mean = make("ocmean", nu=0.1).fit(K)
    assert np.mean(kpca.decision(Ko, np.ones(len(off))) < 0) >= \
           np.mean(mean.decision(Ko, np.ones(len(off))) < 0)


def test_which_descriptions_are_kernel_expansions(world):
    """Two of them keep the co-field; the other two say so instead of pretending."""
    X, _ = world
    K = _kernel(X, X)
    for kind in KINDS:
        sc = make(kind, nu=0.1).fit(K)
        assert sc.is_linear == (kind in ("ocmean", "ocgp"))
        if sc.is_linear:
            # the decision value really is sum_j a_j K(x, x_j) + b
            got = sc.decision(K, np.diag(K), training=True)
            want = K @ sc.coef + sc.bias
            assert np.allclose(got, want, atol=1e-8)
        else:
            with pytest.raises(ValueError):
                _ = sc.coef


def test_the_gaussian_process_can_punish_what_it_knows_nothing_about(world):
    """mean_over_sd uses the posterior width, which the plain mean throws away."""
    X, far = world
    K = _kernel(X, X)
    plain = make("ocgp", nu=0.1, score="mean").fit(K)
    ratio = make("ocgp", nu=0.1, score="mean_over_sd").fit(K)
    Kf = _kernel(far, X)
    assert np.all(plain.decision(Kf, np.ones(len(far))) < 0)
    assert np.all(ratio.decision(Kf, np.ones(len(far))) < 0)
    # the two orderings differ: the ratio is a different description, not a rescaling
    a = plain.decision(K, np.diag(K), training=True)
    b = ratio.decision(K, np.diag(K), training=True)
    assert not np.array_equal(np.argsort(a), np.argsort(b))


def test_the_learner_fits_and_scores_every_method(world):
    X, far = world
    K = _kernel(X, X)
    for kind in KINDS:
        fit = _Learner(kind, {"nu": 0.1, "k": 5, "components": 5, "oc_lam": 0.1}).fit(K, np.ones(len(X)))
        assert fit.oc_ is not None
        here = fit.predict(K)                       # the training molecules, recognised as such
        there = fit.predict(_kernel(far, X), np.ones(len(far)))
        assert here.mean() > there.mean()
        if kind in ("ocmean", "ocgp"):
            assert fit.alpha_ is not None and np.allclose(fit.predict(K), K @ fit.alpha_ + fit.b_,
                                                          atol=1e-8)
        else:
            assert fit.alpha_ is None
            with pytest.raises(ValueError):
                fit.predict(_kernel(far, X))        # it needs the self-kernels


def test_a_description_needs_more_than_one_molecule_and_a_known_name(world):
    X, _ = world
    with pytest.raises(ValueError):
        make("ocmean").fit(_kernel(X[:1], X[:1]))
    with pytest.raises(ValueError):
        OneClassScorer(kind="ocsphere").fit(_kernel(X, X))
    with pytest.raises(ValueError):
        make("ocsvm")
    assert set(KINDS) < set(ONECLASS_METHODS) and all(k in LABELS for k in ONECLASS_METHODS)


def test_enrichment_counts_what_it_says():
    y = np.array([1, 1, 1, 1] + [-1] * 96, dtype=float)
    perfect = np.where(y > 0, 1.0, 0.0) + np.linspace(0, 1e-6, 100)
    assert enrichment_factor(y, perfect, 0.04) == pytest.approx(25.0)
    assert enrichment_factor(y, -perfect, 0.04) == pytest.approx(0.0)
    assert np.isnan(enrichment_factor(-np.ones(10), np.arange(10.0), 0.1))
