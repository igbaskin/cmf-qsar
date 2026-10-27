"""Choosing a batch of compounds at once (plan item MC2).

The cheap rule takes the best scores and skips what looks too similar; the rule this item adds
conditions the posterior covariance on every pick instead. What these tests fix is that the
conditioning is the Cholesky step it claims to be, that the greedy batch it gives is the one of
largest determinant, that a batch of one is still the single best candidate, and that the loop
acquires a whole batch between fits.
"""
import itertools
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.active import (BATCH_RULES, batch_by_cosine, batch_by_covariance,  # noqa: E402
                            condition_on, conditional_variances, learning_curve,
                            log_determinant)


def _covariance(n=8, seed=0, rank=4):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, rank))
    return A @ A.T + 0.05 * np.eye(n)


# ------------------------------------------------------------- conditioning


def test_conditioning_is_one_step_of_a_cholesky():
    """cov - c c' / c_jj is what a pivot does, and the pivot itself ends with no variance left."""
    C = _covariance()
    out = condition_on(C, 3)
    assert out[3, 3] == pytest.approx(0.0, abs=1e-12)
    assert np.allclose(out, out.T)
    want = C - np.outer(C[:, 3], C[3, :]) / C[3, 3]
    assert np.allclose(out, want)


def test_conditioning_never_raises_a_variance():
    C = _covariance()
    before = conditional_variances(C)
    after = conditional_variances(condition_on(C, 0))
    assert np.all(after <= before + 1e-12)
    assert np.any(after < before - 1e-9), "knowing one candidate must tell something about some"


def test_a_candidate_that_says_nothing_new_loses_all_of_its_variance():
    """Two candidates the model cannot tell apart: after one, the other is worth nothing."""
    C = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    after = conditional_variances(condition_on(C, 0))
    assert after[1] == pytest.approx(0.0, abs=1e-12)
    assert after[2] == pytest.approx(1.0)


def test_a_pivot_with_no_variance_left_is_handled_and_not_divided_by():
    C = np.diag([1.0, 0.0, 1.0])
    out = condition_on(C, 1)
    assert np.all(np.isfinite(out))
    assert out[0, 0] == pytest.approx(1.0)


# ------------------------------------------------------------ the batch rules


def test_the_greedy_batch_is_the_one_of_largest_determinant():
    C = _covariance(n=7, seed=3)
    got = batch_by_covariance("variance", np.zeros(7), C, 3)
    best = max(itertools.combinations(range(7), 3), key=lambda r: log_determinant(C, r))
    assert log_determinant(C, got) == pytest.approx(log_determinant(C, best), rel=1e-9)


def test_a_batch_never_repeats_a_candidate_and_has_the_size_asked_for():
    C = _covariance(n=6)
    got = batch_by_covariance("variance", np.zeros(6), C, 4)
    assert len(got) == 4 and len(set(got)) == 4


def test_a_batch_of_one_is_the_best_candidate_by_the_score():
    C = _covariance(n=5)
    mu = np.array([0.0, 5.0, 1.0, 2.0, 3.0])
    assert batch_by_covariance("greedy", mu, C, 1) == [1]


def test_the_two_candidates_the_model_cannot_tell_apart_are_not_both_picked():
    """This is what the cosine filter uses a threshold for and the covariance does not."""
    C = np.array([[1.0, 0.99, 0.0], [0.99, 1.0, 0.0], [0.0, 0.0, 0.9]])
    got = batch_by_covariance("variance", np.zeros(3), C, 2)
    assert set(got) == {0, 2} or set(got) == {1, 2}


def test_the_cosine_rule_skips_what_is_too_similar_and_keeps_the_order():
    score = np.array([3.0, 2.9, 1.0])
    cos = np.array([[1.0, 0.999, 0.1], [0.999, 1.0, 0.1], [0.1, 0.1, 1.0]])
    assert batch_by_cosine(score, cos, 2, max_similarity=0.98) == [0, 2]
    assert batch_by_cosine(score, cos, 2, max_similarity=1.0) == [0, 1]
    assert batch_by_cosine(score, None, 3) == [0, 1, 2]


def test_the_determinant_of_an_empty_batch_is_not_a_number():
    assert log_determinant(_covariance(), []) == float("-inf")


def test_both_rules_are_named_in_the_table():
    assert set(BATCH_RULES) == {"cosine", "covariance"}


# --------------------------------------------------------------- the loop


def _toy(n=40, seed=0):
    """A kernel over molecules on a line, and an activity that follows the position."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n)[:, None]
    K = np.exp(-((x - x.T) ** 2) / 0.02)
    y = np.sin(6.0 * x[:, 0]) + 0.05 * rng.normal(size=n)
    return {1.0: K}, y


def test_a_batch_run_acquires_the_whole_batch_between_fits():
    Ks, y = _toy()
    start = list(range(5))
    out = learning_curve(Ks, y, "variance", start, steps=3, batch=4, batch_rule="covariance")
    assert len(out["taken"]) == 3, "three fits"
    assert all(len(t) == 4 for t in out["taken"]), "four molecules each"
    assert len(out["have"]) == len(start) + 12
    assert len(set(out["have"])) == len(out["have"]), "nothing is acquired twice"


def test_the_two_batch_rules_both_run_and_differ():
    Ks, y = _toy()
    start = list(range(5))
    a = learning_curve(Ks, y, "variance", start, steps=2, batch=5, batch_rule="covariance")
    b = learning_curve(Ks, y, "variance", start, steps=2, batch=5, batch_rule="cosine")
    assert sorted(a["have"]) != sorted(b["have"]), \
        "on a kernel this smooth the two rules must choose differently"


def test_a_batch_of_one_is_the_loop_that_was_there_before():
    Ks, y = _toy()
    start = list(range(5))
    one = learning_curve(Ks, y, "variance", start, steps=4)
    same = learning_curve(Ks, y, "variance", start, steps=4, batch=1, batch_rule="covariance")
    assert one["taken"] == same["taken"]


def test_the_error_after_the_last_batch_is_reported():
    Ks, y = _toy()
    out = learning_curve(Ks, y, "variance", list(range(5)), steps=2, batch=4)
    assert "rmse_end" in out and np.isfinite(out["rmse_end"])
    assert len(out["rmse"]) == 2, "the curve still has one entry per fit"


def test_an_unknown_batch_rule_is_refused():
    Ks, y = _toy()
    with pytest.raises(ValueError, match="unknown batch rule"):
        learning_curve(Ks, y, "variance", list(range(5)), steps=1, batch=3, batch_rule="nope")
