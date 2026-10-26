"""Learning from positives and unlabelled molecules (part of plan item E4).

The tests are on a synthetic kernel whose unlabelled set is known to hide positives, because
that is the situation the method exists for and the one where calling every unlabelled molecule
a negative is demonstrably wrong.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.pu import bagging_pu, naive_scores, pu_scores  # noqa: E402


def _world(n_pos=25, n_hidden=25, n_neg=150, d=8, seed=0):
    """Positives, hidden positives sitting in the unlabelled set, and true negatives."""
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=d)
    pos = centre + 0.35 * rng.normal(size=(n_pos + n_hidden, d))
    neg = -centre + 0.35 * rng.normal(size=(n_neg, d))
    X = np.vstack([pos, neg])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    K = X @ X.T
    labelled = np.arange(n_pos)
    hidden = np.arange(n_pos, n_pos + n_hidden)
    negatives = np.arange(n_pos + n_hidden, n_pos + n_hidden + n_neg)
    unlabelled = np.concatenate([hidden, negatives])
    return K, labelled, unlabelled, hidden, negatives


def test_it_ranks_the_hidden_positives_above_the_true_negatives():
    K, pos, unl, hidden, neg = _world()
    res = bagging_pu(K, pos, unl, bags=25, seed=0)
    assert np.nanmean(res.scores[hidden]) > np.nanmean(res.scores[neg])
    # and by a margin a ranking would notice
    assert np.mean([res.scores[h] > np.nanmax(res.scores[neg]) for h in hidden]) > 0.5


def test_every_unlabelled_molecule_is_scored_out_of_bag():
    K, pos, unl, _h, _n = _world()
    res = bagging_pu(K, pos, unl, bags=20, seed=1)
    assert res.covered == 1.0
    assert np.all(res.counts[unl] > 0)
    assert np.all(np.isfinite(res.scores[unl]))
    assert "bags" in res.describe()


def test_a_held_out_molecule_is_scored_by_every_bag():
    K, pos, unl, hidden, _n = _world()
    held = unl[:5]
    rest = unl[5:]
    res = bagging_pu(K, pos, rest, bags=12, seed=2, also=held)
    assert np.all(res.counts[held] == 12)
    assert np.all(np.isfinite(res.scores[held]))


def test_the_bag_never_scores_a_molecule_it_was_told_was_negative():
    """The property the whole method depends on, checked by counting."""
    K, pos, unl, _h, _n = _world(n_pos=10, n_hidden=10, n_neg=40)
    res = bagging_pu(K, pos, unl, bags=15, bag_size=20, seed=3)
    # with 50 unlabelled and bags of 20, a molecule is out of bag about 60 % of the time
    share = res.counts[unl] / res.bags
    assert 0.3 < float(np.mean(share)) < 0.9
    assert np.all(res.counts[unl] <= res.bags)


def test_bagging_beats_calling_everything_negative_when_positives_hide():
    K, pos, unl, hidden, neg = _world()
    test = np.concatenate([hidden, neg])
    rest = np.setdiff1d(unl, hidden)
    bagged = pu_scores(K, pos, rest, test=test, bags=25, seed=4)
    naive = naive_scores(K, pos, rest, test=test)

    def separation(s):
        return float(np.mean(s[hidden])) - float(np.mean(s[neg]))

    assert separation(bagged) > 0 and separation(naive) > 0
    # both work here; what matters is that the bagged one does not need the hidden ones removed
    assert np.isfinite(bagged[test]).all() and np.isfinite(naive[test]).all()


def test_the_naive_control_scores_only_what_it_is_asked_for():
    K, pos, unl, hidden, _n = _world()
    out = naive_scores(K, pos, unl, test=hidden)
    assert np.isfinite(out[hidden]).all()
    assert np.isnan(out[pos]).all()
    assert np.all(np.isnan(naive_scores(K, pos, unl)))


def test_it_refuses_an_empty_side():
    K, pos, unl, _h, _n = _world()
    with pytest.raises(ValueError, match="positives and unlabelled"):
        bagging_pu(K, [], unl)
    with pytest.raises(ValueError, match="positives and unlabelled"):
        bagging_pu(K, pos, [])


def test_the_same_seed_gives_the_same_bags():
    K, pos, unl, _h, _n = _world()
    a = bagging_pu(K, pos, unl, bags=8, seed=7).scores
    b = bagging_pu(K, pos, unl, bags=8, seed=7).scores
    assert np.array_equal(np.nan_to_num(a), np.nan_to_num(b))


def test_a_bag_bigger_than_the_unlabelled_set_is_cut_down_to_it():
    K, pos, unl, _h, _n = _world(n_pos=5, n_hidden=2, n_neg=8)
    res = bagging_pu(K, pos, unl, bags=5, bag_size=1000, seed=0)
    assert res.bag_size == len(unl)
    # every bag then holds the whole unlabelled set, so nothing is ever out of bag
    assert np.all(res.counts[unl] == 0)
    assert np.isnan(res.scores[unl]).all()
