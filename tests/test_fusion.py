"""Tests for data fusion over several reference actives."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fusion import FUSION_RULES, FusionScorer, fuse, make  # noqa: E402
from cmfqsar.oneclass import make as make_oneclass  # noqa: E402

# three library molecules against four references, chosen so that every rule disagrees
S = np.array([
    [0.90, 0.10, 0.10, 0.10],     # resembles one reference strongly
    [0.40, 0.40, 0.40, 0.40],     # resembles all of them moderately
    [0.20, 0.30, 0.55, 0.05],     # in between
])


def test_every_rule_computes_what_it_says():
    assert fuse(S, "max").tolist() == pytest.approx([0.90, 0.40, 0.55])
    assert fuse(S, "min").tolist() == pytest.approx([0.10, 0.40, 0.05])
    assert fuse(S, "sum").tolist() == pytest.approx([1.20, 1.60, 1.10])
    assert fuse(S, "med").tolist() == pytest.approx([0.10, 0.40, 0.25])
    assert fuse(S, "euc").tolist() == pytest.approx(
        [float(np.sqrt((S[i] ** 2).sum())) for i in range(3)])
    # anz divides the total by how many passed and mnz multiplies by it
    anz = fuse(S, "anz", threshold=0.25)
    mnz = fuse(S, "mnz", threshold=0.25)
    assert anz.tolist() == pytest.approx([0.90, 0.40, 0.425])
    assert mnz.tolist() == pytest.approx([0.90, 6.40, 1.70])
    # and they order the first two molecules oppositely, which is why both are offered
    assert anz[0] > anz[1] and mnz[0] < mnz[1]


def test_sum_ranks_as_the_mean_and_max_as_the_nearest_neighbour():
    """The two rules this program already had, reached from the other direction."""
    assert np.argsort(-fuse(S, "sum")).tolist() == np.argsort(-S.mean(axis=1)).tolist()
    # with a normalised kernel d^2 = 2 - 2s, so the nearest neighbour is the largest similarity:
    # MAX and the one-nearest-neighbour description must rank a library identically
    K = np.array([[1.0, 0.3, 0.2], [0.3, 1.0, 0.25], [0.2, 0.25, 1.0]])
    knn = make_oneclass("ocknn", nu=0.1, k=1).fit(K)
    rows = np.array([[0.9, 0.1, 0.1], [0.4, 0.4, 0.4], [0.2, 0.3, 0.55]])
    by_knn = knn.decision(rows, np.ones(3))
    by_max = fuse(rows, "max")
    assert np.argsort(-by_knn).tolist() == np.argsort(-by_max).tolist()


def test_reciprocal_rank_fusion_uses_ranks_and_not_values():
    """Rescaling one reference's similarities cannot change an RRF score, but changes SUM."""
    T = S.copy()
    T[:, 0] = T[:, 0] * 0.5 + 0.25           # monotone in that column, so its ranking is kept
    assert fuse(S, "rrf").tolist() == pytest.approx(fuse(T, "rrf").tolist())
    assert fuse(S, "sum").tolist() != pytest.approx(fuse(T, "sum").tolist())
    # rank 1 in every reference is the largest possible score
    best = fuse(np.array([[1.0, 1.0], [0.1, 0.1]]), "rrf", rrf_k=60.0)
    assert best[0] == pytest.approx(2.0 / 61.0) and best[1] == pytest.approx(2.0 / 62.0)
    # the chapter's variant counts only the references that put the molecule near the top
    few = fuse(S, "rrf", top_fraction=0.34)
    assert few[0] > 0 and few.shape == (3,)


def test_a_molecule_is_never_its_own_reference():
    """Without this every rule with a maximum in it returns the self-similarity."""
    K = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.3], [0.2, 0.3, 1.0]])
    assert fuse(K, "max").tolist() == pytest.approx([1.0, 1.0, 1.0])     # meaningless
    assert fuse(K, "max", training=True).tolist() == pytest.approx([0.8, 0.8, 0.3])
    assert fuse(K, "sum", training=True).tolist() == pytest.approx([1.0, 1.1, 0.5])
    assert fuse(K, "med", training=True).tolist() == pytest.approx([0.5, 0.55, 0.25])
    with pytest.raises(ValueError):
        fuse(S, "max", training=True)        # not square, so there is no diagonal to drop


def test_a_molecule_comparable_with_nothing_scores_missing():
    """Not zero: a row of missing values must not land anywhere in particular in the ranking."""
    T = np.array([[0.5, 0.4], [np.nan, np.nan], [0.1, 0.2]])
    for rule in FUSION_RULES:
        out = fuse(T, rule)
        assert np.isnan(out[1]), rule
        assert np.isfinite(out[0]) and np.isfinite(out[2]), rule
    # a partly missing row uses what it has
    assert fuse(np.array([[0.5, np.nan]]), "max")[0] == pytest.approx(0.5)
    assert fuse(np.array([[0.5, np.nan]]), "sum")[0] == pytest.approx(0.5)


def test_the_scorer_calibrates_its_offset_on_the_references():
    """nu of the reference actives fall outside, which is what nu means for the 1-SVM too."""
    rng = np.random.default_rng(0)
    # the references cluster about one direction; the outsiders about an orthogonal one, so
    # "unlike the actives" means something here rather than merely "another random draw"
    axis = np.zeros(6)
    axis[0] = 1.0
    X = axis + 0.25 * rng.normal(size=(40, 6))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    K = X @ X.T
    sc = make("max", nu=0.2).fit(K)
    assert sc.n_ref_ == 40 and not sc.is_linear
    inside = sc.decision(K, training=True)
    assert abs(float(np.mean(inside < 0)) - 0.2) <= 0.05
    other = np.zeros(6)
    other[3] = 1.0
    far = other + 0.25 * rng.normal(size=(5, 6))
    far /= np.linalg.norm(far, axis=1, keepdims=True)
    assert float(np.mean(sc.decision(far @ X.T))) < float(np.mean(inside))
    assert float(np.mean(sc.decision(far @ X.T) < 0)) == 1.0      # all of them outside
    with pytest.raises(ValueError):
        FusionScorer().fit(np.ones((1, 1)))
    with pytest.raises(ValueError):
        make("nonsense")
    with pytest.raises(ValueError):
        fuse(S, "nonsense")
