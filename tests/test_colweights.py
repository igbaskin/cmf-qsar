"""Tests for the weighted distance over the channels (plan item M15, second half).

The weights are a measurement choice and live in `benchmarks/run_colweights.py`, so these tests
hold that script to what its table assumes: the column rule agrees with the library's block rule
when a block is one column, a weight of zero really removes a column, the kernel the script builds
is the library's kernel with a scale, and no weight is ever computed from a test row.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from cmfqsar.invariant import (align_block_weights, autocorrelation_kernel,  # noqa: E402
                               expand_block_weights)

import run_colweights as CW  # noqa: E402


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    n, p = 40, 24
    Z = rng.normal(size=(n, p))
    # one column that carries the target, one that is noise, one that is constant
    y = 3.0 * Z[:, 5] + 0.1 * rng.normal(size=n)
    Z[:, 7] = 2.5                                  # constant: no alignment, no weight
    blocks = [6] * 4
    return Z, y, blocks


def test_the_column_rule_is_the_block_rule_with_one_column_blocks(world):
    """`alignment_scores` is what the library's alignment gives a block of one column."""
    Z, y, _ = world
    mine = CW.alignment_scores(Z, y)
    theirs = align_block_weights(Z, y, [1] * Z.shape[1], mode="align") ** 2
    live = mine > 0
    assert live.sum() > 5
    # both are the same quantity up to one overall factor, so their ratio is one number
    ratio = theirs[live] / mine[live]
    assert np.allclose(ratio, ratio[0], rtol=1e-9)
    assert theirs[~live] == pytest.approx(0.0)


def test_the_carrying_column_is_ranked_first_and_the_constant_one_last(world):
    Z, y, _ = world
    h = CW.alignment_scores(Z, y)
    assert int(np.argmax(h)) == 5
    assert h[7] == 0.0                             # a constant column cannot align with anything


def test_the_arms_return_the_weights_they_promise(world):
    Z, y, blocks = world
    p = Z.shape[1]
    assert np.allclose(CW.weights_for(CW.ARM_EQUAL, Z, y, blocks), np.ones(p))
    for share in CW.BEST_SHARES:
        w = CW.weights_for(CW.ARM_BEST, Z, y, blocks, share=share)
        assert set(np.unique(w)) <= {0.0, 1.0}
        assert int(np.sum(w > 0)) == max(1, int(round(share * p)))
        assert w[int(np.argmax(CW.alignment_scores(Z, y)))] == 1.0
    w = CW.weights_for(CW.ARM_BLOCK, Z, y, blocks)
    assert len(w) == p
    # a block weight is one number per block, so it is constant inside a block
    for b in range(len(blocks)):
        sl = slice(b * blocks[b], (b + 1) * blocks[b])
        assert np.allclose(w[sl], w[sl][0])
    assert np.all(CW.weights_for(CW.ARM_FOREST, Z, y, blocks) >= 0)


def test_a_zero_weight_removes_a_column(world):
    """The `best columns` arm has to mean what it says: the other columns are gone."""
    Z, y, blocks = world
    w = np.ones(Z.shape[1])
    w[3] = 0.0
    rows = np.arange(len(y))
    with_zero = CW.cosine_blocks(Z * w, rows, rows)
    without = CW.cosine_blocks(np.delete(Z, 3, axis=1), rows, rows)
    assert np.allclose(with_zero, without, atol=1e-12)


def test_the_script_builds_the_librarys_kernel(world):
    """`cosine_blocks` is `autocorrelation_kernel` with the weights as the scale."""
    Z, y, blocks = world
    w = CW.weights_for(CW.ARM_COLUMN, Z, y, blocks)
    rows = np.arange(len(y))
    mine = CW.cosine_blocks(Z * w, rows, rows)
    theirs = autocorrelation_kernel(Z, scale=w, n_bins=1, blocks=[Z.shape[1]])
    assert np.allclose(mine, theirs, atol=1e-12)
    # and the test block is the rectangle it should be
    te, tr = rows[:7], rows[7:]
    assert CW.cosine_blocks(Z * w, te, tr).shape == (7, len(rows) - 7)
    assert np.allclose(CW.cosine_blocks(Z * w, te, tr), mine[np.ix_(te, tr)], atol=1e-12)


def test_no_weight_is_computed_from_a_row_outside_the_training_part(world):
    """Appending rows that the rule is not shown must not change its weights."""
    Z, y, blocks = world
    rng = np.random.default_rng(1)
    extra = rng.normal(size=(15, Z.shape[1]))
    for arm in (CW.ARM_BLOCK, CW.ARM_COLUMN, CW.ARM_BEST):
        a = CW.weights_for(arm, Z, y, blocks, share=0.3)
        b = CW.weights_for(arm, np.vstack([Z, extra])[:len(y)], y, blocks, share=0.3)
        assert np.allclose(a, b, rtol=1e-12)


def test_the_block_weights_expand_as_the_library_expands_them(world):
    """A block weight reaches its columns through `expand_block_weights`, not by hand."""
    Z, y, blocks = world
    w = align_block_weights(Z, y, blocks, mode="align")
    assert np.allclose(CW.weights_for(CW.ARM_BLOCK, Z, y, blocks),
                       expand_block_weights(w, blocks))


def test_every_arm_has_its_settings_declared():
    """The settings an arm chooses between are declared once, so the stored result can be read."""
    assert CW.candidates(CW.ARM_BEST) == list(CW.BEST_SHARES)
    for arm in CW.ARMS:
        if arm != CW.ARM_BEST:
            assert CW.candidates(arm) == [0.0]
    assert CW.ARM_EQUAL == CW.ARMS[0]              # the reference is the first arm
