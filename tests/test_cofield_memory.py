"""The neighbour list of a co-field evaluation is bounded by pairs, not by points.

`cofields.evaluate_gaussians` reads the grid in chunks and builds the neighbour list of a chunk
in one array.  The chunk counts points, and what costs memory is points times the centres each
one sees: a few hundred for the co-field of a molecule, over a hundred thousand for the co-field
of a periodic model, whose basis is the image-expanded training table.  On a cube over one cell
that difference is 17 GB in a single allocation, which is enough to take a machine down, so the
chunk is cut to keep each neighbour list under :data:`cofields.MAX_PAIRS`.

What the tests check: that the cut happens where it is needed, that it does not happen where it
is not, and that chunking changes no value -- it is only the order in which one sum is taken.
"""
import os
import sys

import numpy as np
import pytest
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cofields import MAX_PAIRS, evaluate_gaussians, points_per_chunk  # noqa: E402


def _dense_sum(points, centres, coeffs, widths):
    """The same sum written out, with nothing dropped and no neighbour list."""
    d2 = ((np.asarray(points, float)[:, None, :]
           - np.asarray(centres, float)[None, :, :]) ** 2).sum(-1)
    return (np.asarray(coeffs, float) * np.exp(-0.5 * np.asarray(widths, float) * d2)).sum(axis=1)


def _grid(n=6, a=4.0):
    ax = np.linspace(0.0, a, n)
    return np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)


def test_a_sparse_basis_keeps_the_chunk_it_was_given():
    """A molecule's co-field: a few hundred Gaussians, and nothing to cut."""
    rng = np.random.RandomState(0)
    centres = rng.randn(200, 3) * 2.0
    tree = cKDTree(centres)
    assert points_per_chunk(_grid(), tree, 6.0, 20000) == 20000


def test_a_dense_basis_cuts_the_chunk():
    """The periodic case: many centres in a small box, so one point sees a great many."""
    rng = np.random.RandomState(0)
    centres = rng.rand(60000, 3) * 4.0          # 60,000 centres in a 4 A box
    tree = cKDTree(centres)
    points = _grid()
    chunk = points_per_chunk(points, tree, 6.0, 20000)
    assert chunk < 20000
    seen = int(np.mean(tree.query_ball_point(points[:8], 6.0, return_length=True)))
    assert chunk * seen <= MAX_PAIRS            # which is what the cut is for
    assert chunk >= 1                           # and it never reaches zero


def test_the_chunk_never_falls_below_one_point():
    """Even if a single point saw every centre of a huge basis, a chunk holds one point."""
    rng = np.random.RandomState(1)
    centres = rng.rand(5000, 3) * 0.5
    tree = cKDTree(centres)
    assert points_per_chunk(_grid(), tree, 10.0, 20000, sample=4) >= 1


def test_chunking_changes_no_value():
    """The sum is the same however it is split, to the tolerance of the display cutoff."""
    rng = np.random.RandomState(2)
    centres = rng.rand(4000, 3) * 3.0
    coeffs = rng.randn(4000) * 0.01
    widths = np.full(4000, 1.2)
    points = _grid(5)
    whole = evaluate_gaussians(points, centres, coeffs, widths, chunk=20000)
    in_bits = evaluate_gaussians(points, centres, coeffs, widths, chunk=7)
    assert np.allclose(whole, in_bits, rtol=1e-12, atol=1e-14)
    assert np.allclose(whole, _dense_sum(points, centres, coeffs, widths),
                       atol=1e-4 * max(1.0, float(np.abs(whole).max())))


def test_an_empty_grid_is_allowed():
    centres = np.zeros((3, 3))
    tree = cKDTree(centres)
    assert points_per_chunk(np.zeros((0, 3)), tree, 5.0, 20000) >= 1
    assert evaluate_gaussians(np.zeros((0, 3)), centres, np.ones(3), np.ones(3)).shape == (0,)


@pytest.mark.parametrize("n_centres", [10, 1000])
def test_the_measured_cost_of_a_chunk_stays_under_the_cap(n_centres):
    rng = np.random.RandomState(3)
    centres = rng.rand(n_centres, 3) * 2.0
    tree = cKDTree(centres)
    points = _grid(8)
    chunk = points_per_chunk(points, tree, 5.0, 20000)
    pairs = chunk * int(np.mean(tree.query_ball_point(points[:8], 5.0, return_length=True)))
    assert pairs <= MAX_PAIRS or chunk == 1
