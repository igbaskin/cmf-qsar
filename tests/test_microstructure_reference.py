"""The reference of the microstructure benchmark that needs no field (stage shape of
benchmarks/run_microstructure.py): the phase fraction and the correlation lengths along the axes."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

RM = pytest.importorskip("run_microstructure")


def test_stripes_along_x_are_long_along_x_and_short_across():
    grid = 31
    cell = np.zeros((grid, grid), np.uint8)
    cell[:, ::4] = 1                     # columns of phase 1: constant along axis 0
    cell[:, 1::4] = 1
    L = RM.correlation_lengths(cell[None])[0]
    assert L[0] == grid // 2             # never falls below 1/e: the half-grid limit
    assert 0.5 < L[1] < 3.0              # across the stripes it falls within a few voxels


def test_a_length_is_finite_and_bounded_on_random_cells():
    rng = np.random.default_rng(0)
    cells = np.stack([RM.generate(2, 21, 0.4, 3.0, 2.0, np.eye(2), rng) for _ in range(5)])
    L = RM.correlation_lengths(cells)
    assert np.all(np.isfinite(L)) and np.all(L >= 0) and np.all(L <= 21 // 2)
    Z = RM.shape_numbers(cells)
    assert Z.shape == (5, 3)
    assert np.allclose(Z[:, 0], cells.reshape(5, -1).mean(axis=1))


def test_the_reference_recovers_a_law_in_its_numbers():
    rng = np.random.RandomState(1)
    Z = rng.uniform(0, 1, (80, 3))
    y = 2.0 + Z[:, 0] - 0.5 * Z[:, 1] ** 2
    tr, te = np.arange(60), np.arange(60, 80)
    r = RM.fit_shape(Z, y, tr, te)
    assert np.sqrt(np.mean((r["pred"] - y[te]) ** 2)) < 0.02
