"""The co-field around a generic residue (plan item PP5).

* The frame: the nitrogen on the x axis, the carbonyl carbon in the xy plane, the virtual beta
  carbon out of that plane at a bond length from the alpha carbon; the plane of the figure is
  spanned by two orthonormal directions through the alpha carbon.
* The co-field is the weighted sum of the normalised fields, the dual coefficients summed over
  the substitutions of one position.
* The error factor is exact: the coefficients of a fit on noisy targets move by G times the
  noise, as the bordered system of the fit says.
* The null keeps every position in its own domain.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "benchmarks"), os.path.join(ROOT, "examples")):
    sys.path.insert(0, p)

pytest.importorskip("rdkit")

import run_site_cofield as R                                            # noqa: E402
from cmfqsar.model import krr_fit                                       # noqa: E402


def test_the_frame():
    at = R.frame_atoms()
    assert np.allclose(at["CA"], 0.0)
    assert at["N"][1] == 0.0 and at["N"][2] == 0.0
    assert at["C"][2] == 0.0
    assert np.linalg.norm(at["CB"]) == pytest.approx(1.53)
    assert abs(at["CB"][2]) > 1.0
    e1, e2 = R.plane_basis()
    assert e1 @ e2 == pytest.approx(0.0, abs=1e-12)
    assert np.linalg.norm(e1) == pytest.approx(1.0) and np.linalg.norm(e2) == pytest.approx(1.0)
    pts, ax = R.plane_points(half=2.0, spacing=1.0)
    assert len(pts) == len(ax) ** 2
    # the nitrogen lies in the plane of the figure
    n = at["N"]
    assert np.linalg.norm(n - (n @ e1) * e1 - (n @ e2) * e2) < 1e-12


def test_the_cofield_is_a_weighted_sum_of_fields():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3, 7)).astype(np.float32)
    pos = np.array([0, 0, 2, 1])
    a = np.array([1.0, 2.0, -1.0, 0.5])
    C = R.cofields(a, pos, X, w=0.5)
    expect = 0.1 * (3.0 * X[:, 0] + 0.5 * X[:, 1] - 1.0 * X[:, 2])
    assert np.allclose(C, expect, atol=1e-6)


def test_the_error_factor_is_exact():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(30, 4))
    K = np.exp(-((A[:, None] - A[None]) ** 2).sum(-1) / 8.0)
    y = rng.normal(size=30)
    e = rng.normal(size=30) * 0.1
    G = R.dual_factor(K, 0.3)
    a0, _ = krr_fit(K, y, 0.3)
    a1, _ = krr_fit(K, y + e, 0.3)
    assert np.allclose(a1 - a0, G @ e, atol=1e-10)


def test_the_null_keeps_every_position_in_its_domain():
    keys = [("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2)]
    perm = R.within_domain_permutation(keys, np.random.default_rng(3))
    assert sorted(perm.tolist()) == list(range(5))
    for i, j in enumerate(perm):
        assert keys[i][0] == keys[j][0]


def test_the_comparison_of_two_maps():
    rng = np.random.default_rng(2)
    c = rng.normal(size=(5, 50))
    cover = np.ones_like(c, bool)
    keep = np.abs(c) > 0.5
    same = R.compare(c, c, cover, keep)
    assert np.allclose(same["r"], 1.0) and np.allclose(same["same_sign"], 1.0)
    turned = R.compare(c, -c, cover, keep)
    assert np.allclose(turned["r"], -1.0) and np.allclose(turned["same_sign"], 0.0)
