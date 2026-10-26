"""Atom functions with a direction (plan item D1).

Everything here is a correctness test, because the whole case for the extension is that the
overlaps stay closed-form: if the formulas are wrong, the speed and the algebra are worth nothing.
Every one of the four is checked against numerical quadrature on a fine grid, which is how they
were derived.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.anisotropic import (AnisotropicFit, PER_WIDTH, basis_size, design,  # noqa: E402
                                 field_overlap, fit_field, overlap_kernel, pair_overlaps)


def _grid(limit=9.0, n=140):
    g = np.linspace(-limit, limit, n)
    X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
    pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    return pts, float(g[1] - g[0]) ** 3


def _basis_values(pts, centre, width, which):
    d = pts - np.asarray(centre, float)
    g = np.exp(-0.5 * width * np.einsum("nu,nu->n", d, d))
    return g if which == 0 else g * d[:, which - 1]


# ------------------------------------------------------------------ the overlaps themselves


def test_every_overlap_matches_quadrature():
    pts, dV = _grid()
    rng = np.random.default_rng(0)
    A = rng.normal(scale=0.5, size=3)
    B = A + rng.normal(scale=0.8, size=3)
    a, b = 1.3, 0.7
    O = pair_overlaps(A[None, :], B[None, :], a, b)
    assert O.shape == (1, 1, PER_WIDTH, PER_WIDTH)
    for i in range(PER_WIDTH):
        for j in range(PER_WIDTH):
            fa = _basis_values(pts, A, a, i)
            fb = _basis_values(pts, B, b, j)
            numeric = float(np.sum(fa * fb) * dV)
            assert O[0, 0, i, j] == pytest.approx(numeric, rel=2e-5, abs=1e-8), (i, j)


def test_the_s_block_is_the_isotropic_overlap_the_program_already_uses():
    """The new formula has to contain the old one, or every published number would move."""
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 3))
    B = rng.normal(size=(3, 3))
    a, b = 0.8, 1.1
    O = pair_overlaps(A, B, a, b)
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    old = np.exp(-(a * b) / (2.0 * (a + b)) * d2) * (2.0 * np.pi / (a + b)) ** 1.5
    assert O[:, :, 0, 0] == pytest.approx(old, rel=1e-12)
    assert pair_overlaps(A, B, a, b, p_type=False)[:, :, 0, 0] == pytest.approx(old, rel=1e-12)


def test_a_p_function_overlaps_itself_positively_and_its_mirror_negatively():
    A = np.zeros((1, 3))
    O = pair_overlaps(A, A, 1.0, 1.0)
    for u in range(3):
        assert O[0, 0, u + 1, u + 1] > 0
        assert O[0, 0, u + 1, 0] == pytest.approx(0.0, abs=1e-12)   # odd against even, same centre
    for u in range(3):
        for v in range(3):
            if u != v:
                assert O[0, 0, u + 1, v + 1] == pytest.approx(0.0, abs=1e-12)


def test_the_overlap_is_symmetric_under_swapping_the_two_sides():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(2, 3))
    B = rng.normal(size=(3, 3))
    a, b = 0.6, 1.4
    O = pair_overlaps(A, B, a, b)
    O_swapped = pair_overlaps(B, A, b, a)
    assert O == pytest.approx(np.transpose(O_swapped, (1, 0, 3, 2)), rel=1e-12, abs=1e-14)


def test_a_field_overlap_matches_quadrature():
    pts, dV = _grid(limit=8.0, n=130)
    rng = np.random.default_rng(3)
    widths = np.array([0.5, 1.5])
    pos_a = rng.normal(scale=0.7, size=(3, 3))
    pos_b = rng.normal(scale=0.7, size=(2, 3))
    ca = rng.normal(size=(3, 2, PER_WIDTH))
    cb = rng.normal(size=(2, 2, PER_WIDTH))
    closed = field_overlap(ca, pos_a, cb, pos_b, widths)
    Ba = design(pts, pos_a, widths)
    Bb = design(pts, pos_b, widths)
    fa = np.einsum("natk,atk->n", Ba, ca)
    fb = np.einsum("natk,atk->n", Bb, cb)
    assert closed == pytest.approx(float(np.sum(fa * fb) * dV), rel=1e-4)


def test_the_kernel_is_symmetric_and_positive_on_the_diagonal():
    rng = np.random.default_rng(4)
    widths = np.array([0.7, 2.0])
    pos = [rng.normal(scale=0.8, size=(4, 3)) for _ in range(5)]
    coef = [rng.normal(size=(4, 2, PER_WIDTH)) for _ in range(5)]
    K = overlap_kernel(coef, pos, widths)
    assert np.allclose(K, K.T)
    assert np.all(np.diag(K) > 0)                 # a field's overlap with itself is its norm
    assert np.all(np.linalg.eigvalsh(K) > -1e-8)  # and the matrix is a Gram matrix


# -------------------------------------------------------------------------- the basis itself


def test_the_basis_has_one_s_and_three_p_per_atom_per_width():
    assert basis_size(4, p_type=True) == 16
    assert basis_size(4, p_type=False) == 4
    pts = np.zeros((7, 3))
    B = design(pts, np.zeros((3, 3)), np.array([0.5, 1.0]))
    assert B.shape == (7, 3, 2, PER_WIDTH)
    assert design(pts, np.zeros((3, 3)), np.array([0.5]), p_type=False).shape == (7, 3, 1, 1)


def test_a_p_function_changes_sign_across_its_own_centre():
    """The property the whole item rests on: an isotropic Gaussian cannot do this."""
    centre = np.zeros((1, 3))
    pts = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    B = design(pts, centre, np.array([1.0]))
    assert B[0, 0, 0, 1] < 0 < B[1, 0, 0, 1]      # px
    assert B[0, 0, 0, 0] == pytest.approx(B[1, 0, 0, 0])   # s is the same on both sides


# ------------------------------------------------------------------------------- the fitting


def test_it_recovers_a_field_that_is_made_of_this_basis():
    rng = np.random.default_rng(5)
    widths = np.array([0.6, 1.8])
    pos = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    truth = rng.normal(size=(2, 2, PER_WIDTH))
    pts, _dV = _grid(limit=6.0, n=70)
    values = np.einsum("natk,atk->n", design(pts, pos, widths), truth)
    got = fit_field(values, pts, pos, widths, ridge=1e-10)
    assert got.rel_error < 1e-6
    assert got.coeffs == pytest.approx(truth, rel=1e-3, abs=1e-4)


def test_it_fits_a_lobe_an_isotropic_basis_cannot():
    """A dipole-shaped target: positive on one side of the atom, negative on the other."""
    widths = np.array([1.0])
    pos = np.zeros((1, 3))
    pts, _dV = _grid(limit=5.0, n=60)
    target = pts[:, 0] * np.exp(-0.5 * np.einsum("nu,nu->n", pts, pts))
    aniso = fit_field(target, pts, pos, widths, ridge=1e-10)
    iso = fit_field(target, pts, pos, widths, p_type=False, ridge=1e-10)
    assert aniso.rel_error < 1e-6
    assert iso.rel_error > 0.99                   # the isotropic fit can only answer zero
    assert abs(aniso.dipoles()[0, 0]) > 1.0
    assert np.allclose(iso.dipoles(), 0.0)


def test_the_p_functions_integrate_to_nothing():
    """They move the field from one side of an atom to the other, not into or out of it."""
    rng = np.random.default_rng(6)
    widths = np.array([0.8, 2.2])
    coeffs = rng.normal(size=(3, 2, PER_WIDTH))
    got = AnisotropicFit(coeffs=coeffs, widths=widths, p_type=True)
    s_only = AnisotropicFit(coeffs=np.concatenate(
        [coeffs[:, :, :1], np.zeros_like(coeffs[:, :, 1:])], axis=2), widths=widths, p_type=True)
    assert got.condensed() == pytest.approx(s_only.condensed())
    assert got.n_coefficients == 3 * 2 * PER_WIDTH


def test_the_dipole_points_where_the_lobe_is():
    widths = np.array([1.0])
    coeffs = np.zeros((1, 1, PER_WIDTH))
    coeffs[0, 0, 2] = 1.0                       # a py lobe
    got = AnisotropicFit(coeffs=coeffs, widths=widths, p_type=True)
    d = got.dipoles()[0]
    assert d[1] > 0 and abs(d[0]) < 1e-12 and abs(d[2]) < 1e-12


def test_the_fitted_field_can_be_evaluated_back_on_its_grid():
    rng = np.random.default_rng(7)
    widths = np.array([1.2])
    pos = rng.normal(size=(2, 3))
    pts, _dV = _grid(limit=4.0, n=40)
    truth = rng.normal(size=(2, 1, PER_WIDTH))
    values = np.einsum("natk,atk->n", design(pts, pos, widths), truth)
    got = fit_field(values, pts, pos, widths, ridge=1e-12)
    assert got.evaluate(pts, pos) == pytest.approx(values, rel=1e-3, abs=1e-6)


def test_a_window_restricts_what_is_fitted():
    rng = np.random.default_rng(8)
    widths = np.array([1.0])
    pos = np.zeros((1, 3))
    pts, _dV = _grid(limit=4.0, n=40)
    values = rng.normal(size=len(pts))
    window = (np.abs(pts[:, 0]) < 1.0).astype(float)
    a = fit_field(values, pts, pos, widths, window=window)
    b = fit_field(values, pts, pos, widths)
    assert not np.allclose(a.coeffs, b.coeffs)


# -------------------------------------------- it has to contain the isotropic path exactly


def test_the_isotropic_kernel_is_the_one_cdft_already_computes():
    """The s-only path must reproduce cmfqsar.cdft bit for bit, or the comparison is not one."""
    from cmfqsar.cdft import overlap_kernel as cdft_kernel
    rng = np.random.default_rng(9)
    widths = np.array([0.4, 1.1, 3.0])
    pos = [rng.normal(scale=0.9, size=(4, 3)) for _ in range(4)]
    flat = [rng.normal(size=(4, 3)) for _ in range(4)]           # (atoms, widths)
    mine = overlap_kernel([c[:, :, None] for c in flat], pos, widths, p_type=False)
    theirs = cdft_kernel(flat, pos, widths)
    assert mine == pytest.approx(theirs, rel=1e-12)


def test_the_isotropic_fit_is_the_one_cdft_already_computes():
    from cmfqsar.cdft import Cube
    from cmfqsar.cdft import fit_field as cdft_fit
    rng = np.random.default_rng(10)
    n = 24
    axis = np.linspace(-4.0, 4.0, n)
    cube = Cube(values=rng.normal(size=(n, n, n)),
                origin=np.array([axis[0]] * 3),
                vectors=np.eye(3) * float(axis[1] - axis[0]),
                numbers=np.array([6, 8]),
                positions=np.array([[0.0, 0.0, 0.0], [1.3, 0.2, 0.0]]))
    widths = np.array([0.5, 2.0])
    theirs = cdft_fit(cube.values, cube, widths=widths, ridge=1e-6)
    mine = fit_field(cube.values, cube.points(), cube.positions, widths, p_type=False,
                     ridge=1e-6)
    assert mine.rel_error == pytest.approx(theirs.rel_error, rel=1e-9)
    assert mine.coeffs[:, :, 0] == pytest.approx(theirs.coeffs, rel=1e-6, abs=1e-9)
    assert mine.condensed() == pytest.approx(theirs.condensed(), rel=1e-6)
