"""Conceptual-DFT fields on a grid, and what atom-centred Gaussians can make of them."""
import os

import numpy as np
import pytest

from cmfqsar.cdft import (BOHR, Cube, cdft_fields, fit_field, overlap_kernel,
                          read_cube, valence_window)
from cmfqsar.probes import basis_widths


def write_cube(path, values, origin, step, numbers, positions):
    """A Gaussian cube, in bohr, as ``orca_plot`` writes them."""
    n = values.shape
    with open(path, "w", newline="\n") as fh:
        fh.write("test cube\nmade by the tests\n")
        fh.write(f"{len(numbers):5d} {origin[0]:12.6f} {origin[1]:12.6f} {origin[2]:12.6f}\n")
        for k in range(3):
            v = [0.0, 0.0, 0.0]
            v[k] = step
            fh.write(f"{n[k]:5d} {v[0]:12.6f} {v[1]:12.6f} {v[2]:12.6f}\n")
        for z, p in zip(numbers, positions):
            fh.write(f"{z:5d} {float(z):12.6f} {p[0]:12.6f} {p[1]:12.6f} {p[2]:12.6f}\n")
        flat = values.reshape(-1)
        for i in range(0, len(flat), 6):
            fh.write("".join(f"{v:13.5E}" for v in flat[i:i + 6]) + "\n")


def gaussian_cube(tmp_path, coeffs, widths, centres_bohr, n=40, step=0.5, name="g.cube"):
    """A cube holding an exact sum of atom-centred Gaussians, so a fit has a right answer."""
    origin = np.array([-(n - 1) * step / 2.0] * 3)
    axes = [origin[k] + np.arange(n) * step for k in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1)          # bohr
    vals = np.zeros((n, n, n))
    for c_row, centre in zip(coeffs, centres_bohr):
        d2 = ((P - centre) ** 2).sum(-1) * BOHR ** 2             # the fit works in angstrom
        for c, b in zip(c_row, widths):
            vals += c * np.exp(-0.5 * b * d2)
    vals *= BOHR ** 3                                            # the reader divides this out
    path = os.path.join(tmp_path, name)
    write_cube(path, vals, origin, step, [6] * len(centres_bohr), centres_bohr)
    return path


def test_reading_a_cube_gives_angstrom_and_the_right_integral(tmp_path):
    n, step = 30, 0.4
    origin = np.array([-6.0, -6.0, -6.0])
    axes = [origin[k] + np.arange(n) * step for k in range(3)]
    P = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    rho = np.exp(-0.5 * (P ** 2).sum(-1))                        # integrates to (2 pi)^{3/2} bohr^3
    path = os.path.join(tmp_path, "rho.cube")
    write_cube(path, rho, origin, step, [8], [[0.0, 0.0, 0.0]])
    cube = read_cube(path)
    assert cube.values.shape == (n, n, n)
    assert cube.symbols == ["O"]
    np.testing.assert_allclose(cube.positions, [[0.0, 0.0, 0.0]], atol=1e-9)
    np.testing.assert_allclose(cube.vectors[0][0], step * BOHR, rtol=1e-12)
    # the reader rescales values to e/A^3, so the integral is unchanged by the unit change
    assert cube.integral() == pytest.approx((2 * np.pi) ** 1.5, rel=1e-3)
    assert cube.points().shape == (n ** 3, 3)


def test_the_fit_is_exact_on_a_field_it_can_represent(tmp_path):
    """A sum of atom-centred Gaussians on the shared widths is recovered to machine accuracy.

    This is what makes the negative result on real conceptual-DFT fields mean something: the
    fitter is not what fails there.
    """
    widths = basis_widths(4)
    centres = np.array([[0.0, 0.0, 0.0], [2.6, 0.0, 0.0], [1.3, 2.2, 0.0]])    # bohr
    rng = np.random.default_rng(0)
    coeffs = rng.normal(size=(3, 4))
    path = gaussian_cube(tmp_path, coeffs, widths, centres, n=48, step=0.45)
    cube = read_cube(path)
    fit = fit_field(cube.values, cube, n_terms=4, ridge=1e-10)
    assert fit.rel_error < 0.02
    assert fit.r2 > 0.999
    np.testing.assert_allclose(fit.coeffs, coeffs, atol=0.05)
    integral = float(np.sum(coeffs @ (2 * np.pi / widths) ** 1.5))
    assert fit.condensed().sum() == pytest.approx(integral, rel=0.05)


def test_conceptual_dft_fields_are_finite_differences_on_one_grid(tmp_path):
    widths, centres = basis_widths(2), np.array([[0.0, 0.0, 0.0]])
    n, step = 24, 0.6
    cubes = {}
    for tag, scale in (("n", 1.0), ("np1", 1.3), ("nm1", 0.8)):
        path = gaussian_cube(tmp_path, scale * np.ones((1, 2)), widths, centres, n=n, step=step,
                             name=f"{tag}.cube")
        cubes[tag] = read_cube(path)
    F = cdft_fields(cubes["n"], cubes["np1"], cubes["nm1"])
    # a cube carries five significant digits, so a difference of two of them carries fewer
    np.testing.assert_allclose(F["fukui_plus"], 0.3 * cubes["n"].values, rtol=1e-4)
    np.testing.assert_allclose(F["fukui_minus"], 0.2 * cubes["n"].values, rtol=1e-4)
    np.testing.assert_allclose(F["dual"], F["fukui_plus"] - F["fukui_minus"], rtol=1e-12)
    np.testing.assert_allclose(F["fukui_zero"], 0.25 * cubes["n"].values, rtol=1e-4)


def test_cubes_on_different_grids_are_refused(tmp_path):
    widths, centres = basis_widths(2), np.array([[0.0, 0.0, 0.0]])
    a = read_cube(gaussian_cube(tmp_path, np.ones((1, 2)), widths, centres, n=20, step=0.5,
                                name="a.cube"))
    b = read_cube(gaussian_cube(tmp_path, np.ones((1, 2)), widths, centres, n=20, step=0.6,
                                name="b.cube"))
    assert not a.same_grid_as(b)
    with pytest.raises(ValueError):
        cdft_fields(a, b)


def test_the_valence_window_closes_on_the_cores():
    rho = np.array([1e-4, 1e-3, 1e-2, 0.2, 1.0, 50.0]) / BOHR ** 3      # e/A^3, as the reader gives
    w = valence_window(rho)
    assert w[0] > 0.999 and w[2] > 0.99      # the valence shell is kept
    assert w[3] == pytest.approx(0.5)        # half closed at RHO_CORE
    assert w[5] < 1e-4                       # a first-row core is gone
    assert np.all(np.diff(w) < 0)


def test_the_overlap_kernel_agrees_with_quadrature():
    """The closed form against a numerical integral of the same two fields.

    This is the formula the probe branch of the kernel evaluates; if it is right, a fitted field
    can be trusted to enter a model as the field it was fitted to.
    """
    widths = np.geomspace(10.0, 0.5, 3)
    rng = np.random.default_rng(7)
    pos = [np.array([[0.0, 0.0, 0.0], [1.4, 0.3, 0.0]]),
           np.array([[0.6, -0.4, 0.2], [2.0, 0.9, -0.3]])]
    coeffs = [rng.normal(size=(2, 3)), rng.normal(size=(2, 3))]
    K = overlap_kernel(coeffs, pos, widths)

    step = 0.08
    axis = np.arange(-7.0, 7.0 + step, step)
    P = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)

    def field(c, centres):
        out = np.zeros(len(P))
        for row, centre in zip(c, centres):
            d2 = ((P - centre) ** 2).sum(-1)
            for w, b in zip(row, widths):
                out += w * np.exp(-0.5 * b * d2)
        return out

    FA, FB = field(coeffs[0], pos[0]), field(coeffs[1], pos[1])
    dV = step ** 3
    assert K[0, 1] == pytest.approx(float(FA @ FB) * dV, rel=2e-3)
    assert K[0, 0] == pytest.approx(float(FA @ FA) * dV, rel=2e-3)
    assert K[1, 1] == pytest.approx(float(FB @ FB) * dV, rel=2e-3)
    np.testing.assert_allclose(K, K.T, rtol=1e-12)
