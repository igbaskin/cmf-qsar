"""Fields and their gradients at anchored points of a shared frame (plan item EC3).

A field is a function of space and a kernel compares two of them everywhere.  These descriptors
read one place instead, which is what a Hammett constant or a metal site is about.  Three things
are tested: that the value is the field the rest of the program would draw there, that the
gradient is the derivative of that value and not something else, and that the Coulomb source is
the physical point-charge field rather than another Gaussian.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.cofields import Grid, field_volume
from cmfqsar.descriptors import (ANCHOR_KINDS, ANCHOR_SOURCES, anchor_names, anchored_matrix,
                                 anchored_row, coulomb_at_points, field_at_points,
                                 field_gradient_at_points)
from cmfqsar.fields import FieldSet, build_fields


def mol(smiles: str = "c1ccccc1N", seed: int = 1) -> Chem.Mol:
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def one(names=("electrostatic",), alpha: float = 1.0):
    fs = FieldSet.create(physchem=list(names), alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit")
    return fs, build_fields(mol(), fs, "m")


POINTS = np.array([[2.5, 0.0, 0.0], [0.0, 0.0, 2.0], [-1.5, 1.5, 0.5]])


# ------------------------------------------------------------------ the value

def test_the_value_is_the_field_the_program_draws_there():
    """The same number that `field_volume` puts on a grid point.

    Not to the last digit.  The routine that draws a field on a grid drops every Gaussian that
    has fallen below ``eps = 1e-5`` of its peak, which is worth it for a million grid points and
    makes the drawing an approximation.  The reading here sums every atom, so the two agree to
    about that tolerance per dropped atom, and the reading is the exact one.  The comparison is
    therefore on an absolute scale: at a point where the field itself is only 0.025, a few
    dropped atoms are 4e-5 of it.
    """
    fs, mf = one()
    for point in POINTS:
        grid = Grid(origin=point, spacing=1.0, shape=(1, 1, 1))
        drawn = field_volume(mf, fs.specs[0], 0, grid)[0, 0, 0]
        got = field_at_points(point[None, :], mf, fs.specs[0], 0)[0]
        assert got == pytest.approx(drawn, abs=2e-5)


def test_the_value_is_the_sum_over_every_atom():
    """Written out by hand, so that a cutoff or a missing atom would show."""
    fs, mf = one()
    want = 0.0
    for i in range(mf.n_atoms):
        d2 = float(np.sum((POINTS[0] - mf.coords[i]) ** 2))
        want += float(mf.weights[i, 0]) * np.exp(-0.5 * fs.specs[0].alpha * d2)
    assert field_at_points(POINTS[:1], mf, fs.specs[0], 0)[0] == pytest.approx(want, rel=1e-12)


def test_a_field_of_zero_weights_is_zero_everywhere():
    fs, mf = one()
    mf.weights = np.zeros_like(mf.weights)
    assert np.allclose(field_at_points(POINTS, mf, fs.specs[0], 0), 0.0)
    assert np.allclose(field_gradient_at_points(POINTS, mf, fs.specs[0], 0), 0.0)


# ------------------------------------------------------------------ the gradient

@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_gradient_is_the_derivative_of_the_value(axis):
    fs, mf = one(("electrostatic", "steric"))
    step = 1e-5
    for f in (0, 1):
        for point in POINTS:
            shift = np.zeros(3)
            shift[axis] = step
            hi = field_at_points((point + shift)[None, :], mf, fs.specs[f], f)[0]
            lo = field_at_points((point - shift)[None, :], mf, fs.specs[f], f)[0]
            got = field_gradient_at_points(point[None, :], mf, fs.specs[f], f)[0, axis]
            assert got == pytest.approx((hi - lo) / (2 * step), rel=1e-4, abs=1e-10)


def test_the_steric_field_uses_its_own_per_atom_widths():
    """The steric field's width is scaled by the atom radius, and the reading must follow it."""
    fs, mf = one(("steric",))
    wide = FieldSet.create(physchem=["steric"], alpha=4.0, charge_model="gasteiger",
                           param_scheme="rdkit")
    narrow = build_fields(mol(), wide, "m")
    a = field_at_points(POINTS[:1], mf, fs.specs[0], 0)[0]
    b = field_at_points(POINTS[:1], narrow, wide.specs[0], 0)[0]
    assert a > b > 0                                  # a wider field reaches further


# ------------------------------------------------------------------ the Coulomb source

def test_the_coulomb_potential_is_one_over_r():
    phi, field = coulomb_at_points(np.array([[3.0, 0.0, 0.0]]), np.zeros((1, 3)),
                                   np.array([1.0]))
    assert phi[0] == pytest.approx(1.0 / 3.0)
    # E = -grad phi = q r_hat / r^2, pointing away from a positive charge
    assert field[0] == pytest.approx([1.0 / 9.0, 0.0, 0.0])


def test_the_coulomb_field_is_the_derivative_of_the_potential():
    coords = np.array([[0.0, 0.0, 0.0], [1.4, 0.3, -0.2]])
    charges = np.array([0.3, -0.5])
    point = np.array([[2.0, 1.0, 0.5]])
    _phi, field = coulomb_at_points(point, coords, charges)
    step = 1e-6
    for axis in range(3):
        shift = np.zeros((1, 3))
        shift[0, axis] = step
        hi = coulomb_at_points(point + shift, coords, charges)[0][0]
        lo = coulomb_at_points(point - shift, coords, charges)[0][0]
        assert field[0, axis] == pytest.approx(-(hi - lo) / (2 * step), rel=1e-4, abs=1e-9)


def test_the_coulomb_potential_of_a_neutral_pair_falls_off_faster_than_one_charge():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    far = np.array([[20.0, 0.0, 0.0]])
    dipole = abs(coulomb_at_points(far, coords, np.array([1.0, -1.0]))[0][0])
    single = abs(coulomb_at_points(far, coords[:1], np.array([1.0]))[0][0])
    assert dipole < 0.1 * single


def test_a_point_on_an_atom_stays_finite():
    phi, field = coulomb_at_points(np.zeros((1, 3)), np.zeros((1, 3)), np.array([1.0]))
    assert np.all(np.isfinite(phi)) and np.all(np.isfinite(field))


# ------------------------------------------------------------------ the block

def test_the_names_say_the_place_the_field_and_the_reading():
    names = anchor_names(["electrostatic", "steric"], ["para", "face"], ("value", "gx"))
    assert names == ["at:para:electrostatic:value", "at:para:electrostatic:gx",
                     "at:para:steric:value", "at:para:steric:gx",
                     "at:face:electrostatic:value", "at:face:electrostatic:gx",
                     "at:face:steric:value", "at:face:steric:gx"]


def test_an_unknown_reading_is_refused():
    with pytest.raises(ValueError):
        anchor_names(["electrostatic"], ["p"], ("value", "curl"))
    fs, mf = one()
    with pytest.raises(ValueError):
        anchored_row(mf, fs, POINTS, ("value",), source="guess")


def test_the_row_matches_the_names_one_for_one():
    fs, mf = one(("electrostatic", "steric"))
    kinds = ("value", "gx", "gy", "gz", "grad")
    row = anchored_row(mf, fs, POINTS, kinds)
    names = anchor_names([s.name for s in fs.specs], ["a", "b", "c"], kinds)
    assert len(row) == len(names) == 3 * 2 * 5
    # the first five entries are the first point and the first field
    value = field_at_points(POINTS[:1], mf, fs.specs[0], 0)[0]
    grad = field_gradient_at_points(POINTS[:1], mf, fs.specs[0], 0)[0]
    assert row[0] == pytest.approx(value)
    assert row[1:4] == pytest.approx(grad)
    assert row[4] == pytest.approx(np.linalg.norm(grad))


def test_the_matrix_has_one_row_per_molecule():
    fs = FieldSet.create(physchem=["electrostatic"], alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(mol(s, seed=i + 1), fs, s)
          for i, s in enumerate(("c1ccccc1N", "c1ccccc1O", "c1ccccc1Cl"))]
    Z = anchored_matrix(mf, fs, POINTS, ("value", "grad"))
    assert Z.shape == (3, 3 * 1 * 2)
    assert np.all(np.isfinite(Z))
    # three different substituents must not give three identical rows
    assert np.abs(Z[0] - Z[1]).max() > 1e-6
    assert np.abs(Z[0] - Z[2]).max() > 1e-6


def test_the_coulomb_source_needs_the_electrostatic_field():
    fs = FieldSet.create(physchem=["steric"], alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = build_fields(mol(), fs, "m")
    with pytest.raises(ValueError):
        anchored_row(mf, fs, POINTS, ("value",), source="coulomb")


def test_the_two_sources_are_different_descriptions():
    """Both read the same charges; a Gaussian field and a 1/r potential are not the same."""
    fs, mf = one()
    gauss = anchored_row(mf, fs, POINTS, ("value", "grad"), source="field")
    coulomb = anchored_row(mf, fs, POINTS, ("value", "grad"), source="coulomb")
    assert len(gauss) == len(coulomb)
    assert not np.allclose(gauss, coulomb)
    assert set(ANCHOR_SOURCES) == {"field", "coulomb"}
    assert ANCHOR_KINDS[0] == "value"
