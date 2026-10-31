"""The periodic overlap kernel and the co-field inside the cell (plan item M4).

What has to be true of this kernel is not a number but a set of properties, and each test below
checks one of them against something independent:

* it is the ordinary overlap kernel of :mod:`cmfqsar.kernels` when the cell is large enough for
  the images not to reach -- the open-boundary limit, which is an exact comparison;
* it does not depend on where the origin of the cell is put, which the open-boundary kernel of
  the same atoms does depend on;
* it does not depend on which cell of the lattice is used, up to the truncation of the image sum;
* the co-field it gives is periodic;
* the per-site contributions add up to the prediction, which is decomposition rule 3.

A set on one parent lattice is the case the item is for, and the tests use the perovskite cell,
which is the smallest structure with more than one kind of site.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ase = pytest.importorskip("ase", reason="the crystal path needs ASE")

from ase import Atoms  # noqa: E402

from cmfqsar import periodic  # noqa: E402
from cmfqsar.cofields import Grid, cofield_volume  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSpec  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

ELEMENTS = ["Sr", "Ba", "Ti", "Zr", "O"]


def perovskite(a=4.0, A="Sr", B="Ti", disp=0.0, origin=(0.0, 0.0, 0.0)):
    """One cubic ABO3 cell: A at the corner, B at the centre, O on the faces.

    `disp` moves the B atom along z, which is the ferroelectric distortion and the only thing
    that makes the three oxygens inequivalent; `origin` shifts every atom in fractional
    coordinates, which changes nothing about the structure.
    """
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5 + disp], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return Atoms(symbols=[A, B, "O", "O", "O"],
                 scaled_positions=(pos + np.asarray(origin, float)[None, :]) % 1.0,
                 cell=[a, a, a], pbc=True)


@pytest.fixture(scope="module")
def fieldset():
    return periodic.inorganic_fieldset(elements=ELEMENTS, alpha=2.0)


@pytest.fixture(scope="module")
def built(fieldset):
    """Six cells on one lattice: two A sites, two B sites and two distorted cells."""
    cells = [perovskite(4.0, "Sr", "Ti"), perovskite(4.0, "Ba", "Ti"),
             perovskite(4.0, "Sr", "Zr"), perovskite(4.0, "Ba", "Zr"),
             perovskite(4.0, "Sr", "Ti", disp=0.05), perovskite(4.0, "Ba", "Zr", disp=0.05)]
    names = ["SrTiO3", "BaTiO3", "SrZrO3", "BaZrO3", "SrTiO3-d", "BaZrO3-d"]
    table = periodic.crystal_table(cells, fieldset, names)
    specs = [s for s in fieldset.specs if s.enabled]
    return table, specs, names


def _fitted(fieldset, table, y, **kw):
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-4}, normalize=True,
                     pbc_overlap=True, **kw)
    model.fit(table, y, cv_folds=0)
    return model


# --------------------------------------------------------------------------- #
# The image expansion
# --------------------------------------------------------------------------- #

def test_the_images_reach_the_radius_and_no_further(built):
    table, specs, _names = built
    r = 6.0
    x, w, rad = periodic.cell_images(table.molecules[0], r)
    assert len(x) == len(w) == len(rad)
    assert len(x) > table.molecules[0].n_atoms          # images were added
    cell = np.asarray(table.molecules[0].cell, float)
    lo = np.zeros(3)
    hi = cell.sum(axis=0)                               # the cubic cell spans [0, a]^3
    d = periodic._box_distance(x, lo, hi)
    assert float(d.max()) <= r + 1e-9
    # the atoms of the cell itself come first and are unchanged
    assert np.allclose(x[:5], table.molecules[0].coords)


def test_a_bigger_radius_keeps_everything_a_smaller_one_does(built):
    table, _specs, _names = built
    small = periodic.cell_images(table.molecules[0], 4.0)[0]
    large = periodic.cell_images(table.molecules[0], 8.0)[0]
    assert len(large) > len(small)
    seen = {tuple(np.round(p, 6)) for p in large}
    assert all(tuple(np.round(p, 6)) in seen for p in small)


def test_the_expansion_keeps_the_structure_boundaries(built):
    table, _specs, _names = built
    images = periodic.expanded_cell_table(table, 5.0)
    assert images.n_mol == table.n_mol
    assert list(images.names) == list(table.names)
    assert images.offsets[-1] == len(images.coords)
    # and it is cached, because a grid search asks for it once per attenuation
    assert periodic.expanded_cell_table(table, 5.0) is images


# --------------------------------------------------------------------------- #
# The kernel
# --------------------------------------------------------------------------- #

def test_the_self_overlap_is_the_diagonal_of_the_kernel(built):
    table, specs, _names = built
    K = periodic.periodic_overlap_kernels(table, table, specs)
    d = periodic.periodic_self_overlaps(table, specs)
    for f in range(K.shape[0]):
        assert np.allclose(np.diag(K[f]), d[f], rtol=1e-10)


def test_the_two_orders_of_the_integral_agree_on_one_lattice(built):
    table, specs, _names = built
    assert periodic.overlap_asymmetry(table, specs) < 1e-6


def test_the_two_orders_disagree_when_the_lattice_does(fieldset):
    cells = [perovskite(4.0, "Sr", "Ti"), perovskite(4.4, "Ba", "Zr")]
    table = periodic.crystal_table(cells, fieldset, ["a", "b"])
    specs = [s for s in fieldset.specs if s.enabled]
    assert periodic.overlap_asymmetry(table, specs) > 1e-3
    K = periodic.periodic_overlap_kernels(table, table, specs)
    for f in range(K.shape[0]):
        assert np.allclose(K[f], K[f].T, rtol=1e-10)      # the average is symmetric regardless


def _cluster(a=40.0, A="Sr", B="Ti"):
    """The five atoms of a perovskite cell kept together in a large empty box."""
    pos = np.array([[0, 0, 0], [2.0, 2.0, 2.0], [2.0, 2.0, 0], [2.0, 0, 2.0], [0, 2.0, 2.0]])
    return Atoms(symbols=[A, B, "O", "O", "O"], positions=pos + 1.0, cell=[a, a, a], pbc=True)


def test_an_isolated_cell_gives_the_open_boundary_kernel(fieldset):
    """With a box far larger than the fields reach, the images contribute nothing at all."""
    table = periodic.crystal_table([_cluster(40.0, "Sr", "Ti"), _cluster(40.0, "Ba", "Zr")],
                                   fieldset, ["a", "b"])
    specs = [s for s in fieldset.specs if s.enabled]
    K_pbc = periodic.periodic_overlap_kernels(table, table, specs)
    K_open = kernel_matrices(table, table, specs, symmetric=True)
    assert np.allclose(K_pbc, K_open, rtol=1e-12, atol=0)


def _unwrapped(a=4.0, A="Sr", B="Ti", origin=(0.6, 0.7, 0.8)):
    """The same crystal with its atoms written outside the cell instead of inside it."""
    pos = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    return Atoms(symbols=[A, B, "O", "O", "O"],
                 scaled_positions=pos + np.asarray(origin, float)[None, :],
                 cell=[a, a, a], pbc=True)


def test_writing_the_atoms_outside_the_cell_changes_nothing(fieldset):
    """A crystal is its lattice sum, and does not care which atoms were called the cell.

    The open-boundary kernel of the same atoms does care: moving an atom to the far side of the
    cell is a different cluster of atoms, and that is the difference the periodic sum removes.
    """
    specs = [s for s in fieldset.specs if s.enabled]
    # only the first structure is written differently; the second is the same file in both, so
    # a change in the kernel can only come from how the first one was written down
    inside = periodic.crystal_table([perovskite(4.0, "Sr", "Ti", origin=(0.6, 0.7, 0.8)),
                                     perovskite(4.0, "Ba", "Zr")], fieldset, ["a", "b"])
    outside = periodic.crystal_table([_unwrapped(4.0, "Sr", "Ti"), perovskite(4.0, "Ba", "Zr")],
                                     fieldset, ["a", "b"])
    far = 20.0            # far enough that every dropped pair is below 1e-12 of a peak overlap
    K1 = periodic.periodic_overlap_kernels(inside, inside, specs, r_max=far)
    K2 = periodic.periodic_overlap_kernels(outside, outside, specs, r_max=far)
    assert np.allclose(K1, K2, rtol=1e-7, atol=1e-9)
    O1 = kernel_matrices(inside, inside, specs, symmetric=True)
    O2 = kernel_matrices(outside, outside, specs, symmetric=True)
    assert not np.allclose(O1, O2, rtol=1e-2)


def test_the_field_norm_does_not_depend_on_the_origin_but_the_overlap_does(fieldset):
    """Shifting one structure alone: its own norm is unchanged, its overlap with another is not.

    This is the frame, in the form a crystal takes it.  Two cells must be written on one lattice
    *and* with one origin for the comparison to mean anything, exactly as two molecules must be
    posed in one frame; :func:`periodic.lattice_spread` reports the first and nothing can report
    the second, so the series has to come with it.
    """
    specs = [s for s in fieldset.specs if s.enabled]
    plain = periodic.crystal_table([perovskite(4.0, "Sr", "Ti"), perovskite(4.0, "Ba", "Zr")],
                                   fieldset, ["a", "b"])
    one_moved = periodic.crystal_table(
        [perovskite(4.0, "Sr", "Ti", origin=(0.6, 0.7, 0.8)), perovskite(4.0, "Ba", "Zr")],
        fieldset, ["a", "b"])
    d1 = periodic.periodic_self_overlaps(plain, specs, r_max=20.0)
    d2 = periodic.periodic_self_overlaps(one_moved, specs, r_max=20.0)
    assert np.allclose(d1, d2, rtol=1e-7, atol=1e-9)
    K1 = periodic.periodic_overlap_kernels(plain, plain, specs, r_max=20.0)
    K2 = periodic.periodic_overlap_kernels(one_moved, one_moved, specs, r_max=20.0)
    assert not np.allclose(K1[0], K2[0], rtol=1e-2)


def test_the_truncation_of_the_image_sum_is_what_is_left(fieldset):
    """How far the images have to reach is a measurement, not a choice.

    The image sum is cut where a pair contributes less than 1e-6 of a peak overlap, exactly as
    the open-boundary kernel is cut (plan item B2).  What the cut costs is a small dependence on
    which atoms were called the cell, and it falls as the radius grows, which is what says that
    the cut is the cause.
    """
    specs = [s for s in fieldset.specs if s.enabled]
    # only the first structure is written differently; the second is the same file in both, so
    # a change in the kernel can only come from how the first one was written down
    inside = periodic.crystal_table([perovskite(4.0, "Sr", "Ti", origin=(0.6, 0.7, 0.8)),
                                     perovskite(4.0, "Ba", "Zr")], fieldset, ["a", "b"])
    outside = periodic.crystal_table([_unwrapped(4.0, "Sr", "Ti"), perovskite(4.0, "Ba", "Zr")],
                                     fieldset, ["a", "b"])
    errors = []
    for r in (7.0, 11.0, 15.0):
        K1 = periodic.periodic_overlap_kernels(inside, inside, specs, r_max=r)
        K2 = periodic.periodic_overlap_kernels(outside, outside, specs, r_max=r)
        busy = np.abs(K1) > 1e-9
        errors.append(float(np.max(np.abs(K1[busy] - K2[busy]) / np.abs(K1[busy]))))
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1e-4


def test_a_supercell_gives_the_same_normalised_kernel(fieldset):
    """The cell is a choice; the structure is not.  The raw value doubles and the cosine does not."""
    specs = [s for s in fieldset.specs if s.enabled]
    cells = [perovskite(4.0, "Sr", "Ti"), perovskite(4.0, "Ba", "Zr"),
             perovskite(4.0, "Sr", "Zr")]
    prim = periodic.crystal_table(cells, fieldset, ["a", "b", "c"])
    sup = periodic.crystal_table([c.repeat((2, 1, 1)) for c in cells], fieldset, ["a", "b", "c"])
    K1 = periodic.periodic_overlap_kernels(prim, prim, specs)
    d1 = periodic.periodic_self_overlaps(prim, specs)
    K2 = periodic.periodic_overlap_kernels(sup, sup, specs)
    d2 = periodic.periodic_self_overlaps(sup, specs)
    busy = np.abs(K1) > 1e-9                   # a field no element of this set carries is zero
    assert np.allclose(K2[busy] / K1[busy], 2.0, rtol=2e-2)   # the integral is twice as big
    n1 = normalize_kernels(K1, d1, d1)
    n2 = normalize_kernels(K2, d2, d2)
    # what is left is the truncation of the image sum, which differs between the two cell shapes
    assert float(np.max(np.abs(n1 - n2))) < 1e-2


def test_a_probe_field_is_refused(fieldset, built):
    table, specs, _names = built
    with pytest.raises(ValueError, match="probe"):
        periodic.periodic_overlap_kernels(
            table, table, list(specs) + [FieldSpec(name="probe:density", kind="probe", alpha=1.0)])


def test_a_structure_without_a_cell_is_refused(fieldset):
    from cmfqsar.fields import MoleculeFields

    mol = MoleculeFields(name="m", coords=np.zeros((1, 3)), weights=np.ones((1, 3)),
                         radii=np.ones(1), symbols=["O"], atom_types=["O"],
                         bonds=np.zeros((0, 2), int), charges=np.zeros(1))
    with pytest.raises(ValueError, match="unit cell"):
        periodic.cell_images(mol, 5.0)


def test_lattice_spread_sees_a_mixed_set(fieldset):
    one = periodic.crystal_table([perovskite(4.0), perovskite(4.0, "Ba")], fieldset, ["a", "b"])
    mixed = periodic.crystal_table([perovskite(4.0), perovskite(4.4, "Ba")], fieldset, ["a", "b"])
    assert periodic.lattice_spread(one)["length_spread"] < 1e-12
    assert periodic.lattice_spread(mixed)["length_spread"] > 0.04
    assert periodic.lattice_spread(one)["atoms_mean"] == 5.0


# --------------------------------------------------------------------------- #
# The model, the co-field and rule 3
# --------------------------------------------------------------------------- #

def test_the_cosine_normalisation_is_required(fieldset, built):
    table, _specs, _names = built
    y = np.arange(table.n_mol, dtype=float)
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-4}, normalize=False,
                     pbc_overlap=True)
    with pytest.raises(ValueError, match="grows with the cell"):
        model.fit(table, y, cv_folds=0)


def test_the_model_predicts_its_own_training_set(fieldset, built):
    table, _specs, _names = built
    y = np.array([1.0, 2.0, 3.0, 4.2, 1.3, 4.4])
    model = _fitted(fieldset, table, y)
    assert np.allclose(model.predict(table), y, atol=0.05)


def test_the_per_site_contributions_add_up_to_the_prediction(fieldset, built):
    """Decomposition rule 3 for a crystal: one contribution per site of the cell."""
    table, _specs, _names = built
    y = np.array([1.0, 2.0, 3.0, 4.2, 1.3, 4.4])
    model = _fitted(fieldset, table, y)
    pred = model.predict(table)
    for i in range(table.n_mol):
        c = model.atom_contributions(table.molecules[i])
        assert len(c) == table.molecules[i].n_atoms
        # the kernel is the average of the two orders of the integral while the decomposition is
        # the one order that splits over the atoms of this cell, so the two agree to the
        # asymmetry of the set, which on one lattice is the truncation of the image sum
        assert float(np.sum(c)) + model.bias == pytest.approx(float(pred[i]), abs=1e-6)


def test_the_equivalent_sites_of_an_undistorted_set_agree(fieldset):
    """The three oxygens of a cubic cell are the same site, and the model must say so."""
    cells = [perovskite(4.0, "Sr", "Ti"), perovskite(4.0, "Ba", "Ti"),
             perovskite(4.0, "Sr", "Zr"), perovskite(4.0, "Ba", "Zr")]
    table = periodic.crystal_table(cells, fieldset, ["a", "b", "c", "d"])
    model = _fitted(fieldset, table, np.array([1.0, 2.0, 3.0, 4.0]))
    c = model.atom_contributions(table.molecules[0])
    assert np.allclose(c[2:], c[2], rtol=1e-6, atol=1e-9)


def _cofield_period_error(model, table, shift=(1, 0, 0)):
    """The largest relative change of the co-field under one lattice translation."""
    cell = np.asarray(table.molecules[0].cell, float)
    rng = np.random.RandomState(0)
    points = rng.rand(12, 3) @ cell
    moved = points + np.asarray(shift, float) @ cell
    worst = 0.0
    for f in (0, 1, 4):
        centres, coeffs, widths = model.cofield_coefficients(f)
        here = _dense(points, centres, coeffs, widths)
        there = _dense(moved, centres, coeffs, widths)
        worst = max(worst, float(np.max(np.abs(here - there))
                                 / max(float(np.abs(here).max()), 1e-12)))
    return worst


def test_the_co_field_is_periodic(fieldset, built):
    """What the item asks for: a co-field inside the cell, which repeats with the lattice.

    It repeats up to the same truncation of the image sum as the kernel: the Gaussians of the
    basis reach `r_max` around the cell, so a point one cell away sees a slightly smaller set of
    them.  The error falls as the radius grows.
    """
    table, _specs, _names = built
    y = np.array([1.0, 2.0, 3.0, 4.2, 1.3, 4.4])
    errors = [_cofield_period_error(_fitted(fieldset, table, y, pbc_overlap_r_max=r), table)
              for r in (8.0, 16.0)]
    assert errors[0] > errors[1]
    assert errors[1] < 0.01


def _dense(points, centres, weights, widths):
    """The Gaussian sum written out, so that nothing is dropped by a display cutoff."""
    d2 = ((np.asarray(points, float)[:, None, :] - np.asarray(centres, float)[None, :, :]) ** 2
          ).sum(-1)
    return (np.asarray(weights, float) * np.exp(-0.5 * np.asarray(widths, float) * d2)).sum(axis=1)


def test_the_co_field_can_be_drawn_on_a_grid(fieldset, built):
    table, _specs, _names = built
    y = np.array([1.0, 2.0, 3.0, 4.2, 1.3, 4.4])
    model = _fitted(fieldset, table, y)
    cell = np.asarray(table.molecules[0].cell, float)
    grid = Grid(origin=np.zeros(3), spacing=float(cell[0, 0]) / 8.0, shape=(8, 8, 8))
    vol = cofield_volume(model, 0, grid)
    assert vol.shape == (8, 8, 8)
    assert np.all(np.isfinite(vol))
    assert float(np.abs(vol).max()) > 0


def test_a_coalition_of_sites_is_refused(fieldset, built):
    """Shapley values need the kernel of a subset of atoms, which a cell does not have."""
    table, _specs, _names = built
    model = _fitted(fieldset, table, np.array([1.0, 2.0, 3.0, 4.2, 1.3, 4.4]))
    with pytest.raises(ValueError, match="periodic cell"):
        model._atom_self_block(table.molecules[0])


def test_the_settings_survive_a_copy_and_a_round_trip(fieldset, built, tmp_path):
    table, _specs, _names = built
    model = _fitted(fieldset, table, np.array([1.0, 2.0, 3.0, 4.2, 1.3, 4.4]),
                    pbc_overlap_r_max=7.0)
    assert model.fresh_copy().pbc_overlap is True
    assert model.fresh_copy().pbc_overlap_r_max == 7.0
    from cmfqsar.dataio import load_model, save_model

    path = str(tmp_path / "pbc.cmf")
    save_model(path, model)
    back, _extra = load_model(path)
    assert back.pbc_overlap is True
    assert back.pbc_overlap_r_max == 7.0
    assert np.allclose(back.predict(table), model.predict(table), atol=1e-10)
