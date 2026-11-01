"""A series on one host: difference fields and the symmetry-averaged periodic overlap (M14).

What has to be true, each checked against something independent:

* the difference field is exact: the overlap of two whole cells equals the overlap of the host
  with itself, plus the host against each difference, plus the two differences -- bilinearity;
* a pristine cell has an empty difference, and the counts of a difference are the defects;
* the symmetry average makes two arrangements related by a host operation identical, leaves
  two that are not related distinct, and is a symmetric kernel whose self-overlaps are its
  diagonal;
* a model on the differences fits without the cosine normalisation, predicts the same number
  for two equivalent arrangements, and survives a round trip through a file.

The host is a 3 x 3 x 1 supercell of a square lattice of two species, the smallest lattice
with a point group large enough (p4mm, eight operations) for the translations and rotations to
be told apart.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ase = pytest.importorskip("ase", reason="the crystal path needs ASE")
pytest.importorskip("spglib", reason="the host symmetry needs spglib")

from ase import Atoms  # noqa: E402

from cmfqsar import periodic  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

A0 = 3.0


def host_cell(n=3):
    """A rock-salt-like square layer, Na and Cl alternating, n x n of a two-atom cell, with
    vacuum along z so that it is a slab in a periodic box."""
    syms, pos = [], []
    for i in range(n):
        for j in range(n):
            syms += ["Na", "Cl"]
            pos += [[i * 2 * A0, j * 2 * A0, 5.0], [i * 2 * A0 + A0, j * 2 * A0 + A0, 5.0]]
            syms += ["Cl", "Na"]
            pos += [[i * 2 * A0 + A0, j * 2 * A0, 5.0], [i * 2 * A0, j * 2 * A0 + A0, 5.0]]
    atoms = Atoms(symbols=syms, positions=pos, cell=[2 * n * A0, 2 * n * A0, 10.0], pbc=True)
    atoms.name = "NaCl layer"
    return atoms


def with_defects(host, remove=(), substitute=()):
    """`host` with the atoms of `remove` deleted and those of `substitute` (index, symbol)
    replaced."""
    a = host.copy()
    syms = list(a.get_chemical_symbols())
    for i, s in substitute:
        syms[i] = s
    a.set_chemical_symbols(syms)
    keep = [i for i in range(len(a)) if i not in set(remove)]
    return a[keep]


@pytest.fixture(scope="module")
def fieldset():
    return periodic.inorganic_fieldset(elements=["Na", "Cl", "K"], alpha=1.0)


@pytest.fixture(scope="module")
def host():
    return host_cell()


@pytest.fixture(scope="module")
def hostf(host, fieldset):
    return periodic.crystal_fields(host, fieldset, "NaCl layer")


@pytest.fixture(scope="module")
def symmetry(host):
    return periodic.host_symmetry(host)


def specs_of(fieldset):
    return [s for s in fieldset.specs if s.enabled]


def site_index(host, x, y):
    """The atom of `host` at (x, y) in units of the lattice constant."""
    pos = host.get_positions()
    d = np.linalg.norm(pos[:, :2] - np.array([x * A0, y * A0])[None, :], axis=1)
    return int(np.argmin(d))


# --------------------------------------------------------------------------- #
# The difference field
# --------------------------------------------------------------------------- #

def test_the_host_symmetry_has_the_translations_and_the_point_group(symmetry):
    rot, tr = symmetry
    # nine primitive translations of the 3 x 3 supercell times the eight operations of p4mm,
    # times two for the mirror through the layer: 144
    assert len(rot) == 144
    assert any(np.allclose(R, np.eye(3)) and np.allclose(t, 0) for R, t in zip(rot, tr))


def test_a_pristine_cell_is_an_empty_difference(host, hostf, fieldset):
    d = periodic.defect_fields(host, hostf, fieldset)
    assert d.n_atoms == 0
    assert d.relative_to == "NaCl layer"


def test_a_vacancy_and_a_substitution_are_the_rows_they_should_be(host, hostf, fieldset):
    i_na = site_index(host, 0, 0)
    i_cl = site_index(host, 1, 0)
    a = with_defects(host, remove=[i_cl], substitute=[(i_na, "K")])
    d = periodic.defect_fields(a, hostf, fieldset)
    assert sorted(d.atom_types) == ["+K", "-Cl", "-Na"]
    assert d.n_atoms == 3
    # the vacancy carries the host's weights with the sign reversed
    k = d.atom_types.index("-Cl")
    assert np.allclose(d.weights[k], -hostf.weights[i_cl])
    # the substituent takes the host's charge for the site: K on a Na site is +1
    k = d.atom_types.index("+K")
    assert d.charges[k] == pytest.approx(hostf.charges[i_na])
    table = periodic.defect_table([host, a], hostf, fieldset, ["pristine", "defective"])
    assert periodic.is_difference_table(table)
    assert list(periodic.defect_count(table)) == [0, 2]


def test_the_difference_is_exact_under_the_bilinear_kernel(host, hostf, fieldset):
    i1 = site_index(host, 0, 0)
    i2 = site_index(host, 2, 1)
    cells = [with_defects(host, remove=[i1]), with_defects(host, remove=[i2]),
             with_defects(host, remove=[i1], substitute=[(i2, "K")])]
    specs = specs_of(fieldset)
    # the whole cells get the host's charge rule (K on a Na site is +1), so that the comparison
    # is of the kernels and not of two charge models: the oxidation-state guess of a cell with a
    # vacancy cannot balance and would give every atom a different charge
    charge = {"Na": 1.0, "K": 1.0, "Cl": -1.0}
    fs_file = fieldset.copy()
    fs_file.charge_model = "file"
    for a in cells:
        a.set_initial_charges([charge[s] for s in a.get_chemical_symbols()])
    whole = periodic.crystal_table(cells, fs_file)
    K_whole = periodic.periodic_overlap_kernels(whole, whole, specs)
    diff = periodic.defect_table(cells, hostf, fieldset)
    hosttab = periodic.crystal_table([host], fieldset, ["host"])
    K_hh = periodic.periodic_overlap_kernels(hosttab, hosttab, specs)
    K_hd = periodic.periodic_overlap_kernels(hosttab, diff, specs)
    K_dd = periodic.periodic_overlap_kernels(diff, diff, specs)
    K_rec = K_hh + K_hd[:, 0, :, None] + K_hd[:, 0, None, :] + K_dd
    # to the truncation of the image sum, which the cutoff rule puts at 1e-6 of a peak
    # overlap per pair and which the sum over the cell's atoms brings to about 1e-8
    for f in range(K_whole.shape[0]):
        scale = max(float(np.max(np.abs(K_whole[f]))), 1e-12)
        assert np.max(np.abs(K_rec[f] - K_whole[f])) < 1e-7 * scale


def test_two_atoms_on_one_site_are_refused(host, hostf, fieldset):
    a = host.copy()
    a.append(ase.Atom("Na", a.get_positions()[0] + [0.01, 0.0, 0.0]))
    with pytest.raises(ValueError, match="two atoms"):
        periodic.defect_fields(a, hostf, fieldset)


# --------------------------------------------------------------------------- #
# The symmetry average
# --------------------------------------------------------------------------- #

def pair_cells(host):
    """Three cells with two Cl vacancies: a pair along x, the same pair moved by a lattice
    translation, and a pair along the diagonal, which is a different arrangement."""
    along_x = [site_index(host, 1, 0), site_index(host, 3, 0)]
    moved = [site_index(host, 1, 2), site_index(host, 3, 2)]
    diagonal = [site_index(host, 1, 0), site_index(host, 3, 2)]
    return [with_defects(host, remove=r) for r in (along_x, moved, diagonal)]


def test_the_average_identifies_equivalent_arrangements_and_keeps_distinct_ones(
        host, hostf, fieldset, symmetry):
    table = periodic.defect_table(pair_cells(host), hostf, fieldset, ["x", "x moved", "diag"])
    specs = specs_of(fieldset)
    K_raw = periodic.periodic_overlap_kernels(table, table, specs)
    K_sym = periodic.periodic_overlap_kernels(table, table, specs, symmetry=symmetry)
    f = [s.name for s in specs].index("steric")
    # in the fixed frame the moved pair hardly overlaps the original (the tails of Gaussians
    # six Angstrom apart): the kernel is all but blind to it
    assert abs(K_raw[f, 0, 1]) < 1e-3 * abs(K_raw[f, 0, 0])
    # under the symmetry average the two are the same structure ...
    assert K_sym[f, 0, 1] == pytest.approx(K_sym[f, 0, 0], rel=1e-6)
    assert K_sym[f, 1, 1] == pytest.approx(K_sym[f, 0, 0], rel=1e-6)
    # ... and so is the diagonal pair: the orbit-averaged field of a vacancy is the same
    # whatever its neighbours, so the linear average is a kernel of the defect counts and
    # knows nothing of the arrangement.  That is what symmetry_layer_kernel is for.
    assert K_sym[f, 0, 2] == pytest.approx(K_sym[f, 0, 0], rel=1e-6)
    # the averaged kernel is symmetric even before the explicit symmetrisation
    one = periodic.periodic_overlap_kernels(table, table, specs, symmetry=symmetry,
                                            symmetrise=False)
    assert np.allclose(one, np.swapaxes(one, 1, 2), rtol=1e-6, atol=1e-10)


def test_the_linear_average_counts_the_defects(host, hostf, fieldset, symmetry):
    """K(A, B) = n_A n_B c for single-species vacancies: a pair against a single is twice a
    single against a single, and a pair against a pair four times."""
    i = site_index(host, 1, 0)
    cells = [with_defects(host, remove=[i])] + pair_cells(host)[:1]
    table = periodic.defect_table(cells, hostf, fieldset)
    K = periodic.periodic_overlap_kernels(table, table, specs_of(fieldset), symmetry=symmetry)
    assert np.allclose(K[:, 0, 1], 2 * K[:, 0, 0], rtol=1e-6, atol=1e-12)
    assert np.allclose(K[:, 1, 1], 4 * K[:, 0, 0], rtol=1e-6, atol=1e-12)


def test_the_averaged_layer_sees_the_arrangement(host, hostf, fieldset, symmetry):
    """The Gaussian layer averaged over the host operations: equivalent arrangements are one
    structure, inequivalent ones are not, and the result is a symmetric kernel with ones on
    the diagonal only for a structure that is its own orbit."""
    table = periodic.defect_table(pair_cells(host), hostf, fieldset, ["x", "x moved", "diag"])
    specs = specs_of(fieldset)
    K = periodic.symmetry_layer_kernel(table, table, specs, symmetry, gammas=[0.05, 0.5])
    assert K.shape == (2, 3, 3)
    for k in range(2):
        assert np.allclose(K[k], K[k].T)
        # the two placements of the pair along x are the same structure: equal rows
        assert np.allclose(K[k, 0], K[k, 1], rtol=1e-6)
        # the diagonal pair is a different structure: its overlap with the x pair is smaller
        # than the x pair's with its own moved copy
        assert K[k, 0, 2] < K[k, 0, 1] - 1e-3
        # a structure against itself: the identity gives exp(0) = 1, the other operations less
        assert K[k, 0, 0] <= 1.0 + 1e-12 and K[k, 0, 0] >= 1.0 / len(symmetry[0])
    # a sharper layer separates more
    assert K[1, 0, 2] / K[1, 0, 1] < K[0, 0, 2] / K[0, 0, 1]


def test_the_compiled_and_the_reference_paths_agree(host, hostf, fieldset, symmetry):
    """The compiled loop takes every pair at its nearest image; the reference sums the images.
    With the cutoff under half the cell they are the same sum, and the linear average of
    either is the kernel of the symmetry-expanded table."""
    from cmfqsar import fastkernel

    if not fastkernel.available():
        pytest.skip("the compiled path needs Numba")
    i = site_index(host, 1, 0)
    cells = [with_defects(host, remove=[i])] + pair_cells(host)
    table = periodic.defect_table(cells, hostf, fieldset)
    specs = specs_of(fieldset)
    r = periodic.nearest_image_cutoff(table, 100.0)       # half the cell width: 9 A
    assert r == pytest.approx(9.0)
    mixing = np.ones(len(specs))
    lin_c, lay_c = periodic.symmetry_averaged_kernels(table, table, specs, symmetry,
                                                      gammas=[0.1, 1.0], mixing=mixing,
                                                      r_max=r, engine="numba")
    lin_n, lay_n = periodic.symmetry_averaged_kernels(table, table, specs, symmetry,
                                                      gammas=[0.1, 1.0], mixing=mixing,
                                                      r_max=r, engine="numpy")
    for f in range(lin_n.shape[0]):
        scale = max(float(np.max(np.abs(lin_n[f]))), 1e-12)
        assert np.max(np.abs(lin_c[f] - lin_n[f])) < 1e-6 * scale
    assert np.allclose(lay_c, lay_n, rtol=1e-6, atol=1e-9)
    K_sym = periodic.periodic_overlap_kernels(table, table, specs, symmetry=symmetry, r_max=r)
    for f in range(lin_n.shape[0]):
        scale = max(float(np.max(np.abs(K_sym[f]))), 1e-12)
        assert np.max(np.abs(lin_n[f] - K_sym[f])) < 1e-6 * scale


def test_the_averaged_layer_is_the_mean_of_the_transformed_layers(host, hostf, fieldset,
                                                                  symmetry):
    """Against the definition, operation by operation, on the identity plus one rotation."""
    table = periodic.defect_table(pair_cells(host), hostf, fieldset)
    specs = specs_of(fieldset)
    rot, tr = symmetry
    pick = [0, 5]
    sub = (rot[pick], tr[pick])
    mixing = np.ones(len(specs))
    K = periodic.symmetry_layer_kernel(table, table, specs, sub, gammas=[0.1], mixing=mixing)
    d = np.tensordot(mixing, periodic.periodic_self_overlaps(table, specs), axes=(0, 0))
    acc = np.zeros((3, 3))
    for R, t in zip(*sub):
        moved = periodic.symmetry_expanded_table(table, R[None], t[None])
        Kg = np.tensordot(mixing, periodic.periodic_overlap_kernels(table, moved, specs,
                                                                     symmetrise=False),
                          axes=(0, 0))
        acc += np.exp(-0.1 * (d[:, None] + d[None, :] - 2 * Kg))
    acc /= 2
    assert np.allclose(K[0], 0.5 * (acc + acc.T), rtol=1e-8, atol=1e-10)


def test_the_self_overlap_is_the_diagonal_under_symmetry(host, hostf, fieldset, symmetry):
    table = periodic.defect_table(pair_cells(host), hostf, fieldset)
    specs = specs_of(fieldset)
    K = periodic.periodic_overlap_kernels(table, table, specs, symmetry=symmetry)
    d = periodic.periodic_self_overlaps(table, specs, symmetry=symmetry)
    assert np.allclose(d, np.einsum("fii->fi", K), rtol=1e-6, atol=1e-12)


def test_the_average_is_no_larger_than_the_fixed_frame_self_overlap(host, hostf, fieldset,
                                                                      symmetry):
    """The orbit average is a mean of fields, so its norm is at most the norm of one of them."""
    table = periodic.defect_table(pair_cells(host), hostf, fieldset)
    specs = specs_of(fieldset)
    d_raw = periodic.periodic_self_overlaps(table, specs)
    d_sym = periodic.periodic_self_overlaps(table, specs, symmetry=symmetry)
    assert np.all(d_sym <= d_raw * (1 + 1e-9))


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

def series(host):
    """Eight cells: single Cl vacancies at four places (equivalent) and four vacancy pairs at
    two distances, each twice in different places."""
    cl = [(1, 0), (3, 0), (1, 2), (5, 4)]
    singles = [with_defects(host, remove=[site_index(host, *p)]) for p in cl]
    near = [[(1, 0), (3, 0)], [(1, 2), (3, 2)]]
    far = [[(1, 0), (5, 2)], [(3, 0), (1, 4)]]
    pairs = [with_defects(host, remove=[site_index(host, *p) for p in pr]) for pr in near + far]
    y = np.array([1.0] * 4 + [2.3, 2.3, 2.0, 2.0])
    return singles + pairs, y


def test_a_model_on_the_differences_needs_no_normalisation_and_sees_the_symmetry(
        host, hostf, fieldset, symmetry, tmp_path):
    cells, y = series(host)
    table = periodic.defect_table(cells, hostf, fieldset)
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-6}, normalize=False,
                     pbc_overlap=True, pbc_symmetry=symmetry)
    model.fit(table, y, cv_folds=0)
    pred = model.predict(table)
    # the four single vacancies are one structure to the model: one prediction
    assert np.ptp(pred[:4]) < 1e-6
    # every pair is two single vacancies to the linear model, near or far: one prediction,
    # which is the mean of the two pair values the series was given
    assert np.ptp(pred[4:]) < 1e-6
    assert pred[4] == pytest.approx(2.15, abs=0.05)
    # the per-site contributions still add up to the prediction (decomposition rule 3): to
    # rounding through the compiled loop, which takes the same nearest-image sum as the
    # prediction, and to the truncation of the image sum at half the cell (9 A here) on the
    # reference path
    from cmfqsar import fastkernel
    contrib = model.atom_contributions(table.molecules[4])
    total = float(np.sum(contrib)) + float(model.bias)
    assert total == pytest.approx(pred[4], rel=1e-8 if fastkernel.available() else 1e-3)
    from cmfqsar.dataio import load_model, save_model

    assert model.fresh_copy().pbc_symmetry is symmetry
    path = str(tmp_path / "defects.cmf")
    save_model(path, model)
    back, _extra = load_model(path)
    assert back.pbc_symmetry is not None and len(back.pbc_symmetry[0]) == len(symmetry[0])
    assert np.allclose(back.predict(table), pred, atol=1e-8)


def test_without_the_symmetry_the_same_defect_elsewhere_is_a_stranger(host, hostf, fieldset):
    cells, y = series(host)
    table = periodic.defect_table(cells, hostf, fieldset)
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-6}, normalize=False,
                     pbc_overlap=True)
    model.fit(table.subset([0, 4]), y[[0, 4]], cv_folds=0)
    pred = model.predict(table)
    # the single vacancy at a site the model never saw has next to no overlap with anything it
    # did, so it is predicted at the bias; the vacancy it did see is predicted as itself
    assert pred[0] == pytest.approx(y[0], abs=1e-3)
    assert abs(pred[3] - float(model.bias)) < 1e-3 * abs(pred[0] - float(model.bias))


def test_whole_cells_still_require_the_normalisation(host, fieldset):
    cells, y = series(host)
    whole = periodic.crystal_table(cells, fieldset)
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-4}, normalize=False,
                     pbc_overlap=True)
    with pytest.raises(ValueError, match="normalize=True"):
        model.fit(whole, y, cv_folds=0)


# --------------------------------------------------------------------------- #
# What the review of 1 November 2026 found: the charge of an added atom, the count of a
# vacancy next to an interstitial, and the self-overlaps a model without normalisation uses
# at prediction time
# --------------------------------------------------------------------------- #

def test_an_interstitial_takes_the_charge_the_balance_gives_it(host, hostf, fieldset):
    from ase import Atom
    a = host.copy()
    a.append(Atom("K", (1.5, 1.5, 5.0)))       # 2.1 A from the nearest site: on no site
    diff = periodic.defect_fields(a, hostf, fieldset)
    assert diff.atom_types == ["+K"]
    assert diff.n_defects == 1
    # Na18 Cl18 K does not balance exactly; the nearest balance is K+ with Na+ and Cl-
    assert diff.charges[0] == pytest.approx(1.0)
    # the electrostatic weight of the row is that charge, not zero
    f = [s.name for s in fieldset.specs].index("electrostatic")
    assert diff.weights[0, f] != 0.0


def test_a_substitution_on_an_elemental_host_has_a_charge_of_zero_and_still_builds():
    syms = ["C"] * 8
    pos = [[i * 2.5, j * 2.5, 5.0] for i in range(2) for j in range(2)]
    pos += [[i * 2.5 + 1.25, j * 2.5 + 1.25, 5.0] for i in range(2) for j in range(2)]
    carbon = Atoms(symbols=syms, positions=pos, cell=[5.0, 5.0, 10.0], pbc=True)
    fs = periodic.inorganic_fieldset(elements=["C", "N"], alpha=1.0)
    hostf = periodic.crystal_fields(carbon, fs, "carbon layer")
    assert np.all(np.asarray(hostf.charges) == 0.0)      # an elemental solid
    a = carbon.copy()
    a.set_chemical_symbols(["N"] + ["C"] * 7)
    diff = periodic.defect_fields(a, hostf, fs)           # used to raise: no non-zero charge
    assert sorted(diff.atom_types) == ["+N", "-C"]
    assert diff.n_defects == 1
    assert np.all(np.asarray(diff.charges) == 0.0)


def test_a_vacancy_and_an_interstitial_count_two(host, hostf, fieldset):
    from ase import Atom
    a = with_defects(host, remove=[0])
    a.append(Atom("K", (1.5, 1.5, 5.0)))
    table = periodic.defect_table([a, host], hostf, fieldset)
    assert sorted(table.molecules[0].atom_types) == ["+K", "-Na"]
    assert periodic.defect_count(table).tolist() == [2, 0]
    # a record without the number falls back to the rows, where a + row on the coordinate of a
    # - row is a substitution: a vacancy next to an interstitial is still two
    table.molecules[0].n_defects = None
    assert periodic.defect_count(table).tolist() == [2, 0]


def test_the_test_side_self_overlaps_are_the_periodic_ones(host, hostf, fieldset, symmetry):
    cells, y = series(host)
    table = periodic.defect_table(cells, hostf, fieldset)
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-6, "gamma": 1.0},
                     normalize=False, pbc_overlap=True, pbc_symmetry=symmetry,
                     kernel_layer="gauss")
    model.fit(table, y, cv_folds=0)
    # the diagonal a new table gets is the diagonal the training kernel was built from: with
    # open-boundary self-kernels instead, every test distance would be overestimated
    d_test = model.joint_self(table)
    d_train = np.asarray(model.base_diag_, float)
    assert np.allclose(d_test, d_train, rtol=1e-8, atol=1e-10)
    # and so the training cells predict themselves as the fit saw them: two equivalent single
    # vacancies get one prediction, as they do for the linear kernel
    pred = model.predict(table)
    assert np.ptp(pred[:4]) < 1e-6
