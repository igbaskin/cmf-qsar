"""The three-body channels and the handedness block of a crystal: plan item M10.

What is tested, in the order the module builds it:

* the neighbourhood of a centre: the centre itself at the vector zero, the periodic images, the
  cutoff, and the periodicity flags of a layer;
* the invariances a descriptor of a crystal must have: a supercell, a shift of the origin and a
  rotation change nothing, exactly, while another structure does change the channels;
* the two blind spots the item is for: two structures that share their distance histogram, which
  the two-point channels cannot tell apart and these can, and a structure against its mirror
  image, which the handedness block separates and the three-body block does not;
* when the handedness block is empty, which is a property of the block and not a fault;
* the pieces around the descriptor: the labels, the blocks, the feature matrix, a site record, a
  fitted model and its file, the reading of that model, and the cases the module refuses.
"""
from itertools import combinations

import numpy as np
import pytest

from cmfqsar import threebody as T
from cmfqsar.fields import AtomTable
from cmfqsar.invariant import chirality_triples, field_pairs
from cmfqsar.model import CMFModel
from cmfqsar.periodic import (PeriodicSpec, crystal_fields, inorganic_fieldset,
                              periodic_autocorrelation, periodic_channel_labels)

ase_build = pytest.importorskip("ase.build")
bulk = ase_build.bulk
Atoms = pytest.importorskip("ase").Atoms

SPEC = T.ThreeBodySpec(r_max=4.0, step=0.5, sigma=0.35)
CHIRAL = T.ThreeBodySpec(r_max=4.5, step=0.5, sigma=0.35, chirality=True, chirality_shells=3)


def worst(a, b):
    """The largest difference between two descriptors, relative to the larger of the two."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))), 1e-30)
    return float(np.max(np.abs(a - b)) / scale)


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


@pytest.fixture(scope="module")
def fieldset():
    return inorganic_fieldset(["Na", "Cl", "K", "O"], charge_model="electronegativity", alpha=1.0)


@pytest.fixture(scope="module")
def nacl(fieldset):
    return crystal_fields(bulk("NaCl", "rocksalt", a=5.64, cubic=True), fieldset, "NaCl")


def chain_cell(sites, fieldset, n=12, step=1.9, name=""):
    """A row of sites along x, periodic in x alone, with an atom on the sites given."""
    pos = np.zeros((len(sites), 3))
    pos[:, 0] = np.asarray(sites, float) * step
    atoms = Atoms(["Na"] * len(sites), positions=pos,
                  cell=[[n * step, 0, 0], [0, 14.0, 0], [0, 0, 14.0]], pbc=(True, False, False))
    return crystal_fields(atoms, fieldset, name)


def generic_cell(fieldset, mirror=False, seed=3, name=""):
    """Four atoms of four species at positions with no symmetry: a cell with a handedness."""
    rng = np.random.RandomState(seed)
    atoms = Atoms(["Na", "Cl", "K", "O"], scaled_positions=rng.rand(4, 3),
                  cell=np.diag([6.0, 6.2, 5.8]), pbc=True)
    if mirror:
        p = atoms.get_positions()
        p[:, 0] *= -1.0
        atoms.set_positions(p)
        atoms.wrap()
    return crystal_fields(atoms, fieldset, name)


# --------------------------------------------------------------------------- #
# The neighbourhood of a centre
# --------------------------------------------------------------------------- #
def test_the_centre_is_in_its_own_neighbourhood(nacl):
    hoods = T.neighbourhoods(nacl, 4.0)
    assert len(hoods) == nacl.n_atoms
    for c, (X, W) in enumerate(hoods):
        assert np.allclose(X[0], 0.0)                       # the centre, at the vector zero
        assert np.allclose(W[0], nacl.weights[c])           # with its own field weights
        assert np.all(np.linalg.norm(X[1:], axis=1) <= 4.0 + 1e-9)


def test_the_neighbourhood_holds_the_periodic_images(nacl):
    """Rocksalt with a = 5.64 has 6 neighbours at 2.82 and 12 at 3.99 around every atom."""
    X, _ = T.neighbourhoods(nacl, 4.2)[0]
    r = np.sort(np.linalg.norm(X, axis=1))
    assert len(r) == 1 + 6 + 12
    assert np.allclose(r[1:7], 2.82, atol=1e-6)
    assert np.allclose(r[7:], 3.9881, atol=1e-3)


def test_a_layer_has_neighbours_in_two_directions_only(fieldset):
    layer = bulk("Na", "fcc", a=4.0, cubic=True)
    layer.set_pbc((True, True, False))
    cell = crystal_fields(layer, fieldset, "layer")
    X, _ = T.neighbourhoods(cell, 6.0)[0]
    assert len(X) > 1
    # nothing is reached across the third direction, so no neighbour sits a whole cell away in z
    assert np.max(np.abs(X[:, 2])) < 4.0


# --------------------------------------------------------------------------- #
# The invariances
# --------------------------------------------------------------------------- #
def test_a_supercell_gives_the_same_channels(nacl, fieldset):
    sup = crystal_fields(bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2)),
                         fieldset, "supercell")
    assert worst(T.threebody_autocorrelation(nacl, SPEC),
                 T.threebody_autocorrelation(sup, SPEC)) < 1e-12


def test_moving_the_origin_changes_nothing(fieldset):
    base = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    moved = base.copy()
    moved.positions += np.array([0.31, -1.7, 2.2])
    moved.wrap()
    assert worst(T.threebody_autocorrelation(crystal_fields(base, fieldset), SPEC),
                 T.threebody_autocorrelation(crystal_fields(moved, fieldset), SPEC)) < 1e-12


def test_turning_the_structure_changes_nothing(fieldset):
    base = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    th = 0.7
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0, 0, 1.0]])
    turned = base.copy()
    turned.set_cell(base.get_cell() @ R.T, scale_atoms=True)
    assert worst(T.threebody_autocorrelation(crystal_fields(base, fieldset), SPEC),
                 T.threebody_autocorrelation(crystal_fields(turned, fieldset), SPEC)) < 1e-12


def test_another_structure_does_change_the_channels(nacl, fieldset):
    other = crystal_fields(bulk("NaCl", "rocksalt", a=5.9, cubic=True), fieldset, "wider")
    assert worst(T.threebody_autocorrelation(nacl, SPEC),
                 T.threebody_autocorrelation(other, SPEC)) > 0.1


# --------------------------------------------------------------------------- #
# The first blind spot: two structures with the same distances
# --------------------------------------------------------------------------- #
def homometric_sets(n=12, k=5):
    """Two sets of sites on a ring of n with the same multiset of distances between them, and
    not related by a shift or a reflection of the ring."""
    def distances(s):
        return tuple(sorted(min((a - b) % n, (b - a) % n) for a in s for b in s if a != b))

    def orbit(s):
        return {tuple(sorted((sign * x + t) % n for x in s))
                for t in range(n) for sign in (1, -1)}

    by = {}
    for s in combinations(range(n), k):
        by.setdefault(distances(s), []).append(s)
    for sets in by.values():
        for a in sets:
            for b in sets:
                if b not in orbit(a):
                    return list(a), list(b)
    raise AssertionError("no homometric pair on this ring")


def test_the_two_point_channels_cannot_tell_a_homometric_pair_apart(fieldset):
    a, b = homometric_sets()
    spec = PeriodicSpec(r_max=12.0, step=0.25, sigma=0.25, taper=0.0, background=True)
    pa = periodic_autocorrelation(chain_cell(a, fieldset, name="A"), spec)
    pb = periodic_autocorrelation(chain_cell(b, fieldset, name="B"), spec)
    assert worst(pa, pb) < 1e-12
    assert cosine(pa, pb) > 1.0 - 1e-12


def test_the_three_body_channels_can(fieldset):
    a, b = homometric_sets()
    spec = T.ThreeBodySpec(r_max=6.0, step=0.5, sigma=0.35)
    ta = T.threebody_autocorrelation(chain_cell(a, fieldset, name="A"), spec)
    tb = T.threebody_autocorrelation(chain_cell(b, fieldset, name="B"), spec)
    assert worst(ta, tb) > 0.1
    assert cosine(ta, tb) < 0.99


# --------------------------------------------------------------------------- #
# The second blind spot: a mirror image
# --------------------------------------------------------------------------- #
def test_the_three_body_block_is_blind_to_a_mirror(fieldset):
    left = generic_cell(fieldset, mirror=False, name="L")
    right = generic_cell(fieldset, mirror=True, name="R")
    assert worst(T.threebody_autocorrelation(left, CHIRAL),
                 T.threebody_autocorrelation(right, CHIRAL)) < 1e-12


def test_the_handedness_block_changes_sign_under_a_mirror(fieldset):
    left = T.periodic_chirality(generic_cell(fieldset, mirror=False), CHIRAL)
    right = T.periodic_chirality(generic_cell(fieldset, mirror=True), CHIRAL)
    assert np.max(np.abs(left)) > 1.0                       # the block is not empty here
    assert np.max(np.abs(left + right)) < 1e-12 * np.max(np.abs(left))


def test_a_centrosymmetric_structure_has_no_handedness(nacl):
    channels = T.periodic_chirality(nacl, CHIRAL)
    assert len(channels) == len(chirality_triples(nacl.weights.shape[1])) * CHIRAL.n_shells
    assert np.max(np.abs(channels)) < 1e-20


def test_two_species_leave_the_handedness_block_empty(fieldset):
    """Weights that depend on the species alone span one vector per species, so two species give
    vectors in a plane and every determinant is zero. This is a property of the block, and what
    is left is the rounding of the arithmetic: the same cell with four species gives channels
    fifteen orders of magnitude larger."""
    fs = inorganic_fieldset(["Na", "Cl"], charge_model="electronegativity", alpha=1.0)
    rng = np.random.RandomState(5)
    atoms = Atoms(["Na", "Cl", "Na", "Cl"], scaled_positions=rng.rand(4, 3),
                  cell=np.diag([6.0, 6.2, 5.8]), pbc=True)
    two = T.periodic_chirality(crystal_fields(atoms, fs, "two species"), CHIRAL)
    four = T.periodic_chirality(generic_cell(fieldset), CHIRAL)
    assert np.max(np.abs(two)) < 1e-10 * np.max(np.abs(four))


def test_the_handedness_block_is_off_by_default(nacl):
    assert not SPEC.chirality
    assert len(T.threebody_features(nacl, SPEC)) == SPEC.n_channels(nacl.weights.shape[1])
    assert len(T.periodic_chirality(nacl, SPEC)) == 0


# --------------------------------------------------------------------------- #
# The layout: blocks, labels, widths
# --------------------------------------------------------------------------- #
def test_the_blocks_and_the_labels_line_up(nacl, fieldset):
    n_fields = nacl.weights.shape[1]
    for spec in (SPEC, CHIRAL):
        blocks = T.threebody_block_sizes(spec, n_fields)
        labels = T.threebody_channel_labels(fieldset, spec)
        row = T.threebody_features(nacl, spec)
        assert sum(blocks) == len(labels) == len(row) == spec.n_channels(n_fields)
        assert len(blocks) == len(field_pairs(n_fields)) + (
            len(chirality_triples(n_fields)) if spec.chirality else 0)
    assert periodic_channel_labels(fieldset, SPEC) == T.threebody_channel_labels(fieldset, SPEC)


def test_a_label_names_two_radii_and_a_distance(fieldset):
    labels = T.threebody_channel_labels(fieldset, SPEC)
    assert "@ 0.00" in labels[0] and "apart" in labels[0]
    assert any("handedness" in s for s in T.threebody_channel_labels(fieldset, CHIRAL))


def test_the_feature_matrix_has_one_row_per_structure(nacl, fieldset):
    other = crystal_fields(bulk("NaCl", "rocksalt", a=5.9, cubic=True), fieldset, "wider")
    X = T.threebody_feature_matrix([nacl, other], SPEC)
    assert X.shape == (2, SPEC.n_channels(nacl.weights.shape[1]))
    assert np.allclose(X[0], T.threebody_features(nacl, SPEC))
    assert T.threebody_feature_matrix([nacl], SPEC, dtype=np.float32).dtype == np.float32


def test_dropping_the_self_terms_changes_the_closest_distances(nacl):
    """The terms in which the two neighbours are one atom sit at distance zero, so dropping them
    empties the first distance channels and leaves the far ones as they were."""
    with_self = T.threebody_autocorrelation(nacl, SPEC)
    spec = T.ThreeBodySpec(r_max=4.0, step=0.5, sigma=0.35, include_self=False)
    without = T.threebody_autocorrelation(nacl, spec)
    assert with_self.shape == without.shape
    change = np.abs(with_self - without)
    distance_index = np.arange(len(change)) % SPEC.n_distances
    near = change[distance_index == 0]
    far = change[distance_index >= 4]                 # 2 angstrom away, which is 5.7 widths
    assert np.max(near) > 0.01 * np.max(np.abs(with_self))
    assert np.max(far) < 1e-6 * np.max(near)


# --------------------------------------------------------------------------- #
# A site of the cell as the sample (plan item M5)
# --------------------------------------------------------------------------- #
def test_a_site_record_gives_the_channels_of_that_site(nacl):
    import dataclasses

    rows = []
    for c in range(nacl.n_atoms):
        rows.append(T.threebody_autocorrelation(dataclasses.replace(nacl, centre=c), SPEC))
    assert worst(np.mean(rows, axis=0), T.threebody_autocorrelation(nacl, SPEC)) < 1e-12
    # the two species of rocksalt are different sites and give different channels
    assert worst(rows[0], rows[1]) > 1e-3


# --------------------------------------------------------------------------- #
# A model, its file and its reading
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_set(fieldset):
    cells, y = [], []
    rng = np.random.RandomState(0)
    for a, el in [(5.6, "NaCl"), (6.3, "KCl"), (5.7, "NaCl"), (6.4, "KCl"),
                  (5.8, "NaCl"), (6.5, "KCl")]:
        cells.append(crystal_fields(bulk(el, "rocksalt", a=a, cubic=True), fieldset, f"{el}{a}"))
        y.append(a + 0.05 * rng.randn())
    return AtomTable.from_molecules(cells), np.asarray(y, float)


def test_a_model_fits_predicts_and_survives_a_file(fieldset, small_set):
    table, y = small_set
    model = CMFModel(fieldset=fieldset, kernel_source="threebody", tb_r_max=4.0, tb_step=0.75)
    model.fit(table, y)
    first = model.predict(table)
    assert first.shape == y.shape and np.all(np.isfinite(first))
    again = CMFModel.from_state(model.state())
    assert again.kernel_source == "threebody" and again.tb_r_max == 4.0
    assert np.allclose(again.predict(table), first)


def test_a_model_with_the_handedness_block_has_one_block_per_triple(fieldset, small_set):
    table, y = small_set
    n_fields = table.weights.shape[1]
    model = CMFModel(fieldset=fieldset, kernel_source="threebody", tb_r_max=4.0, tb_step=0.75,
                     tb_chirality=True, tb_chirality_shells=3)
    model.fit(table, y)
    blocks = model._ac_blocks(n_fields)
    assert len(blocks) == len(model._ac_block_keys(n_fields))
    assert len(blocks) == len(field_pairs(n_fields)) + len(chirality_triples(n_fields))
    assert sum(blocks) == len(periodic_channel_labels(fieldset, model._ac_spec()))


def test_the_pattern_ranks_by_weight_times_spread(fieldset, small_set):
    table, _ = small_set
    X = T.threebody_feature_matrix(table.molecules, SPEC)
    labels = T.threebody_channel_labels(fieldset, SPEC)
    weights = np.zeros(X.shape[1])
    spread = T.channel_spread(X)
    moving = int(np.argmax(spread))
    still = int(np.argmin(spread))
    weights[moving] = 1.0
    weights[still] = 50.0                      # a large weight on a channel that does not move
    rows = T.pattern_table(weights, X, labels, top=2)
    assert rows[0]["channel"] == labels[moving]
    assert abs(rows[0]["effect"]) >= abs(rows[1]["effect"])


# --------------------------------------------------------------------------- #
# What the module refuses
# --------------------------------------------------------------------------- #
def test_a_disordered_cell_is_refused(nacl):
    import dataclasses

    shared = dataclasses.replace(nacl, site_species=[{"Na": 0.5, "K": 0.5}] * nacl.n_atoms)
    with pytest.raises(ValueError, match="more than one species"):
        T.threebody_autocorrelation(shared, SPEC)
    with pytest.raises(ValueError, match="more than one species"):
        T.periodic_chirality(shared, CHIRAL)


def test_structures_from_two_field_sets_are_refused(nacl, fieldset):
    other_fs = inorganic_fieldset(["Na", "Cl"], charge_model="electronegativity", alpha=1.0)
    other = crystal_fields(bulk("NaCl", "rocksalt", a=5.64, cubic=True), other_fs, "other")
    with pytest.raises(ValueError, match="one field set"):
        T.threebody_feature_matrix([nacl, other], SPEC, n_fields=nacl.weights.shape[1])


def test_more_channels_than_fields_are_refused(nacl):
    n_fields = nacl.weights.shape[1]
    with pytest.raises(ValueError, match="fewer than the channels"):
        T.threebody_autocorrelation(nacl, SPEC, pairs=[(0, n_fields + 2)])


def test_a_structure_with_no_atoms_gives_zeros(fieldset):
    empty = Atoms([], cell=np.diag([5.0, 5.0, 5.0]), pbc=True)
    cell = crystal_fields(empty, fieldset, "empty")
    row = T.threebody_autocorrelation(cell, SPEC)
    assert row.shape == (SPEC.n_channels(cell.weights.shape[1]),)
    assert not np.any(row)
