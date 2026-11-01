"""The reciprocal-space channels of a crystal: plan item M6.

What is tested, in the order the module builds it:

* the reciprocal lattice itself (the bound on the search, the lengths, the choice of cell);
* the identity that ties the real-space pair sum to the sum over the reciprocal lattice, which
  is the test of the two pieces of code against each other;
* the invariances: a supercell, a shift of the origin, a rotation, another primitive cell of the
  same lattice all give the same channels, exactly;
* what the channels add: two structures that a short real-space cutoff cannot tell apart, and
  that these channels can;
* the pieces around the descriptor: the labels, the feature matrix, the joint matrix, a fitted
  model, the reading of that model, and the cases the module refuses.
"""
import numpy as np
import pytest

from cmfqsar import reciprocal as R
from cmfqsar.invariant import channel_scale, field_pairs
from cmfqsar.model import CMFModel
from cmfqsar.periodic import (PeriodicSpec, crystal_fields, crystal_table, inorganic_fieldset,
                              periodic_autocorrelation, periodic_channel_labels)

bulk = pytest.importorskip("ase.build").bulk

SPEC = R.ReciprocalSpec(q_max=8.0, step=0.25, sigma=0.25, form_sigma=0.3)


@pytest.fixture(scope="module")
def fieldset():
    return inorganic_fieldset(["Na", "Cl", "Mg", "O", "Si"])


@pytest.fixture(scope="module")
def nacl(fieldset):
    return crystal_fields(bulk("NaCl", "rocksalt", a=5.64), fieldset, "NaCl")


def worst(a, b):
    """The largest difference between two descriptors, relative to the larger of the two."""
    s = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-300)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / s)


# --------------------------------------------------------------------------- #
# The reciprocal lattice
# --------------------------------------------------------------------------- #
def test_reciprocal_cell_is_the_dual_basis(nacl):
    B = R.reciprocal_cell(nacl.cell)
    assert np.allclose(np.asarray(nacl.cell, float) @ B.T, 2.0 * np.pi * np.eye(3), atol=1e-10)


def test_reciprocal_vectors_are_complete_and_inside_the_cutoff(nacl):
    for q_max in (3.0, 6.0, 9.0):
        G, q = R.reciprocal_vectors(nacl.cell, q_max)
        assert np.all(q <= q_max + 1e-9)
        assert np.all(q > 1e-9)                       # G = 0 is left out by default
        assert np.allclose(q, np.linalg.norm(G, axis=1))
        # every vector found by a deliberately wider search that is short enough is here
        wide, wq = R.reciprocal_vectors(nacl.cell, q_max + 4.0)
        inside = wide[wq <= q_max - 1e-9]
        assert len(inside) <= len(G)
        rows = {tuple(np.round(v, 6)) for v in G}
        assert all(tuple(np.round(v, 6)) in rows for v in inside)
    G0, q0 = R.reciprocal_vectors(nacl.cell, 3.0, include_zero=True)
    assert len(G0) == len(R.reciprocal_vectors(nacl.cell, 3.0)[0]) + 1
    assert float(q0.min()) == 0.0


def test_a_cell_without_volume_is_refused(nacl):
    flat = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no volume"):
        R.reciprocal_cell(flat)


def test_structure_factor_at_zero_is_the_sum_of_the_weights(nacl):
    S = R.structure_factors(nacl, np.zeros((1, 3)))
    assert np.allclose(S[0].real, np.asarray(nacl.weights, float).sum(axis=0))
    assert np.allclose(S[0].imag, 0.0)


# --------------------------------------------------------------------------- #
# The identity between the two descriptions (the Parseval test)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("width", [0.3, 0.5, 0.8])
def test_the_real_space_and_reciprocal_sums_agree(nacl, width):
    """The real-space lattice sum of a Gaussian equals the sum over the reciprocal lattice.

    The left side comes from the neighbour list of ``periodic.py`` and the right side from the
    structure factors of this module, so the agreement is evidence for both.
    """
    rec = R.parseval_residual(nacl, width)
    assert rec["worst"] < 1e-6
    assert np.all(np.isfinite(rec["real"])) and np.all(np.isfinite(rec["reciprocal"]))
    assert float(np.abs(rec["real"]).max()) > 0.0


def test_the_identity_holds_for_a_supercell_too(fieldset):
    cell = bulk("Si", "diamond", a=5.43)
    for rep in ((1, 1, 1), (2, 1, 1), (2, 2, 1)):
        c = crystal_fields(cell.repeat(rep), fieldset, f"Si {rep}")
        assert R.parseval_residual(c, 0.5)["worst"] < 1e-6


# --------------------------------------------------------------------------- #
# The invariances
# --------------------------------------------------------------------------- #
def test_the_choice_of_cell_does_not_change_the_channels(fieldset):
    """A supercell, another primitive cell and a shifted origin describe the same material.

    The descriptor is intensive by construction: the sum over the reciprocal lattice grows with
    the number of cells exactly as the expected number of lattice points does, and the extra
    points of a supercell carry no intensity.  This is exact, not approximate, because the one
    quantity that does not depend on G is added from its integral and not from the grid.
    """
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    base = R.reciprocal_autocorrelation(crystal_fields(atoms, fieldset, "NaCl"), SPEC)
    for rep in ((2, 1, 1), (2, 1, 3), (2, 2, 2)):
        sup = crystal_fields(atoms.repeat(rep), fieldset, f"sup {rep}")
        assert worst(base, R.reciprocal_autocorrelation(sup, SPEC)) < 1e-12
    conv = crystal_fields(bulk("NaCl", "rocksalt", a=5.64, cubic=True), fieldset, "conventional")
    assert worst(base, R.reciprocal_autocorrelation(conv, SPEC)) < 1e-12


def test_the_origin_and_the_orientation_do_not_change_the_channels(fieldset):
    atoms = bulk("MgO", "rocksalt", a=4.21)
    base = R.reciprocal_autocorrelation(crystal_fields(atoms, fieldset, "MgO"), SPEC)
    shifted = atoms.copy()
    shifted.positions += np.array([0.31, -1.7, 0.9])
    shifted.wrap()
    assert worst(base, R.reciprocal_autocorrelation(
        crystal_fields(shifted, fieldset, "shifted"), SPEC)) < 1e-12
    turned = atoms.copy()
    turned.rotate(37.0, "xyz", rotate_cell=True)
    assert worst(base, R.reciprocal_autocorrelation(
        crystal_fields(turned, fieldset, "turned"), SPEC)) < 1e-12


def test_a_different_structure_gives_different_channels(fieldset):
    a = R.reciprocal_autocorrelation(
        crystal_fields(bulk("NaCl", "rocksalt", a=5.64), fieldset, "B1"), SPEC)
    b = R.reciprocal_autocorrelation(
        crystal_fields(bulk("NaCl", "cesiumchloride", a=5.64 / 4 ** (1 / 3)), fieldset, "B2"),
        SPEC)
    assert worst(a, b) > 0.01


# --------------------------------------------------------------------------- #
# What the channels add over a real-space cutoff
# --------------------------------------------------------------------------- #
def _two_superstructures(fieldset, repeat, first, second):
    """One host cell with two substituted sites, at two separations along the long axis."""
    cell = bulk("Si", "diamond", a=5.43, cubic=True).repeat(repeat)
    out = []
    for other in (first, second):
        s = cell.copy()
        s.symbols[0] = "Mg"
        s.symbols[other] = "Mg"
        out.append(crystal_fields(s, fieldset, f"Mg at 0 and {other}"))
    return out


def test_the_channels_see_order_beyond_a_real_space_cutoff(fieldset):
    """Two structures that differ only in the distance between two substituted sites.

    The sites are 5.4 and 10.9 angstrom apart in a 21.7 angstrom cell.  Both separations are
    beyond a 4 angstrom cutoff, and the two structures have the same composition and the same
    local environment around every atom, so the real-space descriptor with that cutoff is
    identical for them.  The reciprocal channels are not, because a structure factor is a sum
    over the whole cell and has no cutoff at all.
    """
    a, b = _two_superstructures(fieldset, (4, 1, 1), 8, 16)
    short = PeriodicSpec(r_max=4.0, step=0.5, sigma=0.5)
    xa = periodic_autocorrelation(a, short)
    xb = periodic_autocorrelation(b, short)
    assert worst(xa, xb) < 1e-12                      # real space cannot tell them apart
    ya = R.reciprocal_autocorrelation(a, SPEC)
    yb = R.reciprocal_autocorrelation(b, SPEC)
    assert worst(ya, yb) > 1e-3                       # the reciprocal channels can


def test_the_channels_see_order_beyond_the_longest_cutoff_in_use(fieldset):
    """The same with separations of 10.9 and 21.7 angstrom, beyond a 10 angstrom cutoff as well.

    10 angstrom is the cutoff the benchmarks of this program use for crystals, so this is the
    case the real-space descriptor misses in practice.
    """
    a, b = _two_superstructures(fieldset, (8, 1, 1), 16, 32)
    for r_max in (4.0, 10.0):
        spec = PeriodicSpec(r_max=r_max, step=0.5, sigma=0.5)
        assert worst(periodic_autocorrelation(a, spec), periodic_autocorrelation(b, spec)) < 1e-12
    assert worst(R.reciprocal_autocorrelation(a, SPEC),
                 R.reciprocal_autocorrelation(b, SPEC)) > 1e-4


# --------------------------------------------------------------------------- #
# The spec
# --------------------------------------------------------------------------- #
def test_the_form_factor_is_the_resolution():
    """A wider field damps the high-q channels more: that is what makes the width a resolution."""
    q = np.linspace(0.0, 10.0, 11)
    narrow = R.ReciprocalSpec(form_sigma=0.2).form_factor(q)
    wide = R.ReciprocalSpec(form_sigma=0.6).form_factor(q)
    assert narrow[0] == pytest.approx(1.0) and wide[0] == pytest.approx(1.0)
    assert np.all(np.diff(narrow) < 0) and np.all(np.diff(wide) < 0)
    assert np.all(wide[1:] < narrow[1:])
    assert R.ReciprocalSpec(form_sigma=0.0).form_factor(q) == pytest.approx(np.ones_like(q))


def test_the_window_tapers_to_zero_at_the_cutoff():
    spec = R.ReciprocalSpec(q_max=8.0, taper=1.0)
    assert spec.window(np.array([0.0, 5.0, 6.9])) == pytest.approx(np.ones(3))
    assert 0.0 < float(spec.window(np.array([7.5]))[0]) < 1.0
    assert float(spec.window(np.array([8.0]))[0]) == 0.0
    assert float(R.ReciprocalSpec(taper=0.0).window(np.array([7.999]))[0]) == 1.0


def test_the_form_average_is_near_the_form_factor_at_the_channel():
    """The mean of the form factor over a channel is close to its value at the centre."""
    spec = R.ReciprocalSpec(q_max=8.0, step=0.5, sigma=0.25, form_sigma=0.3)
    avg, point = spec.form_average(), spec.form_factor(spec.centres)
    inside = (spec.centres > 0.5) & (spec.centres < 7.0)
    assert np.allclose(avg[inside], point[inside], rtol=0.25)
    assert np.all(avg > 0) and np.all(avg <= 1.0 + 1e-12)


def test_the_spacing_of_a_channel_is_two_pi_over_q():
    spec = R.ReciprocalSpec(q_max=4.0, step=1.0)
    d = R.spacings(spec)
    assert np.isinf(d[0])
    assert d[1] == pytest.approx(2.0 * np.pi)
    assert d[-1] == pytest.approx(2.0 * np.pi / 4.0)


def test_the_blocks_and_the_labels_line_up(fieldset, nacl):
    n_fields = nacl.weights.shape[1]
    x = R.reciprocal_autocorrelation(nacl, SPEC)
    blocks = R.reciprocal_block_sizes(SPEC, n_fields)
    labels = R.reciprocal_channel_labels(fieldset, SPEC)
    assert sum(blocks) == x.size == len(labels)
    assert len(blocks) == len(field_pairs(n_fields))
    assert all(b == SPEC.n_bins for b in blocks)
    assert labels[0].endswith("@ q 0.00 1/A")
    assert "x" in labels[-1]                           # the last block is a pair of two types
    # the shared helper of periodic.py hands a reciprocal spec to this module
    assert periodic_channel_labels(fieldset, SPEC) == labels


# --------------------------------------------------------------------------- #
# The self-scattering term and partial occupancy
# --------------------------------------------------------------------------- #
def test_dropping_the_self_term_changes_only_a_smooth_curve(nacl):
    """Without the self-scattering term the channels are the distinct scattering.

    The difference between the two is the composition times the mean form factor, which is a
    smooth curve and not a feature of the structure.
    """
    with_self = R.reciprocal_autocorrelation(nacl, SPEC)
    spec = R.ReciprocalSpec(q_max=SPEC.q_max, step=SPEC.step, sigma=SPEC.sigma,
                            form_sigma=SPEC.form_sigma, include_self=False)
    without = R.reciprocal_autocorrelation(nacl, spec)
    n_pairs = len(field_pairs(nacl.weights.shape[1]))
    diff = (with_self - without).reshape(n_pairs, SPEC.n_bins)
    W = np.asarray(nacl.weights, float)
    C = W.T @ W
    fi = np.array([f for f, _ in field_pairs(W.shape[1])], int)
    gi = np.array([g for _, g in field_pairs(W.shape[1])], int)
    mult = np.where(fi == gi, 1.0, 2.0)
    expected = (mult * C[fi, gi])[:, None] * SPEC.form_average()[None, :] / nacl.n_atoms
    assert np.allclose(diff, expected, atol=1e-12)


def test_the_background_is_the_term_at_g_equals_zero(nacl):
    """Keeping G = 0 adds exactly the forward scattering, which is the uniform background."""
    spec = R.ReciprocalSpec(q_max=4.0, step=0.5, sigma=0.5, form_sigma=0.3)
    kept = R.ReciprocalSpec(q_max=4.0, step=0.5, sigma=0.5, form_sigma=0.3, include_zero=True)
    diff = (R.reciprocal_autocorrelation(nacl, kept)
            - R.reciprocal_autocorrelation(nacl, spec))
    # A_fg(0) is the product of the summed weights; it enters the lowest channels only, since a
    # channel at q is a Gaussian about q and G = 0 is 1/sigma widths away from the rest.
    assert float(np.abs(diff).max()) > 0.0
    n_pairs = len(field_pairs(nacl.weights.shape[1]))
    rows = diff.reshape(n_pairs, spec.n_bins)
    assert float(np.abs(rows[:, 0]).max()) > float(np.abs(rows[:, -1]).max())


def test_a_disordered_cell_uses_its_second_moments(fieldset):
    """A site shared by two species carries the mean of the product of the weights.

    The self-scattering term is the one place where the mean of a product is not the product of
    the means, so it is the one place a disordered cell differs from a cell of mean weights.
    """
    from cmfqsar.periodic import crystal_fields as cf

    atoms = bulk("MgO", "rocksalt", a=4.21)
    atoms.info["site_species"] = [{"Mg": 0.5, "Na": 0.5}, {"O": 1.0}]
    atoms.arrays["occupancy"] = np.array([1.0, 1.0])
    fs = inorganic_fieldset(["Mg", "Na", "O"])
    dis = cf(atoms, fs, "half Na")
    x = R.reciprocal_autocorrelation(dis, SPEC)
    assert dis.second_moments is not None
    assert np.all(np.isfinite(x))
    mean_only = cf(bulk("MgO", "rocksalt", a=4.21), fs, "MgO")
    assert worst(x, R.reciprocal_autocorrelation(mean_only, SPEC)) > 1e-6
    # the descriptor of the disordered cell is still independent of the choice of cell
    big = atoms.repeat((2, 1, 1))
    big.info["site_species"] = [{"Mg": 0.5, "Na": 0.5}, {"O": 1.0}] * 2
    big.arrays["occupancy"] = np.ones(4)
    assert worst(x, R.reciprocal_autocorrelation(cf(big, fs, "big"), SPEC)) < 1e-12


# --------------------------------------------------------------------------- #
# What the module refuses
# --------------------------------------------------------------------------- #
def test_a_layer_is_refused(fieldset):
    atoms = bulk("MgO", "rocksalt", a=4.21, cubic=True)
    atoms.pbc = (True, True, False)
    layer = crystal_fields(atoms, fieldset, "layer")
    with pytest.raises(ValueError, match="all three directions"):
        R.reciprocal_autocorrelation(layer, SPEC)


def test_a_site_record_is_refused(fieldset, nacl):
    from cmfqsar.sites import site_fields

    with pytest.raises(ValueError, match="site record"):
        R.reciprocal_autocorrelation(site_fields(nacl, 0), SPEC)


def test_a_set_built_from_two_field_sets_is_refused(fieldset):
    a = crystal_fields(bulk("NaCl", "rocksalt", a=5.64), fieldset, "a")
    b = crystal_fields(bulk("NaCl", "rocksalt", a=5.64), inorganic_fieldset(["Na"]), "b")
    with pytest.raises(ValueError, match="one field set"):
        R.reciprocal_feature_matrix([a, b], SPEC, n_fields=a.weights.shape[1])


# --------------------------------------------------------------------------- #
# The feature matrix, the joint matrix and a fitted model
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def small_set(fieldset):
    """Six crystals, their field table and a target that depends on the structure."""
    structures = {"NaCl": bulk("NaCl", "rocksalt", a=5.64),
                  "MgO": bulk("MgO", "rocksalt", a=4.21),
                  "NaCl-big": bulk("NaCl", "rocksalt", a=5.90),
                  "CsCl-like": bulk("NaCl", "cesiumchloride", a=3.55),
                  "Si": bulk("Si", "diamond", a=5.43),
                  "Si-big": bulk("Si", "diamond", a=5.70)}
    names = list(structures)
    table = crystal_table([structures[k] for k in names], fieldset, names)
    y = np.array([float(abs(np.linalg.det(np.asarray(m.cell, float)))) for m in table.molecules])
    return table, y


def test_the_feature_matrix_has_one_row_per_structure(small_set):
    table, _ = small_set
    F = R.reciprocal_feature_matrix(table.molecules, SPEC,
                                    n_fields=table.weights.shape[1])
    assert F.shape == (len(table.molecules), sum(R.reciprocal_block_sizes(
        SPEC, table.weights.shape[1])))
    assert np.all(np.isfinite(F))
    for k, m in enumerate(table.molecules):
        assert np.allclose(F[k], R.reciprocal_autocorrelation(m, SPEC))
    seen = []
    R.reciprocal_feature_matrix(table.molecules, SPEC, n_fields=table.weights.shape[1],
                               progress=lambda msg, frac: seen.append(msg))
    assert seen and "Reciprocal" in seen[0]


def test_the_joint_matrix_is_the_two_descriptions_side_by_side(small_set, fieldset):
    table, _ = small_set
    real = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)
    X, widths = R.joint_features(table.molecules, real, SPEC,
                                 n_fields=table.weights.shape[1])
    assert X.shape[1] == sum(widths)
    labels = R.joint_labels(fieldset, real, SPEC)
    assert len(labels) == X.shape[1]
    n_pairs = len(field_pairs(table.weights.shape[1]))
    assert np.allclose(X[0, :n_pairs * real.n_bins],
                       periodic_autocorrelation(table.molecules[0], real))
    assert np.allclose(X[0, n_pairs * real.n_bins:],
                       R.reciprocal_autocorrelation(table.molecules[0], SPEC))
    # each block keeps its own scale, so neither description dominates by its units
    s = channel_scale(X, SPEC.n_bins, blocks=widths)
    assert s.shape == (X.shape[1],)
    assert np.all(s >= 0)


def test_a_model_fits_predicts_and_reads_back(small_set, fieldset):
    table, y = small_set
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-6},
                     kernel_source="reciprocal", normalize=True)
    model.rec_q_max, model.rec_step, model.rec_form_sigma = 8.0, 0.5, 0.3
    model.fit(table, y)
    assert model.is_reciprocal and not model.uses_fields
    p = model.predict(table)
    assert p.shape == y.shape and np.all(np.isfinite(p))
    assert np.corrcoef(p, y)[0, 1] > 0.5
    spec = model._ac_spec()
    assert isinstance(spec, R.ReciprocalSpec)
    assert spec.q_max == 8.0 and spec.step == 0.5
    state = CMFModel.from_state(model.state())
    assert state.kernel_source == "reciprocal"
    assert state.rec_q_max == 8.0 and state.rec_step == 0.5 and state.rec_form_sigma == 0.3
    assert np.allclose(state.predict(table), p, atol=1e-10)


def test_the_pattern_of_a_fitted_model_is_one_number_per_channel(small_set, fieldset):
    table, y = small_set
    model = CMFModel(fieldset=fieldset, method="krr", params={"lam": 1e-6},
                     kernel_source="reciprocal", normalize=True)
    model.rec_step = 0.5
    model.fit(table, y)
    P = R.reciprocal_pattern(model, len(fieldset.specs))
    assert P.shape == (len(field_pairs(len(fieldset.specs))), model._ac_spec().n_bins)
    assert np.all(np.isfinite(P)) and float(np.abs(P).max()) > 0.0
    top = R.pattern_table(model, fieldset, n=5)
    assert len(top) == 5
    assert all(0.0 <= q <= model.rec_q_max for _, _, q in top)
    assert abs(top[0][1]) >= abs(top[-1][1])
    assert "1/A" in top[0][0]
