"""A migration path as the sample (plan item BT7, with the periodic environment and hop frame).

What these tests fix: that the environment of a point in a periodic cell does not depend on the
cell chosen to describe the crystal, its origin or its orientation; that a hop's frame puts the
two sites on the z axis and the averaged poses make the kernel blind to how the crystal was
turned; that the enumeration finds the hops a lattice has and the symmetry reduction counts them
right; that the bottleneck is where the chemistry says; that the percolation step reads the
dimensionality of a network correctly through the cell boundary; and that the folds hold out
whole structures.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("ase")
pytest.importorskip("pymatgen.core")

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from cmfqsar.hops import (CENTRE_SYMBOL, Hop, HopSet, bottleneck, enumerate_hops,  # noqa: E402
                          fingerprint, framed_environments, hop_axis_frame,
                          hop_count_features, hop_environment, hop_frames, hop_types, path_barrier,
                          percolation_thresholds, point_environment, posed_hop,
                          read_hop_labels, reduce_by_symmetry, spread_labels,
                          structure_percolation, write_hops)
from cmfqsar.invariant import autocorrelation  # noqa: E402
from cmfqsar.localenv import LOCAL_SPEC, envelope_radius  # noqa: E402
from cmfqsar.periodic import crystal_fields, inorganic_fieldset  # noqa: E402


@pytest.fixture(scope="module")
def world():
    fs = inorganic_fieldset(["Li", "Cl"], charge_model="oxidation")
    licl = bulk("LiCl", "rocksalt", a=5.13)
    li = bulk("Li", "bcc", a=3.5)
    return fs, licl, li


def _rotated(atoms, angle=0.6, shift=(1.1, -0.7, 2.3)):
    """The same crystal turned and moved: nothing physical has changed."""
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ \
        np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    out = atoms.copy()
    out.set_cell(np.asarray(atoms.get_cell()[:]) @ R.T, scale_atoms=False)
    out.set_positions(atoms.get_positions() @ R.T + np.asarray(shift))
    return out, R, np.asarray(shift)


# ------------------------------------------------------------------- the periodic environment


def test_the_environment_has_every_image_within_reach_and_the_marker_first(world):
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    env = point_environment(c, [0.0, 0.0, 0.0], rho=0.1)
    reach = envelope_radius(0.1)
    assert env.symbols[0] == CENTRE_SYMBOL and np.allclose(env.coords[0], 0)
    assert np.all(env.weights[0] == 0)
    d = np.linalg.norm(env.coords[1:], axis=1)
    assert d.max() <= reach + 1e-9 and np.all(np.diff(d) >= -1e-9)
    # brute force: a big supercell, the same sphere
    big = licl.repeat((9, 9, 9))
    pos = big.get_positions() - np.asarray(big.get_cell()[:]).sum(axis=0) * (4 / 9)
    inside = np.linalg.norm(pos, axis=1) <= reach
    assert inside.sum() == env.n_atoms - 1
    assert env.member_atom_map[0] == -1 and set(env.member_atom_map[1:]) <= {0, 1}


def test_a_supercell_an_origin_shift_and_a_rotation_change_nothing(world):
    fs, licl, _ = world
    c1 = crystal_fields(licl, fs)
    c2 = crystal_fields(licl.repeat((2, 1, 3)), fs)
    centre = np.array([0.7, -0.4, 1.9])
    e1 = point_environment(c1, centre, rho=0.1)
    e2 = point_environment(c2, centre + np.asarray(licl.get_cell()[:])[1] * 2, rho=0.1)
    f1 = autocorrelation(e1, LOCAL_SPEC)
    f2 = autocorrelation(e2, LOCAL_SPEC)
    assert e1.n_atoms == e2.n_atoms
    assert np.allclose(f1, f2, rtol=1e-8, atol=1e-10)
    turned, R, shift = _rotated(licl)
    c3 = crystal_fields(turned, fs)
    e3 = point_environment(c3, centre @ R.T + shift, rho=0.1)
    assert e3.n_atoms == e1.n_atoms
    assert np.allclose(np.sort(np.linalg.norm(e3.coords, axis=1)),
                       np.sort(np.linalg.norm(e1.coords, axis=1)))
    assert np.allclose(autocorrelation(e3, LOCAL_SPEC), f1, rtol=1e-8, atol=1e-10)


def test_excluding_the_migrating_ion_removes_exactly_two_atoms(world):
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    hops = enumerate_hops(licl, "Li")
    h = hops[0]
    full = point_environment(c, h.midpoint, rho=0.1)
    env = hop_environment(c, h, rho=0.1)
    assert env.n_atoms == full.n_atoms - 2
    d = np.linalg.norm(env.coords[1:], axis=1)
    # the two sites sit at half the hop length from the midpoint, and no Li is left there (the
    # two Cl of the bottleneck happen to sit at the same distance in rocksalt)
    there = np.abs(d - 0.5 * h.length) < 1e-6
    assert there.sum() == 2 and all(env.symbols[1 + i] == "Cl" for i in np.where(there)[0])
    # the reverse hop excludes the same two ions
    rev = hop_environment(c, h.reversed(), rho=0.1)
    assert rev.n_atoms == env.n_atoms
    assert np.allclose(np.sort(np.linalg.norm(rev.coords, axis=1)), np.sort(d.tolist() + [0.0]))


# ----------------------------------------------------------------------------- the hop frame


def test_the_frame_is_orthonormal_with_z_along_the_hop():
    R = hop_axis_frame([1.0, 2.0, 2.0], reference=[0.0, 0.0, 1.0])
    assert np.allclose(R @ R.T, np.eye(3)) and np.linalg.det(R) > 0
    assert np.allclose(R[2], [1 / 3, 2 / 3, 2 / 3])
    R2 = hop_axis_frame([0.0, 0.0, 1.0], reference=[0.0, 0.0, 2.0])     # parallel: the fallback
    assert np.allclose(R2 @ R2.T, np.eye(3)) and np.allclose(R2[2], [0, 0, 1])
    with pytest.raises(ValueError):
        hop_axis_frame([0.0, 0.0, 0.0])


def test_the_posed_environment_puts_the_sites_on_z_and_averages_the_rotation(world):
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    h = enumerate_hops(licl, "Li")[0]
    env = hop_environment(c, h, rho=0.1)
    frames = hop_frames(env, h.vector, 6)
    assert len(frames) == 6
    v = np.asarray(h.vector, float)
    for R in frames:
        assert np.allclose(R @ v, [0, 0, h.length])
    p = posed_hop(env, h.vector, 6)
    assert p.n_conformers == 6 and p.n_atoms == 6 * env.n_atoms
    # the weights of every pose add up to the weights of the environment
    assert np.allclose(p.weights.reshape(6, env.n_atoms, -1).sum(axis=0), env.weights)


def test_turning_the_crystal_leaves_the_posed_kernel_unchanged(world):
    from cmfqsar.localalign import overlap_kernel
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    h = enumerate_hops(licl, "Li")[0]
    turned, R, shift = _rotated(licl)
    ct = crystal_fields(turned, fs)
    ht = Hop(0, h.a, h.b, h.offset, start=h.start @ R.T + shift, vector=h.vector @ R.T)
    p1 = posed_hop(hop_environment(c, h, rho=0.1), h.vector, 12)
    p2 = posed_hop(hop_environment(ct, ht, rho=0.1), ht.vector, 12)
    _model, _table, K = overlap_kernel([p1, p2], fs)
    assert K[0, 1] == pytest.approx(K[0, 0], abs=1e-6)       # the joint kernel is a sum over fields


# --------------------------------------------------------- enumeration, symmetry, bottleneck


def test_bcc_lithium_has_fourteen_hops_of_two_kinds(world):
    fs, _, li = world
    hops = enumerate_hops(li, "Li", d_min=1.5, d_max=4.5)
    assert len(hops) == 14
    lengths = sorted(round(h.length, 3) for h in hops)
    assert lengths[:8] == [pytest.approx(3.5 * np.sqrt(3) / 2, abs=1e-3)] * 8
    assert lengths[8:] == [pytest.approx(3.5, abs=1e-3)] * 6
    hs = HopSet.build([li], ["Li bcc"], "Li")
    reps = hs.annotate(hs.crystals(fs))
    assert len(reps) == 2
    assert sorted(hs.hops[r].multiplicity for r in reps) == [6, 8]
    assert all(h.representative in reps for h in hs.hops)
    assert "14 hops" in hs.describe() and "2 distinct" in hs.describe()


def test_a_hop_and_its_reverse_differ_unless_the_sites_are_equivalent(world):
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    h = enumerate_hops(licl, "Li")[0]
    e = hop_environment(c, h, rho=0.1)
    r = h.reversed()
    er = hop_environment(c, r, rho=0.1)
    assert fingerprint(e, h.vector) == fingerprint(er, r.vector)   # rocksalt: equivalent sites
    # the reverse hop starts from b in the home cell, so its midpoint is the image of the
    # original one, one lattice shift away
    assert np.allclose(r.midpoint, h.midpoint - h.shift) and np.allclose(r.vector, -h.vector)


def test_the_bottleneck_of_rocksalt_is_two_anions(world):
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    h = enumerate_hops(licl, "Li")[0]
    e = hop_environment(c, h, rho=0.1)
    d, at, atoms = bottleneck(e, h.vector)
    assert d == pytest.approx(5.13 / (2 * np.sqrt(2)), abs=0.02)
    assert at == pytest.approx(0.5, abs=0.03)
    assert atoms == ["Cl", "Cl"]


def test_the_count_reference_counts_what_is_around_the_midpoint(world):
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    hs = HopSet.build([licl], ["LiCl"], "Li")
    envs = hs.environments(hs.crystals(fs), rho=0.1)
    types = hop_types(envs, min_atoms=5)
    assert types == ["Cl", "Li"]
    X = hop_count_features(envs, types)
    assert X.shape[0] == len(envs) and np.all(X[:, :3] == [0, 0, 1])   # the marker's own type
    assert X[0, 3:].sum() == envs[0].n_atoms - 1 - np.sum(
        np.linalg.norm(envs[0].coords[1:], axis=1) >= 8.5)


# ---------------------------------------------------------------------- labels and targets


def test_the_targets_index_the_hops_and_the_folds_hold_out_structures(world):
    fs, licl, li = world
    hs = HopSet.build([licl, li], ["LiCl", "Li"], "Li")
    for k, h in enumerate(hs.hops):
        h.y = float(k)
    t = hs.targets()
    assert len(t) == len(hs.hops) and t.n_mol == 2
    assert set(t.molecule.tolist()) == {0, 1}
    for part in t.folds(2, seed=0):
        assert len(set(t.molecule[part].tolist())) == 1
    sub = hs.targets(rows=hs.of_structure(1), which="reverse")
    assert len(sub) == 0                          # no reverse labels were set


def test_labels_read_from_a_table_land_on_the_right_hop(world, tmp_path):
    fs, licl, li = world
    hs = HopSet.build([li], ["Li bcc"], "Li")
    path = tmp_path / "labels.csv"
    path.write_text("structure,a,b,offset_x,offset_y,offset_z,barrier,barrier_reverse\n"
                    "Li bcc,0,0,1,0,0,0.42,0.42\n"
                    "0,0,0,1,1,1,0.31,\n")
    n = read_hop_labels(str(path), hs)
    assert n == 2
    by = {h.offset: h for h in hs.hops}
    assert by[(1, 0, 0)].y == 0.42 and by[(1, 0, 0)].y_reverse == 0.42
    assert by[(1, 1, 1)].y == 0.31 and np.isnan(by[(1, 1, 1)].y_reverse)
    (tmp_path / "bad.csv").write_text("structure,a,b,barrier\nLi bcc,0,3,0.1\n")
    with pytest.raises(ValueError, match="no hop"):
        read_hop_labels(str(tmp_path / "bad.csv"), hs)
    out = tmp_path / "hops.csv"
    assert write_hops(str(out), hs, {0: 0.5}) == 14
    assert "barrier" in out.read_text().splitlines()[0]


def test_spreading_fills_the_equivalent_hops_and_the_reverse_ones(world):
    fs, _, li = world
    hs = HopSet.build([li], ["Li bcc"], "Li")
    reps = hs.annotate(hs.crystals(fs))
    for r in reps:
        hs.hops[r].y, hs.hops[r].y_reverse = 0.3 + 0.1 * r, 0.3 + 0.1 * r
    filled = spread_labels(hs)
    assert filled == 14 - len(reps)
    assert all(np.isfinite(h.y) for h in hs.hops)


# ------------------------------------------------------------------- the barrier on a landscape


def test_the_minimax_barrier_on_a_slab_landscape():
    """A ridge across the path that cannot be walked around: the barrier is its height."""
    start, vec = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 4.0])
    mid = start + 0.5 * vec

    def energy(pts):
        z = (np.asarray(pts) - mid) @ np.array([0.0, 0.0, 1.0])
        return 0.7 * np.exp(-(z ** 2) / 0.64)

    saddle, ea, eb = path_barrier(energy, start, vec, spacing=0.2, pad=2.0)
    assert ea < 0.01 and eb < 0.01
    assert saddle == pytest.approx(0.7, abs=0.02)


def test_a_detour_lowers_the_barrier_when_there_is_one():
    """A bump on the axis only: the minimax path goes around it."""
    start, vec = np.zeros(3), np.array([0.0, 0.0, 4.0])
    mid = start + 0.5 * vec

    def energy(pts):
        d2 = np.sum((np.asarray(pts) - mid) ** 2, axis=1)
        return 1.0 * np.exp(-d2 / 0.5)

    saddle, _ea, _eb = path_barrier(energy, start, vec, spacing=0.2, pad=2.5)
    assert saddle < 0.2


# ----------------------------------------------------------------------------- percolation


def _hop(a, b, off):
    return Hop(0, a, b, tuple(off), start=np.zeros(3), vector=np.asarray(off, float))


def test_percolation_counts_independent_directions_through_the_boundary():
    hops = [_hop(0, 0, (1, 0, 0)), _hop(0, 0, (0, 1, 0)), _hop(0, 0, (0, 0, 1))]
    out = percolation_thresholds(1, hops, [0.1, 0.2, 0.3])
    assert out == {"E_1D": 0.1, "E_2D": 0.2, "E_3D": 0.3}
    out = percolation_thresholds(1, hops[:1], [0.1])
    assert out["E_1D"] == 0.1 and np.isinf(out["E_2D"]) and np.isinf(out["E_3D"])
    # two sites, a zigzag chain along x: 0 -> 1 in the cell, 1 -> 0 in the next cell
    chain = [_hop(0, 1, (0, 0, 0)), _hop(1, 0, (1, 0, 0))]
    out = percolation_thresholds(2, chain, [0.2, 0.5])
    assert out["E_1D"] == 0.5 and np.isinf(out["E_2D"])
    # a diagonal and an axis together span two directions
    out = percolation_thresholds(1, [_hop(0, 0, (1, 1, 0)), _hop(0, 0, (1, 0, 0))], [0.3, 0.4])
    assert out["E_1D"] == 0.3 and out["E_2D"] == 0.4


def test_the_direction_of_a_hop_does_not_matter_for_percolation():
    forward = [_hop(0, 0, (1, 0, 0)), _hop(0, 0, (-1, 0, 0))]
    out = percolation_thresholds(1, forward, [0.3, 0.3])
    assert out["E_1D"] == 0.3 and np.isinf(out["E_2D"])


def test_structure_percolation_reads_the_saddles_relative_to_the_lowest_site(world):
    fs, _, li = world
    hs = HopSet.build([li], ["Li bcc"], "Li")
    for h in hs.hops:
        h.site_a, h.site_b = 1.0, 1.0
        h.saddle = 1.0 + (0.2 if abs(h.length - 3.5) > 0.1 else 0.6)
    out = structure_percolation(hs, 0)
    assert out == {"E_1D": pytest.approx(0.2), "E_2D": pytest.approx(0.2),
                   "E_3D": pytest.approx(0.2)}


# --------------------------------------------------------------- the bond-valence landscape


def test_the_bond_valence_barrier_of_rocksalt_is_finite_and_symmetric(world):
    pytest.importorskip("bvlain")
    from cmfqsar.hops import bvse_barriers
    fs, licl, _ = world
    hs = HopSet.build([licl], ["LiCl"], "Li")
    reps = hs.annotate(hs.crystals(fs))
    perc = bvse_barriers(hs, rows=reps, resolution=0.3)
    h = hs.hops[reps[0]]
    assert np.isfinite(h.y) and h.y > 0
    assert h.y == pytest.approx(h.y_reverse, abs=0.05)
    assert all(np.isfinite(x.y) for x in hs.hops)            # spread to the equivalent ones
    assert "LiCl" in perc


# ------------------------------------------------- the rotation average in closed form


def test_the_closed_form_equals_many_finite_poses(world):
    from cmfqsar.fields import AtomTable
    from cmfqsar.hops import HopModel, cylindrical_blocks
    from cmfqsar.kernels import kernel_matrices
    from cmfqsar.localalign import pose
    fs, licl, li = world
    c1, c2 = crystal_fields(licl, fs), crystal_fields(li, fs)
    pairs = [(c1, enumerate_hops(licl, "Li")[0]), (c2, enumerate_hops(li, "Li")[0]),
             (c2, enumerate_hops(li, "Li")[-1])]
    envs = [hop_environment(c, h, rho=0.25, r_max=4.5) for c, h in pairs]
    single = [pose(e, hop_frames(e, h.vector, 1)[0]) for e, (_c, h) in zip(envs, pairs)]
    K = cylindrical_blocks(single, single, fs.specs)
    posed = [posed_hop(e, h.vector, 96) for e, (_c, h) in zip(envs, pairs)]
    T = AtomTable.from_molecules(posed)
    Kp = kernel_matrices(T, T, fs.specs)
    assert K.shape == Kp.shape
    for f in range(K.shape[0]):
        scale = max(float(np.abs(Kp[f]).max()), 1e-12)
        assert np.allclose(K[f], Kp[f], atol=1e-6 * scale)
    # one plain field against averaged ones is the same kernel: the average is a projection
    Kx = kernel_matrices(AtomTable.from_molecules(single), T, fs.specs)
    assert np.allclose(Kx[0], Kp[0], atol=1e-6 * float(np.abs(Kp[0]).max()))
    m = HopModel.build(single, fs)
    # the joint kernel is the sum of the normalised field kernels, so its diagonal counts the
    # fields that are not empty: an element missing from a structure contributes nothing
    for j in range(len(single)):
        assert m.K[j, j] == pytest.approx(int(np.sum(m.diag[:, j] > 0)))


def test_turning_the_crystal_leaves_the_closed_form_kernel_unchanged(world):
    from cmfqsar.hops import HopModel
    fs, licl, _ = world
    c = crystal_fields(licl, fs)
    h = enumerate_hops(licl, "Li")[0]
    turned, R, shift = _rotated(licl)
    ct = crystal_fields(turned, fs)
    ht = Hop(0, h.a, h.b, h.offset, start=h.start @ R.T + shift, vector=h.vector @ R.T,
             shift=h.shift @ R.T)
    hs1 = HopSet([licl], ["a"], [h])
    hs2 = HopSet([turned], ["b"], [ht])
    e1 = framed_environments(hs1, [c], rho=0.1)
    e2 = framed_environments(hs2, [ct], rho=0.1)
    m = HopModel.build(e1 + e2, fs)
    assert m.K[0, 1] == pytest.approx(m.K[0, 0], abs=1e-8)


def test_the_hop_model_fits_predicts_and_splits_over_atoms(world):
    from cmfqsar.hops import HopModel, cross_validate_hop_model
    fs, licl, li = world
    wide = bulk("LiCl", "rocksalt", a=5.6)
    hs = HopSet.build([licl, li, wide], ["LiCl", "Li", "LiCl wide"], "Li")
    crystals = hs.crystals(fs)
    reps = hs.annotate(crystals)
    rows = [r for r in reps]
    envs = framed_environments(hs, crystals, rows=rows, rho=0.25, r_max=5.0)
    y = np.array([hs.hops[r].length for r in rows])
    m = HopModel.build(envs, fs).fit(y, lam=1e-6)
    assert np.allclose(m.predict(envs), y, atol=1e-3)
    a = m.atom_contributions(envs[0])
    assert a[0] == 0.0                                      # the marker carries no field
    assert float(a.sum()) == pytest.approx(float(m.predict([envs[0]])[0] - m.bias), abs=1e-8)
    v, sd = m.profile(0, [-1.0, 0.0, 1.0], [0.0, 2.0], sigma=0.1)
    assert v.shape == (3, 2) and sd.shape == (3, 2) and np.all(sd >= 0)
    for r in rows:
        hs.hops[r].y = hs.hops[r].length
    t = hs.targets(rows)
    res = cross_validate_hop_model(m, t, folds=2)
    assert res.n_samples == len(rows) and np.isfinite(res.q2)


# ------------------------------------------------------------------------ the command line


def test_the_command_line_lists_the_hops_of_a_structure(tmp_path, capsys):
    from ase.io import write

    from cmfqsar.cli import main
    path = str(tmp_path / "li.extxyz")
    write(path, bulk("Li", "bcc", a=3.5), format="extxyz")
    main(["hops", path, "--list"])
    text = capsys.readouterr().out
    assert "14 hops" in text and "2 distinct" in text
    assert "bottleneck" in text
