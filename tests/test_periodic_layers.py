"""Tests for layers and chains in the periodic field autocorrelation (plan item M1): the
background per area or repeat length, the images that respect the periodicity flags, and the
height channels of a layer."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ase = pytest.importorskip("ase")
pytest.importorskip("pymatgen.core")

from ase import Atoms  # noqa: E402
from ase.build import bulk, mx2  # noqa: E402

from cmfqsar.fields import AtomTable  # noqa: E402
from cmfqsar.invariant import field_pairs  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.periodic import (CrystalFields, PeriodicSpec, cell_images, centre_layer,  # noqa: E402
                              crystal_fields, crystal_table, image_translations,
                              inorganic_fieldset, lattice_spread, layer_normal,
                              periodic_autocorrelation, periodic_block_sizes,
                              periodic_channel_labels, periodic_measure,
                              periodic_overlap_kernels, periodic_self_overlaps,
                              slab_autocorrelation)


def layer(vacuum=10.0, kind="MoS2", a=3.18, pbc=(True, True, False)):
    s = mx2(formula=kind, kind="2H", a=a, thickness=3.19, size=(1, 1, 1), vacuum=vacuum)
    s.set_pbc(pbc)
    return s


def rotated(atoms, seed=0):
    """The same structure in a turned frame: cell and positions under one rotation."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    out = atoms.copy()
    out.set_cell(atoms.get_cell()[:] @ q.T)
    out.set_positions(atoms.get_positions() @ q.T)
    return out


@pytest.fixture(scope="module")
def fs():
    return inorganic_fieldset(["Mo", "S", "Se", "C"])


@pytest.fixture(scope="module")
def specs():
    return (PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5),
            PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, z_max=3.0, z_step=1.0))


# --------------------------------------------------------------------------- the measure
def test_the_measure_is_the_volume_the_area_or_the_length():
    A = np.diag([2.0, 3.0, 5.0])
    assert periodic_measure(A, (True, True, True)) == pytest.approx(30.0)
    assert periodic_measure(A, (True, True, False)) == pytest.approx(6.0)
    assert periodic_measure(A, (True, False, False)) == pytest.approx(2.0)
    assert periodic_measure(A, (False, False, False)) is None
    n = layer_normal(A, (True, True, False))
    assert np.allclose(n, [0, 0, 1])
    with pytest.raises(ValueError):
        layer_normal(A, (True, True, True))


# --------------------------------------------------------------------------- layers
def test_a_layer_does_not_see_the_vacuum_once_the_flags_say_it_is_a_layer(fs, specs):
    """With two periodic directions the background is per area and the neighbour list stops at
    the vacuum, so the width of the vacuum is not in the descriptor.  Flagged as a crystal the
    same file gives a descriptor that depends on it, which is the defect this fixes."""
    for spec in specs:
        thin = periodic_autocorrelation(crystal_fields(layer(5.0), fs), spec)
        wide = periodic_autocorrelation(crystal_fields(layer(25.0), fs), spec)
        assert np.allclose(thin, wide, rtol=1e-9, atol=1e-12)
        assert np.abs(thin).max() > 0
    spec = specs[0]
    thin = periodic_autocorrelation(crystal_fields(layer(5.0, pbc=(True, True, True)), fs), spec)
    wide = periodic_autocorrelation(crystal_fields(layer(25.0, pbc=(True, True, True)), fs), spec)
    assert not np.allclose(thin, wide, rtol=1e-3)


def test_an_in_plane_supercell_a_rotation_and_a_shift_along_the_normal_change_nothing(fs, specs):
    for spec in specs:
        base = periodic_autocorrelation(crystal_fields(layer(), fs), spec)
        big = layer().repeat((2, 3, 1))
        assert np.allclose(periodic_autocorrelation(crystal_fields(big, fs), spec), base, rtol=1e-9)
        turned = rotated(layer(), seed=3)
        assert np.allclose(periodic_autocorrelation(crystal_fields(turned, fs), spec), base, rtol=1e-8)
        up = layer()
        up.positions[:, 2] += 4.0
        assert np.allclose(periodic_autocorrelation(crystal_fields(up, fs), spec), base, rtol=1e-9)
        # a non-orthogonal cell and a layer leaning out of the xy plane are the same layer
        lean = layer()
        cell = lean.get_cell()[:]
        cell[2] = cell[2] + np.array([3.0, -2.0, 0.0])
        lean.set_cell(cell, scale_atoms=False)
        assert np.allclose(periodic_autocorrelation(crystal_fields(lean, fs), spec), base, rtol=1e-9)


def test_the_structure_still_matters(fs, specs):
    for spec in specs:
        a = periodic_autocorrelation(crystal_fields(layer(a=3.18), fs), spec)
        b = periodic_autocorrelation(crystal_fields(layer(a=3.40), fs), spec)
        c = periodic_autocorrelation(crystal_fields(layer(kind="MoSe2", a=3.32), fs), spec)
        assert not np.allclose(a, b, rtol=1e-3)
        assert not np.allclose(a, c, rtol=1e-3)


def test_a_uniform_layer_is_pure_background():
    """Atoms spread uniformly over the plane, at random heights within a slab: the raw channels
    follow the per-area background of the pairs' height differences, and the subtracted
    channels are noise about zero.  This is the ring integral checked against a simulation."""
    rng = np.random.default_rng(11)
    n, L, t = 2500, 60.0, 3.0
    xyz = np.column_stack([rng.uniform(0, L, size=(n, 2)), rng.uniform(0, t, size=n)])
    cell = np.diag([L, L, 40.0])
    c = CrystalFields(name="gas", coords=xyz, weights=np.ones((n, 1)), radii=np.ones(n),
                      symbols=["Ar"] * n, cell=cell, pbc=(True, True, False))
    raw_spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, background=False, include_self=False)
    sub_spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, background=True, include_self=False)
    raw = periodic_autocorrelation(c, raw_spec)
    sub = periodic_autocorrelation(c, sub_spec)
    far = raw_spec.centres >= 2.0
    assert np.abs(sub[far]).max() < 0.05 * raw[far].max()
    # the background of a pair a height h apart is the tail of the ring integral from h
    bg = raw - sub
    h = np.abs(xyz[:, 2][:, None] - xyz[:, 2][None, :])
    r = np.linspace(0, 8.0, 4000)
    g = np.exp(-((r[:, None] - raw_spec.centres[None, :]) ** 2) / 0.5) * raw_spec.window(r)[:, None]
    f = 2 * np.pi * r[:, None] * g
    hs = np.linspace(0, t, 31)
    J = np.array([np.trapezoid(np.where(r[:, None] >= hh, f, 0.0), r, axis=0) for hh in hs])
    want = np.array([np.interp(h.ravel(), hs, J[:, k]).sum() for k in range(len(raw_spec.centres))])
    want = want / (L * L) / n
    assert np.allclose(bg, want, rtol=2e-2)


def test_a_molecule_sized_layer_gives_the_molecular_descriptor_when_nothing_is_subtracted(fs):
    """A layer whose cell is far wider than the cutoff has no neighbour across any boundary,
    so without the background the periodic sum is the molecular autocorrelation per atom."""
    from cmfqsar.invariant import AutocorrelationSpec, autocorrelation
    s = layer()
    s.set_cell(np.diag([60.0, 60.0, 30.0]), scale_atoms=False)
    s.set_pbc((True, True, False))
    c = crystal_fields(s, fs)
    mol_spec = AutocorrelationSpec(d_max=8.0, step=0.5, sigma=0.5)
    per_spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, taper=0.0, background=False)
    want = autocorrelation(c, mol_spec) / c.n_atoms
    assert np.allclose(periodic_autocorrelation(c, per_spec), want, rtol=1e-9, atol=1e-12)


# --------------------------------------------------------------------------- chains
def chain(box=10.0, n=4, d=1.3):
    s = Atoms("C" * n, positions=[[k * d, 0.0, 0.0] for k in range(n)],
              cell=np.diag([n * d, box, box]), pbc=(True, False, False))
    s.positions[1::2, 1] += 0.4                           # a zigzag, so that the axis matters
    return s


def test_a_chain_does_not_see_its_box_and_a_supercell_changes_nothing(fs, specs):
    spec = specs[0]
    base = periodic_autocorrelation(crystal_fields(chain(10.0), fs), spec)
    assert np.allclose(periodic_autocorrelation(crystal_fields(chain(30.0), fs), spec), base, rtol=1e-9)
    assert np.allclose(periodic_autocorrelation(crystal_fields(chain(10.0).repeat((3, 1, 1)), fs), spec),
                       base, rtol=1e-9)
    assert np.allclose(periodic_autocorrelation(crystal_fields(rotated(chain(), 5), fs), spec),
                       base, rtol=1e-8)
    assert not np.allclose(periodic_autocorrelation(crystal_fields(chain(d=1.5), fs), spec), base, rtol=1e-3)


def test_a_uniform_chain_is_pure_background():
    """Atoms spread uniformly along the axis, at random positions within a tube around it:
    the subtracted channels are noise about zero at distances beyond the tube."""
    rng = np.random.default_rng(7)
    n, L, w = 3000, 900.0, 1.5
    xyz = np.column_stack([rng.uniform(0, L, size=n), rng.uniform(0, w, size=(n, 2))])
    c = CrystalFields(name="tube", coords=xyz, weights=np.ones((n, 1)), radii=np.ones(n),
                      symbols=["Ar"] * n, cell=np.diag([L, 30.0, 30.0]), pbc=(True, False, False))
    raw_spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, background=False, include_self=False)
    sub_spec = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, background=True, include_self=False)
    raw = periodic_autocorrelation(c, raw_spec)
    sub = periodic_autocorrelation(c, sub_spec)
    far = raw_spec.centres >= 3.0
    assert np.abs(sub[far]).max() < 0.08 * raw[far].max()


# --------------------------------------------------------------------------- the images
def test_the_images_respect_the_flags(fs):
    A = np.diag([3.0, 3.0, 20.0])
    T3 = image_translations(A, 5.0)
    T2 = image_translations(A, 5.0, (True, True, False))
    assert np.abs(T3[:, 2]).max() > 0
    assert np.abs(T2[:, 2]).max() == 0
    assert len(T2) < len(T3)
    # a layer 1 A below the top face of its cell: flagged as a crystal its copy across the
    # vacuum is within reach, flagged as a layer it is not
    s = layer(vacuum=1.0)
    s.positions[:, 2] += s.get_cell()[2, 2] - 1.0 - s.positions[:, 2].max()
    as_crystal = s.copy()
    as_crystal.set_pbc(True)
    x3, _, _ = cell_images(crystal_fields(as_crystal, fs), 6.0)
    x2, _, _ = cell_images(crystal_fields(s, fs), 6.0)
    top = s.get_cell()[2, 2]
    assert (x3[:, 2] > top).any()
    assert not (x2[:, 2] > top).any() and not (x2[:, 2] < 0).any()


def test_the_periodic_overlap_of_a_layer_does_not_see_the_vacuum(fs):
    thin = crystal_table([layer(4.0), layer(4.0, a=3.3)], fs)
    wide = crystal_table([layer(24.0), layer(24.0, a=3.3)], fs)
    specs = fs.specs
    K1 = periodic_overlap_kernels(thin, thin, specs)
    K2 = periodic_overlap_kernels(wide, wide, specs)
    assert np.allclose(K1, K2, rtol=1e-8, atol=1e-10)
    assert np.allclose(periodic_self_overlaps(thin, specs), periodic_self_overlaps(wide, specs), rtol=1e-8)
    spread = lattice_spread(thin)
    assert spread["volume_mean"] == pytest.approx(0.5 * (periodic_measure(thin.molecules[0].cell, (1, 1, 0))
                                                         + periodic_measure(thin.molecules[1].cell, (1, 1, 0))))


# --------------------------------------------------------------------------- centring
def test_a_layer_split_across_the_boundary_is_gathered(fs, specs):
    s = layer(vacuum=8.0)
    split = s.copy()
    c = split.get_cell()[2, 2]
    split.positions[:, 2] -= split.positions[:, 2].min() + 1.0     # the bottom S below z = 0
    split.wrap(pbc=(True, True, True))                             # ... and wrapped to the top
    assert split.positions[:, 2].max() > 0.9 * c
    gathered = centre_layer(split)
    f = gathered.get_scaled_positions(wrap=False)[:, 2]
    assert 0.3 < f.min() and f.max() < 0.7
    assert abs(0.5 * (f.min() + f.max()) - 0.5) < 1e-9
    for spec in specs:
        assert np.allclose(periodic_autocorrelation(crystal_fields(gathered, fs), spec),
                           periodic_autocorrelation(crystal_fields(s, fs), spec), rtol=1e-9)
    bulk_si = bulk("Si", "diamond", a=5.43)
    assert np.allclose(centre_layer(bulk_si).positions, bulk_si.positions)


# --------------------------------------------------------------------------- height channels
def test_the_height_grid_the_blocks_and_the_labels_agree(fs):
    spec = PeriodicSpec(r_max=6.0, step=1.0, sigma=0.7, z_max=2.0, z_step=1.0)
    assert np.allclose(spec.heights, [-2, -1, 0, 1, 2])
    assert spec.n_bins == 7 * 25
    row = periodic_autocorrelation(crystal_fields(layer(), fs), spec)
    n_pairs = len(field_pairs(fs.n_fields))
    assert row.shape == (n_pairs * spec.n_bins,)
    assert sum(periodic_block_sizes(spec, fs.n_fields)) == len(row)
    labels = periodic_channel_labels(fs, spec)
    assert len(labels) == len(row)
    assert labels[0].lower().startswith("electrostatic @ rho 0.00 a, z -2.0 / -2.0 a")
    assert PeriodicSpec(r_max=6.0, step=1.0).heights.shape == (1,)


def test_the_height_channels_refuse_a_crystal(fs):
    spec = PeriodicSpec(z_max=3.0)
    with pytest.raises(ValueError, match="two"):
        periodic_autocorrelation(crystal_fields(bulk("NaCl", "rocksalt", a=5.64),
                                                inorganic_fieldset(["Na", "Cl"])), spec)


def test_the_two_faces_of_a_layer(fs):
    """A symmetric layer is its own mirror image.  A Janus layer, S on one face and Se on the
    other, is told from its mirror image by an oriented spec and not by the default one."""
    sym = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, z_max=3.0, z_step=1.0)
    ori = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, z_max=3.0, z_step=1.0, oriented=True)

    def mirrored(s):
        m = s.copy()
        m.positions[:, 2] = s.get_cell()[2, 2] - s.positions[:, 2]
        return m

    mos2 = layer()
    for spec in (sym, ori):
        assert np.allclose(periodic_autocorrelation(crystal_fields(mirrored(mos2), fs), spec),
                           periodic_autocorrelation(crystal_fields(mos2, fs), spec), rtol=1e-9)
    janus = layer()
    janus.symbols[int(np.argmax(janus.positions[:, 2]))] = "Se"
    up, down = crystal_fields(janus, fs), crystal_fields(mirrored(janus), fs)
    assert np.allclose(periodic_autocorrelation(up, sym), periodic_autocorrelation(down, sym), rtol=1e-9)
    assert not np.allclose(periodic_autocorrelation(up, ori), periodic_autocorrelation(down, ori), rtol=1e-3)
    # the symmetric descriptor is the mean of the oriented descriptor over the two faces
    mean = 0.5 * (periodic_autocorrelation(up, ori) + periodic_autocorrelation(down, ori))
    assert np.allclose(mean, periodic_autocorrelation(up, sym), rtol=1e-9)
    # the background is exact: a layer with and without it differ by s s^T R / A in each block
    nobg = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, z_max=3.0, z_step=1.0, background=False)
    diff = (periodic_autocorrelation(up, nobg) - periodic_autocorrelation(up, sym))
    assert np.isfinite(diff).all() and np.abs(diff).max() > 0


def test_the_height_sum_is_the_in_plane_distance_distribution(fs):
    """Summed over both height axes with a grid far wider than the layer, the oriented channels
    reduce to the in-plane distance distribution of the field weights, images included."""
    s = layer()
    c = crystal_fields(s, fs)
    spec = PeriodicSpec(r_max=6.0, step=1.0, sigma=1.0, z_max=12.0, z_step=1.0, oriented=True,
                        background=False)
    nz = len(spec.heights)
    f = 1                                                   # the steric field alone
    row = slab_autocorrelation(c, spec, pairs=[(f, f)]).reshape(len(spec.centres), nz, nz).sum(axis=(1, 2))
    from ase.neighborlist import neighbor_list
    i, j, D = neighbor_list("ijD", s, 12.0)
    rho = np.linalg.norm(D[:, :2], axis=1)
    keep = rho < 6.0
    i, j, rho = i[keep], j[keep], rho[keep]
    w = c.weights[:, f]
    z = s.positions[:, 2] - s.positions[:, 2].mean()
    zsum = np.exp(-((z[:, None] - spec.heights[None, :]) ** 2) / 2.0).sum(axis=1)
    want = np.array([np.sum(w[i] * zsum[i] * w[j] * zsum[j] * np.exp(-(rho - d) ** 2 / 2.0) * spec.window(rho))
                     + np.sum(w * w * zsum * zsum) * np.exp(-d ** 2 / 2.0)
                     for d in spec.centres]) / len(s)
    assert np.allclose(row, want, rtol=1e-6)


def test_a_model_with_height_channels_fits_predicts_and_round_trips(tmp_path, fs):
    rng = np.random.default_rng(2)
    structures, y = [], []
    for k in range(14):
        a = 3.0 + 0.5 * rng.uniform()
        kind = ("MoS2", "MoSe2")[k % 2]
        s = layer(kind=kind, a=a, vacuum=6.0 + 10.0 * rng.uniform())
        s.positions[:, 2] += rng.normal(scale=0.05, size=len(s))
        structures.append(s)
        y.append(a + 0.3 * (k % 2))
    y = np.array(y)
    table = crystal_table(structures, fs)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="periodic",
                     ac_sigma=0.5, ac_d_max=8.0, pbc_z_max=3.0, pbc_z_step=1.0, params={"lam": 1e-4})
    model.fit(table, y)
    assert model.stats.r2 > 0.9
    pred = model.predict(table)
    assert np.allclose(pred, model.stats.extra["y_fit"])
    from cmfqsar.dataio import load_model, save_model
    path = str(tmp_path / "layers.cmf")
    save_model(path, model)
    back = load_model(path)[0]
    assert back.pbc_z_max == 3.0 and back.pbc_z_step == 1.0 and back.pbc_oriented is False
    assert np.allclose(back.predict(table), pred)
    # the spec the model builds carries the heights, and the copy keeps them
    assert model._ac_spec().has_heights and model.fresh_copy().pbc_z_max == 3.0


def test_the_command_line_takes_a_layer(tmp_path, capsys):
    from ase.io import write
    from cmfqsar.cli import main
    rng = np.random.default_rng(4)
    structures = []
    for k in range(12):
        s = layer(kind=("MoS2", "MoSe2")[k % 2], a=3.0 + 0.5 * rng.uniform(), vacuum=8.0)
        s.set_pbc(True)                                  # as a POSCAR would come: no flags
        s.info["name"] = f"L{k}"
        s.info["gap"] = float(s.get_cell()[0, 0]) + 0.2 * (k % 2)
        structures.append(s)
    path = tmp_path / "layers.extxyz"
    write(str(path), structures, format="extxyz")
    main(["build-crystals", str(path), "--target", "gap", "--pbc", "ab", "--heights", "3",
          "--height-step", "1", "--elements", "Mo,S,Se", "--cutoff", "8", "--folds", "3",
          "--lams", "1e-4,1e-2"])
    out = capsys.readouterr().out
    assert "periodic in two directions" in out
    assert "height channels" in out
    assert "q2" in out.lower()
