"""A metal slab as the receptor (plan item EC6).

What can fail quietly here: a slab whose images are missing, so that a molecule near the edge of
the cell sees the surface end; an image-charge energy with the wrong factor, which is a plausible
number for the wrong physics; and a pose search that claims to be constrained to the surface but
turns the molecule over or buries it in the metal. Each has a test.
"""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.fields import FieldSet, build_fields
from cmfqsar.slab import (COULOMB_EV, SLAB_NONZERO_FIELDS, SurfaceScorer, bcc110_slab,
                          dock_on_surface, image_charge_energy, layer_heights, site_profile,
                          slab_receptor, surface_score, surface_sites, surface_transform,
                          top_layer_indices)

ase = pytest.importorskip("ase")

FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")


def fieldset(alpha: float = 1.0) -> FieldSet:
    return FieldSet.create(physchem=list(FIELDS), alpha=alpha, charge_model="gasteiger",
                           param_scheme="rdkit")


def molecule(smiles: str = "c1ccc2[nH]cnc2c1", height: float = 2.2) -> Chem.Mol:
    """A benzimidazole lying flat with its lowest heavy atom at `height` above z = 0."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(m)
    xyz = np.array(m.GetConformer().GetPositions())
    xyz -= xyz.mean(axis=0)
    # lay it flat: the smallest principal axis along z
    u, _s, _vt = np.linalg.svd((xyz - xyz.mean(axis=0)).T @ (xyz - xyz.mean(axis=0)))
    xyz = xyz @ u
    heavy = np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()])
    xyz[:, 2] += height - xyz[heavy, 2].min()
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, xyz[i].tolist())
    return m


# ------------------------------------------------------------------ the slab

def test_the_slab_has_its_layers_and_its_top_at_zero():
    slab = bcc110_slab(nx=4, ny=4, layers=4)
    heights = layer_heights(slab)
    assert len(heights) == 4
    assert heights[0] == pytest.approx(0.0, abs=1e-9)
    spacings = np.diff(heights)
    assert np.allclose(spacings, spacings[0], atol=1e-6)       # evenly spaced layers
    assert abs(spacings[0]) > 1.5                              # about 1.97 A for a = 2.78


def test_the_top_layer_is_found_and_only_the_top_layer():
    slab = bcc110_slab(nx=4, ny=4, layers=4)
    one = top_layer_indices(slab, 1)
    two = top_layer_indices(slab, 2)
    assert len(one) == len(slab) // 4
    assert len(two) == 2 * len(one)
    z = np.asarray(slab.positions)[:, 2]
    assert np.allclose(z[one], 0.0, atol=1e-9)


def test_the_receptor_carries_the_images_of_the_top_layers():
    slab = bcc110_slab(nx=4, ny=4, layers=4)
    fs = fieldset()
    cutoff = 10.0
    without = slab_receptor(slab, fs, top_layers=2, cutoff=0.0)
    with_images = slab_receptor(slab, fs, top_layers=2, cutoff=cutoff)
    # whole cells are added, as many rings as the cutoff needs in each direction
    assert with_images.n_atoms % without.n_atoms == 0
    assert with_images.n_atoms >= 9 * without.n_atoms
    # and the surface then continues at least `cutoff` beyond the primary cell on every side, so
    # a molecule anywhere in the cell sees metal and not an edge
    cell = np.asarray(slab.get_cell())
    for axis, span in ((0, cell[0, 0]), (1, cell[1, 1])):
        assert with_images.coords[:, axis].min() <= -cutoff
        assert with_images.coords[:, axis].max() >= span + cutoff - 2.78


def test_only_the_steric_field_of_a_clean_metal_slab_is_not_zero():
    """Three of the five fields are zero on a one-element metal, and not for want of parameters."""
    slab = bcc110_slab(nx=4, ny=4, layers=2)
    fs = fieldset()
    rec = slab_receptor(slab, fs, top_layers=1)
    nonzero = [s.name for k, s in enumerate(fs.specs)
               if np.abs(rec.weights[:, k]).sum() > 1e-12]
    assert nonzero == list(SLAB_NONZERO_FIELDS) == ["steric"]
    assert np.all(rec.radii > 1.0)                             # the metal has a radius (item EC4)


def test_an_element_indicator_field_marks_the_metal():
    slab = bcc110_slab(nx=4, ny=4, layers=2)
    fs = FieldSet.create(physchem=["steric"], cif_types=["Fe"], alpha=1.0)
    rec = slab_receptor(slab, fs, top_layers=1)
    names = [s.name for s in fs.specs]
    assert np.allclose(rec.weights[:, names.index("cif:Fe")], 1.0)


# ------------------------------------------------------------------ the image charge

def test_one_charge_above_a_plane_is_the_textbook_result():
    for h in (1.0, 2.0, 5.0):
        got = image_charge_energy(np.array([[0.0, 0.0, h]]), np.array([1.0]))
        assert got == pytest.approx(-COULOMB_EV / (4.0 * h), rel=1e-12)


def test_the_image_energy_is_attractive_for_any_distribution():
    rng = np.random.default_rng(0)
    for _ in range(8):
        coords = rng.normal(size=(6, 3)) + np.array([0.0, 0.0, 4.0])
        charges = rng.normal(size=6)
        charges -= charges.mean()                              # a neutral molecule
        assert image_charge_energy(coords, charges) < 0.0


def test_the_image_energy_falls_off_with_height():
    coords = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.5]])
    charges = np.array([0.4, -0.4])
    near = image_charge_energy(coords, charges)
    far = image_charge_energy(coords + np.array([0.0, 0.0, 6.0]), charges)
    assert near < far < 0.0


def test_the_image_energy_cannot_say_where_on_the_lattice_a_molecule_sits():
    """It is invariant to translation in the plane and to rotation about the normal.

    This is why a pose search cannot use it to place a molecule laterally, and the module says so
    rather than leaving the caller to find out.
    """
    rng = np.random.default_rng(1)
    coords = rng.normal(size=(7, 3)) + np.array([0.0, 0.0, 3.5])
    charges = rng.normal(size=7)
    base = image_charge_energy(coords, charges)
    shifted = image_charge_energy(coords + np.array([3.7, -2.1, 0.0]), charges)
    psi = 0.9
    rot = np.array([[np.cos(psi), -np.sin(psi), 0.0], [np.sin(psi), np.cos(psi), 0.0],
                    [0.0, 0.0, 1.0]])
    turned = image_charge_energy(coords @ rot.T, charges)
    assert shifted == pytest.approx(base, rel=1e-12)
    assert turned == pytest.approx(base, rel=1e-12)


def test_a_charge_on_the_plane_stays_finite():
    assert np.isfinite(image_charge_energy(np.zeros((1, 3)), np.array([1.0])))


def test_no_charges_means_no_image_energy():
    assert image_charge_energy(np.array([[0.0, 0.0, 3.0]]), np.array([0.0])) == 0.0


# ------------------------------------------------------------------ the pose

def test_the_transform_does_what_its_parameters_say():
    coords = np.array([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [0.0, 1.0, 2.0]])
    centre = coords.mean(axis=0)
    moved = surface_transform(coords, (1.0, -2.0, 0.5, 0.0, 0.0, 0.0), centre)
    assert np.allclose(moved - coords, np.array([1.0, -2.0, 0.5]))
    # a quarter turn about the normal keeps every height
    turned = surface_transform(coords, (0.0, 0.0, 0.0, np.pi / 2, 0.0, 0.0), centre)
    assert np.allclose(turned[:, 2], coords[:, 2])
    assert not np.allclose(turned[:, :2], coords[:, :2])
    # a tilt changes the heights and keeps the shape
    tilted = surface_transform(coords, (0.0, 0.0, 0.0, 0.0, 0.3, 0.0), centre)
    assert not np.allclose(tilted[:, 2], coords[:, 2])
    for a, b in ((0, 1), (0, 2), (1, 2)):
        assert np.linalg.norm(tilted[a] - tilted[b]) == pytest.approx(
            np.linalg.norm(coords[a] - coords[b]))


def test_docking_stays_on_the_surface():
    slab = bcc110_slab(nx=4, ny=4, layers=2)
    fs = fieldset()
    rec = slab_receptor(slab, fs, top_layers=2, cutoff=8.0)
    mf = build_fields(molecule(height=3.5), fs, "m")
    coords, res = dock_on_surface(mf, rec, fs, starts=4, seed=0, min_height=1.2, max_height=6.0)
    assert 1.0 < res.height_after < 6.5                        # neither buried nor adrift
    assert res.tilt_deg <= 25.0 + 1e-6                         # the tilt bound is respected
    assert res.score_after >= res.score_before - 1e-9
    assert not res.flipped
    # the molecule is moved rigidly: every interatomic distance survives
    before = mf.coords
    assert np.allclose(np.linalg.norm(before[0] - before[5]),
                       np.linalg.norm(coords[0] - coords[5]))


def test_docking_can_be_asked_to_try_the_other_face():
    slab = bcc110_slab(nx=4, ny=4, layers=2)
    fs = fieldset()
    rec = slab_receptor(slab, fs, top_layers=2, cutoff=8.0)
    mf = build_fields(molecule(), fs, "m")
    _c1, one = dock_on_surface(mf, rec, fs, starts=2, seed=0, both_faces=False)
    _c2, two = dock_on_surface(mf, rec, fs, starts=2, seed=0, both_faces=True)
    assert two.score_after >= one.score_after - 1e-9           # a superset of the search
    assert isinstance(two.flipped, bool)


def test_a_score_needs_the_steric_field():
    slab = bcc110_slab(nx=4, ny=4, layers=2)
    fs = FieldSet.create(physchem=["electrostatic"], alpha=1.0)
    rec = slab_receptor(slab, fs, top_layers=1)
    mf = build_fields(molecule(), fs, "m")
    with pytest.raises(ValueError):
        SurfaceScorer(mf, rec, fs)


# ------------------------------------------------------------------ the sites and the score

def test_the_three_kinds_of_site_are_found():
    slab = bcc110_slab(nx=4, ny=4, layers=2)
    labels, points = surface_sites(slab)
    assert set(labels) == {"top", "bridge", "hollow"}
    assert len(points) == len(labels)
    assert np.allclose(points[:, 2], 0.0)
    assert labels.count("top") == len(top_layer_indices(slab, 1))


def test_the_site_profile_says_where_a_molecule_sits():
    slab = bcc110_slab(nx=6, ny=6, layers=2)
    fs = fieldset()
    rec = slab_receptor(slab, fs, top_layers=1)
    sites = surface_sites(slab)
    mf = build_fields(molecule(height=2.2), fs, "m")
    score = surface_score(mf, rec, fs, sites=sites)
    assert set(score.sites) == {"bridge", "hollow", "top"}
    assert all(v >= 0.0 for v in score.sites.values())
    assert score.burial > 0.0
    assert score.height == pytest.approx(2.2, abs=0.05)
    # a molecule lifted far from the surface covers no site worth speaking of
    far = build_fields(molecule(height=12.0), fs, "m")
    high = surface_score(far, rec, fs, sites=sites)
    assert max(high.sites.values()) < 0.01 * max(score.sites.values())
    assert high.burial < 0.01 * score.burial


def test_an_empty_site_list_gives_an_empty_profile():
    assert site_profile(np.zeros((0, 3)), np.zeros(0), [], np.zeros((0, 3))) == {}
