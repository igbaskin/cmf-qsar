"""Tests for cmfqsar/solvent.py: a water occupancy map as a CMF field (plan item D9).

The trajectories are built here, with the water put where the test wants it, so that the claims
can fail. The one that matters most is `test_a_site_appears_where_the_water_actually_sits`: a
water held at a known position in most frames must produce a hydration site there and nowhere
else, and a water that moves every frame must produce none.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import solvent as SV  # noqa: E402
from cmfqsar import trajectory as TJ  # noqa: E402
from cmfqsar.fields import AtomTable, build_fields  # noqa: E402

mdtraj = pytest.importorskip("mdtraj", reason="the solvent map is read from a trajectory")

BOX = 30.0


def _peptide():
    p = Chem.MolFromSequence("AAGAA")
    AllChem.EmbedMolecule(p, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(p)
    return p


def _ligand():
    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(m, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(m)
    for k, a in enumerate(m.GetAtoms()):
        ri = Chem.AtomPDBResidueInfo()
        ri.SetResidueName("LIG")
        ri.SetResidueNumber(900)
        ri.SetIsHeteroAtom(True)
        ri.SetName(f"{a.GetSymbol()}{k:<3d}"[:4])
        a.SetMonomerInfo(ri)
    return m


def _water_mol(n, positions):
    """`n` waters as one molecule, each three atoms, at the given oxygen positions."""
    rw = Chem.RWMol()
    xyz = []
    for k in range(n):
        base = np.asarray(positions[k], float)
        for sym, off in (("O", (0, 0, 0)), ("H", (0.76, 0.59, 0)), ("H", (-0.76, 0.59, 0))):
            a = Chem.Atom(sym)
            ri = Chem.AtomPDBResidueInfo()
            ri.SetResidueName("HOH")
            ri.SetResidueNumber(2000 + k)
            ri.SetIsHeteroAtom(True)
            ri.SetName(f" {sym}  "[:4])
            a.SetMonomerInfo(ri)
            rw.AddAtom(a)
            xyz.append(base + np.asarray(off, float))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*[float(v) for v in p]))
    mol.AddConformer(conf)
    return mol


def _system(tmp_path, held_fraction=1.0, n_frames=20, n_waters=4, tag="sys", seed=0):
    """Peptide, ligand and water; water 0 is held at a fixed spot in `held_fraction` of frames.

    The fixed spot is beside the ligand, so a hydration site there is a site in the map. The
    other waters are moved every frame, so they should raise no site of their own.
    """
    rng = np.random.RandomState(seed)
    pep, lig = _peptide(), _ligand()
    pep_xyz = np.asarray(pep.GetConformer().GetPositions())
    lig_xyz = np.asarray(lig.GetConformer().GetPositions())
    pep_xyz = pep_xyz - pep_xyz.mean(axis=0) + lig_xyz.mean(axis=0) + np.array([5.0, 0.0, 0.0])
    held = lig_xyz.mean(axis=0) + np.array([0.0, 4.0, 0.0])
    water = _water_mol(n_waters, [held] + [held + np.array([0.0, 3.0 * (k + 1), 0.0])
                                           for k in range(n_waters - 1)])
    wat_xyz = np.asarray(water.GetConformer().GetPositions())

    whole = Chem.CombineMols(Chem.CombineMols(pep, lig), water)
    mol = Chem.RWMol(whole).GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(np.vstack([pep_xyz, lig_xyz, wat_xyz])):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*[float(v) for v in p]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)

    n_pep, n_lig = pep.GetNumAtoms(), lig.GetNumAtoms()
    frames = []
    for k in range(n_frames):
        w = wat_xyz.copy()
        if k >= int(round(held_fraction * n_frames)):
            w[0:3] += rng.uniform(-8.0, 8.0, size=3)      # water 0 lets go in these frames
        for j in range(1, n_waters):                       # the others wander every frame
            w[3 * j:3 * j + 3] += rng.uniform(-8.0, 8.0, size=3)
        frames.append(np.vstack([pep_xyz, lig_xyz, w]))

    pdb = str(tmp_path / f"{tag}.pdb")
    Chem.MolToPDBFile(mol, pdb)
    top = mdtraj.load(pdb).topology
    traj = mdtraj.Trajectory(np.stack(frames) / 10.0, top)
    traj.unitcell_vectors = np.tile(np.eye(3) * (BOX / 10.0), (len(frames), 1, 1))
    dcd = str(tmp_path / f"{tag}.dcd")
    traj.save_dcd(dcd)
    ref = str(tmp_path / f"{tag}_ref.pdb")
    only = Chem.RWMol(pep).GetMol()
    rc = Chem.Conformer(only.GetNumAtoms())
    for i, p in enumerate(pep_xyz):
        rc.SetAtomPosition(i, Chem.rdGeometry.Point3D(*[float(v) for v in p]))
    only.RemoveAllConformers()
    only.AddConformer(rc)
    Chem.MolToPDBFile(only, ref)
    return pdb, dcd, ref, lig, held, (n_pep, n_lig)


# ------------------------------------------------------------------------------- the map


def test_a_topology_with_no_water_is_refused(tmp_path):
    lig = _ligand()
    xyz = np.asarray(lig.GetConformer().GetPositions())
    pdb = str(tmp_path / "dry.pdb")
    Chem.MolToPDBFile(lig, pdb)
    t = TJ.load(pdb)
    with pytest.raises(SV.SolventError, match="no water"):
        SV.water_oxygens(t)


def test_the_oxygens_are_found_and_the_hydrogens_are_not(tmp_path):
    pdb, dcd, _ref, _lig, _held, _n = _system(tmp_path, tag="ox")
    t = TJ.load(pdb, dcd)
    ox = SV.water_oxygens(t)
    assert len(ox) == 4, "one oxygen per water"
    assert all(t.elements[i].upper() == "O" for i in ox)
    assert all(t.residues[i] == "HOH" for i in ox)


def test_bulk_occupancy_is_what_the_density_says():
    occ = SV.Occupancy(points=np.zeros((1, 3)), values=np.zeros(1), frames=1, probe=1.0)
    assert occ.bulk == pytest.approx(0.14, abs=0.01), "0.0334 per cubic angstrom in a 1 A sphere"


def test_a_map_needs_a_periodic_cell(tmp_path, monkeypatch):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, tag="nocell")
    real = SV.load

    def without_box(*a, **k):
        t = real(*a, **k)
        t.box = None
        return t

    monkeypatch.setattr(SV, "load", without_box)
    with pytest.raises(SV.SolventError, match="periodic cell"):
        SV.occupancy(pdb, dcd, reference=ref)


# ----------------------------------------------------------------------------- the sites


def test_a_site_appears_where_the_water_actually_sits(tmp_path):
    """The claim the module rests on: a held water becomes a site, a wandering one does not."""
    pdb, dcd, ref, _lig, held, _n = _system(tmp_path, held_fraction=1.0, n_frames=20,
                                            tag="held")
    occ = SV.occupancy(pdb, dcd, reference=ref, spacing=0.5, map_radius=6.0)
    centres, heights = SV.hydration_sites(occ, min_occupancy=0.5, min_distance=2.4)
    assert len(centres) >= 1, "the held water raised no site"
    nearest = float(np.linalg.norm(centres - held, axis=1).min())
    assert nearest < 1.0, f"the strongest site is {nearest:.2f} A from where the water was held"
    assert heights.max() > 0.9, "a water held in every frame must be occupied in every frame"


def test_a_water_that_never_settles_raises_no_site(tmp_path):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=0.0, n_frames=20,
                                             tag="free")
    occ = SV.occupancy(pdb, dcd, reference=ref, spacing=0.5, map_radius=6.0)
    centres, _h = SV.hydration_sites(occ, min_occupancy=0.5, min_distance=2.4)
    assert len(centres) == 0, f"{len(centres)} site(s) from water that moved every frame"


def test_a_water_held_half_the_time_is_occupied_about_half_the_time(tmp_path):
    pdb, dcd, ref, _lig, held, _n = _system(tmp_path, held_fraction=0.5, n_frames=20,
                                            tag="half")
    occ = SV.occupancy(pdb, dcd, reference=ref, spacing=0.5, map_radius=6.0)
    at = occ.values[np.linalg.norm(occ.points - held, axis=1) < 0.4]
    assert at.size and 0.35 < at.max() < 0.65, f"occupancy {at.max():.2f} for half the frames"


def test_two_sites_are_never_closer_than_the_spacing_asked_for(tmp_path):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=1.0, n_waters=3,
                                             tag="apart")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    centres, _h = SV.hydration_sites(occ, min_occupancy=0.3, min_distance=3.0)
    if len(centres) > 1:
        d = np.linalg.norm(centres[:, None, :] - centres[None, :, :], axis=2)
        d += np.eye(len(centres)) * 1e3
        assert d.min() >= 3.0 - 1e-9


def test_the_sites_come_out_strongest_first(tmp_path):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=1.0, n_waters=3,
                                             tag="order")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    _c, heights = SV.hydration_sites(occ, min_occupancy=0.2, min_distance=2.4)
    assert np.all(np.diff(heights) <= 1e-12), "the sites are not in order of occupancy"


def test_the_number_of_sites_can_be_capped(tmp_path):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=1.0, n_waters=4,
                                             tag="cap")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    centres, _h = SV.hydration_sites(occ, min_occupancy=0.1, min_distance=1.0, max_sites=2)
    assert len(centres) <= 2


# ----------------------------------------------------------------------- the coefficients


def test_the_coefficients_reproduce_a_map_made_of_one_gaussian():
    """The fit is a fit: given a map that *is* a Gaussian, it must recover its height."""
    centre = np.array([[0.0, 0.0, 0.0]])
    pts = SV._grid_points(np.array([-3.0, -3.0, -3.0]), (13, 13, 13), 0.5)
    alpha = 0.5
    truth = 0.7 * np.exp(-0.5 * alpha * (pts ** 2).sum(axis=1))
    occ = SV.Occupancy(points=pts, values=truth, frames=1)
    coeff = SV.fit_coefficients(occ, centre, alpha=alpha)
    assert coeff[0] == pytest.approx(0.7, rel=1e-6)


def test_the_coefficients_are_never_negative():
    """An occupancy cannot be negative, and a negative coefficient would read as the opposite."""
    pts = SV._grid_points(np.array([-3.0, -3.0, -3.0]), (13, 13, 13), 0.5)
    rng = np.random.RandomState(0)
    occ = SV.Occupancy(points=pts, values=rng.uniform(0.0, 0.1, len(pts)), frames=1)
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    coeff = SV.fit_coefficients(occ, centres, alpha=0.5)
    assert np.all(coeff >= 0.0)


def test_no_sites_give_no_coefficients():
    occ = SV.Occupancy(points=np.zeros((4, 3)), values=np.zeros(4), frames=1)
    assert SV.fit_coefficients(occ, np.zeros((0, 3)), alpha=0.5).size == 0


# ---------------------------------------------------------------------------- the field


def test_the_water_field_is_a_field_of_hydration_sites():
    centres = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    f = SV.water_field(centres, np.array([0.8, 0.4]), name="x")
    assert f.n_atoms == 2
    assert f.weights.shape == (2, 1)
    assert np.allclose(f.coords, centres)
    assert np.allclose(f.radii, SV.WATER_RADIUS)


def test_a_field_with_no_site_is_refused():
    with pytest.raises(SV.SolventError, match="no hydration site"):
        SV.water_field(np.zeros((0, 3)), np.zeros(0))


def test_a_coefficient_per_site_is_required():
    with pytest.raises(SV.SolventError, match="coefficients"):
        SV.water_field(np.zeros((3, 3)), np.zeros(2))


def test_the_water_field_goes_through_the_kernel_machinery():
    """The point of making it a field: the ordinary kernel code must take it unchanged."""
    from cmfqsar.model import CMFModel

    fs = SV.solvent_fieldset(alpha=0.5)
    fields = [SV.water_field(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
                             np.array([0.8, 0.4]), name="a"),
              SV.water_field(np.array([[0.5, 0.0, 0.0], [3.5, 0.0, 0.0]]),
                             np.array([0.6, 0.5]), name="b"),
              SV.water_field(np.array([[9.0, 9.0, 9.0]]), np.array([0.9]), name="c")]
    table = AtomTable.from_molecules(fields)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = np.asarray(model.joint_kernel(model.prepare_training_kernels(table)), float)
    assert K.shape == (3, 3)
    assert np.allclose(np.diag(K), 1.0), "the cosine normalisation still holds"
    assert K[0, 1] > K[0, 2], "two overlapping maps are more alike than two that do not touch"


def test_the_field_set_carries_one_field_named_for_the_solvent():
    fs = SV.solvent_fieldset(alpha=1.5, name="solvent")
    assert [s.name for s in fs.specs] == ["solvent"]
    assert fs.specs[0].alpha == 1.5 and fs.specs[0].kind == "gauss"


# --------------------------------------------------------------------------- reading it


def test_the_site_weights_say_which_way_the_model_wants_the_water():
    a = SV.water_field(np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]), np.array([0.8, 0.4]))
    b = SV.water_field(np.array([[0.0, 0.0, 0.0]]), np.array([0.5]))
    got = SV.site_contributions(None, [a, b], [2.0, -1.0])
    assert np.allclose(got[0], [1.6, 0.8]), "a positive coefficient asks for more water"
    assert np.allclose(got[1], [-0.5]), "a negative one asks for less"


def test_one_coefficient_per_molecule_is_required():
    a = SV.water_field(np.array([[0.0, 0.0, 0.0]]), np.array([0.8]))
    with pytest.raises(SV.SolventError, match="dual coefficients"):
        SV.site_contributions(None, [a], [1.0, 2.0])


# ------------------------------------------------------------------------- the whole chain


def test_the_chain_gives_a_field_that_can_be_built_on(tmp_path):
    pdb, dcd, ref, _lig, held, _n = _system(tmp_path, held_fraction=1.0, tag="chain")
    field, occ, heights = SV.field_for(pdb, dcd, reference=ref, alpha=0.5, name="one",
                                       min_occupancy=0.5)
    assert field.n_atoms == len(heights) >= 1
    assert occ.frames == 20
    assert float(np.linalg.norm(field.coords - held, axis=1).min()) < 1.0
    fs = SV.solvent_fieldset(alpha=0.5)
    table = AtomTable.from_molecules([field])
    assert table.coords.shape[0] == field.n_atoms


def test_a_ligand_with_no_site_says_so(tmp_path):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=0.0, tag="nosite")
    with pytest.raises(SV.SolventError, match="hydration site"):
        SV.field_for(pdb, dcd, reference=ref, alpha=0.5, min_occupancy=0.9)


# ----------------------------------------------------------------------- the benchmark


def _bench():
    sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
    import run_solvent
    return run_solvent


def test_the_controls_keep_the_same_number_of_sites_per_ligand():
    """A control that changed the count would test two things at once."""
    rs = _bench()
    rng = np.random.RandomState(0)
    sites = {"a": (rng.normal(size=(5, 3)), rng.random(5), rng.random(5)),
             "b": (rng.normal(size=(3, 3)), rng.random(3), rng.random(3))}
    names = ["a", "b"]
    for made in (rs.scrambled_sites(sites, names), rs.shared_sites(sites, names)):
        assert set(made) == set(names)
        for n in names:
            assert len(made[n][1]) == len(made[n][0])
    assert [len(rs.scrambled_sites(sites, names)[n][0]) for n in names] == [5, 3]


def test_random_sites_stay_inside_the_box_the_real_ones_span():
    rs = _bench()
    rng = np.random.RandomState(1)
    sites = {"a": (rng.uniform(0, 10, size=(6, 3)), rng.random(6), rng.random(6))}
    made = rs.scrambled_sites(sites, ["a"], seed=2)
    every = sites["a"][0]
    assert np.all(made["a"][0] >= every.min(axis=0) - 1e-9)
    assert np.all(made["a"][0] <= every.max(axis=0) + 1e-9)


def test_one_shared_set_of_sites_gives_every_ligand_the_same_field():
    rs = _bench()
    rng = np.random.RandomState(3)
    sites = {"a": (rng.normal(size=(4, 3)), rng.random(4), rng.random(4)),
             "b": (rng.normal(size=(2, 3)), rng.random(2), rng.random(2))}
    made = rs.shared_sites(sites, ["a", "b"])
    assert np.allclose(made["a"][0], made["b"][0])
    assert np.allclose(made["a"][0], sites["a"][0]), "the first ligand's sites are the shared set"


def test_mixing_a_kernel_with_itself_changes_nothing():
    rs = _bench()
    K = np.array([[1.0, 0.3], [0.3, 1.0]])
    assert np.allclose(rs.mixed(K, K, 0.4), K)
    assert np.allclose(rs.mixed(K, np.eye(2), 0.0), K)
    assert np.allclose(rs.mixed(K, np.eye(2), 1.0), np.eye(2))


def test_the_places_gather_the_sites_of_several_ligands():
    """Two ligands with a site at the same spot make one place, not two."""
    rs = _bench()
    here = np.array([[0.0, 0.0, 0.0]])
    far = np.array([[10.0, 0.0, 0.0]])
    sites = {"a": (np.vstack([here, far]), np.array([0.5, 0.5]), np.array([0.8, 0.8])),
             "b": (np.vstack([here + 0.3, far]), np.array([0.5, 0.5]), np.array([0.7, 0.7])),
             "c": (np.vstack([here - 0.2, far]), np.array([0.5, 0.5]), np.array([0.9, 0.9]))}
    names = ["a", "b", "c"]
    y = np.array([5.0, 6.0, 7.0])
    places, lam = rs.site_places(sites, names, y, alpha=0.5, folds=3)
    assert len(places) == 2, f"{len(places)} places from two spots"
    assert max(p["ligands"] for p in places) == 3
    assert all(0.0 <= p["occupancy"] <= 1.0 for p in places)
    assert lam > 0


def test_the_verdict_names_the_controls_and_the_scope():
    rs = _bench()
    rec = {"set": "toy", "n": 25, "models": {
        "fields": {"q2": 0.5, "delta": 0.0, "lo": 0.0, "hi": 0.0},
        "water_mixed": {"q2": 0.52, "mix": 0.3, "delta": 0.02, "lo": -0.1, "hi": 0.15},
        "water_alone": {"q2": 0.2, "delta": -0.3, "lo": -0.5, "hi": -0.1},
        "scrambled_alone": {"q2": -0.05, "delta": -0.55, "lo": -0.8, "hi": -0.3},
        "shared_alone": {"q2": -0.1, "delta": -0.6, "lo": -0.9, "hi": -0.3}}}
    text = " ".join(rs.verdict([rec]))
    assert "adds nothing that can be measured" in text
    assert "positions are carrying something" in text
    assert "What this rests on" in text


def test_the_verdict_says_so_when_the_positions_carry_nothing():
    rs = _bench()
    rec = {"set": "toy", "n": 25, "models": {
        "fields": {"q2": 0.5, "delta": 0.0, "lo": 0.0, "hi": 0.0},
        "water_mixed": {"q2": 0.52, "mix": 0.3, "delta": 0.02, "lo": -0.1, "hi": 0.15},
        "water_alone": {"q2": 0.1, "delta": -0.4, "lo": -0.6, "hi": -0.2},
        "scrambled_alone": {"q2": 0.15, "delta": -0.35, "lo": -0.6, "hi": -0.1},
        "shared_alone": {"q2": -0.1, "delta": -0.6, "lo": -0.9, "hi": -0.3}}}
    text = " ".join(rs.verdict([rec]))
    assert "positions are carrying nothing" in text


def test_the_report_says_what_is_missing_when_nothing_has_been_run(tmp_path):
    rs = _bench()
    text = rs.report(results=str(tmp_path), path=str(tmp_path / "t.md"))
    assert "make_md_trajectories.py" in text
    assert os.path.isfile(str(tmp_path / "t.md"))


# --------------------------------------------------------------------------- the sweep


def test_a_coefficient_belongs_to_one_attenuation():
    """The same map fitted at two widths gives two sets, which is why both are stored."""
    pts = SV._grid_points(np.array([-4.0, -4.0, -4.0]), (17, 17, 17), 0.5)
    centre = np.array([[0.0, 0.0, 0.0]])
    occ = SV.Occupancy(points=pts, values=0.6 * np.exp(-0.5 * 0.5 * (pts ** 2).sum(axis=1)),
                       frames=1)
    got = SV.coefficients_at(occ, centre, [0.5, 2.0])
    assert sorted(got) == ["0.5", "2"]
    assert got["0.5"][0] == pytest.approx(0.6, rel=1e-6), "the width it was made at"
    assert abs(got["2"][0] - 0.6) > 0.01, "another width needs another coefficient"


def test_a_stricter_threshold_keeps_fewer_sites(tmp_path):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=1.0, n_waters=4,
                                             tag="strict")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    loose, _h1 = SV.hydration_sites(occ, min_occupancy=0.2, min_distance=2.4)
    tight, _h2 = SV.hydration_sites(occ, min_occupancy=0.8, min_distance=2.4)
    assert len(tight) <= len(loose)


def test_a_wider_spacing_keeps_fewer_sites(tmp_path):
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=1.0, n_waters=4,
                                             tag="wide")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    near, _h1 = SV.hydration_sites(occ, min_occupancy=0.2, min_distance=1.0)
    far, _h2 = SV.hydration_sites(occ, min_occupancy=0.2, min_distance=6.0)
    assert len(far) <= len(near)


def test_the_sweep_names_the_ligands_a_definition_leaves_without_a_site(tmp_path):
    rs = _bench()
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=0.4, tag="miss")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    maps = {"a": occ, "b": occ}
    sites, missing = rs.sites_at(maps, ["a", "b"], 0.5, min_occupancy=0.99, min_distance=2.4)
    assert sites == {} and missing == ["a", "b"]
    sites, missing = rs.sites_at(maps, ["a", "b"], 0.5, min_occupancy=0.2, min_distance=2.4)
    assert missing == [] and set(sites) == {"a", "b"}
    assert len(sites["a"]) == 3, "centres, coefficients and occupancies"


def test_the_sweep_section_says_when_a_control_beats_the_real_maps():
    """The warning that matters: at a loose definition a random block can do as well."""
    rs = _bench()
    sweep = {"set": "toy", "n": 25, "reference": 0.5,
             "occupancies": [0.3, 0.6], "distances": [2.4],
             "rows": [dict(min_occupancy=0.3, min_distance=2.4, missing=0, sites=30.0,
                           sites_min=20, sites_max=40, alpha=2.0, mix=0.3, mixed=0.5,
                           alone=-0.02, control=0.44, delta=0.0, lo=-0.1, hi=0.1),
                      dict(min_occupancy=0.6, min_distance=2.4, missing=0, sites=4.0,
                           sites_min=2, sites_max=6, alpha=2.0, mix=0.3, mixed=0.52,
                           alone=0.55, control=-0.09, delta=0.02, lo=-0.1, hi=0.2)]}
    text = " ".join(rs.sweep_section(sweep))
    assert "random-site control does as well as the real maps or better" in text
    assert "should not be quoted as this item's result" in text
    assert "the stronger the sites, the more they carry" in text


def test_a_count_keeps_only_the_strongest_sites(tmp_path):
    """A definition can be given as a count, which is the only form that travels between sets."""
    rs = _bench()
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=1.0, n_waters=4,
                                             tag="count")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    maps = {"a": occ, "b": occ}
    many, missing = rs.sites_at(maps, ["a", "b"], 0.5, min_occupancy=0.2, min_distance=1.0)
    assert missing == []
    few, missing = rs.sites_at(maps, ["a", "b"], 0.5, min_occupancy=0.2, min_distance=1.0,
                               max_sites=2)
    assert missing == []
    assert len(few["a"][0]) <= 2
    assert len(few["a"][0]) <= len(many["a"][0])
    # the strongest sites are the ones kept, in the same order
    assert np.allclose(few["a"][0], many["a"][0][:len(few["a"][0])])


def test_the_count_section_reports_a_held_out_test_that_failed():
    """A held-out set that carries nothing has to be reported as a failure, not as a caution."""
    rs = _bench()
    counts = [dict(set="one", n=25, max_sites=5, min_occupancy=0.3, min_distance=2.4,
                   sites=5.0, sites_min=5, sites_max=5, strongest=0.8, reference=0.5,
                   reference_alpha=1.0, mixed=0.52, mix=0.3, mixed_alpha=1.0, alone=0.55,
                   alone_alpha=1.0, scrambled=-0.05, shared=-0.1, delta=0.02, lo=-0.1, hi=0.15),
              dict(set="held", n=36, max_sites=5, min_occupancy=0.3, min_distance=2.4,
                   sites=5.0, sites_min=5, sites_max=5, strongest=0.8, reference=0.5,
                   reference_alpha=1.0, mixed=0.49, mix=0.1, mixed_alpha=1.0, alone=-0.04,
                   alone_alpha=1.0, scrambled=-0.02, shared=-0.1, delta=-0.01, lo=-0.12,
                   hi=0.09)]
    text = " ".join(rs.count_section(counts, held_out="held"))
    assert "not supported" in text
    assert "improves no set beyond the interval" in text
    assert "on 1 of the 2 sets" in text, "the set that does carry something is named"


def test_the_count_section_reports_a_held_out_test_that_passed():
    rs = _bench()
    counts = [dict(set="held", n=36, max_sites=5, min_occupancy=0.3, min_distance=2.4,
                   sites=5.0, sites_min=5, sites_max=5, strongest=0.8, reference=0.5,
                   reference_alpha=1.0, mixed=0.62, mix=0.3, mixed_alpha=1.0, alone=0.40,
                   alone_alpha=1.0, scrambled=-0.03, shared=-0.1, delta=0.12, lo=0.02,
                   hi=0.23)]
    text = " ".join(rs.count_section(counts, held_out="held"))
    assert "could have failed and did not" in text
    assert "improves the model beyond the interval on 1 of the 1 sets" in text
    assert "not supported" not in text


def test_the_count_section_says_when_a_count_loses_a_set_its_own_result():
    """A reformulation that gives less than the threshold it replaced is refuted by that set."""
    rs = _bench()
    counts = [dict(set="one", n=25, max_sites=5, min_occupancy=0.3, min_distance=2.4,
                   sites=5.0, sites_min=5, sites_max=5, strongest=0.8, reference=0.5,
                   reference_alpha=1.0, mixed=0.45, mix=0.3, mixed_alpha=1.0, alone=0.17,
                   alone_alpha=1.0, scrambled=-0.13, shared=-0.14, delta=-0.12, lo=-0.32,
                   hi=0.07)]
    sweeps = {"one": {"set": "one", "rows": [
        dict(min_occupancy=0.6, min_distance=2.4, missing=0, sites=3.6, sites_min=1,
             sites_max=7, alpha=2.0, mix=0.3, mixed=0.62, alone=0.55, control=-0.1,
             delta=0.04, lo=-0.19, hi=0.3)]}}
    text = " ".join(rs.count_section(counts, sweeps, held_out="other"))
    assert "does not reproduce what the threshold gave, on 1 of the 1 sets" in text
    assert "+0.170 at 5.0 sites against +0.550 at 3.6 sites" in text
    assert "every site kept is strongly occupied" in text
    quiet = " ".join(rs.count_section(counts, None, held_out="other"))
    assert "does not reproduce" not in quiet, "no sweep, no comparison"


def _rule_row(name, alone, scrambled, delta, lo, hi, floor=0.6, sites=3.6):
    return dict(set=name, n=25, floor=floor, min_distance=2.4, floors=[0.3, 0.6, 0.9],
                sites=sites, sites_min=1, sites_max=7, reference=0.575, reference_alpha=1.0,
                mixed=0.6, mix=0.3, mixed_alpha=1.0, alone=alone, alone_alpha=2.0,
                scrambled=scrambled, shared=-0.14, delta=delta, lo=lo, hi=hi)


def test_the_rule_takes_the_strictest_floor_that_keeps_every_ligand(tmp_path):
    """The rule reads no activity, so it can be carried to a set it was not read from."""
    rs = _bench()
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=1.0, n_waters=4,
                                             tag="rule")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    maps = {"a": occ, "b": occ}
    floor = rs.strictest_floor(maps, ["a", "b"], floors=(0.2, 0.5, 0.9), min_distance=2.4)
    assert floor is not None
    # the floor it picks leaves every ligand a site, and the next one up does not
    for nm in maps:
        assert len(SV.hydration_sites(occ, min_occupancy=floor, min_distance=2.4)[0]) > 0
    higher = [f for f in (0.2, 0.5, 0.9) if f > floor]
    for f in higher[:1]:
        assert len(SV.hydration_sites(occ, min_occupancy=f, min_distance=2.4)[0]) == 0


def test_the_rule_gives_nothing_when_even_the_loosest_floor_is_too_strict(tmp_path):
    rs = _bench()
    pdb, dcd, ref, _lig, _held, _n = _system(tmp_path, held_fraction=0.1, tag="rulenone")
    occ = SV.occupancy(pdb, dcd, reference=ref)
    assert rs.strictest_floor({"a": occ}, ["a"], floors=(0.99,), min_distance=2.4) is None


def test_the_rule_section_reports_a_held_out_test_that_held():
    rs = _bench()
    rules = [_rule_row("one", 0.554, -0.096, 0.042, -0.19, 0.30),
             _rule_row("held", 0.31, -0.03, 0.28, 0.04, 0.56, floor=0.7, sites=6.7)]
    text = " ".join(rs.rule_section(rules, held_out="held"))
    assert "could have failed and did not" in text
    assert "control on 2 of the 2 sets" in text
    assert "on 1 of the 2 sets (held)" in text, "the set it helps is named"


def test_the_rule_section_reports_a_rule_that_misses_its_own_evidence():
    """A rule that does not pick the definition it was read from is refuted by that set."""
    rs = _bench()
    rules = [_rule_row("one", 0.169, 0.100, 0.040, -0.18, 0.22, floor=0.8, sites=3.0)]
    sweeps = {"one": {"set": "one", "rows": [
        dict(min_occupancy=0.7, min_distance=2.4, missing=0, sites=6.7, sites_min=1,
             sites_max=11, alpha=2.0, mix=0.3, mixed=0.536, alone=0.312, control=-0.032,
             delta=0.276, lo=0.037, hi=0.560)]}}
    text = " ".join(rs.rule_section(rules, sweeps, held_out="other"))
    assert "does not pick the definition it was read from, on 1 of the 1 sets" in text
    assert "strictest **tried** was taken for the strictest **possible**" in text
    assert "ahead by less than 0.20" in text, "a small margin is not read as a result"


def test_the_rule_section_reports_a_held_out_test_that_failed():
    rs = _bench()
    rules = [_rule_row("one", 0.554, -0.096, 0.042, -0.19, 0.30),
             _rule_row("held", 0.02, 0.09, -0.08, -0.29, -0.01, floor=0.6, sites=4.1)]
    text = " ".join(rs.rule_section(rules, held_out="held"))
    assert "failed a test on a set they were not read from" in text
    assert "control on 1 of the 2 sets" in text
    assert "makes the model worse beyond the interval" in text


def test_the_rule_section_is_empty_without_a_measurement():
    rs = _bench()
    assert rs.rule_section([]) == []


def test_the_count_section_is_empty_without_a_measurement():
    rs = _bench()
    assert rs.count_section([]) == []


def test_the_sweep_section_says_when_a_definition_leaves_no_sites():
    rs = _bench()
    sweep = {"set": "toy", "n": 25, "reference": 0.5, "occupancies": [0.9],
             "distances": [2.4],
             "rows": [dict(min_occupancy=0.9, min_distance=2.4, missing=7, sites=None)]}
    text = " ".join(rs.sweep_section(sweep))
    assert "7 ligand(s) with no site" in text
    assert "nothing can be read from this" in text


def test_the_stored_sites_can_be_rebuilt_from_the_trajectories():
    """The committed file has to be what the script produces, or it is not provenance."""
    import json

    path = os.path.join(HERE, "..", "examples", "md", "mcl1_water.json")
    runs = os.path.join(HERE, "..", "examples", "md", "work", "mcl1", "runs")
    ready = os.path.join(HERE, "..", "examples", "md", "work", "mcl1", "4HW3_ready.pdb")
    if not (os.path.isfile(path) and os.path.isdir(runs) and os.path.isfile(ready)):
        pytest.skip("the trajectories are not in this checkout")
    with open(path) as fh:
        d = json.load(fh)
    row = d["ligands"][0]
    nm = row["name"]
    field, occ, _h = SV.field_for(os.path.join(runs, f"{nm}.pdb"),
                                  os.path.join(runs, f"{nm}.dcd"), reference=ready,
                                  alpha=d["alpha"], name=nm,
                                  min_occupancy=d["min_occupancy"])
    stored = np.asarray(row["sites"], float)
    assert len(stored) == field.n_atoms
    assert np.allclose(stored[:, :3], field.coords, atol=1e-3)
    assert np.allclose(stored[:, 3], np.asarray(field.weights).ravel(), atol=1e-4)
