"""Tests for cmfqsar/trajectory.py: a simulation read as a weighted conformational ensemble.

Everything here runs on trajectories built for the purpose, so the tests need no simulation
package and no downloaded data. The system is a ligand beside a short peptide with a few waters,
written as a PDB topology and a DCD of frames; what the frames contain is chosen to put each
step of the chain under a claim that can fail.

The claim that matters most is the superposition. A frame of a real simulation carries the whole
system's drift and rotation as well as the ligand's own motion, and only the second is a
conformational difference. `test_a_system_that_only_rotates_gives_one_conformation` builds frames
that differ by nothing but a rigid rotation of everything: if the binding-site fit works, they
collapse to a single conformation, and if it does not, they look like dozens of different poses.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import trajectory as TJ  # noqa: E402
from cmfqsar.fields import FieldSet, conformer_weights  # noqa: E402

mdtraj = pytest.importorskip("mdtraj", reason="the trajectory reader needs mdtraj or MDAnalysis")

HAVE_MDA = "mdanalysis" in TJ.available_engines()


# --------------------------------------------------------------------------- building a system


def _ligand():
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1"))
    AllChem.EmbedMultipleConfs(m, numConfs=8, randomSeed=7)
    AllChem.MMFFOptimizeMoleculeConfs(m)
    for k, a in enumerate(m.GetAtoms()):
        ri = Chem.AtomPDBResidueInfo()
        ri.SetResidueName("LIG")
        ri.SetResidueNumber(900)
        ri.SetIsHeteroAtom(True)
        ri.SetName(f"{a.GetSymbol()}{k:<3d}"[:4])
        a.SetMonomerInfo(ri)
    return m


def _peptide():
    p = Chem.MolFromSequence("AAGAA")
    AllChem.EmbedMolecule(p, randomSeed=3)
    AllChem.MMFFOptimizeMolecule(p)
    return p


def _waters(n, origin):
    """`n` water molecules, each three atoms, spread out from `origin`."""
    rw = Chem.RWMol()
    conf = []
    for k in range(n):
        base = np.asarray(origin, float) + np.array([3.0 * (k + 1), 0.0, 0.0])
        for sym, off in (("O", (0, 0, 0)), ("H", (0.76, 0.59, 0)), ("H", (-0.76, 0.59, 0))):
            a = Chem.Atom(sym)
            ri = Chem.AtomPDBResidueInfo()
            ri.SetResidueName("HOH")
            ri.SetResidueNumber(1000 + k)
            ri.SetIsHeteroAtom(True)
            ri.SetName(f" {sym}  "[:4])
            a.SetMonomerInfo(ri)
            rw.AddAtom(a)
            conf.append(base + np.asarray(off, float))
    mol = rw.GetMol()
    c = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(conf):
        c.SetAtomPosition(i, Chem.rdGeometry.Point3D(*[float(v) for v in xyz]))
    mol.AddConformer(c)
    return mol


def _rotation(angle, axis=(0.0, 0.0, 1.0)):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _write(tmp_path, frames, template_mol, tag="sys"):
    """A PDB topology and a DCD of `frames` (each (n_atoms, 3) in angstrom)."""
    pdb = str(tmp_path / f"{tag}.pdb")
    Chem.MolToPDBFile(template_mol, pdb, confId=template_mol.GetConformer().GetId())
    top = mdtraj.load(pdb).topology
    traj = mdtraj.Trajectory(np.stack(frames) / 10.0, top)
    dcd = str(tmp_path / f"{tag}.dcd")
    traj.save_dcd(dcd)
    return pdb, dcd


def _system(tmp_path, n_waters=3, move_ligand=True, rotate=True, n_frames=12, tag="sys"):
    """Peptide + ligand + waters; the ligand changes conformation, the system rotates."""
    lig, pep = _ligand(), _peptide()
    lig_xyz0 = np.asarray(lig.GetConformer(0).GetPositions())
    pep_xyz = np.asarray(pep.GetConformer().GetPositions())
    # put the peptide beside the ligand, close enough to be a binding site
    pep_xyz = pep_xyz - pep_xyz.mean(axis=0) + lig_xyz0.mean(axis=0) + np.array([5.0, 0.0, 0.0])
    wat = _waters(n_waters, lig_xyz0.mean(axis=0) + np.array([0.0, 12.0, 0.0]))
    wat_xyz = np.asarray(wat.GetConformer().GetPositions())

    whole = Chem.RWMol(Chem.CombineMols(Chem.CombineMols(pep, lig), wat))
    mol = whole.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, xyz in enumerate(np.vstack([pep_xyz, lig_xyz0, wat_xyz])):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*[float(v) for v in xyz]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf)

    n_pep, n_lig = pep.GetNumAtoms(), lig.GetNumAtoms()
    frames = []
    for k in range(n_frames):
        lxyz = np.asarray(lig.GetConformer(k % lig.GetNumConformers()).GetPositions()) \
            if move_ligand else lig_xyz0
        # keep the ligand where it was put, whichever conformation it is in
        lxyz = lxyz - lxyz.mean(axis=0) + lig_xyz0.mean(axis=0)
        xyz = np.vstack([pep_xyz, lxyz, wat_xyz])
        if rotate:
            R = _rotation(0.3 * k)
            xyz = xyz @ R.T + np.array([2.0 * k, -1.0 * k, 0.5 * k])
        frames.append(xyz)
    pdb, dcd = _write(tmp_path, frames, mol, tag=tag)
    return pdb, dcd, lig, (n_pep, n_lig)


# ------------------------------------------------------------------------------- reading


def test_an_engine_is_available():
    assert TJ.available_engines(), "the tests need mdtraj or MDAnalysis"


def test_a_missing_file_is_named():
    with pytest.raises(TJ.TrajectoryError, match="no such file"):
        TJ.load(os.path.join(HERE, "no_such_topology.pdb"))


def test_an_unknown_engine_is_refused(tmp_path):
    pdb, dcd, _lig, _n = _system(tmp_path)
    with pytest.raises(TJ.TrajectoryError, match="mdtraj, mdanalysis or auto"):
        TJ.load(pdb, dcd, engine="gromacs")


def test_the_reader_gives_frames_atoms_and_residues(tmp_path):
    pdb, dcd, _lig, (n_pep, n_lig) = _system(tmp_path, n_waters=3, n_frames=12)
    t = TJ.load(pdb, dcd)
    assert t.n_frames == 12
    assert t.n_atoms == n_pep + n_lig + 9
    assert "LIG" in t.residues and "HOH" in t.residues and "ALA" in t.residues


def test_the_stride_takes_every_nth_frame(tmp_path):
    pdb, dcd, _lig, _n = _system(tmp_path, n_frames=12)
    assert TJ.load(pdb, dcd, stride=3).n_frames == 4


@pytest.mark.skipif(not HAVE_MDA, reason="MDAnalysis is not installed")
def test_the_two_readers_agree(tmp_path):
    """Either reader must give the same geometry, or a result would depend on what is installed."""
    pdb, dcd, _lig, _n = _system(tmp_path, n_frames=6)
    a = TJ.load(pdb, dcd, engine="mdtraj")
    b = TJ.load(pdb, dcd, engine="mdanalysis")
    assert a.n_atoms == b.n_atoms and a.n_frames == b.n_frames
    assert a.elements == b.elements
    assert a.residues == b.residues
    assert np.allclose(a.coords, b.coords, atol=1e-3)


# ------------------------------------------------------------------------------ selecting


def test_the_solvent_and_the_ions_are_dropped(tmp_path):
    pdb, dcd, _lig, (n_pep, n_lig) = _system(tmp_path, n_waters=4)
    t = TJ.load(pdb, dcd)
    stripped, dropped = TJ.strip_solvent(t)
    assert dropped == 12, "four waters of three atoms"
    assert stripped.n_atoms == n_pep + n_lig
    assert "HOH" not in stripped.residues


def test_the_ligand_is_found_by_name_and_by_size(tmp_path):
    pdb, dcd, _lig, (_n_pep, n_lig) = _system(tmp_path)
    t, _d = TJ.strip_solvent(TJ.load(pdb, dcd))
    by_name = TJ.ligand_indices(t, "LIG")
    by_size = TJ.ligand_indices(t)
    assert len(by_name) == n_lig
    assert np.array_equal(by_name, by_size), "the largest non-protein residue is the ligand"


def test_an_unknown_residue_name_says_what_the_file_holds(tmp_path):
    pdb, dcd, _lig, _n = _system(tmp_path)
    t = TJ.load(pdb, dcd)
    with pytest.raises(TJ.TrajectoryError, match="no residue named"):
        TJ.ligand_indices(t, "XYZ")


def test_the_binding_site_is_backbone_atoms_near_the_ligand(tmp_path):
    pdb, dcd, _lig, _n = _system(tmp_path)
    t, _d = TJ.strip_solvent(TJ.load(pdb, dcd))
    lig = TJ.ligand_indices(t, "LIG")
    fit = TJ.site_backbone(t, lig, radius=8.0)
    assert fit.size >= 3
    assert all(t.atom_names[i].upper() in TJ.BACKBONE_ATOMS for i in fit)
    assert all(t.residues[i] in TJ.AMINO_ACIDS for i in fit)
    assert TJ.site_backbone(t, lig, radius=0.1).size == 0, "nothing is near at that radius"


def test_a_system_without_a_protein_has_no_binding_site(tmp_path):
    lig = _ligand()
    xyz = np.asarray(lig.GetConformer(0).GetPositions())
    pdb, dcd = _write(tmp_path, [xyz, xyz + 0.1], lig, tag="ligonly")
    t = TJ.load(pdb, dcd)
    assert TJ.site_backbone(t, TJ.ligand_indices(t, "LIG")).size == 0


# -------------------------------------------------------------------------- superimposing


def test_a_rigid_rotation_is_undone_exactly():
    rng = np.random.RandomState(0)
    base = rng.normal(size=(12, 3))
    R = _rotation(0.7, (0.3, 1.0, -0.2))
    coords = np.stack([base, base @ R.T + np.array([4.0, -2.0, 1.0])])
    out = TJ.superimpose(coords, range(12))
    assert np.allclose(out[1], base, atol=1e-8)


def test_a_superposition_needs_three_atoms():
    with pytest.raises(TJ.TrajectoryError, match="three atoms"):
        TJ.superimpose(np.zeros((2, 5, 3)), [0, 1])


def test_the_whole_system_moves_with_the_fitted_atoms():
    """Only then does what is left in the ligand's coordinates mean motion relative to the fit."""
    rng = np.random.RandomState(1)
    base = rng.normal(size=(20, 3))
    R = _rotation(0.5)
    coords = np.stack([base, base @ R.T])
    out = TJ.superimpose(coords, range(10))        # fitted on the first half
    assert np.allclose(out[1], base, atol=1e-8), "the unfitted half followed the fitted one"


# ------------------------------------------------------------------------------ thinning


def test_identical_frames_become_one_cluster():
    frame = np.random.RandomState(2).normal(size=(10, 3))
    coords = np.stack([frame] * 7)
    reps, counts, used = TJ.leader_cluster(coords, cutoff=0.5)
    assert len(reps) == 1 and counts.tolist() == [7]
    assert used == 0.5, "the radius did not have to grow"


def test_the_populations_are_the_frame_counts():
    """Nine frames of one pose and one of another must weigh 0.9 against 0.1."""
    rng = np.random.RandomState(3)
    a = rng.normal(size=(10, 3))
    b = a + 5.0
    coords = np.stack([a] * 9 + [b])
    reps, counts, _used = TJ.leader_cluster(coords, cutoff=1.0)
    assert len(reps) == 2
    w = counts / counts.sum()
    assert np.allclose(sorted(w), [0.1, 0.9])


def test_the_cap_on_the_number_of_frames_is_met_by_growing_the_cutoff():
    rng = np.random.RandomState(4)
    coords = rng.normal(size=(60, 8, 3)) * 3.0
    reps, counts, used = TJ.leader_cluster(coords, cutoff=0.1, max_frames=6)
    assert len(reps) <= 6
    assert int(counts.sum()) == 60, "every frame belongs to exactly one cluster"
    assert used > 0.1, "the radius had to grow to meet the cap"


def test_clustering_needs_frames():
    with pytest.raises(TJ.TrajectoryError, match="no frames"):
        TJ.leader_cluster(np.zeros((0, 4, 3)))


# ------------------------------------------------------------------------------ the molecule


def test_a_template_fixes_the_bond_orders(tmp_path):
    pdb, dcd, lig, _n = _system(tmp_path)
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", template=Chem.Mol(lig), max_frames=5)
    assert e.mol.GetNumAtoms() == lig.GetNumAtoms()
    assert Chem.MolToSmiles(Chem.RemoveHs(e.mol)) == Chem.MolToSmiles(Chem.RemoveHs(lig))
    assert any(b.GetBondType() == Chem.BondType.AROMATIC for b in e.mol.GetBonds())


def test_a_template_of_the_wrong_size_is_refused(tmp_path):
    pdb, dcd, _lig, _n = _system(tmp_path)
    small = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(small, randomSeed=1)
    with pytest.raises(TJ.TrajectoryError, match="atoms"):
        TJ.read_trajectory(pdb, dcd, ligand="LIG", template=small)


def test_without_a_template_the_bonds_come_from_the_topology(tmp_path):
    pdb, dcd, lig, _n = _system(tmp_path)
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", max_frames=5)
    assert e.mol.GetNumAtoms() == lig.GetNumAtoms()
    assert e.mol.GetNumBonds() > 0


# ------------------------------------------------------------------------------ the chain


def test_a_system_that_only_rotates_gives_one_conformation(tmp_path):
    """The claim of step 3: a frame's global motion is not a conformational difference."""
    pdb, dcd, lig, _n = _system(tmp_path, move_ligand=False, rotate=True, n_frames=10,
                                tag="rot")
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", template=Chem.Mol(lig), max_frames=20,
                           cutoff=0.5)
    assert e.fitted_on == "the binding-site backbone"
    assert e.mol.GetNumConformers() == 1, (
        f"{e.mol.GetNumConformers()} poses from a system that only rotated; the superposition "
        f"did not remove the rotation")
    assert np.allclose(e.weights, [1.0])


def test_a_moving_ligand_gives_several_conformations_with_populations(tmp_path):
    pdb, dcd, lig, _n = _system(tmp_path, move_ligand=True, rotate=True, n_frames=16,
                                tag="move")
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", template=Chem.Mol(lig), max_frames=10,
                           cutoff=0.5)
    assert e.mol.GetNumConformers() > 1
    assert e.n_frames == 16
    assert int(e.populations.sum()) == 16
    assert e.weights.sum() == pytest.approx(1.0)
    assert e.stripped == 9, "three waters"
    assert "frames" in e.describe() and "thinned to" in e.describe()


def test_the_weights_reach_the_field_machinery(tmp_path):
    """The point of the module: `conformer_weighting="given"` must read what it wrote."""
    pdb, dcd, lig, _n = _system(tmp_path, n_frames=16, tag="w")
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", template=Chem.Mol(lig), max_frames=6,
                           cutoff=0.4)
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5,
                         conformers="ensemble", conformer_weighting="given")
    got = conformer_weights(e.mol, fs)
    assert len(got) == e.mol.GetNumConformers()
    assert np.allclose(got, e.weights, atol=1e-6)


def test_the_weights_are_populations_and_not_boltzmann_factors(tmp_path):
    """The item is explicit: an equilibrated trajectory has sampled the distribution already."""
    pdb, dcd, lig, _n = _system(tmp_path, n_frames=16, tag="b")
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", template=Chem.Mol(lig), max_frames=10,
                           cutoff=0.4)
    assert np.allclose(e.weights, e.populations / e.populations.sum())
    assert not e.mol.HasProp("conformer_energies"), "no energies are written, so none can be used"


def test_the_fields_are_built_from_the_ensemble(tmp_path):
    from cmfqsar.fields import build_fields
    pdb, dcd, lig, _n = _system(tmp_path, n_frames=16, tag="f")
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", template=Chem.Mol(lig), max_frames=6,
                           cutoff=0.4)
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5,
                         conformers="ensemble", conformer_weighting="given")
    mf = build_fields(e.mol, fs, "ligand")
    k = e.mol.GetNumConformers()
    assert mf.n_conformers == k, "every kept pose is a member of the ensemble"
    assert mf.coords.shape[0] == k * lig.GetNumAtoms()
    assert np.isfinite(mf.weights).all()


# ------------------------------------------------------------------------------- charges


@pytest.mark.skipif(not HAVE_MDA, reason="MDAnalysis is not installed")
def test_force_field_charges_become_a_per_atom_property(tmp_path):
    """A topology that carries charges gives a prop: field of the charges the run itself used."""
    path = str(tmp_path / "w.pdbqt")
    with open(path, "w") as fh:
        fh.write("REMARK test\n"
                 "ATOM      1  O   LIG A   1       0.000   0.000   0.117  1.00  0.00    -0.560 OA\n"
                 "ATOM      2  H1  LIG A   1       0.000   0.757  -0.469  1.00  0.00     0.280 HD\n"
                 "ATOM      3  H2  LIG A   1       0.000  -0.757  -0.469  1.00  0.00     0.280 HD\n"
                 "END\n")
    t = TJ.load(path, engine="mdanalysis")
    assert t.charges is not None and np.allclose(t.charges, [-0.56, 0.28, 0.28])
    mol = TJ.to_molecule(t, [0], [1.0], template=_water_template())
    assert mol.GetAtomWithIdx(0).GetDoubleProp(TJ.CHARGE_PROP) == pytest.approx(-0.56)


def _water_template():
    m = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(m, randomSeed=1)
    return m


def test_a_topology_without_charges_carries_none(tmp_path):
    pdb, dcd, lig, _n = _system(tmp_path, tag="noq")
    e = TJ.read_trajectory(pdb, dcd, ligand="LIG", template=Chem.Mol(lig), max_frames=4)
    assert not e.mol.GetAtomWithIdx(0).HasProp(TJ.CHARGE_PROP)


# ------------------------------------------------------------------------------- elements


@pytest.mark.parametrize("name,atom_type,element,want", [
    ("C1", "", "C", "C"),                       # the element, when the topology has one
    ("O", "OA", "", "O"),                       # an AutoDock type
    ("H1", "HD", "", "H"),
    ("C5", "CT", "", "C"),                      # a force-field type
    ("CL1", "", "", "Cl"),                      # a two-letter element in an atom name
    ("FE", "", "", "Fe"),
    ("N", "", "", "N"),
])
def test_the_element_is_read_from_whatever_the_topology_has(name, atom_type, element, want):
    assert TJ._element_of(name, atom_type, element) == want


@pytest.mark.parametrize("name,want", [("CA", "C"), ("CB", "C"), ("HG1", "H"), ("ND1", "N"),
                                       ("OG", "O"), ("SD", "S")])
def test_a_protein_atom_name_is_not_mistaken_for_an_element(name, want):
    """CA is an alpha carbon, not calcium; HG a gamma hydrogen, not mercury."""
    assert TJ._element_of(name) == want


def test_an_element_that_cannot_be_read_says_so():
    with pytest.raises(TJ.TrajectoryError, match="could not be worked out"):
        TJ._element_of("1234", "", "")


def test_an_unreadable_symbol_names_itself():
    with pytest.raises(TJ.TrajectoryError, match="not an element symbol"):
        TJ._atomic_number("Qq")


# ------------------------------------------------------------------------- the whole path


def test_the_ensemble_survives_a_round_trip_through_an_sdf(tmp_path):
    """The path a user takes: a trajectory to an SDF to a weighted ensemble in the field set."""
    sys.path.insert(0, os.path.join(HERE, "..", "examples"))
    import trajectory_ensemble as TE
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import build_fields

    pdb, dcd, lig, _n = _system(tmp_path, n_frames=16, tag="trip")
    sdf = str(tmp_path / "ens.sdf")
    tpl = str(tmp_path / "tpl.sdf")
    with Chem.SDWriter(tpl) as w:
        m = Chem.AddHs(Chem.Mol(lig), addCoords=True)
        m.SetProp("_Name", os.path.splitext(os.path.basename(pdb))[0])
        w.write(m)
    code = TE.main([pdb, dcd, "--ligand", "LIG", "--template", tpl, "-o", sdf,
                    "--max-frames", "6", "--cutoff", "0.4"])
    assert code == 0

    ds = load_dataset(sdf, ensemble_by="name")
    assert len(ds.mols) == 1
    mol = ds.mols[0]
    assert mol.GetNumConformers() > 1, "the records were collapsed into one ensemble"
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5,
                         conformers="ensemble", conformer_weighting="given")
    w = conformer_weights(mol, fs)
    assert len(w) == mol.GetNumConformers()
    assert w.sum() == pytest.approx(1.0)
    assert build_fields(mol, fs, "lig").n_conformers == mol.GetNumConformers()
