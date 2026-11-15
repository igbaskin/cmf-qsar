"""Tests of molecules perceived inside crystals (plan item M8, `cmfqsar/molcrystal.py`)."""
import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar import molcrystal as M
from cmfqsar.fields import FieldSet, build_fields

FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]


def fieldset():
    return FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                           param_scheme="rdkit")


def embedded(smiles, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def cell_of(parts, box=12.0, wrap=True):
    """Molecules (RDKit, with a shift each) in a cubic cell, wrapped into it."""
    symbols, pos = [], []
    for m, shift in parts:
        p = m.GetConformer().GetPositions()
        symbols += [a.GetSymbol() for a in m.GetAtoms()]
        pos += list(p - p.mean(axis=0) + np.asarray(shift, float))
    at = Atoms(symbols=symbols, positions=pos, cell=np.eye(3) * box, pbc=True)
    if wrap:
        at.wrap()
    return at


def test_a_molecule_cut_by_the_cell_boundary_is_made_whole():
    benzene = embedded("c1ccccc1")
    at = cell_of([(benzene, [0.3, 0.2, 0.1])], box=9.0)       # centred on a cell corner
    p = at.get_positions()
    assert p[:, 0].max() - p[:, 0].min() > 6.0                  # split across the faces
    mols = M.cell_molecules(at)
    assert len(mols) == 1
    c = mols[0]
    assert c.smiles == "c1ccccc1" and c.charge == 0
    assert c.inchikey == Chem.MolToInchiKey(benzene)
    xyz = c.mol.GetConformer().GetPositions()
    d = np.linalg.norm(xyz[:, None] - xyz[None], axis=2)
    assert d.max() < 5.5                                         # whole: no 9 A gap left
    assert np.any(c.image != 0)


def test_two_molecules_of_a_co_crystal_are_told_apart():
    at = cell_of([(embedded("c1ccccc1"), [3, 3, 3]), (embedded("O"), [8, 8, 8])])
    mols = M.cell_molecules(at)
    found = M.distinct_molecules(mols)
    assert len(found) == 2
    assert sorted(c.smiles for c in mols) == ["O", "c1ccccc1"]


def test_a_salt_gets_charges_that_add_up_to_zero():
    cation = embedded("C[NH3+]")
    chloride = Chem.AddHs(Chem.MolFromSmiles("[Cl-]"))
    AllChem.EmbedMolecule(chloride, randomSeed=0)
    at = cell_of([(cation, [4, 4, 4]), (chloride, [8.0, 4.0, 4.0])], box=10.0)
    mols = M.cell_molecules(at)
    charges = sorted(c.charge for c in mols)
    assert charges == [-1, 1]
    assert sorted(c.smiles for c in mols) == ["C[NH3+]", "[Cl-]"]


def test_copies_of_one_molecule_share_their_perception():
    water = embedded("O")
    at = cell_of([(water, [2, 2, 2]), (water, [7, 7, 7]), (water, [2, 7, 2])], box=10.0)
    mols = M.cell_molecules(at)
    assert len(mols) == 3 and M.distinct_molecules(mols) == {mols[0].inchikey: 3}


def test_a_framework_is_not_a_molecular_crystal():
    chain = Atoms("C2H4", positions=[[0, 0, 0], [1.5, 0, 0], [0, 1.0, 0], [0, -0.5, 0.9],
                                     [1.5, 1.0, 0], [1.5, -0.5, 0.9]],
                  cell=np.diag([3.0, 8.0, 8.0]), pbc=True)
    with pytest.raises(M.PerceptionError, match="framework"):
        M.cell_molecules(chain)


def test_missing_hydrogens_overlaps_and_metals_are_refused():
    heavy = Atoms("C6", positions=[[np.cos(t) * 1.39, np.sin(t) * 1.39, 0]
                                   for t in np.arange(6) * np.pi / 3],
                  cell=np.eye(3) * 10, pbc=True)
    with pytest.raises(M.PerceptionError, match="no hydrogen"):
        M.cell_molecules(heavy)
    two = Atoms("CC", positions=[[0, 0, 0], [0.3, 0, 0]], cell=np.eye(3) * 10, pbc=True)
    with pytest.raises(M.PerceptionError, match="overlapping"):
        M.check_overlaps(two)
    metal = Atoms("ZnH2", positions=[[0, 0, 0], [1.5, 0, 0], [-1.5, 0, 0]],
                  cell=np.eye(3) * 10, pbc=True)
    with pytest.raises(M.PerceptionError, match="outside the organic set: Zn"):
        M.cell_molecules(metal)


def test_partly_occupied_sites_are_refused(tmp_path):
    cif = tmp_path / "x.cif"
    cif.write_text("data_x\n_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 10\n"
                   "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
                   "_symmetry_space_group_name_H-M 'P 1'\nloop_\n_atom_site_label\n"
                   "_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n"
                   "_atom_site_fract_z\n_atom_site_occupancy\nC1 C 0.1 0.1 0.1 0.5\n"
                   "H1 H 0.2 0.1 0.1 1.0\n")
    with pytest.raises(M.PerceptionError, match="partly occupied"):
        M.read_molecular_crystal(str(cif))


def test_every_copy_carries_the_fields_of_its_molecule():
    fs = fieldset()
    acid = embedded("CC(=O)O")
    at = cell_of([(acid, [3, 3, 3]), (acid, [8, 8, 8])], box=11.0)
    mols = M.cell_molecules(at)
    cf = M.molecular_crystal_fields(at, fs, molecules=mols)
    assert cf.weights.shape == (len(at), len(FIELDS)) and cf.pbc == (True, True, True)
    ref = build_fields(mols[0].mol, fs, "acid")
    for c in mols:
        assert np.allclose(cf.weights[c.atom_index], ref.weights)
    assert np.allclose(cf.coords, at.get_positions())           # the cell, not the whole copies
    assert cf.charges.sum() == pytest.approx(0.0, abs=1e-9)


def test_the_periodic_kernel_reads_the_molecular_weights():
    from cmfqsar import periodic

    fs = fieldset()
    at = cell_of([(embedded("CO"), [3, 3, 3]), (embedded("CO", 1), [8, 7, 6])], box=11.0)
    cf = M.molecular_crystal_fields(at, fs)
    spec = periodic.PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)
    row = periodic.periodic_autocorrelation(cf, spec)
    assert np.all(np.isfinite(row)) and np.any(row != 0)
    big = at.repeat((2, 1, 1))
    row2 = periodic.periodic_autocorrelation(M.molecular_crystal_fields(big, fs), spec)
    assert np.allclose(row, row2, rtol=1e-9, atol=1e-12)
