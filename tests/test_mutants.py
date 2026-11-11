"""Point mutants of a protein as fields in the frame of the mutated residue (plan item PP1).

The sample here is a change to one protein, so what these tests fix is different from what the
tests of a molecular series fix:

* a protein read from a PDB file becomes a chemically sound molecule: the right bonds, the
  right formal charges, no radical on a histidine, and the force-field charges where the
  electrostatic field reads them;
* the frame of a residue is a frame: turning the whole protein does not change what the side
  chain of a residue looks like in it, and the frame of two different positions brings their
  side chains onto each other;
* the difference field is the change and nothing else: everything the mutant shares with the
  wild type cancels, so two mutants at the same position are told apart by their side chains;
* a context is local and holds no part of the side chain it surrounds;
* the substitution reference says what it should about volume, hydropathy and charge.

Everything that needs OpenMM or PDBFixer is skipped where they are not installed, as the tests
that need ORCA or xtb are.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels, self_kernels  # noqa: E402
from cmfqsar.mutants import (AA1, AMINO_ACIDS, HYDROPATHY, MAIN_CHAIN,  # noqa: E402
                             Mutation,
                             VOLUME, backbone_frame, frame_shift, in_residue_frame,
                             protein_from_pdb, residue_atoms, residue_name,
                             side_chain_atoms, side_chain_field, side_chain_geometry,
                             site_context, site_difference, substitution_features)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXAMPLES = os.path.join(ROOT, "examples", "mutants")
BUILT = os.path.join(ROOT, "benchmarks", "cache", "megascale", "built")
DOMAIN = "1O6X"


def _openmm_there() -> bool:
    try:
        import openmm  # noqa: F401
        import pdbfixer  # noqa: F401
    except ImportError:
        return False
    return True


needs_openmm = pytest.mark.skipif(not _openmm_there(),
                                  reason="OpenMM and PDBFixer are not installed here")
needs_model = pytest.mark.skipif(not os.path.exists(os.path.join(EXAMPLES, DOMAIN, f"{DOMAIN}.pdb")),
                                 reason="the mutant example set is not built")


# --------------------------------------------------------------------------- #
# A substitution label
# --------------------------------------------------------------------------- #

def test_mutation_reads_and_writes_a_label():
    m = Mutation.parse("V1Q")
    assert (m.wild, m.position, m.new) == ("V", 1, "Q")
    assert m.label == "V1Q"
    assert m.fixer_name == "VAL-1-GLN"


@pytest.mark.parametrize("bad", ["V1", "1VQ", "B3A", "V1J", "", "V-1-Q"])
def test_mutation_refuses_what_is_not_a_substitution(bad):
    with pytest.raises(ValueError):
        Mutation.parse(bad)


def test_substitution_features_say_what_changed():
    rows, names = substitution_features([Mutation.parse("G10W"), Mutation.parse("W10G"),
                                         Mutation.parse("K5E")])
    assert names[:3] == ["d_volume", "d_hydropathy", "d_charge"]
    assert rows.shape == (3, 3 + 2 * len(AMINO_ACIDS))
    # glycine to tryptophan adds volume; the reverse takes the same amount away
    assert rows[0, 0] == pytest.approx(VOLUME["W"] - VOLUME["G"])
    assert rows[1, 0] == pytest.approx(-rows[0, 0])
    assert rows[0, 1] == pytest.approx(HYDROPATHY["W"] - HYDROPATHY["G"])
    # lysine to glutamate turns a positive side chain into a negative one
    assert rows[2, 2] == pytest.approx(-2.0)
    # the one-hot codes are one each
    assert rows[0, 3:3 + len(AMINO_ACIDS)].sum() == 1.0
    assert rows[0, 3 + len(AMINO_ACIDS):].sum() == 1.0


# --------------------------------------------------------------------------- #
# A protein as an RDKit molecule
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def model_pdb():
    path = os.path.join(EXAMPLES, DOMAIN, f"{DOMAIN}.pdb")
    if not os.path.exists(path):
        pytest.skip("the mutant example set is not built")
    with open(path) as fh:
        return fh.read()


@needs_model
def test_a_protein_reads_as_a_sound_molecule(model_pdb):
    mol = protein_from_pdb(model_pdb)
    assert mol.GetNumAtoms() > 500
    assert sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()) == 0
    # every charged atom is one the chemistry allows: a terminal or side-chain nitrogen or oxygen
    for a in mol.GetAtoms():
        if a.GetFormalCharge():
            assert a.GetSymbol() in ("N", "O")
            assert abs(a.GetFormalCharge()) == 1
    # the charge of the whole domain is a whole number of elementary charges
    assert sum(a.GetFormalCharge() for a in mol.GetAtoms()) == int(
        sum(a.GetFormalCharge() for a in mol.GetAtoms()))


@needs_model
def test_histidine_keeps_both_ring_nitrogens_without_a_radical(model_pdb):
    mol = protein_from_pdb(model_pdb)
    his = [a for a in mol.GetAtoms()
           if a.GetPDBResidueInfo().GetResidueName().strip() in ("HIS", "HID", "HIE", "HIP")]
    if not his:
        pytest.skip("this domain has no histidine")
    assert all(a.GetNumRadicalElectrons() == 0 for a in his)


@needs_model
def test_charges_from_a_file_reach_the_electrostatic_field(model_pdb):
    mol = protein_from_pdb(model_pdb)
    n = mol.GetNumAtoms()
    q = np.linspace(-0.4, 0.4, n)
    mol = protein_from_pdb(model_pdb, q)
    f = build_fields(mol, FieldSet.create(charge_model="file"), name="p")
    assert f.charge_source == "file"
    names = [s.name for s in FieldSet.create().specs]
    assert np.allclose(f.weights[:, names.index("electrostatic")], q)


@needs_model
def test_a_wrong_number_of_charges_is_refused(model_pdb):
    with pytest.raises(ValueError):
        protein_from_pdb(model_pdb, np.zeros(3))


# --------------------------------------------------------------------------- #
# The frame of a residue
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def model_fields(model_pdb):
    mol = protein_from_pdb(model_pdb, np.zeros(protein_from_pdb(model_pdb).GetNumAtoms()))
    return mol, build_fields(mol, FieldSet.create(charge_model="file"), name="wild type")


@needs_model
def test_the_frame_of_a_residue_is_orthonormal(model_fields):
    mol, fields = model_fields
    for pos in (3, 10, 20):
        _origin, rot = backbone_frame(fields.coords, residue_atoms(mol, pos))
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(rot) == pytest.approx(1.0, abs=1e-10)


@needs_model
def test_turning_the_protein_does_not_change_what_the_frame_shows(model_fields):
    mol, fields = model_fields
    import dataclasses
    angle = 0.7
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    turned = dataclasses.replace(fields, coords=np.asarray(fields.coords, float) @ R.T + 4.0)
    a = side_chain_field(fields, mol, 10)
    b = side_chain_field(turned, mol, 10)
    assert a.n_atoms == b.n_atoms
    assert np.allclose(a.coords, b.coords, atol=1e-8)
    assert np.allclose(a.weights, b.weights, atol=1e-10)


@needs_model
def test_the_frame_puts_two_positions_on_each_other(model_fields):
    """The alpha carbon is the origin and the nitrogen lies on the first axis, whichever
    residue it is, so the backbones of two positions agree in the frame to a fraction of an
    angstrom even though the residues are different."""
    mol, fields = model_fields
    places = []
    for pos in (5, 12, 25, 40):
        e = in_residue_frame(fields, mol, pos)
        idx = {name: k for k, src in enumerate(e.member_atom_map)
               for name, i in residue_atoms(mol, pos).items() if int(src) == i}
        places.append({k: e.coords[v] for k, v in idx.items() if k in ("N", "CA", "C")})
    for p in places:
        assert np.allclose(p["CA"], 0.0, atol=1e-8)
        assert abs(p["N"][1]) < 1e-8 and abs(p["N"][2]) < 1e-8      # on the first axis
        assert abs(p["C"][2]) < 1e-8                                 # in the plane
    for p in places[1:]:
        assert np.linalg.norm(p["N"] - places[0]["N"]) < 0.1
        assert np.linalg.norm(p["C"] - places[0]["C"]) < 0.35


@needs_model
def test_a_side_chain_holds_no_main_chain_atom_and_is_never_empty(model_fields):
    mol, fields = model_fields
    for pos in range(2, 45):
        keep = side_chain_atoms(mol, pos)
        names = {mol.GetAtomWithIdx(int(i)).GetPDBResidueInfo().GetName().strip() for i in keep}
        assert not (names & MAIN_CHAIN)
        assert len(keep) >= 1, f"residue {pos} came out with no side chain"
        # glycine presents its alpha hydrogens where another residue has a side chain
        if residue_name(mol, pos) == "GLY":
            assert names == {"HA2", "HA3"}


@needs_model
def test_a_context_is_local_and_leaves_out_the_side_chain_it_surrounds(model_fields):
    mol, fields = model_fields
    pos = 20
    own = set(int(i) for i in side_chain_atoms(mol, pos))
    wide = site_context(fields, mol, pos, 0.05)
    narrow = site_context(fields, mol, pos, 0.5)
    assert not (set(int(i) for i in wide.member_atom_map) & own)
    assert narrow.n_atoms < wide.n_atoms                     # a tighter envelope keeps less
    d = np.linalg.norm(np.asarray(narrow.coords, float), axis=1)
    assert d.max() < np.linalg.norm(np.asarray(wide.coords, float), axis=1).max()


# --------------------------------------------------------------------------- #
# What a difference field is
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.isdir(os.path.join(BUILT, DOMAIN)),
                    reason="the mutant structures are not built")
def test_a_mutant_differs_from_the_wild_type_only_in_the_mutated_residue():
    from cmfqsar.mutants import Structure, protein_of
    built = os.path.join(BUILT, DOMAIN)
    wt = Structure.load(os.path.join(built, "wild_type.npz"))
    wt_mol = protein_of(wt)
    labels = [f[:-4] for f in os.listdir(built) if f.endswith(".npz") and f != "wild_type.npz"]
    if not labels:
        pytest.skip("no mutant structures")
    for label in sorted(labels)[:3]:
        s = Structure.load(os.path.join(built, label + ".npz"))
        mol = protein_of(s)
        m = Mutation.parse(label)
        assert AA1.get(residue_name(mol, m.position)) == m.new
        # every backbone outside the mutated residue is held fixed by the build
        shift = frame_shift(mol, wt_mol, m.position)
        assert shift < 1e-6, f"{label} moved a backbone atom by {shift:.3f} A"
        # the side chains around it are free to make room, and sometimes do
        assert frame_shift(mol, wt_mol, m.position, backbone_only=False) < 8.0
        g = side_chain_geometry(s.pdb_block, m.position)
        if g:
            assert 1.40 < g["ca_cb"] < 1.70, f"{label}: CA-CB is {g['ca_cb']:.2f} A"
            assert 95.0 < g["n_ca_cb"] < 130.0, f"{label}: N-CA-CB is {g['n_ca_cb']:.0f} degrees"


@needs_model
def test_the_shared_part_cancels_in_a_difference_field(model_fields):
    """A difference field of a protein with itself is zero, whatever the kernel is asked."""
    mol, fields = model_fields
    d = site_difference(fields, mol, fields, mol, 20, name="none")
    fs = FieldSet.create(charge_model="file")
    table = AtomTable.from_molecules([d])
    assert np.allclose(self_kernels(table, fs.specs), 0.0, atol=1e-12)


@needs_model
def test_a_difference_field_is_the_two_side_chains(model_fields):
    mol, fields = model_fields
    pos = 20
    d = site_difference(fields, mol, fields, mol, pos)
    a = side_chain_field(fields, mol, pos)
    assert d.n_atoms == 2 * a.n_atoms
    assert np.allclose(d.weights[:a.n_atoms], -d.weights[a.n_atoms:])


@needs_model
def test_two_positions_have_kernels_that_are_not_all_the_same(model_fields):
    """Side chains at different positions are comparable in their frames, so a kernel between
    two of them is neither zero (they do not miss each other) nor one (they are not equal)."""
    mol, fields = model_fields
    fs = FieldSet.create(charge_model="file")
    fields_list = [side_chain_field(fields, mol, p, name=str(p)) for p in (5, 12, 25, 40)]
    table = AtomTable.from_molecules(fields_list)
    K = kernel_matrices(table, table, fs.specs)
    d = self_kernels(table, fs.specs)
    K = normalize_kernels(K, d, d)
    steric = K[[s.name for s in fs.specs].index("steric")]
    off = steric[np.triu_indices(len(fields_list), 1)]
    assert np.all(off > 0.0), "two side chains in one frame always overlap somewhere"
    assert np.all(off < 1.0)
    assert off.min() < 0.95, "different residues should not all look alike"


# --------------------------------------------------------------------------- #
# Building a mutant
# --------------------------------------------------------------------------- #

def test_closest_approach_finds_the_two_atoms_that_are_nearest():
    from cmfqsar.mutants import MIN_SEPARATION, closest_approach
    xyz = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [1.5, 0.4, 0.0], [10.0, 0.0, 0.0]])
    assert closest_approach(xyz) == pytest.approx(0.4)
    assert closest_approach(xyz) < MIN_SEPARATION          # this one would be thrown away
    assert closest_approach(np.zeros((1, 3))) == float("inf")
    # a sound structure: nothing closer than a bond from carbon to hydrogen
    sound = np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0], [0.0, 1.52, 0.0]])
    assert closest_approach(sound) > MIN_SEPARATION


@needs_model
def test_a_structure_keeps_what_it_was_given(tmp_path, model_pdb):
    from cmfqsar.mutants import Structure
    s = Structure("V1Q", model_pdb, np.linspace(-0.3, 0.3, 7), 12.0, -34.0, 2)
    back = Structure.load(s.save(str(tmp_path / "s.npz")))
    assert back.name == s.name and back.pdb_block == s.pdb_block
    assert np.allclose(back.charges, s.charges)
    assert (back.energy_before, back.energy_after, back.collapsed) == (12.0, -34.0, 2)


@needs_model
def test_no_built_structure_has_two_atoms_on_top_of_each_other():
    """The Amber hydrogens on nitrogen and oxygen have no repulsion radius, so a minimisation
    can pull a hydrogen into an acceptor and report an energy of minus a hundred thousand.
    Every stored structure has to be clear of that."""
    from cmfqsar.mutants import MIN_SEPARATION, Structure, closest_approach
    built = os.path.join(BUILT, DOMAIN)
    if not os.path.isdir(built):
        pytest.skip("the mutant structures are not built")
    files = sorted(f for f in os.listdir(built) if f.endswith(".npz") and f != "wild_type.npz")
    if not files:
        pytest.skip("no mutant structures")
    for f in files[:40]:
        s = Structure.load(os.path.join(built, f))
        xyz = np.array([[float(line[30 + 8 * k:38 + 8 * k]) for k in range(3)]
                        for line in s.pdb_block.splitlines() if line.startswith("ATOM")])
        assert closest_approach(xyz) > MIN_SEPARATION, f"{f} has two atoms on top of each other"
        assert s.energy_after > -20000.0, f"{f} has an energy no structure can have"


@needs_openmm
@needs_model
def test_building_a_mutant_refuses_a_wrong_wild_type_residue(tmp_path):
    from cmfqsar.mutants import build_mutant, prepare_wild_type
    wt = str(tmp_path / "wt.pdb")
    prepare_wild_type(os.path.join(EXAMPLES, DOMAIN, f"{DOMAIN}.pdb"), wt)
    mol = protein_from_pdb(open(wt).read())
    here = AA1.get(residue_name(mol, 5))
    wrong = next(a for a in AMINO_ACIDS if a != here)
    with pytest.raises(ValueError):
        build_mutant(wt, Mutation(wrong, 5, "A"))
    with pytest.raises(ValueError):
        build_mutant(wt, Mutation(here, 9999, "A"))


@needs_openmm
@needs_model
def test_a_built_mutant_is_sound_and_keeps_the_rest_in_place(tmp_path):
    from cmfqsar.mutants import build_mutant, prepare_wild_type, protein_of
    wt = prepare_wild_type(os.path.join(EXAMPLES, DOMAIN, f"{DOMAIN}.pdb"),
                           str(tmp_path / "wt.pdb"))
    wt_mol = protein_of(wt)
    pos = 20
    here = AA1.get(residue_name(wt_mol, pos))
    new = "W" if here != "W" else "G"
    s = build_mutant(str(tmp_path / "wt.pdb"), Mutation(here, pos, new), rotamers=3, steps=300)
    mol = protein_of(s)
    assert AA1.get(residue_name(mol, pos)) == new
    assert s.energy_after < s.energy_before + 1.0
    g = side_chain_geometry(s.pdb_block, pos)
    if g:
        assert 1.40 < g["ca_cb"] < 1.70
        assert 95.0 < g["n_ca_cb"] < 130.0
    assert frame_shift(mol, wt_mol, pos) < 1e-6
