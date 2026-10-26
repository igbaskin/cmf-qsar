"""Tests for reading protein structures and selecting a binding site."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.protein import (Protein, binding_site, contact_distance,  # noqa: E402
                             covalent_bonds, read_mmcif, read_pdb, read_structure)

# a tiny structure: two residues of a chain, one water, one zinc, one co-crystallised ligand,
# an alternate location and a second model that must be ignored
PDB = """\
HEADER    TEST STRUCTURE
ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00 20.00           N
ATOM      2  CA  ALA A   1      11.458  10.000  10.000  1.00 20.00           C
ATOM      3  C   ALA A   1      12.000  11.420  10.000  1.00 20.00           C
ATOM      4  O   ALA A   1      11.260  12.400  10.000  1.00 20.00           O
ATOM      5  CB  ALA A   1      12.000   9.200  11.200  1.00 20.00           C
ATOM      6  HA  ALA A   1      11.800   9.500   9.100  1.00 20.00           H
ATOM      7  N   SER B  14      30.000  30.000  30.000  1.00 25.00           N
ATOM      8  CA  SER B  14      31.458  30.000  30.000  1.00 25.00           C
ATOM      9  OG ASER B  14      32.000  31.400  30.000  0.50 25.00           O
ATOM     10  OG BSER B  14      32.000  28.600  30.000  0.50 25.00           O
HETATM   11 ZN    ZN A 101      13.500  10.000  10.000  1.00 15.00          ZN
HETATM   12  O   HOH A 201      10.500  11.500  10.500  1.00 30.00           O
HETATM   13  C1  LIG A 301      14.500  12.000  10.000  1.00 18.00           C
HETATM   14  C2  LIG A 301      15.700  12.400  10.300  1.00 18.00           C
ENDMDL
ATOM     99  N   GLY A   9      99.000  99.000  99.000  1.00 20.00           N
END
"""

CIF = """\
data_TEST
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . ALA A 1 10.000 10.000 10.000 1
ATOM 2 C CA . ALA A 1 11.458 10.000 10.000 1
ATOM 3 C C . ALA A 1 12.000 11.420 10.000 1
HETATM 4 O O . HOH A 201 10.500 11.500 10.500 1
ATOM 5 N N . GLY A 9 99.000 99.000 99.000 2
#
"""


@pytest.fixture()
def pdb_file(tmp_path):
    p = tmp_path / "test.pdb"
    p.write_text(PDB)
    return str(p)


def test_read_pdb_parses_atoms_elements_and_residues(pdb_file):
    p = read_pdb(pdb_file, keep_waters=True)
    assert p.n_atoms == 13                      # one of the two alternate locations is dropped
    assert p.n_residues == 5                    # ALA, SER, ZN, HOH, LIG
    assert p.elements[:5] == ["N", "C", "C", "O", "C"]
    assert p.elements[9] == "Zn"                # two-letter element from the element columns
    assert p.chains[6] == "B" and p.resids[6] == 14
    assert p.hetatm[9] and not p.hetatm[0]
    assert p.residue_label(0) == "ALA1.A"
    assert "GLY" not in p.resnames              # the second model is ignored
    # waters and hydrogens can be dropped while reading
    q = read_pdb(pdb_file, keep_waters=False, keep_hydrogens=False)
    assert q.n_atoms == 11
    assert "HOH" not in q.resnames and "H" not in q.elements


def test_bonds_come_from_distances(pdb_file):
    p = read_pdb(pdb_file, keep_waters=True)
    pairs = {tuple(sorted(b)) for b in p.bonds}
    assert (0, 1) in pairs and (1, 2) in pairs and (2, 3) in pairs      # N-CA, CA-C, C=O
    assert (1, 5) in pairs                                             # CA-HA
    assert (6, 11) not in pairs                                        # far apart, no bond
    assert (11, 12) in pairs                                           # the ligand's own bond
    # two hydrogens are never bonded to each other
    h = np.array([e.upper() == "H" for e in p.elements])
    assert not any(h[a] and h[b] for a, b in p.bonds)
    assert covalent_bonds(np.zeros((1, 3)), ["C"]).shape == (0, 2)


def test_binding_site_selects_whole_residues_and_drops_the_crystal_ligand(pdb_file):
    p = read_pdb(pdb_file, keep_waters=True)
    probe = np.array([[12.5, 11.0, 10.0]])      # between the alanine, the zinc and the ligand
    site = binding_site(p, probe, cutoff=3.0)
    names = set(site.resnames)
    assert "ALA" in names                        # a contacting residue comes in whole
    assert sum(1 for r in site.resnames if r == "ALA") == 6
    assert "ZN" in names                         # ions are kept
    assert "HOH" not in names                    # waters off by default
    assert "SER" not in names                    # far away
    # a ligand that lines the site without occupying it stays, like a cofactor
    assert "LIG" in names
    with_water = binding_site(p, probe, cutoff=3.0, keep_waters=True)
    assert "HOH" in set(with_water.resnames)
    # a ligand that the data set overlaps is the co-crystallised one and is removed
    on_ligand = np.array([[14.6, 12.1, 10.0]])
    assert "LIG" not in set(binding_site(p, on_ligand, cutoff=4.0).resnames)
    assert "ALA" in set(binding_site(p, on_ligand, cutoff=4.0).resnames)
    assert "LIG" in set(binding_site(p, on_ligand, cutoff=4.0, drop_ligand=False).resnames)
    # atoms only, without expanding to residues: a tight cutoff touches part of the alanine
    atoms = binding_site(p, probe, cutoff=1.5, whole_residues=False)
    whole = binding_site(p, probe, cutoff=1.5, whole_residues=True)
    assert atoms.n_atoms < whole.n_atoms
    assert sum(1 for r in whole.resnames if r == "ALA") == 6
    assert sum(1 for r in atoms.resnames if r == "ALA") < 6
    # nothing near: an empty structure, not an error
    assert binding_site(p, np.array([[500.0, 0.0, 0.0]]), cutoff=3.0).n_atoms == 0


def test_contact_distance_is_the_frame_check(pdb_file):
    p = read_pdb(pdb_file, keep_waters=True)
    on_top = np.array([[10.0, 10.0, 10.5]])
    assert contact_distance(p, on_top) == pytest.approx(0.5, abs=1e-6)
    far = np.array([[1000.0, 0.0, 0.0]])
    assert contact_distance(p, far) > 900
    assert contact_distance(p, np.zeros((0, 3))) == float("inf")
    # a molecule sitting exactly on the co-crystallised ligand is not a contact with the protein
    on_ligand = np.array([[14.5, 12.0, 10.0]])
    assert contact_distance(p, on_ligand) > 1.0
    assert contact_distance(p, on_ligand, polymer_only=False) == pytest.approx(0.0, abs=1e-6)


def test_binding_site_accepts_many_points(pdb_file):
    """The site of a whole data set is the union over all its atoms."""
    p = read_pdb(pdb_file, keep_waters=True)
    pts = np.array([[12.5, 11.0, 10.0], [31.0, 30.0, 30.0]])
    site = binding_site(p, pts, cutoff=3.0)
    assert {"ALA", "SER", "ZN"} <= set(site.resnames)
    assert site.n_atoms > binding_site(p, pts[:1], cutoff=3.0).n_atoms


def test_subset_keeps_the_annotation_and_rebuilds_bonds(pdb_file):
    p = read_pdb(pdb_file, keep_waters=True)
    s = p.subset([0, 1, 2, 3])
    assert s.n_atoms == 4 and s.n_residues == 1
    assert s.resnames == ["ALA"] * 4
    assert len(s.bonds) == 3
    assert p.subset([]).n_atoms == 0


def test_mmcif_reader(tmp_path):
    path = tmp_path / "test.cif"
    path.write_text(CIF)
    p = read_structure(str(path), keep_waters=True)
    assert p.n_atoms == 4                        # the second model is ignored
    assert p.elements == ["N", "C", "C", "O"]
    assert p.resnames[3] == "HOH" and p.hetatm[3]
    q = read_structure(str(path), keep_waters=False)
    assert q.n_atoms == 3
    with pytest.raises(ValueError):
        bad = tmp_path / "empty.cif"
        bad.write_text("data_NOTHING\n#\n")
        read_mmcif(str(bad))


def test_write_pdb_round_trip(tmp_path, pdb_file):
    from cmfqsar.protein import write_pdb
    p = read_pdb(pdb_file, keep_waters=True)
    out = tmp_path / "out.pdb"
    write_pdb(p, str(out), remarks=["a pocket written by the tests", "x" * 150])
    text = out.read_text()
    assert text.startswith("REMARK 400 a pocket written by the tests")
    assert text.rstrip().endswith("END")
    q = read_pdb(str(out), keep_waters=True)
    assert q.n_atoms == p.n_atoms and q.n_residues == p.n_residues
    assert np.abs(q.coords - p.coords).max() < 1e-3
    assert q.elements == p.elements and q.atom_names == p.atom_names
    assert q.resnames == p.resnames and q.chains == p.chains
    assert list(q.resids) == list(p.resids)
    assert list(q.hetatm) == list(p.hetatm)


def test_find_pocket_prefers_the_matching_folder(tmp_path):
    from cmfqsar.protein import find_pocket
    ex = tmp_path / "examples"
    (ex / "setA").mkdir(parents=True)
    (ex / "setB").mkdir()
    (ex / "pockets" / "setA").mkdir(parents=True)
    (ex / "pockets" / "setB").mkdir()
    (ex / "setA" / "target.sdf").write_text("")
    (ex / "pockets" / "setA" / "target.pdb").write_text(PDB)
    (ex / "pockets" / "setB" / "target.pdb").write_text(PDB)
    found = find_pocket(str(ex / "setA" / "target.sdf"))
    assert found is not None and os.path.normpath(found) == os.path.normpath(
        str(ex / "pockets" / "setA" / "target.pdb"))
    # a structure lying beside the data set takes precedence over the shared folder
    (ex / "setA" / "target_pocket.pdb").write_text(PDB)
    assert os.path.dirname(find_pocket(str(ex / "setA" / "target.sdf"))) == str(ex / "setA")
    # and a set with no pocket anywhere gives nothing rather than the wrong one
    (ex / "setB" / "other.sdf").write_text("")
    assert find_pocket(str(ex / "setB" / "other.sdf")) is None
    assert find_pocket("") is None


def test_fetch_uses_the_cache_without_network(tmp_path):
    """A file already in the cache directory is returned instead of being downloaded."""
    from cmfqsar.protein import fetch_pdb
    (tmp_path / "1ABC.pdb").write_text(PDB)
    assert fetch_pdb("1abc", dest_dir=str(tmp_path)) == str(tmp_path / "1ABC.pdb")
    with pytest.raises(ValueError):
        fetch_pdb("nonsense-code", dest_dir=str(tmp_path))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_residue_anchors_sit_on_the_alpha_carbon(pdb_file):
    from cmfqsar.protein import residue_anchors
    p = read_pdb(pdb_file, keep_waters=True)
    pos, txt = residue_anchors(p)
    assert len(pos) == len(txt) == p.n_residues
    assert txt[0] == "ALA1.A" and txt[1] == "SER14.B"
    # the alanine label sits exactly on its CA, the zinc on the ion itself
    assert np.allclose(pos[0], p.coords[1])
    zn = txt.index("ZN101.A")
    assert np.allclose(pos[zn], p.coords[9])
    # a residue without an alpha carbon is labelled at its centroid
    lig = txt.index("LIG301.A")
    assert np.allclose(pos[lig], p.coords[[11, 12]].mean(axis=0))


def test_residue_labels_drop_the_chain_when_there_is_only_one(pdb_file):
    from cmfqsar.protein import residue_anchors
    p = read_pdb(pdb_file, keep_waters=True)
    one = p.subset([i for i, c in enumerate(p.chains) if c == "A"])
    _, txt = residue_anchors(one)
    assert "ALA1" in txt and "ALA1.A" not in txt
    # with two chains present the suffix distinguishes them
    _, both = residue_anchors(p)
    assert "ALA1.A" in both and "SER14.B" in both
