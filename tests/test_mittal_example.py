"""The Mittal sets: the geometry check of the builder and the repair of an existing file.

No test needs the Supporting Information.  The molecule is the one record of the 38 files that
failed the check on 14 November 2026, ``DHFR_364``.
"""
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolTransforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

from cmfqsar.orca import geometry_faults  # noqa: E402

import make_mittal_examples as M  # noqa: E402

SMILES = "COc1ccc(c(c1)NCc2cnc3c(n2)c(nc(n3)N)N)OC"
SEED = 1000 + 363
#: a second molecule with the same ring system, as the reference of a small file
REFERENCE = "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(cc1)C(=O)O"


def _amine(mol):
    """(carbon, nitrogen, carbon) of the secondary amine between the two ring systems."""
    return mol.GetSubstructMatch(Chem.MolFromSmarts("c[NX3;H1][CH2]"))


def test_the_first_embedding_of_the_record_has_a_linear_nitrogen(monkeypatch):
    monkeypatch.setattr(M, "EMBED_TRIES", 1)
    mol = M.embed(SMILES, SEED)
    assert geometry_faults(mol) == ["bond N9-H31 1.35 A (1.32 of the covalent radii)"]
    assert mol.GetProp("geometry_note").startswith("no embedding of 1 passes")
    assert rdMolTransforms.GetAngleDeg(mol.GetConformer(), *_amine(mol)) > 179.0


def test_the_builder_embeds_it_again_and_says_so():
    mol = M.embed(SMILES, SEED)
    assert geometry_faults(mol) == []
    assert mol.GetProp("geometry_note").startswith(f"embedding seed + {M.SEED_STEP}")
    assert 105.0 < rdMolTransforms.GetAngleDeg(mol.GetConformer(), *_amine(mol)) < 135.0
    # a molecule that passes at once carries no note
    assert not M.embed("CCO", 1000).HasProp("geometry_note")


def test_a_file_is_repaired_without_touching_its_other_records(tmp_path, monkeypatch):
    # a file of two records as the builder wrote it before the check existed
    monkeypatch.setattr(M, "EMBED_TRIES", 1)
    ref = M.embed(REFERENCE, 1000)
    bad = M.embed(SMILES, SEED)
    monkeypatch.undo()
    assert geometry_faults(ref) == [] and geometry_faults(bad) != []
    path = str(tmp_path / "DHFR.sdf")
    with Chem.SDWriter(path) as w:
        for mol, name, smi in ((ref, "DHFR_001", REFERENCE), (bad, "DHFR_364", SMILES)):
            if mol.HasProp("geometry_note"):
                mol.ClearProp("geometry_note")
            mol.SetProp("_Name", name)
            mol.SetProp("activity", "4.640")
            mol.SetProp("SMILES", smi)
            mol.SetProp("align_mcs_atoms", "0")
            mol.SetProp("align_rmsd", "")
            mol.SetProp("reference", "DHFR_001")
            w.write(mol)
    with open(path, newline="") as fh:
        before = fh.read()
    assert M.repair_set(path, mcs_timeout=20) == ["DHFR_364"]
    with open(path, newline="") as fh:
        after = fh.read()
    first = before[:before.index("$$$$") + 4]
    assert after.startswith(first)                              # the other record, to the character
    mols = list(Chem.SDMolSupplier(path, removeHs=False))
    assert [m.GetProp("_Name") for m in mols] == ["DHFR_001", "DHFR_364"]
    assert geometry_faults(mols[1]) == []
    assert mols[1].GetProp("activity") == "4.640" and mols[1].GetProp("SMILES") == SMILES
    assert int(mols[1].GetProp("align_mcs_atoms")) >= 10        # superimposed on the reference
    assert float(mols[1].GetProp("align_rmsd")) < 2.0
    assert M.repair_set(path, mcs_timeout=20) == []             # nothing is left to repair
    with open(path, newline="") as fh:
        assert fh.read() == after


def test_the_record_of_the_repository_passes_and_names_its_repair():
    path = os.path.join(ROOT, "examples", "mittal", "DHFR.sdf")
    mol = [m for m in Chem.SDMolSupplier(path, removeHs=False)
           if m.GetProp("_Name") == "DHFR_364"][0]
    assert geometry_faults(mol) == []
    assert "embedding seed" in mol.GetProp("geometry_note")
    assert np.isfinite(float(mol.GetProp("align_rmsd")))
