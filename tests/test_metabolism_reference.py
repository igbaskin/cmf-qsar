"""The reference of the site-of-metabolism benchmark that needs no field and no geometry:
a few properties of the atom itself (`atom_properties` of benchmarks/run_metabolism.py)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

RMB = pytest.importorskip("run_metabolism")

sys.path.insert(0, ROOT)
from cmfqsar.atomtargets import AtomTargets  # noqa: E402


def _set(smiles):
    """Molecules with every heavy atom measured, and their targets."""
    mols = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        for a in m.GetAtoms():
            a.SetDoubleProp("som", 0.0)
        mols.append(m)
    return mols, AtomTargets.from_molecules(mols, "som")


def test_the_columns_are_the_properties_of_the_atom_itself():
    mols, targets = _set(["c1ccccc1C(=O)N", "CCS"])
    P = RMB.atom_properties(mols, targets)
    # eight scaled properties and one column per hydrogen count, 0 to 3
    assert P.shape == (len(targets), len(RMB.PROPERTY_NAMES) + 4)
    one_hot = P[:, len(RMB.PROPERTY_NAMES):]
    assert np.all(one_hot.sum(axis=1) == 1)
    scaled = P[:, :len(RMB.PROPERTY_NAMES)]
    assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-8)


def test_an_aromatic_carbon_and_a_methyl_differ_in_their_columns():
    mols, targets = _set(["c1ccccc1C"])
    P = RMB.atom_properties(mols, targets)
    mol = mols[0]
    aromatic = next(i for i in range(mol.GetNumAtoms()) if mol.GetAtomWithIdx(i).GetIsAromatic())
    methyl = next(i for i in range(mol.GetNumAtoms())
                  if mol.GetAtomWithIdx(i).GetTotalNumHs() == 3)
    names = list(RMB.PROPERTY_NAMES)
    for key in ("aromatic", "in a ring", "hydrogens on the atom"):
        assert P[aromatic, names.index(key)] != P[methyl, names.index(key)]
    # the methyl is at the end of the molecule, so its longest bond path is the longer one
    assert P[methyl, names.index("relative span")] > P[aromatic, names.index("relative span")]


def test_the_span_is_one_at_the_end_of_a_chain():
    mols, targets = _set(["CCCCC"])
    P = RMB.atom_properties(mols, targets)
    raw_span = P[:, list(RMB.PROPERTY_NAMES).index("relative span")]
    # the two ends share the largest value and the middle atom the smallest
    assert np.argmax(raw_span) in (0, 4) and np.argmin(raw_span) == 2
