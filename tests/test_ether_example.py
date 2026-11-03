"""The reader that turns a relaxed ion-ether complex into a molecule (plan item EC7)."""
import io
import os
import pickle
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import make_ether_examples as M  # noqa: E402


def _complex(smiles="COCCOC", seed=3, ion_at=(0.0, 0.0, 0.0)):
    """An ether with its coordinates and a lithium put at `ion_at`, as the distribution has it."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(mol, p)
    AllChem.MMFFOptimizeMolecule(mol)
    Chem.SanitizeMol(mol)
    coords = mol.GetConformer().GetPositions()
    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], int)
    # the ion first, as the geometry file has it
    numbers = np.concatenate([[3], numbers])
    coords = np.vstack([np.asarray(ion_at, float), coords])
    return smiles, numbers, coords, mol


def test_the_ion_is_removed_and_its_position_kept():
    smiles, numbers, coords, mol = _complex(ion_at=(1.5, -2.0, 0.5))
    out, ion = M.ether_in_complex_geometry(smiles, numbers, coords)
    assert out.GetNumAtoms() == mol.GetNumAtoms(), "the ether keeps every atom but the ion"
    assert not any(a.GetSymbol() == "Li" for a in out.GetAtoms())
    assert np.allclose(ion, [1.5, -2.0, 0.5])


def test_the_atom_order_is_the_order_of_the_smiles():
    """So that a property written per atom belongs to the atom the SMILES names."""
    smiles, numbers, coords, mol = _complex()
    out, _ion = M.ether_in_complex_geometry(smiles, numbers, coords)
    assert ([a.GetSymbol() for a in out.GetAtoms()]
            == [a.GetSymbol() for a in mol.GetAtoms()])
    # the coordinates are the ones given, up to the matching of identical atoms
    got = out.GetConformer().GetPositions()
    want = coords[1:]
    assert sorted(np.round(got.sum(axis=1), 6).tolist()) == \
        sorted(np.round(want.sum(axis=1), 6).tolist())


def test_a_geometry_that_does_not_match_the_smiles_is_refused():
    smiles, numbers, coords, _mol = _complex()
    with pytest.raises(ValueError, match="atoms"):
        M.ether_in_complex_geometry("COCCOCCOC", numbers, coords)


def test_a_complex_without_exactly_one_ion_is_refused():
    smiles, numbers, coords, _mol = _complex()
    numbers = numbers.copy()
    numbers[0] = 6                                    # the lithium replaced by a carbon
    with pytest.raises(ValueError, match="lithium"):
        M.ether_in_complex_geometry(smiles, numbers, coords)


def test_the_geometry_reader_admits_numpy_and_refuses_everything_else():
    """The distributed geometries are a pickle, and a pickle runs what it names."""
    good = pickle.dumps({"a.[Li+]": {"ebind": -1.0, "xyz": np.zeros((2, 4))}})
    out = M._ArraysOnly(io.BytesIO(good)).load()
    assert out["a.[Li+]"]["xyz"].shape == (2, 4)
    bad = pickle.dumps({"x": os.getcwd})
    with pytest.raises(pickle.UnpicklingError, match="refuses"):
        M._ArraysOnly(io.BytesIO(bad)).load()
