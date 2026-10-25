"""Handedness read at a centre: the properties C5's block had, and the one it lacked."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.stereo import (StereoSpec, channel_labels, stereo_centres,  # noqa: E402
                            stereo_channels, stereo_matrix)

FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: four amino acids and four alpha-hydroxy acids, each a centre with four different substituents
SMILES = ["N[C@@H](C)C(=O)O", "N[C@@H](CC)C(=O)O", "N[C@@H](CO)C(=O)O", "N[C@@H](CS)C(=O)O",
          "O[C@@H](C)C(=O)N", "O[C@@H](CC)C(=O)N", "O[C@@H](CCl)C(=O)N", "O[C@@H](CF)C(=O)N"]


def _mol(smi, seed=5):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    assert AllChem.EmbedMolecule(m, p) == 0
    Chem.SanitizeMol(m)
    return m


def _mirror(mol):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, p in enumerate(np.array(conf.GetPositions()) * np.array([1.0, 1.0, -1.0])):
        conf.SetAtomPosition(i, [float(v) for v in p])
    return m


def _turned(mol, seed=0):
    rng = np.random.default_rng(seed)
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0                       # a proper rotation, not a reflection
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    xyz = np.array(conf.GetPositions()) @ R.T + rng.normal(scale=5.0, size=3)
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, [float(v) for v in p])
    return m


def _cip(mol):
    probe = Chem.Mol(mol)
    Chem.AssignStereochemistryFrom3D(probe)
    codes = [a.GetProp("_CIPCode") for a in probe.GetAtoms() if a.HasProp("_CIPCode")]
    return codes[0] if len(codes) == 1 else None


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create(physchem=FIELDS, alpha=0.5, charge_model="gasteiger")
    mols = [_mol(s, 5 + i) for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    return fs, mols, mf


def _unit(v):
    return np.asarray(v, float) / max(float(np.linalg.norm(v)), 1e-12)


# --------------------------------------------------------------- the three properties

def test_the_channels_negate_under_reflection(world):
    fs, mols, mf = world
    for m, f in zip(mols, mf):
        c = stereo_channels(f, n_fields=len(fs.specs))
        mirrored = build_fields(_mirror(m), fs, f.name)
        assert np.linalg.norm(c) > 1e-6, f.name
        assert np.allclose(c, -stereo_channels(mirrored, n_fields=len(fs.specs)), atol=1e-9)


def test_the_channels_survive_a_rigid_motion(world):
    fs, mols, mf = world
    for m, f in zip(mols, mf):
        moved = build_fields(_turned(m), fs, f.name)
        assert np.allclose(stereo_channels(f, n_fields=len(fs.specs)),
                           stereo_channels(moved, n_fields=len(fs.specs)), atol=1e-8)


def test_two_molecules_of_the_same_handedness_point_the_same_way(world):
    """The property C5's whole-molecule block does not have, and the reason for this module.

    A kernel method transfers what it learnt along a direction in feature space, so the channels
    of two *different* molecules of the same configuration have to agree in sign. C5's block
    separated same from opposite handedness by 0.03 against a spread of 0.55; this asks for
    something that is not that.
    """
    fs, mols, mf = world
    V = np.array([_unit(stereo_channels(f, n_fields=len(fs.specs))) for f in mf])
    hand = np.array([1.0 if _cip(m) == "R" else -1.0 for m in mols])
    same = [float(V[i] @ V[j]) for i in range(len(V)) for j in range(i + 1, len(V))
            if hand[i] == hand[j]]
    opposite = [float(V[i] @ V[j]) for i in range(len(V)) for j in range(i + 1, len(V))
                if hand[i] != hand[j]]
    assert same and opposite
    assert float(np.mean(same)) - float(np.mean(opposite)) > 0.5


# ------------------------------------------------------------------------- the reading

def test_a_centre_needs_enough_neighbours(world):
    _fs, _mols, mf = world
    centres = stereo_centres(mf[0])
    assert centres and all(len(n) >= 3 for _i, n in centres)
    assert not stereo_centres(mf[0], StereoSpec(min_neighbours=9))


def test_a_tie_contributes_nothing_rather_than_a_guess():
    """Two identical substituents have no order, and an invented one would be an invented sign."""
    fs = FieldSet.create(physchem=FIELDS, alpha=0.5, charge_model="gasteiger")
    flat = build_fields(_mol("ClC(Cl)(Cl)Cl"), fs, "tetrachloromethane")
    strict = stereo_channels(flat, StereoSpec(min_separation=0.5), n_fields=len(fs.specs))
    assert np.allclose(strict, 0.0)


def test_the_guard_decides_how_many_centres_survive(world):
    fs, _mols, mf = world
    loose = stereo_matrix(mf, StereoSpec(min_separation=0.0), n_fields=len(fs.specs))[1]
    tight = stereo_matrix(mf, StereoSpec(min_separation=2.0), n_fields=len(fs.specs))[1]
    assert loose["with_centre"] >= tight["with_centre"]
    assert tight["with_centre"] < loose["with_centre"]


def test_a_molecule_with_no_usable_centre_says_zero_not_nonsense():
    fs = FieldSet.create(physchem=FIELDS, alpha=0.5, charge_model="gasteiger")
    water = build_fields(_mol("O"), fs, "water")
    assert np.allclose(stereo_channels(water, n_fields=len(fs.specs)), 0.0)


def test_the_matrix_is_one_row_per_molecule_with_a_label_each(world):
    fs, _mols, mf = world
    X, info = stereo_matrix(mf, n_fields=len(fs.specs))
    assert X.shape == (len(mf), len(fs.specs) ** 2)
    assert len(channel_labels(fs)) == X.shape[1]
    assert info["molecules"] == len(mf) and info["with_centre"] > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
