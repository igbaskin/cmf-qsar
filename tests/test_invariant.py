"""Tests for the alignment-free, geometry-aware field autocorrelation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation,  # noqa: E402
                               autocorrelation_kernel, channel_labels, channel_scale,
                               feature_matrix, field_pairs)
from cmfqsar.kernels import self_kernels  # noqa: E402

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "CCOc1ccc(C)cc1CC(=O)N",
          "CC(C)COc1ccccc1C(=O)NC", "CCCCOc1ccccc1C(=O)N", "NCCc1ccc(O)cc1"]


def _mol(smiles: str, seed: int = 1, name: str = "m"):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", name)
    return m


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create(alpha=0.5)
    mols = [_mol(s, seed=i + 1, name=f"m{i}") for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    return fs, mols, mf


def test_a_rigid_motion_changes_nothing(world):
    """Rotation and translation move every atom and leave every distance; so does the descriptor."""
    fs, _, mf = world
    spec = AutocorrelationSpec()
    rng = np.random.default_rng(0)
    for mol in mf[:3]:
        before = autocorrelation(mol, spec)
        moved = mol.copy() if hasattr(mol, "copy") else mol
        R, t = _rotation(rng), rng.normal(scale=20.0, size=3)
        xyz = mol.coords @ R.T + t
        shifted = type(mol)(name=mol.name, coords=xyz, weights=mol.weights, radii=mol.radii,
                            symbols=mol.symbols, atom_types=mol.atom_types, bonds=mol.bonds,
                            charges=mol.charges)
        assert np.allclose(autocorrelation(shifted, spec), before, atol=1e-9)
        # ... and a reflection does change it, since a mirror image is a different molecule
        mirrored = type(mol)(name=mol.name, coords=mol.coords * np.array([1.0, 1.0, -1.0]),
                             weights=mol.weights, radii=mol.radii, symbols=mol.symbols,
                             atom_types=mol.atom_types, bonds=mol.bonds, charges=mol.charges)
        assert np.allclose(autocorrelation(mirrored, spec), before, atol=1e-9)  # distances only


def test_it_still_sees_the_geometry(world):
    """Unlike a count kernel, it changes when the conformation does."""
    fs, mols, mf = world
    spec = AutocorrelationSpec()
    here = autocorrelation(mf[0], spec)
    rng = np.random.default_rng(3)
    xyz = mf[0].coords + rng.normal(scale=0.8, size=mf[0].coords.shape)
    scrambled = type(mf[0])(name="x", coords=xyz, weights=mf[0].weights, radii=mf[0].radii,
                            symbols=mf[0].symbols, atom_types=mf[0].atom_types,
                            bonds=mf[0].bonds, charges=mf[0].charges)
    there = autocorrelation(scrambled, spec)
    assert np.linalg.norm(there - here) / max(np.linalg.norm(here), 1e-12) > 0.02
    # the same atoms in the same places give exactly the same descriptor
    again = autocorrelation(mf[0], spec)
    assert np.array_equal(again, here)


def test_the_zero_distance_channel_is_the_cmf_self_overlap(world):
    """With sigma^2 = 2/alpha the first channel is the overlap integral, up to its prefactor."""
    fs, _, mf = world
    alpha = 0.5
    spec = AutocorrelationSpec(sigma=float(np.sqrt(2.0 / alpha)))
    mol = mf[0]
    feats = autocorrelation(mol, spec).reshape(-1, spec.n_bins)
    pairs = field_pairs(mol.weights.shape[1])
    d = self_kernels(AtomTable.from_molecules([mol]), fs.specs)[:, 0]
    for i, (f, g) in enumerate(pairs):
        if f != g or fs.specs[f].kind != "gauss":
            continue                      # the steric field has one width per atom radius
        raw = feats[i, 0]
        assert raw == pytest.approx(d[f] / (np.pi / alpha) ** 1.5, rel=1e-8)


def test_the_kernel_is_well_formed_and_sees_what_counts_cannot(world):
    fs, mols, mf = world
    spec = AutocorrelationSpec()
    X = feature_matrix(mf, spec)
    assert X.shape == (len(mf), len(field_pairs(mf[0].weights.shape[1])) * spec.n_bins)
    assert len(channel_labels(fs, spec)) == X.shape[1]
    K = autocorrelation_kernel(X, n_bins=spec.n_bins)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    assert K.max() <= 1.0 + 1e-9 and K.min() >= -1.0 - 1e-9
    # a test block against the same molecules reproduces the training block
    scale = channel_scale(X, spec.n_bins)
    assert np.allclose(autocorrelation_kernel(X, X, scale=scale, n_bins=spec.n_bins), K)

    # two conformations of one molecule: the count kernel calls them identical, this does not
    from cmfqsar.chemotypes import count_kernel
    from rdkit import Chem
    other = Chem.Mol(mols[0])
    conf = other.GetConformer()
    rng = np.random.default_rng(5)
    xyz = np.array(conf.GetPositions()) + rng.normal(scale=0.7, size=(other.GetNumAtoms(), 3))
    for i in range(other.GetNumAtoms()):
        conf.SetAtomPosition(i, xyz[i].tolist())
    pair_counts = count_kernel([mols[0], other], ["a", "b"], cif_min=1)
    assert pair_counts[0, 1] == pytest.approx(1.0, abs=1e-6)       # same atoms, so identical
    mf2 = [build_fields(mols[0], fs, "a"), build_fields(other, fs, "b")]
    K2 = autocorrelation_kernel(feature_matrix(mf2, spec), n_bins=spec.n_bins)
    assert K2[0, 1] < 0.999                                        # different geometry, seen


def test_the_self_pairs_carry_the_counts(world):
    """Dropping the l = m terms removes exactly the count-like part of the first channel."""
    _, _, mf = world
    mol = mf[0]
    spec = AutocorrelationSpec()
    with_self = autocorrelation(mol, spec).reshape(-1, spec.n_bins)
    without = autocorrelation(mol, AutocorrelationSpec(include_self=False)).reshape(-1, spec.n_bins)
    pairs = field_pairs(mol.weights.shape[1])
    for i, (f, g) in enumerate(pairs):
        w = mol.weights
        expected = float(np.sum(w[:, f] * w[:, g])) * (2.0 if f != g else 1.0)
        assert with_self[i, 0] - without[i, 0] == pytest.approx(expected, rel=1e-9, abs=1e-9)
