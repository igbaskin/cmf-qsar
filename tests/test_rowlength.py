"""Tests for the row length the invariant kernel divides out (plan item M15).

The kernel rescales the channels and then divides every feature row by its own length, so the
length never reaches the model. `row_lengths` returns that number and `length_kernel` makes a
kernel of it, so that what it is worth can be measured. These tests hold the two functions to
what the measurement assumes about them.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))

from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               channel_scale, feature_matrix, length_kernel, row_lengths)

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
    mf = [build_fields(_mol(s, seed=i + 1, name=f"m{i}"), fs, f"m{i}")
          for i, s in enumerate(SMILES)]
    spec = AutocorrelationSpec()
    X = feature_matrix(mf, spec)
    scale = channel_scale(X, spec.n_bins)
    return fs, mf, spec, X, scale


def test_the_length_is_exactly_what_the_cosine_kernel_divides_out(world):
    """The cosine kernel is the kernel without the division, divided by the two row lengths."""
    _, _, spec, X, scale = world
    raw = autocorrelation_kernel(X, scale=scale, n_bins=spec.n_bins, normalize=False)
    cosine = autocorrelation_kernel(X, scale=scale, n_bins=spec.n_bins, normalize=True)
    ell = row_lengths(X, scale=scale)
    assert np.all(ell > 0)
    assert np.allclose(raw / np.outer(ell, ell), cosine, atol=1e-12)
    # and the self-kernel without the division is the squared length
    assert np.allclose(np.diag(raw), ell ** 2, rtol=1e-12)


def test_the_scale_comes_from_the_rows_when_it_is_not_given(world):
    """Without a scale the length is the length of the row the kernel saw, not of the raw row."""
    _, _, spec, X, scale = world
    assert np.allclose(row_lengths(X, n_bins=spec.n_bins), row_lengths(X, scale=scale))
    # the raw row is longer or shorter by whatever the channel scale is, so the two differ
    assert not np.allclose(row_lengths(X, scale=np.ones(X.shape[1])), row_lengths(X, scale=scale))


def test_a_rigid_motion_does_not_change_the_length(world):
    """The descriptor is invariant to rotation and translation, so its length is too."""
    _, mf, spec, X, scale = world
    rng = np.random.default_rng(0)
    before = row_lengths(X, scale=scale)
    moved = []
    for mol in mf:
        R, t = _rotation(rng), rng.normal(scale=20.0, size=3)
        moved.append(type(mol)(name=mol.name, coords=mol.coords @ R.T + t, weights=mol.weights,
                               radii=mol.radii, symbols=mol.symbols, atom_types=mol.atom_types,
                               bonds=mol.bonds, charges=mol.charges))
    after = row_lengths(feature_matrix(moved, spec), scale=scale)
    assert np.allclose(after, before, rtol=1e-9, atol=1e-9)


def test_a_larger_field_changes_the_length_and_not_the_cosine_kernel(world):
    """This is the mechanism item M15 is about.

    Every channel is a product of two field weights, so multiplying the fields of one molecule by
    two multiplies its row, and its length, by four. The cosine kernel divides that out: its row
    for that molecule does not move at all. So the kernel cannot tell a molecule from the same
    molecule carrying four times the field, and whether that matters is what the benchmark
    measures.
    """
    _, mf, spec, X, scale = world
    bigger = list(mf)
    mol = mf[0]
    bigger[0] = type(mol)(name=mol.name, coords=mol.coords, weights=2.0 * mol.weights,
                          radii=mol.radii, symbols=mol.symbols, atom_types=mol.atom_types,
                          bonds=mol.bonds, charges=mol.charges)
    Xb = feature_matrix(bigger, spec)
    ell = row_lengths(X, scale=scale)
    ellb = row_lengths(Xb, scale=scale)
    assert ellb[0] == pytest.approx(4.0 * ell[0], rel=1e-9)
    assert np.allclose(ellb[1:], ell[1:], rtol=1e-12)
    before = autocorrelation_kernel(X, scale=scale, n_bins=spec.n_bins)
    after = autocorrelation_kernel(Xb, scale=scale, n_bins=spec.n_bins)
    assert np.allclose(after, before, atol=1e-12)


def test_the_length_kernel_is_a_kernel_of_order_one(world):
    """It is an outer product, so it is positive semi-definite, and its entries are near one."""
    _, _, spec, X, scale = world
    ell = row_lengths(X, scale=scale)
    K = length_kernel(ell)
    assert K.shape == (len(ell), len(ell))
    assert np.allclose(K, K.T, atol=1e-12)
    w = np.linalg.eigvalsh(K)
    assert w.min() > -1e-10                      # one positive eigenvalue, the rest zero
    assert np.sum(w > 1e-8 * max(w.max(), 1.0)) == 1
    assert 0.1 < float(np.mean(np.diag(K))) < 10.0
    # the reference only rescales it, which is why the mixing weight is the only parameter
    assert np.allclose(length_kernel(ell, reference=1.0),
                       K * float(np.mean(ell * ell)), rtol=1e-9)


def test_the_test_block_uses_the_training_reference(world):
    """A test row's entry is its own length times a training length, in training units."""
    _, _, spec, X, scale = world
    ell = row_lengths(X, scale=scale)
    train, test = ell[:4], ell[4:]
    ref = float(np.sqrt(np.mean(train * train)))
    K = length_kernel(train, test, reference=ref)
    assert K.shape == (len(test), len(train))
    assert K[0, 0] == pytest.approx(test[0] * train[0] / ref ** 2, rel=1e-12)


def test_the_site_kernel_can_keep_the_length():
    """`normalize=False` is what the measurement needs, and it changes no default."""
    from cmfqsar.sites import site_kernel
    rng = np.random.default_rng(3)
    X = rng.normal(size=(7, 11)) + 3.0
    kept = site_kernel(X, normalize=False)
    cosine = site_kernel(X)
    assert np.allclose(site_kernel(X, normalize=True), cosine, atol=1e-12)
    assert np.allclose(np.diag(cosine), 1.0, atol=1e-12)
    assert np.std(np.diag(kept)) > 0             # the length is in the diagonal now


def test_the_four_arms_of_the_benchmark_are_what_they_say(world):
    """The benchmark's first arm is the kernel the program builds today, and the last is the
    length alone; the mixture is between them."""
    import run_rowlength as RL
    _, _, spec, X, scale = world
    fams, lengths = RL.arm_families({"one": (X, spec.n_bins, None)}, weights=(0.25,))
    assert set(fams) == set(RL.ARMS)
    assert np.allclose(fams[RL.ARM_OUT]["one"],
                       autocorrelation_kernel(X, scale=scale, n_bins=spec.n_bins), atol=1e-12)
    assert np.allclose(lengths["one"], row_lengths(X, scale=scale))
    kept = fams[RL.ARM_KEPT]["one"]
    assert float(np.mean(np.diag(kept))) == pytest.approx(1.0, rel=1e-9)
    mix = fams[RL.ARM_PLUS]["one, weight 0.25"]
    assert np.allclose(mix, 0.75 * fams[RL.ARM_OUT]["one"] + 0.25 * fams[RL.ARM_ONLY]["one"],
                       atol=1e-12)
