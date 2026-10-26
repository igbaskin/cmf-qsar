"""The two-sample test: what it says when there is nothing to find, and when there is."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.mmd import assemble, mmd2, mmd_test, witness  # noqa: E402


def _kernel(A, B, gamma=0.5):
    Z = np.vstack([A, B])
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    K = np.exp(-gamma * d2)
    mask = np.zeros(len(Z), bool)
    mask[:len(A)] = True
    return K, mask


def test_a_set_against_itself_has_no_discrepancy():
    """Exactly zero for the biased estimator; the unbiased one is slightly negative here.

    Handed the same points twice, the cross block carries the self-similarities that the two
    within-set means have had removed, so the unbiased estimator comes out below zero. That is
    the bias correction working as intended, not a difference between the sets.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    K, mask = _kernel(X, X.copy())
    assert abs(mmd2(K, mask, unbiased=False)) < 1e-9
    assert -0.1 < mmd2(K, mask) < 0.0


def test_the_unbiased_estimator_is_centred_on_zero_under_the_null():
    """It takes either sign, which is how 'no difference' looks once the bias is removed."""
    rng = np.random.default_rng(1)
    vals = []
    for t in range(40):
        X = rng.normal(size=(30, 3))
        Y = rng.normal(size=(30, 3))
        K, mask = _kernel(X, Y)
        vals.append(mmd2(K, mask))
    vals = np.array(vals)
    assert abs(float(np.mean(vals))) < 3.0 * float(np.std(vals)) / np.sqrt(len(vals))
    assert (vals < 0).any() and (vals > 0).any()


def test_the_biased_estimator_is_never_negative():
    rng = np.random.default_rng(2)
    K, mask = _kernel(rng.normal(size=(25, 3)), rng.normal(size=(25, 3)))
    assert mmd2(K, mask, unbiased=False) >= -1e-12


def test_two_samples_of_one_population_are_not_distinguished():
    rng = np.random.default_rng(3)
    K, mask = _kernel(rng.normal(size=(60, 4)), rng.normal(size=(60, 4)))
    r = mmd_test(K, mask, permutations=499, seed=0)
    assert r.p_value > 0.05
    assert "not distinguishable" in r.describe()


def test_a_shifted_population_is_found():
    rng = np.random.default_rng(4)
    K, mask = _kernel(rng.normal(size=(60, 4)), rng.normal(size=(60, 4)) + 1.5)
    r = mmd_test(K, mask, permutations=499, seed=0)
    assert r.p_value <= 0.01 and r.z > 3.0
    assert r.mmd > 0.0 and "different regions" in r.describe()


def test_a_difference_in_spread_alone_is_found():
    """A characteristic kernel sees more than a difference of means; a t-test would not."""
    rng = np.random.default_rng(5)
    K, mask = _kernel(rng.normal(size=(80, 3)), rng.normal(size=(80, 3)) * 2.5)
    assert mmd_test(K, mask, permutations=499, seed=0).p_value <= 0.01


def test_the_p_value_is_never_zero():
    """A finite number of shuffles cannot support a claim of impossibility."""
    rng = np.random.default_rng(6)
    K, mask = _kernel(rng.normal(size=(40, 3)), rng.normal(size=(40, 3)) + 8.0)
    r = mmd_test(K, mask, permutations=99, seed=0)
    assert r.p_value == pytest.approx(1.0 / 100.0)


def test_the_witness_points_at_the_set_a_molecule_belongs_to():
    rng = np.random.default_rng(7)
    A = rng.normal(size=(50, 3))
    B = rng.normal(size=(50, 3)) + 4.0
    K, mask = _kernel(A, B)
    w = witness(K, mask)
    assert np.mean(w[:50] > 0) > 0.9 and np.mean(w[50:] < 0) > 0.9


def test_too_few_molecules_give_nan_rather_than_a_number():
    K = np.eye(3)
    assert np.isnan(mmd2(K, np.array([True, False, False])))


def test_the_blocks_assemble_into_the_same_kernel():
    rng = np.random.default_rng(8)
    A, B = rng.normal(size=(12, 3)), rng.normal(size=(9, 3))
    K, mask = _kernel(A, B)
    m = len(A)
    K2, mask2 = assemble(K[:m, :m], K[m:, :m], K[m:, m:])
    assert np.allclose(K, K2) and np.array_equal(mask, mask2)
    with pytest.raises(ValueError):
        assemble(K[:m, :m], K[:m, m:], K[m:, m:])


def test_it_works_on_real_fields():
    """End to end on molecules: two different scaffolds are not one population."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import FieldSet
    from cmfqsar.mmd import kernel_over

    def build(smis, tag):
        mols, names = [], []
        for i, smi in enumerate(smis):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            AllChem.EmbedMolecule(m, randomSeed=11 + i)
            Chem.SanitizeMol(m)
            mols.append(m)
            names.append(f"{tag}{i}")
        return mols, names

    subs = ["C", "CC", "CCC", "O", "N", "Cl", "F", "Br", "OC", "NC", "C(=O)O", "C#N", "CCO",
            "CCN", "S"]
    a = build([f"c1ccccc1{s}" for s in subs], "aromatic")
    b = build([f"CCCCCCCC{s}" for s in subs], "aliphatic")
    fs = FieldSet.create(alpha=0.6)
    K, mask = kernel_over(a[0], a[1], b[0], b[1], fs)
    r = mmd_test(K, mask, permutations=499, seed=0)
    assert r.n_x == 15 and r.n_y == 15
    assert r.p_value < 0.05 and r.mmd > 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
