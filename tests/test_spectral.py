"""Clustering the kernel: what it finds, how many, and how it is compared with the scaffolds."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.spectral import (adjusted_rand, eigengap,  # noqa: E402
                              spectral_clusters)


def _blobs(centres, n=30, scale=0.3, seed=0, gamma=0.5):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(loc=c, scale=scale, size=(n, len(c))) for c in centres])
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2), np.repeat(np.arange(len(centres)), n)


def test_separated_groups_are_recovered_exactly():
    K, truth = _blobs([(0, 0, 0), (6, 0, 0), (0, 6, 0)])
    fit = spectral_clusters(K)
    assert fit.k == 3 and fit.chosen_by_gap
    assert adjusted_rand(fit.labels, truth) == 1.0
    assert sorted(fit.sizes.values()) == [30, 30, 30]


def test_the_number_of_clusters_can_be_given_instead():
    K, _truth = _blobs([(0, 0, 0), (6, 0, 0), (0, 6, 0)])
    fit = spectral_clusters(K, k=4)
    assert fit.k == 4 and not fit.chosen_by_gap and np.isnan(fit.gap)
    assert "as asked" in fit.describe()


def test_one_population_is_not_forced_into_three_pieces_with_a_large_gap():
    """A weak gap is reported as weak; the heuristic is not allowed to hide its uncertainty."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 3))
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    one = spectral_clusters(np.exp(-0.5 * d2))
    K, _ = _blobs([(0, 0, 0), (6, 0, 0), (0, 6, 0)])
    many = spectral_clusters(K)
    assert many.gap > one.gap


def test_negative_similarities_do_not_become_edges():
    """An unnormalised field kernel can be negative; a negative affinity is not a similarity."""
    K, truth = _blobs([(0, 0), (7, 0)], n=20)
    K = K - 0.2                        # push the far pairs below zero
    fit = spectral_clusters(K, k=2)
    assert adjusted_rand(fit.labels, truth) == 1.0


def test_the_eigengap_counts_from_the_smallest_eigenvalues():
    w = [0.0, 0.0, 0.0, 0.9, 0.95, 1.0]
    k, gap = eigengap(w)
    assert k == 3 and gap == pytest.approx(0.9)


def test_adjusted_rand_is_one_for_a_relabelling_and_near_zero_for_noise():
    a = np.repeat([0, 1, 2], 20)
    assert adjusted_rand(a, (a + 1) % 3) == 1.0
    rng = np.random.default_rng(2)
    assert abs(adjusted_rand(a, rng.permutation(a))) < 0.2


def test_a_tiny_set_is_left_alone():
    fit = spectral_clusters(np.eye(2))
    assert fit.k == 1 and np.all(fit.labels == 0)


def test_it_clusters_real_fields_and_can_be_compared_with_the_scaffolds():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.chemotypes import chemotype_labels
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    from cmfqsar.model import CMFModel

    subs = ["C", "CC", "CCC", "O", "N", "Cl", "F", "Br"]
    smis = ([f"c1ccccc1{s}" for s in subs] + [f"c1ccncc1{s}" for s in subs]
            + [f"C1CCCCC1{s}" for s in subs])
    mols, names = [], []
    for i, smi in enumerate(smis):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=5 + i)
        Chem.SanitizeMol(m)
        mols.append(m)
        names.append(f"m{i}")
    fs = FieldSet.create(alpha=0.6)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    K = model.joint_kernel(model.prepare_training_kernels(table))

    fit = spectral_clusters(K, k=3)
    scaffolds, _ = chemotype_labels(mols)
    agreement = adjusted_rand(fit.labels, scaffolds)
    assert fit.k == 3 and len(fit.labels) == len(mols)
    assert -1.0 <= agreement <= 1.0        # the number the benchmark reports


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
