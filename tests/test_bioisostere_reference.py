"""The reference of the bioisostere benchmark that is a model and needs no field:
kernel ridge on Morgan fingerprints, refitted without both members of every pair
(`fingerprint_kernel` and `fingerprint_changes` of benchmarks/run_bioisostere.py)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

RB = pytest.importorskip("run_bioisostere")

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1Br",
          "c1ccccc1CC", "c1ccccc1CO", "c1ccccc1C(=O)N", "c1ccccc1C#N", "c1ccccc1F"]


class _Pair:
    def __init__(self, a, b):
        self.a, self.b = a, b


def test_the_kernel_is_the_tanimoto_similarity():
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    K = RB.fingerprint_kernel(mols)
    assert K.shape == (len(mols), len(mols))
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T) and K.min() >= 0.0 and K.max() <= 1.0 + 1e-12
    # a molecule is more like itself than like any other
    off = K - np.eye(len(mols))
    assert off.max() < 1.0


def test_the_change_is_the_difference_of_two_predictions_without_the_pair():
    from cmfqsar.model import krr_fit
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    y = np.linspace(4.0, 8.0, len(mols))
    pairs = [_Pair(0, 3), _Pair(1, 7)]
    change, q2, lam = RB.fingerprint_changes(mols, y, pairs, folds=5, seed=0)
    assert len(change) == len(pairs) and np.isfinite(q2) and lam in RB.LAMS
    K = RB.fingerprint_kernel(mols)
    for p, got in zip(pairs, change):
        rows = np.asarray([i for i in range(len(y)) if i not in (p.a, p.b)], int)
        coef, bias = krr_fit(K[np.ix_(rows, rows)], y[rows], lam, True)
        pred = K[np.ix_([p.a, p.b], rows)] @ coef + bias
        assert got == pytest.approx(float(pred[1] - pred[0]))


def test_the_two_activities_of_a_pair_do_not_reach_its_own_prediction():
    mols = [Chem.MolFromSmiles(s) for s in SMILES]
    y = np.linspace(4.0, 8.0, len(mols))
    pairs = [_Pair(0, 3)]
    first = RB.fingerprint_changes(mols, y, pairs, folds=5, seed=0)[0][0]
    moved = y.copy()
    moved[0] += 5.0                                   # change only a member of the pair
    second = RB.fingerprint_changes(mols, moved, pairs, folds=5, seed=0)[0][0]
    # the ridge is chosen on all the molecules, so the two need not be equal; they must be close
    assert abs(first - second) < 0.5 * 5.0
