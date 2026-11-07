"""The local kernel without a kernel matrix (plan item LP5).

A per-atom target is a large problem: four hundred molecules are five thousand carbons and the
free databases hold hundreds of thousands. What these tests fix is that the primal route is the
same model and not an approximation of it -- the same predictions, the same ridge, the same folds
-- and that the Nystrom route, which the overlap kernel does need, behaves as an approximation
should.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel  # noqa: E402
from cmfqsar.localenv import (LOCAL_SPEC, choose_lam, cross_validate,  # noqa: E402
                              environments, local_features, local_kernel)
from cmfqsar.locallarge import (NYSTROM_CUTOFF, choose_lam_primal,  # noqa: E402
                                cross_validate_primal, learning_curve, normalised_rows,
                                nystrom_kernel, primal_fit, primal_predict)

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "CCOc1ccccc1N", "CC(C)c1ccccc1O", "CCCCc1ccccc1",
          "CC(=O)Nc1ccc(O)cc1", "COc1ccc(C=O)cc1", "CCc1ccccc1C", "Clc1ccc(O)cc1"]


def _mol(smiles, name="m", seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    m.SetProp("_Name", name)
    return m


@pytest.fixture(scope="module")
def built():
    """Carbons of a few molecules, their environments and the features of those."""
    fs = FieldSet.create()
    mols, fields = [], []
    for i, s in enumerate(SMILES):
        m = _mol(s, f"m{i}", seed=i + 1)
        for a in m.GetAtoms():
            if a.GetSymbol() == "C":
                a.SetDoubleProp("nh", float(sum(1 for nb in a.GetNeighbors()
                                                if nb.GetSymbol() == "H")))
        mols.append(m)
        fields.append(build_fields(m, fs, name=f"m{i}"))
    targets = AtomTargets.from_molecules(mols, "nh", names=[f"m{i}" for i in range(len(mols))])
    envs = environments(fields, targets, rho=0.5)
    X = local_features(envs)
    return fields, targets, X


# --------------------------------------------------------------- the rows are the kernel


def test_the_rows_multiply_back_to_the_kernel(built):
    _fields, _t, X = built
    Z, scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    assert Z.shape == X.shape
    assert np.sum(Z * Z, axis=1) == pytest.approx(np.ones(len(Z)))
    assert Z @ Z.T == pytest.approx(local_kernel(X), abs=1e-10)
    # and the scale that comes back prepares a test set the way a kernel would
    Z2, _ = normalised_rows(X[:3], scale=scale)
    assert Z2 == pytest.approx(Z[:3], abs=1e-12)
    assert Z2 @ Z.T == pytest.approx(
        autocorrelation_kernel(X, X[:3], scale=scale, n_bins=LOCAL_SPEC.n_bins), abs=1e-10)


def test_a_channel_that_is_always_zero_does_not_divide_by_nothing(built):
    _fields, _t, X = built
    dead = np.array(X, float)
    dead[:, : dead.shape[1] // 2] = 0.0
    Z, _scale = normalised_rows(dead, n_bins=LOCAL_SPEC.n_bins)
    assert np.isfinite(Z).all()
    assert np.sum(Z * Z, axis=1) == pytest.approx(np.ones(len(Z)))


# ------------------------------------------------- the primal is the same model as the dual


def test_the_primal_fit_predicts_what_the_dual_fit_predicts(built):
    from cmfqsar.model import krr_fit

    _fields, t, X = built
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    y = np.asarray(t.y, float)
    K = Z @ Z.T
    for lam in (1e-3, 1e-1, 1.0):
        a, b = krr_fit(K, y, lam)
        dual = K @ a + b
        w, c = primal_fit(Z, y, lam)
        assert primal_predict(Z, w, c) == pytest.approx(dual, abs=1e-6)


def test_the_cross_validation_agrees_fold_for_fold(built):
    _fields, t, X = built
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    K = Z @ Z.T
    for lam in (1e-2, 1.0):
        a = cross_validate(K, t, lam, folds=3, seed=0)
        b = cross_validate_primal(Z, t, lam, folds=3, seed=0)
        assert b.predicted == pytest.approx(a.predicted, abs=1e-6)
        assert b.q2 == pytest.approx(a.q2, abs=1e-9)
        assert b.folds == a.folds and b.n_samples == a.n_samples


def test_the_ridge_search_picks_the_same_value(built):
    _fields, t, X = built
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    a = choose_lam(Z @ Z.T, t, folds=3, seed=0, name="dual")
    b = choose_lam_primal(Z, t, folds=3, seed=0, name="primal")
    assert b.lam == a.lam
    assert b.q2 == pytest.approx(a.q2, abs=1e-9)


def test_without_a_bias_the_fit_is_the_plain_ridge(built):
    _fields, t, X = built
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    y = np.asarray(t.y, float)
    w, b = primal_fit(Z, y, 1.0, fit_bias=False)
    assert b == 0.0
    direct = np.linalg.solve(Z.T @ Z + np.eye(Z.shape[1]), Z.T @ y)
    assert w == pytest.approx(direct, abs=1e-8)


# ------------------------------------------------------------------------ the learning curve


def test_the_curve_is_nested_and_reports_what_it_used(built):
    _fields, t, X = built
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    curve = learning_curve(Z, t, sizes=[4, 6, len(SMILES)], folds=3, seed=0)
    assert curve.molecules == [4, 6, len(SMILES)]
    assert curve.sizes == sorted(curve.sizes)          # nested subsets, so more atoms each time
    assert len(curve.q2) == len(curve.error) == len(curve.sizes)
    assert all(s > 0 for s in curve.seconds)
    assert "molecules" in curve.summary()


def test_a_size_with_nothing_in_it_is_skipped(built):
    _fields, t, X = built
    Z, _scale = normalised_rows(X, n_bins=LOCAL_SPEC.n_bins)
    curve = learning_curve(Z, t, sizes=[0, len(SMILES)], folds=3, seed=0)
    assert curve.molecules == [len(SMILES)]


# ------------------------------------------------------- the route the overlap kernel needs


def test_the_nystrom_kernel_is_exact_when_the_landmarks_are_everything(built):
    """Up to the conditioning: the pseudo-inverse drops directions the kernel barely has.

    How exactly is set by the kernel and by `NYSTROM_CUTOFF`, not by the code.  The smallest
    eigenvalue the cut-off keeps in this kernel is 4e-10 against a largest of 42, so the kept
    part has a condition number of 1e11, and dividing by the smallest kept eigenvalue loses
    about `eps * 1e11 = 2e-5` of relative accuracy.  A fixed bound of 1e-6 lies inside that
    range and therefore held on some machines and not on others: the measured error is 7.2e-7
    here and 1.5e-6 on the Ubuntu runner with the pinned versions.  The bound below is the one
    the conditioning allows, and it still fails by a wide margin if the pseudo-inverse is wrong.
    """
    _fields, _t, X = built
    K = local_kernel(X)
    back = nystrom_kernel(K, K)
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    kept = w[w > float(w.max()) * NYSTROM_CUTOFF]
    allowed = float(np.finfo(float).eps * w.max() / kept.min())
    assert np.linalg.norm(K - back) / np.linalg.norm(K) < allowed
    assert back == pytest.approx(K, abs=allowed * float(np.abs(K).max()))


def test_fewer_landmarks_approximate_and_more_landmarks_approximate_better(built):
    _fields, _t, X = built
    K = local_kernel(X)
    n = len(K)
    worse = nystrom_kernel(K[:, :3], K[np.ix_(range(3), range(3))])
    better = nystrom_kernel(K[:, : n - 1], K[np.ix_(range(n - 1), range(n - 1))])
    e_worse = np.linalg.norm(K - worse) / np.linalg.norm(K)
    e_better = np.linalg.norm(K - better) / np.linalg.norm(K)
    assert 0 <= e_better < e_worse
    assert np.linalg.eigvalsh(0.5 * (worse + worse.T)).min() > -1e-8   # still a kernel


def test_repeated_landmarks_do_not_blow_the_approximation_up(built):
    """A pseudo-inverse through the eigendecomposition is what keeps this finite."""
    _fields, _t, X = built
    K = local_kernel(X)
    idx = [0, 0, 1, 2]                                  # the first landmark twice over
    approx = nystrom_kernel(K[:, idx], K[np.ix_(idx, idx)])
    assert np.isfinite(approx).all()
    assert np.linalg.norm(K - approx) / np.linalg.norm(K) < 1.0
