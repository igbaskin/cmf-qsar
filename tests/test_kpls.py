"""Kernel partial least squares (method "kpls"): the dual form equals PLS on explicit features,
the coefficients fit the learner wrapper like every other method, and the number of components
is chosen by cross-validation (run with: python -m pytest tests/test_kpls.py)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import (DEFAULT_COMPONENTS, CMFModel, _Learner, _param_grid,  # noqa: E402
                           cv_predict, grid_search, kpls_fit, krr_fit)


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _linear_problem(n=40, p=12, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, p) @ rng.randn(p, p)
    y = X @ rng.randn(p) + 0.3 * rng.randn(n) + 2.0
    Xt = rng.randn(10, p)
    return X, y, Xt


def test_kpls_equals_pls_on_explicit_features():
    """On the linear kernel K = X X', KPLS with m components is PLS regression with m
    components, on the training molecules and on new ones."""
    PLSRegression = pytest.importorskip("sklearn.cross_decomposition").PLSRegression
    X, y, Xt = _linear_problem()
    K, Kt = X @ X.T, Xt @ X.T
    for m in (1, 2, 3, 5, 8):
        a, b, used = kpls_fit(K, y, m)
        assert used == m
        ref = PLSRegression(n_components=m, scale=False).fit(X, y)
        assert np.allclose(K @ a + b, ref.predict(X).ravel(), atol=1e-9)
        assert np.allclose(Kt @ a + b, ref.predict(Xt).ravel(), atol=1e-9)
        # the centring is folded into the coefficients: they sum to zero, as KRR's with a bias do
        assert abs(a.sum()) < 1e-12


def test_kpls_full_rank_is_least_squares_and_stops_at_the_rank():
    X, y, Xt = _linear_problem(n=40, p=12)
    K = X @ X.T
    a, b, used = kpls_fit(K, y, 12)
    beta = np.linalg.lstsq(np.column_stack([X, np.ones(len(y))]), y, rcond=None)[0]
    assert np.allclose(K @ a + b, X @ beta[:-1] + beta[-1], atol=1e-8)
    # more components than the rank: the extra ones are not used
    a2, b2, used2 = kpls_fit(K, y, 30)
    assert used2 <= 12
    assert np.allclose(K @ a2 + b2, K @ a + b, atol=1e-6)
    # one component on a constant activity: nothing to fit, the prediction is the mean
    a3, b3, used3 = kpls_fit(K, np.full(len(y), 1.5), 3)
    assert used3 == 0 and np.allclose(a3, 0.0) and b3 == 1.5


def test_kpls_learner_wrapper_and_grid():
    X, y, Xt = _linear_problem()
    K = X @ X.T
    lr = _Learner("kpls", {"components": 3}).fit(K, y)
    a, b, used = kpls_fit(K, y, 3)
    assert np.allclose(lr.alpha_, a) and lr.b_ == b and lr.n_components_ == 3
    assert np.allclose(lr.predict(Xt @ X.T), Xt @ X.T @ a + b)
    # the hyper-parameter grid: explicit counts, de-duplicated and sorted, or the default list
    assert _param_grid("kpls", (), (), (), (), components=(4, 2, 2, 8)) == [
        {"components": 2}, {"components": 4}, {"components": 8}]
    assert _param_grid("kpls", (), (), (), ()) == [{"components": c} for c in DEFAULT_COMPONENTS]
    # cross-validated predictions run through the generic path
    ycv = cv_predict(K, y, "kpls", {"components": 2}, folds=5, seed=0)
    assert ycv.shape == y.shape and np.all(np.isfinite(ycv))


def test_kpls_cmf_model_selects_components_and_keeps_the_decomposition():
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "c1ccccc1OC",
              "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C", "c1ccc(Br)cc1O",
              "c1ccc(N)cc1C"]
    mols = [_mol(s, i) for i, s in enumerate(smiles)]
    fs = FieldSet.create(physchem=("electrostatic", "steric", "hydrophobic"), alpha=0.7)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.3])
    model = CMFModel(fieldset=fs, method="kpls", params={"components": 2}, normalize=True)
    res = grid_search(table, y, model, alphas=[0.7], folds=4, seed=0, components=(1, 2, 3, 4))
    assert set(res.best_params) == {"components"}
    assert res.best_params["components"] in (1, 2, 3, 4)
    assert len(res.table) == 4 and all("components" in r for r in res.table)
    model.params.update(res.best_params)
    model.fit(table, y, cv_folds=4)
    assert np.isfinite(model.stats.q2) and model.stats.r2 > 0.0
    assert abs(model.coef.sum()) < 1e-9
    # the model is a kernel expansion: atoms and training molecules sum to y - b
    for mol in mf[:3]:
        yp = model.predict(AtomTable.from_molecules([mol]))[0]
        assert abs(model.atom_contributions(mol).sum() + model.bias - yp) < 1e-8
        assert abs(model.training_contributions(mol).sum() + model.bias - yp) < 1e-8
    # with every component the fit is the unregularised one: it reproduces the training
    # activities, as kernel ridge regression with lambda -> 0 does
    K = model.joint_kernel(model.K_fields_train, table.descriptors)
    a_full, b_full, used = kpls_fit(K, y, len(y) - 1)
    assert used == len(y) - 1
    assert np.allclose(K @ a_full + b_full, y, atol=1e-5)
    a_krr, b_krr = krr_fit(K, y, 1e-12)
    assert np.allclose(K @ a_krr + b_krr, y, atol=1e-5)
