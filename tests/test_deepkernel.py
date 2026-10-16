"""Tests for the nonlinear kernel layer and for alignment-based multiple kernel learning."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import combine_kernels  # noqa: E402
from cmfqsar.model import (CMFModel, align_mixing, grid_search, median_gamma_scale,  # noqa: E402
                           mkl_cv_predict, r2_score)


def _dataset(n=14, seed=0):
    """A small aligned series with an activity that depends on the fields."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1F",
              "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
              "c1ccc(N)cc1C", "c1ccc(Cl)cc1C", "c1ccc(F)cc1N", "c1ccccc1Br",
              "c1ccc(C)cc1C"][:n]
    fs = FieldSet.create(alpha=0.6)
    mf = []
    for i, s in enumerate(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed + i
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        mf.append(build_fields(m, fs, f"m{i}"))
    table = AtomTable.from_molecules(mf)
    rng = np.random.RandomState(seed)
    y = np.array([m.coords.shape[0] for m in mf], float)
    y = (y - y.mean()) / y.std() + 0.1 * rng.normal(size=len(mf))
    return table, y, fs, mf


def test_layer_is_positive_definite_and_linear_layer_changes_nothing():
    table, y, fs, _ = _dataset()
    lin = CMFModel(fieldset=fs.copy(), method="krr", normalize=True).fit(table, y, cv_folds=0)
    K_lin = lin.joint_kernel(lin.K_fields_train)
    for layer in ("gauss", "arccos1", "arccos2"):
        m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True, kernel_layer=layer)
        m.fit(table, y, cv_folds=0)
        K = m.apply_layer(K_lin, np.diag(K_lin), np.diag(K_lin))
        assert K.shape == K_lin.shape
        assert np.allclose(K, K.T, atol=1e-10)
        assert np.linalg.eigvalsh(K).min() > -1e-8          # positive semi-definite
    # "linear" (and the empty string) leave the kernel untouched
    for name in ("linear", "", None):
        m = CMFModel(fieldset=fs.copy(), kernel_layer=name)
        assert m.is_linear
        assert np.array_equal(m.apply_layer(K_lin, np.diag(K_lin), np.diag(K_lin)), K_lin)


def test_joint_self_matches_the_training_diagonal():
    """The self-kernel used for new molecules must agree with the training diagonal.

    A field that is identically zero for a molecule (no H-bond donor, no charge) keeps a zero
    diagonal after cosine normalisation, so the normalised self-kernel is not simply one.
    """
    table, y, fs, _ = _dataset()
    for normalize in (False, True):
        m = CMFModel(fieldset=fs.copy(), method="krr", normalize=normalize)
        m.fit(table, y, cv_folds=0)
        K = m.joint_kernel(m.K_fields_train)
        assert np.allclose(m.joint_self(table), np.diag(K), rtol=1e-9, atol=1e-9)


def test_arccos_self_kernel_equals_the_base_diagonal():
    """The degree-1 arc-cosine kernel keeps k(x, x) = ||x||^2, which the test block relies on."""
    table, y, fs, _ = _dataset()
    m = CMFModel(fieldset=fs.copy(), method="krr", kernel_layer="arccos1").fit(table, y, cv_folds=0)
    K_lin = m.joint_kernel(m.K_fields_train)
    d = np.diag(K_lin)
    K = m.apply_layer(K_lin, d, d)
    assert np.allclose(np.diag(K), d, rtol=1e-8)


def test_gauss_layer_predicts_and_bandwidth_is_in_median_units():
    table, y, fs, mf = _dataset()
    m = CMFModel(fieldset=fs.copy(), method="krr", params={"lam": 1e-3, "gamma": 1.0},
                 normalize=True, kernel_layer="gauss")
    m.fit(table, y, cv_folds=0)
    K_lin = m.joint_kernel(m.K_fields_train)
    assert m.gamma_scale_ == pytest.approx(median_gamma_scale(K_lin))
    # the kernel of a molecule with itself is 1 and the fit reproduces the activities
    K = m.apply_layer(K_lin, np.diag(K_lin), np.diag(K_lin))
    assert np.allclose(np.diag(K), 1.0)
    assert r2_score(y, m.learner_.predict(K)) > 0.9
    # a prediction for the training molecules themselves reproduces the fit
    assert np.allclose(m.predict(table), m.learner_.predict(K), atol=1e-6)


def test_interpretation_is_refused_for_a_nonlinear_kernel_but_rule_2_works():
    table, y, fs, mf = _dataset()
    m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True, kernel_layer="gauss")
    m.fit(table, y, cv_folds=0)
    for call in (lambda: m.atom_contributions(mf[0]), lambda: m.cofield_coefficients(0),
                 lambda: m.predict_with_variance(table)):
        with pytest.raises(ValueError, match="linear"):
            call()
    # decomposition rule 2 (contributions of the training molecules) holds for any kernel
    tc = m.training_contributions(mf[0])
    assert tc.shape == (len(y),)
    assert float(tc.sum() + m.bias) == pytest.approx(float(m.predict(AtomTable.from_molecules([mf[0]]))[0]),
                                                     abs=1e-8)


def test_model_state_round_trip_keeps_the_layer():
    table, y, fs, mf = _dataset()
    m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True, kernel_layer="gauss")
    m.fit(table, y, cv_folds=0)
    p1 = m.predict(AtomTable.from_molecules(mf[:3]))
    m2 = CMFModel.from_state(m.state())
    assert m2.kernel_layer == "gauss"
    assert np.allclose(m2.predict(AtomTable.from_molecules(mf[:3])), p1, atol=1e-10)


def test_grid_search_selects_a_bandwidth_and_records_it():
    table, y, fs, _ = _dataset()
    m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True, kernel_layer="gauss")
    res = grid_search(table, y, m, [0.5, 1.0], lams=[1e-3, 0.1], gammas=[0.5, 1.0, 2.0],
                      folds=4, seed=0)
    assert "gamma" in res.best_params
    assert all("gamma" in row for row in res.table)
    assert len(res.table) == 2 * 2 * 3
    assert m.params["gamma"] == res.best_params["gamma"]
    # the continuous searches of the original scripts are linear-only
    with pytest.raises(ValueError, match="linear"):
        grid_search(table, y, m, [0.5, 1.0], lams=[1e-3, 0.1], lam_search="brent", folds=4)


def test_alignment_weights_favour_the_informative_field():
    """With an activity built from one field only, alignment must put the weight there."""
    rng = np.random.RandomState(0)
    n = 30
    X = rng.normal(size=(n, 4))
    Z = rng.normal(size=(n, 4))
    K = np.stack([X @ X.T, Z @ Z.T])              # field 0 informative, field 1 noise
    y = X[:, 0] * 2.0 + 0.01 * rng.normal(size=n)
    for mode in ("align", "alignf"):
        h = align_mixing(K, y, mode=mode, rho=0.0)
        assert h[0] > 3 * h[1]
        assert np.all(h >= 0)
    # disabled fields get zero weight, and the mean diagonal is preserved
    h = align_mixing(K, y, enabled=[True, False])
    assert h[1] == 0.0
    # the mean diagonal of the joint kernel over the enabled fields is preserved
    assert h[0] * np.mean(np.diag(K[0])) == pytest.approx(np.mean(np.diag(K[0])))
    with pytest.raises(ValueError):
        align_mixing(K, y, mode="nonsense")


def test_mkl_in_the_search_is_fold_safe_and_leaves_the_weights_in_the_model():
    table, y, fs, _ = _dataset()
    m = CMFModel(fieldset=fs.copy(), method="krr", normalize=True)
    res = grid_search(table, y, m, [0.5], lams=[1e-3, 0.1], folds=4, seed=0, mkl="align")
    h = [s.h for s in m.fieldset.specs]
    assert any(abs(v - 1.0) > 1e-6 for v in h)                # the weights really changed
    assert res.best_h == {s.name: s.h for s in m.fieldset.specs}
    assert all("h" in row for row in res.table)
    # the reported cross-validation does not use the left-out activities for the weights
    Kf = m.K_fields_train
    yp, h_full = mkl_cv_predict(Kf, y, m, {"lam": res.best_params["lam"]}, folds=4, seed=0)
    assert r2_score(y, yp) == pytest.approx(res.best_q2, abs=1e-9)
    K_all = combine_kernels(Kf, h_full, m.scale)
    assert np.linalg.eigvalsh(K_all).min() > -1e-8
    with pytest.raises(ValueError, match="either"):
        grid_search(table, y, m, [0.5], lams=[0.1], folds=4, mkl="align", optimize_h=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
