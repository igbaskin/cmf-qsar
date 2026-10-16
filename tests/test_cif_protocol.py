"""Tests for the features added for the benchmark against the 2015 CIF paper: the port of R's
Nelder-Mead minimiser, kernel interpolation between alpha grid points, the simultaneous
(alpha, lambda) search, aggregated statistics of the reshuffled external CV and the
training-variance R2ext of the original scripts."""
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields, cif_types_in  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.model import CMFModel, grid_search, nested_cv  # noqa: E402
from cmfqsar.neldermead import interp_terms, nmmin  # noqa: E402


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _small_set():
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "c1ccccc1OC",
              "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C", "c1ccc(Br)cc1O", "c1ccc(N)cc1C"]
    mols = [_mol(s, i) for i, s in enumerate(smiles)]
    y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4])
    return mols, y


def test_nmmin_reproduces_r_optim():
    # Reference values from R 4.5: optim(c(0.25, 5), fn) with the default Nelder-Mead
    rosen = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
    x, f, n, fail = nmmin(rosen, [0.25, 5])
    assert abs(x[0] - 1.00642115362) < 1e-9 and abs(x[1] - 1.01373140633) < 1e-9
    assert abs(f - 0.000113119206809) < 1e-12 and n == 103 and fail == 0
    f1 = lambda p: (p[0] - 1.3) ** 2 + 3 * (p[1] - 0.7) ** 2 + 0.1 * abs(p[0] * p[1]) + np.sin(3 * p[0])
    x, f, n, fail = nmmin(f1, [0.25, 5])
    assert abs(x[0] - 1.51518258688) < 1e-9 and abs(x[1] - 0.674428149296) < 1e-9 and n == 71


def test_interp_terms():
    grid = [0.01, 0.05, 0.1, 0.25, 0.5]
    t = interp_terms(0.2426456, grid)
    assert t[0][0] == 0.1 and t[1][0] == 0.25 and abs(t[0][1] + t[1][1] - 1) < 1e-12
    assert abs(t[1][1] - (0.2426456 - 0.1) / 0.15) < 1e-12
    assert interp_terms(12.0, grid) == [(0.5, 1.0)] and interp_terms(0.001, grid) == [(0.01, 1.0)]
    assert interp_terms(0.25, grid) == [(0.1, 0.0), (0.25, 1.0)] or interp_terms(0.25, grid) == [(0.25, 1.0)]


def test_interpolated_kernels_and_nelder_mead_search():
    mols, y = _small_set()
    types = cif_types_in(mols, "sybyl", 3)
    assert "C.ar" in types and "H" in types
    fs = FieldSet.create(physchem=[], cif_types=types, alpha=0.7)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])

    # kernels for an alpha between the grid points are linear interpolations
    model = CMFModel(fieldset=fs.copy(), scale_fields=False, alpha_grid=[0.5, 1.0])
    K_lo = kernel_matrices(table, table, FieldSet.create(physchem=[], cif_types=types, alpha=0.5).specs)
    K_hi = kernel_matrices(table, table, FieldSet.create(physchem=[], cif_types=types, alpha=1.0).specs)
    assert np.allclose(model._raw_kernels(table, table), 0.6 * K_lo + 0.4 * K_hi)
    model.fieldset.set_alpha(3.0)
    assert np.allclose(model._raw_kernels(table, table), K_hi)

    # simultaneous Nelder-Mead search over (alpha, lambda)
    alphas = [0.25, 0.5, 1.0, 2.0]
    model = CMFModel(fieldset=fs.copy(), scale_fields=False)
    res = grid_search(table, y, model, alphas, lams=[1.0], folds=4, seed=None, alpha_search="nelder-mead",
                      nm_start=(0.5, 1.0))
    assert alphas[0] <= res.best_alpha <= alphas[-1] and "lam" in res.best_params
    assert res.table[-1].get("nelder_mead") and res.table[-1]["n_eval"] > 5
    assert model.alpha_grid == alphas
    model.fit(table, y, cv_folds=4, cv_seed=None, reuse_kernels=True)
    assert abs(model.stats.q2 - res.best_q2) < 1e-9
    # predictions for new molecules use the same interpolated kernels as the training kernels
    yp = model.predict(table)
    assert np.allclose(yp, model.predict_from_field_kernels(model.K_fields_train))
    assert np.allclose(yp, model.stats.extra["y_fit"])

    # reshuffled external CV: aggregated statistics of the mean prediction
    tmpl = CMFModel(fieldset=fs.copy(), scale_fields=False)
    r = nested_cv(table, y, tmpl, alphas, lams=[1.0], outer_folds=3, inner_folds=4, repeats=2, seed=0,
                  alpha_search="nelder-mead", nm_start=(0.5, 1.0))
    assert np.isfinite(r.q2_ex_aggr) and np.isfinite(r.rmse_ex_aggr) and np.isfinite(r.q2_ex_sd)
    assert len(r.per_repeat) == 2 and len(r.per_repeat[0]["hyper"]) == 3
    assert r.rmse_ex_aggr <= max(d["rmse"] for d in r.per_repeat) + 1e-12
    # the outer folds are reshuffled between repeats even with interleaved inner folds
    r2 = nested_cv(table, y, tmpl, alphas, lams=[1.0], outer_folds=3, inner_folds=4, repeats=2, seed=0,
                   inner_seed=None, alpha_search="nelder-mead", nm_start=(0.5, 1.0))
    assert r2.per_repeat[0]["q2"] != r2.per_repeat[1]["q2"]


def test_r2_pred_with_training_variance():
    mols, y = _small_set()
    fs = FieldSet.create(param_scheme="tripos", alpha=0.7)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    train, test = AtomTable.from_molecules(mf[:9]), AtomTable.from_molecules(mf[9:])
    model = CMFModel(fieldset=fs, scale_fields=False, params={"lam": 1.0})
    model.fit(train, y[:9], cv_folds=0)
    yp = model.evaluate_test(test, y[9:])
    rss = float(np.sum((y[9:] - yp) ** 2))
    expected = 1.0 - (rss / 2) / np.var(y[:9], ddof=1)
    assert abs(model.stats.r2_pred_train_var - expected) < 1e-12
