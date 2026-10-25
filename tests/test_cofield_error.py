"""How much of a co-field is noise (plan item F2).

The decisive test is the first one: the closed-form standard deviation has to agree with what
actually happens when noisy activities are refitted, because the whole value of the thing is that
it replaces that experiment with one matrix product.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cofields import (CofieldError, cofield_variance, cofield_volume,  # noqa: E402
                              cofield_with_error, dual_covariance_factor, make_grid, mask_noise,
                              noise_level, per_molecule_fields)
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(Br)cc1O", "c1ccc(N)cc1C"]
Y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4])


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _fitted(y=Y, normalize=True, fit_bias=True, lam=0.2):
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(_mol(s), fs, f"m{i}") for i, s in enumerate(SMILES)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=normalize,
                     fit_bias=fit_bias)
    model.fit(table, np.asarray(y, float), cv_folds=4, cv_seed=0)
    return model, table


@pytest.fixture(scope="module")
def fitted():
    return _fitted()


@pytest.mark.parametrize("normalize,fit_bias", [(True, True), (False, False)])
def test_the_closed_form_is_what_refitting_noisy_activities_would_have_shown(normalize, fit_bias):
    model, table = _fitted(normalize=normalize, fit_bias=fit_bias)
    grid = make_grid(model.train.coords, margin=1.0, spacing=2.0)
    sigma = 0.3
    closed = cofield_with_error(model, 0, grid, sigma=sigma)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(300):
        m2 = CMFModel(fieldset=model.fieldset.copy(), method="krr", params={"lam": 0.2},
                      normalize=normalize, fit_bias=fit_bias)
        m2.set_training_kernels(table, model.K_fields_train, model.self_diag)
        m2.fit(table, Y + rng.normal(scale=sigma, size=len(Y)), cv_folds=4, cv_seed=0,
               reuse_kernels=True)
        draws.append(cofield_volume(m2, 0, grid))
    empirical = np.std(np.asarray(draws), axis=0, ddof=1)
    ok = closed.sd > 1e-9
    assert ok.sum() > 20
    ratio = empirical[ok] / closed.sd[ok]
    assert abs(float(np.mean(ratio)) - 1.0) < 0.1
    assert 0.75 < float(np.min(ratio)) and float(np.max(ratio)) < 1.35


def test_the_basis_is_the_one_the_co_field_is_made_of(fitted):
    """If these two ever disagreed, the error bars would belong to a different picture."""
    model, _table = fitted
    grid = make_grid(model.train.coords, margin=1.0, spacing=1.5)
    X = per_molecule_fields(model, 1, grid.points())
    assert X.shape == (model.train.n_mol, grid.n_points)
    assert np.allclose(model.coef @ X, cofield_volume(model, 1, grid).ravel(), atol=1e-10)


def test_the_error_is_never_negative_and_vanishes_where_no_molecule_reaches(fitted):
    model, _table = fitted
    grid = make_grid(model.train.coords, margin=12.0, spacing=2.0)
    ce = cofield_with_error(model, 0, grid)
    assert np.all(ce.sd >= 0)
    far = np.abs(ce.mean) < 1e-12
    assert far.any() and np.all(ce.sd[far] < 1e-8)
    assert np.all(ce.zscore[ce.sd == 0] == 0)


def test_the_error_grows_with_the_noise_it_propagates(fitted):
    model, _table = fitted
    grid = make_grid(model.train.coords, margin=1.0, spacing=2.0)
    one = cofield_variance(model, 0, grid, sigma=0.5)
    two = cofield_variance(model, 0, grid, sigma=1.0)
    assert np.allclose(two, 4.0 * one, rtol=1e-10)


def test_what_is_kept_does_not_depend_on_the_units_of_the_activity():
    """Ten times the activity is ten times the co-field and ten times its error."""
    a, _ = _fitted()
    b, _ = _fitted(y=10.0 * Y)
    grid = make_grid(a.train.coords, margin=1.0, spacing=1.5)
    za = cofield_with_error(a, 0, grid)
    zb = cofield_with_error(b, 0, grid)
    assert np.allclose(zb.mean, 10.0 * za.mean, rtol=1e-8)
    assert np.allclose(zb.sd, 10.0 * za.sd, rtol=1e-8)
    assert np.array_equal(np.abs(za.zscore) >= 2.0, np.abs(zb.zscore) >= 2.0)


def test_the_mask_cuts_what_it_says_and_nothing_else():
    mean = np.array([1.0, -3.0, 0.5, 2.0, 0.0])
    sd = np.array([1.0, 1.0, 1.0, 0.1, 0.0])
    kept = mask_noise(mean, sd, z=2.0)
    assert kept.tolist() == [0.0, -3.0, 0.0, 2.0, 0.0]
    assert np.array_equal(mask_noise(mean, sd, z=0.0), mean)
    ce = CofieldError(mean=mean, sd=sd, sigma=1.0, sigma_from="given", z=2.0)
    assert np.array_equal(ce.masked, kept)
    assert 0.0 <= ce.surviving <= 1.0
    assert "standard deviations" in ce.describe()


def test_a_learner_whose_coefficients_are_not_a_linear_map_is_refused(fitted):
    model, _table = fitted
    model.method = "svr"
    try:
        with pytest.raises(ValueError, match="quadratic program"):
            dual_covariance_factor(model)
    finally:
        model.method = "krr"


def test_the_noise_level_is_the_out_of_fold_error_unless_told_otherwise(fitted):
    model, _table = fitted
    assert noise_level(model, 0.42) == (0.42, "given")
    sigma, how = noise_level(model)
    assert sigma == pytest.approx(model.stats.rmse_cv) and "out-of-fold" in how
    keep = model.stats.rmse_cv
    model.stats.rmse_cv = float("nan")
    try:
        sigma, how = noise_level(model)
        assert sigma == pytest.approx(model.stats.rmse_fit) and "understates" in how
        model.stats.rmse_fit = float("nan")
        with pytest.raises(ValueError, match="cross-validate"):
            noise_level(model)
    finally:
        model.stats.rmse_cv = keep
        model.stats.rmse_fit = float(np.sqrt(np.mean((Y - model.predict(model.train)) ** 2)))


def test_a_pharmacophore_can_be_read_off_what_survives(fitted):
    from cmfqsar.pharmacophore import extract

    model, _table = fitted
    loose = extract(model, spacing=1.0, margin=2.0, min_z=0.0)
    assert loose, "the unmasked co-fields should give some features at all"
    strict = extract(model, spacing=1.0, margin=2.0, min_z=50.0)
    assert len(strict) < len(loose)
