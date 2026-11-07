"""The model as the fast level of a hierarchy (plan item MS1): the standard deviation in the
units of the activity, its calibration, and the tolerance rule."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.evidence import noise_scale  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, krr_fit  # noqa: E402
from cmfqsar.surrogate import (Calibration, Handoff, Surrogate, cross_validate,  # noqa: E402
                               defer, tolerance_for_budget)


def _rbf(X, Z, width=1.0):
    d2 = ((X[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2.0 * width ** 2))


def _world(n=60, d=3, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    K = _rbf(X, X)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(scale=noise, size=n) + 2.0
    return X, K, y


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("fit_bias", [True, False])
def test_the_mean_is_the_ridge_prediction_with_its_bias(fit_bias):
    _X, K, y = _world()
    tr, te = np.arange(45), np.arange(45, 60)
    lam = 0.2
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, fit_bias=fit_bias)
    s = Surrogate.fit(K[np.ix_(tr, tr)], y[tr], lam, fit_bias=fit_bias)
    mean, sd = s.predict(K[np.ix_(te, tr)], np.diag(K)[te])
    assert mean == pytest.approx(K[np.ix_(te, tr)] @ a + b, abs=1e-8)
    assert np.all(sd > 0)


def test_the_scale_is_the_mean_squared_whitened_residual():
    """s = r^T A^{-1} r / (n - 1) with r the residual about the marginalised mean."""
    _X, K, y = _world()
    lam, n = 0.2, len(y)
    A = K + lam * np.eye(n)
    Ai = np.linalg.inv(A)
    one = np.ones(n)
    b = (one @ Ai @ y) / (one @ Ai @ one)
    r = y - b
    want = (r @ Ai @ r) / (n - 1)
    assert Surrogate.fit(K, y, lam).scale == pytest.approx(want)
    assert noise_scale(K, y, lam) == pytest.approx(want)
    assert Surrogate.fit(K, y, lam, fit_bias=False).scale == pytest.approx((y @ Ai @ y) / n)


def test_the_variance_has_the_term_of_the_unknown_mean():
    """Eq. 2.42 of Rasmussen and Williams with one basis function, the constant."""
    X, K, y = _world()
    lam, n = 0.2, len(y)
    A = K + lam * np.eye(n)
    Ai = np.linalg.inv(A)
    one = np.ones(n)
    Z = np.array([[0.3, -0.2, 0.1], [8.0, 8.0, 8.0]])          # one inside, one far away
    k = _rbf(Z, X)
    s = Surrogate.fit(K, y, lam)
    _mean, sd = s.predict(k, np.ones(2), with_noise=False)
    t = 1.0 - k @ Ai @ one
    want = s.scale * (1.0 - np.einsum("ij,jk,ik->i", k, Ai, k) + t ** 2 / (one @ Ai @ one))
    assert sd ** 2 == pytest.approx(want, rel=1e-8)
    assert sd[1] > sd[0]
    # far from everything the kernel row is zero: the whole prior plus the uncertain level
    far = s.scale * (1.0 + 1.0 / (one @ Ai @ one))
    assert sd[1] ** 2 == pytest.approx(far, rel=1e-6)


def test_the_deviation_is_in_the_units_of_the_activity():
    """Scale y and the deviation scales with it; scale the kernel and the ridge together and
    nothing changes, because the scale absorbs the units of the kernel."""
    _X, K, y = _world()
    tr, te = np.arange(45), np.arange(45, 60)
    lam = 0.2
    base = Surrogate.fit(K[np.ix_(tr, tr)], y[tr], lam)
    m0, s0 = base.predict(K[np.ix_(te, tr)], np.diag(K)[te])
    scaled = Surrogate.fit(K[np.ix_(tr, tr)], 7.0 * y[tr], lam)
    m1, s1 = scaled.predict(K[np.ix_(te, tr)], np.diag(K)[te])
    assert m1 == pytest.approx(7.0 * m0) and s1 == pytest.approx(7.0 * s0)
    units = Surrogate.fit(3.0 * K[np.ix_(tr, tr)], y[tr], 3.0 * lam)
    m2, s2 = units.predict(3.0 * K[np.ix_(te, tr)], 3.0 * np.diag(K)[te])
    assert m2 == pytest.approx(m0) and s2 == pytest.approx(s0)
    assert base.noise_sd == pytest.approx(np.sqrt(base.scale * lam))


def test_with_and_without_the_noise():
    _X, K, y = _world()
    s = Surrogate.fit(K, y, 0.2)
    _m, sd_y = s.predict(K[:5], np.diag(K)[:5], with_noise=True)
    _m, sd_f = s.predict(K[:5], np.diag(K)[:5], with_noise=False)
    assert sd_y ** 2 == pytest.approx(sd_f ** 2 + s.scale * s.lam)


def test_the_fit_refuses_what_it_cannot_use():
    _X, K, y = _world()
    with pytest.raises(ValueError):
        Surrogate.fit(K[:10, :10], y, 0.2)
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        Surrogate.fit(K, bad, 0.2)
    s = Surrogate.fit(K, y, 0.2)
    with pytest.raises(ValueError):
        s.predict(K[:5, :10], np.ones(5))


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def test_calibration_holds_on_activities_drawn_from_the_model():
    """Data drawn from the Gaussian process the fit assumes: z has unit spread, two standard
    deviations hold about 95 %."""
    rng = np.random.default_rng(3)
    n = 400
    X = rng.normal(size=(n, 2))
    K = _rbf(X, X, width=0.8)
    s_true, lam = 2.5, 0.1
    L = np.linalg.cholesky(s_true * K + 1e-9 * np.eye(n))
    f = L @ rng.normal(size=n) + 5.0
    y = f + rng.normal(scale=np.sqrt(s_true * lam), size=n)
    mean, sd = cross_validate(K, y, lam, folds=10, seed=0)
    cal = Calibration.of(y, mean, sd)
    assert cal.n == n
    assert 0.85 <= cal.z_rms <= 1.15
    assert 0.60 <= cal.coverage_1sd <= 0.76
    assert 0.91 <= cal.coverage_2sd <= 0.99
    assert "calibrated" in cal.describe()
    # the out-of-fold predictions are the model's own cross-validation
    from cmfqsar.model import cv_predict
    assert mean == pytest.approx(cv_predict(K, y, "krr", {"lam": lam}, folds=10, seed=0), abs=1e-8)


def test_a_model_that_is_too_sure_is_told_so_with_the_factor():
    rng = np.random.default_rng(4)
    y = rng.normal(size=200)
    mean = y + rng.normal(scale=0.5, size=200)
    cal = Calibration.of(y, mean, np.full(200, 0.1))
    assert cal.z_rms > 3.0 and cal.factor == cal.z_rms
    assert "surer than it should be" in cal.describe()
    assert cal.coverage_2sd < 0.5
    cal2 = Calibration.of(y, mean, np.full(200, 5.0))
    assert "less sure" in cal2.describe()


def test_too_few_activities_say_so():
    cal = Calibration.of([1.0, 2.0], [1.1, 1.9], [0.2, 0.2])
    assert cal.n == 2 and np.isnan(cal.z_rms) and "too few" in cal.describe()


# --------------------------------------------------------------------------- #
# The rule
# --------------------------------------------------------------------------- #

def test_the_tolerance_and_the_budget_agree():
    rng = np.random.default_rng(5)
    sd = rng.uniform(0.1, 1.0, size=1000)
    tol = tolerance_for_budget(sd, 0.25)
    share = defer(sd, tol).mean()
    assert 0.23 <= share <= 0.27
    assert defer(sd, tolerance_for_budget(sd, 0.0)).sum() == 0
    assert defer(sd, tolerance_for_budget(sd, 1.0)).all()
    with pytest.raises(ValueError):
        tolerance_for_budget(sd, 1.5)


def test_the_handoff_measures_both_sides():
    rng = np.random.default_rng(6)
    n = 300
    sd = rng.uniform(0.1, 1.0, size=n)
    y = rng.normal(size=n)
    mean = y + rng.normal(size=n) * sd           # the error really is of the size the sd says
    h = Handoff.of(sd, 0.5, y=y, mean=mean)
    assert h.n == n and h.n_answered + h.n_deferred == n
    assert h.rmse_answered < h.rmse_deferred
    assert h.mean_sd_answered < 0.5 < h.mean_sd_deferred
    assert "fine level" in h.describe() and "RMSE" in h.describe()
    unknown = Handoff.of(sd, 0.5)
    assert np.isnan(unknown.rmse_answered) and "RMSE" not in unknown.describe()


# --------------------------------------------------------------------------- #
# On a fitted CMF model
# --------------------------------------------------------------------------- #

def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _fitted(**kw):
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
              "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
              "c1ccc(Br)cc1O", "c1ccc(N)cc1C", "c1ccc(F)cc1N", "c1ccc(Br)cc1N"]
    y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4, 1.3, 1.9])
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(_mol(s), fs, f"m{i}") for i, s in enumerate(smiles)]
    tr, te = list(range(10)), list(range(10, 14))
    model = CMFModel(**({"fieldset": fs, "method": "krr", "params": {"lam": 0.1}} | kw))
    train = AtomTable.from_molecules([mf[i] for i in tr])
    model.fit(train, y[tr], cv_folds=5, cv_seed=0)
    return model, AtomTable.from_molecules([mf[i] for i in te]), y[te]


def test_a_fitted_model_gives_the_deviation_and_its_calibration():
    model, test, y_te = _fitted()
    yp, sd = model.predict_with_sd(test)
    assert yp == pytest.approx(model.predict(test), abs=1e-8)
    assert sd.shape == (4,) and np.all(sd > 0)
    _yp, var = model.predict_with_variance(test)
    s = model.surrogate()
    # the deviation is the posterior variance scaled, plus the term of the mean and the noise
    assert np.all(sd ** 2 >= s.scale * (var + s.lam) - 1e-9)
    cal = model.calibration(folds=5, seed=0)
    assert cal.n == 10 and np.isfinite(cal.z_rms)
    assert model.surrogate() is s                      # kept until the next fit
    h = Handoff.of(sd, tolerance_for_budget(sd, 0.5), y=y_te, mean=yp)
    assert h.n_deferred == 2


def test_the_refusals_name_the_reason():
    model, test, _y = _fitted(kernel_layer="gauss")
    with pytest.raises(ValueError, match="linear"):
        model.predict_with_sd(test)
    model, test, _y = _fitted(method="kpls", params={"components": 2})
    with pytest.raises(ValueError, match="krr or gp"):
        model.calibration()
