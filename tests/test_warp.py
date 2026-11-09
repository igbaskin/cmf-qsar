"""A warped Gaussian process: a monotone transform of the activity learned with the GP (ND11).

What has to be true, each checked against something the warp does not compute itself:

* the transform is monotone, its derivative is the derivative, and its inverse undoes it to
  machine precision, whatever the parameters;
* the objective is the log likelihood of the original activities: the Gaussian term of the
  evidence plus the Jacobian, and it does not depend on the units the activities are in;
* the fit finds a transform where one is needed -- an exponentiated or a floored series -- and
  leaves a Gaussian series alone, because six parameters have to beat their own penalty;
* the fold-wise cross-validation returns what the calibration in warped units needs;
* through the model, every prediction and interval comes back in the original units, the warp
  travels with the saved file, and the command line reports it and says which units the
  co-field is read in.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.warp import (DEFAULT_TERMS, Warp, cross_validate_intervals,  # noqa: E402
                          cross_validate_warped, fit_warp, parameter_penalty,
                          warped_log_evidence)


def _world(n=60, seed=0, noise=0.25):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    K = np.exp(-((X[:, None, :] - X[None, :, :]) ** 2).sum(-1) / 2.0)
    g = np.sin(X[:, 0]) + 0.5 * X[:, 1] + rng.normal(scale=noise, size=n)
    return K, g


def _warp():
    return Warp(centre=1.0, width=2.0, a=[0.5, 0.2], b=[1.5, 3.0], c=[-0.3, 0.8])


# ------------------------------------------------------------------ the transform
def test_the_transform_is_monotone_and_its_inverse_undoes_it():
    w = _warp()
    y = np.linspace(-20.0, 20.0, 401)
    z = w.forward(y)
    assert np.all(np.diff(z) > 0)
    assert np.abs(w.inverse(z) - y).max() < 1e-10
    assert np.all(w.derivative(y) > 0)


def test_the_derivative_is_the_derivative():
    w = _warp()
    y = np.linspace(-6.0, 8.0, 50)
    h = 1e-6
    fd = (w.forward(y + h) - w.forward(y - h)) / (2 * h)
    assert np.abs(fd - w.derivative(y)).max() < 1e-7
    assert w.log_jacobian(y) == pytest.approx(float(np.sum(np.log(w.derivative(y)))))


def test_the_identity_is_the_standardisation_alone():
    y = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    w = Warp.identity(y)
    assert w.is_identity and w.n_terms == 0
    assert np.abs(w.inverse(w.forward(y)) - y).max() < 1e-12
    assert w.centre == pytest.approx(np.median(y))
    assert np.allclose(w.forward(y) * w.width + w.centre, y)


def test_what_a_transform_must_not_be():
    with pytest.raises(ValueError, match="monotone"):
        Warp(centre=0.0, width=1.0, a=[-0.1], b=[1.0], c=[0.0])
    with pytest.raises(ValueError, match="monotone"):
        Warp(centre=0.0, width=1.0, a=[0.1], b=[0.0], c=[0.0])
    with pytest.raises(ValueError, match="width"):
        Warp(centre=0.0, width=0.0)
    with pytest.raises(ValueError, match="one value per term"):
        Warp(centre=0.0, width=1.0, a=[0.1, 0.2], b=[1.0], c=[0.0])


def test_the_interval_comes_back_in_the_original_units_and_is_not_symmetric():
    w = Warp(centre=0.0, width=1.0, a=[1.5], b=[2.0], c=[0.0])   # a strong step at the centre
    med, lo, hi = w.interval([0.0, 2.0], [0.5, 0.5], level=0.9)
    assert np.all(lo < med) and np.all(med < hi)
    assert med[0] == pytest.approx(w.inverse(np.array([0.0]))[0])
    # where the transform is steep the interval is narrow in the original units
    assert (hi[0] - lo[0]) < (hi[1] - lo[1])


def test_the_dictionary_round_trip():
    w = _warp()
    w.gain = 3.5
    back = Warp.from_dict(w.to_dict())
    y = np.linspace(-3, 3, 7)
    assert np.allclose(back.forward(y), w.forward(y)) and back.gain == 3.5


# ------------------------------------------------------------------ the objective
def test_the_objective_is_the_evidence_plus_the_jacobian():
    K, y = _world()
    w = _warp()
    want = log_evidence(K, w.forward(y), 0.1, fit_bias=True, profile=True) + w.log_jacobian(y)
    assert warped_log_evidence(K, y, 0.1, w) == pytest.approx(want)


def test_the_units_of_the_activity_do_not_matter():
    """The standardisation absorbs them: the fit on 3 y + 7 is the fit on y."""
    K, y = _world()
    a = fit_warp(K, np.exp(y), 0.1, seed=0)
    b = fit_warp(K, 3.0 * np.exp(y) + 7.0, 0.1, seed=0)
    assert a.gain == pytest.approx(b.gain, rel=1e-4, abs=1e-3)
    assert np.allclose(a.a, b.a, rtol=1e-3, atol=1e-4) and np.allclose(a.c, b.c, atol=1e-3)


# ------------------------------------------------------------------ the fit
def test_a_skewed_series_gets_a_warp_and_a_gaussian_one_is_left_alone():
    K, g = _world()
    plain = fit_warp(K, g, 0.1, seed=0)
    assert plain.is_identity
    assert 0.0 <= plain.gain < parameter_penalty(len(g))
    skewed = fit_warp(K, np.exp(g), 0.1, seed=0)
    assert not skewed.is_identity
    assert skewed.gain > parameter_penalty(len(g))
    floored = fit_warp(K, np.maximum(g, -0.3), 0.1, seed=0)
    assert not floored.is_identity and floored.gain > skewed.gain
    # the warped activities of the exponentiated series are closer to normal than the raw ones
    from scipy.stats import skew
    assert abs(skew(skewed.forward(np.exp(g)))) < abs(skew(np.exp(g)))


def test_the_penalty_is_the_bayesian_information_criterions():
    assert parameter_penalty(60, 2) == pytest.approx(0.5 * 6 * np.log(60))
    assert parameter_penalty(1, 2) == pytest.approx(0.5 * 6 * np.log(2))   # never below two


def test_min_gain_zero_keeps_any_gain_and_the_fit_refuses_what_it_cannot_use():
    K, g = _world()
    w = fit_warp(K, g, 0.1, seed=0, min_gain=0.0)
    assert not w.is_identity and w.gain > 0
    with pytest.raises(ValueError, match="four activities"):
        fit_warp(K[:3, :3], g[:3], 0.1)
    bad = g.copy()
    bad[0] = np.nan
    with pytest.raises(ValueError, match="measured"):
        fit_warp(K, bad, 0.1)
    with pytest.raises(ValueError, match="activities"):
        fit_warp(K[:10, :10], g, 0.1)
    assert fit_warp(K, g, 0.1, n_terms=0).is_identity


# ------------------------------------------------------------------ the folds
def test_the_cross_validation_returns_what_the_calibration_needs():
    K, g = _world()
    y = np.exp(g)
    pred, mean_z, sd_z, z_true = cross_validate_warped(K, y, 0.1, folds=5, seed=1, restarts=1)
    for arr in (pred, mean_z, sd_z, z_true):
        assert arr.shape == y.shape and np.all(np.isfinite(arr))
    assert np.all(sd_z > 0)
    # the held-out prediction is the inverse of the mean, fold by fold, so it is finite and
    # positive like the activities it predicts
    assert np.all(pred > 0)
    out = cross_validate_intervals(K, y, 0.1, level=0.9, folds=5, seed=1, restarts=1)
    for name in ("warped", "plain"):
        p, lo, hi = out[name]
        assert np.all(lo <= p) and np.all(p <= hi)
    assert out["gain"].shape == (5,)
    # the plain interval is symmetric about its prediction and the warped one need not be
    p, lo, hi = out["plain"]
    assert np.allclose(hi - p, p - lo)


# ------------------------------------------------------------------ through the model
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(Br)cc1O", "c1ccc(N)cc1C", "c1ccc(F)cc1N", "c1ccc(Br)cc1N"]
Y0 = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4, 1.3, 1.9])


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


def _fitted(warp=2, min_gain=0.0, skew=True, **kw):
    """A small warped model on a skewed activity, and the molecules it has not seen."""
    y = np.exp(1.2 * Y0) if skew else Y0
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(_mol(s, i + 1), fs, f"m{i}") for i, s in enumerate(SMILES)]
    tr, te = list(range(10)), list(range(10, 14))
    params = {"lam": 0.1}
    if min_gain is not None:
        params["warp_min_gain"] = min_gain
    model = CMFModel(**({"fieldset": fs, "method": "krr", "params": params, "warp": warp} | kw))
    train = AtomTable.from_molecules([mf[i] for i in tr])
    model.fit(train, y[tr], cv_folds=5, cv_seed=0)
    return model, train, AtomTable.from_molecules([mf[i] for i in te]), y[tr], y[te]


def test_a_warped_model_predicts_in_the_original_units():
    model, train, test, y_tr, _y_te = _fitted()
    assert model.warp_ is not None and not model.warp_.is_identity
    K = model._linear_train_kernel()
    in_warped_units = model.learner_.predict(K)
    assert np.allclose(model.predict(train), model.warp_.inverse(in_warped_units))
    # the statistics are in the original units: the fitted values are close to the activities
    assert np.isfinite(model.stats.q2) and model.stats.r2 > 0.5
    assert np.allclose(model.stats.extra["y_fit"], model.predict(train))
    pred = model.predict(test)
    assert pred.shape == (4,) and np.all(pred > 0)         # like the activities, which are exp
    # the coefficients were fitted on the warped activities
    assert np.allclose(model._y_model(), model.warp_.forward(y_tr))


def test_the_interval_and_the_calibration_of_a_warped_model():
    model, _train, test, _y_tr, _y_te = _fitted()
    med, lo, hi = model.predict_warped_interval(test, level=0.9)
    assert np.allclose(med, model.predict(test))
    assert np.all(lo < med) and np.all(med < hi)
    cal = model.calibration(folds=5, seed=0)
    assert cal.n == 10 and np.isfinite(cal.z_rms)
    # the conformal interval is calibrated on residuals in the original units and holds there
    conf = model.calibrate_conformal(level=0.8)
    pred, lo_c, hi_c = model.predict_interval(test)
    assert np.allclose(pred, model.predict(test)) and np.all(lo_c <= pred) and np.all(pred <= hi_c)
    assert conf.usable or model.stats.n_train < 10


def test_a_series_close_to_normal_is_left_alone_through_the_model():
    """Under the default penalty the full fit keeps the identity on this series, and a model
    whose warp is the identity predicts exactly as the plain model does. The folds re-make the
    decision on fewer molecules, where the penalty is lower, so q2 is compared with the identity
    forced in every fold: the identity path has to *be* the plain path."""
    model, train, _test, y_tr, _ = _fitted(min_gain=None, skew=False)
    assert model.warp_ is not None and model.warp_.is_identity
    plain = CMFModel(fieldset=model.fieldset, method="krr", params={"lam": 0.1})
    plain.fit(train, y_tr, cv_folds=5, cv_seed=0)
    assert np.allclose(model.predict(train), plain.predict(train), atol=1e-8)
    forced, _tr, _te, _y_tr, _ = _fitted(min_gain=1e9, skew=False)
    assert forced.warp_.is_identity
    assert forced.stats.q2 == pytest.approx(plain.stats.q2, abs=1e-8)
    assert forced.stats.rmse_cv == pytest.approx(plain.stats.rmse_cv, abs=1e-8)


def test_the_warp_travels_with_the_file(tmp_path):
    from cmfqsar.dataio import load_model, save_model
    model, _train, test, _y_tr, _y_te = _fitted()
    path = str(tmp_path / "w.cmf")
    save_model(path, model)
    back, _extra = load_model(path)
    assert back.warp == 2 and back.warp_ is not None
    assert np.allclose(back.predict(test), model.predict(test))
    med, lo, hi = back.predict_warped_interval(test)
    med0, lo0, hi0 = model.predict_warped_interval(test)
    assert np.allclose(lo, lo0) and np.allclose(hi, hi0)


def test_a_fold_model_keeps_the_warp_setting_and_the_refusals_name_the_reason():
    model, train, _test, y_tr, _ = _fitted()
    assert model.fresh_copy().warp == 2
    with pytest.raises(ValueError, match="ridge fit"):
        CMFModel(fieldset=model.fieldset, method="kpls", params={"components": 2},
                 warp=2).fit(train, y_tr, cv_folds=0)
    low = CMFModel(fieldset=model.fieldset, method="krr", params={"lam": 0.1}, warp=2)
    with pytest.raises(ValueError, match="low-rank"):
        low.fit_low_rank(train, y_tr, train, lam=0.1)
    plain = CMFModel(fieldset=model.fieldset, method="krr", params={"lam": 0.1})
    plain.fit(train, y_tr, cv_folds=0)
    with pytest.raises(ValueError, match="no warp"):
        plain.predict_warped_interval(train)


def test_build_reports_the_warp_and_the_units(tmp_path, capsys):
    """`cmf build --warp 2`: the transform is reported, and so are the units of the co-field."""
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    y = np.exp(1.2 * Y0)
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(SMILES):
            m = _mol(smi, k + 1)
            m.SetProp("_Name", f"m{k}")
            m.SetProp("activity", f"{y[k]:.4f}")
            w.write(m)
    saved = os.path.join(str(tmp_path), "w.cmf")
    capsys.readouterr()
    main(["build", path, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "1.0", "--lams", "0.1", "--test-fraction", "0.25", "--folds", "5",
          "--warp", "2", "--save", saved])
    text = capsys.readouterr().out
    assert "warp:" in text
    if "the identity" not in text:
        assert "read there" in text and "carried back to the original units" in text
    main(["predict", saved, path, "--out", os.path.join(str(tmp_path), "p.csv")])
    assert "Predictions written" in capsys.readouterr().out
