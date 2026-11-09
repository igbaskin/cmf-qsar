"""The standard deviation of item MS1 on the low-rank path of item B6 (plan item MS6).

The claim is that the Gaussian-process reading of a fit -- its scale in the units of the
activity, the deviation of a new structure, the term of the unknown mean -- has an m by m form
through the Woodbury identity, so a model fitted in the space of its landmarks can give it
without ever building the n by n kernel. These tests fix that claim the way the low-rank tests
fix the fit itself: with the training molecules as the landmarks the kernel is reproduced
exactly, so every number the low-rank reading gives must equal the exact one to round-off; with
fewer landmarks it is the consistent reading of the approximate model, which is checked for the
properties a deviation must keep.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cmfqsar import lowrank as LR  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.surrogate import Calibration, LowRankSurrogate, Surrogate, cross_validate  # noqa: E402

SMILES = ["c1ccccc1C(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
          "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O", "c1ccc(Br)cc1C(=O)O",
          "c1ccc(C#N)cc1C(=O)O", "c1ccc(OC)cc1C(=O)O", "c1ccc(CC)cc1C(=O)O",
          "c1ccc(CCl)cc1C(=O)O", "c1ccc(CO)cc1C(=O)O"]
#: the two readings evaluate the same quantities two ways, so they agree to round-off
EXACT = 1e-8


def _series(fields=("electrostatic", "steric"), alpha=0.5):
    fs = FieldSet.create(physchem=list(fields), alpha=float(alpha))
    mols = []
    for k, smi in enumerate(SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        AllChem.EmbedMolecule(m, p)
        mols.append(build_fields(m, fs, f"m{k}"))
    table = AtomTable.from_molecules(mols)
    rng = np.random.default_rng(0)
    y = 4.0 + 0.3 * np.arange(len(mols)) + rng.normal(scale=0.2, size=len(mols))
    return fs, table, y


def _exact(fs, table, y, lam=0.1):
    """The exact model on the first nine molecules, and its joint kernel over all twelve."""
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    model.params["lam"] = float(lam)
    Kf = np.asarray(model.prepare_training_kernels(table))
    K = np.asarray(model.joint_kernel(Kf), float)
    return model, K


def _split(table, y, n_train=9):
    tr, te = np.arange(n_train), np.arange(n_train, table.n_mol)
    return table.subset(tr), table.subset(te), y[tr], y[te], tr, te


# ------------------------------------------------- the landmarks are the molecules: exact
@pytest.mark.parametrize("fit_bias", [True, False])
def test_the_low_rank_reading_is_the_exact_one_when_the_landmarks_span_the_data(fit_bias):
    fs, table, y = _series()
    _model, K = _exact(fs, table, y)
    train, test, y_tr, _y_te, tr, te = _split(table, y)
    lam = 0.1
    exact = Surrogate.fit(K[np.ix_(tr, tr)], y_tr, lam, fit_bias=fit_bias)
    low = LR.build(train, fs, scale=np.ones(2))
    g = LR.gram(low, train, y_tr)
    reading = LowRankSurrogate.fit(g, lam, fit_bias=fit_bias)
    assert reading.scale == pytest.approx(exact.scale, rel=1e-8)
    assert reading.bias == pytest.approx(exact.bias, abs=EXACT)
    assert reading.dof == exact.dof and reading.n_train == exact.n_train
    assert reading.noise_sd == pytest.approx(exact.noise_sd, rel=1e-8)
    if fit_bias:
        assert reading.denom == pytest.approx(exact.denom, rel=1e-8)
    for with_noise in (True, False):
        m_e, s_e = exact.predict(K[np.ix_(te, tr)], np.diag(K)[te], with_noise=with_noise)
        m_l, s_l = reading.predict(low.transform(test), low.joint_self(test),
                                   with_noise=with_noise)
        assert np.abs(m_l - m_e).max() < EXACT
        assert np.abs(s_l - s_e).max() < EXACT


def test_the_scale_is_the_mean_squared_whitened_residual_in_m_by_m_form():
    """quad = (r^T r - (Phi^T r)^T A^-1 Phi^T r) / lam equals r^T (K + lam I)^-1 r."""
    fs, table, y = _series()
    _model, K = _exact(fs, table, y)
    n, lam = table.n_mol, 0.1
    A = K + lam * np.eye(n)
    Ai = np.linalg.inv(A)
    one = np.ones(n)
    b = (one @ Ai @ y) / (one @ Ai @ one)
    r = y - b
    want = (r @ Ai @ r) / (n - 1)
    low = LR.build(table, fs, scale=np.ones(2))
    reading = LowRankSurrogate.fit(LR.gram(low, table, y), lam)
    assert reading.scale == pytest.approx(want, rel=1e-8)


def test_the_folds_give_the_same_deviations_on_both_paths():
    """Same seed, same folds (`_kfold_indices` on both sides), so the out-of-fold numbers agree."""
    fs, table, y = _series()
    _model, K = _exact(fs, table, y)
    lam = 0.1
    m_e, s_e = cross_validate(K, y, lam, folds=4, seed=3)
    low = LR.build(table, fs, scale=np.ones(2))
    m_l, s_l, which = LR.cross_validate(low, table, y, lam, folds=4, seed=3, with_sd=True)
    assert np.abs(m_l - m_e).max() < EXACT
    assert np.abs(s_l - s_e).max() < EXACT
    assert sorted(set(which.tolist())) == [0, 1, 2, 3]
    # the two-value form is unchanged for the callers that do not want the deviation
    pred, which2 = LR.cross_validate(low, table, y, lam, folds=4, seed=3)
    assert np.abs(pred - m_l).max() < EXACT and np.array_equal(which, which2)


# ------------------------------------------------- through the model
def test_a_low_rank_model_gives_the_deviation_and_its_calibration():
    fs, table, y = _series()
    train, test, y_tr, _y_te, _tr, _te = _split(table, y)
    exact = CMFModel(fieldset=fs, method="krr", normalize=True)
    exact.params["lam"] = 0.1
    exact.fit(train, y_tr, cv_folds=3, cv_seed=0)
    low = CMFModel(fieldset=fs, method="krr", normalize=True)
    low.fit_low_rank(train, y_tr, train, lam=0.1, kind="molecules", folds=3, seed=0)
    assert low.is_low_rank
    s = low.surrogate()
    assert isinstance(s, LowRankSurrogate)
    assert s.scale == pytest.approx(exact.surrogate().scale, rel=1e-8)
    m_e, s_e = exact.predict_with_sd(test)
    m_l, s_l = low.predict_with_sd(test)
    assert np.abs(m_l - m_e).max() < EXACT and np.abs(s_l - s_e).max() < EXACT
    assert np.abs(m_l - low.predict(test)).max() < EXACT       # the mean is the prediction
    cal_e = exact.calibration(folds=3, seed=0)
    cal_l = low.calibration(folds=3, seed=0)
    assert isinstance(cal_l, Calibration)
    assert cal_l.z_rms == pytest.approx(cal_e.z_rms, rel=1e-6)
    assert cal_l.rmse == pytest.approx(cal_e.rmse, rel=1e-6)
    # the reading is kept on the model, as the exact one is
    assert low.surrogate() is s


def test_the_deviation_survives_a_file(tmp_path):
    from cmfqsar.dataio import load_model, save_model
    fs, table, y = _series()
    train, test, y_tr, _y_te, _tr, _te = _split(table, y)
    low = CMFModel(fieldset=fs, method="krr", normalize=True)
    low.fit_low_rank(train, y_tr, train, lam=0.1, kind="molecules", folds=3, seed=0)
    _m, sd = low.predict_with_sd(test)
    path = str(tmp_path / "low.cmf")
    save_model(path, low)
    back, _extra = load_model(path)
    assert back.is_low_rank
    _m2, sd2 = back.predict_with_sd(test)
    assert np.abs(sd2 - sd).max() < EXACT


def test_a_model_without_its_gram_says_what_to_do():
    fs, table, y = _series()
    low = CMFModel(fieldset=fs, method="krr", normalize=True)
    low.fit_low_rank(table, y, table, lam=0.1, kind="molecules", folds=3, seed=0)
    low.low_gram_ = None
    low._surrogate = None
    with pytest.raises(RuntimeError, match="fit_low_rank"):
        low.surrogate()


# ------------------------------------------------- fewer landmarks: the approximate model
def test_with_fewer_landmarks_the_deviation_keeps_what_a_deviation_must():
    fs, table, y = _series()
    train, test, y_tr, _y_te, _tr, _te = _split(table, y)
    Z, _rows = LR.landmark_molecules(train, 5, seed=1)
    low = CMFModel(fieldset=fs, method="krr", normalize=True)
    low.fit_low_rank(train, y_tr, Z, lam=0.1, kind="molecules", folds=3, seed=0)
    s = low.surrogate()
    assert Z.n_mol < train.n_mol                     # fewer landmarks than molecules
    assert s.rank == low.low_rank_.rank              # one column per landmark and field
    assert s.scale > 0 and np.isfinite(s.scale)
    m, sd_new = low.predict_with_sd(test)
    _m2, sd_under = low.predict_with_sd(test, with_noise=False)
    assert np.all(sd_new > 0) and np.all(sd_under >= 0)
    assert np.all(sd_new > sd_under)                 # a new measurement carries the noise
    assert np.abs(m - low.predict(test)).max() < EXACT
    # the exact self-kernel is what keeps the deviation up where the landmarks span nothing: the
    # plain subset-of-regressors variance, which drops it, is never larger
    Phi = low.low_rank_.transform(test)
    lam = 0.1
    plain = LR.variance(low.low_gram_, Phi, lam)
    kept = LR.variance(low.low_gram_, Phi, lam, k_self=low._low_rank_self(test))
    assert np.all(kept >= plain - 1e-12)
    cal = low.calibration(folds=3, seed=0)
    assert np.isfinite(cal.z_rms) and cal.n == train.n_mol


def test_the_joint_self_kernel_is_the_models_own():
    fs, table, y = _series()
    low = CMFModel(fieldset=fs, method="krr", normalize=True)
    low.fit_low_rank(table, y, table, lam=0.1, kind="molecules", folds=3, seed=0)
    assert np.abs(low.low_rank_.joint_self(table) - low._low_rank_self(table)).max() < 1e-12
    # under the cosine normalisation every self-kernel is the sum of the field weights
    want = float(np.sum(np.asarray(low.low_rank_.weights, float)))
    assert np.abs(low.low_rank_.joint_self(table) - want).max() < 1e-12


def test_what_the_reading_refuses():
    fs, table, y = _series()
    low = LR.build(table, fs, scale=np.ones(2))
    g = LR.gram(low, table, y)
    reading = LowRankSurrogate.fit(g, 0.1)
    with pytest.raises(ValueError, match="features"):
        reading.predict(np.zeros((2, reading.rank + 1)), [1.0, 1.0])
    with pytest.raises(ValueError, match="self-kernels"):
        reading.predict(np.zeros((2, reading.rank)), [1.0])
    one = LR.gram(low, table.subset([0]), y[:1])
    with pytest.raises(ValueError, match="two activities"):
        LowRankSurrogate.fit(one, 0.1)


# ------------------------------------------------- the command line
def test_build_gives_a_low_rank_model_its_calibration(tmp_path, capsys):
    """`cmf build --low-rank ... --calibration`: two options that could not be combined before."""
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    rng = np.random.default_rng(0)
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(SMILES):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(m, p)
            m.SetProp("_Name", f"m{k}")
            m.SetProp("activity", f"{4.0 + 0.3 * k + rng.normal(scale=0.2):.3f}")
            w.write(m)
    capsys.readouterr()
    main(["build", path, "--activity", "activity", "--fields", "electrostatic,steric",
          "--low-rank", "molecules:6", "--alphas", "0.5", "--lams", "0.1",
          "--test-fraction", "0.25", "--folds", "3", "--calibration"])
    text = capsys.readouterr().out
    assert "Low rank:" in text
    assert "Predictive standard deviation, on the folds: calibration on" in text
    assert "on the test set: calibration on" in text


# ------------------------------------------------- a field that is zero for some molecules
def test_a_field_that_is_zero_for_some_molecules_keeps_the_two_paths_one_model():
    """Esters and a nitrile have no hydrogen-bond donor, so the donor field is identically zero
    for them. The exact path then scales that field by more than one (the mean of its normalised
    diagonal is below one) and gives those molecules a self-kernel of zero for it. Item MS6 found
    the low-rank path doing neither -- a 1 % different kernel on the ACE set -- and this fixes
    both: the fit, the deviation and the calibration must be the exact ones to round-off."""
    smiles = SMILES[:6] + ["c1ccccc1C(=O)OC", "c1ccc(Cl)cc1C(=O)OC", "c1ccccc1C#N",
                           "c1ccc(F)cc1C(=O)OC", "c1ccc(C)cc1C#N", "c1ccccc1OC"]
    fs = FieldSet.create(physchem=["electrostatic", "hbd"], alpha=0.5)
    mols = []
    for k, smi in enumerate(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        AllChem.EmbedMolecule(m, p)
        mols.append(build_fields(m, fs, f"m{k}"))
    table = AtomTable.from_molecules(mols)
    rng = np.random.default_rng(1)
    y = 4.0 + 0.3 * np.arange(len(mols)) + rng.normal(scale=0.2, size=len(mols))
    assert any(np.abs(mf.weights[:, 1]).sum() == 0 for mf in mols[6:])   # no donor at all
    tr = np.array([0, 1, 2, 3, 6, 7, 8, 9, 10])
    te = np.array([4, 5, 11])                       # a donor, a donor, and a molecule without
    train, test = table.subset(tr), table.subset(te)
    exact = CMFModel(fieldset=fs, method="krr", normalize=True)
    exact.params["lam"] = 0.1
    exact.fit(train, y[tr], cv_folds=3, cv_seed=0)
    assert exact.scale[1] > 1.0                      # the donor field is scaled up
    low = CMFModel(fieldset=fs, method="krr", normalize=True)
    low.fit_low_rank(train, y[tr], train, lam=0.1, kind="molecules", folds=3, seed=0)
    assert np.allclose(low.scale, exact.scale)
    Phi = low.low_rank_.transform(train)
    assert np.abs(Phi @ Phi.T - np.asarray(exact._linear_train_kernel(), float)).max() < 1e-8
    assert np.abs(low.predict(test) - exact.predict(test)).max() < EXACT
    assert np.abs(low._low_rank_self(test) - exact.joint_self(test)).max() < 1e-10
    m_e, s_e = exact.predict_with_sd(test)
    m_l, s_l = low.predict_with_sd(test)
    assert np.abs(m_l - m_e).max() < EXACT and np.abs(s_l - s_e).max() < EXACT
    assert low.calibration(folds=3, seed=0).z_rms == pytest.approx(
        exact.calibration(folds=3, seed=0).z_rms, rel=1e-6)
