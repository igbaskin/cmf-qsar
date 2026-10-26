"""Conformal intervals: the guarantee, the refusals, and the normalised form."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.conformal import (Conformal, calibrate, calibrate_cv,  # noqa: E402
                               coverage, difficulty_from_kernel)


def test_the_quantile_is_the_order_statistic_with_the_plus_one():
    c = Conformal(level=0.9, residuals=np.arange(1.0, 101.0))
    assert c.n_calibration == 100
    assert c.rank == 91                       # ceil(101 * 0.9)
    assert c.quantile == 91.0


def test_too_few_errors_refuse_the_level_rather_than_guessing():
    """A rule that cannot reach its level says so; it does not quietly return its widest error."""
    c = Conformal(level=0.95, residuals=np.arange(1.0, 19.0))
    assert not c.usable and c.quantile == float("inf")
    assert c.min_calibration() == 19
    assert Conformal(level=0.95, residuals=np.arange(1.0, 20.0)).usable
    assert "not reachable" in c.describe()


@pytest.mark.parametrize("level", [0.5, 0.8, 0.9, 0.95, 0.99])
def test_the_minimum_calibration_size_is_exact(level):
    n = Conformal(level=level, residuals=np.zeros(1)).min_calibration()
    assert Conformal(level=level, residuals=np.arange(float(n))).usable
    assert not Conformal(level=level, residuals=np.arange(float(n - 1))).usable


@pytest.mark.parametrize("level", [0.8, 0.9, 0.95])
def test_coverage_holds_on_exchangeable_data(level):
    """The guarantee, measured: between the level and the level plus 1/(n+1)."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=4000)
    yp = y + rng.normal(scale=0.7, size=4000)
    cal = calibrate(y[:2000], yp[:2000], level=level)
    lo, hi = cal.interval(yp[2000:])
    got = coverage(y[2000:], lo, hi)
    assert level - 0.02 <= got["coverage"] <= level + 1.0 / 2001 + 0.02


def test_coverage_survives_a_badly_specified_model():
    """A wrong model keeps its coverage at the cost of wider intervals, which is the purpose."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=4000)
    good = y + rng.normal(scale=0.3, size=4000)
    bad = 0.1 * y + rng.normal(scale=2.0, size=4000)
    out = {}
    for name, yp in (("good", good), ("bad", bad)):
        cal = calibrate(y[:2000], yp[:2000], level=0.9)
        lo, hi = cal.interval(yp[2000:])
        out[name] = coverage(y[2000:], lo, hi)
    assert out["good"]["coverage"] > 0.87 and out["bad"]["coverage"] > 0.87
    assert out["bad"]["mean_width"] > 2.5 * out["good"]["mean_width"]


def test_the_normalised_rule_widens_where_the_model_is_extrapolating():
    rng = np.random.default_rng(2)
    hard = rng.random(2000)                       # difficulty in [0, 1)
    y = rng.normal(size=2000)
    yp = y + rng.normal(scale=0.2 + 2.0 * hard)   # error grows with difficulty
    cal = calibrate(y, yp, level=0.9, difficulty=hard, beta=0.2)
    near, far = cal.half_width([0.0]), cal.half_width([1.0])
    assert far[0] > 3.0 * near[0]
    lo, hi = cal.interval(yp, hard)
    assert coverage(y, lo, hi)["coverage"] >= 0.89
    # and the plain rule, on the same data, is one width everywhere
    flat = calibrate(y, yp, level=0.9)
    assert np.ndim(flat.half_width()) == 0


def test_the_normalised_rule_is_narrower_where_it_can_be():
    """Normalisation should gain something: tighter intervals for the easy molecules."""
    rng = np.random.default_rng(3)
    hard = rng.random(3000)
    y = rng.normal(size=3000)
    yp = y + rng.normal(scale=0.1 + 3.0 * hard)
    norm = calibrate(y[:1500], yp[:1500], level=0.9, difficulty=hard[:1500], beta=0.1)
    flat = calibrate(y[:1500], yp[:1500], level=0.9)
    easy = hard[1500:] < 0.2
    wn = np.diff(norm.interval(yp[1500:], hard[1500:]), axis=0)[0][easy].mean()
    wf = np.diff(flat.interval(yp[1500:]), axis=0)[0].mean()
    assert wn < 0.5 * wf


def test_calibrate_cv_uses_out_of_fold_predictions():
    """Calibrating on the fit would give an interval that covers the training set and nothing else."""
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 4))
    K = X @ X.T
    y = X[:, 0] * 2.0 + rng.normal(scale=0.5, size=60)
    cal = calibrate_cv(K, y, "krr", {"lam": 0.1}, level=0.9, folds=10, seed=0)
    assert cal.kind == "cv" and cal.n_calibration == 60 and cal.usable
    from cmfqsar.model import krr_fit
    a, b = krr_fit(K, y, 0.1, True)
    fitted = np.abs(y - (K @ a + b))
    assert cal.scores.mean() > fitted.mean()          # out of fold is harder than the fit


def test_difficulty_is_zero_for_a_training_molecule_and_grows_outside():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 3))
    K = X @ X.T
    far = (X[:3] + 25.0) @ X.T
    k_far = np.sum((X[:3] + 25.0) ** 2, axis=1)
    d_self = difficulty_from_kernel(K, np.diag(K), K)
    d_far = difficulty_from_kernel(far, k_far, K)
    assert d_self.mean() < d_far.mean()
    assert np.all(d_self >= 0.0) and np.all(d_far <= 2.0)


def test_coverage_reports_what_it_was_given():
    got = coverage([1.0, 2.0, 3.0], [0.0, 2.5, 2.9], [2.0, 3.5, 3.1])
    assert got["n"] == 3 and got["coverage"] == pytest.approx(2 / 3)
    assert got["mean_width"] == pytest.approx((2.0 + 1.0 + 0.2) / 3)


def test_a_level_outside_the_unit_interval_is_refused():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            Conformal(level=bad, residuals=np.ones(10))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------- #
# The model's own interval
# --------------------------------------------------------------------------- #

def _small_set(n=24, seed=11):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import AtomTable, FieldSet, build_fields

    subs = ["C", "CC", "CCC", "O", "N", "Cl", "F", "Br", "C=O", "C#N", "OC", "NC"]
    smis = [f"c1ccccc1{s}" for s in subs] + [f"c1ccc(C)cc1{s}" for s in subs]
    rng = np.random.default_rng(seed)
    mols, names = [], []
    for i, smi in enumerate(smis[:n]):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=seed + i)
        Chem.SanitizeMol(m)
        mols.append(m)
        names.append(f"m{i}")
    fs = FieldSet.create(alpha=0.6)
    mf = [build_fields(m, fs, nm) for m, nm in zip(mols, names)]
    y = np.array([m.GetNumHeavyAtoms() for m in mols], float) + rng.normal(scale=0.4, size=len(mols))
    return fs, mf, y, AtomTable


def test_the_model_calibrates_from_the_cross_validation_it_already_ran():
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=8)
    c = model.calibrate_conformal(level=0.9)
    assert c.n_calibration == len(y) and c.kind == "cv"
    # exactly the errors of the stored out-of-fold predictions, nothing recomputed
    assert np.allclose(np.sort(c.scores),
                       np.sort(np.abs(y - model.stats.extra["y_cv"])))


def test_predict_interval_brackets_the_prediction_and_survives_a_round_trip(tmp_path):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf[:18]), y[:18], cv_folds=6)
    model.calibrate_conformal(level=0.9)
    table = AtomTable.from_molecules(mf[18:])
    yp, lo, hi = model.predict_interval(table)
    assert np.all(lo <= yp) and np.all(yp <= hi)
    assert np.allclose(hi - lo, (hi - lo)[0])          # one width, the rule is not normalised

    path = str(tmp_path / "m.cmf")
    save_model(path, model)
    back = load_model(path)
    back = back[0] if isinstance(back, tuple) else back
    yp2, lo2, hi2 = back.predict_interval(table)
    assert np.allclose(lo, lo2) and np.allclose(hi, hi2)


def test_the_normalised_model_interval_varies_by_molecule():
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf[:18]), y[:18], cv_folds=6)
    model.calibrate_conformal(level=0.9, normalise=True, beta=0.1)
    _yp, lo, hi = model.predict_interval(AtomTable.from_molecules(mf[18:]))
    assert np.std(hi - lo) > 0.0


def test_a_model_without_a_calibration_refuses_to_invent_one():
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=6)
    with pytest.raises(ValueError):
        model.predict_interval(AtomTable.from_molecules(mf[:3]))


def test_a_classifier_is_refused_rather_than_given_a_meaningless_interval():
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    model = CMFModel(fieldset=fs, method="svc", task="classification", normalize=True)
    model.fit(AtomTable.from_molecules(mf), (y > np.median(y)).astype(float), cv_folds=4)
    with pytest.raises(ValueError):
        model.calibrate_conformal(level=0.9)
