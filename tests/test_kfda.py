"""Kernel Fisher discriminant analysis (method "kfda", plan item E8).

The learner solves a generalised eigenvalue problem in the kernel, its score agrees with kernel
ridge regression on class-size weighted labels as the regularisation goes to zero, its threshold
and its probabilities are fitted out of fold, and -- the reason the method exists -- a fitted
classifier has a co-field that is a linear map of the class labels, which the support-vector
classifier's kernel expansion is not (run with: python -m pytest tests/test_kfda.py).
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.gpc import brier_score  # noqa: E402
from cmfqsar.model import (KFDA_FOLDS, CMFModel, KFDACalibration, _balanced_threshold,  # noqa: E402
                           _Learner, _logistic_1d, _param_grid, classification_scores,
                           cv_predict, grid_search, kfda_calibrate, kfda_fit, krr_fit)
from cmfqsar.transfer import cofield_overlap  # noqa: E402


def _problem(n=80, d=6, seed=0, noise=0.3):
    """A two-class problem in d dimensions and its linear kernel, scaled to unit mean diagonal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = np.where(X[:, 0] + 0.5 * X[:, 1] + noise * rng.randn(n) > 0, 1.0, -1.0)
    K = X @ X.T
    return K / float(np.mean(np.diag(K))), y, X


def _scatter(K, y):
    """The within-class scatter N = K W K and the class-mean difference m_+ - m_-."""
    K = np.asarray(K, float)
    pos = y > 0
    n_p, n_n = int(pos.sum()), int((~pos).sum())
    A = K.copy()
    A[:, pos] -= K[:, pos].mean(axis=1)[:, None]
    A[:, ~pos] -= K[:, ~pos].mean(axis=1)[:, None]
    return A @ A.T, K @ np.where(pos, 1.0 / n_p, -1.0 / n_n)


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


# --------------------------------------------------------------------------------------------
# the solve itself
# --------------------------------------------------------------------------------------------

def test_the_solve_is_the_direction_that_maximises_the_fisher_quotient():
    """a maximises (m_+ - m_-)'a squared over a'(N + ridge I)a, which is what KFDA is."""
    K, y, _ = _problem()
    n = len(y)
    lam = 0.1
    a, b = kfda_fit(K, y, lam)
    N, M = _scatter(K, y)
    S = N + lam * (np.trace(N) / n) * np.eye(n)

    def quotient(v):
        return float((M @ v) ** 2 / (v @ S @ v))

    best = quotient(a)
    rng = np.random.RandomState(1)
    # nothing in a neighbourhood of the solution beats it, and no random direction comes close
    for _ in range(200):
        step = 0.02 * np.linalg.norm(a) * rng.randn(n)
        assert quotient(a + step) <= best * (1.0 + 1e-9)
    for _ in range(200):
        assert quotient(rng.randn(n)) <= best


def test_the_solve_is_the_top_generalised_eigenvector():
    """(m_+ m_-' difference outer product) a = mu (N + ridge I) a, the stated eigenproblem."""
    sla = pytest.importorskip("scipy.linalg")
    K, y, _ = _problem(n=50, seed=2)
    n = len(y)
    lam = 0.05
    a, _ = kfda_fit(K, y, lam)
    N, M = _scatter(K, y)
    S = N + lam * (np.trace(N) / n) * np.eye(n)
    _, V = sla.eigh(np.outer(M, M), S)
    top = V[:, -1]
    cos = abs(float(top @ a)) / np.linalg.norm(top) / np.linalg.norm(a)
    assert cos > 1.0 - 1e-8


def test_the_score_is_in_within_class_standard_deviations_and_points_at_the_actives():
    K, y, _ = _problem()
    n = len(y)
    a, b = kfda_fit(K, y, 0.1)
    N, _ = _scatter(K, y)
    assert abs(float(a @ N @ a) / (n - 2) - 1.0) < 1e-8
    s = K @ a
    pos = y > 0
    assert s[pos].mean() > s[~pos].mean()            # a higher score means more active
    # the bias puts the decision at the midpoint of the two class means
    assert abs(0.5 * (s[pos].mean() + s[~pos].mean()) + b) < 1e-8


def test_the_score_agrees_with_kernel_ridge_regression_on_weighted_labels():
    """The equivalence the plan entry states: with the ridge going to zero the discriminant is
    kernel ridge regression on labels n/n_+ and -n/n_-, up to the length of the direction."""
    for n, d, seed in ((60, 8, 0), (40, 60, 1), (30, 120, 2)):
        K, y, _ = _problem(n=n, d=d, seed=seed)
        pos = y > 0
        n_p, n_n = int(pos.sum()), int((~pos).sum())
        target = np.where(pos, n / n_p, -n / n_n)
        a, _ = kfda_fit(K, y, 1e-8)
        a_krr, _ = krr_fit(K, target, 1e-8)
        s, s_krr = K @ a, K @ a_krr
        # the two scores differ by a positive factor and a constant, so they correlate exactly
        assert np.corrcoef(s, s_krr)[0, 1] > 1.0 - 1e-6


def test_the_ridge_is_relative_so_the_same_lambda_means_the_same_fit():
    """Multiplying the kernel by a constant rescales the problem and nothing else: the ordering
    of the molecules by score is unchanged, which a ridge in absolute units would not keep."""
    K, y, _ = _problem()
    a1, b1 = kfda_fit(K, y, 0.3)
    a2, b2 = kfda_fit(7.5 * K, y, 0.3)
    assert np.corrcoef(K @ a1, 7.5 * K @ a2)[0, 1] > 1.0 - 1e-10
    assert np.allclose(np.sign(K @ a1 + b1), np.sign(7.5 * K @ a2 + b2))


def test_one_class_alone_is_refused():
    K, y, _ = _problem()
    with pytest.raises(ValueError, match="both classes"):
        kfda_fit(K, np.ones_like(y), 0.1)


# --------------------------------------------------------------------------------------------
# the threshold and the probabilities
# --------------------------------------------------------------------------------------------

def test_the_balanced_threshold_is_the_best_cut_and_ignores_the_class_sizes():
    y = np.array([1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    score = np.array([3.0, 2.5, 2.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    cut, bal = _balanced_threshold(y, score)
    assert 2.0 < cut < 2.5 and abs(bal - 1.0) < 1e-12
    # a cut that follows the larger class would call everything inactive and score 0.5
    assert classification_scores(y, score - cut)["balanced_accuracy"] == pytest.approx(1.0)
    # every cut is tried, so no cut can be better than the one returned
    for c in np.linspace(score.min() - 1, score.max() + 1, 200):
        assert classification_scores(y, score - c)["balanced_accuracy"] <= bal + 1e-12


def test_the_balanced_threshold_does_not_split_equal_scores():
    y = np.array([1.0, 1.0, -1.0, -1.0])
    score = np.array([1.0, 0.0, 0.0, -1.0])        # one active and one inactive share a score
    cut, bal = _balanced_threshold(y, score)
    assert cut not in (0.0,)
    assert classification_scores(y, score - cut)["balanced_accuracy"] == pytest.approx(bal)


def test_the_logistic_map_is_increasing_calibrated_and_finite_when_separated():
    rng = np.random.RandomState(0)
    s = rng.randn(400)
    p_true = 1.0 / (1.0 + np.exp(-(1.5 * s - 0.4)))
    y = np.where(rng.rand(400) < p_true, 1.0, -1.0)
    slope, intercept = _logistic_1d(y, s)
    assert slope > 0
    assert abs(slope - 1.5) < 0.45 and abs(intercept + 0.4) < 0.45
    # a completely separated score gives a finite slope, because the targets are smoothed
    y2 = np.where(s > 0, 1.0, -1.0)
    slope2, _ = _logistic_1d(y2, s)
    assert np.isfinite(slope2) and slope2 > 0


def test_the_calibration_is_fitted_out_of_fold_and_turns_scores_into_probabilities():
    K, y, _ = _problem(n=100, seed=3)
    cal = kfda_calibrate(K, y, 0.1, folds=5, seed=0)
    assert isinstance(cal, KFDACalibration)
    assert cal.folds == 5 and cal.scores is not None and np.all(np.isfinite(cal.scores))
    assert 0.5 < cal.balanced_accuracy <= 1.0
    # the reported accuracy is the one the chosen cut reaches on those out-of-fold scores
    assert classification_scores(y, cal.scores - cal.threshold)["balanced_accuracy"] == \
        pytest.approx(cal.balanced_accuracy)
    # probability() takes a raw score, so the scaled scores are mapped back first
    p = cal.probability(cal.centre + cal.scores * cal.spread)
    assert np.all((p >= 0.0) & (p <= 1.0))
    order = np.argsort(cal.scores)                         # increasing in the score
    assert np.all(np.diff(p[order]) >= -1e-12)
    assert brier_score(y, p) < 0.25                        # better than no better than chance
    # the bias turns the cut back into a boundary on raw scores, and nothing else moves
    a, _ = kfda_fit(K, y, 0.1)
    assert cal.scale(-cal.bias()) == pytest.approx(cal.threshold)
    # too few molecules to split: no calibration rather than a meaningless one
    assert kfda_calibrate(K[:3, :3], y[:3], 0.1, folds=5).folds == 0


# --------------------------------------------------------------------------------------------
# the learner wrapper, the grid and cross-validation
# --------------------------------------------------------------------------------------------

def test_the_learner_wrapper_uses_the_calibrated_threshold():
    K, y, X = _problem()
    lr = _Learner("kfda", {"lam": 0.1}).fit(K, y)
    a, b_mid = kfda_fit(K, y, 0.1)
    assert np.allclose(lr.alpha_, a)
    assert lr.kfda_.folds == KFDA_FOLDS
    assert lr.b_ == pytest.approx(lr.kfda_.bias())
    # the bias puts the chosen cut at zero, in the class-mean units the cut is kept in
    assert lr.kfda_.scale(-lr.b_) == pytest.approx(lr.kfda_.threshold)
    assert lr.support_.shape == y.shape
    # asking for the plain Fisher threshold gives the midpoint and no calibration
    plain = _Learner("kfda", {"lam": 0.1, "kfda_threshold": "midpoint"}).fit(K, y)
    assert plain.b_ == pytest.approx(b_mid) and getattr(plain, "kfda_", None) is None
    # the prediction is K a + b, the kernel expansion every other learner here returns
    Kt = K[:5]
    assert np.allclose(lr.predict(Kt), Kt @ lr.alpha_ + lr.b_)


def test_the_parameter_grid_is_the_lambda_grid_and_cross_validation_runs():
    K, y, _ = _problem()
    assert _param_grid("kfda", (0.1, 1.0), (), (), ()) == [{"lam": 0.1}, {"lam": 1.0}]
    ycv = cv_predict(K, y, "kfda", {"lam": 0.1, "kfda_threshold": "midpoint"}, folds=5, seed=0)
    assert ycv.shape == y.shape and np.all(np.isfinite(ycv))
    sc = classification_scores(y, ycv)
    assert sc["auc"] > 0.8 and sc["balanced_accuracy"] > 0.7


# --------------------------------------------------------------------------------------------
# on fields: the co-field, which is why the method is here
# --------------------------------------------------------------------------------------------

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "c1ccccc1OC",
          "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C", "c1ccc(Br)cc1O",
          "c1ccc(N)cc1C", "c1ccc(F)cc1N", "c1ccc(Cl)cc1C"]
LABELS = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def _field_model(method, params, alpha=0.7):
    mols = [_mol(s, i) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=("electrostatic", "steric", "hydrophobic"), alpha=alpha)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, task="classification", method=method, params=params,
                     normalize=True)
    model.fit(table, LABELS, cv_folds=4)
    return model, table, mf


def test_a_fitted_discriminant_has_a_cofield_and_the_decomposition_rules_hold():
    model, table, mf = _field_model("kfda", {"lam": 0.1})
    assert model.stats.task == "classification"
    assert np.isfinite(model.stats.fit["balanced_accuracy"])
    assert np.isfinite(model.stats.cv["balanced_accuracy"])
    # the co-field exists and is a sum over the training molecules' fields
    for f in range(len(model.fieldset.specs)):
        centres, coeffs, widths = model.cofield_coefficients(f)
        assert len(centres) == len(coeffs) == len(widths)
        assert np.any(np.abs(coeffs) > 0)
    # the score decomposes over atoms and over training molecules, as for any kernel expansion
    for mol in mf[:3]:
        s = model.predict(AtomTable.from_molecules([mol]))[0]
        assert abs(model.atom_contributions(mol).sum() + model.bias - s) < 1e-8
        assert abs(model.training_contributions(mol).sum() + model.bias - s) < 1e-8


def test_the_discriminant_cofield_resembles_the_support_vector_classifier_s():
    """The comparison the plan item's gate asks for: the same molecules, the same fields, two
    classifiers, their co-fields compared by the cosine of plan item F3."""
    a, _, _ = _field_model("kfda", {"lam": 0.1})
    b, _, _ = _field_model("svc", {"C": 1.0})
    ov = cofield_overlap(a, b, "kfda", "svc")
    assert np.isfinite(ov.cosine)
    assert ov.cosine > 0.0           # both classifiers ask the chemistry for the same thing
    assert len(ov.fields) == len(a.fieldset.specs)


def test_probabilities_come_out_of_the_model_and_survive_a_save():
    import pickle

    model, table, mf = _field_model("kfda", {"lam": 0.1})
    p = model.predict_proba(table)
    assert p.shape == LABELS.shape and np.all((p >= 0.0) & (p <= 1.0))
    # the probability rises with the decision value, molecule for molecule; p = 0.5 is not the
    # same cut as the reported class, because the two answer different questions (see
    # CMFModel.predict_proba)
    d = model.predict(table)
    order = np.argsort(d)
    assert np.all(np.diff(p[order]) >= -1e-12)
    back = CMFModel.from_state(pickle.loads(pickle.dumps(model.state())))
    assert back.method == "kfda"
    assert np.allclose(back.predict_proba(table), p)
    assert np.allclose(back.predict(table), model.predict(table))


def test_the_support_vector_classifier_still_has_no_probabilities():
    model, table, _ = _field_model("svc", {"C": 1.0})
    with pytest.raises(ValueError, match="decision value"):
        model.predict_proba(table)


def test_the_benchmark_computes_the_same_auc_and_balanced_accuracy_as_the_library():
    """`benchmarks/run_kfda.py` writes out its own AUC and balanced accuracy, because its paired
    bootstrap asks for one number two thousand times and `classification_scores` computes all of
    them. The two must agree, ties in the decision values included."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import run_kfda as RK

    rng = np.random.RandomState(0)
    for trial in range(30):
        n = int(rng.randint(20, 200))
        y = np.where(rng.rand(n) < rng.uniform(0.05, 0.5), 1.0, -1.0)
        if not ((y > 0).any() and (y < 0).any()):
            continue
        d = rng.randn(n)
        if trial % 3 == 0:
            d = np.round(d, 1)                 # many equal decision values
        ref = classification_scores(y, d)
        assert RK._auc(y, d) == pytest.approx(ref["auc"], abs=1e-10)
        assert RK._balanced_accuracy(y, d) == pytest.approx(ref["balanced_accuracy"], abs=1e-12)


def test_the_benchmark_folds_keep_the_share_of_actives():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import run_kfda as RK

    y = np.array([1.0] * 40 + [-1.0] * 360)
    folds = RK.stratified_folds(y, folds=5, seed=0)
    assert sorted(np.concatenate(folds)) == list(range(len(y)))
    for f in folds:
        assert int(np.sum(y[f] > 0)) == 8      # 40 actives over five folds
        assert len(f) == 80


def test_the_lambda_of_the_discriminant_is_chosen_by_the_grid_search():
    mols = [_mol(s, i) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=("electrostatic", "steric"), alpha=0.7)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    model = CMFModel(fieldset=fs, task="classification", method="kfda", params={"lam": 1.0},
                     normalize=True)
    res = grid_search(table, LABELS, model, alphas=[0.7], lams=(0.01, 0.1, 1.0), folds=4, seed=0)
    assert set(res.best_params) == {"lam"}
    assert res.best_params["lam"] in (0.01, 0.1, 1.0)
    assert len(res.table) == 3
