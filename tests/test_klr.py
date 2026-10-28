"""Kernel logistic regression (method "klr", plan item E9).

The learner is the Laplace step of `cmfqsar/gpc.py` read as a learner of its own, so the tests
that matter are the ones that check it really solves the problem it claims to: the objective is
at its minimum, the solution agrees with logistic regression on explicit features when the kernel
is linear, it is the Gaussian-process mode at the matching amplitude, and the probability is the
likelihood at the mode with no variance averaged into it
(run with: python -m pytest tests/test_klr.py).
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.gpc import LaplaceGPC, brier_score, klr_fit  # noqa: E402
from cmfqsar.model import (CMFModel, _Learner, _param_grid, classification_scores,  # noqa: E402
                           cv_predict, grid_search, klr_threshold, out_of_fold_scores)


def _problem(n=80, d=6, seed=0, noise=0.3):
    """A two-class problem in d dimensions and its linear kernel, scaled to unit mean diagonal."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = np.where(X[:, 0] + 0.5 * X[:, 1] + noise * rng.randn(n) > 0, 1.0, -1.0)
    K = X @ X.T
    return K / float(np.mean(np.diag(K))), y, X


def _objective(K, y, a, lam):
    """The function kernel logistic regression minimises: logistic loss + lam/2 times ||f||^2."""
    f = K @ a
    return float(np.sum(np.logaddexp(0.0, -y * f)) + 0.5 * lam * float(a @ K @ a))


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


# --------------------------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.03, 0.3, 3.0])
def test_the_solution_minimises_the_objective_it_claims_to(lam):
    """Nothing in a neighbourhood of the solution, and no random direction, does better on
    logistic loss plus the ridge."""
    K, y, _ = _problem()
    n = len(y)
    a, b, _ = klr_fit(K, y, lam)
    assert b == 0.0
    best = _objective(K, y, a, lam)
    rng = np.random.RandomState(1)
    scale = float(np.linalg.norm(a))
    for _ in range(300):
        step = 0.02 * scale * rng.randn(n)
        assert _objective(K, y, a + step, lam) >= best - 1e-9
    for _ in range(100):
        assert _objective(K, y, 0.1 * scale * rng.randn(n), lam) >= best


@pytest.mark.parametrize("lam", [0.03, 0.3, 3.0])
def test_the_gradient_of_the_objective_vanishes_at_the_solution(lam):
    """The stationarity condition is K (lam a - (t - pi)) = 0, which with a positive definite
    kernel means lam a = t - pi: the coefficients are the residuals of the probabilities."""
    K, y, _ = _problem()
    a, _, _ = klr_fit(K, y, lam)
    f = K @ a
    pi = 1.0 / (1.0 + np.exp(-f))
    t = 0.5 * (y + 1.0)
    grad = K @ (lam * a - (t - pi))
    assert float(np.max(np.abs(grad))) < 1e-7 * max(1.0, float(np.max(np.abs(f))))
    assert np.allclose(lam * a, t - pi, atol=1e-7)


def test_it_is_logistic_regression_on_the_explicit_features():
    """On the linear kernel K = X X', the latent function is the linear logistic regression with
    the matching ridge, which is an independent implementation of the same problem."""
    LogisticRegression = pytest.importorskip("sklearn.linear_model").LogisticRegression
    for lam in (0.1, 1.0):
        K, y, X = _problem(n=60, d=5, seed=2)
        X = X / np.sqrt(float(np.mean(np.diag(X @ X.T))))     # the same scaling the kernel got
        a, _, _ = klr_fit(K, y, lam)
        # sklearn minimises C * sum(loss) + ||w||^2 / 2, so C = 1 / lam
        ref = LogisticRegression(C=1.0 / lam, fit_intercept=False, tol=1e-12,
                                 max_iter=5000).fit(X, y)
        assert np.allclose(K @ a, X @ ref.coef_.ravel(), atol=1e-5)


def test_it_is_the_gaussian_process_mode_at_the_matching_amplitude():
    """The entry's own claim: this is the Laplace step of the GP classifier with the logistic
    likelihood, at amplitude 1 / lam, and nothing else."""
    K, y, _ = _problem()
    for lam in (0.05, 0.5, 5.0):
        a, _, fit = klr_fit(K, y, lam)
        g = LaplaceGPC(likelihood="logit", amplitude=1.0 / lam).fit(K, y)
        assert np.allclose(a, g.coef, atol=1e-10)
        assert np.allclose(fit.f_, g.f_, atol=1e-10)
        # the probit likelihood is a different problem, so it must not agree
        other = LaplaceGPC(likelihood="probit", amplitude=1.0 / lam).fit(K, y)
        assert not np.allclose(a, other.coef, atol=1e-3)


def test_the_fit_skips_what_it_does_not_need_and_says_so():
    """The saving the plan entry claims: no factorisation for the posterior variance, and no
    marginal likelihood. Both are refused rather than returned wrong."""
    K, y, _ = _problem()
    _, _, fit = klr_fit(K, y, 0.1)
    assert fit.L_ is None
    assert not np.isfinite(fit.log_marginal_likelihood)
    with pytest.raises(RuntimeError, match="evidence=False"):
        fit.latent_variance(K, np.diag(K))
    # the Gaussian process with the same amplitude does provide both
    g = LaplaceGPC(likelihood="logit", amplitude=10.0).fit(K, y)
    assert g.L_ is not None and np.isfinite(g.log_marginal_likelihood)
    assert np.all(g.latent_variance(K, np.diag(K)) >= 0.0)


def test_a_non_positive_ridge_is_refused_and_so_is_one_class():
    K, y, _ = _problem()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="positive lam"):
            klr_fit(K, y, bad)
    with pytest.raises(ValueError, match="both classes"):
        klr_fit(K, np.ones_like(y), 0.1)


def test_the_probability_is_the_likelihood_at_the_mode_with_no_variance_in_it():
    K, y, _ = _problem()
    a, _, fit = klr_fit(K, y, 0.1)
    p = fit.predict_proba(K)
    assert np.allclose(p, 1.0 / (1.0 + np.exp(-(K @ a))), atol=1e-12)
    # p = 0.5 is exactly the decision boundary, which is what "probabilistic classifier" means
    assert np.all((p >= 0.5) == (K @ a >= 0.0))
    assert brier_score(y, p) < 0.25
    # the Gaussian process pulls its probability towards one half, and this does not
    g = LaplaceGPC(likelihood="logit", amplitude=10.0).fit(K, y)
    p_gp = g.predict_proba(K, np.diag(K))
    assert np.all(np.abs(p_gp - 0.5) <= np.abs(p - 0.5) + 1e-12)
    assert float(np.max(np.abs(p_gp - p))) > 1e-3


# --------------------------------------------------------------------------------------------
# the learner wrapper, the grid, the threshold
# --------------------------------------------------------------------------------------------

def test_the_learner_wrapper_leaves_the_boundary_at_one_half():
    K, y, _ = _problem()
    lr = _Learner("klr", {"lam": 0.1}).fit(K, y)
    a, b, _ = klr_fit(K, y, 0.1)
    assert np.allclose(lr.alpha_, a) and lr.b_ == 0.0
    assert lr.klr_ is not None and lr.klr_.converged
    assert lr.support_.shape == y.shape
    assert np.allclose(lr.predict(K[:5]), K[:5] @ a)
    # with the threshold rule asked for, the bias moves and the out-of-fold accuracy is recorded
    tuned = _Learner("klr", {"lam": 0.1, "klr_threshold": "cv"}).fit(K, y)
    assert np.allclose(tuned.alpha_, a)
    assert np.isfinite(tuned.klr_balanced_) and 0.5 <= tuned.klr_balanced_ <= 1.0
    assert getattr(_Learner("klr", {"lam": 0.1}).fit(K, y), "klr_balanced_", None) is None


def test_the_threshold_rule_never_scores_worse_out_of_fold_than_one_half():
    """A set with one active in twenty: the boundary at p = 0.5 follows the larger class and the
    rule has to beat it, since 0.5 balanced accuracy is what calling everything inactive gives."""
    rng = np.random.RandomState(0)
    n, d = 200, 4
    X = rng.randn(n, d)
    score = X[:, 0] + 0.5 * X[:, 1]
    y = np.where(score > np.quantile(score, 0.95), 1.0, -1.0)      # 5 % actives
    K = X @ X.T
    K /= float(np.mean(np.diag(K)))
    plain = _Learner("klr", {"lam": 0.1}).fit(K, y)
    tuned = _Learner("klr", {"lam": 0.1, "klr_threshold": "cv"}).fit(K, y)
    assert tuned.b_ != 0.0
    bal_plain = classification_scores(y, plain.predict(K))["balanced_accuracy"]
    bal_tuned = classification_scores(y, tuned.predict(K))["balanced_accuracy"]
    assert bal_tuned >= bal_plain
    assert tuned.klr_balanced_ > 0.5
    # the ordering is untouched: only the bias differs, so the AUC cannot move
    assert classification_scores(y, plain.predict(K))["auc"] == \
        pytest.approx(classification_scores(y, tuned.predict(K))["auc"])


def test_the_threshold_helper_reports_nothing_when_no_fold_can_be_fitted():
    K, y, _ = _problem()
    bias, bal = klr_threshold(K[:3, :3], y[:3], 0.1, folds=5)
    assert bias == 0.0 and not np.isfinite(bal)


def test_the_out_of_fold_scores_helper_is_shared_and_scales_each_fold():
    """`out_of_fold_scores` is the machinery plan item E8 built, reused here. Every fold's scores
    are put into that fold's own class-mean units, so the pooled scores are on one scale."""
    K, y, _ = _problem(n=100, seed=3)
    sc, used, folds = out_of_fold_scores(K, y, lambda Ktr, ytr: klr_fit(Ktr, ytr, 0.1)[:2],
                                         folds=5, seed=0)
    assert used == 5 and folds == 5 and np.all(np.isfinite(sc))
    assert classification_scores(y, sc)["auc"] > 0.8
    # pooled over folds, the class means sit near -1 and +1, which is what the scaling asks for
    assert abs(float(np.mean(sc[y > 0])) - 1.0) < 0.6
    assert abs(float(np.mean(sc[y < 0])) + 1.0) < 0.6


def test_the_parameter_grid_is_the_lambda_grid_and_cross_validation_runs():
    K, y, _ = _problem()
    assert _param_grid("klr", (0.1, 1.0), (), (), ()) == [{"lam": 0.1}, {"lam": 1.0}]
    ycv = cv_predict(K, y, "klr", {"lam": 0.1}, folds=5, seed=0)
    assert ycv.shape == y.shape and np.all(np.isfinite(ycv))
    sc = classification_scores(y, ycv)
    assert sc["auc"] > 0.8 and sc["balanced_accuracy"] > 0.7


# --------------------------------------------------------------------------------------------
# on fields
# --------------------------------------------------------------------------------------------

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "c1ccccc1OC",
          "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C", "c1ccc(Br)cc1O",
          "c1ccc(N)cc1C", "c1ccc(F)cc1N", "c1ccc(Cl)cc1C"]
LABELS = np.array([1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0])


def _field_model(method, params, alpha=0.7, cv_folds=4):
    mols = [_mol(s, i) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=("electrostatic", "steric", "hydrophobic"), alpha=alpha)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, task="classification", method=method, params=params,
                     normalize=True)
    model.fit(table, LABELS, cv_folds=cv_folds)
    return model, table, mf


def test_a_fitted_model_keeps_the_decomposition_rules_and_has_a_cofield():
    model, table, mf = _field_model("klr", {"lam": 0.1})
    assert model.stats.task == "classification"
    assert np.isfinite(model.stats.fit["balanced_accuracy"])
    assert np.isfinite(model.stats.cv["balanced_accuracy"])
    for f in range(len(model.fieldset.specs)):
        centres, coeffs, widths = model.cofield_coefficients(f)
        assert len(centres) == len(coeffs) == len(widths)
        assert np.any(np.abs(coeffs) > 0)
    for mol in mf[:3]:
        s = model.predict(AtomTable.from_molecules([mol]))[0]
        assert abs(model.atom_contributions(mol).sum() + model.bias - s) < 1e-8
        assert abs(model.training_contributions(mol).sum() + model.bias - s) < 1e-8


def test_probabilities_come_out_of_the_model_and_survive_a_save():
    import pickle

    model, table, _ = _field_model("klr", {"lam": 0.1})
    p = model.predict_proba(table)
    assert p.shape == LABELS.shape and np.all((p >= 0.0) & (p <= 1.0))
    # the class and the probability agree, because the boundary was left at p = 0.5
    assert np.all((p >= 0.5) == (model.predict(table) >= 0.0))
    back = CMFModel.from_state(pickle.loads(pickle.dumps(model.state())))
    assert back.method == "klr"
    assert np.allclose(back.predict_proba(table), p)
    assert np.allclose(back.predict(table), model.predict(table))


def test_the_lambda_is_chosen_by_the_grid_search():
    mols = [_mol(s, i) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=("electrostatic", "steric"), alpha=0.7)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    model = CMFModel(fieldset=fs, task="classification", method="klr", params={"lam": 1.0},
                     normalize=True)
    res = grid_search(table, LABELS, model, alphas=[0.7], lams=(0.01, 0.1, 1.0), folds=4, seed=0)
    assert set(res.best_params) == {"lam"}
    assert res.best_params["lam"] in (0.01, 0.1, 1.0)
    assert len(res.table) == 3


def test_it_ties_the_gaussian_process_classifier_on_this_small_set():
    """The entry expects the two to agree on accuracy, since they share the likelihood and differ
    only in what their hyper-parameter is chosen by. On fourteen molecules that is all that can
    be asked: the orderings should correlate strongly."""
    klr, table, _ = _field_model("klr", {"lam": 0.1})
    gpc, _, _ = _field_model("gpc", {"amplitude": 10.0, "likelihood": "logit"})
    assert np.corrcoef(klr.predict(table), gpc.predict(table))[0, 1] > 0.99
