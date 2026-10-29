"""Laplacian-regularised kernel ridge regression (method "lapkrr", plan item E10).

The learner adds molecules that have no measured activity to a kernel ridge fit through a
smoothness penalty over the graph of field similarity.  What these tests check is that the solve
is the stationary point of the objective it claims to minimise, that it falls back exactly to
kernel ridge regression when the penalty is switched off, that an activity carried by nobody is
marked by a nan and never scored, and -- the reason the method exists -- that a fitted model is
still one kernel expansion whose co-field and whose atom and molecule contributions include the
unlabelled molecules (run with: python -m pytest tests/test_lapkrr.py).
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import (DEFAULT_MANIFOLDS, LAPKRR_NEIGHBOURS, CMFModel,  # noqa: E402
                           _cv_splits, _Learner, _param_grid, cv_predict, grid_search,
                           krr_fit, lapkrr_fit, r2_score)
from cmfqsar.spectral import knn_laplacian  # noqa: E402


def _problem(n=60, d=5, seed=0, noise=0.2, hidden=0.5):
    """A linear regression problem, its kernel, the activities and the half-hidden activities."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    y = X[:, 0] + 0.5 * X[:, 1] + noise * rng.randn(n)
    K = X @ X.T
    K = K / float(np.mean(np.diag(K)))
    hide = rng.permutation(n)[:int(round(hidden * n))]
    y_half = y.copy()
    y_half[hide] = np.nan
    return K, y, y_half


def _objective(K, y, a, b, lam, gamma, neighbours):
    """The function lapkrr_fit minimises: squared error on the labelled molecules, the ridge on
    the norm of the model, and the smoothness penalty over all of them."""
    lab = np.isfinite(y)
    f = K @ a + b
    L = knn_laplacian(K, neighbours=neighbours)
    g = K @ a                                       # the penalty acts on the model, not the bias
    return (float(np.sum((y[lab] - f[lab]) ** 2)) + lam * float(a @ K @ a)
            + gamma * float(g @ L @ g))


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


# --------------------------------------------------------------------------------------------
# the graph the penalty is taken over
# --------------------------------------------------------------------------------------------

def test_the_laplacian_is_symmetric_positive_semidefinite_with_a_zero_eigenvalue():
    K, _y, _ = _problem(n=40, seed=3)
    L = knn_laplacian(K, neighbours=8)
    assert np.allclose(L, L.T, atol=1e-12)
    w = np.linalg.eigvalsh(L)
    assert w.min() > -1e-10                          # a Laplacian has no negative eigenvalue
    assert w.min() < 1e-8                            # and the connected graph has a zero one
    assert w.max() < 2.0 + 1e-10                     # the normalised form is bounded by 2


def test_the_truncation_keeps_every_molecule_the_number_of_neighbours_asked_for():
    K, _y, _ = _problem(n=50, seed=4)
    for k in (3, 5, 15):
        L = knn_laplacian(K, neighbours=k)
        off = ~np.eye(len(K), dtype=bool)
        degree = np.count_nonzero(np.abs(L[off].reshape(len(K), -1)) > 0, axis=1)
        # the graph is symmetrised, so a molecule can have more neighbours than it chose but
        # never fewer
        assert degree.min() >= k
        assert degree.max() < len(K) - 1 or k >= len(K) - 2


def test_a_zero_or_full_neighbourhood_keeps_every_pair_of_positive_similarity():
    """Only pairs of positive similarity: a negative cosine is not a similarity a graph can use,
    and :func:`cmfqsar.spectral._normalised_laplacian` clips it away."""
    K, _y, _ = _problem(n=20, seed=5)
    full = knn_laplacian(K, neighbours=0)
    assert np.allclose(full, knn_laplacian(K, neighbours=len(K)))
    off = ~np.eye(len(K), dtype=bool)
    d = np.sqrt(np.diag(K))
    cos = K / np.outer(d, d)
    assert np.count_nonzero(np.abs(full[off]) > 0) == np.count_nonzero(cos[off] > 0)
    assert np.count_nonzero(np.abs(full[off]) > 0) > 0


def test_the_graph_does_not_change_when_the_kernel_is_rescaled():
    """The affinity is the cosine of the kernel, so the size of a molecule cannot make it a
    neighbour of everything."""
    K, _y, _ = _problem(n=30, seed=6)
    s = np.exp(np.linspace(-1.0, 1.0, len(K)))       # one molecule ten times "bigger" than another
    L = knn_laplacian(K, neighbours=6)
    L_scaled = knn_laplacian(s[:, None] * K * s[None, :], neighbours=6)
    assert np.allclose(L, L_scaled, atol=1e-10)


# --------------------------------------------------------------------------------------------
# the solve
# --------------------------------------------------------------------------------------------

def test_the_solve_is_the_minimum_of_the_objective():
    K, _y, y_half = _problem(n=50, seed=7)
    lam, gamma, k = 0.1, 0.5, 8
    a, b = lapkrr_fit(K, y_half, lam, manifold=gamma, neighbours=k)
    best = _objective(K, y_half, a, b, lam, gamma, k)
    rng = np.random.RandomState(0)
    for _ in range(200):
        step = 0.02 * np.linalg.norm(a) * rng.randn(len(a))
        assert _objective(K, y_half, a + step, b, lam, gamma, k) >= best * (1.0 - 1e-9)
    for db in (-0.2, -0.02, 0.02, 0.2):
        assert _objective(K, y_half, a, b + db, lam, gamma, k) >= best * (1.0 - 1e-9)


def test_the_bias_stays_outside_the_smoothness_penalty():
    """The penalty is gamma (K a)' L (K a) and not gamma (K a + b)' L (K a + b).

    The two are different objectives, because L 1 is not zero for a normalised Laplacian, while
    it is zero for the unnormalised one.  The documented formula therefore has to say which of
    the two is minimised, and this test holds it to that: the fit is the stationary point of the
    first (the test above) and is not a minimum of the second.
    """
    K, _y, y_half = _problem(n=50, seed=11)
    lam, gamma, k = 0.1, 0.5, 8
    L = knn_laplacian(K, neighbours=k)
    assert np.abs(L @ np.ones(len(K))).max() > 1e-6   # a constant is not free under this L
    a, b = lapkrr_fit(K, y_half, lam, manifold=gamma, neighbours=k)
    assert b != 0.0                                   # otherwise the two objectives coincide

    def bias_inside(a, b):
        lab = np.isfinite(y_half)
        f = K @ a + b
        return (float(np.sum((y_half[lab] - f[lab]) ** 2)) + lam * float(a @ K @ a)
                + gamma * float(f @ L @ f))

    base = bias_inside(a, b)
    assert min(bias_inside(a, b + db) for db in (-1e-3, 1e-3)) < base - 1e-9


def test_the_stated_dual_system_is_what_is_solved():
    """(J K + lam I + gamma L K) a = J y, the equation of the plan entry, with b = 0."""
    K, _y, y_half = _problem(n=40, seed=8)
    lam, gamma, k = 0.3, 0.7, 6
    a, b = lapkrr_fit(K, y_half, lam, manifold=gamma, neighbours=k, fit_bias=False)
    assert b == 0.0
    lab = np.isfinite(y_half)
    J = np.diag(lab.astype(float))
    L = knn_laplacian(K, neighbours=k)
    A = J @ K + lam * np.eye(len(K)) + gamma * (L @ K)
    assert np.allclose(A @ a, J @ np.nan_to_num(y_half), atol=1e-9)


def test_without_the_penalty_it_is_kernel_ridge_regression_on_every_molecule():
    K, y, _ = _problem(n=45, seed=9)
    for lam in (0.01, 0.3, 3.0):
        for bias in (True, False):
            a, b = lapkrr_fit(K, y, lam, manifold=0.0, fit_bias=bias)
            a_ref, b_ref = krr_fit(K, y, lam, fit_bias=bias)
            assert np.allclose(a, a_ref, atol=1e-9)
            assert abs(b - b_ref) < 1e-9


def test_without_the_penalty_the_unlabelled_molecules_drop_out_of_the_model():
    """gamma = 0 gives them a coefficient of zero and the fit of ridge on the labelled ones."""
    K, y, y_half = _problem(n=40, seed=10)
    lab = np.isfinite(y_half)
    a, b = lapkrr_fit(K, y_half, 0.2, manifold=0.0)
    assert np.allclose(a[~lab], 0.0, atol=1e-12)
    a_ref, b_ref = krr_fit(K[np.ix_(lab, lab)], y[lab], 0.2)
    assert np.allclose(a[lab], a_ref, atol=1e-9)
    assert abs(b - b_ref) < 1e-9


def test_the_penalty_gives_the_unlabelled_molecules_coefficients_of_their_own():
    K, _y, y_half = _problem(n=40, seed=11)
    lab = np.isfinite(y_half)
    a, _b = lapkrr_fit(K, y_half, 0.2, manifold=0.5, neighbours=8)
    assert np.max(np.abs(a[~lab])) > 1e-3
    assert np.count_nonzero(np.abs(a) > 1e-9) == len(a)


def test_a_stronger_penalty_makes_the_prediction_vary_less_along_the_graph():
    K, _y, y_half = _problem(n=50, seed=12)
    L = knn_laplacian(K, neighbours=8)
    rough = []
    for gamma in (0.0, 0.1, 1.0, 10.0):
        a, b = lapkrr_fit(K, y_half, 0.1, manifold=gamma, neighbours=8)
        f = K @ a
        rough.append(float(f @ L @ f))
    assert rough == sorted(rough, reverse=True)


def test_every_activity_missing_is_refused_with_a_reason():
    K, _y, _ = _problem(n=20, seed=13)
    with pytest.raises(ValueError, match="at least one molecule with a measured activity"):
        lapkrr_fit(K, np.full(20, np.nan), 0.1)


def _two_bands(n_each=30, width=0.1, labelled=1, seed=0):
    """Two thin curved bands of points, one activity level each, a local kernel.

    This is the situation manifold regularisation was written for: the activity is constant
    along each band, a point is similar only to its own neighbours, and almost no point is
    labelled.  Plain ridge regression predicts almost nothing away from the labelled points,
    because the kernel reaches no further; the penalty carries the value along the band.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, np.pi, n_each)
    X = np.vstack([np.column_stack([np.cos(t), np.sin(t)]),
                   np.column_stack([1 - np.cos(t), -np.sin(t) + 0.4])])
    X = X + 0.05 * rng.randn(2 * n_each, 2)
    y = np.array([0.0] * n_each + [1.0] * n_each)
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    K = np.exp(-sq / (2.0 * width ** 2))
    y_half = np.full(2 * n_each, np.nan)
    for start in (0, n_each):
        y_half[start:start + labelled] = y[start:start + labelled]
    return K, y, y_half


def test_it_recovers_what_the_hidden_activities_cost_where_its_assumption_holds():
    """The mechanism the plan entry claims, on data that satisfies the assumption it states.

    The penalty is only as good as the assumption that the activity varies smoothly along the
    data.  Here it holds by construction, and the unlabelled molecules then carry the fit: the
    hidden activities are predicted far better than by ridge on the visible ones alone.  Whether
    the assumption holds for real series is what ``benchmarks/run_lapkrr.py`` measures; nothing
    here says that it does.
    """
    for seed in range(5):
        K, y, y_half = _two_bands(seed=seed)
        lab = np.isfinite(y_half)
        a0, b0 = lapkrr_fit(K, y_half, 0.01, manifold=0.0)
        a1, b1 = lapkrr_fit(K, y_half, 0.01, manifold=10.0, neighbours=5)
        plain = r2_score(y[~lab], (K[~lab] @ a0) + b0)
        manifold = r2_score(y[~lab], (K[~lab] @ a1) + b1)
        assert plain < 0.3
        assert manifold > 0.7
        assert manifold - plain > 0.4


def test_too_strong_a_penalty_flattens_the_prediction():
    """The penalty is a regulariser, so there is a weight above which it costs accuracy.  The
    weight is therefore a setting to be chosen on the folds and not a constant."""
    K, y, y_half = _two_bands(width=1.0, labelled=3, seed=1)
    lab = np.isfinite(y_half)
    spread = []
    for gamma in (0.0, 1.0, 100.0, 10000.0):
        a, b = lapkrr_fit(K, y_half, 0.01, manifold=gamma, neighbours=5)
        spread.append(float(np.std(K[~lab] @ a + b)))
    assert spread == sorted(spread, reverse=True)
    assert spread[-1] < 0.05 * max(spread[0], 1e-12)


# --------------------------------------------------------------------------------------------
# the learner, the folds and the hyper-parameter grid
# --------------------------------------------------------------------------------------------

def test_the_learner_reaches_the_same_coefficients_as_the_solve():
    K, _y, y_half = _problem(n=40, seed=14)
    lr = _Learner("lapkrr", {"lam": 0.2, "manifold": 0.4, "neighbours": 7}).fit(K, y_half)
    a, b = lapkrr_fit(K, y_half, 0.2, manifold=0.4, neighbours=7)
    assert np.allclose(lr.alpha_, a)
    assert abs(lr.b_ - b) < 1e-12
    assert len(lr.support_) == len(K)                # every molecule is a term of the expansion


def test_the_learner_defaults_to_the_documented_neighbourhood():
    K, _y, y_half = _problem(n=30, seed=15)
    lr = _Learner("lapkrr", {"lam": 0.2, "manifold": 0.4}).fit(K, y_half)
    a, _b = lapkrr_fit(K, y_half, 0.2, manifold=0.4, neighbours=LAPKRR_NEIGHBOURS)
    assert np.allclose(lr.alpha_, a)


def test_the_folds_keep_the_unlabelled_molecules_in_every_training_part():
    _K, _y, y_half = _problem(n=40, seed=16)
    unlabelled = np.flatnonzero(~np.isfinite(y_half))
    held_out = []
    for te, tr in _cv_splits(y_half, 5, 0):
        assert set(unlabelled).issubset(set(tr.tolist()))
        assert not set(te.tolist()) & set(unlabelled.tolist())
        held_out += te.tolist()
    # every labelled molecule is left out exactly once, and no unlabelled one ever is
    assert sorted(held_out) == sorted(np.flatnonzero(np.isfinite(y_half)).tolist())


def test_the_folds_are_unchanged_when_every_activity_is_there():
    _K, y, _ = _problem(n=37, seed=17)
    for seed in (None, 0, 3):
        parts = _cv_splits(y, 5, seed)
        assert sorted(np.concatenate([te for te, _tr in parts]).tolist()) == list(range(37))
        for te, tr in parts:
            assert sorted(te.tolist() + tr.tolist()) == list(range(37))


def test_cross_validated_predictions_are_nan_where_there_is_no_activity():
    K, _y, y_half = _problem(n=40, seed=18)
    yp = cv_predict(K, y_half, "lapkrr", {"lam": 0.1, "manifold": 0.3}, folds=5, seed=0)
    lab = np.isfinite(y_half)
    assert np.isfinite(yp[lab]).all()
    assert np.isnan(yp[~lab]).all()


def test_the_grid_runs_over_the_ridge_and_the_weight_and_includes_plain_ridge():
    grid = _param_grid("lapkrr", [0.1, 1.0], (), (), (), manifolds=[0.0, 0.5])
    assert grid == [{"lam": 0.1, "manifold": 0.0}, {"lam": 0.1, "manifold": 0.5},
                    {"lam": 1.0, "manifold": 0.0}, {"lam": 1.0, "manifold": 0.5}]
    fallback = _param_grid("lapkrr", [1.0], (), (), ())
    assert [p["manifold"] for p in fallback] == sorted(DEFAULT_MANIFOLDS)


# --------------------------------------------------------------------------------------------
# a fitted model: one kernel expansion, over labelled and unlabelled molecules together
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fitted():
    """A small model on nine molecules, four of them without an activity."""
    smiles = ["c1ccccc1C", "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1O", "c1ccccc1Cl",
              "c1ccccc1F", "c1ccccc1Br", "c1ccccc1N", "c1ccccc1C(=O)O"]
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    mols = [build_fields(_mol(s), fs, s) for s in smiles]
    table = AtomTable.from_molecules(mols)
    y = np.array([1.0, 1.4, 1.9, 2.3, 2.6, np.nan, np.nan, np.nan, np.nan])
    model = CMFModel(fieldset=fs, method="lapkrr", normalize=True,
                     params={"lam": 0.1, "manifold": 0.5, "neighbours": 3})
    model.fit(table, y, cv_folds=3)
    return model, table, mols, y


def test_the_expansion_covers_the_unlabelled_molecules(fitted):
    model, table, _mols, y = fitted
    assert len(model.coef) == table.n_mol == len(y)
    assert np.max(np.abs(model.coef[~np.isfinite(y)])) > 1e-9


def test_the_statistics_count_only_the_molecules_with_an_activity(fitted):
    model, _table, _mols, y = fitted
    assert model.stats.n_train == int(np.isfinite(y).sum())
    assert np.isfinite(model.stats.r2) and np.isfinite(model.stats.rmse_fit)
    assert np.isfinite(model.stats.q2) and np.isfinite(model.stats.rmse_cv)


def test_the_molecule_and_atom_contributions_sum_to_the_prediction(fitted):
    model, _table, mols, _y = fitted
    for mol in mols[:3]:
        pred = float(model.predict(AtomTable.from_molecules([mol]))[0])
        by_molecule = float(model.training_contributions(mol).sum()) + model.bias
        by_atom = float(model.atom_contributions(mol).sum()) + model.bias
        assert abs(by_molecule - pred) < 1e-8
        assert abs(by_atom - pred) < 1e-8


def test_the_cofield_is_built_from_every_molecule_of_the_expansion(fitted):
    model, table, _mols, y = fitted
    centres, coeffs, _widths = model.cofield_coefficients(0)
    assert len(centres) == len(table.coords)
    unlabelled = np.flatnonzero(~np.isfinite(y))
    counts = np.diff(table.offsets)
    index = np.repeat(np.arange(table.n_mol), counts)
    carried = np.isin(index, unlabelled)
    assert np.max(np.abs(coeffs[carried])) > 0.0     # the library is in the picture


def test_the_cofield_error_band_treats_only_measured_activities_as_noisy(fitted):
    from cmfqsar.cofields import dual_covariance_factor

    model, _table, _mols, y = fitted
    G = dual_covariance_factor(model)
    assert G.shape == (len(y), int(np.isfinite(y).sum()))


def test_a_saved_model_predicts_the_same_after_being_read_back(fitted, tmp_path):
    from cmfqsar.dataio import load_model, save_model

    model, table, mols, _y = fitted
    path = str(tmp_path / "lapkrr.cmf")
    save_model(path, model)
    back = load_model(path)
    back = back[0] if isinstance(back, tuple) else back
    assert back.method == "lapkrr"
    t = AtomTable.from_molecules([mols[0]])
    assert abs(float(back.predict(t)[0]) - float(model.predict(t)[0])) < 1e-9
    assert len(back.coef) == table.n_mol


def test_the_search_can_choose_the_weight_and_falls_back_to_plain_ridge_when_asked(fitted):
    model, table, _mols, y = fitted
    res = grid_search(table, y, model.fresh_copy(), [0.5], lams=[0.1, 1.0],
                      manifolds=[0.0, 0.5], folds=3, seed=0)
    assert set(res.best_params) >= {"lam", "manifold"}
    assert res.best_params["manifold"] in (0.0, 0.5)
    assert np.isfinite(res.best_rmse)
