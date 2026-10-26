"""Two endpoints on one kernel, and the selectivity co-field that comes out of it."""
import numpy as np
import pytest

from cmfqsar.model import krr_fit, r2_score
from cmfqsar.multitask import (MultiTaskFit, Observations, cv_predict, fit, pair_kernel,
                               score_by_task, score_selectivity, search, task_covariance)


def _kernel(n=24, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    K = X @ X.T
    return K / np.mean(np.diag(K)), X


def test_task_covariance_is_positive_definite_where_it_claims_to_be():
    for n in (2, 3, 4):
        for rho in (0.0, 0.5, 0.9):
            T = task_covariance(n, rho)
            assert np.allclose(T, T.T)
            assert np.all(np.linalg.eigvalsh(T) > 0)
    with pytest.raises(ValueError):
        task_covariance(2, 1.0)
    with pytest.raises(ValueError):
        task_covariance(3, -0.6)


def test_rho_zero_reproduces_two_independent_single_task_models():
    """Without sharing, the fit must be exactly the single-task one, task by task.

    This is what makes any later gain measurable: rho = 0 is the baseline, and if it were not the
    single-task model the comparison would be against something else entirely.
    """
    K, _ = _kernel(20, seed=1)
    rng = np.random.default_rng(2)
    Y = np.column_stack([rng.normal(size=20), rng.normal(size=20) + 3.0])
    obs = Observations.from_columns(Y, names=("am", "eu"))
    f = fit(K, obs, rho=0.0, lam=0.1)
    for t in range(2):
        a_single, b_single = krr_fit(K, Y[:, t], 0.1, fit_bias=True)
        np.testing.assert_allclose(f.task_coefficients(t), a_single, atol=1e-9)
        assert f.bias[t] == pytest.approx(b_single, abs=1e-9)


def test_the_effective_coefficients_reproduce_the_pair_kernel_predictions():
    """c_jt is only a regrouping of the dual solution, so both routes must agree exactly."""
    K, _ = _kernel(18, seed=3)
    rng = np.random.default_rng(4)
    Y = np.column_stack([rng.normal(size=18), rng.normal(size=18)])
    Y[::4, 1] = np.nan                          # a missing endpoint, as in real data
    obs = Observations.from_columns(Y)
    f = fit(K, obs, rho=0.6, lam=0.05)
    for p in range(len(obs)):
        m, t = int(obs.molecule[p]), int(obs.task[p])
        long_way = float(sum(f.alpha[q] * K[m, obs.molecule[q]] * f.T[t, obs.task[q]]
                             for q in range(len(obs))) + f.bias[t])
        assert f.predict(K[m][None, :], t)[0] == pytest.approx(long_way, abs=1e-9)


def test_selectivity_is_the_difference_of_the_two_co_field_coefficients():
    K, _ = _kernel(16, seed=5)
    rng = np.random.default_rng(6)
    Y = np.column_stack([rng.normal(size=16), rng.normal(size=16)])
    obs = Observations.from_columns(Y, names=("am", "eu"))
    f = fit(K, obs, rho=0.4, lam=0.1)
    np.testing.assert_allclose(f.selectivity_coefficients(0, 1),
                               f.task_coefficients(0) - f.task_coefficients(1), atol=1e-12)
    # and the predicted difference is the difference of the predictions
    np.testing.assert_allclose(f.predict_selectivity(K, 0, 1),
                               f.predict(K, 0) - f.predict(K, 1), atol=1e-9)


def test_missing_measurements_are_allowed_and_counted():
    rng = np.random.default_rng(7)
    Y = np.column_stack([rng.normal(size=12), rng.normal(size=12)])
    Y[:5, 1] = np.nan
    obs = Observations.from_columns(Y, names=("a", "b"))
    assert obs.counts() == [12, 7]
    assert len(obs) == 19
    K, _ = _kernel(12, seed=8)
    f = fit(K, obs, rho=0.5, lam=0.1)
    assert f.coef.shape == (12, 2)
    assert np.all(np.isfinite(f.coef))


def test_cross_validation_holds_out_whole_molecules():
    """Both measurements of a held-out molecule must be predicted out of fold.

    If one of them stayed in the training set, the task correlation would leak it into the other.
    """
    K, _ = _kernel(20, seed=9)
    rng = np.random.default_rng(10)
    Y = np.column_stack([rng.normal(size=20), rng.normal(size=20)])
    obs = Observations.from_columns(Y)
    pred = cv_predict(K, obs, rho=0.7, lam=0.1, folds=5, seed=0)
    assert np.all(np.isfinite(pred))
    # a leak would show as the second task of a molecule being predicted much better than the
    # first; with a shuffled target neither should be predictable at all
    shuffled = Observations(obs.molecule, obs.task,
                            np.random.RandomState(0).permutation(obs.y), obs.n_mol, obs.n_tasks)
    sp = cv_predict(K, shuffled, rho=0.7, lam=0.1, folds=5, seed=0)
    assert r2_score(shuffled.y, sp) < 0.3


def test_sharing_helps_when_the_endpoints_are_related():
    """Two noisy views of one underlying signal: a positive rho must do better than rho = 0."""
    n = 40
    K, X = _kernel(n, seed=11)
    rng = np.random.default_rng(12)
    w = rng.normal(size=5)
    base = X @ w
    Y = np.column_stack([base + 0.5 * rng.normal(size=n),
                         base + 0.5 * rng.normal(size=n) + 2.0])
    Y[: n // 2, 1] = np.nan              # the second endpoint is half measured, as usual
    obs = Observations.from_columns(Y, names=("am", "eu"))
    rho, lam, score, per = search(K, obs, rhos=(0.0, 0.5, 0.9), lams=(0.01, 0.1, 1.0),
                                  folds=5, seed=0)
    alone = max(np.nanmean([p["q2"] for p in score_by_task(
        obs, cv_predict(K, obs, rho=0.0, lam=l, folds=5, seed=0))]) for l in (0.01, 0.1, 1.0))
    assert rho > 0.0, "sharing should be selected for two views of one signal"
    assert score > alone - 1e-12
    assert score > 0.3


def test_selectivity_is_scored_on_the_molecules_that_have_both():
    n = 30
    K, X = _kernel(n, seed=13)
    rng = np.random.default_rng(14)
    w1, w2 = rng.normal(size=5), rng.normal(size=5)
    Y = np.column_stack([X @ w1, X @ w2 + 1.0])
    Y[:6, 1] = np.nan
    obs = Observations.from_columns(Y, names=("am", "eu"))
    out = score_selectivity(K, obs, 0, 1, rho=0.3, lam=0.01, folds=5, seed=0)
    assert out["n"] == n - 6
    assert out["q2"] > 0.5          # a linear signal in the kernel's own features
    assert len(out["observed"]) == n - 6


def test_the_selectivity_co_field_is_the_difference_of_the_two_co_fields():
    """The claim the module exists for, checked through the program's own co-field code.

    Nothing in :mod:`cmfqsar.cofields` knows about tasks: it is handed a coefficient vector.  So
    if the difference of the two tasks' coefficients produces exactly the difference of their
    co-fields, selectivity is a co-field in the same sense an activity is, and every reader of a
    co-field -- the 3D view, the cube export, the pharmacophore -- works on it unchanged.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.cofields import cofield_volume, make_grid
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    from cmfqsar.model import CMFModel

    mols = []
    for i, smi in enumerate(["c1ccccc1O", "c1ccccc1N", "c1ccc(Cl)cc1O", "c1ccc(C)cc1N",
                             "c1ccc(F)cc1O", "Cc1ccccc1N"]):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=11 + i)
        m.SetProp("_Name", f"m{i}")
        mols.append(m)
    names = [m.GetProp("_Name") for m in mols]
    rng = np.random.default_rng(15)
    Y = np.column_stack([rng.normal(size=len(mols)), rng.normal(size=len(mols))])

    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = model.joint_kernel(model.prepare_training_kernels(table))
    model.fit(table, Y[:, 0], cv_folds=0)          # fills in train, self_diag and scale

    obs = Observations.from_columns(Y, names=("am", "eu"))
    f = fit(K, obs, rho=0.5, lam=0.01)
    grid = make_grid(table.coords, spacing=1.0)
    vols = []
    for c in (f.task_coefficients(0), f.task_coefficients(1), f.selectivity_coefficients(0, 1)):
        model.coef = c
        vols.append(cofield_volume(model, 0, grid))
    difference, computed = vols[0] - vols[1], vols[2]
    assert np.abs(difference).max() > 1e-6, "the two co-fields must actually differ"
    np.testing.assert_allclose(computed, difference, atol=1e-12)
