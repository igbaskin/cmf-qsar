"""Tests of cmfqsar.shift: density ratios, weighted fitting and weighted conformal intervals."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.conformal import Conformal  # noqa: E402
from cmfqsar.model import krr_fit  # noqa: E402
from cmfqsar.shift import (_project_box_sum, distance2, effective_size, gaussian_layer,  # noqa: E402
                           kmm, ulsif, weighted_calibrate, weighted_cv_predict,
                           weighted_krr_fit, weighted_mmd2, weighted_quantile)


def _rbf(a, b, gamma=0.5):
    return np.exp(-gamma * (a[:, None] - b[None, :]) ** 2)


def _normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


# --------------------------------------------------------------------------- weighted fitting

def test_equal_weights_reproduce_krr():
    rng = np.random.RandomState(0)
    x = rng.randn(30)
    K = _rbf(x, x)
    y = np.sin(x) + 0.1 * rng.randn(30)
    for bias in (True, False):
        a0, b0 = krr_fit(K, y, 0.1, bias)
        a1, b1 = weighted_krr_fit(K, y, np.full(30, 3.7), 0.1, bias)
        assert np.allclose(a0, a1, atol=1e-10) and abs(b0 - b1) < 1e-10


def test_zero_weight_removes_the_molecule():
    rng = np.random.RandomState(1)
    x = rng.randn(20)
    K = _rbf(x, x)
    y = np.cos(x)
    w = np.ones(20)
    w[[3, 7]] = 0.0
    keep = np.flatnonzero(w > 0)
    a, b = weighted_krr_fit(K, y, w, 0.05)
    a0, b0 = krr_fit(K[np.ix_(keep, keep)], y[keep], 0.05 * 18 / 20)   # kept weights become 20/18
    assert np.allclose(a[keep], a0, atol=1e-9) and abs(b - b0) < 1e-9
    assert a[3] == 0.0 and a[7] == 0.0


def test_weighted_fit_minimises_its_objective():
    rng = np.random.RandomState(2)
    x = rng.randn(25)
    K = _rbf(x, x) + 1e-10 * np.eye(25)
    y = x ** 2 + 0.2 * rng.randn(25)
    w = rng.uniform(0.2, 3.0, 25)
    w = w / w.mean()
    lam = 0.3
    a, b = weighted_krr_fit(K, y, w, lam)

    def objective(a_, b_):
        r = y - K @ a_ - b_
        return float(np.sum(w * r * r) + lam * a_ @ K @ a_)

    base = objective(a, b)
    for _ in range(20):                    # every small move along the constraint 1'a = 0 is worse
        d = rng.randn(25)
        d -= d.mean()
        assert objective(a + 1e-3 * d, b) >= base - 1e-12
        assert objective(a, b + 1e-3) >= base - 1e-12


def test_weighted_cv_predict_equal_weights_matches_cv_predict():
    from cmfqsar.model import cv_predict

    rng = np.random.RandomState(3)
    x = rng.randn(40)
    K = _rbf(x, x)
    y = np.sin(2 * x)
    p0 = cv_predict(K, y, "krr", {"lam": 0.05}, folds=5, seed=4)
    p1 = weighted_cv_predict(K, y, np.ones(40), 0.05, folds=5, seed=4)
    assert np.allclose(p0, p1, atol=1e-9)


# --------------------------------------------------------------------------- the quantile

@pytest.mark.parametrize("n", [5, 9, 19, 20, 50])
@pytest.mark.parametrize("level", [0.8, 0.9, 0.95])
def test_equal_weights_give_the_ordinary_quantile(n, level):
    rng = np.random.RandomState(n)
    r = np.abs(rng.randn(n))
    q_plain = Conformal(level=level, residuals=r).quantile
    q_w = weighted_quantile(r, np.ones(n), [1.0], level)[0]
    assert (np.isinf(q_plain) and np.isinf(q_w)) or q_plain == q_w


def test_heavy_new_molecule_gets_an_infinite_interval():
    r = np.arange(1.0, 21.0)
    q = weighted_quantile(r, np.ones(20), [0.0, 1.0, 5.0], 0.9)
    assert q[0] == 18.0                    # the new molecule carries nothing: the plain 90 % point
    assert np.isfinite(q[1])
    assert np.isinf(q[2])                  # 5 / 25 = 0.2 > 1 - 0.9 sits at infinity


def test_weighted_conformal_restores_coverage_under_shift():
    """Heteroscedastic errors and a shift towards the noisy region: the true ratio repairs it."""
    rng = np.random.RandomState(5)
    level, trials, n_cal = 0.9, 400, 200
    hit_w = hit_u = 0
    for _ in range(trials):
        x_cal = rng.randn(n_cal)
        r_cal = np.abs((0.2 + np.abs(x_cal)) * rng.randn(n_cal))
        x_new = 1.0 + rng.randn()
        r_new = abs((0.2 + abs(x_new)) * rng.randn())
        w = lambda x: _normal_pdf(x, 1.0, 1.0) / _normal_pdf(x, 0.0, 1.0)   # noqa: E731
        q_w = weighted_quantile(r_cal, w(x_cal), [w(x_new)], level)[0]
        q_u = Conformal(level=level, residuals=r_cal).quantile
        hit_w += r_new <= q_w
        hit_u += r_new <= q_u
    assert hit_w / trials >= level - 0.03
    assert hit_u / trials < level - 0.05


def test_weighted_calibrate_interval_shapes():
    c = weighted_calibrate([1.0, 2.0, 3.0, 4.0] * 5, [1.1, 1.8, 3.3, 4.0] * 5, np.ones(20),
                           level=0.8, difficulty=np.linspace(0, 1, 20), beta=0.5)
    lo, hi = c.interval([0.0, 1.0], [1.0, 1.0], difficulty=[0.0, 1.0])
    assert lo.shape == (2,) and np.all(hi >= lo)
    assert (hi - lo)[1] == pytest.approx(3 * (hi - lo)[0])   # (0.5 + 1) / (0.5 + 0)
    assert c.ess == pytest.approx(20.0)


# --------------------------------------------------------------------------- density ratios

def test_projection_is_feasible_and_optimal():
    from scipy.optimize import minimize

    rng = np.random.RandomState(6)
    for _ in range(5):
        v = 3 * rng.randn(12)
        p = _project_box_sum(v, 2.0, 8.0, 10.0)
        assert np.all(p >= -1e-12) and np.all(p <= 2.0 + 1e-12)
        assert 8.0 - 1e-6 <= p.sum() <= 10.0 + 1e-6
        res = minimize(lambda b: np.sum((b - v) ** 2), np.ones(12), method="SLSQP",
                       bounds=[(0, 2)] * 12,
                       constraints=[{"type": "ineq", "fun": lambda b: b.sum() - 8.0},
                                    {"type": "ineq", "fun": lambda b: 10.0 - b.sum()}])
        assert np.sum((p - v) ** 2) <= res.fun + 1e-6


def test_kmm_reduces_the_discrepancy_and_tracks_the_ratio():
    rng = np.random.RandomState(7)
    x_tr = rng.randn(150)
    x_te = 0.8 + 0.6 * rng.randn(100)
    r = kmm(_rbf(x_tr, x_tr), _rbf(x_tr, x_te), _rbf(x_te, x_te), bound=20.0)
    assert r.mmd2_after < 0.2 * r.mmd2_before
    assert np.all(r.weights >= 0) and np.all(r.weights <= 20.0 + 1e-9)
    n = len(x_tr)
    eps = (np.sqrt(n) - 1) / np.sqrt(n)
    assert n * (1 - eps) - 1e-6 <= r.weights.sum() <= n * (1 + eps) + 1e-6
    true = _normal_pdf(x_tr, 0.8, 0.6) / _normal_pdf(x_tr, 0.0, 1.0)
    assert np.corrcoef(r.weights, true)[0, 1] > 0.8
    assert 1.0 < r.ess < n


def test_ulsif_recovers_a_known_ratio():
    rng = np.random.RandomState(8)
    x_tr = rng.randn(300)
    x_te = 0.7 + 0.8 * rng.randn(200)
    D2_tr_te = (x_tr[:, None] - x_te[None, :]) ** 2
    D2_te_te = (x_te[:, None] - x_te[None, :]) ** 2
    ratio = ulsif(D2_tr_te, D2_te_te, centres=100, seed=0)
    est = ratio.evaluate(D2_tr_te[:, ratio.centres])
    true = _normal_pdf(x_tr, 0.7, 0.8) / _normal_pdf(x_tr, 0.0, 1.0)
    assert np.corrcoef(est, true)[0, 1] > 0.9
    assert 0.7 < est.mean() < 1.3          # the ratio integrates to one under the training density


def test_distance_and_mmd_helpers():
    rng = np.random.RandomState(9)
    X = rng.randn(6, 3)
    K = X @ X.T
    d = np.diag(K)
    D2 = distance2(K, d, d)
    assert np.allclose(D2, ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    assert np.all(gaussian_layer(D2, 1.0) <= 1.0)
    Kt = K[:3, :3]
    assert weighted_mmd2(Kt, K[:3, 3:], K[3:, 3:]) == pytest.approx(
        float(np.sum((X[:3].mean(0) - X[3:].mean(0)) ** 2)))
    assert effective_size([1, 1, 1, 1]) == pytest.approx(4.0)
    assert effective_size([1, 0, 0, 0]) == pytest.approx(1.0)


def test_crossfit_removes_the_self_term_without_shift():
    """Same distribution on both sides: in-sample weights favour the library, cross-fitted do not."""
    from cmfqsar.shift import ulsif_crossfit

    rng = np.random.RandomState(10)
    gap_in, gap_cross = [], []
    for _ in range(10):
        x_tr, x_te = rng.randn(120), rng.randn(60)
        D2_tr_te = (x_tr[:, None] - x_te[None, :]) ** 2
        D2_te_te = (x_te[:, None] - x_te[None, :]) ** 2
        ratio, w_tr, w_te = ulsif_crossfit(D2_tr_te, D2_te_te, seed=0)
        in_tr = ratio.evaluate(D2_tr_te[:, ratio.centres])
        in_te = ratio.evaluate(D2_te_te[:, ratio.centres])
        gap_in.append(in_te.mean() - in_tr.mean())
        gap_cross.append(w_te.mean() - w_tr.mean())
    assert np.mean(gap_in) > 0.0
    assert abs(np.mean(gap_cross)) < abs(np.mean(gap_in))


def test_classifier_ratio_recovers_a_known_ratio():
    from cmfqsar.shift import classifier_ratio

    rng = np.random.RandomState(11)
    x_tr = rng.randn(200)
    x_te = 0.7 + 0.8 * rng.randn(100)
    x = np.concatenate([x_tr, x_te])
    lib = np.r_[np.zeros(200, bool), np.ones(100, bool)]
    r = classifier_ratio((x[:, None] - x[None, :]) ** 2, lib, seed=0)
    true = _normal_pdf(x_tr, 0.7, 0.8) / _normal_pdf(x_tr, 0.0, 1.0)
    # out-of-fold weights: each comes from a classifier that did not see the molecule
    assert np.corrcoef(np.log(r.w_train), np.log(true))[0, 1] > 0.85
    assert 0.6 < r.w_train.mean() < 1.4
    assert r.log_loss < 0.6365                 # better than knowing only the proportions
    new = np.array([-2.0, 0.0, 1.5])
    w_new = r.evaluate((new[:, None] - x[None, :]) ** 2)
    assert w_new[0] < w_new[1] < w_new[2]


# --------------------------------------------------------------------------- the command line

TRAIN_SMILES = ["CCO", "CCCO", "CCCCO", "CC(C)O", "CCN", "CCCN", "CCCCN", "CC(C)N", "CCC(=O)O",
                "CCCC(=O)O", "CC(=O)OC", "CCOC", "CCCOC", "CCCl", "CCCCl", "CCBr", "CC(C)Cl",
                "CC(C)(C)O", "OCCO", "NCCO", "CCS", "CCCS", "c1ccccc1O", "c1ccccc1N"]
LIBRARY_SMILES = ["c1ccccc1CO", "c1ccccc1CCO", "c1ccccc1CN", "Oc1ccc(Cl)cc1", "Nc1ccc(Cl)cc1",
                  "c1ccccc1C(=O)O", "c1ccc2ccccc2c1O", "Cc1ccccc1O", "CCc1ccccc1N", "CCO"]


def _write(path, smiles, activity=True):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(smiles):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            prm = AllChem.ETKDGv3()
            prm.randomSeed = k + 1
            AllChem.EmbedMolecule(m, prm)
            m.SetProp("_Name", f"m{k}")
            if activity:
                m.SetProp("activity", f"{3.0 + 0.1 * m.GetNumHeavyAtoms() + 0.05 * (k % 3):.3f}")
            w.write(m)
    return path


def test_predict_shift_writes_weighted_intervals(tmp_path, capsys):
    import csv

    from cmfqsar.cli import main

    train = _write(os.path.join(str(tmp_path), "train.sdf"), TRAIN_SMILES)
    lib = _write(os.path.join(str(tmp_path), "lib.sdf"), LIBRARY_SMILES, activity=False)
    saved = os.path.join(str(tmp_path), "m.cmf")
    main(["build", train, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--test-fraction", "0", "--folds", "4",
          "--conformal", "0.8", "--save", saved])
    capsys.readouterr()
    out = os.path.join(str(tmp_path), "p.csv")
    main(["predict", saved, lib, "--shift", "--out", out])
    text = capsys.readouterr().out
    assert "Weighted conformal (--shift)" in text
    assert "kernel logistic regression" in text
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(LIBRARY_SMILES)
    assert {"lower", "upper", "shift_weight"} <= set(rows[0])
    w = np.array([float(r["shift_weight"]) for r in rows])
    assert np.all(w >= 0)
    for r in rows:
        if r["lower"] == "" or r["upper"] == "":
            assert r["lower"] == r["upper"] == ""          # the CSV writes an infinite interval empty
            continue
        assert float(r["lower"]) <= float(r["predicted"]) <= float(r["upper"])


def test_predict_shift_needs_a_calibration(tmp_path):
    from cmfqsar.cli import main

    train = _write(os.path.join(str(tmp_path), "train.sdf"), TRAIN_SMILES)
    saved = os.path.join(str(tmp_path), "m.cmf")
    main(["build", train, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--test-fraction", "0", "--folds", "4",
          "--save", saved])
    with pytest.raises(SystemExit, match="--conformal"):
        main(["predict", saved, train, "--shift"])


def test_compare_writes_the_density_ratio(tmp_path, capsys):
    import csv

    from cmfqsar.cli import main

    a = _write(os.path.join(str(tmp_path), "a.sdf"), TRAIN_SMILES)
    b = _write(os.path.join(str(tmp_path), "b.sdf"), LIBRARY_SMILES, activity=False)
    out = os.path.join(str(tmp_path), "w.csv")
    main(["compare", a, b, "--fields", "electrostatic,steric", "--permutations", "49",
          "--weights", out])
    text = capsys.readouterr().out
    assert "Weights that make" in text and "effective size" in text
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(TRAIN_SMILES) + len(LIBRARY_SMILES)
    w_a = np.array([float(r["density_ratio"]) for r in rows if r["set"] == a])
    w_b = np.array([float(r["density_ratio"]) for r in rows if r["set"] == b])
    assert np.all(np.isfinite(w_a)) and np.all(w_a >= 0)
    # p_b / p_a is larger, on average, where the molecules of b are
    assert np.mean(w_b) > np.mean(w_a)


def test_model_blocks_form_one_kernel(tmp_path):
    """The training set given as the library: all three blocks must be the training kernel."""
    from cmfqsar.cli import _model_blocks, main
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import AtomTable, build_fields
    from cmfqsar.dataio import load_model

    train = _write(os.path.join(str(tmp_path), "train.sdf"), TRAIN_SMILES)
    saved = os.path.join(str(tmp_path), "m.cmf")
    main(["build", train, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--test-fraction", "0", "--folds", "4",
          "--save", saved])
    model, _ = load_model(saved)
    ds = load_dataset(train)
    table = AtomTable.from_molecules([build_fields(m, model.fieldset, n)
                                      for m, n in zip(ds.mols, ds.names)])
    K_train, K_cross, K_lib = _model_blocks(model, table)
    assert np.allclose(K_cross, K_train, rtol=1e-9, atol=1e-12)
    assert np.allclose(K_lib, K_train, rtol=1e-9, atol=1e-12)
    assert np.allclose(K_train, model._linear_train_kernel(), rtol=1e-9, atol=1e-12)
