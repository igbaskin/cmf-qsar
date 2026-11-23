"""The molecule vector of a trained device as a kernel view (plan item ND14b)."""
import numpy as np
import pytest

from cmfqsar import deviceview as D


def _data(n=60, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=n)
    return X, y


def test_gaussian_view_is_a_kernel_with_unit_diagonal():
    X, _ = _data()
    K = D.gaussian_view(X, np.arange(40), width=1.0)
    assert K.shape == (60, 60)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9
    assert K.min() >= 0.0


def test_gaussian_view_does_not_depend_on_the_scale_or_the_origin_of_the_vectors():
    X, _ = _data()
    tr = np.arange(40)
    K1 = D.gaussian_view(X, tr)
    K2 = D.gaussian_view(7.0 * X + 100.0, tr)
    assert np.allclose(K1, K2, atol=1e-10)


def test_width_is_a_multiple_of_the_median_distance():
    X, _ = _data()
    tr = np.arange(40)
    wide = D.gaussian_view(X, tr, width=2.0)
    narrow = D.gaussian_view(X, tr, width=0.5)
    off = ~np.eye(60, dtype=bool)
    assert wide[off].mean() > narrow[off].mean()


def test_kernel_correlation_of_a_kernel_with_itself_is_one_and_with_noise_is_not():
    X, _ = _data()
    K = D.gaussian_view(X, np.arange(40))
    assert D.kernel_correlation(K, K) == pytest.approx(1.0)
    # a scaled copy is the same kernel after the normalisation
    assert D.kernel_correlation(K, 3.0 * K) == pytest.approx(1.0)
    Y, _ = _data(seed=5)
    other = D.gaussian_view(Y, np.arange(40))
    assert abs(D.kernel_correlation(K, other)) < 0.5


def test_kernel_correlation_on_rows_uses_only_those_pairs():
    X, _ = _data()
    K = D.gaussian_view(X, np.arange(40))
    rows = np.arange(10, 30)
    direct = D.kernel_correlation(K[np.ix_(rows, rows)], 2.0 * K[np.ix_(rows, rows)])
    assert D.kernel_correlation(K, 2.0 * K, rows=rows) == pytest.approx(direct)


def test_ridge_predict_matches_a_direct_solution():
    X, y = _data()
    K = D.gaussian_view(X, np.arange(40))
    tr, te = np.arange(40), np.arange(40, 60)
    p = D.ridge_predict(K, y, tr, te, 0.01)
    Ktt = K[np.ix_(tr, tr)]
    lam = 0.01 * np.mean(np.diag(Ktt))
    a = np.linalg.solve(Ktt + lam * np.eye(40) + 1e-12 * np.eye(40), y[tr] - y[tr].mean())
    assert np.allclose(p, y[tr].mean() + K[np.ix_(te, tr)] @ a)


def test_ridge_predict_learns_a_smooth_target():
    X, y = _data(n=120)
    tr, te = np.arange(80), np.arange(80, 120)
    K = D.gaussian_view(X, tr, width=1.0)
    p = D.ridge_predict(K, y, tr, te, 0.01)
    assert np.corrcoef(p, y[te])[0, 1] > 0.8


def test_choose_width_returns_values_from_the_grids_and_the_lowest_error():
    X, y = _data(n=120)
    tr, va = np.arange(80), np.arange(80, 100)
    w, lam, e = D.choose_width(X, y, tr, va)
    assert w in D.WIDTHS and lam in D.RIDGES
    for w2 in D.WIDTHS:
        K = D.gaussian_view(X, tr, w2)
        for lam2 in D.RIDGES:
            p = D.ridge_predict(K, y, tr, va, lam2)
            assert np.sqrt(np.mean((p - y[va]) ** 2)) >= e - 1e-12


def test_choose_weight_is_no_worse_than_either_kernel_alone():
    X, y = _data(n=120)
    tr, va = np.arange(80), np.arange(80, 100)
    good = D.gaussian_view(X, tr)
    rng = np.random.default_rng(3)
    other = D.gaussian_view(rng.normal(size=(120, 6)), tr)
    share, lam, e = D.choose_weight(other, good, y, tr, va)
    assert share in D.SHARES and lam in D.RIDGES
    for s in (0.0, 1.0):
        K = (1 - s) * D.cosine_normalise(other) + s * D.cosine_normalise(good)
        alone = min(np.sqrt(np.mean((D.ridge_predict(K, y, tr, va, r) - y[va]) ** 2))
                    for r in D.RIDGES)
        assert e <= alone + 1e-12
    # the informative kernel gets a share, and the one with no information does not win alone
    assert share > 0.0


def test_read_collectors_round_trip(tmp_path):
    p = tmp_path / "x_mp_0.npz"
    n = 12
    np.savez(p, names=np.array([f"m{i}" for i in range(n)]), y=np.arange(n, dtype=np.float32),
             pred=np.zeros(n, np.float32), collectors=np.ones((n, 4), np.float32),
             train=np.arange(8, dtype=np.int32), valid=np.arange(8, 10, dtype=np.int32),
             test=np.arange(10, 12, dtype=np.int32))
    d = D.read_collectors(str(p))
    assert d["collectors"].shape == (n, 4) and list(d["test"]) == [10, 11]
    assert d["names"][3] == "m3" and d["pred"].shape == (n,)


def test_read_collectors_refuses_a_file_without_one_row_per_molecule(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, names=np.array(["a", "b"]), y=np.zeros(2), collectors=np.ones((3, 4)),
             train=np.arange(1), valid=np.arange(1), test=np.arange(1))
    with pytest.raises(ValueError):
        D.read_collectors(str(p))
