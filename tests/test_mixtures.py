"""Mixtures and quantification (plan item SP6, half a): the spectrum and the kernel of a
mixture in `cmfqsar.spectrum`, and the functions of the benchmark.  The spectra are made
here; no test needs a download."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import run_mixtures as X  # noqa: E402
from cmfqsar import spectrum as SP  # noqa: E402


def _sticks(seed, n=8):
    rng = np.random.default_rng(seed)
    return SP.Spectrum(rng.uniform(600.0, 1700.0, n), rng.uniform(0.2, 2.0, n))


# --------------------------------------------------------------------------- #
# The spectrum and the kernel of a mixture
# --------------------------------------------------------------------------- #
def test_the_kernel_of_mixtures_is_the_weighted_sum_of_the_kernels_of_the_components():
    rng = np.random.default_rng(0)
    parts = [_sticks(s) for s in range(4)]
    W = rng.uniform(0.0, 1.0, size=(6, 4))
    V = rng.uniform(0.0, 1.0, size=(3, 4))
    K = SP.exact_kernel(parts, 12.0, normalize=False)
    mixes = [SP.mix(parts, w) for w in W]
    other = [SP.mix(parts, v) for v in V]
    assert np.allclose(SP.exact_kernel(mixes, 12.0, normalize=False), SP.mixture_kernel(K, W))
    assert np.allclose(SP.exact_kernel(mixes, 12.0, others=other, normalize=False),
                       SP.mixture_kernel(K, W, V))
    cos = SP.mixture_kernel(K, W, normalize=True)
    assert np.allclose(cos, SP.exact_kernel(mixes, 12.0)) and np.allclose(np.diag(cos), 1.0)
    # the cosine form removes the total amount, and the other form keeps it
    assert np.allclose(SP.mixture_kernel(K, 3.0 * W, normalize=True), cos)
    assert np.allclose(SP.mixture_kernel(K, 3.0 * W), 9.0 * SP.mixture_kernel(K, W))
    # the cosine kernel of the components is not a start for this product
    wrong = SP.mixture_kernel(SP.exact_kernel(parts, 12.0), W, normalize=True)
    assert np.max(np.abs(wrong - cos)) > 1e-3
    with pytest.raises(ValueError, match="square"):
        SP.mixture_kernel(K[:3], W)
    with pytest.raises(ValueError, match="amounts in every row"):
        SP.mixture_kernel(K, W[:, :3])


def test_mix_adds_the_spectra_times_their_amounts():
    a, b = SP.Spectrum([1000.0, 1200.0], [1.0, 2.0]), SP.Spectrum([1100.0], [4.0])
    m = SP.mix([a, b], [0.5, 2.0], name="two")
    assert m.kind == "sticks" and m.name == "two"
    assert m.x.tolist() == [1000.0, 1100.0, 1200.0] and m.y.tolist() == [0.5, 8.0, 1.0]
    assert np.allclose(SP.mix([a, b], [1.0, 4.0]).y, 2.0 * m.y)   # amounts are not scaled
    x = np.linspace(600.0, 800.0, 11)
    c1 = SP.Spectrum(x, np.sin(x / 50.0) + 2.0, "curve", meta={"yunits": "ABSORBANCE"})
    c2 = SP.Spectrum(x, np.cos(x / 70.0) + 2.0, "curve")
    curve = SP.mix([c1, c2], [0.25, 0.75])
    assert curve.kind == "curve" and np.allclose(curve.y, 0.25 * c1.y + 0.75 * c2.y)
    assert curve.meta == {"yunits": "ABSORBANCE"}
    assert np.allclose(SP.mix([c1], [0.0]).y, 0.0)
    with pytest.raises(ValueError, match="cannot be mixed"):
        SP.mix([a, c1], [1.0, 1.0])
    with pytest.raises(ValueError, match="one grid"):
        SP.mix([c1, SP.Spectrum(x + 1.0, c2.y, "curve")], [1.0, 1.0])
    with pytest.raises(ValueError, match="not negative"):
        SP.mix([a, b], [1.0, -0.1])
    with pytest.raises(ValueError, match="2 components need 2 amounts"):
        SP.mix([a, b], [1.0])
    with pytest.raises(ValueError, match="at least one component"):
        SP.mix([], [])


# --------------------------------------------------------------------------- #
# The mixtures of the benchmark
# --------------------------------------------------------------------------- #
def _pool(n=12, seed=3):
    """Made-up pure spectra on the axis of the benchmark: bands of unit height."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        y = sum(h * np.exp(-0.5 * ((X.AXIS - c) / 9.0) ** 2)
                for c, h in zip(rng.uniform(800, 3700, 10), rng.uniform(0.2, 1.0, 10)))
        out.append(y / y.max())
    return np.vstack(out)


def test_a_mixture_is_the_sum_of_its_components_with_a_baseline_and_noise():
    S = _pool(4)
    rng = np.random.default_rng(1)
    d = X.make_mixtures(S, 50, rng, noise=0.0, flat=20)
    assert d["X"].shape == (50, len(X.AXIS)) and d["amounts"].shape == (50, 4)
    assert np.allclose(d["fractions"].sum(axis=1), 1.0) and d["fractions"].min() >= 0
    total = d["amounts"].sum(axis=1)
    assert total.min() >= X.TOTAL[0] and total.max() <= X.TOTAL[1]
    assert np.allclose(d["amounts"], d["fractions"] * total[:, None])
    assert np.allclose(d["X"][:20], d["amounts"][:20] @ S)          # no baseline there
    base = d["X"][20:] - d["amounts"][20:] @ S
    assert np.allclose(np.diff(base, n=2, axis=1), 0.0, atol=1e-12)  # a straight line
    assert base.max() <= X.OFFSET + X.SLOPE / 2 and np.abs(base).max() > 0.01
    noisy = X.make_mixtures(S, 400, np.random.default_rng(1), flat=400)
    assert np.std(noisy["X"] - noisy["amounts"] @ S) == pytest.approx(X.NOISE, rel=0.02)
    # stray light: the measured absorbance cannot pass -log10(s / (1 + s))
    high = X.make_mixtures(S, 30, np.random.default_rng(2), noise=0.0, flat=30, stray=0.02,
                           scale=50.0)
    assert high["X"].max() == pytest.approx(-np.log10(0.02 / 1.02), abs=1e-3)


def test_the_floor_gives_back_the_amounts_of_a_mixture_without_noise():
    S = _pool(4)
    d = X.make_mixtures(S, 25, np.random.default_rng(4), noise=0.0)
    fit = X.floor_estimate(d["X"], S)
    assert np.allclose(fit["amounts"], d["amounts"], atol=1e-9)
    assert np.allclose(fit["fractions"], d["fractions"], atol=1e-9)
    scaled = X.make_mixtures(S, 5, np.random.default_rng(4), noise=0.0, scale=1.5)
    assert np.allclose(X.floor_estimate(scaled["X"], S, 1.5)["amounts"], scaled["amounts"])
    assert np.allclose(X.as_fractions(np.array([[1.0, 3.0], [0.0, 0.0]])),
                       [[0.25, 0.75], [0.0, 0.0]])


# --------------------------------------------------------------------------- #
# The models
# --------------------------------------------------------------------------- #
def test_kernel_ridge_with_a_constant_is_ridge_regression_on_centred_rows():
    rng = np.random.default_rng(5)
    F = rng.normal(size=(40, 6))
    Y = F @ rng.normal(size=(6, 2)) + 3.0 + 0.05 * rng.normal(size=(40, 2))
    tr, te = np.arange(30), np.arange(30, 40)
    P, alphas, lam = X.ridge_path(F @ F.T, Y, tr, te, ridges=(1e-6, 1e-2))
    centre = F[tr].mean(axis=0)
    Fc = F[tr] - centre
    for k in range(2):
        w = np.linalg.solve(Fc.T @ Fc + lam[k] * np.eye(6), Fc.T @ (Y[tr] - Y[tr].mean(0)))
        assert np.allclose(P[k], (F[te] - centre) @ w + Y[tr].mean(axis=0))
        # the co-spectrum of the centred coefficients is that weight vector
        a = alphas[k] - alphas[k].mean(axis=0, keepdims=True)
        assert np.allclose(SP.co_spectrum(F[tr], a[:, 0], normalize=False), w[:, 0])
    assert lam[1] / lam[0] == pytest.approx(1e4)
    folds = X.inner_folds(30, 3, rng)
    assert sorted(np.concatenate(folds).tolist()) == list(range(30))
    fit = X.kernel_model({"a": F @ F.T, "noise": np.eye(40)}, Y, tr, te, folds)
    assert fit["kernel"] == "a" and X.rmse(fit["prediction"], Y[te]) < 0.2
    assert fit["alpha"].shape == (30, 2)


def test_the_gaussian_layer_and_the_pls_path():
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(6)
    F = rng.normal(size=(50, 30))
    Y = F[:, :4] @ rng.normal(size=(4, 2)) + 0.1 * rng.normal(size=(50, 2))
    tr, te = np.arange(35), np.arange(35, 50)
    K = F @ F.T
    G = X.gaussian_layer(K, tr, 4.0)
    d2 = ((F[:, None, :] - F[None, :, :]) ** 2).sum(axis=2)
    med = np.median(d2[np.ix_(tr, tr)][np.triu_indices(35, 1)])
    assert np.allclose(G, np.exp(-d2 / (8.0 * med))) and np.allclose(np.diag(G), 1.0)
    P = X.pls_path(F, Y, tr, te, 6)
    for k in (1, 3, 6):
        ref = PLSRegression(n_components=k, scale=False).fit(F[tr], Y[tr]).predict(F[te])
        assert np.allclose(P[k - 1], ref)
    folds = X.inner_folds(35, 3, rng)
    fit = X.pls_model({"rows": F, "noise": rng.normal(size=(50, 30))}, Y, tr, te, folds)
    assert fit["rows"] == "rows" and 2 <= fit["components"] <= X.PLS_MAX
    assert X.rmse(fit["prediction"], Y[te]) < 0.5 * np.std(Y[te])


def test_the_net_signal_is_what_the_other_components_cannot_make():
    S = _pool(4)
    ramp = np.linspace(-0.5, 0.5, S.shape[1])[None, :]
    base = np.vstack([np.ones(S.shape[1]), ramp])
    n = X.net_signal(S, 1, base)
    assert np.allclose(np.delete(S, 1, axis=0) @ n, 0.0, atol=1e-9)
    assert np.allclose(base @ n, 0.0, atol=1e-9) and n @ S[1] > 0
    assert X.cosine(n, n) == pytest.approx(1.0) and X.cosine(n, -n) == pytest.approx(-1.0)
    assert np.allclose(X.net_signal(S, 0), S[0] - np.linalg.lstsq(
        S[1:].T, S[0], rcond=None)[0] @ S[1:])


def test_the_rows_of_a_description():
    S = _pool(3)
    plain = X.rows(S, 8.0, "plain", False)
    assert plain.shape == (3, len(X.GRID.x))
    unit = X.rows(S, 8.0, "slope", True)
    assert np.allclose(np.einsum("ij,ij->i", unit, unit), 1.0)
    lifted = X.rows(S + 0.3, 8.0, "slope", False)
    assert np.allclose(lifted, X.rows(S, 8.0, "slope", False), atol=1e-9)   # no offset
    assert X.describe("slope", True) == "slope, cosine"
    assert X.describe("plain", False) == "plain, not normalised"
    assert X.identity_check(S, np.random.default_rng(0)) < 1e-12 if len(S) >= 4 else True


# --------------------------------------------------------------------------- #
# One repeat and the whole run
# --------------------------------------------------------------------------- #
def test_one_repeat_finds_the_amounts_near_the_floor_and_the_fractions_from_them(monkeypatch):
    monkeypatch.setattr(X, "LENGTHS", (16.0,))
    out = X.one_repeat(_pool(), np.random.default_rng(7), n_train=60, n_test=60,
                       widths=(0.0, 8.0))
    plain, slope = X.describe("plain", False), X.describe("slope", False)
    floor = out["floor"]["amounts"]
    for method in ("kernel", "pls"):
        assert out[method]["amounts"][plain] < 1.5 * floor
        assert out[method]["amounts"][slope] < 2.5 * floor
        assert out[method]["fractions"][plain + X.DIVIDED] < 1.5 * out["floor"]["fractions"]
        # a fraction is not a linear function of the spectrum, in either form
        assert out[method]["fractions"][plain] > 5 * out["floor"]["fractions"]
        assert out[method]["fractions"][X.describe("slope", True)] > \
            2 * out[method]["fractions"][plain + X.DIVIDED]
        # the cosine form has divided the total amount out
        assert out[method]["amounts"][X.describe("plain", True)] > 20 * floor
    assert set(out["kernel"]["fractions"]) == set(out["pls"]["fractions"])
    assert len(out["kernel"]["fractions"]) == 6 and len(out["kernel"]["amounts"]) == 4
    assert out["net_signal_cosine"] > 0.8
    assert out["chosen"][slope + "|amounts"] in ("0", "8")


def test_a_baseline_that_was_not_in_the_fit_spares_the_slopes_only(monkeypatch):
    monkeypatch.setattr(X, "LENGTHS", (16.0,))
    out = X.one_repeat(_pool(), np.random.default_rng(8), n_train=60, n_test=60,
                       widths=(8.0,), unseen_baseline=True)
    floor = out["floor"]["amounts"]
    assert out["kernel"]["amounts"][X.describe("slope", False)] < 3 * floor
    assert out["kernel"]["amounts"][X.describe("plain", False)] > 10 * floor
    assert out["pls"]["amounts"][X.describe("plain", False)] > 10 * floor


def test_the_run_writes_its_table(tmp_path, monkeypatch):
    pool = _pool()
    monkeypatch.setattr(X, "LENGTHS", (16.0,))
    monkeypatch.setattr(X, "computed_components", lambda: pool)
    monkeypatch.setattr(X, "measured_components", lambda: None)
    out = X.run(str(tmp_path), repeats=2, n_train=40, n_test=40, widths=(8.0,))
    assert list(out["cases"]) == ["computed", "stray", "unseen"]      # no measured part
    assert out["identity"] < 1e-12 and out["train"] == 40 and out["pool"] == {"computed": 12}
    c = out["cases"]["computed"]
    assert c["repeats"] == 2 and c["linear"] and not out["cases"]["stray"]["linear"]
    assert set(c["differences"]["amounts"][X.describe("plain", False)]) == {
        "kernel_minus_pls", "gaussian_minus_kernel", "kernel_over_floor"}
    assert c["width_chosen_slope_amounts"] == {"8": 2}
    assert "net_signal_cosine" in c and "net_signal_cosine" not in out["cases"]["stray"]
    text = X.report(str(tmp_path))
    assert text.startswith("# Mixtures and quantification")
    for part in ("## Components: computed spectra of benzoic acids",
                 "with a baseline only in the test mixtures", "which is not a floor here",
                 "amounts divided by their sum", "net signal"):
        assert part in text
    with open(tmp_path / "table.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    assert X.report(str(tmp_path / "nothing")) == ""


def test_the_computed_components_are_curves_of_unit_height():
    S = X.computed_components()
    assert S.shape == (189, len(X.AXIS)) and np.allclose(S.max(axis=1), 1.0)
    assert S.min() >= 0.0
