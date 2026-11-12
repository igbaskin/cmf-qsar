"""The tolerance rule of plan item MS1 on the DFT hop barriers (nebDFT2k).

No test needs the downloaded set: the kernels, the rows and the barriers are made up.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import run_nebdft as N  # noqa: E402
import run_nebdft_rule as R  # noqa: E402


def test_the_ridge_as_a_number_on_the_diagonal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 4)) + 5.0
    K = X @ X.T
    Xc = X - X.mean(axis=0)
    assert R.absolute_ridge(K, 0.1) == pytest.approx(0.1 * np.trace(Xc @ Xc.T) / 30)
    assert R.absolute_ridge(K, 1e-3) == pytest.approx(R.absolute_ridge(K, 1.0) * 1e-3)


def test_the_errors_of_the_answered_hops():
    error = np.array([0.1, -0.2, 0.3, -0.4, 1.0, -2.0, 0.1, 0.1, 0.1, 0.1])
    all_of_them = R.answered_error(error, np.abs(error), 0.0)
    assert all_of_them["mae_answered"] == pytest.approx(np.mean(np.abs(error)))
    assert np.isnan(all_of_them["mae_deferred"])
    best = R.answered_error(error, np.abs(error), 0.2)           # the two largest errors go
    assert best["mae_answered"] == pytest.approx(np.mean([0.1, 0.2, 0.3, 0.4, 0.1, 0.1, 0.1,
                                                          0.1]))
    assert best["mae_deferred"] == pytest.approx(1.5)
    worst = R.answered_error(error, -np.abs(error), 0.2)         # the two smallest go
    assert worst["mae_answered"] > all_of_them["mae_answered"] > best["mae_answered"]
    assert best["rmse_answered"] >= best["mae_answered"]
    # half of the mean error (0.45 / 2) is reached when the two largest errors are gone;
    # 15 per cent of ten hops is the first share that rounds to two hops
    assert R.needed_share(error, np.abs(error), 0.225) == pytest.approx(0.15)
    assert R.needed_share(error, np.abs(error), 1e-9) == 1.0     # never reached
    assert R.needed_share(error, np.abs(error), 9.0) == 0.0


def test_the_signals_that_order_the_hops():
    rng = np.random.default_rng(1)
    error, sd, pred = rng.normal(size=20), rng.random(20), rng.random(20)
    s = R.signals(error, sd, pred, None, rng)
    assert set(s) == {R.SIGNALS[0], R.SIGNALS[1], R.SIGNALS[3], R.SIGNALS[4]}
    coarse = pred + 0.5
    s = R.signals(error, sd, pred, coarse, rng)
    assert set(s) == set(R.SIGNALS) and np.allclose(s[R.SIGNALS[2]], 0.5)
    assert s[R.SIGNALS[0]] is sd and np.allclose(s[R.SIGNALS[4]], np.abs(error))


def _made_up(n=120, seed=2):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n // 2), 2)
    X = rng.normal(size=(n, 5))
    hidden = rng.normal(size=n)
    # the last quarter of the hops is noisier, and the description says where it is
    noisy = (np.arange(n) % 4 == 3).astype(float)
    X[:, 4] = 3.0 * noisy + 0.1 * rng.normal(size=n)
    y = X[:, 0] + 0.5 * np.tanh(X[:, 1]) + 0.5 * hidden + noisy * rng.normal(size=n) \
        + 0.05 * rng.normal(size=n)
    bvse = 1.0 + hidden + 0.3 * X[:, 0] + 0.1 * rng.normal(size=n)
    return rng, groups, X, y, bvse


def test_a_fold_of_a_kernel_model_and_of_the_forest(monkeypatch):
    rng, groups, X, y, bvse = _made_up()
    monkeypatch.setattr(N, "FOREST_TREES", 40)
    monkeypatch.setattr(N, "GAMMAS", (1.0,))
    tr, te = np.arange(90), np.arange(90, 120)
    pool = N.variants(N.unit_scale(X @ X.T))
    f = R.kernel_fold(pool, y, groups, tr, te, rng)
    assert f["mean"].shape == f["sd"].shape == (30,) and np.all(f["sd"] > 0)
    assert f["setting"] in pool and 0.3 < f["factor"] < 3.0
    assert np.sqrt(np.mean((f["mean"] - y[te]) ** 2)) < 1.5 * f["claimed"]
    inner = N.group_folds(groups[tr], 3, rng)
    name, relative = R.chosen(pool, y, groups, tr, inner)
    assert name in pool and relative in N.RIDGES
    g = R.forest_fold(np.hstack([X, bvse[:, None]]), y, groups, tr, te, rng)
    assert g["mean"].shape == (30,) and np.all(g["sd"] > 0) and g["setting"] == "forest"
    # the trees disagree more where the barriers are noisier
    noisy = (te % 4 == 3)
    assert g["sd"][noisy].mean() > g["sd"][~noisy].mean()


class _Targets:
    def __init__(self, y, groups):
        self.y, self.molecule, self.atom = y, groups, np.arange(len(y))


class _Hops:
    def __init__(self, y, groups):
        self._t = _Targets(y, groups)

    def targets(self):
        return self._t


def test_the_rule_on_made_up_hops(tmp_path, monkeypatch):
    rng, groups, X, y, bvse = _made_up(n=160, seed=3)
    split = np.array(["train"] * len(y))
    monkeypatch.setattr(N, "load", lambda prepared, m: (_Hops(y, groups), {}, bvse, split))
    kernels = {"overlap": X @ X.T}
    monkeypatch.setattr(N, "descriptions",
                        lambda hs, jobs, cache, tag="": (kernels, {"invariant": X}))
    monkeypatch.setattr(N, "FOREST_TREES", 40)
    monkeypatch.setattr(N, "GAMMAS", (1.0,))
    out = R.run(str(tmp_path), folds=4)
    assert out["hops"] == 160 and out["structures"] == 80 and set(out["arms"]) == set(R.ARMS)
    for arm in R.ARMS:
        a = out["arms"][arm]
        assert 0.5 < a["z_rms"] < 2.0 and 0.0 <= a["rule"]["answered"] <= 1.0
        assert set(a["budgets"]) == {f"{b:g}" for b in R.BUDGETS}
        best = a["signals"][R.SIGNALS[4]]
        assert best["0.5"] < best["0.2"] < a["mae"]              # the true error orders best
        assert all(best["0.2"] <= a["signals"][s]["0.2"] + 1e-12 for s in a["signals"])
        shares = a["share_for_half_the_error"]
        assert shares[R.SIGNALS[4]] <= min(shares.values())
    assert R.SIGNALS[2] not in out["arms"]["bvse"]["signals"]     # no model to disagree with
    assert R.SIGNALS[2] in out["arms"]["fields"]["signals"]
    # the forest sees which hops are noisy: sending its least sure hops lowers the error
    forest = out["arms"]["forest"]
    assert forest["budgets"]["0.3"]["mae_answered"] < 0.9 * forest["mae"]
    assert forest["spearman"] > 0.1

    text = R.report(str(tmp_path))
    assert text.startswith("# Which hops need the DFT calculation?")
    for heading in ("## Does the standard deviation mean what it says?",
                    "## The rule with the tolerance", "## What orders the hops best"):
        assert heading in text
    assert f"| {R.ARM_NAMES['forest']} |" in text and "| true error (the best possible) |" in text
    with open(tmp_path / "rule.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    assert R.report(str(tmp_path / "nothing")) == ""
