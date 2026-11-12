"""The hop models on barriers from density functional theory (plan item BT7, nebDFT2k).

No test needs the downloaded set: the index, the trajectories and the kernels are made up.
"""
import io
import os
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import prepare_nebdft as PREP  # noqa: E402
import run_nebdft as R  # noqa: E402


# --------------------------------------------------------------------------- #
# The preparation
# --------------------------------------------------------------------------- #
def _trajectory(start, end, images=4, others=((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))):
    lines = []
    for k in range(images):
        f = k / (images - 1)
        p = [s + f * (e - s) for s, e in zip(start, end)]
        lines += [str(1 + len(others)), 'Lattice="9 0 0 0 9 0 0 0 9" energy=-1.0',
                  "Li {:.6f} {:.6f} {:.6f} 0.0".format(*p)]
        lines += ["O {:.6f} {:.6f} {:.6f} 0.0".format(*(c + 0.01 * k for c in o))
                  for o in others]
    return "\n".join(lines) + "\n"


def test_the_length_of_a_path_is_read_from_its_trajectory():
    text = _trajectory((1.0, 1.0, 1.0), (4.0, 5.0, 1.0))
    assert PREP.path_length(text) == pytest.approx(5.0)
    assert PREP.path_length("not a file") is None
    one = "\n".join(text.splitlines()[:5])                       # one image only
    assert PREP.path_length(one) is None


def test_a_hop_of_the_index_is_found_in_the_structure():
    symbols = ["Li", "Li", "O"]
    cell = np.diag([4.0, 5.0, 6.0])
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    a, b, off, length = PREP.hop_of("mp-1_0_1_0_0_0", symbols, cell, pos)
    assert (a, b, off) == (0, 1, (0, 0, 0)) and length == pytest.approx(2.0)
    assert PREP.hop_of("mp-1_0_1_-1_0_0", symbols, cell, pos)[3] == pytest.approx(2.0)
    assert PREP.hop_of("mp-1_0_0_0_1_0", symbols, cell, pos)[3] == pytest.approx(5.0)
    assert PREP.hop_of("mp-1_0_2_0_0_0", symbols, cell, pos) is None        # not lithium
    assert PREP.hop_of("mp-1_0_7_0_0_0", symbols, cell, pos) is None        # no such atom


def test_the_set_is_prepared_and_the_left_out_hops_are_counted(tmp_path):
    index = ("material_id,edge_id,chemsys,has_specific_TM,em_bvse,em_dft,_split\n"
             "mp-1,mp-1_0_1_0_0_0,Li-O,False,0.5,0.4,train\n"
             "mp-1,mp-1_0_0_0_1_0,Li-O,False,1.5,0.9,test\n"
             "mp-1,mp-1_0_2_0_0_0,Li-O,False,1.0,1.0,train\n"
             "mp-2,mp-2_0_1_0_0_0,Li-O,False,0.7,0.6,val\n")
    path = tmp_path / "set.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(PREP.INDEX_MEMBER, index)
    with pytest.raises(SystemExit, match="The set has changed"):
        PREP.open_archive(str(path), "0" * 64)
    rows = PREP.read_index(PREP.open_archive(str(path), ""))
    assert len(rows) == 4 and rows[1]["em_dft"] == 0.9 and rows[1]["em_bvse"] == 1.5
    have = {"mp-1": {"formula": "Li2O", "nsites": 3, "cell": np.diag([4.0, 5.0, 6.0]).tolist(),
                     "positions": [[0, 0, 0], [2, 0, 0], [1, 1, 1]],
                     "symbols": ["Li", "Li", "O"]}}
    lengths = {"mp-1_0_1_0_0_0": 2.1, "mp-1_0_0_0_1_0": 3.0, "mp-1_0_2_0_0_0": None}
    out, skipped, dropped = PREP.prepare(rows, have, lengths)
    assert [s["id"] for s in out] == ["mp-1"] and len(out[0]["hops"]) == 1
    hop = out[0]["hops"][0]
    assert hop["y"] == 0.4 and hop["bvse"] == 0.5 and hop["split"] == "train"
    assert hop["length"] == pytest.approx(2.0) and out[0]["oxidation"]["Li"] == 1
    assert skipped == [["mp-2", "not in the Materials Project answer"]]
    assert dropped == {"not two lithium atoms": 1, "length differs from the trajectory": 1,
                       "structure left out": 1}


# --------------------------------------------------------------------------- #
# Kernels and models
# --------------------------------------------------------------------------- #
def _rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    groups = np.repeat(np.arange(n // 3), 3)
    return rng, X, groups


def test_the_kernels_of_a_description():
    rng, X, _ = _rows()
    K = X @ X.T
    G = R.gaussian(K, 1.0)
    assert np.allclose(np.diag(G), 1.0) and np.all(G <= 1.0 + 1e-12)
    d2 = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
    med = np.median(d2[np.triu_indices(len(X), 1)])
    assert np.allclose(G, np.exp(-d2 / med))
    assert np.mean(np.diag(R.unit_scale(K))) == pytest.approx(1.0)
    pool = R.variants(K)
    assert list(pool) == ["linear"] + [f"Gaussian {g:g}" for g in R.GAMMAS]
    assert pool["linear"]() is K and np.allclose(pool["Gaussian 4"](), R.gaussian(K, 4.0))
    b = rng.normal(size=len(X))
    both = R.with_bvse(K, b)
    assert len(both) == len(R.BVSE_WEIGHTS) * (1 + len(R.GAMMAS))
    z = (b - b.mean()) / b.std()
    w = R.BVSE_WEIGHTS[0]
    assert np.allclose(both[f"weight {w:g}, linear"](),
                       R.unit_scale(K) + w * R.unit_scale(np.outer(z, z) + 1.0))


def test_the_ridge_path_and_the_folds_of_whole_structures():
    rng, X, groups = _rows()
    y = X[:, 0] - 0.5 * X[:, 1] + 0.05 * rng.normal(size=len(X))
    K = X @ X.T
    tr, te = np.arange(45), np.arange(45, 60)
    P = R.ridge_path(K, y, tr, te)
    assert P.shape == (len(R.RIDGES), 15)
    Xc = X[tr] - X[tr].mean(axis=0)
    Kc = Xc @ Xc.T
    for k in (1, 5):
        lam = R.RIDGES[k] * np.trace(Kc) / 45
        alpha = np.linalg.solve(Kc + lam * np.eye(45), y[tr] - y[tr].mean())
        assert np.allclose(P[k], (X[te] - X[tr].mean(axis=0)) @ Xc.T @ alpha + y[tr].mean(),
                           atol=1e-8)
    folds = R.group_folds(groups, 4, rng)
    assert sorted(np.concatenate(folds)) == list(range(60))
    assert all(not set(groups[a]) & set(groups[b])
               for i, a in enumerate(folds) for b in folds[i + 1:])


def test_the_kernel_with_information_is_chosen_and_the_scores():
    rng, X, groups = _rows(n=90, seed=1)
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] + 0.05 * rng.normal(size=90)
    noise = rng.normal(size=(90, 6))
    pool = {"noise": lambda: noise @ noise.T, "rows": lambda: X @ X.T,
            "rows, Gaussian": lambda: R.gaussian(X @ X.T, 1.0)}
    tr, te = np.arange(66), np.arange(66, 90)
    R.EDGES.clear()
    p, name = R.kernel_arm(pool, y, groups, tr, te, rng)
    assert name.startswith("rows") and len(R.EDGES) == 1
    s = R.scores(p, y[te])
    assert s["r2"] > 0.7 and s["rmse"] >= s["mae"] > 0
    assert R.scores(y[te], y[te]) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
    f = R.forest_arm(X, y, tr, te)
    assert R.scores(f, y[te])["r2"] > 0.5

    good, bad = y + 0.05 * rng.normal(size=90), y + 0.5 * rng.normal(size=90)
    d = R.paired_mae(good, bad, y, groups, runs=400)
    assert d["high"] < 0 and d["low"] < d["mean"] < d["high"]
    same = R.paired_mae(good, good, y, groups, runs=50)
    assert same == {"mean": 0.0, "low": 0.0, "high": 0.0}


def test_the_names_of_the_models():
    assert R.arm_name("mean of the fit") == "the mean of the fit"
    assert R.arm_name("bvse as it is") == "the bond-valence barrier taken as the prediction"
    assert R.arm_name("bvse|forest") == "random forest on the bond-valence barrier alone"
    assert R.arm_name("overlap|kernel") == \
        "kernel ridge, fields in the frame of the hop (item BT7)"
    assert R.arm_name("invariant+bvse|forest") == \
        "random forest, invariant field about the midpoint, with the bond-valence barrier"


# --------------------------------------------------------------------------- #
# The benchmark on made-up hops
# --------------------------------------------------------------------------- #
class _Targets:
    def __init__(self, y, groups):
        self.y, self.molecule, self.atom = y, groups, np.arange(len(y))


class _Hops:
    def __init__(self, y, groups):
        self._t = _Targets(y, groups)

    def targets(self):
        return self._t


def test_the_benchmark_on_made_up_hops(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    n = 120
    groups = np.repeat(np.arange(n // 2), 2)
    X = rng.normal(size=(n, 5))
    hidden = rng.normal(size=n)                          # what the description does not hold
    y = X[:, 0] + 0.5 * np.tanh(X[:, 1]) + 0.6 * hidden + 0.05 * rng.normal(size=n)
    bvse = 1.0 + hidden + 0.3 * X[:, 0] + 0.1 * rng.normal(size=n)
    split = np.where(np.arange(n) % 10 == 0, "test", np.where(np.arange(n) % 10 == 1, "val",
                                                              "train"))
    split = np.repeat(split[::2], 2)                     # a structure is in one part only
    doc = {"source": "made up", "reference": "nobody", "licence": "MIT", "n_index_hops": n + 5,
           "n_index_structures": n // 2 + 2,
           "dropped_hops": {"not two lithium atoms": 0, "structure left out": 5}}
    monkeypatch.setattr(R, "load", lambda prepared, m: (_Hops(y, groups), doc, bvse, split))
    kernels = {"geometry": X[:, :2] @ X[:, :2].T, "invariant": X @ X.T,
               "overlap": R.gaussian(X @ X.T, 1.0)}
    rows = {"geometry": X[:, :2], "invariant": X}
    monkeypatch.setattr(R, "descriptions", lambda hs, jobs, cache, tag="": (kernels, rows))
    monkeypatch.setattr(R, "FOREST_TREES", 30)
    monkeypatch.setattr(R, "GAMMAS", (1.0,))
    out = R.run(str(tmp_path), folds=4, cache=str(tmp_path / "cache"))
    with np.load(tmp_path / "cache" / "predictions.npz") as z:      # not the real cache
        assert z["y"].shape == (n,) and "folds|overlap|kernel" in z.files
    assert out["hops"] == n and out["structures"] == n // 2
    assert out["published_split"]["structures_in_both_parts"] == 0
    assert out["published_split"]["test"] == int(np.sum(split == "test"))
    cv = out["protocols"]["folds"]["scores"]
    assert {"mean of the fit", "bvse as it is", "bvse|kernel", "bvse|forest",
            "geometry|kernel", "geometry+bvse|kernel", "invariant|kernel",
            "invariant+bvse|kernel", "overlap|kernel", "overlap+bvse|kernel",
            "geometry|forest", "geometry+bvse|forest", "invariant|forest",
            "invariant+bvse|forest"} == set(cv)
    assert max(cv["invariant|kernel"]["mae"], cv["geometry|kernel"]["mae"]) \
        < 0.8 * cv["mean of the fit"]["mae"]
    # the bond-valence barrier holds what the description lacks
    assert cv["invariant+bvse|kernel"]["mae"] < 0.8 * cv["invariant|kernel"]["mae"]
    d = out["differences"]["invariant+bvse|kernel minus invariant|kernel"]
    assert d["high"] < 0
    assert "counts|kernel" not in cv                                # a description not given
    assert set(out["protocols"]["published split"]["scores"]) == set(cv)
    assert 0.0 <= out["ridge_at_an_edge"] <= 1.0
    low, high = cv["invariant|kernel"]["mae_interval"]
    assert low < cv["invariant|kernel"]["mae"] < high

    text = R.report(str(tmp_path))
    assert text.startswith("# Hop barriers from density functional theory (nebDFT2k)")
    assert "| RF-tl | 0.25 | 0.38 | 0.80 |" in text
    assert "| kernel ridge, invariant field about the midpoint, with the bond-valence " \
           "barrier |" in text
    assert "5 hops: structure left out" in text and "not two lithium atoms" not in text
    with open(tmp_path / "table.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    assert R.report(str(tmp_path / "nothing")) == ""
