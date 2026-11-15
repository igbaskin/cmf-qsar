"""Tests of cmfqsar.herding: herding, its exact greedy form, the weights and the references."""
import itertools
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.herding import (herd, kmedoids, maxmin, mmd2_subset, pick,  # noqa: E402
                             pivoted_cholesky, quadrature_weights, target_weights)


def _gauss(x, y, gamma=1.0):
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    y = np.atleast_2d(y.T).T if y.ndim == 1 else y
    d2 = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


def _sample(n=200, seed=0):
    rng = np.random.RandomState(seed)
    return np.r_[rng.randn(n // 2) - 1.5, 0.6 * rng.randn(n - n // 2) + 1.5][:, None]


def test_herding_beats_random_subsets():
    X = _sample()
    K = _gauss(X, X)
    rng = np.random.RandomState(1)
    for k in (5, 10, 20):
        h = mmd2_subset(K, herd(K, k))
        r = np.median([mmd2_subset(K, rng.choice(len(X), k, replace=False)) for _ in range(50)])
        assert h < 0.25 * r


def test_greedy_is_the_exact_greedy_step():
    X = _sample(40, seed=2)
    K = _gauss(X, X, gamma=0.5) + np.diag(np.linspace(0, 0.3, 40))     # a varying diagonal
    idx = list(herd(K, 6, method="greedy"))
    for t in range(1, 6):
        best = min((j for j in range(40) if j not in idx[:t]),
                   key=lambda j: mmd2_subset(K, idx[:t] + [j]))
        assert mmd2_subset(K, idx[:t] + [best]) == pytest.approx(mmd2_subset(K, idx[:t + 1]),
                                                                 rel=1e-12, abs=1e-15)


def test_herding_and_greedy_agree_on_a_constant_diagonal():
    X = _sample(60, seed=3)
    K = _gauss(X, X)
    assert np.array_equal(herd(K, 15, method="herding"), herd(K, 15, method="greedy"))


def test_a_weighted_target_pulls_the_choice():
    X = np.linspace(-3, 3, 61)[:, None]
    K = _gauss(X, X)
    p = np.where(X[:, 0] > 1.0, 1.0, 0.0)
    idx = herd(K, 5, weights=p)
    assert np.all(X[idx, 0] > 0.5)


def test_first_continues_the_rule():
    X = _sample(80, seed=4)
    K = _gauss(X, X)
    full = herd(K, 8)
    rest = herd(K, 5, first=full[:3])
    assert np.array_equal(rest, full[3:])


def test_quadrature_weights():
    X = _sample(100, seed=5)
    K = _gauss(X, X)
    idx = herd(K, 8)
    v = quadrature_weights(K, idx)
    assert np.all(v >= -1e-12) and v.sum() == pytest.approx(1.0)
    assert mmd2_subset(K, idx, v) <= mmd2_subset(K, idx) + 1e-12
    u = quadrature_weights(K, idx, simplex=False)
    assert mmd2_subset(K, idx, u) <= mmd2_subset(K, idx, v) + 1e-10


def test_quadrature_weights_recover_a_weighted_target_exactly():
    rng = np.random.RandomState(6)
    X = rng.randn(12, 3)
    K = X @ X.T + 1.0           # a linear kernel: a weighted mean of the pool is reachable
    p = target_weights(12, rng.uniform(0, 1, 12))
    v = quadrature_weights(K, np.arange(12), weights=p)
    assert mmd2_subset(K, np.arange(12), v, weights=p) < 1e-10


def test_maxmin_takes_the_ends_early():
    X = np.linspace(0, 10, 101)[:, None]
    idx = maxmin(_gauss(X, X, gamma=0.01), 3)
    assert set(X[idx[1:], 0]) == {0.0, 10.0}


def test_kmedoids_finds_the_clusters_and_the_optimum():
    rng = np.random.RandomState(7)
    X = np.r_[rng.randn(10, 2) * 0.2, rng.randn(10, 2) * 0.2 + 5, rng.randn(10, 2) * 0.2 - 5]
    K = X @ X.T                  # the field distance is then the Euclidean distance
    med = kmedoids(K, 3)
    assert sorted(set(m // 10 for m in med)) == [0, 1, 2]
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    cost = lambda m: D[:, list(m)].min(axis=1).sum()      # noqa: E731
    small = np.r_[0:6, 10:16]
    Ds = D[np.ix_(small, small)]
    best = min(itertools.combinations(range(12), 2), key=lambda m: Ds[:, list(m)].min(1).sum())
    got = kmedoids(K[np.ix_(small, small)], 2)
    assert Ds[:, list(got)].min(1).sum() == pytest.approx(Ds[:, list(best)].min(1).sum())
    assert cost(med) < 30 * 0.5


def test_pivoted_cholesky_is_exact_at_the_rank():
    rng = np.random.RandomState(8)
    F = rng.randn(30, 4)
    K = F @ F.T
    idx = pivoted_cholesky(K, 4)
    Kt = K[:, idx] @ np.linalg.pinv(K[np.ix_(idx, idx)]) @ K[idx]
    assert np.allclose(Kt, K, atol=1e-8)


def test_pick_dispatches_and_never_repeats():
    X = _sample(50, seed=9)
    K = _gauss(X, X)
    for m in ("herding", "greedy", "maxmin", "kmedoids", "pivoted_cholesky", "random"):
        idx = pick(K, 10, method=m)
        assert len(idx) == 10 and len(set(idx.tolist())) == 10
    with pytest.raises(ValueError):
        pick(K, 10, method="nonsense")
    with pytest.raises(ValueError):
        herd(K, 51)


# --------------------------------------------------------------------------- the command line

GPB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "examples", "sutherland", "GPB.sdf")


def test_pick_writes_the_choice_in_order(tmp_path, capsys):
    import csv

    from rdkit import Chem

    from cmfqsar.cli import main

    out = os.path.join(str(tmp_path), "p.csv")
    sdf = os.path.join(str(tmp_path), "p.sdf")
    main(["pick", GPB, "--n", "6", "--charges", "file", "--out", out, "--sdf-out", sdf])
    text = capsys.readouterr().out
    assert "6 molecules by herding" in text and "random choices of 6" in text
    with open(out, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [int(r["rank"]) for r in rows] == list(range(1, 7))
    mmd = [float(r["mmd2_of_the_first"]) for r in rows]
    assert mmd[-1] < mmd[0]
    picked = [m for m in Chem.SDMolSupplier(sdf)]
    assert [m.GetProp("pick_rank") for m in picked] == [str(i) for i in range(1, 7)]
    assert [m.GetProp("_Name") for m in picked] == [r["name"] for r in rows]


def test_pick_continues_from_molecules_already_made(tmp_path, capsys):
    import csv

    from rdkit import Chem

    from cmfqsar.cli import main

    mols = [m for m in Chem.SDMolSupplier(GPB, removeHs=False)]
    made, lib = os.path.join(str(tmp_path), "made.sdf"), os.path.join(str(tmp_path), "lib.sdf")
    with Chem.SDWriter(made) as w:
        for m in mols[:10]:
            w.write(m)
    with Chem.SDWriter(lib) as w:
        for m in mols[10:]:
            w.write(m)
    out = os.path.join(str(tmp_path), "p.csv")
    main(["pick", lib, "--n", "4", "--charges", "file", "--made", made, "--out", out])
    with open(out, newline="") as fh:
        chosen = [r["name"] for r in csv.DictReader(fh)]
    assert len(chosen) == 4
    assert not set(chosen) & {m.GetProp("_Name") for m in mols[:10]}
    with pytest.raises(SystemExit, match="--made"):
        main(["pick", lib, "--n", "4", "--charges", "file", "--made", made, "--method", "maxmin"])
