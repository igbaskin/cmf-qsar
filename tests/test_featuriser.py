"""CMF as a featuriser (plan item ND1): the vector, the learners' inputs and the command.

* The whitened landmark rows have the kernel as their inner products, and a ridge regression
  on them is the kernel ridge regression.
* Joining blocks gives each block the same weight, standardises the collectors column by
  column, and keeps the relative column lengths of the CMF vector.
* The paired difference counts a molecule once, however many test parts hold it.
* The stored device files of a split agree with each other and with the data set.
* ``cmf lowrank --features`` writes rows whose inner products are the low-rank kernel, and
  ``--features-for`` gives the same molecule the same row.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

pytest.importorskip("rdkit")

import run_featuriser as F                                              # noqa: E402
from cmfqsar.lowrank import whitener                                    # noqa: E402
from cmfqsar.model import krr_fit                                       # noqa: E402

GPB = os.path.join(ROOT, "examples", "sutherland", "GPB.sdf")


def _kernel(n=30, d=6, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, d))
    return X @ X.T, rng.normal(size=n)


def test_the_vector_has_the_kernel_as_inner_products_and_its_ridge_is_the_kernel_ridge():
    K, y = _kernel()
    tr = np.arange(20)
    Phi = K[:, tr] @ whitener(K[np.ix_(tr, tr)])
    assert np.allclose(Phi[tr] @ Phi[tr].T, K[np.ix_(tr, tr)], atol=1e-8)
    assert np.allclose(Phi @ Phi[tr].T, K[:, tr], atol=1e-8)       # the rows of new molecules
    lam = 0.3
    a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam)
    kernel_pred = K[:, tr] @ a + b
    # the ridge of the benchmark on one unscaled block: centred columns, the mean as the bias
    Xc = Phi - Phi[tr].mean(axis=0)
    w = np.linalg.solve(Xc[tr].T @ Xc[tr] + lam * np.eye(Xc.shape[1]),
                        Xc[tr].T @ (y[tr] - y[tr].mean()))
    assert np.allclose(Xc @ w + y[tr].mean(), kernel_pred, atol=1e-6)


def test_learn_ridge_chooses_on_the_validation_rows_and_predicts_every_row():
    K, _ = _kernel(60, 5, 1)
    rng = np.random.RandomState(2)
    X = rng.normal(size=(60, 5))
    y = X @ [1.0, -1.0, 0.5, 0.0, 2.0] + 0.1 * rng.normal(size=60)
    split = {"train": np.arange(0, 40), "valid": np.arange(40, 50), "test": np.arange(50, 60)}
    pred, chosen = F.learn_ridge(F.scaled_blocks([(X, False)], split["train"]), y, split)
    assert pred.shape == (60,) and chosen["lam"] in F.LAMS
    s = F.scores(y, pred, split["test"])
    assert s["r2"] > 0.95 and len(s["abs_errors"]) == 10


def test_joined_blocks_have_equal_weight_and_the_vector_keeps_its_column_lengths():
    rng = np.random.RandomState(0)
    tr = np.arange(40)
    phi = rng.normal(size=(50, 4)) * [10.0, 3.0, 1.0, 0.1]         # the kernel's own weights
    coll = rng.normal(size=(50, 3)) * [100.0, 1.0, 0.01] + [5.0, -2.0, 0.0]
    const = np.hstack([coll, np.ones((50, 1))])                    # a collector that never moves
    X = F.scaled_blocks([(phi, False), (const, True)], tr)
    assert X.shape == (50, 4 + 3)                                  # the constant column is gone
    a, b = X[:, :4], X[:, 4:]
    for block in (a, b):
        assert np.mean(np.sum(block[tr] ** 2, axis=1)) == pytest.approx(1.0)
        assert np.allclose(block[tr].mean(axis=0), 0.0, atol=1e-12)
    sd = a[tr].std(axis=0)
    assert np.allclose(sd / sd[0], phi[tr].std(axis=0) / phi[tr].std(axis=0)[0])
    assert np.allclose(b[tr].std(axis=0), b[tr].std(axis=0)[0])    # every collector the same


def test_a_molecule_in_several_test_parts_counts_once():
    runs = [{"test": [0, 1, 2], "models": {"a": {"abs_errors": [1.0, 1.0, 1.0]},
                                           "b": {"abs_errors": [0.0, 0.0, 0.0]}}},
            {"test": [0, 3], "models": {"a": {"abs_errors": [3.0, 1.0]},
                                        "b": {"abs_errors": [0.0, 0.0]}}}]
    p = F.paired(runs, "a", "b", draws=200)
    assert p["molecules"] == 4
    assert p["mean"] == pytest.approx((2.0 + 1.0 + 1.0 + 1.0) / 4)
    assert p["excludes_zero"] and p["low"] > 0


def test_the_stored_device_files_agree_with_the_data_set():
    name, seed = "GPB", 0
    if not all(os.path.exists(F.device_file(name, m, seed)) for m in F.DEVICE_METHODS):
        pytest.skip("the device files are not there")
    names, _mols, y = F.load_set(name)
    dev = F.read_device(name, seed, names)
    split = dev["split"]
    every = np.sort(np.concatenate([split[k] for k in ("train", "valid", "test")]))
    assert np.array_equal(every, np.arange(len(y)))                # a partition of the set
    assert len(split["test"]) == round(0.15 * len(y)) or abs(
        len(split["test"]) - 0.15 * len(y)) < 1.0
    for m in F.DEVICE_METHODS:
        assert dev[m]["pred"].shape == (len(y),)
        assert dev[m]["collectors"].shape[0] == len(y)
        assert np.isfinite(dev[m]["collectors"]).all()


def test_the_command_writes_the_vector_and_gives_the_same_molecule_the_same_row(tmp_path):
    from cmfqsar.cli import main
    out = str(tmp_path / "rows.npz")
    main(["lowrank", GPB, "--landmarks", "molecules:30", "--alpha", "0.5", "--charges", "file",
          "--lams", "1", "--folds", "3", "--features", out, "--features-for", GPB])
    z = np.load(out)
    X = z["features"]
    assert X.shape[0] == 2 * 66 and list(z["source"][:66]) == ["data"] * 66
    assert list(z["source"][66:]) == ["other"] * 66
    assert np.allclose(X[:66], X[66:], atol=1e-10)
    # the inner products are a kernel: symmetric, with the cosine-normalised diagonal near the
    # number of fields the molecules carry
    G = X[:66] @ X[:66].T
    assert np.allclose(G, G.T) and np.all(np.diag(G) > 0)
    csv_out = str(tmp_path / "rows.csv")
    main(["lowrank", GPB, "--landmarks", "molecules:30", "--alpha", "0.5", "--charges", "file",
          "--lams", "1", "--folds", "3", "--features", csv_out])
    with open(csv_out, encoding="utf-8") as fh:
        head = fh.readline().strip().split(",")
        first = fh.readline().strip().split(",")
    assert head[:4] == ["name", "source", "measured", "f0"] and len(first) == len(head)
    assert np.allclose([float(v) for v in first[3:]], X[0], rtol=1e-6, atol=1e-8)
