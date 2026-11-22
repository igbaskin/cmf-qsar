"""The soil sorption set and its benchmark (examples/koc, benchmarks/run_koc.py)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
sys.path.insert(0, os.path.join(HERE, ".."))

import run_koc as K  # noqa: E402


def test_the_file_holds_the_two_parts_of_the_opera_paper():
    mols, y, idx = K.load()
    with open(os.path.join(K.DATA, "provenance.json"), encoding="utf-8") as fh:
        prov = json.load(fh)["koc"]
    assert len(mols) == prov["embedded"] == 719
    assert len(idx["train"]) == prov["train_embedded"] == 536
    assert len(idx["test"]) == prov["test_embedded"] == 183
    assert not set(idx["train"]) & set(idx["test"])
    assert list(idx["all"]) == list(range(len(mols)))
    assert y.min() >= 0.0 and y.max() <= 6.5
    keys = [m.GetProp("inchikey14") for m in mols]
    assert len(set(keys)) == len(keys)                     # no structure twice
    assert all(m.GetNumConformers() == 1 and m.GetConformer().Is3D() for m in mols)


def test_the_quick_load_keeps_both_parts():
    mols, y, idx = K.load(quick=20)
    assert len(mols) == 40 and len(idx["train"]) == len(idx["test"]) == 20
    assert len(y) == 40


def test_the_blocks_are_cross_validation_on_two_sets_and_one_external_test():
    idx = {"train": np.arange(0, 30), "test": np.arange(30, 40), "all": np.arange(40)}
    blocks = K.splits(idx)
    assert len(blocks) == 2 * K.REPEATS * K.L.FOLDS + 1
    name, rep, tr, te = blocks[-1]
    assert name == "opera_test" and list(tr) == list(idx["train"]) and list(te) == list(idx["test"])
    for name, rep, tr, te in blocks[:-1]:
        assert not set(tr) & set(te)
        assert set(tr) | set(te) == set(idx[name])


def test_a_linear_signal_is_found_on_every_test():
    rng = np.random.RandomState(0)
    X = rng.randn(120, 3)
    y = X @ np.array([1.0, -0.5, 0.2]) + 0.05 * rng.randn(120)
    idx = {"train": np.arange(0, 90), "test": np.arange(90, 120), "all": np.arange(120)}
    blocks = K.splits(idx)
    m = K.Model("toy")
    m.add("linear", X @ X.T, y, blocks)
    cv, ext, chosen = m.select(y, blocks)
    assert set(chosen) == set(K.TESTS)
    assert K.r2_score(y[idx["test"]], ext["opera_test"]) > 0.95
    for name in K.SETS:
        ii = idx[name]
        assert K.r2_score(y[ii], cv[name][0, ii]) > 0.95


def test_the_report_names_every_model_and_the_published_model(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "RESULTS", str(tmp_path))
    row = {"q2_mean": 0.7, "rmse_mean": 0.6}
    ext = {"r2": 0.7, "rmse": 0.6}
    payload = {"compounds": {"train": 536, "test": 183, "all": 719},
               "y": {p: {"sd": 1.2} for p in ("train", "test", "all")},
               "opera": K.OPERA,
               "models": {m: {"train": row, "all": row, "opera_test": ext} for m in K.MODELS},
               "comparisons": [{"model": K.MODELS[0], "against": K.MODELS[3],
                                "test": "opera_test", "mean": 0.1, "lo": 0.0, "hi": 0.2}]}
    text = K.report(payload)
    for m in K.MODELS:
        assert f"| {m} |" in text
    assert "OPERA model, 12 descriptors" in text and "+0.710 (0.61)" in text
    assert os.path.isfile(os.path.join(str(tmp_path), "table.md"))
