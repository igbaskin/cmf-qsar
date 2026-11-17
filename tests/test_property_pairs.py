"""The reference stage of the four physical-property benchmarks (benchmarks/run_property_pairs.py)."""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
sys.path.insert(0, os.path.join(HERE, ".."))

import run_property_pairs as PP  # noqa: E402


@pytest.mark.parametrize("prop", PP.PROPERTIES)
def test_every_test_has_a_title_and_a_part(prop):
    b = PP.Bench(prop)
    assert len(b.tests) == len(set(b.tests)) >= 3
    for t in b.tests:
        assert b.titles[t]
        assert b.part(t) in b.sets or b.part(t) in b.mod.load.__globals__.get("PARTS", b.sets)


def _toy(prop, n_per_part=40, seed=0):
    """A small data set with the parts of one benchmark and a linear signal."""
    b = PP.Bench(prop)
    parts = sorted(set(b.sets) | {b.part(t) for t in b.tests}
                   | ({"physprop_train"} if prop in ("boiling", "vapour") else set()))
    rng = np.random.RandomState(seed)
    X = rng.randn(n_per_part * len(parts), 3)
    y = X @ np.array([1.0, -0.5, 0.2]) + 0.1 * rng.randn(len(X))
    idx = {p: np.arange(k * n_per_part, (k + 1) * n_per_part) for k, p in enumerate(parts)}
    return b, X, y, idx


@pytest.mark.parametrize("prop", PP.PROPERTIES)
def test_the_predictions_cover_every_test_and_its_compounds(prop):
    b, X, y, idx = _toy(prop)
    blocks = b.splits(idx)
    m = b.Model("toy")
    m.add("linear", X @ X.T, y, blocks)
    preds, scores = b.predictions(m, y, blocks, idx)
    assert set(preds) == set(b.tests) == set(scores)
    for t in b.tests:
        assert preds[t].shape == (len(idx[b.part(t)]),)
        r = scores[t]
        value = r["q2_mean"] if "q2_mean" in r else r["r2"]
        assert value > 0.9                              # a linear kernel finds a linear signal


def test_a_missing_benchmark_cache_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(PP, "CACHE", str(tmp_path))
    with pytest.raises(SystemExit, match="run benchmarks/run_logp.py first"):
        PP.stored("logp")


def test_the_report_has_one_section_per_property(tmp_path, monkeypatch):
    monkeypatch.setattr(PP, "RESULTS", str(tmp_path))
    row = {"q2_mean": 0.9, "rmse_mean": 0.5}
    payload = {"logp": {"tests": ["huuskonen"], "titles": {"huuskonen": "Huuskonen, CV"},
                        "models": {m: {"huuskonen": row} for m in PP.MODELS},
                        "comparisons": [{"model": PP.NEW, "against": "eight counts",
                                         "test": "huuskonen", "mean": 0.01, "lo": -0.01,
                                         "hi": 0.03}]}}
    text = PP.report(payload)
    assert "### logP" in text and "| topological + eight counts | +0.900 (0.5) |" in text
    assert "| topological + eight counts minus eight counts | Huuskonen, CV | +0.010" in text
    assert os.path.isfile(os.path.join(str(tmp_path), "table.md"))
