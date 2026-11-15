"""The reference of the bound: the distance to the nearest training molecule in place of the
posterior standard deviation (`benchmarks/run_bound.py --reference`)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

BD = pytest.importorskip("run_bound")


def test_the_distance_is_rescaled_to_the_mean_deviation():
    rng = np.random.default_rng(0)
    yp, sd, nearest = rng.normal(size=20), rng.uniform(0.2, 0.6, 20), rng.uniform(1.0, 3.0, 20)
    s = BD.reference_scores(yp, sd, nearest, betas=(2.0,))
    assert np.array_equal(s["prediction"], yp)
    assert np.allclose(s["standard deviation, beta 2"], yp - 2.0 * sd)
    penalty = (yp - s["nearest distance, beta 2"]) / 2.0
    assert penalty.mean() == pytest.approx(sd.mean())          # the same mean caution
    assert np.allclose(penalty / penalty.mean(), nearest / nearest.mean())


def test_the_paired_interval_over_the_sets():
    v = BD.paired_over_sets([0.9, 0.8, 0.7, 0.95], [0.85, 0.82, 0.6, 0.9])
    assert v["mean"] == pytest.approx(np.mean([0.05, -0.02, 0.1, 0.05]))
    assert v["low"] <= v["mean"] <= v["high"]
    assert v["ahead"] == 3 and v["n_sets"] == 4


def test_the_blocks_of_one_kernel_reproduce_a_model_fitted_on_the_fold():
    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import FieldSet, build_fields

    ds = load_dataset(os.path.join(BD.SUTHERLAND, "THR.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.flatnonzero(np.isfinite(y))
    mols = [ds.mols[i] for i in keep][:40]
    names = [ds.names[i] for i in keep][:40]
    y = y[keep][:40]
    fs = FieldSet.create(physchem=BD.FIELDS, alpha=BD.ALPHA, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    model, F = BD.field_kernels_of(mf, fs)
    order = np.random.default_rng(1).permutation(len(y))
    test, train = order[:8], np.sort(order[8:])
    yp, sd, nearest = BD.gp_on_blocks(BD.joint_for(model, F, train), y, train, test)
    _scores, sd_m, yp_m = BD._fit_and_rank(mf, y, train, test, fs, (0.0,))
    assert np.allclose(yp, yp_m, atol=1e-8) and np.allclose(sd, sd_m, atol=1e-8)
    assert np.all(nearest >= 0)
    # the scale of a field depends on the training part when the field is zero on some
    # molecules, which is why it is computed again for every fold
    whole = BD.joint_for(model, F, np.arange(len(y)))
    fold = BD.joint_for(model, F, train)
    zero_somewhere = [f for f in range(F.shape[0]) if np.any(np.diag(F[f]) == 0)]
    if zero_somewhere:
        assert not np.allclose(whole, fold)


def test_the_section_of_the_table(tmp_path):
    assert BD.reference_section(str(tmp_path)) == []
    pair = {"mean": 0.02, "low": 0.01, "high": 0.03, "ahead": 7, "n_sets": 8}
    summary = {"folds": 200, "sets": ["A"] * 8, "sd_vs_distance": 0.78,
               "mean": {"prediction": {"bedroc": 0.92, "auc": 0.86}},
               "pairs": {"standard deviation, beta 2 minus prediction": {"bedroc": pair, "auc": pair}}}
    (tmp_path / BD.REFERENCE_JSON).write_text(json.dumps(
        {"seeds": 5, "summary": {"random": summary}}))
    text = "\n".join(BD.reference_section(str(tmp_path)))
    assert "## The reference: the distance to the nearest training molecule" in text
    assert "| prediction | 0.920 | 0.860 |" in text
    assert "| standard deviation, beta 2 minus prediction | +0.020 [+0.010, +0.030] | 7 of 8 |" in text
