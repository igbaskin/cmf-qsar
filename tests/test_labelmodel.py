"""Tests for benchmarks/run_labelmodel.py: gate (b) of plan item BT9.

The models take their kernels from descriptors and shells and never from a label, and the
comparison resamples whole solvent mixtures. Those two are what a mistake would spoil without
any sign, so they are tested first.
"""
import copy
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import run_diffusion as RD                                                   # noqa: E402
import run_labelmodel as LM                                                  # noqa: E402


@pytest.fixture(scope="module")
def rows():
    return LM.series()


def test_the_series_is_the_62_formulations_in_the_order_of_item_bt5(rows):
    assert len(rows) == 62
    assert {r["id"] for r in rows} == set(RD.ALL_FORMULATION_RUNS)
    assert [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    for r in rows:
        assert r["sigma"] > 0 and r["salt"] > 0 and r["measured_sigma"] > 0
        assert r["molarity"] > 0 and r["starts"] in (1, 2)
    assert sum(r["starts"] == 2 for r in rows) == 1          # the reference electrolyte only


def test_the_labels_are_logarithms_of_the_simulated_quantities(rows):
    assert LM.label_vector(rows, "ln_sigma") == pytest.approx(
        np.log([r["sigma"] for r in rows]))
    assert LM.label_vector(rows, "ln_D") == pytest.approx(np.log([r["salt"] for r in rows]))


def test_a_kernel_never_reads_a_label(rows):
    """Change every simulated label: the kernels of the models must stay the same."""
    changed = copy.deepcopy(rows)
    for r in changed:
        r["sigma"] *= 3.0
        r["salt"] *= 0.2
    a = LM.model_kernels(rows, "ln_sigma")
    b = LM.model_kernels(changed, "ln_sigma")
    for model in LM.MODELS:
        assert set(a[model]) == set(b[model])
        for name in a[model]:
            assert np.array_equal(a[model][name], b[model][name]), (model, name)


def test_the_kernels_are_square_symmetric_and_without_negative_directions(rows):
    kernels = LM.model_kernels(rows, "ln_D")
    assert set(kernels) == set(LM.MODELS)
    for model, group in kernels.items():
        for name, K in group.items():
            assert K.shape == (62, 62)
            assert np.allclose(K, K.T, atol=1e-9)
            assert np.linalg.eigvalsh(K).min() > -1e-6, (model, name)


def test_the_reference_reads_the_concentration_only_for_the_diffusion_label(rows):
    d = LM.model_kernels(rows, "ln_D")["reference"]["line"]
    s = LM.model_kernels(rows, "ln_sigma")["reference"]["line"]
    assert not np.allclose(d, s)


def test_the_paired_interval_resamples_whole_groups():
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    a = np.array([0.3, 0.3, 0.2, 0.2, 0.4, 0.4, 0.3, 0.3])
    iv = LM.paired_group_interval(a, a - 0.1, groups)
    assert iv["mean"] == pytest.approx(0.1) and iv["low"] == pytest.approx(0.1)
    assert iv["excludes_zero"]
    same = LM.paired_group_interval(a, a, groups)
    assert same["mean"] == 0.0 and not same["excludes_zero"]
    # a gain in one group of four only has an interval that includes zero
    b = a.copy()
    b[:2] -= 0.3
    part = LM.paired_group_interval(a, b, groups)
    assert part["mean"] > 0 and part["low"] <= 0


def test_the_models_are_read_on_unseen_mixtures_and_the_composition_beats_the_molality(rows):
    y = LM.label_vector(rows, "ln_D")
    kernels = LM.model_kernels(rows, "ln_D")
    groups = [str(r["composition"]) for r in rows]
    out = LM.run_models(rows, y, kernels, groups, models=("conditions", "composition"))
    assert out["composition"]["mae"] < out["conditions"]["mae"]
    for rec in out.values():
        assert len(rec["abs_errors"]) == 62 and len(rec["mae_by_seed"]) == len(LM.SEEDS)
        assert np.isfinite(rec["z_rms"]) and rec["z_rms"] > 0


def test_the_gate_reading_is_the_one_that_was_fixed():
    assert LM.GATE_MODEL == "shells-free"
    assert (LM.Z_LOW, LM.Z_HIGH) == (0.8, 1.25)
    assert LM.SEEDS == (0, 1, 2, 3, 4)
