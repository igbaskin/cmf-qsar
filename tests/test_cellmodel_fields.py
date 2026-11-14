"""The benchmark ``benchmarks/run_cellmodel_fields.py``: the cell model of plan item MS9 with a
field model in every channel.

The channels need the cached rows of three other benchmarks and the cell model needs PyBaMM, so
what is tested here is everything between them: the fit on all labels, the mixed kernel, the
cache of the field rows, the joint draws, and the table.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import run_cellmodel as RC  # noqa: E402
import run_cellmodel_fields as F  # noqa: E402
from cmfqsar.interfaces import CellParameter, CellParameterSet  # noqa: E402


def test_every_predicted_channel_has_a_field_model_and_the_last_two_are_shared():
    labels = [label for label, _ in F.CHANNELS]
    assert labels == [label for label, _ in RC.CHANNELS]
    by = dict(F.CHANNELS)
    assert by["diffusion coefficient"] is F.barrier_fields
    assert by["open-circuit voltage"] is F.voltage_fields
    assert by["electrolyte conductivity"] is F.electrolyte_fields
    # the tortuosity channel was a field kernel from the start, and the exchange current is
    # not predicted in either benchmark
    assert by["electrode tortuosity"] is RC.microstructure_channel
    assert by["exchange current"] is RC.exchange_channel
    assert set(F.MODELS) >= set(F.ERROR_UNITS)
    assert F.RESULTS != RC.RESULTS


def test_the_fit_on_all_labels_chooses_the_kernel_that_predicts_held_out_groups():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.05 * rng.normal(size=60)
    good = RC.unit_kernel(X @ X.T)
    noise = rng.normal(size=(60, 8))
    bad = RC.unit_kernel(noise @ noise.T)
    groups = [i // 3 for i in range(60)]
    name, lam, s = F.fit_whole({"bad": bad, "good": good}, y, groups)
    assert name == "good" and lam in RC.LAMS
    pred, sd = s.predict(good, np.diag(good))
    assert np.sqrt(np.mean((pred - y) ** 2)) < 0.2
    assert np.all(sd > 0)
    # a new point is predicted through its kernels with the training points
    x_new = rng.normal(size=(5, 4))
    d = float(np.mean(np.diag(X @ X.T)))
    p_new, _ = s.predict(x_new @ X.T / d, np.sum(x_new ** 2, axis=1) / d)
    assert p_new == pytest.approx(x_new @ np.array([1.0, -2.0, 0.5, 0.0]), abs=0.3)


def test_the_mixed_kernel_has_a_unit_diagonal_and_the_stated_weights():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(6, 3))
    field, condition = 4.0 * A @ A.T + 0.1 * np.eye(6), np.exp(-np.abs(np.subtract.outer(
        np.arange(6.0), np.arange(6.0))))
    K = F.mixed_kernel(field, condition, 0.5)
    assert np.diag(K) == pytest.approx(np.ones(6))
    d = np.sqrt(np.diag(field))
    assert K[0, 1] == pytest.approx(0.5 * field[0, 1] / (d[0] * d[1]) + 0.5 * condition[0, 1])
    assert F.mixed_kernel(field, condition, 0.0)[0, 1] == pytest.approx(
        field[0, 1] / (d[0] * d[1]))
    assert F.mixed_kernel(field, condition, 1.0) == pytest.approx(condition)
    with pytest.raises(ValueError, match="between 0 and 1"):
        F.mixed_kernel(field, condition, 1.5)


def test_field_rows_are_kept_and_built_again_when_the_names_change(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "CACHE", str(tmp_path))
    calls = []

    def build():
        calls.append(1)
        return np.arange(6.0).reshape(3, 2) + len(calls)

    first = F.cached_rows("rows", ["a", "b", "c"], build)
    again = F.cached_rows("rows", ["a", "b", "c"], build)
    assert len(calls) == 1 and np.array_equal(first, again)
    other = F.cached_rows("rows", ["a", "b", "d"], build)       # another set of structures
    assert len(calls) == 2 and not np.array_equal(first, other)


def _set():
    pset = CellParameterSet(base="made up")
    pset.add(CellParameter(name="a", value=1.0, sd=0.1, kind="scale", distribution="lognormal"))
    pset.add(CellParameter(name="b", value=0.3, sd=1.0, kind="shift", low=0.0, high=0.5))
    pset.add(CellParameter(name="c", value=2.0, distribution="sample",
                           sample=np.array([1.0, 2.0, 3.0])))
    return pset


def test_all_parameters_are_drawn_together_and_each_from_its_own_deviation():
    values = F.joint_values(_set(), 400, seed=3)
    assert list(values) == ["a", "b", "c"]
    assert all(len(v) == 400 for v in values.values())
    assert np.all(values["a"] > 0)
    assert values["b"].min() >= 0.0 and values["b"].max() <= 0.5
    assert set(np.unique(values["c"])) == {1.0, 2.0, 3.0}
    # the same seed gives the same draws, and the parameters are not copies of each other
    same = F.joint_values(_set(), 400, seed=3)
    assert all(np.array_equal(values[k], same[k]) for k in values)
    assert abs(np.corrcoef(np.log(values["a"]), values["c"])[0, 1]) < 0.2


def test_the_error_of_a_channel_is_read_from_its_record():
    barrier = {"training": {"mae": 0.29, "factor": 1.1, "shared": 0.3}}
    assert F.channel_error("Positive particle diffusivity [m2.s-1]", barrier) == {
        "mae": 0.29, "mean": None, "z_rms": 1.1}
    voltage = {"mae": 0.65, "mae_of_the_mean": 1.2, "z_rms": 0.97}
    assert F.channel_error("Positive electrode OCP [V]", voltage)["mean"] == 1.2
    tortuosity = {"mae_log": 0.05, "mae": 9.0, "mae_of_the_mean": 0.3, "z_rms": 1.04}
    assert F.channel_error("Positive electrode Bruggeman coefficient (electrolyte)",
                           tortuosity)["mae"] == 0.05
    # the exchange current is not predicted, so it has no error
    assert F.channel_error("Positive electrode exchange-current density [A.m-2]", {}) is None


def _row(name, predicted=True, **over):
    row = {"name": name, "model": F.MODELS[name][0], "other_model": F.MODELS[name][1],
           "predicted": predicted, "value": 1.0, "units": "", "kind": "scale",
           "published": 1.0,
           "interval": [0.8, 1.25], "calibrated": True if predicted else None,
           "error": {"mae": 0.1, "mean": 0.3, "z_rms": 1.02} if predicted else None,
           "chosen": None, "capacity": [4.9, 5.0], "refused": 0.0, "holds": True,
           "variance": 1e-4, "share": 0.25,
           "other": {"error": {"mae": 0.08, "mean": 0.3, "z_rms": 1.0} if predicted else None,
                     "value": 0.9, "capacity": [4.95, 5.0], "holds": True}}
    row.update(over)
    return row


def test_the_table_sets_the_field_models_beside_the_models_of_the_other_table():
    names = list(F.MODELS)
    rows = [_row(names[0], kind="value", value=3.9e-14, interval=[3.1e-21, 1e-10],
                 published=4e-15, capacity=[0.61, 5.01], share=0.7),
            _row(names[1], kind="shift", published=0.0, holds=False, refused=0.055, share=0.2),
            _row(names[2]), _row(names[3], calibrated=False),
            _row(names[4], predicted=False, share=0.0)]
    res = {"base": "Chen2020", "model": "SPMe", "published": 4.9908, "cycles": 10,
           "rows": rows, "compared": True,
           "joint": {"draws": 200, "runs_that_completed": 190, "refused": 0.05,
                     "interval_of_the_last_cycle": [0.2, 5.3],
                     "median_of_the_last_cycle": 1.1,
                     "holds_the_published_curve": {"all_inside": True}}}
    text = "\n".join(F.table_lines(res))
    assert text.startswith("# Item MS9 with a field model in every channel")
    assert ("| Positive particle diffusivity [m2.s-1] | kernel ridge regression on the "
            "invariant field of the hop | 0.100 | 1.02 | random forest on the same field rows "
            "| 0.080 | eV |") in text
    # the parameter that is not predicted is in the parameters and not among the models
    assert text.count("not predicted: a stated spread") == 0
    assert ("| Positive electrode exchange-current density [A.m-2] | 1 | 0.8 to 1.2 | "
            "multiplies it | 1 | 0.9 | -- |") in text
    assert ("| Positive particle diffusivity [m2.s-1] | 3.9e-14 | 3.1e-21 to 1e-10 | "
            "replaces it | 4e-15 | 0.9 | yes |") in text
    assert ("| Positive electrode OCP [V] | 4.900 to 5.000 | no | 5.5 % | 20.0 % | "
            "4.950 to 5.000 | yes |") in text
    assert "190 runs completed, and the cell model refused 5.0 % of the draws" in text
    assert "between 0.200 and 5.300 A.h" in text and "is inside this interval" in text
    assert "It does not show that the intervals are useful" in text
    # without the joint draws and without the other benchmark the table is still written
    alone = dict(res, joint=None, rows=[dict(r, other=None) for r in rows])
    text = "\n".join(F.table_lines(alone))
    assert "## All parameters drawn together" not in text and "| -- | -- |" in text


def test_a_parameter_is_printed_in_its_own_terms():
    sample = CellParameter(name="d", value=1.0, distribution="sample",
                           sample=np.array([1.0, 2.0, 10.0]), published=4e-15)
    assert F.central_value(sample) == 2.0 and F.published_value(sample) == 4e-15
    scale = CellParameter(name="k", value=0.9, sd=0.1, kind="scale", distribution="lognormal")
    assert F.central_value(scale) == 0.9 and F.published_value(scale) == 1.0
    assert F.published_value(CellParameter(name="v", value=0.3, kind="shift")) == 0.0
    assert F.published_value(CellParameter(name="x", value=0.3)) is None


def test_the_report_needs_a_result(tmp_path):
    with pytest.raises(SystemExit, match="run this benchmark first"):
        F.report(str(tmp_path))
