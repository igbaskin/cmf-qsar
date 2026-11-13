"""A parameter set for a cell model, with its deviations (plan item MS9).

No test needs a download and none needs the benchmark's data.  The tests of the writer and of
the sampler's arithmetic need no cell model either; the few that run one are skipped when PyBaMM
is not installed.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from cmfqsar import cellmodel as CM  # noqa: E402
from cmfqsar.interfaces import (BOLTZMANN_EV, CellParameter, CellParameterSet,  # noqa: E402
                                arrhenius_diffusivity, bruggeman_from_conductivity_ratio,
                                conductivity_scale_from_log10, diffusivity_from_barrier,
                                ladder_to_ocp_curve, ocp_shift)

import run_cellmodel as R  # noqa: E402


# --------------------------------------------------------------------------- #
# The conversions
# --------------------------------------------------------------------------- #
def test_a_barrier_of_zero_leaves_the_prefactor():
    """With no barrier the diffusion coefficient is the random-walk prefactor itself."""
    d = arrhenius_diffusivity(0.0, 3.0, dimension=3)
    assert d == pytest.approx((3.0e-10) ** 2 * 1e13 / 6.0)


def test_the_barrier_enters_as_an_exponential():
    """Two barriers a thermal energy apart give diffusion coefficients a factor e apart."""
    kt = BOLTZMANN_EV * 298.15
    low = arrhenius_diffusivity(0.5, 3.0)
    high = arrhenius_diffusivity(0.5 + kt, 3.0)
    assert low / high == pytest.approx(np.e, rel=1e-9)


def test_more_directions_divide_the_prefactor():
    assert (arrhenius_diffusivity(0.4, 3.0, dimension=2)
            == pytest.approx(1.5 * arrhenius_diffusivity(0.4, 3.0, dimension=3)))


def test_a_deviation_of_a_barrier_becomes_a_deviation_of_a_logarithm():
    """The deviation of the logarithm of the diffusion coefficient is the barrier's over kT."""
    p = diffusivity_from_barrier(0.5, 3.0, "D", sd=0.1)
    assert p.distribution == "lognormal"
    assert p.sd == pytest.approx(0.1 / (BOLTZMANN_EV * 298.15))
    assert p.value == pytest.approx(arrhenius_diffusivity(0.5, 3.0))
    assert p.units == "m2.s-1" and p.kind == "value"


def test_drawn_barriers_are_carried_through_one_by_one():
    """A set of drawn barriers gives the set of their diffusion coefficients, not a fitted one."""
    barriers = np.array([0.3, 0.4, 0.5, 0.6, 0.9])
    p = diffusivity_from_barrier(barriers, 2.9, "D", dimension=2)
    assert p.distribution == "sample"
    expected = sorted(arrhenius_diffusivity(b, 2.9, dimension=2) for b in barriers)
    assert sorted(p.sample) == pytest.approx(expected)
    assert p.value == pytest.approx(arrhenius_diffusivity(0.5, 2.9, dimension=2))


def test_a_logarithmic_difference_of_one_is_a_factor_of_ten():
    p = conductivity_scale_from_log10(1.5, 0.5, "k", sd=0.1)
    assert p.kind == "scale" and p.value == pytest.approx(10.0)
    assert p.sd == pytest.approx(np.log(10.0) * 0.1)


def test_a_voltage_prediction_becomes_a_shift():
    p = ocp_shift(3.47, 3.80, "OCP", sd=0.5)
    assert p.kind == "shift" and p.value == pytest.approx(-0.33)
    assert p.published == pytest.approx(3.80) and p.sd == pytest.approx(0.5)


def test_the_bruggeman_exponent_inverts_the_power_law():
    """An exponent put through the power law comes back out of the conversion."""
    eps, b = 0.335, 1.75
    p = bruggeman_from_conductivity_ratio(eps ** b, eps, "b")
    assert p.value == pytest.approx(b)


def test_a_ratio_outside_its_range_is_refused():
    with pytest.raises(ValueError, match="between zero and one"):
        bruggeman_from_conductivity_ratio(1.4, 0.3, "b")
    with pytest.raises(ValueError, match="porosity"):
        bruggeman_from_conductivity_ratio(0.2, 1.5, "b")


def test_a_ladder_becomes_a_staircase():
    x, v = ladder_to_ocp_curve([0.0, 0.5, 1.0], [4.1, 3.9])
    assert list(x) == [0.0, 0.5, 0.5, 1.0]
    assert list(v) == [4.1, 4.1, 3.9, 3.9]


def test_a_ladder_with_the_wrong_number_of_points_is_refused():
    with pytest.raises(ValueError, match="stoichiometries"):
        ladder_to_ocp_curve([0.0, 1.0], [4.1, 3.9])


# --------------------------------------------------------------------------- #
# The parameter record
# --------------------------------------------------------------------------- #
def test_the_three_kinds_change_a_number_in_three_ways():
    for kind, expected in (("value", 7.0), ("scale", 14.0), ("shift", 9.0)):
        p = CellParameter(name="x", value=7.0, kind=kind)
        assert p.apply(2.0, 7.0) == pytest.approx(expected)


def test_a_function_can_be_scaled_and_shifted_but_not_replaced():
    f = (lambda c, t: 3.0)
    assert CellParameter(name="x", value=2.0, kind="scale").apply(f, 2.0)(1.0, 2.0) == 6.0
    assert CellParameter(name="x", value=2.0, kind="shift").apply(f, 2.0)(1.0, 2.0) == 5.0
    with pytest.raises(TypeError, match="function"):
        CellParameter(name="x", value=2.0, kind="value").apply(f, 2.0)


def test_drawing_follows_the_distribution_and_keeps_the_limits():
    rng = np.random.default_rng(0)
    normal = CellParameter(name="x", value=5.0, sd=1.0).draw(20000, rng)
    assert np.mean(normal) == pytest.approx(5.0, abs=0.05)
    assert np.std(normal) == pytest.approx(1.0, abs=0.05)
    log = CellParameter(name="x", value=5.0, sd=0.5, distribution="lognormal").draw(20000, rng)
    assert np.median(log) == pytest.approx(5.0, rel=0.05)
    assert np.all(log > 0)
    kept = CellParameter(name="x", value=0.1, sd=1.0, low=0.0, high=0.2).draw(500, rng)
    assert kept.min() >= 0.0 and kept.max() <= 0.2


def test_a_drawn_sample_is_resampled_from_itself():
    values = np.array([1.0, 2.0, 3.0])
    p = CellParameter(name="x", value=2.0, distribution="sample", sample=values)
    drawn = p.draw(200, np.random.default_rng(0))
    assert set(np.unique(drawn)).issubset({1.0, 2.0, 3.0})
    assert len(np.unique(drawn)) == 3


def test_what_the_record_refuses():
    with pytest.raises(ValueError, match="not a"):
        CellParameter(name="x", value=1.0, kind="multiply")
    with pytest.raises(ValueError, match="spread is"):
        CellParameter(name="x", value=1.0, distribution="uniform")
    with pytest.raises(ValueError, match="negative deviation"):
        CellParameter(name="x", value=1.0, sd=-1.0)
    with pytest.raises(ValueError, match="above zero"):
        CellParameter(name="x", value=0.0, distribution="lognormal")
    with pytest.raises(ValueError, match="at least"):
        CellParameter(name="x", value=1.0, distribution="sample", sample=[1.0])


@pytest.mark.parametrize("z_rms,expected", [(1.0, True), (0.8, True), (1.25, True),
                                            (0.7, False), (1.3, False), (None, False)])
def test_the_calibration_rule(z_rms, expected):
    """A deviation counts as meaning what it says between 0.8 and 1.25 (plan item MS1)."""
    assert CellParameter(name="x", value=1.0, calibration=z_rms).calibrated is expected


def test_the_set_is_written_and_read_back(tmp_path):
    pset = CellParameterSet(base="Chen2020", note="a test")
    pset.add(CellParameter(name="a", value=2.0, sd=0.5, units="m2.s-1", calibration=1.0))
    pset.add(CellParameter(name="b", value=1.0, sd=0.1, kind="scale", predicted=False))
    path = pset.write(str(tmp_path / "parameters.json"))
    back = json.load(open(path, encoding="utf-8"))
    assert back["base"] == "Chen2020" and len(back["parameters"]) == 2
    assert back["parameters"][0]["name"] == "a"
    assert back["parameters"][0]["calibrated"] is True
    assert back["parameters"][1]["predicted"] is False
    assert len(back["parameters"][0]["interval_95"]) == 2
    table = pset.write_table(str(tmp_path / "parameters.csv"))
    head, *rows = open(table, encoding="utf-8").read().splitlines()
    assert head.startswith("name,kind,value,sd")
    assert len(rows) == 2
    assert pset["a"].value == 2.0 and pset.names == ["a", "b"]
    with pytest.raises(KeyError, match="holds no parameter"):
        pset["c"]
    with pytest.raises(ValueError, match="already"):
        pset.add(CellParameter(name="a", value=3.0))


# --------------------------------------------------------------------------- #
# The sampler's arithmetic, without a cell model
# --------------------------------------------------------------------------- #
def runs_of(curves, failed=0):
    out = [CM.CellRun(capacity=np.asarray(c, float), cycles=len(c)) for c in curves]
    out += [CM.CellRun(capacity=np.zeros(0), cycles=len(curves[0]), ok=False, message="refused")
            for _ in range(failed)]
    return out


def test_the_interval_leaves_out_the_runs_that_did_not_complete():
    runs = runs_of([[5.0, 4.9], [4.0, 3.9], [3.0, 2.9]], failed=1)
    b = CM.band(runs, level=100.0)
    assert b["n"] == 3
    assert b["refused"] == pytest.approx(0.25)
    assert list(b["low"]) == pytest.approx([3.0, 2.9])
    assert list(b["high"]) == pytest.approx([5.0, 4.9])
    assert list(b["median"]) == pytest.approx([4.0, 3.9])


def test_an_interval_of_nothing_is_empty():
    b = CM.band(runs_of([[1.0]], failed=0)[:0] + runs_of([[1.0]], failed=2)[1:])
    assert b["n"] == 0 and len(b["low"]) == 0


def test_whether_the_interval_holds_a_curve():
    b = CM.band(runs_of([[5.0, 4.0], [3.0, 2.0]]), level=100.0)
    inside = CM.contains(b, [4.0, 3.0])
    assert inside["all_inside"] and inside["inside"] == 2
    assert inside["largest_gap"] == pytest.approx(0.0)
    outside = CM.contains(b, [6.0, 3.0])
    assert not outside["all_inside"] and outside["inside"] == 1
    assert outside["largest_gap"] == pytest.approx(1.0)


def test_the_shares_of_the_variance_add_up_to_one():
    shares = CM.variance_shares({"wide": runs_of([[1.0], [5.0], [9.0]]),
                                 "narrow": runs_of([[4.9], [5.0], [5.1]])})
    assert shares["wide"]["share"] + shares["narrow"]["share"] == pytest.approx(1.0)
    assert shares["wide"]["share"] > shares["narrow"]["share"]
    assert shares["wide"]["n"] == 3


def test_the_final_capacity_and_the_fade():
    run = CM.CellRun(capacity=np.array([5.0, 4.5, 4.0]), cycles=3)
    assert run.final == pytest.approx(4.0)
    assert run.fade == pytest.approx(0.8)
    empty = CM.CellRun(capacity=np.zeros(0), cycles=3, ok=False)
    assert not np.isfinite(empty.final)


def test_shrinking_a_sample_keeps_its_middle_and_its_sign():
    values = np.array([1e-20, 1e-18, 1e-16])
    same = CM._shrink_sample(values, 1.0)
    assert same == pytest.approx(values)
    tight = CM._shrink_sample(values, 0.5)
    assert np.all(tight > 0)
    assert np.median(tight) == pytest.approx(np.median(values))
    assert np.log(tight.max() / tight.min()) == pytest.approx(
        0.5 * np.log(values.max() / values.min()))
    flat = CM._shrink_sample(values, 0.0)
    assert flat == pytest.approx(np.full(3, np.median(values)))


def test_an_unknown_cell_model_is_refused():
    pytest.importorskip("pybamm")
    with pytest.raises(ValueError, match="one of"):
        CM.build_model("SPMx")


# --------------------------------------------------------------------------- #
# The benchmark's own machinery
# --------------------------------------------------------------------------- #
def test_the_folds_never_split_a_group():
    groups = [f"g{i // 3}" for i in range(30)]
    parts = R.group_parts(groups, 5)
    assert sorted(np.concatenate(parts)) == list(range(30))
    where = {}
    for k, part in enumerate(parts):
        for i in part:
            where.setdefault(groups[i], set()).add(k)
    assert all(len(v) == 1 for v in where.values())


def test_a_unit_kernel_has_a_mean_diagonal_of_one():
    K = np.array([[4.0, 1.0], [1.0, 6.0]])
    assert np.mean(np.diag(R.unit_kernel(K))) == pytest.approx(1.0)


def test_the_descriptor_kernels_hold_a_linear_and_several_widths():
    Z = np.random.default_rng(0).normal(size=(12, 3))
    kernels = R.descriptor_kernels(Z, gammas=(0.5, 2.0))
    assert set(kernels) == {"linear", "gaussian, width 0.5", "gaussian, width 2"}
    for K in kernels.values():
        assert K.shape == (12, 12)
        assert np.allclose(K, K.T)
        assert np.min(np.linalg.eigvalsh(K)) > -1e-8
    assert np.allclose(np.diag(kernels["gaussian, width 0.5"]), 1.0)


def test_the_fit_reports_a_calibrated_deviation_on_a_quantity_it_can_learn():
    """On data a linear model fits, the deviation of the fit means what it says."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(120, 4))
    y = Z @ np.array([1.0, -0.5, 0.25, 0.0]) + rng.normal(0.0, 0.2, 120)
    groups = [f"g{i // 2}" for i in range(120)]
    fit = R.nested_surrogate(R.descriptor_kernels(Z), y, groups)
    assert fit["mae"] < 0.5 * fit["mae_of_the_mean"]
    assert 0.7 < fit["z_rms"] < 1.4
    assert len(fit["chosen"]) == R.FOLDS


def test_the_table_is_written_from_a_stored_result(tmp_path):
    """The table writer runs on a result of the shape the benchmark stores."""
    pset = CellParameterSet(base="Chen2020")
    pset.add(CellParameter(name="A", value=1e-15, sd=2.0, distribution="lognormal",
                           units="m2.s-1", published=4e-15, calibration=1.05,
                           source="a test channel", note="a note", low=0.0))
    pset.add(CellParameter(name="B", value=1.0, sd=0.5, kind="scale", units="factor",
                           published=1.0, predicted=False, source="a stated spread"))
    per = {}
    for name in ("A", "B"):
        per[name] = {"interval_of_the_last_cycle": [1.0, 4.0],
                     "median_of_the_last_cycle": 2.5, "runs_that_completed": 10,
                     "refused": 0.0,
                     "holds_the_published_curve": {"cycles": 2, "inside": 2, "share": 1.0,
                                                   "largest_gap": 0.0, "all_inside": True},
                     "elasticity": {"elasticity": -0.5, "capacity": 5.0, "step": 0.1},
                     "fade": {}, "curve": {"low": [1.0, 1.0], "median": [2.5, 2.5],
                                           "high": [4.0, 4.0]}}
    result = {"base": "Chen2020", "model": "SPMe", "cycles": 2, "draws": 10,
              "published": {"first": 5.0, "final": 4.99, "fade": 0.998, "curve": [5.0, 4.99]},
              "parameters": pset.as_dict(),
              "channels": {"A": {"label": "a channel", "seconds": 1.0, "z_rms": 1.05,
                                 "mae": 0.1, "mae_of_the_mean": 0.4, "coverage_2sd": 0.95},
                           "B": {"label": "another", "seconds": 0.0}},
              "per_parameter": per,
              "variance_shares": {"A": {"share": 0.9}, "B": {"share": 0.1}},
              "tolerance": [{"scale": 1.0, "width": 3.0, "met": False, "barrier_sd": 0.3},
                            {"scale": 0.1, "width": 0.1, "met": True, "barrier_sd": 0.03}],
              "tolerance_width": 0.25, "seconds": 1.0}
    os.makedirs(tmp_path, exist_ok=True)
    with open(os.path.join(tmp_path, "cellmodel.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    text = open(R.report(str(tmp_path)), encoding="utf-8").read()
    assert "Item MS9" in text
    assert "A" in text and "B" in text
    assert "not predicted" in text            # the parameter that carries a stated spread
    assert "90.0 %" in text                   # the share of the variance, ranked first
    assert "0.030" in text                    # the tolerance row
    assert text.index("| A |") < text.index("| B |")


# --------------------------------------------------------------------------- #
# The cell model itself
# --------------------------------------------------------------------------- #
def test_the_published_set_holds_every_parameter_the_benchmark_changes():
    pytest.importorskip("pybamm")
    p = CM.published_parameters(R.BASE)
    for name in ("Positive particle diffusivity [m2.s-1]", "Positive electrode OCP [V]",
                 "Electrolyte conductivity [S.m-1]",
                 "Positive electrode Bruggeman coefficient (electrolyte)",
                 "Positive electrode exchange-current density [A.m-2]",
                 "Positive electrode porosity"):
        assert name in p


def test_the_cell_model_runs_at_its_published_values():
    pytest.importorskip("pybamm")
    run = CM.run_cell(cycles=2, kind="SPM")
    assert run.ok and len(run.capacity) == 2
    assert 3.0 < run.final < 6.0
    assert run.fade <= 1.0                    # the passivating layer only takes capacity away


def test_a_parameter_the_cell_model_does_not_hold_is_refused():
    pytest.importorskip("pybamm")
    with pytest.raises(KeyError, match="no entry called"):
        CM.run_cell({"Positive electrode magic [V]": 1.0}, cycles=1, kind="SPM")


def test_a_smaller_diffusion_coefficient_gives_less_capacity():
    pytest.importorskip("pybamm")
    p = CM.published_parameters(R.BASE)
    name = "Positive particle diffusivity [m2.s-1]"
    low = CM.run_cell({name: 1e-17}, cycles=1, kind="SPM", parameters=p)
    high = CM.run_cell({name: 4e-14}, cycles=1, kind="SPM", parameters=p)
    assert low.final < high.final


def test_a_shift_needs_the_value_it_shifts():
    pytest.importorskip("pybamm")
    p = ocp_shift(3.5, 3.5, "Positive electrode OCP [V]", sd=0.1)
    p.published = None
    with pytest.raises(ValueError, match="needs the published value"):
        CM.local_sensitivity(p, cycles=1, kind="SPM")
