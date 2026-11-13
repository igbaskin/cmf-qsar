"""Running a cell model once per drawn parameter set, and saying which deviation matters
(plan item MS9).

A cell-level battery model -- the Doyle-Fuller-Newman model and its single-particle reductions,
as PyBaMM implements them -- is a set of partial differential equations over the thickness of the
electrodes and the radius of the particles, with time, boundary conditions and transport in it.
It is not a model this program can be, because this program has no time axis and no transport
equation. It is a model this program can feed: every one of its material parameters is a quantity
that some benchmark here predicts, and the prediction comes with a standard deviation in the
units of the parameter (plan item MS1).

This module is the step after :class:`cmfqsar.interfaces.CellParameterSet`, which holds the
parameters and their deviations. Three things happen here.

* :func:`run_cell` runs the cell model once, with some parameters replaced, and returns the
  discharge capacity of every cycle. A set of parameters the cell model refuses is not an error:
  it is recorded as a run that did not complete, because a deviation wide enough to leave the
  range the cell model accepts is a result about the deviation.
* :func:`sample` draws one parameter many times, with the others at their published values, and
  runs the cell model per draw. The output is a set of capacity-against-cycle curves instead of
  one curve, and :func:`band` turns it into an interval per cycle.
* :func:`variance_shares` says which parameter's deviation moves the last cycle's capacity most,
  as the share of the variance each one explains when it is drawn alone.
  :func:`local_sensitivity` is the other reading, which does not depend on the deviations: how
  far the capacity moves when one parameter is changed by a fixed fraction of itself. The first
  reading says what this program contributes; the second is a property of the cell model alone,
  and it is the one the cell-model literature states.
* :func:`response` runs the cell model once per chosen value of one parameter, which is its
  response curve, and :func:`tolerated_range` reads from that curve how far the parameter may be
  from its measured value before the capacity leaves a stated distance. This is the question to
  ask *before* a parameter is predicted: how accurate does a prediction have to be? (Plan items
  BT9 to BT11, the three parameters of the electrolyte this program does not predict.)
  :func:`nernst_einstein_diffusivity` is the reference such a prediction has to beat for the
  diffusion coefficient: the value an ideal solution would have, from the conductivity alone.

PyBaMM is an optional dependency, as pymatgen is. Nothing in :mod:`cmfqsar.interfaces` needs it,
and every function here imports it when it is called, so the module can be imported without it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .interfaces import GAS_CONSTANT, CellParameter, CellParameterSet

#: the Faraday constant, in coulomb per mole
FARADAY = 96485.33212

#: the published parameter set used when none is named: a graphite / nickel-manganese-cobalt
#: oxide cell measured and fitted by Chen and co-workers (2020), which PyBaMM ships
DEFAULT_BASE = "Chen2020"
#: one cycle: a discharge, a charge and a hold at the top of the charge
DEFAULT_CYCLE = ("Discharge at 1C until 2.5 V", "Charge at 1C until 4.2 V",
                 "Hold at 4.2 V until 50 mA")
#: the options the cell model is built with: the growth of the passivating layer on the negative
#: electrode is what makes the capacity fall from cycle to cycle
DEFAULT_OPTIONS = {"SEI": "solvent-diffusion limited"}
#: which reduction of the cell model to run; the single-particle model with electrolyte is the
#: cheapest one that still has the transport of the electrolyte in it
MODELS = ("SPM", "SPMe", "DFN")
#: the four transport parameters of the electrolyte in the cell model
ELECTROLYTE_ENTRIES = ("Electrolyte conductivity [S.m-1]", "Electrolyte diffusivity [m2.s-1]",
                       "Cation transference number", "Thermodynamic factor")
#: the published set that holds a measured electrolyte with all four of them: LiPF6 in EC:EMC
#: 3:7 by weight, the electrolyte of the Chen 2020 cell, measured by Landesfeind and Gasteiger
#: (2019) between -10 and +50 degrees Celsius and between 0.1 and 3.0 mol/L
MEASURED_ELECTROLYTE = "ORegan2022"
#: the coldest temperature, in kelvin, at which that study's transference numbers and
#: thermodynamic factors are used: the study says their precision is lower below +20 degrees
#: Celsius, its -10 degree values have large errors and go negative, and the fitted transference
#: number is negative at 0 degrees already from 0.5 mol/L up
COLDEST_TRUSTED = 293.15


def discharge(rate: float, cutoff: float = 2.5) -> Tuple[str]:
    """One discharge at `rate` times the one-hour rate down to `cutoff` volt, as a cycle.

    At several times the one-hour rate the transport in the electrolyte starts to limit the
    cell, so this is the cycle that asks whether the electrolyte's parameters matter. Use the
    full Doyle-Fuller-Newman model there: the single-particle model with electrolyte
    approximates the electrolyte and is not meant for high rates.
    """
    rate = float(rate)
    return (f"Discharge at {rate:g}C until {float(cutoff):g} V",)


def _pybamm():
    """PyBaMM, imported late so that this module can be imported without it."""
    try:
        import pybamm
    except Exception as exc:                            # pragma: no cover - no PyBaMM
        raise ImportError(
            "a cell model needs the pybamm package: pip install pybamm. The parameter set and "
            "its deviations (cmfqsar.interfaces) are written without it.") from exc
    return pybamm


def published_parameters(base: str = DEFAULT_BASE):
    """The published parameter set the cell model starts from."""
    return _pybamm().ParameterValues(str(base))


def measured_electrolyte(names: Sequence[str] = ELECTROLYTE_ENTRIES,
                         source: str = MEASURED_ELECTROLYTE,
                         coldest: Optional[float] = COLDEST_TRUSTED) -> Dict[str, object]:
    """The electrolyte functions of the published set `source`, as replacements for `run_cell`.

    The default source holds the functions Landesfeind and Gasteiger (2019) fitted to their
    measurements of the Chen 2020 cell's own electrolyte: each depends on the concentration and
    the temperature, where the Chen 2020 functions depend on the concentration only and two of
    the four are constants. Below `coldest` the transference number and the thermodynamic factor
    are read at `coldest`, since the study's coldest measurements of these two are not reliable;
    None reads them at the cell's temperature. The conductivity and the diffusivity are always
    read at the cell's temperature.
    """
    pybamm = _pybamm()
    p = pybamm.ParameterValues(str(source))
    out = {}
    for name in names:
        if name not in ELECTROLYTE_ENTRIES:
            raise KeyError(f"{name!r} is not one of the electrolyte's transport parameters")
        f = p[name]
        held = name in ELECTROLYTE_ENTRIES[2:] and coldest is not None and callable(f)
        out[name] = _held_above(f, float(coldest)) if held else f
    return out


def _held_above(function, coldest: float):
    """`function(c, T)` read at the larger of T and `coldest`."""
    def held(c_e, T):
        return function(c_e, _pybamm().maximum(T, coldest))
    return held


def build_model(kind: str = "SPMe", options: Optional[Dict[str, object]] = None):
    """The cell model itself."""
    pybamm = _pybamm()
    if kind not in MODELS:
        raise ValueError(f"the cell model is one of {', '.join(MODELS)}, not {kind!r}")
    opts = dict(DEFAULT_OPTIONS if options is None else options)
    return getattr(pybamm.lithium_ion, kind)(opts)


@dataclass
class CellRun:
    """What one run of the cell model gives: the discharge capacity of every cycle.

    `ok` is false when the cell model did not reach the last cycle asked for, and `message` says
    why. `capacity` then holds the cycles it did reach, which may be none of them.
    """

    capacity: np.ndarray
    cycles: int
    ok: bool = True
    message: str = ""
    overrides: Dict[str, float] = field(default_factory=dict)
    #: why the discharge of the last cycle ended, in the cell model's own words; filled only
    #: when the run was asked for details
    termination: str = ""
    #: the lowest and the highest concentration of the electrolyte, in mol/m3, anywhere in the
    #: cell during the discharge of the last cycle; filled only when the run was asked for details
    concentration: Tuple[float, float] = (float("nan"), float("nan"))

    @property
    def at_cutoff(self) -> bool:
        """Did the last discharge end at the lower voltage limit, as a discharge should?  A
        discharge that ended for another reason has no discharge capacity to compare."""
        t = self.termination.lower()
        return "voltage <" in t or "minimum voltage" in t

    @property
    def final(self) -> float:
        """The capacity of the last cycle, or not-a-number when there is none."""
        v = np.asarray(self.capacity, float).ravel()
        return float(v[-1]) if len(v) else float("nan")

    @property
    def fade(self) -> float:
        """How much of the first cycle's capacity is left at the last one."""
        v = np.asarray(self.capacity, float).ravel()
        return float(v[-1] / v[0]) if len(v) > 1 and v[0] > 0 else float("nan")


def run_cell(overrides: Optional[Dict[str, object]] = None, cycles: int = 10,
             base: str = DEFAULT_BASE, kind: str = "SPMe",
             options: Optional[Dict[str, object]] = None,
             cycle: Sequence[str] = DEFAULT_CYCLE, initial_soc: float = 1.0,
             parameters=None, temperature: Optional[float] = None,
             details: bool = False) -> CellRun:
    """One run of the cell model, with the entries in `overrides` put in place of the published
    ones. Returns the discharge capacity of every cycle.

    `overrides` maps a parameter name to the value or the function that replaces it, which is
    what :meth:`cmfqsar.interfaces.CellParameter.apply` produces. `parameters` is for a caller
    that has the published set already, so that reading it is not repeated per run.
    `temperature` sets the temperature of the cell and of its surroundings, in kelvin; how much
    that changes depends on which parameters of the published set depend on temperature at all.
    `details` also records why the discharge of the last cycle ended and the lowest and highest
    concentration of the electrolyte during it, which say whether the capacity is that of an
    ordinary discharge and whether the electrolyte's functions were read inside the range they
    were measured in.
    """
    pybamm = _pybamm()
    p = (published_parameters(base) if parameters is None else parameters).copy()
    if temperature is not None:
        p.update({"Ambient temperature [K]": float(temperature),
                  "Initial temperature [K]": float(temperature)})
    for name, value in dict(overrides or {}).items():
        if name not in p:
            raise KeyError(f"the parameter set {base} holds no entry called {name!r}")
        p[name] = value
    experiment = pybamm.Experiment([tuple(cycle)] * int(cycles))
    numbers = {k: (float(v) if isinstance(v, (int, float)) else float("nan"))
               for k, v in dict(overrides or {}).items()}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sim = pybamm.Simulation(build_model(kind, options), parameter_values=p,
                                    experiment=experiment)
            solution = sim.solve(initial_soc=float(initial_soc))
    except Exception as exc:
        return CellRun(capacity=np.zeros(0), cycles=int(cycles), ok=False,
                       message=f"{type(exc).__name__}: {str(exc)[:200]}", overrides=numbers)
    values = []
    for one in getattr(solution, "cycles", []) or []:
        if one is None or not getattr(one, "steps", None):
            continue
        try:
            values.append(float(one.steps[0]["Discharge capacity [A.h]"].entries[-1]))
        except Exception:                                # pragma: no cover - a cycle with no step
            continue
    done = len(values) == int(cycles)
    termination, concentration = _discharge_details(solution) if details else ("", _NO_RANGE)
    return CellRun(capacity=np.array(values, float), cycles=int(cycles), ok=done,
                   message="" if done else f"reached {len(values)} of {cycles} cycles",
                   overrides=numbers, termination=termination, concentration=concentration)


_NO_RANGE = (float("nan"), float("nan"))


def _discharge_details(solution) -> Tuple[str, Tuple[float, float]]:
    """Why the discharge of the last cycle ended, and the range of the electrolyte's
    concentration during it."""
    last = None
    for one in getattr(solution, "cycles", []) or []:
        if one is not None and getattr(one, "steps", None):
            last = one.steps[0]
    if last is None:
        return "", _NO_RANGE
    termination = str(getattr(last, "termination", "") or "")
    try:
        c = np.asarray(last["Electrolyte concentration [mol.m-3]"].entries, float)
        return termination, (float(np.nanmin(c)), float(np.nanmax(c)))
    except Exception:                                    # a model without that variable
        return termination, _NO_RANGE


def sample(parameter: CellParameter, draws: int = 200, cycles: int = 10,
           base: str = DEFAULT_BASE, kind: str = "SPMe",
           options: Optional[Dict[str, object]] = None, cycle: Sequence[str] = DEFAULT_CYCLE,
           seed: int = 0, progress=None, temperature: Optional[float] = None,
           fixed: Optional[Dict[str, object]] = None) -> List[CellRun]:
    """Run the cell model once per drawn value of one parameter, the others left published.

    This is the one-at-a-time sampling the report reads: it says what this parameter's deviation
    alone does to the cell, which is the question a tolerance rule needs answered. `fixed` holds
    replacements that stay the same in every run, such as a measured electrolyte from
    :func:`measured_electrolyte`; a drawn scale or shift then acts on the fixed value, not on the
    published one.
    """
    p = published_parameters(base)
    if parameter.name not in p:
        raise KeyError(f"the parameter set {base} holds no entry called {parameter.name!r}")
    fixed = dict(fixed or {})
    start = fixed.get(parameter.name, p[parameter.name])
    rng = np.random.default_rng(int(seed))
    values = parameter.draw(int(draws), rng)
    out = []
    for i, v in enumerate(values):
        out.append(run_cell({**fixed, parameter.name: parameter.apply(start, v)},
                            cycles=cycles, base=base, kind=kind, options=options, cycle=cycle,
                            parameters=p, temperature=temperature))
        out[-1].overrides = {parameter.name: float(v)}
        if progress is not None:
            progress(i + 1, len(values))
    return out


def band(runs: Sequence[CellRun], level: float = 95.0) -> Dict[str, np.ndarray]:
    """The interval of the capacity at every cycle, over the runs that completed.

    Returns the lower edge, the median and the upper edge, each one value per cycle, together
    with how many runs they are made of. A run the cell model refused is not in the interval;
    `refused` is the share of runs that did not complete, and it belongs next to the interval
    whenever it is not zero, because an interval of the runs that worked says nothing about the
    draws that did not.
    """
    full = [np.asarray(r.capacity, float) for r in runs if r.ok and len(r.capacity)]
    refused = float(np.mean([not r.ok for r in runs])) if len(runs) else float("nan")
    if not full:
        return {"low": np.zeros(0), "median": np.zeros(0), "high": np.zeros(0), "n": 0,
                "refused": refused}
    n_cycles = min(len(v) for v in full)
    M = np.stack([v[:n_cycles] for v in full])
    half = (100.0 - float(level)) / 2.0
    return {"low": np.percentile(M, half, axis=0), "median": np.percentile(M, 50.0, axis=0),
            "high": np.percentile(M, 100.0 - half, axis=0), "n": int(len(full)),
            "refused": refused}


def contains(interval: Dict[str, np.ndarray], curve: Sequence[float]) -> Dict[str, object]:
    """Does the interval hold the published curve, cycle by cycle?

    Returns how many cycles of the curve lie inside the interval, the share, and the largest
    distance from the interval in the units of the capacity (zero when the curve is inside
    everywhere).
    """
    y = np.asarray(curve, float).ravel()
    lo, hi = np.asarray(interval["low"], float), np.asarray(interval["high"], float)
    n = min(len(y), len(lo))
    if n == 0:
        return {"cycles": 0, "inside": 0, "share": float("nan"), "largest_gap": float("nan"),
                "all_inside": False}
    y, lo, hi = y[:n], lo[:n], hi[:n]
    inside = (y >= lo) & (y <= hi)
    gap = np.maximum(np.maximum(lo - y, y - hi), 0.0)
    return {"cycles": int(n), "inside": int(inside.sum()), "share": float(inside.mean()),
            "largest_gap": float(gap.max()), "all_inside": bool(inside.all())}


def variance_shares(results: Dict[str, Sequence[CellRun]]) -> Dict[str, Dict[str, float]]:
    """Which parameter's deviation moves the last cycle's capacity most.

    For every parameter, the variance of the capacity of the last cycle over its own draws, and
    that variance as a share of the sum over the parameters. This is one-at-a-time attribution:
    it ignores what two deviations do together, which is the right reading when the question is
    which single parameter to measure better, and the wrong one when the parameters interact.
    Runs the cell model refused are left out and counted in `refused`.
    """
    stats = {}
    for name, runs in results.items():
        vals = np.array([r.final for r in runs if r.ok and np.isfinite(r.final)], float)
        stats[name] = {"n": int(len(vals)),
                       "refused": float(np.mean([not r.ok for r in runs])) if len(runs) else 0.0,
                       "mean": float(np.mean(vals)) if len(vals) else float("nan"),
                       "sd": float(np.std(vals)) if len(vals) > 1 else float("nan"),
                       "variance": float(np.var(vals)) if len(vals) > 1 else float("nan"),
                       "low": float(np.percentile(vals, 2.5)) if len(vals) > 1 else float("nan"),
                       "high": float(np.percentile(vals, 97.5)) if len(vals) > 1
                       else float("nan")}
    total = float(np.nansum([s["variance"] for s in stats.values()]))
    for s in stats.values():
        s["share"] = (float(s["variance"] / total) if total > 0 and np.isfinite(s["variance"])
                      else float("nan"))
    return stats


def local_sensitivity(parameter: CellParameter, step: float = 0.1, cycles: int = 10,
                      base: str = DEFAULT_BASE, kind: str = "SPMe",
                      options: Optional[Dict[str, object]] = None,
                      cycle: Sequence[str] = DEFAULT_CYCLE) -> Dict[str, float]:
    """How far the capacity moves when one parameter is changed by a fixed fraction of itself.

    The reading that does not depend on the deviations of this program, so that the ranking can
    be compared with the cell-model literature. The parameter is changed up and down by `step`
    times its own size, around the *published* value and not around the prediction, and the
    answer is the elasticity, the relative change of the capacity of the last cycle over the
    relative change of the parameter. It is therefore a property of the cell model at the point
    its own authors fitted.

    For a parameter that is added to the published value (an open-circuit voltage shifted in
    volt) the size of the change is `step` times the published value, since the shift itself
    starts at zero. A tenth of a voltage is a large change in a battery and a tenth of a
    diffusion coefficient is a small one, so the elasticities of parameters of different kinds
    order the cell model's response to equal *relative* changes and not to equally likely ones.
    """
    p = published_parameters(base)
    if parameter.name not in p:
        raise KeyError(f"the parameter set {base} holds no entry called {parameter.name!r}")
    if parameter.kind == "shift":
        scale = parameter.published
        if scale is None or not np.isfinite(scale) or scale == 0:
            raise ValueError(f"{parameter.name} is a shift, so its sensitivity needs the "
                             "published value it shifts, as published=")
        centre, delta = 0.0, float(step) * abs(float(scale))
    else:
        centre = 1.0 if parameter.kind == "scale" else float(p[parameter.name]) \
            if not callable(p[parameter.name]) else float("nan")
        if not np.isfinite(centre):
            raise ValueError(f"{parameter.name} is a function in the cell model, so it cannot "
                             "be replaced by a number; measure a scale of it instead")
        delta = float(step) * abs(centre)
    runs = {}
    for label, v in (("down", centre - delta), ("up", centre + delta),
                     ("centre", centre)):
        runs[label] = run_cell({parameter.name: parameter.apply(p[parameter.name], v)},
                               cycles=cycles, base=base, kind=kind, options=options,
                               cycle=cycle, parameters=p)
    c0 = runs["centre"].final
    if not np.isfinite(c0) or c0 == 0:
        return {"elasticity": float("nan"), "capacity": c0, "step": float(step),
                "down": runs["down"].final, "up": runs["up"].final,
                "refused": float(sum(not r.ok for r in runs.values()))}
    up, down = runs["up"].final, runs["down"].final
    if not (np.isfinite(up) and np.isfinite(down)):
        e = float("nan")
    else:
        e = float(((up - down) / c0) / (2.0 * float(step)))
    return {"elasticity": e, "capacity": float(c0), "step": float(step),
            "down": float(down), "up": float(up),
            "refused": float(sum(not r.ok for r in runs.values()))}


def response(parameter: CellParameter, values: Sequence[float], cycles: int = 1,
             base: str = DEFAULT_BASE, kind: str = "SPMe",
             options: Optional[Dict[str, object]] = None, cycle: Sequence[str] = DEFAULT_CYCLE,
             temperature: Optional[float] = None, fixed: Optional[Dict[str, object]] = None,
             parameters=None) -> List[CellRun]:
    """The response curve of the cell to one parameter: one run per value in `values`.

    Nothing is drawn. A drawn sample answers what one stated deviation does to the capacity; a
    response curve answers it for every deviation at once, because the interval of the capacity
    for any spread of the parameter can be read from the curve. That is what a tolerance needs,
    and it is why this is the tool for a parameter that has no prediction yet.

    `fixed` holds replacements that stay the same in every run, as in :func:`sample`, and the
    values act on the fixed entry: a factor of 1 or a shift of 0 is then the cell with the fixed
    entries and nothing else changed. Every run carries its details (:func:`run_cell`).
    """
    p = published_parameters(base) if parameters is None else parameters
    if parameter.name not in p:
        raise KeyError(f"the parameter set {base} holds no entry called {parameter.name!r}")
    fixed = dict(fixed or {})
    start = fixed.get(parameter.name, p[parameter.name])
    out = []
    for v in values:
        run = run_cell({**fixed, parameter.name: parameter.apply(start, float(v))},
                       cycles=cycles, base=base, kind=kind, options=options, cycle=cycle,
                       parameters=p, temperature=temperature, details=True)
        run.overrides = {parameter.name: float(v)}
        out.append(run)
    return out


def tolerated_range(values: Sequence[float], capacities: Sequence[float], centre: float,
                    width: float, log: bool = False, level: float = 95.0) -> Dict[str, object]:
    """How far a parameter may be from `centre` before the capacity leaves `width` of the
    capacity there.

    `values` are the points a response curve was run at and `capacities` what the cell gave at
    them; `centre` is the unchanged value and has to be one of the points. The answer is the
    connected range around the centre inside which the capacity stays within `width`
    ampere-hours of the capacity at the centre. Its two ends are found by linear interpolation
    between neighbouring points, in the logarithm of the value when `log` is true, which is the
    right axis for a factor.

    Three things can end the range on a side, and the record says which. The capacity left the
    limit between two points (``"crossed"``). The next point gave no capacity, because the cell
    model refused it or the discharge did not end at the voltage limit (``"refused"``; the end is
    then the last point that did give one). Or the points ran out with the capacity still inside
    (``"open"``; the end is then the outermost point, and the true end is at least that far). A
    tolerance stated from an open end has to say so.

    `sd` is the largest standard deviation of an error whose central interval at `level` per
    cent stays inside the range: of the natural logarithm of the parameter for a factor, of the
    parameter itself otherwise. The nearer end sets it, because a cell responds differently to
    too little and to too much. `sd_is_a_lower_bound` is true when that nearer end is open.
    """
    from statistics import NormalDist

    x = np.asarray(values, float).ravel()
    q = np.asarray(capacities, float).ravel()
    if len(x) != len(q):
        raise ValueError(f"{len(x)} values and {len(q)} capacities")
    if log and np.any(x <= 0):
        raise ValueError("a factor is read on a logarithmic axis, so every value has to be "
                         "above zero")
    order = np.argsort(x)
    x, q = x[order], q[order]
    at = np.flatnonzero(np.isclose(x, float(centre), rtol=1e-12, atol=1e-12))
    if len(at) != 1:
        raise ValueError(f"the centre {centre!r} has to be exactly one of the values")
    k = int(at[0])
    q0 = float(q[k])
    if not np.isfinite(q0):
        raise ValueError("the run at the centre gave no capacity, so there is nothing to "
                         "measure a distance from")
    width = float(width)
    if width <= 0:
        raise ValueError(f"the width has to be above zero, not {width!r}")
    axis = np.log(x) if log else x

    def walk(step: int) -> Tuple[float, str]:
        i = k
        while True:
            j = i + step
            if j < 0 or j >= len(x):
                return float(x[i]), "open"
            if not np.isfinite(q[j]):
                return float(x[i]), "refused"
            d = float(q[j]) - q0
            if abs(d) > width:
                d0 = float(q[i]) - q0
                limit = width if d > 0 else -width
                a = axis[i] + (limit - d0) / (d - d0) * (axis[j] - axis[i])
                return float(np.exp(a) if log else a), "crossed"
            i = j

    low, low_ends = walk(-1)
    high, high_ends = walk(+1)
    z = NormalDist().inv_cdf(0.5 + float(level) / 200.0)
    c = float(centre)
    below = (np.log(c / low) if log else c - low)
    above = (np.log(high / c) if log else high - c)
    nearer = low_ends if below <= above else high_ends
    return {"centre": c, "capacity": q0, "width": width, "low": low, "high": high,
            "low_ends": low_ends, "high_ends": high_ends,
            "sd": float(min(below, above) / z), "sd_is_a_lower_bound": nearer == "open"}


def nernst_einstein_diffusivity(conductivity, transference, factor=1.0, floor: float = 1.0):
    """The salt diffusion coefficient that a conductivity implies for an ideal solution of
    fully dissociated ions, as a function of concentration and temperature.

    For a salt of two singly charged ions the Nernst-Einstein relation ties the conductivity to
    the sum of the two self-diffusion coefficients, ``kappa = F^2 c (D+ + D-) / (R T)``, and the
    transference number divides that sum between the ions, ``t = D+ / (D+ + D-)``. The salt
    diffusion coefficient of a dilute solution is ``2 D+ D- / (D+ + D-)``, so

        D = 2 t (1 - t) kappa R T / (F^2 c)

    with the concentration in mol/m3 and the conductivity in S/m. It needs nothing but the
    conductivity, which is why it is the reference a predicted diffusion coefficient has to
    beat (plan item BT9).

    `conductivity` is a function of the concentration and the temperature; `transference` and
    `factor` are numbers or functions of the same two. `factor` multiplies the result: for a
    concentrated solution the thermodynamic factor belongs there, and 1 is the ideal solution.
    The concentration is read no lower than `floor`, because the relation divides by it.
    """
    def read(x, c, T):
        return x(c, T) if callable(x) else float(x)

    def diffusivity(c_e, T):
        c = _at_least(c_e, float(floor))
        t = read(transference, c, T)
        return (2.0 * t * (1.0 - t) * conductivity(c, T) * GAS_CONSTANT * T
                / (FARADAY ** 2 * c)) * read(factor, c, T)
    return diffusivity


def _at_least(x, low: float):
    """`x` no lower than `low`, for a number or for a symbol of the cell model."""
    if isinstance(x, (int, float, np.integer, np.floating)):
        return max(float(x), float(low))
    return _pybamm().maximum(x, float(low))


def tolerance_for_capacity(parameter: CellParameter, width: float, scales: Sequence[float],
                           published: Optional[float] = None, draws: int = 100, cycles: int = 3,
                           base: str = DEFAULT_BASE, kind: str = "SPMe",
                           options: Optional[Dict[str, object]] = None,
                           cycle: Sequence[str] = DEFAULT_CYCLE, seed: int = 0
                           ) -> List[Dict[str, float]]:
    """How accurate a parameter has to be for the capacity interval to be worth using.

    The deviation of the parameter is made smaller by each factor in `scales`, the cell model is
    run over the draws, and the interval of the last cycle's capacity is reported. This is the
    rule of plan item MS1 read in the units of the cell model.

    An interval is worth using only if it is both **narrow** and **right**, so every row carries
    both: `narrow` is true when the interval is inside `width` ampere-hours, `holds` when it
    still contains the capacity `published` gives, and `useful` when both are. A prediction whose
    error is a bias and not a spread narrows towards the wrong number, and then no deviation is
    small enough: that is what the two columns separate.
    """
    out = []
    for s in scales:
        shrunk = CellParameter(**{**parameter.__dict__,
                                  "sd": float(parameter.sd) * float(s),
                                  "sample": None if parameter.distribution != "sample" else
                                  _shrink_sample(parameter.sample, float(s)),
                                  })
        runs = sample(shrunk, draws=draws, cycles=cycles, base=base, kind=kind, options=options,
                      cycle=cycle, seed=seed)
        b = band(runs)
        if b["n"] < 2:
            out.append({"scale": float(s), "width": float("nan"), "narrow": False,
                        "holds": False, "useful": False, "refused": b["refused"]})
            continue
        lo, hi = float(b["low"][-1]), float(b["high"][-1])
        w = hi - lo
        narrow = bool(w <= float(width))
        holds = (bool(lo <= float(published) <= hi) if published is not None
                 and np.isfinite(published) else None)
        out.append({"scale": float(s), "width": w, "narrow": narrow, "holds": holds,
                    "useful": bool(narrow and (holds is not False)), "low": lo, "high": hi,
                    "refused": float(b["refused"])})
    return out


def _shrink_sample(values: Sequence[float], scale: float) -> np.ndarray:
    """A set of drawn values pulled towards its own median by `scale`, on the logarithmic axis
    when every value is positive, so that a diffusion coefficient stays positive."""
    v = np.asarray(values, float).ravel()
    if np.all(v > 0):
        logs = np.log(v)
        return np.exp(np.median(logs) + float(scale) * (logs - np.median(logs)))
    return float(np.median(v)) + float(scale) * (v - float(np.median(v)))


__all__ = ["CellRun", "DEFAULT_BASE", "DEFAULT_CYCLE", "DEFAULT_OPTIONS", "FARADAY", "MODELS",
           "band", "build_model", "contains", "discharge", "local_sensitivity",
           "measured_electrolyte", "nernst_einstein_diffusivity", "published_parameters",
           "response", "run_cell", "sample", "tolerance_for_capacity", "tolerated_range",
           "variance_shares"]


# --------------------------------------------------------------------------- #
# The activation energy of an exchange-current density (plan item BT12)
# --------------------------------------------------------------------------- #
#: the cell model's names of the exchange-current densities of the two electrodes
EXCHANGE_ENTRIES = ("Negative electrode exchange-current density [A.m-2]",
                    "Positive electrode exchange-current density [A.m-2]")
#: the temperature, in kelvin, at which the Arrhenius laws of the published set equal one
ARRHENIUS_REFERENCE = 298.15


def arrhenius_factor(shift: float, temperature: float,
                     reference: float = ARRHENIUS_REFERENCE) -> float:
    """The factor on a rate at `temperature` when its activation energy is raised by `shift`
    (J/mol) and its value at `reference` is kept.

    The factor is 1 at the reference temperature. In a colder cell a higher activation energy
    gives a smaller rate, and in a warmer cell a larger one.
    """
    return float(np.exp(float(shift) / GAS_CONSTANT
                        * (1.0 / float(reference) - 1.0 / float(temperature))))


def activation_shift(factor: float, temperature: float,
                     reference: float = ARRHENIUS_REFERENCE) -> float:
    """The shift of an activation energy (J/mol) that puts `factor` on a rate at `temperature`:
    the inverse of :func:`arrhenius_factor`. At the reference temperature no shift changes the
    rate, so the question has no answer there and is refused."""
    slope = 1.0 / float(reference) - 1.0 / float(temperature)
    if abs(slope) < 1e-12:
        raise ValueError("at the reference temperature the activation energy does not change "
                         "the rate, so a factor there says nothing about it")
    if not float(factor) > 0:
        raise ValueError(f"a factor on a rate has to be above zero, not {factor!r}")
    return float(GAS_CONSTANT * np.log(float(factor)) / slope)


def exchange_current_at(function, temperature: float, concentration: float = 1000.0,
                        surface: float = 15000.0, maximum: float = 30000.0) -> float:
    """An exchange-current density of the cell model as a number: `function` at one
    concentration of the electrolyte, one concentration at the surface of the particle and one
    maximum concentration (all mol/m3), and one temperature (K)."""
    pybamm = _pybamm()
    v = function(pybamm.Scalar(float(concentration)), pybamm.Scalar(float(surface)),
                 pybamm.Scalar(float(maximum)), pybamm.Scalar(float(temperature)))
    return float(v.evaluate()) if hasattr(v, "evaluate") else float(v)


def activation_energy_of(function, temperatures: Tuple[float, float] = (298.15, 273.15),
                         **where) -> float:
    """The activation energy (J/mol) of an exchange-current density of the cell model, read
    from its values at two temperatures. For an Arrhenius law the result does not depend on
    which two."""
    t1, t2 = (float(t) for t in temperatures)
    v1 = exchange_current_at(function, t1, **where)
    v2 = exchange_current_at(function, t2, **where)
    return float(GAS_CONSTANT * np.log(v1 / v2) / (1.0 / t2 - 1.0 / t1))


@dataclass
class ActivationEnergy:
    """A shift of the activation energy of a rate that the cell model holds as a function.

    It stands where a :class:`cmfqsar.interfaces.CellParameter` stands in :func:`response`: it
    has the `name` of the entry and an `apply`. The value is the shift in J/mol. The function
    is multiplied by the Arrhenius factor of that shift, so its value at `reference` stays the
    published one. The temperature has to be the last argument of the function, as it is for
    the exchange-current densities.
    """

    name: str
    reference: float = ARRHENIUS_REFERENCE

    def apply(self, published, value: float):
        if not callable(published):
            raise TypeError(f"{self.name} is a number in this parameter set, so it has no "
                            "activation energy to shift")
        pybamm = _pybamm()

        def shifted(*a, _f=published, _s=float(value), _t=float(self.reference)):
            return _f(*a) * pybamm.exp(_s / GAS_CONSTANT * (1.0 / _t - 1.0 / a[-1]))
        return shifted
