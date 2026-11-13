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

PyBaMM is an optional dependency, as pymatgen is. Nothing in :mod:`cmfqsar.interfaces` needs it,
and every function here imports it when it is called, so the module can be imported without it.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .interfaces import CellParameter, CellParameterSet

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
             parameters=None) -> CellRun:
    """One run of the cell model, with the entries in `overrides` put in place of the published
    ones. Returns the discharge capacity of every cycle.

    `overrides` maps a parameter name to the value or the function that replaces it, which is
    what :meth:`cmfqsar.interfaces.CellParameter.apply` produces. `parameters` is for a caller
    that has the published set already, so that reading it is not repeated per run.
    """
    pybamm = _pybamm()
    p = (published_parameters(base) if parameters is None else parameters).copy()
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
    return CellRun(capacity=np.array(values, float), cycles=int(cycles), ok=done,
                   message="" if done else f"reached {len(values)} of {cycles} cycles",
                   overrides=numbers)


def sample(parameter: CellParameter, draws: int = 200, cycles: int = 10,
           base: str = DEFAULT_BASE, kind: str = "SPMe",
           options: Optional[Dict[str, object]] = None, cycle: Sequence[str] = DEFAULT_CYCLE,
           seed: int = 0, progress=None) -> List[CellRun]:
    """Run the cell model once per drawn value of one parameter, the others left published.

    This is the one-at-a-time sampling the report reads: it says what this parameter's deviation
    alone does to the cell, which is the question a tolerance rule needs answered.
    """
    p = published_parameters(base)
    if parameter.name not in p:
        raise KeyError(f"the parameter set {base} holds no entry called {parameter.name!r}")
    rng = np.random.default_rng(int(seed))
    values = parameter.draw(int(draws), rng)
    out = []
    for i, v in enumerate(values):
        out.append(run_cell({parameter.name: parameter.apply(p[parameter.name], v)},
                            cycles=cycles, base=base, kind=kind, options=options, cycle=cycle,
                            parameters=p))
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


__all__ = ["CellRun", "DEFAULT_BASE", "DEFAULT_CYCLE", "DEFAULT_OPTIONS", "MODELS", "band",
           "build_model", "contains", "local_sensitivity", "published_parameters", "run_cell",
           "sample", "tolerance_for_capacity", "variance_shares"]
