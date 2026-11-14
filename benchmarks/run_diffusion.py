#!/usr/bin/env python
"""
The salt diffusion coefficient of an electrolyte from simulation (plan item BT9).

What this benchmark does
------------------------
A cell model reads the salt diffusion coefficient of its electrolyte, and this program has no
prediction of it. The tolerance stage of ``run_cellmodel.py`` says how accurate one has to be:
for a capacity that may move by 5 %, no lower than 0.32 of the measured value at the one-hour
rate and 0.83 at 2.5 times that rate, with no limit on the high side at room temperature.
Measured values exist for a few solvent mixtures only, so the labels a model could be fitted on
have to come from simulation. This benchmark asks the two questions that come before any model
(rules 2 and 3 of plan items BT9 to BT11):

* **Is a simulated label right?**  The electrolyte of the reference cell, lithium
  hexafluorophosphate in ethylene carbonate and ethyl methyl carbonate, 3 to 7 by weight, is
  simulated and compared with what was measured for it: the salt diffusion coefficient and the
  conductivity of Landesfeind and Gasteiger (2019), whose fitted functions PyBaMM ships, and
  the conductivity of CALiSol-23.
* **How large a box and how long a run does a label need?**  The error of a coefficient over
  blocks of the run, read at several lengths of one run, and the dependence on the size of the
  box.

The runs
--------
Every run is one box, simulated by :func:`cmfqsar.transport.run_transport`: equilibrated at
constant pressure, then run at constant volume with a thermostat that conserves momentum, with
the unwrapped centre of every molecule stored once per picosecond. The runs of this benchmark
differ in one setting each, and :data:`RUNS` lists them.

* ``full``: the force field as it is, with whole charges on the ions.
* ``scaled``: the charges of the ions multiplied by 0.8, the usual repair for a force field
  without polarisation, which binds a small cation too strongly.
* ``scaled_07``: the charges of the ions multiplied by 0.7. The conductivity rises steeply as
  the charges are scaled down, and a third value of the factor says where the simulated
  conductivity meets the measured one.
* ``scaled_small``, ``scaled_large`` and ``scaled_huge``: the same as ``scaled`` with half, twice
  and four times as many molecules.
* ``scaled_seed2`` and ``scaled_seed3``: the same as ``scaled``, packed again and started with
  other velocities. The error over the blocks of one run does not include the difference
  between two arrangements of the ions that both last longer than the run; these repeats
  measure it.
  A periodic box makes every diffusion coefficient too small by an amount that falls with the
  side of the box, so three sizes give the coefficient of an infinite box and the viscosity.

What is compared
----------------
* The self-diffusion coefficients of the cation, the anion and each solvent.
* The salt diffusion coefficient: ``2 D+ D- / (D+ + D-)`` for an ideal solution, and the same
  multiplied by the measured thermodynamic factor, against the measured function.
* The conductivity from the displacement of the total charge, and the Nernst-Einstein value
  from the two self-diffusion coefficients, against the two measured conductivities.
* The share of the current the cation would carry if every ion moved on its own, against the
  measured transference number, as a first look at item BT10.

Usage
-----
    python benchmarks/run_diffusion.py --stage simulate    # needs the md environment and a GPU
    python benchmarks/run_diffusion.py --stage analyse     # numpy only; PyBaMM for the reference
    python benchmarks/run_diffusion.py --report

The simulation stage is run inside the environment that has OpenMM with a GPU platform, packmol
and AmberTools (on this machine: ``micromamba run -n md`` in WSL). The other stages need none of
that. Every run writes its own directory under ``benchmarks/cache/transport`` and is not run
again, so the benchmark can be continued.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results", "diffusion")
CACHE = os.path.join(HERE, "cache", "transport")

#: the electrolyte of the reference cell: the salt, the solvents by weight, moles of salt per
#: kilogram of solvent. The molality is the one CALiSol-23 measured this mixture at.
SALT = "LiPF6"
SOLVENTS = {"EC": 0.30, "EMC": 0.70}
MOLALITY = 1.11
ELECTROLYTE_ID = "LiPF6_EC30_EMC70_m1p110"
ROOM = 298.15
#: the published set that holds the measured functions of this electrolyte
MEASURED = "ORegan2022"

#: the runs: what each one changes from the defaults of ``TransportSettings``
RUNS: Dict[str, Dict[str, object]] = {
    "full": {"label": "whole charges on the ions", "settings": {}},
    "scaled": {"label": "charges of the ions multiplied by 0.8",
               "settings": {"ion_charge_scale": 0.8}},
    "scaled_small": {"label": "scaled charges, half as many molecules",
                     "settings": {"ion_charge_scale": 0.8, "n_solvent": 125}},
    "scaled_large": {"label": "scaled charges, twice as many molecules",
                     "settings": {"ion_charge_scale": 0.8, "n_solvent": 500}},
    "scaled_07": {"label": "charges of the ions multiplied by 0.7",
                  "settings": {"ion_charge_scale": 0.7}},
    "scaled_seed2": {"label": "scaled charges, a second start",
                     "settings": {"ion_charge_scale": 0.8, "seed": 2}},
    "scaled_seed3": {"label": "scaled charges, a third start",
                     "settings": {"ion_charge_scale": 0.8, "seed": 3}},
    "scaled_huge": {"label": "scaled charges, four times as many molecules",
                    "settings": {"ion_charge_scale": 0.8, "n_solvent": 1000}},
}
#: the runs that differ only in the size of the box, from which the correction for a finite box
#: is read
SIZE_RUNS = ("scaled_small", "scaled", "scaled_large", "scaled_huge")
#: the runs that differ only in how the box was packed and in the first velocities: their
#: scatter is the error that the blocks of one run cannot see
REPEATS = ("scaled", "scaled_seed2", "scaled_seed3")
#: the lengths, in picoseconds, at which one run is read to see how its error falls
LENGTHS = (500.0, 1000.0, 2000.0, 4000.0)
STAGES = ("simulate", "analyse")


def log(msg: str) -> None:
    print(msg, flush=True)


def run_dir(name: str, cache: str = CACHE) -> str:
    return os.path.join(cache, f"{ELECTROLYTE_ID}_{name}")


def formulation(temperature: float = ROOM):
    from cmfqsar.electrolyte import Formulation
    return Formulation(SALT, dict(SOLVENTS), MOLALITY, temperature=temperature)


def settings_of(name: str):
    from cmfqsar.transport import TransportSettings
    if name not in RUNS:
        raise SystemExit(f"no run is called {name!r}; the runs are {', '.join(RUNS)}")
    return TransportSettings(**dict(RUNS[name]["settings"]))


# --------------------------------------------------------------------------- #
# Stage 1: the simulations
# --------------------------------------------------------------------------- #
def stage_simulate(names: Sequence[str], cache: str = CACHE) -> None:
    from cmfqsar.transport import run_transport

    for name in names:
        s = settings_of(name)
        t0 = time.time()
        done = os.path.isfile(os.path.join(run_dir(name, cache), "run.json"))
        run = run_transport(formulation(s.temperature), run_dir(name, cache), s, progress=log)
        log(f"{name}: {run['atoms']} atoms, {run['pairs']} ion pairs, side {run['side']:.2f} A, "
            f"density {run['density']:.3f} g/cm3, {run['molarity']:.3f} mol/L, "
            f"{run['production_ps']:.0f} ps at {run['ns_per_day']:.0f} ns/day"
            + (" (there already)" if done else f" ({time.time() - t0:.0f} s in all)"))


def read_run(name: str, cache: str = CACHE) -> dict:
    path = os.path.join(run_dir(name, cache), "run.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run this benchmark with --stage simulate "
                         f"--runs {name} in the simulation environment")
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    # the record holds the path as the simulation environment saw it
    run["centres"] = os.path.join(run_dir(name, cache), "centres.npz")
    return run


# --------------------------------------------------------------------------- #
# Stage 2: the analysis and the measured reference
# --------------------------------------------------------------------------- #
def measured_reference(molarity: float, temperature: float) -> Dict[str, float]:
    """The four transport parameters of this electrolyte as Landesfeind and Gasteiger measured
    them, at one concentration (mol/L) and one temperature, from the functions PyBaMM ships."""
    import run_cellmodel as RC
    from cmfqsar import cellmodel as CM

    four = CM.measured_electrolyte(coldest=None)
    c = 1000.0 * float(molarity)
    return {"conductivity": RC.electrolyte_value(four[RC.CONDUCTIVITY], temperature, c),
            "diffusivity": RC.electrolyte_value(four[RC.DIFFUSIVITY], temperature, c),
            "transference": RC.electrolyte_value(four[RC.TRANSFERENCE], temperature, c),
            "thermodynamic_factor": RC.electrolyte_value(four[RC.THERMODYNAMIC], temperature, c),
            "source": f"{MEASURED} in PyBaMM: Landesfeind and Gasteiger (2019)"}


def calisol_conductivity() -> Optional[float]:
    """The conductivity of this formulation at 298 K as CALiSol-23 holds it, in S/m."""
    import csv
    path = os.path.join(ROOT, "examples", "electrolyte", "formulations.csv")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("id") == ELECTROLYTE_ID:
                return 0.1 * 10.0 ** float(row["log10_k_298"])        # mS/cm to S/m
    return None


def analyse_run(name: str, cache: str = CACHE) -> dict:
    from cmfqsar import transport as T

    run = read_run(name, cache)
    whole = T.analyse(run)
    ref = measured_reference(whole["molarity"], float(run["temperature"]))
    lengths = []
    for length in LENGTHS:
        if length > whole["length_ps"] + 1e-6:
            continue
        part = T.analyse(run, last_ps=length)
        lengths.append({"length_ps": part["length_ps"],
                        "cation": {k: part["cation"][k] for k in ("d", "se", "exponent")},
                        "anion": {k: part["anion"][k] for k in ("d", "se", "exponent")},
                        "salt": part["salt"],
                        "conductivity": {k: part["conductivity"][k] for k in ("sigma", "se")}})
    factor = ref["thermodynamic_factor"]
    out = {"run": name, "label": RUNS[name]["label"], "settings": run["settings"],
           "atoms": run["atoms"], "pairs": run["pairs"], "side": run["side"],
           "density": run["density"], "molarity": whole["molarity"],
           "realised_molality": run["realised_molality"],
           "temperature": run["temperature"],
           "mean_kinetic_temperature": run["mean_kinetic_temperature"],
           "ns_per_day": run["ns_per_day"], "seconds": run["seconds"],
           "whole": whole, "lengths": lengths, "measured": ref,
           "calisol_conductivity": calisol_conductivity(),
           "salt_ideal": whole["salt"], "salt_with_factor": whole["salt"] * factor,
           "ratio_ideal": whole["salt"] / ref["diffusivity"],
           "ratio_with_factor": whole["salt"] * factor / ref["diffusivity"],
           "ratio_conductivity": whole["conductivity"]["sigma"] / ref["conductivity"],
           "ratio_nernst_einstein": whole["nernst_einstein"] / ref["conductivity"]}
    return out


def stage_analyse(names: Sequence[str], results: str = RESULTS, cache: str = CACHE) -> None:
    os.makedirs(results, exist_ok=True)
    for name in names:
        res = analyse_run(name, cache)
        w = res["whole"]
        log(f"{name}: D+ {w['cation']['d']:.3e} +/- {w['cation']['se']:.1e} "
            f"(exponent {w['cation']['exponent']:.2f}), D- {w['anion']['d']:.3e} +/- "
            f"{w['anion']['se']:.1e}, salt {res['salt_ideal']:.3e}, with the measured factor "
            f"{res['salt_with_factor']:.3e} against the measured "
            f"{res['measured']['diffusivity']:.3e} m2/s (ratio {res['ratio_with_factor']:.2f}); "
            f"conductivity {w['conductivity']['sigma']:.3f} +/- {w['conductivity']['se']:.3f} "
            f"S/m against {res['measured']['conductivity']:.3f}")
        with open(os.path.join(results, f"run_{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)


def _species(row: dict) -> Dict[str, dict]:
    """The self-diffusion records of one result, by species."""
    out = {"cation": row["whole"]["cation"], "anion": row["whole"]["anion"]}
    out.update({name: row["whole"]["solvents"][name] for name in sorted(row["whole"]["solvents"])})
    return out


def repeat_study(rows: Sequence[dict]) -> Optional[dict]:
    """What repeats of one box say about the error of one run.

    Needs at least two of the runs in :data:`REPEATS` among `rows`; returns None otherwise. For
    every species it gives the coefficient of each repeat, their mean, their standard deviation
    and the mean of their block errors. The ratio of the last two is the number of interest: a
    value near one says that the blocks of one run see the whole error, and a larger value says
    that two runs differ by more than either of them knows. `error_factor` is the median of
    that ratio over the species, and never below one; the size study multiplies the block
    errors of single runs by it.
    """
    chosen = [r for r in rows if r["run"] in REPEATS]
    if len(chosen) < 2:
        return None
    n = len(chosen)
    species = {}
    for name in _species(chosen[0]):
        d = np.array([_species(r)[name]["d"] for r in chosen], float)
        se = np.array([_species(r)[name]["se"] for r in chosen], float)
        sd = float(np.std(d, ddof=1))
        species[name] = {"d": [float(v) for v in d], "mean": float(d.mean()), "sd": sd,
                         "se_of_mean": sd / np.sqrt(n), "block_error": float(se.mean()),
                         "ratio": sd / float(se.mean()) if se.mean() > 0 else float("nan")}
    ratios = [v["ratio"] for v in species.values() if np.isfinite(v["ratio"])]

    def spread(values):
        v = np.array(values, float)
        return {"values": [float(x) for x in v], "mean": float(v.mean()),
                "sd": float(np.std(v, ddof=1)), "se_of_mean": float(np.std(v, ddof=1) / np.sqrt(n))}
    return {"runs": [r["run"] for r in chosen], "species": species,
            "error_factor": float(max(1.0, np.median(ratios))) if ratios else 1.0,
            "ratio_with_factor": spread([r["ratio_with_factor"] for r in chosen]),
            "conductivity": spread([r["whole"]["conductivity"]["sigma"] for r in chosen]),
            "conductivity_block_error": float(np.mean(
                [r["whole"]["conductivity"]["se"] for r in chosen])),
            "ionicity": spread([r["whole"]["ionicity"] for r in chosen]),
            "measured": chosen[0]["measured"]}


def repeat_lines(study: dict) -> List[str]:
    """The section of the table on repeats of one box."""
    out: List[str] = []
    w = out.append
    label = {"cation": "Cation", "anion": "Anion"}
    n = len(study["runs"])
    w("## Repeats of one box")
    w("")
    w(f"The same box was packed and started {n} times ({', '.join(study['runs'])}). The error "
      "over the blocks of one run says how well that run determined its own coefficient. It "
      "cannot see a difference between two runs that start from different arrangements of the "
      "ions, if an arrangement lasts longer than the run. The standard deviation between the "
      "repeats does see it, and the ratio of the two is the test.")
    w("")
    w("| Species | " + " | ".join(f"{r}, m2/s" for r in study["runs"])
      + " | Mean, m2/s | Standard deviation between repeats | Mean block error of one run | "
        "Ratio of the two |")
    w("|---|" + "---:|" * n + "---:|---:|---:|---:|")
    for name, v in study["species"].items():
        w(f"| {label.get(name, name)} | " + " | ".join(_e(x) for x in v["d"])
          + f" | {_e(v['mean'])} | {_e(v['sd'], 1)} | {_e(v['block_error'], 1)} | "
            f"{_f(v['ratio'], 1)} |")
    w("")
    factor = study["error_factor"]
    w(f"The median ratio is {_f(factor, 1)}. "
      + ("The blocks of one run see the whole error of its coefficient."
         if factor < 1.5 else
         f"Two runs of this length differ by {_f(factor, 1)} times what the blocks of either "
         "say, so every block error of a single run in this table is too small by about that "
         "factor, and the size study below multiplies them by it.")
      + f" With {n} repeats a standard deviation is itself known only to within a factor of "
        "about two, so this ratio is a size and not a number.")
    w("")
    r, k, m = study["ratio_with_factor"], study["conductivity"], study["measured"]
    w("Over the repeats, the salt diffusion coefficient with the measured thermodynamic "
      f"factor is {_f(r['mean'])} +/- {_f(r['se_of_mean'])} of the measured one (single runs: "
      + ", ".join(_f(x) for x in r["values"]) + f"), and the conductivity from the total "
      f"charge is {_f(k['mean'], 3)} +/- {_f(k['se_of_mean'], 3)} S/m against the measured "
      f"{_f(m['conductivity'], 3)} (single runs: "
      + ", ".join(_f(x, 3) for x in k["values"])
      + f"; the mean block error of one run is {_f(study['conductivity_block_error'], 3)}). "
        "The error after +/- here is the standard error of the mean over the repeats.")
    w("")
    return out


def charge_study(rows: Sequence[dict]) -> Optional[List[dict]]:
    """The runs of the base box by the factor on the charges of the ions, largest first.

    A factor that was run several times (:data:`REPEATS`) enters with the mean over its repeats
    and the standard error of that mean; a factor that was run once enters with its block
    errors. Returns None when fewer than two factors were run.
    """
    base = [r for r in rows if r["settings"].get("n_solvent", 250) == 250]
    scales = sorted({float(r["settings"]["ion_charge_scale"]) for r in base}, reverse=True)
    if len(scales) < 2:
        return None
    out = []
    for scale in scales:
        runs = [r for r in base if float(r["settings"]["ion_charge_scale"]) == scale]
        n = len(runs)

        def mean(get, error):
            v = np.array([get(r) for r in runs], float)
            if n > 1:
                return float(v.mean()), float(np.std(v, ddof=1) / np.sqrt(n))
            return float(v[0]), float(error(runs[0])) if error is not None else float("nan")
        m = runs[0]["measured"]
        entry = {"scale": scale, "runs": [r["run"] for r in runs],
                 "cation": mean(lambda r: r["whole"]["cation"]["d"],
                                lambda r: r["whole"]["cation"]["se"]),
                 "anion": mean(lambda r: r["whole"]["anion"]["d"],
                               lambda r: r["whole"]["anion"]["se"]),
                 "ratio_with_factor": mean(lambda r: r["ratio_with_factor"], None),
                 "conductivity": mean(lambda r: r["whole"]["conductivity"]["sigma"],
                                      lambda r: r["whole"]["conductivity"]["se"]),
                 "ionicity": mean(lambda r: r["whole"]["ionicity"], None),
                 "exponent": float(np.mean([r["whole"]["cation"]["exponent"] for r in runs])),
                 "measured_conductivity": float(m["conductivity"])}
        out.append(entry)
    return out


def charge_lines(study: Sequence[dict], limits: Sequence[Dict[str, float]]) -> List[str]:
    """The section of the table on the charges of the ions."""
    out: List[str] = []
    w = out.append

    def pm(pair, fmt):
        return fmt(pair[0]) + (f" +/- {fmt(pair[1])}" if np.isfinite(pair[1]) else "")
    w("## The charges of the ions")
    w("")
    w("A force field with fixed charges has no polarisation. It binds a small cation to its "
      "neighbours too strongly, and the ions move too slowly. Multiplying the charges of the "
      "ions by a factor below one is the usual repair. The table gives the base box at each "
      "factor that was run. Where a factor was run several times, the numbers are means over "
      "the repeats with the standard error of the mean; where it was run once, the error is "
      "the block error of that run, which the repeats found too small by a factor of about "
      "1.4. None of these numbers is corrected for the size of the box.")
    w("")
    w("| Factor on the charges | Runs | Cation, m2/s | Anion, m2/s | Exponent of the cation | "
      "Salt coefficient with the measured thermodynamic factor, ratio to the measured | "
      "Rates at which the cell tolerates it | Conductivity, S/m | Ratio to the measured | "
      "Ratio to the Nernst-Einstein value |")
    w("|---:|---:|---|---|---:|---|---|---|---:|---:|")
    for e in study:
        served = served_rates(e["ratio_with_factor"][0], limits)
        text = (", ".join(f"{x:g}C" for x in served) if served else "none") if limits \
            else "not run"
        w(f"| {e['scale']:g} | {len(e['runs'])} | {pm(e['cation'], _e)} | "
          f"{pm(e['anion'], _e)} | {_f(e['exponent'])} | {pm(e['ratio_with_factor'], _f)} | "
          f"{text} | {pm(e['conductivity'], lambda v: _f(v, 3))} | "
          f"{_f(e['conductivity'][0] / e['measured_conductivity'])} | "
          f"{_f(e['ionicity'][0])} |")
    w("")
    first, last = study[0], study[-1]
    w(f"From a factor of {first['scale']:g} to {last['scale']:g} the self-diffusion coefficient "
      f"of the cation rises {_f(last['cation'][0] / first['cation'][0], 1)} times and the "
      f"conductivity {_f(last['conductivity'][0] / first['conductivity'][0], 1)} times. The "
      "conductivity rises faster because the ions also pair less: the last column is the share "
      "of the conductivity that is left when the motion of ions together is counted. "
      "Choosing the factor by the measured conductivity of this one electrolyte is a "
      "calibration of one parameter on one number. The test of it is whether the same factor "
      "gives the diffusion coefficient here and the conductivities of other electrolytes.")
    w("")
    return out


def bracket_lines(whole: dict, rest: dict, limits: Sequence[Dict[str, float]]) -> List[str]:
    """What the two size fits say together: a range, where neither can be preferred."""
    out: List[str] = []
    if not (whole.get("determined") and rest.get("determined")
            and "ratio_with_factor" in whole and "ratio_with_factor" in rest):
        return out
    lo, hi = sorted([rest["ratio_with_factor"], whole["ratio_with_factor"]])
    base = whole["runs"].index(rest["runs"][0])
    add = sorted([rest["fit"]["correction"][0], whole["fit"]["correction"][base]])
    cation = rest["d"]["cation"][0]
    big = rest["largest"]
    served_lo, served_hi = served_rates(lo, limits), served_rates(hi, limits)
    out.append("## The two fits read together")
    out.append("")
    out.append(
        "Neither fit can be preferred. The fit over all boxes gives hydrodynamic radii of a "
        "plausible size and takes its slope from the smallest box, which differs from the "
        "others in every quantity. The fit without the smallest box describes its boxes well "
        f"(a residual of {_f(rest['fit']['chi'], 2)} against {_f(whole['fit']['chi'], 2)}) and "
        "gives radii smaller than an atom, which says that the coefficients of the larger "
        "boxes depend on the size less than the correction predicts for a liquid that "
        "diffuses this fast. The two are therefore reported as a range. In the base box "
        f"(`{rest['runs'][0]}`) the correction adds between {_e(add[0])} and {_e(add[1])} m2/s "
        f"to every coefficient, which is {_f(100 * add[0] / cation, 0)} % to "
        f"{_f(100 * add[1] / cation, 0)} % of the cation's. The salt diffusion coefficient of "
        "an infinite box, with the measured thermodynamic factor, is between "
        f"{_f(lo)} and {_f(hi)} of the measured value; the largest box without any correction "
        f"gives {_f(big['ratio_with_factor'])}."
        + (f" The cell tolerates the lower end at {', '.join(f'{x:g}C' for x in served_lo)}"
           f" and the upper end at {', '.join(f'{x:g}C' for x in served_hi)}."
           if limits and served_lo and served_hi else ""))
    out.append("")
    return out


def size_study(rows: Sequence[dict], exclude: Sequence[str] = ()) -> Optional[dict]:
    """The coefficients of an infinite box, from the runs that differ only in their size.

    Needs at least two of the runs in :data:`SIZE_RUNS` among `rows`, the ones in `exclude`
    left out; returns None otherwise. Where the box of one size was repeated
    (:func:`repeat_study`), it enters with the mean of its repeats and the standard error of
    that mean, and the block errors of the other boxes are multiplied by the factor the repeats
    give, because a single run does not see its whole error.
    The species share one slope against one over the side of the box
    (:func:`cmfqsar.transport.joint_size_fit`). The salt diffusion coefficient and its ratio to
    the measured one are then given for the infinite box, with the measured values of the
    largest box, whose concentration is the best determined.
    """
    from cmfqsar import transport as T

    chosen = sorted((r for r in rows if r["run"] in SIZE_RUNS and r["run"] not in exclude),
                    key=lambda r: r["side"])
    if len(chosen) < 2:
        return None
    repeats = repeat_study(rows)
    factor = repeats["error_factor"] if repeats else 1.0
    names = list(_species(chosen[0]))
    d = {k: [_species(r)[k]["d"] for r in chosen] for k in names}
    se = {k: [_species(r)[k]["se"] * factor for r in chosen] for k in names}
    sides = [r["side"] for r in chosen]
    # the box that was repeated enters with the mean of its repeats and the error of that mean
    repeated = [j for j, r in enumerate(chosen) if repeats and r["run"] in repeats["runs"]]
    for j in repeated:
        for k in names:
            d[k][j] = repeats["species"][k]["mean"]
            se[k][j] = repeats["species"][k]["se_of_mean"]
        sides[j] = float(np.mean([r["side"] for r in rows if r["run"] in repeats["runs"]]))
    temperature = float(chosen[0]["temperature"])
    fit = T.joint_size_fit(d, sides, temperature, se=se)
    separate = {k: T.extrapolate_size(d[k], sides, temperature, se=se[k]) for k in d}
    last = chosen[-1]
    m = last["measured"]
    dp = fit["species"]["cation"]["d_infinite"]
    dm = fit["species"]["anion"]["d_infinite"]
    out = {"runs": [r["run"] for r in chosen], "sides": sides,
           "pairs": [r["pairs"] for r in chosen], "d": d, "se": se, "fit": fit,
           "separate": separate, "measured": m, "error_factor": float(factor),
           "repeated": [chosen[j]["run"] for j in repeated], "excluded": list(exclude)}
    # a slope of the wrong sign determines no correction: the coefficients do not rise with
    # the size of the box, and a line through them must not be read at an infinite box
    out["determined"] = bool(np.isfinite(fit["viscosity"]))
    out["largest"] = {"run": last["run"], "ratio_with_factor": last["ratio_with_factor"],
                      "ratio_ideal": last["ratio_ideal"]}
    out["change"] = {k: float(d[k][-1] / d[k][0] - 1.0) for k in names}
    if out["determined"] and dp > 0 and dm > 0:
        salt = T.salt_diffusivity(dp, dm)
        out.update({"salt_ideal": salt,
                    "salt_with_factor": salt * m["thermodynamic_factor"],
                    "ratio_ideal": salt / m["diffusivity"],
                    "ratio_with_factor": salt * m["thermodynamic_factor"] / m["diffusivity"],
                    "nernst_einstein": T.nernst_einstein_conductivity(
                        dp, dm, last["pairs"], last["side"] ** 3, temperature),
                    "transport_number": T.transport_number(dp, dm)})
    return out


def size_lines(study: dict, limits: Sequence[Dict[str, float]]) -> List[str]:
    """The section of the table on the size of the box."""
    out: List[str] = []
    w = out.append
    fit = study["fit"]
    names = list(study["d"])
    label = {"cation": "Cation", "anion": "Anion"}
    w("## The size of the box")
    w("")
    w("A periodic box makes a diffusion coefficient too small, because a molecule interacts "
      "with its own images through the liquid. The amount is 2.837 k T / (6 pi eta L) for a "
      "cubic box of the side L and a liquid of the viscosity eta (Yeh and Hummer, 2004). It is "
      "the same amount for every species of one liquid. The runs below differ only in the "
      "number of molecules. One straight line per species is fitted against 1 / L, with one "
      "slope for all of them, so that the solvents, which have many molecules, fix the slope "
      "for the ions, which have few. The value of a line at 1 / L = 0 is the coefficient of an "
      "infinite box.")
    w("")
    if study.get("repeated"):
        w(f"The box of the run `{study['repeated'][0]}` was repeated, so it enters with the "
          "mean of its repeats and the standard error of that mean. The errors of the other "
          f"boxes are their block errors multiplied by {_f(study['error_factor'], 1)}, the "
          "factor the repeats give.")
        w("")
    if study.get("excluded"):
        w("Left out of this fit: " + ", ".join(f"`{x}`" for x in study["excluded"]) + ".")
        w("")
    w("| Run | Ion pairs | Side, A | " + " | ".join(f"{label.get(n, n)}, m2/s" for n in names)
      + " | What the correction adds, m2/s |")
    w("|---|---:|---:|" + "---|" * len(names) + "---:|")
    for j, run in enumerate(study["runs"]):
        cells = [run, str(study["pairs"][j]), _f(study["sides"][j])]
        cells += [f"{_e(study['d'][n][j])} +/- {_e(study['se'][n][j], 1)}" for n in names]
        cells.append(_e(fit["correction"][j]) if study["determined"] else "not determined")
        w("| " + " | ".join(cells) + " |")
    if not study["determined"]:
        w("")
        change = study["change"]
        low, high = min(change.values()), max(change.values())
        big = study["largest"]
        served = served_rates(big["ratio_with_factor"], limits)
        w("**These boxes determine no correction.** The correction says that a coefficient "
          "rises as the box grows. Here the common slope has the other sign: from the smallest "
          "box of this fit to the largest, the coefficients change by "
          f"{_f(100 * low, 0)} % to {_f(100 * high, 0)} %. A line through them is therefore not "
          "read at an infinite box. The best estimate is the largest box as it is, without a "
          f"correction: in `{big['run']}` the salt diffusion coefficient with the measured "
          f"thermodynamic factor is {_f(big['ratio_with_factor'])} of the measured value"
          + (f", which the cell tolerates at {', '.join(f'{x:g}C' for x in served)}"
             if served else (", which the cell tolerates at none of the four rates"
                             if limits else ""))
          + ". A correction that exists and was not resolved would raise it.")
        w("")
        return out
    w("| infinite box | | | "
      + " | ".join(_e(fit["species"][n]["d_infinite"]) for n in names) + " | 0 |")
    w("")
    w(f"The common slope gives a viscosity of {_f(1000.0 * fit['viscosity'], 2)} mPa s. The "
      f"lines leave a root-mean-square residual of {_f(fit['chi'], 2)} in units of the errors "
      f"of the coefficients; a value near 1 says that one slope describes all species. "
      "The hydrodynamic radius of a species is the radius of a sphere that would diffuse as "
      "fast in a liquid of that viscosity: "
      + ", ".join(f"{label.get(n, n).lower() if n in label else n} "
                  f"{_f(fit['species'][n]['radius'], 1)} A" for n in names) + ".")
    w("")
    w("Fitted one species at a time, without the common slope, the same boxes give: "
      + "; ".join(f"{label.get(n, n).lower() if n in label else n} "
                  f"{_e(study['separate'][n]['d_infinite'])} m2/s"
                  + (f" with a viscosity of "
                     f"{_f(1000.0 * study['separate'][n]['viscosity'], 2)} mPa s"
                     if np.isfinite(study["separate"][n]["viscosity"])
                     else " with a slope of the wrong sign")
                  for n in names)
      + ". The spread of these says how well the boxes determine the slope.")
    w("")
    if "salt_ideal" in study:
        m = study["measured"]
        served = served_rates(study["ratio_with_factor"], limits)
        w("**In an infinite box** the salt diffusion coefficient of an ideal solution is "
          f"{_e(study['salt_ideal'])} m2/s, {_f(study['ratio_ideal'])} of the measured "
          f"{_e(m['diffusivity'])}. Multiplied by the measured thermodynamic factor it is "
          f"{_e(study['salt_with_factor'])} m2/s, {_f(study['ratio_with_factor'])} of the "
          "measured value"
          + (f", which the cell tolerates at {', '.join(f'{x:g}C' for x in served)}"
             if served else (", which the cell tolerates at none of the four rates"
                             if limits else ""))
          + f". The Nernst-Einstein conductivity of the infinite box is "
            f"{_f(study['nernst_einstein'], 3)} S/m against the measured "
            f"{_f(m['conductivity'], 3)}, and the share of the current on the cation, if the "
            f"ions moved alone, is {_f(study['transport_number'], 3)}.")
        w("")
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _e(v, digits: int = 2) -> str:
    """A number in the form 2.69e-10, or "none"."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "none"
    return f"{x:.{digits}e}" if np.isfinite(x) else "none"


def _f(v, digits: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "none"
    return f"{x:.{digits}f}" if np.isfinite(x) else "none"


def tolerance_limits(results: Optional[str] = None) -> List[Dict[str, float]]:
    """The lowest factor of the salt diffusion coefficient the cell tolerates at room
    temperature, per rate, for a capacity that may move by 5 %: what the tolerance stage of
    ``run_cellmodel.py`` stored. Empty when that stage has not been run."""
    folder = os.path.join(HERE, "results", "cellmodel") if results is None else results
    out = []
    for rate in (1.0, 1.5, 2.0, 2.5):
        path = os.path.join(folder, f"tolerance_{ROOM:.0f}K_{rate:g}C.json")
        if not os.path.isfile(path):
            return []
        with open(path, encoding="utf-8") as fh:
            entry = json.load(fh)
        r = entry["parameters"]["Electrolyte diffusivity [m2.s-1]"]["ranges"]["0.05"]
        out.append({"rate": rate, "low": float(r["low"]), "high": float(r["high"]),
                    "high_open": r["high_ends"] == "open"})
    return out


def served_rates(ratio: float, limits: Sequence[Dict[str, float]]) -> List[float]:
    """The rates at which a coefficient that is `ratio` times the measured one is inside the
    tolerated range."""
    return [lim["rate"] for lim in limits
            if ratio >= lim["low"] and (lim["high_open"] or ratio <= lim["high"])]


def read_results(names: Sequence[str], results: str = RESULTS) -> List[dict]:
    rows = []
    for name in names:
        path = os.path.join(results, f"run_{name}.json")
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                rows.append(json.load(fh))
    return rows


def report_lines(rows: Sequence[dict], limits: Sequence[Dict[str, float]]) -> List[str]:
    """The table of this benchmark as lines of Markdown."""
    out: List[str] = []
    w = out.append
    first = rows[0]
    m = first["measured"]
    w("# Item BT9: the salt diffusion coefficient of an electrolyte from simulation")
    w("")
    w("The electrolyte is that of the reference cell: lithium hexafluorophosphate in ethylene "
      "carbonate and ethyl methyl carbonate, 3 to 7 by weight, at "
      f"{MOLALITY:g} mol of salt per kilogram of solvent and {first['temperature']:.0f} K. "
      "Every run is one box of the GAFF force field with charges from the OpenFF graph "
      "network, equilibrated at constant pressure and then run at constant volume with a "
      "Nose-Hoover thermostat. The unwrapped centre of every molecule is stored once per "
      "picosecond, and every number below is computed from those centres "
      "(`cmfqsar/transport.py`).")
    w("")
    w("A self-diffusion coefficient is the slope of the mean squared displacement of one "
      "species against time, divided by six, fitted between a fortieth and a tenth of the "
      "run. The *exponent* is the power with which the mean squared displacement grows in "
      "that window: 1 is diffusion, and a value below 1 says that the molecules had not left "
      "their cages yet and the coefficient is an overestimate of the long-time value. The "
      "error after +/- is the standard error over five blocks of the run. It says how well "
      "the run determined the number and says nothing about the force field.")
    w("")
    w("## The boxes")
    w("")
    w("| Run | What it changes | Atoms | Ion pairs | Side of the box, A | Density, g/cm3 | "
      "Concentration, mol/L | Mean kinetic temperature, K | Length, ns | Speed, ns/day |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        w(f"| {r['run']} | {r['label']} | {r['atoms']} | {r['pairs']} | {_f(r['side'])} | "
          f"{_f(r['density'], 3)} | {_f(r['molarity'], 3)} | "
          f"{_f(r['mean_kinetic_temperature'], 1)} | "
          f"{_f(r['whole']['length_ps'] / 1000.0, 1)} | {_f(r['ns_per_day'], 0)} |")
    w("")
    w("## The simulated electrolyte against the measured one")
    w("")
    w(f"The measured values are those of Landesfeind and Gasteiger (2019) for this "
      f"electrolyte, read from the functions PyBaMM ships at the concentration of the box: "
      f"salt diffusion coefficient {_e(m['diffusivity'])} m2/s, conductivity "
      f"{_f(m['conductivity'], 3)} S/m, transference number {_f(m['transference'], 3)}, "
      f"thermodynamic factor {_f(m['thermodynamic_factor'], 2)}."
      + (f" CALiSol-23 gives a conductivity of {_f(first['calisol_conductivity'], 3)} S/m for "
         f"the same formulation." if first.get("calisol_conductivity") else ""))
    w("")
    w("**Self-diffusion, m2/s.**")
    w("")
    solvents = sorted(first["whole"]["solvents"])
    w("| Run | Cation | Exponent | Anion | Exponent | "
      + " | ".join(f"{s} | Exponent" for s in solvents) + " |")
    w("|---|---|---:|---|---:|" + "---|---:|" * len(solvents))
    for r in rows:
        x = r["whole"]
        cells = [r["run"]]
        for rec in [x["cation"], x["anion"]] + [x["solvents"][s] for s in solvents]:
            cells += [f"{_e(rec['d'])} +/- {_e(rec['se'], 1)}", _f(rec["exponent"])]
        w("| " + " | ".join(cells) + " |")
    w("")
    w("**The salt diffusion coefficient, m2/s.** For an ideal solution it is "
      "2 D+ D- / (D+ + D-). A concentrated solution multiplies that by the thermodynamic "
      "factor, and the measured factor is used here, because this program has no prediction "
      "of it (plan item BT11). The ratio is to the measured coefficient.")
    w("")
    w("| Run | Ideal solution | Ratio to the measured | With the measured thermodynamic "
      "factor | Ratio to the measured | Rates at which the cell tolerates it |")
    w("|---|---:|---:|---:|---:|---|")
    for r in rows:
        served = served_rates(r["ratio_with_factor"], limits)
        text = (", ".join(f"{x:g}C" for x in served) if served else "none") if limits \
            else "the tolerance stage has not been run"
        w(f"| {r['run']} | {_e(r['salt_ideal'])} | {_f(r['ratio_ideal'])} | "
          f"{_e(r['salt_with_factor'])} | {_f(r['ratio_with_factor'])} | {text} |")
    w("")
    if limits:
        w("The last column uses the tolerance stage of `run_cellmodel.py`: for a capacity that "
          "may move by 5 % at 298 K the coefficient may be no lower than "
          + ", ".join(f"{_f(lim['low'])} of the measured value at {lim['rate']:g}C"
                      for lim in limits)
          + ", and every larger value that was tried is tolerated.")
        w("")
    w("None of these coefficients is corrected for the size of the box. A periodic box makes a "
      "diffusion coefficient too small, so the correction raises every one of them.")
    w("")
    w("**The conductivity, S/m.** The first value is from the displacement of the total "
      "charge and includes the motion of ions together. The Nernst-Einstein value is what the "
      "two self-diffusion coefficients would give if every ion moved on its own. Their ratio "
      "says how far the ions are from moving on their own. Both use the whole charges of the "
      "ions, also in the run that scales them in the force field.")
    w("")
    w("| Run | From the total charge | Nernst-Einstein value | Ratio of the two | Ratio of the "
      "first to the measured | Share of the current on the cation if ions moved alone | "
      "Measured transference number |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for r in rows:
        x = r["whole"]
        w(f"| {r['run']} | {_f(x['conductivity']['sigma'], 3)} +/- "
          f"{_f(x['conductivity']['se'], 3)} | {_f(x['nernst_einstein'], 3)} | "
          f"{_f(x.get('ionicity'))} | {_f(r['ratio_conductivity'])} | "
          f"{_f(x['transport_number'], 3)} | {_f(m['transference'], 3)} |")
    w("")
    w("## How the error falls with the length of the run")
    w("")
    w("One run is read at several lengths, each from its start. The window of the fit moves "
      "with the length, so a shorter reading also sits earlier in the caged regime, which the "
      "exponent shows.")
    w("")
    w("| Run | Length, ns | Cation, m2/s | Exponent | Anion, m2/s | Exponent | Salt, ideal, "
      "m2/s | Conductivity, S/m |")
    w("|---|---:|---|---:|---|---:|---:|---|")
    for r in rows:
        for part in r["lengths"]:
            c, a, k = part["cation"], part["anion"], part["conductivity"]
            w(f"| {r['run']} | {_f(part['length_ps'] / 1000.0, 1)} | {_e(c['d'])} +/- "
              f"{_e(c['se'], 1)} | {_f(c['exponent'])} | {_e(a['d'])} +/- {_e(a['se'], 1)} | "
              f"{_f(a['exponent'])} | {_e(part['salt'])} | {_f(k['sigma'], 3)} +/- "
              f"{_f(k['se'], 3)} |")
    w("")
    repeats = repeat_study(rows)
    if repeats is not None:
        out.extend(repeat_lines(repeats))
    charges = charge_study(rows)
    if charges is not None:
        out.extend(charge_lines(charges, limits))
    study = size_study(rows)
    if study is not None:
        out.extend(size_lines(study, limits))
        smallest = study["runs"][0]
        rest = size_study(rows, exclude=(smallest,))
        if rest is not None:
            lines = size_lines(rest, limits)
            lines[0] = "## The size of the box, without the smallest box"
            lines[2] = (f"The same fit without `{smallest}`, the smallest box. If the two fits "
                        "disagree, the smallest box is too small for the correction to "
                        "describe it.")
            out.extend(lines)
            out.extend(bracket_lines(study, rest, limits))
    return out


def report(results: str = RESULTS, names: Optional[Sequence[str]] = None) -> str:
    """Write the table from the stored results. Returns the path of the table."""
    rows = read_results(list(RUNS) if names is None else names, results)
    if not rows:
        raise SystemExit(f"{results} holds no result: run this benchmark with --stage analyse")
    path = os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines(rows, tolerance_limits())) + "\n")
    return path


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--stage", action="append", choices=STAGES,
                    help="run only these stages (default: analyse; simulate needs the "
                         "simulation environment and is never run unasked)")
    ap.add_argument("--runs", default=",".join(RUNS),
                    help="the runs, by name and separated by commas (default: all of them)")
    ap.add_argument("--report", action="store_true",
                    help="write the table from the stored results and stop")
    ap.add_argument("--results", default=RESULTS, help="where the results are written")
    ap.add_argument("--cache", default=CACHE, help="where the simulations are kept")
    args = ap.parse_args(argv)
    names = [n for n in args.runs.split(",") if n]
    if not args.report:
        stages = tuple(args.stage or ("analyse",))
        if "simulate" in stages:
            stage_simulate(names, args.cache)
        if "analyse" in stages:
            stage_analyse(names, args.results, args.cache)
        if "analyse" not in stages:
            return 0
    log(f"the table: {report(args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
