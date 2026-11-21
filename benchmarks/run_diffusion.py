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
* ``conc_c0p1`` to ``conc_c3p0``, and the same names with ``_seed2``: the concentration series
  of gate (a). The measured study has six base concentrations from 0.1 to 3.0 mol/L. Each is
  simulated in a box of 1,000 solvent molecules with the charges of the ions multiplied by
  0.7, and each box is packed and started twice. :data:`CONCENTRATIONS` holds the molalities
  that give those concentrations.
* ``conc_c1p1_long`` and ``conc_c3p0_long``: two boxes of the series run for 20 ns in place of
  4 ns. They say how much a short run overstates a coefficient.
* ``form_<solvents>``: rule 2 on other solvents. One formulation of each of the 22 solvent
  compositions of CALiSol-23, the one nearest to the molality of the reference cell, simulated
  with the settings of the series. Their conductivities against the measured ones say whether
  the factor on the charges, chosen on one electrolyte, holds for other solvents.
  :data:`FORMULATION_RUNS` lists them, and EC:EMC 3:7 is the box of the series.
* ``form_<solvents>_m<molality>``: the 40 other formulations of CALiSol-23, each at its own
  molality (:data:`REST_NAMES`). They are simulated with the same settings. The table reads
  them together with the 22 boxes of the transfer test (:func:`all_formulations_study`).

The cell stage
--------------
Gate (a) compares two coefficients, and the cell stage asks what their difference costs. The
reference cell of ``run_cellmodel.py`` is discharged at 298 K and four rates with the measured
electrolyte, and again with the measured salt diffusion coefficient multiplied by the ratio
that the series found at each concentration. The change of the capacity is compared with the
5 % that the tolerance stage allows.

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
    python benchmarks/run_diffusion.py --stage cell        # the series in the cell model; PyBaMM
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
#: a second published set for the same electrolyte: its diffusion coefficient and conductivity
#: are the functions of Nyman, Behm and Lindbergh (2008), measured at 298 K only and between
#: 0.2 and 2 mol/L
SECOND_MEASURED = "Chen2020"
SECOND_RANGE = (0.2, 2.0)
#: the only number the measured study gives for the uncertainty of its diffusion coefficient. Its
#: error bars are the scatter of two cells with two pulses each, and the authors write that
#: these do not include the systematic error of building a cell. They report that a repeat at
#: the end of a temperature cycle was up to 10 % lower and that this is inside the uncertainty
#: of the measurement. Gate (a) is read with this number. It was fixed before the series ran.
STUDY_UNCERTAINTY = 0.10

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
#: the concentration series of gate (a): the six salt concentrations of the measured study
#: (0.1, 0.5, 1.0, 1.5, 2.0 and 3.0 mol/L) as the molality that gives about each of them. The
#: third is the molality of the reference cell, so that it joins the other runs. The measured
#: functions are read at the concentration the box really has, so these need not be exact.
CONCENTRATIONS: Dict[str, float] = {"c0p1": 0.092, "c0p5": 0.47, "c1p1": MOLALITY,
                                    "c1p5": 1.50, "c2p0": 2.07, "c3p0": 3.32}
#: the factor on the charges of the ions and the number of solvent molecules of the series: what
#: the first part of the item chose
SERIES_SETTINGS = {"ion_charge_scale": 0.7, "n_solvent": 1000}
for _key, _molality in CONCENTRATIONS.items():
    RUNS[f"conc_{_key}"] = {
        "label": f"{_molality:g} mol/kg, charges multiplied by 0.7, 1,000 solvent molecules",
        "molality": _molality, "settings": dict(SERIES_SETTINGS)}
    RUNS[f"conc_{_key}_seed2"] = {
        "label": f"{_molality:g} mol/kg, a second start",
        "molality": _molality, "settings": dict(SERIES_SETTINGS, seed=2)}
#: the runs of the concentration series, by concentration: the first start and its repeat
CONCENTRATION_RUNS: Dict[str, tuple] = {k: (f"conc_{k}", f"conc_{k}_seed2")
                                        for k in CONCENTRATIONS}
#: two boxes of the series run five times as long, with the centres stored every 2 ps. In a
#: short run the fit sits where the molecules have not all left their cages, and the long run
#: says how much the short one overstates. One is the concentration of the reference cell and
#: the other the highest concentration, where the short run is furthest from diffusion.
LONG_RUNS: Dict[str, str] = {"c1p1": "conc_c1p1_long", "c3p0": "conc_c3p0_long"}
for _key, _name in LONG_RUNS.items():
    RUNS[_name] = {
        "label": f"{CONCENTRATIONS[_key]:g} mol/kg, 20 ns",
        "molality": CONCENTRATIONS[_key],
        "settings": dict(SERIES_SETTINGS, production_ps=20000.0, sample_ps=2.0)}
#: rule 2 on other solvents. CALiSol-23 holds the measured conductivity of 62 formulations of
#: LiPF6 in 22 solvent compositions. One formulation of each composition is simulated with the
#: settings of the series: the one whose molality is nearest to that of the reference cell, so
#: that the solvent changes and the concentration stays near 1 mol/kg. For EC:EMC 3:7 that is
#: the box of the series itself, whose two starts are used. The rule was fixed before any of
#: these boxes was run.
FORMULATIONS_CSV = os.path.join(ROOT, "examples", "electrolyte", "formulations.csv")
FORMULATION_TARGET = MOLALITY
#: weight fraction above which a solvent counts as part of a composition in the reading by solvent
SOLVENT_SHARE = 0.20


def calisol_rows(path: str = FORMULATIONS_CSV) -> List[dict]:
    """The formulations of CALiSol-23 as this repository holds them: id, molality, solvent weight
    fractions and the measured conductivity at 298 K in S/m."""
    import csv
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            solvents = {k[2:]: float(v) for k, v in row.items()
                        if k.startswith("w_") and v and float(v) > 0}
            out.append({"id": row["id"], "salt": row["salt"], "molality": float(row["molality"]),
                        "solvents": solvents,
                        "conductivity": 0.1 * 10.0 ** float(row["log10_k_298"])})  # mS/cm to S/m
    return out


def chosen_formulations(rows: Sequence[dict], target: float = FORMULATION_TARGET) -> List[dict]:
    """One formulation of every solvent composition: the one whose molality is nearest to
    `target`, the lower molality on a tie. In the order in which they are run: sorted by the
    measured conductivity and taken alternately from the lowest and the highest, so that a part
    of the runs already spans the range."""
    by_comp: Dict[str, List[dict]] = {}
    for r in rows:
        by_comp.setdefault(r["id"].rsplit("_m", 1)[0], []).append(r)
    chosen = [min(v, key=lambda r: (abs(r["molality"] - target), r["molality"]))
              for v in by_comp.values()]
    return alternating_by_conductivity(chosen)


def alternating_by_conductivity(rows: Sequence[dict]) -> List[dict]:
    """The formulations sorted by the measured conductivity and taken alternately from the lowest
    and the highest, so that the first part of a list already spans the range."""
    ordered = sorted(rows, key=lambda r: r["conductivity"])
    order = []
    lo, hi = 0, len(ordered) - 1
    while lo <= hi:
        order.append(ordered[lo])
        if hi != lo:
            order.append(ordered[hi])
        lo, hi = lo + 1, hi - 1
    return order


def formulation_run_name(fid: str) -> str:
    """The name of the run of one formulation: ``form_`` and the solvents, e.g. form_DEC20_PC80."""
    return "form_" + fid.split("_", 1)[1].rsplit("_m", 1)[0]


#: the run(s) of each composition, by the CALiSol id: a new box for every composition but that
#: of the reference cell, which is the box of the series
FORMULATION_RUNS: Dict[str, tuple] = {}
for _row in chosen_formulations(calisol_rows()):
    if _row["id"] == ELECTROLYTE_ID:
        FORMULATION_RUNS[_row["id"]] = CONCENTRATION_RUNS["c1p1"]
        continue
    _name = formulation_run_name(_row["id"])
    FORMULATION_RUNS[_row["id"]] = (_name,)
    RUNS[_name] = {"label": f"{_row['id']}, charges multiplied by 0.7, 1,000 solvent molecules",
                   "molality": _row["molality"], "solvents": dict(_row["solvents"]),
                   "formulation": _row["id"], "settings": dict(SERIES_SETTINGS)}
#: the new boxes of the transfer test, in the order in which they are run
FORMULATION_NAMES = [n for names in FORMULATION_RUNS.values() for n in names
                     if n.startswith("form_")]
#: the other formulations of CALiSol-23, the 40 that the transfer test did not take: every one is
#: simulated at its own molality with the same settings. A run is named by the id of the
#: formulation without the salt, for example form_DEC20_PC80_m0p282, because several
#: formulations of one composition differ only in the molality. The names are in the order in
#: which they are run: alternately from the lowest and the highest measured conductivity.
REST_NAMES: List[str] = []
for _row in alternating_by_conductivity([r for r in calisol_rows()
                                         if r["id"] not in FORMULATION_RUNS]):
    _name = "form_" + _row["id"].split("_", 1)[1]
    REST_NAMES.append(_name)
    RUNS[_name] = {"label": f"{_row['id']}, charges multiplied by 0.7, 1,000 solvent molecules",
                   "molality": _row["molality"], "solvents": dict(_row["solvents"]),
                   "formulation": _row["id"], "settings": dict(SERIES_SETTINGS)}
#: the smaller box that has the charge factor and the concentration of one box of the series,
#: and the key of that box of the series: two sizes at the factor that the series uses
PAIR_SMALL = "scaled_07"
PAIR_KEY = "c1p1"
#: the runs that differ only in the size of the box, from which the correction for a finite box
#: is read
SIZE_RUNS = ("scaled_small", "scaled", "scaled_large", "scaled_huge")
#: the runs that differ only in how the box was packed and in the first velocities: their
#: scatter is the error that the blocks of one run cannot see
REPEATS = ("scaled", "scaled_seed2", "scaled_seed3")
#: the lengths, in picoseconds, at which one run is read to see how its error falls
LENGTHS = (500.0, 1000.0, 2000.0, 4000.0, 10000.0, 20000.0)
STAGES = ("simulate", "analyse", "cell")
#: the share of the capacity that the cell may move by, as in the tolerance stage
CELL_SHARE = 0.05


def log(msg: str) -> None:
    print(msg, flush=True)


def molality_of(name: str) -> float:
    """Moles of salt per kilogram of solvent in one run."""
    return float(RUNS.get(name, {}).get("molality", MOLALITY))


def solvents_of(name: str) -> Dict[str, float]:
    """The solvent weight fractions of one run."""
    return dict(RUNS.get(name, {}).get("solvents", SOLVENTS))


def run_dir(name: str, cache: str = CACHE) -> str:
    # the same name as Formulation.label() gives, so a box says which electrolyte it holds
    if "formulation" in RUNS.get(name, {}):
        return os.path.join(cache, f"{RUNS[name]['formulation']}_{name}")
    stem = ELECTROLYTE_ID.rsplit("_m", 1)[0]
    return os.path.join(cache, f"{stem}_m{molality_of(name):.3f}".replace(".", "p") + f"_{name}")


def formulation(temperature: float = ROOM, molality: float = MOLALITY,
                solvents: Optional[Dict[str, float]] = None):
    from cmfqsar.electrolyte import Formulation
    return Formulation(SALT, dict(solvents or SOLVENTS), molality, temperature=temperature)


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
        run = run_transport(formulation(s.temperature, molality_of(name), solvents_of(name)),
                            run_dir(name, cache), s, progress=log)
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


def second_reference(molarity: float, temperature: float) -> Optional[Dict[str, float]]:
    """The diffusion coefficient and the conductivity of this electrolyte as Nyman, Behm and
    Lindbergh (2008) measured them, from the functions PyBaMM ships. None away from 298 K,
    where that study has no measurement. `inside` says whether the concentration is one the
    study covered."""
    if abs(float(temperature) - ROOM) > 1.0:
        return None
    import run_cellmodel as RC
    from cmfqsar import cellmodel as CM

    four = CM.measured_electrolyte(source=SECOND_MEASURED, coldest=None)
    c = 1000.0 * float(molarity)
    return {"conductivity": RC.electrolyte_value(four[RC.CONDUCTIVITY], temperature, c),
            "diffusivity": RC.electrolyte_value(four[RC.DIFFUSIVITY], temperature, c),
            "inside": bool(SECOND_RANGE[0] <= float(molarity) <= SECOND_RANGE[1]),
            "source": f"{SECOND_MEASURED} in PyBaMM: Nyman, Behm and Lindbergh (2008)"}


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


def _lengths(run: dict, whole: dict) -> List[dict]:
    """The coefficients of one run read at the lengths of :data:`LENGTHS` that it reaches."""
    from cmfqsar import transport as T
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
    return lengths


def analyse_formulation(name: str, cache: str = CACHE) -> dict:
    """One box of another solvent composition: its coefficients and conductivity against the
    conductivity that CALiSol-23 holds for the formulation. No measured diffusion coefficient
    exists for these, so no ratio of one is formed."""
    from cmfqsar import transport as T

    run = read_run(name, cache)
    whole = T.analyse(run)
    fid = RUNS[name]["formulation"]
    measured = {r["id"]: r for r in calisol_rows()}.get(fid)
    sigma = measured["conductivity"] if measured else float("nan")
    return {"run": name, "label": RUNS[name]["label"], "formulation": fid,
            "solvents": solvents_of(name), "molality": molality_of(name),
            "settings": run["settings"], "atoms": run["atoms"], "pairs": run["pairs"],
            "side": run["side"], "density": run["density"], "molarity": whole["molarity"],
            "realised_molality": run["realised_molality"], "temperature": run["temperature"],
            "mean_kinetic_temperature": run["mean_kinetic_temperature"],
            "ns_per_day": run["ns_per_day"], "seconds": run["seconds"],
            "whole": whole, "lengths": _lengths(run, whole),
            "measured_conductivity": sigma,
            "ratio_conductivity": whole["conductivity"]["sigma"] / sigma,
            "ratio_nernst_einstein": whole["nernst_einstein"] / sigma}


def analyse_run(name: str, cache: str = CACHE) -> dict:
    from cmfqsar import transport as T

    if "formulation" in RUNS.get(name, {}):
        return analyse_formulation(name, cache)
    run = read_run(name, cache)
    whole = T.analyse(run)
    ref = measured_reference(whole["molarity"], float(run["temperature"]))
    lengths = _lengths(run, whole)
    factor = ref["thermodynamic_factor"]
    out = {"run": name, "label": RUNS[name]["label"], "settings": run["settings"],
           "atoms": run["atoms"], "pairs": run["pairs"], "side": run["side"],
           "density": run["density"], "molarity": whole["molarity"],
           "realised_molality": run["realised_molality"],
           "temperature": run["temperature"],
           "mean_kinetic_temperature": run["mean_kinetic_temperature"],
           "ns_per_day": run["ns_per_day"], "seconds": run["seconds"],
           "whole": whole, "lengths": lengths, "measured": ref,
           "second_measured": second_reference(whole["molarity"], float(run["temperature"])),
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
        if "formulation" in res:
            log(f"{name}: D+ {w['cation']['d']:.3e} (exponent {w['cation']['exponent']:.2f}), "
                f"D- {w['anion']['d']:.3e}; conductivity {w['conductivity']['sigma']:.3f} +/- "
                f"{w['conductivity']['se']:.3f} S/m against {res['measured_conductivity']:.3f} "
                f"(ratio {res['ratio_conductivity']:.2f})")
        else:
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


def _pair(values: Sequence[float], single_error: float = float("nan")) -> List[float]:
    """The mean of the starts of one box and its standard error. For two starts the standard
    error of the mean is half their difference. One start has the error given, or none."""
    v = np.array(values, float)
    if len(v) > 1:
        return [float(v.mean()), float(np.std(v, ddof=1) / np.sqrt(len(v)))]
    return [float(v[0]), float(single_error)]


def concentration_study(rows: Sequence[dict],
                        uncertainty: float = STUDY_UNCERTAINTY) -> Optional[dict]:
    """The concentration series: the simulated electrolyte against the measured one at every
    concentration that was run (gate (a) of plan item BT9).

    Uses the runs of :data:`CONCENTRATION_RUNS` among `rows`; returns None when there is none.
    Every concentration enters with the mean over its starts. `inside` says whether the salt
    diffusion coefficient, with the measured thermodynamic factor, is within `uncertainty` of
    the measured one; `inside_with_error` says whether the mean plus or minus two standard
    errors reaches that band, and is None where one start gives no error. `decided` is True
    when every concentration of the series was run with at least two starts.
    """
    by_run = {r["run"]: r for r in rows}
    points = []
    for key, names in CONCENTRATION_RUNS.items():
        runs = [by_run[n] for n in names if n in by_run]
        if not runs:
            continue
        m = runs[0]["measured"]
        molarity = float(np.mean([r["molarity"] for r in runs]))
        # the two starts of one box differ in concentration by less than the last digit
        # shown, so the measured values of the first start stand for both
        ratio = _pair([r["ratio_with_factor"] for r in runs])
        point = {
            "key": key, "runs": [r["run"] for r in runs], "starts": len(runs),
            "molality": molality_of(names[0]), "molarity": molarity,
            "pairs": int(runs[0]["pairs"]),
            "cation": _pair([r["whole"]["cation"]["d"] for r in runs],
                            runs[0]["whole"]["cation"]["se"]),
            "anion": _pair([r["whole"]["anion"]["d"] for r in runs],
                           runs[0]["whole"]["anion"]["se"]),
            "exponent": float(np.mean([r["whole"]["cation"]["exponent"] for r in runs])),
            "salt_ideal": _pair([r["salt_ideal"] for r in runs]),
            "salt_with_factor": _pair([r["salt_with_factor"] for r in runs]),
            "ratio_ideal": _pair([r["ratio_ideal"] for r in runs]),
            "ratio_with_factor": ratio,
            "conductivity": _pair([r["whole"]["conductivity"]["sigma"] for r in runs],
                                  runs[0]["whole"]["conductivity"]["se"]),
            "nernst_einstein": _pair([r["whole"]["nernst_einstein"] for r in runs]),
            "ionicity": _pair([r["whole"]["ionicity"] for r in runs]),
            "transport_number": _pair([r["whole"]["transport_number"] for r in runs]),
            "measured": {k: float(m[k]) for k in ("diffusivity", "conductivity", "transference",
                                                  "thermodynamic_factor")},
            "second": runs[0].get("second_measured"),
        }
        point["ratio_conductivity"] = point["conductivity"][0] / point["measured"]["conductivity"]
        point["inside"] = bool(abs(ratio[0] - 1.0) <= uncertainty)
        point["inside_with_error"] = (
            bool(ratio[0] - 2.0 * ratio[1] <= 1.0 + uncertainty
                 and ratio[0] + 2.0 * ratio[1] >= 1.0 - uncertainty)
            if np.isfinite(ratio[1]) else None)
        points.append(point)
    if not points:
        return None
    points.sort(key=lambda p: p["molarity"])
    best = max(points, key=lambda p: p["conductivity"][0])
    best_measured = max(points, key=lambda p: p["measured"]["conductivity"])
    complete = len(points) == len(CONCENTRATION_RUNS)
    return {"points": points, "uncertainty": float(uncertainty), "complete": complete,
            "decided": complete and all(p["starts"] >= 2 for p in points),
            "inside": [p["key"] for p in points if p["inside"]],
            "largest_conductivity": {"simulated": best["molarity"],
                                     "measured": best_measured["molarity"]},
            "passes": all(p["inside"] for p in points)}


def long_study(rows: Sequence[dict]) -> Optional[List[dict]]:
    """The long runs of :data:`LONG_RUNS` beside the short runs of the same box.

    For every concentration that has a long run and at least one short run among `rows`: the
    coefficients and exponents of the long run, read with its own later window, and the ratio of
    the short runs' mean to it. `overstated` is that ratio for the salt diffusion coefficient: a
    value above one says by how much a short run overstates. None when no such pair is there.
    """
    by_run = {r["run"]: r for r in rows}
    out = []
    for key, name in LONG_RUNS.items():
        short = [by_run[n] for n in CONCENTRATION_RUNS[key] if n in by_run]
        if name not in by_run or not short:
            continue
        long = by_run[name]
        entry = {"key": key, "run": name, "short_runs": [r["run"] for r in short],
                 "molarity": float(long["molarity"]),
                 "length_ps": float(long["whole"]["length_ps"]),
                 "window_ps": list(long["whole"]["cation"]["window_ps"]),
                 "short_window_ps": list(short[0]["whole"]["cation"]["window_ps"]),
                 "ratio_with_factor": float(long["ratio_with_factor"]),
                 "short_ratio_with_factor": float(np.mean([r["ratio_with_factor"]
                                                           for r in short])),
                 "conductivity": dict(long["whole"]["conductivity"]),
                 "measured_conductivity": float(long["measured"]["conductivity"]),
                 "species": {}}
        for sp in _species(long):
            rec = _species(long)[sp]
            mean = float(np.mean([_species(r)[sp]["d"] for r in short]))
            entry["species"][sp] = {
                "d": float(rec["d"]), "se": float(rec["se"]), "exponent": float(rec["exponent"]),
                "short_d": mean,
                "short_exponent": float(np.mean([_species(r)[sp]["exponent"] for r in short])),
                "short_over_long": mean / float(rec["d"])}
        entry["overstated"] = entry["short_ratio_with_factor"] / entry["ratio_with_factor"]
        out.append(entry)
    return out or None


def long_lines(study: Sequence[dict]) -> List[str]:
    """The section of the table on the long runs."""
    out: List[str] = []
    w = out.append
    label = {"cation": "Cation", "anion": "Anion"}
    w("## How much a short run overstates")
    w("")
    first = study[0]
    w("A coefficient is fitted between a fortieth and a tenth of the run. In a run of "
      f"{first['length_ps'] / 5000.0:.0f} ns that window is {first['short_window_ps'][0]:.0f} "
      f"to {first['short_window_ps'][1]:.0f} ps, and in a run of "
      f"{first['length_ps'] / 1000.0:.0f} ns it is {first['window_ps'][0]:.0f} to "
      f"{first['window_ps'][1]:.0f} ps. Where the exponent is below 1 in the earlier window, "
      "the slope there is steeper than the slope at long times, and the short run gives too "
      "large a coefficient. The long run is one start; the short value is the mean of the "
      "short starts of the same box.")
    w("")
    w("| Concentration, mol/L | Species | Short runs, m2/s | Exponent | Long run, m2/s | "
      "Exponent | Short over long |")
    w("|---:|---|---:|---:|---|---:|---:|")
    for e in study:
        for sp, rec in e["species"].items():
            w(f"| {_f(e['molarity'], 2)} | {label.get(sp, sp)} | {_e(rec['short_d'])} | "
              f"{_f(rec['short_exponent'])} | {_e(rec['d'])} +/- {_e(rec['se'], 1)} | "
              f"{_f(rec['exponent'])} | {_f(rec['short_over_long'])} |")
    w("")
    w("| Concentration, mol/L | Salt coefficient with the measured thermodynamic factor, "
      "ratio to the measured: short runs | The same from the long run | Short over long | "
      "Conductivity of the long run, S/m | Measured, S/m |")
    w("|---:|---:|---:|---:|---|---:|")
    for e in study:
        k = e["conductivity"]
        w(f"| {_f(e['molarity'], 2)} | {_f(e['short_ratio_with_factor'])} | "
          f"{_f(e['ratio_with_factor'])} | {_f(e['overstated'])} | "
          f"{_f(k['sigma'], 3)} +/- {_f(k['se'], 3)} | {_f(e['measured_conductivity'], 3)} |")
    w("")
    return out


# --------------------------------------------------------------------------- #
# Stage 3: the simulated coefficient of the series in the cell model
# --------------------------------------------------------------------------- #
def ratio_function(measured, molarities: Sequence[float], ratios: Sequence[float]):
    """The measured function of the cell model multiplied by a ratio that depends on the
    concentration: linear between the concentrations given (in mol/L), and constant below the
    lowest and above the highest. Returns a function of the concentration (mol/m3) and the
    temperature, as the cell model reads one."""
    x = 1000.0 * np.asarray(molarities, float)
    y = np.asarray(ratios, float)
    if len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("the concentrations have to be at least two and rising")

    def scaled(c_e, T):
        import pybamm
        inside = pybamm.minimum(pybamm.maximum(c_e, float(x[0])), float(x[-1]))
        return measured(c_e, T) * pybamm.Interpolant(x, y, inside, interpolator="linear")
    return scaled


def cell_versions(study: dict, correction: Optional[Sequence[float]],
                  factors: Optional[dict] = None) -> Dict[str, List[float]]:
    """The ratios of the series to the measured coefficient, per concentration, in the versions
    that the cell is run with: as simulated, and with the two ends of the size correction.
    With `factors`, the result of :func:`length_factors`, the same three once more with the
    ratios corrected for the length of the run."""
    ratios = [p["ratio_with_factor"][0] for p in study["points"]]
    out = {"as simulated": ratios}
    if correction is not None:
        lo, hi = correction
        out["with the smaller correction for the box"] = [r * (1.0 + lo) for r in ratios]
        out["with the larger correction for the box"] = [r * (1.0 + hi) for r in ratios]
    if factors is not None:
        long = [p["corrected"] for p in factors["points"]]
        out["corrected for the length of the run"] = long
        if correction is not None:
            out["corrected for the length, with the smaller correction for the box"] = [
                r * (1.0 + lo) for r in long]
            out["corrected for the length, with the larger correction for the box"] = [
                r * (1.0 + hi) for r in long]
    return out


def stage_cell(results: str = RESULTS) -> Optional[dict]:
    """Run the reference cell with the simulated salt diffusion coefficient of the series in
    place of the measured one, at 298 K and the four rates of the stress stage. Writes
    ``cell.json``; returns None when the series has fewer than two concentrations."""
    import run_cellmodel as RC
    from cmfqsar import cellmodel as CM

    rows = read_results(list(RUNS), results)
    study = concentration_study(rows)
    if study is None or len(study["points"]) < 2:
        log("the cell stage needs the concentration series: nothing was run")
        return None
    series_names = {n for names in CONCENTRATION_RUNS.values() for n in names}
    lengths = long_study(rows)
    versions = cell_versions(study, size_range([r for r in rows if r["run"] not in series_names]),
                             length_factors(study, lengths) if lengths is not None else None)
    molarities = [p["molarity"] for p in study["points"]]
    four = CM.measured_electrolyte(coldest=None)
    p = CM.published_parameters(RC.BASE)
    out = {"temperature": ROOM, "model": RC.STRESS_MODEL, "share": CELL_SHARE,
           "molarities": molarities, "versions": versions, "starts": study["decided"],
           "rates": []}
    for rate in RC.STRESS_RATES:
        common = dict(cycles=1, base=RC.BASE, kind=RC.STRESS_MODEL, cycle=CM.discharge(rate),
                      temperature=ROOM, parameters=p, details=True)
        ref = CM.run_cell(four, **common)
        entry = {"rate": float(rate), "reference": float(ref.final),
                 "reference_concentration": list(ref.concentration), "runs": {}}
        for name, ratios in versions.items():
            changed = dict(four)
            changed[RC.DIFFUSIVITY] = ratio_function(four[RC.DIFFUSIVITY], molarities, ratios)
            run = CM.run_cell(changed, **common)
            usable = bool(run.ok and run.at_cutoff and np.isfinite(run.final))
            change = float(run.final / ref.final - 1.0) if usable else float("nan")
            entry["runs"][name] = {"capacity": float(run.final) if usable else float("nan"),
                                   "usable": usable, "ended": run.termination,
                                   "change": change,
                                   "concentration": list(run.concentration),
                                   "inside": bool(usable and abs(change) <= CELL_SHARE)}
            log(f"{rate:g}C, {name}: capacity {run.final:.3f} A h against {ref.final:.3f} "
                f"({100 * change:+.1f} %)")
        out["rates"].append(entry)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "cell.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def read_cell(results: str = RESULTS) -> Optional[dict]:
    path = os.path.join(results, "cell.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cell_lines(cell: dict) -> List[str]:
    """The section of the table on the cell run with the simulated coefficient."""
    out: List[str] = []
    w = out.append
    names = list(cell["versions"])
    w("## The simulated coefficient in the cell")
    w("")
    w("Gate (a) compares two coefficients. This section asks what the difference costs. The "
      "reference cell is discharged at 298 K with the measured electrolyte, and again with "
      "the salt diffusion coefficient replaced by the measured function times the ratio of "
      "the series: linear between the six concentrations, constant below the lowest and above "
      "the highest. The other three electrolyte functions stay the measured ones. The numbers "
      "are the change of the discharge capacity; the tolerance stage allows "
      f"{100 * cell['share']:.0f} %.")
    w("")
    w("| Rate | Capacity with the measured coefficient, A h | Lowest and highest "
      "concentration in the cell, mol/L | "
      + " | ".join(f"Change {n}" for n in names) + " |")
    w("|---:|---:|---|" + "---:|" * len(names))
    for e in cell["rates"]:
        lo, hi = e["reference_concentration"]
        cells = []
        for n in names:
            r = e["runs"][n]
            cells.append(f"{100 * r['change']:+.1f} %" if r["usable"]
                         else f"no usable discharge ({r['ended'] or 'failed'})")
        w(f"| {e['rate']:g}C | {_f(e['reference'], 3)} | {_f(lo / 1000.0)} to "
          f"{_f(hi / 1000.0)} | " + " | ".join(cells) + " |")
    w("")
    served = {n: [e["rate"] for e in cell["rates"] if e["runs"][n]["inside"]] for n in names}
    w("Inside the tolerance: " + "; ".join(
        f"{n}, at {', '.join(f'{x:g}C' for x in served[n])}" if served[n]
        else f"{n}, at no rate" for n in names) + ".")
    w("")
    return out


def length_factors(study: dict, lengths: Sequence[dict]) -> Optional[dict]:
    """The ratios of the series corrected for the length of the run: an estimate.

    `lengths` is the result of :func:`long_study`: at each concentration that has a long run,
    the factor by which the short runs overstate the salt diffusion coefficient. For the other
    concentrations the factor is read against the exponent of the cation in the short runs. It
    is linear in the exponent between the measured points, 1 at an exponent of 1 (a
    displacement that grows in proportion to the time already has its final slope), and held
    at the nearest measured value below the lowest measured exponent. Every ratio of `study`
    is divided by its factor. None when no long run has an exponent below 1.
    """
    anchors = sorted((float(e["species"]["cation"]["short_exponent"]), float(e["overstated"]),
                      e["key"]) for e in lengths)
    anchors = [a for a in anchors if a[0] < 1.0]
    if not anchors or len({a[0] for a in anchors}) < len(anchors):
        return None
    x = [a[0] for a in anchors] + [1.0]
    y = [a[1] for a in anchors] + [1.0]
    band = study["uncertainty"]
    measured = {a[2] for a in anchors}
    points = []
    for p in study["points"]:
        factor = float(np.interp(p["exponent"], x, y))
        ratio = float(p["ratio_with_factor"][0])
        points.append({"key": p["key"], "molarity": p["molarity"], "exponent": p["exponent"],
                       "factor": factor, "measured": p["key"] in measured, "ratio": ratio,
                       "corrected": ratio / factor,
                       "inside": bool(abs(ratio / factor - 1.0) <= band)})
    return {"anchors": [list(a) for a in anchors], "points": points, "uncertainty": band,
            "long_exponents": {e["key"]: float(e["species"]["cation"]["exponent"])
                               for e in lengths}}


def length_lines(factors: dict, correction: Optional[Sequence[float]] = None) -> List[str]:
    """The section of the table on the series corrected for the length of the run. With
    `correction`, the range of the correction for the size of the box, the corrected ratios are
    also given with both ends of that range, because the two corrections act in opposite
    directions."""
    out: List[str] = []
    w = out.append
    points = factors["points"]
    band = factors["uncertainty"]
    w("## The series corrected for the length of the run")
    w("")
    w("The long runs say by how much a run of 4 ns overstates the salt diffusion coefficient, "
      f"at {len(factors['anchors'])} of the concentrations. For the other concentrations the "
      "factor is estimated from the exponent of the cation in the short runs. It is linear in "
      "the exponent between the measured points, and it is 1 where the exponent is 1, because "
      "a displacement that grows in proportion to the time already has its final slope. Every "
      "ratio of the series is divided by its factor. This is an estimate: each long run is one "
      "start. A long run whose own exponent is still below 1 also overstates, and the "
      "corrected ratio at that concentration is then an upper limit.")
    w("")
    w("| Concentration, mol/L | Exponent of the cation in the short runs | Factor by which the "
      "short runs overstate | Where the factor comes from | Ratio to the measured, as simulated "
      f"| Ratio corrected for the length | Within {100 * band:.0f} % of the measured |")
    w("|---:|---:|---:|---|---:|---:|---|")
    for p in points:
        source = "the long run" if p["measured"] else "estimated from the exponent"
        if p["measured"]:
            source += f" (its own exponent: {_f(factors['long_exponents'][p['key']])})"
        w(f"| {_f(p['molarity'], 2)} | {_f(p['exponent'])} | {_f(p['factor'])} | {source} | "
          f"{_f(p['ratio'])} | {_f(p['corrected'])} | {'yes' if p['inside'] else 'no'} |")
    w("")
    inside = [p for p in points if p["inside"]]
    w(f"With this correction the ratio is within {100 * band:.0f} % of the measured value at "
      f"{len(inside)} of the {len(points)} concentrations"
      + (" (" + ", ".join(f"{_f(p['molarity'], 2)} mol/L" for p in inside) + ")"
         if inside and len(inside) < len(points) else "")
      + f", and it runs from {_f(min(p['corrected'] for p in points))} to "
      f"{_f(max(p['corrected'] for p in points))}. None of these ratios is corrected for the "
      "size of the box.")
    w("")
    if correction is not None:
        lo, hi = correction
        w("A longer run lowers a coefficient and a larger box raises it, so the two "
          "corrections act in opposite directions. With both of them:")
        w("")
        w("| Concentration, mol/L | Ratio corrected for the length | With the smaller "
          "correction for the box as well | With the larger correction for the box as well |")
        w("|---:|---:|---:|---:|")
        for p in points:
            w(f"| {_f(p['molarity'], 2)} | {_f(p['corrected'])} | "
              f"{_f(p['corrected'] * (1.0 + lo))} | {_f(p['corrected'] * (1.0 + hi))} |")
        w("")
        count = [sum(abs(p["corrected"] * (1.0 + c) - 1.0) <= band for p in points)
                 for c in (lo, hi)]
        w(f"With both corrections the ratio is within {100 * band:.0f} % of the measured value "
          f"at {count[0]} of the {len(points)} concentrations with the smaller correction for "
          f"the box, and at {count[1]} with the larger one.")
        w("")
    return out


def size_pair(rows: Sequence[dict]) -> Optional[dict]:
    """The two sizes of the box at the charge factor of the series.

    The smaller box is the run :data:`PAIR_SMALL` and the larger one the starts of the series at
    :data:`PAIR_KEY`. The correction of Yeh and Hummer is the same amount in m2/s for every
    species, so the mean difference between the two boxes over the species gives a viscosity,
    and with it what the larger box still lacks. `determined` is False when the larger box is
    not the faster one, and no correction is then computed. None when one of the two is not
    among `rows`.
    """
    from cmfqsar import transport as T
    by_run = {r["run"]: r for r in rows}
    large = [by_run[n] for n in CONCENTRATION_RUNS[PAIR_KEY] if n in by_run]
    if PAIR_SMALL not in by_run or not large:
        return None
    small = by_run[PAIR_SMALL]
    out = {"small": PAIR_SMALL, "large": [r["run"] for r in large],
           "small_side": float(small["side"]), "small_pairs": int(small["pairs"]),
           "large_side": float(np.mean([r["side"] for r in large])),
           "large_pairs": int(large[0]["pairs"]), "species": {}}
    for sp, rec in _species(small).items():
        pair = _pair([_species(r)[sp]["d"] for r in large],
                     float(_species(large[0])[sp]["se"]))
        out["species"][sp] = {"small": [float(rec["d"]), float(rec["se"])], "large": pair,
                              "ratio": pair[0] / float(rec["d"]),
                              "difference": pair[0] - float(rec["d"])}
    steps = np.array([v["difference"] for v in out["species"].values()])
    out["difference"] = float(steps.mean())
    out["difference_se"] = float(steps.std(ddof=1) / np.sqrt(len(steps))) if len(steps) > 1 \
        else float("nan")
    out["determined"] = bool(out["difference"] > 0)
    if not out["determined"]:
        return out
    temperature = float(small["temperature"])
    # the correction is proportional to one over the viscosity
    per_unit = (T.yeh_hummer_correction(out["small_side"], 1.0, temperature)
                - T.yeh_hummer_correction(out["large_side"], 1.0, temperature))
    out["viscosity"] = float(per_unit / out["difference"])
    out["remaining"] = T.yeh_hummer_correction(out["large_side"], out["viscosity"], temperature)
    dp, dm = out["species"]["cation"]["large"][0], out["species"]["anion"]["large"][0]
    out["share"] = {"cation": out["remaining"] / dp, "anion": out["remaining"] / dm,
                    "salt": T.salt_diffusivity(dp + out["remaining"], dm + out["remaining"])
                    / T.salt_diffusivity(dp, dm) - 1.0}
    return out


def pair_lines(pair: dict, correction: Optional[Sequence[float]] = None) -> List[str]:
    """The section of the table on the two sizes at the charge factor of the series."""
    out: List[str] = []
    w = out.append
    label = {"cation": "Cation", "anion": "Anion"}
    w("## Two sizes of the box at the charge factor of the series")
    w("")
    w("The size runs above were made at a factor of 0.8 on the charges of the ions, and the "
      f"series uses {SERIES_SETTINGS['ion_charge_scale']:g}. At that factor two sizes exist at "
      f"the concentration of the reference cell: one run of a box with {pair['small_pairs']} "
      f"ion pairs (`{pair['small']}`, a side of {_f(pair['small_side'], 1)} A) and "
      f"{len(pair['large'])} start{'s' if len(pair['large']) != 1 else ''} of the box of the "
      f"series with {pair['large_pairs']} ion "
      f"pairs (a side of {_f(pair['large_side'], 1)} A). The correction of Yeh and Hummer "
      "says that the larger box is faster by the same amount, in m2/s, for every species.")
    w("")
    w("| Species | Smaller box, m2/s | Larger box, m2/s | Larger over smaller | Difference, "
      "m2/s |")
    w("|---|---|---|---:|---:|")
    for sp, v in pair["species"].items():
        large = _e(v["large"][0]) + (f" +/- {_e(v['large'][1], 1)}"
                                     if np.isfinite(v["large"][1]) else "")
        w(f"| {label.get(sp, sp)} | {_e(v['small'][0])} +/- {_e(v['small'][1], 1)} | {large} | "
          f"{_f(v['ratio'])} | {_e(v['difference'])} |")
    w("")
    if not pair["determined"]:
        w("The larger box is not the faster one on average, so the two boxes give no "
          "correction.")
        w("")
        return out
    share = pair["share"]
    text = (f"The mean difference is {_e(pair['difference'])} +/- "
            f"{_e(pair['difference_se'], 1)} m2/s; the error is from the scatter over the "
            "species. Read as the correction of Yeh and Hummer, this difference is a viscosity "
            f"of {_f(1000.0 * pair['viscosity'], 1)} mPa s. With that viscosity the box of the "
            f"series is still too slow by {_e(pair['remaining'])} m2/s, which is "
            f"{100 * share['cation']:.0f} % of the coefficient of the cation, "
            f"{100 * share['anion']:.0f} % of that of the anion and "
            f"{100 * share['salt']:.0f} % of the salt diffusion coefficient.")
    if correction is not None:
        lo, hi = correction
        text += (f" The size runs at a factor of 0.8 gave a range from +{100 * lo:.0f} % to "
                 f"+{100 * hi:.0f} % for the same box, and this estimate is "
                 + ("inside" if lo <= share["salt"] <= hi else "outside") + " it.")
    text += (" The smaller box was run once, so this is an estimate. The range from the size "
             "runs stays the one that the other sections use.")
    w(text)
    w("")
    return out


def _ranks(x: Sequence[float]) -> np.ndarray:
    """Ranks from 0, ties given their mean rank."""
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x), dtype=float)
    for v in np.unique(x):
        tie = x == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """The rank correlation of Spearman."""
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def start_scatter(series: Sequence[dict]) -> Optional[float]:
    """The standard deviation of the logarithm of the conductivity of one start of 4 ns, from the
    pairs of starts of the concentration series: the mean of (ln a - ln b) squared, halved."""
    by_run = {r["run"]: r for r in series}
    d = [np.log(by_run[a]["whole"]["conductivity"]["sigma"]
                / by_run[b]["whole"]["conductivity"]["sigma"])
         for a, b in CONCENTRATION_RUNS.values() if a in by_run and b in by_run]
    if not d:
        return None
    return float(np.sqrt(np.mean(np.square(d)) / 2.0))


def formulation_study(rows: Sequence[dict], series: Sequence[dict], draws: int = 2000,
                      seed: int = 0, runs: Optional[Dict[str, Sequence[str]]] = None
                      ) -> Optional[dict]:
    """The transfer test of the charge factor: the simulated conductivity of one formulation of
    each solvent composition against CALiSol-23.

    *Rank:* Spearman's correlation, with a 95 % interval from resampling the formulations, and
    the correlation that a simulation without any bias would reach with the scatter of one start
    (the median over `draws` sets of measured values, each multiplied by a random factor with the
    scatter of :func:`start_scatter`). *Level:* the geometric mean of the ratios to the measured
    conductivity against the range of ratios that the series found over the concentration of the
    reference electrolyte. The rank part holds if the lower end of the interval is above zero;
    the level part holds if the geometric mean lies inside that range. None with fewer than three
    formulations. `runs` gives the runs of each formulation by its CALiSol id; the default is
    :data:`FORMULATION_RUNS`.
    """
    runs = FORMULATION_RUNS if runs is None else runs
    measured = {r["id"]: r for r in calisol_rows()}
    by_run = {r["run"]: r for r in list(rows) + list(series)}
    points = []
    for fid, names in runs.items():
        got = [by_run[n] for n in names if n in by_run]
        if not got or fid not in measured:
            continue
        sig = [g["whole"]["conductivity"]["sigma"] for g in got]
        points.append({
            "id": fid, "runs": [g["run"] for g in got], "solvents": measured[fid]["solvents"],
            "molality": measured[fid]["molality"], "pairs": int(got[0]["pairs"]),
            "measured": float(measured[fid]["conductivity"]),
            "simulated": float(np.mean(sig)),
            "se": (float(np.std(sig, ddof=1) / np.sqrt(len(sig))) if len(sig) > 1
                   else float(got[0]["whole"]["conductivity"]["se"])),
            "nernst_einstein": float(np.mean([g["whole"]["nernst_einstein"] for g in got])),
            "exponent": float(np.mean([g["whole"]["cation"]["exponent"] for g in got])),
            "reference": fid == ELECTROLYTE_ID})
    if len(points) < 3:
        return None
    for p in points:
        p["ratio"] = p["simulated"] / p["measured"]
        p["ionicity"] = p["simulated"] / p["nernst_einstein"]
    meas = np.array([p["measured"] for p in points])
    sim = np.array([p["simulated"] for p in points])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(draws):
        i = rng.integers(0, len(points), len(points))
        r = spearman(meas[i], sim[i])
        if np.isfinite(r):
            boot.append(r)
    scatter = start_scatter(series)
    ceiling = None
    if scatter is not None:
        ceiling = float(np.median([spearman(meas, meas * np.exp(rng.normal(0.0, scatter,
                                                                           len(meas))))
                                   for _ in range(draws)]))
    conc = [r for r in series if r["run"] in {n for v in CONCENTRATION_RUNS.values() for n in v}]
    series_ratios = []
    for key, names in CONCENTRATION_RUNS.items():
        got = [r for r in conc if r["run"] in names]
        if got:
            series_ratios.append(float(np.mean([r["whole"]["conductivity"]["sigma"]
                                                for r in got]) / got[0]["measured"]["conductivity"]))
    others = [p for p in points if not p["reference"]]
    gmean = float(np.exp(np.mean([np.log(p["ratio"]) for p in others]))) if others else float("nan")
    band = [min(series_ratios), max(series_ratios)] if series_ratios else None
    by_solvent = {}
    for name in sorted({k for p in points for k in p["solvents"]}):
        part = [p for p in points if p["solvents"].get(name, 0.0) >= SOLVENT_SHARE]
        if part:
            by_solvent[name] = {"n": len(part), "ratio": float(np.exp(np.mean(
                [np.log(p["ratio"]) for p in part])))}
    rho = spearman(meas, sim)
    lo, hi = (np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan")))
    return {"points": points, "spearman": rho, "interval": [float(lo), float(hi)],
            "ceiling": ceiling, "scatter": scatter, "geometric_mean": gmean,
            "series_band": band, "by_solvent": by_solvent,
            "complete": len(points) == len(runs),
            "rank_holds": bool(np.isfinite(lo) and lo > 0),
            "level_holds": bool(band is not None and band[0] <= gmean <= band[1]),
            "short": [p["id"] for p in points if p["exponent"] < 0.8]}

#: the runs of all 62 formulations of CALiSol-23 by the CALiSol id: the 22 boxes of the transfer
#: test and the 40 other formulations
ALL_FORMULATION_RUNS: Dict[str, tuple] = dict(FORMULATION_RUNS)
for _name in REST_NAMES:
    ALL_FORMULATION_RUNS[RUNS[_name]["formulation"]] = (_name,)
#: the molality limits (mol/kg) between the three bands of concentration that are read apart
MOLALITY_BANDS = (0.7, 1.4)
#: the factors by which the boxes that are too short are changed to test the level
SHORT_FACTORS = (0.6, 1.5)


def all_formulations_study(rows: Sequence[dict], series: Sequence[dict], draws: int = 2000,
                           seed: int = 0) -> Optional[dict]:
    """The transfer test over all 62 formulations of CALiSol-23, each at its own molality.

    The reading rule is that of :func:`formulation_study`: the rank part holds if the lower end
    of the 95 % interval of the rank correlation is above zero, and the level part holds if the
    geometric mean of the ratios (all but the reference electrolyte, which is the series) lies
    inside the range of ratios of the series. Four more readings are added. The three bands of
    molality (:data:`MOLALITY_BANDS`) are read apart, because the rank over all 62 mixes the
    effect of the concentration with that of the solvent. The rank correlation of the ratio with
    the molality shows whether the factor on the charges depends on the concentration. For the
    compositions that have two or more molalities, the study counts how often the simulation
    moves the conductivity the same way as the measurement, between the lowest and the highest
    molality. And the boxes whose exponent of the cation is below 0.8 are changed by the
    factors of :data:`SHORT_FACTORS`, to show how much the level depends on them.
    """
    study = formulation_study(rows, series, draws=draws, seed=seed, runs=ALL_FORMULATION_RUNS)
    if study is None:
        return None
    pts = study["points"]
    log_ratio = np.log([p["ratio"] for p in pts])
    mol = np.array([p["molality"] for p in pts])
    edges = (0.0,) + tuple(MOLALITY_BANDS) + (float("inf"),)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        part = [p for p in pts if lo <= p["molality"] < hi]
        if not part:
            continue
        bands.append({
            "low": lo, "high": hi, "n": len(part),
            "spearman": spearman([p["measured"] for p in part], [p["simulated"] for p in part]),
            "ratio": float(np.exp(np.mean([np.log(p["ratio"]) for p in part]))),
            "short": sum(p["exponent"] < 0.8 for p in part)})
    study["bands"] = bands
    study["ratio_with_molality"] = spearman(mol, np.exp(log_ratio))
    groups: Dict[tuple, list] = {}
    for p in pts:
        groups.setdefault(tuple(sorted(p["solvents"].items())), []).append(p)
    moved = [0, 0]
    for part in groups.values():
        if len(part) < 2:
            continue
        part = sorted(part, key=lambda p: p["molality"])
        moved[1] += 1
        moved[0] += int(np.log(part[-1]["simulated"] / part[0]["simulated"])
                        * np.log(part[-1]["measured"] / part[0]["measured"]) > 0)
    study["same_direction"] = moved
    others = [p for p in pts if not p["reference"]]
    sensitivity = {}
    for factor in SHORT_FACTORS:
        logs = [np.log(p["ratio"] * (factor if p["exponent"] < 0.8 else 1.0)) for p in others]
        sensitivity[factor] = float(np.exp(np.mean(logs)))
    study["short_sensitivity"] = sensitivity
    return study


def formulation_lines(study: dict) -> List[str]:
    """The section of the table on the transfer test to other solvents."""
    out: List[str] = []
    w = out.append
    pts = sorted(study["points"], key=lambda p: p["measured"])
    w("## Other solvents: does the charge factor transfer?")
    w("")
    w("Rule 2 asks that the simulation reproduces the measured conductivities of CALiSol-23 in "
      "rank and in level before a model is fitted on simulated labels. The factor of 0.7 on the "
      "charges of the ions was chosen on one electrolyte, EC:EMC 3:7. Here one formulation of "
      "each of the 22 solvent compositions is simulated with the same settings: the one whose "
      f"molality is nearest to {FORMULATION_TARGET:g} mol/kg, so that the solvent changes and "
      "the concentration stays near that of the reference cell. Each box is one start of 4 ns, "
      "except EC:EMC 3:7, which is the mean of the two starts of the series. The measured value "
      "is the fit of CALiSol-23 at 298 K for that formulation. The reading rule was fixed before "
      "the boxes were run: the rank part holds if the lower end of the 95 % interval of the rank "
      "correlation is above zero, and the level part holds if the geometric mean of the ratios "
      "of the other compositions lies inside the range of ratios that the series found over the "
      "concentration of EC:EMC 3:7.")
    w("")
    w("| Formulation | Molality, mol/kg | Ion pairs | Measured conductivity, S/m | Simulated, "
      "S/m | Ratio to the measured | Exponent of the cation | Ratio of the conductivity to "
      "the Nernst-Einstein value |")
    w("|---|---:|---:|---:|---|---:|---:|---:|")
    for p in pts:
        name = p["id"] + (" (the reference electrolyte, two starts)" if p["reference"] else "")
        w(f"| {name} | {_f(p['molality'], 3)} | {p['pairs']} | {_f(p['measured'], 3)} | "
          f"{_f(p['simulated'], 3)} +/- {_f(p['se'], 3)} | {_f(p['ratio'])} | "
          f"{_f(p['exponent'])} | {_f(p['ionicity'])} |")
    w("")
    n = len(study["points"])
    lo, hi = study["interval"]
    text = (f"**Rank.** Over {n} formulations the rank correlation of Spearman between the "
            f"simulated and the measured conductivity is {_f(study['spearman'])} (95 % interval "
            f"{_f(lo)} to {_f(hi)}, from resampling the formulations).")
    if study["ceiling"] is not None:
        text += (f" One start of 4 ns scatters by {100 * study['scatter']:.0f} % in the "
                 "logarithm of the conductivity, from the pairs of starts of the series. With "
                 "that scatter alone, a simulation without any bias would reach a rank "
                 f"correlation of about {_f(study['ceiling'])}.")
    text += " The rank part " + ("holds." if study["rank_holds"] else "does not hold.")
    w(text)
    w("")
    band = study["series_band"]
    text = (f"**Level.** The geometric mean of the ratios of the other compositions is "
            f"{_f(study['geometric_mean'])}.")
    if band is not None:
        text += (f" The series of EC:EMC 3:7 gave ratios from {_f(band[0])} to {_f(band[1])} "
                 "over the concentration. The level part "
                 + ("holds." if study["level_holds"] else "does not hold."))
    w(text)
    w("")
    w("**By solvent.** The geometric mean of the ratio over the formulations that hold a "
      f"solvent at {100 * SOLVENT_SHARE:.0f} % of the weight or more:")
    w("")
    w("| Solvent | Formulations | Geometric mean of the ratio to the measured |")
    w("|---|---:|---:|")
    for name, v in study["by_solvent"].items():
        w(f"| {name} | {v['n']} | {_f(v['ratio'])} |")
    w("")
    if study["short"]:
        w("The exponent of the cation is below 0.8 for " + ", ".join(study["short"])
          + ". The concentration series showed that a run of 4 ns then overstates the "
          "coefficients, so these boxes need runs of 20 ns before their values are final.")
        w("")
    if not study["complete"]:
        w(f"Not every composition was run yet: {n} of {len(FORMULATION_RUNS)}. The reading is "
          "therefore not final.")
        w("")
    return out


def all_formulation_lines(study: dict) -> List[str]:
    """The section of the table on all 62 formulations of CALiSol-23."""
    out: List[str] = []
    w = out.append
    pts = sorted(study["points"], key=lambda p: (sorted(p["solvents"].items()), p["molality"]))
    n = len(pts)
    w("## All 62 formulations of CALiSol-23")
    w("")
    w("The transfer test above takes one formulation of each solvent composition. This section "
      "adds the other formulations: every one of the 62 is simulated at its own molality, with "
      "the same settings (charges of the ions multiplied by "
      f"{SERIES_SETTINGS['ion_charge_scale']:g}, {SERIES_SETTINGS['n_solvent']:,} solvent "
      "molecules, one start of 4 ns), except EC:EMC 3:7 at 1.11 mol/kg, which is the mean of "
      "the two starts of the series. The reading rule of the transfer test is kept: the rank "
      "part holds if the lower end of the 95 % interval of the rank correlation is above zero, "
      "and the level part holds if the geometric mean of the ratios of all formulations but the "
      "reference electrolyte lies inside the range of ratios of the series. The four readings "
      "after the rank and the level (the bands of molality, the direction within a "
      "composition, the ratio against the molality and the boxes that are too short) were "
      "chosen after the first numbers had been seen, so they are descriptions and not tests.")
    w("")
    w("| Formulation | Molality, mol/kg | Measured conductivity, S/m | Simulated, S/m | "
      "Ratio to the measured | Exponent of the cation |")
    w("|---|---:|---:|---|---:|---:|")
    for p in pts:
        name = p["id"] + (" (the reference electrolyte, two starts)" if p["reference"] else "")
        w(f"| {name} | {_f(p['molality'], 3)} | {_f(p['measured'], 3)} | "
          f"{_f(p['simulated'], 3)} +/- {_f(p['se'], 3)} | {_f(p['ratio'])} | "
          f"{_f(p['exponent'])} |")
    w("")
    lo, hi = study["interval"]
    text = (f"**Rank.** Over {n} formulations the rank correlation of Spearman between the "
            f"simulated and the measured conductivity is {_f(study['spearman'])} (95 % interval "
            f"{_f(lo)} to {_f(hi)}, from resampling the formulations).")
    if study["ceiling"] is not None:
        text += (f" With the scatter of one start ({100 * study['scatter']:.0f} % in the "
                 "logarithm of the conductivity) alone, a simulation without any bias would "
                 f"reach about {_f(study['ceiling'])}.")
    text += " The rank part " + ("holds." if study["rank_holds"] else "does not hold.")
    w(text)
    w("")
    band = study["series_band"]
    others = [p for p in pts if not p["reference"]]
    ratios = [p["ratio"] for p in others]
    text = (f"**Level.** The geometric mean of the ratios of the {len(others)} formulations "
            f"other than the reference electrolyte is {_f(study['geometric_mean'])}. The ratios "
            f"of single formulations run from {_f(min(ratios))} to {_f(max(ratios))}.")
    if band is not None:
        text += (f" The series gave ratios from {_f(band[0])} to {_f(band[1])}. The level part "
                 + ("holds." if study["level_holds"] else "does not hold."))
    w(text)
    w("")
    w("**Bands of molality.** The rank over all formulations mixes the effect of the "
      "concentration with that of the solvent. Here each band is read apart.")
    w("")
    w("| Molality, mol/kg | Formulations | Rank correlation of Spearman | Geometric mean of "
      "the ratio | Boxes with an exponent below 0.8 |")
    w("|---|---:|---:|---:|---:|")
    for b in study["bands"]:
        limit = (f"below {b['high']:g}" if b["low"] == 0.0 else
                 f"{b['low']:g} or more" if b["high"] == float("inf") else
                 f"{b['low']:g} to below {b['high']:g}")
        w(f"| {limit} | {b['n']} | {_f(b['spearman'])} | {_f(b['ratio'])} | {b['short']} |")
    w("")
    moved = study["same_direction"]
    w(f"**Within one composition.** For the {moved[1]} compositions that have two or more "
      "molalities, the simulation moves the conductivity in the same direction as the "
      f"measurement, from the lowest to the highest molality, in {moved[0]} of {moved[1]}. "
      "The rank correlation of the ratio to the measured conductivity with the molality is "
      f"{_f(study['ratio_with_molality'])}: a value near zero means that the factor on the "
      "charges does not depend on the concentration.")
    w("")
    w("**By solvent.** The geometric mean of the ratio over the formulations that hold a "
      f"solvent at {100 * SOLVENT_SHARE:.0f} % of the weight or more:")
    w("")
    w("| Solvent | Formulations | Geometric mean of the ratio to the measured |")
    w("|---|---:|---:|")
    for name, v in study["by_solvent"].items():
        w(f"| {name} | {v['n']} | {_f(v['ratio'])} |")
    w("")
    if study["short"]:
        low_f, high_f = SHORT_FACTORS
        sens = study["short_sensitivity"]
        w(f"The exponent of the cation is below 0.8 for {len(study['short'])} of the {n} "
          "formulations. The concentration series showed that a run of 4 ns then overstates "
          "the coefficients, so these boxes need runs of 20 ns before their values are final. "
          f"If all their conductivities changed by a factor of {low_f:g} or {high_f:g}, the "
          f"geometric mean of the ratios would be {_f(sens[low_f])} or {_f(sens[high_f])}.")
        w("")
    if not study["complete"]:
        w(f"Not every formulation was run yet: {n} of {len(ALL_FORMULATION_RUNS)}. The "
          "reading is therefore not final.")
        w("")
    return out


def size_range(rows: Sequence[dict]) -> Optional[List[float]]:
    """The share that the correction for the size of the box adds to the salt diffusion
    coefficient of the largest box, as the range between the two size fits: [low, high]. None
    when the size runs are not among `rows` or a fit determines nothing."""
    whole = size_study(rows)
    if whole is None or not whole.get("determined") or "ratio_with_factor" not in whole:
        return None
    rest = size_study(rows, exclude=(whole["runs"][0],))
    if rest is None or not rest.get("determined") or "ratio_with_factor" not in rest:
        return None
    largest = whole["largest"]["ratio_with_factor"]
    lo, hi = sorted([whole["ratio_with_factor"] / largest - 1.0,
                     rest["ratio_with_factor"] / largest - 1.0])
    return [float(lo), float(hi)]


def concentration_lines(study: dict, limits: Sequence[Dict[str, float]],
                        correction: Optional[Sequence[float]] = None) -> List[str]:
    """The section of the table on the concentration series."""
    out: List[str] = []
    w = out.append
    points = study["points"]
    band = study["uncertainty"]

    def pm(pair, fmt):
        return fmt(pair[0]) + (f" +/- {fmt(pair[1])}" if np.isfinite(pair[1]) else "")
    w("## The range of concentrations")
    w("")
    w("The measured study covers salt concentrations from 0.1 to 3.0 mol/L. This series "
      "simulates the same six concentrations at 298 K with the charges of the ions multiplied "
      f"by {SERIES_SETTINGS['ion_charge_scale']:g}, in a box of "
      f"{SERIES_SETTINGS['n_solvent']:,} solvent molecules. Every concentration is run from "
      "two starts: the box is packed again and given other velocities. Where both starts are "
      "there, a number is their mean and the error after +/- is the standard error of that "
      "mean, which for two starts is half their difference. Where one start is there, the "
      "error is the block error of that run. The measured functions are read at the "
      "concentration that the box really has.")
    w("")
    w("**Self-diffusion and the salt diffusion coefficient.**")
    w("")
    w("| Concentration, mol/L | Ion pairs | Starts | Cation, m2/s | Anion, m2/s | Exponent of "
      "the cation | Measured thermodynamic factor | Salt coefficient with that factor, m2/s | "
      "Measured salt coefficient, m2/s | Ratio to the measured | Within "
      f"{100 * band:.0f} % of the measured |")
    w("|---:|---:|---:|---|---|---:|---:|---|---:|---|---|")
    for p in points:
        m = p["measured"]
        verdict = "yes" if p["inside"] else (
            "no, but its error reaches it" if p["inside_with_error"] else "no")
        w(f"| {_f(p['molarity'], 2)} | {p['pairs']} | {p['starts']} | {pm(p['cation'], _e)} | "
          f"{pm(p['anion'], _e)} | {_f(p['exponent'])} | {_f(m['thermodynamic_factor'])} | "
          f"{pm(p['salt_with_factor'], _e)} | {_e(m['diffusivity'])} | "
          f"{pm(p['ratio_with_factor'], _f)} | {verdict} |")
    w("")
    w("**What the study says about its own uncertainty.** Landesfeind and Gasteiger give no "
      "single number for it. Their error bars are the scatter of two cells with two current "
      "pulses each, and they write that these do not include the systematic error of building "
      "a cell. They report that a repeat at the end of a temperature cycle was up to 10 % lower "
      "and call that inside the uncertainty of the measurement. Their fitted function of the "
      "diffusion coefficient has a coefficient of determination of 0.978 for this electrolyte. "
      f"Gate (a) is therefore read with {100 * band:.0f} %. The comparison also multiplies by "
      "their thermodynamic factor, whose fitted function has a coefficient of determination "
      "of 0.943 and, in their words, the highest scatter of the three electrolytes of the "
      "study. The last column therefore includes the error of two measured functions.")
    w("")
    ratios = [p["ratio_with_factor"][0] for p in points]
    inside = [p for p in points if p["inside"]]
    w(f"**Gate (a).** The ratio is within {100 * band:.0f} % of the measured value at "
      f"{len(inside)} of the {len(points)} concentrations"
      + (" (" + ", ".join(f"{_f(p['molarity'], 2)} mol/L" for p in inside) + ")"
         if inside and len(inside) < len(points) else "")
      + f". Over the series it runs from {_f(min(ratios))} to {_f(max(ratios))}. "
      + ("The gate is met at every concentration that was run."
         if study["passes"] else
         f"The gate is not met. With a band twice as wide, {200 * band:.0f} %, the ratio is "
         f"inside at {sum(abs(r - 1.0) <= 2.0 * band for r in ratios)} of the "
         f"{len(points)} concentrations.")
      + ("" if study["decided"] else
         " The series is not complete: " + (
             "not every concentration was run" if not study["complete"]
             else "not every concentration has its second start") + "."))
    w("")
    if correction is not None:
        lo, hi = correction
        w("None of these coefficients is corrected for the size of the box. The correction of "
          "Yeh and Hummer, as a share of a coefficient, is the hydrodynamic radius of the "
          "species divided by the side of the box, times a constant. It does not contain the "
          "viscosity, so the share found at one concentration is a fair estimate at the "
          f"others. The size runs put it between +{100 * lo:.0f} % and +{100 * hi:.0f} % for "
          "this size of box, at a factor of 0.8 on the charges. With that range the ratios "
          "become:")
        w("")
        w("| Concentration, mol/L | Ratio as simulated | With the smaller correction | With "
          "the larger correction | Rates at which the cell tolerates the three |")
        w("|---:|---:|---:|---:|---|")
        for p in points:
            r = p["ratio_with_factor"][0]
            three = [r, r * (1.0 + lo), r * (1.0 + hi)]
            served = [served_rates(x, limits) for x in three]
            text = "; ".join((", ".join(f"{x:g}C" for x in sv) if sv else "none")
                             for sv in served) if limits else "not run"
            w(f"| {_f(p['molarity'], 2)} | {_f(three[0])} | {_f(three[1])} | {_f(three[2])} | "
              f"{text} |")
        w("")
        if limits:
            w("The tolerated rates are those of the tolerance stage, which multiplied the "
              "whole measured function by one number. They are a guide here, because the "
              "ratio of this series changes with the concentration.")
            w("")
    w("**The conductivity, S/m.** The conductivity is the stricter test, because its measured "
      "values carry an error of 3 % and no thermodynamic factor enters. The first value is "
      "from the displacement of the total charge.")
    w("")
    w("| Concentration, mol/L | From the total charge | Measured | Ratio to the measured | "
      "Nernst-Einstein value | Ratio of the first to the Nernst-Einstein value | Share of the "
      "current on the cation if ions moved alone | Measured transference number |")
    w("|---:|---|---:|---:|---:|---:|---:|---:|")
    for p in points:
        m = p["measured"]
        w(f"| {_f(p['molarity'], 2)} | {pm(p['conductivity'], lambda v: _f(v, 3))} | "
          f"{_f(m['conductivity'], 3)} | {_f(p['ratio_conductivity'])} | "
          f"{_f(p['nernst_einstein'][0], 3)} | {_f(p['ionicity'][0])} | "
          f"{_f(p['transport_number'][0], 3)} | {_f(m['transference'], 3)} |")
    w("")
    big = study["largest_conductivity"]
    w(f"The simulated conductivity is largest at {_f(big['simulated'], 2)} mol/L among the "
      f"concentrations that were run, and the measured one at {_f(big['measured'], 2)} mol/L.")
    w("")
    second = [p for p in points if p.get("second")]
    if second:
        w("**A second measurement of the same electrolyte.** Nyman, Behm and Lindbergh (2008) "
          "measured this electrolyte at 298 K between 0.2 and 2 mol/L, and PyBaMM ships their "
          "fitted functions in another parameter set. The table gives their diffusion "
          "coefficient beside that of Landesfeind and Gasteiger. The two functions as shipped "
          "differ by far more than 10 % at most concentrations. Landesfeind and Gasteiger "
          "write that the values of Nyman are slightly higher than theirs, which the shipped "
          "function is not, so the two may not define the coefficient in the same way. This "
          "program has not read the paper of Nyman, and the difference is recorded here and "
          "not used as an uncertainty.")
        w("")
        w("| Concentration, mol/L | Landesfeind and Gasteiger, m2/s | Nyman and others, m2/s | "
          "Ratio of the second to the first | Inside the range of the second study | "
          "Simulated, ratio to the first |")
        w("|---:|---:|---:|---:|---|---|")
        for p in second:
            m, n2 = p["measured"], p["second"]
            w(f"| {_f(p['molarity'], 2)} | {_e(m['diffusivity'])} | {_e(n2['diffusivity'])} | "
              f"{_f(n2['diffusivity'] / m['diffusivity'])} | "
              f"{'yes' if n2['inside'] else 'no'} | {pm(p['ratio_with_factor'], _f)} |")
        w("")
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


def report_lines(rows: Sequence[dict], limits: Sequence[Dict[str, float]],
                 cell: Optional[dict] = None) -> List[str]:
    """The table of this benchmark as lines of Markdown."""
    out: List[str] = []
    w = out.append
    series_names = {n for names in CONCENTRATION_RUNS.values() for n in names}
    long_names = set(LONG_RUNS.values())
    form_names = set(FORMULATION_NAMES) | set(REST_NAMES)
    series = [r for r in rows if r["run"] in series_names]
    longs = [r for r in rows if r["run"] in long_names]
    forms = [r for r in rows if r["run"] in set(FORMULATION_NAMES)]
    other_forms = [r for r in rows if r["run"] in set(REST_NAMES)]
    rows = [r for r in rows if r["run"] not in series_names | long_names | form_names]
    if not rows:
        w("# Item BT9: the salt diffusion coefficient of an electrolyte from simulation")
        w("")
        if forms:
            others = formulation_study(forms, series)
            if others is not None:
                out.extend(formulation_lines(others))
        if other_forms:
            every = all_formulations_study(forms + other_forms, series)
            if every is not None:
                out.extend(all_formulation_lines(every))
        if series:
            out.extend(concentration_lines(concentration_study(series), limits))
        lengths = long_study(series + longs)
        if lengths is not None:
            out.extend(long_lines(lengths))
        return out
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
    if series:
        w("## The boxes of the concentration series")
        w("")
        w("| Run | What it is | Atoms | Ion pairs | Side of the box, A | Density, g/cm3 | "
          "Concentration, mol/L | Length, ns | Speed, ns/day |")
        w("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(series + longs, key=lambda r: (r["molarity"], r["run"])):
            w(f"| {r['run']} | {r['label']} | {r['atoms']} | {r['pairs']} | {_f(r['side'])} | "
              f"{_f(r['density'], 3)} | {_f(r['molarity'], 3)} | "
              f"{_f(r['whole']['length_ps'] / 1000.0, 1)} | {_f(r['ns_per_day'], 0)} |")
        w("")
        out.extend(concentration_lines(concentration_study(series), limits, size_range(rows)))
        pair = size_pair(list(rows) + series)
        if pair is not None:
            out.extend(pair_lines(pair, size_range(rows)))
    lengths = long_study(series + longs)
    if lengths is not None:
        out.extend(long_lines(lengths))
        factors = length_factors(concentration_study(series), lengths)
        if factors is not None:
            out.extend(length_lines(factors, size_range(rows)))
    if cell is not None:
        out.extend(cell_lines(cell))
    if forms and rows:
        others = formulation_study(forms, series)
        if others is not None:
            out.extend(formulation_lines(others))
    if other_forms and rows:
        every = all_formulations_study(forms + other_forms, series)
        if every is not None:
            out.extend(all_formulation_lines(every))
    return out


def report(results: str = RESULTS, names: Optional[Sequence[str]] = None) -> str:
    """Write the table from the stored results. Returns the path of the table."""
    rows = read_results(list(RUNS) if names is None else names, results)
    if not rows:
        raise SystemExit(f"{results} holds no result: run this benchmark with --stage analyse")
    path = os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(report_lines(rows, tolerance_limits(), read_cell(results))) + "\n")
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
            if args.runs == ",".join(RUNS):
                # the default is every run; one that was not simulated yet is named and left out
                there = [n for n in names
                         if os.path.isfile(os.path.join(run_dir(n, args.cache), "run.json"))]
                for n in names:
                    if there and n not in there:
                        log(f"{n}: not simulated yet, left out")
                names = there or names
            stage_analyse(names, args.results, args.cache)
        if "cell" in stages:
            stage_cell(args.results)
        if "analyse" not in stages and "cell" not in stages:
            return 0
    log(f"the table: {report(args.results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
