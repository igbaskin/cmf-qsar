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
}
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
