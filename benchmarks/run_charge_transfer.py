#!/usr/bin/env python
"""
Measured activation energies of charge transfer at an electrode: what the collected values say
(plan item BT12, parts (a) and (b) of its gate).

The exchange-current density of an electrode is not predicted by this program. Its activation
energy is the slope of an Arrhenius plot of the charge-transfer resistance, which is read from
impedance spectra, and it does not need the unknown active area. ``examples/charge_transfer``
holds the values collected from the literature, one row per value, with the paper and the place
in it. A row is *primary* when the number was read in the paper that measured it, and
*secondary* when it was read in another paper that cites it.

No model is fitted here. The script answers the four questions that come before a model:

1. *How well do laboratories agree on one system?*  The reference system is a graphite
   electrode with its surface film, after formation, in a carbonate electrolyte that contains
   ethylene carbonate, with no additive. The spread of the values of different studies for it
   is the repeatability, and no model can have a smaller error.
2. *How much does the electrolyte change the value inside one study?*  For every study that
   measured one electrode in several electrolytes, the range of its values. A study is read on
   its own because laboratories differ in the cell and in the fit of the spectrum.
3. *Does a cold cell need these values?*  The mean for the negative and for the positive
   electrode is compared with the activation energies of the published cell set (35 and 17.8
   kJ/mol), and the loss of capacity that the difference causes is read from the stage
   ``--stage exchange`` of ``run_cellmodel.py``.
4. *Are there enough solvents to test a model of the solvent?*  A *series* is a set of rows of
   one study that share the electrode and the film and differ in the solvent. The script lists
   the series, counts the solvents, and compares one pair of solvents between the series that
   measured both.

The reference of the item is then run (``--stage binding`` and ``--stage regression``). It is
the one-number version of the hypothesis that the removal of the solvent shell sets the barrier:
a linear regression of the activation energy on the binding energy of Li+ with one molecule of
the solvent. The binding energy is computed here for every single solvent of the series, with
ORCA, from the geometries that the builder of plan item BT4 gives. The regression has one slope
for all series and one intercept per series, and it is tested with one solvent held out at a
time, against the mean of the series.

    python benchmarks/run_charge_transfer.py                      # the summary and the table
    python benchmarks/run_charge_transfer.py --stage binding      # the binding energies (ORCA)
    python benchmarks/run_charge_transfer.py --stage regression   # the regression, seconds
    python benchmarks/run_charge_transfer.py --report             # the table from stored results
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "examples", "charge_transfer")
RESULTS = os.path.join(HERE, "results", "charge_transfer")
CELLMODEL = os.path.join(HERE, "results", "cellmodel")
WORK = os.path.join(HERE, "cache", "charge_transfer")

#: kJ/mol in one electron volt per particle
KJ_PER_EV = 96.485
#: kJ/mol in one kilocalorie per mole
KJ_PER_KCAL = 4.184
#: the activation energies of the published cell set (Chen 2020 in PyBaMM), kJ/mol
PUBLISHED = {"negative": 35.0, "positive": 17.8}
#: which classes of electrode stand for the two electrodes of a lithium-ion cell
NEGATIVE = ("graphite",)
POSITIVE = ("layered oxide",)
#: the accuracy a cold cell needs, from the tolerance stage of plan item BT12, kJ/mol
NEEDED = 15.0

#: kJ/mol in one hartree
KJ_PER_HARTREE = 2625.4996
#: the single solvents of the series: the name the label table uses, a short name for the
#: files of the runs, the SMILES, and whether one molecule can wrap the ion with several of its
#: oxygen atoms. The paper of the polyether gives
#: a molar mass of 222 and no formula; tetraethylene glycol dimethyl ether has that molar mass
#: and is computed in its place. Hexamethylphosphoramide is in no series: it is computed because
#: a paper prints its binding energy, which is one more point for the check of the method.
SOLVENTS: Tuple[Tuple[str, str, str, bool], ...] = (
    ("DMC", "DMC", "COC(=O)OC", False), ("EMC", "EMC", "CCOC(=O)OC", False),
    ("EC", "EC", "C1COC(=O)O1", False), ("PC", "PC", "CC1COC(=O)O1", False),
    ("FEC", "FEC", "FC1COC(=O)O1", False), ("DMSO", "DMSO", "CS(=O)C", False),
    ("THF", "THF", "C1CCOC1", False), ("acetonitrile", "acetonitrile", "CC#N", False),
    ("DMF", "DMF", "CN(C)C=O", False),
    ("2,2,2-trifluoroethyl acetate", "TFEAc", "CC(=O)OCC(F)(F)F", False),
    ("water", "water", "O", False), ("diglyme", "diglyme", "COCCOCCOC", True),
    ("poly(ethylene glycol) of molar mass 222", "tetraglyme", "COCCOCCOCCOCCOC", True),
    ("hexamethylphosphoramide", "HMPA", "CN(C)P(=O)(N(C)C)N(C)C", False),
)
#: solvents of the table above that are in no series
CHECK_ONLY = ("hexamethylphosphoramide",)
#: solvents of which one molecule holds the ion with several donor atoms
CHELATING = tuple(name for name, _key, _smiles, wraps in SOLVENTS if wraps)
#: binding energies that the papers of the label table give for Li+ with one solvent molecule,
#: in kJ/mol, with the sign changed so that a stronger binding is a larger number. A value is
#: *printed* or *read from a figure* (the horizontal position of the marker in Figure 3b of
#: Sawayama 2025, to about 1 kJ/mol). All are B3LYP values. Those of Yamada 2009 and Abe 2005
#: are reaction enthalpies at 298 K; those of Sawayama 2025 are differences of the electronic
#: energy with a counterpoise correction, as its supporting information says.
PUBLISHED_BINDING: Tuple[Tuple[str, float, str, str], ...] = (
    ("EC", 210.3, "printed", "Yamada2009"), ("DMC", 185.8, "printed", "Yamada2009"),
    ("PC", 217.1, "printed", "Abe2005"), ("DMSO", 235.5, "printed", "Abe2005"),
    ("FEC", 190.3, "printed", "Abe2005"),
    ("hexamethylphosphoramide", 273.7, "printed", "Sawayama2025"),
    ("2,2,2-trifluoroethyl acetate", 161.1, "read from a figure", "Sawayama2025"),
    ("THF", 182.5, "read from a figure", "Sawayama2025"),
    ("acetonitrile", 185.2, "read from a figure", "Sawayama2025"),
    ("PC", 213.7, "read from a figure", "Sawayama2025"),
    ("DMF", 243.1, "read from a figure", "Sawayama2025"),
)
#: the two levels of the ORCA runs: the geometry and the energy
LEVEL_GEOMETRY = "B97-3c TightSCF"
LEVEL_ENERGY = "B3LYP D3BJ def2-TZVPP RIJCOSX def2/J TightSCF"
#: starting geometries of the complex that the builder makes for one solvent
STARTS = 5
#: a species with more atoms than this is not optimised with ORCA: it keeps the geometry of
#: GFN2-xTB. On the serial ORCA build of this machine an optimisation of the largest complex
#: (38 atoms) would take several hours.
OPT_LIMIT = 20
#: the distance below which a donor atom counts as bound to the ion, in angstrom
SHELL = 2.6
#: draws of the bootstrap and of the permutation test, and their seed
DRAWS = 2000
SEED = 0


def to_kj(value: float, unit: str) -> float:
    """An activation energy in kJ/mol, from kJ/mol, from eV or from kcal/mol."""
    unit = unit.strip()
    if unit == "kJ/mol":
        return float(value)
    if unit == "eV":
        return float(value) * KJ_PER_EV
    if unit == "kcal/mol":
        return float(value) * KJ_PER_KCAL
    raise ValueError("an activation energy is given in kJ/mol, in eV or in kcal/mol, "
                     f"not in {unit!r}")


def read_labels(path: Optional[str] = None) -> List[dict]:
    """The rows of the label table, with the activation energy converted to kJ/mol.

    A row holds either one value or a range. `ea` is the value, or the middle of the range;
    `is_range` says which. `ea_sd` is the stated deviation when there is one.
    """
    path = path or os.path.join(DATA, "activation_energies.csv")
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            unit = r["unit"]
            if r["ea_reported"]:
                r["ea"] = to_kj(float(r["ea_reported"]), unit)
                r["is_range"] = False
                r["ea_low"] = r["ea_high"] = r["ea"]
            elif r["ea_low_reported"] and r["ea_high_reported"]:
                r["ea_low"] = to_kj(float(r["ea_low_reported"]), unit)
                r["ea_high"] = to_kj(float(r["ea_high_reported"]), unit)
                r["ea"] = 0.5 * (r["ea_low"] + r["ea_high"])
                r["is_range"] = True
            else:
                raise ValueError(f"row {r['id']} has neither a value nor a range")
            r["ea_sd"] = to_kj(float(r["ea_sd_reported"]), unit) if r["ea_sd_reported"] else None
            if r["read_from"] not in ("primary", "secondary"):
                raise ValueError(f"row {r['id']}: read_from is primary or secondary")
            rows.append(r)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("two rows of the label table have the same id")
    return rows


def read_sources(path: Optional[str] = None) -> Dict[str, dict]:
    path = path or os.path.join(DATA, "sources.csv")
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def is_reference(r: dict) -> bool:
    """The reference system: graphite with its film after formation, in a carbonate electrolyte
    with ethylene carbonate, with no additive."""
    return (r["electrode_class"] == "graphite" and r["solvent_class"] == "EC-based carbonate"
            and r["film"] == "formed" and r["additive"] == "none")


def repeatability(rows: List[dict]) -> dict:
    """The values of the reference system, one per study, and their spread."""
    by_study: Dict[str, List[float]] = {}
    for r in rows:
        if is_reference(r):
            by_study.setdefault(r["source"], []).append(r["ea"])
    studies = {k: float(np.mean(v)) for k, v in sorted(by_study.items())}
    v = np.array(list(studies.values()), float)
    return {"studies": studies, "n": int(len(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
            "low": float(v.min()), "high": float(v.max())}


def within_study(rows: List[dict]) -> List[dict]:
    """For every study and class of electrode with two or more values: what was varied and the
    range of the values. Ranges given as ranges are left out, since they are not single values."""
    groups: Dict[tuple, List[dict]] = {}
    for r in rows:
        if not r["is_range"]:
            groups.setdefault((r["source"], r["electrode_class"]), []).append(r)
    out = []
    for (source, electrode), g in sorted(groups.items()):
        if len(g) < 2:
            continue
        v = np.array([r["ea"] for r in g], float)
        varied = [name for name, key in (("solvent", "solvent"), ("salt", "salt"),
                                         ("additive", "additive"), ("film", "film"),
                                         ("state", "state"), ("form of the electrode", "form"),
                                         ("temperature range", "temperature_range"))
                  if len({r.get(key, "") for r in g}) > 1]
        out.append({"source": source, "electrode_class": electrode, "n": int(len(v)),
                    "low": float(v.min()), "high": float(v.max()),
                    "range": float(v.max() - v.min()), "varied": varied,
                    "read_from": sorted({r["read_from"] for r in g})})
    return out


def electrode_means(rows: List[dict]) -> Dict[str, dict]:
    """The mean for the two electrodes of a cell with a carbonate electrolyte: films after
    formation, every additive, one value per row, studies weighted by their rows."""
    out = {}
    for name, classes in (("negative", NEGATIVE), ("positive", POSITIVE)):
        v = np.array([r["ea"] for r in rows
                      if r["electrode_class"] in classes
                      and r["solvent_class"] == "EC-based carbonate"
                      and r["film"] in ("formed", "formed and cycled") and not r["is_range"]],
                     float)
        out[name] = {"n": int(len(v)), "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                     "low": float(v.min()), "high": float(v.max()),
                     "published": PUBLISHED[name], "shift": float(v.mean() - PUBLISHED[name])}
    return out


def solvent_name(solvent: str) -> str:
    """The solvent without the basis of its ratio and with 50:50 written as 1:1, so that one
    mixture has one name in every study."""
    name = solvent.replace(" by volume", "").replace(" by weight", "").strip()
    return name.replace("50:50", "1:1")


def solvent_series(rows: List[dict]) -> List[dict]:
    """The series of the table: rows of one study that share the electrode and the film, or the
    way the film was made, and differ in the solvent. The column `solvent_series` names the
    series of a row. Ranges are left out. The rows of a series are in the order of their
    values."""
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("solvent_series") and not r["is_range"]:
            groups.setdefault(r["solvent_series"], []).append(r)
    out = []
    for label, g in sorted(groups.items()):
        g = sorted(g, key=lambda r: r["ea"])
        v = [float(r["ea"]) for r in g]
        names = [solvent_name(r["solvent"]) for r in g]
        out.append({"series": label, "n": len(g), "solvents": names, "values": v,
                    "low": min(v), "high": max(v), "range": max(v) - min(v),
                    "one_solvent": sorted({n for n in names if ":" not in n}),
                    "read_from": sorted({r["read_from"] for r in g})})
    return out


def solvent_count(series: List[dict]) -> dict:
    """How many solvents the series hold, and how many of them more than one series measured."""
    names = sorted({n for s in series for n in s["solvents"]})
    single = sorted({n for s in series for n in s["one_solvent"]})
    shared = [n for n in single if sum(n in s["one_solvent"] for s in series) >= 2]
    return {"n_series": len(series), "n_rows": sum(s["n"] for s in series),
            "largest": max((s["n"] for s in series), default=0),
            "solvents": names, "one_solvent": single, "one_solvent_in_two_series": shared}


def shared_solvents(series: List[dict]) -> List[dict]:
    """Every single solvent that two or more series measured, with its value in each of them."""
    out = []
    for name in solvent_count(series)["one_solvent_in_two_series"]:
        values = [{"series": s["series"],
                   "value": float(np.mean([v for n, v in zip(s["solvents"], s["values"])
                                           if n == name]))}
                  for s in series if name in s["one_solvent"]]
        v = [x["value"] for x in values]
        out.append({"solvent": name, "values": values, "range": max(v) - min(v)})
    return out


def shared_contrasts(series: List[dict]) -> List[dict]:
    """Every pair of solvents that two or more series measured, with the difference of the two
    values in each of those series. A difference that changes from series to series is not a
    property of the two solvents alone."""
    pairs: Dict[tuple, List[dict]] = {}
    for s in series:
        values: Dict[str, List[float]] = {}
        for name, v in zip(s["solvents"], s["values"]):
            values.setdefault(name, []).append(v)
        names = sorted(values)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                pairs.setdefault((a, b), []).append(
                    {"series": s["series"],
                     "difference": float(np.mean(values[a]) - np.mean(values[b]))})
    return [{"first": a, "second": b, "differences": d}
            for (a, b), d in sorted(pairs.items()) if len(d) >= 2]


# ------------------------------------------------------------ the binding energy of Li+


def _orca_energy(job: Tuple[str, object, int, str, bool]) -> Tuple[str, dict, float]:
    """The ORCA energies of one species; ``(name, energies in hartree, seconds)``.

    ``energies["xtb_geometry"]`` is the energy at the geometry the species came with, which is
    the geometry of GFN2-xTB. ``energies["optimised"]`` is the energy after an optimisation with
    ORCA, and it is there only when `optimise` is set. A finished run is reused:
    `cmfqsar.orca.run_job` keeps the output next to the input and runs ORCA again only when
    the input has changed."""
    from cmfqsar import orca as O
    name, mol, charge, work, optimise = job
    t0 = time.time()
    cfg = O.Config(workdir=os.path.join(work, "orca"), maxcore=1500, timeout=6 * 3600.0)
    jobdir = os.path.join(cfg.workdir, O.safe_name(name))
    os.makedirs(jobdir, exist_ok=True)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    start = os.path.join(jobdir, "geom.xyz")
    if not os.path.isfile(start):
        with open(start, "w", newline="\n") as fh:
            fh.write(O.xyz_block(symbols, mol.GetConformer().GetPositions(), name))

    def energy(job_name, xyz):
        return float(O.run_job(jobdir, job_name,
                               O.make_input(LEVEL_ENERGY, charge, 1, xyz, populations=False,
                                            chelpg=False, maxcore=cfg.maxcore), cfg).energy)
    out = {}
    if len(symbols) == 1:                        # the bare ion: nothing to optimise
        out["optimised"] = out["xtb_geometry"] = energy("sp", "geom.xyz")
        return name, out, time.time() - t0
    if optimise:
        O.run_job(jobdir, "opt", O.make_input(LEVEL_GEOMETRY, charge, 1, "geom.xyz", opt=True,
                                              populations=False, chelpg=False,
                                              maxcore=cfg.maxcore), cfg, need_converged=True)
        out["optimised"] = energy("sp", "opt.xyz")
    if not optimise or not name.endswith("_b"):
        out["xtb_geometry"] = energy("sp_xtb", "geom.xyz")
    return name, out, time.time() - t0


def _orca_seconds(folder: str) -> float:
    """The run time that ORCA itself reports, summed over every output under `folder`. It is
    the cost of the stage whether the outputs were made now or are reused."""
    import re
    line = re.compile(r"TOTAL RUN TIME:\s+(\d+) days\s+(\d+) hours\s+(\d+) minutes\s+(\d+) "
                      r"seconds\s+(\d+) msec")
    total = 0.0
    for here, _dirs, files in os.walk(folder):
        for name in files:
            if not name.endswith(".out"):
                continue
            with open(os.path.join(here, name), errors="replace") as fh:
                found = line.findall(fh.read())
            if found:
                d, h, m, sec, ms = (float(x) for x in found[-1])
                total += 86400.0 * d + 3600.0 * h + 60.0 * m + sec + ms / 1000.0
    return total


def _donors_at_ion(jobdir: str, optimised: bool) -> Tuple[int, float]:
    """How many atoms other than hydrogen and carbon are within `SHELL` of the ion in the
    complex, and the shortest of those distances."""
    from cmfqsar import orca as O
    symbols, coords = O.read_xyz(os.path.join(jobdir, "opt.xyz" if optimised else "geom.xyz"))
    ion = [i for i, sym in enumerate(symbols) if sym.capitalize() == "Li"][0]
    d = [float(np.linalg.norm(coords[i] - coords[ion])) for i, sym in enumerate(symbols)
         if i != ion and sym.capitalize() not in ("H", "C")]
    near = sorted(x for x in d if x <= SHELL)
    return len(near), (near[0] if near else float("nan"))


def _atoms_at_ion(mol) -> int:
    """How many atoms other than hydrogen and carbon are within `SHELL` of the ion of a complex
    that the builder made. Fluorine counts, so that a contact of the ion with a fluorine atom
    is seen as another way of binding."""
    pos = np.asarray(mol.GetConformer().GetPositions(), float)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    ion = symbols.index("Li")
    return int(sum(1 for i, sym in enumerate(symbols)
                   if i != ion and sym not in ("H", "C")
                   and float(np.linalg.norm(pos[i] - pos[ion])) <= SHELL))


def binding_stage(work: str = WORK, jobs: int = 14) -> dict:
    """The binding energy of Li+ with one molecule of each solvent, in kJ/mol.

    The builder of plan item BT4 places the ion at the donor atom of the molecule from `STARTS`
    starting geometries, and for a chain with several oxygen atoms also inside the molecule.
    GFN2-xTB optimises every start. ORCA then optimises the start of lowest energy, and also the
    lowest start with another number of atoms at the ion when there is one (oxygen, nitrogen or
    fluorine within `SHELL`), because the two methods can order the two ways of binding
    differently. The complex of lowest ORCA energy
    gives the binding energy: the energy of the ion and of the free molecule minus the energy of
    the complex, each at its own optimised geometry. It is an electronic energy in the gas
    phase, with no thermal correction.

    A solvent of more than `OPT_LIMIT` atoms is not optimised with ORCA. Its free molecule and
    its complex of lowest GFN2-xTB energy keep the geometry of GFN2-xTB, and ORCA gives the
    energy only. The small solvents get that energy too, so the stage measures what the
    geometry of GFN2-xTB changes in a binding energy."""
    from concurrent.futures import ThreadPoolExecutor
    from rdkit import Chem
    sys.path.insert(0, ROOT)
    from cmfqsar import orca as O
    from cmfqsar import solvation as S

    engine = S.Engine(workdir=work)
    e_ion_xtb = S.ion_energy("Li", engine)
    ion = Chem.RWMol()
    atom = Chem.Atom("Li")
    atom.SetFormalCharge(1)
    atom.SetNoImplicit(True)
    ion.AddAtom(atom)
    ion = ion.GetMol()
    ion.AddConformer(Chem.Conformer(1), assignId=True)
    todo = [("ion_Li", ion, 1, work, False)]
    xtb, small = {}, {}
    for k, (name, short, smiles, wraps) in enumerate(SOLVENTS, 1):
        mol = Chem.MolFromSmiles(smiles)
        free, e_free = S.optimise(S._embed(mol, seed=k), f"solvent_{short}", engine,
                                  index=1000 + k)
        tried = []
        for start in range(STARTS):
            built = S.place_shell([mol], ion="Li", seed=10 * k + start)
            m, e = S.optimise(built.mol, f"{short}_1_s{start}", engine,
                              index=2000 + 10 * k + start)
            tried.append((e, _atoms_at_ion(m), m))
        if wraps:
            c, e, _all = S.best_wrapped_complex(mol, f"{short}_wrap", "Li", engine=engine,
                                                index=50 + k)
            tried.append((e, _atoms_at_ion(c.mol), c.mol))
        tried.sort(key=lambda t: t[0])
        xtb[name] = {"binding": -(tried[0][0] - e_ion_xtb - e_free) * KJ_PER_EV,
                     "spread_of_starts": (tried[-1][0] - tried[0][0]) * KJ_PER_EV}
        small[name] = free.GetNumAtoms() <= OPT_LIMIT
        todo.append((f"free_{short}", free, 0, work, small[name]))
        todo.append((f"complex_{short}_a", tried[0][2], 1, work, small[name]))
        other = [t for t in tried if t[1] != tried[0][1]]
        if other and small[name]:
            todo.append((f"complex_{short}_b", other[0][2], 1, work, True))
    todo.sort(key=lambda j: -j[1].GetNumAtoms())            # the longest runs start first
    print(f"    {len(todo)} species for ORCA, {jobs} at a time", flush=True)
    energies, seconds = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as pool:
        for name, e, sec in pool.map(_orca_energy, todo):
            energies[name], seconds[name] = e, sec
            print(f"    {name:<24} " + "  ".join(f"{k} {v:.8f}" for k, v in e.items())
                  + f"  {sec:7.0f} s", flush=True)
    e_ion = energies["ion_Li"]["optimised"]
    rows = []
    for name, short, smiles, wraps in SOLVENTS:
        key = "optimised" if small[name] else "xtb_geometry"
        modes = {k: v[key] for k, v in energies.items()
                 if k.startswith(f"complex_{short}_") and key in v}
        best = min(modes, key=modes.get)
        donors, nearest = _donors_at_ion(os.path.join(work, "orca", O.safe_name(best)),
                                         small[name])
        zero = e_ion + energies[f"free_{short}"][key]
        at_xtb = -(energies[f"complex_{short}_a"]["xtb_geometry"] - e_ion
                   - energies[f"free_{short}"]["xtb_geometry"]) * KJ_PER_HARTREE
        same_way = (-(energies[f"complex_{short}_a"]["optimised"] - zero) * KJ_PER_HARTREE
                    if small[name] else None)
        rows.append({"solvent": name, "smiles": smiles, "wraps": bool(wraps),
                     "atoms": int(Chem.AddHs(Chem.MolFromSmiles(smiles)).GetNumAtoms()),
                     "geometry": "ORCA" if small[name] else "GFN2-xTB",
                     "binding": -(modes[best] - zero) * KJ_PER_HARTREE,
                     "ways_of_binding": sorted(-(v - zero) * KJ_PER_HARTREE
                                               for v in modes.values()),
                     "binding_at_xtb_geometry": at_xtb,
                     "binding_of_the_same_start": same_way,
                     "donors_at_ion": donors, "nearest_donor": nearest,
                     "binding_xtb": xtb[name]["binding"],
                     "spread_of_xtb_starts": xtb[name]["spread_of_starts"],
                     "in_a_series": name not in CHECK_ONLY})
    shift = np.array([r["binding_at_xtb_geometry"] - r["binding_of_the_same_start"]
                      for r in rows if r["binding_of_the_same_start"] is not None], float)
    return {"rows": rows, "level_geometry": LEVEL_GEOMETRY, "level_energy": LEVEL_ENERGY,
            "ion_energy_hartree": e_ion, "starts": STARTS, "opt_limit": OPT_LIMIT,
            "xtb_geometry_shift": {"n": int(len(shift)), "mean": float(shift.mean()),
                                   "sd": float(shift.std(ddof=1)),
                                   "largest": float(np.abs(shift).max())},
            "species": len(todo), "orca_seconds": _orca_seconds(os.path.join(work, "orca")),
            "check": check_binding(rows)}


def check_binding(rows: List[dict]) -> dict:
    """The computed binding energies against the values that the papers give, for the solvents
    that have one: the correlation, and the mean and the standard deviation of the difference.
    Both methods of this stage are checked, ORCA and GFN2-xTB."""
    by = {r["solvent"]: r for r in rows}
    pairs = [(by[name], value, how, source) for name, value, how, source in PUBLISHED_BINDING
             if name in by]
    out = {"n": len(pairs),
           "pairs": [{"solvent": r["solvent"], "published": v, "how": how, "source": src,
                      "computed": r["binding"], "xtb": r["binding_xtb"]}
                     for r, v, how, src in pairs]}
    def compare(chosen, key):
        x = np.array([r[key] for r, _v, _h, _s in chosen], float)
        y = np.array([v for _r, v, _h, _s in chosen], float)
        return {"n": int(len(x)),
                "correlation": float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 else None,
                "mean_difference": float((x - y).mean()),
                "sd_difference": float((x - y).std(ddof=1)) if len(x) > 1 else None}
    out["orca"] = compare(pairs, "binding")
    out["xtb"] = compare(pairs, "binding_xtb")
    # a solvent with two ways of binding: the paper may have computed the other one
    one_way = [p for p in pairs if len(p[0].get("ways_of_binding", [])) <= 1]
    out["two_ways"] = sorted({p[0]["solvent"] for p in pairs
                              if len(p[0].get("ways_of_binding", [])) > 1})
    out["orca_one_way"] = compare(one_way, "binding")
    out["xtb_one_way"] = compare(one_way, "binding_xtb")
    return out


# ------------------------------------------------------------ the regression


def components(solvent: str) -> Tuple[str, ...]:
    """The solvents of an electrolyte, from its name in the table: ``EC:DMC 1:9`` gives
    ``("EC", "DMC")`` and a single solvent gives itself."""
    head = solvent.split(" ")[0]
    return tuple(head.split(":")) if ":" in head else (solvent,)


def regression_points(rows: List[dict], binding: Dict[str, float], mixtures: bool = False,
                      leave_out: Sequence[str] = ()) -> List[dict]:
    """The points of the regression: the rows of the series with one solvent, each with the
    binding energy of its solvent. With `mixtures`, a mixture is a point too and takes the
    binding energy of its most strongly binding solvent, which is the hypothesis that this
    solvent fills the shell of the ion. A solvent named in `leave_out` is not used, and neither
    is a mixture that holds it. A series that keeps fewer than two points is left out."""
    points = []
    for r in rows:
        if not r.get("solvent_series") or r["is_range"]:
            continue
        name = solvent_name(r["solvent"])
        parts = components(name)
        if (len(parts) > 1 and not mixtures) or any(p in leave_out for p in parts):
            continue
        if any(p not in binding for p in parts):
            raise ValueError(f"row {r['id']}: no binding energy for {', '.join(parts)}")
        points.append({"id": r["id"], "series": r["solvent_series"], "solvent": name,
                       "parts": parts, "y": float(r["ea"]),
                       "x": max(float(binding[p]) for p in parts)})
    count: Dict[str, int] = {}
    for p in points:
        count[p["series"]] = count.get(p["series"], 0) + 1
    return [p for p in points if count[p["series"]] >= 2]


def fit_slope(points: List[dict]) -> Tuple[float, Dict[str, Tuple[float, float]]]:
    """One slope for all series and one intercept per series, by least squares.

    Returns the slope and, for every series, the means of x and of y of its points. The slope
    is the regression of the centred y on the centred x; it is 0 when no series holds two
    different values of x."""
    means: Dict[str, Tuple[float, float]] = {}
    for name in {p["series"] for p in points}:
        g = [p for p in points if p["series"] == name]
        means[name] = (float(np.mean([p["x"] for p in g])), float(np.mean([p["y"] for p in g])))
    xc = np.array([p["x"] - means[p["series"]][0] for p in points], float)
    yc = np.array([p["y"] - means[p["series"]][1] for p in points], float)
    sxx = float(xc @ xc)
    return (float(xc @ yc) / sxx if sxx > 1e-12 else 0.0), means


def hold_out_solvents(points: List[dict]) -> List[dict]:
    """Every solvent held out once, with every mixture that holds it.

    For each held-out point there are two predictions. The *mean* is the mean of the points of
    its series that stay in. The *regression* adds the slope, fitted on the points that stay
    in, times the distance of its binding energy from the mean binding energy of those points.
    A point whose series keeps no point is not predicted. A mixture is held out once for each
    of its solvents."""
    out = []
    for solvent in sorted({part for p in points for part in p["parts"]}):
        test = [p for p in points if solvent in p["parts"]]
        train = [p for p in points if solvent not in p["parts"]]
        if not train:
            continue
        slope, means = fit_slope(train)
        for p in test:
            if p["series"] not in means:
                continue
            xm, ym = means[p["series"]]
            out.append({"held_out": solvent, "id": p["id"], "series": p["series"],
                        "solvent": p["solvent"], "y": p["y"], "mean": ym,
                        "regression": ym + slope * (p["x"] - xm), "slope": slope})
    return out


def _errors(predictions: List[dict]) -> dict:
    y = np.array([q["y"] for q in predictions], float)
    out = {"n": int(len(y))}
    for key in ("mean", "regression"):
        e = np.array([q[key] for q in predictions], float) - y
        out[key] = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean()))}
    return out


def gain_interval(predictions: List[dict], draws: int = DRAWS, seed: int = SEED) -> dict:
    """The gain of the regression over the mean of the series: the mean absolute error of the
    mean minus that of the regression, with a 95 % interval from a bootstrap over the held-out
    solvents (plan rule 4). The solvents are drawn, and not the points, because the points of
    one solvent are predicted by one fit."""
    groups: Dict[str, List[dict]] = {}
    for q in predictions:
        groups.setdefault(q["held_out"], []).append(q)
    names = sorted(groups)

    def gain(chosen):
        qs = [q for name in chosen for q in groups[name]]
        y = np.array([q["y"] for q in qs], float)
        return float(np.abs(np.array([q["mean"] for q in qs]) - y).mean()
                     - np.abs(np.array([q["regression"] for q in qs]) - y).mean())
    rng = np.random.default_rng(seed)
    boot = [gain([names[i] for i in rng.integers(0, len(names), len(names))])
            for _ in range(int(draws))]
    return {"gain": gain(names), "low": float(np.percentile(boot, 2.5)),
            "high": float(np.percentile(boot, 97.5)), "groups": len(names)}


def series_disagreement(predictions: List[dict]) -> dict:
    """The part of the error that no model of the solvent alone can remove.

    Take an electrolyte that was held out in two or more series. Its errors in those series
    differ. A better model of the solvent changes the prediction for that electrolyte, and with
    one slope and one level per series it changes it by the same amount in every series. The
    smallest mean absolute error it can reach is therefore the mean distance of the errors from
    their median. The result gives, over the electrolytes that are in two or more series, the
    mean absolute error of the regression and that smallest error."""
    groups: Dict[tuple, List[float]] = {}
    for q in predictions:
        groups.setdefault((q["held_out"], q["solvent"]), []).append(q["regression"] - q["y"])
    shared = {k: np.array(v, float) for k, v in groups.items() if len(v) >= 2}
    if not shared:
        return {"n": 0, "electrolytes": [], "regression_mae": None, "floor_mae": None}
    errors = np.concatenate(list(shared.values()))
    floor = np.concatenate([v - np.median(v) for v in shared.values()])
    return {"n": int(len(errors)), "electrolytes": sorted({k[1] for k in shared}),
            "regression_mae": float(np.abs(errors).mean()),
            "floor_mae": float(np.abs(floor).mean())}


def permutation_test(points: List[dict], draws: int = DRAWS, seed: int = SEED) -> dict:
    """How often a binding energy that belongs to another solvent does as well.

    The binding energies are exchanged at random between the single solvents, each mixture takes
    the value of its strongest solvent again, and the held-out error of the regression is
    computed as before. The result is the share of the draws whose error is not above the error
    with the true assignment."""
    solvents = sorted({part for p in points for part in p["parts"]})
    value = {}
    for p in points:
        if len(p["parts"]) == 1:
            value[p["solvent"]] = p["x"]
    missing = [name for name in solvents if name not in value]
    if missing:                 # a solvent that occurs in mixtures only keeps its place
        solvents = [name for name in solvents if name in value]
    true = _errors(hold_out_solvents(points))["regression"]["mae"]
    rng = np.random.default_rng(seed)
    base = np.array([value[name] for name in solvents], float)
    not_above = 0
    for _ in range(int(draws)):
        shuffled = dict(zip(solvents, rng.permutation(base)))
        moved = [dict(p, x=max(shuffled.get(part, p["x"]) for part in p["parts"]))
                 for p in points]
        if _errors(hold_out_solvents(moved))["regression"]["mae"] <= true + 1e-12:
            not_above += 1
    return {"draws": int(draws), "share_not_above": not_above / float(draws),
            "solvents": len(solvents)}


def analyse(points: List[dict], draws: int = DRAWS, seed: int = SEED) -> dict:
    """The regression on one set of points: the fit on all points, the errors with one solvent
    held out, the gain over the mean with its interval, and the permutation test."""
    slope, means = fit_slope(points)
    xc = np.array([p["x"] - means[p["series"]][0] for p in points], float)
    yc = np.array([p["y"] - means[p["series"]][1] for p in points], float)
    r = float(np.corrcoef(xc, yc)[0, 1]) if xc.std() > 0 and yc.std() > 0 else float("nan")
    resid = yc - slope * xc
    dof = len(points) - len(means) - 1
    predictions = hold_out_solvents(points)
    per_series = []
    for name in sorted(means):
        g = [p for p in points if p["series"] == name]
        x = np.array([p["x"] for p in g], float)
        y = np.array([p["y"] for p in g], float)
        own = float(np.polyfit(x, y, 1)[0]) if len(set(x.tolist())) > 1 else None
        per_series.append({"series": name, "n": len(g), "slope": own,
                           "range_x": float(x.max() - x.min()),
                           "range_y": float(y.max() - y.min())})
    return {"n_points": len(points), "n_series": len(means),
            "n_solvents": len({part for p in points for part in p["parts"]}),
            "slope": slope, "correlation_inside_series": r,
            "sd_y_inside_series": float(np.sqrt((yc ** 2).sum() / max(1, len(points) - len(means)))),
            "residual_sd": float(np.sqrt((resid ** 2).sum() / dof)) if dof > 0 else None,
            "held_out": _errors(predictions), "gain": gain_interval(predictions, draws, seed),
            "permutation": permutation_test(points, draws, seed),
            "between_series": series_disagreement(predictions),
            "per_series": per_series, "predictions": predictions}


#: the sets of points on which the regression is run, with the words the table uses
VARIANTS = (
    ("single", "single solvents only", {}),
    ("single_no_chelate", "single solvents, without the two that wrap the ion",
     {"leave_out": CHELATING}),
    ("mixtures", "single solvents and mixtures; a mixture takes its strongest solvent",
     {"mixtures": True}),
    ("mixtures_no_chelate", "the same, without the two solvents that wrap the ion",
     {"mixtures": True, "leave_out": CHELATING}),
)


def regression_stage(binding: dict, rows: Optional[List[dict]] = None,
                     draws: int = DRAWS) -> dict:
    """The regression of the activation energy on the binding energy, on every set of points
    of `VARIANTS`, with the computed binding energies; and once more with the binding energies
    that the papers give, for the solvents that have one."""
    rows = read_labels() if rows is None else rows
    computed = {r["solvent"]: r["binding"] for r in binding["rows"]}
    out = {"variants": {}, "repeatability": repeatability(rows)["sd"], "draws": int(draws)}
    for key, words, options in VARIANTS:
        points = regression_points(rows, computed, **options)
        out["variants"][key] = dict(analyse(points, draws), words=words)
    printed: Dict[str, List[float]] = {}
    for name, value, _how, _source in PUBLISHED_BINDING:
        printed.setdefault(name, []).append(value)
    published = {name: float(np.mean(v)) for name, v in printed.items()}
    lacking = [name for name in computed if name not in published]
    points = regression_points(rows, published, leave_out=lacking)
    out["variants"]["published"] = dict(
        analyse(points, draws),
        words="single solvents, with the binding energies that the papers give")
    return out


def cold_cost(means: Dict[str, dict], results: str = CELLMODEL) -> List[dict]:
    """What the stated activation energy nearest to each mean costs a cold cell, from the files
    of the exchange-current stage. Empty when that stage has not been run."""
    out = []
    if not os.path.isdir(results):
        return out
    for name in sorted(os.listdir(results)):
        if not (name.startswith("tolerance_exchange_") and name.endswith(".json")):
            continue
        with open(os.path.join(results, name), encoding="utf-8") as fh:
            entry = json.load(fh)
        if not entry.get("energies"):
            continue
        row = {"temperature": entry["temperature"], "rate": entry["rate"]}
        for key in ("negative", "positive"):
            stated = entry["energies"][key]
            nearest = min(stated, key=lambda s: abs(s["energy"] - means[key]["mean"]))
            row[key] = {"energy": nearest["energy"],
                        "share": nearest["share"] if nearest["usable"] else None}
        both = entry["energies"]["both"]
        mean_both = 0.5 * (means["negative"]["mean"] + means["positive"]["mean"])
        nearest = min(both, key=lambda s: abs(s["energy"] - mean_both))
        row["both"] = {"energy": nearest["energy"],
                       "share": nearest["share"] if nearest["usable"] else None}
        out.append(row)
    return sorted(out, key=lambda r: (-r["temperature"], r["rate"]))


def summarise(rows: List[dict], sources: Dict[str, dict]) -> dict:
    missing = sorted({r["source"] for r in rows} - set(sources))
    if missing:
        raise ValueError(f"no record in sources.csv for: {', '.join(missing)}")
    means = electrode_means(rows)
    series = solvent_series(rows)
    return {"n_rows": len(rows),
            "n_primary": sum(r["read_from"] == "primary" for r in rows),
            "n_secondary": sum(r["read_from"] == "secondary" for r in rows),
            "n_studies": len({r["source"] for r in rows}),
            "classes": {c: sum(r["electrode_class"] == c for r in rows)
                        for c in sorted({r["electrode_class"] for r in rows})},
            "repeatability": repeatability(rows), "within_study": within_study(rows),
            "means": means, "cold": cold_cost(means),
            "series": series, "solvent_count": solvent_count(series),
            "shared_solvents": shared_solvents(series),
            "contrasts": shared_contrasts(series),
            "to_read": sorted(k for k, s in sources.items()
                              if not s["status"].startswith("read"))}


def _f(v, nd: int = 1) -> str:
    return "none" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"


def _interval(g: dict) -> str:
    return f"{g['gain']:+.1f} [{g['low']:+.1f}, {g['high']:+.1f}]"


def binding_lines(binding: dict) -> List[str]:
    """The section of the table on the computed binding energies."""
    published: Dict[str, List[str]] = {}
    for name, value, how, _source in PUBLISHED_BINDING:
        published.setdefault(name, []).append(f"{value:.1f}" + ("" if how == "printed" else " (figure)"))
    L = ["### The binding energy of Li+ with one molecule of each solvent", "",
         f"Computed with ORCA: the geometry with `{binding['level_geometry']}`, the energy with "
         f"`{binding['level_energy']}`. The value is the energy of the ion and of the free "
         "molecule minus the energy of the complex, each at its own optimised geometry, in the "
         "gas phase and with no thermal correction. A larger number is a stronger binding. The "
         "starting geometries come from the builder of plan item BT4, which optimises them "
         f"with GFN2-xTB from {binding['starts']} starts. *Published* is the value that a paper "
         "of the label table gives for the same complex, calculated with B3LYP: a reaction "
         "enthalpy or a difference of electronic energies, and, where marked, the position "
         "of a marker in a figure. The ORCA runs of the "
         f"{binding['species']} species took {binding['orca_seconds'] / 60.0:.0f} minutes of "
         "processor time in all.", "",
         "| Solvent | Binding energy, kJ/mol | Geometry from | Donor atoms at the ion | "
         "Shortest distance to the ion, angstrom | With GFN2-xTB alone, kJ/mol | Published, "
         "kJ/mol |",
         "|---|---:|---|---:|---:|---:|---|"]
    for r in sorted(binding["rows"], key=lambda r: r["binding"]):
        name = r["solvent"] + ("" if r["in_a_series"] else " (in no series)")
        L.append(f"| {name} | {_f(r['binding'])} | {r['geometry']} | {r['donors_at_ion']} | "
                 f"{_f(r['nearest_donor'], 2)} | {_f(r['binding_xtb'])} | "
                 f"{', '.join(published.get(r['solvent'], [])) or 'none'} |")
    c = binding["check"]
    g = binding["xtb_geometry_shift"]
    L += ["", f"A solvent of more than {binding['opt_limit']} atoms was not optimised with "
              "ORCA, because the ORCA build of this machine runs on one core. It keeps the "
              "geometry of GFN2-xTB and ORCA gives the energy only. What that changes was "
              f"measured on the {g['n']} smaller solvents, which were computed both ways: the "
              f"binding energy at the geometry of GFN2-xTB is different by {g['mean']:+.1f} "
              f"kJ/mol on average, with a standard deviation of {_f(g['sd'])} kJ/mol and a "
              f"largest difference of {_f(g['largest'])} kJ/mol."]
    L += ["", "The polyether of the label table is given by its paper with a molar mass of 222 "
              "and no formula. Tetraethylene glycol dimethyl ether has that molar mass and was "
              "computed in its place.", "",
          f"Against the {c['n']} published values, the values of ORCA have a correlation of "
          f"{_f(c['orca']['correlation'], 3)}, and they are higher by {_f(c['orca']['mean_difference'])} "
          f"kJ/mol on average, with a standard deviation of the difference of "
          f"{_f(c['orca']['sd_difference'])} kJ/mol. The values of GFN2-xTB, the method of plan "
          f"item BT4, have a correlation of {_f(c['xtb']['correlation'], 3)}, a mean difference "
          f"of {_f(c['xtb']['mean_difference'])} kJ/mol and a standard deviation of "
          f"{_f(c['xtb']['sd_difference'])} kJ/mol. For this reason the regression uses the "
          "values of ORCA.", ""]
    two = [r for r in binding["rows"] if len(r["ways_of_binding"]) > 1]
    if two:
        L += ["A solvent can bind the ion in two ways. The table gives the complex of lower "
              "energy, and a paper may have computed the other one. "
              + " ".join(f"For {r['solvent']} the two complexes give "
                         f"{' and '.join(_f(v) for v in r['ways_of_binding'])} kJ/mol; the one "
                         f"of lower energy holds the ion with {r['donors_at_ion']} atoms."
                         for r in two)
              + f" Without {'this solvent' if len(two) == 1 else 'these solvents'}, the "
              f"{c['orca_one_way']['n']} other published values and the values of ORCA have a "
              f"correlation of {_f(c['orca_one_way']['correlation'], 3)}, a mean difference of "
              f"{_f(c['orca_one_way']['mean_difference'])} kJ/mol and a standard deviation of "
              f"the difference of {_f(c['orca_one_way']['sd_difference'])} kJ/mol; for GFN2-xTB "
              f"the three numbers are {_f(c['xtb_one_way']['correlation'], 3)}, "
              f"{_f(c['xtb_one_way']['mean_difference'])} and "
              f"{_f(c['xtb_one_way']['sd_difference'])}.", ""]
    return L


def regression_lines(reg: dict) -> List[str]:
    """The sections of the table on the regression."""
    L = ["### The regression, with one solvent held out at a time", "",
         "The regression has one slope for all series and one intercept for each series, so it "
         "describes only the differences inside a series. Every solvent is held out once. The "
         "*mean of the series* predicts a held-out value by the mean of the other values of "
         "its series, and the regression adds the slope times the distance of the binding "
         "energy from the mean of those other values. The error is the mean absolute error "
         "over the held-out values, in kJ/mol. The *gain* is the error of the mean minus the "
         "error of the regression, with a 95 % interval from a bootstrap over the held-out "
         f"solvents ({reg['draws']} draws). The last column is a test against chance: the "
         "binding energies are exchanged at random between the solvents, and the column "
         "gives the share of the exchanges whose error is not above the error with the true "
         "binding energies.", "",
         "| Points of the regression | Series | Values | Solvents | Slope | Correlation inside "
         "the series | Error of the mean of the series | Error of the regression | Gain, with "
         "its 95 % interval | Exchanges that do as well |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    for key in reg["variants"]:
        v = reg["variants"][key]
        h = v["held_out"]
        L.append(f"| {v['words']} | {v['n_series']} | {v['n_points']} | {v['n_solvents']} | "
                 f"{v['slope']:+.3f} | {_f(v['correlation_inside_series'], 2)} | "
                 f"{_f(h['mean']['mae'])} | {_f(h['regression']['mae'])} | "
                 f"{_interval(v['gain'])} | {100 * v['permutation']['share_not_above']:.1f} % |")
    L += ["", f"The repeatability between laboratories, from the reference system above, is "
              f"{_f(reg['repeatability'])} kJ/mol. The slope is in kJ/mol of activation energy "
              "per kJ/mol of binding energy. A mixture is held out once for each of its "
              "solvents, so it is counted more than once in the rows with mixtures.", ""]
    L += ["### What a model of the solvent alone cannot remove", "",
          "Some electrolytes were held out in two or more series. The errors of the regression "
          "for one electrolyte differ between its series. A better model of the solvent can "
          "move these errors by one amount for all series and no more, so the mean distance of "
          "the errors from their median is the smallest error that any model of the solvent "
          "alone can reach on these values.", "",
          "| Points of the regression | Values of electrolytes in two or more series | Error "
          "of the regression on them | Smallest error of a model of the solvent alone |",
          "|---|---:|---:|---:|"]
    for key in reg["variants"]:
        v = reg["variants"][key]
        d = v["between_series"]
        L.append(f"| {v['words']} | {d['n']} | {_f(d['regression_mae'])} | "
                 f"{_f(d['floor_mae'])} |")
    L.append("")
    first = reg["variants"]["single"]
    L += ["### The slope of each series, single solvents only", "",
          "| Series | Values | Range of the binding energy, kJ/mol | Range of the activation "
          "energy, kJ/mol | Slope of the series alone |", "|---|---:|---:|---:|---:|"]
    for q in first["per_series"]:
        slope = "none" if q["slope"] is None else f"{q['slope']:+.3f}"
        L.append(f"| {q['series']} | {q['n']} | {_f(q['range_x'])} | {_f(q['range_y'])} | "
                 f"{slope} |")
    for key, words in (("single", "single solvents only"),
                       ("single_no_chelate", "single solvents, without the two that wrap the "
                                             "ion")):
        L += ["", f"### The held-out values: {words}", "",
              "| Held-out solvent | Series | Measured, kJ/mol | Mean of the series | Regression "
              "| Error of the regression |", "|---|---|---:|---:|---:|---:|"]
        for q in reg["variants"][key]["predictions"]:
            L.append(f"| {q['held_out']} | {q['series']} | {_f(q['y'])} | {_f(q['mean'])} | "
                     f"{_f(q['regression'])} | {q['regression'] - q['y']:+.1f} |")
    L.append("")
    return L


def table(res: dict, sources: Dict[str, dict], binding: Optional[dict] = None,
          regression: Optional[dict] = None) -> str:
    rep, means = res["repeatability"], res["means"]
    L = ["# Measured activation energies of charge transfer at an electrode (plan item BT12)", "",
         "The values were collected from the literature and are in "
         "`examples/charge_transfer/activation_energies.csv`, one row per value, with the paper "
         "and the place in it. A row is *primary* when the number was read in the paper that "
         "measured it, and *secondary* when it was read in another paper that cites it. All "
         "values below are in kJ/mol; a value given in eV was multiplied by 96.485 and a value "
         "given in kcal/mol by 4.184. No model is fitted here.", "",
         f"The table holds {res['n_rows']} rows from {res['n_studies']} studies: "
         f"{res['n_primary']} primary and {res['n_secondary']} secondary. By class of electrode: "
         + ", ".join(f"{c} {n}" for c, n in res["classes"].items()) + ".", "",
         "## How well laboratories agree on one system", "",
         "The reference system is a graphite electrode with its surface film after formation, in "
         "a carbonate electrolyte that contains ethylene carbonate, with no additive. One value "
         "per study, the mean of its rows for this system.", "",
         "| Study | Activation energy, kJ/mol |", "|---|---:|"]
    for k, v in rep["studies"].items():
        L.append(f"| {k} | {_f(v)} |")
    L += ["", f"Over these {rep['n']} studies the mean is {_f(rep['mean'])} kJ/mol, the standard "
              f"deviation {_f(rep['sd'])} kJ/mol and the range {_f(rep['low'])} to "
              f"{_f(rep['high'])} kJ/mol. The studies differ in the salt, in the second solvent "
              "and in the kind of graphite, so this spread is an upper estimate of the "
              "repeatability of one system.", "",
          "## How much the value changes inside one study", "",
          "| Study | Electrode | Values | Lowest | Highest | Range | What differs between the rows "
          "| Read as |", "|---|---|---:|---:|---:|---:|---|---|"]
    for w in res["within_study"]:
        L.append(f"| {w['source']} | {w['electrode_class']} | {w['n']} | {_f(w['low'])} | "
                 f"{_f(w['high'])} | {_f(w['range'])} | {', '.join(w['varied']) or 'nothing'} | "
                 f"{' and '.join(w['read_from'])} |")
    L += ["", "## The two electrodes of a cell with a carbonate electrolyte", "",
          "Films after formation, every additive, one value per row. The published value is the "
          "activation energy that the cell model of plan item MS9 holds.", "",
          "| Electrode | Rows | Mean | Standard deviation | Lowest | Highest | Published value | "
          "Mean minus published |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("negative", "positive"):
        m = means[name]
        L.append(f"| {name} | {m['n']} | {_f(m['mean'])} | {_f(m['sd'])} | {_f(m['low'])} | "
                 f"{_f(m['high'])} | {_f(m['published'])} | {m['shift']:+.1f} |")
    if res["cold"]:
        L += ["", "What that difference costs a cold cell, from the stage `--stage exchange` of "
                  "`run_cellmodel.py`. The stage ran stated activation energies in steps of "
                  "10 kJ/mol; the column gives the one nearest to the mean and the difference of "
                  "the capacity, in per cent of the capacity with the published values.", "",
              "| Temperature, K | Rate | Negative electrode alone | Positive electrode alone | "
              "Both electrodes |", "|---:|---:|---|---|---|"]

        def cell(c):
            share = "no usable discharge" if c["share"] is None else f"{100 * c['share']:+.1f} %"
            return f"{c['energy']:g} kJ/mol: {share}"
        for r in res["cold"]:
            L.append(f"| {r['temperature']:.0f} | {r['rate']:g}C | {cell(r['negative'])} | "
                     f"{cell(r['positive'])} | {cell(r['both'])} |")
    count = res["solvent_count"]
    L += ["", "## The series in which only the solvent changes", "",
          "A series is a set of rows of one study that share the electrode and the film, or the "
          "way the film was made, and differ in the solvent. Part (c) of the gate of the item "
          "needs such series: it centres each study on its own mean and holds out every "
          "electrolyte with one solvent together.", "",
          "| Series | Rows | Solvents, from the lowest value to the highest | Lowest | Highest | "
          "Range | Read as |", "|---|---:|---|---:|---:|---:|---|"]
    for q in res["series"]:
        L.append(f"| {q['series']} | {q['n']} | {'; '.join(q['solvents'])} | {_f(q['low'])} | "
                 f"{_f(q['high'])} | {_f(q['range'])} | {' and '.join(q['read_from'])} |")
    L += ["", f"The {count['n_series']} series hold {count['n_rows']} rows, and the largest "
              f"holds {count['largest']}. They hold {len(count['solvents'])} different solvents "
              f"or mixtures, of which {len(count['one_solvent'])} are single solvents: "
              + ", ".join(count["one_solvent"]) + ". "
              f"{len(count['one_solvent_in_two_series'])} single solvents were measured in two "
              "or more series: " + (", ".join(count["one_solvent_in_two_series"]) or "none")
              + ".", ""]
    if res["shared_solvents"]:
        L += ["### One solvent in more than one series", "",
              "The value of a single solvent in each series that measured it. The series differ "
              "in the electrode, in the film, in the salt and in its concentration.", "",
              "| Solvent | Series | Activation energy, kJ/mol |", "|---|---|---:|"]
        for q in res["shared_solvents"]:
            for x in q["values"]:
                L.append(f"| {q['solvent']} | {x['series']} | {_f(x['value'])} |")
        L.append("")
    if res["contrasts"]:
        L += ["### One pair of solvents in more than one series", "",
              "For every pair of solvents that two or more series measured: the value in the "
              "first solvent minus the value in the second, in each series. A difference that "
              "changes from series to series is not a property of the two solvents alone.", "",
              "| First solvent | Second solvent | Series | First minus second, kJ/mol |",
              "|---|---|---|---:|"]
        for c in res["contrasts"]:
            for d in c["differences"]:
                L.append(f"| {c['first']} | {c['second']} | {d['series']} | "
                         f"{d['difference']:+.1f} |")
    else:
        L += ["No pair of solvents was measured in two series."]
    if binding is not None:
        L += ["", "## The reference: a regression on the binding energy of Li+ with the solvent",
              "",
              "The hypothesis that the removal of the solvent shell sets the barrier has a "
              "version with one number: the activation energy rises with the binding energy of "
              "Li+ with one molecule of the solvent. It is the reference that a field model of "
              "the solvent has to be better than (part (c) of the gate).", ""]
        L += binding_lines(binding)
        if regression is not None:
            L += regression_lines(regression)
        while L and L[-1] == "":
            L.pop()
    L += ["", "## Papers that are cited or known and have not been read", "",
          "| Key | Reference | State |", "|---|---|---|"]
    for k in res["to_read"]:
        s = sources[k]
        ref = ", ".join(x for x in (s["authors"], s["journal"], s["year"], s["volume"],
                                    s["pages"]) if x) or s["title"] or s["note"]
        L.append(f"| {k} | {ref} | {s['status']} |")
    L.append("")
    return "\n".join(L)


def run(results: str = RESULTS) -> dict:
    rows, sources = read_labels(), read_sources()
    res = summarise(rows, sources)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


def _stored(results: str, name: str) -> Optional[dict]:
    path = os.path.join(results, name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def report(results: str = RESULTS) -> str:
    res = _stored(results, "summary.json")
    if res is None:
        raise SystemExit(f"{os.path.join(results, 'summary.json')} is missing: run this "
                         "benchmark without --report first")
    out = os.path.join(results, "table.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(table(res, read_sources(), _stored(results, "binding.json"),
                       _stored(results, "regression.json")))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--report", action="store_true",
                    help="write the table from the stored summary and stop")
    ap.add_argument("--results", default=RESULTS, help="where the result is written")
    ap.add_argument("--stage", choices=("summary", "binding", "regression"), default="summary",
                    help="summary: what the collected values say (the default); binding: the "
                         "binding energy of Li+ with each solvent, with ORCA; regression: the "
                         "regression of the activation energy on it")
    ap.add_argument("--jobs", type=int, default=14,
                    help="ORCA processes that run side by side in the stage binding")
    ap.add_argument("--draws", type=int, default=DRAWS,
                    help="draws of the bootstrap and of the permutation test")
    args = ap.parse_args(argv)
    if not args.report and args.stage == "binding":
        os.makedirs(args.results, exist_ok=True)
        out = binding_stage(jobs=args.jobs)
        with open(os.path.join(args.results, "binding.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        c = out["check"]
        print(f"{len(out['rows'])} solvents; against {c['n']} published values ORCA has a "
              f"correlation of {c['orca']['correlation']:.3f} and GFN2-xTB of "
              f"{c['xtb']['correlation']:.3f}")
    elif not args.report and args.stage == "regression":
        binding = _stored(args.results, "binding.json")
        if binding is None:
            raise SystemExit("binding.json is missing: run --stage binding first")
        out = regression_stage(binding, draws=args.draws)
        with open(os.path.join(args.results, "regression.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
        for key, v in out["variants"].items():
            h = v["held_out"]
            print(f"  {key:<20} {v['n_points']:>3} values  slope {v['slope']:+.3f}  error of "
                  f"the mean {h['mean']['mae']:.1f}, of the regression "
                  f"{h['regression']['mae']:.1f} kJ/mol, gain {_interval(v['gain'])}")
    elif not args.report:
        res = run(args.results)
        rep = res["repeatability"]
        print(f"{res['n_rows']} rows from {res['n_studies']} studies; reference system: "
              f"{rep['n']} studies, mean {rep['mean']:.1f}, sd {rep['sd']:.1f} kJ/mol")
        for name, m in res["means"].items():
            print(f"  {name}: mean {m['mean']:.1f} kJ/mol over {m['n']} rows, published "
                  f"{m['published']:g}, difference {m['shift']:+.1f}")
        count = res["solvent_count"]
        print(f"  {count['n_series']} series with {count['n_rows']} rows and "
              f"{len(count['one_solvent'])} single solvents")
    print("the table:", report(args.results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
