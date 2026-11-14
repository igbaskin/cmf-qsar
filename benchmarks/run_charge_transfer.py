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

    python benchmarks/run_charge_transfer.py            # the summary and the table
    python benchmarks/run_charge_transfer.py --report   # the table from the stored summary
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "examples", "charge_transfer")
RESULTS = os.path.join(HERE, "results", "charge_transfer")
CELLMODEL = os.path.join(HERE, "results", "cellmodel")

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


def table(res: dict, sources: Dict[str, dict]) -> str:
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


def report(results: str = RESULTS) -> str:
    path = os.path.join(results, "summary.json")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run this benchmark without --report first")
    with open(path, encoding="utf-8") as fh:
        res = json.load(fh)
    out = os.path.join(results, "table.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(table(res, read_sources()))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--report", action="store_true",
                    help="write the table from the stored summary and stop")
    ap.add_argument("--results", default=RESULTS, help="where the result is written")
    args = ap.parse_args(argv)
    if not args.report:
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
