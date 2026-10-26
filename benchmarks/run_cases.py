#!/usr/bin/env python
"""
Three series worked through with the interpretation tools, and what they say (plan item F1).

``examples/case_studies.py`` is the script a reader runs; this is the part that keeps the numbers,
so that the case-study sections of the manual can be checked against them and rebuilt from them.
It runs the same three cases and records every figure they produce.

The three are free-energy benchmark sets that ship with the crystal structure their ligands were
posed in -- which is what lets the complementarity step happen at all -- chosen by measuring every
set in ``examples/openff`` and ``examples/schindler`` that has a pocket and taking the three best
models among different target families.

Usage
-----
    python benchmarks/run_cases.py
    python benchmarks/run_cases.py --case mcl1 --force
    python benchmarks/run_cases.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "examples"))

import case_studies as CS  # noqa: E402

RESULTS = os.path.join(HERE, "results", "cases")


def run_case(key: str, force: bool = False) -> dict:
    out_dir = os.path.join(RESULTS, key)
    path = os.path.join(out_dir, f"{key}.json")
    if os.path.exists(path) and not force:
        print(f"{key}: already done")
        return json.load(open(path))
    # no HTML report here: it is a megabyte of inlined images and belongs beside the user's own
    # run, not in the result files the tables are rebuilt from
    return CS.run_case(CS.case_of(key), out_dir, want_report=False)


def report(results=RESULTS):
    files = sorted(glob.glob(os.path.join(results, "*", "*.json")))
    data = [json.load(open(f)) for f in files]
    data = [d for d in data if "summary" in d]
    if not data:
        return "No case studies yet; run `python benchmarks/run_cases.py`."
    order = {c.key: i for i, c in enumerate(CS.CASES)}
    data.sort(key=lambda d: order.get(d["summary"]["case"], 99))
    lines = ["# Three series, worked through", "",
             "Free-energy benchmark sets that ship with the crystal structure their ligands were "
             "posed in, chosen by measuring every set in `examples/openff` and "
             "`examples/schindler` that has a pocket and taking the best models among different "
             "target families. The model whose co-fields are read is fitted on the whole series; "
             "the attenuation and the ridge were chosen on two thirds of it and the other third "
             "-- every third molecule by activity rank -- held out.", "",
             "| case | target | n | activity range | α | q²cv | R²pred (held out) | "
             "y-randomised q², worst of 30 | p |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for d in data:
        s = d["summary"]
        lines.append(f"| {s['case']} | {s['target']} | {s['n']} | {s['activity_range']:.2f} | "
                     f"{s['alpha']:g} | {s['q2_cv']:+.3f} | {s['r2_pred']:+.3f} | "
                     f"{s['y_random_max_q2']:+.3f} | {s['y_random_p']:.3f} |")
    weak = [d for d in data if d["summary"]["y_random_max_q2"] >= d["summary"]["q2_cv"]]
    if weak:
        names = ", ".join(d["summary"]["case"] for d in weak)
        lines += ["", f"**{names}: a shuffle of the activities did as well as the activities.** "
                      "On that set the interpretation below is of noise, and the case is kept for "
                      "that reason: it is what the check is for.", ""]

    lines += ["## Which fields the series needs", "",
              "Each field removed in turn and the model refitted with the same attenuation and "
              "ridge: the number is how much cross-validated q² the removal costs, so a large "
              "positive value means the series needs that field and a negative one means the "
              "model is better without it. A kernel norm cannot answer this, because the field "
              "kernels are scaled to unit mean diagonal before they are mixed and then all five "
              "look equally large.", "",
              "| case | " + " | ".join(CS.FIELDS) + " | needs most |",
              "|---|" + "---:|" * len(CS.FIELDS) + "---|"]
    for d in data:
        w = d["fields"]["without"]
        cells = [f"{w[f]['drop']:+.3f}" if f in w else "—" for f in CS.FIELDS]
        lines.append(f"| {d['summary']['case']} | " + " | ".join(cells)
                     + f" | {d['fields']['most_needed']} |")
    lines += ["",
              "Every one of the three can lose at least one field without losing accuracy. "
              + "; ".join(
                  f"**{d['summary']['case']}** needs {d['fields']['most_needed']} most, and "
                  + ", ".join(f"{f} ({w[f]['drop']:+.3f})" for f in CS.FIELDS
                              if (w := d["fields"]["without"]) and w[f]["drop"] < 0)
                  + " can be removed with no loss"
                  for d in data) + ".",
              "",
              "The first of those is worth stopping on: a phosphatase site is charged, and the "
              "electrostatic field is the one field this series does not need.", ""]

    lines += ["## What the co-fields are made of", "",
              "The share of the grid points carrying any field at all where the co-field is more "
              "than two standard deviations from zero -- that is, the part of the map that the "
              "activities' own noise could not have produced (`cofield_with_error`).", "",
              "| case | " + " | ".join(CS.FIELDS) + " | noise level |",
              "|---|" + "---:|" * (len(CS.FIELDS) + 1)]
    for d in data:
        pf = d["cofields"]["per_field"]
        cells = [f"{pf[f]['share_above_2sd']:.0%}" if f in pf else "—" for f in CS.FIELDS]
        lines.append(f"| {d['summary']['case']} | " + " | ".join(cells)
                     + f" | {d['cofields']['noise_level']:.3f} |")

    lines += ["", "## What the pocket knows", "",
              "Field complementarity with the crystal pocket, and the Spearman correlation of "
              "that score with the measured activity. `burial` is the steric overlap alone: how "
              "deep in the site the ligand sits.", "",
              "| case | pocket atoms | complementarity | burial | clashes | ρ(score, activity) | "
              "ρ(burial, activity) |", "|---|---:|---:|---:|---:|---:|---:|"]
    for d in data:
        p = d["pocket"]
        lines.append(f"| {d['summary']['case']} | {p['pocket_atoms']} | "
                     f"{p['mean_score']:+.3f} | {p['mean_burial']:.3f} | "
                     f"{p['mean_clashes']:.1f} | {p['spearman_with_activity']:+.3f} | "
                     f"{p['spearman_burial_activity']:+.3f} |")
    best = max(data, key=lambda d: abs(d["pocket"]["spearman_burial_activity"]))
    lines += ["",
              f"The strongest of those is **{best['summary']['case']}**, where how deeply a "
              f"ligand sits in the site correlates with its activity at "
              f"{best['pocket']['spearman_burial_activity']:+.3f} while the full "
              f"complementarity score reaches only "
              f"{best['pocket']['spearman_with_activity']:+.3f}. A score that mixes five terms "
              "can be worse than one of them.", ""]

    lines += ["## Where a prediction comes from", "",
              "The Shapley decomposition of the most active molecule of each series, by rings and "
              "the connected pieces between them. The values add up to the prediction minus the "
              "prediction of the bare scaffold, so they are in log units of the activity.", "",
              "| case | molecule | measured | predicted | the three largest contributions |",
              "|---|---|---:|---:|---|"]
    for d in data:
        sh = d["shapley"].get("most active")
        if not sh:
            continue
        top = ", ".join(f"{t['group']} {t['value']:+.2f}" for t in sh["top"][:3])
        lines.append(f"| {d['summary']['case']} | {sh['molecule']} | {sh['activity']:+.2f} | "
                     f"{sh['prediction']:+.2f} | {top} |")

    lines += ["", "## What the weakest molecule is missing", "",
              "The field difference between the least and the most active molecule of each "
              "series, as typed lobes. The weights are integrals over the lobe and are not "
              "comparable between cases, only within one.", "",
              "| case | weak | strong | gap | the largest lobes |", "|---|---|---|---:|---|"]
    for d in data:
        df = d["difference"]
        feats = ", ".join(f"{f['kind']} {abs(f['importance']):.1f}" for f in df["features"][:4])
        lines.append(f"| {d['summary']['case']} | {df['weak']} | {df['strong']} | "
                     f"{df['gap']:+.2f} | {feats} |")
    lines += ["",
              "What is **not** here is a chemist's verdict on any of it. The plan item asks for "
              "these cases to be judged by a medicinal chemist, and that has not been done: what "
              "this file holds is every number such a judgement would need.", ""]
    text = "\n".join(lines) + "\n"
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs="*", default=[c.key for c in CS.CASES],
                    choices=[c.key for c in CS.CASES])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for key in a.case:
        run_case(key, a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
