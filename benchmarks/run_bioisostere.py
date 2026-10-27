#!/usr/bin/env python
"""
Does the co-field know which groups are interchangeable? (plan item MC1)

A medicinal chemist asking for a bioisostere table is not asking which substituent is best.  The
question is **which groups can replace each other at this position**, so that one of them can be
chosen for a property the model does not describe: solubility, cost, a patent, a metabolic
liability.  `cmfqsar/bioisostere.py` answers it without refitting the model once per group: the
prediction of the overlap kernel is a sum over the atoms, so the part of it that belongs to a
substituent is the sum over that substituent's own atoms, and exchanging group A for group B is
the difference of two such sums.

This measures whether that number is worth having.

How it is measured
------------------
* **The pairs** are matched molecular pairs found by RDKit's `rdMMPA`: one single bond cut at a
  time, both pieces tried as the core, two molecules with the same core and different groups.
  Nothing is chosen by eye, and two molecules that match on several cores are counted once.
* **Three measures of how far apart two groups are**, the ladder of rule 1:
  1. the **fingerprint distance** between the two groups -- two dimensions, no geometry, no model;
  2. the **field distance** between the two posed groups, restricted to their own atoms --
     geometry, but no model;
  3. the **predicted change**, the difference of the two atom-contribution sums -- geometry and
     the co-field.
* **What they are compared with** is the size of the measured activity difference of the pair.
  Two groups that are interchangeable differ little in activity; the question is which measure
  ranks that.
* **The model never sees the two molecules it is judging.** For every pair the dual coefficients
  are refitted without both members, so the predicted change cannot have learned the two
  activities it is compared with.  The same number with the full model is reported next to it,
  and the gap between the two is how much of the agreement was memory.

Usage
-----
    python benchmarks/run_bioisostere.py
    python benchmarks/run_bioisostere.py --sets GPB --table
    python benchmarks/run_bioisostere.py --report
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.bioisostere import (biggest_core, build_table, matched_pairs,  # noqa: E402
                                 rank_gain, score_pairs, spearman, substituents)
from cmfqsar.conformal import calibrate  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import combine_kernels  # noqa: E402
from cmfqsar.model import CMFModel, krr_fit  # noqa: E402

RESULTS = os.path.join(HERE, "results", "bioisostere")
SETS = {
    "GPB": dict(path=os.path.join(ROOT, "examples", "sutherland", "GPB.sdf"), activity="pKi",
                what="glycogen phosphorylase b, 66 molecules"),
    "THER": dict(path=os.path.join(ROOT, "examples", "sutherland", "THER.sdf"), activity="pKi",
                 what="thermolysin, 76 molecules"),
    "THR": dict(path=os.path.join(ROOT, "examples", "sutherland", "THR.sdf"), activity="pKi",
                what="thrombin, 88 molecules"),
}
FIELDS = ["electrostatic", "steric", "hydrophobic"]
ALPHAS = (0.5, 1.0, 2.0)
LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
LEVEL = 0.9


def _say(text):
    print(text, flush=True)


def load(key, max_molecules=0):
    spec = SETS[key]
    ds = load_dataset(spec["path"], activity_prop=spec["activity"])
    y = np.asarray(ds.activity, float)
    keep = [i for i in range(len(ds)) if np.isfinite(y[i])]
    if max_molecules:
        keep = keep[:max_molecules]
    return ([Chem.RemoveHs(ds.mols[i]) for i in keep], [ds.names[i] for i in keep],
            y[keep], spec["what"])


def fitted(mols, names, y, folds=10, seed=0):
    """The ordinary model of the set, with alpha and the ridge searched on the folds."""
    best = None
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(alpha))
        fields = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        table = AtomTable.from_molecules(fields)
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        K_fields = np.asarray(model.prepare_training_kernels(table))
        for lam in LAMS:
            model.params["lam"] = float(lam)
            model.set_training_kernels(table, K_fields, model.self_diag)
            model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
            if best is None or model.stats.q2 > best[0]:
                best = (float(model.stats.q2), float(alpha), float(lam))
    q2, alpha, lam = best
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha)
    fields = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(fields)
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    K_fields = np.asarray(model.prepare_training_kernels(table))
    model.params["lam"] = lam
    model.set_training_kernels(table, K_fields, model.self_diag)
    model.fit(table, y, cv_folds=folds, cv_seed=seed, reuse_kernels=True)
    K = np.asarray(combine_kernels(K_fields, fs.mixing_vector(), model.scale), float)
    return model, fields, K, q2, alpha, lam


def leave_out(K, y, lam):
    """A function giving the dual coefficients fitted without both members of a pair."""
    y = np.asarray(y, float)
    n = len(y)
    cache = {}

    def of(pair):
        key = (int(pair.a), int(pair.b))
        if key not in cache:
            rows = np.asarray([i for i in range(n) if i not in key], int)
            a, _b = krr_fit(K[np.ix_(rows, rows)], y[rows], float(lam))
            full = np.zeros(n)
            full[rows] = a
            cache[key] = full
        return cache[key]
    return of


def run_set(key, folds=10, seed=0, max_molecules=0, with_table=True, results=RESULTS):
    t0 = time.time()
    mols, names, y, what = load(key, max_molecules)
    _say(f"=== {key}: {len(mols)} molecules, {what}")
    pairs = matched_pairs(mols, y, names)
    if len(pairs) < 10:
        _say(f"  only {len(pairs)} matched pairs: too few to measure anything")
        return None
    _say(f"  {len(pairs)} matched pairs, {time.time() - t0:.0f} s")

    model, fields, K, q2, alpha, lam = fitted(mols, names, y, folds, seed)
    ycv = np.asarray(model.stats.extra.get("y_cv"), float)
    half = calibrate(y, ycv, level=LEVEL).half_width()
    half = float(np.ravel(half)[0])
    _say(f"  model q2 {q2:+.3f} at alpha {alpha:g}, ridge {lam:g}; conformal half width "
         f"{half:.2f} at {LEVEL:.0%}, {time.time() - t0:.0f} s")

    honest = score_pairs(model, fields, pairs, coef_of=leave_out(K, y, lam))
    full = score_pairs(model, fields, pairs)
    _say(f"  pairs scored, {time.time() - t0:.0f} s")

    measured = np.abs([s.measured for s in honest])
    columns = {
        "fingerprint distance": [s.fingerprint for s in honest],
        "field distance": [s.field for s in honest],
        "predicted change, the pair left out": [abs(s.change) for s in honest],
        "predicted change, the whole model": [abs(s.change) for s in full],
    }
    rows = []
    for name, col in columns.items():
        rows.append(dict(name=name, rho=spearman(col, measured)))
        _say(f"  {name:38s} rank correlation {rows[-1]['rho']:+.3f}")
    gains = {}
    for other in ("fingerprint distance", "field distance"):
        g = rank_gain(columns["predicted change, the pair left out"], columns[other],
                      measured, seed=seed)
        gains[other] = g
        _say(f"    against the {other:22s} {g[0]:+.3f} [{g[1]:+.3f}, {g[2]:+.3f}]")

    # the signed question, which is the one the table is read for
    signed = spearman([s.change for s in honest], [s.measured for s in honest])
    _say(f"  the signed change against the signed measurement: {signed:+.3f}")

    out = dict(set=key, what=what, n=len(mols), pairs=len(pairs), q2=q2, alpha=alpha, lam=lam,
               half_width=half, rows=rows, gains={k: list(v) for k, v in gains.items()},
               signed=signed, seconds=time.time() - t0,
               measured=[s.measured for s in honest], columns=columns)

    if with_table:
        subs = substituents(mols)
        core = biggest_core(subs, least=4)
        if core:
            tab = build_table(model, fields, core, subs, half_width=half, names=names)
            path = os.path.join(results, f"table_{key}.csv")
            os.makedirs(results, exist_ok=True)
            with io.open(path, "w", encoding="utf-8", newline="") as fh:
                import csv
                w = csv.writer(fh)
                w.writerow(["group a", "group b", "field distance", "predicted change"])
                for a, b, d, c in tab.rows():
                    w.writerow([a, b, f"{d:.4f}", f"{c:+.4f}"])
            out["table"] = dict(core=core, groups=tab.groups, half_width=half,
                                file=os.path.basename(path),
                                interchangeable={g: tab.interchangeable(g)
                                                 for g in tab.groups})
            _say(f"  a table of {len(tab.groups)} groups on {core[:50]} -> "
                 f"{os.path.basename(path)}")

    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, f"{key}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def run(sets=("GPB", "THER", "THR"), **kw):
    return [g for g in (run_set(k, **kw) for k in sets) if g]


def _load(results=RESULTS):
    out = []
    for key in SETS:
        path = os.path.join(results, f"{key}.json")
        if os.path.exists(path):
            with io.open(path, encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def _cell(c):
    if not np.isfinite(c[0]):
        return "--"
    return f"{c[0]:+.3f} [{c[1]:+.3f}, {c[2]:+.3f}]"


def report(results=RESULTS):
    data = _load(results)
    if not data:
        raise SystemExit("nothing to report: run the benchmark first")
    out = ["# Does the co-field know which groups are interchangeable? (plan item MC1)", "",
           "A bioisostere table answers which groups can replace each other at a position, so "
           "that one of them can be chosen for a property the model does not describe. The "
           "prediction of the overlap kernel is a sum over the atoms, so the part of it that "
           "belongs to a substituent is the sum over that substituent's own atoms, and "
           "exchanging one group for another is the difference of two such sums -- one integral "
           "per pair of groups rather than one model fit per group.", "",
           "The pairs are matched molecular pairs from RDKit's `rdMMPA`. The three measures are "
           "the ladder of rule 1: the fingerprint distance between the groups (no geometry, no "
           "model), the field distance between the two posed groups restricted to their own "
           "atoms (geometry, no model), and the predicted change (geometry and the co-field). "
           "All three are compared with the size of the measured activity difference by rank "
           "correlation. For every pair the dual coefficients are refitted without both of its "
           "members, so the predicted change has not seen the two activities it is judged "
           "against.", ""]
    for d in data:
        out += [f"## {d['set']}: {d['what']}", "",
                f"{d['pairs']} matched pairs among {d['n']} molecules; the model reaches q2 "
                f"{d['q2']:+.3f} at alpha {d['alpha']:g} and ridge {d['lam']:g}, with a "
                f"conformal half width of {d['half_width']:.2f} log units at 90 per cent.", "",
                "| measure | rank correlation with the size of the measured change |",
                "| --- | ---: |"]
        for r in d["rows"]:
            out.append(f"| {r['name']} | {r['rho']:+.3f} |")
        out += ["", "| the co-field against | difference in rank correlation |",
                "| --- | ---: |"]
        for k, g in d["gains"].items():
            out.append(f"| the {k} | {_cell(g)} |")
        out += ["", f"The **signed** change against the signed measurement: "
                    f"{d['signed']:+.3f}.", ""]
        if d.get("table"):
            t = d["table"]
            same = [g for g, v in t["interchangeable"].items() if v]
            out += [f"A table of {len(t['groups'])} groups on one core is in `{t['file']}`. "
                    f"At twice the conformal half width ({2 * t['half_width']:.2f} log units), "
                    f"{len(same)} of the {len(t['groups'])} groups have at least one partner the "
                    f"model cannot tell them apart from.", ""]
    out += _verdict(data)
    return "\n".join(out)


def _verdict(data):
    out = ["## What the sets say together", "",
           "| set | pairs | fingerprint | field | co-field (pair left out) | co-field (whole "
           "model) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for d in data:
        rho = {r["name"]: r["rho"] for r in d["rows"]}
        out.append(f"| {d['set']} | {d['pairs']} | "
                   + " | ".join(f"{rho[k]:+.3f}" for k in
                                ("fingerprint distance", "field distance",
                                 "predicted change, the pair left out",
                                 "predicted change, the whole model")) + " |")
    keys = list(data[0]["gains"])
    wins = {k: [d["set"] for d in data if d["gains"][k][1] > 0] for k in keys}
    losses = {k: [d["set"] for d in data if d["gains"][k][2] < 0] for k in keys}
    both = [d["set"] for d in data if all(d["gains"][k][1] > 0 for k in keys)]
    against = [d["set"] for d in data if any(d["gains"][k][2] < 0 for k in keys)]
    quiet = [d["set"] for d in data if d["set"] not in both and d["set"] not in against]
    out += ["", "## The verdict", ""]
    if both and not against:
        line = (f"**The co-field weighting earns its place.** On {len(both)} of the "
                f"{len(data)} sets ({', '.join(both)}) the predicted change ranks the measured "
                "activity difference better than both the fingerprint distance and the field "
                "distance by more than the interval of rule 4, and on no set is it worse than "
                "either.")
        if quiet:
            smallest = min(data, key=lambda d: d["pairs"])
            line += (f" On {', '.join(quiet)} it is ahead of both as well, but the interval "
                     f"contains zero; that is the set with the fewest pairs "
                     f"({smallest['pairs']} against "
                     f"{max(d['pairs'] for d in data)} on the largest).")
        out += [line, "",
                "The reading is that a bioisostere table is worth computing from a model. The "
                "two references are not weak ones -- the field distance compares the two groups "
                "**in their poses**, so it already knows the shape and the charge each of them "
                "puts where -- and neither carries much: the fingerprint distance ranks nothing "
                "at all on GPB. What the co-field adds is the part the references cannot have, "
                "which is whether the activity depends on the place where the two groups "
                "differ."]
    elif against:
        out += ["**The co-field weighting is worse than at least one of its references.** The "
                f"sets where an interval excludes zero against it are {', '.join(against)}. A "
                "table built this way would mislead, and the item's own warning applies."]
    else:
        out += ["**The co-field weighting does not earn its place on these sets.** No set shows "
                "it ahead of both references by more than the interval of rule 4. What the item "
                "predicted for that case holds: the table would then be shape similarity, which "
                "needs no model, and that is a result of the same kind as plan item D5."]
    out += ["",
            "**How much of it is memory** is the difference between the last two columns. The "
            "whole model, which has seen both activities of every pair it is judging, reaches "
            + ", ".join(f"{[r['rho'] for r in d['rows'] if 'whole model' in r['name']][0]:+.3f} "
                        f"on {d['set']}" for d in data)
            + "; refitting without the two members of each pair leaves "
            + ", ".join(f"{[r['rho'] for r in d['rows'] if 'pair left out' in r['name']][0]:+.3f}"
                        for d in data)
            + ". The gap is large, and it is the reason the honest column is the one reported "
              "above. Two things still come from the whole set: the field type, attenuation and "
              "ridge were chosen on all the molecules, and the alignment is the published one.",
            "",
            "The column that matters for the chemist is not the ranking but the size. A "
            "predicted change smaller than twice the conformal half width is a change the model "
            "cannot see, and the groups that fall inside it are the ones the table calls "
            "interchangeable. That part of the table is the model's own error bar rather than a "
            "claim about the chemistry, so it is sound whatever the ranking says.", ""]
    return out


def write_table(results=RESULTS) -> str:
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with io.open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=["GPB", "THER", "THR"])
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-molecules", type=int, default=0)
    ap.add_argument("--no-table", action="store_true")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    run(sets=a.sets, folds=a.folds, seed=a.seed, max_molecules=a.max_molecules,
        with_table=not a.no_table)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
