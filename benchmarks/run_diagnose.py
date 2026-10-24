#!/usr/bin/env python
"""
The diagnosis, run over sets whose answers are already known (item C6).

``cmfqsar/diagnose.py`` says, before any modelling, whether a frame can exist, whether it carries
signal, whether the geometry is decoration, and which route the set should take.  The way to
check such a thing is to point it at sets this repository has already measured the hard way and
see whether its advice matches what was found:

* **ACE** -- the descriptor-only baseline beat the 3D model, and the frame buys nothing.  The
  diagnosis should say so.
* **Am/Eu** -- a curated frame worth +0.807 against -0.113 for atom counts.  It should say the
  opposite.
* **the corrosion series** -- two-dimensional structures, so every frame is invented, and counts
  beat the fields on the benzenes.  It should warn that the molecules are not in a common frame.
* **the remaining Sutherland sets** -- a mixture, and the advice is the output being judged.

Usage
-----
    python benchmarks/run_diagnose.py [--family sutherland|corrosion|am_eu|all]
    python benchmarks/run_diagnose.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import RDLogger  # noqa: E402

import run_topomer as T  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.diagnose import diagnose, validate_advice  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "diagnose")
SUTHERLAND = os.path.join(ROOT, "examples", "sutherland")
CORROSION = os.path.join(ROOT, "examples", "corrosion")
CORROSION_SETS = ["0_c1ccccc1", "7_c1nncs1", "13_c1c_nH_nn1", "161_c1ccc2scnc2c1"]


def cases(family):
    """(name, molecules, names, activities, charge model, parameterisation) per set."""
    out = []
    if family in ("sutherland", "all"):
        for name in T.SETS:
            ds = load_dataset(os.path.join(SUTHERLAND, f"{name}.sdf"))
            y = np.asarray(ds.activity, float)
            out.append((name, ds.mols, ds.names, y, "file", "rdkit", ds.activity_prop))
    if family in ("corrosion", "all"):
        import run_corrosion as C
        for name in CORROSION_SETS:
            path = os.path.join(CORROSION, f"{name}.csv")
            ds = load_dataset(path)
            y = np.asarray(ds.activity, float)
            ok = np.isfinite(y)
            mols = [m for m, g in zip(ds.mols, ok) if g]
            nm = [n for n, g in zip(ds.names, ok) if g]
            core = C.core_of(mols, path)
            keep = [i for i, m in enumerate(mols) if m.GetSubstructMatch(core)]
            out.append((f"corrosion/{name}", [mols[i] for i in keep], [nm[i] for i in keep],
                        y[ok][keep], "gasteiger", "rdkit", ds.activity_prop))
    if family in ("am_eu", "all"):
        import run_am_eu as A
        heavy, _withh, names, y = A.prepare()
        out.append(("am_eu", heavy, names, y, "file", "tripos", ""))
    return out


def run_family(family="all", results=RESULTS, force=False, folds=10, seed=0):
    os.makedirs(results, exist_ok=True)
    rows = []
    for name, mols, names, y, charges, params, activity_prop in cases(family):
        out_path = os.path.join(results, name.replace("/", "_") + ".json")
        if os.path.exists(out_path) and not force:
            print(f"[skip] {out_path} exists")
            rows.append(json.load(open(out_path)))
            continue
        t0 = time.time()
        print(f"\n=== {name}: {len(mols)} molecules", flush=True)
        d = diagnose(mols, names, y, folds=folds, seed=seed, charge_model=charges,
                     param_scheme=params, activity_prop=activity_prop)
        print(d.report(), flush=True)
        rec = d.to_dict()
        rec["set"] = name
        rec["seconds"] = time.time() - t0
        with open(out_path, "w") as fh:
            json.dump(rec, fh, indent=1)
        rows.append(rec)
    return rows


def validate(family="all", results=RESULTS, seeds=(0, 1, 2), force=False):
    """Follow the advice on held-out molecules: the measurement that judges the recommendation."""
    os.makedirs(results, exist_ok=True)
    rows = []
    for name, mols, names, y, charges, params, activity_prop in cases(family):
        out_path = os.path.join(results, name.replace("/", "_") + "_validate.json")
        if os.path.exists(out_path) and not force:
            rows.append(json.load(open(out_path)))
            continue
        print(f"=== {name}: following the advice on {len(mols)} molecules", flush=True)
        t0 = time.time()
        rec = validate_advice(mols, names, y, seeds=seeds, charge_model=charges,
                              param_scheme=params, activity_prop=activity_prop,
                              progress=lambda m: print(f"    {m}", flush=True))
        rec["set"] = name
        rec["seconds"] = time.time() - t0
        with open(out_path, "w") as fh:
            json.dump(rec, fh, indent=1)
        rows.append(rec)
    report_validation(rows, results)
    return rows


def report_validation(rows, results=RESULTS, out=None):
    """How often the advice was the right road, and what it cost when it was not."""
    lines = ["# Following the advice: was it right?", "",
             "The diagnosis is run on the training part alone, so the advice never sees the",
             "molecules it is judged on; every route is then fitted on that part and asked about",
             "the rest. `advice` is the route recommended, `best` the route that actually won.",
             "", "| set | split | advised | R2pred advised | best route | R2pred best | cost |",
             "|---|---:|---|---:|---|---:|---:|"]
    right = total = 0
    costs = []
    for r in rows:
        for sp in r["splits"]:
            bad = set(sp.get("unusable", ()))
            held = {k: v for k, v in sp["held_out"].items() if v == v and k not in bad}
            if not held:
                continue
            best = max(held, key=held.get)
            adv = sp["advice"] if sp["advice"] in held else "aligned"
            got = held.get(adv, float("nan"))
            cost = held[best] - got
            total += 1
            right += int(adv == best)
            costs.append(cost)
            lines.append(f"| {r['set']} | {sp['seed']} | {adv} | {got:+.3f} | {best} | "
                         f"{held[best]:+.3f} | {cost:+.3f} |")
    if total:
        lines += ["", f"The advice named the best route on **{right} of {total}** splits; when it",
                  f"did not, it cost {np.mean(costs):+.3f} in R2pred on average "
                  f"(worst {max(costs):+.3f}).", ""]
    text = chr(10).join(lines) + chr(10)
    path = out or os.path.join(results, "advice.md")
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written {path}")
    return text


def report(results=RESULTS, out=None):
    rows = []
    for fn in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(results, fn))))
    if not rows:
        raise SystemExit(f"no results in {results}")
    rows.sort(key=lambda r: -r["n"])
    order = ["aligned", "counts", "invariant", "topomer", "descriptors"]
    lines = ["# What the diagnosis says, on sets whose answers are known", "",
             "Each route scored by the same 10-fold cross-validation; the recommendation reads",
             "the paired bootstrap intervals, not the point estimates. The warnings column counts",
             "the questions the set failed.", "",
             "| set | n | " + " | ".join(order) + " | warnings | recommended |",
             "|---|---:|" + "---:|" * len(order) + "---:|---|"]
    for r in rows:
        cells = []
        for k in order:
            v = r["routes"].get(k, {})
            q = v.get("q2")
            cells.append("-" if q is None or not np.isfinite(q) else f"{q:+.3f}")
        warn = sum(1 for f in r["findings"] if f["level"] == "warn")
        lines.append(f"| {r['set']} | {r['n']} | " + " | ".join(cells)
                     + f" | {warn} | {r['recommended']} |")
    text = "\n".join(lines) + "\n"
    path = out or os.path.join(results, "table.md")
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written {path}")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--family", choices=["sutherland", "corrosion", "am_eu", "all"],
                    default="all")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--validate", action="store_true",
                    help="follow the advice on held-out molecules and score it")
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args(argv)
    if args.validate:
        return validate(args.family, seeds=tuple(args.seeds), force=args.force)
    if args.report:
        return report()
    run_family(args.family, force=args.force, folds=args.folds, seed=args.seed)
    report()


if __name__ == "__main__":
    main()
