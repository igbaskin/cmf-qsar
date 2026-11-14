#!/usr/bin/env python
"""
The diagnosis, run over sets whose answers are already known (item C6).

``cmfqsar/diagnose.py`` says, before any modelling, whether a frame can exist, whether it carries
signal, whether the geometry adds nothing, and which route the set should take.  The way to
check such a thing is to run it on sets this repository has already measured in full and
see whether its advice matches what was found:

* **ACE** -- the descriptor-only baseline beat the 3D model, and the frame gains nothing.  The
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


#: a held-out R2 at or above this from the descriptor route, in a set whose descriptor columns
#: the diagnosis itself found to be reading the activity in another split, is the leak again
LEAK_SCORE = 0.95


def validation_splits(rows):
    """One record per split of the validation: the route advised, what it scored on the
    held-out molecules, the best route and the aligned route, and whether the advice was a
    leak.

    The descriptor route of a file can read the activity from the file's own columns.  The
    diagnosis warns of this when the route reaches q2 0.95 on the molecules it is given, and
    such a route is then left out of the comparison (`unusable`).  On 70 % of a small set the
    same columns can score a little below 0.95.  Then the warning is not given, descriptors are
    advised, and the held-out score near 1 is the leak and not a success of the advice.  A
    split is marked `leak` when descriptors were advised, scored `LEAK_SCORE` or more on the
    held-out molecules, and the diagnosis found the descriptor route of the same set unusable
    in another split.
    """
    out = []
    for r in rows:
        known = any("descriptors" in sp.get("unusable", ()) for sp in r["splits"])
        for sp in r["splits"]:
            bad = set(sp.get("unusable", ()))
            held = {k: v for k, v in sp["held_out"].items() if v == v and k not in bad}
            if not held:
                continue
            best = max(held, key=held.get)
            adv = sp["advice"] if sp["advice"] in held else "aligned"
            out.append({"set": r["set"], "seed": sp["seed"], "advised": adv, "got": held[adv],
                        "best": best, "best_score": held[best],
                        "aligned": held.get("aligned", float("nan")),
                        "leak": bool(known and adv == "descriptors"
                                     and held[adv] >= LEAK_SCORE)})
    return out


def advice_summary(splits, runs=5000, seed=0):
    """What following the advice was worth over a list of :func:`validation_splits` records.

    The count of splits in which the advised route was the best one and the mean cost against
    the best route, the same for the rule "always take the aligned model", and the mean over
    the sets of the advised score minus the aligned score, with a bootstrap interval over the
    sets (three splits of one set are not three measurements).
    """
    if not splits:
        return {}
    cost = np.array([s["best_score"] - s["got"] for s in splits])
    cost_aligned = np.array([s["best_score"] - s["aligned"] for s in splits])
    per_set = {}
    for s in splits:
        per_set.setdefault(s["set"], []).append(s["got"] - s["aligned"])
    gain = np.array([np.mean(v) for v in per_set.values()])
    rng = np.random.RandomState(seed)
    draws = [float(np.mean(rng.choice(gain, len(gain)))) for _ in range(int(runs))]
    return {"splits": len(splits), "sets": len(gain),
            "advice_best": int(sum(s["advised"] == s["best"] for s in splits)),
            "advice_cost": float(cost.mean()), "advice_worst": float(cost.max()),
            "aligned_best": int(sum(s["best"] == "aligned" for s in splits)),
            "aligned_cost": float(cost_aligned.mean()),
            "gain": float(gain.mean()), "gain_low": float(np.percentile(draws, 2.5)),
            "gain_high": float(np.percentile(draws, 97.5)),
            "sets_better": int(np.sum(gain > 0.02)), "sets_worse": int(np.sum(gain < -0.02))}


def report_validation(rows, results=RESULTS, out=None):
    """How often the advice was the right route, and what it cost when it was not."""
    lines = ["# Following the advice: was it right?", "",
             "The diagnosis is run on the training part alone, so the advice never sees the",
             "molecules it is judged on; every route is then fitted on that part and asked about",
             "the rest. `advice` is the route recommended, `best` the route that actually won.",
             "", "| set | split | advised | R2pred advised | best route | R2pred best | cost |",
             "|---|---:|---|---:|---|---:|---:|"]
    splits = validation_splits(rows)
    for s in splits:
        lines.append(f"| {s['set']} | {s['seed']} | {s['advised']}{' (leak)' if s['leak'] else ''} | "
                     f"{s['got']:+.3f} | {s['best']} | {s['best_score']:+.3f} | "
                     f"{s['best_score'] - s['got']:+.3f} |")
    if splits:
        everything = advice_summary(splits)
        lines += ["", f"The advice named the best route on **{everything['advice_best']} of "
                      f"{everything['splits']}** splits; over all splits it cost",
                  f"{everything['advice_cost']:+.3f} in R2pred on average against the best route "
                  f"(worst {everything['advice_worst']:+.3f}).", ""]
        leaks = [s for s in splits if s["leak"]]
        clean = advice_summary([s for s in splits if not s["leak"]])
        lines += ["## The same count without the leak", "",
                  f"In {len(leaks)} of these splits the route advised was `descriptors` in a set "
                  "whose descriptor columns read the activity: the diagnosis found that out in "
                  "another split of the same set, and in this one the route scored a little below "
                  "the 0.95 at which the warning is given. Its held-out score near 1 is the leak, "
                  "and it is not a success of the advice. They are marked `(leak)` above.", "",
                  "| | splits | sets | advice names the best route | mean cost of the advice | "
                  "the aligned model is the best route | mean cost of always aligning | "
                  "advice minus aligned, mean over the sets [95 %] | sets better / worse |",
                  "|---|---:|---:|---:|---:|---:|---:|---|---|"]
        for label, s in (("all splits", everything), ("without the leak splits", clean)):
            lines.append(f"| {label} | {s['splits']} | {s['sets']} | {s['advice_best']} | "
                         f"{s['advice_cost']:.3f} | {s['aligned_best']} | {s['aligned_cost']:.3f} | "
                         f"{s['gain']:+.3f} [{s['gain_low']:+.3f}, {s['gain_high']:+.3f}] | "
                         f"{s['sets_better']} / {s['sets_worse']} |")
        lines += ["", "A set counts as better or worse when the mean difference over its splits is "
                      "more than 0.02 in R2pred.", ""]
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
        # the validation writes <set>_validate.json beside <set>.json; it is a different table
        if fn.endswith(".json") and not fn.endswith("_validate.json"):
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
