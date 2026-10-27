#!/usr/bin/env python
"""
Choosing ten compounds at once: does the covariance beat a similarity filter? (plan item MC2)

A laboratory makes ten or twenty compounds at a time, not one.  The acquisition rules measured
by `benchmarks/run_active.py` pick one molecule and refit; a batch has to be chosen before any
of it is measured, and the only thing that stops it from being one molecule ten times over is
whatever diversity rule is used.

`cmfqsar/active.py` has had the cheap rule from the start: take the best scores in order and
skip a candidate whose cosine with a pick already made is above a threshold.  Plan item MC2 asks
for the rule the posterior already implies.  After a pick, the posterior covariance conditioned
on it says how much uncertainty is left everywhere else -- one Cholesky step per pick, no
refitting and no threshold -- and the score is recomputed from that.  With the variance
acquisition this is the batch whose own covariance block has the largest determinant, which is
the classical D-optimal design.

How it is measured
------------------
* The protocol of `benchmarks/run_active.py`: the eight Sutherland sets, twenty molecules to
  start, the attenuation and the ridge reselected by the evidence from what has been acquired,
  and the same starting set for every rule so that the comparison is paired.
* **Batches of ten**, four of them, so forty molecules are acquired in four fits rather than
  forty.
* **E2's two measures**: the best activity acquired, and the error of the model on everything
  not yet acquired.
* **One at a time** is run as well, on the same budget. It is what a batch approximates and it
  costs forty fits instead of four, so it is the ceiling rather than a rival.

Usage
-----
    python benchmarks/run_batch.py
    python benchmarks/run_batch.py --sets GPB THER --repeats 4
    python benchmarks/run_batch.py --report
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

from rdkit import RDLogger  # noqa: E402

from cmfqsar.active import learning_curve  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402

from run_active import BUDGET, DATA, FIELDS, LAMS, SETS, START, STEPS, TOP, kernels  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "batch")
BATCH = 10
#: the share of the pool that is acquired, capped by run_active's own budget of 40 molecules
SHARE = 0.5
#: (name, acquisition, batch size, batch rule)
WAYS = (
    ("one at a time", "ei", 1, "cosine"),
    ("ei, similarity filter", "ei", BATCH, "cosine"),
    ("ei, covariance", "ei", BATCH, "covariance"),
    ("variance, similarity filter", "variance", BATCH, "cosine"),
    ("variance, covariance (largest determinant)", "variance", BATCH, "covariance"),
    ("random", "random", BATCH, "cosine"),
)


def _say(text):
    print(text, flush=True)


def run_set(name, repeats=10, force=False, data=DATA, results=RESULTS, batch=BATCH):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        _say(f"{name}: already done")
        with io.open(out_path, encoding="utf-8") as fh:
            return json.load(fh)
    t0 = time.time()
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    n = len(y)
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, nm) for m, nm in zip(mols, names)])
    Ks = kernels(table)
    n_hit = max(1, int(np.ceil(TOP * n)))
    hits = set(np.argsort(-y)[:n_hit].tolist())
    # a larger share of the pool than run_active takes, because a batch run has to fit several
    # batches in and still leave something to measure the error on
    budget = int(min(STEPS, max(batch, SHARE * (n - START))))
    budget = int(round(budget / batch)) * batch or batch
    _say(f"=== {name}: {n} molecules, {budget} acquired from {START} in batches of {batch}")

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(1000 + rep)
        start = sorted(rng.choice(n, START, replace=False).tolist())
        if set(start) & hits:
            start = sorted(rng.choice(sorted(set(range(n)) - hits), START,
                                      replace=False).tolist())
        for label, kind, size, rule in WAYS:
            steps = budget // size
            out = learning_curve(Ks, y, kind, start, steps=steps, lams=LAMS, seed=rep,
                                 batch=size, batch_rule=rule)
            taken = []
            for t in out["taken"]:
                taken.extend(t if isinstance(t, list) else [t])
            first = next((i + 1 for i, t in enumerate(taken) if t in hits), None)
            rows.append({"repeat": rep, "way": label, "kind": kind, "batch": size, "rule": rule,
                         "fits": steps,
                         "rmse_mean": float(np.mean(out["rmse"])),
                         "rmse_final": float(out.get("rmse_end", out["rmse"][-1])),
                         "best": float(np.max(y[out["have"]])),
                         "steps_to_hit": first, "hit": bool(first is not None),
                         "acquired": len(taken)})
        done = {r["way"]: r for r in rows if r["repeat"] == rep}
        _say("  repeat %d: " % rep + "  ".join(
            f"{k.split(',')[0]} {done[k]['best']:.2f}/{done[k]['rmse_final']:.3f}"
            for k in done))

    out = {"set": name, "n": int(n), "repeats": int(repeats), "start": START,
           "budget": budget, "batch": batch, "n_hit": n_hit,
           "best_possible": float(np.max(y)), "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    with io.open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def run(sets=SETS, repeats=10, force=False, results=RESULTS, batch=BATCH):
    return [run_set(s, repeats=repeats, force=force, results=results, batch=batch) for s in sets]


def _load(results=RESULTS):
    out = []
    for s in SETS:
        path = os.path.join(results, f"{s}.json")
        if os.path.exists(path):
            with io.open(path, encoding="utf-8") as fh:
                out.append(json.load(fh))
    return out


def _paired(data, a, b, key, runs=2000, seed=0):
    """(the mean difference between two ways, the interval), paired by set and repeat."""
    diff = []
    for d in data:
        by = {}
        for r in d["rows"]:
            by.setdefault(r["repeat"], {})[r["way"]] = r
        for rep in by:
            if a in by[rep] and b in by[rep]:
                x, z = by[rep][a][key], by[rep][b][key]
                if x is not None and z is not None:
                    diff.append(float(x) - float(z))
    diff = np.asarray(diff, float)
    if len(diff) < 3:
        return (float("nan"), float("nan"), float("nan"), 0)
    rng = np.random.RandomState(int(seed))
    boot = [float(np.mean(diff[rng.randint(0, len(diff), len(diff))])) for _ in range(int(runs))]
    return (float(diff.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)), int(len(diff)))


def _mean(data, way, key):
    v = [r[key] for d in data for r in d["rows"] if r["way"] == way and r[key] is not None]
    return float(np.mean(v)) if v else float("nan")


def report(results=RESULTS):
    data = _load(results)
    if not data:
        raise SystemExit("nothing to report: run the benchmark first")
    ways = [w[0] for w in WAYS]
    out = ["# Choosing ten compounds at once (plan item MC2)", "",
           "A laboratory makes ten compounds at a time. A batch has to be chosen before any of "
           "it is measured, so something has to keep it from being one molecule ten times over. "
           "The cheap rule skips a candidate whose cosine with a pick already made is above a "
           "threshold. The rule this item asks for conditions the posterior covariance on each "
           "pick instead -- one Cholesky step, no refitting, no threshold -- and recomputes the "
           "score from it; with the variance acquisition that is the batch whose covariance "
           "block has the largest determinant.", "",
           f"The protocol of `benchmarks/run_active.py`: {len(data)} Sutherland sets, "
           f"{data[0]['start']} molecules to start, the attenuation and the ridge reselected by "
           f"the evidence at every fit, {data[0]['repeats']} repeats with the same starting set "
           f"for every rule. Batches of {data[0]['batch']}.", "",
           "| how the batch is chosen | fits | best acquired | error on what is left |",
           "| --- | ---: | ---: | ---: |"]
    for w in ways:
        fits = [r["fits"] for d in data for r in d["rows"] if r["way"] == w]
        out.append(f"| {w} | {int(np.mean(fits)) if fits else 0} | {_mean(data, w, 'best'):.3f} "
                   f"| {_mean(data, w, 'rmse_final'):.3f} |")
    out += ["", "## The gate", "",
            "The item asks whether the covariance rule beats the similarity filter in **both** "
            "of E2's measures by more than the interval, paired run by run.", "",
            "| acquisition | measure | covariance minus filter |", "| --- | --- | ---: |"]
    pairs = []
    for kind, a, b in (("expected improvement", "ei, covariance", "ei, similarity filter"),
                       ("variance", "variance, covariance (largest determinant)",
                        "variance, similarity filter")):
        for key, label, better in (("best", "best acquired", +1),
                                   ("rmse_final", "error on what is left", -1)):
            d = _paired(data, a, b, key)
            pairs.append((kind, key, d, better))
            out.append(f"| {kind} | {label} | {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] |")
    out += [""] + _verdict(data, pairs)
    return "\n".join(out)


def _verdict(data, pairs):
    def wins(kind, key, d, better):
        if not np.isfinite(d[1]):
            return False
        return d[1] > 0 if better > 0 else d[2] < 0

    won = [p for p in pairs if wins(*p)]
    lost = [p for p in pairs if (np.isfinite(p[2][2]) and
                                 (p[2][2] < 0 if p[3] > 0 else p[2][1] > 0))]
    out = ["## The verdict", ""]
    if len(won) == len(pairs):
        out += ["**The gate is met.** The covariance rule beats the similarity filter in both "
                "measures, for both acquisitions, by more than the interval."]
    elif won:
        out += [f"**The gate is not met.** The covariance rule is ahead by more than the "
                f"interval in {len(won)} of the {len(pairs)} comparisons and behind in "
                f"{len(lost)}; the rest contain zero. The item asked for both measures and both "
                f"acquisitions."]
    else:
        out += [f"**The gate is not met: the two rules are the same within the interval.** "
                f"None of the {len(pairs)} comparisons separates them"
                + (f", and the filter is ahead in {len(lost)}." if lost else ".")
                + " Plan item E2 predicted this, and the item itself allows it: the cheap "
                  "filter is enough, and that closes the question rather than leaving it open."]
    best_b = _paired(data, "ei, similarity filter", "one at a time", "best")
    rmse_b = _paired(data, "ei, similarity filter", "one at a time", "rmse_final")
    costs = (np.isfinite(best_b[2]) and best_b[2] < 0) or (np.isfinite(rmse_b[1])
                                                           and rmse_b[1] > 0)
    out += ["",
            f"**What a batch costs against one at a time.** Acquiring the same molecules in "
            f"batches of {data[0]['batch']} needs a tenth of the fits. Against the "
            f"one-at-a-time loop, which sees every measurement before choosing the next "
            f"molecule, expected improvement with the similarity filter differs by "
            f"{best_b[0]:+.3f} [{best_b[1]:+.3f}, {best_b[2]:+.3f}] in the best acquired and "
            f"{rmse_b[0]:+.3f} [{rmse_b[1]:+.3f}, {rmse_b[2]:+.3f}] in the error on what is "
            f"left. "
            + ("That is a cost, and it is what a batch buys its ten-fold saving in fits with."
               if costs else
               "Neither interval excludes zero: on these sets a plate of ten chosen at once is "
               "worth as much as ten chosen one after another, at a tenth of the fitting. That "
               "is the useful finding here, and it holds whichever of the two batch rules is "
               "used."), "",
            "**Why the covariance rule has so little to add** is what E2 already found about "
            "the posterior standard deviation on these sets: it is nearly flat over the pool, "
            "so conditioning on a pick moves the ordering very little, and the cosine filter "
            "removes the near-duplicates that would otherwise be picked twice. Both rules end "
            "up doing the same thing, by different arithmetic.", ""]
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
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    run(sets=a.sets, repeats=a.repeats, force=a.force, batch=a.batch)
    print()
    print(write_table())


if __name__ == "__main__":
    main()
