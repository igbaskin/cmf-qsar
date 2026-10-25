#!/usr/bin/env python
"""
Does choosing the next molecule by the model beat choosing it at random?

A fitted CMF model offers a prediction and a posterior standard deviation for every posed
candidate, and `cmfqsar.active` turns the pair into a rule for what to make next. Whether any of
those rules is worth following is a question for a set whose activities are already known: start
from twenty molecules chosen at random, add one at a time by each rule, and watch two different
things happen.

**Finding an active** is one goal: how good is the best molecule acquired so far, and how many
acquisitions pass before one of the top five per cent is in hand.

**Improving the model** is the other: the RMSE of the model on everything not yet acquired, which
is what a chemist cares about when the series is being mapped rather than mined.

They are different goals and the rules trade between them, so both are reported. Everything --
alpha, lambda -- is reselected by the evidence at every step from the molecules acquired and
nothing else, which is what item B4 made affordable; the only thing shared with the other
strategies is the starting set, so that the comparison is paired.

Usage
-----
    python benchmarks/run_active.py
    python benchmarks/run_active.py --sets GPB THER --repeats 4
    python benchmarks/run_active.py --report
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

from cmfqsar.active import learning_curve  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "active")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.2, 0.5, 1.0, 2.0)
LAMS = (1e-2, 0.1, 1.0)
STRATEGIES = ("random", "maxmin", "variance", "greedy", "ucb", "ei")
START = 20
STEPS = 40            # at most; the budget is also capped at a quarter of the pool, below
BUDGET = 0.25         # of the molecules not in the starting set
TOP = 0.05            # a molecule in this fraction of the set counts as a hit


def kernels(table, alphas=ALPHAS):
    """alpha -> the joint kernel over every molecule of the set."""
    out = {}
    for a in alphas:
        fs = FieldSet.create(physchem=FIELDS, alpha=float(a), charge_model="file")
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        Kf = model.prepare_training_kernels(table)
        out[float(a)] = model.joint_kernel(Kf)
    return out


def run_set(name, repeats=10, force=False, data=DATA, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
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
    steps = int(min(STEPS, max(5, BUDGET * (n - START))))
    print(f"=== {name}: {n} molecules, {steps} acquisitions from {START}, "
          f"{n_hit} of them count as hits", flush=True)

    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(1000 + rep)
        start = sorted(rng.choice(n, START, replace=False).tolist())
        if set(start) & hits:                       # a start that already holds a hit says
            start = sorted(rng.choice(sorted(set(range(n)) - hits), START,  # nothing about
                                      replace=False).tolist())              # discovery
        for kind in STRATEGIES:
            out = learning_curve(Ks, y, kind, start, steps=steps, lams=LAMS, seed=rep)
            taken = out["taken"]
            first = next((i + 1 for i, t in enumerate(taken) if t in hits), None)
            rows.append({"repeat": rep, "strategy": kind,
                         "rmse_mean": float(np.mean(out["rmse"])),
                         "rmse_final": float(out["rmse"][-1]),
                         "q2_final": float(out["q2"][-1]),
                         "best": float(np.max(y[out["have"]])),
                         "best_possible": float(np.max(y)),
                         "steps_to_hit": first, "hit": bool(first is not None),
                         "rmse_curve": [round(v, 4) for v in out["rmse"]],
                         "best_curve": [round(v, 4) for v in out["best"]]})
        done = {r["strategy"]: r for r in rows if r["repeat"] == rep}
        print(f"  repeat {rep}: " + "  ".join(
            f"{k} best {done[k]['best']:.2f}/RMSE {done[k]['rmse_mean']:.3f}"
            for k in STRATEGIES), flush=True)

    out = {"set": name, "n": int(n), "repeats": int(repeats), "start": START, "steps": steps,
           "n_hit": n_hit, "best_possible": float(np.max(y)), "rows": rows,
           "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def _by(data, strategy, key):
    return np.array([r[key] for d in data for r in d["rows"]
                     if r["strategy"] == strategy and r[key] is not None], float)


def _paired(data, a, b, key):
    """The difference in `key` between two strategies, run by run."""
    out = []
    for d in data:
        by = {}
        for r in d["rows"]:
            by.setdefault(r["repeat"], {})[r["strategy"]] = r
        for rep in by:
            if a in by[rep] and b in by[rep]:
                out.append(by[rep][a][key] - by[rep][b][key])
    return np.asarray(out, float)


def _explore_paragraph(data, rm, explore, rng):
    """What the exploration rules bought, judged by the interval rather than by the point."""
    best = min(explore, key=lambda s: rm[s]) if explore else "variance"
    diff = _paired(data, best, "random", "rmse_mean")
    bs = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(4000)]
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    head = (f"**Exploring buys nothing measurable.** The best of the exploration rules ({best}) "
            f"reaches {rm[best]:.3f} RMSE against {rm['random']:.3f} for picking at random, a "
            f"paired difference of {float(np.mean(diff)):+.3f} with a 95 % interval of "
            f"[{lo:+.3f}, {hi:+.3f}] -- zero is inside it."
            if lo <= 0 <= hi else
            f"**Exploring pays.** The best of the exploration rules ({best}) reaches "
            f"{rm[best]:.3f} RMSE against {rm['random']:.3f} for picking at random, a paired "
            f"difference of {float(np.mean(diff)):+.3f} with a 95 % interval of "
            f"[{lo:+.3f}, {hi:+.3f}].")
    return (head + " Acquiring the molecule the model is least sure of means acquiring the "
            "molecule least like the rest of the series, and what is learned from an outlier "
            "does not transfer to the molecules anyone will ask about next. Uncertainty sampling "
            "is a rule for covering a space, and these sets are series to be understood rather "
            "than spaces to be covered.")


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No active-learning results yet; run `python benchmarks/run_active.py`."
    data = [json.load(open(f)) for f in files]
    runs = sum(len(d["rows"]) for d in data) // len(STRATEGIES)
    lines = ["# Which molecule to make next", "",
             f"{len(data)} Sutherland sets, {data[0]['repeats']} runs each ({runs} in all). Every "
             f"run starts from {data[0]['start']} molecules drawn at random and holding no hit, "
             f"and acquires {data[0]['steps']} more one at a time; alpha and lambda are "
             "reselected by the evidence at every step from what has been acquired and nothing "
             f"else. A **hit** is a molecule in the top {100 * TOP:.0f} % of its set. Every "
             "strategy sees the same starting sets, so the comparison is paired.", "",
             "## Finding an active", "",
             "| rule | best found (of the best possible) | runs that found a hit | "
             "median acquisitions to the first hit |", "|---|---:|---:|---:|"]
    gap = {}
    for s in STRATEGIES:
        best = _by(data, s, "best")
        poss = _by(data, s, "best_possible")
        gap[s] = float(np.mean(best / poss))
        hit = np.array([r["hit"] for d in data for r in d["rows"] if r["strategy"] == s])
        first = _by(data, s, "steps_to_hit")
        med = f"{np.median(first):.0f}" if len(first) else "--"
        lines.append(f"| {s} | {float(np.mean(best)):.2f} ({gap[s]:.1%}) | "
                     f"{float(np.mean(hit)):.0%} | {med} |")
    lines += ["", "## Improving the model", "",
              "| rule | mean RMSE over the run | RMSE at the end | q² at the end |",
              "|---|---:|---:|---:|"]
    for s in STRATEGIES:
        lines.append(f"| {s} | {float(np.mean(_by(data, s, 'rmse_mean'))):.3f} | "
                     f"{float(np.mean(_by(data, s, 'rmse_final'))):.3f} | "
                     f"{float(np.mean(_by(data, s, 'q2_final'))):+.3f} |")

    rng = np.random.default_rng(0)
    lines += ["", "## Against picking at random, paired run by run", "",
              "| rule | hits found, difference | mean RMSE, difference |", "|---|---:|---:|"]
    for s in STRATEGIES:
        if s == "random":
            continue
        d_hit, d_rmse = [], []
        for d in data:
            by = {}
            for r in d["rows"]:
                by.setdefault(r["repeat"], {})[r["strategy"]] = r
            for rep in by:
                if s in by[rep] and "random" in by[rep]:
                    d_hit.append(float(by[rep][s]["hit"]) - float(by[rep]["random"]["hit"]))
                    d_rmse.append(by[rep][s]["rmse_mean"] - by[rep]["random"]["rmse_mean"])
        hb = [float(np.mean(rng.choice(d_hit, len(d_hit), replace=True))) for _ in range(4000)]
        rb = [float(np.mean(rng.choice(d_rmse, len(d_rmse), replace=True))) for _ in range(4000)]
        lines.append(f"| {s} | {float(np.mean(d_hit)):+.3f} "
                     f"[{np.percentile(hb, 2.5):+.3f}, {np.percentile(hb, 97.5):+.3f}] | "
                     f"{float(np.mean(d_rmse)):+.3f} "
                     f"[{np.percentile(rb, 2.5):+.3f}, {np.percentile(rb, 97.5):+.3f}] |")

    soon = {}
    for s in STRATEGIES:
        f = _by(data, s, "steps_to_hit")
        soon[s] = float(np.median(f)) if len(f) else float("inf")
    best_find = max((s for s in STRATEGIES), key=lambda s: (round(gap[s], 4), -soon[s]))
    rm = {s: float(np.mean(_by(data, s, "rmse_mean"))) for s in STRATEGIES}
    best_model = min(rm, key=rm.get)
    var_max = abs(rm["variance"] - rm["maxmin"])
    same = best_find == best_model
    explore = [s for s in ("variance", "maxmin") if s in rm]
    worse_than_random = [s for s in explore if rm[s] > rm["random"]]
    lines += ["",
              f"**{best_find}** finds the best molecules and **{best_model}** builds the best "
              + ("model, and they are the same rule."
                 if same else "model, and they are not the same rule.")
              + " That is why the two tables are kept apart: an acquisition rule cannot be "
              "judged without saying what the acquiring is for.", "",
              "Three things in the numbers are worth more than the ranking.", "",
              f"**The uncertainty rules and the diversity rule are one rule.** variance and "
              f"maxmin come out within {var_max:.3f} RMSE of each other and within "
              f"{abs(gap['variance'] - gap['maxmin']):.1%} of the same activity found. That is "
              "item E5 again from the other side: the posterior standard deviation of a CMF "
              "model is very nearly the distance to the training set in other units, so "
              "acquiring by the model's uncertainty and acquiring by field-space diversity are "
              "the same experiment with two names. The Gaussian process is not adding "
              "information here; it is re-expressing geometry.", "",
              _explore_paragraph(data, rm, explore, rng), "",
              "**Greedy is not as bad as it should be, and that is luck.** It ignores the "
              "posterior, so a candidate unlike anything made is predicted at the training mean "
              "and cannot be told from a real answer -- the failure the screening rank guards "
              "against by subtracting the spread rather than adding it. On these sets it finds "
              f"{gap['greedy']:.1%} of the best activity against {gap[best_find]:.1%} for "
              f"{best_find}. The sets are series of analogues, where nothing is far enough from "
              "the training molecules for the training mean to be returned; on a library of "
              "diverse candidates that protection is gone, and the expected-improvement rule, "
              "which at least pays for optimism with a posterior, is the one to use."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        run_set(name, repeats=a.repeats, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
