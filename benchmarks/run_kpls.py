#!/usr/bin/env python
"""
Kernel partial least squares against kernel ridge regression on the eight Sutherland sets.

Kernel PLS (``--method kpls``) is the kernel form of the PLS regression that CoMFA uses.  With
the linear CMF kernel it is PLS in the space of the fields, so the comparison with KRR is the
comparison of CoMFA's learner with the learner of the 2013 paper on the same fields, the same
alignment and the same folds.  The two differ only in how they regularise: KRR shrinks every
direction of the field space by a ridge lambda, PLS keeps a few directions whole and drops the
rest, and the number of directions is its only setting.

Protocol.  The in-repository copies of the Sutherland sets (``examples/sutherland``, with the
file charges and the published training/test split), the RDKit parameterisation, unscaled field
kernels, the alpha grid of the original R scripts and the interleaved folds of those scripts.
Each learner chooses its own setting on the same 10-fold RMSEcv: KRR its lambda on an
eight-point grid, KPLS its number of components among 1 to 12.  The external cross-validation
(5 outer folds, the whole set) repeats that choice inside every outer fold, so q2ex and RMSEcvex
are the fair comparison; q2 and R2pred also carry the choice of alpha made on the training set.

Usage
-----
    python benchmarks/run_kpls.py                 # every set, both learners
    python benchmarks/run_kpls.py --sets THR,ACE --force
    python benchmarks/run_kpls.py --report benchmarks/results/kpls/table.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import cmf_cli  # noqa: E402

RESULTS = os.path.join(HERE, "results", "kpls")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")

R_ALPHAS = "0.01,0.05,0.1,0.25,0.5,0.75,1,2,3,5,10"
COMMON = ["--param", "rdkit", "--charges", "file", "--raw-scale", "--cv-no-shuffle",
          "--alphas", R_ALPHAS, "--nested", "--nested-on", "all"]
MODES = {
    "krr": ["--method", "krr", "--lams", "0.01,0.03,0.1,0.3,1,3,10,20"],
    "kpls": ["--method", "kpls", "--pls-components", "1,2,3,4,5,6,7,8,9,10,11,12"],
}
LABELS = {"krr": "kernel ridge regression (lambda by CV)",
          "kpls": "kernel PLS (components by CV)"}
STATS = ("q2", "rmse_cv", "r2_pred", "rmse_pred", "q2_ex", "rmse_ex")
STAT_LABELS = ("q2", "RMSEcv", "R2pred", "RMSEpred", "q2ex", "RMSEcvex")


def run_one(name: str, mode: str, force: bool = False) -> str:
    out = os.path.join(RESULTS, f"{name}_{mode}.json")
    if os.path.exists(out) and not force:
        print(f"[skip] {out} exists")
        return out
    os.makedirs(RESULTS, exist_ok=True)
    argv = ["build", os.path.join(DATA, f"{name}.sdf")] + COMMON + MODES[mode] + ["--json", out]
    print(f"\n=== {name} / {mode}: cmf_cli.py {' '.join(argv)}", flush=True)
    t0 = time.time()
    cmf_cli.main(argv)
    print(f"=== {name} / {mode} finished in {time.time() - t0:.0f} s", flush=True)
    return out


def load_results():
    res = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        name, mode = os.path.basename(f)[:-5].split("_", 1)
        with open(f) as fh:
            res[(name, mode)] = json.load(fh)
    return res


def fmt(v, nd=2):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{nd}f}" if v == v else "-"


def _setting(r):
    p = r.get("best_params", {})
    if "components" in p:
        return f"{int(p['components'])} comp."
    if "lam" in p:
        return f"lambda {fmt(p['lam'], 3)}"
    return "-"


def report(path: str | None) -> str:
    res = load_results()
    lines = ["| Set | Learner | " + " | ".join(STAT_LABELS) + " | alpha | setting chosen |",
             "|---|---|" + "---|" * len(STATS) + "---|---|"]
    sums = {m: {k: [] for k in STATS} for m in MODES}
    wins = {k: 0 for k in STATS}
    pairs = 0
    for name in SETS:
        first = True
        for mode in MODES:
            r = res.get((name, mode))
            if r is None:
                continue
            vals = [r.get(k) for k in STATS]
            for k, v in zip(STATS, vals):
                if v is not None and v == v:
                    sums[mode][k].append(float(v))
            lines.append(f"| {'**' + name + '**' if first else ''} | {LABELS[mode]} | "
                         + " | ".join(fmt(v) for v in vals)
                         + f" | {fmt(r.get('best_alpha'), 2)} | {_setting(r)} |")
            first = False
        a, b = res.get((name, "krr")), res.get((name, "kpls"))
        if a is not None and b is not None:
            pairs += 1
            for k in STATS:
                va, vb = a.get(k), b.get(k)
                if va is None or vb is None:
                    continue
                better = (vb > va) if k.startswith("q2") or k.startswith("r2") else (vb < va)
                wins[k] += int(better)
    lines.append("")
    lines.append(f"Mean over the sets with both learners ({pairs} sets), and on how many of them "
                 "kernel PLS is the better of the two:")
    lines.append("")
    lines.append("| Learner | " + " | ".join(STAT_LABELS) + " |")
    lines.append("|---|" + "---|" * len(STATS))
    for mode in MODES:
        s = sums[mode]
        lines.append(f"| {LABELS[mode]} | " + " | ".join(
            fmt(sum(v) / len(v), 3) if v else "-" for v in s.values()) + " |")
    lines.append("| kernel PLS better, sets out of " + str(pairs) + " | "
                 + " | ".join(str(wins[k]) for k in STATS) + " |")
    text = "\n".join(lines) + "\n"
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=None, help="comma-separated data sets (default: all eight)")
    ap.add_argument("--modes", default="krr,kpls")
    ap.add_argument("--force", action="store_true", help="recompute existing results")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None,
                    help="only write the Markdown table (default path: results/kpls/table.md)")
    args = ap.parse_args(argv)
    if args.report is None:
        names = [s for s in args.sets.split(",") if s in SETS] if args.sets else list(SETS)
        modes = [m for m in args.modes.split(",") if m in MODES]
        t0 = time.time()
        for name in names:
            for mode in modes:
                run_one(name, mode, force=args.force)
        print(f"\nAll runs finished in {(time.time() - t0) / 60:.1f} min")
        args.report = os.path.join(RESULTS, "table.md")
    print(report(args.report))


if __name__ == "__main__":
    main()
