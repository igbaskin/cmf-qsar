#!/usr/bin/env python
"""
Does ranking by a confidence bound beat ranking by the prediction?

A regression model used to prioritise a library has always been read here as a point estimate:
compute the predicted activity, sort, take the top. The objection is that a screening library is
by construction dominated by molecules unlike the training set, that a kernel method returns the
training mean dressed up for exactly those, and that a point estimate cannot tell that answer
from a real one. The remedy is to rank by a lower bound instead,

    score = prediction - beta * posterior standard deviation

which promotes a molecule only when the model is both optimistic and sure. `CMFModel.rank_score`
computes it; this measures whether it helps, on the two regimes that differ in how much the
uncertainty varies across the library:

    random     a random split of one series, where every held-out molecule has training
               neighbours and the spread of the posterior is narrow
    chemotype  one Bemis-Murcko chemotype held out at a time, where a held-out molecule has no
               scaffold-mate in the training set and the spread is wide

The second is the regime the objection is about, and the one where a bound has room to act. The
metrics are the early-recognition ones, since what a prioritisation is judged by is whether the
actives are at the top: a molecule counts as active when its activity is above the median of the
training part, so the labels never depend on anything held out.

Usage
-----
    python benchmarks/run_bound.py
    python benchmarks/run_bound.py --sets THR,COX2 --betas 0,0.5,1,2
    python benchmarks/run_bound.py --report
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

from cmfqsar.chemotypes import chemotype_labels  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, screening_scores  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "bound")
SUTHERLAND = os.path.join(ROOT, "examples", "sutherland")

SETS = ("COX2", "DHFR", "BZR", "AChE", "THR", "ACE", "GPB", "THER")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)
ALPHA = 0.5
LAM = 0.1


def tags_first(betas):
    """The tag of the first beta, used to record the enrichment ceiling only once per fold."""
    return f"{betas[0]:g}"


def _fit_and_rank(mf, y, train, test, fs, betas):
    """(beta -> ranking score over `test`), plus the spread of the posterior there."""
    tr = AtomTable.from_molecules([mf[i] for i in train])
    te = AtomTable.from_molecules([mf[i] for i in test])
    model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": LAM})
    model.fit(tr, y[train], cv_folds=0)
    out = {}
    for b in betas:
        score, yp, sd = model.rank_score(te, beta=float(b))
        out[f"{b:g}"] = np.asarray(score, float)
    return out, np.asarray(sd, float), np.asarray(yp, float)


def run(sets=SETS, betas=BETAS, out_dir=RESULTS):
    os.makedirs(out_dir, exist_ok=True)
    for name in sets:
        path = os.path.join(SUTHERLAND, f"{name}.sdf")
        if not os.path.exists(path):
            print(f"  {name}: no such file, skipped")
            continue
        t0 = time.time()
        ds = load_dataset(path)
        y = np.asarray(ds.activity, float)
        keep = np.flatnonzero(np.isfinite(y))
        mols = [ds.mols[i] for i in keep]
        names = [ds.names[i] for i in keep]
        y = y[keep]
        fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="file")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        labels, _ = chemotype_labels(mols, min_size=3)
        n = len(y)

        record = {"set": name, "n": int(n), "betas": [float(b) for b in betas],
                  "alpha": ALPHA, "lam": LAM, "regimes": {}}

        # the two regimes, as lists of (train, test) index arrays
        rng = np.random.default_rng(0)
        order = rng.permutation(n)
        folds = [order[i::5] for i in range(5)]
        splits = {"random": [(np.setdiff1d(np.arange(n), f), f) for f in folds],
                  "chemotype": [(np.flatnonzero(labels != g), np.flatnonzero(labels == g))
                                for g in sorted(set(labels.tolist()))]}

        for regime, parts in splits.items():
            acc = {f"{b:g}": {"auc": [], "ef5": [], "ef10": [], "bedroc": []} for b in betas}
            sd_spread, ceiling = [], []
            for train, test in parts:
                if len(test) < 4 or len(train) < 8:
                    continue
                med = float(np.median(y[train]))
                lab = np.where(y[test] > med, 1.0, -1.0)
                if len(set(lab.tolist())) < 2:
                    continue
                try:
                    scores, sd, yp = _fit_and_rank(mf, y, train, test, fs, betas)
                except Exception as exc:                    # noqa: BLE001
                    print(f"    {regime}: a fold failed: {exc}")
                    continue
                sd_spread.append(float(np.std(sd) / max(np.mean(sd), 1e-12)))
                for tag, sc in scores.items():
                    m = screening_scores(lab, sc, fractions=(0.05, 0.10))
                    if tag == tags_first(betas):
                        ceiling.append((m.get("max_ef_5", np.nan), m.get("max_ef_10", np.nan)))
                    acc[tag]["auc"].append(m.get("auc", np.nan))
                    acc[tag]["ef5"].append(m.get("ef_5", np.nan))
                    acc[tag]["ef10"].append(m.get("ef_10", np.nan))
                    acc[tag]["bedroc"].append(m.get("bedroc_20", np.nan))
            record["regimes"][regime] = {
                "relative_sd_spread": float(np.nanmean(sd_spread)) if sd_spread else float("nan"),
                "n_folds": len(sd_spread),
                "max_ef5": float(np.nanmean([c[0] for c in ceiling])) if ceiling else float("nan"),
                "max_ef10": float(np.nanmean([c[1] for c in ceiling])) if ceiling else float("nan"),
                "scores": {tag: {k: (float(np.nanmean(v)) if len(v) else float("nan"))
                                 for k, v in d.items()} for tag, d in acc.items()}}
            print(f"  {name} {regime}: {len(sd_spread)} folds, relative spread of the posterior "
                  f"{record['regimes'][regime]['relative_sd_spread']:.3f}", flush=True)
        with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
            json.dump(record, fh, indent=1)
        print(f"  {name} written, {time.time() - t0:.0f} s", flush=True)


def _load(out_dir=RESULTS):
    out = {}
    for name in SETS:
        p = os.path.join(out_dir, f"{name}.json")
        if os.path.exists(p):
            out[name] = json.load(open(p))
    return out


def report(path=None, out_dir=RESULTS):
    """results/bound/table.md."""
    data = _load(out_dir)
    if not data:
        print("no results yet; run the benchmark first")
        return
    sets = [s for s in SETS if s in data]
    betas = data[sets[0]]["betas"]
    tags = [f"{b:g}" for b in betas]
    lines = ["# Ranking by a confidence bound against ranking by the prediction", "",
             "Early recognition of the actives of a held-out part, a molecule counting as active",
             "when its activity is above the median of the training part. beta is how many",
             "posterior standard deviations are subtracted from the prediction before ranking;",
             "beta = 0 is the point estimate, which is what ranking by a regression model has",
             "always meant. Averaged over the folds of each set, then over the sets.", ""]
    for regime in ("random", "chemotype"):
        spread = [data[s]["regimes"][regime]["relative_sd_spread"] for s in sets
                  if regime in data[s]["regimes"]]
        lines += [f"## {regime} folds", "",
                  f"Relative spread of the posterior standard deviation across a held-out part: "
                  f"{np.nanmean(spread):.3f}. A bound can only act where this is large.", "",
                  f"The actives are about half of a held-out part here, so the enrichment "
                  f"factors saturate early: their ceilings are "
                  f"{np.nanmean([data[s]['regimes'][regime].get('max_ef5', np.nan) for s in sets]):.2f} "
                  f"at 5 per cent and "
                  f"{np.nanmean([data[s]['regimes'][regime].get('max_ef10', np.nan) for s in sets]):.2f} "
                  f"at 10. BEDROC is the metric with room left.", "",
                  "| beta | AUC | EF 5% | EF 10% | BEDROC(20) |", "|---|---:|---:|---:|---:|"]
        for tag in tags:
            row = []
            for key in ("auc", "ef5", "ef10", "bedroc"):
                vals = [data[s]["regimes"][regime]["scores"][tag][key] for s in sets
                        if regime in data[s]["regimes"]]
                row.append(float(np.nanmean(vals)))
            lines.append(f"| {tag} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
        lines.append("")
    lines += ["## Sets", "", "| set | molecules |", "|---|---:|"]
    for s in sets:
        lines.append(f"| {s} | {data[s]['n']} |")
    text = "\n".join(lines) + "\n"
    path = path or os.path.join(out_dir, "table.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"written to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--betas", default=",".join(f"{b:g}" for b in BETAS))
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.report:
        report()
        return
    run(sets=tuple(s for s in args.sets.split(",") if s),
        betas=tuple(float(b) for b in args.betas.split(",") if b))
    report()


if __name__ == "__main__":
    main()
