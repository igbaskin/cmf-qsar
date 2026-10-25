#!/usr/bin/env python
"""
Do the signed three-point channels buy anything?  C5 of the development plan, measured.

The field autocorrelation of `cmfqsar/invariant.py` is alignment-free and geometry-aware, and it
is two-point: it counts pairs of atoms at a distance, which is a quantity a mirror image has in
exactly the same amounts.  The kernel is therefore blind to chirality, and blind as well to the
orientation of one field with respect to another.  The signed triple product of three
displacement vectors is not, and `chirality()` adds it as a second block of channels, resolved
over a few radial shells.

Whether that is worth the columns it costs is an empirical question, and the bar is set by the
descriptor it extends rather than by an aligned model: if the three-point channels add nothing to
the two-point ones, the item is closed.

The test is leave-one-chemotype-out on the five chemotype-split sets, which is the regime the
alignment-free kernel exists for -- a held-out molecule has no scaffold-mate in the training part,
so nothing can be recovered from a neighbour of the same series.

Every variant gets the same freedom and the same honesty: three regularisations by three distance
grids, chosen **inside the training part of every fold** by a 10-fold cross-validation and never
on the fold being predicted.  The number of radial shells is not searched over; each setting is a
column of its own, so that the signed channels are never given more freedom than the kernel they
are being compared with.

Usage
-----
    python benchmarks/run_chirality.py
    python benchmarks/run_chirality.py --sets COX2 DHFR
    python benchmarks/run_chirality.py --report
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
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               block_sizes, channel_scale, feature_matrix)
from cmfqsar.model import cv_predict, krr_fit, r2_score  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "chirality")
DATA = os.path.join(ROOT, "examples")
#: the five sets the README's chemotype table uses
SETS = {
    "COX2": "sutherland/COX2.sdf",
    "DHFR": "sutherland/DHFR.sdf",
    "BZR": "sutherland/BZR.sdf",
    "AChE": "sutherland/AChE.sdf",
    "mittal/ACHE": "mittal/ACHE.sdf",
}
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5                     # the attenuation does not enter this descriptor; it builds fields
LAMS = (0.01, 0.1, 1.0)
#: the three distance grids, as (sigma, step, d_max) -- the freedom the aligned model gets from
#: its three attenuations
GRIDS = ((0.5, 0.5, 10.0), (0.75, 0.75, 12.0), (1.0, 1.0, 14.0))
#: the variants compared: the two-point kernel, and the same with the signed channels over 1, 2
#: and 4 radial shells
VARIANTS = (("two-point", 0), ("signed, 1 shell", 1), ("signed, 2 shells", 2),
            ("signed, 4 shells", 4))


def _specs(shells):
    out = []
    for sigma, step, d_max in GRIDS:
        out.append(AutocorrelationSpec(sigma=sigma, step=step, d_max=d_max,
                                       chirality=bool(shells),
                                       chirality_shells=max(1, int(shells))))
    return out


def _kernels(mf, n_fields, shells):
    """One cosine-normalised kernel over every molecule, per distance grid."""
    out = []
    for spec in _specs(shells):
        X = feature_matrix(mf, spec, n_fields=n_fields)
        blocks = block_sizes(spec, n_fields)
        scale = channel_scale(X, spec.n_bins, blocks=blocks)
        out.append(autocorrelation_kernel(X, scale=scale, n_bins=spec.n_bins, blocks=blocks))
    return out


def _grouped_nested(kernels, y, labels, inner_folds=10, seed=0):
    """Leave one chemotype out, choosing the grid and lambda inside the training part."""
    pred = np.full(len(y), np.nan)
    for g in sorted(set(labels.tolist())):
        te = np.flatnonzero(labels == g)
        tr = np.flatnonzero(labels != g)
        if len(tr) < 8 or not len(te):
            continue
        best = None
        for K in kernels:
            Ktr = K[np.ix_(tr, tr)]
            for lam in LAMS:
                p = cv_predict(Ktr, y[tr], "krr", {"lam": float(lam)},
                               folds=min(inner_folds, len(tr)), seed=seed)
                q = r2_score(y[tr], p)
                if best is None or q > best[0]:
                    best = (q, K, float(lam))
        _q, K, lam = best
        a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
        pred[te] = K[np.ix_(te, tr)] @ a + b
    return pred


def run_set(name, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, name.replace("/", "_") + ".json")
    if os.path.exists(out_path) and not force:
        print(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(DATA, SETS[name]))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    labels, _scaffolds = chemotype_labels(mols, min_size=3)
    labels = np.asarray(labels)
    n_groups = len(set(labels.tolist()))
    print(f"=== {name}: {len(y)} molecules, {n_groups} chemotypes", flush=True)

    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA, charge_model="file")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    n_fields = len(fs.specs)

    row = {"set": name, "n": int(len(y)), "n_groups": int(n_groups), "q2": {}, "columns": {}}
    for label, shells in VARIANTS:
        ks = _kernels(mf, n_fields, shells)
        pred = _grouped_nested(ks, y, labels)
        ok = np.isfinite(pred)
        row["q2"][label] = float(r2_score(y[ok], pred[ok])) if ok.any() else float("nan")
        row["columns"][label] = int(ks[1].shape[0] and feature_matrix(
            mf[:1], _specs(shells)[1], n_fields=n_fields).shape[1])
        print(f"  {label:17s} q2 {row['q2'][label]:+.3f}  ({row['columns'][label]} channels)",
              flush=True)
    row["seconds"] = round(time.time() - t0, 1)
    json.dump(row, open(out_path, "w"), indent=1)
    return row


def report(results=RESULTS):
    import glob

    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No chirality results yet; run `python benchmarks/run_chirality.py`."
    data = [json.load(open(f)) for f in files]
    order = [name for name, _ in VARIANTS]
    lines = ["# Do the signed three-point channels buy anything?", "",
             "Leave-one-chemotype-out on the five chemotype-split sets, the regime the "
             "alignment-free kernel exists for. Three regularisations by three distance grids for "
             "every variant, chosen inside the training part of each fold; the number of radial "
             "shells is a column rather than a search, so the signed channels never get more "
             "freedom than the two-point kernel they extend.", "",
             "| set | n | chemotypes | " + " | ".join(order) + " |",
             "|---|---:|---:|" + "---:|" * len(order)]
    for d in sorted(data, key=lambda r: -r["n"]):
        best = max(d["q2"].values())
        cells = []
        for k in order:
            v = d["q2"].get(k, float("nan"))
            cells.append(f"**{v:+.3f}**" if abs(v - best) < 1e-9 else f"{v:+.3f}")
        lines.append(f"| {d['set']} | {d['n']} | {d['n_groups']} | " + " | ".join(cells) + " |")
    means = {k: float(np.nanmean([d["q2"].get(k, np.nan) for d in data])) for k in order}
    top = max(means.values())
    lines.append("| **mean** |  |  | " + " | ".join(
        (f"**{means[k]:+.3f}**" if abs(means[k] - top) < 1e-9 else f"{means[k]:+.3f}")
        for k in order) + " |")
    cols = {k: int(np.mean([d["columns"].get(k, 0) for d in data])) for k in order}
    lines += ["", "Channels per molecule, averaged over the sets: "
              + ", ".join(f"{k} {cols[k]}" for k in order) + ".", "",
              "## Does the difference survive the scatter?", "",
              "Paired against the two-point kernel, set by set, with a bootstrap interval over "
              "the five sets. Five is a small number of sets and the interval says so.", "",
              "| variant | mean difference | 95 % interval | better on |", "|---|---:|---|---:|"]
    base = np.array([d["q2"].get(order[0], np.nan) for d in data])
    rng = np.random.default_rng(0)
    for k in order[1:]:
        diff = np.array([d["q2"].get(k, np.nan) for d in data]) - base
        diff = diff[np.isfinite(diff)]
        if not len(diff):
            continue
        bt = [float(np.mean(rng.choice(diff, len(diff), replace=True))) for _ in range(5000)]
        lines.append(f"| `{k}` | {float(np.mean(diff)):+.3f} | "
                     f"[{np.percentile(bt, 2.5):+.3f}, {np.percentile(bt, 97.5):+.3f}] | "
                     f"{int((diff > 0).sum())}/{len(diff)} |")
    # the one set that carries the mean, named rather than left to be noticed
    best = order[-1]
    diff = {d["set"]: d["q2"].get(best, np.nan) - d["q2"].get(order[0], np.nan) for d in data}
    worst_first = sorted(diff.items(), key=lambda kv: -abs(kv[1]))
    if worst_first:
        name, value = worst_first[0]
        rest = [v for k, v in diff.items() if k != name and np.isfinite(v)]
        lines += ["", f"At {best.replace('`', '')} the mean difference is "
                  f"{float(np.nanmean(list(diff.values()))):+.3f}, of which **{name} supplies "
                  f"{value:+.3f}**; over the other four sets it is "
                  f"{float(np.mean(rest)):+.3f}."]
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.sets:
        run_set(name, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
