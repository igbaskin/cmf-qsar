#!/usr/bin/env python
"""
The four physical-property benchmarks without a conformer: the topological autocorrelation plus
the eight counts, against the field kernel plus the eight counts.

The benchmarks of logP, water solubility, boiling point and vapour pressure
(``benchmarks/run_logp.py``, ``run_solubility.py``, ``run_boiling.py`` and ``run_vapour.py``)
found one model best on all four properties: the cosine field kernel plus h times the Gaussian
kernel on eight counts. It beats the counts alone, and it beats the field kernel alone. Neither
of these two references tells whether the gain needs the 3D distances of the field kernel. The
same autocorrelation with the distance counted in bonds needs no conformer, and alone it is
close to the field kernel on three of the four properties.

The question, fixed before the measurement
------------------------------------------
Is the field kernel plus the eight counts better than the topological autocorrelation plus the
eight counts? The main comparison is

    invariant + eight counts  minus  topological + eight counts,

the difference of q2 (cross-validation) or R2 (external test) with a 95 % paired bootstrap
interval over the compounds, on each test of each benchmark. If the interval holds zero on most
tests, the gain of the combined model comes from the counts and the graph, and a conformer is not
needed for it.

How the numbers are made
------------------------
Exactly as in the four benchmarks, whose functions are imported: the same compounds, the same
folds, the same evidence selection. The new model is the topological kernel plus h times the
Gaussian kernel on the eight counts. In each block the largest distance in bonds and the count
width are those the single models chose there by the evidence on the same training block; only h
and the ridge are chosen here. This is the rule by which the benchmarks made *invariant + eight
counts*. The predictions of *invariant + eight counts* are read from the cache that each
benchmark writes (``benchmarks/cache/<property>/predictions.npz``), so the benchmark must have
been run first. As a control, the stage computes the topological model and the eight counts
again and checks that their predictions equal the stored ones.

Usage
-----
    python benchmarks/run_property_pairs.py                     # all four (about 2 hours)
    python benchmarks/run_property_pairs.py --property logp     # one of them
    python benchmarks/run_property_pairs.py --report
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem  # noqa: E402

from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.model import r2_score, rmse  # noqa: E402

import run_logp as L  # noqa: E402
import run_resolution as RS  # noqa: E402
from run_logp import (COUNT_WIDTHS, FIELDS, HS, REPEATS, TOPO_MAX, eight_counts,  # noqa: E402
                      gaussian_at, gaussian_stack, log, topological_features,
                      topological_kernel)

RESULTS = os.path.join(HERE, "results", "property_pairs")
CACHE = os.path.join(HERE, "cache")
PROPERTIES = ("logp", "solubility", "boiling", "vapour")
NAMES = {"logp": "logP", "solubility": "water solubility (logS)", "boiling": "boiling point, C",
         "vapour": "vapour pressure (logVP)"}
MODELS = ("eight counts", "topological autocorrelation", "topological + eight counts",
          "invariant + eight counts")
NEW = "topological + eight counts"
#: the stored predictions must equal the recomputed ones to this
TOLERANCE = 1e-6


class Bench:
    """What differs between the four benchmarks: the loader, the folds, the tests and their names."""

    def __init__(self, prop: str):
        self.prop = prop
        self.mod = importlib.import_module(f"run_{prop}")
        if prop == "logp":
            self.Model, self.splits = L.Model, L.splits
            self.sets = L.SETS
            self.externals = {"external": "huuskonen"}
            self.titles = {"huuskonen": "Huuskonen, CV", "sangster_p": "class P, CV",
                           "external": "class P to Huuskonen"}
        elif prop == "solubility":
            self.Model, self.splits = self.mod.Model, self.mod.splits
            self.sets = self.mod.SETS
            self.externals = {"external": self.mod.EXTERNAL[1]}
            self.titles = {"delaney": "Delaney, CV", "aqsoldb_reliable": "AqSolDB, CV",
                           "external": "AqSolDB to Delaney"}
        else:
            B = importlib.import_module("run_boiling")
            self.Model, self.splits = B.Model, B.splits
            self.sets = B.SETS
            self.externals = {t: part for t, (_, part) in B.EXTERNALS.items()}
            self.titles = dict(B.TITLES)
        self.tests = tuple(self.sets) + tuple(self.externals)

    def part(self, test: str) -> str:
        return self.externals.get(test, test)

    def predictions(self, model, y, blocks, idx):
        """Held-out predictions per test (mean over the repeats for CV) and q2 per test."""
        cv, ext, chosen = model.select(y, blocks)
        if not isinstance(ext, dict):
            ext = {"external": ext}
        preds, scores = {}, {}
        for s in self.sets:
            ii = idx[s]
            preds[s] = cv[s][:, ii].mean(axis=0)
            q2 = [float(r2_score(y[ii], cv[s][rep, ii])) for rep in range(REPEATS)]
            scores[s] = {"q2_mean": float(np.mean(q2)),
                         "rmse_mean": float(np.mean([rmse(y[ii], cv[s][rep, ii])
                                                     for rep in range(REPEATS)])),
                         "chosen": chosen[s]}
        for t in self.externals:
            yy = y[idx[self.part(t)]]
            preds[t] = ext[t]
            scores[t] = {"r2": float(r2_score(yy, ext[t])), "rmse": float(rmse(yy, ext[t])),
                         "chosen": chosen[t]}
        return preds, scores


def stored(prop: str):
    path = os.path.join(CACHE, prop, "predictions.npz")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run benchmarks/run_{prop}.py first")
    return np.load(path)


def run_one(prop: str) -> Dict[str, object]:
    t0 = time.time()
    b = Bench(prop)
    z = stored(prop)
    mols, y, idx = b.mod.load(0)
    heavy = [Chem.RemoveHs(m) for m in mols]
    names = np.array([m.GetProp("_Name") for m in mols])
    if not (z["names"].shape == names.shape and np.all(z["names"] == names)):
        raise SystemExit(f"the stored predictions of {prop} are for other compounds")
    if not np.allclose(z["y"], y):
        raise SystemExit(f"the stored predictions of {prop} are for other values")
    blocks = b.splits(idx)
    log(f"{prop}: {len(mols)} compounds, {len(blocks)} blocks")
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mfs = [build_fields(m, fs, str(n)) for m, n in zip(mols, names)]
    n_f = len(fs.specs)
    n_pairs = n_f * (n_f + 1) // 2
    top = max(TOPO_MAX)
    Ftop = topological_features(mols, mfs, top).reshape(len(mols), n_pairs, top + 1)

    def topo(dm):
        return topological_kernel(Ftop[:, :, :dm + 1].reshape(len(mols), -1), dm, n_f)

    models = {k: b.Model(k) for k in MODELS[:3]}
    for dm in TOPO_MAX:
        models["topological autocorrelation"].add(f"up to {dm} bonds", topo(dm), y, blocks)
    log(f"{prop}: topological model done ({time.time() - t0:.0f} s)")
    Z8 = eight_counts(heavy)
    for w, K in zip(COUNT_WIDTHS, gaussian_stack(Z8)):
        models["eight counts"].add(f"width {w:g}", K, y, blocks)
    log(f"{prop}: eight counts done ({time.time() - t0:.0f} s)")
    reach = [models["topological autocorrelation"].best_label(k) for k in range(len(blocks))]
    cwidth = [models["eight counts"].best_label(k) for k in range(len(blocks))]
    for dm in TOPO_MAX:
        here = {k for k in range(len(blocks)) if reach[k] == f"up to {dm} bonds"}
        if not here:
            continue
        Kt = topo(dm)
        for w in COUNT_WIDTHS:
            only = {k for k in here if cwidth[k] == f"width {w:g}"}
            if not only:
                continue
            Kg = gaussian_at(Z8, w)
            for h in HS:
                models[NEW].add(f"up to {dm} bonds, width {w:g}, h {h:g}", Kt + h * Kg, y,
                                blocks, only)
            del Kg
        del Kt
    log(f"{prop}: {NEW} done ({time.time() - t0:.0f} s)")

    preds, scores, control = {}, {}, {}
    for name, m in models.items():
        p, s = b.predictions(m, y, blocks, idx)
        preds[name], scores[name] = p, s
    for name in ("eight counts", "topological autocorrelation"):
        control[name] = float(max(np.max(np.abs(preds[name][t] - z[f"{name}|{t}"]))
                                  for t in b.tests))
        if control[name] > TOLERANCE:
            raise SystemExit(f"{prop}: the recomputed {name} differs from the stored one by "
                             f"{control[name]:.2e}; the benchmark has changed since it was run")
    log(f"{prop}: control passed, largest difference " +
        ", ".join(f"{k} {v:.1e}" for k, v in control.items()))
    preds["invariant + eight counts"] = {t: z[f"invariant + eight counts|{t}"] for t in b.tests}
    with open(os.path.join(HERE, "results", prop, f"{prop}.json"), encoding="utf-8") as fh:
        bench = json.load(fh)
    scores["invariant + eight counts"] = {
        t: {k: v for k, v in bench["models"]["invariant + eight counts"][t].items()}
        for t in b.tests}

    comps = []
    for a, c in ((NEW, "eight counts"), (NEW, "topological autocorrelation"),
                 ("invariant + eight counts", NEW)):
        for t in b.tests:
            g = RS.paired(y[idx[b.part(t)]], preds[a][t], preds[c][t])
            comps.append({"model": a, "against": c, "test": t, **g})
            log(f"  {prop} {a} - {c} [{t}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    return {"compounds": {k: int(len(v)) for k, v in idx.items()}, "tests": list(b.tests),
            "titles": {t: b.titles[t] for t in b.tests}, "models": scores,
            "comparisons": comps, "control_max_difference": control,
            "seconds": round(time.time() - t0, 1)}


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def run(props) -> Dict[str, object]:
    path = os.path.join(RESULTS, "pairs.json")
    payload = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    for prop in props:
        payload[prop] = run_one(prop)
        jdump(payload, path)
    if all(p in payload for p in PROPERTIES):
        report(payload)
    return payload


def _score(r: dict) -> str:
    if "q2_mean" in r:
        return f"{r['q2_mean']:+.3f} ({r['rmse_mean']:.3g})"
    return f"{r['r2']:+.3f} ({r['rmse']:.3g})"


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "pairs.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    lines = ["The four physical-property benchmarks with a model that needs no conformer: the "
             "topological autocorrelation (the five fields, distances counted in bonds) plus h "
             "times the Gaussian kernel on the eight counts. Same compounds, folds and evidence "
             "selection as the benchmarks; the rows of *invariant + eight counts* are those of "
             "the benchmarks. Each cell gives q² (cross-validation) or R² (external test), with "
             "the RMSE in brackets.", ""]
    for prop in PROPERTIES:
        if prop not in payload:
            continue
        r = payload[prop]
        tests = r["tests"]
        lines += [f"### {NAMES[prop]}", "",
                  "| Model | " + " | ".join(r["titles"][t] for t in tests) + " |",
                  "|---|" + "---:|" * len(tests)]
        for name in MODELS:
            lines.append(f"| {name} | " + " | ".join(_score(r["models"][name][t]) for t in tests)
                         + " |")
        lines += ["", "| Difference of q² or R² | Test | Mean [95 % interval] |", "|---|---|---|"]
        for c in r["comparisons"]:
            lines.append(f"| {c['model']} minus {c['against']} | {r['titles'][c['test']]} | "
                         f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] |")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--property", choices=PROPERTIES, action="append",
                    help="run only this property (may be given more than once)")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.property or PROPERTIES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
