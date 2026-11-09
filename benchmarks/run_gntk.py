#!/usr/bin/env python
"""
Does the bond graph add anything to the fields? The graph neural tangent kernel as a view (ND7)

The field kernel is the kernel of an infinitely wide radial-basis network on the aligned fields;
the graph neural tangent kernel (``cmfqsar/gntk.py``) is the kernel of an infinitely wide graph
network on the bond graph, computed in closed form with no training. Stacked under the multiple
kernel learning of item E6 as one more view, it gets a weight by the evidence like any field.
This measures whether that view adds anything, on the eight Sutherland sets and on a random
subset of COMPAS-1, against two references:

* the five fields alone, weighed by the evidence;
* the five fields plus the rotation-invariant autocorrelation kernel of item C5 as a view, which
  is the alignment-free description the program already has -- the plan says that a gain the
  invariant kernel already gives is not a result.

Every variant is cross-validated on the same ten folds with the ridge chosen by the evidence on
the whole set, and the differences in q2 are given with a paired bootstrap over the molecules
(the interval of rule 4). With them: the weight the evidence gives the graph view, its share of
the joint kernel, and the Bayes factor against it doing nothing (``mixing.drop_one``).

Usage
-----
    python benchmarks/run_gntk.py                  # the Sutherland sets
    python benchmarks/run_gntk.py --compas         # a subset of COMPAS-1D
    python benchmarks/run_gntk.py --report
"""
from __future__ import annotations

import argparse
import copy
import glob
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.gntk import gntk_kernel  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, autocorrelation_kernel, feature_matrix  # noqa: E402
from cmfqsar.mixing import drop_one, evidence_mixing, joint_kernel, with_views  # noqa: E402
from cmfqsar.model import cv_predict, r2_score  # noqa: E402
from run_errorpred import DATA, FIELDS, LAMS, SETS, _choose, _kernels  # noqa: E402
from run_sac import bootstrap_difference  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "gntk")
FOLDS = 10
SEED = 0
BLOCKS = 2
COMPAS_SUBSET = 2000
COMPAS_GROUPS = "ABC"
COMPAS_ALPHA = 0.1          # what the COMPAS benchmark chose for every group of this set
VARIANTS = ("fields", "fields + invariant", "fields + graph", "fields + invariant + graph")


def _say(s: str) -> None:
    print(s, flush=True)


def _rmse(y, p):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def _views_for(variant: str, K_inv, K_gntk):
    views = {}
    if "invariant" in variant:
        views["invariant"] = K_inv
    if "graph" in variant:
        views["graph"] = K_gntk
    return views


def measure(name: str, K_fields: np.ndarray, field_names, y: np.ndarray, K_inv: np.ndarray,
            K_gntk: np.ndarray, log=_say) -> dict:
    """Every variant on one set: the weights by the evidence, the ridge by the evidence, the
    out-of-fold predictions on the same folds, and the paired differences."""
    out = {"variants": {}, "seconds": {}}
    preds = {}
    for variant in VARIANTS:
        t0 = time.time()
        stack, names, scale = with_views(K_fields, _views_for(variant, K_inv, K_gntk),
                                         names=list(field_names))
        h = evidence_mixing(stack, y, scale=scale)
        K = joint_kernel(stack, h, scale)
        best = max(LAMS, key=lambda lam: log_evidence(K, y, float(lam)))
        pred = cv_predict(K, y, "krr", {"lam": float(best)}, folds=FOLDS, seed=SEED)
        preds[variant] = pred
        shares = np.asarray(h, float) * np.asarray(scale, float)
        shares = shares / max(float(shares.sum()), 1e-300)
        entry = {"lam": float(best), "q2": float(r2_score(y, pred)), "rmse_cv": _rmse(y, pred),
                 "weights": {n: float(w) for n, w in zip(names, h)},
                 "shares": {n: float(s) for n, s in zip(names, shares)}}
        if "graph" in variant:
            entry["bayes_factor_graph"] = float(drop_one(stack, y, names, scale=scale)["graph"])
        out["variants"][variant] = entry
        out["seconds"][variant] = time.time() - t0
        log(f"  {variant:28s} lambda {best:g}  q2 {entry['q2']:+.3f}  RMSE "
            f"{entry['rmse_cv']:.3f}"
            + (f"  graph share {entry['shares']['graph']:.0%}, Bayes factor "
               f"{entry['bayes_factor_graph']:+.1f}" if "graph" in variant else ""))
    pairs = [("fields + graph", "fields"), ("fields + graph", "fields + invariant"),
             ("fields + invariant + graph", "fields + invariant"),
             ("fields + invariant", "fields")]
    out["pairs"] = []
    for a, b in pairs:
        d, lo, hi = bootstrap_difference(y, preds[a], preds[b], runs=2000, seed=SEED)
        out["pairs"].append({"a": a, "b": b, "delta": d, "low": lo, "high": hi})
        log(f"    {a} against {b}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return out


# --------------------------------------------------------------------------- #
# The Sutherland sets
# --------------------------------------------------------------------------- #
def run_sutherland(name: str, results: str, force: bool = False, log=_say) -> dict:
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        log(f"{name}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    y = y[keep]
    n = len(y)
    log(f"=== {name}: {n} molecules")
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="file")
    mf = [build_fields(m, fs, nm) for m, nm in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    cache = _kernels(table)
    alpha, _lam = _choose(cache, y, np.arange(n))
    Kf, _sd = cache[alpha]
    t1 = time.time()
    K_gntk = gntk_kernel(table, blocks=BLOCKS)
    t_gntk = time.time() - t1
    t1 = time.time()
    F = feature_matrix(table.molecules, AutocorrelationSpec(), fieldset=fs)
    K_inv = autocorrelation_kernel(F)
    t_inv = time.time() - t1
    log(f"  alpha {alpha:g} by the evidence; graph kernel {t_gntk:.1f} s, invariant kernel "
        f"{t_inv:.1f} s")
    out = measure(name, Kf, FIELDS, y, K_inv, K_gntk, log)
    out.update({"set": name, "n": n, "alpha": alpha, "blocks": BLOCKS, "folds": FOLDS,
                "seconds_graph": t_gntk, "seconds_invariant": t_inv,
                "seconds": {**out["seconds"], "total": time.time() - t0}})
    json.dump(out, open(out_path, "w"), indent=1)
    log(f"  {time.time() - t0:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# A subset of COMPAS-1
# --------------------------------------------------------------------------- #
def run_compas(results: str, subset: int = COMPAS_SUBSET, force: bool = False, log=_say) -> dict:
    import run_compas as C

    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, "compas_1D.json")
    if os.path.exists(out_path) and not force:
        log("COMPAS-1D: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds, meta = C.load_prepared("1D")
    rng = np.random.RandomState(SEED)
    idx = np.sort(rng.choice(len(ds.names), int(subset), replace=False))
    sub = copy.copy(ds)
    sub.mols = [ds.mols[i] for i in idx]
    sub.names = [ds.names[i] for i in idx]
    sub.activity = np.asarray(ds.activity, float)[idx]
    y = np.asarray(sub.activity, float)
    log(f"=== COMPAS-1D: {len(y)} of {meta['n']} molecules drawn at random (seed {SEED}), "
        f"target {C.TARGET}")
    fs = C.make_fieldset(COMPAS_GROUPS, meta, {g: COMPAS_ALPHA for g in COMPAS_GROUPS})
    table = C.build_table(sub, fs)
    from cmfqsar.model import CMFModel
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    Kf = np.asarray(model.prepare_training_kernels(table), float)
    field_names = [s.name for s in fs.specs if s.enabled]
    t1 = time.time()
    K_gntk = gntk_kernel(table, blocks=BLOCKS)
    t_gntk = time.time() - t1
    t1 = time.time()
    F = feature_matrix(table.molecules, AutocorrelationSpec(), fieldset=fs)
    K_inv = autocorrelation_kernel(F)
    t_inv = time.time() - t1
    log(f"  {len(field_names)} fields at alpha {COMPAS_ALPHA:g}; graph kernel {t_gntk:.0f} s, "
        f"invariant kernel {t_inv:.0f} s")
    out = measure("COMPAS-1D", Kf, field_names, y, K_inv, K_gntk, log)
    out.update({"set": "COMPAS-1D", "n": int(len(y)), "of": int(meta["n"]), "alpha": COMPAS_ALPHA,
                "fields": field_names, "blocks": BLOCKS, "folds": FOLDS,
                "seconds_graph": t_gntk, "seconds_invariant": t_inv,
                "seconds": {**out["seconds"], "total": time.time() - t0}})
    json.dump(out, open(out_path, "w"), indent=1)
    log(f"  {time.time() - t0:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def report(results: str = RESULTS) -> str:
    compas = os.path.basename(os.path.normpath(results)).endswith("_compas")
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    data = [json.load(open(f)) for f in files]
    data.sort(key=lambda d: -d["n"])
    head = ["# The graph neural tangent kernel as a kernel view (plan item ND7)", ""]
    if not data:
        return "\n".join(head) + "No results yet; run `python benchmarks/run_gntk.py`" + (
            " --compas" if compas else "") + "."
    lines = head + [
        ("A random subset of COMPAS-1D, the fields of the COMPAS benchmark's set ABC" if compas
         else "The eight Sutherland sets, the five fields at the attenuation the evidence chose")
        + f", {data[0]['folds']} folds, the ridge by the evidence. Every variant stacks its "
        "kernels under the weights of plan item E6 (the evidence); *graph* is the graph neural "
        f"tangent kernel of the bond graph with {data[0]['blocks']} blocks, *invariant* the "
        "rotation-invariant autocorrelation of item C5. The graph view's share of the joint "
        "kernel and the Bayes factor against it doing nothing are given for the variant that "
        "holds it with the fields.", "",
        "| set | n | q2 fields | q2 + invariant | q2 + graph | q2 + both | graph share | Bayes "
        "factor, graph | graph vs fields | graph vs invariant | both vs invariant |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    acc = {"gf": [], "gi": [], "bi": [], "share": [], "bf": []}
    for d in data:
        v = d["variants"]
        p = {(x["a"], x["b"]): x for x in d["pairs"]}
        gf = p[("fields + graph", "fields")]
        gi = p[("fields + graph", "fields + invariant")]
        bi = p[("fields + invariant + graph", "fields + invariant")]
        share = v["fields + graph"]["shares"]["graph"]
        bf = v["fields + graph"]["bayes_factor_graph"]
        for k, val in (("gf", gf["delta"]), ("gi", gi["delta"]), ("bi", bi["delta"]),
                       ("share", share), ("bf", bf)):
            acc[k].append(val)
        lines.append(
            f"| {d['set']} | {d['n']} | {v['fields']['q2']:+.3f} | "
            f"{v['fields + invariant']['q2']:+.3f} | {v['fields + graph']['q2']:+.3f} | "
            f"{v['fields + invariant + graph']['q2']:+.3f} | {share:.0%} | {bf:+.1f} | "
            f"{gf['delta']:+.3f} [{gf['low']:+.3f}, {gf['high']:+.3f}] | "
            f"{gi['delta']:+.3f} [{gi['low']:+.3f}, {gi['high']:+.3f}] | "
            f"{bi['delta']:+.3f} [{bi['low']:+.3f}, {bi['high']:+.3f}] |")
    if len(data) > 1:
        lines.append(f"| **mean** |  |  |  |  |  | {np.mean(acc['share']):.0%} | "
                     f"{np.mean(acc['bf']):+.1f} | {np.mean(acc['gf']):+.3f} | "
                     f"{np.mean(acc['gi']):+.3f} | {np.mean(acc['bi']):+.3f} |")
    wins_f = sum(1 for d in data for x in d["pairs"]
                 if (x["a"], x["b"]) == ("fields + graph", "fields") and x["low"] > 0)
    loss_f = sum(1 for d in data for x in d["pairs"]
                 if (x["a"], x["b"]) == ("fields + graph", "fields") and x["high"] < 0)
    wins_i = sum(1 for d in data for x in d["pairs"]
                 if (x["a"], x["b"]) == ("fields + graph", "fields + invariant") and x["low"] > 0)
    loss_i = sum(1 for d in data for x in d["pairs"]
                 if (x["a"], x["b"]) == ("fields + graph", "fields + invariant") and x["high"] < 0)
    lines += ["", "Differences in q2 with their 95 % paired bootstrap intervals over the molecules. "
              f"Against the fields alone the graph view wins on {wins_f} of the {len(data)} sets "
              f"and loses on {loss_f} (an interval that excludes zero); against the fields with "
              f"the invariant kernel it wins on {wins_i} and loses on {loss_i}. The plan reads "
              "the second comparison: a gain the invariant kernel already gives is not a result."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--compas", action="store_true", help="a subset of COMPAS-1D instead")
    ap.add_argument("--subset", type=int, default=COMPAS_SUBSET)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    results = RESULTS + ("_compas" if args.compas else "")
    if not args.report:
        if args.compas:
            run_compas(results, subset=args.subset, force=args.force)
        else:
            for name in args.sets:
                run_sutherland(name, results, force=args.force)
    text = report(results)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
