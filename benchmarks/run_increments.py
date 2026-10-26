#!/usr/bin/env python
"""
Do increments fitted to one series transfer to another? (plan item D5)

The hydrophobic field and the two hydrogen-bond fields are built from one number per (Sybyl atom
type, hydrogen count), taken from regressions of logP and of Abraham's solute constants
(``cmfqsar/increments.py``).  Those numbers were not fitted to the activities being modelled, and
the program's other parameterisation gives a q2 that differs by 0.20 on one Sutherland set, so the
increments are not settled.  ``cmfqsar/learnincrements.py`` fits them to the data: the kernel is a
quadratic form in the increments, so the cross-validated squared error of the model can be
minimised in them directly, with the published values as the starting point and a penalty on the
distance from them.

The question the item asks is not whether fitting lowers the loss it is fitted on -- it must --
but whether what is learned is a property of the atom types, which would transfer to another
series, or of the series, which would make the increments a nuisance parameter.  Four models are
measured on each of the eight Sutherland sets, on their published training and test split:

  published     the increments of the table, which is the model as it stands
  learned       increments fitted on this set's training molecules
  transferred   increments fitted on each *other* set, applied here (the mean over the seven)
  best other    the same, taking the best of the seven, which is the most generous reading

Everything else is held fixed: the same five fields, the same attenuation and ridge grids, the
same ten-fold cross-validation on the training set, the same test set.  The cross-validated q2 of
the learned model is reported as well as its R2 on the test set, because the difference between
them is the size of the selection bias: the increments were fitted on that cross-validation.

Usage
-----
    python benchmarks/run_increments.py                  # fit, then evaluate, then the table
    python benchmarks/run_increments.py --sets GPB THER
    python benchmarks/run_increments.py --report
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

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.learnincrements import (INCREMENT_FIELDS, IncrementFit, basis_from,  # noqa: E402
                                     coverage, fit_from_basis, normalise, other_kernel,
                                     published_increments, spread, type_index, types_in)
from cmfqsar.model import _Learner, cv_predict, r2_score, rmse  # noqa: E402

DATA = os.path.join(ROOT, "examples", "sutherland")
RESULTS = os.path.join(HERE, "results", "increments")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 10
SEED = 0
#: how strongly the fit is held near the published increments.  It cannot be chosen on the
#: cross-validation the increments are fitted on, which always prefers the freest fit, so it is
#: chosen on a part of the training set that the fitting does not see.
RHOS = (0.3, 3.0, 30.0, 300.0)
#: share of the training molecules held back for that choice
VALIDATION = 0.3
#: iterations of the optimiser at most
MAX_ITER = 150


def prepare(name):
    """One set: its molecules, the published split, the increment types and the built fields."""
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    subset = np.asarray(ds.subset, dtype=object)[ok]
    y = y[ok]
    train = np.where(subset != "test")[0]
    test = np.where(subset == "test")[0]
    keys, counts = types_in(mols)
    return {"set": name, "mols": mols, "names": names, "y": y, "train": train, "test": test,
            "keys": keys, "counts": counts, "index": type_index(mols, keys)}


def at_alpha(prep, alpha):
    """The increment basis and the kernel of the other fields, for one attenuation."""
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                         param_scheme="tripos")
    built = [build_fields(m, fs, n, _single=True) for m, n in zip(prep["mols"], prep["names"])]
    basis = basis_from(built, prep["index"], alpha, prep["keys"], prep["counts"])
    return basis, other_kernel(built, fs)


def joint(basis, other, values):
    """The joint kernel of one increment table, over every molecule of the set."""
    K = np.array(other, float)
    for f in range(len(INCREMENT_FIELDS)):
        K = K + normalise(basis.kernel(np.asarray(values)[:, f]))[0]
    return K


def score(K, y, train, test):
    """Best (alpha, lambda) by the training cross-validation, and what it gives on the test set."""
    best = None
    Ktr = K[np.ix_(train, train)]
    for lam in LAMS:
        pred = cv_predict(Ktr, y[train], "krr", {"lam": float(lam)}, folds=FOLDS, seed=SEED)
        q2 = r2_score(y[train], pred)
        if best is None or q2 > best[0]:
            best = (q2, rmse(y[train], pred), float(lam))
    q2, err, lam = best
    learner = _Learner("krr", {"lam": lam}, True).fit(Ktr, y[train])
    out = {"q2": q2, "rmse_cv": err, "lam": lam}
    if len(test):
        pred = learner.predict(K[np.ix_(test, train)])
        out["r2_pred"] = float(r2_score(y[test], pred))
        out["rmse_pred"] = float(rmse(y[test], pred))
    return out


def best_over_alphas(prep, tables, alphas=ALPHAS, note=""):
    """Evaluate several increment tables on one set, each choosing its own (alpha, lambda)."""
    best = {k: None for k in tables}
    for alpha in alphas:
        basis, other = at_alpha(prep, alpha)
        for tag, values in tables.items():
            got = score(joint(basis, other, values), prep["y"], prep["train"], prep["test"])
            got["alpha"] = float(alpha)
            if best[tag] is None or got["q2"] > best[tag]["q2"]:
                best[tag] = got
        del basis, other
        print(f"      alpha {alpha:g} done{note}", flush=True)
    return best


def values_for(keys, table):
    """An increment table as a (len(keys), 3) array, falling back to "other" and then to zero."""
    out = np.zeros((len(keys), len(INCREMENT_FIELDS)))
    other = table.get("other")
    for i, key in enumerate(keys):
        inc = table.get(key, other) or {}
        for f, name in enumerate(INCREMENT_FIELDS):
            out[i, f] = float(inc.get(name, 0.0))
    return out


def as_table(keys, values):
    return {k: {name: float(values[i, f]) for f, name in enumerate(INCREMENT_FIELDS)}
            for i, k in enumerate(keys)}


# ------------------------------------------------------------------- stage one: the fitting


def choose_rho(prep, basis, other, alpha, lam, seed=SEED):
    """Pick the penalty on a part of the training set that the fitting never sees.

    The cross-validated error the increments are fitted on falls whatever the penalty is, so it
    cannot choose one.  The training molecules are split once; the increments are fitted on the
    larger part with each penalty and scored on the smaller one.
    """
    train = prep["train"]
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(train))
    cut = max(3, int(round(VALIDATION * len(train))))
    val, fit_on = train[order[:cut]], train[order[cut:]]
    y = prep["y"]
    rows = []
    for rho in RHOS:
        fit = fit_from_basis([basis] * len(INCREMENT_FIELDS), other, y[fit_on], lam=lam, rho=rho,
                             folds=FOLDS, seed=seed, max_iter=MAX_ITER, use=fit_on)
        K = joint(basis, other, fit.values)
        learner = _Learner("krr", {"lam": lam}, True).fit(K[np.ix_(fit_on, fit_on)], y[fit_on])
        pred = learner.predict(K[np.ix_(val, fit_on)])
        rows.append({"rho": float(rho), "r2_val": float(r2_score(y[val], pred))})
        print(f"      penalty {rho:g}: R2 on the held-back part {rows[-1]['r2_val']:+.3f}",
              flush=True)
    best = max(rows, key=lambda r: r["r2_val"])
    return best["rho"], rows


def fit_set(name, results=RESULTS, force=False):
    """Fit the increments of one set at the operating point its published model chooses."""
    out_path = os.path.join(results, f"fit_{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    prep = prepare(name)
    keys = prep["keys"]
    pub = published_increments(keys)
    print(f"\n=== {name}: {len(prep['y'])} molecules "
          f"({len(prep['train'])} train, {len(prep['test'])} test), {len(keys)} increment types",
          flush=True)
    first = best_over_alphas(prep, {"published": pub})["published"]
    print(f"    published: q2 {first['q2']:+.3f}, alpha {first['alpha']:g}, lam {first['lam']:g}",
          flush=True)
    alpha, lam = first["alpha"], first["lam"]
    basis, other = at_alpha(prep, alpha)
    rho, rho_rows = choose_rho(prep, basis, other, alpha, lam)
    print(f"    penalty {rho:g} chosen", flush=True)
    fit = fit_from_basis([basis] * len(INCREMENT_FIELDS), other, prep["y"][prep["train"]],
                         lam=lam, rho=rho, folds=FOLDS, seed=SEED, max_iter=MAX_ITER,
                         use=prep["train"])
    cos = fit.cosine()
    print(f"    fitted at that point in {fit.iterations} iterations: squared error "
          f"{fit.loss_published:.2f} -> {fit.loss:.2f}; still pointing the same way as the "
          "published increments: " + ", ".join(f"{k} {v:+.2f}" for k, v in cos.items()),
          flush=True)
    res = {"set": name, "n": int(len(prep["y"])), "n_train": int(len(prep["train"])),
           "n_test": int(len(prep["test"])), "keys": keys, "counts": prep["counts"],
           "alpha": alpha, "lam": lam, "rho": rho, "rho_search": rho_rows,
           "values": fit.values.tolist(), "published": pub.tolist(),
           "loss": fit.loss, "loss_published": fit.loss_published,
           "iterations": fit.iterations, "cosine": cos,
           "published_model": first, "seconds": time.time() - t0}
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path} ({res['seconds']:.0f} s)", flush=True)
    return res


# --------------------------------------------------------------- stage two: the evaluation


def evaluate_set(name, fits, results=RESULTS, force=False):
    """Score the published, own and transferred increments on one set."""
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    prep = prepare(name)
    keys = prep["keys"]
    tables = {"published": published_increments(keys),
              "learned": values_for(keys, as_table(fits[name]["keys"],
                                                   np.array(fits[name]["values"])))}
    cov = {}
    for other_set, fit in fits.items():
        if other_set == name:
            continue
        table = as_table(fit["keys"], np.array(fit["values"]))
        tables[f"from {other_set}"] = values_for(keys, table)
        cov[other_set] = coverage(table, prep["mols"])
    print(f"\n=== {name}: {len(tables)} increment tables", flush=True)
    got = best_over_alphas(prep, tables)
    transferred = [v for k, v in got.items() if k.startswith("from ")]
    res = {"set": name, "n": int(len(prep["y"])), "n_train": int(len(prep["train"])),
           "n_test": int(len(prep["test"])), "n_types": len(keys),
           "fields": FIELDS, "alphas": list(ALPHAS), "lams": list(LAMS), "folds": FOLDS,
           "models": got,
           "transfer_mean": {k: float(np.mean([t[k] for t in transferred if k in t]))
                             for k in ("q2", "r2_pred")},
           "transfer_best": {k: float(np.max([t[k] for t in transferred if k in t]))
                             for k in ("q2", "r2_pred")},
           "coverage_mean": float(np.mean([c["exact"] for c in cov.values()])) if cov else 1.0,
           "seconds": time.time() - t0}
    for tag in ("published", "learned"):
        g = got[tag]
        print(f"    {tag:11s} q2 {g['q2']:+.3f}  R2pred {g.get('r2_pred', float('nan')):+.3f}  "
              f"(alpha {g['alpha']:g}, lam {g['lam']:g})", flush=True)
    print(f"    {'transferred':11s} q2 {res['transfer_mean']['q2']:+.3f}  "
          f"R2pred {res['transfer_mean']['r2_pred']:+.3f}  (mean of "
          f"{len(transferred)}; the types of this set covered by "
          f"{res['coverage_mean']:.0%} on average)", flush=True)
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path} ({res['seconds']:.0f} s)", flush=True)
    return res


# ------------------------------------------------------------------------------ the report


def _load(results=RESULTS, prefix=""):
    rows = []
    if not os.path.isdir(results):
        return rows
    for fn in sorted(os.listdir(results)):
        if not fn.endswith(".json"):
            continue
        if prefix and not fn.startswith(prefix):
            continue
        if not prefix and fn.startswith("fit_"):
            continue
        rows.append(json.load(open(os.path.join(results, fn))))
    rows.sort(key=lambda d: d["set"])
    return rows


def report(results=RESULTS):
    rows = _load(results)
    fits = _load(results, "fit_")
    out = ["## Increments fitted to the data, and whether they transfer", "",
           "The hydrophobic field and the two hydrogen-bond fields are built from one increment "
           "per (Sybyl atom type, hydrogen count). `published` takes them from the table, "
           "`learned` fits them on this set's training molecules, and `transferred` applies the "
           "increments fitted on each of the other seven sets. Eight Sutherland sets on their "
           "published training and test split, same five fields, same attenuation and ridge "
           "grids, same ten-fold cross-validation.", ""]
    if rows:
        out += ["| set | n (train/test) | types | q2 published | q2 learned | R2pred published | "
                "R2pred learned | R2pred transferred | R2pred best other |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for d in rows:
            m = d["models"]
            out.append(
                f"| {d['set']} | {d['n_train']}/{d['n_test']} | {d['n_types']} | "
                f"{m['published']['q2']:+.3f} | {m['learned']['q2']:+.3f} | "
                f"{m['published'].get('r2_pred', float('nan')):+.3f} | "
                f"{m['learned'].get('r2_pred', float('nan')):+.3f} | "
                f"{d['transfer_mean']['r2_pred']:+.3f} | {d['transfer_best']['r2_pred']:+.3f} |")

        def mean(key, sub=None):
            if sub:
                return float(np.mean([d[key][sub] for d in rows]))
            return float(np.mean([d["models"][key]["q2"] for d in rows]))

        def mean_pred(tag):
            return float(np.mean([d["models"][tag].get("r2_pred", np.nan) for d in rows]))

        out.append(f"| **mean** | | | **{mean('published'):+.3f}** | {mean('learned'):+.3f} | "
                   f"{mean_pred('published'):+.3f} | {mean_pred('learned'):+.3f} | "
                   f"{mean('transfer_mean', 'r2_pred'):+.3f} | "
                   f"{mean('transfer_best', 'r2_pred'):+.3f} |")
        out += ["",
                f"Fitting lowers the cross-validated error it is fitted on: q2 goes from "
                f"{mean('published'):+.3f} to {mean('learned'):+.3f} on the training set. On the "
                f"test set, which the fitting never saw, R2pred goes from "
                f"{mean_pred('published'):+.3f} to {mean_pred('learned'):+.3f}. The difference "
                "between those two movements is the selection bias of fitting the increments on "
                "the cross-validation that reports the q2.", "",
                f"Increments fitted on another set give {mean('transfer_mean', 'r2_pred'):+.3f} "
                f"on average and {mean('transfer_best', 'r2_pred'):+.3f} for the best of the "
                "seven, against "
                f"{mean_pred('published'):+.3f} for the published table.", ""]
    if fits:
        out += ["### How far the fitted increments move, and whether they agree with each other",
                "",
                "Cosine between the fitted and the published increments of a field, per set: one "
                "means the fit only rescaled them, zero means it learned something unrelated.", "",
                "| set | penalty | hydrophobic | hba | hbd | "
                "squared error, published -> fitted |",
                "| --- | ---: | ---: | ---: | ---: | --- |"]
        for d in fits:
            c = d["cosine"]
            out.append(f"| {d['set']} | {d.get('rho', float('nan')):g} | "
                       f"{c['hydrophobic']:+.2f} | {c['hba']:+.2f} | "
                       f"{c['hbd']:+.2f} | {d['loss_published']:.1f} -> {d['loss']:.1f} |")
        made = [IncrementFit(keys=d["keys"], values=np.array(d["values"]),
                             published=np.array(d["published"]), alpha=d["alpha"], lam=d["lam"],
                             rho=d["rho"], loss=d["loss"], loss_published=d["loss_published"],
                             iterations=d["iterations"]) for d in fits]
        agree = spread(made)
        out += ["",
                "Agreement between the increments fitted on different sets, as the mean cosine "
                "over every pair of sets and the types they have in common: "
                + ", ".join(f"{k} {v:+.2f}" for k, v in agree.items())
                + ". For comparison, every fit starts from the published table, whose cosine "
                  "with itself is +1.00.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fit-only", action="store_true", help="stage one only")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with open(a.report, "w") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    os.makedirs(RESULTS, exist_ok=True)
    fits = {}
    for name in a.sets:
        fits[name] = fit_set(name, force=a.force)
    if not a.fit_only:
        for name in a.sets:
            evaluate_set(name, fits, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
