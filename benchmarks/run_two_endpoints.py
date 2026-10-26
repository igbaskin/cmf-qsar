#!/usr/bin/env python
"""
Two cavities, one model: does sharing help, and what does the selectivity say? (plan item SC1)

``examples/cyclodextrin`` holds association constants of the same guests with different hosts.
496 guests appear in both the alpha-cyclodextrin and the beta-cyclodextrin tables, so the two
cavities are the first two-endpoint problem in the repository that needs no new data.  The
question is the one a two-task model is for: does a molecule measured against one cavity help the
model of the other, and is the *difference* of the two -- the selectivity, which is what a
separation or a formulation depends on -- predictable at all?

What is measured
----------------
One kernel over the guests (the rotation-invariant field autocorrelation, which `run_cyclodextrin`
found to be the best of the three on these tables), one correlation between the endpoints chosen
by the marginal likelihood, and cross-validation that holds out **whole molecules**, so a guest's
value for one cavity is never in the training set while its value for the other is being
predicted.

  shared        both endpoints in one fit, tied by the fitted correlation
  alone         the same code with the correlation set to zero, which is exactly the two
                independent single-task fits
  selectivity   the difference of the two endpoints, scored on the guests measured against both

Every difference comes with a bootstrap interval over the molecules (rule 4).  The selectivity is
then read: against the size of the guest, which is the known difference between the two cavities,
and over the channels of the kernel, which is what an invariant model has instead of a co-field.

Usage
-----
    python benchmarks/run_two_endpoints.py
    python benchmarks/run_two_endpoints.py --pair beta gamma
    python benchmarks/run_two_endpoints.py --report
"""
from __future__ import annotations

import argparse
import csv
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

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar import multitask as MT  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, channel_coefficients, field_pairs  # noqa: E402
from cmfqsar.model import CMFModel, r2_score  # noqa: E402

DATA = os.path.join(ROOT, "examples", "cyclodextrin")
RESULTS = os.path.join(HERE, "results", "two_endpoints")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
LAMS = (1e-2, 0.1, 1.0)
FOLDS = 10
SEED = 0
RUNS = 2000               # bootstrap resamples
FILES = {"alpha": "alpha_cyclodextrin.csv", "beta": "beta_cyclodextrin.csv",
         "gamma": "gamma_cyclodextrin.csv"}


def read(name):
    """(canonical smiles -> (logK, name)) for one host table."""
    out = {}
    with io.open(os.path.join(DATA, FILES[name]), encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            m = Chem.MolFromSmiles(r.get("smiles", ""))
            if m is None:
                continue
            try:
                y = float(r["activity"])
            except (TypeError, ValueError, KeyError):
                continue
            out.setdefault(Chem.MolToSmiles(m), (y, r.get("name", "")))
    return out


def union(first, second):
    """Every guest of either table, with both numbers where both exist."""
    a, b = read(first), read(second)
    keys = sorted(set(a) | set(b))
    rows = []
    for k in keys:
        name = (a.get(k) or b.get(k))[1]
        rows.append({"smiles": k, "name": name,
                     "y1": a[k][0] if k in a else float("nan"),
                     "y2": b[k][0] if k in b else float("nan")})
    return rows


def build(rows, tmp_dir):
    """Embed the guests once, through the loader the rest of the program uses."""
    path = os.path.join(tmp_dir, "_two_endpoints_guests.csv")
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["smiles", "name", "activity"])
        w.writeheader()
        for r in rows:
            y = r["y1"] if np.isfinite(r["y1"]) else r["y2"]
            w.writerow({"smiles": r["smiles"], "name": r["name"], "activity": f"{y:.4f}"})
    ds = load_dataset(path)
    os.remove(path)
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(ds.mols, ds.names)]
    return ds, fs, mf


def bootstrap_difference(truth, a, b, runs=RUNS, seed=SEED):
    """(difference in q2, low, high) for prediction `a` against `b`, resampling the molecules."""
    truth = np.asarray(truth, float)
    ok = np.isfinite(truth) & np.isfinite(a) & np.isfinite(b)
    truth, a, b = truth[ok], np.asarray(a)[ok], np.asarray(b)[ok]
    if len(truth) < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(runs)):
        idx = rng.randint(0, len(truth), len(truth))
        if np.std(truth[idx]) <= 0:
            continue
        out.append(r2_score(truth[idx], a[idx]) - r2_score(truth[idx], b[idx]))
    return (float(r2_score(truth, a) - r2_score(truth, b)),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def selectivity_reading(model, mf, fit, rows, heavy):
    """What the selectivity rests on: the size of the guest, and the channels of the kernel."""
    spec = AutocorrelationSpec(d_max=model.ac_d_max, step=model.ac_sigma, sigma=model.ac_sigma)
    scale = model.ac_column_scale
    out = {}
    for tag, dual in (("endpoint 1", fit.coef[:, 0]), ("endpoint 2", fit.coef[:, 1]),
                      ("selectivity", fit.selectivity_coefficients(0, 1))):
        c = channel_coefficients(model.ac_train_, scale, np.asarray(dual, float))
        out[tag] = _weight_by_distance(c * scale, spec, mf[0].weights.shape[1])
    return out


def _weight_by_distance(gamma, spec, n_fields):
    """Where along the distance axis the weight sits, per pair of field types."""
    pairs = field_pairs(n_fields)
    n_bins = spec.n_bins
    g = np.asarray(gamma, float)[:len(pairs) * n_bins].reshape(len(pairs), n_bins)
    centres = spec.centres
    out = {}
    for i, (f, j) in enumerate(pairs):
        w = np.abs(g[i])
        total = float(w.sum())
        out[f"{f}-{j}"] = {"weight": total,
                           "mean_distance": float((w * centres).sum() / total) if total else 0.0}
    return out


def run_pair(first, second, results=RESULTS, force=False):
    tag = f"{first}_{second}"
    out_path = os.path.join(results, f"{tag}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    rows = union(first, second)
    both = sum(1 for r in rows if np.isfinite(r["y1"]) and np.isfinite(r["y2"]))
    print(f"\n=== {first} and {second}: {len(rows)} guests, {both} measured against both",
          flush=True)
    ds, fs, mf = build(rows, results)
    keep = [i for i, m in enumerate(ds.mols) if m is not None and m.GetNumAtoms() > 0]
    rows = [rows[i] for i in keep]
    mf = [mf[i] for i in keep]
    Y = np.array([[r["y1"], r["y2"]] for r in rows], float)
    heavy = np.array([m.GetNumHeavyAtoms() for m in [ds.mols[i] for i in keep]], float)
    obs = MT.Observations.from_columns(Y, names=(first, second))
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation")
    K = model.training_kernel(table)
    print(f"    kernel {K.shape} ({time.time() - t0:.0f} s)", flush=True)
    best = None
    for lam in LAMS:
        f = MT.evidence_rho(K, obs, lams=[lam])
        if best is None or f.evidence > best.evidence:
            best = f
    rho = float(best.correlations[0, 1])
    lam = float(best.lam)
    print(f"    correlation {rho:+.3f} and lambda {lam:g}, chosen by the evidence "
          f"({best.note})", flush=True)
    T = MT.task_covariance(2, rho)
    pred_shared = MT.cv_predict(K, obs, lam=lam, T=T, folds=FOLDS, seed=SEED)
    pred_alone = MT.cv_predict(K, obs, rho=0.0, lam=lam, folds=FOLDS, seed=SEED)
    shared = MT.score_by_task(obs, pred_shared)
    alone = MT.score_by_task(obs, pred_alone)
    res = {"pair": [first, second], "tag": tag, "n": len(rows), "n_both": both,
           "rho": rho, "lam": lam, "folds": FOLDS, "fields": FIELDS,
           "evidence": float(best.evidence), "tasks": []}
    for t in range(2):
        m = obs.task == t
        d, lo, hi = bootstrap_difference(obs.y[m], pred_shared[m], pred_alone[m])
        res["tasks"].append({"task": shared[t]["task"], "n": shared[t]["n"],
                             "q2_shared": shared[t]["q2"], "q2_alone": alone[t]["q2"],
                             "difference": d, "low": lo, "high": hi})
        r = res["tasks"][-1]
        print(f"    {r['task']:8s} n {r['n']:5d}  q2 {r['q2_shared']:+.3f} shared, "
              f"{r['q2_alone']:+.3f} alone, difference {r['difference']:+.3f} "
              f"[{r['low']:+.3f}, {r['high']:+.3f}]", flush=True)
    sel = MT.score_selectivity(K, obs, 0, 1, rho=rho, lam=lam, folds=FOLDS, seed=SEED)
    res["selectivity"] = {"n": sel["n"], "q2": sel["q2"], "rmse_cv": sel["rmse_cv"]}
    print(f"    selectivity ({first} - {second}) on {sel['n']} guests: q2 {sel['q2']:+.3f}, "
          f"RMSE {sel['rmse_cv']:.3f}", flush=True)
    # does the selectivity follow the size of the guest, as the two cavities say it should?
    mols = sel["molecules"]
    size = heavy[mols]
    res["size"] = {
        "n": int(len(mols)),
        "observed": float(np.corrcoef(size, sel["observed"])[0, 1]),
        "predicted": float(np.corrcoef(size, sel["predicted"][np.isfinite(sel["predicted"])])[0, 1])
        if np.isfinite(sel["predicted"]).all() else float("nan"),
        "heavy_atoms_median": float(np.median(size)),
    }
    print(f"    selectivity against the size of the guest: observed "
          f"{res['size']['observed']:+.3f}, predicted {res['size']['predicted']:+.3f} "
          f"(Pearson, {res['size']['n']} guests)", flush=True)
    fit = MT.fit(K, obs, lam=lam, T=T)
    res["channels"] = selectivity_reading(model, mf, fit, rows, heavy)
    steric = [k for k in res["channels"]["selectivity"] if k == "1-1"]
    if steric:
        a = res["channels"]["endpoint 1"]["1-1"]["mean_distance"]
        s = res["channels"]["selectivity"]["1-1"]["mean_distance"]
        res["steric_distance"] = {"endpoint": a, "selectivity": s}
        print(f"    steric channel: the endpoint's weight sits at {a:.2f} A on average, the "
              f"selectivity's at {s:.2f} A", flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path} ({res['seconds']:.0f} s)", flush=True)
    return res


def report(results=RESULTS):
    rows = []
    if os.path.isdir(results):
        for fn in sorted(os.listdir(results)):
            if fn.endswith(".json"):
                rows.append(json.load(open(os.path.join(results, fn))))
    out = ["## Two cavities in one model", "",
           "Guests of two cyclodextrins, joined on the structure of the guest: a molecule "
           "measured against one cavity and not the other is still a training molecule for both "
           "endpoints. One kernel (the rotation-invariant autocorrelation), one correlation "
           "chosen by the marginal likelihood, ten-fold cross-validation holding out whole "
           "molecules. `alone` is the same code with the correlation set to zero, which is the "
           "two independent single-task fits; the interval is a bootstrap over the molecules.", ""]
    if rows:
        out += ["| pair | guests | both | correlation | endpoint | q2 shared | q2 alone | "
                "difference |", "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |"]
        for d in rows:
            for i, t in enumerate(d["tasks"]):
                head = (f"| {d['pair'][0]} and {d['pair'][1]} | {d['n']} | {d['n_both']} | "
                        f"{d['rho']:+.3f} " if i == 0 else "| | | | ")
                out.append(head + f"| {t['task']} | {t['q2_shared']:+.3f} | {t['q2_alone']:+.3f} "
                                  f"| {t['difference']:+.3f} [{t['low']:+.3f}, {t['high']:+.3f}] |")
        out += ["", "| pair | selectivity q2 | guests | correlation with the guest's size, "
                    "observed | predicted |", "| --- | ---: | ---: | ---: | ---: |"]
        for d in rows:
            s, z = d["selectivity"], d["size"]
            out.append(f"| {d['pair'][0]} - {d['pair'][1]} | {s['q2']:+.3f} | {s['n']} | "
                       f"{z['observed']:+.3f} | {z['predicted']:+.3f} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", default=None,
                    metavar=("FIRST", "SECOND"), help="a pair of hosts (default: beta and alpha)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="?", const="-", default=None, metavar="PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with io.open(a.report, "w", encoding="utf-8") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    os.makedirs(RESULTS, exist_ok=True)
    for first, second in (a.pair or [["beta", "alpha"]]):
        run_pair(first, second, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
