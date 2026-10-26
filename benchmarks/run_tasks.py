#!/usr/bin/env python
"""
Seven endpoints on one kernel: what a free-form task covariance adds, and whether an endpoint
with no measurements at all can be predicted from the related ones (plan item E1).

``benchmarks/run_multitask.py`` fits two endpoints with one correlation between them.  With two
endpoints that is an empty assumption.  With seven it states that every pair of endpoints is
related to the same degree, which is almost never true.  This script measures the two more general
forms in ``cmfqsar/multitask.py`` on a set that has seven real endpoints and descriptors for
them.

The data
--------
``examples/cyclodextrin/`` holds association constants of guest molecules with seven
cyclodextrin hosts: 2671 measurements over 1669 distinct guests, 650 of which were measured
against more than one host.  The hosts are the endpoints.  The reason this set suits E1 is that
the *host* is itself a molecule, so its descriptors are computed from a structure rather than
taken from a label: the cavity size (the number of glucose units), the substituent (methyl,
hydroxypropyl, sulfobutyl), the charge and the lipophilicity.  An endpoint with no measurements is
then a host that nobody has titrated, and predicting it means interpolating between the hosts that
were titrated.

One kernel is built over the union of the guests: one ETKDG conformer each and the
rotation-invariant field autocorrelation.  ``run_cyclodextrin.py`` found that kernel to be the
best of the three available on this data, and it needs no common frame, which this set cannot
provide.  Every row of every table below uses that same kernel, so the rows differ only in the
task covariance.

What is measured
----------------
**Sharing** (10-fold cross-validation over *guests*, never over measurements):

    independent     every host fitted alone on the shared kernel: the floor
    one rho, CV     the covariance (1-rho) I + rho 11', rho chosen by cross-validation
    one rho, evid   the same, chosen by the marginal likelihood
    free rank 1     L L' + diag(d) with L a column, chosen by the marginal likelihood
    free rank 2     the same at rank 2
    descriptors     T(t,s) = k(z_t, z_s) from the host descriptors

**Prediction of an unmeasured endpoint**: each host in turn is removed entirely -- the whole
endpoint, not a fold of its guests -- and predicted from the other six through the descriptor
covariance, with a mean that is a linear function of the descriptors so that it can be evaluated
for the removed host.  Two reference values that need no model are reported beside it: the mean of
the measured hosts' values for the same guest, where another host measured it, and the
single-endpoint model of the host with the most similar descriptors.

Usage
-----
    python benchmarks/run_tasks.py                      # both stages, then the table
    python benchmarks/run_tasks.py --stage share
    python benchmarks/run_tasks.py --report
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
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402

from cmfqsar.align import embed_3d  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               feature_matrix)
from cmfqsar.model import r2_score, rmse  # noqa: E402
from cmfqsar.multitask import (Observations, cv_predict, descriptor_covariance,  # noqa: E402
                               evidence_rho, fit, fit_covariance, log_evidence, score_by_task,
                               search, task_covariance)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "tasks")
DATA = os.path.join(ROOT, "examples", "cyclodextrin")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0, 5.0)
LAMS = (1e-2, 0.1, 1.0)
#: one shared correlation is positive definite only above -1/(tasks - 1), which for seven tasks
#: is -0.167: a grid that reaches further down would be refused, and the floor is near zero anyway
RHOS = (-0.15, -0.1, 0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
FOLDS = 10
SEED = 0
#: host -> (file, printed name); the order is the one the tables use, largest set first
HOSTS = (("beta", "beta_cyclodextrin.csv"),
         ("alpha", "alpha_cyclodextrin.csv"),
         ("gamma", "gamma_cyclodextrin.csv"),
         ("hp_beta", "hydroxypropyl_beta_cyclodextrin.csv"),
         ("sbe_beta", "sulfobutylether_beta_cyclodextrin.csv"),
         ("methyl_beta", "methyl_beta_cyclodextrin.csv"),
         ("dimethyl_beta", "dimethyl_beta_cyclodextrin.csv"))
#: the descriptors of a host, computed from its own structure
DESCRIPTORS = ("glucose units", "clogP", "formal charge", "hydroxyls", "heavy atoms")
#: the columns of the mean: an intercept and two descriptors. The design matrix must have far
#: fewer columns than there are endpoints. With one column per endpoint the mean of an unmeasured
#: endpoint cannot be estimated, for the same reason its covariance row cannot be.
MEAN_COLUMNS = (0, 1)


# --------------------------------------------------------------------------------- the data


def host_rows(path):
    """(canonical smiles, log K) for one host table."""
    out = []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            m = Chem.MolFromSmiles(r["smiles"])
            if m is None:
                continue
            try:
                y = float(r["activity"])
            except (TypeError, ValueError):
                continue
            out.append((Chem.MolToSmiles(m), y))
    return out


def host_descriptors():
    """One row of descriptors per host, from the host structure in ``hosts.csv``."""
    smiles = {}
    with io.open(os.path.join(DATA, "hosts.csv"), encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if r.get("file"):
                smiles[r["file"]] = r["host_smiles"]
    Z = []
    for _key, fname in HOSTS:
        m = Chem.MolFromSmiles(smiles[fname])
        if m is None:
            raise SystemExit(f"the host structure of {fname} does not parse")
        rings = m.GetRingInfo().NumRings()
        hydroxyl = len(m.GetSubstructMatches(Chem.MolFromSmarts("[OX2H]")))
        Z.append([float(rings - 1),                      # glucose units: 6 (alpha) to 8 (gamma)
                  float(Crippen.MolLogP(m)),
                  float(Chem.GetFormalCharge(m)),
                  float(hydroxyl),
                  float(m.GetNumHeavyAtoms())])
    return np.asarray(Z, float)


def build(max_guests=0):
    """The shared kernel over the union of the guests, and the observation table."""
    t0 = time.time()
    per_host = [host_rows(os.path.join(DATA, f)) for _k, f in HOSTS]
    order, index = [], {}
    for rows in per_host:
        for smi, _y in rows:
            if smi not in index:
                index[smi] = len(order)
                order.append(smi)
    if max_guests and len(order) > max_guests:
        keep = set(order[:max_guests])
        order = [s for s in order if s in keep]
        index = {s: i for i, s in enumerate(order)}
    kept = sum(1 for rows in per_host for smi, _y in rows if smi in index)
    print(f"{len(order)} distinct guests, {kept} measurements over {len(HOSTS)} hosts",
          flush=True)

    Y = np.full((len(order), len(HOSTS)), np.nan)
    for t, rows in enumerate(per_host):
        for smi, y in rows:
            if smi in index:
                i = index[smi]
                # a guest measured twice against one host: keep the mean, as the table's own
                # `activity` column already is for repeated literature values
                Y[i, t] = y if not np.isfinite(Y[i, t]) else 0.5 * (Y[i, t] + y)

    mols = []
    for k, smi in enumerate(order):
        m = Chem.MolFromSmiles(smi)
        m = embed_3d(m, seed=k)
        m.SetProp("_Name", f"g{k}")
        mols.append(m)
        if (k + 1) % 250 == 0:
            print(f"    {k + 1}/{len(order)} guests embedded ({time.time() - t0:.0f} s)",
                  flush=True)
    kernels = {}
    spec = AutocorrelationSpec()
    for alpha in ALPHAS:
        fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, m.GetProp("_Name")) for m in mols]
        kernels[alpha] = autocorrelation_kernel(feature_matrix(mf, spec), n_bins=spec.n_bins)
        print(f"    alpha {alpha:g}: kernel {kernels[alpha].shape} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return Y, kernels, host_descriptors()


# ------------------------------------------------------------------------------- the stages


def choose_alpha(Y, kernels):
    """One attenuation for every row of every table, chosen with the independent model.

    Choosing it with the model that shares nothing is neutral between the covariances: the
    independent model has no covariance, so the value cannot be adapted to one of them.
    """
    obs = Observations.from_columns(Y, names=[k for k, _f in HOSTS])
    best = None
    for alpha, K in kernels.items():
        for lam in LAMS:
            pred = cv_predict(K, obs, rho=0.0, lam=lam, folds=FOLDS, seed=SEED)
            per = score_by_task(obs, pred)
            s = float(np.nanmean([p["q2"] for p in per]))
            if best is None or s > best[0]:
                best = (s, float(alpha), float(lam), per)
    print(f"  alpha {best[1]:g}, lambda {best[2]:g}: mean q2 {best[0]:+.3f} independent",
          flush=True)
    return best


def row(name, K, obs, lam, T, note=""):
    """One line of the sharing table: out-of-fold q2 per host under one covariance."""
    t0 = time.time()
    pred = cv_predict(K, obs, rho=0.0, lam=lam, folds=FOLDS, seed=SEED, T=T)
    per = score_by_task(obs, pred)
    out = {"name": name, "lam": float(lam), "note": note,
           "mean_q2": float(np.nanmean([p["q2"] for p in per])),
           "per_task": per, "seconds": round(time.time() - t0, 1)}
    if T is not None:
        out["T"] = np.asarray(T, float).tolist()
        out["evidence"] = float(log_evidence(K, obs, T, lam))
    print(f"  {name:16s} mean q2 {out['mean_q2']:+.3f}   "
          + "  ".join(f"{p['task']} {p['q2']:+.2f}" for p in per), flush=True)
    return out


def stage_share(Y, kernels, Z, force=False):
    path = os.path.join(RESULTS, "share.json")
    if os.path.exists(path) and not force:
        print("share: already done")
        return json.load(open(path))
    names = [k for k, _f in HOSTS]
    obs = Observations.from_columns(Y, names=names)
    score, alpha, lam0, _per = choose_alpha(Y, kernels)
    K = kernels[alpha]
    n_t = obs.n_tasks
    rows = [row("independent", K, obs, lam0, task_covariance(n_t, 0.0), "rho = 0")]

    rho, lam, cv_score, _ = search(K, obs, rhos=RHOS, lams=LAMS, folds=FOLDS, seed=SEED)
    rows.append(row("one rho, CV", K, obs, lam, task_covariance(n_t, rho), f"rho = {rho:+.2f}"))

    one = evidence_rho(K, obs, lams=LAMS)
    rows.append(row("one rho, evidence", K, obs, one.lam, one.T, one.note))

    for rank in (1, 2):
        got = fit_covariance(K, obs, rank=rank, rhos=(0.3, 0.8), lams=(0.1,), maxiter=40)
        rows.append(row(f"free rank {rank}", K, obs, got.lam, got.T, got.note))
        print("    " + got.describe(names).replace("\n", "\n    "), flush=True)

    dc = descriptor_covariance(K, obs, Z, lams=LAMS)
    rows.append(row("descriptors", K, obs, dc.lam, dc.T,
                    f"{dc.kind}, length scale {dc.length_scale:g}, nugget {dc.nugget:g}"))

    out = {"alpha": alpha, "lam_independent": lam0, "folds": FOLDS, "seed": SEED,
           "hosts": names, "counts": obs.counts(), "n_guests": int(Y.shape[0]),
           "descriptors": list(DESCRIPTORS), "Z": Z.tolist(), "rows": rows}
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def stage_cold(Y, kernels, Z, alpha=None, force=False):
    """Every host removed in turn and predicted from the other six."""
    path = os.path.join(RESULTS, "cold.json")
    if os.path.exists(path) and not force:
        print("cold: already done")
        return json.load(open(path))
    names = [k for k, _f in HOSTS]
    if alpha is None:
        alpha = choose_alpha(Y, kernels)[1]
    K = kernels[alpha]
    out = {"alpha": float(alpha), "hosts": names, "mean_columns": list(MEAN_COLUMNS), "held": []}
    for t, name in enumerate(names):
        keep = [s for s in range(len(names)) if s != t]
        truth = Y[:, t]
        ok = np.where(np.isfinite(truth))[0]
        if len(ok) < 10:
            continue
        Zk = np.c_[np.ones(len(keep)), Z[np.ix_(keep, MEAN_COLUMNS)]]
        z_new = np.r_[1.0, Z[t, list(MEAN_COLUMNS)]]
        obs = Observations.from_columns(Y[:, keep], names=[names[s] for s in keep],
                                        task_features=Zk)
        dc = descriptor_covariance(K, obs, Z[keep], lams=LAMS)
        f = fit(K, obs, lam=dc.lam, T=dc.T)
        pred = f.predict_new_task(K, dc.row_for(Z[t]), features=z_new)[ok]

        # baseline 1: the mean of the hosts that were measured, for guests that have one
        others = np.nanmean(Y[np.ix_(ok, keep)], axis=1)
        have = np.isfinite(others)
        # baseline 2: the single-task model of the nearest host in descriptor space
        d = np.linalg.norm((Z[keep] - Z[t]) / np.clip(Z.std(axis=0), 1e-9, None), axis=1)
        near = keep[int(np.argmin(d))]
        single = Observations.from_columns(Y[:, [near]], names=[names[near]])
        fs_ = fit(K, single, rho=0.0, lam=dc.lam)
        near_pred = fs_.predict(K, 0)[ok]

        rec = {"host": name, "n": int(len(ok)), "nearest": names[near],
               "length_scale": dc.length_scale, "nugget": dc.nugget, "lam": dc.lam,
               "row": dc.row_for(Z[t]).tolist(),
               "descriptor": _cold_scores(truth[ok], pred),
               "nearest_host": _cold_scores(truth[ok], near_pred),
               "measured_mean": _cold_scores(truth[ok][have], others[have]) if have.any() else {},
               "n_with_another_host": int(have.sum())}
        out["held"].append(rec)
        print(f"  {name:14s} n {rec['n']:5d}  descriptor q2 "
              f"{rec['descriptor']['q2']:+.3f} rho {rec['descriptor']['spearman']:+.3f}   "
              f"nearest ({names[near]}) q2 {rec['nearest_host']['q2']:+.3f} "
              f"rho {rec['nearest_host']['spearman']:+.3f}", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def _cold_scores(truth, pred):
    from scipy.stats import spearmanr
    truth = np.asarray(truth, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(truth) & np.isfinite(pred)
    if ok.sum() < 5:
        return {"n": int(ok.sum()), "q2": float("nan"), "rmse": float("nan"),
                "spearman": float("nan")}
    return {"n": int(ok.sum()), "q2": float(r2_score(truth[ok], pred[ok])),
            "rmse": float(rmse(truth[ok], pred[ok])),
            "spearman": float(spearmanr(truth[ok], pred[ok]).statistic)}


# -------------------------------------------------------------------------------- the report


def report(results=RESULTS):
    share_p = os.path.join(results, "share.json")
    cold_p = os.path.join(results, "cold.json")
    if not os.path.exists(share_p):
        return "No task results yet; run `python benchmarks/run_tasks.py`."
    sh = json.load(open(share_p))
    lines = ["# Seven endpoints on one kernel", "",
             f"`examples/cyclodextrin/`: {sh['n_guests']} distinct guests measured against the "
             f"{len(sh['hosts'])} cyclodextrin hosts, "
             f"{sum(sh['counts'])} measurements in all "
             f"({', '.join(f'{h} {c}' for h, c in zip(sh['hosts'], sh['counts']))}). One kernel "
             f"for every row -- the rotation-invariant field autocorrelation at alpha "
             f"{sh['alpha']:g}, chosen by the independent model, which has no covariance to "
             f"favour -- and {sh['folds']}-fold cross-validation over **guests**, so that no "
             f"guest has one host in the training set and another in the test set.", "",
             "| task covariance | mean q² | " + " | ".join(sh["hosts"]) + " | what it is |",
             "|---|---:|" + "---:|" * len(sh["hosts"]) + "---|"]
    for r in sh["rows"]:
        by = {p["task"]: p["q2"] for p in r["per_task"]}
        cells = [f"{by.get(h, float('nan')):+.3f}" for h in sh["hosts"]]
        lines.append(f"| {r['name']} | **{r['mean_q2']:+.3f}** | " + " | ".join(cells)
                     + f" | {r['note']} |")
    base = next((r for r in sh["rows"] if r["name"] == "independent"), None)
    best = max(sh["rows"], key=lambda r: r["mean_q2"])
    lines += ["", f"**Sharing is worth {best['mean_q2'] - base['mean_q2']:+.3f} mean q²** "
                  f"({base['mean_q2']:+.3f} independent against {best['mean_q2']:+.3f} for "
                  f"{best['name']}).", ""]
    ev = [(r["name"], r.get("evidence")) for r in sh["rows"] if r.get("evidence") is not None]
    if ev:
        lines += ["The evidence ranks them in this order, which is the ranking it was asked for "
                  "and not the one that matters:", "",
                  "| task covariance | log evidence | mean q² |", "|---|---:|---:|"]
        for name, v in sorted(ev, key=lambda p: -p[1]):
            q = next(r["mean_q2"] for r in sh["rows"] if r["name"] == name)
            lines.append(f"| {name} | {v:+.1f} | {q:+.3f} |")
        lines.append("")
    if os.path.exists(cold_p):
        cold = json.load(open(cold_p))
        lines += ["## An endpoint with no measurements", "",
                  "Each host removed entirely -- the whole endpoint, not a fold of its guests -- "
                  "and predicted from the other six through the descriptor covariance, with a "
                  "mean linear in the descriptors so that there is something to extrapolate. "
                  "Spearman ρ is the column to read: the absolute scale of an endpoint nobody "
                  "measured is not recoverable, and a screen ranks.", "",
                  "| host removed | n | nearest host by descriptors | ρ, descriptor covariance | "
                  "ρ, nearest host alone | ρ, mean of the measured hosts | q², descriptor |",
                  "|---|---:|---|---:|---:|---:|---:|"]
        for r in cold["held"]:
            mm = r.get("measured_mean") or {}
            lines.append(f"| {r['host']} | {r['n']} | {r['nearest']} | "
                         f"{r['descriptor']['spearman']:+.3f} | "
                         f"{r['nearest_host']['spearman']:+.3f} | "
                         f"{mm.get('spearman', float('nan')):+.3f} "
                         f"({mm.get('n', 0)} guests) | {r['descriptor']['q2']:+.3f} |")
        d = float(np.mean([r["descriptor"]["spearman"] for r in cold["held"]]))
        n_ = float(np.mean([r["nearest_host"]["spearman"] for r in cold["held"]]))
        lines += ["", f"Mean ρ over the {len(cold['held'])} hosts: **{d:+.3f}** through the "
                      f"descriptor covariance against **{n_:+.3f}** for the nearest host's own "
                      f"model.", ""]
    text = "\n".join(lines) + "\n"
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", nargs="*", default=["share", "cold"],
                    choices=["share", "cold"])
    ap.add_argument("--max-guests", type=int, default=0, help="for a quick trial run")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    Y, kernels, Z = build(a.max_guests)
    alpha = None
    if "share" in a.stage:
        sh = stage_share(Y, kernels, Z, force=a.force)
        alpha = sh["alpha"]
    if "cold" in a.stage:
        stage_cold(Y, kernels, Z, alpha=alpha, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
