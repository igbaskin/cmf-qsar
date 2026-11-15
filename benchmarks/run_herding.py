#!/usr/bin/env python
"""
Kernel herding against the usual ways of choosing a few molecules from many.

:mod:`cmfqsar.herding` chooses a subset whose mean embedding (for the linear CMF kernel, the
mean field) is close to the mean embedding of the whole set. This script asks whether that is
worth anything in the three places where the program chooses a subset.

Stage ``subsets``: which molecules to make or measure first
    Each of the eight Sutherland sets is split at random into a pool (70 %) and a test set
    (30 %), ten times. From the pool, m = 10, 20 or 40 molecules are chosen *without looking at
    the activity*, a model is fitted on them alone (alpha and lambda by cross-validation on the m
    molecules) and it predicts the test set. The rules: herding on the model's kernel, herding on
    a Gaussian of the field distance, max-min, k-medoids, pivoted Cholesky and random choice
    (five draws per split). The model on the whole pool is the upper reference.

Stage ``landmarks``: which molecules to use as Nystroem landmarks
    On each whole set, with the attenuation and ridge chosen by exact cross-validation, the
    kernel is replaced by its Nystroem approximation on m = 5, 10, 20 or 40 landmark molecules,
    chosen on that same kernel.
    Measured: the relative error of the approximation and the q2 of kernel ridge regression on
    it (10-fold), for the same rules (random: ten draws).

Stage ``members``: which members of an ensemble to keep
    The 62 electrolytes of item BT5, each the mean of 15 or 16 solvation shells. Plan item MS5
    measured the alignment-free model on the k shells of largest weight (k = 1, 2, 4, 8, 16).
    Here the k shells are chosen by herding, and by its exact greedy form, on the field kernel
    of the shells of one electrolyte, with the given weights as the target, and weighted by
    :func:`cmfqsar.herding.quadrature_weights`. References: the k heaviest (the stored records of
    ``benchmarks/results/ensembles/members.json``), k shells drawn at random (three draws for
    k = 4), and all 16. The 4 heaviest are run again as the control of the stored records.

Usage
-----
    python benchmarks/run_herding.py                     # every stage
    python benchmarks/run_herding.py --stage subsets
    python benchmarks/run_herding.py --report
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
from cmfqsar.herding import (distance2_matrix, mmd2_subset, pick,  # noqa: E402
                             quadrature_weights)
from cmfqsar.locallarge import nystrom_kernel  # noqa: E402
from cmfqsar.model import cv_predict, krr_fit  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "herding")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0)
SELECT_ALPHA = 0.3                 # the kernel the subset is chosen on; it never sees the activity
SIZES = (10, 20, 40)
LANDMARKS = (5, 10, 20, 40)
REPEATS = 10
RANDOM_DRAWS = 5
RULES = ("herding", "herding-gauss", "maxmin", "kmedoids", "pivoted_cholesky", "random")


def _gauss_layer(K):
    D2 = distance2_matrix(K)
    off = D2[~np.eye(len(D2), dtype=bool)]
    off = off[off > 1e-12]
    return np.exp(-D2 / (np.median(off) if len(off) else 1.0))


def _choose(K, y, idx, folds=5, seed=0):
    """(lambda, q2) by cross-validation on the molecules `idx` alone."""
    Ks = K[np.ix_(idx, idx)]
    ys = y[idx]
    ss = float(np.sum((ys - ys.mean()) ** 2))
    best = None
    for lam in LAMS:
        p = cv_predict(Ks, ys, "krr", {"lam": float(lam)}, folds=min(folds, len(idx)), seed=seed)
        q2 = 1.0 - float(np.sum((ys - p) ** 2)) / max(ss, 1e-12)
        if best is None or q2 > best[1]:
            best = (float(lam), q2)
    return best


def _load(name):
    from run_conformal import kernels
    ds = load_dataset(os.path.join(DATA, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    return mols, names, y[keep], dict(kernels(mols, names, ALPHAS))


def _select(rule, K_lin, K_gauss, k, seed=0):
    if rule == "herding-gauss":
        return pick(K_gauss, k, method="herding")
    return pick(K_lin, k, method=rule, seed=seed)


# --------------------------------------------------------------------------- stage: subsets

def _fit_and_score(Ks, y, S, te, seed):
    best = None
    for alpha, K in Ks.items():
        lam, q2 = _choose(K, y, S, seed=seed)
        if best is None or q2 > best[2]:
            best = (alpha, lam, q2)
    alpha, lam, _ = best
    K = Ks[alpha]
    a, b = krr_fit(K[np.ix_(S, S)], y[S], lam, True)
    yp = K[np.ix_(te, S)] @ a + b
    res = float(np.sum((y[te] - yp) ** 2))
    return {"rmse": float(np.sqrt(res / len(te))),
            "r2pred": 1.0 - res / max(float(np.sum((y[te] - y[S].mean()) ** 2)), 1e-12),
            "alpha": alpha, "lam": lam}


def _mmd_between(K, A, B):
    return float(K[np.ix_(A, A)].mean() - 2 * K[np.ix_(A, B)].mean() + K[np.ix_(B, B)].mean())


def subsets_set(name, repeats=REPEATS):
    mols, names, y, Ks = _load(name)
    n = len(y)
    K_sel = Ks[SELECT_ALPHA]
    rows = []
    for rep in range(repeats):
        rng = np.random.RandomState(100 + rep)
        order = rng.permutation(n)
        n_te = int(round(0.3 * n))
        te, pool = np.sort(order[:n_te]), np.sort(order[n_te:])
        K_pool = K_sel[np.ix_(pool, pool)]
        K_pool_g = _gauss_layer(K_pool)
        full = _fit_and_score(Ks, y, pool, te, rep)
        rows.append({"repeat": rep, "m": len(pool), "rule": "all", **full})
        for m in SIZES:
            if m > len(pool) - 2:
                continue
            for rule in RULES:
                draws = RANDOM_DRAWS if rule == "random" else 1
                for d in range(draws):
                    local = _select(rule, K_pool, K_pool_g, m, seed=1000 * rep + d)
                    S = pool[np.sort(local)]
                    got = _fit_and_score(Ks, y, S, te, rep)
                    rows.append({"repeat": rep, "m": m, "rule": rule, "draw": d,
                                 "mmd2_pool": mmd2_subset(K_pool, local),
                                 "mmd2_test": _mmd_between(K_sel, S, te), **got})
        print(f"  {name} split {rep}: all-pool RMSE {full['rmse']:.3f}", flush=True)
    return {"set": name, "n": int(n), "activity_sd": float(np.std(y)), "rows": rows}


# --------------------------------------------------------------------------- stage: landmarks

def _cv_q2(K, y, seed=0):
    ss = float(np.sum((y - y.mean()) ** 2))
    out = []
    for lam in LAMS:
        p = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=10, seed=seed)
        out.append(1.0 - float(np.sum((y - p) ** 2)) / ss)
    return max(out)


def landmarks_set(name, draws=10):
    mols, names, y, Ks = _load(name)
    best = max(((a, _cv_q2(K, y)) for a, K in Ks.items()), key=lambda t: t[1])
    alpha, q2_exact = best
    K = Ks[alpha]
    # landmarks are chosen on the kernel they approximate, as a user of the low-rank path would
    K_sel = K
    K_g = _gauss_layer(K)
    norm = float(np.linalg.norm(K))
    rows = []
    for m in LANDMARKS:
        if m >= len(y):
            continue
        for rule in RULES:
            for d in range(draws if rule == "random" else 1):
                S = np.sort(_select(rule, K_sel, K_g, m, seed=d))
                Kt = nystrom_kernel(K[:, S], K[np.ix_(S, S)])
                rows.append({"m": m, "rule": rule, "draw": d,
                             "error": float(np.linalg.norm(K - Kt)) / norm,
                             "q2": _cv_q2(Kt, y)})
        print(f"  {name}: {m} landmarks done", flush=True)
    return {"set": name, "n": len(y), "alpha": alpha, "q2_exact": q2_exact, "rows": rows}


# --------------------------------------------------------------------------- stage: members

def members_stage():
    import run_ensembles as RE
    import run_shells as RS
    from cmfqsar.ensembles import ensemble_molecule, members_of, stored_weights
    from cmfqsar.fields import AtomTable, FieldSet, build_fields
    from cmfqsar.invariant import AutocorrelationSpec, block_sizes, feature_matrix
    from cmfqsar.model import CMFModel

    ds = load_dataset(RE.FILE, activity_prop="ea_298", ensemble_by=RE.GROUP,
                      member_weight=RE.WEIGHT)
    members = [members_of(m) for m in ds.mols]
    weights = [stored_weights(m) for m in ds.mols]
    # the field kernel of every shell, framed, on one scale for all of them
    fs1 = FieldSet.create(physchem=list(RS.FIELDS), alpha=0.5, charge_model="gasteiger")
    flat = [mb for group in members for mb in group]
    mf = [build_fields(mb, fs1, f"s{i}") for i, mb in enumerate(flat)]
    probe = CMFModel(fieldset=fs1, method="krr", normalize=False)
    K_all = probe.joint_kernel(probe.prepare_training_kernels(AtomTable.from_molecules(mf)))
    start = np.cumsum([0] + [len(g) for g in members])
    blocks = [K_all[start[i]:start[i + 1], start[i]:start[i + 1]] for i in range(len(members))]

    # how close each choice comes to the mean field of all shells, relative to its size
    def rel(G, idx, v, p):
        return float(np.sqrt(mmd2_subset(G, idx, v, p) / max(p @ G @ p, 1e-300)))

    distance = {}
    for k in (1, 2, 4, 8):
        vals = {"heaviest": [], "herding, equal weights": [], "herding, fitted weights": [],
                "greedy, equal weights": [], "greedy, fitted weights": [], "random": []}
        for G, w in zip(blocks, weights):
            p = w / w.sum()
            heavy = np.argsort(-w, kind="stable")[:k]
            vals["heaviest"].append(rel(G, heavy, p[heavy] / p[heavy].sum(), p))
            h = pick(G, k, method="herding", weights=p)
            vals["herding, equal weights"].append(rel(G, h, None, p))
            vals["herding, fitted weights"].append(rel(G, h, quadrature_weights(G, h, p), p))
            g = pick(G, k, method="greedy", weights=p)
            vals["greedy, equal weights"].append(rel(G, g, None, p))
            vals["greedy, fitted weights"].append(rel(G, g, quadrature_weights(G, g, p), p))
            rng = np.random.RandomState(k)
            rr = []
            for _ in range(50):
                r = rng.choice(len(G), k, replace=False)
                rr.append(rel(G, r, p[r] / p[r].sum(), p))
            vals["random"].append(float(np.mean(rr)))
        distance[k] = {key: float(np.mean(v)) for key, v in vals.items()}
        print(f"  {k} shells: " + ", ".join(f"{a} {b:.3f}" for a, b in distance[k].items()),
              flush=True)

    def reduced(choice):
        mols = []
        for i, (group, mol) in enumerate(zip(members, ds.mols)):
            idx, v = choice(i)
            m = ensemble_molecule([group[j] for j in idx], weights=v,
                                  name=mol.GetProp("_Name") if mol.HasProp("_Name") else None)
            mols.append(m)
        return mols

    def rows_for(mols):
        spec = AutocorrelationSpec()
        entry = {}
        for alpha in RS.ALPHAS:
            fs = FieldSet.create(physchem=list(RS.FIELDS), alpha=alpha, charge_model="gasteiger",
                                 conformers="ensemble", conformer_weighting="given")
            mfs = [build_fields(m, fs, n) for m, n in zip(mols, ds.names)]
            entry[alpha] = (feature_matrix(mfs, spec, fieldset=fs), spec,
                            block_sizes(spec, fs.n_fields))
        return entry

    def herded(k, method="herding"):
        def choice(i):
            p = weights[i] / weights[i].sum()
            h = pick(blocks[i], k, method=method, weights=p)
            return h, quadrature_weights(blocks[i], h, p)
        return choice

    def drawn(k, seed):
        def choice(i):
            rng = np.random.RandomState(1000 * seed + i)
            r = rng.choice(len(members[i]), k, replace=False)
            return r, weights[i][r]
        return choice

    def heaviest(k):
        def choice(i):
            h = np.argsort(-weights[i], kind="stable")[:k]
            return h, weights[i][h]
        return choice

    variants = {"herding-1": herded(1), "herding-2": herded(2), "herding-4": herded(4),
                "greedy-1": herded(1, "greedy"), "greedy-2": herded(2, "greedy"),
                "greedy-4": herded(4, "greedy"), "greedy-8": herded(8, "greedy"),
                "random-4-a": drawn(4, 1), "random-4-b": drawn(4, 2), "random-4-c": drawn(4, 3),
                "heaviest-4 (control)": heaviest(4)}
    rows = RS.load_series()
    labels = [str(r["composition"]) for r in rows]
    splits = {"random": (RS.random_folds(len(rows)), None),
              "by-composition": (RS.grouped_folds(labels), labels)}
    with open(os.path.join(HERE, "results", "ensembles", "members.json"), encoding="utf-8") as fh:
        stored = json.load(fh)
    out = {"distance": {str(k): v for k, v in distance.items()}, "models": {}, "seconds": {}}
    entries = {}
    for name, choice in variants.items():
        t0 = time.time()
        entries[name] = rows_for(reduced(choice))
        out["seconds"][name] = round(time.time() - t0, 1)
        print(f"  {name}: rows in {out['seconds'][name]:.0f} s", flush=True)
    for target, (tlabel, unit) in RS.TARGETS.items():
        y = np.array([float(r[target]) for r in rows], float)
        for split, (design, group_labels) in splits.items():
            key = f"{target}|{split}"
            recs = {}
            for name, entry in entries.items():
                recs[name] = RS.nested(entry, RS.descriptors(rows, "shells-free"), y, design,
                                       group_labels, (RS.MIX,), label=name, unit=unit)
            for k in (1, 2, 4, 8, 16):
                recs[f"heaviest-{k}"] = stored["models"][key][f"given-{k}"]
            out["models"][key] = recs
    out["control_largest_difference"] = max(
        float(np.max(np.abs(np.asarray(out["models"][key]["heaviest-4 (control)"]["abs_errors"])
                            - np.asarray(out["models"][key]["heaviest-4"]["abs_errors"]))))
        for key in out["models"])
    print(f"  control: the 4 heaviest against the stored records, largest difference "
          f"{out['control_largest_difference']:.3g}")
    return out


# --------------------------------------------------------------------------- the report

def _mean(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.mean(xs)) if xs else float("nan")


def _boot(diffs, reps=10000, seed=0):
    d = np.asarray(diffs, float)
    rng = np.random.RandomState(seed)
    m = d[rng.randint(0, len(d), (reps, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _per_set(data, m, rule, key):
    """The mean of `key` over the splits of every set (random: over its draws too)."""
    out = []
    for d in data:
        v = [r[key] for r in d["rows"] if r["m"] == m and r["rule"] == rule]
        out.append(_mean(v) if v else None)
    return out


def report(results=RESULTS):
    L = ["# Kernel herding against the usual ways of choosing a subset", ""]
    path = os.path.join(results, "subsets.json")
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))["sets"]
        L += ["## Which molecules to measure first", "",
              f"{len(data)} Sutherland sets, {REPEATS} random splits into a pool (70 %) and a test "
              "set (30 %). From the pool, m molecules are chosen without looking at the activity; "
              "a model fitted on them alone predicts the test set. `random` is the mean of "
              f"{RANDOM_DRAWS} draws per split. The kernel the subsets are chosen on is the joint "
              f"kernel at alpha = {SELECT_ALPHA}; `herding-gauss` uses a Gaussian of the field "
              "distance instead. MMD^2 is between the chosen molecules and the pool, in the "
              "joint kernel.", "",
              "| m | rule | R2pred on the test set | RMSE on the test set | MMD^2 to the pool | "
              "MMD^2 to the test set | RMSE minus random, mean over sets [95 % interval] | "
              "sets where better than random |",
              "|---:|---|---:|---:|---:|---:|---:|---:|"]
        for m in SIZES:
            base = _per_set(data, m, "random", "rmse")
            for rule in RULES:
                r2 = _per_set(data, m, rule, "r2pred")
                if all(v is None for v in r2):
                    continue
                rm = _per_set(data, m, rule, "rmse")
                mp = _per_set(data, m, rule, "mmd2_pool")
                mt = _per_set(data, m, rule, "mmd2_test")
                pairs = [(a, b) for a, b in zip(rm, base) if a is not None and b is not None]
                if rule == "random" or not pairs:
                    cmp_cells = "| |"
                else:
                    diff = [a - b for a, b in pairs]
                    lo, hi = _boot(diff)
                    cmp_cells = (f"{np.mean(diff):+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                                 f"{sum(x < 0 for x in diff)} of {len(diff)} |")
                L.append(f"| {m} | `{rule}` | {_mean(r2):+.3f} | {_mean(rm):.3f} | "
                         f"{_mean(mp):.4f} | {_mean(mt):.4f} | " + cmp_cells)
        full = [_mean([r["rmse"] for r in d["rows"] if r["rule"] == "all"]) for d in data]
        full_r2 = [_mean([r["r2pred"] for r in d["rows"] if r["rule"] == "all"]) for d in data]
        L += ["", f"The model on the whole pool: R2pred {np.mean(full_r2):+.3f}, RMSE "
              f"{np.mean(full):.3f}. The activity's standard deviation averages "
              f"{np.mean([d['activity_sd'] for d in data]):.2f}.", ""]
        L += ["### R2pred by set, m = 20", "",
              "| set | n | " + " | ".join(f"`{r}`" for r in RULES) + " |",
              "|---|---:|" + "---:|" * len(RULES)]
        for d in data:
            cells = []
            for rule in RULES:
                v = [r["r2pred"] for r in d["rows"] if r["m"] == 20 and r["rule"] == rule]
                cells.append(f"{np.mean(v):+.2f}" if v else "--")
            L.append(f"| {d['set']} | {d['n']} | " + " | ".join(cells) + " |")
        L.append("")

    path = os.path.join(results, "landmarks.json")
    if os.path.exists(path):
        data = json.load(open(path, encoding="utf-8"))["sets"]
        L += ["## Which molecules to use as Nystroem landmarks", "",
              "The kernel of each whole set, at the attenuation exact cross-validation chooses, is "
              "replaced by its Nystroem approximation on m landmark molecules, chosen on that "
              "kernel. Error is "
              "||K - K~|| / ||K|| (Frobenius); q2 is 10-fold kernel ridge regression on K~, with "
              "the best ridge of the grid, as for the exact kernel. `random` is the mean of ten "
              f"draws. Exact q2 averages {np.mean([d['q2_exact'] for d in data]):+.3f}.", "",
              "| landmarks | rule | relative error of the kernel | q2 | q2 minus exact |",
              "|---:|---|---:|---:|---:|"]
        qx = {d["set"]: d["q2_exact"] for d in data}
        for m in LANDMARKS:
            for rule in RULES:
                e = [_mean([r["error"] for r in d["rows"] if r["m"] == m and r["rule"] == rule])
                     for d in data]
                q = [_mean([r["q2"] for r in d["rows"] if r["m"] == m and r["rule"] == rule])
                     for d in data]
                if not any(np.isfinite(e)):
                    continue
                gap = _mean([a - qx[d["set"]] for a, d in zip(q, data)])
                L.append(f"| {m} | `{rule}` | {_mean(e):.3f} | {_mean(q):+.3f} | {gap:+.3f} |")
        L.append("")

    path = os.path.join(results, "members.json")
    if os.path.exists(path):
        import run_shells as RS
        d = json.load(open(path, encoding="utf-8"))
        L += ["## Which solvation shells to keep", "",
              "The 62 electrolytes of item BT5, each the weighted mean of 15 or 16 shells. "
              "Distance is ||mean field of the kept shells - mean field of all shells|| / "
              "||mean field of all shells||, in the framed field kernel at alpha = 0.5, averaged "
              "over the electrolytes (random: 50 draws each).", "",
              "| shells kept | the heaviest | herding, equal weights | herding, fitted weights | "
              "greedy, equal weights | greedy, fitted weights | random |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
        for k, v in d["distance"].items():
            L.append(f"| {k} | {v['heaviest']:.3f} | {v['herding, equal weights']:.3f} | "
                     f"{v['herding, fitted weights']:.3f} | {v['greedy, equal weights']:.3f} | "
                     f"{v['greedy, fitted weights']:.3f} | {v['random']:.3f} |")
        L += ["", "The alignment-free model of BT5 on the kept shells, nested cross-validation "
              "as in `benchmarks/run_ensembles.py`. Mean absolute error; the interval is the "
              "paired interval over the 62 electrolytes of the gain over the heaviest shells of "
              "the same number, positive when the variant has the smaller error. The heaviest "
              "shells are the stored records of item MS5.", ""]
        order = ["heaviest-1", "herding-1", "greedy-1", "heaviest-2", "herding-2", "greedy-2",
                 "heaviest-4", "herding-4", "greedy-4", "random-4-a", "random-4-b", "random-4-c",
                 "heaviest-8", "greedy-8", "heaviest-16"]
        for key, recs in d["models"].items():
            target, split = key.split("|")
            tlabel, unit = RS.TARGETS[target]
            L += [f"### {tlabel}, {split} folds", "",
                  "| shells | mean absolute error | gain over the same number of heaviest | "
                  "gain over all 16 |", "|---|---:|---:|---:|"]
            for name in order:
                if name not in recs:
                    continue
                r = recs[name]
                k = name.split("-")[1]
                ref = recs.get(f"heaviest-{k}")
                g1 = ("" if name.startswith("heaviest") or ref is None else
                      _interval(RS, ref, r))
                g16 = "" if name == "heaviest-16" else _interval(RS, recs["heaviest-16"], r)
                L.append(f"| {name} | {r['mae_mean']:.4f} | {g1} | {g16} |")
            L.append("")
        L += [f"Control: the 4 heaviest shells run again against the stored records, largest "
              f"difference {d['control_largest_difference']:.3g}.", ""]
    text = "\n".join(L) + "\n"
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def _interval(RS, ref, rec):
    pi = RS.paired_interval(ref["abs_errors"], rec["abs_errors"])
    return f"{pi['mean']:+.4f} [{pi['low']:+.4f}, {pi['high']:+.4f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["subsets", "landmarks", "members", "all"], default="all")
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args()
    if a.report:
        print(report(a.results))
        return
    os.makedirs(a.results, exist_ok=True)

    def save(name, payload):
        with open(os.path.join(a.results, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    if a.stage in ("subsets", "all"):
        t0 = time.time()
        save("subsets", {"sets": [subsets_set(s) for s in a.sets],
                         "seconds": round(time.time() - t0, 1)})
    if a.stage in ("landmarks", "all"):
        t0 = time.time()
        save("landmarks", {"sets": [landmarks_set(s) for s in a.sets],
                           "seconds": round(time.time() - t0, 1)})
    if a.stage in ("members", "all"):
        t0 = time.time()
        out = members_stage()
        out["total_seconds"] = round(time.time() - t0, 1)
        save("members", out)
    print(report(a.results))


if __name__ == "__main__":
    main()
