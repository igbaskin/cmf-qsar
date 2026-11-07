#!/usr/bin/env python
"""
Which columns count: a weighted distance over the channels of the invariant kernel (plan item M15,
second half).

The kernel of this program measures one Euclidean distance over every channel at once. On the
published hop barriers of item BT7 a forest beat that kernel by +0.171 in q2 **on the kernel's own
rows**, and a Gaussian kernel on those rows recovered only about 30 per cent of the difference
(+0.052 [+0.023, +0.080]), so what the forest has is not a curved response to the features. The
reading this script tests is the other obvious one: one distance over 2,601 columns, which is
what the hop environments give, dilutes the few that carry the target among the many that do not,
while a tree reads columns one at a time.

**The arms.** Every arm is the same cosine-normalised linear kernel on the same scaled rows, and
differs only in the weight each column is given. A weight enters the features, so the kernel sees
its square. Every weight is computed **inside the training part** of each outer fold and the test
fold is touched once, with the choice already made.

    equal weights        the kernel as the program builds it: the reference
    block align          one weight per block, that is per pair of field types, by centred kernel
                         alignment (`cmfqsar.invariant.align_block_weights`, mode "align")
    block alignf         the same blocks, with the weights that maximise the alignment of the
                         combination (mode "alignf")
    column align         one weight per column, by the same alignment rule
    best columns         the k columns of the largest alignment, the rest set to zero, with k
                         chosen inside the training part
    forest importances   one weight per column, the square root of a random forest's importance,
                         the forest fitted on the training part alone

The last arm is the sharpest form of the question: if the forest's advantage is *which* columns it
attends to, then handing the kernel the forest's own ranking should recover it.

**A forest on the same rows** is reported beside them for context, so the gap the arms have to
close is measured here and not quoted.

**What is already known, so that it is not measured twice.** The pooled field weights pilot of
1 November 2026 learned weights over 38 molecular sets and did not improve prediction over equal
weights. This is a different case in three ways and the table says so: thousands of columns rather
than a handful of fields, a solid-state target, and a measured gap to another learner on the
identical rows. The first half of item M15 removed one other explanation: the row length, which this kernel
divides out, is worth +0.144 in q2 on these same hop environments, and it is kept out of the way
here by giving every arm the same cosine normalisation.

Usage
-----
    python benchmarks/run_colweights.py --hops        # the hop environments, about twenty minutes
    python benchmarks/run_colweights.py --molecules   # the beta-cyclodextrin guests
    python benchmarks/run_colweights.py --sac         # the single-atom catalysts, metal-out
    python benchmarks/run_colweights.py --report      # the table from the stored results
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

RESULTS = os.path.join(HERE, "results", "colweights")

#: the reference: one distance over every column, each scaled only by its block's root mean square
ARM_EQUAL = "equal weights"
ARM_BLOCK = "block align"
ARM_BLOCKF = "block alignf"
ARM_COLUMN = "column align"
ARM_BEST = "best columns"
ARM_FOREST = "forest importances"
ARMS = (ARM_EQUAL, ARM_BLOCK, ARM_BLOCKF, ARM_COLUMN, ARM_BEST, ARM_FOREST)
#: how many of the best columns the `best columns` arm chooses between, as a share of all columns
BEST_SHARES = (0.02, 0.1, 0.3, 0.6)
#: trees in the forest, both for the weights and for the forest arm
TREES = 300
LAMS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0)
SEED = 0
BOOTSTRAP = 1000
INNER = 5


def log(msg: str) -> None:
    print(msg, flush=True)


def result(name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    return os.path.join(RESULTS, name)


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def jdump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_default)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------- the weights


def alignment_scores(Z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The centred alignment of every single column with the target, column by column.

    For one column this is what :func:`cmfqsar.invariant.align_block_weights` computes for a block
    of one: the squared projection of the centred column on the centred target, divided by the
    column's own squared length. It is the squared correlation times a constant, so it ranks the
    columns the way a correlation does and needs one pass over the matrix.
    """
    Zc = Z - Z.mean(axis=0)
    yc = np.asarray(y, float)
    yc = yc - yc.mean()
    proj = Zc.T @ yc
    nrm = np.sum(Zc * Zc, axis=0)
    out = np.zeros(Z.shape[1])
    live = nrm > max(1e-300, float(nrm.max()) * 1e-12) if Z.shape[1] else nrm > 0
    out[live] = (proj[live] ** 2) / nrm[live]
    return out


def weights_for(arm: str, Z: np.ndarray, y: np.ndarray, blocks: Sequence[int],
                share: float = 0.0, seed: int = SEED) -> np.ndarray:
    """One weight per column, from the training rows alone.

    The weights multiply the features, so the kernel sees them squared; every rule below therefore
    returns the square root of the quantity it wants the kernel to be proportional to.
    """
    from cmfqsar.invariant import align_block_weights, expand_block_weights
    p = Z.shape[1]
    if arm == ARM_EQUAL:
        return np.ones(p)
    if arm in (ARM_BLOCK, ARM_BLOCKF):
        mode = "align" if arm == ARM_BLOCK else "alignf"
        w = align_block_weights(Z, y, blocks, mode=mode)
        return expand_block_weights(w, blocks)
    if arm == ARM_COLUMN:
        h = alignment_scores(Z, y)
        return np.sqrt(h / max(float(h.mean()), 1e-300))
    if arm == ARM_BEST:
        h = alignment_scores(Z, y)
        k = max(1, int(round(share * p)))
        keep = np.argsort(-h)[:k]
        w = np.zeros(p)
        w[keep] = 1.0
        return w
    if arm == ARM_FOREST:
        from sklearn.ensemble import RandomForestRegressor
        f = RandomForestRegressor(n_estimators=TREES, random_state=seed, n_jobs=-1)
        f.fit(Z, y)
        imp = np.asarray(f.feature_importances_, float)
        return np.sqrt(imp / max(float(imp.mean()), 1e-300))
    raise ValueError(f"unknown arm {arm!r}")


def candidates(arm: str) -> List[float]:
    """The settings of an arm that the inner split chooses between."""
    return list(BEST_SHARES) if arm == ARM_BEST else [0.0]


# ----------------------------------------------------------------------------- the fitting


def cosine_blocks(A: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """The cosine-normalised linear kernel between two sets of rows of `A`."""
    d = np.sqrt(np.maximum(np.sum(A * A, axis=1), 1e-300))
    K = A[rows] @ A[cols].T
    return K / np.outer(d[rows], d[cols])


def _ridge_scores(K_tr: np.ndarray, y_tr: np.ndarray, inner: Sequence[Tuple[np.ndarray,
                                                                           np.ndarray]],
                  lams: Sequence[float]) -> np.ndarray:
    """q2 of every ridge on the inner splits, with the splits given as positions inside `K_tr`."""
    from cmfqsar.model import krr_fit, r2_score
    pred = np.full((len(lams), len(y_tr)), np.nan)
    for tr2, te2 in inner:
        Ka = K_tr[np.ix_(tr2, tr2)]
        for j, lam in enumerate(lams):
            a, b = krr_fit(Ka, y_tr[tr2], float(lam))
            pred[j, te2] = K_tr[np.ix_(te2, tr2)] @ a + b
    out = np.full(len(lams), -np.inf)
    for j in range(len(lams)):
        ok = np.isfinite(pred[j])
        if ok.sum() > 2 and np.std(y_tr[ok]) > 0:
            out[j] = r2_score(y_tr[ok], pred[j][ok])
    return out


def nested_arm(arm: str, Z: np.ndarray, y: np.ndarray, blocks: Sequence[int],
               splits: Sequence[Tuple[np.ndarray, np.ndarray]],
               inner_fn: Callable[[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]],
               lams: Sequence[float] = LAMS, seed: int = SEED, log=log) -> dict:
    """Out-of-fold predictions of one arm, with its weights and its ridge chosen on the training
    part of every outer fold."""
    from cmfqsar.model import krr_fit, r2_score, rmse
    y = np.asarray(y, float)
    pred = np.full(len(y), np.nan)
    chosen = []
    t0 = time.time()
    for k, (tr, te) in enumerate(splits):
        inner = inner_fn(tr)                        # positions inside tr
        best = None
        for share in candidates(arm):
            w = weights_for(arm, Z[tr], y[tr], blocks, share=share, seed=seed)
            A = Z * w
            K_tr = cosine_blocks(A, tr, tr)
            scores = _ridge_scores(K_tr, y[tr], inner, lams)
            j = int(np.argmax(scores))
            if best is None or scores[j] > best[0]:
                best = (float(scores[j]), share, float(lams[j]), w)
        _, share, lam, w = best
        A = Z * w
        K_tr = cosine_blocks(A, tr, tr)
        a, b = krr_fit(K_tr, y[tr], lam)
        pred[te] = cosine_blocks(A, te, tr) @ a + b
        chosen.append({"share": share, "lam": lam,
                       "columns_used": int(np.sum(w > 0)),
                       "inner_q2": best[0]})
        log(f"    fold {k}: share {share:g}, ridge {lam:g}, {int(np.sum(w > 0))} columns, "
            f"inner q2 {best[0]:+.3f}")
    ok = np.isfinite(pred)
    out = {"arm": arm, "predicted": pred, "q2": float(r2_score(y[ok], pred[ok])),
           "rmse": float(rmse(y[ok], pred[ok])),
           "mae": float(np.mean(np.abs(y[ok] - pred[ok]))), "n": int(ok.sum()),
           "chosen": chosen, "seconds": time.time() - t0}
    log(f"  {arm}: q2 {out['q2']:+.3f}, error {out['rmse']:.3f}, mean absolute error "
        f"{out['mae']:.3f} ({out['seconds'] / 60:.1f} min)")
    return out


FOREST_SCALED = "a forest on the scaled rows (the length kept)"
FOREST_COSINE = "a forest on the kernel's own rows"


def cosine_rows(Z: np.ndarray) -> np.ndarray:
    """The rows the cosine kernel is really built from: each divided by its own length.

    A forest given `Z` sees how long a row is, and the kernel arms do not, so a forest on `Z`
    would be ahead by whatever the row length is worth -- which is the first half of item M15 and
    not this question. A forest on these rows is given exactly what the kernel is given.
    """
    d = np.sqrt(np.maximum(np.sum(Z * Z, axis=1), 1e-300))
    return Z / d[:, None]


def forest_arm(Z: np.ndarray, y: np.ndarray, splits, seed: int = SEED, log=log,
               name: str = FOREST_SCALED) -> dict:
    """A forest on the given rows, on the same folds: the gap the arms have to close."""
    from sklearn.ensemble import RandomForestRegressor

    from cmfqsar.model import r2_score, rmse
    y = np.asarray(y, float)
    pred = np.full(len(y), np.nan)
    t0 = time.time()
    for tr, te in splits:
        f = RandomForestRegressor(n_estimators=TREES, random_state=seed, n_jobs=-1)
        f.fit(Z[tr], y[tr])
        pred[te] = f.predict(Z[te])
    ok = np.isfinite(pred)
    out = {"arm": name, "predicted": pred,
           "q2": float(r2_score(y[ok], pred[ok])), "rmse": float(rmse(y[ok], pred[ok])),
           "mae": float(np.mean(np.abs(y[ok] - pred[ok]))), "n": int(ok.sum()),
           "chosen": [], "seconds": time.time() - t0}
    log(f"  {name}: q2 {out['q2']:+.3f}, error {out['rmse']:.3f}, "
        f"mean absolute error {out['mae']:.3f} ({out['seconds'] / 60:.1f} min)")
    return out


def paired_interval(y: np.ndarray, a: np.ndarray, b: np.ndarray, groups: Optional[np.ndarray],
                    runs: int = BOOTSTRAP, seed: int = SEED):
    """The q2 difference of two sets of predictions with its interval, drawing whole groups."""
    from cmfqsar.model import r2_score
    y = np.asarray(y, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = np.where(np.isfinite(y) & np.isfinite(a) & np.isfinite(b))[0]
    rng = np.random.RandomState(seed)
    diffs = []
    if groups is None:
        for _ in range(runs):
            i = idx[rng.randint(0, len(idx), len(idx))]
            if np.var(y[i]) > 0:
                diffs.append(r2_score(y[i], a[i]) - r2_score(y[i], b[i]))
    else:
        groups = np.asarray(groups)
        keys = np.unique(groups[idx])
        by = {k: idx[groups[idx] == k] for k in keys}
        for _ in range(runs):
            draw = keys[rng.randint(0, len(keys), len(keys))]
            i = np.concatenate([by[k] for k in draw])
            if np.var(y[i]) > 0:
                diffs.append(r2_score(y[i], a[i]) - r2_score(y[i], b[i]))
    d = float(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    if not diffs:
        return d, float("nan"), float("nan")
    return d, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


def measure(name: str, Z: np.ndarray, y: np.ndarray, blocks: Sequence[int], splits, inner_fn,
            groups: Optional[np.ndarray], args, extra: Optional[dict] = None) -> dict:
    """Every arm on one series, with the intervals against the reference and the forest."""
    y = np.asarray(y, float)
    out = {"series": name, "n": int(len(y)), "columns": int(Z.shape[1]),
           "blocks": int(len(blocks)), "folds": len(splits), "trees": TREES,
           "shares": list(BEST_SHARES), "lams": list(LAMS), "inner": INNER,
           "rows": [], "pairs": []}
    if extra:
        out.update(extra)
    log(f"  {name}: {len(y)} samples, {Z.shape[1]} columns in {len(blocks)} blocks, "
        f"{len(splits)} folds")
    results = {}
    for arm in ARMS:
        results[arm] = nested_arm(arm, Z, y, blocks, splits, inner_fn, seed=args.seed)
    results[FOREST_SCALED] = forest_arm(Z, y, splits, seed=args.seed, name=FOREST_SCALED)
    results[FOREST_COSINE] = forest_arm(cosine_rows(Z), y, splits, seed=args.seed,
                                        name=FOREST_COSINE)
    for key, r in results.items():
        out["rows"].append({k: v for k, v in r.items() if k != "predicted"})
    for key, r in results.items():
        if key == ARM_EQUAL:
            continue
        d, lo, hi = paired_interval(y, r["predicted"], results[ARM_EQUAL]["predicted"], groups,
                                    runs=args.bootstrap, seed=args.seed)
        out["pairs"].append({"a": key, "b": ARM_EQUAL, "diff": d, "low": lo, "high": hi})
        log(f"  {key} against the reference: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    # and how much of the forest's lead each arm closes
    gap = results[FOREST_COSINE]["q2"] - results[ARM_EQUAL]["q2"]
    out["forest_gap"] = float(gap)
    out["forest_gap_with_length"] = float(results[FOREST_SCALED]["q2"]
                                          - results[ARM_EQUAL]["q2"])
    for key, r in results.items():
        if key in (ARM_EQUAL, FOREST_SCALED, FOREST_COSINE):
            continue
        share = (r["q2"] - results[ARM_EQUAL]["q2"]) / gap if abs(gap) > 1e-12 else float("nan")
        out["pairs"].append({"a": key, "b": "share of the forest's lead", "diff": float(share),
                             "low": float("nan"), "high": float("nan")})
    log(f"  the forest's lead over the reference, on the kernel's own rows: {gap:+.3f} "
        f"(with the row length kept: {out['forest_gap_with_length']:+.3f})")
    return out


# ----------------------------------------------------------------------------- the series


def _inner_from_targets(targets, folds: int, seed: int):
    """Inner splits that hold out whole molecules, as positions inside the training part."""
    def inner_fn(tr: np.ndarray):
        inside = targets.subset(tr)
        out = []
        for te2 in inside.folds(folds, seed):
            te2 = np.asarray(te2, int)
            tr2 = np.setdiff1d(np.arange(len(tr)), te2)
            out.append((tr2, te2))
        return out
    return inner_fn


def hops_run(args) -> dict:
    """The hop environments of item BT7: the series the question came from."""
    from cmfqsar.invariant import channel_scale
    from cmfqsar.localenv import LOCAL_SPEC, local_features
    from run_hops import fieldset_for
    from run_litraj import PREPARED, load_prepared

    hs, doc = load_prepared(args.prepared or PREPARED, args.max_structures)
    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    t = hs.targets()
    y = np.asarray(t.y, float)
    t0 = time.time()
    envs = hs.environments(crystals, rows=t.atom, rho=args.rho, r_max=args.r_max)
    X = local_features(envs)
    log(f"  the hop environments: {X.shape[1]} channels, {time.time() - t0:.0f} s")
    from cmfqsar.invariant import block_sizes
    blocks = block_sizes(LOCAL_SPEC, len(fs.names))
    Z = X * channel_scale(X, LOCAL_SPEC.n_bins, blocks=blocks)
    splits = [(np.setdiff1d(np.arange(len(y)), te), np.asarray(te, int))
              for te in t.folds(args.folds, args.seed)]
    return measure("hops", Z, y, blocks, splits, _inner_from_targets(t, INNER, args.seed),
                   np.asarray(t.molecule, int), args,
                   extra={"descriptor": "the hop environment (BT7)", "target": "barrier",
                          "rho": args.rho, "r_max": args.r_max, "fields": list(fs.names),
                          "n_structures": len(hs.structures)})


def molecules_run(args) -> dict:
    """The beta-cyclodextrin guests: a molecular series with no frame."""
    from sklearn.model_selection import KFold

    from cmfqsar.dataio import load_dataset
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.invariant import (AutocorrelationSpec, block_sizes, channel_scale,
                                   feature_matrix)
    import run_cyclodextrin as C

    path = os.path.join(ROOT, "examples", "cyclodextrin", "beta_cyclodextrin.csv")
    ds = load_dataset(path)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    fs = FieldSet.create(physchem=C.FIELDS, alpha=0.5, charge_model="gasteiger",
                         param_scheme="rdkit")
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    spec = AutocorrelationSpec()
    X = feature_matrix(mf, spec)
    blocks = block_sizes(spec, len(fs.names))
    Z = X * channel_scale(X, spec.n_bins, blocks=blocks)
    splits = list(KFold(n_splits=args.folds, shuffle=True, random_state=args.seed).split(y))

    def inner_fn(tr):
        return list(KFold(n_splits=INNER, shuffle=True,
                          random_state=args.seed).split(np.arange(len(tr))))

    return measure("molecules", Z, y, blocks, splits, inner_fn, None, args,
                   extra={"descriptor": "whole-molecule autocorrelation", "target": "log K",
                          "file": "examples/cyclodextrin/beta_cyclodextrin.csv"})


def sac_run(args) -> dict:
    """The single-atom catalysts, held out by metal: the design the pilot found hardest."""
    import run_sac as R
    import run_sites as S5
    from cmfqsar import sites as S
    from cmfqsar.invariant import channel_scale

    data = R.load_set(log)
    sites = [R._metal_index(a) for a in data.structures]
    fs = R.fieldset_for("classic", data.metals)
    crystals = R.crystals_for(data, fs)
    index = S.one_site_targets(sites, np.zeros(len(data)), [c.n_atoms for c in crystals],
                               name="binding")
    X = S.site_features(crystals, index, S5.spec_for(6.0))
    blocks = [X.shape[1]]                       # the site profile is one block per field pair,
    from cmfqsar.periodic import periodic_block_sizes                        # given by the spec
    try:
        blocks = periodic_block_sizes(S5.spec_for(6.0), crystals[0].weights.shape[1])
        if int(sum(blocks)) != X.shape[1]:
            blocks = [X.shape[1]]
    except Exception:                           # a spec this helper does not describe
        blocks = [X.shape[1]]
    Z = X * channel_scale(X, S5.spec_for(6.0).n_bins, blocks=blocks)
    y = np.asarray(data.dE[args.adsorbate], float)
    have = np.where(np.isfinite(y))[0]
    splits = []
    for te_all in R.outer_folds_for(args.design, data, args.seed):
        te = np.intersect1d(te_all, have)
        tr = np.setdiff1d(have, te_all)
        if len(te) and len(tr) > 5:
            splits.append((tr, te))

    def inner_fn(tr):
        out = []
        for te2 in R.inner_folds_for(args.design, data, tr, args.seed):
            te2 = np.asarray(te2, int)
            pos = {int(v): i for i, v in enumerate(tr)}
            te_pos = np.array([pos[int(v)] for v in te2 if int(v) in pos], int)
            if len(te_pos) == 0:
                continue
            tr_pos = np.setdiff1d(np.arange(len(tr)), te_pos)
            out.append((tr_pos, te_pos))
        return out

    return measure("sac", Z, y, blocks, splits, inner_fn, None, args,
                   extra={"descriptor": "site profile (M5), 6 A", "target": f"dE_{args.adsorbate}",
                          "design": args.design, "n_catalysts": len(data)})


# ----------------------------------------------------------------------------- the table

STAGES = ("hops", "molecules", "sac")


def _fmt(x, digits=3, sign=True) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def report(out: str = "") -> str:
    lines = ["# Which columns count: a weighted distance over the channels (plan item M15)", "",
             "Every arm is the same cosine-normalised linear kernel on the same scaled rows and "
             "differs only in the weight each column is given. Every weight is computed inside "
             "the training part of each outer fold. The difference is in q2 against equal "
             "weights, which is what the program does today, with the bootstrap interval of rule "
             "4. The last two rows of each table are forests: on the rows the kernel really "
             "uses, which is the gap the arms have to close, and on the rows as they are, which "
             "also carry the row length that the first half of item M15 measured.", ""]
    found = False
    for stage in STAGES:
        path = result(f"{stage}.json")
        if not os.path.isfile(path):
            continue
        found = True
        doc = jload(path)
        pairs = {p["a"]: p for p in doc.get("pairs", []) if p["b"] == ARM_EQUAL}
        shares = {p["a"]: p["diff"] for p in doc.get("pairs", [])
                  if p["b"] == "share of the forest's lead"}
        lines.append(f"## {stage}")
        lines.append("")
        lines.append(f"{doc.get('descriptor', '')}, target {doc.get('target', '')}, "
                     f"{doc['n']} samples, {doc['columns']} columns in {doc['blocks']} blocks, "
                     f"{doc['folds']} folds.")
        lines.append("")
        lines.append("| arm | q2 | error | mean absolute error | columns used | difference "
                     "against equal weights | share of the forest's lead |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in doc.get("rows", []):
            used = sorted({c["columns_used"] for c in r.get("chosen", [])})
            cols = ("--" if not used else
                    (str(used[0]) if len(used) == 1 else f"{used[0]} to {used[-1]}"))
            if r["arm"] == ARM_EQUAL:
                diff, part = "the reference", "--"
            else:
                p = pairs.get(r["arm"])
                diff = (f"{_fmt(p['diff'])} [{_fmt(p['low'])}, {_fmt(p['high'])}]" if p
                        else "--")
                s = shares.get(r["arm"])
                # a share of the lead means nothing where the forest is behind the kernel
                part = ("--" if s is None or not np.isfinite(s)
                        or float(doc.get("forest_gap") or 0.0) <= 0
                        else f"{100 * s:.0f} per cent")
            lines.append(f"| {r['arm']} | {_fmt(r['q2'])} | {_fmt(r['rmse'], sign=False)} | "
                         f"{_fmt(r['mae'], sign=False)} | {cols} | {diff} | {part} |")
        lines.append("")
        if float(doc.get("forest_gap") or 0.0) <= 0:
            lines.append("On this series the forest is behind the kernel, so there is no lead to "
                         "share and that column is left empty.")
            lines.append("")
        lines.append(f"The forest's lead over equal weights on this series, on the kernel's "
                     f"own rows: {_fmt(doc.get('forest_gap'))}. On the rows as they are, which "
                     f"also carry the row length of the first half of item M15: "
                     f"{_fmt(doc.get('forest_gap_with_length'))}.")
        lines.append("")
    if not found:
        lines.append("Nothing is stored yet: run the stages of `benchmarks/run_colweights.py`.")
        lines.append("")
    text = "\n".join(lines) + "\n"
    path = out or result("table.md")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def write_provenance(args, stage: str, seconds: float) -> None:
    doc = {"stage": stage, "argv": sys.argv[1:], "seed": args.seed, "lams": list(LAMS),
           "shares": list(BEST_SHARES), "trees": TREES, "bootstrap": args.bootstrap,
           "folds": args.folds, "inner": INNER,
           "written": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": round(seconds / 60.0, 1)}
    jdump(doc, result(f"run_{stage}.json"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for stage in STAGES:
        p.add_argument(f"--{stage}", action="store_true", help=f"the {stage} series alone")
    p.add_argument("--report", action="store_true", help="rebuild the table from stored results")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--prepared", default="")
    p.add_argument("--max-structures", type=int, default=0)
    p.add_argument("--rho", type=float, default=0.25)
    p.add_argument("--r-max", type=float, default=6.0)
    p.add_argument("--design", default="metal-out", help="the fold design of the catalyst series")
    p.add_argument("--adsorbate", default="OH", help="which binding energy of the catalysts")
    args = p.parse_args()
    wanted = [s for s in STAGES if getattr(args, s)]
    if args.report and not wanted:
        print(report())
        return 0
    if not wanted:
        wanted = list(STAGES)
    for stage in wanted:
        log(f"=== {stage} ===")
        t0 = time.time()
        doc = globals()[f"{stage}_run"](args)
        jdump(doc, result(f"{stage}.json"))
        write_provenance(args, stage, time.time() - t0)
        log(f"  {stage}: {(time.time() - t0) / 60:.1f} minutes")
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
