#!/usr/bin/env python
"""
The hop models of plan item BT7 on barriers from density functional theory (nebDFT2k).

``benchmarks/prepare_nebdft.py`` prepares the nebDFT2k set of Dembitskiy et al., npj
Computational Materials 11, 131 (2025): 1,681 hops of a lithium ion into a vacancy, each with
the barrier of a path that was relaxed by the nudged elastic band method with density
functional theory (DFT).  Until now the hop models of the program were fitted on barriers
from the bond-valence site-energy method, an empirical energy landscape.  This is the first
fit on barriers from a quantum calculation, and the paper gives numbers to compare with.

The descriptions of a hop
-------------------------
They are those of ``run_hops.py`` and ``run_litraj.py``, built from the rigid structure of the
Materials Project, without a vacancy and without any relaxation:

- the hop length and the bottleneck (the narrowest place of the straight path);
- counts of atoms by type in shells about the midpoint of the hop;
- the two end sites, each with its profile of neighbours (plan item M5);
- the invariant description of the field about the midpoint;
- the fields in the frame of the hop, compared by the rotation-averaged overlap kernel.  This
  is the hop model of item BT7.

The models
----------
Kernel ridge regression on each description, with the linear kernel or a Gaussian kernel on
top of it, and the ridge, all chosen by cross-validation inside the training part on folds
that hold out whole structures.  A random forest on the rows of the descriptions that have
rows, which is the kind of model that the paper reports.

The bond-valence barrier as one more input
------------------------------------------
The index of the set gives the bond-valence barrier of every hop.  The best models of the
paper use it as one more feature ("transfer learning" in its table).  Here it is used in the
same way: alone, as a second kernel added to the kernel of a description, and as one more
column of a forest.

Two protocols
-------------
The published split: the model is fitted on the training and the validation part and scored
on the test part, which is what the table of the paper reports.  And ten folds that hold out
whole structures, over all hops, for the intervals: a difference between two models is given
with a 95 % interval from a bootstrap that draws whole structures.

Usage
-----
    python benchmarks/prepare_nebdft.py --fetch    # once: downloads 69 MB
    python benchmarks/run_nebdft.py                # the benchmark, then the table
    python benchmarks/run_nebdft.py --report
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

RESULTS = os.path.join(HERE, "results", "nebdft")
PREPARED = os.path.join(RESULTS, "prepared.json")
CACHE = os.path.join(HERE, "cache", "nebdft")
RHO = 0.25
R_MAX = 6.0
SITE_R = 6.0
FOLDS = 10
INNER = 4
SEED = 0
BOOTSTRAP = 2000
RIDGES = tuple(10.0 ** e for e in np.arange(-7.0, 1.01, 1.0))
GAMMAS = (0.25, 1.0, 4.0, 16.0)            # times one over the median squared distance
BVSE_WEIGHTS = (0.5, 2.0)                  # the weight of the kernel of the bond-valence barrier
FOREST_TREES = 300
DESCRIPTIONS = ("geometry", "counts", "sites", "invariant", "overlap")
NAMES = {"geometry": "hop length and bottleneck", "counts": "counts about the midpoint",
         "sites": "the two end sites", "invariant": "invariant field about the midpoint",
         "overlap": "fields in the frame of the hop (item BT7)"}
WITH_BVSE = ("geometry", "invariant", "overlap")
FOREST_ROWS = ("geometry", "counts", "invariant")
#: Table 3 of the paper: model, MAE, RMSE (eV), R2, on the test part.  "-tl" marks a model
#: that is given the bond-valence barrier as a feature (for M3GNet: trained on the
#: bond-valence set first).
PUBLISHED = (("RF-tl", 0.25, 0.38, 0.80), ("KR-tl", 0.25, 0.38, 0.80),
             ("XGBRF-tl", 0.25, 0.38, 0.80), ("KR", 0.27, 0.41, 0.76),
             ("SVR-tl", 0.26, 0.42, 0.75), ("M3GNet-tl", 0.28, 0.44, 0.73),
             ("SVR", 0.28, 0.45, 0.72), ("KNN-tl", 0.30, 0.47, 0.69), ("RF", 0.29, 0.48, 0.68),
             ("KNN", 0.31, 0.50, 0.65), ("XGBRF", 0.30, 0.50, 0.65),
             ("M3GNet", 0.40, 0.64, 0.43))
#: the errors of machine-learned potentials that run the elastic band themselves (the text
#: of the paper): without and with transition metals, and after fine-tuning
PUBLISHED_POTENTIALS = "0.07 to 0.08 eV without transition metals, 0.20 to 0.21 eV with " \
                       "them, 0.10 to 0.11 eV after fine-tuning"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The set and its descriptions
# --------------------------------------------------------------------------- #
def load(prepared: str = PREPARED, max_structures: int = 0):
    """The hops as a HopSet, with the bond-valence barrier and the part of the published
    split of every hop, in the order of the hops."""
    import run_litraj as L

    hs, doc = L.load_prepared(prepared, max_structures)
    rows = doc["structures"][:max_structures] if max_structures else doc["structures"]
    bvse = np.array([h["bvse"] for r in rows for h in r["hops"]], float)
    split = np.array([h["split"] for r in rows for h in r["hops"]])
    return hs, doc, bvse, split


def build_descriptions(hs, jobs: int = 8, say: Callable[[str], None] = log
                       ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """The kernel of every description over the hops, and the rows of those that have rows."""
    from cmfqsar.hops import (HopModel, bottleneck, framed_environments, hop_count_features,
                              hop_types)
    from cmfqsar.localenv import count_kernel, local_features, local_kernel
    from cmfqsar.sites import site_kernel
    from run_hops import fieldset_for, site_pair_features

    fs = fieldset_for(hs, "file")
    crystals = hs.crystals(fs)
    t = hs.targets()
    envs = hs.environments(crystals, rows=t.atom, rho=RHO, r_max=R_MAX)
    for r, e in zip(t.atom, envs):
        h = hs.hops[int(r)]
        h.bottleneck, h.bottleneck_at, h.bottleneck_atoms = bottleneck(e, h.vector)
    X = np.array([[hs.hops[r].length, hs.hops[r].bottleneck] for r in t.atom])
    X = (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)
    counts = hop_count_features(envs, hop_types(envs, min_atoms=20))
    sites = site_pair_features(hs, crystals, t.atom, SITE_R)
    local = local_features(envs)
    say(f"  rows: counts {counts.shape[1]}, sites {sites.shape[1]}, invariant "
        f"{local.shape[1]} columns")
    kernels = {"geometry": X @ X.T, "counts": count_kernel(counts),
               "sites": site_kernel(sites), "invariant": local_kernel(local)}
    t0 = time.time()
    framed = framed_environments(hs, crystals, rows=t.atom, rho=RHO, r_max=R_MAX)
    kernels["overlap"] = HopModel.build(framed, fs, n_jobs=jobs).K
    say(f"  overlap kernel in {time.time() - t0:.0f} s")
    return kernels, {"geometry": X, "counts": counts, "invariant": local}


def descriptions(hs, jobs: int = 8, cache: str = CACHE, tag: str = ""):
    """:func:`build_descriptions`, kept in the cache: the overlap kernel takes most of the
    time of a run, and the fits are repeated more often than the kernels."""
    path = os.path.join(cache, f"descriptions{tag}.npz")
    n = len(hs.targets().y)
    if os.path.isfile(path):
        with np.load(path) as z:
            if int(z["n"]) == n:
                return ({k[2:]: z[k] for k in z.files if k.startswith("K_")},
                        {k[2:]: z[k] for k in z.files if k.startswith("X_")})
    kernels, rows = build_descriptions(hs, jobs)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, n=n, **{"K_" + k: v for k, v in kernels.items()},
                        **{"X_" + k: v for k, v in rows.items()})
    return kernels, rows


# --------------------------------------------------------------------------- #
# Kernels and models
# --------------------------------------------------------------------------- #
def unit_scale(K: np.ndarray) -> np.ndarray:
    """A kernel divided by its mean diagonal, so that two kernels can be added."""
    return K / max(float(np.mean(np.diag(K))), 1e-300)


def gaussian(K: np.ndarray, gamma: float) -> np.ndarray:
    """A Gaussian kernel on the rows whose linear kernel is `K`: ``exp(-gamma d^2 / m)`` with
    the squared distance ``d^2 = K_ii + K_jj - 2 K_ij`` and its median ``m`` over the pairs."""
    d = np.diag(K)
    D = np.maximum(d[:, None] + d[None, :] - 2.0 * K, 0.0)
    med = float(np.median(D[np.triu_indices(len(K), 1)]))
    return np.exp(-float(gamma) * D / max(med, 1e-300))


def variants(K: np.ndarray) -> Dict[str, Callable[[], np.ndarray]]:
    """The linear kernel and the Gaussian kernels on top of it, each made when it is asked
    for, so that only one of them is in memory at a time."""
    out: Dict[str, Callable[[], np.ndarray]] = {"linear": lambda: K}
    for g in GAMMAS:
        out[f"Gaussian {g:g}"] = (lambda g=g: gaussian(K, g))
    return out


def ridge_path(K: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               ridges: Sequence[float] = RIDGES) -> np.ndarray:
    """The predictions of kernel ridge regression with a constant for the rows `te`, for
    every ridge (one row each).  A ridge is a multiple of the mean diagonal of the centred
    kernel of the training rows."""
    Ktr = K[np.ix_(tr, tr)]
    mu, mean = Ktr.mean(axis=0), Ktr.mean()
    Kc = Ktr - mu[None, :] - mu[:, None] + mean
    Kte = K[np.ix_(te, tr)]
    Kcte = Kte - mu[None, :] - Kte.mean(axis=1)[:, None] + mean
    e, V = np.linalg.eigh(Kc)
    e = np.maximum(e, 0.0)
    proj = V.T @ (y[tr] - y[tr].mean())
    lam = np.asarray(ridges, float) * max(float(np.trace(Kc)) / len(tr), 1e-300)
    alphas = (proj[None, :] / (e[None, :] + lam[:, None])) @ V.T
    return alphas @ Kcte.T + y[tr].mean()


def group_folds(groups: np.ndarray, k: int, rng) -> List[np.ndarray]:
    """`k` folds, as positions in `groups`, that keep the hops of a structure together."""
    names = rng.permutation(np.unique(groups))
    return [np.sort(np.where(np.isin(groups, part))[0]) for part in np.array_split(names, k)]


#: whether the ridge of a final fit was at an end of its grid (see `run_fluorescence.py`)
EDGES: List[int] = []


def kernel_arm(pool: Dict[str, Callable[[], np.ndarray]], y: np.ndarray, groups: np.ndarray,
               tr: np.ndarray, te: np.ndarray, rng) -> Tuple[np.ndarray, str]:
    """Kernel ridge with the kernel of `pool` and the ridge chosen inside the rows `tr`, on
    folds that hold out whole structures; the prediction for `te` and what was chosen."""
    inner = group_folds(groups[tr], INNER, rng)
    best = None
    for name, make in pool.items():
        K = make()
        err = np.zeros(len(RIDGES))
        for f in inner:
            te_in = tr[f]
            tr_in = np.setdiff1d(tr, te_in)
            P = ridge_path(K, y, tr_in, te_in)
            err += np.sum((P - y[te_in][None, :]) ** 2, axis=1)
        k = int(np.argmin(err))
        if best is None or err[k] < best[0]:
            best = (float(err[k]), name, k)
    _, name, k = best
    EDGES.append(int(k in (0, len(RIDGES) - 1)))
    return ridge_path(pool[name](), y, tr, te)[k], name


def with_bvse(K: np.ndarray, b: np.ndarray) -> Dict[str, Callable[[], np.ndarray]]:
    """The kernels of a description with the bond-valence barrier added as a second kernel,
    for every weight of :data:`BVSE_WEIGHTS`, each linear and with a Gaussian on top."""
    z = (b - b.mean()) / max(float(b.std()), 1e-12)
    Kb = unit_scale(np.outer(z, z) + 1.0)
    out: Dict[str, Callable[[], np.ndarray]] = {}
    base = unit_scale(K)
    for w in BVSE_WEIGHTS:
        for name, make in variants(base + w * Kb).items():
            out[f"weight {w:g}, {name}"] = make
    return out


def forest_arm(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               seed: int = SEED) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    m = RandomForestRegressor(n_estimators=FOREST_TREES, n_jobs=-1, random_state=seed)
    return m.fit(X[tr], y[tr]).predict(X[te])


def scores(p: np.ndarray, y: np.ndarray) -> dict:
    r = np.asarray(p, float) - np.asarray(y, float)
    return {"mae": float(np.mean(np.abs(r))), "rmse": float(np.sqrt(np.mean(r ** 2))),
            "r2": float(1.0 - np.sum(r ** 2) / np.sum((y - np.mean(y)) ** 2))}


def paired_mae(a: np.ndarray, b: np.ndarray, y: np.ndarray, groups: np.ndarray,
               runs: int = BOOTSTRAP, seed: int = SEED) -> dict:
    """The mean absolute error of the predictions `a` minus that of `b`, with a 95 % interval
    from a bootstrap that draws whole structures."""
    ea, eb = np.abs(a - y), np.abs(b - y)
    ids = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in ids}
    sums = np.array([[ea[index[g]].sum(), eb[index[g]].sum(), len(index[g])] for g in ids])
    rng = np.random.default_rng(seed)
    draws = sums[rng.integers(0, len(ids), size=(runs, len(ids)))].sum(axis=1)
    d = (draws[:, 0] - draws[:, 1]) / draws[:, 2]
    return {"mean": float(ea.mean() - eb.mean()), "low": float(np.percentile(d, 2.5)),
            "high": float(np.percentile(d, 97.5))}


def mae_interval(p: np.ndarray, y: np.ndarray, groups: np.ndarray, runs: int = BOOTSTRAP,
                 seed: int = SEED) -> List[float]:
    """A 95 % interval of a mean absolute error, from a bootstrap that draws whole
    structures."""
    e = np.abs(np.asarray(p, float) - np.asarray(y, float))
    ids = np.unique(groups)
    sums = np.array([[e[groups == g].sum(), np.sum(groups == g)] for g in ids])
    rng = np.random.default_rng(seed)
    draws = sums[rng.integers(0, len(ids), size=(runs, len(ids)))].sum(axis=1)
    m = draws[:, 0] / draws[:, 1]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def arms(kernels: Dict[str, np.ndarray], rows: Dict[str, np.ndarray], bvse: np.ndarray
         ) -> Dict[str, Tuple[str, object]]:
    """Every model of the table: its key, and how it is fitted."""
    out: Dict[str, Tuple[str, object]] = {}
    z = (bvse - bvse.mean()) / max(float(bvse.std()), 1e-12)
    out["bvse|kernel"] = ("kernel", variants(np.outer(z, z)))
    out["bvse|forest"] = ("forest", bvse[:, None])
    for d in DESCRIPTIONS:
        if d in kernels:
            out[f"{d}|kernel"] = ("kernel", variants(kernels[d]))
            if d in WITH_BVSE:
                out[f"{d}+bvse|kernel"] = ("kernel", with_bvse(kernels[d], bvse))
    for d in FOREST_ROWS:
        if d in rows:
            out[f"{d}|forest"] = ("forest", rows[d])
            out[f"{d}+bvse|forest"] = ("forest", np.hstack([rows[d], bvse[:, None]]))
    return out


def predictions(todo: Dict[str, Tuple[str, object]], y: np.ndarray, groups: np.ndarray,
                parts: Sequence[Tuple[np.ndarray, np.ndarray]], rng,
                say: Callable[[str], None]) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
    """The predictions of every model for the test rows of `parts` (pairs of training and
    test rows), and what the kernel models chose."""
    pred = {key: np.full(len(y), np.nan) for key in todo}
    chosen: Dict[str, dict] = {}
    for key, (kind, what) in todo.items():
        t0 = time.time()
        for tr, te in parts:
            if kind == "kernel":
                pred[key][te], name = kernel_arm(what, y, groups, tr, te, rng)
                c = chosen.setdefault(key, {})
                c[name] = c.get(name, 0) + 1
            else:
                pred[key][te] = forest_arm(np.asarray(what, float), y, tr, te)
        say(f"    {key}: {time.time() - t0:.0f} s")
    return pred, chosen


def run(results: str = RESULTS, prepared: str = PREPARED, max_structures: int = 0,
        jobs: int = 8, folds: int = FOLDS, tag: str = "", cache: str = CACHE) -> dict:
    t0 = time.time()
    EDGES.clear()
    hs, doc, bvse, split = load(prepared, max_structures)
    t = hs.targets()
    y = np.asarray(t.y, float)
    groups = np.asarray(t.molecule, int)
    order = np.asarray(t.atom, int)
    bvse, split = bvse[order], split[order]
    log(f"{len(y)} hops in {len(np.unique(groups))} structures; DFT barriers mean "
        f"{y.mean():.3f}, standard deviation {y.std(ddof=1):.3f} eV")
    kernels, rows = descriptions(hs, jobs, cache, tag)
    todo = arms(kernels, rows, bvse)
    rng = np.random.default_rng(SEED)
    out = {"source": doc["source"], "reference": doc["reference"], "licence": doc["licence"],
           "hops": int(len(y)), "structures": int(len(np.unique(groups))),
           "index_hops": doc["n_index_hops"], "index_structures": doc["n_index_structures"],
           "dropped_hops": doc["dropped_hops"],
           "y_mean": float(y.mean()), "y_sd": float(y.std(ddof=1)),
           "bvse_mean": float(bvse.mean()), "bvse_sd": float(bvse.std(ddof=1)),
           "bvse_correlation": float(np.corrcoef(bvse, y)[0, 1]),
           "folds": folds, "protocols": {}}
    test = np.where(split == "test")[0]
    train = np.where(split != "test")[0]
    shared = set(groups[test]) & set(groups[train])
    out["published_split"] = {"train": int(len(train)), "test": int(len(test)),
                              "structures_in_both_parts": len(shared)}
    protocols = {"published split": [(train, test)],
                 "folds": [(np.setdiff1d(np.arange(len(y)), f), f)
                           for f in group_folds(groups, folds, rng)]}
    kept = {}
    for name, parts in protocols.items():
        log(f"  {name}")
        pred, chosen = predictions(todo, y, groups, parts, rng, log)
        rows_te = np.concatenate([te for _, te in parts])
        mean_fit = np.full(len(y), np.nan)
        for tr, te in parts:
            mean_fit[te] = y[tr].mean()
        pred["mean of the fit"], pred["bvse as it is"] = mean_fit, bvse
        table = {key: scores(p[rows_te], y[rows_te]) for key, p in pred.items()}
        for key, p in pred.items():
            table[key]["mae_interval"] = mae_interval(p[rows_te], y[rows_te],
                                                      groups[rows_te])
        out["protocols"][name] = {"scores": table, "chosen": {
            k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:3]) for k, v in chosen.items()}}
        kept[name] = (pred, rows_te)
    pred, rows_te = kept["folds"]
    pairs = (("overlap|kernel", "invariant|kernel"), ("overlap|kernel", "geometry|kernel"),
             ("overlap+bvse|kernel", "overlap|kernel"),
             ("invariant+bvse|kernel", "invariant|kernel"),
             ("overlap+bvse|kernel", "bvse|kernel"),
             ("invariant|kernel", "invariant|forest"),
             ("overlap+bvse|kernel", "invariant+bvse|forest"),
             ("invariant+bvse|forest", "invariant|forest"),
             ("invariant|forest", "overlap|kernel"),
             ("invariant|forest", "bvse|kernel"),
             ("overlap|kernel", "bvse|kernel"))
    out["differences"] = {
        f"{a} minus {b}": paired_mae(pred[a][rows_te], pred[b][rows_te], y[rows_te],
                                     groups[rows_te])
        for a, b in pairs if a in pred and b in pred}
    out["ridge_at_an_edge"] = float(np.mean(EDGES)) if EDGES else 0.0
    os.makedirs(cache, exist_ok=True)       # the predictions, for a later look
    np.savez_compressed(os.path.join(cache, f"predictions{tag}.npz"), y=y, groups=groups,
                        bvse=bvse, split=split,
                        **{f"{name}|{key}": p for name, (pr, _) in kept.items()
                           for key, p in pr.items()})
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, f"run{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def arm_name(key: str) -> str:
    if key in ("mean of the fit",):
        return "the mean of the fit"
    if key == "bvse as it is":
        return "the bond-valence barrier taken as the prediction"
    what, kind = key.split("|")
    model = "kernel ridge" if kind == "kernel" else "random forest"
    if what == "bvse":
        return f"{model} on the bond-valence barrier alone"
    plus = what.endswith("+bvse")
    d = what[:-5] if plus else what
    return f"{model}, {NAMES[d]}" + (", with the bond-valence barrier" if plus else "")


def report(results: str = RESULTS, tag: str = "") -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, f"run{tag}.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    pub, cv = d["protocols"]["published split"], d["protocols"]["folds"]
    s = d["published_split"]
    lines = ["# Hop barriers from density functional theory (nebDFT2k)",
             "",
             f"Data: {d['reference']}, {d['source']} ({d['licence']} licence). "
             f"{d['hops']} hops in {d['structures']} structures are used, of "
             f"{d['index_hops']} hops in {d['index_structures']} structures of the set; the "
             "others were left out when the set was prepared ("
             + "; ".join(f"{v} hops: {k}" for k, v in d["dropped_hops"].items() if v) + ").",
             f"The DFT barriers have the mean {d['y_mean']:.3f} eV and the standard deviation "
             f"{d['y_sd']:.3f} eV. The bond-valence barriers of the same hops have the mean "
             f"{d['bvse_mean']:.3f} eV and the standard deviation {d['bvse_sd']:.3f} eV, and "
             f"correlate with the DFT barriers with r = {d['bvse_correlation']:.3f}.",
             "",
             f"Published split: {s['train']} hops to fit and {s['test']} to test, "
             f"{s['structures_in_both_parts']} structures in both parts. Folds: {d['folds']} "
             "folds that hold out whole structures, over all hops.",
             "",
             "An interval is a 95 % interval of the mean absolute error (MAE) from a "
             "bootstrap that draws whole structures.",
             "",
             "| Model | Published test part: MAE, eV | RMSE, eV | R2 | Folds: MAE, eV | "
             "RMSE, eV | R2 |",
             "|---|---|---|---|---|---|---|"]
    keys = sorted(pub["scores"], key=lambda k: pub["scores"][k]["rmse"])
    for key in keys:
        a, b = pub["scores"][key], cv["scores"][key]
        ia, ib = a["mae_interval"], b["mae_interval"]
        lines.append(f"| {arm_name(key)} | {a['mae']:.3f} [{ia[0]:.3f}, {ia[1]:.3f}] | "
                     f"{a['rmse']:.3f} | {a['r2']:.2f} | {b['mae']:.3f} [{ib[0]:.3f}, "
                     f"{ib[1]:.3f}] | {b['rmse']:.3f} | {b['r2']:.2f} |")
    lines += ["",
              "## The published models on the test part (Table 3 of the paper)",
              "",
              "| Model of the paper | MAE, eV | RMSE, eV | R2 |",
              "|---|---|---|---|"]
    for name, mae, rmse, r2 in PUBLISHED:
        lines.append(f"| {name} | {mae:.2f} | {rmse:.2f} | {r2:.2f} |")
    lines += ["",
              "\"-tl\" marks a model of the paper that is given the bond-valence barrier. "
              "Machine-learned potentials that run the elastic band themselves reach "
              f"{PUBLISHED_POTENTIALS} (mean absolute error, the text of the paper).",
              "",
              "## Differences in the mean absolute error, folds",
              "",
              "A 95 % interval from a bootstrap that draws whole structures.",
              "",
              "| First model minus second model | Difference, eV |",
              "|---|---|"]
    for key, v in d["differences"].items():
        a, b = key.split(" minus ")
        lines.append(f"| {arm_name(a)} minus {arm_name(b)} | {v['mean']:+.3f} "
                     f"[{v['low']:+.3f}, {v['high']:+.3f}] |")
    lines += ["",
              "## What the kernel models chose (folds)",
              ""]
    for key, v in cv["chosen"].items():
        lines.append(f"- {arm_name(key)}: " + ", ".join(f"{k} ({n})" for k, n in v.items()))
    lines += ["",
              f"The ridge of a final fit was at an end of its grid in "
              f"{100 * d['ridge_at_an_edge']:.0f} % of the fits.",
              ""]
    text = "\n".join(lines)
    with open(os.path.join(results, "table.md" if not tag else f"table{tag}.md"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--max-structures", type=int, default=0,
                    help="read only the first N structures (a quick look)")
    ap.add_argument("--jobs", type=int, default=8, help="processes for the overlap kernel")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)
    tag = f"_{a.max_structures}" if a.max_structures else ""
    if not a.report:
        run(a.results, max_structures=a.max_structures, jobs=a.jobs, tag=tag)
    print(report(a.results, tag))
    return 0


if __name__ == "__main__":
    sys.exit(main())
