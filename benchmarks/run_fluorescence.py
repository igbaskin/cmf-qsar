#!/usr/bin/env python
"""
Spectra with two axes as fields on a plane (plan item SP4).

`cmfqsar/spectrum2d.py` gives a spectrum with two axes a kernel with one width for each axis.
This benchmark measures it on the two sets of fluorescence maps that
`prepare_fluorescence.py` fetches, and on simulated peak lists.

The reference
-------------
Partial least squares on the map with its values in one long row ("unfolded"), which is what
the literature on such maps reports: PLS regression for a measured property, and PLS
discriminant analysis (PLS-DA) for a class, with one PLS model for each class against the
others and the class of the largest score.  PLS is computed here from the kernel of the rows
(kernel PLS), which gives the predictions of PLS on the rows themselves and needs only the
matrix of the samples.

The kernel models
-----------------
For a measured property, kernel ridge regression.  For a class, the kernel Fisher
discriminant of `cmfqsar/model.py`, one for each class against the others; its score is
measured in within-class standard deviations, so the largest score names the class.

Both get the same descriptions of a map: the map broadened by a Gaussian with one width
along the excitation axis and one along the emission axis, each map as it is or divided by
its own length (the cosine form).  The widths, the form and the setting of the model (the
number of components, or the ridge) are chosen by cross-validation inside the samples of the
fit.  "PLS on the map as measured" is the reference without any of these choices but the
number of components.

Part 1: honey
-------------
104 honeys of four botanical origins; five folds with the classes in proportion, repeated.
Then the authentication case: a description of the real honeys alone
(`cmfqsar/oneclass.py`), and the six false honeys scored against the real honeys that the
description has not seen.

Part 2: olive oil
-----------------
The absorbance at 232 and at 268 nm of 240 oil samples (24 oils at 10 steps of ageing) from
their maps, and the class "within the limits of the regulation for extra virgin oil".  The
30 samples of one oil are not independent, so whole oils are held out; the same with random
folds shows what that costs.  Every sample has three maps: the one of the vial whose
absorbance was measured, and the mean of the three, which is the kernel of the ensemble.

Part 3: simulated peak lists
----------------------------
The two measured sets are smooth maps.  The case that the item is about, sharp peaks whose
places move a little from sample to sample, is simulated: two classes of peak lists that
differ in the intensity of a few peaks.  The closed-form kernel of the peak lists is compared
with PLS-DA on the same peaks put into bins of several sizes.

Usage
-----
    python benchmarks/prepare_fluorescence.py      # once: downloads 77 MB
    python benchmarks/run_fluorescence.py          # the benchmark, then the table
    python benchmarks/run_fluorescence.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import spectrum2d as S2        # noqa: E402
from cmfqsar import model as MODEL          # noqa: E402
from cmfqsar import oneclass as ONE         # noqa: E402
import prepare_fluorescence as PREP          # noqa: E402
import run_spectra_measured as M             # noqa: E402

RESULTS = os.path.join(HERE, "results", "fluorescence")
WIDTHS_EX = (0.0, 10.0, 20.0)              # nm, along the excitation axis
WIDTHS_EM = (0.0, 4.0, 8.0, 16.0)          # nm, along the emission axis
FORMS = (False, True)                      # the cosine form or not
RIDGES = tuple(10.0 ** e for e in np.arange(-10.0, 2.01, 1.0))
PLS_MAX = 20
FOLDS = 5
INNER = 4
SEED = 20261112
HONEY_REPEATS = 20
OLIVE_REPEATS = 10
OLIVE_FOLDS = 6                            # four oils in a fold
#: the scattered excitation light that is taken out, in nm around the first and the second
#: order (`cmfqsar.spectrum2d.scatter_mask`)
SCATTER = {"honey": (25.0, 40.0), "olive": (10.0, 10.0)}
HONEY_BIN = 4                              # emission points in one bin: 0.5 nm -> 2 nm
ONE_CLASS = ("ocmean", "ocknn", "ockpca", "ocgp")
#: the limits of the European Union regulation for extra virgin olive oil
LIMIT_232, LIMIT_268, LIMIT_DELTA = 2.50, 0.22, 0.01
AS_MEASURED = "as measured"


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The two sets
# --------------------------------------------------------------------------- #
def without_scatter(maps: np.ndarray, excitation: np.ndarray, emission: np.ndarray,
                    first: float, second: float) -> np.ndarray:
    """The maps with the scattered excitation light set to NaN."""
    mask = (S2.scatter_mask(excitation, emission, first, orders=(1,))
            | S2.scatter_mask(excitation, emission, second, orders=(2,), below=False))
    out = np.array(maps, dtype=float)
    out[:, mask] = np.nan
    return out


def honey() -> dict:
    d = PREP.load("honey")
    maps = without_scatter(d["maps"], d["excitation"], d["emission"], *SCATTER["honey"])
    em = d["emission"]
    n = (len(em) // HONEY_BIN) * HONEY_BIN
    return {"maps": S2.binned(_keep_nan(maps), 1, HONEY_BIN),
            "excitation": d["excitation"],
            "emission": em[:n].reshape(-1, HONEY_BIN).mean(axis=1),
            "label": d["label"], "false": d["false"]}


def _keep_nan(maps: np.ndarray) -> np.ndarray:
    return maps


def olive() -> dict:
    d = PREP.load("olive")
    maps = without_scatter(d["maps"], d["excitation"], d["emission"], *SCATTER["olive"])
    measured = np.where(np.isfinite(d["absorbance"][:, 0]))[0]
    key = [(o, s) for o, s in zip(d["oil"], d["step"])]
    members = np.array([[i for i, k in enumerate(key) if k == key[m]] for m in measured])
    A = d["absorbance"][measured]
    delta = A[:, 2] - 0.5 * (A[:, 1] + A[:, 3])
    return {"maps": maps, "excitation": d["excitation"], "emission": d["emission"],
            "measured": measured, "members": members, "oil": d["oil"][measured],
            "step": d["step"][measured], "k232": A[:, 0], "k268": A[:, 2], "delta": delta,
            "within": (A[:, 0] <= LIMIT_232) & (A[:, 2] <= LIMIT_268)
            & (np.abs(delta) <= LIMIT_DELTA)}


# --------------------------------------------------------------------------- #
# Descriptions as kernels
# --------------------------------------------------------------------------- #
def name_of(sx: float, sy: float, cosine: bool) -> str:
    return f"{sx:g}|{sy:g}|{int(cosine)}"


def raw_kernels(maps: np.ndarray, step_x: float, step_y: float,
                widths_x: Sequence[float] = WIDTHS_EX,
                widths_y: Sequence[float] = WIDTHS_EM) -> Dict[Tuple[float, float], np.ndarray]:
    """The kernel of the broadened maps, not normalised, for every pair of widths."""
    out = {}
    for sx in widths_x:
        for sy in widths_y:
            F = S2.rows(S2.smoothed(maps, step_x, step_y, sx, sy))
            out[(sx, sy)] = F @ F.T
    return out


def cosine(K: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / d[:, None] / d[None, :]


def descriptions(raw: Dict[Tuple[float, float], np.ndarray],
                 pick: Optional[np.ndarray] = None,
                 mean_of: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Every description as a kernel of the samples.

    `pick` takes rows of the kernels.  `mean_of` holds, for every sample, the rows whose mean
    map is the sample (an ensemble); the kernel of two mean maps is the mean of the kernels
    of their members, and the cosine form is made after that.
    """
    out = {}
    for (sx, sy), K in raw.items():
        if mean_of is not None:
            K = K[mean_of][:, :, mean_of].mean(axis=(1, 3))
        elif pick is not None:
            K = K[np.ix_(pick, pick)]
        scale = float(np.mean(np.diag(K))) or 1.0
        out[name_of(sx, sy, False)] = K / scale
        out[name_of(sx, sy, True)] = cosine(K)
    return out


# --------------------------------------------------------------------------- #
# Models on a kernel
# --------------------------------------------------------------------------- #
def centred(K: np.ndarray, tr: np.ndarray, te: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """The kernel of the rows `tr`, and that of `te` against `tr`, with the mean of the
    training rows taken out of the features: the constant of a linear model."""
    Ktr = K[np.ix_(tr, tr)]
    mu, mean = Ktr.mean(axis=0), Ktr.mean()
    Kte = K[np.ix_(te, tr)]
    return (Ktr - mu[None, :] - mu[:, None] + mean,
            Kte - mu[None, :] - Kte.mean(axis=1)[:, None] + mean)


def ridge_alphas(Kc: np.ndarray, y: np.ndarray,
                 ridges: Sequence[float] = RIDGES) -> np.ndarray:
    """The dual coefficients of kernel ridge regression for every ridge (one row each); the
    target is centred, and a ridge is a multiple of the mean diagonal of the kernel."""
    e, V = np.linalg.eigh(Kc)
    e = np.maximum(e, 0.0)
    proj = V.T @ (y - y.mean())
    lam = np.asarray(ridges, float) * max(float(np.trace(Kc)) / len(Kc), 1e-300)
    return (proj[None, :] / (e[None, :] + lam[:, None])) @ V.T


def pls_alphas(Kc: np.ndarray, y: np.ndarray, most: int = PLS_MAX) -> np.ndarray:
    """PLS for one property from the centred kernel of the rows (kernel PLS), with 1, 2, ...,
    `most` components: one row of dual coefficients each.  The prediction is the centred
    test kernel times the row, plus the mean of the target; it equals that of PLS on the
    rows themselves."""
    n = len(Kc)
    yc = y - y.mean()
    Kd, r = Kc.copy(), yc.copy()
    T, U = [], []
    out = np.zeros((most, n))
    for k in range(most):
        t = Kd @ r
        norm = float(np.sqrt(t @ t))
        if norm < 1e-12 * max(1.0, float(np.trace(Kc))):
            out[k:] = out[k - 1] if k else 0.0
            break
        t = t / norm
        T.append(t)
        U.append(r.copy())
        Kd = Kd - np.outer(t, t @ Kd)
        Kd = Kd - np.outer(Kd @ t, t)
        r = r - t * float(t @ r)
        Tm, Um = np.array(T).T, np.array(U).T
        out[k] = Um @ np.linalg.solve(Tm.T @ Kc @ Um, Tm.T @ yc)
    return out


def regression_path(K: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                    model: str) -> np.ndarray:
    """The predictions for the rows `te` for every setting of the model (one row each)."""
    Kc, Kte = centred(K, tr, te)
    A = pls_alphas(Kc, y[tr], min(PLS_MAX, len(tr) - 2)) if model == "pls" \
        else ridge_alphas(Kc, y[tr])
    return A @ Kte.T + y[tr].mean()


def class_scores(K: np.ndarray, labels: np.ndarray, classes: Sequence, tr: np.ndarray,
                 te: np.ndarray, model: str) -> np.ndarray:
    """The score of every class for the rows `te`, for every setting of the model:
    an array (setting, row, class).  One model for each class against the others."""
    per_class = []
    for c in classes:
        y = np.where(labels == c, 1.0, -1.0)
        if model == "pls":
            per_class.append(regression_path(K, y, tr, te, "pls"))
        else:
            Ktr, Kte = K[np.ix_(tr, tr)], K[np.ix_(te, tr)]
            rows_ = []
            for lam in RIDGES:
                a, b = MODEL.kfda_fit(Ktr, y[tr], lam)
                rows_.append(Kte @ a + b)
            per_class.append(np.array(rows_))
    return np.stack(per_class, axis=2)


def balanced_accuracy(truth: np.ndarray, said: np.ndarray, classes: Sequence) -> float:
    return float(np.mean([np.mean(said[truth == c] == c) for c in classes if np.any(truth == c)]))


def folds_by_class(labels: np.ndarray, k: int, rng) -> List[np.ndarray]:
    """`k` folds with every class spread over them in proportion."""
    folds: List[List[int]] = [[] for _ in range(k)]
    for c in np.unique(labels):
        idx = rng.permutation(np.where(labels == c)[0])
        for j, i in enumerate(idx):
            folds[j % k].append(int(i))
    return [np.sort(np.array(f)) for f in folds]


def folds_by_group(groups: np.ndarray, k: int, rng) -> List[np.ndarray]:
    """`k` folds that keep the samples of a group together."""
    names = rng.permutation(np.unique(groups))
    return [np.sort(np.where(np.isin(groups, part))[0]) for part in np.array_split(names, k)]


def chosen_regression(kernels: Dict[str, np.ndarray], y: np.ndarray, tr: np.ndarray,
                      inner: Sequence[np.ndarray], model: str) -> Tuple[str, int]:
    """The description and the setting with the smallest squared error inside `tr`; `inner`
    holds folds as positions in `tr`."""
    best = None
    for name, K in kernels.items():
        err = 0.0
        for f in inner:
            te_in = tr[f]
            tr_in = np.setdiff1d(tr, te_in)
            P = regression_path(K, y, tr_in, te_in, model)
            e = np.sum((P - y[te_in][None, :]) ** 2, axis=1)
            err = e if isinstance(err, float) else err[:len(e)] + e[:len(err)]
        k = int(np.argmin(err))
        if best is None or err[k] < best[0]:
            best = (float(err[k]), name, k)
    return best[1], best[2]


def chosen_classifier(kernels: Dict[str, np.ndarray], labels: np.ndarray, classes: Sequence,
                      tr: np.ndarray, inner: Sequence[np.ndarray], model: str
                      ) -> Tuple[str, int]:
    """The description and the setting with the most right answers inside `tr`; between
    equal ones the larger margin of the right class decides."""
    best = None
    cls = np.array(list(classes))
    for name, K in kernels.items():
        right, margin = 0.0, 0.0
        for f in inner:
            te_in = tr[f]
            tr_in = np.setdiff1d(tr, te_in)
            S = class_scores(K, labels, classes, tr_in, te_in, model)
            truth = np.array([int(np.where(cls == v)[0][0]) for v in labels[te_in]])
            said = S.argmax(axis=2)
            r = (said == truth[None, :]).sum(axis=1).astype(float)
            own = np.take_along_axis(S, truth[None, :, None], axis=2)[:, :, 0]
            other = np.where(np.arange(len(cls))[None, None, :] == truth[None, :, None],
                             -np.inf, S).max(axis=2)
            scale = np.maximum(S.std(axis=(1, 2)), 1e-12)
            mg = (own - other).mean(axis=1) / scale
            right = r if isinstance(right, float) else right[:len(r)] + r[:len(right)]
            margin = mg if isinstance(margin, float) else margin[:len(mg)] + mg[:len(margin)]
        k = int(np.lexsort((-margin, -right))[0])
        key = (float(right[k]), float(margin[k]))
        if best is None or key > best[0]:
            best = (key, name, k)
    return best[1], best[2]


#: for every model, whether the setting of a final fit was at the edge of its grid: the
#: largest number of PLS components, or the smallest or the largest ridge.  A model that is
#: often at an edge was not given enough room, and a comparison with it is not fair.
EDGES: Dict[str, List[int]] = {}


def _setting(K: np.ndarray, k: int, tr: np.ndarray, model: str) -> int:
    """A whole training part allows more PLS components than a fold of it did."""
    last = (min(PLS_MAX, len(tr) - 2) if model == "pls" else len(RIDGES)) - 1
    k = min(k, last)
    EDGES.setdefault(model, []).append(int(k == last or (model != "pls" and k == 0)))
    return k


def _edges() -> Dict[str, float]:
    """The share of final fits at an edge of the grid since the last call, by model."""
    out = {m: float(np.mean(v)) for m, v in EDGES.items()}
    EDGES.clear()
    return out


# --------------------------------------------------------------------------- #
# Part 1: honey
# --------------------------------------------------------------------------- #
def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """The share of pairs of a positive and a negative score in the right order."""
    p, q = np.asarray(positive, float)[:, None], np.asarray(negative, float)[None, :]
    return float(np.mean((p > q) + 0.5 * (p == q)))


def run_honey(repeats: int = HONEY_REPEATS, **grid) -> dict:
    t0 = time.time()
    EDGES.clear()
    d = honey()
    step_x = float(d["excitation"][1] - d["excitation"][0])
    step_y = float(d["emission"][1] - d["emission"][0])
    raw = raw_kernels(d["maps"], step_x, step_y, **grid)
    real = np.where(~d["false"])[0]
    false = np.where(d["false"])[0]
    labels = d["label"]
    classes = sorted(set(labels[real]))
    kernels = descriptions(raw)
    plain = {AS_MEASURED: kernels[name_of(0.0, 0.0, False)]}
    rng = np.random.default_rng(SEED)
    acc: Dict[str, List[float]] = {}
    bal: Dict[str, List[float]] = {}
    chosen: Dict[str, Dict[str, int]] = {}
    confusion = {m: np.zeros((len(classes), len(classes)), int) for m in ("pls", "kernel")}
    one: Dict[str, List[float]] = {}
    for r in range(repeats):
        folds = folds_by_class(labels[real], FOLDS, rng)
        said = {"pls": np.empty(len(real), dtype=object),
                "kernel": np.empty(len(real), dtype=object),
                "pls " + AS_MEASURED: np.empty(len(real), dtype=object)}
        scores = {kind + "|" + desc: (np.zeros(len(real)), np.zeros(len(false)))
                  for kind in ONE_CLASS for desc in ("plain", "broadened")}
        for f in folds:
            te = real[f]
            tr = np.setdiff1d(real, te)
            inner = folds_by_class(labels[tr], INNER, rng)
            for model, pool, key in (("pls", kernels, "pls"), ("kernel", kernels, "kernel"),
                                     ("pls", plain, "pls " + AS_MEASURED)):
                name, k = chosen_classifier(pool, labels, classes, tr, inner, model)
                S = class_scores(pool[name], labels, classes, tr, te, model)
                said[key][f] = np.array(classes)[S[_setting(pool[name], k, tr, model)]
                                                  .argmax(axis=1)]
                if key in ("pls", "kernel"):
                    chosen.setdefault(key, {})
                    chosen[key][name] = chosen[key].get(name, 0) + 1
            # the authentication case: the real honeys of the fit describe "honey"
            for desc, K in (("plain", kernels[name_of(0.0, 0.0, True)]),
                            ("broadened", kernels[name_of(max(grid.get("widths_x", WIDTHS_EX)),
                                                          max(grid.get("widths_y", WIDTHS_EM)),
                                                          True)])):
                for kind in ONE_CLASS:
                    scorer = ONE.make(kind).fit(K[np.ix_(tr, tr)])
                    scores[kind + "|" + desc][0][f] = scorer.decision(
                        K[np.ix_(te, tr)], np.diag(K)[te])
                    scores[kind + "|" + desc][1][:] += scorer.decision(
                        K[np.ix_(false, tr)], np.diag(K)[false]) / len(folds)
        truth = labels[real]
        for key, v in said.items():
            v = v.astype(str)
            acc.setdefault(key, []).append(float(np.mean(v == truth)))
            bal.setdefault(key, []).append(balanced_accuracy(truth, v, classes))
            if key in confusion:
                for i, a in enumerate(classes):
                    for j, b in enumerate(classes):
                        confusion[key][i, j] += int(np.sum((truth == a) & (v == b)))
        for key, (s_real, s_false) in scores.items():
            one.setdefault(key, []).append(auc(s_real, s_false))
        log(f"  honey: repeat {r + 1} of {repeats}, {time.time() - t0:.0f} s")
    majority = max(np.mean(labels[real] == c) for c in classes)
    out = {"samples": int(len(real)), "false": int(len(false)), "classes": classes,
           "counts": [int(np.sum(labels[real] == c)) for c in classes],
           "points": [int(d["maps"].shape[1]), int(d["maps"].shape[2])],
           "repeats": repeats, "largest_class": float(majority),
           "accuracy": {k: _mean_sd(v) for k, v in acc.items()},
           "balanced_accuracy": {k: _mean_sd(v) for k, v in bal.items()},
           "kernel_minus_pls": M.mean_difference(bal["kernel"], bal["pls"], seed=SEED),
           "kernel_minus_pls_as_measured": M.mean_difference(
               bal["kernel"], bal["pls " + AS_MEASURED], seed=SEED),
           "chosen": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:5])
                      for k, v in chosen.items()},
           "confusion": {k: v.tolist() for k, v in confusion.items()},
           "false_against_real_auc": {k: _mean_sd(v) for k, v in one.items()},
           "setting_at_an_edge": _edges(),
           "seconds": round(time.time() - t0, 1)}
    out["co_map"] = honey_co_map(d, raw, labels, classes, real, out["chosen"]["kernel"])
    return out


def honey_co_map(d: dict, raw: dict, labels: np.ndarray, classes: Sequence, real: np.ndarray,
                 chosen: Dict[str, int]) -> dict:
    """For every class, the places of the map that speak most for and against it: the co-map
    of the kernel Fisher discriminant of that class against the others, fitted on all real
    honeys with the description that was chosen most often and a middle ridge."""
    sx, sy, cos = next(iter(chosen)).split("|")
    sx, sy, cos = float(sx), float(sy), bool(int(cos))
    step_x = float(d["excitation"][1] - d["excitation"][0])
    step_y = float(d["emission"][1] - d["emission"][0])
    maps = S2.smoothed(d["maps"][real], step_x, step_y, sx, sy)
    K = S2.map_kernel(maps, normalize=cos)
    out = {"description": name_of(sx, sy, cos), "classes": {}}
    for c in classes:
        y = np.where(labels[real] == c, 1.0, -1.0)
        a, _ = MODEL.kfda_fit(K, y, 1e-3)
        w = S2.co_map(maps, a, normalize=cos)
        effect = w * np.where(np.isfinite(maps), maps, 0.0).std(axis=0)
        out["classes"][c] = [
            {"excitation": p["x"], "emission": p["y"], "sign": "+" if p["value"] > 0 else "-"}
            for p in S2.strongest_places(d["excitation"], d["emission"], effect, n=3,
                                         separation=(30.0, 40.0))]
    return out


# --------------------------------------------------------------------------- #
# Part 2: olive oil
# --------------------------------------------------------------------------- #
def _mean_sd(values: Sequence[float]) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values))}


def rmse(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def one_number(maps: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    """The reference with one number for a map: a linear regression of the property on the
    sum of all intensities."""
    s = np.where(np.isfinite(maps), maps, 0.0).sum(axis=(1, 2))
    slope, const = np.polyfit(s[tr], y[tr], 1)
    return slope * s[te] + const


def run_olive(repeats: int = OLIVE_REPEATS, **grid) -> dict:
    t0 = time.time()
    EDGES.clear()
    d = olive()
    step_x = float(d["excitation"][1] - d["excitation"][0])
    step_y = float(d["emission"][1] - d["emission"][0])
    raw = raw_kernels(d["maps"], step_x, step_y, **grid)
    inputs = {"one map": descriptions(raw, pick=d["measured"]),
              "mean of three maps": descriptions(raw, mean_of=d["members"])}
    plain = {AS_MEASURED: inputs["one map"][name_of(0.0, 0.0, False)]}
    n = len(d["measured"])
    targets = {"k232": d["k232"], "k268": d["k268"]}
    within = np.where(d["within"], "within", "outside")
    classes = ["outside", "within"]
    rng = np.random.default_rng(SEED)
    err: Dict[str, List[float]] = {}
    hit: Dict[str, List[float]] = {}
    chosen: Dict[str, Dict[str, int]] = {}
    one_map = d["maps"][d["measured"]]
    for r in range(repeats):
        for way in ("whole oils held out", "random folds"):
            folds = folds_by_group(d["oil"], OLIVE_FOLDS, rng) if way.startswith("whole") \
                else [np.sort(f) for f in np.array_split(rng.permutation(n), OLIVE_FOLDS)]
            pred: Dict[str, np.ndarray] = {}
            said: Dict[str, np.ndarray] = {}
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                inner = (folds_by_group(d["oil"][tr], INNER, rng) if way.startswith("whole")
                         else [np.sort(g) for g in np.array_split(rng.permutation(len(tr)),
                                                                  INNER)])
                todo = [("pls", "one map", inputs["one map"]),
                        ("kernel", "one map", inputs["one map"]),
                        ("pls " + AS_MEASURED, "one map", plain)]
                if way.startswith("whole"):
                    todo += [("pls", "mean of three maps", inputs["mean of three maps"]),
                             ("kernel", "mean of three maps", inputs["mean of three maps"])]
                for key, what, pool in todo:
                    model = key.split()[0]
                    for t, y in targets.items():
                        name, k = chosen_regression(pool, y, tr, inner, model)
                        P = regression_path(pool[name], y, tr, f, model)
                        slot = pred.setdefault(f"{t}|{key}|{what}", np.zeros(n))
                        slot[f] = P[_setting(pool[name], k, tr, model)]
                        if way.startswith("whole") and what == "one map" and key != \
                                "pls " + AS_MEASURED:
                            c = chosen.setdefault(f"{t}|{key}", {})
                            c[name] = c.get(name, 0) + 1
                    name, k = chosen_classifier(pool, within, classes, tr, inner, model)
                    S = class_scores(pool[name], within, classes, tr, f, model)
                    slot = said.setdefault(f"{key}|{what}", np.empty(n, dtype=object))
                    slot[f] = np.array(classes)[S[_setting(pool[name], k, tr, model)]
                                                .argmax(axis=1)]
                for t, y in targets.items():
                    pred.setdefault(f"{t}|mean of the fit|", np.zeros(n))[f] = y[tr].mean()
                    pred.setdefault(f"{t}|sum of the intensities|", np.zeros(n))[f] = \
                        one_number(one_map, y, tr, f)
            for key, p in pred.items():
                t = key.split("|")[0]
                err.setdefault(f"{way}|{key}", []).append(rmse(p, targets[t]))
            for key, v in said.items():
                hit.setdefault(f"{way}|{key}", []).append(
                    balanced_accuracy(within, v.astype(str), classes))
        log(f"  olive: repeat {r + 1} of {repeats}, {time.time() - t0:.0f} s")
    way = "whole oils held out"
    differences = {}
    for t in targets:
        differences[t] = {
            "kernel_minus_pls": M.mean_difference(
                err[f"{way}|{t}|kernel|one map"], err[f"{way}|{t}|pls|one map"], seed=SEED),
            "kernel_minus_pls_as_measured": M.mean_difference(
                err[f"{way}|{t}|kernel|one map"],
                err[f"{way}|{t}|pls {AS_MEASURED}|one map"], seed=SEED),
            "mean_of_three_minus_one_map": M.mean_difference(
                err[f"{way}|{t}|kernel|mean of three maps"], err[f"{way}|{t}|kernel|one map"],
                seed=SEED)}
    differences["within"] = {
        "kernel_minus_pls": M.mean_difference(hit[f"{way}|kernel|one map"],
                                              hit[f"{way}|pls|one map"], seed=SEED)}
    return {"samples": n, "oils": int(len(set(d["oil"]))), "maps": int(len(d["maps"])),
            "points": [int(d["maps"].shape[1]), int(d["maps"].shape[2])],
            "repeats": repeats, "folds": OLIVE_FOLDS,
            "spread": {t: float(np.std(y)) for t, y in targets.items()},
            "within_the_limits": int(d["within"].sum()),
            "rmse": {k: _mean_sd(v) for k, v in err.items()},
            "balanced_accuracy": {k: _mean_sd(v) for k, v in hit.items()},
            "differences": differences,
            "chosen": {k: dict(sorted(v.items(), key=lambda kv: -kv[1])[:5])
                       for k, v in chosen.items()},
            "setting_at_an_edge": _edges(),
            "seconds": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------- #
# Part 3: simulated peak lists
# --------------------------------------------------------------------------- #
SIM_PER_CLASS = 50
SIM_PEAKS = 40
SIM_MARKERS = 4
SIM_EFFECT = 0.6
SIM_NOISE = 0.25
SIM_EXTRA = 6
SIM_SIZE = 100.0                           # both axes run from 0 to this
SIM_JITTER = (0.0, 0.5, 1.0, 2.0, 4.0)
SIM_WIDTHS = (0.5, 1.0, 2.0, 4.0, 8.0)
SIM_BINS = (1.0, 2.0, 4.0, 8.0, 16.0)
SIM_REPEATS = 10


def simulated_lists(jitter: float, rng, per_class: int = SIM_PER_CLASS,
                    peaks: int = SIM_PEAKS) -> Tuple[List[S2.PeakList], np.ndarray]:
    """Two classes of peak lists with the same peaks.

    The peaks have fixed places on a square and fixed mean intensities.  In the second class
    the first :data:`SIM_MARKERS` peaks are stronger by the factor ``1 + SIM_EFFECT``.  In a
    sample every peak is moved by a Gaussian of standard deviation `jitter` along each axis,
    its intensity is multiplied by a log-normal factor, and a few peaks at random places are
    added.
    """
    px, py = rng.uniform(5, SIM_SIZE - 5, peaks), rng.uniform(5, SIM_SIZE - 5, peaks)
    base = rng.lognormal(0.0, 0.5, peaks)
    lists, labels = [], []
    for c in (0, 1):
        mean = base.copy()
        if c:
            mean[:SIM_MARKERS] *= 1.0 + SIM_EFFECT
        for _ in range(per_class):
            x = np.concatenate([px + rng.normal(0, jitter, peaks) if jitter else px,
                                rng.uniform(0, SIM_SIZE, SIM_EXTRA)])
            y = np.concatenate([py + rng.normal(0, jitter, peaks) if jitter else py,
                                rng.uniform(0, SIM_SIZE, SIM_EXTRA)])
            w = np.concatenate([mean * rng.lognormal(0.0, SIM_NOISE, peaks),
                                rng.lognormal(-0.5, 0.5, SIM_EXTRA)])
            lists.append(S2.PeakList(x, y, w))
            labels.append(c)
    return lists, np.array(labels)


def binned_rows(lists: Sequence[S2.PeakList], size: float) -> np.ndarray:
    """The peak lists in square bins of the side `size`: the sum of the intensities of the
    peaks of a bin, one row for each list."""
    edges = np.arange(0.0, SIM_SIZE + size, size)
    return np.array([np.histogram2d(np.clip(p.x, 0, SIM_SIZE - 1e-9),
                                    np.clip(p.y, 0, SIM_SIZE - 1e-9), bins=(edges, edges),
                                    weights=p.w)[0].ravel() for p in lists])


def run_simulated(repeats: int = SIM_REPEATS, jitters: Sequence[float] = SIM_JITTER,
                  **sizes) -> dict:
    """The four combinations of a description (the kernel of the peak lists, or bins) and a
    model (the kernel Fisher discriminant, or PLS-DA), for every size of the jitter."""
    t0 = time.time()
    EDGES.clear()
    rng = np.random.default_rng(SEED)
    classes = [0, 1]
    out = {"per_class": sizes.get("per_class", SIM_PER_CLASS),
           "peaks": sizes.get("peaks", SIM_PEAKS), "markers": SIM_MARKERS,
           "effect": SIM_EFFECT, "repeats": repeats, "jitter": {}}
    for jitter in jitters:
        acc: Dict[str, List[float]] = {}
        picked: Dict[str, Dict[str, int]] = {}
        for _ in range(repeats):
            lists, labels = simulated_lists(jitter, rng, **sizes)
            n = len(lists)
            pools = {
                "peaks": {f"width {w:g}": S2.peak_kernel(lists, w, w, normalize=False)
                          for w in SIM_WIDTHS},
                "bins": {}}
            for b in SIM_BINS:
                F = binned_rows(lists, b)
                pools["bins"][f"bin {b:g}"] = F @ F.T
            for pool in pools.values():
                for name in pool:
                    pool[name] = pool[name] / (float(np.mean(np.diag(pool[name]))) or 1.0)
            folds = folds_by_class(labels, FOLDS, rng)
            said: Dict[str, np.ndarray] = {}
            for f in folds:
                tr = np.setdiff1d(np.arange(n), f)
                inner = folds_by_class(labels[tr], INNER, rng)
                for what, pool in pools.items():
                    for model in ("kernel", "pls"):
                        name, k = chosen_classifier(pool, labels, classes, tr, inner, model)
                        S = class_scores(pool[name], labels, classes, tr, f, model)
                        key = f"{what}|{model}"
                        said.setdefault(key, np.zeros(n, int))[f] = \
                            S[_setting(pool[name], k, tr, model)].argmax(axis=1)
                        c = picked.setdefault(key, {})
                        c[name] = c.get(name, 0) + 1
                for name, K in pools["bins"].items():       # PLS-DA at every bin size
                    k_name, k = chosen_classifier({name: K}, labels, classes, tr, inner, "pls")
                    S = class_scores(K, labels, classes, tr, f, "pls")
                    said.setdefault(f"{name}|pls", np.zeros(n, int))[f] = \
                        S[_setting(K, k, tr, "pls")].argmax(axis=1)
            for key, v in said.items():
                acc.setdefault(key, []).append(float(np.mean(v == labels)))
        out["jitter"][f"{jitter:g}"] = {
            "accuracy": {k: _mean_sd(v) for k, v in acc.items()},
            "peaks_minus_bins": M.mean_difference(acc["peaks|kernel"], acc["bins|pls"],
                                                  seed=SEED),
            "chosen": {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in picked.items()}}
        log(f"  simulated: jitter {jitter:g} done, {time.time() - t0:.0f} s")
    out["setting_at_an_edge"] = _edges()
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- #
# The kernel correlation
# --------------------------------------------------------------------------- #
def kernel_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """The correlation of two kernels of the same samples over the pairs of different
    samples.  A new description whose kernel correlates at 0.99 with the old one cannot
    change a model much (the whole-kernel test of plan item EC5)."""
    i, j = np.triu_indices(len(A), 1)
    return float(np.corrcoef(A[i, j], B[i, j])[0, 1])


def run_correlation(**grid) -> dict:
    """The correlation of every broadened description with the map as measured, on the two
    sets, and of the kernel of the peak lists with the kernel of the bins, on one draw of
    the simulated lists for every size of the jitter."""
    t0 = time.time()
    out: dict = {"maps": {}, "simulated": {}}
    h, o = honey(), olive()
    for which, maps, ex, em in (("honey", h["maps"][~h["false"]], h["excitation"],
                                 h["emission"]),
                                ("olive", o["maps"][o["measured"]], o["excitation"],
                                 o["emission"])):
        raw = raw_kernels(maps, float(ex[1] - ex[0]), float(em[1] - em[0]), **grid)
        plain = cosine(raw[(0.0, 0.0)])
        out["maps"][which] = {f"{sx:g}|{sy:g}": kernel_correlation(plain, cosine(K))
                              for (sx, sy), K in raw.items() if (sx, sy) != (0.0, 0.0)}
    rng = np.random.default_rng(SEED)
    for jitter in SIM_JITTER:
        lists, _ = simulated_lists(jitter, rng)
        peak = {w: S2.peak_kernel(lists, w, w) for w in (2.0, 8.0)}
        bins = {}
        for b in (1.0, 8.0):
            F = binned_rows(lists, b)
            bins[b] = cosine(F @ F.T)
        out["simulated"][f"{jitter:g}"] = {
            f"width {w:g}, bins of {b:g}": kernel_correlation(peak[w], bins[b])
            for w in peak for b in bins}
    out["seconds"] = round(time.time() - t0, 1)
    return out


# --------------------------------------------------------------------------- #
PARTS = {"honey": run_honey, "olive": run_olive, "simulated": run_simulated,
         "correlation": run_correlation}


def run(results: str = RESULTS, parts: Sequence[str] = tuple(PARTS), **options) -> dict:
    path = os.path.join(results, "fluorescence.json")
    out = {}
    if os.path.isfile(path):                # a run of one part keeps the others
        with open(path, encoding="utf-8") as fh:
            out = json.load(fh)
    out.update({"sources": PREP.SOURCES, "widths_excitation_nm": list(WIDTHS_EX),
                "widths_emission_nm": list(WIDTHS_EM)})
    for part in parts:
        out[part] = PARTS[part](**options.get(part, {}))
    os.makedirs(results, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def shown(name: str) -> str:
    """A description in words: ``10|4|1`` is 10 nm along the excitation axis, 4 nm along the
    emission axis, the cosine form."""
    sx, sy, cos = name.split("|")
    width = "no broadening" if float(sx) == 0 and float(sy) == 0 else \
        f"{sx} nm by {sy} nm"
    return width + (", each map divided by its length" if int(cos) else "")


def _pm(v: dict, scale: float = 1.0, digits: int = 3) -> str:
    return f"{v['mean'] * scale:.{digits}f} ({v['sd'] * scale:.{digits}f})"


def _interval(v: dict, scale: float = 1.0, digits: int = 3) -> str:
    return (f"{v['mean'] * scale:+.{digits}f} [{v['low'] * scale:+.{digits}f}, "
            f"{v['high'] * scale:+.{digits}f}]")


def _honey_lines(h: dict) -> List[str]:
    lines = ["## Part 1: honey, four botanical origins",
             "",
             f"{h['samples']} real honeys ("
             + ", ".join(f"{n} {c}" for c, n in zip(h["classes"], h["counts"]))
             + f"), maps of {h['points'][0]} by {h['points'][1]} points after the scattered "
             f"light is taken out; five folds, {h['repeats']} repeats. A number in brackets "
             "is the standard deviation over the repeats.",
             "",
             "| Model | Right class, % | Mean over the classes of the share found, % |",
             "|---|---|---|",
             f"| always the largest class | {100 * h['largest_class']:.1f} | "
             f"{100 / len(h['classes']):.1f} |"]
    for key, name in (("pls " + AS_MEASURED, "PLS-DA on the map as measured"),
                      ("pls", "PLS-DA, description chosen inside the fit"),
                      ("kernel", "kernel Fisher discriminant, description chosen inside "
                                 "the fit")):
        lines.append(f"| {name} | {_pm(h['accuracy'][key], 100, 1)} | "
                     f"{_pm(h['balanced_accuracy'][key], 100, 1)} |")
    lines += ["",
              "Kernel Fisher discriminant minus PLS-DA with the same choices, mean over the "
              f"classes: {_interval(h['kernel_minus_pls'], 100, 1)} points; minus PLS-DA on "
              f"the map as measured: {_interval(h['kernel_minus_pls_as_measured'], 100, 1)} "
              "points (95 % bootstrap interval over the repeats).",
              "",
              "Chosen most often: "
              + "; ".join(f"{'PLS-DA' if k == 'pls' else 'kernel Fisher discriminant'}: "
                          + shown(next(iter(v))) + f" ({next(iter(v.values()))} of "
                          f"{FOLDS * h['repeats']} fits)" for k, v in h["chosen"].items())
              + ".",
              "",
              "Where the honeys of a class are put by the kernel Fisher discriminant "
              "(rows: the class; columns: what the model said; all repeats together):",
              "",
              "| | " + " | ".join(h["classes"]) + " |",
              "|---|" + "---|" * len(h["classes"])]
    for c, row in zip(h["classes"], h["confusion"]["kernel"]):
        lines.append(f"| {c} | " + " | ".join(str(v) for v in row) + " |")
    lines += ["",
              "The places of the map that speak most for (+) or against (-) a class "
              f"(description: {shown(h['co_map']['description'])}):",
              "",
              "| Class | Excitation / emission, nm, and sign |",
              "|---|---|"]
    for c, places in h["co_map"]["classes"].items():
        lines.append(f"| {c} | " + "; ".join(
            f"{p['excitation']:.0f} / {p['emission']:.0f} ({p['sign']})" for p in places)
            + " |")
    names = {"ocmean": "distance to the mean map", "ocknn": "distance to the fifth nearest "
             "honey", "ockpca": "distance from the main subspace (5 components)",
             "ocgp": "Gaussian process on a constant"}
    lines += ["",
              "## Part 1: the authentication case",
              "",
              f"A description of the real honeys of the fit alone; the {h['false']} false "
              "honeys against the real honeys that the description has not seen. The number "
              "is the share of pairs of a real and a false honey that are put in the right "
              "order (0.5 is chance).",
              "",
              "| Description of the class | Maps as measured, each divided by its length | "
              "Maps broadened by the largest widths |",
              "|---|---|---|"]
    for kind in ONE_CLASS:
        a = h["false_against_real_auc"]
        lines.append(f"| {names[kind]} | {_pm(a[kind + '|plain'])} | "
                     f"{_pm(a[kind + '|broadened'])} |")
    return lines + [""]


def _olive_lines(o: dict) -> List[str]:
    lines = ["## Part 2: olive oil, the absorbance from the map",
             "",
             f"{o['samples']} oil samples ({o['oils']} oils at 10 steps of ageing) of "
             f"{o['maps']} maps with {o['points'][0]} by {o['points'][1]} points; "
             f"{o['folds']} folds, {o['repeats']} repeats. Root-mean-square errors in "
             "absorbance; a number in brackets is the standard deviation over the repeats. "
             f"The standard deviation of the absorbance at 232 nm is {o['spread']['k232']:.3f} "
             f"and at 268 nm {o['spread']['k268']:.3f}.",
             "",
             "| Model and input | 232 nm, whole oils held out | 232 nm, random folds | "
             "268 nm, whole oils held out | 268 nm, random folds |",
             "|---|---|---|---|---|"]
    rows_ = (("mean of the fit|", "the mean of the fit"),
             ("sum of the intensities|", "linear regression on the sum of the intensities"),
             (f"pls {AS_MEASURED}|one map", "PLS on the map as measured"),
             ("pls|one map", "PLS, description chosen inside the fit"),
             ("kernel|one map", "kernel ridge, description chosen inside the fit"),
             ("pls|mean of three maps", "PLS, the mean of the three maps of a sample"),
             ("kernel|mean of three maps", "kernel ridge, the mean of the three maps"))
    for key, name in rows_:
        cells = []
        for t in ("k232", "k268"):
            for way in ("whole oils held out", "random folds"):
                v = o["rmse"].get(f"{way}|{t}|{key}")
                cells.append(_pm(v, 1.0, 4) if v else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    d = o["differences"]
    lines += ["",
              "Differences with whole oils held out (95 % bootstrap interval over the "
              "repeats):",
              ""]
    for t, wl in (("k232", "232 nm"), ("k268", "268 nm")):
        lines.append(f"- {wl}: kernel ridge minus PLS with the same choices "
                     f"{_interval(d[t]['kernel_minus_pls'], 1.0, 4)}; minus PLS on the map as "
                     f"measured {_interval(d[t]['kernel_minus_pls_as_measured'], 1.0, 4)}; "
                     "the mean of three maps minus one map (kernel ridge) "
                     f"{_interval(d[t]['mean_of_three_minus_one_map'], 1.0, 4)}.")
    lines.append("- Chosen most often with whole oils held out: " + "; ".join(
        f"{k.split('|')[0]}, {'PLS' if k.endswith('pls') else 'kernel ridge'}: "
        + shown(next(iter(v))) for k, v in o["chosen"].items()) + ".")
    lines += ["",
              "## Part 2: within the limits of the regulation or not",
              "",
              f"{o['within_the_limits']} of the {o['samples']} samples are within the three "
              "limits for extra virgin oil. Mean over the two classes of the share found, %.",
              "",
              "| Model and input | Whole oils held out | Random folds |",
              "|---|---|---|"]
    for key, name in ((f"pls {AS_MEASURED}|one map", "PLS-DA on the map as measured"),
                      ("pls|one map", "PLS-DA, description chosen inside the fit"),
                      ("kernel|one map", "kernel Fisher discriminant, description chosen "
                                         "inside the fit"),
                      ("kernel|mean of three maps", "kernel Fisher discriminant, the mean of "
                                                    "the three maps")):
        cells = []
        for way in ("whole oils held out", "random folds"):
            v = o["balanced_accuracy"].get(f"{way}|{key}")
            cells.append(_pm(v, 100, 1) if v else "")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["",
              "Kernel Fisher discriminant minus PLS-DA, whole oils held out: "
              f"{_interval(d['within']['kernel_minus_pls'], 100, 1)} points.",
              ""]
    return lines


def _simulated_lines(s: dict) -> List[str]:
    bins = [k for k in next(iter(s["jitter"].values()))["accuracy"] if k.startswith("bin ")]
    lines = ["## Part 3: simulated peak lists",
             "",
             f"Two classes of {s['per_class']} lists with {s['peaks']} peaks on a square of "
             f"side {SIM_SIZE:g}; in one class {s['markers']} peaks are stronger by "
             f"{100 * s['effect']:.0f} %. The jitter is the standard deviation of the place "
             f"of a peak along each axis. Right class, %; {s['repeats']} repeats.",
             "",
             "| Jitter | Kernel of the peak lists, kernel Fisher discriminant | Kernel of "
             "the peak lists, PLS-DA | Bins, kernel Fisher discriminant | Bins, PLS-DA | "
             + " | ".join(f"PLS-DA, bins of {b.split()[1].split('|')[0]}" for b in bins)
             + " | Width chosen | Bin chosen |",
             "|---|---|---|---|---|" + "---|" * len(bins) + "---|---|"]
    for jitter, v in s["jitter"].items():
        a = v["accuracy"]
        cells = [_pm(a[k], 100, 1) for k in ("peaks|kernel", "peaks|pls", "bins|kernel",
                                             "bins|pls")]
        cells += [_pm(a[b], 100, 1) for b in bins]
        lines.append(f"| {jitter} | " + " | ".join(cells)
                     + f" | {v['chosen']['peaks|kernel'].split()[1]} | "
                     f"{v['chosen']['bins|pls'].split()[1]} |")
    lines += ["",
              "Kernel of the peak lists with the kernel Fisher discriminant minus PLS-DA on "
              "bins, points: " + "; ".join(
                  f"jitter {j}: {_interval(v['peaks_minus_bins'], 100, 1)}"
                  for j, v in s["jitter"].items()) + ".",
              ""]
    return lines


def _correlation_lines(c: dict) -> List[str]:
    lines = ["## The kernel correlation",
             "",
             "The correlation of the kernel of a new description with the kernel of the old "
             "one, over the pairs of different samples. Maps: every broadened description "
             "against the map as measured, both in the cosine form.",
             "",
             "| Set | Lowest correlation over the widths | At the widths | Highest |",
             "|---|---|---|---|"]
    for which, v in c["maps"].items():
        low = min(v, key=v.get)
        sx, sy = low.split("|")
        lines.append(f"| {which} | {v[low]:.4f} | {sx} nm by {sy} nm | {max(v.values()):.4f} |")
    names = list(next(iter(c["simulated"].values())))
    lines += ["",
              "Simulated peak lists: the kernel of the peak lists against the kernel of the "
              "bins, one draw for every size of the jitter.",
              "",
              "| Jitter | " + " | ".join(names) + " |",
              "|---|" + "---|" * len(names)]
    for jitter, v in c["simulated"].items():
        lines.append(f"| {jitter} | " + " | ".join(f"{v[n]:.3f}" for n in names) + " |")
    return lines + [""]


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "fluorescence.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    lines = ["# Spectra with two axes: fluorescence maps and peak lists",
             "",
             "Data: fluorescence maps of honeys (" + d["sources"]["honey"] + ") and of olive "
             "oils during ageing (" + d["sources"]["olive"] + ", CC BY 4.0); no file of the "
             "two sets is in this repository. Widths that can be chosen: "
             + ", ".join(f"{w:g}" for w in d["widths_excitation_nm"])
             + " nm along the excitation axis and "
             + ", ".join(f"{w:g}" for w in d["widths_emission_nm"])
             + " nm along the emission axis.",
             ""]
    if "honey" in d:
        lines += _honey_lines(d["honey"])
    if "olive" in d:
        lines += _olive_lines(d["olive"])
    if "simulated" in d:
        lines += _simulated_lines(d["simulated"])
    if "correlation" in d:
        lines += _correlation_lines(d["correlation"])
    text = "\n".join(lines)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--parts", nargs="+", default=list(PARTS), choices=list(PARTS))
    ap.add_argument("--repeats", type=int, default=0, help="for a trial: repeats of each part")
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)
    if not a.report:
        options = {p: {"repeats": a.repeats} for p in a.parts} if a.repeats else {}
        run(a.results, a.parts, **options)
    print(report(a.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
