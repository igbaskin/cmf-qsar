#!/usr/bin/env python
"""
A model fitted on one instrument reads the spectra of another (plan item SP5).

The data are two near-infrared sets in which the same samples were measured on several
instruments (`prepare_calibration_transfer.py`): 80 samples of corn on three spectrometers
with four measured properties, and 655 tablets on two spectrometers with the content of the
active ingredient.  A model is fitted on the spectra of one instrument, the *source*.  It is
then tested twice on samples that it has not seen: on their spectra from the source (*within*)
and on their spectra from another instrument, the *target* (*crossing*).

Part 1: what differs between the instruments
--------------------------------------------
For every pair of instruments the difference between the two spectra of a sample is taken
apart: the part that is the same for all samples (the mean difference), an offset, a gain and
a sloping baseline per sample, a shift of the wavelength axis and a difference in resolution.
The plan expected a difference in resolution; this part measures what is there.

Part 2: the crossing
--------------------
Two models with the same rows: partial least squares regression (PLS), which is the reference,
and kernel ridge regression on the linear spectrum kernel.  The rows are the spectra as
measured ("plain") or their slopes along the axis ("slope"), both after a broadening whose
width is chosen by cross-validation inside the samples of the fit, on the source alone.

What is done to the spectra of the target before the model reads them:

- nothing ("as measured");
- "each instrument centred": the mean spectrum of the target spectra is taken off and the mean
  spectrum of the source spectra of the fit is put in its place.  This uses the target
  spectra of the test samples, and not their measured properties;
- "shift fitted": the shift of the axis between the two mean spectra is fitted
  (`cmfqsar.spectrum.fit_shift`) and removed;
- both of them;
- with transfer standards (the reference): a few samples of the fit are measured on both
  instruments.  "Mean difference of the standards" adds the mean difference of their spectra.
  "Piecewise direct standardisation" (PDS) fits, for every point of the source axis, a
  regression that makes the source value of the standards from a window of their target
  spectra.  The standards are chosen by the Kennard-Stone rule, and the window and the
  number of principal components of PDS by leave-one-out over the standards.

The floor of a crossing is the model that was fitted on the target instrument itself.

Part 3: a difference in resolution, simulated
---------------------------------------------
The instruments of both sets have the same resolution (part 1), so the claim of the plan
item is tested on a simulated coarser instrument: the target spectra are broadened by a
Gaussian of known width.  The model then reads them as they are, with the widths matched
(`cmfqsar.spectrum.matched_widths`: the source spectra get the larger width, so both
instruments end at one width) from the known difference, and with the difference fitted from
the two mean spectra without a standard (`cmfqsar.spectrum.fit_width`).  This is done with
the source instrument itself as the target, where the resolution is the only difference, and
with another instrument, where the shift is removed and each instrument is centred first.

Usage
-----
    python benchmarks/prepare_calibration_transfer.py   # once: downloads 6.5 MB
    python benchmarks/run_calibration_transfer.py       # the benchmark, then the table
    python benchmarks/run_calibration_transfer.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import spectrum as SP          # noqa: E402
import prepare_calibration_transfer as PREP  # noqa: E402
import run_spectra_measured as M             # noqa: E402

RESULTS = os.path.join(HERE, "results", "calibration_transfer")
FORMS = ("plain", "slope")
MODELS = ("pls", "kernel")
WIDTHS = (0.0, 4.0, 8.0, 16.0)             # nm; 0 leaves the spectra as measured
RIDGES = tuple(10.0 ** e for e in np.arange(-8.0, 0.01, 0.5))
PLS_MAX = 30                               # instrument m5 of the corn set needs more than 20
N_STANDARDS = 8
MORE_STANDARDS = 16                        # PDS is also run with this number
PDS_MORE = "piecewise direct standardisation, more standards"
PDS_HALF = (1, 3, 5, 7)                    # the window is 2 * half + 1 points
PDS_RANK = (1, 2, 3)
MAX_SHIFT = 4.0                            # nm
COARSER = (8.0, 16.0, 32.0)                # nm: the simulated loss of resolution
TRIED_WIDTHS = (2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0)
REPEATS = 20
FOLDS = 5
SEED = 20261112
#: per set: the properties that are modelled, the number of samples of the fit and the part
#: of the axis that is used.  The tablet spectra are noise above 1600 nm on one instrument.
SETS = {
    "corn": {"properties": ("moisture", "oil", "protein", "starch"), "fit": 60,
             "range": (1100.0, 2498.0)},
    "tablets": {"properties": ("assay",), "fit": 155, "range": (600.0, 1600.0)},
}
COARSE_VARIANTS = ("read as they are", "widths matched, difference known",
                   "widths matched, difference fitted")
VARIANTS = ("as measured", "each instrument centred", "shift fitted",
            "shift fitted, each instrument centred", "mean difference of the standards",
            "piecewise direct standardisation")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Rows and models
# --------------------------------------------------------------------------- #
def described(X: np.ndarray, step: float, width: float, form: str) -> np.ndarray:
    """The rows of a description: the spectra broadened by `width`, as they are or as their
    slopes along the axis.  Both steps are linear in the spectrum."""
    F = SP.smoothed(X, step, width)
    return np.gradient(F, step, axis=1) if form == "slope" else F


def pls_weights(F: np.ndarray, y: np.ndarray, most: int) -> Tuple[np.ndarray, np.ndarray]:
    """PLS with 1, 2, ..., `most` components from one fit: one weight over the columns per
    row of the first result and the constants in the second.  The prediction of the model
    with ``k`` components for a row ``f`` is ``f @ W[k - 1] + b[k - 1]``.

    This is the NIPALS algorithm for one property, written out because the benchmark fits
    many thousands of models; `tests/test_calibration_transfer.py` compares it with
    scikit-learn.  The components are found one after the other, so the model with ``k``
    components is the first ``k`` of the model with more.
    """
    xbar, ybar = F.mean(axis=0), float(np.mean(y))
    X = F - xbar
    scale = float(np.sqrt(np.mean(X ** 2))) or 1.0      # one factor for all columns
    X = X / scale
    r = np.asarray(y, float) - ybar
    loadings: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    coef = np.zeros(F.shape[1])
    W = np.zeros((most, F.shape[1]))
    for k in range(most):
        w = X.T @ r
        norm = float(np.sqrt(w @ w))
        if norm < 1e-14:                                # nothing of the property is left
            W[k:] = coef
            break
        w = w / norm
        s = X @ w
        ss = float(s @ s)
        p = X.T @ s / ss
        q = float(r @ s) / ss
        X = X - np.outer(s, p)
        r = r - q * s
        rot = w.copy()                                  # the score as a weight on the rows
        for pj, rj in zip(loadings, rotations):         # before the deflation
            rot -= float(pj @ w) * rj
        loadings.append(p)
        rotations.append(rot)
        coef = coef + q * rot
        W[k] = coef
    W = W / scale
    return W, ybar - W @ xbar


def ridge_weights(F: np.ndarray, y: np.ndarray,
                  ridges: Sequence[float] = RIDGES) -> Tuple[np.ndarray, np.ndarray]:
    """Kernel ridge regression on the linear kernel of the rows, with a constant, for every
    ridge from one eigendecomposition: one weight over the columns per ridge and the
    constants, as :func:`pls_weights` returns them.  A ridge is a multiple of the mean
    diagonal of the centred kernel."""
    xbar, ybar = F.mean(axis=0), float(np.mean(y))
    Fc = F - xbar
    K = Fc @ Fc.T
    e, V = np.linalg.eigh(K)
    e = np.maximum(e, 0.0)
    proj = V.T @ (np.asarray(y, float) - ybar)
    unit = max(float(np.trace(K)) / len(K), 1e-300)
    lam = np.asarray(ridges, float) * unit
    alphas = (proj[None, :] / (e[None, :] + lam[:, None])) @ V.T
    W = alphas @ Fc
    return W, ybar - W @ xbar


def weights_of(model: str, F: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if model == "pls":
        return pls_weights(F, y, min(PLS_MAX, len(F) - 2))
    return ridge_weights(F, y)


def rmse(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def choose(rows: Dict[float, np.ndarray], y: np.ndarray, model: str,
           folds: Sequence[np.ndarray]) -> Tuple[float, int]:
    """The width and the setting of the model (the number of components or the ridge) with
    the smallest error in the cross-validation inside the samples of the fit."""
    best = None
    n = len(y)
    for width, F in rows.items():
        err = 0.0
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            W, b = weights_of(model, F[tr], y[tr])
            err = err + np.sum((F[f] @ W.T + b - y[f][:, None]) ** 2, axis=0)
        k = int(np.argmin(err))
        if best is None or err[k] < best[0]:
            best = (float(err[k]), width, k)
    return best[1], best[2]


def fitted(X: np.ndarray, y: np.ndarray, step: float, form: str, model: str,
           folds: Sequence[np.ndarray], widths: Sequence[float] = WIDTHS) -> dict:
    """A model of one form fitted on the spectra `X`: the width that was chosen, and the
    weight over the columns of the description with the constant."""
    rows = {w: described(X, step, w, form) for w in widths}
    width, k = choose(rows, y, model, folds)
    W, b = weights_of(model, rows[width], y)
    k = min(k, len(W) - 1)                  # the whole fit allows more components than a fold
    return {"width": width, "setting": k, "w": W[k], "b": float(b[k]), "form": form,
            "step": step}


def predict(fit: dict, X: np.ndarray, width: float = None) -> np.ndarray:
    """The prediction of a fitted model for spectra; `width` replaces the width of the fit
    for these spectra (:func:`cmfqsar.spectrum.matched_widths`)."""
    w = fit["width"] if width is None else width
    return described(X, fit["step"], w, fit["form"]) @ fit["w"] + fit["b"]


# --------------------------------------------------------------------------- #
# Transfer standards
# --------------------------------------------------------------------------- #
def kennard_stone(X: np.ndarray, n: int) -> np.ndarray:
    """`n` rows that cover the set: the two rows furthest apart, and then again and again
    the row furthest from the rows already chosen."""
    G = X @ X.T
    d = np.diag(G)
    D = np.maximum(d[:, None] + d[None, :] - 2.0 * G, 0.0)
    i, j = np.unravel_index(int(np.argmax(D)), D.shape)
    chosen = [int(i), int(j)]
    while len(chosen) < min(n, len(X)):
        nearest = D[:, chosen].min(axis=1)
        nearest[chosen] = -1.0
        chosen.append(int(np.argmax(nearest)))
    return np.array(chosen[:n])


def pds_fit(source: np.ndarray, target: np.ndarray, half: int,
            rank: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Piecewise direct standardisation from the spectra of the standards on both instruments.

    For every point ``j`` of the axis the source value of the standards is regressed on the
    ``2 * half + 1`` target values around ``j``, by principal component regression with
    `rank` components.  A window at an end of the axis is moved inside.  Returns the first
    column of every window, the coefficients (one row per point) and the constants.
    """
    n, m = target.shape
    size = 2 * half + 1
    first = np.clip(np.arange(m) - half, 0, m - size)
    cols = first[:, None] + np.arange(size)[None, :]
    W = target[:, cols].transpose(1, 0, 2)              # point, standard, window
    xbar = W.mean(axis=1)
    ybar = source.mean(axis=0)
    U, S, Vt = np.linalg.svd(W - xbar[:, None, :], full_matrices=False)
    r = min(rank, S.shape[1])
    inv = np.where(S[:, :r] > 1e-12 * S[:, :1], 1.0 / np.maximum(S[:, :r], 1e-300), 0.0)
    proj = np.einsum("jnk,nj->jk", U[:, :, :r], source - ybar)
    beta = np.einsum("jkp,jk->jp", Vt[:, :r, :], proj * inv)
    const = ybar - np.einsum("jp,jp->j", xbar, beta)
    return first, beta, const


def pds_apply(pds: Tuple[np.ndarray, np.ndarray, np.ndarray], X: np.ndarray) -> np.ndarray:
    """Target spectra as the source instrument would have measured them."""
    first, beta, const = pds
    cols = first[:, None] + np.arange(beta.shape[1])[None, :]
    return np.einsum("njp,jp->nj", X[:, cols], beta) + const


def pds_chosen(fit: dict, y: np.ndarray,
               loo: Dict[Tuple[int, int], np.ndarray]) -> Tuple[int, int]:
    """The window and the number of components of PDS with the smallest error of the fitted
    model on the standards, each standard transformed without itself (`loo`)."""
    best = None
    for key, T in loo.items():
        err = rmse(predict(fit, T), y)
        if best is None or err < best[0]:
            best = (err, key)
    return best[1]


def pds_leave_one_out(source: np.ndarray,
                      target: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
    """For every setting of PDS, the target spectra of the standards transformed by a PDS
    that was fitted on the other standards."""
    n = len(source)
    out = {}
    for half in PDS_HALF:
        for rank in PDS_RANK:
            if rank > min(n - 2, 2 * half + 1):
                continue
            T = np.empty_like(target)
            for i in range(n):
                rest = np.arange(n) != i
                T[i] = pds_apply(pds_fit(source[rest], target[rest], half, rank),
                                 target[i:i + 1])[0]
            out[(half, rank)] = T
    return out


# --------------------------------------------------------------------------- #
# Part 1: what differs between two instruments
# --------------------------------------------------------------------------- #
def _rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(v) ** 2)))


def per_sample_residual(A: np.ndarray, B: np.ndarray, sloping: bool = True) -> np.ndarray:
    """``B`` minus the best ``offset + gain * A (+ a sloping baseline)``, sample by sample."""
    m = A.shape[1]
    ramp = np.linspace(-1.0, 1.0, m)
    R = np.empty_like(B)
    for i in range(len(A)):
        X = np.array([np.ones(m), A[i]] + ([ramp] if sloping else [])).T
        c, *_ = np.linalg.lstsq(X, B[i], rcond=None)
        R[i] = B[i] - X @ c
    return R


def difference(A: np.ndarray, B: np.ndarray, x: np.ndarray) -> dict:
    """The difference between the spectra of the same samples on two instruments, taken
    apart step by step: first the mean difference, which is the same for all samples, then
    an offset, a gain and a sloping baseline for every sample, then a shift of the axis.
    Every number is a root mean square over the samples and the axis; the part of the axis
    within :data:`MAX_SHIFT` of an end is left out, because a shifted curve has no values
    there.  The shift and the difference in resolution are fitted on the two mean spectra."""
    step = float(x[1] - x[0])
    edge = int(np.ceil(MAX_SHIFT / step)) + 1
    inner = slice(edge, len(x) - edge)
    D = B - A
    shift = SP.fit_shift(A.mean(axis=0), B.mean(axis=0), x, MAX_SHIFT)
    width, _ = SP.fit_width(SP.shifted(A.mean(axis=0), x, shift), B.mean(axis=0), x,
                            TRIED_WIDTHS)
    moved = SP.shifted(A, x, shift)
    Dm = B - moved
    return {
        "spread_of_the_samples": _rms((A - A.mean(axis=0))[:, inner]),
        "difference": _rms(D[:, inner]),
        "after_the_mean_difference": _rms((D - D.mean(axis=0))[:, inner]),
        "after_offset_gain_and_slope_per_sample": _rms(
            per_sample_residual(A, B - D.mean(axis=0))[:, inner]),
        "after_the_shift_too": _rms(per_sample_residual(moved, B - Dm.mean(axis=0))[:, inner]),
        "shift_nm": shift, "width_difference_nm": width,
    }


# --------------------------------------------------------------------------- #
# Part 2 and part 3: one split
# --------------------------------------------------------------------------- #
def target_rows(variant: str, Xs_fit: np.ndarray, Xt: np.ndarray, x: np.ndarray,
                shift: float) -> np.ndarray:
    """The target spectra `Xt` as the model reads them under one of the variants that need no
    standard (:data:`VARIANTS`); `shift` is the fitted shift of the target against the
    source."""
    out = SP.shifted(Xt, x, -shift) if "shift" in variant else Xt
    if "centred" in variant:
        out = out - out.mean(axis=0) + Xs_fit.mean(axis=0)
    return out


def one_split(S: np.ndarray, Y: np.ndarray, x: np.ndarray, fit_idx: np.ndarray,
              test_idx: np.ndarray, rng, n_standards: int = N_STANDARDS,
              more_standards: int = MORE_STANDARDS,
              widths: Sequence[float] = WIDTHS, coarser: Sequence[float] = COARSER) -> dict:
    """All models of one split of the samples.

    `S` holds the spectra (instrument, sample, point) and `Y` one column per property.
    Returns the errors as nested dictionaries: ``within[source][property][model][form]``
    (as measured, and with the test spectra centred, which is the cost of that step when no
    instrument is changed), ``crossing[source->target][property][model][form][variant]`` and
    ``coarser[source->target][property][model][form][width][variant]``, where the target may
    be the source itself.
    """
    step = float(x[1] - x[0])
    folds = [np.sort(f) for f in np.array_split(rng.permutation(len(fit_idx)), FOLDS)]
    k = len(S)
    fits: dict = {}
    out: dict = {"within": {}, "crossing": {}, "coarser": {}, "chosen": {}, "shift": {},
                 "fitted_width": {}}
    for a in range(k):
        out["within"][a] = {}
        Xs_fit, Xs_test = S[a][fit_idx], S[a][test_idx]
        for p in range(Y.shape[1]):
            for model in MODELS:
                for form in FORMS:
                    f = fitted(Xs_fit, Y[fit_idx, p], step, form, model, folds, widths)
                    fits[(a, p, model, form)] = f
                    out["within"][a].setdefault(p, {}).setdefault(model, {})[form] = {
                        VARIANTS[0]: rmse(predict(f, Xs_test), Y[test_idx, p]),
                        VARIANTS[1]: rmse(predict(f, target_rows(VARIANTS[1], Xs_fit, Xs_test,
                                                                 x, 0.0)), Y[test_idx, p])}
                    out["chosen"][f"{a}|{p}|{model}|{form}"] = f["width"]
    cases = [(p, model, form) for p in range(Y.shape[1]) for model in MODELS for form in FORMS]
    for a in range(k):
        wide: dict = {}                     # the models on the broadened source spectra
        for b in range(k):
            key = f"{a}->{b}"
            Xs_fit, Xt = S[a][fit_idx], S[b][test_idx]
            shift = 0.0
            if a != b:
                shift = SP.fit_shift(Xs_fit.mean(axis=0), Xt.mean(axis=0), x, MAX_SHIFT)
                out["shift"][key] = shift
                more = fit_idx[kennard_stone(Xs_fit, more_standards)]
                std = more[:n_standards]                # the rule adds one row at a time
                loo = pds_leave_one_out(S[a][std], S[b][std])
                loo_more = pds_leave_one_out(S[a][more], S[b][more])
                mean_diff = (S[a][std] - S[b][std]).mean(axis=0)
                for p, model, form in cases:
                    f, y = fits[(a, p, model, form)], Y[test_idx, p]
                    slot = out["crossing"].setdefault(key, {}).setdefault(p, {}).setdefault(
                        model, {}).setdefault(form, {})
                    for variant in VARIANTS[:4]:
                        slot[variant] = rmse(
                            predict(f, target_rows(variant, Xs_fit, Xt, x, shift)), y)
                    slot[VARIANTS[4]] = rmse(predict(f, Xt + mean_diff), y)
                    half, rank = pds_chosen(f, Y[std, p], loo)
                    pds = pds_fit(S[a][std], S[b][std], half, rank)
                    slot[VARIANTS[5]] = rmse(predict(f, pds_apply(pds, Xt)), y)
                    half, rank = pds_chosen(f, Y[more, p], loo_more)
                    pds = pds_fit(S[a][more], S[b][more], half, rank)
                    slot[PDS_MORE] = rmse(predict(f, pds_apply(pds, Xt)), y)
                    slot["fitted on the target"] = out["within"][b][p][model][form][VARIANTS[0]]
                    slot["within the source"] = out["within"][a][p][model][form][VARIANTS[0]]
            # part 3: the target loses resolution.  On the source instrument itself nothing
            # else differs, so the spectra are read as measured; on another instrument the
            # shift is removed and each instrument is centred first.
            variant = VARIANTS[3] if a != b else VARIANTS[0]
            for extra in coarser:
                Xc = SP.smoothed(Xt, step, extra)
                signed, _ = SP.fit_width(Xs_fit.mean(axis=0),
                                         SP.shifted(Xc.mean(axis=0), x, -shift), x,
                                         TRIED_WIDTHS)
                out["fitted_width"].setdefault(key, {})[extra] = signed
                for name, diff in ((COARSE_VARIANTS[1], extra),
                                   (COARSE_VARIANTS[2], max(signed, 0.0))):
                    Xw = SP.smoothed(Xs_fit, step, diff)
                    for p, model, form in cases:
                        f, y = fits[(a, p, model, form)], Y[test_idx, p]
                        slot = out["coarser"].setdefault(key, {}).setdefault(p, {}).setdefault(
                            model, {}).setdefault(form, {}).setdefault(extra, {})
                        slot[COARSE_VARIANTS[0]] = rmse(
                            predict(f, target_rows(variant, Xs_fit, Xc, x, shift)), y)
                        if a == b:
                            slot[VARIANTS[1]] = rmse(
                                predict(f, target_rows(VARIANTS[1], Xs_fit, Xc, x, 0.0)), y)
                        if (diff, p, model, form) not in wide:
                            wide[(diff, p, model, form)] = fitted(
                                Xw, Y[fit_idx, p], step, form, model, folds, widths)
                        g = wide[(diff, p, model, form)]
                        slot[name] = rmse(predict(g, target_rows(variant, Xw, Xc, x, shift)), y)
    return out


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def cut(d: Dict[str, np.ndarray], lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    x = d["wavelength"]
    keep = (x >= lo) & (x <= hi)
    return x[keep], d["spectra"][:, :, keep]


def _collect(into: dict, one: dict) -> None:
    """Appends the numbers of `one` to the lists of `into`, which has the same nesting."""
    for key, value in one.items():
        if isinstance(value, dict):
            _collect(into.setdefault(str(key), {}), value)
        else:
            into.setdefault(str(key), []).append(float(value))


def _summarise(node):
    if isinstance(node, dict):
        return {k: _summarise(v) for k, v in node.items()}
    v = np.array(node, float)
    return {"mean": float(v.mean()), "sd": float(v.std())}


OURS = ("kernel", "slope", VARIANTS[3])
REFERENCE = ("pls", "plain", VARIANTS[0])


def gate(crossing: dict, properties: Sequence[str]) -> dict:
    """The comparison that the plan item names, for every crossing and property.

    The kernel on the slopes with the shift fitted and each instrument centred (no standard)
    is compared with PLS on the plain spectra as measured, with PLS with PDS (the better of
    its two forms, which favours the reference) and with PLS on the rows of the kernel.  A
    difference is the first minus the second, with a 95 % bootstrap interval over the
    splits.  The splits share the samples, so the interval says how much the result depends
    on the split, and not how it would change with other samples.
    """
    out = {}
    for key, by_property in crossing.items():
        for p, name in enumerate(properties):
            node = by_property[str(p)]
            ours = np.array(node[OURS[0]][OURS[1]][OURS[2]])
            plain = np.array(node[REFERENCE[0]][REFERENCE[1]][REFERENCE[2]])
            form = min((float(np.mean(node["pls"][f][VARIANTS[5]])), f) for f in FORMS)[1]
            pds = np.array(node["pls"][form][VARIANTS[5]])
            form_more = min((float(np.mean(node["pls"][f][PDS_MORE])), f) for f in FORMS)[1]
            more = np.array(node["pls"][form_more][PDS_MORE])
            same = np.array(node["pls"][OURS[1]][OURS[2]])
            out[f"{key}|{name}"] = {
                "ours": float(ours.mean()), "pls_as_measured": float(plain.mean()),
                "pls_with_pds": float(pds.mean()), "form_of_pds": form,
                "pls_with_pds_more_standards": float(more.mean()),
                "ours_minus_pls_with_pds_more_standards": M.mean_difference(ours, more,
                                                                            seed=SEED),
                "pls_on_the_same_rows": float(same.mean()),
                "ours_minus_pls_as_measured": M.mean_difference(ours, plain, seed=SEED),
                "ours_minus_pls_with_pds": M.mean_difference(ours, pds, seed=SEED),
                "ours_minus_pls_on_the_same_rows": M.mean_difference(ours, same, seed=SEED)}
    return out


def within_comparison(within: dict, properties: Sequence[str]) -> dict:
    """Kernel ridge against PLS within one instrument, for every instrument and property: the
    errors of the four combinations of model and rows, and the differences between the two
    models on the same rows and between kernel ridge on the slopes and PLS on the plain
    spectra, each with a 95 % bootstrap interval over the splits (see :func:`gate` for what
    that interval covers)."""
    out = {}
    for a, by_property in within.items():
        for p, name in enumerate(properties):
            node = by_property[str(p)]
            err = {(m, f): np.array(node[m][f][VARIANTS[0]]) for m in MODELS for f in FORMS}
            row = {f"{m}_{f}": float(v.mean()) for (m, f), v in err.items()}
            row["kernel_minus_pls_on_the_slopes"] = M.mean_difference(
                err[("kernel", "slope")], err[("pls", "slope")], seed=SEED)
            row["kernel_minus_pls_on_the_plain_spectra"] = M.mean_difference(
                err[("kernel", "plain")], err[("pls", "plain")], seed=SEED)
            row["kernel_on_the_slopes_minus_pls_on_the_plain_spectra"] = M.mean_difference(
                err[("kernel", "slope")], err[("pls", "plain")], seed=SEED)
            out[f"{a}|{name}"] = row
    return out


def run_set(which: str, repeats: int = REPEATS, **options) -> dict:
    t0 = time.time()
    d = PREP.load(which)
    spec = SETS[which]
    x, S = cut(d, *spec["range"])
    names = [str(n) for n in d["property_names"]]
    cols = [names.index(p) for p in spec["properties"]]
    Y = d["properties"][:, cols]
    instruments = [str(i) for i in d["instruments"]]
    n = S.shape[1]
    out = {"samples": n, "points": len(x), "range_nm": list(spec["range"]),
           "instruments": instruments, "properties": list(spec["properties"]),
           "property_sd": [float(v) for v in Y.std(axis=0)],
           "fit": spec["fit"], "test": n - spec["fit"], "repeats": repeats,
           "difference": {}}
    for a in range(len(S)):
        for b in range(a + 1, len(S)):
            out["difference"][f"{a}->{b}"] = difference(S[a], S[b], x)
    if len(x) < len(d["wavelength"]):       # what the part of the axis that is left out holds
        full = d["spectra"]
        out["difference_on_the_whole_axis"] = {
            f"{a}->{b}": difference(full[a], full[b], d["wavelength"])
            for a in range(len(full)) for b in range(a + 1, len(full))}
    rng = np.random.default_rng(SEED)
    lists: dict = {}
    for r in range(repeats):
        order = rng.permutation(n)
        one = one_split(S, Y, x, np.sort(order[:spec["fit"]]), np.sort(order[spec["fit"]:]),
                        rng, **options)
        _collect(lists, one)
        log(f"  {which}: split {r + 1} of {repeats}, {time.time() - t0:.0f} s")
    out["gate"] = gate(lists["crossing"], spec["properties"])
    out["within_comparison"] = within_comparison(lists["within"], spec["properties"])
    out.update(_summarise(lists))
    out["seconds"] = round(time.time() - t0, 1)
    return out


def run(results: str = RESULTS, repeats: int = REPEATS, sets: Sequence[str] = tuple(SETS),
        **options) -> dict:
    out = {"source": PREP.PAGE, "licence": PREP.LICENCE,
           "standards": options.get("n_standards", N_STANDARDS),
           "more_standards": options.get("more_standards", MORE_STANDARDS),
           "widths_nm": list(options.get("widths", WIDTHS)), "sets": {}}
    for which in sets:
        out["sets"][which] = run_set(which, repeats, **options)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "transfer.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
MODEL_NAMES = {"pls": "PLS", "kernel": "kernel ridge"}
FORM_NAMES = {"plain": "plain spectra", "slope": "slopes"}


def relative(s: dict, leaf) -> float:
    """The mean over the properties of an error divided by the standard deviation of the
    property; `leaf` takes the index of a property and gives the error."""
    return float(np.mean([leaf(str(p)) / sd for p, sd in enumerate(s["property_sd"])]))


def _pair(s: dict, key: str) -> str:
    a, b = key.split("->")
    return f"{s['instruments'][int(a)]} to {s['instruments'][int(b)]}"


def _difference_lines(which: str, s: dict) -> List[str]:
    lines = []
    for key, v in s["difference"].items():
        lines.append(
            f"| {which}, {_pair(s, key)} | {v['spread_of_the_samples']:.4f} | "
            f"{v['difference']:.4f} | {v['after_the_mean_difference']:.4f} | "
            f"{v['after_offset_gain_and_slope_per_sample']:.4f} | "
            f"{v['after_the_shift_too']:.4f} | {v['shift_nm']:+.2f} | "
            f"{abs(v['width_difference_nm']):g} |")
    return lines


def _within_lines(which: str, s: dict) -> List[str]:
    lines = []
    for a, name in enumerate(s["instruments"]):
        node = s["within"][str(a)]
        cells = []
        for model in MODELS:
            for form in FORMS:
                err = relative(s, lambda p: node[p][model][form][VARIANTS[0]]["mean"])
                cells.append(f"{err:.3f}")
        cost = relative(s, lambda p: node[p][OURS[0]][OURS[1]][VARIANTS[1]]["mean"])
        lines.append(f"| {which}, {name} | " + " | ".join(cells) + f" | {cost:.3f} |")
    return lines


def _crossing_lines(which: str, s: dict, model: str, form: str) -> List[str]:
    lines = []
    columns = ("within the source",) + VARIANTS + (PDS_MORE, "fitted on the target")
    for key, node in s["crossing"].items():
        cells = [f"{relative(s, lambda p: node[p][model][form][c]['mean']):.3f}" for c in columns]
        lines.append(f"| {which}, {_pair(s, key)} | " + " | ".join(cells) + " |")
    return lines


def _gate_counts(s: dict) -> dict:
    g = s["gate"].values()
    return {"cases": len(s["gate"]),
            "below_pls_as_measured": sum(1 for v in g
                                         if v["ours_minus_pls_as_measured"]["high"] < 0),
            "within_pds": sum(1 for v in g if v["ours_minus_pls_with_pds"]["low"] <= 0
                              <= v["ours_minus_pls_with_pds"]["high"]),
            "below_pds": sum(1 for v in g if v["ours_minus_pls_with_pds"]["high"] < 0),
            "above_pds": sum(1 for v in g if v["ours_minus_pls_with_pds"]["low"] > 0),
            "above_pds_more_standards": sum(
                1 for v in g if v["ours_minus_pls_with_pds_more_standards"]["low"] > 0),
            "equal_to_pls_on_the_same_rows": sum(
                1 for v in g if v["ours_minus_pls_on_the_same_rows"]["low"] <= 0
                <= v["ours_minus_pls_on_the_same_rows"]["high"])}


def _coarser_lines(which: str, s: dict, same: bool) -> List[str]:
    lines = []
    keys = [k for k in s["coarser"] if (k.split("->")[0] == k.split("->")[1]) == same]
    extras = list(next(iter(s["coarser"][keys[0]].values()))["pls"]["plain"])
    columns = COARSE_VARIANTS[:1] + ((VARIANTS[1],) if same else ()) + COARSE_VARIANTS[1:]
    for model in MODELS:
        for form in FORMS:
            for extra in extras:
                cells = []
                for variant in columns:
                    cells.append(float(np.mean([relative(
                        s, lambda p: s["coarser"][k][p][model][form][extra][variant]["mean"])
                        for k in keys])))
                found = float(np.mean([s["fitted_width"][k][extra]["mean"] for k in keys]))
                lines.append(f"| {which} | {MODEL_NAMES[model]}, {FORM_NAMES[form]} | "
                             f"{float(extra):g} | " + " | ".join(f"{c:.3f}" for c in cells)
                             + f" | {found:.1f} |")
    return lines


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "transfer.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    sets = d["sets"]
    lines = ["# A model fitted on one instrument reads the spectra of another",
             "",
             f"Data: two near-infrared sets from {d['source']} ({d['licence']}; no file of "
             "the sets is in this repository)."]
    for which, s in sets.items():
        lines.append(
            f"- {which}: {s['samples']} samples on {len(s['instruments'])} instruments "
            f"({', '.join(s['instruments'])}), {s['points']} points from "
            f"{s['range_nm'][0]:g} to {s['range_nm'][1]:g} nm; properties: "
            f"{', '.join(s['properties'])}; {s['fit']} samples to fit and {s['test']} to "
            f"test, {s['repeats']} random splits.")
    lines += ["",
              "An error in the tables below is a root-mean-square error divided by the "
              "standard deviation of the property over all samples, and then the mean over "
              "the properties of the set. A model that predicts the mean has about 1.",
              "",
              "## Part 1: what differs between the instruments",
              "",
              "Every number but the last two is a root mean square of absorbance over the "
              "samples and the axis.",
              "",
              "| Pair of instruments | Spread of the samples on the first | Difference "
              "between the instruments | After the mean difference is taken off | After an "
              "offset, a gain and a sloping baseline per sample | After the fitted shift too "
              "| Fitted shift, nm | Fitted difference in resolution, nm |",
              "|---|---|---|---|---|---|---|---|"]
    for which, s in sets.items():
        lines += _difference_lines(which, s)
    lines += ["",
              "## Part 2: within one instrument",
              "",
              "| Instrument | PLS, plain spectra | PLS, slopes | Kernel ridge, plain spectra "
              "| Kernel ridge, slopes | Kernel ridge, slopes, the test spectra centred on "
              "the mean of the fit |",
              "|---|---|---|---|---|---|"]
    for which, s in sets.items():
        lines += _within_lines(which, s)
    lines += ["",
              "The last column is the cost of centring when no instrument is changed: the "
              "samples of the test do not have exactly the mean of the samples of the fit.",
              "",
              "The same, property by property, as root-mean-square errors in the units of "
              "the property. An interval is a 95 % bootstrap interval over the splits.",
              "",
              "| Instrument and property | PLS, plain spectra | PLS, slopes | Kernel ridge, "
              "plain spectra | Kernel ridge, slopes | Kernel ridge minus PLS, both on the "
              "slopes | Kernel ridge on the slopes minus PLS on the plain spectra |",
              "|---|---|---|---|---|---|---|"]
    for which, s in sets.items():
        for key, v in s["within_comparison"].items():
            a, name = key.split("|")
            same = v["kernel_minus_pls_on_the_slopes"]
            ref = v["kernel_on_the_slopes_minus_pls_on_the_plain_spectra"]
            lines.append(
                f"| {which}, {s['instruments'][int(a)]}, {name} | {v['pls_plain']:.4f} | "
                f"{v['pls_slope']:.4f} | {v['kernel_plain']:.4f} | {v['kernel_slope']:.4f} | "
                f"{same['mean']:+.4f} [{same['low']:+.4f}, {same['high']:+.4f}] | "
                f"{ref['mean']:+.4f} [{ref['low']:+.4f}, {ref['high']:+.4f}] |")
    lines.append("")
    head = ("| Source to target | Within the source | Target as measured | Each instrument "
            "centred | Shift fitted | Shift fitted, each instrument centred | Mean "
            f"difference of {d['standards']} standards | PDS with {d['standards']} standards "
            f"| PDS with {d['more_standards']} standards | Model fitted on the target |")
    for model, form in ((OURS[0], OURS[1]), (REFERENCE[0], REFERENCE[1]), ("pls", "slope"),
                        ("kernel", "plain")):
        lines += [f"## Part 2: the crossing, {MODEL_NAMES[model]} on the "
                  f"{FORM_NAMES[form]}", "", head,
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for which, s in sets.items():
            lines += _crossing_lines(which, s, model, form)
        lines.append("")
    lines += ["## The gate of the plan item, case by case",
              "",
              "A case is one crossing and one property. \"Ours\" is kernel ridge on the "
              "slopes with the shift fitted and each instrument centred, which uses no "
              "standard. An interval is a 95 % bootstrap interval over the splits.",
              "",
              "| Set | Cases | Ours below PLS on the plain spectra as measured | Ours within "
              f"the interval of PLS with PDS ({d['standards']} standards) | Ours below PLS "
              "with PDS | Ours above PLS with PDS | Ours above PLS with PDS and "
              f"{d['more_standards']} standards | Ours within the interval of PLS on the "
              "same rows |",
              "|---|---|---|---|---|---|---|---|"]
    for which, s in sets.items():
        c = _gate_counts(s)
        lines.append(f"| {which} | {c['cases']} | {c['below_pls_as_measured']} | "
                     f"{c['within_pds']} | {c['below_pds']} | {c['above_pds']} | "
                     f"{c['above_pds_more_standards']} | "
                     f"{c['equal_to_pls_on_the_same_rows']} |")
    lines += ["",
              "| Crossing and property | Ours | PLS, plain, as measured | PLS with PDS | PLS "
              "on the same rows | Ours minus PLS with PDS |",
              "|---|---|---|---|---|---|"]
    for which, s in sets.items():
        for key, v in s["gate"].items():
            pair, name = key.split("|")
            df = v["ours_minus_pls_with_pds"]
            lines.append(f"| {which}, {_pair(s, pair)}, {name} | {v['ours']:.4f} | "
                         f"{v['pls_as_measured']:.4f} | {v['pls_with_pds']:.4f} | "
                         f"{v['pls_on_the_same_rows']:.4f} | {df['mean']:+.4f} "
                         f"[{df['low']:+.4f}, {df['high']:+.4f}] |")
    for same, title in ((True, "the target is the source instrument with a lower resolution"),
                        (False, "another instrument with a lower resolution")):
        first = ("Read as measured | Each instrument centred, widths not matched" if same
                 else "Shift fitted and each instrument centred, widths not matched")
        lines += ["",
                  f"## Part 3: a simulated difference in resolution, {title}",
                  "",
                  f"| Set | Model and rows | Target broadened by, nm | {first} | "
                  "Widths matched, difference known | Widths matched, difference fitted | "
                  "Fitted difference, nm |",
                  "|---|---|---|---|---|---|---|" + ("---|" if same else "")]
        if not same:
            lines.insert(-2, "In this table every column has the shift fitted and each "
                             "instrument centred.")
            lines.insert(-2, "")
        for which, s in sets.items():
            lines += _coarser_lines(which, s, same)
    lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
    ap.add_argument("--results", default=RESULTS)
    a = ap.parse_args(argv)
    if not a.report:
        run(a.results, a.repeats, a.sets)
    print(report(a.results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
