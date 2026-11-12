#!/usr/bin/env python
"""
Mixtures and quantification: the spectrum of a mixture as a sum of fields (plan item SP6, half a).

By the Beer-Lambert law the absorbance spectrum of a mixture is the sum of the spectra of its
components, each times its amount.  The kernel of `cmfqsar/spectrum.py` is linear in each of
its two spectra, so the kernel of two mixtures is the same weighted sum over the kernels of
their components (`spectrum.mixture_kernel`).  An amount is then a linear function of the
spectrum, and a kernel ridge model on the linear kernel can learn it from mixtures of known
composition.

**What is made.**  Mixtures are synthesised: four components are drawn, and every mixture is
their weighted sum, with a total amount that varies, a baseline of its own (an offset and a
slope) and noise.  Two kinds of components are used: the computed spectra of the benzoic
acids of plan item SP2, which are close relatives and overlap strongly, and measured spectra
of the Chemotion IR set, which carry real band shapes and real baselines (this part needs
`prepare_chemotion_ir.py` and is left out when its cache is missing).  A third case adds
stray light, which bends the calibration curve at high absorbance.

**What is asked.**  From the spectrum of a mixture, the amount of every component and the
fraction of every component, which does not depend on the total amount.  The form of the
kernel that is not normalised is the one for amounts.  The cosine form divides the total
amount out, and the plan names it for fractions; a fraction is also had from the predicted
amounts, divided by their sum.  All of these are reported.

A fourth case gives the test mixtures a baseline that the mixtures to fit did not have, which
is what a new instrument or a new sample holder does.

**References.**  Partial least squares regression (PLS) on the same rows, which is how
quantification is done in practice; and a least-squares fit with the known spectra of the
pure components and a baseline, which is the best that the noise allows and is called the
floor.  A model that does not know the pure spectra cannot be better than the floor.

Usage
-----
    python benchmarks/run_mixtures.py            # the benchmark, then the table (15 minutes)
    python benchmarks/run_mixtures.py --report
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

from cmfqsar import spectrum as SP  # noqa: E402
import prepare_chemotion_ir as PREP  # noqa: E402
import run_spectra_measured as M  # noqa: E402

RESULTS = os.path.join(HERE, "results", "mixtures")
ACIDS = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir.json")
#: the axis of the synthesised curves, the one of the prepared measured spectra
AXIS = np.arange(PREP.GRID[0], PREP.GRID[1] + 1e-9, PREP.GRID[2])
GRID = M.GRID                       # the grid of the kernel, 50 cm^-1 inside the curves
N_COMPONENTS = 4
N_TRAIN, N_TEST = 120, 120
REPEATS = 20
NOISE = 0.005                       # standard deviation, in units of the strongest pure band
OFFSET, SLOPE = 0.2, 0.1            # the largest offset and the largest rise of a baseline
TOTAL = (0.5, 2.0)                  # the range of the total amount of a mixture
NATURAL_WIDTH = 6.0                 # the width given to a computed line, cm^-1
STRAY = 0.02                        # the share of stray light in the nonlinear case
STRAY_SCALE = 1.5                   # the strongest pure band of that case, absorbance units
WIDTHS = (0.0, 4.0, 8.0, 16.0)
RIDGES = tuple(10.0 ** p for p in range(-9, 1))
LENGTHS = (1.0, 4.0, 16.0, 64.0, 256.0)     # squared length of the Gaussian layer, in medians
PLS_MAX = 10
INNER = 4
FORMS = ("plain", "slope")
SEED = 20261113


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# The components and the mixtures
# --------------------------------------------------------------------------- #
def computed_components(path: str = ACIDS) -> np.ndarray:
    """The computed spectra of the benzoic acids as curves on the axis, one row each: every
    line broadened to the natural width, and every curve scaled so that its strongest band
    is 1."""
    with open(path, encoding="utf-8") as fh:
        acids = json.load(fh)["acids"]
    grid = SP.Grid(float(AXIS[0]), float(AXIS[-1]), float(AXIS[1] - AXIS[0]))
    sticks = [SP.Spectrum(a["wavenumbers"], a["intensities"]) for a in acids]
    F = SP.feature_matrix(sticks, grid, NATURAL_WIDTH)
    return F / F.max(axis=1, keepdims=True)


def measured_components() -> Optional[np.ndarray]:
    """The measured spectra of the Chemotion IR set without the copies, each scaled so that
    its range is 1; None when the set was not prepared."""
    if not (os.path.isfile(PREP.SPECTRA) and os.path.isfile(PREP.TABLE)):
        return None
    x, A, table = PREP.load()
    keep, _ = M.without_copies(A, [r["smiles"] for r in table])
    A = A[keep]
    span = A.max(axis=1) - A.min(axis=1)
    A = A[span > 0.05]                        # a curve with no band is no component
    return A / (A.max(axis=1) - A.min(axis=1))[:, None]


def make_mixtures(S: np.ndarray, n: int, rng, noise: float = NOISE, offset: float = OFFSET,
                  slope: float = SLOPE, total=TOTAL, stray: float = 0.0,
                  scale: float = 1.0, flat: int = 0) -> Dict[str, np.ndarray]:
    """`n` mixtures of the components `S` (one row each, on the axis).

    The fractions are drawn evenly from all compositions, and the total amount evenly from
    `total`.  The spectrum is the sum of the component spectra times their amounts, times
    `scale`.  With `stray` the detector sees that share of light that did not pass the
    sample, which makes the measured absorbance ``-log10((10^-A + s) / (1 + s))`` and bends
    the calibration curve where the absorbance is high.  Every mixture gets a baseline (an
    offset up to `offset` and a straight rise or fall up to `slope` over the axis) and noise.
    The first `flat` mixtures get no baseline.
    """
    m, g = S.shape
    fractions = rng.dirichlet(np.ones(m), size=n)
    amounts = fractions * rng.uniform(total[0], total[1], size=(n, 1))
    A = float(scale) * (amounts @ S)
    if stray > 0:
        A = -np.log10((10.0 ** (-A) + stray) / (1.0 + stray))
    ramp = np.linspace(-0.5, 0.5, g)
    base = (rng.uniform(0.0, offset, size=(n, 1))
            + rng.uniform(-slope, slope, size=(n, 1)) * ramp[None, :])
    base[:flat] = 0.0
    X = A + base + noise * rng.normal(size=(n, g))
    return {"X": X, "amounts": amounts, "fractions": fractions}


def floor_estimate(X: np.ndarray, S: np.ndarray, scale: float = 1.0) -> Dict[str, np.ndarray]:
    """The amounts and the fractions from a least-squares fit of every mixture with the known
    spectra of the pure components, an offset and a straight line.  With white noise and a
    linear response no estimate from one spectrum is better."""
    g = S.shape[1]
    design = np.vstack([float(scale) * S, np.ones(g), np.linspace(-0.5, 0.5, g)]).T
    coef, *_ = np.linalg.lstsq(design, X.T, rcond=None)
    amounts = coef[: len(S)].T
    total = amounts.sum(axis=1, keepdims=True)
    return {"amounts": amounts, "fractions": amounts / np.where(np.abs(total) > 1e-9, total, 1)}


# --------------------------------------------------------------------------- #
# The rows of a description, and the models
# --------------------------------------------------------------------------- #
def finished(F: np.ndarray, form: str, cosine: bool) -> np.ndarray:
    """Broadened curves as the rows of a description: as they are or as slopes, and each
    divided by its length for the cosine form."""
    G = SP.slope(F, GRID) if form == "slope" else np.asarray(F, float)
    if cosine:
        G = G / np.sqrt(np.einsum("ij,ij->i", G, G))[:, None]
    return G


def rows(X: np.ndarray, width: float, form: str, cosine: bool) -> np.ndarray:
    """The curves on the grid of the kernel: broadened to `width`, then :func:`finished`."""
    return finished(M.broadened(X, AXIS, width), form, cosine)


def gaussian_layer(K: np.ndarray, train: np.ndarray, length: float) -> np.ndarray:
    """``exp(-d^2 / (2 l^2))`` with the squared distance ``d^2`` of the linear kernel `K` and
    ``l^2`` equal to `length` times the median squared distance of the training rows."""
    d = np.diag(K)
    D = np.maximum(d[:, None] + d[None, :] - 2.0 * K, 0.0)
    sub = D[np.ix_(train, train)]
    med = float(np.median(sub[np.triu_indices(len(train), 1)]))
    return np.exp(-D / (2.0 * length * max(med, 1e-300)))


def ridge_path(K: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               ridges: Sequence[float] = RIDGES) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Kernel ridge regression with a constant, for every ridge at once.

    The kernel of the training rows is centred, which is the constant of the model.  Every
    ridge is a multiple of the mean diagonal of the centred kernel.  Returns the predictions
    for the rows `te` (one array per ridge), the dual coefficients (the same) and the ridges
    in the units of the kernel.
    """
    Ktr = K[np.ix_(tr, tr)]
    mu, mean = Ktr.mean(axis=0), Ktr.mean()
    Kc = Ktr - mu[None, :] - mu[:, None] + mean
    Kte = K[np.ix_(te, tr)]
    Kcte = Kte - mu[None, :] - Kte.mean(axis=1)[:, None] + mean
    ybar = Y[tr].mean(axis=0)
    e, V = np.linalg.eigh(Kc)
    proj = V.T @ (Y[tr] - ybar)
    lam = np.array(ridges, float) * max(float(np.trace(Kc)) / len(tr), 1e-300)
    alphas = np.array([V @ (proj / (np.maximum(e, 0.0) + r)[:, None]) for r in lam])
    return np.array([Kcte @ a + ybar for a in alphas]), alphas, lam


def rmse(P: np.ndarray, Y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(P) - np.asarray(Y)) ** 2)))


def inner_folds(n: int, k: int, rng) -> List[np.ndarray]:
    return [np.sort(f) for f in np.array_split(rng.permutation(n), k)]


def kernel_model(kernels: Dict[str, np.ndarray], Y: np.ndarray, tr: np.ndarray,
                 te: np.ndarray, folds: Sequence[np.ndarray]) -> dict:
    """Kernel ridge with the kernel (one of `kernels`, by name) and the ridge chosen by
    cross-validation inside the training rows; the prediction for the rows `te`."""
    best = None
    for name, K in kernels.items():
        err = np.zeros(len(RIDGES))
        for f in folds:
            inner_te = tr[f]
            inner_tr = np.setdiff1d(tr, inner_te)
            P, _, _ = ridge_path(K, Y, inner_tr, inner_te)
            err += np.array([np.sum((p - Y[inner_te]) ** 2) for p in P])
        k = int(np.argmin(err))
        if best is None or err[k] < best[0]:
            best = (float(err[k]), name, k)
    _, name, k = best
    P, alphas, _ = ridge_path(kernels[name], Y, tr, te)
    return {"prediction": P[k], "kernel": name, "ridge": float(RIDGES[k]), "alpha": alphas[k]}


def pls_path(F: np.ndarray, Y: np.ndarray, tr: np.ndarray, te: np.ndarray,
             most: int) -> np.ndarray:
    """The predictions of partial least squares regression for the rows `te` with 1, 2, ...,
    `most` components, from one fit.  The components are found one after the other, so the
    model with ``k`` components is the first ``k`` of the model with more."""
    from sklearn.cross_decomposition import PLSRegression

    fit = PLSRegression(n_components=most, scale=False).fit(F[tr], Y[tr])
    scores = (F[te] - F[tr].mean(axis=0)) @ fit.x_rotations_
    ybar = Y[tr].mean(axis=0)
    return np.array([scores[:, :k] @ fit.y_loadings_[:, :k].T + ybar
                     for k in range(1, most + 1)])


def pls_model(features: Dict[str, np.ndarray], Y: np.ndarray, tr: np.ndarray, te: np.ndarray,
              folds: Sequence[np.ndarray], most: int = PLS_MAX) -> dict:
    """Partial least squares regression on the rows of one of `features`, with the rows and
    the number of components chosen by cross-validation inside the training rows."""
    best = None
    top = min(most, len(tr) - len(folds[0]) - 1)
    for name, F in features.items():
        err = np.zeros(top)
        for f in folds:
            inner_te = tr[f]
            inner_tr = np.setdiff1d(tr, inner_te)
            P = pls_path(F, Y, inner_tr, inner_te, top)
            err += np.array([np.sum((p - Y[inner_te]) ** 2) for p in P])
        k = int(np.argmin(err)) + 1
        if best is None or err[k - 1] < best[0]:
            best = (float(err[k - 1]), name, k)
    _, name, k = best
    return {"prediction": pls_path(features[name], Y, tr, te, k)[-1], "rows": name,
            "components": k}


def net_signal(S: np.ndarray, k: int, others: Optional[np.ndarray] = None) -> np.ndarray:
    """The part of the spectrum of component `k` that no combination of the other components
    (and of the rows `others`, a baseline for example) can make: the net analyte signal."""
    rest = np.delete(S, k, axis=0)
    if others is not None:
        rest = np.vstack([rest, others])
    coef, *_ = np.linalg.lstsq(rest.T, S[k], rcond=None)
    return S[k] - rest.T @ coef


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))


# --------------------------------------------------------------------------- #
# One repeat, one case
# --------------------------------------------------------------------------- #
DESCRIPTIONS = tuple((form, cos) for form in FORMS for cos in (False, True))
QUESTIONS = ("amounts", "fractions")
METHODS = ("kernel", "gaussian", "pls")
DIVIDED = ", amounts divided by their sum"


def describe(form: str, cos: bool) -> str:
    return f"{form}, {'cosine' if cos else 'not normalised'}"


def as_fractions(amounts: np.ndarray) -> np.ndarray:
    """Predicted amounts divided by their sum."""
    total = amounts.sum(axis=1, keepdims=True)
    return amounts / np.where(np.abs(total) > 1e-9, total, 1.0)


def one_repeat(pool: np.ndarray, rng, stray: float = 0.0, scale: float = 1.0,
               n_train: int = N_TRAIN, n_test: int = N_TEST, widths=WIDTHS,
               unseen_baseline: bool = False) -> dict:
    """Draw the components and the mixtures, fit every model, and return the errors on the
    test mixtures as ``out[method][question][rows]``.  With `unseen_baseline` the mixtures
    to fit have no baseline and the test mixtures have one."""
    S = pool[rng.choice(len(pool), N_COMPONENTS, replace=False)]
    data = make_mixtures(S, n_train + n_test, rng, stray=stray, scale=scale,
                         flat=n_train if unseen_baseline else 0)
    tr, te = np.arange(n_train), np.arange(n_train, n_train + n_test)
    folds = inner_folds(n_train, INNER, rng)
    floor = floor_estimate(data["X"][te], S, scale)
    out: dict = {"floor": {q: rmse(floor[q], data[q][te]) for q in QUESTIONS},
                 "chosen": {}}
    for method in METHODS:
        out[method] = {q: {} for q in QUESTIONS}
    plain = {w: M.broadened(data["X"], AXIS, w) for w in widths}
    for form, cos in DESCRIPTIONS:
        name = describe(form, cos)
        feats = {f"{w:g}": finished(plain[w], form, cos) for w in widths}
        linear = {w: F @ F.T for w, F in feats.items()}
        curved = {f"{w}|{length:g}": gaussian_layer(K, tr, length)
                  for w, K in linear.items() for length in LENGTHS}
        for q in QUESTIONS:
            Y = data[q]
            fits = {"kernel": kernel_model(linear, Y, tr, te, folds),
                    "gaussian": kernel_model(curved, Y, tr, te, folds),
                    "pls": pls_model(feats, Y, tr, te, folds)}
            for method, fit in fits.items():
                out[method][q][name] = rmse(fit["prediction"], Y[te])
                if q == "amounts" and not cos:
                    out[method]["fractions"][name + DIVIDED] = rmse(
                        as_fractions(fit["prediction"]), data["fractions"][te])
            out["chosen"][f"{name}|{q}"] = fits["kernel"]["kernel"]
            if form == "slope" and not cos and q == "amounts" and stray == 0:
                # the co-spectrum of every component against its net signal, in the rows
                # of the description: the slopes of the pure spectra at the chosen width
                w = float(fits["kernel"]["kernel"])
                alpha = fits["kernel"]["alpha"]
                pure = rows(scale * S, w, "slope", False)
                ramp = rows(np.linspace(-0.5, 0.5, len(AXIS))[None, :], w, "slope", False)
                centre = alpha - alpha.mean(axis=0, keepdims=True)
                F = feats[fits["kernel"]["kernel"]]
                out["net_signal_cosine"] = float(np.mean([
                    cosine(SP.co_spectrum(F[tr], centre[:, k], normalize=False),
                           net_signal(pure, k, ramp)) for k in range(len(S))]))
    return out


def summarise(repeats: List[dict]) -> dict:
    """Means and standard deviations over the repeats, and for every row the differences
    between the methods with a 95 % bootstrap interval over the repeats."""
    out: dict = {"repeats": len(repeats), "floor": {}, "differences": {}}
    for q in QUESTIONS:
        out["floor"][q] = _mean_sd([r["floor"][q] for r in repeats])
        out["differences"][q] = {}
        floor = np.array([r["floor"][q] for r in repeats])
        for method in METHODS:
            out.setdefault(method, {})[q] = {
                name: _mean_sd([r[method][q][name] for r in repeats])
                for name in repeats[0][method][q]}
        for name in repeats[0]["kernel"][q]:
            k, g, p = (np.array([r[m][q][name] for r in repeats]) for m in METHODS)
            out["differences"][q][name] = {
                "kernel_minus_pls": M.mean_difference(k, p, seed=SEED),
                "gaussian_minus_kernel": M.mean_difference(g, k, seed=SEED),
                "kernel_over_floor": float(np.mean(k / floor))}
    widths: Dict[str, int] = {}
    for r in repeats:
        w = r["chosen"][describe("slope", False) + "|amounts"]
        widths[w] = widths.get(w, 0) + 1
    out["width_chosen_slope_amounts"] = dict(sorted(widths.items(), key=lambda kv: -kv[1]))
    cos = [r["net_signal_cosine"] for r in repeats if "net_signal_cosine" in r]
    if cos:
        out["net_signal_cosine"] = _mean_sd(cos)
    return out


def _mean_sd(values: Sequence[float]) -> dict:
    return {"mean": float(np.mean(values)), "sd": float(np.std(values))}


def identity_check(pool: np.ndarray, rng) -> float:
    """The largest difference between the kernel of mixtures made as spectra and the product
    ``W K W^T`` of `spectrum.mixture_kernel`, over the largest element of the kernel."""
    S = pool[rng.choice(len(pool), N_COMPONENTS, replace=False)]
    W = rng.uniform(0.0, 1.0, size=(12, N_COMPONENTS))
    F = rows(S, 8.0, "slope", False)
    direct = rows(W @ S, 8.0, "slope", False)
    K = direct @ direct.T
    return float(np.max(np.abs(K - SP.mixture_kernel(F @ F.T, W))) / np.max(np.abs(K)))


def run(results: str = RESULTS, repeats: int = REPEATS, **sizes) -> dict:
    """All cases; `sizes` (``n_train``, ``n_test``, ``widths``) go to :func:`one_repeat`."""
    t0 = time.time()
    n_train, n_test = sizes.get("n_train", N_TRAIN), sizes.get("n_test", N_TEST)
    computed = computed_components()
    measured = measured_components()
    acids = "computed spectra of benzoic acids"
    cases = [("computed", acids, computed, {})]
    if measured is not None:
        cases.append(("measured", "measured spectra of the Chemotion IR set", measured, {}))
    cases.append(("stray", acids + ", with stray light", computed,
                  {"stray": STRAY, "scale": STRAY_SCALE}))
    cases.append(("unseen", acids + ", with a baseline only in the test mixtures", computed,
                  {"unseen_baseline": True}))
    out = {"components": N_COMPONENTS, "train": n_train, "test": n_test, "noise": NOISE,
           "offset": OFFSET, "slope": SLOPE, "total": list(TOTAL), "stray": STRAY,
           "stray_scale": STRAY_SCALE, "pool": {"computed": int(len(computed))},
           "identity": identity_check(computed, np.random.default_rng(SEED)), "cases": {}}
    if measured is not None:
        out["pool"]["measured"] = int(len(measured))
    for k, (key, label, pool, options) in enumerate(cases):
        rng = np.random.default_rng(SEED + k)
        done = []
        for r in range(repeats):
            done.append(one_repeat(pool, rng, **options, **sizes))
            if (r + 1) % 5 == 0:
                log(f"  {key}: {r + 1} of {repeats} repeats, {time.time() - t0:.0f} s")
        out["cases"][key] = dict(summarise(done), label=label,
                                 linear=not options.get("stray"))
    out["seconds"] = round(time.time() - t0, 1)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "mixtures.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #
def _num(d: dict) -> str:
    return f"{d['mean']:.4f} ({d['sd']:.4f})"


def _interval(d: dict) -> str:
    return f"{d['mean']:+.4f} [{d['low']:+.4f}, {d['high']:+.4f}]"


WORDS = {"amounts": "the amount of every component",
         "fractions": "the fraction of every component"}


def _case_lines(c: dict) -> List[str]:
    lines = [f"## Components: {c['label']}", ""]
    for q in QUESTIONS:
        floor = ("Error of the floor" if c.get("linear", True) else
                 "Error of the least-squares fit with the pure spectra, which is not a floor "
                 "here because it takes the response as linear")
        lines += [f"Question: {WORDS[q]}. {floor}: {_num(c['floor'][q])}.",
                  "",
                  "| Rows of the description | Kernel ridge, linear kernel | Kernel ridge, "
                  "Gaussian layer | PLS on the same rows | Linear kernel minus PLS | "
                  "Gaussian layer minus linear kernel | Linear kernel over the floor |",
                  "|---|---|---|---|---|---|---|"]
        for name in c["kernel"][q]:
            d = c["differences"][q][name]
            lines.append(f"| {name} | {_num(c['kernel'][q][name])} | "
                         f"{_num(c['gaussian'][q][name])} | {_num(c['pls'][q][name])} | "
                         f"{_interval(d['kernel_minus_pls'])} | "
                         f"{_interval(d['gaussian_minus_kernel'])} | "
                         f"{d['kernel_over_floor']:.2f} |")
        lines.append("")
    if "net_signal_cosine" in c:
        lines += ["The co-spectrum of the model for an amount (slopes, not normalised) "
                  "against the net signal of that component, which is the part of its pure "
                  "spectrum that the other components and a baseline cannot make: cosine "
                  f"{_num(c['net_signal_cosine'])}.",
                  ""]
    return lines


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = os.path.join(results, "mixtures.json")
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    lines = [
        "# Mixtures and quantification: the spectrum of a mixture as a sum of fields "
        "(plan item SP6, half a)",
        "",
        f"Every repeat draws {d['components']} components and makes {d['train']} mixtures "
        f"to fit and {d['test']} to test. The total amount of a mixture is between "
        f"{d['total'][0]:g} and {d['total'][1]:g}; every mixture has a baseline (an offset up "
        f"to {d['offset']:g} and a rise or fall up to {d['slope']:g} over the axis) and noise "
        f"of standard deviation {d['noise']:g}, all in units of the strongest band of a pure "
        "component. The numbers are root-mean-square errors on the test mixtures, the mean "
        "over the repeats with the standard deviation in brackets. The width of the "
        "broadening, the ridge, the length of the Gaussian layer and the number of PLS "
        "components are chosen by cross-validation inside the mixtures to fit.",
        "",
        "The floor is a least-squares fit of every test mixture with the known spectra of "
        "the pure components and a baseline. No model that has to learn from mixtures can be "
        "better, as long as the response is linear. A difference is given with a 95 % "
        "bootstrap interval over the repeats; below zero the first of the two is better.",
        "",
        "The identity: the kernel of mixtures made as spectra and the product W K W^T of "
        f"`spectrum.mixture_kernel` differ by {d['identity']:.1e} of the largest element.",
        "",
    ]
    for c in d["cases"].values():
        lines += _case_lines(c)
    text = "\n".join(lines)
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    ap.add_argument("--repeats", type=int, default=REPEATS)
    a = ap.parse_args(argv)
    if not a.report:
        run(repeats=a.repeats)
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
