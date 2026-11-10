#!/usr/bin/env python
"""
A spectrum as a one-dimensional field: a calculated powder pattern as the sample (plan item
SP1, the half that needs no new data).

`cmfqsar/spectrum.py` gives the kernel of two spectra and the co-spectrum.  This script asks
what a powder diffraction pattern, which is what a diffraction experiment on a powder
measures, says about a property of the crystal.  The set is the Matbench phonons task: 1,265
inorganic crystals with the frequency of the highest peak of the phonon density of states.
The pattern of every crystal is calculated from its structure with pymatgen, for X-rays
(Cu K-alpha) and for neutrons of the same wavelength.  No measured pattern is used.

The questions, in the order of the gate of the item:

1. How alike are the kernel of the patterns and the kernel of the structures (the periodic
   autocorrelation of `run_matbench_phonons.py`)?  The correlation of the two kernels is
   reported first.
2. What does the pattern alone predict, against the structural kernel on the same folds,
   against the composition (the fractions of the elements) and against one number of the
   composition (the mass of the lightest atom)?
3. Does the pattern add anything to the structural kernel, or to the composition, when it joins
   them as one more kernel in the mixing of `cmfqsar/mixing.py`?
4. Where on the axis does the model put its weight (the co-spectrum)?

Protocol: the five Matbench folds.  Inside every training part the settings of a description
(the width of the broadening, the power of the intensity, the ridge) are chosen by 10-fold
cross-validation, as `run_matbench_phonons.py` does.  `--repeats` adds other divisions into
five folds.  A difference in the mean absolute error carries a 95 % bootstrap interval over
the structures.

The whole run took 3.5 hours on the author's machine while other jobs ran on it.  More than
half of that are the two rows with a Gaussian kernel on top of the pattern kernel, which try
80 settings in every fold.  The powder patterns, the structural features and the pair distribution
functions are kept in ``benchmarks/cache/spectra`` (not part of the repository).

Usage
-----
    python benchmarks/run_spectra.py                  # every stage, then the table
    python benchmarks/run_spectra.py --stage patterns # the powder patterns, 6 minutes
    python benchmarks/run_spectra.py --stage fit
    python benchmarks/run_spectra.py --stage evidence # the marginal likelihoods again
    python benchmarks/run_spectra.py --stage cospectrum
    python benchmarks/run_spectra.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar import spectrum as SP  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.invariant import autocorrelation_kernel, channel_scale  # noqa: E402
from cmfqsar.mixing import (LAM_GRID, best_ridge_evidence, drop_one,  # noqa: E402
                            evidence_mixing, joint_kernel, with_views)
from cmfqsar.model import _Learner, cv_predict, r2_score, rmse  # noqa: E402
from cmfqsar.periodic import (PeriodicSpec, crystal_fields, inorganic_fieldset,  # noqa: E402
                              periodic_block_sizes, periodic_feature_matrix,
                              structures_from_matbench)
import run_matbench_phonons as P  # noqa: E402

CACHE = os.path.join(HERE, "cache", "spectra")
RESULTS = os.path.join(HERE, "results", "spectra")

#: the X-ray line of a laboratory instrument and its wavelength in angstrom
WAVELENGTH, LAMBDA = "CuKa", 1.54184
#: the range of the diffraction angle 2-theta that a laboratory pattern covers, in degrees
TWO_THETA = (5.0, 90.0)
SOURCES = ("xray", "neutron")
SOURCE_LABEL = {"xray": "X-ray powder pattern", "neutron": "neutron powder pattern"}
#: the widths of the broadening tried, in degrees of 2-theta.  A laboratory instrument has
#: peaks about 0.05 to 0.1 degrees wide; the wide ones compare regions and not single peaks.
SIGMAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
#: the powers of the intensity tried: 1 keeps the intensity, 0.5 takes its square root
POWERS = (1.0, 0.5)
LAMS = P.LAMS
GAMMAS = P.GAMMAS
OUTER, INNER = P.OUTER, P.INNER
#: the structural kernel of the comparison: the variant of run_matbench_phonons.py with the
#: lowest error (three physico-chemical fields and the mass field, background kept)
STRUCTURE_VARIANT = "physchem+mass-nobg"
#: the pair distribution function of plan item M12, on the grid of its short range
PAIR_SPEC = dict(r_min=1.0, r_max=10.0, r_step=0.05, sigma=0.05)
RUNS = 2000
SCRAMBLES = 20
#: the bands of 2-theta, in degrees, over which the co-spectrum is summed for the table
BAND = 10.0


def log(msg: str) -> None:
    print(msg, flush=True)


def result(name: str, results: str = RESULTS) -> str:
    return os.path.join(results, name)


# --------------------------------------------------------------------------- #
# The data and the powder patterns
# --------------------------------------------------------------------------- #
def load():
    """The structures, the target and the names of the Matbench phonons task."""
    atoms, y, names = structures_from_matbench(P.fetch())
    return atoms, y, names


def pack(spectra: Sequence[SP.Spectrum]) -> Dict[str, np.ndarray]:
    """Spectra of different lengths as three arrays, for one file."""
    sizes = np.array([len(s) for s in spectra], int)
    x = np.concatenate([s.x for s in spectra]) if len(spectra) else np.zeros(0)
    y = np.concatenate([s.y for s in spectra]) if len(spectra) else np.zeros(0)
    return {"x": x, "y": y, "sizes": sizes}


def unpack(z, names: Optional[Sequence[str]] = None, axis: str = "two_theta"):
    """The inverse of :func:`pack`."""
    ends = np.cumsum(z["sizes"])
    starts = ends - z["sizes"]
    return [SP.Spectrum(z["x"][a:b], z["y"][a:b], "sticks", axis,
                        names[k] if names is not None else "")
            for k, (a, b) in enumerate(zip(starts, ends))]


def patterns(atoms, names, source: str, force: bool = False, cache: str = CACHE):
    """The powder pattern of every crystal as sticks over 2-theta, computed once and kept."""
    path = os.path.join(cache, f"phonons_{source}_{WAVELENGTH}.npz")
    if os.path.isfile(path) and not force:
        with np.load(path) as z:
            if len(z["sizes"]) == len(atoms):
                return unpack(z, names)
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for a, nm in zip(atoms, names):
            out.append(SP.powder_sticks(a, source, WAVELENGTH, TWO_THETA, nm))
    np.savez_compressed(path, **pack(out))
    sizes = [len(s) for s in out]
    log(f"  {SOURCE_LABEL[source]}: {len(out)} crystals, {int(np.median(sizes))} reflections "
        f"at the median ({min(sizes)} to {max(sizes)}), {time.time() - t0:.0f} s")
    return out


def setting_name(sigma: float, power: float) -> str:
    return f"{sigma:g}|{power:g}"


def setting_values(name: str) -> Tuple[float, float]:
    s, p = name.split("|")
    return float(s), float(p)


def pattern_features(spectra, sigma: float, power: float) -> Tuple[SP.Grid, np.ndarray]:
    """The broadened patterns of the set on the grid made for this width."""
    grid = SP.Grid.for_width(TWO_THETA[0], TWO_THETA[1], sigma)
    return grid, SP.feature_matrix(spectra, grid, sigma, power)


def pattern_kernels(spectra, sigmas=SIGMAS, powers=POWERS) -> Dict[str, np.ndarray]:
    """The cosine kernel of the set for every width and every power of the intensity."""
    out = {}
    for s in sigmas:
        for p in powers:
            out[setting_name(s, p)] = SP.spectrum_kernel(pattern_features(spectra, s, p)[1])
    return out


# --------------------------------------------------------------------------- #
# Descriptions: each gives a training kernel and a test kernel for a setting
# --------------------------------------------------------------------------- #
class Fixed:
    """A description whose kernel does not depend on the training part."""

    def __init__(self, kernels: Dict[str, np.ndarray]):
        self.kernels = kernels
        self.settings = list(kernels)

    def pair(self, setting: str, tr: np.ndarray, te: np.ndarray):
        K = self.kernels[setting]
        return K[np.ix_(tr, tr)], K[np.ix_(te, tr)]


class Layered:
    """A Gaussian kernel on top of the kernels of another description: exp(-gamma D^2 / m),
    with D^2 = 2 - 2 K the squared distance of two normalised samples and m the median of
    D^2 over the training part.  The model is then no longer linear in the spectrum, so it has
    no co-spectrum."""

    def __init__(self, base: Fixed, gammas: Sequence[float] = GAMMAS):
        self.base = base
        self.settings = [f"{s}|g{g:g}" for s in base.settings for g in gammas]

    def pair(self, setting: str, tr: np.ndarray, te: np.ndarray):
        inner, gamma = setting.rsplit("|g", 1)
        K_tr, K_te = self.base.pair(inner, tr, te)
        med = P.median_d2(K_tr)
        return (P.gauss_layer(K_tr, K_tr, float(gamma), med),
                P.gauss_layer(K_te, K_tr, float(gamma), med))


class Structural:
    """The periodic autocorrelation kernel, with the channel scale of the training part."""

    def __init__(self, feats: Dict[str, np.ndarray], specs: Dict[str, PeriodicSpec],
                 n_fields: int):
        self.feats, self.specs, self.n_fields = feats, specs, n_fields
        self.settings = list(feats)

    def pair(self, setting: str, tr: np.ndarray, te: np.ndarray):
        F, spec = self.feats[setting], self.specs[setting]
        blocks = periodic_block_sizes(spec, self.n_fields)
        sc = channel_scale(F[tr], spec.n_bins, blocks=blocks)
        K_tr = autocorrelation_kernel(F[tr], scale=sc, n_bins=spec.n_bins, blocks=blocks)
        K_te = autocorrelation_kernel(F[tr], F[te], scale=sc, n_bins=spec.n_bins,
                                      blocks=blocks)
        return K_tr, K_te


class Composition:
    """The fractions of the elements: the cosine kernel, or a Gaussian kernel on top of it
    whose width is set from the median distance of the training part."""

    def __init__(self, X: np.ndarray, gammas: Sequence[float] = GAMMAS):
        norm = np.sqrt(np.einsum("ij,ij->i", X, X))
        Z = X / np.where(norm > 1e-300, norm, 1.0)[:, None]
        self.K = Z @ Z.T
        self.settings = ["linear"] + [f"gauss {g:g}" for g in gammas]

    def pair(self, setting: str, tr: np.ndarray, te: np.ndarray):
        K_tr, K_te = self.K[np.ix_(tr, tr)], self.K[np.ix_(te, tr)]
        if setting == "linear":
            return K_tr, K_te
        gamma = float(setting.split()[1])
        med = P.median_d2(K_tr)
        return P.gauss_layer(K_tr, K_tr, gamma, med), P.gauss_layer(K_te, K_tr, gamma, med)


def structural_view(atoms, names, force: bool = False, cache: str = CACHE):
    """The structural kernel of the comparison, built as run_matbench_phonons.py builds it,
    and the kernel of the pair distribution functions of the same crystals.  Both are computed
    once and kept, because the fields of 1,265 crystals take several minutes."""
    cfg = P.VARIANTS[STRUCTURE_VARIANT]
    fs = inorganic_fieldset([], physchem=cfg["physchem"], charge_model="oxidation")
    keys = [f"{s:g}" for s in P.SIGMAS]
    specs = {key: PeriodicSpec(r_max=P.R_MAX, step=s, sigma=s,
                               taper=cfg.get("taper", P.TAPER),
                               background=cfg.get("background", True))
             for key, s in zip(keys, P.SIGMAS)}
    path = os.path.join(cache, f"phonons_structure_{STRUCTURE_VARIANT}.npz")
    if os.path.isfile(path) and not force:
        with np.load(path) as z:
            if len(z["pair"]) in (0, len(atoms)) and len(z[keys[0]]) == len(atoms):
                feats = {key: z[key] for key in keys}
                pair = z["pair"] if len(z["pair"]) else None
                return Structural(feats, specs, fs.n_fields), pair
    crystals = [crystal_fields(a, fs, nm) for a, nm in zip(atoms, names)]
    feats = {key: periodic_feature_matrix(crystals, specs[key], n_fields=fs.n_fields)
             for key in keys}
    pair = pair_distribution_kernel(crystals)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, pair=pair if pair is not None else np.zeros((0, 0)), **feats)
    return Structural(feats, specs, fs.n_fields), pair


def element_fractions(atoms) -> Tuple[np.ndarray, List[str]]:
    """One row per crystal: the fraction of the atoms of the cell that each element has."""
    elements = sorted({s for a in atoms for s in a.get_chemical_symbols()})
    index = {el: k for k, el in enumerate(elements)}
    X = np.zeros((len(atoms), len(elements)))
    for i, a in enumerate(atoms):
        symbols = a.get_chemical_symbols()
        for s in symbols:
            X[i, index[s]] += 1.0 / len(symbols)
    return X, elements


def lightest_atom(atoms) -> np.ndarray:
    """One number per crystal: the inverse square root of the mass of its lightest atom.  The
    frequency of a vibration goes as the inverse square root of the mass that moves."""
    return np.array([1.0 / np.sqrt(float(np.min(a.get_masses()))) for a in atoms])


def pair_distribution_kernel(crystals) -> Optional[np.ndarray]:
    """The cosine kernel of the pair distribution functions G(r) of plan item M12 (X-ray
    weights), or None when an element of the set has no weight."""
    from cmfqsar import scattering

    spec = scattering.CurveSpec(**PAIR_SPEC)
    try:
        F = scattering.computed_curves(crystals, spec, "xray")
    except ValueError:
        return None
    return scattering.curve_kernel(F)


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def folds_for(n: int, repeat: int = 0):
    """Five folds: the Matbench division for repeat 0, another division for every other one."""
    from sklearn.model_selection import KFold

    seed = P.SPLIT_SEED + repeat
    return list(KFold(n_splits=OUTER, shuffle=True, random_state=seed).split(np.arange(n)))


def choose(K_tr: np.ndarray, y_tr: np.ndarray, lams=LAMS, inner: int = INNER):
    """The ridge with the lowest inner cross-validated RMSE on one training kernel.

    The inner folds and the model are the ones of ``cv_predict(K, y, "krr", ..., seed=0)``:
    kernel ridge regression with a constant.  The kernel of every inner training part is
    decomposed once, ``K = Q L Q^T``, and every ridge is solved from that decomposition.  With
    ``A = K + lam I`` the constant is ``b = (1^T A^-1 y) / (1^T A^-1 1)`` and the coefficients
    are ``A^-1 (y - b 1)``, which is the solution of the bordered system of
    :func:`cmfqsar.model.krr_fit`.
    """
    from cmfqsar.model import _cv_splits

    y_tr = np.asarray(y_tr, float)
    lams = [float(lam) for lam in lams]
    squares = np.zeros(len(lams))
    for te, tr in _cv_splits(y_tr, inner, 0):
        vals, Q = np.linalg.eigh(K_tr[np.ix_(tr, tr)])
        qy, q1 = Q.T @ y_tr[tr], Q.T @ np.ones(len(tr))
        cross = K_tr[np.ix_(te, tr)] @ Q
        for k, lam in enumerate(lams):
            d = vals + lam
            d = np.where(np.abs(d) > 1e-300, d, 1e-300)
            ay, a1 = qy / d, q1 / d
            b = float(q1 @ ay) / float(q1 @ a1)
            p = cross @ (ay - b * a1) + b
            squares[k] += float(np.sum((p - y_tr[te]) ** 2))
    errs = np.sqrt(squares / len(y_tr))
    k = int(np.argmin(errs))
    return float(errs[k]), lams[k]


def choose_by_refit(K_tr: np.ndarray, y_tr: np.ndarray, lams=LAMS, inner: int = INNER):
    """The same choice as :func:`choose`, with one fit per ridge and inner fold.  It is the
    reference that :func:`choose` is tested against."""
    best = None
    for lam in lams:
        p = cv_predict(K_tr, y_tr, "krr", {"lam": lam}, folds=inner, seed=0)
        err = float(rmse(y_tr, p))
        if best is None or err < best[0]:
            best = (err, float(lam))
    return best


def summary(y: np.ndarray, pred: np.ndarray, folds, per_fold: List[dict]) -> dict:
    maes = np.array([float(np.mean(np.abs(pred[te] - y[te]))) for _, te in folds])
    return {"mae_mean": float(maes.mean()), "mae_sd": float(maes.std(ddof=0)),
            "mae_folds": [float(m) for m in maes],
            "rmse": float(rmse(y, pred)), "r2": float(r2_score(y, pred)),
            "per_fold": per_fold, "pred": [float(v) for v in pred]}


def nested(view, y: np.ndarray, folds, lams=LAMS, inner: int = INNER) -> dict:
    """Out-of-fold predictions of one description, its setting and the ridge chosen inside
    every training part."""
    pred = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in folds:
        best = None
        for s in view.settings:
            K_tr, _ = view.pair(s, tr, tr[:0])
            err, lam = choose(K_tr, y[tr], lams, inner)
            if best is None or err < best[0]:
                best = (err, s, lam)
        err, s, lam = best
        K_tr, K_te = view.pair(s, tr, te)
        pred[te] = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr]).predict(K_te)
        per_fold.append({"setting": s, "lam": lam, "inner_rmse": err})
    return summary(y, pred, folds, per_fold)


def mixed(views: Dict[str, object], singles: Dict[str, dict], y: np.ndarray, folds,
          mode: str = "evidence", lams=LAMS, inner: int = INNER) -> dict:
    """Out-of-fold predictions of several descriptions joined in one kernel.

    Every description enters with the setting it chose alone in the same training part.  With
    `mode` "evidence" the weights maximise the marginal likelihood of the training part
    (:func:`cmfqsar.mixing.evidence_mixing`); with "equal" every description has the same
    weight.  The ridge is chosen inside the training part.
    """
    names = list(views)
    pred = np.full(len(y), np.nan)
    per_fold = []
    for k, (tr, te) in enumerate(folds):
        pairs = [views[nm].pair(singles[nm]["per_fold"][k]["setting"], tr, te) for nm in names]
        stack, _, scale = with_views(pairs[0][0][None],
                                     {nm: pr[0] for nm, pr in zip(names[1:], pairs[1:])},
                                     names=names[:1])
        h = (evidence_mixing(stack, y[tr], scale=scale) if mode == "evidence"
             else np.ones(len(names)))
        K_tr = joint_kernel(stack, h, scale)
        K_te = joint_kernel(np.stack([pr[1] for pr in pairs]), h, scale)
        err, lam = choose(K_tr, y[tr], lams, inner)
        pred[te] = _Learner("krr", {"lam": lam}, True).fit(K_tr, y[tr]).predict(K_te)
        # the scale of a kernel is one over its mean diagonal, so the share of a description
        # in the mean diagonal of the joined kernel is its weight over the sum of the weights
        w = np.asarray(h, float)
        per_fold.append({"lam": lam, "inner_rmse": err,
                         "shares": {nm: float(v) for nm, v in zip(names, w / w.sum())}})
    return summary(y, pred, folds, per_fold)


def constant(y: np.ndarray, folds) -> dict:
    """The mean of the training part as the prediction."""
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        pred[te] = float(np.mean(y[tr]))
    return summary(y, pred, folds, [{} for _ in folds])


def regression(X: np.ndarray, y: np.ndarray, folds) -> dict:
    """A linear regression on the columns of `X` (least squares with a constant), fitted on
    every training part."""
    X = np.asarray(X, float).reshape(len(y), -1)
    A = np.hstack([np.ones((len(y), 1)), X])
    pred = np.full(len(y), np.nan)
    for tr, te in folds:
        coef, *_ = np.linalg.lstsq(A[tr], y[tr], rcond=None)
        pred[te] = A[te] @ coef
    return summary(y, pred, folds, [{} for _ in folds])


def mae_difference(y: np.ndarray, a: np.ndarray, b: np.ndarray, runs: int = RUNS,
                   seed: int = 0) -> Tuple[float, float, float]:
    """The mean absolute error of `a` minus that of `b`, with a 95 % bootstrap interval over
    the structures.  `a` and `b` are predictions, one row per repeat or one row in all.  Below
    zero, `a` has the lower error."""
    y = np.asarray(y, float)
    ea = np.abs(np.atleast_2d(np.asarray(a, float)) - y).mean(axis=0)
    eb = np.abs(np.atleast_2d(np.asarray(b, float)) - y).mean(axis=0)
    d = ea - eb
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (runs, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def kernel_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """The correlation between the off-diagonal elements of two kernels, each centred first.
    Near 1, the two descriptions find the same pairs of samples alike."""
    n = len(a)
    h = np.eye(n) - 1.0 / n
    upper = np.triu_indices(n, 1)
    return float(np.corrcoef((h @ a @ h)[upper], (h @ b @ h)[upper])[0, 1])


def target_alignment(K: np.ndarray, y: np.ndarray) -> float:
    """The correlation between the centred kernel and the products of the centred targets,
    over the pairs of samples: how far alike samples have alike targets."""
    n = len(y)
    h = np.eye(n) - 1.0 / n
    t = np.asarray(y, float) - float(np.mean(y))
    upper = np.triu_indices(n, 1)
    return float(np.corrcoef((h @ K @ h)[upper], np.outer(t, t)[upper])[0, 1])


def fixed_ridge_loss(stack: np.ndarray, y: np.ndarray, names: Sequence[str],
                     scale: np.ndarray, lams=LAM_GRID) -> Dict[str, float]:
    """What :func:`cmfqsar.mixing.drop_one` gave until 10 November 2026: the loss of log
    marginal likelihood when one kernel is taken away, with every marginal likelihood taken at
    the best of a few fixed ridges.  It is kept as the record of a fault: when the best ridge
    is below the list, a sum of more kernels has a larger scale, is nearer to the best ridge
    and gains from that alone."""
    def best(on):
        h = evidence_mixing(stack, y, scale, on)
        K = joint_kernel(stack, h, scale)
        return max(log_evidence(K, y, float(lam)) for lam in lams)

    everything = np.ones(len(names), bool)
    full = best(everything)
    out = {}
    for k, name in enumerate(names):
        keep = everything.copy()
        keep[k] = False
        out[name] = float(full - best(keep))
    return out


def evidence_part(K_structure: np.ndarray, K_pattern: np.ndarray, y: np.ndarray,
                  fixed: Optional[Dict[str, float]] = None) -> dict:
    """The structural kernel and the kernel of the patterns joined on all crystals.

    The weights maximise the marginal likelihood (:func:`cmfqsar.mixing.evidence_mixing`).
    ``drop_one`` is the loss of log marginal likelihood when one of the two kernels is taken
    away (:func:`cmfqsar.mixing.drop_one`): the log Bayes factor against *this kernel adds
    nothing*.  ``log_evidence`` holds the three log marginal likelihoods, each at its best
    ridge.  ``drop_one_fixed_ridges`` is the same loss on five fixed ridges
    (:func:`fixed_ridge_loss`); a value that is given is kept and not computed again.
    """
    stack, names, scale = with_views(K_structure[None], {"xray": K_pattern},
                                     names=["structure"])
    h = np.asarray(evidence_mixing(stack, y, scale=scale), float)
    if fixed is None:
        fixed = fixed_ridge_loss(stack, y, names, scale)
    both = best_ridge_evidence(joint_kernel(stack, h, scale), y)
    alone = {"structure": best_ridge_evidence(K_structure, y),
             "xray": best_ridge_evidence(K_pattern, y)}
    return {"shares": {a: float(v) for a, v in zip(names, h / h.sum())},
            "log_evidence": {"both": both[0], "structure": alone["structure"][0],
                             "xray": alone["xray"][0]},
            "ridge": {"both": both[1], "structure": alone["structure"][1],
                      "xray": alone["xray"][1]},
            "drop_one": {a: float(v) for a, v in
                         drop_one(stack, y, names, scale=scale).items()},
            "drop_one_fixed_ridges": {a: float(v) for a, v in fixed.items()},
            "fixed_ridges": [float(v) for v in LAM_GRID]}


def most_chosen(row: dict, key: str = "setting"):
    values = [f[key] for f in row["per_fold"]]
    return max(set(values), key=values.count)


# --------------------------------------------------------------------------- #
# Stage: fit
# --------------------------------------------------------------------------- #
ROWS = (
    ("mean", "mean of the training part"),
    ("lightest", "linear regression on the inverse square root of the mass of the lightest "
                 "atom"),
    ("composition", "composition: the fractions of the elements"),
    ("structure", "structure: the periodic autocorrelation kernel"),
    ("pair", "pair distribution function G(r) to 10 A, X-ray weights (plan item M12)"),
    ("xray", "X-ray powder pattern"),
    ("neutron", "neutron powder pattern"),
    ("xray, gauss", "X-ray powder pattern, Gaussian kernel on top"),
    ("neutron, gauss", "neutron powder pattern, Gaussian kernel on top"),
    ("xray+neutron", "X-ray and neutron powder patterns"),
    ("composition+xray", "composition and X-ray powder pattern"),
    ("structure+xray", "structure and X-ray powder pattern"),
    ("structure+xray, equal", "structure and X-ray powder pattern, equal weights"),
    ("structure+xray+neutron", "structure, X-ray and neutron powder patterns"),
)
LABEL = dict(ROWS)
MIXES = {
    "xray+neutron": (("xray", "neutron"), "evidence"),
    "composition+xray": (("composition", "xray"), "evidence"),
    "structure+xray": (("structure", "xray"), "evidence"),
    "structure+xray, equal": (("structure", "xray"), "equal"),
    "structure+xray+neutron": (("structure", "xray", "neutron"), "evidence"),
}
#: the comparisons the report gives with an interval: (row, reference)
COMPARE = (
    ("xray", "structure"), ("xray", "composition"), ("xray", "mean"), ("xray", "pair"),
    ("neutron", "xray"), ("neutron", "composition"),
    ("xray, gauss", "xray"), ("xray, gauss", "composition"),
    ("neutron, gauss", "neutron"), ("neutron, gauss", "composition"),
    ("xray+neutron", "xray"),
    ("composition+xray", "composition"),
    ("structure+xray", "structure"), ("structure+xray, equal", "structure"),
    ("structure+xray+neutron", "structure"),
    ("structure", "composition"),
)


def build_views(atoms, names, force: bool = False) -> Dict[str, object]:
    """Every description of the comparison, with its kernels."""
    t0 = time.time()
    views: Dict[str, object] = {}
    views["structure"], K = structural_view(atoms, names, force)
    log(f"  structural features and pair distribution functions, {time.time() - t0:.0f} s")
    views["composition"] = Composition(element_fractions(atoms)[0])
    if K is not None:
        views["pair"] = Fixed({"G(r)": K})
    for source in SOURCES:
        t0 = time.time()
        spectra = patterns(atoms, names, source, force)
        views[source] = Fixed(pattern_kernels(spectra))
        views[source].spectra = spectra
        views[source + ", gauss"] = Layered(views[source])
        log(f"  kernels of the {SOURCE_LABEL[source]}, {time.time() - t0:.0f} s")
    return views


def full_kernel(view, setting: str, n: int) -> np.ndarray:
    everything = np.arange(n)
    return view.pair(setting, everything, everything[:0])[0]


def fit(repeats: int = 1, force: bool = False, results: str = RESULTS) -> dict:
    path = result("fit.json", results)
    if os.path.isfile(path) and not force:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    atoms, y, names = load()
    n = len(y)
    log(f"{n} crystals, target {y.min():.0f} to {y.max():.0f} cm-1 (median "
        f"{np.median(y):.0f}, standard deviation {y.std():.0f})")
    views = build_views(atoms, names)
    light = lightest_atom(atoms)
    out = {"n": n, "repeats": repeats, "wavelength": LAMBDA, "two_theta": list(TWO_THETA),
           "sigmas": list(SIGMAS), "powers": list(POWERS), "lams": list(LAMS),
           "structure_variant": STRUCTURE_VARIANT, "y": [float(v) for v in y],
           "names": list(names), "target_sd": float(y.std()),
           "reflections": {s: [len(sp) for sp in views[s].spectra] for s in SOURCES},
           "hydrogen": [bool("H" in a.get_chemical_symbols()) for a in atoms],
           "rows": {}}
    t0 = time.time()
    for r in range(repeats):
        folds = folds_for(n, r)
        rows: Dict[str, dict] = {"mean": constant(y, folds),
                                 "lightest": regression(light, y, folds)}
        for name in ("composition", "structure", "pair", "xray", "neutron", "xray, gauss",
                     "neutron, gauss"):
            if name in views:
                rows[name] = nested(views[name], y, folds)
                log(f"  repeat {r}: {name}: MAE {rows[name]['mae_mean']:.1f} +/- "
                    f"{rows[name]['mae_sd']:.1f} cm-1  ({time.time() - t0:.0f} s)")
        for name, (parts, mode) in MIXES.items():
            rows[name] = mixed({p: views[p] for p in parts}, rows, y, folds, mode)
            log(f"  repeat {r}: {name}: MAE {rows[name]['mae_mean']:.1f} +/- "
                f"{rows[name]['mae_sd']:.1f} cm-1  ({time.time() - t0:.0f} s)")
        for name, row in rows.items():
            out["rows"].setdefault(name, []).append(row)
    # differences, over all repeats
    pred = {name: np.array([row["pred"] for row in rws]) for name, rws in out["rows"].items()}
    out["differences"] = []
    for a, b in COMPARE:
        if a in pred and b in pred:
            d, lo, hi = mae_difference(y, pred[a], pred[b])
            out["differences"].append({"row": a, "reference": b, "difference": d,
                                       "low": lo, "high": hi})
    # how alike the kernels are, on the whole set at the setting chosen most often
    chosen = {name: most_chosen(out["rows"][name][0])
              for name in ("composition", "structure", "pair", "xray", "neutron")
              if name in out["rows"]}
    full = {name: full_kernel(views[name], s, n) for name, s in chosen.items()}
    out["chosen"] = chosen
    out["alignment"] = {name: target_alignment(K, y) for name, K in full.items()}
    out["agreement"] = {a: {b: kernel_agreement(full[a], full[b]) for b in full} for a in full}
    out["agreement_by_width"] = [
        {"sigma": s, "power": p,
         "structure": kernel_agreement(views["xray"].kernels[setting_name(s, p)],
                                       full["structure"]),
         "composition": kernel_agreement(views["xray"].kernels[setting_name(s, p)],
                                         full["composition"]),
         "alignment": target_alignment(views["xray"].kernels[setting_name(s, p)], y)}
        for s in SIGMAS for p in POWERS]
    # does the marginal likelihood want the pattern beside the structure, on the whole set
    t1 = time.time()
    out["evidence"] = evidence_part(full["structure"], full["xray"], y)
    log(f"  evidence on the whole set, {time.time() - t1:.0f} s")
    out["elapsed_s"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def evidence(results: str = RESULTS) -> dict:
    """The evidence part of a finished fit, computed again and written back into its file.
    The rows of the fit are not touched."""
    path = result("fit.json", results)
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    atoms, y, names = load()
    views = {"structure": structural_view(atoms, names)[0],
             "xray": Fixed(pattern_kernels(patterns(atoms, names, "xray"),
                                           [setting_values(rec["chosen"]["xray"])[0]],
                                           [setting_values(rec["chosen"]["xray"])[1]]))}
    full = {name: full_kernel(views[name], rec["chosen"][name], len(y)) for name in views}
    old = rec.get("evidence", {})
    rec["evidence"] = evidence_part(full["structure"], full["xray"], y,
                                    fixed=old.get("drop_one_fixed_ridges"))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# --------------------------------------------------------------------------- #
# Stage: the co-spectrum
# --------------------------------------------------------------------------- #
def cospectrum(force: bool = False, results: str = RESULTS, scrambles: int = SCRAMBLES) -> dict:
    """The weight over 2-theta of the model fitted on all X-ray patterns, with the setting the
    folds chose most often, and the same reading on targets in random order."""
    path = result("cospectrum.json", results)
    if os.path.isfile(path) and not force:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    fitted = fit(results=results)
    atoms, y, names = load()
    spectra = patterns(atoms, names, "xray")
    row = fitted["rows"]["xray"][0]
    setting = most_chosen(row)
    sigma, power = setting_values(setting)
    lams = [f["lam"] for f in row["per_fold"] if f["setting"] == setting]
    lam = max(set(lams), key=lams.count)
    grid, F = pattern_features(spectra, sigma, power)
    K = SP.spectrum_kernel(F)

    def effect_of(target):
        alpha = _Learner("krr", {"lam": lam}, True).fit(K, target).alpha_
        w = SP.co_spectrum(F, alpha)
        return w, SP.co_spectrum_effect(F, w)

    w, effect = effect_of(y)
    rng = np.random.default_rng(0)
    null = np.array([np.abs(effect_of(rng.permutation(y))[1]).max() for _ in range(scrambles)])
    # the readable form: weight times intensity summed over bands of 2-theta
    edges = np.append(np.arange(TWO_THETA[0], TWO_THETA[1] - 1e-9, BAND), TWO_THETA[1])
    x = grid.x
    Z = F / np.sqrt(np.maximum(np.einsum("ij,ij->i", F, F), 1e-300))[:, None]
    bands = []
    for a, b in zip(edges[:-1], edges[1:]):
        cols = (x >= a) & ((x < b) | (b >= TWO_THETA[1]))
        share = Z[:, cols] @ w[cols]          # what the band adds to each prediction
        inten = np.sqrt(np.einsum("ij,ij->i", Z[:, cols], Z[:, cols]))
        bands.append({"from": float(a), "to": float(b),
                      "d_from": float(SP.two_theta_to_d(b, LAMBDA)),
                      "d_to": float(SP.two_theta_to_d(a, LAMBDA)),
                      "points": int(cols.sum()),
                      "sd": float(share.std()),
                      "r_added": float(np.corrcoef(share, y)[0, 1]),
                      "r_with_target": float(np.corrcoef(inten, y)[0, 1])})
    keep = max(1, len(x) // 800)
    out = {"setting": setting, "sigma": sigma, "power": power, "lam": lam,
           "grid_step": float(grid.step), "grid_points": int(len(x)),
           "fitted_r2": float(r2_score(y, _Learner("krr", {"lam": lam}, True).fit(K, y)
                                       .predict(K))),
           "x": [float(v) for v in x[::keep]],
           "weight": [float(v) for v in w[::keep]],
           "effect": [float(v) for v in effect[::keep]],
           "mean_pattern": [float(v) for v in Z.mean(axis=0)[::keep]],
           "largest_effect": float(np.abs(effect).max()),
           "null_largest_effect": [float(v) for v in null],
           "strongest": [dict(r, d=float(SP.two_theta_to_d(r["x"], LAMBDA)))
                         for r in SP.strongest_regions(x, effect, n=8, separation=2.0)],
           "bands": bands}
    os.makedirs(results, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    figure(out, results)
    return out


def figure(rec: dict, results: str = RESULTS) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.array(rec["x"])
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 7.5), sharex=True)
    axes[0].plot(x, rec["mean_pattern"], color="0.25", lw=0.8)
    axes[0].set_ylabel("mean normalised\nintensity")
    axes[0].set_title("X-ray powder patterns of 1,265 crystals: the mean pattern, the "
                      "co-spectrum and its effect", fontsize=10)
    axes[1].plot(x, rec["weight"], color="tab:blue", lw=0.8)
    axes[1].axhline(0.0, color="0.6", lw=0.6)
    axes[1].set_ylabel("co-spectrum\n(weight)")
    axes[2].plot(x, rec["effect"], color="tab:red", lw=0.8)
    axes[2].axhline(0.0, color="0.6", lw=0.6)
    null = float(np.max(rec["null_largest_effect"])) if rec["null_largest_effect"] else 0.0
    for sign in (-1.0, 1.0):
        axes[2].axhline(sign * null, color="0.5", lw=0.6, ls="--")
    axes[2].set_ylabel("weight x spread\nof the intensity, cm$^{-1}$")
    axes[2].set_xlabel("diffraction angle 2-theta, degrees (Cu K-alpha)")
    top = axes[0].secondary_xaxis(
        "top", functions=(lambda t: SP.two_theta_to_d(np.clip(t, 1.0, 179.0), LAMBDA),
                          lambda d: np.degrees(2 * np.arcsin(np.clip(LAMBDA / (2 * np.maximum(
                              d, LAMBDA / 2 + 1e-9)), 0, 1)))))
    top.set_xticks([10.0, 5.0, 3.0, 2.0, 1.5, 1.2])
    top.set_xlabel("spacing of the lattice planes d, A")
    fig.tight_layout()
    fig.savefig(result("cospectrum.png", results), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #
def _mean_row(rws: List[dict]) -> Tuple[float, float]:
    """The MAE over the folds of the Matbench division, and the mean over all repeats."""
    return rws[0]["mae_mean"], float(np.mean([r["mae_mean"] for r in rws]))


SHORT = {"structure": "structure", "composition": "composition", "pair": "G(r)",
         "xray": "X-ray pattern", "neutron": "neutron pattern"}


def shown(setting) -> str:
    """A setting in words.  The keys of the settings use the character that separates the
    columns of a Markdown table, so they cannot be written into a table as they are."""
    s = str(setting)
    if "|" not in s:
        return s
    gamma = None
    if "|g" in s:
        s, gamma = s.rsplit("|g", 1)
    sigma, power = s.split("|")
    out = f"{sigma} deg, power {power}"
    return out + (f", factor {gamma}" if gamma is not None else "")


def _settings(rws: List[dict]) -> str:
    row = rws[0]
    if "setting" not in row["per_fold"][0]:
        if "shares" in row["per_fold"][0]:
            names = list(row["per_fold"][0]["shares"])
            return "shares " + ", ".join(
                f"{nm} {np.mean([f['shares'][nm] for f in row['per_fold']]):.2f}"
                for nm in names)
        return ""
    chosen = [shown(f["setting"]) for f in row["per_fold"]]
    if len(set(chosen)) == 1:
        return chosen[0] + " in every fold"
    return "; ".join(chosen)


def report(results: str = RESULTS) -> str:
    """The table of the benchmark as text; it is also written to ``table.md``."""
    path = result("fit.json", results)
    if not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    rows = rec["rows"]
    repeats = rec.get("repeats", 1)
    lines = ["# A spectrum as a one-dimensional field: a calculated powder pattern as the "
             "sample (plan item SP1)", "",
             f"The Matbench phonons task: {rec['n']:,} inorganic crystals with the frequency of "
             "the highest peak of the phonon density of states (cm^-1; standard deviation "
             f"{rec['target_sd']:.0f} cm^-1).  The powder diffraction pattern of every crystal "
             "is calculated from its structure with pymatgen, for X-rays (Cu K-alpha, "
             f"{rec['wavelength']:.4f} A) and for neutrons of the same wavelength, between "
             f"{rec['two_theta'][0]:g} and {rec['two_theta'][1]:g} degrees of the diffraction "
             "angle 2-theta.  No measured pattern is used.", "",
             "A pattern is a stick spectrum: one position and one intensity for every "
             "reflection.  The kernel of two patterns is the overlap of the two after every "
             "stick is replaced by a Gaussian (`cmfqsar/spectrum.py`), divided by the norms of "
             "the two.  The width of the Gaussian (one of "
             + ", ".join(f"{s:g}" for s in rec["sigmas"]) + " degrees), the power of the "
             "intensity (1, or 0.5 for the square root) and the ridge are chosen by 10-fold "
             "cross-validation inside every training part.", "",
             f"The crystals have {int(np.median(rec['reflections']['xray']))} X-ray reflections "
             f"at the median ({min(rec['reflections']['xray'])} to "
             f"{max(rec['reflections']['xray'])}).", ""]
    # ---- how alike the kernels are
    ag = rec["agreement"]
    names = [n for n in ("structure", "composition", "pair", "xray", "neutron") if n in ag]
    lines += ["## How alike the kernels are", "",
              "The correlation between the off-diagonal elements of two kernels, each centred "
              "first, on all crystals and at the setting that the folds chose most often.  "
              "Near 1, two descriptions find the same pairs of crystals alike, and the second "
              "cannot add much to the first.  The last column is the same correlation with the "
              "products of the centred targets: how far crystals that a description finds alike "
              "have alike frequencies.", "",
              "| description | setting | " + " | ".join(SHORT[n] for n in names)
              + " | with the target |",
              "|---|---|" + "---:|" * (len(names) + 1)]
    for a in names:
        lines.append(f"| {LABEL[a]} | {shown(rec['chosen'][a])} | "
                     + " | ".join("1" if a == b else f"{ag[a][b]:+.3f}" for b in names)
                     + f" | {rec['alignment'][a]:+.3f} |")
    lines += ["", "The kernel of the X-ray patterns by the width of the broadening and the "
              "power of the intensity:", "",
              "| width, degrees | power of the intensity | correlation with the structural "
              "kernel | correlation with the composition kernel | correlation with the target |",
              "|---:|---:|---:|---:|---:|"]
    for r in rec["agreement_by_width"]:
        lines.append(f"| {r['sigma']:g} | {r['power']:g} | {r['structure']:+.3f} | "
                     f"{r['composition']:+.3f} | {r['alignment']:+.3f} |")
    # ---- the errors
    head = ("| description | MAE, cm^-1 (sd over the five folds) | RMSE, cm^-1 | R2 | "
            "settings chosen in the five folds |")
    if repeats > 1:
        head = head[:-1] + f" MAE, mean of {repeats} divisions into folds |"
    lines += ["", "## The error of every description, on the five Matbench folds", "",
              "MAE is the mean absolute error on the held-out folds.  RMSE and R2 are computed "
              "on the pooled held-out predictions.  A setting of a pattern is its width in "
              "degrees and the power of the intensity.  With a Gaussian kernel on top it also "
              "has a factor: the Gaussian kernel is exp(-factor D^2 / m), with D the distance "
              "of two normalised patterns and m the median of D^2 over the training part.  For "
              "joined kernels the column gives the mean share of every kernel over the folds.",
              "", head, "|---|---:|---:|---:|---|" + ("---:|" if repeats > 1
                                                                   else "")]
    for name, label in ROWS:
        if name not in rows:
            continue
        r0 = rows[name][0]
        line = (f"| {label} | {r0['mae_mean']:.1f} ({r0['mae_sd']:.1f}) | {r0['rmse']:.1f} | "
                f"{r0['r2']:+.3f} | {_settings(rows[name])} |")
        if repeats > 1:
            line += f" {_mean_row(rows[name])[1]:.1f} |"
        lines.append(line)
    lines += ["", "Published errors on the same folds (Matbench v0.1, MAE in cm^-1, read from "
              "the Matbench site in October 2026): the mean as the prediction 324.0; "
              "RF-SCM/Magpie 67.6; CrabNet, which uses the composition only, 55.1; MODNet 34.3; "
              "MEGNet 28.8.", ""]
    # ---- with and without hydrogen
    hyd = np.array(rec["hydrogen"], bool)
    y = np.array(rec["y"])
    lines += ["## The error of the crystals with and without hydrogen", "",
              f"{int(hyd.sum())} of the {rec['n']:,} crystals hold hydrogen.  Their median "
              f"frequency is {np.median(y[hyd]):.0f} cm^-1, and the median of the others is "
              f"{np.median(y[~hyd]):.0f} cm^-1: a bond to hydrogen vibrates at a higher "
              "frequency than any other bond.  A diffraction pattern shows hydrogen weakly, so "
              "the table scores the same held-out predictions for the two groups.  Nothing is "
              "fitted again.", "",
              "| description | MAE of the crystals with hydrogen, cm^-1 | MAE of the crystals "
              "without hydrogen, cm^-1 |", "|---|---:|---:|"]
    for name, label in ROWS:
        if name in rows:
            e = np.abs(np.array(rows[name][0]["pred"]) - y)
            lines.append(f"| {label} | {e[hyd].mean():.1f} | {e[~hyd].mean():.1f} |")
    lines.append("")
    # ---- the differences
    lines += ["## Differences in the mean absolute error", "",
              "The MAE of the first description minus the MAE of the second, with a 95 % "
              "bootstrap interval over the crystals"
              + (f" (the errors of a crystal are averaged over the {repeats} divisions into "
                 "folds first)" if repeats > 1 else "")
              + ".  Below zero, the first description has the lower error.", "",
              "| description | compared with | difference in MAE, cm^-1 | is zero inside the "
              "interval |", "|---|---|---:|---|"]
    for d in rec["differences"]:
        inside = "yes" if d["low"] <= 0.0 <= d["high"] else "no"
        lines.append(f"| {LABEL[d['row']]} | {LABEL[d['reference']]} | {d['difference']:+.1f} "
                     f"[{d['low']:+.1f}, {d['high']:+.1f}] | {inside} |")
    ev = rec["evidence"]
    le = ev["log_evidence"]
    lines += ["", "## The pattern as one more kernel beside the structure", "",
              "The structural kernel and the kernel of the X-ray patterns are joined with "
              "weights that maximise the marginal likelihood of the targets "
              "(`cmfqsar/mixing.py`).  On all crystals the share of the structural kernel is "
              f"{ev['shares']['structure']:.2f} and the share of the pattern is "
              f"{ev['shares']['xray']:.2f}.  The log marginal likelihood, each at its best "
              f"ridge, is {le['both']:.1f} for the joined kernel, {le['structure']:.1f} for "
              f"the structural kernel alone and {le['xray']:.1f} for the pattern alone.  The "
              "log Bayes factor against *the pattern adds nothing* (`mixing.drop_one`: the "
              "loss of log marginal likelihood when the pattern is taken away) is "
              f"{ev['drop_one']['xray']:+.1f}, and against *the structure adds nothing* it is "
              f"{ev['drop_one']['structure']:+.1f}.  The pattern adds nothing to the marginal "
              "likelihood either.", "",
              "The same two numbers with every marginal likelihood taken at the best of five "
              "fixed ridges (" + ", ".join(f"{v:g}" for v in ev["fixed_ridges"]) + "), as "
              "`mixing.drop_one` computed them until 10 November 2026, are "
              f"{ev['drop_one_fixed_ridges']['xray']:+.1f} and "
              f"{ev['drop_one_fixed_ridges']['structure']:+.1f}.  The first of them is not "
              "evidence for the pattern.  The best ridge of the structural kernel is "
              f"{ev['ridge']['structure']:.1e}, below the lowest of the five.  The joined "
              "kernel has twice the scale of one kernel, which acts as half the ridge, so the "
              "joined kernel was nearer to the best ridge and gained from that alone.  This "
              "run found the fault, and `drop_one` now searches the ridge relative to the "
              "scale of the kernel.", ""]
    cpath = result("cospectrum.json", results)
    if os.path.isfile(cpath):
        with open(cpath, encoding="utf-8") as fh:
            co = json.load(fh)
        null = np.array(co["null_largest_effect"])
        lines += ["## The co-spectrum of the X-ray patterns", "",
                  "The model on all crystals with the setting that the folds chose most often "
                  f"(width {co['sigma']:g} degrees, power {co['power']:g}, ridge "
                  f"{co['lam']:g}; {co['grid_points']} grid points, {co['grid_step']:g} "
                  "degrees apart).  Its prediction is a constant plus the sum over the axis of "
                  "a weight times the normalised intensity.  The weight is the co-spectrum "
                  "(`cospectrum.png`).  The effect of a grid point is its weight times the "
                  "standard deviation of the intensity there over the set.", "",
                  f"The largest effect of a grid point is {co['largest_effect']:.1f} cm^-1.  "
                  f"In {len(null)} models fitted to the targets in random order the largest "
                  f"effect is {null.mean():.1f} cm^-1 at the mean and {null.max():.1f} at "
                  f"most.  The model reproduces the targets it was fitted on with R2 "
                  f"{co['fitted_r2']:+.3f}.", "",
                  f"The table sums weight times intensity over bands of {BAND:g} degrees.  The "
                  "second number column is the standard deviation of that sum over the "
                  "crystals: how much the band moves the predictions.  The third is the "
                  "correlation of that sum with the target.  The last is the correlation of "
                  "the intensity in the band with the target, which does not use the model.",
                  "",
                  "| 2-theta, degrees | spacing of the lattice planes d, A | standard deviation "
                  "of what the band adds to the predictions, cm^-1 | correlation of what the "
                  "band adds with the target | correlation of the intensity in the band with "
                  "the target |",
                  "|---|---|---:|---:|---:|"]
        for b in co["bands"]:
            lines.append(f"| {b['from']:g} to {b['to']:g} | {b['d_to']:.2f} to "
                         f"{b['d_from']:.2f} | {b['sd']:.1f} | {b['r_added']:+.2f} | "
                         f"{b['r_with_target']:+.2f} |")
        lines.append("")
    lines += _verdict(rec, co if os.path.isfile(cpath) else None)
    text = "\n".join(lines)
    os.makedirs(results, exist_ok=True)
    with open(result("table.md", results), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    return text


def _verdict(rec: dict, co: Optional[dict] = None) -> List[str]:
    """The reading of the numbers, written from the numbers."""
    rows = rec["rows"]
    diff = {(d["row"], d["reference"]): d for d in rec["differences"]}
    y = np.array(rec["y"])
    hyd = np.array(rec["hydrogen"], bool)

    def mae(name):
        return rows[name][0]["mae_mean"]

    def group(name, mask):
        return float(np.abs(np.array(rows[name][0]["pred"]) - y)[mask].mean())

    def phrase(a, b):
        d = diff[(a, b)]
        side = ("not separated" if d["low"] <= 0.0 <= d["high"]
                else "lower" if d["high"] < 0 else "higher")
        return f"{d['difference']:+.1f} [{d['low']:+.1f}, {d['high']:+.1f}] cm^-1 ({side})"

    ag = rec["agreement"]
    best_width = max(rec["agreement_by_width"], key=lambda r: r["alignment"])
    out = ["## Reading", "",
           f"**The kernels.** The kernel of the X-ray patterns correlates with the structural "
           f"kernel with r {ag['xray']['structure']:+.3f} and with the composition kernel with "
           f"r {ag['xray']['composition']:+.3f}, so it is a different description.  Its "
           f"correlation with the target is {rec['alignment']['xray']:+.3f}, and at no width "
           f"is it above {best_width['alignment']:+.3f}.  The structural kernel has "
           f"{rec['alignment']['structure']:+.3f} and the composition kernel "
           f"{rec['alignment']['composition']:+.3f}.  A kernel that does not find crystals "
           "with alike frequencies alike cannot predict the frequency, and the errors below "
           "agree with that.", "",
           f"**The pattern alone.** The X-ray pattern predicts the frequency with a mean "
           f"absolute error of {mae('xray'):.1f} cm^-1.  The mean of the training part gives "
           f"{mae('mean'):.1f}, the regression on the mass of the lightest atom "
           f"{mae('lightest'):.1f}, the composition {mae('composition'):.1f} and the structural "
           f"kernel {mae('structure'):.1f}.  Against the structural kernel the pattern is "
           f"{phrase('xray', 'structure')}, and against the composition "
           f"{phrase('xray', 'composition')}.  The neutron pattern gives "
           f"{mae('neutron'):.1f} ({phrase('neutron', 'xray')} against the X-ray pattern)."]
    if "pair" in rows:
        out[-1] += (f"  The pair distribution function of plan item M12, which holds the same "
                    f"untyped distances in real space, gives {mae('pair'):.1f} (the X-ray "
                    f"pattern against it: {phrase('xray', 'pair')}).")
    widths = {setting_values(f["setting"])[0] for f in rows["xray"][0]["per_fold"]}
    if widths == {max(rec["sigmas"])}:
        out[-1] += (f"  In every fold the choice inside the folds takes the widest broadening "
                    f"tried, {max(rec['sigmas']):g} degrees.  A pattern broadened that far "
                    "holds only how the intensity is spread over the angle, so the places of "
                    "the single reflections give the model nothing that it can use.")
    if "xray, gauss" in rows:
        out[-1] += (f"  A Gaussian kernel on top lowers the error to {mae('xray, gauss'):.1f} "
                    f"for the X-ray pattern and to {mae('neutron, gauss'):.1f} for the neutron "
                    f"pattern.  The neutron pattern is then still "
                    f"{phrase('neutron, gauss', 'composition')} against the composition.")
    out += ["",
            f"**Why the pattern fails.** The crystals with hydrogen have the highest "
            f"frequencies, and X-rays show hydrogen weakly.  But hydrogen is not the whole "
            f"reason.  For the crystals without hydrogen the X-ray pattern has an error of "
            f"{group('xray', ~hyd):.1f} cm^-1, the mean of the training part "
            f"{group('mean', ~hyd):.1f} and the composition {group('composition', ~hyd):.1f}.  "
            f"For the crystals with hydrogen the three numbers are {group('xray', hyd):.1f}, "
            f"{group('mean', hyd):.1f} and {group('composition', hyd):.1f}.  The frequency of "
            "the highest phonon peak depends on which elements the crystal holds, most of all "
            "on their masses.  A powder pattern holds the geometry of the lattice and the "
            "contrast of the scattering strengths, and it does not name the elements.", "",
            f"**The pattern beside another description.** With the structural kernel the X-ray "
            f"pattern gives {mae('structure+xray'):.1f} cm^-1, which is "
            f"{phrase('structure+xray', 'structure')} against the structural kernel alone "
            f"(with equal weights: {phrase('structure+xray, equal', 'structure')}).  With the "
            f"composition it gives {mae('composition+xray'):.1f}, which is "
            f"{phrase('composition+xray', 'composition')} against the composition alone.  The "
            f"weights that maximise the marginal likelihood give the pattern a share of "
            f"{rec['evidence']['shares']['xray']:.2f} beside the structure.", "",
            f"**The structure against the composition.** The structural kernel is "
            f"{phrase('structure', 'composition')} against a Gaussian kernel on the fractions "
            "of the elements.  On this task the composition carries most of what the "
            "structural kernel predicts.", ""]
    if co is not None:
        null = max(co["null_largest_effect"])
        strongest = max(abs(b["r_with_target"]) for b in co["bands"])
        out += [f"**The co-spectrum.** The model on all X-ray patterns reproduces the targets "
                f"it was fitted on with R2 {co['fitted_r2']:+.3f}.  Its largest effect is "
                f"{co['largest_effect']:.0f} cm^-1, against at most {null:.0f} in models "
                f"fitted to targets in random order, and the intensity of no band of "
                f"{BAND:g} degrees correlates with the target above {strongest:.2f} in size.  "
                "The weights of neighbouring grid points have opposite signs and cancel, "
                "because at this width neighbouring points hold nearly the same intensity.  "
                "The co-spectrum of a model that predicts this little cannot be read as a map.",
                ""]
    gate_structure = diff[("structure+xray", "structure")]["high"] < 0
    gate_composition = diff[("xray", "composition")]["high"] < 0 or (
        diff[("xray", "composition")]["low"] <= 0.0 <= diff[("xray", "composition")]["high"])
    out += ["**The gate of the item.** The item is a result when the structural kernel with "
            "the pattern has a lower error than the structural kernel alone by more than the "
            "interval, or when the pattern alone reaches the composition.  The first "
            f"condition is {'met' if gate_structure else 'not met'}.  The second condition is "
            f"{'met' if gate_composition else 'not met'}."]
    if not gate_structure and not gate_composition:
        out[-1] += ("  This is the outcome that the plan expected, and it is stronger than "
                    "expected: a calculated powder pattern, without the types of the atoms, "
                    "predicts this property little better than the mean does.  The kernel and "
                    "the co-spectrum of `cmfqsar/spectrum.py` are tested on their own; this "
                    "property is not one that a diffraction pattern shows.")
    out.append("")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all",
                    choices=["all", "patterns", "fit", "evidence", "cospectrum"])
    ap.add_argument("--repeats", type=int, default=1,
                    help="divisions into five folds; the first is the Matbench division")
    ap.add_argument("--force", action="store_true", help="compute again what has a result")
    ap.add_argument("--report", action="store_true", help="only write the table")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    if not a.report:
        if a.stage == "patterns":
            atoms, _, names = load()
            for source in SOURCES:
                patterns(atoms, names, source, force=True)
            return 0
        if a.stage in ("all", "fit"):
            fit(a.repeats, force=a.force)
        if a.stage == "evidence":
            evidence()
        if a.stage in ("all", "cospectrum"):
            cospectrum(force=a.force)
    text = report()
    if not text:
        print("no results yet: run the script without --report first")
        return 1
    print(f"Report written to {result('table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
