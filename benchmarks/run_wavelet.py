"""
A multi-scale Mexican-hat kernel against the Gaussian one: the kernel-correlation screen.

A CMF molecular field is a sum of atom-centred Gaussians, and the CMF kernel is the overlap of
two such fields.  A wavelet analysis of that field is possible without any grid: the Mexican-hat
wavelet is the negative Laplacian of a Gaussian, so the wavelet coefficients of a field at scale
``s`` are ``s^2`` times the Laplacian of the same field built with every Gaussian widened by the
smoothing ``s``.  The overlap of two coefficient maps at one scale is then the Laplacian applied
twice (the biharmonic operator) to the ordinary Gaussian overlap, taken with respect to the
separation of the two atoms.  For two atoms with widened widths ``b_l'`` and ``b_m'`` and

    c = b_l' b_m' / (2 (b_l' + b_m'))

the pair integral is the ordinary overlap of the widened Gaussians multiplied by

    s^4 * 4 c^2 * (15 - 20 c d^2 + 4 c^2 d^4).

This script asks one question before anything is built on this kernel: **is it a different
description at all?**  Plan item EC5 found that a kernel-matrix correlation of +0.98 with the
existing kernel predicted that no model built on the new kernel would differ.  The same
correlation is measured here, on the eight Sutherland series, between the cosine-normalised
Gaussian kernel and the cosine-normalised Mexican-hat kernel at four dyadic scales (0.5, 1, 2 and
4 A of smoothing) and their equal-weight sum.  Two readings of agreement are reported:

* the Pearson correlation of the off-diagonal entries of the two cosine kernels (as EC5 did);
* the centred kernel alignment, which compares the two matrices after removing the mean row,
  so that a shared constant offset of all the similarities does not count as agreement.

Because a Mexican-hat kernel at a given scale may order the molecules like a Gaussian kernel of
some *other* width, the correlation is also taken against every width of the usual grid, and the
highest value is reported next to the one at the default width.

As a second, cheap reading the script fits kernel ridge regression under 5-fold cross-validation
repeated three times, with the Gaussian width and the ridge chosen on the training folds by the
log evidence, as ``run_perfield.py`` does.  Three models are compared: the Gaussian kernel, the
multi-scale Mexican-hat kernel, and their sum.

    python benchmarks/run_wavelet.py --stage check
    python benchmarks/run_wavelet.py --stage screen [--sets ACE BZR ...]
    python benchmarks/run_wavelet.py --stage cv [--sets ACE BZR ...]
    python benchmarks/run_wavelet.py --report
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from rdkit import RDLogger                                                  # noqa: E402

from cmfqsar.dataio import load_dataset                                     # noqa: E402
from cmfqsar.evidence import log_evidence                                   # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields                # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels              # noqa: E402
from cmfqsar.model import krr_fit, r2_score, rmse                           # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "wavelet")
DATA = os.path.join(ROOT, "examples", "sutherland")
SETS = ("ACE", "AChE", "BZR", "COX2", "DHFR", "GPB", "THER", "THR")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: the width of the field whose wavelet transform is taken (the package default)
ALPHA0 = 1.0
#: the smoothing scales of the wavelet, in A, one octave apart
SCALES = (0.5, 1.0, 2.0, 4.0)
#: the Gaussian widths the Mexican-hat kernel is compared against
ALPHAS = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS = 5
REPEATS = 3
SEED = 0


# --------------------------------------------------------------------------- #
# The kernel
# --------------------------------------------------------------------------- #

def widened(b: np.ndarray, s: float) -> np.ndarray:
    """The width of a Gaussian ``exp(-b/2 r^2)`` after convolution with a unit-mass Gaussian of
    standard deviation ``s``: ``1 / (1/b + s^2)``."""
    return 1.0 / (1.0 / b + s * s)


def wavelet_pair_factor(bi: np.ndarray, bq: np.ndarray, s: float, d2: np.ndarray) -> np.ndarray:
    """The pair integral of two Mexican-hat coefficient maps at scale ``s``.

    ``bi`` (Ni,) and ``bq`` (Nq,) are the *unsmoothed* widths of the atoms, ``d2`` (Ni, Nq) the
    squared separations.  Returned is the integral over space of ``W_s g_i * W_s g_q`` with
    ``W_s g = s^2 Laplacian(G_s * g)`` for unit-weight atoms; the atomic weights multiply it.
    """
    bi2 = widened(bi, s)[:, None]
    bq2 = widened(bq, s)[None, :]
    amp = (bi2 / bi[:, None]) ** 1.5 * (bq2 / bq[None, :]) ** 1.5      # the convolution's mass
    bs = bi2 + bq2
    c = bi2 * bq2 / (2.0 * bs)
    overlap = amp * (2.0 * np.pi / bs) ** 1.5 * np.exp(-c * d2)
    poly = 4.0 * c * c * (15.0 - 20.0 * c * d2 + 4.0 * c * c * d2 * d2)
    return s ** 4 * overlap * poly


def atom_widths(table: AtomTable, specs, alpha0: float) -> np.ndarray:
    """(A, F) width of every atom's Gaussian in every field of the set."""
    A = len(table.coords)
    B = np.empty((A, len(specs)))
    for f, spec in enumerate(specs):
        if spec.kind == "steric":
            B[:, f] = alpha0 / table.radii ** 2
        else:
            B[:, f] = alpha0
    return B


def wavelet_kernels(table: AtomTable, specs, alpha0: float = ALPHA0,
                    scales: Sequence[float] = SCALES) -> np.ndarray:
    """Raw (not normalised) Mexican-hat kernels, shape (S, F, n, n)."""
    n = table.n_mol
    F = len(specs)
    B = atom_widths(table, specs, alpha0)
    W = table.weights
    starts = table.offsets[:-1]
    out = np.zeros((len(scales), F, n, n))
    shared = [f for f, s in enumerate(specs) if s.kind != "steric"]
    for i in range(n):
        sl = table.slice(i)
        D2 = cdist(table.coords[sl], table.coords, "sqeuclidean")
        for k, s in enumerate(scales):
            if shared:
                f0 = shared[0]
                E = wavelet_pair_factor(B[sl, f0], B[:, f0], s, D2)
                for f in shared:
                    M = (W[sl, f][:, None] * E) * W[:, f][None, :]
                    out[k, f, i] = np.add.reduceat(M, starts, axis=1).sum(axis=0)
            for f, spec in enumerate(specs):
                if spec.kind != "steric":
                    continue
                E = wavelet_pair_factor(B[sl, f], B[:, f], s, D2)
                M = (W[sl, f][:, None] * E) * W[:, f][None, :]
                out[k, f, i] = np.add.reduceat(M, starts, axis=1).sum(axis=0)
    return out


def cosine(K: np.ndarray) -> np.ndarray:
    """Cosine-normalised kernels for an array whose last two axes are molecules."""
    d = np.einsum("...ii->...i", K)
    return normalize_kernels(K.reshape(-1, *K.shape[-2:]),
                             d.reshape(-1, K.shape[-1]), d.reshape(-1, K.shape[-1])).reshape(K.shape)


# --------------------------------------------------------------------------- #
# Agreement of two kernels
# --------------------------------------------------------------------------- #

def pearson_offdiag(K: np.ndarray, L: np.ndarray, good: Optional[np.ndarray] = None) -> float:
    if good is not None:
        K = K[np.ix_(good, good)]
        L = L[np.ix_(good, good)]
    iu = np.triu_indices_from(K, 1)
    a, b = K[iu], L[iu]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def cka(K: np.ndarray, L: np.ndarray, good: Optional[np.ndarray] = None) -> float:
    """Centred kernel alignment (Cortes et al. 2012): the cosine of the two centred matrices."""
    if good is not None:
        K = K[np.ix_(good, good)]
        L = L[np.ix_(good, good)]
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    denom = np.linalg.norm(Kc) * np.linalg.norm(Lc)
    return float((Kc * Lc).sum() / denom) if denom > 1e-300 else float("nan")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_set(name: str, data: str = DATA):
    ds = load_dataset(os.path.join(data, f"{name}.sdf"))
    y = np.asarray(ds.activity, float)
    keep = np.isfinite(y)
    mols = [m for m, k in zip(ds.mols, keep) if k]
    names = [n for n, k in zip(ds.names, keep) if k]
    fs = FieldSet.create(physchem=FIELDS, alpha=ALPHA0, charge_model="file")
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    return table, fs, y[keep]


def gaussian_kernels(table: AtomTable, fs: FieldSet, alphas=ALPHAS) -> Dict[float, np.ndarray]:
    """alpha -> cosine-normalised (F, n, n) Gaussian kernels."""
    out = {}
    for a in alphas:
        specs = [dataclasses.replace(s, alpha=float(a)) for s in fs.specs]
        out[float(a)] = cosine(kernel_matrices(table, table, specs))
    return out


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def check_stage() -> Dict[str, object]:
    """The closed form against a grid integration, and the convolution identity it rests on."""
    out: Dict[str, object] = {}
    h = 0.1
    L = 12.0
    ax = np.arange(-L, L, h)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")

    def gauss(center, b):
        r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        return np.exp(-0.5 * b * r2)

    def mexican(center, b, s):
        """s^2 Laplacian of (G_s * g) in closed form: the Laplacian of the widened Gaussian."""
        b2 = widened(b, s)
        r2 = (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        return s * s * (b2 / b) ** 1.5 * (b2 * b2 * r2 - 3.0 * b2) * np.exp(-0.5 * b2 * r2)

    # 1. the convolution identity, by FFT on the grid
    b = 1.3
    s = 1.0
    g = gauss((0.0, 0.0, 0.0), b)
    r2 = X ** 2 + Y ** 2 + Z ** 2
    G = np.exp(-0.5 * r2 / (s * s)) / (2 * np.pi * s * s) ** 1.5 * h ** 3
    conv = np.real(np.fft.ifftn(np.fft.fftn(g) * np.fft.fftn(np.fft.ifftshift(G))))
    b2 = widened(b, s)
    exact = (b2 / b) ** 1.5 * gauss((0.0, 0.0, 0.0), b2)
    out["convolution_relative_error"] = float(np.abs(conv - exact).max() / exact.max())
    print(f"    convolution with a Gaussian of s = {s} A widens b = {b} to {b2:.4f}: "
          f"largest relative error on the grid {out['convolution_relative_error']:.2e}", flush=True)

    # 2. the pair integral against quadrature, two atoms of unequal width, several separations
    worst = 0.0
    for s in (0.5, 1.0, 2.0):
        for d in (0.0, 1.0, 2.5, 4.0):
            bi, bq = 1.0, 0.6
            f1 = mexican((0.0, 0.0, 0.0), bi, s)
            f2 = mexican((d, 0.0, 0.0), bq, s)
            num = float((f1 * f2).sum() * h ** 3)
            cf = float(wavelet_pair_factor(np.array([bi]), np.array([bq]), s,
                                           np.array([[d * d]]))[0, 0])
            rel = abs(num - cf) / max(abs(cf), 1e-12)
            worst = max(worst, rel)
    out["pair_integral_relative_error"] = float(worst)
    print(f"    the closed-form pair integral against a {len(ax)}^3 grid: worst relative error "
          f"{worst:.2e} over 12 (scale, separation) pairs", flush=True)

    # 3. positive semidefiniteness and the sign changes, on one real series
    table, fs, _y = load_set("GPB")
    Kw = wavelet_kernels(table, fs.specs)
    mn = min(float(np.linalg.eigvalsh(Kw[k, f]).min() / np.abs(np.linalg.eigvalsh(Kw[k, f])).max())
             for k in range(len(SCALES)) for f in range(len(FIELDS))
             if np.abs(Kw[k, f]).max() > 0)
    out["smallest_relative_eigenvalue_gpb"] = mn
    print(f"    GPB: the smallest eigenvalue relative to the largest, over scales and fields, is "
          f"{mn:+.1e} (zero up to round-off means positive semidefinite)", flush=True)
    return out


def screen_set(name: str, force: bool = False) -> Dict[str, object]:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}_screen.json")
    if os.path.exists(path) and not force:
        print(f"{name}: screen already done")
        return json.load(open(path))
    t0 = time.time()
    table, fs, y = load_set(name)
    n = table.n_mol
    print(f"=== {name}: {n} molecules, {len(table.coords)} atoms", flush=True)
    Kg = gaussian_kernels(table, fs)
    Kw_raw = wavelet_kernels(table, fs.specs)
    Kw = cosine(Kw_raw)                                   # (S, F, n, n)
    Kw_sum = Kw.mean(axis=0)                              # (F, n, n), equal scale weights
    F = len(FIELDS)

    def joint(Kf):
        return Kf.sum(axis=0)

    out: Dict[str, object] = {"set": name, "n": int(n), "fields": {}, "joint": {}}
    labels = [f"s={s}" for s in SCALES] + ["sum"]
    cands = [Kw[k] for k in range(len(SCALES))] + [Kw_sum]
    for f, field in enumerate(FIELDS):
        diag_ok = np.array([np.diag(Kw_raw[0, f]) > 1e-12]).ravel() & \
            np.array([np.diag(Kg[ALPHA0][f]) != 0]).ravel()
        good = np.where(diag_ok)[0]
        rec = {"n_used": int(len(good))}
        for lab, Kc in zip(labels, cands):
            r_default = pearson_offdiag(Kg[ALPHA0][f], Kc[f], good)
            a_default = cka(Kg[ALPHA0][f], Kc[f], good)
            r_grid = {str(a): pearson_offdiag(Kg[a][f], Kc[f], good) for a in Kg}
            a_grid = {str(a): cka(Kg[a][f], Kc[f], good) for a in Kg}
            best_r = max(r_grid, key=lambda k: (r_grid[k] if np.isfinite(r_grid[k]) else -9))
            best_a = max(a_grid, key=lambda k: (a_grid[k] if np.isfinite(a_grid[k]) else -9))
            rec[lab] = {"pearson_alpha1": r_default, "cka_alpha1": a_default,
                        "pearson_best": r_grid[best_r], "pearson_best_alpha": float(best_r),
                        "cka_best": a_grid[best_a], "cka_best_alpha": float(best_a)}
        out["fields"][field] = rec
        print(f"    {field:14s} sum of scales: r = {rec['sum']['pearson_alpha1']:+.3f} against "
              f"alpha = 1, best r = {rec['sum']['pearson_best']:+.3f} at alpha = "
              f"{rec['sum']['pearson_best_alpha']}; CKA = {rec['sum']['cka_alpha1']:.3f} / "
              f"{rec['sum']['cka_best']:.3f}", flush=True)
    # the joint kernel the model sees
    Jg = {a: joint(Kg[a]) for a in Kg}
    for lab, Kc in zip(labels, cands):
        J = joint(Kc)
        r_grid = {str(a): pearson_offdiag(Jg[a], J) for a in Jg}
        a_grid = {str(a): cka(Jg[a], J) for a in Jg}
        best_r = max(r_grid, key=r_grid.get)
        best_a = max(a_grid, key=a_grid.get)
        out["joint"][lab] = {"pearson_alpha1": r_grid[str(ALPHA0)], "cka_alpha1": a_grid[str(ALPHA0)],
                             "pearson_best": r_grid[best_r], "pearson_best_alpha": float(best_r),
                             "cka_best": a_grid[best_a], "cka_best_alpha": float(best_a)}
    j = out["joint"]["sum"]
    print(f"    joint kernel, sum of scales: r = {j['pearson_alpha1']:+.3f} against alpha = 1, "
          f"best r = {j['pearson_best']:+.3f} at alpha = {j['pearson_best_alpha']}; "
          f"CKA = {j['cka_alpha1']:.3f} / {j['cka_best']:.3f}", flush=True)
    for lab in labels[:-1]:
        j = out["joint"][lab]
        print(f"      {lab:7s}: r = {j['pearson_alpha1']:+.3f}, best r = {j['pearson_best']:+.3f} "
              f"at alpha = {j['pearson_best_alpha']}, CKA best = {j['cka_best']:.3f}", flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"), indent=1)
    return out


def folds(n: int, k: int, seed: int):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return [np.sort(perm[i::k]) for i in range(k)]


def _select(cands: Dict[str, np.ndarray], y: np.ndarray, tr: np.ndarray):
    """The candidate kernel and ridge with the highest log evidence on the training block."""
    best = None
    for key, K in cands.items():
        Ktr = K[np.ix_(tr, tr)]
        for lam in LAMS:
            e = log_evidence(Ktr, y[tr], float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, key, float(lam))
    return best[1], best[2]


def cv_set(name: str, force: bool = False) -> Dict[str, object]:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{name}_cv.json")
    if os.path.exists(path) and not force:
        print(f"{name}: cv already done")
        return json.load(open(path))
    t0 = time.time()
    table, fs, y = load_set(name)
    n = table.n_mol
    print(f"=== {name}: {n} molecules", flush=True)
    Kg = {a: K.sum(axis=0) for a, K in gaussian_kernels(table, fs).items()}
    Kw = cosine(wavelet_kernels(table, fs.specs)).mean(axis=0).sum(axis=0)
    models = {
        "gaussian": {str(a): Kg[a] for a in Kg},
        "wavelet": {"sum": Kw},
        "gaussian+wavelet": {str(a): Kg[a] + Kw for a in Kg},
    }
    preds = {m: np.zeros((REPEATS, n)) for m in models}
    chosen = {m: [] for m in models}
    for rep in range(REPEATS):
        for te in folds(n, FOLDS, SEED + rep):
            tr = np.setdiff1d(np.arange(n), te)
            for m, cands in models.items():
                key, lam = _select(cands, y, tr)
                K = cands[key]
                a, b = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
                preds[m][rep, te] = K[np.ix_(te, tr)] @ a + b
                chosen[m].append((key, lam))
    out: Dict[str, object] = {"set": name, "n": int(n), "models": {}}
    for m in models:
        r2s = [float(r2_score(y, preds[m][r])) for r in range(REPEATS)]
        rm = [float(rmse(y, preds[m][r])) for r in range(REPEATS)]
        out["models"][m] = {"r2_mean": float(np.mean(r2s)), "r2_sd": float(np.std(r2s)),
                            "rmse_mean": float(np.mean(rm)),
                            "chosen": sorted({f"{k}/{l}" for k, l in chosen[m]})}
        print(f"    {m:17s} q2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}, "
              f"rmse = {np.mean(rm):.3f}", flush=True)
    # paired bootstrap over molecules of the q2 difference, on repeat-averaged predictions
    rng = np.random.RandomState(SEED)
    pg = preds["gaussian"].mean(axis=0)
    for m in ("wavelet", "gaussian+wavelet"):
        pm = preds[m].mean(axis=0)
        diffs = []
        for _ in range(2000):
            idx = rng.randint(0, n, n)
            diffs.append(r2_score(y[idx], pm[idx]) - r2_score(y[idx], pg[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        out["models"][m]["q2_gain_vs_gaussian"] = {
            "mean": float(r2_score(y, pm) - r2_score(y, pg)), "lo": float(lo), "hi": float(hi)}
        print(f"    {m} - gaussian: {out['models'][m]['q2_gain_vs_gaussian']['mean']:+.3f} "
              f"[{lo:+.3f}, {hi:+.3f}]", flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(path, "w"), indent=1)
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def write_table(path: Optional[str] = None) -> str:
    lines: List[str] = []
    lines.append("# The multi-scale Mexican-hat kernel against the Gaussian kernel\n")
    lines.append("Pearson r is over the off-diagonal entries of the two cosine kernels; CKA is the "
                 "centred kernel alignment. 'at alpha 1' compares with the Gaussian kernel of the "
                 "default width; 'best' is the highest value over the width grid "
                 f"{ALPHAS}, with that width in brackets.\n")
    lines.append("## Joint kernel of the five fields, sum of the four scales\n")
    lines.append("| Set | n | r at alpha 1 | best r (alpha) | CKA at alpha 1 | best CKA (alpha) |")
    lines.append("|---|---|---|---|---|---|")
    for s in SETS:
        p = os.path.join(RESULTS, f"{s}_screen.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        j = d["joint"]["sum"]
        lines.append(f"| {s} | {d['n']} | {j['pearson_alpha1']:+.3f} | {j['pearson_best']:+.3f} "
                     f"({j['pearson_best_alpha']}) | {j['cka_alpha1']:.3f} | {j['cka_best']:.3f} "
                     f"({j['cka_best_alpha']}) |")
    lines.append("\n## Joint kernel, one scale at a time (best r over the width grid)\n")
    lines.append("| Set | " + " | ".join(f"s = {s} A" for s in SCALES) + " |")
    lines.append("|---|" + "---|" * len(SCALES))
    for s in SETS:
        p = os.path.join(RESULTS, f"{s}_screen.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cells = [f"{d['joint'][f's={sc}']['pearson_best']:+.3f} "
                 f"({d['joint'][f's={sc}']['pearson_best_alpha']})" for sc in SCALES]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    lines.append("\n## Per field, sum of scales (best r over the width grid)\n")
    lines.append("| Set | " + " | ".join(FIELDS) + " |")
    lines.append("|---|" + "---|" * len(FIELDS))
    for s in SETS:
        p = os.path.join(RESULTS, f"{s}_screen.json")
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        cells = []
        for f in FIELDS:
            r = d["fields"][f]["sum"]
            cells.append(f"{r['pearson_best']:+.3f} ({r['pearson_best_alpha']})")
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    cvs = [s for s in SETS if os.path.exists(os.path.join(RESULTS, f"{s}_cv.json"))]
    if cvs:
        lines.append(f"\n## Kernel ridge regression, {FOLDS}-fold cross-validation repeated "
                     f"{REPEATS} times\n")
        lines.append("q2 is the cross-validated R2 of the pooled predictions, mean over the repeats. "
                     "The gain is the paired bootstrap interval over molecules of the q2 difference "
                     "against the Gaussian kernel.\n")
        lines.append("| Set | Gaussian q2 | Mexican hat q2 | gain | Gaussian + Mexican hat q2 | gain |")
        lines.append("|---|---|---|---|---|---|")
        for s in cvs:
            d = json.load(open(os.path.join(RESULTS, f"{s}_cv.json")))
            m = d["models"]
            gw = m["wavelet"]["q2_gain_vs_gaussian"]
            gb = m["gaussian+wavelet"]["q2_gain_vs_gaussian"]
            lines.append(f"| {s} | {m['gaussian']['r2_mean']:.3f} | {m['wavelet']['r2_mean']:.3f} | "
                         f"{gw['mean']:+.3f} [{gw['lo']:+.3f}, {gw['hi']:+.3f}] | "
                         f"{m['gaussian+wavelet']['r2_mean']:.3f} | "
                         f"{gb['mean']:+.3f} [{gb['lo']:+.3f}, {gb['hi']:+.3f}] |")
    text = "\n".join(lines) + "\n"
    path = path or os.path.join(RESULTS, "wavelet_table.md")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("check", "screen", "cv"), default=None)
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.stage == "check":
        os.makedirs(RESULTS, exist_ok=True)
        out = check_stage()
        json.dump(out, open(os.path.join(RESULTS, "check.json"), "w"), indent=1)
    elif args.stage == "screen":
        for s in args.sets:
            screen_set(s, force=args.force)
    elif args.stage == "cv":
        for s in args.sets:
            cv_set(s, force=args.force)
    if args.report or args.stage is None:
        print(write_table())


if __name__ == "__main__":
    main()
