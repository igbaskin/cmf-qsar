#!/usr/bin/env python
"""
A microstructure as the sample: field-overlap kernels on a voxel grid against two-point statistics.

A two-phase microstructure on a periodic voxel grid is a field: the indicator of each phase is
one channel, one value per voxel. The kernel of this program -- the overlap integral of two
fields -- therefore applies to it without an atom in sight, and the question of this script is
whether it is worth applying. The comparison is the homogenisation workflow of the Materials
Knowledge Systems (MKS, Kalidindi and co-workers): two-point statistics, principal components,
a polynomial regression. Everything here is computed with FFTs on the grid; the atom-centred
Gaussian engine of ``cmfqsar`` is not used, and the fitting code of the package is.

**The data.** Synthetic microstructures, the recipe of ``pymks.generate_multiphase``: white noise
filtered with a Gaussian whose width is the correlation length (the grain size), then cut at the
quantile that gives the phase fraction. The grains are elongated by a random aspect ratio in a
random direction, so that a direction-dependent property exists. The phase fraction, the grain
size, the aspect ratio and the direction all vary from sample to sample. The label is the
effective thermal conductivity of the periodic cell, from a finite-volume solver written here
(``pymks`` 0.4.1 does not build on Python 3.13, and its elastic solver needs ``sfepy``): the
steady conduction equation with a unit mean temperature gradient applied across the periodic
cell, harmonic-mean conductivities on the faces, one sparse solve per direction. The solver is
checked against the exact results for a uniform cell and for the two laminates (series and
parallel) every time the data are built. Two labels are reported: the component along x, which
depends on the direction of the grains, and the mean over the directions, which does not.

**The descriptions.** For every sample and every Gaussian smoothing width ``s`` (voxels):

* *overlap, generating frame* -- the plain kernel of this program: the overlap of the smoothed
  phase fields at zero shift, ``K_ij = sum_x sum_c m_c^i(x) m_c^j(x)``. The samples share a grid
  and nothing else; their origins are random. So this kernel compares two structures at a
  random relative shift, and what it keeps as the smoothing grows is the phase fractions.
* *overlap, rotation-averaged* -- the same, averaged over the point group of the grid (the eight
  operations of the square, the 48 of the cube).
* *autocorrelation* -- the translation-averaged route: the two-point autocorrelation
  ``S_ab(r) = (1/n) sum_x m_a(x) m_b(x + r)`` of every sample (the self-overlap of the field with
  its own translate, which is the two-point statistic of MKS), and the kernel is the overlap of
  two samples' autocorrelations over ``r``. This is the overlap of their power spectra. It is
  reported raw (*linear*), cosine-normalised with one scale per channel block (the package's
  ``autocorrelation_kernel``, *normalised*), averaged over the point group of the grid
  (*lattice-rotation-averaged*), and binned by the length of ``r`` (*radial*, the average over
  every rotation).
* *two-point statistics + PCA* -- the MKS baseline: the autocorrelation and cross-correlation
  of the raw indicator, principal components fitted on the training part, a polynomial of the
  scores of degree one or two, fitted by ridge regression.
* *phase fraction* -- a cubic polynomial of the fraction of phase 1 alone: what the structure
  must add to.

On every kernel a linear model and a Gaussian layer (bandwidth relative to the median distance)
are tried; the smoothing, the layer, the ridge and, for the baseline, the number of components
and the degree are chosen by five-fold cross-validation inside the training part.

**The protocol.** Training sizes 20, 50, 100 and 200 drawn at random from the set, five draws
each, the rest of the set held out; q2 is R2 on the held-out samples and RMSE is in the units
of the label (the conductivity of phase 0 is one). The mean and the spread over the five draws
are reported.

Usage
-----
    python benchmarks/run_microstructure.py                 # 2D 51x51 at contrast 10 and 100,
                                                            # 3D 21^3 at contrast 10, then the table
    python benchmarks/run_microstructure.py --dim 2 --grid 51 --n 400 --contrast 10
    python benchmarks/run_microstructure.py --stage data     # build (or check) the data only
    python benchmarks/run_microstructure.py --stage fit
    python benchmarks/run_microstructure.py --report
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.invariant import autocorrelation_kernel, channel_scale  # noqa: E402
from cmfqsar.model import _Learner, cv_predict, median_gamma_scale, r2_score, rmse  # noqa: E402

CACHE = os.path.join(HERE, "cache", "microstructure")
RESULTS = os.path.join(HERE, "results", "microstructure")

#: the data sets built by default: (dimension, grid points per side, samples, contrast)
CONTRAST = 10.0                 # conductivity of phase 1 over that of phase 0
DEFAULT_SETS: Tuple[Tuple[int, int, int, float], ...] = ((2, 51, 400, CONTRAST),
                                                        (2, 51, 400, 100.0),
                                                        (3, 21, 400, CONTRAST))
FRACTION = (0.2, 0.8)           # phase fraction of phase 1, uniform
LENGTH = (1.5, 8.0)             # correlation length in voxels, log-uniform
ASPECT = (1.0, 3.0)             # aspect ratio of the grains, log-uniform
SMOOTH: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 5.0)     # Gaussian smoothing widths, voxels
GAMMAS: Tuple[float, ...] = (0.5, 1.0, 2.0)               # Gaussian layer, times the median
LAMS: Tuple[float, ...] = tuple(10.0 ** -k for k in range(6, -1, -1))   # ridge, relative
SIZES: Tuple[int, ...] = (20, 50, 100, 200)
REPEATS = 5
INNER = 5
COMPONENTS: Tuple[int, ...] = (2, 3, 5, 10, 20)
DEGREES: Tuple[int, ...] = (1, 2)
RIDGES: Tuple[float, ...] = (1e-8, 1e-4, 1e-2, 1.0)
LABELS = ("xx", "mean")
METHODS = ("phase fraction", "two-point statistics + PCA",
           "overlap, generating frame", "overlap, rotation-averaged",
           "autocorrelation, linear", "autocorrelation, normalised",
           "autocorrelation, lattice-rotation-averaged", "autocorrelation, radial")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


def tag_of(dim: int, grid: int, n: int, contrast: float) -> str:
    return f"{dim}d{grid}_n{n}_c{contrast:g}"


# --------------------------------------------------------------------------- #
# The data: a generator and a solver
# --------------------------------------------------------------------------- #

def _wavevectors(dim: int, grid: int) -> List[np.ndarray]:
    k = 2.0 * np.pi * np.fft.fftfreq(grid)
    return list(np.meshgrid(*([k] * dim), indexing="ij"))


def _rotation(dim: int, rng: np.random.Generator) -> np.ndarray:
    """A random rotation: an angle in 2D, a random orthogonal matrix in 3D."""
    if dim == 2:
        t = rng.uniform(0.0, np.pi)
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q


def generate(dim: int, grid: int, fraction: float, length: float, aspect: float,
             rotation: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One two-phase microstructure: 1 where phase 1 is, 0 elsewhere.

    Gaussian-filtered white noise cut at a quantile, the recipe of ``pymks``. The filter is
    anisotropic: the width is ``length * sqrt(aspect)`` along the first axis of the rotated
    frame and ``length / sqrt(aspect)`` along the others, so the geometric mean of the widths is
    the correlation length whatever the aspect ratio.
    """
    noise = rng.standard_normal((grid,) * dim)
    ks = np.stack(_wavevectors(dim, grid), axis=-1) @ rotation          # rotated wavevectors
    widths = np.full(dim, length / np.sqrt(aspect))
    widths[0] = length * np.sqrt(aspect)
    filt = np.exp(-0.5 * np.sum((ks * widths) ** 2, axis=-1))
    field = np.fft.ifftn(np.fft.fftn(noise) * filt).real
    cut = np.quantile(field, 1.0 - fraction)
    return (field > cut).astype(np.uint8)


def effective_conductivity(phase: np.ndarray, contrast: float = CONTRAST) -> np.ndarray:
    """The effective conductivity tensor of a periodic two-phase cell, phase 0 at one.

    Finite volumes on the voxel grid: the conductivity of a face is the harmonic mean of the two
    voxels it separates. A unit mean gradient is applied along one axis, the periodic
    fluctuation of the temperature solves the conservation equation (one voxel pinned, since the
    equation fixes the temperature up to a constant), and the mean flux is the column of the
    tensor. One sparse solve per axis.
    """
    dim = phase.ndim
    shape = phase.shape
    n = phase.size
    k = np.where(phase > 0, contrast, 1.0)
    idx = np.arange(n).reshape(shape)
    faces = []                                      # per axis: (i, j, k_face), j = i + e_axis
    for ax in range(dim):
        kj = np.roll(k, -1, axis=ax)
        kf = 2.0 * k * kj / (k + kj)
        faces.append((idx.ravel(), np.roll(idx, -1, axis=ax).ravel(), kf.ravel()))
    rows, cols, vals = [], [], []
    for i, j, kf in faces:
        rows += [i, j, i, j]
        cols += [i, j, j, i]
        vals += [-kf, -kf, kf, kf]
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n, n))
    pin = sp.csr_matrix(([1.0], ([0], [0])), shape=(n, n))
    A = A.tolil()
    A[0, :] = 0.0
    A = (A.tocsr() + pin).tocsc()
    solve = spla.factorized(A)
    tensor = np.zeros((dim, dim))
    for g in range(dim):                            # the applied gradient along axis g
        i, j, kf = faces[g]
        rhs = np.zeros(n)
        np.add.at(rhs, i, -kf)                      # sum over faces of k_face (T_j - T_i) = -k_face G
        np.add.at(rhs, j, kf)
        rhs[0] = 0.0
        T = solve(rhs)
        for a in range(dim):                        # the mean flux along axis a
            i, j, kf = faces[a]
            tensor[a, g] = np.mean(kf * ((T[j] - T[i]) + (1.0 if a == g else 0.0)))
    return tensor


def _solver_cases(dim, grid, contrast):
    half = grid // 2
    ones = np.ones((grid,) * dim, np.uint8)
    f = half / grid
    series = 1.0 / ((1 - f) / 1.0 + f / contrast)
    parallel = (1 - f) * 1.0 + f * contrast
    yield "uniform phase 1", ones, contrast * np.eye(dim)
    lam = np.zeros((grid,) * dim, np.uint8)
    lam[:half] = 1                                  # layers perpendicular to x
    exact = parallel * np.eye(dim)
    exact[0, 0] = series
    yield "laminate, layers across x", lam, exact


def check_solver_lines(dim: int, grid: int, contrast: float = CONTRAST) -> List[str]:
    """The exact cases -- a uniform cell, a laminate (series across the layers, parallel along
    them) -- and the symmetry of the tensor on a random structure. Fails loudly."""
    lines = []
    for name, phase, exact in _solver_cases(dim, grid, contrast):
        got = effective_conductivity(phase, contrast)
        err = float(np.max(np.abs(got - exact)))
        lines.append(f"{name}: largest error {err:.1e}")
        if err > 1e-8:
            raise SystemExit(f"the conductivity solver fails the check '{name}': {got} vs {exact}")
    rng = np.random.default_rng(12345)
    got = effective_conductivity(generate(dim, grid, 0.5, 3.0, 2.0, _rotation(dim, rng), rng),
                                 contrast)
    asym = float(np.max(np.abs(got - got.T)))
    lines.append(f"random structure, asymmetry of the tensor {asym:.1e}")
    if asym > 1e-8:
        raise SystemExit(f"the conductivity tensor is not symmetric: {got}")
    return lines


def build_data(dim: int, grid: int, n: int, contrast: float = CONTRAST, seed: int = 0,
               force: bool = False) -> dict:
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, tag_of(dim, grid, n, contrast) + ".npz")
    if os.path.isfile(path) and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    if grid % 2 == 0:
        raise SystemExit("the grid must have an odd number of points, so that a rotation "
                         "about the centre maps the grid onto itself")
    for line in check_solver_lines(dim, grid, contrast):
        log("  solver check: " + line)
    rng = np.random.default_rng(seed)
    phases = np.zeros((n,) + (grid,) * dim, np.uint8)
    params = np.zeros((n, 3))                       # fraction, length, aspect
    tensors = np.zeros((n, dim, dim))
    t0 = time.time()
    for i in range(n):
        fraction = rng.uniform(*FRACTION)
        length = np.exp(rng.uniform(np.log(LENGTH[0]), np.log(LENGTH[1])))
        aspect = np.exp(rng.uniform(np.log(ASPECT[0]), np.log(ASPECT[1])))
        rot = _rotation(dim, rng)
        phases[i] = generate(dim, grid, fraction, length, aspect, rot, rng)
        params[i] = (fraction, length, aspect)
        tensors[i] = effective_conductivity(phases[i], contrast)
        if (i + 1) % 100 == 0:
            log(f"  {i + 1} of {n} samples solved, {time.time() - t0:.0f} s")
    data = {"phase": phases, "params": params, "tensor": tensors,
            "contrast": np.array(contrast), "seed": np.array(seed)}
    np.savez_compressed(path, **data)
    log(f"  data written: {path}, {time.time() - t0:.0f} s")
    return data


def labels(data: dict) -> Dict[str, np.ndarray]:
    t = data["tensor"]
    dim = t.shape[1]
    return {"xx": t[:, 0, 0].copy(), "mean": np.trace(t, axis1=1, axis2=2) / dim}


# --------------------------------------------------------------------------- #
# The descriptions
# --------------------------------------------------------------------------- #

def smoothed_fields(phase: np.ndarray, s: float) -> np.ndarray:
    """The two phase indicators of every sample, smoothed with a periodic Gaussian of width s.

    Returns (n, 2, grid...) with channel 0 the indicator of phase 0 and channel 1 of phase 1.
    """
    m1 = phase.astype(float)
    fields = np.stack([1.0 - m1, m1], axis=1)
    if s <= 0:
        return fields
    dim = phase.ndim - 1
    ks = _wavevectors(dim, phase.shape[-1])
    filt = np.exp(-0.5 * s * s * sum(k * k for k in ks))
    axes = tuple(range(2, 2 + dim))
    return np.fft.ifftn(np.fft.fftn(fields, axes=axes) * filt, axes=axes).real


def point_group(dim: int) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """The operations of the square (8) or the cube (48) as (axis permutation, axes flipped)."""
    ops = []
    for perm in itertools.permutations(range(dim)):
        for flips in itertools.product((0, 1), repeat=dim):
            ops.append((perm, tuple(a for a in range(dim) if flips[a])))
    return ops


def apply_op(x: np.ndarray, op, first_axis: int) -> np.ndarray:
    """One point-group operation on the trailing grid axes of x, about the array centre."""
    perm, flips = op
    lead = tuple(range(first_axis))
    y = np.transpose(x, lead + tuple(first_axis + p for p in perm))
    for a in flips:
        y = np.flip(y, axis=first_axis + a)
    return np.ascontiguousarray(y)


def autocorrelations(fields: np.ndarray) -> np.ndarray:
    """S_11(r) and S_01(r) of every sample, r = 0 at index 0: (n, 2, grid...)."""
    dim = fields.ndim - 2
    axes = tuple(range(2, 2 + dim))
    n_vox = int(np.prod(fields.shape[2:]))
    F = np.fft.fftn(fields, axes=axes)
    S11 = np.fft.ifftn(F[:, 1] * np.conj(F[:, 1]), axes=tuple(a - 1 for a in axes)).real / n_vox
    S01 = np.fft.ifftn(F[:, 0] * np.conj(F[:, 1]), axes=tuple(a - 1 for a in axes)).real / n_vox
    return np.stack([S11, S01], axis=1)


def radial_profile(S: np.ndarray) -> np.ndarray:
    """The autocorrelation binned by the length of r (minimum image), weighted by sqrt(count).

    The linear kernel of the weighted profiles is the sum over the grid of the products of the
    rotation-averaged autocorrelations, which is what makes it the overlap of two
    rotation-averaged fields' statistics.
    """
    dim = S.ndim - 2
    grid = S.shape[-1]
    coords = np.meshgrid(*([np.fft.fftfreq(grid) * grid] * dim), indexing="ij")
    r = np.sqrt(sum(c * c for c in coords))
    shell = np.floor(r + 1e-9).astype(int)
    n_bins = int(shell.max()) + 1
    flat = S.reshape(S.shape[0], S.shape[1], -1)
    counts = np.bincount(shell.ravel(), minlength=n_bins).astype(float)
    out = np.zeros((S.shape[0], S.shape[1], n_bins))
    for c in range(S.shape[1]):
        for i in range(S.shape[0]):
            out[i, c] = np.bincount(shell.ravel(), weights=flat[i, c], minlength=n_bins)
    out /= np.maximum(counts, 1.0)
    return out * np.sqrt(counts)


def group_average(x: np.ndarray, dim: int, centred: bool) -> np.ndarray:
    """The mean of x over the point group of the grid; r = 0 is moved to the centre first when
    the array is an autocorrelation (``centred``), since the group acts about r = 0."""
    axes = tuple(range(x.ndim - dim, x.ndim))
    y = np.fft.fftshift(x, axes=axes) if centred else x
    acc = np.zeros_like(y)
    ops = point_group(dim)
    for op in ops:
        acc += apply_op(y, op, x.ndim - dim)
    acc /= len(ops)
    return np.fft.ifftshift(acc, axes=axes) if centred else acc


def kernels_for(data: dict, s: float) -> Dict[str, np.ndarray]:
    """Every kernel of the kernel methods at one smoothing width, and the block widths."""
    phase = data["phase"]
    n = phase.shape[0]
    dim = phase.ndim - 1
    fields = smoothed_fields(phase, s)
    X = fields.reshape(n, -1)
    K = {"overlap, generating frame": X @ X.T}
    acc = np.zeros((n, n))
    ops = point_group(dim)
    for op in ops:
        Xg = apply_op(fields, op, 2).reshape(n, -1)
        acc += X @ Xg.T
    K["overlap, rotation-averaged"] = 0.5 * (acc + acc.T) / len(ops)
    S = autocorrelations(fields)
    A = S.reshape(n, -1)
    K["autocorrelation, linear"] = A @ A.T
    blocks = [int(np.prod(S.shape[2:]))] * 2
    K["autocorrelation, normalised"] = autocorrelation_kernel(A, blocks=blocks)
    Sg = group_average(S, dim, centred=True).reshape(n, -1)
    K["autocorrelation, lattice-rotation-averaged"] = Sg @ Sg.T
    R = radial_profile(S).reshape(n, -1)
    K["autocorrelation, radial"] = R @ R.T
    return K


def two_point_features(data: dict) -> np.ndarray:
    """The MKS descriptor: S_11(r) and S_01(r) of the raw indicator, flattened."""
    fields = smoothed_fields(data["phase"], 0.0)
    return autocorrelations(fields).reshape(fields.shape[0], -1)


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #

def layered(K: np.ndarray, tr: np.ndarray) -> Dict[str, np.ndarray]:
    """The linear kernel and its Gaussian layers, each scaled to a mean self-kernel of one on
    the training part so that the ridge grid is relative."""
    out = {}
    d = float(np.mean(np.diag(K)[tr]))
    Kl = K / max(d, 1e-300)
    out["linear"] = Kl
    scale = median_gamma_scale(Kl[np.ix_(tr, tr)])
    diag = np.diag(Kl)
    d2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * Kl, 0.0)
    for g in GAMMAS:
        out[f"gauss {g:g}"] = np.exp(-g * scale * d2)
    return out


def fit_kernel_method(kernels: Dict[float, np.ndarray], y: np.ndarray, tr: np.ndarray,
                      te: np.ndarray) -> dict:
    """Nested choice of smoothing, layer and ridge by inner cross-validation on `tr`."""
    best = None
    for s, K in kernels.items():
        for layer, Kl in layered(K, tr).items():
            K_tr = Kl[np.ix_(tr, tr)]
            for lam in LAMS:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, s, layer, lam)
    err, s, layer, lam = best
    Kl = layered(kernels[s], tr)[layer]
    learner = _Learner("krr", {"lam": lam}, True).fit(Kl[np.ix_(tr, tr)], y[tr])
    pred = learner.predict(Kl[np.ix_(te, tr)])
    return {"setting": f"s {s:g}, {layer}, lam {lam:g}", "smoothing": s, "layer": layer,
            "lam": lam, "inner_rmse": float(err), "pred": pred}


def _poly(Z: np.ndarray, degree: int) -> np.ndarray:
    cols = [np.ones(len(Z))]
    for d in range(1, degree + 1):
        for comb in itertools.combinations_with_replacement(range(Z.shape[1]), d):
            cols.append(np.prod(Z[:, comb], axis=1))
    return np.stack(cols, axis=1)


def _ridge(P: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    reg = alpha * np.eye(P.shape[1])
    reg[0, 0] = 0.0                                  # the constant term is not shrunk
    return np.linalg.solve(P.T @ P + reg, P.T @ y)


class _PCA:
    """Principal components of the training part, computed once and shared by every setting."""

    def __init__(self, X_tr: np.ndarray):
        self.mean = X_tr.mean(axis=0)
        _, _, self.Vt = np.linalg.svd(X_tr - self.mean, full_matrices=False)
        self.Z_tr = (X_tr - self.mean) @ self.Vt.T

    def predict(self, y_tr: np.ndarray, X_te: np.ndarray, components: int, degree: int,
                alpha: float) -> np.ndarray:
        """A polynomial of the standardised scores on the first components, ridge regression."""
        c = min(components, self.Vt.shape[0])
        Z_tr = self.Z_tr[:, :c]
        Z_te = (X_te - self.mean) @ self.Vt[:c].T
        sd = np.maximum(Z_tr.std(axis=0), 1e-12)
        P_tr, P_te = _poly(Z_tr / sd, degree), _poly(Z_te / sd, degree)
        return P_te @ _ridge(P_tr, y_tr, alpha)


def _inner_folds(n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.random.RandomState(0).permutation(n)
    folds = [idx[k::INNER] for k in range(INNER)]
    return [(np.setdiff1d(idx, f), f) for f in folds]


def fit_pca_method(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    best = None
    folds = _inner_folds(len(tr))
    pcas = [(_PCA(X[tr[a]]), a, b) for a, b in folds]
    for components in COMPONENTS:
        if components >= len(tr) * (INNER - 1) // INNER:
            continue
        for degree in DEGREES:
            for alpha in RIDGES:
                p = np.full(len(tr), np.nan)
                for pca, a, b in pcas:
                    p[b] = pca.predict(y[tr[a]], X[tr[b]], components, degree, alpha)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, components, degree, alpha)
    err, components, degree, alpha = best
    pred = _PCA(X[tr]).predict(y[tr], X[te], components, degree, alpha)
    return {"setting": f"{components} components, degree {degree}, ridge {alpha:g}",
            "components": components, "degree": degree, "lam": alpha,
            "inner_rmse": float(err), "pred": pred}


def fit_fraction(f: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    P = _poly(f[:, None], 3)
    pred = P[te] @ _ridge(P[tr], y[tr], 1e-8)
    return {"setting": "cubic in the fraction", "pred": pred}


def evaluate(data: dict, kernels: Dict[float, Dict[str, np.ndarray]], X2p: np.ndarray,
             sizes=SIZES, repeats=REPEATS, label_names=LABELS) -> dict:
    ys = labels(data)
    n = len(data["phase"])
    fractions = data["phase"].reshape(n, -1).mean(axis=1)
    out = {}
    for label in label_names:
        y = ys[label]
        for size in sizes:
            if size >= n:
                continue
            for rep in range(repeats):
                perm = np.random.RandomState(1000 * size + rep).permutation(n)
                tr, te = perm[:size], perm[size:]
                t0 = time.time()
                rows = {}
                rows["phase fraction"] = fit_fraction(fractions, y, tr, te)
                rows["two-point statistics + PCA"] = fit_pca_method(X2p, y, tr, te)
                for method in METHODS[2:]:
                    rows[method] = fit_kernel_method({s: kernels[s][method] for s in kernels},
                                                     y, tr, te)
                for method, r in rows.items():
                    r["q2"] = float(r2_score(y[te], r["pred"]))
                    r["rmse"] = float(rmse(y[te], r["pred"]))
                    del r["pred"]
                    out.setdefault(label, {}).setdefault(str(size), {}).setdefault(
                        method, []).append(r)
                log(f"  {label}, n_train {size}, draw {rep}: " + ", ".join(
                    f"{m} {r['q2']:+.3f}" for m, r in rows.items()) +
                    f" ({time.time() - t0:.0f} s)")
    return out


def summarise(runs: List[dict]) -> dict:
    q2 = np.array([r["q2"] for r in runs])
    e = np.array([r["rmse"] for r in runs])
    smooth = [r["smoothing"] for r in runs if "smoothing" in r]
    settings = sorted({r["setting"] for r in runs})
    return {"q2": float(q2.mean()), "q2_sd": float(q2.std(ddof=1)) if len(q2) > 1 else 0.0,
            "rmse": float(e.mean()), "rmse_sd": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
            "smoothing": [float(s) for s in smooth], "settings": settings, "draws": len(runs)}


def run_set(dim: int, grid: int, n: int, contrast: float = CONTRAST, stages=("data", "fit"),
            force: bool = False) -> None:
    tag = tag_of(dim, grid, n, contrast)
    log(f"=== {tag}: {dim}D, {grid} points per side, {n} samples, contrast {contrast:g}")
    t0 = time.time()
    data = build_data(dim, grid, n, contrast, force=force and "data" in stages)
    y = labels(data)
    log(f"  labels: k_xx {y['xx'].min():.2f} to {y['xx'].max():.2f} (sd {y['xx'].std():.3f}), "
        f"mean {y['mean'].min():.2f} to {y['mean'].max():.2f} (sd {y['mean'].std():.3f}); "
        f"anisotropy |k_xx - k_yy| median {np.median(np.abs(data['tensor'][:, 0, 0] - data['tensor'][:, 1, 1])):.3f}")
    if "fit" not in stages:
        return
    t1 = time.time()
    kernels = {s: kernels_for(data, s) for s in SMOOTH}
    X2p = two_point_features(data)
    log(f"  kernels: {len(SMOOTH)} smoothing widths x {len(METHODS) - 2} kernels in "
        f"{time.time() - t1:.0f} s")
    res = evaluate(data, kernels, X2p)
    summary = {label: {size: {m: summarise(runs) for m, runs in by_m.items()}
                       for size, by_m in by_size.items()} for label, by_size in res.items()}
    os.makedirs(RESULTS, exist_ok=True)
    out = {"tag": tag, "dim": dim, "grid": grid, "n": n, "contrast": contrast,
           "solver": "finite volumes, harmonic-mean faces, one voxel pinned",
           "solver_checks": check_solver_lines(dim, grid, contrast),
           "label_spread": {k: {"min": float(v.min()), "max": float(v.max()),
                                "sd": float(v.std())} for k, v in y.items()},
           "grid_searched": {"smoothing": list(SMOOTH), "gammas": list(GAMMAS),
                             "lams": list(LAMS), "components": list(COMPONENTS),
                             "degrees": list(DEGREES), "ridges": list(RIDGES)},
           "sizes": list(SIZES), "repeats": REPEATS, "inner_folds": INNER,
           "minutes": (time.time() - t0) / 60.0,
           "summary": summary, "runs": res}
    with open(os.path.join(RESULTS, tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log(f"  written {tag}.json after {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def _load_results() -> List[dict]:
    found = []
    if not os.path.isdir(RESULTS):
        return found
    for name in sorted(os.listdir(RESULTS)):
        if name.endswith(".json"):
            with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
                found.append(json.load(fh))
    return found


def _smoothing_text(summary: dict) -> str:
    sm = summary.get("smoothing", [])
    if not sm:
        return "-"
    vals, counts = np.unique(np.array(sm), return_counts=True)
    return ", ".join(f"{v:g} ({c})" for v, c in zip(vals, counts))


def report(out: str = "") -> str:
    results = _load_results()
    if not results:
        raise SystemExit("no results under benchmarks/results/microstructure")
    lines = ["# A microstructure as the sample: field-overlap kernels against two-point "
             "statistics", "",
             "Synthetic two-phase microstructures on a periodic voxel grid (Gaussian-filtered "
             "noise cut at the phase fraction; the phase fraction, the grain size, the aspect "
             "ratio and the direction of the grains vary from sample to sample). The label is "
             "the effective thermal conductivity of the periodic cell from a finite-volume "
             "solver (phase 0 at 1, phase 1 at the contrast): the component along x, which "
             "depends on the direction of the grains, and the mean over the directions, which "
             "does not. Training sets of each size drawn five times at random, the rest of the "
             "set held out; every setting chosen by five-fold cross-validation inside the "
             "training part. q2 is R2 on the held-out samples, mean over the five draws with "
             "the spread in brackets; RMSE is in units of the conductivity of phase 0.", ""]
    for r in results:
        lines += [f"## {r['dim']}D, {r['grid']} points per side, {r['n']} samples, "
                  f"contrast {r['contrast']:g}", ""]
        spread = r["label_spread"]
        lines += ["Labels: k_xx from {:.2f} to {:.2f} (sd {:.3f}), mean from {:.2f} to {:.2f} "
                  "(sd {:.3f}). Solver checks: {}. Settings searched: smoothing width in "
                  "{{{}}} voxels; Gaussian layer bandwidth in {{{}}} times the median squared "
                  "distance, or linear; ridge in {{{}}} times the mean self-kernel; for the "
                  "baseline {{{}}} components, degree {{{}}}, ridge {{{}}}. Run time {:.0f} "
                  "min.".format(
                      spread["xx"]["min"], spread["xx"]["max"], spread["xx"]["sd"],
                      spread["mean"]["min"], spread["mean"]["max"], spread["mean"]["sd"],
                      "; ".join(r["solver_checks"]),
                      ", ".join(f"{s:g}" for s in r["grid_searched"]["smoothing"]),
                      ", ".join(f"{g:g}" for g in r["grid_searched"]["gammas"]),
                      ", ".join(f"{l:g}" for l in r["grid_searched"]["lams"]),
                      ", ".join(str(c) for c in r["grid_searched"]["components"]),
                      ", ".join(str(d) for d in r["grid_searched"]["degrees"]),
                      ", ".join(f"{a:g}" for a in r["grid_searched"]["ridges"]),
                      r["minutes"]), ""]
        for label in LABELS:
            if label not in r["summary"]:
                continue
            by_size = r["summary"][label]
            sizes = sorted(by_size, key=int)
            what = ("the component along x (direction-dependent)" if label == "xx"
                    else "the mean over the directions (direction-free)")
            lines += [f"**Label: {what}.**", "",
                      "| method | " + " | ".join(f"q2, n_train {s}" for s in sizes) + " | "
                      + " | ".join(f"RMSE, n_train {s}" for s in sizes) + " |",
                      "|---|" + "---:|" * (2 * len(sizes))]
            for m in METHODS:
                if not all(m in by_size[s] for s in sizes):
                    continue
                cells = [f"{by_size[s][m]['q2']:+.3f} ({by_size[s][m]['q2_sd']:.3f})"
                         for s in sizes]
                cells += [f"{by_size[s][m]['rmse']:.3f} ({by_size[s][m]['rmse_sd']:.3f})"
                          for s in sizes]
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
            lines += ["Smoothing width chosen by the inner cross-validation, in voxels, with the "
                      "number of draws that chose it (five draws per training size):", "",
                      "| method | " + " | ".join(f"n_train {s}" for s in sizes) + " |",
                      "|---|" + "---|" * len(sizes)]
            for m in METHODS[2:]:
                if not all(m in by_size[s] for s in sizes):
                    continue
                lines.append(f"| {m} | " + " | ".join(_smoothing_text(by_size[s][m])
                                                     for s in sizes) + " |")
            lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    out = out or os.path.join(RESULTS, "table.md")
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    print(text)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, default=None, help="2 or 3 (default: both default sets)")
    ap.add_argument("--grid", type=int, default=None, help="points per side, odd")
    ap.add_argument("--n", type=int, default=None, help="number of samples")
    ap.add_argument("--contrast", type=float, default=None,
                    help="conductivity of phase 1 over that of phase 0 (default %g)" % CONTRAST)
    ap.add_argument("--stage", choices=("data", "fit"), default=None)
    ap.add_argument("--force", action="store_true", help="rebuild the data")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    if args.report:
        report()
        return 0
    if args.dim is None and args.grid is None and args.n is None:
        sets = DEFAULT_SETS
    else:
        dim = args.dim or 2
        default = next((s for s in DEFAULT_SETS if s[0] == dim), DEFAULT_SETS[0])
        sets = ((dim, args.grid or default[1], args.n or default[2],
                 args.contrast or CONTRAST),)
    stages = (args.stage,) if args.stage else ("data", "fit")
    for dim, grid, n, contrast in sets:
        run_set(dim, grid, n, contrast, stages=stages, force=args.force)
    if "fit" in stages:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
