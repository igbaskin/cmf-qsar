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
    python benchmarks/run_microstructure.py --protocol gate  # 20 draws, equal budgets, the GP
    python benchmarks/run_microstructure.py --report
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

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
KERNEL_METHODS = ("overlap, generating frame", "overlap, rotation-averaged",
                  "autocorrelation, linear", "autocorrelation, normalised",
                  "autocorrelation, lattice-rotation-averaged", "autocorrelation, radial")
BASELINES = ("phase fraction", "two-point statistics + PCA", "two-point statistics + GP")
METHODS = ("phase fraction", "two-point statistics + PCA") + KERNEL_METHODS


class Protocol:
    """One measurement protocol: how many draws, which settings each method may try.

    The pilot let the kernel methods try 140 settings (five widths, four layers, seven ridges)
    and the baseline 40, and drew every training size five times. The gate gives every method
    the same budget of 40 settings and draws twenty times, so that a paired difference over the
    draws has a standard deviation worth reading. The Gaussian process on the principal
    components (``two-point statistics + GP``) fits its own length scales, one per score, and
    its noise by the marginal likelihood; its cross-validated choice is the number of
    components alone, so its budget is five settings each of which is itself an optimisation.
    """

    def __init__(self, name: str, repeats: int, smooth, gammas, lams, methods, layers_linear=True):
        self.name = name
        self.repeats = repeats
        self.smooth = tuple(float(x) for x in smooth)
        self.gammas = tuple(float(x) for x in gammas)
        self.lams = tuple(float(x) for x in lams)
        self.methods = tuple(methods)
        self.layers_linear = layers_linear

    @property
    def layers(self) -> Tuple[str, ...]:
        return (("linear",) if self.layers_linear else ()) + tuple(f"gauss {g:g}" for g in self.gammas)

    @property
    def kernel_budget(self) -> int:
        return len(self.smooth) * len(self.layers) * len(self.lams)

    def prefix(self) -> str:
        return "" if self.name == "pilot" else self.name + "_"


PILOT = Protocol("pilot", REPEATS, SMOOTH, GAMMAS, LAMS, METHODS)
GATE = Protocol("gate", 20, (0.0, 1.0, 2.0, 5.0), (1.0,), (1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
                BASELINES + KERNEL_METHODS)
PROTOCOLS = {p.name: p for p in (PILOT, GATE)}


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


LABEL_TEXT = {"xx": "the conductivity along x (direction-dependent)",
              "mean": "the conductivity, mean over the directions (direction-free)",
              "C_xx": "the PyMKS elastic stiffness C_xx (direction-dependent)",
              "C_mean": "the PyMKS elastic stiffness, mean of C_xx and C_yy (direction-free)"}


def labels(data: dict, tag: Optional[str] = None) -> Dict[str, np.ndarray]:
    """The conductivity labels, and the PyMKS elastic labels when prepare_elastic.py has
    written them for this set (``elastic_<tag>.npz`` beside the data)."""
    t = data["tensor"]
    dim = t.shape[1]
    out = {"xx": t[:, 0, 0].copy(), "mean": np.trace(t, axis1=1, axis2=2) / dim}
    if tag is not None:
        path = os.path.join(CACHE, "elastic_" + tag + ".npz")
        if os.path.isfile(path):
            e = np.load(path)
            if len(e["cxx"]) == len(t) and np.all(np.isfinite(e["cxx"])):
                out["C_xx"] = e["cxx"].astype(float)
                out["C_mean"] = 0.5 * (e["cxx"] + e["cyy"]).astype(float)
    return out


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

def layered(K: np.ndarray, tr: np.ndarray, protocol: Protocol = PILOT) -> Dict[str, np.ndarray]:
    """The linear kernel and its Gaussian layers, each scaled to a mean self-kernel of one on
    the training part so that the ridge grid is relative."""
    out = {}
    d = float(np.mean(np.diag(K)[tr]))
    Kl = K / max(d, 1e-300)
    if protocol.layers_linear:
        out["linear"] = Kl
    if protocol.gammas:
        scale = median_gamma_scale(Kl[np.ix_(tr, tr)])
        diag = np.diag(Kl)
        d2 = np.maximum(diag[:, None] + diag[None, :] - 2.0 * Kl, 0.0)
        for g in protocol.gammas:
            out[f"gauss {g:g}"] = np.exp(-g * scale * d2)
    return out


def fit_kernel_method(kernels: Dict[float, np.ndarray], y: np.ndarray, tr: np.ndarray,
                      te: np.ndarray, protocol: Protocol = PILOT) -> dict:
    """Nested choice of smoothing, layer and ridge by inner cross-validation on `tr`."""
    best = None
    for s in protocol.smooth:
        K = kernels[s]
        for layer, Kl in layered(K, tr, protocol).items():
            K_tr = Kl[np.ix_(tr, tr)]
            for lam in protocol.lams:
                p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
                err = rmse(y[tr], p)
                if best is None or err < best[0]:
                    best = (err, s, layer, lam)
    err, s, layer, lam = best
    Kl = layered(kernels[s], tr, protocol)[layer]
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

    def scores(self, X_te: np.ndarray, components: int):
        """The standardised scores of the training part and of X_te on the first components."""
        c = min(components, self.Vt.shape[0])
        Z_tr = self.Z_tr[:, :c]
        Z_te = (X_te - self.mean) @ self.Vt[:c].T
        sd = np.maximum(Z_tr.std(axis=0), 1e-12)
        return Z_tr / sd, Z_te / sd

    def predict(self, y_tr: np.ndarray, X_te: np.ndarray, components: int, degree: int,
                alpha: float) -> np.ndarray:
        """A polynomial of the standardised scores on the first components, ridge regression."""
        Z_tr, Z_te = self.scores(X_te, components)
        P_tr, P_te = _poly(Z_tr, degree), _poly(Z_te, degree)
        return P_te @ _ridge(P_tr, y_tr, alpha)

    def predict_gp(self, y_tr: np.ndarray, X_te: np.ndarray, components: int) -> np.ndarray:
        """A Gaussian process on the standardised scores: a squared-exponential kernel with one
        length scale per score, an amplitude and a noise term, all fitted by the marginal
        likelihood from one start (the GP-on-PCA recipe of the MKS literature)."""
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
        Z_tr, Z_te = self.scores(X_te, components)
        c = Z_tr.shape[1]
        kernel = (ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(c), (1e-2, 1e3))
                  + WhiteKernel(1e-2, (1e-8, 1e1)))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0,
                                      random_state=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(Z_tr, y_tr)
        return gp.predict(Z_te)


def _inner_folds(n: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    idx = np.random.RandomState(0).permutation(n)
    folds = [idx[k::INNER] for k in range(INNER)]
    return [(np.setdiff1d(idx, f), f) for f in folds]


def _poly_fits(components: int, degree: int, n_train: int) -> bool:
    """Whether a polynomial of `degree` in `components` scores has fewer terms than the
    training part of an inner fold has samples. Without this rule the inner cross-validation
    on 20 samples sometimes keeps 66 terms on 16 points, and one such draw can put the mean q2
    of a baseline at minus several hundred."""
    n_inner = n_train * (INNER - 1) // INNER
    terms = _poly(np.zeros((1, components)), degree).shape[1]
    return terms < n_inner


def fit_pca_method(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    best = None
    folds = _inner_folds(len(tr))
    pcas = [(_PCA(X[tr[a]]), a, b) for a, b in folds]
    for components in COMPONENTS:
        for degree in DEGREES:
            if not _poly_fits(components, degree, len(tr)):
                continue
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


def fit_gp_method(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    """The number of components by inner cross-validation; everything else by the marginal
    likelihood."""
    best = None
    folds = _inner_folds(len(tr))
    pcas = [(_PCA(X[tr[a]]), a, b) for a, b in folds]
    for components in COMPONENTS:
        if components >= len(tr) * (INNER - 1) // INNER:
            continue
        p = np.full(len(tr), np.nan)
        for pca, a, b in pcas:
            p[b] = pca.predict_gp(y[tr[a]], X[tr[b]], components)
        err = rmse(y[tr], p)
        if best is None or err < best[0]:
            best = (err, components)
    err, components = best
    pred = _PCA(X[tr]).predict_gp(y[tr], X[te], components)
    return {"setting": f"{components} components, GP", "components": components,
            "inner_rmse": float(err), "pred": pred}


def fit_fraction(f: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> dict:
    P = _poly(f[:, None], 3)
    pred = P[te] @ _ridge(P[tr], y[tr], 1e-8)
    return {"setting": "cubic in the fraction", "pred": pred}


def _one_draw(data_phase_fractions, ys, kernels, X2p, label, size, rep, protocol):
    """Every method of the protocol on one draw of a training set; what evaluate collects."""
    fractions = data_phase_fractions
    y = ys[label]
    n = len(y)
    perm = np.random.RandomState(1000 * size + rep).permutation(n)
    tr, te = perm[:size], perm[size:]
    t0 = time.time()
    rows = {}
    for method in protocol.methods:
        if method == "phase fraction":
            rows[method] = fit_fraction(fractions, y, tr, te)
        elif method == "two-point statistics + PCA":
            rows[method] = fit_pca_method(X2p, y, tr, te)
        elif method == "two-point statistics + GP":
            rows[method] = fit_gp_method(X2p, y, tr, te)
        else:
            rows[method] = fit_kernel_method({s: kernels[s][method] for s in kernels},
                                             y, tr, te, protocol)
    for method, r in rows.items():
        r["q2"] = float(r2_score(y[te], r["pred"]))
        r["rmse"] = float(rmse(y[te], r["pred"]))
        del r["pred"]
    return rows, time.time() - t0


def evaluate(data: dict, kernels: Dict[float, Dict[str, np.ndarray]], X2p: np.ndarray,
             sizes=SIZES, repeats=None, label_names=LABELS, protocol: Protocol = PILOT,
             ys: Optional[Dict[str, np.ndarray]] = None, jobs: int = 1) -> dict:
    """Every method on every draw of every training size; the draws of one size run in
    parallel over `jobs` processes when jobs > 1."""
    from joblib import Parallel, delayed
    ys = labels(data) if ys is None else ys
    n = len(data["phase"])
    fractions = data["phase"].reshape(n, -1).mean(axis=1)
    repeats = protocol.repeats if repeats is None else repeats
    out = {}
    for label in label_names:
        for size in sizes:
            if size >= n:
                continue
            t0 = time.time()
            draws = Parallel(n_jobs=jobs)(
                delayed(_one_draw)(fractions, ys, kernels, X2p, label, size, rep, protocol)
                for rep in range(repeats))
            for rep, (rows, seconds) in enumerate(draws):
                for method, r in rows.items():
                    out.setdefault(label, {}).setdefault(str(size), {}).setdefault(
                        method, []).append(r)
            means = {m: np.mean([rows[m]["q2"] for rows, _ in draws]) for m in draws[0][0]}
            log(f"  {protocol.name}, {label}, n_train {size}, {repeats} draws: " + ", ".join(
                f"{m} {v:+.3f}" for m, v in means.items()) +
                f" ({time.time() - t0:.0f} s, {np.mean([t for _, t in draws]):.1f} s per draw)")
    return out


def summarise(runs: List[dict]) -> dict:
    q2 = np.array([r["q2"] for r in runs])
    e = np.array([r["rmse"] for r in runs])
    smooth = [r["smoothing"] for r in runs if "smoothing" in r]
    settings = sorted({r["setting"] for r in runs})
    return {"q2": float(q2.mean()), "q2_sd": float(q2.std(ddof=1)) if len(q2) > 1 else 0.0,
            "rmse": float(e.mean()), "rmse_sd": float(e.std(ddof=1)) if len(e) > 1 else 0.0,
            "smoothing": [float(s) for s in smooth], "settings": settings, "draws": len(runs)}


def paired(runs: List[dict], ref: List[dict]) -> dict:
    """Mean and standard deviation over the draws of (method minus reference), q2 and RMSE."""
    dq = np.array([a["q2"] - b["q2"] for a, b in zip(runs, ref)])
    de = np.array([a["rmse"] - b["rmse"] for a, b in zip(runs, ref)])
    sd = (lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0)
    return {"q2": float(dq.mean()), "q2_sd": sd(dq), "rmse": float(de.mean()), "rmse_sd": sd(de),
            "wins": int(np.sum(dq > 0)), "draws": int(len(dq))}


def summary_of(res: dict) -> dict:
    """Per label and size: the summary of every method and its paired difference against every
    baseline that was run."""
    out = {}
    for label, by_size in res.items():
        for size, by_m in by_size.items():
            block = {m: summarise(runs) for m, runs in by_m.items()}
            for m, runs in by_m.items():
                block[m]["against"] = {b: paired(runs, by_m[b]) for b in BASELINES
                                       if b in by_m and b != m}
            out.setdefault(label, {})[size] = block
    return out


def run_set(dim: int, grid: int, n: int, contrast: float = CONTRAST, stages=("data", "fit"),
            force: bool = False, protocol: Protocol = PILOT, jobs: int = 1,
            label_names: Optional[Sequence[str]] = None) -> None:
    tag = tag_of(dim, grid, n, contrast)
    log(f"=== {tag}: {dim}D, {grid} points per side, {n} samples, contrast {contrast:g}, "
        f"protocol {protocol.name}")
    t0 = time.time()
    data = build_data(dim, grid, n, contrast, force=force and "data" in stages)
    y = labels(data, tag)
    wanted = tuple(label_names) if label_names else tuple(y)
    missing = [k for k in wanted if k not in y]
    if missing:
        raise SystemExit(f"no such label for {tag}: {missing} (the elastic labels need "
                         f"benchmarks/prepare_elastic.py first)")
    log(f"  labels: k_xx {y['xx'].min():.2f} to {y['xx'].max():.2f} (sd {y['xx'].std():.3f}), "
        f"mean {y['mean'].min():.2f} to {y['mean'].max():.2f} (sd {y['mean'].std():.3f}); "
        f"anisotropy |k_xx - k_yy| median {np.median(np.abs(data['tensor'][:, 0, 0] - data['tensor'][:, 1, 1])):.3f}")
    if "fit" not in stages:
        return
    t1 = time.time()
    kernels = {s: kernels_for(data, s) for s in protocol.smooth}
    X2p = two_point_features(data)
    log(f"  kernels: {len(protocol.smooth)} smoothing widths x {len(KERNEL_METHODS)} kernels in "
        f"{time.time() - t1:.0f} s")
    res = evaluate(data, kernels, X2p, protocol=protocol, jobs=jobs, ys=y, label_names=wanted)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, protocol.prefix() + tag + ".json")
    previous = None
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as fh:
            previous = json.load(fh)
    if previous and previous.get("protocol") == protocol.name:
        # another label of the same set and protocol: keep what is there and add to it
        previous_runs = {k: v for k, v in previous.get("runs", {}).items() if k not in res}
        res = {**previous_runs, **res}
    out = {"tag": tag, "dim": dim, "grid": grid, "n": n, "contrast": contrast,
           "protocol": protocol.name,
           "solver": "finite volumes, harmonic-mean faces, one voxel pinned",
           "solver_checks": check_solver_lines(dim, grid, contrast),
           "label_spread": {k: {"min": float(v.min()), "max": float(v.max()),
                                "sd": float(v.std())} for k, v in y.items()},
           "label_text": {k: LABEL_TEXT.get(k, k) for k in y},
           "grid_searched": {"smoothing": list(protocol.smooth), "gammas": list(protocol.gammas),
                             "layers": list(protocol.layers), "lams": list(protocol.lams),
                             "kernel_budget": protocol.kernel_budget,
                             "components": list(COMPONENTS),
                             "degrees": list(DEGREES), "ridges": list(RIDGES)},
           "sizes": list(SIZES), "repeats": protocol.repeats, "inner_folds": INNER,
           "minutes": (time.time() - t0) / 60.0,
           "summary": summary_of(res), "runs": res}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log(f"  written {protocol.prefix()}{tag}.json after {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- #
# Item 3: a property that depends on two scales
# --------------------------------------------------------------------------- #

#: the lower-scale law: the conductivity of phase 1 falls with the defect concentration c of
#: that phase, k1(c) = CONTRAST / (1 + DEFECT_SLOPE * c), c uniform in [0, 1]; a monotone law
#: of the kind point-defect scattering gives, chosen so that k1 spans 10 down to 2.
DEFECT_SLOPE = 4.0
TWOSCALE_METHODS = ("lower-scale descriptor alone", "phase fraction + descriptor",
                    "two-point statistics + descriptor + PCA", "arrangement kernel alone",
                    "sum kernel", "product kernel", "two widths as two channels",
                    "two widths, product with descriptor")
TWOSCALE_WEIGHTS: Tuple[float, ...] = (0.25, 0.5, 0.75)   # weight of the fine width in (e)


def phase1_conductivity(c: np.ndarray, contrast: float = CONTRAST) -> np.ndarray:
    return contrast / (1.0 + DEFECT_SLOPE * np.asarray(c, float))


def build_twoscale(data: dict, tag: str, seed: int = 7, force: bool = False) -> dict:
    """Every cell solved again with its own phase-1 conductivity k1(c): (n,) c and (n, d, d)."""
    path = os.path.join(CACHE, "twoscale_" + tag + ".npz")
    if os.path.isfile(path) and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    rng = np.random.default_rng(seed)
    n = len(data["phase"])
    c = rng.uniform(0.0, 1.0, n)
    k1 = phase1_conductivity(c)
    tensors = np.zeros((n,) + data["tensor"].shape[1:])
    t0 = time.time()
    for i in range(n):
        tensors[i] = effective_conductivity(data["phase"][i], float(k1[i]))
    out = {"c": c, "k1": k1, "tensor": tensors}
    np.savez_compressed(path, **out)
    log(f"  two-scale labels solved in {time.time() - t0:.0f} s: k_xx "
        f"{tensors[:, 0, 0].min():.2f} to {tensors[:, 0, 0].max():.2f}")
    return out


def _unit(K: np.ndarray, tr: np.ndarray) -> np.ndarray:
    return K / max(float(np.mean(np.diag(K)[tr])), 1e-300)


def descriptor_kernels(c: np.ndarray, tr: np.ndarray) -> Dict[str, np.ndarray]:
    """Kernels on the lower-scale descriptor alone: 1 + c c' (a line with a bias), and a
    Gaussian in c with the median-distance bandwidth."""
    out = {"linear": 1.0 + np.outer(c, c)}
    d2 = (c[:, None] - c[None, :]) ** 2
    med = float(np.median(d2[np.ix_(tr, tr)][~np.eye(len(tr), dtype=bool)]))
    out["gauss 1"] = np.exp(-d2 / max(med, 1e-12))
    return out


def _choose_and_fit(candidates: Dict[str, np.ndarray], y, tr, te, lams) -> dict:
    """The candidate kernel and ridge with the smallest inner RMSE, fitted and applied."""
    best = None
    for name, K in candidates.items():
        K_tr = K[np.ix_(tr, tr)]
        for lam in lams:
            p = cv_predict(K_tr, y[tr], "krr", {"lam": lam}, folds=INNER, seed=0)
            err = rmse(y[tr], p)
            if best is None or err < best[0]:
                best = (err, name, lam)
    err, name, lam = best
    K = candidates[name]
    learner = _Learner("krr", {"lam": lam}, True).fit(K[np.ix_(tr, tr)], y[tr])
    return {"setting": f"{name}, lam {lam:g}", "lam": lam, "inner_rmse": float(err),
            "pred": learner.predict(K[np.ix_(te, tr)])}


def _twoscale_draw(fractions, c, ys, kernels, X2p, label, size, rep, protocol):
    """Every two-scale method on one draw: the candidates of each are built here from the
    arrangement kernels (every width and layer of the protocol) and the descriptor kernels."""
    y = ys[label]
    n = len(y)
    perm = np.random.RandomState(1000 * size + rep).permutation(n)
    tr, te = perm[:size], perm[size:]
    t0 = time.time()
    arr = {}                                            # arrangement candidates
    for s_w in protocol.smooth:
        for layer, Kl in layered(kernels[s_w]["autocorrelation, linear"], tr, protocol).items():
            arr[f"s {s_w:g}, {layer}"] = Kl
    desc = {k: _unit(K, tr) for k, K in descriptor_kernels(c, tr).items()}
    rows = {}
    rows["lower-scale descriptor alone"] = _choose_and_fit(desc, y, tr, te, protocol.lams)
    P = np.column_stack([_poly(fractions[:, None], 3), c, c * fractions, c ** 2])
    rows["phase fraction + descriptor"] = {
        "setting": "cubic in the fraction, c, c x fraction, c squared",
        "pred": P[te] @ _ridge(P[tr], y[tr], 1e-8)}
    X_aug = np.column_stack([X2p, c * np.sqrt(X2p.shape[1]) * X2p.std()])
    rows["two-point statistics + descriptor + PCA"] = fit_pca_method(X_aug, y, tr, te)
    rows["arrangement kernel alone"] = _choose_and_fit(arr, y, tr, te, protocol.lams)
    rows["sum kernel"] = _choose_and_fit(
        {f"{a} + {b}": 0.5 * (Ka + Kb) for a, Ka in arr.items() for b, Kb in desc.items()},
        y, tr, te, protocol.lams)
    rows["product kernel"] = _choose_and_fit(
        {f"{a} x {b}": Ka * Kb for a, Ka in arr.items() for b, Kb in desc.items()},
        y, tr, te, protocol.lams)
    fine, coarse = protocol.smooth[0], protocol.smooth[-1]
    two = {}
    for w in TWOSCALE_WEIGHTS:
        K2 = w * _unit(kernels[fine]["autocorrelation, linear"], tr) \
            + (1.0 - w) * _unit(kernels[coarse]["autocorrelation, linear"], tr)
        for layer, Kl in layered(K2, tr, protocol).items():
            two[f"fine weight {w:g}, {layer}"] = Kl
    rows["two widths as two channels"] = _choose_and_fit(two, y, tr, te, protocol.lams)
    rows["two widths, product with descriptor"] = _choose_and_fit(
        {f"{a} x {b}": Ka * Kb for a, Ka in two.items() for b, Kb in desc.items()},
        y, tr, te, protocol.lams)
    for method, r in rows.items():
        r["q2"] = float(r2_score(y[te], r["pred"]))
        r["rmse"] = float(rmse(y[te], r["pred"]))
        del r["pred"]
    return rows, time.time() - t0


def run_twoscale(dim: int = 2, grid: int = 51, n: int = 400, contrast: float = CONTRAST,
                 protocol: Protocol = GATE, jobs: int = 1, force: bool = False) -> None:
    """Item 3: the arrangement and the lower-scale state together."""
    from joblib import Parallel, delayed
    tag = tag_of(dim, grid, n, contrast)
    log(f"=== two-scale property on {tag}")
    t0 = time.time()
    data = build_data(dim, grid, n, contrast)
    two = build_twoscale(data, tag, force=force)
    ys = {"xx": two["tensor"][:, 0, 0].copy(),
          "mean": np.trace(two["tensor"], axis1=1, axis2=2) / dim}
    fractions = data["phase"].reshape(n, -1).mean(axis=1)
    kernels = {s_w: kernels_for(data, s_w) for s_w in protocol.smooth}
    X2p = two_point_features(data)
    res = {}
    for label in LABELS:
        for size in SIZES:
            t1 = time.time()
            draws = Parallel(n_jobs=jobs)(
                delayed(_twoscale_draw)(fractions, two["c"], ys, kernels, X2p, label, size,
                                        rep, protocol) for rep in range(protocol.repeats))
            for rows, _ in draws:
                for method, r in rows.items():
                    res.setdefault(label, {}).setdefault(str(size), {}).setdefault(
                        method, []).append(r)
            means = {m: np.mean([rows[m]["q2"] for rows, _ in draws]) for m in draws[0][0]}
            log(f"  two-scale, {label}, n_train {size}, {protocol.repeats} draws: " + ", ".join(
                f"{m} {v:+.3f}" for m, v in means.items()) + f" ({time.time() - t1:.0f} s)")
    summary = {}
    for label, by_size in res.items():
        for size, by_m in by_size.items():
            block = {m: summarise(runs) for m, runs in by_m.items()}
            for m, runs in by_m.items():
                block[m]["against"] = {b: paired(runs, by_m[b]) for b in
                                       ("arrangement kernel alone", "lower-scale descriptor alone",
                                        "two-point statistics + descriptor + PCA") if b != m}
            summary.setdefault(label, {})[size] = block
    corr = {"c vs k_xx": float(np.corrcoef(two["c"], ys["xx"])[0, 1]),
            "fraction vs k_xx": float(np.corrcoef(fractions, ys["xx"])[0, 1])}
    out = {"tag": tag, "law": f"k1(c) = {contrast:g} / (1 + {DEFECT_SLOPE:g} c), c uniform in [0, 1]",
           "label_spread": {k: {"min": float(v.min()), "max": float(v.max()), "sd": float(v.std())}
                            for k, v in ys.items()},
           "correlations": corr, "repeats": protocol.repeats, "sizes": list(SIZES),
           "minutes": (time.time() - t0) / 60.0, "summary": summary, "runs": res}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "twoscale_" + tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log(f"  written twoscale_{tag}.json after {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- #
# Item 4: the coefficient map over the lag vector
# --------------------------------------------------------------------------- #

def coefficient_map(data: dict, y: np.ndarray, lam: Optional[float] = None,
                    lams=GATE.lams) -> dict:
    """The linear autocorrelation model read back in r-space.

    With the linear kernel K_ij = sum_r S_i(r) S_j(r) and dual coefficients a, the prediction
    is y = sum_r w(r) S(r) + b with w(r) = sum_i a_i S_i(r), one map per channel block. The
    ridge is chosen by five-fold cross-validation on the whole set unless given. Returned with
    the standard deviation of S(r) over the samples, since w(r) times that spread is what a lag
    contributes to the differences between samples.
    """
    fields = smoothed_fields(data["phase"], 0.0)
    S = autocorrelations(fields)                        # (n, 2, grid, grid)
    n = S.shape[0]
    A = S.reshape(n, -1)
    K = A @ A.T
    d = float(np.mean(np.diag(K)))
    Kl = K / d
    if lam is None:
        best = None
        for l in lams:
            p = cv_predict(Kl, y, "krr", {"lam": l}, folds=INNER, seed=0)
            err = rmse(y, p)
            if best is None or err < best[0]:
                best = (err, l)
        lam = best[1]
    learner = _Learner("krr", {"lam": lam}, True).fit(Kl, y)
    w = (learner.alpha_ @ A / d).reshape(S.shape[1:])     # (2, grid, grid)
    sd = S.std(axis=0)
    fit = learner.predict(Kl)
    return {"w": w, "sd": sd, "lam": float(lam), "r2_train": float(r2_score(y, fit)),
            "cv_rmse": float(rmse(y, cv_predict(Kl, y, "krr", {"lam": lam}, folds=INNER, seed=0))),
            "cv_r2": float(r2_score(y, cv_predict(Kl, y, "krr", {"lam": lam}, folds=INNER, seed=0)))}


def _profiles(m: np.ndarray) -> dict:
    """The radial profile (mean over the shell of |r|) and the direction profile (mean of the
    map along the x axis against along the y axis, up to half the grid) of a centred map."""
    grid = m.shape[-1]
    coords = np.meshgrid(*([np.fft.fftfreq(grid) * grid] * 2), indexing="ij")
    r = np.sqrt(coords[0] ** 2 + coords[1] ** 2)
    shell = np.floor(r + 1e-9).astype(int)
    counts = np.bincount(shell.ravel())
    radial = np.bincount(shell.ravel(), weights=m.ravel()) / np.maximum(counts, 1)
    half = grid // 2
    along_x = [float(m[k, 0]) for k in range(half + 1)]             # lags (k, 0)
    along_y = [float(m[0, k]) for k in range(half + 1)]             # lags (0, k)
    return {"radial": radial[:half + 1].tolist(), "along_x": along_x, "along_y": along_y}


def run_map(dim: int = 2, grid: int = 51, n: int = 400, contrast: float = 100.0) -> None:
    """Item 4: the figure and the numbers behind it."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tag = tag_of(dim, grid, n, contrast)
    log(f"=== coefficient map on {tag}")
    data = build_data(dim, grid, n, contrast)
    ys = labels(data)
    out = {"tag": tag, "labels": {}}
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5))
    blocks = ("S11: phase 1 with phase 1", "S01: phase 0 with phase 1")
    for row, label in enumerate(LABELS):
        m = coefficient_map(data, ys[label])
        half = grid // 2
        extent = (-half - 0.5, half + 0.5, -half - 0.5, half + 0.5)
        info = {"lam": m["lam"], "r2_train": m["r2_train"], "cv_r2": m["cv_r2"],
                "cv_rmse": m["cv_rmse"], "blocks": {}}
        for col, name in enumerate(blocks):
            w = m["w"][col]
            effect = w * m["sd"][col]
            for k, (arr, title) in enumerate(((w, "coefficient w(r)"),
                                              (effect, "w(r) x spread of S(r)"))):
                ax = axes[row, 2 * col + k]
                shifted = np.fft.fftshift(arr)
                lim = float(np.max(np.abs(shifted)))
                im = ax.imshow(shifted.T, origin="lower", extent=extent, cmap="RdBu_r",
                               vmin=-lim, vmax=lim)
                ax.set_title(f"{'k_xx' if label == 'xx' else 'mean k'}: {name}\n{title}",
                             fontsize=9)
                ax.set_xlabel("lag along x, voxels")
                ax.set_ylabel("lag along y, voxels")
                fig.colorbar(im, ax=ax, fraction=0.046)
            prof = _profiles(effect)
            abs_prof = _profiles(np.abs(effect))
            info["blocks"][name] = {"effect": prof, "abs_effect": abs_prof,
                                    "coefficient": _profiles(w),
                                    "total_abs_effect": float(np.sum(np.abs(effect)))}
        out["labels"][label] = info
        log(f"  {label}: lam {m['lam']:g}, cross-validated R2 {m['cv_r2']:+.3f}")
    fig.suptitle(f"The linear autocorrelation model read in r-space, {dim}D, {grid} x {grid}, "
                 f"contrast {contrast:g}: red credits a lag, blue debits it", fontsize=11)
    fig.tight_layout()
    os.makedirs(RESULTS, exist_ok=True)
    fig.savefig(os.path.join(RESULTS, "map_" + tag + ".png"), dpi=130)
    plt.close(fig)
    with open(os.path.join(RESULTS, "map_" + tag + ".json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log(f"  written map_{tag}.png and .json")


# --------------------------------------------------------------------------- #
# Item 2: one real micrograph set
# --------------------------------------------------------------------------- #

UHCS_SIZES: Tuple[int, ...] = (20, 50, 100, 200, 500)
UHCS_SMOOTH: Tuple[float, ...] = (0.0, 1.0, 2.0, 4.0)      # pixels of the reduced image
UHCS_KERNELS = ("autocorrelation, grey, linear", "autocorrelation, grey, normalised",
                "autocorrelation, grey, radial", "autocorrelation, two-level, linear",
                "autocorrelation, two-level, radial")
UHCS_BASELINES = ("grey mean and spread", "two-point statistics + PCA",
                  "two-point statistics + GP", "CNN features + ridge")
UHCS_LABELS = ("anneal temperature, C", "log10 anneal time, h", "microconstituent class")
UHCS_SPLITS = ("images at random", "whole specimens")


def _krr_multi(K_tr: np.ndarray, Y_tr: np.ndarray, lam: float):
    """Kernel ridge regression with a bias for several targets at once (one-vs-rest)."""
    n = K_tr.shape[0]
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = K_tr + lam * np.eye(n)
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    rhs = np.vstack([Y_tr, np.zeros((1, Y_tr.shape[1]))])
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return sol[:n], sol[n]


def _class_scores(Y: np.ndarray, idx: np.ndarray, pred: np.ndarray) -> dict:
    truth = Y[idx].argmax(axis=1)
    guess = pred.argmax(axis=1)
    acc = float(np.mean(truth == guess))
    present = np.unique(truth)
    recalls = [float(np.mean(guess[truth == c] == c)) for c in present]
    return {"acc": acc, "bacc": float(np.mean(recalls))}


def _choose_and_fit_classes(candidates: Dict[str, np.ndarray], Y: np.ndarray, tr, te, lams) -> dict:
    """One-vs-rest kernel ridge on the class indicators; the candidate and the ridge by the
    inner accuracy of the largest score."""
    folds = _inner_folds(len(tr))
    best = None
    for name, K in candidates.items():
        for lam in lams:
            pred = np.zeros((len(tr), Y.shape[1]))
            for a, b in folds:
                alpha, bias = _krr_multi(K[np.ix_(tr[a], tr[a])], Y[tr[a]], lam)
                pred[b] = K[np.ix_(tr[b], tr[a])] @ alpha + bias
            acc = _class_scores(Y, tr, pred)["acc"]
            if best is None or acc > best[0]:
                best = (acc, name, lam)
    acc, name, lam = best
    alpha, bias = _krr_multi(candidates[name][np.ix_(tr, tr)], Y[tr], lam)
    pred = candidates[name][np.ix_(te, tr)] @ alpha + bias
    return {"setting": f"{name}, lam {lam:g}", "lam": lam, "inner_acc": float(acc),
            "pred": pred}


def _pca_classes(X: np.ndarray, Y: np.ndarray, tr, te, gp: bool = False) -> dict:
    """The PCA baselines one-vs-rest: the polynomial ridge, or one GP per class."""
    folds = _inner_folds(len(tr))
    pcas = [(_PCA(X[tr[a]]), a, b) for a, b in folds]
    best = None
    settings = ([(c, 0, 0.0) for c in COMPONENTS] if gp else
                [(c, d, a) for c in COMPONENTS for d in DEGREES for a in RIDGES])
    for components, degree, alpha in settings:
        if not _poly_fits(components, degree, len(tr)):
            continue
        pred = np.zeros((len(tr), Y.shape[1]))
        for pca, a, b in pcas:
            if gp:
                pred[b] = np.column_stack([pca.predict_gp(Y[tr[a], k], X[tr[b]], components)
                                           for k in range(Y.shape[1])])
            else:
                pred[b] = pca.predict(Y[tr[a]], X[tr[b]], components, degree, alpha)
        acc = _class_scores(Y, tr, pred)["acc"]
        if best is None or acc > best[0]:
            best = (acc, components, degree, alpha)
    acc, components, degree, alpha = best
    pca = _PCA(X[tr])
    if gp:
        pred = np.column_stack([pca.predict_gp(Y[tr, k], X[te], components)
                                for k in range(Y.shape[1])])
        setting = f"{components} components, GP"
    else:
        pred = pca.predict(Y[tr], X[te], components, degree, alpha)
        setting = f"{components} components, degree {degree}, ridge {alpha:g}"
    return {"setting": setting, "inner_acc": float(acc), "pred": pred}


def uhcs_kernels(images: np.ndarray, protocol_smooth=UHCS_SMOOTH) -> Dict[float, Dict[str, np.ndarray]]:
    """Every kernel of the micrograph methods at every smoothing width."""
    import uhcs_micrographs as U
    out = {}
    for s_w in protocol_smooth:
        sm = U.smooth(images, s_w)
        Sg = U.windowed_autocorrelation(sm)
        A = Sg.reshape(len(images), -1)
        K = {"autocorrelation, grey, linear": A @ A.T,
             "autocorrelation, grey, normalised": autocorrelation_kernel(A, blocks=[A.shape[1]])}
        R = U.radial_profile(Sg)
        K["autocorrelation, grey, radial"] = R @ R.T
        St = U.windowed_autocorrelation(U.two_level(sm))
        B = St.reshape(len(images), -1)
        K["autocorrelation, two-level, linear"] = B @ B.T
        Rt = U.radial_profile(St)
        K["autocorrelation, two-level, radial"] = Rt @ Rt.T
        out[s_w] = K
    return out


def _uhcs_draw(task, kernels, X2p, Zcnn, stats, y, Y, groups, split, size, rep, protocol):
    """Every method on one draw of the micrographs; `task` is "regression" or "classes"."""
    import uhcs_micrographs as U
    n = len(groups)
    if split == "whole specimens":
        tr, te = U.grouped_draw(groups, size, 1000 * size + rep)
    else:
        perm = np.random.RandomState(1000 * size + rep).permutation(n)
        tr, te = perm[:size], perm[size:]
    t0 = time.time()
    rows = {}
    stats_poly = _poly(stats, 2)
    cnn = {}
    Zc = (Zcnn - Zcnn[tr].mean(axis=0)) / np.maximum(Zcnn[tr].std(axis=0), 1e-6)
    Kc = Zc @ Zc.T
    for layer, Kl in layered(Kc, tr, protocol).items():
        cnn[layer] = Kl
    kern = {}
    for m in UHCS_KERNELS:
        cand = {}
        for s_w in kernels:
            for layer, Kl in layered(kernels[s_w][m], tr, protocol).items():
                cand[f"s {s_w:g}, {layer}"] = Kl
        kern[m] = cand
    if task == "regression":
        rows["grey mean and spread"] = {"setting": "quadratic in the mean and the spread",
                                        "pred": stats_poly[te] @ _ridge(stats_poly[tr], y[tr], 1e-8)}
        rows["two-point statistics + PCA"] = fit_pca_method(X2p, y, tr, te)
        rows["two-point statistics + GP"] = fit_gp_method(X2p, y, tr, te)
        rows["CNN features + ridge"] = _choose_and_fit(cnn, y, tr, te, protocol.lams)
        for m in UHCS_KERNELS:
            rows[m] = _choose_and_fit(kern[m], y, tr, te, protocol.lams)
        for method, r in rows.items():
            r["q2"] = float(r2_score(y[te], r["pred"]))
            r["rmse"] = float(rmse(y[te], r["pred"]))
            del r["pred"]
    else:
        rows["grey mean and spread"] = {"setting": "quadratic in the mean and the spread",
                                        "pred": stats_poly[te] @ _ridge(stats_poly[tr], Y[tr], 1e-8)}
        rows["two-point statistics + PCA"] = _pca_classes(X2p, Y, tr, te)
        rows["two-point statistics + GP"] = _pca_classes(X2p, Y, tr, te, gp=True)
        rows["CNN features + ridge"] = _choose_and_fit_classes(cnn, Y, tr, te, protocol.lams)
        for m in UHCS_KERNELS:
            rows[m] = _choose_and_fit_classes(kern[m], Y, tr, te, protocol.lams)
        for method, r in rows.items():
            r.update(_class_scores(Y, te, r["pred"]))
            del r["pred"]
    for r in rows.values():
        r["n_train"], r["n_test"] = int(len(tr)), int(len(te))
    return rows, time.time() - t0


def run_uhcs(protocol: Protocol = GATE, jobs: int = 1, sizes=UHCS_SIZES,
             splits=UHCS_SPLITS, labels_wanted=UHCS_LABELS) -> None:
    """Item 2: the field kernels against CNN features on real micrographs."""
    from joblib import Parallel, delayed
    import uhcs_micrographs as U
    log("=== UHCS micrographs")
    t0 = time.time()
    rows_meta = U.metadata()
    images = U.load_images(rows_meta)
    Zcnn = U.cnn_features(images, rows_meta).astype(np.float64)
    stats = np.column_stack([images.reshape(len(images), -1).mean(axis=1),
                             images.reshape(len(images), -1).std(axis=1)]).astype(np.float64)
    t1 = time.time()
    kernels = uhcs_kernels(images)
    X2p = U.windowed_autocorrelation(images.astype(np.float64)).reshape(len(images), -1)
    log(f"  kernels of {len(images)} micrographs: {len(UHCS_SMOOTH)} widths x "
        f"{len(UHCS_KERNELS)} kernels in {time.time() - t1:.0f} s")
    groups_all = np.array([r["specimen"] for r in rows_meta])
    temp = np.array([r["temperature"] for r in rows_meta])
    logt = np.log10(np.array([r["time_h"] for r in rows_meta]))
    cls = np.array([r["cls"] for r in rows_meta])
    Y_all = np.array([[1.0 if c == k else -1.0 for k in U.CLASSES] for c in cls])
    res, counts = {}, {}
    for label in labels_wanted:
        if label == "microconstituent class":
            keep = np.arange(len(rows_meta))
            task, y, Y = "classes", None, Y_all
        else:
            y_full = temp if label.startswith("anneal") else logt
            keep = np.flatnonzero(np.isfinite(y_full))
            task, y, Y = "regression", y_full[keep], None
        sub = {s_w: {m: K[np.ix_(keep, keep)] for m, K in K_s.items()}
               for s_w, K_s in kernels.items()}
        counts[label] = {"images": int(len(keep)),
                         "specimens": int(len(np.unique(groups_all[keep])))}
        for split in splits:
            for size in sizes:
                if size >= len(keep) - 20:
                    continue
                t2 = time.time()
                draws = Parallel(n_jobs=jobs)(
                    delayed(_uhcs_draw)(task, sub, X2p[keep], Zcnn[keep], stats[keep], y, Y,
                                        groups_all[keep], split, size, rep, protocol)
                    for rep in range(protocol.repeats))
                for rows, _ in draws:
                    for method, r in rows.items():
                        res.setdefault(label, {}).setdefault(split, {}).setdefault(
                            str(size), {}).setdefault(method, []).append(r)
                key = "q2" if task == "regression" else "acc"
                means = {m: np.mean([rows[m][key] for rows, _ in draws]) for m in draws[0][0]}
                log(f"  {label}, {split}, n_train {size}, {protocol.repeats} draws: " + ", ".join(
                    f"{m} {v:+.3f}" for m, v in means.items()) + f" ({time.time() - t2:.0f} s)")
    summary = {}
    for label, by_split in res.items():
        for split, by_size in by_split.items():
            for size, by_m in by_size.items():
                block = {}
                for m, runs in by_m.items():
                    if "q2" in runs[0]:
                        block[m] = summarise(runs)
                    else:
                        acc = np.array([r["acc"] for r in runs])
                        bacc = np.array([r["bacc"] for r in runs])
                        block[m] = {"acc": float(acc.mean()), "acc_sd": float(acc.std(ddof=1)),
                                    "bacc": float(bacc.mean()), "bacc_sd": float(bacc.std(ddof=1)),
                                    "settings": sorted({r["setting"] for r in runs}),
                                    "draws": len(runs)}
                    block[m]["n_train_mean"] = float(np.mean([r["n_train"] for r in runs]))
                    block[m]["n_test_mean"] = float(np.mean([r["n_test"] for r in runs]))
                for m, runs in by_m.items():
                    block[m]["against"] = {}
                    for b in ("CNN features + ridge", "two-point statistics + GP"):
                        if b in by_m and b != m:
                            if "q2" in runs[0]:
                                block[m]["against"][b] = paired(runs, by_m[b])
                            else:
                                d = np.array([a["acc"] - c["acc"] for a, c in zip(runs, by_m[b])])
                                block[m]["against"][b] = {
                                    "acc": float(d.mean()), "acc_sd": float(d.std(ddof=1)),
                                    "wins": int(np.sum(d > 0)), "draws": int(len(d))}
                summary.setdefault(label, {}).setdefault(split, {})[size] = block
    out = {"source": "UHCSDB (DeCost et al. 2017), Zenodo mirror record 23050464, CC BY 3.0 US",
           "images": {"shape": list(images.shape[1:]), "reduce": U.REDUCE,
                      "rows_kept": U.IMAGE_ROWS, "max_lag": U.MAX_LAG},
           "counts": counts, "classes": list(U.CLASSES),
           "class_counts": {k: int(np.sum(cls == k)) for k in U.CLASSES},
           "smoothing": list(UHCS_SMOOTH), "repeats": protocol.repeats,
           "sizes": list(sizes), "minutes": (time.time() - t0) / 60.0,
           "summary": summary, "runs": res}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "uhcs.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log(f"  written uhcs.json after {(time.time() - t0) / 60:.1f} min")


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def _load_results(prefix: str = "") -> List[dict]:
    """The result files of one protocol: the pilot's have no prefix, the gate's start with gate_."""
    found = []
    if not os.path.isdir(RESULTS):
        return found
    for name in sorted(os.listdir(RESULTS)):
        if not name.endswith(".json") or name.startswith(("twoscale", "map", "uhcs")):
            continue
        has_prefix = name.startswith(tuple(p.prefix() for p in PROTOCOLS.values() if p.prefix()))
        if (prefix and not name.startswith(prefix)) or (not prefix and has_prefix):
            continue
        with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
            found.append(json.load(fh))
    return found


def _smoothing_text(summary: dict) -> str:
    sm = summary.get("smoothing", [])
    if not sm:
        return "-"
    vals, counts = np.unique(np.array(sm), return_counts=True)
    return ", ".join(f"{v:g} ({c})" for v, c in zip(vals, counts))


def _block_lines(r: dict, methods: Sequence[str], show_paired: bool) -> List[str]:
    lines = [f"## {r['dim']}D, {r['grid']} points per side, {r['n']} samples, "
             f"contrast {r['contrast']:g}", ""]
    spread = r["label_spread"]
    g = r["grid_searched"]
    lines += ["Labels: k_xx from {:.2f} to {:.2f} (sd {:.3f}), mean from {:.2f} to {:.2f} "
              "(sd {:.3f}). Solver checks: {}. Settings searched: smoothing width in "
              "{{{}}} voxels; layer in {{{}}} (the Gaussian bandwidth is in units of the median "
              "squared distance); ridge in {{{}}} times the mean self-kernel ({} settings per "
              "kernel method); for the polynomial baseline {{{}}} components, degree {{{}}}, "
              "ridge {{{}}} ({} settings); for the GP baseline the components alone, with the "
              "length scales, the amplitude and the noise fitted by the marginal likelihood. "
              "{} draws per training size. Run time {:.0f} min.".format(
                  spread["xx"]["min"], spread["xx"]["max"], spread["xx"]["sd"],
                  spread["mean"]["min"], spread["mean"]["max"], spread["mean"]["sd"],
                  "; ".join(r["solver_checks"]),
                  ", ".join(f"{s:g}" for s in g["smoothing"]),
                  ", ".join(g.get("layers", ["linear"] + [f"gauss {x:g}" for x in g["gammas"]])),
                  ", ".join(f"{l:g}" for l in g["lams"]),
                  g.get("kernel_budget", len(g["smoothing"]) * (1 + len(g["gammas"])) * len(g["lams"])),
                  ", ".join(str(c) for c in g["components"]),
                  ", ".join(str(d) for d in g["degrees"]),
                  ", ".join(f"{a:g}" for a in g["ridges"]),
                  len(g["components"]) * len(g["degrees"]) * len(g["ridges"]),
                  r["repeats"], r["minutes"]), ""]
    for label in [k for k in LABEL_TEXT if k in r["summary"]]:
        by_size = r["summary"][label]
        sizes = sorted(by_size, key=int)
        what = r.get("label_text", {}).get(label, LABEL_TEXT.get(label, label))
        spread_l = r["label_spread"].get(label)
        if spread_l:
            what += f"; from {spread_l['min']:.2f} to {spread_l['max']:.2f}, sd {spread_l['sd']:.3f}"
        lines += [f"**Label: {what}.**", "",
                  "| method | " + " | ".join(f"q2, n_train {s}" for s in sizes) + " | "
                  + " | ".join(f"RMSE, n_train {s}" for s in sizes) + " |",
                  "|---|" + "---:|" * (2 * len(sizes))]
        present = [m for m in methods if all(m in by_size[s] for s in sizes)]
        for m in present:
            cells = [f"{by_size[s][m]['q2']:+.3f} ({by_size[s][m]['q2_sd']:.3f})" for s in sizes]
            cells += [f"{by_size[s][m]['rmse']:.3f} ({by_size[s][m]['rmse_sd']:.3f})"
                      for s in sizes]
            lines.append(f"| {m} | " + " | ".join(cells) + " |")
        lines.append("")
        if show_paired:
            for base in ("two-point statistics + GP", "two-point statistics + PCA"):
                if base not in present:
                    continue
                lines += [f"Paired difference against *{base}* over the draws: mean of "
                          f"(method minus baseline) in q2 and in RMSE, with the standard "
                          f"deviation over the draws in brackets, and the number of draws the "
                          f"method was ahead in q2.", "",
                          "| method | " + " | ".join(f"q2 diff., n_train {s}" for s in sizes)
                          + " | " + " | ".join(f"RMSE diff., n_train {s}" for s in sizes)
                          + " | " + " | ".join(f"ahead, n_train {s}" for s in sizes) + " |",
                          "|---|" + "---:|" * (3 * len(sizes))]
                for m in present:
                    if m == base:
                        continue
                    d = [by_size[s][m].get("against", {}).get(base) for s in sizes]
                    if any(x is None for x in d):
                        continue
                    cells = [f"{x['q2']:+.3f} ({x['q2_sd']:.3f})" for x in d]
                    cells += [f"{x['rmse']:+.3f} ({x['rmse_sd']:.3f})" for x in d]
                    cells += [f"{x['wins']} of {x['draws']}" for x in d]
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
                lines.append("")
        lines += ["Smoothing width chosen by the inner cross-validation, in voxels, with the "
                  f"number of draws that chose it ({r['repeats']} draws per training size):", "",
                  "| method | " + " | ".join(f"n_train {s}" for s in sizes) + " |",
                  "|---|" + "---|" * len(sizes)]
        for m in KERNEL_METHODS:
            if m in present:
                lines.append(f"| {m} | " + " | ".join(_smoothing_text(by_size[s][m])
                                                     for s in sizes) + " |")
        lines.append("")
    return lines


def _jload(name: str):
    path = os.path.join(RESULTS, name)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _twoscale_lines() -> List[str]:
    lines = []
    for name in sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []:
        if not (name.startswith("twoscale_") and name.endswith(".json")):
            continue
        r = _jload(name)
        lines += [f"## A property of two scales, on {r['tag']}", "",
                  f"Phase 1 carries a lower-scale state: a defect concentration c, uniform in "
                  f"[0, 1], that sets its conductivity by {r['law']}. The arrangement is the "
                  f"same {r['tag']} set; every cell is solved again with its own phase-1 "
                  f"conductivity. Correlation of c with k_xx {r['correlations']['c vs k_xx']:+.2f}, "
                  f"of the phase fraction with k_xx {r['correlations']['fraction vs k_xx']:+.2f}. "
                  f"The arrangement kernel is the autocorrelation kernel with its width and layer "
                  f"chosen as in the gate; the descriptor kernel is 1 + c c' or a Gaussian in c; "
                  f"the sum and the product combine one candidate of each; the two-width kernel "
                  f"mixes the finest and the coarsest width with weight {{{', '.join(f'{w:g}' for w in TWOSCALE_WEIGHTS)}}} "
                  f"on the fine one. {r['repeats']} draws per size. Run time {r['minutes']:.0f} min.", ""]
        for label in LABELS:
            if label not in r["summary"]:
                continue
            by_size = r["summary"][label]
            sizes = sorted(by_size, key=int)
            what = "k_xx (direction-dependent)" if label == "xx" else "the mean over the directions"
            lines += [f"**Label: {what}.**", "",
                      "| method | " + " | ".join(f"q2, n_train {s_}" for s_ in sizes) + " | "
                      + " | ".join(f"RMSE, n_train {s_}" for s_ in sizes) + " |",
                      "|---|" + "---:|" * (2 * len(sizes))]
            for m in TWOSCALE_METHODS:
                if not all(m in by_size[s_] for s_ in sizes):
                    continue
                cells = [f"{by_size[s_][m]['q2']:+.3f} ({by_size[s_][m]['q2_sd']:.3f})" for s_ in sizes]
                cells += [f"{by_size[s_][m]['rmse']:.3f} ({by_size[s_][m]['rmse_sd']:.3f})" for s_ in sizes]
                lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
            base = "arrangement kernel alone"
            lines += [f"Paired difference in q2 against *{base}* over the draws (mean, standard "
                      f"deviation in brackets, draws ahead):", "",
                      "| method | " + " | ".join(f"n_train {s_}" for s_ in sizes) + " |",
                      "|---|" + "---:|" * len(sizes)]
            for m in TWOSCALE_METHODS:
                if m == base or not all(m in by_size[s_] for s_ in sizes):
                    continue
                d = [by_size[s_][m]["against"].get(base) for s_ in sizes]
                if any(x is None for x in d):
                    continue
                lines.append(f"| {m} | " + " | ".join(
                    f"{x['q2']:+.3f} ({x['q2_sd']:.3f}), {x['wins']} of {x['draws']}" for x in d) + " |")
            lines.append("")
    return lines


def _map_lines() -> List[str]:
    lines = []
    for name in sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []:
        if not (name.startswith("map_") and name.endswith(".json")):
            continue
        r = _jload(name)
        lines += [f"## The coefficient map of the linear autocorrelation model, on {r['tag']}", "",
                  f"The model y = sum over r of w(r) S(r) + b read back on the lag vector r, "
                  f"fitted on all 400 cells with the ridge chosen by five-fold cross-validation; "
                  f"the figure is `{name.replace('.json', '.png')}`. The numbers below are the "
                  f"effect w(r) times the spread of S(r) over the samples, for the phase-1 "
                  f"autocorrelation block, along the x axis and along the y axis of the lag, "
                  f"and the mean over the shell of each lag length.", ""]
        for label, info in r["labels"].items():
            b = info["blocks"]["S11: phase 1 with phase 1"]
            e = b["effect"]
            lines += [f"**{'k_xx' if label == 'xx' else 'mean k'}**: ridge {info['lam']:g}, "
                      f"cross-validated R2 {info['cv_r2']:+.3f}.", "",
                      "| lag, voxels | " + " | ".join(str(k) for k in range(0, 13)) + " |",
                      "|---|" + "---:|" * 13,
                      "| along x | " + " | ".join(f"{v:+.3f}" for v in e["along_x"][:13]) + " |",
                      "| along y | " + " | ".join(f"{v:+.3f}" for v in e["along_y"][:13]) + " |",
                      "| shell mean | " + " | ".join(f"{v:+.3f}" for v in e["radial"][:13]) + " |", ""]
    return lines


def _uhcs_lines() -> List[str]:
    r = _jload("uhcs.json")
    if r is None:
        return []
    lines = ["## Real micrographs: the UHCS set", "",
             f"{r['source']}. {r['counts'].get('microconstituent class', {}).get('images', '?')} "
             f"micrographs of {r['counts'].get('microconstituent class', {}).get('specimens', '?')} "
             f"specimens; the annealing temperature and time are recorded for "
             f"{r['counts'].get('anneal temperature, C', {}).get('images', '?')} of them. Each "
             f"image is cut above its information bar and reduced {r['images']['reduce']} times "
             f"by block means to {r['images']['shape'][1]} x {r['images']['shape'][0]} pixels; "
             f"the two-point autocorrelation is computed with zero padding and the count of "
             f"overlapping pairs as the denominator, lags up to {r['images']['max_lag']} pixels. "
             f"Smoothing widths {{{', '.join(f'{x:g}' for x in r['smoothing'])}}} pixels. "
             f"Classes: " + ", ".join(f"{k} {v}" for k, v in r["class_counts"].items()) + ". "
             f"Two ways of drawing a training set: images at random (the way most published "
             f"numbers are made; images of one specimen then sit on both sides), and whole "
             f"specimens (the test images come from specimens the model never saw). "
             f"{r['repeats']} draws per size. Run time {r['minutes']:.0f} min.", ""]
    methods = UHCS_BASELINES + UHCS_KERNELS
    for label, by_split in r["summary"].items():
        for split, by_size in by_split.items():
            sizes = sorted(by_size, key=int)
            is_cls = "acc" in next(iter(by_size[sizes[0]].values()))
            ntr = ", ".join(f"{by_size[s_][methods[0]]['n_train_mean']:.0f}" for s_ in sizes)
            lines += [f"**{label}, training sets of {split}** (mean training count {ntr}).", ""]
            if is_cls:
                lines += ["| method | " + " | ".join(f"accuracy, n_train {s_}" for s_ in sizes)
                          + " | " + " | ".join(f"balanced acc., n_train {s_}" for s_ in sizes) + " |",
                          "|---|" + "---:|" * (2 * len(sizes))]
                for m in methods:
                    if not all(m in by_size[s_] for s_ in sizes):
                        continue
                    cells = [f"{by_size[s_][m]['acc']:.3f} ({by_size[s_][m]['acc_sd']:.3f})" for s_ in sizes]
                    cells += [f"{by_size[s_][m]['bacc']:.3f} ({by_size[s_][m]['bacc_sd']:.3f})" for s_ in sizes]
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
            else:
                lines += ["| method | " + " | ".join(f"q2, n_train {s_}" for s_ in sizes) + " | "
                          + " | ".join(f"RMSE, n_train {s_}" for s_ in sizes) + " |",
                          "|---|" + "---:|" * (2 * len(sizes))]
                for m in methods:
                    if not all(m in by_size[s_] for s_ in sizes):
                        continue
                    cells = [f"{by_size[s_][m]['q2']:+.3f} ({by_size[s_][m]['q2_sd']:.3f})" for s_ in sizes]
                    cells += [f"{by_size[s_][m]['rmse']:.3f} ({by_size[s_][m]['rmse_sd']:.3f})" for s_ in sizes]
                    lines.append(f"| {m} | " + " | ".join(cells) + " |")
            lines.append("")
            base = "CNN features + ridge"
            key = "acc" if is_cls else "q2"
            lines += [f"Paired difference in {'accuracy' if is_cls else 'q2'} against *{base}* "
                      f"over the draws (mean, standard deviation in brackets, draws ahead):", "",
                      "| method | " + " | ".join(f"n_train {s_}" for s_ in sizes) + " |",
                      "|---|" + "---:|" * len(sizes)]
            for m in methods:
                if m == base or not all(m in by_size[s_] for s_ in sizes):
                    continue
                d = [by_size[s_][m]["against"].get(base) for s_ in sizes]
                if any(x is None for x in d):
                    continue
                lines.append(f"| {m} | " + " | ".join(
                    f"{x[key]:+.3f} ({x[key + '_sd']:.3f}), {x['wins']} of {x['draws']}" for x in d) + " |")
            lines.append("")
    return lines


def report(out: str = "") -> str:
    pilot = _load_results()
    gate = _load_results(GATE.prefix())
    if not pilot and not gate:
        raise SystemExit("no results under benchmarks/results/microstructure")
    lines = ["# A microstructure as the sample: field-overlap kernels against two-point "
             "statistics", "",
             "Synthetic two-phase microstructures on a periodic voxel grid (Gaussian-filtered "
             "noise cut at the phase fraction; the phase fraction, the grain size, the aspect "
             "ratio and the direction of the grains vary from sample to sample). The label is "
             "the effective thermal conductivity of the periodic cell from a finite-volume "
             "solver (phase 0 at 1, phase 1 at the contrast): the component along x, which "
             "depends on the direction of the grains, and the mean over the directions, which "
             "does not. Training sets of each size drawn at random, the rest of the set held "
             "out; every setting chosen by five-fold cross-validation inside the training part. "
             "q2 is R2 on the held-out samples, mean over the draws with the standard deviation "
             "over the draws in brackets; RMSE is in units of the conductivity of phase 0.", ""]
    if gate:
        lines += ["# The gate: equal budgets, twenty draws, the Gaussian-process baseline", "",
                  "Every method chooses among the same number of settings (40) in the inner "
                  "cross-validation, except the Gaussian process on the principal components, "
                  "which chooses the number of components (five choices) and fits its length "
                  "scales, amplitude and noise by the marginal likelihood. Twenty draws per "
                  "training size. The paired tables give the difference between a method and "
                  "a baseline on the same draws.", ""]
        for r in gate:
            lines += _block_lines(r, BASELINES + KERNEL_METHODS, show_paired=True)
    extra = _twoscale_lines() + _map_lines() + _uhcs_lines()
    if extra:
        lines += ["# The other three tests of the gate", ""] + extra
    if pilot:
        lines += ["# The pilot: five draws, 140 settings per kernel method", ""]
        for r in pilot:
            lines += _block_lines(r, METHODS, show_paired=False)
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
    ap.add_argument("--stage", choices=("data", "fit", "twoscale", "map", "uhcs"), default=None,
                    help="data and fit are the benchmark; twoscale, map and uhcs are items 3, "
                         "4 and 2 of the gate")
    ap.add_argument("--protocol", choices=tuple(PROTOCOLS), default="pilot",
                    help="pilot: 5 draws, 140 settings per kernel method; gate: 20 draws, 40 "
                         "settings for every method and the GP baseline")
    ap.add_argument("--jobs", type=int, default=1, help="processes over the draws")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="labels to fit (default xx and mean; C_xx and C_mean once "
                         "prepare_elastic.py has run); a label already in the result file "
                         "of the same protocol is kept, the new ones are added")
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
    if args.stage == "twoscale":
        run_twoscale(args.dim or 2, args.grid or 51, args.n or 400, args.contrast or CONTRAST,
                     protocol=PROTOCOLS[args.protocol], jobs=args.jobs, force=args.force)
        report()
        return 0
    if args.stage == "uhcs":
        run_uhcs(protocol=PROTOCOLS[args.protocol], jobs=args.jobs)
        report()
        return 0
    if args.stage == "map":
        run_map(args.dim or 2, args.grid or 51, args.n or 400, args.contrast or 100.0)
        report()
        return 0
    stages = (args.stage,) if args.stage else ("data", "fit")
    for dim, grid, n, contrast in sets:
        run_set(dim, grid, n, contrast, stages=stages, force=args.force,
                protocol=PROTOCOLS[args.protocol], jobs=args.jobs, label_names=args.labels)
    if "fit" in stages:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
