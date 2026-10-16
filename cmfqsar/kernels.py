"""
Molecular-field kernels.

For two atoms with Gaussian field contributions
    rho_l(r) = w_l exp(-b_l/2 |r - r_l|^2),   rho_m(r) = w_m exp(-b_m/2 |r - r_m|^2)
the overlap integral over R^3 is analytic:

    k(l, m) = w_l w_m (2 pi / (b_l + b_m))^{3/2} exp(-(b_l b_m) / (2 (b_l + b_m)) |r_l - r_m|^2)

For b_l = b_m = alpha this reduces to Eq. 5 of Baskin & Zhokhova (2013):
    k = w_l w_m (pi/alpha)^{3/2} exp(-alpha/4 |r_l - r_m|^2)
and for the radius-scaled steric field (b = alpha / R_vdw^2) to Eq. 11.

The kernel of molecules i, j for field f is the sum over all atom pairs
(Eq. 2); the joint kernel is the h_f-weighted sum over fields (Eq. 1).
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist

from .fields import AtomTable, FieldSpec


class Cancelled(Exception):
    pass


def _group_fields(specs: Sequence[FieldSpec]):
    gauss = {}
    steric = []
    for f, s in enumerate(specs):
        if s.kind == "gauss":
            gauss.setdefault(float(s.alpha), []).append(f)
        else:
            steric.append((f, float(s.alpha)))
    return gauss, steric


def atom_kernel_block(xi: np.ndarray, wi: np.ndarray, ri: np.ndarray, Q: AtomTable,
                      specs: Sequence[FieldSpec]) -> np.ndarray:
    """Kernels between the atoms of one molecule and every molecule of table Q.

    Returns an array of shape (F, Ni, nQ): the contribution of every atom of
    the probe molecule to the field kernels with every molecule in Q.
    """
    F = len(specs)
    Ni = len(xi)
    out = np.zeros((F, Ni, Q.n_mol))
    D2 = cdist(xi, Q.coords, "sqeuclidean")           # (Ni, AQ)
    starts = Q.offsets[:-1]
    gauss, steric = _group_fields(specs)
    for alpha, fids in gauss.items():
        E = np.exp(-0.25 * alpha * D2)
        E *= (np.pi / alpha) ** 1.5
        for f in fids:
            M = (wi[:, f][:, None] * E) * Q.weights[:, f][None, :]   # (Ni, AQ)
            out[f] = np.add.reduceat(M, starts, axis=1)
    for f, alpha in steric:
        bi = alpha / ri ** 2
        bq = alpha / Q.radii ** 2
        bs = bi[:, None] + bq[None, :]
        E = np.exp(-(bi[:, None] * bq[None, :]) / (2.0 * bs) * D2)
        E *= (2.0 * np.pi / bs) ** 1.5
        M = (wi[:, f][:, None] * E) * Q.weights[:, f][None, :]
        out[f] = np.add.reduceat(M, starts, axis=1)
    return out


def kernel_matrices(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec],
                    progress: Optional[Callable[[int, int], None]] = None,
                    cancel: Optional[Callable[[], bool]] = None) -> np.ndarray:
    """Per-field kernel matrices K_f(i, j) for molecules i in P and j in Q.

    Returns an array of shape (F, nP, nQ).
    """
    F = len(specs)
    K = np.zeros((F, P.n_mol, Q.n_mol))
    for i in range(P.n_mol):
        if cancel is not None and cancel():
            raise Cancelled()
        sl = P.slice(i)
        blk = atom_kernel_block(P.coords[sl], P.weights[sl], P.radii[sl], Q, specs)
        K[:, i, :] = blk.sum(axis=1)
        if progress is not None and (i % 10 == 0 or i == P.n_mol - 1):
            progress(i + 1, P.n_mol)
    return K


def self_kernels(P: AtomTable, specs: Sequence[FieldSpec]) -> np.ndarray:
    """K_f(i, i) for every molecule i of P. Shape (F, nP)."""
    F = len(specs)
    d = np.zeros((F, P.n_mol))
    for i in range(P.n_mol):
        sub = P.subset([i])
        blk = atom_kernel_block(sub.coords, sub.weights, sub.radii, sub, specs)
        d[:, i] = blk.sum(axis=1)[:, 0]
    return d


def normalize_kernels(K: np.ndarray, dP: np.ndarray, dQ: np.ndarray) -> np.ndarray:
    """Cosine-normalised kernels K_f(i,j) / sqrt(K_f(i,i) K_f(j,j)); zero where undefined."""
    Kn = np.zeros_like(K)
    for f in range(K.shape[0]):
        sp = np.sqrt(np.maximum(dP[f], 0.0))
        sq = np.sqrt(np.maximum(dQ[f], 0.0))
        denom = sp[:, None] * sq[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            Kn[f] = np.where(denom > 1e-300, K[f] / denom, 0.0)
    return Kn


def combine_kernels(K: np.ndarray, h: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Joint kernel K = sum_f h_f * scale_f * K_f."""
    return np.tensordot(np.asarray(h, dtype=float) * np.asarray(scale, dtype=float), K, axes=(0, 0))


# --------------------------------------------------------------------------- #
# Reference implementation (numerical quadrature) used in the unit tests
# --------------------------------------------------------------------------- #

def numerical_overlap(x1, w1, b1, x2, w2, b2, spacing=0.1, extent=6.0) -> float:
    """Numerically integrate the product of two Gaussian atom fields (slow; for tests)."""
    lo = np.minimum(x1, x2) - extent
    hi = np.maximum(x1, x2) + extent
    axes = [np.arange(lo[k], hi[k], spacing) for k in range(3)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    f1 = w1 * np.exp(-0.5 * b1 * ((g - x1) ** 2).sum(1))
    f2 = w2 * np.exp(-0.5 * b2 * ((g - x2) ** 2).sum(1))
    return float((f1 * f2).sum() * spacing ** 3)


# --------------------------------------------------------------------------- #
# Derivatives of the field kernels with respect to the probe atom positions
# --------------------------------------------------------------------------- #

def kernel_gradient(xi: np.ndarray, wi: np.ndarray, ri: np.ndarray, Q: AtomTable,
                    specs: Sequence[FieldSpec], u: np.ndarray, chunk: int = 4000) -> np.ndarray:
    """Gradient of ``sum_f sum_j u[f, j] K_f(i, j)`` with respect to the atom positions of i.

    The kernels are sums of Gaussians in the squared interatomic distances, so the
    derivative is a weighted sum of the same Gaussians times the interatomic vectors:

        d/dr_il  w_fil w_fjm g(|r_il - r_jm|^2)  =  2 g'(d^2) w_fil w_fjm (r_il - r_jm)

    with ``g'/g = -alpha_f/4`` for ordinary fields and ``-b_il b_jm / (2 (b_il + b_jm))``
    for the radius-scaled steric field.

    Parameters
    ----------
    u : array of shape (F, nQ), the multiplier of every field/molecule pair.

    Returns
    -------
    array of shape (Ni, 3)
    """
    grad = np.zeros((len(xi), 3))
    u_atom = np.repeat(np.asarray(u, dtype=float), np.diff(Q.offsets), axis=1)   # (F, AQ)
    gauss, steric = _group_fields(specs)
    for s0 in range(0, len(Q.coords), chunk):
        sl = slice(s0, min(s0 + chunk, len(Q.coords)))
        diff = xi[:, None, :] - Q.coords[None, sl, :]            # (Ni, M, 3)
        D2 = np.einsum("lmk,lmk->lm", diff, diff)
        for alpha, fids in gauss.items():
            E = np.exp(-0.25 * alpha * D2) * (np.pi / alpha) ** 1.5
            for f in fids:
                M = (wi[:, f][:, None] * E) * (Q.weights[sl, f] * u_atom[f, sl])[None, :]
                grad += (-0.5 * alpha) * np.einsum("lm,lmk->lk", M, diff)
        for f, alpha in steric:
            bi = alpha / ri ** 2
            bq = alpha / Q.radii[sl] ** 2
            bs = bi[:, None] + bq[None, :]
            fac = (bi[:, None] * bq[None, :]) / bs
            E = np.exp(-0.5 * fac * D2) * (2.0 * np.pi / bs) ** 1.5
            M = (wi[:, f][:, None] * E) * (Q.weights[sl, f] * u_atom[f, sl])[None, :]
            grad += -np.einsum("lm,lmk->lk", M * fac, diff)
    return grad


def so3_right_jacobian(omega: np.ndarray) -> np.ndarray:
    """Right Jacobian of the SO(3) exponential map (identity at omega = 0)."""
    w = np.asarray(omega, dtype=float)
    th = float(np.linalg.norm(w))
    W = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th < 1e-8:
        return np.eye(3) - 0.5 * W
    a = (1.0 - np.cos(th)) / th ** 2
    b = (th - np.sin(th)) / th ** 3
    return np.eye(3) - a * W + b * (W @ W)
