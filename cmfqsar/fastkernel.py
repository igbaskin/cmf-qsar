"""A compiled, threaded atom-pair loop for the field kernels (plan item B5).

The NumPy path in :mod:`cmfqsar.kernels` evaluates one molecule against a table as a dense
``(atoms_of_i, atoms_of_Q)`` matrix and then makes one pass over that matrix per field: an
exponential, a scaling, two multiplications and a segment sum each.  On a 40-atom molecule against
200 molecules that matrix holds 318 000 entries, the block costs 6.7 ms, and about 2 ms of that is
passes over memory rather than arithmetic -- 25 MB of traffic for 2.5 MB of data.

What a compiled loop can and cannot do about it
-----------------------------------------------
It cannot beat NumPy at the exponentials, which are half the time.  ``np.exp`` on an array is
vectorised; the same calls written one at a time in a compiled loop are not, unless Intel's SVML
is installed, and on the machine these numbers come from it is not.  Measured on 200 molecules:
the NumPy path takes 0.67 s, the same arithmetic in a serial compiled loop takes 0.90 s.  **A
compiled kernel is not faster than NumPy for this.**

What it can do is use the other cores.  The NumPy path is single-threaded, and a kernel is a sum
over independent molecule pairs, so the loop over the molecules of the first table is a
``prange``: every iteration writes its own row of the output and nothing is shared.  On 32 cores
the same 0.67 s becomes 0.08 s.  That is the whole value of this module, and it is why it is
written as one fused loop rather than as a translation of the NumPy code: having paid for a
thread, it should not also be making ten passes over memory.

It also gets the cutoff of plan item B2 almost for free.  Dropping a distant atom pair costs one
comparison inside the loop, with no neighbour list to build, so it is useful on exactly the sets
where the NumPy neighbour list is not worth building.

Using it
--------
Numba is an optional dependency: ``pip install numba`` or ``pip install -e .[fast]``.  Without it
:func:`available` is false and everything falls back to NumPy, which stays the reference -- it is
what every table in this repository was produced with, and ``tests/test_fastkernel.py`` holds the
two paths to agreeing.  The compiled loop is compiled with ``fastmath``, so it may reorder a sum;
the measured difference from the NumPy path is 4e-15 relative, which is the same order as the
round-off between two orderings of the NumPy sum itself.

Probe fields (a basis of several Gaussians per atom) are not implemented here and fall back.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

try:                                            # pragma: no cover - depends on the installation
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:                               # pragma: no cover - numba is optional
    HAVE_NUMBA = False

    def njit(*a, **k):                          # type: ignore[misc]
        def wrap(fn):
            return fn
        return wrap

    prange = range                              # type: ignore[assignment]

#: below this many atom pairs the threads cost more than they save
MIN_PAIRS = 100_000
PI = np.pi


def available() -> bool:
    """True when the compiled path can be used at all."""
    return HAVE_NUMBA


def threads() -> int:
    """How many threads the compiled loop will use."""
    if not HAVE_NUMBA:
        return 1
    import numba
    return int(numba.get_num_threads())


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _kernel(p_coords, p_weights, p_radii, p_bounds,
            q_coords, q_weights, q_radii, q_bounds,
            g_alphas, g_offsets, g_fields, g_cut2,
            s_fields, s_alphas, s_cut2, symmetric, out):   # pragma: no cover - compiled
    """Per-field kernels of every molecule of P against every molecule of Q.

    `out` is (n_fields, n_mol_p, n_mol_q) and is written, not added to.  With `symmetric` only the
    pairs with j >= i are evaluated and the caller mirrors them.  The outer loop is threaded:
    iteration i writes only row i of `out`, so there is nothing to synchronise.  A squared cutoff
    of zero means no cutoff for that group of fields.
    """
    n_p = p_bounds.shape[0]
    n_q = q_bounds.shape[0]
    n_g = g_alphas.shape[0]
    n_s = s_fields.shape[0]
    for im in prange(n_p):
        ilo = p_bounds[im, 0]
        ihi = p_bounds[im, 1]
        first = im if symmetric else 0
        for jm in range(first, n_q):
            jlo = q_bounds[jm, 0]
            jhi = q_bounds[jm, 1]
            for b in range(jlo, jhi):
                bx = q_coords[b, 0]
                by = q_coords[b, 1]
                bz = q_coords[b, 2]
                rb = q_radii[b]
                for a in range(ilo, ihi):
                    dx = p_coords[a, 0] - bx
                    dy = p_coords[a, 1] - by
                    dz = p_coords[a, 2] - bz
                    d2 = dx * dx + dy * dy + dz * dz
                    for g in range(n_g):
                        if g_cut2[g] > 0.0 and d2 > g_cut2[g]:
                            continue
                        al = g_alphas[g]
                        e = np.exp(-0.25 * al * d2) * (PI / al) ** 1.5
                        for k in range(g_offsets[g], g_offsets[g + 1]):
                            f = g_fields[k]
                            out[f, im, jm] += p_weights[a, f] * e * q_weights[b, f]
                    for s in range(n_s):
                        if s_cut2[s] > 0.0 and d2 > s_cut2[s]:
                            continue
                        f = s_fields[s]
                        al = s_alphas[s]
                        b_i = al / (p_radii[a] * p_radii[a])
                        b_q = al / (rb * rb)
                        bs = b_i + b_q
                        z = 2.0 * PI / bs
                        e = np.exp(-(b_i * b_q) / (2.0 * bs) * d2) * (z * np.sqrt(z))
                        out[f, im, jm] += p_weights[a, f] * e * q_weights[b, f]
    return out


@njit(cache=True, nogil=True, parallel=True, fastmath=True)
def _defect_kernels(p_frac, p_weights, p_radii, p_bounds,
                    q_frac, q_weights, q_radii, q_bounds,
                    rot, trans, cell,
                    g_alphas, g_offsets, g_fields, g_cut2,
                    s_fields, s_alphas, s_cut2,
                    mixing, dp, dq, gammas, symmetric,
                    out_lin, out_lay):   # pragma: no cover - compiled
    """The periodic overlap of difference fields averaged over the host symmetry (item M14).

    For every pair of structures and every operation ``g`` of the host, the per-field overlap
    ``K_f(P_i, g Q_j)`` is summed over the atom pairs with the minimum-image displacement in
    the shared `cell` (fractional coordinates in, Cartesian distance out), and two things are
    accumulated: its mean over the operations per field (`out_lin`, shape (F, nP, nQ), the
    linear average) and the mean of the Gaussian layer
    ``exp(-gamma (dp_i + dq_j - 2 sum_f mixing_f K_f))`` for every gamma (`out_lay`, shape
    (n_gamma, nP, nQ)).  With `symmetric` only ``j >= i`` is evaluated and the caller mirrors.
    The outer loop is threaded; iteration i writes only row i of both outputs.
    """
    n_p = p_bounds.shape[0]
    n_q = q_bounds.shape[0]
    n_g = g_alphas.shape[0]
    n_s = s_fields.shape[0]
    n_f = mixing.shape[0]
    n_k = gammas.shape[0]
    G = rot.shape[0]
    c00 = cell[0, 0]
    c01 = cell[0, 1]
    c02 = cell[0, 2]
    c10 = cell[1, 0]
    c11 = cell[1, 1]
    c12 = cell[1, 2]
    c20 = cell[2, 0]
    c21 = cell[2, 1]
    c22 = cell[2, 2]
    for im in prange(n_p):
        ilo = p_bounds[im, 0]
        ihi = p_bounds[im, 1]
        kf = np.zeros(n_f)
        first = im if symmetric else 0
        for jm in range(first, n_q):
            jlo = q_bounds[jm, 0]
            jhi = q_bounds[jm, 1]
            base = dp[im] + dq[jm]
            for g in range(G):
                for f in range(n_f):
                    kf[f] = 0.0
                for b in range(jlo, jhi):
                    q0 = q_frac[b, 0]
                    q1 = q_frac[b, 1]
                    q2 = q_frac[b, 2]
                    f0 = rot[g, 0, 0] * q0 + rot[g, 0, 1] * q1 + rot[g, 0, 2] * q2 + trans[g, 0]
                    f1 = rot[g, 1, 0] * q0 + rot[g, 1, 1] * q1 + rot[g, 1, 2] * q2 + trans[g, 1]
                    f2 = rot[g, 2, 0] * q0 + rot[g, 2, 1] * q1 + rot[g, 2, 2] * q2 + trans[g, 2]
                    rb = q_radii[b]
                    for a in range(ilo, ihi):
                        d0 = p_frac[a, 0] - f0
                        d1 = p_frac[a, 1] - f1
                        d2f = p_frac[a, 2] - f2
                        d0 -= np.floor(d0 + 0.5)
                        d1 -= np.floor(d1 + 0.5)
                        d2f -= np.floor(d2f + 0.5)
                        dx = d0 * c00 + d1 * c10 + d2f * c20
                        dy = d0 * c01 + d1 * c11 + d2f * c21
                        dz = d0 * c02 + d1 * c12 + d2f * c22
                        d2 = dx * dx + dy * dy + dz * dz
                        for gg in range(n_g):
                            if g_cut2[gg] > 0.0 and d2 > g_cut2[gg]:
                                continue
                            al = g_alphas[gg]
                            e = np.exp(-0.25 * al * d2) * (PI / al) ** 1.5
                            for k in range(g_offsets[gg], g_offsets[gg + 1]):
                                f = g_fields[k]
                                kf[f] += p_weights[a, f] * e * q_weights[b, f]
                        for s in range(n_s):
                            if s_cut2[s] > 0.0 and d2 > s_cut2[s]:
                                continue
                            f = s_fields[s]
                            al = s_alphas[s]
                            b_i = al / (p_radii[a] * p_radii[a])
                            b_q = al / (rb * rb)
                            bs = b_i + b_q
                            z = 2.0 * PI / bs
                            e = np.exp(-(b_i * b_q) / (2.0 * bs) * d2) * (z * np.sqrt(z))
                            kf[f] += p_weights[a, f] * e * q_weights[b, f]
                joint = 0.0
                for f in range(n_f):
                    out_lin[f, im, jm] += kf[f]
                    joint += mixing[f] * kf[f]
                dist2 = base - 2.0 * joint
                if dist2 < 0.0:
                    dist2 = 0.0
                for k in range(n_k):
                    out_lay[k, im, jm] += np.exp(-gammas[k] * dist2)
            for f in range(n_f):
                out_lin[f, im, jm] /= G
            for k in range(n_k):
                out_lay[k, im, jm] /= G
    return out_lay


def defect_kernels(P, Q, specs: Sequence, cell: np.ndarray, rotations: np.ndarray,
                   translations: np.ndarray, mixing: np.ndarray, dp: np.ndarray, dq: np.ndarray,
                   gammas: Sequence[float], cutoff=None,
                   symmetric: bool = False) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """``(K_lin, K_layer)`` of :func:`_defect_kernels` for two tables of difference fields that
    stand in one `cell`, or None when the compiled loop cannot take the field set.

    `rotations` and `translations` are in fractional coordinates; `mixing` are the per-field
    coefficients of the joint kernel; `dp` and `dq` the joint self-kernels of the two tables;
    `cutoff` a radius in angstroms, which should not exceed half the cell's smallest width,
    because the displacement is taken to the nearest image only.  The caller mirrors the result
    when `symmetric`.
    """
    layout = plan(specs, cutoff=cutoff)
    if layout is None:
        return None
    alphas, offsets, fields, g_cut2, s_fields, s_alphas, s_cut2 = layout
    A = _c(np.asarray(cell, float).reshape(3, 3))
    inv = np.linalg.inv(A)
    F = len(specs)
    # the loop indexes these by field and by structure without a bounds check
    if len(mixing) != F or len(dp) != P.n_mol or len(dq) != Q.n_mol:
        raise ValueError(f"defect_kernels: mixing has {len(mixing)} entries for {F} fields, dp "
                         f"{len(dp)} for {P.n_mol} structures, dq {len(dq)} for {Q.n_mol}")
    gammas = _c(np.asarray(list(gammas), float).reshape(-1))
    out_lin = np.zeros((F, P.n_mol, Q.n_mol))
    out_lay = np.zeros((len(gammas), P.n_mol, Q.n_mol))
    _defect_kernels(_c(P.coords @ inv), _c(P.weights), _c(P.radii), bounds_of(P.offsets),
                    _c(Q.coords @ inv), _c(Q.weights), _c(Q.radii), bounds_of(Q.offsets),
                    _c(np.asarray(rotations, float).reshape(-1, 3, 3)),
                    _c(np.asarray(translations, float).reshape(-1, 3)), A,
                    alphas, offsets, fields, g_cut2, s_fields, s_alphas, s_cut2,
                    _c(mixing), _c(dp), _c(dq), gammas, bool(symmetric), out_lin, out_lay)
    return out_lin, out_lay


def plan(specs: Sequence, cutoff=None, tol: float = 1e-6) -> Optional[Tuple]:
    """The field layout the compiled loop needs, or None when it cannot take this field set.

    A probe field, or a field of a kind the loop does not implement, makes the whole call fall
    back to NumPy rather than being handled halfway.  `cutoff` is a radius in angstroms or None;
    unlike the NumPy path there is no "auto", because inside the loop a cutoff costs one
    comparison per pair and needs no neighbour list, so there is nothing to decide.
    """
    if not HAVE_NUMBA:
        return None
    gauss, steric = {}, []
    for f, s in enumerate(specs):
        if getattr(s, "is_probe", False):
            return None
        if s.kind == "gauss":
            gauss.setdefault(float(s.alpha), []).append(f)
        elif s.kind == "steric":
            steric.append((f, float(s.alpha)))
        else:
            return None
    alphas = np.asarray(sorted(gauss), dtype=np.float64)
    fields: List[int] = []
    offsets = [0]
    for a in alphas:
        fields.extend(gauss[float(a)])
        offsets.append(len(fields))
    cut2 = 0.0 if cutoff is None else float(cutoff) ** 2
    return (alphas, np.asarray(offsets, dtype=np.int64), np.asarray(fields, dtype=np.int64),
            np.full(len(alphas), cut2),
            np.asarray([f for f, _a in steric], dtype=np.int64),
            np.asarray([a for _f, a in steric], dtype=np.float64),
            np.full(len(steric), cut2))


def bounds_of(offsets: np.ndarray, first_mol: int = 0) -> np.ndarray:
    """(n_mol, 2) first and last atom index of every molecule from `first_mol` onwards."""
    off = np.asarray(offsets, dtype=np.int64)
    a0 = int(off[first_mol])
    lo = off[first_mol:-1] - a0
    hi = off[first_mol + 1:] - a0
    return np.ascontiguousarray(np.stack([lo, hi], axis=1))


def _c(a, dtype=np.float64):
    return np.ascontiguousarray(a, dtype=dtype)


def kernel_matrices(P, Q, specs: Sequence, symmetric: bool = False,
                    cutoff=None, out: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """(F, nP, nQ) per-field kernels by the compiled loop, or None when it cannot be used.

    The caller mirrors the result when `symmetric`; this writes only the upper triangle then.
    """
    layout = plan(specs, cutoff=cutoff)
    if layout is None:
        return None
    if P.coords.shape[0] * Q.coords.shape[0] < MIN_PAIRS:
        return None
    alphas, offsets, fields, g_cut2, s_fields, s_alphas, s_cut2 = layout
    F = len(specs)
    if out is None:
        out = np.zeros((F, P.n_mol, Q.n_mol))
    _kernel(_c(P.coords), _c(P.weights), _c(P.radii), bounds_of(P.offsets),
            _c(Q.coords), _c(Q.weights), _c(Q.radii), bounds_of(Q.offsets),
            alphas, offsets, fields, g_cut2, s_fields, s_alphas, s_cut2,
            bool(symmetric), out)
    return out


def warm_up() -> bool:
    """Compile the loop on a two-atom problem, so the first real call is not the compilation.

    Returns False when Numba is not installed.  The compiled function is cached on disk, so this
    costs about a second once per machine and nothing afterwards.
    """
    if not HAVE_NUMBA:
        return False
    x = np.zeros((1, 3))
    w = np.ones((1, 2))
    r = np.ones(1)
    b = np.array([[0, 1]], dtype=np.int64)
    out = np.zeros((2, 1, 1))
    _kernel(x, w, r, b, x, w, r, b,
            np.array([0.5]), np.array([0, 1], dtype=np.int64), np.array([0], dtype=np.int64),
            np.zeros(1), np.array([1], dtype=np.int64), np.array([0.5]), np.zeros(1),
            False, out)
    return True
