"""Atom functions with a direction: Gaussians times a linear factor (plan item D1).

Every field in this program is a sum of isotropic Gaussians on the nuclei, and
:mod:`cmfqsar.cdft` showed what that cannot do.  A Fukui function is essentially a
frontier-orbital density: on acrolein it is pi*, three quarters of its squared magnitude lies more
than 0.5 A off the molecular plane, and there is a node *in* the plane.  A function that is
positive everywhere and centred on a nucleus cannot make a node, and the fit loses half the field
(relative error 77 % on f+, 89 % on the dual descriptor).

The smallest extension that can make a lobe is to multiply the Gaussian by a linear factor.  With
the width convention this program uses, ``exp(-b |r - R|^2 / 2)``, the basis on atom `l` becomes ::

    s_l(r)   = exp(-b |r - R_l|^2 / 2)
    p_l^u(r) = (r_u - R_l,u) exp(-b |r - R_l|^2 / 2),      u = x, y, z

one s and three p functions per atom per width.  A p function is odd about its own centre, so it
*has* a node through the nucleus and a lobe on either side, which is the shape that was missing.
Three Cartesian components span every direction, so nothing has to be oriented along bonds by
hand: the fit chooses the direction, and a lobe along a bond, along a lone pair or perpendicular
to a ring plane are all the same three numbers in a different combination.

The overlaps stay closed-form, which is the whole reason this is the extension to try.  With
``p = a + b``, ``mu = ab/p``, ``d = B - A`` and the s|s overlap
``S = (2 pi / p)^{3/2} exp(-mu |d|^2 / 2)``:

    <s_A | s_B>     = S
    <p_A^u | s_B>   = (b/p) d_u S
    <s_A | p_B^v>   = -(a/p) d_v S
    <p_A^u | p_B^v> = [delta_uv / p - (ab/p^2) d_u d_v] S

:func:`pair_overlaps` evaluates all four over whole arrays of centres, and
``tests/test_anisotropic.py`` holds every one of them against numerical quadrature, which is how
they were derived.  A kernel between two fields is then still a double sum over pairs of atoms and
pairs of basis functions -- four times as many terms per atom pair, with the same cost per term --
so the co-field, the atom contributions and the training-set decomposition all survive.

What this module does not do is decide whether the extension is worth its cost.
``benchmarks/run_anisotropic.py`` measures that the way the plan item asks: the fit error of the
conceptual-DFT fields on the four corrosion series that have ORCA results, with and without the p
functions, and then whether a model built on the resulting kernels notices the difference.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: how many basis functions one width contributes per atom: s, px, py, pz
PER_WIDTH = 4


def basis_size(n_widths: int, p_type: bool = True) -> int:
    """Coefficients per atom for `n_widths` widths, with or without the p functions."""
    return int(n_widths) * (PER_WIDTH if p_type else 1)


def design(points: np.ndarray, positions: np.ndarray, widths: np.ndarray,
           p_type: bool = True) -> np.ndarray:
    """The basis evaluated at `points`: (n_points, n_atoms, n_widths, 1 or 4).

    The last axis is the s function and then the three p components, in the order x, y, z.  It is
    kept as a separate axis rather than flattened so that a caller can drop the p block and get
    exactly the isotropic basis back.
    """
    P = np.asarray(points, float)
    R = np.asarray(positions, float)
    w = np.asarray(widths, float)
    delta = P[:, None, :] - R[None, :, :]                       # (n, A, 3)
    d2 = np.einsum("nau,nau->na", delta, delta)                 # (n, A)
    g = np.exp(-0.5 * w[None, None, :] * d2[:, :, None])        # (n, A, T)
    if not p_type:
        return g[:, :, :, None]
    out = np.empty(g.shape + (PER_WIDTH,))
    out[:, :, :, 0] = g
    for u in range(3):
        out[:, :, :, u + 1] = g * delta[:, :, u][:, :, None]
    return out


def pair_overlaps(A: np.ndarray, B: np.ndarray, a: float, b: float,
                  p_type: bool = True) -> np.ndarray:
    """Overlap integrals between the basis on centres `A` and the basis on centres `B`.

    Returns (n_A, n_B, k, k) with k = 4 (s, px, py, pz) or 1 (s only): entry [l, m, i, j] is
    ``<f_i on A_l | f_j on B_m>`` for Gaussians of width `a` on A and `b` on B.  Derived from the
    Gaussian product rule and checked against quadrature in the tests.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    p = float(a) + float(b)
    mu = float(a) * float(b) / p
    d = B[None, :, :] - A[:, None, :]                           # (A, B, 3)
    d2 = np.einsum("abu,abu->ab", d, d)
    S = np.exp(-0.5 * mu * d2) * (2.0 * np.pi / p) ** 1.5       # (A, B)
    if not p_type:
        return S[:, :, None, None]
    k = PER_WIDTH
    out = np.zeros(S.shape + (k, k))
    out[:, :, 0, 0] = S
    for u in range(3):
        out[:, :, u + 1, 0] = (float(b) / p) * d[:, :, u] * S
        out[:, :, 0, u + 1] = -(float(a) / p) * d[:, :, u] * S
    coef = float(a) * float(b) / (p * p)
    for u in range(3):
        for v in range(3):
            term = -coef * d[:, :, u] * d[:, :, v]
            if u == v:
                term = term + 1.0 / p
            out[:, :, u + 1, v + 1] = term * S
    return out


def field_overlap(coeff_a: np.ndarray, pos_a: np.ndarray, coeff_b: np.ndarray,
                  pos_b: np.ndarray, widths: np.ndarray, p_type: bool = True) -> float:
    """``int F_A(r) F_B(r) dr`` for two fitted fields, in closed form.

    The coefficients are (n_atoms, n_widths, k) with k as in :func:`design`.  The sum runs over
    pairs of atoms, pairs of widths and pairs of basis functions; the atom pairs are the expensive
    axis and are vectorised, the rest is a loop over at most 4 x 4 x n_widths^2 scalars.
    """
    ca = np.asarray(coeff_a, float)
    cb = np.asarray(coeff_b, float)
    w = np.asarray(widths, float)
    total = 0.0
    for t1, a in enumerate(w):
        for t2, b in enumerate(w):
            O = pair_overlaps(pos_a, pos_b, float(a), float(b), p_type=p_type)
            # sum_{l,m,i,j} c_a[l,t1,i] O[l,m,i,j] c_b[m,t2,j]
            total += float(np.einsum("li,lmij,mj->", ca[:, t1, :], O, cb[:, t2, :]))
    return total


def overlap_kernel(coeffs: Sequence[np.ndarray], positions: Sequence[np.ndarray],
                   widths: np.ndarray, p_type: bool = True) -> np.ndarray:
    """The matrix of :func:`field_overlap` between every pair of fitted fields."""
    n = len(coeffs)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            K[i, j] = K[j, i] = field_overlap(coeffs[i], positions[i], coeffs[j], positions[j],
                                              widths, p_type=p_type)
    return K


@dataclass
class AnisotropicFit:
    """A field fitted with s and p functions on every atom."""
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 0)))  # (A, T, k)
    widths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    p_type: bool = True
    rel_error: float = float("nan")      # ||F - fit|| / ||F|| over the fitted region
    r2: float = float("nan")
    name: str = ""

    @property
    def n_coefficients(self) -> int:
        return int(self.coeffs.size)

    def condensed(self) -> np.ndarray:
        """What each atom's share of the fitted field integrates to.

        Only the s functions contribute: a p function is odd about its own centre and integrates
        to zero, which is the point of it -- it moves density from one side of the nucleus to the
        other without changing how much there is.
        """
        return self.coeffs[:, :, 0] @ (2.0 * np.pi / self.widths) ** 1.5

    def dipoles(self) -> np.ndarray:
        """(n_atoms, 3): the first moment the p functions put on each atom.

        ``int (r - R_l) F_l(r) dr`` for atom `l`'s own part of the field, which is the direction
        and size of the lobe the fit chose there.  Zero for an isotropic fit, by construction.
        """
        if not self.p_type:
            return np.zeros((self.coeffs.shape[0], 3))
        w = np.asarray(self.widths, float)
        # int (r_u - R_u) p^u = (1/b)(2 pi / b)^{3/2}, and the s part contributes nothing
        scale = (1.0 / w) * (2.0 * np.pi / w) ** 1.5
        return np.einsum("atu,t->au", self.coeffs[:, :, 1:], scale)

    def evaluate(self, points: np.ndarray, positions: np.ndarray,
                 chunk: int = 100000) -> np.ndarray:
        """The fitted field at arbitrary points, for checking it against the grid it came from."""
        pts = np.asarray(points, float)
        out = np.empty(len(pts))
        for lo in range(0, len(pts), chunk):
            B = design(pts[lo:lo + chunk], positions, self.widths, p_type=self.p_type)
            out[lo:lo + chunk] = np.einsum("natk,atk->n", B, self.coeffs)
        return out


def fit_field(values: np.ndarray, points: np.ndarray, positions: np.ndarray,
              widths: np.ndarray, window: Optional[np.ndarray] = None,
              p_type: bool = True, ridge: float = 1e-6, name: str = "",
              chunk: int = 100000) -> AnisotropicFit:
    """Least squares with s and p functions on every atom, in chunks over the grid.

    The design matrix of a 20-atom molecule on an 80^3 grid with four widths and four functions
    is 160 million entries; what is accumulated instead is the 320 by 320 normal matrix.  `ridge`
    is relative to the mean diagonal, because the p blocks of nearby atoms are nearly collinear
    and the normal matrix is close to singular without it.
    """
    pts = np.asarray(points, float)
    pos = np.asarray(positions, float)
    w = np.asarray(widths, float)
    target = np.asarray(values, float).reshape(-1)
    if window is not None:
        target = target * np.asarray(window, float).reshape(-1)
    k = PER_WIDTH if p_type else 1
    n_col = len(pos) * len(w) * k
    AtA = np.zeros((n_col, n_col))
    Aty = np.zeros(n_col)
    yty = 0.0
    n_rows = 0
    for lo in range(0, len(pts), chunk):
        P, y = pts[lo:lo + chunk], target[lo:lo + chunk]
        B = design(P, pos, w, p_type=p_type).reshape(len(P), n_col)
        AtA += B.T @ B
        Aty += B.T @ y
        yty += float(y @ y)
        n_rows += len(P)
    scale = max(np.trace(AtA) / max(n_col, 1), 1e-30)
    c = np.linalg.solve(AtA + ridge * scale * np.eye(n_col), Aty)
    resid2 = max(yty - 2.0 * float(c @ Aty) + float(c @ AtA @ c), 0.0)
    rel = float(np.sqrt(resid2 / yty)) if yty > 0 else float("nan")
    mean = target.mean()
    ss_tot = yty - n_rows * mean ** 2
    r2 = float(1.0 - resid2 / ss_tot) if ss_tot > 0 else float("nan")
    return AnisotropicFit(c.reshape(len(pos), len(w), k), w, p_type, rel, r2, name)


def fit_cube(values: np.ndarray, cube, positions: Optional[np.ndarray] = None,
             window: Optional[np.ndarray] = None, n_terms: int = 4, p_type: bool = True,
             ridge: float = 1e-6, name: str = "", chunk: int = 100000) -> AnisotropicFit:
    """:func:`fit_field` on a :class:`cmfqsar.cdft.Cube`, with the widths the probe fields share."""
    from .probes import basis_widths
    pos = cube.positions if positions is None else np.asarray(positions, float)
    return fit_field(values, cube.points(), pos, basis_widths(n_terms), window=window,
                     p_type=p_type, ridge=ridge, name=name, chunk=chunk)
