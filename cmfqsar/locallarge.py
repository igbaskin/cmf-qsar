"""Many atoms at once: the local kernel without a kernel matrix (plan item LP5).

A per-atom target is a large problem where a per-molecule one is a small one.  Four hundred
molecules are five thousand carbons; the free NMR databases hold hundreds of thousands, and an
exact kernel over samples costs the square of their number to build and the cube to solve.  Plan
item LP5 asks for a Nystrom or inducing-point approximation to get past that.

For the rotation-invariant local kernel of :mod:`cmfqsar.localenv` no approximation is needed, and
that is what this module is.  The kernel there is a **linear kernel on a finite feature map**: the
field autocorrelation of an environment is a fixed number of channels -- fifteen field pairs times
seventeen distance bins is 255 columns for the five physico-chemical fields -- scaled per block and
then cosine-normalised, and the kernel is the inner product of those rows.  So

    K = Z Z^T,   with Z the scaled and normalised feature rows,

and the ridge regression can be solved in the **primal**, in the space of the 255 coefficients,
at a cost linear in the number of samples:

    (Z^T Z + lam I) w = Z^T y,

a d by d system whatever n is, against the n by n system the dual form solves.  The two give the
same predictions -- the tests check that they agree to round-off -- so this is not an
approximation but the same model computed the other way round.  What it changes is the cost:
memory goes from n^2 to n d, and time from n^3 to n d^2.

The overlap kernel of :mod:`cmfqsar.localalign` is a different matter.  Its feature map is the
field itself, which is a function on space and has no finite basis, so there the dual form is the
only exact one and a Nystrom approximation is the route; :func:`nystrom_kernel` builds one from a
set of landmark environments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .atomtargets import AtomTargets
from .invariant import channel_scale
from .localenv import LAMS, LocalResult
from .model import r2_score, rmse


def normalised_rows(features: np.ndarray, scale: Optional[np.ndarray] = None,
                    n_bins: int = 1, blocks: Optional[Sequence[int]] = None
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """The feature rows whose inner products are the kernel: (rows, the scale used).

    This is :func:`cmfqsar.invariant.autocorrelation_kernel` stopped one step early.  That
    function scales the channel blocks, cosine-normalises and multiplies; this returns the rows
    themselves, so that a model can be fitted in the space they live in.  Pass the `scale` back in
    to prepare test rows with the training set's scaling, as a kernel would.
    """
    X = np.asarray(features, float)
    if scale is None:
        scale = channel_scale(X, n_bins, blocks=blocks)
    Z = X * scale
    d = np.sqrt(np.maximum(np.sum(Z * Z, axis=1), 1e-300))
    return Z / d[:, None], np.asarray(scale, float)


def primal_fit(Z: np.ndarray, y: np.ndarray, lam: float,
               fit_bias: bool = True) -> Tuple[np.ndarray, float]:
    """Ridge regression in the feature space: (weights, bias).

    The bias is handled as the dual form handles it -- the fit is on centred rows and a centred
    target, which is the same model as the bordered system of :func:`cmfqsar.model.krr_fit` for a
    kernel that is an inner product.
    """
    Z = np.asarray(Z, float)
    y = np.asarray(y, float)
    n, d = Z.shape
    if fit_bias:
        zm, ym = Z.mean(axis=0), float(y.mean())
    else:
        zm, ym = np.zeros(d), 0.0
    Zc, yc = Z - zm, y - ym
    A = Zc.T @ Zc + float(lam) * np.eye(d)
    try:
        w = np.linalg.solve(A, Zc.T @ yc)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, Zc.T @ yc, rcond=None)[0]
    return w, float(ym - zm @ w)


def primal_predict(Z: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.asarray(Z, float) @ np.asarray(w, float) + float(b)


def cross_validate_primal(Z: np.ndarray, targets: AtomTargets, lam: float = 1e-3,
                          folds: int = 10, seed: Optional[int] = 0, name: str = "",
                          note: str = "") -> LocalResult:
    """Out-of-fold predictions in the primal, on folds that hold out whole molecules.

    The same folds, the same ridge and the same model as
    :func:`cmfqsar.localenv.cross_validate`; only the arithmetic differs.
    """
    y = np.asarray(targets.y, float)
    parts = targets.folds(folds, seed)
    if not parts:
        raise ValueError("there is nothing to cross-validate: no atom carries the target")
    Z = np.asarray(Z, float)
    pred = np.full(len(y), np.nan)
    for te in parts:
        tr = np.setdiff1d(np.arange(len(y)), te)
        if len(tr) < 2:
            continue
        w, b = primal_fit(Z[tr], y[tr], lam)
        pred[te] = primal_predict(Z[te], w, b)
    ok = np.isfinite(pred)
    return LocalResult(name=name or "primal", predicted=pred, q2=r2_score(y[ok], pred[ok]),
                       error=rmse(y[ok], pred[ok]),
                       mae=float(np.mean(np.abs(y[ok] - pred[ok]))), lam=float(lam),
                       n_samples=int(ok.sum()), n_molecules=targets.n_measured_molecules,
                       folds=len(parts), note=note,
                       molecule=np.asarray(targets.molecule, int))


def choose_lam_primal(Z: np.ndarray, targets: AtomTargets, lams: Sequence[float] = LAMS,
                      folds: int = 10, seed: Optional[int] = 0,
                      name: str = "") -> LocalResult:
    """The ridge that cross-validates best, in the primal.

    The ridge means the same thing in both forms for an inner-product kernel, so the value chosen
    here is the value the dual form would choose.
    """
    best: Optional[LocalResult] = None
    for lam in lams:
        r = cross_validate_primal(Z, targets, lam, folds, seed, name=name)
        if best is None or r.q2 > best.q2:
            best = r
    assert best is not None
    return best


@dataclass
class Curve:
    """What a model of this kind is worth at several training sizes."""
    sizes: List[int]
    molecules: List[int]
    q2: List[float]
    error: List[float]
    lam: List[float]
    seconds: List[float]

    def summary(self) -> str:
        return "; ".join(f"{m} molecules ({n} atoms): q2 {q:+.3f}, error {e:.2f}"
                         for m, n, q, e in zip(self.molecules, self.sizes, self.q2, self.error))


def learning_curve(Z: np.ndarray, targets: AtomTargets, sizes: Sequence[int],
                   lams: Sequence[float] = LAMS, folds: int = 10, seed: Optional[int] = 0,
                   progress=None) -> Curve:
    """Cross-validated error against the number of **molecules** in the training set.

    The sizes are numbers of molecules, not of atoms, because that is what a data set is counted
    in and because the folds hold out molecules.  Each size is a subset of the next, drawn once
    with the given seed, so the curve is nested and the differences along it are not the noise of
    redrawing the set.
    """
    import time

    rng = np.random.RandomState(0 if seed is None else int(seed))
    which = np.unique(np.asarray(targets.molecule, int))
    order = which[rng.permutation(len(which))]
    out = Curve([], [], [], [], [], [])
    for m in sizes:
        keep = set(int(x) for x in order[:int(m)])
        rows = np.array([r for r in range(len(targets))
                         if int(targets.molecule[r]) in keep], dtype=int)
        if len(rows) < 10:
            continue
        sub = targets.subset(rows)
        t0 = time.time()
        res = choose_lam_primal(Z[rows], sub, lams, folds, seed)
        out.sizes.append(len(rows))
        out.molecules.append(int(min(int(m), len(which))))
        out.q2.append(res.q2)
        out.error.append(res.error)
        out.lam.append(res.lam)
        out.seconds.append(time.time() - t0)
        if progress is not None:
            progress(f"{m} molecules: error {res.error:.2f}", 0.0)
    return out


def nystrom_kernel(K_xz: np.ndarray, K_zz: np.ndarray) -> np.ndarray:
    """The rank-m approximation K(X, Z) K(Z, Z)^+ K(Z, X), for a kernel with no finite features.

    This is the route plan item LP5 names, and it is needed for the **overlap** kernel of
    :mod:`cmfqsar.localalign`, whose feature map is a function on space.  For the invariant local
    kernel it is not needed at all: that kernel has 255 features and the primal form above is
    exact.  The pseudo-inverse is taken through a symmetric eigendecomposition, which is what
    keeps a landmark set with near-duplicate members from blowing the approximation up.
    """
    A = np.asarray(K_zz, float)
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    keep = w > max(float(w.max()), 0.0) * 1e-12
    inv = (V[:, keep] / w[keep]) @ V[:, keep].T
    B = np.asarray(K_xz, float)
    return B @ inv @ B.T
