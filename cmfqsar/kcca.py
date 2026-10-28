"""Kernel canonical correlation analysis: what one view of the molecules sees that another does
not (plan item F6).

A *view* is one description of the same molecules, which in this program means one kernel over
them.  The program builds several: the five-field CMF kernel, the quantum fields, the ECFP4
Tanimoto kernel, the pocket complementarity, the same fields over a conformer ensemble.  No tool
here says how much two of them say the same thing, and, more useful, **which part of the field
is the part a second view does not contain**.

Canonical correlation analysis (CCA) answers that question.  Given two descriptions of the same
objects it finds the pair of directions, one in each description, whose projections correlate
most strongly; then the next such pair, uncorrelated with the first; and so on.  The kernel form
(KCCA) does it with kernels in place of coordinates, which is what CMF needs because its
description of a molecule has no coordinate vector.  With centred kernels ``K`` and ``L`` on the
same ``n`` molecules and dual directions ``a`` and ``b`` (one coefficient per molecule, like the
coefficients of a fitted model), the pairs solve the generalised eigenproblem ::

    [ 0    K L ] [a]         [ K^2 + c_K K      0      ] [a]
    [ L K   0  ] [b] = rho   [     0       L^2 + c_L L ] [b]

where ``rho`` is the canonical correlation and ``c_K``, ``c_L`` are ridges.  **The ridge is not
an option.**  With ``c = 0`` and invertible kernels every canonical correlation is exactly 1 and
the directions are arbitrary -- any direction of one view can be matched exactly by some
direction of the other, because both have as many directions as there are molecules.  That is the
classical failure of KCCA, and it is why nothing here is judged by its training correlations:
the ridge is chosen by the correlations on **held-out** molecules, and the correlations are
compared with the ones a **shuffled pairing** of the two views produces.

The eigenproblem is solved through the spectra of the two kernels.  With ``K = U diag(k) U^T``
and ``L = V diag(l) V^T`` the substitution ``a = U diag(k (k + c_K))^(-1/2) p`` turns it into
the singular value decomposition of ::

    M = diag( sqrt(k / (k + c_K)) )  U^T V  diag( sqrt(l / (l + c_L)) )

whose singular values are the canonical correlations and whose singular vectors give the
directions.  The factor ``sqrt(k / (k + c))`` is what the ridge does: a direction along which
the molecules barely vary (small ``k``) is shrunk towards nothing instead of being inflated to
unit variance, and ``||K a_i||^2``, which is 1 without a ridge, says how much of each direction
the ridge left.  That number is reported for every component as its **variance share**.

Two properties make this a CMF tool rather than a general one.  A canonical direction on the
field side is a vector of dual coefficients, the same object as a fitted model's, so
:func:`component_model` turns a component into a model whose co-field
(:meth:`cmfqsar.model.CMFModel.cofield_coefficients`) is the picture of that direction, with no
new interpretation code and with rule 2 of the plan kept: the direction is linear in the field.
And when both views are field kernels over the same fields, the two co-fields of one component
are functions on one space and :func:`component_cosines` compares them by the cosine of plan
item F3.

**The degenerate case, stated first because it is the obvious misuse.**  With the activity as the
second view that view has one dimension, and the first canonical direction is the kernel ridge
regression of the activity on the first view at ridge ``c_K`` -- ``a = (K + c_K I)^-1 y``, up to
a scale factor.  "KCCA-QSAR" in that sense is kernel ridge regression under another name and
predicts nothing new.  The test suite uses the identity as a correctness check and nothing else.

**What the field sees that the other view does not.**  A direction of the field view counts
as *contained* in the other view when that view explains more than half of its variance on
held-out molecules (:func:`contained_directions`).  The field kernel is then split in two:
:func:`projected_kernel` restricts it to the span of the contained directions, and the
difference is the kernel of what is left, the part of the field that is the field's own.  The
split is a projection and not a regression on the canonical variates, because the variates are
whitened by construction and a model on them has lost the variance along each direction, which
is the prior kernel ridge regression rests on; the projected kernels keep it.  Kernel ridge
regression on the two parts, with the regularisation chosen inside the folds
(:func:`activity_on_kernel`), answers the question the item asks, with the bootstrap interval of
the difference; and the model on the field's own part has dual coefficients
(:func:`private_coefficients`), so where that part carries the activity is a co-field.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["KCCAFit", "KCCAResult", "HeldOut", "RidgeChoice", "activity_on_kernel", "analyse",
           "centre", "centre_against", "choose_ridge", "component_cosines", "component_model",
           "contained_directions", "cosine_summary", "held_out_correlations", "kcca",
           "paired_difference", "permutation_null", "private_coefficients", "projected_kernel"]

#: the ridge grid, as multiples of the mean eigenvalue of the centred kernel
RIDGE_GRID = (1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
#: eigenvalues below this share of the largest are not directions but round-off
RCOND = 1e-8
#: regularisation grid of the activity models, as multiples of the mean kernel diagonal
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
#: share of a direction's variance the other view must explain for the direction to count as
#: contained in it (a held-out correlation above sqrt(1/2) = 0.71)
CONTAINED_SHARE = 0.5


# --------------------------------------------------------------------------- #
# Centring
# --------------------------------------------------------------------------- #


def centre(K: np.ndarray) -> np.ndarray:
    """``H K H`` with ``H = I - 11^T / n``: the feature space centred at the mean molecule."""
    K = np.asarray(K, float)
    m = K.mean(axis=0)
    return K - m[None, :] - m[:, None] + K.mean()


def centre_against(K_test_train: np.ndarray, K_train: np.ndarray) -> np.ndarray:
    """The block between new molecules and the training set, centred with the training means.

    A new molecule is placed in the feature space the training set was centred in, so the
    column means and the grand mean are the training set's, and only the row mean is its own.
    """
    K_tt = np.asarray(K_test_train, float)
    K_tr = np.asarray(K_train, float)
    return K_tt - K_tt.mean(axis=1)[:, None] - K_tr.mean(axis=0)[None, :] + K_tr.mean()


def _spectrum(Kc: np.ndarray, rcond: float, max_rank: Optional[int]):
    """Eigenvectors and eigenvalues of a centred kernel, largest first, round-off removed."""
    w, U = np.linalg.eigh(0.5 * (Kc + Kc.T))
    order = np.argsort(w)[::-1]
    w, U = w[order], U[:, order]
    keep = w > max(float(rcond) * max(float(w[0]), 0.0), 0.0)
    if max_rank is not None:
        keep &= np.arange(len(w)) < int(max_rank)
    return U[:, keep], w[keep]


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #


@dataclass
class KCCAFit:
    """The canonical directions of two views on one set of molecules.

    ``a`` and ``b`` hold one dual direction per column, in the order of ``rho`` (largest
    first); ``u = K_c a`` and ``v = L_c b`` are the canonical variates of the training
    molecules.  ``variance_a[i] = ||K_c a_i||^2`` lies in (0, 1] and says how much of the
    direction the ridge left; without a ridge it is 1 for every direction.
    """
    rho: np.ndarray
    a: np.ndarray
    b: np.ndarray
    u: np.ndarray
    v: np.ndarray
    variance_a: np.ndarray
    variance_b: np.ndarray
    ridge: float                      # the relative ridge asked for
    ridge_a: float                    # the absolute ridges it became, one per view
    ridge_b: float
    rank_a: int
    rank_b: int
    col_mean_a: np.ndarray = _field(repr=False, default=None)
    col_mean_b: np.ndarray = _field(repr=False, default=None)
    grand_a: float = 0.0
    grand_b: float = 0.0

    @property
    def n_components(self) -> int:
        return int(len(self.rho))

    def _centre_test(self, K_test_train, col_mean, grand):
        K_tt = np.asarray(K_test_train, float)
        return K_tt - K_tt.mean(axis=1)[:, None] - col_mean[None, :] + grand

    def project_a(self, K_test_train: np.ndarray) -> np.ndarray:
        """Canonical variates of new molecules in view A, from their kernel block to the training set."""
        return self._centre_test(K_test_train, self.col_mean_a, self.grand_a) @ self.a

    def project_b(self, L_test_train: np.ndarray) -> np.ndarray:
        """The same for view B."""
        return self._centre_test(L_test_train, self.col_mean_b, self.grand_b) @ self.b

    def shared_variance(self) -> np.ndarray:
        """Variance of each direction of view A that view B follows: ``v_i rho_i^2``."""
        return self.variance_a * self.rho ** 2

    def private_variance(self) -> np.ndarray:
        """Variance of each direction of view A that view B does not follow: ``v_i (1 - rho_i^2)``."""
        return self.variance_a * (1.0 - self.rho ** 2)


def kcca(K: np.ndarray, L: np.ndarray, ridge: float = 0.1, rcond: float = RCOND,
         max_rank: Optional[int] = None) -> KCCAFit:
    """Regularised kernel canonical correlation analysis of two kernels on the same molecules.

    `K` and `L` are the raw (uncentred) kernels; they are centred here.  `ridge` is relative:
    each view's absolute ridge is ``ridge`` times the mean eigenvalue of its centred kernel, so
    that the same number means the same thing for a kernel of overlap integrals and for a
    Tanimoto coefficient.  `rcond` drops the eigenvalues that are round-off and `max_rank`, if
    given, keeps only that many leading eigenvectors of each view (a cheap second regulariser,
    not used by default).
    """
    K = np.asarray(K, float)
    L = np.asarray(L, float)
    n = K.shape[0]
    if K.shape != (n, n) or L.shape != (n, n):
        raise ValueError(f"the two kernels must be square and of the same size: {K.shape} "
                         f"against {L.shape}")
    if n < 3:
        raise ValueError("canonical correlation needs at least three molecules")
    Kc, Lc = centre(K), centre(L)
    UA, k = _spectrum(Kc, rcond, max_rank)
    UB, l = _spectrum(Lc, rcond, max_rank)
    if len(k) == 0 or len(l) == 0:
        raise ValueError("one of the kernels has no variance over these molecules")
    cA = float(ridge) * float(np.trace(Kc)) / n
    cB = float(ridge) * float(np.trace(Lc)) / n
    wa = np.sqrt(k / (k + cA))
    wb = np.sqrt(l / (l + cB))
    M = (wa[:, None] * (UA.T @ UB)) * wb[None, :]
    P, rho, Qt = np.linalg.svd(M, full_matrices=False)
    Q = Qt.T
    rho = np.clip(rho, 0.0, 1.0)
    a = UA @ (P / np.sqrt(k * (k + cA))[:, None])
    b = UB @ (Q / np.sqrt(l * (l + cB))[:, None])
    var_a = ((k / (k + cA))[:, None] * P ** 2).sum(axis=0)
    var_b = ((l / (l + cB))[:, None] * Q ** 2).sum(axis=0)
    return KCCAFit(rho=rho, a=a, b=b, u=Kc @ a, v=Lc @ b, variance_a=var_a, variance_b=var_b,
                   ridge=float(ridge), ridge_a=cA, ridge_b=cB, rank_a=int(len(k)),
                   rank_b=int(len(l)), col_mean_a=K.mean(axis=0), col_mean_b=L.mean(axis=0),
                   grand_a=float(K.mean()), grand_b=float(L.mean()))


# --------------------------------------------------------------------------- #
# Held-out correlations, the ridge, and the shuffled pairing
# --------------------------------------------------------------------------- #


def _folds(n: int, folds: int, seed: Optional[int]) -> List[np.ndarray]:
    idx = np.arange(n)
    if seed is not None:
        np.random.RandomState(seed).shuffle(idx)
    k = max(2, min(int(folds), n))
    return [idx[i::k] for i in range(k)]


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-300 else float("nan")


@dataclass
class HeldOut:
    """Canonical correlations measured on molecules the directions were not fitted on."""
    per_fold: np.ndarray            # folds x components, nan where a fold had fewer components
    ridge: float

    @property
    def mean(self) -> np.ndarray:
        """The held-out correlation of each component, averaged over the folds."""
        with np.errstate(all="ignore"):
            return np.nanmean(self.per_fold, axis=0)

    @property
    def n_components(self) -> int:
        return int(self.per_fold.shape[1])


def held_out_correlations(K: np.ndarray, L: np.ndarray, ridge: float, folds: int = 5,
                          seed: Optional[int] = 0, components: Optional[int] = None,
                          rcond: float = RCOND, max_rank: Optional[int] = None) -> HeldOut:
    """Fit the directions on four fifths of the molecules, correlate the variates on the fifth.

    This is the number KCCA has to be judged by.  The training correlations approach 1 as the
    ridge goes to zero whatever the two views contain; the held-out correlation does not, and
    falls when the directions were fitted to the particular molecules rather than to what the
    views share.  The correlation is Pearson's, per fold, averaged over the folds.
    """
    K = np.asarray(K, float)
    L = np.asarray(L, float)
    n = K.shape[0]
    parts = _folds(n, folds, seed)
    rows = []
    for test in parts:
        if len(test) < 3:
            continue
        train = np.setdiff1d(np.arange(n), test)
        fit = kcca(K[np.ix_(train, train)], L[np.ix_(train, train)], ridge, rcond, max_rank)
        U = fit.project_a(K[np.ix_(test, train)])
        V = fit.project_b(L[np.ix_(test, train)])
        r = fit.n_components if components is None else min(fit.n_components, int(components))
        rows.append([_pearson(U[:, i], V[:, i]) for i in range(r)])
    if not rows:
        raise ValueError("too few molecules for held-out folds")
    width = max(len(r) for r in rows)
    out = np.full((len(rows), width), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return HeldOut(per_fold=out, ridge=float(ridge))


@dataclass
class RidgeChoice:
    """The ridge the held-out correlations chose, and what every candidate scored."""
    ridge: float
    scores: Dict[float, float]       # candidate -> mean held-out correlation of the leading components
    components: int

    def describe(self) -> str:
        cells = ", ".join(f"{c:g}: {s:+.3f}" for c, s in self.scores.items())
        return (f"ridge {self.ridge:g} (times the mean eigenvalue), chosen by the mean held-out "
                f"correlation over {self.components} pairs; candidates {cells}")


def choose_ridge(K: np.ndarray, L: np.ndarray, grid: Sequence[float] = RIDGE_GRID,
                 folds: int = 5, seed: Optional[int] = 0, components: Optional[int] = None,
                 rcond: float = RCOND) -> RidgeChoice:
    """The ridge with the highest mean held-out correlation over the pairs.

    Over all pairs by default, so that a ridge too small to stop the deep components fitting
    noise is told apart from one that lets them hold out; `components` restricts the mean to
    the leading pairs.  Never the training correlation: that is highest at the smallest ridge of
    any grid.
    """
    scores: Dict[float, float] = {}
    width = 0
    for c in grid:
        ho = held_out_correlations(K, L, float(c), folds=folds, seed=seed, components=components,
                                   rcond=rcond)
        m = ho.mean
        width = max(width, len(m))
        scores[float(c)] = float(np.nanmean(m)) if np.isfinite(m).any() else float("-inf")
    best = max(scores, key=lambda c: scores[c])
    return RidgeChoice(ridge=best, scores=scores, components=width)


def permutation_null(K: np.ndarray, L: np.ndarray, ridge: float, permutations: int = 200,
                     folds: int = 5, seed: Optional[int] = 0, components: Optional[int] = None,
                     rcond: float = RCOND) -> np.ndarray:
    """Held-out correlations when the pairing between the two views is shuffled.

    The molecules of view B are permuted, so every molecule's field is paired with another
    molecule's fingerprint, and the analysis is repeated at the same ridge on the same folds.
    What comes back, `permutations` rows of `components` values, is what the leading held-out
    correlations look like when the two views share nothing but their marginal distributions.
    A real correlation has to clear this; one that does not is the degeneracy of the method,
    not chemistry.
    """
    K = np.asarray(K, float)
    L = np.asarray(L, float)
    n = K.shape[0]
    rng = np.random.RandomState(seed)
    rows = []
    for r in range(int(permutations)):
        p = rng.permutation(n)
        ho = held_out_correlations(K, L[np.ix_(p, p)], ridge, folds=folds, seed=seed,
                                   components=components, rcond=rcond)
        rows.append(ho.mean)
    width = max(len(m) for m in rows)
    out = np.full((len(rows), width), np.nan)
    for r, m in enumerate(rows):
        out[r, :len(m)] = m
    return out


# --------------------------------------------------------------------------- #
# What the field sees that the other view does not
# --------------------------------------------------------------------------- #


def contained_directions(fit: KCCAFit, held_out: HeldOut, null: Optional[np.ndarray] = None,
                         share: float = CONTAINED_SHARE, quantile: float = 0.95) -> np.ndarray:
    """Which directions of view A the other view contains: a boolean per component.

    A direction counts as contained when the other view explains more than `share` of its
    variance on held-out molecules (a held-out correlation above ``sqrt(share)``) and, when the
    shuffled pairing is given, that correlation is also above its `quantile`.  A component the
    folds could not produce (the full fit has a few more than a fold's) is not contained.
    """
    r = fit.n_components
    out = np.zeros(r, bool)
    m = held_out.mean
    k = min(r, len(m))
    thr = float(np.sqrt(share))
    ok = np.isfinite(m[:k]) & (m[:k] > thr)
    if null is not None and null.size:
        with np.errstate(all="ignore"):
            q = np.nanquantile(null, quantile, axis=0)
        kk = min(k, len(q))
        ok[:kk] &= ~np.isfinite(q[:kk]) | (m[:kk] > q[:kk])
    out[:k] = ok
    return out


def projected_kernel(fit: KCCAFit, index: Sequence[int]) -> np.ndarray:
    """The centred kernel of view A restricted to the span of the directions `index`.

    With ``E = Phi^T A_S`` the feature-space vectors of the directions, the projector onto their
    span is ``E (E^T E)^-1 E^T`` and the kernel of the projected features is ::

        K_S = (K_c A_S) (A_S^T K_c A_S)^-1 (K_c A_S)^T = U_S G^-1 U_S^T

    which keeps the variance of the molecules along those directions as it is.  That is the
    point of projecting rather than regressing on the variates: the canonical variates are
    whitened by construction, so a regression on them has no way to prefer the directions the
    molecules actually vary along, which is the prior kernel ridge regression rests on.
    ``K_c - K_S`` is the kernel of what is left, the part of the field outside the span.
    """
    idx = np.asarray(list(index), int)
    n = fit.a.shape[0]
    if idx.size == 0:
        return np.zeros((n, n))
    U = fit.u[:, idx]
    G = fit.a[:, idx].T @ U
    G = 0.5 * (G + G.T)
    return U @ np.linalg.solve(G + 1e-12 * np.trace(G) / len(idx) * np.eye(len(idx)), U.T)


def _krr_cv(K: np.ndarray, y: np.ndarray, lam: float, folds: Sequence[np.ndarray]) -> np.ndarray:
    from .model import krr_fit

    n = len(y)
    ycv = np.full(n, np.nan)
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        a, b = krr_fit(K[np.ix_(train, train)], y[train], float(lam), fit_bias=True)
        ycv[test] = K[np.ix_(test, train)] @ a + b
    return ycv


def _scaled_lams(K: np.ndarray, lams: Sequence[float]) -> List[float]:
    """The regularisation grid as multiples of the mean diagonal, so that a kernel that holds a
    tenth of the field is regularised like the whole field would be."""
    d = float(np.trace(K)) / K.shape[0]
    d = d if d > 1e-300 else 1.0
    return [float(l) * d for l in lams]


def activity_on_kernel(K: np.ndarray, y: np.ndarray, folds: int = 5, seed: Optional[int] = 0,
                       lams: Sequence[float] = LAMS) -> Tuple[float, np.ndarray, float]:
    """Nested cross-validated q2 of kernel ridge regression on a kernel, with the predictions.

    The regularisation is chosen inside every outer fold by an inner cross-validation on the
    training part, so the q2 is not helped by the choice.  `lams` are multiples of the mean
    diagonal of the kernel.  Returns ``(q2, y_cv, lam)`` where `lam` is the regularisation an
    inner cross-validation on the whole set picks, for a final model.
    """
    from .model import r2_score

    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    if float(np.trace(K)) <= 1e-12 * n:
        return float("nan"), np.full(n, np.nan), float("nan")
    grid = _scaled_lams(K, lams)
    outer = _folds(n, folds, seed)
    ycv = np.full(n, np.nan)
    for test in outer:
        train = np.setdiff1d(np.arange(n), test)
        inner = _folds(len(train), folds, None if seed is None else seed + 1)
        Kt, yt = K[np.ix_(train, train)], y[train]
        best = None
        for lam in grid:
            pred = _krr_cv(Kt, yt, lam, inner)
            err = float(np.mean((pred - yt) ** 2))
            if best is None or err < best[0]:
                best = (err, lam)
        ycv[test] = _one_fold(K, y, train, test, best[1])
    best_all = None
    for lam in grid:
        pred = _krr_cv(K, y, lam, outer)
        err = float(np.mean((pred - y) ** 2))
        if best_all is None or err < best_all[0]:
            best_all = (err, lam)
    return float(r2_score(y, ycv)), ycv, float(best_all[1])


def _one_fold(K, y, train, test, lam):
    from .model import krr_fit

    a, b = krr_fit(K[np.ix_(train, train)], y[train], float(lam), fit_bias=True)
    return K[np.ix_(test, train)] @ a + b


def private_coefficients(fit: KCCAFit, contained: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Dual coefficients over the training molecules of a model fitted on the private kernel.

    A model on ``K_P = K_c - K_S`` predicts ``f(x) = phi(x)^T (I - P_S) Phi_c^T beta``, which is
    linear in the field with the weight vector ``Phi_c^T (beta - A_S G^-1 U_S^T beta)``; the
    centring makes the coefficients sum to zero.  Returned as coefficients on the raw training
    Gaussians, so :func:`component_model` turns the private part of the field into a co-field.
    """
    idx = np.flatnonzero(np.asarray(contained, bool))
    beta = np.asarray(beta, float)
    if idx.size:
        U = fit.u[:, idx]
        G = fit.a[:, idx].T @ U
        G = 0.5 * (G + G.T)
        beta = beta - fit.a[:, idx] @ np.linalg.solve(
            G + 1e-12 * np.trace(G) / len(idx) * np.eye(len(idx)), U.T @ beta)
    return beta - beta.mean()


def paired_difference(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, runs: int = 2000,
                      seed: Optional[int] = 0, level: float = 0.95) -> Tuple[float, float, float]:
    """``q2(pred_a) - q2(pred_b)`` with its bootstrap interval over the molecules (rule 4).

    The two predictions are resampled together, molecule by molecule, so the interval is that
    of the difference and not of two separate numbers.
    """
    from .model import r2_score

    y = np.asarray(y, float)
    pa, pb = np.asarray(pred_a, float), np.asarray(pred_b, float)
    ok = np.isfinite(y) & np.isfinite(pa) & np.isfinite(pb)
    y, pa, pb = y[ok], pa[ok], pb[ok]
    n = len(y)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    d = float(r2_score(y, pa) - r2_score(y, pb))
    rng = np.random.RandomState(seed)
    out = np.empty(int(runs))
    for r in range(int(runs)):
        i = rng.randint(0, n, n)
        if np.ptp(y[i]) <= 0:
            out[r] = 0.0
            continue
        out[r] = r2_score(y[i], pa[i]) - r2_score(y[i], pb[i])
    lo = float(np.quantile(out, (1.0 - level) / 2.0))
    hi = float(np.quantile(out, 1.0 - (1.0 - level) / 2.0))
    return d, lo, hi


# --------------------------------------------------------------------------- #
# Everything together
# --------------------------------------------------------------------------- #


@dataclass
class KCCAResult:
    """The analysis of two views: the fit, what held out, what shuffling gave, and the gates."""
    fit: KCCAFit
    held_out: HeldOut
    ridge_choice: Optional[RidgeChoice]
    null: Optional[np.ndarray]                   # permutations x components, or None
    components: int                              # leading components the tables show
    n: int
    contained: Optional[np.ndarray] = None       # per component: does view B contain it
    # gate (ii), when an activity was given
    q2_all: float = float("nan")
    q2_shared: float = float("nan")
    q2_private: float = float("nan")
    q2_difference: Tuple[float, float, float] = (float("nan"), float("nan"), float("nan"))
    ycv_all: Optional[np.ndarray] = None
    ycv_shared: Optional[np.ndarray] = None
    ycv_private: Optional[np.ndarray] = None
    private_coef: Optional[np.ndarray] = None    # dual coefficients of the private-part model
    private_lam: float = float("nan")

    # ---- the numbers of the components
    def null_quantile(self, q: float = 0.95) -> np.ndarray:
        """The `q` quantile of the shuffled held-out correlation, per component."""
        if self.null is None:
            return np.full(self.held_out.n_components, np.nan)
        with np.errstate(all="ignore"):
            return np.nanquantile(self.null, q, axis=0)

    def p_values(self) -> np.ndarray:
        """Share of shuffles whose held-out correlation reached the observed one, per component."""
        obs = self.held_out.mean
        out = np.full(len(obs), np.nan)
        if self.null is None:
            return out
        for i in range(min(len(obs), self.null.shape[1])):
            col = self.null[:, i]
            col = col[np.isfinite(col)]
            if len(col) and np.isfinite(obs[i]):
                out[i] = (1.0 + float((col >= obs[i]).sum())) / (1.0 + len(col))
        return out

    def real_components(self, q: float = 0.95) -> int:
        """How many components hold out above the `q` quantile of the shuffled pairing."""
        obs = self.held_out.mean
        thr = self.null_quantile(q)
        m = min(len(obs), len(thr))
        return int(np.sum(np.isfinite(obs[:m]) & (obs[:m] > thr[:m])))

    def n_contained(self) -> int:
        return int(self.contained.sum()) if self.contained is not None else 0

    def rows(self, components: Optional[int] = None) -> List[Dict[str, float]]:
        """One record per leading component, for a table or a file."""
        obs = self.held_out.mean
        thr = self.null_quantile()
        p = self.p_values()
        k = self.components if components is None else int(components)
        out = []
        for i in range(min(k, self.fit.n_components)):
            out.append({"component": i + 1, "rho_train": float(self.fit.rho[i]),
                        "rho_held_out": float(obs[i]) if i < len(obs) else float("nan"),
                        "null_95": float(thr[i]) if i < len(thr) else float("nan"),
                        "p": float(p[i]) if i < len(p) else float("nan"),
                        "variance_a": float(self.fit.variance_a[i]),
                        "variance_b": float(self.fit.variance_b[i]),
                        "contained": bool(self.contained[i]) if self.contained is not None
                        else False})
        return out

    def describe(self) -> str:
        r = self.fit.n_components
        lines = [f"{self.n} molecules; view A has {self.fit.rank_a} directions, view B "
                 f"{self.fit.rank_b}; {r} canonical pairs, {self.held_out.n_components} of them "
                 f"measured on held-out molecules."]
        if self.ridge_choice is not None:
            lines.append("  " + self.ridge_choice.describe())
        else:
            lines.append(f"  ridge {self.fit.ridge:g} (times the mean eigenvalue), as given")
        lines.append("")
        lines.append("  component   rho (fit)   rho (held out)   shuffled 95 %     p    variance "
                     "share A    B   contained")
        for row in self.rows():
            null = f"{row['null_95']:+.3f}" if np.isfinite(row["null_95"]) else "   -  "
            p = f"{row['p']:.3f}" if np.isfinite(row["p"]) else "  -  "
            lines.append(f"  {row['component']:9d}   {row['rho_train']:+.3f}       "
                         f"{row['rho_held_out']:+.3f}            {null}      {p}         "
                         f"{row['variance_a']:.2f}  {row['variance_b']:.2f}   "
                         f"{'yes' if row['contained'] else 'no'}")
        held = self.held_out.mean
        lines.append("")
        lines.append(f"  mean held-out correlation over all {len(held)} pairs: "
                     f"{float(np.nanmean(held)):+.3f}")
        if self.null is not None:
            k = self.real_components()
            lines.append(f"  (i) {k} of {len(held)} components hold out above the shuffled "
                         f"pairing" + ("" if k else ": nothing the two views share survives "
                                             "holding molecules out"))
        if self.contained is not None:
            nc = self.n_contained()
            lines.append(f"      view B contains {nc} of the {r} directions of view A (more than "
                         f"{CONTAINED_SHARE:.0%} of the direction's variance explained on held-out "
                         f"molecules); {r - nc} directions are view A's own")
        if np.isfinite(self.q2_shared) or np.isfinite(self.q2_private):
            d, lo, hi = self.q2_difference
            lines.append(f"  (ii) activity by kernel ridge regression: on the whole of view A "
                         f"q2 {self.q2_all:+.3f}; on the part view B contains {self.q2_shared:+.3f};"
                         f" on the part it does not {self.q2_private:+.3f}; own minus contained "
                         f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
            if np.isfinite(lo) and lo > 0:
                lines.append("       the part of view A that view B does not contain carries more "
                             "of the activity")
            elif np.isfinite(hi) and hi < 0:
                lines.append("       the part of view A that view B contains carries more of the "
                             "activity; what A adds to B is not where the activity is")
            else:
                lines.append("       the interval covers zero: not a result (rule 4)")
        return "\n".join(lines)


def analyse(K: np.ndarray, L: np.ndarray, y: Optional[np.ndarray] = None,
            ridge: Optional[float] = None, components: int = 3, folds: int = 5,
            seed: Optional[int] = 0, permutations: int = 200,
            grid: Sequence[float] = RIDGE_GRID, rcond: float = RCOND,
            share: float = CONTAINED_SHARE, progress=None) -> KCCAResult:
    """The whole analysis of two views: ridge, held-out correlations, shuffled pairing, gates.

    `ridge` of None chooses it on the folds by the mean held-out correlation over all pairs;
    `permutations` of 0 skips the shuffled pairing; `y` of None skips gate (ii).  `components`
    is how many leading pairs the tables show.  `share` is the share of a direction's variance
    the other view has to explain on held-out molecules for the direction to count as contained.
    """
    K = np.asarray(K, float)
    L = np.asarray(L, float)
    n = K.shape[0]
    choice = None
    if ridge is None:
        if progress:
            progress("choosing the ridge")
        choice = choose_ridge(K, L, grid=grid, folds=folds, seed=seed, components=None,
                              rcond=rcond)
        ridge = choice.ridge
    if progress:
        progress("held-out correlations")
    ho = held_out_correlations(K, L, float(ridge), folds=folds, seed=seed, components=None,
                               rcond=rcond)
    null = None
    if permutations and permutations > 0:
        if progress:
            progress(f"{permutations} shuffled pairings")
        null = permutation_null(K, L, float(ridge), permutations=permutations, folds=folds,
                                seed=seed, components=None, rcond=rcond)
    fit = kcca(K, L, float(ridge), rcond=rcond)
    res = KCCAResult(fit=fit, held_out=ho, ridge_choice=choice, null=null,
                     components=int(components), n=n)
    res.contained = contained_directions(fit, ho, null, share=share)
    if y is not None:
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        if ok.sum() >= 6:
            if progress:
                progress("activity on the contained and on the private part of view A")
            Kc = centre(K)
            K_S = projected_kernel(fit, np.flatnonzero(res.contained))
            K_P = Kc - K_S
            sub = np.ix_(ok, ok)
            q_a, cv_a, _lam_a = activity_on_kernel(Kc[sub], y[ok], folds=folds, seed=seed)
            q_s, cv_s, _lam_s = activity_on_kernel(K_S[sub], y[ok], folds=folds, seed=seed)
            q_p, cv_p, lam_p = activity_on_kernel(K_P[sub], y[ok], folds=folds, seed=seed)
            for name, cv in (("ycv_all", cv_a), ("ycv_shared", cv_s), ("ycv_private", cv_p)):
                full = np.full(n, np.nan)
                full[ok] = cv
                setattr(res, name, full)
            res.q2_all, res.q2_shared, res.q2_private = q_a, q_s, q_p
            if np.isfinite(q_s) and np.isfinite(q_p):
                res.q2_difference = paired_difference(y[ok], cv_p, cv_s, seed=seed)
            if np.isfinite(lam_p):
                from .model import krr_fit
                beta = np.zeros(n)
                beta[ok], _b = krr_fit(K_P[sub], y[ok], lam_p, fit_bias=True)
                res.private_coef = private_coefficients(fit, res.contained, beta)
                res.private_lam = lam_p
    return res


# --------------------------------------------------------------------------- #
# Components as co-fields
# --------------------------------------------------------------------------- #


def component_model(probe, direction: np.ndarray, variate: Optional[np.ndarray] = None):
    """A canonical direction of a field view as a model, so that it has a co-field.

    `probe` is a :class:`cmfqsar.model.CMFModel` whose training kernels were prepared on the
    molecules of the view (``prepare_training_kernels``); the result shares its field set,
    normalisation and kernels and has the direction as its dual coefficients, a bias of zero and
    the canonical variate as its "activity".  Everything that reads a linear model -- the
    co-field export, the atom contributions, the F3 cosine -- works on it unchanged, and it can
    be saved with :func:`cmfqsar.dataio.save_model` and drawn with ``cmf cofield``.
    """
    from .model import ModelStats

    if probe.train is None or probe.K_fields_train is None:
        raise ValueError("the probe model has no training kernels; call "
                         "prepare_training_kernels on the molecules of the view first")
    if not probe.uses_fields or not probe.is_linear:
        raise ValueError("a component has a co-field only on a linear field kernel")
    a = np.asarray(direction, float).reshape(-1)
    if len(a) != probe.train.n_mol:
        raise ValueError(f"the direction has {len(a)} coefficients for {probe.train.n_mol} "
                         "molecules")
    m = probe.fresh_copy()
    m.method = "krr"
    m.set_training_kernels(probe.train, probe.K_fields_train, probe.self_diag)
    m.coef = a.copy()
    m.bias = 0.0
    m.y = (np.asarray(variate, float).copy() if variate is not None
           else np.asarray(m.joint_kernel(m.K_fields_train, probe.train.descriptors), float) @ a)
    m.stats = ModelStats(n_train=int(probe.train.n_mol))
    return m


def component_cosines(models_a: Sequence, models_b: Sequence) -> np.ndarray:
    """The F3 cosine between the co-fields of the components of two field views.

    Entry ``(i, j)`` is the cosine between component ``i`` of view A and component ``j`` of
    view B, over all fields together.  For two views on the same fields in one frame the
    diagonal is what gate (iii) of the item reads: the two co-fields of one shared component
    should resemble each other more than the co-fields of two different components do.
    """
    from .transfer import cofield_overlap

    M = np.full((len(models_a), len(models_b)), np.nan)
    for i, ma in enumerate(models_a):
        for j, mb in enumerate(models_b):
            res = cofield_overlap(ma, mb, f"A{i + 1}", f"B{j + 1}", strict=False)
            M[i, j] = res.cosine
    return M


def cosine_summary(M: np.ndarray) -> Tuple[float, float]:
    """Mean signed cosine on the diagonal and mean absolute cosine off it."""
    M = np.asarray(M, float)
    r = min(M.shape)
    diag = np.array([M[i, i] for i in range(r)], float)
    off = np.array([abs(M[i, j]) for i in range(M.shape[0]) for j in range(M.shape[1]) if i != j],
                   float)
    return (float(np.nanmean(diag)) if np.isfinite(diag).any() else float("nan"),
            float(np.nanmean(off)) if np.isfinite(off).any() else float("nan"))
