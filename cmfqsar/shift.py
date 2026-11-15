"""Covariate shift: weights that make a training set look like a library, and what to do with them.

The two-sample test of :mod:`cmfqsar.mmd` says whether a library of molecules comes from the same
population as the training set.  When it does not, the test gives no remedy.  This module gives
three, all computed from the kernel matrix the program already has.

**The setting.**  *Covariate shift* means that the molecules of the library are distributed
differently from the training molecules, while the activity of a given molecule follows the same
rule in both.  Written with densities: p_lib(x) differs from p_train(x), and p(y | x) is the same.
The quantity that repairs everything is the *density ratio* ::

    w(x) = p_lib(x) / p_train(x).

A training molecule with a large w is typical of the library and rare in the training set, so it
should count more.  Every method below estimates w from kernel values only.

**Two ways to estimate the ratio.**

``kmm``
    Kernel mean matching (Huang, Smola, Gretton, Borgwardt and Schoelkopf, NIPS 2006).  Choose one
    weight per training molecule so that the weighted mean of their feature maps is as close as
    possible to the mean feature map of the library.  The mean feature map is the *kernel mean
    embedding* of a set; for the linear CMF kernel it is the mean field.  The weights are bounded
    by `B` and their mean is kept near one.  It gives weights at the training molecules only.

``ulsif``
    Unconstrained least-squares importance fitting (Kanamori, Hido and Sugiyama, JMLR 10 (2009)
    1391).  The ratio is written as a sum of kernel functions centred on library molecules,
    w(x) = sum_l theta_l k(x, c_l), and theta is the least-squares fit of the true ratio, which
    has a closed form.  It gives a weight at *any* molecule, which is what the weighted conformal
    interval needs: a weight for the molecule being predicted as well as for the calibration ones.

``klr``
    A probabilistic classifier (Bickel, Brueckner and Scheffer, JMLR 10 (2009) 2137).  Kernel
    logistic regression learns p(library | x) from the two sets; by Bayes' rule
    w(x) = exp(f(x)) n_train / n_library, where f is the latent function.  On the Sutherland sets
    it recovers a known ratio better than uLSIF (``benchmarks/run_shift.py``), because the
    logistic loss is not dominated by the few molecules with the largest weights.

Both need a kernel that can tell distributions apart, not only their means.  The linear field
kernel matches mean fields only; the default is therefore a Gaussian of the field distance,
k(x, x') = exp(-gamma |f_x - f_x'|^2), where |f_x - f_x'|^2 = K(x, x) + K(x', x') - 2 K(x, x') is
computed from the CMF kernel itself.  This is the "Gaussian layer" of the model, used here on the
two sets instead of on the activities.

**What the weights are used for.**

* *Weighted fitting.*  Kernel ridge regression that minimises sum_i w_i (y_i - f(x_i))^2 plus the
  usual penalty.  It is the same bordered system as Eq. 12 of the 2013 paper with the ridge of
  molecule i divided by w_i.  Weighted fitting helps only when the model cannot fit the activity
  everywhere at once; with a flexible kernel it often changes little.  The benchmark measures it.
* *Weighted conformal intervals* (Tibshirani, Barber, Candes and Ramdas, NeurIPS 2019).  The
  ordinary conformal quantile treats every calibration error as equally likely to resemble the
  error of a new molecule.  Under shift, errors of calibration molecules that look like the new
  molecule are more relevant.  The weighted rule takes the quantile of the calibration errors
  weighted by w, with an extra point at infinity that carries the weight of the new molecule.
  With the true ratio the coverage guarantee holds again.  A new molecule far from every
  calibration molecule gets an infinite interval, which is the correct answer: there is no
  evidence about its error.
* *Diagnostics.*  The effective sample size (sum w)^2 / sum w^2 says how many training molecules
  the weighted set is worth.  The weighted MMD says how much of the shift the weights removed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["ClassifierRatio", "DensityRatio", "KMMResult", "classifier_ratio", "WeightedConformal", "distance2", "effective_size",
           "gaussian_layer", "kmm", "median_gamma", "shifted_conformal", "ulsif", "ulsif_crossfit",
           "weighted_calibrate",
           "weighted_cv_predict", "weighted_krr_fit", "weighted_mmd2", "weighted_quantile"]


# --------------------------------------------------------------------------- #
# The kernel used to compare the two sets
# --------------------------------------------------------------------------- #

def distance2(Kxy: np.ndarray, dx: Sequence[float], dy: Sequence[float]) -> np.ndarray:
    """Squared field distance |f_x - f_y|^2 = K(x, x) + K(y, y) - 2 K(x, y), never negative."""
    Kxy = np.atleast_2d(np.asarray(Kxy, float))
    dx = np.asarray(dx, float).reshape(-1)
    dy = np.asarray(dy, float).reshape(-1)
    return np.maximum(dx[:, None] + dy[None, :] - 2.0 * Kxy, 0.0)


def median_gamma(D2: np.ndarray) -> float:
    """1 / median of the positive squared distances: the usual starting width of a Gaussian."""
    d = np.asarray(D2, float).ravel()
    d = d[d > 1e-12]
    if len(d) == 0:
        return 1.0
    return 1.0 / float(np.median(d))


def gaussian_layer(D2: np.ndarray, gamma: float) -> np.ndarray:
    """exp(-gamma d^2): a kernel on molecules that compares whole distributions, not only means."""
    return np.exp(-float(gamma) * np.asarray(D2, float))


def effective_size(w: Sequence[float]) -> float:
    """(sum w)^2 / sum w^2: how many equally weighted molecules the weighted set is worth."""
    w = np.asarray(w, float).ravel()
    s2 = float(np.sum(w * w))
    return float(np.sum(w)) ** 2 / s2 if s2 > 0 else 0.0


def weighted_mmd2(K_tr: np.ndarray, K_tr_te: np.ndarray, K_te: np.ndarray,
                  w: Optional[Sequence[float]] = None) -> float:
    """Squared distance between the w-weighted mean embedding of the training set and the library's.

    With w = 1 it is the biased MMD^2 of :func:`cmfqsar.mmd.mmd2`.  The weights are rescaled to a
    mean of one, so only their shape matters.
    """
    K_tr = np.asarray(K_tr, float)
    n = K_tr.shape[0]
    w = np.ones(n) if w is None else np.asarray(w, float).ravel()
    if w.sum() <= 0:
        return float("nan")
    a = w / w.sum()
    return float(a @ K_tr @ a - 2.0 * a @ np.asarray(K_tr_te, float).mean(axis=1)
                 + np.asarray(K_te, float).mean())


# --------------------------------------------------------------------------- #
# Kernel mean matching
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KMMResult:
    weights: np.ndarray              # one per training molecule, mean close to one
    mmd2_before: float               # weighted_mmd2 with equal weights
    mmd2_after: float                # and with these weights
    ess: float                       # effective sample size of the weights
    iterations: int
    bound: float                     # the upper bound B on a weight

    def describe(self) -> str:
        n = len(self.weights)
        return (f"kernel mean matching on {n} training molecules: MMD^2 {self.mmd2_before:.4g} -> "
                f"{self.mmd2_after:.4g}, effective size {self.ess:.1f} of {n}, largest weight "
                f"{float(np.max(self.weights)):.2f} (bound {self.bound:g})")


def _project_box_sum(v: np.ndarray, upper: float, lo: float, hi: float) -> np.ndarray:
    """Euclidean projection onto {0 <= b <= upper, lo <= sum b <= hi}.

    The projection is clip(v - tau, 0, upper) for one shift tau, which is zero when the clipped
    vector already has an allowed sum and is otherwise found by bisection, because the sum falls
    monotonically as tau grows.
    """
    b = np.clip(v, 0.0, upper)
    s = float(b.sum())
    if lo <= s <= hi:
        return b
    target = lo if s < lo else hi
    a, c = float(v.min()) - upper, float(v.max())
    for _ in range(100):
        tau = 0.5 * (a + c)
        if float(np.clip(v - tau, 0.0, upper).sum()) > target:
            a = tau
        else:
            c = tau
    return np.clip(v - 0.5 * (a + c), 0.0, upper)


def kmm(K_tr: np.ndarray, K_tr_te: np.ndarray, K_te: Optional[np.ndarray] = None,
        bound: float = 10.0, eps: Optional[float] = None, max_iter: int = 5000,
        tol: float = 1e-9) -> KMMResult:
    """Weights on the training molecules whose weighted mean embedding matches the library's.

    Solves the quadratic programme of Huang et al. (2006) ::

        minimise   1/2 b' K b - kappa' b,    kappa_i = (n / m) sum_j K(x_i, x'_j)
        subject to 0 <= b_i <= bound,  |sum_i b_i - n| <= n eps

    by accelerated projected gradient (FISTA), with the exact projection onto the box and the band
    on the sum.  `K_tr` is the n x n kernel of the training molecules, `K_tr_te` the n x m kernel
    between them and the library.  `eps` defaults to (sqrt(n) - 1) / sqrt(n), the value the paper
    recommends.  `K_te` is needed only for the MMD values that are reported.
    """
    K = np.asarray(K_tr, float)
    Kx = np.asarray(K_tr_te, float)
    n, m = Kx.shape
    if K.shape != (n, n):
        raise ValueError("the training kernel and the cross kernel do not agree in size")
    if eps is None:
        eps = (np.sqrt(n) - 1.0) / np.sqrt(n)
    kappa = (n / m) * Kx.sum(axis=1)
    lo, hi = n * (1.0 - eps), n * (1.0 + eps)
    L = float(np.linalg.eigvalsh(0.5 * (K + K.T))[-1])
    step = 1.0 / max(L, 1e-12)
    b = _project_box_sum(np.ones(n), bound, lo, hi)
    z, t, it = b.copy(), 1.0, 0
    for it in range(1, int(max_iter) + 1):
        b_new = _project_box_sum(z - step * (K @ z - kappa), bound, lo, hi)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = b_new + ((t - 1.0) / t_new) * (b_new - b)
        moved = float(np.max(np.abs(b_new - b)))
        b, t = b_new, t_new
        if moved < tol * max(1.0, float(np.max(b))):
            break
    if K_te is None:
        before = after = float("nan")
    else:
        before = weighted_mmd2(K, Kx, K_te)
        after = weighted_mmd2(K, Kx, K_te, b)
    return KMMResult(weights=b, mmd2_before=before, mmd2_after=after, ess=effective_size(b),
                     iterations=int(it), bound=float(bound))


# --------------------------------------------------------------------------- #
# A density ratio that can be evaluated anywhere: uLSIF
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DensityRatio:
    """w(x) = sum_l theta_l exp(-gamma d^2(x, c_l)), fitted by uLSIF.

    `evaluate` takes the squared field distances from the molecules to the centres, in the column
    order of `centres` (indices into the library the ratio was fitted on).
    """
    theta: np.ndarray
    gamma: float
    lam: float
    centres: np.ndarray
    score: float                     # the cross-validated uLSIF criterion at the chosen setting
    grid: Tuple = field(default=(), repr=False)

    def evaluate(self, D2_to_centres: np.ndarray) -> np.ndarray:
        Phi = gaussian_layer(np.atleast_2d(D2_to_centres), self.gamma)
        return np.maximum(Phi @ self.theta, 0.0)

    def describe(self) -> str:
        return (f"density ratio by uLSIF: {len(self.centres)} centres, gamma {self.gamma:.3g}, "
                f"lambda {self.lam:g}, cross-validated criterion {self.score:+.4f}")


def _ulsif_theta(Phi_tr: np.ndarray, Phi_te: np.ndarray, lam: float) -> np.ndarray:
    H = Phi_tr.T @ Phi_tr / Phi_tr.shape[0]
    h = Phi_te.mean(axis=0)
    theta = np.linalg.solve(H + lam * np.eye(H.shape[0]), h)
    return np.maximum(theta, 0.0)


def _ulsif_criterion(Phi_tr: np.ndarray, Phi_te: np.ndarray, theta: np.ndarray) -> float:
    """1/2 mean_train w^2 - mean_library w: the squared error to the true ratio, up to a constant."""
    return float(0.5 * np.mean((Phi_tr @ theta) ** 2) - np.mean(Phi_te @ theta))


def ulsif(D2_tr_te: np.ndarray, D2_te_te: np.ndarray, centres: int = 100,
          gamma_factors: Sequence[float] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0),
          lams: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0), folds: int = 5,
          seed: int = 0) -> DensityRatio:
    """Fit w(x) = p_library(x) / p_train(x) from squared field distances.

    `D2_tr_te` holds the squared distances from every training molecule to every library molecule,
    `D2_te_te` those among the library molecules.  The centres are at most `centres` library
    molecules chosen at random.  The width gamma (as multiples of the median heuristic) and the
    ridge lambda are chosen by `folds`-fold cross-validation of the uLSIF criterion, with both sets
    split into folds; the criterion is what the fit minimises, so this chooses the setting that
    estimates the ratio best on molecules it did not see.
    """
    D2_tr_te = np.atleast_2d(np.asarray(D2_tr_te, float))
    D2_te_te = np.atleast_2d(np.asarray(D2_te_te, float))
    n, m = D2_tr_te.shape
    if m < 2 or n < 2:
        raise ValueError("a density ratio needs at least two molecules on each side")
    rng = np.random.RandomState(int(seed))
    c = np.sort(rng.permutation(m)[:min(int(centres), m)])
    g0 = median_gamma(D2_tr_te)
    k = max(2, min(int(folds), n, m))
    f_tr = rng.permutation(n) % k
    f_te = rng.permutation(m) % k
    best, grid = None, []
    for gf in gamma_factors:
        gamma = g0 * float(gf)
        Ptr = gaussian_layer(D2_tr_te[:, c], gamma)
        Pte = gaussian_layer(D2_te_te[:, c], gamma)
        for lam in lams:
            score = 0.0
            for f in range(k):
                th = _ulsif_theta(Ptr[f_tr != f], Pte[f_te != f], float(lam))
                score += _ulsif_criterion(Ptr[f_tr == f], Pte[f_te == f], th)
            score /= k
            grid.append((float(gf), float(lam), float(score)))
            if best is None or score < best[0]:
                best = (score, gamma, float(lam))
    score, gamma, lam = best
    theta = _ulsif_theta(gaussian_layer(D2_tr_te[:, c], gamma),
                         gaussian_layer(D2_te_te[:, c], gamma), lam)
    return DensityRatio(theta=theta, gamma=float(gamma), lam=float(lam), centres=c,
                        score=float(score), grid=tuple(grid))


def ulsif_crossfit(D2_tr_te: np.ndarray, D2_te_te: np.ndarray, folds: int = 5, seed: int = 0,
                   centres: int = 100, **kwargs) -> Tuple[DensityRatio, np.ndarray, np.ndarray]:
    """The ratio, and its values at the training and library molecules by cross-fitting.

    A library molecule is one of the centres of the fitted ratio, so its own kernel function adds
    to its weight: evaluated at the molecules it was fitted on, the ratio is too large at the
    library molecules and too small at the training ones.  In a weighted conformal interval that
    bias makes the new molecule look heavier than every calibration molecule and gives infinite
    intervals even when there is no shift at all.  Cross-fitting removes it.  Both sets are split
    into `folds` folds; the weights of fold f come from a ratio fitted, with the width and ridge
    chosen once on all molecules, on the other folds only.

    Returns (ratio fitted on everything, w at each training molecule, w at each library molecule).
    The first is for molecules that took no part in the fit.
    """
    D2_tr_te = np.atleast_2d(np.asarray(D2_tr_te, float))
    D2_te_te = np.atleast_2d(np.asarray(D2_te_te, float))
    ratio = ulsif(D2_tr_te, D2_te_te, centres=centres, seed=seed, **kwargs)
    n, m = D2_tr_te.shape
    k = max(2, min(int(folds), n, m))
    rng = np.random.RandomState(int(seed) + 1)
    f_tr = rng.permutation(n) % k
    f_te = rng.permutation(m) % k
    w_tr = np.zeros(n)
    w_te = np.zeros(m)
    for f in range(k):
        rest = np.flatnonzero(f_te != f)
        c = np.sort(rng.permutation(rest)[:min(int(centres), len(rest))])
        theta = _ulsif_theta(gaussian_layer(D2_tr_te[np.ix_(f_tr != f, c)], ratio.gamma),
                             gaussian_layer(D2_te_te[np.ix_(rest, c)], ratio.gamma), ratio.lam)
        w_tr[f_tr == f] = gaussian_layer(D2_tr_te[np.ix_(f_tr == f, c)], ratio.gamma) @ theta
        w_te[f_te == f] = gaussian_layer(D2_te_te[np.ix_(f_te == f, c)], ratio.gamma) @ theta
    return ratio, np.maximum(w_tr, 0.0), np.maximum(w_te, 0.0)


# --------------------------------------------------------------------------- #
# A density ratio from a classifier: kernel logistic regression
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ClassifierRatio:
    """w(x) = exp(f(x)) n_train / n_library from kernel logistic regression, with cross-fitting.

    `w_train` and `w_library` are the out-of-fold values at the molecules the classifier was fitted
    on; :meth:`evaluate` gives w at new molecules from the fit on all of them.
    """
    w_train: np.ndarray
    w_library: np.ndarray
    gamma: float
    gamma_factor: float
    lam: float
    log_loss: float                  # cross-validated, per molecule, at the chosen setting
    at_edge: bool                    # the chosen setting lies on the edge of the grid
    prior: float                     # n_train / n_library
    coef: np.ndarray = field(repr=False, default=None)
    bias: float = 1.0
    grid: Tuple = field(default=(), repr=False)

    def evaluate(self, D2_to_fitted: np.ndarray) -> np.ndarray:
        """w at new molecules, from their squared field distances to every fitted molecule."""
        G = gaussian_layer(np.atleast_2d(D2_to_fitted), self.gamma) + self.bias
        return np.exp(np.clip(G @ self.coef, -50.0, 50.0)) * self.prior

    def describe(self) -> str:
        edge = " (on the edge of the grid)" if self.at_edge else ""
        return (f"density ratio by kernel logistic regression: gamma {self.gamma_factor:g} x "
                f"median, lambda {self.lam:g}{edge}, cross-validated log-loss "
                f"{self.log_loss:.3f} per molecule (no information: "
                f"{_null_log_loss(self.prior):.3f})")


def _null_log_loss(prior: float) -> float:
    """Log-loss of the constant probability n_library / (n_train + n_library)."""
    q = 1.0 / (1.0 + float(prior))
    return float(-(q * np.log(q) + (1 - q) * np.log(1 - q)))


def classifier_ratio(D2: np.ndarray, is_library: Sequence[bool],
                     gamma_factors: Sequence[float] = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0),
                     lams: Sequence[float] = (1e-3, 1e-2, 1e-1, 1.0, 10.0), folds: int = 5,
                     seed: int = 0, bias: float = 1.0) -> ClassifierRatio:
    """Fit w(x) = p_library(x) / p_train(x) by kernel logistic regression on both sets.

    `D2` holds the squared field distances among all molecules, training and library together, and
    `is_library` marks the library.  The kernel is exp(-gamma d^2) + `bias`; the constant plays the
    part of an intercept, which the Laplace step of :func:`cmfqsar.gpc.klr_fit` does not have.
    The width (as multiples of the median heuristic) and the ridge are chosen by the
    cross-validated log-loss with folds stratified over the two sets.  The out-of-fold latent
    values at the chosen setting are the cross-fitted weights, so no molecule's weight comes from a
    classifier that saw it.
    """
    from .gpc import klr_fit

    D2 = np.asarray(D2, float)
    lib = np.asarray(is_library, bool).ravel()
    N = len(lib)
    if D2.shape != (N, N):
        raise ValueError("D2 must be the square matrix over the training and library molecules")
    n_tr, n_lib = int((~lib).sum()), int(lib.sum())
    if n_tr < 2 or n_lib < 2:
        raise ValueError("a density ratio needs at least two molecules on each side")
    label = np.where(lib, 1.0, -1.0)
    g0 = median_gamma(D2[np.ix_(~lib, lib)])
    k = max(2, min(int(folds), n_tr, n_lib))
    rng = np.random.RandomState(int(seed))
    fold = np.empty(N, int)
    fold[~lib] = rng.permutation(n_tr) % k
    fold[lib] = rng.permutation(n_lib) % k
    best, grid = None, []
    for gf in gamma_factors:
        G = gaussian_layer(D2, g0 * float(gf)) + float(bias)
        for lam in lams:
            f_oof = np.zeros(N)
            for f in range(k):
                fit, out = fold != f, fold == f
                a, _, _ = klr_fit(G[np.ix_(fit, fit)], label[fit], float(lam))
                f_oof[out] = G[np.ix_(out, fit)] @ a
            loss = float(np.mean(np.logaddexp(0.0, -label * f_oof)))
            grid.append((float(gf), float(lam), loss))
            if best is None or loss < best[0]:
                best = (loss, float(gf), float(lam), f_oof)
    loss, gf, lam, f_oof = best
    at_edge = gf in (min(gamma_factors), max(gamma_factors)) or lam in (min(lams), max(lams))
    G = gaussian_layer(D2, g0 * gf) + float(bias)
    coef, _, _ = klr_fit(G, label, lam)
    prior = n_tr / n_lib
    w = np.exp(np.clip(f_oof, -50.0, 50.0)) * prior
    return ClassifierRatio(w_train=w[~lib], w_library=w[lib], gamma=g0 * gf, gamma_factor=gf,
                           lam=lam, log_loss=loss, at_edge=bool(at_edge), prior=prior,
                           coef=coef, bias=float(bias), grid=tuple(grid))


# --------------------------------------------------------------------------- #
# Weighted kernel ridge regression
# --------------------------------------------------------------------------- #

def weighted_krr_fit(K: np.ndarray, y: Sequence[float], w: Sequence[float], lam: float,
                     fit_bias: bool = True) -> Tuple[np.ndarray, float]:
    """Kernel ridge regression minimising sum_i w_i (y_i - f(x_i))^2 + lam |f|^2.

    The solution is the bordered system of :func:`cmfqsar.model.krr_fit` with the ridge of molecule
    i equal to lam / w_i, after the weights are rescaled to a mean of one so that lam keeps its
    meaning.  A molecule with weight zero has an infinite ridge, so its coefficient is zero; it is
    removed from the system rather than given a ridge of 1e300.  With all weights equal this is
    exactly ``krr_fit``.
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float).ravel()
    w = np.asarray(w, float).ravel()
    if len(w) != len(y) or K.shape != (len(y), len(y)):
        raise ValueError("one weight and one activity per molecule of the kernel")
    if np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be non-negative and not all zero")
    w = w / w.mean()
    use = np.flatnonzero(w > 1e-12)
    Ku = K[np.ix_(use, use)]
    nu = len(use)
    A = np.zeros((nu + 1, nu + 1)) if fit_bias else np.zeros((nu, nu))
    A[:nu, :nu] = Ku + np.diag(float(lam) / w[use])
    rhs = y[use]
    if fit_bias:
        A[:nu, nu] = 1.0
        A[nu, :nu] = 1.0
        rhs = np.concatenate([rhs, [0.0]])
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    a = np.zeros(len(y))
    a[use] = sol[:nu]
    return a, (float(sol[nu]) if fit_bias else 0.0)


def weighted_cv_predict(K: np.ndarray, y: Sequence[float], w: Sequence[float], lam: float,
                        folds: int = 10, seed: Optional[int] = 0,
                        fit_bias: bool = True) -> np.ndarray:
    """Out-of-fold predictions of weighted kernel ridge regression, on the folds of ``cv_predict``.

    The importance-weighted cross-validation error of Sugiyama, Krauledat and Mueller (JMLR 8
    (2007) 985) is then sum_i w_i (y_i - p_i)^2 / sum_i w_i over these predictions; it is the
    estimate of the error on the library, not on the training set, and is what a ridge for the
    weighted fit should be chosen by.
    """
    from .model import _cv_splits

    K = np.asarray(K, float)
    y = np.asarray(y, float).ravel()
    w = np.asarray(w, float).ravel()
    p = np.full(len(y), np.nan)
    for te, tr in _cv_splits(y, folds, seed):
        if not np.any(w[tr] > 0):
            continue
        a, b = weighted_krr_fit(K[np.ix_(tr, tr)], y[tr], w[tr], lam, fit_bias)
        p[te] = K[np.ix_(te, tr)] @ a + b
    return p


# --------------------------------------------------------------------------- #
# Weighted conformal intervals
# --------------------------------------------------------------------------- #

def weighted_quantile(scores: Sequence[float], w_cal: Sequence[float],
                      w_new: Sequence[float], level: float) -> np.ndarray:
    """The weighted conformal quantile of Tibshirani et al. (2019), one per new molecule.

    For a new molecule with weight w_new the calibration score i carries the probability
    w_i / (sum_j w_j + w_new) and a point at +infinity carries w_new / (sum_j w_j + w_new).  The
    quantile is the smallest score whose cumulative probability reaches `level`; it is infinite
    when the finite scores together carry less than `level`.  With all weights equal this is the
    ordinary rule ceil((n + 1) level) of :class:`cmfqsar.conformal.Conformal`.
    """
    s = np.asarray(scores, float).ravel()
    wc = np.asarray(w_cal, float).ravel()
    keep = np.isfinite(s) & np.isfinite(wc)
    s, wc = s[keep], np.maximum(wc[keep], 0.0)
    order = np.argsort(s, kind="mergesort")
    s, cum = s[order], np.cumsum(wc[order])
    total = float(cum[-1]) if len(cum) else 0.0
    w_new = np.maximum(np.atleast_1d(np.asarray(w_new, float)).ravel(), 0.0)
    out = np.full(len(w_new), np.inf)
    for t, wn in enumerate(w_new):
        denom = total + float(wn)
        if denom <= 0:
            continue
        # cumulative probability of the k smallest scores reaches the level; a small tolerance
        # keeps the equal-weight case on the same order statistic as the unweighted rule
        k = int(np.searchsorted(cum / denom, float(level) - 1e-12, side="left"))
        if k < len(s):
            out[t] = s[k]
    return out


@dataclass(frozen=True)
class WeightedConformal:
    """A conformal rule whose calibration errors are weighted by the density ratio.

    `weights` are w(x) at the calibration molecules; the weight of each predicted molecule is given
    to :meth:`interval`.  `difficulty` and `beta` give the normalised form, as in
    :class:`cmfqsar.conformal.Conformal`.
    """
    level: float
    residuals: np.ndarray
    weights: np.ndarray
    kind: str = "cv"
    difficulty: Optional[np.ndarray] = None
    beta: float = 1.0

    def __post_init__(self):
        if not 0.0 < float(self.level) < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        if len(np.ravel(self.weights)) != len(np.ravel(self.residuals)):
            raise ValueError("one weight per calibration molecule")

    @property
    def scores(self) -> np.ndarray:
        r = np.abs(np.asarray(self.residuals, float).ravel())
        if self.difficulty is None:
            return r
        return r / (float(self.beta) + np.asarray(self.difficulty, float).ravel())

    @property
    def ess(self) -> float:
        return effective_size(self.weights)

    def half_width(self, w_new: Sequence[float],
                   difficulty: Optional[Sequence[float]] = None) -> np.ndarray:
        q = weighted_quantile(self.scores, self.weights, w_new, self.level)
        if self.difficulty is None or difficulty is None:
            return q
        return q * (float(self.beta) + np.asarray(difficulty, float).ravel())

    def interval(self, y_pred: Sequence[float], w_new: Sequence[float],
                 difficulty: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        yp = np.asarray(y_pred, float).ravel()
        wn = np.asarray(w_new, float).ravel()
        if len(wn) != len(yp):
            raise ValueError("one weight per predicted molecule")
        h = self.half_width(wn, difficulty)
        return yp - h, yp + h

    def describe(self) -> str:
        n = len(np.ravel(self.residuals))
        norm = "" if self.difficulty is None else ", normalised by the distance to the training set"
        return (f"{self.level:.0%} weighted conformal from {n} calibration errors{norm}; "
                f"effective size of the weights {self.ess:.1f}")


def shifted_conformal(conformal, K_train: np.ndarray, K_lib_train: np.ndarray,
                      K_lib: np.ndarray, seed: int = 0):
    """Weight a model's stored calibration by the density ratio of a library to its training set.

    `conformal` is the :class:`cmfqsar.conformal.Conformal` a model carries (out-of-fold errors,
    one per training molecule, in training order).  The three kernel blocks are the model's own
    linear kernel: among the training molecules, from the library to them, and among the library
    molecules.  The ratio is the classifier's, cross-fitted, so neither a training molecule nor a
    library molecule is weighted by a classifier that saw it.

    Returns (the weighted rule, the ratio, the weight of every library molecule, the difficulty of
    every library molecule or None).  The interval of library molecule j is then
    ``rule.interval(y_pred, w_library, difficulty)``.
    """
    from .conformal import difficulty_from_kernel
    from .mmd import assemble

    K_train = np.asarray(K_train, float)
    n = K_train.shape[0]
    res = np.asarray(conformal.residuals, float).ravel()
    if len(res) != n:
        raise ValueError(f"the calibration holds {len(res)} errors and the training set {n} "
                         "molecules; a weight cannot be matched to its error")
    K, mask = assemble(K_train, K_lib_train, K_lib)
    d = np.diag(K)
    ratio = classifier_ratio(distance2(K, d, d), ~mask, seed=seed)
    difficulty = None
    if conformal.difficulty is not None:
        difficulty = difficulty_from_kernel(np.asarray(K_lib_train, float),
                                            np.diag(np.asarray(K_lib, float)), K_train)
    rule = WeightedConformal(level=float(conformal.level), residuals=res, weights=ratio.w_train,
                             kind=str(conformal.kind), difficulty=conformal.difficulty,
                             beta=float(conformal.beta))
    return rule, ratio, ratio.w_library, difficulty


def weighted_calibrate(y_true: Sequence[float], y_pred: Sequence[float], weights: Sequence[float],
                       level: float = 0.9, difficulty: Optional[Sequence[float]] = None,
                       beta: float = 1.0, kind: str = "cv") -> WeightedConformal:
    """Calibrate on out-of-sample predictions, each error weighted by w at its molecule."""
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    w = np.asarray(weights, float).ravel()
    if not (len(y_true) == len(y_pred) == len(w)):
        raise ValueError("one prediction and one weight per molecule")
    keep = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(w)
    d = None if difficulty is None else np.asarray(difficulty, float).ravel()[keep]
    return WeightedConformal(level=float(level), residuals=np.abs(y_true[keep] - y_pred[keep]),
                             weights=w[keep], kind=str(kind), difficulty=d, beta=float(beta))
