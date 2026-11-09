"""The model as the fast level of a hierarchy: a standard deviation in the units of the activity,
its calibration on the folds, and the rule that sends a structure back to the fine level (plan
item MS1).

A multiscale scheme uses a fitted CMF model between an electronic-structure method and a coarser
model: the fine level labels a few hundred structures, the model answers for the thousands the
coarse model needs.  For that the model has to say, per structure, whether its answer can be
used.  The posterior variance of :meth:`CMFModel.predict_with_variance` is the right quantity,
but it is not usable as it stands, for three reasons this module removes.

* **It is in the units of the kernel.**  The Gaussian process that matches the ridge fit has the
  prior covariance ``s K`` and the noise ``s lambda``, where ``s`` is the one amplitude the
  evidence profiles out in closed form (``quad / dof``, the mean squared whitened residual of the
  training set).  The predictive variance of a new *measurement* is therefore ``s (v + lambda)``,
  with ``v`` the posterior variance in kernel units, and its square root is in the units of
  ``y``.  The constant mean that the ridge fits as its bias is marginalised under a flat prior,
  which adds ``(1 - k^T A^{-1} 1)^2 / (1^T A^{-1} 1)`` to ``v`` (Rasmussen and Williams 2006,
  eq. 2.42 with one basis function); that term is what makes a structure far from every training
  one uncertain about the *level* of the activity and not only about its shape.

* **Nobody has checked that it means what it says.**  On the folds, the standardised residual
  ``z = (y - mean) / sd`` should have a root mean square of one, and the interval of two standard
  deviations should hold about 95 % of the activities.  :class:`Calibration` reports both and the
  factor that would make the root mean square one.  A factor well above one says the model is
  surer than it should be, which is the usual finding when the kernel misses a part of the
  chemistry; the factor is then the honest correction, and it is reported, not applied.

* **A number is not a decision.**  A tolerance in the units of the activity turns it into one:
  :func:`defer` marks every structure whose standard deviation exceeds the tolerance, and
  :class:`Handoff` reports what the rule did, with the error on each side when the activities
  are known.  :func:`tolerance_for_budget` goes the other way, from the share of structures the
  fine level can afford to the tolerance that sends it that many.

The arithmetic is one Cholesky factorisation of the training kernel per fit and one triangular
solve per structure, the same cost as the variance already paid.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import solve_triangular


def _cholesky(A: np.ndarray) -> np.ndarray:
    """A lower Cholesky factor, with a little jitter added if the matrix is not quite definite."""
    n = A.shape[0]
    for jitter in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        try:
            return np.linalg.cholesky(A + jitter * np.eye(n) if jitter else A)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("the kernel is not positive definite even with jitter")


@dataclass
class Surrogate:
    """The Gaussian-process reading of a ridge fit, with its scale in the units of the activity.

    ``scale`` is ``s``: the prior covariance is ``s K`` and the noise is ``s lambda``, so a
    standard deviation this object returns is in the units of ``y``.  The factor of the training
    kernel is kept, so that every later block of structures costs one triangular solve.
    """

    lam: float
    fit_bias: bool
    scale: float
    dof: int
    n_train: int
    L: np.ndarray
    alpha: np.ndarray
    bias: float
    u: Optional[np.ndarray]
    denom: float

    @classmethod
    def fit(cls, K: np.ndarray, y: Sequence[float], lam: float, fit_bias: bool = True) -> "Surrogate":
        """Fit on the training kernel ``K`` and the activities ``y`` with the ridge ``lam``.

        The dual coefficients and the bias are those of :func:`cmfqsar.model.krr_fit`: the
        bordered system it solves is the Gaussian process with the constant mean marginalised,
        so the predictive mean here is the model's own prediction.
        """
        K = np.asarray(K, float)
        y = np.asarray(y, float).ravel()
        n = len(y)
        if K.shape != (n, n):
            raise ValueError(f"the kernel is {K.shape} but there are {n} activities")
        if not np.all(np.isfinite(y)):
            raise ValueError("every training activity must be measured; a nan has no residual")
        lam = float(lam)
        L = _cholesky(K + lam * np.eye(n))

        def solve(b):
            return solve_triangular(L.T, solve_triangular(L, b, lower=True), lower=False)

        w = solve(y)
        if fit_bias:
            one = np.ones(n)
            u = solve(one)
            denom = float(one @ u)
            if denom <= 0:
                raise np.linalg.LinAlgError("the kernel leaves no room for a constant mean")
            bias = float(one @ w) / denom
            alpha = w - bias * u
            quad = float((y - bias) @ alpha)
            dof = n - 1
        else:
            u, denom, bias, alpha = None, float("inf"), 0.0, w
            quad = float(y @ alpha)
            dof = n
        if dof < 1:
            raise ValueError("a scale needs at least two activities")
        scale = max(quad, 1e-300) / dof
        return cls(lam=lam, fit_bias=bool(fit_bias), scale=float(scale), dof=int(dof), n_train=n,
                   L=L, alpha=alpha, bias=bias, u=u, denom=denom)

    def predict(self, K_test_train: np.ndarray, k_test_self: Sequence[float],
                with_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """(prediction, standard deviation) of new structures, both in the units of the activity.

        ``K_test_train`` holds the kernels of the new structures against the training ones and
        ``k_test_self`` their self-kernels.  With ``with_noise`` the deviation is that of a new
        *measurement*, which includes the noise ``s lambda``; without it, that of the underlying
        value, which is what two predictions of the same structure share.
        """
        k = np.atleast_2d(np.asarray(K_test_train, float))
        k_self = np.asarray(k_test_self, float).ravel()
        if k.shape[1] != self.n_train or len(k_self) != k.shape[0]:
            raise ValueError(f"expected {k.shape[0]} rows of {self.n_train} kernels and as many "
                             f"self-kernels, got {k.shape} and {len(k_self)}")
        mean = k @ self.alpha + self.bias
        V = solve_triangular(self.L, k.T, lower=True)
        v = k_self - np.sum(V * V, axis=0)
        if self.fit_bias:
            t = 1.0 - k @ self.u
            v = v + t * t / self.denom
        v = np.maximum(v, 0.0)
        var = self.scale * (v + (self.lam if with_noise else 0.0))
        return mean, np.sqrt(var)

    @property
    def noise_sd(self) -> float:
        """The standard deviation of the noise the fit attributes to one measurement."""
        return float(np.sqrt(self.scale * self.lam))


def cross_validate(K: np.ndarray, y: Sequence[float], lam: float, folds: int = 10,
                   seed: Optional[int] = 0, fit_bias: bool = True,
                   parts: Optional[Sequence[Sequence[int]]] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Out-of-fold (prediction, standard deviation) on a training kernel, in the units of ``y``.

    The folds are the ones the model's own cross-validation uses, so the predictions agree with
    the model's ``y_cv`` and the deviations are the ones a structure would have been given had
    it been new.  The scale is fitted on the training part of each fold.  ``parts`` replaces the
    random folds by given held-out index sets, for folds that hold out whole structures.
    """
    from .model import _cv_splits

    y = np.asarray(y, float).ravel()
    K = np.asarray(K, float)
    diag = np.diag(K)
    mean = np.full(len(y), np.nan)
    sd = np.full(len(y), np.nan)
    if parts is not None:
        splits = [(np.asarray(te, int), np.setdiff1d(np.arange(len(y)), np.asarray(te, int)))
                  for te in parts]
    else:
        splits = _cv_splits(y, folds, seed)
    for te, tr in splits:
        s = Surrogate.fit(K[np.ix_(tr, tr)], y[tr], lam, fit_bias)
        mean[te], sd[te] = s.predict(K[np.ix_(te, tr)], diag[te])
    return mean, sd


@dataclass
class Calibration:
    """Does the standard deviation mean what it says?  Measured on activities the fit did not see.

    ``z_rms`` is the root mean square of ``(y - mean) / sd``: one for a calibrated deviation,
    above one when the model is surer than it should be.  ``coverage_1sd`` and ``coverage_2sd``
    are the shares of the activities inside one and two standard deviations of the prediction,
    68 % and 95 % for a calibrated Gaussian.  ``spearman`` is the rank correlation between the
    deviation and the absolute error, which says whether the deviation at least *orders* the
    structures by their error when its size is off.
    """

    n: int
    rmse: float
    mean_sd: float
    z_rms: float
    coverage_1sd: float
    coverage_2sd: float
    spearman: float

    @classmethod
    def of(cls, y: Sequence[float], mean: Sequence[float], sd: Sequence[float]) -> "Calibration":
        y = np.asarray(y, float).ravel()
        mean = np.asarray(mean, float).ravel()
        sd = np.asarray(sd, float).ravel()
        ok = np.isfinite(y) & np.isfinite(mean) & np.isfinite(sd) & (sd > 0)
        y, mean, sd = y[ok], mean[ok], sd[ok]
        n = len(y)
        if n < 3:
            return cls(n=n, rmse=float("nan"), mean_sd=float("nan"), z_rms=float("nan"),
                       coverage_1sd=float("nan"), coverage_2sd=float("nan"), spearman=float("nan"))
        err = y - mean
        z = err / sd
        from scipy.stats import spearmanr
        rho = float(spearmanr(sd, np.abs(err)).statistic) if n >= 5 and np.ptp(sd) > 0 else float("nan")
        return cls(n=n, rmse=float(np.sqrt(np.mean(err ** 2))), mean_sd=float(np.mean(sd)),
                   z_rms=float(np.sqrt(np.mean(z ** 2))),
                   coverage_1sd=float(np.mean(np.abs(z) <= 1.0)),
                   coverage_2sd=float(np.mean(np.abs(z) <= 2.0)), spearman=rho)

    @property
    def factor(self) -> float:
        """What to multiply the standard deviation by to make the root mean square of z one."""
        return self.z_rms

    def describe(self) -> str:
        if self.n < 3 or not np.isfinite(self.z_rms):
            return f"calibration: too few activities ({self.n}) to measure"
        verdict = ("calibrated" if 0.8 <= self.z_rms <= 1.25 else
                   "surer than it should be" if self.z_rms > 1.25 else "less sure than it could be")
        return (f"calibration on {self.n} activities: RMSE {self.rmse:.3f}, mean sd "
                f"{self.mean_sd:.3f}, z rms {self.z_rms:.2f} ({verdict}; multiply the sd by "
                f"{self.factor:.2f} to correct it), inside 1 sd {self.coverage_1sd:.0%} "
                f"(68 % expected), inside 2 sd {self.coverage_2sd:.0%} (95 % expected), "
                f"sd against |error| Spearman {self.spearman:+.2f}")


def defer(sd: Sequence[float], tolerance: float) -> np.ndarray:
    """True for every structure the fast level must pass to the fine one: ``sd > tolerance``."""
    return np.asarray(sd, float) > float(tolerance)


def tolerance_for_budget(sd: Sequence[float], fraction: float) -> float:
    """The tolerance that sends at most this share of the structures to the fine level.

    ``fraction`` is the share the fine level can afford, between 0 and 1.  The answer is the
    quantile of the deviations at ``1 - fraction``; :func:`defer` at that tolerance marks about
    ``fraction`` of the structures.
    """
    f = float(fraction)
    if not 0.0 <= f <= 1.0:
        raise ValueError("the share of structures must be between 0 and 1")
    sd = np.asarray(sd, float)
    if f >= 1.0:
        return float(np.min(sd)) - 1e-12 if sd.size else 0.0
    return float(np.quantile(sd, 1.0 - f))


@dataclass
class Handoff:
    """What the tolerance rule did to one set of structures.

    ``n_deferred`` structures had a standard deviation above the tolerance and go to the fine
    level; the rest are answered by the model.  When the activities are known the root mean
    square error of each side is measured, so the rule can be judged: the answered side should
    be at or under the tolerance, and the deferred side worse.
    """

    tolerance: float
    n: int
    n_deferred: int
    mean_sd_answered: float
    mean_sd_deferred: float
    rmse_answered: float
    rmse_deferred: float

    @classmethod
    def of(cls, sd: Sequence[float], tolerance: float, y: Optional[Sequence[float]] = None,
           mean: Optional[Sequence[float]] = None) -> "Handoff":
        sd = np.asarray(sd, float).ravel()
        out = defer(sd, tolerance)
        kept = ~out

        def _mean_sd(mask):
            return float(np.mean(sd[mask])) if mask.any() else float("nan")

        def _rmse(mask):
            if y is None or mean is None or not mask.any():
                return float("nan")
            yy = np.asarray(y, float).ravel()[mask]
            mm = np.asarray(mean, float).ravel()[mask]
            ok = np.isfinite(yy) & np.isfinite(mm)
            return float(np.sqrt(np.mean((yy[ok] - mm[ok]) ** 2))) if ok.any() else float("nan")

        return cls(tolerance=float(tolerance), n=int(sd.size), n_deferred=int(out.sum()),
                   mean_sd_answered=_mean_sd(kept), mean_sd_deferred=_mean_sd(out),
                   rmse_answered=_rmse(kept), rmse_deferred=_rmse(out))

    @property
    def n_answered(self) -> int:
        return self.n - self.n_deferred

    @property
    def fraction_answered(self) -> float:
        return self.n_answered / self.n if self.n else float("nan")

    def describe(self) -> str:
        s = (f"tolerance {self.tolerance:.3f}: the model answers {self.n_answered} of {self.n} "
             f"({self.fraction_answered:.0%}), {self.n_deferred} go to the fine level")
        if np.isfinite(self.rmse_answered) or np.isfinite(self.rmse_deferred):
            s += (f"; RMSE of the answered {self.rmse_answered:.3f} (mean sd "
                  f"{self.mean_sd_answered:.3f}), of the deferred {self.rmse_deferred:.3f} "
                  f"(mean sd {self.mean_sd_deferred:.3f})")
        return s
