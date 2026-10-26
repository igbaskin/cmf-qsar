"""Choosing alpha and lambda by the evidence instead of by cross-validation.

Every hyper-parameter in this program is chosen by cross-validation: for each candidate, ten
fits and ten predictions, and the best cross-validated score wins.  That is reliable and slow, and
it spends ten factorisations to learn what one can tell.

A kernel ridge model is the posterior mean of a Gaussian process, and a Gaussian process scores
its own hyper-parameters without holding anything out.  The marginal likelihood -- the
probability of the training activities under the model, with the weights integrated away -- is

    log p(y) = -1/2 (y - 1m)' A^-1 (y - 1m) - 1/2 log|A| - 1/2 log(1' A^-1 1) - (n-1)/2 log 2pi

with ``A = K + lambda I``.  Every term is read off the Cholesky factorisation that the fit needs
anyway, so a candidate costs one factorisation rather than ten, and nothing is held out.

Two details make it usable here rather than merely correct.

**The constant is marginalised, not fitted.**  The program's KRR carries a bias, solved in a
bordered system, and the matching Gaussian process has an unknown constant mean.  Fitting that
mean and then scoring would count it twice; integrating it out under a flat prior gives the
third term above and the `m` that appears in the first, which is the generalised-least-squares
mean rather than the arithmetic one.

**The amplitude is profiled out.**  The kernels here are cosine-normalised and the activities
arrive in whatever units the assay used, so the scale of the prior matters and nobody knows it.
Scaling the kernel and the noise together by `s` leaves the ridge -- and therefore the fitted
model -- unchanged while moving the evidence, and the maximising `s` has a closed form, so it is
substituted rather than searched.  What is left is a criterion that depends on alpha and lambda
alone.  It is a density, so changing the units of the activity still shifts every value by the
same constant -- the ranking is untouched, and it is the ranking that selects.

What the evidence is not is a cross-validated score.  It answers "how probable are these
activities under this model", not "how well would this model predict a molecule it has not
seen", and the two can disagree when the kernel is misspecified -- which for molecular fields is
the normal case.  `benchmarks/run_evidence.py` measures how far apart they are in practice.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["evidence_grid", "log_evidence", "profile_amplitude"]


def _cholesky(A: np.ndarray) -> np.ndarray:
    """A Cholesky factor, with increasing jitter for a kernel that is barely positive."""
    A = 0.5 * (A + A.T)
    scale = float(np.mean(np.diag(A))) or 1.0
    eye = np.eye(A.shape[0])
    for jitter in [0.0] + [10.0 ** p for p in range(-12, -2)]:
        try:
            return np.linalg.cholesky(A + jitter * scale * eye)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("the kernel is not positive definite even with jitter")


def profile_amplitude(quadratic: float, n: int) -> float:
    """The kernel amplitude that maximises the evidence, in closed form.

    Scaling the kernel and the noise together by `s` leaves the predictive mean exactly where it
    was, so this is not a parameter of the model but of the prior's scale; it has an optimum and
    no search is needed for it.
    """
    return float(max(quadratic, 1e-300) / max(n, 1))


def log_evidence(K: np.ndarray, y: Sequence[float], lam: float, fit_bias: bool = True,
                 profile: bool = True) -> float:
    """The log marginal likelihood of `y` under the kernel `K` with ridge `lam`.

    With `fit_bias` the constant mean is marginalised under a flat prior, which is the Gaussian
    process matching the bordered system the program's KRR solves.  With `profile` the kernel
    amplitude is substituted at its maximum, so the value no longer depends on the scale of the
    prior -- which nobody knows -- and the ranking over alpha and lambda no longer depends on the
    units of the activity.  The value itself is a density and still shifts with those units, by
    the same constant for every candidate.
    """
    y = np.asarray(y, float).ravel()
    n = len(y)
    if n < 3:
        return float("nan")
    A = np.asarray(K, float) + float(lam) * np.eye(n)
    try:
        L = _cholesky(A)
    except np.linalg.LinAlgError:
        return float("-inf")
    log_det = 2.0 * float(np.sum(np.log(np.diag(L))))

    def solve(b):
        return np.linalg.solve(L.T, np.linalg.solve(L, b))

    Ay = solve(y)
    if fit_bias:
        one = np.ones(n)
        A1 = solve(one)
        denom = float(one @ A1)
        if denom <= 0:
            return float("-inf")
        mean = float(one @ Ay) / denom
        resid = y - mean
        quad = float(resid @ solve(resid))
        dof = n - 1
        extra = -0.5 * math.log(denom)
    else:
        quad = float(y @ Ay)
        dof = n
        extra = 0.0
    if not np.isfinite(quad) or quad <= 0:
        return float("-inf")
    if profile:
        # s = quad / dof substituted; the terms in s collect into -dof/2 (log(quad/dof) + 1)
        return float(-0.5 * dof * (math.log(quad / dof) + 1.0)
                     - 0.5 * log_det + extra - 0.5 * dof * math.log(2.0 * math.pi))
    return float(-0.5 * quad - 0.5 * log_det + extra - 0.5 * dof * math.log(2.0 * math.pi))


def evidence_grid(kernels: Sequence[Tuple[float, np.ndarray]], y: Sequence[float],
                  lams: Sequence[float], fit_bias: bool = True
                  ) -> Tuple[float, float, Dict[Tuple[float, float], float]]:
    """(best alpha, best lambda, the whole surface) by evidence over a grid of kernels.

    `kernels` is [(alpha, K)]; the evidence is comparable across both axes because the amplitude
    is profiled out of each.
    """
    surface: Dict[Tuple[float, float], float] = {}
    best: Optional[Tuple[float, float, float]] = None
    for alpha, K in kernels:
        for lam in lams:
            e = log_evidence(K, y, float(lam), fit_bias=fit_bias)
            surface[(float(alpha), float(lam))] = e
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, float(alpha), float(lam))
    if best is None:
        return float("nan"), float("nan"), surface
    return best[1], best[2], surface
