"""Does the activity depend on the fields, and does it depend on them beyond a reference?

The question comes before any model. A field model that beats the mean can still be measuring
nothing but size, burial or a count of oxygens, and the usual way to find out is to fit a model on
that number and compare cross-validated errors. This module asks the question directly, with a
test whose null distribution is exact.

**HSIC** (the Hilbert-Schmidt independence criterion; Gretton, Bousquet, Smola and Schoelkopf,
ALT 2005) measures the dependence between two variables from two kernel matrices. With K the field
kernel of n molecules, L a kernel on their activities and H = I - 11'/n the centring matrix ::

    HSIC = tr(H K H H L H) / n^2.

It is zero when the two are independent and positive otherwise, for a characteristic kernel. With
the linear kernel on the activity, L = y y', it is ||sum_i (y_i - mean y) phi(x_i)||^2 / n^2: the
squared norm of the field weighted by the centred activity, which is the covariance of the
activity with the field. That is the default, because it asks whether the *mean* activity
depends on the field, which is what a regression model can use.

**The test.** Under independence, permuting the activities over the molecules leaves the joint
distribution unchanged, so the p-value of the observed HSIC against its permutations is exact for
any n: (1 + #{permuted >= observed}) / (1 + permutations).

**The conditional test: beyond a reference.** The question "do the fields carry information about
the activity that the reference z does not" is the conditional independence y _||_ x | z. When z
takes few values, as a count does, the molecules with equal z form strata. Permuting the
activities *within* every stratum leaves the joint distribution unchanged under the null, so the
p-value is again exact (the stratified permutation test). The statistic is HSIC with both
kernels centred within the strata instead of over the whole set, so that the part of the field and
of the activity that the stratum already explains is removed and only the dependence inside the
strata is measured. A molecule alone in its stratum carries no information and is reported.

A continuous reference (logP, a Hammett constant) has no strata. It can be cut into quantile bins,
and the test is then exact for the binned reference but not for the reference itself: dependence
of the activity on z inside a bin leaks into the statistic. ``benchmarks/run_hsic.py`` measured
the leak at 19 to 22 % false positives at the 5 % level on the Sutherland sets. Passing the
reference itself as `covariate` repairs it: after every permutation the activity is reduced to its
residual from a straight line in z fitted within each bin, so the trend of the activity with z
inside a bin no longer reaches the statistic. That brought the false-positive rate to 5 % or
below. The test is then approximate, and the result says so.

**The alignment** reported beside the p-value is HSIC divided by sqrt(HSIC(K, K) HSIC(L, L)), the
centred kernel alignment (Cortes, Mohri and Rostamizadeh, JMLR 2012), in [0, 1]. It is a size of
effect that does not grow with n, which the p-value does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["HSICResult", "centre_within", "hsic", "hsic_test", "strata_of"]


def strata_of(z: Sequence, bins: Optional[int] = None, max_levels: Optional[int] = None
              ) -> Tuple[np.ndarray, bool]:
    """(stratum of every molecule, whether the strata are the exact values of z).

    `z` is one value per molecule, or a row of values (a count vector), in which case molecules
    with identical rows share a stratum. Without `bins` the distinct values are the strata, which
    is exact. With `bins`, a one-dimensional z is cut at its quantiles into that many strata,
    which is not exact for z itself. `max_levels` refuses exact strata when z has more distinct
    values than that, because a stratum of one molecule carries no information.
    """
    z = np.asarray(z)
    if z.ndim == 1 and bins:
        x = z.astype(float)
        if not np.all(np.isfinite(x)):
            raise ValueError("the reference has missing values")
        edges = np.unique(np.quantile(x, np.linspace(0, 1, int(bins) + 1)[1:-1]))
        return np.searchsorted(edges, x, side="right").astype(int), False
    rows = z.reshape(len(z), -1)
    _, labels = np.unique(rows, axis=0, return_inverse=True)
    labels = np.asarray(labels).ravel().astype(int)
    if max_levels is not None and labels.max() + 1 > max_levels:
        raise ValueError(f"the reference takes {labels.max() + 1} distinct values; cut it into "
                         f"bins, or the strata hold one molecule each and carry no information")
    return labels, True


def centre_within(M: np.ndarray, strata: Optional[Sequence[int]] = None) -> np.ndarray:
    """H_S M H_S, with H_S the centring within every stratum (over the whole set without strata)."""
    M = np.asarray(M, float)
    n = M.shape[0]
    if strata is None:
        r = M.mean(axis=0)
        c = M.mean(axis=1)
        return M - r[None, :] - c[:, None] + M.mean()
    s = np.asarray(strata, int)
    P = np.zeros((n, n))
    for g in np.unique(s):
        idx = np.flatnonzero(s == g)
        P[np.ix_(idx, idx)] = 1.0 / len(idx)
    H = np.eye(n) - P
    return H @ M @ H


def hsic(K: np.ndarray, L: np.ndarray, strata: Optional[Sequence[int]] = None) -> float:
    """HSIC of two kernel matrices, centred over the set or within the strata, divided by n^2."""
    Kc = centre_within(K, strata)
    Lc = centre_within(L, strata)
    n = Kc.shape[0]
    return float(np.sum(Kc * Lc) / (n * n))


@dataclass(frozen=True)
class HSICResult:
    statistic: float                 # HSIC, centred within the strata when there are strata
    alignment: float                 # statistic / sqrt(HSIC(K, K) HSIC(L, L)), in [0, 1]
    p_value: float
    permutations: int
    n: int
    informative: int                 # molecules in strata of two or more
    strata: int                      # 1 for the unconditional test
    exact: bool                      # False when the strata are bins of a continuous reference
    null_mean: float
    null_sd: float
    kernel_y: str
    partial: bool = False            # a line in the covariate was removed within every stratum

    @property
    def z(self) -> float:
        return (self.statistic - self.null_mean) / self.null_sd if self.null_sd > 0 else float("nan")

    def describe(self) -> str:
        if self.strata == 1:
            what = "dependence of the activity on the fields"
        else:
            what = (f"dependence within {self.strata} strata of the reference "
                    f"({self.informative} of {self.n} molecules share a stratum)")
        if self.exact:
            exact = "exact"
        elif self.partial:
            exact = ("approximate: the strata are bins of a continuous value, with a straight "
                     "line in it removed within each bin")
        else:
            exact = "approximate: the strata are bins of a continuous value"
        return (f"{what}: alignment {self.alignment:.3f}, p = {self.p_value:.4g} from "
                f"{self.permutations} permutations ({exact}), z = {self.z:+.1f}")


def _permuted_order(strata: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    order = np.arange(len(strata))
    for g in np.unique(strata):
        idx = np.flatnonzero(strata == g)
        if len(idx) > 1:
            order[idx] = idx[rng.permutation(len(idx))]
    return order


def _within_projection(strata: np.ndarray, covariate: np.ndarray, degree: int) -> np.ndarray:
    """I - P, with P the projection onto a polynomial of `degree` in the covariate within every
    stratum (its mean, slope, ...). Columns that vanish, as the slope does in a stratum where the
    covariate is constant, are dropped by the rank cut of the singular values."""
    n = len(strata)
    cols = []
    for g in np.unique(strata):
        m = strata == g
        zz = np.where(m, covariate - covariate[m].mean(), 0.0)
        for d in range(int(degree) + 1):
            cols.append(np.where(m, zz ** d, 0.0))
    A = np.column_stack(cols)
    U, sv, _ = np.linalg.svd(A, full_matrices=False)
    keep = sv > 1e-10 * max(float(sv.max()), 1e-300)
    U = U[:, keep]
    return np.eye(n) - U @ U.T


def hsic_test(K: np.ndarray, y: Sequence[float], strata: Optional[Sequence[int]] = None,
              exact: bool = True, kernel_y: str = "linear", permutations: int = 999,
              seed: int = 0, block: int = 256, covariate: Optional[Sequence[float]] = None,
              degree: int = 1) -> HSICResult:
    """Permutation test of HSIC between the field kernel `K` and the activities `y`.

    With `strata` the activities are permuted within the strata and both kernels are centred
    within them: the test of dependence beyond the reference the strata come from. `exact`
    records whether the strata are the reference's own values (see :func:`strata_of`).
    `kernel_y` is ``"linear"`` (y y', the covariance of the activity with the field; the default)
    or ``"gauss"`` (a Gaussian of the activity difference, median width, which also sees a
    dependence of the spread). With the linear kernel all permutations are evaluated as one matrix
    product, in blocks of `block`.

    `covariate` (with `strata` and the linear kernel) is the continuous reference the strata are
    bins of. After every permutation the activity is replaced by its residual from a polynomial
    of `degree` in the covariate fitted within each stratum, which removes the trend of the
    activity with the reference inside a bin. The test is then approximate.
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float).ravel()
    n = len(y)
    if K.shape != (n, n):
        raise ValueError("one activity per molecule of the kernel")
    if not np.all(np.isfinite(y)):
        raise ValueError("the activities have missing values; leave those molecules out")
    s = np.zeros(n, int) if strata is None else np.asarray(strata, int).ravel()
    if len(s) != n:
        raise ValueError("one stratum per molecule")
    sizes = np.bincount(s)
    informative = int(np.sum(sizes[s] > 1))
    n_strata = int(len(np.unique(s)))
    Kc = centre_within(K, None if strata is None else s)
    rng = np.random.RandomState(int(seed))
    partial = covariate is not None
    if partial:
        if strata is None or kernel_y != "linear":
            raise ValueError("a covariate needs strata and the linear kernel on the activity")
        c = np.asarray(covariate, float).ravel()
        if len(c) != n or not np.all(np.isfinite(c)):
            raise ValueError("one finite covariate value per molecule")
        R = _within_projection(s, c, degree)
        KR = R @ Kc @ R
        observed = float(y @ KR @ y) / n ** 2
        kk = float(np.sum(KR * KR)) / n ** 2
        ry = R @ y
        ll = float(ry @ ry) ** 2 / n ** 2
        null = np.empty(int(permutations))
        done = 0
        while done < permutations:
            m = min(int(block), int(permutations) - done)
            Y = np.column_stack([y[_permuted_order(s, rng)] for _ in range(m)])
            null[done:done + m] = np.sum(Y * (KR @ Y), axis=0) / n ** 2
            done += m
    elif kernel_y == "linear":
        # y centred within the strata; HSIC = yc' Kc yc / n^2, and permuting y within the strata
        # permutes yc the same way, because the centring is the stratum mean
        means = np.bincount(s, weights=y) / np.maximum(sizes, 1)
        yc = y - means[s]
        observed = float(yc @ Kc @ yc) / n ** 2
        kk = float(np.sum(Kc * Kc)) / n ** 2
        ll = float(yc @ yc) ** 2 / n ** 2
        null = np.empty(int(permutations))
        done = 0
        while done < permutations:
            m = min(int(block), int(permutations) - done)
            Y = np.empty((n, m))
            for j in range(m):
                Y[:, j] = yc[_permuted_order(s, rng)]
            null[done:done + m] = np.sum(Y * (Kc @ Y), axis=0) / n ** 2
            done += m
    elif kernel_y == "gauss":
        d2 = (y[:, None] - y[None, :]) ** 2
        off = d2[np.triu_indices(n, 1)]
        width = float(np.median(off[off > 0])) if np.any(off > 0) else 1.0
        L = np.exp(-d2 / width)
        Lc = centre_within(L, None if strata is None else s)
        observed = float(np.sum(Kc * Lc)) / n ** 2
        kk = float(np.sum(Kc * Kc)) / n ** 2
        ll = float(np.sum(Lc * Lc)) / n ** 2
        null = np.empty(int(permutations))
        for t in range(int(permutations)):
            o = _permuted_order(s, rng)
            null[t] = float(np.sum(Kc * Lc[np.ix_(o, o)])) / n ** 2
    else:
        raise ValueError("kernel_y is 'linear' or 'gauss'")
    p = (1.0 + float(np.sum(null >= observed - 1e-12 * abs(observed)))) / (1.0 + len(null))
    denom = np.sqrt(max(kk, 0.0) * max(ll, 0.0))
    return HSICResult(statistic=observed, alignment=float(observed / denom) if denom > 0 else 0.0,
                      p_value=float(p), permutations=int(permutations), n=n,
                      informative=informative, strata=n_strata,
                      exact=(bool(exact) and not partial) if strata is not None else True,
                      null_mean=float(null.mean()), null_sd=float(null.std()),
                      kernel_y=kernel_y, partial=partial)
