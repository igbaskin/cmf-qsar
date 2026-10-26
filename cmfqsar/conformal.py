"""Prediction intervals with a coverage guarantee, for any learner and any kernel source.

Everything else the program says about an individual prediction is an estimate without
a guarantee.  The Gaussian process
returns a posterior standard deviation, but it is the standard deviation of *that model's own
assumptions*, and when the kernel is misspecified -- which is the normal case for molecular fields
-- it is not a statement about how often the true value falls inside.  The bootstrap intervals of
:mod:`cmfqsar.validation` are intervals on a *statistic*, on q² or on an R²pred, and say nothing
about one molecule.

Conformal prediction (Vovk, Gammerman & Shafer 2005; Lei et al., JASA 113 (2018) 1094) supplies
the missing guarantee, and requires almost nothing in return.  Take the absolute errors the model
makes on molecules it did not fit, let *q* be their (1-alpha) quantile with the finite-sample
correction below, and report y_hat +/- q.  Then, provided the calibration molecules and the new
molecule are exchangeable, the truth lies in that interval at least (1-alpha) of the time --
whatever the learner, whatever the kernel, however wrong the model is.  A badly specified model
does not lose its coverage; its cost is a wider interval, which is a fair cost.

Two ways to get the errors, and they are not equally reliable:

``split``
    Hold out a calibration set, fit on the rest, take the errors there.  The guarantee is exact
    and distribution-free, and the cost is that neither the fit nor the calibration sees all the
    data -- which on a set of thirty molecules is a real cost.

``cv``
    Use the out-of-fold errors of the cross-validation the program runs anyway, then predict with
    the model refitted on everything.  Nothing is held out and nothing extra is computed, but the
    calibration errors come from models fitted on n(1-1/k) molecules while the predictions come
    from a model fitted on n, so exchangeability is only approximate and the guarantee with it.
    It is the variant used in practice, and `benchmarks/run_conformal.py` measures what its
    coverage actually is on real sets rather than asserting that it is fine.

Normalised conformal is the third option, and the one this program is best able to offer: scale
each error by a measure of how hard its molecule is, and the interval widens where the model is
extrapolating instead of staying the same width everywhere.  The measure used here is the
applicability domain the program already computes -- one minus the mean kernel cosine to the
nearest training molecules -- so a molecule outside the domain is not merely flagged, it is given
a wider interval, which is what a user wanted the flag to mean.

Regression only.  Conformal classification produces prediction *sets* rather than intervals and
needs its own vocabulary; it is not implemented here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["Conformal", "calibrate", "calibrate_cv", "coverage", "difficulty_from_kernel"]


def difficulty_from_kernel(K_test_train: np.ndarray, k_test_self: Sequence[float],
                           K_train: np.ndarray, k: int = 3) -> np.ndarray:
    """How far each scored molecule is from the training set, in [0, 2].

    One minus the mean kernel cosine to the `k` nearest training molecules: 0 for a molecule that
    is a training molecule, larger the further it reaches outside.  This is the quantity the
    applicability domain thresholds, reused here as the difficulty of a prediction, so that the
    two cannot disagree about which molecules are the extrapolations.
    """
    from .validation import domain

    d = domain(K_test_train, k_test_self, K_train, k=k)
    return np.clip(1.0 - np.asarray(d.mean_top, float), 0.0, 2.0)


@dataclass(frozen=True)
class Conformal:
    """A calibrated interval rule: absolute errors on molecules the model did not fit.

    `difficulty` and `beta` turn it into the normalised form, where the nonconformity score is
    |y - y_hat| / (beta + difficulty) and the half width grows with the difficulty of the molecule
    being predicted.  `beta` keeps the denominator away from zero and sets how much of the width
    is fixed: with beta large the rule tends back to the plain one.
    """
    level: float
    residuals: np.ndarray
    kind: str = "cv"
    difficulty: Optional[np.ndarray] = None
    beta: float = 1.0

    def __post_init__(self):
        if not 0.0 < float(self.level) < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        r = np.asarray(self.residuals, float).ravel()
        if self.difficulty is not None:
            d = np.asarray(self.difficulty, float).ravel()
            if len(d) != len(r):
                raise ValueError("one difficulty per calibration molecule")

    # ------------------------------------------------------------------ the quantile
    @property
    def n_calibration(self) -> int:
        return int(np.sum(np.isfinite(np.asarray(self.residuals, float).ravel())))

    @property
    def scores(self) -> np.ndarray:
        """The nonconformity scores: absolute errors, normalised when a difficulty is given."""
        r = np.abs(np.asarray(self.residuals, float).ravel())
        if self.difficulty is None:
            return r[np.isfinite(r)]
        s = r / (float(self.beta) + np.asarray(self.difficulty, float).ravel())
        return s[np.isfinite(s)]

    @property
    def rank(self) -> int:
        """Which order statistic of the scores is taken: ceil((n + 1) * level).

        The +1 is what makes the guarantee finite-sample rather than asymptotic, and it is also
        what makes a small calibration set refuse to answer: 19 errors are exactly enough for a
        95 % interval, since ceil(20 * 0.95) = 19 is the largest of them, and 18 are not enough
        at any width, because the rule would have to extrapolate beyond everything it has seen.
        """
        return int(math.ceil((self.n_calibration + 1) * float(self.level)))

    @property
    def quantile(self) -> float:
        """The half width in units of the nonconformity score; inf when there is too little data."""
        s = np.sort(self.scores)
        if len(s) == 0:
            return float("nan")
        r = self.rank
        if r > len(s):
            return float("inf")         # correct: this level is not reachable with this many
        return float(s[r - 1])

    @property
    def usable(self) -> bool:
        return np.isfinite(self.quantile)

    def min_calibration(self) -> int:
        """The smallest calibration set that could support this level.

        Counted rather than solved: level / (1 - level) is the answer in exact arithmetic and
        9.000000000000002 in floating point, which would reject a calibration set of nine.
        """
        n = int(float(self.level) / (1.0 - float(self.level)))
        while math.ceil((n + 1) * float(self.level)) > n:
            n += 1
        return max(n, 1)

    # ------------------------------------------------------------------ using it
    def half_width(self, difficulty: Optional[Sequence[float]] = None) -> np.ndarray:
        """The half width of the interval, one per molecule when a difficulty is given."""
        q = self.quantile
        if self.difficulty is None or difficulty is None:
            return np.asarray(q, float)
        return q * (float(self.beta) + np.asarray(difficulty, float).ravel())

    def interval(self, y_pred: Sequence[float],
                 difficulty: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """(lower, upper) around the predictions."""
        yp = np.asarray(y_pred, float).ravel()
        h = self.half_width(difficulty)
        h = np.full(len(yp), float(h)) if np.ndim(h) == 0 else np.asarray(h, float).ravel()
        if len(h) != len(yp):
            raise ValueError("one difficulty per predicted molecule")
        return yp - h, yp + h

    def describe(self) -> str:
        q = self.quantile
        how = "out-of-fold" if self.kind == "cv" else "held-out calibration"
        norm = "" if self.difficulty is None else ", normalised by the distance to the training set"
        if not np.isfinite(q):
            return (f"{self.level:.0%} conformal: not reachable from {self.n_calibration} "
                    f"{how} errors; {self.min_calibration()} are needed")
        width = f"half width {q:.3f}" if self.difficulty is None else f"half width {q:.3f} x sigma"
        return (f"{self.level:.0%} conformal from {self.n_calibration} {how} errors{norm}: "
                f"{width}")


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def calibrate(y_true: Sequence[float], y_pred: Sequence[float], level: float = 0.9,
              difficulty: Optional[Sequence[float]] = None, beta: float = 1.0,
              kind: str = "cv") -> Conformal:
    """Calibrate on predictions the model made for molecules it did not fit.

    It is the caller's responsibility that those predictions really are out of sample; calibrating
    on the fit would return an interval that covers the training set and nothing else.
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if len(y_true) != len(y_pred):
        raise ValueError("one prediction per molecule")
    keep = np.isfinite(y_true) & np.isfinite(y_pred)
    d = None if difficulty is None else np.asarray(difficulty, float).ravel()[keep]
    return Conformal(level=float(level), residuals=np.abs(y_true[keep] - y_pred[keep]),
                     kind=str(kind), difficulty=d, beta=float(beta))


def calibrate_cv(K: np.ndarray, y: Sequence[float], method: str, params: Dict[str, float],
                 level: float = 0.9, folds: int = 10, seed: Optional[int] = 0,
                 fit_bias: bool = True, difficulty: Optional[Sequence[float]] = None,
                 beta: float = 1.0) -> Conformal:
    """Calibrate from the out-of-fold predictions of the cross-validation already being run."""
    from .model import cv_predict

    y = np.asarray(y, float).ravel()
    yp = cv_predict(np.asarray(K, float), y, method, params, folds=folds, seed=seed,
                    fit_bias=fit_bias)
    return calibrate(y, yp, level=level, difficulty=difficulty, beta=beta, kind="cv")


def coverage(y_true: Sequence[float], lo: Sequence[float], hi: Sequence[float]) -> Dict[str, float]:
    """What the intervals actually did: the share covered, and how wide they had to be."""
    y = np.asarray(y_true, float).ravel()
    lo = np.asarray(lo, float).ravel()
    hi = np.asarray(hi, float).ravel()
    keep = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if not keep.any():
        return {"n": 0, "coverage": float("nan"), "mean_width": float("nan"),
                "median_width": float("nan")}
    inside = (y[keep] >= lo[keep]) & (y[keep] <= hi[keep])
    width = hi[keep] - lo[keep]
    return {"n": int(keep.sum()), "coverage": float(np.mean(inside)),
            "mean_width": float(np.mean(width)), "median_width": float(np.median(width))}
