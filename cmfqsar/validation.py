"""
The three questions a 3D-QSAR model has to answer besides "how good is q²".

**Is it chance?**  A kernel built from molecular fields can fit anything if it is flexible enough,
and a congeneric series whose kernel values all exceed 0.97 is exactly the setting in which a
model can look respectable while carrying no information.  ``y_scramble`` shuffles the activities
and refits, many times, and reports where the real statistic falls in the distribution of
scrambled ones.  This is the check the literature calls y-randomisation, and the program had
everything needed for it except the loop.

**Does the prediction mean anything for *this* molecule?**  A kernel method always returns a
number, and for a molecule unlike anything in the training set that number is the training mean
dressed up.  ``domain`` measures how close a molecule actually is to the training set, in the only
geometry the model knows -- the field kernel itself -- and compares that with how close the
training molecules are to each other.

**How certain is the statistic?**  A difference of 0.02 in q² between two settings is quoted all
over this repository's benchmarks, and it is worth knowing when such a difference is noise.
``bootstrap_ci`` resamples the predictions, ``bootstrap_refit`` resamples the training set and
refits, and both return a percentile interval.

Nothing here needs the fields themselves, only kernels that have already been computed, so all of
it is cheap compared with the fit it validates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .model import cv_predict, r2_score, rmse, _Learner

ProgressFn = Callable[[str, float], None]
CancelFn = Callable[[], bool]


# --------------------------------------------------------------------------- #
# 1. y-randomisation
# --------------------------------------------------------------------------- #

@dataclass
class ScrambleResult:
    """Where the real statistic falls among the ones obtained from shuffled activities."""
    statistic: str = "q2"
    observed: float = float("nan")
    scrambled: np.ndarray = field(default_factory=lambda: np.zeros(0))
    runs: int = 0
    reselected: bool = False          # was the model selection redone on each shuffle?

    @property
    def mean(self) -> float:
        return float(np.mean(self.scrambled)) if len(self.scrambled) else float("nan")

    @property
    def sd(self) -> float:
        return float(np.std(self.scrambled, ddof=1)) if len(self.scrambled) > 1 else float("nan")

    @property
    def p_value(self) -> float:
        """Fraction of shuffles that did as well as the real activities, with the usual +1.

        The +1 in numerator and denominator is the standard correction for a permutation test:
        with 20 shuffles the smallest reportable p is 1/21, never 0, because 20 shuffles cannot
        demonstrate more than that.
        """
        if not len(self.scrambled):
            return float("nan")
        better = int(np.sum(self.scrambled >= self.observed))
        return (better + 1) / (len(self.scrambled) + 1)

    @property
    def z_score(self) -> float:
        """How many standard deviations above the scrambled mean the real statistic lies."""
        sd = self.sd
        if not np.isfinite(sd) or sd <= 0:
            return float("nan")
        return (self.observed - self.mean) / sd

    def caution(self) -> Optional[str]:
        """What the result warrants saying, which is not always what the p-value suggests.

        A permutation test cannot report a p below 1/(runs+1), so a handful of shuffles produces
        a large p even for a model that no shuffle came close to.  Saying "not distinguishable
        from chance" there would be a false alarm; the honest message is that too few shuffles
        were run to say.
        """
        if not len(self.scrambled):
            return None
        if float(np.max(self.scrambled)) >= self.observed:
            return ("WARNING: a shuffle of the activities did as well as the real ones. This "
                    "model is not distinguishable from chance correlation.")
        floor = 1.0 / (self.runs + 1)
        if floor > 0.05:
            return (f"NOTE: no shuffle came close, but {self.runs} of them cannot support a "
                    f"p below {floor:.3f}. Use at least 19 shuffles for p < 0.05, 99 for "
                    f"p < 0.01.")
        if self.p_value > 0.05:
            return ("WARNING: shuffled activities reach this value too often; the model is not "
                    "clearly better than chance correlation.")
        return None

    def describe(self) -> str:
        if not len(self.scrambled):
            return "y-randomisation: not run"
        how = "with the model selection redone" if self.reselected else "at fixed hyper-parameters"
        return (f"y-randomisation ({self.runs} shuffles, {how}): "
                f"{self.statistic} {self.observed:+.3f} against {self.mean:+.3f} "
                f"+/- {self.sd:.3f} scrambled "
                f"(max {np.max(self.scrambled):+.3f}, z = {self.z_score:.1f}, "
                f"p = {self.p_value:.3f})")


def y_scramble(score: Callable[[np.ndarray], float], y: Sequence[float], runs: int = 20,
               seed: int = 0, statistic: str = "q2", reselected: bool = False,
               progress: Optional[ProgressFn] = None,
               cancel: Optional[CancelFn] = None) -> ScrambleResult:
    """Score the real activities, then `runs` shuffles of them, with the same procedure.

    `score` is whatever the caller counts as the model's figure of merit: a closure that takes an
    activity vector and returns a number, larger being better.  Pass one that only refits at fixed
    hyper-parameters and the test asks whether the *fit* can be achieved by chance; pass one that
    repeats the grid search and it asks the stricter question, whether the whole procedure --
    selection included -- can be.  `reselected` only records which of the two was done, for the
    report.
    """
    y = np.asarray(y, dtype=float)
    observed = float(score(y))
    rng = np.random.RandomState(seed)
    out: List[float] = []
    for r in range(int(runs)):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(f"y-randomisation {r + 1}/{runs}", (r + 1) / max(int(runs), 1))
        out.append(float(score(rng.permutation(y))))
    return ScrambleResult(statistic=statistic, observed=observed, scrambled=np.asarray(out),
                          runs=len(out), reselected=reselected)


def scramble_kernel(K: np.ndarray, y: Sequence[float], method: str = "krr",
                    params: Optional[Dict[str, float]] = None, folds: int = 10,
                    cv_seed: Optional[int] = 0, runs: int = 20, seed: int = 0,
                    progress: Optional[ProgressFn] = None,
                    cancel: Optional[CancelFn] = None) -> ScrambleResult:
    """y-randomisation of a cross-validated model on a precomputed kernel."""
    params = dict(params or {"lam": 1.0})

    def score(yy: np.ndarray) -> float:
        pred = cv_predict(K, yy, method, params, folds=folds, seed=cv_seed)
        return r2_score(yy, pred)

    return y_scramble(score, y, runs=runs, seed=seed, statistic=f"q2 ({folds}-fold)",
                      progress=progress, cancel=cancel)


# --------------------------------------------------------------------------- #
# 2. Applicability domain
# --------------------------------------------------------------------------- #

@dataclass
class DomainResult:
    """How close each scored molecule is to the training set, in the model's own geometry."""
    similarity: np.ndarray = field(default_factory=lambda: np.zeros(0))   # to the nearest
    mean_top: np.ndarray = field(default_factory=lambda: np.zeros(0))     # to the k nearest
    distance: np.ndarray = field(default_factory=lambda: np.zeros(0))     # in field space
    threshold: float = float("nan")
    k: int = 3

    @property
    def inside(self) -> np.ndarray:
        if not np.isfinite(self.threshold):
            return np.ones(len(self.similarity), dtype=bool)
        return self.mean_top >= self.threshold

    def describe(self) -> str:
        if not len(self.similarity):
            return "applicability domain: nothing scored"
        n_out = int(np.sum(~self.inside))
        return (f"Applicability domain (mean similarity to the {self.k} nearest training "
                f"molecules, threshold {self.threshold:.3f} from the training set itself): "
                f"{len(self.similarity) - n_out}/{len(self.similarity)} inside, {n_out} outside")


def _cosine(K_cross: np.ndarray, diag_rows: np.ndarray, diag_cols: np.ndarray) -> np.ndarray:
    """Kernel values turned into cosines, which is what makes a threshold transferable."""
    a = np.sqrt(np.maximum(np.asarray(diag_rows, float), 0.0))
    b = np.sqrt(np.maximum(np.asarray(diag_cols, float), 0.0))
    denom = np.outer(a, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denom > 1e-300, np.asarray(K_cross, float) / denom, 0.0)


def domain_threshold(K_train: np.ndarray, k: int = 3, quantile: float = 0.05) -> float:
    """The similarity below which a molecule is outside the domain, taken from the training set.

    Every training molecule is compared with the others, its mean similarity to its `k` nearest
    neighbours is taken, and the `quantile` point of that distribution becomes the threshold.  The
    domain is then defined by the training set's own spread rather than by a number chosen out of
    the air: a series of close analogues gets a tight domain, a diverse set a loose one.
    """
    K = np.asarray(K_train, float)
    n = K.shape[0]
    if n < 2:
        return float("nan")
    S = _cosine(K, np.diag(K), np.diag(K))
    np.fill_diagonal(S, -np.inf)                     # a molecule is not its own neighbour
    kk = int(max(1, min(k, n - 1)))
    nearest = np.sort(S, axis=1)[:, -kk:]
    return float(np.quantile(nearest.mean(axis=1), float(quantile)))


def domain(K_test_train: np.ndarray, k_test_self: Sequence[float], K_train: np.ndarray,
           k: int = 3, quantile: float = 0.05,
           threshold: Optional[float] = None) -> DomainResult:
    """Applicability domain of a set of molecules against the training set of a fitted model.

    Everything is computed from kernels that the prediction needed anyway: the cross kernel
    between the scored molecules and the training set, the scored molecules' self-kernels, and the
    training kernel matrix.  Distances are the field-space distances the kernel implies,
    ``|X - X_j|^2 = K(x,x) + K(j,j) - 2K(x,j)``, which is the same quantity the Gaussian kernel
    layer is built on.
    """
    K_xt = np.atleast_2d(np.asarray(K_test_train, float))
    diag_train = np.diag(np.asarray(K_train, float))
    diag_test = np.asarray(k_test_self, float).ravel()
    S = _cosine(K_xt, diag_test, diag_train)
    kk = int(max(1, min(k, K_xt.shape[1])))
    nearest = np.sort(S, axis=1)[:, -kk:]
    d2 = diag_test[:, None] + diag_train[None, :] - 2.0 * K_xt
    thr = domain_threshold(K_train, k=kk, quantile=quantile) if threshold is None \
        else float(threshold)
    return DomainResult(similarity=S.max(axis=1), mean_top=nearest.mean(axis=1),
                        distance=np.sqrt(np.maximum(d2.min(axis=1), 0.0)), threshold=thr, k=kk)


# --------------------------------------------------------------------------- #
# 3. Bootstrap
# --------------------------------------------------------------------------- #

@dataclass
class BootstrapResult:
    """A percentile confidence interval for one statistic."""
    statistic: str = "q2"
    value: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    level: float = 0.95
    runs: int = 0
    refit: bool = False
    samples: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def describe(self) -> str:
        if not self.runs:
            return "bootstrap: not run"
        how = "refitting on resampled training sets" if self.refit else "resampling the predictions"
        return (f"{self.statistic} = {self.value:+.3f}, {self.level:.0%} bootstrap interval "
                f"[{self.lo:+.3f}, {self.hi:+.3f}] ({self.runs} resamples, {how})")


def _interval(samples: np.ndarray, level: float):
    lo = float(np.quantile(samples, (1.0 - level) / 2.0))
    hi = float(np.quantile(samples, 1.0 - (1.0 - level) / 2.0))
    return lo, hi


def bootstrap_ci(y: Sequence[float], y_pred: Sequence[float], statistic: str = "q2",
                 runs: int = 1000, seed: int = 0, level: float = 0.95,
                 y_ref_mean: Optional[float] = None) -> BootstrapResult:
    """Percentile interval for q2, R2pred or RMSE, by resampling the (observed, predicted) pairs.

    This is the uncertainty of the *statistic* given these predictions -- how much of the number
    is the particular molecules it was computed over -- and it is what tells you whether two
    settings that differ by 0.02 in q2 differ at all.  It does not refit anything, so it does not
    capture the variability of the fit itself; ``bootstrap_refit`` does that.
    """
    y = np.asarray(y, float)
    yp = np.asarray(y_pred, float)
    ok = np.isfinite(y) & np.isfinite(yp)
    y, yp = y[ok], yp[ok]
    n = len(y)
    if n < 3:
        return BootstrapResult(statistic=statistic, runs=0)
    fn = (lambda a, b: rmse(a, b)) if statistic.lower().startswith("rmse") else \
        (lambda a, b: r2_score(a, b, y_ref_mean))
    rng = np.random.RandomState(seed)
    out = np.empty(int(runs))
    for r in range(int(runs)):
        idx = rng.randint(0, n, n)
        out[r] = fn(y[idx], yp[idx])
    lo, hi = _interval(out, level)
    return BootstrapResult(statistic=statistic, value=fn(y, yp), lo=lo, hi=hi, level=level,
                           runs=int(runs), refit=False, samples=out)


def bootstrap_refit(K: np.ndarray, y: Sequence[float], method: str = "krr",
                    params: Optional[Dict[str, float]] = None, runs: int = 100, seed: int = 0,
                    level: float = 0.95, fit_bias: bool = True,
                    progress: Optional[ProgressFn] = None,
                    cancel: Optional[CancelFn] = None) -> BootstrapResult:
    """Cramer's bootstrap: resample the training set with replacement, refit, score what was left.

    Each resample fits on the molecules drawn and predicts the ones that were not (about a third
    of the set), so the statistic is measured out of bag rather than on the molecules the model
    just saw.  The kernel is never recomputed -- a resample is an index into the matrix -- so the
    cost is one linear solve per run.
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    params = dict(params or {"lam": 1.0})
    rng = np.random.RandomState(seed)
    out: List[float] = []
    for r in range(int(runs)):
        if cancel is not None and cancel():
            break
        if progress is not None:
            progress(f"Bootstrap {r + 1}/{runs}", (r + 1) / max(int(runs), 1))
        idx = rng.randint(0, n, n)
        oob = np.setdiff1d(np.arange(n), np.unique(idx))
        if len(oob) < 3 or len(np.unique(y[idx])) < 2:
            continue
        learner = _Learner(method, params, fit_bias).fit(K[np.ix_(idx, idx)], y[idx])
        pred = learner.predict(K[np.ix_(oob, idx)])
        out.append(r2_score(y[oob], pred))
    if not out:
        return BootstrapResult(statistic="q2 (out of bag)", runs=0, refit=True)
    arr = np.asarray(out)
    lo, hi = _interval(arr, level)
    return BootstrapResult(statistic="q2 (out of bag)", value=float(np.mean(arr)), lo=lo, hi=hi,
                           level=level, runs=len(arr), refit=True, samples=arr)


def scramble_selection(kernels: Sequence, y: Sequence[float], method: str = "krr",
                       param_grid: Sequence[Dict[str, float]] = ({"lam": 1.0},),
                       folds: int = 10, cv_seed: Optional[int] = 0, runs: int = 20,
                       seed: int = 0, progress: Optional[ProgressFn] = None,
                       cancel: Optional[CancelFn] = None) -> ScrambleResult:
    """y-randomisation with the model selection repeated inside every shuffle.

    `kernels` is a sequence of precomputed joint kernels -- one per attenuation, say -- and
    `param_grid` the regularisations tried with each.  Every shuffle is scored by the *best* q2
    over that grid, exactly as the real activities were, so the test asks whether the whole
    procedure can reach the reported number by chance rather than whether a single fit can.  That
    is the stricter and more honest question; it costs one cross-validation per grid point per
    shuffle, which is affordable only because the kernels do not depend on the activities and are
    therefore computed once.
    """
    mats = [np.asarray(K, float) for K in kernels]
    grid = [dict(p) for p in param_grid] or [{"lam": 1.0}]

    def score(yy: np.ndarray) -> float:
        best = -np.inf
        for K in mats:
            for params in grid:
                pred = cv_predict(K, yy, method, params, folds=folds, seed=cv_seed)
                best = max(best, r2_score(yy, pred))
        return float(best)

    res = y_scramble(score, y, runs=runs, seed=seed,
                     statistic=f"best q2 over the grid ({folds}-fold)", reselected=True,
                     progress=progress, cancel=cancel)
    return res
