"""A warped Gaussian process: a monotone transform of the activity, learned with the GP (plan
item ND11).

A Gaussian process assumes that the activity, less what the kernel explains, is Gaussian noise
on one scale. A pIC50 series is often not like that: a skewed scale, a floor of inactive
compounds, a handful of very active ones. Snelson, Ghahramani and Rasmussen (NIPS 2003) fit a
monotone, invertible transform of the response *together with* the process, so that the
transformed activity is what the GP models, and every prediction is carried back through the
inverse. The transform has a handful of parameters, so it fits thirty molecules; the model stays
linear in the field in the warped units, which is where the co-field and the decomposition rules
are read, and the output says so.

**The transform.** With ``u = (y - centre) / width`` the standardised activity,

    g(y) = u + sum_i a_i tanh(b_i (u + c_i)),      a_i > 0, b_i > 0,

a sum of a straight line and a few smooth steps: each term stretches the scale around ``-c_i``
and compresses it far away, which is what a skewed or a censored scale needs. ``g'(y) > 0``
everywhere, so the inverse exists, and it is found by bisection, which is exact to machine
precision in sixty steps because the inverse lies within ``sum_i a_i`` of ``z``.

**The fit.** The parameters maximise the log marginal likelihood of the warped activities under
the kernel, with the amplitude profiled out and the constant mean marginalised as
:func:`cmfqsar.evidence.log_evidence` does it, plus the Jacobian ``sum_i log g'(y_i)`` that makes
it the likelihood of the *original* activities. Because the amplitude is profiled, the objective
does not change when the whole transform is scaled, so the only freedom that is fitted is the
shape. The identity (``a = 0``) is always a candidate, and :func:`fit_warp` reports by how much
the warp beat it, so a series whose activities are already close to normal can be left alone.

**The reporting.** The GP gives a mean and a deviation in warped units. The prediction returned
in the original units is the *median*, ``g^-1(mean)``, which is what the inverse of a symmetric
interval's centre is; an interval is ``[g^-1(mean - k sd), g^-1(mean + k sd)]``, which is no
longer symmetric about the prediction, and that asymmetry is the point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: how many tanh terms the warp has by default: enough for a skew and a floor, few enough for
#: thirty molecules
DEFAULT_TERMS = 2
#: the smallest amplitude a term may have before the warp is called the identity
IDENTITY_AMPLITUDE = 1e-6


def parameter_penalty(n: int, n_terms: int = DEFAULT_TERMS) -> float:
    """The log evidence a warp has to gain before it is kept: half its parameter count times
    ``log n``, the penalty of the Bayesian information criterion.

    Six free parameters gain a few nats on any series, Gaussian or not; on sixty activities drawn
    from a Gaussian the fit gains about four where this asks for twelve, so a series whose
    activities are already close to normal keeps the identity, which is what plan item ND11
    requires of it.
    """
    return 0.5 * 3.0 * float(n_terms) * float(np.log(max(int(n), 2)))


@dataclass
class Warp:
    """A monotone transform of the activity, ``g(y) = u + sum a tanh(b (u + c))``.

    ``centre`` and ``width`` standardise the activity first (the median and a robust scale of
    the training activities), so that the term parameters live on one scale whatever the units.
    ``gain`` is the log marginal likelihood the fit gained over the identity transform, in nats;
    zero for the identity itself.
    """

    centre: float
    width: float
    a: np.ndarray = field(default_factory=lambda: np.zeros(0))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))
    c: np.ndarray = field(default_factory=lambda: np.zeros(0))
    gain: float = 0.0

    def __post_init__(self):
        self.a = np.asarray(self.a, float).ravel()
        self.b = np.asarray(self.b, float).ravel()
        self.c = np.asarray(self.c, float).ravel()
        if not (len(self.a) == len(self.b) == len(self.c)):
            raise ValueError("a, b and c must have one value per term")
        if np.any(self.a < 0) or np.any(self.b <= 0):
            raise ValueError("the amplitudes must be non-negative and the steepnesses positive, "
                             "or the transform is not monotone")
        if not (self.width > 0):
            raise ValueError("the width must be positive")

    # ------------------------------------------------------------------ the transform
    @property
    def n_terms(self) -> int:
        return int(len(self.a))

    @property
    def is_identity(self) -> bool:
        """True when every term is too small to matter: the warp is the standardisation alone."""
        return bool(self.n_terms == 0 or np.all(self.a < IDENTITY_AMPLITUDE))

    def _u(self, y) -> np.ndarray:
        return (np.asarray(y, float) - self.centre) / self.width

    def forward(self, y) -> np.ndarray:
        """``g(y)``: the activity in warped units."""
        u = self._u(y)
        z = u.copy()
        for a, b, c in zip(self.a, self.b, self.c):
            z = z + a * np.tanh(b * (u + c))
        return z

    def derivative(self, y) -> np.ndarray:
        """``dg/dy``, positive everywhere."""
        u = self._u(y)
        d = np.ones_like(u)
        for a, b, c in zip(self.a, self.b, self.c):
            t = np.tanh(b * (u + c))
            d = d + a * b * (1.0 - t * t)
        return d / self.width

    def inverse(self, z) -> np.ndarray:
        """``g^-1(z)``: back to the original units, by bisection on the monotone transform."""
        z = np.asarray(z, float)
        reach = float(np.sum(self.a))
        lo = z - reach - 1e-9
        hi = z + reach + 1e-9
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            g = mid.copy()
            for a, b, c in zip(self.a, self.b, self.c):
                g = g + a * np.tanh(b * (mid + c))
            below = g < z
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        return 0.5 * (lo + hi) * self.width + self.centre

    def log_jacobian(self, y) -> float:
        """``sum log g'(y_i)``: what makes the warped likelihood the likelihood of ``y``."""
        return float(np.sum(np.log(self.derivative(y))))

    # ------------------------------------------------------------------ the reporting
    def interval(self, mean_z: Sequence[float], sd_z: Sequence[float],
                 level: float = 0.9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(median, lower, upper) in the original units from a mean and deviation in warped
        units: the central interval of the given level, carried back through the inverse."""
        from scipy.stats import norm
        mean_z = np.asarray(mean_z, float)
        sd_z = np.asarray(sd_z, float)
        k = float(norm.ppf(0.5 + 0.5 * float(level)))
        return (self.inverse(mean_z), self.inverse(mean_z - k * sd_z),
                self.inverse(mean_z + k * sd_z))

    def describe(self) -> str:
        if self.is_identity:
            return (f"warp: the identity (activities standardised about {self.centre:.3g} with "
                    f"width {self.width:.3g}); a transform would have gained {self.gain:+.2f} "
                    f"nats of evidence, which is under the penalty for its parameters")
        terms = ", ".join(f"{a:.3g} tanh({b:.3g} (u {c:+.3g}))"
                          for a, b, c in zip(self.a, self.b, self.c))
        return (f"warp: g(y) = u + {terms}, u = (y - {self.centre:.3g}) / {self.width:.3g}; "
                f"log evidence gained over the identity {self.gain:+.2f}")

    def to_dict(self) -> Dict[str, object]:
        return {"centre": float(self.centre), "width": float(self.width),
                "a": [float(v) for v in self.a], "b": [float(v) for v in self.b],
                "c": [float(v) for v in self.c], "gain": float(self.gain)}

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "Warp":
        return cls(centre=float(d["centre"]), width=float(d["width"]),
                   a=np.asarray(d.get("a", []), float), b=np.asarray(d.get("b", []), float),
                   c=np.asarray(d.get("c", []), float), gain=float(d.get("gain", 0.0)))

    @classmethod
    def identity(cls, y: Sequence[float]) -> "Warp":
        """The standardisation alone, from the training activities."""
        centre, width = _standardisation(y)
        return cls(centre=centre, width=width)


def _standardisation(y: Sequence[float]) -> Tuple[float, float]:
    """The median and a robust width (the interquartile range over 1.349, or the standard
    deviation when that is degenerate) of the activities."""
    y = np.asarray(y, float).ravel()
    centre = float(np.median(y))
    q1, q3 = np.percentile(y, [25.0, 75.0])
    width = float((q3 - q1) / 1.349)
    if not width > 1e-12:
        width = float(np.std(y, ddof=1)) if len(y) > 1 else 1.0
    if not width > 1e-12:
        width = 1.0
    return centre, width


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #
def warped_log_evidence(K: np.ndarray, y: Sequence[float], lam: float, warp: Warp,
                        fit_bias: bool = True) -> float:
    """The log marginal likelihood of the original activities under the warped process.

    ``log p(y) = log N(g(y) | 0, s(K + lam I)) + sum log g'(y)`` with the amplitude ``s``
    profiled out and the constant mean marginalised, as :func:`cmfqsar.evidence.log_evidence`
    computes the first term. Scaling ``g`` by a constant leaves this unchanged.
    """
    from .evidence import log_evidence
    y = np.asarray(y, float).ravel()
    e = log_evidence(K, warp.forward(y), float(lam), fit_bias=fit_bias, profile=True)
    if not np.isfinite(e):
        return float("-inf")
    return float(e + warp.log_jacobian(y))


def _unpack(theta: np.ndarray, n_terms: int, centre: float, width: float) -> Warp:
    th = np.asarray(theta, float)
    a = np.exp(th[:n_terms])
    b = np.exp(th[n_terms:2 * n_terms])
    c = th[2 * n_terms:3 * n_terms]
    return Warp(centre=centre, width=width, a=a, b=b, c=c)


def fit_warp(K: np.ndarray, y: Sequence[float], lam: float, n_terms: int = DEFAULT_TERMS,
             fit_bias: bool = True, restarts: int = 2, seed: Optional[int] = 0,
             min_gain: Optional[float] = None) -> Warp:
    """The transform that maximises the warped marginal likelihood on this kernel.

    Parameters
    ----------
    K, y, lam : the training kernel, the activities and the ridge, as the GP fit has them.
    n_terms : how many tanh terms; two by default.
    restarts : how many random starting points are tried besides the deterministic one; the
        objective has local optima when a term can sit on either side of the data.
    min_gain : the log evidence the warp must gain over the identity to be kept, in nats; by
        default the parameter penalty of :func:`parameter_penalty`, and ``0.0`` keeps any gain.
        The identity is returned when the warp does not clear it, with ``gain`` set to what it
        did gain, so the shortfall can still be read.

    The identity transform is always evaluated, and the result's ``gain`` says by how much the
    warp beat it, which is what the benchmark of this item reads.
    """
    from scipy.optimize import minimize

    y = np.asarray(y, float).ravel()
    K = np.asarray(K, float)
    n = len(y)
    if K.shape != (n, n):
        raise ValueError(f"the kernel is {K.shape} but there are {n} activities")
    if n < 4:
        raise ValueError("a warp needs at least four activities")
    if not np.all(np.isfinite(y)):
        raise ValueError("every training activity must be measured to fit a warp")
    centre, width = _standardisation(y)
    base = Warp(centre=centre, width=width)
    e0 = warped_log_evidence(K, y, lam, base, fit_bias)
    if int(n_terms) <= 0 or not np.isfinite(e0):
        return base
    u = (y - centre) / width
    quantiles = np.quantile(u, np.linspace(0.2, 0.8, int(n_terms)))

    def objective(theta):
        try:
            w = _unpack(theta, int(n_terms), centre, width)
        except ValueError:
            return 1e300
        e = warped_log_evidence(K, y, lam, w, fit_bias)
        return 1e300 if not np.isfinite(e) else -e

    rng = np.random.RandomState(0 if seed is None else int(seed))
    starts = [np.concatenate([np.log(np.full(int(n_terms), 0.3)),
                              np.zeros(int(n_terms)), -quantiles])]
    for _ in range(max(int(restarts), 0)):
        starts.append(np.concatenate([np.log(rng.uniform(0.05, 1.0, int(n_terms))),
                                      np.log(rng.uniform(0.3, 3.0, int(n_terms))),
                                      -quantiles + rng.normal(scale=0.5, size=int(n_terms))]))
    bounds = ([(np.log(1e-4), np.log(20.0))] * int(n_terms)
              + [(np.log(0.05), np.log(50.0))] * int(n_terms)
              + [(float(u.min()) - 5.0, float(-u.min()) + 5.0)] * int(n_terms))
    bounds[2 * int(n_terms):] = [(-float(u.max()) - 5.0, -float(u.min()) + 5.0)] * int(n_terms)
    best: Optional[Tuple[float, np.ndarray]] = None
    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 400})
        if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
            best = (float(res.fun), np.asarray(res.x, float))
    if best is None:
        return base
    gain = -best[0] - e0
    threshold = parameter_penalty(n, int(n_terms)) if min_gain is None else float(min_gain)
    if not np.isfinite(gain) or gain <= threshold:
        base.gain = float(gain) if np.isfinite(gain) else 0.0
        return base
    w = _unpack(best[1], int(n_terms), centre, width)
    w.gain = float(gain)
    return w


def cross_validate_warped(K: np.ndarray, y: Sequence[float], lam: float,
                          n_terms: int = DEFAULT_TERMS, folds: int = 10,
                          seed: Optional[int] = 0, fit_bias: bool = True,
                          restarts: int = 2, min_gain: Optional[float] = None
                          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold (prediction in the original units, mean in warped units, deviation in warped
    units, the held-out activities in the warped units of their own fold), with the warp
    refitted on the training part of every fold.

    The folds are those of the model's own cross-validation for the same seed, so the
    predictions compare with the plain fit's fold by fold. The deviation is the one
    :class:`cmfqsar.surrogate.Surrogate` gives, in the warped units; an interval in the original
    units is :meth:`Warp.interval` applied fold by fold, which :func:`cross_validate_intervals`
    does.
    """
    from .model import _cv_splits
    from .surrogate import Surrogate

    y = np.asarray(y, float).ravel()
    K = np.asarray(K, float)
    diag = np.diag(K)
    pred = np.full(len(y), np.nan)
    mean_z = np.full(len(y), np.nan)
    sd_z = np.full(len(y), np.nan)
    z_true = np.full(len(y), np.nan)
    for te, tr in _cv_splits(y, folds, seed):
        w = fit_warp(K[np.ix_(tr, tr)], y[tr], lam, n_terms=n_terms, fit_bias=fit_bias,
                     restarts=restarts, seed=seed, min_gain=min_gain)
        s = Surrogate.fit(K[np.ix_(tr, tr)], w.forward(y[tr]), lam, fit_bias)
        m, sd = s.predict(K[np.ix_(te, tr)], diag[te])
        mean_z[te], sd_z[te] = m, sd
        z_true[te] = w.forward(y[te])
        pred[te] = w.inverse(m)
    return pred, mean_z, sd_z, z_true


def cross_validate_intervals(K: np.ndarray, y: Sequence[float], lam: float, level: float = 0.9,
                             n_terms: int = DEFAULT_TERMS, folds: int = 10,
                             seed: Optional[int] = 0, fit_bias: bool = True, restarts: int = 2,
                             min_gain: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Out-of-fold intervals of the warped and the plain process at one level, in the original
    units, with the warp refitted on the training part of every fold.

    Returns a dictionary with ``warped`` and ``plain`` entries, each (prediction, lower, upper),
    and ``gain``, the evidence the warp gained over the identity in every fold.
    """
    from scipy.stats import norm

    from .model import _cv_splits
    from .surrogate import Surrogate

    y = np.asarray(y, float).ravel()
    K = np.asarray(K, float)
    diag = np.diag(K)
    k = float(norm.ppf(0.5 + 0.5 * float(level)))
    out = {name: tuple(np.full(len(y), np.nan) for _ in range(3)) for name in ("warped", "plain")}
    gains = []
    for te, tr in _cv_splits(y, folds, seed):
        w = fit_warp(K[np.ix_(tr, tr)], y[tr], lam, n_terms=n_terms, fit_bias=fit_bias,
                     restarts=restarts, seed=seed, min_gain=min_gain)
        gains.append(w.gain)
        s = Surrogate.fit(K[np.ix_(tr, tr)], w.forward(y[tr]), lam, fit_bias)
        m, sd = s.predict(K[np.ix_(te, tr)], diag[te])
        med, lo, hi = w.interval(m, sd, level)
        out["warped"][0][te], out["warped"][1][te], out["warped"][2][te] = med, lo, hi
        s0 = Surrogate.fit(K[np.ix_(tr, tr)], y[tr], lam, fit_bias)
        m0, sd0 = s0.predict(K[np.ix_(te, tr)], diag[te])
        out["plain"][0][te] = m0
        out["plain"][1][te] = m0 - k * sd0
        out["plain"][2][te] = m0 + k * sd0
    result: Dict[str, object] = dict(out)
    result["gain"] = np.asarray(gains, float)
    return result
