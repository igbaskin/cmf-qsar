"""Which of the things that claim to know when a prediction is bad actually knows (plan item E5).

Three quantities in this program offer to tell a chemist that a particular prediction should not be
trusted, and they come from three different places:

* the **Gaussian-process posterior variance**, which is the kernel's own statement about how much
  of this molecule lies in the span of the training set (:meth:`CMFModel.predict_with_variance`);
* the **applicability domain**, a geometric statement about how far the molecule is from its
  nearest training neighbours in field space (:mod:`cmfqsar.validation`);
* the **conformal interval**, whose normalised form widens the interval of a molecule the model
  finds difficult (:mod:`cmfqsar.conformal`).

They are in different units -- a variance, an Angstrom-like distance in field space, a width in log
units of activity -- so they cannot be compared by their numbers.  What can be compared is the
*order* they put the molecules in, because that is all a chemist uses them for: which prediction do
I act on, which do I measure first.  So every estimate is scored against the absolute error it was
supposed to anticipate, by three measures that answer three different questions.

**Does it order the molecules correctly?**  Spearman's rho and Kendall's tau between the estimate
and ``|y - yhat|``.  Scale-free, and unaware of how big the errors are.

**What is gained by acting on it?**  The risk--coverage curve: trust the most confident fraction of
the molecules and measure the RMSE of that fraction.  A useful estimate makes the curve rise with
coverage; a useless one leaves it flat at the overall RMSE.  The area under it (AURC) is one
number, and it is read against two references -- the *oracle*, which orders the molecules by the
error they actually made, and *random*, which is the flat curve.  The efficiency ::

    (AURC_random - AURC) / (AURC_random - AURC_oracle)

is then 1 for the oracle, 0 for an estimate that is no better than ignoring it, and negative for one
that is actively misleading.  It is the number to quote: unlike a correlation, it is in the units of
the thing a chemist cares about and it says how much of the achievable separation was captured.

**What would a two-bin rule do?**  The RMSE of the half the estimate trusts, against the RMSE of
the half it doubts.  This is the crudest and the most practical: the ratio is what "only act on the
confident half" is worth.

A constant estimate -- the unnormalised conformal interval, which gives every molecule the same
width -- is detected and reported as constant rather than scored as if its arbitrary ordering meant
something.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

MIN_COVERAGE = 5          # molecules; an RMSE over fewer than this is noise, not a risk


@dataclass
class PredictorScore:
    """How well one estimate anticipated the errors the model actually made."""

    name: str
    n: int
    spearman: float
    kendall: float
    aurc: float
    aurc_oracle: float
    aurc_random: float
    efficiency: float
    rmse_trusted: float
    rmse_doubted: float
    constant: bool = False

    @property
    def ratio(self) -> float:
        """RMSE of the doubted half over the trusted half; 1 means the split gave no information."""
        return float(self.rmse_doubted / self.rmse_trusted) if self.rmse_trusted > 0 else float("nan")

    def describe(self) -> str:
        if self.constant:
            return (f"{self.name}: the same for every molecule, so it orders nothing "
                    f"(RMSE {self.rmse_trusted:.3f} either way)")
        return (f"{self.name}: rho {self.spearman:+.3f}, efficiency {self.efficiency:+.3f}, "
                f"RMSE {self.rmse_trusted:.3f} on the half it trusts against "
                f"{self.rmse_doubted:.3f} on the half it doubts (x{self.ratio:.2f})")


def risk_coverage(estimate: Sequence[float], error: Sequence[float],
                  min_n: int = MIN_COVERAGE) -> Tuple[np.ndarray, np.ndarray]:
    """(coverage, RMSE) of the most-confident fraction, for every fraction of at least `min_n`.

    The molecules are ordered by `estimate` ascending, so the first prefix is the one the estimate
    is surest about; the RMSE of each prefix is the risk a chemist would take by acting on exactly
    that many predictions.  Ties keep their input order, which is the only neutral choice.
    """
    e = np.asarray(estimate, float)
    r = np.asarray(error, float)
    order = np.argsort(e, kind="stable")
    sq = r[order] ** 2
    n = len(sq)
    k = np.arange(1, n + 1)
    rmse = np.sqrt(np.cumsum(sq) / k)
    lo = min(max(1, int(min_n)), n)
    return k[lo - 1:] / n, rmse[lo - 1:]


def aurc(estimate: Sequence[float], error: Sequence[float], min_n: int = MIN_COVERAGE) -> float:
    """Area under the risk--coverage curve, as the mean RMSE over the coverages it is defined at.

    Every coverage point counts the same, which weights the small, confident subsets as heavily as
    the whole set; that is deliberate, since the point of the estimate is to be right about the
    confident end.
    """
    _cov, rm = risk_coverage(estimate, error, min_n=min_n)
    return float(np.mean(rm))


def score_predictor(name: str, estimate: Sequence[float], error: Sequence[float],
                    min_n: int = MIN_COVERAGE) -> PredictorScore:
    """Score one estimate against the errors, by order, by risk--coverage and by halves."""
    from scipy.stats import kendalltau, spearmanr

    e = np.asarray(estimate, float).ravel()
    r = np.abs(np.asarray(error, float).ravel())
    ok = np.isfinite(e) & np.isfinite(r)
    e, r = e[ok], r[ok]
    n = len(e)
    rmse_all = float(np.sqrt(np.mean(r ** 2))) if n else float("nan")
    if n < max(4, min_n):
        return PredictorScore(name=name, n=n, spearman=float("nan"), kendall=float("nan"),
                              aurc=float("nan"), aurc_oracle=float("nan"),
                              aurc_random=float("nan"), efficiency=float("nan"),
                              rmse_trusted=float("nan"), rmse_doubted=float("nan"))
    spread = float(np.max(e) - np.min(e))
    if spread <= 1e-12 * max(1.0, abs(float(np.mean(e)))):
        # an estimate that is the same everywhere cannot order anything: say so rather than
        # scoring the arbitrary order the sort happens to produce
        return PredictorScore(name=name, n=n, spearman=float("nan"), kendall=float("nan"),
                              aurc=rmse_all, aurc_oracle=aurc(r, r, min_n),
                              aurc_random=rmse_all, efficiency=0.0,
                              rmse_trusted=rmse_all, rmse_doubted=rmse_all, constant=True)

    a = aurc(e, r, min_n)
    a_oracle = aurc(r, r, min_n)
    a_random = rmse_all                      # the flat curve: the risk of ignoring the estimate
    denom = a_random - a_oracle
    eff = float((a_random - a) / denom) if abs(denom) > 1e-12 else float("nan")

    half = n // 2
    order = np.argsort(e, kind="stable")
    trusted, doubted = r[order[:half]], r[order[n - half:]]
    return PredictorScore(name=name, n=n,
                          spearman=float(spearmanr(e, r).statistic),
                          kendall=float(kendalltau(e, r).statistic),
                          aurc=a, aurc_oracle=a_oracle, aurc_random=a_random, efficiency=eff,
                          rmse_trusted=float(np.sqrt(np.mean(trusted ** 2))),
                          rmse_doubted=float(np.sqrt(np.mean(doubted ** 2))))


def compare(estimates: Mapping[str, Sequence[float]], y_true: Sequence[float],
            y_pred: Sequence[float], min_n: int = MIN_COVERAGE) -> List[PredictorScore]:
    """Score every estimate against the same errors, best efficiency first.

    `estimates` maps a name to one number per molecule, oriented so that **larger means a larger
    error is expected**: a variance, a distance from the training set, an interval width.  An
    estimate that is oriented the other way (a similarity) must be negated by the caller, which is
    the one convention this module insists on.
    """
    err = np.asarray(y_true, float) - np.asarray(y_pred, float)
    out = [score_predictor(name, v, err, min_n=min_n) for name, v in estimates.items()]
    return sorted(out, key=lambda s: (-(s.efficiency if np.isfinite(s.efficiency) else -np.inf),
                                      s.name))


def table(scores: Sequence[PredictorScore]) -> str:
    """The comparison as a text table, for the command line and the report."""
    head = (f"  {'estimate':24s} {'rho':>7s} {'tau':>7s} {'efficiency':>11s} "
            f"{'RMSE trusted':>13s} {'RMSE doubted':>13s}")
    lines = [head, "  " + "-" * (len(head) - 2)]
    for s in scores:
        if s.constant:
            lines.append(f"  {s.name:24s} {'--':>7s} {'--':>7s} {'constant':>11s} "
                         f"{s.rmse_trusted:13.3f} {s.rmse_doubted:13.3f}")
            continue
        lines.append(f"  {s.name:24s} {s.spearman:+7.3f} {s.kendall:+7.3f} {s.efficiency:+11.3f} "
                     f"{s.rmse_trusted:13.3f} {s.rmse_doubted:13.3f}")
    if scores:
        s = scores[0]
        lines.append(f"  the oracle, which knows the errors: AURC {s.aurc_oracle:.3f}; "
                     f"ignoring the estimate: {s.aurc_random:.3f}")
    return "\n".join(lines)


def estimates_for(model, test, y_pred: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
    """Every error estimate a fitted model can offer about `test`, by the module's convention.

    Whatever the model cannot provide is left out rather than faked: the posterior variance needs
    the linear kernel, the domain needs the field kernels, and the conformal width needs a
    calibration (``--conformal``).  The returned names are the ones the manual's table uses.
    """
    from .validation import domain

    out: Dict[str, np.ndarray] = {}
    if model.is_linear:
        try:
            _yp, var = model.predict_with_variance(test)
            out["GP variance"] = np.asarray(var, float)
        except (ValueError, RuntimeError):
            pass
    if model.uses_fields:
        try:
            K_xt = model.test_kernel(test)
            dom = domain(K_xt, model.joint_self(test), model.training_kernel(model.train))
            out["distance to training set"] = np.asarray(dom.distance, float)
            # the domain's own statistic is a similarity, so it is oriented the wrong way: negate it
            out["domain dissimilarity"] = 1.0 - np.asarray(dom.mean_top, float)
        except (ValueError, RuntimeError):
            pass
    if getattr(model, "conformal_", None) is not None:
        try:
            _yp, lo, hi = model.predict_interval(test)
            out["conformal width"] = np.asarray(hi, float) - np.asarray(lo, float)
        except (ValueError, RuntimeError):
            pass
    return out
