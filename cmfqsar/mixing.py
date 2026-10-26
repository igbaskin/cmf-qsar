"""A posterior over the field weights (plan item E6).

The joint kernel is a sum, ``K = sum_f h_f c_f K_f``, so choosing the mixing coefficients h_f is
the multiple-kernel-learning problem, and :func:`cmfqsar.model.align_mixing` already solves it by
kernel alignment: one number per field, no statement of how sure it is.  Under the Gaussian
process the same sum has an interpretation that does better.  A sum of kernels is the kernel of a
sum of independent processes,

    f(x) = sum_f f_f(x),    f_f ~ GP(0, h_f c_f K_f),

so h_f is the prior variance of the part of the model carried by field f -- how much of the
activity that field is allowed to explain.  Putting a prior on it and integrating gives a
posterior over the weights, and with it an answer to *does this series care about the steric field
at all* that has an interval on it rather than a point.

The parametrisation is the one the question asks for.  What matters is each field's **share** of
the kernel, not the overall size, which the evidence profiles away anyway: the weights are
written ``h = F softmax(u)``, so the shares ``softmax(u)`` sum to one and can be compared with
1/F, the share a field would have if every field mattered equally.  The prior is
``u_f ~ N(0, sigma^2)`` independently, which is exchangeable over the fields -- it prefers no
field over any other -- and weakly informative: sigma = 1.5 keeps a field's share within about a
factor of twenty of the others before the data are taken into account.  Adding a constant to every
u_f leaves the shares alone, so the likelihood is flat along that direction and the prior settles
it at the centred representative, which is to say the prior acts on the *spread* of the log-shares
and not on their level.  The noise level is sampled with them, as ``log lambda``, because a model
that is allowed to reweigh its fields and not to adjust its regularisation is answering a different
question.

Three things come out, in increasing order of what they cost:

* :func:`evidence_mixing` -- the weights that maximise the posterior.  A point estimate like the
  alignment one, and a direct replacement for it (``--mkl evidence``).
* :func:`sample_mixing` -- the posterior itself, by slice sampling.  Slice sampling because it
  has no step size to tune: a library that asks its user to tune a sampler will be run with the
  default and the default will be wrong for their data.
* :func:`drop_one` -- what the evidence loses when a field is removed outright, which is the
  Bayes factor against *this field does nothing*.  The sharpest form of the question, and the
  cheapest to answer.

**What this does not promise.**  The evidence is not a held-out score, and item D6 measured what
happens when it is given more freedom over the shape of the kernel: it takes the freedom, its own
number improves, and prediction does not move.  The same caution applies here and
``benchmarks/run_mixing.py`` measures it.  What a posterior over the weights is *for* is the
interpretation question -- which fields this series needs -- and that question has no
cross-validated answer to be checked against.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .evidence import log_evidence

#: standard deviation of the prior on the log-shares; see the module docstring
PRIOR_SD = 1.5
#: standard deviation of the prior on log lambda, around the uniform-weight optimum
LAM_PRIOR_SD = 2.0
#: the lambdas tried when a starting point is needed
LAM_GRID = (1e-3, 1e-2, 0.1, 1.0, 10.0)


def _softmax(u: np.ndarray) -> np.ndarray:
    v = np.asarray(u, float) - np.max(u)
    e = np.exp(v)
    return e / e.sum()


def shares_to_h(p: Sequence[float], n_fields: int) -> np.ndarray:
    """Shares that sum to one, written as weights that average one."""
    return np.asarray(p, float) * float(n_fields)


@dataclass
class MixingPosterior:
    """The posterior over the share of the kernel each field carries."""

    names: List[str]
    samples: np.ndarray             # (S, F) shares, one row per draw
    lam: np.ndarray                 # (S,) the regularisation drawn with them
    map_shares: np.ndarray          # (F,) the posterior mode
    map_lam: float
    log_post: np.ndarray            # (S,) log posterior of every draw
    accepted: int = 0
    note: str = ""

    @property
    def mean(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def interval(self, level: float = 0.9) -> np.ndarray:
        """(F, 2) equal-tailed credible interval of every field's share."""
        lo = 100.0 * (1.0 - level) / 2.0
        return np.percentile(self.samples, [lo, 100.0 - lo], axis=0).T

    def carries_less_than_its_share(self, level: float = 0.9) -> List[str]:
        """Fields whose whole interval lies below 1/F: the series needs them least."""
        hi = self.interval(level)[:, 1]
        return [n for n, h in zip(self.names, hi) if h < 1.0 / len(self.names)]

    def describe(self, level: float = 0.9) -> str:
        iv = self.interval(level)
        rows = [f"  {'field':18s} {'share':>8s} {'  ' + str(int(100 * level)) + ' % interval':>20s}"
                f" {'MAP':>8s}"]
        for f, name in enumerate(self.names):
            rows.append(f"  {name:18s} {self.mean[f]:8.3f}   [{iv[f, 0]:6.3f}, {iv[f, 1]:6.3f}]"
                        f" {self.map_shares[f]:8.3f}")
        rows.append(f"  equal shares would be {1.0 / len(self.names):.3f}; "
                    f"lambda {np.median(self.lam):.3g} "
                    f"[{np.percentile(self.lam, 5):.3g}, {np.percentile(self.lam, 95):.3g}]")
        return "\n".join(rows)


def _log_posterior(u: np.ndarray, log_lam: float, K_fields: np.ndarray, y: np.ndarray,
                   scale: np.ndarray, log_lam0: float, prior_sd: float, fit_bias: bool) -> float:
    """Log evidence of the weights plus their prior; -inf where the kernel is not valid."""
    p = _softmax(u)
    h = p * len(p)
    K = np.tensordot(h * scale, K_fields, axes=(0, 0))
    ev = log_evidence(K, y, float(np.exp(log_lam)), fit_bias=fit_bias)
    if not np.isfinite(ev):
        return -np.inf
    lp = -0.5 * float(np.sum((np.asarray(u, float) / prior_sd) ** 2))
    lp -= 0.5 * ((log_lam - log_lam0) / LAM_PRIOR_SD) ** 2
    return float(ev + lp)


def _start(K_fields: np.ndarray, y: np.ndarray, scale: np.ndarray, fit_bias: bool) -> float:
    """log lambda at the evidence optimum with equal weights: where the prior is centred."""
    F = K_fields.shape[0]
    K = np.tensordot(np.ones(F) * scale, K_fields, axes=(0, 0))
    best = None
    for lam in LAM_GRID:
        e = log_evidence(K, y, float(lam), fit_bias=fit_bias)
        if np.isfinite(e) and (best is None or e > best[0]):
            best = (e, float(lam))
    return float(np.log(best[1])) if best else 0.0


def _prepare(K_fields, y, scale, enabled):
    K_fields = np.asarray(K_fields, float)
    F = K_fields.shape[0]
    scale = np.ones(F) if scale is None else np.asarray(scale, float)
    on = np.ones(F, bool) if enabled is None else np.asarray(enabled, bool)
    idx = np.flatnonzero(on)
    return K_fields, np.asarray(y, float), scale, on, idx


def evidence_mixing(K_fields: np.ndarray, y: np.ndarray, scale: Optional[np.ndarray] = None,
                    enabled: Optional[Sequence[bool]] = None, fit_bias: bool = True,
                    prior_sd: float = PRIOR_SD, maxiter: int = 400) -> np.ndarray:
    """The mixing coefficients that maximise the evidence (with the prior above), on the simplex.

    Returned in the same convention as :func:`cmfqsar.model.align_mixing`: one number per field,
    zero where the field is disabled, scaled so that the joint kernel has the mean diagonal it
    would have with h = 1, which keeps the lambda grid comparable.
    """
    from scipy.optimize import minimize

    K_fields, y, scale, on, idx = _prepare(K_fields, y, scale, enabled)
    F = K_fields.shape[0]
    if idx.size == 0:
        return np.zeros(F)
    if idx.size == 1:
        h = np.zeros(F)
        h[idx[0]] = 1.0
        return h
    sub = K_fields[idx]
    ssub = scale[idx]
    log_lam0 = _start(sub, y, ssub, fit_bias)

    def neg(theta):
        return -_log_posterior(theta[:-1], float(theta[-1]), sub, y, ssub, log_lam0, prior_sd,
                               fit_bias)

    x0 = np.concatenate([np.zeros(idx.size), [log_lam0]])
    # the simplex is given explicitly: Nelder-Mead builds its own by perturbing each coordinate
    # by a fraction of its value, and every coordinate here starts at zero, so its own simplex
    # would be 2.5e-4 wide and it would report convergence on the first iteration
    step = np.full(idx.size + 1, 0.6)
    simplex = np.vstack([x0] + [x0 + np.eye(idx.size + 1)[i] * step[i]
                                for i in range(idx.size + 1)])
    res = minimize(neg, x0, method="Nelder-Mead",
                   options={"maxiter": int(maxiter) * (idx.size + 1), "xatol": 1e-3,
                            "fatol": 1e-3, "initial_simplex": simplex})
    p = _softmax(res.x[:-1]) if np.isfinite(res.fun) else np.full(idx.size, 1.0 / idx.size)
    h = np.zeros(F)
    h[idx] = p * idx.size
    return _rescale(h, K_fields, scale, idx)


def _rescale(h: np.ndarray, K_fields: np.ndarray, scale: np.ndarray,
             idx: np.ndarray) -> np.ndarray:
    """Put the joint kernel's mean diagonal back where h = 1 would have it."""
    ref = float(sum(np.mean(np.diag(K_fields[f])) * scale[f] for f in idx))
    cur = float(sum(h[f] * np.mean(np.diag(K_fields[f])) * scale[f] for f in idx))
    if cur > 1e-300 and ref > 1e-300:
        h = h * (ref / cur)
    return h


def _slice_sample(logp, x: np.ndarray, width: float, rng: np.random.Generator,
                  max_steps: int = 8) -> Tuple[np.ndarray, float, int]:
    """One sweep of univariate slice sampling over every coordinate of x (Neal 2003)."""
    x = np.array(x, float)
    cur = logp(x)
    evals = 1
    for i in range(len(x)):
        level = cur + np.log(rng.random())
        lo = x[i] - width * rng.random()
        hi = lo + width
        for _ in range(max_steps):                      # step out
            trial = np.array(x)
            trial[i] = lo
            if logp(trial) <= level:
                break
            evals += 1
            lo -= width
        for _ in range(max_steps):
            trial = np.array(x)
            trial[i] = hi
            if logp(trial) <= level:
                break
            evals += 1
            hi += width
        for _ in range(64):                             # shrink
            cand = np.array(x)
            cand[i] = lo + (hi - lo) * rng.random()
            val = logp(cand)
            evals += 1
            if val > level:
                x, cur = cand, val
                break
            if cand[i] < x[i]:
                lo = cand[i]
            else:
                hi = cand[i]
    return x, cur, evals


def sample_mixing(K_fields: np.ndarray, y: np.ndarray, names: Optional[Sequence[str]] = None,
                  scale: Optional[np.ndarray] = None, enabled: Optional[Sequence[bool]] = None,
                  draws: int = 400, burn: int = 100, thin: int = 1, seed: int = 0,
                  fit_bias: bool = True, prior_sd: float = PRIOR_SD,
                  progress=None) -> MixingPosterior:
    """The posterior over the fields' shares, by slice sampling the log-shares and log lambda."""
    K_fields, y, scale, on, idx = _prepare(K_fields, y, scale, enabled)
    F = K_fields.shape[0]
    names = list(names) if names is not None else [f"field {f + 1}" for f in range(F)]
    if idx.size < 2:
        p = np.zeros(F)
        p[idx] = 1.0
        return MixingPosterior(names=names, samples=p[None, :], lam=np.array([1.0]),
                               map_shares=p, map_lam=1.0, log_post=np.array([np.nan]),
                               note="one field: there is nothing to share")
    sub = K_fields[idx]
    ssub = scale[idx]
    log_lam0 = _start(sub, y, ssub, fit_bias)

    def logp(theta):
        return _log_posterior(theta[:-1], float(theta[-1]), sub, y, ssub, log_lam0, prior_sd,
                              fit_bias)

    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(idx.size), [log_lam0]])
    chain, lams, posts = [], [], []
    best = (-np.inf, x)
    total = int(burn) + int(draws) * int(thin)
    for step in range(total):
        x, cur, _ev = _slice_sample(logp, x, width=1.0, rng=rng)
        if cur > best[0]:
            best = (cur, np.array(x))
        if step >= burn and (step - burn) % max(1, int(thin)) == 0:
            p = np.zeros(F)
            p[idx] = _softmax(x[:-1])
            chain.append(p)
            lams.append(float(np.exp(x[-1])))
            posts.append(cur)
        if progress is not None and step % 25 == 0:
            progress(f"mixing posterior: draw {step}/{total}", step / total)
    pmap = np.zeros(F)
    pmap[idx] = _softmax(best[1][:-1])
    return MixingPosterior(names=names, samples=np.asarray(chain, float),
                           lam=np.asarray(lams, float), map_shares=pmap,
                           map_lam=float(np.exp(best[1][-1])),
                           log_post=np.asarray(posts, float), accepted=len(chain))


def drop_one(K_fields: np.ndarray, y: np.ndarray, names: Optional[Sequence[str]] = None,
             scale: Optional[np.ndarray] = None, enabled: Optional[Sequence[bool]] = None,
             fit_bias: bool = True, prior_sd: float = PRIOR_SD) -> Dict[str, float]:
    """How much log evidence the model loses when each field is taken away entirely.

    The Bayes factor against *this field does nothing*: a large positive number means the
    activities are much more probable with the field than without it, and a number near zero --
    or below it -- means the field is carrying nothing the others do not already carry. This is
    the sharpest form of the question and the cheapest to answer, since it needs one optimisation
    per field rather than a posterior.
    """
    K_fields, y, scale, on, idx = _prepare(K_fields, y, scale, enabled)
    names = list(names) if names is not None else [f"field {f + 1}" for f in range(K_fields.shape[0])]
    full = _best_evidence(K_fields, y, scale, on, fit_bias, prior_sd)
    out: Dict[str, float] = {}
    for f in idx:
        keep = np.array(on)
        keep[f] = False
        if not keep.any():
            out[names[f]] = float("nan")
            continue
        out[names[f]] = float(full - _best_evidence(K_fields, y, scale, keep, fit_bias, prior_sd))
    return out


def _best_evidence(K_fields: np.ndarray, y: np.ndarray, scale: np.ndarray, on: np.ndarray,
                   fit_bias: bool, prior_sd: float) -> float:
    """The evidence at the best weights for the fields that are switched on."""
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return -np.inf
    h = evidence_mixing(K_fields, y, scale, on, fit_bias=fit_bias, prior_sd=prior_sd)
    K = np.tensordot(h * scale, K_fields, axes=(0, 0))
    best = -np.inf
    for lam in LAM_GRID:
        e = log_evidence(K, y, float(lam), fit_bias=fit_bias)
        if np.isfinite(e):
            best = max(best, e)
    return best
