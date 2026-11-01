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

Pooling the weights across series
---------------------------------

A single series of thirty molecules cannot say much about five weights, which is why the
intervals above are wide and why a point estimate fitted on one series is noisy.  Other series
can help.  The prior above is centred at equal shares with one fixed spread; the pooled form
lets other series set both:

    u_s ~ N(mu, tau^2 I)    for every series s,

with `u_s` the centred log-shares of series s, `mu` the log-shares a typical series needs and
`tau` how much series differ from each other.  `mu` and `tau` are chosen by empirical Bayes,
that is, to maximise the probability of all the series together (:func:`pool_priors`).  The
posterior draws of each series under the fixed prior are all that is needed: the probability of
a series under the pooled prior is the average over its draws of the ratio of the two priors
(importance reweighting), so no new sampling is done.  A series then gets its weights from
:func:`evidence_mixing` with ``prior_mean=mu`` and ``prior_sd=tau``: a small series is pulled
towards what the others need, a large series keeps its own weights, and `tau` decides where the
balance lies.  With `tau` near zero the weights are the population's and nothing is fitted on
the series; with `tau` large the pooled prior is the fixed one again.
``benchmarks/run_pooled_mixing.py`` measures whether this predicts better, with every series'
prior learned from the other series only.
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
                   scale: np.ndarray, log_lam0: float, prior_sd: float, fit_bias: bool,
                   prior_mean: Optional[np.ndarray] = None) -> float:
    """Log evidence of the weights plus their prior; -inf where the kernel is not valid.

    The prior on the log-shares is ``N(prior_mean, prior_sd^2 I)``; a mean of None is zero,
    which is equal shares.
    """
    p = _softmax(u)
    h = p * len(p)
    K = np.tensordot(h * scale, K_fields, axes=(0, 0))
    ev = log_evidence(K, y, float(np.exp(log_lam)), fit_bias=fit_bias)
    if not np.isfinite(ev):
        return -np.inf
    d = np.asarray(u, float)
    if prior_mean is not None:
        d = d - np.asarray(prior_mean, float)
    lp = -0.5 * float(np.sum((d / prior_sd) ** 2))
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


def _prior_mean_for(prior_mean, idx: np.ndarray) -> Optional[np.ndarray]:
    """The prior mean over the enabled fields, centred, since only the spread of the
    log-shares is meaningful."""
    if prior_mean is None:
        return None
    m = np.asarray(prior_mean, float)[idx]
    return m - m.mean()


def centred_log_shares(shares: np.ndarray) -> np.ndarray:
    """Shares (rows summing to one) as centred log-shares, the coordinates the prior acts on."""
    u = np.log(np.clip(np.asarray(shares, float), 1e-12, None))
    return u - u.mean(axis=-1, keepdims=True)


def evidence_mixing(K_fields: np.ndarray, y: np.ndarray, scale: Optional[np.ndarray] = None,
                    enabled: Optional[Sequence[bool]] = None, fit_bias: bool = True,
                    prior_sd: float = PRIOR_SD, maxiter: int = 400,
                    prior_mean: Optional[Sequence[float]] = None) -> np.ndarray:
    """The mixing coefficients that maximise the evidence (with the prior above), on the simplex.

    Returned in the same convention as :func:`cmfqsar.model.align_mixing`: one number per field,
    zero where the field is disabled, scaled so that the joint kernel has the mean diagonal it
    would have with h = 1, which keeps the lambda grid comparable.

    ``prior_mean`` (centred log-shares, one per field) and ``prior_sd`` together are the pooled
    prior of :func:`pool_priors`; by default the prior is centred at equal shares.
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
    mean = _prior_mean_for(prior_mean, idx)

    def neg(theta):
        return -_log_posterior(theta[:-1], float(theta[-1]), sub, y, ssub, log_lam0, prior_sd,
                               fit_bias, mean)

    x0 = np.concatenate([np.zeros(idx.size) if mean is None else mean, [log_lam0]])
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
                  progress=None, prior_mean: Optional[Sequence[float]] = None) -> MixingPosterior:
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
    mean = _prior_mean_for(prior_mean, idx)

    def logp(theta):
        return _log_posterior(theta[:-1], float(theta[-1]), sub, y, ssub, log_lam0, prior_sd,
                              fit_bias, mean)

    rng = np.random.default_rng(seed)
    x = np.concatenate([np.zeros(idx.size) if mean is None else mean, [log_lam0]])
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


@dataclass
class PooledPrior:
    """The prior over the log-shares that a collection of series supports."""

    mean: np.ndarray            # (F,) centred log-shares of a typical series
    sd: float                   # how far one series' log-shares sit from the mean, as a rule
    names: List[str]
    n_series: int
    ess: np.ndarray             # (n_series,) effective number of draws behind each series' term
    log_marginal: float         # the pooled log probability at the optimum, up to a constant
    note: str = ""
    method: str = "gaussian"

    @property
    def shares(self) -> np.ndarray:
        """The shares a typical series gives the fields."""
        return _softmax(self.mean)

    def h(self) -> np.ndarray:
        """The population weights in the convention of :func:`evidence_mixing` (mean one)."""
        return shares_to_h(self.shares, len(self.mean))

    def describe(self) -> str:
        rows = [f"  pooled over {self.n_series} series; spread of the log-shares between series "
                f"tau = {self.sd:.3f} (the fixed prior has {PRIOR_SD})",
                f"  {'field':18s} {'typical share':>14s}"]
        for name, s in zip(self.names, self.shares):
            rows.append(f"  {name:18s} {s:14.3f}")
        rows.append(f"  equal shares would be {1.0 / len(self.mean):.3f}"
                    + (f"; smallest effective sample size {float(np.min(self.ess)):.0f} draws"
                       if self.method == "importance" else ""))
        if self.note:
            rows.append("  " + self.note)
        return "\n".join(rows)


#: the spread of the pooled prior is kept in this range: below it the prior pins the weights
#: of every series to the population's, above it the prior is wider than the fixed one by a
#: factor of two and nothing is being pooled
POOLED_SD_RANGE = (0.05, 2.0 * PRIOR_SD)
#: the smallest effective number of draws a series may contribute with under
#: ``method="importance"``; see :func:`pool_priors`
MIN_ESS = 25.0


def _centred_basis(F: int) -> np.ndarray:
    """An orthonormal basis (F, F - 1) of the vectors whose entries sum to zero."""
    q, _ = np.linalg.qr(np.eye(F) - np.full((F, F), 1.0 / F))
    # the last column of q spans the constant direction up to round-off; the first F - 1 span
    # the centred subspace, which is what is needed
    return q[:, :F - 1]


def _gaussian_summaries(zs: Sequence[np.ndarray], prior_sd: float):
    """For every series, the Gaussian picture of its likelihood in the centred coordinates.

    The posterior draws give a mean `m` and a covariance `S`; the likelihood then has precision
    ``Lambda = S^-1 - I / prior_sd^2`` (the prior's precision removed) and ``Lambda c = S^-1 m``
    for its centre `c`.  Lambda is returned by its eigenvalues `w` (negative ones, which are
    round-off or a posterior wider than the prior, set to zero) and eigenvectors `V`, with the
    components of ``Lambda c`` in that basis.
    """
    out = []
    d = zs[0].shape[1]
    for z in zs:
        m = z.mean(axis=0)
        S = np.cov(z, rowvar=False) + 1e-9 * np.eye(d)
        P = np.linalg.inv(S)
        w, V = np.linalg.eigh(P - np.eye(d) / prior_sd ** 2)
        g = V.T @ (P @ m)
        # a direction whose likelihood is wider than twice the prior is one the series says
        # nothing about; the precision left there is sampling noise of the covariance
        keep = w > 0.25 / prior_sd ** 2
        w = np.where(keep, w, 0.0)
        g = np.where(keep, g, 0.0)
        out.append((w, V, g))
    return out


def pool_priors(shares_by_series: Sequence[np.ndarray], names: Optional[Sequence[str]] = None,
                prior_sd: float = PRIOR_SD, sd_range: Tuple[float, float] = POOLED_SD_RANGE,
                method: str = "gaussian", min_ess: float = MIN_ESS,
                maxiter: int = 2000) -> PooledPrior:
    """The population prior ``N(mu, tau^2 I)`` over the centred log-shares, by empirical Bayes.

    ``shares_by_series`` holds, for every series, the posterior draws of its shares under the
    fixed prior (``MixingPosterior.samples``), all with the same fields.  `mu` and `tau` are
    chosen to maximise the probability of all the series together, each series' probability
    being its likelihood integrated against the candidate prior.  Two series are enough to run
    it; the result is only as good as the number of them.

    Two ways to get at a series' likelihood from its draws:

    * ``method="gaussian"`` (the default) summarises the draws by their mean and covariance,
      removes the fixed prior's precision, and treats what is left as a Gaussian likelihood.
      The integral against the candidate prior is then in closed form.  This is the
      random-effects model of meta-analysis, with each series' weights as the estimate and its
      posterior covariance as the uncertainty.  A direction in which the series says nothing
      has zero precision and contributes nothing, as it should.
    * ``method="importance"`` takes the log of the average, over the draws, of the ratio of the
      candidate prior to the fixed one (importance reweighting), and needs no Gaussian
      assumption.  A narrow candidate prior gives almost all the weight to the few draws nearest
      its mean, and the optimiser, left alone, finds such a prior and reports a spread far too
      small.  So every series must keep an effective sample size (the number of equally
      weighted draws its average is worth) of at least ``min_ess``.  With a few hundred draws
      this stops the spread at around a fifth of ``prior_sd``, and the result is noisier than
      the Gaussian one; it is kept as the check that the Gaussian assumption is not doing the
      work.

    The draws must come from the prior of width ``prior_sd`` centred at equal shares, which is
    what :func:`sample_mixing` uses by default.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp

    us = [centred_log_shares(np.atleast_2d(s)) for s in shares_by_series]
    if len(us) < 2:
        raise ValueError("pooling needs at least two series")
    F = us[0].shape[1]
    if any(u.shape[1] != F for u in us):
        raise ValueError("every series must have the same fields")
    if method not in ("gaussian", "importance"):
        raise ValueError("method must be 'gaussian' or 'importance'")
    names = list(names) if names is not None else [f"field {f + 1}" for f in range(F)]
    dim = F - 1                                 # the centred log-shares live in F - 1 dimensions
    B = _centred_basis(F)
    zs = [u @ B for u in us]
    draws = np.array([len(u) for u in us], float)
    lo, hi = float(sd_range[0]), float(sd_range[1])

    if method == "gaussian":
        summaries = _gaussian_summaries(zs, prior_sd)

        def terms(mu, tau):
            mz = B.T @ mu
            out = []
            for w, V, g in summaries:
                mk = V.T @ mz
                a = 1.0 + tau ** 2 * w
                on = w > 0
                q = np.sum((g[on] ** 2 / w[on] - 2.0 * g[on] * mk[on] + w[on] * mk[on] ** 2)
                           / a[on])
                out.append(-0.5 * float(np.sum(np.log(a))) - 0.5 * float(q))
            return np.asarray(out), draws
    else:
        base = [0.5 * np.sum(z ** 2, axis=1) / prior_sd ** 2 for z in zs]

        def terms(mu, tau):
            """log mean importance ratio per series, and the effective sample sizes."""
            mz = B.T @ mu
            out, ess = [], []
            for z, b in zip(zs, base):
                lw = (dim * np.log(prior_sd / tau)
                      - 0.5 * np.sum((z - mz) ** 2, axis=1) / tau ** 2 + b)
                m = logsumexp(lw)
                out.append(float(m - np.log(len(lw))))
                ess.append(float(np.exp(2.0 * m - logsumexp(2.0 * lw))))
            return np.asarray(out), np.asarray(ess)

    def unpack(theta):
        mu = np.asarray(theta[:-1], float)
        mu = mu - mu.mean()
        # a logistic map keeps tau inside the range without a constrained optimiser
        tau = lo + (hi - lo) / (1.0 + np.exp(-float(theta[-1])))
        return mu, tau

    def neg(theta):
        mu, tau = unpack(theta)
        lm, ess = terms(mu, tau)
        val = -float(np.sum(lm))
        if method == "importance":
            short = float(min_ess - np.min(ess))
            # a steep but continuous penalty keeps the search in the region where the averages
            # are worth something
            if short > 0:
                val += 50.0 * short
        return val

    means = np.stack([u.mean(axis=0) for u in us])
    mu0 = means.mean(axis=0)
    tau0 = float(np.clip(np.sqrt(np.mean((means - mu0) ** 2)), lo * 1.5, hi / 1.5))
    t0 = np.log((tau0 - lo) / (hi - tau0))
    x0 = np.concatenate([mu0, [t0]])
    step = np.full(F + 1, 0.5)
    simplex = np.vstack([x0] + [x0 + np.eye(F + 1)[i] * step[i] for i in range(F + 1)])
    res = minimize(neg, x0, method="Nelder-Mead",
                   options={"maxiter": int(maxiter), "xatol": 1e-4, "fatol": 1e-6,
                            "initial_simplex": simplex})
    mu, tau = unpack(res.x)
    lm, ess = terms(mu, tau)
    note = ""
    if method == "gaussian" and all(not np.any(w > 0) for w, _, _ in summaries):
        note = "no series carries information about the weights: the prior is arbitrary"
        mu, tau = np.zeros(F), prior_sd
    elif tau <= lo * 1.05:
        note = "tau is at the lower limit: the series agree and the weights are the population's"
    elif tau >= hi / 1.05:
        note = "tau is at the upper limit: the series do not agree and nothing is pooled"
    elif method == "importance" and float(np.min(ess)) <= 1.2 * min_ess:
        note = (f"tau is held up by the effective sample size ({float(np.min(ess)):.0f} draws): "
                "more draws per series could allow a narrower prior")
    return PooledPrior(mean=mu, sd=float(tau), names=names, n_series=len(us), ess=ess,
                       log_marginal=float(np.sum(lm)), note=note, method=method)
