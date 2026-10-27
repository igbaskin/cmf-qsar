"""Which molecule to make next (plan item E2).

A fitted CMF model predicts an activity and, through the Gaussian-process view, a posterior
standard deviation beside it.  Those two numbers are enough to answer a question the 2019 paper
poses and leaves as a statement -- *maximising the overlap tells how to modify the molecule* --
as an operation with a number: given a set of posed candidates, which one should be made?

The answer depends on what the asker is doing, and the module keeps the two apart rather than
pretending one rule serves both.

* **To find an active compound** the acquisition is optimistic.  Expected improvement asks how
  much a candidate is expected to beat the best thing already made, counting only the gain::

      EI(x) = (mu - y* - xi) Phi(z) + sigma phi(z),   z = (mu - y* - xi) / sigma

  and the upper confidence bound ``mu + beta sigma`` is the cruder form of the same idea.  A
  molecule the model is unsure about is *promoted*, because being wrong about it is cheap and
  being right is valuable.

* **To improve the model** the acquisition ignores the prediction entirely and takes the largest
  posterior standard deviation, or -- with no model at all -- the candidate furthest in field
  space from everything already made (``maxmin``, the classical furthest-first design).

Note that this is the opposite of :meth:`CMFModel.rank_score`, which ranks a *screening* library
by ``mu - beta sigma``: there a molecule the model knows nothing about must be pushed down,
because a kernel method hands back the training mean presented as a prediction and the library
is full of such molecules.  Screening is a decision whose mistakes have a cost; acquisition is a
question that has to be answered.  Optimism is right for one and wrong for the other.

:func:`learning_curve` runs the loop retrospectively on a kernel whose activities are already
known, which is how ``benchmarks/run_active.py`` measures whether any of this beats picking at
random.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: the rules this module knows, in the order the manual introduces them
ACQUISITIONS = ("random", "greedy", "ucb", "ei", "variance", "maxmin")

#: what each one is for, printed by the command line
PURPOSE = {
    "random": "pick at random; the baseline every other rule has to beat",
    "greedy": "the highest prediction; exploitation with no allowance for being wrong",
    "ucb": "prediction plus beta standard deviations; optimism in the face of uncertainty",
    "ei": "expected improvement over the best already made; the usual choice for finding actives",
    "variance": "the largest posterior standard deviation; what most improves the model",
    "maxmin": "furthest in field space from everything already made; design without a model",
}


@dataclass
class Suggestion:
    """One candidate, with the numbers that chose it."""

    index: int
    name: str
    mean: float
    sd: float
    score: float
    distance: float = float("nan")      # to the nearest molecule already made, in field space

    def describe(self, kind: str) -> str:
        return (f"{self.name[:34]:34s} predicted {self.mean:7.3f} +- {self.sd:5.3f}   "
                f"{kind} {self.score:+8.4f}"
                + ("" if not np.isfinite(self.distance) else f"   nearest {self.distance:5.3f}"))


def expected_improvement(mean: Sequence[float], sd: Sequence[float], best: float,
                         xi: float = 0.0) -> np.ndarray:
    """E[max(y - best - xi, 0)] under the posterior, in closed form.

    Zero standard deviation gives the improvement itself when there is one and zero otherwise,
    which is the limit of the expression and keeps a molecule the model is certain about from
    being discarded for its certainty.
    """
    from scipy.stats import norm

    mu = np.asarray(mean, float)
    s = np.asarray(sd, float)
    gap = mu - float(best) - float(xi)
    out = np.maximum(gap, 0.0)
    ok = s > 1e-12
    z = np.zeros_like(mu)
    z[ok] = gap[ok] / s[ok]
    out[ok] = gap[ok] * norm.cdf(z[ok]) + s[ok] * norm.pdf(z[ok])
    return out


def acquisition_scores(kind: str, mean: Sequence[float], sd: Sequence[float],
                       best: Optional[float] = None, beta: float = 1.0, xi: float = 0.0,
                       distance: Optional[Sequence[float]] = None,
                       rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """The score of every candidate under one rule; larger is always better.

    `distance` is the field-space distance of each candidate to the nearest molecule already
    made, which only ``maxmin`` uses; `best` is the best activity so far, which only ``ei`` uses.
    """
    mu = np.asarray(mean, float)
    s = np.asarray(sd, float)
    if kind == "random":
        rng = rng or np.random.default_rng()
        return rng.random(len(mu))
    if kind == "greedy":
        return mu
    if kind == "ucb":
        return mu + float(beta) * s
    if kind == "ei":
        if best is None:
            raise ValueError("expected improvement needs the best activity so far")
        return expected_improvement(mu, s, best, xi)
    if kind == "variance":
        return s
    if kind == "maxmin":
        if distance is None:
            raise ValueError("maxmin needs the distance to the molecules already made")
        return np.asarray(distance, float)
    raise ValueError(f"unknown acquisition {kind!r}; one of {', '.join(ACQUISITIONS)}")


def _distance_to_training(K_xt: np.ndarray, k_self: Sequence[float],
                          k_train_self: Sequence[float]) -> np.ndarray:
    """min_j |x - x_j| in field space, from the kernel: |x - x_j|^2 = K(x,x) + K(j,j) - 2K(x,j)."""
    d2 = (np.asarray(k_self, float)[:, None] + np.asarray(k_train_self, float)[None, :]
          - 2.0 * np.asarray(K_xt, float))
    return np.sqrt(np.maximum(d2.min(axis=1), 0.0))


def suggest(model, candidates, kind: str = "ei", n: int = 1, beta: float = 1.0, xi: float = 0.0,
            names: Optional[Sequence[str]] = None, max_similarity: float = 0.98,
            seed: Optional[int] = 0, batch_rule: str = "cosine") -> List[Suggestion]:
    """Rank posed candidates by one acquisition rule and return the best `n`.

    The model must be a regression on the linear kernel, since that is where a posterior exists.
    With `n` above one a batch is chosen, and `batch_rule` says how (plan item MC2):

    * ``"cosine"`` takes the best scores in order and skips a candidate too similar to one
      already picked (`max_similarity`, a cosine in the model's own kernel).  It is cheap and it
      has a threshold in it.
    * ``"covariance"`` recomputes the score after every pick from the posterior covariance
      conditioned on the picks already made, which needs no threshold and costs one Cholesky
      step per pick.  With ``kind="variance"`` it is the batch of largest determinant.

    Neither refits the model inside the batch, which is what a laboratory would have to pay for;
    `benchmarks/run_batch.py` measures what the two rules are worth against that.
    """
    if kind not in ACQUISITIONS:
        raise ValueError(f"unknown acquisition {kind!r}; one of {', '.join(ACQUISITIONS)}")
    if model.task != "regression":
        raise ValueError("acquisition is defined for a regression model: it compares predicted "
                         "activities with the best one made so far")
    mu, var = model.predict_with_variance(candidates)
    sd = np.sqrt(np.maximum(np.asarray(var, float), 0.0))
    K_xt = model.test_kernel(candidates)
    k_self = model.joint_self(candidates)
    k_train = np.diag(model.joint_kernel(model.K_fields_train, model.train.descriptors)) \
        if model.K_fields_train is not None else model.joint_self(model.train)
    dist = _distance_to_training(K_xt, k_self, k_train)
    best = float(np.max(np.asarray(model.y, float))) if model.y is not None else None
    score = acquisition_scores(kind, mu, sd, best=best, beta=beta, xi=xi, distance=dist,
                               rng=np.random.default_rng(seed))
    nm = list(names) if names is not None else list(getattr(candidates, "names", []))
    if len(nm) != len(mu):
        nm = [f"candidate {i + 1}" for i in range(len(mu))]

    if batch_rule not in BATCH_RULES:
        raise ValueError(f"unknown batch rule {batch_rule!r}; one of {', '.join(BATCH_RULES)}")
    if int(n) > 1 and batch_rule == "covariance":
        _mu, cov = model.predict_with_covariance(candidates)
        chosen = batch_by_covariance(kind, mu, cov, int(n), best=best, beta=beta, xi=xi,
                                     distance=dist, rng=np.random.default_rng(seed))
    else:
        # cosine similarity between the candidates themselves, so that a batch is not one
        # molecule n times over; computed only when a batch is asked for, since it is a block
        cos = None
        if int(n) > 1 and max_similarity < 1.0 and len(mu) > 1:
            dQ = model._self_kernels(candidates) if model.normalize else None
            Kcc = model.joint_kernel(model._field_kernels(candidates, candidates, dQ, dQ))
            d = np.sqrt(np.maximum(np.diag(Kcc), 1e-300))
            cos = Kcc / np.outer(d, d)
        chosen = batch_by_cosine(score, cos, int(n), max_similarity=max_similarity)
    return [Suggestion(index=int(i), name=nm[int(i)], mean=float(mu[int(i)]),
                       sd=float(sd[int(i)]), score=float(score[int(i)]),
                       distance=float(dist[int(i)])) for i in chosen]


#: the batch rules of plan item MC2
BATCH_RULES = ("cosine", "covariance")

#: what each one does, printed by the command line
BATCH_PURPOSE = {
    "cosine": "take the best scores in order, skipping one too similar to a pick already made",
    "covariance": "recompute the score after every pick from the posterior covariance of the "
                  "batch itself, which for the variance rule is the largest determinant",
}


def conditional_variances(cov: np.ndarray) -> np.ndarray:
    """The diagonal of a covariance matrix, which is where a greedy batch starts."""
    return np.maximum(np.diag(np.asarray(cov, float)), 0.0)


def condition_on(cov: np.ndarray, j: int, floor: float = 1e-12) -> np.ndarray:
    """The covariance of everything else once candidate `j` is known.

    One step of a Cholesky factorisation with `j` as the pivot::

        cov' = cov - cov[:, j] cov[j, :] / cov[j, j]

    The variance of `j` itself becomes zero, and the variance of anything correlated with it
    falls.  That is how a batch chosen this way stops being one molecule several times over:
    nothing has to be refitted, because the posterior covariance already knows what a second
    measurement near the first would add.
    """
    c = np.asarray(cov, float)
    pivot = float(c[int(j), int(j)])
    if pivot <= floor:
        out = c.copy()
        out[int(j), :] = 0.0
        out[:, int(j)] = 0.0
        return out
    col = c[:, int(j)]
    return c - np.outer(col, col) / pivot


def batch_by_covariance(kind: str, mean: Sequence[float], cov: np.ndarray, size: int,
                        best: Optional[float] = None, beta: float = 1.0, xi: float = 0.0,
                        distance: Optional[Sequence[float]] = None,
                        rng: Optional[np.random.Generator] = None) -> List[int]:
    """A batch of `size` candidates, the score recomputed after every pick (plan item MC2).

    The cheap rule of :func:`suggest` ranks once and drops a candidate that looks too much like
    one already picked, which is a filter with a threshold in it.  This rule has no threshold:
    after each pick the posterior covariance is conditioned on that pick
    (:func:`condition_on`), the standard deviations are taken from the conditioned covariance,
    and the acquisition is evaluated again.  A candidate whose uncertainty the first pick has
    already removed scores low by itself.

    With ``kind="variance"`` the score **is** the conditional standard deviation, so this is the
    greedy maximisation of the determinant of the batch's own covariance block -- the classical
    D-optimal design -- and ``log det`` of that block is the sum of the logs of the conditional
    variances the picks were chosen by.
    """
    mu = np.asarray(mean, float)
    c = np.asarray(cov, float).copy()
    taken: List[int] = []
    for _ in range(int(size)):
        sd = np.sqrt(conditional_variances(c))
        score = acquisition_scores(kind, mu, sd, best=best, beta=beta, xi=xi,
                                   distance=distance, rng=rng)
        score = np.asarray(score, float).copy()
        if taken:
            score[np.asarray(taken, int)] = -np.inf
        j = int(np.argmax(score))
        if not np.isfinite(score[j]):
            break
        taken.append(j)
        c = condition_on(c, j)
    return taken


def batch_by_cosine(score: Sequence[float], cosine: Optional[np.ndarray], size: int,
                    max_similarity: float = 0.98) -> List[int]:
    """A batch of `size` by score, skipping a candidate too similar to one already picked.

    This is the rule :func:`suggest` has always used, as a function of its own so that the two
    can be measured against each other.
    """
    s = np.asarray(score, float)
    taken: List[int] = []
    for i in np.argsort(-s):
        if len(taken) >= int(size):
            break
        i = int(i)
        if cosine is not None and any(cosine[i, j] > max_similarity for j in taken):
            continue
        taken.append(i)
    return taken


def log_determinant(cov: np.ndarray, rows: Sequence[int]) -> float:
    """log det of the covariance block of a batch, which is what a D-optimal rule maximises."""
    r = np.asarray(rows, int)
    if not len(r):
        return float("-inf")
    block = np.asarray(cov, float)[np.ix_(r, r)]
    sign, value = np.linalg.slogdet(block)
    return float(value) if sign > 0 else float("-inf")


# --------------------------------------------------------------------------- #
# The retrospective loop
# --------------------------------------------------------------------------- #

def _fit_predict(K: np.ndarray, y: np.ndarray, have: np.ndarray, pool: np.ndarray, lam: float,
                 fit_bias: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """(mean, sd) for the pool from a kernel ridge fit on what has been made so far."""
    from .model import krr_fit

    Ktr = K[np.ix_(have, have)]
    a, b = krr_fit(Ktr, y[have], lam, fit_bias)
    Kxt = K[np.ix_(pool, have)]
    mu = Kxt @ a + b
    n = len(have)
    A = Ktr + lam * np.eye(n)
    try:
        L = np.linalg.cholesky(A)
        v = np.linalg.solve(L, Kxt.T)
        var = np.diag(K)[pool] - np.sum(v * v, axis=0)
    except np.linalg.LinAlgError:
        var = np.diag(K)[pool] - np.einsum("ij,jk,ik->i", Kxt, np.linalg.pinv(A), Kxt)
    return mu, np.sqrt(np.maximum(var, 0.0))


def _fit_predict_cov(K: np.ndarray, y: np.ndarray, have: np.ndarray, pool: np.ndarray,
                     lam: float, fit_bias: bool = True):
    """(mean, the whole posterior covariance of the pool) from a fit on what has been made.

    The diagonal of that covariance is what :func:`_fit_predict` returns; a batch rule needs the
    rest of it, because what a second measurement adds depends on the first.
    """
    from .model import krr_fit

    Ktr = K[np.ix_(have, have)]
    a, b = krr_fit(Ktr, y[have], lam, fit_bias)
    Kxt = K[np.ix_(pool, have)]
    mu = Kxt @ a + b
    A = Ktr + lam * np.eye(len(have))
    try:
        L = np.linalg.cholesky(A)
        V = np.linalg.solve(L, Kxt.T)
        cov = K[np.ix_(pool, pool)] - V.T @ V
    except np.linalg.LinAlgError:
        cov = K[np.ix_(pool, pool)] - Kxt @ np.linalg.pinv(A) @ Kxt.T
    return mu, 0.5 * (cov + cov.T)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def learning_curve(kernels: Dict[float, np.ndarray], y: Sequence[float], kind: str,
                   start: Sequence[int], steps: int, lams: Sequence[float] = (1e-2, 0.1, 1.0),
                   beta: float = 1.0, xi: float = 0.0, seed: int = 0,
                   fit_bias: bool = True, batch: int = 1, batch_rule: str = "cosine",
                   max_similarity: float = 0.98) -> Dict[str, list]:
    """Run the acquisition loop over a set whose activities are already known.

    `kernels` maps an attenuation to its full joint kernel over every molecule, so the loop can
    reselect the hyper-parameters at every step -- by the evidence, from what has been acquired
    and nothing else, which is what item B4 made affordable.  Nothing outside `start` and the
    molecules the rule asks for is ever looked at.

    With `batch` above one the model is refitted once per batch and not once per molecule,
    which is what a laboratory that makes ten compounds at a time does (plan item MC2).
    `batch_rule` is then how the rest of the batch is chosen after the first pick: ``"cosine"``
    takes the next best scores and skips what looks too similar, ``"covariance"`` recomputes the
    score from the posterior covariance conditioned on the picks already made.

    Returns, after every acquisition or batch: the RMSE and q2 of the model on the molecules
    *not* yet made, the best activity among those made, and which molecules were taken.
    """
    from .evidence import log_evidence

    y = np.asarray(y, float)
    n = len(y)
    rng = np.random.default_rng(seed)
    var_all = float(np.var(y)) if n > 1 else float("nan")
    have = list(dict.fromkeys(int(i) for i in start))
    pool = [i for i in range(n) if i not in set(have)]
    alphas = sorted(kernels)
    out = {"rmse": [], "q2": [], "best": [], "taken": [], "alpha": [], "lam": []}
    for _step in range(int(steps)):
        if not pool:
            break
        h = np.asarray(have, int)
        p = np.asarray(pool, int)
        # hyper-parameters from what has been made, by the evidence
        pick = None
        for a in alphas:
            Ktr = kernels[a][np.ix_(h, h)]
            for lam in lams:
                e = log_evidence(Ktr, y[h], float(lam), fit_bias=fit_bias)
                if np.isfinite(e) and (pick is None or e > pick[0]):
                    pick = (e, float(a), float(lam))
        _e, alpha, lam = pick
        K = kernels[alpha]
        want = min(int(batch), len(pool))
        if want > 1 and batch_rule == "covariance":
            mu, cov = _fit_predict_cov(K, y, h, p, lam, fit_bias)
            sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        else:
            mu, sd = _fit_predict(K, y, h, p, lam, fit_bias)
            cov = None
        dist = _distance_to_training(K[np.ix_(p, h)], np.diag(K)[p], np.diag(K)[h])
        score = acquisition_scores(kind, mu, sd, best=float(np.max(y[h])), beta=beta, xi=xi,
                                   distance=dist, rng=rng)
        out["rmse"].append(_rmse(y[p], mu))
        # against the variance of the whole set, not of the pool: a rule that acquires the
        # extremes shrinks the pool's variance, and a q2 measured against that would reward it
        # for having made the remaining molecules harder to tell apart
        out["q2"].append(1.0 - (_rmse(y[p], mu) ** 2) / var_all if var_all > 0 else float("nan"))
        out["best"].append(float(np.max(y[h])))
        out["alpha"].append(alpha)
        out["lam"].append(lam)
        if want <= 1:
            chosen = [int(np.argmax(score))]
        elif batch_rule == "covariance":
            chosen = batch_by_covariance(kind, mu, cov, want, best=float(np.max(y[h])),
                                         beta=beta, xi=xi, distance=dist, rng=rng)
        elif batch_rule == "cosine":
            cosine = None
            if max_similarity < 1.0 and len(p) > 1:
                block = K[np.ix_(p, p)]
                d = np.sqrt(np.maximum(np.diag(block), 1e-300))
                cosine = block / np.outer(d, d)
            chosen = batch_by_cosine(score, cosine, want, max_similarity=max_similarity)
        else:
            raise ValueError(f"unknown batch rule {batch_rule!r}; one of "
                             f"{', '.join(BATCH_RULES)}")
        taken = [int(p[int(c)]) for c in chosen]
        out["taken"].append(taken if want > 1 else taken[0])
        for t in taken:
            have.append(t)
            pool.remove(t)
    out["have"] = list(have)
    # the error after the last acquisition, which the loop above never reaches: every entry of
    # "rmse" is measured before the molecules of that step are acquired, and a batch run has few
    # of them
    if pool:
        h, p = np.asarray(have, int), np.asarray(pool, int)
        mu, _sd = _fit_predict(kernels[out["alpha"][-1]], y, h, p, out["lam"][-1], fit_bias)
        out["rmse_end"] = _rmse(y[p], mu)
        out["q2_end"] = (1.0 - (out["rmse_end"] ** 2) / var_all if var_all > 0 else float("nan"))
    else:
        out["rmse_end"] = float("nan")
        out["q2_end"] = float("nan")
    return out
