"""Which molecule to make next (plan item E2).

A fitted CMF model predicts an activity and, through the Gaussian-process view, a posterior
standard deviation beside it.  Those two numbers are enough to answer a question the 2019 paper
poses and leaves as a statement -- *maximising the overlap tells how to modify the molecule* --
as an operation with a number: given a set of posed candidates, which one should be made?

The answer depends on what the asker is doing, and the module keeps the two apart rather than
pretending one rule serves both.

* **To find an active compound** the acquisition is optimistic.  Expected improvement asks how
  much a candidate is expected to beat the best thing already made, counting only the upside::

      EI(x) = (mu - y* - xi) Phi(z) + sigma phi(z),   z = (mu - y* - xi) / sigma

  and the upper confidence bound ``mu + beta sigma`` is the cruder form of the same instinct.  A
  molecule the model is unsure about is *promoted*, because being wrong about it is cheap and
  being right is valuable.

* **To improve the model** the acquisition ignores the prediction entirely and takes the largest
  posterior standard deviation, or -- with no model at all -- the candidate furthest in field
  space from everything already made (``maxmin``, the classical furthest-first design).

Note that this is the opposite of :meth:`CMFModel.rank_score`, which ranks a *screening* library
by ``mu - beta sigma``: there a molecule the model knows nothing about must be pushed down,
because a kernel method hands back the training mean dressed up as a prediction and the library
is full of such molecules.  Screening is a bet that has to be paid for; acquisition is a question
that has to be answered.  Optimism is right for one and wrong for the other.

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
            seed: Optional[int] = 0) -> List[Suggestion]:
    """Rank posed candidates by one acquisition rule and return the best `n`.

    The model must be a regression on the linear kernel, since that is where a posterior exists.
    With `n` above one the picks are taken greedily and a candidate too similar to one already
    picked is skipped (`max_similarity`, a cosine in the model's own kernel): a proper batch rule
    would refit the model after every pick, and this is the cheap approximation to it, which is
    worth knowing when a whole plate is being chosen at once.
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

    # cosine similarity between the candidates themselves, so that a batch is not one molecule
    # n times over; computed only when a batch is asked for, since it is a square block
    cos = None
    if int(n) > 1 and max_similarity < 1.0 and len(mu) > 1:
        dQ = model._self_kernels(candidates) if model.normalize else None
        Kcc = model.joint_kernel(model._field_kernels(candidates, candidates, dQ, dQ))
        d = np.sqrt(np.maximum(np.diag(Kcc), 1e-300))
        cos = Kcc / np.outer(d, d)
    picks: List[Suggestion] = []
    for i in np.argsort(-score):
        if len(picks) >= int(n):
            break
        i = int(i)
        if cos is not None and any(cos[i, p.index] > max_similarity for p in picks):
            continue
        picks.append(Suggestion(index=i, name=nm[i], mean=float(mu[i]), sd=float(sd[i]),
                                score=float(score[i]), distance=float(dist[i])))
    return picks


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


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2)))


def learning_curve(kernels: Dict[float, np.ndarray], y: Sequence[float], kind: str,
                   start: Sequence[int], steps: int, lams: Sequence[float] = (1e-2, 0.1, 1.0),
                   beta: float = 1.0, xi: float = 0.0, seed: int = 0,
                   fit_bias: bool = True) -> Dict[str, list]:
    """Run the acquisition loop over a set whose activities are already known.

    `kernels` maps an attenuation to its full joint kernel over every molecule, so the loop can
    reselect the hyper-parameters at every step -- by the evidence, from what has been acquired
    and nothing else, which is what item B4 made affordable.  Nothing outside `start` and the
    molecules the rule asks for is ever looked at.

    Returns, after every acquisition: the RMSE and q2 of the model on the molecules *not* yet
    made, the best activity among those made, and which molecule was taken.
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
        mu, sd = _fit_predict(K, y, h, p, lam, fit_bias)
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
        taken = int(p[int(np.argmax(score))])
        out["taken"].append(taken)
        have.append(taken)
        pool.remove(taken)
    out["have"] = list(have)
    return out
