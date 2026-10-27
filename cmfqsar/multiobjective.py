"""A design loop with more than one objective (plan item MC11).

Plan items E2 and E3 close the loop on potency alone.  Lead optimisation has at least three
objectives -- the potency, the selectivity against an anti-target panel, and whatever property
of the same series is measured -- and since plan items E1 and MC4 those live on one kernel: a
multi-task model predicts them all, with a posterior for each.

What is missing between that model and a chemist's list is three things, and this module is
them:

* **the Pareto front**, the candidates no other candidate beats on every objective at once.  A
  weighted sum has to be given weights, and nobody knows them before the compounds are made; a
  front does not, and the chemist chooses from it.
* **an acquisition over the objectives**, so that the front can be ordered and a batch taken
  from it.  Expected improvement is computed per objective against the best already made, and
  the objectives are combined by a weighted sum of those improvements, which is scale-free
  because each one is already in its own units of improvement.
* **a sentence per suggestion** saying what it gains and what it loses against the compound it
  is to replace.  A loop that reports only a score hides the trade it is making.

The batch rule of plan item MC2 generalises without a change of idea: after a pick, condition
**every** task's posterior covariance on it (:func:`cmfqsar.active.condition_on`) and compute
the acquisition again, so a plate is not one compound ten times over.  What that conditioning
takes away is the uncertainty, not the promise: two compounds the model is sure are a log unit
better than anything made are both still worth making, and only the part of their score that
came from not knowing them falls.  A rule that spread the batch on the predictions instead
would be refusing a compound for resembling a good one.

Nothing here is specific to two objectives or to the fields: it takes predicted means and
posterior covariances, one per objective, and gives back an order.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .active import condition_on, expected_improvement

#: what to do with an objective: maximise a potency, minimise an off-target activity
MAXIMISE, MINIMISE = 1, -1


@dataclass
class Objective:
    """One thing the loop is optimising, and which way is better."""
    name: str
    direction: int = MAXIMISE
    weight: float = 1.0
    best: Optional[float] = None          # the best value already made, for the improvement

    @property
    def sign(self) -> float:
        return 1.0 if int(self.direction) >= 0 else -1.0

    def __str__(self) -> str:
        way = "as high as possible" if self.sign > 0 else "as low as possible"
        return f"{self.name} ({way})"


def dominates(a: Sequence[float], b: Sequence[float], signs: Sequence[float]) -> bool:
    """Is `a` at least as good as `b` everywhere and better somewhere?"""
    a = np.asarray(a, float) * np.asarray(signs, float)
    b = np.asarray(b, float) * np.asarray(signs, float)
    return bool(np.all(a >= b) and np.any(a > b))


def pareto_layers(values: np.ndarray, objectives: Sequence[Objective]) -> np.ndarray:
    """Which front every candidate belongs to: 1 is the Pareto front, 2 is the next, and so on.

    `values` is (candidates, objectives) of predicted values, in the units of the objectives.
    """
    v = np.asarray(values, float)
    signs = np.array([o.sign for o in objectives], float)
    n = len(v)
    layer = np.zeros(n, int)
    left = np.ones(n, bool)
    current = 1
    while left.any():
        rows = np.where(left)[0]
        front = []
        for i in rows:
            if not any(dominates(v[j], v[i], signs) for j in rows if j != i):
                front.append(i)
        if not front:                      # ties everywhere: everything left is one front
            front = list(rows)
        layer[np.asarray(front, int)] = current
        left[np.asarray(front, int)] = False
        current += 1
    return layer


def pareto_front(values: np.ndarray, objectives: Sequence[Objective]) -> np.ndarray:
    """The indices of the candidates nothing else beats on every objective at once."""
    return np.where(pareto_layers(values, objectives) == 1)[0]


def improvements(values: np.ndarray, sds: np.ndarray,
                 objectives: Sequence[Objective]) -> np.ndarray:
    """(candidates, objectives) of expected improvement, each in its own objective's units."""
    v = np.asarray(values, float)
    s = np.asarray(sds, float)
    out = np.zeros_like(v)
    for j, o in enumerate(objectives):
        if o.best is None:
            out[:, j] = o.sign * v[:, j]
            continue
        out[:, j] = expected_improvement(o.sign * v[:, j], s[:, j], o.sign * float(o.best))
    return out


def acquisition(values: np.ndarray, sds: np.ndarray,
                objectives: Sequence[Objective]) -> np.ndarray:
    """One score per candidate: the weighted sum of the improvements over the objectives.

    Each term is an expected improvement, which is already in the units of its own objective's
    improvement, so the weights say how much a log unit of one is worth against a log unit of
    another and nothing else is hidden in them.
    """
    w = np.array([float(o.weight) for o in objectives], float)
    return improvements(values, sds, objectives) @ w


def batch(values: np.ndarray, covariances: Sequence[np.ndarray], objectives: Sequence[Objective],
          size: int, only: Optional[Sequence[int]] = None) -> List[int]:
    """A batch of `size` candidates, the posterior of every objective conditioned on each pick.

    This is the rule of plan item MC2 with more than one task: after a pick, every objective's
    covariance is conditioned on it, so a candidate whose uncertainty that pick has already
    removed -- in any of the objectives -- scores lower for the next one.  `only` restricts the
    choice to a subset, which is how a batch is taken from the Pareto front alone.
    """
    v = np.asarray(values, float)
    covs = [np.asarray(c, float).copy() for c in covariances]
    allowed = (np.arange(len(v)) if only is None else np.asarray(only, int))
    taken: List[int] = []
    for _ in range(int(size)):
        sds = np.column_stack([np.sqrt(np.maximum(np.diag(c), 0.0)) for c in covs])
        score = acquisition(v, sds, objectives)
        mask = np.full(len(v), -np.inf)
        mask[allowed] = 0.0
        if taken:
            mask[np.asarray(taken, int)] = -np.inf
        best = np.argmax(score + mask)
        if not np.isfinite(score[best] + mask[best]):
            break
        taken.append(int(best))
        covs = [condition_on(c, int(best)) for c in covs]
    return taken


# --------------------------------------------------------------------------- #
# What a suggestion gains and what it loses
# --------------------------------------------------------------------------- #


@dataclass
class Trade:
    """One candidate against one reference compound, objective by objective.

    The numbers in `gains` and `loses` are improvements **in the direction of the objective**:
    positive is better whichever way the objective runs, so a hERG activity that rises by one
    log unit appears as -1.00 in `loses`.
    """
    name: str
    against: str
    gains: List[Tuple[str, float]] = _field(default_factory=list)
    loses: List[Tuple[str, float]] = _field(default_factory=list)
    same: List[str] = _field(default_factory=list)

    def __str__(self) -> str:
        bits = []
        if self.gains:
            bits.append("gains " + ", ".join(f"{n} {v:+.2f}" for n, v in self.gains))
        if self.loses:
            bits.append("loses " + ", ".join(f"{n} {v:+.2f}" for n, v in self.loses))
        if not bits:
            bits.append("no change worth reporting")
        return f"{self.name} against {self.against}: " + "; ".join(bits)


def trade(values: Sequence[float], reference: Sequence[float], objectives: Sequence[Objective],
          name: str = "", against: str = "", indifference: float = 0.0) -> Trade:
    """What one candidate gains and what it loses against a reference, objective by objective.

    `indifference` is the change below which an objective is called unchanged; the sensible
    value is the model's own error, since a loop should not report a trade it cannot see.
    """
    out = Trade(name=name or "the candidate", against=against or "the reference")
    for j, o in enumerate(objectives):
        d = o.sign * (float(values[j]) - float(reference[j]))
        if abs(d) <= float(indifference):
            out.same.append(o.name)
        elif d > 0:
            out.gains.append((o.name, float(d)))
        else:
            out.loses.append((o.name, float(d)))
    return out


def best_so_far(y: np.ndarray, objectives: Sequence[Objective]) -> List[float]:
    """The best value of every objective among the compounds already made."""
    y = np.asarray(y, float)
    out = []
    for j, o in enumerate(objectives):
        col = y[:, j]
        col = col[np.isfinite(col)]
        out.append(float(np.max(o.sign * col) * o.sign) if len(col) else float("nan"))
    return out


def with_best(objectives: Sequence[Objective], y: np.ndarray) -> List[Objective]:
    """The same objectives with `best` filled in from what has been made."""
    tops = best_so_far(y, objectives)
    return [Objective(name=o.name, direction=o.direction, weight=o.weight, best=t)
            for o, t in zip(objectives, tops)]


def report(values: np.ndarray, objectives: Sequence[Objective], names: Sequence[str],
           picks: Sequence[int], reference: Optional[Sequence[float]] = None,
           reference_name: str = "the best already made",
           indifference: float = 0.0) -> List[Trade]:
    """One trade per suggestion, against the best compound already made unless told otherwise."""
    v = np.asarray(values, float)
    if reference is None:
        reference = [o.best for o in objectives]
    return [trade(v[int(i)], reference, objectives, name=str(names[int(i)]),
                  against=reference_name, indifference=indifference) for i in picks]
