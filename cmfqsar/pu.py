"""Learning from positives and unlabelled molecules (part of plan item E4).

A screening library is not a labelled set.  The compounds that came back negative in one assay at
one concentration are *unlabelled* rather than inactive: some are genuinely inert, some were
mis-measured, some would have worked at another concentration, and a few are actives nobody has
recognised yet.  Treating that mixture as a negative class teaches a classifier to reject the
very molecules a screen is looking for, and the error is worst exactly where it matters, at the
top of the ranking.

The remedy used here is **bagging PU** (Mordelet & Vert, *Pattern Recognit. Lett.* **37** (2014)
201), which is the method that needs no estimate of the class prior and no assumption about how
the positives were sampled:

1. draw a small random sample of the unlabelled set and call it negative;
2. fit a classifier on the positives against that sample;
3. score the unlabelled molecules that were *not* in the sample -- out of bag, so the score of a
   molecule is never taken from a model that was told it was negative;
4. repeat, and average each molecule's out-of-bag scores.

An unlabelled molecule that looks like the positives is called negative in only a few bags, and
in the bags where it is left out it scores high; one that is genuinely unlike them is called
negative often and scores low wherever it appears.  The average is the quantity to rank by.

Everything here works on a precomputed kernel, so the CMF kernel enters unchanged and the
positives keep whatever pose they were given.  :func:`bagging_pu` is the general form;
:func:`pu_scores` is the convenience that fits the bags and returns one score per molecule.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: learners the bags may use; both are solved in closed form on the kernel
METHODS = ("krr", "gp")


@dataclass
class PUResult:
    """Out-of-bag scores for every molecule, and how they were arrived at."""

    scores: np.ndarray                  # one per molecule of the kernel, nan where never scored
    bags: int
    bag_size: int
    counts: np.ndarray                  # how many bags scored each molecule out of bag
    positives: np.ndarray
    unlabelled: np.ndarray
    note: str = ""

    @property
    def covered(self) -> float:
        """The share of the unlabelled molecules that got at least one out-of-bag score."""
        c = self.counts[self.unlabelled]
        return float(np.mean(c > 0)) if len(c) else float("nan")

    def describe(self) -> str:
        return (f"{self.bags} bags of {self.bag_size} unlabelled molecules against "
                f"{len(self.positives)} positives; every unlabelled molecule scored out of bag "
                f"{float(np.median(self.counts[self.unlabelled])):.0f} times on median")


def _fit_predict(K: np.ndarray, train: np.ndarray, y: np.ndarray, test: np.ndarray,
                 lam: float) -> np.ndarray:
    """Kernel ridge on labels in {+1, -1}, which is the closed form both methods share."""
    Ktr = K[np.ix_(train, train)]
    n = len(train)
    A = Ktr + float(lam) * np.eye(n)
    try:
        a = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(A, y, rcond=None)[0]
    return K[np.ix_(test, train)] @ a


def bagging_pu(K: np.ndarray, positives: Sequence[int], unlabelled: Sequence[int],
               bags: int = 30, bag_size: Optional[int] = None, lam: float = 0.1,
               seed: int = 0, also: Optional[Sequence[int]] = None) -> PUResult:
    """Bagged positive-unlabelled scores on a precomputed kernel.

    `also` names molecules that are in neither set -- a held-out test fold, say -- which are
    scored by every bag, since no bag ever sees them. Their scores are therefore comparable with
    the out-of-bag scores of the unlabelled set, which is what makes a ranking over both of them
    honest.
    """
    K = np.asarray(K, float)
    pos = np.asarray(positives, int)
    unl = np.asarray(unlabelled, int)
    extra = np.asarray(also if also is not None else [], int)
    n = K.shape[0]
    if len(pos) == 0 or len(unl) == 0:
        raise ValueError("bagging PU needs positives and unlabelled molecules")
    # a bag that holds the whole unlabelled set leaves nothing out of bag, and the method is
    # then the naive classifier wearing its name; half the set is the most that can be asked for
    size = int(bag_size) if bag_size else min(len(pos), max(1, len(unl) // 2))
    size = max(1, min(size, len(unl)))
    total = np.zeros(n)
    counts = np.zeros(n, int)
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.ones(len(pos)), -np.ones(size)])
    for _b in range(int(bags)):
        bag = rng.choice(len(unl), size=size, replace=False)
        in_bag = unl[bag]
        out_bag = np.setdiff1d(unl, in_bag, assume_unique=False)
        train = np.concatenate([pos, in_bag])
        test = np.concatenate([out_bag, extra]) if len(extra) else out_bag
        if len(test) == 0:
            continue
        total[test] += _fit_predict(K, train, y, test, lam)
        counts[test] += 1
    scores = np.full(n, np.nan)
    ok = counts > 0
    scores[ok] = total[ok] / counts[ok]
    return PUResult(scores=scores, bags=int(bags), bag_size=size, counts=counts,
                    positives=pos, unlabelled=unl)


def pu_scores(K: np.ndarray, positives: Sequence[int], unlabelled: Sequence[int],
              test: Optional[Sequence[int]] = None, bags: int = 30,
              bag_size: Optional[int] = None, lam: float = 0.1, seed: int = 0) -> np.ndarray:
    """The ranking a PU model gives, as one number per molecule (nan where never scored)."""
    return bagging_pu(K, positives, unlabelled, bags=bags, bag_size=bag_size, lam=lam,
                      seed=seed, also=test).scores


def naive_scores(K: np.ndarray, positives: Sequence[int], unlabelled: Sequence[int],
                 test: Optional[Sequence[int]] = None, lam: float = 0.1) -> np.ndarray:
    """The ranking you get by calling every unlabelled molecule a negative, once.

    The control bagging PU has to beat: the same learner and the same kernel, with the whole
    unlabelled set used as the negative class and no bagging. Out-of-sample scores are available
    only for `test`, so this is the ranking of the held-out molecules, not of the training ones.
    """
    K = np.asarray(K, float)
    pos = np.asarray(positives, int)
    unl = np.asarray(unlabelled, int)
    train = np.concatenate([pos, unl])
    y = np.concatenate([np.ones(len(pos)), -np.ones(len(unl))])
    out = np.full(K.shape[0], np.nan)
    which = np.asarray(test if test is not None else [], int)
    if len(which):
        out[which] = _fit_predict(K, train, y, which, lam)
    return out
