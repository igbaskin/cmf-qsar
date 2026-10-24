"""Is this set of molecules like that one?  A two-sample test in the field Hilbert space.

The applicability domain of :mod:`cmfqsar.validation` answers the question one molecule at a
time: how close is *this* compound to the training set.  It cannot answer the question a
screening run actually raises, which is about the library as a whole -- are these molecules
drawn from the same population as the ones the model was fitted on, or is the whole exercise an
extrapolation?  Averaging a per-molecule verdict does not answer it either: two sets can have
identical nearest-neighbour similarities and still occupy different regions.

The maximum mean discrepancy (Gretton et al., JMLR 13 (2012) 723) is the natural answer when a
kernel is already in hand.  Every set of molecules has a mean field -- the mean of its feature
maps, which for CMF is a function of space in its own right -- and the MMD is the distance
between the two mean fields:

    MMD^2(X, Y) = ||mean_i phi(x_i) - mean_j phi(y_j)||^2
                = mean K(x,x') - 2 mean K(x,y) + mean K(y,y')

so it is three averages over blocks of the kernel matrix and nothing else.  The unbiased
estimator drops the diagonal terms, which is what lets it be negative on sets that are really
the same -- a feature, not a defect: a small negative value is how "no difference at all" looks
when the diagonal bias has been removed.

Because the kernel is characteristic in practice, MMD = 0 only when the two field distributions
coincide, so the test has power against differences no single summary statistic would show.  The
null distribution has no closed form worth using at these sample sizes, so it is obtained by
permutation: shuffle the labels, recompute, repeat.  That costs three block means per shuffle
and nothing else, since the kernel is computed once.

The **witness** says *where* the difference lies.  It is the function whose expectation separates
the two sets, w(x) = mean_i K(x, x_i) - mean_j K(x, y_j), evaluated at each molecule: large and
positive for molecules typical of the first set, large and negative for the second.  Ranking a
library by it names the compounds that make it unlike the training set, which is the actionable
form of a p-value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["MMDResult", "assemble", "kernel_over", "mmd2", "mmd_test", "witness"]


@dataclass
class MMDResult:
    """The statistic, its null distribution and what it means."""
    mmd2: float                      # the unbiased estimator; may be slightly negative
    mmd: float                       # its square root, zero when the estimator is negative
    p_value: float
    n_x: int
    n_y: int
    permutations: int
    null_mean: float = float("nan")
    null_sd: float = float("nan")
    z: float = float("nan")

    def describe(self) -> str:
        verdict = ("the two sets are not distinguishable in this kernel"
                   if self.p_value > 0.05 else
                   "the two sets occupy different regions of field space")
        return (f"MMD = {self.mmd:.4f} (MMD^2 = {self.mmd2:+.5f}) between {self.n_x} and "
                f"{self.n_y} molecules, p = {self.p_value:.4f} from {self.permutations} "
                f"permutations (z = {self.z:+.1f}): {verdict}")

    def as_dict(self) -> Dict[str, float]:
        return {"mmd": self.mmd, "mmd2": self.mmd2, "p_value": self.p_value, "z": self.z,
                "n_x": self.n_x, "n_y": self.n_y, "permutations": self.permutations}


def _blocks(K: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    i = np.flatnonzero(mask)
    j = np.flatnonzero(~mask)
    return K[np.ix_(i, i)], K[np.ix_(j, j)], K[np.ix_(i, j)]


def mmd2(K: np.ndarray, mask: Sequence[bool], unbiased: bool = True) -> float:
    """MMD^2 between the molecules `mask` selects and the rest, from their kernel matrix.

    The unbiased estimator leaves the self-similarities out of the two within-set means, so that
    it estimates the squared distance of the mean fields rather than that distance plus a bias
    that falls off only as 1/n.  On two samples of the same distribution it is centred on zero
    and takes either sign.
    """
    K = np.asarray(K, float)
    mask = np.asarray(mask, bool)
    Kxx, Kyy, Kxy = _blocks(K, mask)
    m, n = Kxx.shape[0], Kyy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    if not unbiased:
        return float(Kxx.mean() - 2.0 * Kxy.mean() + Kyy.mean())
    sx = (float(Kxx.sum()) - float(np.trace(Kxx))) / (m * (m - 1))
    sy = (float(Kyy.sum()) - float(np.trace(Kyy))) / (n * (n - 1))
    return float(sx + sy - 2.0 * float(Kxy.mean()))


def mmd_test(K: np.ndarray, mask: Sequence[bool], permutations: int = 999,
             seed: int = 0, unbiased: bool = True) -> MMDResult:
    """Permutation test of MMD^2: how often does a reshuffling of the labels do as well?

    The kernel is computed once and every permutation is three block means over it, so the test
    costs a fraction of the kernel it is built on.  The p-value is (1 + #{null >= observed}) /
    (1 + permutations), which never returns zero -- a p-value of 0 would claim more than a finite
    number of shuffles can support.
    """
    K = np.asarray(K, float)
    mask = np.asarray(mask, bool)
    n_x, n_y = int(mask.sum()), int((~mask).sum())
    observed = mmd2(K, mask, unbiased=unbiased)
    if not np.isfinite(observed) or permutations < 1:
        return MMDResult(mmd2=observed, mmd=float(np.sqrt(max(observed, 0.0))),
                         p_value=float("nan"), n_x=n_x, n_y=n_y, permutations=0)
    rng = np.random.RandomState(int(seed))
    null = np.empty(int(permutations))
    shuffled = mask.copy()
    for t in range(int(permutations)):
        rng.shuffle(shuffled)
        null[t] = mmd2(K, shuffled, unbiased=unbiased)
    p = (1.0 + float(np.sum(null >= observed))) / (1.0 + float(permutations))
    sd = float(np.std(null))
    return MMDResult(mmd2=float(observed), mmd=float(np.sqrt(max(observed, 0.0))), p_value=p,
                     n_x=n_x, n_y=n_y, permutations=int(permutations),
                     null_mean=float(np.mean(null)), null_sd=sd,
                     z=float((observed - np.mean(null)) / sd) if sd > 0 else float("nan"))


def witness(K: np.ndarray, mask: Sequence[bool]) -> np.ndarray:
    """w(x) = mean K(x, first set) - mean K(x, second set), one value per molecule.

    Positive where a molecule looks like the first set and negative where it looks like the
    second; its largest magnitudes name the compounds that make the two sets differ, which is
    what a p-value cannot tell you.
    """
    K = np.asarray(K, float)
    mask = np.asarray(mask, bool)
    i, j = np.flatnonzero(mask), np.flatnonzero(~mask)
    if len(i) == 0 or len(j) == 0:
        return np.zeros(K.shape[0])
    return K[:, i].mean(axis=1) - K[:, j].mean(axis=1)


# --------------------------------------------------------------------------- #
# From molecules rather than from a kernel
# --------------------------------------------------------------------------- #

def assemble(Kxx: np.ndarray, Kyx: np.ndarray, Kyy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """One kernel matrix and its mask from the three blocks a fitted model can already give.

    `Kyx` is the (new, train) block a prediction computes anyway, so comparing a library with a
    model's training set costs only the library's own kernel on top of the prediction.
    """
    Kxx = np.asarray(Kxx, float)
    Kyy = np.asarray(Kyy, float)
    Kyx = np.atleast_2d(np.asarray(Kyx, float))
    m, n = Kxx.shape[0], Kyy.shape[0]
    if Kyx.shape != (n, m):
        raise ValueError(f"the cross block should be ({n}, {m}), not {Kyx.shape}")
    K = np.empty((m + n, m + n))
    K[:m, :m] = Kxx
    K[m:, m:] = Kyy
    K[m:, :m] = Kyx
    K[:m, m:] = Kyx.T
    mask = np.zeros(m + n, dtype=bool)
    mask[:m] = True
    return K, mask


def kernel_over(mols_a, names_a, mols_b, names_b, fieldset, normalize: bool = True):
    """The joint CMF kernel over both sets at once, and the mask selecting the first.

    One table of both sets is built so that every molecule is parameterised identically and the
    kernel is the one a model would use; comparing sets with kernels built separately would
    compare their parameterisations as much as their chemistry.
    """
    from .fields import AtomTable, build_fields
    from .model import CMFModel

    fs = fieldset
    mf = [build_fields(m, fs, n) for m, n in zip(list(mols_a) + list(mols_b),
                                                 list(names_a) + list(names_b))]
    model = CMFModel(fieldset=fs, method="krr", normalize=normalize)
    K = model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(mf)))
    mask = np.zeros(len(mf), dtype=bool)
    mask[:len(list(mols_a))] = True
    return K, mask
