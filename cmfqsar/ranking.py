"""Learning to order molecules instead of to reproduce their numbers.

A regression model is asked for the activity and judged on the squared error, which assumes the
numbers mean what they say.  Often they do not.  Potencies pooled from several papers carry
offsets no descriptor can explain, a series measured in two assays carries two scales, and an
IC50 read off the end of a dose-response curve carries a bound rather than a value.  In all three
the *order* survives what the values do not, and a model fitted to the values spends its capacity
on the part that is noise.

The ranking support-vector machine (Herbrich et al. 2000; Joachims, KDD 2002) fits the order.
For every pair of molecules whose activities differ it asks that the more active one score higher
by a margin, and it pays for every pair where that fails:

    minimise  |w|^2 / 2 + C sum_p xi_p
    subject to  <w, X_i - X_j> >= 1 - xi_p  for every pair p = (i, j) with y_i > y_j

which is an ordinary support-vector classification on the *difference* of two molecules' fields.
The kernel of two such differences is a sum of four ordinary kernel entries,

    Kpair((i,j), (k,l)) = K(i,k) - K(i,l) - K(j,k) + K(j,l),

so the whole thing is built from the kernel matrix the program already has, with no new fields
and no new parameterisation.

**The fitted model is still a dual expansion over molecules.**  The weight vector is
w = sum_p a_p (X_{i_p} - X_{j_p}), so collecting the terms gives one coefficient per *molecule*:
the score of a new molecule is K(x, train) @ coef exactly as for any other learner here, with no
bias term, and everything that reads a linear CMF model -- co-fields, atom contributions, Shapley
values, the pharmacophore -- keeps working unchanged.  What changes is only what the numbers
mean: a co-field of a ranking model says where more field *promotes* a molecule, not what it adds
to the activity.

What is given up is the scale.  A ranking score has no units and cannot be compared with an
activity; only its order is meaningful.  The program therefore reports Spearman's rho, Kendall's
tau and the fraction of pairs ordered correctly, and does not report r2 or RMSE, which would be
arithmetic on a quantity that has none.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["RankFit", "concordant_pairs", "fit_rank", "make_pairs", "pair_kernel",
           "rank_scores"]


# --------------------------------------------------------------------------- #
# Which pairs to learn from
# --------------------------------------------------------------------------- #

def make_pairs(y: Sequence[float], gap: float = 0.0, groups: Optional[Sequence] = None,
               max_pairs: Optional[int] = None, seed: int = 0) -> np.ndarray:
    """Ordered pairs (i, j) with y_i > y_j, as an (m, 2) integer array.

    `gap` drops the pairs whose activities differ by less than it.  This is not a speed trick: a
    pair separated by less than the experimental error is a coin toss the model would be asked to
    call correctly, and asking it is how a ranking model learns noise.  Half a log unit is a
    defensible default for pooled literature potencies, zero for one careful assay.

    `groups` restricts the pairs to within a group -- the same assay, the same paper, the same
    series -- which is what makes the method worth having on pooled data: an offset between two
    papers cancels inside each of them and is never compared across.

    `max_pairs` subsamples when the quadratic count is too large; the subsample is drawn once with
    `seed` so that a fit is reproducible.  It matters more than it looks: the solver works on the
    kernel *between pairs*, which is m by m for m pairs, so m = 4000 already costs 128 MB and a
    set of 300 molecules would otherwise form 45,000 pairs and ask for 16 GB.  Subsampling the
    constraints is the standard remedy for the pairwise formulation, and the order of a set is
    heavily over-determined: a few thousand comparisons fix it.
    """
    y = np.asarray(y, float)
    n = len(y)
    ok = np.isfinite(y)
    i, j = np.triu_indices(n, k=1)
    keep = ok[i] & ok[j]
    if groups is not None:
        g = np.asarray(groups)
        keep &= (g[i] == g[j])
    d = y[i] - y[j]
    keep &= np.abs(d) > float(gap)
    i, j, d = i[keep], j[keep], d[keep]
    hi = np.where(d > 0, i, j)                     # the more active one first
    lo = np.where(d > 0, j, i)
    pairs = np.column_stack([hi, lo]).astype(int)
    if max_pairs is not None and len(pairs) > int(max_pairs):
        rng = np.random.RandomState(int(seed))
        pairs = pairs[rng.choice(len(pairs), int(max_pairs), replace=False)]
    return pairs


def pair_kernel(K: np.ndarray, pairs: np.ndarray, signs: Optional[np.ndarray] = None,
                other: Optional[np.ndarray] = None,
                other_signs: Optional[np.ndarray] = None) -> np.ndarray:
    """The kernel between difference vectors: K(i,k) - K(i,l) - K(j,k) + K(j,l).

    With `signs` the rows are flipped -- a pair used as a negative example is the same difference
    taken the other way round -- which is how the one-class problem of "every difference is
    positive" is turned into the balanced binary problem a standard solver expects.
    """
    K = np.asarray(K, float)
    a = np.asarray(pairs, int)
    b = a if other is None else np.asarray(other, int)
    out = (K[np.ix_(a[:, 0], b[:, 0])] - K[np.ix_(a[:, 0], b[:, 1])]
           - K[np.ix_(a[:, 1], b[:, 0])] + K[np.ix_(a[:, 1], b[:, 1])])
    if signs is not None:
        out = out * np.asarray(signs, float)[:, None]
    if other_signs is not None:
        out = out * np.asarray(other_signs, float)[None, :]
    return out


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #

@dataclass
class RankFit:
    """What a ranking fit leaves behind: one coefficient per training molecule."""
    coef: np.ndarray
    pairs: np.ndarray
    alpha: np.ndarray                      # the dual variable of every pair, >= 0
    n_pairs: int = 0
    n_support: int = 0

    def score(self, K_test_train: np.ndarray) -> np.ndarray:
        """The ranking score of new molecules; only its order means anything."""
        return np.asarray(K_test_train, float) @ self.coef


def fit_rank(K: np.ndarray, y: Sequence[float], C: float = 1.0, gap: float = 0.0,
             groups: Optional[Sequence] = None, max_pairs: Optional[int] = 4000,
             seed: int = 0) -> RankFit:
    """Fit a ranking SVM on a precomputed kernel and return molecule-level coefficients.

    The pairs are turned into a balanced binary problem by flipping half of them -- a difference
    taken the other way round with the opposite label is the same constraint -- so that an
    ordinary two-class solver can be used.  The intercept it fits is discarded: a constant added
    to every score changes no order, and the pairs are symmetric by construction.
    """
    from sklearn.svm import SVC

    K = np.asarray(K, float)
    n = K.shape[0]
    pairs = make_pairs(y, gap=gap, groups=groups, max_pairs=max_pairs, seed=seed)
    coef = np.zeros(n)
    if len(pairs) == 0:
        return RankFit(coef=coef, pairs=pairs, alpha=np.zeros(0), n_pairs=0, n_support=0)
    rng = np.random.RandomState(int(seed))
    signs = rng.choice([-1.0, 1.0], size=len(pairs))
    if len(np.unique(signs)) == 1:                 # one pair, or an unlucky draw
        signs[0] = -signs[0]
    Kp = pair_kernel(K, pairs, signs=signs, other_signs=signs)
    # the kernel of difference vectors can be a hair non-symmetric after the products above
    Kp = 0.5 * (Kp + Kp.T)
    est = SVC(kernel="precomputed", C=float(C)).fit(Kp, signs)
    alpha = np.zeros(len(pairs))
    # dual_coef_ is alpha * label, and the label is the sign, so the product is alpha itself
    alpha[est.support_] = np.abs(est.dual_coef_.ravel())
    np.add.at(coef, pairs[:, 0], alpha)
    np.add.at(coef, pairs[:, 1], -alpha)
    return RankFit(coef=coef, pairs=pairs, alpha=alpha, n_pairs=len(pairs),
                   n_support=int(len(est.support_)))


# --------------------------------------------------------------------------- #
# Judging an order
# --------------------------------------------------------------------------- #

def concordant_pairs(y_true: Sequence[float], score: Sequence[float],
                     gap: float = 0.0, groups: Optional[Sequence] = None) -> float:
    """The fraction of comparable pairs the score puts the right way round; ties count a half."""
    pairs = make_pairs(y_true, gap=gap, groups=groups, max_pairs=None)
    if len(pairs) == 0:
        return float("nan")
    s = np.asarray(score, float)
    d = s[pairs[:, 0]] - s[pairs[:, 1]]
    return float(np.mean(np.where(d > 0, 1.0, np.where(d < 0, 0.0, 0.5))))


def rank_scores(y_true: Sequence[float], score: Sequence[float], gap: float = 0.0,
                groups: Optional[Sequence] = None) -> Dict[str, float]:
    """Spearman's rho, Kendall's tau and the share of pairs ordered correctly."""
    from scipy.stats import kendalltau, spearmanr

    y = np.asarray(y_true, float)
    s = np.asarray(score, float)
    ok = np.isfinite(y) & np.isfinite(s)
    if ok.sum() < 3:
        return {"spearman": float("nan"), "kendall": float("nan"), "pairs": float("nan")}
    rho = float(spearmanr(y[ok], s[ok]).statistic)
    tau = float(kendalltau(y[ok], s[ok]).statistic)
    return {"spearman": rho, "kendall": tau,
            "pairs": concordant_pairs(y[ok], s[ok], gap=gap,
                                      groups=None if groups is None
                                      else np.asarray(groups)[ok])}
