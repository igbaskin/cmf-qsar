"""Data fusion: one score from the similarities to several known actives.

A similarity search needs a query, and a project that has found more than one active has more
than one.  The screening literature answers that with *data fusion*: compute the similarity of
each library molecule to every known active and combine the results by a fixed rule (Willett,
*Data fusion in ligand-based virtual screening*, 2013, and the group-fusion rules of Fox & Shaw
1994 as they reached chemoinformatics through Whittle, Gillet, Willett et al. 2004).  No model is
fitted and no inactive is needed, which is why fusion is the next step beyond a single-query search
when a few actives are all there is.

The input is one matrix: `S[i, j]`, the similarity of library molecule `i` to reference active
`j`.  For this program that is the cosine-normalised CMF kernel between two sets of molecular
fields, so the geometry of the comparison is the overlap integral and nothing here has to know
it; the same rules run unchanged on a fingerprint Tanimoto, which is what makes the two
representations comparable on one task.

Eight rules, and the two that matter most are at the ends of the list:

``max``
    the similarity to the *best-matching* active.  The rule the literature most often finds
    best, and the one this program did not have: a molecule resembling one active strongly is a
    better candidate than one resembling all of them weakly, because a series usually binds through
    one arrangement and not through an average of several.
``min``
    the similarity to the worst-matching active, which asks for resemblance to all of them at
    once.  Included because it is the opposite extreme and bounds what the others can do.
``sum``
    the total, which ranks exactly as the mean does.  This is the similarity ensemble, and it is
    the rule :mod:`cmfqsar.oneclass`'s centroid description already implements: for a normalised
    kernel the distance to the centroid of the actives is one minus their mean similarity.  It
    is here for completeness and as the baseline the others have to beat.
``med``
    the median, which is the robust form of ``sum`` and ignores one outlying reference.
``anz``, ``mnz``
    Fox and Shaw's combinations, on the actives whose similarity passes a threshold: ``anz`` is
    the sum divided by how many passed and ``mnz`` the sum multiplied by it.  They act in
    opposite directions, ``anz`` rewarding a molecule that matches few actives well and ``mnz``
    one that matches many at all, so the pair bounds how much the *number* of matches should
    count beside their strength.
``euc``
    the Euclidean norm of the similarity vector, which lies between ``max`` and ``sum`` and
    weights a strong match more than a linear sum does without discarding the rest.
``rrf``
    reciprocal rank fusion: each reference ranks the whole library, and a molecule scores
    `sum_j 1 / (k + r_ij)`.  The only rule here that uses ranks rather than similarity values,
    so it is the one that does not care whether the similarity scales of two references agree,
    and the only one whose score for a molecule depends on the rest of the library.

A molecule is never its own reference.  When the library being scored *is* the reference set,
pass ``training=True`` and the diagonal is excluded, exactly as the nearest-neighbour score of
:mod:`cmfqsar.oneclass` does; without that every rule involving a maximum returns the
self-similarity and the ranking is meaningless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

#: rule keys, in the order they are offered
FUSION_RULES = ("max", "min", "sum", "med", "anz", "mnz", "euc", "rrf")

LABELS = {
    "max": "Similarity to the best-matching active (MAX)",
    "min": "Similarity to the worst-matching active (MIN)",
    "sum": "Sum of the similarities (SUM, ranks as the mean)",
    "med": "Median of the similarities (MED)",
    "anz": "Sum over the actives that pass the threshold (ANZ)",
    "mnz": "Sum times the number that pass the threshold (MNZ)",
    "euc": "Euclidean norm of the similarity vector (EUC)",
    "rrf": "Reciprocal rank fusion over the actives (RRF)",
}


def _masked(S: np.ndarray, training: bool) -> np.ndarray:
    """The similarity matrix with self-matches removed when the library is the reference set."""
    M = np.array(S, dtype=float, copy=True)
    if training:
        n = min(M.shape)
        if M.shape[0] != M.shape[1]:
            raise ValueError("training=True needs a square similarity matrix: the library and "
                             "the reference set must be the same molecules in the same order")
        M[np.arange(n), np.arange(n)] = np.nan
    return M


def fuse(S: np.ndarray, rule: str = "max", threshold: float = 0.0, rrf_k: float = 60.0,
         top_fraction: Optional[float] = None, training: bool = False) -> np.ndarray:
    """Combine the similarities of every library molecule to several actives into one score.

    `S` is (library, references).  Larger is better for every rule, so the result can be ranked,
    thresholded or handed to the screening metrics directly.  `threshold` is what counts as a
    match for ``anz`` and ``mnz``; `rrf_k` is the constant of reciprocal rank fusion, and
    `top_fraction`, if given, restricts the rules that count matches to the actives for which
    the molecule falls in that leading fraction of the library, which is the form in which the
    rank-based rule is usually quoted.

    A row of nothing but missing values scores as missing rather than as zero, which keeps a
    molecule that could not be compared with anything out of the top of the list.
    """
    if rule not in FUSION_RULES:
        raise ValueError(f"unknown fusion rule {rule!r}; use one of {FUSION_RULES}")
    M = _masked(np.atleast_2d(S), training)
    if M.size == 0:
        return np.zeros(M.shape[0])
    ok = np.isfinite(M)
    n_ref = ok.sum(axis=1)
    empty = n_ref == 0
    filled = np.where(ok, M, 0.0)

    if rule == "rrf":
        # every reference ranks the whole library; rank 1 is the most similar
        order = np.argsort(-np.where(ok, M, -np.inf), axis=0)
        ranks = np.empty(M.shape, float)
        rows = np.arange(M.shape[0])
        for j in range(M.shape[1]):
            ranks[order[:, j], j] = rows + 1.0
        use = ok.copy()
        if top_fraction is not None:
            use &= ranks <= max(1.0, float(top_fraction) * M.shape[0])
        contrib = np.where(use, 1.0 / (float(rrf_k) + ranks), 0.0)
        out = contrib.sum(axis=1)
    elif rule == "max":
        out = np.where(empty, np.nan, np.nanmax(np.where(ok, M, -np.inf), axis=1))
    elif rule == "min":
        out = np.where(empty, np.nan, np.nanmin(np.where(ok, M, +np.inf), axis=1))
    elif rule == "sum":
        out = filled.sum(axis=1)
    elif rule == "med":
        # only the rows that have something, so that an empty one raises no all-NaN warning
        out = np.full(M.shape[0], np.nan)
        keep = ~empty
        if keep.any():
            out[keep] = np.nanmedian(M[keep], axis=1)
    elif rule == "euc":
        out = np.sqrt((filled ** 2).sum(axis=1))
    else:                                   # anz, mnz
        passed = ok & (M > float(threshold))
        if top_fraction is not None:
            order = np.argsort(-np.where(ok, M, -np.inf), axis=0)
            ranks = np.empty(M.shape, float)
            rows = np.arange(M.shape[0])
            for j in range(M.shape[1]):
                ranks[order[:, j], j] = rows + 1.0
            passed &= ranks <= max(1.0, float(top_fraction) * M.shape[0])
        nz = passed.sum(axis=1).astype(float)
        total = np.where(passed, M, 0.0).sum(axis=1)
        if rule == "anz":
            out = np.where(nz > 0, total / np.maximum(nz, 1.0), 0.0)
        else:
            out = total * nz
    out = np.asarray(out, float)
    out[empty] = np.nan
    return out


@dataclass
class FusionScorer:
    """A fusion rule with the interface of a fitted one-class description.

    It holds no model: the references *are* the description, and `fit` only records them and
    places the offset so that a fraction `nu` of them falls outside, which is what makes a
    decision value here mean the same thing as one from :mod:`cmfqsar.oneclass`.  The purpose
    of fusion is that nothing is fitted, so there is nothing to overfit and nothing to
    cross-validate; what there is instead is a choice of rule, and the rule has to be chosen on
    data of its own if the choice is not to become the fit.
    """
    kind: str = "max"
    nu: float = 0.1
    threshold: float = 0.0
    rrf_k: float = 60.0
    top_fraction: Optional[float] = None

    offset_: float = 0.0
    n_ref_: int = 0
    diag_: Optional[np.ndarray] = None   # training self-kernels, so the model can detect a refit

    #: Fusion is a rule on the kernel row rather than a kernel expansion with fixed weights, so
    #: no co-field, atom contribution or Shapley value exists for one.  ``sum`` is the exception
    #: in principle, being linear in the row, but it is exactly the centroid description that
    #: :mod:`cmfqsar.oneclass` already offers as ``ocmean`` *with* its co-field, so a second
    #: linear path to the same map would only be a way of getting it wrong twice.
    is_linear: bool = False

    @property
    def label(self) -> str:
        """How this rule is named in a message to the user."""
        return LABELS.get(self.kind, self.kind)

    @property
    def why_not_linear(self) -> str:
        """Why no co-field exists, in a clause that can be dropped into an error message."""
        return {"max": "it is a maximum over the actives",
                "min": "it is a minimum over the actives",
                "med": "it is a median over the actives",
                "euc": "it is a Euclidean norm of the row",
                "rrf": "it is a sum of reciprocal ranks, not of kernel values",
                "anz": "it divides by a count that depends on the molecule",
                "mnz": "it multiplies by a count that depends on the molecule",
                }.get(self.kind, "it is not a sum over the training molecules with fixed weights")

    def fit(self, K: np.ndarray) -> "FusionScorer":
        """Record the reference set and calibrate the offset on it, self-matches excluded."""
        K = np.asarray(K, float)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("fit takes the square similarity matrix of the reference actives")
        if K.shape[0] < 2:
            raise ValueError("data fusion needs at least two reference actives")
        self.n_ref_ = int(K.shape[0])
        self.diag_ = np.diag(K).copy()
        self.offset_ = 0.0
        inside = self.decision(K, training=True)
        good = inside[np.isfinite(inside)]
        if good.size:
            self.offset_ = -float(np.quantile(good, np.clip(self.nu, 0.0, 0.99)))
        return self

    def decision(self, K_test_train: np.ndarray, k_test_self: Optional[np.ndarray] = None,
                 training: bool = False) -> np.ndarray:
        """Fused score of every molecule; at or above zero is inside the description.

        `k_test_self` is accepted and ignored.  Every rule here is a function of the similarity
        row alone, so unlike the centroid and subspace distances of :mod:`cmfqsar.oneclass` this
        needs no self-kernel -- which also means it says nothing unless the kernel is normalised,
        since an unnormalised row ranks by molecular size.
        """
        s = fuse(K_test_train, rule=self.kind, threshold=self.threshold, rrf_k=self.rrf_k,
                 top_fraction=self.top_fraction, training=training)
        return s + self.offset_


def make(rule: str = "max", nu: float = 0.1, **kwargs) -> FusionScorer:
    """Scorer for one of :data:`FUSION_RULES`."""
    if rule not in FUSION_RULES:
        raise ValueError(f"unknown fusion rule {rule!r}; use one of {FUSION_RULES}")
    return FusionScorer(kind=rule, nu=float(nu), **kwargs)
