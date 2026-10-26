"""One-class scores on a precomputed kernel: four geometries besides the 1-SVM's ball.

Similarity-based virtual screening fits a description of the actives alone and ranks everything
else by how well it fits (the 2011 papers on the one-class CMF approach).  The program does that
with a one-class support-vector machine, which encloses the actives in a ball in the field
Hilbert space.  A ball is one choice among several, and the others are cheap here because the
kernel is already computed:

``mean``
    distance to the centroid of the actives, ``||phi(x) - mu||^2``.  With a normalised kernel
    this is one minus the mean field similarity to the training molecules: the similarity
    ensemble of the screening literature, with no hyper-parameter and nothing fitted.  It is the
    baseline the others have to beat.
``knn``
    distance to the k-th nearest active in the field Hilbert space, the classical similarity
    search, with the number of neighbours as its only parameter.
``gp``
    a Gaussian process regressed on a constant target of one over the actives (Kemmler et al.
    2013): the predictive mean falls off both where a molecule is unlike the actives and where
    the model has nothing to say, and ``score="mean_over_sd"`` divides by the posterior standard
    deviation to use the second effect explicitly.
``kpca``
    distance from the principal subspace of the actives (Hoffmann 2007).  A congeneric series
    lies on something much flatter than a ball, and a subspace of a few components can describe
    it far better than a radius can.

Every scorer returns a *decision value* with the same convention as the rest of the program: at
or above zero means inside the description.  The offset is placed so that a fraction `nu` of the
training molecules falls outside, which is what nu means for the 1-SVM, so the methods can be
compared without re-tuning anything.

``mean`` and ``gp`` are linear in the kernel row, ``score(x) = sum_j a_j K(x, x_j) + b``, so the
co-field, the training-molecule contributions and the atom contributions carry over unchanged.
``knn`` and ``kpca`` are not: they are a minimum and a quadratic form, so no co-field exists for
them, and :attr:`OneClassScorer.is_linear` says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Optional

import numpy as np

#: method keys understood by :func:`make`, in the order they are offered
ONECLASS_METHODS = ("ocsvm", "ocmean", "ocknn", "ocgp", "ockpca")

LABELS = {
    "ocsvm": "One-class SVM (1-SVM)",
    "ocmean": "Distance to the centre of the actives (similarity ensemble)",
    "ocknn": "Distance to the k-th nearest active",
    "ocgp": "One-class Gaussian process",
    "ockpca": "Distance from the principal subspace (kernel PCA)",
}


@dataclass
class OneClassScorer:
    """A description of one class, fitted on the kernel of its members alone."""
    kind: str = "ocmean"
    nu: float = 0.1                      # fraction of the training molecules left outside
    k: int = 5                           # ocknn: neighbours
    lam: float = 0.1                     # ocgp: ridge on the constant target
    components: int = 5                  # ockpca: dimension of the subspace
    score: str = "mean"                  # ocgp: "mean" or "mean_over_sd"

    # fitted state
    coef_: Optional[np.ndarray] = None   # linear kinds: dual coefficients
    offset_: float = 0.0
    diag_: Optional[np.ndarray] = None   # training self-kernels
    K_: Optional[np.ndarray] = None      # ocknn needs the training kernel rows
    L_: Optional[np.ndarray] = None      # ocgp: chol(K + lam I)
    vecs_: Optional[np.ndarray] = None   # ockpca: eigenvectors (n, q), scaled
    kmean_: Optional[np.ndarray] = None  # ockpca: column means of the training kernel
    ktotal_: float = 0.0                 # ockpca / mean: grand mean of the training kernel

    # ------------------------------------------------------------------ fit
    @property
    def is_linear(self) -> bool:
        """True when the score is a kernel expansion, so the decomposition rules still apply."""
        return self.kind in ("ocmean", "ocgp")

    @property
    def label(self) -> str:
        """How this description is named in a message to the user."""
        return LABELS.get(self.kind, self.kind)

    @property
    def why_not_linear(self) -> str:
        """Why no co-field exists, in a clause that can be dropped into an error message."""
        return ("it is a minimum over neighbours" if self.kind == "ocknn"
                else "it is a quadratic form" if self.kind == "ockpca"
                else "it is not a sum over the training molecules")

    def fit(self, K: np.ndarray) -> "OneClassScorer":
        K = np.asarray(K, float)
        n = K.shape[0]
        if n < 2:
            raise ValueError("a one-class description needs at least two molecules")
        self.diag_ = np.diag(K).copy()
        self.ktotal_ = float(K.mean())
        if self.kind == "ocmean":
            self.coef_ = np.full(n, 2.0 / n)
        elif self.kind == "ocgp":
            lam = max(float(self.lam), 1e-12)
            self.L_ = np.linalg.cholesky(K + lam * np.eye(n))
            self.coef_ = np.linalg.solve(K + lam * np.eye(n), np.ones(n))
        elif self.kind == "ocknn":
            self.K_ = K
            self.k = int(max(1, min(self.k, n - 1)))
        elif self.kind == "ockpca":
            one = np.full((n, n), 1.0 / n)
            Kc = K - one @ K - K @ one + one @ K @ one
            w, V = np.linalg.eigh(Kc)
            order = np.argsort(w)[::-1]
            q = int(max(1, min(self.components, n - 1)))
            w, V = w[order][:q], V[:, order][:, :q]
            keep = w > max(w[0], 1e-300) * 1e-12
            self.vecs_ = V[:, keep] / np.sqrt(np.maximum(w[keep], 1e-300))
            self.kmean_ = K.mean(axis=1)
        else:
            raise ValueError(f"unknown one-class method {self.kind!r}; "
                             f"use one of {ONECLASS_METHODS}")
        self.offset_ = 0.0
        inside = self.decision(K, self.diag_, training=True)
        self.offset_ = -float(np.quantile(inside, np.clip(self.nu, 0.0, 0.99)))
        return self

    # ------------------------------------------------------------- scoring
    def decision(self, K_test_train: np.ndarray, k_test_self: np.ndarray,
                 training: bool = False) -> np.ndarray:
        """Decision value of new molecules; at or above zero is inside the description.

        `training` says that the rows are the training molecules themselves, which only the
        nearest-neighbour score needs to know: a molecule is not its own neighbour.
        """
        Kt = np.atleast_2d(np.asarray(K_test_train, float))
        d = np.asarray(k_test_self, float)
        if self.kind == "ocmean":
            # -||phi(x) - mu||^2, with the constant that depends only on the training set kept
            # so that the value is comparable across molecules of different self-kernel
            s = Kt @ self.coef_ - d - self.ktotal_
        elif self.kind == "ocgp":
            s = Kt @ self.coef_
            if self.score == "mean_over_sd":
                v = np.maximum(d - np.sum(
                    np.linalg.solve(self.L_, Kt.T) ** 2, axis=0), 1e-12)
                s = s / np.sqrt(v)
        elif self.kind == "ocknn":
            d2 = d[:, None] + self.diag_[None, :] - 2.0 * Kt
            if training:
                np.fill_diagonal(d2, np.inf)     # a molecule is not its own neighbour
            part = np.partition(d2, self.k - 1, axis=1)[:, self.k - 1]
            s = -part
        elif self.kind == "ockpca":
            # squared distance from the principal subspace, in the centred feature space
            kc = Kt - self.kmean_[None, :] - Kt.mean(axis=1)[:, None] + self.ktotal_
            proj = kc @ self.vecs_
            centred_self = d - 2.0 * Kt.mean(axis=1) + self.ktotal_
            s = -(centred_self - np.sum(proj * proj, axis=1))
        else:
            raise ValueError(f"unknown one-class method {self.kind!r}")
        return s + self.offset_

    # the rest of the program uses these two
    @property
    def coef(self) -> np.ndarray:
        if not self.is_linear:
            raise ValueError(f"{LABELS[self.kind]} is not a kernel expansion, so it has no dual "
                             f"coefficients, no co-field and no atom contributions")
        return self.coef_

    @property
    def bias(self) -> float:
        """Constant of the kernel expansion.

        For the centroid distance it absorbs the molecule's own self-kernel, which is a constant
        only when the kernels are normalised -- and a one-class model normalises them, because
        an unnormalised one-class score ranks by molecular size.
        """
        if self.kind == "ocmean":
            return self.offset_ - self.ktotal_ - float(np.mean(self.diag_))
        return self.offset_


def make(method: str, nu: float = 0.1, **kwargs) -> OneClassScorer:
    """Scorer for one of :data:`ONECLASS_METHODS` other than ``ocsvm``."""
    if method == "ocsvm":
        raise ValueError("the one-class SVM is fitted by scikit-learn, not by this module")
    return OneClassScorer(kind=method, nu=float(nu), **kwargs)


def enrichment_factor(y_true: np.ndarray, score: np.ndarray, fraction: float = 0.01) -> float:
    """How many times more actives the top `fraction` of the ranking holds than chance."""
    y = (np.asarray(y_true, float) > 0).astype(float)
    s = np.asarray(score, float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n = len(y)
    if n == 0 or y.sum() == 0:
        return float("nan")
    take = max(1, int(round(fraction * n)))
    top = np.argsort(-s)[:take]
    return float((y[top].mean()) / (y.mean()))
