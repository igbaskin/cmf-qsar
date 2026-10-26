"""Clustering a set by its fields rather than by its scaffolds.

Every grouped validation in this program splits the molecules by their Bemis-Murcko scaffold:
``chemotype_labels`` groups them, ``grouped_cv_predict`` holds one group out at a time, and the
benchmarks report that figure as the reliable one.  The reasoning is sound -- random folds put a
molecule's own scaffold-mates in the training part and overstate the quality of a model on a
mixed series -- but it rests on
an assumption nobody checks: that the scaffold is what makes two molecules look alike *to the
model*.  The model does not see scaffolds.  It sees fields, and its notion of similarity is the
kernel.

Spectral clustering (Ng, Jordan & Weiss 2002) partitions that similarity directly.  The kernel is
a similarity matrix already, so there is nothing to build: form the normalised Laplacian, take the
eigenvectors of its smallest eigenvalues, normalise their rows to the unit sphere and run k-means
there.  Clusters that are not linearly separable in field space are separable in those
coordinates, which is the purpose of the construction and the reason plain k-means on the
fields would not do.

Two uses, and the second is the one that helps most:

* **as a grouping** -- hold out a cluster of the model's own similarity rather than a cluster of
  the chemist's, which is a harder and fairer test whenever the two disagree;
* **as a check on the usual grouping** -- the adjusted Rand index between the two partitions says
  how far the scaffold split and the field split agree.  Where it is high the scaffold grouping
  is doing what it claims.  Where it is low, one of the two is measuring something else, and
  which one matters: a leave-one-scaffold-out figure is only pessimistic if the scaffolds are
  what the model responds to.

How many clusters is decided by the **eigengap** when it is not given: the number of eigenvalues
of the normalised Laplacian below the largest jump, which is the standard heuristic and is
reported along with the gap so that a weak one can be seen for what it is.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

__all__ = ["SpectralFit", "adjusted_rand", "eigengap", "spectral_clusters"]


def _normalised_laplacian(K: np.ndarray, floor: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """(L_sym, degrees) of the affinity, with negative similarities clipped away.

    A CMF kernel of unnormalised fields can be negative -- an electrostatic field of one sign
    meeting one of the other -- and a negative affinity is not a similarity a graph Laplacian can
    use.  Clipping at `floor` is the correct minimum: it says that two molecules whose fields
    oppose are not similar rather than pretending the relationship is a distance.
    """
    A = np.asarray(K, float).copy()
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    A = np.maximum(A, float(floor))
    d = A.sum(axis=1)
    inv = np.where(d > 1e-12, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
    L = np.eye(A.shape[0]) - (inv[:, None] * A * inv[None, :])
    return L, d


def eigengap(values: Sequence[float], k_max: int = 10) -> Tuple[int, float]:
    """(number of clusters, the size of the gap) from the spectrum of the Laplacian."""
    w = np.sort(np.asarray(values, float))[:max(2, int(k_max) + 1)]
    gaps = np.diff(w)
    if len(gaps) == 0:
        return 1, 0.0
    k = int(np.argmax(gaps)) + 1
    return max(k, 2), float(gaps[k - 1])


@dataclass
class SpectralFit:
    """The partition, the spectrum it came from, and how certain the number of clusters is."""
    labels: np.ndarray
    k: int
    eigenvalues: np.ndarray
    gap: float = float("nan")
    chosen_by_gap: bool = False

    @property
    def sizes(self) -> Dict[int, int]:
        u, c = np.unique(self.labels, return_counts=True)
        return {int(a): int(b) for a, b in zip(u, c)}

    def describe(self) -> str:
        sizes = ", ".join(str(n) for n in sorted(self.sizes.values(), reverse=True))
        how = (f"k = {self.k} from the eigengap ({self.gap:.3f})" if self.chosen_by_gap
               else f"k = {self.k} as asked")
        return f"spectral clusters: {how}; sizes {sizes}"


def spectral_clusters(K: np.ndarray, k: Optional[int] = None, k_max: int = 10,
                      seed: int = 0, floor: float = 0.0) -> SpectralFit:
    """Cluster the molecules of a kernel matrix by their field similarity.

    With `k` given it is used; otherwise the eigengap chooses it, over at most `k_max` clusters.
    """
    from sklearn.cluster import KMeans

    K = np.asarray(K, float)
    n = K.shape[0]
    if n < 3:
        return SpectralFit(labels=np.zeros(n, dtype=int), k=1, eigenvalues=np.zeros(0))
    L, _d = _normalised_laplacian(K, floor=floor)
    w, v = np.linalg.eigh(0.5 * (L + L.T))
    chosen = k is None
    if chosen:
        k, gap = eigengap(w, k_max=min(int(k_max), n - 1))
    else:
        gap = float("nan")
    k = int(max(2, min(int(k), n - 1)))
    U = v[:, :k]
    norms = np.linalg.norm(U, axis=1, keepdims=True)
    U = U / np.maximum(norms, 1e-12)              # Ng-Jordan-Weiss: rows on the unit sphere
    labels = KMeans(n_clusters=k, n_init=10, random_state=int(seed)).fit_predict(U)
    return SpectralFit(labels=np.asarray(labels, int), k=k, eigenvalues=w[:max(k + 1, 2)],
                       gap=float(gap), chosen_by_gap=bool(chosen))


def adjusted_rand(a: Sequence, b: Sequence) -> float:
    """Agreement of two partitions, corrected for chance: 1 identical, 0 as good as random."""
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(np.asarray(a), np.asarray(b)))
