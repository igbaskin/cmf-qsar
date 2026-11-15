"""Kernel herding: a few molecules that stand for many, chosen from the kernel alone.

Every set of molecules has a mean field, the mean of its feature maps. For the linear CMF kernel
this is a function of space, and in the language of kernel methods it is the *kernel mean
embedding* of the set (see the manual, "A field is a kernel mean embedding"). A subset stands for
the whole set when its own mean embedding is close to the mean embedding of the set. The distance
between the two is the maximum mean discrepancy (MMD) of :mod:`cmfqsar.mmd`.

**Kernel herding** (Chen, Welling and Smola, UAI 2010) builds such a subset one molecule at a
time. With mu(x) = sum_j p_j K(x, x_j), the target mean embedding evaluated at x, and x_1 .. x_t
the molecules chosen so far, the next one is ::

    x_{t+1} = argmax_x  mu(x) - 1/(t+1) sum_{s<=t} K(x, x_s).

The first term pulls towards molecules typical of the whole set; the second pushes away from
molecules already chosen. The MMD of the herded subset falls as O(1/t) on a finite set, against
O(1/sqrt(t)) for a random subset. ``method="greedy"`` adds the self term K(x, x) / 2 to the second
term, which makes every step the exact greedy minimisation of the MMD with equal weights. For a
cosine-normalised kernel the two rules are the same, because K(x, x) is constant.

Uses in this program:

* **which molecules to make or measure first**, so that a model fitted on them covers the library
  (``cmf pick``);
* **which members of an ensemble to keep**, so that a few conformations or solvation shells give
  the mean field of all of them (:func:`quadrature_weights` then gives the members weights);
* **which molecules to use as Nystroem landmarks** in :mod:`cmfqsar.lowrank`.

Three references come with it, because a new rule must beat the old ones:

``maxmin``
    furthest-first diversity picking, the classical design of chemistry (Kennard and Stone 1969).
    It starts from the most typical molecule and then always takes the molecule furthest from
    everything chosen, so it covers the edges of the set and favours outliers.
``kmedoids``
    k-medoids in field space: the k molecules that minimise the sum of distances from every
    molecule to its nearest chosen one (PAM's BUILD step, then alternating improvement).
``pivoted_cholesky``
    the greedy pivoted Cholesky factorisation: always the molecule worst explained by the span of
    those chosen. It is the usual choice of Nystroem landmarks.

Everything works on a precomputed kernel matrix over the pool of candidates. The cost is one pass
over the matrix for mu and one column per chosen molecule.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = ["METHODS", "distance2_matrix", "herd", "kmedoids", "maxmin", "mmd2_subset",
           "pick", "pivoted_cholesky", "quadrature_weights", "target_weights"]

#: the selection rules :func:`pick` knows
METHODS = ("herding", "greedy", "maxmin", "kmedoids", "pivoted_cholesky", "random")


def target_weights(n: int, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """The weights p of the target mean embedding, non-negative and summing to one."""
    if weights is None:
        return np.full(int(n), 1.0 / int(n))
    p = np.asarray(weights, float).ravel()
    if len(p) != n:
        raise ValueError(f"{len(p)} weights for {n} molecules")
    if np.any(p < 0) or not p.sum() > 0:
        raise ValueError("target weights must be non-negative and not all zero")
    return p / p.sum()


def distance2_matrix(K: np.ndarray) -> np.ndarray:
    """Squared field distances K(i,i) + K(j,j) - 2 K(i,j), never negative."""
    K = np.asarray(K, float)
    d = np.diag(K)
    return np.maximum(d[:, None] + d[None, :] - 2.0 * K, 0.0)


# --------------------------------------------------------------------------- #
# Herding
# --------------------------------------------------------------------------- #

def herd(K: np.ndarray, k: int, weights: Optional[Sequence[float]] = None,
         method: str = "herding", first: Optional[Sequence[int]] = None) -> np.ndarray:
    """The indices of `k` molecules chosen by kernel herding, in the order chosen.

    `K` is the kernel over the pool, `weights` the weights of the target mean embedding (equal by
    default). `method` is ``"herding"`` (Chen et al.) or ``"greedy"`` (the exact greedy MMD step).
    A molecule is never chosen twice. `first` are molecules already chosen (made, measured), which
    the rule continues from; they are not part of the returned indices.
    """
    if method not in ("herding", "greedy"):
        raise ValueError("method is 'herding' or 'greedy'")
    K = np.asarray(K, float)
    n = K.shape[0]
    if K.shape != (n, n):
        raise ValueError("the kernel must be square over the pool")
    k = int(k)
    if not 0 < k <= n:
        raise ValueError(f"cannot choose {k} of {n} molecules")
    mu = K @ target_weights(n, weights)
    diag = np.diag(K)
    acc = np.zeros(n)
    taken = np.zeros(n, bool)
    for j in (() if first is None else np.asarray(first, int).ravel()):
        acc += K[:, int(j)]
        taken[int(j)] = True
    t = int(taken.sum())
    if k > n - t:
        raise ValueError(f"cannot choose {k} more of {n - t} molecules")
    out = []
    for _ in range(k):
        rest = acc + (0.5 * diag if method == "greedy" else 0.0)
        score = mu - rest / (t + 1)
        score[taken] = -np.inf
        j = int(np.argmax(score))
        out.append(j)
        taken[j] = True
        acc += K[:, j]
        t += 1
    return np.asarray(out, int)


def mmd2_subset(K: np.ndarray, idx: Sequence[int], v: Optional[Sequence[float]] = None,
                weights: Optional[Sequence[float]] = None) -> float:
    """||sum_s v_s phi(x_s) - sum_j p_j phi(x_j)||^2: how far the subset is from the whole.

    `v` defaults to equal weights over the subset, `weights` (p) to equal weights over the pool.
    """
    K = np.asarray(K, float)
    idx = np.asarray(idx, int)
    p = target_weights(K.shape[0], weights)
    v = np.full(len(idx), 1.0 / len(idx)) if v is None else np.asarray(v, float).ravel()
    A = K[np.ix_(idx, idx)]
    b = K[idx] @ p
    return float(max(v @ A @ v - 2.0 * v @ b + p @ K @ p, 0.0))


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum w = 1} (Held, Wolfe and Crowder 1974)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1.0
    rho = np.nonzero(u - css / np.arange(1, len(u) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1.0), 0.0)


def quadrature_weights(K: np.ndarray, idx: Sequence[int],
                       weights: Optional[Sequence[float]] = None, simplex: bool = True,
                       max_iter: int = 5000, tol: float = 1e-12) -> np.ndarray:
    """Weights on the chosen molecules that bring their weighted mean closest to the target.

    Minimises mmd2_subset over v. With `simplex` (the default) the weights are kept non-negative
    and summing to one, so the result is still a weighted mean of structures, which is what an
    ensemble needs; the problem is solved by accelerated projected gradient. Without it, the
    unconstrained optimum of Bayesian quadrature, which can give negative weights.
    """
    K = np.asarray(K, float)
    idx = np.asarray(idx, int)
    p = target_weights(K.shape[0], weights)
    A = K[np.ix_(idx, idx)]
    b = K[idx] @ p
    if not simplex:
        jitter = 1e-10 * max(float(np.trace(A)) / len(idx), 1e-300)
        return np.linalg.solve(A + jitter * np.eye(len(idx)), b)
    L = float(np.linalg.eigvalsh(0.5 * (A + A.T))[-1])
    step = 1.0 / max(2.0 * L, 1e-300)
    v = np.full(len(idx), 1.0 / len(idx))
    z, t = v.copy(), 1.0
    for _ in range(int(max_iter)):
        v_new = _project_simplex(z - step * 2.0 * (A @ z - b))
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = v_new + ((t - 1.0) / t_new) * (v_new - v)
        moved = float(np.max(np.abs(v_new - v)))
        v, t = v_new, t_new
        if moved < tol:
            break
    return v


# --------------------------------------------------------------------------- #
# The references
# --------------------------------------------------------------------------- #

def maxmin(K: np.ndarray, k: int, start: Optional[int] = None) -> np.ndarray:
    """Furthest-first picking in field space, from the most typical molecule unless `start`."""
    K = np.asarray(K, float)
    n = K.shape[0]
    k = int(k)
    if not 0 < k <= n:
        raise ValueError(f"cannot choose {k} of {n} molecules")
    D2 = distance2_matrix(K)
    j = int(np.argmax(K.mean(axis=1))) if start is None else int(start)
    out = [j]
    near = D2[j].copy()
    for _ in range(k - 1):
        near[out] = -np.inf
        j = int(np.argmax(near))
        out.append(j)
        near = np.minimum(near, D2[j])
    return np.asarray(out, int)


def kmedoids(K: np.ndarray, k: int, max_iter: int = 100) -> np.ndarray:
    """k medoids in field space: PAM's greedy BUILD, then alternating improvement.

    The cost is the sum over molecules of the field distance to the nearest medoid. Deterministic.
    """
    K = np.asarray(K, float)
    n = K.shape[0]
    k = int(k)
    if not 0 < k <= n:
        raise ValueError(f"cannot choose {k} of {n} molecules")
    D = np.sqrt(distance2_matrix(K))
    med = [int(np.argmin(D.sum(axis=0)))]
    near = D[med[0]].copy()
    for _ in range(k - 1):
        # the gain of adding candidate c: how much it shortens every molecule's nearest distance
        gain = np.maximum(near[:, None] - D, 0.0).sum(axis=0)
        gain[med] = -np.inf
        c = int(np.argmax(gain))
        med.append(c)
        near = np.minimum(near, D[c])
    med = np.asarray(med, int)
    for _ in range(int(max_iter)):
        label = np.argmin(D[:, med], axis=1)
        new = med.copy()
        for c in range(k):
            members = np.flatnonzero(label == c)
            if len(members):
                new[c] = members[int(np.argmin(D[np.ix_(members, members)].sum(axis=0)))]
        if np.array_equal(np.sort(new), np.sort(med)):
            break
        med = new
    return med


def pivoted_cholesky(K: np.ndarray, k: int) -> np.ndarray:
    """Greedy pivoted Cholesky: always the molecule with the largest residual variance."""
    K = np.asarray(K, float)
    n = K.shape[0]
    k = int(k)
    if not 0 < k <= n:
        raise ValueError(f"cannot choose {k} of {n} molecules")
    r = np.diag(K).astype(float).copy()
    L = np.zeros((n, k))
    out = []
    for t in range(k):
        r_masked = r.copy()
        r_masked[out] = -np.inf
        j = int(np.argmax(r_masked))
        out.append(j)
        if r[j] <= 1e-14 * max(float(np.max(np.diag(K))), 1e-300):
            continue        # the rest is already in the span; keep the order, add no column
        col = (K[:, j] - L[:, :t] @ L[j, :t]) / np.sqrt(r[j])
        L[:, t] = col
        r = np.maximum(r - col * col, 0.0)
    return np.asarray(out, int)


def pick(K: np.ndarray, k: int, method: str = "herding",
         weights: Optional[Sequence[float]] = None, seed: int = 0) -> np.ndarray:
    """`k` indices chosen by one of :data:`METHODS`."""
    if method in ("herding", "greedy"):
        return herd(K, k, weights=weights, method=method)
    if method == "maxmin":
        return maxmin(K, k)
    if method == "kmedoids":
        return kmedoids(K, k)
    if method == "pivoted_cholesky":
        return pivoted_cholesky(K, k)
    if method == "random":
        n = np.asarray(K).shape[0]
        return np.sort(np.random.RandomState(int(seed)).choice(n, int(k), replace=False))
    raise ValueError(f"unknown method {method!r}; one of {', '.join(METHODS)}")
