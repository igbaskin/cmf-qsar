"""The molecule vector of a trained device as a kernel view (plan item ND14b).

The QSPR Neural Device (a separate project, ``neurodevice``) reads every atom, bond and
two-bond path of a molecule with small networks that have shared weights, and adds up their
outputs over the molecule. These sums are the *collectors* of the device: one short vector per
molecule. Plan item ND7 stacks a kernel from the bond graph under the field kernels, and plan
item ND8 stacks a kernel from a pretrained model. This module does the same with the collectors
of a device that was trained on the series: a Gaussian kernel on the collector vector is one
more *view* for the mixing of plan item E6 (``cmfqsar.mixing.with_views``).

What the module holds
---------------------
* :func:`read_collectors` reads the file that ``benchmarks/featuriser_device.py`` writes for
  one split: the collectors of every molecule, the activities and the three parts of the split.
* :func:`gaussian_view` makes the kernel. The vectors are centred on the training molecules and
  divided by one number for the whole block, never column by column (a column-by-column
  scaling lowered the accuracy of a network on the CMF vector in plan item ND1). The width of
  the Gaussian is a multiple of the median distance between training molecules. The width is
  the only parameter of the view.
* :func:`kernel_correlation` is the check of plan item EC5: the correlation of two kernels over
  all pairs of molecules, after both are normalised to a diagonal of one. A view whose kernel
  correlates near one with the kernel the model already has adds nothing.
* :func:`ridge_predict`, :func:`choose_width` and :func:`choose_weight` fit the ridge regression
  on a kernel and choose the width of the view and its share of the joint kernel on the
  *validation* molecules.

Why the validation molecules choose
-----------------------------------
The device was trained on the training molecules. The collectors of a training molecule are
therefore fitted to its activity, and the collectors of a molecule the device never saw are not.
The evidence of the mixing of plan item E6 is computed on training molecules, so it can
overstate what the view will do on new molecules. Validation molecules were used by the device
only to stop its training, so their collectors are close to those of new molecules. The weight
of the view and the width of its Gaussian are therefore also chosen on the validation molecules
here (:func:`choose_weight`), and the benchmark reports both ways. The collectors of the
test molecules are never used for any choice.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: the widths tried for the Gaussian, as multiples of the median distance of training molecules
WIDTHS = (0.5, 1.0, 2.0)
#: the ridge parameters tried, as multiples of the mean diagonal of the kernel
RIDGES = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
#: the shares of the view in the joint kernel tried on the validation molecules
SHARES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
#: at most this many molecules are used to find the median distance
MEDIAN_SAMPLE = 2000


def read_collectors(path: str) -> Dict[str, np.ndarray]:
    """The collectors, activities and split that ``featuriser_device.py`` stored for one split.

    Returns a dictionary with ``names``, ``y``, ``collectors`` (molecules, numbers), ``train``,
    ``valid``, ``test`` (indices into the molecules) and ``pred`` (the prediction of the device
    itself, if the file has it).
    """
    with np.load(path, allow_pickle=True) as z:
        out = {"names": np.asarray([str(n) for n in z["names"]]),
               "y": np.asarray(z["y"], float),
               "collectors": np.asarray(z["collectors"], float),
               "train": np.asarray(z["train"], int),
               "valid": np.asarray(z["valid"], int),
               "test": np.asarray(z["test"], int)}
        if "pred" in z.files:
            out["pred"] = np.asarray(z["pred"], float)
    if out["collectors"].ndim != 2 or len(out["collectors"]) != len(out["y"]):
        raise ValueError(f"{path}: the collectors are not one row per molecule")
    return out


def _median_distance(Z: np.ndarray, rows: np.ndarray, seed: int = 0) -> float:
    rows = np.asarray(rows, int)
    if len(rows) > MEDIAN_SAMPLE:
        rows = np.random.default_rng(seed).choice(rows, MEDIAN_SAMPLE, replace=False)
    X = Z[rows]
    sq = np.sum(X ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * X @ X.T, 0.0)
    iu = np.triu_indices(len(rows), 1)
    m = float(np.median(np.sqrt(d2[iu])))
    return m if m > 1e-12 else 1.0


def gaussian_view(vectors: np.ndarray, train: Sequence[int], width: float = 1.0) -> np.ndarray:
    """The Gaussian kernel of a molecule vector, over all molecules, with a diagonal of one.

    ``vectors`` has one row per molecule. The rows are centred on the mean of the training rows
    and divided by one number, the root mean square of the centred training values. The kernel
    is ``exp(-d^2 / (2 s^2))`` with ``s = width`` times the median distance of two training
    molecules.
    """
    V = np.asarray(vectors, float)
    train = np.asarray(train, int)
    Z = V - V[train].mean(axis=0)
    rms = float(np.sqrt(np.mean(Z[train] ** 2)))
    Z = Z / (rms if rms > 1e-12 else 1.0)
    s = float(width) * _median_distance(Z, train)
    sq = np.sum(Z ** 2, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * Z @ Z.T, 0.0)
    return np.exp(-d2 / (2.0 * s * s))


def cosine_normalise(K: np.ndarray) -> np.ndarray:
    """``K(x, z) / sqrt(K(x, x) K(z, z))``."""
    K = np.asarray(K, float)
    d = np.sqrt(np.maximum(np.diag(K), 1e-300))
    return K / d[:, None] / d[None, :]


def kernel_correlation(Ka: np.ndarray, Kb: np.ndarray, rows: Optional[Sequence[int]] = None
                       ) -> float:
    """The correlation of two kernels over the pairs of molecules (plan item EC5).

    Both kernels are cosine-normalised first. The pairs are those of two different molecules
    among ``rows`` (all molecules by default). One number for two complete descriptions: a view
    whose kernel has a correlation near one with the kernel of the model adds no new
    description.
    """
    A, B = np.asarray(Ka, float), np.asarray(Kb, float)
    if A.shape != B.shape:
        raise ValueError(f"kernels of shape {A.shape} and {B.shape}")
    if rows is not None:
        rows = np.asarray(rows, int)
        A, B = A[np.ix_(rows, rows)], B[np.ix_(rows, rows)]
    iu = np.triu_indices(len(A), 1)
    a, b = cosine_normalise(A)[iu], cosine_normalise(B)[iu]
    if np.std(a) < 1e-14 or np.std(b) < 1e-14:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def ridge_predict(K: np.ndarray, y: np.ndarray, train: Sequence[int], rows: Sequence[int],
                  lam: float) -> np.ndarray:
    """Ridge regression on a kernel, with the mean of the training activities as the intercept.

    ``lam`` is a multiple of the mean diagonal of the training block.
    """
    train, rows = np.asarray(train, int), np.asarray(rows, int)
    y = np.asarray(y, float)
    Ktt = K[np.ix_(train, train)]
    ridge = float(lam) * float(np.mean(np.diag(Ktt)))
    mean = float(np.mean(y[train]))
    coef = np.linalg.solve(Ktt + ridge * np.eye(len(train)) + 1e-12 * np.eye(len(train)),
                           y[train] - mean)
    return mean + K[np.ix_(rows, train)] @ coef


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def choose_width(vectors: np.ndarray, y: np.ndarray, train: Sequence[int],
                 valid: Sequence[int], widths: Sequence[float] = WIDTHS,
                 ridges: Sequence[float] = RIDGES) -> Tuple[float, float, float]:
    """The width and the ridge with the lowest error on the validation molecules, for the view
    alone. Returns ``(width, ridge, validation RMSE)``."""
    best = (None, None, np.inf)
    for w in widths:
        K = gaussian_view(vectors, train, w)
        for lam in ridges:
            e = _rmse(ridge_predict(K, y, train, valid, lam), np.asarray(y, float)[valid])
            if e < best[2]:
                best = (float(w), float(lam), e)
    return best


def choose_weight(K_model: np.ndarray, K_view: np.ndarray, y: np.ndarray,
                  train: Sequence[int], valid: Sequence[int],
                  shares: Sequence[float] = SHARES, ridges: Sequence[float] = RIDGES
                  ) -> Tuple[float, float, float]:
    """The share of the view in ``(1 - share) K_model + share K_view`` and the ridge with the
    lowest error on the validation molecules. Both kernels are cosine-normalised first.
    Returns ``(share, ridge, validation RMSE)``."""
    Km, Kv = cosine_normalise(K_model), cosine_normalise(K_view)
    best = (None, None, np.inf)
    for s in shares:
        K = (1.0 - s) * Km + s * Kv
        for lam in ridges:
            e = _rmse(ridge_predict(K, y, train, valid, lam), np.asarray(y, float)[valid])
            if e < best[2]:
                best = (float(s), float(lam), e)
    return best
