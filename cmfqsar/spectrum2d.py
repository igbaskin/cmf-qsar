"""A spectrum with two axes as a field on a plane: the kernel and the co-map (plan item SP4).

Some measurements give an intensity over two variables.  A fluorescence excitation-emission
matrix is an intensity over the excitation and the emission wavelength.  A 1H-13C HSQC
spectrum is a list of cross-peaks, each at a proton shift and a carbon shift.  An LC-MS run is
a list of ions, each at a retention time and a mass.  This module treats such a spectrum as
`cmfqsar/spectrum.py` treats a spectrum with one axis: as a field, here on a plane, with a
kernel between two fields and a weight over the plane that a fitted model gives back.

Two forms
---------
A **peak list** holds positions ``(x_i, y_i)`` and intensities ``a_i`` (:class:`PeakList`).  A
**map** is an array of intensities on a grid of the two axes, ``map[i, j]`` at ``x[i]`` and
``y[j]``; a set of maps is an array with the samples first.

The kernel
----------
Every peak is replaced by a Gaussian with the width ``sigma_x`` along the first axis and
``sigma_y`` along the second.  The two axes have different units, so each has its own width.
The overlap of two broadened peak lists has a closed form, ::

    k(A, B) = SUM_ij a_i b_j exp( -(x_i - x_j)^2 / (4 sigma_x^2) - (y_i - y_j)^2 / (4 sigma_y^2) ),

which is the kernel of `cmfqsar/spectrum.py` with one more factor (:func:`peak_overlap`).  Its
cost is the number of peaks of the one list times that of the other, and no grid enters.  The
usual treatment of such data puts the intensities into bins: two peaks on the two sides of a
bin border then count as different, and two peaks in one bin as the same.  The width does what
the bin size does without a border.

A map enters as the same sum over its grid points.  For a set of maps it is cheaper to broaden
every map on its grid (:func:`smoothed`) and to take dot products of the flattened maps
(:func:`rows`, :func:`map_kernel`); :func:`rasterise` puts a peak list on a grid in the same
way, so that peak lists and maps can be compared.  With a grid step well below the width the
two routes agree, because the integral of the product of two normalised Gaussians of width
``s`` at a distance ``d`` is ``exp(-d^2 / (4 s^2)) / (2 s sqrt(pi))`` along each axis.

Missing values
--------------
A measured map has places that carry no information about the sample.  In a fluorescence map
the excitation light itself is scattered into the detector where the emission wavelength
equals the excitation wavelength, and again at twice that wavelength (:func:`scatter_mask`).
Such values are set to NaN.  :func:`smoothed` broadens a map with missing values by a
normalised convolution: every value is the weighted mean of the measured values around it,
and a place with no measured value within reach is zero.

The co-map
----------
A kernel ridge model on the kernel of the flattened maps is linear in the broadened map, so
its weight is a map over the two axes (:func:`co_map`): positive where intensity raises the
predicted property, or the score of a class, and negative where it lowers it.
:func:`strongest_places` lists the places where a map is largest in size.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import spectrum as SP


@dataclass
class PeakList:
    """Peaks on a plane: positions along the two axes and one intensity for each peak."""
    x: np.ndarray
    y: np.ndarray
    w: np.ndarray
    name: str = ""

    def __post_init__(self):
        self.x = np.atleast_1d(np.asarray(self.x, float))
        self.y = np.atleast_1d(np.asarray(self.y, float))
        self.w = np.atleast_1d(np.asarray(self.w, float))
        if not (self.x.ndim == 1 and self.x.shape == self.y.shape == self.w.shape):
            raise ValueError("a peak list needs one x, one y and one intensity for each peak")
        if not (np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.y))
                and np.all(np.isfinite(self.w))):
            raise ValueError("the positions and the intensities have to be finite numbers")

    def __len__(self) -> int:
        return len(self.x)


def _widths(sigma_x: float, sigma_y: float) -> Tuple[float, float]:
    if not (sigma_x > 0 and sigma_y > 0):
        raise ValueError("both widths have to be positive")
    return float(sigma_x), float(sigma_y)


def peak_overlap(a: PeakList, b: PeakList, sigma_x: float, sigma_y: float) -> float:
    """The kernel of two peak lists, not normalised: the closed form of the module text."""
    sx, sy = _widths(sigma_x, sigma_y)
    if len(a) == 0 or len(b) == 0:
        return 0.0
    dx = a.x[:, None] - b.x[None, :]
    dy = a.y[:, None] - b.y[None, :]
    G = np.exp(-dx * dx / (4.0 * sx * sx) - dy * dy / (4.0 * sy * sy))
    return float(a.w @ G @ b.w)


def peak_kernel(lists: Sequence[PeakList], sigma_x: float, sigma_y: float,
                others: Optional[Sequence[PeakList]] = None,
                normalize: bool = True) -> np.ndarray:
    """The kernel of a set of peak lists, in the cosine form by default.

    Rows are `others` (or `lists` when `others` is not given) and columns are `lists`.  A
    list without intensity has the kernel zero with every list.
    """
    rows_ = lists if others is None else others
    K = np.array([[peak_overlap(r, c, sigma_x, sigma_y) for c in lists] for r in rows_])
    if others is None:
        # k(A, B) and k(B, A) are summed in another order and can differ in the last digit;
        # the kernel of one set is made symmetric to the digit
        K = 0.5 * (K + K.T)
    if not normalize:
        return K
    dc = np.array([peak_overlap(c, c, sigma_x, sigma_y) for c in lists])
    dr = dc if others is None else np.array([peak_overlap(r, r, sigma_x, sigma_y)
                                             for r in rows_])
    norm = np.sqrt(np.maximum(dr[:, None] * dc[None, :], 0.0))
    return K / np.where(norm > 1e-300, norm, 1.0)


def rasterise(peaks: PeakList, x: np.ndarray, y: np.ndarray, sigma_x: float,
              sigma_y: float) -> np.ndarray:
    """A peak list on the grid ``x`` by ``y``, every peak a normalised Gaussian.

    The dot product of two such maps, times the two grid steps, is :func:`peak_overlap`
    divided by ``4 pi sigma_x sigma_y`` when the grid steps are well below the widths and
    the grid reaches a few widths beyond the peaks.
    """
    sx, sy = _widths(sigma_x, sigma_y)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(peaks) == 0:
        return np.zeros((len(x), len(y)))
    gx = np.exp(-0.5 * ((x[None, :] - peaks.x[:, None]) / sx) ** 2) / (sx * np.sqrt(2 * np.pi))
    gy = np.exp(-0.5 * ((y[None, :] - peaks.y[:, None]) / sy) ** 2) / (sy * np.sqrt(2 * np.pi))
    return np.einsum("p,pi,pj->ij", peaks.w, gx, gy)


def scatter_mask(excitation: np.ndarray, emission: np.ndarray, width: float,
                 orders: Sequence[int] = (1, 2), below: bool = True) -> np.ndarray:
    """The places of a fluorescence map that hold scattered excitation light, as True.

    The map has the excitation wavelengths along its first axis and the emission wavelengths
    along its second.  Scattered light is found where the emission wavelength is within
    `width` of the excitation wavelength, and of twice that wavelength for the second order
    (`orders`).  With `below`, the places with an emission wavelength below the excitation
    wavelength are marked too: a molecule does not emit light of a shorter wavelength than
    it absorbed.
    """
    ex = np.asarray(excitation, float)[:, None]
    em = np.asarray(emission, float)[None, :]
    mask = np.zeros((ex.shape[0], em.shape[1]), dtype=bool)
    for order in orders:
        mask |= np.abs(em - order * ex) <= width
    if below:
        mask |= em < ex
    return mask


def smoothed(maps: np.ndarray, step_x: float, step_y: float, sigma_x: float,
             sigma_y: float) -> np.ndarray:
    """Maps on one even grid, each broadened by a Gaussian with one width for each axis.

    `maps` is one map or a set with the samples first.  A width of zero leaves that axis as
    it is.  A NaN is a missing value: the result is the weighted mean of the measured values
    around a place (a normalised convolution), and zero where no measured value is within
    reach.  The first and the last value along an axis are continued beyond its ends.
    """
    from scipy.ndimage import gaussian_filter1d

    M = np.asarray(maps, float)
    if M.ndim not in (2, 3):
        raise ValueError("one map (two axes) or a set of maps (samples first) is expected")
    if sigma_x < 0 or sigma_y < 0 or not (step_x > 0 and step_y > 0):
        raise ValueError("no width below zero, and the steps have to be positive")
    missing = ~np.isfinite(M)
    out = np.where(missing, 0.0, M)
    weight = (~missing).astype(float) if missing.any() else None
    for axis, sigma, step in ((-2, sigma_x, step_x), (-1, sigma_y, step_y)):
        if sigma > 0:
            out = gaussian_filter1d(out, sigma / step, axis=axis, mode="nearest",
                                    truncate=SP.CUT)
            if weight is not None:
                weight = gaussian_filter1d(weight, sigma / step, axis=axis, mode="nearest",
                                           truncate=SP.CUT)
    if weight is None:
        return out
    return np.where(weight > 1e-6, out / np.maximum(weight, 1e-6), 0.0)


def binned(maps: np.ndarray, bins_x: int, bins_y: int) -> np.ndarray:
    """Maps with `bins_x` neighbouring points of the first axis and `bins_y` of the second
    put into one bin: the mean of the measured values of the bin, and zero for a bin without
    one.  Points that do not fill a last bin are left out.  This is the grid on which the
    reference methods work."""
    M = np.asarray(maps, float)
    if bins_x < 1 or bins_y < 1:
        raise ValueError("a bin holds at least one point")
    nx, ny = (M.shape[-2] // bins_x) * bins_x, (M.shape[-1] // bins_y) * bins_y
    if nx == 0 or ny == 0:
        raise ValueError("the bins are larger than the map")
    M = M[..., :nx, :ny]
    B = M.reshape(M.shape[:-2] + (nx // bins_x, bins_x, ny // bins_y, bins_y))
    ok = np.isfinite(B)
    total = np.where(ok, B, 0.0).sum(axis=(-3, -1))
    count = ok.sum(axis=(-3, -1))
    return np.where(count > 0, total / np.maximum(count, 1), 0.0)


def rows(maps: np.ndarray) -> np.ndarray:
    """A set of maps as the rows of a description: one flattened map per sample, with a
    missing value as zero."""
    M = np.asarray(maps, float)
    if M.ndim != 3:
        raise ValueError("a set of maps with the samples first is expected")
    return np.where(np.isfinite(M), M, 0.0).reshape(len(M), -1)


def map_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
               normalize: bool = True) -> np.ndarray:
    """The linear kernel of sets of maps (samples first, or already :func:`rows`), in the
    cosine form by default.  Rows are `test` (or `train`) and columns are `train`."""
    F = rows(train) if np.ndim(train) == 3 else np.asarray(train, float)
    G = None if test is None else (rows(test) if np.ndim(test) == 3
                                   else np.asarray(test, float))
    return SP.spectrum_kernel(F, G, normalize)


def co_map(maps: np.ndarray, alpha: np.ndarray, normalize: bool = True) -> np.ndarray:
    """The weight over the two axes of a kernel model fitted on a set of maps.

    `alpha` holds the dual coefficients of the training maps.  With the cosine kernel the
    model predicts ``b + SUM w * z`` for a new map with the normalised values ``z``; this
    function returns ``w`` in the shape of one map (see :func:`cmfqsar.spectrum.co_spectrum`).
    """
    M = np.asarray(maps, float)
    if M.ndim != 3:
        raise ValueError("a set of maps with the samples first is expected")
    return SP.co_spectrum(rows(M), alpha, normalize).reshape(M.shape[1:])


def strongest_places(x: np.ndarray, y: np.ndarray, M: np.ndarray, n: int = 6,
                     separation: Tuple[float, float] = (0.0, 0.0)) -> List[dict]:
    """The `n` places where a map is largest in size, each with its two positions and its
    value.  A place is left out when it is within `separation` of a place already chosen
    along both axes."""
    x, y, M = np.asarray(x, float), np.asarray(y, float), np.asarray(M, float)
    if M.shape != (len(x), len(y)):
        raise ValueError(f"a map of shape {M.shape} for {len(x)} by {len(y)} grid points")
    order = np.argsort(-np.abs(np.where(np.isfinite(M), M, 0.0)), axis=None)
    chosen: List[dict] = []
    for flat in order:
        i, j = np.unravel_index(int(flat), M.shape)
        if any(abs(x[i] - c["x"]) <= separation[0] and abs(y[j] - c["y"]) <= separation[1]
               for c in chosen):
            continue
        chosen.append({"x": float(x[i]), "y": float(y[j]), "value": float(M[i, j])})
        if len(chosen) == n:
            break
    return chosen
