"""A spectrum as a one-dimensional field: the kernel and the co-spectrum (plan item SP1).

A spectrum is a function of one variable: an intensity over a wavenumber, over a diffraction
angle, over a binding energy or over a chemical shift.  This module treats it as the rest of
the program treats a molecule: as a field, here a field on a line, with a kernel between two
fields and a weight over the line that a fitted model gives back.

Two forms of a spectrum
-----------------------
A **stick spectrum** is a list of positions and intensities.  A quantum program writes a
vibrational spectrum in this form (one stick per normal mode), and a calculated powder
diffraction pattern has this form (one stick per reflection).  A **measured curve** is a list of
intensities on the grid of an instrument.  :class:`Spectrum` holds both.

The kernel
----------
Every stick is replaced by a Gaussian of standard deviation ``sigma``, the *width*.  The
overlap of two broadened spectra with sticks ``(x_i, a_i)`` and ``(y_j, b_j)`` has a closed
form, ::

    k(A, B) = SUM_ij a_i b_j exp( -(x_i - y_j)^2 / (4 sigma^2) ),

which is the steric kernel of the molecular fields in one dimension: atoms become sticks, and
the atom's weight becomes the intensity.  A measured curve enters as the same sum over its grid
points, each point with the weight ``y * dx``.  :func:`stick_overlap` computes this sum
directly.  For a set of spectra it is cheaper to put every broadened spectrum on one grid
(:func:`feature_matrix`) and to take dot products of the rows; with a grid step well below the
width the two agree, because ::

    INT g(x - x_i) g(x - y_j) dx = exp( -(x_i - y_j)^2 / (4 sigma^2) ) / (2 sigma sqrt(pi))

for the normalised Gaussian ``g`` of standard deviation ``sigma``.  The default kernel is the
cosine form (:func:`spectrum_kernel`): every spectrum is divided by its own norm, so the total
intensity, which depends on the amount of sample and on the instrument, does not enter.

The width says how far two peaks may be apart and still count as the same peak.  It is a
hyper-parameter, chosen by cross-validation, and :func:`multi_width_kernel` sums the kernels of
several widths.

The co-spectrum
---------------
A kernel ridge model on the cosine kernel is linear in the normalised broadened spectrum
``z(x)``: its prediction is ``b + INT w(x) z(x) dx``.  The weight ``w`` is the **co-spectrum**
(:func:`co_spectrum`), the one-dimensional analogue of the co-field: a curve over the axis that
is positive where intensity raises the predicted property and negative where it lowers it.

Readers
-------
:func:`read_sticks` and :func:`read_measured` read two-column text (the reader of
``scattering.py`` with the axis left to the caller).  :func:`orca_sticks` takes the infrared
spectrum from an ORCA frequency job.  :func:`powder_sticks` computes the powder diffraction
pattern of a crystal with pymatgen, for X-rays or for neutrons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: the axes this module names, with their units.  The kernel does not depend on the axis: the
#: axis only says what the numbers of ``x`` and the width ``sigma`` mean.
AXES: Dict[str, str] = {
    "wavenumber": "cm^-1",
    "two_theta": "degree",
    "q": "1/A",
    "d_spacing": "A",
    "r": "A",
    "binding_energy": "eV",
    "shift": "ppm",
}

KINDS = ("sticks", "curve")

#: a broadened stick is cut where its Gaussian has fallen to exp(-CUT^2 / 2) of its height
CUT = 6.0


@dataclass
class Spectrum:
    """One spectrum: positions ``x`` and intensities ``y``, sorted by position.

    ``kind`` is ``"sticks"`` for a list of lines and ``"curve"`` for intensities on the grid of
    an instrument.  ``axis`` is one of :data:`AXES` or any other name; it is carried for the
    labels and is not used in a calculation.
    """
    x: np.ndarray
    y: np.ndarray
    kind: str = "sticks"
    axis: str = "wavenumber"
    name: str = ""

    def __post_init__(self):
        x = np.atleast_1d(np.asarray(self.x, float))
        y = np.atleast_1d(np.asarray(self.y, float))
        if x.ndim != 1 or x.shape != y.shape:
            raise ValueError(f"a spectrum needs as many intensities as positions; got "
                             f"{x.shape} positions and {y.shape} intensities")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError(f"the spectrum '{self.name}' holds a value that is not a number")
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}; got '{self.kind}'")
        if self.kind == "curve" and len(x) < 2:
            raise ValueError("a measured curve needs at least two points")
        order = np.argsort(x, kind="stable")
        self.x, self.y = x[order], y[order]

    def __len__(self) -> int:
        return len(self.x)

    @property
    def unit(self) -> str:
        """The unit of the axis, or an empty string for an axis this module does not name."""
        return AXES.get(self.axis, "")

    def lines(self) -> Tuple[np.ndarray, np.ndarray]:
        """The spectrum as sticks: positions and weights.

        A stick spectrum is returned as it is.  A measured curve gives one stick per grid point
        with the weight ``y * dx``, where ``dx`` is the part of the axis the point stands for,
        so that the sum over the sticks is the integral of the curve.
        """
        if self.kind == "sticks":
            return self.x, self.y
        edges = np.concatenate([[self.x[0]], 0.5 * (self.x[1:] + self.x[:-1]), [self.x[-1]]])
        return self.x, self.y * np.diff(edges)

    def within(self, x_min: float, x_max: float) -> "Spectrum":
        """The part of the spectrum between two positions of the axis."""
        keep = (self.x >= x_min) & (self.x <= x_max)
        return Spectrum(self.x[keep], self.y[keep], self.kind, self.axis, self.name)


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_sticks(path: str, axis: str = "wavenumber") -> Spectrum:
    """A stick spectrum from two-column text: a position and an intensity on every line.

    Lines that begin with ``#`` or ``;`` are comments, and columns after the second are
    ignored, as in :func:`cmfqsar.scattering.read_curve`.
    """
    from .scattering import read_curve

    rec = read_curve(path)
    return Spectrum(rec["x"], rec["y"], "sticks", axis, rec["name"])


def read_measured(path: str, axis: str = "wavenumber") -> Spectrum:
    """A measured curve from two-column text, on the grid of the instrument that wrote it."""
    from .scattering import read_curve

    rec = read_curve(path)
    return Spectrum(rec["x"], rec["y"], "curve", axis, rec["name"])


def orca_sticks(source, scale: float = 1.0, name: str = "") -> Spectrum:
    """The infrared spectrum of an ORCA frequency job as sticks.

    `source` is the path of an ORCA output file or a parsed :class:`cmfqsar.orca.OrcaResult`.
    One stick per normal mode with a positive frequency: its position is the harmonic
    wavenumber in cm^-1 times `scale` (the scaling factor of the level of theory), and its
    intensity is the infrared intensity in km/mol.  The modes of translation and rotation
    (frequency zero) and imaginary modes (negative frequency) are left out.
    """
    from . import orca

    if isinstance(source, str):
        with open(source, encoding="utf-8", errors="replace") as fh:
            result = orca.parse_orca_output(fh.read())
        name = name or source
    else:
        result = source
    if result.frequencies is None:
        raise ValueError(f"no vibrational frequencies were found in {name or 'the ORCA result'}")
    if result.ir_intensities is None:
        raise ValueError(f"no infrared intensities were found in {name or 'the ORCA result'}")
    nu = np.asarray(result.frequencies, float)
    inten = np.asarray(result.ir_intensities, float)
    keep = nu > 1e-6
    return Spectrum(nu[keep] * float(scale), inten[keep], "sticks", "wavenumber", name)


def powder_sticks(crystal, source: str = "xray", wavelength="CuKa",
                  two_theta: Tuple[float, float] = (5.0, 90.0), name: str = "") -> Spectrum:
    """The calculated powder diffraction pattern of a crystal as sticks over 2-theta.

    `crystal` is an ASE ``Atoms`` object or a pymatgen ``Structure``.  `source` is ``"xray"``
    or ``"neutron"``.  `wavelength` is in angstrom, or the name of an X-ray line that pymatgen
    knows (``"CuKa"``, ``"MoKa"``).  One stick per reflection (reflections at the same angle
    are merged by pymatgen): its position is the diffraction angle 2-theta in degrees and its
    intensity is the calculated one, with the multiplicity and the Lorentz factor, not scaled
    to 100.  The pattern does not depend on the choice of the cell or on its orientation.
    """
    try:
        from pymatgen.analysis.diffraction.neutron import NDCalculator
        from pymatgen.analysis.diffraction.xrd import WAVELENGTHS, XRDCalculator
        from pymatgen.core import Structure
    except ImportError as exc:                                   # pragma: no cover
        raise ImportError("a powder pattern needs the package pymatgen") from exc
    if source not in ("xray", "neutron"):
        raise ValueError(f"source must be 'xray' or 'neutron'; got '{source}'")
    if isinstance(crystal, Structure):
        structure = crystal
    else:
        structure = Structure(np.asarray(crystal.cell.array, float),
                              crystal.get_chemical_symbols(),
                              np.asarray(crystal.get_positions(), float),
                              coords_are_cartesian=True)
    if source == "xray":
        calc = XRDCalculator(wavelength=wavelength)
    else:
        lam = WAVELENGTHS[wavelength] if isinstance(wavelength, str) else float(wavelength)
        calc = NDCalculator(wavelength=lam)
    pattern = calc.get_pattern(structure, scaled=False,
                               two_theta_range=(float(two_theta[0]), float(two_theta[1])))
    return Spectrum(np.asarray(pattern.x, float), np.asarray(pattern.y, float), "sticks",
                    "two_theta", name)


# --------------------------------------------------------------------------- #
# The kernel in closed form
# --------------------------------------------------------------------------- #
def stick_overlap(a: Spectrum, b: Spectrum, sigma: float, power: float = 1.0) -> float:
    """``SUM_ij a_i b_j exp(-(x_i - y_j)^2 / (4 sigma^2))``, the overlap of two spectra whose
    sticks are broadened to Gaussians of standard deviation `sigma`.

    The sum runs over all pairs of sticks, so it is the exact value and costs the product of
    the two numbers of sticks.  `power` is applied to the weights first (see :func:`broaden`).
    """
    if sigma <= 0:
        raise ValueError("the width sigma must be positive")
    xa, wa = a.lines()
    xb, wb = b.lines()
    if len(xa) == 0 or len(xb) == 0:
        return 0.0
    wa, wb = _powered(wa, power), _powered(wb, power)
    d = xa[:, None] - xb[None, :]
    return float(wa @ np.exp(-d * d / (4.0 * sigma * sigma)) @ wb)


def exact_kernel(spectra: Sequence[Spectrum], sigma: float,
                 others: Optional[Sequence[Spectrum]] = None, power: float = 1.0,
                 normalize: bool = True) -> np.ndarray:
    """The kernel matrix of a set of spectra from :func:`stick_overlap`, pair by pair.

    Rows are `others` (or `spectra` when `others` is not given) and columns are `spectra`.
    With `normalize` every element is divided by the square roots of the two self-overlaps
    (the cosine form).  This is the reference that :func:`spectrum_kernel` is tested against;
    for more than a few hundred spectra use :func:`feature_matrix` and :func:`spectrum_kernel`.
    """
    rows = spectra if others is None else others
    K = np.array([[stick_overlap(r, c, sigma, power) for c in spectra] for r in rows])
    if normalize:
        nr = np.array([stick_overlap(r, r, sigma, power) for r in rows])
        nc = np.array([stick_overlap(c, c, sigma, power) for c in spectra])
        K = K / np.sqrt(np.maximum(nr[:, None] * nc[None, :], 1e-300))
    return K


def _powered(w: np.ndarray, power: float) -> np.ndarray:
    """``sign(w) |w|^power``: 1 leaves the weights as they are, 0.5 takes the square root."""
    if power == 1.0:
        return np.asarray(w, float)
    w = np.asarray(w, float)
    return np.sign(w) * np.abs(w) ** float(power)


# --------------------------------------------------------------------------- #
# Spectra on one grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Grid:
    """An even grid over a part of the axis, from `x_min` to `x_max` in steps of `step`."""
    x_min: float
    x_max: float
    step: float

    def __post_init__(self):
        if not self.step > 0 or not self.x_max > self.x_min:
            raise ValueError("a grid needs a positive step and x_max above x_min")

    @property
    def x(self) -> np.ndarray:
        n = int(np.floor((self.x_max - self.x_min) / self.step + 1e-9)) + 1
        return self.x_min + self.step * np.arange(n)

    @classmethod
    def for_width(cls, x_min: float, x_max: float, sigma: float,
                  points_per_sigma: float = 2.5) -> "Grid":
        """A grid fine enough for the width `sigma`: `points_per_sigma` points in one sigma.
        With 2.5 points the dot product of two rows differs from the closed form by less
        than one part in a million."""
        return cls(float(x_min), float(x_max), float(sigma) / float(points_per_sigma))


def broaden(spectrum: Spectrum, grid: Grid, sigma: float, power: float = 1.0) -> np.ndarray:
    """The spectrum on a grid, every stick replaced by a normalised Gaussian of width `sigma`.

    A measured curve is treated as sticks at its own grid points (:meth:`Spectrum.lines`).
    With ``sigma = 0`` a measured curve is only interpolated onto the grid; sticks need a
    positive width.  `power` is applied to the weights before the broadening: 1 keeps the
    intensities, 0.5 takes their square root, which lowers the share of the strongest line.
    Sticks outside the grid add only the part of their Gaussian that reaches into it.
    """
    gx = grid.x
    out = np.zeros(len(gx))
    if sigma == 0:
        if spectrum.kind != "curve":
            raise ValueError("sticks need a positive width; sigma = 0 is for a measured curve")
        y = _powered(spectrum.y, power)
        vals = np.interp(gx, spectrum.x, y, left=0.0, right=0.0)
        return np.where((gx >= spectrum.x[0]) & (gx <= spectrum.x[-1]), vals, 0.0)
    if sigma < 0:
        raise ValueError("the width sigma must not be negative")
    x, w = spectrum.lines()
    w = _powered(w, power)
    reach = CUT * sigma
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    lo = np.searchsorted(gx, x - reach, side="left")
    hi = np.searchsorted(gx, x + reach, side="right")
    for xi, wi, a, b in zip(x, w, lo, hi):
        if b > a and wi != 0.0:
            d = gx[a:b] - xi
            out[a:b] += wi * norm * np.exp(-0.5 * d * d / (sigma * sigma))
    return out


def feature_matrix(spectra: Sequence[Spectrum], grid: Grid, sigma: float,
                   power: float = 1.0) -> np.ndarray:
    """The broadened spectra of a set on one grid, one row each."""
    if len(spectra) == 0:
        return np.zeros((0, len(grid.x)))
    return np.vstack([broaden(s, grid, sigma, power) for s in spectra])


def spectrum_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
                    normalize: bool = True) -> np.ndarray:
    """The linear kernel of rows of :func:`feature_matrix`, in the cosine form by default.

    Rows are `test` (or `train` when `test` is not given) and columns are `train`.  A spectrum
    with no intensity on the grid has a zero row; its kernel with every spectrum is zero.
    """
    A = np.asarray(train, float)
    B = A if test is None else np.asarray(test, float)
    if not normalize:
        return B @ A.T
    return _unit_rows(B) @ _unit_rows(A).T


def _unit_rows(F: np.ndarray) -> np.ndarray:
    norm = np.sqrt(np.einsum("ij,ij->i", F, F))
    return F / np.where(norm > 1e-300, norm, 1.0)[:, None]


def kernel(spectra: Sequence[Spectrum], x_min: float, x_max: float, sigma: float,
           others: Optional[Sequence[Spectrum]] = None, power: float = 1.0,
           normalize: bool = True, points_per_sigma: float = 2.5) -> np.ndarray:
    """The kernel of a set of spectra at one width, through a grid made for that width."""
    grid = Grid.for_width(x_min, x_max, sigma, points_per_sigma)
    F = feature_matrix(spectra, grid, sigma, power)
    G = None if others is None else feature_matrix(others, grid, sigma, power)
    return spectrum_kernel(F, G, normalize)


def multi_width_kernel(spectra: Sequence[Spectrum], x_min: float, x_max: float,
                       sigmas: Sequence[float], others: Optional[Sequence[Spectrum]] = None,
                       power: float = 1.0, weights: Optional[Sequence[float]] = None,
                       points_per_sigma: float = 2.5) -> np.ndarray:
    """The weighted mean of the cosine kernels of several widths.

    A narrow width tells two spectra apart by the exact places of their lines, and a wide one
    by the regions that hold intensity.  The mean keeps both.  The weights default to equal
    ones and are scaled to sum to one, so the diagonal stays at one.
    """
    sigmas = [float(s) for s in sigmas]
    if not sigmas:
        raise ValueError("at least one width is needed")
    w = np.ones(len(sigmas)) if weights is None else np.asarray(weights, float)
    if len(w) != len(sigmas) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("one weight per width is needed, none below zero and not all zero")
    w = w / w.sum()
    out = None
    for wk, s in zip(w, sigmas):
        K = kernel(spectra, x_min, x_max, s, others, power, True, points_per_sigma)
        out = wk * K if out is None else out + wk * K
    return out


# --------------------------------------------------------------------------- #
# The co-spectrum
# --------------------------------------------------------------------------- #
def co_spectrum(F: np.ndarray, alpha: np.ndarray, normalize: bool = True) -> np.ndarray:
    """The weight over the axis of a kernel model fitted on the rows of `F`.

    `alpha` holds the dual coefficients of the training spectra (``_Learner.alpha_`` of a
    kernel ridge fit).  With the cosine kernel the model predicts ``b + w . z`` for a new
    spectrum with the normalised row ``z``, and this function returns ``w``, one value per
    grid point.  The unit of ``w`` is the unit of the property per unit of normalised
    intensity.
    """
    F = np.asarray(F, float)
    alpha = np.asarray(alpha, float).ravel()
    if len(alpha) != len(F):
        raise ValueError(f"{len(alpha)} coefficients for {len(F)} spectra")
    Z = _unit_rows(F) if normalize else F
    return Z.T @ alpha


def co_spectrum_effect(F: np.ndarray, w: np.ndarray, normalize: bool = True) -> np.ndarray:
    """The weight times the spread of the intensity over the set, one value per grid point.

    A large weight at a place where no spectrum of the set has intensity moves no prediction.
    The product with the standard deviation of the (normalised) intensity is the change of the
    prediction that a typical difference between two spectra of the set causes at that place.
    """
    F = np.asarray(F, float)
    Z = _unit_rows(F) if normalize else F
    return np.asarray(w, float) * Z.std(axis=0)


def strongest_regions(x: np.ndarray, curve: np.ndarray, n: int = 6,
                      separation: float = 0.0) -> List[dict]:
    """The `n` places where a curve over the axis is largest in size, at least `separation`
    apart, each with its position, its value and the index of its grid point."""
    x = np.asarray(x, float)
    c = np.asarray(curve, float)
    out: List[dict] = []
    for k in np.argsort(-np.abs(c)):
        if all(abs(x[k] - r["x"]) > separation for r in out):
            out.append({"x": float(x[k]), "value": float(c[k]), "index": int(k)})
        if len(out) >= n:
            break
    return out


def two_theta_to_d(two_theta: np.ndarray, wavelength: float) -> np.ndarray:
    """The spacing of the lattice planes in angstrom that reflects at the angle 2-theta (in
    degrees) for the given wavelength in angstrom: Bragg's law, ``d = lambda / (2 sin theta)``."""
    theta = np.radians(np.asarray(two_theta, float) / 2.0)
    return float(wavelength) / (2.0 * np.sin(theta))
