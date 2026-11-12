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

The slope of a spectrum
-----------------------
A measured curve stands on a baseline: an offset or a slow drift that comes from the
instrument and from the contact with the sample, and not from the compound.  The kernel above
counts a baseline as intensity.  :func:`slope` takes the derivative of every broadened
spectrum along the axis.  A constant baseline has no slope, and a slow drift has a small one.
The kernel of the slopes is the kernel above with one more factor (:func:`slope_overlap`), ::

    k'(A, B) = SUM_ij a_i b_j (1 - d^2 / (2 sigma^2)) exp( -d^2 / (4 sigma^2) ) / (2 sigma^2)

with ``d = x_i - y_j``.  Two sticks at the same place count as before, two sticks 1.41 widths
apart count as nothing, and two sticks further apart count against each other.

The co-spectrum
---------------
A kernel ridge model on the cosine kernel is linear in the normalised broadened spectrum
``z(x)``: its prediction is ``b + INT w(x) z(x) dx``.  The weight ``w`` is the **co-spectrum**
(:func:`co_spectrum`), the one-dimensional analogue of the co-field: a curve over the axis that
is positive where intensity raises the predicted property and negative where it lowers it.

Mixtures
--------
By the Beer-Lambert law the absorbance spectrum of a mixture is the sum of the spectra of its
components, each times its amount.  The kernel is linear in each of its two spectra, so the
kernel of two mixtures is the same weighted sum over the kernels of their components
(plan item SP6): with the amounts as the rows of ``W`` and the kernel ``K`` of the
components, the kernel of the mixtures is ``W K W^T``.  :func:`mix` makes the spectrum of a
mixture and :func:`mixture_kernel` this product.  The identity holds for the kernel that is
not normalised.  The cosine form divides the total amount out: it is the form for a question
about fractions, and the form that is not normalised is the one for a question about amounts.

Two instruments
---------------
Two instruments give two spectra of one sample (plan item SP5).  Three functions describe
what an instrument does to the axis.  Two Gaussian broadenings add as squares of their
widths, so :func:`matched_widths` gives every instrument the width that brings all of them
to one resolution: a model fitted on the rows of one instrument then reads the rows of
another.  :func:`fit_width` finds the difference in resolution, and :func:`fit_shift` a shift
of the axis, from two curves such as the mean spectra of the two instruments; no sample has
to be measured on both.  :func:`smoothed` and :func:`shifted` apply a width and a shift to
curves that are rows of one grid.  A difference between two instruments that is the same
for all samples is not a matter of the axis: it is removed by taking the mean spectrum of
each instrument off its own spectra, which the benchmark of the item does.

Readers
-------
:func:`read_sticks` and :func:`read_measured` read two-column text (the reader of
``scattering.py`` with the axis left to the caller).  :func:`read_jcamp` reads a JCAMP-DX file,
the format in which instruments and repositories store a measured spectrum, with its
compressed forms; :func:`absorbance` turns a transmittance spectrum into an absorbance
spectrum.  :func:`orca_sticks` takes the infrared spectrum from an ORCA frequency job.
:func:`powder_sticks` computes the powder diffraction pattern of a crystal with pymatgen, for
X-rays or for neutrons.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
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
    #: what the file said about the spectrum, for example the unit of the intensities
    #: (``yunits``); empty for a spectrum that was not read from a file
    meta: Dict[str, str] = field(default_factory=dict)

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
        return Spectrum(self.x[keep], self.y[keep], self.kind, self.axis, self.name,
                        dict(self.meta))


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


# JCAMP-DX compresses a list of ordinates with letters in place of the sign and the first
# digit of a number (the ASDF forms): a squeezed value, a difference from the last value, and
# a repeat count.
_SQUEEZED = {**{c: i for i, c in enumerate("@ABCDEFGHI")},
             **{c: -(i + 1) for i, c in enumerate("abcdefghi")}}
_DIFFERENCE = {**{c: i for i, c in enumerate("%JKLMNOPQR")},
               **{c: -(i + 1) for i, c in enumerate("jklmnopqr")}}
_REPEAT = {c: i + 1 for i, c in enumerate("STUVWXYZs")}
#: a value of a compressed table (no exponent: E and e are squeezed digits there), a value of
#: a table of plain numbers, and the letters that only a compressed table holds
_TOKEN = re.compile(r"[@A-Ia-i%J-Rj-rS-Zs+\-]?[0-9]*\.?[0-9]*")
_PLAIN = re.compile(r"[+\-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[Ee][+\-]?[0-9]+)?")
_ONLY_COMPRESSED = re.compile(r"[@A-DF-Ia-df-i%J-Rj-rS-Zs]")
_AXES_OF_UNITS = {"1/CM": "wavenumber", "CM-1": "wavenumber", "CM^-1": "wavenumber",
                  "PPM": "shift", "EV": "binding_energy", "DEGREES": "two_theta"}


def _label(text: str) -> str:
    """A JCAMP-DX label in the form in which labels are compared: upper case, without
    spaces, dashes, slashes and underscores."""
    return re.sub(r"[\s\-/_]", "", text).upper()


def _tokens(line: str, compressed: bool = True) -> List[Tuple[str, float]]:
    """The values of one line of ordinates as (form, number): the form is "value",
    "difference" or "repeat".  A line of plain numbers may use exponents; a compressed line
    may not, because the letters E and e stand for digits there."""
    out: List[Tuple[str, float]] = []
    if not compressed:
        return [("value", float(m.group(0))) for m in _PLAIN.finditer(line)]
    for m in _TOKEN.finditer(line.replace(",", " ")):
        word = m.group(0)
        if not word:
            continue
        head = word[0]
        if head in _SQUEEZED:
            digits = word[1:]
            value = float(str(abs(_SQUEEZED[head])) + digits)
            out.append(("value", -value if _SQUEEZED[head] < 0 else value))
        elif head in _DIFFERENCE:
            digits = word[1:]
            value = float(str(abs(_DIFFERENCE[head])) + digits)
            out.append(("difference", -value if _DIFFERENCE[head] < 0 else value))
        elif head in _REPEAT:
            out.append(("repeat", float(str(_REPEAT[head]) + word[1:])))
        elif word not in ("+", "-", "."):
            out.append(("value", float(word)))
    return out


def decode_ordinates(lines: Sequence[str]) -> np.ndarray:
    """The ordinates of an ``(X++(Y..Y))`` table of a JCAMP-DX file.

    Every line starts with an abscissa, which is only a check and is dropped.  The ordinates
    that follow are plain numbers or one of the compressed forms: a squeezed value, a
    difference from the value before, or a count that repeats the last value or the last
    difference.  When a line ends with a difference, the next line starts with the value that
    the differences have reached, again as a check, and that value is not a new ordinate.
    """
    y: List[float] = []
    ended_with_difference = False
    compressed = any(_ONLY_COMPRESSED.search(line) for line in lines)
    for line in lines:
        tokens = _tokens(line, compressed)
        if len(tokens) < 2:
            continue
        tokens = tokens[1:]                              # the abscissa of the line
        last = None                                      # the last form on this line
        first = True
        for form, number in tokens:
            if form == "value":
                if first and ended_with_difference and y:
                    last = ("value", number)             # the check value: already in y
                else:
                    y.append(number)
                    last = ("value", number)
            elif form == "difference":
                y.append((y[-1] if y else 0.0) + number)
                last = ("difference", number)
            elif form == "repeat" and last is not None:
                for _ in range(int(number) - 1):
                    y.append(y[-1] + last[1] if last[0] == "difference" else y[-1])
            first = False
        ended_with_difference = last is not None and last[0] == "difference"
    return np.array(y, float)


def parse_jcamp(text: str) -> List[dict]:
    """The blocks of a JCAMP-DX text that hold data.

    A file can hold several blocks, each from ``##TITLE=`` to ``##END=``: a spectrum, a table
    of the peaks that somebody picked, and blocks that only link the others.  Every block with
    an ``XYDATA`` table or a ``PEAKTABLE`` (or ``XYPOINTS``) is returned as a dictionary with
    ``labels`` (the labelled entries of the block, by the label in upper case without spaces,
    dashes, slashes and underscores), ``kind`` ("curve" for ``XYDATA``, "sticks" for a table
    of points) and the arrays ``x`` and ``y``.  The factors ``XFACTOR`` and ``YFACTOR`` are
    applied.  For ``(X++(Y..Y))`` the abscissas run evenly from ``FIRSTX`` to ``LASTX``.
    """
    blocks: List[dict] = []
    labels: Dict[str, str] = {}
    table: Optional[str] = None
    form = ""
    rows: List[str] = []

    def close():
        nonlocal table, rows, form
        if table is None:
            return
        try:
            if "X++(Y..Y)" in form.replace(" ", "").upper():
                y = decode_ordinates(rows) * float(labels.get("YFACTOR", 1.0) or 1.0)
                if len(y) >= 2 and "FIRSTX" in labels and "LASTX" in labels:
                    x = np.linspace(float(labels["FIRSTX"]), float(labels["LASTX"]), len(y))
                    blocks.append({"labels": dict(labels), "kind": "curve", "x": x, "y": y})
            else:
                numbers = [float(w) for row in rows
                           for w in re.split(r"[\s,;()]+", row) if w]
                pairs = np.array(numbers[:len(numbers) // 2 * 2], float).reshape(-1, 2)
                if len(pairs):
                    kind = "curve" if table == "XYDATA" else "sticks"
                    blocks.append({"labels": dict(labels), "kind": kind,
                                   "x": pairs[:, 0] * float(labels.get("XFACTOR", 1.0) or 1.0),
                                   "y": pairs[:, 1] * float(labels.get("YFACTOR", 1.0) or 1.0)})
        except ValueError:
            pass                                          # a table that is not numbers
        table, rows, form = None, [], ""

    for raw in text.splitlines():
        line = raw.split("$$")[0].rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("##"):
            close()
            name, _, value = line.lstrip()[2:].partition("=")
            key = _label(name)
            if key == "TITLE":
                labels = {}
            if key in ("XYDATA", "PEAKTABLE", "XYPOINTS"):
                table, form = key, value
            elif key != "END":
                labels[key] = value.strip()
        elif table is not None:
            rows.append(line)
    close()
    return blocks


def read_jcamp(path: str, name: str = "") -> Spectrum:
    """A spectrum from a JCAMP-DX file: its first block with a table of data.

    A block with ``XYDATA`` gives a measured curve, and a block with a table of points gives
    a stick spectrum; a curve is taken first when the file has both.  The axis is named from
    ``XUNITS`` (``1/CM`` is the wavenumber).  ``meta`` holds ``yunits``, ``xunits``,
    ``title`` and ``datatype`` as the file gives them; see :func:`absorbance`.
    """
    with open(path, encoding="utf-8", errors="replace") as fh:
        blocks = parse_jcamp(fh.read())
    if not blocks:
        raise ValueError(f"{path}: no table of data was found in this JCAMP-DX file")
    block = next((b for b in blocks if b["kind"] == "curve"), blocks[0])
    lab = block["labels"]
    units = lab.get("XUNITS", "").upper().replace(" ", "")
    meta = {"yunits": lab.get("YUNITS", ""), "xunits": lab.get("XUNITS", ""),
            "title": lab.get("TITLE", ""), "datatype": lab.get("DATATYPE", "")}
    return Spectrum(block["x"], block["y"], block["kind"],
                    _AXES_OF_UNITS.get(units, units.lower() or "wavenumber"),
                    name or lab.get("TITLE", ""), meta)


def absorbance(spectrum: Spectrum) -> Spectrum:
    """The same spectrum with absorbance as its intensity.

    The unit is read from ``meta["yunits"]``.  A transmittance ``T`` becomes ``-log10(T)``; a
    transmittance in per cent (the unit says so, or the values go above 1.5) is divided by
    100 first, and values at or below zero are set to a small positive number.  An absorbance
    spectrum is returned as it is.  Another unit is an error, because the kernel compares
    intensities and must not mix two kinds of them.
    """
    units = spectrum.meta.get("yunits", "").upper()
    if "ABSORB" in units:
        return spectrum
    if "TRANSMI" not in units:
        raise ValueError(f"the spectrum '{spectrum.name}' has the intensity unit "
                         f"'{spectrum.meta.get('yunits', '')}', which is neither a "
                         f"transmittance nor an absorbance")
    t = np.asarray(spectrum.y, float)
    if "%" in units or "PERCENT" in units or (len(t) and float(np.max(t)) > 1.5):
        t = t / 100.0
    y = -np.log10(np.clip(t, 1e-6, None))
    return Spectrum(spectrum.x, y, spectrum.kind, spectrum.axis, spectrum.name,
                    dict(spectrum.meta, yunits="ABSORBANCE"))


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


def slope_overlap(a: Spectrum, b: Spectrum, sigma: float, power: float = 1.0) -> float:
    """The overlap of the slopes of two broadened spectra, in closed form: ::

        SUM_ij a_i b_j (1 - d^2 / (2 sigma^2)) exp(-d^2 / (4 sigma^2)) / (2 sigma^2)

    with ``d = x_i - y_j``.  It is ``2 sigma sqrt(pi)`` times the integral of the product of
    the derivatives of the two broadened spectra, as :func:`stick_overlap` is that multiple
    of the integral of the product of the broadened spectra themselves.  A constant added to
    a measured curve changes this sum only through the two ends of the curve.
    """
    if sigma <= 0:
        raise ValueError("the width sigma must be positive")
    xa, wa = a.lines()
    xb, wb = b.lines()
    if len(xa) == 0 or len(xb) == 0:
        return 0.0
    wa, wb = _powered(wa, power), _powered(wb, power)
    u = (xa[:, None] - xb[None, :]) ** 2 / (sigma * sigma)
    return float(wa @ ((1.0 - 0.5 * u) * np.exp(-0.25 * u)) @ wb) / (2.0 * sigma * sigma)


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
    if spectrum.kind == "curve" and len(x) > 2:
        steps = np.diff(x)
        if np.allclose(steps, steps[0], rtol=1e-3, atol=0.0) and sigma >= 2.0 * steps[0]:
            return _broaden_even_curve(x, w, float(np.mean(steps)), gx, sigma)
    reach = CUT * sigma
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    lo = np.searchsorted(gx, x - reach, side="left")
    hi = np.searchsorted(gx, x + reach, side="right")
    for xi, wi, a, b in zip(x, w, lo, hi):
        if b > a and wi != 0.0:
            d = gx[a:b] - xi
            out[a:b] += wi * norm * np.exp(-0.5 * d * d / (sigma * sigma))
    return out


def _broaden_even_curve(x: np.ndarray, w: np.ndarray, step: float, gx: np.ndarray,
                        sigma: float) -> np.ndarray:
    """The sum of the Gaussians of the points of a curve on an even grid, as one convolution.

    A measured curve has thousands of points, and a loop over them is slow.  On an even grid
    the sum at the points of the curve is a convolution of the weights with the Gaussian,
    which is exact there; between two points of the curve the value is interpolated.  The
    caller uses this only when the width is at least two steps of the curve, where the
    interpolation changes the result by less than one per cent of its largest value.
    """
    half = int(math.ceil(CUT * sigma / step))
    offsets = step * np.arange(-half, half + 1)
    kernel = np.exp(-0.5 * offsets * offsets / (sigma * sigma)) / (sigma * np.sqrt(2.0 * np.pi))
    full = np.convolve(w, kernel, mode="full")
    positions = x[0] + step * (np.arange(len(full)) - half)
    return np.interp(gx, positions, full, left=0.0, right=0.0)


def feature_matrix(spectra: Sequence[Spectrum], grid: Grid, sigma: float,
                   power: float = 1.0) -> np.ndarray:
    """The broadened spectra of a set on one grid, one row each."""
    if len(spectra) == 0:
        return np.zeros((0, len(grid.x)))
    return np.vstack([broaden(s, grid, sigma, power) for s in spectra])


def slope(F: np.ndarray, grid: Grid) -> np.ndarray:
    """The derivative along the axis of every row of a :func:`feature_matrix`.

    The cosine kernel of these rows (:func:`spectrum_kernel`) is the kernel of the slopes of
    the spectra, which does not see a constant baseline (:func:`slope_overlap` is its closed
    form).  The derivative is a central difference on the grid, which makes the dot product
    of two rows too small by about ``step^2 / (2 sigma^2)`` of its value: two per cent with
    five grid points in one width.  The cosine form divides most of this out.  A broadened
    measured curve falls to
    zero within a few widths of its two ends, and this fall is a slope that the compound did
    not cause: let the grid end a few widths inside the range of the measured curves.
    """
    F = np.asarray(F, float)
    if F.ndim != 2 or F.shape[1] < 2:
        raise ValueError("the slope needs rows with at least two grid points")
    return np.gradient(F, grid.step, axis=1)


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
# Mixtures
# --------------------------------------------------------------------------- #
def _amounts(weights, n: int) -> np.ndarray:
    w = np.atleast_1d(np.asarray(weights, float))
    if w.ndim != 1 or len(w) != n:
        raise ValueError(f"{n} components need {n} amounts; got an array of shape {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("the amounts have to be numbers that are not negative")
    return w


def mix(spectra: Sequence[Spectrum], weights: Sequence[float], name: str = "") -> Spectrum:
    """The spectrum of a mixture: the sum of the spectra of its components, each times its
    amount (the Beer-Lambert law for absorbance).

    The amounts are used as they are and are not scaled to a sum of one: twice the amount of
    everything gives twice the spectrum.  Stick spectra give a stick spectrum with the sticks
    of all components.  Measured curves have to be on one grid and give a curve on that grid.
    Sticks and curves cannot be mixed here; broaden both onto one grid first
    (:func:`feature_matrix`) and mix the rows.
    """
    if len(spectra) == 0:
        raise ValueError("a mixture needs at least one component")
    w = _amounts(weights, len(spectra))
    kinds = {s.kind for s in spectra}
    if len(kinds) > 1:
        raise ValueError("sticks and curves cannot be mixed; put both on one grid first")
    first = spectra[0]
    if first.kind == "sticks":
        x = np.concatenate([s.x for s in spectra])
        y = np.concatenate([wi * s.y for wi, s in zip(w, spectra)])
        return Spectrum(x, y, "sticks", first.axis, name)
    for s in spectra[1:]:
        if len(s) != len(first) or not np.allclose(s.x, first.x):
            raise ValueError("measured curves have to be on one grid to be mixed")
    y = np.sum([wi * s.y for wi, s in zip(w, spectra)], axis=0)
    return Spectrum(first.x, y, "curve", first.axis, name, dict(first.meta))


def mixture_kernel(K: np.ndarray, amounts: np.ndarray, others: Optional[np.ndarray] = None,
                   normalize: bool = False) -> np.ndarray:
    """The kernel of mixtures from the kernel of their components: ``W' K W^T``.

    `K` is the kernel of the components, not normalised (``spectrum_kernel(F,
    normalize=False)`` or :func:`exact_kernel` with ``normalize=False``).  `amounts` holds
    one row per mixture with the amount of every component; the columns of the result are
    these mixtures.  The rows are the mixtures of `others`, or the same mixtures.  With
    `normalize` the result is put into the cosine form afterwards, which removes the total
    amount of each mixture; the cosine kernel of the components must not be given as `K`,
    because the norm of a mixture is not the sum of the norms of its components.
    """
    K = np.asarray(K, float)
    W = np.atleast_2d(np.asarray(amounts, float))
    V = W if others is None else np.atleast_2d(np.asarray(others, float))
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("the kernel of the components has to be a square matrix")
    if W.shape[1] != len(K) or V.shape[1] != len(K):
        raise ValueError(f"{len(K)} components need {len(K)} amounts in every row")
    out = V @ K @ W.T
    if normalize:
        nw = np.einsum("ij,jk,ik->i", W, K, W)
        nv = np.einsum("ij,jk,ik->i", V, K, V)
        out = out / np.sqrt(np.maximum(nv[:, None] * nw[None, :], 1e-300))
    return out


# --------------------------------------------------------------------------- #
# Two instruments
# --------------------------------------------------------------------------- #
def matched_widths(sigma: float, resolutions: Sequence[float]) -> np.ndarray:
    """The width for the spectra of every instrument, so that all end at one width.

    `resolutions` holds, for every instrument, the width that the instrument has already put
    on its spectra (the standard deviation of a Gaussian, in the units of the axis).  The
    widths of two Gaussian broadenings add as squares, so the instrument with the coarsest
    resolution gets `sigma` and every other instrument gets
    ``sqrt(sigma^2 + r_max^2 - r_k^2)``.  After that every spectrum has the width
    ``sqrt(sigma^2 + r_max^2)``, and a model fitted on one instrument reads the rows of
    another (plan item SP5).
    """
    r = np.asarray(resolutions, float)
    if r.ndim != 1 or len(r) == 0 or np.any(r < 0) or sigma < 0:
        raise ValueError("one resolution per instrument is needed, and no width below zero")
    return np.sqrt(float(sigma) ** 2 + float(r.max()) ** 2 - r ** 2)


def smoothed(rows: np.ndarray, step: float, sigma: float) -> np.ndarray:
    """Curves on one even grid of step `step`, each broadened by a Gaussian of width `sigma`.

    This is :func:`broaden` for curves that are already rows of one grid.  The first and the
    last value of a row are continued beyond its ends, so a curve that stands on a baseline
    does not fall to zero there.  ``sigma = 0`` returns the rows.
    """
    from scipy.ndimage import gaussian_filter1d

    rows = np.asarray(rows, float)
    if sigma < 0 or not step > 0:
        raise ValueError("the width must not be negative and the step has to be positive")
    if sigma == 0:
        return rows
    return gaussian_filter1d(rows, float(sigma) / float(step), axis=-1, mode="nearest",
                             truncate=CUT)


def shifted(rows: np.ndarray, x: np.ndarray, shift: float) -> np.ndarray:
    """Curves on the grid `x`, moved by `shift` along the axis: the result at ``x`` is the
    curve at ``x - shift`` (cubic interpolation).  Beyond the two ends the first and the last
    value are used."""
    from scipy.interpolate import CubicSpline

    rows = np.asarray(rows, float)
    x = np.asarray(x, float)
    if shift == 0:
        return rows
    at = np.clip(x - float(shift), x[0], x[-1])
    return CubicSpline(x, rows, axis=-1)(at)


def _slope_correlation(a: np.ndarray, b: np.ndarray, x: np.ndarray, margin: int) -> float:
    da = np.gradient(a, x)[margin:len(x) - margin]
    db = np.gradient(b, x)[margin:len(x) - margin]
    da, db = da - da.mean(), db - db.mean()
    return float(da @ db / np.sqrt(max((da @ da) * (db @ db), 1e-300)))


def fit_shift(a: np.ndarray, b: np.ndarray, x: np.ndarray, max_shift: float,
              points: int = 20) -> float:
    """The shift of the axis between two curves on one grid: the ``d`` for which ``a`` moved
    by ``d`` (:func:`shifted`) has the slopes of ``b``.

    The curves are, for example, the mean spectra of two instruments; they do not have to
    come from the same samples.  The slopes are compared, so an offset between the curves
    does not count.  Shifts from ``-max_shift`` to ``max_shift`` are tried in steps of one
    grid step divided by `points`, and the part of the axis within `max_shift` of an end is
    left out.  ``shifted(rows_of_b, x, -d)`` puts the spectra of the second instrument on the
    axis of the first.
    """
    a, b, x = (np.asarray(v, float) for v in (a, b, x))
    step = float(x[1] - x[0])
    if not max_shift > 0:
        raise ValueError("max_shift has to be positive")
    margin = int(math.ceil(max_shift / step)) + 1
    if len(x) <= 2 * margin + 3:
        raise ValueError("the curves are too short for this max_shift")
    n = int(round(max_shift / (step / points)))
    tried = (step / points) * np.arange(-n, n + 1)
    score = [_slope_correlation(shifted(a, x, d), b, x, margin) for d in tried]
    return float(tried[int(np.argmax(score))])


def fit_width(a: np.ndarray, b: np.ndarray, x: np.ndarray,
              widths: Sequence[float]) -> Tuple[float, float]:
    """The difference in resolution between two curves on one grid.

    Every width ``w`` of `widths` is tried in both directions: ``a`` broadened by ``w``
    against ``b``, and ``b`` broadened by ``w`` against ``a``.  The direction and the width
    with the highest correlation of the slopes win.  Returns the width with a sign and that
    correlation: a positive width says that ``b`` is the coarser curve and ``a`` has to be
    broadened by it; a negative one says the opposite; zero says that no broadening makes
    the curves more alike.  The curves are, for example, the mean spectra of two instruments
    (:func:`fit_shift`); remove a shift between them first.
    """
    a, b, x = (np.asarray(v, float) for v in (a, b, x))
    step = float(x[1] - x[0])
    margin = int(math.ceil(3.0 * max([0.0] + [float(w) for w in widths]) / step)) + 1
    if len(x) <= 2 * margin + 3:
        raise ValueError("the curves are too short for the largest width")
    best = (_slope_correlation(a, b, x, margin), 0.0)
    for w in widths:
        if w <= 0:
            continue
        for sign, (p, q) in ((1.0, (smoothed(a, step, w), b)), (-1.0, (a, smoothed(b, step, w)))):
            score = _slope_correlation(p, q, x, margin)
            if score > best[0]:
                best = (score, sign * float(w))
    return best[1], best[0]


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
