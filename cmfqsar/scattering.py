"""Measured scattering curves as samples: the total, untyped descriptor (plan item M12).

A diffraction experiment on a powder, a glass or a nanoparticle does not give atomic
coordinates.  It gives one curve: the total structure function ``F(Q)``, or its sine transform
the reduced pair distribution function ``G(r)``.  This module connects that curve to the
descriptors of this program, so that a model can be fitted where no structure is known.

The curve is one channel block of the descriptor
------------------------------------------------
The periodic autocorrelation of ``periodic.py`` is a *typed* pair-distance distribution: one
block per pair of field types.  An experiment cannot separate the types, because every atom
scatters into the same detector.  What it measures is the single block of one field whose weight
is the atom's **scattering length** -- the neutron scattering length ``b``, or for X-rays the
atomic number ``Z``.  Written out, with ``N`` atoms in the cell, number density ``rho_0`` and
``<b>`` the mean scattering length, ::

    G(r) = Phi_bb(r) / ( sigma sqrt(2 pi) * r * <b>^2 ),

where ``Phi_bb`` is :func:`cmfqsar.periodic.periodic_autocorrelation` of that one field, with
the uniform background kept and the self term dropped.  The background term of the descriptor is
exactly the ``- 4 pi r rho_0`` of the standard definition, and the factor
``sigma sqrt(2 pi)`` is the integral of the channel's Gaussian.  The identity is exact as the
channel narrows: the error is of order ``(sigma / r)^2``, and :func:`pair_distribution_residual`
measures it against a direct sum over the neighbour list.

The same holds in reciprocal space.  The Faber-Ziman total structure factor is the one-field
channel of ``reciprocal.py`` divided by ``<b>^2``, ::

    S(Q) = Psi_bb(Q) / <b>^2,        F(Q) = Q [ S(Q) - 1 ],

with the form factor switched off, and ``G`` and ``F`` are a sine transform apart,
``G(r) = (2/pi) INT F(Q) sin(Qr) dQ``.  So the two things an experiment can report and the two
descriptions this program already had are one object seen four ways.

What this buys, and what it costs
---------------------------------
It buys a sample where there is no structure: a measured curve is a feature row like any other,
so a model fitted on computed curves can predict for a measured one, and curves of amorphous or
nanocrystalline materials become usable.  It costs the typing.  Whether that cost is affordable
is a question about the data and not about the code, and `benchmarks/run_scattering.py` measures
it: the total curve against the typed descriptor on the same structures and the same folds.

A measured curve is also not a computed one.  Four things separate them, and
:func:`measured_like` applies all four to a computed curve so that a model can be tested against
them before any experiment is trusted:

* **a finite range in Q.**  The transform runs to the largest Q the instrument reached, which
  puts ripples of period ``2 pi / Q_max`` around every peak;
* **counting noise** on ``F(Q)``;
* **an error in the number density**, which adds a straight line ``-4 pi r delta rho`` to
  ``G(r)``, because ``rho_0`` is what the baseline is drawn from;
* **a limited range in r**: the low-r region is dominated by termination ripples and the high-r
  region by the instrument's resolution.

Reading a measured file
-----------------------
:func:`read_curve` reads the two-column text that PDFgetX3, diffpy and the other common programs
write: comment lines starting with ``#``, then ``r`` and ``G`` (or ``Q`` and ``F``, or ``Q`` and
``S``), with any further columns ignored.  Values the header happens to carry -- ``qmax``,
``rho0``, ``composition`` -- are returned beside the curve when they are there.
"""
from __future__ import annotations

import dataclasses
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: how an atom scatters.  "xray" uses the atomic number, which is the Q -> 0 limit of the X-ray
#: form factor and the weighting most X-ray pair distribution work uses; "neutron" uses the bound
#: coherent scattering length.
SOURCES = ("xray", "neutron")

#: bound coherent neutron scattering lengths in femtometre (Sears, Neutron News 3 (1992) 26).
#: Only the elements listed here are available: an element that is missing is an error rather
#: than a guess, and :func:`scattering_lengths` says so.  Elements with a strongly
#: wavelength-dependent or complex value near a resonance are left out for that reason.
NEUTRON_B: Dict[str, float] = {
    "H": -3.739, "D": 6.671, "He": 3.26, "Li": -1.90, "Be": 7.79, "B": 5.30, "C": 6.646,
    "N": 9.36, "O": 5.803, "F": 5.654, "Ne": 4.566, "Na": 3.63, "Mg": 5.375, "Al": 3.449,
    "Si": 4.1491, "P": 5.13, "S": 2.847, "Cl": 9.577, "Ar": 1.909, "K": 3.67, "Ca": 4.70,
    "Sc": 12.29, "Ti": -3.438, "V": -0.3824, "Cr": 3.635, "Mn": -3.73, "Fe": 9.45, "Co": 2.49,
    "Ni": 10.3, "Cu": 7.718, "Zn": 5.68, "Ga": 7.288, "Ge": 8.185, "As": 6.58, "Se": 7.970,
    "Br": 6.795, "Kr": 7.81, "Rb": 7.09, "Sr": 7.02, "Y": 7.75, "Zr": 7.16, "Nb": 7.054,
    "Mo": 6.715, "Ru": 7.03, "Rh": 5.88, "Pd": 5.91, "Ag": 5.922, "Cd": 4.87, "In": 4.065,
    "Sn": 6.225, "Sb": 5.57, "Te": 5.80, "I": 5.28, "Xe": 4.92, "Cs": 5.42, "Ba": 5.07,
    "La": 8.24, "Ce": 4.84, "Pr": 4.58, "Nd": 7.69, "Ta": 6.91, "W": 4.86, "Re": 9.2,
    "Os": 10.7, "Ir": 10.6, "Pt": 9.60, "Au": 7.63, "Hg": 12.692, "Tl": 8.776, "Pb": 9.405,
    "Bi": 8.532, "Th": 10.31, "U": 8.417,
}


@dataclass(frozen=True)
class CurveSpec:
    """The grids a curve is reported on: ``r`` for ``G(r)`` and ``Q`` for ``F(Q)``.

    ``sigma`` is the width of the channel the pair distribution is computed with, which stands
    for the resolution of the instrument in real space.  The identity with the descriptor is
    exact as ``sigma`` falls, so a narrow channel is the honest setting and a wide one is the
    statement that the measurement is blurred.
    """
    r_min: float = 1.0
    r_max: float = 20.0
    r_step: float = 0.05
    sigma: float = 0.05
    q_min: float = 0.5
    q_max: float = 25.0
    q_step: float = 0.05
    #: the margin, in widths of the channel, kept beyond ``r_max`` while computing, so that the
    #: curve is not cut where the window of the descriptor starts to bite
    margin: float = 6.0

    @property
    def r(self) -> np.ndarray:
        n = max(int(round((self.r_max - self.r_min) / self.r_step)) + 1, 1)
        return self.r_min + np.arange(n, dtype=float) * self.r_step

    @property
    def q(self) -> np.ndarray:
        n = max(int(round((self.q_max - self.q_min) / self.q_step)) + 1, 1)
        return self.q_min + np.arange(n, dtype=float) * self.q_step

    def periodic_spec(self):
        """The descriptor spec whose one-field channels are this curve."""
        from .periodic import PeriodicSpec

        return PeriodicSpec(r_max=self.r_max + self.margin * self.sigma, step=self.r_step,
                            sigma=self.sigma, taper=0.0, include_self=False, background=True)

    def reciprocal_spec(self):
        """The reciprocal spec whose one-field channels are this structure factor.

        The form factor is switched off (``form_sigma`` 0): a scattering experiment carries its
        own, and putting a second one in would damp the curve twice.
        """
        from .reciprocal import ReciprocalSpec

        return ReciprocalSpec(q_max=self.q_max, step=self.q_step, sigma=self.q_step,
                              form_sigma=0.0, include_self=False, taper=0.0)


# --------------------------------------------------------------------------- #
# How an atom scatters
# --------------------------------------------------------------------------- #
def scattering_lengths(symbols: Sequence[str], source: str = "xray") -> np.ndarray:
    """The scattering length of every atom: the atomic number, or the neutron value.

    For X-rays this is the Q -> 0 limit of the form factor.  A real X-ray form factor falls with
    Q, which is why X-ray pair distribution work divides by the average form factor; taking the
    limit instead is the usual approximation and is stated here rather than hidden.
    """
    if source not in SOURCES:
        raise ValueError(f"the scattering source is {source!r}; it is one of {SOURCES}")
    symbols = [str(s) for s in symbols]
    if source == "neutron":
        missing = sorted({s for s in symbols if s not in NEUTRON_B})
        if missing:
            raise ValueError(
                f"no neutron scattering length is tabulated here for {', '.join(missing)}. "
                f"Add the bound coherent value to cmfqsar.scattering.NEUTRON_B, or use "
                f"source='xray', or give the weights yourself.")
        return np.array([NEUTRON_B[s] for s in symbols], float)
    from ase.data import atomic_numbers

    missing = sorted({s for s in symbols if s not in atomic_numbers})
    if missing:
        raise ValueError(f"not an element: {', '.join(missing)}")
    return np.array([float(atomic_numbers[s]) for s in symbols], float)


def scattering_record(crystal, source: str = "xray"):
    """The same structure carrying one field: how each of its atoms scatters.

    Everything else -- the coordinates, the cell, the periodicity flags -- is shared with the
    structure it came from, so this costs nothing but the weights.
    """
    if getattr(crystal, "site_species", None) is not None:
        raise ValueError(
            "a disordered cell has no single scattering length per site in this version: the "
            "mean over the species is right for the pair term and not for the self term, which "
            "the total curve keeps. Use an ordered cell.")
    b = scattering_lengths(crystal.symbols, source)
    return dataclasses.replace(crystal, weights=b[:, None], second_moments=None,
                               descriptors=None, probe_coeffs={},
                               name=f"{crystal.name} ({source})")


# --------------------------------------------------------------------------- #
# The curves a structure would give
# --------------------------------------------------------------------------- #
def pair_distribution(crystal, spec: CurveSpec = CurveSpec(),
                      source: str = "xray") -> Tuple[np.ndarray, np.ndarray]:
    """``(r, G(r))``: the reduced pair distribution function of one structure.

    ``G(r) = 4 pi r rho_0 [g(r) - 1]`` in the usual convention, computed as the one-field
    periodic autocorrelation divided by ``r``, by the mean scattering length squared and by the
    integral of the channel's Gaussian (see the module docstring).
    """
    from .periodic import periodic_autocorrelation

    rec = scattering_record(crystal, source)
    ps = spec.periodic_spec()
    phi = periodic_autocorrelation(rec, ps)
    b = np.asarray(rec.weights, float)[:, 0]
    mean_b = float(np.mean(b)) if len(b) else 0.0
    if abs(mean_b) < 1e-12:
        raise ValueError(f"{crystal.name or 'this structure'} has a mean scattering length of "
                         f"zero, so G(r) is not defined for it")
    r = spec.r
    value = np.interp(r, ps.centres, phi)
    return r, value / (spec.sigma * np.sqrt(2.0 * np.pi) * r * mean_b ** 2)


def structure_function(crystal, spec: CurveSpec = CurveSpec(), source: str = "xray",
                       route: str = "transform") -> Tuple[np.ndarray, np.ndarray]:
    """``(Q, F(Q))`` with ``F(Q) = Q [S(Q) - 1]``: the reduced total structure function.

    ``S(Q)`` is the Faber-Ziman total: the *distinct* scattering, with the term an atom gives
    with itself taken out, so that ``S`` approaches 1 at large Q and ``F`` approaches zero.
    Leaving that term in would add ``(<b^2>/<b>^2 - 1) Q``, a straight line growing with Q, to
    every curve of a structure with more than one element.

    Two routes, because they are the same quantity and not the same computation:

    * ``"transform"`` (the default) is the sine transform of the computed ``G(r)``, which is the
      operational definition and the one that matches the ``G`` this module returns, truncation
      at ``r_max`` included -- exactly as a measured ``F`` matches the measured ``G`` it was
      made from;
    * ``"reciprocal"`` sums over the reciprocal lattice (``reciprocal.py``).  That is what the
      structure factor *is*, and for an infinite crystal it is a set of sharp reflections, which
      a channel grid of finite width samples rather than resolves.  It is the right route for a
      cell whose reflections are dense, and the wrong one for a small cell with a few sharp ones.
    """
    if route not in ("transform", "reciprocal"):
        raise ValueError(f"the route is {route!r}; it is 'transform' or 'reciprocal'")
    q = spec.q
    if route == "transform":
        r, g = pair_distribution(crystal, spec, source)
        return q, inverse_sine_transform(r, g, q)
    from .reciprocal import reciprocal_autocorrelation

    rec = scattering_record(crystal, source)
    rs = spec.reciprocal_spec()
    psi = reciprocal_autocorrelation(rec, rs)
    b = np.asarray(rec.weights, float)[:, 0]
    mean_b = float(np.mean(b)) if len(b) else 0.0
    if abs(mean_b) < 1e-12:
        raise ValueError(f"{crystal.name or 'this structure'} has a mean scattering length of "
                         f"zero, so S(Q) is not defined for it")
    return q, q * np.interp(q, rs.centres, psi) / mean_b ** 2


def pair_distribution_residual(crystal, spec: CurveSpec = CurveSpec(),
                               source: str = "xray") -> dict:
    """How far the descriptor route is from the definition of ``G(r)``.

    The definition is a sum over the neighbour list, smeared by a normalised Gaussian; the
    descriptor route is the one-field autocorrelation.  They are the same in the limit of a
    narrow channel, so the residual says how narrow is narrow enough.
    """
    from .periodic import _neighbour_pairs

    rec = scattering_record(crystal, source)
    b = np.asarray(rec.weights, float)[:, 0]
    n = len(b)
    mean_b = float(np.mean(b))
    rho0 = n / float(abs(np.linalg.det(np.asarray(crystal.cell, float))))
    r = spec.r
    i, j, d = _neighbour_pairs(rec, spec.r_max + spec.margin * spec.sigma)
    w = b[i] * b[j]
    g = (np.exp(-((d[:, None] - r[None, :]) ** 2) / (2.0 * spec.sigma ** 2))
         / (spec.sigma * np.sqrt(2.0 * np.pi)))
    direct = (w[:, None] * g).sum(axis=0) / n / (r * mean_b ** 2) - 4.0 * np.pi * r * rho0
    _, through = pair_distribution(crystal, spec, source)
    scale = max(float(np.abs(direct).max()), 1e-30)
    return {"r": r, "definition": direct, "descriptor": through,
            "worst": float(np.abs(direct - through).max() / scale), "peak": scale}


# --------------------------------------------------------------------------- #
# The two transforms
# --------------------------------------------------------------------------- #
def sine_transform(q: np.ndarray, f: np.ndarray, r: np.ndarray) -> np.ndarray:
    """``G(r) = (2/pi) INT F(Q) sin(Qr) dQ``, by the trapezium rule on the given grid."""
    q = np.asarray(q, float)
    f = np.asarray(f, float)
    r = np.asarray(r, float)
    kernel = np.sin(np.outer(r, q))
    return (2.0 / np.pi) * np.trapezoid(kernel * f[None, :], q, axis=1)


def inverse_sine_transform(r: np.ndarray, g: np.ndarray, q: np.ndarray) -> np.ndarray:
    """``F(Q) = INT G(r) sin(Qr) dr``, the other direction of the same transform."""
    r = np.asarray(r, float)
    g = np.asarray(g, float)
    q = np.asarray(q, float)
    kernel = np.sin(np.outer(q, r))
    return np.trapezoid(kernel * g[None, :], r, axis=1)


# --------------------------------------------------------------------------- #
# What a measurement does to a curve
# --------------------------------------------------------------------------- #
def measured_like(spec: CurveSpec, r: np.ndarray, g: np.ndarray, q_max: Optional[float] = None,
                  noise: float = 0.0, density_error: float = 0.0, density: float = 0.0,
                  seed: int = 0) -> np.ndarray:
    """A computed ``G(r)`` put through what a real measurement does to it.

    ``q_max`` transforms the curve to ``F(Q)``, cuts it there and transforms back, which is what
    an instrument of that range would have reported, ripples and all.  ``noise`` adds Gaussian
    noise to ``F(Q)`` with that standard deviation, in units of the root mean square of ``F``.
    ``density_error`` is the relative error in the number density used to draw the baseline, and
    adds ``-4 pi r rho_0 * density_error`` to the curve; it needs ``density``, the number of
    atoms per cubic angstrom of the structure.  Each is optional and they compose in that order.
    """
    out = np.asarray(g, float).copy()
    r = np.asarray(r, float)
    if q_max is not None and q_max > 0:
        q = np.arange(spec.q_step, float(q_max) + 1e-9, spec.q_step)
        f = inverse_sine_transform(r, out, q)
        if noise > 0:
            rms = float(np.sqrt(np.mean(f ** 2)))
            f = f + np.random.RandomState(seed).normal(0.0, noise * rms, size=f.shape)
        out = sine_transform(q, f, r)
    elif noise > 0:
        rms = float(np.sqrt(np.mean(out ** 2)))
        out = out + np.random.RandomState(seed).normal(0.0, noise * rms, size=out.shape)
    if density_error:
        out = out - 4.0 * np.pi * r * float(density) * float(density_error)
    return out


# --------------------------------------------------------------------------- #
# Reading a measured file
# --------------------------------------------------------------------------- #
#: what the header of a measured file may carry, and the name it is returned under
HEADER_KEYS = {"qmax": "q_max", "qmin": "q_min", "rho0": "density", "number density": "density",
               "composition": "composition", "temperature": "temperature"}


def read_curve(path: str) -> dict:
    """A measured curve from the two-column text the common programs write.

    Returns ``{"x", "y", "path", "header"}`` and whatever of :data:`HEADER_KEYS` the comments
    carried.  Lines beginning with ``#`` are comments, a line of two or more numbers is data, and
    columns after the second (the uncertainties) are ignored.  The file says in its own units
    what ``x`` is: ``r`` in angstrom for a ``.gr``, ``Q`` in inverse angstrom for a ``.fq`` or
    ``.sq``; this reader does not guess, it only reads.
    """
    header: Dict[str, object] = {}
    xs: List[float] = []
    ys: List[float] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            if text.startswith("#") or text.startswith(";"):
                body = text.lstrip("#; ").strip()
                for key, name in HEADER_KEYS.items():
                    m = re.match(rf"{re.escape(key)}\s*[=:]\s*(\S+)", body, re.I)
                    if m:
                        try:
                            header[name] = float(m.group(1))
                        except ValueError:
                            header[name] = m.group(1)
                continue
            parts = text.replace(",", " ").split()
            try:
                x, y = float(parts[0]), float(parts[1])
            except (IndexError, ValueError):
                continue                       # a stray line of text in the data block
            xs.append(x)
            ys.append(y)
    if len(xs) < 2:
        raise ValueError(f"{path}: no two-column numeric data was found in it")
    order = np.argsort(np.asarray(xs, float))
    return {"x": np.asarray(xs, float)[order], "y": np.asarray(ys, float)[order],
            "path": path, "name": os.path.basename(path), "header": header}


def resample(curve: dict, grid: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """A measured curve put on the grid a model uses, with the range it does not cover filled.

    A measurement that stops at 20 angstrom cannot say what happens at 30, and the fill value is
    what the model is told instead; 0 is the right one for ``G(r)``, which oscillates about zero.
    """
    x = np.asarray(curve["x"], float)
    y = np.asarray(curve["y"], float)
    grid = np.asarray(grid, float)
    out = np.interp(grid, x, y, left=fill, right=fill)
    return np.where((grid >= x.min()) & (grid <= x.max()), out, fill)


# --------------------------------------------------------------------------- #
# Curves as samples
# --------------------------------------------------------------------------- #
def computed_curves(crystals: Sequence, spec: CurveSpec = CurveSpec(), source: str = "xray",
                    progress=None) -> np.ndarray:
    """``G(r)`` of every structure of a set, one row each, on the spec's grid."""
    out = np.zeros((len(crystals), len(spec.r)))
    for k, c in enumerate(crystals):
        out[k] = pair_distribution(c, spec, source)[1]
        if progress is not None and k % 25 == 0:
            progress(f"Pair distribution {k + 1}/{len(crystals)}", k / max(len(crystals), 1))
    return out


def curve_matrix(curves: Sequence[dict], grid: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """Measured curves on one grid, one row each: the feature matrix of a set of measurements."""
    return np.vstack([resample(c, grid, fill) for c in curves]) if len(curves) \
        else np.zeros((0, len(grid)))


def curve_kernel(train: np.ndarray, test: Optional[np.ndarray] = None,
                 normalize: bool = True) -> np.ndarray:
    """The cosine-normalised linear kernel of a set of curves, computed or measured.

    It is :func:`cmfqsar.invariant.autocorrelation_kernel` with one channel block, which is what
    a total curve is.  Because the grid is the only thing the two sides share, a measured curve
    and a computed one can be compared with it directly.
    """
    from .invariant import autocorrelation_kernel

    A = np.asarray(train, float)
    B = A if test is None else np.asarray(test, float)
    return autocorrelation_kernel(A, B, scale=np.ones(A.shape[1]), n_bins=A.shape[1],
                                  normalize=normalize)


def curve_labels(spec: CurveSpec) -> List[str]:
    """One label per column of a curve matrix: the distance it reports."""
    return [f"G(r) @ {r:.2f} A" for r in spec.r]
