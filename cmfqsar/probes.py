"""
GRID-like probe fields, written as sums of Gaussians so that CMF can keep its algebra.

Goodford's GRID (J. Med. Chem. 28 (1985) 849) describes a molecule by the energy a small probe --
a water, a methyl, a carbonyl oxygen -- would feel at every point around it.  That is a field in
exactly the sense this program means, and it says something the program's own fields do not: not
"there is hydrophobic character here" but "a probe placed here would be bound".  The obstacle is
arithmetic.  Every kernel in CMF is an overlap integral evaluated in closed form over atom pairs,
which works because a field is a sum of atom-centred Gaussians; a Lennard-Jones term is not one,
has no closed-form overlap, and diverges at the atom, so the integral of its square does not even
exist.

The solution is to fit rather than to integrate numerically, and the fitting has to respect
what atom-centred Gaussians can and cannot do.  Three things were learned by measuring, and all
three shape what is implemented here:

* **The repulsive wall cannot be carried.**  Fitting a curve that is +30 kcal/mol at 2.5 A and
  -0.13 at 3.5 A with a few Gaussians produces coefficients of +/- 1e7 that cancel and a
  worst-case error larger than the well it is meant to represent.  The wall is a truncation
  artefact in every grid implementation anyway, and this program already carries it: the steric
  field *is* a Gaussian van der Waals envelope.  So a probe field carries the **attractive** part
  and the repulsion is left where it is already represented.
* **A well at a radius is a shell, and a shell is not a sum of centred Gaussians.**  Atom-centred
  Gaussians are maximal at the atom; representing a peak at 3.5 A needs differences of broad
  Gaussians, which is that same ill-conditioning again.  Each term is therefore *capped* at its
  own optimum -- the Lennard-Jones part below the contact distance, the hydrogen-bond part below
  its preferred separation -- which makes every term monotone in r, and those a sum of Gaussians
  represents naturally.
* **The r^-6 tail is not Gaussian either**, and it is also negligible: beyond 6 A the dispersion
  is below any threshold that matters, so the energy is tapered to zero between 5 and 7 A, which
  is what GRID's own cut-off does.

What comes out is an attraction envelope for a given probe, reproduced to within roughly 10 to 30
per cent in the volume-weighted norm the kernel actually sees -- 15 on the median
combination of probe and atom, 23 on a realistic worst case, and up to 30 where a dispersion
well and an electrostatic repulsion nearly cancel and the whole curve is below 0.1 kcal/mol -- :func:`fit_report` prints the
number for every element and a test holds it to a stated bound.  It is a probe-specific weighting
of dispersion, electrostatics and hydrogen bonding with the right distance dependence; it is not
a reproduction of GRID's numbers, and the directional part of GRID's hydrogen-bond term is not
represented at all.  The place for that, if it is ever wanted, is Gaussians on lone-pair
extension points rather than on the heteroatom, which this framework can carry, since they are
merely more centres with their own weights.

Ion probes, and what they are and are not
-----------------------------------------
Plan item EC7 asks for a cation probe: the field of an ionophore or an electrolyte solvent read as
the energy a lithium, sodium, potassium or magnesium ion would feel in it.  Four such probes are
defined below.  What distinguishes them deserves to be said plainly, because it is less than the
four names suggest.

An ion carries no hydrogen bond and its dispersion well is negligible beside its charge, so for a
singly charged ion :func:`attraction` reduces to the electrostatic term, which is the molecule's
electrostatic potential under this module's distance-dependent screening.  The three +1 probes
therefore differ from one another in **one** quantity, the distance at which that potential is
sampled: the contact distance of the probe with each atom, which for the lithium probe is the
coordination distance measured on the relaxed complexes of ``examples/ether/``.  That is a real
difference -- a potassium ion cannot reach the inside of a small cavity that a lithium ion fits --
and it is the whole of the difference.  The magnesium probe adds its charge of +2, which
multiplies the field by two; every kernel in this program is normalised by the self-overlap, so a
constant factor on a field is invisible to it, and the magnesium probe is in consequence the
lithium probe at a slightly shorter distance.  A model that needs more than this needs
polarisation, which a fixed probe charge cannot carry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .atomtypes import vdw_params

#: Coulomb constant in kcal/mol with distances in angstrom and charges in electrons
COULOMB = 332.0636
#: radial tolerance of the hydrogen-bond term.  Wider than a hydrogen bond really is: a
#: sharper shoulder than this is not representable by the shared basis, and the error it
#: causes (59% against 23% in the volume-weighted norm) is far worse than the broadening.
HB_WIDTH = 0.6
#: where the energy starts being tapered to zero, and where it reaches zero (angstrom)
TAPER_IN, TAPER_OUT = 5.0, 7.0


@dataclass(frozen=True)
class Probe:
    """A small molecule or group whose interaction energy with the ligand becomes a field."""
    name: str
    label: str
    radius: float          # A
    well: float            # kcal/mol
    charge: float = 0.0    # e
    hbond: float = 0.0     # +1 donor, -1 acceptor, 0 neither


#: the dispersion well used for every ion probe, kcal/mol.  Published Lennard-Jones parameters for
#: alkali cations disagree by two orders of magnitude between force fields, and the term does not
#: matter here: at the contact distance of a lithium ion and an ether oxygen the electrostatic
#: term of :func:`attraction` is about -7 kcal/mol and the dispersion well -0.14, so the well sets
#: a correction of two per cent, well inside the error of the Gaussian expansion itself.
ION_WELL = 0.10
#: Measured on the 1,050 relaxed Li(+)-ether complexes of ``examples/ether/`` (plan item EC7):
#: the oxygens within 2.6 A of the ion sit at 2.01 +/- 0.11 A, and ``vdw_params("O")`` gives the
#: oxygen a radius of 1.52 A, so a lithium probe whose contact distance is the measured one has a
#: radius of 0.49 A.  The other three follow from Shannon's six-coordinate ionic radii, which give
#: Na(+) 0.26 A and K(+) 0.62 A more than Li(+) and Mg(2+) 0.04 A less, because no relaxed
#: geometries of those complexes are distributed with the set.
ION_RADII = {"lithium": 0.49, "sodium": 0.75, "potassium": 1.11, "magnesium": 0.45}

#: the probes offered; deliberately few, because a probe nobody can interpret is of no use
PROBES: Dict[str, Probe] = {
    "water": Probe("water", "Water probe (OH2)", radius=1.7, well=0.15, charge=0.0, hbond=0.5),
    "methyl": Probe("methyl", "Methyl probe (C3)", radius=2.0, well=0.10, charge=0.0, hbond=0.0),
    "carbonyl": Probe("carbonyl", "Carbonyl O probe (acceptor)", radius=1.6, well=0.20,
                      charge=-0.4, hbond=-1.0),
    "amine": Probe("amine", "Amine N probe (donor)", radius=1.75, well=0.16, charge=0.4,
                   hbond=1.0),
    "cation": Probe("cation", "Cation probe (+1)", radius=1.5, well=0.10, charge=1.0, hbond=0.0),
    "lithium": Probe("lithium", "Lithium ion probe (Li+)", radius=ION_RADII["lithium"],
                     well=ION_WELL, charge=1.0, hbond=0.0),
    "sodium": Probe("sodium", "Sodium ion probe (Na+)", radius=ION_RADII["sodium"],
                    well=ION_WELL, charge=1.0, hbond=0.0),
    "potassium": Probe("potassium", "Potassium ion probe (K+)", radius=ION_RADII["potassium"],
                       well=ION_WELL, charge=1.0, hbond=0.0),
    "magnesium": Probe("magnesium", "Magnesium ion probe (Mg2+)", radius=ION_RADII["magnesium"],
                       well=ION_WELL, charge=2.0, hbond=0.0),
}

#: the ion probes, in the order of the ionic radius
ION_PROBES = ("magnesium", "lithium", "sodium", "potassium")


def probe_names() -> List[str]:
    return list(PROBES)


def probe(name: str) -> Probe:
    if name not in PROBES:
        raise KeyError(f"unknown probe {name!r}; the ones defined are {', '.join(PROBES)}")
    return PROBES[name]


# --------------------------------------------------------------------------- #
# The energy being fitted
# --------------------------------------------------------------------------- #

def _taper(r: np.ndarray) -> np.ndarray:
    t = np.clip((TAPER_OUT - r) / (TAPER_OUT - TAPER_IN), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)              # smoothstep: value and slope reach zero together


def attraction(r: np.ndarray, p: Probe, symbol: str, charge: float = 0.0,
               hbond: float = 0.0) -> np.ndarray:
    """What `p` would gain by sitting at distance `r` from one atom, as a field can carry it.

    Every term is capped at its own optimum so that the result falls monotonically with distance,
    and the whole is tapered to zero by 7 A.  Positive values -- the repulsive wall -- are not
    represented; see the module docstring for why, and where they are already represented.
    """
    r = np.maximum(np.asarray(r, float), 1e-3)
    R_i, E_i = vdw_params(symbol)
    R_ij = float(R_i) + p.radius
    E_ij = float(np.sqrt(max(E_i, 1e-6) * max(p.well, 1e-6)))
    x = R_ij / np.maximum(r, R_ij)              # capped at the bottom of the well
    e = E_ij * (x ** 12 - 2.0 * x ** 6)
    if p.charge and charge:
        rc = np.maximum(r, R_ij)
        e = e + COULOMB * p.charge * float(charge) / (4.0 * rc * rc)
    if p.hbond and hbond and p.hbond * float(hbond) < 0:
        rh = np.maximum(r, 2.9)
        e = e - 2.0 * abs(p.hbond * float(hbond)) * np.exp(-((rh - 2.9) ** 2) / (2.0 * HB_WIDTH ** 2))
    return e * _taper(r)


# --------------------------------------------------------------------------- #
# The Gaussian expansion
# --------------------------------------------------------------------------- #

def basis_widths(n_terms: int = 4) -> np.ndarray:
    """The widths, shared by every atom and every probe.

    Sharing them is what keeps the kernel affordable: the overlap of two probe fields is a double
    sum over atom pairs *and* over pairs of terms, and with common widths each pair of terms is
    one pass over the distance matrix under the usual Gaussian-product rule, exactly like an
    ordinary field.  Fitting the widths per atom as well is the source of error the module docstring
    describes.
    """
    return np.geomspace(2.0, 0.08, int(max(1, n_terms)))


@dataclass
class GaussianFit:
    """The fitted expansion of one (probe, atom kind) attraction curve."""
    coeffs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    widths: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rel_error: float = float("nan")    # volume-weighted and relative: what the kernel sees
    depth: float = float("nan")        # the deepest attraction, kcal/mol

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, float)
        return np.sum(self.coeffs[None, :] * np.exp(-0.5 * self.widths[None, :] * r[:, None] ** 2),
                      axis=1)

    def describe(self) -> str:
        terms = ", ".join(f"{c:+.3f} exp(-{b:.3f} r^2/2)" for c, b in zip(self.coeffs, self.widths))
        return (f"{terms}   (deepest {self.depth:+.3f} kcal/mol, relative error "
                f"{self.rel_error:.1%})")


def _solve(r: np.ndarray, e: np.ndarray, widths: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Coefficients by ridge-regularised least squares, weighted by the volume element."""
    w = np.maximum(r, 1e-6)                     # squaring the residual recovers the r^2 of dV
    B = np.exp(-0.5 * widths[None, :] * r[:, None] ** 2) * w[:, None]
    A = np.vstack([B, np.sqrt(ridge) * np.eye(len(widths))])
    y = np.concatenate([e * w, np.zeros(len(widths))])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return np.asarray(c, float)


_FIT_CACHE: Dict[Tuple, GaussianFit] = {}


def fit(p: Probe, symbol: str, charge: float = 0.0, hbond: float = 0.0,
        n_terms: int = 4) -> GaussianFit:
    """The expansion of one atom's contribution to a probe field, memoised.

    `charge` and `hbond` are rounded into the cache key: a fit per distinct partial charge would
    be a fit per atom, which is slow and useless, since the electrostatic term is linear in the
    charge and the rounding moves the energy far less than the capping does.
    """
    q, hb = round(float(charge), 2), round(float(hbond), 1)
    key = (p.name, str(symbol), q, hb, int(n_terms))
    hit = _FIT_CACHE.get(key)
    if hit is not None:
        return hit
    widths = basis_widths(n_terms)
    r = np.linspace(0.2, 9.0, 700)
    e = attraction(r, p, symbol, charge=q, hbond=hb)
    c = _solve(r, e, widths)
    approx = np.sum(c[None, :] * np.exp(-0.5 * widths[None, :] * r[:, None] ** 2), axis=1)
    vol = r ** 2
    rel = float(np.sqrt(np.sum(vol * (approx - e) ** 2) / max(np.sum(vol * e ** 2), 1e-30)))
    out = GaussianFit(coeffs=c, widths=widths, rel_error=rel, depth=float(np.min(e)))
    _FIT_CACHE[key] = out
    return out


def coefficients(symbols: Sequence[str], charges: Sequence[float], hbond: Sequence[float],
                 p: Probe, n_terms: int = 4) -> np.ndarray:
    """(n_atoms, n_terms) expansion coefficients of one molecule's probe field."""
    out = np.zeros((len(symbols), int(n_terms)))
    for i, (s, q, hb) in enumerate(zip(symbols, charges, hbond)):
        out[i] = fit(p, str(s), charge=float(q), hbond=float(hb), n_terms=n_terms).coeffs
    return out


def fit_report(name: str = "water", symbols: Sequence[str] = ("C", "N", "O", "S", "Cl", "H"),
               n_terms: int = 4, charge: float = 0.0, hbond: float = 0.0) -> str:
    """How faithfully the expansion reproduces the attraction, element by element."""
    p = probe(name)
    lines = [f"{p.label}, {n_terms} Gaussians per atom:"]
    for s in symbols:
        lines.append(f"  {s:3s} "
                     f"{fit(p, s, charge=charge, hbond=hbond, n_terms=n_terms).describe()}")
    return "\n".join(lines)
