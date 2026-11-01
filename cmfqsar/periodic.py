"""Continuous molecular fields of crystals: the periodic field autocorrelation.

The autocorrelation kernel of ``invariant.py`` compares two molecules by the distance
distributions of their field weights, which is the real-space form of the orientation-averaged
product of their typed structure factors (the Debye scattering equation).  A crystal is the same
object with three differences, and this module is those differences:

* **The pair sum runs over a neighbour list.**  A molecule has finitely many pairs; a crystal has
  one atom per cell times every neighbour out to a cutoff radius, periodic images included, and
  the sum is divided by the number of atoms in the cell so that the descriptor is intensive: a
  supercell describes the same material, and gives the same channels.

* **The uniform background is removed.**  In an infinite solid the number of pairs at distance r
  grows as r^2, the same for every structure, and dominates everything else at long distance.  What
  the Faber-Ziman partial structure factors keep is the deviation from a uniform distribution of
  each field over the cell volume, and that is what is subtracted here, pair of field types by
  pair of field types.  The l = m self term stays, as it does in the molecular descriptor: it
  carries the composition, the mean of w_f w_g over the atoms.

* **The cutoff is tapered.**  A hard cutoff produces oscillations; the last ``taper`` angstrom
  before ``r_max`` are weighted by a cosine that falls smoothly to zero, so that a pair leaving the
  sphere leaves the descriptor continuously.

Everything else is shared: the Gaussian along the distance axis, the pair-of-field-types
channels, the per-block scaling and cosine normalisation of the kernel, the channel labels, and
the learners.  The descriptor is invariant to rotation, translation, the choice of origin and of
unit cell by construction, and cannot see what any pair-distance distribution cannot see:
enantiomorphs and homometric structures look the same, and so does anything beyond ``r_max``.

Fields for inorganic solids
---------------------------
The classical hydrophobic and hydrogen-bond fields mean nothing in an oxide.  What carries over
is the *electrostatic* field with oxidation states as weights, the *steric* field with the atomic
volume (the cube of the covalent radius), the *electronegativity*, the atomic *mass* or its inverse
square root (``prop:mass``, ``prop:inverse_sqrt_mass``) as property fields, and above all the
*continuous indicator fields* of the 2015 paper, one per element: their pair products are
the partial structure factors of each element pair, which is what a diffraction experiment
measures.  :func:`inorganic_fieldset` builds such a set and :func:`crystal_fields` evaluates the
weights for one structure.  Any per-atom array a structure file carries (Bader charges, magnetic
moments, ...) can be a field as ``prop:<name>``.

Partial occupancy (plan item M2)
-------------------------------
A disordered site -- a Li site occupied in 54 % of the cells, a site shared by Zr and Ta -- is
a configurational ensemble, and the kernels of this program treat an ensemble by its mean field
(the mean-map kernel of :func:`cmfqsar.fields.combine_members`).  When the sites are occupied
independently of each other, the mean field over all configurations is exact and costs nothing:
the weight of every field at a site is the occupancy-weighted mean of the weights of the species
that share it, ``w_f = sum_s p_s w_f(s)``, which is what :func:`crystal_fields` computes when a
structure carries its site species (:func:`site_species`; :func:`read_disordered` reads a CIF
with partial occupancies through pymatgen).  The overlap kernel is bilinear in the fields, so
the kernel of two mean fields is the mean of the kernel over the two ensembles, exactly.  The
periodic autocorrelation is quadratic in one field, so its mean over the ensemble differs from
the descriptor of the mean field where a site meets itself: in the self term at zero distance,
in the pairs of a site with its own periodic images and in the background, where the mean of a
product is not the product of the means,
``E[w_f w_g] = sum_s p_s w_f(s) w_g(s)`` on each site.  :class:`CrystalFields` carries those
second moments and :func:`periodic_autocorrelation` uses them, so that the descriptor is the
exact mean over the configurations as well, per expected atom of the cell.  A structure without disorder carries none of this and goes through the
same code unchanged.

Layers and chains (plan item M1)
--------------------------------
A structure periodic in two directions (a monolayer, a slab) or in one (a polymer chain, a
nanowire) goes through the same neighbour list, which respects the periodicity flags of the
structure (``pbc``), so no pair is counted across the vacuum whatever its width.  Two things do
depend on the number of periodic directions, and both read the flags:

* **The uniform background.**  What is subtracted is the pair histogram of the same weights
  spread uniformly over the *periodic* directions only; across the others every atom keeps its
  place.  For a crystal that is the shell integral over the cell volume; for a layer the count
  of pairs at distance r between two atoms a height h apart grows as a ring, 2 pi r dr over the
  cell area, for r >= h; for a chain as 2 r dr / sqrt(r^2 - t^2) over the repeat length, with t
  the separation from the axis.  :func:`_background_profile` holds those integrals per channel
  and :attr:`CrystalFields.measure` the volume, area or length that divides them.  The
  descriptor then does not change with the vacuum, with an in-plane supercell or with a shift
  of the layer along the normal.

* **The height channels of a layer.**  The distance descriptor cannot see the normal.  With
  ``z_max > 0`` in the spec the descriptor of a layer is resolved by the in-plane distance and
  the heights of both atoms above the layer's mean plane, as the surface kernel of
  ``invariant.py`` resolves a molecule above a surface, with the background removed per unit
  area.  A free-standing layer has no physical "up", so the descriptor is averaged over the two
  faces unless the spec says ``oriented``; an oriented spec keeps the file's own up (the third
  lattice vector) and tells a Janus layer from its mirror image.

:func:`centre_layer` gathers a layer that a database file leaves split across the cell boundary.
The images of the periodic overlap kernel (:func:`image_translations`) read the same flags, so
a layer is never compared with its own copy across the vacuum.

ASE is needed for the neighbour list and to read structure files; pymatgen is needed for the
oxidation-state guess, the electronegativities and the reading of partial occupancies and
nothing else.  Both are imported lazily, so that the rest of the program does not depend on
them.
"""
from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet, FieldSpec
from .invariant import block_sizes, channel_labels, field_pairs

INORGANIC_FIELDS: Tuple[str, ...] = ("electrostatic", "steric", "prop:electronegativity")
OTHER_ELEMENT = "other"          # the indicator field that collects the elements not listed


# --------------------------------------------------------------------------- #
# The descriptor
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PeriodicSpec:
    """The distance grid, cutoff, taper and background treatment of the periodic autocorrelation.

    The grid is laid out like :class:`invariant.AutocorrelationSpec`: channels at 0, step, 2 step,
    ... up to ``r_max``, each a Gaussian of width ``sigma`` along the distance axis.  ``chirality``
    exists only so that the block and label helpers of ``invariant.py`` accept this spec; the
    two-point descriptor of a crystal has no signed block.
    """
    r_max: float = 10.0
    step: float = 0.5
    sigma: float = 0.5
    include_self: bool = True        # the l = m terms: the composition channel at d = 0
    background: bool = True          # subtract the uniform-density term of every field pair
    taper: float = 2.0               # width of the cosine taper below r_max (0: hard cutoff)
    chirality: bool = False          # always False; see the class docstring
    #: height channels of a layer (plan item M1): with ``z_max > 0`` the descriptor of a
    #: structure periodic in two directions is resolved by the in-plane distance and the heights
    #: of both atoms above the mean plane of the layer, on a grid from -z_max to z_max every
    #: z_step; 0 keeps the plain distance descriptor
    z_max: float = 0.0
    z_step: float = 1.0
    z_sigma: float = 0.0             # width of the height channels; 0 takes z_step
    #: True when the normal has a physical direction (the third lattice vector of the file is
    #: "up"); False averages the descriptor over the two faces of the layer
    oriented: bool = False

    @property
    def d_max(self) -> float:
        return self.r_max

    @property
    def has_heights(self) -> bool:
        return self.z_max > 0

    @property
    def heights(self) -> np.ndarray:
        """The height channels, symmetric about the mean plane; one channel at 0 without them."""
        if not self.has_heights:
            return np.zeros(1)
        n = max(int(round(self.z_max / self.z_step)), 0)
        return np.arange(-n, n + 1, dtype=float) * self.z_step

    @property
    def height_sigma(self) -> float:
        return float(self.z_sigma) if self.z_sigma > 0 else float(self.z_step)

    @property
    def centres(self) -> np.ndarray:
        n = max(int(round(self.r_max / self.step)) + 1, 1)
        return np.arange(n, dtype=float) * self.step

    @property
    def n_bins(self) -> int:
        """Channels per pair of field types: the distances, times the heights squared for a
        layer with height channels."""
        n = len(self.centres)
        return n * len(self.heights) ** 2 if self.has_heights else n

    def window(self, r: np.ndarray) -> np.ndarray:
        """1 inside the sphere, a half-cosine over the last ``taper`` angstrom, 0 beyond."""
        r = np.asarray(r, float)
        w = np.where(r < self.r_max, 1.0, 0.0)
        if self.taper > 0:
            r0 = self.r_max - self.taper
            inside = (r > r0) & (r < self.r_max)
            w = np.where(inside, 0.5 * (1.0 + np.cos(np.pi * (r - r0) / self.taper)), w)
        return w

    def shell_integral(self, n_points: int = 4000) -> np.ndarray:
        """``∫ 4π r² exp(-(r - d_k)² / 2σ²) w(r) dr`` for every channel d_k.

        The pair count a uniform distribution of unit density puts into each channel.  It depends
        on the grid, the width and the window only, so it is computed once per spec.
        """
        r = np.linspace(0.0, self.r_max, int(n_points))
        g = np.exp(-((r[:, None] - self.centres[None, :]) ** 2) / (2.0 * self.sigma ** 2))
        f = 4.0 * np.pi * r[:, None] ** 2 * g * self.window(r)[:, None]
        return np.trapezoid(f, r, axis=0)

    def ring_integral(self, n_points: int = 4000) -> np.ndarray:
        """``∫ 2π ρ exp(-(ρ - d_k)² / 2σ²) w(ρ) dρ`` for every channel: the in-plane analogue of
        the shell integral, the pair count a unit density spread over a plane puts into each
        channel of the in-plane distance."""
        r = np.linspace(0.0, self.r_max, int(n_points))
        g = np.exp(-((r[:, None] - self.centres[None, :]) ** 2) / (2.0 * self.sigma ** 2))
        f = 2.0 * np.pi * r[:, None] * g * self.window(r)[:, None]
        return np.trapezoid(f, r, axis=0)


@dataclass
class CrystalFields:
    """Atom-level data of one periodic structure: what :class:`fields.MoleculeFields` is to a
    molecule.  It carries what an :class:`fields.AtomTable` and the periodic kernel need, and
    the unit cell."""
    name: str
    coords: np.ndarray            # (N, 3) Cartesian positions of the atoms of one cell
    weights: np.ndarray           # (N, F) field weights
    radii: np.ndarray             # (N,) covalent radii
    symbols: List[str]
    cell: np.ndarray              # (3, 3) lattice vectors as rows
    pbc: Tuple[bool, bool, bool] = (True, True, True)
    charges: Optional[np.ndarray] = None
    charge_source: str = ""
    atom_types: List[str] = field(default_factory=list)
    bonds: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=int))
    molblock: str = ""
    n_conformers: int = 1
    conformer_weights: Optional[np.ndarray] = None
    atoms_per_conformer: int = 0
    descriptors: Optional[np.ndarray] = None
    probe_coeffs: Dict[str, np.ndarray] = field(default_factory=dict)
    #: total occupancy of every site, or None for an ordered structure (plan item M2)
    occupancy: Optional[np.ndarray] = None
    #: the species sharing every site with their occupancies, or None
    site_species: Optional[List[Dict[str, float]]] = None
    #: (N, F, F) the mean of w_f w_g over the species of every site, or None: the self term of
    #: the periodic autocorrelation needs it, because the mean of a product is not the product
    #: of the means on a disordered site
    second_moments: Optional[np.ndarray] = None
    #: for a difference field (:func:`defect_fields`): the number of sites that changed, which
    #: the rows alone do not give (a vacancy and an added atom are one row each, a substitution
    #: two rows on one site)
    n_defects: Optional[int] = None
    #: the name of the host this structure is a difference from (:func:`defect_fields`), or ""
    #: for a structure whose field is its own.  A difference carries only the sites that changed,
    #: with the sign of the change, and is an exact representation under a bilinear kernel.
    relative_to: str = ""
    #: the site this record stands for (plan item M5), or None for the whole cell.  The periodic
    #: autocorrelation of a record with a centre is the profile of that site alone
    #: (:func:`cmfqsar.sites.site_autocorrelation`), so a site is a sample like any structure.
    centre: Optional[int] = None

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def is_ensemble(self) -> bool:
        return False

    @property
    def is_disordered(self) -> bool:
        return self.site_species is not None

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def n_periodic(self) -> int:
        """How many lattice directions are periodic: 3 for a crystal, 2 for a layer, 1 for a
        chain, 0 for a molecule in a box."""
        return int(sum(bool(p) for p in self.pbc))

    @property
    def measure(self) -> Optional[float]:
        """The size of the cell along its periodic directions: the volume of a crystal, the area
        of a layer, the repeat length of a chain; None for a structure with no periodic
        direction, which has no uniform background to remove."""
        return periodic_measure(self.cell, self.pbc)

    def to_atoms(self):
        from ase import Atoms
        return Atoms(symbols=list(self.symbols), positions=np.asarray(self.coords, float),
                     cell=np.asarray(self.cell, float), pbc=self.pbc)


@functools.lru_cache(maxsize=32)
def _shell_integral(spec: PeriodicSpec) -> np.ndarray:
    """The shell integral of a spec, computed once: it does not depend on the structure."""
    out = spec.shell_integral()
    out.flags.writeable = False
    return out


@functools.lru_cache(maxsize=32)
def _ring_integral(spec: PeriodicSpec) -> np.ndarray:
    out = spec.ring_integral()
    out.flags.writeable = False
    return out


def _pbc_of(crystal) -> Tuple[bool, bool, bool]:
    pbc = getattr(crystal, "pbc", (True, True, True))
    return tuple(bool(p) for p in pbc)


def periodic_measure(cell, pbc) -> Optional[float]:
    """The volume, area or repeat length of a cell along its periodic directions; None when
    no direction is periodic."""
    A = np.asarray(cell, float).reshape(3, 3)
    axes = [k for k in range(3) if pbc[k]]
    if len(axes) == 3:
        return float(abs(np.linalg.det(A)))
    if len(axes) == 2:
        return float(np.linalg.norm(np.cross(A[axes[0]], A[axes[1]])))
    if len(axes) == 1:
        return float(np.linalg.norm(A[axes[0]]))
    return None


def layer_normal(cell, pbc) -> np.ndarray:
    """The unit normal of a structure periodic in two directions, pointing along the
    non-periodic lattice vector: the file's own "up"."""
    A = np.asarray(cell, float).reshape(3, 3)
    axes = [k for k in range(3) if pbc[k]]
    if len(axes) != 2:
        raise ValueError(f"a layer is periodic in two directions; this structure is periodic in "
                         f"{len(axes)} (pbc={tuple(bool(p) for p in pbc)})")
    open_axis = [k for k in range(3) if not pbc[k]][0]
    n = np.cross(A[axes[0]], A[axes[1]])
    if float(n @ A[open_axis]) < 0:
        n = -n
    return n / max(float(np.linalg.norm(n)), 1e-12)


def _normal_separations(coords, cell, pbc) -> np.ndarray:
    """The separation of every ordered pair of atoms across the non-periodic directions, (N, N):
    zero for a crystal, the height difference for a layer, the distance between the two atoms'
    projections on the plane normal to the chain for a chain."""
    X = np.asarray(coords, float)
    A = np.asarray(cell, float).reshape(3, 3)
    axes = [k for k in range(3) if pbc[k]]
    if len(axes) == 3:
        return np.zeros((len(X), len(X)))
    if len(axes) == 2:
        z = X @ layer_normal(cell, pbc)
        return np.abs(z[:, None] - z[None, :])
    if len(axes) == 1:
        a = A[axes[0]] / max(float(np.linalg.norm(A[axes[0]])), 1e-12)
        D = X[:, None, :] - X[None, :, :]
        D = D - (D @ a)[..., None] * a[None, None, :]
        return np.linalg.norm(D, axis=-1)
    return np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)


@functools.lru_cache(maxsize=32)
def _background_profile(spec: PeriodicSpec, n_periodic: int) -> Tuple[np.ndarray, np.ndarray]:
    """``(h, J)``: the pair count that a unit density spread uniformly over the periodic
    directions puts into every channel, for a pair of atoms separated by ``h`` across the
    non-periodic ones.  ``J[a, k]`` belongs to ``h[a]`` and channel ``k``.

    Three periodic directions: ``h`` is 0 alone and ``J`` the shell integral, ``∫ 4π r² G_k w dr``.
    Two: the pairs at distance r lie on a ring of radius sqrt(r² - h²) in the plane, so the
    count is 2π r dr for r >= h and ``J(h) = ∫_h^{r_max} 2π r G_k w dr``.  One: the two points of
    the axis at distance r are ±sqrt(r² - h²) along it, so the count is 2 r dr / sqrt(r² - h²),
    which is ``2 du`` in the axial coordinate u: ``J(h) = ∫_0^{sqrt(r_max² - h²)} 2 G_k(sqrt(u² + h²)) w du``.
    """
    if n_periodic == 3:
        J = np.array(spec.shell_integral())[None, :]
        h = np.zeros(1)
    elif n_periodic == 2:
        n_points = 4000
        r = np.linspace(0.0, spec.r_max, n_points)
        g = (np.exp(-((r[:, None] - spec.centres[None, :]) ** 2) / (2.0 * spec.sigma ** 2))
             * spec.window(r)[:, None])
        f = 2.0 * np.pi * r[:, None] * g
        seg = 0.5 * (f[1:] + f[:-1]) * (r[1] - r[0])
        tail = np.concatenate([seg[::-1].cumsum(axis=0)[::-1], np.zeros((1, f.shape[1]))], axis=0)
        h, J = r, tail
    elif n_periodic == 1:
        h = np.linspace(0.0, spec.r_max, 401)
        J = np.zeros((len(h), spec.centres.size))
        for a, t in enumerate(h):
            U = float(np.sqrt(max(spec.r_max ** 2 - t ** 2, 0.0)))
            if U <= 0.0:
                continue
            u = np.linspace(0.0, U, 2000)
            rr = np.sqrt(u * u + t * t)
            gg = (np.exp(-((rr[:, None] - spec.centres[None, :]) ** 2) / (2.0 * spec.sigma ** 2))
                  * spec.window(rr)[:, None])
            J[a] = np.trapezoid(2.0 * gg, u, axis=0)
    else:
        raise ValueError("a structure with no periodic direction has no uniform background")
    h = np.array(h, float)
    J = np.array(J, float)
    h.flags.writeable = False
    J.flags.writeable = False
    return h, J


def _background_rows(crystal, spec: PeriodicSpec, W: np.ndarray, cov: np.ndarray,
                     fi: np.ndarray, gi: np.ndarray, mult: np.ndarray) -> Optional[np.ndarray]:
    """The uniform background of every pair of field types, (n_pairs, n_bins), before the
    division by the number of atoms; None for a structure with no periodic direction."""
    pbc = _pbc_of(crystal)
    measure = periodic_measure(crystal.cell, pbc)
    if measure is None or measure <= 0:
        return None
    n_periodic = int(sum(pbc))
    h, J = _background_profile(spec, n_periodic)
    if n_periodic == 3:
        S = W.sum(axis=0)
        return (mult * (S[fi] * S[gi] + cov[fi, gi]) / measure)[:, None] * J[0][None, :]
    H = _normal_separations(crystal.coords, crystal.cell, pbc)
    n_fields = W.shape[1]
    T = np.zeros((n_fields, J.shape[1], n_fields))
    flat = H.ravel()
    for k in range(J.shape[1]):
        Jk = np.interp(flat, h, J[:, k]).reshape(H.shape)
        T[:, k, :] = W.T @ Jk @ W
    # H is symmetric, so the two orders of a pair of different types are equal: mult counts both
    rows = T[fi, :, gi] + cov[fi, gi][:, None] * J[0][None, :]
    return mult[:, None] * rows / measure


def centre_layer(atoms):
    """A copy of `atoms` with the structure gathered and centred along every non-periodic
    lattice direction.

    A layer read from a database may sit across the cell boundary along the normal, one part at
    the bottom of the cell and the rest at the top, which breaks the heights and the neighbour
    list of an open direction.  Along each non-periodic direction the atoms are shifted so that
    the widest empty gap of the fractional coordinate, taken around the cycle, straddles the
    boundary, and the structure sits at the middle of the cell.  A structure periodic in every
    direction is returned unchanged.
    """
    out = atoms.copy()
    pbc = [bool(p) for p in out.get_pbc()]
    if all(pbc) or len(out) == 0:
        return out
    frac = np.array(out.get_scaled_positions(wrap=False), float)
    for k in range(3):
        if pbc[k]:
            continue
        f = np.mod(frac[:, k], 1.0)
        order = np.argsort(f)
        fs = f[order]
        gaps = np.diff(np.concatenate([fs, [fs[0] + 1.0]]))
        m = int(np.argmax(gaps))
        start = fs[(m + 1) % len(fs)] if m + 1 < len(fs) else fs[0]
        shifted = np.mod(f - start, 1.0)
        frac[:, k] = shifted + 0.5 * gaps[m]
    out.set_scaled_positions(frac)
    return out


def _neighbour_pairs(crystal: CrystalFields, r_max: float):
    """Ordered atom pairs (i, j, d) with 0 < d < r_max, periodic images included."""
    from ase.neighborlist import neighbor_list
    atoms = crystal.to_atoms()
    i, j, d = neighbor_list("ijd", atoms, float(r_max))
    return np.asarray(i, int), np.asarray(j, int), np.asarray(d, float)


def periodic_autocorrelation(crystal: CrystalFields, spec: PeriodicSpec = PeriodicSpec(),
                             pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """Periodic field autocorrelation of one structure, flattened: (n_pairs * n_bins,).

    For a pair of field types (f, g) and a channel at distance d_k,     ::     ::

        Phi_fg(d_k) = (1/N) [ sum_{l in cell} sum_{m: r_lm < r_max} w_lf w_mg
                                     exp(-(r_lm - d_k)^2 / 2 sigma^2) w(r_lm)
                               - S_f S_g / V * I_k ],

    with N the atoms in the cell, S_f the sum of the weights of field f over the cell, V its
    volume and I_k the shell integral of the spec.  The sum runs over ordered pairs, so a pair of
    different field types is counted in both orders, as the molecular descriptor counts it; the
    background of such a pair is accordingly 2 S_f S_g / V * I_k.

    For a structure periodic in fewer than three directions the pair sum is the same (the
    neighbour list respects the periodicity flags) and the background is the one of a uniform
    spread over the periodic directions alone,
    ``(1/M) sum_{l,m} w_lf w_mg J_k(h_lm)``, with M the area or repeat length of the cell, h_lm
    the separation of the pair across the non-periodic directions and J_k the ring or axis
    integral of :func:`_background_profile`; it reduces to the form above for a crystal.  With
    height channels in the spec (``z_max > 0``) a layer gets :func:`slab_autocorrelation`.
    """
    if getattr(crystal, "centre", None) is not None:
        # a site record (plan item M5): the sum over pairs belongs to the site that stands first
        from .sites import site_autocorrelation
        return site_autocorrelation(crystal, spec, pairs)[int(crystal.centre)]
    if spec.has_heights:
        return slab_autocorrelation(crystal, spec, pairs)
    W = np.asarray(crystal.weights, float)
    n_atoms, n_fields = W.shape
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    centres = spec.centres
    if n_atoms == 0:
        return np.zeros(len(pairs) * len(centres))
    i, j, d = _neighbour_pairs(crystal, spec.r_max)
    inv = 2.0 * spec.sigma ** 2
    if len(d):
        B = np.exp(-((d[:, None] - centres[None, :]) ** 2) / inv) * spec.window(d)[:, None]
    else:
        B = np.zeros((0, len(centres)))
    self_row = np.exp(-(centres ** 2) / inv) if spec.include_self else np.zeros(len(centres))
    # T[f, k, g] = sum over ordered pairs of w_if B_pk w_jg, for every pair of field types at once
    T = np.zeros((n_fields, len(centres), n_fields))
    if len(d):
        Wi, Wj = W[i], W[j]
        for k in range(len(centres)):                      # one BLAS product per channel
            T[:, k, :] = (Wi * B[:, k:k + 1]).T @ Wj
        if crystal.second_moments is not None:
            # a site and its own periodic image are occupied together or not at all, so that
            # pair carries E[w_f w_g] of the site and not the product of the means (M2)
            own = i == j
            if own.any():
                Mo = np.asarray(crystal.second_moments, float)[i[own]]
                D = Mo - W[i[own]][:, :, None] * W[i[own]][:, None, :]
                T += np.einsum("pfg,pk->fkg", D, B[own])
    if crystal.second_moments is not None:
        # disordered sites (M2): the self term and the background need E[w_f w_g] on each site,
        # not the product of the means, and so does the pair of a site with its own periodic
        # image (handled above); the pair term between different sites is exact as it is,
        # because independent sites make the mean of a product the product of the means.
        # The descriptor is per expected atom, so that a supercell of the same disorder gives
        # the same channels, as it does for an ordered cell.
        G = np.asarray(crystal.second_moments, float).sum(axis=0)
        cov = G - W.T @ W                                  # sum over sites of the covariance
        n_norm = float(np.sum(crystal.occupancy)) if crystal.occupancy is not None else n_atoms
    else:
        G = W.T @ W                                        # the self terms, sum_l w_lf w_lg
        cov = np.zeros_like(G)
        n_norm = n_atoms
    fi = np.array([f for f, _ in pairs], int)
    gi = np.array([g for _, g in pairs], int)
    diag = fi == gi
    mult = np.where(diag, 1.0, 2.0)                        # a pair of different types, both ways
    rows = T[fi, :, gi] + np.where(diag[:, None], 0.0, T[gi, :, fi])
    rows = rows + (mult * G[fi, gi])[:, None] * self_row[None, :]
    if spec.background:
        bg = _background_rows(crystal, spec, W, cov, fi, gi, mult)
        if bg is not None:
            rows = rows - bg
    return (rows / max(n_norm, 1e-12)).ravel()


def slab_autocorrelation(crystal: CrystalFields, spec: PeriodicSpec,
                         pairs: Optional[Sequence[Tuple[int, int]]] = None) -> np.ndarray:
    """The descriptor of a layer resolved by height: (n_pairs * n_rho * n_z * n_z,) flattened.

    For a pair of field types (f, g), an in-plane distance channel rho_k and two height
    channels z_a, z_b,     ::

        Phi_fg(rho_k, z_a, z_b) = (1/N) [ sum_l sum_{m: rho_lm < r_max} w_lf w_mg
                                              exp(-(rho_lm - rho_k)^2 / 2 sigma^2) w(rho_lm)
                                              Z_a(z_l) Z_b(z_m)
                                        - (R_k / A) sum_l w_lf Z_a(z_l) sum_m w_mg Z_b(z_m) ],

    with rho_lm the in-plane distance of atom l to atom or image m, z_l the height of atom l
    above the mean plane of the layer, Z_a a Gaussian of width ``height_sigma`` about the
    height channel z_a, R_k the ring integral of the spec and A the area of the cell.  The
    background is exact: a uniform spread over the plane puts 2 pi rho d rho / A pairs at every
    in-plane distance whatever the heights.  A pair of different field types is summed in both
    orders, f at z_a with g at z_b and g at z_a with f at z_b, as the surface kernel of
    ``invariant.py`` sums it; the l = m terms are kept where ``include_self`` is set; a
    disordered site enters with its second moments as in :func:`periodic_autocorrelation`.
    Unless the spec is ``oriented`` the result is the mean over the two faces of the layer,
    which is the same descriptor with both height axes reversed.  It is invariant to the
    choice of cell, origin and in-plane rotation, to the vacuum and to a shift of the layer
    along the normal, and it carries strictly more than the distance descriptor: the distance
    is a function of the in-plane distance and the two heights.
    """
    W = np.asarray(crystal.weights, float)
    n_atoms, n_fields = W.shape
    pbc = _pbc_of(crystal)
    if sum(pbc) != 2:
        raise ValueError(f"{crystal.name or 'this structure'} is periodic in {sum(pbc)} "
                         f"directions; the height channels are defined for a layer, periodic in "
                         f"two: set the periodicity flags so that the normal is the open direction")
    centres, heights = spec.centres, spec.heights
    nb, nz = len(centres), len(heights)
    pairs = list(pairs) if pairs is not None else field_pairs(n_fields)
    if n_atoms == 0:
        return np.zeros(len(pairs) * nb * nz * nz)
    n = layer_normal(crystal.cell, pbc)
    X = np.asarray(crystal.coords, float)
    occ = (np.asarray(crystal.occupancy, float) if crystal.occupancy is not None
           else np.ones(n_atoms))
    z = X @ n
    z = z - float(np.sum(occ * z) / max(float(np.sum(occ)), 1e-12))
    thickness = float(z.max() - z.min())
    # every pair inside the in-plane cutoff is within r_max + thickness in space
    from ase.neighborlist import neighbor_list
    i, j, D = neighbor_list("ijD", crystal.to_atoms(), float(spec.r_max) + thickness)
    i, j = np.asarray(i, int), np.asarray(j, int)
    D = np.asarray(D, float).reshape(-1, 3)
    rho = np.linalg.norm(D - (D @ n)[:, None] * n[None, :], axis=1)
    keep = rho < spec.r_max
    i, j, rho = i[keep], j[keep], rho[keep]
    inv = 2.0 * spec.sigma ** 2
    sz = spec.height_sigma
    Z = np.exp(-((z[:, None] - heights[None, :]) ** 2) / (2.0 * sz ** 2))          # (N, nz)
    U = (W[:, :, None] * Z[:, None, :]).reshape(n_atoms, n_fields * nz)           # (field, height)
    if len(rho):
        B = np.exp(-((rho[:, None] - centres[None, :]) ** 2) / inv) * spec.window(rho)[:, None]
    else:
        B = np.zeros((0, nb))
    self_row = np.exp(-(centres ** 2) / inv) if spec.include_self else np.zeros(nb)
    width = n_fields * nz
    T = np.zeros((nb, width, width))
    for k in range(nb):                                       # one BLAS product per channel
        T[k] = (U[i] * B[:, k:k + 1]).T @ U[j]
    if crystal.second_moments is not None:
        # disordered sites (M2): a site meets itself, in the self term, in the pairs with its
        # own images and in the background, with E[w_f w_g] rather than the product of means
        E = np.asarray(crystal.second_moments, float)
        Dm = E - W[:, :, None] * W[:, None, :]
        own = i == j
        if own.any():
            for l in np.unique(i[own]):
                b_l = B[own & (i == l)].sum(axis=0)                           # (nb,)
                C_l = np.einsum("fg,a,b->fagb", Dm[l], Z[l], Z[l]).reshape(width, width)
                T += b_l[:, None, None] * C_l[None]
        G = np.einsum("lfg,la,lb->fagb", E, Z, Z).reshape(width, width)
        C_self = np.einsum("lfg,la,lb->fagb", Dm, Z, Z).reshape(width, width)
        n_norm = float(np.sum(occ))
    else:
        G = U.T @ U
        C_self = None
        n_norm = float(n_atoms)
    T += self_row[:, None, None] * G[None]
    if spec.background:
        area = periodic_measure(crystal.cell, pbc)
        if area and area > 0:
            s = U.sum(axis=0)
            bg = np.outer(s, s) + (C_self if C_self is not None else 0.0)
            T -= (_ring_integral(spec)[:, None, None] / area) * bg[None]
    T5 = T.reshape(nb, n_fields, nz, n_fields, nz)                           # [k, f, a, g, b]
    out = np.empty((len(pairs), nb, nz, nz))
    for p, (f, g) in enumerate(pairs):
        t = T5[:, f, :, g, :]
        if f != g:
            t = t + T5[:, g, :, f, :]
        out[p] = t
    if not spec.oriented:
        out = 0.5 * (out + out[:, :, ::-1, ::-1])
    return (out / max(n_norm, 1e-12)).ravel()


def periodic_feature_matrix(crystals: Sequence[CrystalFields],
                            spec: PeriodicSpec = PeriodicSpec(),
                            n_fields: Optional[int] = None, progress=None) -> np.ndarray:
    """Periodic autocorrelation features of a set of structures, one row each."""
    if not len(crystals):
        return np.zeros((0, 0))
    n_fields = n_fields if n_fields is not None else crystals[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    width = sum(block_sizes(spec, n_fields))
    out = np.zeros((len(crystals), width))
    for k, c in enumerate(crystals):
        if c.weights.shape[1] != n_fields:
            raise ValueError(
                f"{c.name or f'structure {k}'} carries {c.weights.shape[1]} fields where the set "
                f"carries {n_fields}: build the fields of every structure from one field set.")
        out[k] = periodic_autocorrelation(c, spec, pairs)
        if progress is not None and k % 25 == 0:
            progress(f"Periodic autocorrelation {k + 1}/{len(crystals)}", k / max(len(crystals), 1))
    return out


def periodic_block_sizes(spec: PeriodicSpec, n_fields: int) -> List[int]:
    return block_sizes(spec, n_fields)


def periodic_channel_labels(fieldset: FieldSet, spec: PeriodicSpec) -> List[str]:
    """One label per feature: which pair of field types, at which distance; for a layer with
    height channels, at which in-plane distance and which two heights; for the reciprocal-space
    channels of plan item M6, at which reciprocal length."""
    from .reciprocal import ReciprocalSpec, reciprocal_channel_labels
    if isinstance(spec, ReciprocalSpec):
        return reciprocal_channel_labels(fieldset, spec)
    if not spec.has_heights:
        return channel_labels(fieldset, spec)
    names = [s.label for s in fieldset.specs]
    out = []
    for f, g in field_pairs(len(names)):
        pair = names[f] if f == g else f"{names[f]} x {names[g]}"
        for d in spec.centres:
            for zi in spec.heights:
                for zj in spec.heights:
                    out.append(f"{pair} @ rho {d:.2f} A, z {zi:+.1f} / {zj:+.1f} A")
    return out


# --------------------------------------------------------------------------- #
# The periodic overlap kernel (plan item M4)
# --------------------------------------------------------------------------- #
#
# The autocorrelation above is invariant to rotation, which is what a set of unrelated crystals
# needs and what costs it the co-field: a distance distribution can say which pair of field types
# at which distance matters, never *where* in the cell.  A doped, substituted or defective series
# on **one parent lattice** is the other case.  There the cells share a frame, exactly as the
# molecules of a congeneric series do, and the overlap integral of the fields -- the program's own
# kernel -- applies with one change: the integral runs over one cell and the fields are periodic.
#
# Writing the field of a cell as the lattice sum of its atoms' Gaussians,
#
#     rho_f(r) = sum_{l in cell} sum_T w_lf g_f(|r - r_l - T|),
#
# the overlap over one cell is
#
#     K_f(i, j) = INT_cell rho_f^i rho_f^j dr = sum_{l in cell i} sum_{m in cell j} sum_T k_f(|r_l - r_m - T|),
#
# with k_f the same closed-form Gaussian pair overlap as :mod:`cmfqsar.kernels`: one atom stays in
# the reference cell and the other runs over every periodic image.  So the periodic kernel is the
# ordinary kernel evaluated against an **image-expanded** table, and every shortcut of
# `kernels.py` -- the grouping by width, the cutoff, the compiled loop -- carries over untouched.
#
# The raw value is an integral over one cell, so it grows with the cell: a supercell of the same
# structure gives a multiple of it.  The cosine normalisation removes that factor exactly, which
# is why a model on this kernel requires ``normalize=True``; with it, a set expressed in
# supercells gives the same kernel as the same set in primitive cells, and
# :func:`periodic_overlap_kernels` is a true similarity between two structures on one lattice.


def image_translations(cell: np.ndarray, r_max: float,
                       pbc: Sequence[bool] = (True, True, True)) -> np.ndarray:
    """Every lattice translation that can bring an atom of the cell within `r_max` of the cell.

    The bound is the perpendicular width of the cell along each axis: a translation further than
    ``r_max`` plus the cell's own diameter cannot reach it.  The translations are generated, not
    searched, and :func:`cell_images` then drops the ones whose atoms stay too far away.  No
    translation is made along a direction that is not periodic (`pbc`), so a layer is never
    paired with its own copy across the vacuum.
    """
    A = np.asarray(cell, float).reshape(3, 3)
    volume = abs(float(np.linalg.det(A)))
    if volume <= 0:
        raise ValueError("the cell is singular: three independent lattice vectors are needed")
    diameter = float(np.linalg.norm(A.sum(axis=0)))
    reach = float(r_max) + diameter
    widths = np.array([volume / max(np.linalg.norm(np.cross(A[(k + 1) % 3], A[(k + 2) % 3])), 1e-12)
                       for k in range(3)])
    n = np.ceil(reach / widths).astype(int)
    ranges = [np.arange(-int(n[k]), int(n[k]) + 1) if pbc[k] else np.zeros(1, int)
              for k in range(3)]
    grid = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid @ A


def _box_distance(points: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Distance from every point to an axis-aligned box, zero inside it."""
    d = np.maximum(np.maximum(lo[None, :] - points, points - hi[None, :]), 0.0)
    return np.sqrt((d * d).sum(axis=1))


def cell_images(crystal, r_max: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(coordinates, weights, radii) of the atoms of one cell and of every image within `r_max`.

    "Within `r_max`" is measured from the box the cell spans, not from the atoms of this
    structure, so the same expansion serves every structure of the series: an image that is
    needed by a neighbour's atom is kept even where this structure has no atom near it.
    """
    coords = np.asarray(crystal.coords, float)
    weights = np.asarray(crystal.weights, float)
    radii = np.asarray(crystal.radii, float)
    cell = getattr(crystal, "cell", None)
    if cell is None:
        raise ValueError(f"{crystal.name or 'this structure'} carries no unit cell; the periodic "
                         f"overlap kernel needs crystals built by cmfqsar.periodic.crystal_fields")
    A = np.asarray(cell, float).reshape(3, 3)
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float) @ A
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    out_x, out_w, out_r = [coords], [weights], [radii]
    for T in image_translations(A, float(r_max), _pbc_of(crystal)):
        if not np.any(T):
            continue
        moved = coords + T[None, :]
        keep = _box_distance(moved, lo, hi) <= float(r_max)
        if keep.any():
            out_x.append(moved[keep])
            out_w.append(weights[keep])
            out_r.append(radii[keep])
    return (np.concatenate(out_x, axis=0), np.concatenate(out_w, axis=0),
            np.concatenate(out_r, axis=0))


def overlap_cutoff(specs: Sequence[FieldSpec], radii_max: float = 2.0) -> float:
    """The radius beyond which no field of `specs` contributes to an overlap.

    It is the cutoff rule of plan item B2, read from :mod:`cmfqsar.kernels`: the distance at
    which a Gaussian pair overlap has fallen to 1e-6 of its peak, for the widest Gaussian any
    field in the set puts on an atom.  Using it as the image radius makes the periodic sum as
    exact as the open-boundary kernel with the same cutoff, and no more expensive than it has
    to be.
    """
    from .kernels import _compiled_cutoff

    r = _compiled_cutoff([s for s in specs if s.enabled], True, float(radii_max))
    return float(r) if r else 12.0


def expanded_cell_table(table: AtomTable, r_max: float) -> AtomTable:
    """`table` with the periodic images of every cell added, as one table of the same molecules.

    The molecule boundaries are kept, so the kernel of a structure against this table is summed
    per structure exactly as against the cells themselves.  The expansion is cached on the table,
    because a grid search asks for it once per attenuation and it does not depend on one.
    """
    key = round(float(r_max), 6)
    cached = getattr(table, "_pbc_images", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if not table.molecules:
        raise ValueError("the periodic overlap kernel needs a table built from crystals, which "
                         "carry their cell; this table carries no structures")
    xs, ws, rs = [], [], []
    offsets = [0]
    for mol in table.molecules:
        x, w, r = cell_images(mol, key)
        xs.append(x)
        ws.append(w)
        rs.append(r)
        offsets.append(offsets[-1] + len(x))
    out = AtomTable(coords=np.concatenate(xs, axis=0), weights=np.concatenate(ws, axis=0),
                    radii=np.concatenate(rs, axis=0), offsets=np.array(offsets, int),
                    names=list(table.names), molecules=[])
    table._pbc_images = (key, out)
    return out


def periodic_overlap_kernels(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec],
                             r_max: Optional[float] = None, cutoff=None, engine=None,
                             symmetrise: bool = True, progress=None, cancel=None,
                             symmetry=None) -> np.ndarray:
    """Per-field periodic overlap kernels of two tables of crystals: shape (F, nP, nQ).

    The atoms of `P` stay in their cell and those of `Q` are expanded over the periodic images
    within `r_max` (by default :func:`overlap_cutoff`, the radius at which the widest field of
    the set has fallen to 1e-6 of a peak overlap).  Everything else is
    :func:`cmfqsar.kernels.kernel_matrices`, so `cutoff` and `engine` mean what they mean there.

    The two sets must stand on one parent lattice: the kernel compares the fields point by point
    inside the cell, and two different lattices put the same site at different places, which is
    the frame problem of a molecular series in another guise (:func:`lattice_spread` reports how
    far from one lattice a set is).

    **Why the result is symmetrised.**  The integral runs over the cell of the row and the images
    of the column, so with two different cells the two orders integrate over different domains
    and do not agree; the average of the two is the symmetric form, and it is what a kernel has
    to be.  On one lattice the two orders are equal and the average changes nothing, so the size
    of the correction is a measure of how far the set is from one lattice
    (:func:`overlap_asymmetry` reports it).  ``symmetrise=False`` returns the one-sided sum and
    costs half as much for a block of two different tables.

    **The host symmetry** (plan item M14).  `symmetry` is the pair ``(rotations, translations)``
    of :func:`host_symmetry`: the operations of the space group of the host lattice, in
    fractional coordinates.  With it the kernel is the average of ``K(P, g Q)`` over the
    operations ``g``, which is the kernel of the orbit-averaged fields and so a kernel in its
    own right.  It is needed for a series on one host, because two arrangements of the same
    defects that differ by a symmetry operation of the host -- the same vacancy pair two sites
    along -- have the same property but, in a fixed frame, fields that do not overlap at all.
    The average makes them identical, and leaves every arrangement that is not equivalent as
    distinct as the fields make it.  Without `symmetry` the kernel is the fixed-frame one.
    """
    from .kernels import kernel_matrices

    enabled = [s for s in specs if s.enabled]
    probe = [s.name for s in enabled if getattr(s, "is_probe", False)]
    if probe:
        raise ValueError(f"a probe field carries its Gaussians on the molecules of the table, "
                         f"which the image expansion does not copy, so it cannot be used with "
                         f"the periodic overlap kernel: {', '.join(probe)}")
    radii_max = max(float(np.max(Q.radii)) if len(Q.radii) else 2.0,
                    float(np.max(P.radii)) if len(P.radii) else 2.0)
    r = overlap_cutoff(enabled, radii_max) if r_max is None else float(r_max)
    if symmetry is not None:
        compiled = _compiled_symmetry_average(P, Q, enabled, symmetry, r, engine)
        if compiled is not None:
            return compiled
    Qs = Q if symmetry is None else symmetry_expanded_table(Q, *symmetry)
    K = kernel_matrices(P, expanded_cell_table(Qs, r), enabled, progress=progress, cancel=cancel,
                        symmetric=False, cutoff=cutoff, engine=engine)
    if not symmetrise:
        return K
    if P is Q:
        return 0.5 * (K + np.swapaxes(K, 1, 2))
    Ps = P if symmetry is None else symmetry_expanded_table(P, *symmetry)
    K2 = kernel_matrices(Q, expanded_cell_table(Ps, r), enabled, cancel=cancel,
                         symmetric=False, cutoff=cutoff, engine=engine)
    return 0.5 * (K + np.swapaxes(K2, 1, 2))


def _compiled_symmetry_average(P: AtomTable, Q: AtomTable, enabled: Sequence[FieldSpec],
                               symmetry, r: float, engine,
                               cutoff: Optional[float] = None) -> Optional[np.ndarray]:
    """The linear symmetry average by the compiled loop of :mod:`cmfqsar.fastkernel`, or None
    when that loop is not available, the engine is "numpy", or the tables do not share one cell.

    The loop takes every atom pair at its nearest-image displacement with the cutoff capped
    by :func:`nearest_image_cutoff`, so it agrees with the image sum of the reference path to
    the truncation at that radius (see :func:`symmetry_averaged_kernels`).  `cutoff` gives
    that cap directly, for a caller that has computed it on a larger table than it passes
    here -- the self-overlaps of one structure at a time must be truncated where the kernel
    of the whole table is, or the diagonal of the kernel and the self-overlap differ.
    """
    from .kernels import use_compiled
    from . import fastkernel

    # the reference path copies every cell over the whole group, hundreds of times the work,
    # so the compiled loop is taken whenever it is installed; "numpy" asks for the reference
    if not (use_compiled("auto" if engine is None else engine) and fastkernel.available()):
        return None
    try:
        A = _shared_cell(P)
        if Q is not P and np.max(np.abs(_shared_cell(Q) - A)) > 1e-6:
            return None
    except ValueError:
        return None
    if cutoff is None:
        cut = nearest_image_cutoff(P, r)
        if Q is not P:
            cut = min(cut, nearest_image_cutoff(Q, r))
    else:
        cut = float(cutoff)
    zeros = np.zeros(len(enabled))
    got = fastkernel.defect_kernels(P, Q, enabled, A, symmetry[0], symmetry[1], zeros,
                                    np.zeros(P.n_mol), np.zeros(Q.n_mol), [], cutoff=cut,
                                    symmetric=Q is P)
    if got is None:
        return None
    lin = got[0]
    if Q is P:
        iu = np.triu_indices(P.n_mol, 1)
        for f in range(lin.shape[0]):
            lin[f][(iu[1], iu[0])] = lin[f][iu]
    return lin


def overlap_asymmetry(table: AtomTable, specs: Sequence[FieldSpec],
                      r_max: Optional[float] = None, **kw) -> float:
    """The largest relative disagreement between the two orders of the periodic overlap.

    Zero on one lattice.  It grows with the spread of the cells, because the integral of the
    row's cell and the integral of the column's cell are then not the same integral, and it says
    how much of the kernel is an artefact of that spread rather than of the structures.
    """
    K = periodic_overlap_kernels(table, table, specs, r_max=r_max, symmetrise=False, **kw)
    out = 0.0
    for f in range(K.shape[0]):
        d = np.abs(K[f] - K[f].T)
        s = np.maximum(np.abs(K[f]) + np.abs(K[f].T), 1e-300)
        out = max(out, float(np.max(2.0 * d / s)))
    return out


def periodic_self_overlaps(table: AtomTable, specs: Sequence[FieldSpec],
                           r_max: Optional[float] = None, symmetry=None) -> np.ndarray:
    """``K_f(i, i)`` of every crystal of `table` under the periodic overlap: shape (F, n).

    This is what the cosine normalisation divides by, and it is where the cell size goes: a
    supercell of the same structure has a proportionally larger self-overlap, so the normalised
    kernel does not know which cell was chosen.  `symmetry` is as in
    :func:`periodic_overlap_kernels`: the self-overlap is then that of the orbit-averaged field.
    """
    from .kernels import atom_kernel_block

    enabled = [s for s in specs if s.enabled]
    radii_max = float(np.max(table.radii)) if len(table.radii) else 2.0
    r = overlap_cutoff(enabled, radii_max) if r_max is None else float(r_max)
    if symmetry is not None:
        # one structure at a time through the compiled loop, with the cutoff cap of the whole
        # table, so that the self-overlap is the diagonal of the kernel the same loop gives
        try:
            cut = nearest_image_cutoff(table, r)
        except ValueError:
            cut = r
        parts = [_compiled_symmetry_average(table.subset([i]), table.subset([i]), enabled,
                                            symmetry, r, None, cutoff=cut)
                 for i in range(table.n_mol)]
        if all(p is not None for p in parts):
            return np.stack([p[:, 0, 0] for p in parts], axis=1)
    images = expanded_cell_table(table if symmetry is None
                                 else symmetry_expanded_table(table, *symmetry), r)
    out = np.zeros((len(enabled), table.n_mol))
    for i in range(table.n_mol):
        sl = table.slice(i)
        lo, hi = int(images.offsets[i]), int(images.offsets[i + 1])
        one = AtomTable(coords=images.coords[lo:hi], weights=images.weights[lo:hi],
                        radii=images.radii[lo:hi], offsets=np.array([0, hi - lo], int),
                        names=[table.names[i]], molecules=[])
        blk = atom_kernel_block(table.coords[sl], table.weights[sl], table.radii[sl], one, enabled)
        out[:, i] = blk.sum(axis=1)[:, 0]
    return out


def lattice_spread(table: AtomTable) -> Dict[str, float]:
    """How far a set of crystals is from standing on one lattice.

    Returns the mean cell volume and the largest relative deviation from the mean of the volume
    and of each lattice vector's length.  The periodic overlap kernel is a point-by-point
    comparison inside the cell, so a set whose lattice constants differ by more than a few per
    cent is being compared in a frame that does not quite exist -- the crystal form of a bad
    alignment, and a number a benchmark should state rather than assume.
    """
    cells = [np.asarray(m.cell, float).reshape(3, 3) for m in table.molecules
             if getattr(m, "cell", None) is not None]
    if not cells:
        return {"n": 0.0}
    # the volume of a crystal, the area of a layer, the length of a chain: the vacuum of an
    # open direction is not part of the lattice
    vols = np.array([periodic_measure(A, _pbc_of(m)) or abs(float(np.linalg.det(A)))
                     for A, m in zip(cells, [m for m in table.molecules
                                             if getattr(m, "cell", None) is not None])])
    lens = np.array([np.linalg.norm(A, axis=1) for A in cells])
    mean_len = lens.mean(axis=0)
    return {"n": float(len(cells)), "volume_mean": float(vols.mean()),
            "volume_spread": float(np.max(np.abs(vols - vols.mean())) / max(vols.mean(), 1e-12)),
            "length_spread": float(np.max(np.abs(lens - mean_len[None, :])
                                          / np.maximum(mean_len[None, :], 1e-12))),
            "atoms_mean": float(np.mean([len(m.coords) for m in table.molecules]))}


# --------------------------------------------------------------------------- #
# A series on one host: difference fields and the host symmetry (plan item M14)
# --------------------------------------------------------------------------- #
def host_symmetry(atoms, symprec: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    """The symmetry operations of a host cell: ``(rotations, translations)`` in fractional
    coordinates, shapes (G, 3, 3) and (G, 3), from spglib.

    For a supercell they include the translations by the vectors of the primitive cell, so the
    group of an 8 x 8 cell of a monolayer with twelve point operations has 768 members.  The
    identity is always among them.
    """
    import spglib

    cell = (np.asarray(atoms.get_cell()[:], float), np.asarray(atoms.get_scaled_positions(), float),
            np.asarray(atoms.get_atomic_numbers(), int))
    sym = spglib.get_symmetry(cell, symprec=float(symprec))
    if sym is None:
        raise ValueError(f"spglib found no symmetry for {atoms.get_chemical_formula()}; is the "
                         f"cell singular?")
    return np.asarray(sym["rotations"], float), np.asarray(sym["translations"], float)


def _transformed_structure(mol: CrystalFields, rotations: np.ndarray,
                           translations: np.ndarray) -> CrystalFields:
    """`mol` with its atoms copied under every operation, each copy at 1/G of the weight."""
    G = len(rotations)
    A = np.asarray(mol.cell, float).reshape(3, 3)
    if mol.n_atoms == 0:
        return mol
    frac = np.asarray(mol.coords, float) @ np.linalg.inv(A)
    moved = np.einsum("gij,nj->gni", rotations, frac) + translations[:, None, :]
    moved = np.mod(moved, 1.0).reshape(-1, 3) @ A
    kw = {f.name: getattr(mol, f.name) for f in dataclasses.fields(mol)}
    kw.update(coords=moved, weights=np.tile(np.asarray(mol.weights, float), (G, 1)) / G,
              radii=np.tile(np.asarray(mol.radii, float), G),
              symbols=list(mol.symbols) * G, atom_types=list(mol.atom_types) * G,
              charges=None if mol.charges is None else np.tile(np.asarray(mol.charges), G),
              bonds=np.zeros((0, 2), dtype=int), occupancy=None, site_species=None,
              second_moments=None)
    return CrystalFields(**kw)


def symmetry_expanded_table(table: AtomTable, rotations: np.ndarray,
                            translations: np.ndarray) -> AtomTable:
    """`table` with every structure replaced by the mean of its images under the operations.

    Every atom is copied once per operation at 1/G of its weight, so the field of the new
    structure is the orbit average of the old one, and the overlap of a structure with it is
    the average of the overlaps with the G transformed copies.  The structure boundaries are
    kept, and the expansion is cached on the table, because a grid search asks for it once per
    attenuation and it does not depend on one.
    """
    rotations = np.asarray(rotations, float).reshape(-1, 3, 3)
    translations = np.asarray(translations, float).reshape(-1, 3)
    if len(rotations) != len(translations):
        raise ValueError(f"{len(rotations)} rotations but {len(translations)} translations")
    key = (len(rotations), hash(rotations.tobytes()), hash(translations.tobytes()))
    cached = getattr(table, "_pbc_symmetry", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    if not table.molecules:
        raise ValueError("the symmetry expansion needs a table built from crystals, which "
                         "carry their cell; this table carries no structures")
    out = AtomTable.from_molecules([_transformed_structure(m, rotations, translations)
                                    for m in table.molecules])
    table._pbc_symmetry = (key, out)
    return out


def _shared_cell(table: AtomTable) -> np.ndarray:
    """The one cell every structure of `table` stands in, or an error naming the spread."""
    cells = [np.asarray(m.cell, float).reshape(3, 3) for m in table.molecules]
    A = cells[0]
    worst = max(float(np.max(np.abs(c - A))) for c in cells)
    if worst > 1e-6:
        raise ValueError(f"the structures do not share one cell: lattice vectors differ by up "
                         f"to {worst:.3g} A, and a symmetry operation in fractional coordinates "
                         f"is then a different operation in every cell")
    return A


def _transformed_table(table: AtomTable, A: np.ndarray, R: np.ndarray, t: np.ndarray,
                       r_max: float, images: np.ndarray) -> AtomTable:
    """`table` under one operation (fractional `R`, `t` in the cell `A`), with the periodic
    images within `r_max` of the cell added, built for the whole table at once.

    `images` are the lattice translations of :func:`image_translations`.  The result has no
    molecule objects; it is the right-hand table of one kernel evaluation and nothing else.
    """
    inv = np.linalg.inv(A)
    frac = np.mod((table.coords @ inv) @ R.T + t[None, :], 1.0)
    base = frac @ A
    mol_of = np.repeat(np.arange(table.n_mol), np.diff(table.offsets))
    corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                        [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]], float) @ A
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    xs, ws, rs, ms = [base], [table.weights], [table.radii], [mol_of]
    for T in images:
        if not np.any(T):
            continue
        moved = base + T[None, :]
        keep = _box_distance(moved, lo, hi) <= r_max
        if keep.any():
            xs.append(moved[keep])
            ws.append(table.weights[keep])
            rs.append(table.radii[keep])
            ms.append(mol_of[keep])
    x = np.concatenate(xs, axis=0)
    w = np.concatenate(ws, axis=0)
    r = np.concatenate(rs, axis=0)
    m = np.concatenate(ms)
    order = np.argsort(m, kind="stable")
    counts = np.bincount(m, minlength=table.n_mol)
    offsets = np.zeros(table.n_mol + 1, int)
    offsets[1:] = np.cumsum(counts)
    return AtomTable(coords=x[order], weights=w[order], radii=r[order], offsets=offsets,
                     names=list(table.names), molecules=[])


def nearest_image_cutoff(table: AtomTable, r: float, spread: float = 0.25) -> float:
    """The largest cutoff at which the nearest image of a pair is the only one within reach.

    The compiled path of :func:`symmetry_averaged_kernels` takes every atom pair at its
    nearest-image displacement instead of summing over the images, which is exact when no
    second image lies within the cutoff: the cutoff must then stay under half the cell's width
    along every axis the atoms span.  An axis the atoms do not span -- the vacuum direction of
    a slab, where their fractional spread is below `spread` -- puts no limit, because the
    nearest image along it is the atom itself.  Returns `r` capped accordingly.
    """
    A = _shared_cell(table)
    volume = abs(float(np.linalg.det(A)))
    frac = np.mod(table.coords @ np.linalg.inv(A), 1.0) if len(table.coords) else np.zeros((0, 3))
    cap = float(r)
    for k in range(3):
        width = volume / max(np.linalg.norm(np.cross(A[(k + 1) % 3], A[(k + 2) % 3])), 1e-12)
        spans = len(frac) and float(np.ptp(frac[:, k])) > spread
        if spans:
            cap = min(cap, 0.5 * width)
    return cap


def symmetry_averaged_kernels(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec], symmetry,
                              gammas: Sequence[float] = (), mixing: Optional[np.ndarray] = None,
                              r_max: Optional[float] = None, engine=None,
                              self_p: Optional[np.ndarray] = None,
                              self_q: Optional[np.ndarray] = None,
                              progress=None) -> Tuple[np.ndarray, np.ndarray]:
    """The periodic overlap of two tables of difference fields averaged over the host
    symmetry, in both forms at once: ``(K_lin, K_layer)``.

    ``K_lin``, shape (F, nP, nQ), is ``mean_g K_f(P_i, g Q_j)``: the linear average, the kernel
    of the orbit-averaged fields, which :func:`periodic_overlap_kernels` with `symmetry` also
    gives.  It is additive over the defects -- a vacancy pair is two single vacancies wherever
    they sit -- so it carries the counts of each kind of defect and nothing else.

    ``K_layer``, shape (len(gammas), nP, nQ), is the Gaussian layer averaged instead:
    ``mean_g exp(-gamma (K(P_i, P_i) + K(Q_j, Q_j) - 2 sum_f mixing_f K_f(P_i, g Q_j)))``.
    This is the kernel that sees the *arrangement* of a defect series: the layer is not
    additive, so the pair and triple structure survives the average, and the average is still
    a kernel, the inner product of orbit-averaged features in the layer's own feature space.

    `mixing` are the per-field coefficients of the joint kernel (the field heights divided by
    the field scales, as the model forms them); None weights the fields equally after scaling
    each by its mean self-overlap over `P`.  `gammas` are in absolute units of the joint
    kernel's squared distance; :func:`cmfqsar.model.median_gamma_scale` gives the
    median-distance unit.  `self_p`, `self_q` are the joint self-kernels, the same under every
    operation, computed here when not given.  The structures of `P` and `Q` must share one
    cell, and `r_max` is the image radius (by default :func:`overlap_cutoff`).

    Two implementations agree to the truncation of the image sum: the compiled loop of
    :mod:`cmfqsar.fastkernel` (every atom pair at its nearest-image displacement, the cutoff
    capped by :func:`nearest_image_cutoff`), which is the one for a set of thousands, and the
    NumPy reference, one kernel evaluation per operation on the image-expanded table, which
    `engine="numpy"` selects.
    """
    from .kernels import kernel_matrices, use_compiled
    from . import fastkernel

    rotations = np.asarray(symmetry[0], float).reshape(-1, 3, 3)
    translations = np.asarray(symmetry[1], float).reshape(-1, 3)
    enabled = [s for s in specs if s.enabled]
    radii_max = max(float(np.max(Q.radii)) if len(Q.radii) else 2.0,
                    float(np.max(P.radii)) if len(P.radii) else 2.0)
    r = overlap_cutoff(enabled, radii_max) if r_max is None else float(r_max)
    A = _shared_cell(P)
    if Q is not P:
        AQ = _shared_cell(Q)
        if np.max(np.abs(AQ - A)) > 1e-6:
            raise ValueError("the two tables do not stand in one cell")
    if mixing is None:
        d = periodic_self_overlaps(P, enabled, r_max=r)
        mixing = np.array([1.0 / max(float(np.mean(d[f])), 1e-300) for f in range(len(enabled))])
    mixing = np.asarray(mixing, float)

    def joint_self(T):
        d = periodic_self_overlaps(T, enabled, r_max=r)
        return np.tensordot(mixing, d, axes=(0, 0))

    dp = joint_self(P) if self_p is None else np.asarray(self_p, float)
    dq = dp if (Q is P and self_q is None) else (joint_self(Q) if self_q is None
                                                 else np.asarray(self_q, float))
    gammas = np.asarray(list(gammas), float).reshape(-1)
    compiled = None
    if use_compiled(engine) and fastkernel.available():
        cut = nearest_image_cutoff(P, r)
        if Q is not P:
            cut = min(cut, nearest_image_cutoff(Q, r))
        compiled = fastkernel.defect_kernels(P, Q, enabled, A, rotations, translations, mixing,
                                             dp, dq, gammas, cutoff=cut, symmetric=Q is P)
    if compiled is not None:
        lin, lay = compiled
        if Q is P:
            iu = np.triu_indices(P.n_mol, 1)
            for M in (lin, lay):
                for k in range(M.shape[0]):
                    M[k][(iu[1], iu[0])] = M[k][iu]
        return lin, lay
    images = image_translations(A, r)
    lin = np.zeros((len(enabled), P.n_mol, Q.n_mol))
    lay = np.zeros((len(gammas), P.n_mol, Q.n_mol))
    base = dp[:, None] + dq[None, :]
    G = len(rotations)
    for g in range(G):
        side = _transformed_table(Q, A, rotations[g], translations[g], r, images)
        Kf = kernel_matrices(P, side, enabled, symmetric=False, engine=engine)
        lin += Kf
        d2 = np.maximum(base - 2.0 * np.tensordot(mixing, Kf, axes=(0, 0)), 0.0)
        for k, gamma in enumerate(gammas):
            lay[k] += np.exp(-gamma * d2)
        if progress is not None:
            progress(g + 1, G)
    lin /= G
    lay /= G
    if Q is P:
        lin = 0.5 * (lin + np.swapaxes(lin, 1, 2))
        lay = 0.5 * (lay + np.swapaxes(lay, 1, 2))
    return lin, lay


def symmetry_layer_kernel(P: AtomTable, Q: AtomTable, specs: Sequence[FieldSpec], symmetry,
                          gammas: Sequence[float], **kw) -> np.ndarray:
    """``K_layer`` of :func:`symmetry_averaged_kernels`: shape (len(gammas), nP, nQ)."""
    return symmetry_averaged_kernels(P, Q, specs, symmetry, gammas=gammas, **kw)[1]


def match_sites(atoms, host, tol: float = 0.1) -> np.ndarray:
    """For every atom of `atoms`, the index of the site of `host` it sits on, or -1.

    Distances are taken in the cell of `host` with the minimum-image rule, so an atom written
    on the other side of the cell still finds its site.  Two atoms on one site is an error,
    because a difference field would then count that site twice.
    """
    A = np.asarray(host.cell, float).reshape(3, 3)
    inv = np.linalg.inv(A)
    fa = np.asarray(atoms.get_positions(), float) @ inv
    fh = np.asarray(host.coords, float) @ inv
    d = fa[:, None, :] - fh[None, :, :]
    d -= np.round(d)
    dist = np.linalg.norm(d @ A, axis=2)
    nearest = np.argmin(dist, axis=1)
    out = np.where(dist[np.arange(len(fa)), nearest] <= float(tol), nearest, -1)
    taken = out[out >= 0]
    if len(np.unique(taken)) != len(taken):
        dup = int(np.bincount(taken).argmax())
        raise ValueError(f"two atoms of {atoms.get_chemical_formula()} sit on site {dup} of the "
                         f"host {host.name} within {tol} A")
    return out


def defect_fields(atoms, host: CrystalFields, fieldset: FieldSet, name: str = "",
                  tol: float = 0.1) -> CrystalFields:
    """The field of `atoms` minus the field of its `host`, as a structure of the sites that
    changed: a vacancy is the host's atom with its weights negated, a substitution is that and
    the new atom with its own weights, an added atom is the new atom alone.

    The overlap kernel is bilinear, so the kernel of two differences, plus terms that depend on
    one structure at a time, is the kernel of the two whole cells; a model with a bias on the
    differences is therefore the same model as one on the cells, at a cost that depends on the
    number of defects and not on the size of the supercell.  The charge of an atom that
    replaces another is the host's charge for the site it occupies, so that the electrostatic
    difference is local, as the structural one is; an atom on no site of the host takes the
    oxidation state of its element in the host's composition, or zero when none is assigned.

    `host` is the :class:`CrystalFields` of the pristine cell on the same lattice
    (:func:`crystal_fields`); `atoms` must stand on that lattice within `tol` Angstrom.
    """
    from ase import Atoms

    site = match_sites(atoms, host, tol)
    symbols = list(atoms.get_chemical_symbols())
    hsym = list(host.symbols)
    removed = sorted(set(range(host.n_atoms)) - set(int(s) for s in site if s >= 0))
    changed = [i for i, s in enumerate(site) if s >= 0 and symbols[i] != hsym[int(s)]]
    added = [i for i, s in enumerate(site) if s < 0]
    removed += [int(site[i]) for i in changed]
    new = changed + added
    F = host.weights.shape[1]
    coords = [np.asarray(host.coords, float)[removed]]
    weights = [-np.asarray(host.weights, float)[removed]]
    radii = [np.asarray(host.radii, float)[removed]]
    charges = [None if host.charges is None else -np.asarray(host.charges, float)[removed]]
    syms = [hsym[j] for j in removed]
    types = [f"-{hsym[j]}" for j in removed]
    if new:
        q = np.zeros(len(new))
        if host.charges is not None:
            hq = np.asarray(host.charges, float)
            ox = None
            if added:
                # the states of the host with the added atoms in it: the exact balance when
                # there is one, the nearest balance when there is not, which is the usual case
                # for one extra atom in a cell
                from collections import Counter
                comp = Counter(hsym + [symbols[i] for i in added])
                ox = oxidation_states(list(comp.elements()), composition=dict(comp))
            for k, i in enumerate(new):
                if site[i] >= 0:
                    q[k] = float(hq[int(site[i])])
                elif symbols[i] in hsym:
                    q[k] = float(np.mean(hq[[j for j, s in enumerate(hsym) if s == symbols[i]]]))
                elif ox is not None:
                    q[k] = float(ox.get(symbols[i], 0.0))
        small = Atoms(symbols=[symbols[i] for i in new],
                      positions=np.asarray(atoms.get_positions(), float)[new],
                      cell=np.asarray(host.cell, float), pbc=host.pbc)
        # as a per-atom array, which the "file" charge model reads as it is; initial charges
        # are read only when one of them is not zero, and a zero charge is a legitimate value
        # here (an element of an elemental host, an atom the balance leaves neutral)
        small.set_array("charges", q)
        fs = fieldset.copy()
        fs.charge_model = "file"
        part = crystal_fields(small, fs, name)
        coords.append(part.coords)
        weights.append(part.weights)
        radii.append(part.radii)
        charges.append(None if host.charges is None else q)
        syms += list(part.symbols)
        types += [f"+{s}" for s in part.symbols]
    coords = np.concatenate(coords, axis=0) if coords else np.zeros((0, 3))
    weights = np.concatenate(weights, axis=0) if weights else np.zeros((0, F))
    radii = np.concatenate(radii) if radii else np.zeros(0)
    ch = None if host.charges is None else np.concatenate([c for c in charges if c is not None])
    return CrystalFields(name=name or atoms.get_chemical_formula(), coords=coords.reshape(-1, 3),
                         weights=weights.reshape(-1, F), radii=radii, symbols=syms,
                         cell=np.asarray(host.cell, float), pbc=host.pbc, charges=ch,
                         charge_source=host.charge_source, atom_types=types,
                         relative_to=host.name or "host", n_defects=len(removed) + len(added))


def defect_table(structures: Sequence, host, fieldset: FieldSet,
                 names: Optional[Sequence[str]] = None, tol: float = 0.1) -> AtomTable:
    """An :class:`fields.AtomTable` of difference fields (:func:`defect_fields`) of a series on
    one host.  `host` is an ASE ``Atoms`` of the pristine cell or its :class:`CrystalFields`."""
    if not isinstance(host, CrystalFields):
        host = crystal_fields(host, fieldset, getattr(host, "name", "") or "host")
    names = list(names) if names is not None else [""] * len(structures)
    table = AtomTable.from_molecules([defect_fields(a, host, fieldset, nm, tol)
                                      for a, nm in zip(structures, names)])
    table.host = host
    return table


def is_difference_table(table: AtomTable) -> bool:
    """True when every structure of `table` is a difference from a host."""
    return bool(table.molecules) and all(getattr(m, "relative_to", "") for m in table.molecules)


def defect_count(table: AtomTable) -> np.ndarray:
    """The number of sites that changed in every structure of a difference table: a vacancy
    or an added atom counts one, a substitution one (its two rows share a site).

    :func:`defect_fields` records the number; for a record without it the rows are counted,
    where a ``+`` row at the coordinate of a ``-`` row is a substitution and shares its site.
    """
    out = np.zeros(table.n_mol, int)
    for i, m in enumerate(table.molecules):
        n = getattr(m, "n_defects", None)
        if n is not None:
            out[i] = int(n)
            continue
        minus = [k for k, t in enumerate(m.atom_types) if t.startswith("-")]
        plus = [k for k, t in enumerate(m.atom_types) if t.startswith("+")]
        xyz = np.asarray(m.coords, float)
        shared = 0
        if minus and plus:
            d = np.linalg.norm(xyz[plus][:, None, :] - xyz[minus][None, :, :], axis=2)
            shared = int(np.sum(np.min(d, axis=1) < 1e-6))
        out[i] = len(minus) + len(plus) - shared
    return out


# --------------------------------------------------------------------------- #
# Fields for solids
# --------------------------------------------------------------------------- #
def inorganic_fieldset(elements: Sequence[str] = (), physchem: Sequence[str] = INORGANIC_FIELDS,
                       charge_model: str = "oxidation", other: bool = True,
                       alpha: float = 1.0) -> FieldSet:
    """A field set for solids: the physico-chemical fields named, one indicator field per element
    listed (``cif:<symbol>``) and, with ``other``, one more for every element not listed.

    ``charge_model`` says where the electrostatic weights come from: ``"oxidation"`` guesses
    oxidation states from the composition (pymatgen), falling back to the electronegativity proxy
    when no assignment exists; ``"electronegativity"`` uses that proxy throughout: the deviation
    of each atom's Pauling electronegativity from the mean over the cell, with the sign of a
    charge (an electronegative atom is negative); ``"file"`` reads ``oxi_states``, ``charges`` or
    ``bader_charges`` (what :mod:`cmfqsar.vasp` writes) from the structure's per-atom arrays, or
    its initial charges, and refuses when there are none.
    """
    specs = []
    for name in physchem:
        kind = "steric" if name == "steric" else "gauss"
        specs.append(FieldSpec(name=name, kind=kind, alpha=alpha))
    for el in elements:
        specs.append(FieldSpec(name=f"cif:{el}", kind="gauss", alpha=alpha))
    if other and elements:
        specs.append(FieldSpec(name=f"cif:{OTHER_ELEMENT}", kind="gauss", alpha=alpha))
    return FieldSet(specs=specs, charge_model=charge_model, cif_scheme="element")


def elements_in(structures: Iterable, min_structures: int = 5) -> List[str]:
    """The element symbols present in at least ``min_structures`` of the structures, sorted."""
    counts: Dict[str, int] = {}
    for s in structures:
        for el in set(_symbols_of(s)):
            counts[el] = counts.get(el, 0) + 1
    return sorted(el for el, n in counts.items() if n >= min_structures)


def _symbols_of(s) -> List[str]:
    if isinstance(s, CrystalFields):
        return list(s.symbols)
    return list(s.get_chemical_symbols())


_EN_CACHE: Dict[str, float] = {}
_OXI_CACHE: Dict[str, Optional[Dict[str, float]]] = {}


def electronegativity(symbol: str) -> float:
    """Pauling electronegativity (pymatgen's table); NaN where none is defined (noble gases)."""
    if symbol not in _EN_CACHE:
        from pymatgen.core import Element
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = float(Element(symbol).X)
        _EN_CACHE[symbol] = x
    return _EN_CACHE[symbol]


def oxidation_states(symbols: Sequence[str],
                     composition: Optional[Dict[str, float]] = None) -> Optional[Dict[str, float]]:
    """A charge-balanced assignment of common oxidation states for the composition, or None.

    pymatgen's ``oxi_state_guesses`` on the reduced composition; the first guess is taken,
    which is the one with the most common states.  Elemental solids get zero.  `composition`
    gives the amounts of the elements directly, which a disordered structure needs, since its
    composition is fractional and not a count of its symbols.
    """
    from collections import Counter
    counts = Counter(symbols) if composition is None else dict(composition)
    key = " ".join(f"{el}{n:g}" for el, n in sorted(counts.items()))
    if key in _OXI_CACHE:
        return _OXI_CACHE[key]
    result: Optional[Dict[str, float]] = None
    if len(counts) == 1:
        result = {next(iter(counts)): 0.0}
    else:
        try:
            from pymatgen.core import Composition
            guesses = Composition(key).oxi_state_guesses(max_sites=-1)
            if guesses:
                result = {el: float(v) for el, v in guesses[0].items()}
        except Exception:
            result = None
        if result is None and composition is not None:
            result = _nearest_balance(counts)
    _OXI_CACHE[key] = result
    return result


def _nearest_balance(composition: Dict[str, float], limit: int = 20000) -> Optional[Dict[str, float]]:
    """Common oxidation states that come closest to charge balance for a fractional composition.

    A refined structure with partial occupancies is rarely neutral to the last hundredth, so the
    exact search of pymatgen finds nothing.  This tries every combination of the common states of
    the elements present and keeps the one with the smallest residual charge per atom, the most
    common states first among ties; it refuses when the residual is more than a tenth of a charge
    per atom, which is not an oxide with a misprinted occupancy but a different compound.
    """
    import itertools
    from pymatgen.core import Element
    elements = sorted(composition)
    total = float(sum(composition.values()))
    if total <= 0:
        return None
    # the common states first; every known state only when those cannot balance, which is the
    # mixed-valence case (Ti(III)/Ti(IV) in a lithiated phosphate)
    for attr in ("common_oxidation_states", "oxidation_states"):
        options = [list(getattr(Element(el), attr)) or [0] for el in elements]
        n = 1
        for o in options:
            n *= len(o)
        if n > limit:
            continue
        best, best_res = None, float("inf")
        for combo in itertools.product(*options):
            res = abs(sum(float(composition[el]) * q for el, q in zip(elements, combo))) / total
            if res < best_res - 1e-12:
                best, best_res = combo, res
        if best is not None and best_res <= 0.1:
            return {el: float(q) for el, q in zip(elements, best)}
    return None


def _electronegativity_charges(symbols: Sequence[str]) -> np.ndarray:
    x = np.array([electronegativity(s) for s in symbols], float)
    x = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
    return x.mean() - x


def crystal_charges(atoms, model: str = "oxidation") -> Tuple[np.ndarray, str]:
    """Per-atom charges for the electrostatic field of a solid, and where they came from."""
    symbols = list(atoms.get_chemical_symbols())
    if model == "file":
        for key in ("oxi_states", "charges", "bader_charges"):
            if key in atoms.arrays:
                return np.asarray(atoms.arrays[key], float), key
        q = atoms.get_initial_charges()
        if np.any(q != 0):
            return np.asarray(q, float), "initial_charges"
        raise ValueError(f"structure {atoms.get_chemical_formula()} carries no per-atom "
                         f"'oxi_states' or 'charges' array and no initial charges")
    if model == "oxidation":
        ox = oxidation_states(symbols)
        if ox is not None:
            return np.array([ox[s] for s in symbols], float), "oxidation"
        return _electronegativity_charges(symbols), "electronegativity"
    if model == "electronegativity":
        return _electronegativity_charges(symbols), "electronegativity"
    raise ValueError(f"unknown charge model for solids: {model!r}")


def site_species(atoms) -> Optional[List[Dict[str, float]]]:
    """The species sharing every site with their occupancies, or None for an ordered structure.

    Two conventions are read: ``atoms.info["site_species"]``, one dictionary per site, which
    :func:`read_disordered` writes, and ASE's own CIF convention, ``atoms.info["occupancy"]``
    keyed by the ``spacegroup_kinds`` of the atoms.  A structure in which every site holds one
    species at full occupancy is ordered and gives None, whichever convention it came with.
    """
    n = len(atoms)
    out: Optional[List[Dict[str, float]]] = None
    if "site_species" in atoms.info:
        out = [{str(k): float(v) for k, v in d.items()} for d in atoms.info["site_species"]]
        if len(out) != n:
            raise ValueError(f"{len(out)} site species for {n} atoms")
    elif "occupancy" in atoms.info and "spacegroup_kinds" in atoms.arrays:
        occ = atoms.info["occupancy"]
        kinds = np.asarray(atoms.arrays["spacegroup_kinds"]).reshape(-1)
        out = [{str(k): float(v) for k, v in occ[str(int(kind))].items()}
               if str(int(kind)) in occ else {} for kind in kinds]
    if out is None:
        return None
    symbols = list(atoms.get_chemical_symbols())
    for i, d in enumerate(out):
        if not d:
            out[i] = {symbols[i]: 1.0}
    ordered = all(len(d) == 1 and abs(next(iter(d.values())) - 1.0) < 1e-9
                  and next(iter(d)) == symbols[i] for i, d in enumerate(out))
    return None if ordered else out


def composition_of(atoms) -> Dict[str, float]:
    """The amounts of the elements in the cell, occupancies included."""
    species = site_species(atoms)
    out: Dict[str, float] = {}
    if species is None:
        for s in atoms.get_chemical_symbols():
            out[s] = out.get(s, 0.0) + 1.0
    else:
        for d in species:
            for el, p in d.items():
                out[el] = out.get(el, 0.0) + float(p)
    return out


def read_disordered(path: str, occupancy_tolerance: float = 1.05,
                    site_tolerance: float = 1e-3):
    """A CIF with partial occupancies as an ASE ``Atoms`` carrying its site species (M2).

    pymatgen reads the occupancies; every site gets its majority species as the symbol, its
    total occupancy in ``arrays["occupancy"]`` and the species that share it with their
    occupancies in ``info["site_species"]``, which :func:`crystal_fields` turns into
    occupancy-weighted field weights.  `site_tolerance` is kept small so that split positions a
    fraction of an angstrom apart stay separate sites instead of being merged into one with an
    occupancy above one; `occupancy_tolerance` is what a site may sum to before the file is
    refused.  An ordered CIF comes back as an ordinary structure.
    """
    from ase import Atoms
    from pymatgen.io.cif import CifParser
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parser = CifParser(path, occupancy_tolerance=float(occupancy_tolerance),
                           site_tolerance=float(site_tolerance))
        st = parser.parse_structures(primitive=False)[0]
    symbols, species, occ = [], [], []
    for site in st:
        comp = {sp.symbol: float(v) for sp, v in site.species.items()}
        symbols.append(max(comp, key=comp.get))
        species.append(comp)
        occ.append(sum(comp.values()))
    atoms = Atoms(symbols=symbols, positions=np.asarray(st.cart_coords, float),
                  cell=np.asarray(st.lattice.matrix, float), pbc=True)
    if not st.is_ordered:
        atoms.set_array("occupancy", np.asarray(occ, float))
        atoms.info["site_species"] = species
    atoms.info["source"] = str(path)
    return atoms


def _species_columns(fieldset: FieldSet, species: Dict[str, float], charges_of: Dict[str, float],
                     listed: set, name: str) -> np.ndarray:
    """The field weights of one site occupied by `species` with the given occupancies:
    (n_species, F), one row per species, each row the weights of that species alone."""
    from ase.data import atomic_masses, atomic_numbers, covalent_radii
    rows = []
    for el in species:
        cols = []
        for sp in fieldset.specs:
            if sp.name == "electrostatic":
                cols.append(charges_of[el])
            elif sp.name == "steric":
                cols.append(covalent_radii[atomic_numbers[el]] ** 3)
            elif sp.name == "prop:electronegativity":
                x = electronegativity(el)
                cols.append(x if np.isfinite(x) else 0.0)
            elif sp.name == "prop:mass":
                cols.append(atomic_masses[atomic_numbers[el]])
            elif sp.name == "prop:inverse_sqrt_mass":
                cols.append(atomic_masses[atomic_numbers[el]] ** -0.5)
            elif sp.name.startswith("prop:"):
                raise ValueError(f"structure {name}: the field {sp.name} reads a per-atom array, "
                                 f"which a site shared by several species does not have")
            elif sp.is_cif:
                target = sp.name[4:]
                cols.append((0.0 if el in listed else 1.0) if target == OTHER_ELEMENT
                            else (1.0 if el == target else 0.0))
            else:
                raise ValueError(f"the field {sp.name!r} is not defined for a periodic solid; "
                                 f"use electrostatic, steric, prop:<array> or cif:<element>")
        rows.append(cols)
    return np.asarray(rows, float).reshape(len(species), -1)


def crystal_fields(atoms, fieldset: FieldSet, name: str = "") -> CrystalFields:
    """Evaluate the field weights of one periodic structure (an ASE ``Atoms``).

    A structure with partial occupancies (:func:`site_species`) gets the occupancy-weighted mean
    of the weights of the species sharing each site, and the second moments the self term of the
    periodic autocorrelation needs; see the module docstring.
    """
    from ase.data import atomic_masses, atomic_numbers, covalent_radii
    symbols = list(atoms.get_chemical_symbols())
    n = len(symbols)
    radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols], float)
    listed = {sp.name[4:] for sp in fieldset.specs if sp.is_cif and sp.name[4:] != OTHER_ELEMENT}
    species = site_species(atoms)
    if species is not None:
        return _disordered_fields(atoms, fieldset, species, radii, listed, name)
    charges, source = None, ""
    cols = []
    for sp in fieldset.specs:
        if sp.name == "electrostatic":
            if charges is None:
                charges, source = crystal_charges(atoms, fieldset.charge_model)
            cols.append(charges)
        elif sp.name == "steric":
            cols.append(radii ** 3)
        elif sp.name == "prop:electronegativity":
            x = np.array([electronegativity(s) for s in symbols], float)
            cols.append(np.where(np.isfinite(x), x, 0.0))
        elif sp.name == "prop:mass":
            cols.append(np.array([atomic_masses[atomic_numbers[s]] for s in symbols], float))
        elif sp.name == "prop:inverse_sqrt_mass":
            # a vibrational frequency goes as 1/sqrt(m): the field of the light atoms
            cols.append(np.array([atomic_masses[atomic_numbers[s]] ** -0.5 for s in symbols]))
        elif sp.name.startswith("prop:"):
            key = sp.name[5:]
            if key not in atoms.arrays:
                if fieldset.prop_missing == "zero":
                    cols.append(np.zeros(n))
                    continue
                raise ValueError(f"structure {name or atoms.get_chemical_formula()} carries no "
                                 f"per-atom array {key!r} for the field {sp.name}")
            cols.append(np.asarray(atoms.arrays[key], float).reshape(n))
        elif sp.is_cif:
            el = sp.name[4:]
            if el == OTHER_ELEMENT:
                cols.append(np.array([0.0 if s in listed else 1.0 for s in symbols]))
            else:
                cols.append(np.array([1.0 if s == el else 0.0 for s in symbols]))
        else:
            raise ValueError(f"the field {sp.name!r} is not defined for a periodic solid; use "
                             f"electrostatic, steric, prop:<array> or cif:<element>")
    W = np.column_stack(cols) if cols else np.zeros((n, 0))
    return CrystalFields(name=name or atoms.get_chemical_formula(),
                         coords=np.asarray(atoms.get_positions(), float), weights=W, radii=radii,
                         symbols=symbols, cell=np.asarray(atoms.get_cell()[:], float),
                         pbc=tuple(bool(p) for p in atoms.get_pbc()),
                         charges=charges, charge_source=source, atom_types=list(symbols))


def _disordered_fields(atoms, fieldset: FieldSet, species: List[Dict[str, float]],
                       radii: np.ndarray, listed: set, name: str) -> CrystalFields:
    """The occupancy-weighted fields and their second moments (plan item M2)."""
    from ase.data import atomic_numbers, covalent_radii
    symbols = list(atoms.get_chemical_symbols())
    n = len(symbols)
    label = name or atoms.get_chemical_formula()
    needs_charges = any(sp.name == "electrostatic" for sp in fieldset.specs)
    charges_of: Dict[str, float] = {}
    source = ""
    if needs_charges:
        model = fieldset.charge_model
        if model == "file":
            q, source = crystal_charges(atoms, "file")          # one charge per site, as given
        else:
            ox = oxidation_states(symbols, composition_of(atoms)) if model == "oxidation" else None
            if ox is not None:
                charges_of, source = dict(ox), "oxidation"
            else:
                elements = sorted({el for d in species for el in d})
                x = {el: electronegativity(el) for el in elements}
                mean = float(np.nanmean([v for v in x.values()]))
                charges_of = {el: (mean - v if np.isfinite(v) else 0.0) for el, v in x.items()}
                source = "electronegativity"
    F = fieldset.n_fields
    W = np.zeros((n, F))
    M = np.zeros((n, F, F))
    q_out = np.zeros(n)
    for i, d in enumerate(species):
        if needs_charges and fieldset.charge_model == "file":
            local = {el: float(q[i]) for el in d}               # the site's charge, whatever shares it
        else:
            local = charges_of
        rows = _species_columns(fieldset, d, local, listed, label)
        p = np.array([d[el] for el in d], float)
        W[i] = p @ rows
        M[i] = (rows * p[:, None]).T @ rows
        if needs_charges:
            f = [k for k, sp in enumerate(fieldset.specs) if sp.name == "electrostatic"][0]
            q_out[i] = W[i, f]
        # one radius per site, for the width of the steric field: the occupancy-weighted mean
        radii[i] = float(sum(d[el] * covalent_radii[atomic_numbers[el]] for el in d)
                         / max(sum(d.values()), 1e-12))
    return CrystalFields(name=label, coords=np.asarray(atoms.get_positions(), float), weights=W,
                         radii=radii, symbols=symbols, cell=np.asarray(atoms.get_cell()[:], float),
                         pbc=tuple(bool(p) for p in atoms.get_pbc()),
                         charges=q_out if needs_charges else None, charge_source=source,
                         atom_types=list(symbols),
                         occupancy=np.array([sum(d.values()) for d in species], float),
                         site_species=species, second_moments=M)


def crystal_table(structures: Sequence, fieldset: FieldSet,
                  names: Optional[Sequence[str]] = None) -> AtomTable:
    """An :class:`fields.AtomTable` of periodic structures, which the model takes like any other."""
    names = list(names) if names is not None else [""] * len(structures)
    return AtomTable.from_molecules([crystal_fields(a, fieldset, nm)
                                     for a, nm in zip(structures, names)])


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def read_structures(path: str, index: str = ":") -> list:
    """Every structure of an ASE-readable file (CIF, POSCAR, extended XYZ, ASE JSON, ...).

    A CIF whose sites carry partial occupancies is read through :func:`read_disordered`, so that
    the occupancies reach the fields; every other file goes through ASE as it is.
    """
    from ase.io import read
    out = read(path, index=index)
    out = list(out) if isinstance(out, (list, tuple)) else [out]
    if str(path).lower().endswith(".cif") and any(site_species(a) is not None for a in out):
        try:
            return [read_disordered(path)]
        except Exception:
            return out
    return out


def structures_from_matbench(table: dict, target: Optional[str] = None):
    """The structures and target of a Matbench task file (``{"index", "columns", "data"}``,
    structures as pymatgen dictionaries), as ASE ``Atoms`` and a float array.  Partial
    occupancies are not handled; the Matbench tasks have none."""
    from ase import Atoms
    cols = list(table.get("columns", []))
    if target is None:
        target = next(c for c in cols if c != "structure")
    si, ti = cols.index("structure"), cols.index(target)
    atoms, y, names = [], [], []
    for idx, row in zip(table.get("index", range(len(table["data"]))), table["data"]):
        s = row[si]
        cell = np.asarray(s["lattice"]["matrix"], float)
        symbols, xyz = [], []
        for site in s["sites"]:
            symbols.append(max(site["species"], key=lambda sp: sp.get("occu", 1.0))["element"])
            xyz.append(site["xyz"])
        atoms.append(Atoms(symbols=symbols, positions=np.asarray(xyz, float), cell=cell, pbc=True))
        y.append(float(row[ti]))
        names.append(str(idx))
    return atoms, np.array(y), names


# --------------------------------------------------------------------------- #
# Reading a fitted model: the co-pattern
# --------------------------------------------------------------------------- #
def channel_weights(model) -> np.ndarray:
    """The weight of every channel in a fitted linear model on an invariant kernel.

    The kernel is the dot product of the scaled, unit-normalised channel vectors, so the
    prediction is ``b + sum_c w_c phi_c(x)`` with ``phi`` the scaled and normalised channels of
    ``x`` and ``w = sum_j a_j phi_j`` over the training structures.  That weight vector is to the
    autocorrelation kernels what the co-field is to the overlap kernel: which pair of field types
    at which distance raises the property, and which lowers it.  Defined for the linear layer
    only; a Gaussian or arc-cosine layer has no such vector.
    """
    F = getattr(model, "ac_train_", None)
    if F is None or model.coef is None:
        raise RuntimeError("the model is not fitted on an invariant kernel")
    if not model.is_linear:
        raise ValueError("channel weights are defined for the linear kernel layer only")
    Z = np.asarray(F, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    return Z.T @ np.asarray(model.coef, float)


def top_channels(model, labels: Sequence[str], n: int = 10) -> List[Tuple[str, float, float]]:
    """The ``n`` channels that move the prediction most, as (label, weight, spread).

    ``spread`` is the standard deviation of the scaled, normalised channel over the training
    structures: a weight only matters where the channel varies, so the ranking is by
    ``|weight| * spread`` and both are reported.
    """
    w = channel_weights(model)
    Z = np.asarray(model.ac_train_, float) * np.asarray(model.ac_column_scale, float)
    Z = Z / np.sqrt(np.maximum(np.sum(Z * Z, axis=1, keepdims=True), 1e-300))
    sd = Z.std(axis=0)
    order = np.argsort(-np.abs(w) * sd)[:n]
    return [(labels[c], float(w[c]), float(sd[c])) for c in order]
