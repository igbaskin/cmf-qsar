"""A migration path as the sample: the environment of a hop, its frame and its labels (BT7).

Ionic conductivity in a solid electrolyte is set by the barrier a mobile ion meets between two
sites, not by either site.  That barrier is a field property in the sense of this program: the
ion moving from one site through the bottleneck to the next feels the electrostatic potential,
the repulsion and the polarisability of the framework around the path.  In the language of the
rest of the program the framework is the receptor, the ion is the ligand and the barrier is the
binding energy.

The sample is therefore one **hop**: a pair of sites of the mobile species, in one structure,
with the migrating ion itself removed.  Everything the per-atom items built (plan items LP1 to
LP3) carries over, with three changes that are this module:

* **The environment is cut from a periodic cell.**  :func:`point_environment` is
  :func:`cmfqsar.localenv.environment` for a periodic structure: it collects every atom and every
  periodic image within the reach of the envelope, measured from a *point* rather than from an
  atom, and returns an ordinary :class:`~cmfqsar.fields.MoleculeFields`.  The centre of a hop is
  the midpoint of the two sites, so the first atom of the result is a weightless marker there,
  which keeps the convention of the local modules that the centre is atom 0.
* **The frame is the hop itself.**  The origin is the midpoint and z points from the first site
  to the second, which is the direction the barrier is measured in.  What is left is the rotation
  about z, and :func:`hop_frames` does not choose it: the environment is posed at `n_poses`
  equally spaced angles and averaged, which is a mean-map kernel over the rotation group about
  the path (the device of :mod:`cmfqsar.localalign` for a frame ambiguity).  The starting angle
  is fixed by a rule, the nearest framework atom to the midpoint, so that the discrete set is
  itself a rule.  Reversing a hop reverses z, and the forward and the reverse barrier are two
  samples.
* **The sample index is a pair.**  A hop is (structure, site a, site b, lattice offset of b), and
  the folds hold out whole structures (:meth:`HopSet.targets` builds an
  :class:`~cmfqsar.atomtargets.AtomTargets` whose "molecule" is the structure), since the hops
  of one cell are not independent of each other.

**Labels.**  :func:`bvse_barriers` reads the barrier of every hop from the bond-valence
site-energy landscape (the softBV model, through the ``bvlain`` package): the lowest energy at
which the two sites are connected through the landscape, which is a minimax path and the
quantity the percolation analysis of that method is built from.  It is cheap, so it labels
thousands of hops, and under rule 1 of the plan it is also the reference a model trained on
DFT barriers has to beat.  Measured or DFT barriers come from a table (:func:`read_hop_labels`).

**From hops to conduction.**  :func:`percolation_thresholds` takes one energy per hop of a
structure and reports the level at which the network of hops first connects through the cell in
one, two and three independent directions, which is what the effective activation energy of a
conductor is read from.  The model predicts per hop; this is the step that turns those
predictions into a number per structure.

**The bottleneck.**  :func:`bottleneck` finds the point of least free space on the straight
path, in the spirit of the free-space test of :mod:`cmfqsar.complementarity`, and names the
atoms that form it; it is reported with every hop and is a check that the environment the model
sees is the one a chemist would describe.
"""
from __future__ import annotations

import csv
import itertools
import math
from dataclasses import dataclass, field as _field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .atomtargets import AtomTargets
from .fields import FieldSet, MoleculeFields, combine_members
from .localenv import ENVELOPE_FLOOR, envelope_radius, shell_counts
from .periodic import CrystalFields, crystal_fields

#: default envelope width parameter for a hop, in A^-2: the envelope falls to 1/e at 3.2 A,
#: which reaches the bottleneck atoms and the next shell and not much beyond
DEFAULT_RHO = 0.1
#: poses averaged about the hop axis
DEFAULT_POSES = 12
#: hop lengths considered, in A: shorter than the first is the same site, longer than the
#: second is not a single hop in any known conductor
HOP_MIN, HOP_MAX = 1.5, 4.5
#: the symbol and type of the weightless marker at the centre of an environment
CENTRE_SYMBOL, CENTRE_TYPE = "X", "hop"
#: how far from a site the landscape minimum that stands for it may lie, in A
SITE_SEARCH = 0.6


# --------------------------------------------------------------------------- #
# The periodic environment of a point
# --------------------------------------------------------------------------- #


def _image_range(cell: np.ndarray, pbc: Sequence[bool], reach: float) -> List[range]:
    """How many images along each lattice vector are needed to cover a sphere of radius reach."""
    c = np.asarray(cell, float)
    V = abs(float(np.linalg.det(c)))
    out = []
    for k in range(3):
        if not pbc[k]:
            out.append(range(0, 1))
            continue
        i, j = (k + 1) % 3, (k + 2) % 3
        height = V / max(float(np.linalg.norm(np.cross(c[i], c[j]))), 1e-12)
        n = int(math.ceil(reach / max(height, 1e-12))) + 1
        out.append(range(-n, n + 1))
    return out


def point_environment(crystal: CrystalFields, centre: Sequence[float], rho: float = DEFAULT_RHO,
                      floor: float = ENVELOPE_FLOOR, r_max: Optional[float] = None,
                      exclude: Sequence[Tuple[int, Sequence[int]]] = (),
                      name: str = "") -> MoleculeFields:
    """The fields of a periodic structure seen from a point, under a Gaussian envelope.

    Every atom of the cell and every periodic image of it within the reach of the envelope is
    collected, with the coordinates measured from `centre` and the weights multiplied by
    ``exp(-rho d^2)``, as :func:`cmfqsar.localenv.environment` does for a molecule.  The reach is
    where the envelope falls below `floor`, or `r_max` when that is given and smaller.

    `exclude` lists (atom index, lattice offset) pairs to leave out: for a hop these are the two
    positions of the migrating ion.  Other images of the same atom are other ions and stay.

    The first atom of the result is a weightless marker at the centre, so that the local modules'
    convention that the centre is atom 0 holds; ``member_atom_map`` holds the index in the cell
    of every other atom (-1 for the marker), so that a per-atom quantity can be put back on the
    cell.  ``bonds`` is empty: a framework is described by distances, not by a bond graph.
    """
    reach = envelope_radius(rho, floor)
    if r_max is not None:
        reach = min(reach, float(r_max))
    c0 = np.asarray(centre, float).reshape(3)
    cell = np.asarray(crystal.cell, float)
    coords = np.asarray(crystal.coords, float)
    W = np.asarray(crystal.weights, float)
    radii = np.asarray(crystal.radii, float)
    charges = (np.asarray(crystal.charges, float) if crystal.charges is not None
               and len(crystal.charges) == crystal.n_atoms else np.zeros(crystal.n_atoms))
    skip = {(int(a), tuple(int(x) for x in off)) for a, off in exclude}
    pos, idx, factor = [], [], []
    for off in itertools.product(*_image_range(cell, crystal.pbc, reach)):
        shift = np.asarray(off, float) @ cell
        p = coords + shift - c0
        d2 = np.sum(p * p, axis=1)
        g = np.exp(-float(rho) * d2)
        keep = (d2 <= reach * reach) & (g >= float(floor))
        if skip:
            for a in np.where(keep)[0]:
                if (int(a), tuple(off)) in skip:
                    keep[a] = False
        if keep.any():
            pos.append(p[keep])
            idx.append(np.where(keep)[0])
            factor.append(g[keep])
    if pos:
        pos = np.concatenate(pos)
        idx = np.concatenate(idx)
        factor = np.concatenate(factor)
        order = np.argsort(np.sum(pos * pos, axis=1), kind="stable")
        pos, idx, factor = pos[order], idx[order], factor[order]
    else:
        pos, idx, factor = np.zeros((0, 3)), np.zeros(0, int), np.zeros(0)
    n_f = W.shape[1]
    return MoleculeFields(
        name=name or f"{crystal.name or 'structure'}@{c0.round(2).tolist()}",
        coords=np.vstack([np.zeros((1, 3)), pos]),
        weights=np.vstack([np.zeros((1, n_f)), W[idx] * factor[:, None]]),
        radii=np.concatenate([[1.0], radii[idx]]),
        symbols=[CENTRE_SYMBOL] + [crystal.symbols[i] for i in idx],
        atom_types=[CENTRE_TYPE] + [(crystal.atom_types[i] if crystal.atom_types
                                     else crystal.symbols[i]) for i in idx],
        bonds=np.zeros((0, 2), int),
        charges=np.concatenate([[0.0], charges[idx]]),
        member_atom_map=np.concatenate([[-1], idx]).astype(int),
        charge_source=crystal.charge_source)


# --------------------------------------------------------------------------- #
# Hops
# --------------------------------------------------------------------------- #


@dataclass
class Hop:
    """One migration path: site `a` of a structure to site `b` in the image `offset`."""
    structure: int
    a: int
    b: int
    offset: Tuple[int, int, int]
    start: np.ndarray                     # Cartesian position of a, in the home cell
    vector: np.ndarray                    # from a to b, Cartesian
    #: the Cartesian shift of the image `offset`, so that ``end - shift`` is b in the home cell
    shift: np.ndarray = _field(default_factory=lambda: np.zeros(3))
    #: the forward barrier (from a to b), the reverse one, and the landscape energies behind them
    y: float = float("nan")
    y_reverse: float = float("nan")
    site_a: float = float("nan")
    site_b: float = float("nan")
    saddle: float = float("nan")
    #: least free space on the straight path, in A, where it is, and the atoms that form it
    bottleneck: float = float("nan")
    bottleneck_at: float = float("nan")
    bottleneck_atoms: List[str] = _field(default_factory=list)
    #: which hop stands for this one after symmetry reduction, and how many it stands for
    representative: int = -1
    multiplicity: int = 1
    name: str = ""

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.vector))

    @property
    def midpoint(self) -> np.ndarray:
        return np.asarray(self.start, float) + 0.5 * np.asarray(self.vector, float)

    @property
    def end(self) -> np.ndarray:
        return np.asarray(self.start, float) + np.asarray(self.vector, float)

    def reversed(self) -> "Hop":
        """The same path the other way: b in its image to a in the home cell."""
        off = tuple(-int(x) for x in self.offset)
        return Hop(self.structure, self.b, self.a, off,
                   start=self.end - np.asarray(self.shift, float),
                   vector=-np.asarray(self.vector, float), shift=-np.asarray(self.shift, float),
                   y=self.y_reverse, y_reverse=self.y,
                   site_a=self.site_b, site_b=self.site_a, saddle=self.saddle,
                   bottleneck=self.bottleneck, bottleneck_at=1.0 - self.bottleneck_at,
                   bottleneck_atoms=list(self.bottleneck_atoms), name=self.name)

    def label(self, names: Optional[Sequence[str]] = None) -> str:
        s = names[self.structure] if names is not None else f"structure {self.structure}"
        off = "" if not any(self.offset) else f" {list(self.offset)}"
        return f"{s}: {self.a} -> {self.b}{off}, {self.length:.2f} A"


def enumerate_hops(atoms, mobile: str = "Li", d_min: float = HOP_MIN, d_max: float = HOP_MAX,
                   structure: int = 0) -> List[Hop]:
    """Every pair of sites of the mobile species between `d_min` and `d_max` apart, both ways.

    Periodic images count, so a hop from an atom to its own image in a small cell is a hop.  Each
    unordered pair gives two hops, because the barrier has a direction.
    """
    from ase.neighborlist import neighbor_list
    sym = np.asarray(atoms.get_chemical_symbols())
    where = np.where(sym == mobile)[0]
    if len(where) == 0:
        raise ValueError(f"{atoms.get_chemical_formula()} has no {mobile}")
    i, j, d, D, S = neighbor_list("ijdDS", atoms, float(d_max))
    pos = np.asarray(atoms.get_positions(), float)
    cell = np.asarray(atoms.get_cell()[:], float)
    out: List[Hop] = []
    for k in range(len(i)):
        if sym[i[k]] != mobile or sym[j[k]] != mobile or d[k] < d_min:
            continue
        out.append(Hop(structure, int(i[k]), int(j[k]), tuple(int(x) for x in S[k]),
                       start=pos[i[k]].copy(), vector=np.asarray(D[k], float),
                       shift=np.asarray(S[k], float) @ cell))
    out.sort(key=lambda h: (h.a, h.b, h.offset))
    return out


def hop_environment(crystal: CrystalFields, hop: Hop, rho: float = DEFAULT_RHO,
                    floor: float = ENVELOPE_FLOOR, r_max: Optional[float] = None,
                    name: str = "") -> MoleculeFields:
    """The environment of a hop: the cell seen from the midpoint, without the migrating ion."""
    return point_environment(crystal, hop.midpoint, rho, floor, r_max,
                             exclude=[(hop.a, (0, 0, 0)), (hop.b, hop.offset)],
                             name=name or hop.name)


# --------------------------------------------------------------------------- #
# The frame of a hop
# --------------------------------------------------------------------------- #


def hop_axis_frame(vector: Sequence[float], reference: Optional[Sequence[float]] = None
                   ) -> np.ndarray:
    """The rotation whose rows are x, y, z with z along the hop and x towards `reference`.

    `reference` is a direction whose component perpendicular to the hop fixes x; without one, or
    when it is parallel to the hop, x is taken as the lattice-independent choice of the axis of
    the global frame least aligned with z.
    """
    z = np.asarray(vector, float)
    nz = float(np.linalg.norm(z))
    if nz < 1e-9:
        raise ValueError("a hop of zero length has no frame")
    z = z / nz
    x = None
    if reference is not None:
        v = np.asarray(reference, float)
        v = v - float(v @ z) * z
        if float(np.linalg.norm(v)) > 1e-6:
            x = v / float(np.linalg.norm(v))
    if x is None:
        e = np.eye(3)[int(np.argmin(np.abs(z)))]
        x = e - float(e @ z) * z
        x = x / float(np.linalg.norm(x))
    y = np.cross(z, x)
    return np.stack([x, y, z])


def hop_frames(env: MoleculeFields, vector: Sequence[float], n_poses: int = DEFAULT_POSES
               ) -> List[np.ndarray]:
    """`n_poses` frames of a hop environment: z along the hop, x turned about it in equal steps.

    The starting angle points x at the framework atom nearest to the midpoint (the centre is
    atom 0 and is skipped), so that the set is fixed by the geometry and not by the coordinates
    the file happened to use.
    """
    ref = None
    if env.n_atoms > 1:
        d = np.linalg.norm(np.asarray(env.coords, float)[1:], axis=1)
        ref = np.asarray(env.coords, float)[1 + int(np.argmin(d))]
    R0 = hop_axis_frame(vector, ref)
    n = max(int(n_poses), 1)
    out = []
    for k in range(n):
        t = 2.0 * np.pi * k / n
        c, s = np.cos(t), np.sin(t)
        # a rotation about the hop axis, applied in the hop frame
        T = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        out.append(T @ R0)
    return out


def posed_hop(env: MoleculeFields, vector: Sequence[float], n_poses: int = DEFAULT_POSES,
              name: str = "") -> MoleculeFields:
    """The hop environment in its frame, averaged over the rotation about the hop axis."""
    from .localalign import pose, posed
    frames = hop_frames(env, vector, n_poses)
    if len(frames) == 1:
        return pose(env, frames[0], name=name or env.name)
    return posed(env, frames=frames, name=name or env.name)


# --------------------------------------------------------------------------- #
# The bottleneck and the symmetry reduction
# --------------------------------------------------------------------------- #


def bottleneck(env: MoleculeFields, vector: Sequence[float], n_points: int = 41,
               margin: float = 0.3) -> Tuple[float, float, List[str]]:
    """The point of least free space on the straight path, as (distance, position, atoms).

    The path runs from -vector/2 to +vector/2 through the centre of `env`; at every point the
    distance to the nearest framework atom is taken, and the smallest of those is the
    bottleneck.  `position` is the fraction of the path at which it occurs (0.5 is the midpoint)
    and `atoms` are the symbols of the atoms within `margin` angstrom of that nearest distance,
    which is the ring the ion has to pass through.
    """
    v = 0.5 * np.asarray(vector, float)
    pts = np.linspace(-1.0, 1.0, int(n_points))[:, None] * v[None, :]
    coords = np.asarray(env.coords, float)[1:]
    if len(coords) == 0:
        return float("nan"), float("nan"), []
    d = np.linalg.norm(pts[:, None, :] - coords[None, :, :], axis=2)
    nearest = d.min(axis=1)
    k = int(np.argmin(nearest))
    ring = np.where(d[k] <= nearest[k] + float(margin))[0]
    atoms = sorted(env.symbols[1 + i] for i in ring)
    return float(nearest[k]), float((k / (n_points - 1))), atoms


def fingerprint(env: MoleculeFields, vector: Sequence[float], reach: float = 5.0,
                decimals: int = 2) -> Tuple:
    """What a hop looks like up to a rotation about its axis: the sorted (symbol, distance from
    the midpoint, position along the hop) of the atoms within `reach`.

    Two hops related by a symmetry of the structure have the same fingerprint; a hop and its
    reverse have different ones unless the two sites are equivalent, which is as it should be,
    since their barriers then differ.
    """
    z = np.asarray(vector, float)
    z = z / max(float(np.linalg.norm(z)), 1e-12)
    coords = np.asarray(env.coords, float)[1:]
    d = np.linalg.norm(coords, axis=1)
    keep = d <= float(reach)
    rows = [(env.symbols[1 + i], round(float(d[i]), decimals),
             round(float(coords[i] @ z), decimals)) for i in np.where(keep)[0]]
    return (round(float(np.linalg.norm(vector)), decimals), tuple(sorted(rows)))


def reduce_by_symmetry(hops: Sequence[Hop], envs: Sequence[MoleculeFields]) -> List[int]:
    """Mark equivalent hops of one structure and return the indices of the representatives.

    Equivalence is equality of :func:`fingerprint`.  Every hop gets its `representative` and the
    representative its `multiplicity`, so that a prediction for the representative can be spread
    back to all the hops it stands for, which the percolation step needs.
    """
    seen: Dict[Tuple, int] = {}
    reps: List[int] = []
    for k, (h, e) in enumerate(zip(hops, envs)):
        key = (h.structure, fingerprint(e, h.vector))
        if key in seen:
            r = seen[key]
            h.representative = r
            hops[r].multiplicity += 1
        else:
            seen[key] = k
            h.representative = k
            h.multiplicity = 1
            reps.append(k)
    return reps


# --------------------------------------------------------------------------- #
# A set of structures with their hops
# --------------------------------------------------------------------------- #


@dataclass
class HopSet:
    """Structures, their hops, and the index that turns the hops into a per-sample target."""
    structures: list                      # ASE Atoms
    names: List[str]
    hops: List[Hop]
    mobile: str = "Li"

    @classmethod
    def build(cls, structures: Sequence, names: Optional[Sequence[str]] = None,
              mobile: str = "Li", d_min: float = HOP_MIN, d_max: float = HOP_MAX) -> "HopSet":
        names = list(names) if names is not None else [a.get_chemical_formula()
                                                        for a in structures]
        hops: List[Hop] = []
        for k, a in enumerate(structures):
            for h in enumerate_hops(a, mobile, d_min, d_max, structure=k):
                h.name = h.label(names)
                hops.append(h)
        return cls(list(structures), names, hops, mobile)

    def __len__(self) -> int:
        return len(self.hops)

    def of_structure(self, k: int) -> List[int]:
        return [i for i, h in enumerate(self.hops) if h.structure == int(k)]

    def representatives(self) -> List[int]:
        return [i for i, h in enumerate(self.hops) if h.representative in (-1, i)]

    def crystals(self, fieldset: FieldSet) -> List[CrystalFields]:
        return [crystal_fields(a, fieldset, name=n) for a, n in zip(self.structures, self.names)]

    def environments(self, crystals: Sequence[CrystalFields], rows: Optional[Sequence[int]] = None,
                     rho: float = DEFAULT_RHO, floor: float = ENVELOPE_FLOOR,
                     r_max: Optional[float] = None, posed: bool = False,
                     n_poses: int = DEFAULT_POSES, progress=None) -> List[MoleculeFields]:
        """The environment of every hop in `rows` (all of them by default), posed if asked."""
        rows = list(range(len(self.hops))) if rows is None else list(rows)
        out = []
        for n, r in enumerate(rows):
            h = self.hops[r]
            e = hop_environment(crystals[h.structure], h, rho, floor, r_max)
            out.append(posed_hop(e, h.vector, n_poses) if posed else e)
            if progress is not None and n % 50 == 0:
                progress(f"Hop environments {n + 1}/{len(rows)}", n / max(len(rows), 1))
        return out

    def annotate(self, crystals: Sequence[CrystalFields], rho: float = DEFAULT_RHO,
                 floor: float = ENVELOPE_FLOOR, r_max: Optional[float] = None) -> List[int]:
        """Bottleneck of every hop and the symmetry reduction; returns the representatives."""
        envs = self.environments(crystals, rho=rho, floor=floor, r_max=r_max)
        for h, e in zip(self.hops, envs):
            h.bottleneck, h.bottleneck_at, h.bottleneck_atoms = bottleneck(e, h.vector)
        return reduce_by_symmetry(self.hops, envs)

    def targets(self, rows: Optional[Sequence[int]] = None, which: str = "forward",
                name: str = "barrier") -> AtomTargets:
        """The hops in `rows` with a finite label, as an index whose "molecule" is the structure.

        The folds of the result hold out whole structures, and every tool of the local modules
        that takes an :class:`~cmfqsar.atomtargets.AtomTargets` takes this one.  ``atom`` is the
        index of the hop in :attr:`hops`.
        """
        rows = list(range(len(self.hops))) if rows is None else list(rows)
        key = "y" if which == "forward" else "y_reverse"
        keep = [r for r in rows if np.isfinite(getattr(self.hops[r], key))]
        return AtomTargets(molecule=np.array([self.hops[r].structure for r in keep], int),
                           atom=np.array(keep, int),
                           y=np.array([getattr(self.hops[r], key) for r in keep], float),
                           n_mol=len(self.structures), name=name,
                           per_molecule=[np.zeros(0) for _ in self.structures])

    def describe(self) -> str:
        reps = self.representatives()
        labelled = sum(1 for h in self.hops if np.isfinite(h.y))
        per = [len(self.of_structure(k)) for k in range(len(self.structures))]
        return (f"{len(self.hops)} hops of {self.mobile} in {len(self.structures)} structures "
                f"({min(per) if per else 0} to {max(per) if per else 0} per structure), "
                f"{len(reps)} distinct by symmetry, {labelled} with a barrier")


# --------------------------------------------------------------------------- #
# Labels from the bond-valence landscape
# --------------------------------------------------------------------------- #


def _mobile_charge(atoms, mobile: str, oxidation: Optional[Dict[str, float]]) -> int:
    if oxidation and mobile in oxidation:
        return int(round(oxidation[mobile]))
    return 1


def bvse_calculator(atoms, mobile: str = "Li", oxidation: Optional[Dict[str, float]] = None,
                    resolution: float = 0.25, r_cut: float = 10.0):
    """A ``bvlain`` calculator with the landscape of the mobile ion over the framework.

    The oxidation states come from `oxidation`, else from the structure's ``oxi_states`` array,
    else from the composition (:func:`cmfqsar.periodic.oxidation_states`).  The framework is the
    structure without the mobile species, which is how the bond-valence method is defined.
    """
    from bvlain import Lain
    from pymatgen.io.ase import AseAtomsAdaptor

    from .periodic import oxidation_states

    plain = atoms.copy()
    for key in list(plain.arrays):
        if key not in ("numbers", "positions"):
            del plain.arrays[key]
    st = AseAtomsAdaptor.get_structure(plain)
    symbols = list(atoms.get_chemical_symbols())
    if oxidation is None and "oxi_states" in atoms.arrays:
        st.add_oxidation_state_by_site([float(x) for x in atoms.arrays["oxi_states"]])
    else:
        ox = oxidation if oxidation is not None else oxidation_states(symbols)
        if ox is None:
            raise ValueError(f"no oxidation states for {atoms.get_chemical_formula()}: pass "
                             f"them, or put an 'oxi_states' array on the structure")
        st.add_oxidation_state_by_element({k: int(round(v)) for k, v in ox.items()})
    q = _mobile_charge(atoms, mobile, oxidation)
    calc = Lain(verbose=False)
    calc.read_structure(st, oxi_check=False)
    calc.bvse_distribution(mobile_ion=f"{mobile}{abs(q)}{'+' if q > 0 else '-'}",
                           r_cut=float(r_cut), resolution=float(resolution), k=100)
    return calc


def bvse_energy(calc, points: np.ndarray, f: float = 0.74) -> np.ndarray:
    """The bond-valence site energy of the mobile ion at Cartesian `points`, in eV.

    This evaluates the same Morse plus screened-Coulomb sum ``bvlain`` puts on its grid, at
    points of the caller's choosing: the points are wrapped into the cell and placed in the
    middle of the calculator's supercell, whose neighbour search then applies.
    """
    from scipy.spatial import cKDTree

    from bvlain.potentials import BVSEPotential

    pts = np.asarray(points, float).reshape(-1, 3)
    frac = calc.cell.scaled_positions(pts)
    frac = frac - np.floor(frac)
    sites = calc.cell.cartesian_positions(frac + np.asarray(calc.shift, float))
    tree = cKDTree(calc.supercell.positions)
    distances, ids = tree.query(sites, k=calc.k, distance_upper_bound=calc.r_cut)
    r_min = np.take(calc._get_array("r_min"), ids, axis=-1)
    alpha = np.take(calc._get_array("alpha"), ids, axis=-1)
    r_c = np.take(calc._get_array("r_c"), ids, axis=-1)
    d0 = np.take(calc._get_array("d0"), ids, axis=-1)
    q = np.take(calc._get_array("oxi_states"), ids, axis=-1)
    q = np.where(q * calc.q_mi > 0, q, 0)
    n = np.take(calc._get_array("n"), ids, axis=-1)
    with np.errstate(all="ignore"):
        morse = np.nan_to_num(BVSEPotential.Morse(distances, r_min, d0, alpha), nan=0.0).sum(axis=1)
        coulomb = np.nan_to_num(BVSEPotential.Coulomb(distances, calc.q_mi, q, calc.rc_mi, r_c,
                                                      calc.n_mi, n, f=f), nan=0.0).sum(axis=1)
    return morse + coulomb


def path_barrier(energy: Callable[[np.ndarray], np.ndarray], start: np.ndarray,
                 vector: np.ndarray, spacing: float = 0.2, pad: float = 2.5,
                 site_search: float = SITE_SEARCH) -> Tuple[float, float, float]:
    """(saddle, energy at the start site, energy at the end site) on a landscape.

    A box is laid around the path -- `pad` angstrom beyond each site along it and `pad` to the
    sides -- and sampled at `spacing`; the energy of a site is the lowest value within
    `site_search` of it, so that a site that the landscape puts a little off the given position
    is still that site.  The saddle is the lowest level at which the two minima are connected
    through the box, found by bisection on the level with a connected-components labelling at
    each step: a minimax path, which is what a migration barrier on a grid is.
    """
    from scipy import ndimage

    R = hop_axis_frame(vector)
    L = float(np.linalg.norm(vector))
    mid = np.asarray(start, float) + 0.5 * np.asarray(vector, float)
    half_z = 0.5 * L + float(pad)
    nxy = int(math.ceil(2 * float(pad) / spacing)) + 1
    nz = int(math.ceil(2 * half_z / spacing)) + 1
    xs = np.linspace(-pad, pad, nxy)
    zs = np.linspace(-half_z, half_z, nz)
    X, Y, Z = np.meshgrid(xs, xs, zs, indexing="ij")
    local = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    pts = local @ R + mid                         # rows of R are the axes: local -> global
    E = np.asarray(energy(pts), float).reshape(X.shape)
    E = np.where(np.isfinite(E), E, np.nanmax(E[np.isfinite(E)]) if np.isfinite(E).any() else 0.0)

    def site(z0: float):
        d2 = X * X + Y * Y + (Z - z0) ** 2
        mask = d2 <= site_search * site_search
        sub = np.where(mask, E, np.inf)
        k = np.unravel_index(int(np.argmin(sub)), E.shape)
        return k, float(E[k])

    ka, ea = site(-0.5 * L)
    kb, eb = site(+0.5 * L)
    lo, hi = max(ea, eb), float(E.max())
    structure = ndimage.generate_binary_structure(3, 3)
    if lo >= hi:
        return hi, ea, eb
    saddle = hi
    for _ in range(18):
        t = 0.5 * (lo + hi)
        lab, _n = ndimage.label(E <= t, structure=structure)
        if lab[ka] == lab[kb] and lab[ka] > 0:
            hi, saddle = t, t
        else:
            lo = t
        if hi - lo < 1e-3:
            break
    return float(saddle), ea, eb


def bvse_barriers(hopset: HopSet, rows: Optional[Sequence[int]] = None,
                  oxidation: Optional[Dict[str, float]] = None, resolution: float = 0.25,
                  spacing: float = 0.2, pad: float = 2.5, progress=None,
                  spread: bool = True, grid_percolation: bool = True) -> Dict[str, Dict[str, float]]:
    """Label the hops in `rows` with their bond-valence barriers; returns the percolation
    energies of every structure's landscape as ``bvlain`` reports them, by structure name.

    With `spread`, a hop that was not labelled itself but has a labelled representative takes
    its representative's values, and a hop whose reverse was labelled takes the reverse values.
    `grid_percolation` asks ``bvlain`` for the percolation energies of its own grid, which the
    hop network of :func:`structure_percolation` can be compared with; it can take minutes on a
    landscape with many pockets, so it can be switched off.
    """
    rows = list(range(len(hopset.hops))) if rows is None else list(rows)
    by_structure: Dict[int, List[int]] = {}
    for r in rows:
        by_structure.setdefault(hopset.hops[r].structure, []).append(r)
    out: Dict[str, Dict[str, float]] = {}
    done = 0
    for k, hrows in sorted(by_structure.items()):
        atoms = hopset.structures[k]
        calc = bvse_calculator(atoms, hopset.mobile, oxidation, resolution=resolution)
        out[hopset.names[k]] = {}
        if grid_percolation:
            try:
                perc = calc.percolation_barriers(encut=5.0)
                out[hopset.names[k]] = {kk: float(v) for kk, v in perc.items()}
            except Exception as ex:                  # the landscape stays usable without it
                out[hopset.names[k]] = {"error": str(ex)[:120]}
        energy = lambda pts, c=calc: bvse_energy(c, pts)
        for r in hrows:
            h = hopset.hops[r]
            s, ea, eb = path_barrier(energy, h.start, h.vector, spacing=spacing, pad=pad)
            h.saddle, h.site_a, h.site_b = s, ea, eb
            h.y, h.y_reverse = s - ea, s - eb
            done += 1
            if progress is not None and done % 20 == 0:
                progress(f"Bond-valence barriers {done}/{len(rows)}", done / max(len(rows), 1))
    if spread:
        spread_labels(hopset)
    return out


def spread_labels(hopset: HopSet) -> int:
    """Give every unlabelled hop the label of its representative, or of its reverse; returns
    how many were filled."""
    index = {(h.structure, h.a, h.b, h.offset): i for i, h in enumerate(hopset.hops)}
    filled = 0
    for h in hopset.hops:
        if np.isfinite(h.y):
            continue
        r = hopset.hops[h.representative] if 0 <= h.representative < len(hopset.hops) else None
        src = r if r is not None and np.isfinite(r.y) else None
        if src is None:
            back = index.get((h.structure, h.b, h.a, tuple(-x for x in h.offset)))
            if back is not None and np.isfinite(hopset.hops[back].y_reverse):
                b = hopset.hops[back]
                h.y, h.y_reverse = b.y_reverse, b.y
                h.site_a, h.site_b, h.saddle = b.site_b, b.site_a, b.saddle
                filled += 1
            continue
        h.y, h.y_reverse = src.y, src.y_reverse
        h.site_a, h.site_b, h.saddle = src.site_a, src.site_b, src.saddle
        filled += 1
    return filled


# --------------------------------------------------------------------------- #
# Labels from a table, and the table of a set
# --------------------------------------------------------------------------- #

HOP_COLUMNS = ("structure", "a", "b", "offset_x", "offset_y", "offset_z", "length",
               "barrier", "barrier_reverse", "site_a", "site_b", "saddle", "bottleneck",
               "bottleneck_atoms", "representative", "multiplicity", "predicted")


def read_hop_labels(path: str, hopset: HopSet, column: str = "barrier",
                    reverse_column: str = "barrier_reverse") -> int:
    """Read barriers from a table onto the hops; returns how many hops were labelled.

    The table has the columns ``structure`` (a name of the set, or its index), ``a`` and ``b``
    (atom indices in the structure), optionally ``offset_x``, ``offset_y`` and ``offset_z`` (the
    image of ``b``; the nearest image is taken without them), and the barrier column.  A row
    naming a hop the set does not contain is an error, because a silent miss would be a label
    thrown away.
    """
    names = {n: k for k, n in enumerate(hopset.names)}
    index: Dict[Tuple[int, int, int, Tuple[int, int, int]], int] = {}
    nearest: Dict[Tuple[int, int, int], int] = {}
    for i, h in enumerate(hopset.hops):
        index[(h.structure, h.a, h.b, h.offset)] = i
        key = (h.structure, h.a, h.b)
        if key not in nearest or h.length < hopset.hops[nearest[key]].length:
            nearest[key] = i
    n = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            s = row["structure"].strip()
            k = names[s] if s in names else int(s)
            a, b = int(row["a"]), int(row["b"])
            if all(c in row and row[c].strip() != "" for c in ("offset_x", "offset_y", "offset_z")):
                off = (int(row["offset_x"]), int(row["offset_y"]), int(row["offset_z"]))
                i = index.get((k, a, b, off))
            else:
                i = nearest.get((k, a, b))
            if i is None:
                raise ValueError(f"{path}: no hop {a} -> {b} in {hopset.names[k]} within the "
                                 f"length window; 'cmf hops --list' says which hops there are")
            h = hopset.hops[i]
            h.y = float(row[column])
            if reverse_column in row and row[reverse_column].strip():
                h.y_reverse = float(row[reverse_column])
            n += 1
    return n


def write_hops(path: str, hopset: HopSet, predicted: Optional[Dict[int, float]] = None) -> int:
    """Write every hop with what is known about it; `predicted` maps hop index to prediction."""
    predicted = predicted or {}
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HOP_COLUMNS)
        for i, h in enumerate(hopset.hops):
            w.writerow([hopset.names[h.structure], h.a, h.b, *h.offset, f"{h.length:.4f}",
                        _num(h.y), _num(h.y_reverse), _num(h.site_a), _num(h.site_b),
                        _num(h.saddle), _num(h.bottleneck), " ".join(h.bottleneck_atoms),
                        h.representative, h.multiplicity, _num(predicted.get(i, float("nan")))])
    return len(hopset.hops)


def _num(x: float) -> str:
    return "" if not np.isfinite(x) else f"{float(x):.4f}"


# --------------------------------------------------------------------------- #
# From hops to conduction: percolation through the cell
# --------------------------------------------------------------------------- #


class _Wrapping:
    """Union-find over sites that keeps the lattice offset of every site to its root, so that
    joining two sites already in one component reveals a path that wraps around the cell."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.offset = [np.zeros(3, int) for _ in range(n)]      # offset of i relative to its parent
        self.wraps: Dict[int, List[np.ndarray]] = {}

    def find(self, i: int) -> Tuple[int, np.ndarray]:
        path = []
        while self.parent[i] != i:
            path.append(i)
            i = self.parent[i]
        root = i
        acc = np.zeros(3, int)
        for j in reversed(path):                       # compress, accumulating the offsets
            acc = acc + self.offset[j]
            self.offset[j] = acc.copy()
            self.parent[j] = root
        return root, (self.offset[path[0]] if path else np.zeros(3, int))

    def union(self, a: int, b: int, off: np.ndarray) -> None:
        """Site b sits at the image `off` of the cell relative to site a."""
        ra, oa = self.find(a)
        rb, ob = self.find(b)
        # position of a = oa + root_a ; of b = ob + root_b ; and b - a = off
        if ra == rb:
            wrap = ob - oa - off
            if np.any(wrap):
                self.wraps.setdefault(ra, []).append(wrap)
            return
        self.parent[rb] = ra
        self.offset[rb] = oa + off - ob
        if rb in self.wraps:
            self.wraps.setdefault(ra, []).extend(self.wraps.pop(rb))

    def dimension(self) -> int:
        best = 0
        for root, vs in self.wraps.items():
            if root != self.find(root)[0] or not vs:
                continue
            best = max(best, int(np.linalg.matrix_rank(np.array(vs, float))))
        return best


def percolation_thresholds(n_sites: int, hops: Sequence[Hop], levels: Sequence[float]
                           ) -> Dict[str, float]:
    """The lowest level at which the hops connect through the cell in 1, 2 and 3 directions.

    `levels` is one number per hop, the energy a path crossing it has to reach -- the saddle
    energy relative to the lowest site, or a barrier -- and a hop in either direction carries the
    same connection.  Hops are added in order of increasing level, and the result is
    ``{"E_1D", "E_2D", "E_3D"}`` with *infinity* where the network never wraps in that many
    independent directions.  The sites are the atoms of the mobile species, indexed as the hops
    index them.
    """
    order = np.argsort(np.asarray(levels, float), kind="stable")
    uf = _Wrapping(int(n_sites))
    out = {"E_1D": float("inf"), "E_2D": float("inf"), "E_3D": float("inf")}
    reached = 0
    for k in order:
        h = hops[int(k)]
        lvl = float(levels[int(k)])
        if not np.isfinite(lvl):
            break
        uf.union(h.a, h.b, np.asarray(h.offset, int))
        d = uf.dimension()
        while reached < d:
            reached += 1
            out[f"E_{reached}D"] = lvl
        if reached >= 3:
            break
    return out


def structure_percolation(hopset: HopSet, k: int, values: Optional[Dict[int, float]] = None,
                          relative: bool = True) -> Dict[str, float]:
    """Percolation thresholds of structure `k` from the hops' saddle energies, or from `values`.

    `values` maps a hop index to the energy a path crossing it has to reach; without it the
    bond-valence saddles are used.  With `relative` the thresholds are measured from the lowest
    site energy of the structure, which is what a percolation barrier means.
    """
    rows = hopset.of_structure(k)
    atoms = hopset.structures[k]
    n = len(atoms)
    if values is None:
        lv = [hopset.hops[r].saddle for r in rows]
    else:
        lv = [values.get(r, float("nan")) for r in rows]
    base = 0.0
    if relative:
        sites = [hopset.hops[r].site_a for r in rows if np.isfinite(hopset.hops[r].site_a)]
        base = min(sites) if sites else 0.0
    res = percolation_thresholds(n, [hopset.hops[r] for r in rows], lv)
    return {kk: (v - base if np.isfinite(v) else v) for kk, v in res.items()}


def saddle_from_barrier(hopset: HopSet, predicted: Dict[int, float]) -> Dict[int, float]:
    """Turn predicted forward barriers into saddle energies: site energy plus barrier, where
    the site energy is known, else the barrier itself."""
    out = {}
    for r, v in predicted.items():
        h = hopset.hops[r]
        out[r] = (h.site_a + v) if np.isfinite(h.site_a) else v
    return out


# --------------------------------------------------------------------------- #
# The reference of rule 1, and the co-field read along the path
# --------------------------------------------------------------------------- #


def hop_types(envs: Sequence[MoleculeFields], min_atoms: int = 20) -> List[str]:
    """The atom types of the environments worth a column of the count reference."""
    counts: Dict[str, int] = {}
    for e in envs:
        for t in e.atom_types[1:]:
            counts[t] = counts.get(t, 0) + 1
    return sorted(t for t, c in counts.items() if c >= int(min_atoms))


def hop_count_features(envs: Sequence[MoleculeFields], types: Sequence[str],
                       edges: Optional[Sequence[float]] = None) -> np.ndarray:
    """The atom-centred count of :func:`cmfqsar.localenv.shell_counts` about the midpoint of
    every hop, which is the reference every hop model has to beat."""
    from .localenv import SHELL_EDGES
    edges = SHELL_EDGES if edges is None else edges
    return np.vstack([shell_counts(e, 0, types, edges) for e in envs])


def profile_points(zs: Sequence[float], rs: Sequence[float], n_angles: int = DEFAULT_POSES
                   ) -> np.ndarray:
    """Points of a (z, r) profile in the hop frame: for every z along the hop and every
    radius r from the axis, `n_angles` points around the axis, which are averaged."""
    out = []
    for z in zs:
        for r in rs:
            for k in range(int(n_angles)):
                t = 2.0 * np.pi * k / int(n_angles)
                out.append([r * np.cos(t), r * np.sin(t), z])
    return np.asarray(out, float)


def cofield_profile(model, f: int, zs: Sequence[float], rs: Sequence[float],
                    n_angles: int = DEFAULT_POSES, include_scale: bool = True) -> np.ndarray:
    """The co-field of field `f` as a (len(zs), len(rs)) table in the hop frame.

    The posed kernel averages over the rotation about the hop, so the co-field is a function of
    the position along the hop and the distance from its axis; this reads it at those places and
    averages the residual dependence on the angle that the finite number of poses leaves.
    """
    from .cofields import cofield_at_points
    pts = profile_points(zs, rs, n_angles)
    v = cofield_at_points(model, f, pts, include_scale=include_scale)
    return v.reshape(len(zs), len(rs), int(n_angles)).mean(axis=2)


def format_profile(table: np.ndarray, zs: Sequence[float], rs: Sequence[float],
                   unit: str = "") -> str:
    """A (z, r) table as text: rows along the hop, columns away from its axis."""
    head = "z (A) \\ r (A) |" + "".join(f"{r:8.1f}" for r in rs)
    lines = [head, "-" * len(head)]
    for z, row in zip(zs, table):
        lines.append(f"{z:13.1f} |" + "".join(f"{v:8.2f}" for v in row))
    if unit:
        lines.append(f"(values in {unit})")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The kernel averaged over the rotation about the hop, in closed form
# --------------------------------------------------------------------------- #
#
# Averaging the overlap kernel over the rotation of one environment about the hop axis has a
# closed form.  For two Gaussians of widths b_l and b_m at (rho_l, z_l) and (rho_m, z_m) in
# cylindrical coordinates about the axis, the overlap at a relative angle theta is
#
#     (2 pi / (b_l + b_m))^(3/2) exp(-c [ (z_l - z_m)^2 + rho_l^2 + rho_m^2
#                                         - 2 rho_l rho_m cos theta ]),
#     c = b_l b_m / (2 (b_l + b_m)),
#
# and its mean over theta is the same expression with the cosine term replaced by the modified
# Bessel function I0(2 c rho_l rho_m).  Written with the scaled function i0e(x) = exp(-x) I0(x),
# which is what the library provides and what keeps the number finite,
#
#     k_bar(l, m) = w_l w_m (2 pi / (b_l + b_m))^(3/2)
#                   exp(-c [ (z_l - z_m)^2 + (rho_l - rho_m)^2 ]) i0e(2 c rho_l rho_m).
#
# The rotation average is a projection, so the kernel of two averaged fields equals the kernel of
# one plain field with one averaged field, and this is the mean-map kernel over the whole rotation
# group that `posed_hop` approximates with a finite number of poses.  It costs one Bessel
# evaluation per atom pair, the same as the plain overlap kernel, where the finite poses cost the
# number of poses squared.


def _field_widths(env: MoleculeFields, spec, f: int) -> np.ndarray:
    """The Gaussian width b of every atom for field `f` under `spec`."""
    if spec.kind == "steric":
        r = np.asarray(env.radii, float)
        return float(spec.alpha) / np.maximum(r * r, 1e-12)
    return np.full(env.n_atoms, float(spec.alpha))


def _cylindrical(env: MoleculeFields) -> Tuple[np.ndarray, np.ndarray]:
    """(rho, z) of every atom about the z axis of its frame."""
    c = np.asarray(env.coords, float)
    return np.hypot(c[:, 0], c[:, 1]), c[:, 2]


def _geometric_block(rp, zp, bp, rq, zq, bq):
    """The rotation-averaged overlap of unit Gaussians, for the atoms of P (rows) against the
    atoms of Q (columns): the part of (#eq:hopkernel) that does not depend on the weights."""
    from scipy.special import i0e
    bl, bm = bp[:, None], bq[None, :]
    c = bl * bm / (2.0 * (bl + bm))
    pref = (2.0 * np.pi / (bl + bm)) ** 1.5
    dz = zp[:, None] - zq[None, :]
    dr = rp[:, None] - rq[None, :]
    x = 2.0 * c * rp[:, None] * rq[None, :]
    return pref * np.exp(-c * (dz * dz + dr * dr)) * i0e(x)


def _width_groups(specs) -> List[Tuple[Tuple, List[int]]]:
    """Fields that share one Gaussian width share one geometric block: ``[(key, [f, ...])]``.

    Every ``gauss`` field of one alpha has the same width on every atom, so the exponential and
    the Bessel function are the same for all of them and only the weights differ; the radius-
    scaled steric field has a width per atom and gets a block of its own.
    """
    groups: Dict[Tuple, List[int]] = {}
    for f, sp in enumerate(specs):
        key = ("steric", float(sp.alpha), f) if sp.kind == "steric" else ("gauss", float(sp.alpha))
        groups.setdefault(key, []).append(f)
    return list(groups.items())


def _q_arrays(Q: Sequence[MoleculeFields], specs, groups) -> Dict:
    """The atoms of Q concatenated: cylindrical coordinates, weights, widths per group, owner."""
    rq, zq, wq, owner = [], [], [], []
    bq = {key: [] for key, _ in groups}
    for j, e in enumerate(Q):
        r, z = _cylindrical(e)
        rq.append(r)
        zq.append(z)
        wq.append(np.asarray(e.weights, float))
        owner.append(np.full(e.n_atoms, j, int))
        for key, fs in groups:
            bq[key].append(_field_widths(e, specs[fs[0]], fs[0]))
    return {"r": np.concatenate(rq), "z": np.concatenate(zq), "w": np.concatenate(wq),
            "owner": np.concatenate(owner), "b": {k: np.concatenate(v) for k, v in bq.items()},
            "starts": np.concatenate([[0], np.cumsum([e.n_atoms for e in Q])]), "n": len(Q)}


def _rows_block(P: Sequence[MoleculeFields], i0: int, q: Dict, specs, groups, same: bool,
                chunk: int) -> np.ndarray:
    """The kernel rows of the environments `P` (which are rows i0, i0 + 1, ... of the table)
    against every environment of Q: (F, len(P), nQ)."""
    F, nQ = len(specs), int(q["n"])
    rq, zq, wq, owner = q["r"], q["z"], q["w"], q["owner"]
    out = np.zeros((F, len(P), nQ))
    for n, e in enumerate(P):
        rp, zp = _cylindrical(e)
        wp = np.asarray(e.weights, float)
        first = int(q["starts"][i0 + n]) if same else 0
        for key, fs in groups:
            live = [f for f in fs if np.any(np.abs(wp[:, f]) > 1e-12)]
            if not live:
                continue
            bp = _field_widths(e, specs[fs[0]], fs[0])
            keep_i = np.any(np.abs(wp[:, live]) > 1e-12, axis=1)
            for c0 in range(first, len(rq), int(chunk)):
                c1 = min(c0 + int(chunk), len(rq))
                keep_j = np.any(np.abs(wq[c0:c1][:, live]) > 1e-12, axis=1)
                if not keep_j.any():
                    continue
                G = _geometric_block(rp[keep_i], zp[keep_i], bp[keep_i],
                                     rq[c0:c1][keep_j], zq[c0:c1][keep_j],
                                     q["b"][key][c0:c1][keep_j])
                own = owner[c0:c1][keep_j]
                for f in live:
                    v = (wp[keep_i, f] @ G) * wq[c0:c1, f][keep_j]
                    out[f, n, :] += np.bincount(own, weights=v, minlength=nQ)
    return out


def cylindrical_blocks(P: Sequence[MoleculeFields], Q: Sequence[MoleculeFields],
                       specs, chunk: int = 40000, progress=None, n_jobs: int = 1) -> np.ndarray:
    """The rotation-averaged overlap kernels between the environments of P and of Q: (F, nP, nQ).

    Every environment is in its hop frame, z along the hop.  When `P` is `Q` only the upper
    triangle is computed and mirrored.  The geometric part of the kernel is computed once per
    group of fields that share a width, and the weights are applied per field.  With `n_jobs`
    above one the rows are split over that many processes (joblib), which pays from a few
    hundred environments on; the result does not depend on it.
    """
    same = P is Q
    specs = [s for s in specs if s.enabled]
    F = len(specs)
    nP, nQ = len(P), len(Q)
    groups = _width_groups(specs)
    q = _q_arrays(Q, specs, groups)
    if q["w"].shape[1] != F:
        raise ValueError(f"the environments carry {q['w'].shape[1]} fields and the field set {F}")
    out = np.zeros((F, nP, nQ))
    n_jobs = max(int(n_jobs), 1)
    if n_jobs == 1 or nP < 4 * n_jobs:
        step = 50 if progress is not None else nP
        for i0 in range(0, nP, max(step, 1)):
            i1 = min(i0 + step, nP)
            out[:, i0:i1, :] = _rows_block(P[i0:i1], i0, q, specs, groups, same, chunk)
            if progress is not None:
                progress(f"Rotation-averaged kernel {i1}/{nP}", i1 / max(nP, 1))
    else:
        from joblib import Parallel, delayed
        # the upper triangle makes the late rows cheap, so the blocks are interleaved
        order = np.arange(nP)
        blocks = [order[k::4 * n_jobs] for k in range(4 * n_jobs)]
        results = Parallel(n_jobs=n_jobs)(
            delayed(_rows_block)([P[int(i)] for i in b], 0, q, specs, groups, False, chunk)
            for b in blocks if len(b))
        for b, res in zip([b for b in blocks if len(b)], results):
            out[:, b, :] = res
        same = False                                   # every row was computed in full
    if same:
        for f in range(F):
            U = np.triu(out[f])
            out[f] = U + U.T - np.diag(np.diag(U))
    return out


@dataclass
class HopModel:
    """A kernel ridge model on hop environments with the rotation-averaged overlap kernel.

    The environments are kept in their hop frames; the kernel is the closed-form average over
    the rotation about the hop (:func:`cylindrical_blocks`), one block per field, each divided
    by its diagonal so that no field dominates by its units, and summed with the mixing
    coefficients of the field set.  :meth:`profile` reads the co-field as a function of the
    position along the hop and the distance from its axis, which is what the averaged kernel
    makes the co-field depend on.
    """
    fieldset: FieldSet
    train: List[MoleculeFields]
    K: np.ndarray                         # (n, n) the joint, normalised training kernel
    diag: np.ndarray                      # (F, n) the diagonals of the raw field blocks
    lam: float = 1e-3
    coef: Optional[np.ndarray] = None
    bias: float = 0.0
    y: Optional[np.ndarray] = None
    #: the raw field blocks (F, n, n), kept only when asked for: n^2 per field
    K_fields: Optional[np.ndarray] = None

    @classmethod
    def build(cls, envs: Sequence[MoleculeFields], fieldset: FieldSet, keep_blocks: bool = False,
              progress=None, n_jobs: int = 1) -> "HopModel":
        envs = list(envs)
        blocks = cylindrical_blocks(envs, envs, fieldset.specs, progress=progress, n_jobs=n_jobs)
        d = np.stack([np.diag(blocks[f]) for f in range(blocks.shape[0])])
        model = cls(fieldset=fieldset, train=envs, K=np.zeros((len(envs), len(envs))), diag=d)
        model.K = model._normalised(blocks, d, d)
        if keep_blocks:
            model.K_fields = blocks
        return model

    @property
    def specs(self):
        return [s for s in self.fieldset.specs if s.enabled]

    def _normalised(self, K: np.ndarray, d_rows: np.ndarray, d_cols: np.ndarray) -> np.ndarray:
        F = K.shape[0]
        h = np.array([s.h for s in self.specs], float)
        out = np.zeros(K.shape[1:])
        for f in range(F):
            dr = np.sqrt(np.maximum(d_rows[f], 1e-300))
            dc = np.sqrt(np.maximum(d_cols[f], 1e-300))
            out += h[f] * K[f] / (dr[:, None] * dc[None, :])
        return out

    def fit(self, y: Sequence[float], lam: Optional[float] = None) -> "HopModel":
        from .model import krr_fit
        if lam is not None:
            self.lam = float(lam)
        self.y = np.asarray(y, float)
        self.coef, self.bias = krr_fit(self.K, self.y, self.lam)
        return self

    def kernel_to(self, envs: Sequence[MoleculeFields], progress=None) -> np.ndarray:
        """The (n_test, n_train) joint kernel of new environments against the training ones."""
        envs = list(envs)
        Kt = cylindrical_blocks(envs, self.train, self.specs, progress=progress)
        Kd = cylindrical_blocks(envs, envs, self.specs)
        d = np.stack([np.diag(Kd[f]) for f in range(Kd.shape[0])])
        return self._normalised(Kt, d, self.diag)

    def predict(self, envs: Sequence[MoleculeFields], progress=None) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("the hop model is not fitted")
        return self.kernel_to(envs, progress=progress) @ self.coef + self.bias

    def _averaged_fields(self, f: int, rhos: np.ndarray, zs: np.ndarray) -> np.ndarray:
        """The field of every training environment, averaged about the hop axis, at the
        (rho, z) points, divided by its normalisation: (n_points, n)."""
        from scipy.special import i0e
        spec = self.specs[f]
        out = np.zeros((len(rhos), len(self.train)))
        for j, e in enumerate(self.train):
            r, z = _cylindrical(e)
            w = np.asarray(e.weights, float)[:, f]
            b = _field_widths(e, spec, f)
            keep = np.abs(w) > 1e-12
            if not keep.any():
                continue
            dz = zs[:, None] - z[keep][None, :]
            dr = rhos[:, None] - r[keep][None, :]
            x = b[keep][None, :] * rhos[:, None] * r[keep][None, :]
            g = np.exp(-0.5 * b[keep][None, :] * (dz * dz + dr * dr)) * i0e(x)
            out[:, j] = g @ w[keep]
        return out / np.sqrt(np.maximum(self.diag[f], 1e-300))[None, :]

    def profile(self, f: int, zs: Sequence[float], rs: Sequence[float],
                sigma: Optional[float] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """The co-field of field `f` at every (z, r), as a (len(zs), len(rs)) table, with its
        standard deviation under noise of level `sigma` in the barriers when that is given.

        The co-field is ``sum_j a_j s_fj X_fj(rho, z)`` with ``s_fj`` the normalisation and
        ``X_fj`` the field of environment j averaged about the hop axis; its error follows the
        dual coefficients, whose covariance under independent noise of level sigma is
        ``sigma^2 (K + lam I)^-2``.
        """
        if self.coef is None:
            raise RuntimeError("the hop model is not fitted")
        Z, R = np.meshgrid(np.asarray(zs, float), np.asarray(rs, float), indexing="ij")
        Phi = self._averaged_fields(f, R.ravel(), Z.ravel()) * float(self.specs[f].h)
        values = (Phi @ self.coef).reshape(Z.shape)
        if sigma is None:
            return values, None
        n = len(self.coef)
        M = np.linalg.inv(self.K + self.lam * np.eye(n))
        V = Phi @ M
        sd = float(sigma) * np.sqrt(np.maximum(np.sum(V * V, axis=1), 0.0)).reshape(Z.shape)
        return values, sd

    def atom_contributions(self, env: MoleculeFields) -> np.ndarray:
        """What every atom of an environment adds to its prediction, the bias apart.

        The kernel is a sum over the atoms of the environment, so the prediction splits over
        them exactly, as the atom contributions of the overlap model do.
        """
        if self.coef is None:
            raise RuntimeError("the hop model is not fitted")
        K_self = cylindrical_blocks([env], [env], self.specs)
        h = np.array([s.h for s in self.specs], float)
        out = np.zeros(env.n_atoms)
        for f in range(len(self.specs)):
            d_self = np.sqrt(max(float(K_self[f, 0, 0]), 1e-300))
            rows = _atom_rows(env, self.train, self.specs, f)          # (n_atoms, n_train)
            s = d_self * np.sqrt(np.maximum(self.diag[f], 1e-300))
            out += h[f] * ((rows / s[None, :]) @ self.coef)
        return out


def _atom_rows(env: MoleculeFields, train: Sequence[MoleculeFields], specs, f: int) -> np.ndarray:
    """The rotation-averaged kernel of every atom of `env` with every training environment,
    for one field: (n_atoms, n_train)."""
    spec = specs[f]
    rp, zp = _cylindrical(env)
    wp = np.asarray(env.weights, float)[:, f]
    bp = _field_widths(env, spec, f)
    out = np.zeros((env.n_atoms, len(train)))
    for j, e in enumerate(train):
        r, z = _cylindrical(e)
        w = np.asarray(e.weights, float)[:, f]
        b = _field_widths(e, spec, f)
        keep = np.abs(w) > 1e-12
        if not keep.any():
            continue
        G = _geometric_block(rp, zp, bp, r[keep], z[keep], b[keep])
        out[:, j] = wp * (G @ w[keep])
    return out


def cross_validate_hop_model(model: HopModel, targets: AtomTargets,
                             lams: Optional[Sequence[float]] = None, folds: int = 10,
                             seed: Optional[int] = 0, name: str = ""):
    """The ridge search of :func:`cmfqsar.localenv.choose_lam` on a hop model's kernel."""
    from .localenv import LAMS, choose_lam
    return choose_lam(model.K, targets, lams if lams is not None else LAMS, folds=folds,
                      seed=seed, name=name or "rotation-averaged overlap")


def framed_environments(hopset: HopSet, crystals: Sequence[CrystalFields],
                        rows: Optional[Sequence[int]] = None, rho: float = DEFAULT_RHO,
                        floor: float = ENVELOPE_FLOOR, r_max: Optional[float] = None,
                        progress=None) -> List[MoleculeFields]:
    """The environment of every hop in `rows`, turned into its hop frame (one pose, z along the
    hop), which is what the rotation-averaged kernel takes."""
    from .localalign import pose
    rows = list(range(len(hopset.hops))) if rows is None else list(rows)
    out = []
    for n, r in enumerate(rows):
        h = hopset.hops[r]
        e = hop_environment(crystals[h.structure], h, rho, floor, r_max)
        out.append(pose(e, hop_frames(e, h.vector, 1)[0]))
        if progress is not None and n % 50 == 0:
            progress(f"Hop environments {n + 1}/{len(rows)}", n / max(len(rows), 1))
    return out
