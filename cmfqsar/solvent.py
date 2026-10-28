"""
Solvent as a field: where a ligand's water sits, as a CMF field (plan item D9).

Every field this program has built so far is centred on the atoms of the molecule. The 2015
indicator-field formalism does not require that -- it accepts any continuous density -- and a
water occupancy map from a simulation is one. This module turns that map into a field of the
same class as every other, so a sixth kernel block carries it and a co-field over it says where
displacing a water raises the predicted activity and where keeping one does.

The chain, and the decision in each step.

1. **The map is an occupancy, not a count.** For every point of a grid around the ligand, the
   share of frames in which some water oxygen lies within :data:`PROBE` of it. That is a number
   in [0, 1] with a plain meaning -- "a water sits here nine tenths of the time" -- and it needs
   no normalisation against bulk density to be read. Bulk water gives about 0.13 at a 1 angstrom
   probe, which is why :data:`MIN_OCCUPANCY` defaults well above it.
2. **Every frame is put in the common frame first**, by the same superposition the trajectory
   reader uses: the binding-site backbone onto the one structure every run was built from. Maps
   of two ligands are then in one frame and their hydration sites are comparable, which is the
   whole point of making a field out of them.
3. **The water is re-imaged before anything else.** A periodic box writes every atom wrapped, so
   the water beside the site may be recorded on the far side of the cell; the image nearest the
   site is the one that is physically there. This is the same correction the trajectory reader
   makes for the ligand, and skipping it does not give a worse map, it gives an empty one.
4. **Hydration sites are the peaks of the map**, taken greedily: the highest point above
   :data:`MIN_OCCUPANCY`, then everything within :data:`MIN_DISTANCE` of it excluded, and again.
   A site is therefore a place a water occupies, not a cluster of water molecules, and two sites
   are never closer than one water across.
5. **The coefficients are fitted, not read off the peaks.** With the centres fixed at the sites
   and the width taken from the field set, the occupancy map is fitted by a non-negative least
   squares, so the field reproduces the map rather than the heights of its maxima. Non-negative
   because an occupancy cannot be negative, and a fitted field with negative water in it would
   be read as a place water avoids, which is a different statement.

What comes out of :func:`water_field` is a :class:`~cmfqsar.fields.MoleculeFields` whose atoms
are hydration sites. It is used exactly as the variance field of `cmfqsar/fluctuation.py` is:
built per molecule, kept in a table of its own, turned into a kernel of its own, and mixed with
the kernel of the ordinary fields under one coefficient. It cannot share a table with the
molecule's own fields, because a table has one set of atoms and these atoms are not the
molecule's.

**What this does not claim.** That the water field helps. `benchmarks/run_solvent.py` measures
it against the five physico-chemical fields under rule 1, with the interval of rule 4, and
against two controls: sites placed at random in the same box, and the occupancy map of one
ligand used for every ligand. The first says whether the positions matter and the second whether
anything beyond the shared shape of the site does.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .fields import FieldSet, FieldSpec, MoleculeFields
from .trajectory import (AMINO_ACIDS, SOLVENT_RESIDUES, Trajectory, make_whole,
                         load, reference_positions, site_backbone, superimpose)

#: angstrom; a point counts as occupied in a frame when a water oxygen is this close
PROBE = 1.0
#: angstrom between grid points
SPACING = 0.5
#: angstrom around the ligand that the map covers
MAP_RADIUS = 6.0
#: a peak below this share of frames is bulk water, not a site
MIN_OCCUPANCY = 0.5
#: angstrom; two hydration sites are never closer than this
MIN_DISTANCE = 2.4
#: at most this many sites per ligand, the strongest first
MAX_SITES = 40
#: the radius a hydration site carries, for the steric field only
WATER_RADIUS = 1.4


class SolventError(RuntimeError):
    pass


@dataclass
class Occupancy:
    """A water occupancy map of one ligand, in the common frame."""
    points: np.ndarray            # (M, 3) the grid points
    values: np.ndarray            # (M,) share of frames occupied, in [0, 1]
    frames: int
    name: str = ""
    spacing: float = SPACING
    probe: float = PROBE

    @property
    def bulk(self) -> float:
        """What a point in bulk water would score, for comparison with a peak.

        The expected number of oxygens in the probe sphere at the density of liquid water,
        0.0334 per cubic angstrom, capped at one because an occupancy cannot exceed it.
        """
        return float(min(1.0, 0.0334 * (4.0 / 3.0) * np.pi * self.probe ** 3))

    def describe(self) -> str:
        above = int((self.values >= MIN_OCCUPANCY).sum())
        return (f"{len(self.points)} grid points over {self.frames} frames, "
                f"{above} above {MIN_OCCUPANCY:.0%} occupancy, "
                f"strongest {self.values.max():.0%}, bulk would be {self.bulk:.0%}")


# ----------------------------------------------------------------------------------------------
# the map
# ----------------------------------------------------------------------------------------------

def water_oxygens(traj: Trajectory) -> np.ndarray:
    """The oxygen of every water molecule of the topology."""
    idx = [i for i, (r, e) in enumerate(zip(traj.residues, traj.elements))
           if r in SOLVENT_RESIDUES and str(e).upper() == "O"]
    if not idx:
        raise SolventError("the topology holds no water, so it has no occupancy map; a "
                           "simulation in vacuum or with the solvent stripped cannot answer this")
    return np.asarray(idx, int)


def _reimage_to(coords: np.ndarray, group: np.ndarray, centre: np.ndarray,
                box: np.ndarray) -> np.ndarray:
    """Each atom of `group` moved to its image nearest `centre`, frame by frame."""
    out = coords.copy()
    for k in range(coords.shape[0]):
        cell = np.asarray(box[k], float)
        inv = np.linalg.inv(cell)
        d = out[k][group] - centre[k]
        out[k][group] -= np.round(d @ inv) @ cell
    return out


def occupancy(topology: str, trajectory: str, *, reference: str,
              ligand: Optional[str] = None, spacing: float = SPACING,
              map_radius: float = MAP_RADIUS, probe: float = PROBE, stride: int = 1,
              site_radius: float = 8.0, engine: str = "auto", name: str = "") -> Occupancy:
    """The water occupancy map of one simulation, in the frame of `reference`.

    `reference` is the structure every run of the series was built from; without it two ligands'
    maps would be in two frames and could not be compared, which is the same requirement the
    trajectory reader has.
    """
    traj = load(topology, trajectory, stride=stride, engine=engine)
    if traj.box is None:
        raise SolventError("the trajectory carries no periodic cell, so the water cannot be "
                           "re-imaged and the map would be wrong rather than approximate")
    water = water_oxygens(traj)
    lig = _ligand_indices_with_water(traj, ligand)
    protein = np.asarray([i for i, r in enumerate(traj.residues) if r in AMINO_ACIDS], int)
    if protein.size < 3:
        raise SolventError("the topology holds no protein, so there is no site to put a map in")

    # the ligand first made whole, then brought to the image beside the protein -- the same
    # correction the trajectory reader makes, and for the same reason -- and only then the water
    # brought to the images nearest the ligand
    coords = make_whole(traj.coords, lig, traj.box)
    coords = _reimage_to(coords, lig, coords[:, protein, :].mean(axis=1), traj.box)
    coords = _reimage_to(coords, water, coords[:, lig, :].mean(axis=1), traj.box)

    moved = Trajectory(coords=coords, elements=traj.elements, atom_names=traj.atom_names,
                       residues=traj.residues, resids=traj.resids, bonds=traj.bonds,
                       charges=traj.charges, box=traj.box, engine=traj.engine)
    fit = site_backbone(moved, lig, radius=site_radius)
    if fit.size < 3:
        raise SolventError(f"no protein backbone within {site_radius:g} A of the ligand, so it "
                           f"is not in a site")
    target = reference_positions(reference, moved, fit)
    coords = superimpose(coords, fit, target=target)

    origin, shape = _grid_box(coords[:, lig, :], spacing=spacing, pad=map_radius)
    counts = _occupied_counts(coords[:, water, :], origin, shape, float(spacing), float(probe))
    points = _grid_points(origin, shape, float(spacing))
    return Occupancy(points=points, values=counts.ravel() / max(coords.shape[0], 1),
                     frames=int(coords.shape[0]), name=name, spacing=float(spacing),
                     probe=float(probe))


def _ligand_indices_with_water(traj: Trajectory, resname: Optional[str]) -> np.ndarray:
    """The ligand, chosen among the residues that are neither protein nor solvent nor ions."""
    from .trajectory import ligand_indices
    return ligand_indices(traj, resname)


def _grid_box(ligand_coords: np.ndarray, spacing: float,
              pad: float) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """(origin, shape) of a regular grid around every position the ligand took, padded."""
    flat = ligand_coords.reshape(-1, 3)
    lo = flat.min(axis=0) - float(pad)
    hi = flat.max(axis=0) + float(pad)
    shape = tuple(int(np.floor((hi[d] - lo[d]) / float(spacing))) + 1 for d in range(3))
    return np.asarray(lo, float), shape


def _grid_points(origin: np.ndarray, shape: Tuple[int, int, int],
                 spacing: float) -> np.ndarray:
    axes = [origin[d] + spacing * np.arange(shape[d]) for d in range(3)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def _sphere_offsets(spacing: float, probe: float) -> np.ndarray:
    """Cell offsets whose centre lies within `probe` of the cell at the origin."""
    r = int(np.ceil(float(probe) / float(spacing)))
    rng = np.arange(-r, r + 1)
    offs = np.stack(np.meshgrid(rng, rng, rng, indexing="ij"), axis=-1).reshape(-1, 3)
    return offs


def _occupied_counts(water: np.ndarray, origin: np.ndarray, shape: Tuple[int, int, int],
                     spacing: float, probe: float) -> np.ndarray:
    """How many frames each cell held a water oxygen within `probe` of its centre.

    The grid is regular, so no neighbour search is needed: the cells within `probe` of a point
    form a small cube around the point's own cell, and the distances to their centres are
    computed exactly rather than approximated by the cube. A cell is counted once per frame
    however many waters are in it, because the map is an occupancy and not a density.
    """
    counts = np.zeros(int(np.prod(shape)), np.int32)
    offs = _sphere_offsets(spacing, probe)
    dims = np.asarray(shape, int)
    for k in range(water.shape[0]):
        pos = water[k]
        cell = np.floor((pos - origin) / spacing + 0.5).astype(int)
        inside = np.all((cell >= -offs.max() - 1) & (cell <= dims + offs.max() + 1), axis=1)
        cell = cell[inside]
        pos = pos[inside]
        if not len(cell):
            continue
        cand = (cell[:, None, :] + offs[None, :, :]).reshape(-1, 3)
        centres = origin + cand * spacing
        d2 = ((centres - np.repeat(pos, len(offs), axis=0)) ** 2).sum(axis=1)
        ok = (d2 <= probe * probe) & np.all((cand >= 0) & (cand < dims), axis=1)
        cand = cand[ok]
        if not len(cand):
            continue
        flat = np.unique(np.ravel_multi_index(cand.T, shape))
        counts[flat] += 1
    return counts.reshape(shape)


# ----------------------------------------------------------------------------------------------
# the sites
# ----------------------------------------------------------------------------------------------

def hydration_sites(occ: Occupancy, min_occupancy: float = MIN_OCCUPANCY,
                    min_distance: float = MIN_DISTANCE,
                    max_sites: int = MAX_SITES) -> Tuple[np.ndarray, np.ndarray]:
    """(centres, occupancies) of the peaks of the map, strongest first.

    Greedy: the highest point above `min_occupancy` becomes a site, every point within
    `min_distance` of it is removed, and the next highest is taken. Two sites are therefore never
    closer than one water across, and a site is a place a water occupies rather than a cluster of
    molecules.
    """
    keep = occ.values >= float(min_occupancy)
    pts = occ.points[keep]
    vals = occ.values[keep]
    order = np.argsort(-vals)
    pts, vals = pts[order], vals[order]
    centres: List[np.ndarray] = []
    heights: List[float] = []
    taken = np.zeros(len(pts), bool)
    for i in range(len(pts)):
        if taken[i]:
            continue
        centres.append(pts[i])
        heights.append(float(vals[i]))
        taken |= np.linalg.norm(pts - pts[i], axis=1) < float(min_distance)
        if max_sites and len(centres) >= int(max_sites):
            break
    if not centres:
        return np.zeros((0, 3)), np.zeros(0)
    return np.asarray(centres, float), np.asarray(heights, float)


def fit_coefficients(occ: Occupancy, centres: np.ndarray, alpha: float,
                     max_points: int = 20000, seed: int = 0) -> np.ndarray:
    """Non-negative coefficients of Gaussians at `centres` that reproduce the occupancy map.

    The Gaussians are the field's own, ``exp(-alpha/2 |r - c|^2)``, so what comes out is a field
    of the ordinary kind rather than a new sort of object. Non-negative because an occupancy
    cannot be negative and a negative coefficient would be read as a place water avoids, which
    the map does not say. The grid is subsampled when it is large, since a least squares over
    sixty thousand points and forty columns says nothing a sample of twenty thousand does not.
    """
    from scipy.optimize import nnls

    centres = np.asarray(centres, float)
    if centres.size == 0:
        return np.zeros(0)
    pts, vals = occ.points, occ.values
    if len(pts) > int(max_points):
        rng = np.random.RandomState(int(seed))
        pick = rng.choice(len(pts), int(max_points), replace=False)
        pts, vals = pts[pick], vals[pick]
    d2 = np.sum((pts[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
    A = np.exp(-0.5 * float(alpha) * d2)
    coeff, _residual = nnls(A, vals)
    return np.asarray(coeff, float)


# ----------------------------------------------------------------------------------------------
# the field
# ----------------------------------------------------------------------------------------------

def solvent_fieldset(alpha: float = 0.5, name: str = "solvent") -> FieldSet:
    """The one-field set a water field is evaluated with.

    A water field cannot share a table with the molecule's own fields, because a table carries
    one set of atoms and a hydration site is not an atom of the molecule. It is its own block,
    mixed in with a coefficient, exactly as the variance field of `cmfqsar/fluctuation.py` is.
    """
    fs = FieldSet(specs=[FieldSpec(name=name, kind="gauss", alpha=float(alpha))])
    return fs


def water_field(centres: np.ndarray, coefficients: np.ndarray, name: str = "",
                radius: float = WATER_RADIUS) -> MoleculeFields:
    """The hydration sites of one ligand as a field, one Gaussian per site."""
    centres = np.asarray(centres, float)
    coefficients = np.asarray(coefficients, float).ravel()
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise SolventError("the centres must be an (n, 3) array of positions")
    if len(coefficients) != len(centres):
        raise SolventError(f"{len(coefficients)} coefficients for {len(centres)} sites")
    if len(centres) == 0:
        raise SolventError("a ligand with no hydration site has no water field; lower "
                           "min_occupancy or widen the map")
    n = len(centres)
    return MoleculeFields(
        name=name or "solvent", coords=centres,
        weights=coefficients.reshape(n, 1), radii=np.full(n, float(radius)),
        symbols=["O"] * n, atom_types=["Ow"] * n, bonds=np.zeros((0, 2), int),
        charges=np.zeros(n), charge_source="none")


def field_for(topology: str, trajectory: str, *, reference: str, alpha: float = 0.5,
              ligand: Optional[str] = None, name: str = "",
              **kwargs) -> Tuple[MoleculeFields, Occupancy, np.ndarray]:
    """The whole chain for one ligand: (water field, its map, the site occupancies)."""
    occ = occupancy(topology, trajectory, reference=reference, ligand=ligand, name=name,
                    **{k: v for k, v in kwargs.items()
                       if k in ("spacing", "map_radius", "probe", "stride", "site_radius",
                                "engine")})
    centres, heights = hydration_sites(
        occ, **{k: v for k, v in kwargs.items()
                if k in ("min_occupancy", "min_distance", "max_sites")})
    if len(centres) == 0:
        raise SolventError(f"{name or 'this ligand'} has no point occupied often enough to be a "
                           f"hydration site; the strongest was {occ.values.max():.0%}")
    coeff = fit_coefficients(occ, centres, alpha)
    return water_field(centres, coeff, name=name), occ, heights


# ----------------------------------------------------------------------------------------------
# reading it back
# ----------------------------------------------------------------------------------------------

def site_contributions(model, fields: Sequence[MoleculeFields],
                       coefficients: Sequence[float]) -> List[np.ndarray]:
    """What the model asks of the water at each hydration site, per molecule.

    The co-field of a linear kernel model is ``sum_j a_j X_j``, so the weight the model puts on
    one molecule's site is that molecule's dual coefficient times the site's own coefficient. A
    positive number means the model wants **more** water there -- keeping that water raises the
    prediction -- and a negative one means it wants less, which is where displacing a water is
    worth trying.
    """
    a = np.asarray(coefficients, float)
    if len(a) != len(fields):
        raise SolventError(f"{len(a)} dual coefficients for {len(fields)} molecules")
    return [a[i] * np.asarray(f.weights, float).ravel() for i, f in enumerate(fields)]
