"""
Rigid porous hosts as pockets: zeolites, MOFs and other frameworks read from CIF (plan item M7).

`cmfqsar.protein` and `cmfqsar.complementarity` treat a protein, or a finite host such as a
cyclodextrin, as a pocket: a finite set of atoms whose fields are built with the same
parameters as the guests, so that complementarity, the interaction profile and the rigid-body
placement of `cmfqsar.docking` all work on it. A framework differs in two ways. It repeats, so the
atoms around a pore come from several periodic images of the cell. And it has no residues, so a
profile resolved by residue has nothing to resolve by.

This module removes both differences and changes nothing else:

* :func:`find_pores` maps the free distance -- the distance from a point to the nearest atom's
  surface -- over the cell, with every periodic image counted, and returns its local maxima as
  pores: a centre (the middle of a cage, or the widest point of a channel), a radius (the
  largest sphere that fits there) and an axis (the longest direction of the free space around
  the centre). The centre and the axis are the pore-centred frame a guest is posed in, as
  `cavity_frame` gives it for a macrocycle.
* :func:`framework_pocket` cuts the atoms of every image within a radius of a point into a finite
  :class:`cmfqsar.protein.Protein`. The atoms of the image that holds the point come first and in
  the order of the structure, so for a host that is one molecule in a large box the pocket is the
  host itself, atom for atom. Its "residues" are framework atom types (:func:`atom_types`: the
  element and the number of bonded neighbours, so a bridging oxygen and a terminal one differ), or
  the units a caller gives, so every per-residue table of the pocket code becomes a table per
  atom type.
* :func:`framework_pocket_fields` builds the pocket's fields with the guests' field set. The
  charges of an inorganic framework come from :func:`cmfqsar.periodic.crystal_charges` on the
  periodic structure (formal oxidation states, or electronegativity differences), not from a
  molecular charge scheme that has no parameters for silicon or zinc in a framework.

The rest is the ligand path: `complementarity`, `interaction_profile`, `dock_coords` and
`series_scores` take the pocket as they take a protein pocket.

The limit the plan asks for: a cyclodextrin alone in a cubic box is a framework whose images are
far apart. Once the box is larger than the host plus the radius of the cut, the pocket is the
finite host atom for atom and every score is the finite host's score;
`benchmarks/run_frameworks.py` measures where that happens and what the images do before it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .protein import Protein

#: van der Waals radii in angstrom (Bondi 1964; Alvarez 2013 where Bondi has none) for the free
#: distance; an element missing here gets :data:`DEFAULT_RADIUS`
VDW = {"H": 1.20, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "Na": 2.27, "Mg": 1.73,
       "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "K": 2.75, "Ca": 2.31,
       "Ti": 2.11, "Cr": 2.06, "Mn": 2.05, "Fe": 2.04, "Co": 2.00, "Ni": 1.63, "Cu": 1.40,
       "Zn": 1.39, "Ga": 1.87, "Ge": 2.11, "Br": 1.85, "Zr": 2.23, "Ag": 1.72, "Cd": 1.58,
       "I": 1.98, "Ba": 2.68, "La": 2.43}
DEFAULT_RADIUS = 2.0
#: a bond is two covalent radii plus this, in angstrom, as `protein.covalent_bonds` uses
BOND_TOLERANCE = 0.45
FRAMEWORK_RESNAME = "FWK"


# --------------------------------------------------------------------------- #
# Reading, images and bonds
# --------------------------------------------------------------------------- #

def read_framework(path: str, index: int = 0):
    """A framework structure from a CIF (or any file ASE reads), as an ASE ``Atoms``.

    Only an ordered structure is accepted: a pore needs every atom at one place, and a site
    shared by two species has no single radius or charge.
    """
    from ase.io import read

    atoms = read(path, index=index)
    if not all(atoms.pbc):
        raise ValueError(f"{path}: a framework is periodic in all three directions; this file "
                         f"is periodic in {int(sum(atoms.pbc))}")
    return atoms


def boxed_host(positions: np.ndarray, symbols: Sequence[str], box: float):
    """A finite host in a cubic box of side `box`, at its own coordinates (no shift, no wrap).

    Keeping the coordinates means the image that holds the host is the host itself, number for
    number, which is what makes the large-box limit exact rather than close.
    """
    from ase import Atoms

    pos = np.asarray(positions, float)
    span = pos.max(axis=0) - pos.min(axis=0)
    if box <= float(span.max()):
        raise ValueError(f"a box of {box:g} A is smaller than the host ({span.max():.1f} A)")
    lo = pos.min(axis=0) - 0.5 * (box - span)
    atoms = Atoms(symbols=list(symbols), positions=pos, cell=np.eye(3) * box, pbc=True)
    atoms.info["box_origin"] = lo
    return atoms


def _origin(atoms) -> np.ndarray:
    return np.asarray(atoms.info.get("box_origin", np.zeros(3)), float)


def _fractional(atoms, points: np.ndarray) -> np.ndarray:
    cell = np.asarray(atoms.cell, float)
    return np.linalg.solve(cell.T, (np.asarray(points, float) - _origin(atoms)).T).T


def _image_shifts(cell: np.ndarray, reach: float) -> np.ndarray:
    """Integer image vectors (n, 3), the zero image first, enough to cover `reach` angstrom."""
    heights = 1.0 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)
    m = [int(np.ceil(reach / h)) + 1 for h in heights]
    shifts = [(i, j, k) for i in range(-m[0], m[0] + 1) for j in range(-m[1], m[1] + 1)
              for k in range(-m[2], m[2] + 1)]
    shifts.sort(key=lambda s: (s != (0, 0, 0), abs(s[0]) + abs(s[1]) + abs(s[2]), s))
    return np.array(shifts, int)


def _home_positions(atoms) -> np.ndarray:
    """The atoms' own Cartesian positions; a CIF's atoms are wrapped into the cell already."""
    return np.asarray(atoms.get_positions(), float)


def periodic_bonds(atoms, tolerance: float = BOND_TOLERANCE) -> List[Tuple[int, int, Tuple]]:
    """Every bond of the periodic structure as (i, j, image of j), from covalent radii.

    A bond is two covalent radii plus `tolerance`, the rule the PDB reader of the pocket code
    uses. Hydrogen-hydrogen pairs are never bonds.
    """
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list

    z = atoms.get_atomic_numbers()
    cut = [float(covalent_radii[n]) + 0.5 * tolerance for n in z]
    i, j, S = neighbor_list("ijS", atoms, cut, self_interaction=False)
    out = []
    for a, b, s in zip(i, j, S):
        if z[a] == 1 and z[b] == 1:
            continue
        if a < b or (a == b and tuple(s) > (0, 0, 0)):
            out.append((int(a), int(b), tuple(int(v) for v in s)))
    return out


def atom_types(atoms, tolerance: float = BOND_TOLERANCE) -> List[str]:
    """A framework atom type for every atom: the element and its number of bonded neighbours.

    ``O2`` is an oxygen with two neighbours (the bridging oxygen of a zeolite), ``Si4`` a
    tetrahedral silicon, ``Zn4`` a zinc with four, ``C3`` a carbon with three. The number counts
    neighbours across the cell boundary, so it is a property of the framework and not of the cell
    the file happened to use.
    """
    n = np.zeros(len(atoms), int)
    for a, b, _s in periodic_bonds(atoms, tolerance):
        n[a] += 1
        n[b] += 1
    return [f"{s}{k}" for s, k in zip(atoms.get_chemical_symbols(), n)]


# --------------------------------------------------------------------------- #
# Pores: the free distance and its maxima
# --------------------------------------------------------------------------- #

@dataclass
class Pore:
    """A local maximum of the free distance: where a guest goes, and which way the pore runs.

    The centre is the middle of a cage or the widest point of a channel, and the radius is the
    largest sphere that fits there. The axis is the longest direction of the free space around
    the centre, and `anisotropy` (the longest spread of that space over its shortest) tells a
    channel from a cage.
    """
    centre: np.ndarray
    radius: float                    # free distance at the centre, in A
    axis: np.ndarray                 # unit vector
    anisotropy: float = 1.0
    fractional: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def kind(self) -> str:
        """``channel`` when the free space is at least twice as long one way, else ``cage``."""
        return "channel" if self.anisotropy >= 2.0 else "cage"


_TREES: Dict[tuple, tuple] = {}


def free_distance(atoms, points: np.ndarray, radii: Optional[Dict[str, float]] = None,
                  neighbours: int = 12) -> np.ndarray:
    """Distance from each point to the nearest atom's van der Waals surface, every image counted.

    Negative inside an atom. `radii` overrides the radii of :data:`VDW` by element; a radius of
    zero puts the surface at the nucleus, and an element given a negative radius is left out
    (for a zeolite, whose silicon sits inside a tetrahedron of oxygens, the oxygens alone define
    the pores, which is how pore sizes of zeolites are usually quoted).
    """
    from scipy.spatial import cKDTree

    table = dict(VDW, **(radii or {}))
    cell = np.asarray(atoms.cell, float)
    symbols = atoms.get_chemical_symbols()
    r = np.array([table.get(s, DEFAULT_RADIUS) for s in symbols])
    use = r >= 0
    pos = _home_positions(atoms)[use]
    r = r[use]
    pts = np.atleast_2d(np.asarray(points, float))
    key = (id(atoms), len(atoms), float(cell.sum()), tuple(np.round(r, 4)))
    cached = _TREES.get(key)
    if cached is None:
        shifts = _image_shifts(cell, 12.0)
        imgs = (pos[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
        cached = (cKDTree(imgs), np.tile(r, len(shifts)))
        _TREES.clear()
        _TREES[key] = cached
    tree, rad = cached
    k = min(int(neighbours), tree.n)
    d, idx = tree.query(pts, k=k)
    s = np.asarray(d, float).reshape(len(pts), k) - rad[np.asarray(idx).reshape(len(pts), k)]
    return s.min(axis=1)


def find_pores(atoms, spacing: float = 0.3, min_radius: float = 1.0, axis_reach: float = 6.0,
               probe: float = 1.0, radii: Optional[Dict[str, float]] = None,
               merge: float = 1.0) -> List[Pore]:
    """The pores of a framework: the local maxima of the free distance over one cell.

    The free distance (:func:`free_distance`) is evaluated on a grid over the cell. A grid point
    is a maximum when none of its 26 neighbours (across the cell boundary as well) is larger;
    each maximum is then refined by a local maximisation of the exact function. A maximum
    smaller than `min_radius` is a gap between atoms, not a pore, and maxima closer than `merge`
    angstrom, counting periodic images, are one pore. The axis is the longest principal
    direction of the free space within `axis_reach` of the centre (the points whose free
    distance exceeds `probe`). Returned largest first.

    A macrocycle such as a cyclodextrin has no maximum in its cavity when it stands alone: the
    cavity opens on both sides, so the free distance keeps growing along its axis. Its frame
    comes from its units (:func:`cmfqsar.protein.cavity_frame`), not from here.
    """
    from scipy.optimize import minimize

    cell = np.asarray(atoms.cell, float)
    lengths = np.linalg.norm(cell, axis=1)
    shape = tuple(max(int(np.ceil(L / spacing)), 4) for L in lengths)
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    frac = idx.reshape(-1, 3) / np.asarray(shape, float)
    pts = _origin(atoms) + frac @ cell
    F = free_distance(atoms, pts, radii).reshape(shape)
    is_max = F >= min_radius
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == dj == dk == 0:
                    continue
                is_max &= F >= np.roll(F, (-di, -dj, -dk), axis=(0, 1, 2))
    cand = np.argwhere(is_max)
    cand = cand[np.argsort(-F[tuple(cand.T)])]
    step = float(np.max(np.linalg.norm(cell / np.asarray(shape, float)[:, None], axis=1)))
    pores: List[Pore] = []
    for c in cand:
        x0 = _origin(atoms) + (c / np.asarray(shape, float)) @ cell
        if any(_periodic_distance(x0, p.centre, cell) < merge for p in pores):
            continue
        res = minimize(lambda x: -float(free_distance(atoms, x[None], radii)[0]), x0,
                       method="Nelder-Mead",
                       options={"xatol": 1e-4, "fatol": 1e-6, "initial_simplex":
                                x0 + np.vstack([np.zeros(3), np.eye(3) * 0.5 * step])})
        x = res.x if -res.fun >= F[tuple(c)] else x0
        if any(_periodic_distance(x, p.centre, cell) < merge for p in pores):
            continue
        radius = float(free_distance(atoms, x[None], radii)[0])
        axis, aniso = _pore_axis(atoms, x, axis_reach, probe, spacing, radii)
        fr = np.linalg.solve(cell.T, x - _origin(atoms)) % 1.0
        pores.append(Pore(x, radius, axis, aniso, fr))
    pores.sort(key=lambda p: -p.radius)
    return pores


def _periodic_distance(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> float:
    d = np.linalg.solve(cell.T, np.asarray(a, float) - np.asarray(b, float))
    d -= np.round(d)
    return float(np.linalg.norm(d @ cell))


def _pore_axis(atoms, centre, reach, probe, spacing, radii) -> Tuple[np.ndarray, float]:
    """The longest principal direction of the free space around a centre, and its anisotropy."""
    n = int(np.ceil(reach / spacing))
    g = np.arange(-n, n + 1) * spacing
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts = pts[np.linalg.norm(pts, axis=1) <= reach]
    free = free_distance(atoms, centre + pts, radii) > probe
    sub = pts[free]
    if len(sub) < 4:
        return np.array([0.0, 0.0, 1.0]), 1.0
    sub = sub - sub.mean(axis=0)
    w, v = np.linalg.eigh(sub.T @ sub / len(sub))
    axis = v[:, -1] / np.linalg.norm(v[:, -1])
    k = int(np.argmax(np.abs(axis)))
    if axis[k] < 0:
        axis = -axis                       # a fixed sign, so the same pore gives the same frame
    return axis, float(np.sqrt(max(w[-1], 1e-12) / max(w[0], 1e-12)))


def nearest_pore(pores: Sequence[Pore], point: np.ndarray, atoms) -> Pore:
    """The pore whose centre, in any periodic image, is nearest `point`."""
    if not pores:
        raise ValueError("no pore was found")
    cell = np.asarray(atoms.cell, float)
    return min(pores, key=lambda p: _periodic_distance(p.centre, np.asarray(point, float), cell))


# --------------------------------------------------------------------------- #
# The pocket: a finite cut of the images around a point
# --------------------------------------------------------------------------- #

def framework_pocket(atoms, centre: np.ndarray, radius: float = 12.0,
                     units: Optional[Sequence[int]] = None, unit_name: str = FRAMEWORK_RESNAME,
                     types: Optional[Sequence[str]] = None, name: str = "",
                     whole_molecules: bool = False) -> Protein:
    """Every atom of every image within `radius` of `centre`, as a finite pocket.

    The atoms of the image nearest the centre come first and in the order of the structure; the
    other images follow in order of distance. The "residues" are the framework atom types of
    :func:`atom_types` (or `types`): every type is one residue, named ``FWK1``, ``FWK2``, ...,
    and ``pocket.type_names`` maps the residue number back to the type, so a per-residue profile
    is a profile per atom type. With `units` (an integer per atom, such as the sugars of a
    cyclodextrin) the residues are those units instead, named as a molecule file names them. An
    atom keeps its index in the structure as ``source_index`` and its image as ``source_image``.

    `whole_molecules` keeps every atom of a molecule that has any atom inside the radius, with
    the molecule unwrapped across the cell boundary. That is right for a crystal of discrete
    molecules and impossible for a covalent framework, whose one molecule is infinite, so it is
    refused there.
    """
    cell = np.asarray(atoms.cell, float)
    pos = _home_positions(atoms)
    centre = np.asarray(centre, float)
    symbols = atoms.get_chemical_symbols()
    if types is None and units is None:
        types = atom_types(atoms)
    unwrap = np.zeros((len(atoms), 3), int)
    if whole_molecules:
        found = _molecules(atoms)
        if found is None:
            raise ValueError("whole_molecules was asked for, but the structure is one connected "
                             "framework: its molecule is infinite")
        mol_of, unwrap = found
    upos = pos + unwrap @ cell
    span = float(np.max(np.linalg.norm(upos - upos.mean(axis=0), axis=1))) if len(pos) else 0.0
    shifts = _image_shifts(cell, radius + span + float(np.linalg.norm(cell.sum(axis=0))))
    cart = shifts @ cell
    d = np.linalg.norm(upos[None, :, :] + cart[:, None, :] - centre, axis=2)   # (images, atoms)
    keep = d <= radius
    if whole_molecules:
        for s in range(len(shifts)):
            for m in set(mol_of[keep[s]].tolist()):
                keep[s, mol_of == m] = True
    nearest = np.where(keep, d, np.inf).min(axis=1)
    used = [s for s in range(len(shifts)) if keep[s].any()]
    home = min(used, key=lambda s: (nearest[s], s))
    seq = [home] + sorted((s for s in used if s != home), key=lambda s: (nearest[s], s))
    coords, elements, names, resnames, resids, src, img = [], [], [], [], [], [], []
    type_ids: Dict[str, int] = {}
    counts: Dict[int, int] = {}
    for s in seq:
        for a in np.flatnonzero(keep[s]):
            coords.append(upos[a] + cart[s])
            elements.append(symbols[a])
            res = (int(units[a]) + 1 if units is not None
                   else type_ids.setdefault(types[a], len(type_ids) + 1))
            counts[res] = counts.get(res, 0) + 1
            names.append(f"{symbols[a]}{counts[res]}"[:4])
            resnames.append(unit_name)
            resids.append(res)
            src.append(int(a))
            img.append(tuple(int(v) for v in (shifts[s] + unwrap[a])))
    n = len(coords)
    p = Protein(name=name or "framework", coords=np.asarray(coords, float), elements=elements,
                atom_names=names, resnames=resnames, resids=np.asarray(resids, int),
                icodes=[""] * n, chains=["A"] * n, hetatm=np.zeros(n, bool), path="")
    p.source_index = np.asarray(src, int)
    p.source_image = np.asarray(img, int).reshape(-1, 3)
    p.type_names = {v: k for k, v in type_ids.items()}
    return p


def _molecules(atoms):
    """(molecule per atom, image per atom that makes each molecule whole), or None if infinite.

    A molecule is infinite when following its bonds returns to one of its atoms in another image.
    """
    n = len(atoms)
    adj: Dict[int, List[Tuple[int, np.ndarray]]] = {i: [] for i in range(n)}
    for a, b, s in periodic_bonds(atoms):
        adj[a].append((b, np.array(s, int)))
        adj[b].append((a, -np.array(s, int)))
    image = np.zeros((n, 3), int)
    mol = np.full(n, -1, int)
    m = 0
    for start in range(n):
        if mol[start] >= 0:
            continue
        mol[start] = m
        stack = [start]
        while stack:
            a = stack.pop()
            for b, s in adj[a]:
                want = image[a] + s
                if mol[b] < 0:
                    mol[b] = m
                    image[b] = want
                    stack.append(b)
                elif not np.array_equal(image[b], want):
                    return None
        m += 1
    return mol, image


def framework_pocket_fields(pocket: Protein, fieldset, atoms=None, charges: str = "gasteiger",
                            add_hydrogens: bool = False):
    """The pocket's fields with the guests' field set, and charges that suit the framework.

    `charges` is ``"gasteiger"`` for an organic host (the guests' own scheme, through the
    pocket path of `cmfqsar.complementarity`), or ``"oxidation"``, ``"electronegativity"``,
    ``"dft"`` or ``"file"`` for an inorganic framework, in which case the charges of
    :func:`cmfqsar.periodic.crystal_charges` on `atoms` are put on the pocket atoms they came
    from. Hydrogens are not added by default: a framework file carries the ones it has, and
    hydrogens on the atoms cut at the edge of the pocket would be donors that are not there.
    """
    from .complementarity import pocket_fields, pocket_molecule
    from .fields import FieldSet, build_fields

    if charges == "gasteiger":
        return pocket_fields(pocket, fieldset, add_hydrogens=add_hydrogens,
                             ionise_residues=False)
    if atoms is None:
        raise ValueError(f"charges={charges!r} needs the periodic structure (atoms=)")
    from .periodic import crystal_charges

    q, _source = crystal_charges(atoms, charges)
    mol = pocket_molecule(pocket, add_hydrogens=False, ionise_residues=False)
    if mol.GetNumAtoms() != pocket.n_atoms:
        raise ValueError(f"the pocket has {pocket.n_atoms} atoms and its molecule "
                         f"{mol.GetNumAtoms()}; charges cannot be matched to atoms")
    for a, src in zip(mol.GetAtoms(), pocket.source_index):
        a.SetDoubleProp("partial_charge", float(q[src]))
    fs = FieldSet(specs=fieldset.specs, charge_model="file", cif_scheme=fieldset.cif_scheme,
                  param_scheme=fieldset.param_scheme)
    for attr in ("atom_properties", "probes", "descriptors"):
        if hasattr(fieldset, attr):
            try:
                setattr(fs, attr, getattr(fieldset, attr))
            except Exception:                         # noqa: BLE001 - frozen or derived
                pass
    return build_fields(mol, fs, name=pocket.name or "framework")


def type_profile(profile, pocket: Protein) -> Dict[str, Dict[str, float]]:
    """An interaction profile of the pocket code, keyed by framework atom type.

    `profile` is the :class:`cmfqsar.complementarity.InteractionProfile` of a guest against a
    pocket from :func:`framework_pocket`, whose residues are the atom types; this gives them
    their names back. A pocket cut with `units` keeps its unit labels.
    """
    names = getattr(pocket, "type_names", {}) or {}
    label_type = {}
    for i in range(pocket.n_atoms):
        label_type[pocket.residue_label(i)] = names.get(int(pocket.resids[i]),
                                                        pocket.residue_label(i))
    t = np.asarray(profile.terms, float)
    out: Dict[str, Dict[str, float]] = {}
    for r, res in enumerate(profile.residues):
        key = label_type.get(res, res)
        row = out.setdefault(key, {lab: 0.0 for lab in profile.labels})
        for k, lab in enumerate(profile.labels):
            row[lab] += float(t[r, k])
    return out
