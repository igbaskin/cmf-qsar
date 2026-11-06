"""
Bonds of a periodic structure, perceived from distances, for drawing only.

A crystal read from a CIF or an extended-XYZ file has no bond table: a file of that kind states
where the atoms are and nothing about which of them are bonded.  The molecular view of this
program draws bonds because an RDKit molecule carries them; the crystal view drew spheres alone.
For an inorganic framework that is the honest picture.  For an **organic molecule adsorbed on a
surface**, which is the case items EC2 and EC6 are about, it is not: a reader cannot tell a ring
from a chain in a cloud of spheres.

This module supplies the missing list.  Everything here is **display-only perception**: it
decides which pairs of atoms to draw a stick between and nothing else.  No molecule is
sanitised, no bond order is assigned and no charge is perceived from it -- those belong to plan
item M8, which turns a periodic structure into a molecule, and a bond drawn here never reaches a
field, a kernel or a prediction.

The rule, and the two things it gets right
------------------------------------------
A pair is bonded when ``|r_ij| < R_i + R_j + tolerance`` with covalent radii R and a tolerance of
0.45 angstrom, which is the rule :func:`cmfqsar.protein.covalent_bonds` already applies to a
protein.  Two things are added for the periodic case:

* **the images.**  The pairs are taken over the cell *and* its periodic images, so a bond that
  crosses the cell boundary is found rather than lost.  A benzene ring that a file leaves split
  across the boundary is drawn as a ring, with the sticks leaving the box on one side and the
  image atoms they end on drawn with them.
* **metals get no bonds.**  A distance rule gives every atom of a close-packed metal about twelve
  neighbours, and twelve sticks per atom is noise, not information: it hides the structure it is
  meant to show.  So the rule is applied to the non-metal atoms only (plan item NT8 states it as
  "no bond perception for metals").  A slab therefore stays spheres and the organic molecule on
  it gets its bonds, which is exactly the picture the adsorbate case needs.

What a metal is told instead: :func:`coordination_polyhedra` draws the first coordination shell
of a metal atom as a polyhedron, which is how an inorganic structure is read -- a chain of
corner-sharing octahedra says more than any number of sticks would.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .protein import COVALENT_RADII, covalent_bonds

#: the elements a stick is drawn to: the non-metals, and the metalloids that form covalent
#: frameworks (boron, silicon, germanium, arsenic, antimony, tellurium).  Every other element is
#: treated as a metal and gets no bonds; see the module docstring for why.
BONDING_ELEMENTS = frozenset({
    "H", "D", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar",
    "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn",
})
#: added to the sum of two covalent radii before a pair counts as bonded, in angstrom
BOND_TOLERANCE = 0.45
#: no stick is drawn longer than this, whatever the radii say
MAX_BOND = 2.4
#: covalent radius used for an element no table knows, in angstrom
DEFAULT_RADIUS = 0.77
#: a drawing of more sticks than this is built from lines instead of cylinders
MANY_BONDS = 4000
#: the first coordination shell of a metal reaches this far times its shortest neighbour distance
SHELL_RATIO = 1.30
#: polyhedra drawn at most, so that a large cell does not spend a second on them
MAX_POLYHEDRA = 400


def covalent_radius(symbol: str) -> float:
    """The covalent radius of an element in angstrom.

    The table of :mod:`cmfqsar.protein` first, because the organic elements should have the same
    radius in every part of the program; then ASE's table, which covers every element a crystal
    may hold; then :data:`DEFAULT_RADIUS`.
    """
    s = (symbol or "").strip()
    if not s:
        return DEFAULT_RADIUS
    if s.upper() in COVALENT_RADII:
        return float(COVALENT_RADII[s.upper()])
    try:
        from ase.data import atomic_numbers, covalent_radii
        return float(covalent_radii[atomic_numbers[s.capitalize()]])
    except Exception:                               # noqa: BLE001 - no ASE, or an unknown symbol
        return DEFAULT_RADIUS


def is_bonding(symbol: str) -> bool:
    """Whether a stick may be drawn to an atom of this element.  A metal: no."""
    return (symbol or "").strip().capitalize() in BONDING_ELEMENTS


def perceive_bonds(coords: np.ndarray, symbols: Sequence[str],
                   tolerance: float = BOND_TOLERANCE, max_bond: float = MAX_BOND,
                   skip_metals: bool = True) -> np.ndarray:
    """Bonded pairs of an open (non-periodic) set of atoms, as an (n, 2) array of indices.

    With `skip_metals` a pair counts only when **both** atoms are in
    :data:`BONDING_ELEMENTS`; without it the distance rule is applied to every pair, which is
    what a reader asks for when the structure is a molecular crystal of a metal complex and the
    metal-ligand bonds are the point.
    """
    xyz = np.asarray(coords, float).reshape(-1, 3)
    syms = list(symbols)
    if len(xyz) < 2:
        return np.zeros((0, 2), dtype=int)
    if not skip_metals:
        return covalent_bonds(xyz, syms, tolerance=tolerance, max_bond=max_bond)
    keep = np.flatnonzero(np.array([is_bonding(s) for s in syms], bool))
    if len(keep) < 2:
        return np.zeros((0, 2), dtype=int)
    inner = covalent_bonds(xyz[keep], [syms[int(i)] for i in keep], tolerance=tolerance,
                           max_bond=max_bond)
    return keep[inner] if len(inner) else np.zeros((0, 2), dtype=int)


# --------------------------------------------------------------------------- #
# The periodic case
# --------------------------------------------------------------------------- #
@dataclass
class DrawableStructure:
    """What a viewer needs to draw one cell with its bonds.

    The first `n_cell` rows are the atoms of the cell, **in the cell's own order**, so a site
    number in the Reading table still names the same row.  The rows after them are the image
    atoms that a bond reaches, and only those: an image that is bonded to nothing is not drawn,
    because the images have a drawing of their own.
    """
    coords: np.ndarray
    symbols: List[str]
    radii: np.ndarray
    bonds: np.ndarray
    n_cell: int
    #: for every row, the atom of the cell it is an image of (the identity for the first n_cell)
    source: np.ndarray
    skipped_metals: int = 0
    #: distinct bonds of the structure.  A bond that crosses the cell boundary is **drawn twice**,
    #: once from each of its two ends, because both halves of it are inside the box; this is the
    #: number of chemical bonds behind those sticks, which is the number worth reporting
    n_distinct: int = 0

    @property
    def n_images_drawn(self) -> int:
        return int(len(self.coords) - self.n_cell)

    @property
    def n_crossing(self) -> int:
        """Sticks that leave the cell: one end in it, the other on an image atom."""
        if not len(self.bonds):
            return 0
        a, b = self.bonds[:, 0], self.bonds[:, 1]
        return int(np.sum((a >= self.n_cell) | (b >= self.n_cell)))

    def metal_note(self, n_atoms: Optional[int] = None) -> str:
        """How many atoms got no bonds because they are metal atoms."""
        if not self.skipped_metals:
            return ""
        total = int(n_atoms if n_atoms is not None else self.n_cell)
        if self.skipped_metals >= total:
            return f"all {total} atoms are metal atoms, which get none"
        return f"{self.skipped_metals} of the {total} atoms are metal atoms, which get none"

    def describe(self) -> str:
        if not len(self.bonds):
            note = self.metal_note()
            return f"no bonds: {note}" if note else "no bonds: no pair of atoms is close enough"
        s = f"{self.n_distinct} bonds"
        if self.n_crossing:
            s += (f", {self.n_distinct - (len(self.bonds) - self.n_crossing)} of them crossing "
                  f"the cell boundary and drawn from both ends "
                  f"({len(self.bonds)} sticks in all)")
        note = self.metal_note()
        if note:
            s += "; " + note
        return s


def periodic_bonds(crystal, tolerance: float = BOND_TOLERANCE, max_bond: float = MAX_BOND,
                   skip_metals: bool = True) -> DrawableStructure:
    """The bonds of one cell, with the images a bond reaches (display only).

    The images are expanded to `max_bond`, which is enough: no stick is longer than that, so any
    bonding partner outside the cell lies within `max_bond` of the box.  A pair of two image
    atoms is dropped -- it is a bond of a neighbouring cell and the cell's own copy of it is
    already drawn -- and a pair with one end in the cell is kept, so a bond that crosses the
    boundary is drawn leaving the box, with the atom it ends on drawn beside it.
    """
    from .periodic import cell_image_indices

    n_cell = int(getattr(crystal, "n_atoms", len(crystal.coords)))
    symbols = list(crystal.symbols)
    radii = np.asarray(getattr(crystal, "radii", np.full(n_cell, DEFAULT_RADIUS)), float)
    skipped = int(sum(1 for s in symbols if not is_bonding(s))) if skip_metals else 0
    try:
        coords, source = cell_image_indices(crystal, float(max_bond))
    except (ValueError, AttributeError):
        # no cell: an open structure, which the plain rule already covers
        coords = np.asarray(crystal.coords, float).reshape(-1, 3)
        source = np.arange(len(coords))
    all_symbols = [symbols[int(i)] for i in source]
    pairs = perceive_bonds(coords, all_symbols, tolerance=tolerance, max_bond=max_bond,
                           skip_metals=skip_metals)
    if len(pairs):
        inside = (pairs[:, 0] < n_cell) | (pairs[:, 1] < n_cell)
        pairs = pairs[inside]
    if not len(pairs):
        return DrawableStructure(coords=np.asarray(coords, float)[:n_cell],
                                 symbols=symbols[:n_cell], radii=radii[:n_cell],
                                 bonds=np.zeros((0, 2), dtype=int), n_cell=n_cell,
                                 source=np.arange(n_cell), skipped_metals=skipped,
                                 n_distinct=0)
    # keep the cell, and of the images only those a kept bond ends on
    used = np.unique(pairs)
    extra = np.sort(used[used >= n_cell])
    order = np.concatenate([np.arange(n_cell), extra]).astype(int)
    remap = np.full(len(coords), -1, dtype=int)
    remap[order] = np.arange(len(order))
    return DrawableStructure(
        coords=coords[order], symbols=[all_symbols[int(i)] for i in order],
        radii=radii[source[order]], bonds=remap[pairs], n_cell=n_cell,
        source=source[order], skipped_metals=skipped,
        n_distinct=_distinct_bonds(pairs, coords, source))


def _distinct_bonds(pairs: np.ndarray, coords: np.ndarray, source: np.ndarray) -> int:
    """How many chemical bonds the sticks stand for.

    A bond that crosses the cell boundary is drawn from both of its ends, so it appears twice in
    `pairs`: once as (atom i, image of j) and once as (atom j, image of i).  The two copies join
    the same pair of atoms of the cell and their lattice translations are opposite, so the key
    below -- the unordered pair of source atoms with the length of the translation -- is the same
    for both and different for two genuinely different bonds between the same pair of sites.
    """
    seen = set()
    for a, b in pairs:
        sa, sb = int(source[int(a)]), int(source[int(b)])
        t = coords[int(b)] - coords[int(a)]
        key = (min(sa, sb), max(sa, sb), round(float(np.linalg.norm(t)), 3))
        seen.add(key)
    return len(seen)


# --------------------------------------------------------------------------- #
# What a metal gets instead of bonds
# --------------------------------------------------------------------------- #
@dataclass
class Polyhedron:
    """The first coordination shell of one atom, as a closed surface to draw."""
    centre: int                       # the atom of the cell it belongs to
    element: str
    vertices: np.ndarray              # (n, 3) the neighbours, image atoms included
    faces: np.ndarray                 # (m, 3) triangles over `vertices`
    neighbours: List[str] = field(default_factory=list)
    planar: bool = False              # the shell is flat, so the "polyhedron" is a polygon

    @property
    def coordination(self) -> int:
        return int(len(self.vertices))

    def label(self) -> str:
        kinds: Dict[str, int] = {}
        for s in self.neighbours:
            kinds[s] = kinds.get(s, 0) + 1
        what = "".join(f"{k}{v}" if v > 1 else k for k, v in sorted(kinds.items()))
        return f"{self.element} {self.coordination}-coordinate ({what})"


def _polygon_faces(points: np.ndarray) -> np.ndarray:
    """Triangles of a flat set of points: the polygon in its own plane, as a fan.

    A square-planar or a trigonal-planar shell has no volume, so a convex hull of it fails.  It
    is still a shape worth drawing, and this is that shape: the points ordered by angle about
    their centroid in the plane they span, fanned into triangles.
    """
    c = points.mean(axis=0)
    rel = points - c
    _u, _s, vt = np.linalg.svd(rel, full_matrices=False)
    e1, e2 = vt[0], vt[1]
    angle = np.arctan2(rel @ e2, rel @ e1)
    order = np.argsort(angle)
    faces = [[int(order[0]), int(order[k]), int(order[k + 1])] for k in range(1, len(order) - 1)]
    return np.asarray(faces, dtype=int) if faces else np.zeros((0, 3), dtype=int)


def _expanded(crystal, reach: float):
    """(coordinates, source index, symbols) of the cell and the images within `reach`."""
    from .periodic import cell_image_indices

    symbols = list(crystal.symbols)
    try:
        coords, source = cell_image_indices(crystal, float(reach))
    except (ValueError, AttributeError):
        coords = np.asarray(crystal.coords, float).reshape(-1, 3)
        source = np.arange(len(coords))
    return coords, source, [symbols[int(i)] for i in source]


def _shell(coords: np.ndarray, all_symbols: Sequence[str], tree, i: int, ratio: float,
           max_shell: float) -> List[int]:
    """The first coordination shell of atom `i`: the indices of its vertices.

    The nearest neighbour sets a distance and every neighbour within `ratio` times that distance
    belongs to the shell.  Where the atom has non-metal neighbours, only those are considered,
    which makes the vertices of an ionic structure its anions rather than the next cations out.
    """
    near = [k for k in tree.query_ball_point(coords[i], float(max_shell)) if k != i]
    if not near:
        return []
    d = np.linalg.norm(coords[near] - coords[i], axis=1)
    anions = [k for k, kk in enumerate(near) if is_bonding(all_symbols[kk])]
    if anions:
        near = [near[k] for k in anions]
        d = d[anions]
    cut = float(d.min()) * float(ratio)
    return [k for k, dist in zip(near, d) if dist <= cut]


def coordination_numbers(crystal, ratio: float = SHELL_RATIO, max_shell: float = 3.6,
                         centres: str = "metals") -> List[Tuple[int, str, int]]:
    """``(atom, element, coordination)`` for every candidate centre, small shells included.

    :func:`coordination_polyhedra` leaves out a shell too small to be a shape; this reports every
    centre, so that a count of the polyhedra drawn can be read against the number of atoms that
    could have had one.  The shell rule is the same one, shared, not repeated.
    """
    n_cell = int(getattr(crystal, "n_atoms", len(crystal.coords)))
    coords, _source, all_symbols = _expanded(crystal, float(max_shell) + 1.0)
    tree = cKDTree(coords)
    out: List[Tuple[int, str, int]] = []
    for i in range(n_cell):
        metal = not is_bonding(crystal.symbols[i])
        if centres == "metals" and not metal:
            continue
        if centres == "non-metals" and metal:
            continue
        out.append((i, str(crystal.symbols[i]),
                    len(_shell(coords, all_symbols, tree, i, ratio, max_shell))))
    return out


def coordination_polyhedra(crystal, ratio: float = SHELL_RATIO, max_shell: float = 3.6,
                           min_coordination: int = 3, centres: str = "metals",
                           limit: int = MAX_POLYHEDRA) -> List[Polyhedron]:
    """The first coordination shell of every metal atom of a cell, as polyhedra.

    The shell is found without any chemistry: the nearest neighbour of the centre sets a
    distance, and every neighbour within `ratio` times that distance is a vertex.  That is the
    usual rule, and it needs no parameter per element.  Neighbours are taken over the periodic
    images, so a polyhedron at the edge of the cell is whole; non-metal neighbours are preferred
    where the centre has any, which is what makes the vertices the anions of an ionic structure
    rather than the next cations out.

    `centres` is ``"metals"`` (the atoms that get no bonds, which is what a polyhedron is for),
    ``"all"`` or ``"non-metals"``.  A shell of fewer than `min_coordination` neighbours is left
    out, and a flat shell is drawn as a polygon with :attr:`Polyhedron.planar` set.
    """
    from scipy.spatial import ConvexHull, QhullError

    n_cell = int(getattr(crystal, "n_atoms", len(crystal.coords)))
    symbols = list(crystal.symbols)
    coords, _source, all_symbols = _expanded(crystal, float(max_shell) + 1.0)
    tree = cKDTree(coords)
    out: List[Polyhedron] = []
    for i in range(n_cell):
        metal = not is_bonding(symbols[i])
        if centres == "metals" and not metal:
            continue
        if centres == "non-metals" and metal:
            continue
        chosen = _shell(coords, all_symbols, tree, i, ratio, max_shell)
        if len(chosen) < int(min_coordination):
            continue
        pts = coords[chosen]
        planar = False
        try:
            hull = ConvexHull(pts)
            faces = np.asarray(hull.simplices, dtype=int)
        except (QhullError, ValueError):
            faces = _polygon_faces(pts)
            planar = True
        if not len(faces):
            continue
        out.append(Polyhedron(centre=i, element=symbols[i], vertices=pts, faces=faces,
                              neighbours=[all_symbols[int(k)] for k in chosen], planar=planar))
        if len(out) >= int(limit):
            break
    return out


def polyhedra_mesh(polyhedra: Sequence[Polyhedron]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Every polyhedron as one mesh, so a cell of hundreds costs one drawing item."""
    verts, faces, off = [], [], 0
    for p in polyhedra:
        verts.append(p.vertices)
        faces.append(p.faces + off)
        off += len(p.vertices)
    if not verts:
        return None
    return np.concatenate(verts), np.concatenate(faces)


__all__ = ["BONDING_ELEMENTS", "BOND_TOLERANCE", "DEFAULT_RADIUS", "DrawableStructure",
           "MANY_BONDS", "MAX_BOND", "MAX_POLYHEDRA", "Polyhedron", "SHELL_RATIO",
           "coordination_numbers", "coordination_polyhedra", "covalent_radius", "is_bonding",
           "perceive_bonds", "periodic_bonds", "polyhedra_mesh"]
