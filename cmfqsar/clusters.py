"""
A finite cluster cut from a periodic structure around one site, its cut bonds capped with
hydrogen, read as a metal complex (plan item EC8).

A single-atom catalyst is a metal held by four atoms of a graphene sheet: a molecular active
site inside a solid.  The molecular side of the program -- the five classical fields, the
Gasteiger charges, the metal parameters of plan item EC4 -- reads a molecule, not a cell.  This
module makes the molecule:

1. **The cut.** The atoms within a number of bonds (`hops`) of the site are kept, the bonds
   being found from the covalent radii over the periodic images, so a site at the edge of the
   cell is cut whole. The site itself is bonded to its first coordination shell only (the atoms
   within ``FIRST_SHELL`` times its nearest distance): the covalent radius of a large metal such
   as La reaches the second shell of a graphene sheet, and the cluster would then grow with the
   metal instead of being the same cut for the whole series. An atom left with a single
   neighbour inside the cluster is removed, again and again, so that no chain dangles from the
   edge.
2. **The caps.**  Every bond the cut breaks becomes a hydrogen atom on the kept atom, placed
   along the broken bond at the C-H (1.09 A) or N-H (1.01 A) length, which is the usual way a
   cluster model of a surface or a sheet is terminated.
3. **The complex.**  :func:`cmfqsar.complexes.build_complex` turns the coordinates into an RDKit
   molecule with dative metal-ligand bonds.  The ligand -- everything but the metal -- needs a
   charge before its bond orders can be found, and that charge decides the formal charge the
   metal gets, since the cluster as a whole is neutral.  The charges are tried in turn and the
   first that lets the ligand sanitise is kept.  For a capped divacancy that is minus the number
   of carbon donors: each carbon that holds the metal is a carbanion with a lone pair, and a
   pyridine-like nitrogen has one of its own.  So the metal of an M@C4 site is +4 and that of an
   M@N4 site 0, whatever the metal: the formal charge counts the donors, it is not an oxidation
   state.  A charge that cannot close the shell is not offered (the ligand of an M@C4 site has
   no closed-shell form at -2), so `ligand_charge` is a request, and the report of
   :func:`cluster_complex` says which charge was used.

The cluster keeps the coordinates of the structure, so a series cut from one template stays in
the frame of that template.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: bond lengths of the caps, by the element they sit on, in angstrom
CAP_LENGTH = {"C": 1.09, "N": 1.01, "O": 0.97, "S": 1.34, "B": 1.19, "P": 1.42}
#: two atoms are bonded when closer than this times the sum of their covalent radii
BOND_FACTOR = 1.15
#: the site bonds only to its first shell: the atoms within this times its nearest distance
FIRST_SHELL = 1.25
#: the ligand charge of a cluster: None tries the charges in turn (see the module text)
LIGAND_CHARGE = None


@dataclass
class Cluster:
    """A cut and capped cluster: element symbols and coordinates, the site first."""

    symbols: List[str]
    coords: np.ndarray
    kept: List[int]                    # indices into the structure, the site first
    n_caps: int
    hops: int
    notes: List[str] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def _neighbours(atoms, factor: float = BOND_FACTOR):
    """Bonded neighbours of every atom and the vector to each, over the periodic images."""
    from ase.data import covalent_radii
    from ase.neighborlist import neighbor_list
    cut = factor * covalent_radii[atoms.get_atomic_numbers()]
    i, j, D = neighbor_list("ijD", atoms, cut)
    nb: List[List[Tuple[int, np.ndarray]]] = [[] for _ in range(len(atoms))]
    for p, q, d in zip(i, j, D):
        nb[p].append((int(q), np.asarray(d, float)))
    return nb


def cut_cluster(atoms, site: int, hops: int = 4, factor: float = BOND_FACTOR) -> Cluster:
    """The atoms within `hops` bonds of `site`, the dangling ones pruned, the cut bonds capped.

    Coordinates are unwrapped from the site outwards, so a cluster that crosses the boundary of
    the cell comes out whole and next to the site.
    """
    if hops < 1:
        raise ValueError("a cluster needs at least the first shell: hops >= 1")
    nb = _neighbours(atoms, factor)
    if nb[site]:
        dmin = min(float(np.linalg.norm(d)) for _, d in nb[site])
        first = {q for q, d in nb[site] if np.linalg.norm(d) <= FIRST_SHELL * dmin}
        nb[site] = [(q, d) for q, d in nb[site] if q in first]
        for p in range(len(nb)):
            if p != site and p not in first:
                nb[p] = [(q, d) for q, d in nb[p] if q != site]
    pos = {site: np.asarray(atoms.positions[site], float)}
    depth = {site: 0}
    front = [site]
    for h in range(hops):
        nxt = []
        for p in front:
            for q, d in nb[p]:
                if q not in depth:
                    depth[q] = h + 1
                    pos[q] = pos[p] + d
                    nxt.append(q)
        front = nxt
    keep = set(depth)
    changed = True
    while changed:
        changed = False
        for p in sorted(keep):
            if p != site and sum(q in keep for q, _ in nb[p]) < 2:
                keep.discard(p)
                changed = True
    kept = [site] + sorted(keep - {site})
    symbols = [atoms[p].symbol for p in kept]
    coords = [pos[p] for p in kept]
    n_caps = 0
    notes: List[str] = []
    for p in kept[1:]:
        for q, d in nb[p]:
            if q in keep:
                continue
            length = CAP_LENGTH.get(atoms[p].symbol)
            if length is None:
                notes.append(f"a cut bond of {atoms[p].symbol} has no cap length; capped at 1.0 A")
                length = 1.0
            symbols.append("H")
            coords.append(pos[p] + length * d / np.linalg.norm(d))
            n_caps += 1
    return Cluster(symbols=symbols, coords=np.asarray(coords, float), kept=kept, n_caps=n_caps,
                   hops=int(hops), notes=notes)


def cluster_complex(atoms, site: int, hops: int = 4, ligand_charge: Optional[int] = LIGAND_CHARGE,
                    total_charge: int = 0, name: str = "", origin: Optional[Sequence[float]] = None,
                    flip_z: bool = False):
    """The capped cluster as an RDKit metal complex: (molecule or None, cluster, report).

    `origin` moves the coordinates so that this point of the structure becomes the origin, and
    `flip_z` turns the cluster over (z to -z), which is how a series of sheets is put in one
    frame with the metal on the +z side.  The molecule is None when the complex cannot be built,
    and the report says why.  `ligand_charge` is tried first when it is given; when the ligand
    has no closed shell at that charge, the others are tried, and ``report.ligand_charges`` and
    ``report.metal_charge`` say what was used.
    """
    from .complexes import build_complex
    cl = cut_cluster(atoms, site, hops)
    xyz = cl.coords.copy()
    if origin is not None:
        # the cluster is unwrapped around the site, which keeps its own position, so a point of
        # the structure near the site is the origin as it is
        xyz -= np.asarray(origin, float)
    if flip_z:
        xyz[:, 2] *= -1.0
    cl.coords = xyz
    mol, report = build_complex(cl.symbols, xyz, total_charge=int(total_charge),
                                ligand_charges=None if ligand_charge is None
                                else [int(ligand_charge)], name=name, metal=atoms[site].symbol)
    return mol, cl, report


__all__ = ["BOND_FACTOR", "CAP_LENGTH", "Cluster", "FIRST_SHELL", "LIGAND_CHARGE",
           "cluster_complex", "cut_cluster"]
