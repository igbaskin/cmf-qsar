"""Growing a bound fragment into the gap the pocket leaves (plan item MC7).

A fragment that binds fills part of a site and leaves the rest empty.  What should be put there?
:mod:`cmfqsar.complementarity` already scores a whole ligand against a pocket by field overlap,
with the sign convention that makes a score of this kind mean something: a positive ligand region
should meet a negative protein region, hydrophobic meets hydrophobic, a donor meets an acceptor.
This module turns that into a question about one growth vector:

* **the gap** is what the pocket would still reward, on a grid, after the fragment's own field is
  taken off -- the field a group would have to place to be worth adding;
* **the candidates** are groups built from their SMILES and placed on the vector by rule, with
  the torsion about the new bond scanned, so that nothing about the answer enters the geometry;
* **the score** is the overlap of the grown molecule's added field with the gap, and a placement
  that clashes with the protein is refused.

The gate of plan item MC7 is whether the group that was actually added comes out among the first
ten of such an enumeration.  The plan says this is the item most likely to return nothing,
because a one-body field score has never been shown to rank linkers;
`benchmarks/run_fraggrow.py` is the measurement, and rule 5 then decides what is kept.

Nothing here needs a model: the pocket and the fragment are enough, which is what makes it usable
on a target with no series yet -- the situation a fragment campaign is in.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .complementarity import COMPLEMENTARY_PAIRS
from .fields import FieldSet, MoleculeFields, build_fields
from .localalign import frame_from

#: torsions tried about the new bond, in turns of a full circle
TORSIONS = 12
#: a heavy atom of the ligand may come this close to a protein heavy atom and no closer; the
#: value is a contact rather than a clash, since every bound group touches the protein
CONTACT = 3.0
#: over this distance the room left by the pocket goes from none to all
RAMP = 0.6
#: a placement with any atom closer than this to the protein is refused outright
CLASH = 2.5
#: grid spacing for the gap, in angstroms
SPACING = 0.6
#: how far around the growth vector the gap is looked at, in angstroms
REACH = 7.0


# --------------------------------------------------------------------------- #
# Where a fragment can grow
# --------------------------------------------------------------------------- #


@dataclass
class Vector:
    """A place on a fragment where a group can be added, and the frame that places it."""
    atom: int
    origin: np.ndarray
    rotation: np.ndarray
    hydrogen: Optional[int] = None

    def to_pocket(self, local: np.ndarray, turn: float = 0.0) -> np.ndarray:
        """Local coordinates of a group, turned about the new bond, in the pocket's frame."""
        c, s = np.cos(turn), np.sin(turn)
        spin = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        return np.asarray(self.origin, float) + (spin @ np.asarray(local, float).T).T @ np.asarray(
            self.rotation, float)


def vacant_direction(coords: np.ndarray, atom: int, neighbours: Sequence[int]) -> np.ndarray:
    """Where the next bond of `atom` would point, from the bonds it already has."""
    r = np.asarray(coords, float)
    if not len(neighbours):
        return np.array([1.0, 0.0, 0.0])
    d = np.zeros(3)
    for n in neighbours:
        v = r[int(n)] - r[int(atom)]
        norm = float(np.linalg.norm(v))
        if norm > 1e-9:
            d -= v / norm
    norm = float(np.linalg.norm(d))
    return d / norm if norm > 1e-9 else np.array([1.0, 0.0, 0.0])


def growth_vector(mol, atom: int) -> Optional[Vector]:
    """The frame of the bond that would be made at `atom`, with the hydrogen it replaces.

    The direction is the one the atom's own bonds leave empty, so it is fixed by the fragment
    alone and nothing about what will be added enters it.  The plane is fixed by the atom's
    first heavy neighbour, which makes the frame reproducible.
    """
    coords = np.asarray(mol.GetConformer().GetPositions(), float)
    heavy = [n.GetIdx() for n in mol.GetAtomWithIdx(int(atom)).GetNeighbors()
             if n.GetAtomicNum() > 1]
    hyd = [n.GetIdx() for n in mol.GetAtomWithIdx(int(atom)).GetNeighbors()
           if n.GetAtomicNum() == 1]
    if not heavy:
        return None
    x = vacant_direction(coords, atom, heavy)
    # a virtual point along the vacant direction plays the part of the atom to come
    extended = np.vstack([coords, coords[int(atom)] + x])
    R = frame_from(extended, len(coords), int(heavy[0]), centre=int(atom))
    if R is None:
        return None
    which = None
    if hyd:
        cos = [float((coords[h] - coords[int(atom)]) @ x
                     / max(np.linalg.norm(coords[h] - coords[int(atom)]), 1e-9)) for h in hyd]
        which = int(hyd[int(np.argmax(cos))])
    return Vector(atom=int(atom), origin=coords[int(atom)].copy(), rotation=R, hydrogen=which)


@dataclass
class Case:
    """A fragment, the compound it was grown into, and what was added where."""
    fragment: int
    grown: int
    smiles: str
    atom: int
    added: List[int]
    names: Tuple[str, str] = ("", "")

    def __str__(self) -> str:
        return (f"{self.names[0]} -> {self.names[1]}: adds {self.smiles} "
                f"({len(self.added)} heavy atoms)")


def addition(fragment, grown, max_added: int = 20) -> Optional[Tuple[str, int, List[int]]]:
    """(the added group as SMILES, the fragment atom it hangs on, the added atoms of `grown`).

    Both molecules are taken without hydrogens, and the fragment must be a substructure of the
    grown compound, which is what makes this a growing case rather than a pair of analogues.
    The group is written by the same single-bond cut as the matched pairs of
    :mod:`cmfqsar.bioisostere`, so the two items speak of groups in the same words.
    """
    from .bioisostere import fragment as cut

    match = grown.GetSubstructMatch(fragment)
    if not match:
        return None
    added = sorted(set(range(grown.GetNumAtoms())) - set(match))
    if not added or len(added) > int(max_added):
        return None
    for piece in cut(grown, max_substituent=int(max_added), min_core=0.0):
        if set(piece.atoms.tolist()) == set(added):
            for b in grown.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                if i in added and j in match:
                    return piece.group, list(match).index(j), added
                if j in added and i in match:
                    return piece.group, list(match).index(i), added
    return None


# --------------------------------------------------------------------------- #
# A candidate group, built from nothing but its SMILES
# --------------------------------------------------------------------------- #


@dataclass
class Group:
    """A candidate group in its own frame: the attachment point at the origin, the bond on +x."""
    smiles: str
    mol: object
    local: np.ndarray
    first: int

    @property
    def heavy(self) -> int:
        return int(sum(1 for a in self.mol.GetAtoms() if a.GetAtomicNum() > 1))


def build_group(smiles: str, seed: int = 1) -> Optional[Group]:
    """A group written as SMILES with one attachment point, embedded and put in its own frame.

    The geometry comes from the SMILES and a force field, never from a crystal structure, so a
    candidate that happens to be the right answer is built exactly like every other one.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    dummies = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    m = Chem.AddHs(m)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) if int(seed) else 1
    if AllChem.EmbedMolecule(m, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    except Exception:
        pass
    d = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() == 0][0]
    nb = [n.GetIdx() for n in m.GetAtomWithIdx(d).GetNeighbors()]
    if len(nb) != 1:
        return None
    coords = np.asarray(m.GetConformer().GetPositions(), float)
    third = [n.GetIdx() for n in m.GetAtomWithIdx(nb[0]).GetNeighbors() if n.GetIdx() != d]
    extended = np.vstack([coords, coords[d] + _perpendicular(coords[nb[0]] - coords[d])])
    R = frame_from(coords, nb[0], third[0], centre=d) if third else \
        frame_from(extended, nb[0], len(coords), centre=d)
    if R is None:
        return None
    local = (coords - coords[d]) @ np.asarray(R, float).T
    # drop the dummy; the first real atom keeps the bond to the fragment
    keep = [i for i in range(m.GetNumAtoms()) if i != d]
    ed = Chem.RWMol(m)
    ed.RemoveAtom(int(d))
    out = ed.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return Group(smiles=smiles, mol=out, local=local[keep],
                 first=int(keep.index(nb[0])))


def _perpendicular(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    other = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    w = np.cross(v, other)
    n = float(np.linalg.norm(w))
    return w / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def grow(fragment, vector: Vector, group: Group, turn: float = 0.0):
    """The fragment with the group bonded on, as one molecule with coordinates."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    frag = Chem.RWMol(fragment)
    if vector.hydrogen is not None:
        frag.RemoveAtom(int(vector.hydrogen))
    atom = int(vector.atom) - (1 if vector.hydrogen is not None
                               and vector.hydrogen < vector.atom else 0)
    base = frag.GetNumAtoms()
    combo = Chem.RWMol(Chem.CombineMols(frag.GetMol(), group.mol))
    combo.AddBond(atom, base + int(group.first), Chem.BondType.SINGLE)
    out = combo.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    conf = out.GetConformer()
    # the dummy atom of a SMILES is embedded at about 1.1 A, which is not a bond length between
    # two heavy atoms: the group is slid along the new bond until its first atom sits at the sum
    # of the two covalent radii
    table = Chem.GetPeriodicTable()
    want = (table.GetRcovalent(fragment.GetAtomWithIdx(int(vector.atom)).GetAtomicNum())
            + table.GetRcovalent(group.mol.GetAtomWithIdx(int(group.first)).GetAtomicNum()))
    local = np.asarray(group.local, float).copy()
    local[:, 0] += want - float(local[int(group.first), 0])
    placed = vector.to_pocket(local, turn)
    for k in range(group.mol.GetNumAtoms()):
        x, y, z = placed[k]
        conf.SetAtomPosition(base + k, Chem.rdGeometry.Point3D(float(x), float(y), float(z)))
    return out


# --------------------------------------------------------------------------- #
# The gap the pocket leaves
# --------------------------------------------------------------------------- #


@dataclass
class Gap:
    """What a group would still have to place, on a grid around one growth vector."""
    points: np.ndarray
    reward: Dict[str, np.ndarray]
    room: np.ndarray
    cell: float

    def __len__(self) -> int:
        return len(self.points)


def gap_points(vector: Vector, reach: float = REACH, spacing: float = SPACING) -> np.ndarray:
    """A ball of grid points in front of the growth vector, where a group could go."""
    n = int(np.ceil(reach / spacing))
    ax = spacing * np.arange(-n, n + 1)
    g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
    g = g[(np.linalg.norm(g, axis=1) <= reach) & (g[:, 0] > -1.0)]   # forward of the atom only
    return vector.to_pocket(g)


def gap(pocket: MoleculeFields, fragment: MoleculeFields, fieldset: FieldSet,
        vector: Vector, reach: float = REACH, spacing: float = SPACING) -> Gap:
    """What a group would be rewarded for placing, in the space the fragment leaves free.

    One array per ligand field type, with the sign convention of
    :data:`cmfqsar.complementarity.COMPLEMENTARY_PAIRS`: the pocket's electrostatic field counts
    with a minus, its hydrophobic field with a plus, and its donors pair with the ligand's
    acceptors.  ``room`` is the space left by the protein **and** by the fragment, which is how
    "minus what the fragment already places" is taken here: an offer is open only where a new
    atom could sit.  Shape needs no field of its own, because that is what the room is.
    """
    from .complementarity import _gauss_field_at
    names = [s.name for s in fieldset.specs]
    pts = gap_points(vector, reach, spacing)
    reward: Dict[str, np.ndarray] = {}
    for lig, poc, sign, _label in COMPLEMENTARY_PAIRS:
        if lig not in names or poc not in names:
            continue
        fp = names.index(poc)
        reward[lig] = sign * _gauss_field_at(pts, pocket, fieldset.specs[fp], fp)
    # what the fragment already places is taken off as **space**, not as field: an offer is open
    # only where there is somewhere for a new atom to sit. Subtracting the fragment's own field
    # from the pocket's offer, which is the other reading of the item, compares two different
    # quantities and erases exactly the contact region a new group would use.
    left = room(pts, pocket) * room(pts, fragment)
    # the points where the pocket leaves no room contribute nothing to any score, so they are
    # dropped here rather than multiplied by zero once per candidate
    keep = left > 0
    return Gap(points=pts[keep], reward={k: v[keep] for k, v in reward.items()},
               room=left[keep], cell=float(spacing) ** 3)


def room(points: np.ndarray, pocket: MoleculeFields, contact: float = CONTACT,
         ramp: float = RAMP) -> np.ndarray:
    """How much of a ligand atom the pocket allows at each point, from 0 to 1.

    This is not :func:`cmfqsar.complementarity.free_space`, which asks where a ligand could sit
    comfortably and is right for comparing a co-field with a pocket.  A group that is added to a
    bound fragment **touches** the protein -- that contact is the interaction -- so the mask here
    has to be open at the contact distance and closed only where an atom would not fit.
    """
    from scipy.spatial import cKDTree
    pts = np.asarray(points, float).reshape(-1, 3)
    heavy = [i for i, sym in enumerate(pocket.symbols) if sym != "H"]
    if not heavy or len(pts) == 0:
        return np.ones(len(pts))
    d = cKDTree(np.asarray(pocket.coords, float)[heavy]).query(pts, k=1)[0]
    return np.clip((d - (float(contact) - float(ramp))) / max(float(ramp), 1e-6), 0.0, 1.0)


def score(added: MoleculeFields, fieldset: FieldSet, g: Gap,
          weights: Optional[Dict[str, float]] = None) -> float:
    """How much of the gap one added group fills, as one number.

    The integral is taken only where the pocket leaves room, so a group cannot be rewarded for
    placing field inside the protein.
    """
    from .complementarity import _gauss_field_at
    names = [s.name for s in fieldset.specs]
    room = np.maximum(g.room, 0.0)
    total = 0.0
    for lig, v in g.reward.items():
        f = names.index(lig)
        x = _gauss_field_at(g.points, added, fieldset.specs[f], f)
        w = 1.0 if weights is None else float(weights.get(lig, 1.0))
        total += w * float(np.sum(room * v * x)) * g.cell
    return total


def clashes(mol, pocket: MoleculeFields, atoms: Optional[Sequence[int]] = None,
            contact: float = CLASH) -> int:
    """How many heavy atoms of the molecule sit closer than `contact` to a protein heavy atom."""
    from scipy.spatial import cKDTree
    coords = np.asarray(mol.GetConformer().GetPositions(), float)
    heavy = [i for i in (range(mol.GetNumAtoms()) if atoms is None else atoms)
             if mol.GetAtomWithIdx(int(i)).GetAtomicNum() > 1]
    if not heavy:
        return 0
    prot = [i for i, s in enumerate(pocket.symbols) if s != "H"]
    if not prot:
        return 0
    d = cKDTree(np.asarray(pocket.coords, float)[prot]).query(coords[heavy], k=1)[0]
    return int(np.sum(d < float(contact)))


# --------------------------------------------------------------------------- #
# The enumeration
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    """One group placed on one vector, with the best turn found for it."""
    smiles: str
    score: float
    turn: float
    clashes: int
    heavy: int
    placed: object = None

    def __str__(self) -> str:
        return f"{self.smiles}: {self.score:+.2f} ({self.heavy} atoms, {self.clashes} clashes)"


def try_group(fragment, vector: Vector, group: Group, pocket: MoleculeFields,
              fieldset: FieldSet, g: Gap, turns: int = TORSIONS,
              keep_placed: bool = False) -> Optional[Candidate]:
    """Place one group on the vector, turn it, and keep the best placement that does not clash.

    The molecule is assembled and its fields are built **once**: an atom's type, weight and
    radius come from the bond graph, which the turn about the new bond does not change, so every
    further turn is a rotation of the added atoms' coordinates and nothing else.
    """
    base = fragment.GetNumAtoms() - (1 if vector.hydrogen is not None else 0)
    grown = grow(fragment, vector, group, 0.0)
    if grown is None:
        return None
    added = list(range(base, grown.GetNumAtoms()))
    part = _only(build_fields(grown, fieldset, name=group.smiles), added, group.smiles)
    heavy = np.array([i for i, s in enumerate(part.symbols) if s != "H"], int)
    origin = np.asarray(vector.origin, float)
    R = np.asarray(vector.rotation, float)
    local = (np.asarray(part.coords, float) - origin) @ R.T
    tree = _pocket_tree(pocket)
    best = None
    for k in range(int(turns)):
        turn = 2.0 * np.pi * k / float(turns)
        c, s = np.cos(turn), np.sin(turn)
        spin = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
        part.coords = origin + (spin @ local.T).T @ R
        n_clash = 0
        if tree is not None and len(heavy):
            n_clash = int(np.sum(tree.query(part.coords[heavy], k=1)[0] < CLASH))
        if n_clash and best is not None and best.clashes == 0:
            continue
        cand = Candidate(smiles=group.smiles, score=score(part, fieldset, g), turn=turn,
                         clashes=n_clash, heavy=group.heavy)
        if best is None or (cand.clashes, -cand.score) < (best.clashes, -best.score):
            best = cand
    if best is not None and keep_placed:
        best.placed = grow(fragment, vector, group, best.turn)
    return best


def _pocket_tree(pocket: MoleculeFields):
    """A neighbour tree over the pocket's heavy atoms, cached on the pocket itself."""
    from scipy.spatial import cKDTree
    tree = getattr(pocket, "_fraggrow_tree", None)
    if tree is None:
        heavy = [i for i, s in enumerate(pocket.symbols) if s != "H"]
        tree = cKDTree(np.asarray(pocket.coords, float)[heavy]) if heavy else None
        try:
            object.__setattr__(pocket, "_fraggrow_tree", tree)
        except Exception:
            pass
    return tree


def _only(fields: MoleculeFields, atoms: Sequence[int], name: str) -> MoleculeFields:
    """The fields of some atoms of a molecule, as a field object of its own."""
    from .bioisostere import masked_field
    return masked_field(fields, atoms, name=name)


def enumerate_groups(fragment, vector: Vector, pocket: MoleculeFields, fieldset: FieldSet,
                     groups: Sequence[Group], turns: int = TORSIONS, reach: float = REACH,
                     spacing: float = SPACING, progress=None) -> List[Candidate]:
    """Every group of the library placed on one vector, best first."""
    g = gap(pocket, build_fields(fragment, fieldset, name="fragment"), fieldset, vector,
            reach=reach, spacing=spacing)
    out: List[Candidate] = []
    for i, group in enumerate(groups):
        c = try_group(fragment, vector, group, pocket, fieldset, g, turns=turns)
        if c is not None:
            out.append(c)
        if progress is not None:
            progress(f"Groups {i + 1}/{len(groups)}", (i + 1) / max(len(groups), 1))
    out.sort(key=lambda c: (c.clashes, -c.score))
    return out


def rank_of(candidates: Sequence[Candidate], smiles: str) -> Optional[int]:
    """Where one group came in the enumeration, counting from one."""
    for i, c in enumerate(candidates):
        if c.smiles == smiles:
            return i + 1
    return None
