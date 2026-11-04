"""
A cation with its first solvation shell as one molecule, built, optimised and framed.

Plan item BT4.  For every property of an electrolyte -- the energy a lithium ion gains from a
solvent, what it costs to strip one molecule off again, how many molecules fit around the ion --
the object that carries the property is not the solvent molecule.  It is the complex: the ion with
the molecules that touch it.  Nothing in this program built one, and this module does.

Three steps, and each is separate so that each can be checked on its own.

**Placing.**  A donor atom is an oxygen or a nitrogen with a lone pair pointing somewhere, and
:func:`donor_direction` says where: away from the bisector of an ether oxygen's two bonds, along
the C=O axis of a carbonyl, along the lone pair of a nitrile.  :func:`place_shell` puts the ion at
the origin, chooses as many directions as there are molecules from the polyhedron of that
coordination number (linear, trigonal, tetrahedral), and for each molecule rotates it so that its
donor's lone pair points back at the ion and translates it so the donor sits at the coordination
distance.  Each molecule is then spun about its own ion-donor axis to the angle that leaves the
most room, which is a starting geometry and not a result: the optimiser decides the rest.

**Optimising.**  GFN2-xTB through :mod:`cmfqsar.xtb`, which caches per molecule, per charge, per
method and per starting geometry, so an interrupted run resumes.  ORCA through
:mod:`cmfqsar.orca` is the other engine and is far slower on a complex of fifty atoms.  The
energies this module returns are total energies in eV; a binding energy is a difference of three
of them and a desolvation energy a difference of three others, both written out in the functions
that compute them, because a sign error in either is invisible in the number.

**Framing.**  This is what the complex gives that the free solvent cannot.  A series of free
solvent molecules has no common frame -- a glyme has no scaffold to superimpose -- but a series of
complexes does: the ion is one atom in every one of them, and the polyhedron of its donors fixes
the orientation.  :func:`cation_frame` puts the origin on the ion, the axes along the principal
axes of the donor set, and resolves the four proper sign choices by the third moments of the
donors.  With that frame the overlap kernel applies, and with the overlap kernel there is a
co-field: a map in space of where binding is gained, which for a solvation complex is a statement
about the donor oxygens.

**What the module does not do.**  It does not search conformations.  A glyme wraps a lithium ion
with four or five of its own oxygens, and :func:`wrap_ion` offers the intramolecular case -- the
ion placed at the centroid of the donors that already lie closest together, then optimised -- but
the fold it finds is the fold the starting conformer was near. ``benchmarks/run_solvation.py``
measures how far that gets against 1,050 complexes relaxed with density functional theory, which
is the honest way to state the limit of a builder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

#: 1 Hartree in eV
HARTREE_EV = 27.211386245988

#: the working ions, with the coordination distance to an oxygen in angstrom.  The lithium value
#: is the one measured on the 1,050 relaxed Li(+)-ether complexes of ``examples/ether/``
#: (2.007 +/- 0.113 A); the others add Shannon's six-coordinate radius difference, as the ion
#: probes of :mod:`cmfqsar.probes` do.
ION_OXYGEN_DISTANCE = {"Li": 2.01, "Na": 2.27, "K": 2.63, "Mg": 1.97}
#: the charge of each ion
ION_CHARGE = {"Li": 1, "Na": 1, "K": 1, "Mg": 2}

#: how a donor atom is recognised, in the order tried.  Each pattern's first atom is the donor.
DONOR_SMARTS: Tuple[Tuple[str, str], ...] = (
    ("carbonyl", "[OX1]=[#6]"),
    ("ether", "[OX2]([#6])[#6]"),
    ("hydroxyl", "[OX2H][#6]"),
    ("sulfonyl", "[OX1]=[#16]"),
    ("phosphoryl", "[OX1]=[#15]"),
    ("nitrile", "[NX1]#[#6]"),
    ("amine", "[NX3;!$(N=*);!$(N#*)]"),
    ("fluoride", "[F-]"),
    ("fluorine on phosphorus", "[F][#15]"),
)

#: the directions of the donors for each coordination number, as unit vectors.  Two donors are
#: placed at 180 degrees, three trigonally, four tetrahedrally, five and six as the bipyramid and
#: the octahedron.  A lithium ion in a carbonate or an ether is four- or five-coordinate.
_T = 1.0 / math.sqrt(3.0)
POLYHEDRA: Dict[int, np.ndarray] = {
    1: np.array([[0.0, 0.0, 1.0]]),
    2: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
    3: np.array([[1.0, 0.0, 0.0], [-0.5, math.sqrt(3) / 2, 0.0], [-0.5, -math.sqrt(3) / 2, 0.0]]),
    4: np.array([[_T, _T, _T], [_T, -_T, -_T], [-_T, _T, -_T], [-_T, -_T, _T]]),
    5: np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0],
                 [-0.5, math.sqrt(3) / 2, 0.0], [-0.5, -math.sqrt(3) / 2, 0.0]]),
    6: np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
}

#: the four proper sign choices of a frame: the product of the three signs is +1, so each is a
#: rotation and not a reflection
FLIPS: Tuple[Tuple[float, float, float], ...] = ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0),
                                                 (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0))


class SolvationError(ValueError):
    """A complex that cannot be built or framed, with the reason in the message."""


# --------------------------------------------------------------------------------- the donors


def donor_atoms(mol: Chem.Mol) -> List[Tuple[int, str]]:
    """``[(atom index, kind)]`` of every atom that can point a lone pair at a cation.

    An atom matched by more than one pattern keeps the first kind it matched, and the list is in
    the order of the atom indices so that it does not depend on the order of the patterns.
    """
    found: Dict[int, str] = {}
    for kind, smarts in DONOR_SMARTS:
        query = Chem.MolFromSmarts(smarts)
        if query is None:                                   # pragma: no cover - a fixed table
            raise SolvationError(f"the pattern for {kind} does not parse")
        for match in mol.GetSubstructMatches(query):
            found.setdefault(int(match[0]), kind)
    return [(i, found[i]) for i in sorted(found)]


def donor_direction(mol: Chem.Mol, index: int, coords: Optional[np.ndarray] = None) -> np.ndarray:
    """The unit vector along which a cation should approach the donor `index`.

    It points away from the atoms the donor is bonded to: the reverse of the sum of the bond
    directions, which is the bisector for two bonds and the bond axis itself for one.  A donor
    with no bonds at all -- a bare fluoride -- has no direction, and any direction will do, so
    ``+z`` is returned.
    """
    pos = np.asarray(mol.GetConformer().GetPositions() if coords is None else coords, float)
    atom = mol.GetAtomWithIdx(int(index))
    here = pos[int(index)]
    out = np.zeros(3)
    for nb in atom.GetNeighbors():
        v = here - pos[nb.GetIdx()]
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            out += v / n
    n = float(np.linalg.norm(out))
    if n < 1e-6:
        # two bonds at 180 degrees, or none: the lone pair has no unique direction. Take any
        # direction perpendicular to the bonds, so that the ion does not sit on an atom.
        for nb in atom.GetNeighbors():
            v = here - pos[nb.GetIdx()]
            if float(np.linalg.norm(v)) > 1e-9:
                perp = np.cross(v, [0.0, 0.0, 1.0])
                if float(np.linalg.norm(perp)) < 1e-6:
                    perp = np.cross(v, [0.0, 1.0, 0.0])
                return perp / float(np.linalg.norm(perp))
        return np.array([0.0, 0.0, 1.0])
    return out / n


def best_donor(mol: Chem.Mol, charges: Optional[Sequence[float]] = None) -> int:
    """The donor a cation would choose: the most negative one, or the first carbonyl or ether.

    `charges` are partial charges, one per atom, when the caller has them (from xTB, say).
    Without them the order of :data:`DONOR_SMARTS` decides, which puts a carbonyl oxygen before
    an ether oxygen, as their basicity does.
    """
    donors = donor_atoms(mol)
    if not donors:
        raise SolvationError(f"{Chem.MolToSmiles(Chem.RemoveHs(mol))} has no donor atom that "
                             f"this module recognises")
    if charges is not None:
        q = np.asarray(charges, float)
        return int(min(donors, key=lambda d: q[d[0]])[0])
    order = [kind for kind, _s in DONOR_SMARTS]
    return int(min(donors, key=lambda d: (order.index(d[1]), d[0]))[0])


# --------------------------------------------------------------------------------- the placing


def _rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """A rotation matrix taking the unit vector `a` onto the unit vector `b`."""
    a = np.asarray(a, float) / float(np.linalg.norm(a))
    b = np.asarray(b, float) / float(np.linalg.norm(b))
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    c = float(np.dot(a, b))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel: a half turn about any axis perpendicular to a
        perp = np.cross(a, [0.0, 0.0, 1.0])
        if float(np.linalg.norm(perp)) < 1e-9:
            perp = np.cross(a, [0.0, 1.0, 0.0])
        perp /= float(np.linalg.norm(perp))
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def _rotation_about(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, float) / float(np.linalg.norm(axis))
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def _embed(mol: Chem.Mol, seed: int = 0) -> Chem.Mol:
    """A molecule with hydrogens and one optimised conformer."""
    m = Chem.AddHs(Chem.Mol(mol)) if any(a.GetTotalNumHs() for a in mol.GetAtoms()) else Chem.Mol(mol)
    if m.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = int(seed) + 1
        if AllChem.EmbedMolecule(m, params) != 0:
            raise SolvationError(f"no conformer could be generated for "
                                 f"{Chem.MolToSmiles(Chem.RemoveHs(m))}")
        AllChem.MMFFOptimizeMolecule(m)
    return m


@dataclass
class Complex:
    """A cation with its shell: the molecule, the ion's atom index and what went into it."""
    mol: Chem.Mol
    ion_index: int
    ion: str
    solvents: Tuple[str, ...] = ()
    donors: Tuple[int, ...] = ()
    charge: int = 0
    energy_ev: float = float("nan")          # total energy, eV; NaN until optimised
    optimised: bool = False
    note: str = ""

    @property
    def n_solvents(self) -> int:
        return len(self.solvents)

    def coordination(self, cutoff: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        return coordination(self.mol, self.ion_index, self.ion, cutoff)


def place_shell(solvents: Sequence[Chem.Mol], ion: str = "Li", distance: Optional[float] = None,
                seed: int = 0, spins: int = 12, anion: Optional[Chem.Mol] = None) -> Complex:
    """Put `solvents` (and an `anion`, for a contact ion pair) around one `ion`.

    The ion goes at the origin, the molecules on the directions of the polyhedron for their
    number, each turned so that its donor's lone pair points at the ion and each spun about the
    ion-donor axis to the angle that leaves the largest smallest interatomic distance.  The result
    is a starting geometry for an optimiser.
    """
    if ion not in ION_OXYGEN_DISTANCE:
        raise SolvationError(f"unknown ion {ion!r}; this module knows "
                             f"{', '.join(sorted(ION_OXYGEN_DISTANCE))}")
    members = list(solvents) + ([anion] if anion is not None else [])
    if not members:
        raise SolvationError("a shell needs at least one molecule")
    n = len(members)
    if n not in POLYHEDRA:
        raise SolvationError(f"{n} molecules in a shell: this module places "
                             f"{', '.join(str(k) for k in sorted(POLYHEDRA))}")
    d = float(ION_OXYGEN_DISTANCE[ion] if distance is None else distance)
    directions = POLYHEDRA[n]

    ion_atom = Chem.Atom(ion)
    ion_atom.SetFormalCharge(int(ION_CHARGE[ion]))
    ion_atom.SetNoImplicit(True)
    out = Chem.RWMol()
    out.AddAtom(ion_atom)
    placed = [np.zeros(3)]                        # the ion
    names, donor_indices = [], []

    for k, member in enumerate(members):
        m = _embed(member, seed=seed + k)
        pos = np.asarray(m.GetConformer().GetPositions(), float)
        idx = best_donor(m)
        lone = donor_direction(m, idx, pos)
        target = directions[k] * d
        # the lone pair must point from the donor back at the ion, that is along -direction
        rot = _rotation_between(lone, -directions[k])
        pos = (pos - pos[idx]) @ rot.T
        best = None
        for s in range(max(1, int(spins))):
            spin = _rotation_about(directions[k], 2.0 * math.pi * s / max(1, int(spins)))
            trial = pos @ spin.T + target
            if len(placed) == 1:
                score = float("inf")
            else:
                other = np.vstack(placed[1:]) if len(placed) > 2 else placed[1]
                other = np.atleast_2d(other)
                score = float(np.min(np.linalg.norm(trial[:, None, :] - other[None, :, :],
                                                    axis=-1)))
            if best is None or score > best[0]:
                best = (score, trial)
        coords = best[1]
        offset = out.GetNumAtoms()
        out.InsertMol(m)
        placed.append(coords)
        names.append(Chem.MolToSmiles(Chem.RemoveHs(m)))
        donor_indices.append(offset + int(idx))

    mol = out.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    at = 1
    for coords in placed[1:]:
        for xyz in coords:
            conf.SetAtomPosition(at, Point3D(*[float(v) for v in xyz]))
            at += 1
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    return Complex(mol=mol, ion_index=0, ion=ion, solvents=tuple(names),
                   donors=tuple(donor_indices), charge=int(Chem.GetFormalCharge(mol)),
                   note=f"placed on the {n}-fold polyhedron at {d:.2f} A")


def wrap_ion(solvent: Chem.Mol, ion: str = "Li", n_donors: int = 0, seed: int = 0) -> Complex:
    """Put the ion inside one molecule: the intramolecular case of a glyme or a crown ether.

    The ion is placed at the centroid of the `n_donors` donor atoms that lie closest together in
    the conformer given, or of every donor when `n_donors` is 0, and then pushed out to the
    coordination distance from the nearest of them if it landed too close.  No conformational
    search is done: the fold this gives is the fold of the conformer it was handed.
    """
    m = _embed(solvent, seed=seed)
    pos = np.asarray(m.GetConformer().GetPositions(), float)
    donors = [i for i, _kind in donor_atoms(m)]
    if not donors:
        raise SolvationError(f"{Chem.MolToSmiles(Chem.RemoveHs(m))} has no donor atom")
    want = len(donors) if not n_donors else min(int(n_donors), len(donors))
    if want < len(donors):
        # the tightest group of `want` donors: the one with the smallest spread about its centroid
        best = None
        from itertools import combinations
        for group in combinations(donors, want):
            p = pos[list(group)]
            spread = float(np.mean(np.linalg.norm(p - p.mean(axis=0), axis=1)))
            if best is None or spread < best[0]:
                best = (spread, list(group))
        chosen = best[1]
    else:
        chosen = list(donors)
    centre = pos[chosen].mean(axis=0)
    d = float(ION_OXYGEN_DISTANCE[ion])
    nearest = float(np.min(np.linalg.norm(pos[chosen] - centre, axis=1)))
    if nearest < 0.5 * d:
        # the centroid sits on top of a donor: step away along the centroid's own normal
        p = pos[chosen] - centre
        _u, _s, vt = np.linalg.svd(p - p.mean(axis=0), full_matrices=True)
        centre = centre + vt[-1] * (0.5 * d)

    ion_atom = Chem.Atom(ion)
    ion_atom.SetFormalCharge(int(ION_CHARGE[ion]))
    ion_atom.SetNoImplicit(True)
    out = Chem.RWMol()
    out.AddAtom(ion_atom)
    out.InsertMol(m)
    mol = out.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, Point3D(*[float(v) for v in centre]))
    for i, xyz in enumerate(pos):
        conf.SetAtomPosition(i + 1, Point3D(*[float(v) for v in xyz]))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    return Complex(mol=mol, ion_index=0, ion=ion,
                   solvents=(Chem.MolToSmiles(Chem.RemoveHs(m)),),
                   donors=tuple(i + 1 for i in chosen),
                   charge=int(Chem.GetFormalCharge(mol)),
                   note=f"wrapped on {len(chosen)} donors of one molecule")


# ------------------------------------------------------------------------------- the geometry


def ion_index(mol: Chem.Mol, ion: str = "Li") -> int:
    """Where the cation is. Raises when there is not exactly one."""
    hits = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == ion]
    if len(hits) != 1:
        raise SolvationError(f"the complex holds {len(hits)} {ion} atoms, not one")
    return int(hits[0])


def coordination(mol: Chem.Mol, index: int, ion: str = "Li",
                 cutoff: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """``(atom indices, distances)`` of the donor atoms inside `cutoff` of the ion.

    The default cutoff is 1.3 times the coordination distance of the ion, which for lithium and
    oxygen is 2.6 A -- the value the first shell of the relaxed complexes of ``examples/ether/``
    is counted with.
    """
    pos = np.asarray(mol.GetConformer().GetPositions(), float)
    r = float(1.3 * ION_OXYGEN_DISTANCE.get(ion, 2.0) if cutoff is None else cutoff)
    donors = np.array([i for i, _k in donor_atoms(mol) if i != int(index)], dtype=int)
    if len(donors) == 0:
        return np.zeros(0, int), np.zeros(0)
    d = np.linalg.norm(pos[donors] - pos[int(index)], axis=1)
    keep = d <= r
    order = np.argsort(d[keep])
    return donors[keep][order], d[keep][order]


def _inertia_about_origin(points: np.ndarray) -> np.ndarray:
    """The inertia tensor of `points` about the origin, every point counted once."""
    p = np.asarray(points, float)
    return np.eye(3) * (p ** 2).sum() - p.T @ p


def _axes_with_tiebreak(primary: np.ndarray, tie: Optional[np.ndarray] = None,
                        tol: float = 1e-3) -> np.ndarray:
    """Principal axes of `primary`, with equal moments resolved by the tensor `tie`.

    Two equal moments leave every rotation inside their plane just as good, which happens as soon
    as the coordination polyhedron is regular: three donors at 120 degrees, four at the corners of
    a tetrahedron.  The axes inside such a group are then taken from the second tensor -- the
    atoms of the whole complex rather than the donors alone -- which is a weaker description of
    the shape and therefore the right thing to break a tie with and the wrong thing to lead.
    When the second tensor is degenerate there too, the complex really has that symmetry, and
    every choice gives the same fields.
    """
    moments, vecs = np.linalg.eigh(np.asarray(primary, float))
    axes = vecs.T.copy()
    if tie is not None:
        scale = max(float(np.abs(moments).max()), 1e-12)
        start = 0
        while start < 3:
            stop = start + 1
            while stop < 3 and (moments[stop] - moments[start]) <= tol * scale:
                stop += 1
            if stop - start > 1:
                block = axes[start:stop]
                sub = block @ np.asarray(tie, float) @ block.T
                _m, v = np.linalg.eigh(sub)
                axes[start:stop] = v.T @ block
            start = stop
    axes[2] = np.cross(axes[0], axes[1])                     # right-handed
    return axes


def cation_frame(mol: Chem.Mol, index: Optional[int] = None, ion: str = "Li",
                 cutoff: Optional[float] = None) -> np.ndarray:
    """The coordinates of the complex in the frame its own coordination polyhedron defines.

    The origin is the ion.  The axes are the principal axes of the donor atoms inside the cutoff,
    ordered by their moment, smallest first, which makes the longest extent of the polyhedron the
    x axis.  Four sign choices leave those axes and the handedness alone, and the one with the
    largest third moments of the donors, taken in the order x, y, z, is chosen; a polyhedron
    symmetric enough for all three to vanish is one whose fields are the same in either frame.
    A single donor fixes only one direction: the x axis is put along it and the other two are any
    perpendicular pair, which is stated in the return of :func:`frame_quality`.
    """
    idx = ion_index(mol, ion) if index is None else int(index)
    pos = np.asarray(mol.GetConformer().GetPositions(), float) - np.asarray(
        mol.GetConformer().GetPositions(), float)[idx]
    donors, _d = coordination(mol, idx, ion, cutoff)
    if len(donors) == 0:
        raise SolvationError("the ion has no donor atom within the cutoff, so the polyhedron "
                             "cannot fix a frame")
    p = pos[donors]
    if len(donors) == 1:
        x = p[0] / float(np.linalg.norm(p[0]))
        tmp = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(tmp, x))) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        y = tmp - float(np.dot(tmp, x)) * x
        y /= float(np.linalg.norm(y))
        axes = np.vstack([x, y, np.cross(x, y)])
        return pos @ axes.T
    # the inertia tensor of the donors about the ion leads; the heavy atoms of the whole complex
    # break a tie between two equal moments, which a regular polyhedron always has
    heavy = np.array([a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1], dtype=int)
    axes = _axes_with_tiebreak(_inertia_about_origin(p), _inertia_about_origin(pos[heavy]))
    out = pos @ axes.T
    # the four proper sign choices, decided by the third moments of the donors in the order
    # x, y, z, and by those of the heavy atoms where the donors leave them tied
    alive = list(range(len(FLIPS)))
    for q in (out[donors], out[heavy]):
        keys = np.array([[(s[0] * q[:, 0]) ** 3, (s[1] * q[:, 1]) ** 3,
                          (s[2] * q[:, 2]) ** 3] for s in FLIPS]).sum(axis=2)
        scale = float(np.abs(q).max() ** 3) + 1e-12
        for k in range(3):
            best = max(keys[i, k] for i in alive)
            alive = [i for i in alive if keys[i, k] >= best - 1e-6 * scale]
            if len(alive) == 1:
                break
        if len(alive) == 1:
            break
    return out * np.asarray(FLIPS[alive[0]], float)


def frame_quality(mol: Chem.Mol, index: Optional[int] = None, ion: str = "Li",
                  cutoff: Optional[float] = None) -> Dict[str, float]:
    """How well the polyhedron fixes the frame: the number of donors and the moment ratios.

    Two equal moments leave a rotation free, and the ratios say how near that is.  A caller that
    needs one frame per complex should refuse a complex whose two smallest moments agree to
    within a per cent or so, or accept that its orientation about that axis is arbitrary.
    """
    idx = ion_index(mol, ion) if index is None else int(index)
    donors, d = coordination(mol, idx, ion, cutoff)
    out = {"donors": float(len(donors)),
           "nearest": float(d[0]) if len(d) else float("nan"),
           "mean_distance": float(d.mean()) if len(d) else float("nan")}
    if len(donors) < 2:
        out.update({"moment_ratio_xy": float("nan"), "moment_ratio_yz": float("nan")})
        return out
    pos = np.asarray(mol.GetConformer().GetPositions(), float)
    p = pos[donors] - pos[idx]
    r2 = (p ** 2).sum(axis=1)
    moments = np.linalg.eigvalsh(np.eye(3) * r2.sum() - p.T @ p)
    m = np.maximum(moments, 1e-12)
    out.update({"moment_ratio_xy": float(m[0] / m[1]), "moment_ratio_yz": float(m[1] / m[2])})
    return out


def with_frame(mol: Chem.Mol, index: Optional[int] = None, ion: str = "Li",
               cutoff: Optional[float] = None) -> Chem.Mol:
    """A copy of the complex whose conformer is the one :func:`cation_frame` gives."""
    out = Chem.Mol(mol)
    coords = cation_frame(mol, index, ion, cutoff)
    conf = out.GetConformer()
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in xyz]))
    return out


# ------------------------------------------------------------------------------ the energetics


@dataclass
class Engine:
    """How the complexes are optimised: GFN2-xTB by default, ORCA on request."""
    program: str = "xtb"
    workdir: str = "solvation_work"
    opt: bool = True
    solvent: str = ""                     # an implicit-solvent name, or "" for the gas phase
    accuracy: float = 1.0
    timeout: Optional[float] = 1800.0
    orca_keywords: str = "B3LYP D3BJ def2-SVP"
    #: filled as molecules are run: name -> total energy in eV
    energies: Dict[str, float] = field(default_factory=dict)

    def level(self) -> str:
        if self.program == "xtb":
            from cmfqsar.xtb import Config
            return Config(gfn=2, opt=self.opt, solvent=self.solvent,
                          accuracy=self.accuracy).level()
        return self.orca_keywords + (" opt" if self.opt else "")


def optimise(mol: Chem.Mol, name: str, engine: Optional[Engine] = None,
             index: int = 1) -> Tuple[Chem.Mol, float]:
    """Optimise one molecule or complex and return it with its total energy in eV.

    The call is cached by :mod:`cmfqsar.xtb` on the molecule, its charge, the method and the
    starting geometry, so running this again costs nothing and an interrupted series resumes.
    """
    engine = engine or Engine()
    if engine.program != "xtb":
        raise SolvationError(f"the {engine.program!r} engine is not wired up here yet; "
                             f"use program='xtb'")
    from cmfqsar.xtb import Config, run_molecule
    cfg = Config(gfn=2, opt=engine.opt, fukui=False, solvent=engine.solvent,
                 accuracy=engine.accuracy, timeout=engine.timeout, workdir=engine.workdir)
    outcome = run_molecule(mol, name, int(index), cfg)
    if outcome.error or outcome.mol is None:
        raise SolvationError(f"{name}: {outcome.error or 'no molecule came back'}")
    props = outcome.mol.GetPropsAsDict()
    if "E_total" not in props:
        raise SolvationError(f"{name}: the run reported no total energy")
    e = float(props["E_total"]) * HARTREE_EV
    engine.energies[name] = e
    return outcome.mol, e


def ion_energy(ion: str = "Li", engine: Optional[Engine] = None) -> float:
    """The total energy of the bare cation in eV, which every binding energy needs."""
    engine = engine or Engine()
    key = f"ion_{ion}"
    if key in engine.energies:
        return engine.energies[key]
    atom = Chem.Atom(ion)
    atom.SetFormalCharge(int(ION_CHARGE[ion]))
    atom.SetNoImplicit(True)
    rw = Chem.RWMol()
    rw.AddAtom(atom)
    mol = rw.GetMol()
    conf = Chem.Conformer(1)
    conf.SetAtomPosition(0, Point3D(0.0, 0.0, 0.0))
    mol.AddConformer(conf, assignId=True)
    Chem.SanitizeMol(mol)
    # a single ion has nothing to optimise, and asking for it makes xtb refuse
    bare = Engine(program=engine.program, workdir=engine.workdir, opt=False,
                  solvent=engine.solvent, accuracy=engine.accuracy, timeout=engine.timeout)
    _m, e = optimise(mol, key, bare, index=1)
    engine.energies[key] = e
    return e


def binding_energy(complex_ev: float, ion_ev: float, solvent_evs: Sequence[float]) -> float:
    """``E(complex) - E(ion) - sum E(solvent)``, in eV: negative when the complex is bound.

    The solvent energies are those of the free molecules at their own optimised geometries, so
    this is the binding energy with the deformation of the molecules included, which is how the
    reference set of ``examples/ether/`` defines it as well.
    """
    return float(complex_ev) - float(ion_ev) - float(np.sum(np.asarray(solvent_evs, float)))


def best_wrapped_complex(solvent: Chem.Mol, name: str, ion: str = "Li", n_donors: int = 0,
                         conformers: int = 5, engine: Optional[Engine] = None,
                         index: int = 1) -> Tuple[Complex, float, List[float]]:
    """Wrap the ion in several conformers, optimise each, and keep the one of lowest energy.

    :func:`wrap_ion` does no conformational search, so the fold it finds is the fold of the
    conformer it was handed, and for a chain that wraps a cation with four of its own oxygens
    that is usually not enough. This is the smallest honest remedy: a few starting conformers,
    each optimised, the lowest kept. Returns the complex, its energy in eV and every energy
    tried, so that the spread over the starts can be reported rather than hidden.
    """
    engine = engine or Engine()
    best, tried = None, []
    for c in range(max(1, int(conformers))):
        try:
            built = wrap_ion(solvent, ion=ion, n_donors=n_donors, seed=index * 97 + c)
            mol, e = optimise(built.mol, f"{name}_c{c}", engine, index=index * 100 + c)
        except Exception:                                      # noqa: BLE001 - a start may fail
            continue
        tried.append(e)
        if best is None or e < best[1]:
            built.mol = mol
            built.energy_ev = e
            built.optimised = True
            best = (built, e)
    if best is None:
        raise SolvationError(f"{name}: no starting conformer gave a complex that optimised")
    out = best[0]
    spread = (max(tried) - min(tried)) if len(tried) > 1 else 0.0
    out.note += (f"; lowest of {len(tried)} starting conformers, which spread "
                 f"{spread:.3f} eV")
    return out, best[1], tried


def desolvation_energy(complex_n_ev: float, complex_n_minus_one_ev: float,
                       solvent_ev: float) -> float:
    """What it costs to take one molecule out of the shell, in eV, positive when it costs energy.

    ``E(n-1 complex) + E(free solvent) - E(n complex)``: the two products against the one
    reactant, both at their own optimised geometries.
    """
    return float(complex_n_minus_one_ev) + float(solvent_ev) - float(complex_n_ev)
