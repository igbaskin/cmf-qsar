"""
Alignment onto a plane: the frame of a molecule on a surface, without the surface.

A molecule adsorbed on an electrode, a corrosion inhibitor on steel, a modifier on a copper
cathode, lives in a frame that no reference molecule defines: the normal of the surface is z,
the top atomic layer is z = 0, and the molecule's height, tilt and face are the pose.  The
surface autocorrelation kernel (:mod:`cmfqsar.invariant`) reads such poses without any
in-plane registration, and the Fe(110) study found that the in-plane position over the lattice
carries nothing for the adsorption energy while the height carries everything.  What the surface
kernel cannot give is a co-field, because it has no fixed frame to draw one in.  The overlap
kernel can, if the molecules are in the surface frame; and so far the only way into that frame
has been to relax every molecule on a slab with a machine-learned potential.

This module is the cheap way in.  It poses a molecule on z = 0 by rule, the same rules the UMA
adsorption script uses for its *starting* poses:

* ``flat``: the plane of the two longest principal axes is laid parallel to the surface, the
  longest axis along x, and the flattest face turned down (the out-of-plane atoms point away
  from the surface, which is where a relaxation leaves a substituent that cannot bond);
  the lowest heavy atom is put at `height`;
* ``upright``: the most basic nitrogen (by a rule: aliphatic amine > pyridine-type > imine >
  aniline > pyrrole-type N-H; amide, nitro and quaternary N never; a thioether S or an ether O
  when there is no nitrogen) points its lone pair at the surface and sits at `anchor_height`,
  with the smallest tilt that leaves every other atom above it.

Three further pieces turn a rule into an alignment:

* :func:`surface_field_align` refines a pose by the field similarity to a reference pose, as
  :func:`cmfqsar.align.field_align` does, but with the degrees of freedom a surface leaves:
  translation in the plane and rotation about the normal (``dof="plane"``), plus tilt and
  height (``dof="plane+tilt"``).  The reference is one molecule whose pose is trusted -- a
  relaxed pose, a crystal pose, or the rule applied to the scaffold.
* :func:`plane_align` can try several starting poses (flat on either face, upright) and keep
  the one most similar in field to the reference (``pose="auto"``), which is the alignment
  criterion of the 2013 paper applied to a choice the rules cannot make.
* :func:`in_plane_rmsd` judges a pose against a known one modulo the motions the surface does
  not fix (rotation about z, translation in the plane), so a molecule placed over a different
  lattice site is not counted as misplaced; :func:`height_rmsd` judges the heights alone.

Everything here is rigid: the conformation is never touched.  The measurement that goes with
the module is ``benchmarks/run_fe110.py``, which asks whether rule-posed molecules reproduce
the UMA-relaxed poses, and what the overlap kernel in the rule frame recovers of the adsorption
energy against the surface kernel and against the relaxed frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from scipy.optimize import minimize

from .align import _kernel_score, set_coords
from .fields import AtomTable, FieldSet, build_fields
from .kernels import atom_kernel_block

POSES = ("flat", "flat-up", "upright", "auto")
DOFS = ("plane", "plane+tilt")


# --------------------------------------------------------------------------- #
# The anchor rule
# --------------------------------------------------------------------------- #

def _is_carbonyl_carbon(atom: Chem.Atom) -> bool:
    return atom.GetAtomicNum() == 6 and any(
        b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetAtomicNum() in (8, 16)
        for b in atom.GetBonds())


def anchor_atom(mol: Chem.Mol) -> Optional[int]:
    """The atom that points at the surface in the upright pose: the most basic nitrogen, by a
    simple rule (aliphatic amine > pyridine-type aromatic N > imine > aniline > pyrrole-type
    NH; amide, nitro and quaternary N never), else a thiol/thioether S, else a hydroxyl/ether O.
    ``None`` when the molecule has nothing of the kind.  The rule is the one of the UMA
    adsorption script, with one repair: an oxygen on a nitrogen (nitro, N-oxide) is not an
    anchor, which the script's version let through."""
    best, best_score = None, 0.0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        score = 0.0
        if z == 7:
            if atom.GetFormalCharge() != 0:
                continue
            heavy = [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]
            if any(_is_carbonyl_carbon(n) for n in heavy):
                continue                                                    # amide / sulfonamide-like
            if any(n.GetAtomicNum() in (7, 8) and not n.GetIsAromatic() for n in heavy):
                continue                                                    # nitro, N-oxide, hydrazide
            if atom.GetIsAromatic():
                score = 3.0 if atom.GetTotalNumHs() == 0 and atom.GetDegree() == 2 else 0.5
            elif any(n.GetIsAromatic() for n in heavy):
                score = 1.5                                                 # aniline-type
            elif any(b.GetBondType() == Chem.BondType.DOUBLE for b in atom.GetBonds()):
                score = 2.0                                                 # imine
            else:
                score = 4.0                                                 # aliphatic amine
        elif z == 16:
            if atom.GetDegree() <= 2 and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
                score = 0.4                                                 # thiol / thioether
            elif atom.GetDegree() == 1:
                score = 0.4                                                 # thione S
        elif z == 8:
            if atom.GetFormalCharge() != 0 or any(n.GetAtomicNum() == 7 for n in atom.GetNeighbors()):
                continue                                                    # nitro / N-oxide oxygen
            if atom.GetDegree() <= 2 and all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds()):
                score = 0.2
        if score > best_score:
            best, best_score = atom.GetIdx(), score
    return best


def lone_pair_direction(mol: Chem.Mol, coords: np.ndarray, anchor: int) -> np.ndarray:
    """Where the anchor atom's lone pair points: away from the mean of its bonded neighbours
    (away from the centroid when it has none)."""
    nbrs = [n.GetIdx() for n in mol.GetAtomWithIdx(anchor).GetNeighbors()]
    if not nbrs:
        return coords[anchor] - coords.mean(axis=0)
    return coords[anchor] - coords[nbrs].mean(axis=0)


# --------------------------------------------------------------------------- #
# Rotations
# --------------------------------------------------------------------------- #

def principal_axes(coords: np.ndarray) -> np.ndarray:
    """Rows: the principal axes of the point cloud by decreasing extent, a right-handed frame."""
    centred = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axes = np.eye(3)
    axes[:len(vt)] = vt
    if np.linalg.det(axes) < 0:
        axes[2] *= -1
    return axes


def rotation_to_z(direction: np.ndarray, sign: float = -1.0) -> np.ndarray:
    """The rotation matrix that takes `direction` to ``sign * z``."""
    d = direction / (np.linalg.norm(direction) + 1e-12)
    target = np.array([0.0, 0.0, sign])
    v = np.cross(d, target)
    c = float(np.dot(d, target))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def axis_rotation(axis: Sequence[float], angle: float) -> np.ndarray:
    """Rodrigues' rotation matrix about a unit axis."""
    k = np.asarray(axis, dtype=float)
    k = k / (np.linalg.norm(k) + 1e-12)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * kx @ kx


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _tilt_anchor_lowest(coords: np.ndarray, anchor: int, clearance: float = 0.0) -> np.ndarray:
    """The smallest tilt about the anchor atom that leaves every other atom at least `clearance`
    above it (an azole with its lone pair straight down has the neighbouring C-H lower than the
    N; the tilt lets the N, not the H, meet the surface first).  A grid over tilt axes in the
    plane and tilt angles; the first angle that clears is used.  Where no tilt up to 90 degrees
    clears, the lone pair stays straight down."""
    out = coords - coords[anchor]
    others = np.delete(np.arange(len(out)), anchor)
    if len(others) == 0 or out[others, 2].min() >= clearance - 1e-9:
        return coords
    for angle in np.radians(np.arange(1.0, 91.0, 1.0)):
        for psi in np.radians(np.arange(0.0, 360.0, 10.0)):
            rot = axis_rotation([math.cos(psi), math.sin(psi), 0.0], angle)
            cand = out @ rot.T
            if cand[others, 2].min() >= clearance - 1e-9:
                return cand + coords[anchor]
    return coords


# --------------------------------------------------------------------------- #
# The poses
# --------------------------------------------------------------------------- #

def heavy_mask(mol: Chem.Mol) -> np.ndarray:
    return np.array([a.GetAtomicNum() > 1 for a in mol.GetAtoms()], dtype=bool)


def orient_flat(coords: np.ndarray, heavy: Optional[np.ndarray] = None, face: str = "down") -> np.ndarray:
    """The molecule rotated about its centroid so that the plane of its two longest principal
    axes (over the heavy atoms when `heavy` is given) is the xy plane and the longest axis lies
    along x.  ``face="down"`` turns the flattest face to the surface: the atoms off the plane
    end up at positive z.  ``face="up"`` is the other face; ``face="asis"`` leaves the sign to
    the right-handed frame."""
    centred = coords - coords.mean(axis=0)
    sel = centred[heavy] if heavy is not None and heavy.any() else centred
    axes = principal_axes(sel)
    out = centred @ axes.T
    if face in ("down", "up"):
        zz = (out[heavy, 2] if heavy is not None and heavy.any() else out[:, 2])
        # the side with more out-of-plane material is the side that should point away
        off = np.sum(zz ** 3)
        want_positive = face == "down"
        if (off < 0) == want_positive and abs(off) > 1e-9:
            out = out @ np.diag([1.0, -1.0, -1.0]).T          # a 180-degree turn about x
    elif face != "asis":
        raise ValueError(f"face must be 'down', 'up' or 'asis', not {face!r}")
    return out


def orient_upright(mol: Chem.Mol, coords: np.ndarray, anchor: int) -> np.ndarray:
    """The molecule rotated about its centroid so that the anchor atom's lone pair points at
    the surface (-z), tilted by the least that puts the anchor lowest, and turned about z so
    that its longest in-plane extent lies along x; the anchor over x = y = 0."""
    centred = coords - coords.mean(axis=0)
    direction = lone_pair_direction(mol, centred, anchor)
    out = _tilt_anchor_lowest(centred @ rotation_to_z(direction, sign=-1.0).T, anchor)
    xy = out[:, :2] - out[:, :2].mean(axis=0)
    if len(out) > 1 and np.abs(xy).max() > 1e-6:
        _, _, vt = np.linalg.svd(xy, full_matrices=False)
        cos_t, sin_t = vt[0]
        out = out @ np.array([[cos_t, sin_t, 0.0], [-sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]]).T
    return out - out[anchor] * np.array([1.0, 1.0, 0.0])


def place_at_height(coords: np.ndarray, heavy: np.ndarray, height: float,
                    anchor: Optional[int] = None) -> np.ndarray:
    """Shift along z so that the lowest heavy atom -- or the anchor atom -- is at `height`, and
    in the plane so that the heavy-atom centroid is over x = y = 0 (the anchor stays over the
    origin when there is one)."""
    out = coords.copy()
    low = out[anchor, 2] if anchor is not None else out[heavy, 2].min()
    out[:, 2] += height - low
    if anchor is None:
        out[:, :2] -= out[heavy, :2].mean(axis=0)
    return out


@dataclass
class PlanePose:
    name: str
    label: str                       # flat | flat-up | upright
    anchor: Optional[int]
    similarity: float = float("nan")  # field similarity to the reference, when one was used
    candidates: List[Tuple[str, float]] = field(default_factory=list)

    def describe(self) -> str:
        s = f"{self.name:28s} {self.label:8s}"
        if self.anchor is not None:
            s += f" anchor {self.anchor:3d}"
        if np.isfinite(self.similarity):
            s += f"  similarity {self.similarity:.3f}"
        if len(self.candidates) > 1:
            s += "  (" + ", ".join(f"{l} {v:.3f}" for l, v in self.candidates) + ")"
        return s


def rule_pose(mol: Chem.Mol, label: str, height: float = 2.2, anchor_height: float = 2.0) -> Tuple[np.ndarray, Optional[int]]:
    """The coordinates of `mol` in the pose `label` (``flat``, ``flat-up`` or ``upright``), and
    the anchor atom used (``None`` for the flat poses).  Raises ``ValueError`` for an upright
    pose when the molecule has no anchor."""
    coords = np.array(mol.GetConformer().GetPositions(), dtype=float)
    heavy = heavy_mask(mol)
    if label in ("flat", "flat-up"):
        out = orient_flat(coords, heavy, face="down" if label == "flat" else "up")
        return place_at_height(out, heavy, height), None
    if label == "upright":
        anchor = anchor_atom(mol)
        if anchor is None:
            raise ValueError("no anchor atom for the upright pose")
        out = orient_upright(mol, coords, anchor)
        return place_at_height(out, heavy, anchor_height, anchor=anchor), anchor
    raise ValueError(f"unknown pose {label!r}")


# --------------------------------------------------------------------------- #
# Refinement in the surface frame
# --------------------------------------------------------------------------- #

def _pose_transform(coords: np.ndarray, p: np.ndarray, centre: np.ndarray, dof: str) -> np.ndarray:
    """Rotation about z by p[0], then (plane+tilt) a tilt about the in-plane axis (cos p[1],
    sin p[1]) by p[2], about `centre`; translated by (p[-3], p[-2]) in the plane and, with
    tilt, by p[-1] along z."""
    r = _rot_z(p[0])
    if dof == "plane+tilt":
        r = axis_rotation([math.cos(p[1]), math.sin(p[1]), 0.0], p[2]) @ r
        t = np.array([p[3], p[4], p[5]])
    else:
        t = np.array([p[1], p[2], 0.0])
    return (coords - centre) @ r.T + centre + t


def _n_params(dof: str) -> int:
    if dof == "plane":
        return 3
    if dof == "plane+tilt":
        return 6
    raise ValueError(f"dof must be one of {DOFS}, not {dof!r}")


class _Reference:
    """The reference pose's field tables, built once for a set."""

    def __init__(self, ref: Chem.Mol, fieldset: FieldSet):
        self.fieldset = fieldset
        self.fields = build_fields(ref, fieldset)
        self.table = AtomTable.from_molecules([self.fields])
        self.specs = fieldset.specs
        self.h = fieldset.mixing_vector()
        self.self_kernel = atom_kernel_block(self.fields.coords, self.fields.weights, self.fields.radii,
                                             self.table, self.specs).sum(axis=1)[:, 0]

    def score(self, probe, coords: np.ndarray, probe_self: np.ndarray) -> float:
        return _kernel_score(coords, probe, self.table, self.specs, self.self_kernel, probe_self, self.h)


def _probe(mol: Chem.Mol, fieldset: FieldSet):
    probe = build_fields(mol, fieldset)
    table = AtomTable.from_molecules([probe])
    self_k = atom_kernel_block(probe.coords, probe.weights, probe.radii, table, fieldset.specs).sum(axis=1)[:, 0]
    return probe, self_k


def surface_field_align(mol: Chem.Mol, ref: Chem.Mol, fieldset: FieldSet, dof: str = "plane",
                        restarts: int = 4, seed: int = 0, max_iter: int = 300,
                        max_tilt: float = 30.0, max_dz: float = 1.0,
                        _reference: Optional[_Reference] = None) -> Tuple[float, float]:
    """Refine the pose of `mol` on the surface by the normalised field similarity to `ref`,
    moving it only as a surface allows: in the plane and about the normal (``dof="plane"``),
    or also tilting it by up to `max_tilt` degrees and lifting it by up to `max_dz` A
    (``dof="plane+tilt"``).  Returns (similarity before, similarity after); `mol` is moved in
    place.  The restarts are rotations about z spread over the circle, since the in-plane
    orientation is the one thing a rule leaves open."""
    n = _n_params(dof)
    reference = _reference if _reference is not None else _Reference(ref, fieldset)
    probe, probe_self = _probe(mol, fieldset)
    centre = probe.coords.mean(axis=0)
    tilt_lim = math.radians(max_tilt)

    def objective(p):
        if dof == "plane+tilt" and (abs(p[2]) > tilt_lim or abs(p[5]) > max_dz):
            return 1.0 + abs(p[2]) + abs(p[5])          # outside the box: worse than any similarity
        c = _pose_transform(probe.coords, p, centre, dof)
        return -reference.score(probe, c, probe_self)

    p0 = np.zeros(n)
    before = -objective(p0)
    best_val, best_p = objective(p0), p0
    starts = [p0]
    rng = np.random.RandomState(seed)
    for k in range(restarts):
        q = np.zeros(n)
        q[0] = 2.0 * math.pi * (k + 1) / (restarts + 1) + rng.normal(scale=0.1)
        q[-3:-1] += rng.normal(scale=0.5, size=2) if dof == "plane+tilt" else 0.0
        if dof == "plane":
            q[1:3] += rng.normal(scale=0.5, size=2)
        starts.append(q)
    for q in starts:
        res = minimize(objective, q, method="Powell",
                       options={"maxiter": max_iter, "xtol": 1e-3, "ftol": 1e-6})
        if res.fun < best_val:
            best_val, best_p = res.fun, res.x
    set_coords(mol, _pose_transform(probe.coords, best_p, centre, dof))
    return float(before), float(-best_val)


# --------------------------------------------------------------------------- #
# The aligner
# --------------------------------------------------------------------------- #

def plane_align(mol: Chem.Mol, pose: str = "flat", height: float = 2.2, anchor_height: float = 2.0,
                ref: Optional[Chem.Mol] = None, fieldset: Optional[FieldSet] = None,
                refine: Optional[str] = None, restarts: int = 4, seed: int = 0,
                name: str = "", _reference: Optional[_Reference] = None) -> PlanePose:
    """Pose `mol` on the surface z = 0, in place.

    `pose` is ``flat``, ``flat-up`` (the other face down), ``upright`` or ``auto``.  ``auto`` builds the candidates -- flat with
    either face down and, when the molecule has an anchor atom, upright -- and keeps the one
    with the highest field similarity to `ref` (which then needs `fieldset`); without a
    reference it falls back to ``flat``.  `refine` is ``None``, ``"plane"`` or
    ``"plane+tilt"``: with a reference, each candidate is refined by
    :func:`surface_field_align` with those degrees of freedom before the choice is made.
    """
    if pose not in POSES:
        raise ValueError(f"pose must be one of {POSES}, not {pose!r}")
    if refine is not None and refine not in DOFS:
        raise ValueError(f"refine must be None or one of {DOFS}, not {refine!r}")
    use_ref = ref is not None and fieldset is not None
    if pose == "auto" and not use_ref:
        pose = "flat"
    labels = [pose] if pose in ("flat", "flat-up", "upright") else ["flat", "flat-up", "upright"]
    reference = _reference if (_reference is not None or not use_ref) else _Reference(ref, fieldset)

    results: List[Tuple[str, float, np.ndarray, Optional[int]]] = []
    for label in labels:
        try:
            coords, anchor = rule_pose(mol, label, height=height, anchor_height=anchor_height)
        except ValueError:
            if pose != "auto":
                raise
            continue                                            # no anchor: no upright candidate
        sim = float("nan")
        if use_ref:
            trial = Chem.Mol(mol)
            set_coords(trial, coords)
            if refine is not None:
                _, sim = surface_field_align(trial, ref, fieldset, dof=refine, restarts=restarts,
                                             seed=seed, _reference=reference)
                coords = np.array(trial.GetConformer().GetPositions(), dtype=float)
            else:
                probe, probe_self = _probe(trial, fieldset)
                sim = reference.score(probe, probe.coords, probe_self)
        results.append((label, sim, coords, anchor))
    if not results:
        raise ValueError("no pose could be built")
    if use_ref:
        best = max(results, key=lambda r: r[1])
    else:
        best = results[0]
    set_coords(mol, best[2])
    return PlanePose(name=name or (mol.GetProp("_Name") if mol.HasProp("_Name") else ""),
                     label=best[0], anchor=best[3], similarity=best[1],
                     candidates=[(r[0], r[1]) for r in results])


def plane_align_dataset(mols: Sequence[Chem.Mol], names: Sequence[str], pose: str = "flat",
                        height: float = 2.2, anchor_height: float = 2.0,
                        ref_index: Optional[int] = None, fieldset: Optional[FieldSet] = None,
                        refine: Optional[str] = None, restarts: int = 4, seed: int = 0,
                        ref_as_is: bool = True, progress=None) -> List[PlanePose]:
    """Pose every molecule of a set on the surface, in place.  With `ref_index`, the molecule at
    that index is the reference the others are compared with: it is posed by the rule first
    unless `ref_as_is` is true and its coordinates are already a trusted pose."""
    ref = None
    reference = None
    if ref_index is not None:
        ref = mols[ref_index]
        if not ref_as_is:
            plane_align(ref, pose="flat" if pose == "auto" else pose, height=height,
                        anchor_height=anchor_height, name=names[ref_index])
        if fieldset is not None:
            reference = _Reference(ref, fieldset)
    out: List[PlanePose] = []
    for i, (m, nm) in enumerate(zip(mols, names)):
        if ref_index is not None and i == ref_index:
            out.append(PlanePose(name=nm, label="reference", anchor=None))
        else:
            out.append(plane_align(m, pose=pose, height=height, anchor_height=anchor_height,
                                   ref=ref, fieldset=fieldset, refine=refine, restarts=restarts,
                                   seed=seed, name=nm, _reference=reference))
        if progress:
            progress(i + 1, len(mols))
    return out


# --------------------------------------------------------------------------- #
# Judging a pose
# --------------------------------------------------------------------------- #

def in_plane_rmsd(coords: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """RMSD between `coords` and `target` after the best rotation about z and translation in
    the plane (Kabsch in two dimensions; z is compared as it stands).  This is the pose error a
    surface sees: the lattice site a molecule sits over is not part of it."""
    a = np.asarray(coords, float)
    b = np.asarray(target, float)
    if mask is not None:
        a, b = a[mask], b[mask]
    ca, cb = a[:, :2].mean(axis=0), b[:, :2].mean(axis=0)
    h = (a[:, :2] - ca).T @ (b[:, :2] - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d if d != 0 else 1.0]) @ u.T
    xy = (a[:, :2] - ca) @ r.T + cb
    dz = a[:, 2] - b[:, 2]
    return float(np.sqrt(np.mean(np.sum((xy - b[:, :2]) ** 2, axis=1) + dz ** 2)))


def height_rmsd(coords: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """RMSD of the heights alone."""
    a = np.asarray(coords, float)[:, 2]
    b = np.asarray(target, float)[:, 2]
    if mask is not None:
        a, b = a[mask], b[mask]
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rigid_rmsd(coords: np.ndarray, target: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """RMSD after the best rigid superposition in three dimensions: the floor a rigid pose of
    this conformation can reach, whatever the frame."""
    a = np.asarray(coords, float)
    b = np.asarray(target, float)
    if mask is not None:
        a, b = a[mask], b[mask]
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d if d != 0 else 1.0]) @ u.T
    moved = (a - ca) @ r.T + cb
    return float(np.sqrt(np.mean(np.sum((moved - b) ** 2, axis=1))))
