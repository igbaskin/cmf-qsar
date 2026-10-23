"""Field-point graphs: an alignment that needs neither a common substructure nor a model.

Every alignment this program has is a superposition on something shared.  ``mcs_align`` needs a
common substructure, which a series of several scaffolds does not have; ``field_align`` refines a
pose that something else must first put in roughly the right place, because the CMF kernel is flat
where two molecules do not overlap and there is nothing to climb; ``varalign`` and ``flexalign``
need a fitted model, so they cannot align the training set that the model is fitted on.  The
measurements in this repository all say the same thing about the consequence: the alignment decides
more than the kernel, the learner or the fields, and on a heterogeneous set the frame is where the
method fails.

The construction here is the third road, and it is old (Thorner, Willett, Wright & Taylor 1997 for
field graphs; the field points of Cresset's methodology, and the maximum-common-subgraph matching
of the virtual-screening literature).  Reduce each molecule's *own* fields to a handful of typed
extrema, treat those as the vertices of a graph whose edges carry the distances between them, and
superimpose two molecules on the largest set of extrema whose mutual distances agree.  Nothing is
shared in advance: not a substructure, not a reference frame, not a model.  What is matched is
what the molecule presents to a binding partner, which is the thing the fields are a description
of in the first place.

How it works, in the order the code does it:

1. **Field points.**  A connected region where one field exceeds a fraction of its own largest
   value is one point, with the position, tolerance radius and integrated weight that
   :mod:`cmfqsar.pharmacophore` already computes for a co-field lobe -- the same discretisation,
   applied to a molecule's own field rather than to a model's.  The type of a point is the field
   it came from and the sign of the lobe, so a positive electrostatic extremum can only ever be
   matched to another positive electrostatic extremum.
2. **The correspondence graph.**  One vertex per pair of points of the same type, one from each
   molecule.  Two vertices are joined when they are compatible: they reuse no point, and the
   distance between the two points of one molecule agrees with the distance between their
   partners in the other to within a tolerance.  A clique is then a set of mutually consistent
   pairings, which is a candidate common arrangement.
3. **The cliques.**  Bron-Kerbosch with pivoting, budgeted, since the number of maximal cliques
   is exponential in the worst case and a screening run may not hang.  Three points are the
   minimum that fixes a rigid body, and the superposition that follows uses a proper rotation, so
   a mirror-image arrangement is never accepted as a match.
4. **Choosing among them.**  A graph can have several maximum cliques and they are not
   interchangeable: they are alternative hypotheses about which features correspond.  Each is
   superimposed by Kabsch on the matched extrema and then scored by the quantity the program
   actually cares about, the cosine of the two molecules' fields.  The best wins.  This is the
   one step that uses the CMF kernel, and it uses it to *rank* complete hypotheses rather than to
   search, which is what keeps it out of the flat region that defeats ``field_align``.

**The steric field is left out, and that is measured rather than assumed.**  At every contour
tried, from 0.2 to 0.8 of its peak, the steric field of a drug-sized molecule has exactly one
connected region: it is a single blob covering the molecule.  Its one extremum sits near the
centroid and carries no information about where anything is, so including it would add a vertex
that matches everything and constrains nothing.  The fields that do have localised structure are
the electrostatic one, with two signs, the hydrophobic one, also with two signs because the
Crippen contributions are signed, and the two hydrogen-bond fields; together they give nine to
thirteen points for a ligand of thirty to sixty atoms, which is the regime the field-point
literature works in.

What this does not do: it does not change a conformation, so a molecule can only be placed as
well as the conformation it arrives in allows, and it does not know which of its own points
matter, since a molecule's field says what it presents and not what the target wants.  A fitted
model knows the second thing, which is why the confidence alignment exists and why the two are
worth running in series.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .chemotypes import _kabsch                    # the same proper-rotation Kabsch, not a copy
from .cofields import Grid, field_volume, make_grid
from .fielddiff import overall_similarity
from .fields import FieldSet, MoleculeFields
from .pharmacophore import features_from_volume

#: fields whose extrema are localised enough to carry positional information, with their signs.
#: ``steric`` is deliberately absent; the module docstring says why, and a test asserts the
#: single-region behaviour that is the reason.
DEFAULT_FIELDS = ("electrostatic", "hydrophobic", "hba", "hbd")

#: how a point of each type is written
POINT_LABELS = {
    ("electrostatic", +1): "positive",
    ("electrostatic", -1): "negative",
    ("hydrophobic", +1): "hydrophobic",
    ("hydrophobic", -1): "polar",
    ("hba", +1): "acceptor",
    ("hba", -1): "acceptor-",
    ("hbd", +1): "donor",
    ("hbd", -1): "donor-",
    ("steric", +1): "bulk",
    ("steric", -1): "void",
}


@dataclass
class FieldPoint:
    """One extremum of one field of one molecule: where it is, how big, and how much of it."""
    field: str
    sign: int
    centre: np.ndarray
    radius: float
    weight: float

    @property
    def type_key(self) -> Tuple[str, int]:
        """What a point may be matched to: its field and the sign of the lobe."""
        return (self.field, int(self.sign))

    @property
    def label(self) -> str:
        return POINT_LABELS.get(self.type_key, f"{self.field}{'+' if self.sign > 0 else '-'}")

    def describe(self) -> str:
        return (f"{self.label:12s} at ({self.centre[0]:7.2f},{self.centre[1]:7.2f},"
                f"{self.centre[2]:7.2f})  radius {self.radius:4.2f} A  weight {self.weight:9.3f}")


def field_points(mol: MoleculeFields, fieldset: FieldSet,
                 fields: Optional[Sequence[str]] = None, level: float = 0.3,
                 min_radius: float = 0.5, spacing: float = 0.5, margin: float = 2.0,
                 max_points: int = 16, grid: Optional[Grid] = None) -> List[FieldPoint]:
    """The typed extrema of a molecule's own fields, strongest first.

    The grid is built around this molecule alone, because a field point is a property of the
    molecule and not of any frame it is being compared in; two molecules' point sets are
    therefore computed independently and only their *distances* are ever compared.
    """
    want = set(DEFAULT_FIELDS if fields is None else fields)
    if grid is None:
        grid = make_grid(mol.coords, margin=margin, spacing=spacing)
    out: List[FieldPoint] = []
    for f, spec in enumerate(fieldset.specs):
        if spec.name not in want:
            continue
        vol = field_volume(mol, spec, f, grid)
        for sign in (+1, -1):
            for feat in features_from_volume(vol, grid, spec.name, sign, level=level,
                                             min_radius=min_radius):
                out.append(FieldPoint(field=spec.name, sign=sign,
                                      centre=np.asarray(feat.centre, float),
                                      radius=float(feat.radius),
                                      weight=abs(float(feat.importance))))
    out.sort(key=lambda p: -p.weight)
    return out[:max_points] if max_points and max_points > 0 else out


# --------------------------------------------------------------------------- #
# The correspondence graph and its cliques
# --------------------------------------------------------------------------- #

def correspondence_graph(a: Sequence[FieldPoint], b: Sequence[FieldPoint],
                         tolerance: float = 1.0) -> Tuple[List[Tuple[int, int]], List[set]]:
    """Vertices (pairings of points of the same type) and their compatibility adjacency.

    Two pairings are compatible when they share no point and the distance between the two points
    taken from `a` agrees, to within `tolerance`, with the distance between their partners in
    `b`.  A clique of this graph is a correspondence whose internal geometry is consistent, which
    is a necessary condition for a rigid superposition and not a sufficient one -- the
    superposition itself still has to be computed and checked, since distance agreement alone
    admits arrangements that no rotation can bring together.
    """
    pairs = [(i, j) for i, pa in enumerate(a) for j, pb in enumerate(b)
             if pa.type_key == pb.type_key]
    n = len(pairs)
    if n == 0:
        return [], []
    ca = np.array([p.centre for p in a], float) if len(a) else np.zeros((0, 3))
    cb = np.array([p.centre for p in b], float) if len(b) else np.zeros((0, 3))
    da = np.linalg.norm(ca[:, None, :] - ca[None, :, :], axis=-1) if len(ca) else np.zeros((0, 0))
    db = np.linalg.norm(cb[:, None, :] - cb[None, :, :], axis=-1) if len(cb) else np.zeros((0, 0))
    adj: List[set] = [set() for _ in range(n)]
    tol = float(tolerance)
    for u in range(n):
        iu, ju = pairs[u]
        for v in range(u + 1, n):
            iv, jv = pairs[v]
            if iu == iv or ju == jv:
                continue
            if abs(da[iu, iv] - db[ju, jv]) <= tol:
                adj[u].add(v)
                adj[v].add(u)
    return pairs, adj


def maximal_cliques(adj: Sequence[set], min_size: int = 3, max_cliques: int = 5000,
                    budget: int = 200000) -> List[List[int]]:
    """Maximal cliques of at least `min_size`, by Bron-Kerbosch with pivoting.

    The count of maximal cliques is exponential in the worst case, so the search carries a
    budget and returns what it found when either limit is reached.  That makes the result a
    heuristic under pressure and exact when the budget is not binding; a caller that cares can
    raise both, and a screening run that cares more about finishing can lower them.
    """
    n = len(adj)
    found: List[List[int]] = []
    if n == 0:
        return found
    steps = [0]
    neighbours = [set(s) for s in adj]

    def expand(r: List[int], p: set, x: set) -> None:
        if steps[0] >= budget or len(found) >= max_cliques:
            return
        steps[0] += 1
        if not p and not x:
            if len(r) >= min_size:
                found.append(list(r))
            return
        # pivot: the vertex of P | X with the most neighbours in P
        pux = p | x
        pivot = max(pux, key=lambda v: len(neighbours[v] & p)) if pux else None
        candidates = list(p - neighbours[pivot]) if pivot is not None else list(p)
        for v in candidates:
            if steps[0] >= budget or len(found) >= max_cliques:
                return
            expand(r + [v], p & neighbours[v], x & neighbours[v])
            p = p - {v}
            x = x | {v}

    expand([], set(range(n)), set())
    found.sort(key=lambda c: -len(c))
    return found


# --------------------------------------------------------------------------- #
# Superposition
# --------------------------------------------------------------------------- #

def superpose_pairs(a: Sequence[FieldPoint], b: Sequence[FieldPoint],
                    pairs: Sequence[Tuple[int, int]]) -> Tuple[np.ndarray, np.ndarray, float]:
    """The rigid transform taking the matched points of `a` onto those of `b`, and their RMSD.

    Kabsch, with the proper-rotation correction, so a correspondence that would need a reflection
    is fitted as well as a rotation can fit it and reports the residual rather than being
    silently mirrored into place.
    """
    if len(pairs) < 3:
        raise ValueError("a rigid superposition needs at least three matched points")
    P = np.array([a[i].centre for i, _ in pairs], float)
    Q = np.array([b[j].centre for _, j in pairs], float)
    R, t = _kabsch(P, Q)
    moved = P @ R.T + t
    rmsd = float(np.sqrt(np.mean(np.sum((moved - Q) ** 2, axis=1))))
    return R, t, rmsd


def _transformed(mol: MoleculeFields, R: np.ndarray, t: np.ndarray) -> MoleculeFields:
    """A copy of `mol` with its coordinates moved; the fields themselves are unchanged."""
    import copy
    out = copy.copy(mol)
    out.coords = np.asarray(mol.coords, float) @ np.asarray(R, float).T + np.asarray(t, float)
    return out


@dataclass
class FieldGraphMatch:
    """One hypothesis about how two molecules correspond, and how good it turned out to be."""
    pairs: List[Tuple[int, int]] = _field(default_factory=list)
    size: int = 0                      # matched points
    point_rmsd: float = float("nan")   # of the matched extrema after superposition
    similarity: float = float("nan")   # cosine of the two molecules' fields, once superimposed
    weight: float = 0.0                # total weight of the matched extrema
    R: Optional[np.ndarray] = None
    t: Optional[np.ndarray] = None
    n_points: Tuple[int, int] = (0, 0)
    n_cliques: int = 0

    def describe(self) -> str:
        return (f"{self.size} of {self.n_points[0]}/{self.n_points[1]} points matched, "
                f"point RMSD {self.point_rmsd:.2f} A, field cosine {self.similarity:+.3f}, "
                f"{self.n_cliques} candidate correspondences")


def align(mol: MoleculeFields, reference: MoleculeFields, fieldset: FieldSet,
          tolerance: float = 1.0, min_size: int = 3, level: float = 0.3,
          min_radius: float = 0.5, spacing: float = 0.5, margin: float = 2.0,
          max_points: int = 16, max_candidates: int = 40, max_cliques: int = 5000,
          fields: Optional[Sequence[str]] = None,
          ref_points: Optional[Sequence[FieldPoint]] = None) -> Optional[FieldGraphMatch]:
    """Place `mol` on `reference` by matching their field points; returns the best hypothesis.

    `ref_points` lets a caller aligning a whole series compute the reference's points once.
    Returns None when the two molecules share fewer than `min_size` compatible points or no
    consistent correspondence exists, which is a real answer: two molecules presenting nothing in
    common cannot be superimposed by what they present, and a method that returned a frame anyway
    would be inventing one.
    """
    pa = field_points(mol, fieldset, fields=fields, level=level, min_radius=min_radius,
                      spacing=spacing, margin=margin, max_points=max_points)
    pb = list(ref_points) if ref_points is not None else field_points(
        reference, fieldset, fields=fields, level=level, min_radius=min_radius,
        spacing=spacing, margin=margin, max_points=max_points)
    pairs, adj = correspondence_graph(pa, pb, tolerance=tolerance)
    if not pairs:
        return None
    cliques = maximal_cliques(adj, min_size=min_size, max_cliques=max_cliques)
    if not cliques:
        return None
    best: Optional[FieldGraphMatch] = None
    # the largest cliques first, and only as many as asked for: a bigger correspondence is a
    # stronger hypothesis, but the field cosine decides between them
    for clique in cliques[:max_candidates]:
        chosen = [pairs[v] for v in clique]
        try:
            R, t, rmsd = superpose_pairs(pa, pb, chosen)
        except (ValueError, np.linalg.LinAlgError):
            continue
        sim = overall_similarity(_transformed(mol, R, t), reference, fieldset)
        if not np.isfinite(sim):
            continue
        cand = FieldGraphMatch(pairs=chosen, size=len(chosen), point_rmsd=rmsd, similarity=sim,
                               weight=float(sum(pa[i].weight + pb[j].weight for i, j in chosen)),
                               R=R, t=t, n_points=(len(pa), len(pb)), n_cliques=len(cliques))
        if best is None or cand.similarity > best.similarity:
            best = cand
    return best


def align_dataset(mols: Sequence[MoleculeFields], reference: MoleculeFields,
                  fieldset: FieldSet, progress=None, **kwargs) -> List[Optional[FieldGraphMatch]]:
    """Align a whole series on one reference, computing the reference's points once."""
    ref_points = field_points(
        reference, fieldset, fields=kwargs.get("fields"),
        level=kwargs.get("level", 0.3), min_radius=kwargs.get("min_radius", 0.5),
        spacing=kwargs.get("spacing", 0.5), margin=kwargs.get("margin", 2.0),
        max_points=kwargs.get("max_points", 16))
    out: List[Optional[FieldGraphMatch]] = []
    for i, m in enumerate(mols):
        if progress:
            progress(f"Field-graph alignment {i + 1}/{len(mols)}", i / max(len(mols), 1))
        if m is reference:
            out.append(None)
            continue
        out.append(align(m, reference, fieldset, ref_points=ref_points, **kwargs))
    return out


def summarise(matches: Sequence[Optional[FieldGraphMatch]]) -> Dict[str, float]:
    """How the alignment went over a series: how many were placed, and how well."""
    done = [m for m in matches if m is not None]
    if not done:
        return {"n": float(len(matches)), "placed": 0.0}
    return {"n": float(len(matches)), "placed": float(len(done)),
            "mean_size": float(np.mean([m.size for m in done])),
            "mean_point_rmsd": float(np.mean([m.point_rmsd for m in done])),
            "mean_similarity": float(np.mean([m.similarity for m in done]))}
