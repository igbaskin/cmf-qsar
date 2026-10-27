"""A matched pair as a field placed in space (plan item MC6).

A matched molecular pair is usually a row in a table: a transformation, written as two group
names, and the activity change it produced.  In a series whose molecules are in one frame the
same pair is something else as well -- **a difference of two fields**, a local object sitting
where the two groups differ.  Plan item MC6 asks whether that object carries the effect of the
transformation with it: can a change measured in one series be predicted from changes measured in
others?

Plan item D5 is both the warning and the reference.  Increments fitted on one series did not
transfer to another, and the vectors fitted on two sets agreed only to +0.89 at best.  An
increment is a number attached to an atom type; the field difference of this module is the same
transformation with its geometry kept, so it can tell a methyl that points into a pocket from one
that points at the solvent.  Whether that is enough is a measurement, not an expectation.

What one pair gives
-------------------
* Both members are in the frame of the series, so their fields are comparable with each other
  without further work.  What the two molecules do **not** share with another series is a frame,
  so the difference field is placed in a frame of its own: the origin at the atom of the core
  that carries the group, the x axis along the bond to the group, and the plane fixed by one more
  neighbour of that atom.  That frame is local and it is the same one for both members, so
  nothing about the pair is changed by it.
* The difference field is the field of the second group minus the field of the first, each
  restricted to the atoms of the group itself (the mask of :mod:`cmfqsar.bioisostere`).  It is an
  ordinary :class:`~cmfqsar.fields.MoleculeFields` with signed weights, so every kernel and every
  display of the program applies to it.
* The overlap kernel of two fields is bilinear, so the kernel of two difference fields is
  signed: reversing a pair reverses its kernel row.  A ridge with no intercept therefore predicts
  exactly the opposite change for the reverse of a pair, which is the one property such a model
  must have.

`benchmarks/run_transformfield.py` holds out whole series and asks whether the model beats
predicting no change at all, which is the gate of the item.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .bioisostere import MatchedPair, masked_field
from .fields import MoleculeFields
from .localalign import frame_from
from .localenv import difference_field

#: distances are rounded to this many angstroms when the second frame atom is chosen, so that the
#: same core gives the same frame in the two members of a pair
FRAME_ROUNDING = 0.1


# --------------------------------------------------------------------------- #
# The frame of an attachment point
# --------------------------------------------------------------------------- #


def attachment(mol, group_atoms: Sequence[int]) -> Optional[Tuple[int, int]]:
    """(the core atom carrying the group, the group atom bonded to it), or None.

    `mol` is the RDKit molecule the group was cut out of.  A single cut leaves exactly one bond
    between the two pieces; anything else is not an attachment point and is refused.
    """
    group = set(int(a) for a in group_atoms)
    found = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in group and j not in group:
            found.append((j, i))
        elif j in group and i not in group:
            found.append((i, j))
    return found[0] if len(found) == 1 else None


def attachment_frame(mol, fields: MoleculeFields, core_atom: int,
                     group_atom: int) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """(origin, rotation) of the attachment point, or None where no frame exists.

    The origin is the core atom, the x axis points at the group, and the plane is fixed by one
    more neighbour of the core atom.  That neighbour is chosen by its atom type and then by its
    distance to the group atom, rounded, so that the two members of a pair -- which share the
    core and are superimposed -- choose the same one.
    """
    coords = np.asarray(fields.coords, float)
    others = [n.GetIdx() for n in mol.GetAtomWithIdx(int(core_atom)).GetNeighbors()
              if n.GetIdx() != int(group_atom) and n.GetAtomicNum() > 1]
    if not others:
        return None
    def rank(k):
        d = float(np.linalg.norm(coords[k] - coords[int(group_atom)]))
        return (str(fields.atom_types[k]), round(d / FRAME_ROUNDING))
    second = sorted(others, key=rank)[0]
    R = frame_from(coords, int(group_atom), int(second), centre=int(core_atom))
    if R is None:
        return None
    return coords[int(core_atom)].copy(), R


def posed(fields: MoleculeFields, atoms: Sequence[int], origin: np.ndarray, R: np.ndarray,
          name: str = "") -> MoleculeFields:
    """The group's own atoms, moved into the frame of its attachment point."""
    part = masked_field(fields, atoms, name=name)
    part.coords = (np.asarray(part.coords, float) - np.asarray(origin, float)) @ np.asarray(
        R, float).T
    return part


# --------------------------------------------------------------------------- #
# A pair as a field
# --------------------------------------------------------------------------- #


@dataclass
class Transformation:
    """One matched pair as a difference field in the frame of its attachment point."""
    series: str
    key: Tuple[str, str]
    field: MoleculeFields
    delta: float
    names: Tuple[str, str] = ("", "")

    def __str__(self) -> str:
        return (f"{self.key[0]} -> {self.key[1]} in {self.series}: {self.field.n_atoms} atoms, "
                f"measured {self.delta:+.2f}")


def pair_field(mols, fields: Sequence[MoleculeFields], pair: MatchedPair,
               series: str = "") -> Optional[Transformation]:
    """The difference field of one matched pair, posed, or None where no frame exists.

    The field is the second group minus the first, so its sign goes with `pair.delta`.
    """
    a, b = int(pair.a), int(pair.b)
    at_a = attachment(mols[a], pair.atoms_a)
    at_b = attachment(mols[b], pair.atoms_b)
    if at_a is None or at_b is None:
        return None
    fr_a = attachment_frame(mols[a], fields[a], *at_a)
    fr_b = attachment_frame(mols[b], fields[b], *at_b)
    if fr_a is None or fr_b is None:
        return None
    one = posed(fields[a], pair.atoms_a, fr_a[0], fr_a[1], name=pair.groups[0])
    two = posed(fields[b], pair.atoms_b, fr_b[0], fr_b[1], name=pair.groups[1])
    d = difference_field(two, one, name=f"{pair.groups[0]}->{pair.groups[1]}")
    return Transformation(series=series, key=(pair.groups[0], pair.groups[1]), field=d,
                          delta=float(pair.delta), names=pair.names)


def reversed_of(t: Transformation) -> Transformation:
    """The same pair read the other way round: the field and the change both change sign."""
    f = t.field
    flipped = MoleculeFields(
        name=f"{t.key[1]}->{t.key[0]}", coords=np.array(f.coords, float),
        weights=-np.asarray(f.weights, float), radii=np.array(f.radii, float),
        symbols=list(f.symbols), atom_types=list(f.atom_types),
        bonds=(np.array(f.bonds, int) if f.bonds is not None else None),
        charges=(-np.asarray(f.charges, float) if f.charges is not None else None),
        charge_source=f.charge_source)
    return Transformation(series=t.series, key=(t.key[1], t.key[0]), field=flipped,
                          delta=-t.delta, names=(t.names[1], t.names[0]))


def both_ways(items: Sequence[Transformation]) -> List[Transformation]:
    """Every pair in both directions, which is what makes a model of them antisymmetric."""
    out: List[Transformation] = []
    for t in items:
        out.append(t)
        out.append(reversed_of(t))
    return out


def average_field(items: Sequence[Transformation], name: str = "") -> MoleculeFields:
    """The mean difference field of several pairs of the same transformation.

    A field is a sum of Gaussians, so the mean of several is again a field: its atoms are the
    atoms of all of them with the weights divided by how many there are.  That is the object plan
    item MC6 calls the transformation field, and it can be drawn like any other.
    """
    if not items:
        raise ValueError("no pairs to average")
    w = 1.0 / float(len(items))
    fields = [t.field for t in items]
    coords = np.vstack([np.asarray(f.coords, float) for f in fields])
    weights = np.vstack([w * np.asarray(f.weights, float) for f in fields])
    radii = np.concatenate([np.asarray(f.radii, float) for f in fields])
    charges = np.concatenate([(w * np.asarray(f.charges, float)) if f.charges is not None
                              else np.zeros(f.n_atoms) for f in fields])
    return MoleculeFields(
        name=name or f"{items[0].key[0]}->{items[0].key[1]}",
        coords=coords, weights=weights, radii=radii,
        symbols=[s for f in fields for s in f.symbols],
        atom_types=[t for f in fields for t in f.atom_types],
        bonds=np.zeros((0, 2), int), charges=charges,
        charge_source=fields[0].charge_source)


def by_transformation(items: Sequence[Transformation]) -> Dict[Tuple[str, str],
                                                               List[Transformation]]:
    """The pairs grouped by the transformation they are, in the direction they were read."""
    out: Dict[Tuple[str, str], List[Transformation]] = defaultdict(list)
    for t in items:
        out[t.key].append(t)
    return dict(out)


def shared(items: Sequence[Transformation], least: int = 2) -> Dict[Tuple[str, str], List[str]]:
    """The transformations seen in at least `least` different series, and in which."""
    seen: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for t in items:
        if t.series not in seen[t.key]:
            seen[t.key].append(t.series)
    return {k: v for k, v in seen.items() if len(v) >= int(least)}


# --------------------------------------------------------------------------- #
# The reference of plan item D5: the same transformation, as a number
# --------------------------------------------------------------------------- #


def increment_table(items: Sequence[Transformation]) -> Dict[Tuple[str, str], float]:
    """The mean measured change of every transformation, which is what D5 fitted and transferred.

    This is the reference the field has to beat: the transformation as a number, with no geometry
    in it at all.
    """
    sums: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for t in items:
        sums[t.key].append(float(t.delta))
    return {k: float(np.mean(v)) for k, v in sums.items()}


def increment_predict(table: Dict[Tuple[str, str], float],
                      items: Sequence[Transformation]) -> Tuple[np.ndarray, np.ndarray]:
    """(the predicted change, whether the transformation was seen) for every pair."""
    pred = np.zeros(len(items))
    seen = np.zeros(len(items), bool)
    for i, t in enumerate(items):
        if t.key in table:
            pred[i] = table[t.key]
            seen[i] = True
        elif (t.key[1], t.key[0]) in table:
            pred[i] = -table[(t.key[1], t.key[0])]
            seen[i] = True
    return pred, seen
