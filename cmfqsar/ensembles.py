"""Ensembles as the normal input: several structures that make one sample (plan item MS5).

A CMF field is a sum of atom-centred Gaussian functions and the kernel is bilinear in the
fields, so the weighted mean field of a set of structures is itself a field, and the kernel of
two such mean fields is the weighted mean of the kernels of their members (equation 10.1 of the
manual, the mean-map kernel). The program used this in four places, each with its own code: the
conformations of one molecule, protonation states, tautomers, and the solvation shells of item
BT5. This module is the one entry for all of them, and for anything else that is a set of
structures with weights: frames of a trajectory, sampled clusters, the members of a mixture.

The members of one sample may have different atoms. RDKit keeps a molecule's conformations on
one atom list, so such members cannot be conformations of one molecule. They are packed side by
side into one RDKit molecule instead, and three molecule properties say where each member
starts, what weight it has and, when known, which atom of the first member each of its atoms
corresponds to. :func:`cmfqsar.fields.build_fields` recognises a packed molecule. With
``conformers="ensemble"`` in the field set it builds every member on its own and joins them
with :func:`cmfqsar.fields.combine_members`; with the default ``"single"`` it uses the first
member, so a single structure is the special case of an ensemble with one member.

Members that do share an atom list are kept in the older form, as conformations of one
molecule, so the two forms give the same field for the same input.

The atom map is needed only to sum a per-atom quantity (an atom contribution, a Shapley value)
onto the atoms of the first member. The kernel, the prediction and the co-field in space do not
use it. Without a map, the atoms of a member that has a different atom list are not placed, and
:meth:`cmfqsar.fields.MoleculeFields.unmapped_share` says how much of a per-atom quantity that
loses.
"""

from typing import List, Optional, Sequence

import numpy as np
from rdkit import Chem

#: molecule property of a packed ensemble: the number of atoms of every member, in order
SIZES_PROP = "cmf_member_sizes"
#: molecule property of a packed ensemble: the weight of every member, as given
WEIGHTS_PROP = "cmf_member_weights"
#: molecule property of a packed ensemble: the atom map of all members, joined
MAP_PROP = "cmf_member_atom_map"
#: record property an input file may carry: the atom of the first member that each atom of this
#: record corresponds to, as whitespace-separated integers, -1 where there is none
RECORD_MAP_PROP = "cmf_atom_map"
#: the molecule property the field set reads given conformer weights from (``weight_prop``)
CONFORMER_WEIGHTS_PROP = "conformer_weights"

_OWN = (SIZES_PROP, WEIGHTS_PROP, MAP_PROP)


class EnsembleError(ValueError):
    """An ensemble that cannot be built or read as it stands."""


def same_atoms(a: Chem.Mol, b: Chem.Mol) -> bool:
    """True when two structures have the same atoms in the same order (count and elements)."""
    if a.GetNumAtoms() != b.GetNumAtoms():
        return False
    return all(x.GetAtomicNum() == y.GetAtomicNum() for x, y in zip(a.GetAtoms(), b.GetAtoms()))


def check_weights(weights: Optional[Sequence[float]], n: int) -> Optional[np.ndarray]:
    """The weights as an array, checked: one per member, finite, not negative, not all zero."""
    if weights is None:
        return None
    w = np.asarray([float(v) for v in weights], float)
    if len(w) != n:
        raise EnsembleError(f"{n} members and {len(w)} weights")
    if not np.isfinite(w).all() or w.min() < 0 or w.sum() <= 0:
        raise EnsembleError("the member weights have to be finite, not negative, and not all "
                            "zero")
    return w


def _numbers(values: Sequence[float]) -> str:
    return " ".join(repr(float(v)) for v in values)


def ensemble_molecule(members: Sequence[Chem.Mol], weights: Optional[Sequence[float]] = None,
                      atom_maps: Optional[Sequence[Optional[Sequence[int]]]] = None,
                      name: Optional[str] = None) -> Chem.Mol:
    """One RDKit molecule that stands for a weighted set of structures.

    Every member must hold exactly one conformation; the coordinates are used as they are, so
    the members must already be in the frame the model compares them in. `weights` need not be
    normalised; without them the members count equally, unless the field set's weighting says
    otherwise. `atom_maps` gives, for each member, the atom of the first member each of its
    atoms corresponds to (-1 for none); ``None`` for a member means "not known".

    Members that all have the same atoms become the conformations of one molecule, with the
    weights in the ``conformer_weights`` property, which is the form the field set has always
    read. Otherwise the members are packed side by side (see the module text). The molecule
    properties of the first member are kept, so an activity or a name travels with the sample.
    """
    mols = [m for m in members]
    if not mols:
        raise EnsembleError("an ensemble needs at least one member")
    for i, m in enumerate(mols):
        if m is None:
            raise EnsembleError(f"member {i} is not a molecule")
        if m.GetNumConformers() != 1:
            raise EnsembleError(f"member {i} holds {m.GetNumConformers()} conformations; an "
                                f"ensemble member is one structure, so give each conformation "
                                f"as its own member")
    w = check_weights(weights, len(mols))
    if atom_maps is not None and len(atom_maps) != len(mols):
        raise EnsembleError(f"{len(mols)} members and {len(atom_maps)} atom maps")
    first = mols[0]
    if all(same_atoms(first, m) for m in mols[1:]):
        out = Chem.Mol(first)
        for m in mols[1:]:
            out.AddConformer(Chem.Conformer(m.GetConformer()), assignId=True)
        if w is not None and len(mols) > 1:
            out.SetProp(CONFORMER_WEIGHTS_PROP, ",".join(repr(float(v)) for v in w))
        if name:
            out.SetProp("_Name", str(name))
        return out
    packed = Chem.Mol(first)
    for m in mols[1:]:
        packed = Chem.CombineMols(packed, m)
    for prop in first.GetPropNames():
        if prop.startswith("atom.") or prop in _OWN:
            continue
        packed.SetProp(prop, first.GetProp(prop))
    packed.SetProp("_Name", str(name) if name else
                   (first.GetProp("_Name") if first.HasProp("_Name") else ""))
    packed.SetProp(SIZES_PROP, " ".join(str(m.GetNumAtoms()) for m in mols))
    if w is not None:
        packed.SetProp(WEIGHTS_PROP, _numbers(w))
    if atom_maps is not None and all(mp is not None for mp in atom_maps):
        joined: List[int] = []
        for i, (m, mp) in enumerate(zip(mols, atom_maps)):
            mp = [int(v) for v in mp]
            if len(mp) != m.GetNumAtoms():
                raise EnsembleError(f"the atom map of member {i} has {len(mp)} entries for "
                                    f"{m.GetNumAtoms()} atoms")
            if any(v < -1 or v >= first.GetNumAtoms() for v in mp):
                raise EnsembleError(f"the atom map of member {i} points outside the "
                                    f"{first.GetNumAtoms()} atoms of the first member")
            joined.extend(mp)
        packed.SetProp(MAP_PROP, " ".join(str(v) for v in joined))
    return packed


def is_packed(mol: Chem.Mol) -> bool:
    """True for a molecule that :func:`ensemble_molecule` packed from members with own atoms."""
    return mol is not None and mol.HasProp(SIZES_PROP)


def member_sizes(mol: Chem.Mol) -> List[int]:
    """The number of atoms of every member of a packed ensemble."""
    sizes = [int(v) for v in mol.GetProp(SIZES_PROP).split()]
    if not sizes or min(sizes) < 1 or sum(sizes) != mol.GetNumAtoms():
        raise EnsembleError(f"the packed ensemble says {sum(sizes)} atoms in {len(sizes)} "
                            f"members and holds {mol.GetNumAtoms()}")
    return sizes


def n_members(mol: Chem.Mol) -> int:
    """How many structures a molecule stands for: members, conformations, or one."""
    if is_packed(mol):
        return len(member_sizes(mol))
    return max(mol.GetNumConformers(), 1)


def members_of(mol: Chem.Mol) -> List[Chem.Mol]:
    """The members of a packed ensemble, each a molecule of its own, in order.

    Each member keeps its atoms in their order, their coordinates and their atom properties, and
    it carries the molecule properties of the sample (an activity, a descriptor), so that
    everything :func:`cmfqsar.fields.build_fields` reads from a molecule is there.
    """
    sizes = member_sizes(mol)
    out = []
    start = 0
    total = mol.GetNumAtoms()
    for n in sizes:
        rw = Chem.RWMol(mol)
        for idx in range(total - 1, -1, -1):
            if not start <= idx < start + n:
                rw.RemoveAtom(idx)
        part = rw.GetMol()
        for prop in _OWN:
            if part.HasProp(prop):
                part.ClearProp(prop)
        out.append(part)
        start += n
    return out


def stored_weights(mol: Chem.Mol) -> Optional[np.ndarray]:
    """The member weights a packed ensemble carries, or ``None`` when it carries none."""
    if not mol.HasProp(WEIGHTS_PROP):
        return None
    return check_weights(mol.GetProp(WEIGHTS_PROP).split(), len(member_sizes(mol)))


def stored_atom_maps(mol: Chem.Mol) -> Optional[List[np.ndarray]]:
    """The atom map of every member of a packed ensemble, or ``None`` when it carries none."""
    if not mol.HasProp(MAP_PROP):
        return None
    joined = np.array([int(v) for v in mol.GetProp(MAP_PROP).split()], int)
    sizes = member_sizes(mol)
    if len(joined) != sum(sizes):
        raise EnsembleError(f"the atom map has {len(joined)} entries for {sum(sizes)} atoms")
    cuts = np.cumsum([0] + sizes)
    return [joined[cuts[i]:cuts[i + 1]] for i in range(len(sizes))]


def member_weights(mol: Chem.Mol, fieldset, normalise: bool = True) -> np.ndarray:
    """The weights of the members of a packed ensemble under a field set, summing to one.

    ``uniform`` counts the members equally; ``given`` uses the weights the ensemble carries and
    refuses an ensemble that carries none. A Boltzmann weighting needs an energy per member,
    which a packed ensemble does not carry, so it is refused instead of being replaced by
    another weighting without a word. ``max_conformers`` keeps the members with the largest
    weights. With `normalise` off the weights are returned as given, which is what
    :func:`cmfqsar.fields.combine_members` takes: it divides by the sum itself, and a second
    division would change the last bit of every weight.
    """
    k = len(member_sizes(mol))
    how = getattr(fieldset, "conformer_weighting", "uniform")
    if how == "uniform":
        w = np.ones(k)
    elif how == "given":
        w = stored_weights(mol)
        if w is None:
            raise EnsembleError("the weighting is 'given' and this ensemble carries no member "
                                "weights; read them with member_weight (--member-weight)")
    else:
        raise EnsembleError(f"the weighting {how!r} needs an energy per member, and members "
                            f"with different atoms do not carry one; give the weights "
                            f"themselves with member_weight (--member-weight)")
    keep = int(getattr(fieldset, "max_conformers", 0) or 0)
    if 0 < keep < k:
        order = np.argsort(-w, kind="stable")[:keep]
        cut = np.zeros(k)
        cut[order] = w[order]
        w = cut
    return w / w.sum() if normalise else w


def _maps_for(members: Sequence[Chem.Mol], kept: Sequence[int],
              stored: Optional[List[np.ndarray]]) -> List[List[int]]:
    """Atom maps onto the first kept member: stored ones where they apply, else by identity."""
    ref = kept[0]
    out = []
    for i in kept:
        m = members[i]
        if stored is not None and ref == 0:
            out.append([int(v) for v in stored[i]])
        elif i == ref or same_atoms(members[ref], m):
            out.append(list(range(m.GetNumAtoms())))
        else:
            out.append([-1] * m.GetNumAtoms())
    return out


def packed_fields(mol: Chem.Mol, fieldset, name: str = ""):
    """The field of a packed ensemble under a field set (called by ``build_fields``)."""
    from .fields import build_fields, combine_members

    if (getattr(fieldset, "protonation_ph", None) is not None
            or getattr(fieldset, "tautomers", "")):
        raise EnsembleError("an ensemble of structures and a protonation or tautomer ensemble "
                            "cannot be combined: the product of the two sets of members is not "
                            "built")
    members = members_of(mol)
    if getattr(fieldset, "conformers", "single") != "ensemble":
        return build_fields(members[0], fieldset, name=name, _single=True)
    w = member_weights(mol, fieldset, normalise=False)
    kept = [i for i in range(len(members)) if w[i] > 0]
    fields = [build_fields(members[i], fieldset, name=f"{name}#{i}", _single=True)
              for i in kept]
    maps = _maps_for(members, kept, stored_atom_maps(mol))
    out = combine_members(fields, [w[i] for i in kept], kind="members", atom_maps=maps,
                          name=name)
    if out.descriptors is None:
        out.descriptors = fields[0].descriptors
    return out


def write_ensembles(path: str, samples: Sequence[Sequence[Chem.Mol]],
                    names: Sequence[str], weights: Optional[Sequence[Sequence[float]]] = None,
                    sample_props: Optional[Sequence[dict]] = None,
                    group_prop: str = "sample", weight_prop: str = "member_weight") -> int:
    """Write samples of several structures to one SD file that ``load_dataset`` reads back.

    Every member becomes one record. It carries the sample's name in `group_prop`, its weight in
    `weight_prop` and the sample's own properties (`sample_props`, an activity for example), so
    that ``load_dataset(path, ensemble_by=group_prop, member_weight=weight_prop)`` gives back one
    entry per sample. Returns the number of records written.
    """
    if len(samples) != len(names):
        raise EnsembleError(f"{len(samples)} samples and {len(names)} names")
    written = 0
    writer = Chem.SDWriter(path)
    try:
        for s, (members, nm) in enumerate(zip(samples, names)):
            w = None if weights is None else check_weights(weights[s], len(members))
            extra = {} if sample_props is None else dict(sample_props[s])
            for i, m in enumerate(members):
                rec = Chem.Mol(m)
                rec.SetProp("_Name", str(nm))
                rec.SetProp(group_prop, str(nm))
                if w is not None:
                    rec.SetProp(weight_prop, repr(float(w[i])))
                for key, value in extra.items():
                    rec.SetProp(str(key), str(value))
                writer.write(rec)
                written += 1
    finally:
        writer.close()
    return written
