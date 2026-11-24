"""The continuous reaction field: a reaction as one set of fields (plan item PO3).

A condensed graph of reaction (CGR) is one graph that holds the reactants and the products of a
reaction together.  An atom or a bond that does not change keeps a conventional label, and one
that changes gets a dynamic label: bond made, bond broken, bond order changed, charge changed.
The continuous reaction field (CRF) is the same idea in space.  The fields of the two sides are
put into one coordinate frame, and two kinds of channels are formed from them, for every field f:

* the **conventional channel**, the mean of the two sides, C_f = (P_f + R_f) / 2;
* the **dynamic channel**, the difference, D_f = P_f - R_f.

The fields of this program are linear in the atoms, so both channels are ordinary
:class:`~cmfqsar.fields.MoleculeFields`: the atoms of both sides, with the weights of the
reactant halved (C) or negated (D).  An atom that does not change and does not move cancels
exactly in D.  The dynamic channel of the electrostatic field is the **charge-change field** of a
CGR, so it needs no object of its own.  The **bond-change field** has no molecular counterpart:
one Gaussian at the midpoint of every bond that changes, weighted by the change of bond order
(positive for a bond made, negative for a bond broken).

What a reaction needs
---------------------
An atom-to-atom mapping and one 3D geometry of each side.  :func:`read_reaction` takes the two
sides as mapped SMILES with every hydrogen written and the coordinates in the order of the map
numbers.  The two sides must hold the same atoms; a reaction that does not is refused with
:class:`ReactionMismatch`, because atoms that are missing on one side do not cancel.

Two frames, and only one of them is solved here
-----------------------------------------------
*Inside one reaction* the product is superposed on the reactant, on all heavy atoms or on the
atoms of the reaction centre (:func:`superpose`).  *Between two reactions* there is no shared
frame, and the overlap kernel compares fields point by point.  The overlap kernels of
:func:`overlap_kernels` are therefore exact statements about the coordinates they are given and
nothing more; they are the right kernel for a series of reactions that were placed in one frame
by some other rule.  For reactions with no such rule the description is the rotation-invariant
autocorrelation of :func:`invariant_rows`.

Two identities
--------------
(1) The pair (C, D) is a rotation of the pair (R, P).  For any bilinear kernel k,

    k(R, R') + k(P, P') = 2 k(C, C') + k(D, D') / 2,

so with these weights the channels say exactly what the two sides say as separate blocks.  A gain
can come only from weighting C and D separately.

(2) Reversing a reaction leaves C unchanged and negates D.  The overlap kernel of two dynamic
channels therefore changes sign when one reaction is reversed, and a ridge without intercept on
it predicts the opposite value for the reverse reaction, which is right for a reaction energy.

The autocorrelation is quadratic in a field, so the autocorrelation of D does **not** change sign
when the reaction is reversed.  The part of the invariant description that is odd under reversal
is the cross-correlation of C with D, and it equals half the difference of the autocorrelations
of the two sides:

    C x D = (P x P - R x R) / 2.

It does not depend on the superposition at all.  The superposition enters the invariant
description only through the cross-correlation of the two sides, R x P, which is inside C x C
and D x D.  :func:`invariant_rows` returns the even rows (C x C, D x D) and the odd row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from .fields import AtomTable, FieldSet, FieldSpec, MoleculeFields, build_fields
from .invariant import AutocorrelationSpec, field_pairs
from .kernels import kernel_matrices

#: the two rules for the superposition of the product on the reactant
#: "fragments" is the rule for a side of several molecules that have no position relative to
#: each other: see :func:`superpose_fragments`
SUPERPOSE_ON = ("heavy", "centre")
FRAME_RULES = SUPERPOSE_ON + ("fragments",)
#: attenuation of the bond-change field, in A^-2
BOND_ALPHA = 1.0
#: the spec of the bond-change field: one Gaussian at the midpoint of each changed bond
BOND_SPEC = FieldSpec(name="bondchange", kind="gauss", alpha=BOND_ALPHA)
#: a bond of the graph longer than this many times the sum of the covalent radii is a sign
#: that the coordinates are not in the order of the map numbers
STRETCH_LIMIT = 1.6
#: distance grid of the invariant description: the molecules of a gas-phase reaction set are
#: small, so the grid is shorter and finer than the default for a drug-sized molecule
REACTION_SPEC = AutocorrelationSpec(d_max=8.0, step=0.5, sigma=0.5)


class ReactionMismatch(ValueError):
    """The two sides of a reaction do not hold the same atoms, or a side cannot be read."""


# --------------------------------------------------------------------------- #
# Reading a reaction
# --------------------------------------------------------------------------- #


def mapped_mol(smiles: str, coords: Optional[np.ndarray] = None) -> Chem.Mol:
    """A molecule from a mapped SMILES, with its atoms in the order of the map numbers.

    Every atom must carry a map number, the numbers must be 1..N without gaps, and every hydrogen
    must be written.  Several molecules separated by a dot are read as one RDKit molecule.
    `coords`, when given, are (N, 3) in the order of the map numbers and become the conformer.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ReactionMismatch(f"RDKit cannot read the SMILES {smiles!r}")
    numbers = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    n = len(numbers)
    if sorted(numbers) != list(range(1, n + 1)):
        raise ReactionMismatch(f"the map numbers of {smiles!r} are not 1..{n} without gaps; "
                               f"every atom, also every hydrogen, needs one")
    order = [int(i) for i in np.argsort(numbers)]
    mol = Chem.RenumberAtoms(mol, order)
    if coords is not None:
        xyz = np.asarray(coords, float)
        if xyz.shape != (n, 3):
            raise ReactionMismatch(f"{n} atoms in {smiles!r} against coordinates of shape "
                                   f"{xyz.shape}")
        _set_coords(mol, xyz)
    return mol


def _set_coords(mol: Chem.Mol, xyz: np.ndarray) -> None:
    mol.RemoveAllConformers()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(np.asarray(xyz, float)):
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
    conf.Set3D(True)
    mol.AddConformer(conf, assignId=True)


def _coords(mol: Chem.Mol) -> np.ndarray:
    return np.array(mol.GetConformer().GetPositions(), float)


@dataclass
class Reaction:
    """The two sides of one reaction: the same atoms in the same order, one geometry each."""
    name: str
    reactant: Chem.Mol
    product: Chem.Mol

    @property
    def n_atoms(self) -> int:
        return self.reactant.GetNumAtoms()

    @property
    def symbols(self) -> List[str]:
        return [a.GetSymbol() for a in self.reactant.GetAtoms()]


def read_reaction(reactant_smiles: str, product_smiles: str, reactant_coords: np.ndarray,
                  product_coords: np.ndarray, name: str = "") -> Reaction:
    """One reaction from two mapped SMILES and two geometries in the order of the map numbers.

    Raises :class:`ReactionMismatch` when a side cannot be read or when the two sides do not hold
    the same element under every map number.
    """
    r = mapped_mol(reactant_smiles, reactant_coords)
    p = mapped_mol(product_smiles, product_coords)
    if r.GetNumAtoms() != p.GetNumAtoms():
        raise ReactionMismatch(f"{name or 'the reaction'}: {r.GetNumAtoms()} atoms in the "
                               f"reactant against {p.GetNumAtoms()} in the product")
    for a, b in zip(r.GetAtoms(), p.GetAtoms()):
        if a.GetAtomicNum() != b.GetAtomicNum():
            raise ReactionMismatch(
                f"{name or 'the reaction'}: map number {a.GetIdx() + 1} is {a.GetSymbol()} in "
                f"the reactant and {b.GetSymbol()} in the product")
    return Reaction(name=name, reactant=r, product=p)


def bond_stretch(mol: Chem.Mol) -> float:
    """The longest bond of the graph, as a multiple of the sum of the two covalent radii.

    A value above :data:`STRETCH_LIMIT` says that the coordinates do not belong to the graph,
    which is what happens when their order is not the order of the map numbers.  A molecule
    without bonds gives 0.
    """
    xyz = _coords(mol)
    table = Chem.GetPeriodicTable()
    worst = 0.0
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ideal = (table.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum())
                 + table.GetRcovalent(mol.GetAtomWithIdx(j).GetAtomicNum()))
        worst = max(worst, float(np.linalg.norm(xyz[i] - xyz[j])) / ideal)
    return worst


def reversed_reaction(rx: Reaction) -> Reaction:
    """The same reaction read from the product to the reactant."""
    return Reaction(name=(rx.name + " reversed") if rx.name else "reversed",
                    reactant=Chem.Mol(rx.product), product=Chem.Mol(rx.reactant))


# --------------------------------------------------------------------------- #
# The bonds that change, and the reaction centre
# --------------------------------------------------------------------------- #


def bond_orders(mol: Chem.Mol) -> Dict[Tuple[int, int], float]:
    """Bond order of every bond, keyed by (lower atom index, higher); an aromatic bond is 1.5."""
    out = {}
    for b in mol.GetBonds():
        i, j = sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        out[(i, j)] = float(b.GetBondTypeAsDouble())
    return out


def bond_changes(rx: Reaction) -> List[Tuple[int, int, float]]:
    """(atom, atom, change of bond order) for every bond that differs between the two sides.

    The change is the order in the product minus the order in the reactant: positive for a bond
    that is made or raised, negative for one that is broken or lowered.
    """
    r, p = bond_orders(rx.reactant), bond_orders(rx.product)
    out = []
    for key in sorted(set(r) | set(p)):
        d = p.get(key, 0.0) - r.get(key, 0.0)
        if abs(d) > 1e-9:
            out.append((key[0], key[1], d))
    return out


def reaction_centre(rx: Reaction) -> List[int]:
    """The atoms of every bond that changes, sorted; every atom when no bond changes."""
    atoms = sorted({a for i, j, _d in bond_changes(rx) for a in (i, j)})
    return atoms if atoms else list(range(rx.n_atoms))


def change_kind(order_r: float, order_p: float) -> str:
    """"made", "broken", "raised" or "lowered" for one bond that changes."""
    if order_r == 0.0:
        return "made"
    if order_p == 0.0:
        return "broken"
    return "raised" if order_p > order_r else "lowered"


CHANGE_KINDS = ("made", "broken", "raised", "lowered")


def bond_change_counts(reactions: Sequence[Reaction]) -> Tuple[np.ndarray, List[str]]:
    """The count reference: how many bonds of each element pair are made, broken, raised, lowered.

    Returns (counts, labels): one row per reaction and one column per (element pair, kind) that
    occurs in the set, for example ``C-H broken``.  This is the count version of what the
    dynamic channels may be measuring, to be put beside them in every comparison.
    """
    rows: List[Dict[str, int]] = []
    for rx in reactions:
        sym = rx.symbols
        r, p = bond_orders(rx.reactant), bond_orders(rx.product)
        row: Dict[str, int] = {}
        for i, j, _d in bond_changes(rx):
            pair = "-".join(sorted((sym[i], sym[j])))
            key = f"{pair} {change_kind(r.get((i, j), 0.0), p.get((i, j), 0.0))}"
            row[key] = row.get(key, 0) + 1
        rows.append(row)
    labels = sorted({k for row in rows for k in row})
    out = np.zeros((len(rows), len(labels)))
    for n, row in enumerate(rows):
        for k, v in row.items():
            out[n, labels.index(k)] = v
    return out, labels


# --------------------------------------------------------------------------- #
# The frame of one reaction: the product superposed on the reactant
# --------------------------------------------------------------------------- #


def _kabsch(moving: np.ndarray, fixed: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, centre of moving, centre of fixed) such that (x - cm) @ R.T + cf is the best fit."""
    cm, cf = moving.mean(axis=0), fixed.mean(axis=0)
    H = (moving - cm).T @ (fixed - cf)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d if d != 0 else 1.0]) @ U.T
    return R, cm, cf


def superposition_atoms(rx: Reaction, on: str = "heavy") -> List[int]:
    """The atoms the product is fitted on.

    ``"heavy"``: every atom that is not hydrogen.  ``"centre"``: the atoms of the bonds that
    change; where these are fewer than three, the atoms bonded to them on either side are added.
    Where a rule still gives fewer than three atoms, every atom is used.
    """
    if on not in SUPERPOSE_ON:
        raise ValueError(f"the superposition is on one of {SUPERPOSE_ON}, not {on!r}")
    if on == "heavy":
        atoms = [a.GetIdx() for a in rx.reactant.GetAtoms() if a.GetAtomicNum() > 1]
    else:
        atoms = reaction_centre(rx)
        if len(atoms) < 3:
            grown = set(atoms)
            for mol in (rx.reactant, rx.product):
                for a in atoms:
                    grown.update(n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors())
            atoms = sorted(grown)
    return atoms if len(atoms) >= 3 else list(range(rx.n_atoms))


def superpose(rx: Reaction, on: str = "heavy") -> Tuple[np.ndarray, float]:
    """(coordinates of the product in the frame of the reactant, RMSD over the fitted atoms).

    The reactant is not moved.  The fit is a rigid rotation and translation without a
    reflection, so the handedness of the product is kept.
    """
    atoms = superposition_atoms(rx, on)
    r, p = _coords(rx.reactant), _coords(rx.product)
    if len(atoms) < 2:
        moved = p - p.mean(axis=0) + r.mean(axis=0)
    else:
        R, cm, cf = _kabsch(p[atoms], r[atoms])
        moved = (p - cm) @ R.T + cf
    rmsd = float(np.sqrt(np.mean(np.sum((moved[atoms] - r[atoms]) ** 2, axis=1))))
    return moved, rmsd


def _fit_atoms(mol: Chem.Mol, fragment: Sequence[int]) -> List[int]:
    """The atoms of a fragment a fit uses: its heavy atoms, or all of it when these are fewer
    than three."""
    heavy = [i for i in fragment if mol.GetAtomWithIdx(int(i)).GetAtomicNum() > 1]
    return heavy if len(heavy) >= 3 else [int(i) for i in fragment]


def _place(moving: np.ndarray, fixed: np.ndarray, fragment: Sequence[int],
           fit: Sequence[int]) -> np.ndarray:
    """The coordinates of one fragment of `moving`, fitted on the same atoms of `fixed`."""
    frag, fit = [int(i) for i in fragment], [int(i) for i in fit]
    if len(fit) == 1:
        return moving[frag] - moving[fit[0]] + fixed[fit[0]]
    R, cm, cf = _kabsch(moving[fit], fixed[fit])
    return (moving[frag] - cm) @ R.T + cf


def superpose_fragments(rx: Reaction) -> Tuple[np.ndarray, np.ndarray, float]:
    """(reactant coordinates, product coordinates, RMSD) with every molecule placed by itself.

    The molecules of a side that were optimised separately have no position relative to each
    other, so a rigid fit of the whole side means nothing.  Here the side with fewer molecules
    stays where it is, and each molecule of the other side is fitted on it by itself, on its
    heavy atoms (on all its atoms when it has fewer than three heavy atoms).  With the same
    number of molecules on both sides the reactant stays and the product moves, as under the
    other rules.  The RMSD is over the fitted atoms of all molecules.
    """
    r, p = _coords(rx.reactant), _coords(rx.product)
    frags_r = Chem.GetMolFrags(rx.reactant)
    frags_p = Chem.GetMolFrags(rx.product)
    move_reactant = len(frags_r) > len(frags_p)
    moving, fixed = (r, p) if move_reactant else (p, r)
    mol, frags = (rx.reactant, frags_r) if move_reactant else (rx.product, frags_p)
    out = moving.copy()
    used = []
    for frag in frags:
        fit = _fit_atoms(mol, frag)
        out[list(frag)] = _place(moving, fixed, frag, fit)
        used.extend(fit)
    rmsd = float(np.sqrt(np.mean(np.sum((out[used] - fixed[used]) ** 2, axis=1))))
    return (out, p, rmsd) if move_reactant else (r, out, rmsd)


def frame(rx: Reaction, on: str = "heavy") -> Tuple[np.ndarray, np.ndarray, float]:
    """(reactant coordinates, product coordinates, RMSD) of the two sides in one frame.

    ``"heavy"`` and ``"centre"`` move the product as one body (:func:`superpose`);
    ``"fragments"`` places every molecule by itself (:func:`superpose_fragments`).
    """
    if on == "fragments":
        return superpose_fragments(rx)
    moved, rmsd = superpose(rx, on)
    return _coords(rx.reactant), moved, rmsd


# --------------------------------------------------------------------------- #
# A reaction whose hydrogens carry no map numbers
# --------------------------------------------------------------------------- #


def _skeleton(mol: Chem.Mol) -> Chem.Mol:
    """The molecule as a plain graph: every bond single, no charges, no map numbers."""
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        a.SetFormalCharge(0)
        a.SetAtomMapNum(0)
        a.SetIsAromatic(False)
        a.SetNoImplicit(True)
        a.SetNumExplicitHs(0)
        a.SetNumRadicalElectrons(0)
        a.SetIsotope(0)
    for b in rw.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetIsAromatic(False)
    out = rw.GetMol()
    out.UpdatePropertyCache(strict=False)
    Chem.FastFindRings(out)
    return out


def mol_from_geometry(smiles: str, symbols: Sequence[str], coords: np.ndarray) -> Chem.Mol:
    """One molecule from a SMILES and a geometry whose atoms are in another order.

    The SMILES may carry map numbers on its heavy atoms and its hydrogens implicit.  The bonds
    of the geometry are perceived from the distances, and the atoms of the SMILES are matched to
    it as a graph, without bond orders and charges.  The result has the atoms of the SMILES with
    its hydrogens added, and the coordinates of the geometry.  Raises :class:`ReactionMismatch`
    when the geometry is not the molecule of the SMILES.
    """
    from rdkit.Chem import rdDetermineBonds
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ReactionMismatch(f"RDKit cannot read the SMILES {smiles!r}")
    mol = Chem.AddHs(mol)
    xyz = np.asarray(coords, float)
    if mol.GetNumAtoms() != len(symbols):
        raise ReactionMismatch(f"{mol.GetNumAtoms()} atoms in {smiles!r} against {len(symbols)} "
                               f"in the geometry")
    raw = Chem.RWMol()
    for s in symbols:
        raw.AddAtom(Chem.Atom(s))
    conf = Chem.Conformer(len(symbols))
    for i, q in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(float(q[0]), float(q[1]), float(q[2])))
    raw.AddConformer(conf)
    geom = raw.GetMol()
    try:
        rdDetermineBonds.DetermineConnectivity(geom)
    except Exception as err:
        raise ReactionMismatch(f"the bonds of the geometry cannot be perceived: {err}")
    match = _skeleton(geom).GetSubstructMatch(_skeleton(mol), useChirality=False)
    if len(match) != mol.GetNumAtoms():
        raise ReactionMismatch(f"the geometry is not the molecule {smiles!r}")
    _set_coords(mol, xyz[list(match)])
    return mol


def join_molecules(mols: Sequence[Chem.Mol]) -> Chem.Mol:
    """Several molecules with one conformer each as one RDKit molecule, atoms one after another."""
    out = mols[0]
    for m in mols[1:]:
        out = Chem.CombineMols(out, m)
    out = Chem.Mol(out)
    Chem.SanitizeMol(out)
    return out


def reaction_from_mols(reactant: Chem.Mol, product: Chem.Mol, name: str = "") -> Reaction:
    """A reaction from two sides whose heavy atoms are mapped and whose hydrogens are not.

    Each side is one RDKit molecule (several molecules joined by :func:`join_molecules`) with
    its hydrogens as atoms and one conformer.  The heavy atoms are put in the order of their map
    numbers.  A hydrogen has no map number, so it is paired with a hydrogen on the same heavy
    atom of the other side: the nearest one after every molecule has been placed by
    :func:`superpose_fragments`, with the sum of the distances the smallest possible.  A heavy
    atom that carries another number of hydrogens on the two sides stops the pairing, and the
    reaction is refused with :class:`ReactionMismatch`: a hydrogen moved, and its map is not
    known.  In the result every atom carries a map number, 1..N.
    """
    from scipy.optimize import linear_sum_assignment
    label = name or "the reaction"

    def heavy(mol):
        found = {}
        for a in mol.GetAtoms():
            if a.GetAtomicNum() > 1:
                m = a.GetAtomMapNum()
                if m <= 0 or m in found:
                    raise ReactionMismatch(f"{label}: a heavy atom without a map number, or a "
                                           f"map number used twice")
                found[m] = a.GetIdx()
        return found

    hr, hp = heavy(reactant), heavy(product)
    if sorted(hr) != sorted(hp):
        raise ReactionMismatch(f"{label}: the two sides do not carry the same map numbers")
    maps = sorted(hr)
    for m in maps:
        if (reactant.GetAtomWithIdx(hr[m]).GetAtomicNum()
                != product.GetAtomWithIdx(hp[m]).GetAtomicNum()):
            raise ReactionMismatch(f"{label}: map number {m} is another element on the two sides")

    def hydrogens(mol, idx):
        return [n.GetIdx() for n in mol.GetAtomWithIdx(idx).GetNeighbors()
                if n.GetAtomicNum() == 1]

    for mol in (reactant, product):
        for a in mol.GetAtoms():
            if a.GetAtomicNum() == 1 and not any(n.GetAtomicNum() > 1 for n in a.GetNeighbors()):
                raise ReactionMismatch(f"{label}: a hydrogen that is not bonded to a heavy atom")
    # the heavy atoms in one order, then a first frame from them alone
    order_r, order_p = [hr[m] for m in maps], [hp[m] for m in maps]
    hs_r = {m: hydrogens(reactant, hr[m]) for m in maps}
    hs_p = {m: hydrogens(product, hp[m]) for m in maps}
    for m in maps:
        if len(hs_r[m]) != len(hs_p[m]):
            raise ReactionMismatch(f"{label}: map number {m} carries {len(hs_r[m])} hydrogens "
                                   f"in the reactant and {len(hs_p[m])} in the product")
    first_r = order_r + [h for m in maps for h in hs_r[m]]
    first_p = order_p + [h for m in maps for h in hs_p[m]]
    trial = Reaction(name=name, reactant=Chem.RenumberAtoms(reactant, first_r),
                     product=Chem.RenumberAtoms(product, first_p))
    xr, xp, _rmsd = superpose_fragments(trial)
    # pair the hydrogens of every heavy atom by distance in that frame
    final_r, final_p = list(first_r[:len(maps)]), list(first_p[:len(maps)])
    at = len(maps)
    for m in maps:
        k = len(hs_r[m])
        if k:
            d = np.linalg.norm(xr[at:at + k, None, :] - xp[None, at:at + k, :], axis=-1)
            rows, cols = linear_sum_assignment(d)
            final_r.extend(first_r[at + i] for i in rows)
            final_p.extend(first_p[at + j] for j in cols)
        at += k
    out_r, out_p = Chem.RenumberAtoms(reactant, final_r), Chem.RenumberAtoms(product, final_p)
    for mol in (out_r, out_p):
        for a in mol.GetAtoms():
            a.SetAtomMapNum(a.GetIdx() + 1)
    return Reaction(name=name, reactant=out_r, product=out_p)


def mapped_smiles(mol: Chem.Mol) -> str:
    """The SMILES of one side with every atom, also every hydrogen, written with its map number.

    :func:`mapped_mol` reads it back with the atoms in the same order.
    """
    return Chem.MolToSmiles(mol, allHsExplicit=True)


# --------------------------------------------------------------------------- #
# The channels
# --------------------------------------------------------------------------- #


def _joined(a: MoleculeFields, b: MoleculeFields, wa: float, wb: float,
            name: str) -> MoleculeFields:
    """The field wa * a + wb * b as one field: the atoms of both, with scaled weights."""
    na = a.n_atoms
    bonds_a = np.asarray(a.bonds, int).reshape(-1, 2)
    bonds_b = np.asarray(b.bonds, int).reshape(-1, 2) + na

    def q(m: MoleculeFields) -> np.ndarray:
        c = np.asarray(m.charges, float).reshape(-1) if m.charges is not None else np.zeros(0)
        return c if len(c) == m.n_atoms else np.zeros(m.n_atoms)

    return MoleculeFields(
        name=name,
        coords=np.concatenate([np.asarray(a.coords, float), np.asarray(b.coords, float)]),
        weights=np.concatenate([wa * np.asarray(a.weights, float),
                                wb * np.asarray(b.weights, float)]),
        radii=np.concatenate([np.asarray(a.radii, float), np.asarray(b.radii, float)]),
        symbols=list(a.symbols) + list(b.symbols),
        atom_types=list(a.atom_types) + list(b.atom_types),
        bonds=np.concatenate([bonds_a, bonds_b]),
        charges=np.concatenate([wa * q(a), wb * q(b)]),
        charge_source=a.charge_source)


@dataclass
class ReactionFields:
    """One reaction as fields: the two sides in one frame, and the channels made from them."""
    name: str
    reactant: MoleculeFields
    #: the product, superposed on the reactant
    product: MoleculeFields
    #: (P + R) / 2 for every field
    conventional: MoleculeFields
    #: P - R for every field
    dynamic: MoleculeFields
    #: one pseudo-atom per changed bond, one weight column: the change of bond order
    bonds: MoleculeFields
    #: RMSD of the superposition over the atoms it was fitted on, in A
    rmsd: float = 0.0
    #: the rule the superposition used
    on: str = "heavy"

    def reversed(self) -> "ReactionFields":
        """The reverse reaction **in the same frame**: C is unchanged, D and the bonds change sign.

        This is the reversal that identity (2) of the module text is about.  Building the reverse
        reaction from :func:`reversed_reaction` gives the same fields in the frame of the
        product instead.  The invariant description is the same for both; the overlap kernel is
        not.
        """
        name = (self.name + " reversed") if self.name else "reversed"
        return ReactionFields(
            name=name, reactant=self.product, product=self.reactant,
            conventional=self.conventional,
            dynamic=_joined(self.reactant, self.product, 1.0, -1.0, name),
            bonds=_scaled(self.bonds, -1.0, name), rmsd=self.rmsd, on=self.on)


def _scaled(m: MoleculeFields, w: float, name: str) -> MoleculeFields:
    return MoleculeFields(
        name=name, coords=np.array(m.coords, float), weights=w * np.asarray(m.weights, float),
        radii=np.array(m.radii, float), symbols=list(m.symbols), atom_types=list(m.atom_types),
        bonds=np.asarray(m.bonds, int).reshape(-1, 2), charges=np.zeros(m.n_atoms),
        charge_source=m.charge_source)


def bond_change_field(rx: Reaction, product_coords: np.ndarray, name: str = "",
                      reactant_coords: Optional[np.ndarray] = None) -> MoleculeFields:
    """The bond-change field: one Gaussian at the midpoint of each bond that changes.

    The midpoint of a broken bond is taken in the reactant, that of a made bond in the superposed
    product, and that of a bond whose order changes is the mean of the two.  The single weight
    column is the change of bond order.  A reaction without a changed bond gives one pseudo-atom
    of weight zero at the centroid of the reactant, so that the field exists and overlaps nothing.
    """
    r = _coords(rx.reactant) if reactant_coords is None else np.asarray(reactant_coords, float)
    p = np.asarray(product_coords, float)
    orders_r, orders_p = bond_orders(rx.reactant), bond_orders(rx.product)
    pos, w = [], []
    for i, j, d in bond_changes(rx):
        mid_r, mid_p = 0.5 * (r[i] + r[j]), 0.5 * (p[i] + p[j])
        kind = change_kind(orders_r.get((i, j), 0.0), orders_p.get((i, j), 0.0))
        pos.append(mid_r if kind == "broken" else mid_p if kind == "made"
                   else 0.5 * (mid_r + mid_p))
        w.append(d)
    if not pos:
        pos, w = [r.mean(axis=0)], [0.0]
    n = len(pos)
    return MoleculeFields(
        name=name, coords=np.array(pos, float), weights=np.array(w, float).reshape(n, 1),
        radii=np.ones(n), symbols=["X"] * n, atom_types=["bond"] * n,
        bonds=np.zeros((0, 2), int), charges=np.zeros(n))


def reaction_fields(rx: Reaction, fieldset: FieldSet, on: str = "heavy") -> ReactionFields:
    """The channels of one reaction, with the two sides put into one frame by rule `on`.

    `on` is one of :data:`FRAME_RULES`; see :func:`frame`.
    """
    xr, xp, rmsd = frame(rx, on)
    reactant, product = Chem.Mol(rx.reactant), Chem.Mol(rx.product)
    _set_coords(reactant, xr)
    _set_coords(product, xp)
    R = build_fields(reactant, fieldset, rx.name, _single=True)
    P = build_fields(product, fieldset, rx.name, _single=True)
    return ReactionFields(
        name=rx.name, reactant=R, product=P,
        conventional=_joined(P, R, 0.5, 0.5, rx.name),
        dynamic=_joined(P, R, 1.0, -1.0, rx.name),
        bonds=bond_change_field(rx, xp, rx.name, reactant_coords=xr), rmsd=rmsd, on=on)


# --------------------------------------------------------------------------- #
# Kernels and invariant rows
# --------------------------------------------------------------------------- #

PARTS = ("reactant", "product", "conventional", "dynamic")


def overlap_kernels(items: Sequence[ReactionFields], fieldset: FieldSet,
                    others: Optional[Sequence[ReactionFields]] = None,
                    parts: Sequence[str] = PARTS + ("bonds",)) -> Dict[str, np.ndarray]:
    """Overlap kernels of the named parts, not normalised.

    Every part but ``"bonds"`` gives an array (F, n, m), one kernel per enabled field of the
    field set; ``"bonds"`` gives (n, m).  `others` are the reactions of the columns; without
    them the kernel is of `items` against itself.

    The values are overlap integrals in the coordinates the reactions carry.  They compare two
    reactions point by point, so they mean something only for reactions that share a frame; see
    the module text.
    """
    specs = [s for s in fieldset.specs if s.enabled]
    out: Dict[str, np.ndarray] = {}
    for part in parts:
        use = [BOND_SPEC] if part == "bonds" else specs
        rows = AtomTable.from_molecules([getattr(it, part) for it in items])
        if others is None:
            K = np.asarray(kernel_matrices(rows, rows, use, symmetric=True), float)
        else:
            cols = AtomTable.from_molecules([getattr(it, part) for it in others])
            K = np.asarray(kernel_matrices(rows, cols, use, symmetric=False), float)
        out[part] = K[0] if part == "bonds" else K
    return out


def channel_norms(items: Sequence[ReactionFields], fieldset: FieldSet,
                  part: str = "dynamic") -> np.ndarray:
    """The length of one channel of every reaction, per field: the root of its self-overlap.

    For the dynamic channel this is how much the reaction changes each field.  The
    cosine-normalised kernels divide it out, so it is given here as numbers of its own, to be
    put into the references.  Shape (n, F).
    """
    specs = [s for s in fieldset.specs if s.enabled]
    out = np.zeros((len(items), len(specs)))
    for n, it in enumerate(items):
        t = AtomTable.from_molecules([getattr(it, part)])
        k = np.asarray(kernel_matrices(t, t, specs, symmetric=True), float)[:, 0, 0]
        out[n] = np.sqrt(np.maximum(k, 0.0))
    return out


def invariant_rows(items: Sequence[ReactionFields],
                   spec: AutocorrelationSpec = REACTION_SPEC) -> Dict[str, np.ndarray]:
    """The rotation-invariant description: autocorrelation rows of the parts, not normalised.

    ``reactant``, ``product``, ``conventional`` and ``dynamic`` are the autocorrelations of
    those fields, and ``bonds`` that of the bond-change field.  All of them are even: they do
    not change when the reaction is reversed.  ``odd`` is the cross-correlation of the
    conventional with the dynamic channel, (P x P - R x R) / 2, which changes sign when the
    reaction is reversed and does not depend on the superposition.

    The rows obey identity (1): reactant + product = 2 conventional + dynamic / 2.
    """
    out = {part: autocorrelation_rows([getattr(it, part) for it in items], spec)
           for part in PARTS}
    out["odd"] = 0.5 * (out["product"] - out["reactant"])
    out["bonds"] = autocorrelation_rows([it.bonds for it in items], spec)
    return out


def autocorrelation_rows(fields: Sequence[MoleculeFields],
                         spec: AutocorrelationSpec = REACTION_SPEC) -> np.ndarray:
    """The rows of :func:`cmfqsar.invariant.feature_matrix`, computed for all field pairs at once.

    The library function loops over the pairs of fields; with eighteen fields that is 171
    products per molecule.  This one forms W' B W for every distance channel in one product and
    reads the pairs out of it.  The two agree to rounding, which a test fixes.  It does not
    take probe fields or the signed three-point channels.
    """
    if spec.chirality:
        raise ValueError("the signed three-point channels are not computed here; use "
                         "cmfqsar.invariant.feature_matrix")
    if not len(fields):
        return np.zeros((0, 0))
    n_fields = fields[0].weights.shape[1]
    pairs = field_pairs(n_fields)
    f_idx = np.array([f for f, _g in pairs])
    g_idx = np.array([g for _f, g in pairs])
    twice = np.where(f_idx != g_idx, 2.0, 1.0)[:, None]
    centres = spec.centres
    out = np.zeros((len(fields), len(pairs) * len(centres)))
    for n, m in enumerate(fields):
        W = np.asarray(m.weights, float)
        if W.shape[1] != n_fields:
            raise ValueError(f"{m.name or f'item {n}'} carries {W.shape[1]} fields where the "
                             f"first carries {n_fields}")
        if len(W) == 0:
            continue
        xyz = np.asarray(m.coords, float)
        d = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
        if not spec.include_self:
            np.fill_diagonal(d, np.inf)
        B = np.exp(-((d[:, :, None] - centres[None, None, :]) ** 2) / (2.0 * spec.sigma ** 2))
        M = np.einsum("af,abk,bg->fgk", W, B, W, optimize=True)
        out[n] = (twice * M[f_idx, g_idx]).ravel()
    return out
