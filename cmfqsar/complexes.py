"""
Transition-metal complexes: reading one from coordinates, and the frame its metal gives.

Plan item EC4.  A coordination complex breaks three assumptions the rest of the program makes
about a molecule.  Its metal atom has no entry in the steric table, so the steric field used a
default radius for it; the Gasteiger charge scheme has no parameters for a metal, and one metal
atom made RDKit return "not a number" for *every* atom, so the electrostatic field of a complex
was identically zero; and a metal-ligand bond is not a shared electron pair, so RDKit refuses to
sanitise most complexes written with ordinary single bonds.  This module repairs the third point
and :mod:`cmfqsar.atomtypes` and :mod:`cmfqsar.fields` repair the first two.

What a reader has to decide, and how it is decided here:

* **Which bonds the metal makes.**  They are found by distance (RDKit's own connectivity
  perception) and written as *dative* bonds that point from the donor atom to the metal.  A
  dative bond carries no electrons of the donor's own valence, so the donor keeps the valence it
  has in the free ligand and the molecule sanitises.
* **The bond orders of the ligands.**  The metal is taken out, which leaves the ligands as
  separate fragments, and each fragment's bond orders are determined on its own with
  ``rdDetermineBonds``, the same algorithm the program uses for any set of coordinates.  A
  fragment needs its charge before its bond orders can be found (a cyclopentadienyl ring is
  aromatic at -1 and a radical at 0), so the charges are taken from the data set when it states
  them, and otherwise tried in turn.
* **The charge of the metal.**  It is the total charge of the complex minus the sum of the ligand
  charges.  For a complex whose ligand charges are stated this is the oxidation state of the
  metal, and it is written into the molecule as a formal charge, which is what the repaired
  charge model then places on the metal.
* **The frame.**  The metal is the one atom every member of a series has, so the frame is free in
  the sense of plan item EC4: the metal goes to the origin, and the rotation is fixed either by
  matching the first coordination sphere onto a reference complex (:func:`align_on_first_sphere`)
  or, when the members do not share a coordination sphere, by a rule
  (:func:`principal_metal_frame`).

Nothing here computes a field.  The output is an ordinary RDKit molecule with one conformation,
which :func:`cmfqsar.fields.build_fields` reads like any other.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from .atomtypes import TRANSITION_METALS, is_transition_metal

#: charges tried for a ligand fragment whose charge the data set does not state, in the order in
#: which they are tried.  Neutral first, then the singly and doubly charged anions, which are by
#: far the most common ligands; a charge that does not let the fragment sanitise is skipped.
TRY_LIGAND_CHARGES = (0, -1, -2, 1, -3, 2, -4)

#: how many assignments of donor atoms to reference donor atoms :func:`align_on_first_sphere` is
#: willing to enumerate.  Six donors of one element give 720 assignments; the limit stops a
#: complex with eight identical donors (40320) from taking minutes.
MAX_DONOR_PERMUTATIONS = 5040

#: a ``key=value`` token of a comment line.  Spaces around the equals sign are allowed, because
#: a comment line that separates its fields with vertical bars tends to write ``E(gas) = -1.5``.
_KEY_VALUE = re.compile(r'([A-Za-z_][\w().-]*)\s*=\s*("[^"]*"|\S+)')


# --------------------------------------------------------------------------- #
# reading coordinates
# --------------------------------------------------------------------------- #

@dataclass
class XyzFrame:
    """One block of an XYZ file: the elements, the coordinates and the comment line.

    The comment line of an XYZ file has no agreed format.  Two readings of it are offered and
    both are filled in, because a caller that knows its own file can use whichever fits:
    `tags` holds every ``key=value`` token (the extended-XYZ convention), and `parts` holds the
    comment split on the vertical bar, which several published data sets use to separate fields.
    """
    symbols: List[str]
    coords: np.ndarray
    comment: str = ""
    name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    parts: List[str] = field(default_factory=list)

    @property
    def n_atoms(self) -> int:
        return len(self.symbols)


def parse_comment(comment: str) -> Tuple[Dict[str, str], List[str]]:
    """The ``key=value`` tokens of a comment line, and the line split on the vertical bar."""
    tags = {}
    for key, val in _KEY_VALUE.findall(comment):
        tags[key] = val.strip('"')
    parts = [p.strip() for p in comment.split("|") if p.strip()]
    return tags, parts


def read_xyz(path: str, limit: Optional[int] = None) -> List[XyzFrame]:
    """Every block of a (possibly multi-block) XYZ file.

    The file is read literally: no bonds are perceived and no element is renamed.  `limit` stops
    after that many blocks, which is what a quick check of a large file needs.
    """
    out: List[XyzFrame] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        while True:
            head = fh.readline()
            if not head:
                break
            head = head.strip()
            if not head:
                continue
            try:
                n = int(head.split()[0])
            except (ValueError, IndexError):
                raise ValueError(f"{path}: expected an atom count, found {head!r}")
            comment = fh.readline().rstrip("\n")
            symbols, xyz = [], []
            for _ in range(n):
                line = fh.readline()
                if not line:
                    raise ValueError(f"{path}: the file ends inside a block of {n} atoms")
                p = line.split()
                symbols.append(p[0])
                xyz.append((float(p[1]), float(p[2]), float(p[3])))
            tags, parts = parse_comment(comment)
            name = tags.get("csd_code") or tags.get("name") or (parts[0] if parts else "")
            out.append(XyzFrame(symbols=symbols, coords=np.asarray(xyz, float), comment=comment,
                                name=name, tags=tags, parts=parts))
            if limit is not None and len(out) >= limit:
                break
    return out


def molecule_from_coordinates(symbols: Sequence[str], coords: np.ndarray) -> Chem.Mol:
    """An RDKit molecule with the given elements and coordinates and no bonds at all."""
    rw = Chem.RWMol()
    conf = Chem.Conformer(len(symbols))
    conf.Set3D(True)
    for i, (s, p) in enumerate(zip(symbols, coords)):
        rw.AddAtom(Chem.Atom(str(s)))
        conf.SetAtomPosition(i, Point3D(float(p[0]), float(p[1]), float(p[2])))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)
    return mol


# --------------------------------------------------------------------------- #
# building a complex
# --------------------------------------------------------------------------- #

@dataclass
class ComplexReport:
    """What the builder did, so that a data set can be checked instead of trusted."""
    name: str = ""
    metal: str = ""
    metal_charge: Optional[int] = None
    total_charge: int = 0
    donors: List[int] = field(default_factory=list)
    donor_symbols: List[str] = field(default_factory=list)
    ligand_charges: List[int] = field(default_factory=list)
    ligands_matched: int = 0
    n_ligands: int = 0
    net_charge: Optional[int] = None
    fault: str = ""

    @property
    def ok(self) -> bool:
        return not self.fault


def _canonical(smiles: str) -> Optional[str]:
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m) if m is not None else None


def _fragment_with_bond_orders(symbols, coords, idx, charge: int) -> Optional[Chem.Mol]:
    """One ligand fragment as a molecule, its bond orders determined at the given charge."""
    frag = molecule_from_coordinates([symbols[i] for i in idx], coords[list(idx)])
    try:
        rdDetermineBonds.DetermineBonds(frag, charge=int(charge), useHueckel=False)
        Chem.SanitizeMol(frag)
    except Exception:                                 # noqa: BLE001 - any failure means this
        return None                                   # charge is not the fragment's charge
    if Chem.GetFormalCharge(frag) == int(charge):
        return frag
    # A ligand of one atom is the common case of a mismatch: a chloride, bromide or iodide is one
    # atom with no bond to determine, and the algorithm leaves it neutral whatever charge it is
    # given.  The charge is written on it here, which is what the data set says it is.
    if frag.GetNumAtoms() == 1:
        frag.GetAtomWithIdx(0).SetFormalCharge(int(charge))
        frag.GetAtomWithIdx(0).SetNoImplicit(True)
        try:
            Chem.SanitizeMol(frag)
        except Exception:                             # noqa: BLE001
            return None
        return frag
    # Otherwise the algorithm answered with a radical instead of an ion, and the formal charges
    # it wrote do not add up to the charge it was asked for.  Such an answer is rejected, because
    # the charges of the ligands have to add up to the charge of the complex before the
    # electrostatic field means anything.
    return None


def _assign_ligand_charges(options: Sequence[Sequence[Tuple[int, Chem.Mol, bool]]]
                           ) -> List[Optional[int]]:
    """Give every ligand fragment one of the stated charges, and no charge to two fragments.

    `options` holds, for each fragment, the charges it can take, as positions in the data set's
    list of ligand charges, each marked as confirmed by the ligand's structure or not.  The
    answer gives each fragment the position of its choice inside its own list of options, or
    ``None`` when nothing is left for it.

    The search looks for an assignment that uses a distinct stated charge for every fragment, and
    among those for one with as many confirmed ligands as possible.  Such an assignment makes the
    sum of the ligand charges equal to the sum the data set states, so the charge left for the
    metal is its stated oxidation state.  The fragments are few, so the search is exhaustive with
    a bound: it stops as soon as it finds an assignment in which every ligand is confirmed.
    """
    n = len(options)
    best: List[Optional[int]] = [None] * n
    best_score = (-1, -1)                             # (fragments served, ligands confirmed)
    budget = [20000]                                  # a complex with many like ligands has many
                                                      # assignments; the best found so far is kept

    def walk(i: int, used: frozenset, pick: List[Optional[int]], served: int, confirmed: int):
        nonlocal best, best_score
        budget[0] -= 1
        if budget[0] < 0:
            return
        if i == n:
            score = (served, confirmed)
            if score > best_score:
                best_score, best = score, list(pick)
            return
        if best_score == (n, n):
            return
        for pos, (k, _frag, ok) in enumerate(options[i]):
            if k in used:
                continue
            pick[i] = pos
            walk(i + 1, used | {k}, pick, served + 1, confirmed + int(ok))
            pick[i] = None
            if best_score == (n, n):
                return
        # this fragment may also be left without a stated charge, which the caller then fills in
        pick[i] = None
        walk(i + 1, used, pick, served, confirmed)

    walk(0, frozenset(), [None] * n, 0, 0)
    return best


def build_complex(symbols: Sequence[str], coords: np.ndarray, total_charge: int = 0,
                  ligand_charges: Optional[Sequence[int]] = None,
                  ligand_smiles: Optional[Sequence[str]] = None,
                  name: str = "", multiplicity: Optional[int] = None,
                  metal: Optional[str] = None) -> Tuple[Optional[Chem.Mol], ComplexReport]:
    """Build one mononuclear complex from its coordinates, with dative metal-ligand bonds.

    `total_charge` is the charge of the whole complex.  `ligand_charges` are the charges of the
    ligands as the data set states them, in any order; they are matched to the fragments the
    coordinates give, and `ligand_smiles` (same order) lets the match be checked instead of
    guessed.  `multiplicity` is stored as a property and not used here, so that the quantum stage
    of plan item EC1 can pick the state up later.

    Returns the molecule and a report.  The molecule is ``None`` when the complex could not be
    built, and the report then says why; a data set is expected to lose a few members this way
    and the number lost is part of its description.
    """
    report = ComplexReport(name=name, total_charge=int(total_charge))
    mol = molecule_from_coordinates(symbols, coords)
    try:
        rdDetermineBonds.DetermineConnectivity(mol, useHueckel=False)
    except Exception as exc:                          # noqa: BLE001
        report.fault = f"no connectivity: {exc}"
        return None, report
    metals = [a.GetIdx() for a in mol.GetAtoms()
              if (a.GetSymbol() == metal if metal else is_transition_metal(a.GetSymbol()))]
    if len(metals) != 1:
        report.fault = f"{len(metals)} metal atoms, and this reader builds mononuclear complexes"
        return None, report
    mi = metals[0]
    report.metal = mol.GetAtomWithIdx(mi).GetSymbol()
    donors = sorted(n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors())
    if not donors:
        report.fault = "the metal atom has no neighbour within bonding distance"
        return None, report
    report.donors = list(donors)
    report.donor_symbols = [mol.GetAtomWithIdx(d).GetSymbol() for d in donors]

    # the ligands are what is left when the metal is taken out
    cut = Chem.RWMol(mol)
    for d in donors:
        cut.RemoveBond(mi, d)
    frags = [sorted(f) for f in Chem.GetMolFrags(cut.GetMol()) if mi not in f]
    frags.sort(key=len, reverse=True)
    report.n_ligands = len(frags)

    wanted: List[Tuple[Optional[str], int]] = []
    if ligand_charges is not None:
        smi = list(ligand_smiles) if ligand_smiles is not None else [None] * len(ligand_charges)
        wanted = [(_canonical(s) if s else None, int(q)) for s, q in zip(smi, ligand_charges)]

    # Which of the charges the data set states can each fragment take?  A charge is admissible
    # when the fragment's bond orders come out at that charge, and it is confirmed when the
    # fragment is then the ligand the data set names.  All of them are collected before anything
    # is decided, because each stated charge belongs to one ligand only: taking the first charge
    # that fits the first fragment can leave a later fragment with nothing, and the charge of the
    # metal is then wrong even though the charge of the complex is right.
    options: List[List[Tuple[int, Chem.Mol, bool]]] = []
    for idx in frags:
        opts: List[Tuple[int, Chem.Mol, bool]] = []
        for k, (want_smiles, q) in enumerate(wanted):
            frag = _fragment_with_bond_orders(symbols, coords, idx, q)
            if frag is None:
                continue
            confirmed = False
            if want_smiles is not None:
                try:
                    confirmed = Chem.MolToSmiles(Chem.RemoveHs(frag)) == want_smiles
                except Exception:                     # noqa: BLE001
                    confirmed = False
            opts.append((k, frag, confirmed))
        opts.sort(key=lambda o: (not o[2], o[0]))
        options.append(opts)
    chosen = _assign_ligand_charges(options)

    pieces: List[Tuple[List[int], Chem.Mol, int]] = []
    for idx, opts, pick in zip(frags, options, chosen):
        if pick is not None:
            k, frag, confirmed = opts[pick]
            report.ligands_matched += int(confirmed)
            pieces.append((idx, frag, wanted[k][1]))
            continue
        # no stated charge is left for this fragment: the common ligand charges in turn
        hit = None
        for q in TRY_LIGAND_CHARGES:
            frag = _fragment_with_bond_orders(symbols, coords, idx, q)
            if frag is not None:
                hit = (frag, q)
                break
        if hit is None:
            report.fault = f"the bond orders of a ligand of {len(idx)} atoms could not be found"
            return None, report
        pieces.append((idx, hit[0], hit[1]))

    out = Chem.RWMol()
    conf = Chem.Conformer(len(symbols))
    conf.Set3D(True)
    new: Dict[int, int] = {}
    atom = Chem.Atom(report.metal)
    atom.SetNoImplicit(True)
    metal_charge = int(total_charge) - sum(q for _, _, q in pieces)
    atom.SetFormalCharge(metal_charge)
    new[mi] = out.AddAtom(atom)
    conf.SetAtomPosition(new[mi], Point3D(*(float(v) for v in coords[mi])))
    for idx, frag, _q in pieces:
        for j, old in enumerate(idx):
            new[int(old)] = out.AddAtom(frag.GetAtomWithIdx(j))
            conf.SetAtomPosition(new[int(old)], Point3D(*(float(v) for v in coords[int(old)])))
        for b in frag.GetBonds():
            out.AddBond(new[int(idx[b.GetBeginAtomIdx()])], new[int(idx[b.GetEndAtomIdx()])],
                        b.GetBondType())
    for d in donors:
        # a dative bond points from the donor to the acceptor, so the donor keeps its own valence
        out.AddBond(new[int(d)], new[mi], Chem.BondType.DATIVE)
    built = out.GetMol()
    built.AddConformer(conf, assignId=True)
    try:
        Chem.SanitizeMol(built)
    except Exception as exc:                          # noqa: BLE001
        report.fault = f"the assembled complex does not sanitise: {exc}"
        return None, report
    built.SetProp("_Name", name)
    built.SetProp("charge", str(int(total_charge)))
    if multiplicity is not None:
        built.SetProp("multiplicity", str(int(multiplicity)))
    report.metal_charge = metal_charge
    report.ligand_charges = [q for _, _, q in pieces]
    report.net_charge = Chem.GetFormalCharge(built)
    if report.net_charge != int(total_charge):
        report.fault = (f"the formal charges add up to {report.net_charge:+d} and the data set "
                        f"says the complex has {int(total_charge):+d}")
        return None, report
    return built, report


# --------------------------------------------------------------------------- #
# the metal and its first coordination sphere
# --------------------------------------------------------------------------- #

def metal_index(mol: Chem.Mol) -> Optional[int]:
    """The index of the one transition-metal atom, or ``None`` when there is not exactly one."""
    metals = [a.GetIdx() for a in mol.GetAtoms() if is_transition_metal(a.GetSymbol())]
    return metals[0] if len(metals) == 1 else None


def donor_indices(mol: Chem.Mol) -> List[int]:
    """The atoms bonded to the metal: the first coordination sphere."""
    mi = metal_index(mol)
    if mi is None:
        return []
    return sorted(n.GetIdx() for n in mol.GetAtomWithIdx(mi).GetNeighbors())


def coordination(mol: Chem.Mol) -> Dict[str, float]:
    """Numbers that describe the metal site, for use as descriptors or as a check.

    ``n_donors`` is the coordination number, ``donor_<element>`` the number of donors of that
    element, ``metal_charge`` the formal charge the builder gave the metal, and ``d_min``,
    ``d_mean`` and ``d_max`` the distances from the metal to its donors in angstrom.
    """
    mi = metal_index(mol)
    if mi is None:
        return {}
    donors = donor_indices(mol)
    xyz = mol.GetConformer().GetPositions()
    d = np.linalg.norm(xyz[donors] - xyz[mi], axis=1) if donors else np.zeros(0)
    out: Dict[str, float] = {"n_donors": float(len(donors)),
                             "metal_charge": float(mol.GetAtomWithIdx(mi).GetFormalCharge()),
                             "d_min": float(d.min()) if len(d) else 0.0,
                             "d_mean": float(d.mean()) if len(d) else 0.0,
                             "d_max": float(d.max()) if len(d) else 0.0}
    for i in donors:
        key = "donor_" + mol.GetAtomWithIdx(i).GetSymbol()
        out[key] = out.get(key, 0.0) + 1.0
    return out


def _set_positions(mol: Chem.Mol, xyz: np.ndarray) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for i in range(out.GetNumAtoms()):
        conf.SetAtomPosition(i, Point3D(*(float(v) for v in xyz[i])))
    return out


def metal_origin(mol: Chem.Mol) -> Chem.Mol:
    """The same molecule with its metal atom at the origin."""
    mi = metal_index(mol)
    if mi is None:
        return Chem.Mol(mol)
    xyz = mol.GetConformer().GetPositions()
    return _set_positions(mol, xyz - xyz[mi])


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The rotation that takes `a` closest to `b`, both already centred on the origin.

    A reflection is never returned: two complexes that are mirror images of each other are
    different molecules, and a frame that hid the difference would make the handedness of a
    ligand invisible to the field.
    """
    u, _s, vt = np.linalg.svd(a.T @ b)
    d = np.sign(np.linalg.det(u @ vt))
    if d < 0:
        u[:, -1] *= -1.0
    return u @ vt


def align_on_first_sphere(mols: Sequence[Chem.Mol], ref_index: int = 0,
                          max_permutations: int = MAX_DONOR_PERMUTATIONS):
    """Put every complex in the frame of one of them, using the metal and its donors.

    The metal goes to the origin and the rotation is the one that brings the donor atoms closest
    to the reference complex's donor atoms.  Which donor answers to which reference donor is not
    known, and a coordination sphere is symmetric, so every assignment that pairs donors of the
    same element is tried and the best is kept.  Trying one arbitrary assignment instead would
    report several angstrom for two complexes that already lie on top of each other.

    Returns the aligned molecules and, for each, the root-mean-square distance between its donors
    and the reference donors in angstrom, or ``None`` when its coordination sphere holds a
    different set of elements from the reference's and there is nothing to match.
    """
    ref = mols[ref_index]
    ref_mi = metal_index(ref)
    ref_donors = donor_indices(ref)
    ref_xyz = ref.GetConformer().GetPositions()
    ref_d = ref_xyz[ref_donors] - ref_xyz[ref_mi]
    ref_syms = [ref.GetAtomWithIdx(i).GetSymbol() for i in ref_donors]
    out, rmsds = [], []
    for mol in mols:
        centred = metal_origin(mol)
        mi = metal_index(centred)
        donors = donor_indices(centred)
        if mi is None or len(donors) != len(ref_donors):
            out.append(centred)
            rmsds.append(None)
            continue
        xyz = centred.GetConformer().GetPositions()
        syms = [centred.GetAtomWithIdx(i).GetSymbol() for i in donors]
        if sorted(syms) != sorted(ref_syms):
            out.append(centred)
            rmsds.append(None)
            continue
        # the donors of one element may answer to the reference donors of that element in any
        # order; the assignments are the product of one permutation per element
        groups = {}
        for pos, s in enumerate(syms):
            groups.setdefault(s, []).append(pos)
        ref_groups = {}
        for pos, s in enumerate(ref_syms):
            ref_groups.setdefault(s, []).append(pos)
        count = 1
        for g in groups.values():
            count *= _factorial(len(g))
        order = sorted(groups)
        a = xyz[donors]
        pairing = dict(ref_groups)
        if count > max_permutations:
            # Too many assignments to try them all: the donors of each element are matched by
            # their distance from the metal, on both sides.  That is right whenever the distances
            # tell the donors of one element apart, and wrong by a swap when they do not.
            perms = [tuple(tuple(sorted(groups[s],
                                        key=lambda p: float(np.linalg.norm(a[p])))) for s in order)]
            pairing = {s: sorted(ref_groups[s],
                                 key=lambda p: float(np.linalg.norm(ref_d[p]))) for s in order}
        else:
            perms = product(*(permutations(groups[s]) for s in order))
        best_r, best_rmsd = None, None
        for combo in perms:
            rows_a, rows_b = [], []
            for s, perm in zip(order, combo):
                for p, refp in zip(perm, pairing[s]):
                    rows_a.append(a[p])
                    rows_b.append(ref_d[refp])
            r = _kabsch(np.asarray(rows_a), np.asarray(rows_b))
            val = float(np.sqrt(np.mean(np.sum((np.asarray(rows_a) @ r - np.asarray(rows_b)) ** 2,
                                               axis=1))))
            if best_rmsd is None or val < best_rmsd:
                best_rmsd, best_r = val, r
        out.append(_set_positions(centred, xyz @ best_r))
        rmsds.append(best_rmsd)
    return out, rmsds


def _factorial(n: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


def principal_metal_frame(mol: Chem.Mol, degenerate_tolerance: float = 0.05):
    """A frame for a complex that shares no coordination sphere with any other.

    The metal goes to the origin and the axes are the principal axes of the whole complex, with
    every atom weighted by its atomic number, in order of decreasing moment.  The sign of each
    axis is fixed by the third moment along it, which is the usual way to remove the remaining
    ambiguity: an axis and its opposite give the same second moment, but the heavier side of the
    molecule decides the sign.

    Returns the molecule in that frame and the relative gap between the two closest moments.  A
    small gap means the two axes are interchangeable and the frame is arbitrary in that plane, so
    the number is reported rather than hidden: it says how much of the frame is a real rule.
    """
    centred = metal_origin(mol)
    xyz = centred.GetConformer().GetPositions()
    w = np.array([a.GetAtomicNum() for a in centred.GetAtoms()], float)
    tensor = (xyz * w[:, None]).T @ xyz / max(w.sum(), 1e-12)
    vals, vecs = np.linalg.eigh(tensor)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    for k in range(3):
        proj = xyz @ vecs[:, k]
        third = float(np.sum(w * proj ** 3))
        if third < 0:
            vecs[:, k] *= -1.0
    if np.linalg.det(vecs) < 0:                       # keep a rotation, never a reflection
        vecs[:, 2] *= -1.0
    gaps = [abs(vals[0] - vals[1]), abs(vals[1] - vals[2])]
    scale = max(float(vals[0]), 1e-12)
    return _set_positions(centred, xyz @ vecs), float(min(gaps) / scale)


__all__ = ["TRANSITION_METALS", "TRY_LIGAND_CHARGES", "XyzFrame", "ComplexReport",
           "read_xyz", "parse_comment", "molecule_from_coordinates", "build_complex",
           "metal_index", "donor_indices", "coordination", "metal_origin",
           "align_on_first_sphere", "principal_metal_frame"]
