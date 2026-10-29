"""
Atom typing (Tripos/Sybyl-like) and van der Waals parameters.

Sybyl atom types are used (a) for continuous indicator fields (CIFs) and
(b) to look up Tripos force-field van der Waals parameters for the steric
field (Baskin & Zhokhova 2013, Eq. 10-11).
"""
from __future__ import annotations

from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import rdchem

HybridizationType = rdchem.HybridizationType

# Tripos 5.2 force field van der Waals parameters: (radius [A], well depth E [kcal/mol])
# Clark, Cramer, Van Opdenbosch, J. Comput. Chem. 10 (1989) 982.
_TRIPOS_VDW = {
    "C": (1.70, 0.107),
    "N": (1.55, 0.095),
    "O": (1.52, 0.116),
    "S": (1.80, 0.314),
    "P": (1.80, 0.314),
    "H": (1.50, 0.042),
    "F": (1.47, 0.109),
    "Cl": (1.75, 0.314),
    "Br": (1.85, 0.434),
    "I": (1.98, 0.623),
    "Si": (2.10, 0.402),
    "B": (1.80, 0.100),
    "Se": (1.90, 0.350),
}
_DEFAULT_VDW = (1.80, 0.200)

# Tripos 5.2 parameters by Sybyl atom type (cmf-tripos.prm of the original CMF scripts).
# They coincide with the element values above except for the sulfoxide/sulfone sulfur.
_TRIPOS_VDW_BY_TYPE = {
    "C.1": (1.70, 0.107), "C.2": (1.70, 0.107), "C.3": (1.70, 0.107), "C.ar": (1.70, 0.107),
    "C.cat": (1.70, 0.107),
    "N.1": (1.55, 0.095), "N.2": (1.55, 0.095), "N.3": (1.55, 0.095), "N.4": (1.55, 0.095),
    "N.am": (1.55, 0.095), "N.ar": (1.55, 0.095), "N.pl3": (1.55, 0.095),
    "O.2": (1.52, 0.116), "O.3": (1.52, 0.116), "O.co2": (1.52, 0.116),
    "S.2": (1.80, 0.314), "S.3": (1.80, 0.314), "S.o": (1.70, 0.314), "S.o2": (1.70, 0.314),
    "P.3": (1.80, 0.314), "H": (1.50, 0.042), "F": (1.47, 0.109), "Cl": (1.75, 0.314),
    "Br": (1.85, 0.434), "I": (1.98, 0.623), "Si": (2.10, 0.042),
    "Li": (1.20, 0.400), "Na": (1.20, 0.400), "K": (1.20, 0.400), "Ca": (1.20, 0.600), "Al": (1.20, 0.042),
}


def vdw_params(symbol: str, atom_type: Optional[str] = None):
    """Return (R_vdw, E_vdw) for an element symbol, or for a Sybyl atom type if given."""
    if atom_type is not None:
        t = atom_type
        if t.lower() in ("s.o", "s.o2"):
            t = "S." + t.lower()[2:]
        if t in _TRIPOS_VDW_BY_TYPE:
            return _TRIPOS_VDW_BY_TYPE[t]
    return _TRIPOS_VDW.get(symbol, _DEFAULT_VDW)


# --------------------------------------------------------------------------- #
# Sybyl atom typing
# --------------------------------------------------------------------------- #

def _has_double_to(atom: Chem.Atom, symbols) -> bool:
    for b in atom.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE:
            o = b.GetOtherAtom(atom)
            if o.GetSymbol() in symbols:
                return True
    return False


def _count_neighbors(atom: Chem.Atom, symbol: str) -> int:
    return sum(1 for n in atom.GetNeighbors() if n.GetSymbol() == symbol)


def _is_amide_n(atom: Chem.Atom) -> bool:
    """Nitrogen bonded to a carbon that is double bonded to O or S (amide, thioamide, urea...)."""
    if atom.GetIsAromatic():
        return False
    for n in atom.GetNeighbors():
        if n.GetSymbol() == "C" and _has_double_to(n, ("O", "S")):
            return True
        if n.GetSymbol() == "S" and _count_neighbors(n, "O") >= 2:  # sulfonamide N
            return True
    return False


def _is_carboxylate_like_o(atom: Chem.Atom) -> bool:
    """Oxygen of carboxylate/carboxylic acid, phosphate, nitro, sulfonate groups (Sybyl O.co2)."""
    if atom.GetDegree() != 1:
        return False
    c = atom.GetNeighbors()[0]
    sym = c.GetSymbol()
    if sym == "C":
        # carboxylate / carboxylic acid: carbon with two terminal oxygens (C(=O)O-, C(=O)OH)
        term_o = [n for n in c.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1]
        if len(term_o) >= 2:
            return True
        # carboxylic acid where OH oxygen has H attached (degree 2 with explicit H)
        if atom.GetFormalCharge() == -1:
            return True
        os_ = [n for n in c.GetNeighbors() if n.GetSymbol() == "O"]
        if len(os_) == 2 and _has_double_to(c, ("O",)):
            other = [o for o in os_ if o.GetIdx() != atom.GetIdx()][0]
            # the other oxygen is OH (only H neighbours besides C)
            if all(nn.GetSymbol() in ("C", "H") for nn in other.GetNeighbors()) and other.GetTotalNumHs(includeNeighbors=True) > 0:
                return True
        return False
    if sym == "N":
        # nitro group
        term_o = [n for n in c.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1]
        return len(term_o) >= 2
    if sym in ("P", "S"):
        term_o = [n for n in c.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1]
        return len(term_o) >= 2
    return False


def sybyl_atom_type(atom: Chem.Atom) -> str:
    """Assign an approximate Sybyl (Tripos) atom type to an RDKit atom."""
    sym = atom.GetSymbol()
    hyb = atom.GetHybridization()
    arom = atom.GetIsAromatic()

    if sym == "C":
        if arom:
            return "C.ar"
        # guanidinium-like carbon: three N neighbours, no double bond to O
        if _count_neighbors(atom, "N") == 3 and not _has_double_to(atom, ("O", "S")):
            return "C.cat"
        if hyb == HybridizationType.SP3:
            return "C.3"
        if hyb == HybridizationType.SP2:
            return "C.2"
        if hyb == HybridizationType.SP:
            return "C.1"
        return "C.3"

    if sym == "N":
        if arom:
            return "N.ar"
        if _is_amide_n(atom):
            return "N.am"
        heavy_deg = atom.GetDegree()
        if atom.GetFormalCharge() == 1 and heavy_deg == 4:
            return "N.4"
        # nitro nitrogen
        term_o = [n for n in atom.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1]
        if len(term_o) >= 2:
            return "N.pl3"
        if hyb == HybridizationType.SP:
            return "N.1"
        if hyb == HybridizationType.SP2:
            if _has_double_to(atom, ("C", "N", "O", "S")):
                return "N.2"
            return "N.pl3"  # conjugated planar nitrogen (aniline, enamine)
        if hyb == HybridizationType.SP3:
            if atom.GetFormalCharge() == 1:
                return "N.4"
            # planar if attached to aromatic/sp2 atoms (RDKit may keep SP3 for anilines)
            if any(n.GetIsAromatic() or n.GetHybridization() == HybridizationType.SP2 for n in atom.GetNeighbors()):
                return "N.pl3"
            return "N.3"
        return "N.3"

    if sym == "O":
        if _is_carboxylate_like_o(atom):
            return "O.co2"
        if _has_double_to(atom, ("C", "N", "S", "P")) or (hyb == HybridizationType.SP2 and atom.GetDegree() == 1):
            return "O.2"
        return "O.3"

    if sym == "S":
        n_o = len([n for n in atom.GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1])
        if n_o >= 2:
            return "S.O2"
        if n_o == 1:
            return "S.O"
        if _has_double_to(atom, ("C", "N", "P")) or arom or hyb == HybridizationType.SP2:
            return "S.2"
        return "S.3"

    if sym == "P":
        return "P.3"
    if sym == "H":
        return "H"
    return sym  # F, Cl, Br, I, Si, metals...


def element_atom_type(atom: Chem.Atom) -> str:
    return atom.GetSymbol()


def element_aromatic_atom_type(atom: Chem.Atom) -> str:
    sym = atom.GetSymbol()
    return sym + ".ar" if atom.GetIsAromatic() else sym


def pbh_atom_type(atom: Chem.Atom) -> str:
    """The carbon classes of a benzenoid hydrocarbon: a rim carbon bonded to a hydrogen
    (``C.H``), a carbon shared by two rings (``C.2r``), by three (``C.3r``); see
    :func:`cmfqsar.pbh.pbh_atom_type`."""
    from .pbh import pbh_atom_type as _pbh_atom_type
    return _pbh_atom_type(atom)


TYPING_SCHEMES = {
    "sybyl": sybyl_atom_type,
    "element": element_atom_type,
    "element+aromatic": element_aromatic_atom_type,
    "pbh": pbh_atom_type,
}


def assign_atom_types(mol: Chem.Mol, scheme: str = "sybyl") -> List[str]:
    """Return a list with one atom type string per atom in `mol`.

    If the molecule carries pre-assigned types in the atom property ``_TriposAtomType``
    (e.g. read from a MOL2 file) and ``scheme == "sybyl"`` these are used directly.
    """
    fn = TYPING_SCHEMES[scheme]
    types = []
    for atom in mol.GetAtoms():
        if scheme == "sybyl" and atom.HasProp("_TriposAtomType"):
            types.append(atom.GetProp("_TriposAtomType"))
        elif scheme == "sybyl" and atom.HasProp("sybyl_type"):
            # per-atom property list stored in an SDF (tag "atom.prop.sybyl_type")
            types.append(atom.GetProp("sybyl_type"))
        else:
            types.append(fn(atom))
    return types
