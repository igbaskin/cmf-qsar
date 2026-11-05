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

# Van der Waals parameters for the transition metals, which the Tripos 5.2 table does not hold
# (plan item EC4).  They come from the universal force field (UFF: Rappe, Casewit, Colwell,
# Goddard & Skiff, J. Am. Chem. Soc. 114 (1992) 10024), Table 1, which covers the whole periodic
# table.  The two UFF columns used here are the van der Waals distance x (the separation of two
# atoms of the same element at the minimum of the curve, in angstrom) and the well depth D (in
# kcal/mol, the unit of the Tripos column).  One atom's radius is x / 2.
#
# UFF radii are about 13 % larger than the Tripos ones, because the two force fields split the
# contact distance differently: carbon is 3.851 / 2 = 1.93 angstrom in UFF against 1.70 in
# Tripos.  Every metal radius is therefore multiplied by the ratio of the two carbon radii, so
# that a metal and a carbon atom of the same molecule are measured on one scale.  The well depths
# are taken as they are, because the two force fields agree on carbon (0.105 against 0.107
# kcal/mol).
#
# The entry chosen for each element is its most common oxidation state and coordination in the
# UFF table (Fe3+2 and Fe6+2 share one pair of values, as do the other duplicated forms).
_UFF_NONBONDED = {
    "Sc": (3.295, 0.019), "Ti": (3.175, 0.017), "V": (3.144, 0.016), "Cr": (3.023, 0.015),
    "Mn": (2.961, 0.013), "Fe": (2.912, 0.013), "Co": (2.872, 0.014), "Ni": (2.834, 0.015),
    "Cu": (3.495, 0.005), "Zn": (2.763, 0.124),
    "Y": (3.345, 0.072), "Zr": (3.124, 0.069), "Nb": (3.165, 0.059), "Mo": (3.052, 0.056),
    "Tc": (2.998, 0.048), "Ru": (2.963, 0.056), "Rh": (2.929, 0.053), "Pd": (2.899, 0.048),
    "Ag": (3.148, 0.036), "Cd": (2.848, 0.228),
    "Hf": (3.141, 0.072), "Ta": (3.170, 0.081), "W": (3.069, 0.067), "Re": (2.954, 0.066),
    "Os": (3.120, 0.037), "Ir": (2.840, 0.073), "Pt": (2.754, 0.080), "Au": (3.293, 0.039),
    "Hg": (2.705, 0.385),
}
#: the factor that puts a UFF radius on the scale of the Tripos radii, from their carbon values
UFF_RADIUS_SCALE = 1.70 / (3.851 / 2.0)
_METAL_VDW = {sym: (round(x / 2.0 * UFF_RADIUS_SCALE, 3), d)
              for sym, (x, d) in _UFF_NONBONDED.items()}

#: the d-block metals for which this module carries parameters.  A complex of any other metal
#: still loads, and its metal atom then falls back to the default radius and well depth.
TRANSITION_METALS = frozenset(_METAL_VDW)


def is_transition_metal(symbol: str) -> bool:
    """Is this element symbol one of the d-block metals (plan item EC4)?"""
    return symbol in TRANSITION_METALS


def vdw_params(symbol: str, atom_type: Optional[str] = None):
    """Return (R_vdw, E_vdw) for an element symbol, or for a Sybyl atom type if given."""
    if atom_type is not None:
        t = atom_type
        if t.lower() in ("s.o", "s.o2"):
            t = "S." + t.lower()[2:]
        if t in _TRIPOS_VDW_BY_TYPE:
            return _TRIPOS_VDW_BY_TYPE[t]
        # a metal written by Sybyl carries its coordination in the type ("Co.oh", "Ti.th"), and
        # the parameters belong to the element rather than to the coordination
        base = t.split(".")[0]
        if base in _METAL_VDW:
            return _METAL_VDW[base]
    if symbol in _TRIPOS_VDW:
        return _TRIPOS_VDW[symbol]
    return _METAL_VDW.get(symbol, _DEFAULT_VDW)


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
