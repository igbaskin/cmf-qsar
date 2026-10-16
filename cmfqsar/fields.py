"""
Parameterisation of continuous molecular fields.

Every molecular field of type *f* is approximated by a sum of atom-centred
isotropic Gaussian functions (Baskin & Zhokhova 2013, Eq. 4 and Eq. 10):

    rho_f,il(r) = w_f,il * exp(-1/2 * b_f,il * |r - r_il|^2)

with ``b = alpha_f`` for ordinary fields and ``b = alpha_f / R_vdw^2`` for the
steric field (whose width scales with the van der Waals radius of the atom).

Field types implemented
-----------------------
electrostatic   w = partial atomic charge (Gasteiger, MMFF94 or from file)
steric          w = Tripos vdW well depth E_vdw, width scaled by R_vdw
hydrophobic     w = Wildman-Crippen atomic logP contribution
hba             w = approximate hydrogen-bond acceptor increment (Abraham-B-like)
hbd             w = approximate hydrogen-bond donor increment (Abraham-A-like),
                placed on the polar hydrogen atoms

With ``FieldSet.param_scheme == "tripos"`` the hydrophobic, hba and hbd weights
are instead the in-house increments of the original CMF R scripts (keyed by
Sybyl atom type and number of attached hydrogens, donor increments on the heavy
atom; see :mod:`cmfqsar.increments`), and the steric parameters are looked up
by Sybyl atom type.  ``"tripos-legacy"`` reproduces the 2013 scripts exactly,
including their hydrogen-counting bug (attached hydrogens are only counted when
the hydrogen is the first atom of the bond record, which practically never
happens in Sybyl MOL2 files).
cif:<type>      continuous indicator field: w = 1 for atoms of Sybyl type <type>
                (Sitnikov et al. 2015, Eq. 6)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

from .atomtypes import assign_atom_types, vdw_params
from .increments import tripos_increments

PHYSCHEM_FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")

FIELD_LABELS = {
    "electrostatic": "Electrostatic",
    "steric": "Steric",
    "hydrophobic": "Hydrophobic",
    "hba": "H-bond acceptor",
    "hbd": "H-bond donor",
}


@dataclass
class FieldSpec:
    """Description of one molecular field type."""
    name: str                 # "electrostatic", "steric", ..., or "cif:C.ar"
    kind: str = "gauss"       # "gauss" -> b = alpha ; "steric" -> b = alpha / R^2
    alpha: float = 1.0        # attenuation factor (A^-2 for "gauss"; dimensionless for "steric")
    h: float = 1.0            # mixing coefficient
    enabled: bool = True

    @property
    def label(self) -> str:
        if self.name.startswith("cif:"):
            return "CIF " + self.name[4:]
        return FIELD_LABELS.get(self.name, self.name)

    @property
    def is_cif(self) -> bool:
        return self.name.startswith("cif:")

    @property
    def cif_type(self) -> Optional[str]:
        return self.name[4:] if self.is_cif else None


@dataclass
class FieldSet:
    """The complete set of field types used by a model plus parameterisation options."""
    specs: List[FieldSpec] = field(default_factory=list)
    charge_model: str = "gasteiger"     # "gasteiger" | "mmff" | "file"
    cif_scheme: str = "sybyl"           # "sybyl" | "element" | "element+aromatic"
    param_scheme: str = "rdkit"         # "rdkit" | "tripos" | "tripos-legacy"

    # ----- construction helpers -------------------------------------------------
    @classmethod
    def create(cls, physchem: Sequence[str] = PHYSCHEM_FIELDS, cif_types: Sequence[str] = (),
               alpha: float = 1.0, charge_model: str = "gasteiger", cif_scheme: str = "sybyl",
               param_scheme: str = "rdkit") -> "FieldSet":
        specs = []
        for name in physchem:
            kind = "steric" if name == "steric" else "gauss"
            specs.append(FieldSpec(name=name, kind=kind, alpha=alpha))
        for t in cif_types:
            specs.append(FieldSpec(name=f"cif:{t}", kind="gauss", alpha=alpha))
        return cls(specs=specs, charge_model=charge_model, cif_scheme=cif_scheme, param_scheme=param_scheme)

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.specs]

    @property
    def n_fields(self) -> int:
        return len(self.specs)

    def set_alpha(self, alpha: float, names: Optional[Sequence[str]] = None) -> None:
        for s in self.specs:
            if names is None or s.name in names:
                s.alpha = float(alpha)

    def set_mixing(self, h: Dict[str, float]) -> None:
        for s in self.specs:
            if s.name in h:
                s.h = float(h[s.name])

    def mixing_vector(self) -> np.ndarray:
        return np.array([s.h if s.enabled else 0.0 for s in self.specs], dtype=float)

    def copy(self) -> "FieldSet":
        return FieldSet(specs=[FieldSpec(**vars(s)) for s in self.specs],
                        charge_model=self.charge_model, cif_scheme=self.cif_scheme,
                        param_scheme=getattr(self, "param_scheme", "rdkit"))


# --------------------------------------------------------------------------- #
# Atom-level parameterisation
# --------------------------------------------------------------------------- #

def _partial_charges(mol: Chem.Mol, model: str) -> np.ndarray:
    n = mol.GetNumAtoms()
    if model == "file":
        q = np.zeros(n)
        ok = True
        for a in mol.GetAtoms():
            if a.HasProp("_TriposPartialCharge"):
                q[a.GetIdx()] = a.GetDoubleProp("_TriposPartialCharge")
            elif a.HasProp("partial_charge"):
                q[a.GetIdx()] = a.GetDoubleProp("partial_charge")
            else:
                ok = False
                break
        if ok:
            return q
        model = "gasteiger"  # fall back
    if model == "mmff":
        try:
            # work on a copy: MMFF set-up changes the aromaticity flags of the molecule in place
            props = AllChem.MMFFGetMoleculeProperties(Chem.Mol(mol))
            if props is not None:
                return np.array([props.GetMMFFPartialCharge(i) for i in range(n)], dtype=float)
        except Exception:
            pass
        model = "gasteiger"
    AllChem.ComputeGasteigerCharges(mol)
    q = np.array([float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
                  for a in mol.GetAtoms()])
    q[~np.isfinite(q)] = 0.0
    return q


def _hydrophobic_weights(mol: Chem.Mol) -> np.ndarray:
    contribs = rdMolDescriptors._CalcCrippenContribs(mol)
    return np.array([c[0] for c in contribs], dtype=float)


def _hydrogen_counts(mol: Chem.Mol, legacy: bool = False) -> np.ndarray:
    """Number of hydrogen atoms attached to every atom.

    ``legacy=True`` reproduces the bug of the original R scripts (cmf-params.R), in
    which a hydrogen was only counted when it was the *first* atom of the bond
    record; the bond order of the input file is preserved by RDKit as
    begin/end atom, so the legacy count can be reproduced from the RDKit molecule.
    """
    n = mol.GetNumAtoms()
    nh = np.zeros(n, dtype=int)
    if legacy:
        for b in mol.GetBonds():
            if b.GetBeginAtom().GetAtomicNum() == 1:
                nh[b.GetEndAtomIdx()] += 1
        return nh
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1:
            continue
        nh[a.GetIdx()] = a.GetTotalNumHs(includeNeighbors=True)
    return nh


def _tripos_weights(syb_types: Sequence[str], n_h: np.ndarray):
    """Hydrophobic, acceptor (B) and donor (A) increments of the original CMF scripts."""
    n = len(syb_types)
    hydro, acc, don = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, (t, k) in enumerate(zip(syb_types, n_h)):
        if t == "H":
            continue
        inc = tripos_increments(t, int(k))
        hydro[i], acc[i], don[i] = inc["hydrophobic"], inc["hba"], inc["hbd"]
    return hydro, acc, don


def _hbond_weights(mol: Chem.Mol):
    """Approximate hydrogen-bond acceptor (B-like) and donor (A-like) increments.

    Values are rough Abraham-type fragment increments; they only need to be
    *relative* weights for the corresponding Gaussian fields.
    Donor increments are placed on the polar hydrogen atoms so that the field
    reflects the actual position of the donating hydrogen.
    """
    n = mol.GetNumAtoms()
    acc = np.zeros(n)
    don = np.zeros(n)
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        idx = a.GetIdx()
        nbrs = list(a.GetNeighbors())
        heavy = [x for x in nbrs if x.GetAtomicNum() > 1]
        hs = [x for x in nbrs if x.GetAtomicNum() == 1]
        charge = a.GetFormalCharge()
        if sym == "O":
            dbl = any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())
            if a.GetDegree() == 1 and heavy:
                c = heavy[0]
                csym = c.GetSymbol()
                if csym == "C":
                    term_o = [x for x in c.GetNeighbors() if x.GetSymbol() == "O" and x.GetDegree() == 1]
                    n_n = sum(1 for x in c.GetNeighbors() if x.GetSymbol() == "N")
                    if charge == -1 or len(term_o) >= 2:
                        acc[idx] = 0.45   # carboxylate-type oxygen
                    elif n_n >= 1:
                        acc[idx] = 0.65   # amide carbonyl
                    elif any(x.GetSymbol() == "O" and x.GetIdx() != idx for x in c.GetNeighbors()):
                        acc[idx] = 0.45   # ester / acid carbonyl
                    else:
                        acc[idx] = 0.50   # ketone / aldehyde
                elif csym == "N":
                    acc[idx] = 0.15       # nitro / N-oxide oxygen
                elif csym == "S":
                    acc[idx] = 0.30       # sulfonyl / sulfoxide oxygen
                elif csym == "P":
                    acc[idx] = 0.70       # phosphoryl
                else:
                    acc[idx] = 0.40
            elif hs:  # hydroxyl
                heavy_sym = heavy[0].GetSymbol() if heavy else ""
                arom = heavy[0].GetIsAromatic() if heavy else False
                carbonyl = heavy and heavy_sym == "C" and any(
                    b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(heavy[0]).GetSymbol() == "O"
                    for b in heavy[0].GetBonds())
                if carbonyl:
                    acc[idx] = 0.10
                    dval = 0.59   # carboxylic acid OH
                elif arom:
                    acc[idx] = 0.25
                    dval = 0.60   # phenol
                elif heavy_sym == "N":
                    acc[idx] = 0.25
                    dval = 0.40   # oxime / hydroxylamine
                else:
                    acc[idx] = 0.45
                    dval = 0.37   # alcohol
                for h in hs:
                    don[h.GetIdx()] = dval
            elif not dbl:
                # ether / ester oxygen
                if any(x.GetIsAromatic() for x in heavy):
                    acc[idx] = 0.25
                else:
                    acc[idx] = 0.45
        elif sym == "N":
            arom = a.GetIsAromatic()
            amide = any(x.GetSymbol() == "C" and any(
                b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(x).GetSymbol() in ("O", "S")
                for b in x.GetBonds()) for x in heavy)
            sulfonamide = any(x.GetSymbol() == "S" and sum(1 for y in x.GetNeighbors()
                                                            if y.GetSymbol() == "O") >= 2 for x in heavy)
            nitro = sum(1 for x in heavy if x.GetSymbol() == "O" and x.GetDegree() == 1) >= 2
            aniline = any(x.GetIsAromatic() for x in heavy) and not arom
            triple = any(b.GetBondType() == Chem.BondType.TRIPLE for b in a.GetBonds())
            dbl = any(b.GetBondType() == Chem.BondType.DOUBLE for b in a.GetBonds())
            if charge > 0:
                acc[idx] = 0.0
                dval = 0.50
            elif nitro:
                acc[idx] = 0.0
                dval = 0.0
            elif arom:
                acc[idx] = 0.52 if not hs and a.GetDegree() == 2 else 0.0
                dval = 0.40
            elif amide or sulfonamide:
                acc[idx] = 0.0
                dval = 0.45 if amide else 0.30
            elif triple:
                acc[idx] = 0.33
                dval = 0.0
            elif dbl:
                acc[idx] = 0.50
                dval = 0.25
            elif aniline:
                acc[idx] = 0.40
                dval = 0.15
            else:
                acc[idx] = 0.70 if len(heavy) == 3 else 0.60
                dval = 0.08
            for h in hs:
                don[h.GetIdx()] = dval
        elif sym == "S":
            n_o = sum(1 for x in heavy if x.GetSymbol() == "O")
            if n_o == 0:
                acc[idx] = 0.25
            for h in hs:
                don[h.GetIdx()] = 0.10
        elif sym == "F":
            acc[idx] = 0.05
        elif sym == "Cl":
            acc[idx] = 0.08
        elif sym == "Br":
            acc[idx] = 0.12
        elif sym == "I":
            acc[idx] = 0.18
    return acc, don


# --------------------------------------------------------------------------- #
# Per-molecule container
# --------------------------------------------------------------------------- #

@dataclass
class MoleculeFields:
    """Atom-level data of one molecule needed to evaluate its continuous fields."""
    name: str
    coords: np.ndarray            # (N, 3)
    weights: np.ndarray           # (N, F) weights w_f for every field in the FieldSet
    radii: np.ndarray             # (N,) van der Waals radii (for the steric field)
    symbols: List[str]
    atom_types: List[str]
    bonds: np.ndarray             # (B, 2) atom indices (for drawing)
    charges: np.ndarray           # (N,) partial charges (kept for information)
    molblock: str = ""            # MOL block for later display / export

    @property
    def n_atoms(self) -> int:
        return len(self.coords)


def cif_types_in(mols: Sequence[Chem.Mol], scheme: str = "sybyl", min_molecules: int = 5) -> List[str]:
    """Atom types occurring in at least `min_molecules` molecules (Sitnikov et al. 2015)."""
    counts: Dict[str, int] = {}
    for m in mols:
        for t in set(assign_atom_types(m, scheme)):
            counts[t] = counts.get(t, 0) + 1
    return sorted([t for t, c in counts.items() if c >= min_molecules])


def atom_type_counts(mols: Sequence[Chem.Mol], scheme: str = "sybyl") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for m in mols:
        for t in set(assign_atom_types(m, scheme)):
            counts[t] = counts.get(t, 0) + 1
    return dict(sorted(counts.items()))


def build_fields(mol: Chem.Mol, fieldset: FieldSet, name: str = "") -> MoleculeFields:
    """Compute atom coordinates and per-field weights for one molecule."""
    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions(), dtype=float)
    n = mol.GetNumAtoms()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    scheme = getattr(fieldset, "param_scheme", "rdkit")
    syb = assign_atom_types(mol, "sybyl")       # for vdW parameters and Tripos increments
    types = syb if fieldset.cif_scheme == "sybyl" else assign_atom_types(mol, fieldset.cif_scheme)
    if scheme.startswith("tripos"):
        radii = np.array([vdw_params(s, t)[0] for s, t in zip(symbols, syb)])
        evdw = np.array([vdw_params(s, t)[1] for s, t in zip(symbols, syb)])
    else:
        radii = np.array([vdw_params(s)[0] for s in symbols])
        evdw = np.array([vdw_params(s)[1] for s in symbols])

    need = set(s.name for s in fieldset.specs)
    charges = _partial_charges(mol, fieldset.charge_model) if "electrostatic" in need else np.zeros(n)
    hydro = acc = don = None
    if scheme.startswith("tripos"):
        if need & {"hydrophobic", "hba", "hbd"}:
            hydro, acc, don = _tripos_weights(syb, _hydrogen_counts(mol, legacy=(scheme == "tripos-legacy")))
    else:
        if "hydrophobic" in need:
            hydro = _hydrophobic_weights(mol)
        if "hba" in need or "hbd" in need:
            acc, don = _hbond_weights(mol)

    W = np.zeros((n, fieldset.n_fields))
    for k, spec in enumerate(fieldset.specs):
        if spec.name == "electrostatic":
            W[:, k] = charges
        elif spec.name == "steric":
            W[:, k] = evdw
        elif spec.name == "hydrophobic":
            W[:, k] = hydro
        elif spec.name == "hba":
            W[:, k] = acc
        elif spec.name == "hbd":
            W[:, k] = don
        elif spec.is_cif:
            t = spec.cif_type
            W[:, k] = np.array([1.0 if x == t else 0.0 for x in types])
        else:
            raise ValueError(f"Unknown field type {spec.name}")

    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int).reshape(-1, 2)
    try:
        molblock = Chem.MolToMolBlock(mol)
    except Exception:
        molblock = ""
    return MoleculeFields(name=name or (mol.GetProp("_Name") if mol.HasProp("_Name") else ""),
                          coords=coords, weights=W, radii=radii, symbols=symbols,
                          atom_types=types, bonds=bonds, charges=charges, molblock=molblock)


# --------------------------------------------------------------------------- #
# Table of all atoms of a set of molecules (for vectorised kernel evaluation)
# --------------------------------------------------------------------------- #

@dataclass
class AtomTable:
    coords: np.ndarray      # (A, 3)
    weights: np.ndarray     # (A, F)
    radii: np.ndarray       # (A,)
    offsets: np.ndarray     # (n_mol + 1,) start index of every molecule
    names: List[str]
    molecules: List[MoleculeFields]

    @property
    def n_mol(self) -> int:
        return len(self.offsets) - 1

    def slice(self, i: int) -> slice:
        return slice(int(self.offsets[i]), int(self.offsets[i + 1]))

    def subset(self, idx: Sequence[int]) -> "AtomTable":
        return AtomTable.from_molecules([self.molecules[i] for i in idx])

    @classmethod
    def from_molecules(cls, mols: Sequence[MoleculeFields]) -> "AtomTable":
        if len(mols) == 0:
            raise ValueError("empty molecule list")
        coords = np.concatenate([m.coords for m in mols], axis=0)
        weights = np.concatenate([m.weights for m in mols], axis=0)
        radii = np.concatenate([m.radii for m in mols], axis=0)
        offsets = np.zeros(len(mols) + 1, dtype=int)
        offsets[1:] = np.cumsum([m.n_atoms for m in mols])
        return cls(coords=coords, weights=weights, radii=radii, offsets=offsets,
                   names=[m.name for m in mols], molecules=list(mols))
