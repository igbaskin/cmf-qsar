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
pharm:<feature> the same for one of six pharmacophore features (donor, acceptor, aromatic,
                hydrophobe, cation, anion); see cmfqsar/pharmtypes.py
                (Sitnikov et al. 2015, Eq. 6)
"""
from __future__ import annotations

import re

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

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
        if self.name.startswith("pharm:"):
            return self.name[6:].capitalize()
        if self.name.startswith("prop:"):
            return self.name[5:]
        return FIELD_LABELS.get(self.name, self.name)

    @property
    def is_cif(self) -> bool:
        return self.name.startswith("cif:")

    @property
    def cif_type(self) -> Optional[str]:
        return self.name[4:] if self.is_cif else None

    @property
    def is_pharm(self) -> bool:
        """An indicator field of one of the six pharmacophore features (plan item D2)."""
        return self.name.startswith("pharm:")

    @property
    def pharm_feature(self) -> Optional[str]:
        return self.name[6:] if self.is_pharm else None

    @property
    def is_prop(self) -> bool:
        """A field whose atomic weights are read from a property of the molecule."""
        return self.name.startswith("prop:")

    @property
    def prop(self) -> Optional[str]:
        return self.name[5:] if self.is_prop else None

    @property
    def is_probe(self) -> bool:
        """A GRID-like probe field: every atom carries several Gaussians, not one."""
        return self.name.startswith("probe:")

    @property
    def probe(self) -> Optional[str]:
        return self.name[6:] if self.is_probe else None


@dataclass
class FieldSet:
    """The complete set of field types used by a model plus parameterisation options."""
    specs: List[FieldSpec] = field(default_factory=list)
    charge_model: str = "gasteiger"     # "gasteiger" | "mmff" | "file"
    cif_scheme: str = "sybyl"           # "sybyl" | "element" | "element+aromatic"
    param_scheme: str = "rdkit"         # "rdkit" | "tripos" | "tripos-legacy"
    # what to do when a molecule does not carry a per-atom property a field asks for:
    # "error" says so and names it, "zero" treats the molecule as having none of that quantity
    prop_missing: str = "error"         # "error" | "zero"
    # molecule properties used as non-field descriptors: one number per molecule, which
    # enters the kernel as a block of its own (see cmfqsar/descriptors.py)
    descriptors: Tuple[str, ...] = ()
    descriptor_h: float = 1.0           # the block's mixing coefficient
    #: Gaussians per atom in a probe field; the kernel costs the square of this
    probe_terms: int = 4
    # --- conformations: the fourth dimension (4D-QSAR) --------------------------
    # "single" uses one conformation per molecule, as the papers do, and is the default.
    # "ensemble" represents a molecule by the weighted mean of the fields of all of its
    # conformations, which is the occupancy averaging of 4D-QSAR (Hopfinger et al. 1997).
    conformers: str = "single"          # "single" | "ensemble"
    conformer_weighting: str = "uniform"   # "uniform" | "boltzmann" | "property"
    conformer_kt: float = 0.6           # kcal/mol, about RT at 300 K, for Boltzmann weights
    max_conformers: int = 0             # 0 = use every conformation present
    energy_prop: str = "conformer_energies"   # molecule property holding per-conformer energies
    weight_prop: str = "conformer_weights"    # molecule property holding weights, for "given"

    @property
    def is_ensemble(self) -> bool:
        return self.conformers == "ensemble"

    # ----- construction helpers -------------------------------------------------
    @classmethod
    def create(cls, physchem: Sequence[str] = PHYSCHEM_FIELDS, cif_types: Sequence[str] = (),
               alpha: float = 1.0, charge_model: str = "gasteiger", cif_scheme: str = "sybyl",
               param_scheme: str = "rdkit", conformers: str = "single",
               conformer_weighting: str = "uniform", conformer_kt: float = 0.6,
               max_conformers: int = 0, atom_properties: Sequence[str] = (),
               prop_missing: str = "error", descriptors: Sequence[str] = (),
               descriptor_h: float = 1.0, probes: Sequence[str] = (),
               probe_terms: int = 4,
               pharm_features: Sequence[str] = ()) -> "FieldSet":
        specs = []
        for name in physchem:
            kind = "steric" if name == "steric" else "gauss"
            specs.append(FieldSpec(name=name, kind=kind, alpha=alpha))
        for t in cif_types:
            specs.append(FieldSpec(name=f"cif:{t}", kind="gauss", alpha=alpha))
        for p in pharm_features:
            # an indicator field of a pharmacophore feature rather than of an atom type: six
            # of them instead of twenty-odd, each carried by most molecules of a series
            specs.append(FieldSpec(name=f"pharm:{p}", kind="gauss", alpha=alpha))
        for pr in probes:
            # a GRID-like field: what a probe would gain by sitting at each point. Its alpha
            # is unused -- the widths come from the expansion in cmfqsar/probes.py
            specs.append(FieldSpec(name=f"probe:{pr}", kind="probe", alpha=alpha))
        for q in atom_properties:
            # a field of whatever the file says about every atom: QM charges, a Fukui index,
            # a local softness, anything with one number per atom
            specs.append(FieldSpec(name=f"prop:{q}", kind="gauss", alpha=alpha))
        return cls(specs=specs, charge_model=charge_model, cif_scheme=cif_scheme,
                   param_scheme=param_scheme, conformers=conformers,
                   conformer_weighting=conformer_weighting, conformer_kt=conformer_kt,
                   max_conformers=max_conformers, prop_missing=prop_missing,
                   descriptors=tuple(descriptors), descriptor_h=float(descriptor_h),
                   probe_terms=int(probe_terms))

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
        """The whole set, with the specs themselves copied so they can be retuned separately.

        Every field of the dataclass is carried over by asking the dataclass what its fields are,
        not by listing them here: a copy written by hand quietly drops whatever is added to the
        class afterwards, and a setting lost on the way to the model -- a descriptor, a probe
        width -- shows up as nothing at all, only as a model that is not the one that was asked for.
        """
        return dataclasses.replace(self, specs=[FieldSpec(**vars(s)) for s in self.specs])


# --------------------------------------------------------------------------- #
# Atom-level parameterisation
# --------------------------------------------------------------------------- #

class MissingAtomProperty(ValueError):
    """A molecule does not carry a per-atom property that a field asks for."""


def atom_property_names(mol: Chem.Mol, min_atoms: int = 2) -> List[str]:
    """Names of the per-atom properties a molecule carries, in either convention.

    An atom property is one that every atom has (RDKit keeps these and writes them to SDF as
    ``atom.dprop.<name>``), or a molecule property holding one number per atom, which is how
    most quantum-chemistry packages dump a Mulliken charge or a Fukui index.
    """
    out: List[str] = []
    n = mol.GetNumAtoms()
    if n:
        first = mol.GetAtomWithIdx(0)
        for key in first.GetPropNames():
            if key.startswith("_"):
                continue
            try:
                if all(mol.GetAtomWithIdx(i).HasProp(key) for i in range(n)):
                    float(first.GetProp(key))
                    out.append(key)
            except (ValueError, TypeError):
                continue
    for key in mol.GetPropNames():
        # RDKit decodes "atom.dprop.<name>" into atom properties on reading, so the raw tag is
        # the same numbers again under a second name
        if key.startswith("_") or key in out or key.startswith(("atom.", "bond.")):
            continue
        vals = _as_numbers(mol.GetProp(key))
        if vals is not None and len(vals) >= min_atoms and len(vals) in (n, _n_heavy(mol)):
            out.append(key)
    return out


def _n_heavy(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def _as_numbers(text: str) -> Optional[np.ndarray]:
    """A whitespace-, comma- or newline-separated list of numbers, or None."""
    parts = [p for p in re.split(r"[\s,;]+", str(text).strip()) if p]
    if not parts:
        return None
    try:
        return np.array([float(p) for p in parts], dtype=float)
    except ValueError:
        return None


def atom_property(mol: Chem.Mol, name: str, n_atoms: Optional[int] = None,
                  policy: str = "error", mol_name: str = "") -> np.ndarray:
    """One number per atom, read from the molecule.

    Two conventions are understood, in this order: a property set on every atom (RDKit's own,
    which survives a round trip through SDF as ``atom.dprop.<name>``), and a molecule property
    holding the numbers in atom order, separated by whitespace, commas or semicolons.

    The hydrogens are the usual source of error.  A structure file without 3D coordinates is embedded on loading
    and hydrogens are added, so a list written for the heavy atoms is shorter than the molecule
    is now.  A list as long as the molecule is taken atom for atom; a list as long as the heavy
    atoms is spread over them in order, leaving the hydrogens at zero; anything else is an error
    naming the molecule and both counts, because the alternative is to guess which atom a number
    belongs to.
    """
    n = int(n_atoms if n_atoms is not None else mol.GetNumAtoms())
    label = mol_name or (mol.GetProp("_Name") if mol.HasProp("_Name") else "a molecule")
    vals: Optional[np.ndarray] = None
    if n and all(mol.GetAtomWithIdx(i).HasProp(name) for i in range(n)):
        try:
            vals = np.array([float(mol.GetAtomWithIdx(i).GetProp(name)) for i in range(n)])
        except ValueError as exc:
            raise MissingAtomProperty(
                f"the atom property {name!r} of {label} is not numeric ({exc})") from None
    elif mol.HasProp(name):
        vals = _as_numbers(mol.GetProp(name))
        if vals is None:
            raise MissingAtomProperty(
                f"the property {name!r} of {label} is not a list of numbers; a per-atom field "
                f"needs one number per atom, separated by spaces, commas or newlines")
    if vals is None:
        if policy == "zero":
            return np.zeros(n)
        raise MissingAtomProperty(
            f"{label} carries no per-atom property called {name!r}. Set it on every atom (RDKit "
            f"writes those to SDF as 'atom.dprop.{name}') or give the molecule a property with "
            f"one number per atom; pass prop_missing='zero' to treat it as absent instead.")
    if len(vals) == n:
        return vals.astype(float)
    heavy = [i for i in range(n) if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    if len(vals) == len(heavy):
        out = np.zeros(n)
        out[heavy] = vals
        return out
    raise MissingAtomProperty(
        f"{label} has {n} atoms ({len(heavy)} of them heavy) but the property {name!r} holds "
        f"{len(vals)} numbers; a per-atom field needs one for every atom, or one for every heavy "
        f"atom with the hydrogens left out")


def _cap_dummies(mol: Chem.Mol) -> Chem.Mol:
    """A copy with every attachment point turned into a hydrogen, or the molecule itself.

    A fragment cut off a molecule carries a dummy atom where the bond was, and neither Gasteiger
    nor MMFF can type one: Gasteiger returns nan for *every* atom of such a molecule, which the
    caller then reads as zero, so a fragment-based model -- a topomer model, or the design loop
    of cmfqsar/design.py -- silently loses its whole electrostatic field. Capping the open
    valence with hydrogen is what the charges of a fragment mean anyway: those it would have if
    what it was cut from were a hydrogen.
    """
    if not any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()):
        return mol
    work = Chem.RWMol(Chem.Mol(mol))
    for a in work.GetAtoms():
        if a.GetAtomicNum() == 0:
            a.SetAtomicNum(1)
            a.SetIsotope(0)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetFormalCharge(0)
    out = work.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:                            # noqa: BLE001 - keep the original rather than
        return mol                               # a copy that is not valid
    return out


def _partial_charges(mol: Chem.Mol, model: str) -> Tuple[np.ndarray, str]:
    """The partial charges and where they came from: "file", "mmff" or "gasteiger".

    The source is what was *used*, not what was asked for: a molecule without charges in its
    file gets Gasteiger charges, and the source says so, which is how the contract check of
    ``cmfqsar.contract`` tells a model fitted on file charges that a new molecule has none.
    """
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
            return q, "file"
        model = "gasteiger"  # fall back
    capped = _cap_dummies(mol)
    if model == "mmff":
        try:
            # work on a copy: MMFF set-up changes the aromaticity flags of the molecule in place
            props = AllChem.MMFFGetMoleculeProperties(Chem.Mol(capped))
            if props is not None:
                return (np.array([props.GetMMFFPartialCharge(i) for i in range(n)], dtype=float),
                        "mmff")
        except Exception:
            pass
        model = "gasteiger"
    AllChem.ComputeGasteigerCharges(capped)
    q = np.array([float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
                  for a in capped.GetAtoms()])
    q[~np.isfinite(q)] = 0.0
    return q, "gasteiger"


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
    # set only for an ensemble (4D) field: its atoms are those of n_conformers conformations,
    # one after another, each already scaled by its weight
    n_conformers: int = 1
    conformer_weights: Optional[np.ndarray] = None
    atoms_per_conformer: int = 0
    #: raw (unstandardised) non-field descriptors, in the order of FieldSet.descriptors
    descriptors: Optional[np.ndarray] = None
    #: per-atom Gaussian expansion coefficients of every probe field, (N, n_terms)
    probe_coeffs: Dict[str, np.ndarray] = field(default_factory=dict)
    #: where the partial charges came from: "file", "mmff", "gasteiger", or "" when no field
    #: needed them; recorded so that a model can tell how its molecules were prepared
    charge_source: str = ""

    @property
    def n_atoms(self) -> int:
        return len(self.coords)

    @property
    def is_ensemble(self) -> bool:
        return self.n_conformers > 1

    def per_atom(self, values: np.ndarray) -> np.ndarray:
        """Sum a per-atom quantity of an ensemble field back onto the atoms of the molecule.

        Atom contributions, Shapley values and the like come out per atom *per conformation*;
        a chemist wants them per atom, which is the sum over the conformations.
        """
        v = np.asarray(values, float)
        if not self.is_ensemble or self.atoms_per_conformer <= 0:
            return v
        return v.reshape(self.n_conformers, self.atoms_per_conformer).sum(axis=0)


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


def conformer_weights(mol: Chem.Mol, fieldset: FieldSet) -> np.ndarray:
    """Weights of the conformations of one molecule, summing to one.

    ``uniform`` gives every conformation the same weight.  ``boltzmann`` and ``property`` read
    relative energies in kcal/mol from the molecule property named by ``energy_prop``, which the
    conformer generator writes, and turn them into exp(-dE / kT) normalised.  ``given`` reads the
    weights themselves from the property named by ``weight_prop``, which is how learned weights
    travel with a molecule: the multi-instance fit writes them there and everything downstream,
    the display and the co-fields included, then uses exactly the weighting the model was fitted
    with.  If anything is missing or unusable the weights fall back to uniform, because a silently
    wrong weighting would be worse than an even one.
    """
    k = mol.GetNumConformers()
    if k <= 1:
        return np.ones(max(k, 1))
    keep = k if not fieldset.max_conformers else min(k, int(fieldset.max_conformers))
    w = np.ones(k)
    if fieldset.conformer_weighting == "given":
        prop = getattr(fieldset, "weight_prop", "conformer_weights")
        try:
            if not mol.HasProp(prop):
                raise ValueError("no weights")
            g = np.array([float(x) for x in mol.GetProp(prop).replace(";", ",").split(",")], float)
            if len(g) != k or not np.isfinite(g).all() or g.min() < 0 or g.sum() <= 0:
                raise ValueError("the stored weights do not match the conformations")
            w = g
        except Exception:                        # noqa: BLE001 - documented fallback
            w = np.ones(k)
    elif fieldset.conformer_weighting in ("boltzmann", "property"):
        prop = fieldset.energy_prop
        try:
            if not mol.HasProp(prop):
                raise ValueError("no energies")
            e = np.array([float(x) for x in mol.GetProp(prop).replace(";", ",").split(",")], float)
            if len(e) != k or not np.isfinite(e).all():
                raise ValueError("energy count does not match the conformations")
            kt = max(float(fieldset.conformer_kt), 1e-6)
            w = np.exp(-(e - e.min()) / kt)
        except Exception:                        # noqa: BLE001 - documented fallback
            w = np.ones(k)
    order = np.argsort(-w)[:keep]                # keep the conformations with the largest weights
    out = np.zeros(k)
    out[order] = w[order]
    total = out.sum()
    return out / total if total > 1e-300 else np.full(k, 1.0 / k)


def ensemble_fields(mol: Chem.Mol, fieldset: FieldSet, name: str = "") -> MoleculeFields:
    """Fields of a conformational ensemble: the weighted mean of the fields of its conformations.

    The fourth dimension of 4D-QSAR.  Because a CMF field is a sum of atom-centred Gaussians and
    the kernel is bilinear in the fields, the mean field of an ensemble is itself a field whose
    atoms are the atoms of every conformation with their weights multiplied by that
    conformation's weight.  The kernel of two ensembles is then automatically the weighted mean
    of the kernels of their conformation pairs, which is the mean-map kernel of the two
    conformational distributions, and it stays positive definite.  Everything downstream, the
    co-fields and the decomposition rules included, continues to apply unchanged, because the
    model is still linear in this mean field.

    The cost is quadratic in the number of conformations kept, so ``max_conformers`` is worth
    setting on large series.
    """
    weights = conformer_weights(mol, fieldset)
    use = [i for i, w in enumerate(weights) if w > 0]
    if len(use) <= 1:
        one = Chem.Mol(mol)
        if use and mol.GetNumConformers() > 1:
            keep = mol.GetConformer(use[0])
            one.RemoveAllConformers()
            one.AddConformer(keep, assignId=True)
        return build_fields(one, fieldset, name=name, _single=True)
    base = build_fields(mol, fieldset, name=name, _single=True)      # conformer 0, for the weights
    n = base.n_atoms
    coords, W, bonds = [], [], []
    for j, i in enumerate(use):
        coords.append(np.array(mol.GetConformer(i).GetPositions(), dtype=float))
        W.append(base.weights * float(weights[i]))
        bonds.append(base.bonds + j * n)
    k = len(use)
    out = MoleculeFields(
        name=base.name, coords=np.concatenate(coords, axis=0), weights=np.concatenate(W, axis=0),
        radii=np.tile(base.radii, k), symbols=list(base.symbols) * k,
        atom_types=list(base.atom_types) * k,
        bonds=np.concatenate(bonds, axis=0) if bonds else np.zeros((0, 2), int),
        charges=np.tile(base.charges, k), molblock=base.molblock,
        charge_source=base.charge_source)
    out.n_conformers = k
    out.conformer_weights = np.array([weights[i] for i in use], float)
    out.atoms_per_conformer = n
    return out


def build_fields(mol: Chem.Mol, fieldset: FieldSet, name: str = "",
                 _single: bool = False) -> MoleculeFields:
    """Compute atom coordinates and per-field weights for one molecule.

    With ``conformers="ensemble"`` in the field set and a molecule that carries more than one
    conformation, this returns the ensemble field instead; see :func:`ensemble_fields`.
    """
    if (not _single and getattr(fieldset, "conformers", "single") == "ensemble"
            and mol.GetNumConformers() > 1):
        return ensemble_fields(mol, fieldset, name=name)
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
    probes = [s for s in fieldset.specs if s.is_probe]
    # a probe field weighs dispersion, electrostatics and hydrogen bonding together, so it
    # needs the charges and the donor/acceptor character whether or not those fields are on
    charges, charge_source = (_partial_charges(mol, fieldset.charge_model)
                              if ("electrostatic" in need or probes) else (np.zeros(n), ""))
    hydro = acc = don = None
    pharm = None            # the six pharmacophore features, assigned once if any field asks
    want_hb = bool(need & {"hydrophobic", "hba", "hbd"}) or bool(probes)
    if scheme.startswith("tripos"):
        if want_hb:
            hydro, acc, don = _tripos_weights(syb, _hydrogen_counts(mol, legacy=(scheme == "tripos-legacy")))
    else:
        if "hydrophobic" in need or probes:
            hydro = _hydrophobic_weights(mol)
        if want_hb:
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
        elif spec.is_pharm:
            if pharm is None:
                from .pharmtypes import feature_atoms
                pharm = feature_atoms(mol)
            got = pharm.get(spec.pharm_feature)
            if got is None:
                from .pharmtypes import PHARM_FEATURES
                raise ValueError(f"unknown pharmacophore feature {spec.pharm_feature!r} in "
                                 f"field {spec.name!r}; use one of {PHARM_FEATURES}")
            W[:, k] = got.astype(float)
        elif spec.is_prop:
            W[:, k] = atom_property(mol, spec.prop, n,
                                    policy=getattr(fieldset, "prop_missing", "error"),
                                    mol_name=name)
        elif spec.is_probe:
            # the weight column is unused for a probe field: its atoms carry several
            # Gaussians each, which are kept in probe_coeffs below
            W[:, k] = 0.0
        else:
            raise ValueError(f"Unknown field type {spec.name}")

    bonds = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype=int).reshape(-1, 2)
    try:
        molblock = Chem.MolToMolBlock(mol)
    except Exception:
        molblock = ""
    probe_coeffs = {}
    if probes:
        from .probes import coefficients as _probe_coeffs, probe as _probe
        hb_char = np.zeros(n)
        if don is not None:
            hb_char = hb_char + np.asarray(don, float)        # donors count positive
        if acc is not None:
            hb_char = hb_char - np.asarray(acc, float)        # acceptors negative
        hb_char = np.clip(hb_char, -1.0, 1.0)
        for spec in probes:
            probe_coeffs[spec.name] = _probe_coeffs(
                symbols, charges, hb_char, _probe(spec.probe),
                n_terms=getattr(fieldset, "probe_terms", 4))
    desc = None
    if getattr(fieldset, "descriptors", ()):
        from .descriptors import read_values
        desc = read_values(mol, fieldset.descriptors)
    return MoleculeFields(name=name or (mol.GetProp("_Name") if mol.HasProp("_Name") else ""),
                          coords=coords, weights=W, radii=radii, symbols=symbols,
                          atom_types=types, bonds=bonds, charges=charges, molblock=molblock,
                          descriptors=desc, probe_coeffs=probe_coeffs,
                          charge_source=charge_source)


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

    @property
    def probe_coeffs(self) -> Dict[str, np.ndarray]:
        """(A, n_terms) expansion coefficients of every probe field, over all atoms."""
        if not self.molecules or not self.molecules[0].probe_coeffs:
            return {}
        names = list(self.molecules[0].probe_coeffs)
        return {k: np.concatenate([m.probe_coeffs[k] for m in self.molecules], axis=0)
                for k in names}

    @property
    def descriptors(self) -> Optional[np.ndarray]:
        """(n_mol, n_desc) of raw descriptor values, or None when the field set has none."""
        if not self.molecules or self.molecules[0].descriptors is None:
            return None
        return np.vstack([m.descriptors for m in self.molecules])

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
