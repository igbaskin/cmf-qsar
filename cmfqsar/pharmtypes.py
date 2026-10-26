"""Six pharmacophore features as indicator fields (plan item D2).

An indicator field puts a Gaussian of weight one on every atom of one *kind* and nothing anywhere
else, so its co-field is a map of where that kind of atom helps.  The program has had them keyed
by Sybyl atom type since the 2015 paper (`cif:C.ar`, `cif:N.am`, ...), and they work -- on sets
above about forty molecules.  Below that they do not, and the reason is arithmetic rather than
chemistry: a Sybyl vocabulary has twenty-odd types in a congeneric series, each type is rare, and
a field whose weights are zero on most molecules contributes a kernel column of mostly zeros.

The vocabulary here is six features instead of twenty-odd:

===========  ==========================================================================
`donor`      an N-H, O-H or S-H that can donate a hydrogen bond
`acceptor`   an acceptor lone pair
`aromatic`   an aromatic ring, every atom of it
`hydrophobe` a hydrophobic atom or a lumped hydrophobic group, such as a ring face
`cation`     an ionisable basic group, protonated at physiological pH
`anion`      an ionisable acidic group
===========  ==========================================================================

Each is common enough to appear in most molecules of a series, so each field has something to say
about every molecule; and the vocabulary is the one a medicinal chemist already uses, so the
co-field of `pharm:donor` *is* a statement about where a donor is wanted, with no translation
step.  :mod:`cmfqsar.pharmacophore` knows the six names, so a pharmacophore extracted from such a
model comes out with the right feature types rather than with a field name.

Where the definitions come from
-------------------------------
RDKit's ``BaseFeatures.fdef``, the Gobbi-Poppinger definitions that ship with it, rather than
SMARTS written here: they are the standard set, they are what anyone comparing this with another
program will have, and a definition this program invented would have to be defended.  The eight
families of that file map onto the six above, with ``Hydrophobe`` and ``LumpedHydrophobe`` joined
and ``ZnBinder`` dropped.

An atom belongs to a feature or it does not -- the weight is one or zero, never two, even where
the file gives an atom to both ``Hydrophobe`` and ``LumpedHydrophobe``.  Where a family names a
whole group, every atom of the group gets the weight: an aromatic ring is six atoms of `aromatic`,
a carboxylate is three atoms of `anion`.  That is RDKit's grouping, and for a field it is the
right one, since the character of a carboxylate is not on one of its atoms.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np

#: the vocabulary, in the order the fields are created
PHARM_FEATURES = ("donor", "acceptor", "aromatic", "hydrophobe", "cation", "anion")

#: RDKit feature family -> the name used here; ZnBinder is not a pharmacophore feature of a
#: ligand series and is left out
FAMILY_MAP = {
    "Donor": "donor",
    "Acceptor": "acceptor",
    "Aromatic": "aromatic",
    "Hydrophobe": "hydrophobe",
    "LumpedHydrophobe": "hydrophobe",
    "PosIonizable": "cation",
    "NegIonizable": "anion",
}

#: SMARTS used only when RDKit's feature definitions cannot be read, so that the fields still
#: work rather than failing: deliberately coarse, and reported as such by :func:`definitions_used`
FALLBACK_SMARTS = {
    "donor": ["[$([N;!H0]),$([O;H1]),$([S;H1])]"],
    "acceptor": ["[$([O;!$(*-[#7,#15,#16]=O)]),$([n;X2]),$([N;X1]#*),"
                 "$([N;X2;!$(N-[!#6])]=*)]"],
    "aromatic": ["a"],
    "hydrophobe": ["[C;!$(C=O);!$(C-[N,O,S])]", "[F,Cl,Br,I]"],
    "cation": ["[$([N;H2,H1,H0;!$(N-[!#6]);!$(N=*);!$(N#*);!$(N-C=[O,N,S])]),$([+1,+2,+3])]"],
    "anion": ["[$([C,S](=[O,S,P])-[O;H1,-1]),$([P](=O)([O;H1,-1])),$([-1,-2,-3])]"],
}

_FACTORY = None
_FACTORY_TRIED = False
_SMARTS_CACHE: Dict[str, list] = {}


def _factory():
    """RDKit's feature factory, or None when its definitions cannot be read."""
    global _FACTORY, _FACTORY_TRIED
    if _FACTORY_TRIED:
        return _FACTORY
    _FACTORY_TRIED = True
    try:
        from rdkit import RDConfig
        from rdkit.Chem import ChemicalFeatures
        path = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
        if os.path.isfile(path):
            _FACTORY = ChemicalFeatures.BuildFeatureFactory(path)
    except Exception:                       # noqa: BLE001 - an RDKit without the data files
        _FACTORY = None
    return _FACTORY


def definitions_used() -> str:
    """Which definitions the fields are being built from, for a log line or a model file."""
    return "RDKit BaseFeatures.fdef" if _factory() is not None else "built-in SMARTS (fallback)"


def _smarts(name: str):
    from rdkit import Chem
    if name not in _SMARTS_CACHE:
        _SMARTS_CACHE[name] = [Chem.MolFromSmarts(s) for s in FALLBACK_SMARTS.get(name, [])]
    return [p for p in _SMARTS_CACHE[name] if p is not None]


def feature_atoms(mol) -> Dict[str, np.ndarray]:
    """{feature: boolean array over the atoms} for one molecule.

    An atom that several features of one family name is still weight one: these are indicators,
    not counts, and an atom that RDKit lists as both a hydrophobe and part of a lumped
    hydrophobic group is one hydrophobe.
    """
    n = mol.GetNumAtoms()
    out = {name: np.zeros(n, bool) for name in PHARM_FEATURES}
    fac = _factory()
    if fac is not None:
        for feat in fac.GetFeaturesForMol(mol):
            name = FAMILY_MAP.get(feat.GetFamily())
            if name is None:
                continue
            for i in feat.GetAtomIds():
                if 0 <= i < n:
                    out[name][i] = True
        return out
    for name in PHARM_FEATURES:
        for patt in _smarts(name):
            for match in mol.GetSubstructMatches(patt):
                for i in match:
                    if 0 <= i < n:
                        out[name][i] = True
    return out


def pharm_weights(mol, feature: str) -> np.ndarray:
    """The weight column of one pharmacophore indicator field: one on its atoms, zero elsewhere."""
    if feature not in PHARM_FEATURES:
        raise ValueError(f"unknown pharmacophore feature {feature!r}; "
                         f"use one of {PHARM_FEATURES}")
    return feature_atoms(mol)[feature].astype(float)


def features_in(mols: Sequence, min_molecules: int = 5) -> List[str]:
    """Which of the six occur in at least `min_molecules` molecules of a set.

    The same rule :func:`cmfqsar.fields.cif_types_in` applies to Sybyl types, and the reason the
    item exists: a feature carried by two molecules of thirty is a kernel column of zeros.
    """
    counts = {name: 0 for name in PHARM_FEATURES}
    for m in mols:
        if m is None:
            continue
        got = feature_atoms(m)
        for name, mask in got.items():
            if mask.any():
                counts[name] += 1
    return [name for name in PHARM_FEATURES if counts[name] >= int(min_molecules)]


def coverage(mols: Sequence) -> Dict[str, Dict[str, float]]:
    """Per feature: the share of molecules that carry it and the mean atoms per molecule.

    This is the measurement the item is an argument about, so it is worth being able to print:
    a Sybyl vocabulary has many types each carried by few molecules, and these six are meant to
    be carried by most.
    """
    n_mol = 0
    present = {name: 0 for name in PHARM_FEATURES}
    atoms = {name: 0 for name in PHARM_FEATURES}
    for m in mols:
        if m is None:
            continue
        n_mol += 1
        got = feature_atoms(m)
        for name, mask in got.items():
            k = int(mask.sum())
            atoms[name] += k
            if k:
                present[name] += 1
    if not n_mol:
        return {}
    return {name: {"molecules": present[name] / n_mol,
                   "atoms_per_molecule": atoms[name] / n_mol} for name in PHARM_FEATURES}
