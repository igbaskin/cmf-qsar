"""
Molecules inside crystals: bond and charge perception across the cell boundary (plan item M8).

A molecular crystal read from a CIF is a list of atoms in a cell. The five classical fields of
this program are defined for molecules: they need bond orders, formal charges, aromaticity and
hydrogen-bond roles, which RDKit perceives from a molecule, not from a cloud of atoms. A molecule
of a crystal is also often cut by the cell boundary, so its atoms sit at opposite faces of the
cell. This module turns the cell into its molecules and gives each atom of the cell the field
weights its molecule gives it.

The steps, each of which can fail on its own and is reported by name:

1. **Connectivity across the boundary.** Bonds from covalent radii plus a tolerance, counting the
   periodic images (:func:`cmfqsar.frameworks.periodic_bonds`). The bonded components are the
   molecules; each is made whole by moving every atom to the image that joins it to its
   neighbours. A component that joins itself in another image is a framework, not a molecule,
   and the structure is refused as a molecular crystal.
2. **Bond orders and formal charges.** RDKit's `DetermineBondOrders` (the xyz2mol method) on
   each molecule with its connectivity fixed. A molecule's total charge is not in a CIF; the
   charges 0, +1, -1, +2 and -2 are tried in that order and the first one that gives no radical
   is kept, so a chloride counter-ion becomes Cl- and its cation is then expected to be +1. The
   charges of the cell must add up to zero.
3. **Fields.** Each distinct molecule is given the fields of an ordinary molecule
   (`cmfqsar.fields.build_fields`, with the field set of a molecular data set); every copy of it
   in the cell takes those weights atom by atom. The result is a
   :class:`cmfqsar.periodic.CrystalFields` whose atoms carry molecular weights, which the periodic
   kernels of `cmfqsar.periodic` read as they read the weights of an inorganic solid.

A crystal without hydrogen atoms (common in older X-ray structures) is refused rather than
guessed: the donor field sits on the polar hydrogens, and bond orders perceived without them are
wrong for every heteroatom. So is a structure with partly occupied or overlapping sites.

:func:`molecule_identity` gives the InChI key of a perceived molecule, which is how a crystal is
matched to a compound of a data set given as SMILES: the match tests the perception itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .frameworks import BOND_TOLERANCE, periodic_bonds

#: the total charges tried for a molecule, in order
CHARGES = (0, 1, -1, 2, -2)
#: two atoms closer than this are overlapping sites of a disordered structure, in angstrom
OVERLAP = 0.6
#: the elements an organic molecule of a data set is expected to hold
ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se"}


class PerceptionError(ValueError):
    """A crystal whose molecules cannot be perceived, with the reason in the message."""


@dataclass
class CellMolecule:
    """One molecule of a cell, made whole, with its perceived bonds and charges."""
    mol: object                                  # RDKit Mol with one conformer, unwrapped
    atom_index: np.ndarray                       # indices of its atoms in the cell
    image: np.ndarray                            # (n, 3) image of each atom that makes it whole
    charge: int = 0
    smiles: str = ""
    inchikey: str = ""


def read_molecular_crystal(path: str):
    """A crystal structure file as an ASE ``Atoms``, with the checks a molecular crystal needs.

    Refused: a structure that is not periodic in three directions, one with partly occupied
    sites, and one with two atoms closer than :data:`OVERLAP` (the two positions of a disordered
    group, written out as separate atoms).
    """
    import warnings

    from ase.io import read

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms = read(path, index=0)
    if not all(atoms.pbc):
        raise PerceptionError(f"{path}: not periodic in three directions")
    occ = atoms.info.get("occupancy")
    if occ:
        for site in occ.values():
            if any(float(v) < 0.999 for v in site.values()):
                raise PerceptionError(f"{path}: partly occupied sites (a disordered structure)")
    check_overlaps(atoms)
    return atoms


def check_overlaps(atoms, limit: float = OVERLAP) -> None:
    """Refuse a structure in which two atoms, counting images, are closer than `limit`."""
    from ase.neighborlist import neighbor_list

    i, j, d = neighbor_list("ijd", atoms, limit / 2.0 + 1e-9)
    if len(i):
        k = int(np.argmin(d))
        raise PerceptionError(f"atoms {i[k]} and {j[k]} are {d[k]:.2f} A apart: overlapping "
                              f"sites, as in a disordered structure")


def molecule_components(atoms, tolerance: float = BOND_TOLERANCE):
    """(molecule per atom, image per atom that makes each molecule whole, bond list).

    Raises :class:`PerceptionError` when a component joins itself in another image, which makes
    it a framework (a coordination polymer, a covalent network) rather than a molecule.
    """
    n = len(atoms)
    bonds = periodic_bonds(atoms, tolerance)
    adj: Dict[int, List[Tuple[int, np.ndarray]]] = {i: [] for i in range(n)}
    for a, b, s in bonds:
        adj[a].append((b, np.array(s, int)))
        adj[b].append((a, -np.array(s, int)))
    image = np.zeros((n, 3), int)
    mol = np.full(n, -1, int)
    m = 0
    for start in range(n):
        if mol[start] >= 0:
            continue
        mol[start] = m
        stack = [start]
        while stack:
            a = stack.pop()
            for b, s in adj[a]:
                want = image[a] + s
                if mol[b] < 0:
                    mol[b] = m
                    image[b] = want
                    stack.append(b)
                elif not np.array_equal(image[b], want):
                    raise PerceptionError("a component of the structure is bonded to its own "
                                          "periodic image: a framework, not a molecular crystal")
        m += 1
    return mol, image, bonds


def _rdkit_molecule(symbols, coords, pairs):
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    rw = Chem.RWMol()
    for s in symbols:
        a = Chem.Atom(s)
        a.SetNoImplicit(True)
        rw.AddAtom(a)
    for a, b in pairs:
        rw.AddBond(int(a), int(b), Chem.BondType.SINGLE)
    conf = Chem.Conformer(len(symbols))
    for k, p in enumerate(coords):
        conf.SetAtomPosition(k, Point3D(*[float(v) for v in p]))
    rw.AddConformer(conf, assignId=True)
    return rw.GetMol()


def perceive(symbols: Sequence[str], coords: np.ndarray, pairs, charges=CHARGES):
    """Bond orders and formal charges of one molecule with fixed connectivity.

    Returns (molecule, total charge). The first charge of `charges` that gives a sanitised
    molecule without radical electrons is kept; :class:`PerceptionError` if none does.
    """
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds

    last = "no charge was tried"
    for q in charges:
        m = Chem.Mol(_rdkit_molecule(symbols, coords, pairs))
        try:
            rdDetermineBonds.DetermineBondOrders(m, charge=int(q), allowChargedFragments=True)
            Chem.SanitizeMol(m)
        except Exception as exc:                       # noqa: BLE001 - try the next charge
            last = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            continue
        if any(a.GetNumRadicalElectrons() for a in m.GetAtoms()):
            last = f"charge {q:+d} leaves radicals"
            continue
        Chem.AssignStereochemistryFrom3D(m)
        return m, int(q)
    raise PerceptionError(f"no bond orders found for {''.join(symbols[:6])}...: {last}")


def cell_molecules(atoms, tolerance: float = BOND_TOLERANCE, require_hydrogens: bool = True,
                   organic_only: bool = True) -> List[CellMolecule]:
    """Every molecule of a molecular crystal, made whole, with bond orders and charges.

    Molecules are in the order of their first atom in the cell. Raises
    :class:`PerceptionError` for a framework, a crystal with no hydrogens, an element outside
    :data:`ORGANIC` (with `organic_only`), a molecule whose bonds cannot be perceived, and a
    cell whose charges do not add up to zero.
    """
    symbols = atoms.get_chemical_symbols()
    if organic_only:
        odd = sorted(set(symbols) - ORGANIC)
        if odd:
            raise PerceptionError(f"elements outside the organic set: {', '.join(odd)}")
    if require_hydrogens and "C" in symbols and "H" not in symbols:
        raise PerceptionError("the structure has carbon but no hydrogen atoms: the hydrogens "
                              "were not located, and bond orders without them are guesses")
    mol_of, image, bonds = molecule_components(atoms, tolerance)
    cell = np.asarray(atoms.cell, float)
    pos = np.asarray(atoms.get_positions(), float) + image @ cell
    out: List[CellMolecule] = []
    seen: Dict[Tuple, Tuple] = {}
    for m in range(int(mol_of.max()) + 1 if len(mol_of) else 0):
        idx = np.flatnonzero(mol_of == m)
        local = {int(a): k for k, a in enumerate(idx)}
        pairs = [(local[a], local[b]) for a, b, _s in bonds if a in local and b in local]
        syms = [symbols[a] for a in idx]
        key = (tuple(syms), tuple(sorted(pairs)))
        if key in seen:                          # a copy: the same atoms and bonds in order
            template, q = seen[key]
            from rdkit import Chem
            from rdkit.Geometry import Point3D
            mol = Chem.Mol(template)
            conf = mol.GetConformer()
            for k, p in enumerate(pos[idx]):
                conf.SetAtomPosition(k, Point3D(*[float(v) for v in p]))
            Chem.AssignStereochemistryFrom3D(mol)
        else:
            mol, q = perceive(syms, pos[idx], pairs)
            seen[key] = (mol, q)
        out.append(CellMolecule(mol=mol, atom_index=idx, image=image[idx], charge=q))
    total = sum(c.charge for c in out)
    if total != 0:
        raise PerceptionError(f"the perceived charges of the cell add up to {total:+d}, not zero")
    for c in out:
        c.smiles, c.inchikey = molecule_identity(c.mol)
    return out


def molecule_identity(mol) -> Tuple[str, str]:
    """(canonical SMILES, standard InChI key) of a molecule, '' where RDKit gives none."""
    from rdkit import Chem

    try:
        smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
    except Exception:                                  # noqa: BLE001
        smi = ""
    try:
        key = Chem.MolToInchiKey(mol) or ""
    except Exception:                                  # noqa: BLE001
        key = ""
    return smi, key


def distinct_molecules(molecules: Sequence[CellMolecule]) -> Dict[str, int]:
    """How many copies of each distinct molecule (by InChI key) the cell holds."""
    out: Dict[str, int] = {}
    for c in molecules:
        out[c.inchikey] = out.get(c.inchikey, 0) + 1
    return out


def molecular_crystal_fields(atoms, fieldset, name: str = "",
                             molecules: Optional[Sequence[CellMolecule]] = None):
    """The crystal with molecular field weights on its atoms, for the periodic kernels.

    Every molecule of the cell is given the fields of an ordinary molecule with `fieldset` (the
    field set of a molecular data set: electrostatic, steric, hydrophobic, donor, acceptor), and
    those weights are put on its atoms in the cell, at their positions in the cell. The result is
    a :class:`cmfqsar.periodic.CrystalFields`, so `periodic_autocorrelation` and the periodic
    overlap kernel apply to it unchanged. Hydrogens are atoms of the cell like any other.
    """
    from ase.data import atomic_numbers, covalent_radii

    from .fields import build_fields
    from .periodic import CrystalFields

    molecules = list(molecules) if molecules is not None else cell_molecules(atoms)
    n = len(atoms)
    W = np.zeros((n, len(fieldset.specs)))
    charges = np.zeros(n)
    done = np.zeros(n, bool)
    cache: Dict[str, np.ndarray] = {}
    for c in molecules:
        key = c.inchikey or c.smiles
        if key not in cache or len(cache[key]) != len(c.atom_index):
            mf = build_fields(c.mol, fieldset, name=key)
            if mf.n_atoms != len(c.atom_index):
                raise PerceptionError(f"the fields of {c.smiles} have {mf.n_atoms} atoms and the "
                                      f"molecule {len(c.atom_index)}: hydrogens were added")
            cache[key] = (mf.weights, mf.charges if mf.charges is not None
                          else np.zeros(mf.n_atoms))
        w, q = cache[key]
        W[c.atom_index] = w
        charges[c.atom_index] = q
        done[c.atom_index] = True
    if not done.all():
        raise PerceptionError(f"{int((~done).sum())} atoms of the cell are in no molecule")
    symbols = atoms.get_chemical_symbols()
    radii = np.array([covalent_radii[atomic_numbers[s]] for s in symbols], float)
    return CrystalFields(name=name or atoms.get_chemical_formula(),
                         coords=np.asarray(atoms.get_positions(), float), weights=W,
                         radii=radii, symbols=list(symbols),
                         cell=np.asarray(atoms.cell[:], float),
                         pbc=tuple(bool(p) for p in atoms.pbc), charges=charges,
                         charge_source=fieldset.charge_model, atom_types=list(symbols))
