"""A target that belongs to an atom rather than to a molecule (plan item LP1).

Everything else in this program predicts one number per molecule.  A carbon shift, an acidity, a
site's formation energy and a local reactivity are numbers per *atom*, and a data set of them is
not a list: most atoms of a molecule carry no measurement at all, and the atoms that do are not
independent samples of each other, because they share a molecule.

This module is the data model for that, and nothing more -- there is no model here, and the item
that adds one (the local kernel) needs this first.  It holds three things:

* **reading**, with blanks.  :func:`read_atom_target` returns one number per atom and *not a
  number* where the atom has none, which is the case that separates an atom target from the
  per-atom properties a field is built from (:func:`cmfqsar.fields.atom_property`), where every
  atom must have a value.
* **a sample index.**  :class:`AtomTargets` is the list of measured (molecule, atom) pairs with
  their values, which is what a fit iterates over; the molecule each sample came from is kept
  beside it, because it is needed for every honest split.
* **folds that hold out whole molecules.**  Two shifts of one molecule in different folds would
  let a model see the molecule it is being tested on, and a per-atom q2 obtained that way is
  meaningless.  :meth:`AtomTargets.folds` splits the molecules, as the grouped folds of
  :mod:`cmfqsar.chemotypes` split the chemotypes.

The predictions go back where they came from: :func:`write_atom_predictions` writes an SDF whose
atoms carry the measured and the predicted value, so a shift sits on the carbon it belongs to.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

#: what a blank looks like in a list of numbers written by hand or by another program
BLANKS = ("", "-", "--", "na", "n/a", "nan", "none", "null", ".", "?")
#: the suffix the predictions are written under
PREDICTED_SUFFIX = "_predicted"


def _one(text: str) -> float:
    """One field of a list: a number, or not a number where it is blank."""
    s = str(text).strip()
    if s.lower() in BLANKS:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _split(text: str) -> List[str]:
    """Split a list of numbers, keeping the blanks between two separators.

    ``"1.2,,3.4"`` is three values with the middle one missing, which is how a table with a gap
    arrives; whitespace alone cannot express that, so runs of whitespace are one separator and a
    comma or semicolon is always its own.
    """
    s = str(text).strip()
    if not s:
        return []
    if "," in s or ";" in s:
        parts = re.split(r"[,;]", s)
        return [p.strip() for p in parts]
    return [p for p in re.split(r"\s+", s) if p != ""]


def _n_heavy(mol: Chem.Mol) -> int:
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)


def read_atom_target(mol: Chem.Mol, name: str, mol_name: str = "") -> np.ndarray:
    """One number per atom, *not a number* where the atom has none (plan item LP1).

    Two conventions are read, in this order: a property on the atoms themselves, where an atom
    without it is simply unmeasured, and a molecule property holding the numbers in atom order,
    where a blank field is unmeasured.  A list as long as the molecule is taken atom for atom; a
    list as long as its heavy atoms is spread over them, leaving the hydrogens unmeasured, which
    is what a file written before the hydrogens were added means; any other length is an error
    naming the molecule and both counts, because the alternative is to guess which atom a number
    belongs to.

    Returns an array of length ``mol.GetNumAtoms()``; a molecule carrying nothing comes back as
    all *not a number*, which is a molecule with no measurement rather than an error.
    """
    n = mol.GetNumAtoms()
    out = np.full(n, np.nan)
    if not n:
        return out
    have = [i for i in range(n) if mol.GetAtomWithIdx(i).HasProp(name)]
    if have:
        for i in have:
            out[i] = _one(mol.GetAtomWithIdx(i).GetProp(name))
        return out
    if not mol.HasProp(name):
        return out
    vals = [_one(p) for p in _split(mol.GetProp(name))]
    label = mol_name or (mol.GetProp("_Name") if mol.HasProp("_Name") else "a molecule")
    if len(vals) == n:
        return np.array(vals, float)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if len(vals) == len(heavy):
        for i, v in zip(heavy, vals):
            out[i] = v
        return out
    raise ValueError(
        f"{label}: the atom target '{name}' has {len(vals)} values for a molecule of {n} atoms "
        f"({len(heavy)} of them heavy). Give one value per atom, or one per heavy atom, with a "
        f"blank where the atom was not measured.")


@dataclass
class AtomTargets:
    """The measured (molecule, atom) pairs of a data set, and what each one measured."""
    molecule: np.ndarray = _field(default_factory=lambda: np.zeros(0, int))
    atom: np.ndarray = _field(default_factory=lambda: np.zeros(0, int))
    y: np.ndarray = _field(default_factory=lambda: np.zeros(0))
    n_mol: int = 0
    name: str = ""
    #: one array per molecule, with not-a-number where the atom was not measured
    per_molecule: List[np.ndarray] = _field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def n_measured_molecules(self) -> int:
        return int(len(set(self.molecule.tolist())))

    @classmethod
    def from_molecules(cls, mols: Sequence[Chem.Mol], name: str,
                       names: Optional[Sequence[str]] = None) -> "AtomTargets":
        """Read the target from every molecule and build the index of what was measured."""
        per, mol_i, atom_i, vals = [], [], [], []
        for k, m in enumerate(mols):
            label = names[k] if names is not None and k < len(names) else ""
            v = read_atom_target(m, name, mol_name=label) if m is not None else np.zeros(0)
            per.append(v)
            for i in np.where(np.isfinite(v))[0]:
                mol_i.append(k)
                atom_i.append(int(i))
                vals.append(float(v[i]))
        return cls(molecule=np.asarray(mol_i, int), atom=np.asarray(atom_i, int),
                   y=np.asarray(vals, float), n_mol=len(mols), name=name, per_molecule=per)

    def subset(self, rows: Sequence[int]) -> "AtomTargets":
        rows = np.asarray(rows, int)
        return AtomTargets(self.molecule[rows], self.atom[rows], self.y[rows], self.n_mol,
                           self.name, self.per_molecule)

    def of_molecule(self, k: int) -> np.ndarray:
        """The sample rows that belong to molecule `k`."""
        return np.where(self.molecule == int(k))[0]

    def counts(self) -> Dict[int, int]:
        """How many atoms were measured in each molecule that has any."""
        out: Dict[int, int] = {}
        for m in self.molecule.tolist():
            out[int(m)] = out.get(int(m), 0) + 1
        return out

    def folds(self, k: int = 10, seed: Optional[int] = 0) -> List[np.ndarray]:
        """Test-sample rows for `k` folds that hold out whole **molecules**.

        A molecule's atoms never straddle the split: two shifts of one molecule, one in the
        training set and one in the test set, would let the model see the molecule it is being
        tested on, and the q2 that came out would not mean anything.
        """
        from .model import _kfold_indices
        which = np.unique(self.molecule) if len(self.molecule) else np.zeros(0, int)
        if len(which) == 0:
            return []
        k = max(2, min(int(k), len(which)))
        out = []
        for part in _kfold_indices(len(which), k, seed):
            held = set(int(which[i]) for i in part)
            out.append(np.array([r for r in range(len(self))
                                 if int(self.molecule[r]) in held], dtype=int))
        return [f for f in out if len(f)]

    def describe(self) -> str:
        if not len(self):
            return f"no atom carries '{self.name}'"
        c = self.counts()
        per = np.array(list(c.values()), float)
        return (f"'{self.name}': {len(self)} atoms measured in {len(c)} of {self.n_mol} "
                f"molecules, {per.mean():.1f} per molecule (from {int(per.min())} to "
                f"{int(per.max())}); values {self.y.min():.3f} to {self.y.max():.3f}, "
                f"mean {self.y.mean():.3f}")


def scatter(targets: AtomTargets, values: Sequence[float]) -> List[np.ndarray]:
    """Spread one value per sample back over the molecules, as one array per molecule."""
    v = np.asarray(values, float)
    if len(v) != len(targets):
        raise ValueError(f"{len(v)} values for {len(targets)} samples")
    out = [np.full(len(p), np.nan) for p in targets.per_molecule]
    for r in range(len(targets)):
        out[int(targets.molecule[r])][int(targets.atom[r])] = v[r]
    return out


def write_atom_predictions(path: str, mols: Sequence[Chem.Mol], targets: AtomTargets,
                           predicted: Optional[Sequence[float]] = None,
                           names: Optional[Sequence[str]] = None) -> int:
    """Write an SDF whose atoms carry the measured value and, if given, the predicted one.

    A predicted shift belongs on the carbon it was predicted for, not in a column of a table
    beside the molecule, so it is written in RDKit's own per-atom convention, the molecule
    property ``atom.dprop.<name>`` holding one field per atom.  RDKit decodes that back into
    properties on the atoms themselves when the file is read, an atom whose field is ``n/a``
    gets nothing, and the writer does not drop it as it drops properties set on atoms directly.
    """
    spread = scatter(targets, predicted) if predicted is not None else None
    written = 0
    with Chem.SDWriter(path) as w:
        for k, m in enumerate(mols):
            if m is None:
                continue
            out = Chem.Mol(m)
            if names is not None and k < len(names):
                out.SetProp("_Name", str(names[k]))
            n = out.GetNumAtoms()
            measured = targets.per_molecule[k] if k < len(targets.per_molecule) else None
            if measured is not None and len(measured) == n:
                out.SetProp("atom.dprop." + targets.name, _as_fields(measured))
            if spread is not None and len(spread[k]) == n:
                out.SetProp("atom.dprop." + targets.name + PREDICTED_SUFFIX,
                            _as_fields(spread[k]))
            w.write(out)
            written += 1
    return written


def _as_fields(values: Sequence[float]) -> str:
    """One field per atom, with ``n/a`` where there is no value, which is RDKit's own spelling."""
    return " ".join("n/a" if not np.isfinite(v) else f"{float(v):.6g}" for v in values)


def candidate_targets(mols: Sequence[Chem.Mol], limit: int = 200) -> Dict[str, Tuple[int, int]]:
    """What in these molecules could be a per-atom target: (molecules carrying it, atoms measured).

    This is a looser rule than the one for a per-atom *field*
    (:func:`cmfqsar.fields.atom_property_names`), and it has to be: a field needs a value on every
    atom, while a target is a measurement that most atoms do not have.  A property counts here
    when it is set on at least one atom, or when it is a molecule property whose fields number
    one per atom or one per heavy atom once the blanks are counted in.
    """
    out: Dict[str, Tuple[int, int]] = {}

    def add(name: str, atoms: int) -> None:
        mols_so_far, atoms_so_far = out.get(name, (0, 0))
        out[name] = (mols_so_far + 1, atoms_so_far + atoms)

    for m in [x for x in mols if x is not None][:limit]:
        n = m.GetNumAtoms()
        if not n:
            continue
        seen = set()
        for i in range(n):
            for key in m.GetAtomWithIdx(i).GetPropNames():
                if not key.startswith("_"):
                    seen.add(key)
        for key in sorted(seen):
            v = read_atom_target(m, key)
            if np.isfinite(v).any():
                add(key, int(np.isfinite(v).sum()))
        heavy = _n_heavy(m)
        for key in m.GetPropNames():
            if key.startswith("_") or key in seen or key.startswith(("atom.", "bond.")):
                continue
            fields = _split(m.GetProp(key))
            if len(fields) not in (n, heavy) or len(fields) < 2:
                continue
            vals = np.array([_one(f) for f in fields])
            if np.isfinite(vals).any():
                add(key, int(np.isfinite(vals).sum()))
    return out
