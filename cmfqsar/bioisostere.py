"""Which groups are interchangeable here? A bioisostere table weighted by the co-field (MC1).

`cmf design` ranks candidate substituents by the predicted activity of the whole molecule.  The
table a medicinal chemist asks for answers a different question: **which groups can replace each
other at this position**, so that one of them can be chosen for a property the model knows
nothing about -- solubility, cost, a patent, a metabolic liability.

The model answers it without being refitted once per group.  The prediction of the overlap kernel
is a sum over the atoms of the molecule (decomposition rule 3,
:meth:`cmfqsar.model.CMFModel.atom_contributions`), so the part of the prediction that belongs to
a substituent is the sum over that substituent's own atoms, and the predicted effect of exchanging
group A for group B at one position is the difference of those two sums.  That is one integral per
pair of groups rather than one model fit per group.

Three distances between two groups
----------------------------------
For every pair of substituents at the same position this module computes:

* **the fingerprint distance**, one minus the Tanimoto similarity of their Morgan fingerprints:
  two dimensions, no geometry, no model;
* **the field distance**, the distance between the two posed groups in the field metric of the
  model's own kernel, restricted to the atoms of the groups themselves: geometry, but no model;
* **the predicted change**, the difference of the two atom-contribution sums: geometry and the
  model's co-field, since the contribution of an atom is its field weighted by the co-field.

The gate of plan item MC1 is whether the third ranks the measured activity difference of a matched
molecular pair better than the first two.  If it does not, then what the table shows is shape
similarity and needs no model at all, which would be a result of the same kind as plan item D5.
`benchmarks/run_bioisostere.py` is that measurement.

Finding the pairs
-----------------
The pairs come from RDKit's `rdMMPA`, which cuts one single bond at a time and reports the two
pieces.  Every cut is taken in both directions, so each piece is tried as the core; two molecules
whose cores are the same molecule and whose groups are different are a matched pair at that
position.  Nothing is chosen by eye.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet, MoleculeFields
from .kernels import combine_kernels, kernel_matrices

#: atom map numbers start here, so that they never collide with rdMMPA's attachment labels 1..3
OFFSET = 10
#: a substituent larger than this is not a substituent but a second half of the molecule
MAX_SUBSTITUENT = 12
#: the core has to be at least this share of the molecule
MIN_CORE_SHARE = 0.5


# --------------------------------------------------------------------------- #
# Matched pairs
# --------------------------------------------------------------------------- #


@dataclass
class Substituent:
    """One group of one molecule, with the core it hangs on."""
    molecule: int
    core: str
    group: str
    atoms: np.ndarray

    def __str__(self) -> str:
        return f"{self.group} on {self.core}"


@dataclass
class MatchedPair:
    """Two molecules that differ in one group at one position."""
    a: int
    b: int
    core: str
    groups: Tuple[str, str]
    atoms_a: np.ndarray
    atoms_b: np.ndarray
    delta: float
    names: Tuple[str, str] = ("", "")

    def __str__(self) -> str:
        return (f"{self.names[0]} -> {self.names[1]}: {self.groups[0]} -> {self.groups[1]}, "
                f"measured {self.delta:+.2f}")


def _stripped(frag) -> str:
    """The SMILES of a fragment with the atom numbering taken off, the attachment point kept."""
    from rdkit import Chem
    f = Chem.Mol(frag)
    for a in f.GetAtoms():
        a.SetAtomMapNum(1 if a.GetAtomicNum() == 0 else 0)
    return Chem.MolToSmiles(f)


def fragment(mol, max_substituent: int = MAX_SUBSTITUENT,
             min_core: float = MIN_CORE_SHARE) -> List[Substituent]:
    """Every way of cutting one single bond of `mol`, as (core, group, the group's atoms).

    Both directions of every cut are reported, because which piece is the core is decided by
    what the other molecules of the set have in common and not by this molecule.  The atom
    indices refer to `mol` as it was passed in, which is what lets the group be masked out of the
    molecule's fields afterwards.
    """
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    work = Chem.Mol(mol)
    for a in work.GetAtoms():
        a.SetAtomMapNum(a.GetIdx() + OFFSET)
    n = work.GetNumHeavyAtoms()
    out: List[Substituent] = []
    for _core, chains in rdMMPA.FragmentMol(work, maxCuts=1, resultsAsMols=True):
        if chains is None:
            continue
        parts = Chem.GetMolFrags(chains, asMols=True, sanitizeFrags=False)
        if len(parts) != 2:
            continue
        for i, j in ((0, 1), (1, 0)):
            idx = [a.GetAtomMapNum() - OFFSET for a in parts[j].GetAtoms()
                   if a.GetAtomMapNum() >= OFFSET]
            if not idx or len(idx) > int(max_substituent):
                continue
            if (n - len(idx)) < float(min_core) * n:
                continue
            out.append(Substituent(molecule=-1, core=_stripped(parts[i]),
                                   group=_stripped(parts[j]),
                                   atoms=np.asarray(sorted(idx), int)))
    return out


def substituents(mols: Sequence, **kw) -> List[Substituent]:
    """:func:`fragment` over a set, with the molecule index filled in."""
    out: List[Substituent] = []
    for k, m in enumerate(mols):
        if m is None:
            continue
        for s in fragment(m, **kw):
            s.molecule = k
            out.append(s)
    return out


def matched_pairs(mols: Sequence, y: Sequence[float], names: Optional[Sequence[str]] = None,
                  max_substituent: int = MAX_SUBSTITUENT, min_core: float = MIN_CORE_SHARE,
                  one_per_couple: bool = True) -> List[MatchedPair]:
    """Every matched molecular pair of a set: two molecules, one core, two different groups.

    `delta` is the activity of the second molecule minus that of the first.  With
    `one_per_couple`, two molecules that can be matched on several cores are reported once, on
    the largest core, because the same two measured numbers must not enter a correlation twice.
    """
    y = np.asarray(y, float)
    by_core: Dict[str, List[Substituent]] = defaultdict(list)
    for s in substituents(mols, max_substituent=max_substituent, min_core=min_core):
        if np.isfinite(y[s.molecule]):
            by_core[s.core].append(s)

    found: Dict[Tuple[int, int], MatchedPair] = {}
    out: List[MatchedPair] = []
    for core, rows in by_core.items():
        for p in range(len(rows)):
            for q in range(p + 1, len(rows)):
                a, b = rows[p], rows[q]
                if a.molecule == b.molecule or a.group == b.group:
                    continue
                i, j = (a, b) if a.molecule < b.molecule else (b, a)
                pair = MatchedPair(a=i.molecule, b=j.molecule, core=core,
                                   groups=(i.group, j.group), atoms_a=i.atoms, atoms_b=j.atoms,
                                   delta=float(y[j.molecule] - y[i.molecule]),
                                   names=(names[i.molecule] if names else str(i.molecule),
                                          names[j.molecule] if names else str(j.molecule)))
                if not one_per_couple:
                    out.append(pair)
                    continue
                key = (pair.a, pair.b)
                if key not in found or len(core) > len(found[key].core):
                    found[key] = pair
    return out if not one_per_couple else list(found.values())


# --------------------------------------------------------------------------- #
# The three distances
# --------------------------------------------------------------------------- #


def masked_field(mol: MoleculeFields, atoms: Sequence[int], name: str = "") -> MoleculeFields:
    """The fields of `mol` restricted to `atoms`, as a field object of its own.

    A kernel of this program is a sum over pairs of atoms, so restricting one side of it to a
    subset of the atoms is the same as building the fields of that subset and nothing else.  The
    atoms keep the coordinates they have in the molecule, so two groups compared this way are
    compared **in their poses** and not as detached fragments.
    """
    keep = np.asarray(sorted(set(int(a) for a in atoms)), int)
    if len(keep) == 0 or keep.min() < 0 or keep.max() >= mol.n_atoms:
        raise ValueError(f"{mol.name or 'a molecule'} has {mol.n_atoms} atoms; "
                         f"{list(atoms)} is not a subset of them")
    back = -np.ones(mol.n_atoms, int)
    back[keep] = np.arange(len(keep))
    bonds = (np.asarray(mol.bonds, int).reshape(-1, 2) if mol.bonds is not None
             else np.zeros((0, 2), int))
    if len(bonds):
        ok = (back[bonds[:, 0]] >= 0) & (back[bonds[:, 1]] >= 0)
        bonds = np.stack([back[bonds[ok, 0]], back[bonds[ok, 1]]], axis=1)
    charges = (np.asarray(mol.charges, float)[keep]
               if mol.charges is not None and len(mol.charges) == mol.n_atoms
               else np.zeros(len(keep)))
    return MoleculeFields(
        name=name or f"{mol.name or 'molecule'}:group",
        coords=np.asarray(mol.coords, float)[keep],
        weights=np.asarray(mol.weights, float)[keep],
        radii=np.asarray(mol.radii, float)[keep],
        symbols=[mol.symbols[k] for k in keep],
        atom_types=[mol.atom_types[k] for k in keep],
        bonds=bonds, charges=charges, member_atom_map=keep.astype(int),
        charge_source=mol.charge_source)


def group_distance(a: MoleculeFields, b: MoleculeFields, fieldset: FieldSet,
                   scale: Optional[np.ndarray] = None) -> float:
    """The distance between two posed groups in the field metric, with no model in it.

    ``d^2 = K(a,a) - 2 K(a,b) + K(b,b)`` over the enabled fields, each field scaled as the model
    scales it so that the fields are comparable with one another.  This is the reference the
    co-field weighting has to beat: it knows where the two groups put their field, but not
    whether the activity depends on it.
    """
    specs = [s for s in fieldset.specs if s.enabled]
    P = AtomTable.from_molecules([a, b])
    K = kernel_matrices(P, P, specs, symmetric=True)        # (F, 2, 2)
    h = fieldset.mixing_vector()
    s = np.ones(len(specs)) if scale is None else np.asarray(scale, float)
    d2 = 0.0
    for f in range(len(specs)):
        d2 += float(h[f] * s[f] * (K[f][0, 0] - 2.0 * K[f][0, 1] + K[f][1, 1]))
    return float(np.sqrt(max(d2, 0.0)))


def group_contribution(model, mol: MoleculeFields, atoms: Sequence[int],
                       coef: Optional[np.ndarray] = None) -> float:
    """How much of the prediction for `mol` belongs to the atoms of one group.

    The prediction of the overlap kernel is a sum over the atoms (decomposition rule 3), so this
    is simply that sum restricted to the group.  `coef` replaces the model's dual coefficients,
    which is how a value is obtained from a model that has not seen the molecules being compared.
    """
    keep = np.asarray(sorted(set(int(a) for a in atoms)), int)
    if coef is None:
        per = np.asarray(model.atom_contributions(mol), float)
    else:
        old = model.coef
        try:
            model.coef = np.asarray(coef, float)
            per = np.asarray(model.atom_contributions(mol), float)
        finally:
            model.coef = old
    return float(per[keep].sum())


def predicted_change(model, mol_a: MoleculeFields, atoms_a: Sequence[int],
                     mol_b: MoleculeFields, atoms_b: Sequence[int],
                     coef: Optional[np.ndarray] = None) -> float:
    """What the model says exchanging the first group for the second does to the activity."""
    return (group_contribution(model, mol_b, atoms_b, coef)
            - group_contribution(model, mol_a, atoms_a, coef))


def fingerprint_distance(group_a: str, group_b: str, radius: int = 2, bits: int = 2048) -> float:
    """One minus the Tanimoto similarity of two groups, written as SMILES with an attachment."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import TanimotoSimilarity
    fps = []
    for s in (group_a, group_b):
        m = Chem.MolFromSmiles(s, sanitize=False)
        if m is None:
            return float("nan")
        m.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(m)
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, int(radius), int(bits)))
    return float(1.0 - TanimotoSimilarity(fps[0], fps[1]))


# --------------------------------------------------------------------------- #
# Scoring a set of pairs
# --------------------------------------------------------------------------- #


@dataclass
class PairScore:
    """One matched pair, by the three measures and by what was measured."""
    pair: MatchedPair
    fingerprint: float
    field: float
    change: float

    @property
    def measured(self) -> float:
        return self.pair.delta

    def __str__(self) -> str:
        return (f"{self.pair.groups[0]} -> {self.pair.groups[1]}: fingerprint "
                f"{self.fingerprint:.2f}, field {self.field:.3f}, predicted "
                f"{self.change:+.2f}, measured {self.measured:+.2f}")


def score_pairs(model, fields: Sequence[MoleculeFields], pairs: Sequence[MatchedPair],
                coef_of=None, progress=None) -> List[PairScore]:
    """The three measures for every pair.

    `coef_of(pair)` returns the dual coefficients to use for that pair, which is how the model is
    kept away from the two activities it is being compared with; the default uses the model's own.
    """
    out: List[PairScore] = []
    for k, p in enumerate(pairs):
        a = masked_field(fields[p.a], p.atoms_a, name=p.groups[0])
        b = masked_field(fields[p.b], p.atoms_b, name=p.groups[1])
        coef = None if coef_of is None else coef_of(p)
        out.append(PairScore(
            pair=p,
            fingerprint=fingerprint_distance(p.groups[0], p.groups[1]),
            field=group_distance(a, b, model.fieldset, scale=getattr(model, "scale", None)),
            change=predicted_change(model, fields[p.a], p.atoms_a, fields[p.b], p.atoms_b,
                                    coef=coef)))
        if progress is not None:
            progress(f"Pairs {k + 1}/{len(pairs)}", (k + 1) / max(len(pairs), 1))
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation, which is what "ranks better" means here."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    if np.std(rx) <= 0 or np.std(ry) <= 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _ranks(v: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged."""
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v), float)
    r[order] = np.arange(len(v), dtype=float)
    # average the ranks of equal values
    for value in np.unique(v):
        same = v == value
        if same.sum() > 1:
            r[same] = r[same].mean()
    return r


def rank_gain(better: Sequence[float], worse: Sequence[float], measured: Sequence[float],
              runs: int = 2000, seed: int = 0) -> Tuple[float, float, float]:
    """How much better one measure ranks the measured difference than another, with an interval.

    Both measures are compared with the **size** of the measured activity difference, and the
    bootstrap resamples the pairs, so the interval is the one rule 4 asks for.
    """
    a, b = np.asarray(better, float), np.asarray(worse, float)
    m = np.abs(np.asarray(measured, float))
    keep = np.isfinite(a) & np.isfinite(b) & np.isfinite(m)
    a, b, m = a[keep], b[keep], m[keep]
    if len(a) < 5:
        return (float("nan"), float("nan"), float("nan"))
    point = spearman(a, m) - spearman(b, m)
    rng = np.random.RandomState(int(seed))
    got = []
    for _ in range(int(runs)):
        i = rng.randint(0, len(a), len(a))
        d = spearman(a[i], m[i]) - spearman(b[i], m[i])
        if np.isfinite(d):
            got.append(d)
    if not got:
        return (point, float("nan"), float("nan"))
    return (float(point), float(np.percentile(got, 2.5)), float(np.percentile(got, 97.5)))


# --------------------------------------------------------------------------- #
# The table itself
# --------------------------------------------------------------------------- #


@dataclass
class Table:
    """A square table of groups against groups at one position."""
    core: str
    groups: List[str]
    molecules: List[int]
    distance: np.ndarray                     # (G, G) the field distance between two groups
    change: np.ndarray                       # (G, G) what the model says exchanging them does
    half_width: float = float("nan")         # the model's own conformal half width
    names: List[str] = _field(default_factory=list)

    def interchangeable(self, group: str, within: Optional[float] = None) -> List[str]:
        """The groups whose predicted change against `group` is inside the model's own error.

        Without `within`, the bound is twice the conformal half width: the model cannot tell two
        predictions apart by less than that, so a smaller change is not a change it can see.
        """
        if group not in self.groups:
            raise ValueError(f"{group} is not in this table")
        i = self.groups.index(group)
        bound = (2.0 * self.half_width if within is None else float(within))
        if not np.isfinite(bound):
            raise ValueError("this table has no conformal half width, so a bound is needed")
        return [g for j, g in enumerate(self.groups)
                if j != i and abs(self.change[i, j]) <= bound]

    def rows(self):
        """(group a, group b, distance, change) for every ordered pair, a != b."""
        for i, a in enumerate(self.groups):
            for j, b in enumerate(self.groups):
                if i != j:
                    yield a, b, float(self.distance[i, j]), float(self.change[i, j])


def build_table(model, fields: Sequence[MoleculeFields], core: str,
                subs: Sequence[Substituent], half_width: float = float("nan"),
                names: Optional[Sequence[str]] = None) -> Table:
    """The square table of one position: every group of `subs` against every other.

    One group per molecule is used, so that the table is a table of groups and not of molecules;
    where two molecules carry the same group the first is kept.
    """
    seen: Dict[str, Substituent] = {}
    for s in subs:
        if s.core == core and s.group not in seen:
            seen[s.group] = s
    groups = list(seen)
    if len(groups) < 2:
        raise ValueError(f"only {len(groups)} group on this core: a table needs two")
    masked = [masked_field(fields[seen[g].molecule], seen[g].atoms, name=g) for g in groups]
    part = [group_contribution(model, fields[seen[g].molecule], seen[g].atoms) for g in groups]
    n = len(groups)
    dist = np.zeros((n, n))
    change = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = group_distance(masked[i], masked[j], model.fieldset,
                               scale=getattr(model, "scale", None))
            dist[i, j] = dist[j, i] = d
            change[i, j] = part[j] - part[i]
            change[j, i] = -change[i, j]
    return Table(core=core, groups=groups, molecules=[seen[g].molecule for g in groups],
                 distance=dist, change=change, half_width=float(half_width),
                 names=[(names[seen[g].molecule] if names else str(seen[g].molecule))
                        for g in groups])


def biggest_core(subs: Sequence[Substituent], least: int = 3) -> Optional[str]:
    """The core carrying the most different groups, which is where a table is worth building."""
    count: Dict[str, set] = defaultdict(set)
    for s in subs:
        count[s.core].add(s.group)
    best = [(len(v), k) for k, v in count.items() if len(v) >= int(least)]
    return max(best)[1] if best else None
