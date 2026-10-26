"""Tautomer ensembles: the other half of plan item D4.

An azole can be written with its proton on either nitrogen, and the two forms have different
fields.  The choice is usually made by whatever wrote the file.  The ensemble built here averages
over the forms instead, with the same mathematics as the protonation ensemble in
:mod:`cmfqsar.protonation`: the members are different molecules, and the mean field of the
ensemble is itself a field (:func:`cmfqsar.fields.combine_members`).

**The weights are the weak part, and they are weaker than the protonation ones.** A protonation
state has a population the pKa and the pH give it.  A tautomer has one only if the free energies
are known, and nothing in this program computes them: RDKit enumerates the forms and scores them
with a heuristic that prefers, for instance, an aromatic ring and a carbonyl over their
alternatives, but that score is not an energy and cannot be turned into a population.  So the
weights offered here are

  ``uniform``   every enumerated form counts the same
  ``score``     only the forms RDKit scores highest are kept, counted the same
  ``given``     weights from a molecule property, for energies computed elsewhere

and none of them is a measurement.  ``score`` is the mildest assumption and the default: it keeps
the forms a chemist would draw and drops the ones the enumerator reaches only formally.  Energies
from a semiempirical calculation (plan item D3) would replace the guess with a Boltzmann weight.

The geometry is the parent's: a tautomer has the same heavy atoms in the same order, so the heavy
atoms keep their coordinates and only the hydrogens move.  An aligned series stays aligned.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from rdkit import Chem

from .fields import FieldSet, MoleculeFields, build_fields, combine_members
from .protonation import Ensemble, State

#: how the forms are weighed, in the order they are offered on the command line
WEIGHTINGS = ("score", "uniform", "given")
#: forms kept at most
MAX_FORMS = 8
#: the property a molecule may carry with one relative energy, in kcal/mol, per form
ENERGY_PROP = "tautomer_energies"
#: kT in kcal/mol, about RT at 300 K, for "given" energies
KT = 0.6


class TautomersUnavailable(RuntimeError):
    """This RDKit build has no tautomer enumerator."""


def _enumerator(max_forms: int = MAX_FORMS):
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except Exception as exc:                                    # pragma: no cover - old RDKit
        raise TautomersUnavailable("this RDKit has no tautomer enumerator") from exc
    te = rdMolStandardize.TautomerEnumerator()
    te.SetMaxTautomers(max(1, int(max_forms) * 4))               # the scoring prunes afterwards
    return rdMolStandardize, te


def available() -> bool:
    """Whether the forms can be enumerated at all."""
    try:
        _enumerator()
    except TautomersUnavailable:
        return False
    return True


def tautomer_states(mol: Chem.Mol, weighting: str = "score", max_forms: int = MAX_FORMS,
                    energy_prop: str = ENERGY_PROP, kt: float = KT) -> Ensemble:
    """Enumerate the tautomers of one molecule, on its own geometry, and weigh them.

    Returns an :class:`cmfqsar.protonation.Ensemble` so that the two kinds of ensemble are read
    the same way.  A molecule with one tautomer, or one whose forms cannot be built on this
    geometry, comes back as a single state that is the molecule itself.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"unknown tautomer weighting '{weighting}'; "
                         f"the ones defined are {', '.join(WEIGHTINGS)}")
    single = Ensemble(states=[State(mol=mol, weight=1.0, label="as given",
                                    charge=Chem.GetFormalCharge(mol),
                                    atom_map=np.arange(mol.GetNumAtoms(), dtype=int),
                                    heavy_trace=np.arange(mol.GetNumAtoms(), dtype=int))],
                      ph=float("nan"), sites=[])
    if mol.GetNumConformers() == 0:
        single.note = "no geometry, so the molecule is used as it is"
        return single
    std, te = _enumerator(max_forms)
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    try:
        forms = list(te.Enumerate(heavy))
    except Exception as exc:
        single.note = f"the forms could not be enumerated ({exc})"
        return single
    forms = [f for f in forms if f is not None
             and f.GetNumAtoms() == heavy.GetNumAtoms()
             and [a.GetAtomicNum() for a in f.GetAtoms()]
             == [a.GetAtomicNum() for a in heavy.GetAtoms()]]
    if len(forms) <= 1:
        single.note = "one tautomer"
        return single
    scores = [float(te.ScoreTautomer(f)) for f in forms]
    if weighting == "score":
        best = max(scores)
        keep = [i for i, s in enumerate(scores) if s >= best]
        w = np.ones(len(keep))
    elif weighting == "uniform":
        keep = list(range(len(forms)))
        w = np.ones(len(keep))
    else:
        energies = _given_energies(mol, energy_prop, len(forms))
        if energies is None:
            single.note = f"no per-form energies in '{energy_prop}'"
            return single
        keep = list(range(len(forms)))
        e = np.asarray(energies, float)
        w = np.exp(-(e - e.min()) / max(kt, 1e-9))
    order = np.argsort(w)[::-1][:max_forms]
    keep = [keep[i] for i in order]
    w = np.asarray([w[i] for i in order], float)
    w = w / w.sum()
    states: List[State] = []
    for weight, i in zip(w, keep):
        built = _form_in_place(forms[i], heavy)
        if built is None:
            continue
        n = built.GetNumAtoms()
        trace = np.full(n, -1, int)
        trace[:heavy.GetNumAtoms()] = np.arange(heavy.GetNumAtoms(), dtype=int)
        states.append(State(mol=built, weight=float(weight),
                            label=Chem.MolToSmiles(forms[i]),
                            charge=Chem.GetFormalCharge(built), heavy_trace=trace))
    if not states:
        single.note = "the forms could not be built on this geometry"
        return single
    total = sum(s.weight for s in states)
    for s in states:
        s.weight /= total
    out = Ensemble(states=states, ph=float("nan"), sites=[], dropped=1.0 - total,
                   note=f"{len(states)} tautomers, weighted '{weighting}'")
    _map_forms(out, heavy)
    return out


def _given_energies(mol: Chem.Mol, prop: str, n: int) -> Optional[List[float]]:
    if not mol.HasProp(prop):
        return None
    try:
        vals = [float(x) for x in mol.GetProp(prop).replace(",", " ").split()]
    except ValueError:
        return None
    return vals if len(vals) == n else None


def _form_in_place(form: Chem.Mol, heavy: Chem.Mol) -> Optional[Chem.Mol]:
    """One tautomer on the parent's heavy-atom geometry, with hydrogens added by RDKit."""
    m = Chem.Mol(form)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    conf = Chem.Conformer(m.GetNumAtoms())
    src = heavy.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, src.GetAtomPosition(i))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    try:
        return Chem.AddHs(m, addCoords=True)
    except Exception:
        return None


def _map_forms(ens: Ensemble, heavy: Chem.Mol) -> None:
    """Which atom of the first form every atom corresponds to; a hydrogen goes on its heavy atom."""
    n_heavy = heavy.GetNumAtoms()
    for st in ens.states:
        amap = np.full(st.mol.GetNumAtoms(), -1, int)
        amap[:n_heavy] = np.arange(n_heavy, dtype=int)
        for i, a in enumerate(st.mol.GetAtoms()):
            if amap[i] >= 0 or a.GetAtomicNum() != 1:
                continue
            nbs = [nb.GetIdx() for nb in a.GetNeighbors()]
            if nbs and amap[nbs[0]] >= 0:
                amap[i] = amap[nbs[0]]
        st.atom_map = amap


def tautomer_fields(mol: Chem.Mol, fieldset: FieldSet, weighting: str = "score",
                    max_forms: int = MAX_FORMS, name: str = "",
                    report: Optional[Dict] = None) -> MoleculeFields:
    """The field of one molecule as the mean over its tautomers."""
    ens = tautomer_states(mol, weighting=weighting, max_forms=max_forms)
    members = [build_fields(s.mol, fieldset, name=name or "", _single=True) for s in ens.states]
    if report is not None:
        report.update({"n_forms": len(ens.states), "weights": ens.weights.tolist(),
                       "forms": [s.label for s in ens.states], "note": ens.note})
    if len(members) == 1:
        return members[0]
    return combine_members(members, ens.weights, kind="tautomers",
                           atom_maps=[s.atom_map for s in ens.states], name=name)


def forms_table(mols: Sequence[Chem.Mol], weighting: str = "score",
                max_forms: int = MAX_FORMS) -> List[Ensemble]:
    """The tautomer ensembles of a whole series, in order."""
    return [tautomer_states(m, weighting=weighting, max_forms=max_forms) for m in mols]


def ambiguous_share(ensembles: Sequence[Ensemble]) -> float:
    """What share of a series has more than one form, which is what the ensemble can change."""
    if not ensembles:
        return 0.0
    return float(np.mean([1.0 if len(e) > 1 else 0.0 for e in ensembles]))
