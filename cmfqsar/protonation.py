"""Protonation ensembles: one molecule as the weighted mean of its ionisation states (item D4).

A series measured in acid is usually modelled in one protonation state, chosen by hand or by a
rule.  That choice is a guess wherever a pKa falls near the pH of the experiment: an azole with a
pKa of 1 in 1 M HCl is about half protonated, and neither the neutral nor the cation is the
molecule that was measured.

The mean-map kernel removes the need to choose.  A CMF field is a sum of atom-centred Gaussians
and the kernel is bilinear in the fields, so the mean field of an ensemble is itself a field whose
atoms are the atoms of every member with their weights multiplied by that member's weight
(:func:`cmfqsar.fields.combine_members`).  The kernel of two ensembles is then the weighted mean
of the kernels of their member pairs, which is positive definite, and everything downstream --
co-fields, atom contributions, the decomposition rules -- keeps working because the model is still
linear in this mean field.  Nothing in that argument requires the members to be conformations of
one molecule, which is what this module uses: here they are different molecules.

What the weights are
--------------------
Each ionisable site is treated as independent and given the Henderson-Hasselbalch fraction at its
pKa::

    protonated fraction   = 1 / (1 + 10 ** (pH - pKa))
    deprotonated fraction = 1 / (1 + 10 ** (pKa - pH))

and a state's weight is the product over the sites.  Independent sites are an approximation: a
second deprotonation next to a first one is harder than its own pKa says, and this ignores that.
The pKa values and the patterns that find the sites are Dimorphite-DL's, which are regression
values with a published standard deviation, not measurements of these molecules.  Both limits are
real and are reported with any measurement made with this module.

Geometry
--------
The states are built on the geometry of the molecule as it is given, which matters because a CMF
series is aligned: the heavy atoms keep their coordinates, and only hydrogens are added or
removed, with their positions from RDKit.  So an aligned series stays aligned and the ensemble
costs no further superposition.

Usage
-----
    from cmfqsar.protonation import protonation_fields, protonation_states

    ens = protonation_states(mol, ph=0.0)          # states and weights
    mf = protonation_fields(mol, fieldset, ph=0.0)  # one field for the ensemble
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from .fields import FieldSet, MoleculeFields, build_fields, combine_members

#: the pH window used when the sites are enumerated.  It has to be wide enough that every site is
#: seen as titratable, so that both of its states are generated; the pH that decides the weights
#: is a separate argument.
WIDE_PH = (-10.0, 25.0)
#: how many ionisable sites are enumerated at most; the states double with every site
MAX_SITES = 8
#: states lighter than this share of the whole are dropped
MIN_POPULATION = 0.01
#: states kept at most, the likeliest ones
MAX_STATES = 8


class ProtonationUnavailable(RuntimeError):
    """Dimorphite-DL is not installed, so the states cannot be enumerated."""


@dataclass
class Site:
    """One ionisable site: where it is, what it is, and the pKa used for it."""
    name: str
    atom: int                 # heavy-atom index in the molecule as given, -1 if not placed
    pka: float
    stdev: float

    def fraction(self, ph: float) -> Tuple[float, float]:
        """(deprotonated, protonated) fraction at this pH, from the Henderson-Hasselbalch law."""
        return _fraction(self.pka, ph)


@dataclass
class State:
    """One protonation state of the molecule: a molecule of its own, with a weight."""
    mol: Chem.Mol
    weight: float
    label: str
    charge: int
    #: for every atom, the atom of the first state it corresponds to
    atom_map: np.ndarray = _field(default_factory=lambda: np.zeros(0, int))
    #: for every atom, the heavy atom of the parent molecule it came from, or -1 for a hydrogen
    #: that the state added
    heavy_trace: np.ndarray = _field(default_factory=lambda: np.zeros(0, int))

    @property
    def smiles(self) -> str:
        return Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(self.mol)))


@dataclass
class Ensemble:
    """The states of one molecule at one pH, heaviest first."""
    states: List[State]
    ph: float
    sites: List[Site]
    dropped: float = 0.0          # the weight of the states that were not kept
    note: str = ""

    def __len__(self) -> int:
        return len(self.states)

    @property
    def weights(self) -> np.ndarray:
        return np.array([s.weight for s in self.states], float)

    @property
    def is_mixture(self) -> bool:
        """More than one state carries real weight."""
        return len(self.states) > 1

    @property
    def main(self) -> State:
        return self.states[0]

    @property
    def mean_charge(self) -> float:
        return float(np.sum(self.weights * np.array([s.charge for s in self.states], float)))

    def describe(self) -> str:
        parts = [f"{s.label} {s.weight:.2f}" for s in self.states]
        return f"pH {self.ph:g}: " + ", ".join(parts)


def _fraction(pka: float, ph: float) -> Tuple[float, float]:
    """(deprotonated, protonated) fraction, computed in logs so extreme pKa values survive."""
    # the SMARTS table carries pKa values such as -1000 for a group that is always deprotonated
    dep = -_log10_one_plus_pow10(pka - ph)
    pro = -_log10_one_plus_pow10(ph - pka)
    return 10.0 ** dep, 10.0 ** pro


def _log10_one_plus_pow10(x: float) -> float:
    """log10(1 + 10**x) without evaluating 10**x, which overflows for a pKa of -1000."""
    return max(x, 0.0) + math.log10(1.0 + 10.0 ** -abs(x))


# ---------------------------------------------------------------------------- the enumeration


def _dimorphite():
    try:
        from dimorphite_dl.mol import MoleculeRecord
        from dimorphite_dl.protonate.change import protonate_site, site_log_probabilities
        from dimorphite_dl.protonate.detect import ProtonationSiteDetector
    except Exception as exc:                                     # pragma: no cover - optional
        raise ProtonationUnavailable(
            "protonation ensembles need Dimorphite-DL: pip install dimorphite-dl") from exc
    return MoleculeRecord, ProtonationSiteDetector, protonate_site, site_log_probabilities


def available() -> bool:
    """Whether the states can be enumerated at all."""
    try:
        _dimorphite()
    except ProtonationUnavailable:
        return False
    return True


def _mapped_smiles(heavy: Chem.Mol) -> str:
    """The heavy-atom SMILES with every atom numbered, so the states can be traced back to it."""
    q = Chem.Mol(heavy)
    for i, a in enumerate(q.GetAtoms()):
        a.SetAtomMapNum(i + 1)
    return Chem.MolToSmiles(q)


def _trace_atoms(state: Chem.Mol, heavy: Chem.Mol) -> Optional[List[int]]:
    """For every atom of a state, the heavy atom of the parent it came from.

    The atom numbers written into the SMILES survive the enumeration, except on the atoms that
    the neutralisation step rebuilds -- typically the one nitrogen whose charge it changed.  Those
    are recovered from their neighbours: an atom whose neighbours are known is the parent atom of
    the same element that is bonded to the same parent atoms, and that is unique in every case
    met here.  None if some atom cannot be placed, which makes the caller fall back to the single
    state rather than guess.
    """
    n = state.GetNumAtoms()
    out = [-1] * n
    for i, a in enumerate(state.GetAtoms()):
        num = a.GetAtomMapNum()
        if num > 0:
            out[i] = num - 1
    for _pass in range(3):
        for i, a in enumerate(state.GetAtoms()):
            if out[i] >= 0:
                continue
            anchors = [out[nb.GetIdx()] for nb in a.GetNeighbors() if out[nb.GetIdx()] >= 0]
            if not anchors:
                continue
            cands = set(range(heavy.GetNumAtoms()))
            for j in anchors:
                nbs = {nb.GetIdx() for nb in heavy.GetAtomWithIdx(j).GetNeighbors()}
                cands &= nbs
            cands = {c for c in cands if heavy.GetAtomWithIdx(c).GetAtomicNum()
                     == a.GetAtomicNum() and c not in out}
            if len(cands) == 1:
                out[i] = cands.pop()
        if all(j >= 0 for j in out):
            break
    if any(j < 0 for j in out) or len(set(out)) != n:
        return None
    return out


def _state_in_place(state: Chem.Mol, heavy: Chem.Mol, trace: Sequence[int]) -> Optional[Chem.Mol]:
    """One state on the parent's geometry: the heavy atoms where they were, hydrogens added."""
    m = Chem.Mol(state)
    for a in m.GetAtoms():
        a.SetAtomMapNum(0)
    try:
        Chem.SanitizeMol(m)
    except Exception:
        return None
    conf = Chem.Conformer(m.GetNumAtoms())
    src = heavy.GetConformer()
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, src.GetAtomPosition(int(trace[i])))
    m.RemoveAllConformers()
    m.AddConformer(conf, assignId=True)
    try:
        out = Chem.AddHs(m, addCoords=True)
    except Exception:
        return None
    return out


def _label(mol: Chem.Mol) -> str:
    """A short name for a state: its net charge, which is what distinguishes the states here."""
    q = Chem.GetFormalCharge(mol)
    return "neutral" if q == 0 else (f"{q:+d}" if abs(q) > 1 else ("cation" if q > 0 else "anion"))


def protonation_states(mol: Chem.Mol, ph: float = 0.0, min_population: float = MIN_POPULATION,
                       max_states: int = MAX_STATES, max_sites: int = MAX_SITES) -> Ensemble:
    """Enumerate the protonation states of one molecule and weigh them by their pKa values.

    The molecule is used as it is given: its heavy-atom geometry is kept, so an aligned series
    stays aligned.  Returns an :class:`Ensemble` with the states heaviest first; a molecule with
    no ionisable site, or one the enumeration cannot trace back to its own atoms, comes back as a
    single state that is the molecule itself, so a caller never has to special-case it.
    """
    MoleculeRecord, Detector, protonate_site, site_scores = _dimorphite()
    single = Ensemble(states=[State(mol=mol, weight=1.0, label=_label(mol),
                                    charge=Chem.GetFormalCharge(mol),
                                    atom_map=np.arange(mol.GetNumAtoms(), dtype=int),
                                    heavy_trace=np.arange(mol.GetNumAtoms(), dtype=int))],
                      ph=ph, sites=[])
    if mol.GetNumConformers() == 0:
        single.note = "no geometry, so the molecule is used as it is"
        return single
    heavy = Chem.RemoveHs(Chem.Mol(mol))
    try:
        rec = MoleculeRecord(_mapped_smiles(heavy))
        rec.prepare_for_protonation()
        rec, found = Detector().find_sites(rec)
    except Exception as exc:
        single.note = f"the sites could not be found ({exc})"
        return single
    if not found:
        single.note = "no ionisable site"
        return single
    found = list(found)[:max_sites]
    sites: List[Site] = []
    mols = [rec.mol]
    logw = [0.0]
    for site in found:
        for pka in site.pkas:
            atom = site.idxs_match[pka.idx_site] if pka.idx_site < len(site.idxs_match) else -1
            sites.append(Site(name=site.name, atom=int(atom), pka=float(pka.mean),
                              stdev=float(pka.stdev)))
        try:
            grown = protonate_site(list(mols), site, WIDE_PH[0], WIDE_PH[1], 0.0)
            scores = site_scores(site, WIDE_PH[0], WIDE_PH[1], 0.0, ph)
        except Exception:
            continue
        if not grown or len(grown) != len(mols) * len(scores):
            continue                       # the orders do not line up, so this site is left alone
        mols = grown
        logw = [a + b for a in logw for b in scores]
        if len(mols) > 4 * max_states:     # prune as we go, or the states double with every site
            order = np.argsort(logw)[::-1][:max_states]
            mols = [mols[i] for i in order]
            logw = [logw[i] for i in order]
    w = np.array(logw, float)
    w = np.exp((w - w.max()) * math.log(10.0))
    w /= w.sum()
    order = np.argsort(w)[::-1]
    states: List[State] = []
    kept = 0.0
    for i in order:
        if len(states) >= max_states or w[i] < min_population:
            break
        trace = _trace_atoms(mols[i], heavy)
        built = _state_in_place(mols[i], heavy, trace) if trace is not None else None
        if built is None:
            continue
        full = np.full(built.GetNumAtoms(), -1, int)
        full[:len(trace)] = np.asarray(trace, int)
        states.append(State(mol=built, weight=float(w[i]), label=_label(built),
                            charge=Chem.GetFormalCharge(built), heavy_trace=full))
        kept += float(w[i])
    if not states:
        single.sites = sites
        single.note = "the states could not be built on this geometry"
        return single
    for s in states:                       # the weights are renormalised over what is kept
        s.weight /= kept
    out = Ensemble(states=states, ph=ph, sites=sites, dropped=1.0 - kept)
    _map_states(out, heavy)
    return out


def _map_states(ens: Ensemble, heavy: Chem.Mol) -> None:
    """Say which atom of the first state each atom of every state corresponds to.

    The heavy atoms are the same molecule in every state, so they map one to one through the
    parent.  A hydrogen has no counterpart wherever the states differ, which is the whole point of
    the ensemble, so it is reported on the heavy atom it is bonded to.  That is the atom a chemist
    would read it on: "this nitrogen gains" rather than "this proton gains".
    """
    n_heavy = heavy.GetNumAtoms()
    # where each parent heavy atom sits in the first state's atom list
    place: Dict[int, int] = {}
    for i, j in enumerate(ens.states[0].heavy_trace):
        if j >= 0 and int(j) not in place:
            place[int(j)] = i
    for st in ens.states:
        amap = np.full(st.mol.GetNumAtoms(), -1, int)
        for i, j in enumerate(st.heavy_trace):
            if j >= 0:
                amap[i] = place.get(int(j), -1)
        for i, a in enumerate(st.mol.GetAtoms()):
            if amap[i] >= 0 or a.GetAtomicNum() != 1:
                continue
            nbs = [nb.GetIdx() for nb in a.GetNeighbors()]
            if nbs and amap[nbs[0]] >= 0:
                amap[i] = amap[nbs[0]]
        st.atom_map = amap
    assert n_heavy == len(place), "every heavy atom of the parent has to be placed"


# ------------------------------------------------------------------------------ the field


def protonation_fields(mol: Chem.Mol, fieldset: FieldSet, ph: float = 0.0, name: str = "",
                       min_population: float = MIN_POPULATION, max_states: int = MAX_STATES,
                       report: Optional[Dict] = None) -> MoleculeFields:
    """The field of one molecule as the pKa-weighted mean over its protonation states.

    With one state above the threshold this is exactly :func:`build_fields` on that state, so a
    series where every molecule is unambiguous gives the same numbers as the single-state model
    with that state chosen.  `report`, if given, is filled with what the ensemble was.
    """
    ens = protonation_states(mol, ph=ph, min_population=min_population, max_states=max_states)
    members = [build_fields(s.mol, fieldset, name=name or "", _single=True) for s in ens.states]
    if report is not None:
        report.update({"n_states": len(ens.states), "weights": ens.weights.tolist(),
                       "labels": [s.label for s in ens.states],
                       "charges": [s.charge for s in ens.states],
                       "mean_charge": ens.mean_charge, "dropped": ens.dropped,
                       "sites": [{"name": s.name, "pka": s.pka} for s in ens.sites],
                       "note": ens.note})
    if len(members) == 1:
        return members[0]
    return combine_members(members, ens.weights, kind="protonation",
                           atom_maps=[s.atom_map for s in ens.states], name=name)


def ensemble_table(mols: Sequence[Chem.Mol], names: Sequence[str], ph: float = 0.0,
                   min_population: float = MIN_POPULATION,
                   max_states: int = MAX_STATES) -> List[Ensemble]:
    """The ensembles of a whole series, in order."""
    out = []
    for m, nm in zip(mols, names):
        try:
            out.append(protonation_states(m, ph=ph, min_population=min_population,
                                          max_states=max_states))
        except ProtonationUnavailable:
            raise
        except Exception as exc:                                 # pragma: no cover - defensive
            e = Ensemble(states=[State(mol=m, weight=1.0, label=_label(m),
                                       charge=Chem.GetFormalCharge(m),
                                       atom_map=np.arange(m.GetNumAtoms(), dtype=int))],
                         ph=ph, sites=[], note=f"failed: {exc}")
            out.append(e)
    return out


def mixture_share(ensembles: Sequence[Ensemble]) -> float:
    """What share of a series is a real mixture at this pH, which is what the ensemble can change."""
    if not ensembles:
        return 0.0
    return float(np.mean([1.0 if e.is_mixture else 0.0 for e in ensembles]))
