"""The last step of the design cycle: what to put at a position, ranked by the model (E3).

Everything else in this program scores molecules somebody has already drawn.  This module makes
the suggestion itself: take a parent compound of a series, take one of its attachment points,
replace what is there with every fragment of a library, and rank the results by what the model
expects of them.

The pose problem that stops this being easy is solved by not having one.  A new substituent has
no crystal structure and no alignment, and superimposing it on anything would be a guess; the
topomer rule (:mod:`cmfqsar.topomer`) builds it into a pose that is a deterministic function of
its own topology, so the candidate is posed by exactly the rule that posed the molecules the
model was fitted on.  The comparison is then between comparable things at one attachment point,
which is all a substituent ranking needs.  It is also the reason this works on a topomer model and
not on an aligned one: a model fitted on superimposed whole molecules has no way to place a group
that was never synthesised.

**On a rotation-invariant model there is no pose problem at all** (plan item SC7).  The field
autocorrelation is a function of the molecule's own conformation and of nothing else, so a
candidate needs no frame and no topomer rule: it is built as a whole molecule, embedded once, and
scored.  That is what makes a design loop possible on a guest table, where the molecules share no
scaffold and the superposition the aligned route would need does not exist.  `pose="molecule"`
selects it and is the default for such a model; the annotation that compares the candidate's
fields with the parent's needs a frame and is replaced there by the atom contributions of the
invariant model (:func:`cmfqsar.invariant.atom_contributions`), which say how much of the
prediction comes from the group that was put in.

Three numbers come back for every candidate and they answer different questions:

* the **prediction**, which is what the model expects;
* the **posterior standard deviation**, which says whether the model has any basis for an
  expectation -- a fragment unlike anything in the series gets the training mean and a large
  spread, and the two together are what stop a design loop proposing its own ignorance;
* an **acquisition score** (:mod:`cmfqsar.active`) that combines them according to what the
  chemist is doing, expected improvement by default.

The ranked list is then annotated: whether the fragment is one the series already uses, whether
the whole molecule has already been made (and if so what it measured, which is the only free
check on the suggestion), and what the candidate's fields have that the parent's do not
(:mod:`cmfqsar.fielddiff`), so the answer is *put a wider acceptor here* and not only a number.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, rdBase

from .fields import AtomTable, FieldSet, MoleculeFields, build_fields
from .topomer import (TopomerSpec, attachment_label, series_labels, substituents,
                      topomer_assembly)

#: Common monovalent substituents of medicinal chemistry, as SMILES with one attachment point.
#: A working default rather than a claim: it is a hand-made list of the groups that appear in
#: most series, ordered roughly from small to large, and a real campaign should pass its own
#: library with ``--fragments``.
COMMON_FRAGMENTS: Tuple[str, ...] = (
    "[*][H]", "[*]F", "[*]Cl", "[*]Br", "[*]I", "[*]C", "[*]CC", "[*]C(C)C", "[*]C(C)(C)C",
    "[*]C1CC1", "[*]O", "[*]OC", "[*]OCC", "[*]OC(F)(F)F", "[*]C(F)(F)F", "[*]C#N", "[*][N+](=O)[O-]",
    "[*]N", "[*]NC", "[*]N(C)C", "[*]NC(C)=O", "[*]C(=O)O", "[*]C(=O)OC", "[*]C(=O)N",
    "[*]C(=O)NC", "[*]S(N)(=O)=O", "[*]SC", "[*]S(C)(=O)=O", "[*]c1ccccc1", "[*]c1ccc(F)cc1",
    "[*]c1ccccn1", "[*]c1cccnc1", "[*]c1ccncc1", "[*]N1CCOCC1", "[*]N1CCCCC1", "[*]N1CCN(C)CC1",
    "[*]n1ccnc1", "[*]c1nnn[nH]1", "[*]CO", "[*]CN", "[*]C=C", "[*]C#C", "[*]OC(=O)C",
    "[*]CC(=O)O",
)


@dataclass
class Candidate:
    """One replacement at one position, with everything that was worked out about it."""

    smiles: str                   # the fragment, as SMILES with its attachment point
    label: int                    # the core atom it is attached to
    mean: float = float("nan")
    sd: float = float("nan")
    score: float = float("nan")   # the acquisition score it was ranked by
    delta: float = float("nan")   # prediction minus the parent's prediction
    known: bool = False           # the series already uses this fragment here
    made: Optional[str] = None    # the name of the molecule that is this candidate, if any
    measured: float = float("nan")   # and what it measured
    fields: List = _field(default_factory=list)   # FieldMismatch against the parent
    #: for a candidate built as a whole molecule: how much of the prediction comes from the atoms
    #: of the group that was put in, in the units of the activity, and how many heavy atoms that
    #: group holds out of the molecule's (plan item SC7).  A share would not do: the parts of a
    #: sum that is not sign-definite can each exceed the whole.
    group_value: float = float("nan")
    group_total: float = float("nan")
    group_atoms: float = float("nan")
    #: the candidate itself, as SMILES
    molecule: str = ""

    def describe(self) -> str:
        mark = "known" if self.known else "new  "
        got = "" if self.made is None else f"   made: {self.made} ({self.measured:.2f})"
        share = ("" if self.group_value != self.group_value
                 else f"   the group carries {self.group_value:+.2f} of {self.group_total:+.2f}")
        return (f"{self.smiles[:28]:28s} {mark}  predicted {self.mean:7.3f} +- {self.sd:5.3f}"
                f"   delta {self.delta:+6.3f}   score {self.score:+8.4f}{got}{share}")


@dataclass
class DesignResult:
    """A ranked set of replacements at one position of one parent."""

    parent: str
    label: int
    acquire: str
    candidates: List[Candidate]
    parent_mean: float = float("nan")
    parent_sd: float = float("nan")
    parent_measured: float = float("nan")
    posed: int = 0
    failed: List[str] = _field(default_factory=list)
    #: true when the acquisition score is the same for every candidate, so it ranks nothing.
    #: Expected improvement does this when the best molecule of the series is far above anything
    #: the design can reach, which is the usual case on a table that is not a congeneric series.
    flat_score: bool = False

    def top(self, n: int = 10) -> List[Candidate]:
        return self.candidates[:int(n)]

    def describe(self) -> str:
        new = sum(1 for c in self.candidates if not c.known)
        flat = ("" if not self.flat_score else
                f". WARNING: every candidate scores the same under '{self.acquire}', so the "
                f"order below is not a ranking; with the best molecule of the series far above "
                f"anything on offer, expected improvement is flat. Rank by the prediction "
                f"itself ('greedy') or by the spread ('variance')")
        return (f"{len(self.candidates)} candidates at position {self.label} of {self.parent} "
                f"({new} not used anywhere in the series), ranked by {self.acquire}; "
                f"{len(self.failed)} fragments could not be posed{flat}")


def group_contribution(model, fields: MoleculeFields, mol: Chem.Mol,
                       core: Chem.Mol) -> Tuple[float, float, float]:
    """How much of a prediction comes from the atoms that are not the core (plan item SC7).

    Returns (the group's share of the prediction in the units of the activity, the whole molecule's,
    the share of the heavy atoms the group holds).  The prediction splits over the atoms exactly
    for the invariant kernel, so these are arithmetic rather than an attribution, and the three
    numbers together say whether the group earns its size.  The total is the prediction without
    the bias, so a candidate predicted below the training mean has a negative one.
    """
    nan = float("nan")
    try:
        c = np.asarray(model.atom_contributions(fields), float)
    except (ValueError, RuntimeError):
        return nan, nan, nan
    match = set(mol.GetSubstructMatch(core))
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    if not match or not heavy:
        return nan, nan, nan
    # a hydrogen belongs to the atom it hangs on, as everywhere else
    per = np.array(c, float)
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == 1 and a.GetNeighbors():
            per[a.GetNeighbors()[0].GetIdx()] += per[a.GetIdx()]
            per[a.GetIdx()] = 0.0
    outside = [i for i in heavy if i not in match]
    return (float(per[outside].sum()) if outside else 0.0, float(per[heavy].sum()),
            float(len(outside)) / len(heavy))


def read_fragments(source: str = "common") -> List[str]:
    """A fragment library: the built-in list, or one SMILES per line from a file.

    A line may carry a name after the SMILES, which is ignored: the fragment is identified by
    its own canonical form throughout, so that the same group written two ways is one candidate.
    """
    if source in ("common", "", None):
        return list(COMMON_FRAGMENTS)
    out = []
    with open(source, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line.split()[0])
    return out


def as_fragment(smiles: str, label: int) -> Optional[Chem.Mol]:
    """A library SMILES turned into a fragment this attachment point can accept.

    The attachment is written ``[*]`` or ``*``; it becomes the dummy atom with the isotope that
    :func:`cmfqsar.topomer.substituents` uses to name the core atom a group is attached to, so that
    a fragment from a library and a fragment cut off a molecule are the same kind of object.
    """
    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    dummies = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) != 1:
        return None
    dummies[0].SetIsotope(int(label))
    try:
        Chem.SanitizeMol(mol)
    except Exception:                            # noqa: BLE001 - a fragment that cannot be built
        return None
    return mol


def fragment_smiles(frag) -> str:
    """The canonical name of a fragment, with hydrogens implicit.

    A group cut out of an SDF carries its hydrogens as atoms; the same group written into a
    library by hand does not, and the two are written differently. Everything here -- which
    candidate is which, which are the groups the series already uses, which candidate rebuilds a
    molecule somebody has already made -- is decided by this name, so it has to be one name.
    """
    smi = frag if isinstance(frag, str) else Chem.MolToSmiles(frag)
    with rdBase.BlockLogs():                     # a hydrogen on a dummy atom warns, every time
        mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else smi


def _key(frags: Sequence[Chem.Mol]) -> str:
    """What makes two assemblies the same molecule: the set of (position, group)."""
    return "|".join(sorted(f"{attachment_label(f)}:{fragment_smiles(f)}" for f in frags))


def enumerate_replacements(parent: Chem.Mol, core: Chem.Mol, label: int,
                           fragments: Sequence[str], labels: Optional[Sequence[int]] = None,
                           spec: TopomerSpec = TopomerSpec(),
                           cache: Optional[Dict[str, Chem.Mol]] = None
                           ) -> Tuple[List[Tuple[str, Chem.Mol, str]], List[str]]:
    """Build the parent with each library fragment in place of what sits at `label`.

    Returns (candidates, failures), where a candidate is (fragment SMILES, posed assembly, the
    key naming the whole substitution pattern).  Everything the parent carries at its other
    positions is kept exactly as it is, so the only difference between two candidates is the one
    group being varied -- which is what makes the ranking a ranking of substituents.

    A fragment is one candidate however often the library holds it.  A group cut out of the
    series carries its hydrogens as atoms and the same group in the built-in list does not, so
    a library made of both lists holds such a group under two spellings; it is built once, under
    its canonical name (:func:`fragment_smiles`), and the first spelling met is the one used.
    """
    labels = list(labels) if labels is not None else series_labels([parent], core)
    others = [f for f in substituents(parent, core) if attachment_label(f) != label]
    cache = {} if cache is None else cache
    out, failed, taken = [], [], set()
    for smi in fragments:
        frag = as_fragment(smi, label)
        if frag is None:
            failed.append(smi)
            continue
        name = fragment_smiles(frag)
        if name in taken:                        # the same group, written a second way
            continue
        taken.add(name)
        try:
            built = topomer_assembly(others + [frag], labels, spec, cache)
        except Exception:                        # noqa: BLE001 - one fragment, not the library
            built = None
        if built is None:
            failed.append(smi)
            continue
        out.append((name, built, _key(others + [frag])))
    return out, failed


def whole_molecule(core: Chem.Mol, frags: Sequence[Chem.Mol]) -> Optional[Chem.Mol]:
    """Put the core and its substituents back together into one molecule (plan item SC7).

    The fragments are the ones :func:`cmfqsar.topomer.substituents` produces, whose dummy atom
    carries the index of the core atom it was attached to, so the bond to remake is named by the
    fragment itself.  This is the inverse of the cut, and it is what a design loop on a model
    that needs no frame builds its candidates with.
    """
    rw = Chem.RWMol(core)
    n_core = rw.GetNumAtoms()
    drop = []
    try:
        for frag in frags:
            off = rw.GetNumAtoms()
            rw.InsertMol(frag)
            dummy = next((a.GetIdx() for a in frag.GetAtoms() if a.GetAtomicNum() == 0), None)
            if dummy is None:
                return None
            at = int(frag.GetAtomWithIdx(dummy).GetIsotope())
            nbrs = frag.GetAtomWithIdx(dummy).GetNeighbors()
            if at >= n_core or not nbrs:
                return None
            rw.AddBond(at, off + nbrs[0].GetIdx(), Chem.BondType.SINGLE)
            drop.append(off + dummy)
        for i in sorted(drop, reverse=True):
            rw.RemoveAtom(i)
        out = rw.GetMol()
        Chem.SanitizeMol(out)
    except Exception:                            # noqa: BLE001 - one candidate, not the library
        return None
    return out


def embed(mol: Chem.Mol, seed: int = 1) -> Optional[Chem.Mol]:
    """One conformation of a molecule, the way the guest tables were embedded."""
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.Mol(mol))
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    if AllChem.EmbedMolecule(m, p) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
    except Exception:                            # noqa: BLE001 - keep the embedded geometry
        pass
    return m


def enumerate_molecules(parent: Chem.Mol, core: Chem.Mol, label: int,
                        fragments: Sequence[str], seed: int = 1
                        ) -> Tuple[List[Tuple[str, Chem.Mol, str]], List[str]]:
    """Build the parent with each library fragment in place of what sits at `label`.

    The candidates are whole molecules, each embedded once; nothing is superimposed on anything,
    which is what a model with no frame allows and a guest table needs.  As in
    :func:`enumerate_replacements`, a fragment the library holds twice is built once.  The seed
    of an embedding stays the place of the fragment in the library, so dropping a repeat does
    not change the geometry of any other candidate.
    """
    others = [f for f in substituents(parent, core) if attachment_label(f) != label]
    out, failed, taken = [], [], set()
    for k, smi in enumerate(fragments):
        frag = as_fragment(smi, label)
        if frag is None:
            failed.append(smi)
            continue
        name = fragment_smiles(frag)
        if name in taken:                        # the same group, written a second way
            continue
        taken.add(name)
        built = whole_molecule(core, others + [frag])
        posed = embed(built, seed=seed + k) if built is not None else None
        if posed is None:
            failed.append(smi)
            continue
        out.append((name, posed, _key(others + [frag])))
    return out, failed


def design(model, parent: Chem.Mol, core: Chem.Mol, label: int,
           fragments: Optional[Sequence[str]] = None, labels: Optional[Sequence[int]] = None,
           acquire: str = "ei", beta: float = 1.0, xi: float = 0.0,
           spec: TopomerSpec = TopomerSpec(), parent_name: str = "the parent",
           series: Optional[Sequence[Chem.Mol]] = None,
           series_names: Optional[Sequence[str]] = None,
           series_y: Optional[Sequence[float]] = None,
           known: Optional[Sequence[str]] = None, with_fields: int = 0,
           pose: str = "auto", seed: int = 1, progress=None) -> DesignResult:
    """Rank replacements at one position of `parent` under a fitted model.

    `series`, `series_names` and `series_y` are the molecules the model was fitted on; they are
    used only to say which suggestions have already been made and what those measured, which is
    the one free check there is on a design loop.

    `pose` is how a candidate is built: ``"topomer"`` poses the substituent by the topomer rule,
    which is what a topomer model is fitted on, and ``"molecule"`` builds the whole molecule and
    embeds it once, which a rotation-invariant model can score because it needs no frame (plan
    item SC7).  ``"auto"``, the default, picks the second for an invariant model and the first
    otherwise.
    """
    from .active import acquisition_scores
    from .fielddiff import mismatch

    fs: FieldSet = model.fieldset
    frags = list(fragments) if fragments is not None else list(COMMON_FRAGMENTS)
    labels = list(labels) if labels is not None else series_labels([parent], core)
    cache: Dict[str, Chem.Mol] = {}
    if pose == "auto":
        pose = "molecule" if getattr(model, "kernel_source", "") == "autocorrelation" else "topomer"
    if pose not in ("topomer", "molecule"):
        raise ValueError(f"unknown pose '{pose}'; it is 'topomer', 'molecule' or 'auto'")
    if pose == "molecule" and getattr(model, "uses_fields", True):
        raise ValueError(
            "a candidate built as a whole molecule has no frame in common with the training "
            "molecules, so a kernel that is an overlap integral cannot score it; this route is "
            "for the rotation-invariant kernel. Use pose='topomer' with an aligned or topomer "
            "model.")

    parent_frags = substituents(parent, core)
    if pose == "molecule":
        built, failed = enumerate_molecules(parent, core, label, frags, seed=seed)
        parent_built = embed(whole_molecule(core, parent_frags), seed=seed)
    else:
        built, failed = enumerate_replacements(parent, core, label, frags, labels, spec, cache)
        # the parent itself, posed by the same rule, is what every candidate is compared with
        parent_built = topomer_assembly(parent_frags, labels, spec, cache)
    if progress:
        progress(f"{len(built)} candidates posed", 0.3)
    parent_fields = build_fields(parent_built, fs, parent_name) if parent_built else None
    parent_mean = parent_sd = float("nan")
    if parent_fields is not None:
        pm, pv = model.predict_with_variance(AtomTable.from_molecules([parent_fields]))
        parent_mean = float(pm[0])
        parent_sd = float(np.sqrt(max(float(pv[0]), 0.0)))

    # which substitution patterns exist already, and what they measured
    made: Dict[str, Tuple[str, float]] = {}
    if series is not None:
        names = list(series_names) if series_names is not None else [f"m{i}" for i in
                                                                     range(len(series))]
        ys = (np.asarray(series_y, float) if series_y is not None
              else np.full(len(series), np.nan))
        for mol, nm, yv in zip(series, names, ys):
            if mol is None:
                continue
            try:
                made.setdefault(_key(substituents(mol, core)), (nm, float(yv)))
            except Exception:                    # noqa: BLE001 - a molecule without the core
                continue
    seen = set(known or [])

    fields_mf = [build_fields(b, fs, smi) for smi, b, _k in built]
    mu, var = model.predict_with_variance(AtomTable.from_molecules(fields_mf))
    sd = np.sqrt(np.maximum(np.asarray(var, float), 0.0))
    best = float(np.max(np.asarray(model.y, float))) if model.y is not None else None
    score = acquisition_scores(acquire, mu, sd, best=best, beta=beta, xi=xi,
                               distance=np.zeros(len(mu)) if acquire == "maxmin" else None)
    if progress:
        progress("scored", 0.8)

    cands = []
    for i, (smi, _b, key) in enumerate(built):
        nm, meas = made.get(key, (None, float("nan")))
        cands.append(Candidate(smiles=smi, label=int(label), mean=float(mu[i]),
                               sd=float(sd[i]), score=float(score[i]),
                               delta=float(mu[i] - parent_mean), known=smi in seen,
                               made=nm, measured=meas))
    order = np.argsort(-np.asarray([c.score for c in cands], float))
    cands = [cands[int(i)] for i in order]

    for c in cands:
        i = next(k for k, (smi, _b, _k) in enumerate(built) if smi == c.smiles)
        c.molecule = Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(built[i][1])))             if pose == "molecule" else ""
    for c in cands[:int(with_fields)]:
        i = next(k for k, (smi, _b, _k) in enumerate(built) if smi == c.smiles)
        if pose == "molecule":
            # there is no frame, so there is no field difference to draw; what the invariant
            # model can say is how much of its prediction comes from the group that was put in
            c.group_value, c.group_total, c.group_atoms = group_contribution(
                model, fields_mf[i], built[i][1], core)
            continue
        if parent_fields is None:
            break
        # no grid: the topomer sites sit a hundred angstrom apart and a grid over them would be
        # mostly empty space, while the closed-form numbers are exact anyway
        c.fields = mismatch(fields_mf[i], parent_fields, fs, grid=False)

    parent_key = _key(parent_frags)
    values = np.asarray([c.score for c in cands], float)
    flat = bool(len(values) > 1 and float(np.nanmax(values) - np.nanmin(values)) < 1e-12)
    return DesignResult(parent=parent_name, label=int(label), acquire=acquire, candidates=cands,
                        flat_score=flat,
                        parent_mean=parent_mean, parent_sd=parent_sd,
                        parent_measured=made.get(parent_key, (None, float("nan")))[1],
                        posed=len(built), failed=failed)
