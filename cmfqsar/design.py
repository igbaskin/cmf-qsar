"""Closing the loop: what to put at a position, ranked by the model (plan item E3).

Everything else in this program scores molecules somebody has already drawn.  This module makes
the suggestion itself: take a parent compound of a series, take one of its attachment points,
replace what is there with every fragment of a library, and rank the results by what the model
expects of them.

The pose problem that stops this being easy is solved by not having one.  A new substituent has
no crystal structure and no alignment, and superimposing it on anything would be a guess; the
topomer rule (:mod:`cmfqsar.topomer`) builds it into a pose that is a deterministic function of
its own topology, so the candidate is posed by exactly the rule that posed the molecules the
model was fitted on.  The comparison is then like with like at one attachment point, which is
all a substituent ranking needs.  It is also the reason this works on a topomer model and not on
an aligned one: a model fitted on superimposed whole molecules has no way to place a group that
was never synthesised.

Three numbers come back for every candidate and they answer different questions:

* the **prediction**, which is what the model expects;
* the **posterior standard deviation**, which says whether the model has any business having an
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
#: A working default rather than a claim: it is a hand-made list of the groups that turn up in
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
    label: int                    # the core atom it hangs off
    mean: float = float("nan")
    sd: float = float("nan")
    score: float = float("nan")   # the acquisition score it was ranked by
    delta: float = float("nan")   # prediction minus the parent's prediction
    known: bool = False           # the series already uses this fragment here
    made: Optional[str] = None    # the name of the molecule that is this candidate, if any
    measured: float = float("nan")   # and what it measured
    fields: List = _field(default_factory=list)   # FieldMismatch against the parent

    def describe(self) -> str:
        mark = "known" if self.known else "new  "
        got = "" if self.made is None else f"   made: {self.made} ({self.measured:.2f})"
        return (f"{self.smiles[:28]:28s} {mark}  predicted {self.mean:7.3f} +- {self.sd:5.3f}"
                f"   delta {self.delta:+6.3f}   score {self.score:+8.4f}{got}")


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

    def top(self, n: int = 10) -> List[Candidate]:
        return self.candidates[:int(n)]

    def describe(self) -> str:
        new = sum(1 for c in self.candidates if not c.known)
        return (f"{len(self.candidates)} candidates at position {self.label} of {self.parent} "
                f"({new} not used anywhere in the series), ranked by {self.acquire}; "
                f"{len(self.failed)} fragments could not be posed")


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
    :func:`cmfqsar.topomer.substituents` uses to name the core atom a group hangs off, so that a
    fragment from a library and a fragment cut off a molecule are the same kind of object.
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
    except Exception:                            # noqa: BLE001 - a fragment that will not stand
        return None
    return mol


def fragment_smiles(frag) -> str:
    """The canonical name of a fragment, with hydrogens implicit.

    A group cut out of an SDF carries its hydrogens as atoms; the same group written into a
    library by hand does not, and the two spell themselves differently. Everything here -- which
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
    """
    labels = list(labels) if labels is not None else series_labels([parent], core)
    others = [f for f in substituents(parent, core) if attachment_label(f) != label]
    cache = {} if cache is None else cache
    out, failed = [], []
    for smi in fragments:
        frag = as_fragment(smi, label)
        if frag is None:
            failed.append(smi)
            continue
        try:
            built = topomer_assembly(others + [frag], labels, spec, cache)
        except Exception:                        # noqa: BLE001 - one fragment, not the library
            built = None
        if built is None:
            failed.append(smi)
            continue
        out.append((fragment_smiles(frag), built, _key(others + [frag])))
    return out, failed


def design(model, parent: Chem.Mol, core: Chem.Mol, label: int,
           fragments: Optional[Sequence[str]] = None, labels: Optional[Sequence[int]] = None,
           acquire: str = "ei", beta: float = 1.0, xi: float = 0.0,
           spec: TopomerSpec = TopomerSpec(), parent_name: str = "the parent",
           series: Optional[Sequence[Chem.Mol]] = None,
           series_names: Optional[Sequence[str]] = None,
           series_y: Optional[Sequence[float]] = None,
           known: Optional[Sequence[str]] = None, with_fields: int = 0,
           progress=None) -> DesignResult:
    """Rank replacements at one position of `parent` under a fitted topomer model.

    `series`, `series_names` and `series_y` are the molecules the model was fitted on; they are
    used only to say which suggestions have already been made and what those measured, which is
    the one free check there is on a design loop.
    """
    from .active import acquisition_scores
    from .fielddiff import mismatch

    fs: FieldSet = model.fieldset
    frags = list(fragments) if fragments is not None else list(COMMON_FRAGMENTS)
    labels = list(labels) if labels is not None else series_labels([parent], core)
    cache: Dict[str, Chem.Mol] = {}

    built, failed = enumerate_replacements(parent, core, label, frags, labels, spec, cache)
    if progress:
        progress(f"{len(built)} candidates posed", 0.3)

    # the parent itself, posed by the same rule, is the thing every candidate is compared with
    parent_frags = substituents(parent, core)
    parent_built = topomer_assembly(parent_frags, labels, spec, cache)
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
            except Exception:                    # noqa: BLE001 - a molecule the core misses
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

    for c in cands[:int(with_fields)]:
        if parent_fields is None:
            break
        i = next(k for k, (smi, _b, _k) in enumerate(built) if smi == c.smiles)
        # no grid: the topomer sites sit a hundred angstrom apart and a grid over them would be
        # mostly empty space, while the closed-form numbers are exact anyway
        c.fields = mismatch(fields_mf[i], parent_fields, fs, grid=False)

    parent_key = _key(parent_frags)
    return DesignResult(parent=parent_name, label=int(label), acquire=acquire, candidates=cands,
                        parent_mean=parent_mean, parent_sd=parent_sd,
                        parent_measured=made.get(parent_key, (None, float("nan")))[1],
                        posed=len(built), failed=failed)
