"""Inverse design on prototypes: candidates, an oracle and a loop (plan item M11).

A co-pattern says which pair of field types at which distance a crystal should gain. Acting on
that needs three things this module provides.

**A candidate generator.** A *prototype* is one cell whose sites are grouped into classes, with a
list of the elements each class may take. Substituting them enumerates candidate structures.
Two modes cover what the materials items ask for:

* *composition*: every site of a class takes the same element, which is the ABO3 case -- one
  prototype, many compositions;
* *arrangement*: a fixed number of the sites of a class take a substituent and the others keep
  what the prototype has, which is the point-defect case -- one composition, many arrangements.
  The element name ``"vac"`` removes the atom instead of replacing it, so a vacancy series is an
  arrangement series.

The number of candidates is a product and grows quickly, so nothing is built until it is asked
for: `Prototype.n_candidates` counts them, `candidate_at` builds one from its index, and
`candidates` either walks them all or draws a sample without replacement.

**A test for duplicates.** Two assignments that a symmetry operation of the prototype maps onto
each other are one structure. `site_permutations` reads those operations from spglib and turns
them into permutations of the substitutable sites; `unique` keeps one assignment per orbit. Where
spglib finds no symmetry, `unique_by_channels` compares the descriptor channels instead, which is
also the honest test: two cells with the same channels are the same structure for the model,
whatever their coordinates say.

**An oracle and a loop.** The oracle is what computes the property of a proposed structure.
`LookupOracle` reads it from a table, which is how the loop is measured on a published set;
`CalculatorOracle` runs any ASE calculator, with or without a relaxation, which is how a real
loop runs -- the same interface takes a UMA model or a VASP calculator. `design_loop` then
proposes, computes and refits: at every step it fits the kernel model on what the oracle has
answered so far, chooses the ridge by the evidence, scores every candidate by one of the rules of
`cmfqsar/active.py` and asks the oracle about the best one. The loop never looks at a value the
oracle has not been asked for, and never asks twice about the same candidate.

Minimising is the usual materials goal (a formation energy), and the rules of `active.py` all
maximise, so `design_loop(maximise=False)` negates the target inside. With that, ``"ucb"`` is the
lower confidence bound of plan item E6: a candidate is proposed only when the model is both
optimistic and sure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import product
from math import comb
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .active import ACQUISITIONS, acquisition_scores
from .active import _distance_to_training as _distances
from .active import _fit_predict as _posterior   # the fit of the acquisition study, item E2

#: the element name that removes an atom instead of replacing it
VACANCY = "vac"

#: the rules a loop may be run with, from `cmfqsar/active.py`
RULES = ACQUISITIONS


# --------------------------------------------------------------------------- #
# The prototype and its candidates
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SiteClass:
    """One group of sites of a prototype and the elements they may take.

    `count` is what separates the two modes. ``None`` means every site of the class takes the
    same element, so the class contributes one candidate per element. An integer *k* means
    exactly *k* of the sites take one of `elements` and the rest keep what the prototype has, so
    the class contributes one candidate per element and per choice of *k* sites.
    """

    name: str
    sites: Tuple[int, ...]
    elements: Tuple[str, ...]
    count: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError(f"site class {self.name!r} has no sites")
        if not self.elements:
            raise ValueError(f"site class {self.name!r} has no elements to put on its sites")
        if self.count is not None and not 0 <= int(self.count) <= len(self.sites):
            raise ValueError(f"site class {self.name!r} asks for {self.count} of its "
                             f"{len(self.sites)} sites")

    @property
    def n_options(self) -> int:
        """How many assignments this class alone allows."""
        if self.count is None:
            return len(self.elements)
        return len(self.elements) * comb(len(self.sites), int(self.count))

    def option_at(self, index: int) -> Tuple[str, Tuple[int, ...]]:
        """The `index`-th assignment: the element, and the sites that take it."""
        if not 0 <= index < self.n_options:
            raise ValueError(f"site class {self.name!r} has {self.n_options} assignments, "
                             f"not {index + 1}")
        if self.count is None:
            return self.elements[index], tuple(self.sites)
        per = comb(len(self.sites), int(self.count))
        element = self.elements[index // per]
        return element, _unrank_combination(self.sites, int(self.count), index % per)


def _unrank_combination(items: Sequence[int], k: int, rank: int) -> Tuple[int, ...]:
    """The `rank`-th choice of `k` of `items`, counting in the order `itertools.combinations` does.

    Written out rather than taken from `itertools`, so that one candidate of a very long
    enumeration can be built without walking the ones before it.
    """
    n = len(items)
    out: List[int] = []
    start = 0
    left = int(k)
    r = int(rank)
    while left > 0:
        for i in range(start, n - left + 1):
            block = comb(n - i - 1, left - 1)
            if r < block:
                out.append(items[i])
                start, left = i + 1, left - 1
                break
            r -= block
        else:                                  # pragma: no cover - r was checked by the caller
            raise ValueError("the rank is past the end of the enumeration")
    return tuple(out)


@dataclass(frozen=True)
class Candidate:
    """One proposed structure: where it came from, what it is called, and its cell."""

    index: int
    label: str
    assignment: Tuple[Tuple[str, Tuple[int, ...]], ...]
    atoms: Any

    @property
    def formula(self) -> str:
        """The composition, where the structure can give one, and the label otherwise.

        A candidate the generator built holds an ASE cell; a candidate taken from a published set
        may hold the field record the descriptor was computed from, which has no formula.
        """
        get = getattr(self.atoms, "get_chemical_formula", None)
        return get() if callable(get) else self.label


@dataclass(frozen=True)
class Prototype:
    """One cell, its site classes, and everything the generator needs."""

    atoms: Any
    classes: Tuple[SiteClass, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.classes:
            raise ValueError("a prototype needs at least one site class to substitute on")
        seen: Dict[int, str] = {}
        n = len(self.atoms)
        for c in self.classes:
            for s in c.sites:
                if not 0 <= s < n:
                    raise ValueError(f"site class {c.name!r} names site {s}, and the prototype "
                                     f"has {n} atoms")
                if s in seen:
                    raise ValueError(f"site {s} is in both {seen[s]!r} and {c.name!r}; a site "
                                     f"belongs to one class")
                seen[s] = c.name

    @staticmethod
    def by_symbol(atoms, choices: Mapping[str, Sequence[str]],
                  counts: Optional[Mapping[str, Optional[int]]] = None,
                  name: str = "") -> "Prototype":
        """A prototype whose classes are the elements of `atoms`: ``{"Sr": ["Ca", "Ba"], ...}``.

        `counts` gives the class of an element an arrangement count, as `SiteClass.count`.
        """
        symbols = list(atoms.get_chemical_symbols())
        classes = []
        for symbol, elements in choices.items():
            sites = tuple(i for i, s in enumerate(symbols) if s == symbol)
            if not sites:
                raise ValueError(f"the prototype has no {symbol} site; its elements are "
                                 f"{', '.join(sorted(set(symbols)))}")
            k = None if counts is None else counts.get(symbol)
            classes.append(SiteClass(symbol, sites, tuple(elements), k))
        return Prototype(atoms, tuple(classes), name or atoms.get_chemical_formula())

    @property
    def n_candidates(self) -> int:
        """How many candidates the classes allow, counted and not built."""
        out = 1
        for c in self.classes:
            out *= c.n_options
        return out

    @property
    def sites(self) -> Tuple[int, ...]:
        """Every substitutable site, in order."""
        return tuple(sorted(s for c in self.classes for s in c.sites))

    def assignment_at(self, index: int) -> Tuple[Tuple[str, Tuple[int, ...]], ...]:
        """The `index`-th assignment of elements to sites, as one entry per class."""
        total = self.n_candidates
        if not 0 <= index < total:
            raise ValueError(f"the prototype allows {total} candidates, not {index + 1}")
        rest = int(index)
        out = []
        for k, c in enumerate(self.classes):   # mixed radix, the first class moving slowest
            per = 1
            for later in self.classes[k + 1:]:
                per *= later.n_options
            out.append(c.option_at(rest // per))
            rest = rest % per
        return tuple(out)

    def candidate_at(self, index: int) -> Candidate:
        """Build the `index`-th candidate."""
        assignment = self.assignment_at(index)
        atoms = self.atoms.copy()
        symbols = list(atoms.get_chemical_symbols())
        drop: List[int] = []
        for element, sites in assignment:
            for s in sites:
                if element == VACANCY:
                    drop.append(s)
                else:
                    symbols[s] = element
        atoms.set_chemical_symbols(symbols)
        if drop:
            del atoms[sorted(drop, reverse=True)]
        if not len(atoms):
            raise ValueError(f"candidate {index} of {self.name!r} has no atoms left")
        label = ";".join(f"{c.name}:{element}@" + ",".join(str(s) for s in sites)
                         for c, (element, sites) in zip(self.classes, assignment))
        return Candidate(int(index), label, assignment, atoms)

    def site_codes(self, index: int) -> Tuple[int, ...]:
        """The assignment as one number per substitutable site, for the duplicate test."""
        order = {s: i for i, s in enumerate(self.sites)}
        codes = [0] * len(order)
        assignment = self.assignment_at(index)
        for k, (c, (element, sites)) in enumerate(zip(self.classes, assignment)):
            base = 1 + sum(len(x.elements) for x in self.classes[:k])
            for s in sites:
                codes[order[s]] = base + c.elements.index(element)
        return tuple(codes)


def as_candidates(structures: Sequence[Any],
                  labels: Optional[Sequence[str]] = None) -> List[Candidate]:
    """A pool of candidates from structures that are already built.

    This is what a retrospective measurement needs: the pool is a published set, every candidate
    keeps its row as its index, and the loop cannot tell it from a pool the generator proposed.
    """
    out = []
    for i, atoms in enumerate(structures):
        label = str(labels[i]) if labels is not None else f"row {i}"
        out.append(Candidate(int(i), label, (), atoms))
    return out


def cubic_perovskite(a: float = 3.9, a_site: str = "Sr", b_site: str = "Ti",
                     anion: str = "O"):
    """The five-atom cubic ABO3 cell, which is the prototype the measurement of M11 uses.

    A at the corner, B at the body centre and the three anions at the face centres. ASE does not
    build this structure, and writing it out keeps the site order fixed, which the site classes
    of a prototype depend on.
    """
    from ase import Atoms

    return Atoms([a_site, b_site, anion, anion, anion],
                 scaled_positions=[(0.0, 0.0, 0.0), (0.5, 0.5, 0.5),
                                   (0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)],
                 cell=float(a) * np.eye(3), pbc=True)


def candidates(proto: Prototype, limit: Optional[int] = None, seed: int = 0,
               indices: Optional[Sequence[int]] = None) -> List[Candidate]:
    """Every candidate of a prototype, or a sample of `limit` of them without replacement.

    A sample is drawn over the index range, so a pool of a few thousand can be taken from an
    enumeration of billions without building the ones left out.
    """
    total = proto.n_candidates
    if indices is not None:
        return [proto.candidate_at(int(i)) for i in indices]
    if limit is None or int(limit) >= total:
        return [proto.candidate_at(i) for i in range(total)]
    rng = np.random.default_rng(seed)
    want = int(limit)
    if total <= np.iinfo(np.int64).max:
        chosen = np.sort(rng.choice(total, want, replace=False))
        return [proto.candidate_at(int(i)) for i in chosen]
    taken: Dict[int, None] = {}
    while len(taken) < want:                   # pragma: no cover - needs a huge enumeration
        taken[int(rng.integers(0, 1 << 62)) % total] = None
    return [proto.candidate_at(i) for i in sorted(taken)]


# --------------------------------------------------------------------------- #
# Which candidates are the same structure
# --------------------------------------------------------------------------- #
def site_permutations(proto: Prototype, symprec: float = 1e-3,
                      tol: float = 1e-3) -> np.ndarray:
    """How the symmetry operations of the prototype permute its substitutable sites.

    Shape (G, S): row *g* holds, for every substitutable site, the site that operation *g* sends
    it to. Operations that send a substitutable site onto a site of another class, or onto no
    site at all, are left out, because they do not map the set of candidates onto itself. The
    identity is always the first row.
    """
    from .periodic import host_symmetry

    rotations, translations = host_symmetry(proto.atoms, symprec=float(symprec))
    frac = np.asarray(proto.atoms.get_scaled_positions(), float)
    sites = np.asarray(proto.sites, int)
    order = {int(s): i for i, s in enumerate(sites)}
    rows = []
    for R, t in zip(rotations, translations):
        moved = (frac[sites] @ np.asarray(R, float).T) + np.asarray(t, float)
        row = []
        for point in moved:
            d = frac[sites] - point
            d -= np.round(d)
            hit = np.where(np.all(np.abs(d) < float(tol), axis=1))[0]
            if len(hit) != 1:
                row = []
                break
            row.append(int(hit[0]))
        if row and len(set(row)) == len(row):
            rows.append(row)
    if not rows:
        rows = [[order[int(s)] for s in sites]]
    out = np.asarray(rows, int)
    identity = np.arange(len(sites))
    keep = [i for i in range(len(out)) if not np.array_equal(out[i], identity)]
    return np.vstack([identity[None, :], out[keep]]) if keep else identity[None, :]


def unique(proto: Prototype, pool: Sequence[Candidate],
           perms: Optional[np.ndarray] = None) -> Tuple[List[Candidate], int]:
    """One candidate per symmetry orbit, and how many were dropped as duplicates."""
    perms = site_permutations(proto) if perms is None else np.asarray(perms, int)
    seen: Dict[Tuple[int, ...], None] = {}
    kept: List[Candidate] = []
    for cand in pool:
        codes = np.asarray(proto.site_codes(cand.index), int)
        key = min(tuple(codes[p]) for p in perms)
        if key in seen:
            continue
        seen[key] = None
        kept.append(cand)
    return kept, len(pool) - len(kept)


def channel_keys(crystals: Sequence[Any], spec, n_fields: int,
                 decimals: int = 8) -> List[Tuple]:
    """One hashable key per structure from its descriptor channels, rounded to `decimals`.

    Two structures with the same key are the same structure for a model built on that
    descriptor, which is the test that matters: a two-point descriptor gives one key to a
    homometric pair, and the three-body channels of plan item M10 give them two.
    """
    F = feature_matrix(crystals, spec, n_fields)
    scale = float(np.max(np.abs(F))) or 1.0
    return [tuple(np.round(row / scale, int(decimals))) for row in F]


def unique_by_channels(crystals: Sequence[Any], spec, n_fields: int,
                       decimals: int = 8) -> Tuple[List[int], int]:
    """The rows to keep so that no two structures share their channels, and how many were
    dropped."""
    seen: Dict[Tuple, None] = {}
    keep = []
    for i, key in enumerate(channel_keys(crystals, spec, n_fields, decimals)):
        if key in seen:
            continue
        seen[key] = None
        keep.append(i)
    return keep, len(crystals) - len(keep)


# --------------------------------------------------------------------------- #
# The descriptor of a pool
# --------------------------------------------------------------------------- #
def feature_matrix(crystals: Sequence[Any], spec, n_fields: int,
                   dtype=float) -> np.ndarray:
    """The feature matrix of a pool under whichever periodic descriptor `spec` names."""
    from .periodic import PeriodicSpec, periodic_feature_matrix
    from .reciprocal import ReciprocalSpec, reciprocal_feature_matrix
    from .threebody import ThreeBodySpec, threebody_feature_matrix

    if isinstance(spec, ThreeBodySpec):
        return threebody_feature_matrix(crystals, spec, n_fields=n_fields, dtype=dtype)
    if isinstance(spec, ReciprocalSpec):
        return reciprocal_feature_matrix(crystals, spec, n_fields=n_fields)
    if isinstance(spec, PeriodicSpec):
        return periodic_feature_matrix(crystals, spec, n_fields=n_fields)
    raise ValueError(f"no descriptor is known for a {type(spec).__name__}")


def block_sizes(spec, n_fields: int) -> List[int]:
    """The channel block widths of whichever descriptor `spec` names."""
    from .periodic import PeriodicSpec, periodic_block_sizes
    from .reciprocal import ReciprocalSpec, reciprocal_block_sizes
    from .threebody import ThreeBodySpec, threebody_block_sizes

    if isinstance(spec, ThreeBodySpec):
        return threebody_block_sizes(spec, n_fields)
    if isinstance(spec, ReciprocalSpec):
        return reciprocal_block_sizes(spec, n_fields)
    if isinstance(spec, PeriodicSpec):
        return periodic_block_sizes(spec, n_fields)
    raise ValueError(f"no descriptor is known for a {type(spec).__name__}")


def pool_kernel(F: np.ndarray, blocks: Sequence[int]) -> np.ndarray:
    """The cosine-normalised linear kernel of a pool, each channel block on its own scale."""
    from .invariant import autocorrelation_kernel, channel_scale

    blocks = [int(b) for b in blocks]
    scale = channel_scale(F, blocks[0], blocks=blocks)
    return autocorrelation_kernel(F, scale=scale, n_bins=blocks[0], blocks=blocks)


# --------------------------------------------------------------------------- #
# The oracles
# --------------------------------------------------------------------------- #
class LookupOracle:
    """The property of a candidate, read from a table rather than computed.

    This is what a retrospective measurement uses: the pool is a published set, the values are
    its published labels, and the loop may not see a value it has not asked for. `calls` counts
    the questions, which is the budget the gate of plan item M11 is stated in.
    """

    def __init__(self, values: Mapping[int, float] | Sequence[float], name: str = "lookup"):
        self.values = values
        self.name = name
        self.calls = 0
        self.asked: List[int] = []

    def __call__(self, cand: Candidate) -> float:
        key = cand.index
        if isinstance(self.values, Mapping):
            if key not in self.values:
                raise ValueError(f"no value is tabulated for candidate {key} ({cand.label})")
            value = self.values[key]
        else:
            if not 0 <= key < len(self.values):
                raise ValueError(f"no value is tabulated for candidate {key}")
            value = self.values[key]
        self.calls += 1
        self.asked.append(int(key))
        return float(value)


class CalculatorOracle:
    """The energy of a candidate from an ASE calculator, with an optional relaxation.

    `make` is called with no arguments and returns a fresh calculator, so that a calculator that
    keeps state cannot carry it from one candidate to the next. The same interface takes a
    machine-learned potential or a VASP calculator; what changes is the cost of a call and not
    the loop.
    """

    def __init__(self, make: Callable[[], Any], relax: bool = True, fmax: float = 0.05,
                 steps: int = 60, per_atom: bool = True, name: str = "calculator"):
        self.make = make
        self.relax = bool(relax)
        self.fmax = float(fmax)
        self.steps = int(steps)
        self.per_atom = bool(per_atom)
        self.name = name
        self.calls = 0
        self.seconds = 0.0
        self.relaxed: Dict[int, Any] = {}

    def __call__(self, cand: Candidate) -> float:
        from ase.optimize import BFGS

        t0 = time.time()
        atoms = cand.atoms.copy()
        atoms.calc = self.make()
        if self.relax:
            opt = BFGS(atoms, logfile=None)
            opt.run(fmax=self.fmax, steps=self.steps)
        energy = float(atoms.get_potential_energy())
        self.calls += 1
        self.seconds += time.time() - t0
        self.relaxed[int(cand.index)] = atoms
        return energy / len(atoms) if self.per_atom else energy


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #
def design_loop(pool: Sequence[Candidate], oracle: Callable[[Candidate], float], *,
                start: Sequence[int], budget: int, kernel: Optional[np.ndarray] = None,
                spec=None, n_fields: Optional[int] = None, rule: str = "ucb",
                beta: float = 1.0, xi: float = 0.0,
                lams: Sequence[float] = (1e-3, 1e-2, 0.1, 1.0), maximise: bool = True,
                fit_bias: bool = True, seed: int = 0,
                log: Optional[Callable[[str], None]] = None) -> dict:
    """Propose, compute and refit, until the budget of oracle calls is spent.

    `start` are positions in `pool` whose values the oracle is asked for before the loop begins;
    they are the calculations a design study starts from. `budget` is how many further calls the
    loop may make. The kernel over the pool is given or built from `spec`, and nothing about the
    pool changes while the loop runs, so a candidate the loop never proposes costs nothing beyond
    its descriptor.

    With `maximise` false the target is negated inside, so that every rule still prefers a larger
    score and ``"ucb"`` reads as the lower confidence bound.

    Returns the record: one entry per step with what was proposed and what the oracle answered,
    the best value found, and which candidate holds it.
    """
    from .evidence import log_evidence

    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; one of {', '.join(RULES)}")
    n = len(pool)
    if n == 0:
        raise ValueError("the candidate pool is empty")
    if kernel is None:
        if spec is None or n_fields is None:
            raise ValueError("give the loop a kernel over the pool, or a spec and a field count "
                            "to build one from")
        F = feature_matrix([c.atoms for c in pool], spec, int(n_fields))
        kernel = pool_kernel(F, block_sizes(spec, int(n_fields)))
    K = np.asarray(kernel, float)
    if K.shape != (n, n):
        raise ValueError(f"the kernel is {K.shape} and the pool has {n} candidates")
    sign = 1.0 if maximise else -1.0
    rng = np.random.default_rng(seed)
    values: Dict[int, float] = {}
    have: List[int] = []
    for i in dict.fromkeys(int(x) for x in start):
        if not 0 <= i < n:
            raise ValueError(f"the start set names candidate {i} of a pool of {n}")
        values[i] = float(oracle(pool[i]))
        have.append(i)
    if not have:
        raise ValueError("the loop needs at least one candidate computed to start from")
    rest = [i for i in range(n) if i not in values]
    steps: List[dict] = []
    best = max(values.items(), key=lambda kv: sign * kv[1])
    for step in range(int(budget)):
        if not rest:
            break
        t0 = time.time()
        h, p = np.asarray(have, int), np.asarray(rest, int)
        y = sign * np.asarray([values[i] for i in have], float)
        pick = None
        for lam in lams:
            e = log_evidence(K[np.ix_(h, h)], y, float(lam), fit_bias=fit_bias)
            if np.isfinite(e) and (pick is None or e > pick[0]):
                pick = (float(e), float(lam))
        lam = pick[1] if pick else float(lams[-1])
        full = np.zeros(n)
        for i in have:
            full[i] = sign * values[i]
        mu, sd = _posterior(K, full, h, p, lam, fit_bias)
        dist = _distances(K[np.ix_(p, h)], np.diag(K)[p], np.diag(K)[h])
        score = acquisition_scores(rule, mu, sd, best=float(np.max(y)), beta=beta, xi=xi,
                                   distance=dist, rng=rng)
        j = int(np.argmax(score))
        taken = int(p[j])
        value = float(oracle(pool[taken]))
        values[taken] = value
        have.append(taken)
        rest.remove(taken)
        if sign * value > sign * best[1]:
            best = (taken, value)
        steps.append({"step": step + 1, "taken": taken, "label": pool[taken].label,
                      "formula": pool[taken].formula, "predicted": float(sign * mu[j]),
                      "sd": float(sd[j]), "value": value, "best": float(best[1]),
                      "lam": float(lam), "seconds": time.time() - t0})
        if log is not None:
            log(f"  step {step + 1}: {pool[taken].formula} predicted "
                f"{sign * mu[j]:+.4f} +- {sd[j]:.4f}, computed {value:+.4f}, "
                f"best {best[1]:+.4f}")
    return {"rule": rule, "pool": n, "start": [int(i) for i in start], "budget": int(budget),
            "maximise": bool(maximise), "steps": steps, "best": float(best[1]),
            "best_index": int(best[0]), "best_label": pool[best[0]].label,
            "best_formula": pool[best[0]].formula,
            "start_best": float(max((values[i] for i in dict.fromkeys(int(x) for x in start)),
                                    key=lambda v: sign * v)),
            "computed": len(values), "oracle_calls": int(getattr(oracle, "calls", len(values)))}


def first_better(record: dict, threshold: float) -> Optional[int]:
    """The step at which the loop first found a candidate past `threshold`, or None.

    "Past" means above for a loop that maximises and below for one that minimises, so the same
    call reads a formation energy and a figure of merit.
    """
    sign = 1.0 if record.get("maximise", True) else -1.0
    for entry in record["steps"]:
        if sign * entry["value"] > sign * float(threshold):
            return int(entry["step"])
    return None


def summarise(record: dict) -> str:
    """One line for a log: what the loop spent and what it found."""
    gain = record["best"] - record["start_best"]
    return (f"{record['rule']:8s} {len(record['steps']):4d} calls, best "
            f"{record['best']:+.4f} ({record['best_formula']}), "
            f"{gain:+.4f} against the start set")
