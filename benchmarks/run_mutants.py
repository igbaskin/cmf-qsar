#!/usr/bin/env python
"""
Point mutants of a protein as fields in the frame of the mutated residue (plan item PP1).

The question
------------
Does a field description of what a mutation changes predict how much it changes the stability of
the protein, better than the substitution alone?

The samples are single substitutions in four small natural domains from the mega-scale data set
of Tsuboyama et al. (2023).  The target is ``ddG_ML``: the unfolding free energy of the variant
minus that of the wild type, in kcal/mol, so a negative value is a destabilising mutation.  The
structures are built by `examples/make_mutant_examples.py` with PDBFixer and relaxed with the
Amber ff14SB force field through OpenMM, with every atom outside the mutated residue held fixed.

What the descriptions are
-------------------------
Every one of them lives in the frame of the mutated residue (origin at the alpha carbon, the
first axis along the bond to the nitrogen, the plane through the carbonyl carbon), which is what
makes a change at one position comparable with a change at another.

``difference``   the side chain of the mutant minus the side chain of the wild type, as one
                 signed field (:func:`cmfqsar.mutants.site_difference`).  This is the sample the
                 item proposes: the change itself.
``new``          the side chain of the mutant alone.
``context``      the surroundings of the site in the wild type, without the side chain that is
                 replaced, under a Gaussian envelope.  It is the same for every substitution at
                 one position, so on its own it can only predict the mean of the position; it is
                 here to be joined with the others.
``difference+context``, ``new+context``  the two kernels summed with a weight chosen inside the
                 training part of every fold.

The references
--------------
``substitution`` a linear regression on the substitution alone: the change of residue volume, of
                 hydropathy and of formal charge, and the one-hot codes of the two residues.  It
                 knows nothing about where in the protein the position is.
``counts``       the atom-type counts in six distance shells about the alpha carbon of the site
                 in the wild type, the per-atom reference of items LP2 and SP2, plus the one-hot
                 codes of the two residues.
``burial``       how many atoms surround the site, as a count and as a sum of envelope factors at
                 each envelope width, with the three substitution columns.  The field of the
                 surroundings is what carries the joined description, and burial is the first
                 thing it could be carrying, measured the way a biochemist would.
``mean``         the mean of the training part: q2 = 0 by construction.

The protocol
------------
Two ways of holding out, because they ask different questions:

* **by position** (ten folds): every substitution at one position is in the same fold, so a
  position in the test part was never seen.  This is *not* a clean test of a description that
  uses the surroundings of the site, because the surroundings of one residue and those of the
  next share most of their atoms.
* **by segment** (ten folds): nine consecutive residues at a time, which removes the sequence
  neighbours of every held-out site together with the site.
* **by domain** (four folds, one per protein): the whole protein is held out.  This is harder
  and is the test a user of such a model needs, since a new protein is the case.

The ridge and the kernel are chosen on five inner folds of the same kind inside every training
part (`run_band.nested_choice`), so nothing in the choice sees the held-out samples.  A
difference of q2 carries a 95 % bootstrap interval over the held-out unit.

Before any fit, the correlation of each kernel with the target is reported
(`run_spectra_ir.target_alignment`), which is the measure item EC5 established: it says whether
samples a description finds alike have alike targets, and needs no model.

Usage
-----
    python benchmarks/run_mutants.py --prepare             # the fields, once
    python benchmarks/run_mutants.py                       # the measurement
    python benchmarks/run_mutants.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, MoleculeFields, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices, normalize_kernels, self_kernels  # noqa: E402
from cmfqsar.localenv import count_kernel  # noqa: E402
from cmfqsar.model import krr_fit  # noqa: E402
from cmfqsar import mutants as MU  # noqa: E402

import run_band as B  # noqa: E402

DOMAINS = ("1O6X", "1UFM", "2K5N", "3ZGK")
EXAMPLES = os.path.join(ROOT, "examples", "mutants")
BUILT = os.path.join(ROOT, "benchmarks", "cache", "megascale", "built")
PREPARED = os.path.join(ROOT, "benchmarks", "cache", "megascale", "fields")
RESULTS = os.path.join(ROOT, "benchmarks", "results", "mutants")

#: the envelope decays of the context, in inverse square angstrom; 0.05 reaches about 7.7 A
CONTEXT_RHOS = (0.2, 0.1, 0.05)
#: the envelope decays offered to the inner folds (a subset, to keep the fit affordable)
CONTEXT_CHOICE = (0.2, 0.1)
#: the attenuations of the field kernel
ALPHAS = (0.3, 1.0, 3.0)
#: the attenuation of the context kernel, fixed to keep the number of settings small
CONTEXT_ALPHA = 1.0
#: the weights of the context kernel in the sum offered to the inner folds
MIXES = (0.25, 0.5)
FOLDS = 10
INNER = 3
REPEATS = 3
#: at most this many substitutions per position enter the measurement.  The cost of a kernel
#: grows with the square of the sample count and the cost of a fit with its cube, so a
#: measurement that keeps every position of every domain has to take part of each.  Positions
#: are kept whole because a position is what the folds hold out.
MAX_PER_POSITION = 4
#: how many consecutive residues one group of the `segment` hold-out covers.  The context at
#: the widest envelope reaches about 9.6 angstrom, which along the chain is three or four
#: residues either side of the centre, so a group of nine removes those neighbours.
SEGMENT = 9
LAMS = B.WIDE_LAMS
SHELLS = (1.8, 2.8, 3.8, 5.0, 6.5, 8.5)


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------- #
# Stage 1: the fields of every mutant, saved small
# --------------------------------------------------------------------------- #

def _pack(m: MoleculeFields) -> Dict[str, np.ndarray]:
    return {"coords": np.asarray(m.coords, np.float32), "weights": np.asarray(m.weights, np.float32),
            "radii": np.asarray(m.radii, np.float32),
            "symbols": np.array(list(m.symbols)), "atom_types": np.array(list(m.atom_types))}


def _unpack(z, prefix: str, name: str) -> MoleculeFields:
    return MoleculeFields(
        name=name, coords=np.asarray(z[prefix + "coords"], float),
        weights=np.asarray(z[prefix + "weights"], float), radii=np.asarray(z[prefix + "radii"], float),
        symbols=[str(s) for s in z[prefix + "symbols"]],
        atom_types=[str(s) for s in z[prefix + "atom_types"]],
        bonds=np.zeros((0, 2), int), charges=np.zeros(len(z[prefix + "radii"])), charge_source="file")


def _prepare_domain(domain: str, labels: Sequence[str], built: str, dest: str):
    """The side-chain fields of a list of mutants of one domain, in this process.  The wild
    type is read and typed once, which is most of the work of a single mutant."""
    from cmfqsar.mutants import Mutation, Structure, protein_of, side_chain_field
    fs = FieldSet.create(charge_model="file")
    wt = Structure.load(os.path.join(built, "wild_type.npz"))
    wt_mol = protein_of(wt)
    wt_fields = build_fields(wt_mol, fs, name="wild type")
    done, failed = 0, []
    wt_cache: Dict[int, MoleculeFields] = {}
    t0 = time.time()
    for k, label in enumerate(labels, 1):
        out = os.path.join(dest, label + ".npz")
        if os.path.exists(out):
            done += 1
            continue
        try:
            m = Mutation.parse(label)
            s = Structure.load(os.path.join(built, label + ".npz"))
            mol = protein_of(s)
            f = build_fields(mol, fs, name=label)
            new = side_chain_field(f, mol, m.position, name=label)
            if m.position not in wt_cache:
                wt_cache[m.position] = side_chain_field(wt_fields, wt_mol, m.position, name="wt")
            old = wt_cache[m.position]
            np.savez_compressed(out, label=label, position=m.position, wild=m.wild, new_residue=m.new,
                                **{"new_" + k: v for k, v in _pack(new).items()},
                                **{"old_" + k: v for k, v in _pack(old).items()})
            done += 1
        except Exception as exc:                     # noqa: BLE001
            failed.append((label, f"{type(exc).__name__}: {exc}"))
        if k % 200 == 0:
            log(f"  {domain}: {k}/{len(labels)} after {time.time() - t0:.0f} s, "
                f"{len(failed)} failed")
    return domain, done, failed


def _contexts(domain: str, positions: Sequence[int], built: str, dest: str, rhos=CONTEXT_RHOS):
    """The surroundings of every measured position of one domain, at every envelope decay."""
    from cmfqsar.mutants import Structure, protein_of, site_context
    fs = FieldSet.create(charge_model="file")
    wt = Structure.load(os.path.join(built, "wild_type.npz"))
    wt_mol = protein_of(wt)
    wt_fields = build_fields(wt_mol, fs, name="wild type")
    out: Dict[str, np.ndarray] = {}
    for rho in rhos:
        for p in positions:
            c = site_context(wt_fields, wt_mol, int(p), float(rho), name=f"{domain}:{p}")
            for k, v in _pack(c).items():
                out[f"{rho}|{p}|{k}"] = v
    np.savez_compressed(os.path.join(dest, "contexts.npz"), **out)
    return domain, len(positions) * len(rhos)


def read_table(domain: str) -> List[dict]:
    with open(os.path.join(EXAMPLES, domain, "mutants.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def prepare(domains=DOMAINS, shard: int = 0, shards: int = 1) -> None:
    """Stage 1: the side-chain fields of every mutant and the contexts of every position.

    With `shards` greater than one this process takes every `shards`-th mutant, starting at
    `shard`, so that several processes started side by side share the work; the contexts are
    written by the first share only.
    """
    for d in domains:
        built, dest = os.path.join(BUILT, d), os.path.join(PREPARED, d)
        if not os.path.isdir(built):
            log(f"{d}: no structures built yet")
            continue
        os.makedirs(dest, exist_ok=True)
        rows = read_table(d)
        have = {r["label"] for r in rows if os.path.exists(os.path.join(built, r["label"] + ".npz"))}
        labels = [r["label"] for r in rows if r["label"] in have][shard::shards]
        todo = [lab for lab in labels if not os.path.exists(os.path.join(dest, lab + ".npz"))]
        log(f"{d}: {len(rows)} in the table, {len(have)} built, {len(todo)} to prepare here")
        if todo:
            t0 = time.time()
            _dom, _done, failed = _prepare_domain(d, todo, built, dest)
            if failed:
                log(f"  {len(failed)} failed, first: {failed[0]}")
                with open(os.path.join(dest, f"failed_{shard}.txt"), "w") as fh:
                    for label, err in failed:
                        fh.write(f"{label}\t{err}\n")
            log(f"  prepared in {time.time() - t0:.0f} s")
        positions = sorted({int(r["position"]) for r in rows if r["label"] in have})
        cpath = os.path.join(dest, "contexts.npz")
        if shard == 0 and os.path.exists(cpath):
            # a run made while only part of the structures were built holds only part of the
            # positions, and has to be written again once the rest arrive
            with np.load(cpath, allow_pickle=False) as z:
                stored = {int(k.split("|")[1]) for k in z.files}
            if not set(positions) <= stored:
                log(f"  {len(set(positions) - stored)} positions are new: the contexts are "
                    f"written again")
                os.remove(cpath)
        if shard == 0 and not os.path.exists(cpath):
            t0 = time.time()
            _contexts(d, positions, built, dest)
            log(f"  {len(positions)} contexts at {len(CONTEXT_RHOS)} widths in "
                f"{time.time() - t0:.0f} s")


# --------------------------------------------------------------------------- #
# Stage 2: the samples
# --------------------------------------------------------------------------- #

class Samples:
    """Everything the measurement needs, in one row order."""

    def __init__(self, domains=DOMAINS, max_per_position: Optional[int] = MAX_PER_POSITION,
                 seed: int = 0):
        self.domain: List[str] = []
        self.label: List[str] = []
        self.position: List[int] = []
        self.mutation: List[MU.Mutation] = []
        self.y: List[float] = []
        self.new: List[MoleculeFields] = []
        self.old: List[MoleculeFields] = []
        self.contexts: Dict[Tuple[float, str, int], MoleculeFields] = {}
        rng = np.random.RandomState(int(seed))
        for d in domains:
            dest = os.path.join(PREPARED, d)
            if not os.path.isdir(dest):
                continue
            by_position: Dict[int, List[dict]] = {}
            for r in read_table(d):
                if os.path.exists(os.path.join(dest, r["label"] + ".npz")):
                    by_position.setdefault(int(r["position"]), []).append(r)
            for pos in sorted(by_position):
                rows = by_position[pos]
                if max_per_position is not None and len(rows) > max_per_position:
                    take = rng.choice(len(rows), int(max_per_position), replace=False)
                    rows = [rows[i] for i in sorted(take)]
                for r in rows:
                    with np.load(os.path.join(dest, r["label"] + ".npz"), allow_pickle=False) as z:
                        self.new.append(_unpack(z, "new_", r["label"]))
                        self.old.append(_unpack(z, "old_", "wt"))
                    self.domain.append(d)
                    self.label.append(r["label"])
                    self.position.append(pos)
                    self.mutation.append(MU.Mutation.parse(r["label"]))
                    self.y.append(float(r["ddG_ML"]))
            cpath = os.path.join(dest, "contexts.npz")
            if os.path.exists(cpath):
                with np.load(cpath, allow_pickle=False) as z:
                    for key in {k.rsplit("|", 1)[0] for k in z.files}:
                        rho, pos = key.split("|")
                        self.contexts[(float(rho), d, int(pos))] = _unpack(
                            z, key + "|", f"{d}:{pos}")
        self.y = np.asarray(self.y, float)

    def __len__(self) -> int:
        return len(self.label)

    def group(self, by: str) -> np.ndarray:
        """Sample -> group index, for folds that hold out whole positions, whole stretches of
        the chain, or whole domains.

        Holding out a position does not hold out its surroundings: the context of one residue
        and the context of the next share most of their atoms, so a model trained on the
        neighbours of a held-out site has already seen that site's surroundings.  The
        `segment` grouping holds out :data:`SEGMENT` consecutive residues at a time, which
        removes the sequence neighbours of every held-out site; residues that are far apart in
        the chain and close in space are still split, and only the `domain` grouping removes
        them too.
        """
        if by == "position":
            keys = [(d, p) for d, p in zip(self.domain, self.position)]
        elif by == "segment":
            keys = [(d, p // SEGMENT) for d, p in zip(self.domain, self.position)]
        elif by == "domain":
            keys = list(self.domain)
        else:
            raise ValueError(f"hold out 'position', 'segment' or 'domain', not {by!r}")
        order = {k: i for i, k in enumerate(sorted(set(keys)))}
        return np.array([order[k] for k in keys], int)

    def targets(self, by: str) -> AtomTargets:
        g = self.group(by)
        return AtomTargets(molecule=g, atom=np.arange(len(self)), y=np.asarray(self.y, float),
                           n_mol=int(g.max()) + 1, name=f"ddG, {by} held out")


def difference_fields(s: Samples) -> List[MoleculeFields]:
    from cmfqsar.localenv import difference_field
    return [difference_field(a, b, name=lab) for a, b, lab in zip(s.new, s.old, s.label)]


def context_fields(s: Samples, rho: float) -> List[MoleculeFields]:
    return [s.contexts[(rho, d, p)] for d, p in zip(s.domain, s.position)]


# --------------------------------------------------------------------------- #
# Stage 3: the kernels
# --------------------------------------------------------------------------- #

def field_kernel(fields: Sequence[MoleculeFields], alpha: float, fieldset: Optional[FieldSet] = None,
                 progress: bool = False) -> np.ndarray:
    """The cosine-normalised sum of the five field kernels, as a model with equal weights uses."""
    fs = fieldset or FieldSet.create(charge_model="file", alpha=alpha)
    fs.set_alpha(float(alpha))
    table = AtomTable.from_molecules(list(fields))
    t0 = time.time()
    K = kernel_matrices(table, table, fs.specs, symmetric=True)
    d = self_kernels(table, fs.specs)
    K = normalize_kernels(K, d, d)
    if progress:
        log(f"    kernel of {len(fields)} samples at alpha {alpha} in {time.time() - t0:.0f} s")
    return np.asarray(K.sum(axis=0) / K.shape[0], float)


def unique_context_kernel(s: Samples, rho: float, alpha: float) -> np.ndarray:
    """The context kernel, computed over the distinct positions and expanded to the samples."""
    keys = sorted({(d, p) for d, p in zip(s.domain, s.position)})
    index = {k: i for i, k in enumerate(keys)}
    K = field_kernel([s.contexts[(rho, d, p)] for d, p in keys], alpha)
    rows = np.array([index[(d, p)] for d, p in zip(s.domain, s.position)], int)
    return K[np.ix_(rows, rows)]


def count_rows(s: Samples) -> np.ndarray:
    """Atom-type counts in shells about the alpha carbon of the site, from the wild type, with
    the one-hot codes of the two residues.  The counts come from the context fields at the
    widest envelope, whose atoms are the surroundings of the site."""
    rho = min(CONTEXT_RHOS)
    types = sorted({t for key, c in s.contexts.items() if key[0] == rho for t in c.atom_types})
    index = {t: i for i, t in enumerate(types)}
    rows = []
    sub, _names = MU.substitution_features(s.mutation)
    for k, (d, p) in enumerate(zip(s.domain, s.position)):
        c = s.contexts[(rho, d, p)]
        r = np.zeros(len(types) * len(SHELLS))
        dist = np.linalg.norm(np.asarray(c.coords, float), axis=1)
        which = np.searchsorted(np.asarray(SHELLS, float), dist)
        for t, w in zip(c.atom_types, which):
            if w < len(SHELLS) and t in index:
                r[index[t] * len(SHELLS) + w] += 1.0
        rows.append(r)
    return np.c_[np.asarray(rows, float), sub[:, 3:]]


def burial_rows(s: "Samples") -> np.ndarray:
    """How buried the site is, as the simplest numbers that say it, with the substitution.

    The field of the surroundings is what carries the joined description, and the first thing
    it could be carrying is burial: a site with many atoms around it is in the core, where a
    substitution costs more.  A biochemist would measure that with one number, so the field has
    to be asked whether it says more than one number does.  The columns are the count of atoms
    within each envelope and the sum of the envelope factors, which is a soft count, together
    with the change of residue volume, hydropathy and charge.
    """
    sub, _names = MU.substitution_features(s.mutation)
    rows = []
    for d, p in zip(s.domain, s.position):
        row = []
        for rho in CONTEXT_RHOS:
            c = s.contexts[(rho, d, p)]
            row.append(float(c.n_atoms))
            row.append(float(np.abs(np.asarray(c.weights, float)).sum()))
        rows.append(row)
    return np.c_[np.asarray(rows, float), sub[:, :3]]


def mixed(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    return (1.0 - w) * a + w * b


# --------------------------------------------------------------------------- #
# Stage 4: the measurement
# --------------------------------------------------------------------------- #

def alignment_table(kernels: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
    import run_spectra_ir as SI
    return {k: SI.target_alignment(K, y) for k, K in kernels.items()}


def substitution_agreement(K: np.ndarray, s: "Samples") -> Dict[str, float]:
    """How much of a kernel is the substitution itself, and how much is where it sits.

    The same substitution made at two different positions gives two difference fields that
    differ only in the turn of the side chain and in what the force field did to it.  If the
    kernel of such a pair is close to one, the description carries little beyond the pair
    (wild-type residue, new residue), which the substitution reference already has, and a
    model built on it cannot do better than that reference.  This is the question item LP2
    asked of its own difference field, where the answer -- a mean kernel of 0.998 -- explained
    a failure that the accuracy alone did not.

    Reported as mean kernel values over the pairs of samples of four kinds.
    """
    n = len(s)
    wild = np.array([m.wild for m in s.mutation])
    new = np.array([m.new for m in s.mutation])
    site = np.array([f"{d}:{p}" for d, p in zip(s.domain, s.position)])
    iu = np.triu_indices(n, 1)
    same_sub = (wild[iu[0]] == wild[iu[1]]) & (new[iu[0]] == new[iu[1]]) & (site[iu[0]] != site[iu[1]])
    same_new = (new[iu[0]] == new[iu[1]]) & (wild[iu[0]] != wild[iu[1]])
    same_wild = (wild[iu[0]] == wild[iu[1]]) & (new[iu[0]] != new[iu[1]])
    v = np.asarray(K, float)[iu]
    out = {"same substitution, other position": float(v[same_sub].mean()) if same_sub.any() else float("nan"),
           "same new residue": float(v[same_new].mean()) if same_new.any() else float("nan"),
           "same wild-type residue": float(v[same_wild].mean()) if same_wild.any() else float("nan"),
           "every pair": float(v.mean()),
           "pairs with the same substitution": int(same_sub.sum())}
    return out


def nested(kernels: Dict[str, np.ndarray], tg: AtomTargets, seeds, name: str,
           folds: int = FOLDS):
    results, chosen = [], {}
    for sd in seeds:
        res, picks = B.nested_choice(kernels, tg, lams=LAMS, folds=folds, seed=sd, inner=INNER,
                                     name=name)
        results.append(res)
        for key, _lam in picks:
            chosen[key] = chosen.get(key, 0) + 1
    return results, chosen


def regression(X: np.ndarray, tg: AtomTargets, seeds, name: str, folds: int = FOLDS):
    return [B.regression_reference(X, tg, folds, sd, name) for sd in seeds], {}


def mean_reference(tg: AtomTargets, seeds, folds: int = FOLDS):
    out = []
    y = np.asarray(tg.y, float)
    for sd in seeds:
        pred = np.zeros(len(y))
        for te in tg.folds(folds, sd):
            tr = np.setdiff1d(np.arange(len(y)), te)
            pred[te] = y[tr].mean()
        out.append(B._as_result("the mean of the training part", pred, tg, folds))
    return out, {}


def row_of(results, chosen, tg) -> dict:
    y = np.asarray(tg.y, float)
    q2, sd, err = B.repeated_q2(results, y)
    return {"q2": round(q2, 4), "sd": round(sd, 4), "rmse": round(err, 4),
            "mae": round(float(np.mean([np.mean(np.abs(y - r.predicted)) for r in results])), 4),
            "chosen": chosen}


def compare(a, b, tg) -> List[float]:
    d, lo, hi = B.paired_repeats(a, b, tg)
    return [round(float(d), 4), round(float(lo), 4), round(float(hi), 4)]


COMPARISONS = (("difference", "three columns"), ("difference", "substitution"),
               ("difference", "counts"), ("difference", "mean"),
               ("difference+context", "difference"),
               ("difference+context", "three columns"),
               ("difference+context", "burial"), ("context", "burial"),
               ("burial", "three columns"),
               ("new", "difference"), ("context", "mean"))

KERNEL_ROWS = ("difference", "new", "context", "difference+context")


def run(domains=DOMAINS, repeats: int = REPEATS, max_per_position: Optional[int] = MAX_PER_POSITION,
        results: str = RESULTS, holdouts=("position", "segment", "domain"), seed: int = 0,
        out_name: str = "mutants") -> dict:
    os.makedirs(results, exist_ok=True)
    t0 = time.time()
    s = Samples(domains, max_per_position, seed)
    if not len(s):
        raise SystemExit("no prepared mutants: run with --prepare first")
    log(f"{len(s)} mutants in {len(set(s.domain))} domains, "
        f"{len(set(zip(s.domain, s.position)))} positions; "
        f"ddG from {s.y.min():.2f} to {s.y.max():.2f}, sd {s.y.std():.2f} kcal/mol")
    diffs = difference_fields(s)
    log(f"difference fields: {np.mean([d.n_atoms for d in diffs]):.1f} atoms each")

    kernels: Dict[str, Dict[str, np.ndarray]] = {"difference": {}, "new": {}, "context": {}}
    for a in ALPHAS:
        kernels["difference"][f"alpha {a}"] = field_kernel(diffs, a, progress=True)
        kernels["new"][f"alpha {a}"] = field_kernel(s.new, a, progress=True)
    for rho in CONTEXT_CHOICE:
        t = time.time()
        kernels["context"][f"rho {rho}"] = unique_context_kernel(s, rho, CONTEXT_ALPHA)
        log(f"    context kernel at rho {rho} in {time.time() - t:.0f} s")
    joined = {"alpha 1.0 alone": kernels["difference"]["alpha 1.0"]}
    for kc, Kc in kernels["context"].items():
        for w in MIXES:
            joined[f"alpha 1.0 + {kc} at {w}"] = mixed(kernels["difference"]["alpha 1.0"], Kc, w)
    kernels["difference+context"] = joined

    X_sub, _names = MU.substitution_features(s.mutation)
    X_three = X_sub[:, :3]
    X_counts = count_rows(s)
    X_burial = burial_rows(s)
    log(f"references: {X_three.shape[1]} and {X_sub.shape[1]} substitution columns, "
        f"{X_counts.shape[1]} count columns, {X_burial.shape[1]} burial columns")

    out = {"domains": list(domains), "n": len(s), "n_positions": len(set(zip(s.domain, s.position))),
           "repeats": repeats, "max_per_position": max_per_position, "seed": seed,
           "sd_y": round(float(s.y.std()), 4),
           "atoms_per_difference": round(float(np.mean([d.n_atoms for d in diffs])), 1),
           "per_domain": {d: int(sum(1 for x in s.domain if x == d)) for d in domains},
           "alignment": {k: alignment_table(v, s.y) for k, v in kernels.items() if k in KERNEL_ROWS[:3]},
           "agreement": {k: substitution_agreement(kernels[k]["alpha 1.0"], s)
                         for k in ("difference", "new")},
           "holdout": {}}
    log("kernel correlation with the target: " +
        ", ".join(f"{k} {max(v.values()):+.3f}" for k, v in out["alignment"].items()))
    log("the same substitution at another position gives a difference-field kernel of "
        f"{out['agreement']['difference']['same substitution, other position']:+.3f} "
        f"over {out['agreement']['difference']['pairs with the same substitution']} pairs")

    seeds = list(range(repeats))
    for by in holdouts:
        tg = s.targets(by)
        n_groups = int(tg.molecule.max()) + 1
        folds = min(FOLDS, n_groups) if by != "domain" else min(len(domains), 4)
        log(f"--- holding out whole {by}s ({n_groups} of them, {folds} folds)")
        rows: Dict[str, dict] = {}
        keep: Dict[str, list] = {}
        for name in KERNEL_ROWS:
            t = time.time()
            res, chosen = nested(kernels[name], tg, seeds, name, folds)
            rows[name] = row_of(res, chosen, tg)
            keep[name] = res
            log(f"  {name:22s} q2 {rows[name]['q2']:+.3f}  ({time.time() - t:.0f} s)")
        for name, X in (("three columns", X_three), ("substitution", X_sub),
                        ("counts", X_counts), ("burial", X_burial)):
            res, chosen = regression(X, tg, seeds, name, folds)
            rows[name] = row_of(res, chosen, tg)
            keep[name] = res
            log(f"  {name:22s} q2 {rows[name]['q2']:+.3f}")
        res, chosen = mean_reference(tg, seeds, folds)
        rows["mean"] = row_of(res, chosen, tg)
        keep["mean"] = res
        table = {f"{a} - {b}": compare(keep[a], keep[b], tg) for a, b in COMPARISONS}
        out["holdout"][by] = {"rows": rows, "differences": table, "n_groups": n_groups,
                              "folds": folds}
        with open(os.path.join(results, f"{out_name}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(results, f"{out_name}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    log(f"written to {os.path.join(results, out_name + '.json')} after {out['seconds']:.0f} s")
    return out


# --------------------------------------------------------------------------- #
# The report
# --------------------------------------------------------------------------- #

LABEL = {
    "difference": "difference field: the new side chain minus the old one",
    "new": "the new side chain alone",
    "context": "the surroundings of the site, without the replaced side chain",
    "difference+context": "difference field and surroundings, joined",
    "three columns": "linear regression on the change of residue volume, hydropathy and charge",
    "substitution": "linear regression on those three columns and the one-hot codes of the two "
                    "residues",
    "counts": "linear regression on atom counts in shells about the site, with the one-hot codes",
    "burial": "linear regression on how many atoms surround the site, with the three columns",
    "mean": "the mean of the training part",
}
ORDER = ("mean", "three columns", "substitution", "counts", "burial", "context", "new",
         "difference", "difference+context")


def _iv(d: Sequence[float]) -> str:
    return f"{d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}]"


def report(results: str = RESULTS) -> str:
    with open(os.path.join(results, "mutants.json")) as fh:
        d = json.load(fh)
    second = os.path.join(results, "mutants_seed1.json")
    other = json.load(open(second)) if os.path.exists(second) else None
    out = [f"# Point mutants as fields in the frame of the mutated residue (plan item PP1)", "",
           f"{d['n']} single substitutions at {d['n_positions']} positions of "
           f"{len(d['domains'])} domains ({', '.join(d['domains'])}), from the mega-scale data set "
           f"of Tsuboyama et al. (2023).  The target is ddG_ML in kcal/mol (the unfolding free "
           f"energy of the variant minus that of the wild type), whose standard deviation over "
           f"these samples is {d['sd_y']:.2f}.  A difference field has "
           f"{d['atoms_per_difference']:.0f} atoms on average.  Of the substitutions measured "
           f"at a position, {d['max_per_position']} enter the measurement, drawn with a fixed "
           f"seed: the cost of a kernel grows with the square of the sample count and the cost "
           f"of a fit with its cube, and a position is kept whole because a position is what "
           f"the folds hold out.", "",
           "## Before any fit", "",
           "The correlation of a kernel with the target says whether samples the description "
           "finds alike have alike targets (item EC5).", "",
           "| Kernel | Best correlation with ddG | Setting |", "|---|---|---|"]
    for name, table in d.get("alignment", {}).items():
        best = max(table, key=lambda k: table[k])
        out.append(f"| {LABEL[name]} | {table[best]:+.3f} | {best} |")
    out += ["", "How much of a difference field is the substitution itself: the mean kernel "
                "over pairs of samples of each kind, at an attenuation of 1.", "",
            "| Pairs | Difference field | The new side chain alone |", "|---|---|---|"]
    agree = d.get("agreement", {})
    if agree:
        for key in ("same substitution, other position", "same new residue",
                    "same wild-type residue", "every pair"):
            out.append(f"| {key} | {agree['difference'][key]:+.3f} | {agree['new'][key]:+.3f} |")
        out += ["", f"The first row covers "
                    f"{agree['difference']['pairs with the same substitution']} pairs.", ""]
    for by, part in d["holdout"].items():
        out += [f"## Holding out whole {by}s ({part['n_groups']} of them, {part['folds']} folds)", "",
                "| Description | q2 | sd over repeats | RMSE, kcal/mol | MAE, kcal/mol |",
                "|---|---|---|---|---|"]
        for name in ORDER:
            r = part["rows"].get(name)
            if r:
                out.append(f"| {LABEL[name]} | {r['q2']:+.3f} | {r['sd']:.3f} | {r['rmse']:.3f} | "
                           f"{r['mae']:.3f} |")
        out += ["", "Differences in q2, with the 95 % bootstrap interval over the held-out unit:", "",
                "| Comparison | Difference in q2 |", "|---|---|"]
        for key, val in part["differences"].items():
            out.append(f"| {key} | {_iv(val)} |")
        out.append("")
    if other:
        out += ["## The same measurement on another draw of the substitutions", "",
                "Four substitutions of each position enter the measurement, and which four is a "
                "choice. This is the same measurement with the other draw, as a check that a "
                "margin is not a property of the draw.", "",
                "| Description | " + " | ".join(f"{by}s held out" for by in d["holdout"]) + " |",
                "|---|" + "---|" * len(d["holdout"])]
        for name in ORDER:
            cells = []
            for by in d["holdout"]:
                a = d["holdout"][by]["rows"].get(name, {}).get("q2")
                b = other["holdout"].get(by, {}).get("rows", {}).get(name, {}).get("q2")
                cells.append("--" if a is None or b is None else f"{a:+.3f} / {b:+.3f}")
            out.append(f"| {LABEL[name]} | " + " | ".join(cells) + " |")
        out += ["", f"The second draw holds {other['n']} substitutions against "
                    f"{d['n']}. Each cell is the first draw and then the second.", ""]
    with open(os.path.join(results, "table.md"), "w") as fh:
        fh.write("\n".join(out))
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--prepare", action="store_true", help="stage 1: the fields of every mutant")
    p.add_argument("--shard", type=int, default=0, help="which share of the mutants to prepare")
    p.add_argument("--shards", type=int, default=1, help="how many processes share the work")
    p.add_argument("--domains", nargs="*", default=list(DOMAINS))
    p.add_argument("--max-per-position", type=int, default=MAX_PER_POSITION,
                   help="how many substitutions of each position enter the measurement")
    p.add_argument("--repeats", type=int, default=REPEATS)
    p.add_argument("--holdout", nargs="*", default=["position", "segment", "domain"])
    p.add_argument("--seed", type=int, default=0,
                   help="which substitutions of each position to draw")
    p.add_argument("--out-name", default="mutants", help="the name of the result file")
    p.add_argument("--report", action="store_true")
    a = p.parse_args(argv)
    if a.prepare:
        prepare(a.domains, a.shard, a.shards)
        return
    if a.report:
        print(report())
        return
    run(a.domains, a.repeats, a.max_per_position, holdouts=tuple(a.holdout), seed=a.seed,
        out_name=a.out_name)
    print(report())


if __name__ == "__main__":
    main()
