"""Screening a library without holding it in memory (plan item B3).

Predicting with a kernel method needs only the rows ``K(X, X_train)`` of the kernel: one row per
new molecule, as wide as the training set.  Nothing about a library of 100 000 molecules has to
be in memory at once, and nothing about it has to be kept after its score has been written.  The
rest of this program loads a data set, builds every molecule's fields and keeps them, which is
right for a training set of 200 molecules and impossible for a library.

This module reads the library in blocks and never holds more than one:

1. read `chunk` molecules from the file (a forward supplier, so the file is not indexed);
2. build their fields with the model's own field set, and pose them if asked;
3. compute the (chunk, training) kernel rows and score them;
4. write those rows out, keep the best `top` of them, and discard everything else.

The peak memory is therefore one block of fields plus one ``chunk x n_train`` kernel, whatever the
length of the file.  With the default block of 500 molecules against a 200-molecule training set
that is a 0.8 MB kernel.

What the score is
-----------------
For a regression model it is the prediction, or -- with `beta` above zero -- the lower confidence
bound ``mean - beta * sd``, which is what :meth:`cmfqsar.model.CMFModel.rank_score` computes and
what a library should normally be ranked by: a library is mostly molecules unlike the training
set, a kernel method answers the training mean for those, and a point estimate cannot tell that
answer from a real one.  For a one-class or classification model it is the decision value.

What this does not do
---------------------
It does not generate poses.  A field kernel compares molecules in one frame, so the molecules in
the file must already be in the training set's frame, or `pose` must be set, which places each
one by maximising its field similarity to the training molecules
(:func:`cmfqsar.decoys.align_decoys`) -- the same placement the background sets of the benchmarks
are given, and an optimistic one, since a real library has to choose a pose without knowing the
actives.  The measured cost is about half a second per molecule, which is the dominant cost of a
screening run and the reason posing is off by default.
"""
from __future__ import annotations

import heapq
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, build_fields

#: molecules per block: large enough that the kernel call is not dominated by Python, small
#: enough that the block's kernel stays a few megabytes
DEFAULT_CHUNK = 500


@dataclass
class ScreenSpec:
    """How a library is read and scored."""
    chunk: int = DEFAULT_CHUNK
    top: int = 100                      # how many of the best molecules to keep in memory
    beta: float = 0.0                   # confidence bound: score = mean - beta * sd
    larger_is_better: bool = True
    pose: bool = False                  # place each molecule in the training frame first
    restarts: int = 4                   # posing restarts per molecule
    conformers: int = 0                 # posing: generate this many conformations and keep the best
    cutoff: Optional[object] = None     # passed to the model for the duration of the run (item B2)
    seed: int = 0


@dataclass
class ScreenReport:
    """What a screening run did, and the best of what it found."""
    n_read: int = 0
    n_scored: int = 0
    n_failed: int = 0
    blocks: int = 0
    seconds: float = 0.0
    kernel_seconds: float = 0.0
    pose_seconds: float = 0.0
    peak_kernel_mb: float = 0.0
    out_path: Optional[str] = None
    best: List[Tuple[float, str, float, float]] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)

    @property
    def per_second(self) -> float:
        return self.n_scored / self.seconds if self.seconds > 0 else float("nan")

    def describe(self) -> str:
        lines = [f"{self.n_scored} molecules scored in {self.seconds:.1f} s "
                 f"({self.per_second:.0f} per second), {self.blocks} blocks, "
                 f"peak kernel block {self.peak_kernel_mb:.1f} MB"]
        if self.n_failed:
            lines.append(f"{self.n_failed} molecules could not be read or typed")
        if self.pose_seconds > 0:
            lines.append(f"posing took {self.pose_seconds:.1f} s of that "
                         f"({100 * self.pose_seconds / max(self.seconds, 1e-9):.0f} %)")
        if self.best:
            lines.append("the best of them:")
            for score, name, pred, sd in self.best[:10]:
                lines.append(f"  {name:32s} score {score:+8.3f}   prediction {pred:+8.3f}"
                             + (f"   sd {sd:.3f}" if sd == sd else ""))
        return "\n".join(lines)


def stream_molecules(path: str, chunk: int = DEFAULT_CHUNK,
                     add_hs: bool = True) -> Iterator[List[Tuple[object, str]]]:
    """Blocks of (molecule, name) from an SD file or a SMILES file, without indexing it.

    ``Chem.ForwardSDMolSupplier`` reads the file as a stream, so a 100 000-molecule SD file costs
    one block of memory rather than all of it.  A SMILES file is read line by line and embedded,
    which is slower per molecule and is there for libraries that arrive as SMILES.
    """
    from rdkit import Chem

    ext = os.path.splitext(path)[1].lower()
    block: List[Tuple[object, str]] = []
    if ext in (".sdf", ".sd", ".mol"):
        with open(path, "rb") as fh:
            supplier = Chem.ForwardSDMolSupplier(fh, removeHs=False, sanitize=True)
            for k, mol in enumerate(supplier):
                if mol is None:
                    block.append((None, f"molecule_{k + 1}"))
                else:
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                    block.append((mol, name or f"molecule_{k + 1}"))
                if len(block) >= chunk:
                    yield block
                    block = []
    else:
        from .align import embed_3d
        with open(path, encoding="utf-8", errors="replace") as fh:
            for k, line in enumerate(fh):
                parts = line.split()
                if not parts or parts[0].lower() in ("smiles", "smi"):
                    continue
                mol = Chem.MolFromSmiles(parts[0])
                name = parts[1] if len(parts) > 1 else f"molecule_{k + 1}"
                if mol is None:
                    block.append((None, name))
                else:
                    if add_hs:
                        mol = Chem.AddHs(mol)
                    try:
                        mol = embed_3d(mol, seed=k)
                    except Exception:                       # noqa: BLE001 - a molecule RDKit cannot embed
                        block.append((None, name))
                        continue
                    block.append((mol, name))
                if len(block) >= chunk:
                    yield block
                    block = []
    if block:
        yield block


def _score_block(model, table: AtomTable, spec: ScreenSpec):
    """(score, prediction, standard deviation) for one block of molecules.

    The spread is reported whenever there is a posterior to take it from, not only when `beta`
    asks for a confidence bound: it costs one solve against a factorisation the model keeps, and
    it is what tells the reader that a prediction is the training mean rather than a statement
    about the molecule.
    """
    if model.task == "regression" and model.is_linear:
        return model.rank_score(table, beta=spec.beta,
                                larger_is_better=spec.larger_is_better)
    pred = np.asarray(model.predict(table), float)
    return pred, pred, np.full(len(pred), np.nan)


def screen(model, path: str, spec: Optional[ScreenSpec] = None, out: Optional[str] = None,
           progress: Optional[Callable[[int, float], None]] = None,
           cancel: Optional[Callable[[], bool]] = None) -> ScreenReport:
    """Score every molecule of `path` against `model`, one block at a time.

    `out` is a CSV written as the run goes, so that a run interrupted after an hour has the scores
    of everything it had read.  The report keeps the best ``spec.top`` molecules and nothing else.
    """
    spec = spec or ScreenSpec()
    rep = ScreenReport(out_path=out)
    t0 = time.time()
    best: List[Tuple[float, str, float, float]] = []       # a heap, smallest score first
    fh = open(out, "w", encoding="utf-8", newline="") if out else None
    old_cutoff = getattr(model, "cutoff", None)
    if spec.cutoff is not None:
        model.cutoff = spec.cutoff
    try:
        if fh is not None:
            fh.write("name,score,prediction,sd\n")
        for block in stream_molecules(path, spec.chunk):
            if cancel is not None and cancel():
                break
            rep.blocks += 1
            rep.n_read += len(block)
            # a library file holds whatever a library holds: entries RDKit returns as None, and
            # entries it parses into a molecule with no atoms at all, which has no field to
            # compare and would reach the kernel as an empty segment
            mols = [(m, n) for m, n in block if m is not None and m.GetNumAtoms() > 0]
            rep.n_failed += len(block) - len(mols)
            for m, n in block:
                if (m is None or m.GetNumAtoms() == 0) and len(rep.failures) < 20:
                    rep.failures.append(n if m is None else f"{n}: no atoms")
            if not mols:
                continue
            fields, names, kept = [], [], []
            for m, n in mols:
                try:
                    fields.append(build_fields(m, model.fieldset, n))
                    names.append(n)
                    kept.append(m)
                except Exception as exc:                    # noqa: BLE001 - a molecule it cannot type
                    rep.n_failed += 1
                    if len(rep.failures) < 20:
                        rep.failures.append(f"{n}: {exc}")
            if not fields:
                continue
            if spec.pose:
                tp = time.time()
                fields = _pose(model, fields, kept, spec)
                rep.pose_seconds += time.time() - tp
            table = AtomTable.from_molecules(fields)
            tk = time.time()
            score, pred, sd = _score_block(model, table, spec)
            rep.kernel_seconds += time.time() - tk
            rep.peak_kernel_mb = max(rep.peak_kernel_mb,
                                     8.0 * len(fields) * model.train.n_mol / 1e6)
            rep.n_scored += len(names)
            for k, nm in enumerate(names):
                row = (float(score[k]), nm, float(pred[k]), float(sd[k]))
                if fh is not None:
                    # nine significant digits, because six lost the fifth decimal of a score
                    # around ten and a file that is re-read has to reproduce the ranking
                    fh.write(f"{nm},{row[0]:.9g},{row[2]:.9g},"
                             f"{'' if row[3] != row[3] else format(row[3], '.9g')}\n")
                if spec.top > 0:
                    if len(best) < spec.top:
                        heapq.heappush(best, row)
                    elif row[0] > best[0][0]:
                        heapq.heapreplace(best, row)
            if progress is not None:
                progress(rep.n_scored, time.time() - t0)
    finally:
        if fh is not None:
            fh.close()
        model.cutoff = old_cutoff
    rep.seconds = time.time() - t0
    rep.best = sorted(best, key=lambda r: -r[0])
    return rep


def _pose(model, fields, mols, spec: ScreenSpec):
    """Place one block of molecules in the training set's frame; returns the posed fields.

    The rigid aligner moves the fields it is given, so the same list comes back.  The flexible
    one generates conformations and returns the fields of whichever conformation posed best, so
    there the returned list is a different one.
    """
    from .decoys import align_decoys, align_decoys_flexible

    if spec.conformers and spec.conformers > 1:
        posed, _res = align_decoys_flexible(
            mols, [f.name for f in fields], model.train, model.fieldset,
            n_conformers=int(spec.conformers), restarts=int(spec.restarts), seed=int(spec.seed))
        return posed
    align_decoys(fields, model.train, model.fieldset, restarts=int(spec.restarts),
                 seed=int(spec.seed))
    return fields
