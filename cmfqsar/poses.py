"""
A protocol that can judge an alignment, because q2 cannot.

Every measurement in this program says the alignment decides more than the kernel, the learner or
the fields: an automatic frame costs about 0.30 in q2 where a curated one succeeds, and on series
whose frame is wrong the fields fall below atom-type counts.  Yet the program has no way to tell
whether an alignment method is any good, and the obvious measure is actively misleading.

**Why q2 cannot judge a pose.**  Realigning changes the geometry that q2 is computed on.  Moving a
molecule until the model is confident about it raises the model's agreement with itself, which is
what q2 measures; the README records q2 and external R2pred moving in opposite directions under
exactly this operation.  A criterion that rewards a method for making the data look like the model
cannot be used to choose between methods.

**What can.**  A crystal pose is an answer obtained without the model.  The FEP benchmark sets
under ``examples/openff``, ``examples/schindler`` and ``examples/ross`` carry one for every ligand,
co-crystallised in the same frame, so a posing method can be asked the only two questions that
matter:

1. **Did it find the pose?**  The RMSD of the recovered pose to the crystal pose, over heavy atoms,
   symmetry-aware, and **without superposition** -- ``rdMolAlign.CalcRMS``, never ``GetBestRMS``.
   The frame is the thing being judged, so re-superposing before measuring would report 0.0 for a
   molecule placed in the wrong site: on a molecule displaced by 1 A the two functions return 1.0
   and 0.0 respectively.
2. **Did the prediction improve?**  The external R2pred on held-out molecules posed **without their
   activity**, which is the operational case: a new compound arrives with neither a pose nor a
   measurement.  Its pose comes from the method under test, and its activity is then predicted by a
   model trained on crystal-posed molecules.

**Both, always, and the control first.**  Every run begins by destroying the poses
(:func:`cmfqsar.align.unalign`), so each method is asked to recover a frame rather than to leave
one alone, and every number is reported between two fixed references: the **ceiling**, the crystal
poses themselves, and the **floor**, the scrambled poses left as they are.  A method is worth
something only to the extent that it moves from the floor towards the ceiling, and
:attr:`PoseScore.recovery` is that fraction.  Without the control a method that silently does
nothing scores exactly what the starting poses scored, which on a curated set looks like success.

The protocol is agnostic about the method: :data:`POSERS` holds the ones in the program (MCS
superposition on a reference, field-similarity refinement, and the posterior-variance search of
the 2019 paper), and :func:`judge` takes any callable that re-poses molecules in place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from .align import field_align, mcs_align, set_coords, unalign
from .fields import AtomTable, FieldSet, MoleculeFields, build_fields
from .model import CMFModel, r2_score, rmse


def pose_rmsd(mol: Chem.Mol, ref: Chem.Mol, heavy_only: bool = True) -> float:
    """Symmetry-aware RMSD between two poses of one molecule, **without superposition**.

    This is the deviation in the frame the two structures are already in, which is what judging a
    pose requires.  ``GetBestRMS`` would superpose them first and report how similar the
    conformations are, a different and much easier question: a molecule translated by 1 A scores
    1.0 here and 0.0 there.

    Hydrogens are excluded by default, since a crystal structure rarely places them.
    """
    a, b = (Chem.RemoveHs(Chem.Mol(mol)), Chem.RemoveHs(Chem.Mol(ref))) if heavy_only \
        else (Chem.Mol(mol), Chem.Mol(ref))
    try:
        return float(rdMolAlign.CalcRMS(a, b))
    except Exception:                       # differing atom counts, a failed sanitisation
        pa = np.asarray(a.GetConformer().GetPositions(), float)
        pb = np.asarray(b.GetConformer().GetPositions(), float)
        if pa.shape != pb.shape:
            return float("nan")
        return float(np.sqrt(np.mean(np.sum((pa - pb) ** 2, axis=1))))


@dataclass
class PoseScore:
    """What one posing method achieved on one set: where the poses went, and what they predict."""
    method: str = ""
    n: int = 0
    rmsd: np.ndarray = field(default_factory=lambda: np.zeros(0))   # per molecule, to the crystal
    r2_pred: float = float("nan")           # external, against the training mean
    rmse_pred: float = float("nan")
    predicted: np.ndarray = field(default_factory=lambda: np.zeros(0))
    observed: np.ndarray = field(default_factory=lambda: np.zeros(0))
    seconds: float = 0.0
    floor_rmsd: float = float("nan")        # the median RMSD of the scrambled poses it started from
    #: with a regenerated conformation, the RMSD the best rigid placement would still leave
    conformer_limit: float = 0.0

    @property
    def median_rmsd(self) -> float:
        return float(np.nanmedian(self.rmsd)) if len(self.rmsd) else float("nan")

    def within(self, cutoff: float) -> float:
        """Fraction of molecules whose pose is within `cutoff` angstrom of the crystal pose."""
        if not len(self.rmsd):
            return float("nan")
        return float(np.mean(np.asarray(self.rmsd) <= cutoff))

    @property
    def recovery(self) -> float:
        """How much of the scrambling was undone: 1 is the crystal pose, 0 is no movement.

        Negative means the method left the molecules further from the crystal pose than the
        scrambled poses it was given, which happens and should be reported.
        """
        if not np.isfinite(self.floor_rmsd) or self.floor_rmsd <= 0:
            return float("nan")
        return float((self.floor_rmsd - self.median_rmsd) / self.floor_rmsd)

    def row(self) -> dict:
        return {"method": self.method, "n": self.n, "median_rmsd": self.median_rmsd,
                "conformer_limit": self.conformer_limit,
                "within_1A": self.within(1.0), "within_2A": self.within(2.0),
                "recovery": self.recovery, "r2_pred": self.r2_pred,
                "rmse_pred": self.rmse_pred, "seconds": self.seconds}


# ----------------------------------------------------------------------------------------------
# the posing methods, each of which may use anything except the activity of what it is posing
# ----------------------------------------------------------------------------------------------

def pose_keep(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Leave the poses alone: the floor, and the score any method that does nothing will get."""
    return None


def pose_mcs(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Superimpose each molecule on a reference training molecule through their common substructure."""
    ref = ctx["reference"]
    for m in mols:
        try:
            mcs_align(m, ref)
        except Exception:                   # noqa: BLE001  -- a molecule that shares nothing
            pass


def pose_field(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """MCS first, then refine on the CMF field similarity with the reference."""
    ref, fs = ctx["reference"], ctx["fieldset"]
    for m in mols:
        try:
            mcs_align(m, ref)
        except Exception:                   # noqa: BLE001
            pass
        try:
            field_align(m, ref, fs)
        except Exception:                   # noqa: BLE001
            pass


def pose_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Minimise the model's posterior variance: the proposal of the 2019 paper.

    The activity of the molecule being posed is not used -- only the model's confidence about it,
    which is what makes this legitimate for a molecule that has not been measured.
    """
    from .varalign import variance_align
    fs, model = ctx["fieldset"], ctx["model"]
    mf = [build_fields(m, fs, f"probe{i}") for i, m in enumerate(mols)]
    variance_align(mf, model, max_rmsd=ctx.get("max_rmsd"), restarts=ctx.get("restarts", 2),
                   seed=ctx.get("seed", 0))
    for m, f in zip(mols, mf):
        set_coords(m, f.coords)


def pose_mcs_then_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """The two in series: a substructure frame, then the confidence search from there."""
    pose_mcs(mols, ctx)
    pose_variance(mols, ctx)


def pose_flexible(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Confidence over rigid motion *and* the rotatable bonds (:mod:`cmfqsar.flexalign`)."""
    from .flexalign import flexible_align
    fs, model = ctx["fieldset"], ctx["model"]
    mf = [build_fields(m, fs, f"probe{i}") for i, m in enumerate(mols)]
    flexible_align(mf, list(mols), model, max_rmsd=ctx.get("max_rmsd"),
                   energy_slack=ctx.get("energy_slack", 10.0),
                   energy_weight=ctx.get("energy_weight", 1.0),
                   restarts=ctx.get("restarts", 2), seed=ctx.get("seed", 0),
                   max_torsions=ctx.get("max_torsions", 12),
                   conformers=ctx.get("conformers", 0),
                   prune_rmsd=ctx.get("prune_rmsd", 0.5))
    for m, f in zip(mols, mf):
        set_coords(m, f.coords)


def pose_mcs_then_flexible(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """A substructure frame, then the flexible confidence search from there."""
    pose_mcs(mols, ctx)
    pose_flexible(mols, ctx)


#: name -> poser.  Every one is handed the molecules to pose and a context holding the reference
#: molecule, the field set and the fitted model; none of them may look at the activity.
POSERS: Dict[str, Callable[[Sequence[Chem.Mol], dict], None]] = {
    "scrambled": pose_keep,
    "mcs": pose_mcs,
    "field": pose_field,
    "variance": pose_variance,
    "mcs+variance": pose_mcs_then_variance,
    "flex": pose_flexible,
    "mcs+flex": pose_mcs_then_flexible,
}


# ----------------------------------------------------------------------------------------------
# the protocol
# ----------------------------------------------------------------------------------------------

def _scramble(mols: Sequence[Chem.Mol], fieldset: FieldSet, seed: int,
              max_translation: float = 2.0) -> List[float]:
    """Give every molecule a random rigid pose; returns the displacements."""
    mf = [build_fields(m, fieldset, f"s{i}") for i, m in enumerate(mols)]
    moved = unalign(mf, max_translation=max_translation, seed=seed)
    for m, f in zip(mols, mf):
        set_coords(m, f.coords)
    return list(moved)


def regenerate_conformation(mol: Chem.Mol, seed: int = 0) -> Chem.Mol:
    """The same molecule with a conformation generated from scratch, crystal geometry discarded.

    Without this the protocol hands a held-out molecule its crystal *conformation* and only
    scrambles its pose, which is the one case a flexible method cannot help with: every torsion
    is already correct, so turning one can only do harm.  A molecule that arrives without a
    crystal structure arrives without its conformation too, and that is what a flexible
    alignment is for.
    """
    from rdkit.Chem import AllChem

    out = Chem.Mol(mol)
    if AllChem.EmbedMolecule(out, randomSeed=int(seed) + 1) != 0:
        return Chem.Mol(mol)
    try:
        AllChem.MMFFOptimizeMolecule(out)
        # setting MMFF up alters the aromaticity flags in place, after which the SMARTS that
        # FindMCS returns no longer matches the molecule it came from -- a substructure
        # alignment then fails silently and the molecule is left where it was, which made the
        # MCS baseline look like no alignment at all on five of sixteen sets
        Chem.SanitizeMol(out)
    except Exception:                       # noqa: BLE001
        pass
    # the generated conformation comes back at the origin, which in a crystal frame is tens of
    # angstrom away -- far enough that every field overlap is zero, the objective is flat and no
    # confidence search can move at all.  The site is known in any real workflow, so the molecule
    # is returned to the centroid it had and the control then randomises its orientation.
    new = np.asarray(out.GetConformer().GetPositions(), float)
    ref = np.asarray(mol.GetConformer().GetPositions(), float)
    set_coords(out, new - new.mean(axis=0) + ref.mean(axis=0))
    return out


def conformer_limit(mol: Chem.Mol, ref: Chem.Mol) -> float:
    """The RMSD the best possible rigid placement of this conformation would still leave.

    No rigid method can beat it, so it is the ceiling for a molecule whose conformation is wrong,
    and the number against which a flexible method has to be read.
    """
    try:
        return float(rdMolAlign.GetBestRMS(Chem.RemoveHs(Chem.Mol(mol)),
                                           Chem.RemoveHs(Chem.Mol(ref))))
    except Exception:                       # noqa: BLE001
        return float("nan")


def _predict(model: CMFModel, mols: Sequence[Chem.Mol], names: Sequence[str],
             fieldset: FieldSet) -> np.ndarray:
    table = AtomTable.from_molecules([build_fields(m, fieldset, n)
                                      for m, n in zip(mols, names)])
    return np.asarray(model.predict(table), float)


def judge(mols: Sequence[Chem.Mol], names: Sequence[str], y: Sequence[float],
          train: Sequence[int], test: Sequence[int], fieldset: FieldSet,
          methods: Sequence[str] = ("scrambled", "mcs", "field", "variance"),
          lam: float = 0.1, seed: int = 0, max_translation: float = 2.0,
          restarts: int = 2, max_rmsd: Optional[float] = None,
          posers: Optional[Dict[str, Callable]] = None,
          reconform: bool = False, conformers: int = 0,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, PoseScore]:
    """Judge posing methods against the crystal poses of a set.

    `mols` arrive in their crystal poses.  A model is fitted on the training molecules as they
    are; the test molecules then have their poses destroyed, and each method is asked to recover
    them from that starting point, without ever seeing a test activity.  Returns a score per
    method, plus ``"crystal"`` -- the ceiling, the test molecules in the poses they came in.
    """
    import time

    posers = dict(POSERS) if posers is None else dict(posers)
    y = np.asarray(y, float)
    train, test = [int(i) for i in train], [int(i) for i in test]
    crystal = {i: Chem.Mol(mols[i]) for i in test}

    tr_table = AtomTable.from_molecules([build_fields(mols[i], fieldset, names[i])
                                         for i in train])
    model = CMFModel(fieldset=fieldset, method="gp", normalize=True, params={"lam": float(lam)})
    model.fit(tr_table, y[train], cv_folds=0)
    train_mean = float(np.mean(y[train]))

    out: Dict[str, PoseScore] = {}
    pred = _predict(model, [crystal[i] for i in test], [names[i] for i in test], fieldset)
    out["crystal"] = PoseScore(method="crystal", n=len(test),
                               rmsd=np.zeros(len(test)),
                               r2_pred=float(r2_score(y[test], pred, train_mean)),
                               rmse_pred=float(rmse(y[test], pred)),
                               predicted=pred, observed=y[test], floor_rmsd=float("nan"))
    if progress:
        progress(f"crystal poses: R2pred {out['crystal'].r2_pred:+.3f}")

    for name in methods:
        poser = posers.get(name)
        if poser is None:
            continue
        t0 = time.time()
        probe = [Chem.Mol(crystal[i]) for i in test]            # always from the crystal pose ...
        if reconform:
            probe = [regenerate_conformation(p, seed=seed + j) for j, p in enumerate(probe)]
        _scramble(probe, fieldset, seed=seed, max_translation=max_translation)
        # the floor is measured with the very same yardstick as the result: unalign reports its
        # own displacement over all atoms and without symmetry, and mixing the two measures gave
        # a method that does nothing a recovery of +19% on a twelve-molecule series
        floor = float(np.median([pose_rmsd(p, crystal[i]) for p, i in zip(probe, test)]))
        ctx = {"reference": mols[train[int(np.argmax(y[train]))]], "fieldset": fieldset,
               "model": model, "restarts": restarts, "seed": seed, "max_rmsd": max_rmsd,
               "conformers": int(conformers)}
        poser(probe, ctx)                                       # ... and scrambled, every time
        rmsds = np.array([pose_rmsd(p, crystal[i]) for p, i in zip(probe, test)])
        pred = _predict(model, probe, [names[i] for i in test], fieldset)
        limit = float(np.median([conformer_limit(p, crystal[i])
                                 for p, i in zip(probe, test)])) if reconform else 0.0
        out[name] = PoseScore(method=name, n=len(test), rmsd=rmsds, conformer_limit=limit,
                              r2_pred=float(r2_score(y[test], pred, train_mean)),
                              rmse_pred=float(rmse(y[test], pred)), predicted=pred,
                              observed=y[test], seconds=time.time() - t0, floor_rmsd=floor)
        if progress:
            s = out[name]
            progress(f"{name}: median RMSD {s.median_rmsd:.2f} A (from {floor:.2f}), "
                     f"recovery {s.recovery:+.0%}, R2pred {s.r2_pred:+.3f}")
    return out


def table(scores: Dict[str, PoseScore], title: str = "") -> str:
    """The protocol's answer as a table: where the poses went and what they predict."""
    lines = []
    if title:
        lines += [f"### {title}", ""]
    limit = max((s.conformer_limit for s in scores.values() if s), default=0.0)
    head = "| method | n | median RMSD | within 1 A | within 2 A | recovery | R2pred | RMSEpred |"
    rule = "|---|---:|---:|---:|---:|---:|---:|---:|"
    if limit > 0:
        head = head.replace("| recovery |", "| recovery | conformer limit |")
        rule += "---:|"
    lines += [head, rule]
    order = ["crystal"] + [k for k in scores if k != "crystal"]
    for k in order:
        s = scores.get(k)
        if s is None:
            continue
        rec = "-" if not np.isfinite(s.recovery) else f"{s.recovery:+.0%}"
        lim = f" {s.conformer_limit:.2f} A |" if limit > 0 else ""
        lines.append(f"| {s.method} | {s.n} | {s.median_rmsd:.2f} A | {s.within(1.0):.0%} | "
                     f"{s.within(2.0):.0%} | {rec} |{lim} {s.r2_pred:+.3f} | {s.rmse_pred:.3f} |")
    return "\n".join(lines) + "\n"
