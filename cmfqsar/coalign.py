"""
Co-alignment of a training set, treated as what it is: expectation-maximisation over poses.

:func:`cmfqsar.varalign.cv_align_training_set` already aligns a training set by the
posterior-variance criterion, fold by fold so that no molecule is ever aligned against a model
containing it.  Its own docstring records what measuring it found: the iteration converges, the
contraction of the ensemble is mild, and the predictive gain is *not* reliable -- one set improved,
one degraded, two were neutral, and rounds after the first consistently made the external
prediction worse.  So it is run for one round with a displacement guard, and the outcome is left to
chance.

That is not a method, it is a hope with a default setting.  What the loop actually is, is an EM
iteration: the poses are latent variables, the M-step fits the model to the current poses, the
E-step moves each molecule to where the model (fitted without it) is most confident.  Written that
way it needs the three things an EM iteration needs, and this module adds them.

**An objective, and an acceptance test.**  A round is kept only if it improves the quantity the
iteration is allowed to improve: the **leave-fold-out error of the activities**.  This is a valid
objective here, and the reason is worth stating because the obvious alternative is not.  q2 of the
realigned set is invalid -- realigning changes the geometry q2 is computed on, and in the earlier
tests q2 and the external R2 moved in opposite directions.  But the *pose* of each molecule is
chosen from the model's confidence alone and never from its activity, so predicting that
molecule's activity afterwards, with the fold model that did not contain it, is a fair
measurement.  A round that does not improve it is rejected and the previous poses are restored.

**An annealed ridge.**  Early rounds run with a large lambda, where the model is smooth and the
confidence landscape has few spurious minima, and later rounds sharpen it.  Annealing is what
keeps an EM iteration from settling on a bad frame in the first round.

**A contraction stop.**  The degeneracy of the criterion is that superimposing everything makes
every variance zero, so the iteration is watched for it: the mean off-diagonal normalised kernel
(:func:`cmfqsar.varalign.ensemble_similarity`) is reported every round and the run stops if it
rises past a tolerance, whatever the objective says.

The E-step may be rigid or flexible.  The flexible one is the measured better choice -- C2 found
that confidence with the torsions and conformer-ensemble starts recovers held-out poses to 0.86 A
median against 2.27 A for the rigid search -- and it needs the RDKit molecules, which the field
representation does not carry, so `rdmols` must be passed for it.

Judged under C1 only (``benchmarks/run_poses.py --coalign``): the training poses are destroyed,
recovered by this iteration, and compared with the crystal poses they should have found.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np

from .fields import AtomTable, MoleculeFields
from .varalign import VarianceAligner, ensemble_similarity


@dataclass
class CoalignRound:
    """What one round of the iteration did, and whether it was kept."""
    round: int = 0
    lam: float = float("nan")
    moved_mean: float = 0.0
    moved_max: float = 0.0
    drift: float = 0.0              # cumulative displacement from the poses it started with
    similarity: float = float("nan")
    loo_rmse: float = float("nan")  # leave-fold-out error of the activities, the objective
    accepted: bool = False
    note: str = ""

    def describe(self) -> str:
        verdict = "kept" if self.accepted else f"rejected ({self.note})" if self.note else "rejected"
        return (f"round {self.round} (lambda {self.lam:g}): moved {self.moved_mean:.2f} A "
                f"(max {self.moved_max:.2f}), drift {self.drift:.2f} A, similarity "
                f"{self.similarity:.4f}, leave-fold-out RMSE {self.loo_rmse:.3f} -- {verdict}")


@dataclass
class CoalignResult:
    """The outcome: the poses kept, the objective at every round, and why it stopped."""
    rounds: List[CoalignRound] = field(default_factory=list)
    loo_rmse_start: float = float("nan")
    similarity_start: float = float("nan")
    best_round: int = 0
    stopped: str = ""

    @property
    def loo_rmse(self) -> float:
        kept = [r for r in self.rounds if r.accepted]
        return kept[-1].loo_rmse if kept else self.loo_rmse_start

    @property
    def improved(self) -> float:
        """How much the objective fell; negative means the iteration found nothing."""
        return float(self.loo_rmse_start - self.loo_rmse)

    def describe(self) -> str:
        lines = [f"start: leave-fold-out RMSE {self.loo_rmse_start:.3f}, "
                 f"similarity {self.similarity_start:.4f}"]
        lines += ["  " + r.describe() for r in self.rounds]
        lines.append(f"kept round {self.best_round} ({self.stopped}); "
                     f"objective {self.improved:+.3f}")
        return "\n".join(lines)


def _folds(n: int, folds: int, rng) -> List[np.ndarray]:
    order = rng.permutation(n)
    k = max(2, min(int(folds), n))
    return [order[j::k] for j in range(k)]


def _fit(model_template, mols: Sequence[MoleculeFields], y: np.ndarray, lam: float):
    from .model import CMFModel

    params = dict(model_template.params)
    params["lam"] = float(lam)
    sub = CMFModel(fieldset=model_template.fieldset.copy(), method=model_template.method,
                   params=params, normalize=model_template.normalize,
                   scale_fields=model_template.scale_fields)
    sub.fit(AtomTable.from_molecules(list(mols)), np.asarray(y, float), cv_folds=0)
    return sub


def leave_fold_out_error(mols: Sequence[MoleculeFields], y: np.ndarray, model_template,
                         lam: float, folds: int = 5, seed: int = 0) -> float:
    """RMSE of the activities predicted fold by fold, with the poses as they currently are.

    The objective the iteration is allowed to improve: every molecule is predicted by a model
    that does not contain it, and no pose was chosen using an activity.
    """
    mols = list(mols)
    y = np.asarray(y, float)
    n = len(mols)
    pred = np.full(n, np.nan)
    for part in _folds(n, folds, np.random.RandomState(seed)):
        rest = [i for i in range(n) if i not in set(part.tolist())]
        if len(rest) < 3 or not len(part):
            continue
        sub = _fit(model_template, [mols[i] for i in rest], y[rest], lam)
        table = AtomTable.from_molecules([mols[i] for i in part])
        pred[part] = np.asarray(sub.predict(table), float)
    ok = np.isfinite(pred)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((y[ok] - pred[ok]) ** 2)))


def coalign(mols: Sequence[MoleculeFields], y: Sequence[float], model_template,
            rounds: int = 4, folds: int = 5, max_rmsd: Optional[float] = 1.0,
            lam_start: Optional[float] = None, lam_factor: float = 0.5,
            flexible: bool = False, rdmols: Optional[Sequence] = None,
            conformers: int = 0, restarts: int = 0, max_similarity: Optional[float] = None,
            energy_slack: float = 10.0, tol: float = 0.01, seed: int = 0,
            progress: Optional[Callable[[str, float], None]] = None,
            cancel: Optional[Callable[[], bool]] = None) -> CoalignResult:
    """Co-align a training set by confidence, as an EM iteration with an acceptance test.

    The molecules are modified in place and left in the poses of the last **accepted** round, so
    a run that finds nothing leaves them exactly as they arrived.  `lam_start` defaults to four
    times the model's own ridge and is multiplied by `lam_factor` each round.

    `tol` is the *relative* improvement a round must show to be kept, and it is not a detail.
    With a threshold of machine epsilon the first measurement accepted a round on eg5 whose
    objective fell by less than one part in a hundred thousand -- leave-fold-out RMSE 0.479 to
    0.479 -- while the molecules moved 0.40 A and the frame drifted from 0.07 to 0.25 A from the
    crystal one.  An accepted round with no real gain is worse than no round at all, because the
    poses move.
    """
    mols = list(mols)
    y = np.asarray(y, float)
    n = len(mols)
    if n < 6:
        raise ValueError("a training set of at least six molecules is needed")
    if flexible and (rdmols is None or len(rdmols) != n):
        raise ValueError("the flexible E-step needs one RDKit molecule per field representation")
    specs = model_template.fieldset.specs
    lam0 = float(model_template.params.get("lam", 1.0))
    lam = float(lam_start) if lam_start is not None else 4.0 * lam0

    start_coords = [np.array(m.coords, float) for m in mols]
    kept_coords = [np.array(x, float) for x in start_coords]
    result = CoalignResult(
        loo_rmse_start=leave_fold_out_error(mols, y, model_template, lam0, folds, seed),
        similarity_start=ensemble_similarity(mols, specs))
    best = result.loo_rmse_start
    rng = np.random.RandomState(seed)

    for rnd in range(1, int(rounds) + 1):
        if cancel is not None and cancel():
            result.stopped = "cancelled"
            break
        moved: List[float] = []
        for k, part in enumerate(_folds(n, folds, rng)):
            if cancel is not None and cancel():
                break
            rest = [i for i in range(n) if i not in set(part.tolist())]
            if len(rest) < 3 or not len(part):
                continue
            if progress is not None:
                progress(f"round {rnd}: fold {k + 1}, fitting without {len(part)} molecules",
                         (rnd - 1 + k / max(folds, 1)) / max(rounds, 1))
            sub = _fit(model_template, [mols[i] for i in rest], y[rest], lam)
            if flexible:
                from .flexalign import FlexibleAligner
                aligner = FlexibleAligner(sub, max_rmsd=max_rmsd, energy_slack=energy_slack)
            else:
                aligner = VarianceAligner(sub, max_rmsd=max_rmsd)
            for i in part:
                before = np.array(mols[i].coords, float)
                if flexible:
                    aligner.align_flexible(mols[i], rdmols[int(i)], restarts=restarts,
                                           seed=seed + int(i), conformers=conformers)
                else:
                    aligner.align(mols[i], restarts=restarts, seed=seed + int(i))
                moved.append(float(np.sqrt(((mols[i].coords - before) ** 2).sum(1).mean())))

        sim = ensemble_similarity(mols, specs)
        err = leave_fold_out_error(mols, y, model_template, lam0, folds, seed)
        drift = float(np.mean([np.sqrt(((m.coords - x0) ** 2).sum(1).mean())
                               for m, x0 in zip(mols, start_coords)]))
        rec = CoalignRound(round=rnd, lam=lam, drift=drift, similarity=sim, loo_rmse=err,
                           moved_mean=float(np.mean(moved)) if moved else 0.0,
                           moved_max=float(np.max(moved)) if moved else 0.0)
        contracted = max_similarity is not None and sim > float(max_similarity)
        if contracted:
            rec.note = f"ensemble contracted to {sim:.4f}"
        elif np.isfinite(err) and err < best * (1.0 - float(tol)):
            rec.accepted = True
        else:
            rec.note = (f"the objective did not improve by {tol:.0%}"
                        if np.isfinite(err) else "the objective could not be evaluated")
        result.rounds.append(rec)
        if progress is not None:
            progress(rec.describe(), rnd / max(rounds, 1))
        if rec.accepted:
            best = err
            kept_coords = [np.array(m.coords, float) for m in mols]
            result.best_round = rnd
        else:
            for m, x in zip(mols, kept_coords):      # restore the last accepted poses
                m.coords = np.array(x, float)
            result.stopped = rec.note or "no improvement"
            break
        lam = max(lam * float(lam_factor), lam0)
    else:
        result.stopped = result.stopped or "rounds exhausted"
    if not result.stopped:
        result.stopped = "rounds exhausted"
    for m, x in zip(mols, kept_coords):
        m.coords = np.array(x, float)
    return result
