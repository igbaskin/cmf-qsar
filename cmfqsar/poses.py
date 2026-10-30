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
    # the protocol destroys the orientation, so the restarts must be allowed to draw any of
    # them; a series already in the site keeps the bounded turn
    anywhere = float(ctx.get("max_rotation_deg", 30.0)) >= 180.0
    variance_align(mf, model, max_rmsd=ctx.get("max_rmsd"), restarts=ctx.get("restarts", 2),
                   seed=ctx.get("seed", 0), any_orientation=anywhere)
    for m, f in zip(mols, mf):
        set_coords(m, f.coords)


def pose_mcs_then_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """The two in series: a substructure frame, then the confidence search from there."""
    pose_mcs(mols, ctx)
    pose_variance(mols, ctx)


def pose_pocket(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Place every molecule where the *pocket* prefers it: complementarity less clash.

    The receptor alone, with no model: this is the prior half of item C3's objective, and it is
    here on its own so that the joint pose can be read against it.
    """
    from .docking import dock_coords
    fs, pocket = ctx["fieldset"], ctx["pocket"]
    mf = [build_fields(m, fs, f"probe{i}") for i, m in enumerate(mols)]
    for i, (m, f) in enumerate(zip(mols, mf)):
        try:
            coords, _res = dock_coords(
                f, pocket, fs, restarts=ctx.get("restarts", 2), seed=ctx.get("seed", 0) + i,
                max_rotation_deg=ctx.get("max_rotation_deg", 30.0))
        except Exception:                   # noqa: BLE001 - one molecule, not the set
            continue
        set_coords(m, coords)


def pose_pocket_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """The pocket as the prior and the model's confidence as the likelihood, together (C3)."""
    from .pocketalign import pocket_align
    fs, model, pocket = ctx["fieldset"], ctx["model"], ctx["pocket"]
    mf = [build_fields(m, fs, f"probe{i}") for i, m in enumerate(mols)]
    # the protocol destroys the orientation, so the restarts must be allowed to draw any of
    # them; a series already in the site keeps the bounded turn
    anywhere = float(ctx.get("max_rotation_deg", 30.0)) >= 180.0
    pocket_align(mf, model, pocket, pocket_weight=ctx.get("pocket_weight", 1.0),
                 max_rmsd=ctx.get("max_rmsd"), restarts=ctx.get("restarts", 2),
                 seed=ctx.get("seed", 0), any_orientation=anywhere, fieldset=fs)
    for m, f in zip(mols, mf):
        set_coords(m, f.coords)


def pose_mcs_then_pocket_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """A substructure frame first, then the pocket and the confidence together."""
    pose_mcs(mols, ctx)
    pose_pocket_variance(mols, ctx)


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
                   prune_rmsd=ctx.get("prune_rmsd", 0.5),
                   any_orientation=float(ctx.get("max_rotation_deg", 30.0)) >= 180.0)
    for m, f in zip(mols, mf):
        set_coords(m, f.coords)


def pose_mcs_then_flexible(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """A substructure frame, then the flexible confidence search from there."""
    pose_mcs(mols, ctx)
    pose_flexible(mols, ctx)


def pose_fieldgraph(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Match each molecule's own field extrema to the reference's (:mod:`cmfqsar.fieldgraph`).

    The only poser here that needs neither a common substructure nor a fitted model: what is
    superimposed is what the two molecules present, so a molecule sharing no scaffold with the
    reference can still be placed, and a molecule sharing nothing at all is left where it was.
    """
    from .fieldgraph import align_dataset
    ref, fs = ctx["reference"], ctx["fieldset"]
    ref_fields = build_fields(ref, fs, "reference")
    mf = [build_fields(m, fs, f"probe{i}") for i, m in enumerate(mols)]
    matches = align_dataset(mf, ref_fields, fs,
                            tolerance=ctx.get("fg_tolerance", 1.0),
                            min_size=ctx.get("fg_min_size", 3),
                            level=ctx.get("fg_level", 0.3),
                            max_points=ctx.get("fg_max_points", 16))
    for m, f, res in zip(mols, mf, matches):
        if res is not None and res.R is not None:
            set_coords(m, np.asarray(f.coords, float) @ np.asarray(res.R, float).T + res.t)


def pose_fieldgraph_then_field(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Field points for the frame, then the kernel refinement that needs one to start from."""
    pose_fieldgraph(mols, ctx)
    ref, fs = ctx["reference"], ctx["fieldset"]
    for m in mols:
        try:
            field_align(m, ref, fs)
        except Exception:                   # noqa: BLE001
            pass


def pose_fieldgraph_then_variance(mols: Sequence[Chem.Mol], ctx: dict) -> None:
    """Field points for the frame, then the model's confidence from there."""
    pose_fieldgraph(mols, ctx)
    pose_variance(mols, ctx)


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
    "fieldgraph": pose_fieldgraph,
    "fieldgraph+field": pose_fieldgraph_then_field,
    "fieldgraph+variance": pose_fieldgraph_then_variance,
    # item C3: the receptor alone, and the receptor with the model's confidence
    "pocket": pose_pocket,
    "pocket+variance": pose_pocket_variance,
    "mcs+pocket+variance": pose_mcs_then_pocket_variance,
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

    Without this the protocol gives a held-out molecule its crystal *conformation* and only
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
          restarts: int = 20, max_rotation_deg: float = 180.0,
          max_rmsd: Optional[float] = None,
          posers: Optional[Dict[str, Callable]] = None,
          reconform: bool = False, conformers: int = 0,
          pocket: Optional[MoleculeFields] = None, pocket_weight: float = 1.0,
          progress: Optional[Callable[[str], None]] = None) -> Dict[str, PoseScore]:
    """Judge posing methods against the crystal poses of a set.

    `mols` arrive in their crystal poses.  A model is fitted on the training molecules as they
    are; the test molecules then have their poses destroyed, and each method is asked to recover
    them from that starting point, without ever seeing a test activity.  Returns a score per
    method, plus ``"crystal"`` -- the ceiling, the test molecules in the poses they came in.

    The destruction gives every test molecule a uniformly random orientation, so a method that
    searches orientations must be allowed to draw any of them: `max_rotation_deg` is 180 here,
    where the docking's own default is the 30 degrees a series already in the site needs.
    `restarts` is 20 for the same reason.  Both halves are needed: a random orientation is only
    useful if enough of them are drawn, and more draws of a bounded turn do not help at all.
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
        # the floor is measured with exactly the same measure as the result: unalign reports its
        # own displacement over all atoms and without symmetry, and mixing the two measures gave
        # a method that does nothing a recovery of +19% on a twelve-molecule series
        floor = float(np.median([pose_rmsd(p, crystal[i]) for p, i in zip(probe, test)]))
        ctx = {"reference": mols[train[int(np.argmax(y[train]))]], "fieldset": fieldset,
               "model": model, "restarts": restarts, "seed": seed, "max_rmsd": max_rmsd,
               "conformers": int(conformers), "pocket": pocket,
               "pocket_weight": float(pocket_weight),
               "max_rotation_deg": float(max_rotation_deg)}
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


def global_superpose(probe: Sequence[Chem.Mol], ref: Sequence[Chem.Mol]) -> float:
    """Apply the single rigid motion that best maps the whole `probe` set onto `ref`.

    A training frame is only defined up to a global rigid motion: the CMF kernel depends on where
    the molecules are *relative to each other*, so rotating the entire set changes no kernel entry
    and no prediction.  Judging a recovered frame therefore means removing that freedom first --
    one transform for all molecules, never one per molecule, which would measure conformations
    instead of the frame.  Returns the RMSD of the concatenated coordinates after the fit.
    """
    A = np.vstack([np.asarray(m.GetConformer().GetPositions(), float) for m in probe])
    B = np.vstack([np.asarray(m.GetConformer().GetPositions(), float) for m in ref])
    if A.shape != B.shape:
        return float("nan")
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _s, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    d = np.sign(np.linalg.det(U @ Vt))
    R = U @ np.diag([1.0, 1.0, d]) @ Vt
    for m in probe:
        X = np.asarray(m.GetConformer().GetPositions(), float)
        set_coords(m, (X - ca) @ R + cb)
    A2 = np.vstack([np.asarray(m.GetConformer().GetPositions(), float) for m in probe])
    return float(np.sqrt(np.mean(np.sum((A2 - B) ** 2, axis=1))))


def frame_rmsds(probe: Sequence[Chem.Mol], ref: Sequence[Chem.Mol]) -> np.ndarray:
    """Per-molecule RMSD to the reference frame, after one global superposition of the set."""
    global_superpose(probe, ref)
    return np.array([pose_rmsd(p, r) for p, r in zip(probe, ref)])


def judge_training_frame(mols: Sequence[Chem.Mol], names: Sequence[str], y: Sequence[float],
                         train: Sequence[int], test: Sequence[int], fieldset: FieldSet,
                         methods: Sequence[str] = ("scrambled", "mcs", "coalign",
                                                   "mcs+coalign"),
                         lam: float = 0.1, seed: int = 0, max_translation: float = 2.0,
                         coalign_kwargs: Optional[dict] = None,
                         progress: Optional[Callable[[str], None]] = None
                         ) -> Dict[str, PoseScore]:
    """Judge a method that re-aligns the **training set** itself (item C4).

    The training poses are destroyed and recovered; the held-out molecules keep their crystal
    poses throughout, so what is measured is the frame and nothing else.  Two numbers per method:
    the per-molecule RMSD of the recovered frame to the crystal one, after the global
    superposition that the kernel cannot see, and the external R2pred of the held-out molecules
    posed onto the recovered frame by substructure alignment -- the operational route, since in a
    real application there is no crystal frame to align them to.

    Two questions, and they are not the same.  ``coalign`` starts from the scrambled frame and
    asks whether the iteration can *build* one; it cannot, and the reason is structural -- a model
    fitted on scrambled poses is meaningless, so the confidence it offers is flat, which is what
    C1 found for the rigid search and what the field-graph module was written for.
    ``mcs+coalign`` starts from the substructure frame and asks whether the iteration can
    *improve* a usable one, which is the question C4 is actually about.
    """
    import time

    from .coalign import coalign

    y = np.asarray(y, float)
    train, test = [int(i) for i in train], [int(i) for i in test]
    crystal_train = [Chem.Mol(mols[i]) for i in train]
    crystal_test = [Chem.Mol(mols[i]) for i in test]
    train_mean = float(np.mean(y[train]))
    out: Dict[str, PoseScore] = {}

    def score(probe_train, label, seconds=0.0, floor=float("nan")):
        rmsds = frame_rmsds(probe_train, crystal_train)
        model = CMFModel(fieldset=fieldset, method="gp", normalize=True,
                         params={"lam": float(lam)})
        model.fit(AtomTable.from_molecules(
            [build_fields(m, fieldset, names[i]) for m, i in zip(probe_train, train)]),
            y[train], cv_folds=0)
        ref = probe_train[int(np.argmax(y[train]))]
        posed = []
        for m in crystal_test:
            p = Chem.Mol(m)
            try:
                mcs_align(p, ref)
            except Exception:                   # noqa: BLE001
                pass
            posed.append(p)
        pred = _predict(model, posed, [names[i] for i in test], fieldset)
        return PoseScore(method=label, n=len(train), rmsd=rmsds,
                         r2_pred=float(r2_score(y[test], pred, train_mean)),
                         rmse_pred=float(rmse(y[test], pred)), predicted=pred,
                         observed=y[test], seconds=seconds, floor_rmsd=floor)

    out["crystal"] = score([Chem.Mol(m) for m in crystal_train], "crystal")
    if progress:
        progress(f"crystal training frame: R2pred {out['crystal'].r2_pred:+.3f}")

    for name in methods:
        t0 = time.time()
        probe = [Chem.Mol(m) for m in crystal_train]
        _scramble(probe, fieldset, seed=seed, max_translation=max_translation)
        floor = float(np.median(frame_rmsds([Chem.Mol(p) for p in probe], crystal_train)))
        if name in ("mcs", "mcs+coalign"):
            ref = probe[int(np.argmax(y[train]))]
            for p in probe:
                try:
                    mcs_align(p, ref)
                except Exception:               # noqa: BLE001
                    pass
        if name in ("coalign", "mcs+coalign"):
            mf = [build_fields(p, fieldset, names[i]) for p, i in zip(probe, train)]
            template = CMFModel(fieldset=fieldset, method="gp", normalize=True,
                                params={"lam": float(lam)})
            res = coalign(mf, y[train], template, **(coalign_kwargs or {}))
            for p, f in zip(probe, mf):
                set_coords(p, f.coords)
            if progress:
                progress(f"coalign: {res.stopped}, objective {res.improved:+.3f}")
        elif name not in ("scrambled", "mcs"):
            raise ValueError(f"unknown training-frame method {name!r}")
        s = score(probe, name, seconds=time.time() - t0, floor=floor)
        out[name] = s
        if progress:
            progress(f"{name}: frame RMSD {s.median_rmsd:.2f} A (from {floor:.2f}), "
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
