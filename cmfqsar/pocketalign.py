"""Aligning a molecule inside the pocket (plan item C3).

Two criteria for a pose exist in this program and neither is enough on its own.

:mod:`cmfqsar.varalign` places a molecule where the *model* is most confident about it -- where
its fields most resemble the training set's, as the 2019 paper proposes.  It knows nothing about
the receptor, so nothing stops it putting a molecule where no molecule could go, and on a series
whose training poses are themselves only a superposition it can only reproduce that superposition.

:mod:`cmfqsar.docking` places a molecule where the *pocket* likes it -- complementarity of the
fields with the receptor, less a clash penalty.  It knows nothing about the series, so nothing
makes the pose it finds comparable with the poses the model was fitted on, and a shallow site
leaves it under-determined.

The item puts them together in the obvious way: the complementarity is a **prior** over poses,
the confidence is the **likelihood** of the pose given the model, and the pose to take is the one
that maximises the product.  In the units everything here is already in,

    minimise   sigma^2(pose)  -  w * [ complementarity(pose) - clash(pose) ]

over the six rigid-body parameters, with the displacement guard of the confidence aligner kept so
that neither criterion may walk a molecule out of the site it started in.  `w` is the only new
number: at zero this is the confidence alignment exactly, and as it grows the pocket takes over.

The honest caveats are the ones both halves already carry.  The variance term is not a measure of
the applicability domain once a pose has been chosen to minimise it; the pocket term is not a
docking score, since there is no desolvation, no entropy and no torsional search; and the
internal geometry is never touched, so what is being chosen is a placement and not a
conformation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .docking import PocketScorer
from .fields import FieldSet, MoleculeFields
from .varalign import VarianceAligner

#: how much a unit of pocket complementarity is worth against a unit of posterior variance
POCKET_WEIGHT = 1.0


@dataclass
class PocketAlignResult:
    """What the joint objective did to one molecule."""

    name: str
    coords: np.ndarray
    variance_before: float
    variance_after: float
    pocket_before: float
    pocket_after: float
    clash_before: float
    clash_after: float
    objective_before: float
    objective_after: float
    rmsd: float
    n_eval: int = 0
    limited: bool = False

    def describe(self) -> str:
        return (f"{self.name[:28]:28s} variance {self.variance_before:7.4f} -> "
                f"{self.variance_after:7.4f}   pocket {self.pocket_before:+7.3f} -> "
                f"{self.pocket_after:+7.3f}   moved {self.rmsd:5.2f} A"
                + ("  (at the guard)" if self.limited else ""))


class PocketAligner:
    """Rigid-body alignment by the model's confidence and the pocket's complementarity together.

    The variance half comes from :class:`cmfqsar.varalign.VarianceAligner` and the pocket half
    from :class:`cmfqsar.docking.PocketScorer`, so this class adds no new physics: it is the
    combination, the guard and the search.
    """

    def __init__(self, model, pocket: MoleculeFields, pocket_weight: float = POCKET_WEIGHT,
                 max_rmsd: Optional[float] = 2.0, penalty: float = 50.0,
                 fieldset: Optional[FieldSet] = None, **scorer_kw):
        self.variance = VarianceAligner(model, max_rmsd=max_rmsd, penalty=penalty)
        self.model = model
        self.pocket = pocket
        self.fieldset = fieldset if fieldset is not None else model.fieldset
        self.pocket_weight = float(pocket_weight)
        self.max_rmsd = None if max_rmsd is None else float(max_rmsd)
        self.penalty = float(penalty)
        self.scorer_kw = dict(scorer_kw)

    def _parts(self, mol: MoleculeFields, scorer: PocketScorer, X0: np.ndarray,
              c: np.ndarray, sp, kself, p: np.ndarray):
        """(objective, variance, pocket score, coordinates) at the rigid-body parameters p."""
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        X = (X0 - c) @ R.T + c + p[3:]
        k, _scale = self.variance._kvec(mol, X, sp)
        var = float(kself - k @ self.variance.A @ k)
        poc = float(scorer.score(X))
        obj = var - self.pocket_weight * poc
        if self.max_rmsd is not None:
            msd = float(((X - X0) ** 2).sum() / len(X0))
            lim = self.max_rmsd ** 2
            if msd > lim:
                obj += self.penalty * (msd - lim) ** 2
        return obj, var, poc, X

    def align(self, mol: MoleculeFields, restarts: int = 2, seed: int = 0,
              max_iter: int = 300, apply: bool = True) -> PocketAlignResult:
        """Find the pose that the model and the pocket agree on."""
        scorer = PocketScorer(mol, self.pocket, self.fieldset, **self.scorer_kw)
        X0 = np.array(mol.coords, float)
        c = X0.mean(axis=0)
        sp, kself = self.variance._self_terms(mol)
        counter = {"n": 0}

        def fun(p):
            counter["n"] += 1
            return self._parts(mol, scorer, X0, c, sp, kself, np.asarray(p, float))[0]

        starts = [np.zeros(6)]
        rng = np.random.RandomState(seed)
        for _ in range(max(0, restarts)):
            starts.append(np.concatenate([Rotation.random(random_state=rng).as_rotvec() * 0.35,
                                          rng.normal(scale=0.8, size=3)]))
        best = None
        for p0 in starts:
            # Powell: the pocket term has no analytic gradient (a clash penalty and a tether are
            # not differentiable where they matter), and a six-parameter search does not need one
            res = minimize(fun, p0, method="Powell",
                           options={"maxiter": int(max_iter), "xtol": 1e-3, "ftol": 1e-4})
            if best is None or res.fun < best.fun:
                best = res
        obj0, var0, poc0, _X = self._parts(mol, scorer, X0, c, sp, kself, np.zeros(6))
        obj1, var1, poc1, Xn = self._parts(mol, scorer, X0, c, sp, kself,
                                           np.asarray(best.x, float))
        rmsd = float(np.sqrt(((Xn - X0) ** 2).sum(1).mean()))
        out = PocketAlignResult(
            name=mol.name, coords=Xn, variance_before=var0, variance_after=var1,
            pocket_before=poc0, pocket_after=poc1,
            clash_before=float(scorer.clash(X0)), clash_after=float(scorer.clash(Xn)),
            objective_before=obj0, objective_after=obj1, rmsd=rmsd, n_eval=counter["n"],
            limited=self.max_rmsd is not None and rmsd > 0.995 * self.max_rmsd)
        if apply:
            mol.coords = Xn
        return out


def pocket_align(mols: Sequence[MoleculeFields], model, pocket: MoleculeFields,
                 pocket_weight: float = POCKET_WEIGHT, max_rmsd: Optional[float] = 2.0,
                 restarts: int = 2, seed: int = 0, fieldset: Optional[FieldSet] = None,
                 progress: Optional[Callable[[int, int], None]] = None,
                 cancel: Optional[Callable[[], bool]] = None,
                 **scorer_kw) -> List[PocketAlignResult]:
    """Align every molecule in place by the model's confidence and the pocket together."""
    aligner = PocketAligner(model, pocket, pocket_weight=pocket_weight, max_rmsd=max_rmsd,
                            fieldset=fieldset, **scorer_kw)
    out: List[PocketAlignResult] = []
    for i, mol in enumerate(mols):
        if cancel is not None and cancel():
            break
        out.append(aligner.align(mol, restarts=restarts, seed=seed + i))
        if progress is not None:
            progress(i + 1, len(mols))
    return out


def summarise(results: Sequence[PocketAlignResult]) -> dict:
    """What a run of the joint alignment did, in one line's worth of numbers."""
    if not results:
        return {}
    return {"n": len(results),
            "variance": float(np.median([r.variance_after for r in results])),
            "variance_before": float(np.median([r.variance_before for r in results])),
            "pocket": float(np.median([r.pocket_after for r in results])),
            "pocket_before": float(np.median([r.pocket_before for r in results])),
            "clash": float(np.median([r.clash_after for r in results])),
            "moved": float(np.median([r.rmsd for r in results])),
            "at_the_guard": int(sum(1 for r in results if r.limited)),
            "evaluations": int(np.median([r.n_eval for r in results]))}
