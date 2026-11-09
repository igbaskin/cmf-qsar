"""
Relaxation of a molecule under the gradient of its own predicted activity (plan item MS3).

:mod:`cmfqsar.gradients` makes a prediction a differentiable function of the atom positions, so a
molecule can be *moved* by the model instead of being tried in one pose after another.  This
module does the one motion that leaves the chemistry alone: the rigid motion.  Six parameters --
a rotation vector and a translation, about the molecule's own centroid -- are optimised with the
analytic gradient, so the conformation, the bond lengths and the field weights are exactly what
they were and only the pose changes.

Why the rigid motion and not the atoms
--------------------------------------
Following the gradient with every atom free raises the prediction faster, and it damages the
molecule while doing so: the gradient knows nothing about valence, and
``benchmarks/run_gradient.py`` measures how far the bond lengths drift when the atoms are let go.
A rigid motion cannot do that. It is also the motion a 3D-QSAR model is allowed to make a
statement about, since the training poses define the frame and the kernel compares fields in it.

The companion objective
-----------------------
:mod:`cmfqsar.varalign` moves a molecule by *confidence*: it minimises the posterior variance of
the prediction, which puts the molecule where the model is sure of itself.  This module moves it
by the *prediction*, which puts the molecule where the model says the activity is highest.  The
two objectives are different questions and their answers need not be the same pose; which they
are, on a real series, is measured by ``benchmarks/run_gradient.py --stage relax``.  Confidence
is the one to use when a pose is wanted for prospective prediction, because it uses no activity
value of the molecule being placed; the prediction is the one to use when the question is where a
molecule would have to sit for the model to call it active.

What it cannot do
-----------------
The pose it finds is a statement of the model, not of the physics: nothing here knows about the
receptor, the solvent or a steric clash with anything outside the training set.  The displacement
guard (`max_rmsd`) is what keeps the answer near the pose the molecule arrived in, and a
relaxation that ends at the guard is reported as `limited`, because its answer is the guard's and
not the model's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .fields import MoleculeFields
from .gradients import prediction_gradient
from .kernels import so3_right_jacobian
from .model import CMFModel


@dataclass
class RelaxResult:
    """Outcome of :func:`relax_pose` for one molecule."""
    name: str
    coords: np.ndarray            # the pose it ended in, (N, 3)
    value_before: float           # the prediction in the pose it arrived in
    value_after: float            # the prediction in the pose it ended in
    rmsd: float                   # how far the atoms moved, in angstroms
    n_eval: int                   # objective evaluations the search needed
    limited: bool                 # the displacement guard was active at the end
    converged: bool
    message: str = ""

    @property
    def gain(self) -> float:
        """How much the prediction rose (or fell, when the search was told to minimise)."""
        return float(self.value_after - self.value_before)


def _rigid(X0: np.ndarray, centre: np.ndarray, p: np.ndarray) -> np.ndarray:
    """`X0` turned by the rotation vector `p[:3]` about `centre` and shifted by `p[3:]`."""
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    return (X0 - centre) @ R.T + centre + p[3:]


def relax_pose(model: CMFModel, mol: MoleculeFields, max_rmsd: Optional[float] = 2.0,
               penalty: float = 50.0, maximise: bool = True, max_iter: int = 200,
               gradient: bool = True, apply: bool = False,
               starts: Optional[Sequence[np.ndarray]] = None) -> RelaxResult:
    """Move `mol` as a rigid body to the pose where the model predicts most (or least).

    Parameters
    ----------
    model : a fitted model on the overlap-integral kernel.  An invariant kernel is refused: its
        channels are built from the distances inside the molecule, so a rigid motion does not
        change its prediction and there is nothing to relax.
    mol : the molecule, in the frame of the training set.
    max_rmsd : how far the atoms may move from the arriving pose, in angstroms, before a
        quadratic penalty holds them back; None removes the guard.
    penalty : the weight of that penalty, as in :class:`cmfqsar.varalign.VarianceAligner`.
    maximise : raise the prediction (the default) or lower it.
    gradient : drive the search with the analytic gradient (L-BFGS-B).  With False the same
        objective is searched without it (Nelder-Mead), which is how the benchmark measures what
        the gradient is worth; the pose should be the same and the number of evaluations larger.
    apply : write the pose back into `mol`.
    starts : rigid-body parameters to start from, six numbers each; the arriving pose by default.

    Returns
    -------
    :class:`RelaxResult`
    """
    if model.coef is None or model.train is None:
        raise RuntimeError("the model must be fitted before a molecule can be relaxed under it")
    if not model.uses_fields:
        raise ValueError(
            f"a rigid motion does not change the prediction of the '{model.kernel_source}' "
            f"kernel: its channels are built from the distances inside the molecule and are "
            f"invariant to translation and rotation. There is no pose for it to relax. Fit a "
            f"model on the overlap-integral kernel to refine a pose, or move the atoms "
            f"themselves (benchmarks/run_gradient.py follows the gradient that way).")
    X0 = np.array(mol.coords, float)
    centre = X0.mean(axis=0)
    sign = -1.0 if maximise else 1.0           # the search minimises, so maximising flips the sign
    counter = {"n": 0}

    def guard(X):
        """The displacement penalty and its derivative in the positions."""
        if max_rmsd is None:
            return 0.0, np.zeros_like(X)
        d = X - X0
        msd = float((d * d).sum() / len(X0))
        lim = float(max_rmsd) ** 2
        if msd <= lim:
            return 0.0, np.zeros_like(X)
        return penalty * (msd - lim) ** 2, penalty * 4.0 * (msd - lim) * d / len(X0)

    def value_only(p):
        counter["n"] += 1
        X = _rigid(X0, centre, p)
        g = prediction_gradient(model, mol, coords=X, wrt="coords")
        pen, _ = guard(X)
        return sign * g.value + pen

    def value_and_grad(p):
        counter["n"] += 1
        X = _rigid(X0, centre, p)
        g = prediction_gradient(model, mol, coords=X, wrt="coords")
        pen, dpen = guard(X)
        gx = sign * g.coords + dpen
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        h = gx @ R                              # the derivative seen in the unturned molecule
        out = np.empty(6)
        out[:3] = so3_right_jacobian(p[:3]).T @ np.cross(X0 - centre, h).sum(axis=0)
        out[3:] = gx.sum(axis=0)
        return sign * g.value + pen, out

    tries = [np.zeros(6)] if starts is None else [np.asarray(p, float) for p in starts]
    best = None
    for p0 in tries:
        if gradient:
            res = minimize(value_and_grad, p0, jac=True, method="L-BFGS-B",
                           options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-9})
        else:
            res = minimize(value_only, p0, method="Nelder-Mead",
                           options={"maxiter": max_iter * 20, "fatol": 1e-10, "xatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res

    X = _rigid(X0, centre, best.x)
    before = float(prediction_gradient(model, mol, coords=X0, wrt="coords").value)
    after = float(prediction_gradient(model, mol, coords=X, wrt="coords").value)
    rmsd = float(np.sqrt(((X - X0) ** 2).sum(axis=1).mean()))
    out = RelaxResult(name=mol.name, coords=X, value_before=before, value_after=after,
                      rmsd=rmsd, n_eval=counter["n"],
                      limited=max_rmsd is not None and rmsd > 0.995 * float(max_rmsd),
                      converged=bool(best.success), message=str(best.message))
    if apply:
        mol.coords = X
    return out


def relax_poses(model: CMFModel, mols: Sequence[MoleculeFields], **kwargs) -> List[RelaxResult]:
    """:func:`relax_pose` for every molecule of a list, in order."""
    return [relax_pose(model, m, **kwargs) for m in mols]


def pose_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """The root mean square distance between two poses of the same atoms, atom by atom.

    The atoms are compared where they are, without superposition: two poses of one molecule are
    being compared, and how far apart they sit is the whole question.
    """
    A, B = np.asarray(a, float), np.asarray(b, float)
    if A.shape != B.shape:
        raise ValueError(f"the two poses have shapes {A.shape} and {B.shape}")
    return float(np.sqrt(((A - B) ** 2).sum(axis=1).mean()))
