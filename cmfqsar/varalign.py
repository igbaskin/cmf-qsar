"""
Alignment by prediction confidence: the pose of a molecule is chosen so that the
posterior variance of its predicted activity is as small as possible.

This implements the idea put forward in the Conclusions of Baskin & Zhokhova,
Future Med. Chem. (2019): kernel probabilistic regression provides the posterior
variance of every prediction, that variance depends sharply on the position and
orientation of the molecule, and molecules can therefore be aligned by maximising
the confidence of their prediction, i.e. by moving them into the applicability
domain of the 3D-QSAR model.

    sigma^2(x) = K(x, x) - k(x)^T (K + lambda I)^-1 k(x)

K(x, x) depends only on the interatomic distances inside the molecule, so it is
invariant under rigid motion.  For a rigid-body search, minimising the variance is
therefore exactly maximising the quadratic form k(x)^T (K + lambda I)^-1 k(x):
similarity to the whole training set, with redundant training molecules down-weighted
by the inverse kernel matrix.  No activity value of the aligned molecule is used, so
the procedure is legitimate for prospective prediction.

Two limitations are built into the method and are worth keeping in mind.

1. It aligns molecules against a *fixed* training alignment.  Applied to the training
   set itself the criterion is degenerate: superimposing all molecules maximises mutual
   similarity and drives every variance to zero, while destroying the structure-activity
   signal.  Use a chemically motivated alignment (maximum common substructure) for the
   training set and this method for new molecules.
2. Because the pose is chosen to make the model confident, the posterior variance of an
   aligned molecule is no longer an independent measure of the applicability domain.

For a search over conformations (rather than rigid motion) use normalised kernels:
K(x, x) is then constant, otherwise the variance could be lowered by changing the
conformation rather than by improving the fit to the training set.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .fields import AtomTable, MoleculeFields
from .kernels import atom_kernel_block, kernel_gradient, so3_right_jacobian


@dataclass
class VarianceAlignResult:
    """Outcome of :meth:`VarianceAligner.align` for one molecule."""
    name: str
    coords: np.ndarray
    variance_before: float
    variance_after: float
    prediction_before: float
    prediction_after: float
    rmsd: float                 # displacement from the starting pose
    n_eval: int = 0
    limited: bool = False       # the displacement guard was active

    @property
    def improved(self) -> bool:
        return self.variance_after < self.variance_before


class VarianceAligner:
    """Rigid-body alignment of molecules by minimising the posterior variance of a CMF model.

    The model must be fitted; its training poses define the frame that everything is
    aligned to.  Analytic gradients are used (L-BFGS) with optional random restarts.

    Parameters
    ----------
    model : fitted :class:`~cmfqsar.model.CMFModel`
    max_rmsd : maximum displacement (A) from the starting pose.  Beyond it a quadratic
        penalty is applied, which keeps a molecule near its chemically motivated
        (e.g. maximum-common-substructure) pose.  ``None`` disables the guard.
    penalty : weight of that penalty.
    """

    def __init__(self, model, max_rmsd: Optional[float] = 2.0, penalty: float = 50.0):
        if model.coef is None or model.train is None:
            raise RuntimeError("the model must be fitted before it can be used for alignment")
        self.model = model
        self.specs = model.fieldset.specs
        self.max_rmsd = None if max_rmsd is None else float(max_rmsd)
        self.penalty = float(penalty)
        if model.K_fields_train is None:
            # a model loaded from disk keeps the training poses but not the kernel matrices
            model.K_fields_train = model._field_kernels(model.train, model.train,
                                                        model.self_diag, model.self_diag)
        K = model.joint_kernel(model.K_fields_train)
        lam = float(model.params.get("lam", 1.0)) if model.method in ("krr", "gp") else 1e-3
        n = K.shape[0]
        try:
            self.A = np.linalg.inv(K + lam * np.eye(n))
        except np.linalg.LinAlgError:
            self.A = np.linalg.pinv(K + lam * np.eye(n))
        self.coef = model.fieldset.mixing_vector() * model.scale
        self.sq = (np.sqrt(np.maximum(model.self_diag, 1e-300))
                   if (model.normalize and model.self_diag is not None) else None)

    # ------------------------------------------------------------------ pose terms
    def _self_terms(self, mol: MoleculeFields):
        """Per-field 1/sqrt(K_f(x,x)) (or ones) and the joint K(x,x); both pose-invariant."""
        one = AtomTable.from_molecules([mol])
        d_self = atom_kernel_block(mol.coords, mol.weights, mol.radii, one, self.specs).sum(axis=1)[:, 0]
        if self.sq is None:
            return np.ones(len(self.specs)), float(self.coef @ d_self)
        sp = 1.0 / np.sqrt(np.maximum(d_self, 1e-300))
        return sp, float(self.coef.sum())

    def _kvec(self, mol: MoleculeFields, coords: np.ndarray, sp: np.ndarray):
        """Joint kernel row k(x) and the per-field scaling (needed for the gradient)."""
        kraw = atom_kernel_block(coords, mol.weights, mol.radii, self.model.train, self.specs).sum(axis=1)
        scale = np.ones_like(kraw) if self.sq is None else sp[:, None] / np.maximum(self.sq, 1e-300)
        return self.coef @ (kraw * scale), scale

    def variance(self, mol: MoleculeFields, coords: Optional[np.ndarray] = None) -> float:
        """Posterior variance of the prediction for `mol` in the given pose."""
        sp, kself = self._self_terms(mol)
        k, _ = self._kvec(mol, mol.coords if coords is None else coords, sp)
        return float(kself - k @ self.A @ k)

    def predict(self, mol: MoleculeFields, coords: Optional[np.ndarray] = None) -> float:
        sp, _ = self._self_terms(mol)
        k, _ = self._kvec(mol, mol.coords if coords is None else coords, sp)
        return float(k @ self.model.coef + self.model.bias)

    def objective(self, mol: MoleculeFields, X0: np.ndarray, c: np.ndarray, sp: np.ndarray,
                  kself: float, p: np.ndarray):
        """Variance and its analytic gradient for the rigid-body parameters
        ``p = (rotation vector, translation)`` applied about the centroid `c`."""
        V = X0 - c
        R = Rotation.from_rotvec(p[:3]).as_matrix()
        X = V @ R.T + c + p[3:]
        k, scale = self._kvec(mol, X, sp)
        Ak = self.A @ k
        val = kself - float(k @ Ak)
        u = -2.0 * self.coef[:, None] * scale * Ak[None, :]
        g = kernel_gradient(X, mol.weights, mol.radii, self.model.train, self.specs, u)
        if self.max_rmsd is not None:
            d = X - X0
            msd = float((d * d).sum() / len(X0))
            lim = self.max_rmsd ** 2
            if msd > lim:
                val += self.penalty * (msd - lim) ** 2
                g = g + self.penalty * 4.0 * (msd - lim) * d / len(X0)
        h = g @ R                                   # R^T g_l for every atom (row-wise)
        grad = np.empty(6)
        grad[:3] = so3_right_jacobian(p[:3]).T @ np.cross(V, h).sum(axis=0)
        grad[3:] = g.sum(axis=0)
        return val, grad

    # ------------------------------------------------------------------ optimisation
    def align(self, mol: MoleculeFields, restarts: int = 2, seed: int = 0,
              max_iter: int = 200, apply: bool = True) -> VarianceAlignResult:
        """Find the rigid-body pose of `mol` that minimises the posterior variance."""
        X0 = np.array(mol.coords, dtype=float)
        c = X0.mean(axis=0)
        sp, kself = self._self_terms(mol)
        counter = {"n": 0}

        def fun(p):
            counter["n"] += 1
            return self.objective(mol, X0, c, sp, kself, p)

        starts = [np.zeros(6)]
        rng = np.random.RandomState(seed)
        for _ in range(max(0, restarts)):
            starts.append(np.concatenate([Rotation.random(random_state=rng).as_rotvec() * 0.35,
                                          rng.normal(scale=0.8, size=3)]))
        best = None
        for p0 in starts:
            res = minimize(fun, p0, jac=True, method="L-BFGS-B",
                           options={"maxiter": max_iter, "ftol": 1e-14, "gtol": 1e-9})
            if best is None or res.fun < best.fun:
                best = res
        R = Rotation.from_rotvec(best.x[:3]).as_matrix()
        Xn = (X0 - c) @ R.T + c + best.x[3:]
        rmsd = float(np.sqrt(((Xn - X0) ** 2).sum(1).mean()))
        k0, _ = self._kvec(mol, X0, sp)
        kn, _ = self._kvec(mol, Xn, sp)
        out = VarianceAlignResult(
            name=mol.name,
            coords=Xn,
            variance_before=float(kself - k0 @ self.A @ k0),
            variance_after=float(kself - kn @ self.A @ kn),
            prediction_before=float(k0 @ self.model.coef + self.model.bias),
            prediction_after=float(kn @ self.model.coef + self.model.bias),
            rmsd=rmsd, n_eval=counter["n"],
            limited=self.max_rmsd is not None and rmsd > 0.995 * self.max_rmsd)
        if apply:
            mol.coords = Xn
        return out


def variance_align(mols: Sequence[MoleculeFields], model, max_rmsd: Optional[float] = 2.0,
                   restarts: int = 2, seed: int = 0,
                   progress: Optional[Callable[[int, int], None]] = None,
                   cancel: Optional[Callable[[], bool]] = None) -> List[VarianceAlignResult]:
    """Align every molecule of `mols` in place by minimising the model's posterior variance."""
    aligner = VarianceAligner(model, max_rmsd=max_rmsd)
    out: List[VarianceAlignResult] = []
    for i, mol in enumerate(mols):
        if cancel is not None and cancel():
            break
        out.append(aligner.align(mol, restarts=restarts, seed=seed + i))
        if progress is not None:
            progress(i + 1, len(mols))
    return out


# --------------------------------------------------------------------------- #
# Self-consistent alignment of a training set (cross-validated)
# --------------------------------------------------------------------------- #

@dataclass
class CVAlignRound:
    """Diagnostics of one round of :func:`cv_align_training_set`."""
    round: int
    moved_mean: float
    moved_max: float
    drift: float                 # cumulative displacement from the input poses
    similarity: float            # mean off-diagonal normalised kernel (contraction indicator)

    def describe(self) -> str:
        return (f"round {self.round}: moved {self.moved_mean:.2f} A (max {self.moved_max:.2f}), "
                f"drift {self.drift:.2f} A, mean mutual similarity {self.similarity:.4f}")


def ensemble_similarity(mols: Sequence[MoleculeFields], specs) -> float:
    """Mean off-diagonal normalised kernel of a set of molecules.

    It measures how alike the aligned molecules have become.  A value drifting towards 1
    means the ensemble is contracting and the kernel matrix is losing its ability to
    discriminate between compounds.
    """
    from .kernels import kernel_matrices, normalize_kernels, self_kernels
    table = AtomTable.from_molecules(list(mols))
    K = kernel_matrices(table, table, specs)
    d = self_kernels(table, specs)
    Kn = normalize_kernels(K, d, d).mean(axis=0)
    n = Kn.shape[0]
    return float((Kn.sum() - np.trace(Kn)) / (n * (n - 1))) if n > 1 else 1.0


def cv_align_training_set(mols: Sequence[MoleculeFields], y: np.ndarray, model_template,
                          folds: int = 5, rounds: int = 1, max_rmsd: Optional[float] = 1.0,
                          restarts: int = 0, seed: int = 0,
                          progress: Optional[Callable[[str, float], None]] = None,
                          cancel: Optional[Callable[[], bool]] = None):
    """Align a *training* set with the posterior-variance criterion, cross-validated.

    In every fold the held-out molecules are aligned against a model fitted to the remaining
    ones, so no molecule is ever aligned against a model that contains it.  Repeating over the
    folds gives a self-consistent alignment of the whole set.  The molecules are modified in place.

    Measured behaviour (thrombin, glycogen phosphorylase and two automatically aligned sets):
    the iteration converges and the contraction of the ensemble is mild, but the predictive
    gain is not reliable - one set improved, one degraded, two were neutral - and rounds after
    the first consistently made the external prediction worse.  Hence the defaults of a single
    round and a displacement guard.

    Two warnings that the caller should pass on to the user:

    * q^2 of the realigned set is **not** a valid criterion here.  Realigning the training set
      changes the geometry q^2 is computed on, and in the tests q^2 and the external R^2
      repeatedly moved in opposite directions.  Judge the result on held-out data only.
    * the hyper-parameters were selected on the previous alignment and should be refitted.

    Returns
    -------
    list of :class:`CVAlignRound`
    """
    from .model import CMFModel

    mols = list(mols)
    y = np.asarray(y, dtype=float)
    specs = model_template.fieldset.specs
    X_start = [m.coords.copy() for m in mols]
    n = len(mols)
    folds = max(2, min(int(folds), n))
    rng = np.random.RandomState(seed)
    out = []
    for rnd in range(1, int(rounds) + 1):
        order = rng.permutation(n)
        parts = [order[k::folds] for k in range(folds)]
        moved = []
        for k, part in enumerate(parts):
            if cancel is not None and cancel():
                return out
            rest = [i for i in range(n) if i not in set(part.tolist())]
            if len(rest) < 3:
                continue
            if progress is not None:
                progress(f"round {rnd}/{rounds}, fold {k + 1}/{folds}: fitting the reference model",
                         ((rnd - 1) * folds + k) / (rounds * folds))
            sub = CMFModel(fieldset=model_template.fieldset.copy(), method=model_template.method,
                           params=dict(model_template.params), normalize=model_template.normalize,
                           scale_fields=model_template.scale_fields)
            sub.fit(AtomTable.from_molecules([mols[i] for i in rest]), y[rest], cv_folds=0)
            aligner = VarianceAligner(sub, max_rmsd=max_rmsd)
            for i in part:
                before = mols[i].coords.copy()
                aligner.align(mols[i], restarts=restarts, seed=seed + int(i))
                moved.append(float(np.sqrt(((mols[i].coords - before) ** 2).sum(1).mean())))
        drift = float(np.mean([np.sqrt(((m.coords - x0) ** 2).sum(1).mean())
                               for m, x0 in zip(mols, X_start)]))
        out.append(CVAlignRound(round=rnd,
                                moved_mean=float(np.mean(moved)) if moved else 0.0,
                                moved_max=float(np.max(moved)) if moved else 0.0,
                                drift=drift,
                                similarity=ensemble_similarity(mols, specs)))
        if progress is not None:
            progress(out[-1].describe(), rnd / rounds)
    return out
