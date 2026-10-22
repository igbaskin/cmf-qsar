"""
Two endpoints at once, so that selectivity is a co-field rather than a ratio.

A separation problem is not one property but two: an extractant has an affinity for americium and
an affinity for europium, and what matters is the difference.  Modelling the ratio directly works
-- ``benchmarks/run_am_eu.py`` gets q2 = +0.81 on lg SF -- but it throws information away.  Each
endpoint has its own measurements, often on partly different compounds; a ligand that extracts
both metals strongly looks, to a model of the ratio, exactly like one that extracts neither; and
the ratio cannot say *which* of the two affinities a substituent moved.

This module fits both endpoints on one shared CMF kernel:

    y_t(X)  =  sum_j sum_s a_js T(t,s) K(X, X_j)  +  b_t

with `T` a task covariance -- for two tasks, ``[[1, rho], [rho, 1]]`` with one correlation to
choose -- and one bias per task.  Writing the inner sum as an effective coefficient per molecule
and task,

    c_jt  =  sum_s a_js T(t,s),        y_t(X) = sum_j c_jt K(X, X_j) + b_t,

each task's prediction is **linear in the molecular field with its own dual coefficients**, which
is the whole point: it reduces to exactly the object the rest of the program already knows how to
read.  Hand ``c[:, t]`` to :meth:`cmfqsar.model.CMFModel.cofield_coefficients` and you get that
endpoint's co-field; hand it ``c[:, Am] - c[:, Eu]`` and you get the **selectivity co-field**,

    C_Am(r) - C_Eu(r)  =  sum_j (c_j,Am - c_j,Eu) X_j(r),

a map of where more of a field raises the affinity for one metal more than for the other.  Only a
model linear in the field can produce that; it is available here for the same reason co-fields are
available at all.

Three things the implementation has to get right, and does:

* **Cross-validation leaves out molecules, not measurements.**  With a molecule's Am value in the
  training set and its Eu value in the test set, the task correlation hands the answer over.
  :func:`cv_predict` folds over molecules and moves every observation of a held-out molecule
  together.
* **Missing measurements are the normal case**, so the kernel is built over observed (molecule,
  task) pairs rather than as a Kronecker product over a full grid.
* **rho = 0 must reproduce single-task models exactly**, or the sharing cannot be measured against
  anything; ``tests/test_multitask.py`` holds it to that.

What the selected `rho` means is worth saying: at rho = 0 the endpoints are modelled separately;
as rho rises they borrow strength from each other; and at rho = 1 the task covariance is singular,
the effective coefficients of the two tasks coincide, and the model can only predict a *constant*
selectivity.  So the selectivity structure lives strictly below rho = 1, and the value chosen by
cross-validation says how much of each metal's affinity is the other's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


def task_covariance(n_tasks: int, rho: float) -> np.ndarray:
    """``(1 - rho) I + rho 11^T``: one correlation shared by every pair of tasks.

    Positive definite for ``-1/(n_tasks - 1) < rho < 1``; the caller is expected to keep rho
    strictly below 1, where the matrix is singular and selectivity becomes a constant.
    """
    n = int(n_tasks)
    if n < 1:
        raise ValueError("at least one task")
    rho = float(rho)
    if n > 1 and not (-1.0 / (n - 1) < rho < 1.0):
        raise ValueError(f"rho = {rho} is outside the positive-definite range for {n} tasks")
    return (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))


@dataclass
class Observations:
    """Which (molecule, task) pairs were measured, and what was measured."""
    molecule: np.ndarray            # (P,) index into the training molecules
    task: np.ndarray                # (P,) index into the tasks
    y: np.ndarray                   # (P,) activities
    n_mol: int = 0
    n_tasks: int = 0
    task_names: Tuple[str, ...] = ()

    @classmethod
    def from_columns(cls, values: Sequence[Sequence[float]],
                     names: Sequence[str] = ()) -> "Observations":
        """From one column per task, with NaN for a measurement that was not made."""
        Y = np.asarray(values, dtype=float)
        if Y.ndim != 2:
            raise ValueError("values must be (n_molecules, n_tasks)")
        mol, task, y = [], [], []
        for t in range(Y.shape[1]):
            ok = np.where(np.isfinite(Y[:, t]))[0]
            mol.extend(ok.tolist())
            task.extend([t] * len(ok))
            y.extend(Y[ok, t].tolist())
        return cls(np.asarray(mol, int), np.asarray(task, int), np.asarray(y, float),
                   n_mol=Y.shape[0], n_tasks=Y.shape[1],
                   task_names=tuple(names) if names else tuple(f"task{t}" for t in range(Y.shape[1])))

    def __len__(self) -> int:
        return len(self.y)

    def counts(self) -> List[int]:
        return [int(np.sum(self.task == t)) for t in range(self.n_tasks)]

    def subset(self, rows: Sequence[int]) -> "Observations":
        rows = np.asarray(rows, int)
        return Observations(self.molecule[rows], self.task[rows], self.y[rows],
                            self.n_mol, self.n_tasks, self.task_names)


def pair_kernel(K: np.ndarray, rows: Observations, cols: Observations,
                T: np.ndarray) -> np.ndarray:
    """The kernel between observed pairs: ``K[molecule, molecule] * T[task, task]``."""
    return (np.asarray(K)[np.ix_(rows.molecule, cols.molecule)]
            * np.asarray(T)[np.ix_(rows.task, cols.task)])


@dataclass
class MultiTaskFit:
    """A fitted two-or-more-endpoint model and the per-task coefficients it reduces to."""
    alpha: np.ndarray = field(default_factory=lambda: np.zeros(0))   # (P,) dual, per observation
    bias: np.ndarray = field(default_factory=lambda: np.zeros(0))    # (n_tasks,)
    coef: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))  # (n_mol, n_tasks)
    T: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    rho: float = float("nan")
    lam: float = float("nan")
    obs: Optional[Observations] = None

    def task_coefficients(self, task: int) -> np.ndarray:
        """The dual coefficients of one endpoint, over the training molecules.

        These go straight into the single-task machinery -- set them as a model's ``coef`` and its
        co-field, atom contributions and cube export are that endpoint's.
        """
        return self.coef[:, int(task)]

    def selectivity_coefficients(self, task: int, other: int) -> np.ndarray:
        """``c[:, task] - c[:, other]``: the dual coefficients of the selectivity co-field."""
        return self.coef[:, int(task)] - self.coef[:, int(other)]

    def predict(self, K_cross: np.ndarray, task: int) -> np.ndarray:
        """Predict one endpoint for new molecules; `K_cross` is (n_new, n_train_molecules)."""
        return np.asarray(K_cross) @ self.coef[:, int(task)] + self.bias[int(task)]

    def predict_selectivity(self, K_cross: np.ndarray, task: int, other: int) -> np.ndarray:
        """The difference of the two endpoints, which is what a separation factor is."""
        return (np.asarray(K_cross) @ self.selectivity_coefficients(task, other)
                + self.bias[int(task)] - self.bias[int(other)])


def fit(K: np.ndarray, obs: Observations, rho: float = 0.0, lam: float = 1.0,
        fit_bias: bool = True, T: Optional[np.ndarray] = None) -> MultiTaskFit:
    """Kernel ridge regression over the observed (molecule, task) pairs.

    The bordered system of the single-task case (Eq. 12 of Baskin & Zhokhova 2013) gains one
    column per task, so each endpoint gets its own bias and its own ``sum_j a_j = 0``::

        [ G + lam I   B ] [ a ]   [ y ]
        [ B^T         0 ] [ b ] = [ 0 ]

    with ``G`` the pair kernel and ``B[p, t] = 1`` when observation p belongs to task t.
    """
    T = task_covariance(obs.n_tasks, rho) if T is None else np.asarray(T, float)
    G = pair_kernel(K, obs, obs, T)
    p, n_t = len(obs), obs.n_tasks
    if fit_bias:
        B = np.zeros((p, n_t))
        B[np.arange(p), obs.task] = 1.0
        A = np.zeros((p + n_t, p + n_t))
        A[:p, :p] = G + lam * np.eye(p)
        A[:p, p:] = B
        A[p:, :p] = B.T
        rhs = np.concatenate([obs.y, np.zeros(n_t)])
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        a, b = sol[:p], sol[p:]
    else:
        A = G + lam * np.eye(p)
        try:
            a = np.linalg.solve(A, obs.y)
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(A, obs.y, rcond=None)[0]
        b = np.zeros(n_t)
    # c_jt = sum over the observations of molecule j of a_q T(t, task_q)
    coef = np.zeros((obs.n_mol, n_t))
    for t in range(n_t):
        np.add.at(coef[:, t], obs.molecule, a * T[t, obs.task])
    return MultiTaskFit(alpha=a, bias=b, coef=coef, T=T, rho=float(rho), lam=float(lam), obs=obs)


def cv_predict(K: np.ndarray, obs: Observations, rho: float = 0.0, lam: float = 1.0,
               folds: int = 10, seed: Optional[int] = 0,
               fit_bias: bool = True) -> np.ndarray:
    """Out-of-fold predictions for every observation, holding out whole **molecules**.

    A molecule's measurements never straddle the split: with its americium value in the training
    set and its europium value in the test set, the task correlation would hand over the answer
    and the result would be meaningless.
    """
    y = obs.y
    out = np.full(len(obs), np.nan)
    order = np.arange(obs.n_mol)
    if seed is not None:
        np.random.RandomState(seed).shuffle(order)
    k = max(2, min(int(folds), obs.n_mol))
    groups = np.array_split(order, k)
    for g in groups:
        held = set(int(i) for i in g)
        te = np.array([p for p in range(len(obs)) if int(obs.molecule[p]) in held], dtype=int)
        tr = np.array([p for p in range(len(obs)) if int(obs.molecule[p]) not in held], dtype=int)
        if len(te) == 0 or len(tr) == 0:
            continue
        sub = obs.subset(tr)
        f = fit(K, sub, rho=rho, lam=lam, fit_bias=fit_bias)
        for p in te:
            m, t = int(obs.molecule[p]), int(obs.task[p])
            out[p] = float(K[m] @ f.coef[:, t] + f.bias[t])
    return out


def score_by_task(obs: Observations, pred: np.ndarray) -> List[dict]:
    """q2 and RMSE per task, computed where the predictions exist."""
    from .model import r2_score, rmse
    out = []
    for t in range(obs.n_tasks):
        m = (obs.task == t) & np.isfinite(pred)
        name = obs.task_names[t] if t < len(obs.task_names) else f"task{t}"
        if not np.any(m):
            out.append({"task": name, "n": 0, "q2": float("nan"), "rmse_cv": float("nan")})
            continue
        out.append({"task": name, "n": int(m.sum()),
                    "q2": float(r2_score(obs.y[m], pred[m])),
                    "rmse_cv": float(rmse(obs.y[m], pred[m]))})
    return out


def score_selectivity(K: np.ndarray, obs: Observations, task: int, other: int,
                      rho: float, lam: float, folds: int = 10, seed: Optional[int] = 0,
                      fit_bias: bool = True) -> dict:
    """Cross-validated q2 of the *difference* of the two endpoints.

    Only molecules carrying both measurements can be scored, since only they have an observed
    difference; the model is still fitted on everything, which is the point of doing it this way.
    """
    from .model import r2_score, rmse
    Y = np.full((obs.n_mol, obs.n_tasks), np.nan)
    Y[obs.molecule, obs.task] = obs.y
    both = np.where(np.isfinite(Y[:, task]) & np.isfinite(Y[:, other]))[0]
    truth = Y[both, task] - Y[both, other]
    pred = np.full(len(both), np.nan)
    order = np.arange(obs.n_mol)
    if seed is not None:
        np.random.RandomState(seed).shuffle(order)
    k = max(2, min(int(folds), obs.n_mol))
    for g in np.array_split(order, k):
        held = set(int(i) for i in g)
        tr = np.array([p for p in range(len(obs)) if int(obs.molecule[p]) not in held], dtype=int)
        if len(tr) == 0:
            continue
        f = fit(K, obs.subset(tr), rho=rho, lam=lam, fit_bias=fit_bias)
        for i, m in enumerate(both):
            if int(m) in held:
                pred[i] = float(K[m] @ f.selectivity_coefficients(task, other)
                                + f.bias[task] - f.bias[other])
    ok = np.isfinite(pred)
    return {"n": int(ok.sum()), "q2": float(r2_score(truth[ok], pred[ok])) if ok.any() else float("nan"),
            "rmse_cv": float(rmse(truth[ok], pred[ok])) if ok.any() else float("nan"),
            "predicted": pred, "observed": truth, "molecules": both}


def search(K: np.ndarray, obs: Observations, rhos: Sequence[float] = (0.0, 0.3, 0.5, 0.7, 0.9),
           lams: Sequence[float] = (1e-3, 1e-2, 0.1, 1.0, 10.0), folds: int = 10,
           seed: Optional[int] = 0, fit_bias: bool = True,
           statistic: str = "mean") -> Tuple[float, float, float, List[dict]]:
    """Choose (rho, lambda) by cross-validation; returns (rho, lam, score, per-task scores).

    `statistic` is what the search maximises: ``"mean"`` the mean q2 over the tasks, or the name
    of one task to tune on that endpoint alone.
    """
    best = None
    for rho in rhos:
        for lam in lams:
            pred = cv_predict(K, obs, rho=rho, lam=lam, folds=folds, seed=seed,
                              fit_bias=fit_bias)
            per = score_by_task(obs, pred)
            if statistic == "mean":
                s = float(np.nanmean([p["q2"] for p in per]))
            else:
                s = next((p["q2"] for p in per if p["task"] == statistic), float("nan"))
            if best is None or s > best[2]:
                best = (float(rho), float(lam), s, per)
    return best
