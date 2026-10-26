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

More than two endpoints
-----------------------

One shared `rho` assumes that every pair of endpoints is related to the same degree.  With two
endpoints that assumption is empty.  With seven it is strong, and usually wrong: some endpoints
are almost the same measurement, others are only distantly related.  Two more general forms are
implemented here.  Both are selected by the **marginal likelihood** instead of
cross-validation, because a covariance with ten free parameters needs one matrix factorisation per
candidate this way and ten per candidate with ten-fold cross-validation.

* **A free-form low-rank covariance.**  ``T = L L' + diag(d)``, with `L` of shape (tasks, rank)
  and `d > 0`.  It is positive definite for any parameter values, it has (rank + 1) * tasks
  parameters, and at rank = tasks it can represent any covariance.  :func:`fit_covariance` fits
  it.  Its diagonal is *not* fixed at one, which matters when the endpoints are in different
  units: a potency and a free energy in kJ/mol need separate variances, and one shared rho gives
  them the same variance.

* **A covariance computed from task descriptors.**  ``T(t,s) = k(z_t, z_s)``, where `z` is a
  vector of properties of the endpoint -- the medium, the metal, the temperature, or, in a
  host-guest problem, the structure of the host.  :func:`descriptor_covariance` selects the
  kernel's length scale and nugget by the evidence.  Only this form can say anything about an
  endpoint with **no measurements at all**: a free-form row cannot be estimated for an endpoint
  with no data, while a descriptor row is computed from the descriptors alone.
  :meth:`MultiTaskFit.predict_new_task` then predicts that endpoint from the related ones.

The mean needs the same change.  A separate bias per endpoint is one indicator column per
endpoint, and an endpoint with no measurements has no bias to estimate.  Passing
``task_features`` to :class:`Observations` replaces those indicator columns with the descriptors,
so the mean is a linear function of them and can be evaluated for an endpoint that was never
measured.  Without task features the design matrix is the indicator matrix and the behaviour is
unchanged.

One limitation remains.  An endpoint with no measurements is predicted by *interpolation between
the endpoints that were measured*, so the prediction contains only what those endpoints have in
common.  ``benchmarks/run_tasks.py`` compares it with two reference values that need no model: the
mean of the measured endpoints, and the single most similar endpoint by descriptors.
"""
from __future__ import annotations

import math
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
    #: (n_tasks, d) descriptors of the endpoints themselves, with an intercept column, or None
    #: for a separate free bias per endpoint.  With descriptors the mean is a linear function of
    #: them, so an endpoint that was never measured also has a mean.
    task_features: Optional[np.ndarray] = None

    @classmethod
    def from_columns(cls, values: Sequence[Sequence[float]],
                     names: Sequence[str] = (),
                     task_features: Optional[np.ndarray] = None) -> "Observations":
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
        Z = None if task_features is None else np.asarray(task_features, float)
        if Z is not None and Z.shape[0] != Y.shape[1]:
            raise ValueError("task_features must have one row per task")
        return cls(np.asarray(mol, int), np.asarray(task, int), np.asarray(y, float),
                   n_mol=Y.shape[0], n_tasks=Y.shape[1],
                   task_names=tuple(names) if names else tuple(f"task{t}" for t in range(Y.shape[1])),
                   task_features=Z)

    def design(self) -> np.ndarray:
        """``B``, the (P, d) design matrix of the mean: indicator columns, or the descriptors.

        The bordered system of :func:`fit` imposes ``B' a = 0``. With indicator columns that is
        one constraint per endpoint, namely that its coefficients sum to zero. With descriptor
        columns it is the generalised-least-squares normal equation for a mean that is a linear
        function of the descriptors.
        """
        if self.task_features is None:
            B = np.zeros((len(self.y), self.n_tasks))
            B[np.arange(len(self.y)), self.task] = 1.0
            return B
        return np.asarray(self.task_features, float)[self.task]

    def task_design(self) -> np.ndarray:
        """``(n_tasks, d)``: the row of the design each endpoint would contribute."""
        if self.task_features is None:
            return np.eye(self.n_tasks)
        return np.asarray(self.task_features, float)

    def __len__(self) -> int:
        return len(self.y)

    def counts(self) -> List[int]:
        return [int(np.sum(self.task == t)) for t in range(self.n_tasks)]

    def subset(self, rows: Sequence[int]) -> "Observations":
        rows = np.asarray(rows, int)
        return Observations(self.molecule[rows], self.task[rows], self.y[rows],
                            self.n_mol, self.n_tasks, self.task_names, self.task_features)


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
    #: the coefficients of the mean on the design columns; with indicator columns they are the
    #: biases of the endpoints, and with task descriptors they define the mean of a new endpoint
    mean_coef: np.ndarray = field(default_factory=lambda: np.zeros(0))

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

    def new_task_coefficients(self, t_row: Sequence[float]) -> np.ndarray:
        """Dual coefficients over the training molecules for an endpoint that was never measured.

        `t_row` is the covariance of that endpoint with the fitted ones. Only a covariance
        computed from task descriptors can supply it. The coefficients are
        ``c_j = sum_q a_q T(new, task_q)``, summed over the observations of molecule j. Pass the
        result to the co-field functions and the map obtained is that of the new endpoint.
        """
        t_row = np.asarray(t_row, float).ravel()
        if self.obs is None:
            raise ValueError("the fit does not carry its observations")
        if len(t_row) != self.obs.n_tasks:
            raise ValueError(f"t_row must have one entry per trained task ({self.obs.n_tasks})")
        c = np.zeros(self.obs.n_mol)
        np.add.at(c, self.obs.molecule, self.alpha * t_row[self.obs.task])
        return c

    def predict_new_task(self, K_cross: np.ndarray, t_row: Sequence[float],
                         features: Optional[Sequence[float]] = None) -> np.ndarray:
        """Predict an endpoint with no measurements, from its covariance and its descriptors.

        `features` is the row of the task design matrix for that endpoint. Without task features
        there is no bias to evaluate, because the indicator column of an unmeasured endpoint was
        never estimated. In that case the fitted biases are averaged with weights taken from the
        covariance. That average is an approximation, not an estimate of the new endpoint's bias.
        """
        c = self.new_task_coefficients(t_row)
        if features is not None:
            b = float(np.asarray(features, float).ravel() @ self.mean_coef)
        else:
            w = np.abs(np.asarray(t_row, float).ravel())
            b = float(w @ self.bias / w.sum()) if w.sum() > 0 else float(np.mean(self.bias))
        return np.asarray(K_cross) @ c + b


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
        B = obs.design()
        d = B.shape[1]
        A = np.zeros((p + d, p + d))
        A[:p, :p] = G + lam * np.eye(p)
        A[:p, p:] = B
        A[p:, :p] = B.T
        rhs = np.concatenate([obs.y, np.zeros(d)])
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
        a, w = sol[:p], sol[p:]
    else:
        A = G + lam * np.eye(p)
        try:
            a = np.linalg.solve(A, obs.y)
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(A, obs.y, rcond=None)[0]
        w = np.zeros(obs.task_design().shape[1])
    b = obs.task_design() @ w
    # c_jt = sum over the observations of molecule j of a_q T(t, task_q)
    coef = np.zeros((obs.n_mol, n_t))
    for t in range(n_t):
        np.add.at(coef[:, t], obs.molecule, a * T[t, obs.task])
    return MultiTaskFit(alpha=a, bias=b, coef=coef, T=T, rho=float(rho), lam=float(lam), obs=obs,
                        mean_coef=w)


def cv_predict(K: np.ndarray, obs: Observations, rho: float = 0.0, lam: float = 1.0,
               folds: int = 10, seed: Optional[int] = 0,
               fit_bias: bool = True, T: Optional[np.ndarray] = None) -> np.ndarray:
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
        f = fit(K, sub, rho=rho, lam=lam, fit_bias=fit_bias, T=T)
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
                      fit_bias: bool = True, T: Optional[np.ndarray] = None) -> dict:
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
        f = fit(K, obs.subset(tr), rho=rho, lam=lam, fit_bias=fit_bias, T=T)
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

# ----------------------------------------------------------------- the evidence of a covariance
#
# Choosing a covariance with (rank + 1) * tasks parameters by cross-validation would need ten
# fits for each of a few hundred candidates.  The marginal likelihood needs one matrix
# factorisation for each.  It is the multi-task form of :mod:`cmfqsar.evidence`: the mean is
# integrated out under a flat prior over the design columns (one bias per endpoint, or the task
# descriptors), and the kernel amplitude is substituted at its maximum, so the value does not
# depend on the scale of the prior and the ranking does not depend on the units of the activities.


def log_evidence(K: np.ndarray, obs: Observations, T: np.ndarray, lam: float,
                 fit_bias: bool = True, profile: bool = True) -> float:
    """The log marginal likelihood of the observations under `T` and `lam`.

    With `fit_bias` the mean is integrated out over the design columns of `obs`. This extends the
    single-task case, where one unknown constant is integrated out, to one constant per endpoint,
    or, with task descriptors, to a mean that is a linear function of them. The degrees of freedom
    are reduced by the number of design columns, which puts covariances of different rank on the
    same scale.
    """
    from scipy.linalg import cho_factor, cho_solve

    y = np.asarray(obs.y, float).ravel()
    n = len(y)
    if n < 3:
        return float("nan")
    A = pair_kernel(K, obs, obs, T) + float(lam) * np.eye(n)
    A = 0.5 * (A + A.T)
    # the search visits covariances that are only just positive definite, so a series of
    # increasing jitter values is tried instead of a single attempt; the triangular solves use
    # cho_solve, because a general solve against a Cholesky factor performs another cubic
    # factorisation and this function is called a few thousand times per fit
    scale = float(np.mean(np.diag(A))) or 1.0
    c = None
    for jitter in [0.0] + [10.0 ** p for p in range(-12, -2)]:
        try:
            c = cho_factor(A + jitter * scale * np.eye(n), lower=True, check_finite=False)
            break
        except Exception:                               # noqa: BLE001 - not positive definite
            continue
    if c is None:
        return float("-inf")
    log_det = 2.0 * float(np.sum(np.log(np.abs(np.diag(c[0])))))

    def solve(b):
        return cho_solve(c, b, check_finite=False)

    if fit_bias:
        B = obs.design()
        AB = solve(B)
        M = B.T @ AB
        try:
            sign, log_det_M = np.linalg.slogdet(M)
        except np.linalg.LinAlgError:
            return float("-inf")
        if sign <= 0 or not np.isfinite(log_det_M):
            return float("-inf")
        try:
            w = np.linalg.solve(M, B.T @ solve(y))
        except np.linalg.LinAlgError:
            return float("-inf")
        resid = y - B @ w
        quad = float(resid @ solve(resid))
        dof = n - B.shape[1]
        extra = -0.5 * log_det_M
    else:
        quad = float(y @ solve(y))
        dof = n
        extra = 0.0
    if dof < 1 or not np.isfinite(quad) or quad <= 0:
        return float("-inf")
    if profile:
        return float(-0.5 * dof * (math.log(quad / dof) + 1.0)
                     - 0.5 * log_det + extra - 0.5 * dof * math.log(2.0 * math.pi))
    return float(-0.5 * quad - 0.5 * log_det + extra - 0.5 * dof * math.log(2.0 * math.pi))


# ------------------------------------------------------------- a free-form low-rank covariance


def low_rank_covariance(theta: Sequence[float], n_tasks: int, rank: int) -> np.ndarray:
    """``L L' + diag(exp(2 u))`` from a flat parameter vector, normalised to mean diagonal 1.

    The result is positive definite for every `theta`, so the optimiser needs no constraints. The
    normalisation removes the one direction the evidence cannot distinguish: multiplying `T` and
    `lam` by the same factor leaves the fitted model unchanged, so without the normalisation the
    optimiser would move along that direction without improving anything. The *relative* sizes of
    the diagonal entries stay free, so endpoints in different units get separate variances.
    """
    n, r = int(n_tasks), int(rank)
    theta = np.asarray(theta, float).ravel()
    if len(theta) != n * (r + 1):
        raise ValueError(f"theta must hold {n * (r + 1)} values for {n} tasks at rank {r}")
    L = theta[:n * r].reshape(n, r)
    u = np.clip(theta[n * r:], -8.0, 8.0)
    M = L @ L.T + np.diag(np.exp(2.0 * u))
    scale = float(np.mean(np.diag(M)))
    return M / scale if scale > 1e-300 else np.eye(n)


def _start_vector(n_tasks: int, rank: int, rho: float) -> np.ndarray:
    """A start whose rank-1 part reproduces one shared correlation `rho`."""
    n, r = int(n_tasks), int(rank)
    L = np.zeros((n, r))
    L[:, 0] = math.sqrt(max(rho, 1e-6))
    if r > 1:
        L[:, 1:] = 0.05 * np.eye(n, r - 1)
    u = np.full(n, 0.5 * math.log(max(1.0 - rho, 1e-6)))
    return np.concatenate([L.ravel(), u])


@dataclass
class CovarianceFit:
    """A task covariance chosen by the evidence, and what it cost to choose."""
    T: np.ndarray
    lam: float
    evidence: float
    rank: int
    evaluations: int = 0
    kind: str = "free"
    note: str = ""

    @property
    def correlations(self) -> np.ndarray:
        """`T` scaled to unit diagonal, which is the part that is comparable with `rho`."""
        d = np.sqrt(np.clip(np.diag(self.T), 1e-300, None))
        return self.T / np.outer(d, d)

    def describe(self, names: Sequence[str] = ()) -> str:
        C = self.correlations
        n = C.shape[0]
        names = list(names) or [f"task{t}" for t in range(n)]
        off = [C[i, j] for i in range(n) for j in range(i + 1, n)]
        lines = [f"{self.kind} covariance, rank {self.rank}, lambda {self.lam:g}, "
                 f"log evidence {self.evidence:+.1f}",
                 f"  correlations from {min(off):+.2f} to {max(off):+.2f}" if off else ""]
        for i in range(n):
            lines.append("  " + f"{names[i][:18]:>18s} " +
                         " ".join(f"{C[i, j]:+.2f}" for j in range(n)))
        return "\n".join(ln for ln in lines if ln)


def fit_covariance(K: np.ndarray, obs: Observations, rank: int = 1,
                   lams: Sequence[float] = (0.1,),
                   rhos: Sequence[float] = (0.2, 0.8),
                   fit_bias: bool = True, maxiter: int = 80) -> CovarianceFit:
    """Optimise a free-form low-rank task covariance on the evidence.

    Nelder-Mead over the ``(rank + 1) * tasks`` parameters and ``log lam``. It is started from
    the rank-1 point that reproduces each `rho` of `rhos` at each `lam` of `lams`, and the best
    result is kept. Several starting points are needed because the evidence as a function of the
    covariance is not concave: on real data a start at rho = 0.9 and a start at rho = 0.1 end at
    different optima.
    """
    from scipy.optimize import minimize

    n_t = obs.n_tasks
    if n_t < 2:
        raise ValueError("a task covariance needs at least two tasks")
    best: Optional[CovarianceFit] = None
    evals = 0

    for rho in rhos:
        for lam0 in lams:
            x0 = np.concatenate([_start_vector(n_t, rank, float(rho)),
                                 [math.log(float(lam0))]])

            def neg(x):
                T = low_rank_covariance(x[:-1], n_t, rank)
                lam = math.exp(float(np.clip(x[-1], -14.0, 8.0)))
                v = log_evidence(K, obs, T, lam, fit_bias=fit_bias)
                return -v if np.isfinite(v) else 1e18

            step = np.full(len(x0), 0.3)
            simplex = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * step[i]
                                        for i in range(len(x0))])
            res = minimize(neg, x0, method="Nelder-Mead",
                           options={"maxiter": int(maxiter) * len(x0), "xatol": 1e-3,
                                    "fatol": 1e-3, "initial_simplex": simplex})
            evals += int(res.nfev)
            if not np.isfinite(res.fun) or res.fun >= 1e17:
                continue
            cand = CovarianceFit(T=low_rank_covariance(res.x[:-1], n_t, rank),
                                 lam=float(math.exp(float(np.clip(res.x[-1], -14.0, 8.0)))),
                                 evidence=float(-res.fun), rank=int(rank), kind="free")
            if best is None or cand.evidence > best.evidence:
                best = cand
    if best is None:
        raise RuntimeError("the evidence could not be evaluated anywhere")
    best.evaluations = evals
    best.note = (f"{len(rhos)} x {len(lams)} starts, {evals} evaluations of the evidence; "
                 f"cross-validation would have cost ten fits each")
    return best


# --------------------------------------------------------- a covariance from task descriptors


def standardise(Z: np.ndarray) -> np.ndarray:
    """Columns centred and scaled to unit spread, with constant columns dropped to zero."""
    Z = np.asarray(Z, float)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    out = np.zeros_like(Z)
    ok = sd > 1e-12
    out[:, ok] = (Z[:, ok] - mu[ok]) / sd[ok]
    return out


def task_kernel(Z: np.ndarray, length_scale: float = 1.0, nugget: float = 0.1,
                kind: str = "rbf", Z_new: Optional[np.ndarray] = None) -> np.ndarray:
    """The covariance between endpoints, from their descriptors.

    `kind` is ``"rbf"``, that is ``exp(-|z_t - z_s|^2 / 2 l^2)``, so that endpoints with similar
    descriptors are strongly correlated and distant ones are not; or ``"linear"``, the normalised
    inner product, which can be negative and is the form to use when two endpoints are expected to
    be *negatively* correlated (a potency and a free energy). The nugget is added to the diagonal.
    Without it, two endpoints with identical descriptors would be perfectly correlated, the
    covariance would be singular, and their coefficients would be forced to be equal.

    With `Z_new` the result is the (n_new, n_tasks) cross-covariance, which is what is needed for
    an endpoint with no measurements.
    """
    Z = np.asarray(Z, float)
    rows = Z if Z_new is None else np.asarray(Z_new, float)
    ls = max(float(length_scale), 1e-6)
    if kind == "rbf":
        d2 = (np.sum(rows ** 2, axis=1)[:, None] + np.sum(Z ** 2, axis=1)[None, :]
              - 2.0 * rows @ Z.T)
        Tm = np.exp(-np.clip(d2, 0.0, None) / (2.0 * ls * ls))
    elif kind == "linear":
        nr = np.sqrt(np.clip(np.sum(rows ** 2, axis=1), 1e-12, None))
        nz = np.sqrt(np.clip(np.sum(Z ** 2, axis=1), 1e-12, None))
        Tm = (rows @ Z.T) / np.outer(nr, nz)
        Tm = np.clip(Tm, -1.0, 1.0)
    else:
        raise ValueError(f"unknown task kernel {kind!r}; use 'rbf' or 'linear'")
    eps = float(np.clip(nugget, 0.0, 1.0))
    if Z_new is None:
        Tm = (1.0 - eps) * Tm + eps * np.eye(Tm.shape[0])
    else:
        Tm = (1.0 - eps) * Tm
    return Tm


@dataclass
class DescriptorCovariance:
    """A covariance read off task descriptors, and the rule for a new endpoint's row."""
    T: np.ndarray
    lam: float
    evidence: float
    length_scale: float
    nugget: float
    kind: str
    Z: np.ndarray                      # the standardised descriptors the covariance was built on
    mean: np.ndarray                   # the standardisation, so a new endpoint can be put on it
    scale: np.ndarray

    def row_for(self, z_new: Sequence[float]) -> np.ndarray:
        """The covariance of an endpoint with the trained ones, from its raw descriptors."""
        z = np.asarray(z_new, float).ravel()[None, :]
        ok = self.scale > 1e-12
        zs = np.zeros_like(z)
        zs[:, ok] = (z[:, ok] - self.mean[ok]) / self.scale[ok]
        return task_kernel(self.Z, self.length_scale, self.nugget, self.kind, Z_new=zs)[0]

    def as_fit(self) -> CovarianceFit:
        return CovarianceFit(T=self.T, lam=self.lam, evidence=self.evidence, rank=self.T.shape[0],
                             kind=f"descriptor ({self.kind})",
                             note=f"length scale {self.length_scale:g}, nugget {self.nugget:g}")


def descriptor_covariance(K: np.ndarray, obs: Observations, Z: np.ndarray,
                          length_scales: Sequence[float] = (0.25, 0.5, 1.0, 2.0, 4.0),
                          nuggets: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
                          lams: Sequence[float] = (1e-2, 0.1, 1.0),
                          kinds: Sequence[str] = ("rbf",),
                          fit_bias: bool = True) -> DescriptorCovariance:
    """Choose the task kernel's length scale, nugget and ridge by the evidence.

    A grid rather than Nelder-Mead, because there are only three or four numbers and the grid is
    the thing to show in a table.
    """
    Z = np.asarray(Z, float)
    if Z.shape[0] != obs.n_tasks:
        raise ValueError("Z must have one row per task")
    mean, scale = Z.mean(axis=0), Z.std(axis=0)
    Zs = standardise(Z)
    best = None
    for kind in kinds:
        for ls in length_scales:
            for eps in nuggets:
                T = task_kernel(Zs, ls, eps, kind)
                for lam in lams:
                    v = log_evidence(K, obs, T, float(lam), fit_bias=fit_bias)
                    if np.isfinite(v) and (best is None or v > best[0]):
                        best = (float(v), T, float(lam), float(ls), float(eps), kind)
    if best is None:
        raise RuntimeError("the evidence could not be evaluated on any setting")
    v, T, lam, ls, eps, kind = best
    return DescriptorCovariance(T=T, lam=lam, evidence=v, length_scale=ls, nugget=eps, kind=kind,
                                Z=Zs, mean=mean, scale=scale)


def evidence_rho(K: np.ndarray, obs: Observations, rhos: Sequence[float] = (),
                 lams: Sequence[float] = (1e-2, 0.1, 1.0),
                 fit_bias: bool = True) -> CovarianceFit:
    """The one-correlation covariance, chosen by the evidence instead of by cross-validation.

    This is the reference for the free-form and descriptor covariances: the same objective and the
    same ridge grid, with one parameter.
    """
    n_t = obs.n_tasks
    if not rhos:
        lo = -1.0 / (n_t - 1) if n_t > 1 else -0.9
        rhos = [float(v) for v in np.linspace(lo + 1e-3, 0.95, 25)]
    best = None
    for rho in rhos:
        try:
            T = task_covariance(n_t, float(rho))
        except ValueError:
            continue
        for lam in lams:
            v = log_evidence(K, obs, T, float(lam), fit_bias=fit_bias)
            if np.isfinite(v) and (best is None or v > best[0]):
                best = (float(v), T, float(lam), float(rho))
    if best is None:
        raise RuntimeError("the evidence could not be evaluated on any rho")
    v, T, lam, rho = best
    f = CovarianceFit(T=T, lam=lam, evidence=v, rank=1, kind="one rho")
    f.note = f"rho = {rho:+.3f} over {len(rhos)} values"
    return f
