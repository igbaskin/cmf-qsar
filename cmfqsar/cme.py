"""
The expected field of a molecule with a given activity: the conditional mean embedding.

A CMF field is the kernel mean embedding of a molecule's weighted atoms (manual: "CMF and kernel
mean embeddings"), so a set of molecules has a mean field, and a set of molecules *with a given
activity* has one too. The conditional mean embedding (Song et al., ICML 2009) estimates it from
the training set as a weighted sum of training fields::

    mu(y*) = sum_i beta_i(y*) x_i,      beta(y*) = (L + n lam I)^(-1) l(y*)

where L is a kernel on the activities of the n training molecules and l(y*) its column at the
activity asked for. The weights depend on the activities only; the fields enter as the basis.
So the expected field at an activity is a map on the same grid and in the same basis as the
co-field, and it can be drawn, compared and exported in the same way.

**Two kernels on the activity.**

* ``linear``: L = 1 + u u^T on the standardised activity u. Then the expected field is the mean
  field plus a slope times (u* - mean u), the straight-line regression of the field at every
  point on the activity. The slope ``sum_i u_i x_i / n`` is the field weighted by the centred
  activity: the covariance of the field with the activity. Its squared norm is the HSIC
  statistic of :mod:`cmfqsar.hsic`. In the terms of Haufe et al. (NeuroImage 87, 96-110, 2014)
  it is a forward model, or *pattern*: the field written as a function of the activity. The
  co-field is a backward model, or *filter*: the weights that turn a field into an activity.
  They showed that a filter is not a map of where the signal lies, and a pattern is.
* ``gauss``: L_ij = exp(-gamma (u_i - u_j)^2). The expected field at u* is then a smoothed
  average of the fields of molecules with activities near u*; it can describe an optimum
  (the field at the best activity lies between those of weaker ones), which a straight line
  cannot.

**Choosing gamma and lam.** The conditional mean embedding is a regression whose output lives in
the field space, so it has a loss there (Grunewalder et al., ICML 2012): the squared distance in
the field space between a left-out molecule's field and the expected field at its activity,

    || x_j - mu(y_j) ||^2 = K_jj - 2 beta(y_j)^T K_{.j} + beta(y_j)^T K beta(y_j),

computed from the field kernel K alone. It is cross-validated over the molecules and minimised
over a grid. Divided by the same loss of the plain mean field, it gives the out-of-fold share of
the field variance that the activity explains -- the inverse question to q2.

What this module does not claim: the expected field is an average of the training fields, not
the field of a molecule. Where the asked activity lies outside the training range the Gaussian
weights fade, and the expected field falls back to the mean field; :class:`ExpectedField`
reports how much of it is the mean field, so that this is visible. The straight line instead
extrapolates, as far as it is asked to.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

KERNELS_Y = ("linear", "gauss")
#: widths on the standardised activity; exp(-gamma du^2) halves at du = 0.83 / sqrt(gamma)
GAMMAS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
#: ridges, in units of n (the n lam I of the estimator)
LAMS = (1e-4, 1e-3, 1e-2, 0.1, 1.0)


# --------------------------------------------------------------------------- the weights

@dataclass
class Standardiser:
    """u = (y - mean) / sd, fitted on the training activities."""
    mean: float
    sd: float

    @classmethod
    def fit(cls, y) -> "Standardiser":
        y = np.asarray(y, float)
        sd = float(np.std(y))
        return cls(float(np.mean(y)), sd if sd > 0 else 1.0)

    def __call__(self, y) -> np.ndarray:
        return (np.asarray(y, float) - self.mean) / self.sd


def activity_kernel(u_a, u_b, kernel_y: str = "gauss", gamma: float = 1.0) -> np.ndarray:
    """The kernel between standardised activities: rows `u_a`, columns `u_b`."""
    a = np.atleast_1d(np.asarray(u_a, float))[:, None]
    b = np.atleast_1d(np.asarray(u_b, float))[None, :]
    if kernel_y == "linear":
        return 1.0 + a * b
    if kernel_y == "gauss":
        return np.exp(-float(gamma) * (a - b) ** 2)
    raise ValueError(f"unknown kernel on the activity {kernel_y!r}; use one of {KERNELS_Y}")


def centre_weights(beta: np.ndarray) -> np.ndarray:
    """Give the weight a column lacks to the mean field: beta + (1 - sum beta) / n.

    The plain estimator shrinks the expected field towards zero -- a large ridge, or an activity
    far from every training activity, gives weights that sum to less than one. Estimating the
    expected *departure from the mean field* instead, ``mean + sum_i beta_i (x_i - mean)``, is
    the same as this shift of the weights, and it makes the shrinkage go to the mean field.
    """
    beta = np.asarray(beta, float)
    return beta + (1.0 - beta.sum(axis=0, keepdims=True)) / beta.shape[0]


class _Solver:
    """(L + n lam I)^(-1) for one set of training activities and many ridges, by one eigh."""

    def __init__(self, u_train, kernel_y: str, gamma: float):
        self.u = np.asarray(u_train, float)
        self.kernel_y, self.gamma = kernel_y, float(gamma)
        L = activity_kernel(self.u, self.u, kernel_y, gamma)
        if kernel_y == "linear":
            # the constant of 1 + u u^T is what centring supplies; keeping it would count twice
            L = L - 1.0
        self.d, self.V = np.linalg.eigh(L)

    def weights(self, u_query, lam: float, centre: bool = True) -> np.ndarray:
        n = len(self.u)
        rhs = activity_kernel(self.u, u_query, self.kernel_y, self.gamma)
        if self.kernel_y == "linear":
            rhs = rhs - 1.0
        beta = self.V @ ((self.V.T @ rhs) / (np.maximum(self.d, 0.0) + n * float(lam))[:, None])
        return centre_weights(beta) if centre else beta


def cme_weights(u_train, u_query, kernel_y: str = "gauss", gamma: float = 1.0,
                lam: float = 1e-2, centre: bool = True) -> np.ndarray:
    """(n_train, n_query): beta(u*) = (L + n lam I)^(-1) l(u*) for every activity asked for.

    With `centre` (the default) the weights are shifted by :func:`centre_weights`, so that they
    sum to one and the expected field falls back to the mean field where the data say nothing.
    With the linear kernel the centred weights are ``1/n + u_i u* / (n (1 + lam))``: the mean
    field plus the straight-line slope of the field on the activity.
    """
    return _Solver(u_train, kernel_y, gamma).weights(u_query, lam, centre)


def field_loss(K: np.ndarray, test: np.ndarray, train: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """|| x_j - sum_i beta_ij x_i ||^2 for every molecule j of `test`, from the field kernel.

    `beta` is (len(train), len(test)): one column of weights per test molecule.
    """
    Ktt = K[np.ix_(train, train)]
    Ktj = K[np.ix_(train, test)]
    return (np.diag(K)[test] - 2.0 * np.sum(beta * Ktj, axis=0)
            + np.sum(beta * (Ktt @ beta), axis=0))


def _folds(n: int, folds: int, seed: int) -> List[np.ndarray]:
    order = np.random.RandomState(seed).permutation(n)
    return [order[k::folds] for k in range(min(folds, n))]


@dataclass
class CMEFit:
    """The conditional mean embedding chosen by its cross-validated loss in the field space."""
    kernel_y: str
    gamma: float
    lam: float
    y: np.ndarray
    standardiser: Standardiser
    loss: float                     # out-of-fold loss per molecule at the chosen setting
    loss_mean: float                # the same loss of the plain mean field
    table: Dict[Tuple[float, float], float] = field(default_factory=dict)
    at_edge: List[str] = field(default_factory=list)

    @property
    def explained(self) -> float:
        """The out-of-fold share of the field variance that the activity explains."""
        return 1.0 - self.loss / self.loss_mean if self.loss_mean > 0 else float("nan")

    def weights(self, y_query, centre: bool = True) -> np.ndarray:
        """(n_train, n_query) weights of the training fields at the activities asked for."""
        return cme_weights(self.standardiser(self.y), self.standardiser(y_query),
                           self.kernel_y, self.gamma, self.lam, centre)

    def describe(self) -> str:
        what = ("a straight line in the activity" if self.kernel_y == "linear"
                else f"a Gaussian kernel on the activity, gamma {self.gamma:g}")
        s = (f"{what}, ridge {self.lam:g}: the activity explains {100 * self.explained:.1f} % "
             f"of the field variance out of fold")
        if self.at_edge:
            s += f" (chosen at the edge of the grid of {', '.join(self.at_edge)})"
        return s


def fit_cme(K: np.ndarray, y, kernel_y: str = "gauss", gammas: Sequence[float] = GAMMAS,
            lams: Sequence[float] = LAMS, folds: int = 5, seeds: Sequence[int] = (0, 1, 2)
            ) -> CMEFit:
    """Choose gamma (for ``gauss``) and lam by the out-of-fold loss in the field space.

    The activities are standardised inside every training part, so the left-out molecules do
    not set the scale of their own weights.
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    if kernel_y not in KERNELS_Y:
        raise ValueError(f"unknown kernel on the activity {kernel_y!r}; use one of {KERNELS_Y}")
    if n < 4:
        raise ValueError(f"{n} molecules: too few to cross-validate a conditional mean embedding")
    gammas = tuple(gammas) if kernel_y == "gauss" else (1.0,)
    splits = [(np.setdiff1d(np.arange(n), te), te) for s in seeds for te in _folds(n, folds, s)]
    table: Dict[Tuple[float, float], float] = {(float(g), float(l)): 0.0
                                               for g in gammas for l in lams}
    for tr, te in splits:
        st = Standardiser.fit(y[tr])
        for gm in gammas:
            solver = _Solver(st(y[tr]), kernel_y, gm)
            for lam in lams:
                beta = solver.weights(st(y[te]), lam)
                table[(float(gm), float(lam))] += float(np.sum(field_loss(K, te, tr, beta)))
    table = {k: v / (n * len(seeds)) for k, v in table.items()}
    tot = 0.0
    for tr, te in splits:
        beta = np.full((len(tr), len(te)), 1.0 / len(tr))
        tot += float(np.sum(field_loss(K, te, tr, beta)))
    loss_mean = tot / (n * len(seeds))
    (gm, lam), loss = min(table.items(), key=lambda kv: kv[1])
    edge = []
    if kernel_y == "gauss" and len(gammas) > 1 and gm in (min(gammas), max(gammas)):
        edge.append("gamma")
    if len(lams) > 1 and lam in (min(lams), max(lams)):
        # the smallest ridge is a fine place to stop when the loss no longer moves there
        if not (lam == min(lams) and abs(table[(gm, sorted(lams)[1])] - loss) <= 1e-3 * loss):
            edge.append("lam")
    return CMEFit(kernel_y=kernel_y, gamma=float(gm), lam=float(lam), y=y.copy(),
                  standardiser=Standardiser.fit(y), loss=float(loss), loss_mean=float(loss_mean),
                  table=table, at_edge=edge)


# --------------------------------------------------------------------------- the field

@dataclass
class ExpectedField:
    """The weights of the expected field at one activity, and what they say about it."""
    activity: float
    weights: np.ndarray             # (n_train,), centred: they sum to one
    from_mean: float = 0.0          # 1 - the sum of the raw weights

    # ``from_mean`` is the share of the weight that only restores the mean field. Near 0 the
    # expected field is made of molecules with activities near the one asked for; near 1 the
    # training set says little about that activity (it lies outside the range, or the ridge
    # is large) and the expected field is close to the mean field. With the straight line it
    # is 1 - 1 = 0 by construction: a line always has an answer.

    @property
    def effective_size(self) -> float:
        """How many molecules the expected field is an average of, (sum |b|)^2 / sum b^2."""
        a = np.abs(self.weights)
        s2 = float(np.sum(a * a))
        return float(np.sum(a) ** 2 / s2) if s2 > 0 else 0.0

    def leading(self, names: Sequence[str], k: int = 5) -> List[Tuple[str, float]]:
        """The `k` training molecules with the largest weights."""
        order = np.argsort(-self.weights)[:k]
        return [(str(names[i]), float(self.weights[i])) for i in order]

    def on(self, X: np.ndarray) -> np.ndarray:
        """The expected field on points, from the (n_train, n_points) training fields."""
        return self.weights @ np.asarray(X, float)


def expected_fields(fit: CMEFit, activities: Sequence[float]) -> List[ExpectedField]:
    raw = fit.weights(np.asarray(activities, float), centre=False)
    B = centre_weights(raw)
    lack = (np.clip(1.0 - raw.sum(axis=0), 0.0, 1.0) if fit.kernel_y == "gauss"
            else np.zeros(raw.shape[1]))
    return [ExpectedField(float(a), B[:, k], float(lack[k])) for k, a in enumerate(activities)]


def pattern(y, X: np.ndarray) -> np.ndarray:
    """The covariance of the field with the standardised activity at every point.

    ``sum_i u_i x_i(r) / n``: the slope of the linear conditional mean embedding, a forward
    model in the sense of Haufe et al. (2014). It needs no ridge and no choice of width.
    """
    u = Standardiser.fit(y)(y)
    return u @ np.asarray(X, float) / len(u)


def distance_to(beta: np.ndarray, K_train: np.ndarray, K_test_train: np.ndarray,
                k_test_self: np.ndarray) -> np.ndarray:
    """|| x - mu ||^2 in the field space for every test molecule, mu = sum_i beta_i x_i."""
    beta = np.asarray(beta, float)
    return (np.asarray(k_test_self, float) - 2.0 * np.asarray(K_test_train, float) @ beta
            + float(beta @ np.asarray(K_train, float) @ beta))
