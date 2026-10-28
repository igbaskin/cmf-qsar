"""Gaussian-process classification by the Laplace approximation.

The support-vector classifiers of this program return a decision value and nothing else: to get
a probability out of one, a sigmoid has to be fitted to it afterwards, and there is no posterior
at all.  A Gaussian process puts a prior on the latent function instead and gives three things
the hinge loss cannot:

* **calibrated probabilities**, from the likelihood rather than from a sigmoid fitted afterwards;
* **a posterior variance per molecule**, so "active" and "no idea" are different answers, which
  is what the confidence-based alignment of :mod:`cmfqsar.varalign` needs for a classification
  set;
* **the marginal likelihood**, which scores a kernel without cross-validation and so can choose
  the attenuation, the amplitude and the mixing coefficients from the training molecules alone.

A Bernoulli likelihood is not conjugate to the Gaussian prior, so the posterior is approximated
by a Gaussian at its mode (Rasmussen and Williams, *Gaussian Processes for Machine Learning*,
chapter 3; the Newton iteration is their Algorithm 3.1, the predictions 3.2 and the marginal
likelihood 3.3).  Everything here works from a precomputed kernel, which is what the CMF
machinery provides, and nothing in it looks at a molecule.

The latent mean stays a kernel expansion, ``f(x) = sum_j a_j K(x, x_j)``, so the co-field and the
atom contributions remain exactly as well defined as they are for kernel ridge regression: unlike
a nonlinear kernel layer, this classifier keeps the interpretation intact.

Measured on the six classification sets of `examples/`, the accuracy is equal to that of C-SVC; the
reason to use it is the three things above, not the Matthews correlation.

:func:`klr_fit` (plan item E9) is the same Laplace step read as a learner of its own: kernel
logistic regression is exactly this mode with the logistic likelihood, so it reuses the iteration
here instead of repeating it, and differs only in choosing its ridge on the folds rather than from
the evidence and in not computing the posterior variance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import expit, log_ndtr, ndtr

LIKELIHOODS = ("probit", "logit")
_LOG_2PI = float(np.log(2.0 * np.pi))


# --------------------------------------------------------------------------- #
# Likelihoods: log p(y | f), its gradient, and W = -d2/df2 (diagonal, positive)
# --------------------------------------------------------------------------- #

def _probit(f: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Probit: p(y = +1 | f) = Phi(f).  The ratio phi/Phi is taken through its logarithm, which
    is what keeps a badly misclassified molecule (Phi underflowing) from producing a nan."""
    z = y * f
    log_phi = -0.5 * z * z - 0.5 * _LOG_2PI
    log_Phi = log_ndtr(z)
    r = np.exp(log_phi - log_Phi)
    return float(np.sum(log_Phi)), y * r, r * r + z * r


def _logit(f: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Logistic: p(y = +1 | f) = 1 / (1 + exp(-f))."""
    z = y * f
    pi = expit(f)
    t = 0.5 * (y + 1.0)
    return float(-np.sum(np.logaddexp(0.0, -z))), t - pi, pi * (1.0 - pi)


def _terms(likelihood: str, f: np.ndarray, y: np.ndarray):
    if likelihood == "probit":
        return _probit(f, y)
    if likelihood == "logit":
        return _logit(f, y)
    raise ValueError(f"unknown likelihood {likelihood!r}; use one of {LIKELIHOODS}")


def squash(mean: np.ndarray, var: np.ndarray, likelihood: str = "probit") -> np.ndarray:
    """p(active) from the latent mean and variance, averaged over the latent posterior.

    For the probit likelihood the average is exact, ``Phi(mean / sqrt(1 + var))``; for the
    logistic one it is MacKay's approximation.  Either way the variance pulls the probability
    towards one half, which is the purpose: an uncertain molecule should not be called confidently.
    """
    var = np.maximum(np.asarray(var, float), 0.0)
    if likelihood == "probit":
        return ndtr(np.asarray(mean, float) / np.sqrt(1.0 + var))
    if likelihood == "logit":
        return expit(np.asarray(mean, float) / np.sqrt(1.0 + np.pi * var / 8.0))
    raise ValueError(f"unknown likelihood {likelihood!r}; use one of {LIKELIHOODS}")


# --------------------------------------------------------------------------- #
# The classifier
# --------------------------------------------------------------------------- #

@dataclass
class LaplaceGPC:
    """Binary GP classifier on a precomputed kernel, fitted at the posterior mode.

    `amplitude` multiplies the kernel and is the only hyper-parameter: a large one lets the
    latent function move far from zero, which is what makes the fit confident, and it plays the
    role C plays for a support-vector classifier.  Unlike C it can be chosen by the marginal
    likelihood, without cross-validation; see :func:`select_by_evidence`.
    """
    likelihood: str = "probit"
    amplitude: float = 1.0
    jitter: float = 1e-8
    max_iter: int = 100
    tol: float = 1e-9
    #: with False the factorisation that is only needed afterwards is not made, which leaves the
    #: marginal likelihood and the posterior variance unavailable and the mode unchanged.  That is
    #: what :func:`klr_fit` wants: kernel logistic regression is this mode and nothing else.
    evidence: bool = True

    # fitted quantities
    y_: Optional[np.ndarray] = None
    f_: Optional[np.ndarray] = None              # latent mode
    grad_: Optional[np.ndarray] = None           # d log p(y|f) / df at the mode
    sqrt_W_: Optional[np.ndarray] = None
    L_: Optional[np.ndarray] = None              # chol(I + W^1/2 K W^1/2)
    log_marginal_likelihood: float = float("nan")
    n_iter: int = 0
    converged: bool = False

    # -------------------------------------------------------------- fitting
    def fit(self, K: np.ndarray, y: np.ndarray) -> "LaplaceGPC":
        """Newton iteration on the posterior of the latent values (Algorithm 3.1)."""
        y = np.where(np.asarray(y, float) > 0, 1.0, -1.0)
        if len(np.unique(y)) < 2:
            raise ValueError("binary classification needs molecules of both classes")
        n = len(y)
        K = self.amplitude * np.asarray(K, float)
        if self.jitter:
            K = K + self.jitter * np.eye(n)
        f = np.zeros(n)
        obj = -np.inf
        for it in range(1, self.max_iter + 1):
            loglik, grad, W = _terms(self.likelihood, f, y)
            W = np.maximum(W, 1e-12)
            sW = np.sqrt(W)
            B = np.eye(n) + sW[:, None] * K * sW[None, :]
            L = np.linalg.cholesky(B)
            b = W * f + grad
            a = b - sW * cho_solve((L, True), sW * (K @ b))
            f = K @ a
            new = -0.5 * float(a @ f) + _terms(self.likelihood, f, y)[0]
            self.n_iter = it
            if abs(new - obj) < self.tol * max(1.0, abs(new)):
                obj = new
                self.converged = True
                break
            obj = new
        loglik, grad, W = _terms(self.likelihood, f, y)
        W = np.maximum(W, 1e-12)
        sW = np.sqrt(W)
        if not self.evidence:
            # the mode is all that was asked for.  At the mode f = K grad, so the dual
            # coefficients are grad itself and no factorisation is needed to read them off; the
            # variance and the marginal likelihood stay unavailable, which is what they cost.
            self.y_, self.f_, self.grad_, self.sqrt_W_, self.L_ = y, f, grad, sW, None
            return self
        B = np.eye(n) + sW[:, None] * K * sW[None, :]
        L = np.linalg.cholesky(B)
        a = _dual_from_mode(K, f, sW, L, W, grad)
        self.y_, self.f_, self.grad_, self.sqrt_W_, self.L_ = y, f, grad, sW, L
        self.log_marginal_likelihood = (-0.5 * float(a @ f) + loglik
                                        - float(np.sum(np.log(np.diag(L)))))
        return self

    # ----------------------------------------------------------- predictions
    @property
    def coef(self) -> np.ndarray:
        """Dual coefficients of the latent mean, with the amplitude folded in.

        The latent mean of a new molecule is ``k(x, X) @ coef``, so the rest of the program can
        treat this model exactly like kernel ridge regression: same decomposition rules, same
        co-field, same atom contributions.
        """
        if self.grad_ is None:
            raise RuntimeError("the classifier is not fitted")
        return self.amplitude * self.grad_

    def latent(self, K_test_train: np.ndarray) -> np.ndarray:
        return np.asarray(K_test_train, float) @ self.coef

    def latent_variance(self, K_test_train: np.ndarray, k_test_self: np.ndarray) -> np.ndarray:
        """Posterior variance of the latent function (Algorithm 3.2)."""
        if self.L_ is None:
            if self.grad_ is not None:
                raise RuntimeError("this fit was made with evidence=False, which skips the "
                                   "factorisation the posterior variance is read from; fit again "
                                   "with evidence=True, or use the latent mean alone")
            raise RuntimeError("the classifier is not fitted")
        Ks = self.amplitude * np.asarray(K_test_train, float)
        v = solve_triangular(self.L_, self.sqrt_W_[:, None] * Ks.T, lower=True)
        var = self.amplitude * np.asarray(k_test_self, float) - np.sum(v * v, axis=0)
        return np.maximum(var, 0.0)

    def predict_proba(self, K_test_train: np.ndarray,
                      k_test_self: Optional[np.ndarray] = None) -> np.ndarray:
        """p(active) for new molecules, averaged over the latent posterior where it is known."""
        mean = self.latent(K_test_train)
        if k_test_self is None:
            return squash(mean, np.zeros_like(mean), self.likelihood)
        return squash(mean, self.latent_variance(K_test_train, k_test_self), self.likelihood)


def _dual_from_mode(K: np.ndarray, f: np.ndarray, sW: np.ndarray, L: np.ndarray,
                    W: np.ndarray, grad: np.ndarray) -> np.ndarray:
    """``a = K^-1 f`` at the mode, taken the stable way: a = W f + grad - ... = grad, in exact
    arithmetic, because the mode satisfies f = K grad.  Recomputing it from the factorisation
    keeps the marginal likelihood correct when the Newton iteration stopped early."""
    b = W * f + grad
    return b - sW * cho_solve((L, True), sW * (K @ b))


def klr_fit(K: np.ndarray, y: np.ndarray, lam: float, max_iter: int = 100,
            tol: float = 1e-9) -> Tuple[np.ndarray, float, "LaplaceGPC"]:
    """Kernel logistic regression (plan item E9): the probabilistic classifier that is linear
    in the field.

    It minimises the logistic loss of the latent function plus a ridge on its norm in the
    reproducing-kernel Hilbert space,

        sum_i log(1 + exp(-y_i f(x_i)))  +  (lam / 2) ||f||^2,      f(x) = sum_j a_j K(x, x_j),

    which is a smooth convex problem solved by Newton's method.  **It is the same problem the
    Laplace step of :class:`LaplaceGPC` already solves**, because that step finds the mode of
    ``log p(y | f) - f' (s K)^-1 f / 2`` with the logistic likelihood, and the two agree when the
    amplitude is ``s = 1 / lam``.  So this function runs that step rather than a second Newton
    iteration of its own; ``tests/test_klr.py`` checks that the two give the same coefficients.

    Two things differ from `gpc`, and they are the whole of the difference:

    * **lam is chosen by cross-validation, not by the marginal likelihood.**  The Gaussian process
      picks its amplitude from the evidence, which needs no held-out molecules; this learner picks
      lam the way kernel ridge regression picks its own, on the folds.
    * **there is no posterior variance.**  The probability is the likelihood at the mode,
      ``p(active) = 1 / (1 + exp(-f))``, not the likelihood averaged over a posterior, so a
      molecule unlike anything in the training set is *not* pulled towards one half.  That is what
      makes the fit cheaper -- the factorisation the variance is read from is never made -- and it
      is also what the Gaussian process gives that this does not.

    The decision boundary is ``f = 0``, which is ``p(active) = 0.5``: the class the model reports
    and its probability are the same statement, unlike a discriminant whose threshold is placed
    separately.  On a set with far more inactives than actives that boundary follows the larger
    class, and moving it is then worth considering; `cmfqsar.model` takes a ``klr_threshold``
    parameter for that and leaves it off by default.

    Returns (dual coefficients, bias, the fit), the bias being zero.
    """
    lam = float(lam)
    if not lam > 0.0:
        raise ValueError("kernel logistic regression needs a positive lam (the ridge on the "
                         "norm of the latent function)")
    g = LaplaceGPC(likelihood="logit", amplitude=1.0 / lam, max_iter=int(max_iter),
                   tol=float(tol), evidence=False).fit(K, y)
    return g.coef, 0.0, g


def evidence(K: np.ndarray, y: np.ndarray, amplitude: float = 1.0,
             likelihood: str = "probit") -> float:
    """Approximate log marginal likelihood of a kernel, with no cross-validation anywhere."""
    return LaplaceGPC(likelihood=likelihood, amplitude=amplitude).fit(K, y).log_marginal_likelihood


def select_by_evidence(kernels: Dict[object, np.ndarray], y: np.ndarray,
                       amplitudes: Sequence[float] = (0.3, 1.0, 3.0, 10.0, 30.0),
                       likelihood: str = "probit") -> Tuple[object, float, float]:
    """Pick the kernel and the amplitude with the largest marginal likelihood.

    `kernels` is keyed by whatever labels them, usually the attenuation.  Returns
    (key, amplitude, log marginal likelihood).  This is the cross-validation-free alternative to
    the grid search: every candidate is fitted once, on all the molecules.
    """
    best = (None, float("nan"), -np.inf)
    for key, K in kernels.items():
        for s in amplitudes:
            try:
                lml = evidence(K, y, amplitude=float(s), likelihood=likelihood)
            except (np.linalg.LinAlgError, ValueError):
                continue
            if np.isfinite(lml) and lml > best[2]:
                best = (key, float(s), float(lml))
    if best[0] is None:
        raise ValueError("no candidate kernel could be fitted")
    return best


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of the probabilities; 0 is perfect, 0.25 is no better than chance."""
    t = (np.asarray(y, float) > 0).astype(float)
    return float(np.mean((np.asarray(p, float) - t) ** 2))


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5):
    """Rows of (bin centre, mean predicted probability, observed fraction, count)."""
    t = (np.asarray(y, float) > 0).astype(float)
    p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= 1.0)
        if m.any():
            out.append((0.5 * (lo + hi), float(p[m].mean()), float(t[m].mean()), int(m.sum())))
    return out
