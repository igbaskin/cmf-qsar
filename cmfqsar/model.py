"""
CMF regression models: kernel ridge regression (KRR, the method of the 2013
paper), support vector regression and a Gaussian-process view of KRR that
provides predictive variances.  Includes k-fold cross-validation, grid search
over hyper-parameters (alpha, lambda / C, optionally mixing coefficients h_f)
and external (nested) cross-validation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet, FieldSpec, MoleculeFields
from .kernels import (Cancelled, atom_kernel_block, combine_kernels, kernel_matrices,
                      normalize_kernels, self_kernels)
from .neldermead import interp_terms, nmmin

ProgressFn = Callable[[str, float], None]   # (message, fraction 0..1)
CancelFn = Callable[[], bool]


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def r2_score(y: np.ndarray, yp: np.ndarray, y_ref_mean: Optional[float] = None) -> float:
    y = np.asarray(y, float)
    yp = np.asarray(yp, float)
    m = np.mean(y) if y_ref_mean is None else y_ref_mean
    ss = np.sum((y - m) ** 2)
    if ss <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - yp) ** 2) / ss)


def rmse(y: np.ndarray, yp: np.ndarray) -> float:
    y = np.asarray(y, float)
    yp = np.asarray(yp, float)
    return float(np.sqrt(np.mean((y - yp) ** 2)))


@dataclass
class ModelStats:
    n_train: int = 0
    r2: float = float("nan")          # fit, training set
    rmse_fit: float = float("nan")
    q2: float = float("nan")          # internal cross-validation
    rmse_cv: float = float("nan")
    n_test: int = 0
    r2_pred: float = float("nan")     # external test set (variance of the test set)
    r2_pred_train_mean: float = float("nan")  # external test set (deviations from the training mean)
    r2_pred_train_var: float = float("nan")   # 1 - [RSS/(m-1)] / var(y_train)  (regr_param_ex of the R scripts)
    rmse_pred: float = float("nan")
    q2_ex: float = float("nan")       # external (nested) cross-validation
    rmse_ex: float = float("nan")
    q2_ex_sd: float = float("nan")
    rmse_ex_sd: float = float("nan")
    q2_ex_aggr: float = float("nan")  # external CV with reshuffles: statistics of the mean prediction
    rmse_ex_aggr: float = float("nan")
    cv_folds: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        if self.n_train:
            lines.append(f"Training set: n = {self.n_train}")
            lines.append(f"  r2 (fit)        = {self.r2:.3f}    RMSE (fit)  = {self.rmse_fit:.3f}")
        if np.isfinite(self.q2):
            lines.append(f"  q2 ({self.cv_folds}-fold CV)  = {self.q2:.3f}    RMSEcv      = {self.rmse_cv:.3f}")
        if self.n_test:
            lines.append(f"Test set: m = {self.n_test}")
            lines.append(f"  R2pred          = {self.r2_pred:.3f}    RMSEpred    = {self.rmse_pred:.3f}")
            lines.append(f"  R2pred (train mean) = {self.r2_pred_train_mean:.3f}    "
                         f"R2pred (train variance) = {self.r2_pred_train_var:.3f}")
        if np.isfinite(self.q2_ex):
            sd = f" +/- {self.q2_ex_sd:.3f}" if np.isfinite(self.q2_ex_sd) else ""
            sd2 = f" +/- {self.rmse_ex_sd:.3f}" if np.isfinite(self.rmse_ex_sd) else ""
            lines.append(f"External CV: q2ex = {self.q2_ex:.3f}{sd}    RMSEcvex = {self.rmse_ex:.3f}{sd2}")
            if np.isfinite(getattr(self, "q2_ex_aggr", float("nan"))):
                lines.append(f"  mean prediction over reshuffles: q2ex = {self.q2_ex_aggr:.3f}    "
                             f"RMSEcvex = {self.rmse_ex_aggr:.3f}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Classification statistics
# --------------------------------------------------------------------------- #

def classification_scores(y_true: np.ndarray, decision: np.ndarray) -> Dict[str, float]:
    """Metrics of a binary classifier from its decision values.

    `y_true` holds +1 for the positive (active) class and -1 for the negative one;
    `decision` is the signed decision function, whose sign is the predicted class.
    """
    y = np.asarray(y_true, dtype=float)
    d = np.asarray(decision, dtype=float)
    ok = np.isfinite(y) & np.isfinite(d)
    y, d = y[ok], d[ok]
    pred = np.where(d >= 0, 1.0, -1.0)
    tp = float(np.sum((y > 0) & (pred > 0)))
    tn = float(np.sum((y < 0) & (pred < 0)))
    fp = float(np.sum((y < 0) & (pred > 0)))
    fn = float(np.sum((y > 0) & (pred < 0)))
    n = tp + tn + fp + fn
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    auc = float("nan")
    if (y > 0).any() and (y < 0).any():
        try:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score((y > 0).astype(int), d))
        except Exception:                                   # noqa: BLE001
            auc = float("nan")
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / n if n else float("nan"),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "sensitivity": sens, "specificity": spec,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "mcc": (tp * tn - fp * fn) / den if den > 0 else float("nan"),
        "auc": auc,
    }


def rie_bedroc(ranks: np.ndarray, n: int, n_act: int, alpha: float = 20.0) -> Tuple[float, float]:
    """Robust initial enhancement and BEDROC of Truchon and Bayly (2007).

    The enrichment factor counts the actives above a cut and throws away where they are; RIE
    weights every active by exp(-alpha r / N), so a hit at rank 1 is worth more than one at rank
    50, and BEDROC rescales RIE onto [0, 1] given the active rate.  `alpha` sets how early
    "early" is: 20 puts 80 % of the weight in the top 8 % of the ranking, 160.9 in the top 1 %.
    """
    N, n_act = float(n), float(n_act)
    if n_act <= 0 or n_act >= N or alpha <= 0:
        return float("nan"), float("nan")
    ra = n_act / N
    x = float(np.sum(np.exp(-alpha * np.asarray(ranks, float) / N)))
    denom = (n_act / N) * (1.0 - np.exp(-alpha)) / (np.exp(alpha / N) - 1.0)
    rie = x / denom if denom > 0 else float("nan")
    if not np.isfinite(rie):
        return rie, float("nan")
    scale = ra * np.sinh(alpha / 2.0) / (np.cosh(alpha / 2.0) - np.cosh(alpha / 2.0 - alpha * ra))
    bedroc = rie * scale + 1.0 / (1.0 - np.exp(alpha * (1.0 - ra)))
    return float(rie), float(bedroc)


def screening_scores(y_true: np.ndarray, decision: np.ndarray,
                     fractions: Sequence[float] = (0.005, 0.01, 0.05, 0.10),
                     alphas: Sequence[float] = (20.0, 160.9)) -> Dict[str, float]:
    """Virtual-screening metrics of a ranking: AUC and enrichment at the top of the list.

    `y_true` is +1 for the actives and -1 for the background (decoys), `decision` the score the
    library is ranked by.  The enrichment factor at a fraction f is the share of actives found in
    the top f of the ranking divided by f, so 1.0 is random and 1/(active rate) the maximum.

    A screening result is a property of the background as much as of the model: decoys that differ
    grossly from the actives (in size, charge or shape) inflate every number here, which is what
    property-matched sets such as DUD-E, DEKOIS or MUV exist to avoid.
    """
    y = np.asarray(y_true, dtype=float)
    d = np.asarray(decision, dtype=float)
    ok = np.isfinite(y) & np.isfinite(d)
    y, d = y[ok], d[ok]
    n, n_act = len(y), int(np.sum(y > 0))
    out: Dict[str, float] = {"n": float(n), "n_actives": float(n_act), "n_decoys": float(n - n_act)}
    if n_act == 0 or n_act == n:
        out["auc"] = float("nan")
        return out
    rate = n_act / n
    out["active_rate"] = rate
    out["max_ef"] = 1.0 / rate
    order = np.argsort(-d)
    hits = np.cumsum(y[order] > 0)
    for f in fractions:
        k = max(1, int(round(f * n)))
        tag = f"{f * 100:g}".replace(".", "_")
        out[f"ef_{tag}"] = float((hits[k - 1] / k) / rate)
        out[f"hits_{tag}"] = float(hits[k - 1])
        # the ceiling: at most min(k, n_act) actives can sit in a slice of k molecules, so a
        # small fraction saturates long before the model is perfect
        out[f"max_ef_{tag}"] = float(min(k, n_act) / (k * rate))
    # where in the ranking the actives sit, not merely how many are above a cut: the enrichment
    # factor at a small fraction saturates as soon as that slice is full
    ranks = np.flatnonzero(y[order] > 0) + 1
    for a in alphas:
        rie, bedroc = rie_bedroc(ranks, n, n_act, alpha=float(a))
        tag = f"{a:g}".replace(".", "_")
        out[f"rie_{tag}"] = rie
        out[f"bedroc_{tag}"] = bedroc
    try:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score((y > 0).astype(int), d))
    except Exception:                                       # noqa: BLE001
        out["auc"] = float("nan")
    return out


def enrichment_curve(y_true: np.ndarray, decision: np.ndarray):
    """(fraction of the library screened, fraction of the actives found) of a ranking."""
    y = np.asarray(y_true, dtype=float)
    d = np.asarray(decision, dtype=float)
    ok = np.isfinite(y) & np.isfinite(d)
    y, d = y[ok], d[ok]
    n_act = int(np.sum(y > 0))
    if n_act == 0 or n_act == len(y):
        return None
    order = np.argsort(-d)
    found = np.cumsum(y[order] > 0) / n_act
    screened = np.arange(1, len(y) + 1) / len(y)
    return np.concatenate([[0.0], screened]), np.concatenate([[0.0], found])


@dataclass
class ClassStats:
    """Statistics of a classification (binary) or one-class model."""
    task: str = "classification"
    n_train: int = 0
    n_test: int = 0
    fit: Dict[str, float] = field(default_factory=dict)
    cv: Dict[str, float] = field(default_factory=dict)
    test: Dict[str, float] = field(default_factory=dict)
    cv_folds: int = 0
    inlier_fraction: float = float("nan")      # one-class: training molecules inside the boundary
    n_support: int = 0
    screening: Dict[str, float] = field(default_factory=dict)   # actives against a decoy background
    extra: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def _line(tag: str, s: Dict[str, float]) -> str:
        if not s:
            return ""
        return (f"  {tag:16s} accuracy {s['accuracy']:.3f}  balanced {s['balanced_accuracy']:.3f}  "
                f"sens {s['sensitivity']:.3f}  spec {s['specificity']:.3f}  "
                f"MCC {s['mcc']:.3f}  AUC {s['auc']:.3f}")

    def _screening_lines(self) -> List[str]:
        s = self.screening
        if not s or not np.isfinite(s.get("auc", float("nan"))):
            return []
        out = [f"Virtual screening: {int(s['n_actives'])} actives against {int(s['n_decoys'])} decoys",
               f"  AUC {s['auc']:.3f}   (enrichment factor 1.0 = random, "
               f"{s.get('max_ef', float('nan')):.1f} = perfect)"]
        parts = []
        for tag, pct in (("0_5", "0.5"), ("1", "1"), ("5", "5"), ("10", "10")):
            if f"ef_{tag}" in s:
                ceiling = s.get(f"max_ef_{tag}", float("nan"))
                parts.append(f"top {pct} %: EF {s[f'ef_{tag}']:.2f} of {ceiling:.1f} "
                             f"({int(s[f'hits_{tag}'])} actives)")
        if parts:
            out.append("  " + ";  ".join(parts))
        if "bedroc_20" in s:
            # the enrichment factor stops at its ceiling as soon as the slice is full, and says
            # nothing about where in the slice the actives are; BEDROC keeps weighting the rank
            out.append(f"  BEDROC {s['bedroc_20']:.3f} (alpha 20, weight on the top 8 %), "
                       f"{s.get('bedroc_160_9', float('nan')):.3f} (alpha 160.9, the top 1 %); "
                       f"0 is the worst ranking, 1 the best")
        out.append("  a screening result depends on the background: decoys unlike the actives in size, "
                   "charge or shape inflate these numbers")
        return out

    def summary(self) -> str:
        lines = []
        if self.task == "oneclass":
            lines.append(f"One-class model: n = {self.n_train} target-class molecules")
            if np.isfinite(self.inlier_fraction):
                lines.append(f"  inside the boundary: {self.inlier_fraction * 100:.1f} % of the training set"
                             + (f", {self.n_support} support vectors" if self.n_support else ""))
            for tag, key in (("training set", "y_fit"), ("scored molecules", "y_test_pred")):
                d = np.asarray(self.extra.get(key, []), dtype=float)
                d = d[np.isfinite(d)]
                if len(d):
                    lines.append(f"  decision value, {tag} (n = {len(d)}): "
                                 f"min {d.min():.3f}, median {float(np.median(d)):.3f}, max {d.max():.3f}; "
                                 f"{int(np.sum(d >= 0))} inside the boundary")
            if self.test:
                lines.append(f"Scored set: m = {self.n_test} (labels used only for scoring, never for fitting)")
                lines.append(self._line("ranking", self.test))
            elif self.n_test:
                lines.append(f"Scored set: m = {self.n_test} molecules without class labels; "
                             f"the ranking is in the Scores tab")
            lines += self._screening_lines()
            return "\n".join(lines)
        if self.n_train:
            lines.append(f"Training set: n = {self.n_train}"
                         + (f"  ({int(self.fit.get('tp', 0) + self.fit.get('fn', 0))} active, "
                            f"{int(self.fit.get('tn', 0) + self.fit.get('fp', 0))} inactive)" if self.fit else ""))
            lines.append(self._line("fit", self.fit))
        if self.cv:
            lines.append(self._line(f"{self.cv_folds}-fold CV", self.cv))
        if self.test:
            lines.append(f"Test set: m = {self.n_test}")
            lines.append(self._line("prediction", self.test))
        lines += self._screening_lines()
        return "\n".join(x for x in lines if x)


# --------------------------------------------------------------------------- #
# Kernel-based learners on precomputed kernel matrices
# --------------------------------------------------------------------------- #

def krr_fit(K: np.ndarray, y: np.ndarray, lam: float, fit_bias: bool = True) -> Tuple[np.ndarray, float]:
    """Kernel ridge regression with a bias term, Eq. 12 of Baskin & Zhokhova (2013):

        [ K + lam I   1 ] [ a ]   [ y ]
        [ 1^T         0 ] [ b ] = [ 0 ]

    With ``fit_bias=False`` the plain KRR system (K + lam I) a = y is solved and b = 0
    (the ``set_b_0`` option of the original R scripts).
    """
    n = len(y)
    if not fit_bias:
        A = K + lam * np.eye(n)
        try:
            a = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            a = np.linalg.lstsq(A, y, rcond=None)[0]
        return a, 0.0
    A = np.zeros((n + 1, n + 1))
    A[:n, :n] = K + lam * np.eye(n)
    A[:n, n] = 1.0
    A[n, :n] = 1.0
    rhs = np.concatenate([y, [0.0]])
    try:
        sol = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(A, rhs, rcond=None)[0]
    return sol[:n], float(sol[n])


class _Learner:
    """Thin wrapper around the different kernel learners (dual coefficients + bias)."""

    def __init__(self, method: str, params: Dict[str, float], fit_bias: bool = True):
        self.method = method
        self.params = dict(params)
        self.fit_bias = fit_bias
        self.alpha_: Optional[np.ndarray] = None
        self.b_: float = 0.0
        self.support_: Optional[np.ndarray] = None   # indices of training molecules with a != 0
        self.K_train_: Optional[np.ndarray] = None
        self.gpc_ = None                             # the Laplace fit, for method "gpc"
        self.oc_ = None                              # the scorer, for the one-class methods

    def fit(self, K: np.ndarray, y: np.ndarray) -> "_Learner":
        n = len(y)
        if self.method in ("krr", "gp"):
            a, b = krr_fit(K, y, float(self.params.get("lam", 1.0)), self.fit_bias)
            self.alpha_, self.b_ = a, b
            self.support_ = np.arange(n)
            if self.method == "gp":
                self.K_train_ = K
        elif self.method in ("svr", "nusvr"):
            from sklearn.svm import SVR, NuSVR
            if self.method == "svr":
                est = SVR(kernel="precomputed", C=float(self.params.get("C", 1.0)),
                          epsilon=float(self.params.get("epsilon", 0.1)))
            else:
                est = NuSVR(kernel="precomputed", C=float(self.params.get("C", 1.0)),
                            nu=float(self.params.get("nu", 0.5)))
            est.fit(K, y)
            a = np.zeros(n)
            a[est.support_] = est.dual_coef_.ravel()
            self.alpha_, self.b_ = a, float(est.intercept_[0])
            self.support_ = np.array(est.support_)
        elif self.method in ("svc", "nusvc"):
            # binary classification; dual_coef_ already carries the class sign (y_i * alpha_i),
            # so the decision function is K a + b exactly as for regression
            from sklearn.svm import SVC, NuSVC
            if self.method == "svc":
                est = SVC(kernel="precomputed", C=float(self.params.get("C", 1.0)))
            else:
                est = NuSVC(kernel="precomputed", nu=float(self.params.get("nu", 0.5)))
            labels = np.where(np.asarray(y) > 0, 1, -1)
            if len(np.unique(labels)) < 2:
                raise ValueError("binary classification needs molecules of both classes")
            est.fit(K, labels)
            a = np.zeros(n)
            a[est.support_] = est.dual_coef_.ravel()
            # sklearn orders classes as [-1, +1]; dual_coef_ is signed for the *second* class
            self.alpha_, self.b_ = a, float(est.intercept_[0])
            self.support_ = np.array(est.support_)
            self.classes_ = est.classes_
        elif self.method == "gpc":
            # Gaussian-process classification at the posterior mode: the latent mean is a kernel
            # expansion like every other method here, so the decomposition rules still apply
            from .gpc import LaplaceGPC
            g = LaplaceGPC(likelihood=str(self.params.get("likelihood", "probit")),
                           amplitude=float(self.params.get("amplitude", 1.0)))
            self.gpc_ = g.fit(K, y)
            self.alpha_, self.b_ = g.coef, 0.0
            self.support_ = np.arange(n)
            self.K_train_ = K
        elif self.method in ("ocmean", "ocknn", "ocgp", "ockpca"):
            # the other one-class geometries: centroid, nearest neighbours, GP, principal
            # subspace.  Two of them are kernel expansions and keep the decomposition rules.
            from .oneclass import make
            sc = make(self.method, nu=float(self.params.get("nu", 0.1)),
                      k=int(self.params.get("k", 5)), lam=float(self.params.get("oc_lam", 0.1)),
                      components=int(self.params.get("components", 5)),
                      score=str(self.params.get("oc_score", "mean"))).fit(K)
            self.oc_ = sc
            self.support_ = np.arange(n)
            self.alpha_ = sc.coef if sc.is_linear else None
            self.b_ = sc.bias if sc.is_linear else 0.0
        elif self.method.startswith("fuse-"):
            # data fusion over the actives: a rule on the kernel row, nothing fitted
            from .fusion import make as make_fusion
            sc = make_fusion(self.method[len("fuse-"):],
                             nu=float(self.params.get("nu", 0.1)),
                             threshold=float(self.params.get("fusion_threshold", 0.0)),
                             rrf_k=float(self.params.get("rrf_k", 60.0))).fit(K)
            self.oc_ = sc
            self.support_ = np.arange(n)
            self.alpha_, self.b_ = None, 0.0
        elif self.method == "ocsvm":
            # one-class (novelty detection): no labels are used, every training molecule
            # belongs to the target class
            from sklearn.svm import OneClassSVM
            est = OneClassSVM(kernel="precomputed", nu=float(self.params.get("nu", 0.5)))
            est.fit(K)
            a = np.zeros(n)
            a[est.support_] = est.dual_coef_.ravel()
            self.alpha_, self.b_ = a, float(est.intercept_[0])
            self.support_ = np.array(est.support_)
        else:
            raise ValueError(f"unknown method {self.method}")
        return self

    def predict(self, K_test_train: np.ndarray,
                k_test_self: Optional[np.ndarray] = None) -> np.ndarray:
        oc = getattr(self, "oc_", None)
        if oc is not None:
            training = False
            if k_test_self is None:
                K = np.atleast_2d(K_test_train)
                if K.shape[0] == K.shape[1] == len(oc.diag_):
                    k_test_self, training = oc.diag_, True
                else:
                    raise ValueError("this one-class score needs the self-kernels of the "
                                     "molecules being scored")
            return oc.decision(K_test_train, k_test_self, training=training)
        return K_test_train @ self.alpha_ + self.b_

    def predict_variance(self, K_test_train: np.ndarray, k_test_self: np.ndarray) -> np.ndarray:
        """GP posterior variance (in units of the kernel) with lambda playing the role of noise.

        For the classifier it is the variance of the latent function, which is what the
        probability is averaged over.
        """
        if getattr(self, "gpc_", None) is not None:
            return self.gpc_.latent_variance(K_test_train, k_test_self)
        if self.K_train_ is None:
            return np.full(len(K_test_train), np.nan)
        lam = float(self.params.get("lam", 1.0))
        n = self.K_train_.shape[0]
        try:
            L = np.linalg.cholesky(self.K_train_ + lam * np.eye(n))
            v = np.linalg.solve(L, K_test_train.T)
            var = k_test_self - np.sum(v * v, axis=0)
        except np.linalg.LinAlgError:
            Kinv = np.linalg.pinv(self.K_train_ + lam * np.eye(n))
            var = k_test_self - np.einsum("ij,jk,ik->i", K_test_train, Kinv, K_test_train)
        return np.maximum(var, 0.0)


def _kfold_indices(n: int, folds: int, seed: Optional[int]) -> List[np.ndarray]:
    idx = np.arange(n)
    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
    return [idx[k::folds] for k in range(folds)]


def cv_predict(K: np.ndarray, y: np.ndarray, method: str, params: Dict[str, float],
               folds: int = 10, seed: Optional[int] = 0, fit_bias: bool = True) -> np.ndarray:
    """k-fold cross-validated predictions on a precomputed joint kernel matrix.

    ``seed=None`` uses the deterministic interleaved folds (molecule i goes to fold i mod k)
    of the original R scripts; an integer seed shuffles the molecules first.
    """
    n = len(y)
    folds = max(2, min(folds, n))
    yp = np.zeros(n)
    for te in _kfold_indices(n, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        learner = _Learner(method, params, fit_bias).fit(K[np.ix_(tr, tr)], y[tr])
        yp[te] = learner.predict(K[np.ix_(te, tr)])
    return yp


def median_gamma_scale(K: np.ndarray) -> float:
    """1 / median squared distance of the training molecules in the field Hilbert space.

    The bandwidth of the Gaussian kernel layer is expressed in these units, so that
    gamma = 1 is the median-distance bandwidth whatever the size of the fields; the usual
    "median heuristic" of kernel methods.
    """
    d = np.diag(K)
    d2 = d[:, None] + d[None, :] - 2.0 * K
    n = K.shape[0]
    if n < 2:
        return 1.0
    off = d2[~np.eye(n, dtype=bool)]
    m = float(np.median(np.maximum(off, 0.0)))
    return 1.0 / m if m > 1e-300 else 1.0


# --------------------------------------------------------------------------- #
# The CMF model
# --------------------------------------------------------------------------- #

@dataclass
class CMFModel:
    """A fitted (or to-be-fitted) CMF 3D-QSAR model.

    Attributes set by :meth:`fit`:
      train      - AtomTable of the training set
      y          - training activities
      coef       - dual coefficients a_j
      bias       - b
      self_diag  - K_f(j, j) for training molecules (for normalised kernels)
      scale      - per-field scaling constants c_f (unit mean diagonal) or ones
    """
    fieldset: FieldSet
    # "regression" | "classification" (binary, +1 active / -1 inactive) | "oneclass" (novelty detection)
    task: str = "regression"
    method: str = "krr"                     # krr | gp | svr | nusvr | svc | nusvc | ocsvm
    params: Dict[str, float] = field(default_factory=lambda: {"lam": 1.0, "C": 1.0, "epsilon": 0.1, "nu": 0.5})
    normalize: bool = False                 # cosine-normalise each field kernel
    scale_fields: bool = True               # scale each field kernel to unit mean diagonal
    # where the kernel comes from: the overlap integrals of the fields, which needs the molecules
    # to share a frame, or the rotation-invariant autocorrelation of those fields, which does not
    kernel_source: str = "fields"           # "fields" | "autocorrelation"
    ac_sigma: float = 0.75                  # autocorrelation: width along the distance axis (A)
    ac_d_max: float = 12.0                  # autocorrelation: longest distance resolved (A)
    ac_chirality: bool = False              # autocorrelation: add the three-point signed block
    ac_chirality_shells: int = 4            # autocorrelation: radial shells of that block
    # optional nonlinear layer on top of the CMF kernel: "linear" keeps the model linear in the
    # molecular field, which is what makes the co-field and the atom contributions valid; the other
    # layers predict better on several sets but destroy that interpretation (see the README)
    kernel_layer: str = "linear"            # linear | gauss | arccos1 | arccos2
    # multiple kernel learning: None keeps the mixing coefficients of the FieldSet, "align" or
    # "alignf" derive them by centred kernel alignment (set by grid_search; see align_mixing)
    mkl: Optional[str] = None
    mkl_rho: float = 1.0
    fit_bias: bool = True                   # KRR with bias term (Eq. 12); False -> b = 0
    # alpha grid on which kernels are evaluated; field kernels for alphas between the grid
    # points are linear interpolations of the grid kernels (the convention of the original R
    # scripts when alpha is optimised continuously); None: kernels at the exact alphas
    alpha_grid: Optional[List[float]] = None

    train: Optional[AtomTable] = None
    y: Optional[np.ndarray] = None
    coef: Optional[np.ndarray] = None
    bias: float = 0.0
    self_diag: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None
    K_fields_train: Optional[np.ndarray] = None
    stats: ModelStats = field(default_factory=ModelStats)
    learner_: Optional[_Learner] = None
    # the non-field descriptor block: the training statistics it is standardised with, the
    # standardised training matrix itself, and the constant that puts its mean diagonal at 1
    desc_scaler_: Optional[object] = None
    desc_train_: Optional[np.ndarray] = None
    desc_scale_: float = 1.0
    #: how the training set was prepared, as read from a model file (empty for a model fitted
    #: in this session, whose provenance is written when it is saved); see cmfqsar/contract.py
    provenance: Dict[str, object] = field(default_factory=dict)
    file_version: int = 0

    # ------------------------------------------------------------------ kernels
    def _interp_specs(self):
        """Interpolation terms for alphas off the grid: None, or ((c_lo, specs_lo), (c_hi, specs_hi))
        with per-field coefficient vectors; specs_hi is None when no field needs a second term."""
        if not self.alpha_grid:
            return None
        grid = sorted(float(a) for a in self.alpha_grid)
        c_lo, c_hi = [], []
        lo_specs, hi_specs = [], []
        plain = True
        for s in self.fieldset.specs:
            terms = interp_terms(s.alpha, grid)
            (a1, w1) = terms[0]
            (a2, w2) = terms[1] if len(terms) > 1 else (a1, 0.0)
            if a1 != s.alpha or w2 != 0.0:
                plain = False
            c_lo.append(w1)
            c_hi.append(w2)
            lo_specs.append(FieldSpec(**{**vars(s), "alpha": a1}))
            hi_specs.append(FieldSpec(**{**vars(s), "alpha": a2}))
        if plain:
            return None
        return (np.array(c_lo), lo_specs), (np.array(c_hi), hi_specs if any(c > 0 for c in c_hi) else None)

    def _raw_kernels(self, P: AtomTable, Q: AtomTable, progress=None, cancel=None) -> np.ndarray:
        """Per-field kernels at the current alphas (interpolated between grid alphas if needed)."""
        it = self._interp_specs()
        if it is None:
            return kernel_matrices(P, Q, self.fieldset.specs, progress=progress, cancel=cancel)
        (c_lo, lo), (c_hi, hi) = it
        K = c_lo[:, None, None] * kernel_matrices(P, Q, lo, progress=progress, cancel=cancel)
        if hi is not None:
            K += c_hi[:, None, None] * kernel_matrices(P, Q, hi, progress=progress, cancel=cancel)
        return K

    def _self_kernels(self, P: AtomTable) -> np.ndarray:
        it = self._interp_specs()
        if it is None:
            return self_kernels(P, self.fieldset.specs)
        (c_lo, lo), (c_hi, hi) = it
        d = c_lo[:, None] * self_kernels(P, lo)
        if hi is not None:
            d += c_hi[:, None] * self_kernels(P, hi)
        return d

    def _atom_block(self, mol: MoleculeFields) -> np.ndarray:
        it = self._interp_specs()
        if it is None:
            return atom_kernel_block(mol.coords, mol.weights, mol.radii, self.train, self.fieldset.specs)
        (c_lo, lo), (c_hi, hi) = it
        blk = c_lo[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii, self.train, lo)
        if hi is not None:
            blk += c_hi[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii, self.train, hi)
        return blk

    def _atom_self_block(self, mol: MoleculeFields) -> np.ndarray:
        """Kernels between the atoms of one molecule and each other: shape (F, N, N).

        The field of a subset of atoms is again a field, and the kernel is bilinear, so the
        self-kernel of any subset is the sum of this block over the subset. Used by the Shapley
        decomposition, which needs K(X_S, X_S) for coalitions S of atoms.
        """
        n = mol.n_atoms
        singles = AtomTable(coords=mol.coords, weights=mol.weights, radii=mol.radii,
                            offsets=np.arange(n + 1), names=[str(i) for i in range(n)],
                            molecules=[])
        it = self._interp_specs()
        if it is None:
            return atom_kernel_block(mol.coords, mol.weights, mol.radii, singles,
                                     self.fieldset.specs)
        (c_lo, lo), (c_hi, hi) = it
        blk = c_lo[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii, singles, lo)
        if hi is not None:
            blk += c_hi[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii, singles, hi)
        return blk

    def _field_kernels(self, P: AtomTable, Q: AtomTable, dP: Optional[np.ndarray] = None,
                       dQ: Optional[np.ndarray] = None, progress=None, cancel=None) -> np.ndarray:
        K = self._raw_kernels(P, Q, progress=progress, cancel=cancel)
        if self.normalize:
            if dP is None:
                dP = self._self_kernels(P)
            if dQ is None:
                dQ = self._self_kernels(Q)
            K = normalize_kernels(K, dP, dQ)
        return K

    def _compute_scale(self, K_train: np.ndarray) -> np.ndarray:
        F = K_train.shape[0]
        scale = np.ones(F)
        if self.scale_fields:
            for f in range(F):
                d = np.mean(np.diag(K_train[f]))
                scale[f] = 1.0 / d if d > 1e-300 else 0.0
        return scale

    @property
    def uses_fields(self) -> bool:
        """True when the kernel is the overlap integral, which is what needs a common frame."""
        return self.kernel_source in (None, "", "fields")

    def _ac_spec(self):
        from .invariant import AutocorrelationSpec
        return AutocorrelationSpec(d_max=float(self.ac_d_max), step=float(self.ac_sigma),
                                   sigma=float(self.ac_sigma),
                                   chirality=bool(self.ac_chirality),
                                   chirality_shells=int(self.ac_chirality_shells))

    def _ac_features(self, table: AtomTable) -> np.ndarray:
        from .invariant import feature_matrix
        return feature_matrix(table.molecules, self._ac_spec(),
                              n_fields=table.weights.shape[1])

    def training_kernel(self, train: AtomTable, progress=None, cancel=None) -> np.ndarray:
        """The joint training kernel, whichever source it comes from."""
        if self.uses_fields:
            return self.joint_kernel(self.prepare_training_kernels(train, progress=progress,
                                                                   cancel=cancel),
                                     train.descriptors)
        from .invariant import autocorrelation_kernel, block_sizes, channel_scale
        if progress:
            progress("Field autocorrelation", 0.2)
        self.train = train
        self.self_diag = None
        self.K_fields_train = None
        self.scale = np.ones(1)
        spec = self._ac_spec()
        self.ac_train_ = self._ac_features(train)
        self.ac_blocks_ = block_sizes(spec, train.weights.shape[1])
        self.ac_scale_ = channel_scale(self.ac_train_, spec.n_bins, blocks=self.ac_blocks_)
        return autocorrelation_kernel(self.ac_train_, scale=self.ac_scale_, n_bins=spec.n_bins,
                                      blocks=self.ac_blocks_)

    def test_kernel(self, test: AtomTable, progress=None, cancel=None) -> np.ndarray:
        """The (n_test, n_train) joint kernel block, whichever source it comes from."""
        if self.uses_fields:
            return self.joint_kernel(self.test_kernels(test, progress=progress, cancel=cancel),
                                     test.descriptors)
        from .invariant import autocorrelation_kernel
        if getattr(self, "ac_train_", None) is None:
            raise RuntimeError("model is not fitted")
        return autocorrelation_kernel(self.ac_train_, self._ac_features(test),
                                      scale=self.ac_scale_, n_bins=self._ac_spec().n_bins,
                                      blocks=getattr(self, "ac_blocks_", None))

    def _require_fields_if_atoms(self, what: str) -> None:
        """Atom-level interpretation needs the overlap kernel; the dual form does not."""
        if not self.uses_fields and what in ("co-fields", "atom contributions",
                                             "Shapley values over atoms"):
            self._require_fields(what)

    def _require_fields(self, what: str) -> None:
        if not self.uses_fields:
            raise ValueError(
                f"{what} are defined for the overlap-integral kernel, which is a function of the "
                f"fields in a common frame; this model uses the '{self.kernel_source}' kernel, "
                f"which is invariant to rotation and so has no field in a fixed frame to show. "
                f"The contributions of the training molecules remain valid.")

    def joint_kernel(self, K_fields: np.ndarray, desc: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear (overlap-integral) joint kernel: the sum over field types.

        `desc` holds the raw descriptor values of the molecules the *rows* belong to, when the
        field set names non-field descriptors.  They enter as one more block, standardised with
        the training set's statistics and scaled like a field; passing None leaves the kernel as
        the fields alone, which is what the callers that only want a field kernel do.
        """
        K = combine_kernels(K_fields, self.fieldset.mixing_vector(), self.scale)
        return K + self.descriptor_block(desc)

    def descriptor_block(self, desc: Optional[np.ndarray]) -> float:
        """h_d c_d z(rows) . z(training), or 0.0 when there are no descriptors to add."""
        if desc is None or self.desc_train_ is None:
            return 0.0
        from .descriptors import descriptor_kernel
        Z = self.desc_scaler_.transform(desc)
        return (self.fieldset.descriptor_h * self.desc_scale_
                * descriptor_kernel(Z, self.desc_train_))

    def descriptor_contributions(self, mol: "MoleculeFields") -> List[Tuple[str, float]]:
        """What each non-field descriptor contributed to one molecule's prediction.

        The block is a dot product, so its share of the prediction splits exactly into one term
        per descriptor, and the terms sum to the block's whole contribution.  Reported beside the
        atom contributions, this is what shows at a glance whether a model is standing on its
        geometry or on a descriptor that happens to track the activity.
        """
        if self.desc_train_ is None or self.coef is None:
            return []
        from .descriptors import contributions
        values = np.asarray(mol.descriptors, float).reshape(1, -1)
        z = self.desc_scaler_.transform(values)[0]
        parts = contributions(z, self.desc_train_, self.coef,
                              h=self.fieldset.descriptor_h, scale=self.desc_scale_)
        return list(zip(self.desc_scaler_.names, [float(v) for v in parts]))

    def descriptor_self(self, desc: Optional[np.ndarray]) -> float:
        """The descriptor block's own diagonal, which the variance and one-class scores need."""
        if desc is None or self.desc_train_ is None:
            return 0.0
        from .descriptors import descriptor_self as _self
        Z = self.desc_scaler_.transform(desc)
        return self.fieldset.descriptor_h * self.desc_scale_ * _self(Z)

    @property
    def is_linear(self) -> bool:
        """True when the kernel is the plain overlap integral, so co-fields are meaningful."""
        return self.kernel_layer in (None, "", "linear")

    def joint_self(self, table: AtomTable) -> np.ndarray:
        """Diagonal of the linear joint kernel, i.e. K(x, x) for every molecule of `table`."""
        if not self.uses_fields:
            return np.ones(table.n_mol)          # the autocorrelation kernel is cosine-normalised
        d = self_kernels(table, self.fieldset.specs)
        if self.normalize:
            # the cosine-normalised self-kernel is 1, except for a field that is identically
            # zero for that molecule (no donor, no charge, ...), which normalisation leaves at 0
            d = np.where(d > 1e-300, 1.0, 0.0)
        return ((self.fieldset.mixing_vector() * self.scale) @ d
                + self.descriptor_self(table.descriptors))

    def self_after_layer(self, d_linear: np.ndarray) -> np.ndarray:
        """K(x, x) of the kernel the learner actually sees, from the linear self-kernels.

        The distances of the one-class scores are taken in the Hilbert space of the kernel in
        use, so they need the self-kernel of that kernel: for the Gaussian layer every molecule
        sits on the unit sphere, and the arc-cosine layers leave the self-kernel where it was
        (which is what makes the same diagonals serve both of them).
        """
        if self.is_linear:
            return d_linear
        if self.kernel_layer == "gauss":
            return np.ones_like(np.asarray(d_linear, float))
        return np.asarray(d_linear, float)

    def apply_layer(self, K: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> np.ndarray:
        """Apply the nonlinear layer to a block of the linear kernel.

        gauss    : exp(-gamma d^2), a radial basis function in the Hilbert space of the fields.
        arccos1/2: the degree-1 arc-cosine kernel of Cho & Saul (2009), the kernel of an infinitely
                   wide one- (or two-) hidden-layer network; its self-kernel equals the base one,
                   so the same diagonals serve both layers.
        """
        if self.is_linear:
            return K
        if self.kernel_layer == "gauss":
            g = float(self.params.get("gamma", 1.0)) * float(getattr(self, "gamma_scale_", None) or 1.0)
            d2 = diag_p[:, None] + diag_q[None, :] - 2.0 * K
            return np.exp(-g * np.maximum(d2, 0.0))
        if self.kernel_layer in ("arccos1", "arccos2"):
            def arccos(Kx):
                npn = np.sqrt(np.maximum(diag_p, 1e-300))[:, None]
                nqn = np.sqrt(np.maximum(diag_q, 1e-300))[None, :]
                cos = np.clip(Kx / (npn * nqn), -1.0, 1.0)
                th = np.arccos(cos)
                return (npn * nqn) / np.pi * (np.sin(th) + (np.pi - th) * cos)
            out = arccos(K)
            return arccos(out) if self.kernel_layer == "arccos2" else out
        raise ValueError(f"unknown kernel layer {self.kernel_layer}")

    def _cv_predictions(self, K: np.ndarray, y: np.ndarray, folds: int,
                        seed: Optional[int]) -> np.ndarray:
        """Cross-validated predictions, with the mixing coefficients re-derived per fold if the
        model obtained them by kernel alignment (otherwise they would use the left-out activities)."""
        if self.mkl and self.K_fields_train is not None and len(self.fieldset.specs) > 1:
            yp, _ = mkl_cv_predict(self.K_fields_train, y, self, self.params, folds=folds,
                                   seed=seed, mode=self.mkl, rho=self.mkl_rho)
            return yp
        return cv_predict(K, y, self.method, self.params, folds=folds, seed=seed,
                          fit_bias=self.fit_bias)

    def layered_kernel(self, K_fields: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> np.ndarray:
        return self.apply_layer(self.joint_kernel(K_fields), diag_p, diag_q)

    # ---------------------------------------------------------------- fitting
    def prepare_training_kernels(self, train: AtomTable, progress=None, cancel=None) -> np.ndarray:
        """Compute (and cache) the per-field training kernel matrices for the current alphas."""
        self.train = train
        self.self_diag = self._self_kernels(train) if self.normalize else None
        self.K_fields_train = self._field_kernels(train, train, self.self_diag, self.self_diag,
                                                  progress=progress, cancel=cancel)
        self.scale = self._compute_scale(self.K_fields_train)
        self.prepare_descriptors(train)
        return self.K_fields_train

    def prepare_descriptors(self, train: AtomTable) -> None:
        """Fit the descriptor statistics on the training molecules and scale the block.

        Both belong to the training set and are stored with the model: standardising or rescaling
        according to the molecules being predicted would score two test sets on two scales. The
        scale puts the block's mean diagonal at 1, which is what ``scale_fields`` does to a field,
        so that a descriptor block with ten descriptors does not outweigh the geometry merely by
        being longer.
        """
        from .descriptors import DescriptorScaler, descriptor_self
        names = tuple(getattr(self.fieldset, "descriptors", ()) or ())
        if not names:
            self.desc_scaler_, self.desc_train_, self.desc_scale_ = None, None, 1.0
            return
        values = train.descriptors
        if values is None:
            raise ValueError("the field set names descriptors but the molecules carry none; "
                             "rebuild the fields with the same field set")
        self.desc_scaler_ = DescriptorScaler.fit(
            values, names, missing=getattr(self.fieldset, "prop_missing", "error"))
        self.desc_train_ = self.desc_scaler_.transform(values)
        d = float(np.mean(descriptor_self(self.desc_train_)))
        self.desc_scale_ = 1.0 / d if d > 1e-12 else 0.0

    def set_training_kernels(self, train: AtomTable, K_fields: np.ndarray,
                             self_diag: Optional[np.ndarray] = None) -> np.ndarray:
        """Install precomputed per-field training kernels (already normalised if `normalize`)."""
        self.train = train
        self.self_diag = self_diag
        self.K_fields_train = K_fields
        self.scale = self._compute_scale(K_fields)
        self.prepare_descriptors(train)
        return K_fields

    def fit(self, train: AtomTable, y: np.ndarray, cv_folds: int = 10, cv_seed: Optional[int] = 0,
            progress: Optional[ProgressFn] = None, cancel: Optional[CancelFn] = None,
            reuse_kernels: bool = False) -> "CMFModel":
        y = np.asarray(y, dtype=float)
        if self.uses_fields:
            if not (reuse_kernels and self.K_fields_train is not None and self.train is train):
                self.prepare_training_kernels(
                    train,
                    progress=(lambda i, n: progress(f"Kernel matrix {i}/{n}", 0.8 * i / n)) if progress else None,
                    cancel=cancel)
            K_lin = self.joint_kernel(self.K_fields_train, train.descriptors)
        else:
            K_lin = self.training_kernel(train, progress=progress, cancel=cancel)
        self.y = y
        self.base_diag_ = np.diag(K_lin).copy()
        self.gamma_scale_ = median_gamma_scale(K_lin) if self.kernel_layer == "gauss" else None
        K = self.apply_layer(K_lin, self.base_diag_, self.base_diag_)
        self.learner_ = _Learner(self.method, self.params, self.fit_bias).fit(K, y)
        self.coef = self.learner_.alpha_
        self.bias = self.learner_.b_
        yfit = self.learner_.predict(K)
        if self.task == "oneclass":
            st = ClassStats(task="oneclass", n_train=len(y) if y is not None else K.shape[0])
            st.inlier_fraction = float(np.mean(yfit >= 0))
            st.n_support = int(len(self.learner_.support_)) if self.learner_.support_ is not None else 0
            st.extra["y_fit"] = yfit
        elif self.task == "classification":
            st = ClassStats(task="classification", n_train=len(y),
                            fit=classification_scores(y, yfit))
            if cv_folds and cv_folds >= 2 and len(y) >= 3:
                if progress:
                    progress("Cross-validation", 0.9)
                ycv = self._cv_predictions(K, y, cv_folds, cv_seed)
                st.cv = classification_scores(y, ycv)
                st.cv_folds = min(cv_folds, len(y))
                st.extra["y_cv"] = ycv
            st.extra["y_fit"] = yfit
        else:
            st = ModelStats(n_train=len(y), r2=r2_score(y, yfit), rmse_fit=rmse(y, yfit))
            if cv_folds and cv_folds >= 2 and len(y) >= 3:
                if progress:
                    progress("Cross-validation", 0.9)
                ycv = self._cv_predictions(K, y, cv_folds, cv_seed)
                st.q2 = r2_score(y, ycv)
                st.rmse_cv = rmse(y, ycv)
                st.cv_folds = min(cv_folds, len(y))
                st.extra["y_cv"] = ycv
            st.extra["y_fit"] = yfit
        self.stats = st
        if progress:
            progress("Done", 1.0)
        return self

    # ------------------------------------------------------------- prediction
    def test_kernels(self, test: AtomTable, progress=None, cancel=None) -> np.ndarray:
        dQ = self._self_kernels(test) if self.normalize else None
        return self._field_kernels(test, self.train, dQ, self.self_diag, progress=progress, cancel=cancel)

    def predict(self, test: AtomTable, return_fields: bool = False, progress=None, cancel=None):
        oc = getattr(self.learner_, "oc_", None) if self.learner_ is not None else None
        if self.coef is None and oc is None:
            raise RuntimeError("model is not fitted")
        Kf = self.test_kernels(test, progress=progress, cancel=cancel) if self.uses_fields else None
        K = (self.joint_kernel(Kf, test.descriptors) if self.uses_fields
             else self.test_kernel(test, progress=progress))
        if not self.is_linear:
            K = self.apply_layer(K, self.joint_self(test), self.base_diag_)
        if oc is not None:
            yp = self.learner_.predict(K, self.self_after_layer(self.joint_self(test)))
        else:
            yp = K @ self.coef + self.bias
        if return_fields:
            return yp, Kf
        return yp

    def predict_from_field_kernels(self, K_fields_test: np.ndarray) -> np.ndarray:
        """Prediction from precomputed per-field kernels (F, n_test, n_train) with the training set."""
        if self.coef is None:
            raise RuntimeError("model is not fitted")
        self._require_fields("predictions from precomputed field kernels")
        self._require_linear("predictions from precomputed field kernels")
        if self.desc_train_ is not None:
            # the per-field kernels carry no descriptors, and a prediction that quietly left
            # the descriptor block out would be wrong rather than approximate
            raise ValueError("this model has non-field descriptors; predict from the atom "
                             "table instead of precomputed field kernels")
        return self.joint_kernel(K_fields_test) @ self.coef + self.bias

    def predict_with_variance(self, test: AtomTable) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction and GP posterior variance (only meaningful for method 'gp'/'krr')."""
        if not self.is_linear:
            raise ValueError(
                f"the posterior variance is implemented for the linear CMF kernel; this model uses "
                f"the '{self.kernel_layer}' layer. Fit a linear model (the same fields and alpha) to "
                f"obtain variances, for example for alignment by prediction confidence.")
        K = self.test_kernel(test)
        yp = K @ self.coef + self.bias
        if self.learner_ is None or self.learner_.K_train_ is None:
            learner = _Learner("gp", self.params)
            if self.uses_fields:
                if self.K_fields_train is None:
                    # a model read from a file stores its training atoms, not their kernels:
                    # compute them again, with the stored self-kernels and scale untouched
                    self.K_fields_train = self._field_kernels(self.train, self.train,
                                                              self.self_diag, self.self_diag)
                learner.K_train_ = self.joint_kernel(self.K_fields_train, self.train.descriptors)
            else:
                learner.K_train_ = self.training_kernel(self.train)
        else:
            learner = self.learner_
        var = learner.predict_variance(K, self.joint_self(test))
        return yp, var

    def rank_score(self, test: AtomTable, beta: float = 0.0,
                   larger_is_better: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(score to rank by, prediction, posterior standard deviation) for new molecules.

        With `beta` at zero the score is the prediction itself, which is what ranking a library
        by a regression model has always meant here.  Above zero it is a confidence bound,

            score = mean - beta * sd        (or mean + beta * sd when smaller is better)

        so a molecule is promoted only if the model is both optimistic *and* sure about it.  The
        point is not caution for its own sake: a screening library is by construction dominated
        by molecules unlike the training set, a kernel method returns the training mean dressed
        up for those, and a point estimate cannot tell that answer from a real one.  Subtracting
        the spread puts the molecules the model knows nothing about where they belong.

        This is the same quantity the applicability domain reports and the one-class Gaussian
        process already divides by; `beta` is the only choice, and one standard deviation is the
        conventional starting point.  It needs the linear kernel, since that is where a
        posterior exists.
        """
        yp, var = self.predict_with_variance(test)
        sd = np.sqrt(np.maximum(np.asarray(var, float), 0.0))
        b = float(beta)
        if b == 0.0:
            return yp, yp, sd
        return (yp - b * sd if larger_is_better else yp + b * sd), yp, sd

    def predict_proba(self, test: AtomTable) -> np.ndarray:
        """p(active) for new molecules; only the Gaussian-process classifier provides it.

        The latent posterior variance is averaged over, so a molecule the model has nothing to
        say about is reported near one half rather than confidently.
        """
        learner = self.learner_
        if learner is None or getattr(learner, "gpc_", None) is None:
            raise ValueError("probabilities come from the Gaussian-process classifier; "
                             "fit with method='gpc' (the support-vector classifiers return a "
                             "decision value, which is not a probability)")
        self._require_linear("class probabilities")
        K = self.test_kernel(test)
        return learner.gpc_.predict_proba(K, self.joint_self(test))

    @property
    def log_marginal_likelihood(self) -> float:
        """Approximate evidence of the fitted classifier; nan for every other method."""
        learner = self.learner_
        if learner is None or getattr(learner, "gpc_", None) is None:
            return float("nan")
        return float(learner.gpc_.log_marginal_likelihood)

    def evaluate_test(self, test: AtomTable, y_test: np.ndarray, progress=None, cancel=None) -> np.ndarray:
        yp = self.predict(test, progress=progress, cancel=cancel)
        y_test = np.asarray(y_test, float)
        ok = np.isfinite(y_test)
        if self.task in ("classification", "oneclass"):
            self.stats.n_test = int(ok.sum()) if ok.any() else int(len(yp))
            signs = np.unique(np.sign(y_test[ok])) if ok.any() else np.zeros(0)
            if ok.sum() >= 2 and len(signs) >= 2:
                self.stats.test = classification_scores(y_test[ok], yp[ok])
            self.stats.extra["y_test_pred"] = yp
            return yp
        if ok.sum() >= 2:
            self.stats.n_test = int(ok.sum())
            self.stats.r2_pred = r2_score(y_test[ok], yp[ok])
            self.stats.r2_pred_train_mean = r2_score(y_test[ok], yp[ok], float(np.mean(self.y)))
            self.stats.rmse_pred = rmse(y_test[ok], yp[ok])
            m = int(ok.sum())
            var_train = float(np.var(self.y, ddof=1)) if len(self.y) > 1 else float("nan")
            if m > 1 and var_train > 0:
                rss = float(np.sum((y_test[ok] - yp[ok]) ** 2))
                self.stats.r2_pred_train_var = 1.0 - (rss / (m - 1)) / var_train
        self.stats.extra["y_test_pred"] = yp
        return yp

    # ------------------------------------------------------- interpretation
    def training_contributions(self, mol: MoleculeFields) -> np.ndarray:
        """Decomposition rule 2: contribution a_j K(X, X_j) of every training molecule.

        Valid for any kernel, including a nonlinear layer.
        """
        oc = getattr(self.learner_, "oc_", None) if self.learner_ is not None else None
        if oc is not None and not oc.is_linear:
            raise ValueError(
                f"the contributions of the training molecules are defined for a score that sums "
                f"over them; '{oc.label}' does not ({oc.why_not_linear}). The one-class SVM, the "
                f"centroid distance and the one-class Gaussian process all do.")
        t = AtomTable.from_molecules([mol])
        K = self.test_kernel(t)
        if not self.is_linear:
            K = self.apply_layer(K, self.joint_self(t), self.base_diag_)
        return K[0] * self.coef

    def atom_contributions(self, mol: MoleculeFields, per_field: bool = False) -> np.ndarray:
        """Decomposition rule 3 (Eq. 14 of Sitnikov et al. 2015): contribution of every atom.

        Returns (N,) or, if per_field, (F, N).

        Only defined for the linear (overlap-integral) kernel: the decomposition into atom pairs
        is a property of that kernel and does not survive a nonlinear layer.
        """
        self._require_linear("atom contributions")
        blk = self._atom_block(mol)  # (F, N, n)
        if self.normalize:
            d_test = self._self_kernels(AtomTable.from_molecules([mol]))[:, 0]
            sp = np.sqrt(np.maximum(d_test, 0))
            sq = np.sqrt(np.maximum(self.self_diag, 0))
            denom = sp[:, None, None] * sq[:, None, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                blk = np.where(denom > 1e-300, blk / denom, 0.0)
        coef = self.fieldset.mixing_vector() * self.scale
        contrib = np.einsum("f,fnj,j->fn", coef, blk, self.coef)
        return contrib if per_field else contrib.sum(axis=0)

    def _require_linear(self, what: str) -> None:
        self._require_fields_if_atoms(what)
        oc = getattr(self.learner_, "oc_", None) if self.learner_ is not None else None
        if oc is not None and not oc.is_linear:
            raise ValueError(
                f"{what} are defined for a score that is a sum over the training molecules; "
                f"'{oc.label}' is not one ({oc.why_not_linear}), so no co-field exists for it. "
                f"The one-class SVM, the centroid distance and the one-class Gaussian process "
                f"all keep it.")
        if not self.is_linear:
            raise ValueError(
                f"{what} are defined only for the linear kernel: with the '{self.kernel_layer}' layer "
                f"the prediction is no longer an overlap integral of a co-field with the molecular "
                f"field, so the primal form does not exist. Use the linear kernel to interpret the "
                f"model (the contributions of the training molecules remain valid for any kernel).")

    def cofield_coefficients(self, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centres, coefficients and widths of the Gaussians composing co-field C_f(r).

        C_f(r) = sum_j a_j s_fj sum_m w_fjm exp(-b_jm/2 |r - r_jm|^2),  (Eq. 9)
        with s_fj = 1/sqrt(K_f(j,j)) when kernels are normalised, else 1.
        The field-scaling constant c_f and mixing coefficient h_f are *not* included.
        For a model whose alpha lies between the points of ``alpha_grid`` (interpolated
        kernels) the co-field is evaluated with the nominal alpha, an approximation.
        """
        self._require_linear("co-fields")
        self._require_fields("co-fields")
        spec = self.fieldset.specs[f]
        tr = self.train
        n = tr.n_mol
        s = np.ones(n)
        if self.normalize:
            d = self.self_diag[f]
            s = np.where(d > 1e-300, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
        per_mol = np.repeat(self.coef * s, np.diff(tr.offsets))
        if spec.is_probe:
            # a probe field puts several Gaussians on every atom, so its co-field is the same sum
            # with every atom repeated once per term of the expansion
            from .probes import basis_widths
            C = tr.probe_coeffs.get(spec.name)
            if C is None:
                return tr.coords, np.zeros(len(per_mol)), np.ones(len(per_mol))
            k = C.shape[1]
            return (np.repeat(tr.coords, k, axis=0), (C * per_mol[:, None]).reshape(-1),
                    np.tile(basis_widths(k), tr.coords.shape[0]))
        per_atom_coef = per_mol * tr.weights[:, f]
        b = np.full(len(per_atom_coef), spec.alpha) if spec.kind == "gauss" else spec.alpha / tr.radii ** 2
        return tr.coords, per_atom_coef, b

    def field_scale_for(self, f: int, mol: Optional[MoleculeFields] = None) -> float:
        """Factor multiplying the raw molecular field X_f of `mol` in the primal form (Eq. 7)."""
        if not self.normalize or mol is None:
            return 1.0
        d = self_kernels(AtomTable.from_molecules([mol]), self.fieldset.specs)[f, 0]
        return 1.0 / np.sqrt(d) if d > 1e-300 else 0.0

    # ------------------------------------------------------------ persistence
    #: the newest model-file version this program writes and reads
    FILE_VERSION = 2

    @staticmethod
    def describe_training(model: "CMFModel") -> dict:
        """How the training set was prepared, read off the stored training molecules.

        This is what the model file records as its provenance (version 2 files), and what the
        contract check falls back to for a file that recorded nothing.
        """
        mols = list(model.train.molecules) if model.train is not None else []
        n = len(mols)
        sources: Dict[str, int] = {}
        for m in mols:
            s = getattr(m, "charge_source", "") or "unknown"
            sources[s] = sources.get(s, 0) + 1
        with_h = sum(1 for m in mols if "H" in m.symbols)
        out = {
            "n_train": n,
            "charge_model": model.fieldset.charge_model,
            "param_scheme": model.fieldset.param_scheme,
            "fields": [s.name for s in model.fieldset.specs if s.enabled],
            "charge_sources": sources,
            "charge_source": max(sources, key=sources.get) if sources else None,
            "explicit_hydrogens": (with_h / n) if n else None,
            "conformers": model.fieldset.conformers,
        }
        if model.train is not None and len(model.train.coords):
            out["box_min"] = [float(v) for v in model.train.coords.min(axis=0)]
            out["box_max"] = [float(v) for v in model.train.coords.max(axis=0)]
        if model.y is not None and len(model.y):
            y = np.asarray(model.y, float)
            out["activity"] = {"min": float(np.nanmin(y)), "max": float(np.nanmax(y)),
                               "mean": float(np.nanmean(y))}
        return out

    def _provenance(self) -> dict:
        """The record a model file carries: versions, date and how the training set was made."""
        import platform
        import time
        from . import __version__
        from rdkit import rdBase
        p = dict(self.describe_training(self))
        p.update({"cmfqsar": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "python": platform.python_version(), "numpy": np.__version__,
                  "rdkit": rdBase.rdkitVersion})
        return p

    def state(self) -> dict:
        return {
            "version": self.FILE_VERSION,
            "provenance": self._provenance(),
            "fieldset": self.fieldset,
            "task": self.task,
            "method": self.method,
            "params": self.params,
            "normalize": self.normalize,
            "scale_fields": self.scale_fields,
            "kernel_layer": self.kernel_layer,
            "base_diag": getattr(self, "base_diag_", None),
            "gamma_scale": getattr(self, "gamma_scale_", None),
            "mkl": self.mkl,
            "mkl_rho": self.mkl_rho,
            "fit_bias": self.fit_bias,
            "alpha_grid": self.alpha_grid,
            "train": self.train,
            "y": self.y,
            "coef": self.coef,
            "bias": self.bias,
            "self_diag": self.self_diag,
            "scale": self.scale,
            "stats": self.stats,
            "oneclass": getattr(self.learner_, "oc_", None) if self.learner_ is not None else None,
            "kernel_source": self.kernel_source,
            "ac_sigma": self.ac_sigma,
            "ac_chirality": self.ac_chirality,
            "ac_chirality_shells": self.ac_chirality_shells,
            "ac_d_max": self.ac_d_max,
            "ac_train": getattr(self, "ac_train_", None),
            "ac_scale": getattr(self, "ac_scale_", None),
            # the descriptor block: its standardisation and scale belong to the training set
            "desc_scaler": self.desc_scaler_,
            "desc_train": self.desc_train_,
            "desc_scale": self.desc_scale_,
        }

    @classmethod
    def from_state(cls, st: dict) -> "CMFModel":
        version = int(st.get("version", 1))
        if version > cls.FILE_VERSION:
            raise ValueError(f"this model file is version {version}, written by a newer cmfqsar "
                             f"than this one, which reads up to version {cls.FILE_VERSION}")
        m = cls(fieldset=st["fieldset"], task=st.get("task", "regression"), method=st["method"],
                params=st["params"], normalize=st["normalize"], scale_fields=st["scale_fields"],
                fit_bias=st.get("fit_bias", True), alpha_grid=st.get("alpha_grid"),
                kernel_layer=st.get("kernel_layer", "linear"),
                mkl=st.get("mkl"), mkl_rho=float(st.get("mkl_rho", 1.0)))
        m.kernel_source = st.get("kernel_source", "fields")
        m.ac_sigma = float(st.get("ac_sigma", 0.75))
        m.ac_chirality = bool(st.get("ac_chirality", False))
        m.ac_chirality_shells = int(st.get("ac_chirality_shells", 4))
        m.ac_d_max = float(st.get("ac_d_max", 12.0))
        m.ac_train_ = st.get("ac_train")
        m.ac_scale_ = st.get("ac_scale")
        m.base_diag_ = st.get("base_diag")
        m.gamma_scale_ = st.get("gamma_scale")
        m.train, m.y, m.coef, m.bias = st["train"], st["y"], st["coef"], st["bias"]
        m.self_diag, m.scale, m.stats = st["self_diag"], st["scale"], st["stats"]
        m.provenance = dict(st.get("provenance") or {})
        m.file_version = version
        if "desc_train" in st:
            m.desc_scaler_, m.desc_train_ = st["desc_scaler"], st["desc_train"]
            m.desc_scale_ = float(st.get("desc_scale", 1.0))
        elif getattr(m.fieldset, "descriptors", ()) and m.train is not None:
            # a version-1 file with a descriptor block stored the training molecules and their
            # raw descriptor values but not the standardisation; fit it again on them
            m.prepare_descriptors(m.train)
        if st.get("oneclass") is not None:
            m.learner_ = _Learner(st["method"], st["params"])
            m.learner_.oc_ = st["oneclass"]
            m.learner_.alpha_, m.learner_.b_ = m.coef, m.bias
        return m


# --------------------------------------------------------------------------- #
# Hyper-parameter search
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Multiple kernel learning: mixing coefficients by centred kernel alignment
# --------------------------------------------------------------------------- #

def _centre_kernel(K: np.ndarray) -> np.ndarray:
    """H K H with H = I - 11^T/n (centre the feature space at the mean molecule)."""
    m = K.mean(axis=0)
    return K - m[None, :] - m[:, None] + K.mean()


def align_mixing(K_fields: np.ndarray, y: np.ndarray, scale: Optional[np.ndarray] = None,
                 enabled: Optional[Sequence[bool]] = None, mode: str = "align",
                 rho: float = 1.0) -> np.ndarray:
    """Mixing coefficients h_f by centred kernel alignment (multiple kernel learning).

    The joint kernel of the CMF approach, K = sum_f h_f c_f K_f, is a *linear* combination of
    the per-field overlap-integral kernels, so choosing h_f is exactly the multiple-kernel-
    learning problem.  Alignment measures the cosine between a centred kernel and the ideal
    kernel y y^T of the activities:

      ``mode="align"``  (Cristianini et al. 2001) weights every field by its own alignment,
        h_f ~ <K_f^c, y^c y^c^T>_F / ||K_f^c||_F, which is robust because the fields are
        treated independently;
      ``mode="alignf"`` (Cortes et al. 2012) maximises the alignment of the *combination*,
        i.e. solves min_{h >= 0} || sum_f h_f K_f^c - y^c y^c^T ||_F.  On congeneric series the
        field kernels are strongly correlated and this solution is nearly sparse, so it is
        shrunk towards equal weights with strength `rho` (rho = 0 is the plain alignf).

    The weights are rescaled so that the mean diagonal of the joint kernel is the same as with
    h_f = 1, which keeps the regularisation grid comparable.  The combination stays linear, so
    co-fields and atom contributions remain valid - that is the difference from a nonlinear
    kernel layer, which can buy accuracy but loses the interpretation.
    """
    K_fields = np.asarray(K_fields, float)
    F = K_fields.shape[0]
    if scale is None:
        scale = np.ones(F)
    scale = np.asarray(scale, float)
    on = np.ones(F, bool) if enabled is None else np.asarray(enabled, bool)
    idx = np.flatnonzero(on)
    if idx.size == 0:
        return np.zeros(F)
    yv = np.asarray(y, float)
    yv = yv - yv.mean()
    A = np.stack([_centre_kernel(K_fields[f] * scale[f]).reshape(-1) for f in idx], axis=1)
    b = np.outer(yv, yv).reshape(-1)
    if mode == "align":
        nrm = np.linalg.norm(A, axis=0)
        w = np.where(nrm > 1e-300, np.maximum(A.T @ b, 0.0) / np.maximum(nrm, 1e-300), 0.0)
    elif mode == "alignf":
        from scipy.optimize import nnls
        if rho > 0:
            sc = float(np.mean(np.linalg.norm(A, axis=0)))
            denom = float((A.T @ A).sum())
            h0 = np.full(idx.size, float((A.T @ b).sum()) / denom if denom > 1e-300 else 1.0)
            A = np.vstack([A, np.sqrt(rho) * sc * np.eye(idx.size)])
            b = np.concatenate([b, np.sqrt(rho) * sc * h0])
        w, _ = nnls(A, b)
    else:
        raise ValueError(f"unknown alignment mode {mode}")
    h = np.zeros(F)
    h[idx] = w
    if not np.isfinite(h).all() or h.sum() <= 1e-12:
        h = on.astype(float)                      # degenerate: fall back to equal weights
    ref = float(sum(np.mean(np.diag(K_fields[f])) * scale[f] for f in idx))
    cur = float(sum(h[f] * np.mean(np.diag(K_fields[f])) * scale[f] for f in idx))
    if cur > 1e-300 and ref > 1e-300:
        h = h * (ref / cur)
    return h


def mkl_fold_weights(K_fields: np.ndarray, y: np.ndarray, model: "CMFModel", folds: int = 10,
                     seed: Optional[int] = 0, mode: str = "align",
                     rho: float = 1.0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-fold alignment weights: a list of (test indices, training indices, h)."""
    y = np.asarray(y, float)
    n = len(y)
    folds = max(2, min(folds, n))
    on = [s.enabled for s in model.fieldset.specs]
    f_idx = np.arange(K_fields.shape[0])
    out = []
    for te in _kfold_indices(n, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        out.append((te, tr, align_mixing(K_fields[np.ix_(f_idx, tr, tr)], y[tr], model.scale, on,
                                         mode=mode, rho=rho)))
    return out


def mkl_cv_predict(K_fields: np.ndarray, y: np.ndarray, model: "CMFModel", params: Dict[str, float],
                   folds: int = 10, seed: Optional[int] = 0,
                   fold_h: Optional[List[Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
                   mode: str = "align", rho: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-validated predictions with the mixing coefficients re-derived inside every fold.

    Deriving h_f from all activities and then cross-validating would leak the left-out
    activities into the kernel; the weights are therefore computed from the training part of
    each fold only.  Returns the predictions and the weights of the full training set.
    """
    y = np.asarray(y, float)
    yp = np.zeros(len(y))
    if fold_h is None:
        fold_h = mkl_fold_weights(K_fields, y, model, folds, seed, mode=mode, rho=rho)
    for te, tr, h in fold_h:
        K = combine_kernels(K_fields, h, model.scale)
        dg = np.diag(K).copy()
        K = model.apply_layer(K, dg, dg)
        learner = _Learner(model.method, params, model.fit_bias).fit(K[np.ix_(tr, tr)], y[tr])
        yp[te] = learner.predict(K[np.ix_(te, tr)])
    return yp, align_mixing(K_fields, y, model.scale, [s.enabled for s in model.fieldset.specs],
                            mode=mode, rho=rho)


@dataclass
class SearchResult:
    best_alpha: float
    best_params: Dict[str, float]
    best_h: Dict[str, float]
    best_rmse: float
    best_q2: float
    table: List[dict]          # rows: alpha, params, rmse_cv, q2
    y_cv: Optional[np.ndarray] = None


def _param_grid(method: str, lams, Cs, epsilons, nus):
    if method in ("krr", "gp"):
        return [{"lam": float(l)} for l in lams]
    if method == "svr":
        return [{"C": float(C), "epsilon": float(e)} for C in Cs for e in epsilons]
    if method == "nusvr":
        return [{"C": float(C), "nu": float(nu)} for C in Cs for nu in nus]
    if method == "svc":
        return [{"C": float(C)} for C in Cs]
    if method == "gpc":
        # the amplitude of the kernel plays the part C plays for a support-vector classifier
        return [{"amplitude": float(C)} for C in Cs]
    if method == "nusvc":
        return [{"nu": float(nu)} for nu in nus]
    if method == "ocsvm":
        return [{"nu": float(nu)} for nu in nus]
    raise ValueError(method)


def cv_loss(model: "CMFModel", y: np.ndarray, ycv: np.ndarray) -> Tuple[float, float]:
    """(loss to minimise, quality metric to report) of cross-validated predictions.

    Regression: RMSEcv and q2.  Classification: 1 - AUC (falling back to 1 - balanced
    accuracy when only one class is present in a fold) and the balanced accuracy.
    """
    if model.task == "classification":
        sc = classification_scores(y, ycv)
        quality = sc["auc"] if np.isfinite(sc["auc"]) else sc["balanced_accuracy"]
        if not np.isfinite(quality):
            return 1.0, float("nan")
        return 1.0 - float(quality), float(sc["balanced_accuracy"])
    return rmse(y, ycv), r2_score(y, ycv)


def grid_search(train: AtomTable, y: np.ndarray, model: CMFModel,
                alphas: Sequence[float], lams: Sequence[float] = (1.0,),
                Cs: Sequence[float] = (1.0,), epsilons: Sequence[float] = (0.1,),
                nus: Sequence[float] = (0.5,), gammas: Sequence[float] = (), folds: int = 10,
                seed: Optional[int] = 0,
                optimize_h: bool = False, mkl=False, mkl_rho: float = 1.0,
                h_grid: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0),
                h_passes: int = 2, alpha_fields: Optional[Sequence[str]] = None,
                progress: Optional[ProgressFn] = None, cancel: Optional[CancelFn] = None,
                lam_search: str = "grid", precomputed: Optional[Dict] = None,
                alpha_search: str = "grid", nm_start: Sequence[float] = (0.25, 5.0),
                nm_maxit: int = 500) -> SearchResult:
    """Minimise RMSEcv over alpha (shared by all fields unless `alpha_fields` restricts it)
    and the learner's regularisation parameters.  Optionally refine the mixing
    coefficients h_f by coordinate descent (beware of model-selection bias), or set them by
    centred kernel alignment (``mkl=True``, see :func:`align_mixing`), in which case the
    reported RMSEcv is honest because the weights are re-derived inside every fold.

    ``lam_search="brent"`` (KRR/GP only) additionally minimises RMSEcv over a *continuous*
    lambda in [min(lams), max(lams)] with Brent's bounded method, as the R scripts of the
    2013 paper did (``optimize()`` over gamma with the alpha grid evaluated inside);
    the best of the grid and the Brent result is kept.

    ``alpha_search="nelder-mead"`` (KRR/GP, unnormalised kernels) instead minimises RMSEcv
    over the pair (alpha, lambda) with R's Nelder-Mead simplex (``optim``) started at
    ``nm_start``, the kernels for alphas between the grid points being linear interpolations
    of the grid kernels and the end kernels being used beyond the grid, exactly as
    ``cmf_krr_train_mem`` of the original scripts with ``alpha_grid_search = FALSE`` (the
    protocol of the continuous-indicator-field models of Sitnikov et al. 2015).  The result of
    the simplex search is used as it is (no comparison with the grid), the model's
    ``alpha_grid`` is set so that kernels for new molecules are interpolated in the same way.

    ``precomputed`` maps alpha -> (K_fields, self_diag) for the molecules of `train`
    (used by :func:`nested_cv` to avoid recomputing kernels for every outer fold).
    ``seed=None`` selects the deterministic interleaved folds of the original scripts."""
    y = np.asarray(y, float)
    if model.task == "oneclass":
        raise ValueError(
            "hyper-parameters of a one-class model cannot be selected by cross-validation: "
            "there are no negatives to score against. nu is the expected fraction of outliers and "
            "alpha sets the level of generalisation; choose them directly (see Fig. 5 of Baskin & "
            "Zhokhova 2019) and fit the model.")
    if not model.is_linear:
        if alpha_search == "nelder-mead" or lam_search == "brent":
            raise ValueError("the continuous alpha/lambda searches of the original scripts are "
                             "implemented for the linear kernel only; use the grids instead")
    if mkl and optimize_h:
        raise ValueError("choose either alignment-based multiple kernel learning or the "
                         "coordinate-descent refinement of h, not both")
    grid = _param_grid(model.method, lams, Cs, epsilons, nus)
    rows = []
    h_cache = {}
    best = None
    # alphas=None/empty: keep the per-field alphas already stored in the FieldSet
    alpha_list = list(alphas) if alphas else [None]
    if not model.uses_fields:
        alpha_list = alpha_list[:1]          # the attenuation does not enter this kernel
    n_alpha = len(alpha_list)
    cache = {}
    nelder_mead = alpha_search == "nelder-mead"
    if nelder_mead:
        if model.method not in ("krr", "gp"):
            raise ValueError("Nelder-Mead alpha search is only available for KRR/GP")
        if model.normalize:
            raise ValueError("Nelder-Mead alpha search is not available with normalised kernels")
        if alpha_list == [None]:
            raise ValueError("Nelder-Mead alpha search needs an alpha grid")
        grid = [{"lam": float(nm_start[1])}]   # kernels only; the (alpha, lambda) search follows
    for ia, alpha in enumerate(alpha_list):
        if cancel and cancel():
            raise Cancelled()
        if alpha is not None:
            model.fieldset.set_alpha(alpha, alpha_fields)
        atxt = "table alphas" if alpha is None else f"alpha = {alpha:g}"
        if progress:
            progress(f"{atxt}: computing kernels", ia / n_alpha)
        if not model.uses_fields:
            # the attenuation belongs to the overlap integral; this kernel has its own width
            # along the distance axis, so there is one kernel and the alpha loop runs once
            Kf = None
            K_lin = model.training_kernel(train, cancel=cancel)
            use_mkl = False
            fold_h = None
        elif precomputed is not None and alpha in precomputed:
            Kf_pre, sd_pre = precomputed[alpha]
            Kf = model.set_training_kernels(train, Kf_pre, sd_pre)
        else:
            Kf = model.prepare_training_kernels(
                train, cancel=cancel,
                progress=(lambda i, n, ia=ia, atxt=atxt: progress(f"{atxt}: kernels {i}/{n}", (ia + 0.8 * i / n) / n_alpha)) if progress else None)
        if Kf is not None:
            cache[alpha] = (Kf.copy(), model.scale.copy(),
                            None if model.self_diag is None else model.self_diag.copy())
        use_mkl = Kf is not None and bool(mkl) and len(model.fieldset.specs) > 1
        mkl_mode = mkl if isinstance(mkl, str) else "align"
        model.mkl, model.mkl_rho = (mkl_mode if use_mkl else None), mkl_rho
        if use_mkl:
            if progress:
                progress(f"{atxt}: kernel alignment", (ia + 0.9) / n_alpha)
            fold_h = mkl_fold_weights(Kf, y, model, folds, seed, mode=mkl_mode, rho=mkl_rho)
            h_full = align_mixing(Kf, y, model.scale, [sp.enabled for sp in model.fieldset.specs],
                                  mode=mkl_mode, rho=mkl_rho)
            h_cache[alpha] = {sp.name: float(hv) for sp, hv in zip(model.fieldset.specs, h_full)}
            K_lin = combine_kernels(Kf, h_full, model.scale)
        elif Kf is not None:
            fold_h = None
            K_lin = model.joint_kernel(Kf, train.descriptors)
        dg = np.diag(K_lin).copy()
        if model.kernel_layer == "gauss":
            model.gamma_scale_ = median_gamma_scale(K_lin)
        for gamma in (gammas if (gammas and model.kernel_layer == "gauss") else [None]):
            if gamma is not None:
                model.params["gamma"] = float(gamma)
            K = model.apply_layer(K_lin, dg, dg)
            for params in grid:
                if use_mkl:
                    ycv, _ = mkl_cv_predict(Kf, y, model, params, folds=folds, seed=seed,
                                            fold_h=fold_h)
                else:
                    ycv = cv_predict(K, y, model.method, params, folds=folds, seed=seed,
                                     fit_bias=model.fit_bias)
                r, q2 = cv_loss(model, y, ycv)
                row = {"alpha": float("nan") if alpha is None else alpha, **params,
                       "rmse_cv": r, "q2": q2}
                if gamma is not None:
                    row["gamma"] = float(gamma)
                if use_mkl:
                    row["h"] = dict(h_cache[alpha])
                rows.append(row)
                if best is None or r < best[0]:
                    best = (r, q2, alpha, dict(params, **({"gamma": float(gamma)} if gamma is not None
                                                          else {})), ycv)
    r, q2, alpha, params, ycv = best
    if alpha in h_cache:
        model.fieldset.set_mixing(h_cache[alpha])
    h_best = {s.name: s.h for s in model.fieldset.specs}

    if nelder_mead:
        # Simultaneous minimisation of RMSEcv over (alpha, lambda) with R's Nelder-Mead
        # on linearly interpolated grid kernels (cmf_krr_train_mem, alpha_grid_search = FALSE).
        agrid = sorted(float(a) for a in alpha_list)
        hvec = model.fieldset.mixing_vector()
        evals = []

        def interp_fields(a):
            Kf = None
            for ag, c in interp_terms(a, agrid):
                Kf = c * cache[ag][0] if Kf is None else Kf + c * cache[ag][0]
            return Kf

        def objective(par):
            if cancel and cancel():
                raise Cancelled()
            a_try, lam_try = float(par[0]), float(par[1])
            Kf = interp_fields(a_try)
            sc = model._compute_scale(Kf)
            ycv_t = cv_predict(combine_kernels(Kf, hvec, sc), y, model.method, {"lam": lam_try},
                               folds=folds, seed=seed, fit_bias=model.fit_bias)
            r_t = rmse(y, ycv_t)
            evals.append((r_t, a_try, lam_try, ycv_t))
            if progress:
                progress(f"Nelder-Mead: alpha = {a_try:.4g}, lambda = {lam_try:.4g}, RMSEcv = {r_t:.4f}", 1.0)
            return r_t

        _, _, n_eval, fail = nmmin(objective, list(nm_start), maxit=nm_maxit)
        r, alpha_raw, lam_b, ycv = min(evals, key=lambda t: t[0])
        alpha = min(max(alpha_raw, agrid[0]), agrid[-1])      # as model$alpha of the scripts
        q2 = r2_score(y, ycv)
        params = {"lam": lam_b}
        Kf_b = interp_fields(alpha)
        cache[alpha] = (Kf_b, model._compute_scale(Kf_b), None)
        model.alpha_grid = agrid
        rows.append({"alpha": alpha, "lam": lam_b, "rmse_cv": r, "q2": q2, "nelder_mead": True,
                     "n_eval": n_eval, "fail": fail})

    if lam_search == "brent" and not nelder_mead and model.method in ("krr", "gp") and len(lams) >= 2:
        # Continuous minimisation of RMSEcv over lambda (Brent, bounded) with the
        # best alpha chosen inside the objective, as in the original R scripts.
        from scipy.optimize import minimize_scalar
        lo, hi = float(min(lams)), float(max(lams))
        hvec = model.fieldset.mixing_vector()
        Kj = {a: combine_kernels(cache[a][0], hvec, cache[a][1]) for a in alpha_list}
        evals = []

        def objective(lam):
            if cancel and cancel():
                raise Cancelled()
            loc = None
            for a in alpha_list:
                ycv_a = cv_predict(Kj[a], y, model.method, {"lam": float(lam)}, folds=folds, seed=seed,
                                   fit_bias=model.fit_bias)
                r_a = rmse(y, ycv_a)
                if loc is None or r_a < loc[0]:
                    loc = (r_a, a, ycv_a)
            evals.append((float(lam),) + loc)
            if progress:
                progress(f"lambda search: lambda = {lam:.4g}, RMSEcv = {loc[0]:.4f}", 1.0)
            return loc[0]

        minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1.22e-4})
        lam_b, r_b, alpha_b, ycv_b = min(evals, key=lambda t: t[1])
        rows.append({"alpha": float("nan") if alpha_b is None else alpha_b, "lam": lam_b, "rmse_cv": r_b,
                     "q2": r2_score(y, ycv_b), "brent": True})
        if r_b < r - 1e-12:
            r, q2, alpha, params, ycv = r_b, r2_score(y, ycv_b), alpha_b, {"lam": lam_b}, ycv_b

    if optimize_h and len(model.fieldset.specs) > 1:
        Kf, scale, sd = cache[alpha]
        if alpha is not None:
            model.fieldset.set_alpha(alpha, alpha_fields)
        model.K_fields_train, model.scale, model.self_diag, model.train = Kf, scale, sd, train
        names = [s.name for s in model.fieldset.specs if s.enabled]
        total = h_passes * len(names) * len(h_grid)
        step = 0
        for p in range(h_passes):
            for name in names:
                cur = h_best[name]
                best_local = (r, cur)
                for hv in h_grid:
                    step += 1
                    if cancel and cancel():
                        raise Cancelled()
                    if progress:
                        progress(f"mixing pass {p + 1}: {name} h = {hv:g}", step / total)
                    trial = dict(h_best)
                    trial[name] = hv
                    if all(v == 0 for v in trial.values()):
                        continue
                    model.fieldset.set_mixing(trial)
                    K_lin_t = model.joint_kernel(Kf, train.descriptors)
                    K = model.apply_layer(K_lin_t, np.diag(K_lin_t), np.diag(K_lin_t))
                    ycv_t = cv_predict(K, y, model.method, params, folds=folds, seed=seed, fit_bias=model.fit_bias)
                    rt, q2_t = cv_loss(model, y, ycv_t)
                    if rt < best_local[0] - 1e-12:
                        best_local = (rt, hv)
                        ycv = ycv_t
                        q2 = q2_t
                r, h_best[name] = best_local
                model.fieldset.set_mixing(h_best)
        rows.append({"alpha": float("nan") if alpha is None else alpha, **params, "rmse_cv": r, "q2": q2,
                     "h": dict(h_best)})

    # leave the model in the best configuration
    model.params.update(params)
    if cache:
        Kf, scale, sd = cache[alpha]
        if alpha is not None:
            model.fieldset.set_alpha(alpha, alpha_fields)
        model.fieldset.set_mixing(h_best)
        model.K_fields_train, model.scale, model.self_diag, model.train = Kf, scale, sd, train
    else:
        model.train = train                  # a kernel source with nothing to cache per alpha
    if progress:
        progress("Search finished", 1.0)
    return SearchResult(best_alpha=float("nan") if alpha is None else alpha, best_params=params, best_h=h_best,
                        best_rmse=r, best_q2=q2, table=rows, y_cv=ycv)


# --------------------------------------------------------------------------- #
# External (nested) cross-validation
# --------------------------------------------------------------------------- #

@dataclass
class NestedCVResult:
    q2_ex: float                     # mean over repeats of the per-repeat statistics
    rmse_ex: float
    q2_ex_sd: float
    rmse_ex_sd: float
    y_pred: np.ndarray               # averaged over repeats
    per_repeat: List[dict]
    q2_ex_aggr: float = float("nan")   # statistics of the prediction averaged over repeats
    rmse_ex_aggr: float = float("nan")  # (Q2ecv_aggr / RMSEecv_aggr of the original scripts)


def nested_cv(train: AtomTable, y: np.ndarray, model_template: CMFModel,
              alphas: Sequence[float], lams=(1.0,), Cs=(1.0,), epsilons=(0.1,), nus=(0.5,),
              outer_folds: int = 5, inner_folds: int = 10, repeats: int = 1, seed: Optional[int] = 0,
              optimize_h: bool = False, mkl=False, mkl_rho: float = 1.0, alpha_fields=None,
              gammas: Sequence[float] = (),
              progress: Optional[ProgressFn] = None, cancel: Optional[CancelFn] = None,
              lam_search: str = "grid", precompute: bool = True,
              max_precompute_bytes: float = 2e9, alpha_search: str = "grid",
              nm_start: Sequence[float] = (0.25, 5.0), inner_seed="same") -> NestedCVResult:
    """External k-fold cross-validation with the hyper-parameters selected in an
    inner cross-validation loop on each working set (Baskin & Zhokhova 2013).

    With ``precompute`` the per-field kernel matrices of the whole set are computed once
    for every alpha of the grid and sliced for the outer folds (identical results, much
    faster).  ``seed=None`` uses interleaved (unshuffled) outer and inner folds.  With
    ``repeats > 1`` the molecules are reshuffled before every repeat (``seed`` + repeat index;
    ``seed=None`` gives the same interleaved outer folds in every repeat); besides the mean and
    standard deviation of the per-repeat statistics the result also holds the statistics of
    the prediction averaged over the repeats (``q2_ex_aggr``, ``rmse_ex_aggr``), which is
    what the original scripts (``cmf_krr_ecvr``) and the 2015 paper report.  ``inner_seed``
    is the seed of the inner cross-validation (default: the same as ``seed``); ``None`` gives
    interleaved inner folds as in the scripts, which reshuffle the molecules only once per
    repeat."""
    y = np.asarray(y, float)
    if isinstance(inner_seed, str):
        inner_seed = seed
    n = len(y)
    preds = np.zeros((repeats, n))
    per_repeat = []
    total_steps = repeats * outer_folds
    step = 0
    specs = model_template.fieldset.specs
    pre_full = None
    pre_frac = 0.0
    if precompute and alphas and len(specs) * n * n * len(alphas) * 8 < max_precompute_bytes:
        pre_full = {}
        pre_frac = 0.3
        fs_tmp = model_template.fieldset.copy()
        for ia, alpha in enumerate(alphas):
            if cancel and cancel():
                raise Cancelled()
            if progress:
                progress(f"kernels for alpha = {alpha:g} ({ia + 1}/{len(alphas)})", pre_frac * ia / len(alphas))
            fs_tmp.set_alpha(alpha, alpha_fields)
            Kf = kernel_matrices(train, train, fs_tmp.specs, cancel=cancel)
            sd = None
            if model_template.normalize:
                sd = np.array([np.diag(Kf[f]) for f in range(len(specs))])
                Kf = normalize_kernels(Kf, sd, sd)
            pre_full[alpha] = (Kf, sd)
    for rep in range(repeats):
        folds = _kfold_indices(n, outer_folds, None if seed is None else seed + rep)
        fold_hyper = []     # (alpha, params) selected in every outer fold
        for k, te in enumerate(folds):
            if cancel and cancel():
                raise Cancelled()
            tr = np.setdiff1d(np.arange(n), te)
            sub_train = train.subset(tr)
            model = CMFModel(fieldset=model_template.fieldset.copy(), method=model_template.method,
                             params=dict(model_template.params), normalize=model_template.normalize,
                             scale_fields=model_template.scale_fields, fit_bias=model_template.fit_bias,
                             kernel_layer=model_template.kernel_layer)
            base = pre_frac + (1.0 - pre_frac) * step / total_steps
            span = (1.0 - pre_frac) / total_steps
            pre = None
            if pre_full is not None:
                pre = {a: (Kf[:, tr][:, :, tr], None if sd is None else sd[:, tr]) for a, (Kf, sd) in pre_full.items()}
            res = grid_search(sub_train, y[tr], model, alphas, lams, Cs, epsilons, nus,
                              gammas=gammas, folds=inner_folds,
                              seed=inner_seed, optimize_h=optimize_h, mkl=mkl, mkl_rho=mkl_rho,
                              alpha_fields=alpha_fields,
                              progress=(lambda msg, fr: progress(f"repeat {rep + 1}/{repeats}, fold {k + 1}/{outer_folds}: {msg}", base + span * fr)) if progress else None,
                              cancel=cancel, lam_search=lam_search, precomputed=pre,
                              alpha_search=alpha_search, nm_start=nm_start)
            model.fit(sub_train, y[tr], cv_folds=0, reuse_kernels=True)
            fold_hyper.append((float(res.best_alpha), dict(res.best_params)))
            if pre_full is not None and model.is_linear:
                if res.best_alpha in pre_full:
                    terms = [(res.best_alpha, 1.0)]
                else:   # alpha between grid points (Nelder-Mead): interpolated kernels
                    terms = interp_terms(res.best_alpha, sorted(pre_full))
                Kf_all = sum(c * pre_full[a][0] for a, c in terms)
                preds[rep, te] = model.predict_from_field_kernels(Kf_all[:, te][:, :, tr])
            else:
                preds[rep, te] = model.predict(train.subset(te))
            step += 1
        per_repeat.append({"q2": r2_score(y, preds[rep]), "rmse": rmse(y, preds[rep]), "hyper": fold_hyper})
    q2s = np.array([d["q2"] for d in per_repeat])
    rs = np.array([d["rmse"] for d in per_repeat])
    y_mean = preds.mean(axis=0)
    return NestedCVResult(q2_ex=float(q2s.mean()), rmse_ex=float(rs.mean()),
                          q2_ex_sd=float(q2s.std(ddof=1)) if repeats > 1 else float("nan"),
                          rmse_ex_sd=float(rs.std(ddof=1)) if repeats > 1 else float("nan"),
                          y_pred=y_mean, per_repeat=per_repeat,
                          q2_ex_aggr=r2_score(y, y_mean), rmse_ex_aggr=rmse(y, y_mean))
