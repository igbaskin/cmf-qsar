"""
CMF regression models: kernel ridge regression (KRR, the method of the 2013
paper), kernel partial least squares (KPLS, the kernel form of the PLS that
CoMFA uses), support vector regression and a Gaussian-process view of KRR that
provides predictive variances.  The classifiers are here too: the support-vector
classifiers, the Laplace Gaussian-process classifier of :mod:`cmfqsar.gpc`,
kernel logistic regression (the same mode with its ridge chosen on the folds)
and the kernel Fisher discriminant (KFDA), whose score is linear in the field
and which therefore has a co-field.  Includes k-fold cross-validation, grid search
over hyper-parameters (alpha, lambda / C / number of PLS components, optionally
mixing coefficients h_f) and external (nested) cross-validation.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .fields import AtomTable, FieldSet, FieldSpec, MoleculeFields
from .kernels import (Cancelled, atom_kernel_block, combine_kernels, kernel_matrices,
                      kernel_matrices_multi, normalize_kernels, self_kernels, self_kernels_multi)
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


def mae(y: np.ndarray, yp: np.ndarray) -> float:
    """The mean absolute error."""
    y = np.asarray(y, float)
    yp = np.asarray(yp, float)
    return float(np.mean(np.abs(y - yp)))


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
    #: Spearman, Kendall and the share of pairs ordered correctly; a ranking model has no r2,
    #: because its score has no units and the squared error of one would mean nothing
    rank_fit: Dict[str, float] = field(default_factory=dict)
    rank_cv: Dict[str, float] = field(default_factory=dict)
    rank_test: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, object] = field(default_factory=dict)

    def summary(self) -> str:
        lines = []
        if self.n_train:
            lines.append(f"Training set: n = {self.n_train}")
            if np.isfinite(self.r2):
                lines.append(f"  r2 (fit)        = {self.r2:.3f}    RMSE (fit)  = {self.rmse_fit:.3f}")
            if self.rank_fit:
                lines.append("  " + _rank_line("fit", self.rank_fit))
        if np.isfinite(self.q2):
            lines.append(f"  q2 ({self.cv_folds}-fold CV)  = {self.q2:.3f}    RMSEcv      = {self.rmse_cv:.3f}")
        if self.rank_cv:
            lines.append(f"  " + _rank_line(f"{self.cv_folds}-fold CV", self.rank_cv))
        if self.n_test:
            lines.append(f"Test set: m = {self.n_test}")
            if np.isfinite(self.r2_pred):
                lines.append(f"  R2pred          = {self.r2_pred:.3f}    RMSEpred    = {self.rmse_pred:.3f}")
                lines.append(f"  R2pred (train mean) = {self.r2_pred_train_mean:.3f}    "
                             f"R2pred (train variance) = {self.r2_pred_train_var:.3f}")
            if self.rank_test:
                lines.append("  " + _rank_line("test", self.rank_test))
        if np.isfinite(self.q2_ex):
            sd = f" +/- {self.q2_ex_sd:.3f}" if np.isfinite(self.q2_ex_sd) else ""
            sd2 = f" +/- {self.rmse_ex_sd:.3f}" if np.isfinite(self.rmse_ex_sd) else ""
            lines.append(f"External CV: q2ex = {self.q2_ex:.3f}{sd}    RMSEcvex = {self.rmse_ex:.3f}{sd2}")
            if np.isfinite(getattr(self, "q2_ex_aggr", float("nan"))):
                lines.append(f"  mean prediction over reshuffles: q2ex = {self.q2_ex_aggr:.3f}    "
                             f"RMSEcvex = {self.rmse_ex_aggr:.3f}")
        return "\n".join(lines)


def _rank_line(what: str, r: Dict[str, float]) -> str:
    return (f"{what:14s} rho = {r.get('spearman', float('nan')):+.3f}   "
            f"tau = {r.get('kendall', float('nan')):+.3f}   "
            f"pairs right = {r.get('pairs', float('nan')):.1%}")


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

    The enrichment factor counts the actives above a cut and ignores where they are; RIE
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


def kpls_fit(K: np.ndarray, y: np.ndarray, components: int) -> Tuple[np.ndarray, float, int]:
    """Kernel partial least squares (Rosipal & Trejo 2001) with one response.

    The kernel is centred in the field space and the activity is centred; every component is
    one NIPALS step (one response needs no iteration), after which the kernel and the activity
    residual are deflated by that component's score.  The dual coefficients are
    a = U (T' Kc U)^-1 T' yc on the centred kernel Kc, and are then folded, together with the
    centring, into coefficients on the *uncentred* kernel plus a bias, so that the prediction
    is K(x, train) a + b exactly as for every other learner here.  The coefficients sum to
    zero, as those of kernel ridge regression with a bias term do.

    With the linear CMF kernel this is PLS regression in the space of the fields, that is CoMFA's
    learner with the grid replaced by the continuous field.  Returns (a, b, components used);
    fewer components than asked for are used when the residual or the next score vanishes.
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    components = max(1, int(components))
    ybar = float(np.mean(y))
    yc = y - ybar
    Kc = K - K.mean(axis=0)[None, :] - K.mean(axis=1)[:, None] + K.mean()
    Kd = Kc.copy()
    yr = yc.copy()
    scale = float(np.trace(Kc)) / n if n else 1.0
    tol = 1e-10 * max(scale, 1e-300)
    T, U = [], []
    for _ in range(min(components, max(n - 1, 1))):
        nu = float(np.linalg.norm(yr))
        if nu <= 1e-12 * max(float(np.linalg.norm(yc)), 1e-300):
            break                                  # the activity is fitted exactly
        u = yr / nu
        t = Kd @ u
        nt = float(np.linalg.norm(t))
        if nt <= tol:
            break                                  # nothing of the kernel is left
        t = t / nt
        T.append(t)
        U.append(u)
        Kt = Kd @ t                                # (I - t t') Kd (I - t t'), Kd symmetric
        Kd = Kd - np.outer(t, Kt) - np.outer(Kt, t) + float(t @ Kt) * np.outer(t, t)
        yr = yr - t * float(t @ yr)
    if not T:
        return np.zeros(n), ybar, 0
    T = np.column_stack(T)
    U = np.column_stack(U)
    M = T.T @ Kc @ U
    try:
        c = np.linalg.solve(M, T.T @ yc)
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(M, T.T @ yc, rcond=None)[0]
    a = U @ c
    a = a - a.mean()                               # the centring of the training side
    b = ybar - float(K.sum(axis=0) @ a) / n        # the centring of the test side
    return a, b, int(T.shape[1])


#: how many neighbours every molecule keeps in the graph of the manifold penalty, when the
#: caller asks for no particular number
LAPKRR_NEIGHBOURS = 10
#: weights of the manifold penalty tried by the hyper-parameter search when none are given.
#: Zero is one of them on purpose: it is plain kernel ridge regression on the labelled molecules,
#: so the search can decide that the unlabelled molecules do not help.
DEFAULT_MANIFOLDS = (0.0, 0.01, 0.1, 1.0)


def lapkrr_fit(K: np.ndarray, y: np.ndarray, lam: float, manifold: float = 1.0,
               neighbours: int = LAPKRR_NEIGHBOURS,
               fit_bias: bool = True) -> Tuple[np.ndarray, float]:
    """Kernel ridge regression with a smoothness penalty over labelled *and* unlabelled molecules.

    Manifold regularisation (Belkin, Niyogi & Sindhwani, J. Mach. Learn. Res. 7 (2006)
    2399-2434).  A library being screened, a set of decoys, a batch not yet measured: these
    molecules have fields but no activity, and plain kernel ridge regression cannot use them at
    all.  Here they enter through a second penalty,

        gamma g' L g,        g = K a,

    where g holds the predicted values at all n molecules without the bias term, and L is the
    normalised Laplacian of the graph of field similarity
    (:func:`cmfqsar.spectral.knn_laplacian`).  That term is small when the prediction changes
    slowly between molecules that are close in field space, whether or not they are labelled, so
    the unlabelled molecules say where the prediction may vary and where it may not.  Minimising
    it together with the squared error on the labelled molecules and the usual ridge gives one
    dual solve over all n molecules,

        (J K + lam I + gamma L K) a = J y,

    where J is the diagonal indicator of the molecules that have an activity.  With a bias term
    the stationarity conditions add one row and one column, exactly as in :func:`krr_fit`.

    **The bias stays outside the penalty**, which is what ``g = K a`` says above.  Adding the
    same constant to every activity does not make a model vary faster, so a bias should not be
    charged for.  The distinction matters here, because ``L 1`` is not zero for a normalised
    Laplacian, while it is zero for the unnormalised one.  A bias inside the penalty would
    therefore be charged for being a constant, and the dual above would carry a further
    ``gamma L 1 b`` term in its extra column.

    **The activities of the unlabelled molecules are passed as nan.**  A learner that needs more
    than (K, y) is new in this program, and the alternative -- a second kernel block handed to
    the learner on the side -- would have put the unlabelled molecules outside the training
    table, where the co-field, the molecule contributions and the atom contributions could not
    see them.  The expansion ``f(x) = sum_j a_j K(x, x_j) + b`` runs over all n molecules, so
    the training table has to hold all n; marking the missing activities in y is what keeps the
    model one kernel expansion over one training table and every reader of it unchanged.

    **The assumption.**  gamma g' L g asks the activity to vary smoothly along the data.  An
    activity cliff -- two molecules whose fields are nearly the same and whose activities are
    not -- breaks that assumption, and the penalty then pulls the prediction of both towards
    their average.  How many such pairs a set holds is therefore worth counting before the
    method is used on it (``benchmarks/run_lapkrr.py`` reports the count per set).

    ``manifold = 0`` switches the penalty off; the unlabelled molecules then get coefficients of
    zero and the fit is ridge regression on the labelled molecules alone.
    """
    from .spectral import knn_laplacian

    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    lab = np.isfinite(y)
    if not lab.any():
        raise ValueError("Laplacian-regularised kernel ridge regression needs at least one "
                         "molecule with a measured activity")
    yj = np.where(lab, y, 0.0)
    A = np.zeros((n, n))
    A[lab] = K[lab]                                  # J K
    A[np.diag_indices(n)] += float(lam)              # lam I
    gamma = float(manifold)
    if gamma != 0.0:
        A += gamma * (knn_laplacian(K, neighbours=neighbours) @ K)

    def _solve(M, rhs):
        try:
            return np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(M, rhs, rcond=None)[0]

    if not fit_bias:
        return _solve(A, yj), 0.0
    j = lab.astype(float)
    M = np.zeros((n + 1, n + 1))
    M[:n, :n] = A
    M[:n, n] = j                                     # J 1
    M[n, :n] = j @ K                                 # (J 1)' K
    M[n, n] = float(np.count_nonzero(lab))
    sol = _solve(M, np.concatenate([yj, [float(yj.sum())]]))
    return sol[:n], float(sol[n])


#: folds used to choose the threshold and the probability map of a kernel Fisher discriminant,
#: when the caller asks for no particular number
KFDA_FOLDS = 5


def kfda_fit(K: np.ndarray, y: np.ndarray, lam: float) -> Tuple[np.ndarray, float]:
    """Kernel Fisher discriminant analysis for two classes (Mika et al. 1999).

    The direction in field space along which the two classes are furthest apart relative to
    their own spread.  In the dual it is one linear solve,

        a = (N + lam t I)^-1 (m_+ - m_-),

    where m_c is the mean kernel row of class c, N = K W K is the within-class scatter written
    on the kernel, W = I - 1_+ 1_+' / n_+ - 1_- 1_-' / n_- is the projector that centres each
    class on its own mean, and t = trace(N) / n.

    **What the regularisation is for.**  N has rank at most n - 2, so N on its own is singular
    and the solve has no unique answer: the ridge is what makes the direction defined at all.
    It also bounds the norm of the discriminant, which is the usual control of overfitting --
    without it the quotient being maximised can be made arbitrarily large along directions that
    the training molecules happen to span and new molecules do not.  lam is relative: the ridge
    added is lam times the mean diagonal of N, so one lam grid means the same amount of
    regularisation whatever the size of the kernel, as the lam of kernel ridge regression does
    on a kernel scaled to unit mean diagonal.

    The coefficients are scaled so that the within-class standard deviation of the training
    scores is 1.  A discriminant direction is defined only up to its length, and fixing the
    length this way makes the score readable (it is measured in within-class standard
    deviations) and comparable between fits on different training sets, which is what
    :func:`kfda_calibrate` needs when it pools the scores of several folds.

    The score is K(x, train) a + b, a kernel expansion like every other learner here, so the
    co-field, the molecule contributions, the Shapley values and the applicability domain all
    apply to this classifier without change.  That is what the support-vector classifier cannot
    offer: its decision function is a kernel expansion too, but it is not linear in the field in
    the sense a co-field needs, because the coefficients are the solution of a quadratic program
    on the hinge loss rather than a linear map of the class labels.

    The bias returned puts the decision at the midpoint of the two projected class means, which
    is the Fisher threshold for equal class priors; :func:`kfda_calibrate` replaces it with the
    threshold that scores best out of fold.  Returns (dual coefficients, bias).
    """
    K = np.asarray(K, float)
    y = np.asarray(y, float)
    n = len(y)
    pos = y > 0
    n_p = int(np.count_nonzero(pos))
    n_n = n - n_p
    if n_p == 0 or n_n == 0:
        raise ValueError("the kernel Fisher discriminant needs molecules of both classes")
    # A = K W: every column of K centred on the mean of the columns of its own class
    A = K.copy()
    A[:, pos] -= K[:, pos].mean(axis=1)[:, None]
    A[:, ~pos] -= K[:, ~pos].mean(axis=1)[:, None]
    N = A @ A.T                                 # = K W K, symmetric and non-negative definite
    M = K @ np.where(pos, 1.0 / n_p, -1.0 / n_n)
    t = float(np.trace(N)) / n
    S = N + float(lam) * (t if t > 1e-300 else 1.0) * np.eye(n)
    try:
        a = np.linalg.solve(S, M)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(S, M, rcond=None)[0]
    within = float(a @ N @ a) / max(n - 2, 1)   # variance of the scores inside the classes
    if within > 1e-300:
        a = a / np.sqrt(within)
    centre, _ = kfda_frame(K @ a, pos)
    return a, -centre


def kfda_frame(score: np.ndarray, pos: np.ndarray) -> Tuple[float, float]:
    """(midpoint, half-separation) of the two class means of a discriminant score.

    These two numbers are the units a threshold on this score is kept in: measured from the
    midpoint and divided by the half-separation, the mean active scores +1, the mean inactive
    scores -1, and a cut expressed that way means the same thing for two discriminants fitted on
    different numbers of molecules.  That is what lets :func:`kfda_calibrate` pool the scores of
    several folds, and it is needed: the length of the solve is only fixed up to the within-class
    spread of the molecules it was fitted on, which on a small fold is not the spread of the
    molecules it is scoring.  Fitted on 35 of the 44 training molecules of the GPB set at a weak
    ridge, the out-of-fold scores spread six times as widely as the full fit's, and a cut carried
    over in raw score units then lands in the wrong place.
    """
    s = np.asarray(score, float)
    pos = np.asarray(pos, bool)
    if not pos.any() or not (~pos).any():
        return 0.0, 1.0
    hi, lo = float(np.mean(s[pos])), float(np.mean(s[~pos]))
    half = 0.5 * (hi - lo)
    return 0.5 * (hi + lo), (half if half > 1e-300 else 1.0)


def _balanced_threshold(y: np.ndarray, score: np.ndarray) -> Tuple[float, float]:
    """The cut on `score` with the highest balanced accuracy, and that balanced accuracy.

    Balanced accuracy is the mean of sensitivity and specificity, so the cut it chooses does not
    follow the larger class, which is what a screening set with many more inactives than actives
    would otherwise make every cut do.  A molecule whose score equals the cut counts as active,
    so only cuts strictly between two different scores are tried.

    Balanced accuracy is a step function of the cut, so a whole range of cuts usually reaches the
    best value.  The middle of that range is returned, not its lower edge: the edge sits against
    the score of one molecule and moves with it, while the middle is the most stable point of the
    plateau.
    """
    y = np.asarray(y, float)
    s = np.asarray(score, float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = y > 0
    n_p, n_n = int(np.count_nonzero(pos)), int(np.count_nonzero(~pos))
    if n_p == 0 or n_n == 0:
        return 0.0, float("nan")
    order = np.argsort(s, kind="mergesort")
    s_sorted, p_sorted = s[order], pos[order]
    below_p = np.concatenate([[0], np.cumsum(p_sorted)])        # actives below the cut
    below_n = np.concatenate([[0], np.cumsum(~p_sorted)])       # inactives below the cut
    bal = 0.5 * ((n_p - below_p) / n_p + below_n / n_n)
    # a cut between two equal scores would split molecules the score cannot tell apart
    usable = np.ones(len(bal), bool)
    usable[1:-1] = s_sorted[1:] > s_sorted[:-1]
    bal = np.where(usable, bal, -np.inf)
    best = float(np.max(bal))
    tied = np.flatnonzero(bal >= best - 1e-12)
    k = int(tied[len(tied) // 2])                               # the middle of the plateau

    def cut_at(j: int) -> float:
        if j == 0:
            return float(s_sorted[0]) - 1.0
        if j == len(s_sorted):
            return float(s_sorted[-1]) + 1.0
        return 0.5 * (float(s_sorted[j - 1]) + float(s_sorted[j]))

    return cut_at(k), best


def _logistic_1d(y: np.ndarray, score: np.ndarray, max_iter: int = 100,
                 tol: float = 1e-10) -> Tuple[float, float]:
    """Slope and intercept of p(active) = 1 / (1 + exp(-(slope * score + intercept))).

    Platt's (1999) map from a decision value to a probability, fitted by Newton's method on the
    two parameters.  The targets are smoothed as Platt prescribes -- (n_+ + 1) / (n_+ + 2) for an
    active and 1 / (n_- + 2) for an inactive instead of 1 and 0 -- so that a training set the
    score separates completely gives a finite slope instead of an infinite one.

    The slope is kept at or above zero.  A negative slope would say that a higher score means a
    less active molecule, which contradicts the class the same score is used to report and which
    a set small enough to produce it cannot support; the fit under that constraint is slope zero
    and the base rate for every molecule, which is the honest answer when cross-validation found
    no usable relation between the score and the class.
    """
    y = np.asarray(y, float)
    s = np.asarray(score, float)
    ok = np.isfinite(y) & np.isfinite(s)
    y, s = y[ok], s[ok]
    pos = y > 0
    n_p, n_n = int(np.count_nonzero(pos)), int(np.count_nonzero(~pos))
    if n_p == 0 or n_n == 0 or len(s) < 3:
        return 1.0, 0.0
    t = np.where(pos, (n_p + 1.0) / (n_p + 2.0), 1.0 / (n_n + 2.0))
    # fitted on standardised scores and mapped back, so that the Newton step is well scaled
    mu, sd = float(np.mean(s)), float(np.std(s))
    base = float(np.log(max(np.mean(t), 1e-12) / max(1.0 - np.mean(t), 1e-12)))
    if sd <= 1e-300:
        return 0.0, base
    z = (s - mu) / sd
    w, c = 0.0, 0.0
    for _ in range(int(max_iter)):
        f = w * z + c
        p = 1.0 / (1.0 + np.exp(-np.clip(f, -500.0, 500.0)))
        r = p - t
        g = np.array([float(r @ z), float(np.sum(r))])
        v = p * (1.0 - p)
        H = np.array([[float(v @ (z * z)), float(v @ z)],
                      [float(v @ z), float(np.sum(v))]])
        H = H + 1e-12 * np.eye(2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        w, c = w - float(step[0]), c - float(step[1])
        if float(np.max(np.abs(step))) < tol:
            break
    if w <= 0.0:
        return 0.0, base
    return w / sd, c - w * mu / sd


@dataclass
class KFDACalibration:
    """Where a kernel Fisher discriminant cuts its score, and how it turns it into a probability.

    Both are fitted on scores of molecules the discriminant that scored them had not seen, so
    neither carries the optimism of a threshold read off the training scores.  Both are also
    expressed in the class-mean units of :func:`kfda_frame` rather than in raw score units: the
    scaled score is (raw score - `centre`) / `spread`, which is 0 at the midpoint of the two
    class means and +-1 at the means themselves.
    """
    threshold: float = 0.0                      # cut on the scaled score
    #: p(active) = sigmoid(slope * scaled score + intercept); slope 0 means cross-validation
    #: found no usable relation between the score and the class, so every molecule gets the
    #: base rate of actives
    slope: float = 1.0
    intercept: float = 0.0
    folds: int = 0
    balanced_accuracy: float = float("nan")     # of the out-of-fold scores at that threshold
    scores: Optional[np.ndarray] = None         # the out-of-fold scaled scores themselves
    centre: float = 0.0                         # the final fit's midpoint of the class means
    spread: float = 1.0                         # the final fit's half-separation of those means

    def scale(self, score: np.ndarray) -> np.ndarray:
        """Raw scores K(x, train) a expressed in the class-mean units the threshold is kept in."""
        return (np.asarray(score, float) - self.centre) / self.spread

    def bias(self) -> float:
        """The bias b that puts the chosen threshold at zero, so that the class is sign(K a + b)."""
        return -(self.centre + self.threshold * self.spread)

    def probability(self, score: np.ndarray) -> np.ndarray:
        """p(active) for raw scores K(x, train) a, that is for predictions with the bias removed."""
        f = self.slope * self.scale(score) + self.intercept
        return 1.0 / (1.0 + np.exp(-np.clip(f, -500.0, 500.0)))

    def describe(self) -> str:
        return (f"threshold {self.threshold:+.3f} in class-mean units (0 is the midpoint of the "
                f"two class means, +-1 the means), chosen on {self.folds}-fold out-of-fold "
                f"scores (balanced accuracy {self.balanced_accuracy:.3f} there)")


def kfda_calibrate(K: np.ndarray, y: np.ndarray, lam: float, folds: int = KFDA_FOLDS,
                   seed: Optional[int] = 0,
                   frame: Optional[Tuple[float, float]] = None) -> KFDACalibration:
    """Threshold and probability map of a kernel Fisher discriminant, from out-of-fold scores.

    A discriminant direction says where the classes differ, not where to cut between them: the
    solve of :func:`kfda_fit` is invariant to any shift of the score.  The molecules are split
    into folds, each fold is scored by a discriminant fitted on the other folds, and the cut with
    the best balanced accuracy on those scores is the one kept.  The same scores give the
    one-dimensional logistic map from score to probability.

    Every fold's scores are first put into that fold's own class-mean units
    (:func:`kfda_frame`), which is what makes pooling them sound, and the cut that comes out is
    in those units.  `frame` is the (centre, half-separation) of the fit the threshold will be
    used with, so that the same cut can be turned back into a bias on its raw score; it is
    computed here from a fit on all the molecules when the caller does not supply it.
    """
    sc, used, folds = out_of_fold_scores(K, y, lambda Ktr, ytr: kfda_fit(Ktr, ytr, lam),
                                         folds, seed)
    if not used:
        return KFDACalibration(threshold=0.0, folds=0, scores=None)
    if frame is None:
        a_all, _ = kfda_fit(K, y, lam)
        frame = kfda_frame(K @ a_all, y > 0)
    cut, bal = _balanced_threshold(y, sc)
    slope, intercept = _logistic_1d(y, sc)
    return KFDACalibration(threshold=float(cut), slope=float(slope), intercept=float(intercept),
                           folds=int(folds), balanced_accuracy=float(bal), scores=sc,
                           centre=float(frame[0]), spread=float(frame[1]))


def out_of_fold_scores(K: np.ndarray, y: np.ndarray, fit, folds: int = KFDA_FOLDS,
                       seed: Optional[int] = 0) -> Tuple[np.ndarray, int, int]:
    """Scores of molecules that the fit which scored them had not seen, in class-mean units.

    `fit(K_train, y_train)` returns (dual coefficients, bias) for one fold's training molecules.
    Every fold's scores are put into that fold's own class-mean units (:func:`kfda_frame`) before
    they are pooled, which is what makes pooling them sound: a classifier fitted on four fifths of
    the molecules separates them by a different amount than one fitted on all of them, and a cut
    carried over in raw score units then lands in the wrong place.

    Returns (scores, how many folds contributed, how many folds were asked for).  A fold whose
    training part holds only one class is skipped, and with no usable fold the scores are all nan.
    """
    y = np.asarray(y, float)
    n = len(y)
    folds = max(2, min(int(folds), n))
    sc = np.full(n, np.nan)
    used = 0
    for te in _kfold_indices(n, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        if len(tr) < 4 or not ((y[tr] > 0).any() and (y[tr] <= 0).any()):
            continue
        Ktr = K[np.ix_(tr, tr)]
        a, _ = fit(Ktr, y[tr])
        centre, spread = kfda_frame(Ktr @ a, y[tr] > 0)
        sc[te] = (K[np.ix_(te, tr)] @ a - centre) / spread
        used += 1
    if not np.isfinite(sc).any():
        used = 0
    return sc, used, folds


def klr_threshold(K: np.ndarray, y: np.ndarray, lam: float, folds: int = KFDA_FOLDS,
                  seed: Optional[int] = 0,
                  frame: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
    """Where to cut a kernel logistic regression when p = 0.5 is the wrong place.

    The natural boundary of this learner is f = 0, which is p(active) = 0.5, and that is the
    default because the class it reports and its probability then say the same thing.  It is also
    the boundary that follows the larger class: with one active in twenty, calling a molecule
    inactive is right nineteen times in twenty whatever the fields say, and the balanced accuracy
    of such a model is 0.5.  This returns the bias that puts the best out-of-fold balanced
    accuracy at zero instead, and that accuracy, leaving the probability untouched -- so a model
    using it reports a class and a probability that disagree, which is the price and is stated in
    :meth:`CMFModel.predict_proba`.

    Returns (bias, balanced accuracy out of fold); (0.0, nan) when no fold could be fitted.
    """
    from .gpc import klr_fit

    sc, used, folds = out_of_fold_scores(K, y, lambda Ktr, ytr: klr_fit(Ktr, ytr, lam)[:2],
                                         folds, seed)
    if not used:
        return 0.0, float("nan")
    if frame is None:
        a_all, _, _ = klr_fit(K, y, lam)
        frame = kfda_frame(K @ a_all, np.asarray(y, float) > 0)
    cut, bal = _balanced_threshold(y, sc)
    return -(float(frame[0]) + cut * float(frame[1])), float(bal)


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
        elif self.method == "kpls":
            # the centring plays the part of the bias term, so fit_bias has no effect here
            a, b, used = kpls_fit(K, y, int(self.params.get("components", 5)))
            self.alpha_, self.b_ = a, b
            self.support_ = np.arange(n)
            self.n_components_ = used
        elif self.method == "lapkrr":
            # Laplacian-regularised kernel ridge regression: the molecules without an activity
            # are rows of this kernel and terms of this expansion like any other, and are marked
            # by a nan in y.  Nothing else in the class changes, which is the point of that
            # convention: the coefficients, the bias and the training table keep their meaning.
            a, b = lapkrr_fit(K, y, float(self.params.get("lam", 1.0)),
                              manifold=float(self.params.get("manifold", 1.0)),
                              neighbours=int(self.params.get("neighbours", LAPKRR_NEIGHBOURS)),
                              fit_bias=self.fit_bias)
            self.alpha_, self.b_ = a, b
            self.support_ = np.arange(n)
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
        elif self.method == "kfda":
            # kernel Fisher discriminant: a classifier whose score is a linear functional of the
            # field, so a co-field exists for it; the threshold and the probability map are
            # fitted on out-of-fold scores, not on the training scores
            lam = float(self.params.get("lam", 1.0))
            a, b = kfda_fit(K, y, lam)
            self.alpha_, self.b_ = a, b
            self.support_ = np.arange(n)
            self.classes_ = np.array([-1.0, 1.0])
            if n >= 8 and str(self.params.get("kfda_threshold", "cv")) == "cv":
                cal = kfda_calibrate(K, y, lam,
                                     folds=int(self.params.get("kfda_folds", KFDA_FOLDS)),
                                     seed=int(self.params.get("kfda_seed", 0)),
                                     frame=kfda_frame(K @ a, y > 0))
                if cal.folds and np.isfinite(cal.threshold):
                    self.b_ = cal.bias()
                    self.kfda_ = cal
        elif self.method == "klr":
            # kernel logistic regression: the same Laplace mode the GP classifier finds, with the
            # ridge chosen on the folds instead of from the evidence and no posterior variance.
            # The decision boundary is f = 0, which is p(active) = 0.5, so the class and the
            # probability say the same thing; a threshold rule can be asked for instead.
            from .gpc import klr_fit
            a, b, fit = klr_fit(K, y, float(self.params.get("lam", 1.0)))
            self.alpha_, self.b_ = a, b
            self.support_ = np.arange(n)
            self.classes_ = np.array([-1.0, 1.0])
            self.klr_ = fit
            if n >= 8 and str(self.params.get("klr_threshold", "zero")) == "cv":
                bias, bal = klr_threshold(K, y, float(self.params.get("lam", 1.0)),
                                          folds=int(self.params.get("klr_folds", KFDA_FOLDS)),
                                          seed=int(self.params.get("klr_seed", 0)),
                                          frame=kfda_frame(K @ a, y > 0))
                if np.isfinite(bias) and np.isfinite(bal):
                    self.b_ = bias
                    self.klr_balanced_ = bal
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
        elif self.method == "ranksvm":
            from .ranking import fit_rank
            fit = fit_rank(K, y, C=float(self.params.get("C", 1.0)),
                           gap=float(self.params.get("gap", 0.0)),
                           groups=getattr(self, "groups_", None),
                           max_pairs=int(self.params.get("max_pairs", 20000)),
                           seed=int(self.params.get("seed", 0)))
            # one coefficient per molecule, no bias: a constant changes no order
            self.alpha_, self.b_ = fit.coef, 0.0
            self.support_ = np.flatnonzero(np.abs(fit.coef) > 0)
            self.rank_ = fit
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
        L, Kinv = self._variance_factor(lam)
        if L is not None:
            v = np.linalg.solve(L, K_test_train.T)
            var = k_test_self - np.sum(v * v, axis=0)
        else:
            var = k_test_self - np.einsum("ij,jk,ik->i", K_test_train, Kinv, K_test_train)
        return np.maximum(var, 0.0)

    def predict_covariance(self, K_test_train: np.ndarray,
                           K_test_test: np.ndarray) -> np.ndarray:
        """The whole GP posterior covariance of a block of candidates, not only its diagonal.

        A batch rule needs the off-diagonal part: what a second measurement adds depends on the
        first (plan item MC2).  The factorisation is the one the variance already uses.
        """
        if self.K_train_ is None:
            n = len(K_test_train)
            return np.full((n, n), np.nan)
        lam = float(self.params.get("lam", 1.0))
        L, Kinv = self._variance_factor(lam)
        if L is not None:
            v = np.linalg.solve(L, np.asarray(K_test_train, float).T)
            cov = np.asarray(K_test_test, float) - v.T @ v
        else:
            K = np.asarray(K_test_train, float)
            cov = np.asarray(K_test_test, float) - K @ Kinv @ K.T
        return 0.5 * (cov + cov.T)

    def _variance_factor(self, lam: float):
        """The factorisation of ``K_train + lam I``, computed once and kept.

        Every variance costs a solve against this factor, and the factorisation costs a cube of
        the training set.  Alignment by prediction confidence asks for a variance at every pose it
        tries, and a screening run asks for one per block of the library, so factorising on each
        call was the whole cost of both.  The cache is keyed on the shape, the ridge and a
        checksum of the matrix, which is quadratic and therefore free beside what it saves; a
        kernel that changes -- a realignment, another fold -- misses the cache and is refactorised.
        """
        K = self.K_train_
        key = (K.shape, float(lam), float(K[0, 0]), float(K[-1, -1]), float(K.sum()))
        if getattr(self, "_var_key", None) == key:
            return self._var_factor
        n = K.shape[0]
        try:
            got = (np.linalg.cholesky(K + lam * np.eye(n)), None)
        except np.linalg.LinAlgError:
            got = (None, np.linalg.pinv(K + lam * np.eye(n)))
        self._var_key, self._var_factor = key, got
        return got


def _kfold_indices(n: int, folds: int, seed: Optional[int]) -> List[np.ndarray]:
    idx = np.arange(n)
    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
    return [idx[k::folds] for k in range(folds)]


def _cv_splits(y: np.ndarray, folds: int,
               seed: Optional[int]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """(left-out indices, training indices) of every fold.

    With every activity measured this is the plain k-fold split: the folds of
    :func:`_kfold_indices` and their complements.  A molecule whose activity is nan has nothing
    to predict and is put in the **training part of every fold**, which is how a semi-supervised
    learner (``--method lapkrr``) uses it: the unlabelled molecules are part of the model being
    cross-validated, not part of what is being held out.
    """
    y = np.asarray(y, float)
    n = len(y)
    lab = np.flatnonzero(np.isfinite(y))
    if len(lab) == n:
        folds = max(2, min(folds, n))
        return [(te, np.setdiff1d(np.arange(n), te))
                for te in _kfold_indices(n, folds, seed)]
    folds = max(2, min(folds, max(len(lab), 2)))
    return [(lab[k], np.setdiff1d(np.arange(n), lab[k]))
            for k in _kfold_indices(len(lab), folds, seed)]


def cv_predict(K: np.ndarray, y: np.ndarray, method: str, params: Dict[str, float],
               folds: int = 10, seed: Optional[int] = 0, fit_bias: bool = True,
               groups: Optional[Sequence] = None) -> np.ndarray:
    """k-fold cross-validated predictions on a precomputed joint kernel matrix.

    ``seed=None`` uses the deterministic interleaved folds (molecule i goes to fold i mod k)
    of the original R scripts; an integer seed shuffles the molecules first.
    """
    n = len(y)
    yp = np.full(n, np.nan)
    g = None if groups is None else np.asarray(groups)
    for te, tr in _cv_splits(y, folds, seed):
        learner = _Learner(method, params, fit_bias)
        if g is not None:
            learner.groups_ = g[tr]            # the pairs of a fold stay inside their groups
        learner = learner.fit(K[np.ix_(tr, tr)], y[tr])
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
    method: str = "krr"                     # krr | gp | kpls | lapkrr | svr | nusvr | svc | nusvc | kfda | klr | ocsvm
    params: Dict[str, float] = field(default_factory=lambda: {"lam": 1.0, "C": 1.0, "epsilon": 0.1,
                                                              "nu": 0.5, "components": 5})
    normalize: bool = False                 # cosine-normalise each field kernel
    scale_fields: bool = True               # scale each field kernel to unit mean diagonal
    # how far apart two atoms may be and still be summed (plan item B2).  None computes every
    # pair, which is exact and is what every published table in this repository used; "auto" uses
    # a neighbour list for the fields and molecules where it is faster, which is large molecules
    # at a strong attenuation; a number is a radius in angstroms.  A cutoff makes the kernel an
    # approximation whose dropped pairs were below 1e-6 of a peak overlap each.
    cutoff: Optional[object] = None
    # which implementation of the atom-pair sum to use (plan item B5): None for the library
    # default, which is NumPy and is what every table was produced with, "numba" for the compiled
    # threaded loop, "auto" for that where Numba is installed.  The two differ by about 4e-15.
    engine: Optional[str] = None
    # where the kernel comes from: the overlap integrals of the fields, which needs the molecules
    # to share a frame, or the rotation-invariant autocorrelation of those fields, which does not
    kernel_source: str = "fields"           # "fields" | "autocorrelation" | "surface" |
                                            # "periodic" | "reciprocal" | "threebody"
    ac_sigma: float = 0.75                  # autocorrelation: width along the distance axis (A)
    ac_d_max: float = 12.0                  # autocorrelation: longest distance resolved (A)
    ac_chirality: bool = False              # autocorrelation: add the three-point signed block
    ac_chirality_shells: int = 4            # autocorrelation: radial shells of that block
    # "surface": the in-plane autocorrelation resolved by height above a surface at z = 0
    # (invariant to translation in the plane and rotation about the normal; see invariant.py)
    surf_z_min: float = 1.0                 # surface: lowest height channel (A)
    surf_z_max: float = 6.0                 # surface: highest height channel (A)
    surf_z_step: float = 1.0                # surface: spacing and width of the height channels
    # "periodic": the autocorrelation of a crystal, summed over a neighbour list with periodic
    # images, per atom of the cell, with the uniform background removed and a tapered cutoff
    # (ac_d_max is the cutoff radius, ac_sigma the width and spacing; see periodic.py)
    pbc_background: bool = True             # periodic: subtract the uniform-density term
    pbc_taper: float = 2.0                  # periodic: width of the cosine taper below the cutoff
    # the height channels of a layer (plan item M1): with pbc_z_max > 0 a structure periodic in
    # two directions is described by the in-plane distance and the heights of both atoms above
    # the layer's mean plane, every pbc_z_step from -pbc_z_max to pbc_z_max; the two faces are
    # averaged unless pbc_oriented says the third lattice vector is a physical "up"
    pbc_z_max: float = 0.0
    pbc_z_step: float = 1.0
    pbc_oriented: bool = False
    # "reciprocal": the typed structure factors of a crystal, binned by the length of the
    # reciprocal-lattice vector (plan item M6; see reciprocal.py).  It is a sum over the atoms of
    # one cell with no neighbour list, so it carries order of any range, and rec_form_sigma --
    # the width of the Gaussian field -- sets how much of the fine detail reaches it.
    rec_q_max: float = 8.0                  # longest reciprocal length resolved (1/angstrom)
    rec_step: float = 0.25                  # spacing and width of the channels (1/angstrom)
    rec_form_sigma: float = 0.3             # width of the Gaussian field (angstrom): the resolution
    rec_self: bool = True                   # keep the self-scattering term (the composition)
    # "threebody": the three-body channels of a crystal (plan item M10; see threebody.py).  Every
    # atom of the cell is a centre, and the channels say how much field weight sits at two radii
    # from it and how far apart those two neighbours are.  It separates structures that share
    # their distance histogram, which no two-point descriptor can; with tb_chirality on, the
    # signed-volume block separates a structure from its mirror image as well.
    tb_r_max: float = 4.0                   # how far from a centre the radial grid reaches (A)
    tb_step: float = 0.5                    # spacing of the radial and the distance grid (A)
    tb_sigma: float = 0.35                  # width of the window on every grid point (A)
    tb_chirality: bool = False              # add the handedness block
    tb_chirality_shells: int = 4            # radial windows the handedness is resolved over
    # the *overlap* kernel of crystals (plan item M4): the fields of the cell compared point by
    # point with the periodic images summed in, which needs the structures to stand on one parent
    # lattice and gives a co-field inside the cell.  It is the "fields" kernel source with the
    # images added, so the co-field, the per-site contributions and the learners are unchanged.
    # The raw integral is over one cell and so grows with it, which the cosine normalisation
    # removes: this kernel requires normalize=True, except on a table of difference fields
    # (cmfqsar.periodic.defect_table, plan item M14), whose rows are the changed sites only and
    # whose pristine cell is the empty difference with a norm of zero.
    pbc_overlap: bool = False
    # how far the images reach; None takes the radius at which the widest field of the set has
    # fallen to 1e-6 of a peak overlap (cmfqsar.periodic.overlap_cutoff)
    pbc_overlap_r_max: Optional[float] = None
    # the symmetry operations of the host lattice, (rotations, translations) in fractional
    # coordinates (cmfqsar.periodic.host_symmetry): the periodic overlap is then averaged over
    # them, so that two arrangements of the same defects related by a symmetry of the host are
    # the same structure to the kernel (plan item M14).  None keeps the fixed frame.
    pbc_symmetry: Optional[Tuple[np.ndarray, np.ndarray]] = None
    # mixing coefficients over the channel blocks of the three invariant sources, the analogue of
    # the per-field h_f of the overlap kernel (invariant.py, "Mixing coefficients over the channel
    # blocks"): None keeps every block at one; "balanced" gives every kind of block -- within a
    # group of fields, across two groups -- the same share of the kernel; a mapping names the
    # kinds' weights ({"indicator": 0.3, "indicator*physchem": 0.5}); "align" / "alignf" derive
    # them by centred kernel alignment of the per-block kernels, re-derived inside every
    # cross-validation fold as the field mixing is.  The weights multiply the scaled features
    # before the cosine normalisation, so a weight of zero removes the block exactly.
    ac_block_weights: Union[None, str, Dict[str, float]] = None
    # field name -> group; a field not named falls into the default grouping: the continuous
    # indicator fields (cif:<type>) in one group, every other field in the other
    ac_block_groups: Optional[Dict[str, str]] = None
    ac_block_rho: float = 1.0               # alignf: shrinkage towards equal block weights
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

    def _pair_kernels(self, P: AtomTable, Q: AtomTable, specs, progress=None,
                      cancel=None) -> np.ndarray:
        """The overlap kernels of two tables: open boundaries, or periodic ones (item M4).

        The periodic form is the same integral with the images of `Q` summed in, so everything
        above this method -- the normalisation, the scaling, the learners, the co-field -- is
        the same code for a crystal as for a molecule.
        """
        if not self.pbc_overlap:
            return kernel_matrices(P, Q, specs, progress=progress, cancel=cancel,
                                   cutoff=self.cutoff, engine=self.engine)
        from .periodic import periodic_overlap_kernels
        return periodic_overlap_kernels(P, Q, specs, r_max=self.pbc_overlap_r_max,
                                        cutoff=self.cutoff, engine=self.engine,
                                        progress=progress, cancel=cancel,
                                        symmetry=self.pbc_symmetry)

    def _raw_kernels(self, P: AtomTable, Q: AtomTable, progress=None, cancel=None) -> np.ndarray:
        """Per-field kernels at the current alphas (interpolated between grid alphas if needed)."""
        it = self._interp_specs()
        if it is None:
            return self._pair_kernels(P, Q, self.fieldset.specs, progress=progress, cancel=cancel)
        (c_lo, lo), (c_hi, hi) = it
        K = c_lo[:, None, None] * self._pair_kernels(P, Q, lo, progress=progress, cancel=cancel)
        if hi is not None:
            K += c_hi[:, None, None] * self._pair_kernels(P, Q, hi, progress=progress,
                                                          cancel=cancel)
        return K

    def _self_overlaps(self, P: AtomTable, specs) -> np.ndarray:
        """``K_f(i, i)`` of every structure: open boundaries, or periodic ones (item M4)."""
        if not self.pbc_overlap:
            return self_kernels(P, specs)
        from .periodic import periodic_self_overlaps
        return periodic_self_overlaps(P, specs, r_max=self.pbc_overlap_r_max,
                                      symmetry=self.pbc_symmetry)

    def _self_kernels(self, P: AtomTable) -> np.ndarray:
        it = self._interp_specs()
        if it is None:
            return self._self_overlaps(P, self.fieldset.specs)
        (c_lo, lo), (c_hi, hi) = it
        d = c_lo[:, None] * self._self_overlaps(P, lo)
        if hi is not None:
            d += c_hi[:, None] * self._self_overlaps(P, hi)
        return d

    def _atom_block(self, mol: MoleculeFields) -> np.ndarray:
        """Kernels between the atoms of one molecule and every training molecule: (F, N, n).

        The molecule's probe-field coefficients are passed on.  Without them a probe field has no
        Gaussians on the moving side and its whole row comes back as zero, so everything built on
        this block -- the atom contributions, the Shapley values, the gradients -- would describe
        a model with one field fewer than the one that was fitted, and say nothing about it.
        """
        ci = mol.probe_coeffs or None
        it = self._interp_specs()
        if it is None and not ci and self.pbc_overlap and self.pbc_symmetry is not None:
            blk = self._compiled_atom_block(mol)
            if blk is not None:
                return blk
        side = self._training_side()
        if it is None:
            return atom_kernel_block(mol.coords, mol.weights, mol.radii, side,
                                     self.fieldset.specs, ci=ci)
        (c_lo, lo), (c_hi, hi) = it
        blk = c_lo[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii,
                                                      side, lo, ci=ci)
        if hi is not None:
            blk += c_hi[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii,
                                                           side, hi, ci=ci)
        return blk

    def _compiled_atom_block(self, mol: MoleculeFields) -> Optional[np.ndarray]:
        """The block of :meth:`_atom_block` for a symmetry-averaged periodic model, by the
        compiled loop (plan item M14).

        Every atom of `mol` is taken as a structure of its own against the training cells,
        averaged over the host operations at the nearest image.  That is the sum the prediction
        uses, so the contributions add up to the prediction to rounding, where the
        image-expanded training side of the reference path agrees only to its truncation -- and
        it costs milliseconds per cell where the reference path costs seconds.  None when the
        compiled loop is not available or the cells differ; the caller then takes the
        reference path.
        """
        from dataclasses import fields as dc_fields, replace
        from .periodic import _compiled_symmetry_average, overlap_cutoff

        specs = self.fieldset.specs
        on = [f for f, s in enumerate(specs) if s.enabled]
        enabled = [specs[f] for f in on]
        if mol.n_atoms == 0:
            return np.zeros((len(specs), 0, self.train.n_mol))
        r = self.pbc_overlap_r_max
        if r is None:
            radii_max = max(float(np.max(self.train.radii)) if len(self.train.radii) else 2.0,
                            float(np.max(mol.radii)) if len(mol.radii) else 2.0)
            r = overlap_cutoff(enabled, radii_max)
        # the record may be a MoleculeFields or a CrystalFields (its own dataclass, with the
        # cell): only the fields the record has are replaced, and every per-atom one is sliced
        names = {f.name for f in dc_fields(mol)}
        extra = {k: v for k, v in vars(mol).items() if k not in names}
        n = mol.n_atoms
        singles = []
        for i in range(n):
            def one_atom(v):
                return v[i:i + 1] if v is not None and hasattr(v, "__len__") and len(v) == n \
                    else v
            changes = {"name": f"{mol.name}[{i}]", "coords": mol.coords[i:i + 1],
                       "weights": mol.weights[i:i + 1], "radii": mol.radii[i:i + 1],
                       "symbols": [mol.symbols[i]], "atom_types": one_atom(mol.atom_types),
                       "bonds": np.zeros((0, 2), dtype=int), "charges": one_atom(mol.charges),
                       "n_conformers": 1, "conformer_weights": None, "atoms_per_conformer": 0,
                       "member_sizes": None, "member_atom_map": None, "descriptors": None,
                       "probe_coeffs": {},
                       "occupancy": one_atom(getattr(mol, "occupancy", None)),
                       "site_species": one_atom(getattr(mol, "site_species", None)),
                       "second_moments": one_atom(getattr(mol, "second_moments", None))}
            one = replace(mol, **{k: v for k, v in changes.items() if k in names})
            for k, v in extra.items():
                setattr(one, k, v)
            singles.append(one)
        try:
            got = _compiled_symmetry_average(AtomTable.from_molecules(singles), self.train,
                                             enabled, self.pbc_symmetry, float(r), self.engine)
        except ValueError:
            return None
        if got is None:
            return None
        out = np.zeros((len(specs), mol.n_atoms, self.train.n_mol))
        out[on] = got
        return out

    def _training_side(self) -> AtomTable:
        """The training table a kernel pairs against: its cells, or their periodic images (M4).

        For an ordinary model this is the training table itself.  For the periodic overlap the
        images carry the same weights as the atoms they repeat, so an atom contribution and a
        co-field are the same sums with more Gaussians in them.
        """
        if not self.pbc_overlap:
            return self.train
        from .periodic import expanded_cell_table, overlap_cutoff, symmetry_expanded_table

        r = self.pbc_overlap_r_max
        if r is None:
            radii_max = float(np.max(self.train.radii)) if len(self.train.radii) else 2.0
            r = overlap_cutoff([s for s in self.fieldset.specs if s.enabled], radii_max)
        side = self.train
        if self.pbc_symmetry is not None:
            from .periodic import nearest_image_cutoff
            # the kernel of a symmetry-averaged model is evaluated at the nearest image with
            # the cutoff capped at half the cell; the images of the training side reach the
            # same radius, so that the contributions and the prediction agree to its truncation
            try:
                r = nearest_image_cutoff(side, float(r))
            except ValueError:
                pass
            side = symmetry_expanded_table(side, *self.pbc_symmetry)
        return expanded_cell_table(side, float(r))

    def _atom_self_block(self, mol: MoleculeFields) -> np.ndarray:
        """Kernels between the atoms of one molecule and each other: shape (F, N, N).

        The field of a subset of atoms is again a field, and the kernel is bilinear, so the
        self-kernel of any subset is the sum of this block over the subset. Used by the Shapley
        decomposition, which needs K(X_S, X_S) for coalitions S of atoms.
        """
        if self.pbc_overlap:
            raise ValueError(
                "a coalition of atoms of a periodic cell is not a structure: removing one site "
                "removes it from every image of the cell as well, so the self-kernel of a subset "
                "is not the kernel of anything this model was fitted on. The co-field and the "
                "per-site contributions are defined and are what to read instead.")
        n = mol.n_atoms
        singles = AtomTable(coords=mol.coords, weights=mol.weights, radii=mol.radii,
                            offsets=np.arange(n + 1), names=[str(i) for i in range(n)],
                            molecules=[])
        # every atom is a molecule of its own here, in the molecule's own atom order, so the
        # molecule's probe coefficients serve as the expansion coefficients of both sides
        ci = mol.probe_coeffs or None
        it = self._interp_specs()
        if it is None:
            return atom_kernel_block(mol.coords, mol.weights, mol.radii, singles,
                                     self.fieldset.specs, ci=ci, cq=ci)
        (c_lo, lo), (c_hi, hi) = it
        blk = c_lo[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii,
                                                      singles, lo, ci=ci, cq=ci)
        if hi is not None:
            blk += c_hi[:, None, None] * atom_kernel_block(mol.coords, mol.weights, mol.radii,
                                                           singles, hi, ci=ci, cq=ci)
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

    def fresh_copy(self) -> "CMFModel":
        """An unfitted model with every setting of this one: the field set (copied), learner,
        parameters, normalisation, bias, kernel layer and kernel source with its widths,
        heights and block weights.  The nested cross-validation builds its fold models from
        this, so that a model on an invariant kernel is validated as the kernel it is."""
        return CMFModel(fieldset=self.fieldset.copy(), task=self.task, method=self.method,
                        params=dict(self.params), normalize=self.normalize,
                        scale_fields=self.scale_fields, fit_bias=self.fit_bias,
                        kernel_layer=self.kernel_layer, kernel_source=self.kernel_source,
                        ac_sigma=self.ac_sigma, ac_d_max=self.ac_d_max,
                        ac_chirality=self.ac_chirality, ac_chirality_shells=self.ac_chirality_shells,
                        surf_z_min=self.surf_z_min, surf_z_max=self.surf_z_max,
                        surf_z_step=self.surf_z_step, pbc_background=self.pbc_background,
                        pbc_taper=self.pbc_taper, pbc_overlap=self.pbc_overlap,
                        pbc_z_max=self.pbc_z_max, pbc_z_step=self.pbc_z_step,
                        pbc_oriented=self.pbc_oriented,
                        pbc_overlap_r_max=self.pbc_overlap_r_max,
                        pbc_symmetry=self.pbc_symmetry,
                        ac_block_weights=(dict(self.ac_block_weights)
                                          if isinstance(self.ac_block_weights, dict)
                                          else self.ac_block_weights),
                        ac_block_groups=(dict(self.ac_block_groups)
                                         if self.ac_block_groups else None),
                        ac_block_rho=self.ac_block_rho)

    @property
    def on_surface(self) -> bool:
        """True for the surface kernel: in-plane invariant, height kept."""
        return self.kernel_source == "surface"

    @property
    def is_periodic(self) -> bool:
        """True for the periodic kernel: crystals, summed over the neighbour list of the cell."""
        return self.kernel_source == "periodic"

    @property
    def is_reciprocal(self) -> bool:
        """True for the reciprocal-space channels: the typed structure factors of a crystal."""
        return self.kernel_source == "reciprocal"

    @property
    def is_threebody(self) -> bool:
        """True for the three-body channels of a crystal: the triangles around every atom."""
        return self.kernel_source == "threebody"

    def _ac_spec(self):
        from .invariant import AutocorrelationSpec, SurfaceSpec
        if self.is_threebody:
            from .threebody import ThreeBodySpec
            return ThreeBodySpec(r_max=float(self.tb_r_max), step=float(self.tb_step),
                                 sigma=float(self.tb_sigma), chirality=bool(self.tb_chirality),
                                 chirality_shells=int(self.tb_chirality_shells))
        if self.is_reciprocal:
            from .reciprocal import ReciprocalSpec
            return ReciprocalSpec(q_max=float(self.rec_q_max), step=float(self.rec_step),
                                  sigma=float(self.rec_step),
                                  form_sigma=float(self.rec_form_sigma),
                                  include_self=bool(self.rec_self))
        if self.is_periodic:
            from .periodic import PeriodicSpec
            return PeriodicSpec(r_max=float(self.ac_d_max), step=float(self.ac_sigma),
                                sigma=float(self.ac_sigma), background=bool(self.pbc_background),
                                taper=float(self.pbc_taper), z_max=float(self.pbc_z_max), z_step=float(self.pbc_z_step), oriented=bool(self.pbc_oriented))
        if self.on_surface:
            return SurfaceSpec(d_max=float(self.ac_d_max), step=float(self.ac_sigma),
                               sigma=float(self.ac_sigma), z_min=float(self.surf_z_min),
                               z_max=float(self.surf_z_max), z_step=float(self.surf_z_step),
                               z_sigma=float(self.surf_z_step))
        return AutocorrelationSpec(d_max=float(self.ac_d_max), step=float(self.ac_sigma),
                                   sigma=float(self.ac_sigma),
                                   chirality=bool(self.ac_chirality),
                                   chirality_shells=int(self.ac_chirality_shells))

    def _ac_field_count(self, n_fields: int) -> int:
        """Field columns the autocorrelation works with.

        A probe field has no atom weight column of its own: it is expanded into one channel per
        Gaussian term (:func:`cmfqsar.invariant.expanded_weights`), so a field set with probes
        has more channels than the atom table has columns.
        """
        fs = getattr(self, "fieldset", None)
        if (fs is not None and not self.is_threebody and not self.is_reciprocal
                and not self.is_periodic and not self.on_surface
                and any(s.is_probe for s in fs.specs)):
            from .invariant import expanded_field_count
            return expanded_field_count(fs)
        return int(n_fields)

    def _ac_blocks(self, n_fields: int):
        from .invariant import block_sizes, surface_block_sizes
        spec = self._ac_spec()
        if self.is_threebody:
            from .threebody import threebody_block_sizes
            return threebody_block_sizes(spec, n_fields)
        if self.on_surface:
            return surface_block_sizes(spec, n_fields)
        return block_sizes(spec, self._ac_field_count(n_fields))

    def _ac_features(self, table: AtomTable) -> np.ndarray:
        from .invariant import feature_matrix, surface_feature_matrix
        if self.is_threebody:
            from .threebody import threebody_feature_matrix
            return threebody_feature_matrix(table.molecules, self._ac_spec(),
                                            n_fields=table.weights.shape[1])
        if self.is_reciprocal:
            from .reciprocal import reciprocal_feature_matrix
            return reciprocal_feature_matrix(table.molecules, self._ac_spec(),
                                             n_fields=table.weights.shape[1])
        if self.is_periodic:
            from .periodic import periodic_feature_matrix
            fn = periodic_feature_matrix
        else:
            if self.on_surface:
                fn = surface_feature_matrix
            else:
                # the field set is needed only to expand a probe field into its Gaussian terms
                return feature_matrix(table.molecules, self._ac_spec(),
                                      n_fields=table.weights.shape[1],
                                      fieldset=getattr(self, "fieldset", None))
        return fn(table.molecules, self._ac_spec(), n_fields=table.weights.shape[1])

    # ----------------------------------------------- mixing coefficients of the channel blocks
    @property
    def block_weights_aligned(self) -> bool:
        """True when the block weights come from the activities by kernel alignment, and so
        have to be re-derived inside every cross-validation fold."""
        from .invariant import ALIGN_MODES
        return isinstance(self.ac_block_weights, str) and self.ac_block_weights in ALIGN_MODES

    def _ac_block_keys(self, n_fields: int) -> List[str]:
        """The kind of every channel block (which groups of fields it joins), in layout order."""
        from .invariant import block_group_keys
        names = []
        for sp in self.fieldset.specs:
            # a probe field occupies one column per Gaussian term, and every one of them belongs
            # to the same group, so its name is repeated rather than numbered
            names += [sp.name] * (int(self.fieldset.probe_terms) if sp.is_probe else 1)
        if len(names) != n_fields:
            names = [f"field{i}" for i in range(n_fields)]
        if self.is_threebody:
            return block_group_keys(names, self.ac_block_groups,
                                    chirality=bool(self.tb_chirality))
        chir = bool(self.ac_chirality) and not (self.on_surface or self.is_periodic
                                                or self.is_reciprocal)
        return block_group_keys(names, self.ac_block_groups, chirality=chir)

    def _derive_block_weights(self, X_scaled: np.ndarray, y: Optional[np.ndarray],
                              blocks: Sequence[int], keys: Sequence[str]) -> Optional[np.ndarray]:
        """The block weights of this model on these (scaled) training features, or None when
        every block keeps the weight one."""
        from .invariant import align_block_weights, group_block_weights
        if self.ac_block_weights is None:
            return None
        if self.block_weights_aligned:
            if y is None:
                raise ValueError("the aligned block weights are derived from the activities, "
                                 "which this kernel was asked for without them")
            return align_block_weights(X_scaled, y, blocks, mode=self.ac_block_weights,
                                       rho=self.ac_block_rho)
        return group_block_weights(keys, self.ac_block_weights)

    @property
    def ac_column_scale(self) -> np.ndarray:
        """What every channel column is multiplied by before the kernel is taken: the
        root-mean-square scale of its block times the block's mixing coefficient."""
        from .invariant import expand_block_weights
        sc = np.asarray(self.ac_scale_, float)
        w = getattr(self, "ac_block_weights_", None)
        return sc if w is None else sc * expand_block_weights(w, self.ac_blocks_)

    def block_weight_report(self) -> List[Dict[str, float]]:
        """The fitted block weights by kind of block (see :func:`invariant.block_weight_summary`);
        empty for a model whose blocks all weigh one."""
        from .invariant import block_weight_summary
        w = getattr(self, "ac_block_weights_", None)
        keys = getattr(self, "ac_block_keys_", None)
        if w is None or keys is None:
            return []
        return block_weight_summary(keys, w)

    def training_kernel(self, train: AtomTable, y: Optional[np.ndarray] = None, progress=None,
                        cancel=None, reuse: bool = False) -> np.ndarray:
        """The joint training kernel, whichever source it comes from.

        ``y`` is needed by the invariant sources when the block weights are set by kernel
        alignment.  With ``reuse`` the channel features of a table this model has already
        featurised are kept; the scale and the block weights are always derived afresh."""
        if self.uses_fields:
            return self.joint_kernel(self.prepare_training_kernels(train, progress=progress,
                                                                   cancel=cancel),
                                     train.descriptors)
        from .invariant import autocorrelation_kernel, channel_scale
        if progress:
            progress("Three-body channels" if self.is_threebody else
                     "Reciprocal channels" if self.is_reciprocal else
                     "Periodic autocorrelation" if self.is_periodic else
                     "Surface autocorrelation" if self.on_surface else "Field autocorrelation", 0.2)
        n_fields = train.weights.shape[1]
        if not (reuse and self.train is train and getattr(self, "ac_train_", None) is not None
                and getattr(self, "ac_blocks_", None) is not None):
            self.ac_train_ = self._ac_features(train)
            self.ac_blocks_ = self._ac_blocks(n_fields)
        self.train = train
        self.self_diag = None
        self.K_fields_train = None
        self.scale = np.ones(1)
        spec = self._ac_spec()
        self.ac_scale_ = channel_scale(self.ac_train_, spec.n_bins, blocks=self.ac_blocks_)
        self.ac_block_keys_ = self._ac_block_keys(self._ac_field_count(n_fields))
        self.ac_block_weights_ = self._derive_block_weights(
            self.ac_train_ * self.ac_scale_, None if y is None else np.asarray(y, float),
            self.ac_blocks_, self.ac_block_keys_)
        return autocorrelation_kernel(self.ac_train_, scale=self.ac_column_scale,
                                      n_bins=spec.n_bins, blocks=self.ac_blocks_)

    def test_kernel(self, test: AtomTable, progress=None, cancel=None) -> np.ndarray:
        """The (n_test, n_train) joint kernel block, whichever source it comes from."""
        if self.uses_fields:
            return self.joint_kernel(self.test_kernels(test, progress=progress, cancel=cancel),
                                     test.descriptors)
        from .invariant import autocorrelation_kernel
        if getattr(self, "ac_train_", None) is None:
            raise RuntimeError("model is not fitted")
        return autocorrelation_kernel(self.ac_train_, self._ac_features(test),
                                      scale=self.ac_column_scale, n_bins=self._ac_spec().n_bins,
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
                f"which is invariant to rotation"
                f"{' about the surface normal' if self.on_surface else ''} and so has no field "
                f"in a fixed frame to show. "
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
        atom contributions, this is what shows at a glance whether a model rests on its
        geometry or on a descriptor that happens to follow the activity.
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
        # open boundaries, or the periodic self-overlaps with the host symmetry (items M4 and
        # M14), which the training diagonal was built from and which matters whenever the
        # diagonal is not replaced by one, that is for a model that is not normalised.  At the
        # attenuation itself, not interpolated between grid points: the gradients take the
        # self-kernel there too (tests/test_gradients.py), and the two must agree.
        d = self._self_overlaps(table, self.fieldset.specs)
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

        ::

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
        if (not self.uses_fields and self.block_weights_aligned
                and getattr(self, "ac_train_", None) is not None):
            return block_cv_predict(self, y, self.params,
                                    block_fold_kernels(self, y, folds=folds, seed=seed))
        return cv_predict(K, y, self.method, self.params, folds=folds, seed=seed,
                          fit_bias=self.fit_bias, groups=getattr(self, "rank_groups", None))

    def layered_kernel(self, K_fields: np.ndarray, diag_p: np.ndarray, diag_q: np.ndarray) -> np.ndarray:
        return self.apply_layer(self.joint_kernel(K_fields), diag_p, diag_q)

    # ---------------------------------------------------------------- fitting
    def prepare_training_kernels(self, train: AtomTable, progress=None, cancel=None) -> np.ndarray:
        """Compute (and cache) the per-field training kernel matrices for the current alphas."""
        if self.pbc_overlap and not self.normalize:
            from .periodic import is_difference_table
            if not is_difference_table(train):
                raise ValueError(
                    "the periodic overlap kernel is an integral over one cell, so its raw value "
                    "grows with the cell and ranks structures by how large a cell they were "
                    "written in. Fit it with normalize=True, which divides that out exactly. "
                    "(A table of difference fields, cmfqsar.periodic.defect_table, is exempt: "
                    "a difference is local and does not grow with the cell.)")
        self.train = train
        self.self_diag = self._self_kernels(train) if self.normalize else None
        self.K_fields_train = self._field_kernels(train, train, self.self_diag, self.self_diag,
                                                  progress=progress, cancel=cancel)
        self.scale = self._compute_scale(self.K_fields_train)
        self.prepare_descriptors(train)
        return self.K_fields_train

    def prepare_training_kernels_grid(self, train: AtomTable, alphas: Sequence[float],
                                      alpha_fields: Optional[Sequence[str]] = None,
                                      progress=None, cancel=None) -> Optional[Dict]:
        """Training kernels for a whole grid of attenuations, sharing the distances (item B1).

        Returns ``{alpha: (K_fields, self_diag)}`` in the form :func:`grid_search` accepts as
        ``precomputed``, or None when the shortcut does not apply: a kernel that is not the
        overlap integral has no attenuation to sweep, and an attenuation that has to be
        interpolated between grid points is two field sets rather than one, which the plain path
        already handles.

        The squared interatomic distances do not depend on the attenuation, so one pass over the
        molecules serves every alpha.  The saving is about a quarter of the time on a six-point
        grid; the larger part of the kernel work is the exponential, which has to be done per
        alpha.
        """
        if not self.uses_fields:
            return None
        values = [a for a in alphas if a is not None]
        if len(values) < 2:
            return None
        spec_sets, keys = [], []
        original = [sp.alpha for sp in self.fieldset.specs]
        try:
            for alpha in values:
                self.fieldset.set_alpha(alpha, alpha_fields)
                if self._interp_specs() is not None:
                    return None                      # interpolated alphas: leave it to the caller
                spec_sets.append([FieldSpec(**vars(sp)) for sp in self.fieldset.specs])
                keys.append(alpha)
        finally:
            for sp, a in zip(self.fieldset.specs, original):
                sp.alpha = a
        Ks = kernel_matrices_multi(train, train, spec_sets, progress=progress, cancel=cancel,
                                   engine=self.engine)
        diags = self_kernels_multi(train, spec_sets) if self.normalize else [None] * len(Ks)
        out = {}
        for alpha, K, d in zip(keys, Ks, diags):
            out[alpha] = (normalize_kernels(K, d, d) if self.normalize else K, d)
        return out

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
            K_lin = self.training_kernel(train, y=y, progress=progress, cancel=cancel,
                                         reuse=reuse_kernels)
        self.y = y
        self.base_diag_ = np.diag(K_lin).copy()
        self._surrogate = None
        self.gamma_scale_ = median_gamma_scale(K_lin) if self.kernel_layer == "gauss" else None
        K = self.apply_layer(K_lin, self.base_diag_, self.base_diag_)
        learner = _Learner(self.method, self.params, self.fit_bias)
        learner.groups_ = getattr(self, "rank_groups", None)
        self.learner_ = learner.fit(K, y)
        self.coef = self.learner_.alpha_
        self.bias = self.learner_.b_
        yfit = self.learner_.predict(K)
        # a molecule without a measured activity (--method lapkrr) is part of the model and has
        # nothing to be scored against, so every statistic below is taken on the labelled ones
        lab = np.isfinite(y) if y is not None and y.size else np.zeros(0, bool)
        n_lab = int(np.count_nonzero(lab))
        if self.task == "oneclass":
            st = ClassStats(task="oneclass", n_train=len(y) if y is not None else K.shape[0])
            st.inlier_fraction = float(np.mean(yfit >= 0))
            st.n_support = int(len(self.learner_.support_)) if self.learner_.support_ is not None else 0
            st.extra["y_fit"] = yfit
        elif self.task == "classification":
            st = ClassStats(task="classification", n_train=n_lab,
                            fit=classification_scores(y[lab], yfit[lab]))
            if cv_folds and cv_folds >= 2 and n_lab >= 3:
                if progress:
                    progress("Cross-validation", 0.9)
                ycv = self._cv_predictions(K, y, cv_folds, cv_seed)
                st.cv = classification_scores(y[lab], ycv[lab])
                st.cv_folds = min(cv_folds, n_lab)
                st.extra["y_cv"] = ycv
            st.extra["y_fit"] = yfit
        elif self.method == "ranksvm":
            # a ranking score has no units, so r2 and RMSE have no meaning; what it
            # has are the order statistics, for the fit and for the molecules left out
            from .ranking import rank_scores
            gap = float(self.params.get("gap", 0.0))
            groups = getattr(self, "rank_groups", None)
            st = ModelStats(n_train=len(y))
            st.rank_fit = rank_scores(y, yfit, gap=gap, groups=groups)
            if cv_folds and cv_folds >= 2 and len(y) >= 3:
                if progress:
                    progress("Cross-validation", 0.9)
                ycv = self._cv_predictions(K, y, cv_folds, cv_seed)
                st.rank_cv = rank_scores(y, ycv, gap=gap, groups=groups)
                st.cv_folds = min(cv_folds, len(y))
                st.extra["y_cv"] = ycv
            st.extra["y_fit"] = yfit
        else:
            st = ModelStats(n_train=n_lab, r2=r2_score(y[lab], yfit[lab]),
                            rmse_fit=rmse(y[lab], yfit[lab]))
            if cv_folds and cv_folds >= 2 and n_lab >= 3:
                if progress:
                    progress("Cross-validation", 0.9)
                ycv = self._cv_predictions(K, y, cv_folds, cv_seed)
                st.q2 = r2_score(y[lab], ycv[lab])
                st.rmse_cv = rmse(y[lab], ycv[lab])
                st.cv_folds = min(cv_folds, n_lab)
                st.extra["y_cv"] = ycv
            st.extra["y_fit"] = yfit
        self.stats = st
        if progress:
            progress("Done", 1.0)
        return self

    # --------------------------------------------------------- conformal intervals
    def _linear_train_kernel(self) -> np.ndarray:
        """The joint training kernel before any nonlinear layer, recomputed if it was not kept."""
        if self.uses_fields:
            if self.K_fields_train is None:
                self.K_fields_train = self._field_kernels(self.train, self.train,
                                                          self.self_diag, self.self_diag)
            return self.joint_kernel(self.K_fields_train, self.train.descriptors)
        return self.training_kernel(self.train)

    def calibrate_conformal(self, level: float = 0.9, normalise: bool = False,
                            beta: float = 1.0):
        """Calibrate a prediction interval on the errors of the cross-validation already run.

        Nothing is recomputed: :meth:`fit` leaves the out-of-fold predictions on the statistics,
        and their absolute errors are exactly the nonconformity scores conformal prediction asks
        for.  The guarantee that comes with them is approximate rather than exact, because those
        errors come from models fitted on a fraction of the training set while the predictions
        will come from the model fitted on all of it; `benchmarks/run_conformal.py` measures how
        much that costs, and on the Sutherland sets the answer is nothing measurable.

        With `normalise` the errors are divided by the distance of their molecule from the rest
        of the training set, so that the interval widens where the model is extrapolating.  It
        is off by default because, measured, it gains no sharpness on these sets.
        """
        from .conformal import calibrate, difficulty_from_kernel

        if self.task != "regression":
            raise ValueError("conformal intervals are implemented for regression; a classifier "
                             "needs prediction sets, which this program does not build")
        ycv = None if self.stats is None else self.stats.extra.get("y_cv")
        if ycv is None:
            raise ValueError("no cross-validated predictions to calibrate on; fit the model with "
                             "cv_folds >= 2")
        difficulty = None
        if normalise:
            if self.is_low_rank:
                raise ValueError("a normalised interval needs the distance of every training "
                                 "molecule from every other, which is the n by n kernel a "
                                 "low-rank model was fitted to avoid; leave the interval "
                                 "unnormalised, which on the sets measured in "
                                 "benchmarks/run_conformal.py is no wider")
            K = self._linear_train_kernel()
            difficulty = difficulty_from_kernel(K, np.diag(K), K)
        self.conformal_ = calibrate(self.y, ycv, level=float(level), difficulty=difficulty,
                                    beta=float(beta), kind="cv")
        return self.conformal_

    def predict_interval(self, test: AtomTable, progress=None, cancel=None):
        """(prediction, lower, upper) for new molecules, using the stored calibration."""
        from .conformal import difficulty_from_kernel

        c = getattr(self, "conformal_", None)
        if c is None:
            raise ValueError("the model carries no calibration; call calibrate_conformal first")
        yp = self.predict(test, progress=progress, cancel=cancel)
        d = None
        if c.difficulty is not None:
            K_train = self._linear_train_kernel()
            K_xt = self.test_kernel(test)
            d = difficulty_from_kernel(K_xt, self.joint_self(test), K_train)
        lo, hi = c.interval(yp, d)
        return yp, lo, hi

    # ------------------------------------------------------------- prediction
    def test_kernels(self, test: AtomTable, progress=None, cancel=None) -> np.ndarray:
        dQ = self._self_kernels(test) if self.normalize else None
        return self._field_kernels(test, self.train, dQ, self.self_diag, progress=progress, cancel=cancel)

    # ----------------------------------------------------------------- low rank

    def fit_low_rank(self, train: AtomTable, y: Sequence[float], landmarks: AtomTable,
                     lam: Optional[float] = None, kind: str = "landmarks",
                     folds: int = 10, seed: Optional[int] = 0, block: int = 256,
                     progress=None) -> "CMFModel":
        """Fit in the space of the landmarks instead of the space of the molecules (item B6).

        The model keeps the landmarks and one weight per feature instead of one coefficient per
        training molecule, and everything a prediction needs follows from those: see
        :mod:`cmfqsar.lowrank`.  The training table is still kept, because it is linear in the
        number of molecules and the provenance and the contract are read off it; what is never
        built is the n by n kernel.
        """
        from . import lowrank as LR
        from .model import r2_score, rmse          # local, to keep the import graph flat

        self._require_linear("the low-rank path")
        if not self.uses_fields:
            raise ValueError("the low-rank path is for the overlap kernel; the invariant "
                             "kernel already has explicit features, which cmfqsar.locallarge "
                             "fits directly")
        y = np.asarray(y, float).ravel()
        lam = float(self.params.get("lam", 1.0) if lam is None else lam)
        self.params["lam"] = lam
        specs = [s for s in self.fieldset.specs if s.enabled]
        if self.normalize or not self.scale_fields:
            scale = np.ones(len(specs))
        else:
            d = np.asarray(self_kernels(train, specs), float)
            scale = np.array([1.0 / m if m > 1e-300 else 0.0 for m in d.mean(axis=1)])
        self.scale = scale
        low = LR.build(landmarks, self.fieldset, scale=scale, normalize=self.normalize,
                       kind=kind)
        g = LR.gram(low, train, y, block=block, progress=progress)
        w, b = LR.fit(g, lam, fit_bias=self.fit_bias)
        self.low_rank_ = low
        self.low_gram_ = g
        self.low_weights_ = np.asarray(w, float)
        self.coef = None
        self.bias = float(b)
        self.train, self.y = train, y
        self._surrogate = None
        self.self_diag = np.asarray(self_kernels(train, specs), float)
        pred, _which = LR.cross_validate(low, train, y, lam, folds=folds, seed=seed,
                                         fit_bias=self.fit_bias, block=block)
        fitted = LR.predict(low.transform(train, block=block), w, b)
        self.stats = ModelStats(n_train=len(y), r2=float(r2_score(y, fitted)),
                                rmse_fit=float(rmse(y, fitted)),
                                q2=float(r2_score(y, pred)), rmse_cv=float(rmse(y, pred)),
                                cv_folds=int(folds))
        self.stats.extra["y_cv"] = pred
        self.stats.extra["y_fit"] = fitted
        self.stats.extra["low_rank"] = dict(kind=kind, landmarks=int(landmarks.n_mol),
                                            rank=int(low.rank))
        return self

    @property
    def is_low_rank(self) -> bool:
        """True when this model was fitted in the space of its landmarks."""
        return getattr(self, "low_rank_", None) is not None

    def low_rank_features(self, test: AtomTable, block: int = 256, progress=None) -> np.ndarray:
        if not self.is_low_rank:
            raise RuntimeError("this model was not fitted in the space of landmarks")
        return self.low_rank_.transform(test, block=block, progress=progress)

    def _low_rank_self(self, test: AtomTable) -> np.ndarray:
        """The joint self-kernel of the molecules, which the variance keeps exact."""
        specs = [s for s in self.fieldset.specs if s.enabled]
        d = np.asarray(self_kernels(test, specs), float)
        if self.normalize:
            d = np.ones_like(d)
        h = np.asarray(self.fieldset.mixing_vector(), float)[:len(specs)]
        return (d.T * (h * np.asarray(self.scale, float)[:len(specs)])).sum(axis=1)

    def predict(self, test: AtomTable, return_fields: bool = False, progress=None, cancel=None):
        if self.is_low_rank:
            yp = (self.low_rank_.transform(test, progress=progress) @ self.low_weights_
                  + self.bias)
            return (yp, None) if return_fields else yp
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

    def predict_with_covariance(self, test: AtomTable) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction and the whole posterior covariance over the candidates (plan item MC2).

        :meth:`predict_with_variance` gives the diagonal of the same matrix.  The rest of it is
        what makes a batch more than a list: two candidates the model is equally unsure about
        may be unsure about *the same thing*, and only the covariance says so.
        """
        mu, _var = self.predict_with_variance(test)
        learner = self.learner_ if (self.learner_ is not None
                                    and self.learner_.K_train_ is not None) else \
            getattr(self, "_var_learner", None)
        if learner is None:
            raise RuntimeError("model is not fitted")
        K_ct = self.test_kernel(test)
        dQ = self._self_kernels(test) if self.normalize else None
        # the field part only, as the similarity filter of cmfqsar.active already does: a
        # non-field descriptor block is defined against the training set, not candidate to
        # candidate
        K_cc = self.joint_kernel(self._field_kernels(test, test, dQ, dQ))
        return mu, learner.predict_covariance(K_ct, K_cc)

    def predict_with_variance(self, test: AtomTable) -> Tuple[np.ndarray, np.ndarray]:
        """Prediction and GP posterior variance (only meaningful for method 'gp'/'krr')."""
        if self.is_low_rank:
            from . import lowrank as LR
            Phi = self.low_rank_.transform(test)
            yp = Phi @ self.low_weights_ + self.bias
            var = LR.variance(self.low_gram_, Phi, float(self.params.get("lam", 1.0)),
                              k_self=self._low_rank_self(test))
            return yp, var
        if not self.is_linear:
            raise ValueError(
                f"the posterior variance is implemented for the linear CMF kernel; this model uses "
                f"the '{self.kernel_layer}' layer. Fit a linear model (the same fields and alpha) to "
                f"obtain variances, for example for alignment by prediction confidence.")
        K = self.test_kernel(test)
        yp = K @ self.coef + self.bias
        if self.learner_ is None or self.learner_.K_train_ is None:
            # kept on the model, because a screening run asks for this once per block of the
            # library and a fresh learner each time would refactorise the training kernel
            learner = getattr(self, "_var_learner", None)
            if learner is None or learner.K_train_ is None:
                learner = _Learner("gp", self.params)
                if self.uses_fields:
                    if self.K_fields_train is None:
                        # a model read from a file stores its training atoms, not their kernels:
                        # compute them again, with the stored self-kernels and scale untouched
                        self.K_fields_train = self._field_kernels(self.train, self.train,
                                                                  self.self_diag, self.self_diag)
                    learner.K_train_ = self.joint_kernel(self.K_fields_train,
                                                         self.train.descriptors)
                else:
                    learner.K_train_ = self.training_kernel(self.train)
                self._var_learner = learner
        else:
            learner = self.learner_
        var = learner.predict_variance(K, self.joint_self(test))
        return yp, var

    # ------------------------------------------- the model as the fast level of a hierarchy
    def surrogate(self):
        """The Gaussian-process reading of this fit, with its scale in the units of the activity.

        Plan item MS1.  It is kept on the model, because a library is predicted in blocks and the
        factorisation of the training kernel is the whole cost; a refit discards it.
        """
        from .surrogate import Surrogate

        if self.is_low_rank:
            raise ValueError("the predictive standard deviation needs the n by n training "
                             "kernel, which a low-rank model was fitted to avoid")
        if not self.is_linear:
            raise ValueError(
                f"the predictive standard deviation is implemented for the linear CMF kernel; "
                f"this model uses the '{self.kernel_layer}' layer")
        if self.method not in ("krr", "gp"):
            raise ValueError("the predictive standard deviation belongs to the ridge fit: "
                             f"method krr or gp, not {self.method}")
        if self.y is None or self.train is None:
            raise RuntimeError("model is not fitted")
        s = getattr(self, "_surrogate", None)
        if s is None:
            s = Surrogate.fit(self._linear_train_kernel(), self.y,
                              float(self.params.get("lam", 1.0)), self.fit_bias)
            self._surrogate = s
        return s

    def predict_with_sd(self, test: AtomTable, with_noise: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """(prediction, predictive standard deviation) in the units of the activity.

        The prediction is the model's own.  The standard deviation is the posterior deviation of
        :meth:`predict_with_variance` put into the units of ``y`` by the noise scale of the fit,
        with the uncertainty of the constant mean added; with ``with_noise`` it is the deviation
        of a new measurement, which is what a tolerance on the answer should be compared with.
        """
        s = self.surrogate()
        return s.predict(self.test_kernel(test), self.joint_self(test), with_noise=with_noise)

    def calibration(self, folds: int = 10, seed: Optional[int] = 0):
        """Whether the predictive standard deviation means what it says, on the folds.

        Returns a :class:`cmfqsar.surrogate.Calibration`: the root mean square of the
        standardised out-of-fold residual (one when calibrated), the shares inside one and two
        standard deviations, and the factor that would correct the deviation.
        """
        from .surrogate import Calibration, cross_validate

        self.surrogate()        # the same refusals
        mean, sd = cross_validate(self._linear_train_kernel(), self.y,
                                  float(self.params.get("lam", 1.0)), folds=folds, seed=seed,
                                  fit_bias=self.fit_bias)
        return Calibration.of(self.y, mean, sd)

    def rank_score(self, test: AtomTable, beta: float = 0.0,
                   larger_is_better: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(score to rank by, prediction, posterior standard deviation) for new molecules.

        With `beta` at zero the score is the prediction itself, which is what ranking a library
        by a regression model has always meant here.  Above zero it is a confidence bound,

            score = mean - beta * sd        (or mean + beta * sd when smaller is better)

        so a molecule is promoted only if the model is both optimistic *and* sure about it.  The
        point is not caution for its own sake: a screening library is by construction dominated
        by molecules unlike the training set, a kernel method returns the training mean presented
        as a prediction for those, and a point estimate cannot tell that answer from a real one.
        Subtracting the spread puts the molecules the model knows nothing about where they belong.

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
        """p(active) for new molecules; the two probabilistic classifiers provide it.

        For the Gaussian-process classifier the latent posterior variance is averaged over, so a
        molecule the model has nothing to say about is reported near one half rather than
        confidently.  For the kernel Fisher discriminant the probability is the logistic map that
        was fitted on the out-of-fold discriminant scores, which has no such term: it reports the
        rate of actives seen at this score during cross-validation, wherever the molecule lies.

        For the discriminant, ``p >= 0.5`` and the reported class are two different cuts, and on
        a set with many more inactives than actives they are far apart.  p = 0.5 is the score at
        which actives and inactives were equally frequent; the class comes from the threshold
        that maximises balanced accuracy, which weighs the two classes equally and therefore
        sits lower.  Read the probability as a frequency and the class as a decision.
        """
        learner = self.learner_
        cal = getattr(learner, "kfda_", None) if learner is not None else None
        if cal is not None:
            return cal.probability(self.predict(test) - float(self.bias))
        klr = getattr(learner, "klr_", None) if learner is not None else None
        if klr is not None:
            # the likelihood at the mode, with no variance to average over.  No self-kernels are
            # passed: that is what would bring the posterior variance in, and this learner has
            # none.  The probability belongs to the latent function, so any bias on the decision
            # does not enter it.
            return klr.predict_proba(self.test_kernel(test))
        if learner is None or getattr(learner, "gpc_", None) is None:
            raise ValueError("probabilities come from the Gaussian-process classifier, kernel "
                             "logistic regression or the kernel Fisher discriminant; fit with "
                             "method='gpc', method='klr' or method='kfda' (the support-vector "
                             "classifiers return a decision value, which is not a probability)")
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
        # the activities of the test set are kept beside its predictions, so that a model file
        # alone is enough to draw observed against predicted (cmfqsar/report.py)
        self.stats.extra["y_test"] = y_test
        if self.task in ("classification", "oneclass"):
            self.stats.n_test = int(ok.sum()) if ok.any() else int(len(yp))
            signs = np.unique(np.sign(y_test[ok])) if ok.any() else np.zeros(0)
            if ok.sum() >= 2 and len(signs) >= 2:
                self.stats.test = classification_scores(y_test[ok], yp[ok])
            self.stats.extra["y_test_pred"] = yp
            return yp
        if self.method == "ranksvm":
            from .ranking import rank_scores
            if ok.sum() >= 3:
                self.stats.n_test = int(ok.sum())
                self.stats.rank_test = rank_scores(y_test[ok], yp[ok],
                                                   gap=float(self.params.get("gap", 0.0)))
            self.stats.extra["y_test_pred"] = yp
            return yp
        if ok.sum() >= 2:
            # the training molecules without an activity (--method lapkrr) have no value to
            # take a mean or a variance of, so the two reference figures use the labelled ones
            y_train = np.asarray(self.y, float)
            y_train = y_train[np.isfinite(y_train)]
            self.stats.n_test = int(ok.sum())
            self.stats.r2_pred = r2_score(y_test[ok], yp[ok])
            self.stats.r2_pred_train_mean = r2_score(y_test[ok], yp[ok], float(np.mean(y_train)))
            self.stats.rmse_pred = rmse(y_test[ok], yp[ok])
            m = int(ok.sum())
            var_train = float(np.var(y_train, ddof=1)) if len(y_train) > 1 else float("nan")
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

        Defined for the overlap-integral kernel, where the decomposition into atom pairs is a
        property of the kernel, and for the rotation-invariant autocorrelation kernel, whose
        channels are also sums over atom pairs (plan item SC6,
        :func:`cmfqsar.invariant.atom_contributions`).  Neither survives a nonlinear layer.
        """
        if self.kernel_source == "autocorrelation":
            return self._invariant_atom_contributions(mol, per_field=per_field)
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

    def _invariant_atom_contributions(self, mol: MoleculeFields,
                                      per_field: bool = False) -> np.ndarray:
        """The same decomposition for the rotation-invariant kernel (plan item SC6).

        The autocorrelation channel is a sum over atom pairs, so the prediction splits over the
        atoms exactly as it does for the overlap kernel; what it cannot give is a co-field, since
        there is no frame.  `per_field` would need the split taken over field *pairs*, which is
        what :func:`cmfqsar.invariant.pair_contributions` reports instead.
        """
        from .invariant import atom_contributions as split, channel_coefficients
        if not self.is_linear:
            raise ValueError(
                "atom contributions are defined only for the linear kernel: with the "
                f"'{self.kernel_layer}' layer the prediction is not a sum over the channels. "
                "The Shapley decomposition works for any kernel.")
        if per_field:
            raise ValueError(
                "the invariant kernel has one channel per *pair* of field types, so there is no "
                "contribution per field; cmfqsar.invariant.pair_contributions gives the split "
                "over those pairs.")
        if getattr(self, "ac_train_", None) is None or self.coef is None:
            raise RuntimeError("model is not fitted")
        want = self.train.weights.shape[1]
        got = np.asarray(mol.weights, float).shape[1]
        if got != want:
            raise ValueError(
                f"this model was trained on {want} fields and the molecule carries {got}: the "
                "channels of the two do not match. Build the molecule's fields from the model's "
                "own field set.")
        c = channel_coefficients(self.ac_train_, self.ac_column_scale, self.coef)
        return split(mol, c, scale=self.ac_column_scale, spec=self._ac_spec())

    def invariant_pair_contributions(self, mol: MoleculeFields) -> List[Tuple[str, float]]:
        """What each pair of field types contributed to one prediction (plan item SC6)."""
        from .invariant import channel_coefficients, pair_contributions
        if self.kernel_source != "autocorrelation":
            raise ValueError("the channel pairs belong to the invariant kernel")
        c = channel_coefficients(self.ac_train_, self.ac_column_scale, self.coef)
        return pair_contributions(mol, c, scale=self.ac_column_scale, spec=self._ac_spec(),
                                  fieldset=self.fieldset)

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
                f"model, or the local co-field of cmfqsar.nonlinearity, which is the co-field of "
                f"the linear model tangent to this one at a given molecule (the contributions of "
                f"the training molecules remain valid for any kernel).")

    def cofield_basis(self, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The Gaussians of every training molecule's field f, before the dual coefficients.

        Returns (centres, coefficients, widths, molecule index): the j-th training molecule
        contributes ``x_j(r) = s_fj sum_m w_fjm exp(-b_jm/2 |r - r_jm|^2)``, so that the co-field
        is ``C_f(r) = sum_j a_j x_j(r)`` and anything else linear in the same basis -- the
        variance of the co-field under noise in the activities, say (:mod:`cmfqsar.cofields`) --
        is built from the same numbers rather than from a second copy of them.
        """
        self._require_linear("co-fields")
        return self.training_field_basis(f)

    def training_field_basis(self, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The same four arrays as :meth:`cofield_basis`, with no condition on the kernel layer.

        The basis describes the normalised training fields, not the model, so it exists for
        every model that has fields.  :mod:`cmfqsar.nonlinearity` builds the local co-field of
        a model with a nonlinear layer from it; :meth:`cofield_basis` is this method behind
        the check that the ordinary co-field is defined.
        """
        self._require_fields("the training field basis")
        spec = self.fieldset.specs[f]
        # for the periodic overlap the basis is the images as well as the cell, so that the
        # co-field drawn on a grid is periodic: the field of a cell is its lattice sum (item M4)
        tr = self._training_side()
        n = tr.n_mol
        s = np.ones(n)
        if self.normalize:
            d = self.self_diag[f]
            s = np.where(d > 1e-300, 1.0 / np.sqrt(np.maximum(d, 1e-300)), 0.0)
        counts = np.diff(tr.offsets)
        per_mol = np.repeat(s, counts)
        index = np.repeat(np.arange(n), counts)
        if spec.is_probe:
            # a probe field puts several Gaussians on every atom, so its co-field is the same sum
            # with every atom repeated once per term of the expansion
            from .probes import basis_widths
            C = tr.probe_coeffs.get(spec.name)
            if C is None:
                return tr.coords, np.zeros(len(per_mol)), np.ones(len(per_mol)), index
            k = C.shape[1]
            return (np.repeat(tr.coords, k, axis=0), (C * per_mol[:, None]).reshape(-1),
                    np.tile(basis_widths(k), tr.coords.shape[0]), np.repeat(index, k))
        b = np.full(len(per_mol), spec.alpha) if spec.kind == "gauss" else spec.alpha / tr.radii ** 2
        return tr.coords, per_mol * tr.weights[:, f], b, index

    def cofield_coefficients(self, f: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centres, coefficients and widths of the Gaussians composing co-field C_f(r)::

            C_f(r) = sum_j a_j s_fj sum_m w_fjm exp(-b_jm/2 |r - r_jm|^2),  (Eq. 9)

        with s_fj = 1/sqrt(K_f(j,j)) when kernels are normalised, else 1.
        The field-scaling constant c_f and mixing coefficient h_f are *not* included.
        For a model whose alpha lies between the points of ``alpha_grid`` (interpolated
        kernels) the co-field is evaluated with the nominal alpha, an approximation.
        """
        if self.is_low_rank:
            from . import lowrank as LR
            # the same three arrays, with the index set changed from training molecules to
            # landmarks; the field scale h_f c_f is already inside the feature weights, so it
            # is divided out here to match what this method returns for an exact model
            c, k, w = LR.cofield_gaussians(self.low_rank_, self.low_weights_, f)
            h = float(self.fieldset.specs[f].h) * float(self.scale[f])
            return c, (k / h if abs(h) > 1e-300 else k), w
        centres, coeffs, widths, index = self.cofield_basis(f)
        return centres, coeffs * np.asarray(self.coef, float)[index], widths

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

    def _low_rank_state(self) -> Optional[dict]:
        """The landmarks, the whitening and the weights, or None for an ordinary model."""
        if not self.is_low_rank:
            return None
        low = self.low_rank_
        g = self.low_gram_
        return dict(landmarks=low.landmarks, whiteners=low.whiteners, weights=low.weights,
                    normalize=low.normalize, self_landmarks=low.self_landmarks, kind=low.kind,
                    w=self.low_weights_,
                    gram=dict(PtP=g.PtP, Pty=g.Pty, Pt1=g.Pt1, n=g.n, sum_y=g.sum_y,
                              yty=g.yty) if g is not None else None)

    def _set_low_rank_state(self, st: Optional[dict]) -> None:
        """Rebuild the low-rank state of a model read from a file."""
        if not st:
            self.low_rank_ = None
            self.low_gram_ = None
            self.low_weights_ = None
            return
        from .lowrank import Gram, LowRank
        specs = [s for s in self.fieldset.specs if s.enabled]
        self.low_rank_ = LowRank(landmarks=st["landmarks"], specs=specs,
                                 whiteners=list(st["whiteners"]), weights=st["weights"],
                                 normalize=bool(st.get("normalize", True)),
                                 self_landmarks=st.get("self_landmarks"),
                                 kind=str(st.get("kind", "landmarks")))
        self.low_weights_ = np.asarray(st["w"], float)
        g = st.get("gram")
        self.low_gram_ = (Gram(PtP=g["PtP"], Pty=g["Pty"], Pt1=g["Pt1"], n=int(g["n"]),
                               sum_y=float(g["sum_y"]), yty=float(g["yty"]))
                          if g else None)

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
            # the threshold and the probability map of a kernel Fisher discriminant: a few
            # numbers that cannot be recovered from the coefficients alone
            "kfda": getattr(self.learner_, "kfda_", None) if self.learner_ is not None else None,
            # the fitted mode of a kernel logistic regression, which is what its probability
            # comes from; the coefficients alone give the class but not p(active)
            "klr": getattr(self.learner_, "klr_", None) if self.learner_ is not None else None,
            "kernel_source": self.kernel_source,
            "ac_sigma": self.ac_sigma,
            "ac_chirality": self.ac_chirality,
            "ac_chirality_shells": self.ac_chirality_shells,
            "ac_d_max": self.ac_d_max,
            "surf_z_min": self.surf_z_min,
            "surf_z_max": self.surf_z_max,
            "surf_z_step": self.surf_z_step,
            "pbc_background": self.pbc_background,
            "pbc_taper": self.pbc_taper,
            "pbc_z_max": self.pbc_z_max,
            "pbc_z_step": self.pbc_z_step,
            "pbc_oriented": self.pbc_oriented,
            "rec_q_max": self.rec_q_max,
            "rec_step": self.rec_step,
            "rec_form_sigma": self.rec_form_sigma,
            "rec_self": self.rec_self,
            "tb_r_max": self.tb_r_max,
            "tb_step": self.tb_step,
            "tb_sigma": self.tb_sigma,
            "tb_chirality": self.tb_chirality,
            "tb_chirality_shells": self.tb_chirality_shells,
            "pbc_overlap": self.pbc_overlap,
            "pbc_overlap_r_max": self.pbc_overlap_r_max,
            "pbc_symmetry": (None if self.pbc_symmetry is None else
                             {"rotations": np.asarray(self.pbc_symmetry[0], float).tolist(),
                              "translations": np.asarray(self.pbc_symmetry[1], float).tolist()}),
            "ac_train": getattr(self, "ac_train_", None),
            "ac_scale": getattr(self, "ac_scale_", None),
            "ac_blocks": getattr(self, "ac_blocks_", None),
            "ac_block_weights": self.ac_block_weights,
            "ac_block_groups": self.ac_block_groups,
            "ac_block_rho": self.ac_block_rho,
            "ac_block_weights_fitted": getattr(self, "ac_block_weights_", None),
            "ac_block_keys": getattr(self, "ac_block_keys_", None),
            # the descriptor block: its standardisation and scale belong to the training set
            "desc_scaler": self.desc_scaler_,
            "desc_train": self.desc_train_,
            "desc_scale": self.desc_scale_,
            # the calibrated interval: a few numbers, and useless to recompute later
            "conformal": getattr(self, "conformal_", None),
            # the landmarks and the weights over them, for a model fitted the low-rank way
            "low_rank": self._low_rank_state(),
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
        m.surf_z_min = float(st.get("surf_z_min", 1.0))
        m.surf_z_max = float(st.get("surf_z_max", 6.0))
        m.surf_z_step = float(st.get("surf_z_step", 1.0))
        m.pbc_background = bool(st.get("pbc_background", True))
        m.pbc_taper = float(st.get("pbc_taper", 2.0))
        m.pbc_z_max = float(st.get("pbc_z_max", 0.0))
        m.pbc_z_step = float(st.get("pbc_z_step", 1.0))
        m.pbc_oriented = bool(st.get("pbc_oriented", False))
        m.rec_q_max = float(st.get("rec_q_max", 8.0))
        m.rec_step = float(st.get("rec_step", 0.25))
        m.rec_form_sigma = float(st.get("rec_form_sigma", 0.3))
        m.rec_self = bool(st.get("rec_self", True))
        m.tb_r_max = float(st.get("tb_r_max", 4.0))
        m.tb_step = float(st.get("tb_step", 0.5))
        m.tb_sigma = float(st.get("tb_sigma", 0.35))
        m.tb_chirality = bool(st.get("tb_chirality", False))
        m.tb_chirality_shells = int(st.get("tb_chirality_shells", 4))
        m.pbc_overlap = bool(st.get("pbc_overlap", False))
        r = st.get("pbc_overlap_r_max")
        m.pbc_overlap_r_max = None if r is None else float(r)
        sym = st.get("pbc_symmetry")
        m.pbc_symmetry = None if sym is None else (np.asarray(sym["rotations"], float),
                                                   np.asarray(sym["translations"], float))
        m.ac_train_ = st.get("ac_train")
        m.ac_scale_ = st.get("ac_scale")
        m.ac_blocks_ = st.get("ac_blocks")
        m.ac_block_weights = st.get("ac_block_weights")
        m.ac_block_groups = st.get("ac_block_groups")
        m.ac_block_rho = float(st.get("ac_block_rho", 1.0))
        m.ac_block_weights_ = st.get("ac_block_weights_fitted")
        m.ac_block_keys_ = st.get("ac_block_keys")
        m.base_diag_ = st.get("base_diag")
        m.gamma_scale_ = st.get("gamma_scale")
        m.train, m.y, m.coef, m.bias = st["train"], st["y"], st["coef"], st["bias"]
        m.conformal_ = st.get("conformal")
        m._set_low_rank_state(st.get("low_rank"))
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
        if any(st.get(k) is not None for k in ("oneclass", "kfda", "klr")):
            m.learner_ = _Learner(st["method"], st["params"])
            if st.get("oneclass") is not None:
                m.learner_.oc_ = st["oneclass"]
            if st.get("kfda") is not None:
                m.learner_.kfda_ = st["kfda"]
            if st.get("klr") is not None:
                m.learner_.klr_ = st["klr"]
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
        shrunk towards equal weights with strength `rho` (rho = 0 is the plain alignf);
      ``mode="evidence"`` instead maximises the marginal likelihood of the mixture under the
        prior of :mod:`cmfqsar.mixing`, which is the Bayesian reading of the same problem: h_f is
        the prior variance of the part of the model that field f carries.  It costs an
        optimisation rather than a least-squares solve, and the same module will give the
        posterior over those weights rather than this point.

    The weights are rescaled so that the mean diagonal of the joint kernel is the same as with
    h_f = 1, which keeps the regularisation grid comparable.  The combination stays linear, so
    co-fields and atom contributions remain valid - that is the difference from a nonlinear
    kernel layer, which can gain accuracy but loses the interpretation.
    """
    K_fields = np.asarray(K_fields, float)
    lab = np.isfinite(np.asarray(y, float))
    if not lab.all():
        # a molecule without an activity says nothing about which field matters, so the
        # alignment with the activities is measured on the labelled molecules alone
        K_fields = K_fields[:, lab][:, :, lab]
        y = np.asarray(y, float)[lab]
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
    if mode == "evidence":
        from .mixing import evidence_mixing
        return evidence_mixing(K_fields, yv, scale, on)
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
    on = [s.enabled for s in model.fieldset.specs]
    f_idx = np.arange(K_fields.shape[0])
    out = []
    for te, tr in _cv_splits(y, folds, seed):
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
    yp = np.full(len(y), np.nan)
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


def block_fold_kernels(model: "CMFModel", y: np.ndarray, folds: int = 10,
                       seed: Optional[int] = 0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-fold kernels of an invariant source whose block weights come from the activities.

    Deriving the weights from every activity and then cross-validating would leak the left-out
    activities into the kernel, exactly as for the field mixing; so the channel scale and the
    block weights are derived from the training part of each fold alone, and the whole
    (n, n) kernel is formed under them.  A list of (test indices, training indices, K).
    """
    from .invariant import (align_block_weights, autocorrelation_kernel, channel_scale,
                            expand_block_weights)
    X = np.asarray(model.ac_train_, float)
    blocks = model.ac_blocks_
    n_bins = model._ac_spec().n_bins
    y = np.asarray(y, float)
    n = len(y)
    folds = max(2, min(folds, n))
    out = []
    for te in _kfold_indices(n, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        sc = channel_scale(X[tr], n_bins, blocks=blocks)
        w = align_block_weights(X[tr] * sc, y[tr], blocks, mode=model.ac_block_weights,
                                rho=model.ac_block_rho)
        K = autocorrelation_kernel(X, scale=sc * expand_block_weights(w, blocks), n_bins=n_bins,
                                   blocks=blocks)
        out.append((te, tr, K))
    return out


def block_cv_predict(model: "CMFModel", y: np.ndarray, params: Dict[str, float],
                     fold_kernels: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]) -> np.ndarray:
    """Cross-validated predictions over the per-fold kernels of :func:`block_fold_kernels`."""
    y = np.asarray(y, float)
    yp = np.zeros(len(y))
    for te, tr, K in fold_kernels:
        dg = np.diag(K).copy()
        Kl = model.apply_layer(K, dg, dg)
        learner = _Learner(model.method, params, model.fit_bias).fit(Kl[np.ix_(tr, tr)], y[tr])
        yp[te] = learner.predict(Kl[np.ix_(te, tr)])
    return yp


@dataclass
class SearchResult:
    best_alpha: float
    best_params: Dict[str, float]
    best_h: Dict[str, float]
    best_rmse: float
    best_q2: float
    table: List[dict]          # rows: alpha, params, rmse_cv, q2
    y_cv: Optional[np.ndarray] = None
    # one attenuation per field, when the per-field descent of cmfqsar/perfield.py ran
    best_alphas: Optional[Dict[str, float]] = None
    per_field: Optional[object] = None       # the PerFieldResult, with its trace


#: the numbers of PLS components tried when none are given
DEFAULT_COMPONENTS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12)


def _param_grid(method: str, lams, Cs, epsilons, nus, components=(), manifolds=()):
    if method in ("krr", "gp"):
        return [{"lam": float(l)} for l in lams]
    if method == "lapkrr":
        # two regularisers: the ridge on the norm of the model and the weight of the smoothness
        # penalty over labelled and unlabelled molecules together.  gamma = 0 is ridge regression
        # on the labelled molecules alone, so the reference rule 1 of the repository asks for is
        # inside the grid and the search can decide that the unlabelled molecules do not help.
        gammas = sorted({float(g) for g in (manifolds or DEFAULT_MANIFOLDS)})
        return [{"lam": float(l), "manifold": g} for l in lams for g in gammas]
    if method == "kpls":
        comps = sorted({max(1, int(c)) for c in (components or DEFAULT_COMPONENTS)})
        return [{"components": c} for c in comps]
    if method == "svr":
        return [{"C": float(C), "epsilon": float(e)} for C in Cs for e in epsilons]
    if method == "nusvr":
        return [{"C": float(C), "nu": float(nu)} for C in Cs for nu in nus]
    if method == "ranksvm":
        return [{"C": float(C)} for C in Cs]
    if method == "svc":
        return [{"C": float(C)} for C in Cs]
    if method == "kfda":
        # the ridge of the discriminant, on the grid kernel ridge regression uses: it is relative
        # to the scale of the within-class scatter, so one grid means the same thing for both
        return [{"lam": float(l)} for l in lams]
    if method == "klr":
        # the ridge on the norm of the latent function, which is what the GP classifier's
        # amplitude is the reciprocal of; chosen on the folds, which is the whole of item E9
        return [{"lam": float(l)} for l in lams]
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
    accuracy when only one class is present in a fold) and the balanced accuracy.  Ranking:
    1 - rho and Spearman's rho, because the score of a ranking model has no units and
    minimising its squared error would be selecting on a quantity that does not exist.
    """
    lab = np.isfinite(np.asarray(y, float))
    if not lab.all():
        # unlabelled molecules (--method lapkrr) have no prediction to be scored against
        y, ycv = np.asarray(y, float)[lab], np.asarray(ycv, float)[lab]
    if model.method == "ranksvm":
        from .ranking import rank_scores
        rho = rank_scores(y, ycv, gap=float(model.params.get("gap", 0.0)),
                          groups=getattr(model, "rank_groups", None))["spearman"]
        if not np.isfinite(rho):
            return 1.0, float("nan")
        return 1.0 - float(rho), float(rho)
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
                nm_maxit: int = 500, select: str = "cv",
                components: Sequence[int] = (),
                manifolds: Sequence[float] = ()) -> SearchResult:
    """Minimise RMSEcv over alpha (shared by all fields unless `alpha_fields` restricts it)
    and the learner's regularisation parameters (``lams`` for KRR/GP, ``Cs`` with ``epsilons``
    or ``nus`` for the support-vector methods, ``components`` for KPLS, which falls back to
    :data:`DEFAULT_COMPONENTS` when empty, ``lams`` together with ``manifolds`` for the
    Laplacian-regularised ridge, which falls back to :data:`DEFAULT_MANIFOLDS`).  Optionally refine the mixing
    coefficients h_f by coordinate descent (beware of model-selection bias), or set them by
    centred kernel alignment (``mkl=True``, see :func:`align_mixing`), in which case the
    reported RMSEcv is reliable because the weights are re-derived inside every fold.

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
    if select not in ("cv", "evidence"):
        raise ValueError(f"unknown selection criterion {select!r}: 'cv' or 'evidence'")
    if select == "evidence" and (model.task != "regression" or model.method not in ("krr", "gp")):
        raise ValueError("the evidence is the marginal likelihood of a Gaussian process, so it "
                         "selects for regression with krr or gp; the support-vector methods and "
                         "the classifiers have no such quantity and are selected by "
                         "cross-validation")
    if model.task == "oneclass":
        raise ValueError(
            "hyper-parameters of a one-class model cannot be selected by cross-validation: "
            "there are no negatives to score against. nu is the expected fraction of outliers and "
            "alpha sets the level of generalisation; choose them directly (see Fig. 5 of Baskin & "
            "Zhokhova 2019) and fit the model.")
    if alpha_search == "per-field":
        # the descent reuses the shared grid's kernels and scores with the evidence; both are
        # what make one width per field affordable at all (see cmfqsar/perfield.py)
        if select != "evidence":
            raise ValueError("one attenuation per field is searched by the evidence: a coordinate "
                             "descent over the widths costs ten times more on a cross-validated "
                             "score, and the full grid 5^5 times more; pass --select evidence")
        if not model.uses_fields:
            raise ValueError("this kernel has no per-field attenuation to search")
        if len(list(alphas)) < 2:
            raise ValueError("the per-field search needs an alpha grid with at least two widths")
        if alpha_fields:
            raise ValueError("--alpha-fields restricts one shared width to some of the fields; "
                             "the per-field search gives every field its own and the two "
                             "restrictions cannot both hold")
    if not model.is_linear:
        if alpha_search == "nelder-mead" or lam_search == "brent":
            raise ValueError("the continuous alpha/lambda searches of the original scripts are "
                             "implemented for the linear kernel only; use the grids instead")
    if mkl and optimize_h:
        raise ValueError("choose either alignment-based multiple kernel learning or the "
                         "coordinate-descent refinement of h, not both")
    grid = _param_grid(model.method, lams, Cs, epsilons, nus, components, manifolds)
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
    if precomputed is None and len([a for a in alpha_list if a is not None]) > 1:
        # one pass over the molecules for the whole grid instead of one per alpha (item B1)
        if progress:
            progress("kernels for every alpha", 0.0)
        precomputed = model.prepare_training_kernels_grid(
            train, alpha_list, alpha_fields, cancel=cancel,
            progress=(lambda i, n: progress(f"kernels for every alpha: {i}/{n}", 0.5 * i / n))
            if progress else None)
    for ia, alpha in enumerate(alpha_list):
        if cancel and cancel():
            raise Cancelled()
        fold_blocks = None           # per-fold kernels of the aligned block weights, if any
        if alpha is not None:
            model.fieldset.set_alpha(alpha, alpha_fields)
        atxt = "table alphas" if alpha is None else f"alpha = {alpha:g}"
        if progress:
            progress(f"{atxt}: computing kernels", ia / n_alpha)
        if not model.uses_fields:
            # the attenuation belongs to the overlap integral; this kernel has its own width
            # along the distance axis, so there is one kernel and the alpha loop runs once
            Kf = None
            K_lin = model.training_kernel(train, y=y, cancel=cancel)
            use_mkl = False
            fold_h = None
            if model.block_weights_aligned:
                if progress:
                    progress(f"{atxt}: block alignment", (ia + 0.9) / n_alpha)
                fold_blocks = block_fold_kernels(model, y, folds=folds, seed=seed)
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
                if select == "evidence":
                    # one factorisation per candidate instead of one per fold, and nothing is
                    # held out; the loss is the negative log evidence so that smaller is still better
                    from .evidence import log_evidence
                    ev = log_evidence(K, y, float(params.get("lam", 1.0)),
                                      fit_bias=model.fit_bias)
                    r, q2, ycv = -ev, ev, None
                elif use_mkl:
                    ycv, _ = mkl_cv_predict(Kf, y, model, params, folds=folds, seed=seed,
                                            fold_h=fold_h)
                    r, q2 = cv_loss(model, y, ycv)
                elif fold_blocks is not None:
                    ycv = block_cv_predict(model, y, params, fold_blocks)
                    r, q2 = cv_loss(model, y, ycv)
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
    pf = None
    if alpha_search == "per-field" and cache and len(model.fieldset.specs) > 1:
        # from the best shared width, one field at a time; the model is left holding the result
        from .perfield import descend
        pf = descend(train, y, model, cache, alpha_list, grid, alpha, params,
                     start_evidence=q2, progress=progress, cancel=cancel)
        params, q2, r = dict(pf.params), pf.log_evidence, -pf.log_evidence
        rows.append({"alpha": float("nan"), **params, "rmse_cv": r, "q2": q2,
                     "per_field": dict(pf.alphas)})
    if progress:
        progress("Search finished", 1.0)
    return SearchResult(best_alpha=float("nan") if alpha is None else alpha, best_params=params, best_h=h_best,
                        best_rmse=r, best_q2=q2, table=rows, y_cv=ycv,
                        best_alphas=None if pf is None else dict(pf.alphas), per_field=pf)


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
              nm_start: Sequence[float] = (0.25, 5.0), inner_seed="same",
              components: Sequence[int] = (),
              manifolds: Sequence[float] = ()) -> NestedCVResult:
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
    lab = np.isfinite(y)
    for rep in range(repeats):
        folds = _cv_splits(y, outer_folds, None if seed is None else seed + rep)
        fold_hyper = []     # (alpha, params) selected in every outer fold
        for k, (te, tr) in enumerate(folds):
            if cancel and cancel():
                raise Cancelled()
            sub_train = train.subset(tr)
            model = model_template.fresh_copy()
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
                              alpha_search=alpha_search, nm_start=nm_start,
                              components=components, manifolds=manifolds)
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
        per_repeat.append({"q2": r2_score(y[lab], preds[rep][lab]),
                           "rmse": rmse(y[lab], preds[rep][lab]), "hyper": fold_hyper})
    q2s = np.array([d["q2"] for d in per_repeat])
    rs = np.array([d["rmse"] for d in per_repeat])
    y_mean = preds.mean(axis=0)
    return NestedCVResult(q2_ex=float(q2s.mean()), rmse_ex=float(rs.mean()),
                          q2_ex_sd=float(q2s.std(ddof=1)) if repeats > 1 else float("nan"),
                          rmse_ex_sd=float(rs.std(ddof=1)) if repeats > 1 else float("nan"),
                          y_pred=y_mean, per_repeat=per_repeat,
                          q2_ex_aggr=r2_score(y[lab], y_mean[lab]),
                          rmse_ex_aggr=rmse(y[lab], y_mean[lab]))
