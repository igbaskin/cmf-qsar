"""Multi-instance learning over conformations: let the model choose the pose.

Averaging a molecule's conformations into one mean field, which is what the 4D mode of
:mod:`cmfqsar.conformers` does, assumes that every conformation deserves the same weight.  On an
aligned series that assumption is wrong and measurably harmful: the activity is produced by one
conformation, the bioactive one, and the rest dilute it.

The multi-instance formulation says so directly.  A molecule is a *bag* of conformations, the label
belongs to the bag, and the model decides which conformations in it matter.  Here the decision is a
weight per conformation,

    w_ik = exp(beta * f(x_ik)) / sum_l exp(beta * f(x_il)),

driven by the model's own prediction for that conformation, and the molecule is represented as
before by the weighted mean of its conformational fields.  Two limits bracket everything:

* ``beta = 0``   every conformation weighs the same: the ensemble mode exactly.
* ``beta -> inf`` all the weight goes to the single highest-scoring conformation: hard selection,
  the witness assumption of mi-SVM (Andrews, Tsochantaridis & Hofmann 2002) and of multi-instance
  regression (Ray & Page 2001).

In between the weights are soft, which keeps the whole scheme inside the CMF framework: the bag
kernel stays a weighted mean of conformation kernels,

    K(A, B) = sum_k sum_l w_ak w_bl K(X_A^(k), X_B^(l)) = (W K_conf W^T)_{AB},

so it stays positive definite, the model stays linear in the mean field, and co-fields, the
decomposition rules and the Shapley machinery survive exactly as they do for a fixed weighting.
Hard selection is the one limit that leaves that guarantee intact only because a one-hot weighting
is still a weighting.

Fitting alternates, as multi-instance methods do: fit the regression with the current weights,
score every conformation with it, update the weights, repeat.  The conformation kernel is computed
once and every round after that is matrix algebra on it.  The objective is not convex, so the
result depends on where it starts; the default starts from the pose the molecules arrived in, which
makes the single-conformation model the starting point and the reference to beat.

What this module does *not* do is pretend the choice is free: the weights are fitted to the
training activities, so they must be refitted inside every cross-validation fold, which
:func:`cv_predict` does, and a q2 obtained any other way is not comparable with the
single-conformation one.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .fields import AtomTable, FieldSet, MoleculeFields, build_fields
from .model import _kfold_indices, _Learner, r2_score, rmse


# --------------------------------------------------------------------------- #
# Bags of conformations
# --------------------------------------------------------------------------- #

def flatten_conformers(mols: Sequence, fieldset: FieldSet, names: Sequence[str],
                       max_conformers: int = 0) -> Tuple[List[MoleculeFields], np.ndarray]:
    """Every conformation of every molecule as its own entry, plus the bag it belongs to.

    The field set is used in its single-conformation sense here, whatever its mode: the
    purpose is that the conformations are kept apart so that the model can weigh them.
    """
    from rdkit import Chem
    single = fieldset.copy()
    single.conformers = "single"
    out: List[MoleculeFields] = []
    bags: List[int] = []
    for i, (m, nm) in enumerate(zip(mols, names)):
        ids = [c.GetId() for c in m.GetConformers()]
        if max_conformers and len(ids) > max_conformers:
            ids = ids[:max_conformers]
        if not ids:
            continue
        for j, cid in enumerate(ids):
            one = Chem.Mol(m)
            keep = Chem.Conformer(m.GetConformer(cid))
            one.RemoveAllConformers()
            one.AddConformer(keep, assignId=True)
            out.append(build_fields(one, single, f"{nm}#{j}"))
            bags.append(i)
    return out, np.asarray(bags, dtype=int)


def weight_matrix(weights: np.ndarray, bags: np.ndarray, n_bags: int) -> np.ndarray:
    """(n_bags, n_conformations) matrix W with the weights of each bag in its own row."""
    W = np.zeros((n_bags, len(bags)))
    W[bags, np.arange(len(bags))] = np.asarray(weights, float)
    return W


def uniform_weights(bags: np.ndarray) -> np.ndarray:
    counts = np.bincount(bags, minlength=bags.max() + 1 if len(bags) else 1).astype(float)
    return 1.0 / np.maximum(counts[bags], 1.0)


def first_conformer_weights(bags: np.ndarray) -> np.ndarray:
    """All the weight on the first conformation of each bag: the pose the molecule arrived in."""
    w = np.zeros(len(bags))
    seen = set()
    for i, b in enumerate(bags):
        if b not in seen:
            seen.add(int(b))
            w[i] = 1.0
    return w


def softmax_weights(scores: np.ndarray, bags: np.ndarray, beta: float) -> np.ndarray:
    """Weights within each bag from its conformation scores; beta = 0 is uniform."""
    scores = np.asarray(scores, float)
    out = np.zeros(len(scores))
    if beta <= 0:
        return uniform_weights(bags)
    for b in np.unique(bags):
        m = bags == b
        s = beta * scores[m]
        s = s - s.max()
        e = np.exp(s)
        total = e.sum()
        out[m] = e / total if total > 1e-300 else 1.0 / m.sum()
    return out


def argmax_weights(scores: np.ndarray, bags: np.ndarray) -> np.ndarray:
    """One-hot weights on the highest-scoring conformation of each bag: hard selection."""
    out = np.zeros(len(scores))
    for b in np.unique(bags):
        idx = np.flatnonzero(bags == b)
        out[idx[int(np.argmax(scores[idx]))]] = 1.0
    return out


def bag_kernel(K_conf: np.ndarray, W_rows: np.ndarray, W_cols: Optional[np.ndarray] = None):
    """Kernel between bags from the kernel between conformations: W K W^T."""
    W_cols = W_rows if W_cols is None else W_cols
    return W_rows @ K_conf @ W_cols.T


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #

@dataclass
class MILFit:
    """A fitted multi-instance weighting and the model that goes with it."""
    weights: np.ndarray                  # (n_conformations,) summing to one within each bag
    bags: np.ndarray
    beta: float
    selection: str                       # "soft" | "hard"
    coef: np.ndarray = _field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0
    rounds: int = 0
    moved: List[int] = _field(default_factory=list)      # how many bags changed their best pose
    objective: List[float] = _field(default_factory=list)

    @property
    def n_bags(self) -> int:
        return int(self.bags.max()) + 1 if len(self.bags) else 0

    def W(self) -> np.ndarray:
        return weight_matrix(self.weights, self.bags, self.n_bags)

    def chosen(self) -> np.ndarray:
        """Index of the largest-weight conformation of every bag: the pose the model picked."""
        out = np.zeros(self.n_bags, dtype=int)
        for b in range(self.n_bags):
            idx = np.flatnonzero(self.bags == b)
            out[b] = idx[int(np.argmax(self.weights[idx]))] if len(idx) else -1
        return out

    def effective_counts(self) -> np.ndarray:
        """Exponential of the entropy of each bag's weights: how many conformations matter."""
        out = np.ones(self.n_bags)
        for b in range(self.n_bags):
            w = self.weights[self.bags == b]
            w = w[w > 0]
            out[b] = float(np.exp(-(w * np.log(w)).sum())) if len(w) else 1.0
        return out


def learn_weights(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray, beta: float = 1.0,
                  lam: float = 1.0, rounds: int = 10, selection: str = "soft",
                  init: str = "first", fit_bias: bool = True, method: str = "krr",
                  params: Optional[Dict[str, float]] = None, tol: float = 1e-6,
                  init_weights: Optional[np.ndarray] = None) -> MILFit:
    """Alternate between fitting the model and re-weighting the conformations of each bag.

    `init` is where the alternation starts: ``"first"`` puts all the weight on the conformation a
    molecule arrived with, so the single-conformation model is the starting point, and
    ``"uniform"`` starts from the plain ensemble.  `init_weights` overrides both, which is what
    annealing uses: run with a small beta, hand the weights to a larger one, and so on upwards.
    The iteration stops when no bag changes its highest-weighted conformation and the objective
    settles.

    The objective is not convex and the alternation only finds a local optimum, so where it
    starts matters and is worth varying.
    """
    y = np.asarray(y, float)
    n_bags = int(bags.max()) + 1
    params = dict(params or {})
    params.setdefault("lam", lam)
    if init_weights is not None:
        w = np.asarray(init_weights, float).copy()
    else:
        w = first_conformer_weights(bags) if init == "first" else uniform_weights(bags)
    fit = MILFit(weights=w, bags=np.asarray(bags, int), beta=float(beta), selection=selection)
    previous = None
    for r in range(max(int(rounds), 1)):
        W = weight_matrix(w, bags, n_bags)
        K_bag = bag_kernel(K_conf, W)
        learner = _Learner(method, params, fit_bias).fit(K_bag, y)
        fit.coef, fit.bias = learner.alpha_, learner.b_
        pred_bag = learner.predict(K_bag)
        fit.objective.append(float(rmse(y, pred_bag)))
        # score every conformation with the model fitted on the current bags
        scores = (K_conf @ W.T) @ learner.alpha_ + learner.b_
        w_new = (argmax_weights(scores, bags) if selection == "hard"
                 else softmax_weights(scores, bags, beta))
        moved = int(np.sum(_heaviest(w_new, bags, n_bags) != _heaviest(w, bags, n_bags)))
        fit.moved.append(moved)
        fit.rounds = r + 1
        settled = (previous is not None and abs(previous - fit.objective[-1]) < tol and moved == 0)
        w = w_new
        fit.weights = w
        previous = fit.objective[-1]
        if settled:
            break
    # one last fit so that the stored model matches the stored weights
    W = weight_matrix(w, bags, n_bags)
    learner = _Learner(method, params, fit_bias).fit(bag_kernel(K_conf, W), y)
    fit.coef, fit.bias = learner.alpha_, learner.b_
    fit.objective.append(float(rmse(y, learner.predict(bag_kernel(K_conf, W)))))
    return fit


def anneal_weights(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray,
                   betas: Sequence[float] = (0.5, 1.0, 3.0, 10.0), lam: float = 1.0,
                   rounds: int = 10, selection: str = "soft", init: str = "uniform",
                   **kw) -> MILFit:
    """Raise beta in steps, carrying the weights over, instead of jumping to a hard selection.

    The alternation is a local search, and a hard rule applied to a model that has not settled
    picks the wrong conformation and then trains itself to believe it.  Starting soft keeps every
    conformation as a candidate while the model forms, and sharpening afterwards is the usual
    remedy.
    """
    fit = None
    for beta in betas:
        fit = learn_weights(K_conf, bags, y, beta=float(beta), lam=lam, rounds=rounds,
                            selection=selection, init=init,
                            init_weights=None if fit is None else fit.weights, **kw)
    return fit


def _heaviest(w: np.ndarray, bags: np.ndarray, n_bags: int) -> np.ndarray:
    out = np.zeros(n_bags, dtype=int)
    for b in range(n_bags):
        idx = np.flatnonzero(bags == b)
        if len(idx):
            out[b] = idx[int(np.argmax(w[idx]))]
    return out


def predict_bags(K_new_conf: np.ndarray, fit: MILFit, bags_new: np.ndarray) -> np.ndarray:
    """Predict new bags: score their conformations with the fitted model, weigh, then combine.

    The weighting rule is the one the model was fitted with, applied once: a new molecule has no
    activity to fit its weights to, which is exactly the intention.
    """
    W_train = fit.W()
    scores = (K_new_conf @ W_train.T) @ fit.coef + fit.bias
    w = (argmax_weights(scores, bags_new) if fit.selection == "hard"
         else softmax_weights(scores, bags_new, fit.beta))
    n_new = int(bags_new.max()) + 1 if len(bags_new) else 0
    W_new = weight_matrix(w, bags_new, n_new)
    return (W_new @ K_new_conf @ W_train.T) @ fit.coef + fit.bias


def cv_predict(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray, beta: float = 1.0,
               lam: float = 1.0, folds: int = 10, seed: Optional[int] = 0,
               rounds: int = 10, selection: str = "soft", init: str = "first",
               fit_bias: bool = True, method: str = "krr",
               params: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Cross-validated predictions with the weights relearned inside every fold.

    Splitting is by bag, never by conformation: a molecule is in the training part or in the test
    part as a whole, and its weights are decided by a model that never saw its activity.
    """
    y = np.asarray(y, float)
    n_bags = int(bags.max()) + 1
    folds = max(2, min(int(folds), n_bags))
    out = np.zeros(n_bags)
    for te_bags in _kfold_indices(n_bags, folds, seed):
        te_set = set(int(b) for b in te_bags)
        tr_bags = np.array([b for b in range(n_bags) if b not in te_set], dtype=int)
        tr_mask = np.isin(bags, tr_bags)
        te_mask = ~tr_mask
        remap_tr = {int(b): i for i, b in enumerate(tr_bags)}
        remap_te = {int(b): i for i, b in enumerate(sorted(te_set))}
        bags_tr = np.array([remap_tr[int(b)] for b in bags[tr_mask]], dtype=int)
        bags_te = np.array([remap_te[int(b)] for b in bags[te_mask]], dtype=int)
        sub = learn_weights(K_conf[np.ix_(tr_mask, tr_mask)], bags_tr, y[tr_bags], beta=beta,
                            lam=lam, rounds=rounds, selection=selection, init=init,
                            fit_bias=fit_bias, method=method, params=params)
        pred = predict_bags(K_conf[np.ix_(te_mask, tr_mask)], sub, bags_te)
        for b, p in zip(sorted(te_set), pred):
            out[b] = p
    return out


@dataclass
class MILSearch:
    best_beta: float
    best_lam: float
    best_q2: float
    best_rmse: float
    table: List[dict]
    y_cv: np.ndarray
    fit: MILFit


def search(K_conf: np.ndarray, bags: np.ndarray, y: np.ndarray,
           betas: Sequence[float] = (0.0, 1.0, 3.0, 10.0), lams: Sequence[float] = (0.1, 1.0),
           folds: int = 10, seed: Optional[int] = 0, rounds: int = 10,
           selection: str = "soft", init: str = "first", progress=None) -> MILSearch:
    """Choose beta and lambda by cross-validation, the weights relearned in every fold.

    beta = 0 is the plain ensemble and is always in the grid by default, so the search can always
    fall back to it; hard selection is reached with a large beta or with ``selection="hard"``.
    """
    y = np.asarray(y, float)
    rows, best = [], None
    total = max(len(betas) * len(lams), 1)
    done = 0
    for beta in betas:
        for lam in lams:
            ycv = cv_predict(K_conf, bags, y, beta=beta, lam=lam, folds=folds, seed=seed,
                             rounds=rounds, selection=selection, init=init)
            q2, err = r2_score(y, ycv), rmse(y, ycv)
            rows.append({"beta": float(beta), "lam": float(lam), "q2": q2, "rmse_cv": err})
            done += 1
            if progress:
                progress(f"beta = {beta:g}, lambda = {lam:g}: q2 = {q2:.3f}", done / total)
            if best is None or err < best[1]:
                best = (q2, err, float(beta), float(lam), ycv)
    q2, err, beta, lam, ycv = best
    fit = learn_weights(K_conf, bags, y, beta=beta, lam=lam, rounds=rounds, selection=selection,
                        init=init)
    return MILSearch(best_beta=beta, best_lam=lam, best_q2=q2, best_rmse=err, table=rows,
                     y_cv=ycv, fit=fit)
