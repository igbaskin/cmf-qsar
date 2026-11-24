"""The continuous reaction field against its references (plan item PO3, stage 2).

All 11,961 reactions of the Grambow set, reactant and product only (setting (a) of the item).
Two targets, both in kcal/mol and both differences of the electronic energy plus the zero-point
energy: the activation energy ``ea`` and the reaction energy ``dh``.

The description is rotation-invariant, because reactions of different types share no frame.
Every block below is a table with one row per reaction; a model is a Gaussian kernel on a
weighted sum of the squared distances of its blocks, fitted by kernel ridge regression.

Blocks of the reaction field (eighteen fields: five physicochemical and thirteen Sybyl atom types)

* ``odd``    (P x P - R x R) / 2: changes sign when the reaction is reversed; needs no superposition
* ``even0``  (P x P + R x R) / 2: the two sides without any superposition
* ``conv``   C x C, the autocorrelation of the conventional channel (needs the superposition)
* ``dyn``    D x D, the autocorrelation of the dynamic channel (needs the superposition)
* ``bond``   the autocorrelation of the bond-change field
  (``_h``: product superposed on all heavy atoms; ``_c``: on the atoms of the changed bonds;
  ``_n``: every row divided by its length, which is what a cosine-normalised kernel does)

Blocks of the references

* ``counts``  bonds made, broken, raised and lowered by element pair, and the length of the
  dynamic channel of every field
* ``mdiff``, ``msum``  difference and sum of the Morgan count fingerprints of the two sides
* ``dh``      the true reaction energy (only in the rows marked as not usable)

``odd`` and ``even0`` with equal weights are exactly the two sides as separate blocks
(|dR|^2 + |dP|^2 = 2 |d even0|^2 + 2 |d odd|^2), which is identity (1) of the module.

Stages:  ``--stage features`` (needs RDKit; about ten minutes), ``--stage fit`` (NumPy and SciPy
only; resumable, one file per split and repeat), ``--report``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

CACHE = os.path.join(HERE, "cache", "grambow", "stage2")
RESULTS = os.path.join(HERE, "results", "reactions")
FIELDS = ("electrostatic", "steric", "hydrophobic", "hba", "hbd")
TYPES = ("C.1", "C.2", "C.3", "C.ar", "H", "N.1", "N.2", "N.3", "N.am", "N.ar", "N.pl3",
         "O.2", "O.3")
ALPHA = 0.5
TARGETS = ("ea", "dh")
SPLITS = ("reactant", "random", "type")
FOLDS = 5
GAMMAS = (0.1, 0.3, 1.0, 3.0, 10.0)
LAMS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
WEIGHTS = (0.1, 1.0, 10.0)
SUBSAMPLE = 2000
#: how often the search goes through the blocks
SWEEPS = 1
#: the wider grids of ``--wide``: the first run left ridges and block weights at the edges
WIDE = dict(GAMMAS=(0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0),
            LAMS=(1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0),
            WEIGHTS=(0.01, 0.1, 1.0, 10.0, 100.0), SWEEPS=2)
#: the models that are fitted again with the wider grids
WIDE_MODELS = ("CRF + counts", "counts + morgan", "CRF + counts, rows of unit length",
               "counts", "CRF + counts + morgan")

CRF_H = ["odd", "conv_h", "dyn_h", "bond_h"]
CRF_C = ["odd", "conv_c", "dyn_c", "bond_c"]
#: name -> (blocks, blocks whose weights are tied to each other, targets)
MODELS = {
    "counts": (["counts"], [], TARGETS),
    "morgan": (["mdiff", "msum"], [], TARGETS),
    "counts + morgan": (["counts", "mdiff", "msum"], [], TARGETS),
    "sides": (["odd", "even0"], ["odd", "even0"], TARGETS),
    "sides + counts": (["odd", "even0", "counts"], ["odd", "even0"], TARGETS),
    "CRF": (CRF_H, [], TARGETS),
    "CRF + counts": (CRF_H + ["counts"], [], TARGETS),
    "CRF (centre) + counts": (CRF_C + ["counts"], [], TARGETS),
    "CRF + counts, one weight": (CRF_H + ["counts"], CRF_H, TARGETS),
    "CRF + counts, no superposition": (["odd", "even0", "counts"], [], TARGETS),
    "CRF + counts, no bond field": (["odd", "conv_h", "dyn_h", "counts"], [], TARGETS),
    "CRF + counts, rows of unit length": (["odd_n", "conv_n", "dyn_n", "bond_n", "counts"], [],
                                           TARGETS),
    "odd block only": (["odd"], [], TARGETS),
    "even blocks only": (["conv_h", "dyn_h", "bond_h"], [], TARGETS),
    "CRF + counts + morgan": (CRF_H + ["counts", "mdiff", "msum"], [], TARGETS),
    "not usable: CRF + counts + true dh": (CRF_H + ["counts", "dh"], [], ("ea",)),
    "not usable: counts + morgan + true dh": (["counts", "mdiff", "msum", "dh"], [], ("ea",)),
}
GATE = ("CRF + counts", "counts + morgan")


def _say(text):
    print(text, flush=True)


# --------------------------------------------------------------------------- #
# Stage "features"
# --------------------------------------------------------------------------- #


def _channel_scale(X: np.ndarray, n_bins: int, ref: np.ndarray = None) -> np.ndarray:
    """One scale per field pair: the root mean square over all rows and distance channels."""
    R = X if ref is None else ref
    s = np.sqrt(np.mean(R.reshape(len(R), -1, n_bins) ** 2, axis=(0, 2)))
    s[s <= 0] = 1.0
    return np.repeat(s, n_bins)


def _unit_rows(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n <= 0] = 1.0
    return X / n


def _distances(X: np.ndarray, name: str, chunk: int = 2000) -> float:
    """Write the squared distances of the rows of X, divided by their mean, as float32."""
    X = np.ascontiguousarray(X, float)
    n = len(X)
    sq = np.einsum("ij,ij->i", X, X)
    path = os.path.join(CACHE, f"D_{name}.npy")
    D = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n, n))
    rng = np.random.RandomState(0)
    pick = rng.choice(n, min(n, 2000), replace=False)
    mean = float(np.mean(np.maximum(
        sq[pick][:, None] + sq[pick][None, :] - 2.0 * X[pick] @ X[pick].T, 0.0)))
    mean = mean if mean > 0 else 1.0
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        D[a:b] = (np.maximum(sq[a:b, None] + sq[None, :] - 2.0 * X[a:b] @ X.T, 0.0)
                  / mean).astype(np.float32)
    D.flush()
    del D
    return mean


def features() -> dict:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    import prepare_reactions
    from cmfqsar.fields import FieldSet
    from cmfqsar.reaction import (REACTION_SPEC, autocorrelation_rows, bond_change_counts,
                                  channel_norms, reaction_fields, read_reaction)
    RDLogger.DisableLog("rdApp.*")
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    rows = prepare_reactions.load()
    fieldset = FieldSet.create(physchem=list(FIELDS), cif_types=list(TYPES), alpha=ALPHA)
    spec = REACTION_SPEC
    reactions = [read_reaction(r["rsmi"], r["psmi"], r["r"], r["p"], f"rxn{r['idx']:06d}")
                 for r in rows]
    n = len(reactions)
    _say(f"{n} reactions read, {time.time() - t0:.0f} s")

    X = {}
    norms = None
    for on, tag in (("heavy", "h"), ("centre", "c")):
        items = [reaction_fields(rx, fieldset, on=on) for rx in reactions]
        if tag == "h":
            R = autocorrelation_rows([it.reactant for it in items], spec)
            P = autocorrelation_rows([it.product for it in items], spec)
            X["odd"], X["even0"] = 0.5 * (P - R), 0.5 * (P + R)
            norms = channel_norms(items, fieldset)
            del R, P
        X[f"conv_{tag}"] = autocorrelation_rows([it.conventional for it in items], spec)
        X[f"dyn_{tag}"] = autocorrelation_rows([it.dynamic for it in items], spec)
        X[f"bond_{tag}"] = autocorrelation_rows([it.bonds for it in items], spec)
        del items
        _say(f"rows with the product superposed on '{on}': {time.time() - t0:.0f} s")

    counts, labels = bond_change_counts(reactions)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    def morgan(mol):
        m = Chem.Mol(mol)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        try:
            m = Chem.RemoveHs(m)
        except Exception:
            pass
        return gen.GetCountFingerprintAsNumPy(m).astype(float)

    fr = np.array([morgan(rx.reactant) for rx in reactions])
    fp = np.array([morgan(rx.product) for rx in reactions])
    X["mdiff"], X["msum"] = fp - fr, fp + fr

    # groups: the reactant, the type (the set of bond changes), and the links between a reaction
    # and its reverse, which must never fall on two sides of a split
    def canon(mol):
        m = Chem.Mol(mol)
        for a in m.GetAtoms():
            a.SetAtomMapNum(0)
        return Chem.MolToSmiles(m)

    cr = [canon(rx.reactant) for rx in reactions]
    cp = [canon(rx.product) for rx in reactions]
    types = [" ".join(f"{int(c)}x{lab}" for c, lab in zip(row, labels) if c) for row in counts]
    index = {(a, b): k for k, (a, b) in enumerate(zip(cr, cp))}
    reverse = np.array([index.get((b, a), -1) for a, b in zip(cr, cp)], int)

    def ids(values):
        seen = {}
        return np.array([seen.setdefault(v, len(seen)) for v in values], int)

    # scaling: one scale per field pair for the row blocks (odd and even0 share theirs, so that
    # the two with equal weights are the two sides), a z-score for the counts
    nb = spec.n_bins
    scale_sides = _channel_scale(X["even0"], nb)
    Z = {"odd": X["odd"] / scale_sides, "even0": X["even0"] / scale_sides}
    for key in ("conv_h", "dyn_h", "bond_h", "conv_c", "dyn_c", "bond_c"):
        Z[key] = X[key] / _channel_scale(X[key], nb)
    own_odd = X["odd"] / _channel_scale(X["odd"], nb)
    Z["odd_n"] = _unit_rows(own_odd)
    for key in ("conv", "dyn", "bond"):
        Z[f"{key}_n"] = _unit_rows(Z[f"{key}_h"])
    C = np.hstack([counts, norms])
    sd = C.std(axis=0)
    sd[sd <= 0] = 1.0
    Z["counts"] = (C - C.mean(axis=0)) / sd
    Z["mdiff"], Z["msum"] = X["mdiff"], X["msum"]
    ea = np.array([r["ea"] for r in rows])
    dh = np.array([r["dh"] for r in rows])
    Z["dh"] = dh[:, None]
    means = {}
    for key in sorted(Z):
        means[key] = _distances(Z[key], key)
        _say(f"distances of block {key} ({Z[key].shape[1]} columns): {time.time() - t0:.0f} s")
    np.savez_compressed(
        os.path.join(CACHE, "tables.npz"), ea=ea, dh=dh, reactant=ids(cr), type=ids(types),
        reverse=reverse, idx=np.array([r["idx"] for r in rows], int),
        counts=counts, count_labels=np.array(labels), norms=norms)
    info = dict(reactions=n, fields=list(FIELDS) + list(TYPES), alpha=ALPHA,
                spec=dict(d_max=spec.d_max, step=spec.step, sigma=spec.sigma),
                columns={k: int(v.shape[1]) for k, v in Z.items()},
                mean_squared_distance=means, distinct_reactants=int(len(set(cr))),
                distinct_types=int(len(set(types))),
                reactions_with_their_reverse_in_the_set=int(np.sum(reverse >= 0)),
                seconds=time.time() - t0)
    with open(os.path.join(CACHE, "features.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


# --------------------------------------------------------------------------- #
# Folds
# --------------------------------------------------------------------------- #


def units(tables, split: str) -> np.ndarray:
    """The unit every reaction belongs to: units, not reactions, are dealt into folds.

    ``random``: a reaction, joined with its reverse where the set holds it.  ``reactant``: every
    reaction of one reactant.  ``type``: every reaction with the same set of bond changes.  In
    all three a reaction and its reverse are put into one unit.
    """
    n = len(tables["ea"])
    base = {"random": np.arange(n), "reactant": tables["reactant"], "type": tables["type"]}[split]
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for k, r in enumerate(tables["reverse"]):
        if r >= 0:
            a, b = find(int(base[k])), find(int(base[int(r)]))
            if a != b:
                parent[a] = b
    return np.array([find(int(v)) for v in base], int)


def fold_of(unit: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """Fold number of every reaction: whole units are dealt at random, largest first."""
    rng = np.random.RandomState(seed)
    names, sizes = np.unique(unit, return_counts=True)
    order = np.lexsort((rng.rand(len(names)), -sizes))
    load = np.zeros(folds)
    where = {}
    for k in order:
        f = int(np.argmin(load + 1e-9 * rng.rand(folds)))
        where[int(names[k])] = f
        load[f] += sizes[k]
    return np.array([where[int(u)] for u in unit], int)


# --------------------------------------------------------------------------- #
# Stage "fit"
# --------------------------------------------------------------------------- #


class Blocks:
    """The distance matrices on disk, opened once and read where they are needed."""

    def __init__(self):
        self._open = {}

    def get(self, name: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        if name not in self._open:
            self._open[name] = np.load(os.path.join(CACHE, f"D_{name}.npy"), mmap_mode="r")
        D = self._open[name]
        if len(rows) == D.shape[0]:                               # every row: read it in order
            return np.asarray(D)[:, cols]
        return np.asarray(D[rows])[:, cols]


def _mix(parts, w):
    total = float(sum(w))
    S = np.zeros(parts[0].shape, np.float64)
    for p, wi in zip(parts, w):
        S += (wi / total) * p
    return S


class Search:
    """The search for the weights, the width and the ridge of one model on one subsample.

    `parts` are the squared-distance matrices of the blocks over the subsample, `targets` the
    target values as columns, `fit` and `val` index the subsample.  A kernel is factorised once
    for all targets, and only the validation errors are kept, so the search of the second
    target costs almost nothing.
    """

    def __init__(self, parts, targets, fit, val):
        self.parts, self.fit, self.val = parts, fit, val
        self.Y = np.asarray(targets, float).reshape(len(parts[0]), -1)
        self.mean = self.Y[fit].mean(axis=0)
        self._rmse = {}                       # (weights, gamma) -> (ridges, targets)

    def _errors(self, key, g, S):
        from scipy.linalg import cho_factor, cho_solve
        if (key, g) not in self._rmse:
            K = np.exp(-g * S())
            Kff, Kvf = K[np.ix_(self.fit, self.fit)], K[np.ix_(self.val, self.fit)]
            out = np.zeros((len(LAMS), self.Y.shape[1]))
            for i, lam in enumerate(LAMS):
                A = Kff.copy()
                A[np.diag_indices_from(A)] += lam
                c = cho_factor(A, lower=True, overwrite_a=True, check_finite=False)
                pred = Kvf @ cho_solve(c, self.Y[self.fit] - self.mean,
                                       check_finite=False) + self.mean
                out[i] = np.sqrt(np.mean((pred - self.Y[self.val]) ** 2, axis=0))
            self._rmse[(key, g)] = out
        return self._rmse[(key, g)]

    def score(self, w, target):
        key = tuple(w)
        mixed = {}

        def S():
            if "S" not in mixed:
                mixed["S"] = _mix(self.parts, w)
            return mixed["S"]

        best = (np.inf, None, None)
        for g in GAMMAS:
            err = self._errors(key, g, S)[:, target]
            i = int(np.argmin(err))
            if err[i] < best[0]:
                best = (float(err[i]), g, LAMS[i])
        return best

    def choose(self, target: int = 0, tied=()):
        """(weights, gamma, ridge, validation RMSE) for one target.

        The search starts from equal weights, tries each free block at 0.1 and at 10 once, and
        takes gamma and the ridge from their grids every time.  Tied blocks move together.
        """
        names = list(range(len(self.parts)))
        groups = [[i for i in names if i in tied]] if len(tied) else []
        groups += [[i] for i in names if i not in tied]
        w = [1.0] * len(self.parts)
        best = self.score(w, target)
        if len(groups) > 1:
            for _sweep in range(SWEEPS):
                for grp in groups:
                    for value in WEIGHTS:
                        trial = list(w)
                        for i in grp:
                            trial[i] = value
                        s = self.score(trial, target)
                        if s[0] < best[0]:
                            best, w = s, trial
        return w, best[1], best[2], best[0]


def choose(parts, y, fit, val, tied):
    """:meth:`Search.choose` for one target."""
    return Search(parts, y, fit, val).choose(0, tied)


def fit_split(split: str, repeat: int, models=None, folds: int = FOLDS, tag: str = "") -> str:
    """Fit every model on every fold of one split and repeat; write and return the file.

    `tag` is added to the file name; the folds do not depend on it, so a tagged run is on the
    same folds as the plain one.
    """
    from scipy.linalg import cho_factor, cho_solve
    tables = dict(np.load(os.path.join(CACHE, "tables.npz")))
    n = len(tables["ea"])
    Y = {"ea": tables["ea"], "dh": tables["dh"]}
    unit = units(tables, split)
    fold = fold_of(unit, folds, seed=1000 * repeat + SPLITS.index(split))
    path = os.path.join(CACHE, f"fit_{split}_{repeat}{tag}.json")
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            done = json.load(f)
    done.setdefault("fold", fold.tolist())
    done.setdefault("models", {})
    blocks = Blocks()
    names = list(models) if models else list(MODELS)
    for name in names:
        use, tied_names, targets = MODELS[name]
        tied = [use.index(b) for b in tied_names]
        entry = done["models"].setdefault(name, {})
        if all(t in entry and len(entry[t]["chosen"]) == folds for t in targets):
            continue
        t0 = time.time()
        out = {t: dict(pred=[None] * n, chosen=[]) for t in targets}
        for k in range(folds):
            train, test = np.where(fold != k)[0], np.where(fold == k)[0]
            rng = np.random.RandomState(7 + k + 100 * repeat)
            sub = np.sort(rng.choice(train, min(SUBSAMPLE, len(train)), replace=False))
            # the inner split follows the outer one: whole units are held out
            inner = fold_of(unit[sub], 4, seed=k)
            fit, val = np.where(inner != 0)[0], np.where(inner == 0)[0]
            parts = [blocks.get(b, sub, sub).astype(np.float64) for b in use]
            search = Search(parts, np.column_stack([Y[t][sub] for t in targets]), fit, val)
            picks = {t: search.choose(i, tied) for i, t in enumerate(targets)}
            del parts, search
            solved = {}
            for t in targets:
                w, g, lam, val_rmse = picks[t]
                key = (tuple(w), g, lam)
                if key not in solved:
                    S = np.zeros((n, len(train)), np.float32)
                    total = float(sum(w))
                    rows_all = np.arange(n)
                    for b, wi in zip(use, w):
                        S += np.float32(wi / total) * blocks.get(b, rows_all, train)
                    K = np.exp(-g * S.astype(np.float64))
                    del S
                    Ktt = K[train]
                    Ktt[np.diag_indices_from(Ktt)] += lam
                    c = cho_factor(Ktt, lower=True, overwrite_a=True, check_finite=False)
                    solved[key] = (c, K[test])
                    del K
                c, Kte = solved[key]
                m = float(np.mean(Y[t][train]))
                a = cho_solve(c, Y[t][train] - m, check_finite=False)
                pred = Kte @ a + m
                for i, p in zip(test, pred):
                    out[t]["pred"][int(i)] = float(p)
                out[t]["chosen"].append(dict(weights=dict(zip(use, w)), gamma=g, ridge=lam,
                                             inner_rmse=val_rmse))
            del solved
        for t in targets:
            entry[t] = out[t]
        with open(path, "w") as f:
            json.dump(done, f)
        line = ", ".join(f"{t} MAE {np.mean(np.abs(np.array(out[t]['pred']) - Y[t])):.2f}"
                         for t in targets)
        _say(f"[{split} {repeat}] {name}: {line}  ({time.time() - t0:.0f} s)")
    return path


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #


def _stats(y, pred):
    e = pred - y
    return dict(mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                q2=float(1.0 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)))


def simple_references(tables, fold):
    """Out-of-fold predictions of the references that need no kernel.

    ``mean``: the mean of the training part.  ``reactant mean``: the mean over the training
    reactions of the same reactant, and the training mean where there is none; with whole
    reactants held out it must equal ``mean``, which audits the folds.  ``linear in true dh``:
    the Bell-Evans-Polanyi relation, a linear regression of the barrier on the true reaction
    energy; it needs the quantity a model would have to predict first.
    """
    ea, dh, reactant = tables["ea"], tables["dh"], tables["reactant"]
    out = {t: {"mean": np.zeros(len(ea)), "reactant mean": np.zeros(len(ea))} for t in TARGETS}
    out["ea"]["not usable: linear in true dh"] = np.zeros(len(ea))
    shared = 0
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        for t, y in (("ea", ea), ("dh", dh)):
            m = float(y[tr].mean())
            out[t]["mean"][te] = m
            sums, cnts = {}, {}
            for r, v in zip(reactant[tr], y[tr]):
                sums[r] = sums.get(r, 0.0) + v
                cnts[r] = cnts.get(r, 0) + 1
            out[t]["reactant mean"][te] = [sums[r] / cnts[r] if r in sums else m
                                           for r in reactant[te]]
        shared += int(sum(r in set(reactant[tr]) for r in reactant[te]))
        A = np.vstack([dh[tr], np.ones(tr.sum())]).T
        coef = np.linalg.lstsq(A, ea[tr], rcond=None)[0]
        out["ea"]["not usable: linear in true dh"][te] = coef[0] * dh[te] + coef[1]
    return out, shared


def _paired(y, a, b, groups, runs=1000, seed=0):
    """Difference b - a in q2 and in MAE with a 95 % interval from resampling whole groups."""
    rng = np.random.RandomState(seed)
    names = np.unique(groups)
    members = [np.where(groups == g)[0] for g in names]

    def diff(idx):
        sa, sb = _stats(y[idx], a[idx]), _stats(y[idx], b[idx])
        return sb["q2"] - sa["q2"], sb["mae"] - sa["mae"]

    point = diff(np.arange(len(y)))
    draws = np.array([diff(np.concatenate([members[i] for i in
                                           rng.randint(0, len(names), len(names))]))
                      for _ in range(runs)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return dict(q2=[point[0], float(lo[0]), float(hi[0])],
                mae=[point[1], float(lo[1]), float(hi[1])])


def kernel_correlations(run: dict, names, n: int = 2000, seed: int = 0) -> dict:
    """Correlation of whole kernels: the centred alignment of every pair of the named models.

    Each kernel is built on `n` reactions drawn at random, with the settings its model chose
    for the activation energy in the first fold.  1 means that two descriptions order the pairs
    of reactions in the same way; a model can only add to another where this is below 1.
    """
    rng = np.random.RandomState(seed)
    total = len(run["fold"])
    pick = np.sort(rng.choice(total, min(n, total), replace=False))
    blocks = Blocks()
    kernels = {}
    for name in names:
        if name not in run["models"] or "ea" not in run["models"][name]:
            continue
        c = run["models"][name]["ea"]["chosen"][0]
        use = list(c["weights"])
        S = _mix([blocks.get(b, pick, pick).astype(np.float64) for b in use],
                 [c["weights"][b] for b in use])
        K = np.exp(-c["gamma"] * S)
        K = K - K.mean(axis=0, keepdims=True)
        K = K - K.mean(axis=1, keepdims=True)
        kernels[name] = K / np.linalg.norm(K)
    out = {}
    have = list(kernels)
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            out[f"{a} | {b}"] = float(np.sum(kernels[a] * kernels[b]))
    return out


def collect() -> dict:
    tables = dict(np.load(os.path.join(CACHE, "tables.npz")))
    Y = {"ea": tables["ea"], "dh": tables["dh"]}
    with open(os.path.join(CACHE, "features.json")) as f:
        info = json.load(f)
    out = dict(features=info, splits={})
    for split in SPLITS:
        runs = []
        r = 0
        while os.path.exists(os.path.join(CACHE, f"fit_{split}_{r}.json")):
            with open(os.path.join(CACHE, f"fit_{split}_{r}.json")) as f:
                runs.append(json.load(f))
            r += 1
        if not runs:
            continue
        res = dict(repeats=len(runs), models={}, simple={}, paired={}, chosen={})
        shared_all = []
        preds = {t: {} for t in TARGETS}
        # preds[target][model][repeat]: a model need not have every repeat
        for r, run in enumerate(runs):
            fold = np.array(run["fold"])
            simple, shared = simple_references(tables, fold)
            shared_all.append(shared)
            for t in TARGETS:
                for name, p in simple[t].items():
                    preds[t].setdefault(name, {})[r] = p
                for name, entry in run["models"].items():
                    if t in entry and all(v is not None for v in entry[t]["pred"]):
                        preds[t].setdefault(name, {})[r] = np.array(entry[t]["pred"])
        res["test_reactions_whose_reactant_is_in_training"] = shared_all
        # the Bell-Evans-Polanyi line with a reaction energy that a model predicted: fitted on
        # the true energies of the training part, applied to the predicted ones of the test part
        for r, dh_hat in list(preds["dh"].get(GATE[0], {}).items()):
            fold = np.array(runs[r]["fold"])
            line = np.zeros(len(fold))
            for k in np.unique(fold):
                tr, te = fold != k, fold == k
                A = np.vstack([Y["dh"][tr], np.ones(tr.sum())]).T
                coef = np.linalg.lstsq(A, Y["ea"][tr], rcond=None)[0]
                line[te] = coef[0] * dh_hat[te] + coef[1]
            preds["ea"].setdefault(f"linear in dh predicted by {GATE[0]}", {})[r] = line
        res["kernel_correlation"] = kernel_correlations(
            runs[0], ["CRF + counts", "counts + morgan", "sides", "CRF", "counts", "morgan"])
        for t in TARGETS:
            res["models"][t] = {}
            for name, plist in preds[t].items():
                per = [_stats(Y[t], plist[r]) for r in sorted(plist)]
                res["models"][t][name] = dict(
                    repeats=len(per),
                    mae=float(np.mean([s["mae"] for s in per])),
                    rmse=float(np.mean([s["rmse"] for s in per])),
                    q2=float(np.mean([s["q2"] for s in per])),
                    q2_spread=float(np.std([s["q2"] for s in per])) if len(per) > 1 else 0.0)
        # the paired comparisons, with whole reactants resampled, over the repeats both have
        pairs = [GATE, ("CRF + counts, rows of unit length", GATE[1]),
                 ("CRF + counts", "counts"), ("CRF + counts", "sides + counts"),
                 ("CRF + counts", "CRF + counts, one weight"),
                 ("CRF + counts", "CRF + counts, no superposition"),
                 ("CRF + counts", "CRF + counts, no bond field"),
                 ("CRF + counts", "CRF + counts, rows of unit length"),
                 ("CRF + counts", "CRF (centre) + counts"),
                 ("CRF + counts + morgan", "counts + morgan"),
                 ("CRF", "sides"), ("CRF", "morgan")]
        for t in TARGETS:
            res["paired"][t] = {}
            for cand, ref in pairs:
                both = sorted(set(preds[t].get(cand, {})) & set(preds[t].get(ref, {})))
                if not both:
                    continue
                joined = _paired(
                    np.tile(Y[t], len(both)),
                    np.concatenate([preds[t][ref][r] for r in both]),
                    np.concatenate([preds[t][cand][r] for r in both]),
                    np.tile(tables["reactant"], len(both)))
                joined["repeats"] = len(both)
                # the same difference on each set of folds by itself: a margin that holds on
                # one set of folds only is a property of those folds
                joined["each"] = [
                    _paired(Y[t], preds[t][ref][r], preds[t][cand][r], tables["reactant"],
                            runs=300) for r in both] if len(both) > 1 else []
                res["paired"][t][f"{cand} minus {ref}"] = joined
        # what the search chose, to see a setting at the edge of its grid
        for name in runs[0]["models"]:
            res["chosen"][name] = {}
            for t in TARGETS:
                picks = [c for run in runs if name in run["models"] and t in run["models"][name]
                         for c in run["models"][name][t]["chosen"]]
                if not picks:
                    continue
                res["chosen"][name][t] = dict(
                    gamma=sorted({c["gamma"] for c in picks}),
                    ridge=sorted({c["ridge"] for c in picks}),
                    weights={b: sorted({c["weights"][b] for c in picks})
                             for b in picks[0]["weights"]})
        wide = os.path.join(CACHE, f"fit_{split}_0_wide.json")
        if os.path.exists(wide):
            with open(wide) as f:
                run = json.load(f)
            res["wide"] = dict(models={}, paired={}, chosen={})
            for t in TARGETS:
                got = {name: np.array(e[t]["pred"]) for name, e in run["models"].items()
                       if t in e and all(v is not None for v in e[t]["pred"])}
                res["wide"]["models"][t] = {
                    name: dict(_stats(Y[t], p), narrow_mae=res["models"][t].get(name, {}).get("mae"))
                    for name, p in got.items()}
                res["wide"]["paired"][t] = {
                    f"{a} minus {b}": _paired(Y[t], got[b], got[a], tables["reactant"])
                    for a, b in (GATE, ("CRF + counts, rows of unit length", GATE[1]),
                                 ("CRF + counts + morgan", GATE[1]), (GATE[0], "counts"))
                    if a in got and b in got}
            for name, e in run["models"].items():
                picks = e.get("ea", {}).get("chosen", [])
                if picks:
                    res["wide"]["chosen"][name] = dict(
                        gamma=sorted({c["gamma"] for c in picks}),
                        ridge=sorted({c["ridge"] for c in picks}),
                        weights={b: sorted({c["weights"][b] for c in picks})
                                 for b in picks[0]["weights"]})
        out["splits"][split] = res
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "stage2.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


SPLIT_NAMES = {"reactant": "whole reactants held out", "random": "random folds",
               "type": "whole reaction types held out"}
TARGET_NAMES = {"ea": "activation energy", "dh": "reaction energy"}


def report() -> str:
    """Write the table.  Without the cache it is rebuilt from the committed stage2.json."""
    if os.path.exists(os.path.join(CACHE, "tables.npz")):
        d = collect()
    else:
        with open(os.path.join(RESULTS, "stage2.json")) as f:
            d = json.load(f)
    info = d["features"]
    L = ["# The continuous reaction field against its references (plan item PO3, stage 2)", "",
         f"{info['reactions']} reactions of the Grambow set, reactant and product only. "
         f"{info['distinct_reactants']} distinct reactants, {info['distinct_types']} distinct "
         f"sets of bond changes; {info['reactions_with_their_reverse_in_the_set']} reactions "
         f"have their reverse in the set and are kept on one side of every split. Energies in "
         f"kcal/mol. Every model is a Gaussian kernel on its blocks, fitted by kernel ridge "
         f"regression, five folds; the settings are chosen inside the training part.", ""]
    for split, res in d["splits"].items():
        for t in TARGETS:
            if t not in res["models"]:
                continue
            L += [f"## {TARGET_NAMES[t].capitalize()}, {SPLIT_NAMES[split]} "
                  f"(up to {res['repeats']} set{'s' if res['repeats'] > 1 else ''} of folds)", "",
                  "| Model | Sets of folds | MAE | RMSE | q2 |", "|---|---|---|---|---|"]
            order = sorted(res["models"][t].items(), key=lambda kv: kv[1]["mae"])
            for name, s in order:
                L.append(f"| {name} | {s['repeats']} | {s['mae']:.2f} | {s['rmse']:.2f} | "
                         f"{s['q2']:+.3f} |")
            L += ["", "| Difference (first minus second) | Sets of folds | in q2 "
                  "[95 % interval] | in MAE [95 % interval] | in q2 on each set of folds |",
                  "|---|---|---|---|---|"]
            for name, p in res["paired"][t].items():
                each = "; ".join(f"{e['q2'][0]:+.3f} [{e['q2'][1]:+.3f}, {e['q2'][2]:+.3f}]"
                                 for e in p.get("each", []))
                L.append(f"| {name} | {p['repeats']} | {p['q2'][0]:+.3f} [{p['q2'][1]:+.3f}, "
                         f"{p['q2'][2]:+.3f}] | {p['mae'][0]:+.2f} [{p['mae'][1]:+.2f}, "
                         f"{p['mae'][2]:+.2f}] | {each} |")
            L.append("")
        if res.get("wide"):
            W = res["wide"]
            L += [f"### The gate models again with wider grids, {SPLIT_NAMES[split]}, the "
                  "same folds", "",
                  f"Width factor {'/'.join(f'{x:g}' for x in WIDE['GAMMAS'])}; ridge "
                  f"{'/'.join(f'{x:g}' for x in WIDE['LAMS'])}; block weights "
                  f"{'/'.join(f'{x:g}' for x in WIDE['WEIGHTS'])}; the search goes through the "
                  f"blocks {WIDE['SWEEPS']} times.", ""]
            for t in TARGETS:
                L += [f"| Model, {TARGET_NAMES[t]} | MAE | MAE with the first grids | RMSE | q2 |",
                      "|---|---|---|---|---|"]
                for name, s in sorted(W["models"][t].items(), key=lambda kv: kv[1]["mae"]):
                    old = f"{s['narrow_mae']:.2f}" if s.get("narrow_mae") is not None else ""
                    L.append(f"| {name} | {s['mae']:.2f} | {old} | {s['rmse']:.2f} | "
                             f"{s['q2']:+.3f} |")
                L += ["", "| Difference (first minus second) | in q2 [95 % interval] | "
                      "in MAE [95 % interval] |", "|---|---|---|"]
                for name, pr in W["paired"][t].items():
                    L.append(f"| {name} | {pr['q2'][0]:+.3f} [{pr['q2'][1]:+.3f}, "
                             f"{pr['q2'][2]:+.3f}] | {pr['mae'][0]:+.2f} [{pr['mae'][1]:+.2f}, "
                             f"{pr['mae'][2]:+.2f}] |")
                L.append("")
            L += ["| Model | Width factor | Ridge | Block weights |", "|---|---|---|---|"]
            for name, c in W["chosen"].items():
                w = "; ".join(f"{b} {'/'.join(f'{x:g}' for x in v)}"
                              for b, v in c["weights"].items())
                L.append(f"| {name} | {'/'.join(f'{x:g}' for x in c['gamma'])} | "
                         f"{'/'.join(f'{x:g}' for x in c['ridge'])} | {w} |")
            L.append("")
        if res.get("kernel_correlation"):
            L += [f"### Correlation of whole kernels, {SPLIT_NAMES[split]}", "",
                  "Centred alignment on 2,000 reactions, with the settings each model chose "
                  "for the activation energy in the first fold.", "",
                  "| Two models | Correlation |", "|---|---|"]
            L += [f"| {k} | {v:.3f} |" for k, v in res["kernel_correlation"].items()]
            L.append("")
        L += ["### Settings the search chose, activation energy, "
              f"{SPLIT_NAMES[split]}", "",
              "| Model | Width factor | Ridge | Block weights |", "|---|---|---|---|"]
        for name, c in res["chosen"].items():
            if "ea" in c:
                w = "; ".join(f"{b} {'/'.join(f'{x:g}' for x in v)}"
                              for b, v in c["ea"]["weights"].items())
                L.append(f"| {name} | {'/'.join(f'{x:g}' for x in c['ea']['gamma'])} | "
                         f"{'/'.join(f'{x:g}' for x in c['ea']['ridge'])} | {w} |")
        L.append("")
        L += [f"Test reactions whose reactant is also in the training part, "
              f"{SPLIT_NAMES[split]}: {res['test_reactions_whose_reactant_is_in_training']} "
              f"(per repeat).", ""]
    text = "\n".join(L) + "\n"
    with open(os.path.join(RESULTS, "stage2.md"), "w", newline="\n") as f:
        f.write(text)
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--stage", choices=("features", "fit"))
    ap.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--first-repeat", type=int, default=0)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--wide", action="store_true",
                    help="fit the gate models again with wider grids, on the same folds")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    if args.stage == "features":
        _say(json.dumps(features(), indent=1))
    if args.stage == "fit":
        for r in range(args.first_repeat, args.first_repeat + args.repeats):
            for split in args.splits:
                if args.wide:
                    globals().update(WIDE)
                    fit_split(split, r, args.models or WIDE_MODELS, tag="_wide")
                else:
                    fit_split(split, r, args.models)
    if args.report:
        _say(report())


if __name__ == "__main__":
    main()
