#!/usr/bin/env python
"""
Measured logP with the rotation-invariant field kernel, against the references a logP model has
to beat.

The sets are made by ``benchmarks/prepare_logp.py`` from the SangsterLogP data set (CC BY 4.0):
``examples/logp/huuskonen.sdf`` (the Huuskonen set) and ``examples/logp/sangster_p.sdf`` (3,000
compounds sampled from class P of SangsterLogP, the values reported as logP by their authors,
none of them in the Huuskonen set). One ETKDG + MMFF94 conformer per compound.

The question, fixed before the measurement
------------------------------------------
logP is a property of the whole molecule and is close to additive: Wildman and Crippen's clogP
is a sum of one increment per atom type. The program's hydrophobic field puts exactly those
increments on the atoms, so the sum of the hydrophobic field over a molecule *is* clogP. The
field kernel can only be worth something on logP if it predicts better than a linear regression
on that sum. The main comparison is therefore

    invariant kernel (as the program builds it) minus the linear regression on clogP,

the difference of cross-validated q2 with a 95 % paired bootstrap interval over the compounds,
on each set. The other models say where the difference comes from.

The first full run answered it: the invariant kernel was ahead by about 0.07 on both sets, with
widths of 0.03 to 0.05 A chosen in most folds. Two questions were then added, and two references
with them:

* Is the gain the increments, not the geometry? clogP uses increments fitted by Wildman and
  Crippen on other data. **Refitting them inside each fold** is the fair reference.
* Is the gain the 3D distances, or only which atoms are bonded to which? At 0.05 A the channels
  separate bond lengths and angles, which the molecular graph also gives. **The same
  autocorrelation over distances counted in bonds** answers that.

The models
----------
* **invariant, cosine** -- the field autocorrelation of the five fields (Gasteiger
  electrostatic, steric, hydrophobic, acceptor, donor), cosine-normalised: what
  ``cmf build --kernel autocorrelation`` fits. Fifteen smoothing widths from 0.01 to 6 A, step
  equal to the width, distances up to 12 A.
* **invariant, not normalised** -- the same rows without the cosine normalisation, the kernel
  divided by its mean diagonal. The cosine removes the length of a row, which grows with the
  size of the molecule, and logP grows with size too.
* **invariant + clogP** -- the cosine kernel plus h times a linear kernel on clogP: does the
  field add anything to the sum of its own increments? In each fold the width is the one the
  cosine model chose there, and h and the ridge are chosen by the evidence.
* **invariant + five descriptors** -- the cosine kernel plus h times the descriptor block the
  manual describes (molecular weight, polar surface area, donors, acceptors, rotatable bonds,
  standardised, a dot product scaled to mean diagonal 1), the width taken the same way.
* **linear regression on clogP** -- ridge regression on RDKit's Crippen MolLogP alone.
* **field totals** -- a Gaussian kernel on the five sums of the field weights of a molecule:
  the one-number version of each field (total charge, total bulk, clogP, total acceptor and
  donor strength), without the distances.
* **eight counts** -- the Gaussian kernel on the eight counts of the melting-point benchmark.
* **ECFP4** -- the Tanimoto kernel on 2048-bit Morgan fingerprints of radius 2.
* **refitted Crippen increments** -- ridge regression on the counts of Wildman and Crippen's atom
  types (RDKit's labels, hydrogens included): one increment per type, fitted on the training
  block, shrunk towards zero by the ridge. A linear kernel on the raw counts.
* **refitted increments + clogP** -- the same kernel plus h times the clogP kernel: the published
  increments with fitted corrections, which shrink towards the published values instead.
* **topological autocorrelation** -- the field autocorrelation of the same five fields with the
  distance between two atoms counted in bonds (the shortest path in the graph, hydrogens
  included), one channel per distance from 0 bonds up to a maximum chosen by the evidence. It
  is cosine-normalised and scaled exactly as the 3D one. It needs no conformer.
* **topological + clogP** -- as *invariant + clogP*, with the maximum distance the topological
  model chose in each fold.

How the numbers are made
------------------------
Kernel ridge regression with a bias, the program's own model. In every fold the kernel setting
(width, mixing coefficient h) and the ridge are chosen by the log evidence on the training block,
as the program does with *Evidence* selection; the evidence and the fit come from one
eigendecomposition of the training block per setting, which gives the same numbers as
``cmfqsar.evidence.log_evidence`` and ``krr_fit`` (checked in :func:`selfcheck`). Five folds
repeated three times on each set. A third test fits on all of the class P sample and predicts
the Huuskonen set, which it has never seen. The channel scale of the invariant kernel and the
standardisation of the descriptors are computed over all compounds of both files; neither uses
the logP values.

Usage
-----
    python benchmarks/run_logp.py               # everything; the features are cached
    python benchmarks/run_logp.py --quick 300   # a smoke test on 300 compounds per set
    python benchmarks/run_logp.py --report
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors  # noqa: E402

from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation_kernel,  # noqa: E402
                               block_sizes, channel_scale, feature_matrix)
from cmfqsar.model import r2_score, rmse  # noqa: E402

import run_resolution as RS  # noqa: E402
import run_wavelet as RW  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "logp")
RESULTS = os.path.join(HERE, "results", "logp")
CACHE = os.path.join(HERE, "cache", "logp")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: smoke tests chose the widest width offered (1.5 of 0.05-1.5 on 600 compounds, then 3 on
#: 240), so the list goes on to 6; the first full run chose 0.05, the narrowest, in 57 of 62
#: blocks, so it goes down to 0.01
SIGMAS = (0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
D_MAX = 12.0
#: the same smoke test chose h = 3, the largest of 0.3 to 3, for clogP, so the list goes on to 30;
#: the second full run chose 0.3, the smallest, for the refitted increments in 27 of 31 blocks,
#: so it goes down to 0.03
HS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
#: the first full run chose 0.25, the narrowest, for the field totals in 13 of 15 folds
COUNT_WIDTHS = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
#: the largest distance, in bonds, of the topological autocorrelation; the second full run chose
#: 20, the largest, in every block, so it goes on to 40, which is more than the largest number of
#: bonds between two atoms of any compound in the two sets (38, hydrogens included)
TOPO_MAX = (2, 3, 4, 6, 8, 12, 20, 30, 40)
LAMS = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)
FOLDS, REPEATS, SEED = 5, 3, 0
SETS = ("huuskonen", "sangster_p")
MAIN = ("invariant, cosine", "linear regression on clogP")
#: the models in the order of the table; the json stores them with sorted keys
MODELS = ("invariant, cosine", "invariant, not normalised", "invariant + clogP",
          "invariant + five descriptors", "linear regression on clogP", "field totals",
          "eight counts", "ECFP4", "refitted Crippen increments", "refitted increments + clogP",
          "topological autocorrelation", "topological + clogP")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


# --------------------------------------------------------------------------- the fit

class Fit:
    """Evidence and held-out predictions of KRR with a bias for every ridge of :data:`LAMS`.

    One eigendecomposition of the training block serves every ridge. The bordered system of
    ``krr_fit`` has the solution b = 1'A^-1 y / 1'A^-1 1 and a = A^-1 (y - b), with
    A = K + lam I, and the profiled log evidence of ``log_evidence`` uses the same two solves.
    """

    def __init__(self, Ktr: np.ndarray, ytr: np.ndarray):
        s, U = np.linalg.eigh(0.5 * (Ktr + Ktr.T))
        self.s = np.maximum(s, 0.0)
        self.U = U
        self.Uy = U.T @ ytr
        self.U1 = U.T @ np.ones(len(ytr))
        self.n = len(ytr)

    def evidence(self, lam: float) -> float:
        d = self.s + lam
        n = self.n
        denom = float(np.sum(self.U1 ** 2 / d))
        if denom <= 0:
            return float("-inf")
        mean = float(np.sum(self.U1 * self.Uy / d)) / denom
        r = self.Uy - mean * self.U1
        quad = float(np.sum(r ** 2 / d))
        if not np.isfinite(quad) or quad <= 0:
            return float("-inf")
        dof = n - 1
        return float(-0.5 * dof * (math.log(quad / dof) + 1.0) - 0.5 * float(np.sum(np.log(d)))
                     - 0.5 * math.log(denom) - 0.5 * dof * math.log(2.0 * math.pi))

    def coefficients(self, lam: float):
        d = self.s + lam
        denom = float(np.sum(self.U1 ** 2 / d))
        b = float(np.sum(self.U1 * self.Uy / d)) / denom
        a = self.U @ ((self.Uy - b * self.U1) / d)
        return a, b


def selfcheck() -> None:
    """The eigendecomposition route against the program's own functions, on a random kernel."""
    from cmfqsar.evidence import log_evidence
    from cmfqsar.model import krr_fit
    rng = np.random.RandomState(1)
    X = rng.randn(60, 8)
    K = np.exp(-0.1 * np.sum((X[:, None] - X[None]) ** 2, axis=2))
    y = X[:, 0] + 0.3 * rng.randn(60)
    tr, te = np.arange(45), np.arange(45, 60)
    f = Fit(K[np.ix_(tr, tr)], y[tr])
    for lam in (1e-3, 0.1, 1.0):
        e0 = log_evidence(K[np.ix_(tr, tr)], y[tr], lam, fit_bias=True)
        a0, b0 = krr_fit(K[np.ix_(tr, tr)], y[tr], lam, True)
        a, b = f.coefficients(lam)
        assert abs(f.evidence(lam) - e0) < 1e-6 * max(1.0, abs(e0)), (f.evidence(lam), e0)
        assert np.allclose(K[np.ix_(te, tr)] @ a + b, K[np.ix_(te, tr)] @ a0 + b0, atol=1e-8)
    log("self-check: evidence and predictions agree with log_evidence and krr_fit")


def splits(idx: Dict[str, np.ndarray]):
    """(test name, repeat, training indices, test indices) for every block of the measurement."""
    out = []
    for name in SETS:
        ii = idx[name]
        for rep in range(REPEATS):
            for te in RW.folds(len(ii), FOLDS, SEED + rep):
                tr = np.setdiff1d(np.arange(len(ii)), te)
                out.append((name, rep, ii[tr], ii[te]))
    out.append(("external", 0, idx["sangster_p"], idx["huuskonen"]))
    return out


def evaluate(K: np.ndarray, y: np.ndarray, blocks, only=None) -> Dict[str, np.ndarray]:
    """Evidence (block x ridge) and held-out predictions (block x ridge x test) of one kernel.

    `only` restricts the work to some blocks; the others get an evidence of minus infinity, so
    that the selection never picks this kernel there.
    """
    ev = np.full((len(blocks), len(LAMS)), -np.inf)
    preds = []
    for k, (_, _, tr, te) in enumerate(blocks):
        if only is not None and k not in only:
            preds.append(None)
            continue
        f = Fit(K[np.ix_(tr, tr)], y[tr])
        Kte = K[np.ix_(te, tr)]
        p = np.zeros((len(LAMS), len(te)))
        for j, lam in enumerate(LAMS):
            ev[k, j] = f.evidence(lam)
            a, b = f.coefficients(lam)
            p[j] = Kte @ a + b
        preds.append(p)
    return {"ev": ev, "pred": preds}


class Model:
    """The candidates of one model: one evaluated kernel per setting, chosen per block."""

    def __init__(self, name: str):
        self.name = name
        self.labels: List[str] = []
        self.cands: List[Dict[str, np.ndarray]] = []

    def add(self, label: str, K: np.ndarray, y: np.ndarray, blocks, only=None) -> None:
        self.labels.append(label)
        self.cands.append(evaluate(K, y, blocks, only))

    def best_label(self, k: int) -> str:
        """The setting of highest evidence in block `k`."""
        ev = np.array([c["ev"][k] for c in self.cands])
        return self.labels[np.unravel_index(int(np.nanargmax(ev)), ev.shape)[0]]

    def select(self, y: np.ndarray, blocks):
        """Held-out predictions with the setting and ridge of highest evidence in each block."""
        n = len(y)
        cv = {name: np.zeros((REPEATS, n)) for name in SETS}
        external = None
        chosen: Dict[str, Dict[str, int]] = {name: {} for name in SETS + ("external",)}
        for k, (name, rep, _, te) in enumerate(blocks):
            ev = np.array([c["ev"][k] for c in self.cands])
            c, j = np.unravel_index(int(np.nanargmax(ev)), ev.shape)
            p = self.cands[c]["pred"][k][j]
            key = f"{self.labels[c]}, ridge {LAMS[j]:g}"
            chosen[name][key] = chosen[name].get(key, 0) + 1
            if name == "external":
                external = p
            else:
                cv[name][rep, te] = p
        return cv, external, chosen


# --------------------------------------------------------------------------- the descriptions

def load(quick: int = 0):
    mols, y, tag = [], [], []
    for name in SETS:
        sup = Chem.SDMolSupplier(os.path.join(DATA, f"{name}.sdf"), removeHs=False)
        block = [m for m in sup if m is not None]
        if quick:
            keep = np.linspace(0, len(block) - 1, min(quick, len(block))).astype(int)
            block = [block[i] for i in keep]
        mols += block
        y += [float(m.GetProp("logP")) for m in block]
        tag += [name] * len(block)
    tag = np.array(tag)
    idx = {name: np.where(tag == name)[0] for name in SETS}
    return mols, np.array(y), idx


def features(mfs, fs, sigma: float, names: np.ndarray, use_cache: bool) -> np.ndarray:
    """The autocorrelation rows at one width, cached with the names of the compounds."""
    path = os.path.join(CACHE, f"features_{sigma:g}_{len(names)}.npz")
    if use_cache and os.path.isfile(path):
        z = np.load(path)
        if z["names"].shape == names.shape and np.all(z["names"] == names):
            return z["F"].astype(float)
    spec = AutocorrelationSpec(d_max=D_MAX, step=sigma, sigma=sigma)
    F = feature_matrix(mfs, spec, fieldset=fs)
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, F=F.astype(np.float32), names=names)
    return F


def invariant_kernels(F: np.ndarray, sigma: float, n_fields: int):
    spec = AutocorrelationSpec(d_max=D_MAX, step=sigma, sigma=sigma)
    blocks = block_sizes(spec, n_fields)
    sc = channel_scale(F, spec.n_bins, blocks=blocks)
    Kc = autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks)
    Ku = autocorrelation_kernel(F, scale=sc, n_bins=spec.n_bins, blocks=blocks, normalize=False)
    return Kc, Ku / float(np.mean(np.diag(Ku)))


def standardise(X: np.ndarray) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    return (X - X.mean(axis=0)) / np.maximum(X.std(axis=0), 1e-12)


def linear_block(Z: np.ndarray) -> np.ndarray:
    K = Z @ Z.T
    return K / float(np.mean(np.diag(K)))


def gaussian_stack(Z: np.ndarray):
    """One Gaussian kernel per width, made one at a time (an n x n x d array would be large)."""
    sq = np.sum(Z * Z, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0) / Z.shape[1]
    for w in COUNT_WIDTHS:
        yield np.exp(-d2 / (2.0 * w * w))


def eight_counts(mols) -> np.ndarray:
    rows = [[Descriptors.MolWt(m), Lipinski.NumHDonors(m), Lipinski.NumHAcceptors(m),
             rdMolDescriptors.CalcNumRotatableBonds(m), rdMolDescriptors.CalcNumRings(m),
             rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcTPSA(m),
             rdMolDescriptors.CalcFractionCSP3(m)] for m in mols]
    return standardise(np.array(rows))


def five_descriptors(mols) -> np.ndarray:
    rows = [[Descriptors.MolWt(m), rdMolDescriptors.CalcTPSA(m), Lipinski.NumHDonors(m),
             Lipinski.NumHAcceptors(m), rdMolDescriptors.CalcNumRotatableBonds(m)] for m in mols]
    return standardise(np.array(rows))


def crippen_counts(mols):
    """How many atoms of each Wildman-Crippen type a molecule has, hydrogens included.

    RDKit fills the type labels only into lists that already have one element per atom. Returns
    the count matrix and the type labels; a ridge regression on it refits the increments.
    """
    per_mol, labels = [], set()
    for m in mols:
        n = m.GetNumAtoms()
        types, lab = [0] * n, [""] * n
        rdMolDescriptors._CalcCrippenContribs(m, True, types, lab)
        counts: Dict[str, int] = {}
        for t in lab:
            counts[t] = counts.get(t, 0) + 1
        per_mol.append(counts)
        labels.update(counts)
    labels = sorted(labels)
    col = {t: j for j, t in enumerate(labels)}
    C = np.zeros((len(mols), len(labels)))
    for i, counts in enumerate(per_mol):
        for t, c in counts.items():
            C[i, col[t]] = c
    return C, labels


def topological_features(mols, mfs, d_max: int) -> np.ndarray:
    """The field autocorrelation with the distance counted in bonds, one channel per distance.

    The channel of pair (f, g) at distance k is sum over atom pairs l, m that are k bonds apart of
    w_lf w_mg, counted both ways round for f != g, exactly as :func:`cmfqsar.invariant.
    autocorrelation` counts it with a Gaussian of distance; here the Gaussian is replaced by the
    exact count, since a number of bonds is an integer. Distance 0 is the atom with itself.
    """
    n_f = mfs[0].weights.shape[1]
    pairs = [(f, f) for f in range(n_f)] + [(f, g) for f in range(n_f) for g in range(f + 1, n_f)]
    out = np.zeros((len(mols), len(pairs) * (d_max + 1)))
    for i, (m, mf) in enumerate(zip(mols, mfs)):
        W = np.asarray(mf.weights, float)
        if W.shape[0] != m.GetNumAtoms():
            raise ValueError(f"{mf.name}: {W.shape[0]} field atoms, {m.GetNumAtoms()} graph atoms")
        D = Chem.GetDistanceMatrix(m)
        block = np.zeros((len(pairs), d_max + 1))
        for k in range(d_max + 1):
            G = W.T @ (D == k).astype(float) @ W                # (n_f, n_f)
            for p, (f, g) in enumerate(pairs):
                block[p, k] = G[f, g] * (1.0 if f == g else 2.0)
        out[i] = block.ravel()
    return out


def topological_kernel(F: np.ndarray, d_max: int, n_fields: int) -> np.ndarray:
    n_pairs = n_fields * (n_fields + 1) // 2
    blocks = [d_max + 1] * n_pairs
    sc = channel_scale(F, d_max + 1, blocks=blocks)
    return autocorrelation_kernel(F, scale=sc, n_bins=d_max + 1, blocks=blocks)


# --------------------------------------------------------------------------- the run

def run(quick: int = 0) -> Dict[str, object]:
    t0 = time.time()
    selfcheck()
    mols, y, idx = load(quick)
    heavy = [Chem.RemoveHs(m) for m in mols]
    names = np.array([m.GetProp("_Name") for m in mols])
    blocks = splits(idx)
    log(f"{len(mols)} compounds: " + ", ".join(f"{k} {len(v)}" for k, v in idx.items()))
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mfs = [build_fields(m, fs, str(n)) for m, n in zip(mols, names)]
    totals = np.array([mf.weights.sum(axis=0) for mf in mfs])
    clogp = np.array([Crippen.MolLogP(m) for m in heavy])
    hyd = FIELDS.index("hydrophobic")
    agree = float(np.max(np.abs(totals[:, hyd] - clogp)))
    log(f"fields built ({time.time() - t0:.0f} s); the hydrophobic sum differs from Crippen "
        f"MolLogP by at most {agree:.2e}")

    Kclogp = linear_block(standardise(clogp[:, None]))
    Kdesc = linear_block(five_descriptors(heavy))
    models = {k: Model(k) for k in MODELS}
    models["linear regression on clogP"].add("linear", Kclogp, y, blocks)
    C, crippen_types = crippen_counts(mols)
    Kcounts = linear_block(C)
    models["refitted Crippen increments"].add("linear", Kcounts, y, blocks)
    for h in HS:
        models["refitted increments + clogP"].add(f"h {h:g}", Kcounts + h * Kclogp, y, blocks)
    log(f"{len(crippen_types)} Crippen atom types in the two sets ({time.time() - t0:.0f} s)")
    def topo(dm):
        return topological_kernel(topological_features(mols, mfs, dm), dm, len(fs.specs))

    for dm in TOPO_MAX:
        models["topological autocorrelation"].add(f"up to {dm} bonds", topo(dm), y, blocks)
    reach = [models["topological autocorrelation"].best_label(k) for k in range(len(blocks))]
    for dm in TOPO_MAX:
        only = {k for k, w in enumerate(reach) if w == f"up to {dm} bonds"}
        if not only:
            continue
        Kt = topo(dm)
        for h in HS:
            models["topological + clogP"].add(f"up to {dm} bonds, h {h:g}", Kt + h * Kclogp,
                                              y, blocks, only)
    log(f"topological models done ({time.time() - t0:.0f} s)")
    for w, K in zip(COUNT_WIDTHS, gaussian_stack(standardise(totals[:, totals.std(axis=0) > 0]))):
        models["field totals"].add(f"width {w:g}", K, y, blocks)
    for w, K in zip(COUNT_WIDTHS, gaussian_stack(eight_counts(heavy))):
        models["eight counts"].add(f"width {w:g}", K, y, blocks)
    models["ECFP4"].add("Tanimoto", fingerprint_kernel(heavy), y, blocks)
    log(f"references done ({time.time() - t0:.0f} s)")
    for s in SIGMAS:
        F = features(mfs, fs, s, names, use_cache=True)
        Kc, Ku = invariant_kernels(F, s, len(fs.specs))
        del F
        models["invariant, cosine"].add(f"sigma {s:g}", Kc, y, blocks)
        models["invariant, not normalised"].add(f"sigma {s:g}", Ku, y, blocks)
        del Ku
        log(f"sigma {s:g} done ({time.time() - t0:.0f} s)")
    # the two combined models take, in each block, the width the cosine model chose there by the
    # evidence on the same training block, and choose only h and the ridge themselves: ten widths
    # times five values of h would cost five times the rest of the run
    width = [models["invariant, cosine"].best_label(k) for k in range(len(blocks))]
    for s in SIGMAS:
        only = {k for k, w in enumerate(width) if w == f"sigma {s:g}"}
        if not only:
            continue
        Kc, _ = invariant_kernels(features(mfs, fs, s, names, use_cache=True), s, len(fs.specs))
        for h in HS:
            models["invariant + clogP"].add(f"sigma {s:g}, h {h:g}", Kc + h * Kclogp, y, blocks,
                                            only)
            models["invariant + five descriptors"].add(f"sigma {s:g}, h {h:g}", Kc + h * Kdesc,
                                                       y, blocks, only)
        log(f"combined models at sigma {s:g}, {len(only)} blocks ({time.time() - t0:.0f} s)")

    res, preds = {}, {}
    for name, m in models.items():
        cv, ext, chosen = m.select(y, blocks)
        r = {}
        for set_name in SETS:
            ii = idx[set_name]
            yy = y[ii]
            q2 = [float(r2_score(yy, cv[set_name][rep, ii])) for rep in range(REPEATS)]
            r[set_name] = {"q2_mean": float(np.mean(q2)), "q2_sd": float(np.std(q2)),
                           "rmse_mean": float(np.mean([rmse(yy, cv[set_name][rep, ii])
                                                       for rep in range(REPEATS)])),
                           "chosen": chosen[set_name]}
            preds[(name, set_name)] = cv[set_name][:, ii].mean(axis=0)
        yy = y[idx["huuskonen"]]
        r["external"] = {"r2": float(r2_score(yy, ext)), "rmse": float(rmse(yy, ext)),
                         "chosen": chosen["external"]}
        preds[(name, "external")] = ext
        res[name] = r
        log(f"  {name:30s} q2 Huuskonen {r['huuskonen']['q2_mean']:+.3f} "
            f"({r['huuskonen']['rmse_mean']:.3f})  class P {r['sangster_p']['q2_mean']:+.3f} "
            f"({r['sangster_p']['rmse_mean']:.3f})  P to Huuskonen {r['external']['r2']:+.3f} "
            f"({r['external']['rmse']:.3f})")

    pairs = [("invariant, cosine", "linear regression on clogP"),
             ("invariant, not normalised", "invariant, cosine"),
             ("invariant + clogP", "linear regression on clogP"),
             ("invariant + five descriptors", "invariant, cosine"),
             ("invariant, cosine", "field totals"),
             ("invariant, cosine", "eight counts"),
             ("invariant, cosine", "ECFP4"),
             ("invariant + clogP", "ECFP4"),
             ("invariant, cosine", "refitted Crippen increments"),
             ("invariant + clogP", "refitted increments + clogP"),
             ("invariant, cosine", "topological autocorrelation"),
             ("invariant + clogP", "topological + clogP"),
             ("topological autocorrelation", "refitted Crippen increments")]
    comps = []
    for a, b in pairs:
        for test in SETS + ("external",):
            yy = y[idx["huuskonen"] if test == "external" else idx[test]]
            g = RS.paired(yy, preds[(a, test)], preds[(b, test)])
            comps.append({"model": a, "against": b, "test": test, **g})
            log(f"  {a} - {b} [{test}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    payload = {"compounds": {k: int(len(v)) for k, v in idx.items()}, "quick": quick,
               "y": {k: {"mean": float(y[v].mean()), "sd": float(y[v].std()),
                         "min": float(y[v].min()), "max": float(y[v].max())}
                     for k, v in idx.items()},
               "hydrophobic_sum_minus_clogp_max": agree,
               "models": res, "comparisons": comps, "sigmas": list(SIGMAS), "hs": list(HS),
               "lams": list(LAMS), "count_widths": list(COUNT_WIDTHS), "d_max": D_MAX,
               "folds": FOLDS, "repeats": REPEATS, "topo_max": list(TOPO_MAX),
               "crippen_types": len(crippen_types), "seconds": round(time.time() - t0, 1)}
    jdump(payload, os.path.join(RESULTS, "quick.json" if quick else "logp.json"))
    # the held-out predictions, so that a later reference can be compared without a full rerun
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "predictions_quick.npz" if quick else "predictions.npz"),
             names=names, y=y, **{f"{m}|{t}": p for (m, t), p in preds.items()})
    if not quick:
        report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "logp.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    n = payload["compounds"]
    sd = payload["y"]
    lines = [f"Measured logP: Huuskonen set ({n['huuskonen']} compounds, spread "
             f"{sd['huuskonen']['sd']:.2f}) and a class P sample of SangsterLogP "
             f"({n['sangster_p']} compounds, spread {sd['sangster_p']['sd']:.2f}). Five folds "
             f"repeated three times on each set; the last two columns fit on the class P sample "
             f"and predict the Huuskonen set. The setting and the ridge are chosen by the "
             f"evidence on the training block.", "",
             "| Model | q², Huuskonen | RMSE, Huuskonen | q², class P | RMSE, class P | "
             "R², class P to Huuskonen | RMSE, class P to Huuskonen |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        r = payload["models"][name]
        lines.append(f"| {name} | {r['huuskonen']['q2_mean']:+.3f} | "
                     f"{r['huuskonen']['rmse_mean']:.3f} | {r['sangster_p']['q2_mean']:+.3f} | "
                     f"{r['sangster_p']['rmse_mean']:.3f} | {r['external']['r2']:+.3f} | "
                     f"{r['external']['rmse']:.3f} |")
    lines += ["", "| Difference of q² | Test | Mean [95 % interval] |", "|---|---|---|"]
    tests = {"huuskonen": "Huuskonen, CV", "sangster_p": "class P, CV",
             "external": "class P to Huuskonen"}
    for c in payload["comparisons"]:
        lines.append(f"| {c['model']} minus {c['against']} | {tests[c['test']]} | "
                     f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] |")
    text = "\n".join(lines) + "\n"
    # the table is written here, so that run_all.py --tables and --check can rebuild it
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0,
                    help="use this many compounds of each set and do not touch the cache")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
