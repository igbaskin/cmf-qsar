#!/usr/bin/env python
"""
Measured critical micelle concentrations of surfactants with the rotation-invariant field
kernel, against the references a CMC model has to beat. The fitting, the kernels and most
references come from ``benchmarks/run_logp.py``; the design follows the other physical property
benchmarks (logP, solubility, boiling point, vapour pressure).

The set is made by ``benchmarks/prepare_cmc.py`` from SurfPro (Hoedl et al., Digital Discovery 4
(2025) 1176; CC BY-NC-SA 4.0): ``examples/cmc/train.sdf`` (the training part, 1,203 surfactants)
and ``examples/cmc/test.sdf`` (the paper's test set of 140). The target is pCMC, the negative
decimal logarithm of the CMC in mol/L. The structure is the surfactant ion without its
counterion, one ETKDG + MMFF94 conformer each.

The question, fixed before the measurement
------------------------------------------
Is the field kernel, and the field kernel plus the eight counts that was the best model on the
four other properties, better than the classical rule for the CMC? The rule is Klevens's: within
one class of surfactants, log CMC falls linearly with the number of carbon atoms in the tail. The
main comparison is

    invariant kernel (as the program builds it) minus Klevens refitted,

with the difference of q2 and a 95 % paired bootstrap interval, on each test.

The tests
---------
* **random folds**: five folds repeated three times on the training part, as for the other
  properties.
* **series held out**: five folds repeated three times on the training part, in which every
  member of a homologous series is in the same fold. A series is the set of surfactants that
  are the same once every carbon of a pure hydrocarbon or fluorocarbon chain is removed
  (:func:`series_key`): sodium decyl, dodecyl and tetradecyl sulfate are one series, and so are
  the gemini surfactants 12-4-12 and 12-6-12. Random folds put close members of a series on
  both sides of the split; this test asks for a new head group. Its bootstrap draws whole
  series, because the members of one series are not independent.
* **SurfPro test set**: a fit on the whole training part predicts the 140 surfactants of the
  paper's test set. The paper reports, for its graph neural networks (AttentiveFP, an ensemble of
  the ten fold models), MAE 0.235 to 0.279 and RMSE 0.346 to 0.428 pCMC units, and for RDKit
  fingerprints with a random forest MAE 0.630 and RMSE 0.840 (Table 3). Those numbers are for
  the first version of the data; the files here are SurfProV2, in which many structures were
  corrected.

The models
----------
* **invariant, cosine** and **invariant, not normalised** -- the field kernel of the five
  classical fields, with and without the cosine normalisation; the width is chosen per block.
* **Klevens refitted** -- a ridge regression on the descriptors of :func:`klevens`: the class of
  the surfactant (six classes, from the source), the counterion, the number of carbons in the
  longest carbon chain, that number once more for each class (so that each class has its own
  slope), the length of the second longest chain (the second tail of a gemini surfactant), the
  number of fluorinated carbons and the number of ether oxygens (ethylene oxide units).
* **linear regression on the tail length** -- the longest chain alone.
* **invariant + Klevens descriptors**, **topological + Klevens descriptors** -- the kernel plus h
  times the linear kernel of the Klevens descriptors; the kernel's own setting is the one it
  chose alone, and only h and the ridge are chosen here.
* **invariant + topological** and **invariant + eight counts** -- as in the other property
  benchmarks.
* **field totals**, **eight counts**, **ECFP4**, **refitted Crippen increments**,
  **topological autocorrelation** -- the references of the logP benchmark.

No model sees the temperature, as in the paper; 91 % of the values are at 25 C, and 16 have
no temperature in the source.

Usage
-----
    python benchmarks/run_cmc.py               # everything; the features are cached
    python benchmarks/run_cmc.py --quick 200   # a smoke test on 200 surfactants per part
    python benchmarks/run_cmc.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem  # noqa: E402

from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, feature_matrix  # noqa: E402
from cmfqsar.model import r2_score, rmse  # noqa: E402

import run_logp as L  # noqa: E402
from run_logp import (COUNT_WIDTHS, FIELDS, LAMS, REPEATS, add_pairs,  # noqa: E402
                      crippen_counts, eight_counts, gaussian_at,
                      gaussian_stack, invariant_kernels, jdump, linear_block, log, selfcheck,
                      standardise, topological_features, topological_kernel)
import run_resolution as RS  # noqa: E402
import run_wavelet as RW  # noqa: E402

# Surfactants are longer than the molecules of the other property sets: a C16 tail is about 20 A
# long and a gemini surfactant about twice that. A first run with the grids of run_logp.py chose
# the widest field width (6 A), the longest topological reach (40 bonds) and the smallest weight
# of the Klevens block (0.03) almost everywhere, and its distances stopped at 12 A. The grids are
# therefore widened here, and set in run_logp as well, because its helpers read them there.
D_MAX = 30.0
SIGMAS = L.SIGMAS + (8.0, 12.0)
HS = (0.001, 0.003, 0.01) + L.HS
TOPO_MAX = L.TOPO_MAX + (60, 80)
L.D_MAX, L.SIGMAS, L.HS, L.TOPO_MAX = D_MAX, SIGMAS, HS, TOPO_MAX

DATA = os.path.join(ROOT, "examples", "cmc")
RESULTS = os.path.join(HERE, "results", "cmc")
CACHE = os.path.join(HERE, "cache", "cmc")

PARTS = ("train", "test")
#: the two cross-validations, both on the training part
SETS = ("random", "series")
TESTS = SETS + ("external",)
TITLES = {"random": "training part, CV, random folds",
          "series": "training part, CV, series held out",
          "external": "the SurfPro test set"}
#: the surfactant classes of the source
CLASSES = ("anionic", "cationic", "gemini cationic", "non-ionic", "sugar-based non-ionic",
           "zwitterionic")
MODELS = ("invariant, cosine", "invariant, not normalised", "invariant + Klevens descriptors",
          "invariant + eight counts", "invariant + topological", "Klevens refitted",
          "linear regression on the tail length", "field totals", "eight counts", "ECFP4",
          "refitted Crippen increments", "topological autocorrelation",
          "topological + Klevens descriptors")
#: the published errors on the test set (Table 3 of the paper, first version of the data)
PUBLISHED = {"AttentiveFP, best (all properties, 96 channels)": (0.235, 0.346),
             "AttentiveFP, one property, 64 channels": (0.250, 0.382),
             "RDKit fingerprint, random forest": (0.630, 0.840)}


# ------------------------------------------------------------------ chains, Klevens and series

def _chain_carbons(m) -> List[int]:
    """The carbons outside rings with four single bonds (sp3): the atoms a tail is made of."""
    return [a.GetIdx() for a in m.GetAtoms()
            if a.GetSymbol() == "C" and not a.IsInRing() and not a.GetIsAromatic()
            and a.GetHybridization() == Chem.HybridizationType.SP3]


def chain_lengths(m) -> List[int]:
    """The number of carbons on the longest path of each chain, longest first.

    A chain is a connected set of carbons from :func:`_chain_carbons`; it is a tree, because no
    atom of it is in a ring, so its longest path is found by two breadth-first searches.
    """
    keep = set(_chain_carbons(m))
    nbr = {i: [n.GetIdx() for n in m.GetAtomWithIdx(i).GetNeighbors() if n.GetIdx() in keep]
           for i in keep}

    def farthest(start):
        dist, todo = {start: 1}, [start]
        while todo:
            nxt = []
            for i in todo:
                for j in nbr[i]:
                    if j not in dist:
                        dist[j] = dist[i] + 1
                        nxt.append(j)
            todo = nxt
        far = max(dist, key=dist.get)
        return far, dist[far], set(dist)

    out, seen = [], set()
    for i in sorted(keep):
        if i in seen:
            continue
        a, _, comp = farthest(i)
        _, n, _ = farthest(a)
        seen |= comp
        out.append(n)
    return sorted(out, reverse=True) or [0]


def klevens(mols, types, counterions) -> np.ndarray:
    """The descriptors of the refitted Klevens rule, one row per surfactant (not standardised)."""
    ions = sorted(set(counterions))
    rows = []
    for m, t, c in zip(mols, types, counterions):
        n = chain_lengths(m)
        cls = [float(t == k) for k in CLASSES]
        n_cf = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "C"
                   and any(x.GetSymbol() == "F" for x in a.GetNeighbors()))
        n_eo = sum(1 for a in m.GetAtoms() if a.GetSymbol() == "O" and not a.IsInRing()
                   and a.GetDegree() == 2
                   and all(x.GetSymbol() == "C" and not x.GetIsAromatic()
                           and x.GetHybridization() == Chem.HybridizationType.SP3
                           for x in a.GetNeighbors()))
        rows.append(cls + [float(c == k) for k in ions] + [n[0]] + [n[0] * x for x in cls]
                    + [n[1] if len(n) > 1 else 0, n_cf, n_eo])
    return np.array(rows, float)


def series_key(m) -> str:
    """The surfactant without its hydrocarbon and fluorocarbon chains: the head of a series.

    Every carbon of :func:`_chain_carbons` whose heavy neighbours are only carbon or fluorine is
    removed, with its fluorines; the canonical SMILES of what remains is the key. The carbon that
    binds the head (the first carbon of the tail) stays, so the key keeps the way the tail is
    attached.
    """
    heavy = Chem.RemoveHs(m)
    drop = set()
    for i in _chain_carbons(heavy):
        a = heavy.GetAtomWithIdx(i)
        if all(n.GetSymbol() in ("C", "F") for n in a.GetNeighbors()):
            drop.add(i)
            drop |= {n.GetIdx() for n in a.GetNeighbors() if n.GetSymbol() == "F"}
    rw = Chem.RWMol(heavy)
    for i in sorted(drop, reverse=True):
        rw.RemoveAtom(i)
    # the cut leaves atoms with fewer bonds; the SMILES of the unsanitised remainder is enough
    # for a key, and sanitising could refuse it
    text = Chem.MolToSmiles(rw.GetMol(), canonical=True)
    return ".".join(sorted(text.split("."))) or "only chains"


def series_folds(groups: np.ndarray, k: int, seed: int) -> List[np.ndarray]:
    """`k` folds of about equal size in which every member of a series is in the same fold."""
    rng = np.random.RandomState(seed)
    keys = np.unique(groups)
    rng.shuffle(keys)
    size = {g: int(np.sum(groups == g)) for g in keys}
    order = sorted(keys, key=lambda g: -size[g])           # stable: ties keep the shuffled order
    load, members = [0] * k, [[] for _ in range(k)]
    for g in order:
        f = int(np.argmin(load))
        members[f].append(g)
        load[f] += size[g]
    return [np.sort(np.where(np.isin(groups, mm))[0]) for mm in members]


# ------------------------------------------------------------------------- data and splits

def load(quick: int = 0):
    mols, y, tag = [], [], []
    for part in PARTS:
        sup = Chem.SDMolSupplier(os.path.join(DATA, f"{part}.sdf"), removeHs=False)
        block = [m for m in sup if m is not None]
        if quick:
            pick = np.linspace(0, len(block) - 1, min(quick, len(block))).astype(int)
            block = [block[i] for i in pick]
        mols += block
        y += [float(m.GetProp("pCMC")) for m in block]
        tag += [part] * len(block)
    tag = np.array(tag)
    return mols, np.array(y), {p: np.where(tag == p)[0] for p in PARTS}


def splits(idx: Dict[str, np.ndarray], groups: np.ndarray):
    out = []
    ii = idx["train"]
    for rep in range(REPEATS):
        for te in RW.folds(len(ii), L.FOLDS, L.SEED + rep):
            out.append(("random", rep, ii[np.setdiff1d(np.arange(len(ii)), te)], ii[te]))
    for rep in range(REPEATS):
        for te in series_folds(groups[ii], L.FOLDS, L.SEED + rep):
            out.append(("series", rep, ii[np.setdiff1d(np.arange(len(ii)), te)], ii[te]))
    out.append(("external", 0, idx["train"], idx["test"]))
    return out


class Model(L.Model):
    """The model of the logP benchmark, with the two cross-validations and the test of this one."""

    def select(self, y: np.ndarray, blocks):
        n = len(y)
        cv = {name: np.zeros((REPEATS, n)) for name in SETS}
        external = None
        chosen: Dict[str, Dict[str, int]] = {name: {} for name in TESTS}
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


def paired_by_series(y, a, b, groups, seed: int = RS.SEED) -> Dict[str, float]:
    """q2 of a minus q2 of b, with a paired bootstrap interval that draws whole series."""
    rng = np.random.RandomState(seed)
    keys = np.unique(groups)
    where = {g: np.where(groups == g)[0] for g in keys}
    diffs = []
    for _ in range(RS.DRAWS):
        idx = np.concatenate([where[g] for g in rng.choice(keys, len(keys))])
        diffs.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"mean": float(r2_score(y, a) - r2_score(y, b)), "lo": float(lo), "hi": float(hi)}


def features(mfs, fs, sigma: float, names: np.ndarray, use_cache: bool) -> np.ndarray:
    path = os.path.join(CACHE, f"features_{sigma:g}_{D_MAX:g}_{len(names)}.npz")
    if use_cache and os.path.isfile(path):
        z = np.load(path)
        if z["names"].shape == names.shape and np.all(z["names"] == names):
            return z["F"].astype(float)
    F = feature_matrix(mfs, AutocorrelationSpec(d_max=D_MAX, step=sigma, sigma=sigma),
                       fieldset=fs)
    if use_cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez(path, F=F.astype(np.float32), names=names)
    return F


def klevens_selfcheck() -> None:
    """The chain lengths and series of three surfactants whose answer is known."""
    sds = Chem.MolFromSmiles("CCCCCCCCCCCCOS(=O)(=O)[O-]")
    gem = Chem.MolFromSmiles("CCCCCCCCCCCC[N+](C)(C)CCCCCC[N+](C)(C)CCCCCCCCCCCC")
    c10 = Chem.MolFromSmiles("CCCCCCCCCCOS(=O)(=O)[O-]")
    ok = (chain_lengths(sds) == [12] and chain_lengths(gem)[:3] == [12, 12, 6]
          and series_key(sds) == series_key(c10))
    if not ok:
        raise SystemExit(f"Klevens self-check failed: {chain_lengths(sds)} {chain_lengths(gem)} "
                         f"{series_key(sds)} {series_key(c10)}")


# --------------------------------------------------------------------------------- the run

def run(quick: int = 0) -> Dict[str, object]:
    t0 = time.time()
    selfcheck()
    klevens_selfcheck()
    mols, y, idx = load(quick)
    heavy = [Chem.RemoveHs(m) for m in mols]
    names = np.array([m.GetProp("_Name") for m in mols])
    types = [m.GetProp("surfactant_type") for m in mols]
    ions = [m.GetProp("counterion") if m.HasProp("counterion") else "" for m in mols]
    groups = np.array([series_key(m) for m in heavy])
    blocks = splits(idx, groups)
    n_series = len(np.unique(groups[idx["train"]]))
    log(f"{len(mols)} surfactants: train {len(idx['train'])} in {n_series} series, "
        f"test {len(idx['test'])}")
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mfs = [build_fields(m, fs, str(n)) for m, n in zip(mols, names)]
    totals = np.array([mf.weights.sum(axis=0) for mf in mfs])
    Zk = klevens(heavy, types, ions)
    tail = Zk[:, len(CLASSES) + len(set(ions))]
    log(f"fields built ({time.time() - t0:.0f} s)")

    Kkl = linear_block(standardise(Zk[:, Zk.std(axis=0) > 0]))
    models = {k: Model(k) for k in MODELS}
    models["Klevens refitted"].add("linear", Kkl, y, blocks)
    models["linear regression on the tail length"].add(
        "linear", linear_block(standardise(tail[:, None])), y, blocks)
    C, crippen_types = crippen_counts(mols)
    models["refitted Crippen increments"].add("linear", linear_block(C), y, blocks)
    log(f"{len(crippen_types)} Crippen atom types ({time.time() - t0:.0f} s)")

    n_f = len(fs.specs)
    n_pairs = n_f * (n_f + 1) // 2
    top = max(TOPO_MAX)
    Ftop = topological_features(mols, mfs, top).reshape(len(mols), n_pairs, top + 1)

    def topo(dm):
        return topological_kernel(Ftop[:, :, :dm + 1].reshape(len(mols), -1), dm, n_f)

    for dm in TOPO_MAX:
        models["topological autocorrelation"].add(f"up to {dm} bonds", topo(dm), y, blocks)
    reach = [models["topological autocorrelation"].best_label(k) for k in range(len(blocks))]
    for dm in TOPO_MAX:
        only = {k for k, w in enumerate(reach) if w == f"up to {dm} bonds"}
        if not only:
            continue
        Kt = topo(dm)
        for h in HS:
            models["topological + Klevens descriptors"].add(f"up to {dm} bonds, h {h:g}",
                                                            Kt + h * Kkl, y, blocks, only)
    log(f"topological models done ({time.time() - t0:.0f} s)")
    for w, K in zip(COUNT_WIDTHS, gaussian_stack(standardise(totals[:, totals.std(axis=0) > 0]))):
        models["field totals"].add(f"width {w:g}", K, y, blocks)
    for w, K in zip(COUNT_WIDTHS, gaussian_stack(eight_counts(heavy))):
        models["eight counts"].add(f"width {w:g}", K, y, blocks)
    models["ECFP4"].add("Tanimoto", fingerprint_kernel(heavy), y, blocks)
    log(f"references done ({time.time() - t0:.0f} s)")
    for s in SIGMAS:
        Kc, Ku = invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s, n_f)
        models["invariant, cosine"].add(f"sigma {s:g}", Kc, y, blocks)
        models["invariant, not normalised"].add(f"sigma {s:g}", Ku, y, blocks)
        del Kc, Ku
        log(f"sigma {s:g} done ({time.time() - t0:.0f} s)")
    width = [models["invariant, cosine"].best_label(k) for k in range(len(blocks))]
    for s in SIGMAS:
        only = {k for k, w in enumerate(width) if w == f"sigma {s:g}"}
        if not only:
            continue
        Kc, _ = invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s, n_f)
        for h in HS:
            models["invariant + Klevens descriptors"].add(f"sigma {s:g}, h {h:g}",
                                                          Kc + h * Kkl, y, blocks, only)
        log(f"combined model at sigma {s:g}, {len(only)} blocks ({time.time() - t0:.0f} s)")
    cwidth = [models["eight counts"].best_label(k) for k in range(len(blocks))]
    Z8 = eight_counts(heavy)
    add_pairs(models, y, blocks, width, reach, cwidth,
              lambda s: invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s,
                                          n_f)[0],
              topo, lambda w: gaussian_at(Z8, w))
    log(f"pair models done ({time.time() - t0:.0f} s)")

    tr, te = idx["train"], idx["test"]
    res, preds = {}, {}
    for name in MODELS:
        cv, ext, chosen = models[name].select(y, blocks)
        r = {}
        for set_name in SETS:
            yy = y[tr]
            q2 = [float(r2_score(yy, cv[set_name][rep, tr])) for rep in range(REPEATS)]
            r[set_name] = {"q2_mean": float(np.mean(q2)), "q2_sd": float(np.std(q2)),
                           "rmse_mean": float(np.mean([rmse(yy, cv[set_name][rep, tr])
                                                       for rep in range(REPEATS)])),
                           "chosen": chosen[set_name]}
            preds[(name, set_name)] = cv[set_name][:, tr].mean(axis=0)
        r["external"] = {"r2": float(r2_score(y[te], ext)), "rmse": float(rmse(y[te], ext)),
                         "mae": float(np.mean(np.abs(y[te] - ext))),
                         "chosen": chosen["external"]}
        preds[(name, "external")] = ext
        res[name] = r
        log(f"  {name:38s} " + "  ".join(
            f"{t} {r[t]['q2_mean'] if t in SETS else r[t]['r2']:+.3f} "
            f"({r[t]['rmse_mean'] if t in SETS else r[t]['rmse']:.3f})" for t in TESTS))

    pairs = [("invariant, cosine", "Klevens refitted"),
             ("invariant + Klevens descriptors", "Klevens refitted"),
             ("invariant, cosine", "linear regression on the tail length"),
             ("invariant, not normalised", "invariant, cosine"),
             ("invariant, cosine", "field totals"),
             ("invariant, cosine", "eight counts"),
             ("invariant, cosine", "ECFP4"),
             ("invariant, cosine", "refitted Crippen increments"),
             ("invariant, cosine", "topological autocorrelation"),
             ("invariant + Klevens descriptors", "topological + Klevens descriptors"),
             ("invariant + topological", "invariant, cosine"),
             ("invariant + eight counts", "invariant, cosine"),
             ("invariant + eight counts", "eight counts"),
             ("invariant + Klevens descriptors", "invariant + eight counts")]
    comps = []
    for a, b in pairs:
        for test in TESTS:
            if test == "external":
                g = RS.paired(y[te], preds[(a, test)], preds[(b, test)])
            elif test == "series":
                g = paired_by_series(y[tr], preds[(a, test)], preds[(b, test)], groups[tr])
            else:
                g = RS.paired(y[tr], preds[(a, test)], preds[(b, test)])
            comps.append({"model": a, "against": b, "test": test, **g})
            log(f"  {a} - {b} [{test}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    sizes = np.bincount(np.unique(groups[tr], return_inverse=True)[1])
    payload = {"surfactants": {k: int(len(v)) for k, v in idx.items()}, "quick": quick,
               "series": {"count": int(n_series), "largest": int(sizes.max()),
                          "single_members": int(np.sum(sizes == 1))},
               "y": {k: {"mean": float(y[v].mean()), "sd": float(y[v].std()),
                         "min": float(y[v].min()), "max": float(y[v].max())}
                     for k, v in idx.items()},
               "published_test_errors": {k: {"mae": v[0], "rmse": v[1]}
                                         for k, v in PUBLISHED.items()},
               "models": res, "comparisons": comps, "sigmas": list(SIGMAS), "hs": list(HS),
               "lams": list(LAMS), "count_widths": list(COUNT_WIDTHS), "d_max": D_MAX,
               "folds": L.FOLDS, "repeats": REPEATS, "topo_max": list(TOPO_MAX),
               "crippen_types": len(crippen_types), "seconds": round(time.time() - t0, 1)}
    jdump(payload, os.path.join(RESULTS, "quick.json" if quick else "cmc.json"))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "predictions_quick.npz" if quick else "predictions.npz"),
             names=names, y=y, groups=groups,
             **{f"{m}|{t}": p for (m, t), p in preds.items()})
    if not quick:
        report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "cmc.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    n, sd, s = payload["surfactants"], payload["y"], payload["series"]
    lines = [f"Measured critical micelle concentrations, pCMC = -log10(CMC in mol/L), from "
             f"SurfPro: the training part ({n['train']} surfactants in {s['count']} series, "
             f"spread {sd['train']['sd']:.2f}) and the paper's test set ({n['test']}, spread "
             f"{sd['test']['sd']:.2f}). Five folds repeated three times on the training part, "
             f"once with random folds and once with every series in one fold; the test set is "
             f"predicted by a fit on the whole training part. The setting and the ridge are "
             f"chosen by the evidence on the training block. Each cell gives q² "
             f"(cross-validation) or R² (test set), with the RMSE in pCMC units in brackets; "
             f"the last column is the mean absolute error (MAE) on the test set.", "",
             "| Model | " + " | ".join(TITLES[t] for t in TESTS) + " | MAE, test set |",
             "|---|" + "---:|" * (len(TESTS) + 1)]
    for name in MODELS:
        r = payload["models"][name]
        cells = [f"{r[t]['q2_mean']:+.3f} ({r[t]['rmse_mean']:.3f})" if t in SETS
                 else f"{r[t]['r2']:+.3f} ({r[t]['rmse']:.3f})" for t in TESTS]
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {r['external']['mae']:.3f} |")
    lines += ["", "Published errors on the test set (Table 3 of the SurfPro paper, on the first "
              "version of the data):", "", "| Published model | MAE | RMSE |", "|---|---:|---:|"]
    for k, v in payload["published_test_errors"].items():
        lines.append(f"| {k} | {v['mae']:.3f} | {v['rmse']:.3f} |")
    lines += ["", "| Difference of q² or R² | Test | Mean [95 % interval] |", "|---|---|---|"]
    for c in payload["comparisons"]:
        lines.append(f"| {c['model']} minus {c['against']} | {TITLES[c['test']]} | "
                     f"{c['mean']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] |")
    text = "\n".join(lines) + "\n"
    # the table is written here, so that run_all.py --tables and --check can rebuild it
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", type=int, default=0,
                    help="use this many surfactants of each part and do not touch the cache")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
