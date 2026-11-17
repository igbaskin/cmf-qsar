#!/usr/bin/env python
"""
Measured water solubility with the rotation-invariant field kernel, against the references a
solubility model has to beat. The design is that of ``benchmarks/run_logp.py``, whose fitting,
selection and kernels are imported from it.

The sets are made by ``benchmarks/prepare_solubility.py``: ``examples/solubility/delaney.sdf``
(the Delaney set, the classic solubility benchmark) and ``examples/solubility/
aqsoldb_reliable.sdf`` (the compounds of AqSolDB measured more than once with agreeing results,
none of them in the Delaney set). The target is logS, the decimal logarithm of the solubility
in mol/L. One ETKDG + MMFF94 conformer per compound.

The question, fixed before the measurement
------------------------------------------
The solubility of a solid has two parts. One is how much the molecule prefers water to its own
liquid, which is close to logP. The other is how much it costs to take the molecule out of its
crystal, which grows with the melting point. Yalkowsky's general solubility equation says
logS = 0.5 - logP - 0.01 (melting point in C - 25). A field of one molecule sees the first part.
It sees the second only through molecular features that go with a high melting point: flat
aromatic rings, hydrogen-bond donors, symmetry, size. On logP the field kernel was ahead of every
reference. Here the question is whether that remains true when part of the property belongs to
the crystal, which no model in this benchmark is given.

Delaney's ESOL equation is the reference a solubility model has to beat: a linear regression on
four descriptors, clogP, molecular weight, the number of rotatable bonds and the aromatic
proportion (aromatic atoms over heavy atoms). The main comparison is

    invariant kernel (as the program builds it) minus ESOL refitted in each fold,

the difference of cross-validated q2 with a 95 % paired bootstrap interval over the compounds,
on each set. The other models say where the difference comes from.

The models
----------
* **invariant, cosine** -- the field autocorrelation of the five fields (Gasteiger
  electrostatic, steric, hydrophobic, acceptor, donor), cosine-normalised, as in the logP
  benchmark: fifteen smoothing widths from 0.01 to 6 A, distances up to 12 A.
* **invariant, not normalised** -- the same rows without the cosine normalisation. Solubility
  falls with size, and the cosine removes the length of a row, which grows with size.
* **invariant + ESOL descriptors** -- the cosine kernel plus h times a linear kernel on the four
  ESOL descriptors, the width taken from the cosine model in each fold.
* **linear regression on clogP** -- ridge regression on RDKit's Crippen MolLogP alone.
* **ESOL refitted** -- ridge regression on the four ESOL descriptors, computed by RDKit (Delaney
  used the Daylight clogP), fitted in each fold.
* **field totals**, **eight counts**, **ECFP4**, **refitted Crippen increments**,
  **topological autocorrelation** -- as in the logP benchmark.
* **refitted increments + ESOL descriptors** and **topological + ESOL descriptors** -- the
  kernel plus h times the ESOL kernel; for the topological one with the maximum distance it
  chose in each fold.

The value of the published ESOL equation, which the Delaney file gives for each compound, is
reported for the Delaney set as a fixed prediction with no fitting. Delaney fitted the equation
on a larger set of 2,874 measurements, which may include these compounds.

How the numbers are made
------------------------
As in the logP benchmark: kernel ridge regression with a bias; the setting and the ridge chosen
by the log evidence on each training block; five folds repeated three times on each set; a third
test fits on all of the AqSolDB set and predicts the Delaney set.

Usage
-----
    python benchmarks/run_solubility.py               # everything; the features are cached
    python benchmarks/run_solubility.py --quick 300   # a smoke test on 300 compounds per set
    python benchmarks/run_solubility.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem  # noqa: E402
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors  # noqa: E402

from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, feature_matrix  # noqa: E402
from cmfqsar.model import r2_score, rmse  # noqa: E402

import run_logp as L  # noqa: E402
from run_logp import (COUNT_WIDTHS, D_MAX, FIELDS, HS, LAMS, REPEATS, SIGMAS,  # noqa: E402
                      TOPO_MAX, crippen_counts, eight_counts, gaussian_stack, invariant_kernels,
                      jdump, linear_block, log, selfcheck, standardise, topological_features,
                      topological_kernel)
import run_resolution as RS  # noqa: E402
import run_wavelet as RW  # noqa: E402

DATA = os.path.join(ROOT, "examples", "solubility")
RESULTS = os.path.join(HERE, "results", "solubility")
CACHE = os.path.join(HERE, "cache", "solubility")
SETS = ("delaney", "aqsoldb_reliable")
#: the external test fits on the first and predicts the second
EXTERNAL = ("aqsoldb_reliable", "delaney")
TITLES = {"delaney": "Delaney", "aqsoldb_reliable": "AqSolDB G3+G5"}
#: the models in the order of the table; the json stores them with sorted keys
MODELS = ("invariant, cosine", "invariant, not normalised", "invariant + ESOL descriptors",
          "linear regression on clogP", "ESOL refitted", "field totals", "eight counts",
          "ECFP4", "refitted Crippen increments", "refitted increments + ESOL descriptors",
          "topological autocorrelation", "topological + ESOL descriptors")


class Model(L.Model):
    """The model of the logP benchmark, with the sets of this one."""

    def select(self, y: np.ndarray, blocks):
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


def splits(idx: Dict[str, np.ndarray]):
    out = []
    for name in SETS:
        ii = idx[name]
        for rep in range(REPEATS):
            for te in RW.folds(len(ii), L.FOLDS, L.SEED + rep):
                tr = np.setdiff1d(np.arange(len(ii)), te)
                out.append((name, rep, ii[tr], ii[te]))
    out.append(("external", 0, idx[EXTERNAL[0]], idx[EXTERNAL[1]]))
    return out


def load(quick: int = 0):
    mols, y, tag = [], [], []
    for name in SETS:
        sup = Chem.SDMolSupplier(os.path.join(DATA, f"{name}.sdf"), removeHs=False)
        block = [m for m in sup if m is not None]
        if quick:
            keep = np.linspace(0, len(block) - 1, min(quick, len(block))).astype(int)
            block = [block[i] for i in keep]
        mols += block
        y += [float(m.GetProp("logS")) for m in block]
        tag += [name] * len(block)
    tag = np.array(tag)
    idx = {name: np.where(tag == name)[0] for name in SETS}
    return mols, np.array(y), idx


def features(mfs, fs, sigma: float, names: np.ndarray, use_cache: bool) -> np.ndarray:
    path = os.path.join(CACHE, f"features_{sigma:g}_{len(names)}.npz")
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


def esol_descriptors(mols) -> np.ndarray:
    """Delaney's four: clogP, molecular weight, rotatable bonds, aromatic proportion."""
    rows = []
    for m in mols:
        heavy = m.GetNumHeavyAtoms()
        arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic() and a.GetAtomicNum() > 1)
        rows.append([Crippen.MolLogP(m), Descriptors.MolWt(m),
                     rdMolDescriptors.CalcNumRotatableBonds(m), arom / heavy])
    return np.array(rows)


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
    log(f"fields built ({time.time() - t0:.0f} s)")

    Kclogp = linear_block(standardise(clogp[:, None]))
    Kesol = linear_block(standardise(esol_descriptors(heavy)))
    models = {k: Model(k) for k in MODELS}
    models["linear regression on clogP"].add("linear", Kclogp, y, blocks)
    models["ESOL refitted"].add("linear", Kesol, y, blocks)
    C, crippen_types = crippen_counts(mols)
    Kcounts = linear_block(C)
    models["refitted Crippen increments"].add("linear", Kcounts, y, blocks)
    for h in HS:
        models["refitted increments + ESOL descriptors"].add(f"h {h:g}", Kcounts + h * Kesol,
                                                             y, blocks)
    log(f"{len(crippen_types)} Crippen atom types ({time.time() - t0:.0f} s)")

    # the topological rows at the largest distance hold those at every smaller one
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
            models["topological + ESOL descriptors"].add(f"up to {dm} bonds, h {h:g}",
                                                         Kt + h * Kesol, y, blocks, only)
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
        del Ku
        log(f"sigma {s:g} done ({time.time() - t0:.0f} s)")
    width = [models["invariant, cosine"].best_label(k) for k in range(len(blocks))]
    for s in SIGMAS:
        only = {k for k, w in enumerate(width) if w == f"sigma {s:g}"}
        if not only:
            continue
        Kc, _ = invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s, n_f)
        for h in HS:
            models["invariant + ESOL descriptors"].add(f"sigma {s:g}, h {h:g}", Kc + h * Kesol,
                                                       y, blocks, only)
        log(f"combined model at sigma {s:g}, {len(only)} blocks ({time.time() - t0:.0f} s)")

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
        yy = y[idx[EXTERNAL[1]]]
        r["external"] = {"r2": float(r2_score(yy, ext)), "rmse": float(rmse(yy, ext)),
                         "chosen": chosen["external"]}
        preds[(name, "external")] = ext
        res[name] = r
        log(f"  {name:38s} q2 Delaney {r['delaney']['q2_mean']:+.3f} "
            f"({r['delaney']['rmse_mean']:.3f})  AqSolDB {r['aqsoldb_reliable']['q2_mean']:+.3f} "
            f"({r['aqsoldb_reliable']['rmse_mean']:.3f})  AqSolDB to Delaney "
            f"{r['external']['r2']:+.3f} ({r['external']['rmse']:.3f})")

    ii = idx["delaney"]
    pub = np.array([float(mols[i].GetProp("esol_published")) for i in ii])
    published = {"r2": float(r2_score(y[ii], pub)), "rmse": float(rmse(y[ii], pub))}
    log(f"  published ESOL equation on the Delaney set: R2 {published['r2']:+.3f} "
        f"(RMSE {published['rmse']:.3f}), no fitting")

    pairs = [("invariant, cosine", "ESOL refitted"),
             ("invariant + ESOL descriptors", "ESOL refitted"),
             ("invariant, cosine", "linear regression on clogP"),
             ("invariant, not normalised", "invariant, cosine"),
             ("invariant, cosine", "field totals"),
             ("invariant, cosine", "eight counts"),
             ("invariant, cosine", "ECFP4"),
             ("invariant + ESOL descriptors", "ECFP4"),
             ("invariant, cosine", "refitted Crippen increments"),
             ("invariant + ESOL descriptors", "refitted increments + ESOL descriptors"),
             ("invariant, cosine", "topological autocorrelation"),
             ("invariant + ESOL descriptors", "topological + ESOL descriptors"),
             ("topological autocorrelation", "refitted Crippen increments")]
    comps = []
    for a, b in pairs:
        for test in SETS + ("external",):
            yy = y[idx[EXTERNAL[1]] if test == "external" else idx[test]]
            g = RS.paired(yy, preds[(a, test)], preds[(b, test)])
            comps.append({"model": a, "against": b, "test": test, **g})
            log(f"  {a} - {b} [{test}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    payload = {"compounds": {k: int(len(v)) for k, v in idx.items()}, "quick": quick,
               "y": {k: {"mean": float(y[v].mean()), "sd": float(y[v].std()),
                         "min": float(y[v].min()), "max": float(y[v].max())}
                     for k, v in idx.items()},
               "published_esol_on_delaney": published,
               "models": res, "comparisons": comps, "sigmas": list(SIGMAS), "hs": list(HS),
               "lams": list(LAMS), "count_widths": list(COUNT_WIDTHS), "d_max": D_MAX,
               "folds": L.FOLDS, "repeats": REPEATS, "topo_max": list(TOPO_MAX),
               "crippen_types": len(crippen_types), "seconds": round(time.time() - t0, 1)}
    jdump(payload, os.path.join(RESULTS, "quick.json" if quick else "solubility.json"))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "predictions_quick.npz" if quick else "predictions.npz"),
             names=names, y=y, **{f"{m}|{t}": p for (m, t), p in preds.items()})
    if not quick:
        report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "solubility.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    n = payload["compounds"]
    sd = payload["y"]
    pub = payload["published_esol_on_delaney"]
    lines = [f"Measured water solubility, logS in mol/L: the Delaney set ({n['delaney']} "
             f"compounds, spread {sd['delaney']['sd']:.2f}) and the AqSolDB compounds of groups "
             f"G3 and G5 ({n['aqsoldb_reliable']} compounds, spread "
             f"{sd['aqsoldb_reliable']['sd']:.2f}). Five folds repeated three times on each set; "
             f"the last two columns fit on the AqSolDB set and predict the Delaney set. The "
             f"setting and the ridge are chosen by the evidence on the training block. The "
             f"published ESOL equation, with no fitting, gives R² {pub['r2']:+.3f} and RMSE "
             f"{pub['rmse']:.3f} on the Delaney set.", "",
             "| Model | q², Delaney | RMSE, Delaney | q², AqSolDB | RMSE, AqSolDB | "
             "R², AqSolDB to Delaney | RMSE, AqSolDB to Delaney |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name in MODELS:
        r = payload["models"][name]
        lines.append(f"| {name} | {r['delaney']['q2_mean']:+.3f} | "
                     f"{r['delaney']['rmse_mean']:.3f} | "
                     f"{r['aqsoldb_reliable']['q2_mean']:+.3f} | "
                     f"{r['aqsoldb_reliable']['rmse_mean']:.3f} | {r['external']['r2']:+.3f} | "
                     f"{r['external']['rmse']:.3f} |")
    lines += ["", "| Difference of q² | Test | Mean [95 % interval] |", "|---|---|---|"]
    tests = {"delaney": "Delaney, CV", "aqsoldb_reliable": "AqSolDB, CV",
             "external": "AqSolDB to Delaney"}
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
