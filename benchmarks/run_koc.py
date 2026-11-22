#!/usr/bin/env python
"""
Measured soil sorption coefficients (logKoc) with the rotation-invariant field kernel, against
the references a soil sorption model has to beat. The design is that of
``benchmarks/run_logp.py`` and ``benchmarks/run_boiling.py``; the fitting, the selection and the
kernels are imported from the first.

Koc is the soil organic carbon-water partition coefficient: the concentration of a compound
sorbed to soil, per unit mass of soil organic carbon, divided by its concentration in the soil
water at equilibrium, in L/kg. The target is its decimal logarithm.

The set is made by ``benchmarks/prepare_koc.py`` from the data of the OPERA models (Mansouri et
al., J. Cheminform. 10 (2018) 10; MIT licence): ``examples/koc/koc.sdf``, the PHYSPROP logKoc
values curated for OPERA, with the paper's training and test parts. One ETKDG + MMFF94
conformer per compound.

The question, fixed before the measurement
------------------------------------------
Sorption to soil organic matter is a partition between water and an organic phase, so logKoc
follows the octanol-water partition coefficient. The classical model is one straight line,
logKoc = a logP + b. The program's hydrophobic field puts Wildman and Crippen's logP increments
on the atoms, so the sum of that field over a molecule is clogP. As for logP itself, the field
kernel is only worth something here if it predicts better than that line. The main comparison
is

    invariant kernel (as the program builds it) minus the linear regression on clogP,

with the difference of q2 or R2 and a 95 % paired bootstrap interval over the compounds, on
each test. The models with clogP added say whether the fields hold anything beyond clogP.

The models
----------
* **invariant, cosine**, **invariant, not normalised**, **invariant + clogP**, **linear
  regression on clogP**, **field totals**, **eight counts**, **ECFP4**, **refitted Crippen
  increments**, **topological autocorrelation**, **topological + clogP**, **invariant +
  topological** and **invariant + eight counts** -- as in the logP benchmark.

How the numbers are made
------------------------
As in the logP benchmark: kernel ridge regression with a bias; the setting and the ridge chosen
by the log evidence on each training block; five folds repeated three times on the training
part and on all compounds. The external test fits on the whole training part and predicts the
test part, which is the test of the OPERA paper. The model report of OPERA gives R2 0.71 and
RMSE 0.61 there for its model with 12 descriptors (and Q2 0.81, RMSE 0.55 in fivefold
cross-validation on the training part). OPERA's parts hold 545 and 184 compounds; 536 and 183
of them are used here (see ``examples/koc/README.md``), so the comparison is close, not exact.

Usage
-----
    python benchmarks/run_koc.py               # everything; the features are cached
    python benchmarks/run_koc.py --quick 150   # a smoke test on 150 compounds per part
    python benchmarks/run_koc.py --report
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
from rdkit.Chem import Crippen  # noqa: E402

from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, feature_matrix  # noqa: E402
from cmfqsar.model import r2_score, rmse  # noqa: E402

import run_logp as L  # noqa: E402
from run_logp import (COUNT_WIDTHS, D_MAX, FIELDS, HS, LAMS, REPEATS, SIGMAS,  # noqa: E402
                      TOPO_MAX, add_pairs, crippen_counts, eight_counts, gaussian_at,
                      gaussian_stack, invariant_kernels, jdump, linear_block, log, selfcheck,
                      standardise, topological_features, topological_kernel)
import run_resolution as RS  # noqa: E402
import run_wavelet as RW  # noqa: E402

DATA = os.path.join(ROOT, "examples", "koc")
RESULTS = os.path.join(HERE, "results", "koc")
CACHE = os.path.join(HERE, "cache", "koc")
#: the parts of the data: the two parts of the OPERA paper, and both together
PARTS = ("train", "test", "all")
#: the parts with cross-validation
SETS = ("train", "all")
#: the external test: name -> (the part fitted, the part predicted)
EXTERNALS = {"opera_test": ("train", "test")}
TESTS = SETS + tuple(EXTERNALS)
TITLES = {"train": "Training part, CV", "all": "All compounds, CV",
          "opera_test": "Test part (the OPERA test)"}
#: the models in the order of the table; the json stores them with sorted keys
MODELS = ("invariant, cosine", "invariant, not normalised", "invariant + clogP",
          "linear regression on clogP", "field totals", "eight counts", "ECFP4",
          "refitted Crippen increments", "topological autocorrelation", "topological + clogP",
          "invariant + topological", "invariant + eight counts")
#: what the model report of OPERA gives for its own model (QMRF Q17-26-0017)
OPERA = {"descriptors": 12, "train_r2": 0.81, "train_rmse": 0.54, "cv_q2": 0.81,
         "cv_rmse": 0.55, "test_r2": 0.71, "test_rmse": 0.61, "train_n": 545, "test_n": 184}


class Model(L.Model):
    """The model of the logP benchmark, with the parts and the external test of this one."""

    def select(self, y: np.ndarray, blocks):
        n = len(y)
        cv = {name: np.zeros((REPEATS, n)) for name in SETS}
        external = {}
        chosen: Dict[str, Dict[str, int]] = {name: {} for name in TESTS}
        for k, (name, rep, _, te) in enumerate(blocks):
            ev = np.array([c["ev"][k] for c in self.cands])
            c, j = np.unravel_index(int(np.nanargmax(ev)), ev.shape)
            p = self.cands[c]["pred"][k][j]
            key = f"{self.labels[c]}, ridge {LAMS[j]:g}"
            chosen[name][key] = chosen[name].get(key, 0) + 1
            if name in EXTERNALS:
                external[name] = p
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
    for name, (fit, pred) in EXTERNALS.items():
        out.append((name, 0, idx[fit], idx[pred]))
    return out


def load(quick: int = 0):
    sup = Chem.SDMolSupplier(os.path.join(DATA, "koc.sdf"), removeHs=False)
    mols = [m for m in sup if m is not None]
    parts = [m.GetProp("split") for m in mols]
    if quick:
        keep = []
        for part in ("train", "test"):
            ii = [i for i, p in enumerate(parts) if p == part]
            keep += [ii[j] for j in np.linspace(0, len(ii) - 1, min(quick, len(ii))).astype(int)]
        keep = sorted(keep)
        mols = [mols[i] for i in keep]
        parts = [parts[i] for i in keep]
    tag = np.array(parts)
    idx = {name: np.where(tag == name)[0] for name in ("train", "test")}
    idx["all"] = np.arange(len(mols))
    return mols, np.array([float(m.GetProp("logKoc")) for m in mols]), idx


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
    agree = float(np.max(np.abs(totals[:, FIELDS.index("hydrophobic")] - clogp)))
    log(f"fields built ({time.time() - t0:.0f} s); the hydrophobic sum differs from Crippen "
        f"MolLogP by at most {agree:.2e}")

    Kclogp = linear_block(standardise(clogp[:, None]))
    models = {k: Model(k) for k in MODELS}
    models["linear regression on clogP"].add("linear", Kclogp, y, blocks)
    C, crippen_types = crippen_counts(mols)
    models["refitted Crippen increments"].add("linear", linear_block(C), y, blocks)
    log(f"{len(crippen_types)} Crippen atom types ({time.time() - t0:.0f} s)")

    n_f = len(fs.specs)
    n_pairs = n_f * (n_f + 1) // 2
    top = max(TOPO_MAX)
    Ftop = topological_features(mols, mfs, top).reshape(len(mols), n_pairs, top + 1)

    def topo(dm):
        return topological_kernel(Ftop[:, :, :dm + 1].reshape(len(mols), -1), dm, n_f)

    def field(s):
        return invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s, n_f)

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
        Kc, Ku = field(s)
        models["invariant, cosine"].add(f"sigma {s:g}", Kc, y, blocks)
        models["invariant, not normalised"].add(f"sigma {s:g}", Ku, y, blocks)
        del Kc, Ku
        log(f"sigma {s:g} done ({time.time() - t0:.0f} s)")
    width = [models["invariant, cosine"].best_label(k) for k in range(len(blocks))]
    for s in SIGMAS:
        only = {k for k, w in enumerate(width) if w == f"sigma {s:g}"}
        if not only:
            continue
        Kc, _ = field(s)
        for h in HS:
            models["invariant + clogP"].add(f"sigma {s:g}, h {h:g}", Kc + h * Kclogp, y,
                                            blocks, only)
        log(f"combined model at sigma {s:g}, {len(only)} blocks ({time.time() - t0:.0f} s)")
    cwidth = [models["eight counts"].best_label(k) for k in range(len(blocks))]
    Z8 = eight_counts(heavy)
    add_pairs(models, y, blocks, width, reach, cwidth, lambda s: field(s)[0], topo,
              lambda w: gaussian_at(Z8, w))
    log(f"pair models done ({time.time() - t0:.0f} s)")

    res, preds = {}, {}
    for name in MODELS:
        cv, ext, chosen = models[name].select(y, blocks)
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
        for test, (_, part) in EXTERNALS.items():
            yy = y[idx[part]]
            r[test] = {"r2": float(r2_score(yy, ext[test])), "rmse": float(rmse(yy, ext[test])),
                       "chosen": chosen[test]}
            preds[(name, test)] = ext[test]
        res[name] = r
        log(f"  {name:32s} " + "  ".join(
            f"{t} {r[t]['q2_mean'] if t in SETS else r[t]['r2']:+.3f} "
            f"({r[t]['rmse_mean'] if t in SETS else r[t]['rmse']:.2f})" for t in TESTS))

    pairs = [("invariant, cosine", "linear regression on clogP"),
             ("invariant + clogP", "linear regression on clogP"),
             ("invariant + clogP", "invariant, cosine"),
             ("invariant, not normalised", "invariant, cosine"),
             ("invariant, cosine", "field totals"),
             ("invariant, cosine", "eight counts"),
             ("invariant, cosine", "ECFP4"),
             ("invariant, cosine", "refitted Crippen increments"),
             ("invariant, cosine", "topological autocorrelation"),
             ("invariant + clogP", "topological + clogP"),
             ("invariant + topological", "invariant, cosine"),
             ("invariant + eight counts", "invariant, cosine"),
             ("invariant + eight counts", "eight counts")]
    comps = []
    for a, b in pairs:
        for test in TESTS:
            part = EXTERNALS[test][1] if test in EXTERNALS else test
            g = RS.paired(y[idx[part]], preds[(a, test)], preds[(b, test)])
            comps.append({"model": a, "against": b, "test": test, **g})
            log(f"  {a} - {b} [{test}]: {g['mean']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    payload = {"compounds": {k: int(len(v)) for k, v in idx.items()}, "quick": quick,
               "y": {k: {"mean": float(y[v].mean()), "sd": float(y[v].std()),
                         "min": float(y[v].min()), "max": float(y[v].max())}
                     for k, v in idx.items()},
               "opera": OPERA, "models": res, "comparisons": comps, "sigmas": list(SIGMAS),
               "hs": list(HS), "lams": list(LAMS), "count_widths": list(COUNT_WIDTHS),
               "d_max": D_MAX, "folds": L.FOLDS, "repeats": REPEATS, "topo_max": list(TOPO_MAX),
               "crippen_types": len(crippen_types), "seconds": round(time.time() - t0, 1)}
    jdump(payload, os.path.join(RESULTS, "quick.json" if quick else "koc.json"))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "predictions_quick.npz" if quick else "predictions.npz"),
             names=names, y=y, **{f"{m}|{t}": p for (m, t), p in preds.items()})
    if not quick:
        report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "koc.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    n = payload["compounds"]
    sd = payload["y"]
    op = payload["opera"]
    lines = [f"Measured soil sorption coefficients, logKoc (Koc in L/kg): the PHYSPROP set "
             f"curated for OPERA, its training part ({n['train']} compounds, spread "
             f"{sd['train']['sd']:.2f}) and test part ({n['test']} compounds, spread "
             f"{sd['test']['sd']:.2f}). Five folds repeated three times on the training part "
             f"and on all {n['all']} compounds; the external test fits on the whole training "
             f"part. The setting and the ridge are chosen by the evidence on the training "
             f"block. Each cell gives q² (cross-validation) or R² (external test), with the "
             f"RMSE in brackets.", "",
             "| Model | " + " | ".join(TITLES[t] for t in TESTS) + " |",
             "|---|" + "---:|" * len(TESTS)]
    for name in MODELS:
        r = payload["models"][name]
        cells = [f"{r[t]['q2_mean']:+.3f} ({r[t]['rmse_mean']:.2f})" if t in SETS
                 else f"{r[t]['r2']:+.3f} ({r[t]['rmse']:.2f})" for t in TESTS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append(f"| OPERA model, {op['descriptors']} descriptors, as its report gives it "
                 f"({op['train_n']} and {op['test_n']} compounds) | "
                 f"{op['cv_q2']:+.3f} ({op['cv_rmse']:.2f}) | -- | "
                 f"{op['test_r2']:+.3f} ({op['test_rmse']:.2f}) |")
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
                    help="use this many compounds of each part and do not touch the cache")
    ap.add_argument("--report", action="store_true", help="print the table of the last run")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
        return 0
    run(a.quick)
    return 0


if __name__ == "__main__":
    sys.exit(main())
