#!/usr/bin/env python
"""
Measured vapour pressures with the rotation-invariant field kernel, against the references a
vapour pressure model has to beat. The design, the parts of the data and the two external tests
are those of ``benchmarks/run_boiling.py``, whose model selection, splits and Joback groups are
imported from it; the fitting and the kernels come from ``benchmarks/run_logp.py``.

The sets are made by ``benchmarks/prepare_vapour.py`` from the data of the OPERA models
(Mansouri et al., J. Cheminform. 10 (2018) 10; MIT licence): ``examples/vapour/physprop.sdf``
(the PHYSPROP vapour pressures curated for OPERA, with the paper's training and test parts) and
``examples/vapour/echemportal.sdf`` (vapour pressures from registration dossiers that OPERA 2.9
added, none of them in PHYSPROP). The target is logVP, the decimal logarithm of the vapour
pressure in mmHg, near 25 C. One ETKDG + MMFF94 conformer per compound.

The question, fixed before the measurement
------------------------------------------
The vapour pressure of a liquid is set by the same forces between molecules as its boiling
point; that of a solid also by the cost of leaving the crystal, as for the solubility. The
benchmark asks whether the field kernel keeps the lead it had on boiling points when part of the
property belongs to the crystal. The main comparison is

    invariant kernel (as the program builds it) minus Joback's groups refitted in each fold,

with the difference of q2 and a 95 % paired bootstrap interval over the compounds, on each test.
Joback's method has no vapour pressure equation, so here its groups are only a set of counts
for a ridge regression; no published equation is reported.

The models and the tests are those of the boiling point benchmark. The OPERA paper reports, for
its vapour pressure model with 12 descriptors, Q2 0.91 and RMSE 1.08 log units in fivefold
cross-validation on the training part (2,034 compounds) and R2 0.92 and RMSE 1.00 on the test
part (679 compounds).

Usage
-----
    python benchmarks/run_vapour.py               # everything; the features are cached
    python benchmarks/run_vapour.py --quick 300   # a smoke test on 300 compounds per part
    python benchmarks/run_vapour.py --report
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
from rdkit.Chem import Descriptors  # noqa: E402

from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, feature_matrix  # noqa: E402
from cmfqsar.model import r2_score, rmse  # noqa: E402

import run_logp as L  # noqa: E402
from run_logp import (COUNT_WIDTHS, D_MAX, FIELDS, HS, LAMS, REPEATS, SIGMAS,  # noqa: E402
                      TOPO_MAX, add_pairs, crippen_counts, eight_counts, gaussian_at,
                      gaussian_stack, invariant_kernels,
                      jdump, linear_block, log, selfcheck, standardise, topological_features,
                      topological_kernel)
from run_boiling import (EXTERNALS, JOBACK, MODELS, PARTS, SETS, TESTS, TITLES,  # noqa: E402
                         Model, joback, joback_selfcheck, splits)
import run_resolution as RS  # noqa: E402

DATA = os.path.join(ROOT, "examples", "vapour")
RESULTS = os.path.join(HERE, "results", "vapour")
CACHE = os.path.join(HERE, "cache", "vapour")


def load(quick: int = 0):
    mols, y, tag = [], [], []
    for fname in ("physprop", "echemportal"):
        sup = Chem.SDMolSupplier(os.path.join(DATA, f"{fname}.sdf"), removeHs=False)
        block = [m for m in sup if m is not None]
        parts = ([f"physprop_{m.GetProp('split')}" for m in block] if fname == "physprop"
                 else ["echemportal"] * len(block))
        if quick:
            keep = []
            for part in sorted(set(parts)):
                ii = [i for i, p in enumerate(parts) if p == part]
                keep += [ii[j] for j in np.linspace(0, len(ii) - 1,
                                                    min(quick, len(ii))).astype(int)]
            keep = sorted(keep)
            block = [block[i] for i in keep]
            parts = [parts[i] for i in keep]
        mols += block
        y += [float(m.GetProp("logVP")) for m in block]
        tag += parts
    tag = np.array(tag)
    idx = {name: np.where(tag == name)[0] for name in PARTS}
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


def run(quick: int = 0) -> Dict[str, object]:
    t0 = time.time()
    selfcheck()
    joback_selfcheck()
    mols, y, idx = load(quick)
    heavy = [Chem.RemoveHs(m) for m in mols]
    names = np.array([m.GetProp("_Name") for m in mols])
    blocks = splits(idx)
    log(f"{len(mols)} compounds: " + ", ".join(f"{k} {len(v)}" for k, v in idx.items()))
    fs = FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                         param_scheme="rdkit")
    mfs = [build_fields(m, fs, str(n)) for m, n in zip(mols, names)]
    totals = np.array([mf.weights.sum(axis=0) for mf in mfs])
    mw = np.array([Descriptors.MolWt(m) for m in heavy])
    Cj, _ = joback(heavy)
    log(f"fields built ({time.time() - t0:.0f} s)")

    Kmw = linear_block(standardise(mw[:, None]))
    Kjob = linear_block(Cj)
    models = {k: Model(k) for k in MODELS}
    models["linear regression on molecular weight"].add("linear", Kmw, y, blocks)
    models["Joback refitted"].add("linear", Kjob, y, blocks)
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
            models["topological + Joback groups"].add(f"up to {dm} bonds, h {h:g}",
                                                      Kt + h * Kjob, y, blocks, only)
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
            models["invariant + Joback groups"].add(f"sigma {s:g}, h {h:g}", Kc + h * Kjob,
                                                    y, blocks, only)
        log(f"combined model at sigma {s:g}, {len(only)} blocks ({time.time() - t0:.0f} s)")
    cwidth = [models["eight counts"].best_label(k) for k in range(len(blocks))]
    Z8 = eight_counts(heavy)
    add_pairs(models, y, blocks, width, reach, cwidth,
              lambda s: invariant_kernels(features(mfs, fs, s, names, use_cache=not quick), s,
                                          n_f)[0],
              topo, lambda w: gaussian_at(Z8, w))
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
        log(f"  {name:38s} " + "  ".join(
            f"{t} {r[t]['q2_mean'] if t in SETS else r[t]['r2']:+.3f} "
            f"({r[t]['rmse_mean'] if t in SETS else r[t]['rmse']:.2f})" for t in TESTS))

    pairs = [("invariant, cosine", "Joback refitted"),
             ("invariant + Joback groups", "Joback refitted"),
             ("invariant, cosine", "linear regression on molecular weight"),
             ("invariant, not normalised", "invariant, cosine"),
             ("invariant, cosine", "field totals"),
             ("invariant, cosine", "eight counts"),
             ("invariant, cosine", "ECFP4"),
             ("invariant, cosine", "refitted Crippen increments"),
             ("invariant, cosine", "topological autocorrelation"),
             ("invariant + Joback groups", "topological + Joback groups"),
             ("topological autocorrelation", "refitted Crippen increments"),
             ("invariant + topological", "invariant, cosine"),
             ("invariant + eight counts", "invariant, cosine"),
             ("invariant + eight counts", "eight counts"),
             ("invariant + Joback groups", "invariant + eight counts")]
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
               "joback_groups": len(JOBACK),
               "models": res, "comparisons": comps, "sigmas": list(SIGMAS), "hs": list(HS),
               "lams": list(LAMS), "count_widths": list(COUNT_WIDTHS), "d_max": D_MAX,
               "folds": L.FOLDS, "repeats": REPEATS, "topo_max": list(TOPO_MAX),
               "crippen_types": len(crippen_types), "seconds": round(time.time() - t0, 1)}
    jdump(payload, os.path.join(RESULTS, "quick.json" if quick else "vapour.json"))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(os.path.join(CACHE, "predictions_quick.npz" if quick else "predictions.npz"),
             names=names, y=y, **{f"{m}|{t}": p for (m, t), p in preds.items()})
    if not quick:
        report(payload)
    return payload


def report(payload: Optional[dict] = None) -> str:
    if payload is None:
        with open(os.path.join(RESULTS, "vapour.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
    n = payload["compounds"]
    sd = payload["y"]
    lines = [f"Measured vapour pressures, logVP in mmHg near 25 C: the PHYSPROP set curated for "
             f"OPERA, its training part ({n['physprop_train']} compounds, spread "
             f"{sd['physprop_train']['sd']:.2f}) and test part ({n['physprop_test']}), and "
             f"vapour pressures from eChemPortal ({n['echemportal']} compounds, spread "
             f"{sd['echemportal']['sd']:.2f}). Five folds repeated three times on the training "
             f"part and on the eChemPortal set; the external tests fit on the whole training "
             f"part. The setting and the ridge are chosen by the evidence on the training "
             f"block. Each cell gives q² (cross-validation) or R² (external test), with the "
             f"RMSE in log units in brackets.", "",
             "| Model | " + " | ".join(TITLES[t] for t in TESTS) + " |",
             "|---|" + "---:|" * len(TESTS)]
    for name in MODELS:
        r = payload["models"][name]
        cells = [f"{r[t]['q2_mean']:+.3f} ({r[t]['rmse_mean']:.2f})" if t in SETS
                 else f"{r[t]['r2']:+.3f} ({r[t]['rmse']:.2f})" for t in TESTS]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
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
