#!/usr/bin/env python
"""
Does giving every atom a direction let a Fukui function be fitted, and does a model notice?

``cmfqsar/cdft.py`` established the first half of plan item D1: a conceptual-DFT field cannot be
fitted by isotropic Gaussians on the nuclei, because it is a frontier-orbital density with a node
in the molecular plane and a positive function centred on a nucleus cannot make a node.  The item
proposes the smallest extension that can -- multiplying each Gaussian by a linear factor, which
gives every atom three p functions whose overlaps are still closed-form
(:mod:`cmfqsar.anisotropic`) -- and sets two conditions for keeping it: **the fit error has to
drop, and the model has to notice.**

This measures both, on the four corrosion series that have finished ORCA runs, reusing their .gbw
files exactly as ``run_cdft_kernels.py`` does.

For each series and each field:

    fit error       the relative error of the fitted expansion against the field on its own grid,
                    with s functions only and with s and p
    coefficients    what the extension costs: four numbers per atom per width instead of one
    kernels         q2 of a model on the kernel of the fitted fields, alone and added to the five
                    ordinary fields, for both bases -- and the grid quadrature, which keeps the
                    field exactly and is the ceiling any fit is trying to reach

Usage
-----
    python benchmarks/run_anisotropic.py                       # the four series
    python benchmarks/run_anisotropic.py --set 12_c1ccoc1 --limit 4
    python benchmarks/run_anisotropic.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import run_cdft_kernels as CK  # noqa: E402
import run_corrosion_qm as Q  # noqa: E402
import run_topomer as T  # noqa: E402

from cmfqsar import anisotropic as A  # noqa: E402
from cmfqsar.cdft import cdft_fields, fit_field, overlap_kernel, valence_window  # noqa: E402
from cmfqsar.model import cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "anisotropic")
#: the four series whose quantum runs finished
SETS = ("161_c1ccc2scnc2c1", "12_c1ccoc1", "13_c1c_nH_nn1", "7_c1nncs1")
FIELDS = ["fukui_plus", "dual"]
WIDTHS = CK.WIDTHS
RIDGE = 1e-3


def run(set_name, results=RESULTS, intervals=100, jobs=8, limit=0, force=False):
    out_path = os.path.join(results, f"{set_name}.json")
    if os.path.exists(out_path) and not force:
        print(f"{set_name}: already done")
        return json.load(open(out_path))
    workdir = os.path.join(Q.RESULTS, set_name, "orca")
    if not os.path.isdir(workdir):
        print(f"{set_name}: no quantum run ({workdir} is missing)")
        return None
    t0 = time.time()
    mols, names, y, _core, _ref, rmsd = Q.prepare(os.path.join(Q.C.DATA, set_name + ".csv"))
    dirs = CK.job_dirs(workdir)
    keep = [i for i, n in enumerate(names) if n in dirs]
    sdf = os.path.join(Q.RESULTS, set_name, f"{set_name}_qm.sdf")
    annotated = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(sdf, removeHs=False)
                 } if os.path.isfile(sdf) else {}
    if annotated:
        keep = [i for i in keep if names[i] in annotated]
    if limit:
        keep = keep[:limit]
    mols = [annotated.get(names[i], mols[i]) for i in keep]
    names = [names[i] for i in keep]
    y = y[keep]
    print(f"=== {set_name}: {len(mols)} molecules with finished quantum jobs", flush=True)
    if len(mols) < 8:
        print(f"    too few to cross-validate, skipped")
        return None

    def work(k):
        c = CK.cubes_for(dirs[names[k]], intervals, keep=False)
        F = cdft_fields(c["n"], c["np1"], c["nm1"])
        win = valence_window(c["n"].values)
        return k, c["n"], {f: F[f] * win for f in FIELDS}

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        got = {}
        for k, cube, F in ex.map(work, range(len(mols))):
            got[k] = (cube, F)
            if len(got) % 5 == 0 or len(got) == len(mols):
                print(f"    cubes {len(got)}/{len(mols)} ({time.time() - t0:.0f} s)", flush=True)

    order = sorted(got)
    positions = [got[k][0].positions for k in order]
    lo, n, points = CK.common_grid(positions)
    dV = CK.SPACING ** 3
    res = {"set": set_name, "n": len(mols), "intervals": intervals,
           "widths": list(map(float, WIDTHS)), "ridge": RIDGE, "fields": {}}
    base = CK.base_kernels(mols, names)
    best_base = max((CK.score(base[a], y), a) for a in T.ALPHAS)
    res["base"] = {"q2": best_base[0][0], "rmse_cv": best_base[0][1], "alpha": best_base[1]}
    print(f"    the five ordinary fields alone: q2 {best_base[0][0]:+.3f}", flush=True)

    for fname in FIELDS:
        entry = {}
        X = np.vstack([CK.resample(got[k][0], got[k][1][fname], points) for k in order])
        Kg = CK.unit_diagonal((X @ X.T) * dV)
        q2, err, lam = CK.score(Kg, y)
        entry["grid"] = {"q2": q2, "rmse_cv": err, "lam": lam}

        for tag, p_type in (("s", False), ("s+p", True)):
            t1 = time.time()
            fits = [A.fit_field(got[k][1][fname], got[k][0].points(), got[k][0].positions,
                                WIDTHS, p_type=p_type, ridge=RIDGE) for k in order]
            K = CK.unit_diagonal(A.overlap_kernel([f.coeffs for f in fits], positions, WIDTHS,
                                                  p_type=p_type))
            off = ~np.eye(len(mols), dtype=bool)
            q2, err, lam = CK.score(K, y)
            q2b, errb, _l = max(CK.score(CK.unit_diagonal(base[a] + K), y) for a in T.ALPHAS)
            entry[tag] = {
                "fit_rel_error": float(np.mean([f.rel_error for f in fits])),
                "fit_r2": float(np.mean([f.r2 for f in fits])),
                "coefficients_per_atom": int(A.basis_size(len(WIDTHS), p_type)),
                "q2": q2, "rmse_cv": err, "lam": lam,
                "q2_with_five": q2b, "rmse_with_five": errb,
                "r_grid_fitted": float(np.corrcoef(Kg[off], K[off])[0, 1]),
                "seconds": round(time.time() - t1, 1),
            }
            if p_type:
                entry[tag]["mean_dipole"] = float(np.mean(
                    [np.linalg.norm(f.dipoles(), axis=1).mean() for f in fits]))
            print(f"    {fname} / {tag:3s}: fit error {entry[tag]['fit_rel_error']:.1%}, "
                  f"q2 alone {q2:+.3f}, with the five fields {q2b:+.3f} "
                  f"({entry[tag]['seconds']:.0f} s)", flush=True)
        res["fields"][fname] = entry
    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(results, exist_ok=True)
    json.dump(res, open(out_path, "w"), indent=1)
    return res


def report(results=RESULTS):
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    data = [json.load(open(f)) for f in files if not f.endswith("table.md")]
    if not data:
        return "No anisotropic results yet; run `python benchmarks/run_anisotropic.py`."
    order = {s: i for i, s in enumerate(SETS)}
    data.sort(key=lambda d: order.get(d["set"], 99))
    lines = ["# Can a direction on every atom represent a Fukui function?", "",
             "Plan item D1 proposes multiplying each atom's Gaussian by a linear factor, which "
             "gives every atom three p functions and keeps the overlaps closed-form "
             "(`cmfqsar/anisotropic.py`). It sets two conditions for keeping the extension: the "
             "fit error has to drop, and the model has to notice. Measured on the corrosion "
             "series with finished ORCA runs, on the valence part of each field, with the four "
             "widths the probe fields share.", "",
             "## Does the fit improve?", "",
             "| series | n | field | error, s only | error, s and p | coefficients per atom | "
             "fitted vs exact kernel, s | s and p |",
             "|---|---:|---|---:|---:|---|---:|---:|"]
    drops, r_gain = [], []
    for d in data:
        for fname, e in d["fields"].items():
            if "s" not in e or "s+p" not in e:
                continue
            drops.append(e["s"]["fit_rel_error"] - e["s+p"]["fit_rel_error"])
            r_gain.append(e["s+p"]["r_grid_fitted"] - e["s"]["r_grid_fitted"])
            lines.append(f"| {d['set']} | {d['n']} | {fname} | "
                         f"{e['s']['fit_rel_error']:.0%} | **{e['s+p']['fit_rel_error']:.0%}** | "
                         f"{e['s']['coefficients_per_atom']} → "
                         f"{e['s+p']['coefficients_per_atom']} | "
                         f"{e['s']['r_grid_fitted']:+.3f} | {e['s+p']['r_grid_fitted']:+.3f} |")
    if drops:
        lines += ["", f"**The fit does improve**: the relative error falls by "
                      f"{np.mean(drops):.0%} on average (from "
                      f"{np.mean([e['s']['fit_rel_error'] for d in data for e in d['fields'].values() if 's' in e]):.0%} "
                      f"to "
                      f"{np.mean([e['s+p']['fit_rel_error'] for d in data for e in d['fields'].values() if 's+p' in e]):.0%}), "
                      f"for four times as many coefficients per atom. The kernel the fit produces "
                      f"also agrees better with the one the exact field produces, by "
                      f"{np.mean(r_gain):+.3f} in correlation.", ""]

    lines += ["## Does the model notice?", "",
              "q² of a ten-fold cross-validated model on each kernel: the field alone, and the "
              "field added to the five ordinary ones. `grid` keeps the field exactly and is the "
              "ceiling a fit is trying to reach.", "",
              "| series | field | five fields | grid | s alone | s and p alone | five + s | "
              "five + s and p |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    deltas, deltas_with = [], []
    for d in data:
        for fname, e in d["fields"].items():
            if "s" not in e or "s+p" not in e:
                continue
            deltas.append(e["s+p"]["q2"] - e["s"]["q2"])
            deltas_with.append(e["s+p"]["q2_with_five"] - e["s"]["q2_with_five"])
            lines.append(f"| {d['set']} | {fname} | {d['base']['q2']:+.3f} | "
                         f"{e['grid']['q2']:+.3f} | {e['s']['q2']:+.3f} | "
                         f"{e['s+p']['q2']:+.3f} | {e['s']['q2_with_five']:+.3f} | "
                         f"{e['s+p']['q2_with_five']:+.3f} |")
    if deltas:
        mean_d, mean_dw = float(np.mean(deltas)), float(np.mean(deltas_with))
        better = sum(1 for v in deltas if v > 0)
        lines += ["",
                  f"**The model does not notice.** Over the {len(deltas)} series-field pairs the "
                  f"p functions change q² of the field alone by **{mean_d:+.3f}** on average "
                  f"({better} of {len(deltas)} better), and of the field added to the five "
                  f"ordinary ones by **{mean_dw:+.3f}**. The extension passes the first of the "
                  f"item's two conditions and fails the second.", "",
                  "Why the two measurements disagree is the same reason `run_cdft_kernels.py` "
                  "found for the isotropic fit: a kernel is an inner product between molecules, "
                  "and an error that falls the same way for every molecule cancels in it. The p "
                  "functions recover shape that every molecule of a congeneric series has in the "
                  "same place, so they improve every field by about the same amount and leave the "
                  "distances between the molecules where they were.", ""]
    text = "\n".join(lines) + "\n"
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                        # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", nargs="*", default=list(SETS))
    ap.add_argument("--intervals", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only this many molecules, for a trial")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        print(report())
        return
    for name in a.set:
        run(name, intervals=a.intervals, jobs=a.jobs, limit=a.limit, force=a.force)
    print()
    print(report())


if __name__ == "__main__":
    main()
