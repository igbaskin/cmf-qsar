#!/usr/bin/env python
"""
Does the shape of a conceptual-DFT field buy a model anything the condensed numbers do not?

``cmfqsar/cdft.py`` establishes that a Fukui function cannot be fitted by atom-centred isotropic
Gaussians -- it is a frontier-orbital density with a node in the molecular plane, and the fit
loses half of it.  What that does *not* settle is whether the model cares.  A kernel is an inner
product between molecules, and a per-molecule error that cancels in the same way for every
molecule costs a model nothing.  Only three kernels on the same molecules can say.

  grid        the field kept exactly, on one common grid, the kernel a quadrature:
              K_AB = sum_g F_A(g) F_B(g) dV.  Shape preserved, atom decomposition gone.
  fitted      the field fitted with Gaussians on the nuclei, the kernel the closed-form overlap
              CMF uses.  Everything of CMF's algebra kept, half the field lost.
  condensed   one Gaussian per atom weighted by the Hirshfeld-condensed index -- what the program
              already does with a prop: field, and what costs no grid at all.

Each is scored alone and added to the five ordinary fields, on the series' own alignment, with
the same lambda grid and the same 10-fold cross-validation as every other benchmark here.

This reuses the quantum calculations of ``run_corrosion_qm.py``: the .gbw files are already in its
working directories, so the only new cost is one ``orca_plot`` call per state per molecule.

Usage
-----
    python benchmarks/run_cdft_kernels.py --set 161_c1ccc2scnc2c1 [--intervals 100] [--jobs 8]
"""
from __future__ import annotations

import argparse
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
from scipy.ndimage import map_coordinates  # noqa: E402

import run_corrosion_qm as Q  # noqa: E402
import run_topomer as T  # noqa: E402

from cmfqsar.cdft import (cdft_fields, fit_field, overlap_kernel, read_cube,  # noqa: E402
                          run_orca_plot, valence_window)
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "cdft_kernels")
#: narrow enough that the coefficients do not cancel (see cmfqsar/cdft.py)
WIDTHS = np.geomspace(20.0, 0.5, 4)
FIELDS = ["fukui_plus", "dual"]
PAD = 4.0          # angstrom of vacuum around the series on the common grid
SPACING = 0.25     # angstrom of the common grid


def job_dirs(workdir):
    """{molecule name: directory} for every finished job of a previous quantum run."""
    out = {}
    for entry in sorted(os.listdir(workdir)):
        status = os.path.join(workdir, entry, "status.json")
        if not os.path.isfile(status):
            continue
        with open(status) as fh:
            st = json.load(fh)
        if st.get("status") == "done":
            out[st["name"]] = os.path.join(workdir, entry)
    return out


def cubes_for(path, intervals, keep=False):
    """The three densities of one molecule, as cubes on that molecule's own grid."""
    made = {}
    for tag, gbw in (("n", "sp_n.gbw"), ("np1", "sp_np1.gbw"), ("nm1", "sp_nm1.gbw")):
        made[tag] = run_orca_plot(os.path.join(path, gbw), intervals=intervals)
    cubes = {k: read_cube(v) for k, v in made.items()}
    if not keep:
        for v in made.values():
            os.remove(v)
    return cubes


def common_grid(positions, pad=PAD, spacing=SPACING):
    lo = np.min([p.min(0) for p in positions], axis=0) - pad
    hi = np.max([p.max(0) for p in positions], axis=0) + pad
    n = np.maximum(np.ceil((hi - lo) / spacing).astype(int) + 1, 2)
    axes = [lo[k] + np.arange(n[k]) * spacing for k in range(3)]
    return lo, n, np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def resample(cube, values, points):
    """Trilinear interpolation of one molecule's field onto the common grid; outside is zero."""
    idx = ((points - cube.origin) / np.array([cube.vectors[k][k] for k in range(3)])).T
    return map_coordinates(values, idx, order=1, mode="constant", cval=0.0).astype(np.float32)


def unit_diagonal(K):
    """Scale to unit mean diagonal, as the model does to every field kernel."""
    return K / max(float(np.mean(np.diag(K))), 1e-30)


def score(K, y, lams=T.LAMS):
    best = None
    for lam in lams:
        pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=T.FOLDS, seed=T.SEED)
        q2 = r2_score(y, pred)
        if best is None or q2 > best[0]:
            best = (q2, rmse(y, pred), float(lam))
    return best


def base_kernels(mols, names, alphas=T.ALPHAS):
    """The five ordinary fields at each attenuation, scaled as the model scales them."""
    out = {}
    for alpha in alphas:
        fs = FieldSet.create(physchem=T.FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(mf)))
    return out


def condensed_kernels(mols, names, prop, alphas=T.ALPHAS):
    """One Gaussian per atom weighted by the condensed index: the prop: field as it ships."""
    out = {}
    for alpha in alphas:
        fs = FieldSet.create(physchem=[], alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit", atom_properties=[prop])
        mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        out[alpha] = model.joint_kernel(model.prepare_training_kernels(AtomTable.from_molecules(mf)))
    return out


def run(set_name, results=RESULTS, qm_results=None, intervals=100, jobs=8, fields=FIELDS,
        keep_cubes=False, limit=0):
    qm_results = qm_results or Q.RESULTS
    workdir = os.path.join(qm_results, set_name, "orca")
    if not os.path.isdir(workdir):
        raise SystemExit(f"no quantum run for {set_name}: {workdir} is missing")
    t0 = time.time()
    mols, names, y, core, ref_i, rmsd = Q.prepare(os.path.join(Q.C.DATA, set_name + ".csv"))
    dirs = job_dirs(workdir)
    keep = [i for i, n in enumerate(names) if n in dirs]
    sdf = os.path.join(qm_results, set_name, f"{set_name}_qm.sdf")
    annotated = {m.GetProp("_Name"): m for m in Chem.SDMolSupplier(sdf, removeHs=False)
                 } if os.path.isfile(sdf) else {}
    keep = [i for i in keep if names[i] in annotated] if annotated else keep
    if limit:
        keep = keep[:limit]
    mols = [annotated.get(names[i], mols[i]) for i in keep]
    names = [names[i] for i in keep]
    y = y[keep]
    print(f"=== {set_name}: {len(mols)} molecules with finished quantum jobs "
          f"(median alignment RMSD {rmsd:.2f} A)", flush=True)

    def work(k):
        c = cubes_for(dirs[names[k]], intervals, keep=keep_cubes)
        F = cdft_fields(c["n"], c["np1"], c["nm1"])
        win = valence_window(c["n"].values)
        return k, c["n"], {f: F[f] * win for f in fields}

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        got = {}
        for k, cube, F in ex.map(work, range(len(mols))):
            got[k] = (cube, F)
            print(f"    cubes {len(got)}/{len(mols)}: {names[k]} ({time.time() - t0:.0f} s)",
                  flush=True)

    lo, n, points = common_grid([got[k][0].positions for k in got])
    dV = SPACING ** 3
    print(f"    common grid {tuple(n)} at {SPACING} A = {len(points)} points", flush=True)

    res = {"set": set_name, "n": len(mols), "intervals": intervals, "spacing": SPACING,
           "widths": list(map(float, WIDTHS)), "lams": list(T.LAMS), "folds": T.FOLDS}
    base = base_kernels(mols, names)
    q2, err, lam = score(base[T.ALPHAS[0]], y)
    best_base = max((score(base[a], y), a) for a in T.ALPHAS)
    res["base"] = {"q2": best_base[0][0], "rmse_cv": best_base[0][1], "alpha": best_base[1],
                   "lam": best_base[0][2]}
    print(f"    {'base':22s} q2 {best_base[0][0]:+.3f}", flush=True)

    for fname in fields:
        X = np.vstack([resample(got[k][0], got[k][1][fname], points) for k in sorted(got)])
        Kg = unit_diagonal((X @ X.T) * dV)
        fits = [fit_field(got[k][1][fname], got[k][0], widths=WIDTHS, ridge=1e-3)
                for k in sorted(got)]
        Kf = unit_diagonal(overlap_kernel([f.coeffs for f in fits],
                                          [got[k][0].positions for k in sorted(got)], WIDTHS))
        Kc = condensed_kernels(mols, names, fname)
        off = ~np.eye(len(mols), dtype=bool)
        r_gf = float(np.corrcoef(Kg[off], Kf[off])[0, 1])
        rel = float(np.mean([f.rel_error for f in fits]))
        print(f"    --- {fname}: mean fit error {rel:.1%}, "
              f"grid vs fitted kernel r {r_gf:+.3f}", flush=True)
        entry = {"fit_rel_error": rel, "r_grid_fitted": r_gf}
        for tag, K in (("grid", Kg), ("fitted", Kf)):
            q2, err, lam = score(K, y)
            entry[tag] = {"q2": q2, "rmse_cv": err, "lam": lam}
            q2b, errb, lamb = max(score(unit_diagonal(base[a] + K), y) for a in T.ALPHAS)
            entry["base+" + tag] = {"q2": q2b, "rmse_cv": errb, "lam": lamb}
            print(f"    {tag:10s} alone q2 {q2:+.3f}   with the five fields q2 {q2b:+.3f}",
                  flush=True)
        best_c = max((score(Kc[a], y), a) for a in T.ALPHAS)
        entry["condensed"] = {"q2": best_c[0][0], "rmse_cv": best_c[0][1], "alpha": best_c[1],
                              "lam": best_c[0][2]}
        best_cb = max((score(unit_diagonal(base[a] + Kc[a]), y), a) for a in T.ALPHAS)
        entry["base+condensed"] = {"q2": best_cb[0][0], "rmse_cv": best_cb[0][1],
                                   "alpha": best_cb[1], "lam": best_cb[0][2]}
        r_gc = float(np.corrcoef(Kg[off], Kc[best_c[1]][off])[0, 1])
        entry["r_grid_condensed"] = r_gc
        print(f"    {'condensed':10s} alone q2 {best_c[0][0]:+.3f}   with the five fields "
              f"q2 {best_cb[0][0]:+.3f}   (grid vs condensed kernel r {r_gc:+.3f})", flush=True)
        res[fname] = entry

    res["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    out = os.path.join(results, f"{set_name}.json")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out} ({res['seconds']:.0f} s)", flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="161_c1ccc2scnc2c1")
    ap.add_argument("--intervals", type=int, default=100, help="orca_plot grid intervals")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--fields", nargs="*", default=FIELDS)
    ap.add_argument("--keep-cubes", action="store_true", help="do not delete the cubes (GB)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N molecules (a probe)")
    ap.add_argument("--qm-results", default=None)
    args = ap.parse_args(argv)
    run(args.set, intervals=args.intervals, jobs=args.jobs, fields=args.fields,
        keep_cubes=args.keep_cubes, qm_results=args.qm_results, limit=args.limit)


if __name__ == "__main__":
    main()
