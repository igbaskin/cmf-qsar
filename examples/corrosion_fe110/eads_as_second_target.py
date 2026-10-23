"""Does the UMA adsorption energy help a potency model, as a descriptor or as a second task?

Three ways of letting E_ads into a CMF model of the benzimidazole node, in the MCS frame and in
the adsorbed-pose frame (registered in the surface plane), all under the same external 5-fold
cross-validation repeated 10 times:

- ``fields``: the five physicochemical fields alone (the reference from compare_alignments.py);
- ``fields+E_ads``: E_ads as a non-field descriptor block beside the fields (one number per
  molecule, standardised on the training fold, mixing coefficient h_d);
- ``multitask``: potency and E_ads as two endpoints on one field kernel with a task correlation
  rho chosen by the inner cross-validation (rho = 0 is the single-task model on the same folds,
  reported beside it so the sharing is measured against something);
- ``E_ads only``: ridge regression on the single number, the floor.

Targets: the raw MWILA in 1 M HCl (`activity`), the molecular-weight-adjusted one
(`mwila_hcl_adj`) and the papers' Langmuir free energy (`dG_ads`, 59 molecules).  The molecule
whose gas-phase reference is an artefact is left out everywhere.

    python examples/corrosion_fe110/eads_as_second_target.py [--repeats 10] [--outer 5]

Writes eads_second_target.csv next to this script and prints the table.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import compare_alignments as ca  # noqa: E402
from cmfqsar import multitask as mt  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import combine_kernels, kernel_matrices  # noqa: E402
from cmfqsar.model import _kfold_indices, r2_score, rmse  # noqa: E402

TARGETS = {"potency": "activity", "mwila_adj": "mwila_hcl_adj", "dG_ads": "dG_ads"}
RHOS = (0.0, 0.3, 0.5, 0.7, 0.9)


def prop(m, name):
    try:
        return float(m.GetProp(name)) if m.HasProp(name) and m.GetProp(name).strip() else np.nan
    except ValueError:
        return np.nan


def combined_kernels(mols, names, alphas):
    """alpha -> the joint field kernel over all molecules, each field scaled to unit mean diagonal
    (what CMFModel does with scale_fields=True and no cosine normalisation)."""
    out = {}
    for a in alphas:
        fs = FieldSet.create(alpha=a)
        table = AtomTable.from_molecules([build_fields(m, fs, name=n) for m, n in zip(mols, names)])
        K_f = kernel_matrices(table, table, fs.specs)
        scale = np.array([1.0 / max(np.mean(np.diag(K_f[f])), 1e-12) for f in range(K_f.shape[0])])
        out[a] = combine_kernels(K_f, fs.mixing_vector(), scale)
    return out


def multitask_cv(Ks, Y, outer, repeats, seed, lams, rhos=RHOS):
    """External CV of the two-task model; returns (pred with rho searched, pred with rho = 0),
    both for task 0, averaged over repeats, and the rhos chosen."""
    n = Y.shape[0]
    preds = {"searched": np.zeros((repeats, n)), "rho0": np.zeros((repeats, n))}
    chosen = []
    for rep in range(repeats):
        folds = _kfold_indices(n, outer, None if seed is None else seed + rep)
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            obs_tr = mt.Observations.from_columns(Y[train], names=("y", "E_ads"))
            best = None
            for a, K in Ks.items():
                K_tr = K[np.ix_(train, train)]
                rho, lam, score, _ = mt.search(K_tr, obs_tr, rhos=rhos, lams=lams, folds=10,
                                                seed=seed, statistic="y")
                if best is None or score > best[0]:
                    best = (score, a, rho, lam)
            _, a, rho, lam = best
            K = Ks[a]
            K_tr = K[np.ix_(train, train)]
            K_te = K[np.ix_(test, train)]
            fit = mt.fit(K_tr, obs_tr, rho=rho, lam=lam)
            preds["searched"][rep, test] = fit.predict(K_te, 0)
            # the same folds and alpha, no sharing
            best0 = None
            for a0, K0 in Ks.items():
                r0, l0, s0, _ = mt.search(K0[np.ix_(train, train)], obs_tr, rhos=(0.0,), lams=lams,
                                          folds=10, seed=seed, statistic="y")
                if best0 is None or s0 > best0[0]:
                    best0 = (s0, a0, l0)
            _, a0, l0 = best0
            fit0 = mt.fit(Ks[a0][np.ix_(train, train)], obs_tr, rho=0.0, lam=l0)
            preds["rho0"][rep, test] = fit0.predict(Ks[a0][np.ix_(test, train)], 0)
            chosen.append(rho)
    return preds, chosen


def ridge_cv(x, y, outer, repeats, seed):
    """Out-of-fold predictions of a one-variable ridge (lambda tiny: it is one slope)."""
    n = len(y)
    preds = np.zeros((repeats, n))
    for rep in range(repeats):
        for test in _kfold_indices(n, outer, None if seed is None else seed + rep):
            train = np.setdiff1d(np.arange(n), test)
            A = np.column_stack([x[train], np.ones(len(train))])
            coef = np.linalg.lstsq(A, y[train], rcond=None)[0]
            preds[rep, test] = coef[0] * x[test] + coef[1]
    return preds


def stats(y, preds):
    q2 = [r2_score(y, p) for p in preds]
    return float(np.mean(q2)), float(np.std(q2)), float(np.mean([rmse(y, p) for p in preds]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--descriptor-h", type=float, nargs="+", default=[1.0])
    a = ap.parse_args(argv)
    fr = ca.frames()
    rows = []
    for frame in ("mcs", "pose+mcs"):
        mols, names, _, e_ads, gas_ok = fr[frame]
        # the pose molecules carry E_ads and the CSV columns; the MCS ones need E_ads attached
        pose_by_name = {n: m for m, n in zip(fr["pose+mcs"][0], fr["pose+mcs"][1])}
        for m, n in zip(mols, names):
            src = pose_by_name[n]
            for p in ("E_ads_kJmol", "mwila_hcl_adj", "dG_ads", "activity"):
                if src.HasProp(p) and not m.HasProp(p):
                    m.SetProp(p, src.GetProp(p))
        Ks = None
        for target, pname in TARGETS.items():
            y_all = np.array([prop(m, pname) for m in mols])
            keep = np.isfinite(y_all) & np.isfinite(e_ads) & gas_ok
            idx = np.flatnonzero(keep)
            sub_mols = [mols[i] for i in idx]
            sub_names = [names[i] for i in idx]
            y = y_all[idx]
            x = e_ads[idx]
            t0 = time.time()
            # fields alone and fields + descriptor block
            for h in [None] + list(a.descriptor_h):
                fs = FieldSet.create() if h is None else FieldSet.create(descriptors=["E_ads_kJmol"], descriptor_h=h)
                res, dt = ca.evaluate(sub_mols, sub_names, y, fs, a.outer, a.repeats, a.seed)
                label = "fields" if h is None else f"fields+E_ads (h_d={h:g})"
                rows.append({"target": target, "frame": frame, "model": label, "n": len(idx),
                             "q2": round(res.q2_ex, 3), "q2_sd": round(res.q2_ex_sd, 3),
                             "rmse": round(res.rmse_ex, 3), "sd_y": round(float(np.std(y)), 3)})
                print(f"{target:10s} {frame:9s} {label:24s} n={len(idx):3d} q2 = {res.q2_ex:6.3f} +- {res.q2_ex_sd:.3f} "
                      f"rmse {res.rmse_ex:.3f}", flush=True)
            # E_ads alone
            q2, sd, rm = stats(y, ridge_cv(x, y, a.outer, a.repeats, a.seed))
            rows.append({"target": target, "frame": frame, "model": "E_ads only (ridge)", "n": len(idx),
                         "q2": round(q2, 3), "q2_sd": round(sd, 3), "rmse": round(rm, 3), "sd_y": round(float(np.std(y)), 3)})
            print(f"{target:10s} {frame:9s} {'E_ads only (ridge)':24s} n={len(idx):3d} q2 = {q2:6.3f} +- {sd:.3f} rmse {rm:.3f}", flush=True)
            # multitask: potency + E_ads on one kernel
            Ks = combined_kernels(sub_mols, sub_names, ca.ALPHAS)
            Y = np.column_stack([y, x])
            preds, chosen = multitask_cv(Ks, Y, a.outer, a.repeats, a.seed, ca.LAMS)
            for key, label in (("rho0", "multitask rho=0 (single)"), ("searched", "multitask rho searched")):
                q2, sd, rm = stats(y, preds[key])
                row = {"target": target, "frame": frame, "model": label, "n": len(idx),
                       "q2": round(q2, 3), "q2_sd": round(sd, 3), "rmse": round(rm, 3), "sd_y": round(float(np.std(y)), 3)}
                if key == "searched":
                    row["rho_median"] = round(float(np.median(chosen)), 2)
                rows.append(row)
                print(f"{target:10s} {frame:9s} {label:24s} n={len(idx):3d} q2 = {q2:6.3f} +- {sd:.3f} rmse {rm:.3f}"
                      + (f"  rho median {np.median(chosen):.1f}" if key == "searched" else ""), flush=True)
            print(f"  ({time.time() - t0:.0f} s)", flush=True)
    out = os.path.join(HERE, "eads_second_target.csv")
    keys = ["target", "frame", "model", "n", "q2", "q2_sd", "rmse", "sd_y", "rho_median"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
