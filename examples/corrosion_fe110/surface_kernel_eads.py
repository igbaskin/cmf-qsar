"""The E_ads model with the surface kernel: no in-plane alignment at all.

The surface autocorrelation kernel (cmfqsar.invariant.SurfaceSpec, `kernel_source="surface"`)
describes a molecule over a surface by the in-plane distance distribution of its field weights
resolved by the heights of both atoms: invariant to translation in the plane and rotation about
the normal, sensitive to height and tilt.  Here it is run on the UMA poses exactly as the
adsorption script wrote them (z = 0 at the top Fe layer), with no MCS registration in the plane,
and on a copy of those poses turned and shifted at random in the plane, which must give the same
numbers.  Beside it: the three-dimensional autocorrelation kernel, which is invariant to every
rigid motion and so throws the height away, and the overlap-integral model in the registered
frame (compare_alignments.py, q2 0.52).  Same molecules, same nested cross-validation.

    python examples/corrosion_fe110/surface_kernel_eads.py [--repeats 10]

Writes surface_kernel_eads.csv next to this script and prints the table.
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
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import compare_alignments as ca  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, nested_cv  # noqa: E402


def shuffled_in_plane(mols, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for m in mols:
        mm = Chem.Mol(m)
        xyz = np.array(mm.GetConformer().GetPositions())
        t = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
        c = xyz.mean(axis=0)
        new = (xyz - c) @ R.T + c + np.array([rng.uniform(-6, 6), rng.uniform(-6, 6), 0.0])
        conf = mm.GetConformer()
        for i, p in enumerate(new):
            conf.SetAtomPosition(i, [float(v) for v in p])
        out.append(mm)
    return out


def evaluate(mols, names, y, fs, model, outer, repeats, seed, alphas):
    mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
    train = AtomTable.from_molecules(mf)
    t0 = time.time()
    res = nested_cv(train, y, model, alphas, lams=ca.LAMS, outer_folds=outer, inner_folds=10,
                    repeats=repeats, seed=seed, precompute=model.uses_fields)
    return res, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--csv", default=ca.CSV, help="the node CSV (default: the benzimidazole node)")
    ap.add_argument("--sdf", default=ca.SDF, help="its UMA poses (default: 157_benzimidazole_uma.sdf)")
    ap.add_argument("--tag", default="", help="suffix of the output table")
    ap.add_argument("--scramble", type=int, default=0, help="y-scrambling runs of the surface kernel (0 = none)")
    a = ap.parse_args(argv)
    fr = ca.frames(a.csv, a.sdf)
    fs = FieldSet.create()
    rows = []

    def run(label, mols, names, y, model, alphas):
        res, dt = evaluate(mols, names, y, fs, model, a.outer, a.repeats, a.seed, alphas)
        rows.append({"model": label, "n": len(y), "q2": round(res.q2_ex, 3), "q2_sd": round(res.q2_ex_sd, 3),
                     "rmse": round(res.rmse_ex, 2), "seconds": round(dt)})
        print(f"{label:52s} n={len(y)} q2 = {res.q2_ex:6.3f} +- {res.q2_ex_sd:.3f}  rmse {res.rmse_ex:.1f}  {dt:.0f} s", flush=True)

    for frame, label in (("pose", "poses as written"), ("pose+mcs", "poses registered in the plane")):
        mols, names, _, e_ads, gas_ok = fr[frame]
        keep = np.flatnonzero(np.isfinite(e_ads) & gas_ok)
        mols = [mols[i] for i in keep]; names = [names[i] for i in keep]; y = e_ads[keep]
        run(f"overlap fields, {label}", mols, names, y, CMFModel(fieldset=fs, method="krr", fit_bias=True), ca.ALPHAS)
        if frame == "pose":
            for sigma, (z0, z1, dz) in ((0.75, (1.0, 6.0, 1.0)), (0.5, (1.0, 6.0, 0.5)), (1.0, (1.0, 7.0, 1.5))):
                m = CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="surface",
                             ac_sigma=sigma, surf_z_min=z0, surf_z_max=z1, surf_z_step=dz)
                run(f"surface kernel sigma {sigma:g}, z {z0:g}-{z1:g} step {dz:g}, {label}", mols, names, y, m, [1.0])
            shuffled = shuffled_in_plane(mols, a.seed)
            run("surface kernel sigma 0.75, z 1-6 step 1, poses shuffled in the plane", shuffled, names, y,
                CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="surface"), [1.0])
            run("3D autocorrelation (height thrown away), poses as written", mols, names, y,
                CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="autocorrelation"), [1.0])
            run("overlap fields, poses shuffled in the plane (what alignment-free costs the overlap)", shuffled, names, y,
                CMFModel(fieldset=fs, method="krr", fit_bias=True), ca.ALPHAS)
            if a.scramble:
                rng_s = np.random.default_rng(a.seed + 1)
                q2s = []
                for _ in range(a.scramble):
                    res, _ = evaluate(mols, names, rng_s.permutation(y), fs,
                                      CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="surface"),
                                      a.outer, 1, a.seed, [1.0])
                    q2s.append(res.q2_ex)
                rows.append({"model": f"surface kernel, y scrambled x{a.scramble}: max q2", "n": len(y),
                             "q2": round(max(q2s), 3), "q2_sd": round(float(np.std(q2s)), 3), "rmse": "", "seconds": ""})
                print(f"surface kernel, y scrambled: max q2 {max(q2s):.3f}, mean {np.mean(q2s):.3f} over {a.scramble}", flush=True)
    out = os.path.join(HERE, "surface_kernel_eads" + (f"_{a.tag}" if a.tag else "") + ".csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
