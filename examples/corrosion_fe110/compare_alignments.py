"""Does the adsorbed-pose frame beat the MCS alignment as the frame of a CMF model?

The same molecules (the benzimidazole node), the same fields, learner and nested
cross-validation, three frames:

- ``mcs``: the conventional one, gas-phase ETKDG+MMFF conformers superimposed on the node
  scaffold (benzimidazole, the first molecule) by their maximum common substructure, in 3D;
- ``pose``: the UMA-relaxed adsorbed poses as written (z = surface normal, the top Fe layer at
  z = 0), which fixes tilt and height physically but leaves the in-plane rotation and position
  where the relaxation left them;
- ``pose+mcs``: the adsorbed poses registered in the surface plane only: the MCS atoms are
  superimposed on the reference pose by a rotation about z and a translation in x, y (Kabsch in
  two dimensions), so the physics keeps the vertical and the common substructure sets the lateral.

Targets: the electrochemical potency (`activity`) in every frame, and the UMA adsorption energy
`E_ads_kJmol` (without the one molecule whose gas-phase reference is an artefact) in the frames
that carry it and in the MCS frame.  KRR on the five physicochemical fields, alpha and lambda
by 10-fold inner cross-validation, 5 outer folds, 3 repeats, the same seeds for every frame.

    python examples/corrosion_fe110/compare_alignments.py [--repeats 3] [--outer 5]

Writes alignment_comparison.csv next to this script and prints the table.
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

from cmfqsar.align import align_dataset, mcs_atom_map  # noqa: E402
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, nested_cv, r2_score  # noqa: E402

CSV = os.path.join(ROOT, "examples", "corrosion", "157_c1ccc2_nH_cnc2c1.csv")
SDF = os.path.join(HERE, "157_benzimidazole_uma.sdf")
ALPHAS = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
LAMS = [10.0 ** k for k in range(-4, 2)]


def coords(m):
    return np.array(m.GetConformer().GetPositions(), dtype=float)


def set_coords(m, xyz):
    conf = m.GetConformer()
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, [float(v) for v in p])


def register_in_plane(mol: Chem.Mol, ref: Chem.Mol) -> tuple:
    """Rotate `mol` about z and translate it in x, y so that its MCS atoms best match those of
    `ref` (Kabsch in two dimensions); z is untouched.  Returns (rmsd_xy, n_matched)."""
    amap = mcs_atom_map(mol, ref)
    p = coords(mol)
    q = coords(ref)
    a = np.array([p[i, :2] for i, _ in amap])
    b = np.array([q[j, :2] for _, j in amap])
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    h = (a - ca).T @ (b - cb)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T                       # a proper rotation in the plane
    xy = (p[:, :2] - ca) @ r.T + cb
    out = p.copy()
    out[:, :2] = xy
    set_coords(mol, out)
    rmsd = float(np.sqrt(((a - ca) @ r.T + cb - b).__pow__(2).sum(axis=1).mean()))
    return rmsd, len(amap)


def frames():
    """name -> (mols, names, potency, E_ads or None, gas_intact)."""
    out = {}
    ds = load_dataset(CSV, activity_prop="activity")
    log = align_dataset(ds.mols, 0, use_mcs=True)
    failed = [l for l in log if "FAILED" in l]
    print(f"mcs frame: {len(ds)} molecules aligned on {ds.names[0]}, {len(failed)} failures")
    out["mcs"] = (ds.mols, ds.names, np.asarray(ds.activity, float), None, np.ones(len(ds), bool))

    ds = load_dataset(SDF, activity_prop="activity")
    e_ads = np.array([float(m.GetProp("E_ads_kJmol")) for m in ds.mols])
    gas_ok = np.array([m.GetProp("gas_intact") == "1" if m.HasProp("gas_intact") else True for m in ds.mols])
    bound = np.array([m.GetProp("bound") == "1" for m in ds.mols])
    print(f"pose frame: {len(ds)} molecules, {bound.sum()} chemisorbed, {(~gas_ok).sum()} with an "
          f"unreliable gas-phase reference")
    out["pose"] = ([Chem.Mol(m) for m in ds.mols], ds.names, np.asarray(ds.activity, float), e_ads, gas_ok)

    mols = [Chem.Mol(m) for m in ds.mols]
    rms = []
    for m in mols[1:]:
        try:
            rms.append(register_in_plane(m, mols[0])[0])
        except Exception as exc:  # noqa: BLE001
            print(f"  in-plane registration failed for {m.GetProp('_Name')}: {exc}")
    print(f"pose+mcs frame: in-plane MCS registration on {ds.names[0]}, median rmsd {np.median(rms):.2f} A, "
          f"max {max(rms):.2f} A")
    out["pose+mcs"] = (mols, ds.names, np.asarray(ds.activity, float), e_ads, gas_ok)
    # the E_ads of the MCS frame is the same molecule's, by name
    by_name = dict(zip(ds.names, e_ads))
    ok_by_name = dict(zip(ds.names, gas_ok))
    m_mols, m_names, m_y, _, _ = out["mcs"]
    out["mcs"] = (m_mols, m_names, m_y, np.array([by_name.get(n, np.nan) for n in m_names]),
                  np.array([ok_by_name.get(n, True) for n in m_names]))
    return out


def evaluate(mols, names, y, fs, outer, repeats, seed):
    mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
    train = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", fit_bias=True)
    t0 = time.time()
    res = nested_cv(train, y, model, ALPHAS, lams=LAMS, outer_folds=outer, inner_folds=10,
                    repeats=repeats, seed=seed, precompute=not fs.descriptors)   # a descriptor block needs the table
    return res, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--outer", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scramble", type=int, default=0, help="y-scrambling runs per model (0 = none)")
    a = ap.parse_args(argv)
    fs = FieldSet.create()                     # the five physicochemical fields, Gasteiger charges
    fr = frames()
    rows = []
    for target in ("potency", "E_ads"):
        for name in ("mcs", "pose", "pose+mcs"):
            mols, names, pot, e_ads, gas_ok = fr[name]
            if target == "potency":
                y, keep = pot, np.isfinite(pot)
            else:
                y, keep = e_ads, np.isfinite(e_ads) & gas_ok
            idx = np.flatnonzero(keep)
            res, dt = evaluate([mols[i] for i in idx], [names[i] for i in idx], y[idx], fs,
                               a.outer, a.repeats, a.seed)
            yp = res.y_pred
            r_pear = float(np.corrcoef(y[idx], yp)[0, 1])
            row = {"target": target, "frame": name, "n": len(idx), "q2_ex": round(res.q2_ex, 3),
                   "q2_sd": round(res.q2_ex_sd, 3), "rmse_ex": round(res.rmse_ex, 3),
                   "q2_aggr": round(res.q2_ex_aggr, 3), "r_pred": round(r_pear, 3),
                   "sd_y": round(float(np.std(y[idx])), 3), "seconds": round(dt)}
            if a.scramble:
                from cmfqsar.validation import y_scramble
                rng = np.random.default_rng(a.seed)
                q2s = []
                for _ in range(a.scramble):
                    ys = rng.permutation(y[idx])
                    rs, _ = evaluate([mols[i] for i in idx], [names[i] for i in idx], ys, fs, a.outer, 1, a.seed)
                    q2s.append(rs.q2_ex)
                row["q2_scrambled_max"] = round(max(q2s), 3)
            rows.append(row)
            print(f"{target:8s} {name:9s} n={len(idx):3d}  q2_ex = {res.q2_ex:6.3f} +- {res.q2_ex_sd:.3f}  "
                  f"rmse_ex = {res.rmse_ex:.3f} (sd y {np.std(y[idx]):.3f})  r = {r_pear:.3f}  {dt:.0f} s", flush=True)
    out = os.path.join(HERE, "alignment_comparison.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
