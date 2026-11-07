#!/usr/bin/env python
"""
Posed by rule against relaxed on a slab: the planar aligner on the Fe(110) adsorption sets (EC2).

The Fe(110) sets (``examples/corrosion_fe110``) hold, for two nodes of the corrosion-inhibitor
scaffold tree, the adsorption energy of every molecule on bcc Fe(110) from the UMA potential and
the relaxed adsorbed pose in the surface frame: the normal along z, the top Fe layer at z = 0.
Those poses cost about five minutes of GPU per molecule.  ``cmfqsar/planealign.py`` poses a
molecule in the same frame by rule in milliseconds -- flat with the flattest face down, upright on
its most basic nitrogen, or whichever of those is most similar in field to one reference pose --
and this script asks what the rule is worth, in three measurements.

1. **Pose recovery.**  Starting from the relaxed geometry turned at random (so only the
   orientation has to be found), and from an ETKDG+MMFF conformer of the SMILES (what a user
   without a slab has), how far is the rule pose from the relaxed one?  Measured as the RMSD over
   heavy atoms after the best rotation about the normal and translation in the plane, since the
   lattice site a molecule sits over is not part of a pose, and as the RMSD of the heights alone.
   Reported for every molecule and for the chemisorbed ones, whose relaxed poses mean something;
   a detached molecule's pose is wherever the relaxation left it.

2. **The adsorption-energy model in each frame.**  The overlap kernel on the five fields in the
   relaxed frame (the ceiling the Fe(110) study established) and in the rule frames, the
   surface autocorrelation kernel on the same frames, and the conventional MCS frame of the
   gas-phase conformers.  Same molecules, same nested cross-validation (5 outer folds, 10 inner,
   10 repeats, one seed), each difference with its paired bootstrap interval over molecules.

3. **The surface co-field.**  The overlap model fitted in the best rule frame of the conformers is
   linear in the fields, so its co-field can be drawn on planes parallel to the surface; the
   surface kernel cannot draw one.  The co-field at 2.0 and 3.5 A is compared with the co-field
   of the same model in the relaxed frame, field by field, as a correlation over the plane.

Usage
-----
    python benchmarks/run_fe110.py --node benzimidazole
    python benchmarks/run_fe110.py --node quinoline
    python benchmarks/run_fe110.py --report
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.align import align_dataset, embed_3d, random_rigid_transform, set_coords  # noqa: E402
from cmfqsar.cofields import evaluate_gaussians  # noqa: E402
from cmfqsar.dataio import load_dataset, relative_path  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, grid_search, nested_cv, r2_score  # noqa: E402
from cmfqsar.planealign import (height_rmsd, in_plane_rmsd, plane_align_dataset, rigid_rmsd,  # noqa: E402
                                rule_pose)

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "fe110")
NODES = {
    "benzimidazole": os.path.join(ROOT, "examples", "corrosion_fe110", "157_benzimidazole_uma.sdf"),
    "quinoline": os.path.join(ROOT, "examples", "corrosion_fe110", "158_quinoline_uma.sdf"),
}
ALPHAS = [0.3, 0.5, 0.8, 1.2, 2.0, 3.0]
LAMS = [10.0 ** k for k in range(-4, 2)]
OUTER, INNER, REPEATS, SEED = 5, 10, 10, 0
HEIGHT = 1.9          # where the chemisorbed rings end up (median lowest heavy atom, both nodes); 2.2 is where they start

# the frames, in the order the table shows them
FRAMES = [
    ("relaxed", "relaxed poses as written"),
    ("relaxed/level", "relaxed, lowered to the common height"),
    ("relaxed/flat", "rule: flat, from the relaxed geometry"),
    ("relaxed/flat-up", "rule: flat, the other face, relaxed geometry"),
    ("relaxed/auto+plane", "rule: auto by field similarity, refined in the plane, relaxed geometry"),
    ("relaxed/auto+tilt", "rule: auto, refined with tilt and height, relaxed geometry"),
    ("conformer/flat", "rule: flat, from an MMFF conformer"),
    ("conformer/auto+plane", "rule: auto, refined in the plane, MMFF conformer"),
    ("conformer/auto+tilt", "rule: auto, refined with tilt and height, MMFF conformer"),
    ("conformer/mcs", "MCS alignment of the conformers (the conventional frame)"),
]
SURFACE_FRAMES = ["relaxed", "relaxed/level", "relaxed/flat", "conformer/flat", "conformer/auto+tilt"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_node(node):
    ds = load_dataset(NODES[node], activity_prop="activity")
    e_ads = np.array([float(m.GetProp("E_ads_kJmol")) for m in ds.mols])
    gas_ok = np.array([m.GetProp("gas_intact") == "1" if m.HasProp("gas_intact") else True for m in ds.mols])
    bound = np.array([m.GetProp("bound") == "1" for m in ds.mols])
    labels = [m.GetProp("pose_label") if m.HasProp("pose_label") else "?" for m in ds.mols]
    smiles = [m.GetProp("smiles") for m in ds.mols]
    keep = np.flatnonzero(np.isfinite(e_ads) & gas_ok)
    mols = [Chem.Mol(ds.mols[i]) for i in keep]
    names = [ds.names[i] for i in keep]
    return (mols, names, e_ads[keep], bound[keep], [labels[i] for i in keep], [smiles[i] for i in keep])


def heavy(m):
    return np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()])


def xyz(m):
    return np.array(m.GetConformer().GetPositions(), dtype=float)


def conformer_of(smiles, relaxed, seed=0):
    """An ETKDG+MMFF conformer of `smiles` with its heavy atoms in the order of `relaxed`, so
    that the two can be compared atom by atom.  The relaxed molecule came from the same SMILES
    through RDKit, so the orders agree; a substructure match checks it and repairs the rare
    case where they do not."""
    conf = embed_3d(Chem.MolFromSmiles(smiles), seed=seed)
    ref_heavy = Chem.RemoveHs(relaxed)
    con_heavy = Chem.RemoveHs(conf)
    match = ref_heavy.GetSubstructMatch(con_heavy)
    if len(match) != con_heavy.GetNumAtoms():
        raise RuntimeError(f"conformer does not match the relaxed molecule: {smiles}")
    order = list(match)                       # conformer heavy atom i  ->  relaxed heavy atom order[i]
    return conf, np.argsort(order)            # relaxed heavy atom j  <-  conformer heavy atom argsort[j]


def pose_errors(posed, relaxed, perm=None):
    """(in-plane RMSD, height RMSD, rigid floor) over heavy atoms."""
    a = xyz(posed)[heavy(posed)]
    b = xyz(relaxed)[heavy(relaxed)]
    if perm is not None:
        a = a[perm]
    return in_plane_rmsd(a, b), height_rmsd(a, b), rigid_rmsd(a, b)


# --------------------------------------------------------------------------- #
# Frames
# --------------------------------------------------------------------------- #

def scrambled(mols, seed):
    rng = np.random.RandomState(seed)
    out = []
    for m in mols:
        mm = Chem.Mol(m)
        set_coords(mm, random_rigid_transform(xyz(mm), rng, max_rotation=None, max_translation=3.0))
        out.append(mm)
    return out


def levelled(mols, height):
    """The relaxed poses with every molecule shifted along z so that its lowest heavy atom is at
    `height`: tilt, face and in-plane arrangement kept, the outcome of the relaxation (bound at 2 A,
    detached at 4-5 A) removed."""
    out = []
    for m in mols:
        mm = Chem.Mol(m)
        c = xyz(mm)
        c[:, 2] += height - c[heavy(mm), 2].min()
        set_coords(mm, c)
        out.append(mm)
    return out


def make_frames(mols, names, smiles, fs, seed=SEED, height=HEIGHT, progress=print):
    """name -> (molecules in that frame, permutation of heavy atoms to the relaxed order or None)."""
    frames = {"relaxed": (mols, None), "relaxed/level": (levelled(mols, height), None)}
    t0 = time.time()
    conformers, perms = [], []
    for s, m in zip(smiles, mols):
        c, p = conformer_of(s, m, seed=seed)
        conformers.append(c)
        perms.append(p)
    progress(f"  conformers: {len(conformers)} in {time.time() - t0:.0f} s")

    def posed(start, pose, refine, tag):
        t = time.time()
        ms = [Chem.Mol(m) for m in start]
        if pose in ("flat", "flat-up"):
            for m in ms:
                set_coords(m, rule_pose(m, pose, height=height)[0])
        else:
            # the reference keeps its relaxed pose: one trusted pose is the premise of auto
            ms[0] = Chem.Mol(mols[0])
            plane_align_dataset(ms, names, pose="auto", height=height, ref_index=0, fieldset=fs,
                                refine=refine, restarts=4, seed=seed, ref_as_is=True)
        progress(f"  {tag}: {time.time() - t:.0f} s")
        return ms

    start_relaxed = scrambled(mols, seed)
    frames["relaxed/flat"] = (posed(start_relaxed, "flat", None, "relaxed/flat"), None)
    frames["relaxed/flat-up"] = (posed(start_relaxed, "flat-up", None, "relaxed/flat-up"), None)
    frames["relaxed/auto+plane"] = (posed(start_relaxed, "auto", "plane", "relaxed/auto+plane"), None)
    frames["relaxed/auto+tilt"] = (posed(start_relaxed, "auto", "plane+tilt", "relaxed/auto+tilt"), None)
    # the conformer frames: the reference molecule's conformer is replaced by its relaxed pose in
    # the auto frames (one known pose), and posed by rule like the rest in the flat frame
    frames["conformer/flat"] = (posed(conformers, "flat", None, "conformer/flat"), perms)
    frames["conformer/auto+plane"] = (posed(conformers, "auto", "plane", "conformer/auto+plane"), perms)
    frames["conformer/auto+tilt"] = (posed(conformers, "auto", "plane+tilt", "conformer/auto+tilt"), perms)
    t = time.time()
    mcs = [Chem.Mol(c) for c in conformers]
    log = align_dataset(mcs, 0, use_mcs=True)
    progress(f"  conformer/mcs: {sum('FAILED' in l for l in log)} failures, {time.time() - t:.0f} s")
    frames["conformer/mcs"] = (mcs, perms)
    return frames


def recovery(frames, mols, bound, labels):
    """Pose errors of every rule frame against the relaxed poses."""
    out = {}
    for name, (ms, perms) in frames.items():
        if name in ("relaxed", "relaxed/level", "conformer/mcs"):
            continue
        rows = []
        for i, (p, r) in enumerate(zip(ms, mols)):
            if i == 0 and "auto" in name:
                continue                                  # the reference was given its pose
            perm = None if perms is None else perms[i]
            ip, hz, floor = pose_errors(p, r, perm)
            rows.append((ip, hz, floor, bool(bound[i]), labels[i]))
        arr = np.array([(a, b, c) for a, b, c, _, _ in rows])
        isb = np.array([b for _, _, _, b, _ in rows])
        lab = np.array([l for _, _, _, _, l in rows])
        res = {"n": len(rows), "n_bound": int(isb.sum()),
               "in_plane_median": float(np.median(arr[:, 0])),
               "in_plane_median_bound": float(np.median(arr[isb, 0])) if isb.any() else float("nan"),
               "height_median": float(np.median(arr[:, 1])),
               "height_median_bound": float(np.median(arr[isb, 1])) if isb.any() else float("nan"),
               "floor_median": float(np.median(arr[:, 2])),
               "within_1A_bound": float(np.mean(arr[isb, 0] < 1.0)) if isb.any() else float("nan"),
               "by_label": {}}
        for l in sorted(set(lab)):
            sel = (lab == l) & isb
            if sel.any():
                res["by_label"][l] = {"n": int(sel.sum()), "in_plane_median": float(np.median(arr[sel, 0]))}
        out[name] = dict(res, per_molecule=[[a, b, c] for a, b, c, _, _ in rows])
    # the better of the two faces, molecule by molecule: what the face ambiguity costs
    a, b = out.get("relaxed/flat"), out.get("relaxed/flat-up")
    if a and b:
        best = np.minimum(np.array(a["per_molecule"]), np.array(b["per_molecule"]))
        isb = np.array([bool(x) for x in bound])
        lab = np.array(labels)
        res = {"n": a["n"], "n_bound": a["n_bound"],
               "in_plane_median": float(np.median(best[:, 0])),
               "in_plane_median_bound": float(np.median(best[isb, 0])),
               "height_median": float(np.median(best[:, 1])),
               "height_median_bound": float(np.median(best[isb, 1])),
               "floor_median": float(np.median(best[:, 2])),
               "within_1A_bound": float(np.mean(best[isb, 0] < 1.0)), "by_label": {}}
        for l in sorted(set(lab)):
            sel = (lab == l) & isb
            if sel.any():
                res["by_label"][l] = {"n": int(sel.sum()), "in_plane_median": float(np.median(best[sel, 0]))}
        out["relaxed/flat, better face"] = res
    for v in out.values():
        v.pop("per_molecule", None)
    return out


def baselines(mols, y, bound):
    """Leave-one-out q2 of a straight line on the lowest heavy-atom height of the relaxed pose,
    and on the chemisorbed indicator: how much of E_ads the relaxed frame carries as its outcome."""
    def loo(x):
        X = np.column_stack([np.ones(len(y)), x])
        pred = np.empty(len(y))
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            beta = np.linalg.lstsq(X[k], y[k], rcond=None)[0]
            pred[i] = X[i] @ beta
        return float(r2_score(y, pred))
    hmin = np.array([xyz(m)[heavy(m)][:, 2].min() for m in mols])
    return {"height_loo_q2": loo(hmin), "bound_loo_q2": loo(bound.astype(float)),
            "height_median_bound": float(np.median(hmin[bound])),
            "height_median_unbound": float(np.median(hmin[~bound])) if (~bound).any() else float("nan"),
            "e_ads_bound_mean": float(y[bound].mean()),
            "e_ads_unbound_mean": float(y[~bound].mean()) if (~bound).any() else float("nan")}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def evaluate(mols, names, y, fs, model):
    mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
    train = AtomTable.from_molecules(mf)
    t0 = time.time()
    alphas = ALPHAS if model.uses_fields else [1.0]
    res = nested_cv(train, y, model, alphas, lams=LAMS, outer_folds=OUTER, inner_folds=INNER,
                    repeats=REPEATS, seed=SEED, precompute=model.uses_fields)
    return {"q2": float(res.q2_ex), "q2_sd": float(res.q2_ex_sd), "rmse": float(res.rmse_ex),
            "y_pred": [float(v) for v in res.y_pred], "seconds": round(time.time() - t0, 1)}


def paired(y, a, b, runs=2000, seed=0):
    """(q2 difference a - b, low, high) over bootstrap resamples of the molecules."""
    y = np.asarray(y, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(runs):
        idx = rng.randint(0, len(y), len(y))
        if np.std(y[idx]) <= 0:
            continue
        out.append(r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
    return (r2_score(y, a) - r2_score(y, b), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def models(frames, names, y, fs, subset=None, progress=print):
    out = {}
    sel = np.arange(len(y)) if subset is None else np.flatnonzero(subset)
    names = [names[i] for i in sel]
    y = np.asarray(y, float)[sel]
    for key, label in FRAMES:
        ms = [frames[key][0][i] for i in sel]
        r = evaluate(ms, names, y, fs, CMFModel(fieldset=fs, method="krr", fit_bias=True))
        out["overlap/" + key] = dict(r, label=label, kernel="overlap")
        progress(f"  overlap  {key:22s} q2 {r['q2']:+.3f} +- {r['q2_sd']:.3f}  rmse {r['rmse']:.1f}  {r['seconds']:.0f} s")
    for key in SURFACE_FRAMES:
        ms = [frames[key][0][i] for i in sel]
        m = CMFModel(fieldset=fs, method="krr", fit_bias=True, kernel_source="surface",
                     ac_sigma=0.75, surf_z_min=1.0, surf_z_max=6.0, surf_z_step=1.0)
        r = evaluate(ms, names, y, fs, m)
        out["surface/" + key] = dict(r, label=dict(FRAMES)[key], kernel="surface")
        progress(f"  surface  {key:22s} q2 {r['q2']:+.3f} +- {r['q2_sd']:.3f}  rmse {r['rmse']:.1f}  {r['seconds']:.0f} s")
    # paired differences against the two references
    for ref in ("overlap/relaxed", "surface/relaxed"):
        for k, v in out.items():
            if k == ref:
                continue
            d, lo, hi = paired(y, v["y_pred"], out[ref]["y_pred"])
            v.setdefault("against", {})[ref] = [d, lo, hi]
    return out


# --------------------------------------------------------------------------- #
# The co-field
# --------------------------------------------------------------------------- #

def fitted_model(mols, names, y, fs):
    mf = [build_fields(m, fs, name=n) for m, n in zip(mols, names)]
    train = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", fit_bias=True)
    grid_search(train, y, model, ALPHAS, lams=LAMS, folds=10, seed=0)
    model.fit(train, y, cv_folds=10, cv_seed=0, reuse_kernels=True)
    return model


def cofield_slice(model, f, z, xs, ys):
    centres, coeffs, widths = model.cofield_coefficients(f)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    pts = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    v = evaluate_gaussians(pts, centres, coeffs, widths)
    v *= model.fieldset.specs[f].h * model.scale[f]
    return v.reshape(X.shape)


def cofields(frames, names, y, fs, node, planes=(2.0, 3.5), half_width=8.0, spacing=0.25,
             rule_frame="conformer/flat", progress=print):
    """The co-fields of the overlap model in the rule frame against those in the relaxed frame."""
    xs = np.arange(-half_width, half_width + 1e-9, spacing)
    ys = xs.copy()
    out = {"rule_frame": rule_frame, "planes": list(planes), "fields": {}}
    fitted = {}
    for key in ("relaxed", rule_frame):
        ms, _ = frames[key]
        fitted[key] = fitted_model(ms, names, y, fs)
        progress(f"  co-field model, {key}: alpha {fitted[key].fieldset.specs[0].alpha:g}, {fitted[key].params}")
    slices = {}
    for f, spec in enumerate(fs.specs):
        out["fields"][spec.name] = {}
        for z in planes:
            a = cofield_slice(fitted["relaxed"], f, z, xs, ys)
            b = cofield_slice(fitted[rule_frame], f, z, xs, ys)
            slices[(spec.name, z)] = (a, b)
            r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
            out["fields"][spec.name][f"{z:g}"] = {"r": r, "range_relaxed": [float(a.min()), float(a.max())],
                                                  "range_rule": [float(b.min()), float(b.max())]}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nf = len(fs.specs)
        fig, axes = plt.subplots(2 * len(planes), nf, figsize=(2.6 * nf, 2.6 * 2 * len(planes)), squeeze=False)
        ref_xy = xyz(frames["relaxed"][0][0])[heavy(frames["relaxed"][0][0])][:, :2]
        for f, spec in enumerate(fs.specs):
            for pi, z in enumerate(planes):
                a, b = slices[(spec.name, z)]
                for row, (grid, tag) in enumerate(((a, "relaxed"), (b, "rule"))):
                    # each panel on its own scale: the two models differ in alpha and so in the
                    # size of their coefficients, and the comparison is of shape, not magnitude
                    vmax = max(np.abs(grid).max(), 1e-12)
                    ax = axes[2 * pi + row, f]
                    ax.imshow(grid, origin="lower", extent=[xs[0], xs[-1], ys[0], ys[-1]], cmap="RdBu",
                              vmin=-vmax, vmax=vmax)
                    ax.scatter(ref_xy[:, 0], ref_xy[:, 1], s=4, c="k")
                    ax.set_title(f"{spec.name}, z = {z:g} A, {tag}", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"{node}: co-fields of the E_ads model, relaxed frame against the rule frame "
                     f"({rule_frame}); blue = stronger adsorption; each panel on its own scale", fontsize=9)
        fig.tight_layout()
        path = os.path.join(RESULTS, f"cofields_{node}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        out["figure"] = relative_path(path, ROOT)
    except Exception as exc:  # noqa: BLE001
        out["figure"] = f"not drawn: {exc}"
    return out


# --------------------------------------------------------------------------- #
# Driver and report
# --------------------------------------------------------------------------- #

def run_node(node, results=RESULTS, force=False):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, f"{node}.json")
    if os.path.exists(out_path) and not force:
        print(f"{node}: {out_path} exists, use --force to recompute")
        return
    mols, names, y, bound, labels, smiles = load_node(node)
    print(f"{node}: {len(mols)} molecules with a reliable E_ads, {int(bound.sum())} chemisorbed, "
          f"reference {names[0]}")
    fs = FieldSet.create()
    print("frames")
    frames = make_frames(mols, names, smiles, fs, height=HEIGHT)
    base = baselines(mols, y, bound)
    print(f"baselines: LOO q2 on the lowest height alone {base['height_loo_q2']:.3f}, on the chemisorbed "
          f"indicator {base['bound_loo_q2']:.3f}; lowest heavy atom median {base['height_median_bound']:.2f} A "
          f"chemisorbed, {base['height_median_unbound']:.2f} A detached")
    print("pose recovery")
    rec = recovery(frames, mols, bound, labels)
    for k, v in rec.items():
        print(f"  {k:22s} in-plane rmsd median {v['in_plane_median']:.2f} A (chemisorbed {v['in_plane_median_bound']:.2f}, "
              f"{v['within_1A_bound']:.0%} within 1 A), heights {v['height_median_bound']:.2f} A, "
              f"rigid floor {v['floor_median']:.2f} A")
    print("models, all molecules")
    mod = models(frames, names, y, fs)
    print("models, chemisorbed molecules")
    mod_bound = models(frames, names, y, fs, subset=bound)
    print("co-fields")
    cof = cofields(frames, names, y, fs, node)
    for fname, planes in cof["fields"].items():
        print("  " + fname + ": " + ", ".join(f"z {z}: r = {v['r']:.2f}" for z, v in planes.items()))
    with open(out_path, "w") as fh:
        json.dump({"node": node, "n": len(mols), "n_bound": int(bound.sum()), "reference": names[0],
                   "protocol": {"outer": OUTER, "inner": INNER, "repeats": REPEATS, "seed": SEED,
                                "alphas": ALPHAS, "lams": LAMS},
                   "height": HEIGHT, "baselines": base,
                   "recovery": rec, "models": mod, "models_bound": mod_bound, "cofields": cof}, fh, indent=1)
    print("->", out_path)


def _fmt_against(v, ref):
    if "against" not in v or ref not in v["against"]:
        return "-"
    d, lo, hi = v["against"][ref]
    return f"{d:+.2f} [{lo:+.2f}, {hi:+.2f}]"


def report(results=RESULTS, out=None):
    nodes = []
    for fn in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        if fn.endswith(".json"):
            nodes.append(json.load(open(os.path.join(results, fn))))
    if not nodes:
        raise SystemExit(f"no results in {results}")
    lines = ["# Posed by rule against relaxed on a slab: the planar aligner on the Fe(110) sets", "",
             "The relaxed poses are UMA's (`examples/corrosion_fe110`), the normal along z and the top",
             "Fe layer at z = 0. The rule poses are `cmfqsar/planealign.py`: flat with the flattest",
             "face down, or auto (flat on either face, or upright on the most basic nitrogen,",
             "whichever is most similar in field to the one reference pose), refined on that",
             "similarity in the plane only or also in tilt and height. Pose errors are RMSDs over",
             "heavy atoms after the best rotation about the normal and translation in the plane.",
             "Models: KRR on the five fields, 5 outer folds, 10 inner, 10 repeats, one seed, alpha",
             "and lambda by the inner loop; differences with their 95% paired bootstrap interval over",
             "molecules.", ""]
    for nd in nodes:
        lines += [f"## {nd['node']}: {nd['n']} molecules, {nd['n_bound']} chemisorbed, reference {nd['reference']}", "",
                  "### Pose recovery (A)", "",
                  "| frame | in-plane RMSD, all | chemisorbed | within 1 A | height RMSD, chemisorbed | rigid floor |",
                  "|---|---:|---:|---:|---:|---:|"]
        for k, v in nd["recovery"].items():
            lines.append(f"| {k} | {v['in_plane_median']:.2f} | {v['in_plane_median_bound']:.2f} | "
                         f"{v['within_1A_bound']:.0%} | {v['height_median_bound']:.2f} | {v['floor_median']:.2f} |")
        bl = {k: v["by_label"] for k, v in nd["recovery"].items()}
        labs = sorted({l for d in bl.values() for l in d})
        if labs:
            lines += ["", "Chemisorbed molecules by the pose UMA kept, in-plane RMSD median (n):", "",
                      "| frame | " + " | ".join(labs) + " |", "|---|" + "---:|" * len(labs)]
            for k, d in bl.items():
                lines.append(f"| {k} | " + " | ".join(f"{d[l]['in_plane_median']:.2f} ({d[l]['n']})" if l in d else "-"
                                                      for l in labs) + " |")
        b = nd["baselines"]
        lines += ["", "### What the relaxed frame carries", "",
                  f"The lowest heavy atom of a relaxed pose sits at {b['height_median_bound']:.2f} A (median) when the "
                  f"molecule is chemisorbed and at {b['height_median_unbound']:.2f} A when it detached; E_ads averages "
                  f"{b['e_ads_bound_mean']:.0f} kJ/mol for the first and {b['e_ads_unbound_mean']:.0f} for the second. "
                  f"A straight line on that one height predicts E_ads with a leave-one-out q2 of "
                  f"**{b['height_loo_q2']:.2f}**, the chemisorbed indicator alone with {b['bound_loo_q2']:.2f}. That is "
                  f"the outcome of the relaxation, not a property of the molecule, and a rule cannot know it; the rows "
                  f"marked *relaxed, lowered* keep the relaxed tilt and face but put every molecule at the common "
                  f"height of {nd['height']:.1f} A, which is what a rule can do.", ""]
        for which, title in (("models", "all molecules"), ("models_bound", "chemisorbed molecules only")):
            lines += ["", f"### Models of E_ads (kJ/mol), {title}", "",
                      "| kernel | frame | q2 | RMSE | against overlap, relaxed | against surface, relaxed |",
                      "|---|---|---:|---:|---|---|"]
            for k, v in nd[which].items():
                lines.append(f"| {v['kernel']} | {v['label']} | {v['q2']:+.3f} +- {v['q2_sd']:.3f} | {v['rmse']:.1f} | "
                             f"{_fmt_against(v, 'overlap/relaxed')} | {_fmt_against(v, 'surface/relaxed')} |")
        cof = nd["cofields"]
        lines += ["", f"### The co-field in the rule frame ({cof['rule_frame']}) against the relaxed frame", "",
                  "Correlation over the plane of the two co-fields, per field and height:", "",
                  "| field | " + " | ".join(f"z = {z:g} A" for z in cof["planes"]) + " |",
                  "|---|" + "---:|" * len(cof["planes"])]
        for fname, planes in cof["fields"].items():
            lines.append(f"| {fname} | " + " | ".join(f"{planes[f'{z:g}']['r']:+.2f}" for z in cof["planes"]) + " |")
        if "figure" in cof:
            lines += ["", f"Figure: `{cof['figure']}`"]
        lines.append("")
    text = "\n".join(lines) + "\n"
    path = out or os.path.join(results, "table.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print("->", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", choices=list(NODES) + ["both"], default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.node:
        for nd in (list(NODES) if args.node == "both" else [args.node]):
            run_node(nd, force=args.force)
    if args.report or not args.node:
        report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
