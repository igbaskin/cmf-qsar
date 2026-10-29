#!/usr/bin/env python
"""
Does a ligand's trajectory in its pocket model a series better than its crystal pose?

That is plan item D8, and it is the question the fourth-dimension study of this repository could
not ask. That study built its ensembles by generating conformations of a molecule on its own and
found they did not help; what it never had was a pose that is known to be right. A free-energy
set gives one: every ligand is co-crystallised, so the curated frame is the crystal frame, and a
short simulation of the ligand inside that site produces an ensemble around a pose that is not a
guess.

Five models, on the same molecules, the same alpha and lambda grids and the same 10-fold
cross-validation:

  crystal              the crystal pose alone, one conformation per ligand. The reference of
                       rule 1, and the ceiling the earlier study never had.
  conformers           an ensemble of generated conformations, superimposed on the crystal pose.
                       The control that separates "molecular dynamics helps" from "any ensemble
                       helps": it is what the fourth-dimension study did.
  md_uniform           the trajectory ensemble, every cluster weighing the same. This is the
                       comparison the item names.
  md_population        the trajectory ensemble, each cluster weighted by how many frames fell in
                       it, which is what `cmfqsar/trajectory.py` writes. The item asks for the
                       uniform one; this says whether the weighting matters.
  md_mil               the multi-instance fit over the trajectory clusters, which lets the model
                       choose which frames count instead of averaging them.

Every difference is reported against `crystal` with the bootstrap interval of rule 4, so a gain
below the interval is not a gain.

The trajectories are made by `examples/make_md_trajectories.py`, which needs OpenMM and the
OpenFF stack; this script needs only the ensemble file that script writes, so the table can be
rebuilt on a machine with no simulation software.

Usage
-----
    python benchmarks/run_md.py
    python benchmarks/run_md.py --sets mcl1 p38 hif2a
    python benchmarks/run_md.py --report
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

import kernel_cache as KC  # noqa: E402

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, cv_predict, r2_score, rmse  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "md")
MD_DATA = os.path.join(ROOT, "examples", "md")
DOCKED_DATA = os.path.join(ROOT, "examples", "docked")
CRYSTAL_DATA = os.path.join(ROOT, "examples", "openff")
SETS = ("mcl1", "p38", "hif2a")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
BETAS = (0.0, 1.0, 3.0, 10.0)
FOLDS = 10
SEED = 0
RUNS = 2000
#: conformations generated for the control ensemble, to match the trajectory ensemble's size
N_CONFORMERS = 20
#: the attenuation and lambda at which the normalisation check is run, one setting for all sets
NORM_ALPHA = 0.3
NORM_LAM = 0.1


def _say(text):
    print(text, flush=True)


# ----------------------------------------------------------------------------------------------
# the data
# ----------------------------------------------------------------------------------------------

def crystal_set(name, activity="pAffinity"):
    """The ligands as the collection gives them: one crystal pose each, with their activities."""
    path = os.path.join(CRYSTAL_DATA, name + ".sdf")
    ds = load_dataset(path, activity_prop=activity)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    return ([m for m, g in zip(ds.mols, ok) if g],
            [n for n, g in zip(ds.names, ok) if g], y[ok])


def docked_set(name, activity="pAffinity", method=None):
    """The poses found while each ligand was held out, one pose per ligand.

    `examples/make_docked_poses.py` writes them: no ligand is posed with the help of its own
    crystal pose or its own activity.
    """
    method = method or DOCK_METHOD
    path = os.path.join(DOCKED_DATA, f"{name}_{method}.sdf")
    if not os.path.isfile(path):
        return None
    ds = load_dataset(path, activity_prop=activity)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    return ([m for m, g in zip(ds.mols, ok) if g],
            [n for n, g in zip(ds.names, ok) if g], y[ok])


def md_set(name, activity="pAffinity", suffix="_md"):
    """The trajectory ensembles, one molecule per ligand carrying its representative frames."""
    path = os.path.join(MD_DATA, name + suffix + ".sdf")
    if not os.path.isfile(path):
        return None
    ds = load_dataset(path, activity_prop=activity, ensemble_by="name")
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    return ([m for m, g in zip(ds.mols, ok) if g],
            [n for n, g in zip(ds.names, ok) if g], y[ok])


def conformer_control(mols, names, n=N_CONFORMERS, seed=SEED):
    """Generated conformations superimposed on each molecule's own crystal pose.

    What the fourth-dimension study did, so that a gain from the trajectories can be told from a
    gain that any ensemble would give.
    """
    from cmfqsar.conformers import generate_conformers
    out = []
    for i, m in enumerate(mols):
        out.append(generate_conformers(m, n=n, seed=seed + i, align_to="self", keep_input=True))
    return out


def matched(crystal, md):
    """The ligands both sets have, in one order, so every model sees the same molecules."""
    c_mols, c_names, c_y = crystal
    m_mols, m_names, m_y = md
    index = {n: k for k, n in enumerate(m_names)}
    keep = [i for i, n in enumerate(c_names) if n in index]
    cm = [c_mols[i] for i in keep]
    cn = [c_names[i] for i in keep]
    cy = c_y[keep]
    mm = [m_mols[index[n]] for n in cn]
    return (cm, cn, cy), (mm, cn, cy)


# ----------------------------------------------------------------------------------------------
# the models
# ----------------------------------------------------------------------------------------------

def field_set(alpha, mode="single", weighting="uniform"):
    return FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                           param_scheme="rdkit", conformers=mode,
                           conformer_weighting=weighting)


def kernel_for(mols, names, alpha, mode="single", weighting="uniform"):
    fs = field_set(alpha, mode, weighting)
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    return KC.joint_training_kernel(model, table)


def best_over_grid(mols, names, y, mode="single", weighting="uniform",
                   alphas=ALPHAS, lams=LAMS, folds=FOLDS, seed=SEED):
    """Best (q2, RMSEcv, alpha, lambda, predictions) over the grid, scored by its own CV."""
    best = None
    for alpha in alphas:
        K = kernel_for(mols, names, alpha, mode=mode, weighting=weighting)
        for lam in lams:
            pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=folds, seed=seed)
            q2 = r2_score(y, pred)
            if best is None or q2 > best[0]:
                best = (float(q2), float(rmse(y, pred)), float(alpha), float(lam),
                        np.asarray(pred, float))
    return best


def best_mil(mols, names, y, alphas=ALPHAS, lams=LAMS, betas=BETAS, folds=FOLDS, seed=SEED):
    """The multi-instance fit, its attenuation chosen on the same grid as the others."""
    from cmfqsar import mil

    best = None
    for alpha in alphas:
        fs = field_set(alpha, "single")
        flat, bags = mil.flatten_conformers(mols, fs, names)
        table = AtomTable.from_molecules(flat)
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        K_conf = KC.joint_training_kernel(model, table)
        got = mil.search(K_conf, bags, y, betas=betas, lams=lams, folds=folds, seed=seed)
        if best is None or got.best_q2 > best[0]:
            best = (float(got.best_q2), float(got.best_rmse), float(alpha),
                    float(got.best_lam), np.asarray(got.y_cv, float), float(got.best_beta))
    return best


def bootstrap_difference(truth, a, b, runs=RUNS, seed=SEED):
    """(difference in q2, low, high) for prediction `a` against `b`, resampling the molecules."""
    truth = np.asarray(truth, float)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(truth) & np.isfinite(a) & np.isfinite(b)
    truth, a, b = truth[ok], a[ok], b[ok]
    if len(truth) < 4:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(runs)):
        idx = rng.randint(0, len(truth), len(truth))
        if np.std(truth[idx]) <= 0:
            continue
        out.append(r2_score(truth[idx], a[idx]) - r2_score(truth[idx], b[idx]))
    if not out:
        return float("nan"), float("nan"), float("nan")
    return (float(r2_score(truth, a) - r2_score(truth, b)),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


# ----------------------------------------------------------------------------------------------
# one set
# ----------------------------------------------------------------------------------------------

def run_set(name, results=RESULTS, force=False):
    out_path = os.path.join(results, f"{name}.json")
    if os.path.exists(out_path) and not force:
        with open(out_path) as fh:
            return json.load(fh)
    md = md_set(name)
    if md is None:
        _say(f"  {name}: no trajectory ensemble in examples/md; run "
             f"examples/make_md_trajectories.py first")
        return None
    crystal, md = matched(crystal_set(name), md)
    c_mols, names, y = crystal
    m_mols, _n, _y = md
    frames = [m.GetNumConformers() for m in m_mols]
    _say(f"  {name}: {len(names)} ligands, {np.mean(frames):.1f} trajectory conformations each")

    t0 = time.time()
    rows = {}
    q2, rms, alpha, lam, pred = best_over_grid(c_mols, names, y, mode="single")
    rows["crystal"] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam, pred=pred.tolist())
    _say(f"    crystal        q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} lambda {lam:g}")

    conf_mols = conformer_control(c_mols, names)
    q2, rms, alpha, lam, pred = best_over_grid(conf_mols, names, y, mode="ensemble",
                                               weighting="uniform")
    rows["conformers"] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam, pred=pred.tolist())
    _say(f"    conformers     q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} lambda {lam:g}")

    for tag, weighting in (("md_uniform", "uniform"), ("md_population", "given")):
        q2, rms, alpha, lam, pred = best_over_grid(m_mols, names, y, mode="ensemble",
                                                   weighting=weighting)
        rows[tag] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam, pred=pred.tolist())
        _say(f"    {tag:14s} q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} lambda {lam:g}")

    q2, rms, alpha, lam, pred, beta = best_mil(m_mols, names, y)
    rows["md_mil"] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam, beta=beta, pred=pred.tolist())
    _say(f"    md_mil         q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} lambda {lam:g} "
         f"beta {beta:g}")

    own = md_set(name, suffix="_md_ownframe")
    if own is not None:
        # the same frames, each trajectory fitted to its own first frame instead of to the one
        # structure they were all built from: what a field model loses without a common frame
        _c, own = matched(crystal_set(name), own)
        o_mols, _on, _oy = own
        if len(o_mols) == len(names):
            q2, rms, alpha, lam, pred = best_over_grid(o_mols, names, y, mode="ensemble",
                                                       weighting="uniform")
            rows["md_ownframe"] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam,
                                       pred=pred.tolist())
            _say(f"    md_ownframe    q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} "
                 f"lambda {lam:g}")

    base = np.asarray(rows["crystal"]["pred"], float)
    for tag in rows:
        d, lo, hi = bootstrap_difference(y, np.asarray(rows[tag]["pred"], float), base)
        rows[tag].update(delta=d, lo=lo, hi=hi)

    if "md_ownframe" in rows and "md_uniform" in rows:
        # the control is also read against the common-frame ensemble, because the alignment is
        # the only thing that differs between those two: same frames, same weighting
        d, lo, hi = bootstrap_difference(y, np.asarray(rows["md_ownframe"]["pred"], float),
                                         np.asarray(rows["md_uniform"]["pred"], float))
        rows["md_ownframe"].update(vs_uniform=dict(delta=d, lo=lo, hi=hi))

    rec = {"set": name, "n": int(len(y)), "frames_mean": float(np.mean(frames)),
           "frames_min": int(np.min(frames)), "frames_max": int(np.max(frames)),
           "seconds": float(time.time() - t0),
           # the measured activities, in the order of the predictions, so that a reading can be
           # combined over the sets afterwards without fitting anything again
           "activity": [float(v) for v in y], "models": rows}
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ----------------------------------------------------------------------------------------------
# poses that are not known
# ----------------------------------------------------------------------------------------------

def _conformer_rmsds(mol, ref):
    """How far every conformation of `mol` is from the single pose `ref`, no superposition."""
    from cmfqsar.poses import pose_rmsd
    out = []
    for conf in mol.GetConformers():
        one = Chem.Mol(mol)
        one.RemoveAllConformers()
        one.AddConformer(Chem.Conformer(conf), assignId=True)
        out.append(float(pose_rmsd(one, ref)))
    return [v for v in out if v == v]


def stage_docked(name, results=RESULTS, method=None, force=False):
    """The same question where the pose is not known, which is where an ensemble should help.

    Five models of the same ligands on the same folds and grids: the crystal pose as the ceiling,
    the pose that was found while the ligand was held out, and three readings of a simulation
    started from that found pose. Every difference is against the found pose, because that is
    what a modeller would have without a crystal structure; the difference from the crystal pose
    is reported beside it, to say how much of the ceiling is lost.
    """
    method = method or DOCK_METHOD
    out_path = os.path.join(results, f"{name}_docked.json")
    if os.path.exists(out_path) and not force:
        with open(out_path) as fh:
            return json.load(fh)
    docked = docked_set(name, method=method)
    ens = md_set(name, suffix="_docked_md")
    if docked is None or ens is None:
        _say(f"  {name}: no poses found without the crystal pose, or no simulation from them; "
             f"run examples/make_docked_poses.py and then "
             f"examples/make_md_trajectories.py --ligands")
        return None

    cm, cn, cy = crystal_set(name)
    di = dict(zip(docked[1], docked[0]))
    ei = dict(zip(ens[1], ens[0]))
    keep = [i for i, n in enumerate(cn) if n in di and n in ei]
    names = [cn[i] for i in keep]
    y = cy[keep]
    c_mols = [cm[i] for i in keep]
    d_mols = [di[n] for n in names]
    e_mols = [ei[n] for n in names]
    frames = [m.GetNumConformers() for m in e_mols]
    _say(f"  {name}: {len(names)} ligands posed without their crystal poses, "
         f"{np.mean(frames):.1f} trajectory conformations each")

    t0 = time.time()
    rows = {}
    for tag, mols in (("crystal", c_mols), ("docked", d_mols)):
        q2, rms, alpha, lam, pred = best_over_grid(mols, names, y, mode="single")
        rows[tag] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam, pred=pred.tolist())
        _say(f"    {tag:18s} q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} lambda {lam:g}")
    for tag, weighting in (("docked_uniform", "uniform"), ("docked_population", "given")):
        q2, rms, alpha, lam, pred = best_over_grid(e_mols, names, y, mode="ensemble",
                                                   weighting=weighting)
        rows[tag] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam, pred=pred.tolist())
        _say(f"    {tag:18s} q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} lambda {lam:g}")
    q2, rms, alpha, lam, pred, beta = best_mil(e_mols, names, y)
    rows["docked_mil"] = dict(q2=q2, rmse=rms, alpha=alpha, lam=lam, beta=beta,
                              pred=pred.tolist())
    _say(f"    docked_mil         q2 {q2:+.3f}  RMSEcv {rms:.3f}  alpha {alpha:g} "
         f"lambda {lam:g} beta {beta:g}")

    # the found pose is the reference, because that is what a modeller has without a crystal
    # structure; the crystal pose is the ceiling and is reported as a second difference
    base = np.asarray(rows["docked"]["pred"], float)
    ceiling = np.asarray(rows["crystal"]["pred"], float)
    for tag in rows:
        pred = np.asarray(rows[tag]["pred"], float)
        d, lo, hi = bootstrap_difference(y, pred, base)
        rows[tag].update(delta=d, lo=lo, hi=hi)
        d, lo, hi = bootstrap_difference(y, pred, ceiling)
        rows[tag].update(vs_crystal=dict(delta=d, lo=lo, hi=hi))

    from cmfqsar.poses import pose_rmsd
    before = [float(pose_rmsd(d, c)) for d, c in zip(d_mols, c_mols)]
    nearest, middle = [], []
    for e, c in zip(e_mols, c_mols):
        rs = _conformer_rmsds(e, c)
        if rs:
            nearest.append(float(np.min(rs)))
            middle.append(float(np.median(rs)))
    rec = {"set": name, "method": method, "n": int(len(y)),
           "frames_mean": float(np.mean(frames)), "frames_min": int(np.min(frames)),
           "frames_max": int(np.max(frames)), "seconds": float(time.time() - t0),
           "activity": [float(v) for v in y],
           "poses": {"median_rmsd": float(np.median(before)),
                     "max_rmsd": float(np.max(before)),
                     "within_1A": float(np.mean(np.asarray(before) <= 1.0)),
                     "within_2A": float(np.mean(np.asarray(before) <= 2.0))},
           "repair": {"nearest_median": float(np.median(nearest)) if nearest else None,
                      "conformation_median": float(np.median(middle)) if middle else None},
           "models": rows}
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def load_docked(results=RESULTS):
    """Every result of the pose-not-known stage, biggest set first."""
    out = []
    for path in sorted(glob.glob(os.path.join(results, "*_docked.json"))):
        with open(path) as fh:
            out.append(json.load(fh))
    return sorted(out, key=lambda r: -int(r.get("n") or 0))


def docked_section(rows):
    """What an ensemble is worth when the pose it covers is not known."""
    if not rows:
        return []
    out = ["## Poses that are not known", ""]
    for rec in rows:
        m = rec["models"]
        p, rp = rec.get("poses") or {}, rec.get("repair") or {}
        out += [f"Every number above rests on crystal poses, and the item itself says that an "
                f"ensemble should help most where the pose is *not* known. On **{rec['set']}** "
                f"it is asked there too. Every ligand was posed while it was held out: its pose "
                f"was destroyed, a random placement within 2 A of the site, and then the "
                f"receptor's own fields put it back where the pocket prefers it -- no model, no "
                f"activity of its own and no crystal frame of any kind. Those poses sit a "
                f"median {p.get('median_rmsd', float('nan')):.2f} A from the crystal poses "
                f"({p.get('within_1A', 0.0):.0%} within 1 A, "
                f"{p.get('within_2A', 0.0):.0%} within 2 A, at most "
                f"{p.get('max_rmsd', float('nan')):.2f} A). Each ligand was then simulated from "
                f"its found pose, for the same length as everywhere above.", "",
                f"{rec['n']} ligands, {rec['frames_mean']:.1f} conformations each.", "",
                "| model | q2 | change against the pose that was found | "
                "change against the crystal pose |",
                "| --- | ---: | --- | --- |"]
        for tag in DOCK_TAGS:
            if tag not in m:
                continue
            r = m[tag]
            vc = r.get("vs_crystal") or {}
            first = ("the reference" if tag == "docked" else
                     f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
            second = ("the ceiling" if tag == "crystal" else
                      f"{vc['delta']:+.3f} [{vc['lo']:+.3f}, {vc['hi']:+.3f}]"
                      if vc else "-")
            out.append(f"| {DOCK_LABEL[tag]} | {r['q2']:+.3f} | {first} | {second} |")
        out.append("")

        if rp.get("nearest_median") is not None:
            moved = ("nearer to" if rp["nearest_median"] < p.get("median_rmsd", 0.0)
                     else "no nearer to")
            # the nearest of several conformations is a minimum over them, so it is lower
            # than a single pose even when nothing moved nearer: say so
            drifted = (rp["conformation_median"] > p.get("median_rmsd", 0.0))
            caution = (" The middle conformation is farther from the crystal pose than the "
                       "starting pose was, so the ensemble as a whole did not move towards it."
                       if drifted else "")
            out += [f"**Did the simulation repair the pose?** The nearest kept conformation of "
                    f"each ligand is a median {rp['nearest_median']:.2f} A from its crystal "
                    f"pose and the middle conformation {rp['conformation_median']:.2f} A, "
                    f"against {p.get('median_rmsd', float('nan')):.2f} A for the pose the "
                    f"simulation started from. So two nanoseconds in the site bring the ligand "
                    f"{moved} the pose it should have had. The nearest conformation is "
                    f"chosen after the fact, out of {rec['frames_mean']:.1f} of "
                    f"them, so part of that is a smallest of several and not repair."
                    + caution, ""]

        fixed = m.get("docked_uniform")
        if fixed:
            if fixed["lo"] > 0:
                says = (f"**Where the pose is not known, the ensemble is worth something.** The "
                        f"reading fixed in advance gives {fixed['delta']:+.3f} "
                        f"[{fixed['lo']:+.3f}, {fixed['hi']:+.3f}] against the single pose it "
                        f"started from, which clears the interval. That is the opposite of what "
                        f"the same reading does on crystal poses, and it is what the item "
                        f"expected: an ensemble covers a pose that is wrong.")
            elif fixed["hi"] < 0:
                says = (f"**The ensemble is worse than the single pose it started from**, "
                        f"{fixed['delta']:+.3f} [{fixed['lo']:+.3f}, {fixed['hi']:+.3f}] for "
                        f"the reading fixed in advance, beyond the interval.")
            else:
                says = (f"**The ensemble gains nothing here either.** The reading fixed in "
                        f"advance gives {fixed['delta']:+.3f} [{fixed['lo']:+.3f}, "
                        f"{fixed['hi']:+.3f}] against the single pose it started from, which "
                        f"does not clear the interval. So the place where an ensemble should "
                        f"help most does not show it helping, on this set and at this length.")
            best = max((t for t in DOCK_TAGS if t.startswith("docked_") and t in m),
                       key=lambda t: m[t]["delta"], default=None)
            if best and best != "docked_uniform":
                b = m[best]
                says += (f" The largest of the three readings is {DOCK_LABEL[best]}, "
                         f"{b['delta']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}], and taking the "
                         f"largest of three is what the section above measured the cost of.")
            out += [says, ""]
        cr = (m.get("docked") or {}).get("vs_crystal") or {}
        if cr:
            out += [f"What the whole exercise costs: posing the ligands instead of reading their "
                    f"crystal poses gives {cr['delta']:+.3f} [{cr['lo']:+.3f}, "
                    f"{cr['hi']:+.3f}] of q2. That is the size of the thing an ensemble would "
                    f"have to repair.", ""]
    return out


# ----------------------------------------------------------------------------------------------
# where the kernel is normalised
# ----------------------------------------------------------------------------------------------

def stage_normalisation(name, results=RESULTS, alpha=NORM_ALPHA, lam=NORM_LAM, force=False):
    """Two paths that should be the same model, measured against each other.

    `cmfqsar/mil.py` says beta = 0 is the plain ensemble with every conformation weighing the
    same, so the multi-instance fit at beta = 0 and the uniform ensemble are one model. They do
    not give one q2. The difference is where the kernel is normalised: the ensemble path
    normalises the kernel of the whole ensemble, the multi-instance path normalises every
    conformation and then averages. Switching the normalisation off is what shows that, so the
    four numbers are measured at one attenuation and one lambda, with no grid and no choice.
    """
    from cmfqsar import mil

    out_path = os.path.join(results, f"{name}_norm.json")
    if os.path.exists(out_path) and not force:
        with open(out_path) as fh:
            return json.load(fh)
    md = md_set(name)
    if md is None:
        return None
    _c, md = matched(crystal_set(name), md)
    mols, names, y = md

    rows = {}
    for normalize in (True, False):
        fs = field_set(alpha, "ensemble", "uniform")
        table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
        model = CMFModel(fieldset=fs, method="krr", normalize=normalize)
        K = KC.joint_training_kernel(model, table)
        pred = cv_predict(K, y, "krr", {"lam": float(lam)}, folds=FOLDS, seed=SEED)
        rows[f"ensemble_{'normalised' if normalize else 'plain'}"] = float(r2_score(y, pred))

        fs = field_set(alpha, "single")
        flat, bags = mil.flatten_conformers(mols, fs, names)
        table = AtomTable.from_molecules(flat)
        model = CMFModel(fieldset=fs, method="krr", normalize=normalize)
        K_conf = KC.joint_training_kernel(model, table)
        ycv = mil.cv_predict(K_conf, bags, y, beta=0.0, lam=float(lam), folds=FOLDS, seed=SEED)
        rows[f"mil_{'normalised' if normalize else 'plain'}"] = float(r2_score(y, ycv))

    rec = {"set": name, "n": int(len(y)), "alpha": float(alpha), "lam": float(lam),
           "q2": rows}
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=1)
    _say(f"  {name}: ensemble {rows['ensemble_normalised']:+.3f} against multi-instance at "
         f"beta = 0 {rows['mil_normalised']:+.3f}; with the normalisation off "
         f"{rows['ensemble_plain']:+.3f} against {rows['mil_plain']:+.3f}")
    return rec


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

TAGS = ("conformers", "md_uniform", "md_population", "md_mil", "md_ownframe")
LABEL = {"crystal": "the crystal pose", "conformers": "generated conformations",
         "md_uniform": "trajectory, clusters equal", "md_population": "trajectory, by population",
         "md_mil": "trajectory, multi-instance",
         "md_ownframe": "trajectory, each in its own frame"}


#: file endings in the results directory that are not a set's own result
OTHER_RESULTS = ("_norm.json", "_docked.json")

#: the stage for poses that are not known: the posing method, and the models it compares
DOCK_METHOD = "pocket"
DOCK_TAGS = ("crystal", "docked", "docked_uniform", "docked_population", "docked_mil")
DOCK_LABEL = {"crystal": "the crystal pose", "docked": "the pose that was found",
              "docked_uniform": "trajectory from it, clusters equal",
              "docked_population": "trajectory from it, by population",
              "docked_mil": "trajectory from it, multi-instance"}

#: the one reading of the trajectory that was fixed before the sets were compared with each
#: other: the ensemble with every conformation weighing the same. It contains no choice -- there
#: is no weighting to pick and no beta to search -- which is what makes it reportable when the
#: best of several readings is not.
FIXED_TAG = "md_uniform"

#: readings of the trajectory that are controls rather than candidate models. The own-frame
#: reading is built to be wrong on purpose -- it drops the common frame -- so it is left out of
#: the comparison between the sets and out of the count of readings that were tried.
CONTROLS = ("md_ownframe",)


def load_all(results=RESULTS):
    out = []
    for p in sorted(glob.glob(os.path.join(results, "*.json"))):
        if p.endswith(OTHER_RESULTS):
            continue
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def load_norms(results=RESULTS):
    out = []
    for p in sorted(glob.glob(os.path.join(results, "*_norm.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
    return out


def norm_section(norms):
    """What the normalisation check settles, when it has been run.

    It is reported because a result depends on it: on a set where the only reading of the
    trajectory that clears the interval is the multi-instance fit at beta = 0, that reading is
    the uniform ensemble with the kernel normalised in another place.
    """
    if not norms:
        return []
    out = ["## Where the kernel is normalised", "",
           f"The multi-instance fit at beta = 0 is the uniform ensemble: every conformation "
           f"weighs the same. The two are the same model and they do not give the same q2. Each "
           f"row is one attenuation ({norms[0]['alpha']:g}) and one lambda "
           f"({norms[0]['lam']:g}), with no grid and nothing chosen, so these numbers are not "
           f"comparable with the table above, where every model is taken at its own best "
           f"attenuation.", "",
           "| set | ligands | ensemble | multi-instance, beta = 0 | ensemble, no normalisation "
           "| multi-instance, no normalisation |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in norms:
        q = r["q2"]
        out.append(f"| {r['set']} | {r['n']} | {q['ensemble_normalised']:+.3f} | "
                   f"{q['mil_normalised']:+.3f} | {q['ensemble_plain']:+.3f} | "
                   f"{q['mil_plain']:+.3f} |")
    out.append("")

    gaps = [abs(r["q2"]["mil_normalised"] - r["q2"]["ensemble_normalised"]) for r in norms]
    plain = [abs(r["q2"]["mil_plain"] - r["q2"]["ensemble_plain"]) for r in norms]
    narrower = sum(1 for a, b in zip(gaps, plain) if b < a)
    out += [f"**The two paths differ by {min(gaps):.3f} to {max(gaps):.3f} of q2, and the "
            f"difference narrows to {min(plain):.3f} to {max(plain):.3f} when the normalisation "
            f"is switched off** -- on {narrower} of the {len(norms)} sets. So most of what "
            f"separates two paths through one model is where the kernel is normalised: the "
            f"ensemble path normalises the kernel of the whole ensemble, the multi-instance "
            f"path normalises every conformation and then averages. Up to {max(plain):.3f} is "
            f"left over, so the normalisation is not all of it.", "",
            f"Both places are defensible, which is why this matters for reading the table: a "
            f"gain of this size can come from the choice alone. Where the only reading of a set "
            f"that clears the interval is the multi-instance fit at beta = 0, that reading is "
            f"the uniform ensemble with the kernel normalised in the other place, and the "
            f"uniform ensemble itself does not clear it.", ""]
    return out


def report(results=RESULTS, path=None):
    rows = load_all(results)
    out = ["# Does a trajectory beat the crystal pose?", "",
           "Plan item D8. Five models of the same ligands on the same folds, with the same "
           "alpha and lambda grids: the crystal pose alone, an ensemble of generated "
           "conformations (what the fourth-dimension study did), and three readings of a short "
           "simulation of each ligand in its own crystal site. Every difference is against the "
           "crystal pose, with the bootstrap interval of rule 4.", ""]
    if not rows:
        out += ["No result yet. `examples/make_md_trajectories.py` produces the trajectories, "
                "which needs OpenMM and the OpenFF stack; this table is rebuilt from the "
                "ensemble file it writes.", ""]
        text = "\n".join(out) + "\n"
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", newline="\n") as fh:
                fh.write(text)
        return text

    for rec in rows:
        m = rec["models"]
        out += [f"## {rec['set']}", "",
                f"{rec['n']} ligands, {rec['frames_mean']:.1f} trajectory conformations each "
                f"(from {rec['frames_min']} to {rec['frames_max']}).", "",
                "| model | q2 | RMSEcv | attenuation | change against the crystal pose |",
                "| --- | ---: | ---: | ---: | --- |"]
        for tag in ("crystal",) + TAGS:
            if tag not in m:
                continue
            r = m[tag]
            change = ("the reference" if tag == "crystal" else
                      f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
            out.append(f"| {LABEL[tag]} | {r['q2']:+.3f} | {r['rmse']:.3f} | "
                       f"{r['alpha']:g} | {change} |")
        out.append("")
    out += verdict(rows)
    out += fixed_section(rows)
    out += ownframe_section(rows)
    out += docked_section(load_docked(results))
    out += norm_section(load_norms(results))
    text = "\n".join(out) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
    return text


#: a gain whose interval comes this close to zero is reported as an edge result
EDGE = 0.05


def across_sets(rows):
    """Whether the sets give the same answer, when there is more than one.

    One set can only say what that set does. The paragraph this returns is the reason for
    measuring more than one: it names each set's best reading of the trajectory with its
    interval and says whether they agree.
    """
    if len(rows) < 2:
        return []
    best = []
    for rec in rows:
        m = rec.get("models") or {}
        tags = [t for t in TAGS if t.startswith("md_") and t not in CONTROLS and t in m]
        if not tags:
            continue
        top = max(tags, key=lambda t: m[t]["delta"])
        clears = [t for t in tags if m[t]["lo"] > 0]
        best.append((rec["set"], int(rec["n"]), top, m[top], clears, tags,
                     [float(m[t]["lo"]) for t in clears]))
    if len(best) < 2:
        return []

    parts = "; ".join(f"**{nm}** ({n} ligands) {d['delta']:+.3f} "
                      f"[{d['lo']:+.3f}, {d['hi']:+.3f}], {LABEL[t]}"
                      for nm, n, t, d, _c, _a, _l in best)
    # a set counts as ahead when any reading of its frames clears the interval, which need not
    # be the reading with the largest gain
    ahead = [b for b in best if b[4]]
    behind = [b for b in best if b[3]["hi"] < 0]

    if not ahead:
        head = (f"**No set shows a gain from the trajectory.** On each of the {len(best)} sets "
                f"the best reading of the simulation is at or below the crystal pose: {parts}.")
        tail = (f" {len(behind)} of the {len(best)} are behind it beyond the interval."
                if behind else
                f" None of the {len(best)} intervals excludes zero in either direction.")
        close = (" So the answer this item gave on one set is not a property of that set: an "
                 "ensemble around a pose that is already right adds nothing on any of them.")
    elif len(ahead) == len(best):
        head = (f"**Every set shows a gain from the trajectory.** On all {len(best)} of them "
                f"some reading of it beats the crystal pose beyond the interval. The largest "
                f"gain on each set: {parts}.")
        tail = ""
        close = (" That is the opposite of what one set said, so the earlier reading was a "
                 "property of that set and the item has to be read again.")
    else:
        head = (f"**The sets disagree.** Some reading of the trajectory beats the crystal pose "
                f"beyond the interval on {len(ahead)} of the {len(best)} sets "
                f"({', '.join(b[0] for b in ahead)}) and on none of the others. The largest "
                f"gain on each set: {parts}.")
        tail = ""
        close = (" A result that holds on some sets and not on others is not a method yet; what "
                 "separates them is the next thing to measure.")

    edge = ""
    if ahead:
        tried = sum(len(b[5]) for b in best)
        lows = [lo for b in best for lo in b[6]]
        near = min(lows) if lows else 0.0
        if near < EDGE:
            edge = (f" **Read that carefully.** {tried} readings of a trajectory were compared "
                    f"with their crystal pose across the {len(best)} sets, and the nearest edge "
                    f"of a clearing interval is {near:+.3f} from zero. One result at the edge "
                    f"of {tried} comparisons is about what chance produces, so what the sets "
                    f"support is a weak statement and not a method.")
    return [head + tail + close + edge, ""]


def combined_difference(rows, tag=FIXED_TAG, runs=RUNS, seed=SEED):
    """One difference in q2 for the whole comparison, from one reading on every set.

    Each set keeps its own molecules: they are resampled inside their set, never across sets,
    because two sets have different targets and different activity scales. The sets are then
    averaged with a weight equal to the number of molecules they contribute.
    """
    sets = []
    for rec in rows:
        m = rec.get("models") or {}
        y = rec.get("activity")
        if not y or tag not in m or "crystal" not in m:
            continue
        sets.append((np.asarray(y, float), np.asarray(m[tag]["pred"], float),
                     np.asarray(m["crystal"]["pred"], float)))
    if len(sets) < 2:
        return None
    w = np.asarray([len(y) for y, _a, _b in sets], float)
    w = w / w.sum()
    point = float(sum(wi * (r2_score(y, a) - r2_score(y, b))
                      for wi, (y, a, b) in zip(w, sets)))
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(int(runs)):
        total, ok = 0.0, True
        for wi, (y, a, b) in zip(w, sets):
            idx = rng.randint(0, len(y), len(y))
            if np.std(y[idx]) <= 0:
                ok = False
                break
            total += wi * (r2_score(y[idx], a[idx]) - r2_score(y[idx], b[idx]))
        if ok:
            out.append(total)
    if not out:
        return None
    return {"delta": point, "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "sets": len(sets),
            "n": int(sum(len(y) for y, _a, _b in sets))}


def fixed_section(rows, tag=FIXED_TAG):
    """The one reading that was fixed first, instead of the best of several.

    Every gain in the tables above is the largest of several readings of the same frames, and
    that is what makes a single interval at the edge of zero so weak. This section reports one
    reading of the trajectory on every set, chosen for having no free parameter in it rather
    than for winning, and one combined number for the three sets together.
    """
    have = [r for r in rows if tag in (r.get("models") or {}) and "crystal" in r["models"]]
    if not have:
        return []
    # whether the fixed reading is also the one that won somewhere, which the reader has to be
    # told either way: a rule that names the winner is not a rule
    top = []
    for rec in have:
        m = rec["models"]
        cand = [t for t in TAGS if t.startswith("md_") and t not in CONTROLS and t in m]
        if cand and max(cand, key=lambda t: m[t]["delta"]) == tag:
            top.append(rec["set"])
    if not top:
        won = f"It is not the largest gain on any of the {len(have)} sets."
    elif len(top) == len(have):
        won = "It is also the largest gain on every one of these sets."
    else:
        won = (f"It is also the largest gain on {len(top)} of the {len(have)} sets "
               f"({', '.join(top)}).")
    out = ["## One reading fixed in advance", "",
           f"Every gain above is the largest of several readings of the same frames. This "
           f"section reports one reading instead: the trajectory with every conformation "
           f"weighing the same, the \"{LABEL[tag]}\" row of the tables above. It was "
           f"fixed before the sets were compared with "
           f"each other, and it contains no choice -- there is no weighting to pick and no beta "
           f"to search -- so it cannot be the reading that happened to win. {won}", "",
           "| set | ligands | the crystal pose | " + LABEL[tag] +
           " | change against the crystal pose |",
           "| --- | ---: | ---: | ---: | --- |"]
    for rec in have:
        m = rec["models"]
        r, c = m[tag], m["crystal"]
        out.append(f"| {rec['set']} | {rec['n']} | {c['q2']:+.3f} | {r['q2']:+.3f} | "
                   f"{r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] |")
    out.append("")

    comb = combined_difference(have, tag=tag)
    if comb:
        out += [f"**Combined over the {comb['sets']} sets: {comb['delta']:+.3f} "
                f"[{comb['lo']:+.3f}, {comb['hi']:+.3f}]**, over {comb['n']} molecules. Each "
                f"set is weighted by the molecules it contributes, and the molecules are "
                f"resampled inside their own set, never across sets: two sets have different "
                f"targets and their activities are not on one scale.", ""]
    clears = [r["set"] for r in have if r["models"][tag]["lo"] > 0]
    behind = [r["set"] for r in have if r["models"][tag]["hi"] < 0]
    ahead = comb is not None and comb["lo"] > 0
    if not clears and not ahead:
        close = (f"**Fixed in advance, the trajectory shows no gain.** It clears the interval on "
                 f"none of the {len(have)} sets, and the combined estimate does not clear it "
                 f"either. The gains in the tables above are what taking the best of several "
                 f"readings produces. This is the number the item should be read as giving.")
    elif clears and ahead:
        close = (f"**Fixed in advance, the trajectory still shows a gain**, on "
                 f"{len(clears)} of the {len(have)} sets ({', '.join(clears)}) and in the "
                 f"combined estimate, so it does not depend on choosing the reading after the "
                 f"sets were seen.")
    elif ahead:
        close = (f"**The combined estimate clears the interval although no single set does.** "
                 f"Pooling {comb['n']} molecules can do that, and it rests on the sets being "
                 f"comparable enough to pool, which three targets with three activity scales "
                 f"are only in part.")
    else:
        close = (f"**One set clears the interval and the combined estimate does not** "
                 f"({', '.join(clears)}), so what is measured there belongs to that set rather "
                 f"than to the method.")
    if behind:
        close += (f" The reading is behind the crystal pose beyond the interval on "
                  f"{len(behind)} of the {len(have)} sets ({', '.join(behind)}).")
    out += [close, ""]
    out += ["This is not a pre-registration. The sets had been measured before the rule was "
            "written down, and no rule written afterwards can undo that. What it does avoid is "
            "the other fault: deciding which of several readings to believe after seeing which "
            "of them won. The attenuation and lambda are still chosen by cross-validation "
            "inside every model, which the crystal pose gets in the same measure, so that "
            "choice is not what separates the two columns.", ""]
    return out


def ownframe_section(rows):
    """What a field model loses when the series is not in one frame.

    The same frames read twice. Everywhere else in this table a trajectory is superimposed on
    the one structure all the runs of its set were built from, so the whole series sits in a
    single frame. Here each trajectory is superimposed on its own first frame instead, which
    leaves every ligand in a frame of its own.
    """
    have = [r for r in rows if "md_ownframe" in (r.get("models") or {})
            and "md_uniform" in r["models"]]
    if not have:
        return []
    out = ["## What the common frame is worth", "",
           "The same frames, read twice, with the same weighting (clusters equal). Everywhere "
           "above, a trajectory is superimposed on the one structure all the runs of its set "
           "were built from, so the whole series sits in a single frame. Here each trajectory "
           "is superimposed on its own first frame instead, which leaves every ligand in a "
           "frame of its own.", "",
           "| set | ligands | one common frame | each in its own frame | "
           "change against the common frame |",
           "| --- | ---: | ---: | ---: | --- |"]
    for rec in have:
        m = rec["models"]
        u, o = m["md_uniform"], m["md_ownframe"]
        v = o.get("vs_uniform") or {}
        change = (f"{v['delta']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]" if v
                  else f"{o['q2'] - u['q2']:+.3f}")
        out.append(f"| {rec['set']} | {rec['n']} | {u['q2']:+.3f} | {o['q2']:+.3f} | "
                   f"{change} |")
    gaps = [r["models"]["md_ownframe"]["q2"] - r["models"]["md_uniform"]["q2"] for r in have]
    one_way = all(g < 0 for g in gaps)
    if one_way:
        size = (f"{abs(gaps[0]):.3f} of q2" if len(have) == 1 else
                f"between {min(abs(g) for g in gaps):.3f} and "
                f"{max(abs(g) for g in gaps):.3f} of q2")
        where = "this set" if len(have) == 1 else f"all {len(have)} sets"
        head = (f"**The common frame is what makes any of this measurable.** Dropping it costs "
                f"{size} on {where}.")
    else:
        head = (f"**Dropping the common frame does not change q2 in one direction.** The change "
                f"is between {min(gaps):+.3f} and {max(gaps):+.3f} of q2 over the {len(have)} "
                f"sets, and the sign is not the same on all of them.")
    # the loss has an interval of its own, measured on the same molecules, so the one-set
    # reading of this control can be stated as weakly as it deserves
    clear = [r for r in have
             if (r["models"]["md_ownframe"].get("vs_uniform") or {}).get("hi", 1.0) < 0]
    if not clear:
        says = (" No interval excludes zero, so these sets do not settle how much the common "
                "frame is worth.")
    elif len(clear) == len(have):
        says = (" The interval excludes zero on every set, so the loss is larger than the "
                f"resampling of the molecules explains.")
    else:
        names = ", ".join(r["set"] for r in clear)
        says = (f" The interval excludes zero on {len(clear)} of the {len(have)} sets "
                f"({names}), so only there is the change larger than the resampling of the "
                f"molecules explains.")
    why = (" Two runs of the same protein equilibrate to conformations a few tenths of an "
           "angstrom apart, so fitting each ligand to its own run leaves the series in as many "
           "frames as it has molecules.")
    if one_way:
        close = (" The misalignment is far smaller than the ones part C of the plan is about, "
                 "which is a measure of how much a field kernel asks of an alignment.")
    else:
        close = (" A field kernel was expected to lose from that, and on one set it does. On "
                 "the others it does not, so the measurement does not support the general "
                 "statement that a common frame is what makes this comparison possible. What "
                 "an alignment of a few tenths of an angstrom is worth is set by set here, not "
                 "by the method.")
    ahead = [r for r in have if r["models"]["md_ownframe"].get("delta", 0.0) > 0]
    if ahead:
        extra = "; ".join(f"{r['set']} {r['models']['md_ownframe']['delta']:+.3f}"
                          for r in ahead)
        close += (f" One more reason to read the gains of the tables above with care: on "
                  f"{len(ahead)} of the {len(have)} sets this control is itself above the "
                  f"crystal pose ({extra}), and it was built to be wrong.")
    out += ["", head + says + why + close, ""]
    return out


def verdict(rows):
    out = ["## What it settles", ""]
    for rec in rows:
        m = rec["models"]
        name = rec["set"]
        best_tag, best = None, None
        for tag in TAGS:
            if tag.startswith("md_") and tag in m:
                if best is None or m[tag]["delta"] > best["delta"]:
                    best_tag, best = tag, m[tag]
        if best is None:
            continue
        ahead = best["lo"] > 0
        behind = best["hi"] < 0
        if ahead:
            out += [f"**On {name} the trajectory beats the crystal pose.** The best reading of "
                    f"it, {LABEL[best_tag]}, gains {best['delta']:+.3f} in q2 "
                    f"[{best['lo']:+.3f}, {best['hi']:+.3f}], which clears the interval. The "
                    f"crystal pose reaches {m['crystal']['q2']:+.3f} and it reaches "
                    f"{best['q2']:+.3f}.", ""]
        elif behind:
            out += [f"**On {name} the trajectory is worse than the crystal pose.** The best "
                    f"reading of it, {LABEL[best_tag]}, is {best['delta']:+.3f} "
                    f"[{best['lo']:+.3f}, {best['hi']:+.3f}] behind, which clears the interval "
                    f"in the wrong direction. Averaging a ligand over its own motion costs "
                    f"information that the single curated pose keeps.", ""]
        else:
            others = [t for t in TAGS
                      if t.startswith("md_") and t in m and m[t]["lo"] > 0]
            cost = ("" if others else
                    ", so on this set the simulation is not worth its cost")
            out += [f"**On {name} the trajectory neither beats the crystal pose nor loses to "
                    f"it.** The best reading of it, {LABEL[best_tag]}, differs by "
                    f"{best['delta']:+.3f} [{best['lo']:+.3f}, {best['hi']:+.3f}], an interval "
                    f"that contains zero{cost}. The crystal pose reaches "
                    f"{m['crystal']['q2']:+.3f} and it reaches {best['q2']:+.3f}.", ""]
        clears = [t for t in TAGS if t.startswith("md_") and t in m and m[t]["lo"] > 0]
        if clears and not ahead:
            t = max(clears, key=lambda k: m[k]["lo"])
            r = m[t]
            # at beta = 0 the multi-instance fit is the uniform ensemble, so the reading to
            # compare it with is that one: the same model, and it does not clear the interval
            same = (t == "md_mil" and float(m[t].get("beta", -1.0)) == 0.0
                    and "md_uniform" in m)
            if same:
                u = m["md_uniform"]
                why = (f"At beta = 0 the multi-instance fit is the uniform ensemble, so the "
                       f"reading to hold it against is {LABEL['md_uniform']}, which is the "
                       f"same model: that gives {u['delta']:+.3f} [{u['lo']:+.3f}, "
                       f"{u['hi']:+.3f}] and does not clear the interval. What separates them "
                       f"is where the kernel is normalised, which the last section of this "
                       f"table measures.")
            else:
                why = "The two differ in how the frames are weighted."
            out += [f"**A different reading of the same frames does clear the interval.** On "
                    f"{name}, {LABEL[t]} is {r['delta']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] "
                    f"ahead of the crystal pose, while the largest gain, {LABEL[best_tag]}, is "
                    f"not. {why} So what this set supports is the weaker statement: some "
                    f"reading of the trajectory is ahead here, and which reading that is was "
                    f"not fixed before the measurement.", ""]

        if "conformers" in m:
            c = m["conformers"]
            out += [f"The control says where that comes from. An ensemble of generated "
                    f"conformations -- what the fourth-dimension study built -- changes q2 by "
                    f"{c['delta']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}] against the same "
                    f"crystal pose. ", ""]
        if "md_uniform" in m and "md_population" in m:
            u, p = m["md_uniform"], m["md_population"]
            gap = abs(u["q2"] - p["q2"])
            nearest = min(abs(u["delta"]), abs(p["delta"]))
            better = "equal weights" if u["q2"] > p["q2"] else "weights by population"
            if gap > nearest:
                out += [f"**How the clusters are weighted matters more than whether the "
                        f"trajectory is used at all.** Equal weights give {u['q2']:+.3f} and "
                        f"weights by population {p['q2']:+.3f}, a gap of {gap:.3f}, which is "
                        f"larger than the {nearest:.3f} between the better of the two and the "
                        f"crystal pose. {better.capitalize()} is the better reading here. The "
                        f"item asks for equal weights; weights by population are what "
                        f"`cmfqsar/trajectory.py` writes, on the argument that an equilibrated "
                        f"trajectory has already sampled the distribution. That argument is "
                        f"about the physics and this measurement is about the model, and on "
                        f"this set they disagree.", ""]
            else:
                out += [f"How the clusters are weighted changes little: equal weights give "
                        f"{u['q2']:+.3f} and weights by population {p['q2']:+.3f}, a gap of "
                        f"{gap:.3f}, smaller than the {nearest:.3f} between the better of them "
                        f"and the crystal pose.", ""]
        if "md_mil" in m:
            mi = m["md_mil"]
            beta = float(mi.get("beta", float("nan")))
            chose = (" The search chose beta = 0, which is the plain ensemble with every "
                     "conformation weighing the same, so on this set letting the model pick a "
                     "bioactive frame gained nothing over averaging them." if beta == 0.0 else
                     f" The search chose beta = {beta:g}, so the model did concentrate the "
                     f"weight on some frames rather than averaging them evenly.")
            out += [f"Letting the model choose the frames gives {mi['q2']:+.3f}, "
                    f"{mi['delta']:+.3f} [{mi['lo']:+.3f}, {mi['hi']:+.3f}] against the crystal "
                    f"pose.{chose}", ""]
    out += across_sets(rows)
    out += [f"**What this rests on.** "
            + ("One set, simulated" if len(rows) < 2 else
               f"{len(rows)} sets, each simulated") +
            " with GAFF in TIP3P water for the length "
            "stated in `examples/md/README.md`. A few nanoseconds samples the ligand's motion "
            "around its crystal pose and not a change of binding mode, which is the fair "
            "scope of the question asked here. The ligands stayed where they started -- the "
            "data statement gives the median distance from the crystal pose -- so this measures "
            "what an ensemble around a correct pose is worth, which is exactly the question, "
            "and not whether a simulation can find a pose.", ""]
    lost = [rec["set"] for rec in rows if _loses(rec)]
    won = [rec["set"] for rec in rows if not _loses(rec)]
    if lost:
        where = ("here" if len(rows) < 2 else
                 f"on {len(lost)} of the {len(rows)} sets ({', '.join(lost)})")
        out += [f"**What follows for the plan.** The item said that if the trajectory mean does "
                f"not beat the crystal pose then plan items D9 and D10 are not built on it. No "
                f"reading of the trajectory beats the crystal pose {where}, so D9 -- solvent as "
                f"a field, read off these trajectories -- does not follow from this result. D10 "
                f"was built before this measurement and on conformer ensembles rather than "
                f"trajectories, and its own verdict already stands.", ""]
        if won:
            out += [f"The gate is no longer a plain no, because some reading is ahead beyond the "
                    f"interval on {', '.join(won)}. D9 was built in any case, at the author's "
                    f"instruction, and it failed on its own evidence: two readings of its "
                    f"definition sweep were written down beforehand and both failed on a set "
                    f"they were not read from. So the gate reopening changes what is still open "
                    f"here, which is the trajectory itself, and not the verdict of D9.", ""]
    return out


def _loses(rec):
    """True when no reading of the trajectory is ahead of the crystal pose beyond the interval."""
    m = rec.get("models") or {}
    for tag in TAGS:
        if tag.startswith("md_") and tag not in CONTROLS and tag in m:
            lo = m[tag].get("lo")
            if lo is not None and lo == lo and lo > 0:
                return False
    return True


# ----------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sets", nargs="*", default=list(SETS))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="?", const=os.path.join(RESULTS, "table.md"), default=None)
    ap.add_argument("--docked", action="store_true",
                    help="measure the same comparison on poses that were found rather than "
                         "read: needs examples/make_docked_poses.py and a simulation started "
                         "from its poses")
    ap.add_argument("--normalisation", action="store_true",
                    help="measure the uniform ensemble against the multi-instance fit at "
                         "beta = 0, which is the same model with the kernel normalised "
                         "elsewhere")
    a = ap.parse_args(argv)

    if a.report:
        print(report(path=a.report))
        print(f"written {a.report}")
        return 0
    os.makedirs(RESULTS, exist_ok=True)
    for name in a.sets:
        if a.normalisation:
            stage_normalisation(name, force=a.force)
        elif a.docked:
            stage_docked(name, force=a.force)
        else:
            run_set(name, force=a.force)
    report(path=os.path.join(RESULTS, "table.md"))
    print(f"written {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
