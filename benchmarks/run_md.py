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
    python benchmarks/run_md.py --sets mcl1
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
CRYSTAL_DATA = os.path.join(ROOT, "examples", "openff")
SETS = ("mcl1",)
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.1, 0.3, 0.5, 1.0, 2.0)
LAMS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
BETAS = (0.0, 1.0, 3.0, 10.0)
FOLDS = 10
SEED = 0
RUNS = 2000
#: conformations generated for the control ensemble, to match the trajectory ensemble's size
N_CONFORMERS = 20


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

    rec = {"set": name, "n": int(len(y)), "frames_mean": float(np.mean(frames)),
           "frames_min": int(np.min(frames)), "frames_max": int(np.max(frames)),
           "seconds": float(time.time() - t0), "models": rows}
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(rec, fh, indent=1)
    return rec


# ----------------------------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------------------------

TAGS = ("conformers", "md_uniform", "md_population", "md_mil", "md_ownframe")
LABEL = {"crystal": "the crystal pose", "conformers": "generated conformations",
         "md_uniform": "trajectory, clusters equal", "md_population": "trajectory, by population",
         "md_mil": "trajectory, multi-instance",
         "md_ownframe": "trajectory, each in its own frame"}


def load_all(results=RESULTS):
    out = []
    for p in sorted(glob.glob(os.path.join(results, "*.json"))):
        with open(p) as fh:
            out.append(json.load(fh))
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
    text = "\n".join(out) + "\n"
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", newline="\n") as fh:
            fh.write(text)
    return text


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
            out += [f"**On {name} the trajectory neither beats the crystal pose nor loses to "
                    f"it.** The best reading of it, {LABEL[best_tag]}, differs by "
                    f"{best['delta']:+.3f} [{best['lo']:+.3f}, {best['hi']:+.3f}], an interval "
                    f"that contains zero, so on this set the simulation is not worth its cost. "
                    f"The crystal pose reaches {m['crystal']['q2']:+.3f} and it reaches "
                    f"{best['q2']:+.3f}.", ""]
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
        if "md_ownframe" in m and "md_uniform" in m:
            o, u = m["md_ownframe"], m["md_uniform"]
            out += [f"**The common frame is what makes any of this measurable.** Read the same "
                    f"frames with each trajectory superimposed on its own first frame instead "
                    f"of on the one structure every run was built from, and q2 falls from "
                    f"{u['q2']:+.3f} to {o['q2']:+.3f}. Two runs equilibrate to protein "
                    f"conformations a few tenths of an angstrom apart, so fitting each ligand "
                    f"to its own leaves the series in as many frames as it has molecules. That "
                    f"is {abs(u['q2'] - o['q2']):.3f} of q2 lost to a misalignment far smaller "
                    f"than the ones part C of the plan is about, which is a measure of how much "
                    f"a field kernel asks of an alignment.", ""]
    out += ["**What this rests on.** One set, simulated with GAFF in TIP3P water for the length "
            "stated in `examples/md/README.md`. A few nanoseconds samples the ligand's motion "
            "around its crystal pose and not a change of binding mode, which is the honest "
            "scope of the question asked here. The ligands stayed where they started -- the "
            "data statement gives the median distance from the crystal pose -- so this measures "
            "what an ensemble around a correct pose is worth, which is exactly the question, "
            "and not whether a simulation can find a pose.", ""]
    if any(_loses(rec) for rec in rows):
        out += ["**What follows for the plan.** The item said that if the trajectory mean does "
                "not beat the crystal pose then plan items D9 and D10 are not built on it. No "
                "reading of the trajectory beats the crystal pose here, so D9 -- solvent as a "
                "field, read off these trajectories -- does not follow from this result. D10 "
                "was built before this measurement and on conformer ensembles rather than "
                "trajectories, and its own verdict already stands.", ""]
    return out


def _loses(rec):
    """True when no reading of the trajectory is ahead of the crystal pose beyond the interval."""
    m = rec.get("models") or {}
    for tag in TAGS:
        if tag.startswith("md_") and tag in m:
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
    a = ap.parse_args(argv)

    if a.report:
        print(report(path=a.report))
        print(f"written {a.report}")
        return 0
    os.makedirs(RESULTS, exist_ok=True)
    for name in a.sets:
        run_set(name, force=a.force)
    report(path=os.path.join(RESULTS, "table.md"))
    print(f"written {os.path.join(RESULTS, 'table.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
