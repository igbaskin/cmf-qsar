#!/usr/bin/env python
"""
What the ligand field carries that its contacts with the pocket do not: kernel CCA of the two.

The chemogenomics use of canonical correlation pairs a kernel over ligands with a kernel over
targets, and needs several targets whose ligands share a frame. This repository has one pocket
per data set, so that form is not available. What it does have is a pocket's description of
every ligand: the interaction profile of ``cmfqsar/complementarity.py``, the complementarity of
the ligand's fields with the pocket's fields resolved by receptor residue and interaction type
(electrostatic, hydrophobic, donor to acceptor, acceptor to donor, burial). That profile is the
pocket's view of the ligand, and it is what a chemist means by "the contacts the molecule makes".

So the pairing measured here is, for each series with a stored binding site: view A, the five
fields of the ligand; view B, the linear kernel of its interaction profile with the pocket. The
directions of the field the profile contains are the field's contacts; the directions it does
not contain are the parts of the field that change no contact, which is where a model of the
ligand alone is fitting something other than an interaction with this pocket. Gate (ii) asks
whether those directions carry activity. The profile is computed from the fields, so this is
not two independent measurements of the molecules: it is the field seen through the pocket
against the field seen on its own, and the question is how much of the second the first keeps.

The analysis is the one of ``run_kcca.py``; the two views are not fields on the same space, so
the co-field cosine of gate (iii) does not apply. The sets are the free-energy collections that
ship a binding site (``examples/pockets/``) and have at least twenty ligands.

Usage
-----
    python benchmarks/run_kcca_pocket.py
    python benchmarks/run_kcca_pocket.py --sets openff/thrombin schindler/syk --force
    python benchmarks/run_kcca_pocket.py --report
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

from rdkit import RDLogger  # noqa: E402

import kernel_cache as KCACHE  # noqa: E402
from cmfqsar import kcca as KC  # noqa: E402
from cmfqsar.complementarity import (interaction_profile, pocket_fields,  # noqa: E402
                                     pocket_residues, profile_matrix)
from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.evidence import log_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.protein import find_pocket, read_structure  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RESULTS = os.path.join(HERE, "results", "kcca_pocket")
COLLECTIONS = ("openff", "schindler")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHAS = (0.3, 0.5, 1.0, 2.0)
LAMS = (1e-2, 0.1, 1.0)
COMPONENTS = 3
PERMUTATIONS = 200
FOLDS = 5
SEED = 0
MIN_MOLECULES = 20
#: a ligand further than this from every pocket atom (median over its atoms) is not in the pocket
MAX_APPROACH = 6.0
#: series that are another series' molecules again, in the same frame, against a pocket file that
#: differs in a loop the ligands do not touch; counted once
DUPLICATES = {"schindler/eg5_alternativeloop": "schindler/eg5"}


def _fieldset(alpha):
    return FieldSet.create(physchem=FIELDS, alpha=float(alpha), charge_model="gasteiger")


def _field_view(mols, names, alpha):
    fs = _fieldset(alpha)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    probe = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = probe.joint_kernel(KCACHE.training_kernels(probe, AtomTable.from_molecules(mf)))
    return fs, mf, K


def _choose_alpha(mols, names, y):
    best = None
    for a in ALPHAS:
        _fs, _mf, K = _field_view(mols, names, a)
        for lam in LAMS:
            e = log_evidence(K, y, float(lam), fit_bias=True)
            if np.isfinite(e) and (best is None or e > best[0]):
                best = (e, float(a))
    return best[1]


def sets_with_pockets(collections=COLLECTIONS):
    """[(tag, sdf path, pocket path)] of every series that has a stored binding site."""
    out = []
    for coll in collections:
        for path in sorted(glob.glob(os.path.join(ROOT, "examples", coll, "*.sdf"))):
            tag = f"{coll}/{os.path.splitext(os.path.basename(path))[0]}"
            if tag in DUPLICATES:
                continue
            try:
                from rdkit import Chem
                first = next((m for m in Chem.SDMolSupplier(path) if m is not None), None)
            except Exception:                           # noqa: BLE001
                first = None
            code = first.GetProp("pdb") if first is not None and first.HasProp("pdb") else None
            pocket = find_pocket(path, code)
            if pocket:
                out.append((tag, path, pocket))
    return out


def _closest_approach(mols, prot):
    """Median over the ligands' atoms of the distance to the nearest pocket atom, in angstroms."""
    from scipy.spatial.distance import cdist
    L = np.vstack([m.GetConformer().GetPositions() for m in mols])
    return float(np.median(cdist(L, np.asarray(prot.coords, float)).min(axis=1)))


def run_set(tag, path, pocket_path, force=False, results=RESULTS):
    os.makedirs(results, exist_ok=True)
    out_path = os.path.join(results, tag.replace("/", "__") + ".json")
    if os.path.exists(out_path) and not force:
        print(f"{tag}: already done")
        return json.load(open(out_path))
    t0 = time.time()
    ds = load_dataset(path)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    if int(ok.sum()) < MIN_MOLECULES:
        print(f"{tag}: only {int(ok.sum())} molecules with an activity; skipped")
        return None
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    prot = read_structure(pocket_path, keep_waters=False)
    approach = _closest_approach(mols, prot)
    if approach > MAX_APPROACH:
        print(f"{tag}: the ligands sit {approach:.0f} A from {os.path.basename(pocket_path)}, "
              "which is not their frame; skipped")
        return None
    alpha = _choose_alpha(mols, names, y)
    fs, mf, K = _field_view(mols, names, alpha)
    pocket = pocket_fields(prot, fs)
    labels, order = pocket_residues(prot, pocket)
    profiles = [interaction_profile(m, pocket, fs, labels, order) for m in mf]
    P = np.nan_to_num(profile_matrix(profiles))
    L = P @ P.T
    print(f"=== {tag}: {len(mols)} molecules, alpha {alpha:g}, pocket {prot.n_residues} residues, "
          f"profile of {P.shape[1]} terms", flush=True)
    res = KC.analyse(K, L, y=y, ridge=None, components=COMPONENTS, folds=FOLDS, seed=SEED,
                     permutations=PERMUTATIONS)
    s = res.summary()
    # the profile on its own as a predictor, for context
    q2_profile, _cv, _lam = KC.activity_on_kernel(KC.centre(L), y, folds=FOLDS, seed=SEED)
    d, lo, hi = s["own_minus_contained"]
    print(f"  held-out rho {np.round(s['rho_held_out'][:3], 3)}, {s['real']} of "
          f"{s['pairs_held_out']} real, {s['contained']} of {s['pairs']} contained; q2 field "
          f"{s['q2_all']:+.3f} contained {s['q2_contained']:+.3f} own {s['q2_own']:+.3f} "
          f"({d:+.3f} [{lo:+.3f}, {hi:+.3f}]); profile alone {q2_profile:+.3f}", flush=True)
    out = {"set": tag, "n": len(mols), "alpha": alpha, "residues": int(prot.n_residues),
           "closest_approach": approach, "profile_terms": int(P.shape[1]),
           "q2_profile": float(q2_profile), "field": s, "seconds": round(time.time() - t0, 1)}
    json.dump(out, open(out_path, "w"), indent=1)
    return out


def report(results=RESULTS):
    files = sorted(glob.glob(os.path.join(results, "*.json")))
    if not files:
        return "No results yet; run `python benchmarks/run_kcca_pocket.py`."
    data = sorted([json.load(open(f)) for f in files], key=lambda d: -d["n"])
    lines = ["# What the ligand field carries that its contacts with the pocket do not", "",
             "View A is the five fields of the ligand at the attenuation the evidence picks for the "
             "series; view B is the linear kernel of its interaction profile with the stored "
             "binding site, the complementarity of the fields resolved by receptor residue and "
             "interaction type. *Held-out ρ* is the mean canonical correlation over all pairs on "
             "held-out molecules; *real* counts the pairs above the 95th percentile of a shuffled "
             "pairing; *contained* counts the directions of the field of which the profile explains "
             "more than half the variance on held-out molecules. The q² values are kernel ridge "
             "regression on the whole field, on its part inside the contained directions (the "
             "contacts), on the rest, and on the profile alone, with the regularisation chosen "
             "inside the folds.", "",
             "| series | n | α | residues | mean ρ | real | contained | q² field | q² contacts | "
             "q² field's own | own − contacts [95 %] | q² profile alone |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|"]
    qa, qc, qo, qp, dd = [], [], [], [], []
    above = below = 0
    for d in data:
        f = d["field"]
        diff, lo, hi = f["own_minus_contained"]
        above += int(np.isfinite(lo) and lo > 0)
        below += int(np.isfinite(hi) and hi < 0)
        qa.append(f["q2_all"]); qc.append(f["q2_contained"]); qo.append(f["q2_own"])
        qp.append(d["q2_profile"]); dd.append(diff)
        lines.append(f"| {d['set']} | {d['n']} | {d['alpha']:g} | {d['residues']} | "
                     f"{f['rho_held_out_mean']:+.3f} | {f['real']}/{f['pairs_held_out']} | "
                     f"{f['contained']}/{f['pairs']} | {f['q2_all']:+.3f} | {f['q2_contained']:+.3f} "
                     f"| {f['q2_own']:+.3f} | {diff:+.3f} [{lo:+.3f}, {hi:+.3f}] | "
                     f"{d['q2_profile']:+.3f} |")
    n = len(data)
    lines.append(f"| **mean** |  |  |  |  |  |  | **{np.nanmean(qa):+.3f}** | **{np.nanmean(qc):+.3f}** "
                 f"| **{np.nanmean(qo):+.3f}** | **{np.nanmean(dd):+.3f}** | **{np.nanmean(qp):+.3f}** |")
    share = np.mean([d["field"]["contained"] / max(d["field"]["pairs"], 1) for d in data])
    real = np.mean([d["field"]["real"] / max(d["field"]["pairs_held_out"], 1) for d in data])
    lines += ["",
              f"Over {n} series: {100 * real:.0f} % of the canonical pairs hold out above the "
              f"shuffled pairing, and the profile contains {100 * share:.0f} % of the field's "
              f"directions on average. The contacts predict the activity at q² "
              f"**{np.nanmean(qc):+.3f}** on average, the field's own part at **{np.nanmean(qo):+.3f}**, "
              f"the whole field at **{np.nanmean(qa):+.3f}** and the profile alone at "
              f"**{np.nanmean(qp):+.3f}**; the difference (own minus contacts) is "
              f"**{np.nanmean(dd):+.3f}**, with its interval above zero on {above} and below zero on "
              f"{below} of the {n} series.", ""]
    if below > n / 2:
        lines.append("So the field directions that change a contact with the pocket are where the "
                     "activity sits, and the part of the field that touches nothing carries less: "
                     "the models of these series are fitting interactions, not the alignment.")
    elif above > n / 2:
        lines.append("So the activity sits largely in field directions that change no contact "
                     "with the stored pocket, which points at something the pocket as stored does "
                     "not show: a moving side chain, a water, a solvent-exposed effect, or the "
                     "alignment itself.")
    else:
        lines.append(f"So on {n - above - below} of the {n} series the interval covers zero, and "
                     "the few that do not fall on both sides: on sets of twenty to forty ligands "
                     "neither part can be said to carry more of the activity (rule 4). What the "
                     "table does show is that the contact profile is not a predictor on its own, "
                     "and that the whole field predicts better than either of its parts.")
    lines.append("")
    lines.append(f"Series whose ligands sit more than {MAX_APPROACH:g} Å from the stored pocket are "
                 "left out, because the structure is not in their frame; a series that is "
                 "another's molecules again, against a pocket file differing in a loop the ligands "
                 "do not touch, is counted once.")
    text = "\n".join(lines) + "\n"
    with open(os.path.join(results, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:          # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=None, help="tags such as openff/thrombin")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if not a.report:
        for tag, path, pocket in sets_with_pockets():
            if a.sets and tag not in a.sets:
                continue
            try:
                run_set(tag, path, pocket, force=a.force)
            except Exception as exc:                     # noqa: BLE001 - one series must not stop the rest
                print(f"  {tag}: FAILED -- {type(exc).__name__}: {exc}", flush=True)
    print(report())


if __name__ == "__main__":
    main()
