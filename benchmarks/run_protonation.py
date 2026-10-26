#!/usr/bin/env python
"""
Does a protonation ensemble help, on a series measured in acid? (plan item D4)

``examples/corrosion_hcl`` holds the iron corrosion-inhibitor series protonated for 1 M HCl: one
state per molecule, chosen when the data was prepared.  That choice is a guess wherever a pKa
falls near the pH of the experiment.  The mean-map kernel removes the need to make it: a CMF field
is a sum of atom-centred Gaussians and the kernel is bilinear in the fields, so the mean field of
an ensemble is itself a field, and the members do not have to be conformations of one molecule --
they can be its protonation states (``cmfqsar/protonation.py``).

Three models are measured on each series, with the same fields, the same (alpha, lambda) grids and
the same ten-fold cross-validation, and with the same alignment -- every molecule rigidly
superimposed on the most active large member through the scaffold atoms, as in
``run_corrosion.py``:

  file        the state the data file gives, which is the model as it stands
  dominant    the likeliest state at this pH by its pKa values: one state again, chosen by pKa
              rather than by hand
  ensemble    the pKa-weighted mean field over the states
  tautomer    the mean field over the tautomers of the state the file gives, which is the other
              ensemble of different molecules (``cmfqsar/tautomers.py``).  Its weights are weaker
              than the pKa ones: a tautomer has a population only if the free energies are known,
              and none are computed here, so the forms RDKit scores highest are counted the same

``file`` against ``dominant`` asks whether the prepared state was the right one.  ``dominant``
against ``ensemble`` asks whether being a mixture matters, which is the part only the ensemble can
do.  The pH scan reports, without fitting anything, what share of each series is a real mixture at
each pH, which says where an ensemble can make any difference at all.

What the weights are, and what they are not: each site is treated as independent and given its
Henderson-Hasselbalch fraction, with Dimorphite-DL's patterns finding the sites and its regression
pKa values setting the weights.  Independent sites overestimate multiple ionisation, because a
second proton on a molecule that already carries one is harder to add than its own pKa says, and
the pKa values are regression values for these groups rather than measurements of these molecules.
Both limits are reported with the numbers.

Usage
-----
    python benchmarks/run_protonation.py                     # the scan, then the models
    python benchmarks/run_protonation.py --scan-only
    python benchmarks/run_protonation.py --sets 1_c1ccncc1 --ph 0
    python benchmarks/run_protonation.py --ph 3     # where the mixtures are
    python benchmarks/run_protonation.py --report
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

import run_corrosion as C  # noqa: E402  - the alignment and the scaffold of each series
import run_topomer as T  # noqa: E402  - the fields, the grids and the cross-validation

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields, combine_members  # noqa: E402
from cmfqsar.protonation import available, ensemble_table, mixture_share  # noqa: E402
from cmfqsar.tautomers import ambiguous_share, forms_table, tautomer_fields  # noqa: E402

DATA = os.path.join(ROOT, "examples", "corrosion_hcl")
RESULTS = os.path.join(HERE, "results", "protonation")
#: 1 M HCl, which is what the series in this folder were measured in
PH = 0.0
#: the pH values the scan reports, from strong acid to blood
PH_SCAN = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.4)
MIN_MOLECULES = 15
MODELS = ("file", "dominant", "ensemble", "tautomer")
#: how the tautomers are weighed; see cmfqsar/tautomers.py for why this is the mildest choice
TAUTOMER_WEIGHTING = "score"


def series_files(names=None):
    paths = sorted(glob.glob(os.path.join(DATA, "*.csv")))
    if names:
        want = set(names)
        paths = [p for p in paths if os.path.basename(p)[:-4] in want]
    return paths


def prepare(path, min_molecules=MIN_MOLECULES):
    """Load one series, keep the molecules that carry the scaffold, and align them on it."""
    ds = load_dataset(path)
    y = np.asarray(ds.activity, float)
    ok = np.isfinite(y)
    mols = [m for m, g in zip(ds.mols, ok) if g]
    names = [n for n, g in zip(ds.names, ok) if g]
    y = y[ok]
    core = C.core_of(mols, path)
    if core is None:
        return None
    keep = [i for i, m in enumerate(mols) if m.GetSubstructMatch(core)]
    mols = [mols[i] for i in keep]
    names = [names[i] for i in keep]
    y = y[keep]
    if len(mols) < min_molecules:
        return None
    ref, rmsds = C.align_on_core(mols, y, core)
    finite = [r for r in rmsds if np.isfinite(r)]
    return {"mols": mols, "names": names, "y": y, "core": core, "reference": names[ref],
            "align_rmsd_median": float(np.median(finite)) if finite else None}


# ------------------------------------------------------------------------------ the pH scan


def scan_set(path, ph_values=PH_SCAN, min_molecules=MIN_MOLECULES):
    """How much of one series is a real mixture, at each pH.  No model is fitted."""
    prep = prepare(path, min_molecules)
    if prep is None:
        return None
    name = os.path.basename(path)[:-4]
    out = {"set": name, "n": len(prep["mols"]), "ph": {}}
    for ph in ph_values:
        ens = ensemble_table(prep["mols"], prep["names"], ph=ph)
        q_file = np.array([Chem.GetFormalCharge(m) for m in prep["mols"]], float)
        q_dom = np.array([e.main.charge for e in ens], float)
        out["ph"][f"{ph:g}"] = {
            "mixture_share": mixture_share(ens),
            "states_mean": float(np.mean([len(e) for e in ens])),
            "charge_agrees": float(np.mean(q_dom == q_file)),
            "mean_charge_file": float(q_file.mean()),
            "mean_charge_pka": float(np.mean([e.mean_charge for e in ens])),
            "sites_mean": float(np.mean([len(e.sites) for e in ens])),
        }
        print(f"    pH {ph:4.1f}: {out['ph'][f'{ph:g}']['mixture_share']:5.0%} of the molecules a "
              f"mixture, the likeliest state has the file's charge for "
              f"{out['ph'][f'{ph:g}']['charge_agrees']:5.0%}, mean charge "
              f"{out['ph'][f'{ph:g}']['mean_charge_pka']:+.2f} against "
              f"{out['ph'][f'{ph:g}']['mean_charge_file']:+.2f} in the file", flush=True)
    return out


# ------------------------------------------------------------------------------ the models


def _fields_for(prep, ens, fs):
    """The four field lists of one series at one alpha."""
    names, mols = prep["names"], prep["mols"]
    as_file = [build_fields(m, fs, n, _single=True) for m, n in zip(mols, names)]
    dominant = [build_fields(e.main.mol, fs, n, _single=True) for e, n in zip(ens, names)]
    mixed = []
    for e, n in zip(ens, names):
        members = [build_fields(s.mol, fs, n, _single=True) for s in e.states]
        if len(members) == 1:
            mixed.append(members[0])
        else:
            mixed.append(combine_members(members, e.weights, kind="protonation",
                                         atom_maps=[s.atom_map for s in e.states], name=n))
    forms = [tautomer_fields(m, fs, weighting=TAUTOMER_WEIGHTING, name=n)
             for m, n in zip(mols, names)]
    return {"file": as_file, "dominant": dominant, "ensemble": mixed, "tautomer": forms}


def run_set(path, ph=PH, results=RESULTS, force=False, min_molecules=MIN_MOLECULES):
    name = os.path.basename(path)[:-4]
    out_path = os.path.join(results, f"{name}__ph{ph:g}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    prep = prepare(path, min_molecules)
    if prep is None:
        print(f"=== {name}: too small or no scaffold, skipped", flush=True)
        return None
    y = prep["y"]
    print(f"\n=== {name}: {len(y)} molecules, aligned on {prep['reference']} "
          f"(median RMSD {prep['align_rmsd_median']:.2f} A)", flush=True)
    ens = ensemble_table(prep["mols"], prep["names"], ph=ph)
    forms = forms_table(prep["mols"], weighting=TAUTOMER_WEIGHTING)
    q_file = np.array([Chem.GetFormalCharge(m) for m in prep["mols"]], float)
    q_dom = np.array([e.main.charge for e in ens], float)
    res = {"set": name, "file": os.path.relpath(path, ROOT), "n": len(y), "ph": ph,
           "core": Chem.MolToSmiles(prep["core"]), "reference": prep["reference"],
           "align_rmsd_median": prep["align_rmsd_median"],
           "mixture_share": mixture_share(ens),
           "states_mean": float(np.mean([len(e) for e in ens])),
           "charge_agrees": float(np.mean(q_dom == q_file)),
           "mean_charge_file": float(q_file.mean()),
           "mean_charge_pka": float(np.mean([e.mean_charge for e in ens])),
           "tautomer_share": ambiguous_share(forms),
           "tautomer_forms_mean": float(np.mean([len(e) for e in forms])),
           "tautomer_weighting": TAUTOMER_WEIGHTING,
           "fields": T.FIELDS, "alphas": list(T.ALPHAS), "lams": list(T.LAMS),
           "folds": T.FOLDS, "models": {}}
    print(f"    at pH {ph:g}: {res['mixture_share']:.0%} a mixture, the likeliest state has the "
          f"file's charge for {res['charge_agrees']:.0%}, mean charge "
          f"{res['mean_charge_pka']:+.2f} against {res['mean_charge_file']:+.2f}; "
          f"{res['tautomer_share']:.0%} with more than one tautomer "
          f"({res['tautomer_forms_mean']:.1f} forms on average)", flush=True)
    per_alpha = {k: [] for k in MODELS}
    for alpha in T.ALPHAS:
        fs = FieldSet.create(physchem=T.FIELDS, alpha=alpha, charge_model="gasteiger",
                             param_scheme="rdkit")
        built = _fields_for(prep, ens, fs)
        for k in MODELS:
            per_alpha[k].append((fs, built[k]))
        print(f"    alpha {alpha:g}: fields built ({time.time() - t0:.0f} s)", flush=True)
    for k in MODELS:
        q2, err, alpha, lam = T.best_q2(per_alpha[k], y, T.ALPHAS, T.LAMS)
        res["models"][k] = {"q2": q2, "rmse_cv": err, "alpha": alpha, "lam": lam}
        print(f"    {k:10s} q2 {q2:+.3f}  RMSEcv {err:.3f}  (alpha {alpha:g}, lam {lam:g})",
              flush=True)
    res["seconds"] = time.time() - t0
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(f"    written to {out_path}  ({res['seconds']:.0f} s)", flush=True)
    return res


# ------------------------------------------------------------------------------- the report


def _load(results=RESULTS, kind="models"):
    rows = []
    if not os.path.isdir(results):
        return rows
    for fn in sorted(os.listdir(results)):
        if not fn.endswith(".json") or fn == "table.md":
            continue
        d = json.load(open(os.path.join(results, fn)))
        if kind == "models" and "models" in d:
            rows.append(d)
        if kind == "scan" and "ph" in d and "models" not in d:
            rows.append(d)
    rows.sort(key=lambda d: -d["n"])
    return rows


def report(results=RESULTS):
    models = _load(results, "models")
    scan = _load(results, "scan")
    out = ["## Protonation ensembles on the corrosion series in HCl", "",
           "A protonation ensemble is the same mathematics as a conformational ensemble: the "
           "mean field of the states, weighted by their pKa populations. `file` is the state the "
           "data file gives, `dominant` the likeliest state at this pH by pKa, `ensemble` the "
           "weighted mean over the states, and `tautomer` the mean over the tautomers of the "
           "file's state. Same five fields, same attenuation and ridge grids, same ten-fold "
           "cross-validation, same scaffold alignment as `run_corrosion.py`.", ""]
    if scan:
        out += ["### Where an ensemble can make a difference", "",
                "Share of the molecules of a series that is a real mixture (a second state above "
                "1% of the molecule), and how often the likeliest state carries the charge the "
                "file gives it. No model is fitted for this table.", "",
                "| pH | mixtures | likeliest state agrees with the file | mean charge by pKa | "
                "mean charge in the file |", "| --- | --- | --- | --- | --- |"]
        keys = list(scan[0]["ph"].keys())
        for ph in keys:
            got = [d["ph"][ph] for d in scan if ph in d["ph"]]
            out.append(f"| {ph} | {np.mean([g['mixture_share'] for g in got]):.0%} | "
                       f"{np.mean([g['charge_agrees'] for g in got]):.0%} | "
                       f"{np.mean([g['mean_charge_pka'] for g in got]):+.2f} | "
                       f"{np.mean([g['mean_charge_file'] for g in got]):+.2f} |")
        out += ["", f"{len(scan)} series, "
                    f"{sum(d['n'] for d in scan)} molecules in all.", ""]
    for ph in sorted({d["ph"] for d in models}):
        here = [d for d in models if d["ph"] == ph]
        out += [f"### The models at pH {ph:g}", "",
                "| set | n | mixtures | state agrees | several tautomers | "
                + " | ".join(f"q2 {k}" for k in MODELS) + " |",
                "| --- | --- | --- | --- | --- | " + " | ".join("---" for _ in MODELS) + " |"]
        for d in here:
            best = max(MODELS, key=lambda k: d["models"][k]["q2"])
            cells = []
            for k in MODELS:
                q = f"{d['models'][k]['q2']:+.3f}"
                cells.append(f"**{q}**" if k == best else q)
            out.append(f"| {d['set']} | {d['n']} | {d['mixture_share']:.0%} | "
                       f"{d['charge_agrees']:.0%} | {d.get('tautomer_share', 0.0):.0%} | "
                       + " | ".join(cells) + " |")
        means = {k: float(np.mean([d["models"][k]["q2"] for d in here])) for k in MODELS}
        best = max(MODELS, key=lambda k: means[k])
        out.append("| **mean** | " + f"{int(np.sum([d['n'] for d in here]))}" + " | "
                   + f"{np.mean([d['mixture_share'] for d in here]):.0%}" + " | "
                   + f"{np.mean([d['charge_agrees'] for d in here]):.0%}" + " | "
                   + f"{np.mean([d.get('tautomer_share', 0.0) for d in here]):.0%}" + " | "
                   + " | ".join((f"**{means[k]:+.3f}**" if k == best else f"{means[k]:+.3f}")
                                for k in MODELS) + " |")
        wins = {k: sum(1 for d in here
                       if max(MODELS, key=lambda m: d["models"][m]["q2"]) == k) for k in MODELS}
        pairs = [("dominant", "file"), ("ensemble", "dominant"), ("ensemble", "file"),
                 ("tautomer", "file")]
        out += ["", "Best on " + ", ".join(f"{wins[k]} of {len(here)} series for `{k}`"
                                           for k in MODELS) + ".", "",
                "Mean difference: " + ", ".join(
                    f"`{a}` - `{b}` "
                    f"{np.mean([d['models'][a]['q2'] - d['models'][b]['q2'] for d in here]):+.3f}"
                    for a, b in pairs) + ".", ""]
    if models:
        out += ["Three limits of the weights bear on these numbers. Each ionisable site is "
                "treated as independent, which overestimates multiple ionisation: a second "
                "proton is harder to add than its own pKa says, so the mean charge by pKa is an "
                "upper bound. The pKa values are Dimorphite-DL's regression values for these "
                "groups, not measurements of these molecules. And the tautomer weights are not "
                "populations at all: the forms RDKit scores highest are counted the same, "
                "because no free energies are computed here.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=None, help="series to run (default: all of them)")
    ap.add_argument("--ph", type=float, default=PH, help=f"the pH of the models (default {PH:g})")
    ap.add_argument("--scan-only", action="store_true", help="the pH scan, no models")
    ap.add_argument("--min-molecules", type=int, default=MIN_MOLECULES)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", nargs="?", const="-", default=None,
                    metavar="PATH", help="print the table, or write it to PATH")
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    if a.report is not None:
        text = report()
        if a.report == "-":
            print(text)
        else:
            with open(a.report, "w") as fh:
                fh.write(text + "\n")
            print(f"written to {a.report}")
        return
    if not available():
        raise SystemExit("this benchmark needs Dimorphite-DL: pip install dimorphite-dl")
    os.makedirs(RESULTS, exist_ok=True)
    paths = series_files(a.sets)
    print(f"{len(paths)} series in {os.path.relpath(DATA, ROOT)}")
    scan_path = os.path.join(RESULTS, "ph_scan.json")
    if a.force or not os.path.exists(scan_path):
        scans = []
        for p in paths:
            print(f"\n=== {os.path.basename(p)[:-4]}", flush=True)
            s = scan_set(p, min_molecules=a.min_molecules)
            if s is not None:
                scans.append(s)
        with open(scan_path, "w") as fh:
            json.dump({"sets": scans, "ph_values": list(PH_SCAN)}, fh, indent=1)
        for s in scans:                    # one file per set, so the report can read them alone
            with open(os.path.join(RESULTS, f"scan_{s['set']}.json"), "w") as fh:
                json.dump(s, fh, indent=1)
    else:
        print(f"[skip] {scan_path} exists")
    if not a.scan_only:
        for p in paths:
            run_set(p, ph=a.ph, force=a.force, min_molecules=a.min_molecules)
    print()
    print(report())


if __name__ == "__main__":
    main()
