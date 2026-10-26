#!/usr/bin/env python
"""
Judging the alignment methods against crystal poses (item C1 of the development plan).

q2 cannot say whether an alignment method works, because realigning changes the geometry q2 is
computed on: a method is rewarded for making the data look like the model, and the README records
q2 and external R2pred moving in opposite directions under exactly that operation.  The judge has
to come from outside the model, and the FEP benchmark sets supply it -- every ligand of
``examples/openff`` is co-crystallised in one frame, so the pose a method should have found is
known.

For each set: fit a model on the training molecules in their crystal poses, destroy the poses of
the held-out molecules, and ask each method to recover them without ever seeing a held-out
activity.  Report, for every method, both of the things that matter -- how near the crystal pose
it got, and what the model then predicts -- between the two fixed references:

    crystal     the held-out molecules in their crystal poses: the ceiling
    scrambled   the same molecules left as the control scrambled them: the floor

A method is worth what it moves from the floor towards the ceiling.  ``recovery`` is that fraction
of the pose error removed; it can be negative, and when it is, the table says so.

Usage
-----
    python benchmarks/run_poses.py                      # the default sets
    python benchmarks/run_poses.py --sets jnk1 p38 --methods mcs field variance
    python benchmarks/run_poses.py --report
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
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.fields import FieldSet  # noqa: E402
from cmfqsar.poses import POSERS, judge, judge_training_frame, table  # noqa: E402

RDLogger.DisableLog("rdApp.*")

DATA = os.path.join(ROOT, "examples", "openff")
RESULTS = os.path.join(HERE, "results", "poses")

#: the openff sets with enough ligands for a model and a held-out tenth
SETS = ["syk", "hif2a", "bace", "pde10", "bace_hunt", "pfkfb3", "cdk8", "p38", "eg5", "tnks2",
        "mcl1", "shp2", "ptp1b", "thrombin", "jnk1", "pde2"]
DEFAULT_SETS = ["jnk1", "thrombin", "p38", "mcl1"]
METHODS = ("scrambled", "mcs", "field", "variance", "mcs+variance")
#: the flexible search of C2 has a conformation to fix as well as a pose, so it is
#: judged with --reconform, where the held-out conformation is generated rather than read
FLEX_METHODS = ("scrambled", "mcs", "mcs+variance", "flex", "mcs+flex")
#: the field-point graph of fieldgraph.py, judged beside the substructure alignment it is meant
#: to do without, and in series with the kernel and confidence refinements
GRAPH_METHODS = ("scrambled", "mcs", "field", "fieldgraph", "fieldgraph+field",
                 "fieldgraph+variance")
#: item C3: the receptor as the prior and the model's confidence as the likelihood. These need
#: the pocket, so they are judged with --pocket and only on the sets a structure exists for
POCKET_METHODS = ("scrambled", "mcs", "variance", "mcs+variance", "pocket",
                  "pocket+variance", "mcs+pocket+variance")
POCKETS = os.path.join(ROOT, "examples", "pockets", "openff")
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5
LAM = 0.1
TEST_FRACTION = 0.3


def load(name, data=DATA):
    """The ligands of one FEP set in their crystal poses, with pAffinity as the activity."""
    path = os.path.join(data, name + ".sdf")
    mols, names, y = [], [], []
    for i, m in enumerate(Chem.SDMolSupplier(path, removeHs=False)):
        if m is None or not m.HasProp("pAffinity"):
            continue
        try:
            value = float(m.GetProp("pAffinity"))
        except ValueError:
            continue
        label = m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name") else f"{name}-{i + 1}"
        m.SetProp("_Name", label)
        mols.append(m)
        names.append(label)
        y.append(value)
    return mols, names, np.asarray(y, float)


def load_pocket(name, fieldset, pockets=POCKETS):
    """The fields of the receptor's binding site, in the same parameterisation as the ligands."""
    from cmfqsar.complementarity import pocket_fields
    from cmfqsar.protein import read_structure

    path = os.path.join(pockets, name + ".pdb")
    if not os.path.exists(path):
        return None
    return pocket_fields(read_structure(path, keep_waters=False), fieldset)


def split(n, fraction=TEST_FRACTION, seed=0):
    """A random split; the held-out molecules are the ones whose poses are destroyed."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    k = max(3, int(round(fraction * n)))
    return sorted(int(i) for i in idx[k:]), sorted(int(i) for i in idx[:k])


def run_set(name, methods=METHODS, results=RESULTS, seed=0, alpha=ALPHA, lam=LAM,
            restarts=2, max_rmsd=None, force=False, reconform=False, suffix="",
            conformers=0, pocket_weight=1.0):
    out_path = os.path.join(results, f"{name}{suffix}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    mols, names, y = load(name)
    tr, te = split(len(mols), seed=seed)
    print(f"\n=== {name}: {len(mols)} ligands with crystal poses, "
          f"{len(tr)} training / {len(te)} held out; pAffinity "
          f"{y.min():.1f} to {y.max():.1f} (sd {y.std():.2f})", flush=True)
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit")
    pocket = None
    if any("pocket" in m for m in methods):
        pocket = load_pocket(name, fs)
        if pocket is None:
            print(f"    no pocket structure for {name}: the pocket methods are skipped")
            methods = tuple(m for m in methods if "pocket" not in m)
        else:
            print(f"    pocket: {pocket.n_atoms} atoms of the binding site", flush=True)
    scores = judge(mols, names, y, tr, te, fs, methods=methods, lam=lam, seed=seed,
                   restarts=restarts, max_rmsd=max_rmsd, reconform=reconform,
                   conformers=conformers, pocket=pocket, pocket_weight=pocket_weight,
                   progress=lambda msg: print(f"    {msg}", flush=True))
    res = {"set": name, "n": len(mols), "n_train": len(tr), "n_test": len(te),
           "fields": FIELDS, "alpha": alpha, "lam": lam, "seed": seed,
           "restarts": restarts, "max_rmsd": max_rmsd, "reconform": reconform,
           "conformers": conformers,
           "pocket": None if pocket is None else int(pocket.n_atoms),
           "pocket_weight": float(pocket_weight) if pocket is not None else None,
           "activity": {"min": float(y.min()), "max": float(y.max()), "sd": float(y.std())},
           "methods": {k: s.row() for k, s in scores.items()},
           "rmsd": {k: [float(v) for v in s.rmsd] for k, s in scores.items()},
           "seconds": time.time() - t0}
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(table(scores, title=name), flush=True)
    print(f"    written to {out_path}  ({res['seconds']:.0f} s)", flush=True)
    return res


def variant(row):
    """What a result file is a run of, as a key and as a sentence."""
    if row.get("frame"):
        # a training-frame run answers a different question from a pose run, so it never shares
        # a table with one
        co = row.get("coalign", {})
        posers = tuple(sorted(m for m in row.get("methods", {}) if m != "crystal"))
        what = (f"the training frame destroyed and recovered: {co.get('rounds', '?')} rounds, "
                f"guard {co.get('max_rmsd')} A"
                + (", flexible E-step" if co.get("flexible") else ", rigid E-step"))
        return (2, int(bool(co.get("flexible"))), 0.0, posers), what
    recon = bool(row.get("reconform"))
    confs = int(row.get("conformers") or 0)
    guard = row.get("max_rmsd")
    # the set of methods is part of what a run is: two runs of the same protocol over different
    # posers are not rows of one table, and merging them gives a set two rows with missing cells in both
    posers = tuple(sorted(m for m in row.get("methods", {}) if m != "crystal"))
    key = (recon, confs, -1.0 if guard is None else float(guard), posers)
    what = ("regenerated conformations" if recon else "crystal conformations, pose scrambled")
    if recon:
        what += (f", {confs} generated starting conformations" if confs
                 else ", the generated conformation alone")
    what += (f", displacement guard {guard:g} A" if guard else ", no displacement guard")
    if any(p.startswith("fieldgraph") for p in posers):
        what += "; field-point graph alignment"
    return key, what


def _table(group, methods, label):
    lines = [f"## {label}", "",
             "### Median RMSD to the crystal pose, angstrom", "",
             "| set | n | " + " | ".join(methods) + " |",
             "|---|---:|" + "---:|" * len(methods)]
    for r in group:
        cells = [("-" if m not in r["methods"] else f"{r['methods'][m]['median_rmsd']:.2f}")
                 for m in methods]
        lines.append(f"| {r['set']} | {r['n']} | " + " | ".join(cells) + " |")
    lines += ["", "### External R2pred of the held-out molecules in those poses", "",
              "| set | n | " + " | ".join(methods) + " |",
              "|---|---:|" + "---:|" * len(methods)]
    for r in group:
        cells = [("-" if m not in r["methods"] else f"{r['methods'][m]['r2_pred']:+.3f}")
                 for m in methods]
        lines.append(f"| {r['set']} | {r['n']} | " + " | ".join(cells) + " |")
    if len(group) > 1:
        lines += ["", "### Mean over the sets", "",
                  "| method | median RMSD | within 1 A | within 2 A | recovery | R2pred | "
                  "conformer limit |", "|---|---:|---:|---:|---:|---:|---:|"]
        for m in methods:
            got = [r["methods"][m] for r in group if m in r["methods"]]
            if not got:
                continue
            rec = np.nanmean([g["recovery"] for g in got])
            lim = np.mean([g.get("conformer_limit", 0.0) for g in got])
            lines.append(f"| {m} | {np.mean([g['median_rmsd'] for g in got]):.2f} A | "
                         f"{np.mean([g['within_1A'] for g in got]):.0%} | "
                         f"{np.mean([g['within_2A'] for g in got]):.0%} | "
                         + ("-" if not np.isfinite(rec) else f"{rec:+.0%}") + " | "
                         f"{np.mean([g['r2_pred'] for g in got]):+.3f} | {lim:.2f} A |")
    return lines + [""]


def run_frame_set(name, results=RESULTS, seed=0, alpha=ALPHA, lam=LAM, force=False,
                  rounds=4, folds=4, max_rmsd=1.0, max_similarity=0.98, flexible=False,
                  conformers=0, suffix="_frame"):
    """Item C4: destroy the *training* frame and ask a method to recover it."""
    out_path = os.path.join(results, f"{name}{suffix}.json")
    if os.path.exists(out_path) and not force:
        print(f"[skip] {out_path} exists")
        return json.load(open(out_path))
    t0 = time.time()
    mols, names, y = load(name)
    tr, te = split(len(mols), seed=seed)
    print(f"\n=== {name}: training frame of {len(tr)} ligands destroyed and recovered, "
          f"{len(te)} held out in their crystal poses", flush=True)
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit")
    kw = {"rounds": rounds, "folds": folds, "max_rmsd": max_rmsd,
          "max_similarity": max_similarity, "flexible": flexible, "conformers": conformers,
          "seed": seed}
    if flexible:
        kw["rdmols"] = [mols[i] for i in tr]
    scores = judge_training_frame(mols, names, y, tr, te, fs, lam=lam, seed=seed,
                                  coalign_kwargs=kw,
                                  progress=lambda m: print(f"    {m}", flush=True))
    res = {"set": name, "n": len(mols), "n_train": len(tr), "n_test": len(te),
           "fields": FIELDS, "alpha": alpha, "lam": lam, "seed": seed, "frame": True,
           "coalign": {k: v for k, v in kw.items() if k != "rdmols"},
           "methods": {k: v.row() for k, v in scores.items()},
           "rmsd": {k: [float(x) for x in v.rmsd] for k, v in scores.items()},
           "seconds": time.time() - t0}
    os.makedirs(results, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(res, fh, indent=1)
    print(table(scores, title=f"{name}: the training frame"), flush=True)
    print(f"    written to {out_path}  ({res['seconds']:.0f} s)", flush=True)
    return res


def report(results=RESULTS, out=None):
    """Every variant in the results folder, each as its own labelled table.

    The folder holds several protocols -- crystal conformations or regenerated ones, with and
    without the conformer ensemble, at several guard settings -- and they are not comparable row
    by row.  Mixing them into one table, which an earlier version did, produces three unlabelled
    rows per set and a mean over runs that answer different questions.
    """
    rows = []
    for fn in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(results, fn))))
    if not rows:
        raise SystemExit(f"no results in {results}")
    groups = {}
    for r in rows:
        key, what = variant(r)
        groups.setdefault((key, what), []).append(r)
    known = (["crystal"] + [m for m in list(METHODS) + list(FLEX_METHODS)
                                   + list(GRAPH_METHODS)] + ["coalign", "mcs+coalign"])
    seen = []
    for m in known:
        if m not in seen:
            seen.append(m)
    lines = ["# Judging an alignment against crystal poses", "",
             "Every ligand of these FEP sets is co-crystallised in one frame, so the pose a",
             "method should have found is known. A model is fitted on the training molecules in",
             "their crystal poses; the held-out molecules then have their poses destroyed, and",
             "each method recovers them without seeing a held-out activity. `crystal` is the",
             "ceiling, `scrambled` the floor, `recovery` the fraction of the pose error a method",
             "removed, and the conformer limit the RMSD the best possible *rigid* placement of",
             "the starting conformation would still leave.", "",
             f"Fields: {', '.join('`%s`' % f for f in FIELDS)}; alpha {rows[0]['alpha']}, "
             f"lambda {rows[0]['lam']}, {int(100 * TEST_FRACTION)}% held out.", ""]
    for (key, what), group in sorted(groups.items(), key=lambda kv: kv[0][0]):
        group.sort(key=lambda r: -r["n"])
        methods = [m for m in seen if any(m in r["methods"] for r in group)]
        lines += _table(group, methods, what)
    text = "\n".join(lines) + "\n"
    path = out or os.path.join(results, "table.md")
    with open(path, "w") as fh:
        fh.write(text)
    print(text)
    print(f"written {path}")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="*", default=DEFAULT_SETS,
                    help=f"openff sets; 'all' runs {len(SETS)} of them")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="which posers to judge; the default depends on --flex and --pocket")
    ap.add_argument("--pocket", action="store_true",
                    help="judge the pocket methods of item C3 (needs examples/pockets/openff)")
    ap.add_argument("--pocket-weight", type=float, default=1.0, metavar="W",
                    help="how much a unit of pocket complementarity is worth against a unit of "
                         "posterior variance in the joint objective")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--max-rmsd", type=float, default=None,
                    help="displacement guard for the confidence search (angstrom)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reconform", action="store_true",
                    help="generate the held-out conformations instead of using the crystal ones")
    ap.add_argument("--conformers", type=int, default=0,
                    help="starting configurations from the conformer generator")
    ap.add_argument("--suffix", default=None, help="tag for the result files")
    ap.add_argument("--coalign", action="store_true",
                    help="item C4: destroy the training frame and recover it (cmfqsar/coalign.py)")
    ap.add_argument("--coalign-rounds", type=int, default=4)
    ap.add_argument("--coalign-flexible", action="store_true",
                    help="use the flexible E-step, which needs the molecules")
    ap.add_argument("--flex", action="store_true",
                    help="shorthand for --reconform with the flexible methods")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        return report()
    sets = SETS if args.sets == ["all"] else args.sets
    if args.coalign:
        for name in sets:
            run_frame_set(name, seed=args.seed, alpha=args.alpha, lam=args.lam,
                          force=args.force, rounds=args.coalign_rounds,
                          max_rmsd=args.max_rmsd if args.max_rmsd is not None else 1.0,
                          flexible=args.coalign_flexible,
                          suffix="_frame_flex" if args.coalign_flexible else "_frame")
        return report()
    methods = (tuple(args.methods) if args.methods else
               tuple(POCKET_METHODS) if args.pocket else
               tuple(FLEX_METHODS) if args.flex else tuple(METHODS))
    reconform = args.reconform or args.flex
    for name in sets:
        default_suffix = ("_flex" if reconform else "")
        if reconform and args.conformers:
            default_suffix += f"_c{args.conformers}"
        if args.max_rmsd is not None:
            default_suffix += f"_g{args.max_rmsd:g}"
        if args.pocket:
            # a pocket run answers a different question from a pose run over the same sets, so
            # it keeps its own files and its own table; the weight is part of what the run is
            default_suffix += "_pocket"
            if abs(args.pocket_weight - 1.0) > 1e-9:
                default_suffix += f"_w{args.pocket_weight:g}".replace(".", "")
        run_set(name, methods=methods, seed=args.seed, alpha=args.alpha,
                lam=args.lam, restarts=args.restarts, max_rmsd=args.max_rmsd,
                force=args.force, reconform=reconform, conformers=args.conformers,
                pocket_weight=args.pocket_weight,
                suffix=args.suffix if args.suffix is not None else default_suffix)
    if len(sets) > 1:
        report()


if __name__ == "__main__":
    main()
