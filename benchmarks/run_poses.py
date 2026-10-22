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
from cmfqsar.poses import POSERS, judge, table  # noqa: E402

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


def split(n, fraction=TEST_FRACTION, seed=0):
    """A random split; the held-out molecules are the ones whose poses are destroyed."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    k = max(3, int(round(fraction * n)))
    return sorted(int(i) for i in idx[k:]), sorted(int(i) for i in idx[:k])


def run_set(name, methods=METHODS, results=RESULTS, seed=0, alpha=ALPHA, lam=LAM,
            restarts=2, max_rmsd=None, force=False, reconform=False, suffix=""):
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
    scores = judge(mols, names, y, tr, te, fs, methods=methods, lam=lam, seed=seed,
                   restarts=restarts, max_rmsd=max_rmsd, reconform=reconform,
                   progress=lambda msg: print(f"    {msg}", flush=True))
    res = {"set": name, "n": len(mols), "n_train": len(tr), "n_test": len(te),
           "fields": FIELDS, "alpha": alpha, "lam": lam, "seed": seed,
           "restarts": restarts, "max_rmsd": max_rmsd, "reconform": reconform,
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


def report(results=RESULTS, out=None):
    rows = []
    for fn in sorted(os.listdir(results)) if os.path.isdir(results) else []:
        if fn.endswith(".json"):
            rows.append(json.load(open(os.path.join(results, fn))))
    if not rows:
        raise SystemExit(f"no results in {results}")
    rows.sort(key=lambda r: -r["n"])
    order = ["crystal"] + [m for m in METHODS]
    lines = ["# Judging an alignment against crystal poses", "",
             "Every ligand of these FEP sets is co-crystallised in one frame, so the pose a",
             "method should have found is known. A model is fitted on the training molecules in",
             "their crystal poses; the held-out molecules then have their poses destroyed, and",
             "each method recovers them without seeing a held-out activity. `crystal` is the",
             "ceiling, `scrambled` the floor, and `recovery` the fraction of the pose error a",
             "method removed.", "",
             f"Fields: {', '.join('`%s`' % f for f in FIELDS)}; alpha {rows[0]['alpha']}, "
             f"lambda {rows[0]['lam']}, {int(100 * TEST_FRACTION)}% held out.", "",
             "## Median RMSD to the crystal pose, angstrom", "",
             "| set | n | " + " | ".join(order) + " |",
             "|---|---:|" + "---:|" * len(order)]
    for r in rows:
        cells = []
        for m in order:
            v = r["methods"].get(m)
            cells.append("-" if v is None else f"{v['median_rmsd']:.2f}")
        lines.append(f"| {r['set']} | {r['n']} | " + " | ".join(cells) + " |")
    lines += ["", "## External R2pred of the held-out molecules in those poses", "",
              "| set | n | " + " | ".join(order) + " |",
              "|---|---:|" + "---:|" * len(order)]
    for r in rows:
        cells = []
        for m in order:
            v = r["methods"].get(m)
            cells.append("-" if v is None else f"{v['r2_pred']:+.3f}")
        lines.append(f"| {r['set']} | {r['n']} | " + " | ".join(cells) + " |")
    if len(rows) > 1:
        lines += ["", "## Mean over the sets", "",
                  "| method | median RMSD | within 2 A | recovery | R2pred |",
                  "|---|---:|---:|---:|---:|"]
        for m in order:
            got = [r["methods"][m] for r in rows if m in r["methods"]]
            if not got:
                continue
            rec = np.nanmean([g["recovery"] for g in got])
            lines.append(f"| {m} | {np.mean([g['median_rmsd'] for g in got]):.2f} A | "
                         f"{np.mean([g['within_2A'] for g in got]):.0%} | "
                         + ("-" if not np.isfinite(rec) else f"{rec:+.0%}") + " | "
                         f"{np.mean([g['r2_pred'] for g in got]):+.3f} |")
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
    ap.add_argument("--methods", nargs="*", default=list(METHODS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--lam", type=float, default=LAM)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--max-rmsd", type=float, default=None,
                    help="displacement guard for the confidence search (angstrom)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reconform", action="store_true",
                    help="generate the held-out conformations instead of using the crystal ones")
    ap.add_argument("--flex", action="store_true",
                    help="shorthand for --reconform with the flexible methods")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.report:
        return report()
    sets = SETS if args.sets == ["all"] else args.sets
    methods = tuple(FLEX_METHODS) if args.flex else tuple(args.methods)
    reconform = args.reconform or args.flex
    for name in sets:
        run_set(name, methods=methods, seed=args.seed, alpha=args.alpha,
                lam=args.lam, restarts=args.restarts, max_rmsd=args.max_rmsd,
                force=args.force, reconform=reconform,
                suffix="_flex" if reconform else "")
    if len(sets) > 1:
        report()


if __name__ == "__main__":
    main()
