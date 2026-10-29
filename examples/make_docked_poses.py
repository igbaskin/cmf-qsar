#!/usr/bin/env python
"""Poses found without the crystal pose, for plan item D8.

Plan item D8 asks whether a conformational ensemble from a short simulation models a series
better than the single pose it started from. Measured on crystal poses the answer is no, and the
item's own reading of that is that the question is about a set whose poses are **not** known:
there an ensemble would be covering uncertainty in the pose, not motion around a pose that is
already right.

This script produces such a set from a set whose poses *are* known, so that the comparison still
has a crystal pose to be judged against. Every ligand is posed while it is held out:

1. the ligands are split into folds (five by default) with a fixed seed,
2. for each fold, a model is fitted on the other ligands in their crystal poses,
3. the fold's ligands have their poses destroyed, exactly as `benchmarks/run_poses.py` destroys
   them: a random rigid placement within 2 A of the site,
4. one posing method, fixed before the run, recovers them. With the default `pocket` method that
   is `cmfqsar/docking.py`: the receptor's own fields, complementarity less clash, with no model
   and no crystal frame of any kind,
5. the pose that was found is written out with its distance from the crystal pose.

So no ligand is ever posed with the help of its own crystal pose, and none is posed with the help
of its own activity. What is still borrowed from the crystal is the **conformation**: the pose is
rigid, so the ligand keeps the shape the collection gives it and only its placement is searched.

The result is one SDF in `examples/docked/`, which `examples/make_md_trajectories.py --ligands`
simulates and `benchmarks/run_md.py --docked` measures.

Usage
-----
    python examples/make_docked_poses.py --set mcl1
    python examples/make_docked_poses.py --set mcl1 --method mcs --folds 5
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

NEWLINE = chr(10)
from typing import Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.poses import POSERS, _scramble, pose_rmsd  # noqa: E402

RDLogger.DisableLog("rdApp.*")

LIGANDS = os.path.join(HERE, "openff")
POCKETS = os.path.join(HERE, "pockets", "openff")
OUT_DIR = os.path.join(HERE, "docked")

#: the same parameterisation the pose benchmark of item C1 uses, so that the median error
#: reported there is the error this script starts the simulations from
FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
ALPHA = 0.5
LAM = 0.1
RESTARTS = 2
MAX_TRANSLATION = 2.0
FOLDS = 5
SEED = 0
ACTIVITY = "pAffinity"


def load(name, data=LIGANDS):
    """The ligands of one free-energy set in their crystal poses, with their activities."""
    path = os.path.join(data, name + ".sdf")
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing; this script works on the openff collection")
    mols, names, y = [], [], []
    for i, m in enumerate(Chem.SDMolSupplier(path, removeHs=False)):
        if m is None or not m.HasProp(ACTIVITY):
            continue
        try:
            value = float(m.GetProp(ACTIVITY))
        except ValueError:
            continue
        label = (m.GetProp("_Name") if m.HasProp("_Name") and m.GetProp("_Name")
                 else f"{name}-{i + 1}")
        m.SetProp("_Name", label)
        mols.append(m)
        names.append(label)
        y.append(value)
    return mols, names, np.asarray(y, float)


def load_pocket(name, fieldset, pockets=POCKETS):
    """The fields of the binding site, in the same parameterisation as the ligands."""
    from cmfqsar.complementarity import pocket_fields
    from cmfqsar.protein import read_structure

    path = os.path.join(pockets, name + ".pdb")
    if not os.path.isfile(path):
        return None
    return pocket_fields(read_structure(path, keep_waters=False), fieldset)


def folds_of(n, folds=FOLDS, seed=SEED):
    """Every index in exactly one fold, so that every ligand is posed while it is held out."""
    idx = np.arange(int(n))
    np.random.RandomState(int(seed)).shuffle(idx)
    return [sorted(int(i) for i in part) for part in np.array_split(idx, int(folds))]


def recover(name, method="pocket", folds=FOLDS, seed=SEED, alpha=ALPHA, lam=LAM,
            restarts=RESTARTS, out_dir=OUT_DIR, log=print):
    """Pose every ligand of a set while it is held out, and write the poses that were found."""
    if method not in POSERS:
        raise SystemExit(f"no posing method called {method}; "
                         f"available: {', '.join(sorted(POSERS))}")
    mols, names, y = load(name)
    fs = FieldSet.create(physchem=FIELDS, alpha=alpha, charge_model="gasteiger",
                         param_scheme="rdkit")
    pocket = None
    if "pocket" in method:
        pocket = load_pocket(name, fs)
        if pocket is None:
            raise SystemExit(f"{method} needs a pocket structure and {name} has none in "
                             f"{POCKETS}")
        log(f"  pocket: {pocket.n_atoms} atoms of the binding site")

    t0 = time.time()
    found, rmsds, folded = [None] * len(mols), [float("nan")] * len(mols), [0] * len(mols)
    for k, test in enumerate(folds_of(len(mols), folds, seed)):
        held = set(test)
        train = [i for i in range(len(mols)) if i not in held]
        table = AtomTable.from_molecules([build_fields(mols[i], fs, names[i]) for i in train])
        model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": float(lam)})
        model.fit(table, y[train], cv_folds=0)
        probe = [Chem.Mol(mols[i]) for i in test]
        _scramble(probe, fs, seed=int(seed), max_translation=MAX_TRANSLATION)
        ctx = {"reference": mols[train[int(np.argmax(y[train]))]], "fieldset": fs,
               "model": model, "restarts": int(restarts), "seed": int(seed),
               "max_rmsd": None, "conformers": 0, "pocket": pocket, "pocket_weight": 1.0}
        POSERS[method](probe, ctx)
        for i, p in zip(test, probe):
            found[i], rmsds[i], folded[i] = p, float(pose_rmsd(p, mols[i])), k + 1
        log(f"  fold {k + 1} of {folds}: {len(test)} ligands posed on a model of "
            f"{len(train)}, median {np.median([rmsds[i] for i in test]):.2f} A from their "
            f"crystal poses")

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{name}_{method}.sdf")
    with Chem.SDWriter(out) as w:
        for m, nm, value, r, k in zip(found, names, y, rmsds, folded):
            m.SetProp("_Name", nm)
            m.SetProp(ACTIVITY, f"{float(value):.6g}")
            m.SetProp("pose_rmsd", f"{r:.4f}")
            m.SetProp("pose_method", method)
            m.SetProp("pose_fold", str(k))
            w.write(m)
    rec = {"set": name, "method": method, "n": len(mols), "folds": int(folds),
           "seed": int(seed), "fields": FIELDS, "alpha": float(alpha), "lam": float(lam),
           "restarts": int(restarts), "max_translation": MAX_TRANSLATION,
           "pocket_atoms": None if pocket is None else int(pocket.n_atoms),
           "median_rmsd": float(np.median(rmsds)), "max_rmsd": float(np.max(rmsds)),
           "within_1A": float(np.mean(np.asarray(rmsds) <= 1.0)),
           "within_2A": float(np.mean(np.asarray(rmsds) <= 2.0)),
           "rmsd": {nm: round(float(r), 4) for nm, r in zip(names, rmsds)},
           "seconds": float(time.time() - t0)}
    with open(os.path.join(out_dir, f"{name}_{method}.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    log(f"  {len(mols)} poses -> {out}")
    log(f"  median {rec['median_rmsd']:.2f} A from the crystal pose, at most "
        f"{rec['max_rmsd']:.2f} A; {rec['within_1A']:.0%} within 1 A, "
        f"{rec['within_2A']:.0%} within 2 A")
    return rec


README = """# Poses found without the crystal pose

One SDF per set and posing method: every ligand of a free-energy set in the pose that was found
for it **while it was held out**. Plan item D8 asks what a conformational ensemble is worth where
the pose is not known, and these files are the starting point for that question;
`examples/make_md_trajectories.py --ligands` simulates them and `benchmarks/run_md.py --docked`
measures them.

## What each file is

`<set>_<method>.sdf`, one record per ligand under the ligand's own title, carrying

* `pAffinity`, the activity as the source collection gives it,
* `pose_rmsd`, how far the pose that was found is from the crystal pose, in angstrom, measured
  in the frame both are already in and without superposition,
* `pose_method`, the posing method, and `pose_fold`, the fold the ligand was held out in.

`<set>_<method>.json` holds the same numbers as a record of the run, including every ligand's
`pose_rmsd`.

## How they were made

{recipe}

{entries}

## Where the data comes from

* The ligands and their activities: the Open Force Field protein-ligand-benchmark, free under
  MIT, already in `examples/openff/` with its licence.
* The binding sites: `examples/pockets/openff/`, cut from the PDB entries those files name. The
  wwPDB distributes its holdings in the public domain under CC0.
"""


def write_readme(records, out_dir=OUT_DIR):
    """The data statement, built from what was actually run."""
    if not records:
        return
    r = records[0]
    recipe = (f"`examples/make_docked_poses.py` is the script and the whole recipe: nothing was "
              f"done by hand. The ligands of a set are split into **{r['folds']} folds** with "
              f"seed {r['seed']}. For each fold a CMF model is fitted on the other ligands in "
              f"their crystal poses (fields {', '.join(r['fields'])}, attenuation "
              f"{r['alpha']:g}, lambda {r['lam']:g}), the fold's ligands have their poses "
              f"destroyed -- a random rigid placement within {r['max_translation']:g} A of the "
              f"site, the same destruction `benchmarks/run_poses.py` uses -- and one posing "
              f"method puts them back. So no ligand is posed with the help of its own crystal "
              f"pose or its own activity. What is still taken from the crystal structure is the "
              f"*conformation*: the posing is rigid, so only the placement is searched.")
    entries = NEWLINE.join(
        f"* `{x['set']}_{x['method']}.sdf`: {x['n']} ligands, method **{x['method']}**"
        + (f" ({x['pocket_atoms']} atoms of the binding site)" if x.get("pocket_atoms") else "")
        + f". The poses sit a median {x['median_rmsd']:.2f} A from the crystal poses, at most "
          f"{x['max_rmsd']:.2f} A; {x['within_1A']:.0%} within 1 A, {x['within_2A']:.0%} "
          f"within 2 A."
        for x in records)
    with open(os.path.join(out_dir, "README.md"), "w", newline=NEWLINE) as fh:
        fh.write(README.format(recipe=recipe, entries=entries))


def merge_records(records, out_dir=OUT_DIR):
    """This run's records, plus the ones it did not make, so the statement covers every file."""
    out = []
    for x in records:
        out.append({k: v for k, v in x.items() if k != "rmsd"})
    fresh = {(x["set"], x["method"]) for x in out}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
        try:
            with open(path) as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(old, dict) and (old.get("set"), old.get("method")) not in fresh:
            out.append({k: v for k, v in old.items() if k != "rmsd"})
    return sorted(out, key=lambda x: (x.get("set", ""), x.get("method", "")))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", dest="sets", action="append", default=None,
                    help="a set of examples/openff (repeat for several); default mcl1")
    ap.add_argument("--method", default="pocket",
                    help="the posing method, fixed before the run (default pocket: the "
                         "receptor's fields alone, with no model and no crystal frame)")
    ap.add_argument("--folds", type=int, default=FOLDS,
                    help=f"how many held-out folds the posing is done in (default {FOLDS})")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=OUT_DIR)
    a = ap.parse_args(argv)
    records = []
    for name in (a.sets or ["mcl1"]):
        print(f"{name}: posing every ligand while it is held out, method {a.method}")
        records.append(recover(name, method=a.method, folds=a.folds, seed=a.seed,
                               out_dir=a.out))
    write_readme(merge_records(records, out_dir=a.out), out_dir=a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
