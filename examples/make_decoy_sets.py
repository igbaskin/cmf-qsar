"""
Build ready-to-use background (decoy) sets in ``examples/decoys/``.

For each target the background is drawn from the *other* example series, so the decoys are
molecules with measured activity against a different protein: presumed inactive against this
target, and chemically unrelated to its ligands.  Every decoy is then placed in the frame of the
training actives by maximising its field similarity to them (``cmfqsar.decoys``), which gives it
its best possible pose and makes the resulting enrichment conservative.

The decoys are written with ``subset = decoy``, so the program loads them as a background set
automatically (GUI: *Add decoys…*; CLI: ``--decoys``).

Note the limits of such a background: ligands of unrelated targets differ from the actives in size,
shape and charge, which makes the screening task easy and the enrichment optimistic.  For a serious
evaluation use a property-matched set (DUD-E, DEKOIS 2.0) or measured inactives (LIT-PCBA); see the
README.

Usage:
    python examples/make_decoy_sets.py                      # default targets
    python examples/make_decoy_sets.py --targets THR,GPB --n-decoys 60 --restarts 4
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.decoys import align_decoys  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SOURCE = ("background set for virtual-screening evaluation: ligands of other targets from "
          "examples/sutherland, posed by maximising their CMF field similarity to the training "
          "actives of this target (examples/make_decoy_sets.py)")


def build(target: str, n_decoys: int, restarts: int, alpha: float, out_dir: str, seed: int = 0) -> None:
    t0 = time.time()
    path = os.path.join(HERE, "sutherland", f"{target}.sdf")
    if not os.path.isfile(path):
        print(f"{target}: {path} not found, skipped")
        return
    ds = load_dataset(path)
    ok = np.isfinite(ds.activity)
    tr = [i for i in ds.train_indices if ok[i]]
    if not tr:
        print(f"{target}: no training molecules with activity, skipped")
        return
    # actives of the target: the more potent half of the training set
    thr = float(np.median(ds.activity[tr]))
    act = [i for i in tr if ds.activity[i] >= thr]
    fs = FieldSet.create(alpha=alpha, charge_model="file")
    ref = AtomTable.from_molecules([build_fields(ds.mols[i], fs, ds.names[i]) for i in act])

    pool = []
    for other in sorted(glob.glob(os.path.join(HERE, "sutherland", "*.sdf"))):
        name = os.path.basename(other)[:-4]
        if name == target:
            continue
        od = load_dataset(other)
        pool += [(name, m, nm) for m, nm in zip(od.mols, od.names)]
    rng = np.random.RandomState(seed)
    pick = rng.choice(len(pool), size=min(n_decoys, len(pool)), replace=False)

    mf, labels = [], []
    for k in pick:
        src, m, nm = pool[k]
        mf.append(build_fields(m, fs, f"{src}_{nm}"))
        labels.append((src, m, nm))
    res = align_decoys(mf, ref, fs, restarts=restarts,
                       progress=lambda i, n: (print("   %d/%d" % (i, n), flush=True)
                                              if (i % 25 == 0 or i == n) else None))

    out = os.path.join(out_dir, f"{target}_decoys.sdf")
    w = Chem.SDWriter(out)
    for (src, m, nm), mol, r in zip(labels, mf, res):
        mm = Chem.Mol(m)
        conf = mm.GetConformer()
        for a, xyz in enumerate(mol.coords):
            conf.SetAtomPosition(a, [float(v) for v in xyz])
        for p in list(mm.GetPropNames()):
            if p not in ("pIC50", "pKi"):
                mm.ClearProp(p)
        mm.SetProp("_Name", f"decoy_{src}_{nm}")
        mm.SetProp("subset", "decoy")
        mm.SetProp("origin", src)
        mm.SetProp("target_of_this_background", target)
        mm.SetProp("field_similarity", f"{r.similarity_after:.3f}")
        mm.SetProp("closest_active", r.best_match)
        mm.SetProp("source", SOURCE)
        w.write(mm)
    w.close()
    sim = np.array([r.similarity_after for r in res])
    before = np.array([r.similarity_before for r in res])
    print(f"{target:5s} {len(res):4d} decoys from {len(set(l[0] for l in labels))} other targets, "
          f"aligned to {len(act)} actives: similarity {before.mean():.3f} -> {sim.mean():.3f} "
          f"(max {sim.max():.3f}), {time.time() - t0:.0f} s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="THR,GPB,ACE,AChE")
    ap.add_argument("--n-decoys", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(HERE, "decoys"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for t in [x.strip() for x in a.targets.split(",") if x.strip()]:
        build(t, a.n_decoys, a.restarts, a.alpha, a.out)


if __name__ == "__main__":
    main()
