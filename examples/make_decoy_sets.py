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


#: targets that appear under more than one name, so that the same protein is never its own decoy
ALIASES = {"ache": {"ache"}, "thr": {"thr", "thrombin"}, "ace": {"ace"}, "gpb": {"gpb"},
           "cox2": {"cox2"}, "bzr": {"bzr"}, "dhfr": {"dhfr"}}


def _same_target(a: str, b: str) -> bool:
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    for names in ALIASES.values():
        if a in names and b in names:
            return True
    return False


def collect_pool(target: str, wide: bool) -> list:
    """Candidate decoys: every other series shipped here, or only the Sutherland ones."""
    pats = ["sutherland/*.sdf"] if not wide else ["*/*.sdf", "*.sdf"]
    pool = []
    seen = set()
    for pat in pats:
        for other in sorted(glob.glob(os.path.join(HERE, pat))):
            name = os.path.basename(other)[:-4]
            if _same_target(name, target) or "decoy" in other.replace("\\", "/"):
                continue
            try:
                od = load_dataset(other)
            except Exception:                            # noqa: BLE001 - not a data set
                continue
            for m, nm in zip(od.mols, od.names):
                if m is None:
                    continue
                key = Chem.MolToSmiles(m)
                if key in seen:
                    continue
                seen.add(key)
                pool.append((name, m, nm))
    return pool


def build(target: str, n_decoys: int, restarts: int, alpha: float, out_dir: str, seed: int = 0,
          matched: bool = False, max_similarity: float = 0.35, wide: bool = True,
          align_ref_max: int = 0, conformers: int = 1) -> None:
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
    ref_idx = act
    if align_ref_max and len(act) > align_ref_max:
        # posing is the expensive part and its cost grows with the reference; a subset of the
        # actives is enough to define the frame, and the model is still fitted on all of them
        step = len(act) / float(align_ref_max)
        ref_idx = [act[int(k * step)] for k in range(align_ref_max)]
        print(f"{target:5s} posing against {len(ref_idx)} of {len(act)} actives")
    ref = AtomTable.from_molecules([build_fields(ds.mols[i], fs, ds.names[i]) for i in ref_idx])

    pool = collect_pool(target, wide=wide or matched)
    if matched:
        from _matched_decoys import select
        pick, report = select([ds.mols[i] for i in act], [p[1] for p in pool], n_decoys,
                              max_similarity=max_similarity, seed=seed)
        print(f"{target:5s} property matching: {report['dissimilar']}/{report['pool']} pool "
              f"molecules below Tanimoto {max_similarity:g} to every active, {report['selected']} "
              f"selected")
        for name, (a, d) in report["property_gap"].items():
            print(f"        {name:18s} actives {a:8.2f}   decoys {d:8.2f}")
    else:
        rng = np.random.RandomState(seed)
        pick = rng.choice(len(pool), size=min(n_decoys, len(pool)), replace=False)

    mf, labels = [], []
    for k in pick:
        src, m, nm = pool[k]
        mf.append(build_fields(m, fs, f"{src}_{nm}"))
        labels.append((src, m, nm))
    prog = lambda i, n: (print("   %d/%d" % (i, n), flush=True)
                         if (i % 25 == 0 or i == n) else None)
    if conformers > 1:
        from cmfqsar.decoys import align_decoys_flexible
        chosen, res = align_decoys_flexible([l[1] for l in labels], [l[2] for l in labels],
                                            ref, fs, n_conformers=conformers,
                                            restarts=restarts, progress=prog)
        mf = chosen
    else:
        res = align_decoys(mf, ref, fs, restarts=restarts, progress=prog)

    out = os.path.join(out_dir, f"{target}_decoys{'_matched' if matched else ''}.sdf")
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
    if conformers > 1:
        moved = sum(1 for r in res if r.conformer > 0)
        print(f"{target:5s} a generated conformation won for {moved} of {len(res)} decoys")
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
    ap.add_argument("--matched", action="store_true",
                    help="choose decoys that match the physical properties of the actives while "
                         "being topologically unlike them, which is what makes a background hard")
    ap.add_argument("--max-similarity", type=float, default=0.35,
                    help="a decoy's ECFP4 Tanimoto to every active must stay below this")
    ap.add_argument("--conformers", type=int, default=1, metavar="N",
                    help="generate N conformations of every decoy and keep the one that poses "
                         "best; the conformation in the file is always among the candidates")
    ap.add_argument("--align-ref-max", type=int, default=0, metavar="N",
                    help="pose the decoys against at most N of the actives, which is what makes "
                         "a large background affordable")
    ap.add_argument("--wide-pool", action="store_true",
                    help="draw from every example series, not only the Sutherland ones")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for t in [x.strip() for x in a.targets.split(",") if x.strip()]:
        build(t, a.n_decoys, a.restarts, a.alpha, a.out, matched=a.matched,
              max_similarity=a.max_similarity, wide=a.wide_pool, align_ref_max=a.align_ref_max,
              conformers=a.conformers)


if __name__ == "__main__":
    main()
