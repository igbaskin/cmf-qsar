#!/usr/bin/env python
"""
Point-mutant stability series from the mega-scale data set of Tsuboyama et al. (plan item PP1).

Source
------
Tsuboyama, K., Dauparas, J., Chen, J. et al. Mega-scale experimental analysis of protein folding
stability in biology and design. Nature 620, 434-444 (2023).  Data: Zenodo record 7992926,
licence CC BY 4.0.  Two files of that record are used:

* ``Processed_K50_dG_datasets.zip`` -> ``Tsuboyama2023_Dataset2_Dataset3_20230416.csv``, the
  folding free energies inferred from the proteolysis assay (``dG_ML``, kcal/mol, the unfolding
  free energy of the variant; ``ddG_ML`` = dG of the variant minus dG of the wild type, so that a
  negative value is a destabilising mutation);
* ``AlphaFold_model_PDBs.zip``, one model of every wild-type domain with hydrogens, numbered
  from 1 along the sequence that was measured.

What is kept
------------
For each domain named in :data:`DOMAINS`: every single substitution (a label such as ``V1Q``;
insertions, deletions and double mutants are left out) whose ``dG_ML`` lies in the range the
assay resolves, -1 to 5 kcal/mol, and whose ``ddG_ML`` is present.  The domains are natural
proteins (named by a PDB code) from different fold clusters of the data set, with a wild-type
stability above 3 kcal/mol so that destabilising mutations of a few kcal/mol stay inside the
range, and with the most mutants that pass.

Output
------
``examples/mutants/<domain>/mutants.csv`` (label, position, wild-type residue, new residue,
dG_ML, ddG_ML, 95 % interval of dG) and ``<domain>.pdb`` (the AlphaFold model, unchanged).

With ``--build`` the script also builds every mutant structure with PDBFixer and OpenMM
(:func:`cmfqsar.mutants.build_mutant`) into ``benchmarks/cache/megascale/built/<domain>/``,
which `benchmarks/run_mutants.py` reads.  Each one is started from six turns of the new side
chain's first torsion and relaxed with the Amber ff14SB force field, with the mutated residue
and the side chains of its neighbours within 6 angstrom free and every other atom fixed; the
lowest-energy result is kept.  That takes about three seconds per mutant, and ``--jobs`` runs
several at once.

Usage
-----
    python examples/make_mutant_examples.py                        # the tables and the models
    python examples/make_mutant_examples.py --build                # every structure, in order
    python examples/make_mutant_examples.py --build --shard 0 --shards 12   # one share of 12
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.mutants import Mutation  # noqa: E402

CACHE = os.path.join(ROOT, "benchmarks", "cache", "megascale")
TABLE = os.path.join(CACHE, "Tsuboyama2023_Dataset2_Dataset3_20230416.csv")
MODELS = os.path.join(CACHE, "pdb")
OUT = os.path.join(HERE, "mutants")
BUILT = os.path.join(CACHE, "built")

#: PDB code -> fold cluster of the data set, wild-type dG_ML (kcal/mol)
DOMAINS = {"1O6X": (129, 4.17), "1UFM": (174, 3.28), "2K5N": (105, 4.09), "3ZGK": (16, 3.56)}
DG_RANGE = (-1.0, 5.0)


def select(table: str = TABLE, out: str = OUT) -> Dict[str, int]:
    """Write the mutant tables and copy the models; returns the number of mutants per domain."""
    wanted = {f"{d}.pdb": d for d in DOMAINS}
    rows: Dict[str, List[dict]] = {d: [] for d in DOMAINS}
    wild: Dict[str, dict] = {}
    with open(table, newline="") as fh:
        for r in csv.DictReader(fh):
            d = wanted.get(r["WT_name"])
            if d is None:
                continue
            if r["mut_type"] == "wt":
                wild.setdefault(d, r)
                continue
            try:
                m = Mutation.parse(r["mut_type"])
            except ValueError:
                continue
            try:
                dg, ddg = float(r["dG_ML"]), float(r["ddG_ML"])
            except ValueError:
                continue
            if not DG_RANGE[0] <= dg <= DG_RANGE[1]:
                continue
            rows[d].append({"label": m.label, "position": m.position, "wild": m.wild, "new": m.new,
                            "dG_ML": f"{dg:.4f}", "ddG_ML": f"{ddg:.4f}",
                            "dG_95CI": r.get("deltaG_95CI", "")})
    counts = {}
    for d, lst in rows.items():
        seen, kept = set(), []
        for r in lst:
            if r["label"] in seen:
                continue
            seen.add(r["label"])
            kept.append(r)
        kept.sort(key=lambda r: (r["position"], r["new"]))
        folder = os.path.join(out, d)
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "mutants.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(kept[0]))
            w.writeheader()
            w.writerows(kept)
        shutil.copyfile(os.path.join(MODELS, f"{d}.pdb"), os.path.join(folder, f"{d}.pdb"))
        wt = wild.get(d, {})
        with open(os.path.join(folder, "wild_type.txt"), "w") as fh:
            fh.write(f"sequence {wt.get('aa_seq', '')}\n")
            fh.write(f"dG_ML {wt.get('dG_ML', '')}\n")
            fh.write(f"cluster {wt.get('WT_cluster', '')}\n")
        counts[d] = len(kept)
        print(f"{d}: {len(kept)} single mutants kept at "
              f"{len(set(r['position'] for r in kept))} positions; wild type dG_ML {wt.get('dG_ML', '?')}")
    return counts


def build(domains=tuple(DOMAINS), out: str = OUT, built: str = BUILT,
          shard: int = 0, shards: int = 1) -> None:
    """Build the mutant structures of each domain, one after another in this process.

    With `shards` greater than one this process takes only every `shards`-th mutant, starting
    at `shard`, so that several processes started side by side share the work.  Separate
    processes are used rather than a pool of workers because a pool on Windows loses its
    children when the parent is interrupted, and a build of several thousand structures is
    long enough for that to matter.
    """
    os.environ.setdefault("OPENMM_CPU_THREADS", "1")
    from cmfqsar.mutants import Mutation, build_mutant, prepare_wild_type
    for d in domains:
        folder, dest = os.path.join(out, d), os.path.join(built, d)
        os.makedirs(dest, exist_ok=True)
        wt_pdb = os.path.join(dest, "wild_type.pdb")
        if not os.path.exists(wt_pdb):
            prepare_wild_type(os.path.join(folder, f"{d}.pdb"), wt_pdb).save(
                os.path.join(dest, "wild_type.npz"))
        with open(os.path.join(folder, "mutants.csv"), newline="") as fh:
            labels = [r["label"] for r in csv.DictReader(fh)]
        mine = labels[shard::shards]
        todo = [lab for lab in mine if not os.path.exists(os.path.join(dest, f"{lab}.npz"))]
        print(f"{d}: {len(labels)} mutants, {len(mine)} in this share, {len(todo)} to build",
              flush=True)
        t0, failed = time.time(), []
        for k, label in enumerate(todo, 1):
            try:
                build_mutant(wt_pdb, Mutation.parse(label)).save(
                    os.path.join(dest, f"{label}.npz"))
            except Exception as exc:                 # noqa: BLE001 - reported, not raised
                failed.append((label, f"{type(exc).__name__}: {exc}"))
            if k % 50 == 0 or k == len(todo):
                print(f"  {d} share {shard}: {k}/{len(todo)} after {time.time() - t0:.0f} s, "
                      f"{len(failed)} failed", flush=True)
        if failed:
            with open(os.path.join(dest, f"failed_{shard}.txt"), "w") as fh:
                for label, err in failed:
                    fh.write(f"{label}\t{err}\n")
            print(f"  failed: {', '.join(l for l, _ in failed[:10])}"
                  f"{' ...' if len(failed) > 10 else ''}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build", action="store_true", help="also build the mutant structures")
    p.add_argument("--domains", nargs="*", default=list(DOMAINS))
    p.add_argument("--shard", type=int, default=0, help="which share of the mutants to build")
    p.add_argument("--shards", type=int, default=1, help="how many processes share the work")
    a = p.parse_args(argv)
    if not a.build or not os.path.exists(os.path.join(OUT, a.domains[0], "mutants.csv")):
        select()
    if a.build:
        build(a.domains, shard=a.shard, shards=a.shards)


if __name__ == "__main__":
    main()
