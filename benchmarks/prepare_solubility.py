#!/usr/bin/env python
"""
Two measured water solubility sets with one 3D structure per compound, for
``benchmarks/run_solubility.py``.

Solubility is given as logS, the decimal logarithm of the solubility in mol/L, near 25 C.

Sources
-------
* **The Delaney set** (J. S. Delaney, J. Chem. Inf. Comput. Sci. 44 (2004) 1000-1005), as
  MoleculeNet distributes it (``delaney-processed.csv``, 1,128 compounds, from
  https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv). It is the
  classic solubility benchmark, often called "ESOL" after the equation of the paper. The file
  also holds the value the published ESOL equation gives for each compound. No licence is
  stated with the file. AqSolDB, below, contains the same set (its data set G) under CC0.
* **AqSolDB** (M. C. Sorkun, A. Khetan, S. Er, Sci. Data 6 (2019) 143,
  doi:10.1038/s41597-019-0151-1; Harvard Dataverse doi:10.7910/DVN/OVHAW8, licence CC0 1.0),
  9,982 compounds merged from nine public sets. The paper sorts the compounds into reliability
  groups. G1: found in one source only. G2 and G3: found twice, with a standard deviation of the
  two values of at least 0.5 (G2) or below 0.5 log units (G3). G4 and G5: found three times or
  more, standard deviation at least 0.5 (G4) or below 0.5 (G5). Only G3 and G5 are used here:
  the compounds whose solubility was measured more than once with agreeing results.

Two files are written to ``examples/solubility``:

* ``delaney.sdf`` -- every compound of the Delaney file that RDKit reads as one neutral organic
  molecule and can embed in 3D.
* ``aqsoldb_reliable.sdf`` -- the AqSolDB compounds of groups G3 and G5, without the compounds of
  the Delaney file (by the first block of the InChIKey) and without compounds above a molecular
  weight of 700, so that the two files share no compound.

The rules for refusing a compound and the 3D structures (ETKDGv3 with seed 42, then MMFF94) are
those of ``benchmarks/prepare_logp.py``, imported from it.

Usage:
    python benchmarks/prepare_solubility.py [--workers 8]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit.Chem import Descriptors  # noqa: E402
from rdkit import Chem  # noqa: E402

from prepare_logp import clean, count, embed_all, md5, skeleton, write  # noqa: E402

CACHE = os.path.join(HERE, "cache", "solubility")
OUT = os.path.join(ROOT, "examples", "solubility")
URLS = {"delaney.csv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
                       "delaney-processed.csv",
        "aqsoldb.csv": "https://dataverse.harvard.edu/api/access/datafile/3407241?format=original"}
#: the checksum Harvard Dataverse publishes for the AqSolDB file
AQSOLDB_MD5 = "02f732e52de921b8adaf4ef03cf747bf"
RELIABLE = ("G3", "G5")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch(name: str) -> str:
    path = os.path.join(CACHE, name)
    if not os.path.isfile(path):
        os.makedirs(CACHE, exist_ok=True)
        log(f"downloading {URLS[name]}")
        urllib.request.urlretrieve(URLS[name], path)
    return path


def read_csv(path: str):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    d_path, q_path = fetch("delaney.csv"), fetch("aqsoldb.csv")
    if md5(q_path) != AQSOLDB_MD5:
        raise SystemExit(f"{q_path}: the checksum is not the one Harvard Dataverse publishes")
    prov = {"delaney": {"source": "J. S. Delaney, J. Chem. Inf. Comput. Sci. 44 (2004) "
                                  "1000-1005, as distributed by MoleculeNet; no licence stated",
                        "url": URLS["delaney.csv"], "md5": md5(d_path)},
            "aqsoldb_reliable": {"source": "AqSolDB, Sorkun, Khetan and Er, Sci. Data 6 (2019) "
                                           "143; Harvard Dataverse doi:10.7910/DVN/OVHAW8, "
                                           "CC0 1.0", "url": URLS["aqsoldb.csv"],
                                 "md5": md5(q_path), "groups": list(RELIABLE)}}

    # the Delaney set
    rows = read_csv(d_path)
    dela, why, seen, dup = [], {}, set(), 0
    esol_published = []
    for r in rows:
        m, reason = clean(r["smiles"].strip())
        if m is None:
            count(why, reason)
            continue
        key = skeleton(m)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        rec = {"name": f"delaney-{len(dela) + 1}", "smiles": Chem.MolToSmiles(m),
               "logS": float(r["measured log solubility in mols per litre"]),
               "compound": r["Compound ID"].strip(),
               "esol_published": float(r["ESOL predicted log solubility in mols per litre"]),
               "inchikey14": key}
        dela.append(rec)
    prov["delaney"].update({"read": len(rows), "refused": why, "duplicate_skeletons": dup,
                            "kept_before_embedding": len(dela)})

    # AqSolDB, groups G3 and G5, without the Delaney compounds
    rows = read_csv(q_path)
    pool, why, overlap, keys = [], {}, 0, set()
    groups: dict = {}
    for r in rows:
        count(groups, r["Group"])
        if r["Group"] not in RELIABLE:
            continue
        m, reason = clean(r["SMILES"])
        if m is None:
            count(why, reason)
            continue
        if Descriptors.MolWt(m) > 700:
            count(why, "molecular weight above 700")
            continue
        key = skeleton(m)
        if key in seen:
            overlap += 1
            continue
        if key in keys:
            # stereoisomers of one skeleton; the first one is kept
            count(why, "duplicate skeleton")
            continue
        keys.add(key)
        pool.append({"name": r["ID"], "smiles": Chem.MolToSmiles(m),
                     "logS": float(r["Solubility"]), "compound": r["Name"],
                     "sd": float(r["SD"]), "occurrences": int(r["Ocurrences"]),
                     "group": r["Group"], "inchikey14": key})
    prov["aqsoldb_reliable"].update({"read": len(rows), "groups_in_file": groups,
                                     "in_reliable_groups": sum(groups.get(g, 0)
                                                               for g in RELIABLE),
                                     "refused": why, "in_delaney": overlap,
                                     "kept_before_embedding": len(pool)})
    log(f"Delaney {len(dela)}, AqSolDB G3+G5 {len(pool)} ({overlap} in Delaney)")

    os.makedirs(OUT, exist_ok=True)
    for tag, recs in (("delaney", dela), ("aqsoldb_reliable", pool)):
        t0 = time.time()
        mols, failed = embed_all(recs, a.workers)
        write(mols, os.path.join(OUT, f"{tag}.sdf"))
        prov[tag]["embedded"] = len(mols)
        prov[tag]["embedding_failed"] = failed
        log(f"{tag}: {len(mols)} embedded, {len(failed)} failed ({time.time() - t0:.0f} s)")
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
