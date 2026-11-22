#!/usr/bin/env python
"""
A measured soil sorption set with one 3D structure per compound, for ``benchmarks/run_koc.py``.

Koc is the soil organic carbon-water partition coefficient: the concentration of a compound
sorbed to soil, per unit mass of soil organic carbon, divided by its concentration in the soil
water at equilibrium, in L/kg. The target is its decimal logarithm, logKoc. Dividing by the
organic carbon content removes most of the dependence on the individual soil, so logKoc is
close to a property of the molecule alone.

Source
------
The data of the OPERA models (K. Mansouri, C. M. Grulke, R. S. Judson, A. J. Williams,
J. Cheminform. 10 (2018) 10, doi:10.1186/s13321-018-0263-1), which are kept in the file
``OPERA_Data.zip`` of the repository github.com/kmansouri/OPERA (MIT licence). The member
``KOC_QR.sdf`` holds 728 logKoc values of the PHYSPROP database, curated for the OPERA paper,
with "QSAR-ready" structures (salts and solvents removed) and the paper's split into a training
part (75 %) and a test part (25 %). The model report of OPERA (QMRF Q17-26-0017, in
``QMRFs.zip`` of the same repository) gives, for its model with 12 descriptors: R2 0.81 and
RMSE 0.54 on the training part, Q2 0.81 and RMSE 0.55 in fivefold cross-validation, and R2 0.71
and RMSE 0.61 on the test part.

The download is pinned to one commit of the repository, so the file cannot change without
notice.

One file is written to ``examples/koc``:

* ``koc.sdf`` -- every compound that RDKit reads as one neutral organic molecule and can embed
  in 3D, with ``split`` = ``train`` or ``test`` as in the OPERA paper. A second copy of a
  structure (the same first block of the InChIKey) is left out.

The rules for refusing a compound and the 3D structures (ETKDGv3 with seed 42, then MMFF94) are
those of ``benchmarks/prepare_logp.py``, imported from it.

Usage:
    python benchmarks/prepare_koc.py [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402

from prepare_logp import clean, count, embed_all, md5, skeleton, write  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CACHE = os.path.join(HERE, "cache", "koc")
OUT = os.path.join(ROOT, "examples", "koc")
COMMIT = "01b07c769dd86eadb71843e057783a64a364c885"
URL = f"https://raw.githubusercontent.com/kmansouri/OPERA/{COMMIT}/OPERA_Data.zip"
MEMBER = "KOC_QR.sdf"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch() -> dict:
    """The Koc file of the archive, extracted into the cache; the archive is downloaded once."""
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, MEMBER)
    archive = os.path.join(CACHE, "OPERA_Data.zip")
    if not os.path.isfile(path):
        if not os.path.isfile(archive):
            log(f"downloading {URL}")
            urllib.request.urlretrieve(URL, archive)
        with zipfile.ZipFile(archive) as z:
            with z.open(MEMBER) as src, open(path, "wb") as dst:
                dst.write(src.read())
    return {"archive": archive if os.path.isfile(archive) else None, MEMBER: path}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    files = fetch()
    prov = {"source": "OPERA data (Mansouri et al., J. Cheminform. 10 (2018) 10), "
                      "github.com/kmansouri/OPERA, MIT licence",
            "url": URL, "commit": COMMIT, "files_md5": {MEMBER: md5(files[MEMBER])}}
    if files["archive"]:
        prov["archive_md5"] = md5(files["archive"])

    recs, why, seen, dup, n_read = [], {}, set(), [], 0
    for rec in Chem.SDMolSupplier(files[MEMBER]):
        n_read += 1
        if rec is None:
            count(why, "RDKit cannot read the record")
            continue
        try:
            koc = float(rec.GetProp("LogKOC"))
        except (KeyError, ValueError):
            count(why, "no logKoc value")
            continue
        m, reason = clean(rec.GetProp("Canonical_QSARr"))
        if m is None:
            count(why, reason)
            continue
        key = skeleton(m)
        name = f"koc-{rec.GetProp('ChemID')}"
        if key in seen:
            dup.append(name)
            continue
        seen.add(key)
        recs.append({"name": name, "smiles": Chem.MolToSmiles(m), "logKoc": koc,
                     "split": "train" if rec.GetProp("Tr_1_Tst_0").strip() == "1" else "test",
                     "compound": rec.GetProp("preferred_name"), "cas": rec.GetProp("CAS"),
                     "dtxsid": rec.GetProp("dsstox_substance_id"),
                     "reference": rec.GetProp("KocRef"), "inchikey14": key})
    prov["koc"] = {"read": n_read, "refused": why, "duplicate_skeletons": dup,
                   "kept_before_embedding": len(recs),
                   "train": sum(r["split"] == "train" for r in recs),
                   "test": sum(r["split"] == "test" for r in recs)}
    log(f"{len(recs)} of {n_read} records kept ({prov['koc']['train']} train, "
        f"{prov['koc']['test']} test); refused {why}; {len(dup)} second copies")

    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    mols, failed = embed_all(recs, a.workers)
    write(mols, os.path.join(OUT, "koc.sdf"))
    prov["koc"]["embedded"] = len(mols)
    prov["koc"]["embedding_failed"] = failed
    prov["koc"]["train_embedded"] = sum(m.GetProp("split") == "train" for m in mols)
    prov["koc"]["test_embedded"] = sum(m.GetProp("split") == "test" for m in mols)
    log(f"{len(mols)} embedded, {len(failed)} failed ({time.time() - t0:.0f} s)")
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
