#!/usr/bin/env python
"""
Two measured logP sets with one 3D structure per compound, for ``benchmarks/run_logp.py``.

The source is the SangsterLogP data set (T. Cirino, J. Sangster, L. Charochkina, I. Tetko;
Zenodo record 19387552, doi:10.5281/zenodo.19387552, licence CC BY 4.0), described in Cirino et
al., Scientific Data 2026, doi:10.1038/s41597-026-07357-2. Its spreadsheet has three sheets: the
main table (23,520 compounds), the Huuskonen set (1,870 compounds; J. Huuskonen et al., J. Chem.
Inf. Comput. Sci. 2000) and a prospective set (327 compounds).

Two files are written to ``examples/logp``:

* ``huuskonen.sdf`` -- the Huuskonen sheet, every compound that RDKit reads as one neutral
  organic molecule and can embed in 3D. It is the classic logP benchmark.
* ``sangster_p.sdf`` -- a random sample of class P of the main table. Class P holds the values
  that the original authors reported as logP, which the data descriptor calls the most reliable
  class; the other classes were measured as logD, and some of them were corrected to logP.
  Compounds of the Huuskonen sheet (by the first block of the InChIKey) and compounds with a
  molecular weight above 700 are left out before sampling, so the two files share no compound.

Each compound gets one conformer: RDKit ETKDGv3 with a fixed seed, then MMFF94
(:func:`cmfqsar.align.embed_3d`). logP is an equilibrium property and in principle an average
over conformers; one conformer is the simplest choice and the one the program makes by default.

Usage:
    python benchmarks/prepare_logp.py [--n 3000] [--workers 8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import Descriptors  # noqa: E402

RDLogger.DisableLog("rdApp.*")

URL = "https://zenodo.org/records/19387552/files/Datasets.xlsx?download=1"
CACHE = os.path.join(HERE, "cache", "logp")
OUT = os.path.join(ROOT, "examples", "logp")
ORGANIC = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
SEED = 0
_M = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch() -> str:
    path = os.path.join(CACHE, "Datasets.xlsx")
    if not os.path.isfile(path):
        os.makedirs(CACHE, exist_ok=True)
        log(f"downloading {URL}")
        urllib.request.urlretrieve(URL, path)
    return path


def md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def read_sheets(path: str):
    """Every sheet of the workbook as a list of dicts keyed by the header row.

    The standard library reads it: an .xlsx file is a zip of XML files, and the program does not
    depend on a spreadsheet package.
    """
    z = zipfile.ZipFile(path)
    strings = ["".join(t.text or "" for t in si.iter(_M + "t"))
               for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).find(_M + "sheets")]
    out = {}
    for k, title in enumerate(names, 1):
        rows = []
        for r in ET.fromstring(z.read(f"xl/worksheets/sheet{k}.xml")).find(_M + "sheetData"):
            d = {}
            for c in r:
                v = c.find(_M + "v")
                col = "".join(ch for ch in c.get("r") if ch.isalpha())
                d[col] = None if v is None else (strings[int(v.text)] if c.get("t") == "s"
                                                 else v.text)
            rows.append(d)
        head = rows[0]
        out[title] = [{head[c]: r.get(c) for c in head if head[c]} for r in rows[1:]]
    return out


def clean(smiles: str):
    """One neutral organic molecule, or (None, the reason it was refused)."""
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return None, "RDKit cannot read the SMILES"
    if len(Chem.GetMolFrags(m)) > 1:
        return None, "more than one component"
    if any(a.GetSymbol() not in ORGANIC for a in m.GetAtoms()):
        return None, "an element outside the organic set"
    if sum(a.GetFormalCharge() for a in m.GetAtoms()):
        return None, "a net charge"
    for a in m.GetAtoms():
        # a nitro group, an N-oxide or an azide is written with a +1 and a -1 on bonded atoms
        # and is a neutral group; a charge with no opposite charge beside it is a zwitterion
        q = a.GetFormalCharge()
        if q and not any(n.GetFormalCharge() * q < 0 for n in a.GetNeighbors()):
            return None, "a zwitterion"
    if m.GetNumHeavyAtoms() < 2:
        return None, "fewer than two heavy atoms"
    return Chem.RemoveHs(m), None


def value(text):
    """The logP of a cell, or None. A few cells of the main table read "5.74 +- 0.22"."""
    try:
        return float(str(text).split("+-")[0])
    except (TypeError, ValueError):
        return None


def skeleton(m) -> str:
    """The first block of the InChIKey: the same compound whatever its stereo labels."""
    return Chem.MolToInchiKey(m).split("-")[0]


def _embed(args):
    from cmfqsar.align import embed_3d
    smi, seed = args
    try:
        return Chem.MolToMolBlock(embed_3d(Chem.MolFromSmiles(smi), seed=seed))
    except Exception as exc:                                    # noqa: BLE001
        return f"ERROR {exc}"


def embed_all(records, workers: int):
    with ProcessPoolExecutor(max_workers=workers) as ex:
        blocks = list(ex.map(_embed, [(r["smiles"], 42) for r in records], chunksize=16))
    kept, failed = [], []
    for r, b in zip(records, blocks):
        if b.startswith("ERROR"):
            failed.append(r["name"])
            continue
        m = Chem.MolFromMolBlock(b, removeHs=False)
        for k, v in r.items():
            m.SetProp(k, str(v))
        m.SetProp("_Name", r["name"])
        kept.append(m)
    return kept, failed


def write(mols, path):
    w = Chem.SDWriter(path)
    for m in mols:
        w.write(m)
    w.close()


def count(d: dict, reason: str) -> None:
    d[reason] = d.get(reason, 0) + 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="compounds sampled from class P")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    path = fetch()
    sheets = read_sheets(path)
    main_name = [k for k in sheets if k.startswith("SangsterLogP")][0]
    huus_name = [k for k in sheets if k.startswith("Huuskonen")][0]
    prov = {"source": "SangsterLogP, Zenodo record 19387552 (doi:10.5281/zenodo.19387552), "
                      "CC BY 4.0; Cirino et al., Scientific Data 2026, "
                      "doi:10.1038/s41597-026-07357-2",
            "url": URL, "md5": md5(path), "bytes": os.path.getsize(path),
            "sheets": {k: len(v) for k, v in sheets.items()}}

    # the Huuskonen sheet
    huus, why, seen, dup = [], {}, set(), 0
    for r in sheets[huus_name]:
        if not r.get("ID") and not r.get("SMILES"):
            count(why, "an empty row")
            continue
        if value(r.get("logP")) is None:
            # 21 rows: a comma in the compound name moved the rest of the name into the logP
            # column when the sheet was made, and the value itself is not in the row
            count(why, "no logP value in the row (a comma in the name shifted the columns)")
            continue
        m, reason = clean(r["SMILES"])
        if m is None:
            count(why, reason)
            continue
        key = skeleton(m)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        huus.append({"name": r["ID"], "smiles": Chem.MolToSmiles(m), "logP": value(r["logP"]),
                     "compound": r.get("Name") or "", "source": r.get("Source") or "",
                     "inchikey14": key})
    prov["huuskonen"] = {"read": len(sheets[huus_name]), "refused": why,
                         "duplicate_skeletons": dup, "kept_before_embedding": len(huus)}

    # class P of the main table, without the Huuskonen compounds
    pool, why, overlap, pool_keys = [], {}, 0, set()
    for r in sheets[main_name]:
        if r.get("Ionisation-based class") != "P":
            continue
        if value(r.get("logP")) is None:
            count(why, "no logP value")
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
        if key in pool_keys:
            count(why, "duplicate skeleton")
            continue
        pool_keys.add(key)
        pool.append({"name": r["ID"], "smiles": Chem.MolToSmiles(m), "logP": value(r["logP"]),
                     "compound": r.get("Name") or "", "method": r.get("Exp Method") or "",
                     "recommended": r.get("Recommended") or "", "source": r.get("Source") or "",
                     "inchikey14": key})
    rng = np.random.RandomState(SEED)
    pick = sorted(rng.choice(len(pool), size=min(a.n, len(pool)), replace=False))
    sample = [pool[i] for i in pick]
    prov["sangster_p"] = {"class_P_rows": sum(1 for r in sheets[main_name]
                                              if r.get("Ionisation-based class") == "P"),
                          "refused": why, "in_huuskonen": overlap, "pool": len(pool),
                          "sampled": len(sample), "seed": SEED}
    log(f"Huuskonen {len(huus)}, class P pool {len(pool)} ({overlap} in Huuskonen), "
        f"sampled {len(sample)}")

    os.makedirs(OUT, exist_ok=True)
    for tag, recs in (("huuskonen", huus), ("sangster_p", sample)):
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
