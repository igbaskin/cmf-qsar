#!/usr/bin/env python
"""
Two measured vapour pressure sets with one 3D structure per compound, for
``benchmarks/run_vapour.py``.

The target is logVP, the decimal logarithm of the vapour pressure in mmHg, near 25 C.

Sources
-------
Both sets come from the data of the OPERA models (K. Mansouri, C. M. Grulke, R. S. Judson,
A. J. Williams, J. Cheminform. 10 (2018) 10, doi:10.1186/s13321-018-0263-1), kept in the file
``OPERA_Data.zip`` of the repository github.com/kmansouri/OPERA (MIT licence), as for the boiling
points of ``benchmarks/prepare_boiling.py``.

* **PHYSPROP** (``VP_QR.sdf``): 2,819 vapour pressures of the PHYSPROP database curated for the
  OPERA paper, with "QSAR-ready" structures and the paper's split into a training part and a
  test part. Most values are at 25 C (2,368) and most of the rest at 20 C (287); the record
  gives the temperature. 108 records belong to neither part and carry no temperature; they are
  left out. The paper reports, for its vapour pressure model, Q2 0.91 and RMSE 1.08 log units in
  fivefold cross-validation on the training part and R2 0.92 and RMSE 1.00 on the test part.
* **eChemPortal** (``VP_2.9_update_checked_chemicalbook_log10mmHg.xlsx``, sheet
  ``VP_Checked``): the 3,945 vapour pressures that version 2.9 of OPERA added, with the curators'
  value (``corrected_log10_mmHg``) and status. Only the rows with eChemPortal among their
  sources are kept: values from the registration dossiers of the OECD eChemPortal. A row is kept
  only when its status says that the value was retained or confirmed (``STATUS_KEPT``) and the
  value is a single number; rows marked as a range, a lower bound, a value at another
  temperature, or implausible are left out.

Two files are written to ``examples/vapour``:

* ``physprop.sdf`` -- every PHYSPROP compound of the two parts that RDKit reads as one neutral
  organic molecule and can embed in 3D, with ``split`` = ``train`` or ``test``.
* ``echemportal.sdf`` -- the eChemPortal compounds, without the PHYSPROP compounds (by the first
  block of the InChIKey).

Compounds above a molecular weight of 700 are left out of both. The rules for refusing a
compound and the 3D structures (ETKDGv3 with seed 42, then MMFF94) are those of
``benchmarks/prepare_logp.py``, imported from it.

Usage:
    python benchmarks/prepare_vapour.py [--workers 8]
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
from rdkit.Chem import Descriptors  # noqa: E402

from prepare_logp import clean, count, embed_all, md5, read_sheets, skeleton, write  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CACHE = os.path.join(HERE, "cache", "vapour")
OUT = os.path.join(ROOT, "examples", "vapour")
URL = "https://github.com/kmansouri/OPERA/raw/master/OPERA_Data.zip"
#: the archive the boiling point preparation downloads, used when it is there
SHARED = os.path.join(HERE, "cache", "boiling", "opera", "OPERA_Data.zip")
MEMBERS = ("VP_QR.sdf", "VP_2.9_update_checked_chemicalbook_log10mmHg.xlsx")
#: the statuses of a row whose value was kept as it was or confirmed by the curators
STATUS_KEPT = ("not_checked_retained_original", "chemicalbook_confirmed_or_minor_rounding",
               "chemicalbook_no_vp_found_retained_original", "chemicalbook_numeric_correction")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch() -> dict:
    folder = os.path.join(CACHE, "opera")
    os.makedirs(folder, exist_ok=True)
    archive = SHARED if os.path.isfile(SHARED) else os.path.join(folder, "OPERA_Data.zip")
    if not os.path.isfile(archive):
        log(f"downloading {URL}")
        urllib.request.urlretrieve(URL, archive)
    out = {"archive": archive}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            base = name.split("/")[-1]
            if base in MEMBERS:
                path = os.path.join(folder, base)
                if not os.path.isfile(path):
                    with z.open(name) as src, open(path, "wb") as dst:
                        dst.write(src.read())
                out[base] = path
    missing = [m for m in MEMBERS if m not in out]
    if missing:
        raise SystemExit(f"{archive}: missing {missing}")
    return out


def number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    files = fetch()
    prov = {"source": "OPERA data (Mansouri et al., J. Cheminform. 10 (2018) 10), "
                      "github.com/kmansouri/OPERA, MIT licence",
            "url": URL, "archive_md5": md5(files["archive"]),
            "files_md5": {m: md5(files[m]) for m in MEMBERS}, "unit": "log10 mmHg"}

    # PHYSPROP, the two parts of the OPERA paper
    phys, why, seen, dup, temps = [], {}, set(), 0, {}
    n_read = 0
    for rec in Chem.SDMolSupplier(files["VP_QR.sdf"]):
        n_read += 1
        if rec is None:
            count(why, "RDKit cannot read the record")
            continue
        part = rec.GetProp("Tr_1_Tst_0").strip()
        if part not in ("0", "1"):
            count(why, "in neither part of the OPERA split, and no temperature recorded")
            continue
        vp = number(rec.GetProp("LogVP"))
        if vp is None:
            count(why, "no vapour pressure")
            continue
        m, reason = clean(rec.GetProp("Canonical_QSARr"))
        if m is None:
            count(why, reason)
            continue
        if Descriptors.MolWt(m) > 700:
            count(why, "molecular weight above 700")
            continue
        key = skeleton(m)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        t = rec.GetProp("VP Temperature").strip()
        count(temps, t)
        phys.append({"name": f"physprop-{rec.GetProp('ChemID')}", "smiles": Chem.MolToSmiles(m),
                     "logVP": vp, "split": "train" if part == "1" else "test",
                     "temperature": t, "compound": rec.GetProp("preferred_name"),
                     "cas": rec.GetProp("CAS"), "reference": rec.GetProp("VP Reference"),
                     "inchikey14": key})
    prov["physprop"] = {"read": n_read, "refused": why, "duplicate_skeletons": dup,
                        "temperatures_C": temps, "kept_before_embedding": len(phys),
                        "train": sum(r["split"] == "train" for r in phys),
                        "test": sum(r["split"] == "test" for r in phys)}

    # eChemPortal values of the OPERA 2.9 update
    rows = read_sheets(files["VP_2.9_update_checked_chemicalbook_log10mmHg.xlsx"])["VP_Checked"]
    echem, why, overlap, keys = [], {}, 0, set()
    for r in rows:
        src = r.get("pub_source_name") or ""
        if "eChemPortal" not in src:
            count(why, "no eChemPortal value among the sources")
            continue
        st = r.get("curation_status") or ""
        if st not in STATUS_KEPT:
            count(why, f"curation status {st}")
            continue
        vp = number(r.get("corrected_log10_mmHg"))
        if vp is None:
            count(why, "the checked value is not a single number")
            continue
        m, reason = clean(r.get("Canonical_QSARr"))
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
            count(why, "duplicate skeleton")
            continue
        keys.add(key)
        echem.append({"name": f"echem-{r.get('DSSTOX_SUBSTANCE_ID')}",
                      "smiles": Chem.MolToSmiles(m), "logVP": vp,
                      "compound": r.get("PREFERRED_NAME") or "", "cas": r.get("CASRN") or "",
                      "sources": src, "curation_status": st, "inchikey14": key})
    prov["echemportal"] = {"read": len(rows), "refused": why, "in_physprop": overlap,
                           "kept_before_embedding": len(echem)}
    log(f"PHYSPROP {len(phys)} ({prov['physprop']['train']} train, "
        f"{prov['physprop']['test']} test), eChemPortal {len(echem)} ({overlap} in PHYSPROP)")

    os.makedirs(OUT, exist_ok=True)
    for tag, recs in (("physprop", phys), ("echemportal", echem)):
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
