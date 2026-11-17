#!/usr/bin/env python
"""
Two measured boiling point sets with one 3D structure per compound, for
``benchmarks/run_boiling.py``.

The boiling point is the normal boiling point, in degrees C.

Sources
-------
Both sets come from the data of the OPERA models (K. Mansouri, C. M. Grulke, R. S. Judson,
A. J. Williams, J. Cheminform. 10 (2018) 10, doi:10.1186/s13321-018-0263-1), which are kept in
the file ``OPERA_Data.zip`` of the repository github.com/kmansouri/OPERA (MIT licence).

* **PHYSPROP** (``BP_QR.sdf``): 5,434 boiling points of the PHYSPROP database, curated for the
  OPERA paper, with "QSAR-ready" structures (salts and solvents removed) and the paper's split
  into a training part (75 %) and a test part (25 %). The paper reports, for its boiling point
  model, Q2 0.93 and RMSE 22.46 C in fivefold cross-validation on the training part and R2 0.93
  and RMSE 22.08 C on the test part.
* **eChemPortal** (``BP_2.9_update_checked_targeted.xlsx``, sheet ``Checked_BP``): the 2,426
  boiling points that version 2.9 of OPERA added, with the curators' checked value
  (``corrected_BP_C``) and status. Only the rows with eChemPortal among their sources are kept:
  values from the registration dossiers of the OECD eChemPortal. Rows whose only sources are
  web catalogues (LookChem, ChemicalBook) are left out, because such pages often give a
  predicted value. Rows the curators flagged or left for a manual review are left out
  as well.

Two files are written to ``examples/boiling``:

* ``physprop.sdf`` -- every PHYSPROP compound that RDKit reads as one neutral organic molecule
  and can embed in 3D, with ``split`` = ``train`` or ``test`` as in the OPERA paper.
* ``echemportal.sdf`` -- the eChemPortal compounds, without the PHYSPROP compounds (by the first
  block of the InChIKey).

Compounds above a molecular weight of 700 are left out of both. The rules for refusing a
compound and the 3D structures (ETKDGv3 with seed 42, then MMFF94) are those of
``benchmarks/prepare_logp.py``, imported from it.

Usage:
    python benchmarks/prepare_boiling.py [--workers 8]
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

CACHE = os.path.join(HERE, "cache", "boiling")
OUT = os.path.join(ROOT, "examples", "boiling")
URL = "https://github.com/kmansouri/OPERA/raw/master/OPERA_Data.zip"
MEMBERS = ("BP_QR.sdf", "BP_2.9_update_checked_targeted.xlsx")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch() -> dict:
    """The two files of the archive, extracted into the cache; the archive is downloaded once."""
    folder = os.path.join(CACHE, "opera")
    os.makedirs(folder, exist_ok=True)
    archive = os.path.join(folder, "OPERA_Data.zip")
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


def keep(m, why: dict) -> bool:
    if Descriptors.MolWt(m) > 700:
        count(why, "molecular weight above 700")
        return False
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    files = fetch()
    prov = {"source": "OPERA data (Mansouri et al., J. Cheminform. 10 (2018) 10), "
                      "github.com/kmansouri/OPERA, MIT licence",
            "url": URL, "archive_md5": md5(files["archive"]),
            "files_md5": {m: md5(files[m]) for m in MEMBERS}}

    # PHYSPROP, with the training and test parts of the OPERA paper
    phys, why, seen, dup = [], {}, set(), 0
    sup = Chem.SDMolSupplier(files["BP_QR.sdf"])
    n_read = 0
    for rec in sup:
        n_read += 1
        if rec is None:
            count(why, "RDKit cannot read the record")
            continue
        bp = number(rec.GetProp("BP"))
        if bp is None:
            count(why, "no boiling point")
            continue
        m, reason = clean(rec.GetProp("Canonical_QSARr"))
        if m is None:
            count(why, reason)
            continue
        if not keep(m, why):
            continue
        key = skeleton(m)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        phys.append({"name": f"physprop-{rec.GetProp('ChemID')}", "smiles": Chem.MolToSmiles(m),
                     "bp": bp,
                     "split": "train" if rec.GetProp("Tr_1_Tst_0").strip() == "1" else "test",
                     "compound": rec.GetProp("preferred_name"), "cas": rec.GetProp("CAS"),
                     "dtxsid": rec.GetProp("dsstox_substance_id"), "inchikey14": key})
    prov["physprop"] = {"read": n_read, "refused": why, "duplicate_skeletons": dup,
                        "kept_before_embedding": len(phys),
                        "train": sum(r["split"] == "train" for r in phys),
                        "test": sum(r["split"] == "test" for r in phys)}

    # eChemPortal values of the OPERA 2.9 update
    rows = read_sheets(files["BP_2.9_update_checked_targeted.xlsx"])["Checked_BP"]
    echem, why, overlap, keys, status = [], {}, 0, set(), {}
    for r in rows:
        src = r.get("pub_source_name") or ""
        if "eChemPortal" not in src:
            count(why, "no eChemPortal value among the sources")
            continue
        st = r.get("curation_status") or ""
        count(status, st)
        if "manual_review" in st or st.startswith("flag_"):
            # "flag_no_clean_finite_BP_decomposition_or_NA" and the review statuses
            count(why, "flagged or left for a manual review by the curators")
            continue
        bp = number(r.get("corrected_BP_C"))
        if bp is None:
            count(why, "no checked value")
            continue
        m, reason = clean(r.get("Canonical_QSARr"))
        if m is None:
            count(why, reason)
            continue
        if not keep(m, why):
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
                      "smiles": Chem.MolToSmiles(m), "bp": bp,
                      "compound": r.get("PREFERRED_NAME") or "", "cas": r.get("CASRN") or "",
                      "sources": src, "curation_status": st, "inchikey14": key})
    prov["echemportal"] = {"read": len(rows), "refused": why, "in_physprop": overlap,
                           "curation_status_of_echemportal_rows": status,
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
