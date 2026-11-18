#!/usr/bin/env python
"""
The glass transition temperatures of homopolymers from PolyMetriX, as a table the polymer
benchmark reads.

Source: S. Kunchapu and K. M. Jablonka, "Curated Glass Transition Temperature for Polymers",
Zenodo record 15789599 (the latest version of 15783761), licence CC BY 4.0; described in
Kunchapu and Jablonka, npj Computational Materials 11, 312 (2025),
doi:10.1038/s41524-025-01823-y.  The file holds 7,367 homopolymers, each a repeat unit written as
SMILES with two attachment points (the PSMILES), the curated Tg in kelvin, the polymer class, a
reliability class, and the 100 hierarchical descriptors of the PolyMetriX library (side chain,
backbone and full polymer).

What this script does: download the file into ``benchmarks/cache/polymer_tg/raw/`` and check its
MD5 checksum; refuse the rows the field description cannot represent, each with its reason; write
``examples/polymer_tg/tg.csv`` (identifier, PSMILES, Tg, class, reliability, source) and
``examples/polymer_tg/provenance.json``.  The PolyMetriX descriptors stay in the cached download,
which the benchmark reads for its reference models; they are 6 MB of numbers that the download
reproduces exactly.

Refused rows:

* an element outside those the Gasteiger charge scheme covers (``GASTEIGER_ELEMENTS`` in
  ``cmfqsar/fields.py``): one such atom leaves the whole molecule without an electrostatic field;
* a net formal charge different from zero: a repeat unit with a counter-ion that is not written;
* a repeat unit the builder refuses (``cmfqsar.polymer.parse_repeat_unit``).

Usage
-----
    python benchmarks/prepare_polymer_tg.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.fields import GASTEIGER_ELEMENTS  # noqa: E402
from cmfqsar.polymer import parse_repeat_unit  # noqa: E402

RDLogger.DisableLog("rdApp.*")

RECORD = "15789599"
FILE = "LAMALAB_CURATED_Tg_structured_polymerclass_with_embeddings.csv"
URL = f"https://zenodo.org/api/records/{RECORD}/files/{FILE}/content"
MD5 = "31400a5f499394e3a6e0658488a464bb"
CACHE = os.path.join(HERE, "cache", "polymer_tg", "raw")
RAW = os.path.join(CACHE, "tg.csv")
OUT = os.path.join(ROOT, "examples", "polymer_tg")
#: the PolyMetriX descriptors: the columns of the three hierarchical levels
FEATURE_PREFIXES = ("sidechainlevel.features.", "backbonelevel.features.",
                    "fullpolymerlevel.features.")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch() -> str:
    """The raw file, downloaded once and checked against the checksum Zenodo publishes."""
    if not os.path.isfile(RAW) or md5(RAW) != MD5:
        os.makedirs(CACHE, exist_ok=True)
        print(f"downloading {URL} ...", flush=True)
        urllib.request.urlretrieve(URL, RAW)
    got = md5(RAW)
    if got != MD5:
        raise SystemExit(f"{RAW}: MD5 {got}, expected {MD5}; the record has changed")
    return RAW


def read_raw():
    csv.field_size_limit(1 << 30)
    with open(fetch(), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def feature_columns(row) -> list:
    return [c for c in row if c.startswith(FEATURE_PREFIXES)]


def refusal(psmiles: str) -> str:
    """Why a repeat unit is left out, or an empty string."""
    try:
        parse_repeat_unit(psmiles)
    except ValueError as exc:
        return f"builder: {exc}"
    mol = Chem.MolFromSmiles(psmiles)
    other = sorted({a.GetSymbol() for a in mol.GetAtoms()
                    if a.GetAtomicNum() > 0 and a.GetSymbol() not in GASTEIGER_ELEMENTS})
    if other:
        return "element without Gasteiger charges: " + ", ".join(other)
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return "net formal charge"
    return ""


def main() -> int:
    rows = read_raw()
    kept, refused = [], Counter()
    for k, r in enumerate(rows):
        why = refusal(r["PSMILES"])
        if why:
            refused[why.split(":")[0] if why.startswith("builder") else why] += 1
            continue
        kept.append({"id": f"tg{k + 1:04d}", "psmiles": r["PSMILES"],
                     "tg_k": r["labels.Exp_Tg(K)"], "polymer_class": r["meta.polymer_class"],
                     "reliability": r["meta.reliability"], "source": r["meta.source"]})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "tg.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(kept)
    prov = {"record": f"https://doi.org/10.5281/zenodo.{RECORD}", "file": FILE, "md5": MD5,
            "licence": "CC BY 4.0",
            "paper": "Kunchapu and Jablonka, npj Comput. Mater. 11, 312 (2025), "
                     "doi:10.1038/s41524-025-01823-y",
            "rows": len(rows), "kept": len(kept), "refused": dict(sorted(refused.items())),
            "polymetrix_feature_columns": len(feature_columns(rows[0])),
            "classes": dict(Counter(r["polymer_class"] for r in kept).most_common())}
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    print(f"{len(rows)} rows, {len(kept)} kept; refused: {dict(refused)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
