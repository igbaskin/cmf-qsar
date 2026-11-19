#!/usr/bin/env python
"""
Two polymer properties that depend on how chains pack, as tables the PL2 benchmark reads.

**Density from molecular dynamics.**  RadonPy PI1070 (Hayashi et al., npj Comput. Mater. 8, 222
(2022)): 1,077 amorphous homopolymers, each simulated by the same all-atom protocol (GAFF2, ten
chains of about 1,000 atoms, 300 K, 1 atm), with the density and many other properties.  The file
is ``data/PI1070.csv`` of github.com/RadonPy/RadonPy at commit 1387bc8, BSD 3-Clause licence.
The polymer class is the number RadonPy gives (1 to 21).

**Gas permeability, measured.**  The permeability of 786 polymers to CH4, CO2, H2, N2 and O2 in
Barrer, released with torch-molecule (MIT licence; github.com/liugangcode/torch-molecule) and read
here from its copy on Hugging Face, ``liuganghuggingface/moldata-polymer_gas_permeability`` at
revision 2492824, through the dataset server, which returns the rows as JSON.  The target is
log10 of the permeability.

What this script does: download both into ``benchmarks/cache/polymer_packing/raw/`` (the RadonPy
file is checked against its MD5 checksum, the Hugging Face revision against the one above); refuse
the repeat units the field description cannot represent, each with its reason; write
``examples/polymer_packing/pi1070.csv``, ``permeability.csv`` and ``permeability_pl6.csv``, the two
licence texts and ``provenance.json``.

Refused rows of ``pi1070.csv`` and ``permeability.csv`` (plan item PL2): a repeat unit the builder
refuses (``cmfqsar.polymer.parse_repeat_unit``) or one with four attachment points, which leaves
out ladder polymers such as the polymers of intrinsic microporosity; an element outside the
Gasteiger scheme; a net formal charge.  These two tables keep the SMILES as the sources give them.

**The table of plan item PL6** (``permeability_pl6.csv``).  The same permeability rows, with two
changes.

* *Ladder polymers are kept.*  The builder joins four attachment points since PL6.  Four of the 31
  are refused: a junction ring of fewer than five atoms (perm558); a Troger's base written with two
  bridges per unit, so that the junction would be a ring of four nitrogens and four carbons
  (perm054); and two rows that repeat the five values of another row (perm094 repeats perm499,
  written with a broken ring; perm618 is a regioisomer of perm005).
* *Two faults of the source are repaired.*  In 99 rows (115 rings) a benzene ring is written as a
  cyclohexa-1,3-diene: two adjacent ring carbons carry a hydrogen and no double bond, as in a
  fluorene written ``C3C=CC=CC3C3C=CC=CC32``.  No polymer in these papers has such a ring; it is a
  benzene ring whose aromatic bonds were lost.  The repair makes the bond between the two carbons
  double, which makes the ring aromatic.  In 12 rows (24 atoms) the carbonyl carbon of an imide
  or an ester carries ``=CH2`` in place of ``=O`` (``c(=C)n``, ``OC(=C)C``); the repair makes it an
  oxygen.  The column ``source_psmiles`` keeps the
  SMILES of the source, and ``repair`` says what was changed.  The density and glass transition
  tables have neither fault.

Not used: RadonPy's van der Waals volume per repeat unit, which would give the packing
coefficient k = V_vdW rho N_A / M.  It is above 1 -- the atoms filling more than the whole volume --
for several nylons (1.18 and 1.20), so the volume itself is wrong there.

Usage
-----
    python benchmarks/prepare_polymer_packing.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
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

RADONPY_COMMIT = "1387bc882feb9677a7741f4cce462110918d7761"
RADONPY_URL = f"https://raw.githubusercontent.com/RadonPy/RadonPy/{RADONPY_COMMIT}/data/PI1070.csv"
RADONPY_LICENSE = f"https://raw.githubusercontent.com/RadonPy/RadonPy/{RADONPY_COMMIT}/LICENSE"
RADONPY_MD5 = "4148919e3491461818a25b81bfabd5c6"
HF_DATASET = "liuganghuggingface/moldata-polymer_gas_permeability"
HF_REVISION = "24928245e5168b5b4c71d7579bda16efe80f6c3a"
HF_ROWS = 786
TORCH_MOLECULE_LICENSE = "https://raw.githubusercontent.com/liugangcode/torch-molecule/HEAD/LICENSE"
GASES = ("CH4", "CO2", "H2", "N2", "O2")
CACHE = os.path.join(HERE, "cache", "polymer_packing", "raw")
OUT = os.path.join(ROOT, "examples", "polymer_packing")


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: str) -> str:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def radonpy_rows():
    path = fetch(RADONPY_URL, os.path.join(CACHE, "PI1070.csv"))
    got = md5(path)
    if got != RADONPY_MD5:
        raise SystemExit(f"{path}: MD5 {got}, expected {RADONPY_MD5}")
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def permeability_rows():
    """The 786 rows as JSON, cached; refused if the dataset has moved past the pinned revision."""
    path = os.path.join(CACHE, "permeability_rows.json")
    if not os.path.isfile(path):
        with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{HF_DATASET}") as fh:
            sha = json.load(fh)["sha"]
        if sha != HF_REVISION:
            raise SystemExit(f"{HF_DATASET} is at revision {sha}, not {HF_REVISION}; the dataset "
                             f"server serves the newest one, so check the change before using it")
        rows = []
        for off in range(0, HF_ROWS, 100):
            url = (f"https://datasets-server.huggingface.co/rows?dataset={HF_DATASET}"
                   f"&config=default&split=full&offset={off}&length=100")
            with urllib.request.urlopen(url) as fh:
                rows += [x["row"] for x in json.load(fh)["rows"]]
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
    with open(path, encoding="utf-8") as fh:
        rows = json.load(fh)
    if len(rows) != HF_ROWS:
        raise SystemExit(f"{path}: {len(rows)} rows, expected {HF_ROWS}")
    return rows


def refusal(psmiles: str) -> str:
    """Why a repeat unit is left out of the PL2 tables, or an empty string."""
    mol = Chem.MolFromSmiles(psmiles)
    if mol is not None and sum(a.GetAtomicNum() == 0 for a in mol.GetAtoms()) == 4:
        return "builder: 4 attachment points"     # a ladder polymer; PL2 had no builder for it
    return chemistry_refusal(psmiles)


def chemistry_refusal(psmiles: str) -> str:
    """Why the builder or the field description cannot take a repeat unit, or an empty string."""
    try:
        parse_repeat_unit(psmiles)
    except ValueError as exc:
        text = str(exc)
        return ("builder: " + text.split(" has ", 1)[1].split(";")[0]) if " has " in text \
            else "builder: " + text
    mol = Chem.MolFromSmiles(psmiles)
    other = sorted({a.GetSymbol() for a in mol.GetAtoms()
                    if a.GetAtomicNum() > 0 and a.GetSymbol() not in GASTEIGER_ELEMENTS})
    if other:
        return "element without Gasteiger charges: " + ", ".join(other)
    if sum(a.GetFormalCharge() for a in mol.GetAtoms()) != 0:
        return "net formal charge"
    return ""


#: a benzene ring written as a cyclohexa-1,3-diene: two adjacent sp3 ring carbons with hydrogen
BROKEN_BENZENE = Chem.MolFromSmarts(
    "[#6;R;!a]1=[#6;!a][#6;!a]=[#6;!a][#6;X4;!H0][#6;X4;!H0]1")
#: a carbonyl carbon of an imide, amide or ester with a methylene where the oxygen belongs
CARBONYL_METHYLENE = Chem.MolFromSmarts("[#6](=[CH2;D1])~[#7,#8]")
#: rows of the PL6 table refused by hand, with the reason; the junction-ring rule finds perm558
PL6_FAULTS = {
    "perm054": "Troger's base written with two bridges per unit; the junction would be a ring "
               "of four N and four C",
    "perm094": "repeats the five values of perm499, written with a broken aromatic ring",
    "perm618": "repeats the five values of perm005 (within 1 %); a regioisomer of its unit",
}


def repair(psmiles: str):
    """(repaired SMILES, broken benzene rings made aromatic, carbonyl oxygens restored).

    The atom order of the input is kept (``canonical=False``), so that the attachment points stay
    in the order written, which decides the head and the tail.
    """
    m = Chem.RWMol(Chem.MolFromSmiles(psmiles))
    rings = 0
    while True:
        hit = m.GetSubstructMatch(BROKEN_BENZENE)
        if not hit:
            break
        m.GetBondBetweenAtoms(hit[4], hit[5]).SetBondType(Chem.BondType.DOUBLE)
        for i in (hit[4], hit[5]):
            a = m.GetAtomWithIdx(i)
            a.SetNumExplicitHs(0)
            a.SetNoImplicit(False)
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.SanitizeMol(m)
        rings += 1
    oxygens = 0
    for hit in m.GetSubstructMatches(CARBONYL_METHYLENE):
        m.GetAtomWithIdx(hit[1]).SetAtomicNum(8)
        oxygens += 1
    Chem.SanitizeMol(m)
    if not rings and not oxygens:
        return psmiles, 0, 0
    return Chem.MolToSmiles(m, canonical=False), rings, oxygens


def junction_ring(psmiles: str) -> int:
    """The size of the ring that forms where two units of a ladder polymer meet (0 if linear)."""
    unit = parse_repeat_unit(psmiles)
    if not unit.ladder:
        return 0
    m = unit.mol
    nb = lambda i: m.GetAtomWithIdx(i).GetNeighbors()[0].GetIdx()      # noqa: E731
    D = Chem.GetDistanceMatrix(m)
    return int(D[nb(unit.head), nb(unit.head2)] + D[nb(unit.tail), nb(unit.tail2)] + 2)


def pl6_table(raw):
    """The rows of ``permeability_pl6.csv`` and the counts for the provenance."""
    kept, refused, repairs = [], Counter(), Counter()
    for k, r in enumerate(raw):
        rid = f"perm{k + 1:03d}"
        src = r["SMILES"]
        why = chemistry_refusal(src)
        if not why and rid in PL6_FAULTS:
            why = "fault: " + PL6_FAULTS[rid]
        if not why and 0 < junction_ring(src) < 5:
            why = f"fault: the junction ring would have {junction_ring(src)} atoms"
        if why:
            refused[why] += 1
            continue
        fixed, rings, oxygens = repair(src)
        what = []
        if rings:
            what.append(f"{rings} benzene ring{'s' if rings > 1 else ''} made aromatic")
            repairs["rows with benzene rings made aromatic"] += 1
            repairs["benzene rings made aromatic"] += rings
        if oxygens:
            what.append(f"{oxygens} carbonyl oxygen{'s' if oxygens > 1 else ''} restored")
            repairs["rows with carbonyl oxygens restored"] += 1
            repairs["carbonyl oxygens restored"] += oxygens
        if chemistry_refusal(fixed):
            raise SystemExit(f"{rid}: the repaired SMILES {fixed} is refused: "
                             f"{chemistry_refusal(fixed)}")
        row = {"id": rid, "psmiles": fixed, "ladder": int(parse_repeat_unit(fixed).ladder)}
        for g in GASES:
            v = r.get(g)
            row[f"log10_{g}"] = (f"{math.log10(float(v)):.4f}"
                                 if v not in (None, "") and float(v) > 0 else "")
        row["repair"] = "; ".join(what)
        row["source_psmiles"] = src if what else ""
        kept.append(row)
    return kept, refused, repairs


def write(name: str, rows, fields) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    prov = {}
    # ---- density
    raw = radonpy_rows()
    kept, refused = [], Counter()
    for r in raw:
        why = refusal(r["smiles"])
        if why:
            refused[why] += 1
            continue
        kept.append({"id": r["monomer_ID"].lower(), "psmiles": r["smiles"],
                     "polymer_class": r["polymer_class"], "tacticity": r["tacticity"],
                     "density": r["density"], "density_std": r["density_std"],
                     "degree_of_polymerisation": r["DP"], "mn": r["Mn"]})
    write("pi1070.csv", kept, list(kept[0]))
    prov["pi1070"] = {"source": RADONPY_URL, "md5": RADONPY_MD5, "licence": "BSD 3-Clause",
                      "paper": "Hayashi et al., npj Comput. Mater. 8, 222 (2022), "
                               "doi:10.1038/s41524-022-00906-4",
                      "rows": len(raw), "kept": len(kept), "refused": dict(refused),
                      "classes": dict(sorted(Counter(r["polymer_class"] for r in kept).items(),
                                             key=lambda kv: int(kv[0])))}
    print(f"PI1070: {len(raw)} rows, {len(kept)} kept, refused {dict(refused)}")
    # ---- permeability
    raw = permeability_rows()
    kept, refused = [], Counter()
    for k, r in enumerate(raw):
        why = refusal(r["SMILES"])
        if why:
            refused[why] += 1
            continue
        row = {"id": f"perm{k + 1:03d}", "psmiles": r["SMILES"]}
        for g in GASES:
            v = r.get(g)
            row[f"log10_{g}"] = (f"{math.log10(float(v)):.4f}"
                                 if v not in (None, "") and float(v) > 0 else "")
        kept.append(row)
    write("permeability.csv", kept, list(kept[0]))
    prov["permeability"] = {"source": f"https://huggingface.co/datasets/{HF_DATASET}",
                            "revision": HF_REVISION, "licence": "MIT (torch-molecule)",
                            "unit": "log10 of the permeability in Barrer",
                            "rows": len(raw), "kept": len(kept), "refused": dict(refused),
                            "labels": {g: sum(1 for r in kept if r[f"log10_{g}"]) for g in GASES}}
    print(f"permeability: {len(raw)} rows, {len(kept)} kept, refused {dict(refused)}")
    # ---- the PL6 table: ladders kept, two faults of the source repaired
    kept6, refused6, repairs6 = pl6_table(raw)
    write("permeability_pl6.csv", kept6, list(kept6[0]))
    prov["permeability_pl6"] = {
        "source": "the permeability rows above", "kept": len(kept6),
        "ladders": sum(r["ladder"] for r in kept6), "refused": dict(sorted(refused6.items())),
        "repairs": dict(sorted(repairs6.items())),
        "labels": {g: sum(1 for r in kept6 if r[f"log10_{g}"]) for g in GASES}}
    print(f"permeability, PL6: {len(kept6)} kept ({prov['permeability_pl6']['ladders']} "
          f"ladders), refused {dict(refused6)}, repairs {dict(repairs6)}")
    # ---- licences and provenance
    shutil.copyfile(fetch(RADONPY_LICENSE, os.path.join(CACHE, "LICENSE_RadonPy.txt")),
                    os.path.join(OUT, "LICENSE_RadonPy.txt"))
    shutil.copyfile(fetch(TORCH_MOLECULE_LICENSE, os.path.join(CACHE, "LICENSE_torch-molecule.txt")),
                    os.path.join(OUT, "LICENSE_torch-molecule.txt"))
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
