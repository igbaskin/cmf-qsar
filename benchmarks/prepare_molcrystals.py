#!/usr/bin/env python
"""
Melting points with crystal structures: the data set of plan item M8.

No public set gives a melting point together with the crystal structure it belongs to, so this
script makes one from two open sources:

* **Melting points**: the ONS Open Melting Point Collection (ONSMP029, Bradley, Lang, Williams,
  Curtin; figshare 10.6084/m9.figshare.1031638 record 93086, CC BY 4.0): 2,706 compounds whose
  melting points were measured at least twice and agree, with a SMILES string each.
* **Crystal structures**: the Crystallography Open Database (COD, public domain), searched by
  the sum formula of each compound.

A COD entry is accepted for a compound only when the molecules perceived from its cell by
`cmfqsar.molcrystal` (bonds across the cell boundary, then bond orders and charges) are the
compound: every distinct molecule of the cell must have the InChI skeleton (the first block of
the InChI key) of one component of the compound's SMILES, and every component must be found.
That excludes solvates and co-crystals, and it is a test of the perception at the same time. The
candidates of a compound are tried in order of their R factor, entries measured under pressure or
marked as duplicates are left out, and at most `--tries` are downloaded.

Stages
------
    python benchmarks/prepare_molcrystals.py --stage labels    # the ONS table, fetched once
    python benchmarks/prepare_molcrystals.py --stage query     # one COD search per formula
    python benchmarks/prepare_molcrystals.py --stage match     # CIFs fetched and perceived
    python benchmarks/prepare_molcrystals.py --stage write     # examples/melting_points

The searches and the CIF files are kept in `benchmarks/cache/molcrystals` (not in the
repository); what is kept in the repository is the result: `examples/melting_points` with the
matched cells as extended XYZ, the labels and the COD identifiers.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.rdMolDescriptors import CalcMolFormula  # noqa: E402

RDLogger.DisableLog("rdApp.*")

ONS_URL = "https://ndownloader.figshare.com/files/94753"
COD_SEARCH = "https://www.crystallography.net/cod/result.php?formula={formula}&format=csv"
COD_CIF = "https://www.crystallography.net/cod/{cod}.cif"
CACHE = os.path.join(HERE, "cache", "molcrystals")
OUT = os.path.join(ROOT, "examples", "melting_points")
#: seconds between two requests to one server, so that the run stays polite
PAUSE = 1.0
ALLOWED = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)
        fh.write("\n")


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def fetch(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cmfqsar-benchmark"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #

def cod_formula(mol) -> str:
    """The sum formula as COD writes it: Hill order, element and count separated by spaces,
    a count of one left out (``C6 H6``, ``C H4 O``)."""
    f = re.sub(r"[+-]\d*$", "", CalcMolFormula(mol))
    parts = re.findall(r"([A-Z][a-z]?)(\d*)", f)
    return " ".join(f"{el}{n if n not in ('', '1') else ''}" for el, n in parts)


def skeletons(mol) -> List[str]:
    """The InChI skeleton (first block of the key) of every component of a molecule."""
    out = []
    for frag in Chem.GetMolFrags(mol, asMols=True):
        key = Chem.MolToInchiKey(frag) or ""
        if key:
            out.append(key.split("-")[0])
    return sorted(set(out))


def stage_labels(source: Optional[str] = None) -> List[Dict[str, object]]:
    """The ONS table as rows with a parsed molecule, its formula and its skeletons."""
    import openpyxl

    path = source or os.path.join(CACHE, "ONSMeltingPointsDoubleValidated.xlsx")
    if not os.path.isfile(path):
        os.makedirs(CACHE, exist_ok=True)
        data = fetch(ONS_URL)
        with open(path, "wb") as fh:
            fh.write(data)
        log(f"ONS table: {len(data)} bytes to {path}")
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    head = [str(h) for h in rows[0]]
    col = {h: i for i, h in enumerate(head)}
    out = []
    for r in rows[1:]:
        smi = r[col["SMILES"]]
        if not smi:
            continue
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        mh = Chem.AddHs(m)
        if set(a.GetSymbol() for a in mh.GetAtoms()) - ALLOWED:
            continue
        lo = [h for h in head if h.startswith("Low")][0]
        hi = [h for h in head if h.startswith("High")][0]
        av = [h for h in head if h.startswith("Ave")][0]
        out.append({"csid": str(r[col["CSID"]]), "name": str(r[col["name"]]),
                    "smiles": str(smi), "mp_low": float(r[col[lo]]),
                    "mp_high": float(r[col[hi]]), "mp": float(r[col[av]]),
                    "formula": cod_formula(mh), "skeletons": skeletons(mh),
                    "heavy_atoms": int(m.GetNumHeavyAtoms())})
    jdump(out, os.path.join(CACHE, "labels.json"))
    log(f"labels: {len(out)} compounds with organic elements and a SMILES RDKit reads")
    return out


# --------------------------------------------------------------------------- #
# COD searches
# --------------------------------------------------------------------------- #

def _query_path(formula: str) -> str:
    return os.path.join(CACHE, "query", formula.replace(" ", "_") + ".csv")


def stage_query(limit: int = 0) -> Dict[str, int]:
    """One COD search per distinct formula, kept as CSV; an empty file is a search with no hit."""
    labels = jload(os.path.join(CACHE, "labels.json"))
    formulas = sorted({r["formula"] for r in labels})
    if limit:
        formulas = formulas[:limit]
    os.makedirs(os.path.join(CACHE, "query"), exist_ok=True)
    done = hits = 0
    t0 = time.time()
    for f in formulas:
        path = _query_path(f)
        if os.path.isfile(path):
            continue
        url = COD_SEARCH.format(formula=urllib.parse.quote(f))
        try:
            text = fetch(url).decode("utf-8", "replace")
        except Exception as exc:                     # noqa: BLE001 - retried on the next run
            log(f"  {f}: {exc}")
            time.sleep(5 * PAUSE)
            continue
        body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        done += 1
        hits += int(body.count("\n") > 0)
        if done % 100 == 0:
            log(f"  {done} searches, {hits} with entries ({time.time() - t0:.0f} s)")
        time.sleep(PAUSE)
    log(f"query: {done} new searches of {len(formulas)} formulas")
    return {"formulas": len(formulas), "new": done}


def _plain(name: str) -> str:
    """A name reduced to letters and digits in lower case, for comparing names."""
    return re.sub(r"[^a-z0-9]", "", str(name or "").lower())


def candidates(formula: str, name: str = "") -> List[Dict[str, str]]:
    """The COD entries of a formula that can be used, best first.

    An entry whose common or chemical name is the compound's name comes first, because a sum
    formula is shared by every isomer; then the rest. Within each group the lower R factor
    comes first.
    """
    path = _query_path(formula)
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(io.StringIO(fh.read())))
    out = []
    for r in rows:
        if r.get("cellpressure") or r.get("diffrpressure") or r.get("duplicateof"):
            continue
        if "has coordinates" not in (r.get("flags") or ""):
            continue
        if (r.get("status") or "") in ("errors", "retracted"):
            continue
        try:
            rf = float(r.get("Robs") or r.get("Rall") or 1.0)
        except ValueError:
            rf = 1.0
        names = {_plain(r.get("commonname")), _plain(r.get("chemname"))} - {""}
        want = _plain(name)
        named = bool(want) and any(want == n or (len(want) > 5 and want in n) for n in names)
        out.append({"cod": r["file"], "R": rf, "T": r.get("diffrtemp") or r.get("celltemp"),
                    "sg": r.get("sg", ""), "Z": r.get("Z", ""), "year": r.get("year", ""),
                    "named": named})
    out.sort(key=lambda d: (not d["named"], d["R"], d["cod"]))
    return out


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def _cif(cod: str) -> str:
    path = os.path.join(CACHE, "cif", f"{cod}.cif")
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = fetch(COD_CIF.format(cod=cod))
        with open(path, "wb") as fh:
            fh.write(data)
        time.sleep(PAUSE)
    return path


def judge(path: str, want: List[str]) -> Dict[str, object]:
    """Perceive the molecules of one CIF and say whether they are the compound."""
    from cmfqsar import molcrystal as M

    try:
        atoms = M.read_molecular_crystal(path)
        mols = M.cell_molecules(atoms)
    except Exception as exc:                         # noqa: BLE001 - the reason is the result
        text = (str(exc).splitlines() or [""])[0][:160]
        return {"ok": False, "reason": f"{type(exc).__name__}: {text}"}
    found = sorted({c.inchikey.split("-")[0] for c in mols if c.inchikey})
    ok = found == sorted(want)
    return {"ok": ok, "found": found, "atoms": len(atoms), "molecules": len(mols),
            "smiles": sorted({c.smiles for c in mols}),
            "reason": "" if ok else "the molecules of the cell are not the compound"}


def match_one(row: Dict[str, object], tries: int) -> Dict[str, object]:
    cands = candidates(str(row["formula"]), str(row["name"]))
    rec = {"csid": row["csid"], "candidates": len(cands),
           "named": sum(1 for c in cands if c["named"]), "tried": []}
    for c in cands[:tries]:
        try:
            path = _cif(c["cod"])
        except Exception as exc:                     # noqa: BLE001
            rec["tried"].append({"cod": c["cod"], "ok": False, "reason": f"download: {exc}"})
            continue
        j = judge(path, list(row["skeletons"]))
        j["cod"] = c["cod"]
        j["R"] = c["R"]
        rec["tried"].append(j)
        if j["ok"]:
            rec["cod"] = c["cod"]
            break
    return rec


def stage_match(tries: int = 5, jobs: int = 2, limit: int = 0) -> Dict[str, int]:
    labels = jload(os.path.join(CACHE, "labels.json"))
    if limit:
        labels = labels[:limit]
    path = os.path.join(CACHE, "match.json")
    done = jload(path) if os.path.isfile(path) else {}
    todo = [r for r in labels if r["csid"] not in done and candidates(str(r["formula"]))]
    log(f"match: {len(todo)} compounds with COD entries to try ({len(done)} done before)")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for i, rec in enumerate(ex.map(lambda r: match_one(r, tries), todo)):
            done[rec["csid"]] = rec
            if (i + 1) % 50 == 0:
                jdump(done, path)
                n_ok = sum(1 for v in done.values() if v.get("cod"))
                log(f"  {i + 1}/{len(todo)}: {n_ok} matched ({time.time() - t0:.0f} s)")
    jdump(done, path)
    n_ok = sum(1 for v in done.values() if v.get("cod"))
    log(f"match: {n_ok} compounds matched to a COD structure")
    return {"tried": len(done), "matched": n_ok}


# --------------------------------------------------------------------------- #
# The data set
# --------------------------------------------------------------------------- #

def stage_write() -> Dict[str, object]:
    """examples/melting_points: the matched cells, the labels and how every compound fared."""
    from ase.io import write

    from cmfqsar import molcrystal as M

    labels = {r["csid"]: r for r in jload(os.path.join(CACHE, "labels.json"))}
    match = jload(os.path.join(CACHE, "match.json"))
    cells, table = [], []
    reasons: Dict[str, int] = {}
    for csid, r in labels.items():
        m = match.get(csid)
        if not m or not candidates(str(r["formula"])):
            reasons["no COD entry with the formula"] = reasons.get(
                "no COD entry with the formula", 0) + 1
            continue
        if not m.get("cod"):
            last = m["tried"][-1]["reason"] if m["tried"] else "nothing tried"
            key = last.split(":")[0] if last.startswith(("PerceptionError", "download")) \
                else last
            reasons[key] = reasons.get(key, 0) + 1
            continue
        atoms = M.read_molecular_crystal(os.path.join(CACHE, "cif", f"{m['cod']}.cif"))
        atoms.info = {"name": f"cod_{m['cod']}", "cod": m["cod"], "csid": csid}
        cells.append(atoms)
        table.append({"name": f"cod_{m['cod']}", "cod": m["cod"], "csid": csid,
                      "compound": r["name"], "smiles": r["smiles"], "mp_C": r["mp"],
                      "mp_low_C": r["mp_low"], "mp_high_C": r["mp_high"],
                      "atoms_in_cell": len(atoms),
                      "molecules_in_cell": [t for t in m["tried"] if t["ok"]][0]["molecules"]})
    os.makedirs(OUT, exist_ok=True)
    write(os.path.join(OUT, "crystals.extxyz"), cells, format="extxyz")
    with open(os.path.join(OUT, "melting_points.csv"), "w", encoding="utf-8",
              newline="\n") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    summary = {"compounds": len(labels), "matched": len(table), "not_matched": reasons}
    jdump(summary, os.path.join(OUT, "provenance.json"))
    log(f"write: {len(table)} crystals to {OUT}; not matched: {reasons}")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("labels", "query", "match", "write"), action="append",
                    required=True)
    ap.add_argument("--source", help="a local copy of the ONS table (stage labels)")
    ap.add_argument("--tries", type=int, default=5, help="COD entries tried per compound")
    ap.add_argument("--jobs", type=int, default=2, help="downloads at a time")
    ap.add_argument("--limit", type=int, default=0, help="only the first N (a pilot)")
    args = ap.parse_args(argv)
    for st in args.stage:
        if st == "labels":
            stage_labels(args.source)
        elif st == "query":
            stage_query(args.limit)
        elif st == "match":
            stage_match(args.tries, args.jobs, args.limit)
        elif st == "write":
            stage_write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
