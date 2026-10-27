#!/usr/bin/env python
"""
Build ``examples/antitarget/h3_herg.sdf``: a series with a potency and an anti-target (item MC4).

Medicinal chemistry asks of every series where field may be added for the target that is **not**
also added for the off-target.  The two-task model of this program answers that as a difference
of co-fields, and plan item MC4 says what was missing: a series in the program's format with a
potency and an anti-target activity measured on the same compounds.

This script fetches one from ChEMBL:

    Lazewska D. et al., "Novel and highly potent histamine H3 receptor ligands. Part 1:
    withdrawing of hERG activity", Bioorg. Med. Chem. Lett. 21 (2011) 5378-5382,
    https://doi.org/10.1016/j.bmcl.2011.07.006  (ChEMBL document CHEMBL1821550)

The paper is about exactly this question: how to keep the histamine H3 affinity of a series of
basic amines while taking its hERG activity away.  Every compound carries both numbers, measured
**in the same laboratory in one assay each**, which is what makes them comparable; an
anti-target number from another paper would be a different measurement and is not used.

    histamine H3   displacement of [125I]iodoproxyfan from the human recombinant receptor, Ki
    hERG           displacement of [3H]dofetilide from human recombinant ERG, IC50

Written tags: ``pKi_H3`` (the primary activity, -log10 Ki in M), ``pIC50_hERG`` (the second
endpoint), ``selectivity`` (their difference), the ChEMBL identifiers of the molecule, the two
assays and the document, the two assay descriptions, and the alignment note.  The assay
conditions travel with the file as columns, as the item requires.

Usage
-----
    python examples/make_antitarget_examples.py                 # fetch and build
    python examples/make_antitarget_examples.py --cache x.json  # keep the download
    python examples/make_antitarget_examples.py --offline       # build from the cache only

The download is a few hundred small records from the ChEMBL web service, which is free and needs
no key; ChEMBL data is CC BY-SA 3.0, so the file may be redistributed with attribution.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from cmfqsar.align import mcs_align  # noqa: E402

RDLogger.DisableLog("rdApp.*")

BASE = "https://www.ebi.ac.uk/chembl/api/data"
DOCUMENT = "CHEMBL1821550"
PRIMARY = ("CHEMBL264", "histamine H3 receptor", "pKi_H3")
ANTI = ("CHEMBL240", "hERG (KCNH2)", "pIC50_hERG")
#: a molecule whose maximum common substructure with the reference is smaller than this share of
#: its own heavy atoms is not in the same series and is left out
MCS_SHARE = 0.5
#: and the common part has to have at least this many atoms for the frame to mean anything
MCS_ATOMS = 12


def get(path, **params):
    params.setdefault("format", "json")
    url = f"{BASE}/{path}?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=180) as fh:
                return json.load(fh)
        except Exception as exc:
            if attempt == 3:
                raise
            print(f"   retrying after {type(exc).__name__}", flush=True)
            time.sleep(5)


def page(path, **params):
    offset = 0
    while True:
        d = get(path, limit=1000, offset=offset, **params)
        key = next(k for k in d if k != "page_meta")
        rows = d[key]
        if not rows:
            return
        for r in rows:
            yield r
        if not d["page_meta"].get("next"):
            return
        offset += 1000


def fetch(document=DOCUMENT):
    """Every binding activity of the paper, with the molecule and the assay it belongs to."""
    out = []
    for r in page("activity", document_chembl_id=document, assay_type="B",
                  pchembl_value__isnull="false",
                  only="molecule_chembl_id,target_chembl_id,target_pref_name,standard_type,"
                       "standard_value,standard_units,pchembl_value,assay_chembl_id,"
                       "assay_description,canonical_smiles,document_chembl_id"):
        out.append(r)
    info = get(f"document/{document}")
    return out, {k: info.get(k) for k in ("doi", "journal", "year", "volume", "first_page",
                                          "title", "authors")}


def pair_up(rows):
    """{molecule: {'smiles', primary tag, anti tag, the two assays}} for the compounds with both."""
    by_mol = {}
    for r in rows:
        t = r.get("target_chembl_id")
        if t not in (PRIMARY[0], ANTI[0]):
            continue
        m = r.get("molecule_chembl_id")
        if not m or not r.get("pchembl_value"):
            continue
        slot = by_mol.setdefault(m, {})
        key = "primary" if t == PRIMARY[0] else "anti"
        slot.setdefault(key, []).append(r)
        slot.setdefault("smiles", r.get("canonical_smiles"))
    out = {}
    for m, slot in by_mol.items():
        if "primary" not in slot or "anti" not in slot or not slot.get("smiles"):
            continue
        p = slot["primary"]
        a = slot["anti"]
        out[m] = dict(
            smiles=slot["smiles"],
            primary=float(np.mean([float(r["pchembl_value"]) for r in p])),
            anti=float(np.mean([float(r["pchembl_value"]) for r in a])),
            primary_assay=p[0].get("assay_chembl_id"),
            anti_assay=a[0].get("assay_chembl_id"),
            primary_description=p[0].get("assay_description"),
            anti_description=a[0].get("assay_description"),
            primary_type=p[0].get("standard_type"),
            anti_type=a[0].get("standard_type"),
            repeats=max(len(p), len(a)))
    return out


def embed(mol, seed=0xC0FFEE):
    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed) & 0x7FFFFFFF
    if AllChem.EmbedMolecule(m, params) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=400)
    except Exception:
        pass
    return m


def choose_reference(mols):
    """The molecule most like the rest, by the size of its common substructure with them."""
    from rdkit.Chem import rdFMCS
    best = None
    sample = mols[:: max(1, len(mols) // 25)]
    for i, m in enumerate(mols):
        total = 0
        for other in sample:
            if other is m:
                continue
            res = rdFMCS.FindMCS([Chem.RemoveHs(m), Chem.RemoveHs(other)], timeout=2,
                                 ringMatchesRingOnly=True, completeRingsOnly=False)
            total += res.numAtoms if not res.canceled else 0
        if best is None or total > best[0]:
            best = (total, i)
    return best[1]


def build(records, seed=0xC0FFEE, verbose=True):
    """Embed, align to the reference by the common substructure, and keep what belongs."""
    from rdkit.Chem import rdFMCS

    names = sorted(records)
    mols, kept = [], []
    for m in names:
        mol = Chem.MolFromSmiles(records[m]["smiles"])
        if mol is None:
            continue
        got = embed(mol, seed)
        if got is None:
            continue
        got.SetProp("_Name", m)
        mols.append(got)
        kept.append(m)
    if verbose:
        print(f"  {len(mols)} of {len(names)} molecules embedded")
    ref_i = choose_reference(mols)
    ref = mols[ref_i]
    if verbose:
        print(f"  the frame is {kept[ref_i]}, the molecule most like the rest")
    out, dropped = [], 0
    for mol, name in zip(mols, kept):
        heavy = Chem.RemoveHs(mol)
        res = rdFMCS.FindMCS([heavy, Chem.RemoveHs(ref)], timeout=5,
                             ringMatchesRingOnly=True, completeRingsOnly=False)
        share = res.numAtoms / max(heavy.GetNumHeavyAtoms(), 1)
        if res.canceled or res.numAtoms < MCS_ATOMS or share < MCS_SHARE:
            dropped += 1
            continue
        if mol is not ref:
            try:
                mcs_align(mol, ref, timeout=5)
            except Exception:
                dropped += 1
                continue
        out.append((name, mol, res.numAtoms, share))
    if verbose:
        print(f"  {len(out)} kept, {dropped} left out by the common-substructure rule")
    return out, kept[ref_i]


def write(path, aligned, records, info, reference):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    n = 0
    with Chem.SDWriter(path) as w:
        for name, mol, common, share in aligned:
            r = records[name]
            mol.SetProp("_Name", name)
            mol.SetProp(PRIMARY[2], f"{r['primary']:.3f}")
            mol.SetProp(ANTI[2], f"{r['anti']:.3f}")
            mol.SetProp("selectivity", f"{r['primary'] - r['anti']:.3f}")
            mol.SetProp("activity", f"{r['primary']:.3f}")
            mol.SetProp("molecule_chembl_id", name)
            mol.SetProp("document_chembl_id", DOCUMENT)
            mol.SetProp("primary_target", f"{PRIMARY[1]} ({PRIMARY[0]})")
            mol.SetProp("primary_assay", str(r["primary_assay"]))
            mol.SetProp("primary_assay_description", str(r["primary_description"]))
            mol.SetProp("primary_standard_type", str(r["primary_type"]))
            mol.SetProp("antitarget", f"{ANTI[1]} ({ANTI[0]})")
            mol.SetProp("antitarget_assay", str(r["anti_assay"]))
            mol.SetProp("antitarget_assay_description", str(r["anti_description"]))
            mol.SetProp("antitarget_standard_type", str(r["anti_type"]))
            mol.SetProp("reference", f"{info.get('journal')} {info.get('year')} "
                                     f"{info.get('volume')}:{info.get('first_page')} "
                                     f"doi:{info.get('doi')}")
            mol.SetProp("common_atoms_with_frame", str(common))
            mol.SetProp("frame", reference)
            w.write(mol)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(HERE, "antitarget", "h3_herg.sdf"))
    ap.add_argument("--cache", default=os.path.join(HERE, "antitarget", "chembl_raw.json"),
                    help="where the download is kept, so the build can be repeated offline")
    ap.add_argument("--offline", action="store_true", help="build from the cache, fetch nothing")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    a = ap.parse_args()

    if a.offline:
        if not os.path.exists(a.cache):
            raise SystemExit(f"--offline needs {a.cache}, which does not exist")
        with io.open(a.cache, encoding="utf-8") as fh:
            blob = json.load(fh)
        rows, info = blob["rows"], blob["document"]
        print(f"{len(rows)} activities from the cache")
    else:
        print(f"fetching document {DOCUMENT} from ChEMBL")
        rows, info = fetch()
        print(f"  {len(rows)} binding activities with a pChEMBL value")
        os.makedirs(os.path.dirname(a.cache), exist_ok=True)
        with io.open(a.cache, "w", encoding="utf-8") as fh:
            json.dump({"rows": rows, "document": info}, fh, indent=1)
        print(f"  kept in {a.cache}")

    records = pair_up(rows)
    print(f"  {len(records)} molecules carry both {PRIMARY[1]} and {ANTI[1]}")
    aligned, reference = build(records, seed=a.seed)
    n = write(a.out, aligned, records, info, reference)
    p = np.array([records[name]["primary"] for name, _m, _c, _s in aligned])
    q = np.array([records[name]["anti"] for name, _m, _c, _s in aligned])
    print(f"{n} molecules -> {a.out}")
    print(f"  {PRIMARY[2]} {p.min():.2f} to {p.max():.2f} (spread {p.std():.2f})")
    print(f"  {ANTI[2]} {q.min():.2f} to {q.max():.2f} (spread {q.std():.2f})")
    print(f"  their correlation {np.corrcoef(p, q)[0, 1]:+.3f}; selectivity "
          f"{(p - q).min():.2f} to {(p - q).max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
