#!/usr/bin/env python
"""
The copolymer table of plan item PL4: computed electron affinities and ionization potentials of
binary copolymers with a known composition and chain architecture.

**Source.**  The VIPEA set of M. Aldeghi and C. W. Coley, "A graph representation of molecular
ensembles for polymer property prediction", *Chem. Sci.* 13, 10486 (2022),
doi:10.1039/D2SC02839E: 42,966 copolymers of 9 aryl diboronic acids (monomer A) and 682 aryl
dibromides (monomer B), each as alternating (1:1), random or block copolymer at A:B = 1:1, 1:3
and 3:1.  The file is ``datasets/vipea/dataset.csv`` of github.com/coleygroup/polymer-chemprop-data
at commit 2af92df, MIT licence.  The values were computed with xTB on octamers ended by
hydrogen, Boltzmann-averaged over 8 conformers and averaged over the sequences of each copolymer
(``datasets/vipea/calculations/make-polymers.py`` of the same commit makes the sequences: every
distinct order of the units for a random copolymer, a sequence and its reverse counted once, up
to 32; AAAABBBB, AABBBBBB or AAAAAABB for a block copolymer; ABABABAB for the alternating one).
They are computed, not measured.

**Repeat units.**  The two B(OH)2 groups of monomer A and the two bromine atoms on aromatic
carbon of monomer B are the attachment points, because the octamers were made by joining these
groups.  Two dibromides carry a third bromine and are left out.

**The subset.**  A kernel model on all 42,966 rows, with folds repeated, costs days.  25 of the
680 usable dibromides are drawn with a fixed seed, and every copolymer of each with the 9
diboronic acids is kept: 225 pairs, 1,575 copolymers.

What this script does: download the file and the licence into ``benchmarks/cache/copolymers/raw/``
(the file is checked against its SHA-256), make the repeat units, draw the subset, and write
``examples/copolymers/vipea.csv``, ``LICENSE_polymer-chemprop-data.txt`` and ``provenance.json``.

Usage
-----
    python benchmarks/prepare_copolymers.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.polymer import parse_repeat_unit  # noqa: E402

RDLogger.DisableLog("rdApp.*")

COMMIT = "2af92df9560c9f1caf6ddc8a71e2290768a71674"
BASE = f"https://raw.githubusercontent.com/coleygroup/polymer-chemprop-data/{COMMIT}"
DATA_URL = f"{BASE}/datasets/vipea/dataset.csv"
LICENSE_URL = f"{BASE}/LICENSE"
SHA256 = "6f9d4d8c0cb8154a2150235569ba71f116f9bda1417303e19184e46d64a8bc46"
CACHE = os.path.join(HERE, "cache", "copolymers", "raw")
OUT = os.path.join(ROOT, "examples", "copolymers")
N_DIBROMIDES = 25
SEED = 0

BORONIC = Chem.MolFromSmarts("[BX3](-[OX2H1])-[OX2H1]")
ARYL_BROMINE = Chem.MolFromSmarts("[Br;$(Br-c)]")
STAR = Chem.MolFromSmiles("*")


def fetch(url: str, path: str) -> str:
    if not os.path.isfile(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading {url} ...", flush=True)
        urllib.request.urlretrieve(url, path)
    return path


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def unit_of(smiles: str, kind: str) -> str:
    """The repeat unit of a monomer: its two reactive groups as attachment points, or ""."""
    mol = Chem.MolFromSmiles(smiles)
    patt = BORONIC if kind == "A" else ARYL_BROMINE
    if len(mol.GetSubstructMatches(patt)) != 2:
        return ""
    out = Chem.ReplaceSubstructs(mol, patt, STAR, replaceAll=True)[0]
    Chem.SanitizeMol(out)
    smi = Chem.MolToSmiles(out)
    parse_repeat_unit(smi)                       # refuses anything that is not a linear unit
    return smi


def main() -> int:
    path = fetch(DATA_URL, os.path.join(CACHE, "dataset.csv"))
    digest = sha256(path)
    if digest != SHA256:
        raise SystemExit(f"{path}: SHA-256 {digest}, expected {SHA256}")
    with open(path, encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    a_smiles = {r["poly_id"].split("_")[0]: r["monoA"] for r in raw}
    b_smiles = {r["poly_id"].split("_")[1]: r["monoB"] for r in raw}
    units_a = {k: unit_of(s, "A") for k, s in a_smiles.items()}
    units_b = {k: unit_of(s, "B") for k, s in b_smiles.items()}
    bad_a = sorted(k for k, u in units_a.items() if not u)
    bad_b = sorted((k for k, u in units_b.items() if not u), key=int)
    if bad_a:
        raise SystemExit(f"diboronic acids without two B(OH)2 groups: {bad_a}")
    usable = sorted((k for k, u in units_b.items() if u), key=int)
    chosen = sorted(np.random.RandomState(SEED).choice(usable, size=N_DIBROMIDES,
                                                        replace=False).tolist(), key=int)
    rows = []
    for r in raw:
        a, b = r["poly_id"].split("_")
        if b not in chosen:
            continue
        na, nb = (int(x[:-1]) for x in r["comp"].split("_"))
        rows.append({"id": f"{r['poly_id']}_{r['poly_type']}_{r['comp']}", "pair": r["poly_id"],
                     "a": a, "b": b, "architecture": r["poly_type"], "n_a": na, "n_b": nb,
                     "frac_a": r["fracA"], "frac_b": r["fracB"],
                     "ea_ev": f"{float(r['EA (eV)']):.6f}", "ip_ev": f"{float(r['IP (eV)']):.6f}",
                     "unit_a": units_a[a], "unit_b": units_b[b],
                     "monomer_a": r["monoA"], "monomer_b": r["monoB"]})
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "vipea.csv"), "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    shutil.copyfile(fetch(LICENSE_URL, os.path.join(CACHE, "LICENSE")),
                    os.path.join(OUT, "LICENSE_polymer-chemprop-data.txt"))
    prov = {"source": DATA_URL, "commit": COMMIT, "sha256": digest, "licence": "MIT",
            "paper": "M. Aldeghi and C. W. Coley, Chem. Sci. 13, 10486 (2022), "
                     "doi:10.1039/D2SC02839E",
            "rows": len(raw), "diboronic_acids": len(units_a), "dibromides": len(units_b),
            "dibromides_refused": {k: "a third bromine on aromatic carbon" for k in bad_b},
            "dibromides_kept": chosen, "seed": SEED, "kept": len(rows),
            "kinds": dict(sorted(Counter(f"{r['architecture']} {r['n_a']}A_{r['n_b']}B"
                                         for r in rows).items()))}
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    print(f"{len(raw)} rows; {len(usable)} usable dibromides, {len(bad_b)} refused; kept "
          f"{len(rows)} copolymers of {len(chosen)} dibromides with {len(units_a)} diboronic "
          f"acids; file SHA-256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
