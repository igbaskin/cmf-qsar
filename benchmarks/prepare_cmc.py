#!/usr/bin/env python
"""
Measured critical micelle concentrations of surfactants with one 3D structure per surfactant
ion, for ``benchmarks/run_cmc.py``.

The target is pCMC, the negative decimal logarithm of the critical micelle concentration (CMC) in
mol/L, as the source gives it: a surfactant that forms micelles at a lower concentration has a
higher pCMC.

Source
------
SurfPro (S. L. Hoedl, L. Hermans, P. F. J. Dankloff, A. Piruska, W. T. S. Huck,
W. E. Robinson, Digital Discovery 4 (2025) 1176, doi:10.1039/D4DD00393D), the files
``data/surfpro_train.csv`` and ``data/surfpro_test.csv`` of the repository
github.com/BigChemistry-RobotLab/SurfPro at the commit in ``COMMIT``. This is the second version
of the data (SurfProV2), in which the authors corrected many structures after the paper. The
data are under the licence CC BY-NC-SA 4.0: they may be shared and adapted for non-commercial
use, with attribution, and an adapted version must carry the same licence.

* ``surfpro_train.csv``: the training part, with the paper's ten cross-validation folds; 1,204
  rows have a pCMC.
* ``surfpro_test.csv``: the paper's test set of 140 surfactants, drawn with stratification by
  surfactant type. The paper reports its errors on this set (Table 3).

Every row carries the surfactant type the authors gave it (anionic, cationic, gemini cationic,
non-ionic, sugar-based non-ionic, zwitterionic) and the temperature of the measurement (mostly
25 C). All temperatures are kept, as in the paper.

What is changed
---------------
A row is a salt: the surfactant ion and its counterions (Na+, K+, Li+, NH4+, Cl-, Br-, I-). The
3D structure is that of the surfactant ion alone, with its net charge; the counterion is kept as
the text property ``counterion``. A row is refused when its SMILES cannot be read, when it has
more than one organic component, when an element is outside H, B, C, N, O, F, Si, P, S, Cl, Br
and I, or when the ion has more than ``MAX_HEAVY`` heavy atoms. Unlike the other physical
property sets there is no limit on the molecular weight below that, because many non-ionic and
gemini surfactants weigh more than 700.

Two files are written to ``examples/cmc``: ``train.sdf`` and ``test.sdf``, with the properties
``pCMC``, ``smiles`` (the surfactant ion), ``salt_smiles`` (the salt as the source writes it),
``counterion``, ``surfactant_type``, ``temperature``, ``fold`` (the paper's fold, training part
only) and ``inchikey14`` (the first block of the InChIKey of the salt). A second row of the same
salt, the same isomeric SMILES, is left out; stereoisomers are kept. The 3D structures are those of ``benchmarks/prepare_logp.py`` (ETKDGv3 with seed
42, then MMFF94).

Usage:
    python benchmarks/prepare_cmc.py [--workers 8]
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

from rdkit import Chem, RDLogger  # noqa: E402

from prepare_logp import ORGANIC, count, embed_all, md5, skeleton, write  # noqa: E402

RDLogger.DisableLog("rdApp.*")

CACHE = os.path.join(HERE, "cache", "cmc", "surfpro")
OUT = os.path.join(ROOT, "examples", "cmc")
REPO = "BigChemistry-RobotLab/SurfPro"
#: the commit of 20 May 2026, the last change of the data files when this script was written
COMMIT = "35b8906bec72b4f792267c5713d7ec393fbe0122"
FILES = ("surfpro_train.csv", "surfpro_test.csv")
#: the largest surfactant ion kept; one ion of the source has 319 heavy atoms, the next 135
MAX_HEAVY = 150
#: the fragments that are counterions rather than surfactants
COUNTERIONS = {"[Na+]", "[K+]", "[Li+]", "[NH4+]", "[Cl-]", "[Br-]", "[I-]"}


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fetch() -> dict:
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for name in FILES:
        path = os.path.join(CACHE, f"{COMMIT[:7]}_{name}")
        if not os.path.isfile(path):
            url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/data/{name}"
            log(f"downloading {url}")
            urllib.request.urlretrieve(url, path)
        out[name] = path
    return out


def split_salt(smiles: str):
    """The surfactant ion and the counterion text, or (None, the reason it was refused)."""
    m = Chem.MolFromSmiles(smiles or "")
    if m is None:
        return None, "RDKit cannot read the SMILES"
    ions, organic = [], []
    for frag in Chem.GetMolFrags(m, asMols=True):
        text = Chem.MolToSmiles(frag)
        (ions if text in COUNTERIONS else organic).append(text)
    if len(organic) != 1:
        return None, "more than one organic component"
    ion = Chem.MolFromSmiles(organic[0])
    if any(a.GetSymbol() not in ORGANIC for a in ion.GetAtoms()):
        return None, "an element outside the organic set"
    if ion.GetNumHeavyAtoms() > MAX_HEAVY:
        return None, f"more than {MAX_HEAVY} heavy atoms"
    return (ion, ".".join(sorted(ions))), None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    files = fetch()
    prov = {"source": "SurfPro (Hoedl et al., Digital Discovery 4 (2025) 1176, "
                      "doi:10.1039/D4DD00393D), github.com/" + REPO + ", version SurfProV2",
            "licence": "CC BY-NC-SA 4.0", "commit": COMMIT,
            "files_md5": {n: md5(p) for n, p in files.items()}, "unit": "pCMC = -log10(CMC / M)"}
    seen = set()
    os.makedirs(OUT, exist_ok=True)
    for name, tag in (("surfpro_test.csv", "test"), ("surfpro_train.csv", "train")):
        with open(files[name], encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        recs, why, dup, types, temps = [], {}, 0, {}, {}
        for i, r in enumerate(rows):
            if not (r.get("pCMC") or "").strip():
                count(why, "no pCMC")
                continue
            got, reason = split_salt(r["SMILES"])
            if got is None:
                count(why, reason)
                continue
            ion, counterion = got
            salt = Chem.MolFromSmiles(r["SMILES"])
            key = skeleton(salt)
            # a second row of the same salt; stereoisomers are different salts, which the
            # first block of the InChIKey would not tell apart
            same = Chem.MolToSmiles(salt) + tag
            if same in seen:
                dup += 1
                continue
            seen.add(same)
            count(types, r["type"])
            t = f"{float(r['temp']):g}" if (r.get("temp") or "").strip() else ""
            count(temps, t)
            rec = {"name": f"surfpro-{tag}-{i}", "smiles": Chem.MolToSmiles(ion),
                   "pCMC": float(r["pCMC"]), "salt_smiles": r["SMILES"],
                   "counterion": counterion, "surfactant_type": r["type"], "temperature": t,
                   "inchikey14": key}
            if tag == "train":
                rec["fold"] = r["fold"]
            recs.append(rec)
        t0 = time.time()
        mols, failed = embed_all(recs, a.workers)
        write(mols, os.path.join(OUT, f"{tag}.sdf"))
        prov[tag] = {"rows": len(rows), "refused": why, "duplicate_structures": dup,
                     "types": types, "temperatures_C": temps,
                     "kept_before_embedding": len(recs), "embedded": len(mols),
                     "embedding_failed": failed}
        log(f"{tag}: {len(recs)} kept, {len(mols)} embedded, {len(failed)} failed "
            f"({time.time() - t0:.0f} s); refused {why}")
    with open(os.path.join(OUT, "provenance.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(prov, fh, indent=1)
        fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
