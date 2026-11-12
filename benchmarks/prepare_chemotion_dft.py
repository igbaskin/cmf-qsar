#!/usr/bin/env python
"""
A density-functional infrared spectrum for a part of the compounds of the Chemotion IR set.

`prepare_chemotion_xtb.py` computes a GFN2-xTB spectrum for 1,645 compounds.  A measured
spectrum finds its compound in that library in few cases (`run_spectra_computed.py`), and the
probable reason is the level of theory.  This script computes the spectra of the small
compounds again with a density functional, so that the two levels can be compared on the same
compounds, from the same conformer.

What is computed
----------------
Every compound of the GFN2-xTB library with at most ``--max-atoms`` atoms (25 by default) is
taken.  ORCA optimises it at B97-3c, starting from the geometry that xtb found, and computes
the harmonic wavenumbers and the infrared intensities of one molecule in the gas phase.  A
compound that ends with an imaginary frequency is moved along that mode and optimised once
more (`cmfqsar.orca.retry_band`).

The driver is the one of plan item SP2 (`cmfqsar.orca.run_band`), which optimises a molecule,
computes its frequencies and reads the band of one bond.  This script gives it the first bond
of the molecule and does not use the band: the whole spectrum is read from the output of the
job.

The result is `benchmarks/cache/chemotion_ir/computed_b97-3c.json`, in the layout of the
GFN2-xTB file, with wavenumbers that are not scaled.  The jobs are kept beside it, so a
second run computes only what is missing.  Nothing is written into the repository.

What it costs
-------------
The 231 compounds took 13.5 hours with 14 jobs at a time on the author's machine, which is
164 hours of one processor core (ORCA runs one core for a job here).  A compound takes 43
minutes on average and the longest took 4.8 hours.  One compound, the diyne
``OCC#CC#Cc1ccc(cc1)Br``, had no converged geometry after 190 steps and was stopped by
hand; without that its job runs to the limit of 260 steps and is then reported as failed.

Usage
-----
    python benchmarks/prepare_chemotion_dft.py --jobs 14
    python benchmarks/prepare_chemotion_dft.py --limit 4 --jobs 4       # a trial
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import prepare_chemotion_ir as PREP  # noqa: E402
import prepare_chemotion_xtb as XTB  # noqa: E402

LEVEL = "B97-3c TightSCF"
MAX_ATOMS = 25
OUT = os.path.join(PREP.CACHE, "computed_b97-3c.json")
WORK = os.path.join(PREP.CACHE, "orca_work_b97-3c")


def log(msg: str) -> None:
    print(msg, flush=True)


def xtb_geometry(index: int, work: str = XTB.WORK):
    """The molecule of a compound with the geometry that xtb found for it: the bond table of
    the start geometry and the coordinates of the optimisation (of the repeated one when that
    was kept)."""
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    from cmfqsar.orca import read_xyz

    mol = Chem.MolFromMolFile(XTB.start_path(index, work), removeHs=False)
    if mol is None:
        raise RuntimeError("the start geometry is not read")
    job = os.path.join(work, f"{index:05d}_c{index:05d}")
    with open(os.path.join(job, "status.json")) as fh:
        status = json.load(fh)
    place = os.path.join(job, "retry") if status.get("retried") else job
    symbols, coords = read_xyz(os.path.join(place, "xtbopt.xyz"))
    if [s.upper() for s in symbols] != [a.GetSymbol().upper() for a in mol.GetAtoms()]:
        raise RuntimeError("the xtb geometry lists different atoms")
    conf = mol.GetConformer()
    for a, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(a, Point3D(float(x), float(y), float(z)))
    return mol


def one_spectrum(index: int, smiles: str, cfg, xtb_work: str = XTB.WORK) -> dict:
    """The density-functional stick spectrum of one compound, or the error that stopped it."""
    from cmfqsar import orca
    from cmfqsar import spectrum as SP

    t0 = time.time()
    row = {"index": index, "smiles": smiles}
    try:
        mol = xtb_geometry(index, xtb_work)
        bond = mol.GetBondWithIdx(0)
        pair = (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        first = orca.run_band(mol, f"c{index:05d}", index, cfg, pair)
        if first.mol is None:
            raise RuntimeError(first.error)
        best = orca.retry_band(first, cfg, pair) if first.n_imaginary else first
        path = os.path.join(cfg.workdir, best.mol.GetProp("qm_dir"),
                            best.mol.GetProp("qm_job") + ".out")
        sticks = SP.orca_sticks(path, scale=1.0)
    except Exception as exc:               # a compound that fails is reported
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["seconds"] = round(time.time() - t0, 1)
        return row
    row.update({"atoms": mol.GetNumAtoms(),
                "wavenumbers": [round(float(x), 2) for x in sticks.x],
                "intensities": [round(float(x), 5) for x in sticks.y],
                "n_imaginary": int(best.n_imaginary), "retried": best is not first,
                "reused": bool(first.reused), "seconds": round(time.time() - t0, 1)})
    return row


def compute(out: str = OUT, work: str = WORK, jobs: int = 8, limit: int = 0,
            max_atoms: int = MAX_ATOMS, level: str = LEVEL) -> dict:
    """Compute the spectra of the small compounds and write the file; returns its head."""
    from cmfqsar import orca

    xhead, xspectra = XTB.load()
    todo: List[Tuple[int, str]] = sorted(
        (r["index"], s) for s, r in xspectra.items() if r["atoms"] <= max_atoms)
    if limit:
        todo = todo[:limit]
    cfg = orca.Config()
    cfg.level = level
    cfg.fukui = False
    cfg.workdir = work
    cfg.nprocs = 1
    os.makedirs(work, exist_ok=True)
    log(f"{len(xspectra)} compounds have a GFN2-xTB spectrum; {len(todo)} with at most "
        f"{max_atoms} atoms are computed at '{level}', {jobs} at a time")
    t0 = time.time()
    done: List[Optional[dict]] = [None] * len(todo)
    count = [0]

    def task(k: int) -> None:
        index, smiles = todo[k]
        done[k] = one_spectrum(index, smiles, cfg)
        count[0] += 1
        if "error" in done[k]:
            log(f"[{count[0]}/{len(todo)}] {index:05d} FAILED -- {done[k]['error'][:200]}")
        else:
            log(f"[{count[0]}/{len(todo)}] {index:05d} {done[k]['atoms']} atoms, "
                f"{done[k]['seconds']:.0f} s"
                + (", imaginary mode" if done[k]["n_imaginary"] else "")
                + f"; {time.time() - t0:.0f} s in all")

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(task, range(len(todo))))
    else:
        for k in range(len(todo)):
            task(k)
    good = [r for r in done if r and "error" not in r]
    failed = [r for r in done if r and "error" in r]
    head = {"what": "computed infrared stick spectra of the small compounds of the Chemotion "
                    "IR set, each optimised from its GFN2-xTB geometry, one molecule in the "
                    "gas phase",
            "source_of_the_structures": PREP.DOI, "licence_of_the_structures": PREP.LICENCE,
            "qm_level": level, "axis": "wavenumber", "unit": "cm^-1",
            "intensity_unit": "km/mol", "frequency_scale": 1.0, "max_atoms": max_atoms,
            "compounds": len(todo), "computed": len(good), "left_out": {},
            "failed": [{"index": r["index"], "smiles": r["smiles"], "error": r["error"]}
                       for r in failed],
            "with_an_imaginary_mode": sum(1 for r in good if r["n_imaginary"]),
            "seconds": round(time.time() - t0, 1),
            "job_seconds": round(float(np.sum([r["seconds"] for r in good])), 1)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(head, spectra=good), fh)
    log(f"wrote {len(good)} spectra to {out}; {len(failed)} failed; "
        f"{head['with_an_imaginary_mode']} keep an imaginary mode; {head['seconds']:.0f} s")
    return head


def load(path: str = OUT) -> Tuple[dict, Dict[str, dict]]:
    """The head of the file and its spectra by SMILES."""
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run benchmarks/prepare_chemotion_dft.py, which "
                         "computes it with ORCA (hours)")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    spectra = {r["smiles"]: r for r in d.pop("spectra")}
    return d, spectra


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8, help="ORCA jobs at a time")
    ap.add_argument("--limit", type=int, default=0, help="only the first N compounds")
    ap.add_argument("--max-atoms", type=int, default=MAX_ATOMS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    compute(a.out, WORK, a.jobs, a.limit, a.max_atoms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
