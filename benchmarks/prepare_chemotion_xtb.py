#!/usr/bin/env python
"""
A computed infrared spectrum for the compounds of the Chemotion IR set (GFN2-xTB).

`prepare_chemotion_ir.py` prepares the measured spectra of the set.  This script computes a
spectrum for the compounds from their structures alone, so that a measured spectrum can be
compared with a computed library (`run_spectra_computed.py`).

What is computed
----------------
A compound is taken when its SMILES is one neutral molecule without unpaired electrons, of
the elements H, B, C, N, O, F, Si, P, S, Cl, Br, I and Se, with at most ``--max-atoms`` atoms.
Salts, mixtures, metal complexes and radicals are left out: their SMILES does not give a
geometry that can be trusted.  For every compound that is taken:

1. RDKit makes up to ten three-dimensional conformers from the SMILES and relaxes each with
   the MMFF94 force field (UFF where MMFF94 has no parameters).  The conformer of lowest
   force-field energy is kept.  This is a cheap choice of one conformer and not a search.
   The start geometries are made first, in processes of their own, and kept as MOL files.
2. xtb optimises that geometry with GFN2-xTB and computes the harmonic wavenumbers and the
   infrared intensities (`cmfqsar.xtb.run_vibrations`, one molecule in the gas phase).

The result is `benchmarks/cache/chemotion_ir/computed_xtb.json`: one stick spectrum per
compound, by its SMILES.  The jobs are kept under `xtb_work` beside it, so a second run
computes only what is missing.  Nothing is written into the repository: the structures come
from a set with the licence CC BY-SA 4.0.

Usage
-----
    python benchmarks/prepare_chemotion_xtb.py --jobs 14         # all compounds
    python benchmarks/prepare_chemotion_xtb.py --limit 20        # the first 20, as a trial
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

import prepare_chemotion_ir as PREP  # noqa: E402

ELEMENTS = frozenset(("H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Se"))
MAX_ATOMS = 100
N_CONFORMERS = 10
SEED = 42
OUT = os.path.join(PREP.CACHE, "computed_xtb.json")
WORK = os.path.join(PREP.CACHE, "xtb_work")
#: threads of one xtb job; the number of jobs at a time is the argument --jobs
THREADS_PER_JOB = 2


def log(msg: str) -> None:
    print(msg, flush=True)


def compounds(rows: Sequence[dict]) -> List[str]:
    """The SMILES of the compounds of the prepared set, each once, in a fixed order.  The
    place of a compound in this list names its job, so the list must not depend on a
    filter."""
    return sorted({r["smiles"] for r in rows})


def usable(smiles: str, max_atoms: int = MAX_ATOMS):
    """The molecule of a SMILES when a spectrum can be computed for it, and the reason when
    not: ``(mol, "")`` or ``(None, reason)``."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "the SMILES is not read"
    if len(Chem.GetMolFrags(mol)) > 1:
        return None, "more than one molecule (a salt, a solvate or a complex)"
    other = sorted({a.GetSymbol() for a in mol.GetAtoms()} - ELEMENTS)
    if other:
        return None, "an element outside the list (a metal in most cases)"
    if Chem.GetFormalCharge(mol) != 0:
        return None, "an ion"
    if Descriptors.NumRadicalElectrons(mol) > 0:
        return None, "unpaired electrons"
    if Chem.AddHs(mol).GetNumAtoms() > max_atoms:
        return None, f"more than {max_atoms} atoms"
    return mol, ""


def start_geometry(mol, n_conformers: int = N_CONFORMERS, seed: int = SEED):
    """The molecule with all hydrogens and one conformer: of up to `n_conformers` conformers
    made by RDKit and relaxed with a force field, the one of lowest force-field energy."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    m = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    # without a limit the embedding of a strained ring system (a [2.2]paracyclophane, of
    # which the set has many) tries for minutes and then gives nothing
    params.maxIterations = 30
    ids = list(AllChem.EmbedMultipleConfs(m, int(n_conformers), params))
    if not ids:
        # plain distance geometry, without the knowledge of flat rings and preferred
        # torsions that such a ring system contradicts; the force field relaxes the result
        params.useBasicKnowledge = False
        params.useExpTorsionAnglePrefs = False
        ids = list(AllChem.EmbedMultipleConfs(m, int(n_conformers), params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(m, int(n_conformers), params))
    if not ids:
        raise RuntimeError("RDKit made no three-dimensional geometry")
    try:
        if AllChem.MMFFHasAllMoleculeParams(m):
            done = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
        else:
            done = AllChem.UFFOptimizeMoleculeConfs(m, maxIters=2000)
        energies = [e for _, e in done]
    except Exception:                      # no force field at all: the first conformer
        energies = [0.0] * len(ids)
    best = ids[min(range(len(ids)), key=lambda k: energies[k])]
    one = Chem.Mol(m, False, best)
    # setting up MMFF changes the aromaticity flags in place; give RDKit's model back
    Chem.SanitizeMol(one)
    return one


def start_path(index: int, work: str) -> str:
    return os.path.join(work, "start", f"{index:05d}.mol")


def make_start(task: Tuple[int, str, str]) -> Tuple[int, str]:
    """Make the start geometry of one compound and keep it as a MOL file; returns the number
    of the compound and an error text, which is empty when the file is there.

    The geometry is made once: a second run must give xtb the same geometry, or its finished
    job would not be reused.  The function takes one argument and lives at the top of the
    module, so that a process of its own can run it.
    """
    from rdkit import Chem

    index, smiles, work = task
    path = start_path(index, work)
    if os.path.isfile(path) and Chem.MolFromMolFile(path, removeHs=False) is not None:
        return index, ""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        made = start_geometry(Chem.MolFromSmiles(smiles))
        Chem.MolToMolFile(made, path + ".part")
        os.replace(path + ".part", path)
    except Exception as exc:               # a compound without a geometry is reported
        return index, f"{type(exc).__name__}: {exc}"
    return index, ""


def make_starts(todo: Sequence[Tuple[int, str]], work: str, jobs: int) -> Dict[int, str]:
    """The start geometries of all compounds; returns the errors by the number of the
    compound.  The force field is slow and runs in Python, so the work is given to `jobs`
    processes and not to threads."""
    tasks = [(index, smiles, work) for index, smiles in todo
             if not os.path.isfile(start_path(index, work))]
    errors: Dict[int, str] = {}
    if not tasks:
        return errors
    if jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(make_start, tasks, chunksize=4))
    else:
        results = [make_start(task) for task in tasks]
    for index, error in results:
        if error:
            errors[index] = error
    return errors


def one_spectrum(index: int, smiles: str, cfg, work: str) -> dict:
    """The stick spectrum of one compound from its start geometry, or the error that stopped
    it."""
    from rdkit import Chem
    from cmfqsar import xtb

    t0 = time.time()
    row = {"index": index, "smiles": smiles}
    path = start_path(index, work)
    start = Chem.MolFromMolFile(path, removeHs=False) if os.path.isfile(path) else None
    if start is None:
        row["error"] = "the start geometry is not read"
        return row
    v = xtb.run_vibrations(start, f"c{index:05d}", index, cfg)
    row["atoms"] = start.GetNumAtoms()
    row["seconds"] = round(time.time() - t0, 1)
    if v.error:
        row["error"] = v.error
        return row
    row.update({"wavenumbers": [round(float(x), 2) for x in v.wavenumbers],
                "intensities": [round(float(x), 5) for x in v.intensities],
                "n_imaginary": int(v.n_imaginary), "reused": bool(v.reused)})
    return row


def compute(out: str = OUT, work: str = WORK, jobs: int = 8, limit: int = 0,
            max_atoms: int = MAX_ATOMS, timeout: float = 3600.0,
            threads: int = THREADS_PER_JOB) -> dict:
    """Compute the spectra of all usable compounds and write the file; returns its head."""
    from cmfqsar import xtb

    _, _, rows = PREP.load()
    names = compounds(rows)
    left_out: Dict[str, int] = {}
    todo: List[Tuple[int, str]] = []
    for index, smiles in enumerate(names, start=1):
        mol, reason = usable(smiles, max_atoms)
        if mol is None:
            left_out[reason] = left_out.get(reason, 0) + 1
        else:
            todo.append((index, smiles))
    if limit:
        todo = todo[:limit]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = str(int(threads))        # the xtb jobs inherit it
    os.environ.setdefault("OMP_STACKSIZE", "2G")
    cfg = xtb.Config(xtb=xtb.find_xtb(None), fukui=False, workdir=work, timeout=timeout)
    log(f"{len(names)} compounds, {len(todo)} to compute with {jobs} jobs at a time; "
        f"left out: {left_out}")
    t0 = time.time()
    no_start = make_starts(todo, work, jobs)
    log(f"start geometries: {len(todo) - len(no_start)} made or kept, {len(no_start)} failed, "
        f"{time.time() - t0:.0f} s")
    done: List[Optional[dict]] = [None] * len(todo)
    count = [0]

    def task(k: int) -> None:
        index, smiles = todo[k]
        if index in no_start:
            done[k] = {"index": index, "smiles": smiles, "error": no_start[index]}
        else:
            done[k] = one_spectrum(index, smiles, cfg, work)
        count[0] += 1
        if "error" in done[k]:
            log(f"[{count[0]}/{len(todo)}] {index:05d} FAILED -- {done[k]['error'][:150]}")
        elif count[0] % 50 == 0:
            log(f"[{count[0]}/{len(todo)}] {time.time() - t0:.0f} s")

    if jobs > 1:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            list(pool.map(task, range(len(todo))))
    else:
        for k in range(len(todo)):
            task(k)
    good = [r for r in done if r and "error" not in r]
    failed = [r for r in done if r and "error" in r]
    head = {"what": "computed infrared stick spectra of compounds of the Chemotion IR set, "
                    "one conformer each, one molecule in the gas phase",
            "source_of_the_structures": PREP.DOI, "licence_of_the_structures": PREP.LICENCE,
            "qm_level": xtb.vibration_level(cfg), "xtb": xtb.xtb_version(cfg.xtb),
            "axis": "wavenumber", "unit": "cm^-1", "intensity_unit": "km/mol",
            "frequency_scale": 1.0, "conformers_tried": N_CONFORMERS, "max_atoms": max_atoms,
            "compounds": len(names), "computed": len(good), "left_out": left_out,
            "failed": [{"index": r["index"], "smiles": r["smiles"], "error": r["error"]}
                       for r in failed],
            "with_an_imaginary_mode": sum(1 for r in good if r["n_imaginary"]),
            "seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(dict(head, spectra=good), fh)
    log(f"wrote {len(good)} spectra to {out}; {len(failed)} failed; "
        f"{head['with_an_imaginary_mode']} keep an imaginary mode; {head['seconds']:.0f} s")
    return head


def load(path: str = OUT) -> Tuple[dict, Dict[str, dict]]:
    """The head of the file and its spectra by SMILES."""
    if not os.path.isfile(path):
        raise SystemExit(f"{path} is missing: run benchmarks/prepare_chemotion_xtb.py, which "
                         "computes it with xtb (about an hour)")
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    spectra = {r["smiles"]: r for r in d.pop("spectra")}
    return d, spectra


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=8, help="xtb jobs at a time")
    ap.add_argument("--limit", type=int, default=0, help="only the first N usable compounds")
    ap.add_argument("--max-atoms", type=int, default=MAX_ATOMS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    compute(a.out, WORK, a.jobs, a.limit, a.max_atoms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
