#!/usr/bin/env python
"""
The carbonyl band of substituted benzoic acids, computed with ORCA (plan item SP2)

Plan item SP2 asks for a vibrational band as a *local* target: one number on one atom, predicted
from the field around that atom as the shifts of plan item LP2 were.  The series is the one
Hammett defined his constants on, the substituted benzoic acids: every substituent of the Hammett
set of this repository (`examples/hammett/`, 95 of them), placed meta and para to the carboxyl
group, which gives 189 acids (X = H is the same acid in both series and is kept once).  The band
is the stretch of the carbonyl bond, the strongest band of an acid and the one whose shift with
the substituent is textbook chemistry: an electron donor lowers it, an acceptor raises it.

Two stages, both from this script.

1. **The structures**, from SMILES: each acid is embedded with ETKDG, optimised with MMFF94, and
   the lowest-energy conformer with the carboxyl in its usual *syn* arrangement (the hydroxyl
   hydrogen on the side of the carbonyl oxygen) is kept.  An SF5 group is then set to a regular
   octahedron, because the force field that has parameters for it (UFF) distorts it, and every
   bond length of every acid is checked against the covalent radii.  Written to
   `examples/spectra/benzoic_acids.sdf` with the substituent, its position, the Hammett constants
   and the constant that applies to the position.

2. **The band**, with ORCA (``--orca``): a geometry optimisation and a harmonic frequency
   calculation at r2SCAN-3c (`cmfqsar.orca.run_band`), the carbonyl stretch picked as the normal
   mode that moves the carbonyl carbon and oxygen apart most between 1600 and 1900 cm^-1, scaled
   by the conventional factor of the level (`cmfqsar.orca.FREQUENCY_SCALE`).  Written to
   `examples/spectra/benzoic_acids_co.sdf` with the optimised geometry, the band on the carbonyl
   carbon as ``atom.dprop.co_stretch`` and the details as molecule properties.  Every job is
   cached under `benchmarks/results/band/orca_work/`, so an interrupted run resumes.

Usage
-----
    python examples/make_band_examples.py                 # stage 1
    python examples/make_band_examples.py --orca --jobs 10   # stage 2, 10 molecules at a time
    python examples/make_band_examples.py --orca --limit 4   # a quick look at the first four

On the serial ORCA build of this machine (one core per job) benzoic acid alone takes 5 minutes.
With ten to twenty acids running at the same time one acid takes 16 to 69 minutes and about
200 MB of memory, and the whole series takes about 8 hours.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, rdMolTransforms  # noqa: E402
from rdkit.Geometry import Point3D  # noqa: E402

from make_hammett_examples import SUBSTITUENTS  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT_DIR = os.path.join(HERE, "spectra")
STRUCTURES = os.path.join(OUT_DIR, "benzoic_acids.sdf")
BANDS = os.path.join(OUT_DIR, "benzoic_acids_co.sdf")
WORKDIR = os.path.join(ROOT, "benchmarks", "results", "band", "orca_work")

#: the ring with the carboxyl group, written so that the substituent SMILES of the Hammett table
#: attaches to the carbon named by the position
SERIES = {"para": "OC(=O)c1ccc(cc1)", "meta": "OC(=O)c1cc(ccc1)"}
#: the carbonyl carbon and its oxygen of a carboxylic acid; the band is written on the first atom
CARBONYL = "[CX3](=O)[OX2H1]"
#: the hydroxyl hydrogen of the acid, with the atoms that fix its dihedral: O=C-O-H
ACID_DIHEDRAL = "O=[CX3][OX2][H]"
BAND_PROP = "co_stretch"
WINDOW = (1600.0, 1900.0)
SOURCE = ("Hammett sigma_m / sigma_p after Hansch, Leo & Taft, Chem. Rev. 1991, 91, 165, Table I, "
          "as in examples/hammett; structures X-C6H4-COOH built with RDKit ETKDG + MMFF94, the "
          "syn carboxyl conformer kept (examples/make_band_examples.py)")


def smiles_for(position: str, fragment: str) -> str:
    return SERIES[position] + fragment


def acid_dihedral(mol: Chem.Mol, conf_id: int = -1) -> Optional[float]:
    """The O=C-O-H dihedral in degrees, near 0 for the syn carboxyl; None without the group."""
    match = mol.GetSubstructMatch(Chem.MolFromSmarts(ACID_DIHEDRAL))
    if not match:
        return None
    return float(rdMolTransforms.GetDihedralDeg(mol.GetConformer(conf_id), *match))


#: bond lengths of an aryl-SF5 group in angstrom, from gas-phase structures of C6H5SF5
SF5_BONDS = {"C-S": 1.80, "S-F": 1.59}
#: a start structure is accepted when every bond length is between these multiples of the sum of
#: the covalent radii; the 187 acids without SF5 lie between 0.79 and 1.03
START_BOND_LIMITS = (0.7, 1.3)


def regular_sf5(mol: Chem.Mol) -> int:
    """Set every aryl-SF5 group of `mol` to a regular octahedron, in place; returns the number
    of groups.

    UFF, the only force field of RDKit with parameters for this group, leaves it far from an
    octahedron: S-F lengths from 1.25 to 2.17 A and F-S-F angles down to 25 degrees in these
    acids.  A quantum optimisation from such a start breaks the group apart.  The construction
    puts the sulfur on the line that bisects the ring at its carbon, one fluorine on the same
    line (axial) and four in the plane perpendicular to it (equatorial), turned by 45 degrees
    from the ring plane, which is the arrangement the gas-phase structure has.
    """
    conf = mol.GetConformer()
    r = np.asarray(conf.GetPositions(), float)
    done = 0
    for s in mol.GetAtoms():
        if s.GetSymbol() != "S":
            continue
        fluorines = [n.GetIdx() for n in s.GetNeighbors() if n.GetSymbol() == "F"]
        others = [n for n in s.GetNeighbors() if n.GetSymbol() != "F"]
        if len(fluorines) != 5 or len(others) != 1 or not others[0].GetIsAromatic():
            continue
        c = others[0]
        ring = [n.GetIdx() for n in c.GetNeighbors() if n.GetIdx() != s.GetIdx()]
        if len(ring) != 2:
            continue
        axis = r[c.GetIdx()] - 0.5 * (r[ring[0]] + r[ring[1]])
        axis /= np.linalg.norm(axis)
        normal = np.cross(r[ring[0]] - r[c.GetIdx()], r[ring[1]] - r[c.GetIdx()])
        normal -= axis * float(normal @ axis)
        normal /= np.linalg.norm(normal)
        in_plane = np.cross(normal, axis)
        centre = r[c.GetIdx()] + SF5_BONDS["C-S"] * axis
        w = 1.0 / np.sqrt(2.0)
        directions = [axis, w * (in_plane + normal), w * (in_plane - normal),
                      -w * (in_plane + normal), -w * (in_plane - normal)]
        r[s.GetIdx()] = centre
        for f, d in zip(fluorines, directions):
            r[f] = centre + SF5_BONDS["S-F"] * d
        done += 1
    if done:
        for a, (x, y, z) in enumerate(r):
            conf.SetAtomPosition(a, Point3D(float(x), float(y), float(z)))
    return done


def wrong_bonds(mol: Chem.Mol) -> List[str]:
    """The bonds of a start structure whose length is outside `START_BOND_LIMITS`."""
    from cmfqsar import orca

    return orca.changed_bonds(mol, low=START_BOND_LIMITS[0], high=START_BOND_LIMITS[1])


def build(position: str, fragment: str, n_embed: int = 20, seed: int = 2026) -> Chem.Mol:
    """X-C6H4-COOH with hydrogens: the lowest-energy force-field conformer with a syn carboxyl.

    MMFF94 is used where it has parameters for the molecule and UFF where it has not (SF5,
    SeCH3, B(OH)2, as in the Hammett set); the energy is only used to rank the conformers of one
    molecule.  A conformer whose carboxyl is anti (hydroxyl hydrogen away from the carbonyl
    oxygen) is passed over as long as a syn one exists, because the two differ in the band by
    tens of cm^-1 and the series has to be one conformer family.  An SF5 group is rebuilt as a
    regular octahedron (:func:`regular_sf5`), and a structure with a bond length far from the
    sum of the covalent radii is an error, not a start geometry.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles_for(position, fragment)))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.3
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_embed, params=params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_embed, params=params))
    if not ids:
        raise RuntimeError(f"embedding failed for {smiles_for(position, fragment)}")
    work = Chem.Mol(mol)                 # the force field re-perceives aromaticity in place
    if AllChem.MMFFHasAllMoleculeParams(Chem.Mol(mol)):
        results = AllChem.MMFFOptimizeMoleculeConfs(work, maxIters=2000)
        field = "MMFF94"
    else:
        results = AllChem.UFFOptimizeMoleculeConfs(work, maxIters=2000)
        field = "UFF"
    energies = np.array([e if (conv == 0 and e == e) else np.inf for conv, e in results], float)
    for k, cid in enumerate(ids):
        c = np.asarray(work.GetConformer(cid).GetPositions(), float)
        d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))
        if d[np.triu_indices(len(c), 1)].min() < 0.7:
            energies[k] = np.inf
    if not np.isfinite(energies).any():
        energies = np.zeros(len(ids))
        work = Chem.Mol(mol)
    order = list(np.argsort(energies, kind="stable"))
    syn = [k for k in order if np.isfinite(energies[k])
           and abs(acid_dihedral(work, ids[k]) or 0.0) < 90.0]
    pick = syn[0] if syn else order[0]
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    conf = Chem.Conformer(work.GetConformer(ids[pick]))
    conf.SetId(0)
    out.AddConformer(conf, assignId=False)
    Chem.SanitizeMol(out)
    rebuilt = regular_sf5(out)
    wrong = wrong_bonds(out)
    if wrong:
        raise RuntimeError(f"{smiles_for(position, fragment)}: the force-field structure has "
                           f"bond lengths far from the covalent radii: {', '.join(wrong)}")
    out.SetDoubleProp("ff_energy", float(energies[pick]) if np.isfinite(energies[pick]) else 0.0)
    out.SetProp("force_field", field + (", SF5 set to a regular octahedron" if rebuilt else ""))
    out.SetProp("carboxyl", "syn" if syn else "anti")
    return out


def make_structures(seed: int = 2026, n_embed: int = 20, log=print) -> List[Chem.Mol]:
    mols: List[Chem.Mol] = []
    for label, frag, sm, sp, pi in SUBSTITUENTS:
        positions = ("para", "meta") if frag else ("none",)
        for pos in positions:
            m = build("para" if pos == "none" else pos, frag, n_embed=n_embed, seed=seed)
            name = label if pos == "none" else f"{pos}-{label}"
            m.SetProp("_Name", name)
            m.SetProp("substituent", label)
            m.SetProp("position", pos)
            m.SetProp("smiles", Chem.MolToSmiles(Chem.RemoveHs(m)))
            m.SetDoubleProp("sigma_m", sm)
            m.SetDoubleProp("sigma_p", sp)
            m.SetDoubleProp("sigma", 0.0 if pos == "none" else (sp if pos == "para" else sm))
            if pi is not None:
                m.SetDoubleProp("pi", pi)
            m.SetProp("source", SOURCE)
            mols.append(m)
            if m.GetProp("carboxyl") != "syn":
                log(f"  {name}: no syn carboxyl conformer among the embeddings; anti kept")
    return mols


def write_sdf(path: str, mols: Sequence[Chem.Mol]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    w = Chem.SDWriter(path)
    for m in mols:
        w.write(m)
    w.close()


def read_sdf(path: str) -> List[Chem.Mol]:
    return [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]


def run_orca(structures: str = STRUCTURES, out: str = BANDS, workdir: str = WORKDIR,
             jobs: int = 10, limit: int = 0, level: Optional[str] = None, log=print,
             retry: bool = True) -> int:
    """Stage 2: the band of every acid, cached job by job; returns the number of failures."""
    from cmfqsar import orca

    mols = read_sdf(structures)
    names = [m.GetProp("_Name") for m in mols]
    if limit:
        mols, names = mols[:limit], names[:limit]
    cfg = orca.Config()
    cfg.level = level or orca.FAST_LEVEL
    cfg.fukui = False
    cfg.workdir = workdir
    if orca.orca_has_mpi(cfg.orca):
        cfg.nprocs = 1
    bonds = [orca.bond_from_smarts(m, CARBONYL) for m in mols]
    log(f"{len(mols)} acids from {os.path.relpath(structures, ROOT)}; level '{cfg.level}', "
        f"optimisation and frequencies, {jobs} molecule(s) at a time -> "
        f"{os.path.relpath(out, ROOT)}; jobs cached under {os.path.relpath(workdir, ROOT)}")
    t0 = time.time()
    outcomes = orca.band_positions(mols, names, cfg, bonds, prop=BAND_PROP, low=WINDOW[0],
                                   high=WINDOW[1], jobs=jobs, log=log)
    again = [k for k, oc in enumerate(outcomes) if oc.mol is not None and oc.n_imaginary]
    if retry and again:
        # a stationary point with an imaginary frequency is a saddle point: one step along
        # that mode and a second optimisation, in a directory of its own
        log(f"{len(again)} acids ended with an imaginary frequency; each is moved along that "
            f"mode and optimised once more")

        def second(k):
            return k, orca.retry_band(outcomes[k], cfg, bonds[k], prop=BAND_PROP,
                                      low=WINDOW[0], high=WINDOW[1])

        with ThreadPoolExecutor(max_workers=max(jobs, 1)) as ex:
            for k, oc in ex.map(second, again):
                first = outcomes[k]
                if oc is first:
                    log(f"  {names[k]}: the second optimisation is not better; the first "
                        f"result is kept ({first.n_imaginary} imaginary)")
                else:
                    log(f"  {names[k]}: {first.frequency:.1f} -> {oc.frequency:.1f} cm^-1, "
                        f"{oc.n_imaginary} imaginary after the second optimisation")
                outcomes[k] = oc
    done = [oc for oc in outcomes if oc.mol is not None]
    failed = [oc for oc in outcomes if oc.mol is None]
    for oc in done:
        d = acid_dihedral(oc.mol)
        oc.mol.SetProp("carboxyl_after_opt",
                       "syn" if d is not None and abs(d) < 90.0 else "anti")
        oc.mol.SetDoubleProp("acid_dihedral", d if d is not None else float("nan"))
    write_sdf(out, [oc.mol for oc in done])
    anti = sum(1 for oc in done if oc.mol.GetProp("carboxyl_after_opt") != "syn")
    imag = sum(1 for oc in done if oc.n_imaginary)
    shares = np.array([oc.share for oc in done], float)
    moved = sum(1 for oc in done if oc.mol.HasProp(BAND_PROP + "_retry"))
    log(f"{moved} acids took the result of a second optimisation")
    log(f"{len(done)} done, {len(failed)} failed, {anti} ended anti, {imag} with an imaginary "
        f"frequency; share of the mode on the carbonyl bond {shares.min():.2f} to "
        f"{shares.max():.2f}; {(time.time() - t0) / 60:.1f} min")
    for oc in failed:
        log(f"  failed: {oc.name}: {oc.error}")
    return len(failed)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--orca", action="store_true", help="stage 2: run ORCA on the structures")
    ap.add_argument("--jobs", type=int, default=10, help="molecules at a time in stage 2")
    ap.add_argument("--limit", type=int, default=0, help="stage 2 on the first N acids only")
    ap.add_argument("--level", default=None, help="level of theory (default r2SCAN-3c)")
    ap.add_argument("--no-retry", action="store_true",
                    help="stage 2 without the second optimisation of an acid that ended "
                         "with an imaginary frequency")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args(argv)
    structures = os.path.join(a.out_dir, os.path.basename(STRUCTURES))
    bands = os.path.join(a.out_dir, os.path.basename(BANDS))
    if a.orca:
        if not os.path.exists(structures):
            raise SystemExit(f"{structures} is missing: run the script without --orca first")
        return 1 if run_orca(structures, bands, jobs=a.jobs, limit=a.limit, level=a.level,
                             retry=not a.no_retry) else 0
    mols = make_structures(seed=a.seed)
    write_sdf(structures, mols)
    positions = {p: sum(1 for m in mols if m.GetProp("position") == p)
                 for p in ("para", "meta", "none")}
    print(f"wrote {len(mols)} acids to {os.path.relpath(structures, ROOT)} "
          f"({positions['para']} para, {positions['meta']} meta, {positions['none']} unsubstituted); "
          f"atoms {min(m.GetNumAtoms() for m in mols)} to {max(m.GetNumAtoms() for m in mols)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
