#!/usr/bin/env python
"""Chiral phosphoric acid catalysts in one frame, with the enantioselectivity of 1,075 reactions.

The series is the high-throughput screen of Zahrt, Henle, Rose, Wang, Darrow and Denmark,
Science 363 (2019) eaau5631: 43 BINOL-derived chiral phosphoric acids, each run against the
25 combinations of 5 N-acyl imines and 5 thiols, forming N,S-acetals.  Every reaction carries its
enantiomeric excess, and the published target is the free-energy difference between the two
diastereomeric transition states, ``ddG = R T ln((1 + ee) / (1 - ee))`` at 298.15 K.  That is a
difference of two rate constants, so this is the first series in the repository whose activity
is a kinetic selectivity rather than a binding constant or a computed property.

Why the series fits the method.  Every catalyst is the same phosphoric acid on the same
1,1'-binaphthyl (or octahydro-binaphthyl) scaffold, carrying one substituent at both 3 and 3'
positions.  The acid is the reaction centre, so the frame is given by construction, as the
benzene ring was for the Hammett constants: phosphorus at the origin, the two ring oxygens
defining the x axis, the acid oxygens pointing along +z.  The substituents sit around the +z
pocket where the imine and the thiol meet, and a model of the selectivity is a map of where bulk
and charge in that pocket favour one face of the imine.

The source files
----------------
The structures are the fragments the Science paper's supplementary material distributes: two
scaffolds (``i``, the binaphthyl phosphate; ``vi``, its 5,5',6,6',7,7',8,8'-octahydro form),
38 substituents as radicals, 5 imines and 5 thiols, all as 2D ChemDraw files, with the 1,075
measured enantiomeric excesses in one table whose row names are
``<substituent>_<scaffold>_<imine>_<thiol>``.  This script reads them from the public copy in
the DeepReac repository (github.com/bm2-lab/DeepReac, ``Data/DatasetC``, which redistributes the
Science data files beside its own code under the Apache 2.0 licence).  The data files
themselves are the paper's supplementary material, so the status of the copy built here is
**unclear** in the sense of ``docs/public_3dqsar_datasets.md``: the facts are the paper's,
redistribution is customary in the field but not explicitly permitted, and this script rebuilds
every file from the original distribution, whose SHA-256 it records.

What is built
-------------
Each catalyst is assembled from its scaffold and two copies of its substituent, bonded at the
3 and 3' carbons (the ring carbon next to the phosphate oxygen that is not the biaryl carbon).
The catalyst is then embedded with ETKDG, every conformer is minimised with MMFF94 and the
lowest one is kept.  One catalyst (181_i) has two pentafluorosulfanyl groups, which no force
field of RDKit holds: its conformers are found with CF3 in place of each SF5 group, and each
group is then set to a regular octahedron (:func:`sf5_stand_in`).  The scaffold has an axial
stereocentre that a 2D file does not carry and
that ETKDG assigns at random; every conformer is reflected, where needed, so that the dihedral
C2-C1-C1'-C2' of the biaryl bond is positive in every catalyst.  Whether that is (R) or (S)
is not known from the source; what matters is that the 43 catalysts are the same enantiomer,
as they were in the screen, so that the sign of the selectivity means the same thing for all.

The frame.  The phosphorus at the origin; z along the bisector of the two ring oxygens, from
their midpoint through the phosphorus, so that the P=O and P-OH oxygens point to +z; x along
the line from one ring oxygen to the other, orthogonalised; y completing the right-handed set.
The catalyst has a C2 axis along z, so the frame has one arbitrary choice: which ring oxygen is
on +x.  The single-conformer file fixes it by a convention (the hydroxyl oxygen on +y; the
two acid oxygens lie in the yz plane, so the x sign cannot be read from them).  The
symmetrised file writes both images (the frame and its rotation by 180 degrees about z) as two
conformers of each catalyst, so that the ensemble mode of the field set averages them and the
field has the symmetry of the molecule; this also averages the P=O and P-OH oxygens, which
the acid exchanges on a time scale far shorter than the reaction.

Outputs (``examples/denmark/``)
-------------------------------
    catalysts.sdf        43 catalysts in frame, one conformer each
    catalysts_sym.sdf    the same, with the two C2 images as conformers of one record name
    imines.sdf           the 5 imines, 3D, lowest MMFF conformer, not in any frame
    thiols.sdf           the 5 thiols, likewise
    reactions.csv        the 1,075 reactions: catalyst, imine, thiol, ee and ddG in kcal/mol
    README.md            the provenance, the files and the reading of the numbers

Usage
-----
    python examples/make_denmark_examples.py --fetch     # download the source files, then build
    python examples/make_denmark_examples.py             # build from an existing upstream copy
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from make_hammett_examples import regular_sf5  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT_DIR = os.path.join(HERE, "denmark")
UPSTREAM = os.path.join(OUT_DIR, "upstream")
RAW = "https://raw.githubusercontent.com/bm2-lab/DeepReac/master/Data/DatasetC"

#: the 50 fragment files and the table, as the distribution names them
SCAFFOLDS = ("i", "vi")
SUBSTITUENTS = ("1", "5", "7", "29", "61", "71", "72", "73", "76", "87", "99", "144", "145", "147",
                "157", "166", "181", "182", "202", "205", "207", "210", "223", "229", "230", "242",
                "245", "246", "249", "251", "253", "262", "276", "286", "328", "365", "371", "382")
IMINES = ("01", "02", "03", "04", "05")
THIOLS = ("A", "B", "C", "D", "E")
TABLE = "dataset_D.csv"

#: SHA-256 over the 51 source files in path order, so that the set has one identity.  A fetch
#: that gives another digest is reported, not refused: the distribution may have been corrected.
SOURCE_DIGEST = "77261b1db80084e30b167ebf1c7b1ff83deed99253353011b319dfe54d689423"

#: the published conversion: R in kcal/(mol K) and the temperature of the screen
R_KCAL = 1.98720e-3
T_K = 298.15

SOURCE = ("Zahrt, Henle, Rose, Wang, Darrow & Denmark, Science 363 (2019) eaau5631, "
          "supplementary data as redistributed at github.com/bm2-lab/DeepReac (Data/DatasetC); "
          "catalysts assembled from the scaffold and substituent fragments, embedded with RDKit "
          "ETKDG + MMFF94, lowest conformer, one axial configuration, phosphorus at the origin "
          "and the acid oxygens on +z (examples/make_denmark_examples.py)")


# ------------------------------------------------------------------------------- the download


def wanted_files() -> List[str]:
    out = [TABLE]
    out += [f"sdf/{s}.sdf" for s in SCAFFOLDS]
    out += [f"sdf/{s}.sdf" for s in SUBSTITUENTS]
    out += [f"sdf/{s}.sdf" for s in IMINES]
    out += [f"sdf/{s}.sdf" for s in THIOLS]
    return out


def fetch_upstream(force: bool = False) -> str:
    """Download the source files into ``upstream/`` and return the digest of the whole set."""
    os.makedirs(os.path.join(UPSTREAM, "sdf"), exist_ok=True)
    paths = wanted_files()
    print(f"  {len(paths)} source files")
    for i, path in enumerate(paths, 1):
        dest = os.path.join(UPSTREAM, path.replace("/", os.sep))
        if os.path.isfile(dest) and not force:
            continue
        with urllib.request.urlopen(f"{RAW}/{path}", timeout=120) as fh:   # noqa: S310 - fixed https
            blob = fh.read()
        with open(dest, "wb") as out:
            out.write(blob)
        if i % 10 == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    digest = upstream_digest(paths)
    if digest == SOURCE_DIGEST:
        print(f"  digest of the set: {digest} (as recorded)")
    else:
        print(f"  digest of the set: {digest}")
        print(f"  WARNING: the recorded digest is {SOURCE_DIGEST}.")
        print("  The source has changed since this script was written, so the numbers in the "
              "manual were produced from other files than these.")
    return digest


def upstream_digest(paths: Optional[Sequence[str]] = None) -> str:
    """One SHA-256 over the source files, in path order, so the set has a single identity."""
    h = hashlib.sha256()
    for path in paths or wanted_files():
        dest = os.path.join(UPSTREAM, path.replace("/", os.sep))
        h.update(path.encode())
        with open(dest, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


# ------------------------------------------------------------------------------ the fragments


def read_fragment(name: str) -> Chem.Mol:
    """A fragment file as a sanitised 2D molecule; radicals stay radicals."""
    path = os.path.join(UPSTREAM, "sdf", f"{name}.sdf")
    m = Chem.MolFromMolFile(path, sanitize=False, removeHs=False)
    if m is None:
        raise ValueError(f"{path}: not readable")
    m = Chem.RemoveHs(m, sanitize=False)
    Chem.SanitizeMol(m)
    return m


def fragment_smiles(name: str) -> str:
    return Chem.MolToSmiles(read_fragment(name))


def position_3_carbons(scaffold: Chem.Mol) -> List[int]:
    """The two ring carbons that carry the 3,3'-substituents.

    Position 2 is the aromatic carbon bonded to a phosphate oxygen; its ring neighbours are the
    biaryl carbon (three heavy neighbours) and position 3 (two heavy neighbours, one hydrogen).
    """
    out = []
    for a in scaffold.GetAtoms():
        if a.GetSymbol() != "O":
            continue
        nbrs = [n for n in a.GetNeighbors()]
        if not any(n.GetSymbol() == "P" for n in nbrs):
            continue
        aryl = [n for n in nbrs if n.GetIsAromatic()]
        if len(aryl) != 1:
            continue
        c2 = aryl[0]
        for c in c2.GetNeighbors():
            if c.GetIsAromatic() and c.GetIdx() != a.GetIdx() and c.GetDegree() == 2 \
                    and c.GetTotalNumHs() == 1:
                out.append(c.GetIdx())
    if len(out) != 2:
        raise ValueError(f"expected two 3,3' carbons on the scaffold, found {out}")
    return sorted(out)


def assemble(scaffold: Chem.Mol, substituent: Chem.Mol) -> Chem.Mol:
    """The catalyst: the scaffold with a copy of the substituent bonded at each 3 and 3' carbon.

    The substituent is a radical whose radical atom is the attachment point; the new bond
    replaces the ring hydrogen of the scaffold and the unpaired electron of the substituent.
    """
    radical = [a.GetIdx() for a in substituent.GetAtoms() if a.GetNumRadicalElectrons() == 1]
    if len(radical) != 1:
        raise ValueError("the substituent must have exactly one radical atom")
    sites = position_3_carbons(scaffold)
    mol = Chem.RWMol(scaffold)
    for site in sites:
        offset = mol.GetNumAtoms()
        mol.InsertMol(substituent)
        head = mol.GetAtomWithIdx(offset + radical[0])
        head.SetNumRadicalElectrons(0)
        head.SetNoImplicit(False)
        mol.AddBond(site, offset + radical[0], Chem.BondType.SINGLE)
        ring = mol.GetAtomWithIdx(site)
        ring.SetNumExplicitHs(0)
        ring.SetNoImplicit(False)
    out = mol.GetMol()
    for a in out.GetAtoms():
        a.SetNumRadicalElectrons(0)
    Chem.SanitizeMol(out)
    return Chem.MolFromSmiles(Chem.MolToSmiles(out))


# --------------------------------------------------------------------------- the conformation


def biaryl_dihedral_atoms(mol: Chem.Mol) -> Tuple[int, int, int, int]:
    """(C2, C1, C1', C2'): the oxygen-bearing carbons and the biaryl carbons of the scaffold."""
    p = [a for a in mol.GetAtoms() if a.GetSymbol() == "P"]
    if len(p) != 1:
        raise ValueError("one phosphorus expected")
    c2s = []
    for o in p[0].GetNeighbors():
        aryl = [n for n in o.GetNeighbors() if n.GetIsAromatic()]
        if len(aryl) == 1:
            c2s.append(aryl[0])
    if len(c2s) != 2:
        raise ValueError("two phosphate ring oxygens expected")
    a2, b2 = c2s
    a1 = b1 = None
    for n in a2.GetNeighbors():
        for m in b2.GetNeighbors():
            if n.GetIdx() != a2.GetIdx() and m.GetIdx() != b2.GetIdx() and \
                    mol.GetBondBetweenAtoms(n.GetIdx(), m.GetIdx()) is not None:
                a1, b1 = n, m
    if a1 is None:
        raise ValueError("no biaryl bond found between the two 2-carbons")
    return a2.GetIdx(), a1.GetIdx(), b1.GetIdx(), b2.GetIdx()


def biaryl_dihedral(mol: Chem.Mol, conf_id: int = -1) -> float:
    i, j, k, l = biaryl_dihedral_atoms(mol)
    return float(rdMolTransforms.GetDihedralDeg(mol.GetConformer(conf_id), i, j, k, l))


def sf5_stand_in(mol: Chem.Mol) -> Tuple[Optional[Chem.Mol], List[int]]:
    """The molecule with a CF3 group in place of every SF5 group, and for every atom of that
    stand-in the index of the same atom in `mol`; (None, []) for a molecule without the group.

    No force field of RDKit holds an SF5 group.  MMFF94 has no parameters for hexavalent
    sulfur.  UFF, which was used for this catalyst until 14 November 2026, moved the fluorines
    6 to 8 A away from the sulfur, put two of them on one point and opened the biaryl dihedral
    to 71 degrees, where the other catalysts of the scaffold have 54 to 59.  So the conformers
    are found for the stand-in, with the force field of the other catalysts, and the group
    itself is then built by a rule (:func:`make_hammett_examples.regular_sf5`).  The stand-in
    has the same ring carbon, the same atom on it and three of the five fluorines.
    """
    sulfurs, dropped = [], []
    for s in mol.GetAtoms():
        if s.GetSymbol() != "S" or s.GetDegree() != 6:
            continue
        fluorines = sorted(n.GetIdx() for n in s.GetNeighbors() if n.GetSymbol() == "F")
        if len(fluorines) == 5:
            sulfurs.append(s.GetIdx())
            dropped += fluorines[3:]
    if not sulfurs:
        return None, []
    rw = Chem.RWMol(mol)
    for s in sulfurs:
        rw.GetAtomWithIdx(s).SetAtomicNum(6)
    for a in sorted(dropped, reverse=True):      # from the last atom, so the others keep their order
        rw.RemoveAtom(a)
    stand_in = rw.GetMol()
    Chem.SanitizeMol(stand_in)
    return stand_in, [a for a in range(mol.GetNumAtoms()) if a not in set(dropped)]


def embed_lowest(mol: Chem.Mol, n: int = 30, seed: int = 2026,
                 same_axis: bool = True) -> Tuple[Chem.Mol, float, int]:
    """AddHs, embed `n` conformers, minimise each with MMFF94, keep the lowest.

    With `same_axis` every conformer whose biaryl dihedral is negative is reflected first, so
    the lowest conformer has a positive dihedral whatever the embedding produced; a reflection
    does not change an MMFF energy.  A molecule with an SF5 group is embedded and minimised as
    its stand-in with CF3 (:func:`sf5_stand_in`), and the energy returned is that of the
    stand-in.  Returns (molecule with one conformer, energy, conformers).
    """
    mh = Chem.AddHs(mol)
    # the dihedral atoms are found on the sanitised graph: setting MMFF up rewrites the
    # aromaticity flags in place, so the search would fail after it
    dihedral = biaryl_dihedral_atoms(mh) if same_axis else None
    stand_in, kept = sf5_stand_in(mh)
    work = mh if stand_in is None else stand_in
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.useSmallRingTorsions = True
    params.pruneRmsThresh = -1.0
    ids = list(AllChem.EmbedMultipleConfs(work, numConfs=int(n), params=params))
    if not ids:
        params.useRandomCoords = True
        ids = list(AllChem.EmbedMultipleConfs(work, numConfs=int(n), params=params))
    if not ids:
        raise RuntimeError(f"no conformer could be embedded for {Chem.MolToSmiles(mol)}")
    props = AllChem.MMFFGetMoleculeProperties(work)
    # UFF stands in where MMFF94 has no parameters; no catalyst of the series needs it now
    force_field = "MMFF94" if props is not None else "UFF"
    energies = []
    for cid in ids:
        if props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(work, props, confId=cid)
        else:
            ff = AllChem.UFFGetMoleculeForceField(work, confId=cid)
        ff.Minimize(maxIts=2000)
        energies.append(float(ff.CalcEnergy()))
    if stand_in is not None:
        # the conformers of the stand-in, on the atoms of the catalyst itself; the sulfur and
        # the fluorines are placed by `regular_sf5` below, which needs the ring atoms only
        for cid in ids:
            pos = np.zeros((mh.GetNumAtoms(), 3))
            pos[kept] = np.array(work.GetConformer(cid).GetPositions(), float)
            conf = Chem.Conformer(mh.GetNumAtoms())
            for atom, p in enumerate(pos):
                conf.SetAtomPosition(atom, p.tolist())
            conf.SetId(cid)
            mh.AddConformer(conf, assignId=False)
        force_field += " on the stand-in with CF3, SF5 set to a regular octahedron"
    mh.SetProp("force_field", force_field)
    Chem.SanitizeMol(mh)                 # restores the aromaticity MMFF set-up altered
    if same_axis:
        i, j, k, l = dihedral
        for cid in ids:
            if rdMolTransforms.GetDihedralDeg(mh.GetConformer(cid), i, j, k, l) < 0:
                conf = mh.GetConformer(cid)
                pos = np.array(conf.GetPositions(), float)
                pos[:, 0] *= -1.0
                for atom, p in enumerate(pos):
                    conf.SetAtomPosition(atom, p.tolist())
    if stand_in is not None:
        for cid in ids:
            regular_sf5(mh, cid)
    best = int(np.argmin(energies))
    keep = Chem.Mol(mh)
    keep.RemoveAllConformers()
    keep.AddConformer(Chem.Conformer(mh.GetConformer(ids[best])), assignId=True)
    keep.SetProp("force_field", force_field)
    return keep, float(energies[best]), len(ids)


# ------------------------------------------------------------------------------------ the frame


def frame_atoms(mol: Chem.Mol) -> Tuple[int, int, int, int]:
    """(P, ring oxygen a, ring oxygen b, hydroxyl oxygen)."""
    p = [a for a in mol.GetAtoms() if a.GetSymbol() == "P"][0]
    ring, acid = [], []
    for o in p.GetNeighbors():
        if o.GetSymbol() != "O":
            continue
        if any(n.GetIsAromatic() for n in o.GetNeighbors()):
            ring.append(o.GetIdx())
        else:
            acid.append(o)
    if len(ring) != 2 or len(acid) != 2:
        raise ValueError("a phosphoric acid diester with two ring oxygens was expected")
    oh = [o.GetIdx() for o in acid if any(n.GetSymbol() == "H" for n in o.GetNeighbors())]
    if len(oh) != 1:
        raise ValueError("one P-OH oxygen expected")
    return p.GetIdx(), ring[0], ring[1], oh[0]


def place_in_frame(mol: Chem.Mol) -> Chem.Mol:
    """P at the origin, the acid oxygens on +z, the ring oxygens along x, the hydroxyl on +x."""
    ip, ia, ib, ioh = frame_atoms(mol)
    conf = mol.GetConformer()
    pos = np.array(conf.GetPositions(), float)
    pos -= pos[ip]
    mid = 0.5 * (pos[ia] + pos[ib])
    z = -mid / np.linalg.norm(mid)
    x = pos[ib] - pos[ia]
    x -= np.dot(x, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.vstack([x, y, z])
    pos = pos @ R.T
    # the acid oxygens lie in the yz plane, so the hydroxyl's y sign is what the C2 rotation
    # changes; put it on +y
    if pos[ioh, 1] < 0:
        pos[:, 0] *= -1.0
        pos[:, 1] *= -1.0
    out = Chem.Mol(mol)
    c = out.GetConformer()
    for i, p in enumerate(pos):
        c.SetAtomPosition(i, p.tolist())
    return out


def c2_images(mol: Chem.Mol) -> List[Chem.Mol]:
    """The molecule and its rotation by 180 degrees about z, with every property carried over."""
    base = np.array(mol.GetConformer().GetPositions(), float)
    out = []
    for s, label in ((1.0, "C2+"), (-1.0, "C2-")):
        m = Chem.Mol(mol)
        conf = m.GetConformer()
        for i, p in enumerate(base * np.array([s, s, 1.0])):
            conf.SetAtomPosition(i, p.tolist())
        m.SetProp("symmetry_image", label)
        out.append(m)
    return out


# ---------------------------------------------------------------------------------- the table


def read_table() -> List[Dict[str, object]]:
    """The 1,075 reactions with their enantiomeric excess and its free-energy form."""
    rows = []
    with open(os.path.join(UPSTREAM, TABLE), encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            name = r["name"].strip()
            sub, scaf, imine, thiol = name.split("_")
            ee = float(r["stereo"])
            rows.append({"reaction": name, "catalyst": f"{sub}_{scaf}", "substituent": sub,
                         "scaffold": scaf, "imine": imine, "thiol": thiol,
                         "ee_percent": ee, "ddG_kcal": ddg_from_ee(ee)})
    return rows


def ddg_from_ee(ee_percent: float, temperature: float = T_K) -> float:
    """``R T ln((1 + ee) / (1 - ee))`` in kcal/mol; the published form of the selectivity."""
    e = float(ee_percent) / 100.0
    if not -1.0 < e < 1.0:
        raise ValueError(f"an enantiomeric excess of {ee_percent} % has no finite free energy")
    return R_KCAL * temperature * math.log((1.0 + e) / (1.0 - e))


# ---------------------------------------------------------------------------------- the build


def build_catalysts(n_conformers: int = 30, seed: int = 2026,
                    names: Optional[Sequence[str]] = None) -> List[Chem.Mol]:
    scaffolds = {s: read_fragment(s) for s in SCAFFOLDS}
    subs = {s: read_fragment(s) for s in SUBSTITUENTS}
    out = []
    t0 = time.time()
    for k, name in enumerate(names, 1):
        sub, scaf = name.split("_")
        mol = assemble(scaffolds[scaf], subs[sub])
        mh, energy, tried = embed_lowest(mol, n=n_conformers, seed=seed)
        mh = place_in_frame(mh)
        mh.SetProp("_Name", name)
        mh.SetProp("catalyst", name)
        mh.SetProp("substituent", sub)
        mh.SetProp("scaffold", scaf)
        mh.SetProp("smiles", Chem.MolToSmiles(Chem.RemoveHs(mh)))
        mh.SetDoubleProp("mmff_energy", energy)
        mh.SetIntProp("conformers_tried", tried)
        mh.SetDoubleProp("biaryl_dihedral", biaryl_dihedral(mh))
        mh.SetProp("source", SOURCE)
        out.append(mh)
        print(f"    {k:2d}/{len(names)} {name:8s} {mh.GetNumHeavyAtoms():3d} heavy atoms, "
              f"{tried} conformers, {energy:9.2f} kcal/mol, dihedral {biaryl_dihedral(mh):+6.1f} "
              f"({time.time() - t0:.0f} s)", flush=True)
    return out


def build_substrates(names: Sequence[str], labels: Sequence[str], kind: str,
                     n_conformers: int = 20, seed: int = 2026) -> List[Chem.Mol]:
    out = []
    for name, label in zip(names, labels):
        mol = read_fragment(name)
        mh, energy, tried = embed_lowest(mol, n=n_conformers, seed=seed, same_axis=False)
        mh.SetProp("_Name", label)
        mh.SetProp(kind, label)
        mh.SetProp("smiles", Chem.MolToSmiles(Chem.RemoveHs(mh)))
        mh.SetDoubleProp("mmff_energy", energy)
        mh.SetProp("source", SOURCE)
        out.append(mh)
    return out


def write_sdf(path: str, mols: Sequence[Chem.Mol]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    w = Chem.SDWriter(path)
    for m in mols:
        w.write(m)
    w.close()


def write_reactions(path: str, rows: Sequence[Dict[str, object]]) -> None:
    cols = ["reaction", "catalyst", "substituent", "scaffold", "imine", "thiol", "ee_percent",
            "ddG_kcal"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] if c != "ddG_kcal" else f"{r[c]:.4f}" for c in cols])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the source files first")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--conformers", type=int, default=30, help="ETKDG conformers tried per catalyst")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--only", type=int, default=0, help="build only the first N catalysts (a test)")
    a = ap.parse_args(argv)

    if a.fetch:
        print("fetching the source files")
        fetch_upstream(force=a.force)
    elif not os.path.isfile(os.path.join(UPSTREAM, TABLE)):
        raise SystemExit(f"{UPSTREAM} has no source files; run with --fetch")
    else:
        d = upstream_digest()
        print(f"source digest {d}" + (" (as recorded)" if d == SOURCE_DIGEST else " (NOT as recorded)"))

    rows = read_table()
    catalysts = sorted({r["catalyst"] for r in rows},
                       key=lambda s: (int(s.split("_")[0]), s.split("_")[1]))
    print(f"{len(rows)} reactions, {len(catalysts)} catalysts, "
          f"{len({r['imine'] for r in rows})} imines, {len({r['thiol'] for r in rows})} thiols")
    if a.only:
        catalysts = catalysts[:a.only]

    print("building the catalysts")
    mols = build_catalysts(n_conformers=a.conformers, seed=a.seed, names=catalysts)
    os.makedirs(a.out_dir, exist_ok=True)
    write_sdf(os.path.join(a.out_dir, "catalysts.sdf"), mols)
    images = []
    for m in mols:
        pair = c2_images(m)
        for im in pair:
            im.SetProp("conformer_energies", ",".join(f"{m.GetDoubleProp('mmff_energy'):.4f}"
                                                      for _ in pair))
        images.extend(pair)
    write_sdf(os.path.join(a.out_dir, "catalysts_sym.sdf"), images)

    print("building the substrates")
    imines = build_substrates(IMINES, ["1", "2", "3", "4", "5"], "imine", seed=a.seed)
    thiols = build_substrates(THIOLS, list(THIOLS), "thiol", seed=a.seed)
    write_sdf(os.path.join(a.out_dir, "imines.sdf"), imines)
    write_sdf(os.path.join(a.out_dir, "thiols.sdf"), thiols)
    write_reactions(os.path.join(a.out_dir, "reactions.csv"), rows)
    y = np.array([r["ddG_kcal"] for r in rows])
    print(f"wrote {len(mols)} catalysts, {len(imines)} imines, {len(thiols)} thiols and "
          f"{len(rows)} reactions to {a.out_dir}")
    print(f"ddG: mean {y.mean():.3f}, sd {y.std():.3f}, range {y.min():.3f} to {y.max():.3f} kcal/mol")
    return 0


if __name__ == "__main__":
    sys.exit(main())
