#!/usr/bin/env python
"""
The AMPA / glycine-NMDA selectivity set of the 2003 selectivity-fields paper, as SDF files.

    Baskin I.I., Tikhonova I.G., Palyulin V.A., Zefirov N.S. Selectivity fields: comparative
    molecular field analysis (CoMFA) of the glycine/NMDA and AMPA receptors.
    J. Med. Chem. 46 (2003) 4063-4069.

The paper proposes to model the *difference* of two affinities directly: the resulting co-field
(the "selectivity field") shows where a change of the molecular field moves the binding
preference from one receptor to the other.  Its data are 57 tricyclic antagonists (52 training,
5 test) with inhibition constants against [3H]AMPA and [3H]glycine binding, taken from three
papers of the Florence group (refs 12-14 of the paper):

    scaffold 1   4,5-dihydro-1,2,4-triazolo[1,5-a]quinoxalin-4-ones, 2-substituent R and
                 ring substituents R1-R4 at positions 9, 8, 7, 6 (Catarzi et al. 1999, 2001)
    scaffold 2   compound 39, the N5-ethyl analogue of compound 19
    scaffold 3   pyrazolo[1,5-c]quinazolin-5(6H)-ones (Varano et al. 2002)
    scaffold 4   their 1,10b-dihydro analogues (one stereocentre, as modelled)

The three-dimensional structures are the Sybyl models of I. Tikhonova (Gasteiger-Hueckel
charges, carboxylic acids as carboxylates), which exist on the author's machine as docked poses
in the two receptor models.  This script needs that folder (``--data-root``); the SDF files it
writes are in the repository, so nobody else needs it.

What the files are
------------------
``examples/selectivity/ampa_glycine.sdf``
    the 57 compounds superimposed on the tricyclic core, which is the rule of Figure 1 of the
    paper (benzene ring, lactam, five-membered ring); the reference is compound 1.  Every record
    carries ``pKi_AMPA``, ``pKi_glycine``, their difference ``delta_AMPA_glycine`` (the paper's
    Delta, positive for AMPA-selective compounds), the published split ``subset``, the paper's
    compound number as the name (``cpd_01`` ... ``cpd_57``) and in ``paper_no``, the scaffold and
    substituents, and the Sybyl atom types and charges as per-atom property lists
    (``atom.prop.sybyl_type``, ``atom.dprop.partial_charge``) for ``--charges file``.
``examples/selectivity/docked_ampa.sdf``, ``docked_glycine.sdf``
    the same compounds as docked poses in the frame of each receptor model, and
``examples/selectivity/receptor_ampa.pdb``, ``receptor_glycine.pdb``
    the receptor models they were docked into (the AMPA ligand-binding core after PDB 1FTL and
    the homology model of the glycine site of the NR1 subunit).

Corrections to Table 1 of the paper
-----------------------------------
The modelled structures differ from the printed substituent names in four places, and the
structures are what the published models were built from, so the files follow the structures:
"imidazol-1-yl" is 1,2,4-triazol-4-yl (compounds 1 and 46); "pyrazol-1-yl" of compounds 2 and 3
is pyrrol-1-yl (compound 6 is the pyrazolyl one); "2-formyl-pyrazol-1-yl" is 3-formylpyrrol-1-yl
(4, 5, 47, 48); and scaffold 3 is drawn with a 5-carboxylic acid where the modelled compounds
carry a 5-oxo group.  Compounds 26 and 27 are the same compounds as 9 and 8, entered twice from
two papers; both copies were in the published training set, so both are kept here, each with a
``duplicate_of`` tag so that they can be dropped.

The file charges are kept as they are, but they are not all usable.  For the fifteen compounds of
scaffolds 3 and 4 they sum to one unit above the formal charge (an ester sums to +1, a
carboxylate to 0), and for compound 46 they sum to -1.74; the other 41 compounds sum to their
formal charge.  Every record carries ``file_charge_sum`` and ``formal_charge`` so that this can be
seen, and ``benchmarks/run_selectivity.py`` uses Gasteiger charges computed by RDKit for all
compounds, with the file charges as a variant.  The file of compound 46 also lacks the two C-H
hydrogens of its triazole ring; they are added here (tag ``fixed``).

Usage
-----
    python examples/make_selectivity_examples.py --data-root <folder with ampa/ and glycine/>
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolAlign

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

from cmfqsar.dataio import read_molecules  # noqa: E402
from make_sutherland_examples import _fix_for_sdf  # noqa: E402

DEFAULT_ROOT = r"C:\Users\baski\OneDrive\people\Tikhonova"
OUT = os.path.join(HERE, "selectivity")
SOURCE = ("Baskin, Tikhonova, Palyulin, Zefirov, J. Med. Chem. 46 (2003) 4063, Tables 1-2; "
          "structures: Sybyl models of I. Tikhonova (Gasteiger-Hueckel charges)")
CHARGES = "Gasteiger-Hueckel (Sybyl)"

#: Tikhonova's file id -> (paper number, scaffold, R, R1, R2, R3, R4, reference, split).
#: Substituent positions follow scaffold 1 (R1 = 9, R2 = 8, R3 = 7, R4 = 6); for scaffolds 3 and
#: 4 they are the positions that overlay those in the Figure 1 superposition (R2 = 9, R3 = 8).
#: Names are those of the modelled structures (see the module docstring).
TABLE = {
    2: (1, 1, "COOH", "H", "1,2,4-triazol-4-yl", "Cl", "H", 12, "train"),
    7: (2, 1, "COOEt", "H", "pyrrol-1-yl", "Cl", "H", 12, "train"),
    8: (3, 1, "COOH", "H", "pyrrol-1-yl", "Cl", "H", 12, "train"),
    9: (4, 1, "COOEt", "H", "3-formylpyrrol-1-yl", "Cl", "H", 12, "train"),
    10: (5, 1, "COOH", "H", "3-formylpyrrol-1-yl", "Cl", "H", 12, "train"),
    14: (6, 1, "COOH", "H", "pyrazol-1-yl", "Cl", "H", 12, "train"),
    17: (7, 1, "COOEt", "H", "NO2", "Cl", "H", 12, "train"),
    18: (8, 1, "COOH", "H", "NO2", "Cl", "H", 12, "train"),
    20: (9, 1, "COOH", "H", "NO2", "H", "H", 12, "train"),
    22: (10, 1, "COOH", "H", "NH2", "Cl", "H", 12, "train"),
    27: (11, 1, "COOEt", "H", "H", "H", "H", 13, "train"),
    28: (12, 1, "COOEt", "H", "H", "Cl", "H", 13, "train"),
    29: (13, 1, "COOEt", "H", "Cl", "H", "H", 13, "train"),
    30: (14, 1, "COOEt", "Cl", "Cl", "H", "H", 13, "train"),
    31: (15, 1, "COOEt", "H", "Cl", "Cl", "H", 13, "train"),
    33: (16, 1, "COOEt", "H", "H", "NH2", "H", 13, "train"),
    34: (17, 1, "COOEt", "H", "H", "Br", "H", 13, "train"),
    35: (18, 1, "COOH", "H", "H", "H", "H", 13, "train"),
    36: (19, 1, "COOH", "H", "H", "Cl", "H", 13, "train"),
    37: (20, 1, "COOH", "H", "Cl", "H", "H", 13, "train"),
    38: (21, 1, "COOH", "Cl", "H", "Cl", "H", 13, "train"),
    39: (22, 1, "COOH", "H", "Cl", "Cl", "H", 13, "train"),
    40: (23, 1, "COOH", "H", "H", "CF3", "H", 13, "train"),
    41: (24, 1, "COOH", "H", "H", "NO2", "H", 13, "train"),
    42: (25, 1, "COOH", "H", "H", "Br", "H", 13, "train"),
    43: (26, 1, "COOH", "H", "NO2", "H", "H", 13, "train"),
    44: (27, 1, "COOH", "H", "NO2", "Cl", "H", 13, "train"),
    47: (28, 1, "CH2OH", "H", "H", "Cl", "H", 13, "train"),
    48: (29, 1, "CH2CN", "H", "H", "Cl", "H", 13, "train"),
    49: (30, 1, "CH2COOEt", "H", "H", "Cl", "H", 13, "train"),
    50: (31, 1, "CH2COOH", "H", "H", "Cl", "H", 13, "train"),
    51: (32, 1, "CONHOH", "H", "H", "Cl", "H", 13, "train"),
    52: (33, 1, "CONHNH2", "H", "H", "Cl", "H", 13, "train"),
    53: (34, 1, "CONH2", "H", "H", "Cl", "H", 13, "train"),
    54: (35, 1, "CONH-5-tetrazolyl", "H", "H", "Cl", "H", 13, "train"),
    55: (36, 1, "CONHCH2CH2OH", "H", "H", "Cl", "H", 13, "train"),
    56: (37, 1, "CONHCH2COOCH3", "H", "H", "Cl", "H", 13, "train"),
    57: (38, 1, "CONHCH2COOH", "H", "H", "Cl", "H", 13, "train"),
    46: (39, 2, "COOH", "H", "H", "Cl", "H", 13, "train"),      # N5-ethyl
    58: (40, 3, "COOEt", "H", "H", "H", "H", 14, "train"),
    59: (41, 3, "COOH", "H", "H", "H", "H", 14, "train"),
    60: (42, 3, "COOEt", "H", "H", "Cl", "H", 14, "train"),
    61: (43, 3, "COOH", "H", "H", "Cl", "H", 14, "train"),
    65: (44, 3, "COOH", "H", "NO2", "Cl", "H", 14, "train"),
    69: (45, 3, "COOH", "H", "Cl", "Cl", "H", 14, "train"),
    71: (46, 3, "COOH", "H", "1,2,4-triazol-4-yl", "Cl", "H", 14, "train"),
    72: (47, 3, "COOEt", "H", "3-formylpyrrol-1-yl", "Cl", "H", 14, "train"),
    73: (48, 3, "COOH", "H", "3-formylpyrrol-1-yl", "Cl", "H", 14, "train"),
    79: (49, 4, "COOH", "H", "H", "H", "H", 14, "train"),
    81: (50, 4, "COOH", "H", "H", "Cl", "H", 14, "train"),
    82: (51, 4, "COOMe", "H", "Cl", "Cl", "H", 14, "train"),
    83: (52, 4, "COOH", "H", "Cl", "Cl", "H", 14, "train"),
    21: (53, 1, "COOEt", "H", "NH2", "Cl", "H", 12, "test"),
    32: (54, 1, "COOEt", "H", "H", "CF3", "H", 13, "test"),
    45: (55, 1, "COOH", "H", "Cl", "H", "NO2", 13, "test"),
    64: (56, 3, "COOEt", "H", "NO2", "Cl", "H", 14, "test"),
    80: (57, 4, "COOMe", "H", "H", "Cl", "H", 14, "test"),
}
DUPLICATES = {26: 9, 27: 8}   # paper number -> the compound it repeats
REFERENCE_NO = 1              # the molecule the others are superimposed on

#: The tricyclic core common to the four scaffolds, with any atom where they differ: carbonyl O,
#: lactam C, lactam N(H), the benzene ring, the atom that joins the benzene ring to the five-
#: membered ring (N in scaffold 1, C10b in 3 and 4), the three outer atoms of that ring (the
#: middle one carries R) and the atom it shares with the lactam ring.  Fourteen atoms; the
#: superposition fits them, which is the rule of Figure 1 of the paper.
CORE = Chem.MolFromSmarts("O=[#6]1~[#7]~[#6]2~[#6]~[#6]~[#6]~[#6]~[#6]2~[#6,#7]2~[#7,#6]~[#6]~[#7,#6]~[#6,#7]12")


def read_activities(root):
    """id -> (pKi AMPA, pKi glycine, Delta).  glyc.txt holds pKi(glycine) - pKi(AMPA)."""
    def table(path):
        out = {}
        with open(path) as fh:
            for row in csv.reader(fh):
                if row:
                    out[int(row[0])] = float(row[1])
        return out
    ampa = table(os.path.join(root, "ampa", "ampa.txt"))
    sel = table(os.path.join(root, "glycine", "glyc.txt"))
    return {i: (ampa[i], ampa[i] + sel[i], -sel[i]) for i in ampa}


def read_ligand(path):
    mols, _, _ = read_molecules(path, add_hs=False)
    if len(mols) != 1 or mols[0] is None:
        raise RuntimeError(f"{path}: expected one molecule")
    return mols[0]


def complete_hydrogens(mol, h_charge):
    """Add a hydrogen to every atom the file left with an unpaired electron.

    The Sybyl file of compound 46 (id 71) lacks the two C-H hydrogens of its triazole ring, which
    RDKit reads as two radical carbons.  They are added with coordinates and given `h_charge`,
    the charge the same hydrogens carry in compound 1; the carbons keep their file charges.
    Returns the number of hydrogens added.
    """
    bare = [a.GetIdx() for a in mol.GetAtoms() if a.GetNumRadicalElectrons() > 0]
    if not bare:
        return 0
    for i in bare:
        a = mol.GetAtomWithIdx(i)
        a.SetNumRadicalElectrons(0)
        a.SetNumExplicitHs(1)
    n_before = mol.GetNumAtoms()
    full = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=bare)
    for a in full.GetAtoms():
        if a.GetIdx() >= n_before:
            a.SetProp("_TriposAtomType", "H")
            a.SetDoubleProp("_TriposPartialCharge", h_charge)
    full.SetProp("fixed", f"{full.GetNumAtoms() - n_before} hydrogens added that the source file lacks; "
                          f"charge {h_charge:.4f} as in compound {REFERENCE_NO}")
    return full


def ring_h_charge(mol):
    """The mean file charge of the hydrogens on five-membered aromatic ring carbons."""
    patt = Chem.MolFromSmarts("[#1][c;r5]")
    q = [mol.GetAtomWithIdx(h).GetDoubleProp("_TriposPartialCharge")
         for h, _ in mol.GetSubstructMatches(patt)]
    if not q:
        raise RuntimeError("the reference compound has no hydrogens on a five-membered aromatic ring")
    return float(np.mean(q))


def annotate(mol, fid, act):
    no, scaffold, r, r1, r2, r3, r4, ref, subset = TABLE[fid]
    mm = Chem.Mol(mol)
    mm.SetProp("_Name", f"cpd_{no:02d}")
    mm.SetProp("paper_no", str(no))
    mm.SetProp("tikhonova_id", str(fid))
    mm.SetProp("pKi_AMPA", f"{act[0]:.3f}")
    mm.SetProp("pKi_glycine", f"{act[1]:.3f}")
    mm.SetProp("delta_AMPA_glycine", f"{act[2]:.3f}")
    mm.SetProp("subset", subset)
    mm.SetProp("scaffold", str(scaffold))
    for tag, v in (("R", r), ("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4)):
        mm.SetProp(tag, v)
    if scaffold == 2:
        mm.SetProp("N5", "Et")
    mm.SetProp("ref", str(ref))
    if no in DUPLICATES:
        mm.SetProp("duplicate_of", f"cpd_{DUPLICATES[no]:02d}")
    mm.SetProp("charges", CHARGES)
    mm.SetProp("source", SOURCE)
    for a in mm.GetAtoms():
        a.SetProp("sybyl_type", a.GetProp("_TriposAtomType") if a.HasProp("_TriposAtomType") else a.GetSymbol())
        q = a.GetDoubleProp("_TriposPartialCharge") if a.HasProp("_TriposPartialCharge") else 0.0
        a.SetDoubleProp("partial_charge", round(q, 4))
    Chem.CreateAtomStringPropertyList(mm, "sybyl_type")
    Chem.CreateAtomDoublePropertyList(mm, "partial_charge")
    _fix_for_sdf(mm)
    q = sum(a.GetDoubleProp("partial_charge") for a in mm.GetAtoms())
    mm.SetProp("file_charge_sum", f"{q:.3f}")
    mm.SetProp("formal_charge", str(Chem.GetFormalCharge(mm)))
    return mm


def core_atoms(mol):
    hits = mol.GetSubstructMatches(CORE, uniquify=True)
    if len(hits) != 1:
        raise RuntimeError(f"{mol.GetProp('_Name')}: {len(hits)} matches of the core, expected 1")
    return list(hits[0])


def superimpose(mols):
    """Move every molecule onto the reference by the fourteen core atoms; returns the RMSDs."""
    ref = next(m for m in mols if int(m.GetProp("paper_no")) == REFERENCE_NO)
    ref_atoms = core_atoms(ref)
    out = []
    for m in mols:
        amap = list(zip(core_atoms(m), ref_atoms))
        rmsd = rdMolAlign.AlignMol(m, ref, atomMap=amap)
        m.SetProp("align_rmsd", f"{rmsd:.3f}")
        m.SetProp("align_atoms", str(len(amap)))
        out.append(rmsd)
    return out


def write(mols, path):
    w = Chem.SDWriter(path)
    for m in sorted(mols, key=lambda m: int(m.GetProp("paper_no"))):
        w.write(m)
    w.close()


def convert(root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    act = read_activities(root)
    missing = sorted(set(TABLE) ^ set(act))
    if missing:
        raise RuntimeError(f"ids in the table and in ampa.txt differ: {missing}")
    aligned, docked = [], {"ampa": [], "glycine": []}
    ref_id = next(fid for fid, row in TABLE.items() if row[0] == REFERENCE_NO)
    h_charge = ring_h_charge(read_ligand(os.path.join(root, "ampa", "ampa.mdb", f"{ref_id}.mol2")))
    for fid in sorted(TABLE):
        for rec, folder in (("ampa", "ampa.mdb"), ("glycine", "cscore_files1.mdb")):
            m = read_ligand(os.path.join(root, rec, folder, f"{fid}.mol2"))
            fixed = complete_hydrogens(m, h_charge)
            if fixed:
                m = fixed
                print(f"  id {fid}: {m.GetProp('fixed')} ({rec})")
            m = annotate(m, fid, act[fid])
            m.SetProp("frame", f"docked in the {rec} receptor model (receptor_{rec}.pdb)")
            docked[rec].append(m)
            if rec == "ampa":
                a = Chem.Mol(m)
                a.SetProp("frame", "superimposed on the tricyclic core of compound 1 (Figure 1 of the paper)")
                aligned.append(a)
    rmsd = superimpose(aligned)
    write(aligned, os.path.join(out_dir, "ampa_glycine.sdf"))
    for rec, mols in docked.items():
        write(mols, os.path.join(out_dir, f"docked_{rec}.sdf"))
        shutil.copyfile(os.path.join(root, rec, "receptor.pdb"), os.path.join(out_dir, f"receptor_{rec}.pdb"))
    n_tr = sum(1 for m in aligned if m.GetProp("subset") == "train")
    n_te = len(aligned) - n_tr
    print(f"{len(aligned)} compounds (train {n_tr}, test {n_te}) -> {out_dir}")
    print(f"superposition on {len(core_atoms(aligned[0]))} core atoms: RMSD median {np.median(rmsd):.3f}, "
          f"max {max(rmsd):.3f} A ({aligned[int(np.argmax(rmsd))].GetProp('_Name')})")
    for m in aligned:
        print(f"  {m.GetProp('_Name')}  id {m.GetProp('tikhonova_id'):>2s}  scaffold {m.GetProp('scaffold')}  "
              f"pKi {m.GetProp('pKi_AMPA'):>7s} {m.GetProp('pKi_glycine'):>7s}  delta {m.GetProp('delta_AMPA_glycine'):>7s}  "
              f"{m.GetProp('subset'):5s} rmsd {m.GetProp('align_rmsd')}  "
              f"{Chem.MolToSmiles(Chem.RemoveHs(m))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_ROOT, help="folder with ampa/ and glycine/ (Tikhonova's files)")
    p.add_argument("--out", default=OUT)
    a = p.parse_args()
    RDLogger.DisableLog("rdApp.*")
    convert(a.data_root, a.out)


if __name__ == "__main__":
    main()
