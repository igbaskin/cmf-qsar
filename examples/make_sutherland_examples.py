"""
Convert the eight Sutherland 3D-QSAR data sets (Sutherland, O'Brien & Weaver, J. Med. Chem. 47
(2004) 5541-5554; supplementary material) from the original MOL2 + activity-table distribution
into self-contained SDF files in ``examples/sutherland/``.

Every SDF record carries
  * the activity (``pIC50`` or ``pKi``) and the published train/test split (``subset``),
  * the original Sybyl atom types and partial charges (MNDO ESP-fit, Gasteiger-Marsili for THER)
    as per-atom property lists (``atom.prop.sybyl_type``, ``atom.dprop.partial_charge``), which the
    program uses for continuous indicator fields and for the electrostatic field when
    "Partial charges: file" is selected,
  * the aligned 3D coordinates exactly as distributed (no re-alignment).

Usage:  python examples/make_sutherland_examples.py --data-root <folder with <set>_data subfolders>
"""
import argparse
import os
import sys

import numpy as np
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cmfqsar.dataio import apply_activity_table, read_activity_table, read_molecules  # noqa: E402

SETS = {  # name: (folder, mol2 prefix, activity label, charge model)
    "ACE": ("ace_data", "ace-3dqsar-mndo", "pIC50", "MNDO ESP-fit (scaled)"),
    "AChE": ("ache_data", "ache-3dqsar-mndo", "pIC50", "MNDO ESP-fit (scaled)"),
    "BZR": ("bzr_data", "bzr-3dqsar-mndo", "pIC50", "MNDO ESP-fit (scaled)"),
    "COX2": ("cox2_data", "cox2-3dqsar-mndo", "pIC50", "MNDO ESP-fit (scaled)"),
    "DHFR": ("dhfr_data", "dhfr-3dqsar-mndo", "pIC50", "MNDO ESP-fit (scaled)"),
    "GPB": ("gpb_data", "gpb-3dqsar-mndo", "pKi", "MNDO ESP-fit (scaled)"),
    "THER": ("ther_data", "ther-3dqsar-gasteiger", "pKi", "Gasteiger-Marsili"),
    "THR": ("thr_data", "thr-3dqsar-mndo", "pKi", "MNDO ESP-fit (scaled)"),
}
TARGETS = {
    "ACE": "angiotensin-converting enzyme inhibitors",
    "AChE": "acetylcholinesterase inhibitors",
    "BZR": "benzodiazepine receptor ligands",
    "COX2": "cyclooxygenase-2 inhibitors",
    "DHFR": "dihydrofolate reductase inhibitors",
    "GPB": "glycogen phosphorylase b inhibitors",
    "THER": "thermolysin inhibitors",
    "THR": "thrombin inhibitors",
}
SOURCE = ("Sutherland J.J., O'Brien L.A., Weaver D.F. J. Med. Chem. 47 (2004) 5541 (supplementary material): "
          "aligned MMFF94s-minimised conformations; split as published")


def _fix_for_sdf(mol: Chem.Mol) -> None:
    """MOL2 'ar' bonds on non-ring groups (e.g. carboxylate C-O) cannot be kekulised: make them single."""
    try:
        Chem.FastFindRings(mol)
    except Exception:
        pass
    ri = mol.GetRingInfo()
    changed = False
    for b in mol.GetBonds():
        if b.GetIsAromatic() and ri.NumBondRings(b.GetIdx()) == 0:
            b.SetIsAromatic(False)
            b.SetBondType(Chem.BondType.SINGLE)
            changed = True
    for a in mol.GetAtoms():
        if a.GetIsAromatic() and ri.NumAtomRings(a.GetIdx()) == 0:
            a.SetIsAromatic(False)
            changed = True
    # pentavalent neutral nitrogen (nitro / N-oxide written as N(=O)=O): use the charge-separated form
    mol.UpdatePropertyCache(strict=False)
    for a in mol.GetAtoms():
        if a.GetSymbol() == "N" and a.GetFormalCharge() == 0 and a.GetExplicitValence() > 3:
            for b in a.GetBonds():
                o = b.GetOtherAtom(a)
                if b.GetBondType() == Chem.BondType.DOUBLE and o.GetSymbol() == "O" and o.GetDegree() == 1:
                    b.SetBondType(Chem.BondType.SINGLE)
                    o.SetFormalCharge(-1)
                    a.SetFormalCharge(1)
                    changed = True
                    break
            mol.UpdatePropertyCache(strict=False)
    # azide written as N=N#N with an over-charged central N: use R-N=[N+]=[N-]
    for a in mol.GetAtoms():
        if a.GetSymbol() == "N" and sum(b.GetBondTypeAsDouble() for b in a.GetBonds()) > 4:
            trip = [b for b in a.GetBonds() if b.GetBondType() == Chem.BondType.TRIPLE
                    and b.GetOtherAtom(a).GetSymbol() == "N" and b.GetOtherAtom(a).GetDegree() == 1]
            if trip:
                trip[0].SetBondType(Chem.BondType.DOUBLE)
                trip[0].GetOtherAtom(a).SetFormalCharge(-1)
                a.SetFormalCharge(1)
                changed = True
    mol.UpdatePropertyCache(strict=False)
    if changed:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            mol.UpdatePropertyCache(strict=False)


def convert(data_root: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for name, (folder, prefix, label, charges) in SETS.items():
        mol2 = os.path.join(data_root, folder, f"{prefix}-all.mol2")
        table = os.path.join(data_root, folder, f"{folder.split('_')[0]}-data-sybyl-all.txt")
        mols, names, _ = read_molecules(mol2, add_hs=False)
        from cmfqsar.dataio import Dataset
        ds = Dataset(mols=mols, names=names, activity=np.full(len(mols), np.nan),
                     subset=np.array(["train"] * len(mols), dtype=object))
        n_act = apply_activity_table(ds, read_activity_table(table), use_subset=True)
        out = os.path.join(out_dir, f"{name}.sdf")
        w = Chem.SDWriter(out)
        n_written = 0
        for m, nm, y, sub in zip(ds.mols, ds.names, ds.activity, ds.subset):
            if not np.isfinite(y):
                continue   # compounds without activity (the "inactive" ones) are not part of the sets
            mm = Chem.Mol(m)
            mm.SetProp("_Name", nm)
            mm.SetProp(label, f"{y:.2f}")
            mm.SetProp("subset", str(sub))
            mm.SetProp("target", TARGETS[name])
            mm.SetProp("charges", charges)
            mm.SetProp("source", SOURCE)
            for a in mm.GetAtoms():
                a.SetProp("sybyl_type", a.GetProp("_TriposAtomType") if a.HasProp("_TriposAtomType") else a.GetSymbol())
                q = a.GetDoubleProp("_TriposPartialCharge") if a.HasProp("_TriposPartialCharge") else 0.0
                a.SetDoubleProp("partial_charge", round(q, 4))
            Chem.CreateAtomStringPropertyList(mm, "sybyl_type")
            Chem.CreateAtomDoublePropertyList(mm, "partial_charge")
            _fix_for_sdf(mm)
            w.write(mm)
            n_written += 1
        w.close()
        n_tr = int(np.sum((ds.subset == "train") & np.isfinite(ds.activity)))
        n_te = int(np.sum((ds.subset == "test") & np.isfinite(ds.activity)))
        print(f"{name:5s} {len(mols):4d} molecules in MOL2, {n_act:4d} with activity -> {n_written:4d} written "
              f"(train {n_tr}, test {n_te}) -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sutherland"))
    a = p.parse_args()
    convert(a.data_root, a.out)
