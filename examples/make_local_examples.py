#!/usr/bin/env python
"""Prepare the two per-atom data sets of plan item LP2.

Both sets hold a target that belongs to an atom (plan item LP1), written in the convention
:mod:`cmfqsar.atomtargets` reads: the molecule property ``atom.dprop.<name>`` with one field per
atom and ``n/a`` where the atom was not measured.

**Carbon shifts** (``examples/local/shifts_cdcl3.sdf``) come from nmrshiftdb2, a free database of
measured NMR spectra with the signals assigned to atoms.  Only the 13C spectra measured in
deuterated chloroform are kept, because a shift depends on the solvent and most records do not say
which one was used.  The file is 160 MB and is not part of this repository; download it once:

    curl -L -o nmrshiftdb2withsignals.sd \\
        https://sourceforge.net/projects/nmrshiftdb2/files/data/nmrshiftdb2withsignals.sd/download

**Carboxylic acid pKa** (``examples/local/carboxylic_pka.sdf``) comes from the data sets collected
by Baltruschat and Czodrowski (F1000Research 9:113, 2020), which are free to download:

    https://github.com/czodrowskilab/Machine-learning-meets-pKa/tree/master/datasets

``AvLiLuMoVe.sdf`` holds literature values for neutral molecules and
``combined_training_datasets_unique.sdf`` holds values extracted from ChEMBL, with the molecules
stored in the ionised form.  Only molecules with exactly one carboxylic acid group and a value
between 0.5 and 6.5 are kept, so that the value can only be the acidity of that group: a
protonated amine is a base and would sit above 7.  The target is written on the carboxyl carbon.

Structures in both sources are two-dimensional, so a conformation is generated here with ETKDG and
refined with MMFF, and the atom order is left untouched so that the measured atom keeps its index.

Usage
-----
    python examples/make_local_examples.py --shifts PATH/nmrshiftdb2withsignals.sd
    python examples/make_local_examples.py --pka-dir PATH/with/the/two/sdf/files
    python examples/make_local_examples.py --shifts ... --pka-dir ... --max-shifts 400
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402

OUT = os.path.join(HERE, "local")
ELEMENTS = {"C", "H", "N", "O", "F", "S", "Cl", "Br", "I"}
ACID = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
CARBOXYLATE = Chem.MolFromSmarts("[CX3](=O)[OX1-]")
OTHER_ACIDS = [Chem.MolFromSmarts(s) for s in
               ("[SX4](=O)(=O)[OX2H1,OX1-]", "[PX4](=O)([OX2H1,OX1-])", "c1nnn[nH]1",
                "[SX3](=O)[OX2H1,OX1-]")]
SHIFT_RANGE = (-20.0, 240.0)
PKA_RANGE = (0.5, 6.5)


# --------------------------------------------------------------------------- common


def ordinary(mol, lo_heavy=5, hi_heavy=24):
    """Is this a small, neutral, single-piece organic molecule of the usual elements?"""
    if mol is None:
        return False
    if any(a.GetSymbol() not in ELEMENTS for a in mol.GetAtoms()):
        return False
    if Chem.GetFormalCharge(mol) != 0:
        return False
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    return lo_heavy <= mol.GetNumHeavyAtoms() <= hi_heavy


def embed(mol, seed=1):
    """A three-dimensional conformation, with the atom order left as it is."""
    m = Chem.AddHs(mol, addCoords=True)
    m.RemoveAllConformers()
    p = AllChem.ETKDGv3()
    p.randomSeed = int(seed)
    if AllChem.EmbedMolecule(m, p) != 0:
        p.useRandomCoords = True
        if AllChem.EmbedMolecule(m, p) != 0:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=500)
    except Exception:
        pass
    return m


def as_fields(values):
    """One field per atom, ``n/a`` where there is no value."""
    return " ".join("n/a" if not np.isfinite(v) else f"{float(v):.6g}" for v in values)


def write(path, records, name):
    """Write the molecules with their per-atom target."""
    folder = os.path.dirname(path) or "."
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with Chem.SDWriter(path) as w:
        for mol, values, title, extra in records:
            out = Chem.Mol(mol)
            out.SetProp("_Name", str(title))
            out.SetProp("atom.dprop." + name, as_fields(values))
            for k, v in extra.items():
                out.SetProp(str(k), str(v))
            w.write(out)
    return len(records)


# --------------------------------------------------------------------------- carbon shifts


def parse_spectrum(text):
    """``shift;intensity;atom|...`` as a dict from atom index to the mean shift reported for it."""
    got = {}
    for part in str(text).split("|"):
        f = part.strip().split(";")
        if len(f) < 3:
            continue
        try:
            shift = float(f[0])
            atom = int(f[2])
        except ValueError:
            continue
        got.setdefault(atom, []).append(shift)
    return {a: float(np.mean(v)) for a, v in got.items()}


def in_chloroform(mol):
    """Was the first spectrum of this record measured in deuterated chloroform?"""
    if not mol.HasProp("Solvent"):
        return False
    first = str(mol.GetProp("Solvent")).strip().split(" 1:")[0]
    return "Chloroform-D1" in first


def shift_records(path, limit, seed=1, verbose=True, any_solvent=False):
    """Molecules of nmrshiftdb2 with their assigned 13C shifts, prepared in three dimensions.

    `any_solvent` keeps the records whose solvent is unreported as well, which is most of the
    database.  A shift depends on the solvent, so that set is noisier by a few ppm; it is what
    plan item LP5 uses to show that the method runs at the size the database really has.
    """
    out, keys, read, embedded = [], set(), 0, 0
    for mol in Chem.ForwardSDMolSupplier(path, removeHs=False):
        read += 1
        if mol is None or not mol.HasProp("Spectrum 13C 0"):
            continue
        if not (any_solvent or in_chloroform(mol)) or not ordinary(mol):
            continue
        assigned = parse_spectrum(mol.GetProp("Spectrum 13C 0"))
        n = mol.GetNumAtoms()
        if not assigned or any(not 0 <= a < n for a in assigned):
            continue
        if any(mol.GetAtomWithIdx(a).GetSymbol() != "C" for a in assigned):
            continue
        if len(assigned) < 3:
            continue
        if any(not SHIFT_RANGE[0] <= s <= SHIFT_RANGE[1] for s in assigned.values()):
            continue
        key = Chem.MolToInchiKey(mol)
        if not key or key in keys:
            continue
        keys.add(key)
        m = embed(mol, seed)
        if m is None:
            continue
        embedded += 1
        values = np.full(m.GetNumAtoms(), np.nan)
        for a, s in assigned.items():
            values[a] = s
        title = (mol.GetProp("nmrshiftdb2 ID") if mol.HasProp("nmrshiftdb2 ID")
                 else f"shift{embedded}")
        out.append((m, values, title, {"activity": f"{np.nanmean(values):.3f}",
                                       "carbons_measured": str(len(assigned))}))
        if verbose and embedded % 50 == 0:
            print(f"  {embedded} molecules kept out of {read} read", flush=True)
        if len(out) >= int(limit):
            break
    return out, read


# --------------------------------------------------------------------------- carboxylic acid pKa


def neutralise(mol):
    """Put the proton back on a carboxylate, so that the molecule is the acid."""
    rw = Chem.RWMol(mol)
    for atom in rw.GetAtoms():
        if (atom.GetSymbol() == "O" and atom.GetFormalCharge() == -1 and atom.GetDegree() == 1
                and atom.GetNeighbors()[0].GetSymbol() == "C"):
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(1)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def acid_value(mol):
    """The reported pKa of this record, or None when there is none in the acid range."""
    if not mol.HasProp("pKa"):
        return None
    try:
        p = float(mol.GetProp("pKa"))
    except ValueError:
        return None
    return p if PKA_RANGE[0] <= p <= PKA_RANGE[1] else None


def one_acid(mol):
    """The molecule as the neutral acid with exactly one carboxylic group, or None."""
    if mol is None:
        return None
    if any(mol.HasSubstructMatch(o) for o in OTHER_ACIDS):
        return None
    charge = Chem.GetFormalCharge(mol)
    if charge == 0:
        acid = mol
    elif charge == -1 and len(mol.GetSubstructMatches(CARBOXYLATE)) == 1:
        acid = neutralise(mol)
    else:
        return None
    if acid is None or not ordinary(acid, hi_heavy=30):
        return None
    return acid if len(acid.GetSubstructMatches(ACID)) == 1 else None


def pka_records(paths, seed=1, verbose=True):
    """Carboxylic acids with their pKa on the carboxyl carbon, prepared in three dimensions."""
    found = {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for mol in Chem.ForwardSDMolSupplier(path, removeHs=False):
            if mol is None:
                continue
            value = acid_value(mol)
            if value is None:
                continue
            acid = one_acid(mol)
            if acid is None:
                continue
            key = Chem.MolToInchiKey(acid)
            if not key:
                continue
            found.setdefault(key, [acid, []])[1].append(value)
    out = []
    for key in sorted(found):
        acid, values = found[key]
        m = embed(acid, seed)
        if m is None:
            continue
        match = m.GetSubstructMatches(ACID)
        if len(match) != 1:
            continue
        carbon = match[0][0]
        values_a = np.full(m.GetNumAtoms(), np.nan)
        values_a[carbon] = float(np.mean(values))
        out.append((m, values_a, key, {"activity": f"{np.mean(values):.3f}",
                                       "measurements": str(len(values))}))
        if verbose and len(out) % 100 == 0:
            print(f"  {len(out)} acids prepared out of {len(found)}", flush=True)
    return out, len(found)


# --------------------------------------------------------------------------- main


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shifts", default="", help="path to nmrshiftdb2withsignals.sd")
    ap.add_argument("--pka-dir", default="", help="directory with the two pKa SDF files")
    ap.add_argument("--max-shifts", type=int, default=400,
                    help="molecules to keep for the shift set (default 400)")
    ap.add_argument("--seed", type=int, default=1, help="seed of the conformation generator")
    ap.add_argument("--any-solvent", action="store_true",
                    help="keep the 13C spectra whose solvent is unreported as well, which is most "
                         "of the database and is noisier by a few ppm (plan item LP5)")
    ap.add_argument("--out", default=OUT, metavar="DIR",
                    help="where the prepared files go (default examples/local). A set of "
                         "thousands of molecules belongs outside the repository; plan item LP5 "
                         "builds one that way and the benchmark reads it with --data-root")
    a = ap.parse_args(argv)
    RDLogger.DisableLog("rdApp.*")
    if not a.shifts and not a.pka_dir:
        ap.error("give --shifts, --pka-dir, or both")

    if a.shifts:
        print(f"Carbon shifts from {a.shifts}")
        records, read = shift_records(a.shifts, a.max_shifts, a.seed,
                                      any_solvent=a.any_solvent)
        path = os.path.join(a.out, "shifts_cdcl3.sdf")
        n = write(path, records, "shift")
        atoms = sum(int(np.isfinite(v).sum()) for _m, v, _t, _e in records)
        print(f"  {n} molecules, {atoms} measured carbons, from {read} records -> {path}")

    if a.pka_dir:
        print(f"Carboxylic acid pKa from {a.pka_dir}")
        paths = [os.path.join(a.pka_dir, f) for f in
                 ("AvLiLuMoVe.sdf", "combined_training_datasets_unique.sdf")]
        records, unique = pka_records(paths, a.seed)
        path = os.path.join(a.out, "carboxylic_pka.sdf")
        n = write(path, records, "pka")
        print(f"  {n} acids of {unique} unique ones -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
