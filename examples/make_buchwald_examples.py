"""Build the Buchwald-Hartwig yield set: 3,955 couplings with their components in 3D frames.

The data are the high-throughput screen of Ahneman, Estrada, Lin, Dreher and Doyle (Science
360 (2018) 186): the palladium-catalysed C-N coupling of 15 aryl halides with 4-methylaniline,
with 4 biaryl phosphine ligands, 3 bases and 22 isoxazole additives, and the yield of each
combination measured. The copy read here is the workbook of the Yield-BERT repository
(github.com/rxn4chemistry/rxn_yields, MIT licence), which also holds the published splits:
ten random 70/30 splits (sheets FullCV_01 to FullCV_10) and four splits that hold out five
additives each (sheets Test1 to Test4).

What this script writes to examples/buchwald/:

* reactions.csv: one row per reaction, the four components by label and the yield in percent.
* components.csv: every component with its role, label and SMILES.
* splits.csv: for every published split, whether each reaction is in the training or the
  test part. The training part of a sheet is its first rows, as the Yield-BERT code reads it.
* aryl_halides.sdf: the 15 aryl halides in the frame of the bond that breaks, the ipso carbon
  at the origin, the halogen on +x and the ring in the xy plane. A planar aryl halide can be
  turned half a turn about that bond and reflected through its ring plane without changing
  what it is, so every one is written as the four images that these two operations produce,
  under one name; the field model averages over them.
* additives.sdf: the 22 additives in the frame of their N-O bond, the midpoint of the bond at
  the origin, the nitrogen on +x and the centre of the isoxazole ring on +y, so the ring lies
  in the xy plane. The two faces of the ring are equivalent, so every additive is written as
  itself and its reflection through the ring plane.

The ligands and the bases are not built: with four ligands and three bases a model can only
tell them apart, which their labels already do.

Usage
-----
    python examples/make_buchwald_examples.py --fetch   # download the workbook, then build
    python examples/make_buchwald_examples.py           # build from the downloaded copy
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from cmfqsar.orca import geometry_faults  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT_DIR = os.path.join(HERE, "buchwald")
UPSTREAM = os.path.join(OUT_DIR, "upstream")
SOURCE_URL = ("https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/"
              "Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx")
LICENSE_URL = "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/LICENSE"
WORKBOOK = "Dreher_and_Doyle_input_data.xlsx"
#: SHA-256 of the workbook as downloaded on 14 November 2026
SOURCE_DIGEST = "d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e"

COLUMNS = ("Ligand", "Base", "Aryl halide", "Additive")
ROLES = {"Ligand": "ligand", "Base": "base", "Aryl halide": "aryl_halide", "Additive": "additive"}
#: the published splits and the first row of the test part of each sheet, counted from 1, as
#: the Yield-BERT training script reads them (train = rows before it)
SPLITS: Dict[str, int] = {**{f"FullCV_{i:02d}": 2768 for i in range(1, 11)},
                          "Test1": 3058, "Test2": 3056, "Test3": 3059, "Test4": 3056}
SPLIT_NAMES = {**{f"FullCV_{i:02d}": f"random_{i:02d}" for i in range(1, 11)},
               **{f"Test{i}": f"additive_test{i}" for i in range(1, 5)}}


# ----------------------------------------------------------------------------- the source


def fetch(force: bool = False) -> str:
    os.makedirs(UPSTREAM, exist_ok=True)
    path = os.path.join(UPSTREAM, WORKBOOK)
    for url, dest in ((SOURCE_URL, path), (LICENSE_URL, os.path.join(UPSTREAM, "LICENSE"))):
        if force or not os.path.exists(dest):
            print("downloading", url)
            urllib.request.urlretrieve(url, dest)
    return path


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_sheets(path: str) -> Dict[str, List[Tuple[str, str, str, str, float]]]:
    """Every split sheet as a list of (ligand, base, aryl halide, additive, yield) rows."""
    import pandas as pd
    out = {}
    for sheet in SPLITS:
        df = pd.read_excel(path, sheet_name=sheet)
        missing = [c for c in COLUMNS + ("Output",) if c not in df.columns]
        if missing:
            raise ValueError(f"sheet {sheet} of {path} has no column {missing}")
        out[sheet] = [(str(r["Ligand"]), str(r["Base"]), str(r["Aryl halide"]),
                       str(r["Additive"]), float(r["Output"])) for _, r in df.iterrows()]
    return out


# ----------------------------------------------------------------------------- labels


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"cannot read SMILES {smiles}")
    return Chem.MolToSmiles(mol)


def ligand_label(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    methoxy = mol.HasSubstructMatch(Chem.MolFromSmarts("c[OX2][CH3]"))
    # adamantyl first: its rings are cyclohexane rings too, so it also matches the XPhos pattern
    if mol.HasSubstructMatch(Chem.MolFromSmarts("P(C12CC3CC(CC(C3)C1)C2)")):
        return "AdBrettPhos"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("P([CH1]1CCCCC1)[CH1]1CCCCC1")):
        return "XPhos"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("P(C(C)(C)C)C(C)(C)C")):
        return "t-BuBrettPhos" if methoxy else "t-BuXPhos"
    raise ValueError(f"unknown ligand {smiles}")


def base_label(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol.HasSubstructMatch(Chem.MolFromSmarts("P=N")):
        return "P2Et"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("C1CCN2CCCN=C2N1")) or \
            mol.HasSubstructMatch(Chem.MolFromSmarts("CN1CCCN2CCCN=C12")):
        return "MTBD"
    if mol.HasSubstructMatch(Chem.MolFromSmarts("CC(C)(C)N=C(N(C)C)N(C)C")):
        return "BTMG"
    raise ValueError(f"unknown base {smiles}")


def reacting_bond(mol: Chem.Mol) -> Tuple[int, int]:
    """(ipso carbon, halogen) of an aryl halide."""
    hits = mol.GetSubstructMatches(Chem.MolFromSmarts("[c;r6]-[Cl,Br,I]"))
    if len(hits) != 1:
        raise ValueError(f"{Chem.MolToSmiles(mol)}: expected one aryl-halogen bond, found {len(hits)}")
    return hits[0]


def aryl_label(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    ipso, x = reacting_bond(mol)
    halide = mol.GetAtomWithIdx(x).GetSymbol()
    ring = Chem.MolFromSmarts("n1ccccc1")
    if mol.HasSubstructMatch(ring):
        n = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "N"][0]
        path = len(Chem.GetShortestPath(mol, ipso, n)) - 1
        arene = {1: "2-pyridyl", 2: "3-pyridyl"}[path]
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("cC(F)(F)F")):
        arene = "4-CF3-phenyl"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("c[OX2][CH3]")):
        arene = "4-MeO-phenyl"
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("c[CH2][CH3]")):
        arene = "4-Et-phenyl"
    else:
        raise ValueError(f"unknown aryl halide {smiles}")
    return f"{arene}-{halide}"


def isoxazole_bond(mol: Chem.Mol) -> Tuple[int, int, List[int]]:
    """(oxygen, nitrogen, atoms of the five-membered ring) of the isoxazole N-O bond."""
    hits = mol.GetSubstructMatches(Chem.MolFromSmarts("[o,O;r5]:,-[n,N;r5]"))
    if len(hits) != 1:
        raise ValueError(f"{Chem.MolToSmiles(mol)}: expected one N-O bond in a five-membered "
                         f"ring, found {len(hits)}")
    o, n = hits[0]
    for ring in mol.GetRingInfo().AtomRings():
        if len(ring) == 5 and o in ring and n in ring:
            return o, n, list(ring)
    raise ValueError(f"{Chem.MolToSmiles(mol)}: the N-O bond is in no five-membered ring")


# ----------------------------------------------------------------------------- 3D structures


def embed_lowest(smiles: str, n: int = 20, seed: int = 2026) -> Chem.Mol:
    """Hydrogens added, `n` ETKDG conformers minimised with MMFF94, the lowest kept."""
    mh = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    params.pruneRmsThresh = -1.0
    ids = list(AllChem.EmbedMultipleConfs(mh, numConfs=int(n), params=params))
    if not ids:
        raise RuntimeError(f"no conformer could be embedded for {smiles}")
    props = AllChem.MMFFGetMoleculeProperties(mh)
    if props is None:
        raise RuntimeError(f"MMFF94 has no parameters for {smiles}")
    energies = []
    for cid in ids:
        ff = AllChem.MMFFGetMoleculeForceField(mh, props, confId=cid)
        ff.Minimize(maxIts=2000)
        energies.append(float(ff.CalcEnergy()))
    Chem.SanitizeMol(mh)                  # MMFF set-up rewrites the aromaticity flags in place
    best = ids[int(np.argmin(energies))]
    out = Chem.Mol(mh)
    out.RemoveAllConformers()
    out.AddConformer(Chem.Conformer(mh.GetConformer(best)), assignId=True)
    out.SetProp("force_field", "MMFF94")
    out.SetProp("mmff_energy", f"{min(energies):.4f}")
    faults = geometry_faults(out)
    if faults:
        raise RuntimeError(f"{smiles}: the built structure is wrong: {'; '.join(faults)}")
    return out


def frame(origin: np.ndarray, x_towards: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Rows x, y, z of a right-handed frame: x along `x_towards`, y the part of `y_hint`
    perpendicular to x."""
    x = x_towards / np.linalg.norm(x_towards)
    y = y_hint - np.dot(y_hint, x) * x
    y /= np.linalg.norm(y)
    return np.vstack([x, y, np.cross(x, y)])


def with_coordinates(mol: Chem.Mol, pos: np.ndarray, name: str, image: str) -> Chem.Mol:
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    for atom, p in enumerate(pos):
        conf.SetAtomPosition(atom, p.tolist())
    out.SetProp("_Name", name)
    out.SetProp("image", image)
    return out


def place_aryl_halide(mol: Chem.Mol, name: str) -> List[Chem.Mol]:
    """The four images of an aryl halide in the frame of its C-X bond."""
    ipso, x = reacting_bond(mol)
    pos = np.array(mol.GetConformer().GetPositions(), float)
    ring = [r for r in mol.GetRingInfo().AtomRings() if ipso in r][0]
    centre = pos[list(ring)].mean(axis=0)
    # y in the ring plane: from the ring centre to the ortho carbon of lower index projects
    # to a direction perpendicular to C-X; the images below make its sign irrelevant
    ortho = sorted(a.GetIdx() for a in mol.GetAtomWithIdx(ipso).GetNeighbors() if a.GetIdx() in ring)[0]
    R = frame(pos[ipso], pos[x] - pos[ipso], pos[ortho] - centre)
    local = (pos - pos[ipso]) @ R.T
    images = {"identity": np.array([1, 1, 1.0]), "half turn": np.array([1, -1, -1.0]),
              "mirror z": np.array([1, 1, -1.0]), "mirror y": np.array([1, -1, 1.0])}
    return [with_coordinates(mol, local * s, name, k) for k, s in images.items()]


def place_additive(mol: Chem.Mol, name: str) -> List[Chem.Mol]:
    """The two images of an additive in the frame of its N-O bond."""
    o, n, ring = isoxazole_bond(mol)
    pos = np.array(mol.GetConformer().GetPositions(), float)
    mid = 0.5 * (pos[o] + pos[n])
    R = frame(mid, pos[n] - pos[o], pos[ring].mean(axis=0) - mid)
    local = (pos - mid) @ R.T
    return [with_coordinates(mol, local, name, "identity"),
            with_coordinates(mol, local * np.array([1, 1, -1.0]), name, "mirror z")]


# ----------------------------------------------------------------------------- the build


def build(path: str, out_dir: str = OUT_DIR, conformers: int = 20, seed: int = 2026) -> dict:
    sheets = read_sheets(path)
    first = sheets["FullCV_01"]
    # labels: ligands and bases by name, aryl halides by arene and halogen, additives A01..A22
    # in the order of their canonical SMILES
    smiles = {c: sorted({canonical(r[i]) for r in first}) for i, c in enumerate(COLUMNS)}
    label: Dict[Tuple[str, str], str] = {}
    for s in smiles["Ligand"]:
        label[("Ligand", s)] = ligand_label(s)
    for s in smiles["Base"]:
        label[("Base", s)] = base_label(s)
    for s in smiles["Aryl halide"]:
        label[("Aryl halide", s)] = aryl_label(s)
    for k, s in enumerate(smiles["Additive"], start=1):
        label[("Additive", s)] = f"A{k:02d}"
    for c in COLUMNS:
        names = [label[(c, s)] for s in smiles[c]]
        if len(set(names)) != len(names):
            raise ValueError(f"two {c} components got the same label: {names}")

    def key(row) -> Tuple[str, str, str, str]:
        return tuple(label[(c, canonical(row[i]))] for i, c in enumerate(COLUMNS))

    # the reactions, numbered in the order of the first sheet
    reactions = []
    index: Dict[Tuple[str, str, str, str], int] = {}
    for i, row in enumerate(first):
        k = key(row)
        if k in index:
            raise ValueError(f"reaction {k} appears twice in FullCV_01")
        index[k] = i
        reactions.append({"reaction": f"R{i + 1:04d}", "ligand": k[0], "base": k[1],
                          "aryl_halide": k[2], "additive": k[3], "yield": row[4]})
    # the splits: every sheet is the same reactions in another order
    split = {}
    for sheet, start in SPLITS.items():
        rows = sheets[sheet]
        if len(rows) != len(first):
            raise ValueError(f"sheet {sheet} has {len(rows)} rows, FullCV_01 {len(first)}")
        part = np.empty(len(first), dtype=object)
        for pos, row in enumerate(rows):
            i = index[key(row)]
            if abs(reactions[i]["yield"] - row[4]) > 1e-6:
                raise ValueError(f"{sheet}: reaction {reactions[i]['reaction']} has yield "
                                 f"{row[4]} here and {reactions[i]['yield']} in FullCV_01")
            part[i] = "train" if pos < start - 1 else "test"
        split[SPLIT_NAMES[sheet]] = part
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "reactions.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reaction", "ligand", "base", "aryl_halide", "additive", "yield_percent"])
        for r in reactions:
            w.writerow([r["reaction"], r["ligand"], r["base"], r["aryl_halide"], r["additive"],
                        f"{r['yield']:.6f}"])
    with open(os.path.join(out_dir, "splits.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["reaction"] + list(split))
        for i, r in enumerate(reactions):
            w.writerow([r["reaction"]] + [split[s][i] for s in split])
    with open(os.path.join(out_dir, "components.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["role", "label", "smiles"])
        for c in COLUMNS:
            for s in smiles[c]:
                w.writerow([ROLES[c], label[(c, s)], s])
    # the 3D structures of the aryl halides and the additives
    for c, fname, place in (("Aryl halide", "aryl_halides.sdf", place_aryl_halide),
                            ("Additive", "additives.sdf", place_additive)):
        w = Chem.SDWriter(os.path.join(out_dir, fname))
        for s in smiles[c]:
            mol = embed_lowest(s, n=conformers, seed=seed)
            for image in place(mol, label[(c, s)]):
                image.SetProp("smiles", s)
                w.write(image)
        w.close()
        print(f"wrote {len(smiles[c])} {ROLES[c].replace('_', ' ')}s to {fname}")
    counts = {ROLES[c]: len(smiles[c]) for c in COLUMNS}
    print(f"{len(reactions)} reactions; " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return {"reactions": len(reactions), **counts}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fetch", action="store_true", help="download the workbook first")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--conformers", type=int, default=20, help="ETKDG conformers tried per molecule")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)
    path = fetch(args.force) if args.fetch else os.path.join(UPSTREAM, WORKBOOK)
    if not os.path.exists(path):
        print(f"{path} is missing: run with --fetch", file=sys.stderr)
        return 1
    found = digest(path)
    if found != SOURCE_DIGEST:
        print(f"warning: the workbook has SHA-256 {found}, not the {SOURCE_DIGEST} this script "
              f"was written for", file=sys.stderr)
    build(path, args.out_dir, args.conformers, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
