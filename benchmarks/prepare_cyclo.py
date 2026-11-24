"""Prepare the second reaction set of plan item PO3: the [3+2] cycloadditions of Cyclo-23-TS.

The set is that of Stuyver, Jorner and Coley (5,269 reactions, activation free energies at
B3LYP-D3(BJ)/def2-TZVP//B3LYP-D3(BJ)/def2-SVP), in the copy that the benchmark of van Gerwen,
Briling, Calvino Alonso, Franke and Corminboeuf uses (Digital Discovery 3 (2024) 932;
github.com/lcmd-epfl/benchmark-barrier-learning, folder ``data/cyclo``; the data of that paper
are CC BY 4.0 on Zenodo, 10.5281/zenodo.8309465).

``--download`` fetches the folder with a partial clone (about 660 MB) into
``benchmarks/cache/cyclo/repo``.  Nothing of it is run; the table and the xyz files are read as
data.

What the set gives, and what this script has to add:

* ``full_dataset.csv``: one row per reaction with the mapped reaction SMILES, the activation
  free energy ``G_act`` and the reaction free energy ``G_r`` in kcal/mol.  The map numbers are
  on the heavy atoms only.
* ``xyz/<id>/``: one geometry file per reactant molecule (``r0``, ``r1``) and per product
  (``p0``), each optimised by itself.  The atoms are not in the order of the SMILES.

So every molecule is matched to its geometry as a graph (:func:`cmfqsar.reaction.mol_from_geometry`),
the molecules of a side are joined, and the hydrogens of the two sides are paired
(:func:`cmfqsar.reaction.reaction_from_mols`).  The result is written in the form of
``prepare_reactions.py``: SMILES with every atom mapped, and coordinates in the order of the map
numbers, so that the same reader takes both sets.  A reaction that cannot be read is refused and
counted, with the reason.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.reaction import (STRETCH_LIMIT, ReactionMismatch, bond_changes,  # noqa: E402
                              bond_stretch, join_molecules, mapped_smiles, mol_from_geometry,
                              reaction_from_mols, read_reaction, superpose_fragments)

CACHE = os.path.join(HERE, "cache", "cyclo")
REPO = os.path.join(CACHE, "repo")
DATA = os.path.join(REPO, "data", "cyclo")
URL = "https://github.com/lcmd-epfl/benchmark-barrier-learning.git"
WANTED = ["data/cyclo/full_dataset.csv", "data/cyclo/xyz/", "data/cyclo/*_10_fold.npy",
          "results/", "src/hypers.py", "README.md"]
TABLE = os.path.join(CACHE, "cyclo_geometries.npz")
REPORT = os.path.join(CACHE, "cyclo_prepared.json")


def _say(text):
    print(text, flush=True)


def download() -> None:
    """A partial clone that holds only the cycloaddition data and the result files."""
    if os.path.exists(os.path.join(DATA, "full_dataset.csv")):
        _say("the data are present")
        return
    os.makedirs(CACHE, exist_ok=True)
    env = dict(os.environ, MSYS_NO_PATHCONV="1")
    run = lambda *a, cwd=None: subprocess.run(a, cwd=cwd, check=True, env=env)   # noqa: E731
    run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--depth", "1", URL, REPO)
    run("git", "sparse-checkout", "set", "--no-cone", *WANTED, cwd=REPO)
    run("git", "checkout", "-q", "master", cwd=REPO)


def read_xyz(path: str):
    with open(path) as f:
        lines = f.read().split("\n")
    n = int(lines[0])
    symbols, xyz = [], []
    for line in lines[2:2 + n]:
        q = line.split()
        symbols.append(q[0])
        xyz.append([float(v) for v in q[1:4]])
    return symbols, np.array(xyz, float)


def geometry_files(folder: str, letter: str):
    """The geometry files of one side.  3,038 folders also hold a second geometry of one
    reactant, named ``..._alt.xyz``; the first one is taken and the second left out."""
    return [f for f in sorted(glob.glob(os.path.join(folder, letter + "*.xyz")))
            if not f.endswith("_alt.xyz")]


def side(smiles: str, files) -> Chem.Mol:
    """One side of a reaction: every molecule of the SMILES matched to one of the geometries."""
    geoms = [read_xyz(f) for f in files]
    parts = smiles.split(".")
    if len(parts) != len(geoms):
        raise ReactionMismatch(f"{len(parts)} molecules in the SMILES against {len(geoms)} "
                               f"geometry files")
    used, mols = set(), []
    for part in parts:
        found, last = None, None
        for k, (symbols, xyz) in enumerate(geoms):
            if k in used:
                continue
            try:
                found = mol_from_geometry(part, symbols, xyz)
                used.add(k)
                break
            except ReactionMismatch as err:
                last = err
        if found is None:
            raise last
        mols.append(found)
    return join_molecules(mols)


def prepare(limit: int = 0) -> dict:
    RDLogger.DisableLog("rdApp.*")
    with open(os.path.join(DATA, "full_dataset.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    if limit:
        rows = rows[:limit]
    refused = {}
    kept, sizes, numbers, rsmi, psmi = [], [], [], [], []
    coords = {"r": [], "p": []}
    energies, stretch, rmsd, changed = [], [], [], []
    reactant_molecules = {}
    for row in rows:
        folder = os.path.join(DATA, "xyz", row["rxn_id"])
        left, right = row["rxn_smiles"].split(">>")
        try:
            rx = reaction_from_mols(side(left, geometry_files(folder, "r")),
                                    side(right, geometry_files(folder, "p")),
                                    f"cyclo{int(row['rxn_id']):05d}")
            s_r, s_p = mapped_smiles(rx.reactant), mapped_smiles(rx.product)
            x_r = np.array(rx.reactant.GetConformer().GetPositions(), float)
            x_p = np.array(rx.product.GetConformer().GetPositions(), float)
            back = read_reaction(s_r, s_p, x_r, x_p, rx.name)       # the form the models read
            if back.symbols != rx.symbols:
                raise ReactionMismatch("the written SMILES do not give the atoms back in order")
            worst = max(bond_stretch(back.reactant), bond_stretch(back.product))
            if worst > STRETCH_LIMIT:
                raise ReactionMismatch("a bond of the graph is too long in its geometry")
        except ReactionMismatch as err:
            why = str(err).split(": ")[-1][:60]
            why = "".join(c for c in why if not c.isdigit())
            refused[why] = refused.get(why, 0) + 1
            continue
        except Exception as err:                                   # RDKit sanitisation and the like
            why = type(err).__name__
            refused[why] = refused.get(why, 0) + 1
            continue
        n_r = len(Chem.GetMolFrags(back.reactant))
        reactant_molecules[n_r] = reactant_molecules.get(n_r, 0) + 1
        kept.append(int(row["rxn_id"]))
        sizes.append(back.n_atoms)
        numbers.append([a.GetAtomicNum() for a in back.reactant.GetAtoms()])
        rsmi.append(s_r)
        psmi.append(s_p)
        coords["r"].append(x_r)
        coords["p"].append(x_p)
        # columns 3 and 4 are the two targets, as in the table of prepare_reactions.py
        energies.append([np.nan, np.nan, np.nan, float(row["G_act"]), float(row["G_r"])])
        stretch.append(worst)
        rmsd.append(superpose_fragments(back)[2])
        changed.append(len(bond_changes(back)))
        if len(kept) % 500 == 0:
            _say(f"  {len(kept)} reactions prepared")
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    np.savez_compressed(
        TABLE, idx=np.array(kept, int), offsets=offsets,
        numbers=np.concatenate(numbers).astype(np.int8),
        r=np.concatenate(coords["r"]).astype(np.float32),
        p=np.concatenate(coords["p"]).astype(np.float32),
        energies=np.array(energies, float), rsmi=np.array(rsmi), psmi=np.array(psmi))
    sizes = np.array(sizes)
    heavy = np.array([sum(1 for z in zs if z > 1) for zs in numbers])
    table = Chem.GetPeriodicTable()
    report = dict(
        rows_in_table=len(rows), kept=len(kept), refused=refused,
        atoms=dict(least=int(sizes.min()), most=int(sizes.max()), mean=float(sizes.mean())),
        heavy_atoms=dict(least=int(heavy.min()), most=int(heavy.max())),
        elements=sorted({table.GetElementSymbol(int(z)) for zs in numbers for z in zs}),
        reactant_molecules={str(k): v for k, v in sorted(reactant_molecules.items())},
        changed_bonds=dict(least=int(min(changed)), most=int(max(changed)),
                           median=float(np.median(changed))),
        rmsd_after_placing_each_molecule=dict(median=float(np.median(rmsd)),
                                              q90=float(np.quantile(rmsd, 0.9)),
                                              largest=float(np.max(rmsd))),
        longest_bond_as_multiple_of_covalent_sum=float(np.max(stretch)),
        source="github.com/lcmd-epfl/benchmark-barrier-learning, data/cyclo; "
               "Zenodo 10.5281/zenodo.8309465, CC BY 4.0", limit=limit)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=1)
    return report


def load():
    """The prepared table as a list of dicts, one per reaction, in the form of the first set."""
    with np.load(TABLE) as z:
        a = {name: z[name] for name in ("idx", "offsets", "rsmi", "psmi", "r", "p", "energies")}
    out = []
    for k, idx in enumerate(a["idx"]):
        sl = slice(int(a["offsets"][k]), int(a["offsets"][k + 1]))
        out.append(dict(idx=int(idx), rsmi=str(a["rsmi"][k]), psmi=str(a["psmi"][k]),
                        r=a["r"][sl].astype(float), p=a["p"][sl].astype(float),
                        ea=float(a["energies"][k, 3]), dh=float(a["energies"][k, 4])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--download", action="store_true", help="fetch the data with a partial clone")
    ap.add_argument("--limit", type=int, default=0, help="only the first reactions (for a trial)")
    args = ap.parse_args()
    if args.download:
        download()
    _say(json.dumps(prepare(args.limit), indent=1))


if __name__ == "__main__":
    main()
