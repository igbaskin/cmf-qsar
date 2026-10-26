#!/usr/bin/env python
"""Build 3D structures of the cyclodextrin hosts, for use as pockets (plan item SC3).

``examples/cyclodextrin/hosts.csv`` carries each host as SMILES, which has no geometry, and a
pocket is a geometry. This embeds them and writes one SDF per host.

A cyclodextrin is a ring of identical sugars and is close to C(n) symmetric, so the conformer to
keep is the one whose units sit at the same distance from the centre of the ring: the selection
rule is the standard deviation of that distance, and it uses the symmetry of the molecule rather
than anything about the guests or their constants. The chosen conformer is reported with the
radius of its unit ring and the narrowest radius of its cavity, so that a structure that came out
collapsed can be seen to have done so.

    python examples/make_host_structures.py
    python examples/make_host_structures.py --hosts beta_cyclodextrin.csv --conformers 60
"""
from __future__ import annotations

import argparse
import csv
import io
import os

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "cyclodextrin")
OUT = os.path.join(DATA, "hosts")
#: how many conformations to try before choosing one
CONFORMERS = 40


def cavity(host, centre, axis, zs=np.linspace(-2.5, 2.5, 11), window=1.5):
    """The narrowest and the mean radius of the free channel along the axis, over heavy atoms."""
    heavy = np.array([e != "H" for e in host.elements])
    rel = host.coords[heavy] - centre
    along = rel @ axis
    radial = np.linalg.norm(rel - np.outer(along, axis), axis=1)
    out = []
    for z in zs:
        near = np.abs(along - z) < window
        if near.any():
            out.append(float(radial[near].min()))
    return (min(out), float(np.mean(out))) if out else (float("nan"), float("nan"))


def build(smiles, conformers=CONFORMERS, seed0=1):
    """The most symmetric of several embedded conformations, with its numbers."""
    from cmfqsar.protein import cavity_frame, host_from_molecule
    best = None
    for k in range(int(conformers)):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        p = AllChem.ETKDGv3()
        p.useMacrocycleTorsions = True
        p.randomSeed = seed0 + k
        if AllChem.EmbedMolecule(m, p) != 0:
            continue
        try:
            AllChem.MMFFOptimizeMolecule(m, maxIters=4000)
        except Exception:                        # noqa: BLE001 - keep the embedded geometry
            pass
        host = host_from_molecule(m, name="host")
        centre, axis = cavity_frame(host)
        keys = sorted(set(int(r) for r in host.resids))
        mids = np.array([host.coords[host.resids == k2].mean(axis=0) for k2 in keys])
        radius = np.linalg.norm(mids - centre, axis=1)
        narrow, mean_r = cavity(host, centre, axis)
        got = {"seed": seed0 + k, "units": len(keys), "unit_radius": float(radius.mean()),
               "unit_spread": float(radius.std()), "cavity_min": narrow, "cavity_mean": mean_r}
        if best is None or got["unit_spread"] < best[0]["unit_spread"]:
            best = (got, m)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hosts", nargs="*", default=None,
                    help="which rows of hosts.csv (by their 'file' column)")
    ap.add_argument("--conformers", type=int, default=CONFORMERS)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    RDLogger.DisableLog("rdApp.*")
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    os.makedirs(a.out, exist_ok=True)
    rows = list(csv.DictReader(io.open(os.path.join(DATA, "hosts.csv"), encoding="utf-8")))
    want = set(a.hosts) if a.hosts else None
    made = 0
    for r in rows:
        key = r.get("file") or ""
        if not key or (want and key not in want):
            continue
        smiles = r.get("host_smiles") or ""
        if not smiles:
            continue
        got = build(smiles, conformers=a.conformers)
        if got is None:
            print(f"{key}: no conformation could be embedded")
            continue
        info, mol = got
        name = os.path.splitext(key)[0]
        mol.SetProp("_Name", name)
        for k, v in info.items():
            mol.SetProp(k, f"{v:g}" if isinstance(v, float) else str(v))
        path = os.path.join(a.out, name + ".sdf")
        with Chem.SDWriter(path) as w:
            w.write(mol)
        made += 1
        print(f"{path}: {info['units']} units, ring radius {info['unit_radius']:.2f} A "
              f"(spread {info['unit_spread']:.2f}), cavity {info['cavity_min']:.2f} to "
              f"{info['cavity_mean']:.2f} A, from conformation {info['seed']}")
    print(f"{made} host structures written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
