"""Prepare the reaction set of plan item PO3: geometries of reactant, product and transition state.

The data are the gas-phase reactions of Grambow, Pattanaik and Green, *Reactants, products, and
transition states of elementary chemical reactions based on quantum chemistry*, Scientific Data
7 (2020) 137; Zenodo 10.5281/zenodo.3715478, CC BY 4.0.  This script uses the
wB97X-D3/def2-TZVP part:

* ``wb97xd3.csv`` (2.7 MB): one row per reaction, with the mapped SMILES of the two sides, the
  activation energy ``ea`` and the reaction energy ``dh``, both in kcal/mol.  Both are
  differences of the electronic energy plus the zero-point energy, so ``dh`` is a reaction
  energy at 0 K although the source calls it an enthalpy;
* ``wb97xd3.tar.gz`` (1.5 GB): one folder ``rxnNNNNNN`` per reaction with three Q-Chem log
  files, ``r``, ``p`` and ``ts``.  Each log holds a geometry optimisation and a frequency job.

Nothing of the archive is written to disk.  The script reads it as a stream, takes from every
log the last geometry, the last SCF energy, the zero-point energy and the enthalpy correction,
and writes one compressed table, ``benchmarks/cache/grambow/wb97xd3_geometries.npz``.

What it checks, and reports in ``wb97xd3_prepared.json`` beside the table:

* the three geometries of a reaction have the elements of the mapped SMILES, in the order of the
  map numbers (this is what :func:`cmfqsar.reaction.read_reaction` needs);
* no bond of either graph is longer than 1.6 times the sum of the covalent radii in its
  geometry, which would say that the coordinates do not belong to the graph;
* the activation energy and the reaction energy of the table are reproduced from the logs.

Run ``python benchmarks/prepare_reactions.py --download`` once; without the flag the two files
must already be in the cache folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.reaction import (STRETCH_LIMIT, ReactionMismatch, bond_stretch,  # noqa: E402
                              read_reaction)

CACHE = os.path.join(HERE, "cache", "grambow")
RECORD = "https://zenodo.org/api/records/3715478/files/{name}/content"
FILES = {"wb97xd3.csv": "37a67af1780fc3c1d46b4d743ebbca9b",
         "wb97xd3.tar.gz": "a5fff7110cf4426c402c7a209b51a0a1"}
TABLE = os.path.join(CACHE, "wb97xd3_geometries.npz")
REPORT = os.path.join(CACHE, "wb97xd3_prepared.json")
HARTREE = 627.5094740631                                      # kcal/mol
SIDES = ("r", "p", "ts")

_GEOM = re.compile(r"Standard Nuclear Orientation \(Angstroms\)\s*\n.*?\n\s*-+\s*\n(.*?)\n\s*-+",
                   re.S)
_ENERGY = re.compile(r"Total energy in the final basis set =\s*(-?\d+\.\d+)")
_ZPE = re.compile(r"Zero point vibrational energy:\s*(-?\d+\.\d+)\s*kcal/mol")
_ENTH = re.compile(r"Total Enthalpy:\s*(-?\d+\.\d+)\s*kcal/mol")


def _say(text):
    print(text, flush=True)


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download() -> None:
    os.makedirs(CACHE, exist_ok=True)
    for name, digest in FILES.items():
        path = os.path.join(CACHE, name)
        if os.path.exists(path) and md5(path) == digest:
            _say(f"{name}: present, checksum right")
            continue
        _say(f"{name}: downloading")
        req = urllib.request.Request(RECORD.format(name=name),
                                     headers={"User-Agent": "python-urllib"})
        with urllib.request.urlopen(req) as src, open(path, "wb") as dst:
            for block in iter(lambda: src.read(1 << 20), b""):
                dst.write(block)
        if md5(path) != digest:
            raise RuntimeError(f"{name}: the checksum is not {digest}")


def parse_log(text: str):
    """(symbols, coordinates, SCF energy in hartree, zero-point energy and enthalpy in kcal/mol).

    The geometry is the last one printed, which is the one of the frequency job.  A value that
    the log does not hold comes back as nan.
    """
    blocks = _GEOM.findall(text)
    if not blocks:
        raise ValueError("no geometry block")
    symbols, xyz = [], []
    for line in blocks[-1].splitlines():
        parts = line.split()
        if len(parts) == 5:
            symbols.append(parts[1])
            xyz.append([float(v) for v in parts[2:]])

    def last(rx):
        found = rx.findall(text)
        return float(found[-1]) if found else float("nan")

    return symbols, np.array(xyz, float), last(_ENERGY), last(_ZPE), last(_ENTH)


def read_table():
    rows = {}
    with open(os.path.join(CACHE, "wb97xd3.csv"), newline="") as f:
        for row in csv.DictReader(f):
            rows[int(row["idx"])] = (row["rsmi"], row["psmi"], float(row["ea"]), float(row["dh"]))
    return rows


def read_archive(limit: int = 0):
    """{reaction index: {side: parsed log}} from the archive, read as a stream."""
    got = {}
    t0 = time.time()
    with tarfile.open(os.path.join(CACHE, "wb97xd3.tar.gz"), "r|gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".log"):
                continue
            base = os.path.basename(m.name)[:-4]
            side = "ts" if base.startswith("ts") else base[0]
            if side not in SIDES:
                continue
            idx = int(base[len(side):])
            text = tf.extractfile(m).read().decode("utf-8", "replace")
            try:
                got.setdefault(idx, {})[side] = parse_log(text)
            except ValueError:
                got.setdefault(idx, {})[side] = None
            if len(got) % 1000 == 0 and side == "ts":
                _say(f"  {len(got)} reactions read, {time.time() - t0:.0f} s")
            if limit and len(got) > limit:
                got.pop(idx)
                break
    return got


def prepare(limit: int = 0) -> dict:
    RDLogger.DisableLog("rdApp.*")
    table = read_table()
    logs = read_archive(limit)
    refused = {}
    kept, coords, sizes, numbers = [], {s: [] for s in SIDES}, [], []
    energies = []
    stretch = []
    worst_ea = worst_dh = 0.0
    products = {}
    for idx in sorted(logs):
        entry = logs[idx]

        def refuse(why):
            refused[why] = refused.get(why, 0) + 1

        if idx not in table:
            refuse("no row in the table")
            continue
        if any(entry.get(s) is None for s in SIDES):
            refuse("a log is missing or has no geometry")
            continue
        rsmi, psmi, ea, dh = table[idx]
        try:
            rx = read_reaction(rsmi, psmi, entry["r"][1], entry["p"][1], f"rxn{idx:06d}")
        except ReactionMismatch as err:
            refuse("not read: " + str(err).split(":")[0][:40])
            continue
        want = rx.symbols
        if any(entry[s][0] != want for s in SIDES):
            refuse("the elements of a geometry are not in the order of the map numbers")
            continue
        s_r, s_p = bond_stretch(rx.reactant), bond_stretch(rx.product)
        if max(s_r, s_p) > STRETCH_LIMIT:
            refuse(f"a bond of the graph is longer than {STRETCH_LIMIT} times the covalent sum")
            continue
        # the two targets, from the logs: differences of the electronic energy plus the
        # zero-point energy, to the transition state and to the product
        e = {s: entry[s][2] * HARTREE for s in SIDES}
        ea_log = (e["ts"] + entry["ts"][3]) - (e["r"] + entry["r"][3])
        dh_log = (e["p"] + entry["p"][3]) - (e["r"] + entry["r"][3])
        if np.isfinite(ea_log):
            worst_ea = max(worst_ea, abs(ea_log - ea))
        if np.isfinite(dh_log):
            worst_dh = max(worst_dh, abs(dh_log - dh))
        n_prod = len(Chem.GetMolFrags(rx.product))
        products[n_prod] = products.get(n_prod, 0) + 1
        kept.append(idx)
        sizes.append(rx.n_atoms)
        numbers.append([a.GetAtomicNum() for a in rx.reactant.GetAtoms()])
        for s in SIDES:
            coords[s].append(entry[s][1])
        energies.append([e["r"], e["p"], e["ts"], ea, dh, ea_log, dh_log])
        stretch.append([s_r, s_p])
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(int)
    np.savez_compressed(
        TABLE, idx=np.array(kept, int), offsets=offsets,
        numbers=np.concatenate(numbers).astype(np.int8),
        r=np.concatenate(coords["r"]).astype(np.float32),
        p=np.concatenate(coords["p"]).astype(np.float32),
        ts=np.concatenate(coords["ts"]).astype(np.float32),
        energies=np.array(energies, float), stretch=np.array(stretch, float),
        rsmi=np.array([table[i][0] for i in kept]), psmi=np.array([table[i][1] for i in kept]))
    sizes = np.array(sizes)
    heavy = np.array([sum(1 for z in zs if z > 1) for zs in numbers])
    report = dict(
        rows_in_table=len(table), reactions_in_archive=len(logs), kept=len(kept),
        refused=refused, atoms=dict(least=int(sizes.min()), most=int(sizes.max()),
                                    mean=float(sizes.mean())),
        heavy_atoms=dict(least=int(heavy.min()), most=int(heavy.max())),
        elements=sorted({Chem.GetPeriodicTable().GetElementSymbol(int(z))
                         for zs in numbers for z in zs}),
        product_molecules={str(k): v for k, v in sorted(products.items())},
        largest_difference_kcal=dict(activation_energy=worst_ea, reaction_energy=worst_dh),
        longest_bond_as_multiple_of_covalent_sum=float(np.max(stretch)),
        source="Zenodo 10.5281/zenodo.3715478, CC BY 4.0", limit=limit)
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=1)
    return report


def load():
    """The prepared table as a list of dicts, one per reaction."""
    # every array is read once: an .npz file decompresses an array again at each access
    with np.load(TABLE) as z:
        a = {name: z[name] for name in ("idx", "offsets", "rsmi", "psmi", "r", "p", "ts",
                                        "energies")}
    out = []
    for k, idx in enumerate(a["idx"]):
        sl = slice(int(a["offsets"][k]), int(a["offsets"][k + 1]))
        out.append(dict(idx=int(idx), rsmi=str(a["rsmi"][k]), psmi=str(a["psmi"][k]),
                        r=a["r"][sl].astype(float), p=a["p"][sl].astype(float),
                        ts=a["ts"][sl].astype(float), ea=float(a["energies"][k, 3]),
                        dh=float(a["energies"][k, 4])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--download", action="store_true", help="fetch the two files from Zenodo")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many reactions of the archive (for a trial)")
    args = ap.parse_args()
    if args.download:
        download()
    report = prepare(args.limit)
    _say(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
