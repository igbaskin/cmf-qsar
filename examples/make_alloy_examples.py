#!/usr/bin/env python
"""
The bimetallic alloy surfaces of Mamun et al. as a data set of H and OH adsorption energies
(plan item EC12).

Source: O. Mamun, K. T. Winther, J. R. Boes and T. Bligaard, Scientific Data 6 (2019) 76, "High-
throughput calculations of catalytic properties of bimetallic alloy surfaces" (CC BY 4.0).  The raw
Quantum ESPRESSO logs (BEEF-vdW) are in the Materials Cloud record xxdhq-f1v83
(doi 10.24435/materialscloud:2019.0015).  Two of its archives are used:

    H.tar.gz           17,212 relaxation logs of slabs with H or OH on them
    references.tar.gz  4,060 logs of bare slabs, bulk cells and gas molecules

Put both in ``benchmarks/cache/alloy_her`` (git-ignored).  Check them against the MD5 sums of the
record: H.tar.gz 3280ca9c6445ae4b670978928a42291e, references.tar.gz
f9c4b46d909ef69ef877fd1b9f974faf.

Stages
------
--scan    read every log once and keep the last structure and the energy (no extraction to disk)
--build   match every adsorbed slab to its bare slab, compute the adsorption energy, find the
          site, remove duplicates, and write ``examples/alloy``

Every slab is a 2 x 2 cell of three metal layers (12 atoms); the logs have numbered names, so the
composition comes from the atoms.  The adsorption energies are

    E_ads(H)  = E(slab + H)  - E(slab) - 1/2 E(H2)
    E_ads(OH) = E(slab + OH) - E(slab) - (E(H2O) - 1/2 E(H2))

Four traps of the logs are handled here (see the README of ``examples/alloy``):

1. The last ATOMIC_POSITIONS block of a relaxation is the next step, which was never computed.
2. ASE's reader fails on a spin-polarised log whose band count changes; this reader does not use
   it, but it takes ASE's value of the Rydberg constant, so that the energies equal ASE's.
3. 458 logs are several single-point runs written one after another: the last run counts.
4. The adsorbed slab and its bare slab can be written with other cell vectors, another origin, or
   as a rotated or mirrored copy.  They are matched by the distances of the atoms over the 12
   operations of the hexagonal layer and every shift of one atom onto another.

Usage
-----
    python examples/make_alloy_examples.py --scan
    python examples/make_alloy_examples.py --build [--workers 4]
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import pickle
import re
import sys
import tarfile
import time
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "benchmarks", "cache", "alloy_her")
OUT = os.path.join(HERE, "alloy")
ARCHIVES = {"H": "H.tar.gz", "ref": "references.tar.gz"}

BOHR = 0.529177210903
#: a relaxation that reached this number of steps stopped at the step limit of the source
STEP_LIMIT = 50
#: atoms within this height of the top atom form the top layer, in angstrom
LAYER_TOLERANCE = 0.6
#: a bare slab matches an adsorbed slab if every atom is nearer than this, in angstrom
MATCH_LIMIT = 1.0
#: the search stops at a bare slab that matches this well
MATCH_GOOD = 0.35
#: neighbours of the adsorbate in the top layer: within this much of the nearest one, in angstrom
BIND_WINDOW = 0.5
#: two logs are the same adsorption geometry if the adsorbate sits within this (in plane), the
#: heights differ by less than DUP_HEIGHT and the energies by less than DUP_ENERGY
DUP_PLANE, DUP_HEIGHT, DUP_ENERGY = 0.4, 0.3, 0.02
#: the adsorbate is below the top layer by more than this: subsurface, in angstrom
SUBSURFACE = -0.3


# --------------------------------------------------------------------------- #
# Reading the logs
# --------------------------------------------------------------------------- #
def rydberg() -> float:
    from ase.io.espresso import units
    return float(units["Ry"])


def parse_log(data: str, ry: float) -> dict:
    """The last computed structure, the energy and the bookkeeping of one Quantum ESPRESSO log."""
    m = re.search(r"lattice parameter \(alat\)\s*=\s*([\d.]+)\s*a\.u\.", data)
    alat = float(m.group(1)) * BOHR
    axes = [[float(x) for x in g] for g in re.findall(
        r"a\(\d\)\s*=\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\)", data)][-3:]
    cell = np.array(axes) * alat
    energies = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", data, re.M)
    if not energies:
        raise ValueError("no energy")
    blocks = re.findall(r"ATOMIC_POSITIONS \((\w+)\)\n((?:\s*[A-Z][a-z]?\d*\s+-?[\d.]+\s+-?[\d.]+"
                        r"\s+-?[\d.]+[^\n]*\n)+)", data)
    if not blocks:
        # single-point runs written one after another: the table of the last run
        # (a run that stopped before its first energy is skipped: its table has no energy)
        runs = [r for r in re.split(r"Program PWSCF", data) if re.search(r"^!\s+total energy", r, re.M)]
        run = runs[-1] if runs else data
        site = re.findall(r"^\s+\d+\s+([A-Z][a-z]?)\d*\s+tau\(\s*\d+\)\s*=\s*\(\s*(-?[\d.]+)\s+"
                          r"(-?[\d.]+)\s+(-?[\d.]+)\s*\)", run, re.M)
        if not site:
            raise ValueError("no positions")
        nat = int(re.findall(r"number of atoms/cell\s*=\s*(\d+)", run)[-1])
        blocks = [("alat", "\n".join(f"{a} {x} {y} {z}" for a, x, y, z in site[-nat:]) + "\n")]
        energies = re.findall(r"^!\s+total energy\s+=\s+(-?[\d.]+)\s+Ry", run, re.M)[-1:]

    def read_block(block):
        unit, text = block
        sym, pos = [], []
        for line in text.strip().splitlines():
            t = line.split()
            sym.append(re.sub(r"\d+$", "", t[0]))
            pos.append([float(x) for x in t[1:4]])
        pos = np.array(pos)
        if unit == "alat":
            pos = pos * alat
        elif unit == "crystal":
            pos = pos @ cell
        elif unit == "bohr":
            pos = pos * BOHR
        return sym, pos

    # the block after step k holds the positions of step k + 1, so the last computed step is
    # the one before the last block
    last = len(energies) - 2 if len(energies) >= 2 and len(blocks) >= len(energies) - 1 else -1
    sym, pos = read_block(blocks[last])
    mag = re.findall(r"total magnetization\s+=\s+(-?[\d.]+)", data)
    return {"symbols": sym, "pos": pos, "cell": cell, "energy": float(energies[-1]) * ry,
            "nsteps": len(energies), "mag": float(mag[-1]) if mag else None,
            "finished": "JOB DONE" in data}


def scan(archive: str, out: str, log=print) -> None:
    ry = rydberg()
    rows, fails = [], []
    t0 = time.time()
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".log") or "/._" in m.name:
                continue
            data = tf.extractfile(m).read().decode("utf-8", "replace")
            try:
                r = parse_log(data, ry)
                r["name"] = m.name
                rows.append(r)
            except Exception as e:           # noqa: BLE001 - the reason is kept
                fails.append((m.name, repr(e)[:120]))
            if (len(rows) + len(fails)) % 2000 == 0:
                log(f"  {len(rows)} read, {len(fails)} failed, {time.time() - t0:.0f} s")
    with open(out, "wb") as fh:
        pickle.dump({"rows": rows, "fails": fails}, fh)
    log(f"{os.path.basename(archive)}: {len(rows)} logs, {len(fails)} failed, "
        f"{time.time() - t0:.0f} s")


# --------------------------------------------------------------------------- #
# Matching an adsorbed slab to its bare slab
# --------------------------------------------------------------------------- #
def layer_operations() -> List[np.ndarray]:
    """The 12 operations of the hexagonal layer about the surface normal."""
    ops = []
    for k in range(6):
        a = np.pi * k / 3
        rot = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        ops.append(rot)
        ops.append(rot @ np.diag([1.0, -1.0, 1.0]))
    return ops


OPS = layer_operations()


def match_one(sym: List[str], p: np.ndarray, cell: np.ndarray, bare: dict) -> Tuple[float, np.ndarray]:
    """(worst atom distance, shift) of the adsorbate-free slab on a bare slab, shifting in plane.

    The shift puts one atom of the rarest element on each atom of that element of the bare slab
    in turn; the other atoms must then find an atom of their own element, each used once.
    """
    inv = np.linalg.inv(cell)
    symq = np.array(bare["symbols"])
    q = bare["pos"]
    best, best_shift = 99.0, np.zeros(3)
    rare = min(set(sym), key=sym.count)
    i0 = sym.index(rare)
    for j0 in np.where(symq == rare)[0]:
        shift = q[j0] - p[i0]
        shift[2] = 0.0
        pp = p + shift
        worst, used, ok = 0.0, set(), True
        for s, pos in zip(sym, pp):
            dmin, bj = 9.0, None
            for j in np.where(symq == s)[0]:
                d = (pos - q[j]) @ inv
                d[:2] -= np.round(d[:2])
                r = float(np.linalg.norm(d @ cell))
                if r < dmin:
                    dmin, bj = r, j
            if bj is None or bj in used:
                ok = False
                break
            used.add(bj)
            worst = max(worst, dmin)
        if ok and worst < best:
            best, best_shift = worst, shift.copy()
    return best, best_shift


def metals_of(sym) -> tuple:
    return tuple(sorted(collections.Counter(s for s in sym if s not in ("H", "O")).items()))


_STATE: dict = {}


def _init(slabs, by_metals, logs):
    _STATE.update(slabs=slabs, by_metals=by_metals, logs=logs)


def _match_log(k: int):
    """(log index, bare slab, distance, operation, shift) of one adsorbed slab, or None."""
    x = _STATE["logs"][k]
    sym_all = x["symbols"]
    keep = [i for i, s in enumerate(sym_all) if s not in ("H", "O")]
    sym = [sym_all[i] for i in keep]
    p = x["pos"][keep]
    best = (None, 99.0, 0, np.zeros(3))
    for j in _STATE["by_metals"].get(metals_of(sym), []):
        for o, rot in enumerate(OPS):
            d, shift = match_one(sym, p @ rot.T, x["cell"], _STATE["slabs"][j])
            if d < best[1]:
                best = (j, d, o, shift)
            if d < MATCH_GOOD:
                return (k,) + best
    return (k,) + best


# --------------------------------------------------------------------------- #
# Sites
# --------------------------------------------------------------------------- #
def wrap_in_plane(p: np.ndarray, cell: np.ndarray) -> np.ndarray:
    f = p @ np.linalg.inv(cell)
    f[:2] -= np.floor(f[:2])
    return f @ cell


def describe_site(bare: dict, pa: np.ndarray) -> dict:
    """Where an adsorbate atom at `pa` sits on the bare slab: height, the binding atoms, the site."""
    sym = np.array(bare["symbols"])
    q = bare["pos"]
    cell = bare["cell"]
    inv = np.linalg.inv(cell)
    top_z = q[:, 2].max()
    top = np.where(q[:, 2] > top_z - LAYER_TOLERANCE)[0]
    h = float(pa[2] - q[top, 2].mean())

    def dist(i):
        d = (pa - q[i]) @ inv
        d[:2] -= np.round(d[:2])
        return float(np.linalg.norm(d @ cell))

    dt = {int(i): dist(i) for i in top}
    d1 = min(dt.values())
    bind = sorted(i for i, d in dt.items() if d < d1 + BIND_WINDOW)
    n = len(bind)
    if h < SUBSURFACE:
        site = "subsurface"
    elif n == 1:
        site = "top"
    elif n == 2:
        site = "bridge"
    else:
        # hcp when an atom of the second layer lies right below, fcc when none does
        second = [i for i in range(len(q)) if i not in set(top.tolist())
                  and q[i, 2] > top_z - 3.5]
        below = False
        for i in second:
            d = (pa - q[i]) @ inv
            d[:2] -= np.round(d[:2])
            d[2] = 0.0
            if np.linalg.norm(d @ cell) < 0.6:
                below = True
        site = "hcp" if below else "fcc"
    return {"height": h, "site": site, "n_bind": n, "d_nearest": d1,
            "bind_elements": "-".join(sorted(sym[bind]))}


def plane_distance(a: np.ndarray, b: np.ndarray, cell: np.ndarray) -> float:
    d = (a - b) @ np.linalg.inv(cell)
    d[2] = 0.0
    d[:2] -= np.round(d[:2])
    return float(np.linalg.norm(d @ cell))


# --------------------------------------------------------------------------- #
# The data set
# --------------------------------------------------------------------------- #
def load(name: str) -> List[dict]:
    with open(os.path.join(CACHE, name), "rb") as fh:
        return pickle.load(fh)["rows"]


def build(workers: int, log=print) -> None:
    ref = load("ref_logs.pkl")
    logs = load("H_logs.pkl")
    slabs = [x for x in ref if len(x["symbols"]) == 12 and x["finished"]]
    by_metals: Dict[tuple, List[int]] = collections.defaultdict(list)
    for j, x in enumerate(slabs):
        by_metals[metals_of(x["symbols"])].append(j)
    gas = {"".join(sorted(x["symbols"])): x for x in ref if len(x["symbols"]) <= 3 and x["finished"]}
    e_h2, e_h2o = gas["HH"]["energy"], gas["HHO"]["energy"]
    mu_oh = e_h2o - 0.5 * e_h2
    log(f"{len(slabs)} bare slabs; E(H2) = {e_h2:.3f} eV ({gas['HH']['name']}), "
        f"E(H2O) = {e_h2o:.3f} eV ({gas['HHO']['name']})")

    todo = []
    for k, x in enumerate(logs):
        s = x["symbols"]
        n_h, n_o = s.count("H"), s.count("O")
        if n_h == 1 and n_o == 0 and len(s) == 13:
            todo.append(k)
        elif n_h == 1 and n_o == 1 and len(s) == 14:
            todo.append(k)
    log(f"{len(todo)} logs have one H or one OH on a 12-atom slab; matching them to bare slabs")
    t0 = time.time()
    with Pool(workers, initializer=_init, initargs=(slabs, dict(by_metals), logs)) as pool:
        matches = pool.map(_match_log, todo, chunksize=20)
    log(f"matched in {time.time() - t0:.0f} s")

    rows, skipped = [], collections.Counter()
    for k, j, d, o, shift in matches:
        x = logs[k]
        if j is None:
            skipped["no bare slab of this composition"] += 1
            continue
        if d >= MATCH_LIMIT:
            skipped["no bare slab within 1 A"] += 1
            continue
        if not x["finished"]:
            skipped["log not finished"] += 1
            continue
        s = x["symbols"]
        bare = slabs[j]
        ads = "OH" if "O" in s else "H"
        e_ads = x["energy"] - bare["energy"] - (mu_oh if ads == "OH" else 0.5 * e_h2)
        ia = s.index("O") if ads == "OH" else s.index("H")
        pa = x["pos"][ia] @ OPS[o].T + shift
        pa = wrap_in_plane(pa, bare["cell"])
        row = {"log": x["name"], "slab": j, "adsorbate": ads, "e_ads": e_ads,
               "x": pa[0], "y": pa[1], "z": pa[2], "match_distance": d,
               "steps": x["nsteps"], "step_limit": x["nsteps"] >= STEP_LIMIT}
        row.update(describe_site(bare, pa))
        rows.append(row)
    log("skipped: " + ", ".join(f"{v} {k}" for k, v in skipped.items()))

    # duplicates: the same slab, adsorbate and place, the same energy.  The lowest energy stays.
    rows.sort(key=lambda r: (r["slab"], r["adsorbate"], r["e_ads"]))
    kept: Dict[tuple, List[dict]] = collections.defaultdict(list)
    for r in rows:
        r["duplicate_of"] = ""
        pa = np.array([r["x"], r["y"], r["z"]])
        cell = slabs[r["slab"]]["cell"]
        for q in kept[(r["slab"], r["adsorbate"])]:
            pq = np.array([q["x"], q["y"], q["z"]])
            if (plane_distance(pa, pq, cell) < DUP_PLANE and abs(r["z"] - q["z"]) < DUP_HEIGHT
                    and abs(r["e_ads"] - q["e_ads"]) < DUP_ENERGY):
                r["duplicate_of"] = q["log"]
                break
        else:
            kept[(r["slab"], r["adsorbate"])].append(r)
    n_dup = sum(1 for r in rows if r["duplicate_of"])
    log(f"{len(rows)} adsorbed slabs matched, {n_dup} duplicates, {len(rows) - n_dup} kept")

    os.makedirs(OUT, exist_ok=True)
    used = sorted({r["slab"] for r in rows})
    write_slabs(os.path.join(OUT, "slabs.extxyz"), slabs, used)
    write_table(os.path.join(OUT, "adsorption.csv"), rows, slabs)
    with open(os.path.join(OUT, "gas_references.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"E_H2": e_h2, "E_H2O": e_h2o, "log_H2": gas["HH"]["name"],
                   "log_H2O": gas["HHO"]["name"], "unit": "eV",
                   "note": "ASE's value of the Rydberg constant; BEEF-vdW"}, fh, indent=1)
        fh.write("\n")
    for ads in ("H", "OH"):
        r = [q for q in rows if q["adsorbate"] == ads and not q["duplicate_of"]]
        log(f"  {ads}: {len(r)} geometries on {len({q['slab'] for q in r})} slabs; "
            + ", ".join(f"{k} {v}" for k, v in collections.Counter(q['site'] for q in r).most_common()))


def formula(sym) -> str:
    c = collections.Counter(sym)
    return "".join(f"{e}{n if n > 1 else ''}" for e, n in sorted(c.items()))


def write_slabs(path: str, slabs: List[dict], used: List[int]) -> None:
    from ase import Atoms
    from ase.io import write
    frames = []
    for j in used:
        x = slabs[j]
        a = Atoms(x["symbols"], positions=x["pos"], cell=x["cell"], pbc=True)
        a.info.update(slab=j, formula=formula(x["symbols"]), energy_eV=x["energy"],
                      source_log=x["name"], steps=x["nsteps"])
        frames.append(a)
    write(path, frames, format="extxyz")


def write_table(path: str, rows: List[dict], slabs: List[dict]) -> None:
    cols = ["log", "slab", "formula", "adsorbate", "e_ads", "x", "y", "z", "height", "site",
            "n_bind", "bind_elements", "d_nearest", "match_distance", "steps", "step_limit",
            "duplicate_of"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            r = dict(r, formula=formula(slabs[r["slab"]]["symbols"]))
            w.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c] for c in cols])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", action="store_true", help="read the logs of both archives")
    ap.add_argument("--build", action="store_true", help="match, compute and write examples/alloy")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if not (args.scan or args.build):
        ap.error("give --scan or --build")
    if args.scan:
        for key, out in (("ref", "ref_logs.pkl"), ("H", "H_logs.pkl")):
            scan(os.path.join(CACHE, ARCHIVES[key]), os.path.join(CACHE, out))
    if args.build:
        build(args.workers)


if __name__ == "__main__":
    main()
