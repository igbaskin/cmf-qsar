#!/usr/bin/env python
"""
Structures that are not molecules, read from six file formats (plan item NT1).

`cmfqsar/structures.py` reads crystals, layers, chains and particles into the same `Dataset` the
molecules use, with a name, a target and a training or test label for each.  This benchmark asks
what each file format keeps of such a set, and whether what comes back is the structure that was
written: the same names, the same target values, the same periodic directions, the same atoms,
and the same periodic channels.

Stage formats
-------------
Four sets, one of each kind of boundary:

* crystals -- the 60 smallest cells of the melting-point set (`examples/melting_points`),
  target the melting point in degrees Celsius;
* layers -- the 139 bare sheets of the single-atom catalyst set (`examples/sac`), periodic in
  two directions (the file says three, as most files do), target the DFT energy of the sheet;
* chains -- 39 carbon nanotubes (n, m) built by ASE, periodic along the tube only, with no
  lattice vector across it, target the tube diameter;
* particles -- 30 metal clusters (icosahedra, decahedra and octahedra of Cu, Ag, Au, Pd, Pt,
  13 to 309 atoms) built by ASE with no cell, target the EMT energy per atom (ASE's effective
  medium potential), stored as a calculator result where a format allows it.

Each set is written in six formats -- extended XYZ, an ASE SQLite database, an ASE JSON
database, an ASE trajectory, a folder of CIF files and a folder of POSCAR files -- with the
target as an info field where the format holds one and in a CSV table beside the folder where it
does not.  Each is read back by `load_structures` as the program reads it, and once more with the
periodic directions given (`pbc`), which is the repair for a format that cannot state them.
Compared with what was written: how many structures came back, the fraction with the right name
and the right target, the fraction with the right periodic directions, the largest atom
displacement (through the cell along a periodic direction), the largest change of a distance
between two atoms, and the largest difference of the periodic autocorrelation rows divided by
the largest row entry.  For the clusters, the rows are also compared with those of each cluster
alone in a 60 A periodic box with the background term switched off.

Stage speed
-----------
The time to read all 665 cells of the melting-point set from each format, with the target.

Usage
-----
    python benchmarks/run_structures.py --stage formats     # about 2 minutes
    python benchmarks/run_structures.py --stage speed       # about 1 minute
    python benchmarks/run_structures.py --report
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
import warnings
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import structures as S  # noqa: E402
from cmfqsar.periodic import (PeriodicSpec, crystal_table, elements_in,  # noqa: E402
                              inorganic_fieldset, periodic_feature_matrix)

RESULTS = os.path.join(HERE, "results", "structures")
CACHE = os.path.join(HERE, "cache", "structures")
MELT = os.path.join(ROOT, "examples", "melting_points")
SAC = os.path.join(ROOT, "examples", "sac")
FORMATS = ("extxyz", "db", "json", "traj", "cif", "poscar")
FORMAT_NAMES = {"extxyz": "extended XYZ", "db": "ASE database (SQLite)",
                "json": "ASE database (JSON)", "traj": "ASE trajectory",
                "cif": "folder of CIF files", "poscar": "folder of POSCAR files"}
#: the periodic directions of each set, as the letters the reader takes
PBC = {"crystals": "abc", "layers": "ab", "chains": "c", "particles": "none"}
SPEC = PeriodicSpec(r_max=8.0, step=0.5, sigma=0.5, taper=2.0)
#: fields that need no oxidation states: guessing them for a sheet of 60 carbon and nitrogen
#: atoms around a metal is a combinatorial search that takes minutes per sheet
PHYSCHEM = ("steric", "prop:electronegativity")


def jdump(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def jload(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# The four sets
# --------------------------------------------------------------------------- #
def crystal_set(n: Optional[int] = 60) -> list:
    ds = S.load_structures(os.path.join(MELT, "crystals.extxyz"),
                           target_file=os.path.join(MELT, "melting_points.csv"),
                           target_column="mp_C")
    order = np.argsort([len(a) for a in ds.structures], kind="stable")
    keep = order if n is None else order[:n]
    out = []
    for i in keep:
        a = ds.structures[i].copy()
        a.info = {"name": ds.names[i], "target": float(ds.activity[i])}
        out.append(a)
    return out


def layer_set() -> list:
    ds = S.load_structures(os.path.join(SAC, "substrates.extxyz"),
                           target_file=os.path.join(SAC, "sac_hoer.csv"),
                           target_column="E_sub", pbc="ab")
    out = []
    for a, nm, y in zip(ds.structures, ds.names, ds.activity):
        if not np.isfinite(y):
            continue
        b = a.copy()
        b.arrays.pop("momenta", None)
        b.info = {"name": nm, "target": float(y)}
        out.append(b)
    return out


def chain_set() -> list:
    from ase.build import nanotube
    out = []
    for n in range(4, 11):
        for m in range(0, n + 1):
            if len(out) >= 39:
                break
            t = nanotube(n, m, length=1, bond=1.42)
            if len(t) > 400:
                continue
            d = 1.42 * np.sqrt(3.0) * np.sqrt(n * n + n * m + m * m) / np.pi
            t.info = {"name": f"cnt_{n}_{m}", "target": float(d)}
            out.append(t)
    return out


def particle_set() -> list:
    from ase.calculators.emt import EMT
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.cluster import Decahedron, Icosahedron, Octahedron
    out = []
    for el in ("Cu", "Ag", "Au", "Pd", "Pt"):
        shapes = [("ico", Icosahedron(el, 2)), ("ico", Icosahedron(el, 3)),
                  ("ico", Icosahedron(el, 4)), ("ico", Icosahedron(el, 5)),
                  ("deca", Decahedron(el, 3, 2, 0)), ("octa", Octahedron(el, 5, 1))]
        for shape, c in shapes:
            c.center()
            c.calc = EMT()
            e = float(c.get_potential_energy())
            c.calc = SinglePointCalculator(c, energy=e)
            c.info = {"name": f"{el}{len(c)}_{shape}", "target": e / len(c)}
            out.append(c)
    return out


SETS = {"crystals": crystal_set, "layers": layer_set, "chains": chain_set,
        "particles": particle_set}


# --------------------------------------------------------------------------- #
# Writing and reading
# --------------------------------------------------------------------------- #
def write_set(structs: list, fmt: str, folder: str) -> Dict[str, str]:
    """Write the set in one format; returns the path to read and the CSV of targets, if any."""
    import ase.db
    from ase.io import write
    if os.path.isdir(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)
    csv_path = os.path.join(folder, "targets.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "target"])
        for a in structs:
            w.writerow([a.info["name"], repr(a.info["target"])])
    if fmt == "extxyz":
        path = os.path.join(folder, "set.extxyz")
        write(path, structs, format="extxyz")
        return {"path": path, "target": "target"}
    if fmt in ("db", "json"):
        path = os.path.join(folder, f"set.{fmt}")
        db = ase.db.connect(path, append=False)
        for a in structs:
            db.write(a, name=a.info["name"], target=a.info["target"])
        return {"path": path, "target": "target"}
    if fmt == "traj":
        path = os.path.join(folder, "set.traj")
        write(path, structs, format="traj")
        return {"path": path, "target": "target"}
    sub = os.path.join(folder, "files")
    os.makedirs(sub)
    for a in structs:
        if fmt == "cif":
            write(os.path.join(sub, a.info["name"] + ".cif"), a, format="cif")
        else:
            write(os.path.join(sub, a.info["name"] + ".poscar"), a, format="vasp", direct=True)
    return {"path": sub, "target_file": csv_path}


def read_back(where: Dict[str, str], pbc: Optional[str]):
    return S.load_structures(where["path"], target=where.get("target"),
                             target_file=where.get("target_file"), pbc=pbc)


def displacement(a, b) -> float:
    """The largest distance between matching atoms, through the cell along periodic directions."""
    if len(a) != len(b) or a.get_chemical_symbols() != b.get_chemical_symbols():
        return float("inf")
    d = b.positions - a.positions
    pbc = np.array(a.get_pbc(), bool)
    cell = np.asarray(a.get_cell(), float)
    if pbc.any() and abs(np.linalg.det(cell)) > 1e-9:
        f = np.linalg.solve(cell.T, d.T).T
        f[:, pbc] -= np.round(f[:, pbc])
        d = f @ cell
    elif pbc.any():
        for k in np.flatnonzero(pbc):
            ln = float(np.linalg.norm(cell[k]))
            u = cell[k] / ln
            d = d - np.outer(ln * np.round((d @ u) / ln), u)
    return float(np.max(np.linalg.norm(d, axis=1))) if len(d) else 0.0


def distance_change(a, b) -> float:
    """The largest change of a distance between two atoms (the nearest images where periodic).

    A rotation or a shift of the whole structure changes no distance, so this separates a
    structure that was moved from one that was changed, which the displacement cannot.
    """
    if len(a) != len(b) or a.get_chemical_symbols() != b.get_chemical_symbols():
        return float("inf")
    return float(np.max(np.abs(a.get_all_distances(mic=True) - b.get_all_distances(mic=True))))


def features(structs: list, fs) -> np.ndarray:
    table = crystal_table(structs, fs, [a.info.get("name", "") for a in structs])
    return periodic_feature_matrix(table.molecules, SPEC, n_fields=len(fs.specs))


def box_identity(structs: list, fs, F0: np.ndarray, side: float = 60.0) -> float:
    """A cluster with no cell against the same cluster alone in a periodic box.

    With no periodic direction the periodic autocorrelation has no background term, so it should
    equal the descriptor of the cluster in a box wider than the cutoff with the background term
    switched off.  Returns the largest difference divided by the largest entry.
    """
    from dataclasses import replace
    boxed = []
    for a in structs:
        b = a.copy()
        b.set_cell([side, side, side])
        b.center()
        b.set_pbc(True)
        boxed.append(b)
    table = crystal_table(boxed, fs, [a.info["name"] for a in structs])
    B = periodic_feature_matrix(table.molecules, replace(SPEC, background=False),
                                n_fields=len(fs.specs))
    return float(np.max(np.abs(F0 - B)) / max(np.max(np.abs(B)), 1e-300))


def compare(written: list, ds, F0: Optional[np.ndarray], fs) -> Dict:
    by_name = {nm: i for i, nm in enumerate(ds.names)}
    got = [by_name.get(a.info["name"]) for a in written]
    n_named = sum(g is not None for g in got)
    y_ok = pbc_ok = 0
    disp, dist = [], []
    for a, g in zip(written, got):
        if g is None:
            continue
        b = ds.structures[g]
        y = ds.activity[g]
        if np.isfinite(y) and abs(y - a.info["target"]) <= 1e-9 * max(1.0, abs(y)):
            y_ok += 1
        if list(map(bool, b.get_pbc())) == list(map(bool, a.get_pbc())):
            pbc_ok += 1
        disp.append(displacement(a, b))
        dist.append(distance_change(a, b))
    out = {"read": len(ds), "named": n_named / len(written), "target": y_ok / len(written),
           "pbc": pbc_ok / len(written),
           "max_displacement": float(max(disp)) if disp else None,
           "max_distance_change": float(max(dist)) if dist else None,
           "kernel": None, "kernel_note": ""}
    if F0 is not None and n_named == len(written) and np.isfinite(out["max_displacement"]):
        try:
            F1 = features([ds.structures[g] for g in got], fs)
            out["kernel"] = float(np.max(np.abs(F1 - F0)) / max(np.max(np.abs(F0)), 1e-300))
        except Exception as exc:                     # noqa: BLE001 - recorded, not hidden
            out["kernel_note"] = f"{type(exc).__name__}: {exc}"[:160]
    return out


def stage_formats() -> None:
    warnings.filterwarnings("ignore")
    rows = []
    for kind, make in SETS.items():
        t0 = time.time()
        structs = make()
        fs = inorganic_fieldset(elements_in(structs, 1), physchem=PHYSCHEM)
        F0, note, boxed = None, "", None
        try:
            F0 = features(structs, fs)
        except Exception as exc:                     # noqa: BLE001
            note = f"{type(exc).__name__}: {exc}"[:160]
        if kind == "particles" and F0 is not None:
            boxed = box_identity(structs, fs, F0)
        print(f"{kind}: {len(structs)} structures, {min(map(len, structs))}-"
              f"{max(map(len, structs))} atoms, features {time.time() - t0:.1f} s {note}")
        for fmt in FORMATS:
            folder = os.path.join(CACHE, kind, fmt)
            try:
                where = write_set(structs, fmt, folder)
            except Exception as exc:                 # noqa: BLE001 - a format that cannot hold it
                rows.append({"set": kind, "format": fmt, "pbc_given": False,
                             "error": f"writing: {type(exc).__name__}: {exc}"[:160]})
                continue
            for given in (False, True):
                try:
                    ds = read_back(where, PBC[kind] if given else None)
                    r = compare(structs, ds, F0, fs)
                except Exception as exc:             # noqa: BLE001
                    r = {"error": f"reading: {type(exc).__name__}: {exc}"[:160]}
                r.update({"set": kind, "format": fmt, "pbc_given": given})
                rows.append(r)
                print(f"  {fmt:7s} pbc given {str(given):5s} {r}")
        rows.append({"set": kind, "summary": True, "n": len(structs),
                     "atoms": [int(min(map(len, structs))), int(max(map(len, structs)))],
                     "reference_note": note, "box_identity": boxed})
    jdump({"rows": rows, "spec": {"r_max": SPEC.r_max, "step": SPEC.step,
                                  "sigma": SPEC.sigma, "taper": SPEC.taper}},
          os.path.join(RESULTS, "formats.json"))


def stage_speed() -> None:
    warnings.filterwarnings("ignore")
    structs = crystal_set(None)
    rows = []
    for fmt in FORMATS:
        folder = os.path.join(CACHE, "speed", fmt)
        where = write_set(structs, fmt, folder)
        size = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(folder)
                   for f in fs if f != "targets.csv")
        times = []
        for _ in range(3):
            t0 = time.perf_counter()
            ds = read_back(where, None)
            times.append(time.perf_counter() - t0)
        ok = int(np.isfinite(ds.activity).sum())
        rows.append({"format": fmt, "structures": len(ds), "with_target": ok,
                     "atoms": int(sum(len(a) for a in ds.structures)),
                     "seconds": float(min(times)), "megabytes": size / 1e6})
        print(rows[-1])
    jdump({"rows": rows}, os.path.join(RESULTS, "speed.json"))


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _num(x) -> str:
    return "--" if x is None else ("atoms differ" if not np.isfinite(x) else f"{x:.1e}")


def _frac(x) -> str:
    return "--" if x is None else f"{100 * x:.0f} %"


def report() -> str:
    lines = ["# Structures that are not molecules, from six file formats (plan item NT1)", ""]
    fp = os.path.join(RESULTS, "formats.json")
    if os.path.isfile(fp):
        R = jload(fp)
        sums = {r["set"]: r for r in R["rows"] if r.get("summary")}
        lines += ["## What each format keeps", "",
                  "Each set is written in the format and read back as the program reads it "
                  "('as read'), then again with the periodic directions given ('directions "
                  "given'). The kernel difference is the largest difference of the periodic "
                  "autocorrelation rows (cutoff "
                  f"{R['spec']['r_max']:g} A, channels every {R['spec']['step']:g} A) divided "
                  "by the largest entry. A displacement with no change of a distance is a "
                  "rotation or a shift of the whole structure: a CIF file stores the lengths "
                  "and angles of the cell, so the reader places the cell in the standard "
                  "orientation. 'Refused' gives the reason ASE gives when it cannot write the "
                  "structure in that format.", ""]
        for kind in ("crystals", "layers", "chains", "particles"):
            if kind not in sums:
                continue
            s = sums[kind]
            lines += [f"### {kind.capitalize()}: {s['n']} structures, {s['atoms'][0]} to "
                      f"{s['atoms'][1]} atoms", "",
                      "| Format | Directions | Read | Right name | Right target | Right "
                      "periodic directions | Largest atom displacement, A | Largest change "
                      "of a distance, A | Kernel difference |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in R["rows"]:
                if r.get("summary") or r["set"] != kind:
                    continue
                given = "given" if r["pbc_given"] else "as read"
                if "error" in r:
                    lines.append(f"| {FORMAT_NAMES[r['format']]} | {given} | refused: "
                                 f"{r['error']} | | | | | | |")
                    continue
                k = r["kernel"]
                ktxt = ("--" if k is None and not r["kernel_note"] else
                        "refused" if k is None else f"{k:.1e}")
                d = r["max_displacement"]
                dtxt = "--" if d is None else ("atoms differ" if not np.isfinite(d)
                                               else f"{d:.1e}")
                lines.append(f"| {FORMAT_NAMES[r['format']]} | {given} | {r['read']} | "
                             f"{_frac(r['named'])} | {_frac(r['target'])} | "
                             f"{_frac(r['pbc'])} | {dtxt} | {_num(r.get('max_distance_change'))} "
                             f"| {ktxt} |")
            notes = [r["kernel_note"] for r in R["rows"]
                     if not r.get("summary") and r["set"] == kind and r.get("kernel_note")]
            if s.get("reference_note") or notes:
                lines += ["", "Kernel refused: " + (s.get("reference_note") or notes[0])]
            if s.get("box_identity") is not None:
                lines += ["", "With no periodic direction the periodic autocorrelation has no "
                          "background term. The channels of each cluster with no cell equal "
                          "those of the same cluster alone in a 60 A periodic box with the "
                          f"background term switched off to {s['box_identity']:.1e} of the "
                          "largest entry."]
            lines.append("")
    sp = os.path.join(RESULTS, "speed.json")
    if os.path.isfile(sp):
        rows = jload(sp)["rows"]
        lines += ["## Reading speed", "",
                  f"All {rows[0]['structures']} cells of the melting-point set "
                  f"({rows[0]['atoms']:,} atoms), the best of three reads, with the target.", "",
                  "| Format | Structures read | With a target | Seconds | Milliseconds per "
                  "structure | Size on disk, MB |", "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            lines.append(f"| {FORMAT_NAMES[r['format']]} | {r['structures']} | "
                         f"{r['with_target']} | {r['seconds']:.2f} | "
                         f"{1000 * r['seconds'] / max(r['structures'], 1):.2f} | "
                         f"{r['megabytes']:.1f} |")
        lines.append("")
    text = "\n".join(lines)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "table.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("formats", "speed"), action="append",
                    help="stage(s) to run; the report always follows")
    ap.add_argument("--report", action="store_true", help="only rebuild the table")
    args = ap.parse_args(argv)
    for st in args.stage or []:
        if st == "formats":
            stage_formats()
        elif st == "speed":
            stage_speed()
    print(report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
