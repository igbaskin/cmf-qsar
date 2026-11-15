#!/usr/bin/env python3
"""
Static VASP runs that leave the three grids of plan item M9: CHGCAR, ELFCAR and LOCPOT.

This script runs *inside WSL*, where VASP is installed; `benchmarks/run_gridfields.py` writes the
cells it needs to an extended-XYZ file and calls it. Each cell is one self-consistent calculation
without relaxation, in a directory of its own under the work directory. After a run only the
three grids and the OUTCAR are kept, compressed with gzip; the wave functions are not written.

The settings are one set for every cell, so that the grids of two cells are comparable:

* PBE, PAW, a plane-wave cutoff of 450 eV for every cell;
* no spin polarisation: the grids are fields, and the target of the set is not recomputed;
* Gaussian smearing of 0.1 eV, a Gamma-centred 6 x 6 x 6 k-point mesh (the cells are 3.6 to 4.6 A);
* LCHARG for the valence density, LELF for the electron localisation function, and LVHAR for the
  electrostatic (Hartree plus ionic) potential in LOCPOT, without exchange and correlation.

Usage (inside WSL, after `source ~/vasp_env.sh`):

    python3 grid_dft_wsl.py CELLS.extxyz WORKDIR [--ranks 8] [--first 0] [--last N]

A cell whose directory already holds `done.json` is skipped, so the script can be stopped and
started again.
"""
import argparse
import gzip
import json
import os
import shutil
import subprocess
import time

from ase.io import read

#: the POTCAR of each element: the plain one where the local PAW set has it, otherwise the
#: smallest set with semicore states that it does have
SEMICORE = {"Ba": "_sv", "Ca": "_pv", "Cs": "_sv", "K": "_pv", "Nb": "_pv", "Rb": "_pv",
            "Sr": "_sv", "Y": "_sv", "Zr": "_sv"}
INCAR = {"ENCUT": 450, "PREC": "Normal", "ISPIN": 1, "ISMEAR": 0, "SIGMA": 0.1,
         "EDIFF": 1e-5, "NELM": 150, "ALGO": "Normal", "IBRION": -1, "NSW": 0,
         "LCHARG": ".TRUE.", "LAECHG": ".FALSE.", "LELF": ".TRUE.", "LVHAR": ".TRUE.",
         "LWAVE": ".FALSE.", "LORBIT": 0}
KPOINTS = "Gamma-centred mesh\n0\nGamma\n6 6 6\n0 0 0\n"
KEEP = ("CHGCAR", "ELFCAR", "LOCPOT", "OUTCAR")


def potcar(species, pp_dir):
    parts = []
    for s in species:
        path = os.path.join(pp_dir, "potpaw_PBE", s + SEMICORE.get(s, ""), "POTCAR")
        with open(path) as fh:
            parts.append(fh.read())
    return "".join(parts)


def write_inputs(atoms, run, pp_dir):
    os.makedirs(run, exist_ok=True)
    order = sorted(range(len(atoms)), key=lambda i: atoms[i].symbol)
    atoms = atoms[order]
    species = list(dict.fromkeys(atoms.get_chemical_symbols()))
    counts = [atoms.get_chemical_symbols().count(s) for s in species]
    lines = [atoms.info.get("name", "cell"), "1.0"]
    lines += ["  %.10f %.10f %.10f" % tuple(v) for v in atoms.cell]
    lines += ["  " + " ".join(species), "  " + " ".join(map(str, counts)), "Direct"]
    lines += ["  %.10f %.10f %.10f" % tuple(p) for p in atoms.get_scaled_positions(wrap=True)]
    with open(os.path.join(run, "POSCAR"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(run, "INCAR"), "w") as fh:
        fh.write("".join(f"{k} = {v}\n" for k, v in INCAR.items()))
    with open(os.path.join(run, "KPOINTS"), "w") as fh:
        fh.write(KPOINTS)
    with open(os.path.join(run, "POTCAR"), "w") as fh:
        fh.write(potcar(species, pp_dir))


def converged(run):
    try:
        with open(os.path.join(run, "OUTCAR"), errors="ignore") as fh:
            text = fh.read()
    except OSError:
        return False
    return "aborting loop because EDIFF is reached" in text


def run_one(atoms, run, ranks, pp_dir):
    done = os.path.join(run, "done.json")
    if os.path.isfile(done):
        return json.load(open(done))
    write_inputs(atoms, run, pp_dir)
    t0 = time.time()
    with open(os.path.join(run, "vasp.log"), "w") as log:
        code = subprocess.call(["mpirun", "-np", str(ranks), "vasp_std"], cwd=run,
                               stdout=log, stderr=subprocess.STDOUT)
    ok = code == 0 and converged(run) and all(
        os.path.isfile(os.path.join(run, f)) for f in KEEP)
    for f in KEEP:
        path = os.path.join(run, f)
        if os.path.isfile(path):
            with open(path, "rb") as src, gzip.open(path + ".gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
    for f in os.listdir(run):
        if f not in ("POSCAR", "INCAR", "KPOINTS", "done.json", "vasp.log") \
                and not f.endswith(".gz"):
            os.remove(os.path.join(run, f))
    rec = {"name": atoms.info.get("name", ""), "ok": bool(ok), "exit": int(code),
           "seconds": round(time.time() - t0, 1)}
    with open(done, "w") as fh:
        json.dump(rec, fh)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("cells", help="extended-XYZ file with a 'name' in each frame's info")
    ap.add_argument("workdir", help="directory that receives one subdirectory per cell")
    ap.add_argument("--ranks", type=int, default=8, help="MPI ranks per VASP run")
    ap.add_argument("--first", type=int, default=0, help="index of the first cell to run")
    ap.add_argument("--last", type=int, default=None, help="index after the last cell to run")
    args = ap.parse_args()
    pp_dir = os.environ["VASP_PP_PATH"]
    cells = read(args.cells, index=":")
    for i, atoms in enumerate(cells[args.first:args.last], start=args.first):
        name = atoms.info.get("name", f"cell_{i:04d}")
        rec = run_one(atoms, os.path.join(args.workdir, name), args.ranks, pp_dir)
        print(time.strftime("%H:%M:%S"), i, name, "ok" if rec["ok"] else "FAILED",
              rec["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
