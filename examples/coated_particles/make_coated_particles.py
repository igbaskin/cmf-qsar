#!/usr/bin/env python
"""
Write `particles.extxyz`: 72 thiolate-coated gold particles, core and shell in one structure.

The shells are those of the model stage of plan item NT5, built by `coated()` in
`benchmarks/run_ligand_shells.py`, so this file and that benchmark describe the same particles:
gold icosahedra of 55 and 147 atoms, chains of 2, 4 or 6 carbon atoms on a sulfur atom, ending
in O, N or C, at 20, 35, 50 or 65 % of the surface gold atoms, one shell of each recipe (the
first of the benchmark's two random shells).

Each structure has no cell vector that matters: it sits in a box with 8 A of vacuum and is
periodic in no direction.  Its info holds `name` and `accessibility`, the mean fraction of the
end atoms' spheres (covalent radius plus a 1.4 A probe) that no other atom covers.  The
accessibility is computed from the geometry; it is not a measurement.

    python examples/coated_particles/make_coated_particles.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from ase import Atoms                                            # noqa: E402
from ase.io import write                                         # noqa: E402

import run_ligand_shells as shells                               # noqa: E402

#: icosahedron shells of the cores: 55 and 147 gold atoms
CORES = (3, 4)


def main(out: str = os.path.join(HERE, "particles.extxyz")) -> None:
    frames = []
    for k in CORES:
        for coverage in shells.COVERAGES:
            for chain in shells.CHAINS:
                for end in shells.ENDS:
                    shell, core, y = shells.coated(k, coverage, chain, end, 0)
                    a = Atoms(["Au"] * len(core) + shell.get_chemical_symbols(),
                              positions=np.vstack([core, shell.positions]))
                    a.center(vacuum=8.0)
                    a.pbc = False
                    a.info["name"] = f"Au{len(core)}-c{int(100 * coverage)}-n{chain}-{end}"
                    a.info["accessibility"] = round(float(y), 6)
                    frames.append(a)
    write(out, frames)
    print(f"{len(frames)} coated particles written to {out}")


if __name__ == "__main__":
    main()
