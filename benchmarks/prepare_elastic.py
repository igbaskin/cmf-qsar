#!/usr/bin/env python
"""
The PyMKS elastic label for the microstructure benchmark, computed in a separate environment.

`benchmarks/run_microstructure.py` labels its synthetic two-phase cells with an effective
thermal conductivity from its own finite-volume solver. The published MKS homogenisation
benchmark uses instead the effective elastic stiffness of the same kind of cell, from the
finite-element solver of PyMKS (`pymks.solve_fe`, which drives `sfepy`). This script computes
that label for the cells already in `benchmarks/cache/microstructure/`, so that the gate can
be read on the published property too.

It must run where PyMKS runs. PyMKS 0.4.1, the last release, does not install on a current
Python without help: its build script calls two `configparser` methods removed in Python 3.12,
its import needs `pkg_resources`, `pandas`, `pyarrow` and `deprecated`, which it does not
declare, and its solver calls a `sfepy` method renamed since 2021
(`create_output_dict` -> `create_output`). On the author's machine it runs in a WSL virtual
environment (Python 3.12, sfepy 2026.3) after those three patches; see the notes.

For every cell the solver applies a unit macroscopic strain along x, with periodic boundary
conditions, and returns the stress field; the effective stiffness component C_xx is the mean
stress along x divided by the applied strain. The cell is then transposed and solved again, so
that C_yy is the same quantity along y. Phase 0 has Young's modulus 1, phase 1 the contrast
(10 or 100, matching the conductivity contrast of the set); Poisson's ratio is 0.3 in both.

Usage (inside the PyMKS environment; the paths are read as given):
    python benchmarks/prepare_elastic.py benchmarks/cache/microstructure/2d51_n400_c10.npz
    python benchmarks/prepare_elastic.py <npz> --limit 3 --time     # time a few cells
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("npz", help="a data file of run_microstructure.py (phase, tensor, contrast)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--poisson", type=float, default=0.3)
    ap.add_argument("--time", action="store_true", help="only time the first cells")
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    import tempfile
    npz = os.path.abspath(args.npz)
    # sfepy writes its mesh (domain.vtk) into the working directory; work in a folder of our
    # own, so that two runs do not collide and nothing lands in the repository
    os.chdir(tempfile.mkdtemp(prefix="pymks_elastic_"))
    from pymks import solve_fe
    d = np.load(npz)
    phase = d["phase"]
    contrast = float(d["contrast"])
    n = len(phase) if args.limit is None else min(args.limit, len(phase))
    out = os.path.join(os.path.dirname(npz), "elastic_" + os.path.basename(npz))
    cxx = np.full(n, np.nan)
    cyy = np.full(n, np.nan)
    t0 = time.time()
    batch = 10
    for start in range(0, n, batch):
        cells = phase[start:min(start + batch, n)].astype(int)
        for direction, arr in (("x", cells), ("y", np.transpose(cells, (0, 2, 1)))):
            r = solve_fe(arr, elastic_modulus=(1.0, contrast),
                         poissons_ratio=(args.poisson, args.poisson), macro_strain=1.0)
            stress = np.asarray(r["stress"])            # (b, nx, ny, 3): xx, yy, xy
            c = stress[..., 0].mean(axis=(1, 2))
            if direction == "x":
                cxx[start:start + len(cells)] = c
            else:
                cyy[start:start + len(cells)] = c
        done = min(start + batch, n)
        print(f"  {done} of {n} cells, {time.time() - t0:.0f} s", flush=True)
        if args.time:
            break
    if args.time:
        print(f"{(time.time() - t0) / (2 * min(batch, n)):.2f} s per solve")
        return 0
    np.savez_compressed(out, cxx=cxx, cyy=cyy, contrast=np.array(contrast),
                        poisson=np.array(args.poisson),
                        note=np.array("PyMKS 0.4.1 solve_fe, sfepy, periodic, unit macro strain"))
    print(f"written {out}: C_xx {np.nanmin(cxx):.3f} to {np.nanmax(cxx):.3f}, "
          f"{(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
