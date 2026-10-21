"""Per-atom quantum-chemical properties from ORCA, written as an SDF for the ``prop:`` fields.

A corrosion inhibitor works through the atoms that bind the metal, and the quantities chemists use
to say *which* atoms those are -- condensed Fukui functions, local softness, local
electrophilicity, atomic charges, frontier-orbital populations -- are per-atom numbers that the
program cannot compute itself.  It can, however, build a molecular field from any per-atom
property carried by the input file (``--fields prop:NAME``), so this script closes the gap: it
takes a data set the program reads, runs ORCA on every molecule and writes one SDF in which every
atom carries the quantum-chemical descriptors and every molecule the global ones, together with
everything the input had (name, activity, ...).

What is computed, per molecule
------------------------------
* Geometry: by default the program's own conformer (ETKDG + MMFF from SMILES, or the coordinates
  of the SDF as given), so that the field is built on exactly the geometry the properties were
  computed on, hydrogens included.  ``--opt`` optimises it with ORCA first and the SDF then holds
  the optimised coordinates.
* The N-electron state (charge from the formal charges of the molecule, multiplicity from the
  electron count): Mulliken, Loewdin, Hirshfeld and CHELPG charges, orbital energies, the
  Mulliken population of every atom in every MO (for the HOMO/LUMO populations), energy, dipole.
* Unless ``--no-fukui``: the N+1 and N-1 electron states at the same geometry, which give the
  condensed Fukui functions by finite differences of the atomic charges (Yang & Mortier 1986),

      f+ = q(N) - q(N+1)     (attack by a nucleophile; where the molecule accepts an electron)
      f- = q(N-1) - q(N)     (attack by an electrophile; where it gives one up -- the metal
                              surface sees this side of an inhibitor)
      f0 = (f+ + f-) / 2,    dual descriptor = f+ - f-

  from the Hirshfeld charges (the Mulliken-based set is stored too, under ``*_mulliken``), and the
  global reactivity indices from the three energies: IP = E(N-1) - E(N), EA = E(N) - E(N+1),
  hardness = (IP - EA)/2, softness = 1/(2 hardness), chemical potential mu = -(IP + EA)/2,
  electrophilicity = mu^2/(2 hardness).  Local softness s+/- = softness * f+/-, local
  electrophilicity = electrophilicity * f+.  Energies in eV, softness in 1/eV.

The anion is computed with the same basis as the neutral; a small basis without diffuse functions
can leave the extra electron poorly bound, which is the usual compromise for condensed Fukui
functions and the reason the EA is reported alongside -- a negative one tells you how far to
trust f+.

Running ORCA
------------
Every molecule gets its own directory ``<workdir>/<index>_<name>/`` with the inputs, outputs and a
``status.json`` holding a key made of (canonical SMILES, charge, level, opt, coordinates).  A
rerun skips molecules whose key is present and done, and within a molecule every ORCA job whose
output already says ``ORCA TERMINATED NORMALLY`` is not run again, so a crashed series resumes
where it stopped.  A molecule whose ORCA job fails is reported and left out of the SDF, not fatal.

``--nprocs`` sets ``%pal nprocs`` for ORCA's MPI parallelism, which needs the MPI build of ORCA
(``orca_startup_mpi.exe`` beside ``orca.exe``) and ``mpiexec`` on the PATH.  The build installed
on this desktop is the serial one, so the script falls back to serial runs with a warning and the
way to use the cores is ``--jobs N``, N molecules at a time.

Usage
-----
    python examples/orca_atom_properties.py examples/corrosion/157_c1ccc2_nH_cnc2c1.csv \\
        --limit 3 --jobs 3 -o bzim_qm.sdf
    python cmf_cli.py build bzim_qm.sdf --fields steric,prop:fukui_minus --alphas 0.5 --lams 0.1

``--level`` overrides the level of theory (default B3LYP-D3BJ/def2-SVP with RIJCOSX; ``--fast``
selects the composite r2SCAN-3c, several times faster and good enough for charges).
"""
import argparse
import os
import sys
import time
from typing import Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.orca import (  # noqa: E402
    ATOM_PROPS_FUKUI, ATOM_PROPS_N, Config, DEFAULT_LEVEL, DEFAULT_ORCA, FAST_LEVEL,
    MOL_PROPS_FUKUI, MOL_PROPS_N, find_orca, orca_has_mpi, process_dataset)

# The work itself lives in cmfqsar/orca.py, so that the application can run exactly the same code
# from Tools -> Quantum properties (ORCA). What is left here is the command line.


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="CSV with a smiles column, .smi, or SDF")
    ap.add_argument("-o", "--out", help="output SDF (default: <dataset>_qm.sdf)")
    ap.add_argument("--workdir", help="directory for the ORCA jobs (default: <out stem>_orca)")
    ap.add_argument("--level", default=DEFAULT_LEVEL, help=f"ORCA keywords (default '{DEFAULT_LEVEL}')")
    ap.add_argument("--fast", action="store_true", help=f"use '{FAST_LEVEL}' instead")
    ap.add_argument("--opt", action="store_true", help="optimise the geometry with ORCA first")
    ap.add_argument("--no-fukui", action="store_true", help="skip the N+1 / N-1 states")
    ap.add_argument("--nprocs", type=int, default=4,
                    help="%%pal nprocs per ORCA job (default 4; needs the MPI build of ORCA)")
    ap.add_argument("--jobs", type=int, default=1, help="molecules run concurrently (default 1)")
    ap.add_argument("--maxcore", type=int, default=2000, help="%%maxcore in MB per process (default 2000)")
    ap.add_argument("--limit", type=int, help="only the first N molecules")
    ap.add_argument("--timeout", type=float, help="seconds allowed per ORCA job")
    ap.add_argument("--orca", help=f"ORCA executable (default $ORCA_EXE or {DEFAULT_ORCA})")
    a = ap.parse_args(argv)

    out = a.out or os.path.splitext(a.dataset)[0] + "_qm.sdf"
    cfg = Config(orca=find_orca(a.orca), level=FAST_LEVEL if a.fast else a.level, opt=a.opt,
                 fukui=not a.no_fukui, nprocs=max(1, a.nprocs), maxcore=a.maxcore, timeout=a.timeout,
                 workdir=a.workdir or os.path.splitext(out)[0] + "_orca")
    if cfg.nprocs > 1 and not orca_has_mpi(cfg.orca):
        print(f"warning: {cfg.orca} is the serial ORCA build (no orca_startup_mpi); running with nprocs 1. "
              f"Use --jobs N to run N molecules at a time instead.")
        cfg.nprocs = 1
    t0 = time.time()
    outcomes = process_dataset(a.dataset, out, cfg, limit=a.limit, jobs=max(1, a.jobs))
    print(f"total wall time {time.time() - t0:.0f} s")
    rel = os.path.relpath(out, os.path.join(HERE, ".."))
    print(f"  try:  python cmf_cli.py build {rel} --fields steric,prop:fukui_minus --alphas 0.5 --lams 0.1")
    return 0 if any(oc.mol is not None for oc in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
