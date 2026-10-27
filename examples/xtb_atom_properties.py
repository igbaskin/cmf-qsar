"""Per-atom properties from GFN2-xTB, written as an SDF for the ``prop:`` fields (item D3).

The same job as ``examples/orca_atom_properties.py`` and about a thousand times faster. ORCA
solves the Kohn-Sham equations and needs two to three minutes for a molecule of forty atoms;
GFN2-xTB solves a parameterised tight-binding problem and needs a fraction of a second for the
same molecule, including the two ionic states a Fukui index is built from. That is what makes a
quantum stage over a whole collection of series affordable rather than a single-series exercise.

What is computed, per molecule
------------------------------
* Geometry: the one in the file, unchanged, so the field is built on the geometry the properties
  were computed on. ``--opt`` relaxes it with xtb instead, and the SDF then holds the relaxed
  coordinates -- which breaks a superposition, so use it only on a set that is not aligned.
* Per atom: the GFN2 partial charge ``q_xtb``, the atomic polarisability ``alpha_xtb``, and,
  unless ``--no-fukui``, the condensed Fukui functions

      f+ = q(N) - q(N+1)     (where the molecule accepts an electron)
      f- = q(N-1) - q(N)     (where it gives one up)
      f0 = (f+ + f-) / 2,    dual descriptor = f+ - f-

  under the names ``fukui_plus``, ``fukui_minus``, ``fukui_zero`` and ``dual``, which are the
  names ``examples/orca_atom_properties.py`` writes for the same quantities.
* Per molecule: the total energy, the frontier orbital energies, their gap and the dipole.

The charges are not the ORCA ones. ``q_xtb`` is a Mulliken-like tight-binding population and
``q_hirshfeld`` is a Hirshfeld charge, so the two are different quantities with different names;
a model fitted on one must be predicted with the same one. The Fukui indices are defined by the
same differences in both, which ``tests/test_xtb.py`` checks.

Usage
-----
    python examples/xtb_atom_properties.py examples/corrosion/set01.sdf
    python examples/xtb_atom_properties.py set.sdf -o set_xtb.sdf --jobs 8
    python examples/xtb_atom_properties.py set.sdf --solvent water --no-fukui

xtb itself is not part of this repository. The Windows and Linux builds are released at
https://github.com/grimme-lab/xtb/releases; give the path with ``--xtb``, or set ``XTB_EXE``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.xtb import DEFAULT_XTB, Config, XtbError, find_xtb, process_dataset, xtb_version


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="CSV with a smiles column, .smi, or SDF")
    ap.add_argument("-o", "--out", help="output SDF (default: <dataset>_xtb.sdf)")
    ap.add_argument("--workdir", help="directory for the xtb jobs (default: <out stem>_xtb)")
    ap.add_argument("--gfn", type=int, default=2, choices=[0, 1, 2],
                    help="the GFN level (default 2, which is GFN2-xTB)")
    ap.add_argument("--opt", action="store_true",
                    help="relax the geometry with xtb first; this breaks a superposition")
    ap.add_argument("--no-fukui", action="store_true",
                    help="skip the two ionic states, so no Fukui functions")
    ap.add_argument("--solvent", default="",
                    help="an ALPB solvent name, e.g. water; the default is the gas phase")
    ap.add_argument("--accuracy", type=float, default=1.0,
                    help="xtb's --acc; smaller is tighter (default 1.0)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="molecules run at a time (default 1; xtb uses the cores of one anyway)")
    ap.add_argument("--limit", type=int, help="only the first N molecules")
    ap.add_argument("--timeout", type=float, default=600.0,
                    help="seconds allowed for one molecule (default 600)")
    ap.add_argument("--xtb", help=f"the xtb executable (default $XTB_EXE or {DEFAULT_XTB})")
    a = ap.parse_args(argv)

    out = a.out or os.path.splitext(a.dataset)[0] + "_xtb.sdf"
    try:
        exe = find_xtb(a.xtb)
    except XtbError as exc:
        print(exc)
        return 2
    print(f"{exe}: {xtb_version(exe)}")
    cfg = Config(xtb=exe, gfn=a.gfn, opt=a.opt, fukui=not a.no_fukui, solvent=a.solvent,
                 accuracy=a.accuracy, timeout=a.timeout,
                 workdir=a.workdir or os.path.splitext(out)[0] + "_xtb")
    t0 = time.time()
    outcomes = process_dataset(a.dataset, out, cfg, limit=a.limit, jobs=max(1, a.jobs))
    print(f"total wall time {time.time() - t0:.0f} s")
    rel = os.path.relpath(out, os.path.join(HERE, ".."))
    print(f"  try:  python cmf_cli.py build {rel} --fields steric,prop:fukui_minus "
          f"--alphas 0.5 --lams 0.1")
    return 0 if any(oc.mol is not None for oc in outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
