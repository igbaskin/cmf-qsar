"""A molecular-dynamics trajectory written as a conformational ensemble in an SDF (item D7).

`cmfqsar/trajectory.py` turns a simulation into the object the field machinery already takes: one
molecule with a few dozen conformations and a weight for each. This script is the command line
in front of it. It reads one trajectory, or one per ligand of a series, and writes an SDF that
`cmf build --ensemble-by name --conformations ensemble --conformer-weighting given` reads.
An SDF record holds one conformation, so the ensemble is written as one record per conformation
under the same title, which is what `--ensemble-by name` collapses again.

    python examples/trajectory_ensemble.py sys.pdb sys.xtc --ligand LIG -o ligand.sdf
    python examples/trajectory_ensemble.py --directory runs/ --template series.sdf -o series.sdf
    cmf build series.sdf --activity pKi --ensemble-by name --conformations ensemble         --conformer-weighting given

With `--directory`, every pair of files in it that share a stem (``abc.pdb`` with ``abc.dcd``,
``abc.xtc`` or ``abc.nc``) is read as one ligand, and the molecule of `--template` whose name
matches that stem gives the bond orders. That is how a series simulated one ligand at a time
becomes one data set.

What the chain does, and why, is in the module's own documentation: the solvent is stripped, the
frames are superimposed on the binding-site backbone when there is a receptor, they are thinned
by clustering, and the weight of a conformation is its cluster population -- never a Boltzmann
factor, because an equilibrated trajectory has already sampled the distribution.

The activity is not in a trajectory. Give it with `--activity-file` (a CSV of name and value, as
`cmf build --activity-file` takes) or add it to the SDF afterwards.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from rdkit import Chem, RDLogger  # noqa: E402

from cmfqsar.trajectory import (DEFAULT_CUTOFF, DEFAULT_MAX_FRAMES, DEFAULT_SITE_RADIUS,
                                TrajectoryError, available_engines, read_trajectory)  # noqa: E402

RDLogger.DisableLog("rdApp.*")

TRAJECTORY_SUFFIXES = (".xtc", ".dcd", ".nc", ".netcdf", ".trr", ".h5", ".dtr", ".trj")
TOPOLOGY_SUFFIXES = (".pdb", ".psf", ".prmtop", ".parm7", ".gro", ".tpr", ".mol2", ".pdbqt")


def pairs_in(directory: str) -> List[tuple]:
    """(stem, topology, trajectory) for every pair of files in `directory` that share a stem."""
    tops: Dict[str, str] = {}
    trjs: Dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(directory, "*"))):
        stem, ext = os.path.splitext(os.path.basename(p))
        if ext.lower() in TOPOLOGY_SUFFIXES:
            tops.setdefault(stem, p)
        elif ext.lower() in TRAJECTORY_SUFFIXES:
            trjs.setdefault(stem, p)
    return [(s, tops[s], trjs.get(s)) for s in sorted(tops) if s in trjs or tops[s]]


def templates_from(path: Optional[str]) -> Dict[str, Chem.Mol]:
    """{name: molecule} from an SDF, for the bond orders and the atom order."""
    if not path:
        return {}
    out = {}
    for m in Chem.SDMolSupplier(path, removeHs=False):
        if m is None:
            continue
        out[m.GetProp("_Name") if m.HasProp("_Name") else f"mol{len(out)}"] = m
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("topology", nargs="?", help="PDB, PSF, PRMTOP, GRO, ... (or use --directory)")
    ap.add_argument("trajectory", nargs="?", help="XTC, DCD, NetCDF, TRR; omit for a multi-model "
                                                  "topology that holds the frames itself")
    ap.add_argument("--directory", help="read every topology/trajectory pair in this directory")
    ap.add_argument("-o", "--out", required=True, help="the SDF to write")
    ap.add_argument("--ligand", help="the residue name of the ligand (default: the largest "
                                     "residue that is neither protein nor solvent)")
    ap.add_argument("--template", help="an SDF of the ligands, which fixes the bond orders; the "
                                       "molecule whose name matches the file stem is used")
    ap.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                    help=f"conformations to keep per ligand (default {DEFAULT_MAX_FRAMES})")
    ap.add_argument("--cutoff", type=float, default=DEFAULT_CUTOFF,
                    help=f"clustering radius in angstrom (default {DEFAULT_CUTOFF})")
    ap.add_argument("--stride", type=int, default=1, help="read every Nth frame")
    ap.add_argument("--site-radius", type=float, default=DEFAULT_SITE_RADIUS,
                    help=f"what counts as the binding site, in angstrom "
                         f"(default {DEFAULT_SITE_RADIUS})")
    ap.add_argument("--extra-solvent", default="",
                    help="further residue names to drop, comma separated")
    ap.add_argument("--engine", default="auto", choices=["auto", "mdtraj", "mdanalysis"])
    ap.add_argument("--activity-file", help="a CSV of name and value, copied onto the molecules")
    a = ap.parse_args(argv)

    if not available_engines():
        print("reading a trajectory needs mdtraj or MDAnalysis; neither is installed")
        return 2
    if not a.directory and not a.topology:
        print("give a topology (and a trajectory), or --directory")
        return 2

    jobs = (pairs_in(a.directory) if a.directory
            else [(os.path.splitext(os.path.basename(a.topology))[0], a.topology, a.trajectory)])
    if not jobs:
        print(f"no topology/trajectory pair in {a.directory}")
        return 1
    templates = templates_from(a.template)
    extra = [s for s in (a.extra_solvent or "").split(",") if s]

    activities = {}
    if a.activity_file:
        import csv
        with open(a.activity_file, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 2:
                    try:
                        activities[row[0].strip()] = float(row[1])
                    except ValueError:
                        pass                       # the header line

    mols = []
    for stem, top, trj in jobs:
        try:
            e = read_trajectory(top, trj, ligand=a.ligand, template=templates.get(stem),
                                max_frames=a.max_frames, cutoff=a.cutoff, stride=a.stride,
                                site_radius=a.site_radius, extra_solvent=extra,
                                engine=a.engine, name=stem)
        except TrajectoryError as exc:
            print(f"{stem}: skipped -- {exc}")
            continue
        if stem in activities:
            e.mol.SetProp("activity", f"{activities[stem]:.6g}")
        print(f"{stem}: {e.describe()}")
        mols.append(e.mol)
    if not mols:
        print("nothing could be read")
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    # an SDF record holds one conformation, so an ensemble is written as one record per
    # conformation, all under the same title; `--ensemble-by name` collapses them again on
    # loading, and the weights travel as a property every record carries
    records = 0
    with Chem.SDWriter(a.out) as w:
        for m in mols:
            for conf in m.GetConformers():
                w.write(m, confId=conf.GetId())
                records += 1
    print(f"{len(mols)} ligands in {records} records written to {a.out}")
    print(f"  try:  cmf build {a.out} --activity activity --ensemble-by name "
          f"--conformations ensemble --conformer-weighting given")
    return 0


if __name__ == "__main__":
    sys.exit(main())
