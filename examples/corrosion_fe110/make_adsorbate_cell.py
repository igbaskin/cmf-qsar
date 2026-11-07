#!/usr/bin/env python
"""One file holding both the Fe(110) slab and the molecules adsorbed on it.

``157_benzimidazole_uma.sdf`` holds the relaxed poses of the inhibitors **alone**, in the surface
frame: the normal along z, the top atomic layer at z = 0.  That is what a molecular data set
needs, and the slab is left out of it because every molecule sits on the same one.  The window
for crystals needs the other thing: one structure holding the slab and the molecule together,
periodic in the two directions of the surface, which is what a relaxation of an adsorbed state
writes and what a user of a surface study already has as a ``CONTCAR`` or a CIF.  This script
builds that file from the two halves this repository does ship, so that the surface side of the
application can be opened without running a relaxation first.

What it writes
--------------
An extended-XYZ file with one frame per molecule.  Each frame is the 6 x 6 x 4 Fe(110) slab that
the poses were relaxed on (``cmfqsar.slab.bcc110_slab``: a = 2.78 angstrom, the minimum of the
UMA oc20 head for body-centred cubic iron, with its top layer at z = 0) followed by the atoms of
one molecule at its relaxed pose.  The cell is the slab's, periodic in the two surface
directions and open along the normal, and the frame carries the name of the molecule and its
adsorption energy as info fields, so that

    cmf build-crystals benzimidazole_on_fe110.extxyz --pbc ab --target E_ads_kJmol

fits a periodic model on it and

    cmf-app   (File -> Crystals and other periodic structures -> Open one structure file)

opens it.  With *Periodic in* set to ``ab: a layer`` and *Atoms* set to ``ball and stick`` the
molecule comes out with its rings while the iron stays spheres, because a metal atom gets no
perceived bonds (see the manual, *How the atoms are drawn, and the bonds*).

Where the atoms are put
-----------------------
Along the normal the whole structure is shifted so that it sits inside its cell, with the vacuum
above and below it, which is what a relaxation of an adsorbed state writes and what the viewer's
cell box expects.  The ``z = 0`` of the surface frame is a convention for the *poses* of the
molecules alone, in ``157_benzimidazole_uma.sdf``; a periodic cell wants its atoms in its box.

In the plane the relaxed poses are given about x = y = 0, because the slab they were relaxed on
was centred there, while ``bcc110_slab`` builds its cell from the origin outwards.
``--place centre``, the default, moves each molecule by half the two surface vectors, which is a
translation by half a cell and changes no distance, and leaves it in the middle of the picture.
``--place corner`` keeps the pose where the SDF has it and then wraps the cell, so the molecule
comes back into the box in two pieces: that is the structure to open to see a bond drawn across
the cell boundary, with the image atoms it ends on drawn beside it.  Wrapping is what makes the
difference -- a molecule that merely sits outside the box is still one piece and crosses nothing.

Usage
-----
    python examples/corrosion_fe110/make_adsorbate_cell.py
    python examples/corrosion_fe110/make_adsorbate_cell.py --count 20 --place corner \
        --out corner.extxyz

Needs ASE and RDKit.  The output is deterministic: the same inputs give the same file, byte for
byte, so the file in the repository can be rebuilt and compared.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

#: the poses, and the properties of each that are worth carrying into the frame
SDF = os.path.join(HERE, "157_benzimidazole_uma.sdf")
CARRY = ("E_ads_eV", "E_ads_kJmol", "pose_label", "height_A", "smiles")
#: how many molecules the shipped file holds: enough to be a set in the Structures panel, few
#: enough that the file stays small, since every frame repeats the 144 atoms of the slab
COUNT = 8
#: the slab of the reference poses
SLAB = dict(element="Fe", nx=6, ny=6, layers=4, a=2.78, vacuum=15.0)
DEFAULT_OUT = os.path.join(HERE, "benzimidazole_on_fe110.extxyz")


def build(count: int = COUNT, place: str = "centre", sdf: str = SDF):
    """The frames, as a list of ASE structures: the slab with one molecule each."""
    from ase import Atoms
    from rdkit import Chem, RDLogger

    from cmfqsar.slab import bcc110_slab
    RDLogger.DisableLog("rdApp.*")

    slab = bcc110_slab(**SLAB)
    cell = np.asarray(slab.get_cell(), float)
    shift = 0.5 * (cell[0] + cell[1]) if place == "centre" else np.zeros(3)
    slab_symbols = [a.symbol for a in slab]
    frames = []
    for mol in Chem.SDMolSupplier(sdf, removeHs=False, sanitize=True):
        if mol is None or len(frames) >= int(count):
            continue
        if mol.HasProp("gas_intact") and mol.GetProp("gas_intact") != "1":
            continue                      # its adsorption energy is an artefact
        xyz = np.array(mol.GetConformer().GetPositions(), float) + shift
        symbols = [a.GetSymbol() for a in mol.GetAtoms()]
        positions = np.vstack([np.asarray(slab.positions, float), xyz])
        # along the normal: the whole structure inside its cell, vacuum above and below
        positions[:, 2] += 0.5 * (float(cell[2, 2]) - np.ptp(positions[:, 2])) \
            - positions[:, 2].min()
        frame = Atoms(symbols=slab_symbols + symbols, positions=positions,
                      cell=cell, pbc=(True, True, False))
        if place == "corner":
            # the pose reaches outside the box; wrapping brings it back in two pieces, which is
            # what puts a bond across the cell boundary
            frame.wrap()
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"molecule_{len(frames) + 1}"
        frame.info["name"] = name
        frame.info["slab_atoms"] = int(len(slab))
        frame.info["slab"] = f"{SLAB['element']}(110) {SLAB['nx']}x{SLAB['ny']}x{SLAB['layers']}"
        frame.info["a_slab"] = float(SLAB["a"])
        frame.info["place"] = str(place)
        for key in CARRY:
            if mol.HasProp(key):
                frame.info[key] = mol.GetProp(key)
        frames.append(frame)
    return frames


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--count", type=int, default=COUNT,
                    help=f"molecules written, in the order of the SDF (default {COUNT})")
    ap.add_argument("--place", choices=("centre", "corner"), default="centre",
                    help="where the molecule sits in the cell: in the middle of the picture "
                         "(default), or where the SDF has it, which straddles the boundary")
    ap.add_argument("--out", default=DEFAULT_OUT, help="the extended-XYZ file to write")
    ap.add_argument("--sdf", default=SDF, help="the SDF of relaxed poses to read")
    args = ap.parse_args(argv)

    from ase.io import write
    frames = build(count=args.count, place=args.place, sdf=args.sdf)
    if not frames:
        raise SystemExit(f"no molecule with a usable pose in {args.sdf}")
    # newline="\n" so that the file is the same on every platform and its hash is one number
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        write(fh, frames, format="extxyz")
    atoms = [len(f) for f in frames]
    print(f"{len(frames)} frames of {min(atoms)} to {max(atoms)} atoms written to {args.out}")
    print(f"  the slab is {frames[0].info['slab_atoms']} atoms of "
          f"{frames[0].info['slab']}, the molecule is placed at the {args.place}")
    print("  open it with: cmf-app, File -> Crystals and other periodic structures -> "
          "Open one structure file, Periodic in 'ab: a layer', Atoms 'ball and stick'")


if __name__ == "__main__":
    main()
