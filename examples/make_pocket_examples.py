"""Build ``examples/pockets/<collection>/<target>.pdb``: the binding site of every benchmark set.

The free-energy collections give their ligands in the coordinate frame of a reference crystal
structure and name that structure in every record, so a pocket trimmed from the entry drops
straight onto the molecules with no superposition.  Storing the trimmed site rather than the whole
entry keeps the files small, tens of kilobytes instead of a megabyte, and lets the application show
the protein without network access.

What is written: every residue with an atom within ``--cutoff`` of any ligand of the set, taken
whole, with crystallographic waters removed and the co-crystallised ligand removed when it occupies
the same space as the data set.  Ions and cofactors lining the site are kept.  Coordinates are
untouched, so the file is in the frame of the ligands.

Structures come from RCSB and are in the public domain (CC0); the provenance of each file is
recorded in its REMARK 400 lines.

    python examples/make_pocket_examples.py                 # openff and schindler
    python examples/make_pocket_examples.py --all           # and the Schrodinger sets too
    python examples/make_pocket_examples.py --cutoff 10
"""
import argparse
import datetime
import glob
import os
import sys

import numpy as np
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cmfqsar.protein import (binding_site, contact_distance, fetch_pdb,  # noqa: E402
                             read_structure, write_pdb)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_COLLECTIONS = ["openff", "schindler"]
ALL_COLLECTIONS = ["openff", "schindler", "ross"]
CODE_PROPS = ("pdb", "PDB", "pdb_id", "pdbid")


def pdb_code(path: str):
    """The PDB identifier recorded in the first record of an SD file, if any."""
    for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=False):
        if m is None:
            continue
        for prop in CODE_PROPS:
            if m.HasProp(prop):
                code = m.GetProp(prop).strip()
                if code:
                    return code
        return None
    return None


def ligand_points(path: str):
    pts = []
    for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=False):
        if m is None or m.GetNumConformers() == 0:
            continue
        pts.append(m.GetConformer().GetPositions())
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collections", default=None,
                    help="comma-separated folders of examples/ to process")
    ap.add_argument("--all", action="store_true", help="include the Schrodinger (ross) sets")
    ap.add_argument("--cutoff", type=float, default=8.0,
                    help="residues with an atom within this distance of a ligand (default 8 A)")
    ap.add_argument("--max-contact", type=float, default=5.0,
                    help="skip a set whose ligands are further than this from the structure, "
                         "which means the two are not in the same frame; correctly framed poses "
                         "touch the protein within about 2 A (default 5 A)")
    ap.add_argument("--out", default=os.path.join(HERE, "pockets"))
    ap.add_argument("--cache", default=None, help="where downloaded entries are kept")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    cols = (args.collections.split(",") if args.collections
            else (ALL_COLLECTIONS if args.all else DEFAULT_COLLECTIONS))
    today = datetime.date.today().isoformat()
    made, skipped, failed = 0, 0, 0
    for col in cols:
        root = os.path.join(HERE, col)
        if not os.path.isdir(root):
            print(f"{col}: no such folder, skipped")
            continue
        for sdf in sorted(glob.glob(os.path.join(root, "**", "*.sdf"), recursive=True)):
            rel = os.path.relpath(sdf, HERE)
            stem = os.path.splitext(os.path.basename(sdf))[0]
            dest_dir = os.path.join(args.out, os.path.relpath(os.path.dirname(sdf), HERE))
            dest = os.path.join(dest_dir, stem + ".pdb")
            if os.path.isfile(dest) and not args.overwrite:
                skipped += 1
                continue
            code = pdb_code(sdf)
            if not code:
                print(f"{rel}: no PDB identifier in the file, skipped")
                skipped += 1
                continue
            try:
                entry = fetch_pdb(code, dest_dir=args.cache)
                prot = read_structure(entry, keep_waters=True)
                pts = ligand_points(sdf)
                d = contact_distance(prot, pts)
                if d > args.max_contact:
                    print(f"{rel}: {code} is {d:.1f} A from the ligands, not the same frame; skipped")
                    skipped += 1
                    continue
                site = binding_site(prot, pts, cutoff=args.cutoff, keep_waters=False)
                if site.n_atoms == 0:
                    print(f"{rel}: {code} gave an empty site, skipped")
                    skipped += 1
                    continue
                os.makedirs(dest_dir, exist_ok=True)
                write_pdb(site, dest, remarks=[
                    f"CMF 3D-QSAR example pocket for {rel}",
                    f"Source entry {code.upper()} from RCSB (wwPDB), public domain (CC0).",
                    f"Residues with an atom within {args.cutoff:g} A of any ligand of the set, taken",
                    "whole; crystallographic waters and the co-crystallised ligand removed, ions and",
                    "cofactors kept. Coordinates are those of the entry, i.e. the frame the ligands",
                    f"are given in; closest ligand approach {d:.2f} A.",
                    f"Written by examples/make_pocket_examples.py on {today}.",
                ])
                kb = os.path.getsize(dest) / 1024
                print(f"{rel}: {code} -> {os.path.relpath(dest, HERE)} "
                      f"({site.n_residues} residues, {site.n_atoms} atoms, {kb:.0f} kB, "
                      f"contact {d:.2f} A)")
                made += 1
            except Exception as exc:                       # noqa: BLE001 - reported per set
                print(f"{rel}: {code}: {exc}")
                failed += 1
    print(f"\n{made} pockets written, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
