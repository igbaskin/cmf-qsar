"""
Two series of transition-metal complexes with computed redox potentials, for plan item EC4.

Item EC4 asks for the parts a coordination complex needs before the program can model one: van
der Waals parameters for the metals, a reader that writes metal-ligand bonds as dative bonds, and
the frame that the metal and its first coordination sphere give for free. It then asks for a
measurement on "a tabulated series with a shared frame". Two series are built here, because the
two things the item wants measured need different data.

**The iron series: does the method work on complexes at all?**

    Bhuiyan F. H., Harb H., Vazquez Mayagoitia A. "Redox potential prediction of Fe(II)/Fe(III)
    complexes: a density functional theory and graph neural network approach", Digital Discovery
    5 (2026) 1098. Data and code: https://github.com/argonne-lcf/gnnredox (MIT licence).

2267 mononuclear iron(II) complexes taken from the CSD through the tmQM collection, with a
computed Fe(III)/Fe(II) reduction potential for each. Every complex is iron(II), so the metal
does not vary; what varies is the ligand set, and it varies completely -- 2164 of the 2267
complexes have a ligand set that no other complex in the series shares. This is therefore the
series that tests the machinery and the frameless kernel, and it is the series that shows what a
metal-centred frame is worth when the members share nothing but the metal.

**The nickel series: a congeneric series in a shared frame, with its own quantum control.**

    Burlingame J., Matthews B., Handy S., Volkov A. "Spin-state-dependent coordination changes
    and computed redox potentials in a series of substituted triaqua Ni-bis(pyrazolyl)pyridine
    complexes", Polyhedron, in revision (2026). Data:
    https://doi.org/10.5281/zenodo.23630692 (CC BY 4.0).

25 substituted triaqua nickel bis(pyrazolyl)pyridine complexes, each computed in six charge and
spin states, with the reduction potentials of the Ni(III)/Ni(II) and Ni(II)/Ni(I) couples in two
solvents. All 25 share one ligand skeleton and one coordination sphere -- three nitrogen donors
of the ligand and three water oxygens -- so the first coordination sphere aligns the series and
the co-field can be read. The record also carries the natural population analysis of every state,
partitioned into the metal, the ligand, the substituents and the waters, which is the "quantum
descriptors alone" control the item's gate asks for: it is the same DFT calculation the target
comes from, reduced to a handful of numbers.

What the targets are, and what they are not
-------------------------------------------
Both targets are **computed** potentials, not measured ones. The iron potentials come from a
density-functional calculation of the two oxidation states with a micro-solvated correction; the
nickel potentials come from a thermodynamic cycle with continuum solvation, referenced to the
ferrocenium/ferrocene couple computed with the same protocol. So both series say how well a
representation orders complexes that one calculation produced, in the way the COMPAS study of
polybenzenoid gaps does. Neither says how well CMF predicts a measurement.

The nickel series needs one more statement. Its quantum control and its target come from the same
calculation, so a field that beats the control has beaten the strongest competitor available, and
a field that does not has added nothing to what a few global numbers already say.

Usage
-----
    python examples/make_metal_examples.py --fetch           # download the source files
    python examples/make_metal_examples.py --iron            # build complexes_fe/fe_redox.sdf
    python examples/make_metal_examples.py --nickel          # build complexes_ni/ni_bpp.sdf
    python examples/make_metal_examples.py --summary         # what the files hold

The downloaded files stay in an ``upstream/`` directory of each data set, which is not part of
the repository: the iron data are MIT-licensed and the nickel data CC BY 4.0, so both may be
redistributed, but the built SDF is what the benchmark reads and the raw files are large.
"""
from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import sys
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cmfqsar.complexes import (align_on_first_sphere, build_complex,  # noqa: E402
                               coordination, donor_indices, metal_index, read_xyz)

RDLogger.DisableLog("rdApp.*")

IRON_DIR = os.path.join(HERE, "complexes_fe")
NICKEL_DIR = os.path.join(HERE, "complexes_ni")
IRON_SDF = os.path.join(IRON_DIR, "fe_redox.sdf")
NICKEL_SDF = os.path.join(NICKEL_DIR, "ni_bpp.sdf")

_GNNREDOX = "https://raw.githubusercontent.com/argonne-lcf/gnnredox/main/"
_ZENODO = "https://zenodo.org/records/23630692/files/"

#: the files downloaded for each series: (directory, file name, address)
SOURCES = [
    (IRON_DIR, "desolvated_all_tmcs.xyz", _GNNREDOX + "Data/desolvated_all_tmcs.xyz"),
    (IRON_DIR, "tmqm_redox_data_full_data.csv", _GNNREDOX + "Data/tmqm_redox_data_full_data.csv"),
    (IRON_DIR, "LICENSE", _GNNREDOX + "LICENSE"),
    (NICKEL_DIR, "SI_coordinates_all_150.xyz", _ZENODO + "SI_coordinates_all_150.xyz?download=1"),
    (NICKEL_DIR, "redox_potentials_all_complexes.csv",
     _ZENODO + "redox_potentials_all_complexes.csv?download=1"),
    (NICKEL_DIR, "S2_and_fragment_spin_charge_NPA.csv",
     _ZENODO + "S2_and_fragment_spin_charge_NPA.csv?download=1"),
    (NICKEL_DIR, "README.md", _ZENODO + "README.md?download=1"),
]

#: the state of the nickel complexes whose geometry is used: the resting nickel(II) state, which
#: is the oxidised side of one couple and the reduced side of the other, and the only state that
#: both targets share
NICKEL_STATE = "Ni(II) triplet, pseudo-octahedral"


def upstream(directory: str, name: str) -> str:
    return os.path.join(directory, "upstream", name)


def fetch_upstream(force: bool = False) -> None:
    """Download the source files of both series into their ``upstream/`` directories."""
    for directory, name, url in SOURCES:
        dest = upstream(directory, name)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.exists(dest) and not force:
            print(f"  have    {os.path.relpath(dest, ROOT)}")
            continue
        with urllib.request.urlopen(url, timeout=300) as fh, open(dest, "wb") as out:
            out.write(fh.read())
        print(f"  fetched {os.path.relpath(dest, ROOT)} ({os.path.getsize(dest)} bytes)")


# --------------------------------------------------------------------------- #
# the iron series
# --------------------------------------------------------------------------- #

def _donor_formula(mol: Chem.Mol) -> str:
    """The first coordination sphere as a formula, such as ``N4O2``."""
    counts: Dict[str, int] = {}
    for i in donor_indices(mol):
        s = mol.GetAtomWithIdx(i).GetSymbol()
        counts[s] = counts.get(s, 0) + 1
    return "".join(f"{s}{counts[s]}" for s in sorted(counts))


def build_iron(limit: int = 0) -> None:
    """Build the iron series from the coordinates and the table of potentials."""
    xyz_path = upstream(IRON_DIR, "desolvated_all_tmcs.xyz")
    csv_path = upstream(IRON_DIR, "tmqm_redox_data_full_data.csv")
    for p in (xyz_path, csv_path):
        if not os.path.exists(p):
            raise SystemExit(f"{p} is missing; run with --fetch first")
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = {r["csd_code"]: r for r in csv.DictReader(fh)}
    frames = read_xyz(xyz_path, limit=limit or None)
    print(f"  {len(frames)} blocks of coordinates, {len(rows)} rows of potentials")
    t0 = time.time()
    built, faults = [], {}
    matched = 0
    for frame in frames:
        row = rows.get(frame.name)
        if row is None:
            faults["no potential in the table"] = faults.get("no potential in the table", 0) + 1
            continue
        mol, report = build_complex(
            frame.symbols, frame.coords, total_charge=int(row["q"]),
            ligand_charges=ast.literal_eval(row["ligands_q_list"]),
            ligand_smiles=ast.literal_eval(row["ligands_list"]),
            name=frame.name)
        if mol is None:
            # the size of the ligand is part of the message; the count is per kind of fault
            key = re.sub(r"\d+", "N", report.fault.split(":")[0])
            faults[key] = faults.get(key, 0) + 1
            continue
        # The table states the oxidation state of the metal, and the builder works it out from
        # the charges of the ligands.  Where the two disagree, one ligand charge was placed on
        # the wrong ligand, so the charge inside the complex is wrong even though its total is
        # right.  Such a complex is left out rather than repaired by hand.
        if report.metal_charge != int(row["tm_oxs"]):
            key = "the metal charge does not match the stated oxidation state"
            faults[key] = faults.get(key, 0) + 1
            continue
        matched += report.ligands_matched == report.n_ligands
        c = coordination(mol)
        mol.SetProp("reduction_pot", f"{float(row['reduction_pot']):.6f}")
        mol.SetProp("csd_code", frame.name)
        mol.SetProp("metal", report.metal)
        mol.SetProp("metal_charge", str(report.metal_charge))
        mol.SetProp("n_donors", str(int(c["n_donors"])))
        mol.SetProp("donors", _donor_formula(mol))
        mol.SetProp("d_mean", f"{c['d_mean']:.4f}")
        mol.SetProp("n_atoms", str(mol.GetNumAtoms()))
        mol.SetProp("formula", row["Stoichiometry"])
        mol.SetProp("n_ligands", str(report.n_ligands))
        built.append(mol)
    print(f"  built {len(built)} of {len(frames)} complexes in {time.time() - t0:.0f} s")
    print(f"  every ligand charge confirmed by the ligand's own structure in {matched} of them")
    for key, n in sorted(faults.items(), key=lambda kv: -kv[1]):
        print(f"    not built, {n}: {key}")
    if not built:
        raise SystemExit("nothing was built")
    _report_iron(built)
    os.makedirs(IRON_DIR, exist_ok=True)
    writer = Chem.SDWriter(IRON_SDF)
    for mol in built:
        writer.write(mol)
    writer.close()
    print(f"  wrote {os.path.relpath(IRON_SDF, ROOT)} "
          f"({os.path.getsize(IRON_SDF) / 1e6:.1f} MB)")


def _report_iron(mols: List[Chem.Mol]) -> None:
    y = np.array([float(m.GetProp("reduction_pot")) for m in mols])
    nd = np.array([int(m.GetProp("n_donors")) for m in mols])
    mq = np.array([int(m.GetProp("metal_charge")) for m in mols])
    na = np.array([m.GetNumAtoms() for m in mols])
    donors: Dict[str, int] = {}
    for m in mols:
        key = m.GetProp("donors")
        donors[key] = donors.get(key, 0) + 1
    print(f"  potential: {y.min():.2f} to {y.max():.2f} V, mean {y.mean():.2f}, "
          f"standard deviation {y.std():.2f}")
    print(f"  atoms per complex: {na.min()} to {na.max()}, mean {na.mean():.0f}")
    print(f"  coordination number: {np.bincount(nd).argmax()} most often, "
          f"{nd.min()} to {nd.max()}")
    print(f"  metal charge: " + ", ".join(f"{v:+d} in {int((mq == v).sum())}"
                                          for v in sorted(set(mq.tolist()))))
    top = sorted(donors.items(), key=lambda kv: -kv[1])[:5]
    print("  first coordination sphere: " + ", ".join(f"{k} in {v}" for k, v in top)
          + f" ({len(donors)} distinct)")


# --------------------------------------------------------------------------- #
# the nickel series
# --------------------------------------------------------------------------- #

def _nickel_targets(csv_path: str) -> Dict[str, Dict[str, float]]:
    """The potentials of each complex, keyed by laboratory label then by target name."""
    out: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["branch"] != "fully_relaxed":
                continue
            name = ("E_red_" if r["couple"] == "Ni2_to_Ni1" else "E_ox_") + r["solvent"]
            out.setdefault(r["complex"], {})[name] = float(r["E_vs_Fc_V"])
    return out


def _nickel_quantum(csv_path: str, state: str = "Ni2_triplet") -> Dict[str, Dict[str, float]]:
    """The population analysis of one state of each complex, keyed by laboratory label.

    These are the quantum descriptors the gate compares the fields with: the natural charge on
    the metal, on the ligand, on the substituents and on the water molecules, and the spin
    density on the metal and on the ligand.
    """
    keep = {"NPA_spin_Ni": "npa_spin_Ni", "NPA_spin_BPP(py+pz+subst)": "npa_spin_BPP",
            "NPA_spin_substituents": "npa_spin_subst", "NPA_spin_waters": "npa_spin_waters",
            "NPA_charge_Ni": "npa_charge_Ni", "NPA_charge_BPP(py+pz+subst)": "npa_charge_BPP",
            "NPA_charge_substituents": "npa_charge_subst",
            "NPA_charge_waters": "npa_charge_waters", "S2_after_annih": "s2_after"}
    out: Dict[str, Dict[str, float]] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["state"] != state:
                continue
            out[r["lab"]] = {new: float(r[old]) for old, new in keep.items() if r.get(old)}
            out[r["lab"]]["complex_no"] = float(r["complex"])
    return out


def build_nickel() -> None:
    """Build the nickel series: one geometry per complex, aligned on its first sphere."""
    xyz_path = upstream(NICKEL_DIR, "SI_coordinates_all_150.xyz")
    pot_path = upstream(NICKEL_DIR, "redox_potentials_all_complexes.csv")
    npa_path = upstream(NICKEL_DIR, "S2_and_fragment_spin_charge_NPA.csv")
    for p in (xyz_path, pot_path, npa_path):
        if not os.path.exists(p):
            raise SystemExit(f"{p} is missing; run with --fetch first")
    targets = _nickel_targets(pot_path)
    quantum = _nickel_quantum(npa_path)
    frames = [f for f in read_xyz(xyz_path) if NICKEL_STATE in f.comment]
    print(f"  {len(frames)} blocks in the state {NICKEL_STATE!r}")
    built, labels = [], []
    for frame in frames:
        # "Complex 7 (lab 019)" is the first field of the comment line
        head = frame.parts[0]
        number = head.split()[1]
        lab = head.split("lab")[-1].strip(" )")
        charge = int(frame.parts[2].split()[-1])
        mult = int(frame.parts[3].split()[-1])
        mol, report = build_complex(frame.symbols, frame.coords, total_charge=charge,
                                    name=f"ni_{int(number):02d}", multiplicity=mult)
        if mol is None:
            print(f"    complex {number} (lab {lab}) not built: {report.fault}")
            continue
        mol.SetProp("complex_no", str(int(number)))
        mol.SetProp("lab", lab)
        mol.SetProp("donors", _donor_formula(mol))
        mol.SetProp("metal_charge", str(report.metal_charge))
        for key, val in targets.get(lab, {}).items():
            mol.SetProp(key, f"{val:.6f}")
        for key, val in quantum.get(lab, {}).items():
            if key != "complex_no":
                mol.SetProp(key, f"{val:.6f}")
        built.append(mol)
        labels.append(lab)
    missing = [m.GetProp("_Name") for m in built if not m.HasProp("E_red_AN")]
    if missing:
        print(f"    no potential for {len(missing)}: {', '.join(missing)}")
    aligned, rmsds = align_on_first_sphere(built, ref_index=0)
    good = [r for r in rmsds if r is not None]
    print(f"  aligned on the metal and its donors: "
          f"{np.mean(good):.2f} +/- {np.std(good):.2f} angstrom, worst {max(good):.2f}, "
          f"{len(rmsds) - len(good)} with no match")
    for mol, r in zip(aligned, rmsds):
        if r is not None:
            mol.SetProp("donor_rmsd", f"{r:.4f}")
    _report_nickel(aligned)
    os.makedirs(NICKEL_DIR, exist_ok=True)
    writer = Chem.SDWriter(NICKEL_SDF)
    for mol in aligned:
        writer.write(mol)
    writer.close()
    print(f"  wrote {os.path.relpath(NICKEL_SDF, ROOT)} "
          f"({os.path.getsize(NICKEL_SDF) / 1e3:.0f} kB)")


def _report_nickel(mols: List[Chem.Mol]) -> None:
    for name in ("E_red_AN", "E_ox_AN", "E_red_DMF", "E_ox_DMF"):
        y = np.array([float(m.GetProp(name)) for m in mols if m.HasProp(name)])
        if len(y):
            print(f"  {name}: {y.min():+.2f} to {y.max():+.2f} V, mean {y.mean():+.2f}, "
                  f"standard deviation {y.std():.2f}")
    na = np.array([m.GetNumAtoms() for m in mols])
    print(f"  atoms per complex: {na.min()} to {na.max()}")
    spheres = sorted({m.GetProp("donors") for m in mols})
    print(f"  first coordination sphere: {', '.join(spheres)}")


# --------------------------------------------------------------------------- #

def summarise() -> None:
    for path in (IRON_SDF, NICKEL_SDF):
        if not os.path.exists(path):
            print(f"  {os.path.relpath(path, ROOT)}: not built")
            continue
        mols = [m for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=True)]
        n_bad = sum(1 for m in mols if m is None)
        mols = [m for m in mols if m is not None]
        dative = np.array([sum(1 for b in m.GetBonds()
                               if b.GetBondType() == Chem.BondType.DATIVE) for m in mols])
        print(f"  {os.path.relpath(path, ROOT)}: {len(mols)} complexes "
              f"({os.path.getsize(path) / 1e6:.1f} MB), {n_bad} unreadable, "
              f"{dative.mean():.1f} dative bonds each")
        props = sorted(set(mols[0].GetPropNames()))
        print(f"    properties: {', '.join(props)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--fetch", action="store_true", help="download the source files")
    ap.add_argument("--force", action="store_true", help="download again even if a file is there")
    ap.add_argument("--iron", action="store_true", help="build the iron series")
    ap.add_argument("--nickel", action="store_true", help="build the nickel series")
    ap.add_argument("--summary", action="store_true", help="what the built files hold")
    ap.add_argument("--limit", type=int, default=0, help="only the first N complexes")
    args = ap.parse_args()
    if not any((args.fetch, args.iron, args.nickel, args.summary)):
        ap.error("nothing to do: pass --fetch, --iron, --nickel or --summary")
    if args.fetch:
        print("[fetch]")
        fetch_upstream(force=args.force)
    if args.iron:
        print("[iron]")
        build_iron(limit=args.limit)
    if args.nickel:
        print("[nickel]")
        build_nickel()
    if args.summary:
        print("[summary]")
        summarise()


if __name__ == "__main__":
    main()
