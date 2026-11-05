"""
Phenazine derivatives with their one-electron reduction potentials, for plan item EC1.

The geometries and the potentials come from the data repository of

    Ghule S., Dash S. R., Bagchi S., Joshi K., Vanka K. "Predicting the Redox Potentials of
    Phenazine Derivatives Using DFT-Assisted Machine Learning", ACS Omega 7 (2022) 11742-11755.
    https://doi.org/10.1021/acsomega.1c06856
    Data at https://github.com/siddharth-ncl-work/ml_redox_potential-DATA

Why this series is here
-----------------------
Item EC1 adds the oxidation state, the spin state and a continuum solvent to the quantum stage,
and asks whether the fields those give -- Hirshfeld charges, Fukui functions and the spin
population of the radical -- predict a redox potential better than the five physicochemical
fields, and better than the global ionisation energy and electron affinity alone. A phenazine
series is the case the item names, and it is the right case for three separate reasons.

* **The frame is free.** Every member shares the phenazine core, so a common frame needs no
  argument, which is the one hard requirement of the method. The Hammett study is the precedent:
  a congeneric series in a shared frame is where a field model is expected to work.
* **The reduced form is a radical anion.** Its unpaired electron is what the new spin field
  measures, and a reduction potential is exactly the property that electron decides. Without item
  EC1 the spin populations were computed and thrown away.
* **The solvent matters and is known.** The potentials were computed in 1,2-dimethoxyethane, so
  the descriptors are computed in the same solvent through the new ``solvent`` setting rather
  than in the gas phase.

What the target is, and what it is not
--------------------------------------
``E0_V`` is a **computed** one-electron reduction potential in volts, not a measured one. It is
the free-energy difference between the neutral molecule and its anion in a continuum solvent,
placed on the volt scale by the measured potential of phenazine itself (-1.74 V), which the source
uses as its anchor. So the series says how well a representation orders these molecules, in the
way the COMPAS study of polybenzenoid gaps does, and it does not say how well CMF predicts a
measurement. Item EC1's gate is a comparison between descriptor sets on one target, which a
computed target answers; a claim about absolute accuracy would need measured potentials, and none
was found in machine-readable form.

One consequence has to be stated plainly. The target comes from a calculation of the anion, and
the spin field also comes from a calculation of the anion. The two are not the same quantity -- the
target is a global free energy in a solvent and the field is a local spin density -- but they are
not independent either, and a gain of the spin field over the five fields has to be read with
that in mind. The global electron affinity is in the benchmark as the control that makes this
readable: it is the nearest global quantity to the target, so a field that cannot beat it has
added nothing.

What the source files need before they can be read
--------------------------------------------------
The geometry files were written by GaussView and have three faults, each of which makes RDKit
refuse the file or the molecule. All three are repaired here, and :func:`read_geometry` reports
which repairs it made so that nothing is silently changed:

1. **The counts line carries no version string.** Where a molfile must end that line with
   ``V2000``, these files have a bare ``0``. The line is rewritten from the atom and bond counts
   it does contain.
2. **There is no ``M  END``.** The file simply stops after the last bond, and the reader then
   reads past the end. The terminator is appended.
3. **Two sulfonic acids have the double bond on the wrong oxygen.** The file gives the hydroxyl
   oxygen a double bond to sulfur and a single bond to its hydrogen, which is a valence of three.
   The geometry settles which oxygen carries the hydrogen (the O-H distance is 0.96 angstrom), so
   the hydroxyl bond is made single and the double bond moves to a terminal oxygen of the same
   centre. The result is a clean ``S(=O)(=O)O``.

Phenazine itself is the first row of all three tables and is the anchor rather than a data point,
and the repository holds no geometry for it, so it is not in the series. What is left is 70
derivatives.

Status under ``docs/public_3dqsar_datasets.md``
-----------------------------------------------
The distributor states no licence, so this is **unclear**: ``upstream/`` is fetched rather than
redistributed and is git-ignored, its digest is recorded here, and what is kept in the repository
is the SDF this script derives.

Usage
-----
    python examples/make_phenazine_examples.py --fetch      # into examples/phenazine/upstream/
    python examples/make_phenazine_examples.py --select     # phenazine.sdf, aligned, with targets
    python examples/make_phenazine_examples.py --quantum --jobs 4
    python examples/make_phenazine_examples.py --summary
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger                                              # noqa: E402

from cmfqsar import orca as O                                                 # noqa: E402
from cmfqsar.align import align_dataset                                       # noqa: E402

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "phenazine")
UPSTREAM = os.path.join(OUT, "upstream")
RAW = "https://raw.githubusercontent.com/siddharth-ncl-work/ml_redox_potential-DATA/master"
API = "https://api.github.com/repos/siddharth-ncl-work/ml_redox_potential-DATA"

#: the three tables of the source and the folder of geometries each one names
TABLES = (
    ("external_test_set", "dft_calculations/external_test_set.csv",
     "molecules/external_test_set"),
    ("two_groups", "dft_calculations/multiple_functional_groups_dataset/two_functional_groups.csv",
     "molecules/multiple_functional_groups_datset/two_functional_groups"),
    ("three_groups",
     "dft_calculations/multiple_functional_groups_dataset/three_functional_groups.csv",
     "molecules/multiple_functional_groups_datset/three_functional_groups"),
)
#: the anchor row of every table: a reference point, not a data point, and it has no geometry file
ANCHOR = "PHENAZINE"
#: SHA-256 over the 73 source files in path order, so that the set has one identity.  A fetch
#: that gives a different digest means the source changed, which is reported rather than ignored.
SOURCE_DIGEST = "b0f6caf38cea0c09e553340da7047395cc4bd2a7889c6e9a051efa8ef76a625f"
#: the solvent the source computed its potentials in
SOLVENT = "DME"
TARGET = "E0_V"


class PhenazineError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------- fetching

def _get(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": "cmfqsar-make-phenazine"})
    with urllib.request.urlopen(req, timeout=120) as fh:
        data = fh.read()
    return data if binary else data.decode("utf-8", "replace")


def wanted_files() -> list:
    """Every source file this script reads: the three tables and the geometries they name."""
    out = [path for _, path, _ in TABLES]
    tree = json.loads(_get(f"{API}/git/trees/master?recursive=1"))
    folders = tuple(folder for _, _, folder in TABLES)
    for entry in tree.get("tree", []):
        p = entry.get("path", "")
        if entry.get("type") == "blob" and p.endswith(".mol") and p.startswith(folders):
            out.append(p)
    return sorted(out)


def fetch_upstream(force: bool = False) -> str:
    """Download the source files into ``upstream/`` and return the digest of the whole set."""
    os.makedirs(UPSTREAM, exist_ok=True)
    paths = wanted_files()
    print(f"  {len(paths)} source files")
    for i, path in enumerate(paths, 1):
        dest = os.path.join(UPSTREAM, path.replace("/", os.sep))
        if os.path.isfile(dest) and not force:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as fh:
            fh.write(_get(f"{RAW}/{path}", binary=True))
        if i % 25 == 0:
            print(f"    {i}/{len(paths)}", flush=True)
    digest = upstream_digest(paths)
    if digest == SOURCE_DIGEST:
        print(f"  digest of the set: {digest} (as recorded)")
    else:
        print(f"  digest of the set: {digest}")
        print(f"  WARNING: the recorded digest is {SOURCE_DIGEST}.")
        print("  The source has changed since this script was written, so the numbers in the "
              "manual were produced from other files than these.")
    return digest


def upstream_digest(paths=None) -> str:
    """One SHA-256 over the source files, in path order, so the set has a single identity."""
    paths = paths or sorted(_local_sources())
    h = hashlib.sha256()
    for path in paths:
        dest = os.path.join(UPSTREAM, path.replace("/", os.sep))
        h.update(path.encode())
        with open(dest, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def _local_sources() -> list:
    out = []
    for base, _, files in os.walk(UPSTREAM):
        for f in files:
            rel = os.path.relpath(os.path.join(base, f), UPSTREAM)
            out.append(rel.replace(os.sep, "/"))
    return out


# ------------------------------------------------------------------- reading a source geometry

def read_geometry(path: str):
    """(molecule, repairs) for one source geometry file, with the three faults above repaired."""
    lines = io.open(path, encoding="utf-8", errors="replace").read().splitlines()
    repairs = []
    if len(lines) > 3 and "V2000" not in lines[3] and "V3000" not in lines[3]:
        fields = lines[3].split()
        if len(fields) < 2:
            raise PhenazineError(f"{os.path.basename(path)}: no counts line")
        lines[3] = "%3d%3d  0  0  0  0  0  0  0  0999 V2000" % (int(fields[0]), int(fields[1]))
        repairs.append("counts line had no version string")
    if not any(ln.strip() == "M  END" for ln in lines):
        lines.append("M  END")
        repairs.append("no M END terminator")
    mol = Chem.MolFromMolBlock("\n".join(lines) + "\n", removeHs=False, sanitize=False)
    if mol is None:
        raise PhenazineError(f"{os.path.basename(path)}: unreadable even after the repairs")
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "O":
            continue
        if not any(n.GetSymbol() == "H" for n in atom.GetNeighbors()):
            continue
        heavy = [b for b in atom.GetBonds() if b.GetOtherAtom(atom).GetAtomicNum() > 1]
        if len(heavy) != 1 or heavy[0].GetBondType() != Chem.BondType.DOUBLE:
            continue
        centre = heavy[0].GetOtherAtom(atom)
        heavy[0].SetBondType(Chem.BondType.SINGLE)
        for bond in centre.GetBonds():
            other = bond.GetOtherAtom(centre)
            if (other.GetSymbol() == "O" and other.GetDegree() == 1
                    and bond.GetBondType() == Chem.BondType.SINGLE):
                bond.SetBondType(Chem.BondType.DOUBLE)
                repairs.append(f"the double bond moved from the hydroxyl oxygen of "
                               f"{centre.GetSymbol()}{centre.GetIdx()} to a terminal one")
                break
        else:
            repairs.append("a hydroxyl bond was made single")
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:                                      # noqa: BLE001
        raise PhenazineError(f"{os.path.basename(path)}: {type(exc).__name__}: {exc}")
    if mol.GetNumConformers() == 0:
        raise PhenazineError(f"{os.path.basename(path)}: no geometry")
    # A phenazine is flat, so most of these geometries are planar to within a thousandth of an
    # angstrom and RDKit marks them 2D. Planar is the right answer here, not a missing geometry,
    # so the test is that the atoms are somewhere rather than that they leave a plane.
    conf = mol.GetConformer()
    xyz = conf.GetPositions()
    if float(np.ptp(xyz, axis=0).max()) < 0.1:
        raise PhenazineError(f"{os.path.basename(path)}: every atom is at the same place")
    if not conf.Is3D():
        conf.Set3D(True)
        repairs.append("the geometry is planar and was marked 2D")
    return mol, repairs


def read_table(path: str) -> list:
    """(name, E0 in volts) of every row of one source table, the anchor row excluded."""
    rows = []
    with io.open(path, encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            name = (row.get("molecule") or "").strip()
            if not name or name.upper() == ANCHOR:
                continue
            try:
                rows.append((name, float(row["E0(V)"])))
            except (KeyError, TypeError, ValueError):
                print(f"    {name}: no usable E0(V), dropped")
    return rows


# ---------------------------------------------------------------------------------- the series

def build_series() -> list:
    """Every molecule of the series: the geometry, the target and which table it came from."""
    out, all_repairs = [], {}
    for subset, table, folder in TABLES:
        table_path = os.path.join(UPSTREAM, table.replace("/", os.sep))
        if not os.path.isfile(table_path):
            raise PhenazineError(f"{table} is not in upstream/; run --fetch first")
        rows = read_table(table_path)
        kept = 0
        for name, e0 in rows:
            geom = os.path.join(UPSTREAM, folder.replace("/", os.sep), f"{name}.mol")
            if not os.path.isfile(geom):
                print(f"    {subset}/{name}: no geometry file, dropped")
                continue
            try:
                mol, repairs = read_geometry(geom)
            except PhenazineError as exc:
                print(f"    {exc}")
                continue
            ident = f"{subset}_{name}"
            mol.SetProp("_Name", ident)
            mol.SetProp("id", ident)
            mol.SetProp("subset", subset)
            mol.SetProp("source_name", name)
            mol.SetProp(TARGET, f"{e0:.6f}")
            # The neutral molecule is closed-shell; item EC1 reads these two from the record
            # rather than guessing, and the anion it derives from them is the doublet.
            mol.SetProp("charge", "0")
            mol.SetProp("multiplicity", "1")
            out.append(mol)
            kept += 1
            if repairs:
                all_repairs[ident] = repairs
        print(f"  {subset}: {kept} of {len(rows)} rows kept")
    if all_repairs:
        print(f"  {len(all_repairs)} geometries needed a repair:")
        counts = {}
        for repairs in all_repairs.values():
            for r in repairs:
                key = r.split(" of ")[0]
                counts[key] = counts.get(key, 0) + 1
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"    {n}x {key}")
    return out


def align(mols: list) -> list:
    """Put every member in one frame on the phenazine core it shares with the smallest member.

    The smallest member is the reference because it is the one whose common substructure with
    every other member is the whole core and nothing else.  `align_dataset` moves the molecules
    in place and returns a line per molecule, which is read here for the two things that can go
    wrong: a match that covers less than the phenazine core, and a match that fails outright.
    """
    ref = min(range(len(mols)), key=lambda i: mols[i].GetNumAtoms())
    print(f"  aligning on {mols[ref].GetProp('id')} ({mols[ref].GetNumAtoms()} atoms) "
          f"by maximum common substructure")
    log = align_dataset(mols, ref_index=ref, use_mcs=True)
    matched, rmsds, failed = [], [], []
    for mol, line in zip(mols, log):
        if "FAILED" in line:
            failed.append((mol.GetProp("id"), line))
            continue
        if line.startswith("MCS "):
            parts = line.split()
            matched.append(int(parts[1]))
            rmsds.append(float(parts[-1]))
    if matched:
        m, r = np.array(matched), np.array(rmsds)
        print(f"  matched {m.mean():.1f} +/- {m.std():.1f} atoms ({m.min()} to {m.max()}), "
              f"RMSD on the match {r.mean():.2f} +/- {r.std():.2f} A")
    if failed:
        print(f"  {len(failed)} alignments failed:")
        for ident, line in failed:
            print(f"    {ident}: {line}")
    return mols


def write_series(mols: list) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "phenazine.sdf")
    writer = Chem.SDWriter(path)
    for mol in mols:
        writer.write(mol)
    writer.close()
    y = np.array([float(m.GetProp(TARGET)) for m in mols])
    print(f"  wrote {path}: {len(mols)} molecules, "
          f"{TARGET} from {y.min():.3f} to {y.max():.3f} V "
          f"(predicting the mean costs {np.mean(np.abs(y - y.mean())):.3f} V)")
    return path


# --------------------------------------------------------------------------- the quantum stage

def quantum(limit: int = 0, jobs: int = 4, maxcore: int = 1500) -> str:
    """Run the quantum stage of item EC1 on the series and write the annotated SDF."""
    src = os.path.join(OUT, "phenazine.sdf")
    if not os.path.isfile(src):
        raise PhenazineError("phenazine.sdf is not there; run --select first")
    mols = [m for m in Chem.SDMolSupplier(src, removeHs=False) if m is not None]
    if limit:
        mols = mols[:limit]
    names = [m.GetProp("id") for m in mols]
    cfg = O.Config(level=O.DEFAULT_LEVEL, fukui=True, solvent=SOLVENT, maxcore=maxcore,
                   workdir=os.path.join(OUT, "qm_work"))
    print(f"  {len(mols)} molecules, {O.DEFAULT_LEVEL} in {SOLVENT}, {jobs} at a time")
    print(f"  measured cost: 3 min for 22 atoms, 27 min for 44, 39 min for 58, three states each")
    t0 = time.time()
    done, failed = [], []

    def progress(message, fraction):
        print(f"    [{fraction * 100:5.1f}%] {message} "
              f"({(time.time() - t0) / 60:.0f} min elapsed)", flush=True)

    outcomes = O.annotate_molecules(mols, names, cfg, jobs=jobs, progress=progress)
    for oc in outcomes:
        (failed if oc.error else done).append(oc)
    path = os.path.join(OUT, "phenazine_qm.sdf")
    writer = Chem.SDWriter(path)
    for oc in outcomes:
        if oc.mol is not None:
            writer.write(oc.mol)
    writer.close()
    print(f"  wrote {path}: {len(done)} molecules, {len(failed)} failures, "
          f"{(time.time() - t0) / 3600:.2f} h")
    for oc in failed:
        print(f"    {oc.name}: {oc.error}")
    return path


# -------------------------------------------------------------------------------- the summary

def summarise() -> None:
    path = os.path.join(OUT, "phenazine_qm.sdf")
    if not os.path.isfile(path):
        path = os.path.join(OUT, "phenazine.sdf")
    mols = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    if not mols:
        print("  nothing to summarise")
        return
    print(f"  {len(mols)} molecules in {os.path.basename(path)}")
    sizes = np.array([m.GetNumAtoms() for m in mols])
    print(f"  atoms per molecule {sizes.mean():.1f} +/- {sizes.std():.1f} "
          f"({sizes.min()} to {sizes.max()})")
    subsets = {}
    for m in mols:
        subsets[m.GetProp("subset")] = subsets.get(m.GetProp("subset"), 0) + 1
    print("  by table: " + ", ".join(f"{k} {v}" for k, v in sorted(subsets.items())))
    y = np.array([float(m.GetProp(TARGET)) for m in mols])
    print(f"  {TARGET} {y.min():.3f} to {y.max():.3f} V, mean {y.mean():.3f}, "
          f"predicting the mean costs {np.mean(np.abs(y - y.mean())):.3f} V")
    for name in list(O.ATOM_PROPS_N) + list(O.ATOM_PROPS_SPIN):
        have = sum(1 for m in mols if m.HasProp(f"atom.dprop.{name}"))
        if have:
            print(f"  {name}: on {have} molecules")
    for name in ("IP", "EA", "gap", "spin_total_anion"):
        vals = [float(m.GetProp(name)) for m in mols if m.HasProp(name)]
        if vals:
            v = np.array(vals)
            print(f"  {name} {v.mean():+.3f} +/- {v.std():.3f} ({v.min():+.3f} to {v.max():+.3f})"
                  f" on {len(v)} molecules")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--fetch", action="store_true", help="download the source files")
    ap.add_argument("--force", action="store_true", help="download again even if a file is there")
    ap.add_argument("--select", action="store_true", help="build and align phenazine.sdf")
    ap.add_argument("--quantum", action="store_true", help="run the ORCA stage")
    ap.add_argument("--summary", action="store_true", help="what the files hold")
    ap.add_argument("--limit", type=int, default=0, help="only the first N molecules")
    ap.add_argument("--jobs", type=int, default=4, help="molecules at a time (ORCA here is serial)")
    ap.add_argument("--maxcore", type=int, default=1500, help="MB per ORCA job")
    args = ap.parse_args()
    if not any((args.fetch, args.select, args.quantum, args.summary)):
        ap.error("nothing to do: pass --fetch, --select, --quantum or --summary")
    if args.fetch:
        print("[fetch]")
        fetch_upstream(force=args.force)
    if args.select:
        print("[select]")
        mols = build_series()
        write_series(align(mols))
    if args.quantum:
        print("[quantum]")
        quantum(limit=args.limit, jobs=args.jobs, maxcore=args.maxcore)
    if args.summary:
        print("[summary]")
        summarise()


if __name__ == "__main__":
    main()
