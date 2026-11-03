"""
Ether molecules with the energy a cation gains by binding them, for plan item EC7.

    Wang W., Yang T., Harris W. H., Gomez-Bombarelli R.
    "Active learning and neural network potentials accelerate molecular screening of
    ether-based solvate ionic liquids", Chem. Commun. 56 (2020) 8920.
    Data at https://github.com/learningmatter-mit/MLElectrolytes (branch `master`).

Why this set is here
--------------------
Item EC7 asks for a cation probe: the field of a solvent molecule read as the energy a lithium,
sodium or potassium ion would feel in it.  Measuring such a probe needs a set whose target *is*
that energy, and this is one.  It holds 1,086 saturated acyclic ethers -- glymes and their
branched relatives, carbon, hydrogen and oxygen only -- with the binding energy of Li(+), Na(+)
and Mg(2+) computed at wB97X-D3/def2-TZVP and at BP86-D3/def2-SVP, and the relaxed geometry of
every lithium complex.

Three properties of the set decide what can be measured on it:

* **Three cations on the same molecules.**  The selectivity between two cations, which is what an
  ionophore is designed for, is a difference of two columns of the same table.  That makes this
  the multi-task problem of item E1 with descriptors of the endpoint -- the ion's radius, charge
  and polarisability -- rather than a label per endpoint.
* **The geometry is the bound one.**  A glyme is a flexible chain whose free conformers are many
  and whose bound conformer wraps the ion.  A field computed on a conformer picked by a
  generator would not be a field of the molecule that was measured.  The distributed geometries
  are the relaxed complexes, so this script writes the ether in the conformation it has when the
  lithium sits in it, with the ion's position kept as a property.  The cost is stated plainly:
  the geometries exist for lithium only, so a sodium or magnesium model built on them uses a
  lithium-shaped conformer.
* **The simple answer is already in the file.**  The number of oxygens and the carbon-to-oxygen
  ratio are columns of the distribution, and the oxygen count alone explains about 38 per cent
  of the lithium binding energy (correlation -0.62 on the 1,049 molecules that carry it).  Any
  field model has to be measured against them, as item BT3 was measured against the composition.

What is kept and what is dropped
--------------------------------
From ``data/cluster.csv`` this script keeps the SMILES, the abbreviation, the oxygen count, the
carbon-to-oxygen ratio, the molecule's HOMO energy and the six binding energies, and it drops the
ion HOMO energies, the dipoles and the absolute energies, which no model here uses.  A cell that
is empty upstream stays empty.  From ``geoms/ether_li.pkl`` it keeps the complexes whose atom
counts agree with their SMILES; each becomes one record of ``geometries.sdf`` holding the ether
alone, in the complex geometry, with the lithium position in the ``ion_xyz`` property.

The geometry file is a Python pickle.  Opening a pickle runs whatever code it names, so this
script opens it with an unpickler that admits numpy arrays and nothing else, and refuses any
other class.

Status of the copy
------------------
The distribution states no licence.  Under ``docs/public_3dqsar_datasets.md`` that makes the
files written here **unclear**: the facts are the paper's and redistribution is customary in the
field but not explicitly permitted.  The row in that document names the origin and this script,
which rebuilds both files from the original distribution.

Usage
-----
    python examples/make_ether_examples.py --fetch      # download, then write examples/ether/
    python examples/make_ether_examples.py              # write from an existing upstream copy
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import pickle
import sys
import urllib.request

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "ether")
DEFAULT_ROOT = os.path.join(DEFAULT_OUT, "upstream")

RAW = "https://raw.githubusercontent.com/learningmatter-mit/MLElectrolytes/master/"
#: file name -> (path in the distribution, SHA-256 of the copy fetched on 4 November 2026)
UPSTREAM = {
    "cluster.csv": ("data/cluster.csv",
                    "8dbaef837d9498dc7720708be05f4d5626ae0cc3d67871d0126b90a6bed02062"),
    "ether_li.pkl": ("geoms/ether_li.pkl",
                     "774a50b5d867c7307bf620a6c87ba38488ba6488fac1f7d06d02b6a5bad74fe2"),
}

#: the columns written out, with the upstream name each comes from
COLUMNS = (
    ("name", None),
    ("smiles", "smiles"),
    ("abbr", "abbr"),
    ("o_count", "O_count"),
    ("c_to_o", "C:O"),
    ("homo", "wbx_mol_homo"),
    ("li_ebind", "wbx_li_ebind"),
    ("na_ebind", "wbx_na_ebind"),
    ("mg_ebind", "wbx_mg_ebind"),
    ("li_ebind_bp86", "bp86_li_ebind"),
    ("na_ebind_bp86", "bp86_na_ebind"),
    ("mg_ebind_bp86", "bp86_mg_ebind"),
)

#: what the unpickler is allowed to build: numpy arrays and their dtypes, nothing else. The
#: distributed file was written with numpy 1, which names the array constructor
#: numpy.core.multiarray._reconstruct; numpy 2 writes numpy._core.multiarray._reconstruct, so
#: both spellings of the same function are admitted.
PICKLE_ALLOWED = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}

SYMBOL = {1: "H", 3: "Li", 6: "C", 8: "O"}
#: a bond longer than this between atoms the SMILES bonds means the geometry was not understood
MAX_BOND = 1.9


class _ArraysOnly(pickle.Unpickler):
    """An unpickler that refuses every class but numpy's array constructors."""

    def find_class(self, module, name):
        if (module, name) not in PICKLE_ALLOWED:
            raise pickle.UnpicklingError(
                f"the geometry file names {module}.{name}, which this reader refuses; "
                f"only numpy arrays are admitted")
        return super().find_class(module, name)


# --------------------------------------------------------------------------------- the download


def fetch_upstream(root: str) -> None:
    """Download both distribution files into `root` and check their hashes."""
    os.makedirs(root, exist_ok=True)
    for name, (path, want) in UPSTREAM.items():
        url = RAW + path
        print("downloading " + url)
        with urllib.request.urlopen(url) as fh:          # noqa: S310 - a fixed https URL
            blob = fh.read()
        got = hashlib.sha256(blob).hexdigest()
        if got != want:
            raise SystemExit(f"{name}: SHA-256 mismatch. The published file this script was "
                             f"written against has {want}; this download has {got} "
                             f"({len(blob)} bytes). The distribution has changed, so check it "
                             f"before using it.")
        with open(os.path.join(root, name), "wb") as out:
            out.write(blob)
        print(f"  {len(blob)} bytes, SHA-256 as published")


# ------------------------------------------------------------------------------------ the table


def read_table(path: str):
    """The upstream rows, with the kept columns under their new names.

    One molecule, hexaglyme, has two identical rows upstream; the second is dropped, and so is
    any other repeat, because a molecule measured once must not become two samples.
    """
    out, seen, repeats = [], {}, []
    with io.open(path, encoding="utf-8", errors="replace") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            rec = {"name": f"e{i:04d}"}
            for new, old in COLUMNS:
                if old is None:
                    continue
                value = (row.get(old) or "").strip()
                rec[new] = value
            if not rec.get("smiles"):
                continue
            first = seen.get(rec["smiles"])
            if first is not None:
                same = all(rec[new] == first[new] for new, old in COLUMNS if old != "smiles"
                           and old is not None and new != "name")
                repeats.append((rec["smiles"], "identical" if same else "different values"))
                continue
            seen[rec["smiles"]] = rec
            out.append(rec)
    for smiles, how in repeats:
        print(f"    repeated row dropped ({how}): {smiles}")
    return out


def read_geometries(path: str):
    """``{smiles: (ebind, atomic numbers, coordinates)}`` from the relaxed complexes."""
    with open(path, "rb") as fh:
        raw = _ArraysOnly(io.BytesIO(fh.read())).load()
    if not isinstance(raw, dict):
        raise SystemExit(f"{os.path.basename(path)} holds a {type(raw).__name__}, not a table")
    out = {}
    for key, rec in raw.items():
        smiles = str(key).split(".[")[0]
        xyz = np.asarray(rec["xyz"], float)
        if xyz.ndim != 2 or xyz.shape[1] != 4:
            raise SystemExit(f"{smiles}: the geometry has shape {xyz.shape}, not (n, 4)")
        out[smiles] = (float(rec["ebind"]), xyz[:, 0].astype(int), xyz[:, 1:])
    return out


# -------------------------------------------------------------------------------- the molecules


def ether_in_complex_geometry(smiles: str, numbers, coords):
    """The ether alone, in the geometry it has in the complex, plus the ion's position.

    The ion is dropped and the remaining atoms are read as a molecule by perceiving the bonds
    from the coordinates, then matched to the SMILES so that the atom order is the SMILES order.
    These molecules are saturated, acyclic and made of carbon, hydrogen and oxygen only, so the
    perception has no bond order to guess.
    """
    numbers = np.asarray(numbers, int)
    coords = np.asarray(coords, float)
    ion = np.where(numbers == 3)[0]
    if len(ion) != 1:
        raise ValueError(f"{smiles}: the complex holds {len(ion)} lithium atoms, not one")
    ion_xyz = coords[ion[0]].copy()
    keep = np.ones(len(numbers), bool)
    keep[ion[0]] = False
    numbers, coords = numbers[keep], coords[keep]
    unknown = sorted(set(int(z) for z in numbers) - set(SYMBOL))
    if unknown:
        raise ValueError(f"{smiles}: the complex holds the elements {unknown}, which this "
                         f"script does not expect")

    ref = Chem.AddHs(Chem.MolFromSmiles(smiles))
    if ref.GetNumAtoms() != len(numbers):
        raise ValueError(f"{smiles}: the SMILES has {ref.GetNumAtoms()} atoms and the geometry "
                         f"{len(numbers)}")
    block = [str(len(numbers)), smiles]
    block += [f"{SYMBOL[int(z)]} {x:.6f} {y:.6f} {z_:.6f}"
              for z, (x, y, z_) in zip(numbers, coords)]
    from_xyz = Chem.MolFromXYZBlock("\n".join(block) + "\n")
    if from_xyz is None:
        raise ValueError(f"{smiles}: the coordinates cannot be read")
    rdDetermineBonds.DetermineConnectivity(from_xyz)
    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    query = Chem.AdjustQueryProperties(Chem.RWMol(ref), params)
    match = from_xyz.GetSubstructMatch(query, useChirality=False)
    if len(match) != ref.GetNumAtoms():
        raise ValueError(f"{smiles}: the bonds perceived from the coordinates do not match "
                         f"the SMILES")

    conf = Chem.Conformer(ref.GetNumAtoms())
    for i, j in enumerate(match):
        conf.SetAtomPosition(i, Point3D(*[float(v) for v in coords[j]]))
    ref.RemoveAllConformers()
    ref.AddConformer(conf, assignId=True)
    pos = ref.GetConformer().GetPositions()
    for b in ref.GetBonds():
        d = float(np.linalg.norm(pos[b.GetBeginAtomIdx()] - pos[b.GetEndAtomIdx()]))
        if d > MAX_BOND:
            raise ValueError(f"{smiles}: two bonded atoms are {d:.2f} A apart in the geometry")
    return ref, ion_xyz


# ----------------------------------------------------------------------------------- the output


def write_table(rows, path: str) -> None:
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([new for new, _old in COLUMNS])
        for r in rows:
            w.writerow([r.get(new, "") for new, _old in COLUMNS])


def write_geometries(rows, geoms, path: str):
    """One SDF record per molecule whose geometry is available and understood."""
    written, refused = 0, []
    writer = Chem.SDWriter(path)
    try:
        for r in rows:
            smiles = r["smiles"]
            if smiles not in geoms:
                refused.append((smiles, "no relaxed complex in the distribution"))
                continue
            _ebind, numbers, coords = geoms[smiles]
            try:
                mol, ion_xyz = ether_in_complex_geometry(smiles, numbers, coords)
            except ValueError as exc:
                refused.append((smiles, str(exc).split(": ", 1)[-1]))
                continue
            mol.SetProp("_Name", r["name"])
            mol.SetProp("smiles", smiles)
            mol.SetProp("ion_xyz", " ".join(f"{v:.6f}" for v in ion_xyz))
            for key in ("abbr", "o_count", "c_to_o", "homo",
                        "li_ebind", "na_ebind", "mg_ebind",
                        "li_ebind_bp86", "na_ebind_bp86", "mg_ebind_bp86"):
                if r.get(key):
                    mol.SetProp(key, r[key])
            writer.write(mol)
            written += 1
    finally:
        writer.close()
    return written, refused


README = """\
# Ether molecules with cation binding energies (plan item EC7)

Built by `examples/make_ether_examples.py --fetch` from the MLElectrolytes distribution,
<https://github.com/learningmatter-mit/MLElectrolytes>, branch `master`, of

> Wang W., Yang T., Harris W. H., Gomez-Bombarelli R. "Active learning and neural network
> potentials accelerate molecular screening of ether-based solvate ionic liquids",
> Chem. Commun. 56 (2020) 8920.

The distributor states no licence, so the status of this copy is **unclear** in the sense of
`docs/public_3dqsar_datasets.md`: the facts are the paper's, redistribution is customary in the
field but not explicitly permitted, and the script above rebuilds both files from the original
distribution.

## The files

| File | Content |
|---|---|
| `ether_ions.csv` | {n_rows} saturated acyclic ethers: SMILES, oxygen count, carbon-to-oxygen ratio, HOMO energy, and the binding energy of Li(+), Na(+) and Mg(2+) in eV at wB97X-D3/def2-TZVP and at BP86-D3/def2-SVP |
| `geometries.sdf` | {n_geom} of the same molecules in the conformation of the relaxed lithium complex, with the ion position in the `ion_xyz` property and the binding energies repeated as properties |

An empty cell means the quantity is missing upstream, not zero. Of the {n_rows} molecules,
{n_li} carry a lithium binding energy, {n_na} a sodium one and {n_mg} a magnesium one at
wB97X-D3/def2-TZVP.

## What the geometries are, and what that costs

The coordinates are the relaxed **complex**: the ether wrapped around the lithium ion, with the
ion removed and its position kept. That is the conformation whose energy was measured, so a
field computed on it is a field of the molecule the number belongs to. The distribution holds
lithium complexes only, so a sodium or magnesium model built on these coordinates uses a
lithium-shaped conformer. The binding energies of the three ions correlate (+0.80 for lithium
against sodium, +0.85 against magnesium), so the error this causes is not visible in a
single-ion model; it is a limit on what the selectivity between two ions can be read as.

## The simple baselines

The oxygen count and the carbon-to-oxygen ratio are columns of the table because any model of
these energies has to be compared with them. On the molecules that carry a lithium binding
energy the oxygen count alone has a correlation of -0.62 with it, and the carbon-to-oxygen ratio
+0.04. The difference between the lithium and sodium energies -- the selectivity -- has a
correlation of -0.05 with the oxygen count, so the count says nothing about it.
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build the ether-cation example set from the MLElectrolytes distribution.")
    ap.add_argument("--data-root", default=DEFAULT_ROOT,
                    help="folder holding cluster.csv and ether_li.pkl "
                         "(default: examples/ether/upstream)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT,
                    help="where the table and the geometries are written")
    ap.add_argument("--fetch", action="store_true",
                    help="download both files from the distribution into --data-root first")
    args = ap.parse_args()

    if args.fetch:
        fetch_upstream(args.data_root)
    table_path = os.path.join(args.data_root, "cluster.csv")
    geom_path = os.path.join(args.data_root, "ether_li.pkl")
    for path in (table_path, geom_path):
        if not os.path.exists(path):
            raise SystemExit(f"{path} is missing; run this script with --fetch first")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_table(table_path)
    geoms = read_geometries(geom_path)
    print(f"{len(rows)} rows in the table, {len(geoms)} relaxed complexes")

    out_table = os.path.join(args.out_dir, "ether_ions.csv")
    write_table(rows, out_table)
    print(f"wrote {out_table}")

    out_geom = os.path.join(args.out_dir, "geometries.sdf")
    written, refused = write_geometries(rows, geoms, out_geom)
    print(f"wrote {out_geom}: {written} molecules, {len(refused)} refused")
    for smiles, why in refused[:10]:
        print(f"    {smiles}: {why}")
    if len(refused) > 10:
        print(f"    ... and {len(refused) - 10} more")

    counts = {}
    for key in ("li_ebind", "na_ebind", "mg_ebind"):
        counts[key] = sum(1 for r in rows if r.get(key))
    with io.open(os.path.join(args.out_dir, "README.md"), "w", encoding="utf-8",
                 newline="\n") as fh:
        fh.write(README.format(n_rows=len(rows), n_geom=written, n_li=counts["li_ebind"],
                               n_na=counts["na_ebind"], n_mg=counts["mg_ebind"]))
    print(f"wrote {os.path.join(args.out_dir, 'README.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
