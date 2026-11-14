"""Every structure of every example file is a structure of its molecule.

On 14 November 2026 two example files were found to hold a structure that was not the molecule:
an SF5 group that a force field without parameters for it had left with two fluorines on one
point.  No model score showed either of them.  A scan of all example files then found three
more kinds of the same fault in files that this repository builds: hydrochlorides embedded with
the chloride on top of the compound, further SF5 groups, and a nitrogen that the refinement had
left on one line with its two carbons.

The test below is that scan.  It reads every record of every SDF under ``examples/`` and applies
:func:`cmfqsar.orca.geometry_faults`: no bond length outside 0.7 to 1.3 times the sum of the
covalent radii, and no two atoms without a bond closer than 1.3 A.  A record may fail only if it
is listed in ``examples/geometry_exceptions.csv``, with the reason.  That list holds two kinds of
record and no other:

``source``
    the structure is distributed like this by its authors (a published aligned structure, a
    docked pose, a relaxed complex), and this repository does not change the coordinates of a
    source;
``marked``
    the molecule broke in a calculation and the data set says so in a column of its own.

A new file with a faulty record fails the test, and so does a listed record that no longer
fails, so the list cannot go out of date in either direction.
"""
import csv
import os
import subprocess

from rdkit import Chem, RDLogger

from cmfqsar.orca import geometry_faults

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EXAMPLES = os.path.join(ROOT, "examples")
EXCEPTIONS = os.path.join(EXAMPLES, "geometry_exceptions.csv")
CLASSES = ("source", "marked")


def example_files():
    """The SDF files under ``examples/``, as paths with forward slashes from the repository
    root: the tracked ones in a git checkout, every one on disk otherwise."""
    try:
        out = subprocess.run(["git", "ls-files", "examples"], cwd=ROOT, capture_output=True,
                             text=True, check=True).stdout.split("\n")
        found = sorted(f for f in out if f.lower().endswith(".sdf"))
    except (OSError, subprocess.CalledProcessError):
        found = []
    if not found:
        for folder, _dirs, names in os.walk(EXAMPLES):
            for name in names:
                if name.lower().endswith(".sdf"):
                    path = os.path.relpath(os.path.join(folder, name), ROOT)
                    found.append(path.replace(os.sep, "/"))
        found.sort()
    return found


def scan(files):
    """{(file, position of the record from 1, name): the faults as one text} and the number of
    records read."""
    RDLogger.DisableLog("rdApp.*")
    found, n = {}, 0
    try:
        for f in files:
            # the check reads the bond table and the coordinates, so the records are not sanitised
            supplier = Chem.SDMolSupplier(os.path.join(ROOT, f), removeHs=False, sanitize=False)
            for position, mol in enumerate(supplier, start=1):
                n += 1
                if mol is None:
                    found[(f, position, "")] = "the record cannot be read"
                    continue
                if mol.GetNumConformers() == 0 or not mol.GetConformer().Is3D():
                    continue
                faults = geometry_faults(mol)
                if faults:
                    name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""
                    found[(f, position, name)] = "; ".join(faults)
    finally:
        RDLogger.EnableLog("rdApp.*")
    return found, n


def read_exceptions():
    with open(EXCEPTIONS, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return {(r["file"], int(r["record"]), r["name"]): r for r in rows}


def test_the_list_of_exceptions_names_a_class_and_a_reason_for_every_record():
    rows = read_exceptions()
    assert rows, "the list of exceptions is empty or missing"
    for key, row in rows.items():
        assert row["class"] in CLASSES, (key, row["class"])
        assert row["finding"].strip(), key
        assert row["reason"].strip(), key
        assert os.path.isfile(os.path.join(ROOT, row["file"])), key


def test_no_record_of_any_example_file_has_an_impossible_geometry_unless_it_is_listed():
    files = example_files()
    assert len(files) > 300          # the folder is there and is read
    found, n = scan(files)
    assert n > 30000
    listed = read_exceptions()
    new = sorted(k for k in found if k not in listed)
    assert not new, ("records that fail the geometry check and are not in "
                     "examples/geometry_exceptions.csv: "
                     + "; ".join(f"{f} record {i} ({name}): {found[(f, i, name)]}"
                                 for f, i, name in new[:20]))
    gone = sorted(k for k in listed if k not in found)
    assert not gone, ("records listed in examples/geometry_exceptions.csv that pass the check "
                      "now; take them off the list: "
                      + "; ".join(f"{f} record {i} ({name})" for f, i, name in gone[:20]))
    changed = sorted(k for k in listed if found[k] != listed[k]["finding"])
    assert not changed, ("listed records whose fault is not the one on the list: "
                         + "; ".join(f"{f} record {i} ({name}): {found[(f, i, name)]}"
                                     for f, i, name in changed[:20]))


def test_the_check_finds_a_long_bond_and_a_close_pair_and_passes_a_whole_molecule():
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    mol = Chem.AddHs(Chem.MolFromSmiles("CCN"))
    assert AllChem.EmbedMolecule(mol, randomSeed=7) == 0
    assert geometry_faults(mol) == []
    conf = mol.GetConformer()
    start = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]

    def moved(atom, to):
        m = Chem.Mol(mol)
        m.GetConformer().SetAtomPosition(atom, Point3D(*to))
        return m

    # a bond that is too long: the nitrogen 2.5 A from its carbon
    c, n = start[1], start[2]
    length = c.Distance(n)
    far = [c.x + 2.5 * (n.x - c.x) / length, c.y + 2.5 * (n.y - c.y) / length,
           c.z + 2.5 * (n.z - c.z) / length]
    faults = geometry_faults(moved(2, far))
    assert any(f.startswith("bond C2-N3 2.50 A (1.70 of the covalent radii)") for f in faults)
    # two atoms without a bond on nearly one point: a hydrogen of the methyl group on the nitrogen
    faults = geometry_faults(moved(3, [n.x + 0.3, n.y, n.z]))
    assert any(f.startswith("no bond") for f in faults)
    # the limits are arguments
    assert geometry_faults(mol, contact=3.0) != []
