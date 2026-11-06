"""Bonds of a periodic structure, perceived from distances for drawing.

A structure file states where the atoms are and nothing about which of them are bonded, so the
crystal view perceives the bonds.  The tests check the three things the drawing rests on, each
against something that is not the perception itself:

* **the molecule's own bonds.** The molecules of the Fe(110) sets came from SDF files, which do
  carry a bond table, so the perceived set can be compared with the real one on real molecules
  with their hydrogens -- in the open, and in a periodic cell that cuts the molecule in two.
* **a metal gets none.** A distance rule gives every atom of a close-packed slab about twelve
  neighbours; the rule is applied to the non-metal atoms only, so a slab stays spheres and the
  organic molecule on it gets its bonds. That is the case this feature exists for.
* **what a metal gets instead**: the first coordination shell as a polyhedron, with a
  coordination number that is right where it is known in advance.

Nothing here is a chemical claim: a bond perceived for drawing never reaches a field, a kernel or
a prediction, and the tests on the field weights of a cell elsewhere in the suite are what check
that.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("ase", reason="a periodic structure is an ASE structure")

from ase import Atoms                                                   # noqa: E402

from cmfqsar.bonds import (BONDING_ELEMENTS, MAX_BOND, Polyhedron,      # noqa: E402
                           coordination_polyhedra, covalent_radius, is_bonding,
                           perceive_bonds, periodic_bonds, polyhedra_mesh)
from cmfqsar.periodic import crystal_fields, inorganic_fieldset         # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FE110_SDF = os.path.join(ROOT, "examples", "corrosion_fe110", "157_benzimidazole_uma.sdf")


@pytest.fixture(scope="module")
def fieldset():
    return inorganic_fieldset([], physchem=["steric"], charge_model="electronegativity")


def cell_of(atoms, fieldset, name="test"):
    return crystal_fields(atoms, fieldset, name)


def benzene_ring(a=8.0, centre=None):
    """A flat benzene ring in a cubic cell; `centre` puts it where it is asked for."""
    r = 1.39
    xyz = np.array([(r * math.cos(k * math.pi / 3), r * math.sin(k * math.pi / 3), 0.0)
                    for k in range(6)])
    if centre is not None:
        xyz = xyz + np.asarray(centre, float)
    return Atoms(symbols=["C"] * 6, positions=xyz, cell=[a] * 3, pbc=True)


def rocksalt(a=5.64, cation="Na"):
    pos = np.array([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5],
                    [.5, 0, 0], [0, .5, 0], [0, 0, .5], [.5, .5, .5]])
    return Atoms(symbols=[cation] * 4 + ["Cl"] * 4, scaled_positions=pos, cell=[a] * 3,
                 pbc=True)


# --------------------------------------------------------------------------- #
# The rule itself
# --------------------------------------------------------------------------- #
def test_the_organic_elements_bond_and_the_metals_do_not():
    for s in ("C", "N", "O", "H", "S", "Cl", "Si", "P"):
        assert is_bonding(s), s
    for s in ("Fe", "Na", "Li", "Ni", "Pt", "La", "Al"):
        assert not is_bonding(s), s
    assert "C" in BONDING_ELEMENTS and "Fe" not in BONDING_ELEMENTS


def test_the_radius_of_an_organic_element_is_the_one_the_rest_of_the_program_uses():
    from cmfqsar.protein import COVALENT_RADII
    assert covalent_radius("C") == pytest.approx(COVALENT_RADII["C"])
    assert covalent_radius("O") == pytest.approx(COVALENT_RADII["O"])


def test_an_element_outside_the_protein_table_still_has_a_radius():
    """A crystal holds elements a protein never does; ASE's table covers them."""
    assert 1.0 < covalent_radius("La") < 2.5
    assert 0.5 < covalent_radius("Ti") < 2.0
    assert covalent_radius("") > 0                       # and nothing raises


def test_a_benzene_ring_gets_six_bonds():
    bonds = perceive_bonds(np.array(benzene_ring().positions), ["C"] * 6)
    assert len(bonds) == 6
    # every atom has two neighbours, which is what a ring is
    counts = np.bincount(np.asarray(bonds, int).ravel(), minlength=6)
    assert list(counts) == [2] * 6


def test_two_hydrogens_are_never_bonded_to_each_other():
    xyz = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    assert len(perceive_bonds(xyz, ["H", "H"])) == 0


def test_no_bond_is_drawn_longer_than_the_limit():
    xyz = np.array([[0.0, 0.0, 0.0], [MAX_BOND + 0.3, 0.0, 0.0]])
    assert len(perceive_bonds(xyz, ["I", "I"])) == 0


def test_a_metal_pair_gets_no_bond():
    xyz = np.array([[0.0, 0.0, 0.0], [2.48, 0.0, 0.0]])          # the Fe-Fe distance of bcc iron
    assert len(perceive_bonds(xyz, ["Fe", "Fe"])) == 0


def test_a_metal_ligand_bond_can_be_asked_for():
    """A molecular crystal of a metal complex is the case where the metal bonds are the point."""
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])           # a typical Fe-N distance
    assert len(perceive_bonds(xyz, ["Fe", "N"])) == 0
    assert len(perceive_bonds(xyz, ["Fe", "N"], skip_metals=False)) == 1


# --------------------------------------------------------------------------- #
# The periodic case
# --------------------------------------------------------------------------- #
def test_a_ring_inside_the_cell_has_six_bonds_and_needs_no_images(fieldset):
    d = periodic_bonds(cell_of(benzene_ring(centre=[4.0, 4.0, 4.0]), fieldset))
    assert d.n_distinct == 6
    assert d.n_crossing == 0
    assert d.n_images_drawn == 0
    assert len(d.coords) == d.n_cell == 6


def test_the_same_ring_wrapped_across_the_boundary_is_still_a_ring(fieldset):
    """This is what the images are for: a bond that leaves the cell is found, not lost."""
    at = benzene_ring()
    at.wrap()                                     # the ring now straddles three faces
    d = periodic_bonds(cell_of(at, fieldset))
    assert d.n_distinct == 6                      # the same six bonds
    assert d.n_crossing > 0                       # some of them leave the box
    assert d.n_images_drawn > 0                   # and the atoms they end on are drawn
    # every drawn image atom is an image of an atom of the cell
    assert np.all(d.source[d.n_cell:] < d.n_cell)
    # the first rows are still the cell, in its own order, so a site number still names its row
    assert np.allclose(d.coords[:d.n_cell], np.asarray(at.positions, float))
    assert list(d.source[:d.n_cell]) == list(range(d.n_cell))


def test_a_rock_salt_cell_gets_no_bonds_and_says_why(fieldset):
    d = periodic_bonds(cell_of(rocksalt(), fieldset))
    assert len(d.bonds) == 0
    assert d.skipped_metals == 4
    assert "4 of the 8 atoms are metal atoms" in d.describe()


def test_a_structure_without_a_cell_falls_back_to_the_plain_rule(fieldset):
    """A record with no lattice is drawn by the open rule rather than refused."""
    from cmfqsar.fields import FieldSet, build_fields
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from cmfqsar.align import embed_3d
    mol = embed_3d(Chem.AddHs(Chem.MolFromSmiles("c1ccccc1")))
    mf = build_fields(mol, FieldSet.create(physchem=["steric"]), "benzene")
    d = periodic_bonds(mf)
    assert d.n_distinct == 12                     # six ring bonds and six C-H bonds
    assert d.n_crossing == 0


@pytest.mark.skipif(not os.path.exists(FE110_SDF), reason="the Fe(110) set is not in the tree")
def test_the_perceived_bonds_are_the_molecules_own_bonds(fieldset):
    """Real molecules with their hydrogens, against the bond table their SDF file carries."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    exact = total = 0
    for mol in Chem.SDMolSupplier(FE110_SDF, removeHs=False, sanitize=True):
        if mol is None or not mol.GetNumBonds():
            continue
        total += 1
        truth = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
        got = {frozenset(map(int, p)) for p in
               perceive_bonds(np.array(mol.GetConformer().GetPositions(), float),
                              [a.GetSymbol() for a in mol.GetAtoms()])}
        exact += int(got == truth)
    assert total > 50
    # the benchmark measures this properly; here it only has to be right nearly always
    assert exact >= total - 2


@pytest.mark.skipif(not os.path.exists(FE110_SDF), reason="the Fe(110) set is not in the tree")
def test_an_organic_molecule_on_a_metal_slab_keeps_its_bonds_and_the_slab_keeps_none(fieldset):
    """The case this feature exists for, and the one the manual shows."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    from cmfqsar.slab import bcc110_slab
    mol = next(m for m in Chem.SDMolSupplier(FE110_SDF, removeHs=False, sanitize=True)
               if m is not None)
    xyz = np.array(mol.GetConformer().GetPositions(), float)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    truth = {frozenset((b.GetBeginAtomIdx(), b.GetEndAtomIdx())) for b in mol.GetBonds()}
    slab = bcc110_slab()
    placed = xyz.copy()
    placed[:, :2] -= placed[:, :2].mean(axis=0)          # over the corner, so it is cut in two
    at = Atoms(symbols=[a.symbol for a in slab] + symbols,
               positions=np.vstack([slab.positions, placed]),
               cell=np.asarray(slab.get_cell(), float), pbc=(True, True, False))
    at.wrap()
    d = periodic_bonds(cell_of(at, fieldset, "adsorbate"))
    n_slab = len(slab)
    assert d.skipped_metals == n_slab
    # not one stick reaches the metal
    for a, b in d.bonds:
        assert int(d.source[int(a)]) >= n_slab and int(d.source[int(b)]) >= n_slab
    # and the molecule's own bonds are all there, although the cell cuts it
    got = {frozenset(sorted(int(d.source[int(x)]) - n_slab for x in p)) for p in d.bonds}
    got = {p for p in got if len(p) == 2}
    assert got == truth
    assert d.n_crossing > 0


# --------------------------------------------------------------------------- #
# What a metal gets instead
# --------------------------------------------------------------------------- #
def test_rock_salt_is_six_coordinate(fieldset):
    polys = coordination_polyhedra(cell_of(rocksalt(), fieldset))
    assert len(polys) == 4                        # one per sodium
    assert {p.coordination for p in polys} == {6}
    assert all(p.element == "Na" for p in polys)
    assert all(set(p.neighbours) == {"Cl"} for p in polys)
    assert all("Na 6-coordinate (Cl6)" == p.label() for p in polys)
    assert not any(p.planar for p in polys)


def test_the_polyhedra_of_a_cell_are_one_mesh(fieldset):
    polys = coordination_polyhedra(cell_of(rocksalt(), fieldset))
    mesh = polyhedra_mesh(polys)
    assert mesh is not None
    verts, faces = mesh
    assert len(verts) == sum(p.coordination for p in polys)
    assert len(faces) == sum(len(p.faces) for p in polys)
    assert faces.max() < len(verts)               # the indices stay inside the joined mesh
    assert polyhedra_mesh([]) is None


def test_a_cell_of_non_metals_alone_has_no_polyhedra(fieldset):
    """Carbon gets bonds, so it does not need a polyhedron and does not get one."""
    assert coordination_polyhedra(cell_of(benzene_ring(centre=[4.0, 4.0, 4.0]), fieldset)) == []


def test_a_flat_shell_is_drawn_as_a_polygon(fieldset):
    """A square-planar shell has no volume, so a convex hull of it fails; it is still a shape."""
    a = 12.0
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0], [0.0, -2.0, 0.0]]) + a / 2
    at = Atoms(symbols=["Ni", "Cl", "Cl", "Cl", "Cl"], positions=pos, cell=[a] * 3, pbc=True)
    polys = coordination_polyhedra(cell_of(at, fieldset))
    assert len(polys) == 1
    p = polys[0]
    assert p.planar is True
    assert p.coordination == 4
    assert len(p.faces) == 2                      # a square is two triangles
    assert isinstance(p, Polyhedron)


def test_a_shell_of_too_few_neighbours_is_left_out(fieldset):
    """Three neighbours are a triangle and are drawn; asking for four leaves them out."""
    a = 12.0
    pos = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                    [-1.0, 1.73, 0.0], [-1.0, -1.73, 0.0]]) + a / 2
    at = Atoms(symbols=["Ni", "Cl", "Cl", "Cl"], positions=pos, cell=[a] * 3, pbc=True)
    assert coordination_polyhedra(cell_of(at, fieldset), min_coordination=4) == []
    polys = coordination_polyhedra(cell_of(at, fieldset), min_coordination=3)
    assert len(polys) == 1 and polys[0].coordination == 3 and polys[0].planar


def test_the_number_of_polyhedra_is_capped(fieldset):
    polys = coordination_polyhedra(cell_of(rocksalt(), fieldset), limit=2)
    assert len(polys) == 2

def test_the_coordination_number_is_reported_for_every_centre(fieldset):
    """Including the centres whose shell is too small to draw, which is why a cell can hold
    more metal atoms than polyhedra."""
    from cmfqsar.bonds import coordination_numbers
    cell = cell_of(rocksalt(), fieldset)
    numbers = coordination_numbers(cell)
    assert [n for _i, _e, n in numbers] == [6, 6, 6, 6]
    assert {e for _i, e, _n in numbers} == {"Na"}
    # the shell rule is shared, so the polyhedra are exactly the centres with a big enough shell
    polys = coordination_polyhedra(cell)
    assert len(polys) == sum(1 for _i, _e, n in numbers if n >= 3)
    assert [p.coordination for p in polys] == [n for _i, _e, n in numbers]


def test_a_centre_with_one_near_neighbour_is_counted_but_not_drawn(fieldset):
    """A site whose nearest neighbour is much closer than the next has a shell of one atom."""
    from cmfqsar.bonds import coordination_numbers
    a = 14.0
    pos = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [5.0, 0.0, 0.0],
                    [0.0, 5.0, 0.0]]) + a / 2
    at = Atoms(symbols=["Li", "O", "O", "O"], positions=pos, cell=[a] * 3, pbc=True)
    cell = cell_of(at, fieldset)
    assert [n for _i, _e, n in coordination_numbers(cell)] == [1]
    assert coordination_polyhedra(cell) == []

# --------------------------------------------------------------------------- #
# The example file the manual tells a reader to open
# --------------------------------------------------------------------------- #
ADSORBATE_CELL = os.path.join(ROOT, "examples", "corrosion_fe110",
                              "benzimidazole_on_fe110.extxyz")


@pytest.mark.skipif(not os.path.exists(ADSORBATE_CELL), reason="the example cell is not built")
def test_the_example_cell_is_a_slab_with_a_molecule_on_it(fieldset):
    """`examples/corrosion_fe110/benzimidazole_on_fe110.extxyz`, which the manual opens.

    It exists so that the surface side of the window can be opened without running a relaxation,
    so the two things it has to be are checked here: the slab is in the surface frame, and the
    molecule on it gets bonds while the iron gets none.
    """
    from cmfqsar.periodic import read_crystal_set

    cells, names = read_crystal_set(ADSORBATE_CELL, pbc="ab")
    assert len(cells) >= 8
    assert names[0] == "Benzimidazole"
    for atoms, name in zip(cells, names):
        assert atoms.info.get("slab_atoms") == 144, name
        assert float(atoms.info["E_ads_kJmol"]) < 0.0, name
        iron = [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == "Fe"]
        assert len(iron) == 144, name
        z = np.asarray(atoms.positions, float)[:, 2]
        # every atom inside its cell, as a relaxation of an adsorbed state writes it, with the
        # molecule above the top layer of the slab
        assert z.min() >= -1e-6 and z.max() <= float(atoms.cell[2, 2]) + 1e-6, name
        assert z[len(iron):].min() > z[iron].max(), name
        assert tuple(bool(p) for p in atoms.pbc) == (True, True, False), name

    crystal = cell_of(cells[0], fieldset, names[0])
    d = periodic_bonds(crystal)
    assert d.skipped_metals == 144
    assert d.n_distinct == 16                      # benzimidazole with its hydrogens
    for a, b in d.bonds:                           # and not one stick reaches the iron
        assert int(d.source[int(a)]) >= 144 and int(d.source[int(b)]) >= 144


@pytest.mark.skipif(not os.path.exists(ADSORBATE_CELL), reason="the example cell is not built")
def test_the_script_rebuilds_the_example_cell_byte_for_byte(tmp_path):
    """The file is derived, so the script that writes it must be the record of how."""
    import subprocess
    script = os.path.join(ROOT, "examples", "corrosion_fe110", "make_adsorbate_cell.py")
    out = tmp_path / "again.extxyz"
    subprocess.run([sys.executable, script, "--out", str(out)], check=True,
                   capture_output=True)
    with open(ADSORBATE_CELL, "rb") as fh:
        shipped = fh.read().replace(b"\r\n", b"\n")
    with open(out, "rb") as fh:
        again = fh.read().replace(b"\r\n", b"\n")
    assert again == shipped


@pytest.mark.skipif(not os.path.exists(ADSORBATE_CELL), reason="the example cell is not built")
def test_the_corner_placement_puts_a_bond_across_the_cell_boundary(fieldset, tmp_path):
    """`--place corner` is what the manual names for seeing a bond leave the box."""
    import subprocess
    script = os.path.join(ROOT, "examples", "corrosion_fe110", "make_adsorbate_cell.py")
    out = tmp_path / "corner.extxyz"
    subprocess.run([sys.executable, script, "--place", "corner", "--count", "1",
                    "--out", str(out)], check=True, capture_output=True)
    from cmfqsar.periodic import read_crystal_set
    cells, names = read_crystal_set(str(out), pbc="ab")
    d = periodic_bonds(cell_of(cells[0], fieldset, names[0]))
    assert d.n_distinct == 16                      # the same molecule, the same bonds
    assert d.n_crossing > 0                        # but some of them leave the cell
    assert d.n_images_drawn > 0
