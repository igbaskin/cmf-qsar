"""Tests of rigid porous hosts as pockets (plan item M7, `cmfqsar/frameworks.py`)."""
import numpy as np
import pytest
from ase import Atoms
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar import frameworks as F
from cmfqsar.complementarity import (complementarity, interaction_profile, pocket_fields,
                                     pocket_residues)
from cmfqsar.fields import FieldSet, build_fields
from cmfqsar.protein import host_from_molecule

FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]


def fieldset():
    return FieldSet.create(physchem=FIELDS, alpha=1.0, charge_model="gasteiger",
                           param_scheme="rdkit")


def cubic_framework(a=8.0):
    """Si on the corners and O on the edge midpoints of a cube: one cage at the cube centre."""
    return Atoms("SiO3", positions=[[0, 0, 0], [a / 2, 0, 0], [0, a / 2, 0], [0, 0, a / 2]],
                 cell=np.eye(3) * a, pbc=True)


def molecule(smiles, seed=0):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


# --------------------------------------------------------------------------- pores

def test_the_cage_of_a_cubic_framework_has_its_exact_radius():
    a = 8.0
    at = cubic_framework(a)
    pores = F.find_pores(at, spacing=0.5, min_radius=3.0)
    assert len(pores) == 1
    p = pores[0]
    assert np.allclose(p.centre, [a / 2] * 3, atol=1e-3)
    assert p.radius == pytest.approx(a / np.sqrt(2) - F.VDW["O"], abs=1e-4)
    assert p.kind == "cage" and p.anisotropy == pytest.approx(1.0, abs=0.05)
    assert np.allclose(p.fractional, [0.5, 0.5, 0.5], atol=1e-3)


def test_walls_of_atoms_leave_a_channel_between_them():
    """Two planes of atoms, x = 0 and y = 0, repeat into square tubes along z."""
    wall = [[0.0, y, 0.0] for y in np.arange(0.0, 8.0, 1.6)]
    wall += [[x, 0.0, 0.0] for x in np.arange(1.6, 8.0, 1.6)]
    at = Atoms(f"C{len(wall)}", positions=wall, cell=np.diag([8.0, 8.0, 1.6]), pbc=True)
    pores = F.find_pores(at, spacing=0.25, min_radius=2.0)
    assert pores
    p = pores[0]
    assert p.kind == "channel"
    assert abs(p.axis[2]) == pytest.approx(1.0, abs=1e-3)
    assert np.allclose(p.centre[:2], [4.0, 4.0], atol=1e-3)


def test_free_distance_counts_images_and_can_leave_an_element_out():
    at = cubic_framework(8.0)
    centre = np.array([[4.0, 4.0, 4.0]])
    with_si = F.free_distance(at, centre)[0]
    oxygen_only = F.free_distance(at, centre, radii={"Si": -1.0})[0]
    assert with_si == pytest.approx(oxygen_only)          # the oxygens are nearer here
    near_si = np.array([[0.5, 0.5, 0.5]])
    assert F.free_distance(at, near_si)[0] < F.free_distance(at, near_si, radii={"Si": -1.0})[0]
    # the point next to the far corner sees the image of the atom at the origin
    corner = np.array([[7.9, 7.9, 7.9]])
    assert F.free_distance(at, corner)[0] < 0.0


def test_nearest_pore_counts_periodic_images():
    at = cubic_framework(8.0)
    pores = F.find_pores(at, spacing=0.5, min_radius=3.0)
    assert F.nearest_pore(pores, [-4.0, -4.0, -4.0], at) is pores[0]
    with pytest.raises(ValueError, match="no pore"):
        F.nearest_pore([], [0, 0, 0], at)


# --------------------------------------------------------------------------- bonds and types

def test_atom_types_count_neighbours_across_the_boundary():
    at = cubic_framework(3.2)                              # Si-O 1.6 A: every edge is a bond
    types = F.atom_types(at)
    assert types == ["Si6", "O2", "O2", "O2"]
    assert F._molecules(at) is None                         # one infinite framework


def test_a_molecule_split_by_the_boundary_is_made_whole():
    at = Atoms("N2", positions=[[0.5, 5.0, 5.0], [9.4, 5.0, 5.0]], cell=np.eye(3) * 10.0,
               pbc=True)
    mol, image = F._molecules(at)
    assert mol[0] == mol[1]
    p = F.framework_pocket(at, [0.5, 5.0, 5.0], radius=0.5, whole_molecules=True)
    assert p.n_atoms == 2
    assert np.linalg.norm(p.coords[0] - p.coords[1]) == pytest.approx(1.1, abs=1e-9)
    with pytest.raises(ValueError, match="infinite"):
        F.framework_pocket(cubic_framework(3.2), [1.6, 1.6, 1.6], whole_molecules=True)


# --------------------------------------------------------------------------- the pocket

def test_the_pocket_is_the_host_atom_for_atom_once_the_box_is_large():
    host = host_from_molecule(molecule("c1ccccc1O"))
    centre = host.coords.mean(axis=0)
    reach = float(np.max(np.linalg.norm(host.coords - centre, axis=1)))
    cut = reach + 0.5
    fs = fieldset()
    ref = pocket_fields(host, fs)
    big = F.boxed_host(host.coords, host.elements, reach + cut + 2.0)
    p = F.framework_pocket(big, centre, radius=cut, units=host.resids - 1, unit_name="MON")
    assert p.n_atoms == host.n_atoms and not np.any(p.source_image)
    assert np.array_equal(p.coords, host.coords)
    got = pocket_fields(p, fs)
    assert np.array_equal(got.coords, ref.coords) and np.array_equal(got.weights, ref.weights)
    box = 2.0 * reach + 1.0
    small = F.boxed_host(host.coords, host.elements, box)
    q = F.framework_pocket(small, centre, radius=box, units=host.resids - 1, unit_name="MON")
    assert q.n_atoms > host.n_atoms and np.any(q.source_image)
    assert np.array_equal(q.coords[:host.n_atoms], host.coords)    # the home image first


def test_boxed_host_keeps_the_coordinates_and_refuses_a_small_box():
    pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    at = F.boxed_host(pos, ["C", "C"], 10.0)
    assert np.array_equal(at.get_positions(), pos)
    with pytest.raises(ValueError, match="smaller than the host"):
        F.boxed_host(pos, ["C", "C"], 2.5)


def test_residues_are_atom_types_and_their_names_come_back():
    at = cubic_framework(3.2)
    p = F.framework_pocket(at, [1.6, 1.6, 1.6], radius=4.0)
    assert set(p.type_names.values()) == {"Si6", "O2"}
    assert {p.type_names[int(r)] for r in p.resids} == {"Si6", "O2"}
    assert set(p.resnames) == {F.FRAMEWORK_RESNAME}
    assert p.source_index.shape == (p.n_atoms,) and p.source_image.shape == (p.n_atoms, 3)


def test_framework_charges_come_from_the_periodic_structure():
    at = cubic_framework(3.2)
    fs = fieldset()
    p = F.framework_pocket(at, [1.6, 1.6, 1.6], radius=3.5)
    fields = F.framework_pocket_fields(p, fs, atoms=at, charges="electronegativity")
    from cmfqsar.periodic import crystal_charges

    q, _ = crystal_charges(at, "electronegativity")
    e = [s.name for s in fs.specs].index("electrostatic")
    assert fields.n_atoms == p.n_atoms
    assert np.allclose(fields.weights[:, e], q[p.source_index])
    with pytest.raises(ValueError, match="needs the periodic structure"):
        F.framework_pocket_fields(p, fs, charges="oxidation")


def test_the_type_profile_adds_up_to_the_complementarity_terms():
    host = host_from_molecule(molecule("OCC(O)CO"))
    at = F.boxed_host(host.coords, host.elements, 30.0)
    centre = host.coords.mean(axis=0)
    p = F.framework_pocket(at, centre, radius=10.0, types=["T"] * host.n_atoms)
    fs = fieldset()
    fields = pocket_fields(p, fs)
    g = build_fields(molecule("CO", seed=1), fs, "methanol")
    g.coords = g.coords - g.coords.mean(axis=0) + centre + np.array([0.0, 0.0, 3.0])
    labels, order = pocket_residues(p, fields)
    prof = interaction_profile(g, fields, fs, labels, order)
    by_type = F.type_profile(prof, p)
    assert list(by_type) == ["T"]
    c = complementarity(g, fields, fs)
    for k, v in c.terms.items():
        if np.isfinite(v):
            assert by_type["T"][k] == pytest.approx(v, abs=1e-12)


def test_a_file_that_is_not_periodic_in_three_directions_is_refused(tmp_path):
    from ase.io import write

    path = str(tmp_path / "slab.xyz")
    write(path, Atoms("C", positions=[[0, 0, 0]]))
    with pytest.raises(ValueError, match="periodic in all three"):
        F.read_framework(path)
