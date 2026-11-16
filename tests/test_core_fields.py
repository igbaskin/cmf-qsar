"""Plan item NT6: fields for the core of a particle.

Three per-atom fields of where an atom sits, computed from the positions: the coordination
number, the generalized coordination number and the exposure (``prop:coordination``,
``prop:generalized_coordination``, ``prop:exposure``).
"""
import numpy as np
import pytest

ase = pytest.importorskip("ase")
pytest.importorskip("scipy")
from ase.build import bulk, fcc100, fcc111  # noqa: E402
from ase.cluster import Icosahedron, Octahedron  # noqa: E402

from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.periodic import (CORE_FIELDS, coordination_numbers, core_columns,  # noqa: E402
                              crystal_fields, crystal_table, generalized_coordination,
                              inorganic_fieldset)

PHYSCHEM = ("steric", "prop:electronegativity")


def _open(a):
    a.cell[:] = 0.0
    a.pbc = False
    return a


def test_the_icosahedron_of_55_atoms_has_vertices_edges_and_a_full_inside():
    cn = coordination_numbers(_open(Icosahedron("Cu", 3)))
    assert np.bincount(cn.astype(int)).tolist()[6::2] == [12, 30, 0, 13]


def test_the_generalized_coordination_of_a_vertex_counts_its_neighbours_neighbours():
    a = _open(Icosahedron("Cu", 3))
    g = generalized_coordination(a)
    cn = coordination_numbers(a)
    # a vertex: five edge atoms of 8 neighbours and the centre-shell atom below it of 12
    assert np.allclose(g[cn == 6], (5 * 8 + 12) / 12.0)


@pytest.mark.parametrize("build, value", [(fcc111, 7.5), (fcc100, 20.0 / 3.0)])
def test_terrace_atoms_have_the_published_generalized_coordination(build, value):
    slab = build("Pt", (3, 3, 5), vacuum=10.0)
    z = slab.get_positions()[:, 2]
    assert np.allclose(generalized_coordination(slab)[z > z.max() - 0.1], value)


def test_an_atom_of_a_crystal_counts_its_periodic_images():
    one = bulk("Au", "fcc", a=4.08)              # one atom in the cell
    assert coordination_numbers(one).tolist() == [12.0]
    assert generalized_coordination(one).tolist() == [12.0]
    assert coordination_numbers(bulk("Si", "diamond", a=5.43)).tolist() == [4.0, 4.0]


def test_an_ionic_crystal_counts_cations_closer_than_their_covalent_radii():
    # the documented limit of the rule: magnesium has 6 oxygen and 12 magnesium neighbours
    assert coordination_numbers(bulk("MgO", "rocksalt", a=4.21)).tolist() == [18.0, 6.0]
    assert coordination_numbers(bulk("NaCl", "rocksalt", a=5.64)).tolist() == [6.0, 6.0]


def test_exposed_atoms_are_the_atoms_with_fewer_than_twelve_neighbours():
    a = _open(Octahedron("Cu", 6, cutoff=1))
    cols = core_columns(a, CORE_FIELDS)
    assert np.array_equal(cols["prop:exposure"] > 0, cols["prop:coordination"] < 12)
    assert np.all((cols["prop:exposure"] >= 0) & (cols["prop:exposure"] <= 1))


def test_the_fields_do_not_change_when_the_particle_is_turned_and_moved():
    a = _open(Icosahedron("Pd", 4))
    b = a.copy()
    b.rotate(37, (1, 2, 3), center=(0, 0, 0))
    b.translate((3.1, -7.2, 0.4))
    ca, cb = core_columns(a, CORE_FIELDS), core_columns(b, CORE_FIELDS)
    for name in ("prop:coordination", "prop:generalized_coordination"):
        assert np.array_equal(ca[name], cb[name])
    # the 200 probe points do not turn with the particle, so the exposure moves by a few of them
    assert np.max(np.abs(ca["prop:exposure"] - cb["prop:exposure"])) <= 0.05


def test_a_slab_wrapped_across_its_cell_has_the_same_coordination():
    slab = fcc111("Cu", (3, 3, 4), vacuum=8.0)
    moved = slab.copy()
    moved.translate((1.3, 0.9, 0.0))
    moved.wrap()
    assert np.array_equal(coordination_numbers(slab), coordination_numbers(moved))


def test_the_field_set_adds_the_core_fields_once_after_the_physchem_ones():
    fs = inorganic_fieldset(["Au"], physchem=PHYSCHEM + ("prop:exposure",), core=True)
    names = [sp.name for sp in fs.specs]
    assert names == ["steric", "prop:electronegativity", "prop:exposure", "prop:coordination",
                     "prop:generalized_coordination", "cif:Au", "cif:other"]
    assert [sp.name for sp in inorganic_fieldset(["Au"]).specs] == [
        "electrostatic", "steric", "prop:electronegativity", "cif:Au", "cif:other"]


def test_the_crystal_fields_carry_the_computed_columns():
    a = _open(Icosahedron("Cu", 3))
    fs = inorganic_fieldset(["Cu"], physchem=PHYSCHEM, core=True)
    c = crystal_fields(a, fs, "ico")
    names = [sp.name for sp in fs.specs]
    cols = core_columns(a, CORE_FIELDS)
    for name in CORE_FIELDS:
        assert np.array_equal(c.weights[:, names.index(name)], cols[name])


def test_a_shared_site_takes_the_value_of_its_position_times_the_occupancy():
    cell = bulk("Cu", "fcc", a=3.7, cubic=True)
    full = [{"Cu": 0.5, "Au": 0.5}] * len(cell)
    half = [{"Cu": 0.25, "Au": 0.25}] * len(cell)
    fs = inorganic_fieldset(["Cu", "Au"], physchem=("prop:coordination",))
    a = cell.copy()
    a.info["site_species"] = full
    c = crystal_fields(a, fs)
    assert np.allclose(c.weights[:, 0], 12.0)
    assert np.allclose(c.second_moments[:, 0, 0], 144.0)
    b = cell.copy()
    b.info["site_species"] = half
    assert np.allclose(crystal_fields(b, fs).weights[:, 0], 6.0)


def test_a_model_fits_with_the_core_fields():
    structs, y = [], []
    for el in ("Cu", "Ag", "Au"):
        for k in (2, 3, 4):
            structs.append(_open(Icosahedron(el, k)))
            y.append(k + 0.1 * len(el))
    fs = inorganic_fieldset(["Cu", "Ag", "Au"], physchem=PHYSCHEM, core=True)
    table = crystal_table(structs, fs, [f"p{i}" for i in range(len(structs))])
    m = CMFModel(fieldset=fs, method="krr", kernel_source="periodic", ac_d_max=6.0,
                 normalize=True)
    m.fit(table, np.asarray(y))
    assert np.all(np.isfinite(m.predict(table)))
