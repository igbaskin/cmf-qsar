"""Plan items BT1 to BT3: charges of a part-lithiated cell, crystal descriptors, insertion pairs.

Nothing here touches the network.  The insertion pairs are built from structures made by hand
with the energies they would have carried, so that the arithmetic of a voltage is tested without
a service being up.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar import battery, descriptors                                   # noqa: E402
from cmfqsar.periodic import (INORGANIC_FIELDS, composition_of, crystal_charges,  # noqa: E402
                              crystal_table, inorganic_fieldset, oxidation_states)

F_PER_MOLE = 26801.5           # mAh per mole of charge, which is Faraday's constant over 3.6


# --------------------------------------------------------------------------- #
# structures
# --------------------------------------------------------------------------- #
def rock_salt(symbols, a=4.2):
    """A cubic cell with the sites of rock salt, filled with the symbols given (8 sites)."""
    from ase import Atoms
    sites = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5),
             (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5)]
    if len(symbols) != len(sites):
        raise AssertionError("eight symbols for eight sites")
    return Atoms(symbols=list(symbols), scaled_positions=sites, cell=[a, a, a], pbc=True)


def layered_lixcoo2(n_li):
    """A cell of Li(x)CoO2 with four cobalt sites and `n_li` of the four lithium sites filled."""
    from ase import Atoms
    a, c = 2.82, 14.05
    symbols, positions = [], []
    for k in range(4):
        z = (k + 0.5) / 4.0
        symbols.append("Co")
        positions.append((0.0, 0.0, z))
        for dz, el in ((0.25 / 4.0, "O"), (-0.25 / 4.0, "O")):
            symbols.append(el)
            positions.append((1.0 / 3.0, 2.0 / 3.0, z + dz))
    for k in range(n_li):
        symbols.append("Li")
        positions.append((2.0 / 3.0, 1.0 / 3.0, k / 4.0))
    return Atoms(symbols=symbols, scaled_positions=positions, cell=[a, a, c], pbc=True)


def with_energy(atoms, formation_per_atom, hull=0.0, ident="x"):
    atoms = atoms.copy()
    atoms.info.update(id=ident, formation_energy_per_atom=float(formation_per_atom),
                      energy_above_hull=float(hull), functional="test")
    return atoms


# --------------------------------------------------------------------------- #
# BT1: the charge model
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_li,co_charge", [(4, 3.0), (3, 3.25), (2, 3.5)])
def test_a_part_lithiated_cell_is_neutral_and_its_metal_takes_the_charge(n_li, co_charge):
    """Li(x)CoO2 for x = 1, 0.75 and 0.5: the cell is neutral and the cobalt carries the loss."""
    atoms = layered_lixcoo2(n_li)
    q, source = crystal_charges(atoms, "oxidation")
    assert source == "oxidation"
    assert abs(float(np.sum(q))) < 1e-6
    symbols = atoms.get_chemical_symbols()
    cobalt = {round(float(v), 6) for v, s in zip(q, symbols) if s == "Co"}
    assert cobalt == {co_charge}


def test_the_cobalt_charge_rises_as_the_lithium_leaves():
    """The point of item BT1: the charge on the metal has to move with the state of charge."""
    charges = []
    for n_li in (4, 3, 2):
        q, _ = crystal_charges(layered_lixcoo2(n_li), "oxidation")
        symbols = layered_lixcoo2(n_li).get_chemical_symbols()
        charges.append(float(np.mean([v for v, s in zip(q, symbols) if s == "Co"])))
    assert charges[0] < charges[1] < charges[2]


def test_a_fractional_composition_balances_on_the_metal():
    """Li(0.5)CoO2 as a composition, which has no assignment in whole oxidation states."""
    states = oxidation_states(["Li", "Co", "O", "O"], composition={"Li": 0.5, "Co": 1.0, "O": 2.0})
    assert states is not None
    assert states["Co"] == pytest.approx(3.5)
    assert states["Li"] == pytest.approx(1.0) and states["O"] == pytest.approx(-2.0)


def test_a_fractional_phosphate_balances_too():
    """Li(0.5)FePO4: the iron is half Fe(II) and half Fe(III), so its average is +2.5."""
    states = oxidation_states(["Li", "Fe", "P", "O"],
                              composition={"Li": 0.5, "Fe": 1.0, "P": 1.0, "O": 4.0})
    assert states["Fe"] == pytest.approx(2.5)
    net = sum(n * states[el] for el, n in {"Li": 0.5, "Fe": 1.0, "P": 1.0, "O": 4.0}.items())
    assert abs(net) < 1e-9


def test_an_assignment_that_does_not_balance_is_refused():
    """Whatever the guess, a non-neutral set of states is not returned."""
    for composition in ({"Li": 0.5, "Co": 1.0, "O": 2.0}, {"Li": 0.75, "Mn": 1.0, "O": 2.0},
                        {"Na": 0.5, "Fe": 1.0, "P": 1.0, "O": 4.0}):
        states = oxidation_states(list(composition), composition=composition)
        if states is None:
            continue
        net = sum(n * states[el] for el, n in composition.items())
        assert abs(net) < 1e-6, (composition, states)


def test_mixed_valence_is_not_put_on_an_anion():
    """A half-occupied lithium chloride does not become Cl(-0.5); the guess refuses instead."""
    assert battery is not None                        # the import is what this file is about
    states = oxidation_states(["Li", "Cl", "Li", "Cl"], composition={"Li": 1.0, "Cl": 2.0})
    assert states is None


def test_the_dft_charge_model_prefers_what_a_calculation_gave():
    atoms = layered_lixcoo2(4)
    bader = np.linspace(-1.0, 1.0, len(atoms))
    atoms.set_array("bader_charges", bader)
    q, source = crystal_charges(atoms, "dft")
    assert source == "bader_charges"
    assert np.allclose(q, bader)


def test_the_dft_charge_model_falls_back_to_the_formal_states():
    q, source = crystal_charges(layered_lixcoo2(4), "dft")
    assert source == "oxidation"
    assert abs(float(np.sum(q))) < 1e-6


# --------------------------------------------------------------------------- #
# BT2: the crystal descriptors
# --------------------------------------------------------------------------- #
def test_the_capacity_of_lithium_metal_is_the_textbook_number():
    from ase.build import bulk
    value = descriptors.compute(bulk("Li", "fcc", a=4.0), "calc:capacity")
    assert value == pytest.approx(3862.0, abs=2.0)    # 3860 mAh/g in every battery textbook


def test_the_capacity_of_lithium_cobalt_oxide_is_the_textbook_number():
    value = descriptors.compute(layered_lixcoo2(4), "calc:capacity")
    assert value == pytest.approx(274.0, abs=2.0)     # 274 mAh/g for one lithium per LiCoO2


def test_the_density_and_the_volume_come_from_the_cell():
    atoms = layered_lixcoo2(4)
    mass = float(np.sum(atoms.get_masses()))
    volume = float(atoms.get_volume())
    assert descriptors.compute(atoms, "calc:density") == pytest.approx(1.66054 * mass / volume)
    assert descriptors.compute(atoms, "calc:volume_per_atom") == pytest.approx(volume / len(atoms))
    assert descriptors.compute(atoms, "calc:volume_per_formula") == pytest.approx(volume / 4.0)


def test_the_lithium_content_counts_per_formula_unit():
    assert descriptors.compute(layered_lixcoo2(4), "calc:ion_count") == pytest.approx(1.0)
    assert descriptors.compute(layered_lixcoo2(2), "calc:ion_count") == pytest.approx(0.5)
    assert descriptors.compute(layered_lixcoo2(2), "calc:ion_fraction") == pytest.approx(2.0 / 14.0)


def test_the_shell_of_the_working_ion_is_measured_through_the_cell_boundary():
    atoms = layered_lixcoo2(4)
    count = descriptors.compute(atoms, "calc:ion_coordination")
    distance = descriptors.compute(atoms, "calc:ion_oxygen_distance")
    assert count > 0 and math.isfinite(distance)
    assert 1.5 < distance < 3.0                       # a lithium-oxygen distance in an oxide


def test_another_working_ion_is_named_after_an_at_sign():
    atoms = layered_lixcoo2(4)
    assert descriptors.compute(atoms, "calc:ion_count@Na") == pytest.approx(0.0)
    assert math.isnan(descriptors.compute(atoms, "calc:ion_coordination@Na"))


def test_an_unknown_crystal_descriptor_says_what_there_is():
    with pytest.raises(KeyError, match="capacity"):
        descriptors.compute(layered_lixcoo2(4), "calc:capacitty")


def test_a_cell_keeps_its_own_numbers_as_descriptors():
    """A number a loader left in atoms.info enters a model like a molecule's property tag."""
    atoms = with_energy(layered_lixcoo2(4), -2.0, hull=0.01)
    values = descriptors.read_values(atoms, ("formation_energy_per_atom", "calc:ion_count",
                                             "no_such_number"))
    assert values[0] == pytest.approx(-2.0)
    assert values[1] == pytest.approx(1.0)
    assert math.isnan(values[2])


def test_the_descriptors_reach_the_table_the_model_is_fitted_on():
    fs = inorganic_fieldset(["Li", "Co", "O"], physchem=INORGANIC_FIELDS,
                            charge_model="electronegativity")
    fs = fs.__class__(**{**fs.__dict__, "descriptors": ("calc:capacity", "calc:density")})
    table = crystal_table([layered_lixcoo2(4), layered_lixcoo2(2)], fs, ["full", "half"])
    Z = table.descriptors
    assert Z is not None and Z.shape == (2, 2)
    assert Z[0, 0] > Z[1, 0]                          # the lithiated cell carries more charge
    assert np.all(np.isfinite(Z))


# --------------------------------------------------------------------------- #
# BT3: pairs and voltages
# --------------------------------------------------------------------------- #
def test_the_host_framework_is_the_cell_without_the_working_ion():
    assert battery.host_framework(layered_lixcoo2(4)) == (("Co", 1), ("O", 2))
    assert battery.host_framework(layered_lixcoo2(0)) == (("Co", 1), ("O", 2))
    assert battery.framework_label((("Co", 1), ("O", 2))) == "CoO2"


def test_the_content_is_counted_per_host_formula_unit():
    assert battery.host_units(layered_lixcoo2(4)) == pytest.approx(4.0)
    assert battery.ion_content(layered_lixcoo2(4)) == pytest.approx(1.0)
    assert battery.ion_content(layered_lixcoo2(1)) == pytest.approx(0.25)


def test_the_voltage_is_the_difference_of_the_formation_energies():
    """Minus the change in formation energy per host unit, divided by the lithium taken in."""
    charged = with_energy(layered_lixcoo2(0), -1.0, ident="charged")        # 12 atoms
    discharged = with_energy(layered_lixcoo2(4), -2.0, ident="discharged")  # 16 atoms
    # per host formula unit: -1.0 * 12 / 4 = -3.0 and -2.0 * 16 / 4 = -8.0; dx = 1
    assert battery.formation_energy_per_host(charged) == pytest.approx(-3.0)
    assert battery.formation_energy_per_host(discharged) == pytest.approx(-8.0)
    assert battery.average_voltage(charged, discharged) == pytest.approx(5.0)


def test_the_voltage_does_not_depend_on_the_size_of_the_cell():
    charged = with_energy(layered_lixcoo2(0), -1.0)
    discharged = with_energy(layered_lixcoo2(4), -2.0)
    big = with_energy(layered_lixcoo2(4).repeat((2, 1, 1)), -2.0)
    assert battery.average_voltage(charged, big) == pytest.approx(
        battery.average_voltage(charged, discharged))


def test_a_ladder_keeps_the_most_stable_cell_at_every_step():
    good = with_energy(layered_lixcoo2(2), -2.0, hull=0.00, ident="good")
    bad = with_energy(layered_lixcoo2(2), -1.0, hull=0.20, ident="bad")
    rungs = battery.ladders([bad, good])
    assert rungs["CoO2"][0.5].info["id"] == "good"


def test_the_pairs_are_the_neighbouring_steps_of_a_ladder():
    # formation energies that make both steps a four-volt one: -3, -5 and -7 eV per host unit
    cells = [with_energy(layered_lixcoo2(0), -3.0 / 3.0, ident="x0"),
             with_energy(layered_lixcoo2(2), -5.0 / 3.5, ident="x2"),
             with_energy(layered_lixcoo2(4), -7.0 / 4.0, ident="x4")]
    pairs = battery.insertion_pairs(cells)
    assert [(p.x_charged, p.x_discharged) for p in pairs] == [(0.0, 0.5), (0.5, 1.0)]
    assert [round(p.voltage, 6) for p in pairs] == [4.0, 4.0]
    assert all(p.framework == "CoO2" for p in pairs)
    assert all(np.isfinite(p.voltage) for p in pairs)
    assert pairs[0].capacity > 0


def test_a_pair_whose_voltage_is_outside_the_window_is_refused():
    """Two phases that share a framework formula and nothing else are not an electrode."""
    cells = [with_energy(layered_lixcoo2(0), -1.0, ident="a"),
             with_energy(layered_lixcoo2(4), 40.0, ident="b")]
    assert battery.insertion_pairs(cells) == []


def test_a_pair_becomes_a_table_of_structures_and_voltages():
    cells = [with_energy(layered_lixcoo2(0), -1.0, ident="x0"),
             with_energy(layered_lixcoo2(2), -5.0 / 3.5, ident="x2"),
             with_energy(layered_lixcoo2(4), -1.75, ident="x4")]
    pairs = battery.insertion_pairs(cells)
    structures, y, names = battery.pair_table(pairs, frame="discharged")
    assert len(structures) == len(y) == len(names) == len(pairs)
    structures, y, names = battery.pair_table(pairs, frame="both")
    assert len(structures) == 2 * len(pairs) and y[0] == y[1]
    with pytest.raises(ValueError, match="discharged"):
        battery.pair_table(pairs, frame="middle")


def test_the_family_of_a_host_is_named():
    from ase import Atoms
    oxide = with_energy(layered_lixcoo2(4), -2.0)
    pair = battery.InsertionPair(framework="CoO2", charged=oxide, discharged=oxide,
                                 x_charged=0.0, x_discharged=1.0, voltage=4.0)
    assert battery.chemistry(pair) == "oxide"
    phosphate = Atoms(symbols=["Li", "Fe", "P", "O", "O", "O", "O"],
                      positions=np.zeros((7, 3)), cell=[6, 6, 6], pbc=True)
    pair = battery.InsertionPair(framework="FeO4P", charged=phosphate, discharged=phosphate,
                                 x_charged=0.0, x_discharged=1.0, voltage=3.4)
    assert battery.chemistry(pair) == "phosphate"


def test_an_optimade_entry_becomes_a_structure_with_its_energies():
    entry = {"id": "mp-1", "attributes": {
        "species_at_sites": ["Li", "O"],
        "species": [{"name": "Li", "chemical_symbols": ["Li"]},
                    {"name": "O", "chemical_symbols": ["O"]}],
        "cartesian_site_positions": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        "lattice_vectors": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        "nsites": 2,
        "_mp_stability": {"gga_gga+u": {"formation_energy_per_atom": -1.5,
                                        "energy_above_hull": 0.02}}}}
    atoms = battery.atoms_from_optimade(entry)
    assert atoms.get_chemical_symbols() == ["Li", "O"]
    assert atoms.info["id"] == "mp-1"
    assert atoms.info["formation_energy_per_atom"] == pytest.approx(-1.5)
    assert atoms.info["energy_above_hull"] == pytest.approx(0.02)
    assert atoms.info["functional"] == "gga_gga+u"


def test_a_disordered_optimade_entry_is_refused_by_name():
    entry = {"id": "mp-2", "attributes": {
        "species_at_sites": ["A"],
        "species": [{"name": "A", "chemical_symbols": ["Li", "Na"], "concentration": [0.5, 0.5]}],
        "cartesian_site_positions": [[0.0, 0.0, 0.0]],
        "lattice_vectors": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        "nsites": 1, "_mp_stability": {}}}
    with pytest.raises(ValueError, match="shares its site"):
        battery.atoms_from_optimade(entry)


def test_the_records_survive_a_round_trip(tmp_path):
    records = [{"id": "mp-1", "attributes": {"nsites": 2}}]
    path = battery.write_records(records, str(tmp_path / "cache" / "sys.json"))
    assert battery.read_records(path) == records


def test_the_electrode_documents_say_what_a_key_is_for(monkeypatch):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    monkeypatch.delenv("PMG_MAPI_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key"):
        battery.insertion_electrodes()


def test_the_folds_keep_a_framework_together():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "benchmarks", "run_battery.py")
    spec = importlib.util.spec_from_file_location("run_battery", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    groups = ["CoO2"] * 3 + ["FeO4P"] * 2 + ["MnO2"] * 4 + ["NiO2"] + ["VO2"] * 2
    folds = module.grouped_folds(groups)
    assert len(folds) == 5
    for tr, te in folds:
        assert not ({groups[i] for i in te} & {groups[i] for i in tr})
    assert sorted(i for _, te in folds for i in te) == list(range(len(groups)))


def test_the_composition_features_hold_the_fractions_and_the_descriptors():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "benchmarks", "run_battery.py")
    spec = importlib.util.spec_from_file_location("run_battery2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    elements = ["Co", "Li", "O"]
    X = module.composition_features([layered_lixcoo2(4), layered_lixcoo2(2)], elements)
    assert X.shape == (2, len(elements) + len(module.CRYSTAL_DESCRIPTORS))
    assert X[0, :3].sum() == pytest.approx(1.0)
    assert X[0, elements.index("Li")] > X[1, elements.index("Li")]
