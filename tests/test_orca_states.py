"""The oxidation state, the spin state and the solvent of the quantum stage (plan item EC1).

Nothing here runs ORCA. What is tested is the part that decides what ORCA is asked to do and the
part that reads what it answered, because both fail silently: a wrong charge or spin state gives a
plausible number for the wrong state, and a spin population read from the wrong column gives a
plausible field for the wrong quantity.
"""
import numpy as np
import pytest
from rdkit import Chem

from cmfqsar.cdft import (PLOT_DENSITY, PLOT_POTENTIAL, PLOT_SPIN, _PLOT_KEYS,
                          _PLOT_KEYS_POTENTIAL, _PLOT_SUFFIX, CDFT_FIELDS)
from cmfqsar.orca import (DEFAULT_LEVEL, ECP_FROM_Z, HEAVY_LEVEL, OrcaError, SOLVENT_CONSTANTS,
                          _parse_charges, _parse_spin, cache_key, derive_properties,
                          electron_count, ion_multiplicity, make_input, multiplicity_for,
                          needs_ecp, parse_orca_output, solvent_lines, state_from_properties)

PHENAZINE = "c1ccc2nc3ccccc3nc2c1"


def mol(smiles=PHENAZINE, **props):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    for k, v in props.items():
        m.SetProp(k, str(v))
    return m


# ------------------------------------------------------------------ the state of a molecule

def test_without_properties_the_state_is_the_formal_charge_and_the_lowest_spin():
    m = mol()
    assert state_from_properties(m) == (0, 1)


def test_a_charge_and_a_multiplicity_in_the_record_are_used():
    m = mol(charge=-1, multiplicity=2)
    assert state_from_properties(m) == (-1, 2)


def test_a_capitalised_property_name_is_read_too():
    m = mol(Charge=1, Multiplicity=2)
    assert state_from_properties(m) == (1, 2)


def test_a_quintet_iron_complex_keeps_its_spin_state():
    # the case the item names: nothing in a connection table says Fe(II) is a quintet
    m = mol("[Fe]", charge=2, multiplicity=5)
    assert state_from_properties(m) == (2, 5)


def test_a_multiplicity_of_the_wrong_parity_is_refused():
    m = mol(multiplicity=2)              # phenazine has an even number of electrons
    with pytest.raises(OrcaError, match="does not fit"):
        state_from_properties(m)


def test_a_multiplicity_below_one_is_refused():
    m = mol(multiplicity=0)
    with pytest.raises(OrcaError, match="not a spin state"):
        state_from_properties(m)


def test_a_charge_in_the_record_overrides_the_formal_charge():
    m = mol("[NH4+]")
    assert Chem.GetFormalCharge(m) == 1
    m.SetProp("charge", "0")
    # 0 is refused for ammonium only if the parity disagrees; here it is read as given
    charge, _ = state_from_properties(m)
    assert charge == 0


def test_the_electron_count_follows_the_charge():
    m = mol()
    assert electron_count(m, 0) - electron_count(m, -1) == -1


# ------------------------------------------------------------------ the ionic states

def test_a_closed_shell_parent_gives_the_doublet_the_fukui_indices_use():
    assert ion_multiplicity(1) == 2


def test_an_open_shell_parent_pairs_the_electron_where_it_can():
    assert ion_multiplicity(2) == 1
    assert ion_multiplicity(3) == 2
    assert ion_multiplicity(5) == 4


def test_the_ion_of_a_closed_shell_molecule_is_what_it_always_was():
    m = mol()
    assert ion_multiplicity(multiplicity_for(m, 0)) == multiplicity_for(m, -1)


# ------------------------------------------------------------------ the solvent

def test_the_gas_phase_adds_nothing():
    assert solvent_lines(None) == ([], [])
    assert "CPCM" not in make_input(DEFAULT_LEVEL, 0, 1, "g.xyz")


def test_a_solvent_orca_names_becomes_a_keyword():
    keywords, block = solvent_lines("water")
    assert keywords == ["CPCM(water)"]
    assert block == []
    assert "CPCM(water)" in make_input(DEFAULT_LEVEL, 0, 1, "g.xyz", solvent="water")


def test_a_solvent_orca_does_not_name_becomes_a_block_with_its_constants():
    keywords, block = solvent_lines("DME")
    assert keywords == ["CPCM"]
    assert "%cpcm" in block
    assert any("epsilon 7.2" in ln for ln in block)
    assert any("refrac" in ln for ln in block)


def test_the_two_constants_may_be_given_directly():
    _, block = solvent_lines((36.6, 1.3441))
    assert any("epsilon 36.6" in ln for ln in block)


def test_the_solvent_name_is_matched_whatever_its_case():
    assert solvent_lines("dme") == solvent_lines("DME")
    assert "dme" in SOLVENT_CONSTANTS


def test_an_optimisation_in_a_solvent_still_prints_nothing_extra():
    text = make_input(DEFAULT_LEVEL, 0, 1, "g.xyz", opt=True, solvent="water")
    assert "Opt" in text and "CPCM(water)" in text
    assert "Print[P_Hirshfeld]" not in text and "CHELPG" not in text


# ------------------------------------------------------------------ the basis and the ECP

def test_a_light_molecule_needs_no_core_potential():
    assert not needs_ecp(mol())


def test_a_heavy_element_needs_one():
    assert needs_ecp(Chem.MolFromSmiles("Cl[Ru](Cl)Cl"))
    assert needs_ecp(Chem.MolFromSmiles("[Ir]"))


def test_the_threshold_is_rubidium():
    assert needs_ecp(Chem.MolFromSmiles("[Rb]"))
    assert not needs_ecp(Chem.MolFromSmiles("[Kr]"))
    assert ECP_FROM_Z == 37


def test_the_heavy_level_names_a_def2_basis_and_its_core_potential():
    assert "def2-TZVP" in HEAVY_LEVEL and "def2-ECP" in HEAVY_LEVEL


# ------------------------------------------------------------------ the cache key

def test_a_gas_phase_run_at_the_lowest_spin_keeps_the_key_it_had():
    # the ORCA results already cached for the other benchmarks must not be thrown away
    plain = cache_key("c1ccccc1", 0, DEFAULT_LEVEL, False)
    explicit = cache_key("c1ccccc1", 0, DEFAULT_LEVEL, False, solvent=None, mult=None)
    assert plain == explicit


def test_a_solvent_changes_the_key():
    plain = cache_key("c1ccccc1", 0, DEFAULT_LEVEL, False)
    assert cache_key("c1ccccc1", 0, DEFAULT_LEVEL, False, solvent="water") != plain


def test_two_solvents_have_different_keys():
    a = cache_key("c1ccccc1", 0, DEFAULT_LEVEL, False, solvent="water")
    b = cache_key("c1ccccc1", 0, DEFAULT_LEVEL, False, solvent="DME")
    assert a != b


def test_a_spin_state_changes_the_key():
    plain = cache_key("[Fe]", 2, DEFAULT_LEVEL, False)
    assert cache_key("[Fe]", 2, DEFAULT_LEVEL, False, mult=5) != plain
    assert (cache_key("[Fe]", 2, DEFAULT_LEVEL, False, mult=5)
            != cache_key("[Fe]", 2, DEFAULT_LEVEL, False, mult=3))


# ------------------------------------------------------------------ reading the spin populations

CLOSED = """
MULLIKEN ATOMIC CHARGES
-----------------------
   0 C :   -0.123456
   1 N :    0.234567
Sum of atomic charges:    0.1111
"""

OPEN = """
MULLIKEN ATOMIC CHARGES AND SPIN POPULATIONS
--------------------------------------------
   0 C :   -0.223456    0.345678
   1 N :    0.334567   -0.045678
Sum of atomic charges         :   0.1111
Sum of atomic spin populations:   0.3000
"""


def test_a_closed_shell_block_has_charges_and_no_spin():
    lines = CLOSED.splitlines()
    assert _parse_charges(lines, "MULLIKEN ATOMIC CHARGES") == pytest.approx([-0.123456, 0.234567])
    assert _parse_spin(lines, "MULLIKEN ATOMIC CHARGES") is None


def test_an_open_shell_block_gives_both_columns():
    lines = OPEN.splitlines()
    assert _parse_charges(lines, "MULLIKEN ATOMIC CHARGES") == pytest.approx([-0.223456, 0.334567])
    assert _parse_spin(lines, "MULLIKEN ATOMIC CHARGES") == pytest.approx([0.345678, -0.045678])


def test_the_spin_is_not_the_charge():
    lines = OPEN.splitlines()
    q = _parse_charges(lines, "MULLIKEN ATOMIC CHARGES")
    s = _parse_spin(lines, "MULLIKEN ATOMIC CHARGES")
    assert not np.allclose(q, s)


def test_a_missing_block_is_not_an_error():
    assert _parse_spin("nothing here".splitlines(), "MULLIKEN ATOMIC CHARGES") is None


def test_the_output_as_a_whole_carries_the_spin_only_when_it_is_open_shell():
    assert parse_orca_output(CLOSED).spin is None
    assert parse_orca_output(OPEN).spin == pytest.approx([0.345678, -0.045678])


# ------------------------------------------------------------------ the spin reaches the fields

def _result(charges, spin=None):
    r = parse_orca_output("")
    r.mulliken = np.asarray(charges, float)
    r.hirshfeld = np.asarray(charges, float)
    r.energy = -1.0
    r.spin = None if spin is None else np.asarray(spin, float)
    return r


def test_an_open_shell_parent_gets_a_spin_field():
    n = _result([0.1, -0.1], spin=[0.9, 0.1])
    atom, molprops = derive_properties(n, charge=0, mult=2)
    assert atom["spin"] == pytest.approx([0.9, 0.1])
    assert molprops["spin_total"] == pytest.approx(1.0)


def test_a_closed_shell_parent_gets_no_spin_field():
    atom, molprops = derive_properties(_result([0.1, -0.1]), charge=0, mult=1)
    assert "spin" not in atom
    assert "spin_total" not in molprops


def test_both_radical_ions_give_their_own_spin_field():
    n = _result([0.1, -0.1])
    anion = _result([-0.4, -0.6], spin=[0.7, 0.3])
    cation = _result([0.6, 0.4], spin=[0.2, 0.8])
    atom, molprops = derive_properties(n, anion, cation, charge=0, mult=1)
    assert atom["spin_anion"] == pytest.approx([0.7, 0.3])
    assert atom["spin_cation"] == pytest.approx([0.2, 0.8])
    assert molprops["spin_total_anion"] == pytest.approx(1.0)
    assert molprops["spin_total_cation"] == pytest.approx(1.0)


def test_the_spin_field_is_per_atom_and_as_long_as_the_molecule():
    n = _result([0.1, -0.1])
    anion = _result([-0.4, -0.6], spin=[0.7, 0.3])
    atom, _ = derive_properties(n, anion, _result([0.6, 0.4]), charge=0, mult=1)
    assert len(atom["spin_anion"]) == len(n.hirshfeld)


# ------------------------------------------------------------------ the potential cube

def test_the_potential_is_its_own_plot_type():
    assert PLOT_POTENTIAL == 43
    assert PLOT_POTENTIAL not in (PLOT_DENSITY, PLOT_SPIN)


def test_every_plot_type_has_a_file_suffix():
    for kind in (PLOT_DENSITY, PLOT_SPIN, PLOT_POTENTIAL):
        assert _PLOT_SUFFIX[kind]


def test_the_potential_asks_for_a_density_by_name_and_the_others_accept_one():
    keys = _PLOT_KEYS_POTENTIAL.format(kind=PLOT_POTENTIAL, intervals=40, density="job.scfp")
    assert "job.scfp" in keys
    # the density and spin plots answer a yes/no question instead
    assert "\ny\n" in _PLOT_KEYS.format(kind=PLOT_DENSITY, intervals=40)
    assert "\ny\n" not in keys


def test_the_potential_is_one_of_the_fields_this_module_offers():
    assert "potential" in CDFT_FIELDS
