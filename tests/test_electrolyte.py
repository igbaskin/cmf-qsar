"""Tests for cmfqsar/electrolyte.py: a bulk electrolyte box, for plan item BT5.

Nothing here runs a simulation. What is tested is the part that decides what goes into the box
and in what order, because that is where a mistake is silent: a box with the wrong number of ions
still runs and still gives shells, and a topology assembled in a different order from the packed
coordinates gives a molecule made of the wrong atoms.

`test_the_counts_reproduce_the_molality` and `test_a_mixture_keeps_its_weight_fractions` are the
two that fail if the composition arithmetic is wrong, and
`test_the_order_of_the_species_is_the_same_everywhere` is the one that fails if the packing and
the topology could disagree.
"""
import os
import sys

import pytest
from rdkit import Chem

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import electrolyte as EL                                        # noqa: E402


# ---------------------------------------------------------------------------- what is refused


def test_a_borate_salt_is_refused_by_name():
    """GAFF has no boron, so LiBF4 and LiBOB cannot be approximated into the series."""
    with pytest.raises(EL.ElectrolyteError) as err:
        EL.Formulation(salt="LiBF4", solvents={"PC": 1.0}, molality=1.0)
    assert "boron" in str(err.value)


def test_an_unknown_solvent_is_refused_by_name():
    with pytest.raises(EL.ElectrolyteError) as err:
        EL.Formulation(salt="LiPF6", solvents={"water": 1.0}, molality=1.0)
    assert "water" in str(err.value)


def test_a_formulation_needs_a_solvent_and_a_molality():
    with pytest.raises(EL.ElectrolyteError):
        EL.Formulation(salt="LiPF6", solvents={"PC": 0.0}, molality=1.0)
    with pytest.raises(EL.ElectrolyteError):
        EL.Formulation(salt="LiPF6", solvents={"PC": 1.0}, molality=0.0)


def test_the_weight_fractions_are_normalised():
    form = EL.Formulation(salt="LiPF6", solvents={"EC": 30.0, "DMC": 70.0}, molality=1.0)
    assert form.solvents == pytest.approx({"EC": 0.3, "DMC": 0.7})


def test_an_unknown_smiles_is_refused():
    with pytest.raises(EL.ElectrolyteError):
        EL.molar_mass("this is not a smiles")


# -------------------------------------------------------------------------- the composition


def test_the_molar_mass_counts_the_hydrogens():
    assert EL.molar_mass("COC(=O)OC") == pytest.approx(90.08, abs=0.02)      # DMC
    assert EL.molar_mass("[Li+]") == pytest.approx(6.94, abs=0.02)
    assert EL.molar_mass("F[P-](F)(F)(F)(F)F") == pytest.approx(144.96, abs=0.05)


def test_the_counts_reproduce_the_molality():
    form = EL.Formulation(salt="LiPF6", solvents={"PC": 1.0}, molality=1.0)
    counts = form.counts(n_solvent=250)
    assert counts["PC"] == 250
    assert counts["ION"] == counts["ANION"]
    # 250 PC is 25.52 g, so one mole per kilogram is 25.5 ion pairs
    assert counts["ION"] == 26
    assert form.realised_molality(counts) == pytest.approx(1.0, abs=0.03)


def test_the_solvent_count_is_the_one_asked_for():
    form = EL.Formulation(salt="LiPF6", solvents={"EC": 0.3, "DMC": 0.4, "DEC": 0.3},
                          molality=1.0)
    counts = form.counts(n_solvent=211)
    assert sum(counts[s] for s in form.solvents) == 211


def test_a_mixture_keeps_its_weight_fractions():
    """Equal weights of two solvents means a mole ratio of their molar masses, reversed."""
    form = EL.Formulation(salt="LiPF6", solvents={"DMC": 0.5, "DEC": 0.5}, molality=1.0)
    counts = form.counts(n_solvent=400)
    ratio = counts["DMC"] / counts["DEC"]
    want = EL.molar_mass(EL.SOLVENTS["DEC"][0]) / EL.molar_mass(EL.SOLVENTS["DMC"][0])
    assert ratio == pytest.approx(want, rel=0.02)
    mass_dmc = counts["DMC"] * EL.molar_mass(EL.SOLVENTS["DMC"][0])
    mass_dec = counts["DEC"] * EL.molar_mass(EL.SOLVENTS["DEC"][0])
    assert mass_dmc / (mass_dmc + mass_dec) == pytest.approx(0.5, abs=0.01)


def test_a_dilute_electrolyte_still_gets_enough_ions():
    """At a low molality the rounded ion count would be one or two, which samples nothing."""
    form = EL.Formulation(salt="LiPF6", solvents={"PC": 1.0}, molality=0.05)
    counts = form.counts(n_solvent=100, min_ions=4)
    assert counts["ION"] == 4
    assert form.realised_molality(counts) > form.molality


def test_the_realised_molality_is_reported_and_differs_by_rounding():
    form = EL.Formulation(salt="LiPF6", solvents={"PC": 1.0}, molality=0.282)
    counts = form.counts(n_solvent=250)
    got = form.realised_molality(counts)
    assert got != form.molality
    assert got == pytest.approx(form.molality, rel=0.1)


# ------------------------------------------------------------------------------- the box


def test_the_starting_density_is_the_ideal_mixture():
    form = EL.Formulation(salt="LiPF6", solvents={"PC": 1.0}, molality=1.0)
    assert form.start_density() == pytest.approx(EL.SOLVENTS["PC"][1])
    mix = EL.Formulation(salt="LiPF6", solvents={"DMC": 0.5, "DEC": 0.5}, molality=1.0)
    want = 1.0 / (0.5 / EL.SOLVENTS["DMC"][1] + 0.5 / EL.SOLVENTS["DEC"][1])
    assert mix.start_density() == pytest.approx(want)


def test_the_box_side_follows_the_mass_and_the_density():
    form = EL.Formulation(salt="LiPF6", solvents={"PC": 1.0}, molality=1.0)
    counts = form.counts(n_solvent=250)
    side = form.box_side(counts, loosen=1.0)
    volume = form.mass(counts) / EL.AVOGADRO / form.start_density() * 1e24
    assert side ** 3 == pytest.approx(volume, rel=1e-6)
    assert form.box_side(counts, loosen=1.1) == pytest.approx(side * 1.1)
    assert 20.0 < side < 60.0, "a box of 250 solvent molecules is a few tens of angstrom"


def test_the_order_of_the_species_is_the_same_everywhere():
    """The packing and the topology are matched atom for atom, so one order serves both."""
    form = EL.Formulation(salt="LiPF6", solvents={"EC": 0.3, "DMC": 0.7}, molality=1.0)
    counts = form.counts(n_solvent=100)
    order = EL.species_order(form, counts)
    assert [name for name, _ in order] == ["DMC", "EC", "ANI", "LI"]
    assert dict(order)["LI"] == counts["ION"]
    assert dict(order)["ANI"] == counts["ANION"]
    assert dict(order)["EC"] == counts["EC"]


def test_the_label_is_usable_as_a_file_name():
    form = EL.Formulation(salt="LiPF6", solvents={"EC": 0.3, "DMC": 0.7}, molality=1.25)
    label = form.label()
    assert label == "LiPF6_DMC70_EC30_m1p250"
    assert "." not in label and " " not in label and "/" not in label


# --------------------------------------------------------------------------- the templates


def test_a_template_has_a_conformation_and_its_residue_name():
    mol = EL.template(EL.SOLVENTS["DMC"][0], "DMC")
    assert mol.GetNumConformers() == 1
    info = mol.GetAtomWithIdx(0).GetMonomerInfo()
    assert info is not None and info.GetResidueName().strip() == "DMC"
    assert mol.GetNumAtoms() == 12                         # C3H6O3 with hydrogens


def test_a_template_is_the_same_molecule_every_time():
    """The atom order is the one the box is packed in, so it must not move between calls."""
    a = EL.template(EL.SOLVENTS["EC"][0], "EC")
    b = EL.template(EL.SOLVENTS["EC"][0], "EC")
    assert [x.GetSymbol() for x in a.GetAtoms()] == [x.GetSymbol() for x in b.GetAtoms()]
    pa = a.GetConformer().GetPositions()
    pb = b.GetConformer().GetPositions()
    assert pa.shape == pb.shape
    assert float(abs(pa - pb).max()) < 1e-6


def test_the_anion_template_carries_its_charge():
    form = EL.Formulation(salt="LiPF6", solvents={"PC": 1.0}, molality=1.0)
    tmpl = EL.templates(form)
    assert set(tmpl) == {"PC", "ANI"}
    assert Chem.GetFormalCharge(tmpl["ANI"]) == -1
    assert Chem.GetFormalCharge(tmpl["PC"]) == 0


def test_the_shell_templates_are_the_packing_templates():
    form = EL.Formulation(salt="LiPF6", solvents={"EC": 0.5, "DMC": 0.5}, molality=1.0)
    tmpl = EL.shell_templates(form)
    assert set(tmpl) == {"EC", "DMC", "ANI"}
    for name, mol in tmpl.items():
        assert mol.GetNumConformers() == 1


def test_the_ion_file_holds_one_atom_of_the_right_element(tmp_path):
    path = str(tmp_path / "li.pdb")
    EL.ion_pdb(path, "Li")
    text = open(path, encoding="ascii").read()
    assert text.count("HETATM") == 1
    mol = Chem.MolFromPDBFile(path, removeHs=False, sanitize=False)
    assert mol is not None and mol.GetNumAtoms() == 1
    assert mol.GetAtomWithIdx(0).GetSymbol() == "Li"


# ------------------------------------------------------------------------- what is installed


def test_the_report_of_what_is_installed_matches_what_require_does():
    state = EL.available()
    assert set(state) >= {"packmol", "antechamber", "openmm", "openff", "nagl"}
    if all(state.values()):
        EL.require()
    else:
        with pytest.raises(EL.ElectrolyteError) as err:
            EL.require()
        for name, ok in state.items():
            if not ok:
                assert name in str(err.value)


def test_putting_the_tools_on_the_path_does_not_fail_without_them():
    found = EL.tools_on_path()
    assert isinstance(found, list)
    assert all(t in ("antechamber", "packmol") for t in found)
