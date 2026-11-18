"""Polymers as capped oligomers, and the field autocorrelation of one repeat unit.

The tests fix what the builder promises -- a repeat unit needs exactly two attachment points with
one bond order, the units are joined head to tail, every atom knows its unit, the caps close the
open valences -- and what the descriptor promises: with every atom counted it is the program's own
autocorrelation, and the rows of a partition of the atoms add up to it.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from rdkit import Chem, RDLogger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import AutocorrelationSpec, autocorrelation  # noqa: E402
from cmfqsar.polymer import (UNIT_PROP, autocorrelation_rows, build, build_many,  # noqa: E402
                             central_unit, oligomer, parse_repeat_unit, unit_labels)

RDLogger.DisableLog("rdApp.*")
STYRENE = "[*]CC([*])c1ccccc1"


@pytest.fixture(scope="module")
def fs():
    return FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic", "hba", "hbd"],
                           alpha=1.0, charge_model="gasteiger", param_scheme="rdkit")


@pytest.fixture(scope="module")
def trimer():
    mol, info = build(STYRENE, n=3, conformers=4)
    return mol, info


@pytest.mark.parametrize("smiles, why", [
    ("CCc1ccccc1", "0 attachment points"),
    ("[*]CC", "1 attachment points"),
    ("[*]C([*])C[*]", "3 attachment points"),
    ("[*]C=C[*].[Na+]", "more than one fragment"),
    ("[*]=CC[*]", "different order"),
    ("not smiles", "not valid SMILES"),
])
def test_a_repeat_unit_needs_two_attachment_points_of_one_bond_order(smiles, why):
    with pytest.raises(ValueError, match=why):
        parse_repeat_unit(smiles)


def test_the_units_are_joined_head_to_tail_and_capped():
    # polystyrene: the head is the CH2, the tail the CH(Ph); a trimer capped with hydrogen is
    # CH3-CH(Ph)-CH2-CH(Ph)-CH2-CH2Ph
    m = oligomer(STYRENE, n=3, stereo="random", add_hydrogens=False)
    ref = Chem.MolFromSmiles("CC(CC(CCc1ccccc1)c1ccccc1)c1ccccc1")
    assert Chem.MolToSmiles(Chem.RemoveHs(m)) == Chem.MolToSmiles(ref)
    assert not any(a.GetAtomicNum() == 0 for a in m.GetAtoms())


def test_every_atom_knows_its_unit_and_the_caps_are_minus_one():
    m = oligomer(STYRENE, n=3)
    labels = unit_labels(m)
    assert sorted(set(labels.tolist())) == [-1, 0, 1, 2]
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    for k in (0, 1, 2):
        assert sum(labels[i] == k for i in heavy) == 8          # C8 per styrene unit
    assert int(np.sum(labels == -1)) == 2                        # two hydrogen caps
    for a in m.GetAtoms():                                       # a hydrogen joins its atom's unit
        if a.GetAtomicNum() == 1 and labels[a.GetIdx()] >= 0:
            assert labels[a.GetIdx()] == labels[a.GetNeighbors()[0].GetIdx()]


@pytest.mark.parametrize("smiles, cap", [("[*]#CC#[*]", "C#C"), ("[*]=CC=[*]", "C=C")])
def test_a_multiple_bond_at_the_end_gets_a_carbon_cap(smiles, cap):
    m = oligomer(smiles, n=2, add_hydrogens=False)
    assert Chem.SanitizeMol(m) == Chem.SanitizeFlags.SANITIZE_NONE
    assert m.HasSubstructMatch(Chem.MolFromSmarts(cap))


def test_isotactic_copies_one_configuration_and_random_leaves_it_open():
    iso = oligomer("[*]CC([*])C(=O)OC", n=3, add_hydrogens=False)
    tags = [a.GetChiralTag() for a in iso.GetAtoms() if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED]
    assert len(tags) >= 2 and len(set(tags)) == 1
    rnd = oligomer("[*]CC([*])C(=O)OC", n=3, stereo="random", add_hydrogens=False)
    assert all(a.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for a in rnd.GetAtoms())


def test_the_given_configuration_is_kept():
    # the two hands of one repeat unit give the two mirror-image oligomers
    left = Chem.MolToSmiles(Chem.RemoveHs(oligomer("[*]C[C@H]([*])Cl", n=3, add_hydrogens=False)))
    right = Chem.MolToSmiles(Chem.RemoveHs(oligomer("[*]C[C@@H]([*])Cl", n=3, add_hydrogens=False)))
    assert left != right and left.count("@") >= 2
    mirror = left.replace("@@", "!").replace("@", "@@").replace("!", "@")
    assert Chem.CanonSmiles(mirror) == Chem.CanonSmiles(right)


def test_the_embedded_chain_is_the_most_extended_and_optimised(trimer):
    mol, info = trimer
    assert mol.GetNumConformers() == 1
    assert info["force_field"] == "MMFF94"
    assert info["units"] == 3 and info["central_atoms"] == 16   # C8H8 in the middle unit
    assert 4.0 < info["span"] < 7.0                              # five C-C bonds, not folded back


def test_the_middle_unit_needs_an_odd_count():
    m = oligomer(STYRENE, n=2)
    with pytest.raises(ValueError, match="no middle unit"):
        central_unit(m)
    assert int(central_unit(oligomer(STYRENE, n=3)).sum()) == 16


@pytest.mark.parametrize("sigma", [0.05, 0.5, 6.0])
def test_with_every_atom_the_rows_are_the_programs_autocorrelation(trimer, fs, sigma):
    mf = build_fields(trimer[0], fs, "ps")
    spec = AutocorrelationSpec(d_max=12.0, step=sigma, sigma=sigma)
    dense = autocorrelation(mf, spec)
    sparse = autocorrelation_rows(mf, [spec])[0]
    assert np.allclose(sparse, dense, rtol=1e-10, atol=1e-12 * np.abs(dense).max())


def test_the_rows_of_a_partition_add_up_to_the_whole(trimer, fs):
    mol = trimer[0]
    mf = build_fields(mol, fs, "ps")
    specs = [AutocorrelationSpec(d_max=12.0, step=s, sigma=s) for s in (0.1, 1.0)]
    whole = autocorrelation_rows(mf, specs)
    labels = unit_labels(mol)
    parts = [autocorrelation_rows(mf, specs, centre=labels == k) for k in sorted(set(labels))]
    for j in range(len(specs)):
        assert np.allclose(sum(p[j] for p in parts), whole[j], rtol=1e-10)


def test_the_central_row_does_not_see_the_caps(fs):
    # the same middle unit with different caps two units away gives the same central row as long
    # as the caps are beyond the reach of the channels
    mol, _ = build("[*]CC[*]", n=3, conformers=4)
    mf = build_fields(mol, fs, "pe")
    spec = AutocorrelationSpec(d_max=1.2, step=0.1, sigma=0.1)
    centre = central_unit(mol)
    row = autocorrelation_rows(mf, [spec], centre=centre)[0]
    caps = unit_labels(mol) == -1
    far = mf.coords[caps]
    near = mf.coords[centre]
    gap = np.min(np.linalg.norm(far[:, None] - near[None], axis=-1))
    assert gap > spec.d_max + 8 * spec.sigma - 1e-9 or gap > 2.0
    W = mf.weights.copy()
    W[caps] *= 3.0                       # change what the caps carry
    mf2 = type(mf)(**{**mf.__dict__, "weights": W})
    row2 = autocorrelation_rows(mf2, [spec], centre=centre)[0]
    if gap > spec.d_max + 8 * spec.sigma:
        assert np.allclose(row, row2)
    assert not np.allclose(autocorrelation_rows(mf, [spec])[0], autocorrelation_rows(mf2, [spec])[0])


def test_build_many_keeps_the_order_and_reports_failures():
    out = build_many([STYRENE, "[*]CC", "[*]CC[*]"], n=3, conformers=2, jobs=1)
    assert [m is not None for m, _, _ in out] == [True, False, True]
    assert "attachment points" in out[1][2]
    m = out[2][0]
    assert sorted(set(unit_labels(m).tolist())) == [-1, 0, 1, 2]
    assert m.HasProp(f"atom.iprop.{UNIT_PROP}")


def test_the_command_writes_an_sdf_with_the_labels_and_the_columns(tmp_path):
    table = tmp_path / "units.csv"
    table.write_text("name,psmiles,tg\nPS,[*]CC([*])c1ccccc1,373\nbad,[*]CC,0\nPE,[*]CC[*],195\n",
                     encoding="utf-8")
    out = tmp_path / "units.sdf"
    r = subprocess.run([sys.executable, "-m", "cmfqsar.cli", "polymers", str(table), str(out),
                        "--smiles-column", "psmiles", "--name-column", "name", "--conformers", "2"],
                       cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr
    assert "1 repeat unit(s) could not be built" in r.stdout
    mols = [m for m in Chem.SDMolSupplier(str(out), removeHs=False)]
    assert [m.GetProp("_Name") for m in mols] == ["PS", "PE"]
    assert mols[0].GetProp("tg") == "373"
    assert sorted(set(unit_labels(mols[0]).tolist())) == [-1, 0, 1, 2]
