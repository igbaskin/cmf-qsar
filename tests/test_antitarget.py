"""A potency and an anti-target on the same compounds (plan item MC4).

The science of this item is in the two-task model, which plan item SC1 already tested. What is
new is the data and the way the difference co-field is read, so these tests fix those: the
records are paired only where both measurements exist, the basic nitrogen is found and an amide
nitrogen is not mistaken for one, the atom groups partition the molecule, and the file in the
repository carries what the benchmark and the manual say it does.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

import make_antitarget_examples as MK  # noqa: E402
import run_antitarget as RA  # noqa: E402

DATA = RA.DATA


def _row(mol, target, value, assay, kind="IC50"):
    return {"molecule_chembl_id": mol, "target_chembl_id": target, "pchembl_value": value,
            "assay_chembl_id": assay, "assay_description": f"an assay on {target}",
            "standard_type": kind, "canonical_smiles": "CCN1CCCCC1"}


# ------------------------------------------------------------------ pairing up


def test_only_the_molecules_with_both_measurements_are_kept():
    rows = [_row("A", MK.PRIMARY[0], "8.0", "a1", "Ki"), _row("A", MK.ANTI[0], "5.0", "a2"),
            _row("B", MK.PRIMARY[0], "7.0", "a1", "Ki"),
            _row("C", MK.ANTI[0], "6.0", "a2")]
    got = MK.pair_up(rows)
    assert set(got) == {"A"}
    assert got["A"]["primary"] == pytest.approx(8.0)
    assert got["A"]["anti"] == pytest.approx(5.0)
    assert got["A"]["primary_assay"] == "a1" and got["A"]["anti_assay"] == "a2"


def test_repeated_measurements_are_averaged_and_counted():
    rows = [_row("A", MK.PRIMARY[0], "8.0", "a1", "Ki"), _row("A", MK.PRIMARY[0], "9.0", "a1",
                                                              "Ki"),
            _row("A", MK.ANTI[0], "5.0", "a2")]
    got = MK.pair_up(rows)["A"]
    assert got["primary"] == pytest.approx(8.5)
    assert got["repeats"] == 2


def test_another_target_in_the_same_paper_is_ignored():
    rows = [_row("A", MK.PRIMARY[0], "8.0", "a1", "Ki"), _row("A", MK.ANTI[0], "5.0", "a2"),
            _row("A", "CHEMBL999", "7.0", "a3")]
    got = MK.pair_up(rows)["A"]
    assert got["primary"] == pytest.approx(8.0) and got["anti"] == pytest.approx(5.0)


# -------------------------------------------------------------- the basic nitrogen


def _mol3d(smiles, seed=1):
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def test_an_aliphatic_amine_is_basic_and_the_other_nitrogens_are_not():
    cases = {
        "CCN1CCCCC1": 1,              # a piperidine: the one hERG responds to
        "CC(=O)NC": 0,                # an amide
        "c1ccncc1": 0,                # pyridine, aromatic
        "CC#N": 0,                    # a nitrile
        "CS(=O)(=O)NC": 0,            # a sulfonamide
        "CCN(CC)CCN1CCCC1": 2,        # two amines
    }
    for smiles, n in cases.items():
        mol = Chem.MolFromSmiles(smiles)
        assert len(RA.basic_nitrogens(mol)) == n, smiles


def test_the_groups_cover_every_heavy_atom_once():
    mol = _mol3d("c1ccc(OCCCN2CCCC2)cc1")
    basic, beside, rest = RA.atom_groups(mol)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    assert sorted(basic + beside + rest) == sorted(heavy)
    assert len(set(basic) & set(beside)) == 0
    assert basic, "this molecule has a pyrrolidine nitrogen"
    assert beside, "and carbons beside it"


def test_a_molecule_with_no_basic_nitrogen_puts_everything_in_the_rest():
    mol = _mol3d("c1ccccc1C(=O)O")
    basic, beside, rest = RA.atom_groups(mol)
    assert basic == [] and beside == []
    assert len(rest) == Chem.RemoveHs(mol).GetNumHeavyAtoms()


def test_only_the_carbons_near_the_nitrogen_count_as_beside_it():
    mol = _mol3d("CN1CCCCC1", seed=3)
    basic, beside, rest = RA.atom_groups(mol)
    coords = np.asarray(mol.GetConformer().GetPositions(), float)
    for i in beside:
        assert mol.GetAtomWithIdx(i).GetSymbol() == "C"
        assert min(float(np.linalg.norm(coords[i] - coords[n])) for n in basic) <= RA.BESIDE


# ------------------------------------------------------------------ the statistic


def test_the_paired_difference_of_two_identical_predictions_is_zero():
    rng = np.random.RandomState(0)
    y = rng.normal(size=40)
    p = y + 0.3 * rng.normal(size=40)
    d, lo, hi = RA._paired_q2(y, p, p, runs=200)
    assert d == pytest.approx(0.0) and lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


def test_a_better_prediction_has_a_positive_paired_difference():
    rng = np.random.RandomState(1)
    y = rng.normal(size=60)
    good = y + 0.2 * rng.normal(size=60)
    bad = y + 1.0 * rng.normal(size=60)
    d, lo, hi = RA._paired_q2(y, good, bad, runs=300)
    assert d > 0 and lo > 0 and lo < d < hi


# -------------------------------------------------------------------- the data


@pytest.mark.skipif(not os.path.exists(DATA), reason="the anti-target set is not built")
def test_the_file_carries_both_endpoints_and_the_conditions_they_were_measured_in():
    mols = [m for m in Chem.SDMolSupplier(DATA, removeHs=False) if m is not None]
    assert len(mols) >= 50
    for m in mols:
        for tag in (RA.PRIMARY, RA.ANTI, "selectivity", "molecule_chembl_id",
                    "document_chembl_id", "primary_assay", "antitarget_assay",
                    "primary_assay_description", "antitarget_assay_description", "reference"):
            assert m.HasProp(tag), tag
        assert np.isfinite(float(m.GetProp(RA.PRIMARY)))
        assert np.isfinite(float(m.GetProp(RA.ANTI)))
        assert float(m.GetProp("selectivity")) == pytest.approx(
            float(m.GetProp(RA.PRIMARY)) - float(m.GetProp(RA.ANTI)), abs=2e-3)


@pytest.mark.skipif(not os.path.exists(DATA), reason="the anti-target set is not built")
def test_every_molecule_was_measured_in_the_same_two_assays():
    """An anti-target number from another laboratory would be a different measurement."""
    mols = [m for m in Chem.SDMolSupplier(DATA, removeHs=False) if m is not None]
    assert len({m.GetProp("primary_assay") for m in mols}) == 1
    assert len({m.GetProp("antitarget_assay") for m in mols}) == 1
    assert len({m.GetProp("document_chembl_id") for m in mols}) == 1


@pytest.mark.skipif(not os.path.exists(DATA), reason="the anti-target set is not built")
def test_the_series_is_superimposed_and_has_a_basic_nitrogen():
    mols = [m for m in Chem.SDMolSupplier(DATA, removeHs=False) if m is not None]
    # the molecules differ in size, so their centres do too; the median is what says they are in
    # one frame, and the one compound whose second amine is matched first is not evidence against
    centres = np.array([np.asarray(m.GetConformer().GetPositions(), float).mean(axis=0)
                        for m in mols])
    assert float(np.median(np.linalg.norm(centres - np.median(centres, axis=0), axis=1))) < 3.0
    amines = []
    for m in mols:
        heavy = Chem.RemoveHs(m)
        found = RA.basic_nitrogens(heavy)
        if found:
            amines.append(np.asarray(heavy.GetConformer().GetPositions(), float)[found[0]])
    amines = np.asarray(amines, float)
    assert len(amines) >= 0.9 * len(mols), \
        "these are basic amines; that is why hERG is the question"
    spread = np.linalg.norm(amines - np.median(amines, axis=0), axis=1)
    assert float(np.median(spread)) < 3.0, \
        "the common part of a superimposed series lands in the same place"


# ------------------------------------------------------------ the counter-ion


def test_a_salt_is_built_without_its_counter_ion():
    # ChEMBL writes a hydrochloride as the compound and "Cl" in one SMILES; the candidate file
    # held 30 such records until 14 November 2026, with the chloride on top of the compound
    salt = Chem.MolFromSmiles("CN(C)[C@@H]1Cc2ccccc2[C@@H](C2CCCCC2)C1.Cl")
    compound, left_out = MK.without_counter_ions(salt)
    assert left_out == "Cl"
    assert Chem.MolToSmiles(compound) == Chem.MolToSmiles(
        Chem.MolFromSmiles("CN(C)[C@@H]1Cc2ccccc2[C@@H](C2CCCCC2)C1"))
    twice = Chem.MolFromSmiles("C[C@@H]1CCCN1CCc1ccc(NC(=O)CN2CCOCC2)cc1.Cl.Cl")
    compound, left_out = MK.without_counter_ions(twice)
    assert left_out == "Cl.Cl" and len(Chem.GetMolFrags(compound)) == 1
    # a record that is one molecule comes back as the same object, so its embedding is unchanged
    free = Chem.MolFromSmiles("CCN1CCCCC1")
    assert MK.without_counter_ions(free) == (free, "")


@pytest.mark.parametrize("name", ["h3_herg.sdf", "h3_candidates.sdf"])
def test_every_record_of_the_two_files_is_one_molecule(name):
    path = os.path.join(os.path.dirname(DATA), name)
    if not os.path.exists(path):
        pytest.skip(f"{name} is not built")
    for m in Chem.SDMolSupplier(path, removeHs=False):
        assert len(Chem.GetMolFrags(m)) == 1, m.GetProp("_Name")
