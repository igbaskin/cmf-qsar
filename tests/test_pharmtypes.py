"""Six pharmacophore features as indicator fields (plan item D2).

The vocabulary comes from RDKit's own feature definitions, so what is tested here is not the
chemistry of the definitions but that the fields are built from them correctly: the weights are
indicators, the right atoms carry them, a feature nothing carries makes an empty field rather than
an error, and the co-field of `pharm:donor` comes out of the pharmacophore extractor as a donor.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.pharmtypes import (FALLBACK_SMARTS, PHARM_FEATURES, coverage,  # noqa: E402
                                definitions_used, feature_atoms, features_in, pharm_weights)

SMILES = {
    "benzoic acid": "c1ccccc1C(=O)O",
    "aniline": "c1ccccc1N",
    "acetamide": "CC(N)=O",
    "butylamine": "CCCCN",
    "cyclohexane": "C1CCCCC1",
    "phenol": "c1ccccc1O",
}


def _mol(smi, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    return m


def _set():
    return [_mol(s, k + 1) for k, s in enumerate(SMILES.values())]


# ----------------------------------------------------------------------- the assignment


def test_the_vocabulary_is_the_six_the_item_names():
    assert PHARM_FEATURES == ("donor", "acceptor", "aromatic", "hydrophobe", "cation", "anion")
    assert set(FALLBACK_SMARTS) == set(PHARM_FEATURES)
    assert "RDKit" in definitions_used() or "SMARTS" in definitions_used()


def test_every_feature_gets_a_boolean_column_of_the_right_length():
    m = _mol(SMILES["benzoic acid"])
    got = feature_atoms(m)
    assert set(got) == set(PHARM_FEATURES)
    for name, mask in got.items():
        assert mask.dtype == bool and len(mask) == m.GetNumAtoms(), name


def test_a_weight_is_one_or_zero_and_never_two():
    """An atom that is both a hydrophobe and part of a lumped hydrophobic group is one hydrophobe."""
    m = _mol(SMILES["cyclohexane"])
    w = pharm_weights(m, "hydrophobe")
    assert set(np.unique(w)) <= {0.0, 1.0}
    assert w.sum() > 0


def test_an_aromatic_ring_is_six_aromatic_atoms():
    m = _mol(SMILES["phenol"])
    assert pharm_weights(m, "aromatic").sum() == pytest.approx(6.0)


def test_a_carboxylic_acid_is_an_anion_and_an_amine_is_a_cation():
    acid = feature_atoms(_mol(SMILES["benzoic acid"]))
    amine = feature_atoms(_mol(SMILES["butylamine"]))
    assert acid["anion"].any()
    assert not acid["cation"].any()
    assert amine["cation"].any()
    assert not amine["anion"].any()


def test_a_donor_and_an_acceptor_land_on_the_right_atoms():
    m = _mol(SMILES["acetamide"])
    got = feature_atoms(m)
    donors = [m.GetAtomWithIdx(int(i)).GetSymbol() for i in np.where(got["donor"])[0]]
    acceptors = [m.GetAtomWithIdx(int(i)).GetSymbol() for i in np.where(got["acceptor"])[0]]
    assert "N" in donors
    assert "O" in acceptors


def test_an_unknown_feature_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown pharmacophore feature"):
        pharm_weights(_mol(SMILES["phenol"]), "halogen")


# -------------------------------------------------------------------------- the coverage


def test_coverage_counts_molecules_and_atoms():
    cov = coverage(_set())
    assert set(cov) == set(PHARM_FEATURES)
    for name, d in cov.items():
        assert 0.0 <= d["molecules"] <= 1.0, name
        assert d["atoms_per_molecule"] >= 0.0
    assert cov["aromatic"]["molecules"] == pytest.approx(0.5)     # three of the six
    assert cov["hydrophobe"]["molecules"] == pytest.approx(1.0)   # every one of them


def test_features_in_applies_the_same_rule_as_the_sybyl_types():
    mols = _set()
    assert set(features_in(mols, 1)) >= set(features_in(mols, 5))
    assert features_in(mols, 1000) == []
    assert "aromatic" in features_in(mols, 2)


def test_coverage_of_nothing_is_nothing():
    assert coverage([]) == {}
    assert features_in([], 1) == []


# ------------------------------------------------------------------------ the field set


def test_a_field_set_can_be_built_from_the_features():
    fs = FieldSet.create(physchem=["steric"], pharm_features=list(PHARM_FEATURES), alpha=0.5)
    names = [sp.name for sp in fs.specs]
    assert names == ["steric"] + [f"pharm:{p}" for p in PHARM_FEATURES]
    pharm = [sp for sp in fs.specs if sp.is_pharm]
    assert len(pharm) == len(PHARM_FEATURES)
    assert [sp.pharm_feature for sp in pharm] == list(PHARM_FEATURES)
    assert [sp.label for sp in pharm] == [p.capitalize() for p in PHARM_FEATURES]
    assert not fs.specs[0].is_pharm and fs.specs[0].pharm_feature is None


def test_the_weights_reach_the_atom_table():
    fs = FieldSet.create(physchem=[], pharm_features=["donor", "aromatic"], alpha=0.5)
    mols = _set()
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                      for i, m in enumerate(mols)])
    assert table.weights.shape[1] == 2
    for k in range(2):
        assert set(np.unique(table.weights[:, k])) <= {0.0, 1.0}
    assert table.weights[:, 1].sum() > table.weights[:, 0].sum()   # more aromatic than donor


def test_a_field_of_a_feature_nothing_carries_is_empty_not_an_error():
    fs = FieldSet.create(physchem=[], pharm_features=["anion"], alpha=0.5)
    mol = build_fields(_mol(SMILES["cyclohexane"]), fs, "m")
    assert mol.weights.shape[1] == 1
    assert mol.weights.sum() == 0.0


def test_a_field_naming_a_feature_that_does_not_exist_fails_clearly():
    fs = FieldSet.create(physchem=[], pharm_features=["halogen"], alpha=0.5)
    with pytest.raises(ValueError, match="unknown pharmacophore feature"):
        build_fields(_mol(SMILES["phenol"]), fs, "m")


def test_the_fields_can_be_asked_for_through_a_field_list():
    """`--fields pharm:donor,steric` has to work as well as the --pharm flag."""
    from cmfqsar.cli import _split_fields
    physchem, props = _split_fields("steric,pharm:donor")
    assert props == []
    fs = FieldSet.create(physchem=physchem, alpha=0.5)
    assert [sp.name for sp in fs.specs] == ["steric", "pharm:donor"]
    mol = build_fields(_mol(SMILES["acetamide"]), fs, "m")
    assert mol.weights[:, 1].sum() > 0


# ------------------------------------------------------- it reads directly as a pharmacophore


def test_a_cofield_of_a_feature_comes_out_as_that_feature():
    """The claim of the item: no translation step between the field and the pharmacophore."""
    from cmfqsar.pharmacophore import FEATURE_TYPES
    for name in PHARM_FEATURES:
        good = FEATURE_TYPES[(f"pharm:{name}", +1)]
        bad = FEATURE_TYPES[(f"pharm:{name}", -1)]
        assert good[2] is True and bad[2] is False
        assert name in good[1] and name in bad[1]
        assert good[0] in ("donor", "acceptor", "aromatic", "hydrophobic", "cation", "anion")
        assert bad[0].endswith("_bad")


def test_every_feature_type_has_an_element_for_the_pdb_export():
    from cmfqsar.pharmacophore import FEATURE_TYPES, PDB_ELEMENT
    for (field, _sign), (kind, _meaning, _good) in FEATURE_TYPES.items():
        if field.startswith("pharm:"):
            assert kind in PDB_ELEMENT, kind


def test_a_pharmacophore_extracted_from_such_a_model_is_typed(tmp_path):
    from cmfqsar.model import CMFModel
    from cmfqsar.pharmacophore import extract
    fs = FieldSet.create(physchem=[], pharm_features=["donor", "aromatic", "hydrophobe"],
                         alpha=0.5)
    mols = [_mol(s, k + 1) for k, s in enumerate(
        ["c1ccccc1C(=O)O", "c1ccc(N)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
         "c1ccc(CN)cc1C(=O)O", "c1ccc(CCN)cc1C(=O)O"])]
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}")
                                      for i, m in enumerate(mols)])
    y = np.linspace(4.0, 8.0, len(mols))
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(table, y, cv_folds=0)
    feats = extract(model, level=0.5, max_features=8, min_z=0.0)
    assert feats
    kinds = {f.kind for f in feats}
    assert kinds & {"donor", "aromatic", "hydrophobic", "donor_bad", "aromatic_bad",
                    "hydrophobic_bad"}
    for f in feats:
        assert f.field.startswith("pharm:")
        assert f.meaning                                  # every feature says what it asks for
