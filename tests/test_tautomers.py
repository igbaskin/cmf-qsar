"""Tautomer ensembles: the other half of plan item D4.

The mathematics is the protonation ensemble's, so what is tested here is the enumeration and the
weights: the forms keep the parent's heavy-atom geometry, the weights are what each setting says
they are, and the kernel of two ensembles is still the weighted mean of the kernels of their
member pairs.  The weights themselves are assumptions, not populations, and the tests say so by
checking that the module refuses to invent energies it does not have.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.tautomers import (ENERGY_PROP, WEIGHTINGS, ambiguous_share,  # noqa: E402
                               available, forms_table, tautomer_fields, tautomer_states)

SMILES = {
    "triazole": "c1nnc[nH]1",            # the proton sits on either nitrogen
    "benzimidazole": "Cc1nc2ccccc2[nH]1",
    "uracil": "O=c1cc[nH]c(=O)[nH]1",    # many forms, most of them formal
    "benzene": "c1ccccc1",               # one form
}

needs_rdkit_tautomers = pytest.mark.skipif(not available(),
                                           reason="this RDKit has no tautomer enumerator")


def _mol(smiles, seed=3, name="m"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", name)
    return m


# --------------------------------------------------------------------------- the enumeration


@needs_rdkit_tautomers
def test_a_molecule_with_one_form_is_one_state():
    e = tautomer_states(_mol(SMILES["benzene"]))
    assert len(e) == 1 and e.states[0].weight == 1.0
    assert "one tautomer" in e.note


@needs_rdkit_tautomers
def test_the_two_forms_of_a_triazole_are_found_and_weighed_equally():
    e = tautomer_states(_mol(SMILES["triazole"]))
    assert len(e) == 2
    assert e.weights == pytest.approx([0.5, 0.5])
    assert e.weights.sum() == pytest.approx(1.0)
    assert len({s.label for s in e.states}) == 2        # two different forms, not one twice


@needs_rdkit_tautomers
def test_the_score_weighting_keeps_fewer_forms_than_the_uniform_one():
    """Uracil has many forms; most are ones the enumerator reaches only formally."""
    scored = tautomer_states(_mol(SMILES["uracil"]), weighting="score")
    every = tautomer_states(_mol(SMILES["uracil"]), weighting="uniform")
    assert len(every) > len(scored)
    assert every.weights == pytest.approx(np.full(len(every), 1.0 / len(every)))


@needs_rdkit_tautomers
def test_the_cap_on_the_number_of_forms_is_kept():
    e = tautomer_states(_mol(SMILES["uracil"]), weighting="uniform", max_forms=3)
    assert len(e) == 3
    assert e.weights.sum() == pytest.approx(1.0)


@needs_rdkit_tautomers
def test_given_energies_become_boltzmann_weights():
    mol = _mol(SMILES["triazole"])
    n = len(tautomer_states(mol, weighting="uniform"))
    mol.SetProp(ENERGY_PROP, " ".join(["0.0"] + ["10.0"] * (n - 1)))
    e = tautomer_states(mol, weighting="given", kt=0.6)
    assert e.weights[0] > 0.99                 # ten kcal/mol is out of reach at room temperature
    assert e.weights.sum() == pytest.approx(1.0)


@needs_rdkit_tautomers
def test_energies_that_are_not_there_are_not_invented():
    e = tautomer_states(_mol(SMILES["triazole"]), weighting="given")
    assert len(e) == 1 and "no per-form energies" in e.note
    mol = _mol(SMILES["triazole"])
    mol.SetProp(ENERGY_PROP, "0.0")            # one number for two forms
    assert len(tautomer_states(mol, weighting="given")) == 1


def test_an_unknown_weighting_is_refused_by_name():
    with pytest.raises(ValueError, match="unknown tautomer weighting"):
        tautomer_states(_mol(SMILES["triazole"]), weighting="boltzmann")
    assert WEIGHTINGS == ("score", "uniform", "given")


@needs_rdkit_tautomers
def test_a_molecule_without_geometry_comes_back_as_itself():
    flat = Chem.AddHs(Chem.MolFromSmiles(SMILES["triazole"]))
    e = tautomer_states(flat)
    assert len(e) == 1 and "no geometry" in e.note


@needs_rdkit_tautomers
def test_the_heavy_atoms_keep_the_geometry_they_were_given():
    mol = _mol(SMILES["benzimidazole"])
    want = np.array(Chem.RemoveHs(Chem.Mol(mol)).GetConformer().GetPositions())
    e = tautomer_states(mol, weighting="uniform")
    assert len(e) > 1
    for st in e.states:
        got = np.array(Chem.RemoveHs(Chem.Mol(st.mol)).GetConformer().GetPositions())
        assert np.allclose(got, want, atol=1e-6)


# ------------------------------------------------------------------------------- the field


@needs_rdkit_tautomers
def test_one_form_gives_the_same_field_as_building_the_molecule():
    fs = FieldSet.create(alpha=0.5)
    mol = _mol(SMILES["benzene"])
    a = tautomer_fields(mol, fs, name="b")
    b = build_fields(mol, fs, "b", _single=True)
    assert np.array_equal(a.coords, b.coords) and np.array_equal(a.weights, b.weights)
    assert not a.is_ensemble


@needs_rdkit_tautomers
def test_the_ensemble_field_is_the_weighted_mean_of_the_forms():
    fs = FieldSet.create(alpha=0.5)
    mol = _mol(SMILES["triazole"])
    e = tautomer_states(mol)
    mf = tautomer_fields(mol, fs, name="t")
    assert mf.is_ensemble and mf.member_kind == "tautomers"
    assert len(mf.sizes) == len(e.states)
    for i, st in enumerate(e.states):
        one = build_fields(st.mol, fs, "s", _single=True)
        assert np.allclose(mf.weights[mf.member_slice(i)], one.weights * st.weight)


@needs_rdkit_tautomers
def test_the_kernel_of_two_ensembles_is_the_mean_of_the_member_kernels():
    fs = FieldSet.create(alpha=0.6)
    mols = [_mol(SMILES["triazole"], seed=1), _mol(SMILES["benzimidazole"], seed=2)]
    ens = [tautomer_states(m, weighting="uniform") for m in mols]
    assert all(len(e) > 1 for e in ens)
    tab = AtomTable.from_molecules([tautomer_fields(m, fs, weighting="uniform", name=f"m{i}")
                                    for i, m in enumerate(mols)])
    K = kernel_matrices(tab, tab, fs.specs)
    for i, a in enumerate(ens):
        for j, b in enumerate(ens):
            acc = np.zeros(len(fs.specs))
            for sa in a.states:
                for sb in b.states:
                    pair = AtomTable.from_molecules(
                        [build_fields(sa.mol, fs, "a", _single=True),
                         build_fields(sb.mol, fs, "b", _single=True)])
                    acc += sa.weight * sb.weight * kernel_matrices(pair, pair, fs.specs)[:, 0, 1]
            assert K[:, i, j] == pytest.approx(acc, rel=1e-9, abs=1e-12)


@needs_rdkit_tautomers
def test_atom_contributions_come_back_on_the_atoms_of_the_molecule():
    fs = FieldSet.create(alpha=0.5)
    mf = tautomer_fields(_mol(SMILES["triazole"]), fs, name="t")
    v = np.ones(mf.n_atoms)
    got = mf.per_atom(v)
    assert len(got) == mf.sizes[0]
    assert got.sum() == pytest.approx(mf.n_atoms)      # a moving proton lands on its heavy atom
    assert mf.unmapped_share(v) == 0.0


@needs_rdkit_tautomers
def test_the_field_set_carries_the_setting_and_build_fields_uses_it():
    fs = FieldSet.create(alpha=0.5, tautomers="uniform")
    assert fs.is_tautomer_ensemble
    mf = build_fields(_mol(SMILES["triazole"]), fs, "t")
    assert mf.is_ensemble and mf.member_kind == "tautomers"
    assert not build_fields(_mol(SMILES["triazole"]), FieldSet.create(alpha=0.5), "t").is_ensemble


@needs_rdkit_tautomers
def test_the_two_kinds_of_ensemble_cannot_be_combined_yet():
    fs = FieldSet.create(alpha=0.5, tautomers="score", protonation_ph=0.0)
    with pytest.raises(ValueError, match="cannot be combined"):
        build_fields(_mol(SMILES["triazole"]), fs, "t")


@needs_rdkit_tautomers
def test_a_whole_series_can_be_described_at_once():
    mols = [_mol(SMILES[k], seed=i + 1) for i, k in enumerate(["triazole", "benzene"])]
    forms = forms_table(mols)
    assert len(forms) == 2
    assert ambiguous_share(forms) == pytest.approx(0.5)
    assert ambiguous_share([]) == 0.0


@needs_rdkit_tautomers
def test_a_series_of_azoles_trains_and_the_kernel_stays_positive_definite():
    from cmfqsar.model import CMFModel
    fs = FieldSet.create(alpha=0.5, tautomers="uniform")
    smis = ["c1nnc[nH]1", "Cc1nnc[nH]1", "CCc1nnc[nH]1", "Cc1nc2ccccc2[nH]1", "c1ccccc1",
            "O=c1cc[nH]c(=O)[nH]1"]
    mols = [_mol(s, seed=i + 1) for i, s in enumerate(smis)]
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    K = kernel_matrices(table, table, fs.specs)
    for f in range(K.shape[0]):
        assert np.linalg.eigvalsh(K[f]).min() > -1e-8, f"field {f} is not positive semidefinite"
    y = np.linspace(4.0, 9.0, len(mols))
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(table, y, cv_folds=0)
    assert model.stats is not None and model.stats.r2 > 0.5


@needs_rdkit_tautomers
def test_the_report_says_what_the_ensemble_was():
    rep = {}
    tautomer_fields(_mol(SMILES["triazole"]), FieldSet.create(alpha=0.5), name="t", report=rep)
    assert rep["n_forms"] == 2
    assert sum(rep["weights"]) == pytest.approx(1.0)
    assert len(rep["forms"]) == 2 and all(rep["forms"])
