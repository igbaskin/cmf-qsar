"""What an invariant model says about one molecule, atom by atom (plan item SC6).

The rotation-invariant kernel has no co-field, because it has no frame, but every one of its
channels is a sum over atom pairs, so a prediction still splits over the atoms of a molecule.
The claim being tested is that the split is exact: the atom numbers plus the bias are the
prediction, to the last digit, for every molecule and every learner whose score is a sum over the
training molecules. The rest of the tests fix the properties that make the numbers readable: an
atom with no weight in any field contributes nothing, the split does not depend on how the
molecule is turned, and a pair of field types gets the same total either way it is summed.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, atom_contributions,  # noqa: E402
                               autocorrelation_kernel, channel_coefficients, feature_matrix,
                               field_pairs, pair_contributions)
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "CCOc1ccc(C)cc1CC(=O)N",
          "CC(C)COc1ccccc1C(=O)NC", "CCCCOc1ccccc1C(=O)N", "CCCOc1ccccc1C(=O)NCCC",
          "CCOc1ccccc1C(=O)NCC", "COc1ccccc1C(=O)NCC", "CCCOc1ccc(Cl)cc1C(=O)NCC",
          "CCCOc1ccc(C)cc1C(=O)NCC"]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


@pytest.fixture(scope="module")
def fitted():
    mols = [_mol(s, i + 1) for i, s in enumerate(SMILES)]
    names = [f"m{i}" for i in range(len(mols))]
    y = np.linspace(4.0, 8.5, len(mols))
    fs = FieldSet.create(alpha=0.5)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1},
                     kernel_source="autocorrelation")
    model.fit(table, y, cv_folds=0)
    return {"model": model, "fields": mf, "table": table, "y": y, "mols": mols, "fs": fs}


# ------------------------------------------------------------------ the split is exact


def test_the_atom_numbers_add_up_to_the_prediction(fitted):
    model, mf, table = fitted["model"], fitted["fields"], fitted["table"]
    pred = model.predict(table)
    for i, mol in enumerate(mf):
        c = model.atom_contributions(mol)
        assert len(c) == mol.n_atoms
        assert c.sum() + model.bias == pytest.approx(pred[i], rel=1e-10, abs=1e-12)


def test_it_also_adds_up_for_a_molecule_the_model_never_saw(fitted):
    model, fs = fitted["model"], fitted["fs"]
    mol = build_fields(_mol("CCCOc1ccccc1C(=O)NCCCC", 99), fs, "new")
    pred = float(model.predict(AtomTable.from_molecules([mol]))[0])
    assert model.atom_contributions(mol).sum() + model.bias == pytest.approx(pred, rel=1e-10)


def test_it_adds_up_for_a_support_vector_model_too(fitted):
    """Any learner whose score is a sum over the training molecules has the same decomposition."""
    model = CMFModel(fieldset=fitted["fs"], method="svr", normalize=True,
                     params={"C": 10.0, "epsilon": 0.05}, kernel_source="autocorrelation")
    model.fit(fitted["table"], fitted["y"], cv_folds=0)
    pred = model.predict(fitted["table"])
    for i, mol in enumerate(fitted["fields"]):
        assert model.atom_contributions(mol).sum() + model.bias == pytest.approx(pred[i],
                                                                                rel=1e-9)


def test_the_pair_split_adds_up_to_the_same_total(fitted):
    model, mf = fitted["model"], fitted["fields"]
    for mol in mf[:3]:
        total = model.atom_contributions(mol).sum()
        pairs = model.invariant_pair_contributions(mol)
        assert len(pairs) == len(field_pairs(mol.weights.shape[1]))
        assert sum(v for _n, v in pairs) == pytest.approx(total, rel=1e-9, abs=1e-12)
        assert all(" with " in n or n in [s.name for s in model.fieldset.specs]
                   for n, _v in pairs)


# -------------------------------------------------------- what makes the numbers readable


def test_turning_the_molecule_does_not_change_the_split(fitted):
    """The kernel is invariant to rotation; so is every atom's share of the prediction."""
    model, fs = fitted["model"], fitted["fs"]
    mol = _mol(SMILES[2], 7)
    before = model.atom_contributions(build_fields(mol, fs, "a"))
    turned = Chem.Mol(mol)
    conf = turned.GetConformer()
    theta = 0.7
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                  [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    pos = np.array(conf.GetPositions()) @ R.T + np.array([3.0, -2.0, 1.5])
    for i in range(turned.GetNumAtoms()):
        conf.SetAtomPosition(i, pos[i])
    after = model.atom_contributions(build_fields(turned, fs, "a"))
    assert after == pytest.approx(before, rel=1e-8, abs=1e-10)


def test_an_atom_with_no_weight_anywhere_contributes_nothing(fitted):
    """An atom that no field sees cannot take a share: every channel is weighted by its weights."""
    model, fs = fitted["model"], fitted["fs"]
    mol = build_fields(_mol(SMILES[0], 5), fs, "a")
    W = np.asarray(mol.weights, float)
    silent = np.where(np.abs(W).sum(axis=1) == 0)[0]
    if len(silent) == 0:                       # the five fields reach every atom of this molecule
        silent = np.array([int(np.argmin(np.abs(W).sum(axis=1)))])
        mol.weights[silent] = 0.0
    c = model.atom_contributions(mol)
    assert np.abs(c[silent]).max() < 1e-12


def test_a_bigger_group_takes_a_bigger_share_than_one_of_its_atoms(fitted):
    """Sanity: the split is over atoms, so the ring is the sum of its carbons, not a new entity."""
    model, mf = fitted["model"], fitted["fields"]
    c = model.atom_contributions(mf[0])
    heavy = [i for i, s in enumerate(mf[0].symbols) if s != "H"]
    assert abs(c[heavy].sum()) > abs(c[heavy[0]])


# --------------------------------------------------------------- the plumbing around it


def test_the_function_can_be_used_without_a_model(fitted):
    """The same split from the features alone, which is what a script or another kernel needs."""
    model, mf = fitted["model"], fitted["fields"]
    spec = AutocorrelationSpec(d_max=model.ac_d_max, step=model.ac_sigma, sigma=model.ac_sigma)
    c = channel_coefficients(model.ac_train_, model.ac_column_scale, model.coef)
    mine = atom_contributions(mf[1], c, scale=model.ac_column_scale, spec=spec)
    assert mine == pytest.approx(model.atom_contributions(mf[1]), rel=1e-12)
    per_pair = atom_contributions(mf[1], c, scale=model.ac_column_scale, spec=spec, per_pair=True)
    assert per_pair.shape == (len(field_pairs(mf[1].weights.shape[1])), mf[1].n_atoms)
    assert per_pair.sum(axis=0) == pytest.approx(mine, rel=1e-12)


def test_an_empty_molecule_gives_an_empty_split(fitted):
    """A molecule with no atoms has nothing to split, and must not raise on the way there."""
    import dataclasses
    model = fitted["model"]
    spec = AutocorrelationSpec(d_max=model.ac_d_max, step=model.ac_sigma, sigma=model.ac_sigma)
    c = channel_coefficients(model.ac_train_, model.ac_column_scale, model.coef)
    full = build_fields(_mol("C", 2), fitted["fs"], "c")
    empty = dataclasses.replace(full, coords=np.zeros((0, 3)),
                                weights=np.zeros((0, full.weights.shape[1])),
                                radii=np.zeros(0), symbols=[], atom_types=[],
                                charges=np.zeros(0), bonds=np.zeros((0, 2), int))
    assert len(atom_contributions(empty, c, scale=model.ac_column_scale, spec=spec)) == 0


def test_a_model_with_a_nonlinear_layer_is_refused_by_name(fitted):
    model = CMFModel(fieldset=fitted["fs"], method="krr", normalize=True,
                     params={"lam": 0.1}, kernel_source="autocorrelation",
                     kernel_layer="gauss")
    model.fit(fitted["table"], fitted["y"], cv_folds=0)
    with pytest.raises(ValueError, match="linear"):
        model.atom_contributions(fitted["fields"][0])


def test_per_field_says_what_to_use_instead(fitted):
    with pytest.raises(ValueError, match="pair"):
        fitted["model"].atom_contributions(fitted["fields"][0], per_field=True)


def test_the_signed_three_point_channels_are_refused(fitted):
    """A triple is not a pair, so it has no place in a split over two atoms."""
    fs = fitted["fs"]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1},
                     kernel_source="autocorrelation", ac_chirality=True, ac_chirality_shells=2)
    model.fit(fitted["table"], fitted["y"], cv_folds=0)
    with pytest.raises(ValueError, match="three-point"):
        model.atom_contributions(fitted["fields"][0])


def test_a_molecule_with_other_fields_is_refused_rather_than_mixed(fitted):
    model = fitted["model"]
    other = build_fields(_mol(SMILES[0], 3), FieldSet.create(physchem=["steric"], alpha=0.5), "x")
    with pytest.raises(ValueError, match="five fields|do not match|carries"):
        model.atom_contributions(other)


def test_the_overlap_kernel_still_decomposes_as_before(fitted):
    """The dispatch must not change the model the decomposition rules were written for."""
    fs = FieldSet.create(alpha=0.5)
    mf = fitted["fields"]
    table = fitted["table"]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(table, fitted["y"], cv_folds=0)
    pred = model.predict(table)
    c = model.atom_contributions(mf[0])
    assert c.sum() + model.bias == pytest.approx(pred[0], rel=1e-9)
    assert model.atom_contributions(mf[0], per_field=True).shape == (len(fs.specs), mf[0].n_atoms)
