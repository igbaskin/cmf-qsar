"""Tests for ranking by a confidence bound rather than by the point estimate."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, MoleculeFields, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "CCOc1ccc(C)cc1CC(=O)N",
          "CC(C)COc1ccccc1C(=O)NC", "CCCCOc1ccccc1C(=O)N", "CCOc1ccccc1C(=O)NCC"]


def _mol(smiles, seed=1, name="m"):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", name)
    return m


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create(alpha=0.5)
    mf = [build_fields(_mol(s, seed=i + 1, name=f"m{i}"), fs, f"m{i}")
          for i, s in enumerate(SMILES)]
    y = np.array([5.0, 5.5, 6.0, 6.4, 7.0, 7.3])
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="gp", normalize=True, params={"lam": 0.1})
    model.fit(table, y, cv_folds=0)
    return fs, mf, y, model, table


def test_a_bound_of_zero_is_the_prediction_itself(world):
    """The default has to be what ranking by a model has always meant here."""
    fs, mf, y, model, table = world
    score, yp, sd = model.rank_score(table, beta=0.0)
    assert np.allclose(score, yp)
    assert np.allclose(yp, model.predict(table))
    assert np.all(sd >= 0) and np.all(np.isfinite(sd))


def test_the_bound_is_the_prediction_minus_that_many_deviations(world):
    fs, mf, y, model, table = world
    _, var = model.predict_with_variance(table)
    expect = np.sqrt(np.maximum(var, 0.0))
    for beta in (0.25, 1.0, 3.0):
        score, yp, sd = model.rank_score(table, beta=beta)
        assert np.allclose(sd, expect)
        assert np.allclose(score, yp - beta * sd)
        # and it is a lower bound: never above the prediction
        assert np.all(score <= yp + 1e-12)


def test_the_sign_follows_which_end_is_active(world):
    """For an activity where a smaller value is more active the bound is added, not subtracted."""
    fs, mf, y, model, table = world
    lo, yp, sd = model.rank_score(table, beta=1.0, larger_is_better=False)
    assert np.allclose(lo, yp + sd)
    hi, _, _ = model.rank_score(table, beta=1.0, larger_is_better=True)
    assert np.allclose(hi, yp - sd)


def test_the_bound_demotes_the_molecule_the_model_knows_nothing_about(world):
    """The purpose: a molecule unlike the training set gets a wide posterior and is pushed down."""
    fs, mf, y, model, table = world
    stranger = build_fields(_mol("NCCc1ccc(O)cc1", seed=99, name="stranger"), fs, "stranger")
    far = AtomTable.from_molecules([mf[0], stranger])
    score0, yp0, sd0 = model.rank_score(far, beta=0.0)
    score1, yp1, sd1 = model.rank_score(far, beta=2.0)
    assert sd1[1] > sd1[0]                       # the unfamiliar molecule is the uncertain one
    # the bound lowers the unfamiliar molecule more than the training one, so its rank can only fall
    drop = (yp1 - score1)
    assert drop[1] > drop[0]
    assert score1[1] - score1[0] < score0[1] - score0[0]


def test_a_nonlinear_kernel_has_no_posterior_and_says_so(world):
    fs, mf, y, model, table = world
    deep = CMFModel(fieldset=fs, method="krr", normalize=True,
                    params={"lam": 0.1, "gamma": 1.0}, kernel_layer="gauss")
    deep.fit(table, y, cv_folds=0)
    with pytest.raises(ValueError, match="posterior variance"):
        deep.rank_score(table, beta=1.0)


def test_the_ranking_can_change_and_only_at_the_top(world):
    """A bound reorders a list; it does not change how well the model separates overall."""
    fs, mf, y, model, table = world
    stranger = build_fields(_mol("NCCc1ccc(O)cc1", seed=5, name="s"), fs, "s")
    lib = AtomTable.from_molecules(list(mf) + [stranger])
    by_mean = np.argsort(-model.rank_score(lib, beta=0.0)[0])
    by_bound = np.argsort(-model.rank_score(lib, beta=2.0)[0])
    assert list(by_mean) != list(by_bound)
