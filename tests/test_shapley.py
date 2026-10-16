"""Tests for the Shapley decomposition of a CMF prediction."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.shapley import (AtomGame, FieldGame, atom_shapley, common_core,  # noqa: E402
                             field_shapley, shapley, substituent_groups)

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1F",
          "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(N)cc1C", "c1ccc(Cl)cc1C", "c1ccc(F)cc1N"]


def _dataset(seed=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fs = FieldSet.create(alpha=0.6)
    mols, mf = [], []
    for i, s in enumerate(SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed + i
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        mols.append(m)
        mf.append(build_fields(m, fs, f"m{i}"))
    table = AtomTable.from_molecules(mf)
    y = np.array([m.coords.shape[0] for m in mf], float)
    y = (y - y.mean()) / y.std()
    return mols, mf, table, y, fs


def _model(fs, table, y, **kw):
    m = CMFModel(fieldset=fs.copy(), method="krr", params=dict({"lam": 1e-3}, **kw.pop("params", {})),
                 **kw)
    return m.fit(table, y, cv_folds=0)


def test_shapley_reproduces_the_published_atom_contributions():
    """For the linear kernel the game is additive, so the values are exactly rule 3."""
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=False)
    mol = mf[0]
    res = atom_shapley(m, mol, exact_limit=20, interactions=True)
    assert res.exact
    assert np.allclose(res.values, m.atom_contributions(mol), atol=1e-10)
    assert res.efficiency_error < 1e-10
    # an additive game has no interactions at all
    assert np.abs(res.interactions).max() < 1e-10
    # and the values add up to the prediction, the empty field contributing only the bias
    assert res.baseline == pytest.approx(m.bias, abs=1e-10)
    assert res.prediction == pytest.approx(float(m.predict(AtomTable.from_molecules([mol]))[0]),
                                           abs=1e-10)


@pytest.mark.parametrize("normalize", [False, True])
def test_field_game_reproduces_the_per_field_contributions_of_a_linear_model(normalize):
    """Each field is normalised on its own, so the field game is additive for any linear model."""
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=normalize)
    mol = mf[1]
    res = field_shapley(m, mol, interactions=True)
    per_field = m.atom_contributions(mol, per_field=True).sum(axis=1)
    assert np.allclose(res.values, per_field, atol=1e-10)
    assert res.efficiency_error < 1e-10
    assert np.abs(res.interactions).max() < 1e-10


def test_normalised_linear_model_is_not_additive_but_stays_efficient():
    """Cosine normalisation couples the atoms, so Shapley redistributes and rule 3 differs."""
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True)
    mol = mf[0]
    res = atom_shapley(m, mol, exact_limit=20, interactions=True)
    assert res.efficiency_error < 1e-10
    assert not np.allclose(res.values, m.atom_contributions(mol), atol=1e-3)
    assert np.abs(res.interactions).max() > 1e-6
    # both decompositions still describe the same prediction
    assert res.values.sum() + res.baseline == pytest.approx(res.prediction, abs=1e-10)


@pytest.mark.parametrize("layer", ["gauss", "arccos1"])
def test_nonlinear_models_can_be_decomposed_exactly(layer):
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True, kernel_layer=layer, params={"gamma": 1.0})
    mol = mf[5]
    with pytest.raises(ValueError, match="linear"):
        m.atom_contributions(mol)                     # the published rule is unavailable
    res = atom_shapley(m, mol, exact_limit=20, interactions=True)
    assert res.exact and res.efficiency_error < 1e-9
    assert res.prediction == pytest.approx(float(m.predict(AtomTable.from_molecules([mol]))[0]),
                                           abs=1e-10)
    assert np.abs(res.interactions).max() > 1e-6      # the layer really does couple the atoms
    fres = field_shapley(m, mol)
    assert fres.efficiency_error < 1e-9
    assert fres.values.sum() == pytest.approx(fres.prediction - fres.baseline, abs=1e-9)


def test_sampling_agrees_with_the_exact_values():
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True, kernel_layer="gauss", params={"gamma": 1.0})
    mol = mf[5]
    exact = atom_shapley(m, mol, exact_limit=20)
    sampled = atom_shapley(m, mol, exact_limit=4, samples=400, seed=1)
    assert not sampled.exact and sampled.stderr is not None
    assert np.max(np.abs(sampled.values - exact.values)) < 0.05
    # the sampled estimate is still efficient: the permutations telescope to the full coalition
    assert sampled.efficiency_error < 1e-9
    assert np.all(sampled.stderr >= 0)


def test_a_fixed_core_changes_the_baseline_not_the_total():
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True, kernel_layer="gauss", params={"gamma": 1.0})
    mol = mf[6]
    core = list(range(6))                     # hold six atoms fixed in every coalition
    res = atom_shapley(m, mol, core=core, exact_limit=20)
    assert len(res.labels) == mol.n_atoms - len(core)
    assert res.prediction == pytest.approx(float(m.predict(AtomTable.from_molecules([mol]))[0]),
                                           abs=1e-10)
    assert res.efficiency_error < 1e-9
    # the baseline is now the prediction for the core alone, not for empty space
    game = AtomGame(m, mol, core=core)
    assert res.baseline == pytest.approx(float(game.value(np.zeros((1, game.n_players), bool))[0]))
    empty = AtomGame(m, mol)
    assert res.baseline != pytest.approx(float(empty.value(np.zeros((1, empty.n_players), bool))[0]))


def test_groups_pool_atoms_and_the_scaffold_helpers_work():
    mols, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True)
    i = 5                                      # 4-fluorophenol against the series
    core = common_core(mols[i], [mols[k] for k in range(len(mols)) if k != i])
    groups, labels = substituent_groups(mols[i], core)
    assert groups and len(labels) == len(groups)
    # every atom belongs to the core or to exactly one group
    flat = [a for g in groups for a in g]
    assert len(set(flat)) == len(flat)
    assert set(flat) | set(core) == set(range(mols[i].GetNumAtoms()))
    res = atom_shapley(m, mf[i], groups=groups, core=core, labels=labels, exact_limit=20)
    assert len(res.values) == len(groups)
    assert res.efficiency_error < 1e-9
    # pooling atoms into one player is the same as summing their values when the game is additive
    lin = _model(fs, table, y, normalize=False)
    pooled = atom_shapley(lin, mf[i], groups=groups, core=core, labels=labels, exact_limit=20)
    atoms = lin.atom_contributions(mf[i])
    assert np.allclose(pooled.values, [atoms[g].sum() for g in groups], atol=1e-9)


def test_interaction_sampling_and_the_player_limit():
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True, kernel_layer="gauss", params={"gamma": 1.0})
    mol = mf[0]
    groups = [[i] for i in range(6)] + [list(range(6, mol.n_atoms))]
    exact = atom_shapley(m, mol, groups=groups, exact_limit=20, interactions=True)
    sampled = atom_shapley(m, mol, groups=groups, exact_limit=3, samples=200, seed=0,
                           interactions=True, interaction_samples=400)
    assert np.max(np.abs(sampled.interactions - exact.interactions)) < 0.1
    assert exact.interactions_exact and not sampled.interactions_exact
    # too many players for pairwise indices: the caller is told to group the atoms
    with pytest.raises(ValueError, match="at most"):
        shapley(AtomGame(m, mol), exact_limit=2, samples=20, interactions=True,
                interaction_limit=4)


def test_atom_field_game_reproduces_the_per_field_atom_contributions():
    """The basis of a Shapley field: for a linear kernel it is exactly rule 3, field by field."""
    from cmfqsar.shapley import atom_field_shapley
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=False)
    mol = mf[0]
    res = atom_field_shapley(m, mol, exact_limit=0, samples=50, seed=0)
    assert res.kind == "atom-fields"
    assert len(res.labels) == mol.n_atoms * len(fs.specs)
    got = res.field_atom_values(mol.n_atoms, len(fs.specs))
    assert np.allclose(got, m.atom_contributions(mol, per_field=True), atol=1e-10)
    assert res.efficiency_error < 1e-9
    with pytest.raises(ValueError, match="no field axis"):
        atom_shapley(m, mol, exact_limit=20).field_atom_values(mol.n_atoms, len(fs.specs))


def test_shapley_field_integrates_to_the_attributed_value():
    from cmfqsar.cofields import make_grid, shapley_volume
    from cmfqsar.shapley import atom_field_shapley
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True, kernel_layer="gauss", params={"gamma": 1.0})
    mol = mf[5]
    res = atom_field_shapley(m, mol, samples=200, seed=0)
    vals = res.field_atom_values(mol.n_atoms, len(fs.specs))
    assert res.efficiency_error < 1e-9
    assert vals.sum() == pytest.approx(res.prediction - res.baseline, abs=1e-9)
    grid = make_grid(mol.coords, margin=6.0, spacing=0.3)
    dv = grid.spacing ** 3
    total = 0.0
    for f, spec in enumerate(fs.specs):
        vol = shapley_volume(mol, spec, vals[f], grid)
        assert vol.sum() * dv == pytest.approx(float(vals[f].sum()), abs=2e-3)
        total += vol.sum() * dv
    assert total == pytest.approx(res.prediction - res.baseline, abs=1e-2)
    # without the normalising prefactor the map keeps the shape but not the integral
    raw = shapley_volume(mol, fs.specs[0], vals[0], grid, normalized=False)
    assert abs(raw.sum() * dv) > abs(float(vals[0].sum()))


def test_atom_field_game_can_use_groups_and_a_core():
    from cmfqsar.shapley import AtomFieldGame, atom_field_shapley
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=True, kernel_layer="arccos1")
    mol = mf[2]
    groups = [[0, 1], [2], [3]]
    core = list(range(4, mol.n_atoms))
    res = atom_field_shapley(m, mol, groups=groups, core=core, exact_limit=20)
    assert len(res.labels) == len(groups) * len(fs.specs)
    assert res.efficiency_error < 1e-9
    game = AtomFieldGame(m, mol, groups=groups, core=core)
    assert game.value(np.ones((1, game.n_players), bool))[0] == pytest.approx(res.prediction)
    # the core alone is the baseline, and it is not the empty field
    assert game.value(np.zeros((1, game.n_players), bool))[0] == pytest.approx(res.baseline)


def test_atom_values_spread_group_values_and_the_table_is_sorted():
    _, mf, table, y, fs = _dataset()
    m = _model(fs, table, y, normalize=False)
    mol = mf[0]
    groups = [[0, 1], [2], [3]]
    res = atom_shapley(m, mol, groups=groups, core=list(range(4, mol.n_atoms)), exact_limit=20)
    spread = res.atom_values(mol.n_atoms)
    assert spread[0] == pytest.approx(res.values[0] / 2)
    assert spread[2] == pytest.approx(res.values[1])
    assert spread[mol.n_atoms - 1] == 0.0          # core atoms are not players
    vals = [abs(v) for _, v, _ in res.table()]
    assert vals == sorted(vals, reverse=True)
    with pytest.raises(ValueError, match="no players"):
        shapley(AtomGame(m, mol, groups=[], core=list(range(mol.n_atoms))))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
