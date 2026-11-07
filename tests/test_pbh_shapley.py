"""The per-ring reading of the layered model (plan item PBH6).

The Gaussian kernel layer halves the error of the field model on the polybenzenoid gap and gives
up rule 2 for it: a model that is not linear in the field has no co-field and no exact per-atom
decomposition. Shapley values with one ring per player are its reading, and these tests hold that
reading to what the benchmark's table claims about it: it is exact, it adds up to the prediction
the model makes, and it describes the layered model and not the linear one.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import pbh  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.align import set_coords  # noqa: E402
from cmfqsar.shapley import atom_shapley  # noqa: E402

#: five benzenoids and their measured gaps, enough for a model and for rings of three classes
SMILES = {
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "phenanthrene": "c1ccc2c(c1)ccc1ccccc12",
    "triphenylene": "c1ccc2c(c1)c1ccccc1c1ccccc21",
    "chrysene": "c1ccc2c(c1)ccc1c3ccccc3ccc21",
}
GAPS = np.array([4.78, 3.56, 4.70, 4.87, 4.06])


def _mol(name: str, seed: int = 1):
    m = Chem.AddHs(Chem.MolFromSmiles(SMILES[name]))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    set_coords(m, pbh.canonical_frame(m))
    m.SetProp("_Name", name)
    return m


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create(physchem=("steric", "electrostatic"), cif_types=pbh.PBH_TYPES,
                         cif_scheme="pbh", ring_classes=pbh.RING_CLASSES + ("sextet",),
                         alpha=0.5)
    mols = [_mol(n) for n in SMILES]
    mfs = [pbh.build_pbh_fields(m, fs) for m in mols]
    table = AtomTable.from_molecules(mfs)
    linear = CMFModel(fieldset=fs, method="krr", params={"lam": 0.05})
    linear.fit(table, GAPS, cv_folds=0)
    layered = CMFModel(fieldset=fs, method="krr", kernel_layer="gauss",
                       params={"lam": 0.05, "gamma": 0.25})
    layered.fit(table, GAPS, cv_folds=0)
    rings = []
    for m in mols:
        r = pbh.ring_classes(m)
        pbh.clar_weights(m, r)
        rings.append(r)
    return fs, mols, mfs, rings, linear, layered


def _players(mf, rings):
    """Every atom in exactly one ring's group, which is what the game needs."""
    return pbh.ring_players(mf, rings)


def test_the_layered_model_is_not_linear_in_the_field(world):
    """Which is why it needs Shapley values: rule 2 and rule 3 do not hold for it."""
    _, _, _, _, linear, layered = world
    assert linear.is_linear
    assert not layered.is_linear
    assert layered.kernel_layer == "gauss"


def test_one_ring_is_one_player_and_the_values_are_exact(world):
    _, _, mfs, rings, _, layered = world
    for mf, r in zip(mfs, rings):
        sh = atom_shapley(layered, mf, groups=_players(mf, r), exact_limit=14)
        assert len(sh.values) == len(r)
        assert sh.exact                                  # at most 14 players: no sampling
        assert sh.n_evaluations >= len(r) + 2            # the empty and the full coalition at least
        assert sh.efficiency_error < 1e-8
        assert sh.stderr is None


def test_the_values_add_up_to_the_prediction_the_model_makes(world):
    """The reading must explain the model being read, not a nearby one."""
    _, _, mfs, rings, _, layered = world
    for mf, r in zip(mfs, rings):
        sh = atom_shapley(layered, mf, groups=_players(mf, r), exact_limit=14)
        direct = float(layered.predict(AtomTable.from_molecules([mf]))[0])
        assert sh.prediction == pytest.approx(direct, abs=1e-8)
        assert float(sh.values.sum() + sh.baseline) == pytest.approx(direct, abs=1e-8)


def test_the_layered_reading_is_not_the_linear_one(world):
    """Two different models give two different readings, so the stage has to compute both."""
    _, _, mfs, rings, linear, layered = world
    apart = 0
    for mf, r in zip(mfs, rings):
        a = atom_shapley(linear, mf, groups=_players(mf, r), exact_limit=14).values
        b = atom_shapley(layered, mf, groups=_players(mf, r), exact_limit=14).values
        assert len(a) == len(b)
        if not np.allclose(a, b, atol=1e-6):
            apart += 1
    assert apart >= 1


def test_the_linear_models_two_decompositions_describe_one_prediction(world):
    """Rule 3 and the Shapley values of the linear model are different decompositions, and both
    add up to that model's own prediction. The benchmark reports them side by side, so this holds
    them to that."""
    _, _, mfs, rings, linear, _ = world
    for mf, r in zip(mfs, rings):
        per_ring, _contrib, lost = pbh.ring_contributions(linear, mf, r)
        yp = float(linear.predict(AtomTable.from_molecules([mf]))[0])
        assert lost == 0.0
        assert float(per_ring.sum() + linear.bias) == pytest.approx(yp, abs=1e-8)
        sh = atom_shapley(linear, mf, groups=_players(mf, r), exact_limit=14)
        assert float(sh.values.sum() + sh.baseline) == pytest.approx(yp, abs=1e-8)


def test_the_wider_grid_brackets_the_linear_limit_and_is_stored_apart():
    """PBH6's first open question was a grid that stopped at its own edge.

    As the width goes to zero the layer tends to the linear kernel, so a grid whose smallest
    point is the chosen one says nothing about where the best width is. The wider grid reaches
    three points below that point, far enough to show a turn if there were one, and it is a
    measurement of its own: it is stored under another name, so running it cannot overwrite the
    default grid's result or be mistaken for it. The measurement found no better width there, so
    the default grid keeps its edge.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import run_compas
    g, w = run_compas.GAMMAS, run_compas.WIDE_GAMMAS
    assert list(g) == sorted(g) and list(w) == sorted(w)
    assert all(x > 0 for x in w)
    assert set(g) < set(w)                      # the wider grid keeps every default point
    assert min(g) == 0.25                       # the point chosen on nine of ten searches
    assert min(w) <= min(g) / 4                 # and the wider grid reaches well below it
    assert (run_compas.results_path("1D", "resolution")
            != run_compas.results_path("1D", "resolution_wide"))
    assert "ringshapley" in run_compas.STAGES


def test_the_wider_grid_is_reachable_from_the_command_line():
    """A stored result needs one documented command that reproduces it (rule G6)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    import run_compas
    with pytest.raises(SystemExit):
        run_compas.main(["--help"])              # the parser is built without error
    import inspect
    assert "wide" in inspect.signature(run_compas.resolution).parameters
    src = inspect.getsource(run_compas.main)
    assert "--wide-gammas" in src and "wide=args.wide_gammas" in src
def test_the_linear_models_shapley_values_are_rule_3_over_the_same_players(world):
    """The claim the report makes, which keeps the comparison in that table honest.

    A model linear in the field is additive over the atoms, so a group of atoms has no
    interaction with the rest and its Shapley value is just the sum of its own shares under
    decomposition rule 3. The stored per-ring decomposition differs from the Shapley reading only
    because rule 3 can give half of a shared atom to each of two rings while a player cannot be
    shared, and this test separates the two: over the same players the numbers are the same.
    """
    _, _, mfs, rings, linear, _ = world
    for mf, r in zip(mfs, rings):
        players = _players(mf, r)
        sh = atom_shapley(linear, mf, groups=players, exact_limit=14)
        _per_ring, contrib, _lost = pbh.ring_contributions(linear, mf, r)
        by_player = np.array([float(np.sum(np.asarray(contrib)[g])) for g in players])
        assert np.max(np.abs(by_player - sh.values)) < 1e-10


def test_cancellation_counts_how_much_a_reading_cancels():
    """One when the values add up without cancelling, larger when they fight each other."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
    from run_compas import _cancellation
    assert _cancellation([1.0, 2.0], prediction=3.0, baseline=0.0) == pytest.approx(1.0)
    assert _cancellation([1.0, 2.0], prediction=5.0, baseline=2.0) == pytest.approx(1.0)
    assert _cancellation([5.0, -4.0], prediction=1.0, baseline=0.0) == pytest.approx(9.0)
    assert np.isnan(_cancellation([1.0, -1.0], prediction=2.0, baseline=2.0))
