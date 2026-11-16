"""The HOSE-code reference of the local benchmark and the element channels that stand in for the
fields (`environment_ids`, `hose_lookup` and `element_environment` of benchmarks/run_local.py)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

RL = pytest.importorskip("run_local")


class _Targets:
    """The two index arrays `environment_ids` reads, for a hand-made list of atoms."""

    def __init__(self, molecule, atom):
        self.molecule = np.asarray(molecule, int)
        self.atom = np.asarray(atom, int)

    def __len__(self):
        return len(self.atom)


def _mol(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def test_equivalent_carbons_share_every_sphere_and_different_ones_do_not():
    # the two methyl carbons of propane are the same at every depth; the middle one is not
    m = _mol("CCC")
    ids = RL.environment_ids([m], _Targets([0, 0, 0], [0, 2, 1]), depth=4)
    assert ids.shape == (3, 5)
    assert (ids[0] == ids[1]).all()
    assert ids[0, 0] != ids[2, 0]          # a CH3 is not a CH2 even at radius 0


def test_the_deeper_sphere_separates_what_the_shallow_one_cannot():
    # the methyl carbons of ethanol and of propane are both CH3 bonded to a carbon, so they agree
    # at radius 0 and differ once the sphere reaches the oxygen
    ids = RL.environment_ids([_mol("CCO"), _mol("CCC")], _Targets([0, 1], [0, 0]), depth=3)
    assert ids[0, 0] == ids[1, 0]
    assert ids[0, 2] != ids[1, 2]


def test_the_lookup_reads_the_deepest_shared_sphere():
    # three training atoms; the test atom matches atom 0 down to radius 2 and atom 1 only at 0
    ids = np.array([[1, 10, 100],
                    [1, 11, 101],
                    [1, 12, 102],
                    [1, 10, 100]])
    y = np.array([5.0, 50.0, 500.0, -1.0])
    pred, depth = RL.hose_lookup(ids, y, [np.array([3])])
    assert depth[3] == 2 and pred[3] == pytest.approx(5.0)


def test_an_unseen_atom_falls_back_to_a_shallower_sphere_and_then_to_the_mean():
    ids = np.array([[1, 10], [1, 11], [1, 99], [7, 77]])
    y = np.array([2.0, 4.0, 0.0, 0.0])
    pred, depth = RL.hose_lookup(ids, y, [np.array([2, 3])])
    assert depth[2] == 0 and pred[2] == pytest.approx(3.0)     # mean of the two radius-0 matches
    assert depth[3] == -1 and pred[3] == pytest.approx(3.0)    # nothing shared: training mean


def test_a_test_atom_never_reads_its_own_value():
    ids = np.array([[1, 10], [1, 10], [1, 10]])
    y = np.array([1.0, 3.0, 1000.0])
    folds = [np.array([2]), np.array([0]), np.array([1])]
    first, _ = RL.hose_lookup(ids, y, folds)
    moved = y.copy()
    moved[2] = -1000.0
    second, _ = RL.hose_lookup(ids, moved, folds)
    assert first[2] == pytest.approx(second[2]) == pytest.approx(2.0)


def _toy(n_mol=12, per=3, seed=0):
    """A kernel, targets, molecule labels and folds for a small made-up set."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_mol * per, 4))
    K = np.exp(-0.5 * ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    y = X @ np.array([1.0, -2.0, 0.5, 0.0]) + 0.1 * rng.normal(size=len(X))
    molecule = np.repeat(np.arange(n_mol), per)
    folds = [np.where(np.isin(molecule, part))[0] for part in np.array_split(np.arange(n_mol), 4)]
    return K, y, molecule, folds


def test_the_whole_share_of_the_curve_is_the_ordinary_cross_validation():
    from cmfqsar.localenv import cross_validate
    from cmfqsar.atomtargets import AtomTargets

    K, y, molecule, folds = _toy()
    pred, _ = RL._subsampled_predictions(K, y, molecule, folds, 1.0, 1e-3,
                                         np.random.RandomState(0))
    targets = AtomTargets(molecule, np.zeros(len(y), int), y, int(molecule.max()) + 1, "t", [])
    targets.folds = lambda k=10, seed=0: folds
    ref = cross_validate(K, targets, lam=1e-3, folds=len(folds))
    assert np.allclose(pred, ref.predicted)


def test_a_smaller_share_drops_whole_molecules_and_keeps_the_test_part():
    K, y, molecule, folds = _toy()
    _pred, tr = RL._subsampled_predictions(K, y, molecule, folds, 0.5, 1e-3,
                                           np.random.RandomState(3))
    last_test = folds[-1]
    assert not set(tr) & set(last_test)
    kept = np.unique(molecule[tr])
    for m in kept:                                    # a kept molecule comes with all its atoms
        assert set(np.where(molecule == m)[0]) <= set(tr)
    assert len(kept) == 4                              # half of the 9 training molecules, rounded
    ids = np.stack([molecule % 3, molecule], axis=1)
    full, _ = RL.hose_lookup(ids, y, folds)
    same = RL._subsampled_lookup(ids, y, molecule, folds, 1.0, np.random.RandomState(0))
    assert np.allclose(full, same)


def test_element_channels_keep_the_geometry_and_the_envelope():
    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.localenv import environment
    from cmfqsar.triangles import envelope_factors

    m = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1O"))
    AllChem.EmbedMolecule(m, randomSeed=7)
    f = build_fields(m, FieldSet.create(), name="phenol")
    env = environment(f, 0, 0.5)
    el = RL.element_environment(env, 0.5)
    assert el.weights.shape == (env.n_atoms, len(RL.ELEMENT_CHANNELS) + 1)
    assert np.allclose(el.coords, env.coords)
    factor = envelope_factors(env, 0.5)
    for a, s in enumerate(env.symbols):
        col = list(RL.ELEMENT_CHANNELS).index(s) if s in RL.ELEMENT_CHANNELS else -1
        assert el.weights[a, col] == pytest.approx(factor[a])
        assert np.count_nonzero(el.weights[a]) == 1
