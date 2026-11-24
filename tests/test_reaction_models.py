"""The folds and the search of the reaction benchmark (plan item PO3, stage 2).

What these tests fix: whole units are dealt into folds, a reaction and its reverse are in one
unit under every split, the reactant mean equals the plain mean when whole reactants are held
out (the audit of the folds), and the search finds the block that carries the target.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import run_reactions_models as rm  # noqa: E402


def _tables(n=240, seed=0):
    rng = np.random.RandomState(seed)
    reactant = rng.randint(0, 30, n)
    kind = rng.randint(0, 12, n)
    reverse = np.full(n, -1)
    for a, b in ((0, 1), (10, 57), (100, 200)):                   # three reverse pairs
        reverse[a], reverse[b] = b, a
    return dict(ea=rng.randn(n) * 20 + 80, dh=rng.randn(n) * 30, reactant=reactant, type=kind,
                reverse=reverse)


def test_a_reaction_and_its_reverse_are_one_unit_under_every_split():
    t = _tables()
    for split in rm.SPLITS:
        unit = rm.units(t, split)
        for k, r in enumerate(t["reverse"]):
            if r >= 0:
                assert unit[k] == unit[r]
        for seed in range(3):
            fold = rm.fold_of(unit, 5, seed)
            for k, r in enumerate(t["reverse"]):
                if r >= 0:
                    assert fold[k] == fold[r]


def test_whole_units_are_held_out_and_the_folds_are_of_similar_size():
    t = _tables()
    unit = rm.units(t, "reactant")
    fold = rm.fold_of(unit, 5, seed=1)
    for u in np.unique(unit):
        assert len(set(fold[unit == u])) == 1
    sizes = np.bincount(fold, minlength=5)
    assert sizes.min() > 0.7 * sizes.max()
    # another seed gives another assignment
    assert not np.array_equal(fold, rm.fold_of(unit, 5, seed=2))


def test_the_reactant_mean_is_the_plain_mean_when_reactants_are_held_out():
    t = _tables()
    fold = rm.fold_of(rm.units(t, "reactant"), 5, seed=0)
    simple, shared = rm.simple_references(t, fold)
    assert shared == 0
    for target in rm.TARGETS:
        assert np.allclose(simple[target]["reactant mean"], simple[target]["mean"])
    # with random folds the reactants are shared, and the two are not the same
    fold = rm.fold_of(rm.units(t, "random"), 5, seed=0)
    simple, shared = rm.simple_references(t, fold)
    assert shared > 0
    assert not np.allclose(simple["ea"]["reactant mean"], simple["ea"]["mean"])


def test_the_search_gives_the_weight_to_the_block_that_carries_the_target():
    rng = np.random.RandomState(0)
    n = 300
    good, noise = rng.randn(n, 2), rng.randn(n, 6)
    y = np.sin(good[:, 0]) + good[:, 1] ** 2

    def dist(X):
        d = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
        return d / d.mean()

    fit, val = np.arange(0, 220), np.arange(220, n)
    w, gamma, lam, rmse = rm.choose([dist(good), dist(noise)], y, fit, val, tied=[])
    assert w[0] > w[1]
    assert rmse < 0.5 * float(np.std(y[val]))
    assert gamma in rm.GAMMAS and lam in rm.LAMS
    # tied blocks keep one weight
    w, _g, _l, _r = rm.choose([dist(good), dist(noise)], y, fit, val, tied=[0, 1])
    assert w[0] == w[1]


def test_the_paired_interval_contains_zero_for_two_equal_models():
    rng = np.random.RandomState(1)
    y = rng.randn(400)
    a = y + 0.3 * rng.randn(400)
    groups = rng.randint(0, 40, 400)
    same = rm._paired(y, a, a.copy(), groups, runs=200)
    assert same["q2"] == [0.0, 0.0, 0.0]
    better = rm._paired(y, a, y + 0.05 * rng.randn(400), groups, runs=200)
    assert better["q2"][1] > 0 and better["mae"][2] < 0
