"""Which molecule to make next (plan item E2).

The arithmetic of expected improvement is checked against Monte Carlo, the rules are checked to
pick what they claim to pick, and the retrospective loop is checked to be a loop: it may look at
what it has acquired and at nothing else.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.active import (ACQUISITIONS, acquisition_scores, expected_improvement,  # noqa: E402
                            learning_curve, suggest)
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(Br)cc1O", "c1ccc(N)cc1C"]
Y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4])


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


@pytest.fixture(scope="module")
def fitted():
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(_mol(s), fs, f"m{i}") for i, s in enumerate(SMILES)]
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf[:8]), Y[:8], cv_folds=4, cv_seed=0)
    return model, table, mf


def test_expected_improvement_is_what_sampling_the_posterior_would_give():
    rng = np.random.default_rng(0)
    mu = np.array([0.5, 1.0, 2.0, -1.0])
    sd = np.array([0.3, 1.0, 0.2, 0.7])
    best = 1.0
    draws = rng.normal(mu[None, :], sd[None, :], size=(200000, len(mu)))
    mc = np.mean(np.maximum(draws - best, 0.0), axis=0)
    assert np.allclose(expected_improvement(mu, sd, best), mc, atol=0.01)


def test_expected_improvement_at_the_edges():
    ei = expected_improvement([2.0, 0.5], [0.0, 0.0], best=1.0)
    assert ei[0] == pytest.approx(1.0) and ei[1] == pytest.approx(0.0)
    assert np.all(expected_improvement([0.0], [1.0], best=5.0) > 0)       # never exactly zero
    wide = expected_improvement([1.0], [2.0], best=1.0)
    narrow = expected_improvement([1.0], [0.5], best=1.0)
    assert wide > narrow                   # uncertainty is worth something when it is free


def test_each_rule_picks_what_it_says_it_picks():
    mu = np.array([1.0, 3.0, 2.0])
    sd = np.array([2.0, 0.1, 0.5])
    dist = np.array([0.1, 0.2, 5.0])
    pick = lambda k, **kw: int(np.argmax(acquisition_scores(k, mu, sd, **kw)))  # noqa: E731
    assert pick("greedy") == 1
    assert pick("variance") == 0
    assert pick("maxmin", distance=dist) == 2
    assert pick("ucb", beta=0.0) == 1              # no optimism: the prediction alone
    assert pick("ucb", beta=5.0) == 0              # much optimism: the one it is unsure about
    assert pick("ei", best=2.5) == 1               # only one candidate can beat 2.5 on the mean
    assert pick("ei", best=6.0) == 0               # out of reach: the widest posterior wins
    a = acquisition_scores("random", mu, sd, rng=np.random.default_rng(3))
    b = acquisition_scores("random", mu, sd, rng=np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_the_rules_refuse_what_they_cannot_do():
    with pytest.raises(ValueError, match="unknown acquisition"):
        acquisition_scores("hunch", [1.0], [1.0])
    with pytest.raises(ValueError, match="best activity"):
        acquisition_scores("ei", [1.0], [1.0])
    with pytest.raises(ValueError, match="distance"):
        acquisition_scores("maxmin", [1.0], [1.0])


def test_suggest_ranks_real_candidates(fitted):
    model, table, _mf = fitted
    mu, var = model.predict_with_variance(table)
    sd = np.sqrt(np.maximum(var, 0.0))
    picks = suggest(model, table, kind="greedy", n=3)
    assert [p.index for p in picks][0] == int(np.argmax(mu))
    assert [p.score for p in picks] == sorted([p.score for p in picks], reverse=True)
    assert len({p.index for p in picks}) == 3
    assert suggest(model, table, kind="variance", n=1)[0].index == int(np.argmax(sd))
    assert all(np.isfinite([p.mean, p.sd, p.distance]).all() for p in picks)
    assert "predicted" in picks[0].describe("greedy")


def test_a_batch_does_not_pick_the_same_molecule_twice(fitted):
    """The candidate set here holds every molecule twice over."""
    model, _table, mf = fitted
    doubled = AtomTable.from_molecules(mf + mf)
    names = [f"m{i}" for i in range(len(mf))] * 2
    picks = suggest(model, doubled, kind="greedy", n=4, names=names, max_similarity=0.98)
    assert len({p.name for p in picks}) == len(picks)
    loose = suggest(model, doubled, kind="greedy", n=4, names=names, max_similarity=1.0)
    assert len({p.name for p in loose}) < len(loose)       # without the filter, duplicates


def test_a_classifier_is_refused(fitted):
    model, table, _mf = fitted
    model.task = "classification"
    try:
        with pytest.raises(ValueError, match="regression"):
            suggest(model, table, kind="ei")
    finally:
        model.task = "regression"


# --------------------------------------------------------------------------- #
# The retrospective loop
# --------------------------------------------------------------------------- #

def _world(n=60, d=5, noise=0.1, seed=0):
    """A kernel and activities that really are a function of the features it measures."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = X @ rng.normal(size=d) + rng.normal(scale=noise, size=n)
    return {1.0: X @ X.T}, y


def test_the_loop_acquires_one_molecule_at_a_time_and_never_the_same_one():
    kernels, y = _world()
    start = list(range(10))
    out = learning_curve(kernels, y, "ei", start, steps=15, seed=0)
    assert len(out["taken"]) == 15 and len(out["rmse"]) == 15
    assert len(set(out["taken"])) == 15
    assert not (set(out["taken"]) & set(start))
    assert out["have"][:10] == start
    assert np.all(np.diff(out["best"]) >= -1e-12)          # the best so far cannot fall


def test_the_loop_stops_when_the_pool_runs_out():
    kernels, y = _world(n=20)
    out = learning_curve(kernels, y, "variance", list(range(15)), steps=50, seed=0)
    assert len(out["taken"]) == 5


def test_looking_for_actives_finds_them_sooner_than_picking_at_random():
    """Averaged over starts, because any single run is a coin toss."""
    kernels, y = _world(n=80, seed=1)
    found = {}
    for kind in ("ei", "random"):
        best = []
        for seed in range(8):
            rng = np.random.default_rng(100 + seed)
            start = sorted(rng.choice(len(y), 12, replace=False).tolist())
            out = learning_curve(kernels, y, kind, start, steps=12, seed=seed)
            best.append(float(np.max(y[out["have"]])))
        found[kind] = float(np.mean(best))
    assert found["ei"] > found["random"]


def test_an_unknown_rule_stops_the_loop():
    kernels, y = _world(n=20)
    with pytest.raises(ValueError, match="unknown acquisition"):
        learning_curve(kernels, y, "intuition", list(range(10)), steps=2)


def test_every_rule_the_module_advertises_can_run_the_loop():
    kernels, y = _world(n=40)
    for kind in ACQUISITIONS:
        out = learning_curve(kernels, y, kind, list(range(10)), steps=3, seed=0)
        assert len(out["taken"]) == 3
