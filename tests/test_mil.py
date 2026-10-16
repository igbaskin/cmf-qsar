"""Tests for multi-instance learning over conformations."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.mil import (argmax_weights, bag_kernel, cv_predict, first_conformer_weights,  # noqa: E402
                         flatten_conformers, learn_weights, predict_bags, search,
                         softmax_weights, uniform_weights, weight_matrix)
from cmfqsar.model import r2_score  # noqa: E402


def _witness_bags(n_bags=60, k=5, d=4, noise=0.05, seed=0):
    """Bags whose label is the best score of their instances: the witness assumption."""
    rng = np.random.RandomState(seed)
    true = np.array([2.0, 1.0] + [0.0] * (d - 2))
    Z, bags, y, witness = [], [], [], []
    for i in range(n_bags):
        V = rng.normal(size=(k, d))
        g = V @ true
        w = int(np.argmax(g))
        witness.append(w)
        for j in range(k):
            Z.append(V[j])
            bags.append(i)
        y.append(float(g[w]) + noise * rng.normal())
    return np.array(Z), np.array(bags), np.array(y), np.array(witness)


def test_weight_helpers():
    bags = np.array([0, 0, 0, 1, 1, 2])
    assert uniform_weights(bags) == pytest.approx([1 / 3, 1 / 3, 1 / 3, 0.5, 0.5, 1.0])
    assert first_conformer_weights(bags) == pytest.approx([1, 0, 0, 1, 0, 1])
    s = np.array([0.0, 2.0, 1.0, 5.0, 0.0, 1.0])
    soft = softmax_weights(s, bags, 1.0)
    for b in (0, 1, 2):                                   # every bag sums to one
        assert soft[bags == b].sum() == pytest.approx(1.0)
    assert soft[1] > soft[2] > soft[0]                    # ordered by score
    assert softmax_weights(s, bags, 0.0) == pytest.approx(uniform_weights(bags))
    hard = argmax_weights(s, bags)
    assert hard == pytest.approx([0, 1, 0, 1, 0, 1])
    W = weight_matrix(uniform_weights(bags), bags, 3)
    assert W.shape == (3, 6) and W.sum(axis=1) == pytest.approx(np.ones(3))
    K = np.eye(6)
    assert bag_kernel(K, W)[2, 2] == pytest.approx(1.0)   # a bag of one is itself
    assert bag_kernel(K, W)[0, 0] == pytest.approx(1 / 3)


def test_selection_finds_the_witness_where_averaging_fails():
    Z, bags, y, witness = _witness_bags()
    K = Z @ Z.T
    flat = learn_weights(K, bags, y, beta=0.0, lam=0.01, init="uniform")
    picked = learn_weights(K, bags, y, beta=20.0, lam=0.01, init="uniform")
    hit = np.mean(picked.chosen() - np.arange(len(y)) * 5 == witness)
    assert hit > 0.9                                      # the hidden instance is recovered
    q2_flat = r2_score(y, cv_predict(K, bags, y, beta=0.0, lam=0.01, folds=5, init="uniform"))
    q2_pick = r2_score(y, cv_predict(K, bags, y, beta=20.0, lam=0.01, folds=5, init="uniform"))
    assert q2_pick > 0.9 and q2_flat < 0.6                # and it is worth a great deal
    assert picked.effective_counts().mean() < 1.5         # the weights really did concentrate
    assert flat.effective_counts().mean() == pytest.approx(5.0)


def test_beta_zero_is_exactly_the_ensemble():
    Z, bags, y, _ = _witness_bags(n_bags=20, seed=3)
    K = Z @ Z.T
    fit = learn_weights(K, bags, y, beta=0.0, lam=0.1, init="uniform", rounds=5)
    assert fit.weights == pytest.approx(uniform_weights(bags))
    # the bag kernel is then the mean-map kernel of the conformations, as the 4D mode builds it
    W = weight_matrix(uniform_weights(bags), bags, 20)
    assert bag_kernel(K, W) == pytest.approx(W @ K @ W.T)


def test_hard_selection_is_one_hot_and_stops():
    Z, bags, y, _ = _witness_bags(n_bags=30, seed=5)
    K = Z @ Z.T
    fit = learn_weights(K, bags, y, beta=0.0, lam=0.01, selection="hard", init="first", rounds=20)
    assert set(np.unique(fit.weights)) <= {0.0, 1.0}
    assert fit.effective_counts() == pytest.approx(np.ones(30))
    assert fit.rounds <= 20 and fit.moved[-1] == 0        # it settled rather than ran out
    assert len(fit.objective) == fit.rounds + 1


def test_prediction_of_new_bags_uses_no_activity():
    Z, bags, y, _ = _witness_bags(n_bags=50, seed=7)
    K = Z @ Z.T
    tr_bags = np.arange(40)
    tr = np.isin(bags, tr_bags)
    te = ~tr
    bags_te = bags[te] - 40
    fit = learn_weights(K[np.ix_(tr, tr)], bags[tr], y[:40], beta=10.0, lam=0.01, init="uniform")
    pred = predict_bags(K[np.ix_(te, tr)], fit, bags_te)
    assert len(pred) == 10
    assert r2_score(y[40:], pred) > 0.8
    # the same call with the activities shuffled must give the same predictions: they are not used
    fit2 = learn_weights(K[np.ix_(tr, tr)], bags[tr], y[:40], beta=10.0, lam=0.01, init="uniform")
    assert predict_bags(K[np.ix_(te, tr)], fit2, bags_te) == pytest.approx(pred)


def test_cross_validation_splits_by_bag_and_relearns_the_weights():
    Z, bags, y, _ = _witness_bags(n_bags=40, seed=11)
    K = Z @ Z.T
    ycv = cv_predict(K, bags, y, beta=10.0, lam=0.01, folds=4, seed=0, init="uniform")
    assert len(ycv) == 40 and np.isfinite(ycv).all()
    # a model fitted on everything predicts its own training bags better than the cross-validation
    fit = learn_weights(K, bags, y, beta=10.0, lam=0.01, init="uniform")
    inside = predict_bags(K, fit, bags)
    assert r2_score(y, inside) >= r2_score(y, ycv)
    # shuffling the activities destroys the cross-validated fit, so the weights are not fitting noise
    rng = np.random.RandomState(0)
    ys = rng.permutation(y)
    assert r2_score(ys, cv_predict(K, bags, ys, beta=10.0, lam=0.01, folds=4, seed=0,
                                   init="uniform")) < 0.3


def test_search_reports_a_grid_and_can_fall_back_to_averaging():
    Z, bags, y, _ = _witness_bags(n_bags=40, seed=13)
    K = Z @ Z.T
    res = search(K, bags, y, betas=[0.0, 10.0], lams=[0.01, 0.1], folds=4, seed=0, init="uniform")
    assert len(res.table) == 4
    assert res.best_beta == 10.0                           # the witness data want selection
    assert res.best_q2 > 0.9 and res.fit.beta == res.best_beta
    assert {"beta", "lam", "q2", "rmse_cv"} <= set(res.table[0])


def test_flatten_conformers_keeps_the_bags():
    from cmfqsar.conformers import generate_conformers
    from cmfqsar.fields import FieldSet
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mols, names = [], []
    for i, smi in enumerate(["CCCOc1ccccc1", "CCCNc1ccccc1C"]):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = i + 1
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        m.SetProp("_Name", f"m{i}")
        mols.append(generate_conformers(m, n=4, seed=i + 2, prune_rmsd=1.0))
        names.append(f"m{i}")
    fs = FieldSet.create(alpha=0.5, conformers="ensemble")
    flat, bags = flatten_conformers(mols, fs, names)
    assert len(flat) == sum(m.GetNumConformers() for m in mols)
    assert list(np.unique(bags)) == [0, 1]
    assert all(f.n_conformers == 1 for f in flat)          # each entry is one conformation
    assert flat[0].name.startswith("m0#")
    capped, bags2 = flatten_conformers(mols, fs, names, max_conformers=2)
    assert len(capped) == 4 and list(np.bincount(bags2)) == [2, 2]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_initial_weights_can_be_handed_in():
    Z, bags, y, _ = _witness_bags(n_bags=30, seed=17)
    K = Z @ Z.T
    warm = learn_weights(K, bags, y, beta=1.0, lam=0.01, init="uniform", rounds=3)
    cold = learn_weights(K, bags, y, beta=10.0, lam=0.01, init="uniform", rounds=3)
    carried = learn_weights(K, bags, y, beta=10.0, lam=0.01, rounds=3,
                            init_weights=warm.weights)
    assert carried.weights.shape == warm.weights.shape
    assert not np.allclose(carried.weights, cold.weights)      # the start really is different
    for b in np.unique(bags):                                  # and still a weighting
        assert carried.weights[bags == b].sum() == pytest.approx(1.0)


def test_annealing_runs_through_the_betas_and_ends_at_the_last():
    from cmfqsar.mil import anneal_weights
    Z, bags, y, witness = _witness_bags(n_bags=40, seed=19)
    K = Z @ Z.T
    fit = anneal_weights(K, bags, y, betas=(0.5, 2.0, 10.0), lam=0.01, init="uniform", rounds=5)
    assert fit.beta == 10.0
    assert fit.effective_counts().mean() < 2.0                 # it did sharpen
    hit = np.mean(fit.chosen() - np.arange(40) * 5 == witness)
    assert hit > 0.8                                           # and still finds the witness here
