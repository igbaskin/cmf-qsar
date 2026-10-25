"""Scoring the things that claim to know when a prediction is bad (plan item E5)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.errorpred import (aurc, compare, estimates_for, risk_coverage,  # noqa: E402
                               score_predictor, table)
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402


def _errors(n=40, seed=0):
    """Errors whose size is known, so that an estimate can be right about them on purpose."""
    rng = np.random.default_rng(seed)
    size = np.linspace(0.1, 2.0, n)
    return size, size * rng.choice([-1.0, 1.0], n)


def test_the_curve_runs_from_the_confident_end_to_the_whole_set():
    size, err = _errors()
    cov, rm = risk_coverage(size, err, min_n=5)
    assert cov[0] == pytest.approx(5 / len(err)) and cov[-1] == pytest.approx(1.0)
    assert rm[-1] == pytest.approx(np.sqrt(np.mean(err ** 2)))
    assert np.all(np.diff(rm) > -1e-12)          # a perfect estimate never lowers the risk


def test_nothing_beats_the_oracle_and_reversing_it_is_worse_than_random():
    size, err = _errors()
    assert aurc(np.abs(err), err) <= aurc(size, err) + 1e-12
    assert score_predictor("oracle", np.abs(err), err).efficiency == pytest.approx(1.0)
    assert score_predictor("reversed", -np.abs(err), err).efficiency < 0.0


def test_an_estimate_that_knows_nothing_lands_near_zero():
    size, err = _errors(n=200, seed=1)
    rng = np.random.default_rng(2)
    s = score_predictor("noise", rng.normal(size=len(err)), err)
    assert abs(s.efficiency) < 0.25
    assert abs(s.spearman) < 0.25
    assert 0.7 < s.ratio < 1.4                   # the two halves are the same set, split at random


def test_an_estimate_that_is_the_same_everywhere_is_reported_as_such():
    size, err = _errors()
    s = score_predictor("flat", np.full(len(err), 1.75), err)
    assert s.constant and s.efficiency == 0.0
    assert s.rmse_trusted == s.rmse_doubted == pytest.approx(np.sqrt(np.mean(err ** 2)))
    assert np.isnan(s.spearman)
    assert "constant" in table([s])


def test_the_halves_are_what_acting_on_the_estimate_would_buy():
    size, err = _errors()
    s = score_predictor("size", size, err)
    assert s.rmse_doubted > s.rmse_trusted and s.ratio > 2.0
    assert s.spearman > 0.9


def test_too_few_molecules_to_say_anything_says_nothing():
    s = score_predictor("tiny", [1.0, 2.0, 3.0], [0.1, -0.2, 0.3])
    assert s.n == 3 and np.isnan(s.efficiency) and np.isnan(s.spearman)


def test_compare_ranks_the_estimates_by_efficiency():
    size, err = _errors(n=60, seed=3)
    rng = np.random.default_rng(4)
    scores = compare({"useless": rng.normal(size=len(err)), "oracle": np.abs(err),
                      "half-right": size + rng.normal(scale=0.4, size=len(err))},
                     np.zeros(len(err)), -err)
    assert [s.name for s in scores][0] == "oracle"
    assert scores[0].efficiency >= scores[1].efficiency >= scores[2].efficiency
    assert all(s.n == len(err) for s in scores)


# --------------------------------------------------------------------------- #
# What a fitted model can offer
# --------------------------------------------------------------------------- #

def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _fitted():
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
              "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
              "c1ccc(Br)cc1O", "c1ccc(N)cc1C", "c1ccc(F)cc1N", "c1ccc(Br)cc1N"]
    y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4, 1.3, 1.9])
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(_mol(s), fs, f"m{i}") for i, s in enumerate(smiles)]
    tr, te = list(range(10)), list(range(10, 14))
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1})
    train = AtomTable.from_molecules([mf[i] for i in tr])
    model.fit(train, y[tr], cv_folds=5, cv_seed=0)
    return model, AtomTable.from_molecules([mf[i] for i in te]), y[te]


def test_a_fitted_model_offers_the_variance_and_the_domain():
    model, test, _y = _fitted()
    est = estimates_for(model, test)
    assert set(est) == {"GP variance", "distance to training set", "domain dissimilarity"}
    assert all(len(v) == test.n_mol and np.isfinite(v).all() for v in est.values())


def test_the_normalised_conformal_width_is_the_domain_in_the_units_of_the_activity():
    """Worth a test because it is the reason the comparison has three answers and not four."""
    from scipy.stats import spearmanr

    model, test, y = _fitted()
    model.calibrate_conformal(level=0.8, normalise=True)
    est = estimates_for(model, test)
    assert "conformal width" in est
    rho = spearmanr(est["conformal width"], est["domain dissimilarity"]).statistic
    assert abs(rho) == pytest.approx(1.0)        # the same ordering, so the same efficiency
    scores = {s.name: s for s in compare(est, y, model.predict(test), min_n=2)}
    assert scores["conformal width"].efficiency == pytest.approx(
        scores["domain dissimilarity"].efficiency)


def test_the_unnormalised_conformal_width_orders_nothing():
    model, test, y = _fitted()
    model.calibrate_conformal(level=0.8, normalise=False)
    est = estimates_for(model, test)
    assert score_predictor("conformal width", est["conformal width"], y - model.predict(test),
                           min_n=2).constant
