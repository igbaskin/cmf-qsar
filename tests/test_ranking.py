"""The ranking SVM: the pairs it forms, the order it learns, and what it refuses to report."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar.ranking import (concordant_pairs, fit_rank, make_pairs,  # noqa: E402
                             pair_kernel, rank_scores)


# --------------------------------------------------------------------------- the pairs

def test_pairs_are_ordered_most_active_first():
    p = make_pairs([3.0, 1.0, 2.0])
    assert len(p) == 3
    y = np.array([3.0, 1.0, 2.0])
    assert np.all(y[p[:, 0]] > y[p[:, 1]])


def test_the_gap_drops_pairs_too_close_to_call():
    y = [1.0, 1.2, 3.0]
    assert len(make_pairs(y)) == 3
    assert len(make_pairs(y, gap=0.5)) == 2          # the 1.0 / 1.2 pair goes
    assert len(make_pairs(y, gap=5.0)) == 0


def test_groups_keep_the_pairs_inside_a_source():
    y = [1.0, 2.0, 11.0, 12.0]
    g = ["a", "a", "b", "b"]
    p = make_pairs(y, groups=g)
    assert len(p) == 2
    assert all(g[int(i)] == g[int(j)] for i, j in p)


def test_a_missing_activity_takes_its_pairs_with_it():
    assert len(make_pairs([1.0, np.nan, 3.0])) == 1


def test_subsampling_is_reproducible():
    y = np.arange(40.0)
    a = make_pairs(y, max_pairs=50, seed=3)
    b = make_pairs(y, max_pairs=50, seed=3)
    c = make_pairs(y, max_pairs=50, seed=4)
    assert len(a) == 50 and np.array_equal(a, b) and not np.array_equal(a, c)


def test_the_pair_kernel_is_the_kernel_of_the_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    K = X @ X.T
    p = np.array([[0, 1], [2, 3]])
    Kp = pair_kernel(K, p)
    d = X[p[:, 0]] - X[p[:, 1]]
    assert np.allclose(Kp, d @ d.T)


# --------------------------------------------------------------------------- the fit

def test_a_linear_order_is_recovered_out_of_sample():
    rng = np.random.default_rng(1)
    X, w = rng.normal(size=(60, 5)), rng.normal(size=5)
    y = X @ w + rng.normal(scale=0.2, size=60)
    fit = fit_rank(X @ X.T, y, C=1.0)
    Xt = rng.normal(size=(40, 5))
    yt = Xt @ w + rng.normal(scale=0.2, size=40)
    assert rank_scores(yt, fit.score(Xt @ X.T))["spearman"] > 0.95


def test_the_fit_is_one_coefficient_per_molecule_and_sums_to_zero():
    """Every pair adds +a to one molecule and -a to another, so the weights cancel.

    That is what makes a ranking model a dual expansion like any other here, and so what lets
    the co-fields and the atom contributions go on working unchanged.
    """
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 4))
    y = X[:, 0] + rng.normal(scale=0.1, size=30)
    fit = fit_rank(X @ X.T, y, C=1.0)
    assert fit.coef.shape == (30,)
    assert abs(float(fit.coef.sum())) < 1e-8
    assert np.all(fit.alpha >= 0.0)


def test_a_constant_activity_gives_no_pairs_and_a_zero_model():
    fit = fit_rank(np.eye(6), np.ones(6), C=1.0)
    assert fit.n_pairs == 0 and np.allclose(fit.coef, 0.0)


def test_the_order_is_invariant_to_a_monotone_change_of_the_activity():
    """Fitting the order means the units cannot matter; a regression would not survive this."""
    rng = np.random.default_rng(3)
    X, w = rng.normal(size=(50, 4)), rng.normal(size=4)
    y = X @ w
    a = fit_rank(X @ X.T, y, C=1.0).score(X @ X.T)
    b = fit_rank(X @ X.T, np.exp(y / 2.0), C=1.0).score(X @ X.T)
    assert rank_scores(a, b)["spearman"] > 0.99


def test_groups_protect_the_model_from_an_offset_between_sources():
    """Two papers on the same chemistry, reported on different scales."""
    rng = np.random.default_rng(4)
    X, w = rng.normal(size=(80, 5)), rng.normal(size=5)
    truth = X @ w
    groups = np.array(["a"] * 40 + ["b"] * 40)
    reported = truth + np.where(groups == "a", -4.0, 4.0)       # a large offset, no signal
    K = X @ X.T
    blind = fit_rank(K, reported, C=1.0)
    grouped = fit_rank(K, reported, C=1.0, groups=groups)
    rho_grouped = rank_scores(truth, grouped.score(K))["spearman"]
    rho_blind = rank_scores(truth, blind.score(K))["spearman"]
    assert rho_grouped > rho_blind
    # the cross-group pairs are pure noise and there are more of them than the honest ones, so
    # the blind model is pulled by them; grouping leaves less than a tenth of the disorder
    assert (1.0 - rho_grouped) < 0.1 * (1.0 - rho_blind)


def test_concordant_pairs_counts_what_it_says():
    assert concordant_pairs([3.0, 2.0, 1.0], [3.0, 2.0, 1.0]) == 1.0
    assert concordant_pairs([3.0, 2.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    assert concordant_pairs([2.0, 1.0], [5.0, 5.0]) == 0.5       # a tie is half a point


# --------------------------------------------------------------------------- the model

def _small_set(n=24, seed=7):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import AtomTable, FieldSet, build_fields

    subs = ["C", "CC", "CCC", "O", "N", "Cl", "F", "Br", "C=O", "C#N", "OC", "NC"]
    smis = [f"c1ccccc1{s}" for s in subs] + [f"c1ccc(C)cc1{s}" for s in subs]
    mols = []
    for i, smi in enumerate(smis[:n]):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=seed + i)
        Chem.SanitizeMol(m)
        mols.append(m)
    fs = FieldSet.create(alpha=0.6)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    rng = np.random.default_rng(seed)
    y = np.array([m.GetNumHeavyAtoms() for m in mols], float) + rng.normal(scale=0.3, size=len(mols))
    return fs, mf, y, AtomTable


def test_the_model_reports_an_order_and_not_an_r2():
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    m = CMFModel(fieldset=fs, method="ranksvm", normalize=True, fit_bias=False,
                 params={"C": 1.0})
    m.fit(AtomTable.from_molecules(mf), y, cv_folds=6)
    assert not np.isfinite(m.stats.r2) and not np.isfinite(m.stats.q2)
    assert m.stats.rank_fit["spearman"] > 0.8
    assert "rho" in m.stats.summary() and "r2" not in m.stats.summary()
    assert set(m.stats.rank_cv) == {"spearman", "kendall", "pairs"}


def test_the_model_scores_a_test_set_by_its_order():
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    m = CMFModel(fieldset=fs, method="ranksvm", normalize=True, fit_bias=False,
                 params={"C": 1.0})
    m.fit(AtomTable.from_molecules(mf[:18]), y[:18], cv_folds=6)
    m.evaluate_test(AtomTable.from_molecules(mf[18:]), y[18:])
    assert m.stats.n_test == 6 and np.isfinite(m.stats.rank_test["spearman"])
    assert not np.isfinite(m.stats.r2_pred)


def test_a_ranking_model_still_has_a_co_field():
    """The expansion is over molecules, so everything that reads a linear model keeps working."""
    from cmfqsar.model import CMFModel

    fs, mf, y, AtomTable = _small_set()
    m = CMFModel(fieldset=fs, method="ranksvm", normalize=True, fit_bias=False,
                 params={"C": 1.0})
    m.fit(AtomTable.from_molecules(mf), y, cv_folds=6)
    centres, coeffs, widths = m.cofield_coefficients(0)
    assert len(centres) == len(coeffs) == len(widths) and np.any(coeffs != 0.0)


def test_the_search_selects_a_ranking_model_on_rho():
    from cmfqsar.model import CMFModel, cv_loss

    fs, mf, y, AtomTable = _small_set()
    m = CMFModel(fieldset=fs, method="ranksvm", normalize=True, fit_bias=False,
                 params={"C": 1.0})
    good = np.asarray(y, float)
    loss, quality = cv_loss(m, y, good)
    assert quality == pytest.approx(1.0) and loss == pytest.approx(0.0)
    loss_bad, quality_bad = cv_loss(m, y, -good)
    assert quality_bad == pytest.approx(-1.0) and loss_bad == pytest.approx(2.0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
