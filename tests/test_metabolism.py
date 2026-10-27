"""Which atom does the enzyme attack (plan item MC8): the baseline and the two measures.

The model here is the one plan items LP2 and LP5 already built; what is new is the count over
bonds that the item names as the reference, and the two measures the site-of-metabolism
literature uses.  These tests fix what those three pieces mean: the count over bonds counts over
the bond graph and nothing else, the top-k measure is per molecule and not per atom, and the area
under the curve agrees with its definition as a probability.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

from cmfqsar.atomtargets import AtomTargets  # noqa: E402
from cmfqsar.localenv import (BOND_SHELLS, topological_counts,  # noqa: E402
                              topological_features, type_index)

import run_metabolism as rm  # noqa: E402

TYPES = ["C.3", "C.ar", "O.3", "O.2", "N.am", "N.ar"]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    return m


def _targets(values_per_molecule):
    """AtomTargets built straight from the numbers, without a file."""
    mol_i, atom_i, vals = [], [], []
    for k, v in enumerate(values_per_molecule):
        for i, x in enumerate(v):
            if np.isfinite(x):
                mol_i.append(k)
                atom_i.append(i)
                vals.append(float(x))
    return AtomTargets(np.asarray(mol_i, int), np.asarray(atom_i, int), np.asarray(vals, float),
                       len(values_per_molecule), "som",
                       [np.asarray(v, float) for v in values_per_molecule])


# --------------------------------------------------------------- the count over bonds


def test_the_row_is_the_centre_and_four_shells_of_counts():
    mol = _mol("CCO")
    row = topological_counts(mol, 0, TYPES)
    assert row.shape == ((len(TYPES) + 1) * (1 + BOND_SHELLS),)
    n_t = len(TYPES) + 1
    assert row[:n_t].sum() == 1.0, "the centre is one atom"


def test_the_centre_block_names_the_centre():
    mol = _mol("CCO")
    idx = type_index(TYPES)
    n_t = len(TYPES) + 1
    assert topological_counts(mol, 2, TYPES)[idx["O.3"]] == 1.0
    assert topological_counts(mol, 0, TYPES)[idx["C.3"]] == 1.0
    # and nothing else is in that block
    assert topological_counts(mol, 2, TYPES)[:n_t].sum() == 1.0


def test_it_counts_over_bonds_and_not_over_space():
    """The two ends of butane are four bonds apart whatever the conformation does."""
    n_t = len(TYPES) + 1
    rows = []
    for seed in (1, 2, 3, 4, 5):
        mol = _mol("CCCC", seed=seed)
        rows.append(topological_counts(mol, 0, TYPES))
    for r in rows[1:]:
        assert np.array_equal(rows[0], r), "a conformation changed a topological count"
    # the far carbon of butane sits in the three-bond shell, with the hydrogens of C2 in it too
    idx = type_index(TYPES)
    assert rows[0][n_t * 3 + idx["C.3"]] == 1.0


def test_hydrogens_land_in_the_unknown_column():
    """A type not in the list is counted, but in the one column kept for everything else."""
    mol = _mol("C")
    row = topological_counts(mol, 0, TYPES)
    n_t = len(TYPES) + 1
    assert row[n_t * 1 + len(TYPES)] == 4.0, "four hydrogens one bond from the carbon"


def test_the_shell_count_is_honoured():
    mol = _mol("CCCCCCCC")
    short = topological_counts(mol, 0, TYPES, shells=1)
    long_ = topological_counts(mol, 0, TYPES, shells=6)
    n_t = len(TYPES) + 1
    assert short.shape == (n_t * 2,)
    assert long_.shape == (n_t * 7,)
    assert np.array_equal(short, long_[:n_t * 2]), "the near shells must not change"
    assert long_[n_t * 6:].sum() > 0, "something is six bonds from the first carbon"


def test_the_feature_matrix_follows_the_row_order_of_the_targets():
    mols = [_mol("CCO"), _mol("CCN")]
    values = [[0.0, 1.0, np.nan] + [np.nan] * 20, [np.nan, 0.0, 1.0] + [np.nan] * 20]
    values = [np.asarray(v[:m.GetNumAtoms()], float) for v, m in zip(values, mols)]
    t = _targets(values)
    X = topological_features(mols, t, TYPES)
    assert X.shape[0] == len(t)
    for r in range(len(t)):
        want = topological_counts(mols[int(t.molecule[r])], int(t.atom[r]), TYPES)
        assert np.array_equal(X[r], want)


# ------------------------------------------------------------------------ the measures


def test_top_k_asks_one_question_per_molecule():
    """Two molecules of five atoms each; the ranking is inside a molecule, never across them."""
    values = [np.asarray([1.0, 0.0, 0.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0, 0.0, 1.0])]
    t = _targets(values)
    # the first molecule is ranked right, the second is not
    pred = np.asarray([9.0, 1.0, 0.0, 0.0, 0.0] + [0.0, 9.0, 8.0, 1.0, 0.5])
    assert rm.per_molecule_hits(t, pred, 1).tolist() == [1.0, 0.0]
    assert rm.per_molecule_hits(t, pred, 2).tolist() == [1.0, 0.0]
    assert rm.per_molecule_hits(t, pred, 3).tolist() == [1.0, 0.0]
    assert rm.per_molecule_hits(t, pred, 5).tolist() == [1.0, 1.0]


def test_a_large_prediction_in_one_molecule_cannot_help_another():
    """The scale of a molecule's predictions drops out, because each is ranked on its own."""
    values = [np.asarray([1.0, 0.0, 0.0]), np.asarray([1.0, 0.0, 0.0])]
    t = _targets(values)
    pred = np.asarray([0.01, 0.00, -0.01, 100.0, 99.0, 98.0])
    assert rm.per_molecule_hits(t, pred, 1).tolist() == [1.0, 1.0]


def test_a_molecule_with_no_site_is_not_counted():
    values = [np.asarray([1.0, 0.0, 0.0]), np.asarray([0.0, 0.0, 0.0])]
    t = _targets(values)
    hits = rm.per_molecule_hits(t, np.arange(6.0), 1)
    assert len(hits) == 1, "a molecule the set says nothing about asks no question"


def test_the_area_under_the_curve_is_the_probability_of_ranking_a_site_higher():
    rng = np.random.RandomState(0)
    y = (rng.rand(300) < 0.2).astype(float)
    pred = y * 0.8 + rng.rand(300)
    pos, neg = pred[y > 0], pred[y == 0]
    by_hand = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
    assert rm.auc(y, pred) == pytest.approx(by_hand, abs=2e-3)


def test_a_perfect_and_a_reversed_ranking():
    y = np.asarray([0.0, 0.0, 1.0, 1.0])
    assert rm.auc(y, np.asarray([0.0, 1.0, 2.0, 3.0])) == pytest.approx(1.0)
    assert rm.auc(y, np.asarray([3.0, 2.0, 1.0, 0.0])) == pytest.approx(0.0)
    assert np.isnan(rm.auc(np.zeros(4), np.arange(4.0)))


def test_the_paired_interval_brackets_the_difference_it_reports():
    rng = np.random.RandomState(1)
    a = (rng.rand(200) < 0.7).astype(float)
    b = (rng.rand(200) < 0.5).astype(float)
    point, lo, hi = rm.paired(a, b, seed=0, runs=400)
    assert point == pytest.approx(a.mean() - b.mean())
    assert lo < point < hi
    same = rm.paired(a, a, seed=0, runs=400)
    assert same[0] == 0.0 and same[1] == 0.0 and same[2] == 0.0


# ------------------------------------------------------------------ the whole benchmark


@pytest.mark.skipif(not os.path.exists(rm.DATA), reason="the metabolism set is not built")
def test_the_benchmark_runs_and_reports(tmp_path):
    out = rm.run(rhos=(0.25,), folds=3, seed=0, max_molecules=12, results=str(tmp_path))
    assert out["n_molecules"] >= 10
    names = [r["name"] for r in out["rows"]]
    assert names[0].startswith("count over bonds")
    assert any(n.startswith("count and field") for n in names)
    for r in out["rows"]:
        assert 0.0 <= r["top1"] <= r["top2"] <= 1.0, r["name"]
    text = rm.report(str(tmp_path))
    assert "top-2" in text and "plan item MC8" in text
