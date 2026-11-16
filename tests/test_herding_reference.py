"""The references of the herding benchmark that need no field: max-min on fingerprints and a
spread over the number of heavy atoms (rules `fingerprints` and `size` of
benchmarks/run_herding.py)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

RH = pytest.importorskip("run_herding")

SMILES = ["CCO", "c1ccccc1", "CCCCCCCC", "CC(=O)O", "c1ccncc1", "CCN(CC)CC", "OCCO",
          "CCCCCCCCCCCC", "c1ccc2ccccc2c1", "CC(C)(C)O"]


def _mols():
    return [Chem.MolFromSmiles(s) for s in SMILES]


def test_the_fingerprint_pick_is_distinct_and_repeatable():
    mols = _mols()
    a = RH.fingerprint_maxmin(mols, 5, seed=3)
    b = RH.fingerprint_maxmin(mols, 5, seed=3)
    assert len(set(a.tolist())) == 5 and a.tolist() == b.tolist()
    assert all(0 <= i < len(mols) for i in a)


def test_the_size_spread_takes_the_middle_of_equal_parts():
    mols = _mols()
    heavy = np.array([m.GetNumHeavyAtoms() for m in mols])
    chosen = RH.size_spread(mols, 2)
    # sorted by size, the two parts of five; their middles are the 3rd and the 8th smallest
    order = np.argsort(heavy, kind="stable")
    assert chosen.tolist() == [order[2], order[7]]
    assert len(set(RH.size_spread(mols, len(mols)).tolist())) == len(mols)


def test_the_cheap_rules_run_in_the_splits_beside_the_others():
    assert RH.SUBSET_RULES[-1] == "random"
    assert set(RH.CHEAP_RULES) <= set(RH.SUBSET_RULES)
    assert not set(RH.CHEAP_RULES) & set(RH.RULES)          # the landmark stage is unchanged
    rec = RH.subsets_set("GPB", repeats=1)
    for m in RH.SIZES:
        got = {r["rule"] for r in rec["rows"] if r["m"] == m}
        assert set(RH.SUBSET_RULES) <= got
        for rule in RH.CHEAP_RULES:
            row = next(r for r in rec["rows"] if r["m"] == m and r["rule"] == rule)
            assert np.isfinite(row["rmse"]) and row["mmd2_pool"] >= -1e-12
