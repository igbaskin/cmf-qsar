"""The worked case studies (plan item F1).

The cases themselves take seconds to minutes and live in `benchmarks/results/cases`; what is
tested here is that the script's own machinery is right -- the split, the grouping of atoms into
chemical pieces, and that one case runs from end to end and produces every section it promises.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

import case_studies as CS  # noqa: E402


# ------------------------------------------------------------------------------- the split


def test_the_test_set_spans_the_range_rather_than_one_end_of_it():
    y = np.linspace(4.0, 9.0, 24)
    rng = np.random.default_rng(0)
    y = y[rng.permutation(len(y))]
    tr, te = CS.split_by_rank(y, every=3)
    assert len(tr) + len(te) == len(y)
    assert not set(tr.tolist()) & set(te.tolist())
    # the held-out third covers almost the whole range, which a random third need not
    assert np.ptp(y[te]) > 0.9 * np.ptp(y)
    assert np.ptp(y[tr]) > 0.9 * np.ptp(y)


def test_every_molecule_is_in_exactly_one_part():
    for n in (6, 7, 22, 25, 33):
        y = np.arange(n, dtype=float)
        tr, te = CS.split_by_rank(y)
        assert sorted(tr.tolist() + te.tolist()) == list(range(n))


def test_a_larger_stride_holds_out_fewer():
    y = np.arange(30, dtype=float)
    assert len(CS.split_by_rank(y, every=3)[1]) > len(CS.split_by_rank(y, every=5)[1])


# ------------------------------------------------------------------ the chemical grouping


def _fields_of(smiles, name="m"):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cmfqsar.fields import FieldSet, build_fields
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = 1
    AllChem.EmbedMolecule(m, p)
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5)
    return build_fields(m, fs, name)


def test_the_players_are_rings_and_the_pieces_between_them():
    mol = _fields_of("c1ccccc1CC(=O)Nc1ccccc1Cl")
    groups, labels = CS._ring_groups(mol)
    assert len(groups) == len(labels)
    assert len(groups) >= 2                       # at least the two rings
    assert sum(len(g) for g in groups) == mol.n_atoms
    flat = [a for g in groups for a in g]
    assert sorted(flat) == list(range(mol.n_atoms))
    assert any(label.startswith("ring") for label in labels)


def test_the_grouping_stays_small_enough_to_be_exact():
    """Shapley values are exact up to fourteen players; beyond that the decomposition is sampled."""
    mol = _fields_of("c1ccc2c(c1)c1ccccc1c1ccccc21")
    groups, labels = CS._ring_groups(mol)
    assert len(groups) <= 14
    assert len(labels) == len(groups)


def test_a_molecule_without_a_molblock_is_still_grouped():
    mol = _fields_of("CCCCCCCCCCCC")
    mol.molblock = ""
    groups, labels = CS._ring_groups(mol)
    assert groups and len(groups) == len(labels)
    assert sum(len(g) for g in groups) == mol.n_atoms


# ----------------------------------------------------------------------------- the cases


def test_the_three_cases_name_files_that_exist():
    root = os.path.join(os.path.dirname(__file__), "..")
    assert len(CS.CASES) == 3
    for case in CS.CASES:
        assert os.path.exists(os.path.join(root, case.dataset)), case.dataset
        assert os.path.exists(os.path.join(root, case.pocket)), case.pocket
        assert case.target and case.why


def test_an_unknown_case_is_refused():
    with pytest.raises(SystemExit, match="unknown case"):
        CS.case_of("not-a-target")
    assert CS.case_of("mcl1").key == "mcl1"


@pytest.mark.slow
def test_one_case_runs_from_end_to_end(tmp_path, monkeypatch):
    """The smallest case, with a one-point grid, so that the whole chain is exercised quickly."""
    monkeypatch.setattr(CS, "ALPHAS", (0.5,))
    monkeypatch.setattr(CS, "LAMS", (0.1,))
    monkeypatch.setattr(CS, "FOLDS", 5)
    got = CS.run_case(CS.case_of("ptp1b"), str(tmp_path), want_report=False, verbose=False)
    for section in ("summary", "fields", "cofields", "shapley", "difference", "pocket"):
        assert section in got, section
    s = got["summary"]
    assert s["n"] == s["n_train"] + s["n_test"]
    assert np.isfinite(s["q2_cv"]) and np.isfinite(s["r2_pred"])
    assert np.isfinite(s["y_random_mean_q2"]) and 0.0 <= s["y_random_p"] <= 1.0
    assert set(got["cofields"]["per_field"]) == set(CS.FIELDS)
    for name, d in got["cofields"]["per_field"].items():
        assert 0.0 <= d["share_above_2sd"] <= 1.0, name
    assert {"best predicted", "worst predicted", "most active"} <= set(got["shapley"])
    for label, d in got["shapley"].items():
        assert d["players"] >= 1 and d["efficiency_error"] < 1e-6 or not d["exact"]
        assert d["top"] and abs(sum(t["share"] for t in d["top"])) <= 1.0 + 1e-9
    assert got["pocket"]["pocket_atoms"] > 100
    assert -1.0 <= got["pocket"]["spearman_with_activity"] <= 1.0
    assert os.path.exists(os.path.join(str(tmp_path), "ptp1b.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "ptp1b_pharmacophore.csv"))


def test_the_recorded_results_are_readable_and_say_what_the_table_says():
    """The benchmark's own files, so that a changed number cannot pass unnoticed."""
    import json
    root = os.path.join(os.path.dirname(__file__), "..")
    folder = os.path.join(root, "benchmarks", "results", "cases")
    if not os.path.isdir(folder):
        pytest.skip("the case studies have not been run in this checkout")
    table = os.path.join(folder, "table.md")
    assert os.path.exists(table)
    text = open(table, encoding="utf-8").read()
    for case in CS.CASES:
        p = os.path.join(folder, case.key, f"{case.key}.json")
        assert os.path.exists(p), p
        d = json.load(open(p))
        assert d["summary"]["case"] == case.key
        assert f"| {case.key} |" in text
    # the warning about a case that does not beat its own shuffle has to be in the table when
    # one of them does not
    weak = [case.key for case in CS.CASES
            if (lambda s: s["y_random_max_q2"] >= s["q2_cv"])(
                json.load(open(os.path.join(folder, case.key, f"{case.key}.json")))["summary"])]
    if weak:
        assert "did as well as the activities" in text
