"""Tests for benchmarks/run_interphase.py: part (a) of plan item BT13.

The pairs of species and the potentials are formed from the graphs of the species. The tests
check what a mistake would spoil without any sign: the key must not depend on the numbering of
the atoms, the sign of a reduction and of an oxidation must be right, only closed-shell reactants
may be paired, and the models must be fitted without the held-out group.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import run_interphase as RI                                                  # noqa: E402

#: ethylene carbonate with its hydrogen atoms
EC = (["C", "O", "O", "O", "C", "C"] + ["H"] * 4,
      [[0, 1], [0, 2], [0, 3], [2, 4], [3, 5], [4, 5], [4, 6], [4, 7], [5, 8], [5, 9]])


def renumber(species, bonds, perm):
    """The same graph with atom i called perm[i]."""
    new = [None] * len(species)
    for i, p in enumerate(perm):
        new[p] = species[i]
    return new, [[perm[a], perm[b]] for a, b in bonds]


def test_the_graph_key_does_not_depend_on_the_numbering_of_the_atoms():
    sp, bonds = EC
    rng = np.random.default_rng(1)
    key = RI.graph_key(sp, bonds)
    for _ in range(5):
        perm = [int(v) for v in rng.permutation(len(sp))]
        assert RI.graph_key(*renumber(sp, bonds, perm)) == key


def test_the_graph_key_tells_two_isomers_apart():
    sp, bonds = EC
    # the open chain with the same formula: C-C-O-C(=O)-O with a ring bond removed
    chain = [b for b in bonds if b != [3, 5]]
    assert RI.graph_key(sp, chain) != RI.graph_key(sp, bonds)


def test_atoms_that_the_connectivity_cannot_tell_apart_share_a_label():
    sp, bonds = EC
    lab = RI.atom_labels(sp, bonds)
    assert lab[6] == lab[7] == lab[8] == lab[9]          # the four hydrogen atoms of the ring
    assert lab[2] == lab[3]                              # the two ring oxygen atoms
    assert lab[1] != lab[2]                              # the carbonyl oxygen is another atom


def test_the_potential_has_the_right_sign_and_reference():
    # a reduction is easier (higher potential) when the product has the lower free energy
    easy = RI.potential(-10.0, -11.0, "reduction")
    hard = RI.potential(-10.0, -9.0, "reduction")
    assert easy > hard
    assert easy == pytest.approx(1.0 - RI.LI_REFERENCE_V)
    # an oxidation is easier (lower potential) when the product has the lower free energy
    assert RI.potential(-10.0, -11.0, "oxidation") < RI.potential(-10.0, -9.0, "oxidation")
    assert RI.potential(-10.0, -9.0, "oxidation") == pytest.approx(1.0 - RI.LI_REFERENCE_V)
    assert RI.LI_REFERENCE_V == pytest.approx(1.40)


def _record(sid, charge, mult, g, species=None, bonds=None):
    sp, bd = species or EC[0], bonds or EC[1]
    return {"id": sid, "charge": charge, "mult": mult, "species": sp, "xyz": [[0, 0, 0]] * len(sp),
            "bonds": bd, "g": {RI.FREE_ENERGY: g, "shifted_rrho_eV": g, "eV": g},
            "resp": None, "mulliken": None, "spin": None}


def test_only_a_closed_shell_reactant_is_paired_with_the_radical_ion_of_its_graph():
    recs = [_record("a", 0, 1, -10.0), _record("b", -1, 2, -11.0), _record("c", 1, 2, -9.0),
            _record("d", -1, 1, -5.0)]       # a closed-shell anion of the same graph: no pair
    prs = RI.pairs(recs)
    kinds = {(p["kind"], p["id"], p["partner"]) for p in prs}
    assert ("reduction", "a", "b") in kinds
    assert ("oxidation", "a", "c") in kinds
    assert all(p["id"] == "a" for p in prs)           # 'd' has no radical partner at charge -2
    red = [p for p in prs if p["kind"] == "reduction"][0]
    assert red["E"] == pytest.approx(RI.potential(-10.0, -11.0, "reduction"))


def test_the_lowest_free_energy_of_a_graph_charge_and_spin_is_kept():
    recs = [_record("a", 0, 1, -10.0), _record("b1", -1, 2, -10.5), _record("b2", -1, 2, -11.0)]
    prs = RI.pairs(recs)
    assert [p["partner"] for p in prs if p["kind"] == "reduction"] == ["b2"]


def test_a_reduction_that_breaks_a_bond_is_not_a_pair():
    sp, bonds = EC
    opened = [b for b in bonds if b != [3, 5]]
    recs = [_record("a", 0, 1, -10.0), _record("b", -1, 2, -11.0, sp, opened)]
    assert RI.pairs(recs) == []


def test_families_are_given_in_the_order_phosphorus_sulfur_fluorine_carbonate_other():
    sp, bonds = EC
    assert RI.family_of(sp, bonds) == "carbonate"
    assert RI.family_of(["P"] + ["F"] * 6, [[0, i] for i in range(1, 7)]) == "phosphorus"
    assert RI.family_of(["S", "O", "O", "F", "C", "F", "F"], []) == "sulfur"
    assert RI.family_of(["C", "F", "H", "H", "H"], [[0, 1], [0, 2], [0, 3], [0, 4]]) == "fluorinated"
    assert RI.family_of(["C", "H", "H", "H", "H"], [[0, 1], [0, 2], [0, 3], [0, 4]]) == "other"


def test_the_spin_of_the_radical_ion_is_averaged_over_equivalent_atoms_whatever_its_numbering():
    sp, bonds = EC
    perm = [5, 4, 3, 2, 1, 0, 9, 8, 7, 6]
    sp2, bonds2 = renumber(sp, bonds, perm)
    spin = np.zeros(len(sp2))
    spin[perm[1]] = 0.6                              # the carbonyl oxygen
    spin[perm[8]] = 0.4                              # one ring hydrogen
    species = {
        "r": {"species": sp, "xyz": np.random.default_rng(0).normal(size=(10, 3)).tolist(),
              "bonds": bonds, "resp": [0.1] * 10, "charge": 0, "mult": 1},
        "q": {"species": sp2, "xyz": [[0, 0, 0]] * 10, "bonds": bonds2, "spin": spin.tolist(),
              "resp": [0.0] * 10, "charge": -1, "mult": 2},
    }
    xtb = {"r": {"fukui_plus": [0.1] * 10, "fukui_minus": [0.2] * 10}}
    row = {"id": "r", "partner": "q"}
    _, cols, _ = RI.species_table(row, species, xtb, "reduction")
    s = cols[:, RI.COLUMNS.index("spin")]
    assert s[1] == pytest.approx(0.6)                # the carbonyl oxygen keeps its population
    assert s[6:].tolist() == pytest.approx([0.1] * 4)   # the 0.4 is shared by the four hydrogens
    assert cols[:, RI.COLUMNS.index("fukui")].tolist() == pytest.approx([0.1] * 10)
    _, cols, _ = RI.species_table(row, species, xtb, "oxidation")
    assert cols[:, RI.COLUMNS.index("fukui")].tolist() == pytest.approx([0.2] * 10)


def test_the_nested_fit_uses_no_row_of_the_held_out_group_and_finds_a_line():
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=(n, 1))
    y = 2.0 * x[:, 0] + 0.5
    K = x @ x.T + 1.0                       # the constant gives the line an intercept
    groups = [str(i // 4) for i in range(n)]
    fit = RI.nested_ridge({"line": K}, y, groups, folds=5, inner=3, seed=0)
    assert np.all(np.isfinite(fit["predicted"]))
    assert np.max(np.abs(fit["predicted"] - y)) < 0.05
    # a label that no descriptor knows is not predicted from the other groups
    y2 = rng.normal(size=n)
    fit2 = RI.nested_ridge({"line": K}, y2, groups, folds=5, inner=3, seed=0)
    assert np.mean(np.abs(fit2["predicted"] - y2)) > 0.5 * np.mean(np.abs(y2 - y2.mean()))


def test_the_rank_correlation_of_a_monotone_map_is_one():
    a = np.array([1.0, 3.0, 2.0, 5.0])
    assert RI.spearman(a, np.exp(a)) == pytest.approx(1.0)
    assert RI.spearman(a, -a) == pytest.approx(-1.0)
