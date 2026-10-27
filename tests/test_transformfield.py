"""A matched pair as a field placed in space (plan item MC6).

What these tests fix is the construction: the attachment point is found, the frame is the same
one for both members of a pair and does not depend on how the molecule was turned, the
difference field changes sign when the pair is read the other way round, and the kernel of two
difference fields is signed with it -- which is what makes a ridge without an intercept
antisymmetric.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.bioisostere import matched_pairs  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.model import krr_fit  # noqa: E402
from cmfqsar.transformfield import (attachment, attachment_frame, average_field,  # noqa: E402
                                    both_ways, by_transformation, increment_predict,
                                    increment_table, pair_field, posed, reversed_of, shared)

SERIES = [("chloro", "Clc1ccc(C(=O)NC2CC2)cc1", 6.0),
          ("methyl", "Cc1ccc(C(=O)NC2CC2)cc1", 5.4),
          ("fluoro", "Fc1ccc(C(=O)NC2CC2)cc1", 5.8),
          ("methoxy", "COc1ccc(C(=O)NC2CC2)cc1", 6.3)]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def _series(fields=("electrostatic", "steric")):
    mols = [_mol(s, seed=k + 1) for k, (_n, s, _y) in enumerate(SERIES)]
    names = [n for n, _s, _y in SERIES]
    y = np.array([v for _n, _s, v in SERIES])
    fs = FieldSet.create(physchem=list(fields), alpha=1.0)
    built = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    return mols, names, y, fs, built


def _first_pair():
    mols, names, y, fs, built = _series()
    pairs = matched_pairs(mols, y, names)
    assert pairs, "the series must have matched pairs"
    return mols, built, fs, pairs


# ------------------------------------------------------------- the attachment point


def test_the_cut_bond_is_found():
    mols, names, y, fs, built = _series()
    pairs = matched_pairs(mols, y, names)
    p = pairs[0]
    at = attachment(mols[p.a], p.atoms_a)
    assert at is not None
    core_atom, group_atom = at
    assert group_atom in set(p.atoms_a.tolist())
    assert core_atom not in set(p.atoms_a.tolist())
    assert mols[p.a].GetBondBetweenAtoms(core_atom, group_atom) is not None


def test_a_group_joined_by_two_bonds_is_not_an_attachment_point():
    """A ring cut in two places leaves two bonds, and there is then no single frame."""
    mol = _mol("C1CCCCC1")
    assert attachment(mol, [0, 1, 2]) is None


def test_the_frame_does_not_depend_on_how_the_molecule_is_turned():
    mols, built, fs, pairs = _first_pair()
    p = pairs[0]
    core_atom, group_atom = attachment(mols[p.a], p.atoms_a)
    origin, R = attachment_frame(mols[p.a], built[p.a], core_atom, group_atom)
    a = posed(built[p.a], p.atoms_a, origin, R)

    # turn the molecule and rebuild everything from the turned coordinates
    rng = np.random.RandomState(0)
    Q, _r = np.linalg.qr(rng.randn(3, 3))
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    turned = built[p.a].__class__(**{**built[p.a].__dict__})
    turned.coords = np.asarray(built[p.a].coords, float) @ Q.T + np.array([3.0, -2.0, 1.0])
    origin2, R2 = attachment_frame(mols[p.a], turned, core_atom, group_atom)
    b = posed(turned, p.atoms_a, origin2, R2)
    assert np.allclose(a.coords, b.coords, atol=1e-8)


def test_the_frame_puts_the_group_on_the_x_axis():
    mols, built, fs, pairs = _first_pair()
    p = pairs[0]
    core_atom, group_atom = attachment(mols[p.a], p.atoms_a)
    origin, R = attachment_frame(mols[p.a], built[p.a], core_atom, group_atom)
    moved = posed(built[p.a], p.atoms_a, origin, R)
    first = list(moved.member_atom_map).index(group_atom)
    r = np.asarray(moved.coords, float)[first]
    assert r[0] > 0 and abs(r[1]) < 1e-8 and abs(r[2]) < 1e-8


# -------------------------------------------------------------- the difference field


def test_a_pair_becomes_one_field_with_both_groups_in_it():
    mols, built, fs, pairs = _first_pair()
    t = pair_field(mols, built, pairs[0], "test")
    assert t is not None
    assert t.field.n_atoms == len(pairs[0].atoms_a) + len(pairs[0].atoms_b)
    assert t.delta == pytest.approx(pairs[0].delta)
    assert t.key == pairs[0].groups
    assert "test" in str(t)


def test_reading_the_pair_the_other_way_round_turns_everything_round():
    mols, built, fs, pairs = _first_pair()
    t = pair_field(mols, built, pairs[0], "test")
    r = reversed_of(t)
    assert r.delta == pytest.approx(-t.delta)
    assert r.key == (t.key[1], t.key[0])
    assert np.allclose(r.field.coords, t.field.coords)
    assert np.allclose(r.field.weights, -np.asarray(t.field.weights, float))
    assert np.allclose(reversed_of(r).field.weights, np.asarray(t.field.weights, float))


def test_the_kernel_of_a_reversed_pair_is_the_negative():
    """The overlap kernel is bilinear, which is what makes a model of pairs antisymmetric."""
    mols, built, fs, pairs = _first_pair()
    specs = [s for s in fs.specs if s.enabled]
    t = pair_field(mols, built, pairs[0], "test")
    u = pair_field(mols, built, pairs[1], "test")
    P = AtomTable.from_molecules([t.field, reversed_of(t).field, u.field])
    K = kernel_matrices(P, P, specs, symmetric=True)
    for f in range(len(specs)):
        assert K[f][0, 1] == pytest.approx(-K[f][0, 0], rel=1e-9)
        assert K[f][1, 2] == pytest.approx(-K[f][0, 2], rel=1e-9)


def test_a_ridge_without_an_intercept_predicts_the_opposite_for_the_opposite_pair():
    mols, built, fs, pairs = _first_pair()
    specs = [s for s in fs.specs if s.enabled]
    items = [pair_field(mols, built, p, "test") for p in pairs]
    items = [t for t in items if t is not None]
    P = AtomTable.from_molecules([t.field for t in items])
    K = np.asarray(kernel_matrices(P, P, specs, symmetric=True), float).sum(axis=0)
    y = np.array([t.delta for t in items])
    a, b = krr_fit(K, y, 1e-2, fit_bias=False)
    assert b == 0.0
    flipped = [reversed_of(t) for t in items]
    Q = AtomTable.from_molecules([t.field for t in flipped])
    Kq = np.asarray(kernel_matrices(Q, P, specs), float).sum(axis=0)
    assert np.allclose(Kq @ a, -(K @ a), atol=1e-9)


def test_both_ways_doubles_the_set_and_keeps_the_pairs_together():
    mols, built, fs, pairs = _first_pair()
    items = [t for t in (pair_field(mols, built, p, "test") for p in pairs) if t]
    doubled = both_ways(items)
    assert len(doubled) == 2 * len(items)
    assert sum(t.delta for t in doubled) == pytest.approx(0.0)


# ---------------------------------------------------------- averaging and the tables


def test_the_average_field_is_the_mean_of_its_members():
    mols, built, fs, pairs = _first_pair()
    items = [t for t in (pair_field(mols, built, p, "test") for p in pairs) if t][:3]
    avg = average_field(items, name="mean")
    assert avg.n_atoms == sum(t.field.n_atoms for t in items)
    assert avg.name == "mean"
    # a field is a sum of Gaussians, so the mean carries 1/k of every member's weight
    total = sum(float(np.abs(t.field.weights).sum()) for t in items)
    assert float(np.abs(avg.weights).sum()) == pytest.approx(total / len(items))


def test_averaging_nothing_is_refused():
    with pytest.raises(ValueError, match="no pairs"):
        average_field([])


def test_the_transformations_are_grouped_and_the_shared_ones_found():
    mols, built, fs, pairs = _first_pair()
    items = [t for t in (pair_field(mols, built, p, "one") for p in pairs) if t]
    other = [t for t in (pair_field(mols, built, p, "two") for p in pairs) if t]
    groups = by_transformation(items)
    assert sum(len(v) for v in groups.values()) == len(items)
    assert shared(items, least=2) == {}, "one series cannot share anything"
    both = shared(items + other, least=2)
    assert both and all(len(v) == 2 for v in both.values())


def test_the_increment_table_is_the_mean_change_of_a_transformation():
    mols, built, fs, pairs = _first_pair()
    items = [t for t in (pair_field(mols, built, p, "one") for p in pairs) if t]
    table = increment_table(items)
    key = items[0].key
    same = [t.delta for t in items if t.key == key]
    assert table[key] == pytest.approx(float(np.mean(same)))


def test_the_increment_prediction_knows_what_it_has_not_seen():
    mols, built, fs, pairs = _first_pair()
    items = [t for t in (pair_field(mols, built, p, "one") for p in pairs) if t]
    table = increment_table(items[:1])
    pred, seen = increment_predict(table, items)
    assert seen[0] and pred[0] == pytest.approx(items[0].delta)
    assert not all(seen), "a transformation it has never met must be marked"
    assert all(p == 0.0 for p, s in zip(pred, seen) if not s)


def test_a_transformation_met_the_other_way_round_is_predicted_with_the_opposite_sign():
    mols, built, fs, pairs = _first_pair()
    items = [t for t in (pair_field(mols, built, p, "one") for p in pairs) if t]
    table = increment_table([reversed_of(items[0])])
    pred, seen = increment_predict(table, [items[0]])
    assert seen[0] and pred[0] == pytest.approx(items[0].delta)


# ------------------------------------------------------------------- the benchmark


def test_the_benchmark_measures_what_it_says(tmp_path):
    sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
    import run_transformfield as rt
    y = np.array([1.0, -1.0, 2.0, -2.0])
    series = ["a", "a", "b", "b"]
    assert rt.q2_within(y, y, series) == pytest.approx(1.0)
    assert rt.q2_within(y, np.zeros(4), series) == pytest.approx(0.0)
    assert rt.q2_within(y, -y, series) < 0
