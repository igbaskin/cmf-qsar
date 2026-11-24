"""The continuous reaction field (plan item PO3).

What these tests fix is the construction: a reaction is read with its atoms in the order of the
map numbers, a reaction whose sides do not hold the same atoms is refused, the product is
superposed on the reactant without a reflection, an atom that does not change and does not move
cancels in the dynamic channel, and the two identities of the module hold to rounding:

(1) k(R, R') + k(P, P') = 2 k(C, C') + k(D, D') / 2, for the overlap kernel and for the rows of
    the invariant description;
(2) reversing a reaction negates its dynamic kernel row and leaves its conventional one, so a
    ridge without intercept predicts the opposite value for the reverse reaction.
"""
import os
import sys

import numpy as np
import pytest
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.fields import FieldSet  # noqa: E402
from cmfqsar.invariant import feature_matrix  # noqa: E402
from cmfqsar.model import krr_fit  # noqa: E402
from cmfqsar.reaction import (REACTION_SPEC, ReactionMismatch,  # noqa: E402
                              autocorrelation_rows, bond_change_counts, bond_changes,
                              bond_stretch, channel_norms, invariant_rows, mapped_mol,
                              overlap_kernels, reaction_centre, reaction_fields, read_reaction,
                              reversed_reaction, superpose, superposition_atoms)

#: name, reactant, product: every atom mapped, every hydrogen written
REACTIONS = [
    ("keto-enol",
     "[C:1]([H:4])([H:5])([H:6])[C:2](=[O:3])[H:7]",
     "[C:1]([H:4])([H:5])=[C:2]([O:3][H:6])[H:7]"),
    ("ring opening",
     "[C:1]1([H:5])([H:6])[C:2]([H:7])([H:8])[C:3]([H:9])=[C:4]1[H:10]",
     "[C:1]([H:5])([H:6])=[C:4]([H:10])[C:3]([H:9])=[C:2]([H:7])[H:8]"),
    ("dehydration",
     "[C:1]([H:4])([H:5])([H:6])[C:2]([H:7])([H:8])[O:3][H:9]",
     "[C:1]([H:4])([H:5])=[C:2]([H:7])[H:8].[O:3]([H:6])[H:9]"),
    ("imine-enamine",
     "[C:1]([H:4])([H:5])([H:6])[C:2]([H:7])=[N:3][H:8]",
     "[C:1]([H:4])([H:5])=[C:2]([H:7])[N:3]([H:6])[H:8]"),
]


def _coords(smiles, seed=1):
    mol = mapped_mol(smiles)
    assert AllChem.EmbedMolecule(mol, randomSeed=seed) == 0
    return np.array(mol.GetConformer().GetPositions(), float)


def _reaction(k):
    name, r, p = REACTIONS[k]
    return read_reaction(r, p, _coords(r, seed=k + 1), _coords(p, seed=k + 11), name)


def _fieldset():
    return FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5)


def _items(on="heavy"):
    fs = _fieldset()
    return fs, [reaction_fields(_reaction(k), fs, on=on) for k in range(len(REACTIONS))]


# ------------------------------------------------------------------- reading


def test_the_atoms_are_in_the_order_of_the_map_numbers():
    mol = mapped_mol("[O:3]=[C:2]([H:7])[C:1]([H:4])([H:5])[H:6]")
    assert [a.GetAtomMapNum() for a in mol.GetAtoms()] == [1, 2, 3, 4, 5, 6, 7]
    assert [a.GetSymbol() for a in mol.GetAtoms()] == ["C", "C", "O", "H", "H", "H", "H"]


def test_a_side_without_every_map_number_is_refused():
    with pytest.raises(ReactionMismatch):
        mapped_mol("[C:1]([H:4])([H:5])([H:6])[C:2](=O)[H:7]")
    with pytest.raises(ReactionMismatch):
        mapped_mol("[CH3:1][C:2](=[O:3])[H:7]")


def test_two_sides_with_different_atoms_are_refused():
    name, r, _p = REACTIONS[0]
    other = "[C:1]([H:4])([H:5])=[C:2]([N:3][H:6])[H:7]"          # map number 3 is N, not O
    with pytest.raises(ReactionMismatch, match="map number 3"):
        read_reaction(r, other, _coords(r), _coords(other), name)
    short = "[C:1]([H:4])([H:5])=[C:2]([H:6])[O:3]"
    with pytest.raises(ReactionMismatch):
        read_reaction(r, short, _coords(r), _coords(short), name)


def test_coordinates_in_another_order_are_seen():
    name, r, _p = REACTIONS[0]
    xyz = _coords(r)
    good = mapped_mol(r, xyz)
    assert bond_stretch(good) < 1.3
    bad = mapped_mol(r, xyz[::-1].copy())
    assert bond_stretch(bad) > 1.6


# ------------------------------------------------- the bonds that change


def test_the_changed_bonds_of_a_hydrogen_shift():
    rx = _reaction(0)
    got = {(i + 1, j + 1): d for i, j, d in bond_changes(rx)}
    # C1-H6 broken, O3-H6 made, C1=C2 raised, C2=O3 lowered
    assert got == {(1, 6): -1.0, (3, 6): 1.0, (1, 2): 1.0, (2, 3): -1.0}
    assert [a + 1 for a in reaction_centre(rx)] == [1, 2, 3, 6]


def test_the_count_reference_names_what_changes():
    counts, labels = bond_change_counts([_reaction(k) for k in range(len(REACTIONS))])
    row = dict(zip(labels, counts[0]))
    assert row["C-H broken"] == 1 and row["H-O made"] == 1
    assert row["C-C raised"] == 1 and row["C-O lowered"] == 1
    assert counts.shape == (len(REACTIONS), len(labels))
    # dehydration: C-O and C-H broken, H-O made, C-C raised
    row = dict(zip(labels, counts[2]))
    assert row["C-O broken"] == 1 and row["C-H broken"] == 1 and row["H-O made"] == 1


# ------------------------------------------------------- the superposition


def test_a_turned_product_comes_back():
    rx = _reaction(0)
    xyz = np.array(rx.reactant.GetConformer().GetPositions(), float)
    a = 0.7
    turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    same = read_reaction(REACTIONS[0][1], REACTIONS[0][1], xyz, xyz @ turn.T + [3.0, -1.0, 2.0])
    for on in ("heavy", "centre"):
        moved, rmsd = superpose(same, on)
        assert rmsd < 1e-10
        assert np.allclose(moved, xyz, atol=1e-9)


def test_the_superposition_keeps_the_handedness():
    rx = _reaction(1)
    xyz = np.array(rx.reactant.GetConformer().GetPositions(), float)
    mirror = read_reaction(REACTIONS[1][1], REACTIONS[1][1], xyz, xyz * [1.0, 1.0, -1.0])
    moved, _rmsd = superpose(mirror, "heavy")
    # the four carbons of cyclobutene are nearly planar, so they fit; the hydrogens above and
    # below the plane cannot be brought onto each other without a reflection
    assert not np.allclose(moved, xyz, atol=0.05)


def test_the_centre_rule_fits_on_the_atoms_that_change():
    rx = _reaction(0)
    assert [a + 1 for a in superposition_atoms(rx, "centre")] == [1, 2, 3, 6]
    assert [a + 1 for a in superposition_atoms(rx, "heavy")] == [1, 2, 3]
    with pytest.raises(ValueError):
        superposition_atoms(rx, "all")


# ------------------------------------------------------------ the channels


def test_an_unchanged_reaction_has_no_dynamic_field():
    name, r, _p = REACTIONS[0]
    xyz = _coords(r)
    fs = _fieldset()
    it = reaction_fields(read_reaction(r, r, xyz, xyz + 5.0, name), fs)
    K = overlap_kernels([it], fs)
    assert np.allclose(K["dynamic"], 0.0, atol=1e-10)
    assert np.allclose(K["conventional"], K["reactant"], rtol=1e-10)
    assert float(np.abs(it.bonds.weights).sum()) == 0.0
    assert np.allclose(channel_norms([it], fs), 0.0, atol=1e-5)


def test_identity_one_for_the_overlap_kernel():
    for on in ("heavy", "centre"):
        fs, items = _items(on)
        K = overlap_kernels(items, fs)
        sides = K["reactant"] + K["product"]
        channels = 2.0 * K["conventional"] + 0.5 * K["dynamic"]
        assert np.allclose(sides, channels, rtol=1e-10, atol=1e-10)
        # and the two channels are not the same thing as one side
        assert not np.allclose(K["dynamic"], 0.0, atol=1e-3)


def test_identity_one_for_the_invariant_rows():
    _fs, items = _items()
    rows = invariant_rows(items)
    assert np.allclose(rows["reactant"] + rows["product"],
                       2.0 * rows["conventional"] + 0.5 * rows["dynamic"],
                       rtol=1e-10, atol=1e-10)


def test_identity_two_reversal_negates_the_dynamic_kernel_row():
    fs, items = _items()
    back = [it.reversed() for it in items]
    K = overlap_kernels(items, fs, others=items)
    Kb = overlap_kernels(back, fs, others=items)
    assert np.allclose(Kb["dynamic"], -K["dynamic"], rtol=1e-10, atol=1e-10)
    assert np.allclose(Kb["bonds"], -K["bonds"], rtol=1e-10, atol=1e-10)
    assert np.allclose(Kb["conventional"], K["conventional"], rtol=1e-10, atol=1e-10)


def test_a_ridge_without_intercept_predicts_the_opposite_for_the_reverse():
    fs, items = _items()
    y = np.array([9.8, -12.5, 10.1, 3.9])                         # reaction energies
    K = overlap_kernels(items, fs)["dynamic"].sum(axis=0)
    K = K / np.mean(np.diag(K))
    dual, b = krr_fit(K, y, 1e-3, fit_bias=False)
    assert b == 0.0
    back = [it.reversed() for it in items]
    Kb = overlap_kernels(back, fs, others=items)["dynamic"].sum(axis=0) / np.mean(
        np.diag(overlap_kernels(items, fs)["dynamic"].sum(axis=0)))
    assert np.allclose(Kb @ dual, -(K @ dual), rtol=1e-9, atol=1e-9)


def test_the_invariant_rows_under_reversal():
    fs, items = _items()
    rows = invariant_rows(items)
    # reversed in the same frame, and rebuilt from the product side in the product's frame
    back = invariant_rows([it.reversed() for it in items])
    rebuilt = invariant_rows([reaction_fields(reversed_reaction(_reaction(k)), fs)
                              for k in range(len(REACTIONS))])
    for other in (back, rebuilt):
        assert np.allclose(other["odd"], -rows["odd"], rtol=1e-8, atol=1e-8)
        for even in ("conventional", "dynamic", "bonds"):
            assert np.allclose(other[even], rows[even], rtol=1e-6, atol=1e-6)
    # the odd row is the cross-correlation of C with D; it is not zero for a real reaction
    assert np.abs(rows["odd"]).max() > 1e-3


def test_the_odd_row_does_not_depend_on_the_superposition():
    _fs, heavy = _items("heavy")
    _fs, centre = _items("centre")
    a, b = invariant_rows(heavy), invariant_rows(centre)
    assert np.allclose(a["odd"], b["odd"], rtol=1e-9, atol=1e-9)
    # the even rows do, because the cross-correlation of the two sides is inside them
    assert not np.allclose(a["dynamic"], b["dynamic"], atol=1e-3)


def test_the_length_of_the_dynamic_channel_is_its_self_overlap():
    fs, items = _items()
    norms = channel_norms(items, fs)
    K = overlap_kernels(items, fs)["dynamic"]
    assert norms.shape == (len(items), 2)
    for f in range(2):
        assert np.allclose(norms[:, f] ** 2, np.diag(K[f]), rtol=1e-9)


def test_the_fast_rows_are_the_rows_of_the_library():
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic"],
                         cif_types=["C.3", "C.2", "H"], alpha=0.5)
    items = [reaction_fields(_reaction(k), fs) for k in range(len(REACTIONS))]
    for part in ("reactant", "dynamic", "conventional"):
        fields = [getattr(it, part) for it in items]
        slow = feature_matrix(fields, REACTION_SPEC)
        fast = autocorrelation_rows(fields, REACTION_SPEC)
        assert fast.shape == slow.shape
        assert np.allclose(fast, slow, rtol=1e-10, atol=1e-10)
