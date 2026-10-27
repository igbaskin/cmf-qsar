"""Growing a bound fragment into the gap the pocket leaves (plan item MC7).

What these tests fix is the geometry and the bookkeeping, which is where a growing tool goes
wrong without saying so: the growth vector points where the fragment has no bond, a group built
from its SMILES lands at a sensible bond length and turns about the new bond and nothing else,
the gap is what the pocket offers minus what the fragment already places, and the score counts
only where there is room.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.fields import FieldSet, MoleculeFields, build_fields  # noqa: E402
from cmfqsar.fraggrow import (CLASH, Candidate, Gap, addition, build_group,  # noqa: E402
                              clashes, enumerate_groups, gap, gap_points, grow, growth_vector,
                              rank_of, room, score, try_group, vacant_direction)


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _fieldset(fields=("electrostatic", "steric", "hydrophobic", "hba", "hbd")):
    return FieldSet.create(physchem=list(fields), alpha=1.0)


def _wall(x=6.0, n=5, spacing=2.0, channels=5, only=None):
    """A flat sheet of carbon atoms standing in for a pocket wall at x = `x`.

    `only` puts the weight in one channel, which is how a test isolates one kind of offer.
    """
    pts = []
    for i in range(n):
        for j in range(n):
            pts.append([x, (i - n // 2) * spacing, (j - n // 2) * spacing])
    coords = np.asarray(pts, float)
    k = len(coords)
    w = np.zeros((k, int(channels)))
    if only is None:
        w[:] = 0.1
    else:
        w[:, int(only)] = 0.1
    return MoleculeFields(name="wall", coords=coords, weights=w,
                          radii=np.full(k, 1.7), symbols=["C"] * k, atom_types=["C.3"] * k,
                          bonds=np.zeros((0, 2), int), charges=np.zeros(k))


# ------------------------------------------------------------- the growth vector


def test_the_vacant_direction_points_away_from_the_bonds_that_are_there():
    coords = np.array([[0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    d = vacant_direction(coords, 0, [1])
    assert np.allclose(d, [1.0, 0.0, 0.0])
    coords = np.array([[0.0, 0.0, 0.0], [-1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    d = vacant_direction(coords, 0, [1, 2])
    assert d[0] > 0.9 and abs(d[1]) < 1e-9


def test_the_growth_vector_replaces_a_hydrogen_of_the_atom_it_grows_from():
    mol = _mol("c1ccccc1C")
    carbon = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "C"
              and a.GetIsAromatic() and any(n.GetAtomicNum() == 1 for n in a.GetNeighbors())][0]
    v = growth_vector(mol, carbon)
    assert v is not None and v.hydrogen is not None
    coords = mol.GetConformer().GetPositions()
    to_h = coords[v.hydrogen] - coords[carbon]
    x = np.asarray(v.rotation, float)[0]
    assert float(to_h @ x / np.linalg.norm(to_h)) > 0.9, "x must point where the hydrogen was"


def test_a_frame_needs_a_heavy_neighbour():
    mol = _mol("C")
    assert growth_vector(mol, 0) is None, "methane has no heavy neighbour to fix a plane"


# ---------------------------------------------------------------- building a group


def test_a_group_is_built_from_its_smiles_in_its_own_frame():
    g = build_group("CC[*:1]")
    assert g is not None and g.heavy == 2
    assert g.local.shape[1] == 3
    first = g.local[g.first]
    assert first[0] > 0 and abs(first[1]) < 1e-6 and abs(first[2]) < 1e-6, \
        "the bond to the fragment lies on +x"
    assert g.mol.GetNumAtoms() == g.local.shape[0], "the dummy is gone, its position is not"


def test_a_smiles_without_exactly_one_attachment_point_is_refused():
    assert build_group("CC") is None
    assert build_group("[*:1]CC[*:1]") is None
    assert build_group("not a molecule") is None


def test_growing_makes_one_molecule_with_a_sensible_new_bond():
    mol = _mol("c1ccccc1")
    v = growth_vector(mol, 0)
    g = build_group("C[*:1]")
    grown = grow(mol, v, g, 0.0)
    assert grown is not None
    assert Chem.MolToSmiles(Chem.RemoveHs(grown)) == "Cc1ccccc1"
    coords = grown.GetConformer().GetPositions()
    base = mol.GetNumAtoms() - 1                     # one hydrogen was taken off
    d = float(np.linalg.norm(coords[base + g.first] - coords[v.atom]))
    assert 1.3 < d < 1.7, f"a carbon-carbon bond of {d:.2f} A is not a bond"


def test_turning_the_group_moves_it_about_the_new_bond_and_nothing_else():
    mol = _mol("c1ccccc1")
    v = growth_vector(mol, 0)
    g = build_group("CC[*:1]")
    a = grow(mol, v, g, 0.0).GetConformer().GetPositions()
    b = grow(mol, v, g, np.pi / 2).GetConformer().GetPositions()
    base = mol.GetNumAtoms() - 1
    assert np.allclose(a[:base], b[:base]), "the fragment must not move"
    assert np.allclose(a[base + g.first], b[base + g.first], atol=1e-8), \
        "the first atom is on the axis, so it stays"
    assert not np.allclose(a[base:], b[base:]), "the rest of the group must turn"


# -------------------------------------------------------------------- the gap


def test_the_room_is_open_at_a_contact_distance_and_closed_inside_the_wall():
    wall = _wall(x=0.0)
    pts = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.5, 0.0, 0.0], [8.0, 0.0, 0.0]])
    r = room(pts, wall)
    assert r[0] == 0.0 and r[1] == 0.0
    assert r[2] == pytest.approx(1.0)
    assert r[3] == pytest.approx(1.0)


def test_the_gap_keeps_only_the_points_in_front_with_room():
    mol = _mol("c1ccccc1")
    v = growth_vector(mol, 0)
    fs = _fieldset()
    pts = gap_points(v, reach=5.0, spacing=1.0)
    local = (pts - v.origin) @ np.asarray(v.rotation, float).T
    assert local[:, 0].min() > -1.01, "nothing behind the atom is asked about"
    g = gap(_wall(x=50.0), build_fields(mol, fs, "f"), fs, v, reach=5.0, spacing=1.0)
    assert len(g) > 0 and len(g.points) == len(g.room)
    assert set(g.reward) == {"electrostatic", "hydrophobic", "hbd", "hba"}, \
        "steric is not an offer: a protein gives space, not steric field"


def test_the_one_sided_offers_are_never_negative_and_the_fragment_can_use_them_up():
    mol = _mol("c1ccccc1C")
    v = growth_vector(mol, 0)
    fs = _fieldset()
    g = gap(_wall(x=3.5), build_fields(mol, fs, "f"), fs, v, reach=5.0, spacing=1.0)
    for name in ("hydrophobic", "hbd", "hba"):
        assert float(np.min(g.reward[name])) >= 0.0


def test_a_group_that_fills_the_gap_scores_more_than_one_that_does_not():
    """With one kind of offer and nothing else, more of the right field is worth more."""
    mol = _mol("c1ccccc1")
    v = growth_vector(mol, 0)
    fs = _fieldset(("hydrophobic",))
    wall = _wall(x=float(v.origin[0]) + 8.0, channels=1, only=0)
    g = gap(wall, build_fields(mol, fs, "f"), fs, v, reach=6.0, spacing=0.8)
    big = try_group(mol, v, build_group("CCCC[*:1]"), wall, fs, g)
    small = try_group(mol, v, build_group("C[*:1]"), wall, fs, g)
    assert big is not None and small is not None
    assert big.score > small.score > 0, "more hydrophobic atoms reach more of a hydrophobic gap"


# ------------------------------------------------------------------ the clashes


def test_a_group_driven_into_the_wall_is_counted_as_clashing():
    mol = _mol("c1ccccc1")
    v = growth_vector(mol, 0)
    fs = _fieldset()
    close = _wall(x=float(v.origin[0]) + 1.0)
    g = gap(close, build_fields(mol, fs, "f"), fs, v, reach=5.0, spacing=1.0)
    c = try_group(mol, v, build_group("CCCC[*:1]"), close, fs, g)
    assert c is not None and c.clashes > 0


def test_counting_clashes_looks_at_the_atoms_it_is_given():
    mol = _mol("c1ccccc1")
    coords = mol.GetConformer().GetPositions()
    wall = _wall(x=float(coords[0][0]))
    assert clashes(mol, wall) > 0
    far = _wall(x=float(coords[:, 0].max()) + 20.0)
    assert clashes(mol, far) == 0


# ------------------------------------------------------------- the enumeration


def test_the_enumeration_is_ordered_and_a_group_can_be_looked_up_in_it():
    mol = _mol("c1ccccc1")
    v = growth_vector(mol, 0)
    fs = _fieldset()
    wall = _wall(x=float(v.origin[0]) + 8.0)
    groups = [build_group(s) for s in ("C[*:1]", "CC[*:1]", "O[*:1]", "CCCC[*:1]")]
    got = enumerate_groups(mol, v, wall, fs, [g for g in groups if g], reach=6.0, spacing=0.8)
    assert len(got) == 4
    keys = [(c.clashes, -c.score) for c in got]
    assert keys == sorted(keys), "best first"
    assert rank_of(got, got[2].smiles) == 3
    assert rank_of(got, "nothing like it") is None


def test_the_addition_of_a_grown_compound_is_found():
    small = Chem.MolFromSmiles("c1ccccc1O")
    big = Chem.MolFromSmiles("Cc1ccccc1O")
    got = addition(small, big)
    assert got is not None
    smiles, atom, added = got
    assert smiles == "C[*:1]"
    assert len(added) == 1
    assert small.GetAtomWithIdx(atom).GetSymbol() == "C"


def test_a_molecule_that_is_not_contained_has_no_addition():
    assert addition(Chem.MolFromSmiles("c1ccncc1"), Chem.MolFromSmiles("Cc1ccccc1O")) is None


def test_an_addition_that_hangs_on_two_bonds_is_not_a_growth():
    """A ring fused onto the fragment is not something one bond can add."""
    small = Chem.MolFromSmiles("c1ccccc1")
    big = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    assert addition(small, big) is None
