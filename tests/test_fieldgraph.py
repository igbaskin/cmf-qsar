"""Tests for field-point graphs and the alignment built on them."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cofields import field_volume, make_grid  # noqa: E402
from cmfqsar.fieldgraph import (DEFAULT_FIELDS, FieldPoint, align,  # noqa: E402
                                align_dataset, correspondence_graph, field_points,
                                maximal_cliques, summarise, superpose_pairs, _transformed)
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.pharmacophore import features_from_volume  # noqa: E402

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "NCCc1ccc(O)cc1"]


def _mol(smiles, seed=1, name="m"):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", name)
    return m


def _rigid(rng, scale=5.0):
    """A random proper rotation and translation."""
    R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1.0
    return R, rng.normal(scale=scale, size=3)


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hydrophobic", "hba", "hbd"],
                         alpha=0.5)
    mf = [build_fields(_mol(s, seed=i + 1, name=f"m{i}"), fs, f"m{i}")
          for i, s in enumerate(SMILES)]
    return fs, mf


def _point(field, sign, centre, weight=1.0, radius=1.0):
    return FieldPoint(field=field, sign=sign, centre=np.asarray(centre, float),
                      radius=radius, weight=weight)


# --------------------------------------------------------------------- points

def test_the_steric_field_is_one_blob_at_every_contour(world):
    """The measured reason the steric field is left out of the graph."""
    fs, mf = world
    names = [s.name for s in fs.specs]
    f = names.index("steric")
    grid = make_grid(mf[0].coords, margin=2.0, spacing=0.5)
    vol = field_volume(mf[0], fs.specs[f], f, grid)
    for level in (0.2, 0.4, 0.6, 0.8):
        regions = features_from_volume(vol, grid, "steric", +1, level=level, min_radius=0.5)
        assert len(regions) == 1, f"level {level} gave {len(regions)} regions"
    assert "steric" not in DEFAULT_FIELDS


def test_field_points_are_typed_and_ordered_by_weight(world):
    fs, mf = world
    pts = field_points(mf[0], fs)
    assert 4 <= len(pts) <= 16
    assert all(p.field in DEFAULT_FIELDS for p in pts)
    assert all(p.weight >= 0 for p in pts)
    assert [p.weight for p in pts] == sorted([p.weight for p in pts], reverse=True)
    assert {p.sign for p in pts} <= {+1, -1}
    assert any(p.label in ("positive", "negative") for p in pts)
    assert "at (" in pts[0].describe()
    # max_points caps the list, keeping the strongest
    few = field_points(mf[0], fs, max_points=3)
    assert len(few) == 3 and few[0].weight == pts[0].weight


def test_field_points_move_with_the_molecule_and_keep_their_distances(world):
    """A field point is a property of the molecule, so a rigid motion must only carry it along."""
    fs, mf = world
    rng = np.random.default_rng(3)
    R, t = _rigid(rng)
    a = field_points(mf[1], fs)
    b = field_points(_transformed(mf[1], R, t), fs)
    assert len(a) == len(b)
    da = np.linalg.norm(np.array([p.centre for p in a])[:, None]
                        - np.array([p.centre for p in a])[None], axis=-1)
    db = np.linalg.norm(np.array([p.centre for p in b])[:, None]
                        - np.array([p.centre for p in b])[None], axis=-1)
    # the grid is rebuilt around the moved molecule, so allow the discretisation its say
    assert float(np.abs(da - db).max()) < 0.5


# ---------------------------------------------------------------- the graph

def test_only_points_of_the_same_type_can_be_paired():
    a = [_point("electrostatic", +1, [0, 0, 0]), _point("hbd", +1, [3, 0, 0])]
    b = [_point("electrostatic", +1, [5, 0, 0]), _point("hba", +1, [8, 0, 0])]
    pairs, adj = correspondence_graph(a, b)
    assert pairs == [(0, 0)]                     # the donor has no acceptor to match
    assert adj == [set()]
    # opposite signs of one field are different types
    assert correspondence_graph([_point("electrostatic", +1, [0, 0, 0])],
                                [_point("electrostatic", -1, [0, 0, 0])])[0] == []


def test_compatibility_is_distance_agreement_within_the_tolerance():
    a = [_point("hbd", +1, [0, 0, 0]), _point("hba", +1, [4.0, 0, 0])]
    b = [_point("hbd", +1, [0, 0, 0]), _point("hba", +1, [4.4, 0, 0])]
    pairs, adj = correspondence_graph(a, b, tolerance=0.5)
    assert len(pairs) == 2 and adj[0] == {1}
    _, tight = correspondence_graph(a, b, tolerance=0.2)
    assert tight[0] == set()                     # 0.4 A of disagreement is now too much
    # a pairing never reuses a point, so two pairings of the same point are incompatible
    a2 = [_point("hbd", +1, [0, 0, 0])]
    b2 = [_point("hbd", +1, [0, 0, 0]), _point("hbd", +1, [1, 0, 0])]
    pairs2, adj2 = correspondence_graph(a2, b2, tolerance=10.0)
    assert len(pairs2) == 2 and adj2[0] == set() and adj2[1] == set()


def test_maximal_cliques_finds_them_and_respects_its_limits():
    # a triangle 0-1-2 plus a pendant 3 joined only to 0
    adj = [{1, 2, 3}, {0, 2}, {0, 1}, {0}]
    cliques = maximal_cliques(adj, min_size=2)
    as_sets = [set(c) for c in cliques]
    assert {0, 1, 2} in as_sets and {0, 3} in as_sets
    assert cliques[0] == sorted(cliques[0], key=lambda v: v) or len(cliques[0]) == 3
    assert len(cliques[0]) == 3                  # sorted largest first
    assert maximal_cliques(adj, min_size=4) == []
    assert maximal_cliques([], min_size=3) == []
    # the budget bounds the work and still returns something usable
    big = [set(range(12)) - {i} for i in range(12)]
    assert len(maximal_cliques(big, min_size=3, max_cliques=1)) == 1


# -------------------------------------------------------------- superposition

def test_superposition_recovers_a_known_transform_exactly():
    rng = np.random.default_rng(7)
    base = rng.normal(size=(5, 3)) * 3.0
    R, t = _rigid(rng)
    a = [_point("hbd", +1, c) for c in base]
    b = [_point("hbd", +1, c) for c in base @ R.T + t]
    pairs = [(i, i) for i in range(5)]
    R2, t2, rmsd = superpose_pairs(a, b, pairs)
    assert rmsd == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(base @ R2.T + t2, base @ R.T + t, atol=1e-9)
    with pytest.raises(ValueError):
        superpose_pairs(a, b, pairs[:2])         # two points do not fix a rigid body


def test_a_mirror_image_is_not_fitted_as_a_rotation():
    """Four points of one chirality cannot be superimposed on their reflection."""
    base = np.array([[0.0, 0, 0], [3.0, 0, 0], [0.0, 3.0, 0], [0.0, 0, 3.0]])
    mirrored = base * np.array([1.0, 1.0, -1.0])
    a = [_point("hbd", +1, c) for c in base]
    b = [_point("hbd", +1, c) for c in mirrored]
    _, _, rmsd = superpose_pairs(a, b, [(i, i) for i in range(4)])
    assert rmsd > 1.0                            # a proper rotation cannot do it, and says so


# -------------------------------------------------------------- the alignment

def test_alignment_recovers_a_molecule_onto_its_own_pose(world):
    """The one case with a known answer: a molecule moved at random, put back by its fields."""
    fs, mf = world
    rng = np.random.default_rng(0)
    for m in mf:
        R0, t0 = _rigid(rng)
        moved = _transformed(m, R0, t0)
        res = align(moved, m, fs)
        assert res is not None, m.name
        back = moved.coords @ res.R.T + res.t
        rmsd = float(np.sqrt(((back - m.coords) ** 2).sum(axis=1).mean()))
        assert rmsd < 0.35, f"{m.name}: {rmsd:.3f} A"
        assert res.similarity > 0.99
        assert res.size >= 3 and res.n_cliques >= 1
        assert "points matched" in res.describe()


def test_two_molecules_with_nothing_in_common_are_left_alone(world):
    """No frame is invented: a molecule presenting nothing comparable returns None."""
    fs, mf = world
    import copy
    bare = copy.copy(mf[0])
    bare.weights = np.zeros_like(mf[0].weights)      # a molecule with no fields at all
    assert align(bare, mf[1], fs) is None
    assert field_points(bare, fs) == []


def test_align_dataset_shares_the_reference_points_and_summarises(world):
    fs, mf = world
    rng = np.random.default_rng(5)
    moved = [_transformed(m, *_rigid(rng)) for m in mf]
    matches = align_dataset(moved, mf[0], fs)
    assert len(matches) == len(moved)
    stats = summarise(matches)
    assert stats["n"] == 3 and stats["placed"] >= 2
    assert np.isfinite(stats["mean_similarity"])
    # the reference itself is skipped rather than aligned to itself
    with_ref = align_dataset(list(mf) + [mf[0]], mf[0], fs)
    assert with_ref[0] is None and with_ref[-1] is None
    assert summarise([None, None])["placed"] == 0.0
