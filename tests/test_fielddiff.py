"""Tests for the field difference between a candidate and the molecule it imitates."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cofields import Grid, difference_volume, field_volume, make_grid  # noqa: E402
from cmfqsar.fielddiff import (difference_features, mismatch,  # noqa: E402
                               overall_similarity, rank_against)
from cmfqsar.fields import FieldSet, MoleculeFields, build_fields  # noqa: E402

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


def _moved(mol: MoleculeFields, shift) -> MoleculeFields:
    return MoleculeFields(name=mol.name + "_moved", coords=mol.coords + np.asarray(shift, float),
                          weights=mol.weights, radii=mol.radii, symbols=mol.symbols,
                          atom_types=mol.atom_types, bonds=mol.bonds, charges=mol.charges)


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create(alpha=0.5)
    mf = [build_fields(_mol(s, seed=i + 1, name=f"m{i}"), fs, f"m{i}")
          for i, s in enumerate(SMILES)]
    return fs, mf


def test_a_molecule_matches_itself_exactly(world):
    fs, mf = world
    rows = mismatch(mf[0], mf[0], fs, spacing=0.6)
    assert len(rows) == len(fs.specs)
    for r in rows:
        assert r.cosine == pytest.approx(1.0, abs=1e-9)
        assert r.distance == pytest.approx(0.0, abs=1e-6)
        assert r.missing == pytest.approx(0.0, abs=1e-6)
        assert r.added == pytest.approx(0.0, abs=1e-6)
        assert r.norm_ref == pytest.approx(r.norm_cand, rel=1e-12)
    assert overall_similarity(mf[0], mf[0], fs) == pytest.approx(1.0, abs=1e-9)
    assert difference_features(mf[0], mf[0], fs, spacing=0.6) == []


def test_the_distance_is_the_integral_it_claims_to_be(world):
    """||Xa - Xb||^2 = K(a,a) + K(b,b) - 2K(a,b), checked against the map on a fine grid."""
    fs, mf = world
    a, b = mf[0], _moved(mf[0], [1.2, -0.4, 0.3])
    grid = make_grid(np.vstack([a.coords, b.coords]), margin=5.0, spacing=0.25)
    rows = mismatch(a, b, fs, grid=grid)
    for f, (spec, row) in enumerate(zip(fs.specs, rows)):
        d = difference_volume(a, b, spec, f, grid)
        numeric = float((d ** 2).sum() * grid.spacing ** 3)
        assert row.distance ** 2 == pytest.approx(numeric, rel=0.05, abs=1e-6)


def test_moving_a_molecule_makes_what_is_missing_and_what_is_added(world):
    fs, mf = world
    a = mf[0]
    b = _moved(a, [2.0, 0.0, 0.0])
    rows = mismatch(b, a, fs, spacing=0.4)          # candidate b, reference a
    assert all(r.cosine < 0.999 for r in rows)
    assert all(r.distance > 0 for r in rows)
    # the same field, only elsewhere: what is missing and what is added are about equal
    steric = next(r for r in rows if r.field == "steric")
    assert steric.missing > 0 and steric.added > 0
    assert steric.missing == pytest.approx(steric.added, rel=0.15)
    feats = difference_features(b, a, fs, spacing=0.4, max_features=8)
    assert feats
    kinds = {f.kind.split()[0] for f in feats}
    assert kinds <= {"missing", "extra"} and len(kinds) == 2
    for f in feats:
        assert (f.importance > 0) == f.kind.startswith("extra")
        assert f.radius > 0


def test_ranking_puts_the_reference_first_and_stays_in_range(world):
    fs, mf = world
    order = rank_against(mf[0], mf, fs)
    assert order[0][0] == 0 and order[0][1] == pytest.approx(1.0, abs=1e-9)
    assert [s for _, s in order] == sorted([s for _, s in order], reverse=True)
    assert all(-1.0 - 1e-9 <= s <= 1.0 + 1e-9 for _, s in order)
    # a molecule far away shares no field with it
    far = _moved(mf[0], [60.0, 0.0, 0.0])
    assert overall_similarity(far, mf[0], fs) == pytest.approx(0.0, abs=1e-6)


def test_the_closed_form_needs_no_grid(world):
    fs, mf = world
    rows = mismatch(mf[1], mf[0], fs, grid=False)
    assert all(np.isfinite(r.cosine) and np.isfinite(r.distance) for r in rows)
    assert all(not np.isfinite(r.missing) and not np.isfinite(r.added) for r in rows)
    on_grid = mismatch(mf[1], mf[0], fs, spacing=0.6)
    for a, b in zip(rows, on_grid):
        assert a.cosine == pytest.approx(b.cosine, abs=1e-12)
        assert a.distance == pytest.approx(b.distance, abs=1e-12)
