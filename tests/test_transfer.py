"""Comparing two models by their co-fields (plan item F3).

The claim is that an integral over all space is a matrix product, so the test that matters is the
first one: the closed form against the same overlap taken on a grid.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cofields import cofield_volume, make_grid  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.transfer import (cofield_overlap, describe_matrix, incompatible,  # noqa: E402
                              overlap_matrix)

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(Br)cc1O", "c1ccc(N)cc1C"]
Y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1, 1.7, 1.4])


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _fields(alpha=1.0):
    fs = FieldSet.create(alpha=alpha)
    return fs, [build_fields(_mol(s), fs, f"m{i}") for i, s in enumerate(SMILES)]


def _model(fs, mf, idx, y, **kw):
    model = CMFModel(fieldset=fs.copy(), method="krr", params={"lam": 0.1}, normalize=True, **kw)
    model.fit(AtomTable.from_molecules([mf[i] for i in idx]), np.asarray(y, float)[idx],
              cv_folds=3, cv_seed=0)
    return model


@pytest.fixture(scope="module")
def two():
    fs, mf = _fields()
    a = _model(fs, mf, list(range(6)), Y)
    b = _model(fs, mf, list(range(6, 12)), Y)
    return a, b, fs, mf


def test_the_closed_form_is_the_integral_it_claims_to_be(two):
    a, b, _fs, _mf = two
    res = cofield_overlap(a, b)
    grid = make_grid(np.vstack([a.train.coords, b.train.coords]), margin=4.0, spacing=0.4)
    cell = grid.spacing ** 3
    for f, row in enumerate(res.fields):
        CA = cofield_volume(a, f, grid).ravel()
        CB = cofield_volume(b, f, grid).ravel()
        num = float(CA @ CB) * cell
        cos = num / float(np.sqrt((CA @ CA) * cell) * np.sqrt((CB @ CB) * cell))
        assert row.overlap == pytest.approx(num, rel=0.02)
        assert row.cosine == pytest.approx(cos, abs=0.01)


def test_a_model_against_itself_is_one(two):
    a, _b, _fs, _mf = two
    res = cofield_overlap(a, a, "a", "again")
    assert res.cosine == pytest.approx(1.0)
    assert all(r.cosine == pytest.approx(1.0) for r in res.fields)


def test_a_model_against_its_opposite_is_minus_one():
    fs, mf = _fields()
    a = _model(fs, mf, list(range(6)), Y)
    b = _model(fs, mf, list(range(6)), -Y)
    res = cofield_overlap(a, b)
    assert res.cosine == pytest.approx(-1.0)
    assert all(r.cosine == pytest.approx(-1.0) for r in res.fields)


def test_the_cosine_does_not_move_when_an_activity_is_rescaled(two):
    a, b, fs, mf = two
    plain = cofield_overlap(a, b).cosine
    scaled = cofield_overlap(a, _model(fs, mf, list(range(6, 12)), 10.0 * Y)).cosine
    assert scaled == pytest.approx(plain, abs=1e-6)
    shifted = cofield_overlap(a, _model(fs, mf, list(range(6, 12)), Y + 5.0)).cosine
    assert np.isfinite(shifted)       # a shift moves the bias, not the shape, but not exactly


def test_it_is_symmetric(two):
    a, b, _fs, _mf = two
    assert cofield_overlap(a, b).cosine == pytest.approx(cofield_overlap(b, a).cosine)


def test_two_models_in_different_places_are_refused():
    fs, mf = _fields()
    a = _model(fs, mf, list(range(6)), Y)
    b = _model(fs, mf, list(range(6, 12)), Y)
    b.train.coords = b.train.coords + 500.0
    assert "same region of space" in (incompatible(a, b) or "")
    with pytest.raises(ValueError, match="same region of space"):
        cofield_overlap(a, b)
    res = cofield_overlap(a, b, strict=False)
    assert np.isnan(res.cosine) and res.notes


def test_different_widths_are_refused():
    fs1, mf1 = _fields(alpha=1.0)
    fs2, mf2 = _fields(alpha=0.5)
    a = _model(fs1, mf1, list(range(6)), Y)
    b = _model(fs2, mf2, list(range(6, 12)), Y)
    why = incompatible(a, b)
    assert why and "different widths" in why


def test_a_nonlinear_model_has_no_co_field_to_compare(two):
    a, b, _fs, _mf = two
    b.kernel_layer = "gauss"
    try:
        assert "no co-field" in (incompatible(a, b) or "")
    finally:
        b.kernel_layer = None


def test_the_matrix_is_square_symmetric_and_one_on_the_diagonal(two):
    a, b, fs, mf = two
    c = _model(fs, mf, [0, 2, 4, 6, 8, 10], Y)
    M, pairs = overlap_matrix([a, b, c], ["a", "b", "c"])
    assert M.shape == (3, 3)
    assert np.allclose(np.diag(M), 1.0)
    assert np.allclose(M, M.T, equal_nan=True)
    assert len(pairs) == 3
    text = describe_matrix(M, ["a", "b", "c"])
    assert "a" in text and "+1.000" in text


def test_the_per_field_numbers_are_reported_with_their_norms(two):
    a, b, _fs, _mf = two
    res = cofield_overlap(a, b)
    assert len(res.fields) == len(a.fieldset.specs)
    for r in res.fields:
        assert r.norm_a > 0 and r.norm_b > 0
        assert abs(r.cosine) <= 1.0 + 1e-9
        assert r.cosine == pytest.approx(r.overlap / (r.norm_a * r.norm_b))
        assert r.label in r.describe()
    assert "cosine" in res.describe()
