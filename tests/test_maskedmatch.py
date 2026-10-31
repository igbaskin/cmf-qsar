"""Scoring a candidate only where the co-field has weight (plan item MC5).

The arithmetic is the whole claim here: the mask must multiply the co-field exactly, the masked
overlap must be the same integral as the one in the prediction, and the cosine must not grow with
the size of the molecule.  Each of those is checked against something independent -- a grid
integral, the model's own `predict`, or an invariance -- rather than against a stored number.
"""
import copy
import math
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import AtomTable, CMFModel, FieldSet, build_fields  # noqa: E402
from cmfqsar.cofields import (CofieldMask, make_grid, mask_width,  # noqa: E402
                              masked_cofield, top_atom_mask)
from cmfqsar.kernels import expansion_overlap, gaussian_product, numerical_overlap  # noqa: E402
from cmfqsar.maskedmatch import MaskedScorer, score_candidates, screen_conformers  # noqa: E402

SMILES = ["c1ccccc1C(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
          "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O", "c1ccc(Br)cc1C(=O)O"]
Y = np.array([1.0, 2.0, 1.5, 2.5, 3.0, 2.0, 2.2])


def _dense(points, centres, weights, widths):
    """The Gaussian sum written out, with nothing dropped.

    `cofields.evaluate_gaussians` truncates every Gaussian at the radius where it falls below
    1e-5 of its peak, which is right for drawing and wrong for checking algebra: a product
    expansion has wider Gaussians than its factors, so the two sides would be truncated
    differently and the comparison would measure the truncation.
    """
    points = np.atleast_2d(np.asarray(points, float))
    centres = np.atleast_2d(np.asarray(centres, float))
    weights = np.asarray(weights, float).reshape(-1)
    widths = np.asarray(widths, float).reshape(-1)
    d2 = ((points[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    return (weights * np.exp(-0.5 * widths * d2)).sum(axis=1)


def _mols(n=len(SMILES), conformers=1):
    out = []
    for k, smi in enumerate(SMILES[:n]):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        if conformers > 1:
            AllChem.EmbedMultipleConfs(m, numConfs=conformers, params=p)
        else:
            AllChem.EmbedMolecule(m, p)
        out.append(m)
    return out


@pytest.fixture(scope="module")
def fitted():
    mols = _mols()
    fs = FieldSet.create(physchem=("steric", "electrostatic"), alpha=0.5)
    mf = [build_fields(m, fs, f"m{k}") for k, m in enumerate(mols)]
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), Y, cv_folds=3)
    return model, mf, mols


# --------------------------------------------------------------------- the two primitives

def test_the_pair_overlap_is_the_formula_of_the_module():
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=3), rng.normal(size=3)
    w1, w2, b1, b2 = 1.3, -0.7, 0.6, 0.9
    closed = expansion_overlap(x1[None], [w1], [b1], x2[None], [w2], [b2])
    assert closed == pytest.approx(numerical_overlap(x1, w1, b1, x2, w2, b2, spacing=0.05),
                                   rel=1e-4)
    # the same number written out by hand
    s = b1 + b2
    want = w1 * w2 * (2 * math.pi / s) ** 1.5 * math.exp(-(b1 * b2) / (2 * s)
                                                         * float(((x1 - x2) ** 2).sum()))
    assert closed == pytest.approx(want, rel=1e-12)


def test_the_product_of_two_expansions_is_the_pointwise_product():
    rng = np.random.default_rng(2)
    x1, w1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4), rng.uniform(0.3, 1.5, 4)
    x2, w2, b2 = rng.normal(size=(3, 3)), rng.normal(size=3), rng.uniform(0.3, 1.5, 3)
    c, w, b = gaussian_product(x1, w1, b1, x2, w2, b2)
    assert len(c) == 12
    points = rng.normal(size=(20, 3)) * 1.5
    lhs = _dense(points, x1, w1, b1) * _dense(points, x2, w2, b2)
    rhs = _dense(points, c, w, b)
    assert np.allclose(lhs, rhs, atol=1e-12)


def test_the_integral_of_the_product_is_the_overlap():
    rng = np.random.default_rng(3)
    x1, w1, b1 = rng.normal(size=(4, 3)), rng.normal(size=4), rng.uniform(0.3, 1.5, 4)
    x2, w2, b2 = rng.normal(size=(3, 3)), rng.normal(size=3), rng.uniform(0.3, 1.5, 3)
    c, w, b = gaussian_product(x1, w1, b1, x2, w2, b2)
    integral = float(np.sum(w * (2.0 * np.pi / b) ** 1.5))
    assert integral == pytest.approx(expansion_overlap(x1, w1, b1, x2, w2, b2), rel=1e-12)


def test_an_empty_expansion_overlaps_with_nothing():
    assert expansion_overlap(np.zeros((0, 3)), [], [], np.zeros((1, 3)), [1.0], [1.0]) == 0.0
    c, w, b = gaussian_product(np.zeros((0, 3)), [], [], np.zeros((1, 3)), [1.0], [1.0])
    assert len(c) == 0 and len(w) == 0 and len(b) == 0


def test_the_tolerance_drops_only_negligible_terms():
    x1 = np.array([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0]])      # one atom far away
    x2 = np.array([[0.1, 0.0, 0.0]])
    full = gaussian_product(x1, [1.0, 1.0], [0.5, 0.5], x2, [1.0], [0.5])
    cut = gaussian_product(x1, [1.0, 1.0], [0.5, 0.5], x2, [1.0], [0.5], tol=1e-9)
    assert len(full[0]) == 2 and len(cut[0]) == 1
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.2, 0.0]])
    assert np.allclose(_dense(points, *full), _dense(points, *cut))


# --------------------------------------------------------------------- the mask

def test_the_mask_sits_on_the_atoms_that_contribute_most(fitted):
    model, mf, _ = fitted
    contrib = np.concatenate([model.atom_contributions(m) for m in mf])
    coords = np.vstack([m.coords for m in mf])
    mask = top_atom_mask(model, mf, k=4)
    assert len(mask.centres) == 4
    want = coords[np.argsort(-np.abs(contrib))[:4]]
    assert np.allclose(np.sort(mask.centres, axis=0), np.sort(want, axis=0))
    assert np.allclose(np.sort(np.abs(mask.contributions))[::-1],
                       np.sort(np.abs(contrib))[::-1][:4])


def test_the_normalised_mask_peaks_at_one(fitted):
    model, mf, _ = fitted
    assert top_atom_mask(model, mf, k=6).peak() == pytest.approx(1.0, rel=1e-9)
    raw = top_atom_mask(model, mf, k=6, normalise=False)
    assert raw.peak() >= 1.0                       # overlapping Gaussians add up


def test_the_mask_width_is_the_attenuation_the_kernel_uses(fitted):
    model, mf, _ = fitted
    assert mask_width(model) == pytest.approx(0.5)
    assert mask_width(model, 1.25) == pytest.approx(1.25)
    assert np.allclose(top_atom_mask(model, mf, k=3, width=1.25).widths, 1.25)


def test_the_masked_cofield_is_the_cofield_times_the_mask(fitted):
    model, mf, _ = fitted
    mask = top_atom_mask(model, mf, k=5)
    grid = make_grid(mf[0].coords, margin=2.0, spacing=1.0)
    points = grid.points()[::7]
    for f in range(len(model.fieldset.specs)):
        c, w, b = model.cofield_coefficients(f)
        want = _dense(points, c, w, b) * _dense(points, *mask.expansion())
        got = _dense(points, *masked_cofield(model, f, mask, tol=0.0))
        assert np.allclose(got, want, atol=1e-10)


def test_a_mask_of_one_atom_everywhere_leaves_the_cofield_alone(fitted):
    """A mask of weight 1 and width 0 is no mask; the limit is checked with a very wide one."""
    model, mf, _ = fitted
    wide = CofieldMask(centres=np.zeros((1, 3)), weights=np.ones(1), widths=np.full(1, 1e-6))
    points = make_grid(mf[0].coords, margin=1.0, spacing=1.5).points()[::5]
    c, w, b = model.cofield_coefficients(0)
    plain = _dense(points, c, w, b)
    masked = _dense(points, *masked_cofield(model, 0, wide, tol=0.0))
    assert np.allclose(masked, plain, rtol=1e-3, atol=1e-8)


# --------------------------------------------------------------------- the score

def test_inside_and_outside_add_up_to_the_prediction(fitted):
    """Rule 3: the unmasked overlap summed over the fields is the prediction itself."""
    model, mf, _ = fitted
    scorer = MaskedScorer(model, top_atom_mask(model, mf, k=6))
    for m in mf:
        s = scorer.score(m)
        own = float(model.predict(AtomTable.from_molecules([m]))[0])
        assert s.prediction == pytest.approx(own, rel=1e-9, abs=1e-9)
        assert s.inside + s.outside + model.bias == pytest.approx(own, rel=1e-9, abs=1e-9)


def test_the_cosine_does_not_grow_with_the_size_of_the_field(fitted):
    """Twice the field is twice the overlap and the same cosine, which is why it is reported."""
    model, mf, _ = fitted
    scorer = MaskedScorer(model, top_atom_mask(model, mf, k=6))
    one = scorer.score(mf[0])
    twice = copy.deepcopy(mf[0])
    twice.weights = twice.weights * 2.0
    two = scorer.score(twice)
    assert two.inside == pytest.approx(2.0 * one.inside, rel=1e-9)
    assert two.cosine == pytest.approx(one.cosine, rel=1e-9)


def test_a_bigger_mask_covers_everything_a_smaller_one_does(fitted):
    """The top-k atoms are nested, so without the peak normalisation the mask only grows.

    The normalised mask is *not* monotone -- more atoms overlap, so the peak it is divided by is
    larger -- and neither is the masked overlap, which is an integral of a signed product. What
    is monotone is the region itself, and that is what the mask is.
    """
    model, mf, _ = fitted
    points = np.vstack([m.coords for m in mf])[::3]
    small = top_atom_mask(model, mf, k=2, normalise=False)
    large = top_atom_mask(model, mf, k=8, normalise=False)
    assert np.all(_dense(points, *large.expansion())
                  >= _dense(points, *small.expansion()) - 1e-12)


def test_the_candidates_come_back_ranked(fitted):
    model, mf, _ = fitted
    out = score_candidates(model, mf, top_atom_mask(model, mf, k=6))
    assert len(out) == len(mf)
    assert [s.cosine for s in out] == sorted((s.cosine for s in out), reverse=True)
    assert "cosine" in str(out[0])


def test_a_conformer_library_keeps_the_best_conformation(fitted):
    model, mf, _ = fitted
    mols = _mols(n=3, conformers=3)
    for i, m in enumerate(mols):
        m.SetProp("_Name", f"cand{i}")
    out = screen_conformers(model, mols, top_atom_mask(model, mf, k=6))
    assert len(out) == 3                                  # one row per molecule, not per conformer
    assert {s.name for s in out} == {"cand0", "cand1", "cand2"}
    assert all(0 <= s.conformer < 3 for s in out)
    assert [s.cosine for s in out] == sorted((s.cosine for s in out), reverse=True)


def test_a_nonlinear_model_is_refused(fitted):
    """A masked co-field needs a co-field, and a nonlinear kernel layer has none."""
    model, mf, _ = fitted
    layered = copy.deepcopy(model)
    layered.kernel_layer = "gaussian"
    with pytest.raises(Exception):
        MaskedScorer(layered, top_atom_mask(model, mf, k=4))
