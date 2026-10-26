"""Increments fitted to the data instead of taken from the table (plan item D5).

The module rests on one identity: an increment gives every atom of a type the same weight, so the
field kernel is a quadratic form in the increments.  The first test checks that identity against
the kernel the program builds normally -- if it did not hold, everything fitted here would be
fitted to the wrong kernel.  The rest check the derivatives, because a gradient that is almost
right is the kind of error that still converges, to the wrong place: every link of the chain
(the cross-validated loss in the kernel, the cosine normalisation, the increments themselves) is
compared with finite differences.
"""
import json
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.learnincrements import (INCREMENT_FIELDS, OTHER, _normalised_gradient,  # noqa: E402
                                     basis_from, coverage, cv_loss_and_grad, fit_increments,
                                     folds_of, increment_key, increment_types, normalise,
                                     other_kernel, published_increments, read_increments, spread,
                                     type_index, types_in)
from cmfqsar.model import cv_predict  # noqa: E402

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "CCOc1ccc(C)cc1CC(=O)N",
          "CC(C)COc1ccccc1C(=O)NC", "CCCCOc1ccccc1C(=O)N", "CCCOc1ccccc1C(=O)NCCC",
          "CCOc1ccccc1C(=O)NCC", "COc1ccccc1C(=O)NCC", "CCCOc1ccc(Cl)cc1C(=O)NCC",
          "CCCOc1ccc(C)cc1C(=O)NCC"]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


@pytest.fixture(scope="module")
def series():
    mols = [_mol(s, i + 1) for i, s in enumerate(SMILES)]
    names = [f"m{i}" for i in range(len(mols))]
    rng = np.random.default_rng(0)
    y = np.array([len(s) for s in SMILES], float) + rng.normal(0, 0.3, len(SMILES))
    return mols, names, y


@pytest.fixture(scope="module")
def prepared(series):
    mols, names, y = series
    keys, counts = types_in(mols)
    fs = FieldSet.create(alpha=0.5, param_scheme="tripos")
    built = [build_fields(m, fs, n, _single=True) for m, n in zip(mols, names)]
    basis = basis_from(built, type_index(mols, keys), 0.5, keys, counts, dtype=np.float64)
    return {"mols": mols, "names": names, "y": y, "keys": keys, "fs": fs, "built": built,
            "basis": basis, "published": published_increments(keys)}


# --------------------------------------------------------------------------- the types


def test_a_type_is_the_sybyl_type_and_the_hydrogen_count(series):
    mols, _, _ = series
    keys = increment_types(mols[0])
    assert len(keys) == mols[0].GetNumAtoms()
    assert all(k == "" for k, a in zip(keys, mols[0].GetAtoms()) if a.GetAtomicNum() == 1)
    heavy = [k for k in keys if k]
    assert all("/" in k for k in heavy)
    assert increment_key("C.3", 2) == "C.3/2"
    assert "C.ar/1" in heavy


def test_the_published_values_are_the_table(prepared):
    keys = prepared["keys"]
    pub = published_increments(keys)
    assert pub.shape == (len(keys), 3)
    from cmfqsar.increments import tripos_increments
    i = keys.index("C.ar/1")
    want = tripos_increments("C.ar", 1)
    assert pub[i, 0] == pytest.approx(want["hydrophobic"])
    assert pub[i, 1] == pytest.approx(want["hba"])
    assert pub[i, 2] == pytest.approx(want["hbd"])


def test_rare_types_can_be_merged_and_are_not_by_default(series):
    mols, _, _ = series
    everything, counts = types_in(mols)
    assert OTHER not in everything
    few, _ = types_in(mols, min_atoms=10_000, max_types=2)
    assert OTHER in few and len(few) <= 3


def test_every_heavy_atom_gets_a_type_index(prepared):
    idx = type_index(prepared["mols"], prepared["keys"])
    for m, ix in zip(prepared["mols"], idx):
        heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
        assert len(ix) == m.GetNumAtoms()
        assert (ix[heavy] >= 0).all()


# ------------------------------------------------- the identity the whole module rests on


def test_the_quadratic_form_is_the_kernel_the_program_builds(prepared):
    """Every increment field kernel has to come out of the tensor exactly, or nothing holds."""
    fs, built, basis = prepared["fs"], prepared["built"], prepared["basis"]
    table = AtomTable.from_molecules(built)
    K = kernel_matrices(table, table, fs.specs)
    names = [sp.name for sp in fs.specs]
    for f, name in enumerate(INCREMENT_FIELDS):
        mine = basis.kernel(prepared["published"][:, f])
        assert mine == pytest.approx(K[names.index(name)], rel=1e-9, abs=1e-10)


def test_the_tensor_is_symmetric_under_swapping_the_two_molecules(prepared):
    M = prepared["basis"].M
    assert M == pytest.approx(np.transpose(M, (2, 3, 0, 1)), rel=1e-6, abs=1e-9)


def test_doubling_every_increment_multiplies_the_kernel_by_four(prepared):
    c = prepared["published"][:, 0]
    a = prepared["basis"].kernel(c)
    b = prepared["basis"].kernel(2 * c)
    assert b == pytest.approx(4 * a, rel=1e-9)
    # ... and leaves the cosine-normalised kernel exactly where it was
    assert normalise(b)[0] == pytest.approx(normalise(a)[0], rel=1e-9, abs=1e-12)


# ----------------------------------------------------------------------- the derivatives


def test_the_loss_is_the_one_the_model_reports(prepared):
    y = prepared["y"]
    K = normalise(prepared["basis"].kernel(prepared["published"][:, 0]))[0]
    pred = cv_predict(K, y, "krr", {"lam": 0.2}, folds=5, seed=0)
    mine, _ = cv_loss_and_grad(K, y, 0.2, folds_of(len(y), 5, 0))
    assert mine == pytest.approx(float(np.sum((pred - y) ** 2)), rel=1e-10)


def test_the_gradient_in_the_kernel_matches_finite_differences(prepared):
    y = prepared["y"]
    n = len(y)
    K = normalise(prepared["basis"].kernel(prepared["published"][:, 0]))[0] + 0.1 * np.eye(n)
    parts = folds_of(n, 5, 0)
    for bias in (True, False):
        _, G = cv_loss_and_grad(K, y, 0.2, parts, fit_bias=bias)
        h = 1e-6
        for i, j in ((0, 0), (0, 3), (2, 5), (4, 4)):
            E = np.zeros((n, n))
            E[i, j] = h
            E[j, i] = h
            num = (cv_loss_and_grad(K + E, y, 0.2, parts, fit_bias=bias)[0]
                   - cv_loss_and_grad(K - E, y, 0.2, parts, fit_bias=bias)[0]) / (2 * h)
            want = G[i, i] if i == j else 2 * G[i, j]
            assert num == pytest.approx(want, rel=1e-4, abs=1e-6)


def test_the_gradient_through_the_normalisation_matches_finite_differences(prepared):
    y = prepared["y"]
    n = len(y)
    Kraw = prepared["basis"].kernel(prepared["published"][:, 0])
    parts = folds_of(n, 5, 0)

    def loss(K):
        return cv_loss_and_grad(normalise(K)[0], y, 0.2, parts)

    _, G = loss(Kraw)
    Gr = _normalised_gradient(G, Kraw, normalise(Kraw)[1])
    h = 1e-4 * np.abs(Kraw).max()
    for i, j in ((0, 0), (1, 4), (3, 3)):
        E = np.zeros((n, n))
        E[i, j] = h
        E[j, i] = h
        num = (loss(Kraw + E)[0] - loss(Kraw - E)[0]) / (2 * h)
        want = Gr[i, i] if i == j else 2 * Gr[i, j]
        assert num == pytest.approx(want, rel=1e-4, abs=1e-6)


def test_the_gradient_in_the_increments_matches_finite_differences(prepared):
    """The whole chain at once: increments -> kernel -> normalisation -> cross-validated loss."""
    y = prepared["y"]
    basis = prepared["basis"]
    parts = folds_of(len(y), 5, 0)
    c = prepared["published"][:, 0] + 0.01

    def pieces(cc):
        Kraw = basis.kernel(cc)
        Kn, d = normalise(Kraw)
        L, G = cv_loss_and_grad(Kn, y, 0.2, parts)
        return L, G, Kraw, d

    L, G, Kraw, d = pieces(c)
    grad = 2.0 * np.einsum("mn,tmn->t", _normalised_gradient(G, Kraw, d), basis.half(c),
                           optimize=True)
    h = 1e-6
    for t in range(len(c)):
        up, down = c.copy(), c.copy()
        up[t] += h
        down[t] -= h
        num = (pieces(up)[0] - pieces(down)[0]) / (2 * h)
        assert num == pytest.approx(grad[t], rel=1e-4, abs=1e-5)


# ---------------------------------------------------------------------------- the fitting


def test_fitting_lowers_the_loss_it_is_fitted_on(series):
    mols, names, y = series
    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1.0, max_iter=60)
    assert fit.loss <= fit.loss_published
    assert fit.values.shape == fit.published.shape
    assert fit.iterations > 0
    assert set(fit.increments) == set(fit.keys)
    assert set(fit.increments[fit.keys[0]]) == set(INCREMENT_FIELDS)


def test_a_large_penalty_keeps_the_published_increments(series):
    mols, names, y = series
    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1e8, max_iter=60)
    assert fit.values == pytest.approx(fit.published, abs=1e-3)
    assert all(v > 0.99 for v in fit.cosine().values())


def test_the_fitted_values_are_reported_at_the_published_length(series):
    """The normalised kernel cannot see the length, so the report fixes it to the table's."""
    mols, names, y = series
    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=0.01, max_iter=60)
    for f in range(len(INCREMENT_FIELDS)):
        assert np.linalg.norm(fit.values[:, f]) == pytest.approx(
            np.linalg.norm(fit.published[:, f]), rel=1e-6)


def test_the_cosine_says_how_far_the_fit_moved(series):
    mols, names, y = series
    tight = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1e8, max_iter=40)
    loose = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1e-3, max_iter=120)
    assert min(tight.cosine().values()) > max(loose.cosine().values()) - 1e-9


def test_the_fit_can_be_written_and_read_back(series, tmp_path):
    mols, names, y = series
    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1.0, max_iter=40)
    path = str(tmp_path / "inc.json")
    fit.to_json(path)
    got = read_increments(path)
    assert set(got) == set(fit.keys)
    assert got[fit.keys[0]]["hydrophobic"] == pytest.approx(fit.values[0, 0])
    with open(path) as fh:
        assert "cosine" in json.load(fh)


def test_a_field_set_builds_the_fields_with_a_fitted_table(series):
    mols, names, y = series
    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1e-3, max_iter=80)
    fs = FieldSet.create(alpha=0.5, param_scheme="tripos")
    before = build_fields(mols[0], fs, "m", _single=True).weights.copy()
    fs.increments = fit.increments
    after = build_fields(mols[0], fs, "m", _single=True).weights
    names_f = [sp.name for sp in fs.specs]
    for name in INCREMENT_FIELDS:
        k = names_f.index(name)
        assert not np.allclose(before[:, k], after[:, k])
    for name in ("steric", "electrostatic"):
        k = names_f.index(name)
        assert np.allclose(before[:, k], after[:, k])


def test_a_type_the_table_does_not_know_falls_back_and_is_reported(series):
    mols, names, _ = series
    table = {"C.ar/1": {"hydrophobic": 1.0, "hba": 0.0, "hbd": 0.0}}
    cov = coverage(table, mols)
    assert 0.0 < cov["exact"] < 1.0
    assert cov["other"] == 0.0 and cov["missing"] == pytest.approx(1.0 - cov["exact"])
    fs = FieldSet.create(alpha=0.5, param_scheme="tripos")
    fs.increments = table
    w = build_fields(mols[0], fs, "m", _single=True).weights
    k = [sp.name for sp in fs.specs].index("hydrophobic")
    assert set(np.unique(w[:, k])) <= {0.0, 1.0}      # only the one type the table carries
    table["other"] = {"hydrophobic": 0.5, "hba": 0.0, "hbd": 0.0}
    assert coverage(table, mols)["missing"] == 0.0


def test_the_fitted_table_survives_being_saved_with_a_model(series, tmp_path):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel
    mols, names, y = series
    fit = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1.0, max_iter=40)
    fs = FieldSet.create(alpha=0.5, param_scheme="tripos")
    fs.increments = fit.increments
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(table, y, cv_folds=0)
    path = str(tmp_path / "m.cmf")
    save_model(path, model)
    back, _ = load_model(path)
    assert back.fieldset.increments is not None
    assert set(back.fieldset.increments) == set(fit.keys)


def test_the_field_set_copy_carries_the_table(series):
    fs = FieldSet.create(alpha=0.5, param_scheme="tripos")
    fs.increments = {"C.3/2": {"hydrophobic": 1.0, "hba": 0.0, "hbd": 0.0}}
    assert fs.copy().increments == fs.increments


def test_the_other_fields_enter_as_a_constant(prepared):
    K = other_kernel(prepared["built"], prepared["fs"])
    assert K.shape == (len(prepared["built"]),) * 2
    assert np.allclose(np.diag(K), 2.0, atol=1e-8)       # two normalised fields, each 1 on itself
    assert np.linalg.eigvalsh(0.5 * (K + K.T)).min() > -1e-8


def test_the_agreement_between_two_fits_is_reported(series):
    mols, names, y = series
    a = fit_increments(mols, names, y, alpha=0.5, lam=0.1, folds=5, rho=1.0, max_iter=40)
    b = fit_increments(mols, names, y[::-1], alpha=0.5, lam=0.1, folds=5, rho=1.0, max_iter=40)
    got = spread([a, b])
    assert set(got) == set(INCREMENT_FIELDS)
    assert all(-1.0 <= v <= 1.0 for v in got.values())
    assert np.isnan(spread([a])["hydrophobic"])     # one fit on its own has no pair to compare
