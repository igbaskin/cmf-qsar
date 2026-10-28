"""Kernel canonical correlation analysis (plan item F6).

The method's classical failure is that without a ridge every correlation is 1, so the tests that
matter are the ones about what is *not* found: a shuffled pairing holds out at nothing, and the
activity as a second view gives back kernel ridge regression and nothing new.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import kcca as KC  # noqa: E402
from cmfqsar.descriptors import fingerprint_kernel  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.transfer import cofield_overlap  # noqa: E402


def _two_views(n=60, d=6, shared=2, noise=0.3, seed=0):
    """Two linear kernels over the same objects that share `shared` latent directions."""
    rng = np.random.RandomState(seed)
    Z = rng.normal(size=(n, shared))                # what the two views have in common
    A = np.hstack([Z, rng.normal(size=(n, d - shared))]) @ rng.normal(size=(d, d))
    B = np.hstack([Z, rng.normal(size=(n, d - shared))]) @ rng.normal(size=(d, d))
    A += noise * rng.normal(size=A.shape)
    B += noise * rng.normal(size=B.shape)
    return A @ A.T, B @ B.T, Z


# ------------------------------------------------------------------ the solver


def test_the_generalised_eigenproblem_is_solved():
    """The directions satisfy K L b = rho (K^2 + c K) a and its mirror image."""
    K, L, _ = _two_views()
    fit = KC.kcca(K, L, ridge=0.1)
    Kc, Lc = KC.centre(K), KC.centre(L)
    for i in range(3):
        a, b, rho = fit.a[:, i], fit.b[:, i], fit.rho[i]
        lhs = Kc @ (Lc @ b)
        rhs = rho * (Kc @ (Kc @ a) + fit.ridge_a * (Kc @ a))
        assert np.allclose(lhs, rhs, atol=1e-8 * np.abs(rhs).max())
        lhs = Lc @ (Kc @ a)
        rhs = rho * (Lc @ (Lc @ b) + fit.ridge_b * (Lc @ b))
        assert np.allclose(lhs, rhs, atol=1e-8 * np.abs(rhs).max())
    # the normalisation a^T (K^2 + cK) a = 1, and the variates are uncorrelated across pairs
    G = fit.a.T @ (Kc @ Kc + fit.ridge_a * Kc) @ fit.a
    assert np.allclose(G, np.eye(G.shape[0]), atol=1e-8)
    assert np.all(np.diff(fit.rho) <= 1e-12)


def test_without_a_ridge_every_correlation_is_one():
    """The degeneracy the item warns about, reproduced: full-rank views, c = 0, rho = 1."""
    K, L, _ = _two_views(n=30, d=40)       # more latent dimensions than molecules: full rank
    fit = KC.kcca(K, L, ridge=0.0)
    assert np.allclose(fit.rho[:10], 1.0, atol=1e-6)
    assert np.allclose(fit.variance_a, 1.0, atol=1e-8)
    # and the ridge is what makes the number mean something
    fit = KC.kcca(K, L, ridge=1.0)
    assert fit.rho[0] < 0.999
    assert np.all(fit.variance_a < 1.0)


def test_shared_directions_are_found_and_the_rest_are_not():
    K, L, Z = _two_views(n=80, d=6, shared=2, noise=0.1)
    fit = KC.kcca(K, L, ridge=0.01)
    assert fit.rho[0] > 0.9 and fit.rho[1] > 0.7
    assert fit.rho[2] < 0.5
    # the two leading variates span what the views share
    U = fit.u[:, :2]
    coef, *_ = np.linalg.lstsq(U, Z, rcond=None)
    r2 = 1 - ((Z - U @ coef) ** 2).sum() / ((Z - Z.mean(0)) ** 2).sum()
    assert r2 > 0.75


def test_projection_of_new_molecules_matches_the_training_variates():
    K, L, _ = _two_views()
    fit = KC.kcca(K, L, ridge=0.1)
    assert np.allclose(fit.project_a(K), fit.u)
    assert np.allclose(fit.project_b(L), fit.v)


def test_the_activity_as_a_second_view_is_kernel_ridge_regression():
    """CCA against a one-dimensional view returns (K + cI)^-1 y, up to a scale factor.

    Exactly so when the kernel has full rank.  When it has not, the two dual vectors differ by
    a part in the null space of K, which contributes nothing to any prediction, and it is the
    fitted values K a that agree.
    """
    for d, full_rank in ((80, True), (6, False)):
        K, _, Z = _two_views(n=60, d=d)
        y = Z[:, 0] + 0.2 * Z[:, 1]
        yc = y - y.mean()
        L = np.outer(yc, yc)
        fit = KC.kcca(K, L, ridge=0.3)
        assert fit.n_components == 1
        Kc = KC.centre(K)
        krr = np.linalg.solve(Kc + fit.ridge_a * np.eye(len(y)), yc)
        a = fit.a[:, 0]
        pa, pk = Kc @ a, Kc @ krr
        assert abs(pa @ pk) / np.linalg.norm(pa) / np.linalg.norm(pk) > 1 - 1e-8
        if full_rank:
            assert abs(a @ krr) / np.linalg.norm(a) / np.linalg.norm(krr) > 1 - 1e-8


# ------------------------------------------------------ held out, ridge, shuffled


def test_held_out_correlations_survive_for_a_real_pairing_and_not_for_a_shuffled_one():
    K, L, _ = _two_views(n=100, d=6, shared=2, noise=0.1, seed=3)
    ho = KC.held_out_correlations(K, L, ridge=0.1, folds=5, seed=0, components=3)
    assert ho.mean[0] > 0.8 and ho.mean[1] > 0.8
    null = KC.permutation_null(K, L, ridge=0.1, permutations=20, folds=5, seed=0, components=3)
    assert null.shape == (20, 3)
    assert np.nanmean(null[:, 0]) < 0.5
    assert np.nanquantile(null[:, 0], 0.95) < ho.mean[0]


def test_the_ridge_is_chosen_on_held_out_molecules_not_on_the_fit():
    """With nearly as many directions as molecules the fit says 1 at every small ridge; the
    held-out correlation says much less, and that is the number the choice is made on."""
    K, L, _ = _two_views(n=40, d=30, shared=2, noise=0.1, seed=1)
    grid = (1e-4, 1e-2, 1.0)
    choice = KC.choose_ridge(K, L, grid=grid, folds=5, seed=0, components=2)
    assert set(choice.scores) == set(grid)
    assert choice.ridge == max(choice.scores, key=lambda c: choice.scores[c])
    assert KC.kcca(K, L, ridge=1e-4).rho[0] > 0.999       # the fit is the degeneracy
    assert max(choice.scores.values()) < 0.5               # the held-out number is not
    assert "held-out" in choice.describe()


def test_analyse_reports_the_gates_and_the_table():
    K, L, Z = _two_views(n=80, d=6, shared=2, noise=0.1, seed=5)
    rng = np.random.RandomState(0)
    # the activity sits on a direction the two views do NOT share
    fit0 = KC.kcca(K, L, ridge=0.01)
    private = fit0.u[:, -1]
    y = private / private.std() + 0.1 * rng.normal(size=len(private))
    res = KC.analyse(K, L, y=y, ridge=0.01, components=2, folds=5, seed=0, permutations=15)
    rows = res.rows()
    assert len(rows) == 2 and rows[0]["rho_held_out"] > 0.8 and rows[0]["contained"]
    assert res.real_components() >= 1
    assert res.p_values()[0] < 0.1
    assert res.contained.sum() >= 1 and (~res.contained).sum() >= 1
    assert res.q2_private > res.q2_shared and res.q2_private > 0.5
    d, lo, hi = res.q2_difference
    assert d > 0 and lo <= d <= hi
    assert res.private_coef is not None and abs(res.private_coef.sum()) < 1e-8
    text = res.describe()
    assert "(i)" in text and "(ii)" in text and "held out" in text and "contained" in text


def test_the_projection_splits_the_kernel_exactly():
    K, L, _ = _two_views(n=60, d=8, shared=2, seed=2)
    fit = KC.kcca(K, L, ridge=0.3)
    Kc = KC.centre(K)
    every = np.arange(fit.n_components)
    # all directions together span the whole centred kernel
    assert np.allclose(KC.projected_kernel(fit, every), Kc, atol=1e-8)
    K_S = KC.projected_kernel(fit, [0, 1])
    K_P = Kc - K_S
    # a projection: applying it twice changes nothing, and the two parts are orthogonal
    w, U = np.linalg.eigh(Kc)
    keep = w > 1e-10 * w.max()
    Kc_inv = (U[:, keep] / w[keep]) @ U[:, keep].T       # pseudo-inverse of Kc
    assert np.allclose(K_S @ Kc_inv @ K_S, K_S, atol=1e-8)
    assert np.allclose(K_S @ Kc_inv @ K_P, 0.0, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(K_P) > -1e-8)
    assert KC.projected_kernel(fit, []).shape == Kc.shape and not KC.projected_kernel(fit, []).any()


def test_contained_directions_need_a_held_out_correlation_above_the_null_and_the_share():
    K, L, _ = _two_views(n=80, d=6, shared=2, noise=0.1, seed=5)
    fit = KC.kcca(K, L, ridge=0.01)
    ho = KC.held_out_correlations(K, L, 0.01, folds=5, seed=0)
    c = KC.contained_directions(fit, ho, null=None, share=0.5)
    assert c[0] and c[1] and not c[-1]
    # a null that nothing clears
    null = np.ones((5, ho.n_components))
    assert not KC.contained_directions(fit, ho, null=null).any()
    # the share can be made impossible
    assert not KC.contained_directions(fit, ho, share=1.0).any()


def test_private_coefficients_reproduce_the_model_on_the_private_kernel():
    K, L, _ = _two_views(n=60, d=8, shared=2, seed=2)
    fit = KC.kcca(K, L, ridge=0.3)
    Kc = KC.centre(K)
    contained = np.zeros(fit.n_components, bool)
    contained[:2] = True
    K_P = Kc - KC.projected_kernel(fit, np.flatnonzero(contained))
    rng = np.random.RandomState(1)
    beta = rng.normal(size=K.shape[0])
    pred = K_P @ beta
    coef = KC.private_coefficients(fit, contained, beta)
    assert abs(coef.sum()) < 1e-8
    # a model with these coefficients on the raw kernel predicts the same thing up to a constant
    other = Kc @ coef
    assert np.allclose(pred - pred.mean(), other - other.mean(), atol=1e-8)


def test_activity_on_kernel_is_a_nested_cross_validation():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 4))
    K = X @ X.T
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.normal(size=60)
    q2, ycv, lam = KC.activity_on_kernel(K, y, folds=5, seed=0)
    assert q2 > 0.9 and np.isfinite(ycv).all() and lam > 0
    q2_noise, _, _ = KC.activity_on_kernel(K, rng.normal(size=60), folds=5, seed=0)
    assert q2_noise < 0.2
    q2_zero, ycv_zero, _ = KC.activity_on_kernel(np.zeros((60, 60)), y)
    assert np.isnan(q2_zero) and np.isnan(ycv_zero).all()


# ------------------------------------------------------------ fields and co-fields


SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(Br)cc1O", "c1ccc(N)cc1C", "c1ccc(Br)cc1C", "c1ccc(F)cc1N"]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


@pytest.fixture(scope="module")
def field_views():
    mols = [_mol(s) for s in SMILES]
    fs = FieldSet.create(alpha=1.0)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    probe = CMFModel(fieldset=fs, method="krr", normalize=True)
    K = probe.joint_kernel(probe.prepare_training_kernels(table))
    L = fingerprint_kernel(mols)
    return mols, table, probe, K, L


def test_the_fingerprint_kernel_is_a_tanimoto_kernel(field_views):
    mols, _t, _p, _K, L = field_views
    n = len(mols)
    assert L.shape == (n, n) and np.allclose(np.diag(L), 1.0) and np.allclose(L, L.T)
    assert np.all(L >= 0) and np.all(L <= 1)
    assert np.all(np.linalg.eigvalsh(L) > -1e-10)


def test_a_component_is_a_model_with_a_cofield(field_views):
    _m, table, probe, K, L = field_views
    fit = KC.kcca(K, L, ridge=0.3)
    m = KC.component_model(probe, fit.a[:, 0], fit.u[:, 0])
    assert np.allclose(m.coef, fit.a[:, 0]) and m.bias == 0.0
    centres, coeffs, widths = m.cofield_coefficients(0)
    assert len(centres) == len(coeffs) == len(widths) == table.coords.shape[0]
    # the component predicts its own variate on the training molecules
    pred = m.joint_kernel(m.K_fields_train) @ m.coef
    assert np.allclose(pred - pred.mean(), fit.u[:, 0] - fit.u[:, 0].mean(), atol=1e-8)


def test_the_two_cofields_of_one_component_agree_between_two_field_views(field_views):
    """View B is the same molecules with every pose slightly moved: a shared component should
    give the same co-field from either side, and two different components should not."""
    mols, table, probe, K, _L = field_views
    rng = np.random.RandomState(0)
    moved = []
    for m in mols:
        mm = Chem.Mol(m)
        conf = mm.GetConformer()
        X = np.array(conf.GetPositions(), float)
        X = X + rng.normal(scale=0.15, size=X.shape)
        for i, xyz in enumerate(X):
            conf.SetAtomPosition(i, [float(v) for v in xyz])
        moved.append(mm)
    fs = probe.fieldset.copy()
    table_b = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(moved)])
    probe_b = CMFModel(fieldset=fs, method="krr", normalize=True)
    Kb = probe_b.joint_kernel(probe_b.prepare_training_kernels(table_b))
    fit = KC.kcca(K, Kb, ridge=0.1)
    assert fit.rho[0] > 0.9
    ma = [KC.component_model(probe, fit.a[:, i], fit.u[:, i]) for i in range(3)]
    mb = [KC.component_model(probe_b, fit.b[:, i], fit.v[:, i]) for i in range(3)]
    M = KC.component_cosines(ma, mb)
    diag, off = KC.cosine_summary(M)
    assert M.shape == (3, 3)
    assert M[0, 0] > 0.8
    assert diag > off
    # the same cosine the transfer command would print
    assert cofield_overlap(ma[0], mb[0], strict=False).cosine == pytest.approx(M[0, 0])


def test_a_component_model_refuses_a_probe_without_kernels(field_views):
    _m, _t, probe, _K, _L = field_views
    with pytest.raises(ValueError, match="training kernels"):
        KC.component_model(CMFModel(fieldset=probe.fieldset.copy()), np.zeros(3))
    with pytest.raises(ValueError, match="coefficients"):
        KC.component_model(probe, np.zeros(3))


def test_mismatched_kernels_are_refused():
    with pytest.raises(ValueError, match="same size"):
        KC.kcca(np.eye(5), np.eye(4))


# ------------------------------------------------------------------ the command


def test_the_kcca_command_runs_both_pairings_and_writes_what_it_says(tmp_path, capsys):
    from cmfqsar.cli import main
    from cmfqsar.dataio import load_model

    path = os.path.join(str(tmp_path), "set.sdf")
    rng = np.random.RandomState(0)
    with Chem.SDWriter(path) as w:
        for i, smi in enumerate(SMILES):
            m = _mol(smi)
            m.SetProp("_Name", f"m{i}")
            m.SetProp("activity", f"{rng.normal():.3f}")
            w.write(m)
    comp = os.path.join(str(tmp_path), "components")
    csv_out = os.path.join(str(tmp_path), "variates.csv")
    main(["kcca", path, "--activity", "activity", "--alpha", "1.0", "--folds", "3",
          "--permutations", "5", "--components", "2", "--quiet", "--save-components", comp,
          "--out", csv_out])
    text = capsys.readouterr().out
    assert "View B: Tanimoto" in text and "(i)" in text and "(ii)" in text
    assert "a fingerprint has no co-field" in text
    assert os.path.exists(os.path.join(comp, "A1.cmf"))
    assert os.path.exists(os.path.join(comp, "private.cmf"))
    model, extra = load_model(os.path.join(comp, "A1.cmf"))
    assert extra["kcca_component"] == 1 and model.coef.shape == (len(SMILES),)
    centres, coeffs, _w = model.cofield_coefficients(0)
    assert len(coeffs) == len(centres)
    with open(csv_out) as fh:
        header = fh.readline().strip().split(",")
    assert header == ["name", "activity", "u1", "u2", "v1", "v2"]
    # the same molecules as a second field view: the co-field cosines are printed
    main(["kcca", path, "--alpha", "1.0", "--folds", "3", "--permutations", "0",
          "--components", "2", "--quiet", "--other", "dataset", "--other-dataset", path])
    text = capsys.readouterr().out
    assert "(iii) cosine between the co-field" in text
    assert "readable from either side" in text
    # two attenuations draw two different functions: the cosine is refused with the reason
    main(["kcca", path, "--alpha", "1.0", "--folds", "3", "--permutations", "0",
          "--components", "2", "--quiet", "--other", "dataset", "--other-dataset", path,
          "--other-alpha", "0.7"])
    text = capsys.readouterr().out
    assert "cannot be compared by a cosine" in text and "alpha" in text
    # different fields on the two sides have no common cosine
    main(["kcca", path, "--fields", "steric,hydrophobic", "--alpha", "1.0", "--folds", "3",
          "--permutations", "0", "--quiet", "--other", "fields",
          "--other-fields", "electrostatic,hba,hbd"])
    text = capsys.readouterr().out
    assert "cannot be compared by a cosine" in text and "different fields" in text


def test_the_kcca_command_refuses_what_it_cannot_match(tmp_path):
    from cmfqsar.cli import main

    path = os.path.join(str(tmp_path), "set.sdf")
    other = os.path.join(str(tmp_path), "other.sdf")
    with Chem.SDWriter(path) as w, Chem.SDWriter(other) as w2:
        for i, smi in enumerate(SMILES):
            m = _mol(smi)
            m.SetProp("_Name", f"m{i}")
            w.write(m)
            m.SetProp("_Name", f"x{i}")
            w2.write(m)
    with pytest.raises(SystemExit, match="matched by name"):
        main(["kcca", path, "--other", "dataset", "--other-dataset", other, "--quiet",
              "--permutations", "0"])
    with pytest.raises(SystemExit, match="other-fields"):
        main(["kcca", path, "--other", "fields", "--quiet", "--permutations", "0"])
