"""Low-rank kernels for large training sets (plan item B6).

The claim of the item is that a model fitted in the space of m landmarks is the same model as
the exact one when the landmarks span the data, and a controlled approximation when they do
not. These tests fix that claim piece by piece: the feature map reproduces the Nystrom kernel,
the primal fit is the dual fit, the evidence and the variance are the same quantities computed
another way, the co-field keeps its form with the index set changed from molecules to
landmarks, and the memory of a fit does not depend on the number of molecules.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar import lowrank as LR  # noqa: E402
from cmfqsar.cofields import evaluate_gaussians  # noqa: E402
from cmfqsar.evidence import log_evidence as exact_evidence  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, krr_fit  # noqa: E402

SMILES = ["c1ccccc1C(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
          "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O", "c1ccc(Br)cc1C(=O)O",
          "c1ccc(C#N)cc1C(=O)O", "c1ccc(OC)cc1C(=O)O", "c1ccc(CC)cc1C(=O)O"]


def _series(fields=("electrostatic", "steric"), alpha=0.5):
    fs = FieldSet.create(physchem=list(fields), alpha=float(alpha))
    mols = []
    for k, smi in enumerate(SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        AllChem.EmbedMolecule(m, p)
        mols.append(build_fields(m, fs, f"m{k}"))
    table = AtomTable.from_molecules(mols)
    y = np.array([4.0 + 0.3 * k for k in range(len(mols))])
    return fs, table, y


def _exact(fs, table, y, lam=0.1):
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    model.params["lam"] = float(lam)
    Kf = np.asarray(model.prepare_training_kernels(table))
    K = np.asarray(model.joint_kernel(Kf), float)
    model.fit(table, y, cv_folds=3, cv_seed=0, reuse_kernels=True)
    return model, K


# ------------------------------------------------------------- the feature map


def test_the_features_reproduce_the_kernel_when_the_landmarks_are_the_molecules():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    Phi = low.transform(table)
    assert Phi.shape[0] == table.n_mol
    assert np.allclose(Phi @ Phi.T, K, atol=1e-10)


def test_the_whitener_is_the_pseudo_inverse_square_root():
    A = np.array([[2.0, 1.0], [1.0, 2.0]])
    W = LR.whitener(A)
    assert np.allclose(W @ W.T, np.linalg.inv(A))


def test_a_landmark_set_with_a_duplicate_does_not_blow_up():
    """The eigenvalue that belongs to the duplicate is dropped, not divided by."""
    A = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    W = LR.whitener(A)
    assert W.shape[1] == 2, "a rank-two matrix gives two columns"
    assert np.all(np.isfinite(W))


def test_the_features_are_built_the_same_way_in_blocks():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    assert np.allclose(low.transform(table, block=3), low.transform(table, block=1000))


def test_a_grid_of_centres_covers_the_molecules_and_nothing_else():
    fs, table, y = _series()
    Z = LR.gaussian_centres(table, 2.0, 2)
    assert Z.n_mol > 10
    from scipy.spatial import cKDTree
    d, _i = cKDTree(table.coords).query(Z.coords, k=1)
    assert float(d.max()) <= 2.0 + 1e-9, "a centre far from every atom carries nothing"


def test_landmark_molecules_are_a_subset_and_are_reproducible():
    fs, table, y = _series()
    a, rows_a = LR.landmark_molecules(table, 4, seed=1)
    b, rows_b = LR.landmark_molecules(table, 4, seed=1)
    assert a.n_mol == 4 and list(rows_a) == list(rows_b)
    assert LR.landmark_molecules(table, 999, seed=1)[0].n_mol == table.n_mol


# ------------------------------------------------------------------- the fit


def test_the_primal_fit_is_the_dual_fit():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    Phi = low.transform(table)
    g = LR.gram(low, table, y)
    w, b = LR.fit(g, 0.1)
    a, b2 = krr_fit(K, y, 0.1)
    assert np.allclose(LR.predict(Phi, w, b), K @ a + b2, atol=1e-8)


def test_the_fit_can_be_made_without_a_bias():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    g = LR.gram(low, table, y)
    w, b = LR.fit(g, 0.1, fit_bias=False)
    a, b2 = krr_fit(K, y, 0.1, fit_bias=False)
    assert b == 0.0 and b2 == 0.0
    assert np.allclose(LR.predict(low.transform(table), w, 0.0), K @ a, atol=1e-8)


def test_the_gram_does_not_depend_on_the_block_size():
    fs, table, y = _series()
    low = LR.build(table, fs)
    a = LR.gram(low, table, y, block=2)
    b = LR.gram(low, table, y, block=1000)
    assert np.allclose(a.PtP, b.PtP) and np.allclose(a.Pty, b.Pty)
    assert a.n == b.n and a.sum_y == pytest.approx(b.sum_y)


def test_the_memory_of_a_fit_is_the_rank_and_not_the_molecules():
    small = LR.memory(1000, 300, 5)
    large = LR.memory(100000, 300, 5)
    assert small == large, "the Gram is m by m whatever n is"
    assert LR.memory(100000, 300, 5, exact=True) > 1e11


# ------------------------------------------------- the evidence and the variance


def test_the_evidence_is_the_same_number_computed_another_way():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    g = LR.gram(low, table, y)
    for lam in (0.01, 0.1, 1.0):
        assert LR.log_evidence(g, lam) == pytest.approx(exact_evidence(K, y, lam), abs=1e-6)


def test_the_evidence_without_a_bias_matches_too():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    g = LR.gram(low, table, y)
    assert LR.log_evidence(g, 0.1, fit_bias=False) == pytest.approx(
        exact_evidence(K, y, 0.1, fit_bias=False), abs=1e-6)


def test_the_variance_is_the_gaussian_process_variance():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    Phi = low.transform(table)
    g = LR.gram(low, table, y)
    lam = 0.1
    exact = np.diag(K) - np.einsum("ij,jk,ik->i", K,
                                   np.linalg.inv(K + lam * np.eye(len(y))), K)
    k_self = np.full(len(y), float(np.sum(np.asarray(fs.mixing_vector())[:len(low.specs)]
                                          * model.scale)))
    assert np.allclose(LR.variance(g, Phi, lam, k_self=k_self), np.maximum(exact, 0.0),
                       atol=1e-8)


def test_without_the_self_kernel_the_variance_is_the_smaller_one():
    """The term that is dropped is what keeps the variance up away from the landmarks."""
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    few, _rows = LR.landmark_molecules(table, 3, seed=0)
    low = LR.build(few, fs, scale=model.scale)
    Phi = low.transform(table)
    g = LR.gram(low, table, y)
    k_self = np.full(len(y), float(np.sum(np.asarray(fs.mixing_vector())[:len(low.specs)]
                                          * model.scale)))
    kept = LR.variance(g, Phi, 0.1, k_self=k_self)
    plain = LR.variance(g, Phi, 0.1)
    assert np.all(kept >= plain - 1e-12)
    assert float(np.mean(kept - plain)) > 1e-3, "three landmarks cannot span ten molecules"


# ------------------------------------------------------------- the interpretation


def test_the_cofield_is_the_same_function_over_the_landmarks():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    low = LR.build(table, fs, scale=model.scale)
    g = LR.gram(low, table, y)
    w, _b = LR.fit(g, 0.1)
    # the exact model with the same ridge, so the two co-fields are of one model
    a, _b2 = krr_fit(K, y, 0.1)
    model.coef = a
    pts = np.asarray(table.coords, float)[:50]
    for f in range(len(low.specs)):
        c0, k0, w0 = model.cofield_coefficients(f)
        want = evaluate_gaussians(pts, c0, k0, w0) * fs.specs[f].h * model.scale[f]
        c1, k1, w1 = LR.cofield_gaussians(low, w, f)
        assert np.allclose(evaluate_gaussians(pts, c1, k1, w1), want, atol=1e-8)


def test_the_landmark_coefficients_are_one_number_per_landmark_per_field():
    fs, table, y = _series()
    low = LR.build(table, fs)
    g = LR.gram(low, table, y)
    w, _b = LR.fit(g, 0.1)
    per = LR.landmark_coefficients(low, w)
    assert len(per) == len(low.specs)
    assert all(len(a) == table.n_mol for a in per)


# -------------------------------------------------------------- the whole path


def test_cross_validation_holds_molecules_out_and_predicts_them():
    fs, table, y = _series()
    pred, which = LR.cross_validate(LR.build(table, fs), table, y, 0.1, folds=5, seed=0)
    assert len(pred) == len(y) and set(which) == set(range(5))
    assert np.all(np.isfinite(pred))


def test_a_grid_of_centres_fits_and_is_not_far_from_the_exact_model():
    fs, table, y = _series()
    model, K = _exact(fs, table, y)
    Z = LR.gaussian_centres(table, 1.5, len([s for s in fs.specs if s.enabled]))
    low = LR.build(Z, fs, scale=model.scale, kind="grid")
    Phi = low.transform(table)
    rel = float(np.linalg.norm(Phi @ Phi.T - K) / np.linalg.norm(K))
    assert rel < 0.2, f"a 1.5 A grid should span a broad field; the error is {rel:.3f}"


def test_the_command_runs_and_reports_what_it_used(tmp_path, capsys):
    from cmfqsar.cli import main
    fs, table, y = _series()
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(SMILES):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(m, p)
            m.SetProp("_Name", f"m{k}")
            m.SetProp("activity", f"{4.0 + 0.3 * k:.2f}")
            w.write(m)
    capsys.readouterr()
    main(["lowrank", path, "--activity", "activity", "--landmarks", "molecules:5",
          "--fields", "electrostatic,steric", "--lams", "0.1", "--folds", "3"])
    text = capsys.readouterr().out
    assert "landmarks" in text and "columns of features" in text
    assert "chosen by the evidence" in text


# ------------------------------------------------------- the model, saved and loaded


def _fitted_low_rank(fs, table, y, lam=0.1, kind="molecules"):
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    Z = (table if kind == "molecules"
         else LR.gaussian_centres(table, 1.5, len([s for s in fs.specs if s.enabled])))
    model.fit_low_rank(table, y, Z, lam=lam, kind=kind, folds=3, seed=0)
    return model


def test_the_model_knows_it_was_fitted_the_low_rank_way():
    fs, table, y = _series()
    model = _fitted_low_rank(fs, table, y)
    assert model.is_low_rank and model.coef is None
    assert model.low_rank_.rank == model.low_weights_.size
    assert model.stats.extra["low_rank"]["landmarks"] == table.n_mol


def test_predicting_from_the_model_is_the_feature_fit():
    fs, table, y = _series()
    model = _fitted_low_rank(fs, table, y)
    Phi = model.low_rank_features(table)
    assert np.allclose(model.predict(table), Phi @ model.low_weights_ + model.bias)


def test_the_low_rank_model_is_the_exact_model_when_every_molecule_is_a_landmark():
    fs, table, y = _series()
    low = _fitted_low_rank(fs, table, y, lam=0.1, kind="molecules")
    model, K = _exact(fs, table, y, 0.1)
    a, b = krr_fit(K, y, 0.1)
    assert np.allclose(low.predict(table), K @ a + b, atol=1e-8)


def test_the_variance_comes_out_of_the_model_too():
    fs, table, y = _series()
    model = _fitted_low_rank(fs, table, y)
    mu, var = model.predict_with_variance(table)
    assert np.allclose(mu, model.predict(table))
    assert np.all(var >= 0) and np.all(np.isfinite(var))


def test_the_cofield_of_a_low_rank_model_is_over_its_landmarks():
    fs, table, y = _series()
    model = _fitted_low_rank(fs, table, y, kind="grid")
    centres, coeffs, widths = model.cofield_coefficients(0)
    assert len(centres) == model.low_rank_.landmarks.coords.shape[0]
    assert len(coeffs) == len(centres) and len(widths) == len(centres)


def test_a_low_rank_model_is_saved_and_read_back(tmp_path):
    from cmfqsar.dataio import load_model, save_model
    fs, table, y = _series()
    model = _fitted_low_rank(fs, table, y)
    path = os.path.join(str(tmp_path), "m.cmf")
    save_model(path, model, {"dataset": "toy"})
    back, extra = load_model(path)
    assert back.is_low_rank and extra["dataset"] == "toy"
    assert np.allclose(back.predict(table), model.predict(table))
    assert np.allclose(back.predict_with_variance(table)[1],
                       model.predict_with_variance(table)[1])
    c0, k0, w0 = model.cofield_coefficients(0)
    c1, k1, w1 = back.cofield_coefficients(0)
    assert np.allclose(c0, c1) and np.allclose(k0, k1) and np.allclose(w0, w1)


def test_an_ordinary_model_has_no_low_rank_state(tmp_path):
    from cmfqsar.dataio import load_model, save_model
    fs, table, y = _series()
    model, _K = _exact(fs, table, y)
    assert not model.is_low_rank
    path = os.path.join(str(tmp_path), "m.cmf")
    save_model(path, model)
    back, _extra = load_model(path)
    assert not back.is_low_rank
    assert np.allclose(back.predict(table), model.predict(table))


def test_the_low_rank_path_refuses_a_kernel_that_has_features_already():
    fs, table, y = _series()
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    model.kernel_source = "autocorrelation"
    with pytest.raises(ValueError, match="invariant kernel"):
        model.fit_low_rank(table, y, table, lam=0.1)


def test_build_saves_a_low_rank_model_that_predict_can_read(tmp_path, capsys):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(SMILES):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(m, p)
            m.SetProp("_Name", f"m{k}")
            m.SetProp("activity", f"{4.0 + 0.3 * k:.2f}")
            w.write(m)
    saved = os.path.join(str(tmp_path), "m.cmf")
    capsys.readouterr()
    main(["build", path, "--activity", "activity", "--fields", "electrostatic,steric",
          "--low-rank", "molecules:6", "--alphas", "0.5", "--lams", "0.1",
          "--test-fraction", "0", "--folds", "3", "--save", saved])
    text = capsys.readouterr().out
    assert "Low rank:" in text and "columns of features" in text
    assert os.path.exists(saved)
    main(["predict", saved, path, "--out", os.path.join(str(tmp_path), "p.csv")])
    out = capsys.readouterr().out
    assert "Predictions written" in out


# ------------------------------------------------------------- landmark atoms


def test_the_atoms_of_the_molecules_become_landmarks_where_they_stand():
    fs, table, y = _series()
    Z = LR.landmark_atoms(table, 2, min_distance=0.0)
    assert Z.n_mol == table.coords.shape[0], "one landmark per atom when nothing is thinned"
    assert np.allclose(np.sort(Z.coords, axis=0), np.sort(np.asarray(table.coords, float),
                                                          axis=0))
    assert np.allclose(Z.weights, 1.0), "a landmark is a basis function, not a copy of an atom"
    assert Z.weights.shape[1] == 2


def test_a_landmark_atom_keeps_the_radius_of_the_atom_it_stands_on():
    fs, table, y = _series()
    Z = LR.landmark_atoms(table, 2, min_distance=0.0, seed=None)
    assert np.allclose(Z.radii, np.asarray(table.radii, float))


def test_thinning_keeps_the_landmarks_apart_and_fewer():
    fs, table, y = _series()
    loose = LR.landmark_atoms(table, 2, min_distance=1.0, seed=0)
    tight = LR.landmark_atoms(table, 2, min_distance=0.0, seed=0)
    assert loose.n_mol < tight.n_mol
    from scipy.spatial import cKDTree
    d = cKDTree(loose.coords).query(loose.coords, k=2)[0][:, 1]
    assert float(d.min()) >= 1.0 - 1e-9


def test_the_atoms_of_a_subset_of_molecules_can_be_taken():
    fs, table, y = _series()
    part = LR.landmark_atoms(table, 2, molecules=3, min_distance=0.0, seed=0)
    whole = LR.landmark_atoms(table, 2, molecules=0, min_distance=0.0, seed=0)
    assert part.n_mol < whole.n_mol


def test_the_number_of_landmarks_can_be_capped():
    fs, table, y = _series()
    Z = LR.landmark_atoms(table, 2, min_distance=0.0, max_landmarks=17, seed=0)
    assert Z.n_mol == 17


def test_landmark_atoms_follow_a_narrow_field_where_a_grid_of_the_same_size_does_not():
    """The case plan item B6 named: a Gaussian narrow enough to fall between grid points."""
    fs, table, y = _series(alpha=5.0)
    model, K = _exact(fs, table, y)
    n_f = len([s for s in fs.specs if s.enabled])
    grid = LR.build(LR.gaussian_centres(table, 2.0, n_f), fs, scale=model.scale, kind="grid")
    atoms = LR.build(LR.landmark_atoms(table, n_f, min_distance=0.5, seed=0), fs,
                     scale=model.scale, kind="atoms")

    def error(low):
        Phi = low.transform(table)
        return float(np.linalg.norm(Phi @ Phi.T - K) / np.linalg.norm(K))

    assert error(atoms) < 0.2 * error(grid), (
        f"atoms {error(atoms):.2e} should be far better than a grid {error(grid):.2e} "
        f"at a narrow attenuation")


def test_the_command_takes_atoms_as_landmarks(tmp_path, capsys):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for k, smi in enumerate(SMILES):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(m, p)
            m.SetProp("_Name", f"m{k}")
            m.SetProp("activity", f"{4.0 + 0.3 * k:.2f}")
            w.write(m)
    capsys.readouterr()
    main(["lowrank", path, "--activity", "activity", "--landmarks", "atoms:0.75",
          "--fields", "electrostatic,steric", "--lams", "0.1", "--folds", "3"])
    text = capsys.readouterr().out
    assert "the atoms of" in text and "thinned to" in text
