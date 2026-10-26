"""Tests for binary and one-class classification in the CMF framework."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel, classification_scores, cv_loss, grid_search  # noqa: E402

# two chemotypes: substituted phenols (class +1) and aliphatic amines (class -1)
ACTIVE = ["c1ccccc1O", "c1ccc(C)cc1O", "c1ccc(F)cc1O", "c1ccc(Cl)cc1O", "c1ccc(OC)cc1O", "c1ccc(N)cc1O"]
INACTIVE = ["CCCCN", "CCCCCN", "CC(C)CN", "CCCCCCN", "CCN(C)C", "CCCN(C)C"]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


def _set():
    mols = [_mol(s, i) for i, s in enumerate(ACTIVE + INACTIVE)]
    classes = np.array([1.0] * len(ACTIVE) + [-1.0] * len(INACTIVE))
    fs = FieldSet.create(alpha=0.6)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    return fs, mf, classes


def test_classification_scores_on_a_known_confusion_matrix():
    y = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=float)
    d = np.array([2.0, 1.0, 0.5, -0.5, -2.0, -1.0, 0.5, 1.5])      # 3 TP, 1 FN, 2 TN, 2 FP
    s = classification_scores(y, d)
    assert (s["tp"], s["fn"], s["tn"], s["fp"]) == (3.0, 1.0, 2.0, 2.0)
    assert s["accuracy"] == pytest.approx(5 / 8)
    assert s["sensitivity"] == pytest.approx(0.75)
    assert s["specificity"] == pytest.approx(0.5)
    assert s["balanced_accuracy"] == pytest.approx(0.625)
    assert s["mcc"] == pytest.approx(0.25819889, abs=1e-6)
    assert 0.0 <= s["auc"] <= 1.0
    # a perfect ranking gives AUC 1, a reversed one gives 0
    assert classification_scores(y, y)["auc"] == pytest.approx(1.0)
    assert classification_scores(y, -y)["auc"] == pytest.approx(0.0)


def test_binary_classification_separates_two_chemotypes():
    fs, mf, classes = _set()
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, task="classification", method="svc", params={"C": 10.0}, normalize=True)
    model.fit(table, classes, cv_folds=4)
    st = model.stats
    assert st.task == "classification"
    assert st.fit["accuracy"] == pytest.approx(1.0)        # the two chemotypes are separable
    assert st.cv["balanced_accuracy"] > 0.8
    assert 0.0 <= st.cv["auc"] <= 1.0
    # the decision values have the right sign per class
    d = st.extra["y_fit"]
    assert np.all(d[classes > 0] > 0) and np.all(d[classes < 0] < 0)
    # the model is sparse: fewer support vectors than training molecules is possible but not required
    assert len(model.learner_.support_) <= len(classes)


def test_classification_decomposition_rules_hold():
    """Atom and training-molecule contributions must sum to the decision value."""
    fs, mf, classes = _set()
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, task="classification", method="svc", params={"C": 10.0}, normalize=True)
    model.fit(table, classes, cv_folds=0)
    for mol in mf[:3]:
        d = model.predict(AtomTable.from_molecules([mol]))[0]
        assert model.atom_contributions(mol).sum() + model.bias == pytest.approx(d, abs=1e-8)
        assert model.training_contributions(mol).sum() + model.bias == pytest.approx(d, abs=1e-8)
    # the co-field of a classifier is still a combination of the training-set fields
    centers, coeffs, widths = model.cofield_coefficients(0)
    assert len(centers) == len(coeffs) == len(widths)
    assert np.any(coeffs > 0) and np.any(coeffs < 0)       # both classes contribute, with opposite signs


def test_one_class_model_fits_the_target_class_only():
    """The decision function must agree with scikit-learn's own one-class convention.

    (The asymptotic bound "outliers <= nu" is not asserted here: on a few nearly
    identical molecules the kernel is almost constant, most points end exactly on the
    boundary, and scikit-learn itself reports more outliers than nu.  On a real data set
    the bound is respected - 31 thrombin actives give 94 % inside at nu = 0.05.)
    """
    from sklearn.svm import OneClassSVM
    fs, mf, classes = _set()
    actives = [m for m, c in zip(mf, classes) if c > 0]
    table = AtomTable.from_molecules(actives)
    for nu in (0.1, 0.5):
        model = CMFModel(fieldset=fs, task="oneclass", method="ocsvm", params={"nu": nu}, normalize=True)
        model.fit(table, np.ones(len(actives)), cv_folds=0)
        assert model.stats.task == "oneclass"
        K = model.joint_kernel(model.K_fields_train)
        reference = OneClassSVM(kernel="precomputed", nu=nu).fit(K)
        assert model.stats.inlier_fraction == pytest.approx(float(np.mean(reference.predict(K) > 0)))
        assert np.allclose(model.predict(table), reference.decision_function(K), atol=1e-8)
    # molecules of the other chemotype should score lower than the target class
    inactives = [m for m, c in zip(mf, classes) if c < 0]
    d_act = model.predict(AtomTable.from_molecules(actives))
    d_inact = model.predict(AtomTable.from_molecules(inactives))
    assert d_act.mean() > d_inact.mean()


def test_one_class_rejects_cross_validated_hyperparameter_search():
    fs, mf, classes = _set()
    actives = [m for m, c in zip(mf, classes) if c > 0]
    model = CMFModel(fieldset=fs, task="oneclass", method="ocsvm", normalize=True)
    with pytest.raises(ValueError, match="one-class"):
        grid_search(AtomTable.from_molecules(actives), np.ones(len(actives)), model, [0.5], nus=[0.1, 0.5])


def test_cv_loss_is_task_aware():
    fs, mf, classes = _set()
    model = CMFModel(fieldset=fs, task="classification")
    loss, quality = cv_loss(model, classes, classes)            # a perfect classifier
    assert loss == pytest.approx(0.0) and quality == pytest.approx(1.0)
    loss_bad, _ = cv_loss(model, classes, -classes)             # a perfectly wrong one
    assert loss_bad == pytest.approx(1.0)
    reg = CMFModel(fieldset=fs, task="regression")
    loss_r, q2 = cv_loss(reg, classes, classes)
    assert loss_r == pytest.approx(0.0) and q2 == pytest.approx(1.0)


def test_model_state_keeps_the_task():
    fs, mf, classes = _set()
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, task="classification", method="svc", normalize=True)
    model.fit(table, classes, cv_folds=0)
    back = CMFModel.from_state(model.state())
    assert back.task == "classification" and back.method == "svc"
    assert np.allclose(back.coef, model.coef)


def test_screening_scores_and_enrichment_curve():
    from cmfqsar.model import enrichment_curve, screening_scores
    y = np.array([1.0] * 10 + [-1.0] * 90)                 # 10 % actives
    perfect = np.concatenate([np.full(10, 5.0), np.zeros(90)])
    s = screening_scores(y, perfect)
    assert s["auc"] == pytest.approx(1.0)
    assert s["max_ef"] == pytest.approx(10.0)
    for pct in (1, 5, 10):
        assert s[f"ef_{pct}"] == pytest.approx(10.0)       # the maximum attainable
    assert s["n_actives"] == 10 and s["n_decoys"] == 90
    rev = screening_scores(y, -perfect)
    assert rev["auc"] == pytest.approx(0.0) and rev["ef_10"] == pytest.approx(0.0)
    # a ranking with no negatives has no screening metrics
    assert not np.isfinite(screening_scores(np.ones(5), np.arange(5.0))["auc"])
    x, found = enrichment_curve(y, perfect)
    assert x[0] == 0.0 and found[0] == 0.0
    assert x[-1] == pytest.approx(1.0) and found[-1] == pytest.approx(1.0)
    assert found[np.searchsorted(x, 0.10)] == pytest.approx(1.0)   # all actives in the first 10 %
    assert enrichment_curve(np.ones(5), np.arange(5.0)) is None


def test_early_recognition_sees_what_the_enrichment_factor_cannot():
    """At a small fraction the enrichment factor saturates; BEDROC still separates the rankings."""
    from cmfqsar.model import rie_bedroc, screening_scores
    n_act, n = 30, 2000
    y = np.r_[np.ones(n_act), -np.ones(n - n_act)]
    rng = np.random.default_rng(0)
    rest = rng.normal(size=n - n_act)
    perfect = np.r_[np.full(n_act, 10.0), rest]
    s_perfect = screening_scores(y, perfect)
    assert s_perfect["auc"] == pytest.approx(1.0)
    assert s_perfect["bedroc_20"] == pytest.approx(1.0, abs=1e-6)
    assert s_perfect["bedroc_160_9"] == pytest.approx(1.0, abs=1e-6)
    # the maximum is reported, and the factor reaches it
    assert s_perfect["ef_1"] == pytest.approx(s_perfect["max_ef_1"])

    # two rankings that fill the top 1 % equally but differ below it: the factor cannot
    # distinguish them, the early-recognition metrics can
    top = int(round(0.01 * n))
    a = np.zeros(n); b = np.zeros(n)
    a[:top] = 10.0                      # the first actives on top, the others just below
    a[top:n_act] = 9.0
    b[:top] = 10.0
    b[top:n_act] = -5.0                 # the rest of the actives pushed to the bottom
    a[n_act:] = rng.uniform(-1, 1, n - n_act)
    b[n_act:] = rng.uniform(-1, 1, n - n_act)
    sa, sb = screening_scores(y, a), screening_scores(y, b)
    assert sa["ef_1"] == pytest.approx(sb["ef_1"])          # identical, and both at the maximum
    assert sa["bedroc_20"] > sb["bedroc_20"] + 0.1
    assert sa["auc"] > sb["auc"]

    # a random ranking scores near zero, a reversed one at zero
    assert screening_scores(y, rng.normal(size=n))["bedroc_20"] < 0.2
    assert screening_scores(y, -perfect)["bedroc_20"] == pytest.approx(0.0, abs=1e-6)
    # and the component functions refuse a degenerate problem rather than inventing a number
    assert not np.isfinite(rie_bedroc(np.arange(1, 4), 3, 3)[0])


def test_decoys_are_a_separate_subset_and_count_as_inactive():
    from cmfqsar.dataio import Dataset, _subset_label
    assert _subset_label("decoy") == "decoy" and _subset_label("background") == "decoy"
    assert _subset_label("3") == "decoy" and _subset_label("1") == "train"
    mols = [_mol(s, i) for i, s in enumerate(ACTIVE[:4])]
    ds = Dataset(mols=mols, names=list("abcd"), activity=np.array([8.0, 5.0, 9.0, np.nan]),
                 subset=np.array(["train", "train", "decoy", "decoy"], dtype=object))
    assert list(ds.decoy_indices) == [2, 3]
    c = ds.classes_from_threshold(6.0)
    # molecule 2 is highly "active" by its activity value but is a decoy, so it stays negative
    assert list(c[:3]) == [1.0, -1.0, -1.0] and c[3] == -1.0


def test_a_flexible_decoy_is_never_posed_worse_than_a_rigid_one():
    """The conformation from the file is among the candidates, so the best can only improve."""
    from cmfqsar.decoys import align_decoys, align_decoys_flexible
    fs, mf, classes = _set()
    ref = AtomTable.from_molecules([m for m, c in zip(mf, classes) if c > 0])
    mol = _mol(INACTIVE[0], 11)
    rigid = align_decoys([build_fields(mol, fs, "decoy")], ref, fs, restarts=1)[0]

    # one conformation is the rigid alignment itself, to the last digit
    one_f, one = align_decoys_flexible([mol], ["decoy"], ref, fs, n_conformers=1, restarts=1)
    assert one[0].similarity_after == pytest.approx(rigid.similarity_after, abs=1e-12)
    assert one[0].n_conformers == 1 and one[0].conformer == 0
    assert one_f[0].coords.shape == mf[0].coords.shape[:1] + (3,) or one_f[0].coords.shape[1] == 3

    # several conformations can only do better, and the winner is reported
    many_f, many = align_decoys_flexible([mol], ["decoy"], ref, fs, n_conformers=5, restarts=1)
    r = many[0]
    assert r.similarity_after >= one[0].similarity_after - 1e-12
    assert 1 <= r.n_conformers <= 5 and 0 <= r.conformer < r.n_conformers
    assert 0.0 <= r.similarity_after <= 1.0
    # the molecule keeps its atoms: the chosen conformation is written back atom for atom
    assert many_f[0].coords.shape[0] == mol.GetNumAtoms()
    assert np.allclose(many_f[0].coords, r.coords)


def test_decoy_aligner_gradient_and_improvement():
    """The decoy aligner must have an exact gradient and give a molecule a better pose."""
    from cmfqsar.decoys import DecoyAligner, align_decoys
    fs, mf, classes = _set()
    ref = AtomTable.from_molecules([m for m, c in zip(mf, classes) if c > 0])
    al = DecoyAligner(ref, fs)
    probe = build_fields(_mol(INACTIVE[0], 11), fs, "decoy")
    X0 = probe.coords.copy()
    c = X0.mean(axis=0)
    sp = al._self_norm(probe)
    rng = np.random.RandomState(0)
    for _ in range(2):
        p = np.concatenate([rng.normal(scale=0.4, size=3), rng.normal(scale=0.6, size=3)])
        _, g = al.objective(probe, X0, c, sp, p)
        num = np.zeros(6)
        for k in range(6):
            h = 1e-6
            pp = p.copy(); pp[k] += h
            pm = p.copy(); pm[k] -= h
            num[k] = (al.objective(probe, X0, c, sp, pp)[0] - al.objective(probe, X0, c, sp, pm)[0]) / (2 * h)
        assert np.max(np.abs(g - num)) < 1e-5 * max(1.0, np.max(np.abs(num)))
    # the similarity is a cosine, so it lies in [0, 1], and alignment can only improve it
    res = align_decoys([probe], ref, fs, restarts=3)
    assert len(res) == 1
    r = res[0]
    assert 0.0 <= r.similarity_before <= 1.0 and 0.0 <= r.similarity_after <= 1.0
    assert r.similarity_after >= r.similarity_before - 1e-9
    assert r.best_match in ref.names
    assert np.allclose(probe.coords, r.coords)


def test_decoy_similarity_is_pose_invariant_in_its_self_term():
    """The normalisation of the probe must not depend on where the probe sits."""
    from cmfqsar.decoys import DecoyAligner
    from scipy.spatial.transform import Rotation
    fs, mf, classes = _set()
    ref = AtomTable.from_molecules([m for m, c in zip(mf, classes) if c > 0])
    al = DecoyAligner(ref, fs)
    probe = build_fields(_mol(INACTIVE[1], 12), fs, "decoy")
    s0 = al._self_norm(probe)
    probe.coords = probe.coords @ Rotation.from_rotvec([0.3, -0.2, 0.7]).as_matrix().T + np.array([3.0, -2.0, 1.0])
    assert np.allclose(al._self_norm(probe), s0, rtol=1e-10)


def test_dataset_class_labels():
    from cmfqsar.dataio import Dataset
    mols = [_mol(s, i) for i, s in enumerate(ACTIVE[:3])]
    for m, v in zip(mols, ("active", "inactive", "1")):
        m.SetProp("flag", v)
    ds = Dataset(mols=mols, names=["a", "b", "c"], activity=np.array([7.0, 5.0, np.nan]),
                 subset=np.array(["train"] * 3, dtype=object))
    c = ds.classes_from_threshold(6.0)
    assert c[0] == 1.0 and c[1] == -1.0 and np.isnan(c[2])
    c2 = ds.classes_from_threshold(6.0, active_above=False)
    assert c2[0] == -1.0 and c2[1] == 1.0
    c3 = ds.classes_from_property("flag")
    assert list(c3) == [1.0, -1.0, 1.0]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
