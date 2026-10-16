"""Tests for alignment by prediction confidence (GP posterior variance)."""
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import so3_right_jacobian  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.varalign import VarianceAligner, variance_align  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N"]
Y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 0.8, 0.9, 2.2])


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


def _model(normalize=True, alpha=0.6):
    mols = [_mol(s, i) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(alpha=alpha)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.05}, normalize=normalize)
    model.fit(AtomTable.from_molecules(mf), Y, cv_folds=0)
    return model, fs, mols


def test_self_kernel_is_rigid_invariant():
    """K(x, x) must not depend on the pose, otherwise the variance criterion is not well posed."""
    model, fs, mols = _model()
    al = VarianceAligner(model, max_rmsd=None)
    mol = build_fields(mols[2], fs, "probe")
    _, kself0 = al._self_terms(mol)
    rng = np.random.RandomState(0)
    for _ in range(3):
        R = Rotation.random(random_state=rng).as_matrix()
        moved = build_fields(mols[2], fs, "moved")
        moved.coords = mol.coords @ R.T + rng.normal(scale=4.0, size=3)
        _, kself = al._self_terms(moved)
        assert abs(kself - kself0) <= 1e-10 * max(1.0, abs(kself0))


def test_analytic_gradient_matches_finite_differences():
    for normalize in (True, False):
        model, fs, mols = _model(normalize=normalize)
        mol = build_fields(mols[2], fs, "probe")
        X0 = mol.coords.copy()
        c = X0.mean(axis=0)
        rng = np.random.RandomState(0)
        for max_rmsd in (None, 0.3):          # 0.3 A forces the displacement penalty to be active
            al = VarianceAligner(model, max_rmsd=max_rmsd)
            sp, kself = al._self_terms(mol)
            p = np.concatenate([rng.normal(scale=0.4, size=3), rng.normal(scale=0.6, size=3)])
            _, g = al.objective(mol, X0, c, sp, kself, p)
            num = np.zeros(6)
            for k in range(6):
                h = 1e-6
                pp = p.copy(); pp[k] += h
                pm = p.copy(); pm[k] -= h
                num[k] = (al.objective(mol, X0, c, sp, kself, pp)[0]
                          - al.objective(mol, X0, c, sp, kself, pm)[0]) / (2 * h)
            assert np.max(np.abs(g - num)) < 1e-5 * max(1.0, np.max(np.abs(num)))


def test_so3_right_jacobian_limits():
    assert np.allclose(so3_right_jacobian(np.zeros(3)), np.eye(3))
    w = np.array([0.3, -0.2, 0.1])
    J = so3_right_jacobian(w)
    # numerical check: d/dt [R(w + t e_k) v] at t=0 equals (-R [v]_x J)[:, k]
    v = np.array([1.0, 2.0, -0.5])
    R = Rotation.from_rotvec(w).as_matrix()
    Vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    analytic = -R @ Vx @ J
    num = np.zeros((3, 3))
    for k in range(3):
        h = 1e-7
        wp = w.copy(); wp[k] += h
        wm = w.copy(); wm[k] -= h
        num[:, k] = (Rotation.from_rotvec(wp).as_matrix() @ v
                     - Rotation.from_rotvec(wm).as_matrix() @ v) / (2 * h)
    assert np.max(np.abs(analytic - num)) < 1e-6


def test_alignment_recovers_a_perturbed_pose():
    """A molecule of the training set, displaced and re-aligned, must come back."""
    model, fs, mols = _model()
    al = VarianceAligner(model, max_rmsd=None)
    probe = build_fields(mols[4], fs, "probe")
    X0 = probe.coords.copy()
    c = X0.mean(axis=0)
    ax = np.array([0.3, -0.5, 0.8]); ax /= np.linalg.norm(ax)
    R = Rotation.from_rotvec(ax * np.deg2rad(35)).as_matrix()
    probe.coords = (X0 - c) @ R.T + c + np.array([1.0, -0.8, 0.6])
    start = float(np.sqrt(((probe.coords - X0) ** 2).sum(1).mean()))
    res = al.align(probe, restarts=2, seed=0)
    end = float(np.sqrt(((probe.coords - X0) ** 2).sum(1).mean()))
    assert start > 1.5
    assert end < 0.5 * start and end < 1.0
    assert res.variance_after <= res.variance_before + 1e-12
    assert np.allclose(probe.coords, res.coords)


def test_displacement_guard_limits_the_move():
    model, fs, mols = _model()
    probe = build_fields(mols[4], fs, "probe")
    X0 = probe.coords.copy()
    c = X0.mean(axis=0)
    R = Rotation.from_rotvec(np.array([0.0, 0.0, 1.0]) * np.deg2rad(40)).as_matrix()
    probe.coords = (X0 - c) @ R.T + c + np.array([2.0, 0.0, 0.0])
    res = VarianceAligner(model, max_rmsd=0.5).align(probe, restarts=1, seed=0)
    assert res.rmsd < 0.75          # the penalty keeps the molecule near its starting pose


def test_variance_align_batch_runs_and_reports():
    model, fs, mols = _model()
    probes = [build_fields(m, fs, f"p{i}") for i, m in enumerate(mols[:3])]
    seen = []
    res = variance_align(probes, model, max_rmsd=1.0, restarts=0,
                         progress=lambda i, n: seen.append((i, n)))
    assert len(res) == 3 and seen[-1] == (3, 3)
    for r in res:
        assert np.isfinite(r.variance_after) and r.variance_after <= r.variance_before + 1e-9
        assert r.name.startswith("p")


def test_unalign_is_rigid_bounded_and_reproducible():
    """Randomising poses must preserve the internal geometry and respect its bounds."""
    from scipy.spatial.distance import pdist
    from cmfqsar.align import unalign
    _, fs, mols = _model()
    mf = [build_fields(m, fs, f"u{i}") for i, m in enumerate(mols)]
    d_before = [pdist(m.coords) for m in mf]
    moved = unalign(mf, max_rotation=30.0, max_translation=1.0, seed=7)
    d_after = [pdist(m.coords) for m in mf]
    # rigid: all interatomic distances unchanged
    for a, b in zip(d_before, d_after):
        assert np.max(np.abs(a - b)) < 1e-9
    assert all(0.0 < r for r in moved)
    # a 30 degree rotation about the centroid plus 1 A cannot displace more than
    # 2 sin(15 deg) * (max distance from the centroid) + 1 A
    for mol, r in zip(mf, moved):
        reach = float(np.max(np.linalg.norm(mol.coords - mol.coords.mean(axis=0), axis=1)))
        assert r <= 2 * np.sin(np.deg2rad(15.0)) * reach + 1.0 + 1e-6
    # same seed, same result
    mf2 = [build_fields(m, fs, f"u{i}") for i, m in enumerate(mols)]
    assert np.allclose(moved, unalign(mf2, max_rotation=30.0, max_translation=1.0, seed=7))
    # a different seed gives a different scramble
    mf3 = [build_fields(m, fs, f"u{i}") for i, m in enumerate(mols)]
    assert not np.allclose(moved, unalign(mf3, max_rotation=30.0, max_translation=1.0, seed=8))


def test_unalign_then_variance_align_recovers_the_pose():
    """The control experiment: scramble a molecule, realign it, measure the recovery."""
    from cmfqsar.align import unalign
    model, fs, mols = _model()
    al = VarianceAligner(model, max_rmsd=None)
    probe = build_fields(mols[4], fs, "probe")
    reference = probe.coords.copy()
    moved = unalign([probe], max_rotation=35.0, max_translation=1.2, seed=4)[0]
    al.align(probe, restarts=2, seed=0)
    recovered = float(np.sqrt(((probe.coords - reference) ** 2).sum(1).mean()))
    assert moved > 0.8                       # the scramble really displaced the molecule
    assert recovered < 0.2 * moved and recovered < 0.3


def test_cv_training_alignment_runs_and_reports_diagnostics():
    """The fold-wise training-set alignment must run, converge and report its diagnostics."""
    from cmfqsar.varalign import cv_align_training_set, ensemble_similarity
    model, fs, mols = _model()
    mf = [build_fields(m, fs, f"t{i}") for i, m in enumerate(mols)]
    start = [m.coords.copy() for m in mf]
    sim0 = ensemble_similarity(mf, fs.specs)
    rows = cv_align_training_set(mf, Y, model, folds=4, rounds=2, max_rmsd=1.0, restarts=0)
    assert len(rows) == 2
    assert rows[0].round == 1 and rows[1].round == 2
    # the molecules were moved, within the guard, and the drift is cumulative
    assert all(r.moved_max <= 1.05 for r in rows)
    assert rows[1].drift >= rows[0].drift - 1e-9
    # steps shrink: the iteration converges rather than running away
    assert rows[1].moved_mean <= rows[0].moved_mean + 1e-9
    # the diagnostics are finite and the similarity is a correlation-like number
    for r in rows:
        assert 0.0 <= r.similarity <= 1.0 + 1e-9
        assert "round" in r.describe()
    assert ensemble_similarity(mf, fs.specs) >= sim0 - 1e-6     # alignment does not reduce overlap
    assert any(not np.allclose(a.coords, b) for a, b in zip(mf, start))


def test_cv_training_alignment_respects_cancel():
    from cmfqsar.varalign import cv_align_training_set
    model, fs, mols = _model()
    mf = [build_fields(m, fs, f"t{i}") for i, m in enumerate(mols)]
    rows = cv_align_training_set(mf, Y, model, folds=4, rounds=3, max_rmsd=1.0, restarts=0,
                                 cancel=lambda: True)
    assert rows == []


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
