"""Tests for the heterogeneous-series workflow: several chemotypes, no one common substructure."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.chemotypes import (ACYCLIC, apply_transform, align_within_groups,  # noqa: E402
                                alignment_free_q2, chemotype_labels, evaluate, group_members,
                                group_reference, grouped_cv_predict, grouped_folds, place_groups,
                                scaffold_smiles, summarise_groups, _kabsch)
from cmfqsar.fields import FieldSet  # noqa: E402

#: two ring systems and a pair of acyclic molecules: three chemotypes by construction
BENZENES = ["Cc1ccccc1", "CCc1ccccc1", "CCCc1ccccc1", "Clc1ccccc1", "COc1ccccc1"]
PYRIDINES = ["Cc1ccncc1", "CCc1ccncc1", "CCCc1ccncc1", "Clc1ccncc1"]
ACYCLICS = ["CCCCO", "CCCCCN"]


def _mol(smiles: str, seed: int = 1, name: str = "m"):
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


@pytest.fixture(scope="module")
def mixed():
    """Molecules, names and labels of a three-chemotype set."""
    smiles = BENZENES + PYRIDINES + ACYCLICS
    mols = [_mol(s, seed=i + 1, name=f"m{i}") for i, s in enumerate(smiles)]
    labels, names = chemotype_labels(mols)
    return mols, [f"m{i}" for i in range(len(mols))], labels, names


def _centroid(mol):
    return np.array(mol.GetConformer().GetPositions()).mean(axis=0)


def _random_motion(mol, rng):
    """Move a molecule rigidly, by a random rotation about its centre and a random shift."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    xyz = np.array(mol.GetConformer().GetPositions())
    mid = xyz.mean(axis=0)
    apply_transform(mol, R, mid + rng.normal(scale=4.0, size=3) - R @ mid)


# --------------------------------------------------------------------------- 1. clustering

def test_the_framework_strips_the_side_chains():
    assert scaffold_smiles(_mol("Cc1ccccc1")) == "c1ccccc1"
    assert scaffold_smiles(_mol("CCCCO")) == ACYCLIC
    # a molecule with no rings has no framework, whatever its size
    assert scaffold_smiles(_mol("CCCCCCCCO")) == ACYCLIC
    # the generic framework forgets the heteroatom, so benzene and pyridine become one chemotype
    assert (scaffold_smiles(_mol("Cc1ccncc1"), generic=True)
            == scaffold_smiles(_mol("Cc1ccccc1"), generic=True))


def test_the_groups_come_out_largest_first(mixed):
    mols, _, labels, names = mixed
    assert len(names) == 3 and len(labels) == len(mols)
    sizes = [len(m) for m in group_members(labels)]
    assert sizes == [5, 4, 2]                    # benzenes, pyridines, acyclics
    assert names[0] == "c1ccccc1" and names[2] == ACYCLIC
    assert set(labels[:5]) == {0} and set(labels[5:9]) == {1} and set(labels[9:]) == {2}
    assert summarise_groups(labels, names).startswith("3 chemotypes: 5x c1ccccc1")


def test_small_groups_are_pooled_so_that_the_folds_stay_meaningful(mixed):
    mols, _, _, _ = mixed
    labels, names = chemotype_labels(mols, min_size=5)
    assert len(names) == 2                       # the benzenes, and everything else together
    assert names[1] == "other (small groups)"
    assert [len(m) for m in group_members(labels)] == [5, 6]


# --------------------------------------------------------------------------- 4. honest folds

def test_folds_never_split_a_chemotype(mixed):
    _, _, labels, _ = mixed
    folds = grouped_folds(labels)
    assert len(folds) == 3
    for te in folds:
        assert len(set(labels[te])) == 1         # one chemotype per fold, whole
    assert sorted(np.concatenate(folds)) == list(range(len(labels)))


def test_bundling_keeps_whole_chemotypes_together(mixed):
    _, _, labels, _ = mixed
    folds = grouped_folds(labels, max_folds=2)
    assert len(folds) == 2
    assert sorted(np.concatenate(folds)) == list(range(len(labels)))
    for te in folds:                             # a bundle is a union of whole chemotypes
        for g in set(labels[te]):
            assert set(np.flatnonzero(labels == g)) <= set(te)


def test_holding_out_a_chemotype_exposes_what_random_folds_hide():
    """With an activity that is a property of the scaffold, random folds are pure leakage."""
    labels = np.repeat(np.arange(5), 8)
    y = labels.astype(float) * 2.0
    f = y + 1e-3 * np.arange(len(y))             # a feature that all but names the chemotype
    K = np.exp(-0.5 * (f[:, None] - f[None, :]) ** 2)
    grouped = grouped_cv_predict(K, y, labels, params={"lam": 1e-6})
    random = grouped_cv_predict(K, y, np.arange(len(y)) % 5, params={"lam": 1e-6})
    assert np.isfinite(grouped).all() and np.isfinite(random).all()
    from cmfqsar.model import r2_score
    assert r2_score(y, random) > 0.99            # the scaffold-mates give the answer away
    assert r2_score(y, grouped) < 0.0            # held out, the model knows nothing of it


def test_grouped_predictions_are_fitted_without_the_held_out_group():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(24, 3))
    K = X @ X.T + np.eye(24) * 0.1
    y = rng.normal(size=24)
    labels = np.repeat(np.arange(3), 8)
    got = grouped_cv_predict(K, y, labels, params={"lam": 0.5})
    from cmfqsar.model import _Learner
    for g in range(3):
        te = np.flatnonzero(labels == g)
        tr = np.flatnonzero(labels != g)
        want = _Learner("krr", {"lam": 0.5}, True).fit(
            K[np.ix_(tr, tr)], y[tr]).predict(K[np.ix_(te, tr)])
        assert np.allclose(got[te], want)


# --------------------------------------------------------------------------- 2 and 3. geometry

def test_the_reference_of_a_group_is_its_largest_molecule(mixed):
    mols, _, labels, _ = mixed
    idx = np.flatnonzero(labels == 0)
    ref = group_reference(mols, idx)
    assert mols[ref].GetNumHeavyAtoms() == max(mols[i].GetNumHeavyAtoms() for i in idx)


def test_kabsch_recovers_the_motion_that_was_applied():
    rng = np.random.default_rng(7)
    before = rng.normal(size=(12, 3))
    mol = _mol("Cc1ccccc1")
    xyz = np.array(mol.GetConformer().GetPositions())
    _random_motion(mol, rng)
    after = np.array(mol.GetConformer().GetPositions())
    R, t = _kabsch(xyz, after)
    assert np.allclose(xyz @ R.T + t, after, atol=1e-6)
    assert np.allclose(np.linalg.det(R), 1.0)    # a rotation, never a reflection
    # the same transform carries any other molecule the same way
    other = _mol("CCc1ccccc1")
    kept = np.array(other.GetConformer().GetPositions())
    apply_transform(other, R, t)
    assert np.allclose(np.array(other.GetConformer().GetPositions()), kept @ R.T + t,
                       atol=1e-6)


def test_aligning_inside_a_group_superimposes_the_shared_rings(mixed):
    from rdkit import Chem
    mols, _, labels, _ = mixed
    mols = [Chem.Mol(m) for m in mols]
    rng = np.random.default_rng(11)
    for m in mols:
        _random_motion(m, rng)
    idx = np.flatnonzero(labels == 0)
    ref = group_reference(mols, idx)
    ring = np.array([a.GetIdx() for a in mols[ref].GetAtoms() if a.GetIsAromatic()])

    def ring_rmsd(i):
        here = np.array([a.GetIdx() for a in mols[i].GetAtoms() if a.GetIsAromatic()])
        a = np.array(mols[i].GetConformer().GetPositions())[here]
        b = np.array(mols[ref].GetConformer().GetPositions())[ring]
        return float(np.sqrt(((np.sort(a, axis=0) - np.sort(b, axis=0)) ** 2).sum(axis=1).mean()))

    before = [ring_rmsd(i) for i in idx if i != ref]
    log = align_within_groups(mols, labels)
    after = [ring_rmsd(i) for i in idx if i != ref]
    assert len(log) == len(mols) and not [l for l in log if "FAILED" in l]
    assert max(after) < 0.5 and max(after) < max(before)


def test_placing_moves_a_whole_chemotype_without_disturbing_it(mixed):
    from rdkit import Chem
    mols, _, labels, _ = mixed
    mols = [Chem.Mol(m) for m in mols]
    rng = np.random.default_rng(13)
    for m in mols:
        _random_motion(m, rng)
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5)
    idx1 = np.flatnonzero(labels == 1)
    kept = np.array([np.array(mols[i].GetConformer().GetPositions()) for i in np.flatnonzero(labels == 0)],
                    dtype=object)
    inside = [np.linalg.norm(_centroid(mols[i]) - _centroid(mols[j]))
              for i in idx1 for j in idx1 if i < j]
    ref0 = group_reference(mols, np.flatnonzero(labels == 0))
    ref1 = group_reference(mols, idx1)
    far = np.linalg.norm(_centroid(mols[ref1]) - _centroid(mols[ref0]))
    log = place_groups(mols, labels, fs, mode="field", reference_group=0, restarts=1, seed=0)
    assert "left where it is" in log[0] and not [l for l in log if "FAILED" in l]
    # the reference chemotype never moves
    for k, i in enumerate(np.flatnonzero(labels == 0)):
        assert np.allclose(np.array(mols[i].GetConformer().GetPositions()), kept[k])
    # and the chemotype that does move keeps its own internal arrangement
    after = [np.linalg.norm(_centroid(mols[i]) - _centroid(mols[j]))
             for i in idx1 for j in idx1 if i < j]
    assert np.allclose(after, inside, atol=1e-6)
    # it has come into the same frame as the reference chemotype
    near = np.linalg.norm(_centroid(mols[ref1]) - _centroid(mols[ref0]))
    assert near < 2.0 and near < far


# --------------------------------------------------------------------------- 5. the baseline

def test_the_alignment_free_baseline_ignores_the_poses(mixed):
    """The smaller the attenuation, the less the poses can matter: that is the whole point."""
    from rdkit import Chem
    mols, names, labels, _ = mixed
    y = np.arange(len(mols), dtype=float) * 0.3
    moved = [Chem.Mol(m) for m in mols]
    rng = np.random.default_rng(17)
    for m in moved:
        _random_motion(m, rng)
    gaps = []
    for alpha in (2e-3, 2e-4, 2e-5):
        q2, rms = alignment_free_q2(mols, names, y, labels, alpha=alpha, cif_min=1)
        q2b, rmsb = alignment_free_q2(moved, names, y, labels, alpha=alpha, cif_min=1)
        assert np.isfinite(q2) and np.isfinite(q2b)
        gaps.append(abs(q2 - q2b))
    assert gaps[0] > gaps[1] > gaps[2]           # it converges on a pose-free kernel
    assert gaps[2] < 0.01
    assert q2 == pytest.approx(q2b, abs=0.01) and rms == pytest.approx(rmsb, abs=0.01)


# --------------------------------------------------------------------------- the whole workflow

def test_evaluate_reports_the_three_numbers_for_every_chemotype(mixed):
    mols, names, labels, group_names = mixed
    rng = np.random.default_rng(19)
    y = labels.astype(float) + rng.normal(scale=0.3, size=len(mols))
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    res = evaluate(mols, names, y, labels, group_names, fs, alphas=[0.5], lams=[0.1, 1.0])
    assert np.isfinite([res.q2_random, res.q2_grouped, res.q2_baseline, res.rmse_grouped]).all()
    assert res.alpha == 0.5 and res.lam in (0.1, 1.0)
    assert len(res.per_group) == 3
    assert sum(g["n"] for g in res.per_group) == len(mols)
    assert {g["name"] for g in res.per_group} == set(group_names)
    assert isinstance(res.verdict(), str) and res.verdict()


def test_a_congeneric_series_has_nothing_to_hold_out():
    mols = [_mol(s, seed=i + 1) for i, s in enumerate(BENZENES)]
    names = [f"m{i}" for i in range(len(mols))]
    labels, group_names = chemotype_labels(mols)
    assert len(group_names) == 1
    res = evaluate(mols, names, np.arange(len(mols), dtype=float), labels, group_names,
                   FieldSet.create(physchem=["steric"], alpha=1.0))
    assert not np.isfinite(res.q2_grouped)
    assert "nothing to hold out" in " ".join(res.log)
    assert "not enough chemotypes" in res.verdict()
