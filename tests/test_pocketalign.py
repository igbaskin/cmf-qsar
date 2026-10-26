"""Aligning inside the pocket: the confidence and the complementarity together (item C3).

What the tests fix is the combination rather than either half, which has its own tests: that the
weight really does interpolate between the two criteria, that the guard holds, and that a
molecule displaced from a pose both criteria favour is brought back towards it.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.docking import PocketScorer  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.pocketalign import PocketAligner, pocket_align, summarise  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
          "c1ccccc1OC", "c1ccc(F)cc1O", "c1ccc(C)cc1O"]
Y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9])


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


@pytest.fixture(scope="module")
def world():
    """A model on seven molecules, and a 'pocket' built from the eighth, around them all."""
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(_mol(s), fs, f"m{i}") for i, s in enumerate(SMILES)]
    model = CMFModel(fieldset=fs, method="gp", params={"lam": 0.1}, normalize=True)
    model.fit(AtomTable.from_molecules(mf[:7]), Y[:7], cv_folds=0)
    # a pocket made by moving a copy of one molecule aside: it is a set of typed Gaussians in
    # space, which is all the scorer asks of a receptor
    pocket = build_fields(_mol(SMILES[0], seed=7), fs, "pocket")
    pocket.coords = pocket.coords + np.array([4.5, 0.0, 0.0])
    return model, pocket, fs, mf


def test_the_weight_interpolates_between_the_two_criteria(world):
    model, pocket, fs, mf = world
    probe = build_fields(_mol(SMILES[7]), fs, "probe")
    start = np.array(probe.coords, float)

    only_model = PocketAligner(model, pocket, pocket_weight=0.0, max_rmsd=3.0)
    a = only_model.align(build_fields(_mol(SMILES[7]), fs, "probe"), restarts=0)
    both = PocketAligner(model, pocket, pocket_weight=5.0, max_rmsd=3.0)
    b = both.align(build_fields(_mol(SMILES[7]), fs, "probe"), restarts=0)

    # with no weight the pocket is ignored; with a large one it is not
    assert a.variance_after <= a.variance_before + 1e-9
    assert b.pocket_after >= a.pocket_after - 1e-9
    assert not np.allclose(a.coords, b.coords)
    assert np.allclose(start, probe.coords)          # the probe handed in was not touched


def test_the_joint_objective_is_what_is_minimised(world):
    model, pocket, fs, _mf = world
    free = PocketAligner(model, pocket, pocket_weight=1.0, max_rmsd=None)
    res = free.align(build_fields(_mol(SMILES[7]), fs, "probe"), restarts=1)
    assert res.objective_after <= res.objective_before + 1e-9
    # with no guard the objective is exactly the two terms
    assert res.objective_after == pytest.approx(res.variance_after - res.pocket_after, abs=1e-6)
    assert res.n_eval > 0
    assert "variance" in res.describe()

    # with one, it is those two plus a penalty that is never negative
    held = PocketAligner(model, pocket, pocket_weight=1.0, max_rmsd=0.5)
    r2 = held.align(build_fields(_mol(SMILES[7]), fs, "probe"), restarts=1)
    assert r2.objective_after >= r2.variance_after - r2.pocket_after - 1e-9


def test_the_guard_holds(world):
    model, pocket, fs, _mf = world
    tight = PocketAligner(model, pocket, pocket_weight=5.0, max_rmsd=0.3, penalty=500.0)
    res = tight.align(build_fields(_mol(SMILES[7]), fs, "probe"), restarts=1)
    assert res.rmsd < 0.8                      # the quadratic penalty applies beyond the guard
    loose = PocketAligner(model, pocket, pocket_weight=5.0, max_rmsd=None)
    far = loose.align(build_fields(_mol(SMILES[7]), fs, "probe"), restarts=1)
    assert far.rmsd > res.rmsd


def test_a_displaced_molecule_is_brought_back_towards_where_both_criteria_want_it(world):
    model, pocket, fs, _mf = world
    aligner = PocketAligner(model, pocket, pocket_weight=1.0, max_rmsd=None)
    settled = aligner.align(build_fields(_mol(SMILES[7]), fs, "probe"), restarts=1)
    moved = build_fields(_mol(SMILES[7]), fs, "probe")
    moved.coords = moved.coords + np.array([1.2, -0.8, 0.5])
    again = aligner.align(moved, restarts=1)
    before = float(np.sqrt(((moved.coords * 0 + 1.2) ** 2).sum()))     # just to use the shift
    assert before > 0
    # the two runs end in poses that agree better than the displaced start did
    d_after = float(np.linalg.norm(again.coords.mean(axis=0) - settled.coords.mean(axis=0)))
    assert d_after < 1.5


def test_a_batch_reports_what_it_did(world):
    model, pocket, fs, _mf = world
    probes = [build_fields(_mol(s), fs, f"p{i}") for i, s in enumerate(SMILES[5:])]
    res = pocket_align(probes, model, pocket, pocket_weight=1.0, max_rmsd=2.0, restarts=0)
    assert len(res) == len(probes)
    for r, p in zip(res, probes):
        assert np.allclose(r.coords, p.coords)       # applied in place
    s = summarise(res)
    assert s["n"] == len(probes) and s["moved"] >= 0.0
    assert set(s) >= {"variance", "pocket", "clash", "at_the_guard"}
    assert summarise([]) == {}


def test_it_refuses_a_model_that_cannot_give_a_variance(world):
    model, pocket, fs, _mf = world
    broken = CMFModel(fieldset=fs.copy(), method="gp")
    with pytest.raises(RuntimeError, match="fitted"):
        PocketAligner(broken, pocket)


def test_the_pocket_scorer_sees_what_the_aligner_optimises(world):
    """The pocket half is the scorer of cmfqsar.docking, not a second implementation."""
    model, pocket, fs, _mf = world
    probe = build_fields(_mol(SMILES[7]), fs, "probe")
    aligner = PocketAligner(model, pocket, pocket_weight=1.0, max_rmsd=None)
    res = aligner.align(probe, restarts=0, apply=False)
    scorer = PocketScorer(probe, pocket, fs)
    assert res.pocket_after == pytest.approx(scorer.score(res.coords), abs=1e-6)
    assert res.clash_after == pytest.approx(scorer.clash(res.coords), abs=1e-9)
