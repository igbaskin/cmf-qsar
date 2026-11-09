"""Relaxing a pose under the gradient of the prediction (plan item MS3).

What is checked here is that the rigid-body search does what it says: it keeps the molecule
rigid, it reaches the same optimum with the analytic gradient as the same objective reaches
without one, it respects the displacement guard, and it refuses the kernel that has no pose.
Whether the pose it finds is the pose the confidence aligner finds is a measurement and not a
property, and it is made in ``benchmarks/run_gradient.py --stage relax``.
"""
import dataclasses

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation

from cmfqsar.fields import AtomTable, FieldSet, build_fields
from cmfqsar.model import CMFModel
from cmfqsar.relax import pose_rmsd, relax_pose, relax_poses

SMILES = ["CCCCO", "CCCCN", "CCCOC", "CCCCC", "CCOCC", "CCCCF", "CCCCCl", "CCC(C)O"]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)              # MMFF set-up alters the aromaticity flags in place
    m.SetProp("_Name", smiles)
    return m


def _fitted(**model_args):
    """A small fitted model on the overlap kernel, and a molecule that is not in it."""
    mols = [_mol(s, seed=i + 1) for i, s in enumerate(SMILES)]
    fs = FieldSet.create(physchem=["steric", "electrostatic"], alpha=0.5)
    table = AtomTable.from_molecules([build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)])
    y = np.array([4.0 + 0.2 * m.GetNumHeavyAtoms() + 0.05 * i for i, m in enumerate(mols)])
    args = dict(method="krr", normalize=True, params={"lam": 0.1})
    args.update(model_args)
    model = CMFModel(fieldset=fs, **args)
    model.fit(table, y, cv_folds=0)
    return model, fs, build_fields(_mol("CCCCS", seed=11), fs, "probe")


def _turned(mol, angle=0.35, shift=0.5, seed=0):
    rng = np.random.RandomState(seed)
    rv = Rotation.random(random_state=rng).as_rotvec()
    R = Rotation.from_rotvec(rv / np.linalg.norm(rv) * angle).as_matrix()
    X = np.asarray(mol.coords, float)
    c = X.mean(axis=0)
    return dataclasses.replace(mol, coords=(X - c) @ R.T + c + rng.normal(scale=shift, size=3))


def _bond_lengths(mol, coords):
    i, j = np.asarray(mol.bonds, int).T
    return np.linalg.norm(np.asarray(coords)[i] - np.asarray(coords)[j], axis=1)


def test_the_relaxation_raises_the_prediction_and_keeps_the_molecule_rigid():
    model, _fs, probe = _fitted()
    start = _turned(probe)
    out = relax_pose(model, start, max_rmsd=2.0)
    assert out.value_after > out.value_before
    assert out.gain == pytest.approx(out.value_after - out.value_before)
    # every bond length is what it was: the molecule moved, it did not change
    before = _bond_lengths(start, start.coords)
    after = _bond_lengths(start, out.coords)
    assert np.abs(after - before).max() < 1e-9
    # and the molecule itself was not written to
    assert np.abs(np.asarray(start.coords) - _turned(probe).coords).max() < 1e-12


def test_the_gradient_reaches_what_the_same_objective_reaches_without_it():
    """The analytic gradient is a way to the optimum, not a different optimum."""
    model, _fs, probe = _fitted()
    start = _turned(probe)
    driven = relax_pose(model, start, max_rmsd=None)
    plain = relax_pose(model, start, max_rmsd=None, gradient=False)
    assert driven.value_after == pytest.approx(plain.value_after, abs=5e-4)
    assert driven.n_eval < plain.n_eval            # and it needs far fewer evaluations


def test_the_displacement_guard_holds_the_molecule_back():
    model, _fs, probe = _fitted()
    start = _turned(probe)
    near = relax_pose(model, start, max_rmsd=0.5)
    far = relax_pose(model, start, max_rmsd=2.0)
    assert near.rmsd < 0.55 and near.limited
    assert far.rmsd > near.rmsd
    assert far.value_after > near.value_after      # more room reaches a higher prediction


def test_minimising_lowers_the_prediction():
    model, _fs, probe = _fitted()
    out = relax_pose(model, _turned(probe), max_rmsd=1.0, maximise=False)
    assert out.value_after < out.value_before


def test_applying_the_pose_writes_it_back():
    model, _fs, probe = _fitted()
    start = _turned(probe)
    X0 = np.array(start.coords)
    out = relax_pose(model, start, max_rmsd=1.0, apply=True)
    assert np.abs(np.asarray(start.coords) - out.coords).max() < 1e-12
    assert pose_rmsd(X0, start.coords) > 0.1


def test_several_molecules_at_once():
    model, _fs, probe = _fitted()
    outs = relax_poses(model, [_turned(probe, seed=0), _turned(probe, seed=1)], max_rmsd=1.0)
    assert len(outs) == 2
    assert all(o.value_after > o.value_before for o in outs)


def test_an_invariant_kernel_has_no_pose_to_relax():
    model, _fs, probe = _fitted(kernel_source="autocorrelation", ac_d_max=6.0)
    with pytest.raises(ValueError, match="invariant to translation and rotation"):
        relax_pose(model, probe)


def test_an_unfitted_model_is_refused():
    model, _fs, probe = _fitted()
    fresh = model.fresh_copy()
    with pytest.raises(RuntimeError, match="fitted"):
        relax_pose(fresh, probe)


def test_two_poses_of_different_molecules_cannot_be_compared():
    with pytest.raises(ValueError, match="shapes"):
        pose_rmsd(np.zeros((3, 3)), np.zeros((4, 3)))
