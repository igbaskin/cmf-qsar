"""Alignment onto a plane: the rules, the refinement in the surface frame, and how a pose is judged."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from cmfqsar.fields import FieldSet
from cmfqsar.planealign import (anchor_atom, height_rmsd, in_plane_rmsd, orient_flat, plane_align,
                                plane_align_dataset, rigid_rmsd, rule_pose, surface_field_align)


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


def _xyz(m):
    return np.array(m.GetConformer().GetPositions(), dtype=float)


def _heavy(m):
    return np.array([a.GetAtomicNum() > 1 for a in m.GetAtoms()])


def _distances(xyz):
    return np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)


def test_a_flat_aromatic_lies_in_a_plane_at_the_height_with_its_long_axis_along_x():
    m = _mol("c1ccc2ccccc2c1")                       # naphthalene
    d0 = _distances(_xyz(m))
    pose = plane_align(m, pose="flat", height=2.2)
    xyz, heavy = _xyz(m), _heavy(m)
    assert pose.label == "flat" and pose.anchor is None
    assert np.allclose(xyz[heavy, 2], 2.2, atol=0.05)             # the ring is the plane
    assert xyz[heavy, 2].min() == pytest.approx(2.2, abs=1e-6)    # the lowest heavy atom at the height
    span = xyz[heavy].max(axis=0) - xyz[heavy].min(axis=0)
    assert span[0] > span[1] > span[2]                             # long axis along x
    assert np.allclose(xyz[heavy, :2].mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(_distances(xyz), d0, atol=1e-8)             # rigid


def test_the_flattest_face_is_turned_down_so_substituents_point_away_from_the_surface():
    m = _mol("CC(C)(C)c1ccccc1")                     # tert-butylbenzene: the methyls stick out of the ring plane
    plane_align(m, pose="flat")
    xyz = _xyz(m)
    ring = [a.GetIdx() for a in m.GetAtoms() if a.GetIsAromatic()]
    methyl = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "C" and a.GetTotalNumHs(includeNeighbors=True) == 3]
    assert xyz[methyl, 2].mean() > xyz[ring, 2].mean()            # the bulk is above the ring
    other = orient_flat(_xyz(m), _heavy(m), face="up")
    assert other[methyl, 2].mean() < other[ring, 2].mean()        # the other face is the mirror choice


def test_the_upright_pose_puts_the_most_basic_nitrogen_lowest_with_its_lone_pair_down():
    m = _mol("c1ccncc1")                             # pyridine
    n = anchor_atom(m)
    assert m.GetAtomWithIdx(n).GetSymbol() == "N"
    pose = plane_align(m, pose="upright", anchor_height=2.0)
    xyz, heavy = _xyz(m), _heavy(m)
    assert pose.label == "upright" and pose.anchor == n
    assert xyz[n, 2] == pytest.approx(2.0, abs=1e-6)
    assert xyz[n, 2] <= xyz[:, 2].min() + 1e-6                     # nothing below the anchor, hydrogens included
    nbrs = [a.GetIdx() for a in m.GetAtomWithIdx(n).GetNeighbors()]
    assert xyz[nbrs, 2].mean() > xyz[n, 2]                         # the lone pair points at the surface
    assert np.allclose(xyz[n, :2], 0.0, atol=1e-6)                 # the anchor over the origin


def test_the_anchor_rule_prefers_an_amine_to_a_pyridine_and_never_takes_an_amide():
    amine_and_pyridine = Chem.MolFromSmiles("NCc1ccncc1")
    a = anchor_atom(amine_and_pyridine)
    assert amine_and_pyridine.GetAtomWithIdx(a).GetTotalNumHs() == 2       # the aliphatic amine
    amide = Chem.MolFromSmiles("CC(=O)Nc1ccccc1")
    assert anchor_atom(amide) is None
    nitro = Chem.MolFromSmiles("[O-][N+](=O)c1ccccc1")
    assert anchor_atom(nitro) is None
    assert anchor_atom(Chem.MolFromSmiles("c1ccccc1")) is None
    with pytest.raises(ValueError):
        rule_pose(_mol("c1ccccc1"), "upright")


def test_the_in_plane_rmsd_ignores_what_the_surface_does_not_fix_and_sees_what_it_does():
    m = _mol("c1ccc2[nH]cnc2c1")
    plane_align(m, pose="flat")
    a = _xyz(m)
    t = 0.7
    r = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    turned = a @ r.T + np.array([3.0, -1.5, 0.0])
    assert in_plane_rmsd(turned, a) < 1e-6
    assert height_rmsd(turned, a) < 1e-9
    lifted = a + np.array([0.0, 0.0, 1.0])
    assert in_plane_rmsd(lifted, a) == pytest.approx(1.0, abs=1e-6)
    assert height_rmsd(lifted, a) == pytest.approx(1.0, abs=1e-6)
    tilt = np.array([[1, 0, 0], [0, np.cos(0.5), -np.sin(0.5)], [0, np.sin(0.5), np.cos(0.5)]])
    tilted = (a - a.mean(0)) @ tilt.T + a.mean(0)
    assert in_plane_rmsd(tilted, a) > 0.3
    assert rigid_rmsd(tilted, a) < 1e-6                            # a rigid motion is not a pose error in 3D


def test_refinement_in_the_plane_recovers_a_turned_copy_and_never_changes_the_heights():
    fs = FieldSet.create()
    ref = _mol("c1ccc2[nH]cnc2c1")
    plane_align(ref, pose="flat")
    probe = Chem.Mol(ref)
    a = _xyz(probe)
    t = 1.1
    r = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    turned = a @ r.T + np.array([2.0, 1.0, 0.0])
    conf = probe.GetConformer()
    for i, p in enumerate(turned):
        conf.SetAtomPosition(i, [float(v) for v in p])
    before, after = surface_field_align(probe, ref, fs, dof="plane", restarts=6)
    assert after >= before - 1e-9
    assert after > 0.98                                             # the same molecule, found again
    assert np.allclose(_xyz(probe)[:, 2], a[:, 2], atol=1e-9)       # z untouched
    assert in_plane_rmsd(_xyz(probe), a) < 0.3


def test_refinement_with_tilt_moves_the_heights_only_within_the_box():
    fs = FieldSet.create()
    ref = _mol("c1ccc2[nH]cnc2c1")
    plane_align(ref, pose="flat")
    probe = Chem.Mol(ref)
    z0 = _xyz(probe)[:, 2].copy()
    conf = probe.GetConformer()
    for i, p in enumerate(_xyz(probe) + np.array([0.0, 0.0, 0.6])):
        conf.SetAtomPosition(i, [float(v) for v in p])
    before, after = surface_field_align(probe, ref, fs, dof="plane+tilt", restarts=2, max_dz=1.0)
    assert after >= before - 1e-9
    z1 = _xyz(probe)[:, 2]
    assert np.abs(z1 - z0).mean() < 0.6                              # it came back down
    assert np.abs(z1 - z0 - 0.6).max() <= 1.0 + 0.4 + 1e-6           # inside the lift box plus a tilt


def test_auto_picks_the_candidate_most_similar_in_field_to_the_reference():
    fs = FieldSet.create()
    ref = _mol("c1ccc2[nH]cnc2c1")
    plane_align(ref, pose="flat")
    flat_copy = Chem.Mol(ref)
    pose = plane_align(flat_copy, pose="auto", ref=ref, fieldset=fs)
    assert pose.label in ("flat", "flat-up")                        # the reference lies flat, so should the copy
    labels = [l for l, _ in pose.candidates]
    assert "upright" in labels and len(pose.candidates) == 3
    sims = dict(pose.candidates)
    assert sims[pose.label] >= sims["upright"]
    up_ref = _mol("c1ccc2[nH]cnc2c1")
    plane_align(up_ref, pose="upright")
    pose2 = plane_align(Chem.Mol(up_ref), pose="auto", ref=up_ref, fieldset=fs)
    assert pose2.label == "upright"


def test_the_dataset_aligner_keeps_the_reference_pose_and_poses_the_rest():
    fs = FieldSet.create()
    mols = [_mol(s) for s in ("c1ccc2[nH]cnc2c1", "Cc1nc2ccccc2[nH]1", "c1ccc2nc(-c3ccccc3)[nH]c2c1")]
    plane_align(mols[0], pose="flat", height=2.4)
    ref_before = _xyz(mols[0]).copy()
    res = plane_align_dataset(mols, ["a", "b", "c"], pose="auto", ref_index=0, fieldset=fs, refine="plane")
    assert res[0].label == "reference"
    assert np.allclose(_xyz(mols[0]), ref_before)
    for m, r in zip(mols[1:], res[1:]):
        assert r.label in ("flat", "flat-up", "upright")
        assert np.isfinite(r.similarity)
        assert _xyz(m)[_heavy(m), 2].min() > 1.5                    # on the surface, not through it
