"""Tests for placing conformations inside the pocket."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.docking import (PocketScorer, dock_coords, dock_conformers,  # noqa: E402
                             dock_dataset, summarise)
from cmfqsar.fields import FieldSet, FieldSpec, MoleculeFields  # noqa: E402

#: an electrostatic and a steric field, which is all these tests need
FS = FieldSet(specs=[FieldSpec(name="electrostatic", kind="gauss", alpha=0.5),
                     FieldSpec(name="steric", kind="steric", alpha=0.5)])


def _fields(coords, charges, radius=1.7, name="probe", steric=None):
    """A molecule with an electrostatic and a steric field, built by hand."""
    n = len(coords)
    w = np.zeros((n, 2))
    w[:, 0] = charges
    w[:, 1] = 1.0 if steric is None else steric
    return MoleculeFields(name=name, coords=np.array(coords, float), weights=w,
                          radii=np.full(n, radius), symbols=["C"] * n, atom_types=["C.3"] * n,
                          bonds=np.zeros((0, 2), int), charges=np.array(charges, float))


def test_the_score_prefers_complementary_charges_and_hates_clashes():
    pocket = _fields([[6.0, 0, 0]], [-1.0], name="pocket")
    cation = _fields([[0.0, 0, 0]], [+1.0])
    anion = _fields([[0.0, 0, 0]], [-1.0])
    s_good = PocketScorer(cation, pocket, FS, tether_weight=0.0)
    s_bad = PocketScorer(anion, pocket, FS, tether_weight=0.0)
    here = cation.coords
    assert s_good.score(here) > s_bad.score(here)
    # moving on top of the pocket atom is a clash and must be punished
    on_top = np.array([[6.0, 0.0, 0.0]])
    assert s_good.clash(on_top) > 0 and s_good.clash(here) == 0
    assert s_good.score(on_top) < s_good.score(here)


def test_docking_moves_a_clashing_pose_out_of_the_wall():
    pocket = _fields([[0.0, 0, 0], [3.0, 0, 0], [6.0, 0, 0]], [-0.3, -0.3, -0.3], name="pocket")
    probe = _fields([[3.0, 0.4, 0.0]], [+1.0])            # sitting inside a pocket atom
    scorer = PocketScorer(probe, pocket, FS)
    assert scorer.clash(probe.coords) > 0
    coords, res = dock_coords(probe, pocket, FS, restarts=2, seed=0)
    assert res.clash_after < res.clash_before
    assert res.score_after > res.score_before
    assert res.moved > 0.1
    assert coords.shape == probe.coords.shape


def test_the_tether_keeps_a_pose_from_wandering_off():
    pocket = _fields([[0.0, 0, 0]], [-1.0], name="pocket")
    probe = _fields([[4.0, 0, 0]], [-1.0])                # repelled: it would rather leave
    far = np.array([[40.0, 0.0, 0.0]])
    tight = PocketScorer(probe, pocket, FS, tether_radius=2.0, tether_weight=5.0)
    loose = PocketScorer(probe, pocket, FS, tether_weight=0.0)
    assert tight.score(far) < loose.score(far)            # the tether is what makes leaving costly
    coords, _ = dock_coords(probe, pocket, FS, restarts=1, seed=0)
    drift = np.linalg.norm(coords.mean(axis=0) - probe.coords.mean(axis=0))
    assert drift < 12.0                                   # it stays in the neighbourhood


def test_burial_pulls_a_pose_into_the_site():
    """Without the burial term the field terms alone do not hold a ligand in a cavity."""
    ring = _fields([[np.cos(a) * 5.0, np.sin(a) * 5.0, 0.0] for a in np.linspace(0, 2 * np.pi, 9)],
                   [0.0] * 9, name="pocket")
    probe = _fields([[0.0, 0.0, 0.0]], [0.0])             # in the middle of the cavity
    inside = probe.coords
    outside = probe.coords + np.array([12.0, 0.0, 0.0])
    with_burial = PocketScorer(probe, ring, FS, burial_weight=2.0, tether_weight=0.0)
    without = PocketScorer(probe, ring, FS, burial_weight=0.0, tether_weight=0.0)
    assert with_burial.score(inside) > with_burial.score(outside)
    assert without.score(inside) == pytest.approx(without.score(outside), abs=1e-9)


def test_docking_a_molecule_keeps_its_conformations_and_internal_geometry():
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cmfqsar.conformers import generate_conformers
    m = Chem.AddHs(Chem.MolFromSmiles("CCCOc1ccccc1"))
    p = AllChem.ETKDGv3()
    p.randomSeed = 1
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    m.SetProp("_Name", "probe")
    m = generate_conformers(m, n=3, seed=2, prune_rmsd=1.0)
    centre = np.array(m.GetConformer(0).GetPositions()).mean(axis=0)
    pocket = _fields([centre + np.array([7.0, 0, 0]), centre + np.array([-7.0, 0, 0])],
                     [-0.5, -0.5], name="pocket")
    docked, results = dock_conformers(m, pocket, FS, restarts=1, seed=0)
    assert docked.GetNumConformers() == m.GetNumConformers()
    assert len(results) == m.GetNumConformers()
    assert docked.GetProp("_Name") == "probe"
    # rigid body only: every internal distance is preserved
    for c in range(m.GetNumConformers()):
        a = np.array(m.GetConformer(c).GetPositions())
        b = np.array(docked.GetConformer(c).GetPositions())
        da = np.linalg.norm(a[:, None] - a[None, :], axis=-1)
        db = np.linalg.norm(b[:, None] - b[None, :], axis=-1)
        assert np.abs(da - db).max() < 1e-6
    # keep_best trims to the best-scoring poses
    few, res_few = dock_conformers(m, pocket, FS, restarts=0, seed=0, keep_best=2)
    assert few.GetNumConformers() == 2 and len(res_few) == 2
    info = summarise(results)
    assert info["poses"] == m.GetNumConformers()
    assert set(info) >= {"score_before", "score_after", "clash_before", "clash_after", "moved"}
    assert summarise([]) == {}


def test_dock_dataset_survives_a_molecule_it_cannot_place():
    from rdkit import Chem
    broken = Chem.Mol()                                   # no atoms, no conformations
    from rdkit.Chem import AllChem
    good = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    p = AllChem.ETKDGv3()
    p.randomSeed = 1
    AllChem.EmbedMolecule(good, p)
    pocket = _fields([[8.0, 0, 0]], [-0.5], name="pocket")
    out, res = dock_dataset([good, broken], pocket, FS, restarts=0, seed=0)
    assert len(out) == 2
    assert out[1].GetNumAtoms() == 0                      # handed back untouched
    assert len(res) >= 1


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
