"""The descriptions of a protein site that plan item PP3 measures against a count of atoms.

* The anchored points lie along the direction of the side chain: the virtual beta carbon built
  from the ideal backbone points within a few degrees of the real beta carbons of a wild-type
  structure, in the frame of each residue.
* The anisotropy of a field is its amount, its centre and its principal second moments, which a
  hand-made example checks, and a rigid move of the surroundings moves the centre and leaves
  the amount and the moments alone.
* The accessible area is in square angstrom, the main chain is not counted in the side chain,
  and a glycine has no side-chain area.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, ROOT)

pytest.importorskip("rdkit")

import run_site_fields as R                                             # noqa: E402

PDB = os.path.join(ROOT, "examples", "mutants", "1UFM", "1UFM.pdb")


def test_the_anchored_points():
    pts = R.anchor_points()
    assert pts.shape == (7, 3)
    u = pts[0] / np.linalg.norm(pts[0])
    for k, t in enumerate(R.ANCHOR_ALONG):
        assert np.allclose(pts[k], u * t, atol=1e-9)
    mid = pts[1]
    for p in pts[3:]:
        off = p - mid
        assert np.linalg.norm(off) == pytest.approx(R.ANCHOR_SIDE)
        assert abs(off @ u) < 1e-9


def test_the_virtual_beta_carbon_points_along_the_real_ones():
    pytest.importorskip("pdbfixer")
    import tempfile
    from cmfqsar import mutants as MU
    u = R.anchor_points()[0]
    u = u / np.linalg.norm(u)
    with tempfile.TemporaryDirectory() as tmp:
        st = MU.prepare_wild_type(PDB, os.path.join(tmp, "w.pdb"))
        mol = MU.protein_of(st)
    xyz = mol.GetConformer().GetPositions()
    angles = []
    numbers = sorted({a.GetPDBResidueInfo().GetResidueNumber() for a in mol.GetAtoms()})
    for p in numbers:
        atoms = MU.residue_atoms(mol, p)
        if not all(k in atoms for k in ("N", "CA", "C", "CB")):
            continue
        origin, rot = MU.backbone_frame(xyz, atoms)
        cb = (xyz[atoms["CB"]] - origin) @ rot.T
        angles.append(np.degrees(np.arccos(np.clip(cb @ u / np.linalg.norm(cb), -1, 1))))
    assert len(angles) > 20
    assert np.median(angles) < 10.0 and max(angles) < 25.0


def _context(coords, weights):
    return SimpleNamespace(coords=np.asarray(coords, float), weights=np.asarray(weights, float))


def test_the_anisotropy_of_a_field():
    coords = [[1.0, 0, 0], [-1.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0]]
    w = np.zeros((4, 5))
    w[:, 1] = 1.0                      # the steric field: four equal atoms
    w[0, 0], w[1, 0] = 2.0, -2.0       # the electrostatic field: a pair of opposite signs
    ctx = {(float(R.RHO), "x", 1): _context(coords, w)}
    X, labels = R.anisotropy_rows(ctx, [("x", 1)])
    row = dict(zip(labels, X[0]))
    names = labels[0].split(":")[0], labels[7].split(":")[0]
    steric = names[1]
    assert row[f"{steric}:amount"] == pytest.approx(4.0)
    assert [row[f"{steric}:{k}"] for k in "xyz"] == pytest.approx([0, 0, 0], abs=1e-12)
    # moments about the centre: 2 (along y), 0.5 (along x), 0
    assert [row[f"{steric}:m{k}"] for k in (1, 2, 3)] == pytest.approx([2.0, 0.5, 0.0])
    # the electrostatic field counts by magnitude: amount 4, centre at the origin
    elec = names[0]
    assert row[f"{elec}:amount"] == pytest.approx(4.0)
    # a field with no weight is all zeros
    assert all(row[f"{labels[14].split(':')[0]}:{k}"] == 0.0
               for k in ("amount", "x", "y", "z", "m1", "m2", "m3"))


def test_a_rigid_move_moves_only_the_centre():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(30, 3)) * 3
    w = np.abs(rng.normal(size=(30, 5)))
    shift = np.array([1.0, -2.0, 0.5])
    ctx = {(float(R.RHO), "x", 1): _context(coords, w),
           (float(R.RHO), "x", 2): _context(coords + shift, w)}
    X, labels = R.anisotropy_rows(ctx, [("x", 1), ("x", 2)])
    for j, lab in enumerate(labels):
        kind = lab.split(":")[1]
        if kind in "xyz":
            assert X[1, j] - X[0, j] == pytest.approx(shift["xyz".index(kind)])
        else:
            assert X[1, j] == pytest.approx(X[0, j])


def test_the_accessible_area():
    pytest.importorskip("mdtraj")
    pytest.importorskip("pdbfixer")
    area = R.accessible_area(("1UFM",))
    totals = np.array([v[0] for v in area.values()])
    sides = np.array([v[1] for v in area.values()])
    assert (sides <= totals + 1e-9).all() and (totals >= 0).all()
    assert totals.max() < 400.0 and totals.mean() > 20.0
    from cmfqsar import mutants as MU
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        st = MU.prepare_wild_type(PDB, os.path.join(tmp, "w.pdb"))
        mol = MU.protein_of(st)
    gly = {a.GetPDBResidueInfo().GetResidueNumber() for a in mol.GetAtoms()
           if a.GetPDBResidueInfo().GetResidueName() == "GLY"}
    for p in gly:
        assert area[("1UFM", p)][1] == pytest.approx(0.0)
