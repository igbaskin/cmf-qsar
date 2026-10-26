"""Tests for reading a pharmacophore out of a co-field."""
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cofields import Grid, make_grid  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.pharmacophore import (FEATURE_TYPES, Feature, extract,  # noqa: E402
                                   features_from_volume, pocket_exclusions, summary,
                                   write_csv, write_pdb)

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "CCOc1ccc(C)cc1CC(=O)N",
          "CC(C)COc1ccccc1C(=O)NC", "CCCCOc1ccccc1C(=O)N"]


def _mol(smiles, seed=1, name="m"):
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


def _bumps(centres, amplitudes, grid, width=1.0):
    """A volume with a Gaussian bump of the given amplitude at every centre."""
    pts = grid.points()
    v = np.zeros(len(pts))
    for c, a in zip(centres, amplitudes):
        v += a * np.exp(-((pts - np.asarray(c)) ** 2).sum(axis=1) / (2 * width ** 2))
    return v.reshape(grid.shape)


@pytest.fixture(scope="module")
def grid():
    return Grid(origin=np.array([-6.0, -6.0, -6.0]), spacing=0.5, shape=(25, 25, 25))


def test_a_lobe_becomes_a_feature_where_it_is(grid):
    centres = [(-2.0, 0.0, 0.0), (2.5, 1.0, -1.0)]
    vol = _bumps(centres, [1.0, 0.7], grid)
    # the grid is coarse here, so the smallest lobe worth keeping is smaller than the default
    feats = features_from_volume(vol, grid, "hbd", +1, level=0.5, min_radius=0.3)
    assert len(feats) == 2
    feats.sort(key=lambda f: -f.importance)
    assert feats[0].kind == "donor" and feats[0].favourable
    for f, c in zip(sorted(feats, key=lambda f: f.centre[0]), sorted(centres)):
        assert np.allclose(f.centre, c, atol=0.3)
        assert 0.5 < f.radius < 3.0
        assert f.importance > 0
    # the stronger bump scores higher and the weaker one is still found
    assert feats[0].importance > feats[1].importance > 0


def test_the_contour_level_and_the_minimum_size_do_what_they_say(grid):
    vol = _bumps([(0.0, 0.0, 0.0)], [1.0], grid)
    loose = features_from_volume(vol, grid, "hydrophobic", +1, level=0.3, min_radius=0.3)[0]
    tight = features_from_volume(vol, grid, "hydrophobic", +1, level=0.8, min_radius=0.3)[0]
    assert tight.radius < loose.radius            # a higher contour is a tighter sphere
    assert np.allclose(tight.centre, loose.centre, atol=0.2)
    assert features_from_volume(vol, grid, "hydrophobic", +1, level=0.8, min_radius=5.0) == []
    # and the default minimum quietly drops a lobe that a 0.5 A grid resolves with a handful of
    # cells, which is what keeps a real co-field from producing features from its own noise
    assert features_from_volume(vol, grid, "hydrophobic", +1, level=0.8) == []


def test_the_sign_decides_what_the_feature_asks_for(grid):
    vol = _bumps([(0.0, 0.0, 0.0)], [-1.0], grid)       # a negative lobe
    assert features_from_volume(vol, grid, "steric", +1, level=0.5, min_radius=0.3) == []
    excl = features_from_volume(vol, grid, "steric", -1, level=0.5, min_radius=0.3)
    assert len(excl) == 1 and excl[0].kind == "exclusion"
    assert excl[0].favourable is False and excl[0].importance < 0
    # a negative electrostatic lobe is a feature asking for negative charge, not a warning
    neg = features_from_volume(vol, grid, "electrostatic", -1, level=0.5, min_radius=0.3)[0]
    assert neg.kind == "negative" and neg.favourable and neg.importance > 0
    assert all(len(v) == 3 for v in FEATURE_TYPES.values())


def test_extraction_from_a_fitted_model_and_what_it_refuses(tmp_path):
    fs = FieldSet.create(alpha=0.5)
    mols = [_mol(s, seed=i + 1, name=f"m{i}") for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    y = np.array([5.0, 5.6, 6.1, 6.4, 7.0])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    feats = extract(model, level=0.6, max_features=4, spacing=0.8)
    assert feats and len(feats) <= 8                      # 4 favourable + at most 4 exclusions
    assert sum(1 for f in feats if f.favourable) <= 4
    assert all(f.radius > 0 and np.isfinite(f.importance) for f in feats)
    assert "features" in summary(feats)
    # strongest first among the favourable ones
    good = [f.importance for f in feats if f.favourable]
    assert good == sorted(good, key=lambda v: -abs(v))

    csv_path, pdb_path = str(tmp_path / "ph.csv"), str(tmp_path / "ph.pdb")
    write_csv(csv_path, feats)
    write_pdb(pdb_path, feats)
    rows = open(csv_path).read().strip().splitlines()
    assert len(rows) == len(feats) + 1 and rows[0].startswith("type,x,y,z,radius")
    lines = [l for l in open(pdb_path) if l.startswith("HETATM")]
    assert len(lines) == len(feats)
    # the B-factor column carries the tolerance radius
    assert float(lines[0][60:66]) == pytest.approx(feats[0].radius, abs=0.01)

    # a kernel with no frame has no co-field, so there is nothing to discretise
    invariant = CMFModel(fieldset=fs, method="krr", normalize=True,
                         kernel_source="autocorrelation", params={"lam": 0.1})
    invariant.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    with pytest.raises(ValueError, match="overlap-integral kernel"):
        extract(invariant)


def test_pocket_exclusions_thin_the_receptor_to_a_few_spheres(grid):
    rng = np.random.default_rng(0)
    inside = rng.uniform(-5.0, 5.0, size=(200, 3))
    outside = rng.uniform(20.0, 30.0, size=(50, 3))
    protein = types.SimpleNamespace(
        coords=np.vstack([inside, outside]),
        elements=["C"] * len(inside) + ["C"] * len(outside))
    feats = pocket_exclusions(protein, grid, max_spheres=10, min_separation=2.0)
    assert len(feats) == 10
    assert all(f.kind == "pocket" and not f.favourable for f in feats)
    lo, hi = grid.extent
    for f in feats:
        assert np.all(f.centre >= lo) and np.all(f.centre <= hi)   # nothing from outside the box
    for i, a in enumerate(feats):
        for b in feats[i + 1:]:
            assert np.linalg.norm(a.centre - b.centre) >= 2.0 - 1e-9
    # hydrogens produce no exclusion spheres
    protein.elements = ["H"] * len(protein.coords)
    assert pocket_exclusions(protein, grid) == []
