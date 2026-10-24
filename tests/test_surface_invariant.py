"""The surface autocorrelation kernel: invariant in the plane, aware of the height and the tilt."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (SurfaceSpec, autocorrelation, field_pairs,  # noqa: E402
                               surface_autocorrelation, surface_block_sizes,
                               surface_channel_labels, surface_feature_matrix)

SMILES = ["c1ccc2[nH]cnc2c1", "Cc1nc2ccccc2[nH]1", "Nc1nc2ccccc2[nH]1", "c1ccc(-c2nc3ccccc3[nH]2)cc1",
          "O=[N+]([O-])c1ccc2nc[nH]c2c1", "Sc1nc2ccccc2[nH]1"]


def _flat_on_surface(smiles, seed, height=2.0):
    """A molecule embedded, laid flat (plane of the two largest principal axes in xy) and put
    `height` A above z = 0: the pose convention of the adsorption script."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    xyz = np.array(m.GetConformer().GetPositions())
    c = xyz - xyz.mean(axis=0)
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    if np.linalg.det(vt) < 0:
        vt[2] *= -1
    out = c @ vt.T
    out[:, 2] += height - out[:, 2].min()
    conf = m.GetConformer()
    for i, p in enumerate(out):
        conf.SetAtomPosition(i, [float(v) for v in p])
    return m


def _moved(mol, rng, dz=0.0, tilt=0.0):
    """A copy turned about z, shifted in the plane, and optionally lifted or tilted about x."""
    from rdkit import Chem
    out = Chem.Mol(mol)
    xyz = np.array(out.GetConformer().GetPositions())
    t = rng.uniform(0, 2 * np.pi)
    R = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1.0]])
    c = xyz.mean(axis=0)
    new = (xyz - c) @ R.T + c + np.array([rng.uniform(-5, 5), rng.uniform(-5, 5), dz])
    if tilt:
        Rx = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])
        new = (new - new.mean(axis=0)) @ Rx.T + new.mean(axis=0)
    conf = out.GetConformer()
    for i, p in enumerate(new):
        conf.SetAtomPosition(i, [float(v) for v in p])
    return out


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create()
    mols = [_flat_on_surface(s, 7 + i) for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    return fs, mols, mf


def test_the_grid_and_the_labels_agree(world):
    fs, _, mf = world
    spec = SurfaceSpec(d_max=6.0, step=1.5, z_min=1.0, z_max=4.0, z_step=1.5)
    assert len(spec.centres) == 5 and len(spec.heights) == 3 and spec.n_bins == 45
    row = surface_autocorrelation(mf[0], spec)
    assert row.shape == (len(field_pairs(5)) * 45,)
    assert sum(surface_block_sizes(spec, 5)) == len(row)
    labels = surface_channel_labels(fs, spec)
    assert len(labels) == len(row) and labels[0].lower().startswith("electrostatic @ rho 0.00 a, z 1.0 / 1.0")


def test_a_motion_in_the_plane_changes_nothing(world):
    fs, mols, mf = world
    rng = np.random.default_rng(3)
    spec = SurfaceSpec()
    for i, m in enumerate(mols[:3]):
        here = surface_autocorrelation(mf[i], spec)
        there = surface_autocorrelation(build_fields(_moved(m, rng), fs, "moved"), spec)
        assert np.allclose(here, there, rtol=1e-9, atol=1e-9)


def test_but_a_change_of_height_or_tilt_does(world):
    fs, mols, mf = world
    rng = np.random.default_rng(4)
    spec = SurfaceSpec()
    here = surface_autocorrelation(mf[0], spec)
    lifted = surface_autocorrelation(build_fields(_moved(mols[0], rng, dz=1.5), fs, "up"), spec)
    tilted = surface_autocorrelation(build_fields(_moved(mols[0], rng, tilt=0.6), fs, "tilt"), spec)
    rel = lambda a, b: np.linalg.norm(a - b) / np.linalg.norm(a)
    assert rel(here, lifted) > 0.1 and rel(here, tilted) > 0.05
    # the three-dimensional autocorrelation sees neither: it is the one that throws the height away
    assert np.allclose(autocorrelation(mf[0]), autocorrelation(build_fields(_moved(mols[0], rng, dz=1.5), fs, "up")))


def test_the_height_sum_recovers_the_in_plane_distribution(world):
    """Summed over both height axes with wide height channels, the channels reduce to the plain
    in-plane distance distribution of the field weights."""
    _, _, mf = world
    spec = SurfaceSpec(d_max=6.0, step=1.0, sigma=1.0, z_min=0.0, z_max=8.0, z_step=1.0, z_sigma=1.0)
    row = surface_autocorrelation(mf[1], spec, pairs=[(1, 1)]).reshape(len(spec.centres), -1).sum(axis=1)
    W = mf[1].weights[:, 1]
    xy = mf[1].coords[:, :2]
    rho = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
    z = mf[1].coords[:, 2]
    # exact: the height Gaussians sum to the same constant for every atom when the grid is wide
    zsum = np.exp(-((z[:, None] - spec.heights[None, :]) ** 2) / 2.0).sum(axis=1)
    want = np.array([np.einsum("a,b,ab->", W * zsum, W * zsum, np.exp(-(rho - d) ** 2 / 2.0)) for d in spec.centres])
    assert np.allclose(row, want, rtol=1e-6)


def test_a_model_on_this_kernel_needs_no_in_plane_alignment(world):
    from cmfqsar.model import CMFModel
    fs, mols, mf = world
    y = np.array([5.0, 5.6, 6.1, 6.4, 7.0, 7.3])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="surface", params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf[:5]), y[:5], cv_folds=0)
    here = model.predict(AtomTable.from_molecules([mf[5]]))
    rng = np.random.default_rng(11)
    moved = build_fields(_moved(mols[5], rng), fs, "moved")
    assert np.allclose(model.predict(AtomTable.from_molecules([moved])), here, atol=1e-9)
    lifted = build_fields(_moved(mols[5], rng, dz=2.0), fs, "lifted")
    assert not np.allclose(model.predict(AtomTable.from_molecules([lifted])), here, atol=1e-3)
    with pytest.raises(ValueError, match="surface normal"):
        model.cofield_coefficients(0)


def test_the_surface_model_survives_a_round_trip(tmp_path, world):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel
    fs, _, mf = world
    y = np.linspace(5.0, 7.0, 5)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="surface",
                     surf_z_min=1.5, surf_z_max=5.5, surf_z_step=0.5, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf[:5]), y, cv_folds=0)
    want = model.predict(AtomTable.from_molecules(mf[5:]))
    path = str(tmp_path / "surface.cmf")
    save_model(path, model)
    back = load_model(path)
    back = back[0] if isinstance(back, tuple) else back
    assert back.kernel_source == "surface" and back.surf_z_step == pytest.approx(0.5)
    assert np.allclose(back.predict(AtomTable.from_molecules(mf[5:])), want)


def test_the_feature_matrix_refuses_mixed_field_sets(world):
    fs, mols, mf = world
    other = build_fields(mols[0], FieldSet.create(physchem=["steric", "hbd"]), "other")
    with pytest.raises(ValueError):
        surface_feature_matrix([mf[0], other], SurfaceSpec(), n_fields=5)


def test_the_nested_cross_validation_keeps_the_kernel(world):
    """nested_cv used to rebuild its fold models from a short list of settings that dropped the
    kernel source, so an invariant kernel was validated as the overlap kernel at one alpha."""
    from cmfqsar.model import CMFModel, nested_cv
    fs, mols, mf = world
    y = np.array([5.0, 5.6, 6.1, 6.4, 7.0, 7.3])
    rng = np.random.default_rng(21)
    shuffled = [build_fields(_moved(m, rng), fs, f"s{i}") for i, m in enumerate(mols)]
    kw = dict(alphas=[1.0], lams=[0.1, 1.0], outer_folds=3, inner_folds=2, repeats=1, seed=0, precompute=False)
    surf = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="surface")
    a = nested_cv(AtomTable.from_molecules(mf), y, surf, **kw)
    b = nested_cv(AtomTable.from_molecules(shuffled), y, surf, **kw)
    assert np.allclose(a.y_pred, b.y_pred, atol=1e-8)                 # in-plane invariant, as fitted
    ac = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation")
    c = nested_cv(AtomTable.from_molecules(mf), y, ac, **kw)
    assert not np.allclose(a.y_pred, c.y_pred, atol=1e-6)             # and not the 3D kernel either
    over = CMFModel(fieldset=fs, method="krr", normalize=True)
    d = nested_cv(AtomTable.from_molecules(mf), y, over, **kw)
    assert not np.allclose(a.y_pred, d.y_pred, atol=1e-6)             # nor the overlap kernel
    assert surf.fresh_copy().kernel_source == "surface" and surf.fresh_copy().surf_z_step == surf.surf_z_step
