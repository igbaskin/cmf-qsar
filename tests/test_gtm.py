"""Tests for the chemical space map: kernel PCA and Generative Topographic Mapping."""
import os
import sys

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.gtm import (GTM, CMFMap, KernelPCA, class_landscape, density,  # noqa: E402
                         landscape, predict_from_landscape)


def test_kernel_pca_reproduces_feature_space_distances():
    """With every component kept, the coordinates must reproduce the original distances."""
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 6))
    kp = KernelPCA(n_components=40).fit(X @ X.T)
    Z = kp.transform()
    assert np.max(np.abs(pdist(Z) - pdist(X))) < 1e-10
    # the in-sample projection through the kernel agrees with the stored coordinates
    assert np.allclose(kp.transform(X @ X.T), Z, atol=1e-9)
    # and a genuinely new molecule lands at the right distance from the training ones
    Y = rng.normal(size=(5, 6))
    assert np.max(np.abs(cdist(kp.transform(Y @ X.T), Z) - cdist(Y, X))) < 1e-9
    assert kp.explained_variance_ratio.sum() == pytest.approx(1.0)


def test_gtm_likelihood_increases_monotonically():
    rng = np.random.RandomState(1)
    T = rng.normal(size=(60, 5))
    g = GTM(grid=10, rbf=4, n_iter=30).fit(T)
    ll = np.asarray(g.log_likelihood_)
    assert len(ll) > 3
    assert np.all(np.diff(ll) > -1e-6)                  # EM never decreases the likelihood
    R = g.responsibilities(T)
    assert R.shape == (100, 60)
    assert np.allclose(R.sum(axis=0), 1.0)              # a posterior over the nodes
    P = g.transform(T)
    assert P.shape == (60, 2) and np.abs(P).max() <= 1.0 + 1e-9    # inside the latent square
    assert g.modes(T).shape == (60, 2)


def test_gtm_separates_two_clusters():
    rng = np.random.RandomState(2)
    T = np.vstack([rng.normal(size=(30, 4)), rng.normal(size=(30, 4)) + 12.0])
    g = GTM(grid=15, rbf=4, n_iter=60).fit(T)
    P = g.transform(T)
    separation = np.linalg.norm(P[:30].mean(0) - P[30:].mean(0))
    spread = max(np.linalg.norm(P[:30] - P[:30].mean(0), axis=1).mean(),
                 np.linalg.norm(P[30:] - P[30:].mean(0), axis=1).mean())
    assert separation > 2 * spread


def test_landscapes_and_prediction():
    rng = np.random.RandomState(3)
    T = np.vstack([rng.normal(size=(30, 4)), rng.normal(size=(30, 4)) + 10.0])
    values = np.concatenate([np.zeros(30), np.ones(30)])
    g = GTM(grid=12, rbf=4, n_iter=50).fit(T)
    R = g.responsibilities(T)
    land = landscape(R, values)
    # supported nodes carry a value, empty ones do not
    assert np.isfinite(land).any() and not np.isfinite(land).all()
    assert np.nanmin(land) >= -1e-9 and np.nanmax(land) <= 1.0 + 1e-9
    pred = predict_from_landscape(R, land)
    good = np.isfinite(pred)
    assert good.sum() == len(values)
    assert np.corrcoef(pred[good], values[good])[0, 1] > 0.9
    # the density is the data support and sums to the number of molecules
    d = density(R)
    assert d.sum() == pytest.approx(len(values))
    # a class landscape is the weighted fraction of the positive class
    cl = class_landscape(R, np.where(values > 0, 1.0, -1.0))
    assert np.nanmin(cl) >= -1e-9 and np.nanmax(cl) <= 1.0 + 1e-9
    # missing values are ignored rather than propagated
    v2 = values.copy()
    v2[:5] = np.nan
    assert np.isfinite(landscape(R, v2)).any()


def test_cmf_map_on_molecules_and_projection():
    """A map built from real CMF kernels places new molecules inside the latent square."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cmfqsar.fields import AtomTable, FieldSet, build_fields

    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O",
              "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "CCCCN", "CCCCCN", "CC(C)CN", "CCN(C)C"]
    mols = []
    for i, s in enumerate(smiles):
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        p = AllChem.ETKDGv3()
        p.randomSeed = i
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        mols.append(m)
    fs = FieldSet.create(alpha=0.6)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    cmap = CMFMap(n_components=6, grid=10, rbf=3, n_iter=40).fit(AtomTable.from_molecules(mf[:9]), fs)
    assert cmap.coords_.shape == (9, 2) and np.abs(cmap.coords_).max() <= 1.0 + 1e-9
    assert cmap.R_.shape == (100, 9)
    land = cmap.activity_landscape(np.arange(9, dtype=float))
    assert cmap.as_image(land).shape == (10, 10)
    assert cmap.density().shape == (100,)
    coords, R = cmap.project(AtomTable.from_molecules(mf[9:]))
    assert coords.shape == (3, 2) and np.abs(coords).max() <= 1.0 + 1e-9
    assert np.allclose(R.sum(axis=0), 1.0)
    # a molecule of the training set projects back onto (almost) its own place
    back, _ = cmap.project(AtomTable.from_molecules([mf[0]]))
    assert np.linalg.norm(back[0] - cmap.coords_[0]) < 0.2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
