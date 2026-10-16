"""Unit tests for the cmfqsar core (run with: python -m pytest tests)."""
import os
import sys

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.atomtypes import assign_atom_types  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields, cif_types_in  # noqa: E402
from cmfqsar.kernels import (atom_kernel_block, kernel_matrices, numerical_overlap,  # noqa: E402
                             self_kernels)
from cmfqsar.model import CMFModel, cv_predict, grid_search, krr_fit, nested_cv  # noqa: E402
from cmfqsar.cofields import (cofield_volume, make_grid, molfield_volume, overlap_volume,  # noqa: E402
                              isosurface)


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


def test_gauss_kernel_matches_numerical_integration():
    x1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.2, -0.4, 0.7])
    alpha = 0.8
    w1, w2 = 0.6, -1.3
    analytic = w1 * w2 * (np.pi / alpha) ** 1.5 * np.exp(-alpha / 4 * np.sum((x1 - x2) ** 2))
    numeric = numerical_overlap(x1, w1, alpha, x2, w2, alpha, spacing=0.1, extent=5.0)
    assert abs(analytic - numeric) < 1e-3 * abs(analytic)


def test_steric_kernel_matches_numerical_integration():
    # rho = E exp(-1/2 alpha (|r-r0|/R)^2)  ->  b = alpha / R^2   (Eq. 10-11)
    x1 = np.array([0.0, 0.0, 0.0])
    x2 = np.array([1.5, 0.3, -0.2])
    alpha = 1.0
    R1, R2 = 1.7, 1.52
    E1, E2 = 0.107, 0.116
    b1, b2 = alpha / R1 ** 2, alpha / R2 ** 2
    analytic = E1 * E2 * np.sqrt(8 * np.pi ** 3 / (b1 + b2) ** 3) * np.exp(-(b1 * b2) / (2 * (b1 + b2)) * np.sum((x1 - x2) ** 2))
    numeric = numerical_overlap(x1, E1, b1, x2, E2, b2, spacing=0.1, extent=7.0)
    assert abs(analytic - numeric) < 1e-3 * abs(analytic)

    # and the vectorised implementation gives the same number
    fs = FieldSet.create(physchem=("steric",), alpha=alpha)
    from cmfqsar.fields import MoleculeFields
    m1 = MoleculeFields("a", x1[None, :], np.array([[E1]]), np.array([R1]), ["C"], ["C.3"], np.zeros((0, 2), int), np.zeros(1))
    m2 = MoleculeFields("b", x2[None, :], np.array([[E2]]), np.array([R2]), ["O"], ["O.3"], np.zeros((0, 2), int), np.zeros(1))
    K = kernel_matrices(AtomTable.from_molecules([m1]), AtomTable.from_molecules([m2]), fs.specs)
    assert abs(K[0, 0, 0] - analytic) < 1e-10


def test_atom_types():
    m = Chem.AddHs(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)[O-]"))
    t = assign_atom_types(m, "sybyl")
    heavy = [t[a.GetIdx()] for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    assert heavy[0] == "C.3"
    assert heavy[1] == "C.2"
    assert heavy[2] == "O.2"
    assert heavy[3] == "O.3"
    assert heavy[4] == "C.ar"
    assert heavy[-1] == "O.co2" and heavy[-2] == "O.co2"
    m2 = Chem.AddHs(Chem.MolFromSmiles("CC(=O)NC1=CC=NC=C1"))
    t2 = assign_atom_types(m2, "sybyl")
    assert t2[3] == "N.am"
    assert "N.ar" in t2


def test_kernel_matrix_symmetric_psd_and_fit():
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "c1ccccc1OC",
              "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C"]
    mols = [_mol(s, i) for i, s in enumerate(smiles)]
    cif = cif_types_in(mols, "sybyl", min_molecules=3)
    fs = FieldSet.create(cif_types=cif, alpha=0.7)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    K = kernel_matrices(table, table, fs.specs)
    assert K.shape == (fs.n_fields, 10, 10)
    assert np.allclose(K, np.transpose(K, (0, 2, 1)), atol=1e-9)
    d = self_kernels(table, fs.specs)
    assert np.allclose(d, np.array([np.diag(K[f]) for f in range(fs.n_fields)]))
    for f in range(fs.n_fields):
        ev = np.linalg.eigvalsh(K[f])
        assert ev.min() > -1e-8 * max(1.0, ev.max())

    y = np.array([1.0, 2.0, 0.5, 1.5, 3.0, 1.2, 0.8, 0.9, 2.2, 1.1])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 1e-3}, normalize=True, scale_fields=True)
    model.fit(table, y, cv_folds=5)
    yfit = model.predict(table)
    assert model.stats.r2 > 0.95
    assert np.allclose(yfit, model.stats.extra["y_fit"])

    # decomposition rules: atoms and training molecules sum to y - b
    for mol in mf[:3]:
        yp = model.predict(AtomTable.from_molecules([mol]))[0]
        assert abs(model.atom_contributions(mol).sum() + model.bias - yp) < 1e-8
        assert abs(model.training_contributions(mol).sum() + model.bias - yp) < 1e-8
        pf = model.atom_contributions(mol, per_field=True)
        assert pf.shape == (fs.n_fields, mol.n_atoms)

    # GP variance is non-negative and finite
    yp2, var = model.predict_with_variance(table)
    assert np.all(np.isfinite(var)) and np.all(var >= 0)

    # SVR path
    m2 = CMFModel(fieldset=fs, method="svr", params={"C": 10.0, "epsilon": 0.05}, normalize=True)
    m2.fit(table, y, cv_folds=3)
    assert np.isfinite(m2.stats.q2)


def test_primal_equals_dual_on_grid():
    """The overlap integral of co-field and molecular field reproduces the kernel prediction."""
    smiles = ["CCO", "CCN", "CCC", "CC(=O)O", "CCF", "CCCl"]
    mols = [_mol(s, i) for i, s in enumerate(smiles)]
    fs = FieldSet.create(physchem=("electrostatic", "steric", "hydrophobic"), alpha=1.0)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    y = np.array([0.1, 0.9, -0.5, 1.2, 0.3, 0.2])
    for normalize in (False, True):
        model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.01}, normalize=normalize)
        model.fit(table, y, cv_folds=0)
        grid = make_grid(table.coords, margin=5.0, spacing=0.25)
        dv = grid.spacing ** 3
        for mol in mf[:2]:
            total = model.bias
            for f in range(fs.n_fields):
                total += overlap_volume(model, mol, f, grid).sum() * dv
            yp = model.predict(AtomTable.from_molecules([mol]))[0]
            assert abs(total - yp) < 0.02 * max(1.0, abs(yp)), (normalize, total, yp)
    vol = cofield_volume(model, 1, grid)
    surf = isosurface(vol, 0.5 * vol.max(), grid)
    assert surf is not None and len(surf[0]) > 0


def test_krr_bias_system():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 3))
    K = X @ X.T
    y = X[:, 0] * 2 + 5.0
    a, b = krr_fit(K, y, 1e-6)
    assert abs(b - 5.0) < 1e-3
    assert np.allclose(K @ a + b, y, atol=1e-3)
    assert abs(a.sum()) < 1e-6


def test_grid_search_and_nested_cv():
    smiles = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1C(=O)O", "c1ccccc1OC",
              "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C", "c1ccc(Br)cc1", "c1ccc(N)cc1C"]
    mols = [_mol(s, i) for i, s in enumerate(smiles)]
    fs = FieldSet.create(physchem=("electrostatic", "steric"), alpha=1.0)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    y = np.arange(12, dtype=float) * 0.3 + np.sin(np.arange(12))
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    res = grid_search(table, y, model, alphas=[0.5, 1.0], lams=[0.01, 1.0], folds=4, optimize_h=True,
                      h_grid=(0.5, 1.0, 2.0), h_passes=1)
    assert res.best_alpha in (0.5, 1.0)
    assert model.fieldset.specs[0].alpha == res.best_alpha
    model.fit(table, y, cv_folds=4, reuse_kernels=True)
    assert np.isfinite(model.stats.q2)
    ncv = nested_cv(table, y, model, alphas=[0.5, 1.0], lams=[0.01, 1.0], outer_folds=3, inner_folds=3, repeats=2)
    assert len(ncv.y_pred) == 12 and np.isfinite(ncv.rmse_ex)


def test_embedding_keeps_rdkit_aromaticity():
    """MMFF optimisation must not leave MMFF aromaticity flags behind (breaks MCS alignment and typing)."""
    from cmfqsar.align import embed_3d, mcs_align
    smi = "COCCOCn1c(Sc2ccccc2)c(C)c(=O)[nH]c1=O"   # HEPT-type HIV-RT inhibitor (pyrimidinedione)
    m = embed_3d(Chem.MolFromSmiles(smi), seed=3)
    arom_ref = [a.GetIsAromatic() for a in Chem.MolFromSmiles(smi).GetAtoms()]
    arom_emb = [a.GetIsAromatic() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    assert arom_emb == arom_ref
    m2 = embed_3d(Chem.MolFromSmiles("COCn1c(Sc2ccccc2)c(C)c(=O)[nH]c1=O"), seed=4)
    rmsd, n = mcs_align(m2, m)
    assert n >= 17 and rmsd < 2.0   # (before the fix the match failed with n = 0)
    # MMFF charges must not modify the molecule either
    fs = FieldSet.create(physchem=("electrostatic",), charge_model="mmff")
    build_fields(m, fs)
    assert [a.GetIsAromatic() for a in m.GetAtoms() if a.GetAtomicNum() > 1] == arom_ref


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
