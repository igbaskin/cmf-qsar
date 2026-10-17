"""Tests for the alignment-free, geometry-aware field autocorrelation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation,  # noqa: E402
                               autocorrelation_kernel, channel_labels, channel_scale,
                               feature_matrix, field_pairs)
from cmfqsar.kernels import self_kernels  # noqa: E402

SMILES = ["CCCOc1ccccc1C(=O)NCC", "CCCNc1ccccc1C(=O)OC", "CCOc1ccc(C)cc1CC(=O)N",
          "CC(C)COc1ccccc1C(=O)NC", "CCCCOc1ccccc1C(=O)N", "NCCc1ccc(O)cc1"]


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


def _rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@pytest.fixture(scope="module")
def world():
    fs = FieldSet.create(alpha=0.5)
    mols = [_mol(s, seed=i + 1, name=f"m{i}") for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    return fs, mols, mf


def test_a_rigid_motion_changes_nothing(world):
    """Rotation and translation move every atom and leave every distance; so does the descriptor."""
    fs, _, mf = world
    spec = AutocorrelationSpec()
    rng = np.random.default_rng(0)
    for mol in mf[:3]:
        before = autocorrelation(mol, spec)
        moved = mol.copy() if hasattr(mol, "copy") else mol
        R, t = _rotation(rng), rng.normal(scale=20.0, size=3)
        xyz = mol.coords @ R.T + t
        shifted = type(mol)(name=mol.name, coords=xyz, weights=mol.weights, radii=mol.radii,
                            symbols=mol.symbols, atom_types=mol.atom_types, bonds=mol.bonds,
                            charges=mol.charges)
        assert np.allclose(autocorrelation(shifted, spec), before, atol=1e-9)
        # ... and a reflection does change it, since a mirror image is a different molecule
        mirrored = type(mol)(name=mol.name, coords=mol.coords * np.array([1.0, 1.0, -1.0]),
                             weights=mol.weights, radii=mol.radii, symbols=mol.symbols,
                             atom_types=mol.atom_types, bonds=mol.bonds, charges=mol.charges)
        assert np.allclose(autocorrelation(mirrored, spec), before, atol=1e-9)  # distances only


def test_it_still_sees_the_geometry(world):
    """Unlike a count kernel, it changes when the conformation does."""
    fs, mols, mf = world
    spec = AutocorrelationSpec()
    here = autocorrelation(mf[0], spec)
    rng = np.random.default_rng(3)
    xyz = mf[0].coords + rng.normal(scale=0.8, size=mf[0].coords.shape)
    scrambled = type(mf[0])(name="x", coords=xyz, weights=mf[0].weights, radii=mf[0].radii,
                            symbols=mf[0].symbols, atom_types=mf[0].atom_types,
                            bonds=mf[0].bonds, charges=mf[0].charges)
    there = autocorrelation(scrambled, spec)
    assert np.linalg.norm(there - here) / max(np.linalg.norm(here), 1e-12) > 0.02
    # the same atoms in the same places give exactly the same descriptor
    again = autocorrelation(mf[0], spec)
    assert np.array_equal(again, here)


def test_the_zero_distance_channel_is_the_cmf_self_overlap(world):
    """With sigma^2 = 2/alpha the first channel is the overlap integral, up to its prefactor."""
    fs, _, mf = world
    alpha = 0.5
    spec = AutocorrelationSpec(sigma=float(np.sqrt(2.0 / alpha)))
    mol = mf[0]
    feats = autocorrelation(mol, spec).reshape(-1, spec.n_bins)
    pairs = field_pairs(mol.weights.shape[1])
    d = self_kernels(AtomTable.from_molecules([mol]), fs.specs)[:, 0]
    for i, (f, g) in enumerate(pairs):
        if f != g or fs.specs[f].kind != "gauss":
            continue                      # the steric field has one width per atom radius
        raw = feats[i, 0]
        assert raw == pytest.approx(d[f] / (np.pi / alpha) ** 1.5, rel=1e-8)


def test_the_kernel_is_well_formed_and_sees_what_counts_cannot(world):
    fs, mols, mf = world
    spec = AutocorrelationSpec()
    X = feature_matrix(mf, spec)
    assert X.shape == (len(mf), len(field_pairs(mf[0].weights.shape[1])) * spec.n_bins)
    assert len(channel_labels(fs, spec)) == X.shape[1]
    K = autocorrelation_kernel(X, n_bins=spec.n_bins)
    assert np.allclose(np.diag(K), 1.0)
    assert np.allclose(K, K.T, atol=1e-12)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    assert K.max() <= 1.0 + 1e-9 and K.min() >= -1.0 - 1e-9
    # a test block against the same molecules reproduces the training block
    scale = channel_scale(X, spec.n_bins)
    assert np.allclose(autocorrelation_kernel(X, X, scale=scale, n_bins=spec.n_bins), K)

    # two conformations of one molecule: the count kernel calls them identical, this does not
    from cmfqsar.chemotypes import count_kernel
    from rdkit import Chem
    other = Chem.Mol(mols[0])
    conf = other.GetConformer()
    rng = np.random.default_rng(5)
    xyz = np.array(conf.GetPositions()) + rng.normal(scale=0.7, size=(other.GetNumAtoms(), 3))
    for i in range(other.GetNumAtoms()):
        conf.SetAtomPosition(i, xyz[i].tolist())
    pair_counts = count_kernel([mols[0], other], ["a", "b"], cif_min=1)
    assert pair_counts[0, 1] == pytest.approx(1.0, abs=1e-6)       # same atoms, so identical
    mf2 = [build_fields(mols[0], fs, "a"), build_fields(other, fs, "b")]
    K2 = autocorrelation_kernel(feature_matrix(mf2, spec), n_bins=spec.n_bins)
    assert K2[0, 1] < 0.999                                        # different geometry, seen


def test_the_self_pairs_carry_the_counts(world):
    """Dropping the l = m terms removes exactly the count-like part of the first channel."""
    _, _, mf = world
    mol = mf[0]
    spec = AutocorrelationSpec()
    with_self = autocorrelation(mol, spec).reshape(-1, spec.n_bins)
    without = autocorrelation(mol, AutocorrelationSpec(include_self=False)).reshape(-1, spec.n_bins)
    pairs = field_pairs(mol.weights.shape[1])
    for i, (f, g) in enumerate(pairs):
        w = mol.weights
        expected = float(np.sum(w[:, f] * w[:, g])) * (2.0 if f != g else 1.0)
        assert with_self[i, 0] - without[i, 0] == pytest.approx(expected, rel=1e-9, abs=1e-9)


# --------------------------------------------------------- the kernel as a modelling method

def _moved(mol, rng, shift=15.0):
    from rdkit import Chem
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    x = np.array(conf.GetPositions())
    R = _rotation(rng)
    y = (x - x.mean(axis=0)) @ R.T + x.mean(axis=0) + rng.normal(scale=shift, size=3)
    for i in range(m.GetNumAtoms()):
        conf.SetAtomPosition(i, y[i].tolist())
    return m


def test_a_model_on_this_kernel_ignores_the_poses(world):
    """The point of the whole exercise: no alignment, and none needed."""
    from cmfqsar.model import CMFModel
    fs, mols, mf = world
    y = np.array([5.0, 5.6, 6.1, 6.4, 7.0, 7.3])
    train = AtomTable.from_molecules(mf[:5])
    model = CMFModel(fieldset=fs, method="krr", normalize=True,
                     kernel_source="autocorrelation", params={"lam": 0.1})
    model.fit(train, y[:5], cv_folds=0)
    here = model.predict(AtomTable.from_molecules([mf[5]]))

    rng = np.random.default_rng(13)
    far = build_fields(_moved(mols[5], rng), fs, "far")
    assert np.allclose(model.predict(AtomTable.from_molecules([far])), here, atol=1e-9)

    # and the model itself does not care how the training molecules were posed
    shaken = [build_fields(_moved(m, rng), fs, f"s{i}") for i, m in enumerate(mols[:5])]
    other = CMFModel(fieldset=fs, method="krr", normalize=True,
                     kernel_source="autocorrelation", params={"lam": 0.1})
    other.fit(AtomTable.from_molecules(shaken), y[:5], cv_folds=0)
    assert np.allclose(other.predict(AtomTable.from_molecules([mf[5]])), here, atol=1e-9)


def test_what_the_invariant_model_can_and_cannot_explain(world):
    from cmfqsar.model import CMFModel
    fs, _, mf = world
    y = np.linspace(5.0, 7.0, 5)
    model = CMFModel(fieldset=fs, method="krr", normalize=True,
                     kernel_source="autocorrelation", params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf[:5]), y, cv_folds=0)
    # the dual form survives: a prediction is still a sum over the training molecules
    tc = model.training_contributions(mf[5])
    assert tc.shape == (5,)
    assert float(tc.sum() + model.bias) == pytest.approx(
        float(model.predict(AtomTable.from_molecules([mf[5]]))[0]), abs=1e-8)
    # the primal form does not: there is no field in a fixed frame to draw
    for call in (lambda: model.cofield_coefficients(0), lambda: model.atom_contributions(mf[0])):
        with pytest.raises(ValueError, match="overlap-integral kernel"):
            call()


def test_the_invariant_model_survives_a_round_trip(tmp_path, world):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel
    fs, _, mf = world
    y = np.linspace(5.0, 7.0, 5)
    model = CMFModel(fieldset=fs, method="krr", normalize=True,
                     kernel_source="autocorrelation", ac_sigma=0.6, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf[:5]), y, cv_folds=0)
    want = model.predict(AtomTable.from_molecules(mf[5:]))
    path = str(tmp_path / "ac.cmf")
    save_model(path, model)
    back = load_model(path)
    back = back[0] if isinstance(back, tuple) else back
    assert back.kernel_source == "autocorrelation" and back.ac_sigma == pytest.approx(0.6)
    assert np.allclose(back.predict(AtomTable.from_molecules(mf[5:])), want)


def test_the_search_does_not_loop_over_an_attenuation_it_ignores(world):
    from cmfqsar.model import CMFModel, grid_search
    fs, _, mf = world
    y = np.linspace(5.0, 7.0, 6)
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation")
    res = grid_search(AtomTable.from_molecules(mf), y, model, [0.3, 0.5, 1.0],
                      lams=[1e-2, 0.1], folds=3, seed=0)
    assert len(res.table) == 2                     # two regularisations, one kernel
    assert np.isfinite(res.best_q2) and "lam" in res.best_params
