"""Tests for the alignment-free, geometry-aware field autocorrelation."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.invariant import (AutocorrelationSpec, autocorrelation,  # noqa: E402
                               autocorrelation_kernel, block_sizes, channel_labels,
                               channel_scale, chirality, chirality_triples, feature_matrix,
                               field_pairs)
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
    """The purpose of the whole method: no alignment, and none needed."""
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

    # and the model itself is unaffected by how the training molecules were posed
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
    # the co-field does not: there is no fixed frame to draw a field in
    with pytest.raises(ValueError, match="overlap-integral kernel"):
        model.cofield_coefficients(0)
    # the split over atoms does survive, because every channel is a sum over atom pairs
    # (plan item SC6); it is tested in full in test_invariant_atoms.py
    c = model.atom_contributions(mf[5])
    assert float(c.sum() + model.bias) == pytest.approx(
        float(model.predict(AtomTable.from_molecules([mf[5]]))[0]), abs=1e-8)


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


# ----------------------------------------------------------------------------------------------
# the three-point signed channels, which are what makes enantiomers distinguishable
# ----------------------------------------------------------------------------------------------

CHIRAL = AutocorrelationSpec(chirality=True)


def _mirrored(mol):
    """The true enantiomer of a conformation: every coordinate reflected, the fields unchanged."""
    from cmfqsar.fields import MoleculeFields
    return MoleculeFields(name=mol.name + "_mirror",
                          coords=np.asarray(mol.coords, float) * np.array([1.0, 1.0, -1.0]),
                          weights=mol.weights, radii=mol.radii, symbols=mol.symbols,
                          atom_types=mol.atom_types, bonds=mol.bonds, charges=mol.charges)


def _rigidly_moved(mol, R, t):
    from cmfqsar.fields import MoleculeFields
    return MoleculeFields(name=mol.name + "_moved",
                          coords=np.asarray(mol.coords, float) @ np.asarray(R, float).T + t,
                          weights=mol.weights, radii=mol.radii, symbols=mol.symbols,
                          atom_types=mol.atom_types, bonds=mol.bonds, charges=mol.charges)


def test_the_default_is_untouched(world):
    """Chirality is off unless asked for, so a model built before it existed is unchanged."""
    fs, _, mf = world
    spec = AutocorrelationSpec()
    assert spec.chirality is False
    n_pairs = len(field_pairs(len(fs.specs)))
    plain = autocorrelation(mf[0], spec)
    assert plain.shape == (n_pairs * spec.n_bins,)
    assert block_sizes(spec, len(fs.specs)) == [spec.n_bins] * n_pairs
    assert len(channel_labels(fs, spec)) == len(plain)
    # and the chirality block is appended, not mixed in: the pair part is bit-identical
    with_chi = autocorrelation(mf[0], CHIRAL)
    assert np.array_equal(with_chi[:len(plain)], plain)
    assert len(with_chi) == len(plain) + len(chirality_triples(len(fs.specs))) * CHIRAL.n_shells
    assert len(channel_labels(fs, CHIRAL)) == len(with_chi)


def test_the_triples_are_the_independent_ones():
    assert chirality_triples(3) == [(0, 1, 2)]
    assert len(chirality_triples(5)) == 10                  # C(5, 3)
    assert all(f < g < h for f, g, h in chirality_triples(5))
    assert chirality_triples(2) == [] and chirality_triples(0) == []


def test_the_two_point_form_cannot_tell_a_molecule_from_its_mirror(world):
    """The gap this block exists to close, stated as a measurement."""
    fs, _, mf = world
    for m in mf[:3]:
        plain = AutocorrelationSpec()
        assert np.allclose(autocorrelation(m, plain), autocorrelation(_mirrored(m), plain),
                           atol=1e-12)


def test_the_signed_channels_negate_under_reflection(world):
    fs, _, mf = world
    for m in mf[:3]:
        c = chirality(m, CHIRAL)
        assert np.linalg.norm(c) > 0, m.name
        assert np.allclose(c, -chirality(_mirrored(m), CHIRAL), rtol=1e-10, atol=1e-12)


def test_the_signed_channels_survive_a_rigid_motion(world):
    """Rotation and translation leave every channel exactly where it was."""
    fs, _, mf = world
    rng = np.random.default_rng(0)
    for m in mf[:3]:
        R = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        if np.linalg.det(R) < 0:
            R[:, 0] *= -1.0
        moved = _rigidly_moved(m, R, rng.normal(scale=6.0, size=3))
        assert np.allclose(chirality(m, CHIRAL), chirality(moved, CHIRAL), atol=1e-9)


def test_the_factorisation_is_the_triple_sum_it_claims_to_be(world):
    """The purpose of the block: det[A_f, A_g, A_h] IS the sum over every atom triple."""
    import itertools
    fs, _, mf = world
    mol = mf[0]
    X = np.asarray(mol.coords, float)
    X = X - X.mean(axis=0)
    W = np.asarray(mol.weights, float)
    spec = AutocorrelationSpec(chirality=True, chirality_shells=1)
    got = chirality(mol, spec).reshape(len(chirality_triples(W.shape[1])), 1)[:, 0]
    trips = chirality_triples(W.shape[1])
    # the explicit O(N^3) sum, over a molecule small enough to afford it
    idx = range(len(X))
    for i, (f, g, h) in enumerate(trips):
        total = 0.0
        for l, m, n in itertools.product(idx, idx, idx):
            total += W[l, f] * W[m, g] * W[n, h] * float(np.dot(X[l], np.cross(X[m], X[n])))
        assert total == pytest.approx(got[i], rel=1e-7, abs=1e-9)


def test_a_planar_molecule_has_no_handedness():
    """Three vectors in a plane span no volume, and a planar molecule is achiral."""
    from cmfqsar.fields import MoleculeFields
    rng = np.random.default_rng(1)
    n = 9
    xy = rng.normal(size=(n, 2)) * 2.0
    coords = np.column_stack([xy, np.zeros(n)])
    mol = MoleculeFields(name="flat", coords=coords, weights=rng.normal(size=(n, 5)),
                         radii=np.full(n, 1.7), symbols=["C"] * n, atom_types=["C.3"] * n,
                         bonds=np.zeros((0, 2), int), charges=np.zeros(n))
    assert np.allclose(chirality(mol, CHIRAL), 0.0, atol=1e-10)


def test_the_kernel_separates_enantiomers_that_it_could_not_before(world):
    """One number for the whole claim: the kernel between a molecule and its mirror image."""
    fs, _, mf = world
    pairs = [mf[0], _mirrored(mf[0])]
    plain = AutocorrelationSpec()
    Kp = autocorrelation_kernel(feature_matrix(pairs, plain), n_bins=plain.n_bins)
    assert Kp[0, 1] == pytest.approx(1.0, abs=1e-12)        # indistinguishable, exactly
    X = feature_matrix(pairs, CHIRAL)
    Kc = autocorrelation_kernel(X, n_bins=CHIRAL.n_bins,
                                blocks=block_sizes(CHIRAL, len(fs.specs)))
    assert Kc[0, 1] < 0.95
    # on the signed block alone they are exact antipodes
    n_pair_cols = len(field_pairs(len(fs.specs))) * CHIRAL.n_bins
    trips = len(chirality_triples(len(fs.specs)))
    Konly = autocorrelation_kernel(X[:, n_pair_cols:], n_bins=CHIRAL.n_shells,
                                   blocks=[CHIRAL.n_shells] * trips)
    assert Konly[0, 1] == pytest.approx(-1.0, abs=1e-9)


def test_the_scaling_honours_blocks_of_different_widths(world):
    """A volume scales as a length cubed, so the block needs its own scale or it is negligible."""
    fs, _, mf = world
    X = feature_matrix(mf, CHIRAL)
    blocks = block_sizes(CHIRAL, len(fs.specs))
    assert sum(blocks) == X.shape[1]
    s = channel_scale(X, CHIRAL.n_bins, blocks=blocks)
    Xs = X * s
    start = 0
    for width in blocks:
        block = Xs[:, start:start + width]
        if np.abs(block).max() > 0:
            assert float(np.sqrt(np.mean(block ** 2))) == pytest.approx(1.0, rel=1e-9)
        start += width
    # without the explicit blocks the uniform assumption splits wrongly, which is why it is passed
    assert len(channel_scale(X, CHIRAL.n_bins)) == X.shape[1]


def test_a_model_can_be_fitted_on_the_kernel_with_chirality(world):
    """It is a kernel like any other: the learner, cross-validation and save/load all work."""
    from cmfqsar.model import CMFModel
    fs, _, mf = world
    table = AtomTable.from_molecules(mf)
    y = np.array([5.0, 5.4, 6.1, 6.6, 7.0, 7.4])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1},
                     kernel_source="autocorrelation", ac_chirality=True)
    model.fit(table, y, cv_folds=0)
    pred = model.predict(table)
    assert pred.shape == (len(y),) and np.isfinite(pred).all()
    # the chirality setting is part of what the model is, so it must survive a round trip
    assert model._ac_spec().chirality is True
    assert sum(model.ac_blocks_) == model.ac_train_.shape[1]


# ----------------------------------------------------------------------------------------------
# mixing coefficients over the channel blocks: the h_f of the invariant kernels
# ----------------------------------------------------------------------------------------------

def test_block_weights_of_one_are_the_kernel_as_it_was(world):
    from cmfqsar.invariant import block_group_keys, group_block_weights
    from cmfqsar.model import CMFModel
    fs, _, mf = world
    table = AtomTable.from_molecules(mf)
    y = np.array([5.0, 5.4, 6.1, 6.6, 7.0, 7.4])
    plain = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation",
                     params={"lam": 0.1})
    ones = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation",
                    params={"lam": 0.1}, ac_block_weights={"physchem": 1.0, "x*physchem": 1.0})
    assert np.allclose(plain.training_kernel(table), ones.training_kernel(table), atol=1e-12)
    assert ones.ac_block_weights_.shape == (len(ones.ac_blocks_),)
    assert np.all(ones.ac_block_weights_ == 1.0)
    plain.fit(AtomTable.from_molecules(mf[:5]), y[:5], cv_folds=0)
    ones.fit(AtomTable.from_molecules(mf[:5]), y[:5], cv_folds=0)
    assert np.allclose(plain.predict(table), ones.predict(table), atol=1e-12)
    # the default grouping puts every field of this set in one group, and an empty mapping
    # weighs every block one
    keys = block_group_keys([sp.name for sp in fs.specs])
    assert set(keys) == {"physchem"} and len(keys) == len(field_pairs(len(fs.specs)))
    assert np.all(group_block_weights(keys, {}) == 1.0)
    assert plain.block_weight_report() == []


def test_a_zero_weight_removes_a_block_exactly(world):
    """Two fields kept, every block touching the others weighed zero: the kernel of a field set
    that holds the two fields alone, to the last digit, since a zero block enters neither the
    dot products nor the norms."""
    from cmfqsar.model import CMFModel
    fs, mols, mf = world
    table = AtomTable.from_molecules(mf)
    groups = {"electrostatic": "kept", "steric": "kept"}
    full = CMFModel(fieldset=fs, normalize=True, kernel_source="autocorrelation",
                    ac_block_weights={"physchem": 0.0, "physchem*kept": 0.0},
                    ac_block_groups=groups)
    K_full = full.training_kernel(table)
    fs2 = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5)
    table2 = AtomTable.from_molecules([build_fields(m, fs2, f"m{i}") for i, m in enumerate(mols)])
    small = CMFModel(fieldset=fs2, normalize=True, kernel_source="autocorrelation")
    assert np.allclose(K_full, small.training_kernel(table2), atol=1e-10)
    kinds = {r["kind"]: r for r in full.block_weight_report()}
    assert set(kinds) == {"kept", "physchem", "kept*physchem"}
    assert kinds["kept"]["n_blocks"] == 3 and kinds["kept"]["min"] == 1.0
    assert kinds["physchem"]["max"] == 0.0 and kinds["kept*physchem"]["max"] == 0.0
    assert kinds["kept"]["share"] == pytest.approx(1.0)
    # either order of the groups names the same kind of block
    other = CMFModel(fieldset=fs, normalize=True, kernel_source="autocorrelation",
                     ac_block_weights={"physchem": 0.0, "kept*physchem": 0.0},
                     ac_block_groups=groups)
    assert np.allclose(other.training_kernel(table), K_full, atol=1e-12)


def test_balanced_weights_give_every_kind_of_block_the_same_share(world):
    from cmfqsar.invariant import block_sizes
    from cmfqsar.model import CMFModel
    fs, _, mf = world
    table = AtomTable.from_molecules(mf)
    model = CMFModel(fieldset=fs, normalize=True, kernel_source="autocorrelation",
                     ac_block_weights="balanced", ac_block_groups={"electrostatic": "q"})
    model.training_kernel(table)
    rows = model.block_weight_report()
    assert {r["kind"] for r in rows} == {"q", "physchem", "physchem*q"}
    for r in rows:
        assert r["share"] == pytest.approx(1.0 / 3)
        assert r["min"] == pytest.approx(1.0 / np.sqrt(r["n_blocks"]))
    # with the signed three-point block on, the triples are blocks of their own kind too
    chiral = CMFModel(fieldset=fs, normalize=True, kernel_source="autocorrelation",
                      ac_chirality=True, ac_block_weights="balanced",
                      ac_block_groups={"electrostatic": "q"})
    chiral.training_kernel(table)
    assert len(chiral.ac_block_weights_) == len(block_sizes(CHIRAL, len(fs.specs)))
    assert len(chiral.ac_block_keys_) == len(chiral.ac_block_weights_)


def test_aligned_weights_follow_the_activity_and_vanish_on_a_dead_block():
    from scipy.optimize import nnls
    from cmfqsar.invariant import align_block_weights
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 12))
    X[:, 9:] = 0.0                                             # a block that is zero throughout
    y = X[:, :3].sum(axis=1) + 0.1 * rng.normal(size=60)       # the activity lives in block 0
    w = align_block_weights(X, y, [3, 3, 3, 3])
    assert w.shape == (4,) and np.isfinite(w).all() and (w >= 0).all()
    assert w[3] == 0.0 and w[0] > w[1] and w[0] > w[2]
    # the joint alignment is the non-negative least-squares fit of the centred block kernels to
    # the ideal kernel, computed here in feature space; check it against the kernels themselves
    Xc, yc = X - X.mean(axis=0), y - y.mean()
    A = np.stack([(Xc[:, 3 * b:3 * b + 3] @ Xc[:, 3 * b:3 * b + 3].T).ravel() for b in range(3)],
                 axis=1)
    h, _ = nnls(A, np.outer(yc, yc).ravel())
    md = np.array([np.mean(np.sum(X[:, 3 * b:3 * b + 3] ** 2, axis=1)) for b in range(3)])
    h *= md.sum() / (h * md).sum()
    wf = align_block_weights(X, y, [3, 3, 3, 3], mode="alignf", rho=0.0)
    assert np.allclose(wf[:3], np.sqrt(h), atol=1e-6) and wf[3] == 0.0
    # the shrinkage pulls the joint solution towards equal weights
    ws = align_block_weights(X, y, [3, 3, 3, 3], mode="alignf", rho=1.0)
    assert ws[1] > wf[1] and ws[2] > wf[2] and ws[0] < wf[0]
    with pytest.raises(ValueError, match="alignment mode"):
        align_block_weights(X, y, [3, 3, 3, 3], mode="nonsense")


def test_the_aligned_model_round_trips_and_re_derives_its_weights_per_fold(tmp_path, world):
    from cmfqsar.dataio import load_model, save_model
    from cmfqsar.model import CMFModel, cv_predict
    fs, _, mf = world
    y = np.array([5.0, 5.4, 6.1, 6.6, 7.0, 7.4])
    train = AtomTable.from_molecules(mf[:5])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation",
                     params={"lam": 0.1}, ac_block_weights="align",
                     ac_block_groups={"electrostatic": "q"})
    model.fit(train, y[:5], cv_folds=3)
    assert np.isfinite(model.stats.q2)
    assert model.ac_block_weights_.shape == (len(model.ac_blocks_),)
    assert (model.ac_block_weights_ >= 0).all() and model.ac_block_weights_.max() > 0
    want = model.predict(AtomTable.from_molecules(mf[5:]))
    # the cross-validated predictions were made with weights from the training part of each
    # fold, not with the weights of the whole set, which would have seen the left-out activities
    K = model.training_kernel(train, y=y[:5])
    leaky = cv_predict(K, y[:5], "krr", {"lam": 0.1}, folds=3, seed=0)
    assert not np.allclose(model.stats.extra["y_cv"], leaky, atol=1e-6)
    path = str(tmp_path / "aligned.cmf")
    save_model(path, model)
    back = load_model(path)[0]
    assert back.ac_block_weights == "align" and back.ac_block_groups == {"electrostatic": "q"}
    assert np.allclose(back.ac_block_weights_, model.ac_block_weights_)
    assert back.block_weight_report() == model.block_weight_report()
    assert np.allclose(back.predict(AtomTable.from_molecules(mf[5:])), want, atol=1e-12)
    # asked for the kernel without the activities the weights come from, it says so
    with pytest.raises(ValueError, match="activities"):
        CMFModel(fieldset=fs, normalize=True, kernel_source="autocorrelation",
                 ac_block_weights="align").training_kernel(train)


def test_the_search_and_the_nested_cross_validation_keep_the_block_weights(world):
    from cmfqsar.model import CMFModel, grid_search, nested_cv
    fs, mols, mf = world
    table = AtomTable.from_molecules(mf)
    y = np.array([5.0, 5.4, 6.1, 6.6, 7.0, 7.4])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation",
                     ac_block_weights="alignf", ac_block_rho=0.5,
                     ac_block_groups={"electrostatic": "q"})
    fresh = model.fresh_copy()
    assert fresh.ac_block_weights == "alignf" and fresh.ac_block_rho == 0.5
    assert fresh.ac_block_groups == {"electrostatic": "q"}
    res = grid_search(table, y, model, [1.0], lams=[0.01, 0.1], folds=3, seed=0)
    assert len(res.table) == 2 and np.isfinite(res.best_rmse)
    kw = dict(alphas=[1.0], lams=[0.01, 0.1], outer_folds=3, inner_folds=2, repeats=1, seed=0,
              precompute=False)
    ncv = nested_cv(table, y, model, **kw)
    assert ncv.y_pred.shape == (6,) and np.isfinite(ncv.y_pred).all()
    # a mapping that weighs everything but one group zero validates as that group's field set
    only = CMFModel(fieldset=fs, method="krr", normalize=True, kernel_source="autocorrelation",
                    ac_block_weights={"physchem": 0.0, "physchem*q": 0.0},
                    ac_block_groups={"electrostatic": "q"})
    fs_q = FieldSet.create(physchem=["electrostatic"], alpha=0.5)
    table_q = AtomTable.from_molecules([build_fields(m, fs_q, f"m{i}") for i, m in enumerate(mols)])
    alone = CMFModel(fieldset=fs_q, method="krr", normalize=True, kernel_source="autocorrelation")
    assert np.allclose(nested_cv(table, y, only, **kw).y_pred,
                       nested_cv(table_q, y, alone, **kw).y_pred, atol=1e-8)


# --------------------------------------------------------------------------- probe fields


def _probe_mols(n=5):
    return [_mol(SMILES[i], seed=i + 1, name=f"m{i}") for i in range(n)]


def test_a_probe_field_needs_the_field_set_to_be_expanded():
    """It used to give a kernel of zeros, which says every molecule is alike."""
    fs = FieldSet.create(physchem=[], probes=["lithium"], alpha=0.5)
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in _probe_mols(3)]
    with pytest.raises(ValueError, match="probe field"):
        feature_matrix(mf, AutocorrelationSpec())


def test_the_expanded_probe_field_carries_the_terms_of_its_expansion():
    from cmfqsar.invariant import expanded_field_count, expanded_field_labels, expanded_weights
    fs = FieldSet.create(physchem=["steric"], probes=["lithium"], alpha=0.5, probe_terms=4)
    mf = build_fields(_probe_mols(1)[0], fs, "m0")
    W = expanded_weights(mf, fs)
    assert W.shape == (len(mf.coords), 5), "one steric column and four probe terms"
    assert np.allclose(W[:, 0], mf.weights[:, 0])
    assert np.allclose(W[:, 1:], mf.probe_coeffs["probe:lithium"])
    assert expanded_field_count(fs) == 5
    assert expanded_field_labels(fs)[1].endswith("term 1")


def test_the_probe_kernel_distinguishes_molecules_once_expanded():
    fs = FieldSet.create(physchem=[], probes=["lithium"], alpha=0.5, probe_terms=4)
    mols = _probe_mols(5)
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in mols]
    spec = AutocorrelationSpec()
    F = feature_matrix(mf, spec, fieldset=fs)
    from cmfqsar.invariant import expanded_field_count
    assert F.shape == (5, sum(block_sizes(spec, expanded_field_count(fs))))
    K = autocorrelation_kernel(F, n_bins=spec.n_bins,
                               blocks=block_sizes(spec, expanded_field_count(fs)))
    off = K[np.triu_indices_from(K, k=1)]
    assert np.all(np.diag(K) > 0.999), "the kernel is cosine-normalised"
    assert off.std() > 1e-3, "a kernel of one constant would say the molecules are alike"


def test_passing_the_field_set_changes_nothing_without_a_probe():
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5)
    mf = [build_fields(m, fs, m.GetProp("_Name")) for m in _probe_mols(3)]
    spec = AutocorrelationSpec()
    assert np.allclose(feature_matrix(mf, spec), feature_matrix(mf, spec, fieldset=fs))
