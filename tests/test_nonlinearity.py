"""The local co-field of a model with a nonlinear kernel layer (cmfqsar.nonlinearity).

The decisive test is the derivative one: the local co-field claims to be the derivative of the
prediction with respect to the normalised field, so changing one atom's weight by a small amount
and refitting nothing must move the prediction by the overlap of the local co-field with the
change of the normalised field.  Everything else -- the layer derivatives, the spread map, the
index -- is checked against that object or against the linear model, where all of it must
reduce to the ordinary co-field.
"""
import copy
import csv
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.cofields import cofield_volume, make_grid  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import expansion_overlap  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.nonlinearity import (cofield_spread, layer_derivatives, local_cofield_coefficients,  # noqa: E402
                                  local_cofield_volume, local_duals, nonlinearity_indices)

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccccc1F",
          "c1ccc(F)cc1O", "c1ccc(C)cc1O", "c1ccc(Cl)cc1N", "c1ccc(O)cc1C",
          "c1ccc(N)cc1C", "c1ccc(Cl)cc1C", "c1ccc(F)cc1N", "c1ccccc1Br",
          "c1ccc(C)cc1C"]


def _dataset(seed=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fs = FieldSet.create(alpha=0.6)
    mf = []
    for i, s in enumerate(SMILES):
        m = Chem.AddHs(Chem.MolFromSmiles(s))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed + i
        AllChem.EmbedMolecule(m, p)
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        mf.append(build_fields(m, fs, f"m{i}"))
    rng = np.random.RandomState(seed)
    y = np.array([m.coords.shape[0] for m in mf], float)
    y = (y - y.mean()) / y.std() + 0.1 * rng.normal(size=len(mf))
    # an activity that is not additive in the fields: a product of two of its pieces
    y = y + 0.5 * y * np.roll(y, 1)
    return mf, y, fs


@pytest.fixture(scope="module")
def data():
    return _dataset()


def _fit(data, layer, normalize, gamma=1.0, train=None, lam=1e-2):
    mf, y, fs = data
    idx = list(range(len(mf))) if train is None else list(train)
    table = AtomTable.from_molecules([mf[i] for i in idx])
    m = CMFModel(fieldset=fs.copy(), method="krr", params={"lam": lam, "gamma": gamma},
                 normalize=normalize, kernel_layer=layer)
    m.fit(table, y[idx], cv_folds=0)
    return m, table


def _gauss_field(model):
    """A field type with one Gaussian of a known width per atom (the electrostatic one)."""
    return next(f for f, s in enumerate(model.fieldset.specs) if s.enabled and s.kind == "gauss")


# ------------------------------------------------------------------------------ the layer

@pytest.mark.parametrize("layer", ["linear", "gauss", "arccos1", "arccos2"])
def test_layer_derivatives_are_the_layer_and_its_finite_differences(data, layer):
    m, table = _fit(data, layer, normalize=False)
    K = m.joint_kernel(m.K_fields_train)
    d = np.diag(K)
    Kb, dp = K[:5].copy(), d[:5].copy()
    g, g_s, g_n = layer_derivatives(m, Kb, dp, d)
    assert np.allclose(g, m.apply_layer(Kb, dp, d), rtol=1e-12, atol=1e-12)
    # away from a molecule's pair with itself, where the layer clips, the two partial
    # derivatives are what central differences give
    off = ~np.eye(len(d), dtype=bool)[:5]
    eps = 1e-6 * float(np.median(d))
    fd_s = (m.apply_layer(Kb + eps, dp, d) - m.apply_layer(Kb - eps, dp, d)) / (2 * eps)
    fd_n = (m.apply_layer(Kb, dp + eps, d) - m.apply_layer(Kb, dp - eps, d)) / (2 * eps)
    scale = max(float(np.abs(g_s).max()), 1e-12)
    assert np.allclose(g_s[off], fd_s[off], rtol=1e-5, atol=1e-6 * scale)
    assert np.allclose(g_n[off], fd_n[off], rtol=1e-5, atol=1e-6 * scale)
    if layer == "linear":
        assert np.all(g_s == 1.0) and np.all(g_n == 0.0)


# ---------------------------------------------------------------------- the linear limit

def test_for_a_linear_model_everything_reduces_to_the_cofield(data):
    mf, y, fs = data
    m, table = _fit(data, "linear", normalize=True)
    f = _gauss_field(m)
    c0, k0, w0 = m.cofield_coefficients(f)
    for mol in mf[:3]:
        c, k, w = local_cofield_coefficients(m, f, mol)
        assert np.array_equal(c, c0) and np.array_equal(w, w0)
        assert np.allclose(k, k0, rtol=1e-12, atol=1e-14)
    B, b0 = local_duals(m)
    assert np.allclose(B, np.tile(m.coef, (len(y), 1)))
    assert np.all(b0 == 0.0)
    grid = make_grid(table.coords, margin=1.0, spacing=1.5)
    sp = cofield_spread(m, f, grid)
    assert np.allclose(sp.mean, cofield_volume(m, f, grid), atol=1e-12)
    assert float(np.abs(sp.sd).max()) < 1e-12 * float(np.abs(sp.mean).max())
    for idx in nonlinearity_indices(m):
        assert idx.spread == pytest.approx(0.0, abs=1e-10)
        assert idx.cosine == pytest.approx(1.0, abs=1e-8)


# ------------------------------------------------------------------------ the derivative

@pytest.mark.parametrize("layer", ["gauss", "arccos1", "arccos2"])
@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("held_out", [False, True])
def test_the_local_cofield_is_the_derivative_of_the_prediction(data, layer, normalize, held_out):
    """Move one atom's weight in one field by a small amount: the prediction must move by the
    overlap of the local co-field with the change of the normalised field, Eq. (2)."""
    mf, y, fs = data
    train = range(len(mf) - 1) if held_out else None
    m, table = _fit(data, layer, normalize=normalize, train=train)
    mol = mf[-1] if held_out else mf[3]
    f = _gauss_field(m)
    spec = m.fieldset.specs[f]
    hc = float(m.fieldset.mixing_vector()[f] * m.scale[f])
    w = np.asarray(mol.weights[:, f], float)
    k = int(np.argmax(np.abs(w)))

    def predict_with(delta):
        p = copy.deepcopy(mol)
        p.weights = np.array(p.weights, float, copy=True)
        p.weights[k, f] += delta
        return float(m.predict(AtomTable.from_molecules([p]))[0])

    eps = 1e-4 * max(float(np.abs(w).max()), 1e-3)
    dy = (predict_with(eps) - predict_with(-eps)) / (2 * eps)

    centres, coeffs, widths = local_cofield_coefficients(m, f, mol)
    X_c, X_w, X_b = mol.coords, w, np.full(len(w), spec.alpha)
    G_c, G_w, G_b = mol.coords[k:k + 1], np.ones(1), np.full(1, spec.alpha)
    if normalize:
        # the change of the normalised field: the atom's Gaussian, minus its projection on
        # the field, over the field's norm
        norm = np.sqrt(expansion_overlap(X_c, X_w, X_b, X_c, X_w, X_b))
        proj = expansion_overlap(X_c, X_w / norm, X_b, G_c, G_w, G_b)
        d_c = np.vstack([X_c, G_c])
        d_w = np.concatenate([-(X_w / norm) * proj / norm, G_w / norm])
        d_b = np.concatenate([X_b, G_b])
    else:
        d_c, d_w, d_b = G_c, G_w, G_b
    predicted = hc * expansion_overlap(centres, coeffs, widths, d_c, d_w, d_b)
    assert predicted == pytest.approx(dy, rel=2e-3, abs=1e-9)
    # and the ordinary co-field, if one took it anyway, would be a different object
    if not held_out:
        with pytest.raises(ValueError, match="nonlinearity"):
            m.cofield_coefficients(f)


def test_the_self_term_is_kept_only_without_normalisation(data):
    mf, y, fs = data
    m_n, _ = _fit(data, "gauss", normalize=True)
    m_r, _ = _fit(data, "gauss", normalize=False)
    f = _gauss_field(m_n)
    n_basis = len(m_n.training_field_basis(f)[0])
    assert len(local_cofield_coefficients(m_n, f, mf[0])[0]) == n_basis
    assert len(local_cofield_coefficients(m_r, f, mf[0])[0]) == n_basis + mf[0].coords.shape[0]
    _B, b0 = local_duals(m_r)
    assert np.any(b0 != 0.0)


# ------------------------------------------------------------------------------ the spread

@pytest.mark.parametrize("normalize", [True, False])
def test_the_spread_map_is_the_spread_of_the_local_cofields(data, normalize):
    mf, y, fs = data
    m, table = _fit(data, "gauss", normalize=normalize)
    f = _gauss_field(m)
    grid = make_grid(table.coords, margin=1.0, spacing=1.5)
    sp = cofield_spread(m, f, grid, chunk=37)
    vols = np.stack([local_cofield_volume(m, f, mol, grid) for mol in mf])
    assert sp.n == len(mf)
    assert np.allclose(sp.mean, vols.mean(axis=0), atol=1e-10)
    assert np.allclose(sp.sd, vols.std(axis=0), atol=1e-10)
    assert float(sp.sd.max()) > 1e-2 * float(np.abs(sp.mean).max())
    scaled = cofield_spread(m, f, grid, include_scale=True)
    s = m.fieldset.specs[f].h * m.scale[f]
    assert np.allclose(scaled.sd, sp.sd * abs(s)) and np.allclose(scaled.mean, sp.mean * s)


# ------------------------------------------------------------------------------- the index

def test_the_index_is_between_zero_and_one_and_grows_with_the_bandwidth(data):
    spreads, cosines = [], []
    for gamma in (0.02, 1.0, 8.0):
        m, _ = _fit(data, "gauss", normalize=True, gamma=gamma)
        idx = {i.field: i for i in nonlinearity_indices(m)}
        assert "joint" in idx
        for i in idx.values():
            assert 0.0 <= i.spread <= 1.0
        spreads.append(idx["joint"].spread)
        cosines.append(idx["joint"].cosine)
    # a Gaussian layer with a small bandwidth is the linear model plus a small correction
    assert spreads[0] < 0.05
    assert spreads[0] < spreads[1] < spreads[2]
    assert cosines[0] > cosines[1] > cosines[2]


def test_the_index_agrees_with_the_grid(data):
    """The closed form and a fine grid must say the same about one field."""
    mf, y, fs = data
    m, table = _fit(data, "gauss", normalize=True, gamma=2.0)
    f = _gauss_field(m)
    grid = make_grid(table.coords, margin=4.0, spacing=0.35)
    sp = cofield_spread(m, f, grid)
    dv = grid.spacing ** 3
    var = float(np.sum(sp.sd ** 2)) * dv
    total = var + float(np.sum(sp.mean ** 2)) * dv
    closed = {i.field: i for i in nonlinearity_indices(m)}[m.fieldset.specs[f].name]
    assert var / total == pytest.approx(closed.spread, rel=0.05)


# ----------------------------------------------------------------------- what is refused

def test_refusals(data):
    m, table = _fit(data, "gauss", normalize=True)
    with pytest.raises(ValueError, match="nonlinearity"):
        m.cofield_coefficients(0)
    ac = CMFModel(fieldset=data[2].copy(), method="krr", kernel_source="autocorrelation",
                  kernel_layer="gauss")
    ac.fit(table, data[1], cv_folds=0)
    with pytest.raises(ValueError):
        nonlinearity_indices(ac)


# ------------------------------------------------------------------------- the command

def test_cmf_cofield_local_and_spread(tmp_path, capsys):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cmfqsar.cli import main
    sdf = os.path.join(str(tmp_path), "train.sdf")
    with Chem.SDWriter(sdf) as w:
        for k, smi in enumerate(SMILES[:8]):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(mol, p)
            mol.SetProp("_Name", f"m{k}")
            mol.SetProp("activity", f"{4.0 + 0.6 * k + 0.1 * k * k:.2f}")
            w.write(mol)
    out = os.path.join(str(tmp_path), "m.cmf")
    main(["build", sdf, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--kernel-layer", "gauss", "--save", out])
    with pytest.raises(SystemExit, match="local"):
        main(["cofield", out, "--out-prefix", os.path.join(str(tmp_path), "a")])
    prefix = os.path.join(str(tmp_path), "s")
    table = os.path.join(str(tmp_path), "s.csv")
    main(["cofield", out, "--kind", "spread", "--spacing", "1.5", "--out-prefix", prefix,
          "--csv", table])
    text = capsys.readouterr().out
    assert "non-linearity index" in text and "joint:" in text
    assert os.path.exists(f"{prefix}_spread_electrostatic.cube")
    assert os.path.exists(f"{prefix}_spreadmean_electrostatic.cube")
    with open(table, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(0.0 <= float(r["nonlinearity"]) <= 1.0 for r in rows)
    main(["cofield", out, "--kind", "local", "--molecule", sdf, "--spacing", "1.5",
          "--out-prefix", prefix])
    assert os.path.exists(f"{prefix}_local_electrostatic.cube")


# =========================================================================== second derivatives

from cmfqsar.cofields import Grid  # noqa: E402
from cmfqsar.nonlinearity import (Tangent, atom_interactions, atom_probes, curvature_field,  # noqa: E402
                                  gradient_field, interaction_modes, layer_second_derivatives,
                                  pair_interaction_field, probe_gradient, probe_hessian, probe_set)


def _with_weight(mol, f, k, delta):
    p = copy.deepcopy(mol)
    p.weights = np.array(p.weights, float, copy=True)
    p.weights[k, f] += delta
    return p


def _with_probes(mol, f, centres, amplitudes):
    """`mol` with extra atoms of the given weights in field f and nothing in the others."""
    p = copy.deepcopy(mol)
    c = np.atleast_2d(np.asarray(centres, float))
    rows = np.zeros((len(c), p.weights.shape[1]))
    rows[:, f] = np.asarray(amplitudes, float)
    p.coords = np.vstack([p.coords, c])
    p.weights = np.vstack([np.asarray(p.weights, float), rows])
    p.radii = np.concatenate([np.asarray(p.radii, float), np.full(len(c), 1.7)])
    p.charges = np.concatenate([np.asarray(p.charges, float), np.zeros(len(c))])
    p.symbols = list(p.symbols) + ["X"] * len(c)
    p.atom_types = list(p.atom_types) + ["X"] * len(c)
    return p


def _second_difference(predict, eps):
    """d2y/de1 de2 by central differences of a function of two amplitudes."""
    return (predict(eps, eps) - predict(eps, -eps) - predict(-eps, eps) + predict(-eps, -eps)) / (4 * eps * eps)


def test_layer_second_derivatives_are_the_finite_differences_of_the_first(data):
    m, table = _fit(data, "gauss", normalize=False)
    K = m.joint_kernel(m.K_fields_train)
    d = np.diag(K)
    Kb, dp = K[:4].copy(), d[:4].copy()
    g_ss, g_sn, g_nn = layer_second_derivatives(m, Kb, dp, d)
    off = ~np.eye(len(d), dtype=bool)[:4]
    eps = 1e-5 * float(np.median(d))

    def first(Kx, dx):
        return layer_derivatives(m, Kx, dx, d)[1:]

    fd_ss = (first(Kb + eps, dp)[0] - first(Kb - eps, dp)[0]) / (2 * eps)
    fd_sn = (first(Kb, dp + eps)[0] - first(Kb, dp - eps)[0]) / (2 * eps)
    fd_nn = (first(Kb, dp + eps)[1] - first(Kb, dp - eps)[1]) / (2 * eps)
    scale = float(np.abs(g_ss).max())
    assert np.allclose(g_ss[off], fd_ss[off], rtol=1e-4, atol=1e-6 * scale)
    assert np.allclose(g_sn[off], fd_sn[off], rtol=1e-4, atol=1e-6 * scale)
    assert np.allclose(g_nn[off], fd_nn[off], rtol=1e-4, atol=1e-6 * scale)
    lin, _ = _fit(data, "linear", normalize=False)
    assert all(np.all(x == 0.0) for x in layer_second_derivatives(lin, Kb, dp, d))
    arc, _ = _fit(data, "arccos1", normalize=False)
    with pytest.raises(ValueError, match="cusp"):
        layer_second_derivatives(arc, Kb, dp, d)


@pytest.mark.parametrize("layer", ["linear", "gauss"])
@pytest.mark.parametrize("normalize", [True, False])
def test_the_atom_interactions_are_the_second_differences_of_the_prediction(data, layer, normalize):
    mf, y, fs = data
    m, table = _fit(data, layer, normalize=normalize)
    mol = mf[3]
    f = _gauss_field(m)
    H, g = atom_interactions(m, mol, f)
    N = mol.n_atoms
    assert H.shape == (N, N) and np.allclose(H, H.T, atol=1e-12)
    w = np.asarray(mol.weights[:, f], float)
    heavy = [k for k in range(N) if mol.symbols[k] != "H"]
    eps = 1e-3 * max(float(np.abs(w).max()), 1e-3)

    def predict(k, l, e1, e2):
        p = _with_weight(_with_weight(mol, f, k, e1), f, l, e2)
        return float(m.predict(AtomTable.from_molecules([p]))[0])

    scale = max(float(np.abs(H).max()), 1e-12)
    for k, l in [(heavy[0], heavy[1]), (heavy[0], heavy[-1]), (heavy[2], heavy[2])]:
        fd = _second_difference(lambda a, b: predict(k, l, a, b), eps)
        # the second difference carries round-off of order 1e-13 / eps^2, hence the floor
        assert H[k, l] == pytest.approx(fd, rel=2e-3, abs=max(1e-5 * scale, 5e-6))
        fd1 = (predict(k, l, eps, 0.0) - predict(k, l, -eps, 0.0)) / (2 * eps)
        assert g[k] == pytest.approx(fd1, rel=1e-4, abs=1e-9)
    if layer == "linear" and not normalize:
        # a linear model without normalisation is additive: no curvature at all
        assert np.abs(H).max() < 1e-10
    if layer == "linear" and normalize:
        # with cosine normalisation even the linear model is not: the normalisation saturates
        assert np.abs(H).max() > 1e-4


@pytest.mark.parametrize("normalize", [True, False])
def test_probes_anywhere_in_space_match_added_atoms(data, normalize):
    """Two probe atoms added to the molecule, each with a small amplitude: the pair field, the
    curvature and the gradient at those two places are what the added atoms do."""
    mf, y, fs = data
    m, table = _fit(data, "gauss", normalize=normalize)
    mol = mf[5]
    f = _gauss_field(m)
    p1 = mol.coords[0] + np.array([1.3, 0.4, 0.2])
    p2 = mol.coords[4] + np.array([-0.3, 1.1, 0.6])
    a = probe_set(m, f, [p1])
    b = probe_set(m, f, [p2])
    T = Tangent(m, mol)
    H = T.hessian(a, b)
    eps = 2e-3

    def predict(e1, e2):
        p = _with_probes(mol, f, [p1, p2], [e1, e2])
        return float(m.predict(AtomTable.from_molecules([p]))[0])

    fd = _second_difference(predict, eps)
    assert H[0, 0] == pytest.approx(fd, rel=2e-3, abs=1e-9)
    # the gradient at p1 is the first difference
    fd1 = (predict(eps, 0.0) - predict(-eps, 0.0)) / (2 * eps)
    assert probe_gradient(m, mol, a)[0] == pytest.approx(fd1, rel=1e-4, abs=1e-9)
    # the pair field on a one-point grid at p2 is the same number; the curvature at p2 is the
    # second derivative along p2 alone
    grid = Grid(origin=np.asarray(p2, float), spacing=1.0, shape=(1, 1, 1))
    assert pair_interaction_field(m, mol, f, p1, grid)[0, 0, 0] == pytest.approx(H[0, 0], rel=1e-10)
    fd22 = (predict(0.0, eps) - 2 * predict(0.0, 0.0) + predict(0.0, -eps)) / (eps * eps)
    assert curvature_field(m, mol, f, grid)[0, 0, 0] == pytest.approx(fd22, rel=2e-3, abs=1e-9)
    fd2 = (predict(0.0, eps) - predict(0.0, -eps)) / (2 * eps)
    assert gradient_field(m, mol, f, grid)[0, 0, 0] == pytest.approx(fd2, rel=1e-4, abs=1e-9)
    assert probe_hessian(m, mol, b)[0, 0] == pytest.approx(fd22, rel=2e-3, abs=1e-9)


def test_probes_of_two_field_types_interact_too(data):
    mf, y, fs = data
    m, table = _fit(data, "gauss", normalize=True)
    mol = mf[2]
    f1 = _gauss_field(m)
    f2 = next(i for i, s in enumerate(m.fieldset.specs) if s.enabled and i != f1
              and float(np.abs(mol.weights[:, i]).max()) > 0)
    k, l = 0, 1
    H = probe_hessian(m, mol, atom_probes(m, f1, mol), atom_probes(m, f2, mol))
    assert H.shape == (mol.n_atoms, mol.n_atoms)
    eps = 1e-3

    def predict(e1, e2):
        p = _with_weight(_with_weight(mol, f1, k, e1), f2, l, e2)
        return float(m.predict(AtomTable.from_molecules([p]))[0])

    fd = _second_difference(predict, eps)
    assert H[k, l] == pytest.approx(fd, rel=2e-3, abs=1e-6 * max(float(np.abs(H).max()), 1e-12))


def test_interaction_modes_decompose_the_matrix(data):
    mf, y, fs = data
    m, table = _fit(data, "gauss", normalize=True)
    mol = mf[1]
    f = _gauss_field(m)
    H, _g = atom_interactions(m, mol, f)
    modes = interaction_modes(m, mol, f, k=mol.n_atoms)
    rebuilt = sum(md.value * np.outer(md.vector, md.vector) for md in modes)
    assert np.allclose(rebuilt, H, atol=1e-10)
    vals = [abs(md.value) for md in modes]
    assert vals == sorted(vals, reverse=True)
    assert all(np.linalg.norm(md.vector) == pytest.approx(1.0) for md in modes)
    assert len(interaction_modes(m, mol, f, k=2)) == 2


def test_a_missing_field_is_refused_under_normalisation(data):
    mf, y, fs = data
    m, table = _fit(data, "gauss", normalize=True)
    mol = next(x for x in mf if any(float(np.abs(x.weights[:, i]).max()) == 0.0
                                    for i, s in enumerate(m.fieldset.specs) if s.enabled))
    f = next(i for i, s in enumerate(m.fieldset.specs) if s.enabled
             and float(np.abs(mol.weights[:, i]).max()) == 0.0)
    with pytest.raises(ValueError, match="normalised"):
        atom_interactions(m, mol, f)


def test_cmf_cofield_pair_curvature_and_modes(tmp_path, capsys):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from cmfqsar.cli import main
    sdf = os.path.join(str(tmp_path), "train.sdf")
    with Chem.SDWriter(sdf) as w:
        for k, smi in enumerate(SMILES[:8]):
            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(mol, p)
            mol.SetProp("_Name", f"m{k}")
            mol.SetProp("activity", f"{4.0 + 0.6 * k + 0.1 * k * k:.2f}")
            w.write(mol)
    out = os.path.join(str(tmp_path), "m.cmf")
    main(["build", sdf, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--kernel-layer", "gauss", "--save", out])
    prefix = os.path.join(str(tmp_path), "p")
    main(["cofield", out, "--kind", "pair", "--molecule", sdf, "--probe-atom", "0",
          "--fields", "electrostatic", "--spacing", "1.5", "--out-prefix", prefix])
    assert os.path.exists(f"{prefix}_pair_electrostatic.cube")
    main(["cofield", out, "--kind", "curvature", "--molecule", sdf, "--fields", "electrostatic",
          "--spacing", "1.5", "--out-prefix", prefix])
    assert os.path.exists(f"{prefix}_curvature_electrostatic.cube")
    table = os.path.join(str(tmp_path), "h.csv")
    main(["cofield", out, "--kind", "modes", "--molecule", sdf, "--fields", "electrostatic",
          "--csv", table])
    text = capsys.readouterr().out
    assert "mode 1" in text
    with open(table, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {"field", "atom_i", "atom_j", "value"} <= set(rows[0])
    with pytest.raises(SystemExit, match="probe-atom"):
        main(["cofield", out, "--kind", "pair", "--molecule", sdf, "--out-prefix", prefix])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
