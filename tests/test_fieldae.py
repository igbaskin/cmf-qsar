"""A field autoencoder with an analytic decoder (plan item E11).

The claim of this module is that the loss is the *exact* field distance and that its gradient is
the exact derivative of that distance, so the training needs no grid.  Every test here checks one
of those against something independent: the NumPy overlap kernels of
:mod:`cmfqsar.kernels`, a central finite difference, the ``ockpca`` one-class score, or an
invariance the construction must have.  Nothing is compared with a stored number.

The tests that need no network run without PyTorch installed; the rest are skipped.
"""
import dataclasses
import math
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar import fieldae  # noqa: E402
from cmfqsar.fields import FieldSpec, MoleculeFields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402

SMILES = ["c1ccccc1C(=O)O", "c1ccc(Cl)cc1C(=O)O", "c1ccc(C)cc1C(=O)O", "c1ccc(O)cc1C(=O)O",
          "c1ccc(N)cc1C(=O)O", "c1ccc(F)cc1C(=O)O", "c1ccc(Br)cc1C(=O)O", "c1ccc(I)cc1C(=O)O"]

needs_torch = pytest.mark.skipif(not fieldae.available(), reason="PyTorch is not installed")


def _mols(n=len(SMILES)):
    out = []
    for k, smi in enumerate(SMILES[:n]):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = k + 1
        AllChem.EmbedMolecule(m, p)
        out.append(m)
    return out


@pytest.fixture(scope="module")
def built():
    """A small table with a radius-scaled field and three shared-width ones."""
    fs = FieldSet.create(physchem=("steric", "electrostatic", "hydrophobic"), alpha=0.5)
    mf = [build_fields(m, fs, f"m{k}") for k, m in enumerate(_mols())]
    table = AtomTable.from_molecules(mf)
    geom = fieldae.geometry(fs, scale=fieldae.field_scales(table, fs))
    return fs, table, geom, fieldae.tensors(table, fs, geom)


@pytest.fixture(scope="module")
def gauss_only():
    """The same molecules with no radius-scaled field, where a copy of the atoms is exact.

    The steric field has a width per atom and a pseudo-atom carries one radius, so a pseudo-atom
    placed on a carbon is not that carbon's steric Gaussian.  Several tests below need a case
    where copying the atoms reproduces the field exactly, and this is it.
    """
    fs = FieldSet.create(physchem=("electrostatic", "hydrophobic"), alpha=0.5)
    mf = [build_fields(m, fs, f"m{k}") for k, m in enumerate(_mols(4))]
    table = AtomTable.from_molecules(mf)
    geom = fieldae.geometry(fs, scale=fieldae.field_scales(table, fs))
    return fs, table, geom, fieldae.tensors(table, fs, geom)


def _torch_overlap(tens, geom, rows_a, rows_b, dtype=None):
    import torch

    dtype = dtype or torch.float64
    a = np.asarray(rows_a, int)
    b = np.asarray(rows_b, int)
    na = int(tens.counts[a].max())
    nb = int(tens.counts[b].max())
    xa = torch.as_tensor(tens.coords[a, :na], dtype=dtype)
    wa = torch.as_tensor(tens.weights[a, :na], dtype=dtype)
    ra = torch.as_tensor(tens.radii[a, :na], dtype=dtype)
    xb = torch.as_tensor(tens.coords[b, :nb], dtype=dtype)
    wb = torch.as_tensor(tens.weights[b, :nb], dtype=dtype)
    rb = torch.as_tensor(tens.radii[b, :nb], dtype=dtype)
    return fieldae.overlap(xa, wa, ra, xb, wb, rb, geom).numpy()


# --------------------------------------------------------------------------- #
# The field set, and what it refuses
# --------------------------------------------------------------------------- #

def test_the_scaling_brings_every_field_to_unit_mean_self_kernel(built):
    fs, table, geom, tens = built
    # k(f, f) averaged over the set is the number of fields, one per field
    assert tens.self_k.mean() == pytest.approx(geom.n_fields, rel=1e-10)


def test_a_probe_field_is_refused():
    fs = FieldSet.create(physchem=("electrostatic",), alpha=0.5)
    # a probe field is one whose name says so: every atom carries several Gaussians
    fs.specs.append(FieldSpec(name="probe:density", kind="probe", alpha=0.5))
    with pytest.raises(ValueError, match="pseudo-atom"):
        fieldae.geometry(fs)


def test_a_descriptor_block_is_refused():
    fs = FieldSet.create(physchem=("electrostatic",), alpha=0.5)
    fs.descriptors = ("MolLogP",)
    with pytest.raises(ValueError, match="descriptor"):
        fieldae.geometry(fs)


def test_the_geometry_groups_the_fields_by_width(built):
    fs, table, geom, tens = built
    gauss, steric = geom.groups()
    assert sorted(len(v) for v in gauss.values()) == [2]      # two fields share alpha = 0.5
    assert len(steric) == 1                                   # the steric field stands alone
    assert geom.names[steric[0]] == "steric"


def test_require_raises_without_torch(monkeypatch):
    monkeypatch.setattr(fieldae, "HAVE_TORCH", False)
    with pytest.raises(ImportError, match="pip install torch"):
        fieldae.require()


# --------------------------------------------------------------------------- #
# The overlap: the formula of the module, for tensors
# --------------------------------------------------------------------------- #

@needs_torch
def test_the_overlap_is_the_kernel_of_the_program(built):
    fs, table, geom, tens = built
    K = fieldae.joint_kernel(table, table, fs, geom, symmetric=True)
    rows = np.arange(table.n_mol)
    got = _torch_overlap(tens, geom, rows, rows)               # molecule i against molecule i
    assert np.allclose(got, np.diag(K), rtol=1e-12, atol=0)
    pairs_a, pairs_b = [0, 1, 2, 3], [4, 5, 6, 7]
    got = _torch_overlap(tens, geom, pairs_a, pairs_b)
    assert np.allclose(got, K[pairs_a, pairs_b], rtol=1e-12, atol=0)


@needs_torch
def test_the_self_overlap_is_the_stored_field_norm(built):
    fs, table, geom, tens = built
    rows = np.arange(table.n_mol)
    assert np.allclose(_torch_overlap(tens, geom, rows, rows), tens.self_k, rtol=1e-12)


@needs_torch
def test_the_overlap_of_pseudo_atoms_is_the_kernel_of_their_table(built):
    fs, table, geom, tens = built
    import torch

    rng = np.random.RandomState(0)
    pos = rng.randn(9, 3) * 2.0
    w = rng.randn(9, geom.n_fields) * 0.3
    pt = fieldae.pseudo_table(pos, w)
    K = fieldae.joint_kernel(pt, table, fs, geom, symmetric=False)[0]
    xa = torch.as_tensor(pos[None], dtype=torch.float64)
    wa = torch.as_tensor(w[None], dtype=torch.float64)
    ra = torch.full((1, 9), fieldae.PSEUDO_RADIUS, dtype=torch.float64)
    for i in range(table.n_mol):
        k = int(tens.counts[i])
        got = fieldae.overlap(xa, wa, ra,
                              torch.as_tensor(tens.coords[i, :k][None], dtype=torch.float64),
                              torch.as_tensor(tens.weights[i, :k][None], dtype=torch.float64),
                              torch.as_tensor(tens.radii[i, :k][None], dtype=torch.float64),
                              geom).numpy()
        assert got[0] == pytest.approx(K[i], rel=1e-12)


@needs_torch
def test_padding_with_zero_weights_changes_nothing(built):
    """The reason a batch of molecules of different size needs no mask."""
    fs, table, geom, tens = built
    import torch

    k = int(tens.counts[0])
    x = torch.as_tensor(tens.coords[0, :k][None], dtype=torch.float64)
    w = torch.as_tensor(tens.weights[0, :k][None], dtype=torch.float64)
    r = torch.as_tensor(tens.radii[0, :k][None], dtype=torch.float64)
    plain = fieldae.overlap(x, w, r, x, w, r, geom).numpy()
    pad = 7
    xp = torch.cat([x, torch.full((1, pad, 3), 123.0, dtype=torch.float64)], dim=1)
    wp = torch.cat([w, torch.zeros((1, pad, geom.n_fields), dtype=torch.float64)], dim=1)
    rp = torch.cat([r, torch.ones((1, pad), dtype=torch.float64)], dim=1)
    padded = fieldae.overlap(xp, wp, rp, xp, wp, rp, geom).numpy()
    assert padded[0] == pytest.approx(plain[0], rel=1e-14)


@needs_torch
def test_the_gradient_is_the_derivative_of_the_overlap(built):
    """Autograd against a central finite difference of the NumPy kernel.

    This is the one thing the training depends on that no other test covers: the derivative of
    the closed-form overlap with respect to a pseudo-atom's position and its weights.
    """
    fs, table, geom, tens = built
    import torch

    rng = np.random.RandomState(1)
    pos = rng.randn(5, 3) * 1.5
    w = rng.randn(5, geom.n_fields) * 0.4
    row = 2
    k = int(tens.counts[row])
    xm = torch.as_tensor(tens.coords[row, :k][None], dtype=torch.float64)
    wm = torch.as_tensor(tens.weights[row, :k][None], dtype=torch.float64)
    rm = torch.as_tensor(tens.radii[row, :k][None], dtype=torch.float64)

    tp = torch.tensor(pos[None], dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(w[None], dtype=torch.float64, requires_grad=True)
    tr = torch.full((1, 5), fieldae.PSEUDO_RADIUS, dtype=torch.float64)
    value = fieldae.overlap(xm, wm, rm, tp, tw, tr, geom).sum()
    value.backward()
    g_pos = tp.grad.numpy()[0]
    g_w = tw.grad.numpy()[0]

    def numeric(p, ww):
        pt = fieldae.pseudo_table(p, ww)
        return fieldae.joint_kernel(pt, table.subset([row]), fs, geom, symmetric=False)[0, 0]

    h = 1e-6
    for a in range(5):
        for c in range(3):
            up, dn = pos.copy(), pos.copy()
            up[a, c] += h
            dn[a, c] -= h
            fd = (numeric(up, w) - numeric(dn, w)) / (2 * h)
            assert g_pos[a, c] == pytest.approx(fd, rel=2e-5, abs=1e-8)
    for a in (0, 3):
        for f in range(geom.n_fields):
            up, dn = w.copy(), w.copy()
            up[a, f] += h
            dn[a, f] -= h
            fd = (numeric(pos, up) - numeric(pos, dn)) / (2 * h)
            assert g_w[a, f] == pytest.approx(fd, rel=2e-5, abs=1e-8)


# --------------------------------------------------------------------------- #
# The reconstruction error
# --------------------------------------------------------------------------- #

@needs_torch
def test_copying_the_atoms_reconstructs_the_field_exactly(gauss_only):
    """``||f - f'|| = 0`` when the pseudo-atoms are the molecule's own atoms."""
    fs, table, geom, tens = gauss_only
    import torch

    row = 1
    k = int(tens.counts[row])
    x = torch.as_tensor(tens.coords[row, :k][None], dtype=torch.float64)
    w = torch.as_tensor(tens.weights[row, :k][None], dtype=torch.float64)
    r = torch.as_tensor(tens.radii[row, :k][None], dtype=torch.float64)
    cross = fieldae.overlap(x, w, r, x, w, r, geom).numpy()[0]
    sq = tens.self_k[row] - 2.0 * cross + cross
    assert abs(sq) < 1e-12 * tens.self_k[row]


@needs_torch
def test_the_relative_error_of_the_network_is_the_numpy_one(gauss_only):
    """What the training minimises and what the pre-image gives are the same number."""
    fs, table, geom, tens = gauss_only
    feats = np.eye(table.n_mol)[:, :4] + 0.1    # any features will do for this identity
    ae = fieldae.FieldAutoencoder(geom, feats.shape[1], code=3, n_pseudo=6, hidden=(16,),
                                  box=tens.box(), weight_scale=tens.weight_scale(), seed=0)
    rows = np.arange(table.n_mol)
    from_torch = fieldae.relative_errors(ae, feats, tens, rows)
    pos, wts = fieldae.decode_rows(ae, feats, rows)
    for i, row in enumerate(rows):
        mf = fieldae.pre_image_fields(pos[i], wts[i])
        pt = AtomTable.from_molecules([mf])
        k_ab = fieldae.joint_kernel(pt, table.subset([row]), fs, geom, symmetric=False)[0, 0]
        k_bb = fieldae.joint_self(pt, fs, geom)[0]
        k_aa = tens.self_k[row]
        rel = math.sqrt(max(k_aa - 2 * k_ab + k_bb, 0.0) / k_aa)
        assert from_torch[i] == pytest.approx(rel, rel=2e-6)


@needs_torch
def test_fitting_pseudo_atoms_to_one_field_removes_the_error(gauss_only):
    """With as many pseudo-atoms as atoms and no code, the fit can be exact -- and nearly is."""
    fs, table, geom, tens = gauss_only
    rows = [0, 2]
    m = int(tens.counts[rows].max())
    rel, pos, w = fieldae.fit_pseudo_atoms(tens, rows, geom, n_pseudo=m, steps=60, lr=0.02)
    assert pos.shape == (2, m, 3)
    assert w.shape == (2, m, geom.n_fields)
    assert float(np.max(rel)) < 0.05          # the start is already close; the fit keeps it there


@needs_torch
def test_fewer_pseudo_atoms_cannot_fit_better_than_more(gauss_only):
    fs, table, geom, tens = gauss_only
    rows = [0, 1, 2, 3]
    small, _p, _w = fieldae.fit_pseudo_atoms(tens, rows, geom, n_pseudo=3, steps=120, lr=0.05)
    large, _p, _w = fieldae.fit_pseudo_atoms(tens, rows, geom, n_pseudo=12, steps=120, lr=0.05)
    assert float(np.mean(large)) < float(np.mean(small))


@needs_torch
def test_the_fit_reduces_the_reconstruction_error(gauss_only):
    fs, table, geom, tens = gauss_only
    rng = np.random.RandomState(0)
    feats = rng.randn(table.n_mol, 6)
    ae = fieldae.FieldAutoencoder(geom, feats.shape[1], code=2, n_pseudo=8, hidden=(32,),
                                  box=tens.box(), weight_scale=tens.weight_scale(), seed=0)
    rows = np.arange(table.n_mol)
    before = float(np.mean(fieldae.relative_errors(ae, feats, tens, rows)))
    hist = fieldae.fit(ae, feats, tens, rows, None, epochs=40, batch=2, lr=5e-3, patience=0)
    after = float(np.mean(fieldae.relative_errors(ae, feats, tens, rows)))
    assert after < before
    assert hist.train_error == pytest.approx(after, rel=1e-6)


@needs_torch
def test_an_unknown_loss_is_refused(gauss_only):
    fs, table, geom, tens = gauss_only
    feats = np.zeros((table.n_mol, 3))
    ae = fieldae.FieldAutoencoder(geom, 3, code=2, n_pseudo=4, hidden=(8,))
    with pytest.raises(ValueError, match="relative"):
        fieldae.fit(ae, feats, tens, np.arange(table.n_mol), None, loss="l2")


# --------------------------------------------------------------------------- #
# The decoder: the box, the collapse, the pre-image
# --------------------------------------------------------------------------- #

@needs_torch
def test_the_pseudo_atoms_stay_inside_the_box(gauss_only):
    fs, table, geom, tens = gauss_only
    import torch

    box = np.array([4.0, 3.0, 2.0])
    dec = fieldae.PseudoAtomDecoder(code=2, n_pseudo=10, n_fields=geom.n_fields, hidden=(8,),
                                    box=box)
    code = torch.randn(5, 2) * 50.0            # far outside anything the encoder would give
    pos, w = dec(code)
    assert np.all(np.abs(pos.detach().numpy()) <= box[None, None, :] + 1e-6)
    assert w.shape == (5, 10, geom.n_fields)


@needs_torch
def test_the_repulsion_penalty_sees_a_collapse_and_nothing_else():
    import torch

    far = torch.tensor([[[0.0, 0, 0], [3.0, 0, 0], [0.0, 3, 0]]])
    close = torch.tensor([[[0.0, 0, 0], [0.1, 0, 0], [0.0, 3, 0]]])
    assert float(fieldae._repulsion(far, 0.75)) == 0.0
    assert float(fieldae._repulsion(close, 0.75)) > 0.0


def test_the_minimum_distance_finds_the_closest_pair():
    pos = np.array([[[0.0, 0, 0], [2.0, 0, 0], [2.0, 0.5, 0]]])
    assert fieldae.min_pseudo_distance(pos) == pytest.approx(0.5)


@needs_torch
def test_the_pre_image_is_a_field_the_program_can_read(gauss_only):
    fs, table, geom, tens = gauss_only
    rng = np.random.RandomState(0)
    feats = rng.randn(table.n_mol, 5)
    ae = fieldae.FieldAutoencoder(geom, 5, code=4, n_pseudo=7, hidden=(16,), box=tens.box(),
                                  weight_scale=tens.weight_scale(), seed=0)
    mf = ae.pre_image(np.zeros(4))
    assert isinstance(mf, MoleculeFields)
    assert mf.coords.shape == (7, 3)
    assert mf.weights.shape == (7, geom.n_fields)
    # and it goes through the ordinary kernel path without any conversion
    K = kernel_matrices(AtomTable.from_molecules([mf]), table,
                        [s for s in fs.specs if s.enabled], symmetric=False)
    assert K.shape == (geom.n_fields, 1, table.n_mol)


@needs_torch
def test_sampling_without_spread_returns_the_mean_code(gauss_only):
    fs, table, geom, tens = gauss_only
    rng = np.random.RandomState(0)
    feats = rng.randn(table.n_mol, 5)
    ae = fieldae.FieldAutoencoder(geom, 5, code=3, n_pseudo=5, hidden=(16,), box=tens.box(),
                                  weight_scale=tens.weight_scale(), seed=0)
    ae.set_code_distribution(ae.codes(feats))
    got = ae.sample(3, seed=0, spread=0.0)
    mean = ae.pre_image(ae.code_mean_.detach().numpy())
    assert len(got) == 3
    for mf in got:
        assert np.allclose(mf.coords, mean.coords)
    spread = ae.sample(2, seed=0, spread=1.0)
    assert not np.allclose(spread[0].coords, mean.coords)


# --------------------------------------------------------------------------- #
# Drawing and exporting, which never enter the loss
# --------------------------------------------------------------------------- #

def test_the_drawn_field_is_the_sum_of_the_gaussians(gauss_only):
    """The display path against the formula written out, at a few points."""
    fs, table, geom, tens = gauss_only
    from cmfqsar.cofields import Grid

    mf = fieldae.pre_image_fields(np.array([[0.0, 0, 0], [1.5, 0, 0]]),
                                  np.array([[1.0] * geom.n_fields, [-0.5] * geom.n_fields]))
    grid = Grid(origin=np.array([-1.0, -1.0, 0.0]), spacing=0.5, shape=(5, 5, 1))
    vol = fieldae.decoded_volume(mf, geom, 0, grid)
    pts = grid.points()
    b = fieldae.field_width(geom, 0)
    d2 = ((pts[:, None, :] - mf.coords[None, :, :]) ** 2).sum(-1)
    want = (mf.weights[:, 0] * np.exp(-0.5 * b * d2)).sum(axis=1).reshape(grid.shape)
    assert np.allclose(vol, want, atol=1e-5)


def test_the_width_of_a_radius_scaled_field_follows_the_radius(built):
    fs, table, geom, tens = built
    f = geom.kinds.index("steric")
    assert fieldae.field_width(geom, f, radius=2.0) == pytest.approx(geom.alphas[f] / 4.0)
    g = 1 - f if geom.kinds[1 - f] == "gauss" else 2
    assert fieldae.field_width(geom, g, radius=2.0) == pytest.approx(geom.alphas[g])


def test_the_exported_cube_reads_back_as_the_same_field(tmp_path, gauss_only):
    fs, table, geom, tens = gauss_only
    from cmfqsar.cdft import read_cube

    mf = fieldae.pre_image_fields(np.array([[0.0, 0, 0], [1.2, 0.3, 0]]),
                                  np.array([[0.7] * geom.n_fields, [-0.4] * geom.n_fields]))
    from cmfqsar.cofields import Grid

    path = fieldae.export_decoded(str(tmp_path / "decoded.cube"), mf, geom, 0, spacing=0.6)
    cube = read_cube(path)
    assert list(cube.numbers) == [0, 0]            # pseudo-atoms are not elements
    assert np.allclose(cube.positions, mf.coords, atol=1e-4)
    grid = Grid(origin=cube.origin, spacing=float(cube.vectors[0][0]), shape=cube.values.shape)
    vol = fieldae.decoded_volume(mf, geom, 0, grid)
    # the reader converts a density from e/bohr^3 to e/A^3; a field is not a density, so the
    # values come back scaled by that one constant and nothing else
    from cmfqsar.cdft import BOHR

    assert np.allclose(np.asarray(cube.values, float) * BOHR ** 3, vol, atol=1e-6)


# --------------------------------------------------------------------------- #
# What the autoencoder is compared with
# --------------------------------------------------------------------------- #

def test_kernel_pca_error_is_the_one_class_score_of_the_program(built):
    """:func:`kpca_relative_errors` against ``ockpca``, which is the same distance with an offset."""
    from cmfqsar import oneclass

    fs, table, geom, tens = built
    K = fieldae.joint_kernel(table, table, fs, geom, symmetric=True)
    train = np.arange(6)
    test = np.array([6, 7])
    q = 3
    sc = oneclass.make("ockpca", components=q).fit(K[np.ix_(train, train)])
    decision = sc.decision(K[np.ix_(test, train)], tens.self_k[test])
    squared = sc.offset_ - decision                      # the squared distance from the subspace
    want = np.sqrt(np.maximum(squared, 0.0) / tens.self_k[test])
    got = fieldae.kpca_relative_errors(K[np.ix_(train, train)], K[np.ix_(test, train)],
                                       tens.self_k[test], q)
    assert np.allclose(got, want, rtol=1e-8, atol=1e-10)


def test_kernel_pca_cannot_get_worse_with_more_components(built):
    fs, table, geom, tens = built
    K = fieldae.joint_kernel(table, table, fs, geom, symmetric=True)
    train = np.arange(6)
    test = np.array([6, 7])
    errs = [float(np.mean(fieldae.kpca_relative_errors(
        K[np.ix_(train, train)], K[np.ix_(test, train)], tens.self_k[test], q)))
        for q in (1, 2, 3, 4, 5)]
    assert all(b <= a + 1e-9 for a, b in zip(errs[:-1], errs[1:]))


def test_the_whole_training_set_is_reconstructed_by_as_many_components(built):
    """With n - 1 components the training molecules themselves are reconstructed exactly."""
    fs, table, geom, tens = built
    K = fieldae.joint_kernel(table, table, fs, geom, symmetric=True)
    train = np.arange(table.n_mol)
    err = fieldae.kpca_relative_errors(K, K, tens.self_k[train], table.n_mol - 1)
    assert float(np.max(err)) < 1e-6
