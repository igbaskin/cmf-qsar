"""The learnable CMF layer of plan item ND2 (cmfqsar/learnable.py)."""
import os

import numpy as np
import pytest

from cmfqsar import AtomTable, FieldSet, build_fields, load_dataset
from cmfqsar import learnable as L
from cmfqsar.kernels import expansion_overlap, kernel_matrices, normalize_kernels, self_kernels
from cmfqsar.model import krr_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_torch = pytest.mark.skipif(not L.available(), reason="PyTorch is not installed")
ALPHAS = np.array([0.3, 0.7, 1.2])


@pytest.fixture(scope="module")
def series():
    """Forty molecules of the ACE series with three fields, each with its own attenuation."""
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", "ACE.sdf"))
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hba"], alpha=0.5,
                         charge_model="file")
    for spec, a in zip(fs.specs, ALPHAS):
        spec.alpha = float(a)
    mols, names = ds.mols[:40], ds.names[:40]
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    y = np.asarray(ds.activity, float)[:40]
    rng = np.random.RandomState(0)
    order = rng.permutation(40)
    return {"fs": fs, "table": table, "y": y, "packed": L.pack(table, fs.specs),
            "train": np.sort(order[:28]), "valid": np.sort(order[28:34]),
            "test": np.sort(order[34:])}


def test_the_packed_atoms_hold_only_the_atoms_that_carry_weight(series):
    table, P = series["table"], series["packed"]
    assert P["names"] == ["electrostatic", "steric", "hba"]
    assert P["kinds"] == ["gauss", "steric", "gauss"]
    W = np.asarray(table.weights)
    for f in range(3):
        per_molecule = [int(np.count_nonzero(W[table.slice(i), f])) for i in range(table.n_mol)]
        assert P["w"][f].shape == (table.n_mol, max(max(per_molecule), 1))
        assert [int(np.count_nonzero(row)) for row in P["w"][f]] == per_molecule
        assert np.isclose(P["w"][f].sum(), W[:, f].sum())
    # the acceptor field uses far fewer atoms than the steric field
    assert P["w"][2].shape[1] < P["w"][1].shape[1]
    assert np.all(P["c"][0] == 1.0) and np.any(P["c"][1] != 1.0)


def test_a_field_kind_without_a_kernel_here_is_refused(series):
    class Spec:
        name, kind = "x", "efield"
    with pytest.raises(ValueError, match="efield"):
        L.pack(series["table"], [Spec()])


def test_packed_atoms_survive_a_file(series, tmp_path):
    path = str(tmp_path / "atoms.npz")
    L.save_packed(path, series["packed"], y=series["y"])
    P, extra = L.load_packed(path)
    assert P["names"] == series["packed"]["names"] and P["n"] == 40
    assert np.array_equal(extra["y"], series["y"])
    for f in range(3):
        assert np.array_equal(P["xyz"][f], series["packed"]["xyz"][f])
        assert np.array_equal(P["w"][f], series["packed"]["w"][f])


@needs_torch
def test_the_kernels_are_those_of_the_program(series):
    table, fs, P = series["table"], series["fs"], series["packed"]
    rows, land = np.arange(40), np.arange(0, 40, 3)
    kern = L.kernel_pass(P, rows, land, ALPHAS, device="cpu")
    ref = np.asarray(kernel_matrices(table, table.subset(land), fs.specs, symmetric=False))
    assert np.allclose(kern.K.numpy(), ref, rtol=1e-10, atol=1e-12 * np.abs(ref).max())
    d = np.asarray(self_kernels(table, fs.specs))
    assert np.allclose(kern.dx.numpy(), d, rtol=1e-10)
    assert np.allclose(kern.dz.numpy(), d[:, land], rtol=1e-10)
    assert np.allclose(kern.Kzz.numpy(), ref[:, land, :], rtol=1e-9,
                       atol=1e-12 * np.abs(ref).max())


@needs_torch
def test_the_derivative_in_log_alpha_agrees_with_a_finite_difference(series):
    P = series["packed"]
    rows, land = np.arange(40), np.arange(0, 40, 4)
    kern = L.kernel_pass(P, rows, land, ALPHAS, device="cpu")
    eps = 1e-4
    up = L.kernel_pass(P, rows, land, ALPHAS * np.exp(eps), derivative=False, device="cpu")
    dn = L.kernel_pass(P, rows, land, ALPHAS * np.exp(-eps), derivative=False, device="cpu")
    for got, a, b in ((kern.dK, up.K, dn.K), (kern.dKzz, up.Kzz, dn.Kzz),
                      (kern.ddx, up.dx, dn.dx), (kern.ddz, up.dz, dn.dz)):
        num = ((a - b) / (2 * eps)).numpy()
        assert np.abs(got.numpy() - num).max() < 1e-6 * np.abs(num).max()
    assert up.dK is None and up.ddx is None


@needs_torch
def test_with_nothing_trained_the_model_is_the_kernel_ridge_of_the_program(series):
    table, fs, P, y = series["table"], series["fs"], series["packed"], series["y"]
    tr = series["train"]
    res = L.train(P, y, tr, tr, valid=series["valid"], alpha=ALPHAS, lam=0.3, learn=(),
                  device="cpu")
    K = np.asarray(kernel_matrices(table, table, fs.specs))
    d = np.asarray(self_kernels(table, fs.specs))
    Kn = normalize_kernels(K, d, d).sum(axis=0)
    a, b = krr_fit(Kn[np.ix_(tr, tr)], y[tr], 0.3)
    assert np.allclose(res.pred, Kn[:, tr] @ a + b, atol=2e-4)
    assert res.passes == 1 and len(res.history) == 1 and res.learned == ()
    # the other convention: every field kernel divided by the mean self-kernel of the landmarks
    res = L.train(P, y, tr, tr, alpha=ALPHAS, lam=0.3, learn=(), normalize="mean", device="cpu")
    Km = sum(K[f] / d[f][tr].mean() for f in range(3))
    a, b = krr_fit(Km[np.ix_(tr, tr)], y[tr], 0.3)
    assert np.allclose(res.pred, Km[:, tr] @ a + b, atol=2e-4)


@needs_torch
def test_the_gradient_of_the_criterion_agrees_with_finite_differences(series):
    P, y, tr = series["packed"], series["y"], series["train"]

    def start(alpha, h, lam):
        return L.train(P, y, tr, tr, alpha=alpha, h=h, lam=lam, steps=0, device="cpu",
                       learn=("alpha", "weights", "lam"), predict=[]).start

    h0 = np.array([0.8, 1.3, 0.9])
    s = start(ALPHAS, h0, 0.3)
    assert s["layout"] == [["alpha", 3], ["weights", 3], ["lam", 1]]
    g = np.asarray(s["gradient"])
    eps = 1e-4
    for f in range(3):
        up, dn = ALPHAS.copy(), ALPHAS.copy()
        up[f] *= np.exp(eps)
        dn[f] *= np.exp(-eps)
        num = (start(up, h0, 0.3)["cv"] - start(dn, h0, 0.3)["cv"]) / (2 * eps)
        assert abs(g[f] - num) < 1e-5 + 1e-3 * abs(num)
        # a weight logit: h = T softmax(u), with the sum T held
        u = np.log(h0)
        hs = []
        for sign in (1, -1):
            v = u.copy()
            v[f] += sign * eps
            hs.append(h0.sum() * np.exp(v) / np.exp(v).sum())
        num = (start(ALPHAS, hs[0], 0.3)["cv"] - start(ALPHAS, hs[1], 0.3)["cv"]) / (2 * eps)
        assert abs(g[3 + f] - num) < 1e-5 + 1e-3 * abs(num)
    num = (start(ALPHAS, h0, 0.3 * np.exp(eps))["cv"]
           - start(ALPHAS, h0, 0.3 * np.exp(-eps))["cv"]) / (2 * eps)
    assert abs(g[6] - num) < 1e-5 + 1e-3 * abs(num)


@needs_torch
def test_training_lowers_the_criterion_and_keeps_what_is_not_trained(series):
    P, y, tr, va = series["packed"], series["y"], series["train"], series["valid"]
    res = L.train(P, y, tr, tr, valid=va, alpha=0.5, lam=0.3, steps=6, device="cpu",
                  learn=("alpha", "weights", "lam"), keep_criterion=True)
    low = res.by_criterion
    cvs = [h["cv"] for h in res.history]
    assert min(cvs) < res.start["cv"] - 1e-3
    assert np.isclose(low.h.sum(), 3.0) and np.isclose(res.h.sum(), 3.0)
    assert not np.allclose(low.alpha, 0.5) and low.lam != 0.3
    # early stopping keeps the evaluation with the lowest validation error
    best = int(np.nanargmin([h["valid_rmse"] for h in res.history]))
    assert np.allclose(res.alpha, res.history[best]["alpha"])
    assert np.allclose(low.alpha, res.history[int(np.argmin(cvs))]["alpha"])
    assert res.passes >= 2 and np.all(np.isfinite(res.pred))
    only = L.train(P, y, tr, tr, valid=va, alpha=0.5, lam=0.3, steps=4, device="cpu",
                   learn=("weights",))
    assert np.allclose(only.alpha, 0.5) and only.lam == 0.3 and only.passes == 1
    with pytest.raises(ValueError, match="layer"):
        L.train(P, y, tr, tr, alpha=0.5, learn=("gamma",), device="cpu")


@needs_torch
def test_the_cofield_of_a_trained_model_gives_its_prediction_back(series):
    table, fs, P, y = series["table"], series["fs"], series["packed"], series["y"]
    tr = series["train"]
    res = L.train(P, y, tr, tr, alpha=0.5, lam=0.3, steps=3, normalize="mean", device="cpu",
                  learn=("alpha", "weights", "lam"), select="cv")
    Z = table.subset(tr)
    for i in series["test"][:3]:
        mol = table.molecules[int(i)]
        total = res.bias
        for f in range(3):
            centres, coeffs, widths = L.cofield_gaussians(res, Z, f)
            b = (np.full(mol.n_atoms, res.alpha[f]) if res.kinds[f] == "gauss"
                 else res.alpha[f] / np.asarray(mol.radii) ** 2)
            total += expansion_overlap(centres, coeffs, widths, mol.coords,
                                       mol.weights[:, f], b)
        assert abs(total - res.pred[int(i)]) < 1e-6 * max(1.0, abs(total))
    learned = L.fieldset_of(res, fs)
    assert [s.alpha for s in learned.specs] == pytest.approx(list(res.alpha))
    assert [s.h for s in learned.specs] == pytest.approx(list(res.h))
    assert [s.alpha for s in fs.specs] == pytest.approx(list(ALPHAS))    # the original is kept
    with pytest.raises(ValueError, match="landmarks"):
        L.cofield_gaussians(res, table, 0)


@needs_torch
def test_the_gaussian_layer_trains_its_width_and_has_no_cofield(series):
    table, P, y, tr = series["table"], series["packed"], series["y"], series["train"]
    res = L.train(P, y, tr, tr, valid=series["valid"], alpha=0.5, lam=0.1, gamma=1.0,
                  layer="gauss", steps=5, device="cpu", learn=("lam", "gamma"), select="cv")
    assert res.gamma is not None and res.gamma != 1.0 and res.med > 0
    assert min(h["cv"] for h in res.history) <= res.start["cv"]
    assert np.all(np.isfinite(res.pred))
    with pytest.raises(ValueError, match="co-field"):
        L.cofield_gaussians(res, table.subset(tr), 0)


@needs_torch
def test_a_network_on_the_layer_trains_with_and_without_the_fields(series):
    P, y, tr, va = series["packed"], series["y"], series["train"], series["valid"]
    fixed = L.train_network(P, y, tr, va, tr, alpha=0.5, learn=(), hidden=(8,), epochs=30,
                            batch=14, patience=30, seed=0, device="cpu")
    assert fixed.passes == 1 and np.allclose(fixed.alpha, 0.5)
    assert np.all(np.isfinite(fixed.pred)) and fixed.learned == ()
    fit = np.sqrt(np.mean((fixed.pred[tr] - y[tr]) ** 2))
    assert fit < np.std(y[tr])
    extra = np.random.RandomState(1).normal(size=(40, 2))
    both = L.train_network(P, y, tr, va, tr, alpha=0.5, extra=extra, hidden=(8,), epochs=20,
                           batch=14, lr_fields=0.05, trust=0.05, patience=20, seed=0,
                           learn=("alpha", "weights"), device="cpu")
    assert both.passes > 2                       # the kernels were recomputed as alpha moved
    assert {h["phase"] for h in both.history} == {"joint", "network alone"}
    assert np.isclose(both.h.sum(), 3.0) and np.all(np.isfinite(both.pred))
    assert np.all(both.alpha >= L.ALPHA_BOUNDS[0]) and np.all(both.alpha <= L.ALPHA_BOUNDS[1])


@needs_torch
def test_the_command_line_trains_the_fields_and_writes_the_cofield(tmp_path, capsys):
    from cmfqsar.cli import main
    gpb = os.path.join(ROOT, "examples", "sutherland", "GPB.sdf")
    prefix = str(tmp_path / "map")
    common = ["lowrank", gpb, "--landmarks", "molecules:30", "--alpha", "0.5", "--charges",
              "file", "--fields", "steric,hba", "--train-steps", "3"]
    main(common + ["--train", "widths,weights", "--train-holdout", "0", "--cofield", prefix])
    out = capsys.readouterr().out
    assert "trained weights and widths by gradient" in out
    assert "no molecules were held out" in out
    for name in ("steric", "hba"):
        path = f"{prefix}_{name}.cube"
        assert os.path.getsize(path) > 1000
        with open(path) as fh:
            assert fh.readline().strip() == f"CMF co-field of the field {name}"
    # the co-field is written at the trained attenuation, which is no longer the start
    line = next(ln for ln in out.splitlines() if ln.strip().startswith("co-field of steric"))
    assert "attenuation 0.5)" not in line
    main(common + ["--train", "weights"])
    assert "early stopping on 10 held-out molecules" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="molecules:N"):
        main(["lowrank", gpb, "--landmarks", "grid:3.0", "--charges", "file", "--train",
              "widths"])
    with pytest.raises(SystemExit, match="widths, weights"):
        main(["lowrank", gpb, "--landmarks", "molecules:30", "--charges", "file", "--train",
              "ridge"])
