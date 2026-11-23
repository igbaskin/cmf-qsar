"""Atom amplitudes from a network on the atoms, plan item ND14a (cmfqsar/amplitudes.py)."""
import copy
import os

import numpy as np
import pytest

from cmfqsar import AtomTable, FieldSet, build_fields, load_dataset
from cmfqsar import amplitudes as A
from cmfqsar import learnable as L
from cmfqsar.kernels import expansion_overlap, kernel_matrices, normalize_kernels, self_kernels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_torch = pytest.mark.skipif(not A.available(), reason="PyTorch is not installed")
ALPHAS = np.array([0.3, 0.7, 1.2])
SMALL = dict(hidden=5, rounds=2, penalty=0.1, device="cpu")


@pytest.fixture(scope="module")
def series():
    """Thirty molecules of the ACE series with three fields."""
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", "ACE.sdf"))
    fs = FieldSet.create(physchem=["electrostatic", "steric", "hba"], alpha=0.5,
                         charge_model="file")
    for spec, a in zip(fs.specs, ALPHAS):
        spec.alpha = float(a)
    mols, names = ds.mols[:30], ds.names[:30]
    table = AtomTable.from_molecules([build_fields(m, fs, n) for m, n in zip(mols, names)])
    y = np.asarray(ds.activity, float)[:30]
    order = np.random.RandomState(0).permutation(30)
    return {"fs": fs, "table": table, "y": y, "graph": A.pack_graph(table, fs.specs),
            "train": np.sort(order[:20]), "valid": np.sort(order[20:25]),
            "test": np.sort(order[25:])}


def test_the_graph_holds_the_packed_atoms_and_the_input_of_every_atom(series):
    table, G = series["table"], series["graph"]
    P = L.pack(table, series["fs"].specs)
    W = np.asarray(table.weights)
    for f in range(3):
        assert np.array_equal(G["w"][f], P["w"][f]) and np.array_equal(G["xyz"][f], P["xyz"][f])
        atom = G["atom"][f]
        assert atom.shape == G["w"][f].shape
        assert np.array_equal(atom >= 0, G["w"][f] != 0.0)
        assert np.array_equal(W[atom[atom >= 0], f], G["w"][f][atom >= 0])
        # an atom index belongs to the molecule of its row
        mol = np.searchsorted(G["offsets"], atom[atom >= 0], side="right") - 1
        assert np.array_equal(mol, np.nonzero(atom >= 0)[0])
        assert np.isclose(G["scale"][f], np.sqrt(np.mean(W[W[:, f] != 0, f] ** 2)))
    T = len(G["types"])
    x = G["x"]
    assert x.shape == (len(W), T + 3 + 2) and len(G["columns_x"]) == x.shape[1]
    assert np.all(x[:, :T].sum(axis=1) == 1.0)                 # one atom type per atom
    assert np.allclose(x[:, T:T + 3], W / G["scale"])
    # the two last columns count the bonded atoms; every bond is listed in both directions
    mf = table.molecules[0]
    degree = np.bincount(np.asarray(mf.bonds).ravel(), minlength=mf.n_atoms)
    assert np.allclose(4.0 * x[:mf.n_atoms, T + 3:].sum(axis=1), degree)
    e = G["edges"]
    assert e.shape[1] == 2 * sum(len(m.bonds) for m in table.molecules)
    assert {(a, b) for a, b in e.T} == {(b, a) for a, b in e.T}


def test_the_graph_of_other_molecules_uses_the_columns_of_the_training_graph(series):
    table, fs, G = series["table"], series["fs"], series["graph"]
    few = table.subset([3, 7])
    H = A.pack_graph(few, fs.specs, like=G)
    assert H["types"] == G["types"] and np.array_equal(H["scale"], G["scale"])
    lo, hi = table.offsets[3], table.offsets[4]
    assert np.array_equal(H["x"][:hi - lo], G["x"][lo:hi])
    # an atom type that the training graph does not have gets no column
    less = dict(G, types=[t for t in G["types"] if t != "H"])
    H = A.pack_graph(few, fs.specs, like=less)
    hydrogen = np.array([s == "H" for m in few.molecules for s in m.symbols])
    assert np.all(H["x"][hydrogen, :len(less["types"])].sum(axis=1) == 0.0)
    assert np.all(H["x"][~hydrogen, :len(less["types"])].sum(axis=1) == 1.0)
    with pytest.raises(ValueError, match="number of fields"):
        A.pack_graph(few, fs.specs[:2], like=G)


def test_a_graph_survives_a_file(series, tmp_path):
    path = str(tmp_path / "graph.npz")
    A.save_graph(path, series["graph"], y=series["y"])
    G, extra = A.load_graph(path)
    assert np.array_equal(extra["y"], series["y"]) and list(extra) == ["y"]
    for key in ("x", "edges", "offsets", "scale"):
        assert np.array_equal(G[key], series["graph"][key])
    assert G["types"] == series["graph"]["types"]
    assert G["columns_x"] == series["graph"]["columns_x"]
    for f in range(3):
        assert np.array_equal(G["atom"][f], series["graph"]["atom"][f])
        assert np.array_equal(G["w"][f], series["graph"]["w"][f])


@needs_torch
def test_the_two_passes_agree_with_automatic_differentiation(series):
    import torch
    G = series["graph"]
    rng = np.random.RandomState(1)
    for f, kind in enumerate(G["kinds"]):
        rows, land = np.arange(8), np.array([1, 4, 6])
        t = lambda a: torch.as_tensor(np.asarray(a, float))             # noqa: E731
        xp, cp = t(G["xyz"][f][rows]), t(G["c"][f][rows])
        ap = t(G["w"][f][rows] + 0.1 * rng.normal(size=G["w"][f][rows].shape)).requires_grad_()
        aq = t(G["w"][f][land] * 1.3).requires_grad_()
        # the kernels with the whole array of atom pairs kept, as the reference
        b, n_a = ap.shape
        E = A._pair_overlap(xp, cp, xp[land], cp[land], float(ALPHAS[f]), kind)
        K_ref = torch.einsum("i,izj,zj->iz", ap.reshape(-1), E, aq).view(b, n_a, -1).sum(1)
        Es = A._self_overlap(xp, cp, float(ALPHAS[f]), kind)
        d_ref = torch.einsum("bi,bij,bj->b", ap, Es, ap)
        K, d = A._field_pass(xp, ap.detach(), cp, xp[land], aq.detach(), cp[land],
                             float(ALPHAS[f]), kind, budget=2000.0)
        assert torch.allclose(K, K_ref.detach(), rtol=1e-10, atol=1e-12)
        assert torch.allclose(d, d_ref.detach(), rtol=1e-10)
        GK, Gd = t(rng.normal(size=K.shape)), t(rng.normal(size=d.shape))
        ((GK * K_ref).sum() + (Gd * d_ref).sum()).backward()
        gp, gq = A._field_pass(xp, ap.detach(), cp, xp[land], aq.detach(), cp[land],
                               float(ALPHAS[f]), kind, 2000.0, GK, Gd)
        assert torch.allclose(gp, ap.grad, rtol=1e-9, atol=1e-11)
        assert torch.allclose(gq, aq.grad, rtol=1e-9, atol=1e-11)


@needs_torch
def test_with_no_network_the_model_is_the_model_of_the_learnable_layer(series):
    G, y, tr, va = series["graph"], series["y"], series["train"], series["valid"]
    P = L.pack(series["table"], series["fs"].specs)
    for normalize, layer in (("cosine", None), ("mean", None), ("cosine", "gauss")):
        ref = L.train(P, y, tr, tr, valid=va, alpha=ALPHAS, lam=0.3, learn=(),
                      normalize=normalize, layer=layer, device="cpu")
        res = A.train(G, y, tr, tr, valid=va, fields=[], alpha=ALPHAS, lam=0.3,
                      learn_lam=False, normalize=normalize, layer=layer, device="cpu")
        assert np.allclose(res.pred, ref.pred, atol=1e-9)
        assert res.learned == [] and res.state == {} and len(res.history) == 1
        assert res.start["stop"] == "nothing trained" and res.lam == 0.3
    # the ridge alone can still be trained, as in the learnable layer
    ref = L.train(P, y, tr, tr, valid=va, alpha=ALPHAS, lam=0.3, learn=("lam",), steps=5,
                  device="cpu", select="cv")
    res = A.train(G, y, tr, tr, valid=va, fields=[], alpha=ALPHAS, lam=0.3, steps=5,
                  device="cpu", select="cv")
    assert res.lam != 0.3 and min(h["cv"] for h in res.history) < res.start["cv"]
    assert min(h["cv"] for h in res.history) == pytest.approx(
        min(h["cv"] for h in ref.history), rel=1e-3)


@needs_torch
def test_the_start_of_a_training_is_the_model_with_the_fixed_amplitudes(series):
    G, y, tr = series["graph"], series["y"], series["train"]
    fixed = A.train(G, y, tr, tr, fields=[], alpha=ALPHAS, lam=0.3, learn_lam=False,
                    device="cpu")
    res = A.train(G, y, tr, tr, alpha=ALPHAS, lam=0.3, steps=0, **SMALL)
    assert res.learned == G["names"] and res.start["weights"] > 0
    assert np.allclose(res.pred, fixed.pred, atol=1e-10)
    assert all(v == 0.0 for v in res.correction.values())
    for f, a in enumerate(A.amplitudes(res, G)):
        assert np.array_equal(a, G["w"][f])


@needs_torch
@pytest.mark.parametrize("normalize,layer", [("cosine", None), ("mean", None),
                                             ("cosine", "gauss")])
def test_the_gradient_agrees_with_finite_differences(series, monkeypatch, normalize, layer):
    import scipy.optimize
    G, y, tr = series["graph"], series["y"], series["train"]
    seen = {}

    def spy(fun, x0, **kw):
        seen["fun"], seen["n"] = fun, len(x0)
        return scipy.optimize.OptimizeResult(message="spy")
    monkeypatch.setattr(scipy.optimize, "minimize", spy)
    A.train(G, y, tr, tr[:12], valid=series["valid"], alpha=ALPHAS, lam=0.3,
            normalize=normalize, layer=layer, steps=1, **SMALL)
    fun = seen["fun"]
    rng = np.random.RandomState(2)
    z = 0.5 * rng.normal(size=seen["n"])              # away from the start: every layer moves
    _value, grad = fun(z)
    assert np.linalg.norm(grad) > 0
    for _ in range(3):
        d = rng.normal(size=len(z))
        d /= np.linalg.norm(d)
        fd = (fun(z + 1e-3 * d)[0] - fun(z - 1e-3 * d)[0]) / 2e-3
        assert fd == pytest.approx(float(grad @ d), rel=1e-4, abs=1e-8)


@needs_torch
def test_training_lowers_the_objective_and_only_corrects_the_named_fields(series):
    G, y, tr, va = series["graph"], series["y"], series["train"], series["valid"]
    res = A.train(G, y, tr, tr, valid=va, fields=["electrostatic", "hba"], alpha=ALPHAS,
                  lam=0.3, steps=6, select="cv", keep_criterion=True, **SMALL)
    assert res.learned == ["electrostatic", "hba"] and res.by_criterion is res
    objective = [h["objective"] for h in res.history]
    assert min(objective) < objective[0] - 1e-3
    assert res.evaluations == len(res.history) and res.pass_seconds > 0
    assert set(res.correction) == {"electrostatic", "hba"}
    assert all(v > 0 for v in res.correction.values())
    amps = A.amplitudes(res, G)
    assert np.array_equal(amps[1], G["w"][1])                    # the steric field is fixed
    for f in (0, 2):
        changed = amps[f] != G["w"][f]
        assert changed.any() and not changed[G["w"][f] == 0.0].any()   # only atoms of the field
    # early stopping keeps the evaluation with the lowest validation error
    res = A.train(G, y, tr, tr, valid=va, alpha=ALPHAS, lam=0.3, steps=6, **SMALL)
    assert res.history[0]["valid_rmse"] >= min(h["valid_rmse"] for h in res.history)
    with pytest.raises(ValueError, match="no field 'shape'"):
        A.train(G, y, tr, tr, fields=["shape"], **SMALL)
    with pytest.raises(ValueError, match="training part"):
        A.train(G, y, tr, series["test"], **SMALL)


@needs_torch
def test_the_trained_amplitudes_are_weights_that_the_program_can_use(series):
    table, fs, G, y = series["table"], series["fs"], series["graph"], series["y"]
    tr = series["train"]
    res = A.train(G, y, tr, tr[:15], alpha=ALPHAS, lam=0.3, steps=4, select="cv", **SMALL)
    W = A.atom_weights(res, G, table.weights)
    assert W.shape == table.weights.shape and not np.array_equal(W, table.weights)
    assert np.array_equal(W == 0.0, np.asarray(table.weights) == 0.0)
    # a table with these weights and the kernels of the program give the predictions back
    mols = []
    for i, mf in enumerate(table.molecules):
        m = copy.copy(mf)
        m.weights = W[table.slice(i)]
        mols.append(m)
    T2 = AtomTable.from_molecules(mols)
    land = res.landmarks
    K = np.asarray(kernel_matrices(T2, T2.subset(land), fs.specs, symmetric=False))
    d = np.asarray(self_kernels(T2, fs.specs))
    Kn = normalize_kernels(K, d, d[:, land]).sum(axis=0)
    assert np.allclose(Kn @ res.coef + res.bias, res.pred, atol=1e-8)
    assert np.allclose(res.dz, d[:, land], rtol=1e-9)
    # the same molecules as another table, with the columns of the training graph
    rows = series["test"]
    H = A.pack_graph(table.subset(rows), fs.specs, like=G)
    assert np.allclose(A.predict(res, H, device="cpu"), res.pred[rows], atol=1e-9)
    assert np.allclose(A.predict(res, G, rows, device="cpu"), res.pred[rows], atol=1e-12)


@needs_torch
def test_the_cofield_of_a_trained_model_gives_its_prediction_back(series):
    table, G, y, tr = series["table"], series["graph"], series["y"], series["train"]
    res = A.train(G, y, tr, tr, alpha=ALPHAS, lam=0.3, steps=4, normalize="mean", select="cv",
                  **SMALL)
    W = A.atom_weights(res, G, table.weights)
    for i in series["test"][:3]:
        mol = table.molecules[int(i)]
        total = res.bias
        for f in range(3):
            centres, coeffs, widths = A.cofield_gaussians(res, f)
            b = (np.full(mol.n_atoms, res.alpha[f]) if res.kinds[f] == "gauss"
                 else res.alpha[f] / np.asarray(mol.radii) ** 2)
            total += expansion_overlap(centres, coeffs, widths, mol.coords,
                                       W[table.slice(int(i)), f], b)
        assert abs(total - res.pred[int(i)]) < 1e-8 * max(1.0, abs(total))
    layer = A.train(G, y, tr, tr, alpha=ALPHAS, lam=0.3, layer="gauss", steps=2, **SMALL)
    assert layer.gamma == 1.0 and np.all(np.isfinite(layer.pred))
    with pytest.raises(ValueError, match="co-field"):
        A.cofield_gaussians(layer, 0)


@needs_torch
def test_without_rounds_an_atom_does_not_read_its_neighbours(series):
    G, y, tr = series["graph"], series["y"], series["train"]
    cut = dict(G, edges=np.zeros((2, 0), np.int64))
    alone = A.train(G, y, tr, tr, alpha=ALPHAS, lam=0.3, steps=3, select="cv", hidden=5,
                    rounds=0, device="cpu")
    for a, b in zip(A.amplitudes(alone, G), A.amplitudes(alone, cut)):
        assert np.array_equal(a, b)
    near = A.train(G, y, tr, tr, alpha=ALPHAS, lam=0.3, steps=3, select="cv", **SMALL)
    assert any(not np.allclose(a, b) for a, b in zip(A.amplitudes(near, G),
                                                     A.amplitudes(near, cut)))
    # two atoms with the same input get the same correction when the atom is read alone
    f = 0
    atom = G["atom"][f]
    x = G["x"][atom[atom >= 0]]
    amp = A.amplitudes(alone, G)[f][atom >= 0]
    _rows, first, inverse = np.unique(x, axis=0, return_index=True, return_inverse=True)
    assert len(first) < len(x) and np.allclose(amp, amp[first][inverse.ravel()], atol=1e-12)
