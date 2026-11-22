"""The co-field as an atom feature and the atom network of plan item ND14c (cmfqsar/device.py)."""
import os

import numpy as np
import pytest

from cmfqsar import AtomTable, FieldSet, build_fields, load_dataset
from cmfqsar import device as D
from cmfqsar import learnable as L
from cmfqsar.cofields import evaluate_gaussians, probe_basis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
needs_torch = pytest.mark.skipif(not D.available(), reason="PyTorch is not installed")
ALPHAS = np.array([0.3, 0.7, 1.2])
LAM = 0.05


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
    order = np.random.RandomState(0).permutation(40)
    # every atom of every molecule, padded: the places where the co-field is read
    counts = [table.slice(i).stop - table.slice(i).start for i in range(40)]
    A = max(counts)
    points, radii, mask = np.zeros((40, A, 3)), np.ones((40, A)), np.zeros((40, A), bool)
    weights = np.zeros((40, A, 3))
    for i in range(40):
        sl = table.slice(i)
        points[i, :counts[i]] = np.asarray(table.coords)[sl]
        radii[i, :counts[i]] = np.asarray(table.radii)[sl]
        weights[i, :counts[i]] = np.asarray(table.weights)[sl]
        mask[i, :counts[i]] = True
    return {"fs": fs, "table": table, "y": y, "packed": L.pack(table, fs.specs),
            "train": np.sort(order[:28]), "valid": np.sort(order[28:34]),
            "test": np.sort(order[34:]), "points": points, "radii": radii, "mask": mask,
            "weights": weights}


def models_of(series, normalize, **kw):
    tr = series["train"]
    return D.fold_models(series["packed"], series["y"], tr, tr, alpha=ALPHAS, lam=LAM,
                         normalize=normalize, folds=4, seed=3, device="cpu", **kw)


def read(series, models, rows, **kw):
    return D.atom_cofield(series["packed"], models, rows, series["points"][rows],
                          mask=series["mask"][rows], radii=series["radii"][rows], device="cpu",
                          **kw)


@needs_torch
def test_the_last_model_is_the_model_of_the_whole_training_part(series):
    tr = series["train"]
    for normalize in ("cosine", "mean"):
        M = models_of(series, normalize)
        ref = L.train(series["packed"], series["y"], tr, tr, alpha=ALPHAS, lam=LAM, learn=(),
                      normalize=normalize, device="cpu")
        assert M.coef.shape == (5, len(tr)) and M.folds == 4
        assert np.allclose(M.coef[-1], ref.coef, rtol=1e-8, atol=1e-10)
        assert np.isclose(M.bias[-1], ref.bias)
        assert np.allclose(M.fit, ref.pred[tr], atol=1e-8)
        assert np.allclose(M.norms, ref.norms)
        # every training row is in exactly one fold, and the folds have nearly equal sizes
        sizes = np.bincount(M.fold_of, minlength=4)
        assert sizes.sum() == len(tr) and sizes.max() - sizes.min() <= 1


@needs_torch
def test_the_model_of_a_fold_does_not_see_the_activities_of_its_fold(series):
    M = models_of(series, "cosine")
    y2 = series["y"].copy()
    in_fold = series["train"][M.fold_of == 1]
    y2[in_fold] += 7.0
    s2 = dict(series, y=y2)
    M2 = models_of(s2, "cosine")
    assert np.allclose(M2.coef[1], M.coef[1]) and np.isclose(M2.bias[1], M.bias[1])
    assert not np.allclose(M2.coef[0], M.coef[0])
    assert not np.allclose(M2.coef[-1], M.coef[-1])
    # the out-of-fold prediction of a row is the prediction of the model of its fold
    assert np.allclose(M2.cv[M.fold_of == 1], M.cv[M.fold_of == 1])


@needs_torch
def test_the_values_times_the_weights_sum_to_the_prediction(series):
    rows = np.arange(40)
    W = series["weights"]
    for normalize in ("mean", "cosine"):
        M = models_of(series, normalize)
        v = read(series, M, rows, molecule_norm=True)
        assert v.shape == W.shape
        total = (v * W).sum(axis=(1, 2)) + M.bias[-1]
        ref = L.train(series["packed"], series["y"], series["train"], series["train"],
                      alpha=ALPHAS, lam=LAM, learn=(), normalize=normalize, device="cpu").pred
        assert np.allclose(total, ref, atol=1e-8)
        # without the division by the molecule's own norm the two conventions differ
        plain = read(series, M, rows)
        assert np.allclose(plain, v) == (normalize == "mean")


@needs_torch
def test_the_values_are_the_atom_effect_of_the_cofield(series):
    """The same number from the co-field as Gaussians, convolved with the atom's Gaussian."""
    M = models_of(series, "mean")
    rows = series["test"]
    v = read(series, M, rows)
    trained = L.Trained(names=M.names, kinds=M.kinds, alpha=M.alpha, h=M.h, lam=M.lam,
                        gamma=None, med=1.0, normalize="mean", layer=None,
                        landmarks=M.landmarks, coef=M.coef[-1], bias=float(M.bias[-1]),
                        pred=np.zeros(0), learned=(), norms=M.norms)
    Z = series["table"].subset(M.landmarks)
    for f, kind in enumerate(M.kinds):
        centres, coeffs, widths = L.cofield_gaussians(trained, Z, f)
        for i, row in enumerate(rows):
            keep = series["mask"][row]
            pts = series["points"][row][keep]
            if kind == "steric":
                own = ALPHAS[f] / series["radii"][row][keep] ** 2
                ref = np.array([evaluate_gaussians(p[None], *probe_basis(centres, coeffs, widths,
                                                                         b), eps=1e-12)[0]
                                for p, b in zip(pts, own)])
            else:
                ref = evaluate_gaussians(pts, *probe_basis(centres, coeffs, widths, ALPHAS[f]),
                                         eps=1e-12)
            assert np.allclose(v[i][keep][:, f], ref, rtol=1e-6, atol=1e-9 * np.abs(ref).max())


@needs_torch
def test_a_training_molecule_reads_the_model_of_its_fold(series):
    M = models_of(series, "cosine")
    rows = np.arange(40)
    version = M.versions(rows)
    assert np.array_equal(version[series["train"]], M.fold_of)
    assert np.all(version[series["valid"]] == -1) and np.all(version[series["test"]] == -1)
    assert np.all(M.versions(rows, out_of_fold=False) == -1)
    full = read(series, M, rows)
    oof = read(series, M, rows, version=version)
    other = np.concatenate([series["valid"], series["test"]])
    assert np.array_equal(full[other], oof[other])
    assert not np.allclose(full[series["train"]], oof[series["train"]])
    # the out-of-fold values give the out-of-fold prediction back
    W = series["weights"]
    for k in range(M.folds):
        r = series["train"][M.fold_of == k]
        v = read(series, M, r, version=np.full(len(r), k), molecule_norm=True)
        assert np.allclose((v * W[r]).sum(axis=(1, 2)) + M.bias[k], M.cv[M.fold_of == k],
                           atol=1e-8)


@needs_torch
def test_padded_places_read_zero_and_wrong_input_is_refused(series):
    M = models_of(series, "cosine")
    rows = np.arange(6)
    v = read(series, M, rows)
    assert np.all(v[~series["mask"][rows]] == 0.0)
    with pytest.raises(ValueError, match="radii"):
        D.atom_cofield(series["packed"], M, rows, series["points"][rows], device="cpu")
    with pytest.raises(ValueError, match="points"):
        D.atom_cofield(series["packed"], M, rows, series["points"][:3],
                       radii=series["radii"][rows], device="cpu")
    with pytest.raises(ValueError, match="fold"):
        read(series, M, rows, version=np.full(6, 9))


def sums_of_atoms(seed=0, n=400, atoms=9, d=3):
    """Molecules whose activity is a sum over the atoms of a function of the atom features."""
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, atoms, d))
    mask = np.arange(atoms)[None, :] < rng.randint(3, atoms + 1, size=n)[:, None]
    per_atom = np.tanh(X[:, :, 0] * X[:, :, 1]) + 0.5 * X[:, :, 2] ** 2
    y = (per_atom * mask).sum(axis=1)
    return X, mask, y


@needs_torch
def test_the_device_learns_a_sum_over_the_atoms():
    X, mask, y = sums_of_atoms()
    tr, va, te = np.arange(0, 280), np.arange(280, 340), np.arange(340, 400)
    res = D.train_device(X, mask, y, tr, va, hidden=(32, 32), collectors=8, brain=(16,),
                         epochs=300, patience=40, batch=32, lr=3e-3, seed=1, device="cpu")
    e = res.pred[te] - y[te]
    assert 1.0 - np.sum(e ** 2) / np.sum((y[te] - y[te].mean()) ** 2) > 0.9
    assert res.collectors.shape == (400, 8) and res.epochs == len(res.history)
    assert np.isclose(res.valid_rmse, min(h["valid_rmse"] for h in res.history))
    # the order of the atoms and the content of the padded places do not matter
    perm = np.random.RandomState(2).permutation(X.shape[1])
    assert np.allclose(res.predict(X[te][:, perm], mask[te][:, perm]), res.pred[te])
    X2 = X.copy()
    X2[~mask] = 123.0
    assert np.allclose(res.predict(X2[te], mask[te]), res.pred[te])
    p, c = res.predict(X[te], mask[te], return_collectors=True)
    assert np.allclose(p, res.pred[te]) and np.allclose(c, res.collectors[te])


@needs_torch
def test_the_device_takes_columns_of_the_whole_molecule():
    X, mask, y = sums_of_atoms(seed=5, n=300)
    extra = np.random.RandomState(6).normal(size=(300, 2))
    y2 = y + 3.0 * extra[:, 0]
    tr, va, te = np.arange(0, 200), np.arange(200, 250), np.arange(250, 300)
    kw = dict(hidden=(32,), collectors=8, brain=(16,), epochs=200, patience=30, batch=32,
              lr=3e-3, seed=0, device="cpu")
    with_extra = D.train_device(X, mask, y2, tr, va, extra=extra, **kw)
    without = D.train_device(X, mask, y2, tr, va, **kw)

    def rmse(p):
        return float(np.sqrt(np.mean((p[te] - y2[te]) ** 2)))
    assert rmse(with_extra.pred) < 0.6 * rmse(without.pred)
    with pytest.raises(ValueError, match="extra"):
        with_extra.predict(X[te], mask[te])
    with pytest.raises(ValueError, match="validation"):
        D.train_device(X, mask, y2, tr, [], **kw)


# ----------------------------------------------------------------- the low-rank model, the CLI
def lowrank_model(series, lam=0.3):
    from cmfqsar import lowrank as LR
    table, fs, y = series["table"], series["fs"], series["y"]
    Z, _rows = LR.landmark_molecules(table, 20, seed=0)
    low = LR.build(Z, fs, normalize=True, kind="molecules")
    Phi = low.transform(table)
    return LR, low, Phi, table, y, lam


def test_the_fold_weights_of_the_low_rank_model_are_those_of_its_cross_validation(series):
    LR, low, Phi, table, y, lam = lowrank_model(series)
    W, B, which = D.lowrank_fold_weights(low, Phi, y, lam, folds=5, seed=0)
    pred, folds = LR.cross_validate(low, table, y, lam, folds=5, seed=0)
    assert np.array_equal(which, folds) and W.shape == (6, low.rank)
    mine = np.array([Phi[i] @ W[which[i]] + B[which[i]] for i in range(len(y))])
    assert np.allclose(mine, pred)
    w, b = LR.fit(LR.gram(low, table, y), lam)
    assert np.allclose(W[-1], w) and np.isclose(B[-1], b)


def test_the_low_rank_values_times_the_weights_sum_to_the_prediction(series):
    from cmfqsar.kernels import self_kernels
    LR, low, Phi, table, y, lam = lowrank_model(series)
    W, B, _which = D.lowrank_fold_weights(low, Phi, y, lam, folds=5, seed=0)
    v = D.lowrank_atom_cofield(low, W[-1], table)
    assert v.shape == (len(np.asarray(table.coords)), 3)
    d = np.asarray(self_kernels(table, low.specs), float)             # (fields, molecules)
    weights = np.asarray(table.weights, float)
    total = np.full(table.n_mol, B[-1])
    for i in range(table.n_mol):
        sl = table.slice(i)
        per_field = (v[sl] * weights[sl]).sum(axis=0)
        total[i] += float(np.sum(per_field / np.sqrt(np.maximum(d[:, i], 1e-300))))
    assert np.allclose(total, LR.predict(Phi, W[-1], B[-1]), atol=1e-6)


def test_the_command_line_writes_the_cofield_at_the_atoms(tmp_path, capsys):
    import csv

    from cmfqsar.cli import main
    gpb = os.path.join(ROOT, "examples", "sutherland", "GPB.sdf")
    common = ["lowrank", gpb, "--landmarks", "molecules:30", "--alpha", "0.5", "--charges",
              "file", "--fields", "steric,hba", "--lams", "1.0", "--folds", "5"]
    table = str(tmp_path / "atoms.csv")
    main(common + ["--atom-cofield", table])
    out = capsys.readouterr().out
    assert "read a model fitted without their fold (5 folds, ridge 1)" in out
    with open(table, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ds = load_dataset(gpb)
    assert len(rows) == sum(m.GetNumAtoms() for m in ds.mols)
    assert list(rows[0])[:7] == ["name", "source", "atom", "symbol", "x", "y", "z"]
    assert list(rows[0])[7:] == ["steric", "hba"] and {r["source"] for r in rows} == {"data"}
    assert rows[0]["atom"] == "1" and rows[0]["symbol"] == ds.mols[0].GetAtomWithIdx(0).GetSymbol()
    # the same molecules given as other molecules read the model of the whole set instead
    archive = str(tmp_path / "atoms.npz")
    main(common + ["--atom-cofield", archive, "--features-for", gpb])
    assert "the model of the whole data set" in capsys.readouterr().out
    z = np.load(archive)
    n = len(rows)
    assert z["values"].shape == (2 * n, 2) and list(z["fields"]) == ["steric", "hba"]
    assert list(z["source"]) == ["data"] * len(ds.mols) + ["other"] * len(ds.mols)
    assert int(z["counts"].sum()) == 2 * n and np.allclose(z["xyz"][:n], z["xyz"][n:])
    data = np.array([[float(r["steric"]), float(r["hba"])] for r in rows])
    assert np.allclose(z["values"][:n], data, rtol=1e-6, atol=1e-8)
    assert not np.allclose(z["values"][:n], z["values"][n:])
    with pytest.raises(SystemExit, match="landmarks in space"):
        main(["lowrank", gpb, "--landmarks", "fourier:64", "--charges", "file",
              "--atom-cofield", table])


# -------------------------------------------------------------------- the benchmark scripts
def benchmark(name):
    import importlib
    import sys
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    return importlib.import_module(name)


def test_the_atoms_are_put_in_the_order_of_the_smiles():
    from rdkit import Chem
    R = benchmark("run_device")
    ds = load_dataset(os.path.join(ROOT, "examples", "sutherland", "GPB.sdf"))
    for mol in ds.mols[:10]:
        smiles, order, degree = R.smiles_order(mol)
        back = Chem.MolFromSmiles(smiles)
        assert sorted(order) == [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
        assert [mol.GetAtomWithIdx(i).GetAtomicNum() for i in order] == \
            [a.GetAtomicNum() for a in back.GetAtoms()]
        assert degree == [a.GetDegree() for a in back.GetAtoms()]
    assert R.sensor_name("prop:q_xtb") == "COF_PROP_Q_XTB"
    assert R.sensor_name("cif:C.2r") == "COF_CIF_C_2R"


def test_the_sensor_file_is_what_the_device_runner_reads(tmp_path):
    # the device is a separate project: installed, named by NEURODEVICE_ROOT, or beside this one
    import sys
    for root in (os.environ.get("NEURODEVICE_ROOT"), os.path.join(os.path.dirname(ROOT), "bpz")):
        if root and os.path.isdir(os.path.join(root, "neurodevice")) and root not in sys.path:
            sys.path.append(root)
    pytest.importorskip("neurodevice")
    R = benchmark("run_device")
    F = benchmark("featuriser_device")
    from neurodevice.chem import sensors as S
    from neurodevice.chem.io import mol_from_smiles
    smiles = {"a": "CCO", "b": "c1ccccc1C"}
    mols = [mol_from_smiles(s, name=n) for n, s in smiles.items()]
    A = max(m.n_atoms for m in mols)
    pts = {"names": np.array(list(smiles)), "mask": np.zeros((2, A), bool),
           "degree": np.zeros((2, A), int)}
    values = np.zeros((2, A, 2))
    for i, m in enumerate(mols):
        pts["mask"][i, :m.n_atoms] = True
        pts["degree"][i, :m.n_atoms] = S.sensor_deg(m)
        values[i, :m.n_atoms] = np.arange(m.n_atoms)[:, None] + np.array([10.0 * i, 0.5])
    path = str(tmp_path / "sensors.npz")
    R.write_sensors(path, pts, ["T14_ONE", "T14_TWO"], values)
    try:
        new, check = F.register_atom_sensors(path)
        assert new == ["T14_ONE", "T14_TWO"]
        check(mols)
        got = S.atom_sensor_matrix(mols[1], ["T14_ONE", "T14_TWO", "DEG"])
        assert np.allclose(got[:, :2], values[1, :mols[1].n_atoms])
        with pytest.raises(SystemExit, match="already has"):
            F.register_atom_sensors(path)
        wrong = mol_from_smiles("OCC", name="b")
        with pytest.raises(SystemExit, match="order"):
            check([wrong])
    finally:
        for name in ("T14_ONE", "T14_TWO"):
            S.ATOM_SENSORS.pop(name, None)


def test_the_hop_protocol_never_divides_a_structure():
    H = benchmark("run_device_hops")
    rng = np.random.RandomState(0)
    molecule = np.sort(rng.randint(0, 60, size=400))
    outer = [np.flatnonzero(molecule % 4 == k) for k in range(4)]
    plans = H.protocol(molecule, 0, outer)
    for plan in plans:
        parts = [set(molecule[plan[k]]) for k in ("held", "valid", "fit")]
        assert not (parts[0] & parts[1]) and not (parts[0] & parts[2])
        assert not (parts[1] & parts[2])
        assert len(plan["held"]) + len(plan["valid"]) + len(plan["fit"]) == 400
        assert len(plan["inner"]) == len(plan["fit"])
        assert set(plan["inner"]) == set(range(H.INNER))
        for s in parts[2]:
            assert len(set(plan["inner"][molecule[plan["fit"]] == s])) == 1
    v = H.versions(400, plans)
    assert v.shape == (400, 4)
    assert np.all(v[plans[0]["held"], 0] == H.INNER)
    assert np.all(v[plans[0]["valid"], 0] == H.INNER)
    assert np.array_equal(v[plans[0]["fit"], 0], plans[0]["inner"])


def test_the_hop_models_of_a_fold_and_the_paired_interval():
    H = benchmark("run_device_hops")
    rng = np.random.RandomState(1)
    X = rng.normal(size=(120, 6))
    K = X @ X.T
    y = X @ rng.normal(size=6) + 0.1 * rng.normal(size=120)
    molecule = np.repeat(np.arange(30), 4)
    plan = H.protocol(molecule, 0, [np.arange(0, 20)])[0]
    lam, coef, bias, cv = H.fit_fold(K, y, plan, lams=(1e-3, 1.0))
    assert lam == 1e-3 and coef.shape == (H.INNER + 1, 120)
    fit, inner = plan["fit"], plan["inner"]
    for k in range(H.INNER):
        out = fit[inner == k]
        assert np.all(coef[k, out] == 0.0) and np.all(coef[k, plan["held"]] == 0.0)
        assert np.allclose(K[out] @ coef[k] + bias[k], cv[inner == k])
    assert np.all(coef[-1, plan["valid"]] == 0.0) and np.all(coef[-1, fit] != 0.0)
    worse = y + rng.normal(size=120)
    p = H.paired(y, y + 0.01, worse, molecule, draws=200)
    assert p["mean"] < 0 and p["high"] < 0 and p["excludes_zero"]
    same = H.paired(y, worse, worse, molecule, draws=50)
    assert same["mean"] == 0.0 and not same["excludes_zero"]
