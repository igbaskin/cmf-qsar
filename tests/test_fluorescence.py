"""Plan item SP4: spectra with two axes as fields on a plane.

No test needs the downloaded sets: the files, the maps and the peak lists are made up here.
"""
import hashlib
import io
import os
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from cmfqsar import spectrum as SP  # noqa: E402
from cmfqsar import spectrum2d as S2  # noqa: E402
import prepare_fluorescence as PREP  # noqa: E402
import run_fluorescence as R  # noqa: E402


# --------------------------------------------------------------------------- #
# cmfqsar.spectrum2d
# --------------------------------------------------------------------------- #
def _lists(seed=0, n=7):
    rng = np.random.default_rng(seed)
    a = S2.PeakList(rng.uniform(20, 80, n), rng.uniform(20, 80, n), rng.uniform(0.5, 2, n))
    b = S2.PeakList(a.x + rng.normal(0, 1, n), a.y + rng.normal(0, 1, n), a.w[::-1].copy())
    return a, b


def test_the_closed_form_of_two_peak_lists_and_the_grid_agree():
    a, b = _lists()
    sx, sy = 3.0, 2.0
    x = y = np.arange(0.0, 100.0, 0.5)
    ra, rb = S2.rasterise(a, x, y, sx, sy), S2.rasterise(b, x, y, sx, sy)
    on_grid = float((ra * rb).sum()) * 0.5 * 0.5 * 4.0 * np.pi * sx * sy
    assert on_grid == pytest.approx(S2.peak_overlap(a, b, sx, sy), rel=1e-9)
    assert S2.peak_overlap(a, b, sx, sy) == pytest.approx(S2.peak_overlap(b, a, sx, sy))
    assert float(ra.sum()) * 0.25 == pytest.approx(a.w.sum(), rel=1e-6)    # the area is kept
    # one peak against itself is its squared intensity; far apart it is nothing
    one = S2.PeakList([10.0], [10.0], [3.0])
    far = S2.PeakList([90.0], [10.0], [3.0])
    assert S2.peak_overlap(one, one, 1.0, 1.0) == pytest.approx(9.0)
    assert S2.peak_overlap(one, far, 1.0, 1.0) < 1e-100
    # a peak that is moved along one axis: only the width of that axis counts
    moved = S2.PeakList([10.0], [12.0], [3.0])
    assert S2.peak_overlap(one, moved, 50.0, 1.0) == pytest.approx(9.0 * np.exp(-1.0))
    assert S2.peak_overlap(one, moved, 1.0, 50.0) == pytest.approx(9.0, rel=1e-3)


def test_the_kernel_of_peak_lists_and_its_errors():
    a, b = _lists()
    K = S2.peak_kernel([a, b], 3.0, 2.0)
    assert np.allclose(np.diag(K), 1.0) and 0.5 < K[0, 1] < 1.0 and K[0, 1] == K[1, 0]
    raw = S2.peak_kernel([a, b], 3.0, 2.0, normalize=False)
    assert raw[0, 0] == pytest.approx(S2.peak_overlap(a, a, 3.0, 2.0))
    cross = S2.peak_kernel([a], 3.0, 2.0, others=[a, b])
    assert cross.shape == (2, 1) and cross[0, 0] == pytest.approx(1.0)
    empty = S2.PeakList([], [], [])
    assert len(empty) == 0 and S2.peak_overlap(a, empty, 1.0, 1.0) == 0.0
    assert np.all(S2.peak_kernel([a, empty], 1.0, 1.0)[1] == 0.0)
    assert S2.rasterise(empty, np.arange(4.0), np.arange(3.0), 1.0, 1.0).shape == (4, 3)
    with pytest.raises(ValueError):
        S2.PeakList([1.0, 2.0], [1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        S2.PeakList([np.nan], [1.0], [1.0])
    with pytest.raises(ValueError):
        S2.peak_overlap(a, b, 0.0, 1.0)


def test_scattered_light_is_marked():
    ex = np.array([300.0, 350.0])
    em = np.arange(280.0, 721.0, 10.0)
    mask = S2.scatter_mask(ex, em, 15.0)
    assert mask.shape == (2, len(em))
    row = dict(zip(em, mask[0]))
    assert row[280.0] and row[300.0] and row[310.0]          # below, and the first order
    assert not row[320.0] and not row[500.0]
    assert row[590.0] and row[600.0] and row[610.0] and not row[620.0]    # the second order
    only = S2.scatter_mask(ex, em, 15.0, orders=(2,), below=False)
    assert not only[0][em == 300.0][0] and only[0][em == 600.0][0]
    assert only[1][em == 700.0][0] and not only[1][em == 280.0][0]


def test_broadening_a_map_with_missing_values():
    rng = np.random.default_rng(1)
    flat = np.full((2, 12, 15), 4.0)
    flat[0, 3:6, 4:9] = np.nan
    assert np.allclose(S2.smoothed(flat, 1.0, 2.0, 3.0, 4.0), 4.0)     # a hole is filled
    hole = np.full((30, 30), np.nan)
    hole[0, 0] = 1.0
    out = S2.smoothed(hole, 1.0, 1.0, 1.0, 1.0)
    assert out[0, 0] == pytest.approx(1.0) and out[20, 20] == 0.0       # nothing within reach
    M = rng.normal(size=(3, 20, 40))
    both = S2.smoothed(M, 2.0, 0.5, 4.0, 3.0)
    by_hand = SP.smoothed(np.swapaxes(SP.smoothed(np.swapaxes(M, 1, 2), 2.0, 4.0), 1, 2),
                          0.5, 3.0)
    assert np.allclose(both, by_hand)
    assert np.allclose(S2.smoothed(M, 2.0, 0.5, 0.0, 0.0), M)
    only_y = S2.smoothed(M, 2.0, 0.5, 0.0, 3.0)
    assert np.allclose(only_y, SP.smoothed(M, 0.5, 3.0))
    for bad in ((-1.0, 1.0), (1.0, -1.0)):
        with pytest.raises(ValueError):
            S2.smoothed(M, 1.0, 1.0, *bad)
    with pytest.raises(ValueError):
        S2.smoothed(np.zeros(5), 1.0, 1.0, 1.0, 1.0)


def test_bins_rows_the_kernel_and_the_co_map():
    M = np.arange(2 * 4 * 7, dtype=float).reshape(2, 4, 7)
    B = S2.binned(M, 2, 3)
    assert B.shape == (2, 2, 2) and B[0, 0, 0] == pytest.approx(M[0, :2, :3].mean())
    M[0, 0, 0] = np.nan
    assert S2.binned(M, 2, 3)[0, 0, 0] == pytest.approx(np.nanmean(M[0, :2, :3]))
    M[1, :2, :3] = np.nan
    assert S2.binned(M, 2, 3)[1, 0, 0] == 0.0                # a bin without a measured value
    with pytest.raises(ValueError):
        S2.binned(M, 9, 1)
    F = S2.rows(M)
    assert F.shape == (2, 28) and F[0, 0] == 0.0 and np.isfinite(F).all()
    rng = np.random.default_rng(2)
    maps = rng.random((6, 5, 8))
    K = S2.map_kernel(maps)
    assert np.allclose(np.diag(K), 1.0) and np.allclose(K, S2.map_kernel(S2.rows(maps)))
    assert S2.map_kernel(maps, maps[:2]).shape == (2, 6)
    alpha = rng.normal(size=6)
    w = S2.co_map(maps, alpha)
    assert w.shape == (5, 8)
    new = rng.random((1, 5, 8))
    z = S2.rows(new)[0] / np.linalg.norm(S2.rows(new)[0])
    by_kernel = (S2.map_kernel(maps, new) @ alpha).item()
    assert float((w.ravel() * z).sum()) == pytest.approx(by_kernel)
    x, y = np.arange(5.0), np.arange(8.0)
    peak = np.zeros((5, 8))
    peak[1, 2], peak[1, 3], peak[4, 7] = 5.0, -4.0, 3.0
    places = S2.strongest_places(x, y, peak, n=2, separation=(1.0, 1.0))
    assert [(p["x"], p["y"], p["value"]) for p in places] == [(1.0, 2.0, 5.0), (4.0, 7.0, 3.0)]
    assert S2.strongest_places(x, y, peak, n=2)[1]["value"] == -4.0
    with pytest.raises(ValueError):
        S2.strongest_places(x, y, peak.T)


# --------------------------------------------------------------------------- #
# The reader of the two archives
# --------------------------------------------------------------------------- #
def _instrument_file(samples, excitations, rng, shift=0.03):
    """A file as the fluorescence instrument writes it: pairs of columns, a block of
    numbers, an empty line and a log."""
    grid = np.arange(300.0, 311.0, 2.0)
    measured = grid + shift * (grid - 300.0) / 10.0                    # not exactly even
    names, cols, truth = [], [], {}
    for ex in excitations:
        for s in samples:
            y = rng.random(len(grid))
            names += [f"{s}_EX_{ex:.2f}", ""]
            cols += [measured, y]
            truth[(s, ex)] = y
    lines = [",".join(names) + ",", ",".join(["Wavelength (nm)", "Intensity (a.u.)"]
                                             * (len(names) // 2)) + ","]
    for k in range(len(grid)):
        lines.append(",".join(f"{c[k]:.7f}" for c in cols) + ",")
    lines += ["", f"{samples[0]}_EX_300.00,", "Collection Time: 5/12/2021", "Instrument  Cary"]
    return "\n".join(lines), truth, grid


def _absorption_file(values):
    lines = ["Advanced Reads Report", "", "Wavelength (nm)  ,232.00, 264.00, 268.00, 272.00  ,",
             "       Sample        , F  ,Abs(232.00) ,    Abs(264.00) ,", "______________"]
    for s, v in values.items():
        lines.append(f"{s}                   ,    ,     {v[0]:.4f} ,         {v[1]:.4f} ,"
                     f"         {v[2]:.4f} ,         {v[3]:.4f} ,")
    return "\n".join(lines + ["", "Results Flags Legend"])


def test_the_files_of_the_two_instruments_are_read():
    rng = np.random.default_rng(3)
    text, truth, grid = _instrument_file(["C2", "A1"], [300.0, 310.0, 320.0], rng)
    found = PREP.fluorescence_file(text, grid)
    assert set(found) == {"C2", "A1"}
    ex, block = found["A1"]
    assert np.allclose(ex, [300.0, 310.0, 320.0]) and block.shape == (3, len(grid))
    assert np.allclose(block[1], truth[("A1", 310.0)], atol=0.02)      # interpolated a little
    values = {"F1": (2.1783, 0.2366, 0.2420, 0.2316), "N0": (2.5216, 0.1896, 0.1947, 0.1856)}
    read = PREP.absorbance_file(_absorption_file(values))
    assert set(read) == {"F1", "N0"} and np.allclose(read["N0"], values["N0"])


def test_the_olive_set_is_put_together_from_its_files(monkeypatch):
    rng = np.random.default_rng(4)
    monkeypatch.setattr(PREP, "OLIVE_EMISSION", (300.0, 310.0, 2.0))
    files = {}
    for step in (0, 1):
        text, _, _ = _instrument_file(["A0", "A1", "A2", "B0"], [300.0, 310.0], rng)
        files[f"Aging Step {step}/Fluorescence/2021051{step}_0750_AS{step}_A0A1A2B0.csv"] = \
            text.encode()
        files[f"Aging Step {step}/Absorption/2021051{step}_0940_AS{step}_A0B0.csv"] = \
            _absorption_file({"A0": (2.0 + step, 0.2, 0.25, 0.2),
                              "B0": (1.5, 0.1, 0.1, 0.1)}).encode()
    again, truth, _ = _instrument_file(["B0"], [300.0, 310.0], rng)      # measured once more
    files["Aging Step 1/Fluorescence/20210512_0900_AS1_B0.csv"] = again.encode()
    files["Aging Step 1/Fluorescence/20210617_1117_AS1_EFH_WhiteWine.csv"] = b"x"
    o = PREP.olive(files)
    assert o["maps"].shape == (8, 2, 6) and o["maps"].dtype == np.float32
    assert list(o["oil"]) == ["A", "A", "A", "B"] * 2 and list(o["step"]) == [0] * 4 + [1] * 4
    assert list(o["replicate"]) == [0, 1, 2, 0] * 2
    assert np.isfinite(o["absorbance"][:, 0]).sum() == 4               # one vial of three
    row = np.where((o["oil"] == "A") & (o["step"] == 1) & (o["replicate"] == 0))[0][0]
    assert np.allclose(o["absorbance"][row], [3.0, 0.2, 0.25, 0.2])
    later = np.where((o["oil"] == "B") & (o["step"] == 1))[0][0]
    assert np.allclose(o["maps"][later][1], truth[("B0", 310.0)], atol=0.02)
    assert "WhiteWine" in o["notes"][0] and "step 1, sample B0" in o["notes"][1]

    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    outer = io.BytesIO()
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("Dataset/Raw_data.zip", inner.getvalue())
    assert set(PREP.olive_members(outer.getvalue())) == set(files)


def test_the_honey_set_is_read_from_its_matlab_file(tmp_path):
    from scipy.io import savemat

    rng = np.random.default_rng(5)
    data = rng.random((5, 7, 3))                                       # sample, emission, excitation
    data[1, :, 2] = np.nan
    cells = np.empty((3, 2), dtype=object)
    classes = np.empty((3, 2), dtype=object)
    lookup = np.empty((3, 1), dtype=object)
    for i in range(3):
        for j in range(2):
            cells[i, j] = np.zeros((0, 0))
            classes[i, j] = np.zeros((0, 0))
        lookup[i, 0] = np.zeros((0, 0))
    cells[1, 0] = np.arange(270.0, 273.5, 0.5)[None, :]
    cells[2, 0] = np.array([[240.0, 245.0, 250.0]])
    classes[0, 0] = np.array([[1, 2, 1, 3, 3]], dtype=np.uint8)
    table = np.empty((3, 2), dtype=object)
    for k, name in enumerate(["Acacia", "False acacia", "Linden"]):
        table[k, 0], table[k, 1] = np.array([[k + 1]]), name
    lookup[0, 0] = table
    buffer = io.BytesIO()
    savemat(buffer, {"X": {"data": data, "axisscale": cells, "class": classes,
                           "classlookup": lookup}})
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("HoneyEEM.mat", buffer.getvalue())
    h = PREP.honey(archive.getvalue())
    assert h["maps"].shape == (5, 3, 7) and np.allclose(h["maps"][0, 1], data[0, :, 1])
    assert np.isnan(h["maps"][1, 2]).all()
    assert list(h["label"]) == ["acacia", "false acacia", "acacia", "linden", "linden"]
    assert list(h["false"]) == [False, True, False, False, False]
    assert np.allclose(h["excitation"], [240.0, 245.0, 250.0]) and len(h["emission"]) == 7

    path = tmp_path / "a.zip"
    path.write_bytes(archive.getvalue())
    assert PREP.checked(str(path), hashlib.sha256(archive.getvalue()).hexdigest())
    with pytest.raises(SystemExit, match="The set has changed"):
        PREP.checked(str(path), "0" * 64)
    with pytest.raises(SystemExit, match="prepare_fluorescence.py"):
        PREP.load("honey", str(tmp_path / "nothing"))


# --------------------------------------------------------------------------- #
# The models on a kernel
# --------------------------------------------------------------------------- #
def test_pls_from_the_kernel_is_pls_on_the_rows():
    from sklearn.cross_decomposition import PLSRegression

    rng = np.random.default_rng(6)
    F = rng.normal(size=(40, 200))
    y = F[:, :5].sum(axis=1) + 0.1 * rng.normal(size=40)
    K = F @ F.T
    tr, te = np.arange(30), np.arange(30, 40)
    P = R.regression_path(K, y, tr, te, "pls")
    assert P.shape == (R.PLS_MAX, 10)
    scale = F[tr].std()
    for k in (1, 4, 12):
        ref = PLSRegression(n_components=k, scale=False).fit(F[tr] / scale, y[tr])
        assert np.allclose(P[k - 1], ref.predict(F[te] / scale).ravel(), atol=1e-8)
    ridge = R.regression_path(K, y, tr, te, "kernel")
    Fc = F[tr] - F[tr].mean(axis=0)
    Kc = Fc @ Fc.T
    lam = R.RIDGES[5] * np.trace(Kc) / 30
    alpha = np.linalg.solve(Kc + lam * np.eye(30), y[tr] - y[tr].mean())
    assert np.allclose(ridge[5], (F[te] - F[tr].mean(axis=0)) @ Fc.T @ alpha + y[tr].mean(),
                       atol=1e-8)
    flat = R.pls_alphas(np.zeros((6, 6)), np.arange(6.0), 3)            # nothing to fit
    assert np.allclose(flat, 0.0)


def test_folds_scores_and_descriptions():
    rng = np.random.default_rng(7)
    labels = np.array(["a"] * 20 + ["b"] * 9 + ["c"] * 6)
    folds = R.folds_by_class(labels, 5, rng)
    assert sorted(np.concatenate(folds)) == list(range(35))
    assert all(np.sum(labels[f] == "a") == 4 and 1 <= np.sum(labels[f] == "c") <= 2
               for f in folds)
    groups = np.repeat(np.arange(8), 5)
    parts = R.folds_by_group(groups, 4, rng)
    assert sorted(np.concatenate(parts)) == list(range(40))
    assert all(len(set(groups[p])) == 2 for p in parts)
    assert not set(groups[parts[0]]) & set(groups[parts[1]])            # a group stays whole
    truth = np.array(["a", "a", "a", "a", "b", "b"])
    said = np.array(["a", "a", "a", "a", "a", "b"])
    assert R.balanced_accuracy(truth, said, ["a", "b"]) == pytest.approx(0.75)
    assert R.auc([3.0, 2.0], [1.0, 2.0]) == pytest.approx((1 + 1 + 1 + 0.5) / 4)
    assert R.rmse([1.0, 3.0], [0.0, 0.0]) == pytest.approx(np.sqrt(5.0))

    maps = rng.random((9, 4, 6))
    raw = R.raw_kernels(maps, 10.0, 2.0, widths_x=(0.0, 10.0), widths_y=(0.0,))
    assert set(raw) == {(0.0, 0.0), (10.0, 0.0)}
    assert np.allclose(raw[(0.0, 0.0)], S2.rows(maps) @ S2.rows(maps).T)
    members = np.arange(9).reshape(3, 3)
    d = R.descriptions(raw, mean_of=members)
    mean_rows = S2.rows(maps).reshape(3, 3, -1).mean(axis=1)
    K = mean_rows @ mean_rows.T
    assert np.allclose(d[R.name_of(0.0, 0.0, False)], K / np.mean(np.diag(K)))
    assert np.allclose(np.diag(d[R.name_of(0.0, 0.0, True)]), 1.0)
    picked = R.descriptions(raw, pick=np.array([0, 3, 6]))
    assert picked[R.name_of(10.0, 0.0, False)].shape == (3, 3)
    assert R.shown("10|4|1") == "10 nm by 4 nm, each map divided by its length"
    assert R.shown("0|0|0") == "no broadening"


def test_a_class_is_found_and_the_edge_of_a_grid_is_counted():
    rng = np.random.default_rng(8)
    centres = rng.normal(size=(3, 30))
    labels = np.repeat(np.array(["x", "y", "z"]), 16)
    F = centres[np.repeat(np.arange(3), 16)] + 0.4 * rng.normal(size=(48, 30))
    good = F @ F.T
    noise = rng.normal(size=(48, 30))
    kernels = {"noise": noise @ noise.T, "good": good}
    tr = np.arange(48)[np.arange(48) % 4 != 0]
    te = np.arange(48)[np.arange(48) % 4 == 0]
    inner = R.folds_by_class(labels[tr], 3, rng)
    R.EDGES.clear()
    for model in ("pls", "kernel"):
        name, k = R.chosen_classifier(kernels, labels, ["x", "y", "z"], tr, inner, model)
        assert name == "good"
        S = R.class_scores(good, labels, ["x", "y", "z"], tr, te, model)
        said = np.array(["x", "y", "z"])[S[R._setting(good, k, tr, model)].argmax(axis=1)]
        assert np.mean(said == labels[te]) > 0.9
    assert set(R.EDGES) == {"pls", "kernel"} and all(len(v) == 1 for v in R.EDGES.values())
    assert R._setting(good, 10 ** 6, tr, "pls") == R.PLS_MAX - 1        # counted as an edge
    edges = R._edges()
    assert edges["pls"] == 0.5 and not R.EDGES
    y = F[:, 0] + 0.05 * rng.normal(size=48)
    idx = [np.sort(f) for f in np.array_split(rng.permutation(len(tr)), 3)]
    assert R.chosen_regression(kernels, y, tr, idx, "kernel")[0] == "good"


# --------------------------------------------------------------------------- #
# The benchmark on made-up sets
# --------------------------------------------------------------------------- #
def _band(ex, em, e0, m0, se=12.0, sm=6.0):
    return np.exp(-0.5 * ((ex[:, None] - e0) / se) ** 2 - 0.5 * ((em[None, :] - m0) / sm) ** 2)


def _load(which, cache=None):
    rng = np.random.default_rng(9)
    if which == "honey":
        ex, em = np.arange(240.0, 311.0, 10.0), np.arange(350.0, 390.0, 0.5)
        names = ["acacia", "linden", "meadow", "sunflower"]
        places = [(250.0, 358.0), (300.0, 358.0), (250.0, 380.0), (300.0, 380.0)]
        maps, label = [], []
        for c, (e0, m0) in zip(names, places):
            for _ in range(10):
                maps.append(_band(ex, em, 270.0, 370.0, 30.0, 12.0)
                            + rng.uniform(0.8, 1.2) * _band(ex, em, e0, m0)
                            + 0.02 * rng.normal(size=(len(ex), len(em))))
                label.append(c)
        for _ in range(4):                                  # a band that no real honey has
            maps.append(_band(ex, em, 275.0, 369.0, 5.0, 3.0) * 3.0
                        + 0.02 * rng.normal(size=(len(ex), len(em))))
            label.append("false acacia")
        label = np.array(label)
        return {"maps": np.array(maps, dtype=np.float32), "excitation": ex, "emission": em,
                "label": label, "false": np.array([s.startswith("false") for s in label])}
    ex, em = np.arange(300.0, 351.0, 10.0), np.arange(400.0, 440.0, 2.0)
    maps, oil, step, rep, absorb = [], [], [], [], []
    for o in "ABCDEF":
        level = rng.uniform(0.5, 1.0)
        for s in range(4):
            for r in range(3):
                maps.append(level * _band(ex, em, 310.0, 410.0)
                            + 0.4 * s * _band(ex, em, 340.0, 430.0)
                            + 0.05 * rng.normal(size=(len(ex), len(em))))
                oil.append(o)
                step.append(s)
                rep.append(r)
                k232 = 2.0 + 0.3 * s + 0.01 * rng.normal()
                absorb.append([k232, 0.1, 0.1 + 0.05 * s, 0.1] if r == 0 else [np.nan] * 4)
    return {"maps": np.array(maps, dtype=np.float32), "excitation": ex, "emission": em,
            "oil": np.array(oil), "step": np.array(step), "replicate": np.array(rep),
            "absorbance": np.array(absorb)}


def test_the_two_sets_are_prepared_for_the_benchmark(monkeypatch):
    monkeypatch.setattr(R.PREP, "load", _load)
    h = R.honey()
    assert h["maps"].shape == (44, 8, 20)                    # four emission points in a bin
    assert h["emission"][0] == pytest.approx(350.75) and h["false"].sum() == 4
    o = R.olive()
    assert len(o["measured"]) == 24 and o["members"].shape == (24, 3)
    assert all(m[0] == i for m, i in zip(o["members"], o["measured"]))  # its own map first
    assert np.allclose(o["delta"], o["k268"] - 0.1)
    assert list(o["within"][:4]) == [True, False, False, False]        # the third limit
    masked = R.without_scatter(np.ones((1, 2, 5)), np.array([300.0, 400.0]),
                               np.array([300.0, 350.0, 400.0, 600.0, 800.0]), 10.0, 10.0)
    assert np.isnan(masked[0, 0, [0, 3]]).all() and np.isfinite(masked[0, 0, [1, 2, 4]]).all()
    assert np.isnan(masked[0, 1, [0, 1, 2, 4]]).all() and np.isfinite(masked[0, 1, 3])


def test_the_benchmark_on_made_up_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(R.PREP, "load", _load)
    monkeypatch.setattr(R, "OLIVE_FOLDS", 3)
    grid = dict(widths_x=(0.0,), widths_y=(0.0, 4.0))
    out = R.run(str(tmp_path), parts=("honey", "olive", "simulated"),
                honey=dict(repeats=1, **grid), olive=dict(repeats=1, **grid),
                simulated=dict(repeats=1, jitters=(0.0, 2.0), per_class=12, peaks=10))
    h = out["honey"]
    assert h["samples"] == 40 and h["false"] == 4 and h["counts"] == [10, 10, 10, 10]
    assert h["accuracy"]["kernel"]["mean"] > 0.9 and h["accuracy"]["pls"]["mean"] > 0.9
    assert np.array(h["confusion"]["kernel"]).sum() == 40
    assert set(h["co_map"]["classes"]) == set(h["classes"])
    assert all(v["mean"] > 0.9 for k, v in h["false_against_real_auc"].items()
               if k.endswith("plain"))
    assert set(h["setting_at_an_edge"]) == {"pls", "kernel"}

    o = out["olive"]
    assert o["samples"] == 24 and o["oils"] == 6 and o["maps"] == 72
    way = "whole oils held out"
    assert o["rmse"][f"{way}|k232|kernel|one map"]["mean"] \
        < 0.5 * o["rmse"][f"{way}|k232|mean of the fit|"]["mean"]
    assert f"{way}|k232|kernel|mean of three maps" in o["rmse"]
    assert "random folds|k232|kernel|mean of three maps" not in o["rmse"]
    assert set(o["differences"]["k232"]) == {
        "kernel_minus_pls", "kernel_minus_pls_as_measured", "mean_of_three_minus_one_map"}
    assert o["balanced_accuracy"][f"{way}|kernel|one map"]["mean"] > 0.8

    s = out["simulated"]
    assert set(s["jitter"]) == {"0", "2"}
    a = s["jitter"]["0"]["accuracy"]
    assert {"peaks|kernel", "peaks|pls", "bins|kernel", "bins|pls", "bin 1|pls"} <= set(a)
    assert a["peaks|kernel"]["mean"] > 0.6

    text = R.report(str(tmp_path))
    assert text.startswith("# Spectra with two axes")
    for heading in ("## Part 1: honey", "## Part 1: the authentication case",
                    "## Part 2: olive oil", "## Part 2: within the limits",
                    "## Part 3: simulated peak lists"):
        assert heading in text
    assert "| kernel Fisher discriminant, description chosen inside the fit |" in text
    with open(tmp_path / "table.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    again = R.run(str(tmp_path), parts=("simulated",),
                  simulated=dict(repeats=1, jitters=(1.0,), per_class=10, peaks=8))
    assert "honey" in again and set(again["simulated"]["jitter"]) == {"1"}  # the rest is kept
    assert R.report(str(tmp_path / "nothing")) == ""

    monkeypatch.setattr(R, "SIM_JITTER", (0.0, 3.0))
    c = R.run(str(tmp_path), parts=("correlation",), correlation=grid)["correlation"]
    assert set(c["maps"]) == {"honey", "olive"} and set(c["maps"]["honey"]) == {"0|4"}
    assert 0.9 < c["maps"]["olive"]["0|4"] <= 1.0           # a smooth map changes little
    still, moved = c["simulated"]["0"], c["simulated"]["3"]
    assert still["width 2, bins of 1"] > moved["width 2, bins of 1"]
    assert "## The kernel correlation" in R.report(str(tmp_path))


def test_the_correlation_of_two_kernels():
    rng = np.random.default_rng(11)
    F = rng.normal(size=(12, 5))
    K = F @ F.T
    assert R.kernel_correlation(K, 3.0 * K + 2.0) == pytest.approx(1.0)
    assert R.kernel_correlation(K, -K) == pytest.approx(-1.0)
    other = rng.normal(size=(12, 5))
    assert abs(R.kernel_correlation(K, other @ other.T)) < 0.6


def test_simulated_peak_lists_and_their_bins():
    rng = np.random.default_rng(10)
    lists, labels = R.simulated_lists(0.0, rng, per_class=6, peaks=9)
    assert len(lists) == 12 and list(labels) == [0] * 6 + [1] * 6
    assert all(len(p) == 9 + R.SIM_EXTRA for p in lists)
    assert np.allclose(lists[0].x[:9], lists[7].x[:9])                  # no jitter: one place
    strong = np.mean([p.w[:R.SIM_MARKERS].mean() for p in lists[6:]])
    weak = np.mean([p.w[:R.SIM_MARKERS].mean() for p in lists[:6]])
    assert strong / weak == pytest.approx(1.0 + R.SIM_EFFECT, rel=0.2)
    moved, _ = R.simulated_lists(3.0, rng, per_class=6, peaks=9)
    assert not np.allclose(moved[0].x[:9], moved[7].x[:9])
    F = R.binned_rows(lists, 10.0)
    assert F.shape == (12, 100) and np.allclose(F.sum(axis=1), [p.w.sum() for p in lists])
