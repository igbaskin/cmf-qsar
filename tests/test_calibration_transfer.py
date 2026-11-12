"""Plan item SP5: a model fitted on one instrument reads the spectra of another.

No test needs the downloaded sets: the files and the spectra are made up here.
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
import prepare_calibration_transfer as PREP  # noqa: E402
import run_calibration_transfer as T  # noqa: E402


# --------------------------------------------------------------------------- #
# cmfqsar.spectrum: two instruments
# --------------------------------------------------------------------------- #
def _bands(x, centres, heights, width):
    return sum(h * np.exp(-0.5 * ((x - c) / width) ** 2) for c, h in zip(centres, heights))


def test_matched_widths_bring_every_instrument_to_one_width():
    w = SP.matched_widths(4.0, [0.0, 3.0])
    assert np.allclose(w, [5.0, 4.0])                   # 3-4-5: the sharper one gets more
    assert np.allclose(w ** 2 + np.array([0.0, 9.0]), 25.0)
    assert np.allclose(SP.matched_widths(0.0, [2.0, 2.0]), 0.0)
    for bad in ([], [-1.0]):
        with pytest.raises(ValueError):
            SP.matched_widths(1.0, bad)
    with pytest.raises(ValueError):
        SP.matched_widths(-1.0, [1.0])


def test_two_broadenings_add_as_squares():
    x = np.arange(0.0, 400.0, 1.0)
    a = _bands(x, (120.0, 200.0, 290.0), (1.0, 0.5, 0.8), 2.0)[None, :]
    twice = SP.smoothed(SP.smoothed(a, 1.0, 3.0), 1.0, 4.0)
    once = SP.smoothed(a, 1.0, 5.0)
    assert np.abs(twice - once).max() < 1e-6
    assert SP.smoothed(a, 1.0, 0.0) is not None and np.array_equal(SP.smoothed(a, 1.0, 0.0), a)
    flat = np.full((2, 50), 3.0)                        # a baseline does not fall at the ends
    assert np.allclose(SP.smoothed(flat, 2.0, 9.0), 3.0)
    with pytest.raises(ValueError):
        SP.smoothed(a, 1.0, -1.0)


def test_a_shift_is_applied_and_found_again():
    x = np.arange(1000.0, 1400.0, 2.0)
    a = _bands(x, (1100.0, 1210.0, 1330.0), (1.0, 0.6, 0.9), 12.0)
    moved = SP.shifted(a, x, 1.3)
    exact = _bands(x, (1101.3, 1211.3, 1331.3), (1.0, 0.6, 0.9), 12.0)
    assert np.abs(moved - exact)[5:-5].max() < 1e-4
    assert SP.shifted(a, x, 0.0) is a
    b = exact + 0.4 + 0.001 * (x - 1200.0)              # an offset and a sloping baseline
    assert SP.fit_shift(a, b, x, 4.0) == pytest.approx(1.3, abs=0.11)
    assert SP.fit_shift(b, a, x, 4.0) == pytest.approx(-1.3, abs=0.11)
    back = SP.shifted(b[None, :], x, -SP.fit_shift(a, b, x, 4.0))[0]
    assert np.abs(np.gradient(back, x) - np.gradient(a, x))[5:-5].max() < 2e-3
    with pytest.raises(ValueError):
        SP.fit_shift(a, b, x, 0.0)
    with pytest.raises(ValueError):
        SP.fit_shift(a[:6], b[:6], x[:6], 4.0)


def test_a_difference_in_resolution_is_found_with_its_direction():
    x = np.arange(1000.0, 1600.0, 2.0)
    a = _bands(x, (1100.0, 1210.0, 1330.0, 1480.0), (1.0, 0.6, 0.9, 0.7), 6.0)
    coarse = SP.smoothed(a, 2.0, 8.0) + 0.3
    tried = (2.0, 4.0, 8.0, 16.0)
    width, score = SP.fit_width(a, coarse, x, tried)
    assert width == 8.0 and score > 0.999               # the second curve is the coarser one
    width, _ = SP.fit_width(coarse, a, x, tried)
    assert width == -8.0
    width, score = SP.fit_width(a, a + 1.0, x, tried)
    assert width == 0.0 and score == pytest.approx(1.0)
    with pytest.raises(ValueError):
        SP.fit_width(a[:20], a[:20], x[:20], tried)


# --------------------------------------------------------------------------- #
# The reader of the two archives
# --------------------------------------------------------------------------- #
def _dataset(data, scale=None, names=None):
    """A record with the fields of a "DataSet" object that the reader uses."""
    cells = np.empty((2, 2), dtype=object)
    labels = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            cells[i, j] = np.zeros((0, 0))
            labels[i, j] = np.array([], dtype="U1")
    if scale is not None:
        cells[1, 0] = np.asarray(scale, float)[None, :]
    if names is not None:
        width = max(len(n) for n in names)
        labels[1, 0] = np.array([n.ljust(width) for n in names])
    return {"data": np.asarray(data, float), "axisscale": cells, "label": labels}


def _archive(path, member, variables):
    from scipy.io import savemat

    buffer = io.BytesIO()
    savemat(buffer, variables)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(member, buffer.getvalue())
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def test_the_reader_takes_the_tables_out_of_the_matlab_files(tmp_path):
    rng = np.random.default_rng(0)
    x = np.arange(1100.0, 1120.0, 2.0)
    variables = {"propvals": _dataset(rng.normal(size=(6, 4)),
                                      names=["Moisture", "Oil", "Protein", "Starch"])}
    for k, name in enumerate(PREP.CORN_INSTRUMENTS):
        variables[name + "spec"] = _dataset(rng.normal(size=(6, 10)) + k, x)
        variables[name + "nbs"] = _dataset(rng.normal(size=(3, 10)), x)
    path = str(tmp_path / "corn.zip")
    want = _archive(path, "corn.mat", variables)
    found = PREP.records(path, "corn.mat", want)
    data, scale, names = PREP.dataset(found["propvals"])
    assert data.shape == (6, 4) and scale.size == 0 and names[2] == "Protein"
    corn = PREP.corn(found)
    assert corn["spectra"].shape == (3, 6, 10) and np.array_equal(corn["wavelength"], x)
    assert list(corn["property_names"]) == ["moisture", "oil", "protein", "starch"]
    assert np.allclose(corn["spectra"][2], variables["mp6spec"]["data"])
    assert corn["standards_mp5"].shape == (3, 10)
    with pytest.raises(SystemExit, match="The set has changed"):
        PREP.records(path, "corn.mat", "0" * 64)
    variables["mp5spec"] = _dataset(rng.normal(size=(6, 10)), x + 1.0)
    _archive(path, "corn.mat", variables)
    with pytest.raises(ValueError, match="another wavelength scale"):
        PREP.corn(PREP.records(path, "corn.mat"))

    sizes = {"calibrate": 5, "validate": 2, "test": 7}
    variables = {}
    for part, n in sizes.items():
        for k in (1, 2):
            variables[f"{part}_{k}"] = _dataset(rng.normal(size=(n, 10)) + k, x)
        variables[part + "_Y"] = _dataset(rng.normal(size=(n, 3)),
                                          names=["weight", "hardness", "assay"])
    path = str(tmp_path / "tablets.zip")
    _archive(path, "t.mat", variables)
    tab = PREP.tablets(PREP.records(path, "t.mat"))
    assert tab["spectra"].shape == (2, 14, 10) and tab["properties"].shape == (14, 3)
    assert list(tab["part"]) == ["calibrate"] * 5 + ["validate"] * 2 + ["test"] * 7
    assert np.allclose(tab["spectra"][1][5:7], variables["validate_2"]["data"])
    assert list(tab["property_names"]) == ["weight", "hardness", "assay"]
    with pytest.raises(SystemExit, match="prepare_calibration_transfer.py"):
        PREP.load("corn", str(tmp_path / "nothing"))


# --------------------------------------------------------------------------- #
# The models and the standards
# --------------------------------------------------------------------------- #
def _made_up(n=48, seed=0, shift=0.8, noise=0.002):
    """Spectra of three bands on two instruments.  The second instrument has a shifted axis,
    another baseline and a gain.  The property is the amount of the first band."""
    rng = np.random.default_rng(seed)
    x = np.arange(1100.0, 1500.0, 2.0)
    amounts = rng.uniform(0.5, 1.5, size=(n, 3))
    centres = (1180.0, 1290.0, 1400.0)

    def measured(c, moved):
        clean = sum(a * np.exp(-0.5 * ((x - c0 - moved) / 9.0) ** 2)
                    for a, c0 in zip(c, centres))
        return clean + rng.normal(0.0, noise, len(x))

    first = np.array([measured(c, 0.0) for c in amounts])
    second = np.array([measured(c, shift) for c in amounts])
    second = 1.02 * second + 0.3 + 0.2 * np.sin((x - 1100.0) / 150.0)
    return x, np.array([first, second]), amounts[:, :1]


def test_pls_and_the_ridge_agree_with_their_references():
    from sklearn.cross_decomposition import PLSRegression

    x, S, Y = _made_up()
    F, y = T.described(S[0], 2.0, 4.0, "slope"), Y[:, 0]
    W, b = T.pls_weights(F[:36], y[:36], 8)
    assert W.shape == (8, F.shape[1]) and b.shape == (8,)
    scale = F[:36].std()
    for k in (1, 3, 8):
        ref = PLSRegression(n_components=k, scale=False).fit(F[:36] / scale, y[:36])
        assert np.allclose(F[36:] @ W[k - 1] + b[k - 1], ref.predict(F[36:] / scale).ravel(),
                           atol=1e-9)
    W, b = T.ridge_weights(F[:36], y[:36], (1e-3, 1e-1))
    Fc = F[:36] - F[:36].mean(axis=0)
    K = Fc @ Fc.T
    for k, ridge in enumerate((1e-3, 1e-1)):
        lam = ridge * np.trace(K) / 36
        alpha = np.linalg.solve(K + lam * np.eye(36), y[:36] - y[:36].mean())
        ref = (F[36:] - F[:36].mean(axis=0)) @ Fc.T @ alpha + y[:36].mean()
        assert np.allclose(F[36:] @ W[k] + b[k], ref, atol=1e-9)
    same, _ = T.pls_weights(np.ones((6, 4)), np.arange(6.0), 3)     # no column varies
    assert np.allclose(same, 0.0)


def test_the_description_is_linear_and_the_model_predicts_within_one_instrument():
    x, S, Y = _made_up()
    A = S[0]
    for form in T.FORMS:
        both = T.described(A[:2].sum(axis=0)[None, :], 2.0, 8.0, form)
        assert np.allclose(both, T.described(A[:2], 2.0, 8.0, form).sum(axis=0))
    flat = np.full((1, len(x)), 5.0)
    assert np.allclose(T.described(flat, 2.0, 4.0, "slope"), 0.0)    # no slope in an offset
    rng = np.random.default_rng(1)
    folds = [np.sort(f) for f in np.array_split(rng.permutation(36), 4)]
    for model in T.MODELS:
        fit = T.fitted(A[:36], Y[:36, 0], 2.0, "slope", model, folds, (0.0, 4.0))
        assert fit["width"] in (0.0, 4.0)
        assert T.rmse(T.predict(fit, A[36:]), Y[36:, 0]) < 0.03       # the spread is 0.29
        wider = T.predict(fit, A[36:], width=16.0)                    # another width is read
        assert not np.allclose(wider, T.predict(fit, A[36:]))


def test_the_standards_cover_the_set_and_pds_restores_the_source_spectra():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [1.0, 0.0], [0.5, 0.6], [0.5, 0.1]])
    picked = T.kennard_stone(X, 3)
    assert set(picked[:2]) == {0, 2} and picked[2] == 3              # the two ends, then the top
    assert len(T.kennard_stone(X, 9)) == 5
    x, S, Y = _made_up(noise=0.0)
    std = T.kennard_stone(S[0][:36], 8)
    pds = T.pds_fit(S[0][std], S[1][std], 3, 2)
    restored = T.pds_apply(pds, S[1][36:])
    before = np.sqrt(np.mean((S[1][36:] - S[0][36:]) ** 2))
    after = np.sqrt(np.mean((restored - S[0][36:]) ** 2))
    assert after < 0.02 * before
    loo = T.pds_leave_one_out(S[0][std], S[1][std])
    assert set(loo) <= {(h, r) for h in T.PDS_HALF for r in T.PDS_RANK}
    assert all(v.shape == S[1][std].shape for v in loo.values())
    rng = np.random.default_rng(2)
    folds = [np.sort(f) for f in np.array_split(rng.permutation(36), 4)]
    fit = T.fitted(S[0][:36], Y[:36, 0], 2.0, "plain", "pls", folds, (0.0,))
    assert T.pds_chosen(fit, Y[std, 0], loo) in loo


def test_the_difference_between_two_instruments_is_taken_apart():
    x, S, _ = _made_up(noise=0.0)
    d = T.difference(S[0], S[1], x)
    assert d["shift_nm"] == pytest.approx(0.8, abs=0.11) and d["width_difference_nm"] == 0.0
    assert d["difference"] > d["after_the_mean_difference"] \
        > d["after_offset_gain_and_slope_per_sample"] > d["after_the_shift_too"]
    assert d["after_the_shift_too"] < 0.02 * d["difference"]
    coarse = SP.smoothed(S[0], 2.0, 8.0)
    assert T.difference(S[0], coarse, x)["width_difference_nm"] == 8.0


def test_what_is_done_to_the_target_spectra():
    x, S, _ = _made_up(noise=0.0)
    fit, target = S[0][:30], S[1][30:]
    assert T.target_rows(T.VARIANTS[0], fit, target, x, 0.8) is target
    centred = T.target_rows(T.VARIANTS[1], fit, target, x, 0.8)
    assert np.allclose(centred.mean(axis=0), fit.mean(axis=0))
    assert np.allclose(centred - centred.mean(axis=0), target - target.mean(axis=0))
    moved = T.target_rows(T.VARIANTS[2], fit, target, x, 0.8)
    assert np.allclose(moved, SP.shifted(target, x, -0.8))
    both = T.target_rows(T.VARIANTS[3], fit, target, x, 0.8)
    assert np.allclose(both.mean(axis=0), fit.mean(axis=0))
    assert np.allclose(both - both.mean(axis=0), moved - moved.mean(axis=0))


# --------------------------------------------------------------------------- #
# The benchmark on made-up instruments
# --------------------------------------------------------------------------- #
def _load(which, cache=None):
    x, S, Y = _made_up(n=60, seed=3)
    return {"wavelength": x, "spectra": S, "instruments": np.array(["a", "b"]),
            "properties": np.hstack([Y, Y ** 2]),
            "property_names": np.array(["first", "square"])}


def test_the_benchmark_on_two_made_up_instruments(tmp_path, monkeypatch):
    monkeypatch.setattr(T.PREP, "load", _load)
    monkeypatch.setitem(T.SETS, "made up", {"properties": ("first",), "fit": 40,
                                            "range": (1100.0, 1498.0)})
    out = T.run(str(tmp_path), repeats=3, sets=("made up",), widths=(0.0, 4.0),
                coarser=(16.0,), n_standards=6, more_standards=8)
    s = out["sets"]["made up"]
    assert s["samples"] == 60 and s["test"] == 20 and s["instruments"] == ["a", "b"]
    assert s["difference"]["0->1"]["shift_nm"] == pytest.approx(0.8, abs=0.11)
    assert set(s["crossing"]) == {"0->1", "1->0"} and set(s["coarser"]) == {
        "0->0", "0->1", "1->0", "1->1"}
    plain = s["crossing"]["0->1"]["0"]["kernel"]["plain"]
    node = s["crossing"]["0->1"]["0"]["kernel"]["slope"]
    assert set(T.VARIANTS) <= set(node) and T.PDS_MORE in node
    assert out["standards"] == 6 and out["more_standards"] == 8
    # the baseline of the second instrument ruins the model on the plain spectra; centring
    # each instrument repairs it, and so does PDS; the slopes do not see the baseline
    assert plain[T.VARIANTS[0]]["mean"] > 5 * plain[T.VARIANTS[1]]["mean"]
    assert plain[T.VARIANTS[5]]["mean"] < 0.05 * plain[T.VARIANTS[0]]["mean"]
    assert node[T.VARIANTS[0]]["mean"] < 0.1 * plain[T.VARIANTS[0]]["mean"]
    within = s["within"]["0"]["0"]["kernel"]["slope"]
    assert node["fitted on the target"]["mean"] == pytest.approx(
        s["within"]["1"]["0"]["kernel"]["slope"][T.VARIANTS[0]]["mean"])
    assert node["within the source"]["mean"] == pytest.approx(within[T.VARIANTS[0]]["mean"])
    # centring has a cost of its own: the samples of the test do not have the mean of the
    # samples of the fit.  It is the same cost within one instrument and at the crossing.
    assert within[T.VARIANTS[1]]["mean"] > 10 * within[T.VARIANTS[0]]["mean"]
    assert node[T.VARIANTS[3]]["mean"] == pytest.approx(within[T.VARIANTS[1]]["mean"], rel=0.1)
    # a target of lower resolution: matched widths repair it, and the difference is found
    low = s["coarser"]["0->0"]["0"]["kernel"]["slope"]["16.0"]
    assert low[T.COARSE_VARIANTS[1]]["mean"] < 0.1 * low[T.COARSE_VARIANTS[0]]["mean"]
    assert low[T.COARSE_VARIANTS[2]]["mean"] == pytest.approx(low[T.COARSE_VARIANTS[1]]["mean"])
    assert s["fitted_width"]["0->0"]["16.0"]["mean"] == 16.0
    other = s["coarser"]["0->1"]["0"]["kernel"]["slope"]["16.0"]
    assert other[T.COARSE_VARIANTS[1]]["mean"] < other[T.COARSE_VARIANTS[0]]["mean"]
    # centring alone is also read on the source instrument, where nothing else differs
    assert T.VARIANTS[1] in low and T.VARIANTS[1] not in other
    assert low[T.COARSE_VARIANTS[1]]["mean"] < low[T.VARIANTS[1]]["mean"]
    g = s["gate"]["0->1|first"]
    assert g["ours"] == pytest.approx(node[T.VARIANTS[3]]["mean"])
    assert g["ours_minus_pls_as_measured"]["high"] < 0
    assert set(g["ours_minus_pls_with_pds"]) >= {"mean", "low", "high"}

    w = s["within_comparison"]["0|first"]
    assert w["kernel_slope"] == pytest.approx(within[T.VARIANTS[0]]["mean"])
    assert w["kernel_minus_pls_on_the_slopes"]["mean"] == pytest.approx(
        w["kernel_slope"] - w["pls_slope"])
    assert set(s["within_comparison"]) == {"0|first", "1|first"}

    text = T.report(str(tmp_path))
    assert "| made up, a, first |" in text
    assert text.startswith("# A model fitted on one instrument reads the spectra of another")
    assert "| made up, a to b |" in text and "## Part 3" in text
    assert "| Read as measured | Each instrument centred, widths not matched |" in text
    assert "In this table every column has the shift fitted" in text
    assert "| made up | 2 |" in text                                  # two cases at the gate
    with open(tmp_path / "table.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    assert T.report(str(tmp_path / "nothing")) == ""


def test_relative_errors_and_the_counts_of_the_gate():
    s = {"property_sd": [2.0, 4.0]}
    assert T.relative(s, lambda p: {"0": 1.0, "1": 1.0}[p]) == pytest.approx(0.375)
    yes = {"mean": -1.0, "low": -2.0, "high": -0.5}
    around = {"mean": 0.0, "low": -1.0, "high": 1.0}
    above = {"mean": 1.0, "low": 0.5, "high": 2.0}
    cases = {"gate": {
        "a": {"ours_minus_pls_as_measured": yes, "ours_minus_pls_with_pds": around,
              "ours_minus_pls_with_pds_more_standards": above,
              "ours_minus_pls_on_the_same_rows": around},
        "b": {"ours_minus_pls_as_measured": yes, "ours_minus_pls_with_pds": above,
              "ours_minus_pls_with_pds_more_standards": above,
              "ours_minus_pls_on_the_same_rows": above},
        "c": {"ours_minus_pls_as_measured": around, "ours_minus_pls_with_pds": yes,
              "ours_minus_pls_with_pds_more_standards": around,
              "ours_minus_pls_on_the_same_rows": around}}}
    c = T._gate_counts(cases)
    assert c == {"cases": 3, "below_pls_as_measured": 2, "within_pds": 1, "below_pds": 1,
                 "above_pds": 1, "above_pds_more_standards": 2,
                 "equal_to_pls_on_the_same_rows": 2}
