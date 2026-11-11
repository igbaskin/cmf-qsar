"""Measured spectra: the JCAMP-DX reader, the absorbance, the slope of a spectrum, the script
that prepares the Chemotion IR set, and the benchmark of a measured spectrum against other
measured spectra.  No test needs the set itself: the files are made here."""
import io
import json
import lzma
import os
import sys
import tarfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import prepare_chemotion_ir as PREP  # noqa: E402
import run_spectra_measured as M  # noqa: E402
from cmfqsar import spectrum as SP  # noqa: E402


# --------------------------------------------------------------------------- #
# A writer of the compressed forms, independent of the reader
# --------------------------------------------------------------------------- #
def _lettered(value: int, positive: str, negative: str) -> str:
    """A whole number with its sign and first digit as one letter."""
    digits = str(abs(value))
    head = positive[int(digits[0])] if value >= 0 else negative[int(digits[0]) - 1]
    return head + digits[1:]


def _squeezed(value: int) -> str:
    return _lettered(value, "@ABCDEFGHI", "abcdefghi")


def _difference(value: int) -> str:
    return _lettered(value, "%JKLMNOPQR", "jklmnopqr")


def _repeat(count: int) -> str:
    digits = str(count)
    return "STUVWXYZs"[int(digits[0]) - 1] + digits[1:]


def difdup_lines(values, per_line=6):
    """The lines of an (X++(Y..Y)) table in the difference form with repeat counts.  Every
    line starts with the number of its first point and that point as a squeezed value; the
    last point of a line is given again at the start of the next line, as the check that the
    format asks for, and a last line holds only this check."""
    lines, start = [], 0
    values = [int(v) for v in values]
    while start < len(values) - 1:
        chunk = values[start:start + per_line]
        words = [_squeezed(chunk[0])]
        diffs = [b - a for a, b in zip(chunk, chunk[1:])]
        i = 0
        while i < len(diffs):
            n = 1
            while i + n < len(diffs) and diffs[i + n] == diffs[i]:
                n += 1
            words.append(_difference(diffs[i]) + (_repeat(n) if n > 1 else ""))
            i += n
        lines.append(str(start) + "".join(words))
        start += len(chunk) - 1
    lines.append(str(start) + _squeezed(values[-1]))
    return lines


def jcamp_text(values, first_x=4000.0, last_x=600.0, yunits="TRANSMITTANCE", yfactor=0.001,
               lines=None, title="a test spectrum"):
    head = [f"##TITLE={title}", "##JCAMP-DX=5.0", "##DATA TYPE=INFRARED SPECTRUM",
            "##XUNITS=1/CM", f"##YUNITS={yunits}", "##XFACTOR=1.0", f"##YFACTOR={yfactor}",
            f"##FIRSTX={first_x}", f"##LASTX={last_x}", f"##NPOINTS={len(values)}",
            "##XYDATA=(X++(Y..Y))"]
    return "\n".join(head + (lines or difdup_lines(values)) + ["##END="]) + "\n"


# --------------------------------------------------------------------------- #
# The reader
# --------------------------------------------------------------------------- #
def test_the_compressed_forms_give_back_the_numbers_that_were_written():
    rng = np.random.default_rng(0)
    values = np.concatenate([[950, 950, 950, 950, 951, 952, 953, -4, -4, 0, 0, 12],
                             rng.integers(-2000, 2000, 300), [7, 7, 7]])
    for per_line in (2, 5, 9, 400):
        got = SP.decode_ordinates(difdup_lines(values, per_line))
        assert np.array_equal(got, values), per_line


def test_squeezed_values_with_a_point_counter_as_the_chemotion_files_have_them():
    """One squeezed value after the other; the letters E and e are the digits 5 and -5."""
    lines = ["0E12e3@A0", "4I99999a"]
    assert np.array_equal(SP.decode_ordinates(lines), [512, -53, 0, 10, 999999, -1])


def test_plain_numbers_with_exponents_are_read_as_numbers():
    lines = ["4000.0 1.5E-1 2.0e+0 -3", "3997.0, 4 5.25"]
    assert np.allclose(SP.decode_ordinates(lines), [0.15, 2.0, -3.0, 4.0, 5.25])


def test_a_file_with_several_blocks_gives_the_curve_and_the_peak_table():
    values = (1000 * (0.9 - 0.5 * np.exp(-0.5 * ((np.arange(120) - 60) / 4.0) ** 2)))
    text = ("##TITLE=link\n##JCAMP-DX=5.0\n##DATA TYPE=LINK\n##BLOCKS=2\n"
            + jcamp_text(np.round(values), 3000.0, 2762.0)
            + "##TITLE=peaks\n##DATA TYPE=INFRARED PEAK TABLE\n##XUNITS=1/CM\n"
              "##PEAKTABLE=(XY..XY)\n2880.0, 0.4; 2990.5, 0.9\n##END=\n##END=\n")
    blocks = SP.parse_jcamp(text)
    assert [b["kind"] for b in blocks] == ["curve", "sticks"]
    curve, sticks = blocks
    assert curve["labels"]["YUNITS"] == "TRANSMITTANCE"
    assert np.allclose(curve["x"], np.linspace(3000.0, 2762.0, 120))
    assert np.allclose(curve["y"], np.round(values) * 0.001)
    assert np.allclose(sticks["x"], [2880.0, 2990.5]) and np.allclose(sticks["y"], [0.4, 0.9])
    assert "YUNITS" not in sticks["labels"]            # a new title starts new labels


def test_read_jcamp_gives_a_spectrum_sorted_by_wavenumber(tmp_path):
    values = np.round(1000 * np.linspace(0.2, 0.9, 50))
    path = tmp_path / "one.dx"
    path.write_text(jcamp_text(values, 4000.0, 3902.0), encoding="utf-8")
    s = SP.read_jcamp(str(path))
    assert s.kind == "curve" and s.axis == "wavenumber" and s.name == "a test spectrum"
    assert s.meta["yunits"] == "TRANSMITTANCE" and s.meta["xunits"] == "1/CM"
    assert s.x[0] == 3902.0 and s.x[-1] == 4000.0      # the file runs downwards
    assert np.allclose(s.y, values[::-1] * 0.001)
    empty = tmp_path / "empty.dx"
    empty.write_text("##TITLE=nothing\n##END=\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no table of data"):
        SP.read_jcamp(str(empty))


def test_absorbance_from_a_transmittance_as_a_fraction_and_in_per_cent():
    x = np.linspace(600.0, 4000.0, 5)
    t = np.array([1.0, 0.5, 0.1, 0.01, 0.0])
    want = np.array([0.0, np.log10(2.0), 1.0, 2.0, 6.0])      # zero is cut at 1e-6
    a = SP.absorbance(SP.Spectrum(x, t, "curve", meta={"yunits": "TRANSMITTANCE"}))
    assert np.allclose(a.y, want) and a.meta["yunits"] == "ABSORBANCE"
    b = SP.absorbance(SP.Spectrum(x, 100 * t, "curve", meta={"yunits": "% TRANSMITTANCE"}))
    assert np.allclose(b.y, want)
    c = SP.absorbance(SP.Spectrum(x, 100 * t, "curve", meta={"yunits": "TRANSMITTANCE"}))
    assert np.allclose(c.y, want)                             # the values say per cent
    same = SP.Spectrum(x, t, "curve", meta={"yunits": "Absorbance"})
    assert SP.absorbance(same) is same
    with pytest.raises(ValueError, match="neither a transmittance nor an absorbance"):
        SP.absorbance(SP.Spectrum(x, t, "curve", meta={"yunits": "KUBELKA-MUNK"}))
    with pytest.raises(ValueError):
        SP.absorbance(SP.Spectrum(x, t, "curve"))             # no unit is not a licence


# --------------------------------------------------------------------------- #
# Broadening a curve, and the slope
# --------------------------------------------------------------------------- #
def _curve(seed=0, offset=0.0, n=900):
    rng = np.random.default_rng(seed)
    x = 600.0 + 2.0 * np.arange(n)
    y = np.zeros(n)
    for c, h, w in zip(rng.uniform(700, 2300, 12), rng.uniform(0.1, 1.0, 12),
                       rng.uniform(4, 15, 12)):
        y += h * np.exp(-0.5 * ((x - c) / w) ** 2)
    return SP.Spectrum(x, y + offset, "curve")


def test_the_convolution_of_an_even_curve_is_the_sum_over_its_points():
    s = _curve()
    grid = SP.Grid(650.0, 2350.0, 3.0)
    for sigma in (4.0, 9.0, 30.0):
        fast = SP.broaden(s, grid, sigma)
        x, w = s.lines()
        d = grid.x[:, None] - x[None, :]
        exact = (np.exp(-0.5 * d * d / sigma ** 2) @ w) / (sigma * np.sqrt(2 * np.pi))
        assert np.max(np.abs(fast - exact)) < 0.01 * np.max(exact), sigma
    # an uneven curve and a width below two steps take the sum itself
    uneven = SP.Spectrum(s.x + 0.4 * np.sin(s.x), s.y, "curve")
    x, w = uneven.lines()
    d = grid.x[:, None] - x[None, :]
    exact = (np.exp(-0.5 * d * d / 81.0) @ w) / (9.0 * np.sqrt(2 * np.pi))
    assert np.allclose(SP.broaden(uneven, grid, 9.0), exact)


def test_the_dot_product_of_two_slopes_is_the_closed_form():
    rng = np.random.default_rng(3)
    a = SP.Spectrum(rng.uniform(900, 1700, 9), rng.uniform(0.2, 2.0, 9))
    b = SP.Spectrum(rng.uniform(900, 1700, 7), rng.uniform(0.2, 2.0, 7))
    sigma = 10.0
    grid = SP.Grid.for_width(700.0, 1900.0, sigma, points_per_sigma=20)
    F = SP.slope(SP.feature_matrix([a, b], grid, sigma), grid)
    scale = 2.0 * sigma * np.sqrt(np.pi) * grid.step
    for p, q, u, v in ((a, b, 0, 1), (a, a, 0, 0), (b, b, 1, 1)):
        assert scale * (F[u] @ F[v]) == pytest.approx(SP.slope_overlap(p, q, sigma), rel=2e-3)
    # one stick against one stick: the same place, 1.41 widths apart, and further apart
    one = SP.Spectrum([1000.0], [1.0])
    at = lambda d: SP.slope_overlap(one, SP.Spectrum([1000.0 + d], [1.0]), sigma)  # noqa: E731
    assert at(0.0) == pytest.approx(1.0 / (2.0 * sigma ** 2))
    assert at(np.sqrt(2.0) * sigma) == pytest.approx(0.0, abs=1e-12)
    assert at(3.0 * sigma) < 0
    with pytest.raises(ValueError):
        SP.slope_overlap(one, one, 0.0)
    with pytest.raises(ValueError):
        SP.slope(np.zeros((3, 1)), grid)


def test_the_slope_does_not_see_a_constant_baseline_and_the_plain_kernel_does():
    low, high = _curve(1), _curve(1, offset=0.5)
    other = _curve(2)
    x = low.x
    A = np.vstack([low.y, high.y, other.y])
    grid = SP.Grid(650.0, 2350.0, 2.0)                      # ends 50 cm^-1 inside the curves
    F = M.broadened(A, x, 8.0, grid)
    slope = M.formed(F, "slope", grid)
    assert slope[0] @ slope[1] == pytest.approx(1.0, abs=1e-6)
    centred = M.formed(F, "own mean removed", grid)
    assert centred[0] @ centred[1] == pytest.approx(1.0, abs=1e-6)
    plain = M.formed(F, "plain", grid)
    assert plain[0] @ plain[1] < 0.9
    assert plain[1] @ plain[2] > plain[0] @ plain[2]        # the baseline is counted as likeness
    for rows in (slope, centred, plain):
        assert np.allclose(np.einsum("ij,ij->i", rows, rows), 1.0)
    with pytest.raises(ValueError, match="form must be one of"):
        M.formed(F, "curvature", grid)


def test_a_drifting_baseline_costs_the_slope_kernel_less_than_the_other_forms():
    s = _curve(4)
    drift = 0.3 + 0.4 * (s.x - s.x[0]) / (s.x[-1] - s.x[0])
    A = np.vstack([s.y, s.y + drift])
    grid = SP.Grid(650.0, 2350.0, 2.0)
    F = M.broadened(A, s.x, 4.0, grid)
    like = {form: float(np.prod(M.formed(F, form, grid), axis=0).sum()) for form in M.FORMS}
    assert like["slope"] > 0.999 > like["own mean removed"] > like["plain"]


# --------------------------------------------------------------------------- #
# The script that prepares the set
# --------------------------------------------------------------------------- #
def _add(tar, name, data: bytes):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _archive(path, meta, files):
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w") as tar:
        for name, text in files.items():
            _add(tar, "exp/" + name, text.encode("utf-8"))
    with tarfile.open(path, "w") as tar:
        _add(tar, "bag/data/dataset/meta_data.json", json.dumps(meta).encode("utf-8"))
        _add(tar, "bag/data/dataset/JCAMP-DX Files/IR_data.tar.xz",
             lzma.compress(inner.getvalue()))
        _add(tar, "bag/README.md", b"not read")


def _entry(sample, smiles, identifier, kind="CHMO:0000630 | infrared absorption spectroscopy (IR)",
           filename="IR.dx"):
    return {"sample_id": sample, "analysis_id": 10 * sample, "cano_smiles": smiles,
            "kind": kind, "content": "IR (ATR): 1700 cm-1",
            "datasets": [{"instrument": "Bruker ALPHA",
                          "attacments": [{"filename": filename, "identifier": identifier}]}]}


def test_prepare_reads_an_archive_without_writing_its_files(tmp_path):
    n = 1701
    half = np.full(n, 500)                                   # a transmittance of 0.5
    files = {
        "aaa": jcamp_text(half, 4000.0, 600.0),
        "bbb": jcamp_text(np.full(n, 0.25), 600.0, 4000.0, yunits="ABSORBANCE", yfactor=1.0,
                          lines=[f"{i} 0.25 0.25 0.25" for i in range(n // 3)]),
        "ccc": "##TITLE=only peaks\n##XUNITS=1/CM\n##PEAKTABLE=(XY..XY)\n1700, 1\n##END=\n",
        "ddd": jcamp_text(half, 800.0, 200.0).replace("1/CM", "NANOMETERS"),
        "eee": jcamp_text(half[:400], 4000.0, 3202.0),       # does not cover the grid
        "fff": jcamp_text(half, 4000.0, 600.0, yunits="KUBELKA-MUNK"),
    }
    meta = [_entry(1, "CCO", "7/aaa"), _entry(2, "CCN", "7/bbb"), _entry(3, "CCC", "7/ccc"),
            _entry(4, "CCCl", "7/ddd", kind="CHMO:0000804 | UV-VIS-NIR absorption"),
            _entry(5, "CCBr", "7/eee"), _entry(6, "CCF", "7/fff"),
            _entry(7, "CCI", "7/missing")]
    archive = tmp_path / "download" / "set.tar"
    archive.parent.mkdir()
    _archive(str(archive), meta, files)
    cache = tmp_path / "cache"
    with pytest.raises(SystemExit, match="SHA-256"):
        PREP.prepare(str(archive), str(cache))                # not the published set
    done = PREP.prepare(str(archive), str(cache), check=False)
    assert done["entries"] == 7 and done["kept"] == 2 and done["compounds"] == 2
    assert done["left_out"] == {"no curve in the file": 1, "not an infrared spectrum": 1,
                                "the curve does not cover the grid": 1,
                                "the intensity is in KUBELKA-MUNK": 1,
                                "no file of the measurement was found": 1}
    assert sorted(os.listdir(cache)) == ["spectra.npz", "table.json"]
    grid, A, rows = PREP.load(str(cache))
    assert grid[0] == PREP.GRID[0] and grid[-1] == PREP.GRID[1] and A.shape == (2, len(grid))
    assert np.allclose(A[0], np.log10(2.0), atol=1e-6) and np.allclose(A[1], 0.25)
    assert [r["smiles"] for r in rows] == ["CCO", "CCN"]
    assert rows[0]["instrument"] == "Bruker ALPHA" and rows[0]["atr"] is True
    assert rows[0]["step"] == pytest.approx(2.0)
    with pytest.raises(SystemExit, match="prepare_chemotion_ir.py"):
        PREP.load(str(tmp_path / "nothing"))


def test_an_archive_without_the_data_files_is_refused(tmp_path):
    archive = tmp_path / "other.tar"
    with tarfile.open(archive, "w") as tar:
        _add(tar, "bag/README.md", b"nothing else")
    with pytest.raises(SystemExit, match="does not hold"):
        PREP.members(str(archive), check=False)


# --------------------------------------------------------------------------- #
# The benchmark
# --------------------------------------------------------------------------- #
def test_copies_within_a_compound_and_across_compounds():
    rng = np.random.default_rng(5)
    base = rng.uniform(0, 1, size=(6, 40))
    #        0        1        2        3        4        5        6        7
    A = np.vstack([base[0], base[0], base[1], base[2], base[2], base[3], base[4], base[0]])
    compounds = ["a", "a", "a", "b", "c", "d", "d", "a"]
    assert sorted(M.identical_sets(A)) == [[0, 1, 7], [3, 4]]
    keep, left = M.without_copies(A, compounds)
    assert keep.tolist() == [True, False, True, False, False, True, True, False]
    assert left == {"copies_within_a_compound": 2,
                    "sets_shared_by_different_compounds": [[3, 4]]}
    assert M.identical_sets(base) == []
    near = np.vstack([base[0], base[0] + 1e-3])
    assert M.identical_sets(near) == []                      # alike is not the same
    assert M.grouped(["x", "y", "x", "z", "y"]) == [[0, 2], [1, 4], [3]]


def _unit(rows):
    rows = np.asarray(rows, float)
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_pair_ranks_puts_the_target_into_the_library_in_place_of_the_first_spectrum():
    # compound 0: rows 0, 1, 2; compound 1: row 3; compound 2: rows 4, 5
    U = _unit([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.2, 1.0, 0.0],
               [0.3, 0.9, 0.1],
               [0.0, 0.1, 1.0], [0.0, 0.0, 1.0]])
    groups = [[0, 1, 2], [3], [4, 5]]
    r = M.pair_ranks(U, groups)
    pairs = list(zip(r["query"].tolist(), r["target"].tolist()))
    assert pairs == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (4, 5), (5, 4)]
    assert r["compound"].tolist() == [0] * 6 + [2] * 2
    # the query 0 finds row 1 at once; its other target, row 2, lies behind compound 1
    assert r["rank"].tolist() == [1, 2, 1, 2, 2, 2, 1, 1]
    assert r["named"][0] == 1 and r["named"][1] == 3 and r["named"][4] == 3
    # a library spectrum as similar as the target counts as better
    tie = _unit([[1.0, 0.0], [0.6, 0.8], [0.6, 0.8]])
    assert M.pair_ranks(tie, [[0, 1], [2]])["rank"][0] == 2
    assert M.pair_ranks(tie, [[0, 1], [2]])["named"][0] == 2
    ids, means = M.per_compound(np.array([1.0, 0, 0, 0, 0, 0, 1, 1]), r["compound"])
    assert ids.tolist() == [0, 2] and np.allclose(means, [1 / 6, 1.0])
    s = M.scores(r["rank"], r["compound"])
    assert np.allclose(s["first"], [2 / 6, 1.0]) and np.allclose(s["top3"], [1.0, 1.0])
    assert np.allclose(s["rr"], [(2 + 4 * 0.5) / 6, 1.0])
    unit = _unit(U - U.mean(axis=1, keepdims=True))
    far = M.misses(r, unit, beyond=1)
    assert far["pairs"] == 4 and far["compounds"] == 1 and far["beyond"] == 1
    assert M.misses(r, unit, beyond=5)["correlation_median"] is None


def test_the_choice_of_a_setting_is_made_on_one_half_and_scored_on_the_other():
    rng = np.random.default_rng(8)
    n = 40
    good = {"first": np.ones(n), "top3": np.ones(n), "top10": np.ones(n), "rr": np.ones(n)}
    bad = {k: np.zeros(n) for k in good}
    mixed = {k: (rng.uniform(size=n) < 0.5).astype(float) for k in good}
    per = {"plain|0": bad, "slope|4": good, "slope|8": mixed}
    fit = M.nested(per, list(per), repeats=20, seed=1)
    assert fit["chosen"] == {"slope|4": 20} and fit["first"] == 1.0 and fit["first_sd"] == 0.0
    assert all(fit["scored"]) and np.allclose(fit["per_compound_first"], 1.0)
    only = M.nested(per, ["plain|0", "slope|8"], repeats=20, seed=1)
    assert only["chosen"] == {"slope|8": 20}
    assert only["first"] == pytest.approx(np.mean(mixed["first"]), abs=0.1)
    assert M.nested(per, list(per), repeats=20, seed=1) == fit
    d = M.mean_difference(fit["per_compound_first"], only["per_compound_first"])
    assert d["low"] <= d["mean"] <= d["high"] and d["low"] > 0
    zero = M.mean_difference(np.ones(9), np.ones(9))
    assert zero == {"mean": 0.0, "low": 0.0, "high": 0.0}
    assert M.shown("slope|4") == "slope, width 4 cm^-1"
    assert M.shown("plain|0") == "plain, not broadened"
    assert M.setting_name("own mean removed", 16.0) == "own mean removed|16"


def _measured_set(n_compounds=60, n_twice=24, seed=11):
    """Spectra of made-up compounds on the grid of the prepared set.  Every compound has its
    own bands; the second sample of a compound has the same bands with other heights, a
    baseline of its own and noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(PREP.GRID[0], PREP.GRID[1] + 1e-9, PREP.GRID[2])
    rows, smiles = [], []
    for c in range(n_compounds):
        centres = rng.uniform(750, 3700, 14)
        heights = rng.uniform(0.05, 0.6, 14)
        widths = rng.uniform(5, 14, 14)
        for sample in range(2 if c < n_twice else 1):
            h = heights * rng.uniform(0.8, 1.25, 14)
            y = sum(hi * np.exp(-0.5 * ((x - ci) / wi) ** 2)
                    for ci, hi, wi in zip(centres, h, widths))
            y = y + rng.uniform(0.0, 0.6) + rng.uniform(-0.3, 0.3) * (x - x[0]) / 3250.0
            rows.append(y + 0.004 * rng.normal(size=len(x)))
            smiles.append("C" * (c + 1) + "O")
    return x, np.vstack(rows), smiles


def test_on_spectra_with_a_baseline_the_slope_form_finds_more_than_the_plain_form():
    x, A, smiles = _measured_set()
    groups = M.grouped(smiles)
    F = M.broadened(A, x, 4.0)
    first = {}
    for form in M.FORMS:
        r = M.pair_ranks(M.formed(F, form), groups)
        first[form] = float(np.mean(M.scores(r["rank"], r["compound"])["first"]))
    assert first["slope"] > 0.95
    assert first["slope"] > first["own mean removed"] > first["plain"]


def test_the_benchmark_runs_on_a_made_up_set_and_writes_its_table(tmp_path, monkeypatch):
    x, A, smiles = _measured_set()
    A = np.vstack([A, A[0], A[30]])                # a copy within a compound, and one across
    smiles = smiles + [smiles[0], "c1ccccc1"]
    table = [{"smiles": s, "sample_id": 100 + i} for i, s in enumerate(smiles)]
    monkeypatch.setattr(M.PREP, "load", lambda: (x, A, table))
    out = M.run(str(tmp_path), repeats=6)
    assert out["copies_within_a_compound"] == 1
    assert out["sample_ids_of_curves_shared_by_different_compounds"] == [[130, 185]]
    assert out["spectra"] == len(smiles) - 3 and out["compounds"] == 60
    assert out["compounds_with_several_spectra"] == 23
    assert out["spectra_per_compound"] == {"2": 23} and out["pairs"] == 46
    assert set(out["nested"]) == {"all forms", "plain", "own mean removed", "slope"}
    assert out["nested"]["all forms"]["first"] > out["nested"]["plain"]["first"]
    assert out["chosen_minus_form"]["plain"]["low"] > 0
    assert out["setting_of_the_readings"].startswith("slope|")
    assert set(out["regions"]) == {name for name, _, _ in M.REGIONS}
    assert out["chance"] == pytest.approx(1 / 60)
    assert out["composition_alone"] == 1.0         # every made-up compound has its own formula
    assert out["nearest_other_compound"]["n"] == 60
    text = M.report(str(tmp_path))
    assert text.startswith("# A measured infrared spectrum against other measured spectra")
    for part in ("## How often the compound of the query is found",
                 "## Every setting on all compounds", "## Reading",
                 "kernel of the slopes of the spectra"):
        assert part in text
    assert "|4" not in text and "|0" not in text  # no setting key inside a table cell
    with open(tmp_path / "table.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    assert M.report(str(tmp_path / "nothing")) == ""
