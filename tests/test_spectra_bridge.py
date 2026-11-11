"""The spectrum as the bridge between a structure and a spectrum (plan item SP3): the
frequency route of the xtb driver, the data file it gives, and the benchmark's own functions."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

import run_spectra_bridge as BR  # noqa: E402
from cmfqsar import kcca as KC  # noqa: E402
from cmfqsar import spectrum as SP  # noqa: E402
from cmfqsar import xtb as QX  # noqa: E402

HAVE_XTB = True
try:
    QX.find_xtb()
except QX.XtbError:
    HAVE_XTB = False
needs_xtb = pytest.mark.skipif(not HAVE_XTB, reason="xtb is not installed here")

SPECTRA = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir_xtb.json")
STRUCTURES = os.path.join(ROOT, "examples", "spectra", "benzoic_acids.sdf")

VIBSPECTRUM = """$vibrational spectrum
#  mode     symmetry     wave number   IR intensity    selection rules
#                         cm**(-1)      (km*mol-1)        IR
     1                      -0.00         0.00000          -
     2                       0.00         0.00000          -
     3        a           -120.50         3.10000         YES
     4        a           1538.11       131.48210         YES
     5        a1          3651.09         2.61190         YES
     6        b2          3644.20         9.55310         YES
$end
"""


# --------------------------------------------------------------------------- #
# The frequency route of the xtb driver
# --------------------------------------------------------------------------- #
def test_the_spectrum_file_of_xtb_is_read_mode_by_mode(tmp_path):
    path = tmp_path / "vibspectrum"
    path.write_text(VIBSPECTRUM, encoding="utf-8")
    freqs, intens = QX.read_vibspectrum(str(path))
    assert list(freqs) == [-0.0, 0.0, -120.5, 1538.11, 3651.09, 3644.2]
    assert list(intens) == [0.0, 0.0, 3.1, 131.4821, 2.6119, 9.5531]
    empty = tmp_path / "empty"
    empty.write_text("$vibrational spectrum\n$end\n", encoding="utf-8")
    with pytest.raises(QX.XtbError, match="holds no mode"):
        QX.read_vibspectrum(str(empty))
    assert QX.vibration_level(QX.Config()) == "GFN2-xTB acc=1 ohess"
    assert "alpb=water" in QX.vibration_level(QX.Config(solvent="water"))


@needs_xtb
def test_the_normal_modes_of_water(tmp_path):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    AllChem.EmbedMolecule(water, randomSeed=1)
    cfg = QX.Config(workdir=str(tmp_path / "work"), fukui=False)
    v = QX.run_vibrations(water, "water", 1, cfg)
    assert v.error == "" and v.n_imaginary == 0 and not v.reused
    # three atoms that are not on one line have three modes: the bend and two stretches
    assert len(v.wavenumbers) == 3 and np.all(np.diff(v.wavenumbers) >= 0)
    assert 1300 < v.wavenumbers[0] < 1900 and 3300 < v.wavenumbers[1] < 4000
    assert v.coords.shape == (3, 3) and v.level == "GFN2-xTB acc=1 ohess"
    again = QX.run_vibrations(water, "water", 1, cfg)
    assert again.reused and np.array_equal(again.wavenumbers, v.wavenumbers)
    assert np.array_equal(again.intensities, v.intensities)
    both = QX.process_vibrations([water, water], ["a", "b"], cfg, jobs=2)
    assert [x.name for x in both] == ["a", "b"] and all(x.error == "" for x in both)


def test_a_molecule_that_cannot_be_run_is_reported_and_not_fatal(tmp_path):
    from rdkit import Chem

    water = Chem.AddHs(Chem.MolFromSmiles("O"))
    lines = []
    cfg = QX.Config(xtb=str(tmp_path / "no such program.exe"), workdir=str(tmp_path / "work"))
    out = QX.process_vibrations([water], ["water"], cfg, log=lines.append)
    assert len(out) == 1 and out[0].error and out[0].wavenumbers is None
    assert "FAILED" in lines[0] and "water" in lines[0]


# --------------------------------------------------------------------------- #
# The data file
# --------------------------------------------------------------------------- #
def test_the_file_holds_a_second_spectrum_of_every_acid():
    from rdkit import Chem

    with open(SPECTRA, encoding="utf-8") as fh:
        head = json.load(fh)
    rows = {r["name"]: r for r in head["acids"]}
    acids = [m for m in Chem.SDMolSupplier(STRUCTURES, removeHs=False)]
    assert len(rows) == len(acids) == 189
    assert head["frequency_scale"] == [1.0] and head["qm_level"] == ["GFN2-xTB acc=1 ohess"]
    for mol in acids:
        row = rows[mol.GetProp("_Name")]
        x = np.array(row["wavenumbers"])
        assert len(x) == len(row["intensities"]) and np.all(np.diff(x) >= 0) and np.all(x > 0)
        assert len(x) == 3 * mol.GetNumAtoms() - 6 - row["n_imaginary"]
        # every acid has a strong line where the carbonyl stretch of this method lies
        part = [(w, i) for w, i in zip(row["wavenumbers"], row["intensities"])
                if 1650 < w < 1850]
        assert part and max(i for _w, i in part) > 100.0
    assert sum(1 for r in rows.values() if r["n_imaginary"]) == 10
    names, table = BR.read_names()
    assert names == [m.GetProp("_Name") for m in acids]
    assert table["position"][0] == "none" and len(set(table["substituent"])) == 95


@needs_xtb
def test_stage_5_gives_the_stored_spectra(tmp_path):
    import make_band_examples as M

    out = str(tmp_path / "ir_xtb.json")
    lines = []
    failed = M.write_infrared_xtb(STRUCTURES, out, workdir=str(tmp_path / "work"), jobs=2,
                                  limit=4, log=lines.append)
    assert failed == 0 and "4 acids" in lines[-1]
    with open(out, encoding="utf-8") as fh:
        new = json.load(fh)["acids"]
    with open(SPECTRA, encoding="utf-8") as fh:
        stored = json.load(fh)["acids"][:4]
    for a, b in zip(new, stored):
        assert a["name"] == b["name"] and a["n_imaginary"] == b["n_imaginary"]
        assert np.allclose(a["wavenumbers"], b["wavenumbers"], atol=0.5)
        assert np.allclose(a["intensities"], b["intensities"], rtol=0.02, atol=0.05)


# --------------------------------------------------------------------------- #
# Part (b): the acid of a spectrum among the candidates
# --------------------------------------------------------------------------- #
def _library(n=24, seed=0, shared=True):
    """Stick spectra of `n` made-up acids: three lines that all of them share, and three
    lines of their own."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        own_x, own_y = rng.uniform(500.0, 1500.0, 3), rng.uniform(20.0, 60.0, 3)
        x = np.concatenate([[700.0, 1730.0, 3600.0], own_x]) if shared else own_x
        y = np.concatenate([[300.0, 400.0, 200.0], own_y]) if shared else own_y
        out.append(SP.Spectrum(x, y, name=f"acid {k}"))
    return out


def _table(n=24):
    """Twelve substituents, each at two positions; the two acids of one share a formula."""
    return {"substituent": [f"s{k // 2}" for k in range(n)],
            "position": ["meta", "para"] * (n // 2),
            "formula": [f"C{7 + k // 2}H6O2" for k in range(n)]}


def test_the_rank_of_the_right_candidate():
    S = np.array([[0.9, 0.5, 0.1],
                  [0.7, 0.6, 0.8],
                  [0.3, 0.3, 0.3]])
    assert list(BR.ranks(S)) == [1.0, 3.0, 2.0]          # a tie counts half


def test_the_axis_of_a_query_is_mapped_below_and_above_the_cut():
    s = SP.Spectrum([1000.0, 2000.0, 3000.0], [1.0, 2.0, 3.0])
    m = BR.mapped(s, 0.98, 1.05)
    assert np.allclose(m.x, [980.0, 1960.0, 3150.0]) and list(m.y) == [1.0, 2.0, 3.0]
    assert BR.factor_grid("none") == [(1.0, 1.0)]
    assert BR.factor_grid("one factor", one=(0.99, 1.0)) == [(0.99, 0.99), (1.0, 1.0)]
    assert len(BR.factor_grid("two factors", low=(0.98, 0.99), high=(1.04, 1.05, 1.06))) == 6
    assert BR.shown("two factors|50|0.5|0.985|1.040") == \
        "50 cm^-1, power 0.5, factors 0.985 and 1.040"
    assert BR.shown("one factor|5|1|0.990|0.990") == "5 cm^-1, power 1, factor 0.990"
    assert BR.shown("none|20|1|1.000|1.000") == "20 cm^-1, power 1"


def test_a_measured_like_query_is_a_curve_with_noise():
    s = SP.Spectrum([1000.0, 1730.0], [50.0, 400.0])
    rng = np.random.default_rng(0)
    c = BR.measured_like(s, sigma=4.0, step=4.0, noise=0.02, rng=rng)
    assert c.kind == "curve" and c.x[1] - c.x[0] == 4.0
    assert abs(c.x[int(np.argmax(c.y))] - 1730.0) <= 4.0
    quiet = BR.measured_like(s, 4.0, 4.0, 0.0, rng)
    assert np.std(c.y - quiet.y) == pytest.approx(0.02 * quiet.y.max(), rel=0.2)


def test_centring_takes_the_shared_bands_out_of_the_comparison():
    """The two levels put the bands that all acids share at different places.  The shared
    bands of a query then fall on the own lines of other candidates, and the plain kernel
    ranks those candidates first.  Centred spectra hold no shared band."""
    rng = np.random.default_rng(0)
    library, queries = [], []
    for _ in range(40):
        own_x, own_y = rng.uniform(500.0, 1500.0, 3), rng.uniform(20.0, 60.0, 3)
        shared = [300.0, 400.0, 200.0]
        library.append(SP.Spectrum(np.r_[[700.0, 1730.0, 3600.0], own_x], np.r_[shared, own_y]))
        queries.append(SP.Spectrum(np.r_[[900.0, 1100.0, 1300.0], own_x], np.r_[shared, own_y]))
    grid = SP.Grid.for_width(400.0, 4000.0, 30.0)
    lib = SP.feature_matrix(library, grid, 30.0)
    qry = SP.feature_matrix(queries, grid, 30.0)
    cal = np.arange(20)
    plain, centred = BR.similarities(lib, qry), BR.similarities(lib, qry, cal)
    assert plain.shape == centred.shape == (40, 40)
    first_plain = float(np.mean(BR.ranks(plain) == 1))
    first_centred = float(np.mean(BR.ranks(centred) == 1))
    assert first_plain < 0.8 < first_centred


def test_the_scores_of_a_set_of_queries():
    S = np.eye(6) + 0.1
    S[2, 3] = 5.0                                     # query 2 prefers its twin
    twins = {0: 1, 1: 0, 2: 3, 3: 2}
    same = np.zeros((6, 6), bool)
    for i, j in twins.items():
        same[i, i] = same[i, j] = True
    same[4, 4] = same[5, 5] = True
    got = BR.scores(S, np.arange(6), twins, same)
    assert got["top1"] == pytest.approx(5 / 6) and got["top3"] == 1.0
    assert got["isomer"] == pytest.approx(3 / 4)
    assert got["top1_formula"] == pytest.approx(5 / 6) and got["median"] == 1.0
    assert np.isnan(BR.scores(S, np.array([4, 5]), twins)["isomer"])


def test_the_splits_keep_the_two_acids_of_a_substituent_together():
    table = _table()
    cal, test = BR.split_substituents(table, 0)
    assert len(cal) == len(test) == 12 and not set(cal) & set(test)
    subs = np.array(table["substituent"])
    assert not set(subs[cal]) & set(subs[test])
    one, two = BR.halves(cal, table, 0)
    assert sorted(np.concatenate([one, two])) == sorted(cal)
    assert not set(subs[one]) & set(subs[two])
    table["position"] = ["meta", "para"] * 12
    assert BR.twin_rows(table) == {**{2 * k: 2 * k + 1 for k in range(12)},
                                   **{2 * k + 1: 2 * k for k in range(12)}}
    folds = BR.substituent_folds(table, 4, 0)
    assert sorted(np.concatenate(folds)) == list(range(24))
    assert all(len(set(subs[f])) * 2 == len(f) for f in folds)


def test_the_calibration_finds_the_factor_between_two_levels():
    """The queries are the library spectra with every line at 0.97 of its place.  At a narrow
    width nothing matches without a map, and the map with one factor finds its acid."""
    library = _library(shared=False)
    queries = [SP.Spectrum(s.x * 0.97, s.y) for s in library]
    table = _table()
    twins = BR.twin_rows(table)
    formulas = np.array(table["formula"])
    same = formulas[:, None] == formulas[None, :]
    splits = [BR.split_substituents(table, r) for r in range(2)]
    best = BR.best_settings(library, queries, splits, twins, same, table,
                            kinds=("none", "one factor"), centred=(False,), widths=(3.0,),
                            powers=(1.0,), one=(1.0, 1.031, 1.06))
    assert set(best) == {"none", "one factor"}
    for row in best["one factor"]:
        assert row["setting"] == "one factor|3|1|1.031|1.031" and row["top1"] == 1.0
        assert row["isomer"] == 1.0 and row["top1_formula"] == 1.0
    assert all(row["top1"] < 0.5 for row in best["none"])
    chosen = BR.chosen_map(best, ["none", "one factor"])
    assert [row["map"] for row in chosen] == ["one factor"] * 2
    summary = BR.summarise(chosen)
    assert summary["top1"] == {"mean": 1.0, "sd": 0.0}
    assert summary["maps"] == {"one factor": 2} and "inner" not in summary


def test_the_references_of_part_b():
    library = _library()
    queries = [SP.Spectrum(s.x * 1.01, s.y) for s in library]
    table = _table()
    cal, test = BR.split_substituents(table, 0)
    band = BR.band_scores(library, queries, cal, test, BR.twin_rows(table))
    # every made-up acid has its strongest line of the carbonyl region at the same place
    assert band["top1"] == 0.0 and band["median"] == pytest.approx(12.5)
    f = BR.formula_scores(table)
    assert f["top1"] == 0.5 and f["largest_group"] == 2 and f["share_unique"] == 0.0


# --------------------------------------------------------------------------- #
# Part (a): the pairs of directions of two descriptions
# --------------------------------------------------------------------------- #
def _two_views(n=60, seed=0):
    """Two descriptions that share one hidden number and hold noise beside it."""
    rng = np.random.default_rng(seed)
    hidden = rng.normal(size=n)
    A = np.c_[hidden + 0.1 * rng.normal(size=n), rng.normal(size=(n, 3))]
    Bv = np.c_[rng.normal(size=(n, 3)), -hidden + 0.1 * rng.normal(size=n)]
    return A @ A.T, Bv @ Bv.T, hidden, Bv


def test_a_shared_direction_is_found_on_held_out_samples():
    K, L, hidden, _B = _two_views()
    table = {"substituent": [f"s{k // 2}" for k in range(60)]}
    folds = BR.substituent_folds(table, 5, 0)
    ridge, seen = BR.choose_ridge(K, L, folds, 2, grid=(0.01, 0.1, 1.0))
    assert ridge in (0.01, 0.1, 1.0) and len(seen) == 3
    ref = KC.kcca(K, L, ridge)
    U, V, rho = BR.held_out_pairs(K, L, ridge, folds, 2, reference=ref)
    assert U.shape == V.shape == (60, 2) and rho.shape == (5, 2)
    assert np.nanmean(rho[:, 0]) > 0.8
    # with the signs of the folds made one, the held-out variate follows the hidden number
    assert abs(np.corrcoef(U[:, 0], hidden)[0, 1]) > 0.8
    assert abs(np.corrcoef(V[:, 0], hidden)[0, 1]) > 0.8
    # a shuffled pairing shares nothing
    p = np.random.default_rng(1).permutation(60)
    shuffled = BR.held_out_pairs(K, L[np.ix_(p, p)], ridge, folds, 2)[2]
    assert abs(np.nanmean(shuffled[:, 0])) < 0.5 < np.nanmean(rho[:, 0])


def test_the_region_that_carries_a_pair():
    """The second description is a set of spectra whose line in the carbonyl region moves
    with the hidden number.  That region carries the first pair."""
    rng = np.random.default_rng(2)
    n = 60
    hidden = rng.uniform(-1.0, 1.0, n)
    A = np.c_[hidden, 0.3 * rng.normal(size=(n, 2))]
    spectra = [SP.Spectrum(np.concatenate([[1725.0 + 8.0 * h], rng.uniform(450.0, 950.0, 3)]),
                           np.concatenate([[300.0], rng.uniform(50.0, 150.0, 3)]))
               for h in hidden]
    grid = SP.Grid.for_width(*BR.RANGE, 5.0)
    F = SP.feature_matrix(spectra, grid, 5.0)
    Z = BR.unit_rows(F)
    K = A @ A.T
    table = {"substituent": [f"s{k}" for k in range(n)]}
    folds = BR.substituent_folds(table, 5, 0)
    ref = KC.kcca(K, Z @ Z.T, 0.1)
    rows = BR.region_reading(Z, grid.x, K, 0.1, folds, ref, 2)
    labels = [r[2] for r in BR.IR.REGIONS]
    assert len(rows) == len(labels) and len(rows[0]) == 2
    first = [abs(row[0]) for row in rows]
    assert labels[int(np.argmax(first))] == "C=O stretch" and max(first) > 0.6


def test_the_stored_table_is_the_report():
    stored = os.path.join(BR.RESULTS, "table.md")
    if not os.path.isfile(os.path.join(BR.RESULTS, "bridge.json")):
        pytest.skip("the benchmark has not been run")
    with open(stored, encoding="utf-8") as fh:
        before = fh.read()
    assert BR.report() + "\n" == before
    assert "## Reading" in before and "The gate of the item" in before


# --------------------------------------------------------------------------- #
# The second density functional
# --------------------------------------------------------------------------- #
def test_the_file_of_the_second_functional_is_near_the_first():
    from rdkit import Chem

    path = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir_b97-3c.json")
    with open(path, encoding="utf-8") as fh:
        head = json.load(fh)
    with open(os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir.json"),
              encoding="utf-8") as fh:
        first = {r["name"]: r for r in json.load(fh)["acids"]}
    rows = {r["name"]: r for r in head["acids"]}
    acids = [m for m in Chem.SDMolSupplier(STRUCTURES, removeHs=False)]
    assert len(rows) == len(acids) == 189
    assert head["frequency_scale"] == [0.98] and head["qm_level"] == ["B97-3c TightSCF"]
    shifts = []
    for mol in acids:
        row = rows[mol.GetProp("_Name")]
        x = np.array(row["wavenumbers"])
        assert len(x) == len(row["intensities"]) and np.all(np.diff(x) >= 0) and np.all(x > 0)
        assert len(x) in (3 * mol.GetNumAtoms() - 6, 3 * mol.GetNumAtoms() - 7)
        other = np.array(first[row["name"]]["wavenumbers"])
        if len(other) == len(x):
            shifts.append(np.abs(x - other))
    # two density functionals put the lines of an acid at nearly the same places
    assert len(shifts) > 180 and np.median(np.concatenate(shifts)) < 10.0


def test_the_levels_of_the_queries_and_their_factors():
    import make_band_examples as M

    assert M.level_slug("B97-3c TightSCF") == "b97-3c" and M.level_slug("") == "level"
    assert set(BR.LEVELS) == {"xtb", "b97-3c", "xtb, ORCA geometry"}
    for cfg in BR.LEVELS.values():
        assert {"path", "label", "about", "one", "low", "high", "curve_widths"} <= set(cfg)
    dft = BR.LEVELS["b97-3c"]
    assert os.path.basename(dft["path"]) == "benzoic_acids_ir_b97-3c.json"
    assert 1.0 in dft["one"] and 1.0 in dft["low"] and 1.0 in dft["high"]
    # the control is computed when it is run, and its spectra are not kept in the repository
    control = BR.LEVELS["xtb, ORCA geometry"]
    assert os.path.normpath(control["path"]).startswith(os.path.normpath(BR.CONTROL_CACHE))
    assert control["one"] == BR.LEVELS["xtb"]["one"]
