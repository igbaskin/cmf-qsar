"""A measured spectrum against a library of computed spectra: the script that computes the
library with xtb, and the benchmark's own functions.  No test needs the Chemotion IR set; the
spectra are made here."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import prepare_chemotion_xtb as XTB  # noqa: E402
import run_spectra_computed as C  # noqa: E402
from cmfqsar import spectrum as SP  # noqa: E402
from cmfqsar import xtb as QX  # noqa: E402

HAVE_XTB = True
try:
    QX.find_xtb()
except QX.XtbError:
    HAVE_XTB = False
needs_xtb = pytest.mark.skipif(not HAVE_XTB, reason="xtb is not installed here")

PARACYCLOPHANE = "CC(=O)c1cc2CCc3ccc(CCc1cc2)cc3"


# --------------------------------------------------------------------------- #
# Which compounds get a computed spectrum, and their start geometry
# --------------------------------------------------------------------------- #
def test_a_compound_is_usable_when_it_is_one_neutral_organic_molecule():
    mol, why = XTB.usable("CCO")
    assert mol is not None and why == ""
    for smiles, part in (("CC(=O)[O-].[Na+]", "more than one molecule"),
                         ("C[Pt]Cl", "an element outside the list"),
                         ("C[N+](C)(C)C", "an ion"),
                         ("C[CH2]", "unpaired electrons"),
                         ("not a smiles", "not read")):
        mol, why = XTB.usable(smiles)
        assert mol is None and part in why, smiles
    mol, why = XTB.usable("C" * 40)                      # 122 atoms with the hydrogens
    assert mol is None and why == "more than 100 atoms"
    assert XTB.usable("C" * 40, max_atoms=130)[0] is not None
    assert XTB.usable("C[N+](=O)[O-]")[0] is not None    # charged atoms, no net charge
    rows = [{"smiles": "CCO"}, {"smiles": "CC"}, {"smiles": "CCO"}, {"smiles": "C"}]
    assert XTB.compounds(rows) == ["C", "CC", "CCO"]


def test_the_start_geometry_has_all_hydrogens_and_one_conformer():
    from rdkit import Chem

    mol = Chem.MolFromSmiles("OC(=O)c1ccccc1")
    start = XTB.start_geometry(mol)
    assert start.GetNumAtoms() == 15 and start.GetNumConformers() == 1
    assert sum(a.GetIsAromatic() for a in start.GetAtoms()) == 6   # the flags of RDKit
    again = XTB.start_geometry(mol)
    assert np.allclose(start.GetConformer().GetPositions(),
                       again.GetConformer().GetPositions())        # the same seed


def test_a_strained_ring_system_gets_a_geometry_without_a_long_wait():
    import time
    from rdkit import Chem

    t0 = time.time()
    start = XTB.start_geometry(Chem.MolFromSmiles(PARACYCLOPHANE))
    assert time.time() - t0 < 60.0
    assert start.GetNumConformers() == 1 and start.GetNumAtoms() == 37
    xyz = start.GetConformer().GetPositions()
    d = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    assert d[np.triu_indices(len(xyz), 1)].min() > 0.7             # no two atoms on one place


def test_a_start_geometry_is_made_once_and_kept(tmp_path):
    work = str(tmp_path)
    assert XTB.make_start((7, "CCO", work)) == (7, "")
    path = XTB.start_path(7, work)
    assert os.path.isfile(path) and path.endswith(os.path.join("start", "00007.mol"))
    with open(path) as fh:
        first = fh.read()
    with open(path, "w") as fh:
        fh.write(first.replace("RDKit", "kept "))                  # a mark in the header
    assert XTB.make_start((7, "CCO", work)) == (7, "")
    with open(path) as fh:
        assert "kept " in fh.read()                                # not made again
    index, error = XTB.make_start((8, "not a smiles", work))
    assert index == 8 and error and not os.path.isfile(XTB.start_path(8, work))
    errors = XTB.make_starts([(7, "CCO"), (8, "not a smiles"), (9, "CC")], work, jobs=1)
    assert list(errors) == [8] and os.path.isfile(XTB.start_path(9, work))
    row = XTB.one_spectrum(8, "not a smiles", None, work)
    assert row["error"] == "the start geometry is not read"


@needs_xtb
def test_xtb_gives_the_spectrum_of_a_small_molecule(tmp_path):
    work = str(tmp_path / "work")
    assert XTB.make_start((1, "CO", work)) == (1, "")
    cfg = QX.Config(xtb=QX.find_xtb(None), fukui=False, workdir=work, timeout=300.0)
    row = XTB.one_spectrum(1, "CO", cfg, work)
    assert "error" not in row, row
    assert row["atoms"] == 6 and len(row["wavenumbers"]) == 12     # 3 N - 6 modes
    assert row["n_imaginary"] == 0 and row["wavenumbers"] == sorted(row["wavenumbers"])
    assert 3000 < row["wavenumbers"][-1] < 4000                    # the O-H stretch
    again = XTB.one_spectrum(1, "CO", cfg, work)
    assert again["reused"] and again["wavenumbers"] == row["wavenumbers"]
    with pytest.raises(SystemExit, match="prepare_chemotion_xtb.py"):
        XTB.load(str(tmp_path / "nothing.json"))


# --------------------------------------------------------------------------- #
# Spectra on the grid
# --------------------------------------------------------------------------- #
def _sticks(seed, n=40):
    rng = np.random.default_rng(seed)
    return np.sort(rng.uniform(500.0, 3900.0, n)), rng.uniform(0.5, 50.0, n)


def test_sticks_shared_between_grid_points_and_broadened_are_the_broadened_sticks():
    sticks = [_sticks(1), _sticks(2)]
    grid = SP.Grid(C.RANGE[0], C.RANGE[1], C.STEP)
    for width, tolerance in ((4.0, 0.04), (8.0, 0.01), (32.0, 0.001)):
        rows = C.cut(C.smooth(C.deposit(sticks), width, "sticks"), C.WIDE[0], *C.RANGE)
        exact = SP.feature_matrix([SP.Spectrum(x, y) for x, y in sticks], grid, width)
        assert rows.shape == exact.shape
        # the grid step is the area of one grid point
        assert np.max(np.abs(rows / C.STEP - exact)) < tolerance * np.max(exact), width


def test_deposit_keeps_the_intensity_and_applies_the_factor_the_power_and_the_depth():
    x, y = np.array([1000.0, 1001.0, 2000.5]), np.array([4.0, 2.0, 9.0])
    D = C.deposit([(x, y)])
    first = C.WIDE[0]
    at = lambda v: int(round((v - first) / C.STEP))                 # noqa: E731
    assert D.sum() == pytest.approx(15.0)
    assert D[0, at(1000.0)] == pytest.approx(4.0 + 1.0) and D[0, at(1002.0)] == pytest.approx(1.0)
    assert D[0, at(2000.0)] == pytest.approx(6.75) and D[0, at(2002.0)] == pytest.approx(2.25)
    moved = C.deposit([(x, y)], factor=0.5)
    assert moved[0, at(500.0)] == pytest.approx(4.0 + 1.5)          # 500 and 500.5
    root = C.deposit([(np.array([1000.0]), np.array([9.0]))], power=0.5)
    assert root.sum() == pytest.approx(3.0)
    deep = C.deposit([(np.array([500.0, 2000.0]), np.array([1.0, 1.0]))], depth=True)
    assert deep[0, at(500.0)] == pytest.approx(2.0) and deep[0, at(2000.0)] == pytest.approx(0.5)
    scaled = C.deposit([(np.array([2000.0]), np.array([1.0]))], factor=0.5, depth=True)
    assert scaled[0, at(1000.0)] == pytest.approx(1.0)              # the depth of the new place
    outside = C.deposit([(np.array([100.0, 9000.0, 1000.0]), np.array([5.0, 5.0, 1.0]))])
    assert outside.sum() == pytest.approx(1.0)


def test_a_measured_curve_keeps_its_baseline_at_the_ends_and_sticks_do_not():
    flat = np.full((1, 400), 0.3)
    assert np.allclose(C.smooth(flat, 32.0, "curve"), 0.3)
    assert C.smooth(flat, 32.0, "sticks")[0, 0] < 0.2
    A = np.array([[0.5, 0.2, 0.9], [-0.1, 0.3, 0.3]])
    assert C.powered_curves(A, 1.0) is not None and np.array_equal(C.powered_curves(A, 1.0), A)
    assert np.allclose(C.powered_curves(A, 0.5), np.sqrt([[0.3, 0.0, 0.7], [0.0, 0.4, 0.4]]))


def test_the_forms_on_a_region_are_rows_of_unit_length():
    rng = np.random.default_rng(3)
    x = np.arange(650.0, 3900.0 + 1e-9, 2.0)
    curve = sum(h * np.exp(-0.5 * ((x - c) / 9.0) ** 2)
                for c, h in zip(rng.uniform(800, 3600, 10), rng.uniform(0.1, 1, 10)))
    rows = np.vstack([curve, curve + 0.4])
    for form in C.FORMS:
        U = C.formed(rows, form, 650.0, 700.0, 1800.0)
        assert U.shape == (2, 551) and np.allclose(np.einsum("ij,ij->i", U, U), 1.0)
    assert C.formed(rows, "slope", 650.0, 700.0, 1800.0)[0] == pytest.approx(
        C.formed(rows, "slope", 650.0, 700.0, 1800.0)[1])           # no baseline in a slope
    mean_off = C.formed(rows, "own mean removed", 650.0, 700.0, 3850.0)
    assert mean_off[0] @ mean_off[1] == pytest.approx(1.0)
    plain = C.formed(rows, "plain", 650.0, 700.0, 3850.0)
    assert plain[0] @ plain[1] < 0.9
    with pytest.raises(ValueError, match="form must be one of"):
        C.formed(rows, "curvature", 650.0, 700.0, 1800.0)
    assert np.array_equal(C.cut(np.arange(10.0)[None, :], 100.0, 104.0, 110.0)[0], [2, 3, 4, 5])
    Z = C.centred(np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]))
    assert np.allclose(np.einsum("ij,ij->i", Z, Z), 1.0)


def test_ranks_count_a_tie_against_the_right_compound():
    L = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.6, 0.8]])
    Q = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6]])
    rank, top = C.ranks(Q, L, np.array([0, 2, 2, 1]))
    assert rank.tolist() == [1, 3, 2, 4] and top.tolist()[:2] == [0, 1]
    key = C.setting_name("own mean removed", 32.0, 0.97, "700-1800", 0.5, True, True)
    assert key == "own mean removed|32|0.97|700-1800|0.5|1|1"
    assert C.setting_values(key) == {"form": "own mean removed", "width": 32.0, "factor": 0.97,
                                     "region": "700-1800", "power": 0.5, "centred": True,
                                     "depth": True}
    text = C.shown(key)
    assert "|" not in text and "700 to 1800 cm^-1" in text and "square root" in text
    assert "over the wavenumber" in text and text.endswith("centred")
    assert C.shown(C.setting_name("slope", 8, 1.0, "whole", 1.0, False)) == (
        "slope, width 8 cm^-1, wavenumbers times 1, the whole range, intensities")
    best = C.best_of_option({"slope|8|1|whole|1|0|0": {"first": 0.2},
                             "slope|8|0.97|whole|1|0|0": {"first": 0.5},
                             "plain|8|0.97|whole|1|0|0": {"first": 0.3}})
    assert best["form"] == {"slope": 0.5, "plain": 0.3}
    assert best["factor"] == {"1.0": 0.2, "0.97": 0.5}


# --------------------------------------------------------------------------- #
# The benchmark on a made-up set
# --------------------------------------------------------------------------- #
def _made_up(n=50, seed=4):
    """Compounds with made-up bands.  The computed sticks stand too high by the factor
    1 / 0.97; the measured curve has the true bands with a width, intensities that fall with
    the wavenumber as in an ATR measurement, a baseline of its own and noise."""
    rng = np.random.default_rng(seed)
    x = np.arange(650.0, 3900.0 + 1e-9, 2.0)
    computed, rows, table = {}, [], []
    for c in range(n):
        smiles = "C" * (c + 1) + "O"
        true = np.sort(rng.uniform(750.0, 3700.0, 16))
        strength = rng.uniform(5.0, 60.0, 16)
        computed[smiles] = {"smiles": smiles, "index": c + 1,
                            "wavenumbers": (true / 0.97).tolist(),
                            "intensities": strength.tolist(), "n_imaginary": int(c % 9 == 0)}
        for sample in range(2 if c < 8 else 1):
            h = strength * rng.uniform(0.85, 1.2, 16) * 1000.0 / true
            y = sum(hi * np.exp(-0.5 * ((x - ci) / 7.0) ** 2) for ci, hi in zip(true, h))
            y = y / y.max() + rng.uniform(0.0, 0.5) + 0.002 * rng.normal(size=len(x))
            rows.append(y)
            table.append({"smiles": smiles, "sample_id": 100 * c + sample})
    head = {"qm_level": "made up", "left_out": {"an ion": 2}, "failed": [{"index": 0}],
            "with_an_imaginary_mode": 6, "seconds": 1.0, "max_atoms": 100}
    return x, np.vstack(rows), table, head, computed


def test_the_benchmark_finds_the_factor_and_writes_its_table(tmp_path, monkeypatch):
    x, A, table, head, computed = _made_up()
    monkeypatch.setattr(C.PREP, "load", lambda: (x, A, table))
    monkeypatch.setattr(C.XTB, "load", lambda: (dict(head), dict(computed)))
    out = C.run(str(tmp_path), repeats=4, widths=(8.0, 16.0), factors=(0.97, 1.0),
                powers=(1.0,))
    assert out["library"] == 50 and out["queries"] == 58 and out["compounds_with_a_query"] == 50
    assert out["settings"] == 3 * 2 * 2 * 2 * 2 * 2
    best = C.setting_values(out["setting_of_the_readings"])
    assert best["factor"] == 0.97                       # the factor that was put in
    assert out["nested"]["all"]["first"] > 0.9
    assert out["best_of_option"]["factor"]["0.97"] > out["best_of_option"]["factor"]["1.0"]
    assert out["chance"] == pytest.approx(1 / 50) and out["median_rank"] == 1.0
    assert out["library_failed"] == 1 and out["library_left_out"] == {"an ion": 2}
    assert sum(g["compounds"] for name, g in out["groups"].items() if "donor" in name) == 50
    assert out["with_an_imaginary_mode"]["compounds"] == 6
    text = C.report(str(tmp_path))
    assert text.startswith("# A measured infrared spectrum against a library of computed")
    for part in ("## How often the compound of the query is found",
                 "## What every option is worth", "## Which compounds are found",
                 "all settings open to the choice"):
        assert part in text
    assert "|0|" not in text and "|1|" not in text      # no setting key inside a table cell
    with open(tmp_path / "table.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    assert C.report(str(tmp_path / "nothing")) == ""


def test_a_library_spectrum_without_a_query_stays_a_candidate(tmp_path, monkeypatch):
    x, A, table, head, computed = _made_up(n=20)
    extra = {"smiles": "c1ccccc1", "index": 99, "wavenumbers": [800.0, 1500.0, 3100.0],
             "intensities": [10.0, 20.0, 5.0], "n_imaginary": 0}
    computed = dict(computed, **{"c1ccccc1": extra})
    del computed["CO"]                                  # a measured compound without a spectrum
    monkeypatch.setattr(C.PREP, "load", lambda: (x, A, table))
    monkeypatch.setattr(C.XTB, "load", lambda: (dict(head), dict(computed)))
    out = C.run(str(tmp_path), repeats=2, widths=(8.0,), factors=(0.97,), powers=(1.0,),
                forms=("slope",))
    assert out["library"] == 20 and out["compounds_with_a_query"] == 19
    assert out["queries"] == 26                         # the two spectra of "CO" are no query
