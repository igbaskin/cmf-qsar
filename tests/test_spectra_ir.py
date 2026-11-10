"""The infrared spectra of the benzoic acids as samples (plan item SP1, second measurement):
the data file, the builder stage that writes it, and the benchmark's own functions."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

import run_spectra_ir as IR  # noqa: E402
from cmfqsar import spectrum as SP  # noqa: E402
from cmfqsar.atomtargets import AtomTargets  # noqa: E402

SPECTRA = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_ir.json")
BANDS = os.path.join(ROOT, "examples", "spectra", "benzoic_acids_co.sdf")


# --------------------------------------------------------------------------- #
# The data file
# --------------------------------------------------------------------------- #
def _acids():
    from rdkit import Chem

    return [m for m in Chem.SDMolSupplier(BANDS, removeHs=False) if m is not None]


def test_the_file_holds_the_spectrum_of_every_acid_with_its_carbonyl_band():
    with open(SPECTRA, encoding="utf-8") as fh:
        head = json.load(fh)
    rows = {r["name"]: r for r in head["acids"]}
    acids = _acids()
    assert len(rows) == len(acids) == 189
    assert head["unit"] == "cm^-1" and head["frequency_scale"] == [0.97]
    for mol in acids:
        row = rows[mol.GetProp("_Name")]
        x, y = np.array(row["wavenumbers"]), np.array(row["intensities"])
        assert len(x) == len(y) and np.all(np.diff(x) >= 0) and np.all(x > 0)
        # 3 N - 6 modes, without the imaginary ones
        modes = 3 * mol.GetNumAtoms() - 6 - int(mol.GetProp("n_imaginary"))
        assert len(x) == modes
        # the band of plan item SP2 is one of the lines
        assert np.min(np.abs(x - float(mol.GetProp("co_stretch")))) < 2e-3
    spectra = IR.read_spectra(SPECTRA, [m.GetProp("_Name") for m in acids])
    assert [s.name for s in spectra[:2]] == ["H", acids[1].GetProp("_Name")]
    assert all(s.axis == "wavenumber" and s.kind == "sticks" for s in spectra)
    with pytest.raises(SystemExit, match="no spectrum for"):
        IR.read_spectra(SPECTRA, ["H", "an acid that is not there"])


def test_stage_4_writes_the_same_file_from_the_orca_outputs(tmp_path):
    import make_band_examples as M

    if not os.path.isdir(os.path.join(M.WORKDIR, "001_H")):
        pytest.skip("the ORCA outputs of plan item SP2 are not on this machine")
    out = str(tmp_path / "ir.json")
    lines = []
    assert M.write_infrared(BANDS, out, log=lines.append) == 0
    with open(out, "rb") as a, open(SPECTRA, "rb") as b:
        assert a.read().replace(b"\r\n", b"\n") == b.read().replace(b"\r\n", b"\n")
    assert "189 acids" in lines[-1]
    # an acid whose output is missing is counted and named
    assert M.write_infrared(BANDS, out, workdir=str(tmp_path), log=lines.append) == 189
    assert "no spectrum" in lines[-2] and "189 acids have none" in lines[-1]


# --------------------------------------------------------------------------- #
# The benchmark's own functions
# --------------------------------------------------------------------------- #
def _spectra(n=50, seed=0):
    """Spectra with one line whose position carries the property (it moves by up to 20
    cm^-1), and four lines at random places that carry nothing."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(-1.0, 1.0, n)
    spectra = []
    for value in y:
        x = np.concatenate([[1720.0 + 10.0 * value], rng.uniform(500.0, 1500.0, 4)])
        inten = np.concatenate([[300.0], rng.uniform(20.0, 200.0, 4)])
        spectra.append(SP.Spectrum(x, inten))
    return spectra, y


def test_a_setting_has_a_name_that_can_be_read_back():
    name = IR.setting_name(5.0, 0.5)
    assert name == "5 cm^-1, power 0.5" and "|" not in name
    assert IR.setting_values(name) == (5.0, 0.5)
    assert IR.setting_values(name + ", factor 4") == (5.0, 0.5)


def test_the_kernels_of_a_region_and_of_the_rest():
    spectra, _y = _spectra()
    cut = IR.outside(spectra[0], *IR.CARBONYL)
    assert len(cut) == 4 and np.all((cut.x < IR.CARBONYL[0]) | (cut.x > IR.CARBONYL[1]))
    kernels = IR.spectrum_kernels(spectra, *IR.CARBONYL, widths=(5.0, 20.0), powers=(1.0,))
    assert list(kernels) == ["5 cm^-1, power 1", "20 cm^-1, power 1"]
    for K in kernels.values():
        assert K.shape == (50, 50) and np.allclose(np.diag(K), 1.0)
    # in the carbonyl region there is one line, so the kernel is the closed form of its shift
    line = [float(s.within(*IR.CARBONYL).x[0]) for s in spectra[:2]]
    want = np.exp(-(line[0] - line[1]) ** 2 / (4 * 5.0 ** 2))
    assert kernels["5 cm^-1, power 1"][0, 1] == pytest.approx(want, abs=1e-5)


def test_a_gaussian_kernel_on_top_and_a_sum_of_two_kernels():
    spectra, _y = _spectra()
    base = IR.spectrum_kernels(spectra, *IR.RANGE, widths=(5.0,), powers=(1.0, 0.5))
    top = IR.layered(base, gammas=(0.5, 2.0))
    assert len(top) == 4 and "5 cm^-1, power 1, factor 2" in top
    K = base["5 cm^-1, power 1"]
    d2 = 2.0 - 2.0 * K
    med = np.median(d2[np.triu_indices(50, 1)])
    assert np.allclose(top["5 cm^-1, power 1, factor 2"], np.exp(-2.0 * d2 / med))
    other = {"a": np.eye(50)}
    both = IR.joined(other, base, shares=(0.25, 0.75))
    assert len(both) == 4
    key = "a + 5 cm^-1, power 1, share 0.25"
    assert np.allclose(both[key], 0.75 * np.eye(50) + 0.25 * K)
    assert all(np.allclose(np.diag(v), 1.0) for v in both.values())


def test_the_choice_inside_the_folds_reads_the_position_of_a_line():
    spectra, y = _spectra(n=60)
    targets = AtomTargets.from_arrays([[v] for v in y], name="made up")
    assert np.array_equal(IR.with_target(targets, 2 * y, "twice").y, 2 * y)
    kernels = IR.spectrum_kernels(spectra, *IR.RANGE, widths=(5.0, 200.0), powers=(1.0,))
    results, chosen = IR.nested(kernels, targets, [0, 1], "spectrum")
    assert len(results) == 2 and sum(chosen.values()) == 2 * IR.FOLDS
    # a width of 200 cm^-1 cannot see a shift of 10 cm^-1; a width of 5 can
    assert IR.most_chosen(chosen) == "5 cm^-1, power 1"
    table = {"position": ["meta"] * 30 + ["para"] * 30,
             "substituent": [f"s{k}" for k in range(30)] * 2}
    row = IR.row_of(results, chosen, targets, table)
    assert row["q2"] > 0.5 and row["twin_q2"] is not None
    # against a regression on a column of noise the spectrum is above, with its interval
    noise = np.random.default_rng(1).normal(size=60)
    flat, none = IR.regression(noise, targets, [0, 1], "noise")
    assert none == {}
    d = IR.compare(results, flat, targets)
    assert d[1] > 0 and d[1] < d[0] < d[2]
    # the number that needs no fit says the same
    assert IR.target_alignment(kernels["5 cm^-1, power 1"], y) > \
        IR.target_alignment(kernels["200 cm^-1, power 1"], y) + 0.05


def test_the_regions_of_the_mid_infrared():
    assert IR.region_of(1727.9) == "C=O stretch"
    assert IR.region_of(3642.0) == "O-H stretch"
    assert IR.region_of(3300.0) == "between the named regions"


def test_the_stored_table_is_the_report():
    stored = os.path.join(IR.RESULTS, "table.md")
    if not os.path.isfile(os.path.join(IR.RESULTS, "ir.json")):
        pytest.skip("the benchmark has not been run")
    with open(stored, encoding="utf-8") as fh:
        before = fh.read()
    assert IR.report() + "\n" == before
    assert "## Reading" in before and "The gate of the item" in before
