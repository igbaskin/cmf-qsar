"""A spectrum as a one-dimensional field: plan item SP1.

The item rests on one identity: the dot product of two broadened spectra on a grid is the
closed-form sum over the pairs of their sticks.  These tests check the identity first, then
what the kernel ignores and what it keeps, the readers, the powder pattern, and the
co-spectrum that a fitted model gives back.
"""
import os
import sys

import numpy as np
import pytest

from cmfqsar import spectrum as SP
from cmfqsar.model import _Learner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _random_sticks(n, seed, lo=10.0, hi=80.0):
    rng = np.random.default_rng(seed)
    return SP.Spectrum(rng.uniform(lo, hi, n), rng.uniform(0.1, 5.0, n), axis="two_theta")


@pytest.fixture(scope="module")
def spectra():
    return [_random_sticks(n, k) for k, n in enumerate((5, 40, 120, 3))]


# --------------------------------------------------------------------------- #
# The kernel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sigma", [0.05, 0.3, 2.0])
def test_the_kernel_on_a_grid_equals_the_closed_form(spectra, sigma):
    exact = SP.exact_kernel(spectra, sigma)
    on_grid = SP.kernel(spectra, 10.0 - 7 * sigma, 80.0 + 7 * sigma, sigma)
    assert np.abs(exact - on_grid).max() < 1e-6
    assert np.allclose(np.diag(exact), 1.0)


def test_the_dot_product_of_two_rows_is_the_sum_over_the_pairs_of_sticks(spectra):
    sigma = 0.3
    grid = SP.Grid.for_width(0.0, 90.0, sigma)
    F = SP.feature_matrix(spectra, grid, sigma)
    closed = SP.stick_overlap(spectra[1], spectra[2], sigma)
    on_grid = 2.0 * sigma * np.sqrt(np.pi) * grid.step * float(F[1] @ F[2])
    assert on_grid == pytest.approx(closed, rel=1e-6)
    # every stick is a normalised Gaussian, so a row integrates to the sum of the intensities
    assert grid.step * F[1].sum() == pytest.approx(spectra[1].y.sum(), rel=1e-6)


def test_the_width_decides_whether_a_shifted_line_is_the_same_line():
    a = SP.Spectrum([30.0], [2.0])
    for shift, sigma in ((0.1, 0.05), (0.1, 0.5), (1.0, 0.5)):
        b = SP.Spectrum([30.0 + shift], [7.0])
        k = SP.exact_kernel([a], sigma, others=[b])[0, 0]
        assert k == pytest.approx(np.exp(-shift ** 2 / (4 * sigma ** 2)))
    # far apart: nothing in common
    far = SP.Spectrum([60.0], [1.0])
    assert SP.exact_kernel([a], 0.5, others=[far])[0, 0] < 1e-12


def test_the_cosine_kernel_ignores_the_total_intensity(spectra):
    louder = [SP.Spectrum(s.x, 7.0 * s.y) for s in spectra]
    assert np.allclose(SP.exact_kernel(spectra, 0.4), SP.exact_kernel(louder, 0.4))
    assert np.allclose(SP.kernel(spectra, 0, 90, 0.4), SP.kernel(louder, 0, 90, 0.4))
    # without the normalisation the intensity enters
    raw = SP.exact_kernel(spectra, 0.4, normalize=False)
    assert np.allclose(SP.exact_kernel(louder, 0.4, normalize=False), 49.0 * raw)


def test_the_power_of_the_intensity_is_applied_to_the_sticks():
    a = SP.Spectrum([20.0, 40.0], [100.0, 1.0])
    b = SP.Spectrum([20.0, 40.0], [10.0, 1.0])
    assert SP.stick_overlap(a, a, 0.2, power=0.5) == pytest.approx(
        SP.stick_overlap(b, b, 0.2, power=1.0))
    grid = SP.Grid(0.0, 60.0, 0.05)
    assert np.allclose(SP.broaden(a, grid, 0.2, power=0.5), SP.broaden(b, grid, 0.2))
    # the square root lowers the share of the strongest line
    weak = SP.Spectrum([40.0], [1.0])
    k1 = SP.exact_kernel([a], 0.2, others=[weak])[0, 0]
    k2 = SP.exact_kernel([a], 0.2, others=[weak], power=0.5)[0, 0]
    assert k2 > 5 * k1


def test_a_measured_curve_enters_as_the_sum_over_its_grid_points(spectra):
    """A curve that is a stick spectrum broadened by an instrument of width s1 has, with a
    stick spectrum, the closed-form overlap of the two with the widths combined."""
    s1, sigma = 0.15, 0.6
    sticks, other = spectra[1], spectra[2]
    fine = SP.Grid(0.0, 90.0, 0.01)
    curve = SP.Spectrum(fine.x, SP.broaden(sticks, fine, s1), kind="curve", axis="two_theta")
    got = SP.stick_overlap(curve, other, sigma)
    d = sticks.x[:, None] - other.x[None, :]
    var = 2 * sigma ** 2 + s1 ** 2
    want = float(sticks.y @ (np.sqrt(2 * sigma ** 2 / var) * np.exp(-d * d / (2 * var)))
                 @ other.y)
    assert got == pytest.approx(want, rel=1e-5)
    # with sigma = 0 a curve is only put on the grid; sticks need a width
    coarse = SP.Grid(10.0, 80.0, 0.5)
    assert np.allclose(SP.broaden(curve, coarse, 0.0), np.interp(coarse.x, curve.x, curve.y))
    with pytest.raises(ValueError, match="positive width"):
        SP.broaden(sticks, coarse, 0.0)


def test_a_spectrum_with_no_line_on_the_grid_has_a_zero_row(spectra):
    outside = SP.Spectrum([500.0], [3.0])
    empty = SP.Spectrum([], [])
    K = SP.kernel(spectra[:2] + [outside, empty], 0.0, 90.0, 0.5)
    assert np.all(np.isfinite(K))
    assert np.allclose(K[2], 0.0) and np.allclose(K[3], 0.0) and np.allclose(K[:, 2], 0.0)
    assert SP.stick_overlap(empty, spectra[0], 0.5) == 0.0


def test_the_kernel_of_several_widths_is_the_mean_of_the_kernels(spectra):
    sigmas = (0.1, 1.0, 5.0)
    K = SP.multi_width_kernel(spectra, 0.0, 90.0, sigmas)
    parts = [SP.kernel(spectra, 0.0, 90.0, s) for s in sigmas]
    assert np.allclose(K, np.mean(parts, axis=0))
    assert np.allclose(np.diag(K), 1.0)
    weighted = SP.multi_width_kernel(spectra, 0.0, 90.0, sigmas, weights=(2.0, 0.0, 0.0))
    assert np.allclose(weighted, parts[0])
    with pytest.raises(ValueError, match="one weight per width"):
        SP.multi_width_kernel(spectra, 0.0, 90.0, sigmas, weights=(1.0, 1.0))
    with pytest.raises(ValueError, match="at least one width"):
        SP.multi_width_kernel(spectra, 0.0, 90.0, ())


# --------------------------------------------------------------------------- #
# The spectrum and its readers
# --------------------------------------------------------------------------- #
def test_a_spectrum_is_sorted_and_checked():
    s = SP.Spectrum([3.0, 1.0, 2.0], [30.0, 10.0, 20.0], axis="shift")
    assert list(s.x) == [1.0, 2.0, 3.0] and list(s.y) == [10.0, 20.0, 30.0]
    assert s.unit == "ppm" and len(s) == 3
    assert list(s.within(1.5, 3.0).x) == [2.0, 3.0]
    assert SP.Spectrum([1.0], [1.0], axis="my own axis").unit == ""
    with pytest.raises(ValueError, match="as many intensities as positions"):
        SP.Spectrum([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="not a number"):
        SP.Spectrum([1.0, np.nan], [1.0, 2.0], name="bad")
    with pytest.raises(ValueError, match="kind must be"):
        SP.Spectrum([1.0], [1.0], kind="bars")
    with pytest.raises(ValueError, match="at least two points"):
        SP.Spectrum([1.0], [1.0], kind="curve")
    # a curve as sticks: the weights add up to the integral of the curve
    curve = SP.Spectrum([0.0, 1.0, 3.0], [2.0, 2.0, 2.0], kind="curve")
    assert curve.lines()[1].sum() == pytest.approx(6.0)


def test_the_readers_take_two_column_text(tmp_path):
    path = tmp_path / "ir.txt"
    path.write_text("# wavenumber  intensity\n1700.5 320.0 0.1\n; a comment\n"
                    "3050.0, 12.5\n650.0 40.0\n", encoding="utf-8")
    sticks = SP.read_sticks(str(path))
    assert sticks.kind == "sticks" and sticks.axis == "wavenumber"
    assert list(sticks.x) == [650.0, 1700.5, 3050.0] and list(sticks.y) == [40.0, 320.0, 12.5]
    curve = SP.read_measured(str(path), axis="binding_energy")
    assert curve.kind == "curve" and curve.unit == "eV" and curve.name == "ir.txt"


def test_the_infrared_sticks_of_an_orca_result():
    from cmfqsar.orca import OrcaResult

    res = OrcaResult()
    res.frequencies = np.array([-35.0, 0.0, 0.0, 0.0, 650.0, 1780.0, 3700.0])
    res.ir_intensities = np.array([1.0, 0.0, 0.0, 0.0, 40.0, 330.0, 60.0])
    s = SP.orca_sticks(res, scale=0.97, name="acid")
    assert np.allclose(s.x, 0.97 * np.array([650.0, 1780.0, 3700.0]))    # no zero, no imaginary
    assert list(s.y) == [40.0, 330.0, 60.0] and s.axis == "wavenumber"
    res.ir_intensities = None
    with pytest.raises(ValueError, match="no infrared intensities"):
        SP.orca_sticks(res, name="acid")
    res.frequencies = None
    with pytest.raises(ValueError, match="no vibrational frequencies"):
        SP.orca_sticks(res, name="acid")


def test_the_powder_pattern_of_rock_salt():
    pytest.importorskip("pymatgen")
    bulk = pytest.importorskip("ase.build").bulk
    primitive = SP.powder_sticks(bulk("NaCl", "rocksalt", a=5.64))
    cubic = SP.powder_sticks(bulk("NaCl", "rocksalt", a=5.64, cubic=True))
    assert primitive.axis == "two_theta" and primitive.kind == "sticks"
    # the 111 and the 200 reflection, and the 200 one is the strongest
    assert np.allclose(primitive.x[:2], [27.39, 31.73], atol=0.02)
    assert int(np.argmax(primitive.y)) == 1
    assert SP.two_theta_to_d(primitive.x[1], 1.54184) == pytest.approx(5.64 / 2, abs=0.005)
    # the pattern does not depend on the choice of the cell
    assert np.allclose(primitive.x, cubic.x, atol=1e-6)
    assert np.allclose(primitive.y / primitive.y.max(), cubic.y / cubic.y.max(), atol=1e-6)
    assert SP.exact_kernel([primitive], 0.1, others=[cubic])[0, 0] == pytest.approx(1.0)
    # neutrons see the same reflections with other intensities
    neutron = SP.powder_sticks(bulk("NaCl", "rocksalt", a=5.64), source="neutron")
    assert np.allclose(neutron.x, primitive.x, atol=1e-6)
    assert not np.allclose(neutron.y / neutron.y.max(), primitive.y / primitive.y.max(),
                           atol=0.05)
    # a larger cell moves every reflection to a smaller angle
    larger = SP.powder_sticks(bulk("NaCl", "rocksalt", a=5.80))
    assert larger.x[0] < primitive.x[0]
    with pytest.raises(ValueError, match="'xray' or 'neutron'"):
        SP.powder_sticks(bulk("NaCl", "rocksalt", a=5.64), source="electron")


# --------------------------------------------------------------------------- #
# The co-spectrum
# --------------------------------------------------------------------------- #
def _series(n=60, seed=0):
    """Spectra with one line at 50 whose intensity carries the property and three lines at
    random places that carry nothing."""
    rng = np.random.default_rng(seed)
    y = rng.uniform(0.0, 1.0, n)
    spectra = []
    for value in y:
        x = np.concatenate([[20.0, 50.0], rng.uniform(60.0, 85.0, 3)])
        inten = np.concatenate([[3.0, 3.0 * value], rng.uniform(0.5, 1.5, 3)])
        spectra.append(SP.Spectrum(x, inten, axis="two_theta"))
    return spectra, y


def test_the_co_spectrum_is_the_weight_that_gives_the_prediction():
    spectra, y = _series()
    grid = SP.Grid.for_width(0.0, 90.0, 0.5)
    F = SP.feature_matrix(spectra, grid, 0.5)
    train, test = np.arange(45), np.arange(45, 60)
    K = SP.spectrum_kernel(F[train])
    learner = _Learner("krr", {"lam": 1e-3}, True).fit(K, y[train])
    w = SP.co_spectrum(F[train], learner.alpha_)
    assert w.shape == grid.x.shape
    Z = F[test] / np.sqrt(np.einsum("ij,ij->i", F[test], F[test]))[:, None]
    assert np.allclose(learner.predict(SP.spectrum_kernel(F[train], F[test])),
                       learner.b_ + Z @ w)
    with pytest.raises(ValueError, match="coefficients for"):
        SP.co_spectrum(F[train], learner.alpha_[:-1])


def test_the_co_spectrum_finds_the_line_that_carries_the_property():
    spectra, y = _series()
    grid = SP.Grid.for_width(0.0, 90.0, 0.5)
    F = SP.feature_matrix(spectra, grid, 0.5)
    learner = _Learner("krr", {"lam": 1e-3}, True).fit(SP.spectrum_kernel(F), y)
    w = SP.co_spectrum(F, learner.alpha_)
    effect = SP.co_spectrum_effect(F, w)
    top = SP.strongest_regions(grid.x, effect, n=3, separation=5.0)
    assert abs(top[0]["x"] - 50.0) < 1.0 and top[0]["value"] > 0
    assert all(abs(a["x"] - b["x"]) > 5.0 for a in top for b in top if a is not b)
    # where no spectrum has intensity the effect is zero, whatever the weight
    assert np.allclose(effect[grid.x < 10.0], 0.0)


def test_braggs_law():
    assert SP.two_theta_to_d(60.0, 1.0) == pytest.approx(1.0)
    assert np.allclose(SP.two_theta_to_d([180.0], 1.54184), [1.54184 / 2])


# --------------------------------------------------------------------------- #
# The benchmark's own functions
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def bench():
    pytest.importorskip("sklearn")
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import run_spectra

    return run_spectra


def test_spectra_of_different_lengths_survive_one_file(bench, spectra, tmp_path):
    path = tmp_path / "p.npz"
    np.savez_compressed(path, **bench.pack(spectra))
    with np.load(path) as z:
        back = bench.unpack(z, [f"s{k}" for k in range(len(spectra))])
    for a, b in zip(spectra, back):
        assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
    assert back[2].name == "s2" and back[0].axis == "two_theta"


def test_the_difference_in_the_error_has_a_sign_and_an_interval(bench):
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    good = y + 0.1 * rng.normal(size=300)
    bad = y + 1.0 * rng.normal(size=300)
    d, lo, hi = bench.mae_difference(y, good, bad, runs=500)
    assert d < 0 and hi < 0 and lo < d < hi
    d2, lo2, hi2 = bench.mae_difference(y, good, good, runs=200)
    assert d2 == 0.0 and lo2 == 0.0 and hi2 == 0.0
    # several repeats: the errors of a sample are averaged first
    d3, _, _ = bench.mae_difference(y, np.vstack([good, bad]), np.vstack([bad, good]), runs=200)
    assert d3 == pytest.approx(0.0, abs=1e-12)


def test_the_choice_inside_the_folds_takes_the_setting_that_predicts(bench):
    rng = np.random.default_rng(1)
    n = 120
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.05 * rng.normal(size=n)
    noise = rng.normal(size=(n, 40))
    view = bench.Fixed({"noise": noise @ noise.T / 40.0, "signal": X @ X.T})
    folds = bench.folds_for(n)
    row = bench.nested(view, y, folds, lams=(1e-3, 1e-1), inner=5)
    assert [f["setting"] for f in row["per_fold"]] == ["signal"] * 5
    assert row["r2"] > 0.95 and len(row["pred"]) == n
    assert bench.most_chosen(row) == "signal"
    flat = bench.constant(y, folds)
    line = bench.regression(X, y, folds)
    assert flat["r2"] < 0.05 and line["r2"] > 0.99
    assert row["mae_mean"] < 0.2 * flat["mae_mean"]


def test_joined_kernels_keep_the_one_that_predicts(bench):
    rng = np.random.default_rng(2)
    n = 100
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.05 * rng.normal(size=n)
    noise = rng.normal(size=(n, 30))
    views = {"signal": bench.Fixed({"a": X @ X.T / 3.0}),
             "noise": bench.Fixed({"b": noise @ noise.T / 30.0})}
    folds = bench.folds_for(n)
    lams = (1e-3, 1e-1)
    singles = {nm: bench.nested(v, y, folds, lams=lams, inner=5) for nm, v in views.items()}
    both = bench.mixed(views, singles, y, folds, "evidence", lams=lams, inner=5)
    shares = [f["shares"] for f in both["per_fold"]]
    assert all(s["signal"] > 0.9 for s in shares)
    assert both["r2"] > 0.95
    equal = bench.mixed(views, singles, y, folds, "equal", lams=lams, inner=5)
    assert all(f["shares"]["signal"] == pytest.approx(0.5) for f in equal["per_fold"])


def test_how_alike_two_kernels_are(bench):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 4))
    K = X @ X.T
    other = rng.normal(size=(50, 4))
    assert bench.kernel_agreement(K, 3.0 * K + 2.0) == pytest.approx(1.0)
    assert abs(bench.kernel_agreement(K, other @ other.T)) < 0.3
    y = X[:, 0]
    assert bench.target_alignment(np.outer(y, y), y) == pytest.approx(1.0)


def test_the_stored_table_is_the_report(bench):
    stored = os.path.join(bench.RESULTS, "table.md")
    if not os.path.isfile(os.path.join(bench.RESULTS, "fit.json")):
        pytest.skip("the benchmark has not been run")
    with open(stored, encoding="utf-8") as fh:
        before = fh.read()
    assert bench.report() + "\n" == before
    assert "## Reading" in before and "The gate of the item" in before


def test_the_fast_choice_of_the_ridge_equals_one_fit_per_ridge(bench):
    rng = np.random.default_rng(4)
    n = 90
    X = rng.normal(size=(n, 6))
    y = 300.0 + 50.0 * (X @ rng.normal(size=6)) + 5.0 * rng.normal(size=n)
    K = X @ X.T / 6.0
    lams = (1e-6, 1e-4, 1e-2, 1.0)
    fast = bench.choose(K, y, lams, inner=5)
    slow = bench.choose_by_refit(K, y, lams, inner=5)
    assert fast[1] == slow[1]
    assert fast[0] == pytest.approx(slow[0], rel=1e-8)


def test_a_setting_is_written_without_the_separator_of_table_columns(bench):
    assert bench.setting_name(10.0, 1.0) == "10|1"
    assert bench.setting_values("0.05|0.5") == (0.05, 0.5)
    assert bench.shown("10|1") == "10 deg, power 1"
    assert bench.shown("5|0.5|g2") == "5 deg, power 0.5, factor 2"
    assert bench.shown("gauss 1") == "gauss 1"
    assert "|" not in bench.shown("0.2|1|g0.25")


def test_a_gaussian_kernel_on_top_of_a_description(bench):
    rng = np.random.default_rng(5)
    X = rng.normal(size=(40, 5))
    Z = X / np.sqrt((X * X).sum(axis=1))[:, None]
    base = bench.Fixed({"a": Z @ Z.T})
    layered = bench.Layered(base, gammas=(0.5, 2.0))
    assert layered.settings == ["a|g0.5", "a|g2"]
    tr, te = np.arange(30), np.arange(30, 40)
    K_tr, K_te = layered.pair("a|g2", tr, te)
    assert K_tr.shape == (30, 30) and K_te.shape == (10, 30)
    assert np.allclose(np.diag(K_tr), 1.0)
    d2 = 2.0 - 2.0 * (Z @ Z.T)
    med = np.median(d2[np.ix_(tr, tr)][np.triu_indices(30, 1)])
    assert np.allclose(K_te, np.exp(-2.0 * d2[np.ix_(te, tr)] / med))


def test_the_marginal_likelihood_does_not_credit_a_kernel_for_its_scale(bench):
    """A kernel of noise beside the kernel that predicts.  With every marginal likelihood
    taken at its best ridge on a fine grid, the joined kernel gains nothing over the kernel
    that predicts."""
    rng = np.random.default_rng(6)
    n = 80
    X = rng.normal(size=(n, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + 0.01 * rng.normal(size=n)
    noise = rng.normal(size=(n, 30))
    lams = tuple(10.0 ** np.arange(-8.0, 0.01, 0.5))
    part = bench.evidence_part(X @ X.T / 3.0, noise @ noise.T / 30.0, y,
                               drop={"structure": 9.0, "xray": 7.0}, lams=lams)
    assert part["shares"]["structure"] > 0.9
    le = part["log_evidence"]
    assert le["both"] - le["structure"] < 2.0
    assert le["both"] - le["xray"] > 50.0
    assert part["drop_one"] == {"structure": 9.0, "xray": 7.0}       # kept as given
    assert part["ridges_tried"] == [1e-8, 1.0]
