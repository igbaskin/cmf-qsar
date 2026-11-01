"""Measured scattering curves as samples: plan item M12.

The claim the item rests on is that a measured curve is the *total, untyped* version of the
descriptor this program already computes.  These tests check that claim first, then the pieces
built on it: the two transforms, what a measurement does to a curve, the reader, and the kernel
that compares a measured curve with a computed one.
"""
import os

import numpy as np
import pytest

from cmfqsar import scattering as S
from cmfqsar.periodic import crystal_fields, inorganic_fieldset

bulk = pytest.importorskip("ase.build").bulk

SPEC = S.CurveSpec(r_min=1.0, r_max=20.0, r_step=0.05, sigma=0.05, q_min=0.5, q_max=25.0,
                   q_step=0.05)


@pytest.fixture(scope="module")
def fieldset():
    return inorganic_fieldset([], physchem=["steric"])


@pytest.fixture(scope="module")
def nacl(fieldset):
    return crystal_fields(bulk("NaCl", "rocksalt", a=5.64), fieldset, "NaCl")


@pytest.fixture(scope="module")
def crystals(fieldset):
    return {name: crystal_fields(atoms, fieldset, name) for name, atoms in
            (("NaCl", bulk("NaCl", "rocksalt", a=5.64)),
             ("MgO", bulk("MgO", "rocksalt", a=4.21)),
             ("Si", bulk("Si", "diamond", a=5.43)))}


# --------------------------------------------------------------------------- #
# How an atom scatters
# --------------------------------------------------------------------------- #
def test_the_xray_weight_is_the_atomic_number():
    assert S.scattering_lengths(["H", "C", "O", "Fe"], "xray") == pytest.approx([1, 6, 8, 26])


def test_the_neutron_weights_are_the_tabulated_ones():
    b = S.scattering_lengths(["H", "C", "O", "Ti"], "neutron")
    assert b[0] < 0 and b[3] < 0            # hydrogen and titanium scatter with a sign change
    assert b == pytest.approx([S.NEUTRON_B[s] for s in ("H", "C", "O", "Ti")])


def test_an_element_without_a_tabulated_value_is_an_error_not_a_guess():
    with pytest.raises(ValueError, match="no neutron scattering length"):
        S.scattering_lengths(["O", "Pm"], "neutron")
    with pytest.raises(ValueError, match="not an element"):
        S.scattering_lengths(["O", "Xx"], "xray")
    with pytest.raises(ValueError, match="scattering source"):
        S.scattering_lengths(["O"], "electron")


def test_the_record_carries_one_field_and_shares_everything_else(nacl):
    rec = S.scattering_record(nacl, "xray")
    assert rec.weights.shape == (nacl.n_atoms, 1)
    assert rec.weights[:, 0] == pytest.approx(S.scattering_lengths(nacl.symbols, "xray"))
    assert np.shares_memory(rec.coords, nacl.coords)
    assert rec.cell is nacl.cell and rec.pbc == nacl.pbc


def test_a_disordered_cell_is_refused(fieldset):
    atoms = bulk("MgO", "rocksalt", a=4.21)
    atoms.info["site_species"] = [{"Mg": 0.5, "Na": 0.5}, {"O": 1.0}]
    atoms.arrays["occupancy"] = np.array([1.0, 1.0])
    dis = crystal_fields(atoms, inorganic_fieldset(["Mg", "Na", "O"]), "half Na")
    with pytest.raises(ValueError, match="disordered"):
        S.scattering_record(dis, "xray")


# --------------------------------------------------------------------------- #
# The claim: the curve is one block of the descriptor
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source", ["xray", "neutron"])
def test_the_descriptor_route_is_the_definition_of_g(crystals, source):
    """G(r) from the one-field autocorrelation against G(r) summed over the neighbour list.

    This is what plan item M12 rests on: a measured curve is the untyped version of the
    descriptor, not a different object.
    """
    for name, c in crystals.items():
        rec = S.pair_distribution_residual(c, SPEC, source)
        assert rec["peak"] > 1.0, name
        assert rec["worst"] < 1e-3, f"{name} ({source}): {rec['worst']:.2e}"


def test_the_two_routes_agree_better_as_the_channel_narrows(nacl):
    """The identity is exact in the limit of a narrow channel, so the error must fall with it."""
    worst = []
    for sigma in (0.1, 0.05, 0.02):
        spec = S.CurveSpec(r_min=1.0, r_max=15.0, r_step=sigma, sigma=sigma)
        worst.append(S.pair_distribution_residual(nacl, spec)["worst"])
    assert worst[0] > worst[1] > worst[2]
    assert worst[-1] < 1e-5


def test_g_has_the_peaks_of_the_structure(nacl):
    """The peaks of rocksalt NaCl sit where its neighbour shells are.

    The nearest neighbour is Na-Cl at half the lattice constant, 2.82 A; the second is Na-Na and
    Cl-Cl at a / sqrt(2), 3.99 A, and that one is the taller because it has twelve neighbours in
    it against six.
    """
    r, g = S.pair_distribution(nacl, SPEC)

    def peak_near(d, window=0.3):
        near = np.abs(r - d) < window
        return float(g[near].max()), float(r[near][np.argmax(g[near])])

    first_height, first_r = peak_near(5.64 / 2.0)
    second_height, second_r = peak_near(5.64 / np.sqrt(2.0))
    assert first_r == pytest.approx(2.82, abs=0.1)
    assert second_r == pytest.approx(3.99, abs=0.1)
    assert first_height > 5.0 and second_height > first_height
    assert np.mean(g) == pytest.approx(0.0, abs=2.0)   # it oscillates about zero
    # The tallest peak of this curve is neither of those two but one far out, because a perfect
    # crystal has more neighbours in a distant shell than a near one and nothing here broadens
    # them: there is no thermal motion in the model and no instrument. A measured curve decays
    # with r for both of those reasons, which measured_like puts back in part.
    assert float(r[np.argmax(g)]) > 5.0


def test_g_does_not_change_with_the_choice_of_cell(fieldset):
    """A supercell is the same material, so it has the same pair distribution function."""
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    base = S.pair_distribution(crystal_fields(atoms, fieldset, "NaCl"), SPEC)[1]
    for rep in ((2, 1, 1), (2, 2, 2)):
        sup = crystal_fields(atoms.repeat(rep), fieldset, f"sup {rep}")
        other = S.pair_distribution(sup, SPEC)[1]
        assert np.abs(base - other).max() / np.abs(base).max() < 1e-9


def test_a_structure_with_no_scattering_power_is_refused(nacl, monkeypatch):
    """A "null matrix": a composition whose mean neutron scattering length cancels.

    Real ones exist -- a titanium-zirconium alloy is used as a sample holder because of it -- and
    G(r) is not defined for them, since its definition divides by that mean.  The tabulated value
    is moved here to make the cancellation exact.
    """
    monkeypatch.setitem(S.NEUTRON_B, "Cl", -S.NEUTRON_B["Na"])
    with pytest.raises(ValueError, match="mean scattering length of zero"):
        S.pair_distribution(nacl, SPEC, "neutron")


# --------------------------------------------------------------------------- #
# The two transforms
# --------------------------------------------------------------------------- #
def test_the_sine_transform_inverts_itself(nacl):
    """G(r) -> F(Q) -> G(r) returns the curve it started from."""
    # wide grids in both r and Q: the round trip is limited by whichever is shorter, and the
    # point here is the transform and not the truncation
    spec = S.CurveSpec(r_min=0.5, r_max=40.0, r_step=0.02, sigma=0.05, q_min=0.05, q_max=40.0,
                       q_step=0.02)
    r, g = S.pair_distribution(nacl, spec)
    q, f = S.structure_function(nacl, spec, route="transform")
    back = S.sine_transform(q, f, r)
    keep = (r > 2.0) & (r < 25.0)
    assert np.corrcoef(g[keep], back[keep])[0, 1] > 0.99
    assert np.abs(g[keep] - back[keep]).max() / np.abs(g[keep]).max() < 0.15


def test_the_structure_function_goes_to_zero_at_large_q(nacl):
    """F(Q) = Q [S(Q) - 1] with the self term taken out, so S approaches 1 and F approaches 0.

    Keeping that term would add a line growing as (<b^2>/<b>^2 - 1) Q to every curve of a
    structure with more than one element, which is what this guards against.
    """
    spec = S.CurveSpec(r_min=0.5, r_max=30.0, r_step=0.02, sigma=0.05, q_min=0.05, q_max=30.0,
                       q_step=0.02)
    q, f = S.structure_function(nacl, spec, route="transform")
    high = q > 25.0
    assert np.abs(f[high]).mean() < 0.2 * np.abs(f).max()


def test_the_reciprocal_route_is_available_and_refuses_an_impossible_request(nacl):
    q, f = S.structure_function(nacl, SPEC, route="reciprocal")
    assert q.shape == f.shape and np.all(np.isfinite(f))
    with pytest.raises(ValueError, match="route"):
        S.structure_function(nacl, SPEC, route="nowhere")


# --------------------------------------------------------------------------- #
# What a measurement does to a curve
# --------------------------------------------------------------------------- #
def test_a_finite_range_in_q_broadens_the_peaks(nacl):
    r, g = S.pair_distribution(nacl, SPEC)
    sharp = S.measured_like(SPEC, r, g, q_max=25.0)
    blunt = S.measured_like(SPEC, r, g, q_max=10.0)
    assert np.abs(blunt).max() < np.abs(sharp).max() < np.abs(g).max()
    assert np.corrcoef(blunt, g)[0, 1] < np.corrcoef(sharp, g)[0, 1] < 1.0


def test_noise_and_a_density_error_do_what_they_say(nacl):
    r, g = S.pair_distribution(nacl, SPEC)
    assert S.measured_like(SPEC, r, g) == pytest.approx(g)        # nothing asked, nothing done
    quiet = S.measured_like(SPEC, r, g, q_max=25.0)
    noisy = S.measured_like(SPEC, r, g, q_max=25.0, noise=0.1)
    assert np.abs(noisy - quiet).max() > 0.0
    assert np.corrcoef(noisy, quiet)[0, 1] > 0.5
    rho = nacl.n_atoms / nacl.volume
    sloped = S.measured_like(SPEC, r, g, density_error=0.1, density=rho)
    line = sloped - g
    assert line == pytest.approx(-4.0 * np.pi * r * rho * 0.1)
    # the same seed gives the same noise, so a measurement can be repeated exactly
    assert S.measured_like(SPEC, r, g, q_max=25.0, noise=0.1, seed=3) == pytest.approx(
        S.measured_like(SPEC, r, g, q_max=25.0, noise=0.1, seed=3))


# --------------------------------------------------------------------------- #
# Reading a measured file
# --------------------------------------------------------------------------- #
def _write(path, r, g, header=("# qmax = 25.0", "# rho0 = 0.0446", "#L r G(r) dr dG(r)")):
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for line in header:
            fh.write(line + "\n")
        for x, y in zip(r, g):
            fh.write(f"{x:.4f} {y:.6f} 0.0000 0.0100\n")


def test_a_written_curve_is_read_back(tmp_path, nacl):
    r, g = S.pair_distribution(nacl, SPEC)
    path = os.path.join(str(tmp_path), "nacl.gr")
    _write(path, r, g)
    curve = S.read_curve(path)
    assert curve["name"] == "nacl.gr"
    assert curve["header"]["q_max"] == pytest.approx(25.0)
    assert curve["header"]["density"] == pytest.approx(0.0446)
    assert curve["x"] == pytest.approx(r)
    assert curve["y"] == pytest.approx(g, abs=1e-5)
    assert S.resample(curve, SPEC.r) == pytest.approx(g, abs=1e-5)


def test_the_reader_survives_the_things_files_do(tmp_path, nacl):
    r, g = S.pair_distribution(nacl, SPEC)
    path = os.path.join(str(tmp_path), "odd.gr")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("; a semicolon comment\n\n#### start data\n")
        fh.write("a line of words that is not data\n")
        for x, y in zip(r[::-1], g[::-1]):          # written backwards
            fh.write(f"{x:.4f},{y:.6f}\n")
    curve = S.read_curve(path)
    assert np.all(np.diff(curve["x"]) > 0)          # it comes back in order
    assert curve["y"] == pytest.approx(g, abs=1e-5)
    empty = os.path.join(str(tmp_path), "empty.gr")
    with open(empty, "w", encoding="utf-8") as fh:
        fh.write("# nothing but a header\n")
    with pytest.raises(ValueError, match="two-column"):
        S.read_curve(empty)


def test_a_curve_that_stops_early_is_filled_and_not_extrapolated(tmp_path, nacl):
    r, g = S.pair_distribution(nacl, SPEC)
    curve = {"x": r[:100], "y": g[:100]}
    out = S.resample(curve, SPEC.r)
    assert out[:100] == pytest.approx(g[:100])
    assert np.all(out[101:] == 0.0)


# --------------------------------------------------------------------------- #
# Curves as samples
# --------------------------------------------------------------------------- #
def test_the_curves_of_a_set_make_a_feature_matrix(crystals):
    cells = list(crystals.values())
    F = S.computed_curves(cells, SPEC, "xray")
    assert F.shape == (len(cells), len(SPEC.r))
    assert np.all(np.isfinite(F))
    for k, c in enumerate(cells):
        assert F[k] == pytest.approx(S.pair_distribution(c, SPEC, "xray")[1])
    assert len(S.curve_labels(SPEC)) == F.shape[1]
    seen = []
    S.computed_curves(cells, SPEC, "xray", progress=lambda m, f: seen.append(m))
    assert seen and "Pair distribution" in seen[0]


def test_the_kernel_compares_a_measured_curve_with_a_computed_one(tmp_path, crystals):
    """A model is fitted on computed curves and asked about a file: the same grid, one kernel."""
    cells = list(crystals.values())
    computed = S.computed_curves(cells, SPEC, "xray")
    path = os.path.join(str(tmp_path), "measured.gr")
    r, g = S.pair_distribution(cells[1], SPEC, "xray")
    _write(path, r, S.measured_like(SPEC, r, g, q_max=25.0))
    measured = S.curve_matrix([S.read_curve(path)], SPEC.r)
    K = S.curve_kernel(computed, measured)
    assert K.shape == (1, len(cells))
    assert np.all(np.abs(K) <= 1.0 + 1e-9)
    assert int(np.argmax(K[0])) == 1              # it matches the structure it was made from
    assert np.diag(S.curve_kernel(computed)) == pytest.approx(np.ones(len(cells)))


def test_an_empty_set_of_curves_is_an_empty_matrix():
    assert S.curve_matrix([], SPEC.r).shape == (0, len(SPEC.r))
