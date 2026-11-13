"""Transport coefficients of a simulated electrolyte (plan item BT9).

Everything here runs on made-up trajectories whose answer is known: a random walk has a mean
squared displacement that grows by the variance of one step per frame. No simulation is run; the
one test that needs OpenMM builds a system of three particles and is skipped without it.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from cmfqsar import transport as T  # noqa: E402


def walk(frames, particles, step, seed=0):
    """Unwrapped positions of independent random walkers: every coordinate moves by a normal
    step of the standard deviation `step` per frame."""
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0.0, step, (frames, particles, 3)), axis=0)


def direct_msd(r, lag):
    d = r[lag:] - r[:-lag]
    return float((d ** 2).sum(axis=2).mean())


# ------------------------------------------------------------------ mean squared displacement

def test_the_fast_sum_is_the_direct_sum():
    r = walk(60, 7, 0.3, seed=1)
    fast = T.msd(r)
    assert fast[0] == 0.0
    for lag in (1, 2, 5, 17, 59):
        assert fast[lag] == pytest.approx(direct_msd(r, lag), rel=1e-9)


def test_batches_do_not_change_the_sum():
    r = walk(40, 9, 0.5, seed=2)
    assert T.msd(r, batch=2) == pytest.approx(T.msd(r, batch=64), rel=1e-10)
    assert len(T.msd(r, max_lag=10)) == 11


def test_a_random_walk_grows_by_three_step_variances_per_frame():
    r = walk(2000, 200, 0.2, seed=3)
    curve = T.msd(r, 200)
    assert curve[100] == pytest.approx(3 * 0.2 ** 2 * 100, rel=0.05)


def test_what_the_displacement_refuses():
    with pytest.raises(T.TransportError, match="frames, particles, 3"):
        T.msd(np.zeros((10, 3)))
    with pytest.raises(T.TransportError, match="at least two frames"):
        T.msd(np.zeros((1, 4, 3)))


def test_a_drift_of_the_whole_box_is_taken_out():
    r = walk(50, 6, 0.3, seed=4)
    masses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    drift = np.linspace(0.0, 30.0, 50)[:, None, None] * np.array([1.0, -2.0, 0.5])
    assert T.remove_drift(r + drift, masses) == pytest.approx(T.remove_drift(r, masses))
    still = T.remove_drift(r, masses)
    com = (still * masses[None, :, None]).sum(axis=1) / masses.sum()
    assert np.abs(com).max() < 1e-12
    with pytest.raises(T.TransportError, match="masses"):
        T.remove_drift(r, masses[:3])


# ------------------------------------------------------------------ the diffusion coefficient

def test_the_window_starts_late_and_ends_at_a_tenth():
    assert T.default_window(4000.0) == (100.0, 400.0)
    assert T.default_window(2000.0) == (50.0, 200.0)
    # too short for a start at 50 ps: a twentieth to a quarter of the run
    assert T.default_window(400.0) == (20.0, 100.0)


def test_a_slope_needs_points_inside_the_window():
    t = np.arange(10.0)
    assert T.fit_slope(t, 3.0 * t + 1.0, (2.0, 8.0))["slope"] == pytest.approx(3.0)
    assert T.fit_slope(t, 2.0 * t, (2.0, 8.0))["exponent"] == pytest.approx(1.0)
    with pytest.raises(T.TransportError, match="too short"):
        T.fit_slope(t, t, (20.0, 30.0))


def test_the_diffusion_coefficient_of_a_random_walk():
    step, dt = 0.25, 1.0                              # angstrom per frame, picoseconds
    res = T.diffusion(walk(4000, 150, step, seed=5), dt)
    expected = step ** 2 / (2.0 * dt) * T.A2_PER_PS   # slope 3 s^2 / dt, divided by six
    assert res["d"] == pytest.approx(expected, rel=0.06)
    assert res["exponent"] == pytest.approx(1.0, abs=0.08)
    # 4000 frames one picosecond apart are 3999 ps, so the window is a little under 100 to 400
    assert len(res["blocks"]) == 5 and res["window_ps"] == pytest.approx([99.975, 399.9])
    assert res["particles"] == 150 and res["length_ps"] == pytest.approx(3999.0)
    assert len(res["lags_ps"]) == len(res["msd"]) == 16


def test_the_error_is_the_error_of_the_estimate_it_stands_next_to():
    """Over many made-up runs the scatter of the coefficient is what its block error says, to
    within the factor that five blocks can resolve. An error from blocks fitted in shorter
    windows of their own was three times too small on the same walks."""
    step, expected = 0.25, 0.25 ** 2 / 2.0 * T.A2_PER_PS
    values, errors = [], []
    for seed in range(12):
        res = T.diffusion(walk(2000, 40, step, seed=100 + seed), 1.0)
        values.append(res["d"])
        errors.append(res["se"])
    scatter = float(np.std(values, ddof=1))
    assert np.mean(values) == pytest.approx(expected, rel=0.05)
    assert 0.5 * scatter < float(np.mean(errors)) < 2.0 * scatter


def test_more_particles_give_a_smaller_error():
    few = [T.diffusion(walk(2000, 8, 0.25, seed=k), 1.0)["se"] for k in range(6)]
    many = [T.diffusion(walk(2000, 256, 0.25, seed=k), 1.0)["se"] for k in range(6)]
    assert np.mean(many) < 0.5 * np.mean(few)


def test_one_block_has_no_error_and_a_short_run_is_refused():
    res = T.diffusion(walk(400, 20, 0.25, seed=7), 1.0, blocks=1)
    assert np.isnan(res["se"]) and res["blocks"] == []
    with pytest.raises(T.TransportError, match="too short"):
        T.diffusion(walk(40, 20, 0.25, seed=7), 1.0, window=(30.0, 60.0))
    with pytest.raises(T.TransportError, match="frames, particles, 3"):
        T.diffusion(np.zeros((40, 3)), 1.0)


# ------------------------------------------------------------------ salt, share, concentration

def test_the_salt_diffuses_as_the_harmonic_mean_of_its_ions():
    assert T.salt_diffusivity(2e-10, 2e-10) == pytest.approx(2e-10)
    assert T.salt_diffusivity(1e-10, 3e-10) == pytest.approx(1.5e-10)
    assert T.salt_diffusivity(1e-10, 3e-10, factor=2.0) == pytest.approx(3.0e-10)
    with pytest.raises(T.TransportError, match="above zero"):
        T.salt_diffusivity(0.0, 1e-10)


def test_the_share_of_the_cation_and_the_concentration():
    assert T.transport_number(1e-10, 3e-10) == pytest.approx(0.25)
    # one ion pair in 1660.54 cubic angstrom is one mole per litre
    assert T.molarity(1, 1.0e27 / T.AVOGADRO) == pytest.approx(1.0)
    assert T.molarity(30, 30.0e27 / T.AVOGADRO) == pytest.approx(1.0)


def test_the_conductivity_of_ions_that_move_on_their_own():
    volume, temperature = 1.0e27 / T.AVOGADRO, 300.0      # one mole per litre per pair
    sigma = T.nernst_einstein_conductivity(1e-10, 2e-10, 1, volume, temperature)
    molar = 96485.33212 ** 2 * 3e-10 / (8.314462618 * temperature)    # F^2 (D+ + D-) / (R T)
    assert sigma == pytest.approx(molar * 1000.0, rel=1e-6)           # 1000 mol/m3
    assert T.nernst_einstein_conductivity(1e-10, 2e-10, 1, volume, temperature, charge=0.5) \
        == pytest.approx(sigma / 4.0)


def test_ions_that_move_as_pairs_carry_no_current():
    r = walk(500, 20, 0.3, seed=8)
    res = T.conductivity(r, r + 5.0, 1.0, 4.0e4, 298.15, window=(20.0, 100.0))
    assert abs(res["sigma"]) < 1e-12


def test_ions_that_move_on_their_own_conduct_as_nernst_and_einstein_say():
    step, dt, n, volume, temperature = 0.25, 1.0, 40, 6.0e4, 298.15
    cations = walk(20000, n, step, seed=9)
    anions = np.zeros((20000, n, 3))                     # the anions stand still
    res = T.conductivity(cations, anions, dt, volume, temperature, window=(20.0, 200.0))
    d = step ** 2 / (2.0 * dt) * T.A2_PER_PS
    expected = T.nernst_einstein_conductivity(d, 0.0, n, volume, temperature)
    assert res["sigma"] == pytest.approx(expected, rel=0.4)
    assert res["se"] > 0
    with pytest.raises(T.TransportError, match="frames"):
        T.conductivity(cations[:10], anions, dt, volume, temperature)


# ------------------------------------------------------------------ the finite box

def test_boxes_of_several_sizes_give_the_infinite_box_and_the_viscosity():
    temperature, viscosity, d_inf = 298.15, 3.0e-3, 2.5e-10
    sides = np.array([30.0, 40.0, 55.0])
    d = np.array([d_inf - T.yeh_hummer_correction(L, viscosity, temperature) for L in sides])
    fit = T.extrapolate_size(d, sides, temperature)
    assert fit["d_infinite"] == pytest.approx(d_inf, rel=1e-9)
    assert fit["viscosity"] == pytest.approx(viscosity, rel=1e-9)
    assert fit["residual"] < 1e-20
    # the same line as a factor: one radius corrects a box of any size
    radius = T.hydrodynamic_radius(fit["d_infinite"], fit["viscosity"], temperature)
    for L, value in zip(sides, d):
        assert value * T.size_factor(radius, L) == pytest.approx(d_inf, rel=1e-9)


def test_the_errors_weight_the_line():
    sides = np.array([30.0, 40.0, 55.0])
    d = np.array([1.0e-10, 1.3e-10, 1.2e-10])
    plain = T.extrapolate_size(d, sides, 298.15)
    weighted = T.extrapolate_size(d, sides, 298.15, se=[1e-12, 1e-12, 1e-10])
    assert weighted["d_infinite"] != pytest.approx(plain["d_infinite"], rel=1e-3)
    # a coefficient that rises as the box shrinks gives no viscosity
    assert np.isnan(T.extrapolate_size([2e-10, 1e-10], [30.0, 60.0], 298.15)["viscosity"])
    with pytest.raises(T.TransportError, match="at least two boxes"):
        T.extrapolate_size([1e-10], [30.0], 298.15)


def test_species_of_one_liquid_share_the_slope_against_the_size():
    temperature, viscosity = 298.15, 2.0e-3
    sides = [27.0, 34.0, 43.0]
    limits = {"cation": 1.0e-10, "anion": 1.6e-10, "EMC": 3.0e-10}
    d = {k: [v - T.yeh_hummer_correction(L, viscosity, temperature) for L in sides]
         for k, v in limits.items()}
    fit = T.joint_size_fit(d, sides, temperature)
    assert fit["viscosity"] == pytest.approx(viscosity, rel=1e-8)
    for k, v in limits.items():
        assert fit["species"][k]["d_infinite"] == pytest.approx(v, rel=1e-8)
        assert fit["species"][k]["radius"] == pytest.approx(
            T.hydrodynamic_radius(v, viscosity, temperature), rel=1e-7)
    assert fit["correction"] == pytest.approx(
        [T.yeh_hummer_correction(L, viscosity, temperature) for L in sides])
    assert fit["chi"] < 1e-6


def test_the_species_known_best_fixes_the_slope_for_the_others():
    """The ions are few and their coefficients scatter; the solvent is exact. With the errors
    given, the common slope is the solvent's, and the ions get the right limit on average."""
    temperature, viscosity = 298.15, 2.0e-3
    sides = [27.0, 34.0, 43.0]
    exact = [3.0e-10 - T.yeh_hummer_correction(L, viscosity, temperature) for L in sides]
    noisy = [1.0e-10 - T.yeh_hummer_correction(L, viscosity, temperature) + e
             for L, e in zip(sides, (1.5e-11, -2.0e-11, 1.0e-11))]
    fit = T.joint_size_fit({"cation": noisy, "EMC": exact}, sides, temperature,
                           se={"cation": [2e-11] * 3, "EMC": [1e-13] * 3})
    assert fit["viscosity"] == pytest.approx(viscosity, rel=0.01)
    alone = T.extrapolate_size(noisy, sides, temperature)
    assert abs(fit["species"]["cation"]["d_infinite"] - 1.0e-10) \
        < abs(alone["d_infinite"] - 1.0e-10)


def test_what_the_joint_fit_refuses():
    with pytest.raises(T.TransportError, match="at least two boxes"):
        T.joint_size_fit({"cation": [1e-10]}, [30.0], 298.15)
    with pytest.raises(T.TransportError, match="coefficients for"):
        T.joint_size_fit({"cation": [1e-10]}, [30.0, 40.0], 298.15)
    # coefficients that fall as the box grows give no viscosity
    fit = T.joint_size_fit({"cation": [2e-10, 1e-10]}, [30.0, 60.0], 298.15)
    assert np.isnan(fit["viscosity"]) and np.isnan(fit["species"]["cation"]["radius"])


def test_a_box_too_small_for_its_species_is_refused():
    assert T.size_factor(0.0, 30.0) == 1.0
    assert T.size_factor(3.0, 30.0) == pytest.approx(1.0 / (1.0 - T.YEH_HUMMER * 0.1))
    with pytest.raises(T.TransportError, match="too small"):
        T.size_factor(12.0, 30.0)


# ------------------------------------------------------------------ the run and its record

def test_the_centres_of_molecules_whose_atoms_are_contiguous():
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    atom_mass = np.array([1.0, 3.0, 7.0])
    starts = np.array([0, 2])
    centres = T._centres(positions, starts, np.add.reduceat(atom_mass, starts), atom_mass)
    assert centres == pytest.approx(np.array([[1.5, 0.0, 0.0], [10.0, 10.0, 10.0]]))


def test_an_unknown_thermostat_is_refused_before_anything_is_built(tmp_path):
    form = T.E.Formulation("LiPF6", {"EC": 0.3, "EMC": 0.7}, 1.0)
    with pytest.raises(T.TransportError, match="nose-hoover or langevin"):
        T.run_transport(form, str(tmp_path), T.TransportSettings(thermostat="berendsen"))


def test_the_steps_of_a_stretch_of_time():
    s = T.TransportSettings(timestep_fs=2.0)
    assert s.steps(1.0) == 500 and s.steps(0.0005) == 1


def test_scaled_charges_scale_the_listed_pairs_too():
    openmm = pytest.importorskip("openmm")
    from openmm import unit
    system = openmm.System()
    force = openmm.NonbondedForce()
    for q in (1.0, -0.4, 0.2):
        system.addParticle(1.0)
        force.addParticle(q, 0.3, 0.1)
    force.addException(0, 1, -0.4, 0.3, 0.1)       # both scaled
    force.addException(1, 2, -0.08, 0.3, 0.1)      # one scaled
    system.addForce(force)
    T.scale_ion_charges(system, [0, 1], 0.8)
    charges = [force.getParticleParameters(i)[0].value_in_unit(unit.elementary_charge)
               for i in range(3)]
    assert charges == pytest.approx([0.8, -0.32, 0.2])
    products = [force.getExceptionParameters(k)[2].value_in_unit(unit.elementary_charge ** 2)
                for k in range(2)]
    assert products == pytest.approx([-0.4 * 0.64, -0.08 * 0.8])


def _stored_run(tmp_path, frames=600, pairs=12, solvent=30, seed=11):
    """A made-up run: cations, anions and one solvent as random walkers of three speeds."""
    rng = np.random.default_rng(seed)
    names = ["EC"] * solvent + ["ANI"] * pairs + ["LI"] * pairs
    steps = np.array([0.4] * solvent + [0.3] * pairs + [0.2] * pairs)
    centres = np.cumsum(rng.normal(0.0, 1.0, (frames, len(names), 3)) * steps[None, :, None],
                        axis=0)
    path = os.path.join(str(tmp_path), "centres.npz")
    np.savez_compressed(path, centres=centres.astype(np.float32), names=np.array(names),
                        masses=np.array([88.0] * solvent + [145.0] * pairs + [6.9] * pairs),
                        charges=np.array([0.0] * solvent + [-1.0] * pairs + [1.0] * pairs),
                        dt_ps=1.0, side=30.0)
    return {"centres": path, "ion_residue": "LI", "anion_residue": "ANI",
            "temperature": 298.15, "volume": 27000.0, "side": 30.0}


def test_a_stored_run_is_read_back_and_analysed(tmp_path):
    run = _stored_run(tmp_path)
    data = T.read_centres(run["centres"])
    assert data["centres"].shape == (600, 54, 3) and data["dt_ps"] == 1.0
    res = T.analyse(run)
    assert res["pairs"] == 12 and set(res["solvents"]) == {"EC"}
    # the three species were given three speeds, and they come out in that order
    assert res["cation"]["d"] < res["anion"]["d"] < res["solvents"]["EC"]["d"]
    assert res["salt"] == pytest.approx(T.salt_diffusivity(res["cation"]["d"],
                                                           res["anion"]["d"]))
    assert 0.0 < res["transport_number"] < 0.5
    assert res["molarity"] == pytest.approx(T.molarity(12, 27000.0))
    assert "ionicity" in res and np.isfinite(res["conductivity"]["sigma"])


def test_a_run_can_be_cut_to_read_what_a_shorter_one_would_give(tmp_path):
    run = _stored_run(tmp_path)
    whole = T.analyse(run)
    half = T.analyse(run, last_ps=299.0)
    late = T.analyse(run, first_ps=300.0)
    assert whole["length_ps"] == pytest.approx(599.0)
    assert half["length_ps"] == pytest.approx(299.0)
    assert late["length_ps"] == pytest.approx(299.0)
    assert half["cation"]["d"] != late["cation"]["d"]


def test_a_run_without_its_ions_is_refused(tmp_path):
    run = _stored_run(tmp_path)
    run["ion_residue"] = "NA"
    with pytest.raises(T.TransportError, match="no cation or no anion"):
        T.analyse(run)
