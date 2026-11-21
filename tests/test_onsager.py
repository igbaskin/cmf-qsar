"""Tests for benchmarks/run_onsager.py: the transference number from collective displacements.

Three exact cases fix the definition. Independent ions give the self-diffusion ratio. Ions that
move as pairs carry no current, so the number is undefined. And a motion that is common to every
molecule changes nothing, because the coordinates are taken relative to the solvent.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import run_onsager as ON                                                     # noqa: E402

FRAMES = 3000
DT = 1.0


def _walks(n, sigma, rng):
    return np.cumsum(rng.normal(0.0, sigma, (FRAMES, n, 3)), axis=1 * 0)


def _box(dp=1.0, dm=3.0, n_ion=150, n_solvent=60, pairs=False, seed=3):
    """Cations and anions that walk independently (or as pairs) among solvent molecules that
    stay where they are. Variance of one step is 2 D dt per axis."""
    rng = np.random.default_rng(seed)
    cat = _walks(n_ion, np.sqrt(2 * dp * DT), rng)
    ani = cat.copy() if pairs else _walks(n_ion, np.sqrt(2 * dm * DT), rng)
    solv = np.zeros((FRAMES, n_solvent, 3)) + rng.uniform(0, 50, (1, n_solvent, 3))
    centres = np.concatenate([cat, ani, solv], axis=1)
    names = ["LI"] * n_ion + ["ANI"] * n_ion + ["S"] * n_solvent
    masses = np.concatenate([np.full(n_ion, 7.0), np.full(n_ion, 145.0),
                             np.full(n_solvent, 90.0)])
    return centres, names, masses


def test_independent_ions_give_the_self_diffusion_ratio():
    """One box gives one sample of a collective displacement, so a single box is noisy whatever
    the number of ions; the mean over many boxes is the ratio of the self-diffusion coefficients."""
    values = []
    for seed in range(24):
        centres, names, masses = _box(dp=1.0, dm=3.0, n_ion=20, seed=seed)
        y_c, y_a, n_c, n_a = ON.collective_trajectories(centres, names, masses, "LI", "ANI")
        res = ON.transference(y_c, y_a, DT, window=(50.0, 300.0))
        assert n_c == n_a == 20
        values.append(res["t"])
    assert np.nanmean(values) == pytest.approx(1.0 / 4.0, abs=0.04)
    assert np.nanstd(values) > 0.02, "a single box is not precise: that is why the pilot looks"


def test_ions_that_move_as_pairs_carry_no_current():
    centres, names, masses = _box(pairs=True)
    y_c, y_a, _, _ = ON.collective_trajectories(centres, names, masses, "LI", "ANI")
    res = ON.transference(y_c, y_a, DT, window=(50.0, 300.0))
    assert res["L_charge"] < 1e-6 * res["L_pp"] or np.isnan(res["t"])


def test_a_motion_common_to_every_molecule_changes_nothing():
    centres, names, masses = _box(n_ion=40)
    rng = np.random.default_rng(11)
    drift = np.cumsum(rng.normal(0, 2.0, (FRAMES, 1, 3)), axis=0)
    a = ON.collective_trajectories(centres, names, masses, "LI", "ANI")
    b = ON.collective_trajectories(centres + drift, names, masses, "LI", "ANI")
    assert np.allclose(a[0], b[0], atol=1e-6) and np.allclose(a[1], b[1], atol=1e-6)


def test_the_solvent_is_the_frame():
    """A solvent that moves while the ions stay put shows up as an opposite motion of both ions
    relative to it, in the same direction for both, so it carries no current."""
    rng = np.random.default_rng(5)
    n_ion, n_solv = 30, 30
    solv_walk = np.cumsum(rng.normal(0.0, 1.0, (FRAMES, 1, 3)), axis=0)
    solv = np.zeros((FRAMES, n_solv, 3)) + solv_walk
    ions = np.zeros((FRAMES, 2 * n_ion, 3))
    centres = np.concatenate([ions, solv], axis=1)
    names = ["LI"] * n_ion + ["ANI"] * n_ion + ["S"] * n_solv
    y_c, y_a, _, _ = ON.collective_trajectories(centres, names, np.ones(60 + 0), "LI", "ANI") \
        if False else ON.collective_trajectories(centres, names, np.ones(len(names)), "LI",
                                                 "ANI")
    assert np.allclose(y_c, y_a)                      # the same shift, so no charge moves
    assert np.allclose(y_c, -n_ion * solv_walk[:, 0, :])


def test_a_run_without_ions_or_solvent_is_refused():
    centres = np.zeros((10, 3, 3))
    with pytest.raises(ON.T.TransportError):
        ON.collective_trajectories(centres, ["S", "S", "S"], np.ones(3), "LI", "ANI")
    with pytest.raises(ON.T.TransportError):
        ON.collective_trajectories(centres, ["LI", "ANI", "LI"], np.ones(3), "LI", "ANI")


def test_the_scatter_of_one_start_is_the_root_mean_square_difference_over_root_two():
    names = [n for v in ON.RD.CONCENTRATION_RUNS.values() for n in v]
    series = {}
    for i, n in enumerate(names):
        series[n] = {"t": 0.30 + (0.1 if i % 2 else 0.0)}
    sc = ON.start_scatter(series)
    assert sc["pairs"] == 6
    assert sc["scatter"] == pytest.approx(0.1 / np.sqrt(2.0))


def test_the_pilot_reads_a_usable_scatter_of_a_twentieth():
    assert ON.USABLE_SCATTER == 0.05 and ON.BLOCKS == 5
