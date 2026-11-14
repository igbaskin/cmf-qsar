"""From hop barriers to a diffusion coefficient and a conductivity (plan item MS8).

The long-time diffusion coefficient of a periodic network is checked against cases with a
closed answer (a cubic lattice, a chain of two kinds of bond, a landscape with site energies),
against a kinetic Monte Carlo simulation, and for its independence of the window of rates.
"""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar import conduction as C  # noqa: E402

A = 2.0


def _cubic():
    edges, vecs = [], []
    for ax in range(3):
        for s in (1, -1):
            o = [0, 0, 0]
            o[ax] = s
            edges.append((0, 0, tuple(o)))
            v = np.zeros(3)
            v[ax] = s * A
            vecs.append(v)
    return edges, np.array(vecs)


def _chain():
    """Two sites in a cell of length 2A; bond 0-1 inside the cell, bond 1-0 across it."""
    edges = [(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0)), (1, 0, (1, 0, 0)), (0, 1, (-1, 0, 0))]
    vecs = np.array([[A, 0, 0], [-A, 0, 0], [A, 0, 0], [-A, 0, 0]], float)
    return edges, vecs


def test_a_cubic_lattice():
    edges, vecs = _cubic()
    D = C.network_diffusion(1, edges, vecs, np.full(6, 3.0))
    assert np.allclose(D, 3.0 * A * A * np.eye(3))


def test_two_kinds_of_bond_in_series():
    edges, vecs = _chain()
    g1, g2 = 1.0, 1e-3
    D = C.network_diffusion(2, edges, vecs, np.array([g1, g1, g2, g2]))
    assert D[0, 0] == pytest.approx(A * A * 2 * g1 * g2 / (g1 + g2), rel=1e-12)
    assert np.allclose(D[1:, 1:], 0.0)


def test_a_network_that_does_not_cross_the_cell_does_not_diffuse():
    edges, vecs = _chain()
    assert np.allclose(C.network_diffusion(2, edges[:2], vecs[:2], np.ones(2)), 0.0)
    assert np.allclose(C.network_diffusion(2, [], np.zeros((0, 3)), np.zeros(0)), 0.0)


def test_the_critical_rate_is_the_slowest_hop_a_crossing_needs():
    edges, _ = _chain()
    assert C.critical_rate(2, edges, np.array([1.0, 1.0, 1e-3, 1e-3])) == pytest.approx(1e-3)


def test_a_landscape_with_site_energies():
    edges, vecs = _chain()
    kt = C.KB_EV * 298.15
    E = np.array([0.0, 0.12])
    saddle = np.array([0.30, 0.30, 0.45, 0.45])
    rates, pi = C.landscape_rates(edges, E, saddle, nu=1.0)
    assert np.isclose(pi[0] * rates[0], pi[1] * rates[1])              # reversible
    D = C.network_diffusion(2, edges, vecs, rates, site_weights=pi)
    Z = np.exp(-E / kt).sum()
    c1, c2 = np.exp(-0.30 / kt) / Z, np.exp(-0.45 / kt) / Z
    assert D[0, 0] == pytest.approx((2 * A) ** 2 / (1 / c1 + 1 / c2), rel=1e-9)
    # equal site energies give back the form without weights
    rates0, pi0 = C.landscape_rates(edges, np.zeros(2), saddle, nu=1.0)
    D0 = C.network_diffusion(2, edges, vecs, rates0, site_weights=pi0)
    assert np.allclose(D0, C.network_diffusion(2, edges, vecs, rates0))


def _random_network(seed=1, n=4, size=4.0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, size, size=(n, 2))
    edges, vecs = [], []
    for i in range(n):
        for j in range(n):
            for ox in (-1, 0, 1):
                for oy in (-1, 0, 1):
                    d = pos[j] + np.array([ox, oy]) * size - pos[i]
                    if (i != j or ox or oy) and np.linalg.norm(d) < 2.6:
                        edges.append((i, j, (ox, oy, 0)))
                        vecs.append([d[0], d[1], 0.0])
    return rng, edges, np.array(vecs)


def test_the_closed_form_agrees_with_a_kinetic_monte_carlo_simulation():
    rng, edges, vecs = _random_network()
    bar = C.pair_barriers(edges, rng.uniform(0.10, 0.25, len(edges)))
    rates = C.rates_from_barriers(bar, nu=1.0, temperature=600.0)
    exact = C.network_diffusion(4, edges, vecs, rates)
    sim = C.simulate_walk(4, edges, vecs, rates, walkers=3000, steps=1500, seed=2)
    assert np.trace(sim) == pytest.approx(np.trace(exact), rel=0.1)


def test_the_window_of_rates_does_not_move_the_result():
    rng, edges, vecs = _random_network()
    bar = C.pair_barriers(edges, rng.uniform(0.1, 2.0, len(edges)))
    rates = C.rates_from_barriers(bar)
    d5 = np.trace(C.network_diffusion(4, edges, vecs, rates, window=1e5))
    d7 = np.trace(C.network_diffusion(4, edges, vecs, rates, window=1e7))
    assert d5 == pytest.approx(d7, rel=1e-4)


def test_the_two_directions_of_a_hop_share_their_mean_barrier():
    edges = [(0, 1, (0, 0, 0)), (1, 0, (0, 0, 0)), (1, 2, (0, 0, 1))]
    out = C.pair_barriers(edges, [0.2, 0.4, 0.7])
    assert np.allclose(out, [0.3, 0.3, 0.7])


def test_the_lattice_gas_conductivity():
    assert C.lattice_gas_conductivity(1e10, 10, 0.0, 1000.0) == 0.0
    assert C.lattice_gas_conductivity(1e10, 10, 1.0, 1000.0) == 0.0
    # q^2 (N/V) c (1-c) D / kT, by hand, in S/cm
    s = C.lattice_gas_conductivity(1e10, 10, 0.5, 1000.0, temperature=300.0)
    by_hand = (C.CHARGE ** 2 * (10 / 1e-27) * 0.25 * 1e-10 / (C.KB_J * 300.0)) / 100.0
    assert s == pytest.approx(by_hand, rel=1e-12)
    with pytest.raises(ValueError):
        C.lattice_gas_conductivity(1.0, 1, 1.5, 1.0)


# ---------------------------------------------------------------------- the benchmark


def _bench():
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import run_conductivity as RC
    return RC


def test_the_interval_counts_a_draw_without_a_path():
    RC = _bench()
    finite = np.linspace(-5, -1, 390)
    v = np.concatenate([np.full(10, -np.inf), finite])
    lo, mid, hi = RC._interval(v)
    # 400 draws: the 97.5 % point is the value at index ceil(0.975 * 399) = 390 of the sorted
    # draws, which is finite value 380
    assert lo == -np.inf and np.isfinite(mid) and hi == pytest.approx(finite[380])
    assert RC._interval(np.full(10, -np.inf)) == [-np.inf, -np.inf, -np.inf]


def _structure(label, point, lo, hi, measured=None, bv=None):
    s = {"structure": label, "compound": label, "disorder": "", "sites": 4, "occupancy": 0.5,
         "volume": 100.0, "hops": 8, "log10_sigma_point": point,
         "log10_sigma_interval": [lo, (lo + hi) / 2, hi],
         "measured": None if measured is None else {"sigma": 10 ** measured}}
    if bv is not None:
        s["log10_sigma_bond_valence"] = bv
    return s


def test_the_comparison_and_its_gate():
    RC = _bench()
    rows = [_structure("a", -3.0, -6.0, 0.0, -2.0, -2.5), _structure("b", -9.0, -12.0, -5.0, -3.0),
            _structure("c", -1.0, -2.0, 1.0, -1.0), _structure("d", -4.0, -8.0, 0.0)]
    c = RC.compare(rows)
    assert c["structures"] == 3 and c["inside"] == 2 and c["gate_met"]
    assert c["rows"][1]["point_minus_measured"] == pytest.approx(-6.0)
    assert c["rows"][0]["others_mean_minus_measured"] == pytest.approx(0.0)
    assert not RC.compare(rows[:2])["gate_met"]


def test_the_table_is_written(tmp_path):
    RC = _bench()
    rows = [_structure("cod_7215190", -2.0, -5.0, 1.0, -1.9, -5.8)]
    d = {"temperature": 298.15, "draws": 10, "structures": rows, "comparison": RC.compare(rows),
         "sensitivity": [{"barrier_sd": 0.5, "width": 20.0}],
         "width_attempt_frequency_only": 1.0, "width_barriers_only": 19.0}
    with open(tmp_path / "conductivity.json", "w") as fh:
        json.dump(d, fh)
    text = RC.report(str(tmp_path))
    assert "Against the measured conductivities" in text and (tmp_path / "table.md").exists()
