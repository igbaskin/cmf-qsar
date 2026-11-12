"""From DFT barriers to the activation energy of a conductor (the first half of plan item MS8).

No test needs the downloaded set or the kernels: the hops and the rows are made up.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

import run_conductors as R  # noqa: E402
import run_nebdft as N  # noqa: E402


class _Network:
    """Structures with one lithium site each and hops to its own images: the barrier along
    x, y and z decides the percolation barrier in one, two and three directions."""

    def __init__(self, barriers):
        self.names = [f"cod_{c}" for c in ("7215190", "1570310", "4002499")][:len(barriers)]
        self.structures = [[0] for _ in barriers]            # one site per structure
        self.hops = []
        for k, b in enumerate(barriers):
            first = len(self.hops)
            for axis, level in enumerate(b):
                for sign in (1, -1):
                    off = [0, 0, 0]
                    off[axis] = sign
                    self.hops.append(SimpleNamespace(
                        structure=k, a=0, b=0, offset=tuple(off), length=3.0 + axis,
                        representative=first + 2 * axis, y=level + 0.1, saddle=float("nan"),
                        site_a=float("nan"), site_b=float("nan")))

    def of_structure(self, k):
        return [i for i, h in enumerate(self.hops) if h.structure == k]

    def representatives(self):
        return [i for i, h in enumerate(self.hops) if h.representative == i]


def test_the_share_that_the_hops_of_a_structure_have_in_common():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(400), 4)
    common = rng.normal(size=400)[groups]
    own = rng.normal(size=1600)
    assert R.shared_share(np.sqrt(0.5) * common + np.sqrt(0.5) * own, groups) == \
        pytest.approx(0.5, abs=0.07)
    assert R.shared_share(own, groups) < 0.08
    assert R.shared_share(common, groups) == pytest.approx(1.0)
    assert R.shared_share(own[:5], np.arange(5)) == 0.0          # no group with two members
    assert R.shared_share(np.ones(8), np.repeat(np.arange(4), 2)) == 0.0


def test_values_are_spread_to_the_equivalent_hops_and_give_a_percolation_barrier():
    net = _Network([(0.3, 0.5, 0.9), (0.2, 0.2, 0.7)])
    reps = net.representatives()
    assert len(reps) == 6 and len(net.hops) == 12
    values = R.spread_values(net, reps, np.array([0.3, 0.5, 0.9, 0.2, 0.2, 0.7]))
    assert len(values) == 12 and values[1] == values[0] == 0.3 and values[11] == 0.7
    assert R.thresholds(net, 0, values) == (0.3, 0.9)
    assert R.thresholds(net, 1, values) == (0.2, 0.7)
    empty = _Network([(0.3, 0.5, 0.9)])
    empty.hops = []
    assert R.thresholds(empty, 0, {}) == (float("inf"), float("inf"))


def test_the_barriers_are_drawn_with_a_common_and_an_own_part():
    net = _Network([(0.3, 0.5, 0.9), (0.2, 0.2, 0.7)])
    reps = net.representatives()
    pred = np.array([0.3, 0.5, 0.9, 0.2, 0.2, 0.7])
    exact = R.drawn_thresholds(net, reps, pred, np.zeros(6), 0.5, draws=5)
    assert np.allclose(exact[0], [0.3, 0.9]) and np.allclose(exact[1], [0.2, 0.7])
    sd = np.full(6, 0.2)
    together = R.drawn_thresholds(net, reps, pred, sd, 1.0, draws=300)
    # with only a common shift the order of the hops stays: three minus one direction is fixed
    gap = together[0][:, 1] - together[0][:, 0]
    assert np.allclose(gap[together[0][:, 0] > 0], 0.6)
    apart = R.drawn_thresholds(net, reps, pred, sd, 0.0, draws=300)
    assert np.std(apart[0][:, 1] - apart[0][:, 0]) > 0.05
    assert np.all(apart[0] >= 0.0)                               # no barrier below zero
    # the lowest of several noisy barriers is lower than the lowest of their means
    assert np.median(apart[1][:, 0]) < 0.2
    low, mid, high = R._interval(apart[0][:, 0])
    assert low < mid < high and R._interval(np.array([np.nan])) != R._interval(apart[0][:, 0])
    rows = R.sensitivity(net, reps, pred, sd, 0.3, apart, scales=(0.5, 0.1), draws=200)
    assert [r["scale"] for r in rows] == [1.0, 0.5, 0.1]
    assert rows[0]["interval_width"] > rows[1]["interval_width"] > rows[2]["interval_width"]
    assert rows[2]["barrier_sd"] == pytest.approx(0.02)


def test_the_saddles_of_the_bond_valence_landscape_are_read(tmp_path):
    net = _Network([(0.3, 0.5, 0.9)])
    path = tmp_path / "hops.csv"
    path.write_text(
        "structure,a,b,offset_x,offset_y,offset_z,barrier,site_a,site_b,saddle\n"
        "cod_7215190,0,0,1,0,0,0.4,-0.7,-0.7,-0.3\n"
        "cod_7215190,0,0,0,1,0,0.6,-0.7,-0.7,\n"
        "cod_0000000,0,0,1,0,0,0.4,-0.7,-0.7,-0.3\n")
    assert R.read_saddles(str(path), net) == 1
    assert net.hops[0].saddle == -0.3 and net.hops[0].site_a == -0.7
    assert np.isnan(net.hops[2].saddle)


def _structure(point, one, three, measured, bv=None):
    s = {"measured": measured, "forest": {"point": list(point), "one": list(one),
                                           "three": list(three)}}
    if bv:
        s["bond_valence"] = {"one": bv[0], "three": bv[1]}
    return s


def test_the_comparison_with_the_measured_values():
    rows = [
        _structure((0.30, 0.40), (0.1, 0.25, 0.5), (0.2, 0.35, 0.6), {"low": 0.2, "high": 0.3},
                   bv=(0.5, 0.6)),
        _structure((0.90, 0.90), (0.6, 0.80, 1.0), (0.6, 0.80, 1.0), {"low": 0.4, "high": 0.4}),
        _structure((0.5, 0.5), (0.3, 0.5, 0.7), (0.3, 0.5, 0.7), None)]
    c = R.compare(rows)
    assert c["n"] == 2
    assert c["predicted_one_direction"] == pytest.approx((0.05 + 0.5) / 2)
    assert c["forest_one_direction"] == pytest.approx((0.0 + 0.4) / 2)
    assert c["mean_of_the_measured"] == pytest.approx(0.075)
    assert c["interval_meets_measured"] == 1                     # 0.6 to 1.0 misses 0.4
    assert c["mean_interval_width"] == pytest.approx(0.4)
    assert c["bond_valence_n"] == 1 and c["bond_valence_one_direction"] == pytest.approx(0.25)
    assert R.compare([rows[2]]) == {"n": 0}


def test_the_chain_on_made_up_hops(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    n = 120
    groups = np.repeat(np.arange(n // 2), 2)
    X = rng.normal(size=(n, 6))
    y = 0.8 + 0.3 * X[:, 0] + 0.1 * rng.normal(size=n)
    hops = [SimpleNamespace(length=2.5 + 0.01 * i) for i in range(n)]
    hs = SimpleNamespace(hops=hops, targets=lambda: SimpleNamespace(
        y=y, molecule=groups, atom=np.arange(n)))
    doc = {"structures": [{"hops": [{"bvse": float(v)} for v in 1.0 + y + 0.1
                                    * rng.normal(size=n)]}]}
    monkeypatch.setattr(N, "load", lambda prepared, m: (hs, doc, None, None))
    monkeypatch.setattr(N, "descriptions", lambda hs, jobs, cache, tag="": ({},
                                                                            {"invariant": X}))
    monkeypatch.setattr(N, "FOREST_TREES", 30)
    net = _Network([(0.3, 0.5, 0.9), (0.2, 0.2, 0.7), (0.6, 0.6, 0.6)])
    reps = net.representatives()
    Xc = np.zeros((len(reps), 6))
    Xc[:, 0] = np.linspace(-2.0, 1.0, len(reps))                 # low to high barriers
    monkeypatch.setattr(R, "conductor_rows", lambda fs, say=None: (net, reps, Xc, ["Ge"]))
    import run_hops as H
    monkeypatch.setattr(H, "fieldset_for", lambda hs, model: None)
    out = R.run(str(tmp_path), draws=40, folds=4, scales=(0.5,))
    tr = out["training"]
    assert tr["hops"] == n and tr["mae"] < 0.25 and 0.3 < tr["factor"] < 3.0
    assert tr["bond_valence_correlation"] > 0.5 and "predicted" not in tr
    assert tr["shortest_hop"] == pytest.approx(2.5)
    assert out["conductors"]["distinct"] == 9 and out["conductors"]["hops"] == 18
    assert out["conductors"]["elements_without_a_field"] == ["Ge"]
    first = out["structures"][0]
    assert first["structure"] == "cod_7215190" and first["measured"]["low"] == 0.22
    low, mid, high = first["forest"]["one"]
    assert 0.0 <= low <= mid <= high and len(first["forest"]["point"]) == 2
    assert first["shorter_than_training"] == 0.0 and "bond_valence" not in first
    assert out["comparison"]["n"] == 3
    assert [s["scale"] for s in out["sensitivity"]] == [1.0, 0.5]

    text = R.report(str(tmp_path))
    assert text.startswith("# From DFT barriers to the activation energy of a conductor")
    for part in ("Does the model carry over?", "## Against the measured activation energies",
                 "## How the error of one barrier moves the result", "| page read |"):
        assert part in text
    with open(tmp_path / "table.md", encoding="utf-8") as fh:
        assert fh.read() == text + "\n"
    assert R.report(str(tmp_path / "nothing")) == ""


def test_every_measured_value_names_its_source():
    assert len(R.MEASURED) >= 3                                  # the gate asks for three
    for label, m in R.MEASURED.items():
        assert label.startswith("cod_") and 0.0 < m["low"] <= m["high"] < 2.0
        assert m["source"] and m["sample"]
        assert m["checked"] in ("page read", "search result")
