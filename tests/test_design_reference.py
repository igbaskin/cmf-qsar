"""The reference of the design tutorial: the loop of plan item M11 with a model that needs no
field (`benchmarks/run_prototypes.py --stage cheap`).

No test needs a download: the cells, the pools and the stored results are made up here.
"""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

ase_build = pytest.importorskip("ase.build")
R = pytest.importorskip("run_prototypes")
from cmfqsar import prototypes as P  # noqa: E402


def test_the_linear_kernel_keeps_a_count():
    X = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    K = R.linear_kernel(X)
    assert np.mean(np.diag(K)) == pytest.approx(1.0)
    # no row is normalised: twice the count is four times the diagonal
    assert K[1, 1] == pytest.approx(4.0 * K[0, 0])
    # a column of zeros is dropped and does not change the kernel
    K2 = R.linear_kernel(np.hstack([X, np.zeros((4, 1))]))
    assert np.allclose(K, K2)


def test_the_names_of_the_elements_are_read_site_by_site(monkeypatch):
    from ase import Atoms

    def cell(a, b, xs):
        return Atoms([a, b] + list(xs), scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5), (0.5, 0, 0.5),
                                                          (0.5, 0.5, 0), (0, 0.5, 0.5)],
                     cell=[4.0] * 3, pbc=True)

    atoms = [cell("Sr", "Ti", "OOO"), cell("Ba", "Ti", "OON"), cell("Sr", "Zr", "FOO")]
    monkeypatch.setattr(R.M6, "fetch", lambda task: {})
    monkeypatch.setattr(R.periodic, "structures_from_matbench",
                        lambda table, target=None: (atoms, [0.1, 0.2, 0.3], ["a", "b", "c"]))
    X, y, names = R.element_name_columns("perovskites", 3)
    assert len(X) == 3 and np.allclose(sorted(y), [0.1, 0.2, 0.3])
    assert names == sorted(names)
    assert set(names) == {"A Sr", "A Ba", "B Ti", "B Zr", "X O", "X N", "X F"}
    row = {n: X[0, k] for k, n in enumerate(names)}
    # every row: one A, one B and three X sites
    assert np.allclose(X[:, [k for k, n in enumerate(names) if n.startswith("A ")]].sum(axis=1), 1)
    assert np.allclose(X[:, [k for k, n in enumerate(names) if n.startswith("X ")]].sum(axis=1), 3)
    assert sum(row.values()) == 5.0


def test_a_cell_that_is_not_five_atoms_is_refused(monkeypatch):
    from ase import Atoms

    monkeypatch.setattr(R.M6, "fetch", lambda task: {})
    monkeypatch.setattr(R.periodic, "structures_from_matbench",
                        lambda table, target=None: ([Atoms("SrTiO2", cell=[4.0] * 3, pbc=True)],
                                                    [0.0], ["a"]))
    with pytest.raises(ValueError, match="has 4 atoms and not 5"):
        R.element_name_columns("perovskites", 1)


def test_the_neighbour_pairs_of_an_alloy():
    host = ase_build.bulk("Cu", "fcc", a=3.6, cubic=True) * (2, 2, 2)
    d = host.get_all_distances(mic=True)
    first = float(np.min(d[d > 0.1]))
    i, j = 0, int(np.argmin(np.where(d[0] > 0.1, np.abs(d[0] - first), 9.0)))
    k = int(np.argmax(d[0]))                              # the furthest site from site 0

    def placed(sites):
        a = host.copy()
        symbols = a.get_chemical_symbols()
        for s in sites:
            symbols[s] = "Au"
        a.set_chemical_symbols(symbols)
        return P.Candidate(0, "x", (), a)

    X = R.neighbour_pair_columns([placed([i, j]), placed([0, k]), placed([i, j, 8])], host)
    assert X.shape == (3, R.CHEAP_CLOSED_SHELLS)
    assert X[0].tolist() == [1.0, 0.0, 0.0]               # one pair of nearest neighbours
    assert X[1].sum() == 0.0                              # further apart than the third shell
    assert X[2].sum() >= 1.0


def test_the_ordering_finds_the_best_candidate_of_an_additive_pool():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 4, size=(300, 3)).astype(float)
    y = X @ np.array([1.0, 2.0, 3.0]) + 0.01 * rng.normal(size=300)
    K = R.linear_kernel(X)
    o = R.ordering_rows(K, y, R.starts_for(len(y), 30, 3))
    assert o["spearman_median"] > 0.99 and o["r2_median"] > 0.99
    assert o["place_median"] is not None and o["place_median"] <= 3
    assert len(o["runs"]) == 3


def _fake_record(name, model, size, n):
    rule = {"best_mean": -1.0, "rank_median": 1.0, "beat_start_runs": 9, "of_runs": 10,
            "reached_top_runs": 10, "reached_top_median": 2.0, "runs": []}
    return {"pool_name": name, "model": model, "start": size, "limit": n, "pool": {"n": n},
            "greedy": rule, "ucb": rule,
            "ordering": {"spearman_median": 0.9, "r2_median": 0.8, "place_median": 1.0}}


def test_the_section_of_the_table(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "RESULTS", str(tmp_path))
    assert R.cheap_section() == []
    field_rule = {"best_mean": -0.5, "rank_median": 3.0, "beat_start_runs": 7, "of_runs": 10,
                  "reached_top_runs": 8, "reached_top_median": 16.0}
    loops = {f"perovskites | three-body | 4000 | start {s}":
             {"pool": {"n": 4000}, "random": field_rule, "greedy": field_rule, "ucb": field_rule}
             for s in R.STARTS}
    (tmp_path / "loop.json").write_text(json.dumps(loops))
    cheap = {f"perovskites | element names | 4000 | start {s}":
             _fake_record("perovskites", "element names", s, 4000) for s in R.STARTS}
    closed = _fake_record("closed loop", "neighbour pairs", 10, 400)
    for r in ("greedy", "ucb"):
        closed[r] = dict(closed[r], found_global_runs=5, of_runs=5)
    cheap["closed loop | neighbour pairs"] = closed
    (tmp_path / "cheap.json").write_text(json.dumps(cheap))
    text = "\n".join(R.cheap_section())
    assert "## The reference: the same loop with a model that needs no field" in text
    assert "| perovskites | random choice | 25 | random |" in text
    assert "| perovskites | three-body fields | 100 | greedy |" in text
    assert ("| perovskites | element names | 25 | greedy | -1.0000 | 1 of 4000 | 9/10 | 10/10 | "
            "2 | 1 | +0.900 |") in text
    assert "| neighbour pairs | greedy | -1.0000 | 1 of 400 | 5/5 | 1 | +0.900 |" in text
