"""The shell references of the site tasks of benchmarks/run_alloy.py (plan item EC12)."""
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))
sys.path.insert(0, os.path.join(HERE, ".."))

import run_alloy as A  # noqa: E402

DATA = os.path.join(A.EXAMPLES, "slabs.extxyz")
needs_data = pytest.mark.skipif(not os.path.exists(DATA), reason="examples/alloy is not here")


def test_the_light_gaussian_kernels_equal_the_benchmark_ones():
    X = np.random.RandomState(0).randn(30, 5)
    old = A.gauss_family(X, "x")
    new = A.gauss_light(X, "x", factors=(0.25, 1.0, 4.0))
    assert sorted(old) == sorted(new)
    for key in old:
        assert np.allclose(old[key], new[key], atol=1e-12)


@pytest.fixture(scope="module")
def data():
    return A.AlloySet()


@needs_data
def test_the_rows_of_the_stored_h_task_are_drawn_again(data):
    path = os.path.join(A.RESULTS, "data_site.json")
    stored = next(t for t in json.load(open(path, encoding="utf-8"))["tasks"]
                  if t["task"] == "site H [all]")
    rows = A.site_rows(data, "H", "all", A.MAX_OVERLAP_SITES)
    assert [r["log"] for r in rows] == stored["rows"]


@needs_data
def test_a_top_site_has_one_atom_in_the_first_shell_and_a_hollow_three(data):
    # most, not all: a relaxed H is not exactly over its site, and some top layers are buckled
    rows = A.select(data, "H", "all")
    tops = [r for r in rows if r["site"] == "top"][:200]
    hollows = [r for r in rows if r["site"] == "fcc"][:200]
    P, E, S = A.shell_columns(data, tops + hollows, "H")
    first = P[:, 0]                          # the number of atoms in the first shell
    assert np.mean(first[:len(tops)] == 1) > 0.6
    assert np.mean(first[len(tops):] == 3) > 0.9
    # the element counts of a shell add up to its number of atoms
    n_e = len(data.elements)
    assert np.allclose(E[:, :n_e].sum(axis=1), first)
    assert np.all(S.sum(axis=1) == 1)


@needs_data
def test_the_shell_references_are_kernels_on_every_site(data):
    rows = A.select(data, "H", "all")[:40]
    fams = A.shell_references(data, rows, "H")
    assert set(fams) == set(A.SHELL_REFERENCES)
    for fam in fams.values():
        assert len(fam) == 1 + len(A.SHELL_FACTORS)
        for K in fam.values():
            assert K.shape == (40, 40)
            assert np.allclose(K, K.T)


@needs_data
def test_the_count_references_give_the_kernels_of_gauss_family(data):
    # the count references use gauss_light, which must give the kernels of the stored run
    rows = A.select(data, "H", "all")[:40]
    N, Q = A.neighbour_counts(data, rows)
    for name, K in A.count_references(data, rows)["neighbour counts of each element in four shells"].items():
        if name != "linear":
            assert np.allclose(K, A.gauss_family(N, "counts")[name], atol=1e-12)
