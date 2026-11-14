"""The second stage of the point-mutant benchmark: every position, from the wild type alone.

The stage reads the wild-type structure and the label table of each domain that the repository
holds, draws a few substitutions for each position, and compares a model of the number of atoms
around the site with the field of the surroundings.  These tests check the parts that can be
wrong without a failure in the measurement: the kernel on columns, the draw of substitutions, and
the expansion of a kernel over positions to a kernel over substitutions.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import run_mutants as RM  # noqa: E402
from cmfqsar import mutants as MU  # noqa: E402


def test_a_kernel_on_columns_is_symmetric_with_ones_on_the_diagonal():
    X = np.random.RandomState(0).normal(size=(12, 3))
    K = RM.gaussian_columns(X)
    assert np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1.0)
    assert K.min() > 0.0 and K.max() <= 1.0 + 1e-12


def test_a_kernel_on_columns_does_not_depend_on_the_unit_of_a_column():
    X = np.random.RandomState(1).normal(size=(15, 4))
    Y = X * np.array([1.0, 100.0, 0.01, 7.0]) + np.array([5.0, -3.0, 0.0, 2.0])
    assert np.allclose(RM.gaussian_columns(X), RM.gaussian_columns(Y))


def test_a_wider_kernel_finds_every_pair_more_alike():
    X = np.random.RandomState(2).normal(size=(10, 3))
    narrow, wide = RM.gaussian_columns(X, 0.5), RM.gaussian_columns(X, 2.0)
    off = ~np.eye(10, dtype=bool)
    assert np.all(wide[off] > narrow[off])


def test_a_constant_column_is_refused_and_not_silently_divided_by_zero():
    X = np.c_[np.arange(6.0), np.ones(6)]
    with pytest.warns(RuntimeWarning):
        K = RM.gaussian_columns(X)
    assert not np.all(np.isfinite(K))


@pytest.fixture(scope="module")
def draw():
    return RM.SiteSamples({}, RM.DOMAINS, 4, 0)


def test_the_draw_keeps_a_position_whole_and_takes_at_most_four_of_it(draw):
    sites = {}
    for d, p in zip(draw.domain, draw.position):
        sites[(d, p)] = sites.get((d, p), 0) + 1
    assert len(sites) == 280
    assert max(sites.values()) == 4
    assert len(draw) == sum(sites.values())


def test_the_same_seed_draws_the_same_substitutions_and_another_seed_does_not(draw):
    again = RM.SiteSamples({}, RM.DOMAINS, 4, 0)
    other = RM.SiteSamples({}, RM.DOMAINS, 4, 1)
    assert draw.label == again.label
    assert draw.label != other.label
    assert sorted(set(zip(draw.domain, draw.position))) == sorted(set(zip(other.domain, other.position)))


def test_every_drawn_label_is_in_the_table_and_has_its_own_value(draw):
    table = {(d, r["label"]): float(r["ddG_ML"]) for d in RM.DOMAINS for r in RM.read_table(d)}
    for d, lab, y in zip(draw.domain, draw.label, draw.y):
        assert table[(d, lab)] == y


def test_substitutions_of_one_position_share_a_row_of_the_expanded_kernel(draw):
    keys = sorted(set(zip(draw.domain, draw.position)))
    K = np.random.RandomState(3).normal(size=(len(keys), len(keys)))
    K = K + K.T
    big = RM.expand_position_kernel(keys, K, draw)
    site = np.array([f"{d}:{p}" for d, p in zip(draw.domain, draw.position)])
    first = {s: i for i, s in reversed(list(enumerate(site)))}
    for i in range(0, len(draw), 37):
        assert np.array_equal(big[i], big[first[site[i]]])
    i, j = 0, len(draw) - 1
    ki, kj = keys.index((draw.domain[i], draw.position[i])), keys.index((draw.domain[j], draw.position[j]))
    assert big[i, j] == K[ki, kj]


@pytest.fixture(scope="module")
def one_domain():
    return RM.site_contexts(("1O6X",))


def test_the_contexts_cover_every_position_at_every_envelope(one_domain):
    positions = {int(r["position"]) for r in RM.read_table("1O6X")}
    for rho in RM.CONTEXT_RHOS:
        assert {p for (r, d, p) in one_domain if r == float(rho)} == positions


def test_a_wider_envelope_holds_more_atoms_around_every_site(one_domain):
    narrow, wide = max(RM.CONTEXT_RHOS), min(RM.CONTEXT_RHOS)
    positions = sorted({p for (_r, _d, p) in one_domain})
    for p in positions:
        assert one_domain[(float(wide), "1O6X", p)].n_atoms >= one_domain[(float(narrow), "1O6X", p)].n_atoms


def test_the_burial_count_is_the_atom_count_of_the_envelope_with_the_three_columns(one_domain):
    s = RM.SiteSamples(one_domain, ("1O6X",), 4, 0)
    X = RM.burial_count_rows(s)
    sub, _names = MU.substitution_features(s.mutation)
    assert X.shape == (len(s), 4)
    assert np.array_equal(X[:, 1:], sub[:, :3])
    for k in (0, len(s) // 2, len(s) - 1):
        assert X[k, 0] == one_domain[(RM.SITE_RHO, "1O6X", s.position[k])].n_atoms
    assert X[:, 0].min() > 0


def test_a_buried_site_has_more_atoms_around_it_than_an_exposed_one(one_domain):
    """The first and the last residue of a chain are at the ends, where fewer atoms surround them
    than at the centre of a small domain."""
    counts = {p: one_domain[(RM.SITE_RHO, "1O6X", p)].n_atoms
              for (r, d, p) in one_domain if r == RM.SITE_RHO}
    positions = sorted(counts)
    ends = np.mean([counts[positions[0]], counts[positions[-1]]])
    assert max(counts.values()) > 1.5 * ends
