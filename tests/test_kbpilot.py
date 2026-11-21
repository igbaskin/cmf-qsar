"""Tests for benchmarks/run_kbpilot.py: the thermodynamic factor from number fluctuations.

Two exact limits fix the definition. Independent ions give a factor of 1, whatever the solvent.
Ions that sit in pairs, each cation with one anion, give 1/2, because the pairs are then the
neutral species.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "benchmarks"))

import run_kbpilot as KB                                                      # noqa: E402

SIDE = 50.0


def _box(frames=400, pairs=False, n_ion=60, solvents=(300, 200), seed=1):
    rng = np.random.default_rng(seed)
    names = ["LI"] * n_ion + ["ANI"] * n_ion
    for i, n in enumerate(solvents):
        names += [f"S{i}"] * n
    out = np.empty((frames, len(names), 3))
    for f in range(frames):
        x = rng.uniform(0.0, SIDE, (len(names), 3))
        if pairs:
            x[n_ion:2 * n_ion] = np.mod(x[:n_ion] + rng.normal(0.0, 0.05, (n_ion, 3)), SIDE)
        out[f] = x
    return out, names


def _gamma(centres, names, share=0.25):
    species, order = KB.species_of(names)
    counts = KB.cube_counts(centres, species, len(order), SIDE, share, origins=60, stride=1)
    return KB.block_error(counts, (share * SIDE) ** 3)


def test_independent_ions_give_a_factor_of_one():
    centres, names = _box()
    res = _gamma(centres, names)
    assert res["gamma"] == pytest.approx(1.0, abs=0.12)


def test_ions_in_pairs_give_a_factor_of_one_half():
    centres, names = _box(pairs=True)
    res = _gamma(centres, names)
    assert res["gamma"] == pytest.approx(0.5, abs=0.08)


def test_the_species_are_the_cation_the_anion_and_the_sorted_solvents():
    species, order = KB.species_of(["EMC", "LI", "ANI", "EC", "EC"])
    assert order == ["LI", "ANI", "EC", "EMC"]
    assert species.tolist() == [3, 0, 1, 2, 2]


def test_a_cube_counts_every_molecule_once_and_wraps_around_the_box():
    centres = np.array([[[1.0, 1.0, 1.0], [49.0, 1.0, 1.0], [25.0, 25.0, 25.0]]])
    species = np.array([0, 0, 1])
    counts = KB.cube_counts(centres, species, 2, SIDE, 0.5, origins=500, stride=1)
    assert counts.shape == (1, 500, 2)
    assert counts.sum(axis=2).max() <= 3
    # the mean count in a cube that is an eighth of the box is an eighth of the molecules
    assert counts[0, :, 0].mean() == pytest.approx(2 / 8, abs=0.06)


def test_the_loader_wraps_the_unwrapped_centres(tmp_path):
    path = str(tmp_path / "centres.npz")
    c = np.array([[[-3.0, 120.0, 55.0]]])
    np.savez(path, centres=c, names=np.array(["LI"]), side=SIDE)
    centres, names, side = KB.load_box(path)
    assert side == SIDE and names == ["LI"]
    assert centres[0, 0] == pytest.approx([47.0, 20.0, 5.0])
