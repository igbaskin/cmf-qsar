"""Plan item NT4: item M1 measured on a set of nanotubes (benchmarks/run_nanotubes.py).

The tight-binding gaps of the set, the chain kernel on tubes (one cell against three, a mirror
image, a box mistaken for a crystal), and the handedness that item M10 cannot read from a
structure of one element.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

ase = pytest.importorskip("ase")
pytest.importorskip("scipy")
R = pytest.importorskip("run_nanotubes")

from cmfqsar.periodic import PeriodicSpec, crystal_table  # noqa: E402
from cmfqsar.threebody import ThreeBodySpec, periodic_chirality  # noqa: E402

SPEC = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5, taper=2.0)


def test_an_armchair_tube_is_metallic_and_the_others_follow_n_minus_m():
    assert R.tight_binding_gap(R.tube(5, 5)) < 1e-6           # no curvature gap by symmetry
    zigzag_metal = R.tight_binding_gap(R.tube(9, 0))
    assert 0.0 < zigzag_metal < R.METAL_GAP                    # a small gap from curvature
    for n, m in ((10, 0), (8, 0), (6, 4), (7, 0)):
        assert (n - m) % 3 != 0
        assert R.tight_binding_gap(R.tube(n, m)) > 0.5


def test_the_gap_of_a_semiconducting_tube_falls_as_one_over_its_diameter():
    small, large = R.tight_binding_gap(R.tube(7, 0)), R.tight_binding_gap(R.tube(16, 0))
    ratio = R.diameter(16, 0) / R.diameter(7, 0)
    assert 0.7 * ratio < small / large < 1.3 * ratio


def test_the_tube_set_is_as_described():
    rows = R.tube_list()
    assert all(R.diameter(n, m) >= R.MIN_DIAMETER and a <= R.MAX_ATOMS for n, m, a in rows)
    assert (10, 10) in [(n, m) for n, m, _ in rows]
    assert abs(R.chiral_angle(6, 6) - 30.0) < 1e-9 and R.chiral_angle(9, 0) == 0.0


def test_one_cell_of_a_tube_gives_the_channels_of_three():
    F = R._rows([R.tube(6, 3), R.tube(6, 3, length=3)], SPEC)
    assert R._rel(F[0], F[1]) < 1e-12


def test_a_tube_and_its_mirror_image_are_the_same_to_the_pair_kernel():
    t = R.tube(6, 3)
    F = R._rows([t, R.mirror(t)], SPEC)
    assert R._rel(F[0], F[1]) < 1e-12


def test_item_m10_has_no_handedness_block_for_one_element():
    # recorded so that a change to item M10 is noticed: the block needs three different fields
    recs = crystal_table([R.tube(6, 3)], R._fields(), ["t"]).molecules
    assert len(periodic_chirality(recs[0], ThreeBodySpec(chirality=True))) == 0


def test_the_handedness_across_shells_changes_sign_in_a_mirror():
    spec = ThreeBodySpec(chirality=True)
    t = R.tube(6, 3)
    recs = crystal_table([t, R.mirror(t)], R._fields(), ["a", "b"]).molecules
    a, b = (R.shell_handedness(r, spec) for r in recs)
    assert np.max(np.abs(a)) > 1e-4 and np.allclose(a, -b, atol=1e-14)
    achiral = crystal_table([R.tube(8, 0)], R._fields(), ["z"]).molecules[0]
    assert np.max(np.abs(R.shell_handedness(achiral, spec))) < 1e-12


def test_a_tube_in_a_box_is_not_a_chain():
    t = R.tube(8, 0)
    boxed = t.copy()
    boxed.cell[0, 0] = boxed.cell[1, 1] = 20.0
    boxed.center(axis=(0, 1))
    chain = boxed.copy()
    boxed.pbc = (True, True, True)
    chain.pbc = (False, False, True)
    F = R._rows([chain, boxed], SPEC)
    assert R._rel(F[0], F[1]) > 0.5
    plain = R._rows([t], SPEC)[0]
    assert R._rel(plain, F[0]) < 1e-12        # a box across a chain changes nothing
