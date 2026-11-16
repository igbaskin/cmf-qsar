"""The count references of the particle benchmark (stage counts of benchmarks/run_particles.py):
neighbour pairs per atom in the shells of the bulk lattice, the coordination histogram, and the
kernel with its own coefficients for each metal."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

RP = pytest.importorskip("run_particles")
ase = pytest.importorskip("ase")


def _cluster(symbol, positions):
    from ase import Atoms
    a = Atoms(symbol * len(positions), positions=positions)
    a.center(vacuum=8.0)
    a.pbc = False
    return a


def test_a_dimer_at_the_nearest_neighbour_distance_fills_the_first_shell():
    from ase.data import atomic_numbers, reference_states
    lattice = reference_states[atomic_numbers["Cu"]]["a"]
    nn = lattice / np.sqrt(2.0)
    dimer = _cluster("Cu", [(0, 0, 0), (nn, 0, 0)])
    far = _cluster("Cu", [(0, 0, 0), (lattice, 0, 0)])          # the second shell
    S = RP.shell_counts([dimer, far])
    assert S.shape == (2, RP.SHELLS)
    assert S[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert S[1].tolist() == [0.0, 1.0, 0.0, 0.0]
    H = RP.coordination_histogram([dimer, far])
    assert H[0, 1] == 1.0 and H[1, 0] == 1.0 and H.shape == (2, 13)


def test_the_bulk_shells_of_a_large_particle_approach_twelve_and_six():
    from ase.cluster import Octahedron
    p = Octahedron("Cu", 14, cutoff=4)
    p.center(vacuum=8.0)
    p.pbc = False
    s = RP.shell_counts([p])[0]
    assert 9.0 < s[0] < 12.0 and 4.0 < s[1] < 6.0                 # bulk: 12 and 6


def test_the_kernel_has_no_coupling_between_metals():
    el = np.array(["Cu", "Cu", "Ag"])
    X = np.array([[1.0, 2.0], [2.0, 0.0], [0.5, 0.5]])
    K = RP.per_metal_kernel(el, X)
    assert K[0, 2] == 0 and K[1, 2] == 0 and K[2, 2] > 0
    Z = X - X.mean(0)
    assert K[0, 1] == pytest.approx(1.0 + Z[0] @ Z[1])
