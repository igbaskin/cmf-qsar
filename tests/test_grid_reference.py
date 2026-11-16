"""The reference of the grid benchmark: the total energy of the same DFT run with the element
counts (stages energies and reference of benchmarks/run_gridfields.py)."""
import gzip
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

GF = pytest.importorskip("run_gridfields")
ase = pytest.importorskip("ase")

OUTCAR = """ some header
  energy without entropy =      -21.10000000  energy(sigma->0) =      -21.11000000
  free  energy   TOTEN  =       -21.12000000 eV
 more lines
  energy  without entropy=      -21.84554479  energy(sigma->0) =      -21.85228007
 the end
"""


def test_the_last_energy_of_an_outcar_is_read_plain_and_compressed(tmp_path):
    plain = tmp_path / "OUTCAR"
    plain.write_text(OUTCAR)
    packed = tmp_path / "OUTCAR.gz"
    with gzip.open(packed, "wt") as fh:
        fh.write(OUTCAR)
    assert GF.outcar_energy(str(plain)) == pytest.approx(-21.85228007)
    assert GF.outcar_energy(str(packed)) == pytest.approx(-21.85228007)
    assert GF.outcar_energy(str(tmp_path / "missing.gz")) is None


def test_the_columns_are_the_energy_and_the_element_counts(tmp_path, monkeypatch):
    from ase import Atoms
    cells = [Atoms("SrTiO3"), Atoms("BaTiO3"), Atoms("SrZrO3")]
    names = ["SrTiO3_1", "BaTiO3_2", "SrZrO3_3"]
    path = tmp_path / "run_energies.csv"
    path.write_text("cell,energy_eV\nSrTiO3_1,-40.0\nBaTiO3_2,-41.5\nSrZrO3_3,-42.25\n")
    monkeypatch.setattr(GF, "ENERGIES", str(path))
    E, C, elements = GF.run_energy_columns(names, cells)
    assert E.tolist() == [-40.0, -41.5, -42.25]
    assert elements == ["Ba", "O", "Sr", "Ti", "Zr"]
    assert C[0].tolist() == [0, 3, 1, 1, 0] and C[2].tolist() == [0, 3, 1, 0, 1]


def test_the_linear_kernel_is_a_regression_on_standardised_columns():
    rng = np.random.RandomState(0)
    X = np.c_[rng.randn(20) * 50 + 7, rng.randn(20), np.ones(20)]     # a constant column too
    K = GF.linear_kernel(X)
    Z = (X[:, :2] - X[:, :2].mean(0)) / X[:, :2].std(0)
    assert np.allclose(K, Z @ Z.T)
    assert np.allclose(K, K.T)
