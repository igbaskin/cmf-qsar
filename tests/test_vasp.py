"""Per-atom arrays from a VASP run (plan item M3).

What these tests fix: that a net Bader charge is valence electrons minus Bader electrons and
nothing else, that the species order of the OUTCAR is respected, that a magnetisation block is
found and a run without one gives no array, and that a mismatch in atom counts is refused rather
than silently misassigned.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("ase")

from cmfqsar.vasp import (describe_arrays, net_bader_charges, read_acf,  # noqa: E402
                          read_outcar_energy, read_outcar_magmoms, read_outcar_species,
                          structure_with_arrays, write_structure)

POSCAR = """Li2 O1
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Li O
2 1
Direct
0.0 0.0 0.0
0.5 0.5 0.0
0.25 0.25 0.5
"""

ACF = """    #         X           Y           Z       CHARGE      MIN DIST   ATOMIC VOL
 --------------------------------------------------------------------------------
    1    0.000000    0.000000    0.000000    2.150000     0.60         3.1
    2    2.000000    2.000000    0.000000    2.150000     0.60         3.1
    3    1.000000    1.000000    2.000000    7.700000     0.90        15.8
 --------------------------------------------------------------------------------
    VACUUM CHARGE:               0.0000
    VACUUM VOLUME:               0.0000
    NUMBER OF ELECTRONS:        12.0000
"""

OUTCAR = """ vasp.6.4.1
   VRHFIN =Li: 1s2s2p
   POMASS =    7.010; ZVAL   =    3.000    mass and valenz
   VRHFIN =O: s2p4
   POMASS =   16.000; ZVAL   =    6.000    mass and valenz
   ions per type =               2   1
   ZVAL   =   3.00  6.00
  FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)
  energy  without entropy=      -20.123456  energy(sigma->0) =      -20.123456
 magnetization (x)

# of ion       s       p       d       tot
------------------------------------------
    1        0.010   0.000   0.000   0.010
    2        0.010   0.000   0.000   0.010
    3        0.100   0.300   0.000   0.400
--------------------------------------------------
tot          0.120   0.300   0.000   0.420
  energy  without entropy=      -20.654321  energy(sigma->0) =      -20.654321
"""


@pytest.fixture
def run(tmp_path):
    (tmp_path / "CONTCAR").write_text(POSCAR)
    (tmp_path / "ACF.dat").write_text(ACF)
    (tmp_path / "OUTCAR").write_text(OUTCAR)
    return str(tmp_path)


def test_the_bader_column_and_the_species_are_read(run):
    e = read_acf(os.path.join(run, "ACF.dat"))
    assert e.tolist() == [2.15, 2.15, 7.7]
    sp = read_outcar_species(os.path.join(run, "OUTCAR"))
    assert sp["symbols"] == ["Li", "O"] and sp["counts"] == [2, 1] and sp["zval"] == [3.0, 6.0]


def test_a_net_charge_is_valence_minus_bader_electrons():
    q = net_bader_charges([2.15, 2.15, 7.7], ["Li", "Li", "O"], {"Li": 3.0, "O": 6.0})
    assert np.allclose(q, [0.85, 0.85, -1.7])
    assert abs(q.sum()) < 1e-9
    with pytest.raises(ValueError, match="ZVAL"):
        net_bader_charges([1.0], ["Na"], {"Li": 3.0})


def test_the_structure_carries_the_arrays(run):
    atoms = structure_with_arrays(run)
    assert len(atoms) == 3 and atoms.get_chemical_symbols() == ["Li", "Li", "O"]
    assert np.allclose(atoms.arrays["bader_charges"], [0.85, 0.85, -1.7])
    assert np.allclose(atoms.arrays["bader_electrons"], [2.15, 2.15, 7.7])
    assert np.allclose(atoms.arrays["magnetic_moment"], [0.01, 0.01, 0.4])
    assert atoms.info["vasp_energy"] == pytest.approx(-20.654321)
    assert "bader_charges" in describe_arrays(atoms)
    assert np.allclose(atoms.get_initial_charges(), [0.85, 0.85, -1.7])


def test_the_magnetisation_block_is_the_last_one_and_may_be_absent(run):
    mm = read_outcar_magmoms(os.path.join(run, "OUTCAR"))
    assert mm.tolist() == [0.01, 0.01, 0.4]
    assert read_outcar_energy(os.path.join(run, "OUTCAR")) == pytest.approx(-20.654321)
    plain = os.path.join(run, "plain")
    os.makedirs(plain)
    with open(os.path.join(plain, "OUTCAR"), "w") as fh:
        fh.write(OUTCAR.split(" magnetization (x)")[0])
    assert read_outcar_magmoms(os.path.join(plain, "OUTCAR")) is None


def test_a_count_mismatch_is_refused(run):
    with open(os.path.join(run, "ACF.dat"), "w") as fh:
        fh.write("\n".join(ACF.splitlines()[:4]) + "\n")
    with pytest.raises(ValueError, match="2 atoms"):
        structure_with_arrays(run)


def test_without_bader_the_run_is_refused_unless_asked(run):
    os.remove(os.path.join(run, "ACF.dat"))
    with pytest.raises(FileNotFoundError, match="ACF.dat"):
        structure_with_arrays(run)
    atoms = structure_with_arrays(run, require_bader=False)
    assert "bader_charges" not in atoms.arrays and "magnetic_moment" in atoms.arrays


def test_the_arrays_survive_a_round_trip_through_extended_xyz(run, tmp_path):
    from ase.io import read
    atoms = structure_with_arrays(run)
    path = str(tmp_path / "with_arrays.extxyz")
    write_structure(atoms, path)
    back = read(path)
    # ASE reads a "charges" column back as the initial charges, which the periodic charge
    # model also accepts
    assert np.allclose(back.arrays["bader_charges"], atoms.arrays["bader_charges"])
    from cmfqsar.periodic import crystal_charges
    assert np.allclose(crystal_charges(back, "file")[0], atoms.arrays["bader_charges"])
    assert np.allclose(back.arrays["magnetic_moment"], atoms.arrays["magnetic_moment"])


def test_the_periodic_fields_take_the_net_charges(run):
    from cmfqsar.periodic import crystal_charges, crystal_fields, inorganic_fieldset
    atoms = structure_with_arrays(run)
    q, src = crystal_charges(atoms, "file")
    assert src == "bader_charges" and np.allclose(q, [0.85, 0.85, -1.7])
    fs = inorganic_fieldset(["Li", "O"], charge_model="file")
    c = crystal_fields(atoms, fs)
    assert np.allclose(c.weights[:, 0], [0.85, 0.85, -1.7])
