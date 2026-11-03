"""Plan item BT6: the difference between two cells of one ladder as a field in the cell.

The pairs here are built by hand, so the construction is tested without the network and without
a lattice that happens to be convenient.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmfqsar import celldiff                                               # noqa: E402
from cmfqsar.periodic import (INORGANIC_FIELDS, inorganic_fieldset,        # noqa: E402
                              periodic_autocorrelation, PeriodicSpec)

FIELDSET = inorganic_fieldset(["Li", "Co", "O"], physchem=INORGANIC_FIELDS,
                              charge_model="electronegativity")


def cell(n_li, a=4.2, jitter=0.0):
    """A cubic cell with four cobalt sites, eight oxygens and `n_li` of four lithium sites."""
    from ase import Atoms
    symbols, scaled = [], []
    for k, site in enumerate([(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]):
        symbols.append("Co")
        scaled.append(site)
    for site in [(0.25, 0.25, 0.25), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75), (0.25, 0.75, 0.75),
                 (0.25, 0.25, 0.75), (0.75, 0.75, 0.75), (0.75, 0.25, 0.25), (0.25, 0.75, 0.25)]:
        symbols.append("O")
        scaled.append(site)
    for site in [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5), (0.5, 0.5, 0.5)][:n_li]:
        symbols.append("Li")
        scaled.append(site)
    atoms = Atoms(symbols=symbols, scaled_positions=scaled, cell=[a, a, a], pbc=True)
    if jitter:
        rng = np.random.default_rng(0)
        atoms.set_positions(atoms.get_positions() + jitter * rng.standard_normal((len(atoms), 3)))
    return atoms


class Pair:
    """What the battery module hands over: two cells, a target and a group."""

    def __init__(self, charged, discharged, voltage=4.0, framework="CoO2", name="pair"):
        self.charged, self.discharged = charged, discharged
        self.voltage, self.framework, self.name = voltage, framework, name
        self.x_charged, self.x_discharged = 0.0, 1.0


# --------------------------------------------------------------------------- #
# what a pair must satisfy
# --------------------------------------------------------------------------- #
def test_two_cells_on_one_lattice_have_no_mismatch():
    assert celldiff.lattice_mismatch(cell(0), cell(2)) == pytest.approx(0.0)
    assert celldiff.lattice_mismatch(cell(0, a=4.2), cell(2, a=4.41)) == pytest.approx(0.05)


def test_the_lattice_test_does_not_care_how_the_vectors_are_ordered():
    from ase import Atoms
    a = Atoms("Li", cell=[3.0, 4.0, 5.0], pbc=True)
    b = Atoms("Li", cell=[5.0, 3.0, 4.0], pbc=True)
    assert celldiff.lattice_mismatch(a, b) == pytest.approx(0.0)


def test_a_pair_whose_cells_are_too_far_apart_is_refused():
    ok, why = celldiff.can_difference(cell(0, a=4.2), cell(2, a=5.0))
    assert not ok and "cell lengths differ" in why
    with pytest.raises(ValueError, match="cannot carry a difference field"):
        celldiff.difference_fields(cell(0, a=4.2), cell(2, a=5.0), FIELDSET)


def test_a_pair_with_different_host_atoms_is_refused():
    other = cell(2)
    other.symbols[0] = "Ni"
    ok, why = celldiff.can_difference(cell(0), other)
    assert not ok and "different host atoms" in why


def test_a_pair_that_loses_the_working_ion_is_refused():
    ok, why = celldiff.can_difference(cell(4), cell(2))
    assert not ok and "no more of the working ion" in why


def test_a_proper_pair_is_accepted():
    ok, why = celldiff.can_difference(cell(0), cell(4))
    assert ok and why == ""


# --------------------------------------------------------------------------- #
# the difference field
# --------------------------------------------------------------------------- #
def test_the_difference_holds_both_cells_with_one_sign_each():
    d = celldiff.difference_fields(cell(0), cell(2), FIELDSET, name="half")
    assert d.name == "half"
    assert len(d.symbols) == len(cell(0)) + len(cell(2))
    assert d.weights.shape == (len(d.symbols), FIELDSET.n_fields)
    first, second = d.weights[:len(cell(2))], d.weights[len(cell(2)):]
    names = [sp.name for sp in FIELDSET.specs]
    # a field that is a property of the atom alone cancels on the host, which is the point of the
    # construction: twelve of the fourteen atoms of the discharged cell are in the charged one too
    for name in ("steric", "cif:Co", "cif:O"):
        k = names.index(name)
        assert np.allclose(second[:12, k], -first[:12, k]), name
    # the electrostatic field does not cancel, and that is not a fault: the electronegativity
    # proxy is the deviation from the mean over the cell, and the mean moves when lithium enters
    k = names.index("electrostatic")
    assert not np.allclose(second[:12, k], -first[:12, k])


def test_the_host_cancels_and_the_inserted_ion_is_what_is_left():
    """The steric field of the difference sums to the volume of the inserted atoms alone."""
    d = celldiff.difference_fields(cell(0), cell(2), FIELDSET)
    steric = [k for k, sp in enumerate(FIELDSET.specs) if sp.name == "steric"][0]
    lithium = celldiff.difference_fields(cell(0), cell(2), FIELDSET)
    total = float(lithium.weights[:, steric].sum())
    from ase.data import atomic_numbers, covalent_radii
    expected = 2.0 * covalent_radii[atomic_numbers["Li"]] ** 3
    assert total == pytest.approx(expected, rel=1e-9)
    assert abs(float(d.weights[:, steric].sum()) - expected) < 1e-9


def test_a_pair_that_inserts_nothing_is_refused():
    one = cell(2)
    ok, why = celldiff.can_difference(one, one.copy())
    assert not ok and "no more of the working ion" in why


def test_the_difference_of_a_cell_with_itself_is_empty():
    """The identity the construction rests on: +W and -W on the same sites give no field.

    The pair gate refuses two identical cells, because they are not an insertion pair, so the
    signed stack is built here directly: what is tested is that the kernel sees nothing, which is
    why the host of a real pair cancels.
    """
    from cmfqsar.periodic import CrystalFields, crystal_fields
    one = crystal_fields(cell(2), FIELDSET, "one")
    empty = CrystalFields(name="nothing", coords=np.vstack([one.coords, one.coords]),
                          weights=np.vstack([one.weights, -one.weights]),
                          radii=np.concatenate([one.radii, one.radii]),
                          symbols=list(one.symbols) * 2, cell=one.cell, pbc=one.pbc,
                          atom_types=list(one.atom_types) * 2)
    assert np.allclose(empty.weights.sum(axis=0), 0.0)
    spec = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)
    assert np.allclose(periodic_autocorrelation(empty, spec), 0.0, atol=1e-9)


def test_the_change_of_the_metal_charge_survives_in_the_difference():
    """Item BT1 inside item BT6: the cobalt is Co(IV) empty and Co(III) full, and the difference
    carries that change, not only the lithium that went in."""
    fs = inorganic_fieldset(["Li", "Co", "O"], physchem=INORGANIC_FIELDS,
                            charge_model="oxidation")
    d = celldiff.difference_fields(cell(0), cell(4), fs)
    k = [j for j, sp in enumerate(fs.specs) if sp.name == "electrostatic"][0]
    cobalt = np.array([w for w, s in zip(d.weights[:, k], d.symbols) if s == "Co"], float)
    assert len(cobalt) == 8                                   # four in each half of the pair
    assert float(cobalt.sum()) == pytest.approx(4.0 * (3.0 - 4.0))


def test_the_charged_cell_is_moved_into_the_lattice_of_the_other():
    """The two cells of a real pair differ in size; the difference lives in one cell."""
    charged, discharged = cell(0, a=4.2), cell(2, a=4.3)
    d = celldiff.difference_fields(charged, discharged, FIELDSET)
    assert np.allclose(d.cell, discharged.get_cell()[:])
    # the cobalt that sat at the origin of both cells is at the origin of the difference twice
    at_origin = np.sum(np.all(np.abs(d.coords) < 1e-9, axis=1))
    assert at_origin == 2


def test_the_difference_is_a_structure_every_kernel_takes():
    d = celldiff.difference_fields(cell(0), cell(2), FIELDSET)
    spec = PeriodicSpec(r_max=6.0, step=0.5, sigma=0.5)
    row = periodic_autocorrelation(d, spec)
    assert row.shape[0] > 0 and np.all(np.isfinite(row))
    assert np.any(row != 0.0)


# --------------------------------------------------------------------------- #
# the sites the ions went into, and the set
# --------------------------------------------------------------------------- #
def test_the_inserted_sites_are_the_ones_with_no_partner():
    inserted = celldiff.inserted_positions(cell(1), cell(3))
    assert inserted.shape == (2, 3)
    first = cell(3).get_positions()[-2:]
    assert np.allclose(np.sort(inserted, axis=0), np.sort(first, axis=0))


def test_every_site_is_inserted_when_the_charged_cell_has_none():
    assert celldiff.inserted_positions(cell(0), cell(4)).shape == (4, 3)


def test_the_shell_of_the_inserted_ion_is_measured_in_the_cell():
    pair = Pair(cell(0), cell(4))
    kept, refused = celldiff.usable_pairs([pair], FIELDSET)
    assert len(kept) == 1 and not refused
    shell = celldiff.shell_of_the_inserted_ion(kept[0], anion="O", cutoff=3.0)
    assert shell["n_sites"] == 4
    assert shell["coordination"] > 0
    assert 1.5 < shell["distance"] < 3.0


def test_a_set_reports_what_it_had_to_refuse():
    good = Pair(cell(0), cell(2), framework="CoO2", name="good")
    bad = Pair(cell(0, a=4.2), cell(2, a=5.2), framework="CoO2", name="bad")
    kept, refused = celldiff.usable_pairs([good, bad], FIELDSET)
    assert [k.name for k in kept] == ["good"]
    assert len(refused) == 1 and refused[0]["name"] == "bad"
    assert "cell lengths differ" in refused[0]["reason"]
    assert kept[0].target == pytest.approx(4.0) and kept[0].group == "CoO2"
    assert kept[0].n_inserted == 2 and kept[0].discharged is not None


def test_the_tolerance_can_be_widened():
    pair = Pair(cell(0, a=4.2), cell(2, a=4.62))          # 10 per cent apart
    assert not celldiff.usable_pairs([pair], FIELDSET)[0]
    kept, _ = celldiff.usable_pairs([pair], FIELDSET, tol=0.15)
    assert len(kept) == 1
    assert kept[0].lattice_mismatch == pytest.approx(0.1)
