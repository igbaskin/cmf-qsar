"""The solvation-complex builder, its frame and its energy bookkeeping (plan item BT4).

Nothing here calls xTB: the tests cover the placing, the geometry and the arithmetic, which is
what can be checked without a quantum program. The benchmark checks the energies against a
published relaxation.
"""
import math
import os
import sys

import numpy as np
import pytest
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import solvation as S  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields  # noqa: E402

RDLogger.DisableLog("rdApp.*")

EC = "C1COC(=O)O1"          # ethylene carbonate: one carbonyl and two ether oxygens
DME = "COCCOC"              # dimethoxyethane: two ether oxygens
TETRAGLYME = "COCCOCCOCCOCCOC"


def _mol(smiles, seed=3):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    return m


# --------------------------------------------------------------------------------- the donors


def test_the_donors_of_a_carbonate_are_its_three_oxygens():
    got = S.donor_atoms(_mol(EC))
    kinds = sorted(kind for _i, kind in got)
    assert len(got) == 3
    assert kinds == ["carbonyl", "ether", "ether"]


def test_a_carbonyl_oxygen_is_preferred_to_an_ether_oxygen():
    """Which donor a cation takes, when no partial charges say otherwise."""
    m = _mol(EC)
    idx = S.best_donor(m)
    kinds = dict(S.donor_atoms(m))
    assert kinds[idx] == "carbonyl"


def test_partial_charges_decide_the_donor_when_they_are_given():
    m = _mol(EC)
    donors = [i for i, _k in S.donor_atoms(m)]
    charges = np.zeros(m.GetNumAtoms())
    charges[donors[-1]] = -1.0                      # make the last donor the most negative
    assert S.best_donor(m, charges) == donors[-1]


def test_the_lone_pair_of_an_ether_oxygen_points_away_from_both_bonds():
    m = _mol(DME)
    pos = m.GetConformer().GetPositions()
    idx = [i for i, k in S.donor_atoms(m) if k == "ether"][0]
    u = S.donor_direction(m, idx)
    assert u.shape == (3,)
    assert np.isclose(np.linalg.norm(u), 1.0)
    for nb in m.GetAtomWithIdx(idx).GetNeighbors():
        to_neighbour = pos[nb.GetIdx()] - pos[idx]
        to_neighbour /= np.linalg.norm(to_neighbour)
        assert float(np.dot(u, to_neighbour)) < 0.0, "the lone pair points away from every bond"


def test_a_molecule_with_no_donor_is_refused_by_name():
    with pytest.raises(S.SolvationError, match="no donor atom"):
        S.best_donor(_mol("CCCC"))


# -------------------------------------------------------------------------------- the placing


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_every_solvent_reaches_the_ion_at_the_coordination_distance(n):
    c = S.place_shell([Chem.MolFromSmiles(EC)] * n, ion="Li")
    assert c.n_solvents == n
    assert c.charge == 1
    idx, d = c.coordination()
    assert len(idx) >= n, "each molecule contributes at least its own donor"
    assert np.allclose(d[:n], S.ION_OXYGEN_DISTANCE["Li"], atol=1e-6)


def test_the_placed_molecules_do_not_sit_on_top_of_each_other():
    c = S.place_shell([Chem.MolFromSmiles(EC)] * 4, ion="Li")
    pos = c.mol.GetConformer().GetPositions()
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    assert d.min() > 0.9, f"closest pair of atoms {d.min():.2f} A apart"


def test_the_ion_is_the_first_atom_and_carries_the_charge():
    c = S.place_shell([Chem.MolFromSmiles(DME)], ion="Na")
    assert c.ion_index == 0
    atom = c.mol.GetAtomWithIdx(0)
    assert atom.GetSymbol() == "Na" and atom.GetFormalCharge() == 1
    assert Chem.GetFormalCharge(c.mol) == 1


def test_a_different_ion_is_placed_at_its_own_distance():
    for ion in ("Li", "Na", "K"):
        c = S.place_shell([Chem.MolFromSmiles(EC)], ion=ion)
        _idx, d = c.coordination()
        assert np.isclose(d[0], S.ION_OXYGEN_DISTANCE[ion], atol=1e-6)


def test_an_anion_joins_the_shell_and_changes_the_total_charge():
    c = S.place_shell([Chem.MolFromSmiles(EC)] * 2, ion="Li",
                      anion=Chem.MolFromSmiles("[F-]"))
    assert c.n_solvents == 3, "the anion is one of the three placed molecules"
    assert c.charge == 0, "a contact ion pair of Li+ and F- is neutral"


def test_a_shell_this_module_cannot_place_is_refused_with_the_numbers_it_can():
    with pytest.raises(S.SolvationError, match="1, 2, 3, 4, 5, 6"):
        S.place_shell([Chem.MolFromSmiles(EC)] * 7, ion="Li")


def test_an_unknown_ion_is_refused_by_name():
    with pytest.raises(S.SolvationError, match="unknown ion"):
        S.place_shell([Chem.MolFromSmiles(EC)], ion="Cs")


def test_wrapping_puts_the_ion_among_the_donors_of_one_molecule():
    c = S.wrap_ion(Chem.MolFromSmiles(TETRAGLYME), ion="Li", n_donors=4)
    assert c.n_solvents == 1
    assert len(c.donors) == 4
    pos = c.mol.GetConformer().GetPositions()
    # the ion is inside the chosen donors: no further from their centroid than they are
    centre = pos[list(c.donors)].mean(axis=0)
    spread = float(np.mean(np.linalg.norm(pos[list(c.donors)] - centre, axis=1)))
    assert float(np.linalg.norm(pos[c.ion_index] - centre)) <= spread + 1e-6


# ---------------------------------------------------------------------------------- the frame


def test_the_frame_puts_the_ion_at_the_origin():
    c = S.place_shell([Chem.MolFromSmiles(EC)] * 3, ion="Li")
    coords = S.cation_frame(c.mol, c.ion_index, "Li")
    assert np.allclose(coords[c.ion_index], 0.0, atol=1e-9)


def _turned(mol, seed=4):
    """The same complex moved and turned at random."""
    rng = np.random.RandomState(seed)
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    out = Chem.Mol(mol)
    conf = out.GetConformer()
    pos = np.asarray(conf.GetPositions(), float) @ R.T + np.array([3.0, -2.0, 1.5])
    for i, p in enumerate(pos):
        conf.SetAtomPosition(i, Chem.rdGeometry.Point3D(*[float(v) for v in p]))
    return out


def test_the_frame_does_not_depend_on_how_the_complex_was_turned():
    """The point of a frame: two copies of one complex in different poses give one geometry.

    The shell holds three different molecules, so the polyhedron has no symmetry. A complex of
    three identical molecules placed at 120 degrees does have one, and then no rule can pick
    between the frames its symmetry relates; the test below covers that case instead.
    """
    c = S.place_shell([Chem.MolFromSmiles(EC), Chem.MolFromSmiles(DME),
                       Chem.MolFromSmiles("C1CCOC1")], ion="Li", seed=1)
    a = S.cation_frame(c.mol, 0, "Li")
    b = S.cation_frame(_turned(c.mol), 0, "Li")
    assert np.allclose(a, b, atol=1e-5), f"largest difference {np.abs(a - b).max():.2e}"


def test_a_symmetric_complex_gives_the_same_fields_in_either_frame():
    """Three identical molecules at 120 degrees: the frames its symmetry relates are equivalent,
    so what must agree is the kernel and not the coordinates."""
    from cmfqsar.fields import AtomTable
    from cmfqsar.kernels import kernel_matrices
    c = S.place_shell([Chem.MolFromSmiles(EC)] * 3, ion="Li", seed=1)
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5,
                         charge_model="gasteiger")
    one = build_fields(S.with_frame(c.mol, 0, "Li"), fs, "a")
    two = build_fields(S.with_frame(_turned(c.mol), 0, "Li"), fs, "b")
    table = AtomTable.from_molecules([one, two])
    for K in kernel_matrices(table, table, fs.specs):
        norm = K / np.sqrt(np.outer(np.diag(K), np.diag(K)))
        assert norm[0, 1] > 0.98, f"the two frames give different fields: {norm[0, 1]:.3f}"


def test_the_frame_is_a_rotation_and_keeps_every_distance():
    c = S.place_shell([Chem.MolFromSmiles(DME)] * 2, ion="Li")
    pos = np.asarray(c.mol.GetConformer().GetPositions(), float)
    coords = S.cation_frame(c.mol, 0, "Li")
    before = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    after = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    assert np.allclose(before, after, atol=1e-8)


def test_the_frame_quality_reports_a_rotation_it_cannot_fix():
    """Four donors on a regular tetrahedron leave every orientation equally good."""
    c = S.place_shell([Chem.MolFromSmiles(EC)] * 4, ion="Li")
    q = S.frame_quality(c.mol, 0, "Li")
    assert q["donors"] >= 4
    assert abs(q["moment_ratio_xy"] - 1.0) < 0.01, "the placed polyhedron is symmetric"


def test_a_complex_with_no_donor_in_reach_cannot_be_framed():
    c = S.place_shell([Chem.MolFromSmiles(EC)], ion="Li", distance=6.0)
    with pytest.raises(S.SolvationError, match="polyhedron"):
        S.cation_frame(c.mol, 0, "Li")


def test_with_frame_returns_a_molecule_the_field_builder_accepts():
    c = S.place_shell([Chem.MolFromSmiles(EC)] * 2, ion="Li")
    framed = S.with_frame(c.mol, 0, "Li")
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=0.5,
                         charge_model="gasteiger")
    mf = build_fields(framed, fs, "complex")
    assert np.isfinite(mf.weights).all()
    assert np.isclose(mf.weights[0, 0], 1.0, atol=0.2), "the ion carries the positive charge"
    assert np.allclose(mf.coords[0], 0.0, atol=1e-6), "and sits at the origin of the frame"


# ----------------------------------------------------------------------------- the energetics


def test_a_binding_energy_is_the_complex_minus_its_parts():
    assert S.binding_energy(-10.0, -1.0, [-7.0, -3.0]) == pytest.approx(1.0)
    assert S.binding_energy(-12.0, -1.0, [-7.0, -3.0]) == pytest.approx(-1.0)


def test_a_desolvation_energy_costs_energy_when_the_shell_is_bound():
    # a bound four-complex at -12, the three-complex at -9, the free solvent at -3: taking one
    # molecule out leaves the parts exactly where they were, so it costs nothing
    assert S.desolvation_energy(-12.0, -9.0, -3.0) == pytest.approx(0.0)
    # a more bound four-complex costs energy to open
    assert S.desolvation_energy(-13.0, -9.0, -3.0) == pytest.approx(1.0)


def test_the_hartree_conversion_is_the_one_the_module_documents():
    assert S.HARTREE_EV == pytest.approx(27.2113862, abs=1e-6)


def test_the_polyhedra_are_unit_vectors_that_spread_out():
    for n, dirs in S.POLYHEDRA.items():
        assert dirs.shape == (n, 3)
        assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)
        if n > 1:
            d = dirs @ dirs.T
            np.fill_diagonal(d, -1.0)
            assert d.max() < math.cos(math.radians(80.0)), \
                f"two of the {n} directions are closer than 80 degrees"
