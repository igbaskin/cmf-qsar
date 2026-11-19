"""The cavity frame of plan item SC5.

* The donors are the O, N and S atoms of the macrocycle, not those of a side arm.
* The frame is a property of the host alone: moving the input does not move the result.
* For a crown ether z is the ring normal; for a cryptand it runs through the bridgeheads.
* A molecule without a macrocycle is refused, and the align command logs it.
* An ion that is too large for a ring starts above and below it, at its contact distance.
* A contact is a donor within the sum of the two radii and a stated slack.
* A start that lies on an atom is moved off it, and the misfit of a perfect ring is zero.
* With xtb present, a complex is built whose ion touches every donor of the ring at the sum of
  the radii, from starts that differ from each other.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip("rdkit")
from rdkit import Chem                                                  # noqa: E402
from rdkit.Chem import AllChem                                          # noqa: E402
from rdkit.Geometry import Point3D                                      # noqa: E402

from cmfqsar import cavity as C                                         # noqa: E402

CROWN6 = "C1COCCOCCOCCOCCOCCO1"
BENZO15 = "c1ccc2c(c1)OCCOCCOCCOCCO2"
CRYPT222 = "C1COCCOCCN2CCOCCOCCN1CCOCCOCC2"
LARIAT = "COCCC1COCCOCCOCCOCCO1"            # 15-crown-5 with a methoxyethyl side arm


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(m, randomSeed=seed) == 0
    AllChem.MMFFOptimizeMolecule(m)
    return m


def _moved(m, seed=3):
    rng = np.random.RandomState(seed)
    q, _r = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    out = Chem.Mol(m)
    conf = out.GetConformer()
    xyz = np.asarray(conf.GetPositions()) @ q.T + rng.normal(size=3) * 5.0
    for i, p in enumerate(xyz):
        conf.SetAtomPosition(i, Point3D(*p))
    return out


def test_the_donors_are_the_atoms_of_the_macrocycle():
    assert len(C.donor_atoms(_mol(CROWN6))) == 6
    assert len(C.donor_atoms(_mol(BENZO15))) == 5
    assert len(C.donor_atoms(_mol(CRYPT222))) == 8             # six oxygens, two nitrogens
    lariat = _mol(LARIAT)
    donors = C.donor_atoms(lariat)
    assert len(donors) == 5
    arm = [a.GetIdx() for a in lariat.GetAtoms() if a.GetSymbol() == "O"
           and a.GetIdx() not in donors]
    assert len(arm) == 1


def test_the_frame_does_not_depend_on_where_the_host_was():
    m = _mol(BENZO15)
    a, b = C.cavity_frame(m), C.cavity_frame(_moved(m))
    assert np.allclose(a, b, atol=1e-6)
    d = C.donor_atoms(m)
    assert np.allclose(a[d].mean(axis=0), 0.0, atol=1e-9)


def test_z_is_the_ring_normal_of_a_crown_and_the_cage_axis_of_a_cryptand():
    crown = C.cavity_frame(_mol(CROWN6))
    d = C.donor_atoms(_mol(CROWN6))
    assert crown[d][:, 2].std() < 0.6 < np.hypot(crown[d][:, 0], crown[d][:, 1]).mean()
    m = _mol(CRYPT222)
    xyz = C.cavity_frame(m)
    n = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == "N"]
    axis = xyz[n[0]] - xyz[n[1]]
    assert abs(axis[2]) / np.linalg.norm(axis) > 0.9


def test_a_molecule_without_a_macrocycle_is_refused():
    m = _mol("CCOCCOCCOCC")
    with pytest.raises(C.CavityError):
        C.cavity_frame(m)
    log = C.cavity_align_dataset([m, _mol(CROWN6)])
    assert log[0].startswith("FAILED") and "6 donors" in log[1]


def _ring(radius, n=6):
    phi = 2 * np.pi * np.arange(n) / n
    return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), np.zeros(n)])


def test_an_ion_too_large_for_a_ring_starts_above_and_below_it():
    reach = C.ION_RADII["K"] + C.DONOR_RADIUS
    small = _ring(1.9, 4)
    starts = C.ion_starts(small, range(4), "K")
    assert len(starts) == 2
    for s in starts:
        assert np.allclose(np.linalg.norm(small - s, axis=1), reach, atol=1e-9)
    assert np.allclose(starts[0] + starts[1], 0.0, atol=1e-9)        # one on each side
    wide = _ring(3.0)
    assert len(C.ion_starts(wide, range(6), "K")) == 1
    assert np.allclose(C.ion_starts(wide, range(6), "K")[0], 0.0, atol=1e-9)
    # a ring that is only a little too small keeps the centre: the height would be under 0.5 A
    assert len(C.ion_starts(_ring(reach - 0.02), range(6), "K")) == 1


def test_a_start_on_an_atom_is_moved_off_it_and_a_perfect_ring_has_no_misfit():
    reach = C.ION_RADII["K"] + C.DONOR_RADIUS
    ring = _ring(reach)
    assert C.donor_misfit(ring, range(6), np.zeros(3), "K") == pytest.approx(0.0, abs=1e-12)
    assert C.donor_misfit(ring, range(6), np.zeros(3), "Na") == pytest.approx(1.38 - 1.02)
    blocked = np.vstack([ring, [[0.2, 0.0, 0.0]]])              # an atom at the centre
    start = C.ion_starts(blocked, range(6), "K")[0]
    assert np.linalg.norm(blocked - start, axis=1).min() >= 1.8
    assert np.linalg.norm(start) <= 3.0 * np.sqrt(3) + 1e-9
    # a free centre is left where it is
    assert np.allclose(C._clear_point(np.zeros(3), ring), 0.0)


def test_a_contact_is_a_donor_within_the_radii_and_the_slack():
    limit = C.contact_distance("Na")
    assert abs(limit - (1.02 + C.DONOR_RADIUS + C.CONTACT_SLACK)) < 1e-12
    ring = _ring(limit - 0.01)
    assert C.ion_contacts(ring, range(6), np.zeros(3), "Na") == 6
    assert C.ion_contacts(ring, range(6), np.array([0.5, 0.0, 0.0]), "Na") < 6
    assert C.contact_distance("K") - limit == pytest.approx(1.38 - 1.02)


def test_a_complex_holds_its_ion_among_the_donors():
    from cmfqsar.xtb import XtbError, find_xtb
    try:
        find_xtb()
    except XtbError:
        pytest.skip("xtb is not installed")
    r = C.host_complex("C1COCCOCCOCCO1", "Li", conformers=2, threads=1)
    m = r["complex"]
    xyz = m.GetConformer().GetPositions()
    donors = C.donor_atoms(r["host"])
    d = np.linalg.norm(xyz[donors] - xyz[r["ion_index"]], axis=1)
    assert m.GetAtomWithIdx(r["ion_index"]).GetSymbol() == "Li"
    # two conformers, and two starts for a conformer whose ring is too small for the ion
    assert r["kept"] >= 1 and r["starts"] in (2, 3, 4)
    assert not r["released"] and r["contacts"] == 4
    # every oxygen is held at the sum of the two radii
    assert np.allclose(d, C.ION_RADII["Li"] + C.DONOR_RADIUS, atol=0.1)
    assert r["misfit"] < 0.1
    # the two conformers differ: RDKit gives one conformer many times for a seed of zero
    from rdkit.Chem import AllChem
    h = Chem.AddHs(Chem.MolFromSmiles("C1COCCOCCOCCO1"))
    p = AllChem.ETKDGv3()
    p.randomSeed = 1
    ids = list(AllChem.EmbedMultipleConfs(h, 2, p))
    assert np.abs(h.GetConformer(ids[0]).GetPositions()
                  - h.GetConformer(ids[1]).GetPositions()).max() > 0.1
    assert r["host"].GetNumAtoms() == m.GetNumAtoms() - 1
