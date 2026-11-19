"""The cavity frame of plan item SC5.

* The donors are the O, N and S atoms of the macrocycle, not those of a side arm.
* The frame is a property of the host alone: moving the input does not move the result.
* For a crown ether z is the ring normal; for a cryptand it runs through the bridgeheads.
* A molecule without a macrocycle is refused, and the align command logs it.
* With xtb present, a complex is built whose ion sits among the donors.
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
    assert d.min() > 1.7 and d.max() < 3.0
    assert r["host"].GetNumAtoms() == m.GetNumAtoms() - 1
