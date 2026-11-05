"""The Denmark catalyst series: assembly, frame and free-energy conversion (plan item PO1).

The builder (``examples/make_denmark_examples.py``) assembles each chiral phosphoric acid from
a scaffold and two copies of a substituent radical, embeds it, fixes one axial configuration,
and puts the phosphoric acid in a frame.  These tests pin the parts that do not need the
downloaded files: the attachment positions, the assembled graph, the sign of the biaryl
dihedral, the frame, the two C2 images and the published conversion of an enantiomeric excess
to a free-energy difference.  The built files are checked only when they are present.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
import make_denmark_examples as D  # noqa: E402

SCAFFOLD_I = "O=P1(O)Oc2ccc3ccccc3c2-c2c(ccc3ccccc23)O1"
SCAFFOLD_VI = "O=P1(O)Oc2ccc3c(c2-c2c(ccc4c2CCCC4)O1)CCCC3"
PHENYL = "[c]1ccccc1"
BENZYL = "[CH2]c1ccc(OC)cc1"
BROMO = "[Br]"


def test_position_3_is_next_to_the_phosphate_oxygen_and_has_a_hydrogen():
    for smiles in (SCAFFOLD_I, SCAFFOLD_VI):
        scaf = Chem.MolFromSmiles(smiles)
        sites = D.position_3_carbons(scaf)
        assert len(sites) == 2
        for i in sites:
            c3 = scaf.GetAtomWithIdx(i)
            assert c3.GetTotalNumHs() == 1
            c2 = [n for n in c3.GetNeighbors()
                  if any(m.GetSymbol() == "O" for m in n.GetNeighbors())]
            assert c2, "position 3 must neighbour the carbon that carries the phosphate oxygen"


@pytest.mark.parametrize("sub, expected", [
    (PHENYL, "O=P1(O)Oc2c(-c3ccccc3)cc3ccccc3c2-c2c(c(-c3ccccc3)cc3ccccc23)O1"),
    (BROMO, "O=P1(O)Oc2c(Br)cc3ccccc3c2-c2c(c(Br)cc3ccccc23)O1"),
])
def test_assembly_puts_the_same_substituent_at_3_and_3prime(sub, expected):
    cat = D.assemble(Chem.MolFromSmiles(SCAFFOLD_I), Chem.MolFromSmiles(sub))
    assert Chem.MolToSmiles(cat) == Chem.MolToSmiles(Chem.MolFromSmiles(expected))
    assert all(a.GetNumRadicalElectrons() == 0 for a in cat.GetAtoms())


def test_a_benzyl_radical_attaches_through_its_methylene():
    cat = D.assemble(Chem.MolFromSmiles(SCAFFOLD_VI), Chem.MolFromSmiles(BENZYL))
    patt = Chem.MolFromSmarts("c:c(-[CH2]-c1ccc(OC)cc1):c")
    assert len(cat.GetSubstructMatches(patt)) == 2


def test_embedding_gives_one_axial_configuration_and_the_frame():
    cat = D.assemble(Chem.MolFromSmiles(SCAFFOLD_I), Chem.MolFromSmiles(PHENYL))
    for seed in (1, 2, 3):
        # several conformers, so that more than one of them needs the reflection
        mh, energy, tried = D.embed_lowest(cat, n=8, seed=seed)
        assert tried == 8 and np.isfinite(energy)
        assert D.biaryl_dihedral(mh) > 0
        framed = D.place_in_frame(mh)
        pos = np.array(framed.GetConformer().GetPositions())
        ip, ia, ib, ioh = D.frame_atoms(framed)
        assert np.allclose(pos[ip], 0.0, atol=1e-9)
        # the ring oxygens straddle the yz plane, at nearly the same height below the phosphorus
        # (their P-O distances differ by what the minimisation left); the acid oxygens are above
        assert abs(pos[ia, 0] + pos[ib, 0]) < 1e-6 and abs(pos[ia, 2] - pos[ib, 2]) < 0.05
        assert pos[ia, 2] < 0 and pos[ib, 2] < 0
        assert pos[ioh, 2] > 0 and pos[ioh, 1] > 0
        acid = [a.GetIdx() for a in framed.GetAtoms() if a.GetSymbol() == "O"
                and not any(n.GetIsAromatic() for n in a.GetNeighbors())]
        assert all(pos[i, 2] > 0 for i in acid)
        # the dihedral survives the frame, which is a rotation
        assert D.biaryl_dihedral(framed) == pytest.approx(D.biaryl_dihedral(mh), abs=1e-6)


def test_the_two_c2_images_are_the_rotation_about_z():
    cat = D.assemble(Chem.MolFromSmiles(SCAFFOLD_I), Chem.MolFromSmiles(BROMO))
    mh, _, _ = D.embed_lowest(cat, n=2, seed=5)
    framed = D.place_in_frame(mh)
    a, b = D.c2_images(framed)
    pa = np.array(a.GetConformer().GetPositions())
    pb = np.array(b.GetConformer().GetPositions())
    assert np.allclose(pa * np.array([-1.0, -1.0, 1.0]), pb)
    assert a.GetProp("symmetry_image") != b.GetProp("symmetry_image")
    # the rotated image is a valid frame of the same molecule: the same P, the same dihedral
    assert D.biaryl_dihedral(b) == pytest.approx(D.biaryl_dihedral(a), abs=1e-6)


def test_the_published_free_energy_conversion():
    # catalyst 145_i on imine 1 with thiol A: ee 90 %, and the paper's table says 1.745 kcal/mol
    assert D.ddg_from_ee(90.0) == pytest.approx(1.745, abs=0.002)
    assert D.ddg_from_ee(0.0) == 0.0
    assert D.ddg_from_ee(-34.0) == pytest.approx(-D.ddg_from_ee(34.0))
    with pytest.raises(ValueError):
        D.ddg_from_ee(100.0)


@pytest.mark.skipif(not os.path.isfile(os.path.join(D.OUT_DIR, "reactions.csv")),
                    reason="the built files are not present")
def test_the_built_files_are_complete():
    with open(os.path.join(D.OUT_DIR, "reactions.csv"), encoding="utf-8") as fh:
        lines = fh.read().strip().splitlines()
    assert len(lines) == 1076
    cats = {m.GetProp("_Name") for m in Chem.SDMolSupplier(os.path.join(D.OUT_DIR, "catalysts.sdf"),
                                                           removeHs=False)}
    assert len(cats) == 43
    sym = [m for m in Chem.SDMolSupplier(os.path.join(D.OUT_DIR, "catalysts_sym.sdf"), removeHs=False)]
    assert len(sym) == 86
    dihedrals = [float(m.GetProp("biaryl_dihedral"))
                 for m in Chem.SDMolSupplier(os.path.join(D.OUT_DIR, "catalysts.sdf"), removeHs=False)]
    assert all(d > 0 for d in dihedrals)
