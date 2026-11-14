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


SF5_PHENYL = "[c]1ccc(S(F)(F)(F)(F)F)cc1"


def _faults(mol):
    """Bonds outside 0.7 to 1.3 times the sum of the covalent radii, and the closest pair of
    atoms without a bond when it is below 1.3 A."""
    from cmfqsar import orca

    out = list(orca.changed_bonds(mol, low=0.7, high=1.3))
    r = np.array(mol.GetConformer().GetPositions(), float)
    d = np.linalg.norm(r[:, None, :] - r[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    for b in mol.GetBonds():
        d[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = d[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = np.inf
    if d.min() < 1.3:
        out.append("contact %.2f" % d.min())
    return out


def test_an_sf5_group_is_built_on_the_conformers_of_a_stand_in():
    # no force field of RDKit holds the group: UFF moved its fluorines 6 to 8 A from the sulfur
    cat = D.assemble(Chem.MolFromSmiles(SCAFFOLD_I), Chem.MolFromSmiles(SF5_PHENYL))
    stand_in, kept = D.sf5_stand_in(Chem.AddHs(cat))
    assert stand_in.GetNumAtoms() == Chem.AddHs(cat).GetNumAtoms() - 4 == len(kept)
    assert not any(a.GetSymbol() == "S" for a in stand_in.GetAtoms())
    mh, energy, tried = D.embed_lowest(cat, n=4, seed=1)
    assert tried == 4 and np.isfinite(energy)
    assert "stand-in" in mh.GetProp("force_field")
    assert 45.0 < D.biaryl_dihedral(mh) < 65.0
    r = np.array(mh.GetConformer().GetPositions(), float)
    sulfurs = [a for a in mh.GetAtoms() if a.GetSymbol() == "S"]
    assert len(sulfurs) == 2
    for s in sulfurs:
        for n in s.GetNeighbors():
            length = float(np.linalg.norm(r[s.GetIdx()] - r[n.GetIdx()]))
            assert length == pytest.approx(1.59 if n.GetSymbol() == "F" else 1.80, abs=1e-6)
    assert _faults(mh) == []


def test_a_catalyst_without_the_group_has_no_stand_in():
    cat = Chem.AddHs(D.assemble(Chem.MolFromSmiles(SCAFFOLD_I), Chem.MolFromSmiles(PHENYL)))
    assert D.sf5_stand_in(cat) == (None, [])


@pytest.mark.parametrize("name", ["catalysts.sdf", "catalysts_sym.sdf", "imines.sdf", "thiols.sdf"])
def test_no_record_of_the_built_files_has_an_impossible_bond(name):
    # the check that would have found the catalyst with the SF5 groups: every bond length
    # between 0.7 and 1.3 times the sum of the covalent radii, and no two atoms without a bond
    # closer than 1.3 A
    path = os.path.join(D.OUT_DIR, name)
    if not os.path.isfile(path):
        pytest.skip("the built files are not present")
    faults = [(mol.GetProp("_Name"), f) for mol in Chem.SDMolSupplier(path, removeHs=False)
              for f in _faults(mol)]
    assert faults == []
