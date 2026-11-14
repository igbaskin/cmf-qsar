"""Series in which both enantiomers are measured: the builder and the measurement (item MC3).

No test needs the download.  The records are made up, and the molecules are embedded here.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, os.path.join(ROOT, "examples"))

from cmfqsar.fields import FieldSet, build_fields  # noqa: E402
from cmfqsar.stereo import StereoSpec, centre_channels, stereo_channels  # noqa: E402

import make_enantiomer_examples as B  # noqa: E402
import run_chirality as C  # noqa: E402
import run_enantiomers as R  # noqa: E402

FIELDS = ["electrostatic", "steric", "hydrophobic", "hba", "hbd"]
#: lactic acid, its mirror image, one of its diastereomers' parents, and propranolol
LACTIC = ("C[C@H](O)C(=O)O", "C[C@@H](O)C(=O)O")
PROPRANOLOL = ("CC(C)NC[C@H](O)COc1cccc2ccccc12", "CC(C)NC[C@@H](O)COc1cccc2ccccc12")


# --------------------------------------------------------------------------- #
# Is this a pair of mirror images?
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("a,b,centres", [
    (LACTIC[0], LACTIC[1], 1),                                  # a pair
    (PROPRANOLOL[0], PROPRANOLOL[1], 1),                        # the pair the literature names
    ("C[C@H](O)[C@H](N)C(=O)O", "C[C@@H](O)[C@@H](N)C(=O)O", 2),  # both centres inverted
    (LACTIC[0], LACTIC[0], None),                               # the same molecule twice
    ("CC(O)C(=O)O", "CC(O)C(=O)O", None),                       # no centre at all
    (LACTIC[0], "CC(O)C(=O)O", None),                           # a configuration not given
    ("C[C@H](O)[C@H](N)C(=O)O", "C[C@@H](O)[C@H](N)C(=O)O", None),  # diastereomers
    (LACTIC[0], "C[C@H](N)C(=O)O", None),                       # different molecules
])
def test_only_mirror_images_are_a_pair(a, b, centres):
    got = B.enantiomers(a, b)
    assert (got["centres"] if got else None) == centres


def test_the_salt_and_the_free_base_are_the_same_molecule():
    assert B.enantiomers(LACTIC[0] + ".Cl", LACTIC[1]) is not None
    assert Chem.MolToSmiles(B.parse(LACTIC[0] + ".Cl")) == Chem.MolToSmiles(B.parse(LACTIC[0]))
    assert B.parse("[Na+].[Cl-]") is not None                   # two parts, neither organic
    assert B.parse("") is None and B.parse("not a molecule") is None


def _record(mol, assay, value, type_="Ki", smiles=None, year=2001, assay_id=None):
    return {"molecule_chembl_id": mol, "canonical_smiles": smiles, "standard_relation": "=",
            "standard_units": "nM", "pchembl_value": str(value), "standard_type": type_,
            "assay_chembl_id": assay_id or assay, "assay_description": "a made-up assay",
            "document_chembl_id": "CHEMBL_D" + str(year), "document_year": year,
            "data_validity_comment": None}


def test_a_pair_needs_both_potencies_from_one_assay():
    rows = [_record("M1", "A1", 8.0, smiles=LACTIC[0]),
            _record("M2", "A1", 6.0, smiles=LACTIC[1]),
            # the same pair, but the two values come from two assays: not a pair
            _record("M3", "A2", 9.0, smiles=PROPRANOLOL[0]),
            _record("M4", "A3", 5.0, smiles=PROPRANOLOL[1])]
    found = B.pairs_of_target(rows)
    assert len(found) == 1
    # which member is "first" follows the structures and not the identifiers, so the two
    # potencies are checked as a set
    assert {found[0]["first_p"], found[0]["second_p"]} == {8.0, 6.0}
    assert found[0]["first_smiles"] < found[0]["second_smiles"]
    assert found[0]["assay"] == "A1" and found[0]["centres"] == 1


def test_a_record_that_cannot_be_used_is_left_out():
    assert not B.usable(_record("M", "A", 8.0) | {"standard_relation": ">"})
    assert not B.usable(_record("M", "A", 8.0) | {"data_validity_comment": "outside typical"})
    assert not B.usable(_record("M", "A", 8.0) | {"standard_units": "ug.mL-1"})
    assert not B.usable(_record("M", "A", 8.0) | {"pchembl_value": None})
    assert B.usable(_record("M", "A", 8.0))


def test_a_download_that_names_no_assay_is_refused():
    good = [_record("M1", "A1", 8.0), _record("M2", None, 6.0)]
    good[1]["assay_chembl_id"] = None
    assert B.complete(good)                                      # one record names its assay
    assert B.complete([])
    assert not B.complete([{"assay_chembl_id": None}, {"molecule_chembl_id": "M"}])


def test_the_first_member_is_decided_by_the_structure_and_not_by_the_identifier():
    low, high = sorted([Chem.MolToSmiles(B.parse(s)) for s in LACTIC])
    for ids in (("M1", "M2"), ("M2", "M1")):
        rows = [_record(ids[0], "A1", 8.0, smiles=LACTIC[0]),
                _record(ids[1], "A1", 6.0, smiles=LACTIC[1])]
        found = B.pairs_of_target(rows)
        assert found[0]["first_smiles"] == low and found[0]["second_smiles"] == high


def test_one_assay_for_each_pair_the_one_with_the_most_records():
    rows = [_record("M1", "A1", 8.0, smiles=LACTIC[0]), _record("M2", "A1", 6.0,
                                                                smiles=LACTIC[1]),
            _record("M1", "A2", 7.0, smiles=LACTIC[0]), _record("M2", "A2", 5.0,
                                                                smiles=LACTIC[1]),
            _record("M2", "A2", 5.2, smiles=LACTIC[1])]
    found = B.pairs_of_target(rows)
    assert len(found) == 1 and found[0]["assay"] == "A2" and found[0]["records"] == 3


def test_the_two_structures_are_mirror_images_of_one_conformation():
    rows = [_record("M1", "A1", 8.0, smiles=PROPRANOLOL[0]),
            _record("M2", "A1", 6.0, smiles=PROPRANOLOL[1])]
    p = B.pairs_of_target(rows)[0]
    built, why = B.build_pair(p)
    assert why is None
    first, second = built
    a = np.array(first.GetConformer().GetPositions())
    b = np.array(second.GetConformer().GetPositions())
    assert np.allclose(a * np.array([1.0, 1.0, -1.0]), b)        # a reflection, atom for atom
    assert B.cip_codes(first) != B.cip_codes(second)             # and the configuration flips
    assert [c for _i, c in B.cip_codes(first)] in (["R"], ["S"])


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def _fields(smiles_list, seed=5):
    fs = FieldSet.create(physchem=FIELDS, alpha=0.5, charge_model="gasteiger")
    mols, mf = [], []
    for k, smi in enumerate(smiles_list):
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        p = AllChem.ETKDGv3()
        p.randomSeed = seed + k
        assert AllChem.EmbedMolecule(m, p) == 0
        Chem.SanitizeMol(m)
        mols.append(m)
        mf.append(build_fields(m, fs, f"m{k}"))
    return mols, mf, len(fs.specs)


def test_the_kernel_with_a_block_from_outside_is_the_one_item_c7_measures():
    smiles = [LACTIC[0], LACTIC[1], PROPRANOLOL[0], PROPRANOLOL[1]]
    mols, mf, n_fields = _fields(smiles)
    S = np.vstack([stereo_channels(m, StereoSpec(), n_fields=n_fields) for m in mf])
    mine = R.kernels_with(mf, n_fields, S)
    theirs = C._stereo_kernels(mf, n_fields)
    assert len(mine) == len(theirs)
    for a, b in zip(mine, theirs):
        assert np.allclose(a, b)


def _reflected(mol):
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    for i, p in enumerate(np.array(conf.GetPositions()) * np.array([1.0, 1.0, -1.0])):
        conf.SetAtomPosition(i, [float(v) for v in p])
    Chem.AssignStereochemistryFrom3D(m)
    return m


def test_the_channels_split_into_the_stereocentres_and_the_rest():
    # the second molecule is the reflection of the first, as the builder makes it: two separate
    # embeddings of the two enantiomers differ in their torsions as well, and then the channels
    # of the two are not exact negatives
    mols, mf, n_fields = _fields([PROPRANOLOL[0], LACTIC[0]])
    fs = FieldSet.create(physchem=FIELDS, alpha=0.5, charge_model="gasteiger")
    mirror = _reflected(mols[0])
    mols = [mols[0], mirror, mols[1]]
    mf = [mf[0], build_fields(mirror, fs, "mirror"), mf[1]]
    centres = [R.stereocentres(m) for m in mols]
    assert all(len(c) == 1 for c in centres)                     # one stereocentre each
    whole = R.stereo_rows(mf, mols, centres, "every centre", n_fields)
    cip = R.stereo_rows(mf, mols, centres, "stereocentres only", n_fields)
    rest = R.stereo_rows(mf, mols, centres, "every other centre", n_fields)
    assert np.allclose(whole, cip + rest)                        # the parts add up
    assert np.any(np.abs(cip) > 1e-9)
    # the channels of a molecule and of its mirror image are exact negatives, part by part
    assert np.allclose(cip[0], -cip[1]) and np.allclose(rest[0], -rest[1])
    assert np.allclose(whole, np.vstack([stereo_channels(m, StereoSpec(), n_fields=n_fields)
                                         for m in mf]))


def test_the_two_point_rows_of_a_pair_are_identical():
    mols, mf, n_fields = _fields([PROPRANOLOL[0]])
    mirror = _reflected(mols[0])
    fs = FieldSet.create(physchem=FIELDS, alpha=0.5, charge_model="gasteiger")
    both = [mf[0], build_fields(mirror, fs, "mirror")]
    assert R.blindness(both, n_fields, [(0, 1)]) == 0.0


def test_the_difference_inside_a_pair_and_how_it_is_scored():
    y = np.array([8.0, 6.0, 5.0, 5.0, 7.0, 9.0])
    pairs = [(0, 1), (2, 3), (4, 5)]
    half = R.centred(y, pairs)
    assert np.allclose(half, [1.0, -1.0, 0.0, 0.0, -1.0, 1.0])
    # a model that ranks every pair the right way round, except the one that cannot be ranked
    pred = np.array([1.0, -1.0, 0.0, 0.0, -0.5, 0.5])
    r = R.ranking(pred, y, pairs)
    assert r["pairs with two equal potencies, set aside"] == 1
    assert r["margin 0.0"]["pairs"] == 2 and r["margin 0.0"]["share"] == 1.0
    assert r["margin 1.0"]["pairs"] == 2
    # a model blind to handedness gives the two members of a pair the same number: half a point
    blind = np.array([7.0, 7.0, 5.0, 5.0, 8.0, 8.0])
    rb = R.ranking(blind, y, pairs)
    assert rb["margin 0.0"]["share"] == 0.5
    # and one that is always wrong
    rw = R.ranking(-pred, y, pairs)
    assert rw["margin 0.0"]["share"] == 0.0
    assert rw["margin 0.5"]["pairs"] == 2


def test_the_held_out_pairs_of_every_target_are_pooled():
    y = np.array([8.0, 6.0, 5.0, 7.0])
    pairs = [(0, 1), (2, 3)]
    pred = np.array([1.0, -1.0, np.nan, np.nan])             # the second pair was not predicted
    assert R.differences(pred, y, pairs) == [[2.0, 2.0]]
    rows = [{"variants": {"p": {"g": {"differences": [[1.0, 2.0], [-1.0, -2.0]]}}}},
            {"variants": {"p": {"g": {"differences": [[1.0, -2.0], [0.0, 1.0]]}}}}]
    got = R.pooled_ranking(rows, "p", "g")
    # two named right, one named wrong, one not named at all: (2 + 0.5) / 4
    assert got["margin 0.0"]["pairs"] == 4
    assert got["margin 0.0"]["share"] == pytest.approx(0.625)
    assert got["margin 1.0"]["pairs"] == 4 and R.pooled_ranking([], "p", "g") == {}


def test_the_interval_of_a_share():
    assert R.wilson(5, 10)[0] < 0.5 < R.wilson(5, 10)[1]
    assert R.wilson(0, 10) == [0.0, pytest.approx(0.278, abs=0.01)]
    assert R.wilson(10, 10)[1] == pytest.approx(1.0)
    assert R.wilson(0, 0)[0] != R.wilson(0, 0)[0]                 # not a number, with no pair
    wide, narrow = R.wilson(30, 60), R.wilson(300, 600)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_a_pair_is_never_split_between_training_and_test():
    groups = [f"p{i // 2}" for i in range(40)]
    folds = R.group_folds(groups, folds=5, seed=0)
    assert len(folds) == 5
    for tr, te in folds:
        assert not (set(groups[i] for i in tr) & set(groups[i] for i in te))
        assert len(tr) + len(te) == 40
    assert sorted(i for _tr, te in folds for i in te) == list(range(40))


#: a real pair of the PI3-kinase set whose stereocentre is a ring atom, so it survives into the
#: Bemis-Murcko scaffold.  Read with the configuration left in, the two members have different
#: scaffolds and would be put in different groups.
RING_CENTRE = ("CS(=O)(=O)Cc1cc([C@@]23CCOC[C@@H]2C3)nc(-c2cccc(O)c2)n1",
               "CS(=O)(=O)Cc1cc([C@]23CCOC[C@H]2C3)nc(-c2cccc(O)c2)n1")


def test_a_pair_whose_centre_is_in_the_ring_is_not_split_by_the_scaffold():
    from cmfqsar.chemotypes import scaffold_smiles
    mols = [Chem.MolFromSmiles(s) for s in RING_CENTRE]
    # the scaffolds with the configuration left in differ, which is the trap
    assert scaffold_smiles(mols[0]) != scaffold_smiles(mols[1])
    # the grouping takes the configuration off first, so both members land in one group
    groups = R.scaffold_groups(mols)
    assert groups[0] == groups[1]
    assert R.check_groups(groups, [(0, 1)]) == 1
    with pytest.raises(SystemExit):
        R.check_groups(["c0", "c1"], [(0, 1)])


def test_the_same_pair_measured_against_two_targets_stays_two_groups():
    d = {"group": ["t1:P", "t1:P", "t2:P", "t2:P"], "member": ["first", "second"] * 2}
    assert R.pair_index(d) == [(0, 1), (2, 3)]
    one = {"group": ["t:P", "t:P"], "member": ["first", "first"]}
    assert R.pair_index(one) == []                                # a member missing: no pair


# --------------------------------------------------------------------------- #
# The geometry of a built structure
# --------------------------------------------------------------------------- #
def test_an_sf5_group_is_built_as_a_regular_octahedron():
    # MMFF94 has no parameters for the group, so the files held the raw embedding of four pairs
    # until 14 November 2026, with two fluorines 1.0 A apart
    from cmfqsar.orca import geometry_faults

    mol = B.embed("O=C(O)C1=Cc2cc(S(F)(F)(F)(F)F)ccc2O[C@@H]1C(F)(F)F")
    assert mol is not None and geometry_faults(mol) == []
    assert "SF5" in mol.GetProp("geometry_note")
    r = np.array(mol.GetConformer().GetPositions())
    s = [a for a in mol.GetAtoms() if a.GetSymbol() == "S"][0]
    f = [n.GetIdx() for n in s.GetNeighbors() if n.GetSymbol() == "F"]
    c = [n.GetIdx() for n in s.GetNeighbors() if n.GetSymbol() == "C"]
    assert len(f) == 5 and len(c) == 1
    assert np.allclose(np.linalg.norm(r[f] - r[s.GetIdx()], axis=1), 1.59, atol=1e-3)
    assert np.linalg.norm(r[c[0]] - r[s.GetIdx()]) == pytest.approx(1.80, abs=1e-3)
    between = [np.linalg.norm(r[i] - r[j]) for k, i in enumerate(f) for j in f[k + 1:]]
    assert min(between) == pytest.approx(1.59 * np.sqrt(2.0), abs=1e-3)     # 90 degrees
    # the rest of the molecule is refined: the CF3 group on the ring has its usual bond length
    cf3 = mol.GetSubstructMatch(Chem.MolFromSmarts("[CX4](F)(F)F"))
    assert np.linalg.norm(r[cf3[0]] - r[cf3[1]]) == pytest.approx(1.34, abs=0.03)
    # and the configuration of the record is kept
    assert [code for _i, code in B.cip_codes(mol)] == ["S"]


def test_an_embedding_that_fails_the_geometry_check_is_made_again():
    # with the seed of the builder this molecule comes out with the two carbons of its secondary
    # amine on one line with the nitrogen, and MMFF94 does not leave that point
    from cmfqsar.orca import geometry_faults

    smiles = "Cc1nc2cccc(-c3cc4c([nH]3)CCNC4=O)c2nc1N[C@@H](C)C(N)=O"
    raw = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = int(B.SEED) & 0x7FFFFFFF
    assert AllChem.EmbedMolecule(raw, params) == 0
    AllChem.MMFFOptimizeMolecule(raw, maxIters=500)
    assert geometry_faults(raw) != []
    mol = B.embed(smiles)
    assert mol is not None and geometry_faults(mol) == []
    assert mol.GetProp("geometry_note").startswith("embedding seed + 1")
    # a molecule that passes at once carries no note
    plain = B.embed(PROPRANOLOL[0])
    assert not plain.HasProp("geometry_note")
