"""Tests for the topomer construction: a frame that comes from the topology."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem, RDLogger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.topomer import (SITE_SPACING, TopomerSpec, attachment_label,  # noqa: E402
                             canonical, common_core, cuttable, cuttable_fraction,
                             largest_substituent, series_labels, substituents,
                             topomer_fields, topomer_molecule, topomer_pose,
                             unique_substituents)

RDLogger.DisableLog("rdApp.*")
CORE = Chem.MolFromSmiles("c1ccccc1C(=O)N")
SERIES = ["CCCOc1ccccc1C(=O)NCC", "CCOc1ccccc1C(=O)NCC", "Clc1ccccc1C(=O)NC",
          "CC(C)Oc1ccccc1C(=O)NCC"]


def test_the_same_substituent_always_lands_in_the_same_place():
    """Cut from different molecules, written in different atom orders: one pose."""
    a = Chem.MolFromSmiles("CCCOc1ccccc1C(=O)NCC")
    b = Chem.MolFromSmiles("CCCOc1ccc(Cl)cc1C(=O)NCC")        # same R-group, other molecule
    fa, fb = largest_substituent(a, CORE), largest_substituent(b, CORE)
    assert Chem.MolToSmiles(fa) == Chem.MolToSmiles(fb)
    xa = np.array(topomer_pose(fa).GetConformer().GetPositions())
    xb = np.array(topomer_pose(fb).GetConformer().GetPositions())
    assert xa.shape == xb.shape
    assert np.array_equal(xa, xb)                              # not merely close: identical

    # and a fragment whose atoms are numbered differently poses the same way too
    order = [int(i) for i in np.random.default_rng(0).permutation(fa.GetNumAtoms())]
    shuffled = Chem.RenumberAtoms(fa, order)
    xs = np.array(topomer_pose(shuffled).GetConformer().GetPositions())
    assert np.allclose(np.sort(xs, axis=0), np.sort(xa, axis=0), atol=1e-9)


def test_the_pose_convention_is_what_it_says():
    frag = largest_substituent(Chem.MolFromSmiles(SERIES[0]), CORE)
    posed = topomer_pose(frag)
    xyz = np.array(posed.GetConformer().GetPositions())
    dummy = [a.GetIdx() for a in posed.GetAtoms() if a.GetAtomicNum() == 0][0]
    root = [n.GetIdx() for n in posed.GetAtomWithIdx(dummy).GetNeighbors()][0]
    assert np.allclose(xyz[root], 0.0, atol=1e-9)              # attachment atom at the origin
    v = xyz[dummy] / np.linalg.norm(xyz[dummy])
    assert np.allclose(v, [-1.0, 0.0, 0.0], atol=1e-6)         # open valence along -x
    assert xyz[:, 0].max() > 0.5                               # the body extends the other way


def test_each_attachment_point_keeps_its_own_region_of_space():
    mols = [Chem.MolFromSmiles(s) for s in SERIES]
    labels = series_labels(mols, CORE)
    assert len(labels) >= 2
    built = topomer_molecule(mols[0], CORE, labels)
    xyz = np.array(built.GetConformer().GetPositions())
    sites = sorted({round(float(x) / SITE_SPACING) for x in xyz[:, 0]})
    assert len(sites) >= 2                                     # two substituents, two sites
    assert (max(sites) - min(sites)) * SITE_SPACING >= SITE_SPACING - 1e-9


def test_the_kernel_is_the_sum_over_attachment_points():
    """Which is what topomer CoMFA gets by giving every position its own columns."""
    fs = FieldSet.create(alpha=0.5)
    mols = [Chem.MolFromSmiles(s) for s in SERIES[:2]]
    labels = series_labels(mols, CORE)
    whole = [build_fields(topomer_molecule(m, CORE, labels), fs, f"m{i}")
             for i, m in enumerate(mols)]
    k_all = kernel_matrices(AtomTable.from_molecules(whole), AtomTable.from_molecules(whole),
                            fs.specs)[:, 0, 1]
    # the same number, assembled from one attachment point at a time
    total = np.zeros_like(k_all)
    for label in labels:
        per = []
        for i, m in enumerate(mols):
            built = topomer_molecule(m, CORE, [label])
            per.append(build_fields(built, fs, f"m{i}") if built is not None else None)
        if any(p is None for p in per):
            continue
        tab = AtomTable.from_molecules(per)
        total += kernel_matrices(tab, tab, fs.specs)[:, 0, 1]
    assert np.allclose(k_all, total, atol=1e-9)


def test_a_core_that_cannot_be_cut_off_is_reported_not_ignored():
    fused = Chem.MolFromSmiles("O=C1Nc2ccccc2C(c2ccccc2)=NC1")      # a benzodiazepine
    plain = Chem.MolFromSmiles("CCOc1ccccc1C(=O)NCC")
    benzene = Chem.MolFromSmiles("c1ccccc1")
    assert cuttable(plain, benzene)
    assert not cuttable(fused, benzene)                        # removing it would open a ring
    assert cuttable_fraction([plain, fused], benzene) == pytest.approx(0.5)
    fs = FieldSet.create(alpha=0.5)
    _, log = topomer_fields([fused, fused], ["a", "b"], fs, core=benzene)
    assert any("comes off cleanly" in line for line in log)


def test_fields_are_built_for_a_series_and_identical_r_groups_agree():
    fs = FieldSet.create(alpha=0.5)
    mols = [Chem.MolFromSmiles(s) for s in SERIES]
    mf, log = topomer_fields(mols, [f"m{i}" for i in range(len(mols))], fs, core=CORE)
    assert all(f is not None for f in mf) and not log
    assert len(unique_substituents(mols, CORE)) >= 3
    # two molecules sharing every substituent have the same fields, atom for atom
    twin = [Chem.MolFromSmiles(SERIES[0]), Chem.MolFromSmiles(SERIES[0])]
    a, b = topomer_fields(twin, ["x", "y"], fs, core=CORE)[0]
    assert np.array_equal(a.coords, b.coords) and np.array_equal(a.weights, b.weights)


def test_the_helpers_do_what_their_names_say():
    mol = Chem.MolFromSmiles(SERIES[0])
    subs = substituents(mol, CORE)
    assert len(subs) >= 2 and all(any(a.GetAtomicNum() == 0 for a in f.GetAtoms()) for f in subs)
    assert {attachment_label(f) for f in subs} <= set(series_labels([mol], CORE))
    assert Chem.MolToSmiles(canonical(subs[0])) == Chem.MolToSmiles(subs[0])
    assert common_core([Chem.MolFromSmiles(s) for s in SERIES]) is not None
