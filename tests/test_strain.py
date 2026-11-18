"""The strain a point mutation leaves (plan item PP4).

* The wild type relaxed at a position is relaxed as a mutant there is: only the residue and
  the side chains around it move, and the energy falls.
* The split energy adds up: the residue's own terms, the terms of the other atoms and the
  interaction between them give the total, and the total is the energy the force field gives
  the whole structure.
* A mutant and the wild type at the same position have the same surroundings, so a structure
  compared with itself gives zero in every energy column.
* The logarithmic scale of an energy is odd, monotone and linear near zero.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
sys.path.insert(0, ROOT)

pytest.importorskip("rdkit")

import run_strain as R                                                  # noqa: E402
from cmfqsar import mutants as MU                                       # noqa: E402

PDB = os.path.join(ROOT, "examples", "mutants", "1UFM", "1UFM.pdb")


def _openmm():
    pytest.importorskip("openmm")
    pytest.importorskip("pdbfixer")


@pytest.fixture(scope="module")
def relaxed(tmp_path_factory):
    _openmm()
    tmp = tmp_path_factory.mktemp("strain")
    wt_pdb = str(tmp / "wt.pdb")
    MU.prepare_wild_type(PDB, wt_pdb)
    return wt_pdb, MU.relax_wild_type(wt_pdb, 20, steps=300)


def test_the_wild_type_is_relaxed_around_one_position(relaxed):
    wt_pdb, st = relaxed
    assert st.energy_after < st.energy_before
    before = MU.protein_from_pdb(open(wt_pdb).read()).GetConformer().GetPositions()
    mol = MU.protein_of(st)
    after = mol.GetConformer().GetPositions()
    assert before.shape == after.shape
    moved = np.linalg.norm(after - before, axis=1) > 1e-3
    numbers = np.array([a.GetPDBResidueInfo().GetResidueNumber() for a in mol.GetAtoms()])
    names = np.array([a.GetPDBResidueInfo().GetName().strip() for a in mol.GetAtoms()])
    # no backbone atom moves except those of the residue itself
    off = moved & (numbers != 20) & np.isin(names, sorted(MU.BACKBONE))
    assert not off.any()
    assert moved[numbers == 20].any()
    # residues far from the site do not move at all
    far = np.linalg.norm(before - before[numbers == 20].mean(axis=0), axis=1) > 20.0
    assert not moved[far].any()


def test_the_split_energy_adds_up(relaxed):
    _wt, st = relaxed
    t = MU.energy_terms(st, 20)
    parts = sum(t[p][k] for p in ("site", "surroundings", "interaction") for k in MU.ENERGY_KINDS)
    assert parts == pytest.approx(t["total"], abs=1e-6)
    assert t["total"] == pytest.approx(st.energy_after, abs=0.5)
    # the bonds that join the residue to the chain are in the interaction, and at a minimum
    # they are small
    assert abs(t["interaction"]["bond"]) < 5.0
    assert t["residue"] in MU.AA1


def test_a_structure_against_itself_gives_zero():
    t = {"site": {k: 1.0 + i for i, k in enumerate(MU.ENERGY_KINDS)},
         "surroundings": {k: -50.0 for k in MU.ENERGY_KINDS},
         "interaction": {k: -3.0 * i for i, k in enumerate(MU.ENERGY_KINDS)},
         "total": 0.0, "residue": "LEU"}
    wt = {("X", 5): t}
    mut = {("X", "L5L"): t}
    ref = R.residue_references(wt, mut)
    assert ref == {"LEU": pytest.approx(sum(t["site"].values()))}

    class S:
        domain, position, label = ["X"], [5], ["L5L"]

    rows, total = R.energy_rows(S, wt, mut, ref)
    assert rows.shape == (1, len(R.ENERGY_COLUMNS))
    assert np.allclose(rows, 0.0) and np.allclose(total, 0.0)


def test_the_residue_reference_removes_what_only_names_the_residue():
    def terms(residue, site):
        return {"site": {k: site / len(MU.ENERGY_KINDS) for k in MU.ENERGY_KINDS},
                "surroundings": {k: 0.0 for k in MU.ENERGY_KINDS},
                "interaction": {k: 0.0 for k in MU.ENERGY_KINDS},
                "total": site, "residue": residue}
    wt = {("X", 1): terms("ALA", 10.0), ("X", 2): terms("ALA", 12.0)}
    mut = {("X", "A1W"): terms("TRP", 100.0), ("X", "A2W"): terms("TRP", 104.0)}
    ref = R.residue_references(wt, mut)

    class S:
        domain, position, label = ["X", "X"], [1, 2], ["A1W", "A2W"]

    rows, total = R.energy_rows(S, wt, mut, ref)
    # the total difference is mostly the change of residue; the residue column is not
    assert total[:, 0] == pytest.approx([90.0, 92.0])
    assert rows[:, 4] == pytest.approx([-2.0 - (-1.0), 2.0 - 1.0])


def test_the_log_scale():
    x = np.array([-1000.0, -4.184, -1e-3, 0.0, 1e-3, 4.184, 1000.0])
    y = R.signed_log(x)
    assert np.allclose(y, -y[::-1])
    assert np.all(np.diff(y) > 0)
    assert y[4] == pytest.approx(1e-3 / R.LOG_SCALE, rel=1e-3)
    assert y[5] == pytest.approx(np.log(2.0))
