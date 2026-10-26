"""A macrocyclic host read as a pocket (plan item SC3).

`complement` was written for proteins: a structure with residues, and profiles per residue. A
cyclodextrin, a cucurbituril or a calixarene comes as a molecule file with no residues at all,
and what takes their place is the host's own repeating units. These tests fix the split -- the
small ring systems are the units, the macrocycle that joins them is not one, and every atom
outside a ring goes to the unit it hangs on -- and the frame that says where the cavity is.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.protein import (cavity_frame, host_from_molecule, monomer_units,  # noqa: E402
                             read_molecule_host, read_structure)

#: a small cyclodextrin-like macrocycle is slow to embed; alpha-cyclodextrin itself is the real
#: case and is built once for the whole module
ALPHA_CD = ("OC[C@H]1O[C@@H]2O[C@H]3[C@@H](O)[C@H](O)[C@@H](O[C@@H]4O[C@H](CO)[C@@H](O[C@@H]5O"
            "[C@H](CO)[C@@H](O[C@@H]6O[C@H](CO)[C@@H](O[C@@H]7O[C@H](CO)[C@@H](O[C@@H]1[C@H](O)"
            "[C@@H]2O)[C@H](O)[C@H]7O)[C@H](O)[C@H]6O)[C@H](O)[C@H]5O)[C@H](O)[C@H]4O)O[C@@H]3CO")


def _embed(smiles, seed=1, macrocycle=True):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.useMacrocycleTorsions = macrocycle
    p.randomSeed = seed
    if AllChem.EmbedMolecule(m, p) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(m, maxIters=1000)
    except Exception:                            # noqa: BLE001
        pass
    return m


@pytest.fixture(scope="module")
def alpha():
    m = _embed(ALPHA_CD, seed=3)
    if m is None:
        pytest.skip("this RDKit could not embed the macrocycle")
    return m


# --------------------------------------------------------------------------- the split


def test_a_cyclodextrin_splits_into_its_sugars(alpha):
    unit = monomer_units(alpha)
    assert len(unit) == alpha.GetNumAtoms()
    units = sorted(set(unit))
    assert len(units) == 6                       # alpha-cyclodextrin is six glucoses
    heavy = [sum(1 for i, a in enumerate(alpha.GetAtoms())
                 if unit[i] == u and a.GetAtomicNum() > 1) for u in units]
    assert heavy == [11] * 6                     # and each one is C6O5


def test_the_macrocycle_itself_is_not_a_unit(alpha):
    """Without the size limit every sugar would be fused into one unit through the big ring."""
    assert len(set(monomer_units(alpha, max_ring=9))) == 6
    assert len(set(monomer_units(alpha, max_ring=100))) == 1


def test_an_atom_outside_every_ring_joins_the_unit_it_hangs_on(alpha):
    unit = monomer_units(alpha)
    for a in alpha.GetAtoms():
        if a.IsInRing():
            continue
        nbrs = [n.GetIdx() for n in a.GetNeighbors()]
        assert unit[a.GetIdx()] in {unit[j] for j in nbrs}


def test_a_molecule_with_no_rings_is_one_unit():
    m = _embed("CCCCCO", seed=1, macrocycle=False)
    assert len(set(monomer_units(m))) == 1


def test_two_separate_rings_are_two_units():
    m = _embed("c1ccccc1.C1CCCCC1", seed=1, macrocycle=False)
    if m is None:
        pytest.skip("could not embed")
    assert len(set(monomer_units(m))) == 2


# ------------------------------------------------------------------------- the host object


def test_the_host_has_one_residue_per_unit(alpha):
    host = host_from_molecule(alpha, name="alpha-cyclodextrin")
    assert host.n_atoms == alpha.GetNumAtoms()
    assert host.n_residues == 6
    assert sorted(set(int(r) for r in host.resids)) == [1, 2, 3, 4, 5, 6]
    assert host.residue_label(0).startswith("MON")
    assert len(host.bonds) == alpha.GetNumBonds()
    assert not host.is_water().any()


def test_a_host_without_a_conformation_is_refused():
    flat = Chem.AddHs(Chem.MolFromSmiles("c1ccccc1"))
    with pytest.raises(ValueError, match="needs a conformation"):
        host_from_molecule(flat)


def test_the_host_can_be_read_from_a_file(alpha, tmp_path):
    path = os.path.join(str(tmp_path), "host.sdf")
    with Chem.SDWriter(path) as w:
        w.write(alpha)
    host = read_molecule_host(path)
    assert host.n_residues == 6 and host.n_atoms == alpha.GetNumAtoms()
    assert host.name == "host"
    # and through the reader that chooses by extension, as the pocket tools call it
    assert read_structure(path).n_residues == 6


def test_an_unreadable_extension_says_what_it_takes(tmp_path):
    path = os.path.join(str(tmp_path), "host.xyz")
    open(path, "w").write("1\n\nC 0 0 0\n")
    with pytest.raises(ValueError, match=r"\.sdf"):
        read_molecule_host(path)


# ------------------------------------------------------------------------- where the cavity is


def test_the_cavity_frame_is_the_middle_of_the_ring_of_units(alpha):
    host = host_from_molecule(alpha)
    centre, axis = cavity_frame(host)
    keys = sorted(set(int(r) for r in host.resids))
    mids = np.array([host.coords[host.resids == k].mean(axis=0) for k in keys])
    assert centre == pytest.approx(mids.mean(axis=0))
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    # the units stand around the centre, not along the axis
    rel = mids - centre
    along = np.abs(rel @ axis)
    radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
    assert radial.mean() > 2.0 * along.mean()


def test_the_cavity_is_a_hole(alpha):
    """There has to be room at the centre, or it is not a pocket."""
    host = host_from_molecule(alpha)
    centre, _axis = cavity_frame(host)
    heavy = np.array([e != "H" for e in host.elements])
    d = np.linalg.norm(host.coords[heavy] - centre, axis=1)
    assert d.min() > 2.0


def test_a_host_with_two_units_still_gives_a_frame():
    m = _embed("c1ccccc1.C1CCCCC1", seed=1, macrocycle=False)
    if m is None:
        pytest.skip("could not embed")
    centre, axis = cavity_frame(host_from_molecule(m))
    assert centre.shape == (3,) and np.linalg.norm(axis) == pytest.approx(1.0)


# ----------------------------------------------------------- it works as a pocket downstream


def test_the_host_makes_pocket_fields(alpha):
    from cmfqsar.complementarity import pocket_fields
    from cmfqsar.fields import FieldSet
    host = host_from_molecule(alpha)
    fs = FieldSet.create(alpha=0.5)
    pocket = pocket_fields(host, fs)
    assert pocket.n_atoms == host.n_atoms
    assert pocket.weights.shape == (host.n_atoms, len(fs.specs))
    assert np.isfinite(pocket.coords).all()


def test_a_guest_in_the_cavity_scores_against_the_host(alpha):
    from cmfqsar.complementarity import complementarity, pocket_fields
    from cmfqsar.fields import FieldSet, build_fields
    host = host_from_molecule(alpha)
    fs = FieldSet.create(alpha=0.5)
    pocket = pocket_fields(host, fs)
    centre, _axis = cavity_frame(host)
    guest = _embed("CCCCC(=O)O", seed=2, macrocycle=False)
    conf = guest.GetConformer()
    pos = np.array(conf.GetPositions())
    pos = pos - pos.mean(axis=0) + centre
    for i in range(guest.GetNumAtoms()):
        conf.SetAtomPosition(i, [float(v) for v in pos[i]])
    mf = build_fields(guest, fs, "guest")
    got = complementarity(mf, pocket, fs)
    assert np.isfinite(got.score)
    assert got.terms and all(np.isfinite(v) for v in got.terms.values())
