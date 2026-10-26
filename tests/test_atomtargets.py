"""A target that belongs to an atom rather than to a molecule (plan item LP1).

There is no model over atoms yet; what these tests fix is the part that has to be right before
one can exist. A per-atom target is unlike the per-atom properties a field is built from in one
way that decides everything downstream: **most atoms have no value**, and the ones that do are
not independent samples, because they share a molecule. So the reading has to keep the blanks,
the index has to remember which molecule each sample came from, the folds have to hold out whole
molecules, and a prediction has to go back onto the atom it belongs to.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.atomtargets import (AtomTargets, read_atom_target, scatter,  # noqa: E402
                                 write_atom_predictions)
from cmfqsar.dataio import load_dataset  # noqa: E402

SMILES = ["CC(=O)Oc1ccccc1C(=O)O", "c1ccccc1C(=O)O", "CCOc1ccccc1N", "CC(C)c1ccccc1O",
          "CCCCc1ccccc1"]


def _mol(smiles, name="m", seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    m.SetProp("_Name", name)
    return m


def _with_gaps(smiles, name, prop="shift", every=3, seed=1):
    """A molecule whose heavy atoms carry a value, with a gap at every `every`-th one."""
    m = _mol(smiles, name, seed)
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    m.SetProp(prop, ",".join("" if k % every == 0 else f"{120 + k:.1f}"
                             for k in range(len(heavy))))
    return m


# ------------------------------------------------------------------------- reading the blanks


def test_a_blank_is_not_a_zero():
    m = _with_gaps(SMILES[0], "aspirin")
    v = read_atom_target(m, "shift")
    assert len(v) == m.GetNumAtoms()
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    assert not np.isfinite(v[heavy[0]])            # the first heavy atom was left blank
    assert v[heavy[1]] == pytest.approx(121.0)
    assert not np.isfinite(v[[a.GetIdx() for a in m.GetAtoms()
                              if a.GetAtomicNum() == 1]]).any()   # hydrogens are unmeasured


def test_every_spelling_of_a_blank_is_one():
    m = _mol("CCO", "e")
    heavy = [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() > 1]
    m.SetProp("t", "1.0, , n/a")
    v = read_atom_target(m, "t")
    assert v[heavy[0]] == pytest.approx(1.0)
    assert not np.isfinite(v[heavy[1]]) and not np.isfinite(v[heavy[2]])


def test_a_value_on_the_atoms_themselves_is_read_and_a_missing_one_stays_missing():
    m = _mol("CCO", "e")
    for i in (0, 2):
        m.GetAtomWithIdx(i).SetDoubleProp("t", 10.0 + i)
    v = read_atom_target(m, "t")
    assert v[0] == pytest.approx(10.0) and v[2] == pytest.approx(12.0)
    assert not np.isfinite(v[1])


def test_one_value_per_atom_is_taken_atom_for_atom():
    m = _mol("CCO", "e")
    m.SetProp("t", " ".join(str(i) for i in range(m.GetNumAtoms())))
    v = read_atom_target(m, "t")
    assert v == pytest.approx(np.arange(m.GetNumAtoms(), dtype=float))


def test_a_list_of_the_wrong_length_names_both_counts():
    m = _mol("CCO", "ethanol")
    m.SetProp("t", "1 2 3 4")
    with pytest.raises(ValueError, match="ethanol.*4 values.*atoms"):
        read_atom_target(m, "t")


def test_a_molecule_that_carries_nothing_is_not_an_error():
    v = read_atom_target(_mol("CCO", "e"), "nothing_here")
    assert len(v) and not np.isfinite(v).any()


# -------------------------------------------------------------------------- the sample index


@pytest.fixture(scope="module")
def targets():
    mols = [_with_gaps(s, f"m{i}", seed=i + 1) for i, s in enumerate(SMILES)]
    return mols, AtomTargets.from_molecules(mols, "shift",
                                            names=[f"m{i}" for i in range(len(mols))])


def test_the_index_is_the_measured_pairs_only(targets):
    mols, t = targets
    assert len(t) == sum(int(np.isfinite(read_atom_target(m, "shift")).sum()) for m in mols)
    assert len(t.molecule) == len(t.atom) == len(t.y) == len(t)
    assert np.isfinite(t.y).all()
    for r in range(len(t)):
        k, i = int(t.molecule[r]), int(t.atom[r])
        assert mols[k].GetAtomWithIdx(i).GetAtomicNum() > 1
        assert t.per_molecule[k][i] == pytest.approx(t.y[r])


def test_it_remembers_which_molecule_each_sample_came_from(targets):
    _mols, t = targets
    assert t.n_mol == len(SMILES)
    assert t.n_measured_molecules == len(SMILES)
    counts = t.counts()
    assert sum(counts.values()) == len(t)
    assert set(counts) == set(range(len(SMILES)))
    assert len(t.of_molecule(0)) == counts[0]


def test_the_description_says_what_there_is(targets):
    _mols, t = targets
    text = t.describe()
    assert "shift" in text and "atoms measured in" in text
    assert AtomTargets(name="nothing").describe() == "no atom carries 'nothing'"


def test_a_subset_keeps_the_molecule_of_every_sample(targets):
    _mols, t = targets
    rows = [0, 1, len(t) - 1]
    s = t.subset(rows)
    assert len(s) == 3
    assert s.molecule.tolist() == [int(t.molecule[r]) for r in rows]
    assert s.n_mol == t.n_mol


# ----------------------------------------------------------------- folds that keep a molecule


def test_a_molecule_is_never_split_between_two_folds(targets):
    _mols, t = targets
    folds = t.folds(k=3, seed=0)
    assert folds
    seen = set()
    for f in folds:
        here = {int(t.molecule[r]) for r in f}
        assert not (here & seen), "a molecule appeared in two folds"
        seen |= here
    assert sum(len(f) for f in folds) == len(t)
    assert seen == set(t.counts())


def test_asking_for_more_folds_than_molecules_gives_one_fold_each(targets):
    _mols, t = targets
    folds = t.folds(k=50, seed=0)
    assert len(folds) == t.n_measured_molecules


def test_folds_of_an_empty_index_are_no_folds():
    assert AtomTargets(name="x").folds(5) == []


# ----------------------------------------------------------- a prediction goes back on the atom


def test_a_prediction_is_spread_back_over_the_molecules(targets):
    _mols, t = targets
    pred = t.y + 0.5
    back = scatter(t, pred)
    assert len(back) == t.n_mol
    for r in range(len(t)):
        k, i = int(t.molecule[r]), int(t.atom[r])
        assert back[k][i] == pytest.approx(pred[r])
    # and an atom that was never measured has no prediction either
    assert np.isnan(back[0][[a.GetIdx() for a in _mols[0].GetAtoms()
                             if a.GetAtomicNum() == 1][0]])


def test_the_wrong_number_of_values_is_refused(targets):
    _mols, t = targets
    with pytest.raises(ValueError, match="values for"):
        scatter(t, np.zeros(len(t) + 1))


def test_the_file_written_can_be_read_back_atom_for_atom(targets, tmp_path):
    mols, t = targets
    path = os.path.join(str(tmp_path), "out.sdf")
    pred = t.y + 0.5
    assert write_atom_predictions(path, mols, t, pred,
                                  names=[f"m{i}" for i in range(len(mols))]) == len(mols)
    back = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    assert len(back) == len(mols)
    again = AtomTargets.from_molecules(back, "shift")
    assert len(again) == len(t)
    assert again.y == pytest.approx(t.y)
    assert again.atom.tolist() == t.atom.tolist()
    # the prediction came back on the same atoms, half a unit above
    p = AtomTargets.from_molecules(back, "shift_predicted")
    assert p.atom.tolist() == t.atom.tolist()
    assert p.y == pytest.approx(t.y + 0.5)


def test_the_measured_values_alone_can_be_written(targets, tmp_path):
    mols, t = targets
    path = os.path.join(str(tmp_path), "measured.sdf")
    write_atom_predictions(path, mols, t)
    back = [m for m in Chem.SDMolSupplier(path, removeHs=False) if m is not None]
    assert AtomTargets.from_molecules(back, "shift").y == pytest.approx(t.y)
    assert len(AtomTargets.from_molecules(back, "shift_predicted")) == 0


# ------------------------------------------------------------------- through the data set


def test_a_data_set_carries_the_target(tmp_path):
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for i, s in enumerate(SMILES[:3]):
            m = _with_gaps(s, f"m{i}", seed=i + 1)
            m.SetProp("activity", f"{5.0 + i:.2f}")
            w.write(m)
    ds = load_dataset(path, atom_activity="shift")
    assert ds.atom_activity_prop == "shift"
    assert ds.atom_targets is not None and len(ds.atom_targets) > 0
    assert ds.atom_targets.n_mol == len(ds)
    assert np.isfinite(ds.activity).all()        # the molecule-level activity is still there
    # and it can be asked for afterwards instead
    plain = load_dataset(path)
    assert plain.atom_targets is None
    got = plain.set_atom_activity("shift")
    assert len(got) == len(ds.atom_targets)


def test_the_command_reports_what_a_file_carries(tmp_path, capsys):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        for i, s in enumerate(SMILES[:3]):
            w.write(_with_gaps(s, f"m{i}", seed=i + 1))
    out = os.path.join(str(tmp_path), "written.sdf")
    capsys.readouterr()
    main(["atoms", path, "--target", "shift", "--folds", "3", "--out", out])
    text = capsys.readouterr().out
    assert "atoms measured in" in text
    assert "no molecule is split between two of them" in text
    assert os.path.exists(out)
    main(["atoms", path, "--list"])
    assert "shift" in capsys.readouterr().out


def test_a_target_nothing_carries_is_refused_by_name(tmp_path):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(path) as w:
        w.write(_mol("CCO", "e"))
    with pytest.raises(SystemExit, match="no atom carries"):
        main(["atoms", path, "--target", "shift"])
