"""Designing on a model that needs no frame (plan item SC7).

`cmf design` built its candidates with the topomer rule, which exists to give a substituent
nobody has made a pose. A rotation-invariant model needs no pose, so a candidate can be the whole
molecule, embedded once -- which is what makes a design loop possible on a table of guests, where
there is no common scaffold to superimpose on. These tests fix the two halves of that: the
molecule is rebuilt correctly from the core and its groups, and the route is refused for a kernel
that is an overlap integral, where a candidate with no frame cannot be scored.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.design import (COMMON_FRAGMENTS, as_fragment, design,  # noqa: E402
                            embed, enumerate_molecules, group_contribution, whole_molecule)
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.topomer import substituents  # noqa: E402

CORE = "C(=O)O"
PARENT = "CC(=O)O"
SERIES = ["CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "CCCCC(=O)O", "O=C(O)C1CCCCC1",
          "O=C(O)c1ccccc1", "O=C(O)Cc1ccccc1", "O=C(O)C1CCCC1", "CC(C)C(=O)O",
          "O=C(O)CC1CCCCC1"]
FRAGMENTS = ["[*]C", "[*]CC", "[*]C1CCCCC1", "[*]c1ccccc1", "[*]C12CC3CC(CC(C3)C1)C2"]


def _mol3d(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    AllChem.MMFFOptimizeMolecule(m)
    Chem.SanitizeMol(m)
    return m


@pytest.fixture(scope="module")
def invariant():
    mols = [_mol3d(s, i + 1) for i, s in enumerate(SERIES)]
    names = [f"m{i}" for i in range(len(mols))]
    y = np.array([1.3, 1.5, 2.0, 2.5, 3.6, 2.1, 2.4, 3.2, 1.8, 3.5])
    fs = FieldSet.create(alpha=1.0)
    mf = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1},
                     kernel_source="autocorrelation")
    model.fit(AtomTable.from_molecules(mf), y, cv_folds=0)
    return {"model": model, "mols": mols, "names": names, "y": y, "fs": fs}


# ------------------------------------------------------------------- putting a molecule back


def test_the_molecule_is_rebuilt_from_its_core_and_its_groups():
    core = Chem.MolFromSmiles(CORE)
    parent = Chem.MolFromSmiles(PARENT)
    frags = substituents(parent, core)
    back = whole_molecule(core, frags)
    assert Chem.MolToSmiles(back) == Chem.MolToSmiles(parent)


def test_a_fragment_from_the_library_makes_the_molecule_it_should():
    core = Chem.MolFromSmiles(CORE)
    frag = as_fragment("[*]C12CC3CC(CC(C3)C1)C2", 0)
    got = whole_molecule(core, [frag])
    assert Chem.MolToSmiles(got) == Chem.MolToSmiles(
        Chem.MolFromSmiles("O=C(O)C12CC3CC(CC(C3)C1)C2"))      # adamantane-1-carboxylic acid


def test_a_fragment_that_cannot_be_attached_gives_nothing():
    core = Chem.MolFromSmiles(CORE)
    frag = as_fragment("[*]C", 9)              # there is no atom 9 in this core
    assert whole_molecule(core, [frag]) is None


def test_the_candidate_is_embedded_in_three_dimensions():
    m = embed(Chem.MolFromSmiles("O=C(O)C1CCCCC1"), seed=3)
    assert m is not None and m.GetNumConformers() == 1
    pos = np.array(m.GetConformer().GetPositions())
    assert pos.shape[1] == 3 and np.ptp(pos, axis=0).min() > 0.5
    assert any(a.GetAtomicNum() == 1 for a in m.GetAtoms())     # hydrogens are there


def test_every_other_group_of_the_parent_is_kept():
    core = Chem.MolFromSmiles("c1ccccc1")
    parent = Chem.MolFromSmiles("Cc1ccc(C(=O)O)cc1")            # two groups on a ring
    frags = substituents(parent, core)
    label = max(f.GetAtomWithIdx(next(a.GetIdx() for a in f.GetAtoms()
                                      if a.GetAtomicNum() == 0)).GetIsotope() for f in frags)
    built, failed = enumerate_molecules(parent, core, int(label), ["[*]Cl"], seed=1)
    assert not failed and len(built) == 1
    smi = Chem.MolToSmiles(Chem.RemoveHs(built[0][1]))
    assert "Cl" in smi
    assert Chem.MolFromSmiles(smi).HasSubstructMatch(Chem.MolFromSmiles("C(=O)O")) \
        or Chem.MolFromSmiles(smi).HasSubstructMatch(Chem.MolFromSmiles("Cc1ccccc1"))


# ------------------------------------------------------------------------- the design itself


def test_the_design_runs_on_an_invariant_model(invariant):
    res = design(invariant["model"], Chem.MolFromSmiles(PARENT), Chem.MolFromSmiles(CORE), 0,
                 fragments=FRAGMENTS, acquire="greedy", pose="molecule",
                 series=invariant["mols"], series_names=invariant["names"],
                 series_y=invariant["y"], with_fields=len(FRAGMENTS))
    assert len(res.candidates) == len(FRAGMENTS)
    assert all(c.molecule for c in res.candidates)
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(np.isfinite(c.mean) and c.sd >= 0 for c in res.candidates)


def test_the_route_is_chosen_by_the_kernel(invariant):
    res = design(invariant["model"], Chem.MolFromSmiles(PARENT), Chem.MolFromSmiles(CORE), 0,
                 fragments=FRAGMENTS[:2], acquire="greedy", pose="auto")
    assert all(c.molecule for c in res.candidates)          # the molecule route was taken


def test_an_overlap_kernel_refuses_the_molecule_route(invariant):
    fs = invariant["fs"]
    mf = [build_fields(m, fs, n) for m, n in zip(invariant["mols"], invariant["names"])]
    aligned = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    aligned.fit(AtomTable.from_molecules(mf), invariant["y"], cv_folds=0)
    with pytest.raises(ValueError, match="no frame in common"):
        design(aligned, Chem.MolFromSmiles(PARENT), Chem.MolFromSmiles(CORE), 0,
               fragments=FRAGMENTS[:2], pose="molecule")


def test_an_unknown_route_is_refused_by_name(invariant):
    with pytest.raises(ValueError, match="unknown pose"):
        design(invariant["model"], Chem.MolFromSmiles(PARENT), Chem.MolFromSmiles(CORE), 0,
               fragments=FRAGMENTS[:2], pose="sideways")


def test_the_group_contribution_adds_up_to_the_prediction(invariant):
    """The split over atoms is exact, so the group and the core are the whole of it."""
    model = invariant["model"]
    mol = _mol3d("O=C(O)C12CC3CC(CC(C3)C1)C2", 5)
    fields = build_fields(mol, invariant["fs"], "candidate")
    group, total, atoms = group_contribution(model, fields, mol, Chem.MolFromSmiles(CORE))
    predicted = float(model.predict(AtomTable.from_molecules([fields]))[0])
    assert total == pytest.approx(predicted - model.bias, rel=1e-8, abs=1e-10)
    assert 0.0 < atoms < 1.0
    assert abs(group) <= abs(total) + abs(total - group) + 1e-9


def test_a_flat_acquisition_score_is_reported_as_such(invariant):
    """Expected improvement is flat when nothing on offer can beat the best molecule."""
    res = design(invariant["model"], Chem.MolFromSmiles(PARENT), Chem.MolFromSmiles(CORE), 0,
                 fragments=FRAGMENTS, acquire="ei", pose="molecule")
    if res.flat_score:
        assert "not a ranking" in res.describe()
    else:
        assert len({round(c.score, 12) for c in res.candidates}) > 1


def test_the_known_strong_binder_comes_first(invariant):
    """The point of the thing: adamantyl is the strongest group a cyclodextrin cavity takes."""
    res = design(invariant["model"], Chem.MolFromSmiles(PARENT), Chem.MolFromSmiles(CORE), 0,
                 fragments=FRAGMENTS, acquire="greedy", pose="molecule")
    top = Chem.MolToSmiles(Chem.MolFromSmiles(res.candidates[0].smiles))
    assert top == Chem.MolToSmiles(Chem.MolFromSmiles("[*]C12CC3CC(CC(C3)C1)C2"))


# ----------------------------------------------------------------------------- the command


def test_the_command_takes_the_invariant_route(tmp_path, capsys):
    from cmfqsar.cli import main
    path = os.path.join(str(tmp_path), "guests.sdf")
    with Chem.SDWriter(path) as w:
        for i, (s, y) in enumerate(zip(SERIES, [1.3, 1.5, 2.0, 2.5, 3.6, 2.1, 2.4, 3.2, 1.8,
                                                3.5])):
            m = _mol3d(s, i + 1)
            m.SetProp("_Name", f"m{i}")
            m.SetProp("activity", f"{y:.2f}")
            w.write(m)
    capsys.readouterr()
    main(["design", path, "--kernel", "autocorrelation", "--core", CORE, "--parent", "m0",
          "--fragments", "series", "--top", "3", "--folds", "3", "--acquire", "greedy",
          "--lams", "0.1"])
    text = capsys.readouterr().out
    assert "Invariant model on 10 molecules" in text
    assert "nothing was superimposed on anything" in text
