"""Co-field-guided design: what to put at a position (plan item E3).

The loop has one property it must not lose: replacing a group with itself has to give the parent
back, atom for atom. Everything else -- the ranking, the annotations -- is only as good as that.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.design import (COMMON_FRAGMENTS, Candidate, as_fragment, design,  # noqa: E402
                            enumerate_replacements, read_fragments)
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.topomer import (attachment_label, series_labels, substituents,  # noqa: E402
                             topomer_assembly, topomer_fields, topomer_molecule)

CORE = Chem.MolFromSmiles("c1ccccc1")
SERIES = ["Oc1ccccc1C", "Nc1ccccc1C", "Clc1ccccc1C", "Oc1ccccc1CC", "Nc1ccccc1CC",
          "Clc1ccccc1CC", "Oc1ccccc1F", "Nc1ccccc1F", "C(=O)(O)c1ccccc1C", "Cc1ccccc1C"]
Y = np.array([1.0, 2.0, 0.8, 1.3, 2.4, 1.1, 0.6, 1.9, 3.0, 0.9])


def _mols():
    return [Chem.MolFromSmiles(s) for s in SERIES]


@pytest.fixture(scope="module")
def world():
    mols = _mols()
    names = [f"m{i}" for i in range(len(mols))]
    labels = series_labels(mols, CORE)
    fs = FieldSet.create(alpha=1.0)
    mf, _log = topomer_fields(mols, names, fs, core=CORE, labels=labels)
    keep = [i for i, f in enumerate(mf) if f is not None]
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1}, normalize=True)
    model.fit(AtomTable.from_molecules([mf[i] for i in keep]), Y[keep], cv_folds=3, cv_seed=0)
    return model, mols, names, labels, keep


def test_a_library_smiles_becomes_a_fragment_of_the_right_site():
    frag = as_fragment("[*]C(=O)N", 7)
    assert frag is not None and attachment_label(frag) == 7
    assert as_fragment("CCO", 1) is None                 # nothing to attach by
    assert as_fragment("[*]CC[*]", 1) is None            # two attachment points, not a group
    assert as_fragment("not a molecule", 1) is None


def test_the_built_in_library_is_made_of_attachable_groups():
    frags = read_fragments("common")
    assert frags == list(COMMON_FRAGMENTS) and len(frags) > 20
    for smi in frags:
        f = as_fragment(smi, 3)
        assert f is not None, smi
        assert sum(1 for a in f.GetAtoms() if a.GetAtomicNum() == 0) == 1


def test_a_library_file_is_read_and_its_comments_ignored(tmp_path):
    p = tmp_path / "frags.smi"
    p.write_text("# a comment\n[*]F  fluoro\n\n[*]Cl chloro\n", encoding="utf-8")
    assert read_fragments(str(p)) == ["[*]F", "[*]Cl"]


def test_replacing_a_group_with_itself_gives_the_parent_back():
    """The identity that makes the ranking a ranking of substituents."""
    mols = _mols()
    parent = mols[0]
    labels = series_labels(mols, CORE)
    frags = substituents(parent, CORE)
    label = attachment_label(frags[0])
    own = Chem.MolToSmiles(frags[0])
    built, _failed = enumerate_replacements(parent, CORE, label, [own], labels)
    assert len(built) == 1
    made = Chem.MolToSmiles(built[0][1])
    assert made == Chem.MolToSmiles(topomer_assembly(frags, labels))
    assert made == Chem.MolToSmiles(topomer_molecule(parent, CORE, labels))


def test_the_other_positions_are_left_alone():
    mols = _mols()
    parent = mols[0]                       # two substituents, one of which is varied
    labels = series_labels(mols, CORE)
    frags = substituents(parent, CORE)
    assert len(frags) == 2
    label = attachment_label(frags[0])
    kept = Chem.MolToSmiles(frags[1])
    built, _failed = enumerate_replacements(parent, CORE, label, ["[*]F", "[*]C#N"], labels)
    assert len(built) == 2
    for _smi, _mol, key in built:
        assert f"{attachment_label(frags[1])}:{kept}" in key


def test_a_fragment_the_rule_cannot_pose_is_reported_not_dropped_silently():
    mols = _mols()
    labels = series_labels(mols, CORE)
    label = attachment_label(substituents(mols[0], CORE)[0])
    built, failed = enumerate_replacements(mols[0], CORE, label,
                                           ["[*]F", "[*]CC[*]", "nonsense"], labels)
    assert [b[0] for b in built] == ["[2*]F"] or len(built) == 1
    assert set(failed) == {"[*]CC[*]", "nonsense"}


def test_the_electrostatic_field_survives_being_cut_into_fragments():
    """Gasteiger returns nan for every atom of a molecule holding a dummy; capping fixes it."""
    mols = _mols()
    labels = series_labels(mols, CORE)
    fs = FieldSet.create(alpha=1.0)
    mf = build_fields(topomer_molecule(mols[0], CORE, labels), fs, "x")
    e = np.abs(mf.weights[:, 0]).sum()
    assert e > 0.1, "the topomer assembly has no electrostatic field at all"
    assert np.isfinite(mf.weights).all()


def test_the_ranking_comes_back_ordered_and_annotated(world):
    model, mols, names, labels, keep = world
    parent = mols[0]
    label = attachment_label(substituents(parent, CORE)[0])
    own = [Chem.MolToSmiles(f) for m in mols for f in substituents(m, CORE)
           if attachment_label(f) == label]
    res = design(model, parent, CORE, label, fragments=sorted(set(own)) + ["[*]C#N"],
                 labels=labels, parent_name=names[0], series=mols, series_names=names,
                 series_y=Y, known=set(own), with_fields=2)
    assert res.candidates and res.posed == len(res.candidates)
    scores = [c.score for c in res.candidates]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(c, Candidate) for c in res.candidates)
    assert np.isfinite(res.parent_mean) and np.isfinite(res.parent_sd)
    # the parent is one of the series, so its measured activity is known
    assert res.parent_measured == pytest.approx(Y[0])
    # a candidate that reproduces a molecule of the series carries that molecule's measurement
    named = [c for c in res.candidates if c.made is not None]
    assert named, "replacing a group by one the series uses should rebuild a known molecule"
    for c in named:
        assert np.isfinite(c.measured)
    # the new group is marked new, the ones the series uses are marked known
    assert any(not c.known for c in res.candidates)
    assert any(c.known for c in res.candidates)
    assert "position" in res.describe()


def test_the_field_difference_is_computed_only_where_it_was_asked_for(world):
    model, mols, names, labels, _keep = world
    label = attachment_label(substituents(mols[0], CORE)[0])
    res = design(model, mols[0], CORE, label, fragments=["[*]F", "[*]Cl", "[*]C#N", "[*]O"],
                 labels=labels, parent_name=names[0], with_fields=2)
    assert [bool(c.fields) for c in res.candidates][:2] == [True, True]
    assert not any(c.fields for c in res.candidates[2:])
    rows = res.candidates[0].fields
    assert len(rows) == len(model.fieldset.specs)
    assert all(np.isfinite(r.distance) for r in rows)


def test_the_delta_is_against_the_parent_posed_the_same_way(world):
    model, mols, names, labels, _keep = world
    label = attachment_label(substituents(mols[0], CORE)[0])
    res = design(model, mols[0], CORE, label, fragments=["[*]F", "[*]Cl"], labels=labels,
                 parent_name=names[0])
    for c in res.candidates:
        assert c.delta == pytest.approx(c.mean - res.parent_mean)

def test_a_group_is_one_group_however_its_hydrogens_are_written():
    """A series read from a file carries explicit hydrogens; a library does not.

    The two also disagree about what a substituent is: with explicit hydrogens every free ring
    position carries one, and those are groups the loop can replace -- which is how it proposes
    putting something where there was nothing. What must agree is the name of a real group.
    """
    from cmfqsar.design import fragment_smiles

    explicit = Chem.AddHs(Chem.MolFromSmiles("Oc1ccccc1C"))
    frags = substituents(explicit, CORE)
    assert any(a.GetAtomicNum() == 1 for f in frags for a in f.GetAtoms()), "no explicit H to test"
    named = {fragment_smiles(f) for f in frags}
    plain = {fragment_smiles(f) for f in substituents(_mols()[0], CORE)}
    assert plain and plain <= named
    extra = named - plain
    assert extra and all("[H]" in e for e in extra)


def test_a_candidate_that_rebuilds_a_molecule_of_a_file_series_is_recognised():
    """The annotation that failed silently when the fragment names did not agree."""
    mols = [Chem.AddHs(m) for m in _mols()]
    names = [f"m{i}" for i in range(len(mols))]
    labels = series_labels(mols, CORE)
    fs = FieldSet.create(alpha=1.0)
    mf, _log = topomer_fields(mols, names, fs, core=CORE, labels=labels)
    keep = [i for i, f in enumerate(mf) if f is not None]
    model = CMFModel(fieldset=fs, method="krr", params={"lam": 0.1}, normalize=True)
    model.fit(AtomTable.from_molecules([mf[i] for i in keep]), Y[keep], cv_folds=3, cv_seed=0)
    label = attachment_label(substituents(mols[0], CORE)[0])
    library = sorted({Chem.MolToSmiles(f) for m in mols for f in substituents(m, CORE)
                      if attachment_label(f) == label})
    res = design(model, mols[0], CORE, label, fragments=library, labels=labels,
                 parent_name=names[0], series=mols, series_names=names, series_y=Y)
    assert any(c.made is not None for c in res.candidates),         "replacing a group by one the series uses must rebuild a molecule of the series"
    assert np.isfinite(res.parent_measured)
