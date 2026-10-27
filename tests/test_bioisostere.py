"""A bioisostere table weighted by the co-field (plan item MC1).

The claim the module rests on is that the prediction of the overlap kernel is a sum over the
atoms, so the part of it that belongs to a substituent can be read off without refitting
anything.  These tests fix that identity, the masking it needs, the pair rule, and the two
summaries built on top of them.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.bioisostere import (MatchedPair, Substituent, biggest_core,  # noqa: E402
                                 build_table, fingerprint_distance, fragment, group_contribution,
                                 group_distance, masked_field, matched_pairs, predicted_change,
                                 rank_gain, score_pairs, spearman, substituents)
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.kernels import kernel_matrices  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

#: four para-substituted benzamides: one core, four groups
SERIES = [("amide", "NC(=O)c1ccc(C)cc1", 5.0), ("acid", "OC(=O)c1ccc(C)cc1", 4.2),
          ("nitrile", "N#Cc1ccc(C)cc1", 5.6), ("aldehyde", "O=Cc1ccc(C)cc1", 4.8),
          ("ester", "COC(=O)c1ccc(C)cc1", 5.2), ("ketone", "CC(=O)c1ccc(C)cc1", 5.4)]


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def _series():
    mols = [_mol(s, seed=k + 1) for k, (_n, s, _y) in enumerate(SERIES)]
    return mols, [n for n, _s, _y in SERIES], np.array([y for _n, _s, y in SERIES])


def _model(mols, names, y, fields=("electrostatic", "steric")):
    fs = FieldSet.create(physchem=list(fields), alpha=1.0)
    built = [build_fields(m, fs, n) for m, n in zip(mols, names)]
    model = CMFModel(fieldset=fs, method="krr", normalize=True)
    model.params["lam"] = 1e-2
    model.fit(AtomTable.from_molecules(built), y, cv_folds=3, cv_seed=0)
    return model, built


# ------------------------------------------------------------------ cutting a molecule


def test_a_cut_is_reported_in_both_directions():
    """Which piece is the core is decided by the set, so one cut gives two readings."""
    got = fragment(_mol("NC(=O)c1ccc(C)cc1"), max_substituent=12, min_core=0.0)
    pairs = {(f.core, f.group) for f in got}
    assert any(g == "NC(=O)[*:1]" for _c, g in pairs), "the amide must be a group"
    assert any(c == "NC(=O)[*:1]" for c, _g in pairs), "and also a core"


def test_the_atom_indices_point_into_the_molecule_that_was_cut():
    mol = _mol("NC(=O)c1ccc(C)cc1")
    for f in fragment(mol, min_core=0.0):
        assert f.atoms.min() >= 0 and f.atoms.max() < mol.GetNumAtoms()
        assert len(set(f.atoms.tolist())) == len(f.atoms)


def test_a_group_larger_than_the_limit_is_not_a_group():
    mol = _mol("NC(=O)c1ccc(C)cc1")
    small = fragment(mol, max_substituent=2, min_core=0.0)
    assert small, "something must still be cut off"
    assert max(len(f.atoms) for f in small) <= 2


def test_the_core_has_to_be_most_of_the_molecule():
    mol = _mol("NC(=O)c1ccc(C)cc1")
    assert all(len(f.atoms) <= mol.GetNumAtoms() / 2
               for f in fragment(mol, min_core=0.5))


# -------------------------------------------------------------------- matched pairs


def test_two_molecules_with_one_core_and_two_groups_are_a_pair():
    mols, names, y = _series()
    pairs = matched_pairs(mols[:2], y[:2], names[:2])
    assert len(pairs) == 1
    p = pairs[0]
    assert p.a == 0 and p.b == 1
    assert set(p.groups) == {"NC(=O)[*:1]", "O=C(O)[*:1]"}
    assert p.delta == pytest.approx(y[1] - y[0]), "the second minus the first"
    assert "amide" in str(p) and "acid" in str(p)


def test_a_couple_is_reported_once_even_when_several_cores_match():
    mols, names, y = _series()
    once = matched_pairs(mols, y, names)
    many = matched_pairs(mols, y, names, one_per_couple=False)
    couples = {(p.a, p.b) for p in once}
    assert len(once) == len(couples), "one row per couple"
    assert len(many) >= len(once)


def test_a_molecule_without_an_activity_makes_no_pair():
    mols, names, y = _series()
    y = y.copy()
    y[1] = np.nan
    assert all(1 not in (p.a, p.b) for p in matched_pairs(mols, y, names))


def test_the_core_with_the_most_groups_is_the_one_a_table_is_built_on():
    mols, names, _y = _series()
    subs = substituents(mols)
    core = biggest_core(subs, least=3)
    assert core is not None
    assert len({s.group for s in subs if s.core == core}) >= 3


# ------------------------------------------------------------------------- masking


def test_the_masked_field_keeps_the_atoms_where_they_were():
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["electrostatic"], alpha=1.0)
    whole = build_fields(mols[0], fs, "amide")
    part = masked_field(whole, [0, 1, 2])
    assert part.n_atoms == 3
    assert np.allclose(part.coords, np.asarray(whole.coords)[[0, 1, 2]])
    assert np.allclose(part.weights, np.asarray(whole.weights)[[0, 1, 2]])
    assert list(part.member_atom_map) == [0, 1, 2]


def test_masking_a_molecule_in_two_splits_its_kernel_exactly():
    """The kernel is a sum over atom pairs, which is the identity the whole item rests on."""
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    specs = [s for s in fs.specs if s.enabled]
    whole = build_fields(mols[0], fs, "amide")
    left = list(range(3))
    right = [i for i in range(whole.n_atoms) if i not in left]
    a, b = masked_field(whole, left), masked_field(whole, right)
    K = kernel_matrices(AtomTable.from_molecules([whole]),
                        AtomTable.from_molecules([whole]), specs, symmetric=True)
    P = AtomTable.from_molecules([a, b])
    k = kernel_matrices(P, P, specs, symmetric=True)
    for f in range(len(specs)):
        assert K[f][0, 0] == pytest.approx(
            k[f][0, 0] + 2.0 * k[f][0, 1] + k[f][1, 1], rel=1e-9)


def test_masking_asks_for_atoms_that_exist():
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["electrostatic"], alpha=1.0)
    whole = build_fields(mols[0], fs, "amide")
    with pytest.raises(ValueError, match="subset"):
        masked_field(whole, [0, 10 ** 6])


# ------------------------------------------------- the part of the prediction it owns


def test_the_parts_of_a_molecule_add_up_to_its_prediction():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    whole = built[0]
    left = list(range(4))
    right = [i for i in range(whole.n_atoms) if i not in left]
    total = (group_contribution(model, whole, left)
             + group_contribution(model, whole, right))
    predicted = float(model.predict(AtomTable.from_molecules([whole]))[0])
    assert total + model.bias == pytest.approx(predicted, rel=1e-8, abs=1e-8)


def test_other_coefficients_give_another_number_and_do_not_stay_behind():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    before = group_contribution(model, built[0], [0, 1, 2])
    other = np.zeros(len(model.coef))
    other[0] = 1.0
    changed = group_contribution(model, built[0], [0, 1, 2], coef=other)
    assert changed != pytest.approx(before)
    assert group_contribution(model, built[0], [0, 1, 2]) == pytest.approx(before), \
        "the model's own coefficients must be put back"


def test_the_predicted_change_turns_round_with_the_pair():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    pairs = matched_pairs(mols, y, names)
    p = pairs[0]
    there = predicted_change(model, built[p.a], p.atoms_a, built[p.b], p.atoms_b)
    back = predicted_change(model, built[p.b], p.atoms_b, built[p.a], p.atoms_a)
    assert there == pytest.approx(-back)


# ----------------------------------------------------------------- the two distances


def test_the_field_distance_is_zero_between_a_group_and_itself():
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    whole = build_fields(mols[0], fs, "amide")
    a = masked_field(whole, [0, 1, 2])
    assert group_distance(a, a, fs) == pytest.approx(0.0, abs=1e-9)


def test_the_field_distance_grows_when_the_groups_differ_more():
    mols, names, y = _series()
    fs = FieldSet.create(physchem=["electrostatic", "steric"], alpha=1.0)
    whole = build_fields(mols[0], fs, "amide")
    one = masked_field(whole, [0])
    two = masked_field(whole, [0, 1])
    three = masked_field(whole, [0, 1, 2])
    assert group_distance(one, two, fs) < group_distance(one, three, fs)


def test_the_fingerprint_distance_is_zero_for_the_same_group_and_one_for_strangers():
    assert fingerprint_distance("NC(=O)[*:1]", "NC(=O)[*:1]") == pytest.approx(0.0)
    assert fingerprint_distance("C[*:1]", "c1ccc2ccccc2c1[*:1]") > 0.8
    assert np.isnan(fingerprint_distance("NC(=O)[*:1]", "not a molecule at all"))


# ------------------------------------------------------------------- the two summaries


def test_scoring_a_set_of_pairs_fills_every_column():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    pairs = matched_pairs(mols, y, names)
    scores = score_pairs(model, built, pairs)
    assert len(scores) == len(pairs)
    for s in scores:
        assert np.isfinite(s.fingerprint) and s.field >= 0
        assert s.measured == s.pair.delta
    assert "->" in str(scores[0])


def test_other_coefficients_reach_the_scores():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    pairs = matched_pairs(mols, y, names)
    zero = score_pairs(model, built, pairs, coef_of=lambda p: np.zeros(len(model.coef)))
    assert all(s.change == pytest.approx(0.0) for s in zero), \
        "with no coefficients the model predicts no change"


def test_the_rank_correlation_and_its_ties():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) != spearman([1, 2, 3, 4], [1, 2, 3, 4])
    assert np.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))
    # ties are averaged, so a measure that cannot separate two pairs is not rewarded
    assert spearman([1, 1, 2, 2], [1, 2, 3, 4]) == pytest.approx(0.894, abs=1e-3)


def test_the_rank_gain_says_which_of_two_measures_is_better():
    rng = np.random.RandomState(0)
    measured = rng.rand(60)
    good = measured + 0.05 * rng.randn(60)
    bad = rng.rand(60)
    d, lo, hi = rank_gain(good, bad, measured, runs=300)
    assert d > 0.5 and lo > 0 and hi > lo
    back = rank_gain(bad, good, measured, runs=300)
    assert back[0] == pytest.approx(-d, abs=1e-12)


# ----------------------------------------------------------------------- the table


def test_the_table_is_antisymmetric_in_the_change_and_symmetric_in_the_distance():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    subs = substituents(mols)
    core = biggest_core(subs, least=3)
    t = build_table(model, built, core, subs, half_width=0.5, names=names)
    assert len(t.groups) >= 3
    assert np.allclose(t.distance, t.distance.T)
    assert np.allclose(t.change, -t.change.T)
    assert np.allclose(np.diag(t.distance), 0.0) and np.allclose(np.diag(t.change), 0.0)
    assert len(list(t.rows())) == len(t.groups) * (len(t.groups) - 1)


def test_interchangeable_follows_the_bound_it_is_given():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    subs = substituents(mols)
    core = biggest_core(subs, least=3)
    t = build_table(model, built, core, subs, half_width=0.0, names=names)
    g = t.groups[0]
    assert t.interchangeable(g, within=0.0) == [], "nothing is interchangeable at zero"
    assert len(t.interchangeable(g, within=1e9)) == len(t.groups) - 1
    with pytest.raises(ValueError, match="not in this table"):
        t.interchangeable("[*:1]C#N#C")


def test_a_table_needs_two_groups():
    mols, names, y = _series()
    model, built = _model(mols, names, y)
    with pytest.raises(ValueError, match="needs two"):
        build_table(model, built, "a core nothing has", [], half_width=0.5)


# ------------------------------------------------------------------------ the command


def test_the_command_prints_the_nearest_partners_and_writes_the_table(tmp_path, capsys):
    from cmfqsar.cli import main
    mols, names, y = _series()
    data = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(data) as w:
        for m, n, v in zip(mols, names, y):
            m.SetProp("_Name", n)
            m.SetProp("activity", f"{v:.2f}")
            w.write(m)
    model = os.path.join(str(tmp_path), "m.cmf")
    main(["build", data, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "1.0", "--lams", "0.1", "--save", model])
    csv = os.path.join(str(tmp_path), "table.csv")
    capsys.readouterr()
    main(["isosteres", model, "--molecules", data, "--activity", "activity", "--csv", csv])
    text = capsys.readouterr().out
    assert "groups on it" in text and "conformal half width" in text
    rows = open(csv, encoding="utf-8").read().strip().splitlines()
    assert len(rows) > 3 and "predicted change" in rows[0]


def test_the_command_says_so_when_no_position_carries_enough_groups(tmp_path):
    from cmfqsar.cli import main
    mols, names, y = _series()
    data = os.path.join(str(tmp_path), "set.sdf")
    with Chem.SDWriter(data) as w:
        for m, n, v in zip(mols, names, y):
            m.SetProp("_Name", n)
            m.SetProp("activity", f"{v:.2f}")
            w.write(m)
    model = os.path.join(str(tmp_path), "m.cmf")
    main(["build", data, "--activity", "activity", "--fields", "electrostatic",
          "--alphas", "1.0", "--lams", "0.1", "--save", model])
    with pytest.raises(SystemExit, match="different groups"):
        main(["isosteres", model, "--molecules", data, "--least", "99"])
