"""Scoring a conformational restriction (plan item MC9).

What these tests fix is the meaning of the three pieces: the pair rule accepts a ring closure and
refuses a different molecule, the two amounts are the integrals they claim to be, and the score
of a pair is a share that does not depend on how the grid was laid out.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from cmfqsar.cofields import field_volume, make_grid  # noqa: E402
from cmfqsar.fields import FieldSet, build_fields, combine_members  # noqa: E402
from cmfqsar.fluctuation import variance_field, variance_fieldset  # noqa: E402
from cmfqsar.restriction import (Amounts, Pair, amounts_from, correlation,  # noqa: E402
                                 grid_for, grid_volumes, matched_pairs, score_pair,
                                 score_pairs, weighted_amounts)

#: a flexible molecule, the same eleven atoms with the chain tied into a ring, and a stranger
FLEXIBLE = "CCCCc1ccccc1O"
RESTRICTED = "CC1CCc2ccccc2O1"
STRANGER = "CC(=O)Nc1ccc(S(N)(=O)=O)cc1"


def _mol(smiles, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(m, randomSeed=seed)
    AllChem.MMFFOptimizeMolecule(m)
    return Chem.RemoveHs(m)


def _fieldset(fields=("electrostatic",)):
    return FieldSet.create(physchem=list(fields), alpha=2.0)


def _ensemble(mol, fs, n=3, seed=1, name="m"):
    """The mean field and the variance field of one molecule over a few conformations."""
    from cmfqsar.conformers import generate_conformers, single_conformer
    ens = generate_conformers(mol, n=n, seed=seed)
    k = max(ens.GetNumConformers(), 1)
    members = [build_fields(single_conformer(ens, i) if ens.GetNumConformers() > 1 else ens,
                            fs, name=f"{name}#{i}") for i in range(k)]
    mean = (combine_members(members, np.ones(len(members)), kind="conformers", name=name)
            if len(members) > 1 else members[0])
    var = variance_field(members if len(members) > 1 else members * 2, fs, floor=1e-2, name=name)
    return mean, var


# ----------------------------------------------------------------- the pair rule


def test_a_ring_closure_is_a_pair_and_the_direction_is_fixed():
    mols = [_mol(FLEXIBLE), _mol(RESTRICTED)]
    pairs = matched_pairs(mols, [6.0, 7.0], names=["flex", "tied"])
    assert len(pairs) == 1
    p = pairs[0]
    assert (p.flexible, p.restricted) == (0, 1), "the molecule with more rings is the restricted"
    assert p.rotatable_removed >= 1
    assert p.common == pytest.approx(1.0), "the same eleven atoms, one bond more"
    assert p.gain == pytest.approx(1.0), "the gain is the restricted minus the flexible"
    assert "flex" in str(p) and "tied" in str(p)


def test_the_fingerprint_these_two_molecules_would_fail_is_not_what_decides():
    """The rule has to accept a ring closure, which a fingerprint sees as a different molecule."""
    from rdkit.Chem import AllChem
    from rdkit.DataStructs import TanimotoSimilarity
    a, b = Chem.MolFromSmiles(FLEXIBLE), Chem.MolFromSmiles(RESTRICTED)
    fp = [AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) for m in (a, b)]
    assert TanimotoSimilarity(*fp) < 0.3, "if this rises, the note in matched_pairs needs a look"
    assert len(matched_pairs([_mol(FLEXIBLE), _mol(RESTRICTED)], [6.0, 7.0])) == 1


def test_a_changed_substituent_is_not_a_restriction():
    """One more ring and one atom swapped: the composition test keeps it out."""
    mols = [_mol("CCCCc1ccccc1O"), _mol("CC1CCc2ccccc2N1")]
    assert matched_pairs(mols, [6.0, 7.0], max_composition=0) == []


def test_a_different_molecule_is_not_a_pair():
    mols = [_mol(FLEXIBLE), _mol(STRANGER)]
    assert matched_pairs(mols, [6.0, 7.0]) == []


def test_a_molecule_without_a_measured_value_makes_no_pair():
    mols = [_mol(FLEXIBLE), _mol(RESTRICTED)]
    assert matched_pairs(mols, [6.0, np.nan]) == []


def test_the_common_substructure_condition_can_refuse_a_ring_closure():
    """Raising the share asked for refuses a pair whose two members differ elsewhere too."""
    mols = [_mol(FLEXIBLE), _mol(RESTRICTED)]
    assert len(matched_pairs(mols, [6.0, 7.0], mcs_share=0.5)) == 1
    assert matched_pairs(mols, [6.0, 7.0], mcs_share=1.01) == []


# ------------------------------------------------------------------ the amounts


def test_the_amounts_are_the_integrals_they_claim_to_be():
    """Summed against the weight by hand, voxel by voxel, they must agree."""
    fs = _fieldset()
    specs = [s for s in fs.specs if s.enabled]
    vspecs = [s for s in variance_fieldset(fs).specs if s.enabled]
    mean, var = _ensemble(_mol(FLEXIBLE), fs)
    grid = grid_for([mean], spacing=1.0)
    rng = np.random.RandomState(0)
    weight = [rng.rand(grid.n_points)]
    amounts = weighted_amounts(weight, [mean], [var], specs, vspecs, grid)
    xbar = field_volume(mean, specs[0], 0, grid).reshape(-1)
    v = np.maximum(field_volume(var, vspecs[0], 0, grid).reshape(-1), 0.0)
    cell = grid.spacing ** 3
    assert amounts.relevant[0] == pytest.approx(
        np.sum(weight[0] ** 2 * xbar ** 2) * cell, rel=1e-6)
    assert amounts.intermittent[0] == pytest.approx(np.sum(weight[0] ** 2 * v) * cell, rel=1e-6)


def test_both_amounts_are_positive_and_the_share_is_their_ratio():
    fs = _fieldset()
    specs = [s for s in fs.specs if s.enabled]
    vspecs = [s for s in variance_fieldset(fs).specs if s.enabled]
    mean, var = _ensemble(_mol(FLEXIBLE), fs)
    grid = grid_for([mean], spacing=1.0)
    a = weighted_amounts([np.ones(grid.n_points)], [mean], [var], specs, vspecs, grid)
    assert a.relevant[0] > 0 and a.intermittent[0] > 0
    assert a.intermittent_share[0] == pytest.approx(a.intermittent[0] / a.relevant[0])


def test_an_ensemble_that_does_not_move_has_no_intermittent_field():
    """The amount is built on the variance, so one repeated conformation must give nothing.

    Whether a ring closure lowers the amount is the question the benchmark measures and is not
    fixed here: with the electrostatic field alone the variance is carried by the charged atoms,
    and a flexible alkyl chain can move a great deal while carrying almost no charge.
    """
    fs = _fieldset()
    specs = [s for s in fs.specs if s.enabled]
    vspecs = [s for s in variance_fieldset(fs).specs if s.enabled]
    mol = _mol(FLEXIBLE)
    one = build_fields(mol, fs, name="still")
    still = variance_field([one, one, one], fs, floor=1e-2, name="still")
    moving = _ensemble(mol, fs, n=5, name="moving")
    grid = grid_for([one], spacing=1.0)
    a = weighted_amounts([np.ones(grid.n_points)], [one, moving[0]], [still, moving[1]],
                         specs, vspecs, grid)
    assert a.intermittent[0] == pytest.approx(0.0, abs=1e-9)
    assert a.intermittent[1] > 1e-6, "an ensemble that does move must have some"


def test_the_volumes_can_be_built_once_and_weighted_twice():
    fs = _fieldset()
    specs = [s for s in fs.specs if s.enabled]
    vspecs = [s for s in variance_fieldset(fs).specs if s.enabled]
    mean, var = _ensemble(_mol(FLEXIBLE), fs)
    grid = grid_for([mean], spacing=1.0)
    volumes = grid_volumes([mean], [var], specs, vspecs, grid)
    assert volumes.shape == (1, len(specs), 2, grid.n_points)
    w = [np.ones(grid.n_points)]
    direct = weighted_amounts(w, [mean], [var], specs, vspecs, grid)
    cached = amounts_from(w, volumes, grid)
    assert cached.relevant[0] == pytest.approx(direct.relevant[0], rel=1e-5)
    assert cached.intermittent[0] == pytest.approx(direct.intermittent[0], rel=1e-5)


def test_only_the_rows_asked_for_are_built():
    fs = _fieldset()
    specs = [s for s in fs.specs if s.enabled]
    vspecs = [s for s in variance_fieldset(fs).specs if s.enabled]
    a = _ensemble(_mol(FLEXIBLE), fs, name="a")
    b = _ensemble(_mol(RESTRICTED), fs, name="b")
    grid = grid_for([a[0], b[0]], spacing=1.0)
    all_rows = grid_volumes([a[0], b[0]], [a[1], b[1]], specs, vspecs, grid)
    one = grid_volumes([a[0], b[0]], [a[1], b[1]], specs, vspecs, grid, rows=[1])
    assert one.shape[0] == 1
    assert np.allclose(one[0], all_rows[1])


# -------------------------------------------------------------------- the score


def test_the_score_is_a_share_of_the_flexible_member():
    amounts = Amounts(relevant=np.array([2.0, 1.0]), intermittent=np.array([4.0, 1.0]))
    s = score_pair(amounts, Pair(0, 1, 0.9, 2, gain=0.5, names=("a", "b")))
    assert s.kept == pytest.approx(0.5), "the restricted member places half the relevant field"
    assert s.removed == pytest.approx(0.75), "three quarters of the intermittent part is gone"
    assert s.gain == 0.5
    assert "a" in str(s) and "b" in str(s)


def test_a_restriction_that_changes_nothing_scores_zero_removed():
    amounts = Amounts(relevant=np.array([3.0, 3.0]), intermittent=np.array([2.0, 2.0]))
    s = score_pair(amounts, Pair(0, 1, 1.0, 1, gain=0.0))
    assert s.kept == pytest.approx(1.0) and s.removed == pytest.approx(0.0)


def test_a_molecule_with_no_field_gives_not_a_number_rather_than_an_exception():
    amounts = Amounts(relevant=np.array([0.0, 1.0]), intermittent=np.array([0.0, 1.0]))
    s = score_pair(amounts, Pair(0, 1, 1.0, 1, gain=0.0))
    assert np.isnan(s.kept) and np.isnan(s.removed)


def test_score_pairs_keeps_the_order_it_was_given():
    amounts = Amounts(relevant=np.array([1.0, 2.0, 4.0]), intermittent=np.array([1.0, 2.0, 4.0]))
    pairs = [Pair(0, 1, 1.0, 1, gain=0.1), Pair(0, 2, 1.0, 1, gain=0.2)]
    got = score_pairs(amounts, pairs)
    assert [s.gain for s in got] == [0.1, 0.2]
    assert got[0].kept == pytest.approx(2.0) and got[1].kept == pytest.approx(4.0)


# -------------------------------------------------------------- the correlation


def test_the_correlation_and_its_interval():
    rng = np.random.RandomState(0)
    x = rng.rand(80)
    y = 2.0 * x + 0.1 * rng.randn(80)
    r, lo, hi = correlation(x, y, runs=300)
    assert r > 0.9 and lo > 0 and lo < r < hi


def test_a_column_that_is_all_the_same_has_no_correlation():
    r, lo, hi = correlation(np.ones(20), np.arange(20.0))
    assert np.isnan(r) and np.isnan(lo) and np.isnan(hi)


def test_pairs_with_a_missing_score_are_dropped_and_not_counted():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([1.0, 2.0, 99.0, 4.0, 5.0, 6.0])
    assert correlation(x, y, runs=100)[0] == pytest.approx(1.0)


# ------------------------------------------------------------------ the command

#: four molecules holding two ring closures, so that the command has pairs to find
CLI_SMILES = [("flex1", "CCCCc1ccccc1O", 5.0), ("tied1", "CC1CCc2ccccc2O1", 6.2),
              ("flex2", "CCCCc1ccc(Cl)cc1O", 5.4), ("tied2", "CC1CCc2ccc(Cl)cc2O1", 6.0)]


def _cli_set(path):
    with Chem.SDWriter(path) as w:
        for k, (name, smi, y) in enumerate(CLI_SMILES):
            m = Chem.AddHs(Chem.MolFromSmiles(smi))
            p = AllChem.ETKDGv3()
            p.randomSeed = k + 1
            AllChem.EmbedMolecule(m, p)
            m.SetProp("_Name", name)
            m.SetProp("activity", f"{y:.2f}")
            w.write(m)
    return path


def test_the_command_finds_the_pairs_and_scores_them(tmp_path, capsys):
    from cmfqsar.cli import main
    data = _cli_set(os.path.join(str(tmp_path), "set.sdf"))
    model = os.path.join(str(tmp_path), "m.cmf")
    main(["build", data, "--activity", "activity", "--fields", "electrostatic",
          "--alphas", "0.5", "--lams", "0.1", "--save", model])
    csv = os.path.join(str(tmp_path), "pairs.csv")
    capsys.readouterr()
    main(["restriction", model, "--molecules", data, "--activity", "activity",
          "--conformers", "2", "--spacing", "1.5", "--csv", csv])
    text = capsys.readouterr().out
    # flex1 pairs with both rings, since a chlorine is inside the two atoms the rule allows
    assert "3 pairs" in text
    assert "flex1 -> tied1" in text and "of the relevant field" in text
    assert "over 3 pairs" in text, "the correlation is printed once there are three"
    assert os.path.exists(csv)
    rows = open(csv, encoding="utf-8").read().strip().splitlines()
    assert len(rows) == 4, "a header and one row per pair"
    assert "intermittent field removed" in rows[0]


def test_the_command_says_so_when_there_is_no_pair(tmp_path):
    from cmfqsar.cli import main
    data = _cli_set(os.path.join(str(tmp_path), "set.sdf"))
    model = os.path.join(str(tmp_path), "m.cmf")
    main(["build", data, "--activity", "activity", "--fields", "electrostatic",
          "--alphas", "0.5", "--lams", "0.1", "--save", model])
    one = os.path.join(str(tmp_path), "one.sdf")
    with Chem.SDWriter(one) as w:
        m = Chem.AddHs(Chem.MolFromSmiles(FLEXIBLE))
        AllChem.EmbedMolecule(m, randomSeed=1)
        m.SetProp("_Name", "alone")
        m.SetProp("activity", "5.0")
        w.write(m)
    with pytest.raises(SystemExit, match="no pair"):
        main(["restriction", model, "--molecules", one, "--activity", "activity"])
