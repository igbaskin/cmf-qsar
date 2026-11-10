"""The band benchmark's own arithmetic (plan item SP2): the linear regressions on the Hammett
constant and their use as a reference, on a made-up series where the band is a known function
of sigma."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "benchmarks"))

import run_band  # noqa: E402
from cmfqsar.atomtargets import AtomTargets  # noqa: E402


def _series(n_each=20, seed=0):
    rng = np.random.RandomState(seed)
    sigma_m = rng.uniform(-0.2, 0.8, n_each)
    sigma_p = rng.uniform(-0.8, 0.8, n_each)
    # the band follows sigma with one slope for meta and another for para, plus noise
    y = np.concatenate([[1740.0], 1740.0 + 12.0 * sigma_m, 1740.0 + 18.0 * sigma_p])
    y = y + rng.normal(0, 0.5, len(y))
    per = [[v] for v in y]
    targets = AtomTargets.from_arrays(per, name="co_stretch")
    table = {"position": ["none"] + ["meta"] * n_each + ["para"] * n_each,
             "sigma": [0.0] + list(sigma_m) + list(sigma_p),
             "substituent": [f"s{k}" for k in range(len(y))],
             "anti": [False] * len(y), "imaginary": [0] * len(y)}
    return targets, table


def test_hammett_lines_recover_the_slopes():
    targets, table = _series()
    lines = run_band.hammett_lines(targets, table)
    assert set(lines) == {"all", "meta", "para"}
    assert lines["meta"]["slope"] == pytest.approx(12.0, abs=1.0)
    assert lines["para"]["slope"] == pytest.approx(18.0, abs=1.0)
    assert lines["meta"]["n"] == 21 and lines["para"]["n"] == 21 and lines["all"]["n"] == 41
    assert lines["para"]["r"] > 0.98


def test_two_lines_predict_better_than_one_when_the_slopes_differ():
    targets, table = _series()
    one = run_band.hammett_reference(targets, table, folds=5, seed=0, two_lines=False)
    two = run_band.hammett_reference(targets, table, folds=5, seed=0, two_lines=True)
    assert np.all(np.isfinite(one.predicted)) and np.all(np.isfinite(two.predicted))
    assert two.q2 > one.q2 > 0.5
    assert two.n_samples == len(targets) and two.folds == 5
    assert "fitted in fold" in two.note


def test_q2_and_rmse_helpers():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert run_band.q2_of(y, y) == pytest.approx(1.0)
    assert run_band.q2_of(y, np.full(4, y.mean())) == pytest.approx(0.0)
    assert run_band.rmse_of(y, y + 2.0) == pytest.approx(2.0)


def test_position_statistics():
    assert run_band.para_over_meta({"para": 3.0, "meta": 1.0}) == pytest.approx(0.5)
    assert run_band.para_over_meta({"para": -1.0, "meta": 1.0}) == pytest.approx(0.0)
    assert run_band.para_over_meta({"para": 0.0, "meta": 0.0}) == 0.0
    v = {"ipso": 5.0, "ortho": 2.0, "meta": 0.0, "para": 2.0}
    assert run_band.alternation(v) == pytest.approx(1.0)


def test_ring_positions_are_found_on_an_acid_with_explicit_hydrogens():
    """The carbonyl oxygen is told from the hydroxyl oxygen by its one neighbour, so the test
    works on a molecule whose hydrogens are atoms -- the form the ORCA stage writes."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.localalign import posed_environments

    mols, per = [], []
    for smiles in ("OC(=O)c1ccccc1", "OC(=O)c1ccc(cc1)[N+](=O)[O-]", "OC(=O)c1cc(ccc1)Cl"):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(m, randomSeed=7) == 0
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        carbon = m.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))[0]
        v = np.full(m.GetNumAtoms(), np.nan)
        v[carbon] = 1740.0 + len(mols)
        mols.append(m)
        per.append(v)
    targets = AtomTargets.from_arrays(per, name="co_stretch")
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=f"m{k}") for k, m in enumerate(mols)]
    envs = posed_environments(fields, targets, rho=0.1)
    means, four, counts = run_band.ring_points(mols, targets, envs)
    assert sorted(four) == ["ipso", "meta", "ortho", "para"]
    assert counts["ipso"] == 3 and counts["para"] == 3
    assert counts["ortho (carbonyl side)"] == 3 and counts["ortho (hydroxyl side)"] == 3
    # the frame at the carbonyl carbon puts the ring on one axis: para lies beyond ipso
    d = {k: float(np.linalg.norm(v)) for k, v in four.items()}
    assert d["ipso"] < d["ortho"] < d["meta"] < d["para"]
    assert d["ipso"] == pytest.approx(1.48, abs=0.08) and d["para"] == pytest.approx(4.3, abs=0.2)


def _examples():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "examples")
    if path not in sys.path:
        sys.path.insert(0, path)
    import make_band_examples
    return make_band_examples


def test_the_sf5_group_of_a_start_structure_is_a_regular_octahedron():
    mb = _examples()
    mol = mb.build("para", "S(F)(F)(F)(F)F", n_embed=4)
    assert "SF5" in mol.GetProp("force_field")
    r = mol.GetConformer().GetPositions()
    s = next(a for a in mol.GetAtoms() if a.GetSymbol() == "S")
    fluorines = [n.GetIdx() for n in s.GetNeighbors() if n.GetSymbol() == "F"]
    carbon = next(n.GetIdx() for n in s.GetNeighbors() if n.GetSymbol() == "C")
    assert len(fluorines) == 5
    lengths = [np.linalg.norm(r[f] - r[s.GetIdx()]) for f in fluorines]
    assert np.allclose(lengths, mb.SF5_BONDS["S-F"], atol=1e-3)
    assert np.linalg.norm(r[carbon] - r[s.GetIdx()]) == pytest.approx(mb.SF5_BONDS["C-S"], abs=1e-3)
    unit = [(r[x] - r[s.GetIdx()]) / np.linalg.norm(r[x] - r[s.GetIdx()])
            for x in fluorines + [carbon]]
    cosines = sorted(float(unit[a] @ unit[b]) for a in range(6) for b in range(a + 1, 6))
    # six neighbours of an octahedron: three pairs opposite, twelve pairs at a right angle
    assert np.allclose(cosines[:3], -1.0, atol=1e-3) and np.allclose(cosines[3:], 0.0, atol=1e-3)
    assert mb.wrong_bonds(mol) == []
    # no fluorine closer to another atom than a contact allows
    others = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in fluorines + [s.GetIdx()]]
    nearest = min(np.linalg.norm(r[f] - r[o]) for f in fluorines for o in others)
    assert nearest > 2.0


def test_a_structure_without_sf5_is_not_touched():
    mb = _examples()
    mol = mb.build("meta", "C(F)(F)F", n_embed=4)
    before = mol.GetConformer().GetPositions().copy()
    assert mb.regular_sf5(mol) == 0
    assert np.array_equal(before, mol.GetConformer().GetPositions())
    assert mol.GetProp("force_field") == "MMFF94"


def _acids_with_set_bond_lengths(lengths, seed=0):
    """Benzoic acid copies whose C=O bond has the given lengths, with a band that falls
    linearly with the length plus a little noise."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D

    base = Chem.AddHs(Chem.MolFromSmiles("OC(=O)c1ccccc1"))
    assert AllChem.EmbedMolecule(base, randomSeed=7) == 0
    AllChem.MMFFOptimizeMolecule(base)
    Chem.SanitizeMol(base)
    carbon, oxygen, _ = base.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))
    rng = np.random.RandomState(seed)
    mols, per = [], []
    for k, d in enumerate(lengths):
        m = Chem.Mol(base)
        m.SetProp("_Name", f"acid{k}")
        p = m.GetConformer().GetPositions()
        u = (p[oxygen] - p[carbon]) / np.linalg.norm(p[oxygen] - p[carbon])
        new = p[carbon] + d * u
        m.GetConformer().SetAtomPosition(oxygen, Point3D(*[float(x) for x in new]))
        v = np.full(m.GetNumAtoms(), np.nan)
        v[carbon] = 1740.0 - 3500.0 * (d - 1.21) + rng.normal(0, 0.2)
        mols.append(m)
        per.append(v)
    return mols, AtomTargets.from_arrays(per, name="co_stretch"), carbon, oxygen


def test_the_bond_length_line_recovers_a_band_that_follows_the_length():
    lengths = np.linspace(1.205, 1.220, 24)
    mols, targets, _c, _o = _acids_with_set_bond_lengths(lengths)
    found = run_band.carbonyl_lengths(mols, targets)
    order = np.argsort(np.asarray(targets.molecule))
    assert np.allclose(found[order], lengths, atol=1e-6)
    res = run_band.bond_length_reference(mols, targets, folds=4, seed=0)
    assert res.q2 > 0.99 and np.all(np.isfinite(res.predicted))
    assert "r -" in res.note and "1.2050 to 1.2200" in res.note


def test_the_bond_length_line_predicts_the_mean_when_the_lengths_are_all_equal():
    mols, targets, _c, _o = _acids_with_set_bond_lengths([1.21] * 12)
    res = run_band.bond_length_reference(mols, targets, folds=4, seed=0)
    assert np.all(np.isfinite(res.predicted))
    assert res.q2 <= 0.05


def test_start_geometries_replace_the_coordinates_and_keep_the_bands(tmp_path):
    from rdkit import Chem

    mols, targets, carbon, oxygen = _acids_with_set_bond_lengths([1.206, 1.212, 1.218])
    names = [m.GetProp("_Name") for m in mols]
    path = str(tmp_path / "start.sdf")
    writer = Chem.SDWriter(path)
    for m in reversed(mols):                       # another order in the file: matched by name
        s = Chem.Mol(m)
        p = s.GetConformer().GetPositions() + np.array([0.5, 0.0, 0.0])
        for i in range(s.GetNumAtoms()):
            s.GetConformer().SetAtomPosition(i, [float(x) for x in p[i]])
        writer.write(s)
    writer.close()
    out = run_band.with_start_geometries(mols, names, path)
    for m, c in zip(mols, out):
        assert c is not m and c.GetProp("_Name") == m.GetProp("_Name")
        shift = c.GetConformer().GetPositions() - m.GetConformer().GetPositions()
        assert np.allclose(shift, [0.5, 0.0, 0.0], atol=1e-3)
        assert [a.GetIdx() for a in c.GetAtoms()] == [a.GetIdx() for a in m.GetAtoms()]
    # the given molecules are not changed, and a name that is not in the file stops the run
    assert np.allclose(run_band.carbonyl_lengths(mols, targets)[np.argsort(targets.molecule)],
                       [1.206, 1.212, 1.218], atol=1e-6)
    with pytest.raises(SystemExit):
        run_band.with_start_geometries(mols, ["acid0", "acid1", "no such acid"], path)


def test_the_report_writes_the_table_it_returns(tmp_path):
    """`run_all.py` rebuilds a table by calling `report`; a report that only returned its text
    would leave a stale table unnoticed."""
    import shutil

    stored = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "benchmarks", "results", "band", "band.json")
    if not os.path.exists(stored):
        pytest.skip("the stored result of the band benchmark is not there")
    shutil.copy(stored, str(tmp_path / "band.json"))
    text = run_band.report(str(tmp_path))
    with open(str(tmp_path / "table.md"), encoding="utf-8") as fh:
        assert fh.read() == text
    assert "linear regression on the Hammett constant, all acids" in text
    assert "Hammett line" not in text and "one line" not in text
    other = str(tmp_path / "other.md")
    assert run_band.report(str(tmp_path), other) == text and os.path.exists(other)


# --------------------------------------------------------------------------- #
# part 7: the indicator fields
# --------------------------------------------------------------------------- #

def _twins(n=12, seed=0):
    """A made-up series with every substituent twice, meta and para, and one acid without a
    substituent.  The band of an acid follows the Hammett constant of its substituent."""
    rng = np.random.RandomState(seed)
    sigma = rng.uniform(-0.6, 0.8, n)
    y = np.concatenate([[1740.0], 1740.0 + 10.0 * sigma, 1740.0 + 14.0 * sigma])
    y = y + rng.normal(0, 0.3, len(y))
    targets = AtomTargets.from_arrays([[v] for v in y], name="co_stretch")
    names = [f"s{k}" for k in range(n)]
    table = {"position": ["none"] + ["meta"] * n + ["para"] * n,
             "sigma": [0.0] + list(sigma) + list(sigma),
             "substituent": ["H"] + names + names}
    return targets, table


def test_folds_over_substituents_hold_out_both_acids_of_a_substituent():
    targets, table = _twins()
    grouped = run_band.substituent_targets(targets, table)
    assert len(grouped) == len(targets) and grouped.n_mol == 13
    assert np.array_equal(grouped.y, targets.y) and np.array_equal(grouped.atom, targets.atom)
    subs = np.array(table["substituent"])
    seen = []
    for fold in grouped.folds(5, seed=0):
        held = set(subs[fold])
        # no substituent of the test rows is left in the training rows
        assert not held & set(np.delete(subs, fold))
        seen += list(fold)
    assert sorted(seen) == list(range(len(targets)))
    # the folds over acids do separate the two acids of a substituent
    split = any(set(subs[f]) & set(np.delete(subs, f)) for f in targets.folds(5, seed=0))
    assert split


def test_twin_correlation_reads_the_pairs_of_a_substituent():
    targets, table = _twins()
    t = run_band.twin_correlation(targets, table)
    assert t["n"] == 12
    assert t["r_sigma"] == pytest.approx(1.0)
    assert 0.9 < t["r_band"] < 1.0
    only_meta = {k: v[:13] for k, v in table.items()}
    few = run_band.twin_correlation(AtomTargets.from_arrays([[v] for v in targets.y[:13]]),
                                    only_meta)
    assert few["n"] == 0 and np.isnan(few["r_band"])


def _results(targets, noise, seeds):
    y = np.asarray(targets.y, float)
    return [run_band._as_result("made up", y + np.random.RandomState(s).normal(0, noise, len(y)),
                                targets, 5) for s in seeds]


def test_the_q2_of_repeats_is_their_mean():
    targets, _table = _twins()
    y = np.asarray(targets.y, float)
    res = _results(targets, 1.0, range(4))
    q, sd, err = run_band.repeated_q2(res, y)
    each = [run_band.q2_of(y, r.predicted) for r in res]
    assert q == pytest.approx(np.mean(each)) and sd == pytest.approx(np.std(each, ddof=1))
    e = run_band.squared_errors(res, y)
    assert e.shape == y.shape and err == pytest.approx(np.sqrt(e.mean()))
    # the mean q2 is one minus the summed mean squared errors over the total sum of squares
    assert q == pytest.approx(1.0 - e.sum() / np.sum((y - y.mean()) ** 2))
    one = run_band.repeated_q2(res[:1], y)
    assert one[0] == pytest.approx(each[0]) and one[1] == 0.0


def test_the_paired_interval_of_repeats():
    targets, table = _twins(n=30)
    y = np.asarray(targets.y, float)
    good, poor = _results(targets, 0.5, range(5)), _results(targets, 3.0, range(10, 15))
    d = run_band.paired_repeats(good, poor, targets, runs=300, seed=0)
    assert d[0] == pytest.approx(run_band.repeated_q2(good, y)[0]
                                 - run_band.repeated_q2(poor, y)[0])
    assert 0 < d[1] < d[0] < d[2]
    back = run_band.paired_repeats(poor, good, targets, runs=300, seed=0)
    assert back[0] == pytest.approx(-d[0]) and back[2] < 0
    same = run_band.paired_repeats(good, good, targets, runs=50, seed=0)
    assert same == (0.0, 0.0, 0.0)
    # the units of the bootstrap are the substituents when the copy of the targets is given
    grouped = run_band.substituent_targets(targets, table)
    g = run_band.paired_repeats(good, poor, grouped, runs=300, seed=0)
    assert g[0] == pytest.approx(d[0]) and (g[1], g[2]) != (d[1], d[2])


def test_kernel_agreement():
    rng = np.random.RandomState(0)
    x = rng.normal(size=(15, 4))
    k = x @ x.T
    assert run_band.kernel_agreement(k, 3.0 * k) == pytest.approx(1.0)
    # a constant added to a kernel is removed by the centring
    assert run_band.kernel_agreement(k, k + 5.0) == pytest.approx(1.0)
    z = rng.normal(size=(15, 4))
    assert abs(run_band.kernel_agreement(k, z @ z.T)) < 0.5


def test_the_first_atom_of_a_substituent_is_found_with_its_type_and_place():
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import FieldSet, build_fields
    from cmfqsar.localalign import posed_environments

    mols, per = [], []
    for smiles in ("OC(=O)c1ccccc1", "OC(=O)c1ccc(cc1)[N+](=O)[O-]", "OC(=O)c1cc(ccc1)Cl",
                   "OC(=O)c1ccc(cc1)OC"):
        m = Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(m, randomSeed=7) == 0
        AllChem.MMFFOptimizeMolecule(m)
        Chem.SanitizeMol(m)
        carbon = m.GetSubstructMatch(Chem.MolFromSmarts("[CX3](=O)[OX2H1]"))[0]
        v = np.full(m.GetNumAtoms(), np.nan)
        v[carbon] = 1740.0 + len(mols)
        mols.append(m)
        per.append(v)
    targets = AtomTargets.from_arrays(per, name="co_stretch")
    fieldset = FieldSet.create()
    fields = [build_fields(m, fieldset, name=f"m{k}") for k, m in enumerate(mols)]
    envs = posed_environments(fields, targets, rho=0.01)
    first = run_band.first_atoms(mols, targets, envs)
    assert first[0] is None
    assert first[1][:2] == ("N.pl3", "para") and first[3][:2] == ("O.3", "para")
    assert first[2][0] == "Cl" and first[2][1] in ("meta (carbonyl side)", "meta (hydroxyl side)")
    assert np.linalg.norm(first[1][2]) == pytest.approx(5.8, abs=0.2)
    assert np.linalg.norm(first[2][2]) == pytest.approx(5.2, abs=0.3)


def _stored(name):
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "benchmarks", "results", "band", name)


def test_the_report_adds_the_indicator_part_when_its_result_is_there(tmp_path):
    import shutil

    if not (os.path.exists(_stored("band.json")) and os.path.exists(_stored("indicator.json"))):
        pytest.skip("the stored results of the band benchmark are not there")
    shutil.copy(_stored("band.json"), str(tmp_path / "band.json"))
    without = run_band.report(str(tmp_path))
    assert "Indicator fields" not in without
    shutil.copy(_stored("indicator.json"), str(tmp_path / "indicator.json"))
    text = run_band.report(str(tmp_path))
    with open(str(tmp_path / "table.md"), encoding="utf-8") as fh:
        assert fh.read() == text
    assert text.startswith(without)
    assert "## Indicator fields: one field per atom type" in text
    assert "whole substituents held out" in text and "**Reading of the indicator fields.**" in text
    assert "Hammett line" not in text and "one line" not in text
