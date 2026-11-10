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


# --------------------------------------------------------------------------- #
# part 8: quantum charges
# --------------------------------------------------------------------------- #

def _write_quantum(path, mols, shift=0.0, skip=()):
    """An SD file like the one the xtb stage writes: the same acids with three numbers on
    every atom.  Returns the charge given to every atom, one array per acid."""
    from rdkit import Chem

    charges = []
    writer = Chem.SDWriter(path)
    for k, m in enumerate(mols):
        s = Chem.Mol(m)
        q = np.linspace(-0.5, 0.5, s.GetNumAtoms()) + 0.01 * k
        charges.append(q)
        for a in s.GetAtoms():
            a.SetDoubleProp(run_band.QUANTUM_CHARGE, float(q[a.GetIdx()]))
            for p in run_band.QUANTUM_FUKUI:
                a.SetDoubleProp(p, 0.001 * a.GetIdx())
        for p in (run_band.QUANTUM_CHARGE,) + tuple(run_band.QUANTUM_FUKUI):
            Chem.CreateAtomDoublePropertyList(s, p)
        if shift:
            pos = s.GetConformer().GetPositions() + np.array([shift, 0.0, 0.0])
            for i in range(s.GetNumAtoms()):
                s.GetConformer().SetAtomPosition(i, [float(x) for x in pos[i]])
        if m.GetProp("_Name") not in skip:
            writer.write(s)
    writer.close()
    return charges


def test_quantum_charges_are_set_on_copies_of_the_same_atoms(tmp_path):
    from cmfqsar.fields import FieldSet, build_fields

    mols, _targets, _carbon, _oxygen = _acids_with_set_bond_lengths([1.206, 1.212, 1.218])
    names = [m.GetProp("_Name") for m in mols]
    path = str(tmp_path / "quantum.sdf")
    charges = _write_quantum(path, mols)
    out = run_band.with_quantum_charges(mols, names, path)
    for m, c, q in zip(mols, out, charges):
        assert c is not m and not m.GetAtomWithIdx(0).HasProp("partial_charge")
        got = [a.GetDoubleProp("partial_charge") for a in c.GetAtoms()]
        assert np.allclose(got, q, atol=1e-6)
        assert c.GetAtomWithIdx(3).GetDoubleProp("fukui_plus") == pytest.approx(0.003)
        assert np.allclose(c.GetConformer().GetPositions(), m.GetConformer().GetPositions())
    # a field set that reads its charges from the file builds its fields on these numbers
    built = build_fields(out[1], FieldSet.create(physchem=("electrostatic",),
                                                 charge_model="file"), name=names[1])
    assert np.allclose(built.charges, charges[1], atol=1e-6)


def test_quantum_charges_of_another_geometry_are_refused(tmp_path):
    mols, _targets, _carbon, _oxygen = _acids_with_set_bond_lengths([1.206, 1.212])
    names = [m.GetProp("_Name") for m in mols]
    with pytest.raises(SystemExit, match="--xtb"):
        run_band.with_quantum_charges(mols, names, str(tmp_path / "not_there.sdf"))
    moved = str(tmp_path / "moved.sdf")
    _write_quantum(moved, mols, shift=0.05)
    with pytest.raises(SystemExit, match="differs by 0.050 A"):
        run_band.with_quantum_charges(mols, names, moved)
    # a shift below the tolerance is the rounding of the file, and is accepted
    assert len(run_band.with_quantum_charges(mols, names, moved, tolerance=0.1)) == 2
    short = str(tmp_path / "short.sdf")
    _write_quantum(short, mols, skip=("acid1",))
    with pytest.raises(SystemExit, match="no structure named acid1"):
        run_band.with_quantum_charges(mols, names, short)


def test_the_four_atoms_of_the_carboxyl_group():
    mols, targets, carbon, oxygen = _acids_with_set_bond_lengths([1.206, 1.212])
    found = run_band.carboxyl_atoms(mols[0], carbon)
    assert found[:2] == (carbon, oxygen)
    assert [mols[0].GetAtomWithIdx(i).GetSymbol() for i in found] == ["C", "O", "O", "H"]
    assert mols[0].GetAtomWithIdx(found[2]).GetDegree() == 2
    assert mols[0].GetBondBetweenAtoms(found[2], found[3]) is not None
    ring = next(a.GetIdx() for a in mols[0].GetAtoms() if a.GetIsAromatic())
    assert run_band.carboxyl_atoms(mols[0], ring) is None
    assert len(run_band.CARBOXYL_ATOMS) == 4
    # the charges of the four atoms, in that order, one line per sample
    charges = [np.arange(m.GetNumAtoms()) + 100.0 * k for k, m in enumerate(mols)]
    got = run_band.carboxyl_charges(mols, targets, charges)
    assert got.shape == (2, 4)
    for r in range(2):
        k = int(targets.molecule[r])
        assert np.array_equal(got[r], np.array(found, float) + 100.0 * k)
    # a target on an atom that is not the carbon of a carboxyl group gives no charges
    per = [np.where(np.arange(m.GetNumAtoms()) == ring, 1.0, np.nan) for m in mols]
    other = AtomTargets.from_arrays(per, name="co_stretch")
    assert np.all(np.isnan(run_band.carboxyl_charges(mols, other, charges)))


def _linear_series(n=40, seed=0, noise=0.2):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(n, 3))
    y = 1740.0 + x @ np.array([3.0, -2.0, 1.0]) + rng.normal(0, noise, n)
    return x, AtomTargets.from_arrays([[v] for v in y], name="co_stretch")


def test_a_regression_on_columns_predicts_out_of_fold():
    x, targets = _linear_series()
    y = np.asarray(targets.y, float)
    res = run_band.regression_reference(x, targets, folds=5, seed=0, name="made up")
    assert res.name == "made up" and res.q2 > 0.99
    # every prediction comes from the equation fitted without the sample
    for te in targets.folds(5, 0):
        tr = np.setdiff1d(np.arange(len(y)), te)
        coef = np.linalg.lstsq(np.c_[np.ones(len(tr)), x[tr]], y[tr], rcond=None)[0]
        assert np.allclose(res.predicted[te], np.c_[np.ones(len(te)), x[te]] @ coef)
    # one column is a simple linear regression, and it explains less than the three
    one = run_band.regression_reference(x[:, 0], targets, folds=5, seed=0)
    assert 0.3 < one.q2 < res.q2
    # a sample without its numbers takes the mean of the band
    holes = x.copy()
    holes[4] = np.nan
    gap = run_band.regression_reference(holes, targets, folds=5, seed=0)
    assert gap.predicted[4] == pytest.approx(y.mean())
    assert np.all(np.isfinite(gap.predicted))


def test_the_choice_inside_the_folds_takes_the_kernel_that_predicts():
    x, targets = _linear_series()
    y = np.asarray(targets.y, float)
    z = np.random.RandomState(5).normal(size=(len(y), 3))
    kernels = {"informative": x @ x.T, "noise": z @ z.T}
    res, picks = run_band.nested_choice(kernels, targets, lams=(1e-4, 1e-2), folds=5, seed=0,
                                        inner=3, name="made up")
    assert len(picks) == 5 and all(key == "informative" for key, _lam in picks)
    assert all(lam in (1e-4, 1e-2) for _key, lam in picks)
    assert res.q2 > 0.95 and "inside every training part" in res.note
    # the held-out samples take no part in the choice or in the fit: another band on the
    # samples of the first fold leaves the predictions for these samples as they were
    first = targets.folds(5, 0)[0]
    changed = y.copy()
    changed[first] = 1700.0 + 50.0 * np.arange(len(first))
    other = AtomTargets.from_arrays([[v] for v in changed], name="co_stretch")
    again, _picks = run_band.nested_choice(kernels, other, lams=(1e-4, 1e-2), folds=5, seed=0,
                                           inner=3)
    assert np.allclose(again.predicted[first], res.predicted[first])
    rest = np.setdiff1d(np.arange(len(y)), first)
    assert not np.allclose(again.predicted[rest], res.predicted[rest])


def test_the_largest_errors_and_the_score_without_some_samples():
    targets, _table = _twins()
    y = np.asarray(targets.y, float)
    names = [f"acid{k}" for k in range(len(y))]
    res = []
    for seed in range(3):
        pred = y + np.random.RandomState(seed).normal(0, 0.3, len(y))
        pred[3] += 20.0
        res.append(run_band._as_result("made up", pred, targets, 5))
    top = run_band.largest_errors(res, y, names, n=2)
    assert [t["name"] for t in top][0] == "acid3" and len(top) == 2
    assert top[0]["error"] == pytest.approx(20.0, abs=1.0) and top[0]["share"] > 0.95
    assert top[0]["share"] + top[1]["share"] <= 1.0
    cut, sub = run_band.without_rows(res, targets, [3], folds=5)
    assert len(sub) == len(y) - 1 and len(cut) == 3
    keep = np.delete(np.arange(len(y)), 3)
    # nothing is fitted again: the predictions of the other samples are the same numbers
    assert np.array_equal(cut[0].predicted, np.asarray(res[0].predicted)[keep])
    assert np.array_equal(sub.y, y[keep])
    assert run_band.repeated_q2(cut, sub.y)[0] > 0.98 > run_band.repeated_q2(res, y)[0]


def test_the_difference_between_the_para_acid_and_the_meta_acid():
    targets, table = _twins()
    y = np.asarray(targets.y, float)
    pairs = run_band.twin_pairs(table)
    assert pairs.shape == (12, 2)
    pos, subs = np.array(table["position"]), np.array(table["substituent"])
    assert set(pos[pairs[:, 0]]) == {"meta"} and set(pos[pairs[:, 1]]) == {"para"}
    assert np.array_equal(subs[pairs[:, 0]], subs[pairs[:, 1]])
    exact = run_band.twin_difference([run_band._as_result("exact", y, targets, 5)], targets,
                                     table)
    assert exact["n"] == 12 and exact["q2"] == pytest.approx(1.0)
    assert exact["r"] == pytest.approx(1.0) and exact["error"] == pytest.approx(0.0)
    assert exact["spread"] == pytest.approx(np.std(y[pairs[:, 1]] - y[pairs[:, 0]]))
    # a model that gives both acids of a substituent the mean of the two bands predicts the
    # band well and the difference not at all
    mean = y.copy()
    mean[pairs[:, 0]] = mean[pairs[:, 1]] = 0.5 * (y[pairs[:, 0]] + y[pairs[:, 1]])
    flat = [run_band._as_result("pair mean", mean, targets, 5)]
    d = run_band.twin_difference(flat, targets, table)
    assert run_band.repeated_q2(flat, y)[0] > 0.9
    assert d["q2"] <= 0.0 and d["r"] is None
    # fewer than three pairs give no numbers
    few = {k: v[:3] for k, v in table.items()}
    none = run_band.twin_difference([run_band._as_result("exact", y[:3], targets.subset([0, 1, 2]),
                                                         5)], targets.subset([0, 1, 2]), few)
    assert none["n"] == 0 and none["q2"] is None


def test_the_error_by_position():
    targets, table = _twins()
    y = np.asarray(targets.y, float)
    pos = np.array(table["position"])
    pred = y + np.where(pos == "meta", 2.0, 0.0)
    e = run_band.position_errors([run_band._as_result("made up", pred, targets, 5)], y, table)
    assert e["meta"] == pytest.approx(2.0) and e["para"] == pytest.approx(0.0)
    only_para = {k: [v for v, p in zip(vals, pos) if p != "meta"] for k, vals in table.items()}
    rows = np.where(pos != "meta")[0]
    sub = targets.subset(rows)
    e = run_band.position_errors([run_band._as_result("made up", pred[rows], sub, 5)], y[rows],
                                 only_para)
    assert e["meta"] is None and e["para"] == pytest.approx(0.0)


def test_the_xtb_stage_writes_charges_that_the_benchmark_accepts(tmp_path):
    """Stage 3 of the example builder on two acids: the same atoms at the same places, with a
    charge on every atom whose sum is the charge of the acid."""
    from rdkit import Chem

    from cmfqsar import xtb

    try:
        xtb.find_xtb(None)
    except xtb.XtbError:
        pytest.skip("xtb is not installed here")
    mb = _examples()
    if not os.path.exists(mb.STRUCTURES):
        pytest.skip("the start structures of the benzoic acids are not there")
    out = str(tmp_path / "quantum.sdf")
    said = []
    failed = mb.run_xtb(mb.STRUCTURES, out, workdir=str(tmp_path / "work"), jobs=2, limit=2,
                        log=said.append)
    assert failed == 0 and "wrote 2 acids" in said[-1]
    mols = [m for m in Chem.SDMolSupplier(mb.STRUCTURES, removeHs=False)][:2]
    names = [m.GetProp("_Name") for m in mols]
    got = run_band.with_quantum_charges(mols, names, out)
    for c in got:
        q = [a.GetDoubleProp("partial_charge") for a in c.GetAtoms()]
        assert abs(sum(q)) < 1e-4 and max(abs(v) for v in q) > 0.2
        assert all(a.HasProp("fukui_plus") and a.HasProp("fukui_minus") for a in c.GetAtoms())
    # a second build reads the finished jobs and writes the same bytes
    again = str(tmp_path / "again.sdf")
    assert mb.run_xtb(mb.STRUCTURES, again, workdir=str(tmp_path / "work"), jobs=2, limit=2,
                      log=said.append) == 0
    with open(out, "rb") as a, open(again, "rb") as b:
        assert a.read() == b.read()


def test_the_report_adds_the_quantum_charge_part_when_its_result_is_there(tmp_path):
    import shutil

    stored = [_stored(n) for n in ("band.json", "indicator.json", "charges.json")]
    if not all(os.path.exists(p) for p in stored):
        pytest.skip("the stored results of the band benchmark are not there")
    for p in stored[:2]:
        shutil.copy(p, str(tmp_path / os.path.basename(p)))
    without = run_band.report(str(tmp_path))
    assert "Quantum charges" not in without
    shutil.copy(stored[2], str(tmp_path / "charges.json"))
    text = run_band.report(str(tmp_path))
    with open(str(tmp_path / "table.md"), encoding="utf-8") as fh:
        assert fh.read() == text
    assert text.startswith(without)
    assert "## Quantum charges: the electrostatic field from GFN2-xTB charges" in text
    assert "width and ridge chosen inside the folds" in text
    assert "**Reading of the quantum charges.**" in text and "**Answer.**" in text
    assert "Hammett line" not in text and "one line" not in text
    # the stored table is the one this report writes
    with open(_stored("table.md"), encoding="utf-8") as fh:
        assert fh.read() == text
