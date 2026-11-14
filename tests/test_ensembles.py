"""Ensembles of structures as the normal input (plan item MS5).

A sample may be several structures with weights, and the structures may have different atoms.
The tests check the packing and the split, that the packed field is the mean field the older
code built, that the reader keeps every record, that the command line builds and predicts with
such samples, and -- on three electrolytes of item BT5 -- that the general path gives exactly
the kernel of BT5's own code.
"""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.dataio import load_dataset  # noqa: E402
from cmfqsar.ensembles import (EnsembleError, ensemble_molecule, is_packed,  # noqa: E402
                               member_weights, members_of, n_members, write_ensembles)
from cmfqsar.fields import (AtomTable, FieldSet, build_fields, combine_members,  # noqa: E402
                            ensemble_fields)

FIELDS = ["electrostatic", "steric", "hydrophobic"]


def _mol(smiles, seed=1, name=""):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    assert AllChem.EmbedMolecule(m, p) == 0
    if name:
        m.SetProp("_Name", name)
    return m


def _fs(**kw):
    return FieldSet.create(physchem=FIELDS, alpha=0.5, charge_model="gasteiger", **kw)


def _same_fields(a, b):
    assert a.coords.shape == b.coords.shape
    assert np.array_equal(a.coords, b.coords)
    assert np.array_equal(a.weights, b.weights)
    assert np.array_equal(a.radii, b.radii)
    assert list(a.symbols) == list(b.symbols)


# ------------------------------------------------------------------ packing and splitting


def test_members_with_the_same_atoms_stay_conformations():
    a, b = _mol("CCO", 1), _mol("CCO", 7)
    m = ensemble_molecule([a, b], weights=[3.0, 1.0])
    assert not is_packed(m)
    assert m.GetNumConformers() == 2 and n_members(m) == 2
    fs = _fs(conformers="ensemble", conformer_weighting="given")
    got = build_fields(m, fs)
    want = ensemble_fields(m, fs)
    _same_fields(got, want)
    assert np.allclose(got.conformer_weights, [0.75, 0.25])


def test_members_with_their_own_atoms_are_packed_and_split_back_exactly():
    members = [_mol("CCO", 1), _mol("CO", 2), _mol("CCCO", 3)]
    for m in members:
        for at in m.GetAtoms():
            at.SetDoubleProp("tag", float(at.GetIdx()))
    packed = ensemble_molecule(members, weights=[0.5, 0.3, 0.2], name="mixture")
    assert is_packed(packed)
    assert n_members(packed) == 3
    assert packed.GetProp("_Name") == "mixture"
    back = members_of(packed)
    for got, want in zip(back, members):
        assert got.GetNumAtoms() == want.GetNumAtoms()
        assert [a.GetAtomicNum() for a in got.GetAtoms()] == \
            [a.GetAtomicNum() for a in want.GetAtoms()]
        assert np.array_equal(got.GetConformer().GetPositions(),
                              want.GetConformer().GetPositions())
        assert [a.GetDoubleProp("tag") for a in got.GetAtoms()] == \
            [a.GetDoubleProp("tag") for a in want.GetAtoms()]


def test_the_packed_field_is_the_mean_field_of_its_members():
    members = [_mol("CCO", 1), _mol("CO", 2), _mol("CCCO", 3)]
    raw = [0.5, 0.3, 0.2]
    packed = ensemble_molecule(members, weights=raw)
    fs = _fs(conformers="ensemble", conformer_weighting="given")
    got = build_fields(packed, fs)
    parts = [build_fields(m, fs) for m in members]
    want = combine_members(parts, raw, kind="members",
                           atom_maps=[list(range(m.GetNumAtoms())) if i == 0
                                      else [-1] * m.GetNumAtoms()
                                      for i, m in enumerate(members)])
    _same_fields(got, want)
    assert got.n_conformers == 3
    assert np.allclose(got.conformer_weights, raw)


def test_a_single_structure_is_the_special_case():
    members = [_mol("CCO", 1), _mol("CO", 2)]
    packed = ensemble_molecule(members, weights=[0.1, 0.9])
    got = build_fields(packed, _fs())                   # conformers="single", the default
    _same_fields(got, build_fields(members[0], _fs()))
    assert not got.is_ensemble


def test_the_weighting_rules_of_a_packed_ensemble():
    members = [_mol("CCO", 1), _mol("CO", 2), _mol("CCCO", 3)]
    packed = ensemble_molecule(members, weights=[0.2, 0.5, 0.3])
    assert np.allclose(member_weights(packed, _fs(conformers="ensemble")), [1 / 3] * 3)
    top = member_weights(packed, _fs(conformers="ensemble", conformer_weighting="given",
                                     max_conformers=2))
    assert np.allclose(top, [0.0, 0.5 / 0.8, 0.3 / 0.8])
    with pytest.raises(EnsembleError, match="energy per member"):
        member_weights(packed, _fs(conformers="ensemble", conformer_weighting="boltzmann"))
    plain = ensemble_molecule(members)
    with pytest.raises(EnsembleError, match="carries no member weights"):
        build_fields(plain, _fs(conformers="ensemble", conformer_weighting="given"))


def test_bad_inputs_are_refused():
    a, b = _mol("CCO", 1), _mol("CO", 2)
    with pytest.raises(EnsembleError, match="at least one"):
        ensemble_molecule([])
    with pytest.raises(EnsembleError, match="2 members and 1 weights"):
        ensemble_molecule([a, b], weights=[1.0])
    with pytest.raises(EnsembleError, match="not all zero"):
        ensemble_molecule([a, b], weights=[0.0, 0.0])
    two = Chem.Mol(a)
    two.AddConformer(Chem.Conformer(a.GetConformer()), assignId=True)
    with pytest.raises(EnsembleError, match="2 conformations"):
        ensemble_molecule([two, b])
    with pytest.raises(EnsembleError, match="points outside"):
        ensemble_molecule([a, b], atom_maps=[list(range(a.GetNumAtoms())),
                                             [99] * b.GetNumAtoms()])
    packed = ensemble_molecule([a, b])
    with pytest.raises(EnsembleError, match="cannot be combined"):
        build_fields(packed, _fs(conformers="ensemble", tautomers="score"))


def test_an_atom_map_places_a_per_atom_quantity_and_its_absence_is_reported():
    a, b = _mol("CCO", 1), _mol("CCOC", 2)
    # map the first four heavy atoms of b onto the three of a where they correspond
    amap_b = [-1] * b.GetNumAtoms()
    amap_b[0], amap_b[1], amap_b[2] = 0, 1, 2
    fs = _fs(conformers="ensemble")
    with_map = build_fields(ensemble_molecule([a, b], atom_maps=[list(range(a.GetNumAtoms())),
                                                                   amap_b]), fs)
    without = build_fields(ensemble_molecule([a, b]), fs)
    values = np.ones(with_map.n_atoms)
    assert with_map.per_atom(values).sum() == pytest.approx(a.GetNumAtoms() + 3)
    assert without.per_atom(values).sum() == pytest.approx(a.GetNumAtoms())
    assert without.unmapped_share(values) == pytest.approx(
        b.GetNumAtoms() / (a.GetNumAtoms() + b.GetNumAtoms()))


def test_descriptors_survive_an_ensemble():
    a, b = _mol("CCO", 1), _mol("CCO", 5)
    a.SetProp("logS", "-1.5")
    conf = ensemble_molecule([a, b])
    fs = _fs(conformers="ensemble", descriptors=("logS",))
    assert build_fields(conf, fs).descriptors is not None
    c = _mol("CO", 2)
    packed = ensemble_molecule([a, c])
    got = build_fields(packed, fs)
    assert got.descriptors is not None and np.allclose(got.descriptors, [-1.5])


# ------------------------------------------------------------------------------- the reader


def _mixtures(path, weights=True):
    samples = [[_mol("CCO", 1), _mol("CO", 2)], [_mol("CCCO", 3), _mol("CCO", 4), _mol("O", 5)],
               [_mol("CC(C)O", 6)], [_mol("CO", 7), _mol("CCCCO", 8)],
               [_mol("OCCO", 9), _mol("CO", 10)], [_mol("CCOCC", 11), _mol("CCO", 12)]]
    w = [[0.6, 0.4], [0.5, 0.3, 0.2], [1.0], [0.7, 0.3], [0.5, 0.5], [0.8, 0.2]]
    props = [{"activity": f"{4.0 + 0.5 * k:.2f}"} for k in range(len(samples))]
    n = write_ensembles(path, samples, [f"s{k}" for k in range(len(samples))],
                        weights=w if weights else None, sample_props=props)
    return samples, w, n


def test_the_reader_keeps_every_record_and_reads_the_weights(tmp_path):
    path = str(tmp_path / "mix.sdf")
    samples, w, n = _mixtures(path)
    assert n == sum(len(s) for s in samples)
    ds = load_dataset(path, activity_prop="activity", ensemble_by="sample",
                      member_weight="member_weight")
    assert list(ds.names) == [f"s{k}" for k in range(len(samples))]
    assert [n_members(m) for m in ds.mols] == [len(s) for s in samples]
    assert np.allclose(ds.activity, [4.0 + 0.5 * k for k in range(len(samples))])
    fs = _fs(conformers="ensemble", conformer_weighting="given")
    got = build_fields(ds.mols[1], fs)
    assert np.allclose(got.conformer_weights, w[1])
    assert got.n_atoms == sum(m.GetNumAtoms() for m in samples[1])


def test_the_reader_refuses_a_record_without_its_weight(tmp_path):
    path = str(tmp_path / "mix.sdf")
    _mixtures(path, weights=False)
    with pytest.raises(EnsembleError, match="carries no 'member_weight'"):
        load_dataset(path, activity_prop="activity", ensemble_by="sample",
                     member_weight="member_weight")


def test_records_with_the_same_atoms_still_become_conformations(tmp_path):
    path = str(tmp_path / "conf.sdf")
    write_ensembles(path, [[_mol("CCO", 1), _mol("CCO", 2), _mol("CCO", 3)]], ["e"],
                    weights=[[1.0, 2.0, 1.0]])
    ds = load_dataset(path, ensemble_by="name", member_weight="member_weight")
    m = ds.mols[0]
    assert not is_packed(m) and m.GetNumConformers() == 3
    got = build_fields(m, _fs(conformers="ensemble", conformer_weighting="given"))
    assert np.allclose(got.conformer_weights, [0.25, 0.5, 0.25])


# ------------------------------------------------------------------------- the command line


def test_build_and_predict_with_ensembles_from_the_command_line(tmp_path, capsys):
    from cmfqsar.cli import main
    path = str(tmp_path / "mix.sdf")
    samples, _w, _n = _mixtures(path)
    model = str(tmp_path / "m.cmf")
    main(["build", path, "--activity", "activity", "--fields", "electrostatic,steric",
          "--alphas", "0.5", "--lams", "0.1", "--conformations", "ensemble",
          "--ensemble-by", "sample", "--member-weight", "member_weight", "--save", model])
    assert os.path.exists(model)
    capsys.readouterr()
    main(["predict", model, path, "--ensemble-by", "sample", "--member-weight",
          "member_weight"])
    text = capsys.readouterr().out
    assert all(f"s{k} " in text for k in range(len(samples)))


def test_multi_instance_learning_refuses_a_packed_ensemble(tmp_path):
    from cmfqsar.cli import main
    path = str(tmp_path / "mix.sdf")
    _mixtures(path)
    with pytest.raises(SystemExit, match="different atoms"):
        main(["mil", path, "--activity", "activity", "--ensemble-by", "sample"])


# ------------------------------------------------------------------- the gate on BT5's files


def test_the_general_path_gives_bt5s_kernel_exactly(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import run_shells as RS
    from cmfqsar.model import CMFModel

    rows = RS.load_series(limit=3)
    shells = RS.shell_molecules(rows)
    path = str(tmp_path / "e.sdf")
    write_ensembles(path, shells, [str(r["id"]) for r in rows],
                    weights=[[float(m.GetProp("shell_weight")) for m in s] for s in shells])
    ds = load_dataset(path, ensemble_by="sample", member_weight="member_weight")
    assert all(is_packed(m) for m in ds.mols)
    old_fs = FieldSet.create(physchem=list(RS.FIELDS), alpha=1.0, charge_model="gasteiger")
    new_fs = FieldSet.create(physchem=list(RS.FIELDS), alpha=1.0, charge_model="gasteiger",
                             conformers="ensemble", conformer_weighting="given")
    old = [RS.ensemble_from_molecules(s, old_fs, name=str(r["id"]))
           for r, s in zip(rows, shells)]
    new = [build_fields(m, new_fs, n) for m, n in zip(ds.mols, ds.names)]
    for a, b in zip(old, new):
        _same_fields(a, b)
        assert np.array_equal(a.conformer_weights, b.conformer_weights)
    kernels = []
    for fs, mf in ((old_fs, old), (new_fs, new)):
        table = AtomTable.from_molecules(mf)
        model = CMFModel(fieldset=fs, method="krr", normalize=True)
        kernels.append(model.joint_kernel(model.prepare_training_kernels(table),
                                          table.descriptors))
    assert np.array_equal(kernels[0], kernels[1])


# ------------------------------------------------------------------------ the benchmark script


def _bench():
    sys.path.insert(0, os.path.join(ROOT, "benchmarks"))
    import run_ensembles as RE
    return RE


def test_the_record_comparison_counts_every_number_and_ignores_the_time():
    RE = _bench()
    a = {"mae_mean": 1.0, "per_fold": [{"alpha": 0.5, "lam": 0.1}], "abs_errors": [0.1, 0.2],
         "elapsed_s": 3.0}
    b = {"mae_mean": 1.0, "per_fold": [{"alpha": 0.5, "lam": 0.1}], "abs_errors": [0.1, 0.25],
         "elapsed_s": 9.0}
    assert RE._largest_difference(a, a) == 0.0
    assert RE._largest_difference(a, b) == pytest.approx(0.05)
    assert RE._largest_difference(a, {**b, "extra": 1}) == float("inf")
    assert RE._largest_difference({"x": [1, 2]}, {"x": [1]}) == float("inf")


def test_the_table_says_whether_the_gate_is_met(tmp_path):
    import json
    RE = _bench()
    with open(tmp_path / "file.json", "w") as fh:
        json.dump({"records": 4, "electrolytes": 2, "members_per_electrolyte": {"min": 1, "max": 3},
                   "atoms_per_member": {"min": 5, "max": 9},
                   "electrolytes_whose_members_differ_in_atoms": 1}, fh)
    gate = {"kernels": {"0.5": {"framed_kernel": 0.0, "alignment_free_rows": 0.0}},
            "readings": [{"target": "ea_298", "split": "random", "model": "shells",
                          "stored_mae": 1.0, "bt5_mae": 1.0, "bt5_difference": 0.0,
                          "general_mae": 1.0, "general_difference": 0.0}],
            "largest_difference_control": 0.0, "largest_difference_general": 0.0,
            "largest_difference_tables": 0.0, "gate_met": True}
    with open(tmp_path / "gate.json", "w") as fh:
        json.dump(gate, fh)
    assert "**Met.**" in RE.write_table(str(tmp_path))
    assert (tmp_path / "table.md").exists()
    gate.update(gate_met=False, largest_difference_general=1e-9)
    with open(tmp_path / "gate.json", "w") as fh:
        json.dump(gate, fh)
    assert "**Not met.**" in RE.write_table(str(tmp_path))
