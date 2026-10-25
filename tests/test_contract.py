"""The model file records how its training set was prepared, and predicting checks it."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar import __version__  # noqa: E402
from cmfqsar.contract import FRAME_MARGIN, check_molecules, summary  # noqa: E402
from cmfqsar.dataio import load_model, save_model  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402

SMILES = ["c1ccccc1O", "c1ccccc1N", "c1ccccc1C", "c1ccccc1Cl", "c1ccc(F)cc1O", "c1ccc(C)cc1O",
          "c1ccc(Cl)cc1N", "c1ccc(O)cc1C"]


def _mol(smiles, seed, charges=True, hydrogens=True):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    if not hydrogens:
        m = Chem.RemoveHs(m)
    if charges:
        AllChem.ComputeGasteigerCharges(m)
        for a in m.GetAtoms():        # "file" charges: the Gasteiger ones, stored as the file's
            a.SetDoubleProp("partial_charge", float(a.GetProp("_GasteigerCharge")) * 1.5)
    return m


def _fitted(charge_model="file", hydrogens=True):
    fs = FieldSet.create(alpha=0.6, charge_model=charge_model)
    mols = [_mol(s, 11 + i, charges=True, hydrogens=hydrogens) for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    y = np.linspace(4.0, 7.0, len(mf))
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y)
    return model, fs


def _codes(findings, level=None):
    return {f.code for f in findings if level is None or f.level == level}


def test_the_file_carries_its_provenance_and_comes_back(tmp_path):
    model, fs = _fitted()
    path = str(tmp_path / "m.cmf")
    save_model(path, model)
    back, _ = load_model(path)
    p = back.provenance
    assert back.file_version == CMFModel.FILE_VERSION
    assert p["cmfqsar"] == __version__ and p["n_train"] == len(SMILES)
    assert p["charge_source"] == "file" and p["charge_sources"] == {"file": len(SMILES)}
    assert p["explicit_hydrogens"] == 1.0 and p["param_scheme"] == "rdkit"
    assert "electrostatic" in p["fields"] and len(p["box_min"]) == 3
    assert "charges from the file" in summary(back) and "explicit hydrogens" in summary(back)


def test_a_file_from_a_newer_program_is_refused(tmp_path):
    model, fs = _fitted()
    st = model.state()
    st["version"] = CMFModel.FILE_VERSION + 1
    with pytest.raises(ValueError, match="newer"):
        CMFModel.from_state(st)


def test_an_old_file_without_provenance_loads_and_the_check_says_so():
    model, fs = _fitted()
    st = model.state()
    st["version"] = 1
    del st["provenance"]
    old = CMFModel.from_state(st)
    assert old.provenance == {}
    mf = [build_fields(_mol("c1ccccc1Br", 99), fs, "new")]
    findings = check_molecules(old, mf)
    assert "no-provenance" in _codes(findings, "note")
    assert not _codes(findings, "refuse")


def test_molecules_prepared_like_the_training_set_pass():
    model, fs = _fitted()
    mf = [build_fields(_mol("c1ccccc1Br", 99), fs, "new")]
    findings = check_molecules(model, mf)
    assert not _codes(findings, "refuse") and not _codes(findings, "warn")


def test_a_molecule_without_file_charges_is_refused_when_the_model_used_them():
    model, fs = _fitted(charge_model="file")
    good = build_fields(_mol("c1ccccc1Br", 99), fs, "with")
    bad = build_fields(_mol("c1ccccc1I", 98, charges=False), fs, "without")
    assert good.charge_source == "file" and bad.charge_source == "gasteiger"
    findings = check_molecules(model, [good, bad])
    refused = [f for f in findings if f.level == "refuse"]
    assert len(refused) == 1 and refused[0].code == "charges" and "without" in refused[0].message


def test_computed_charges_never_refuse():
    model, fs = _fitted(charge_model="gasteiger")
    bad = build_fields(_mol("c1ccccc1I", 98, charges=False), fs, "without")
    assert not _codes(check_molecules(model, [bad]), "refuse")


def test_missing_hydrogens_are_a_warning():
    model, fs = _fitted()
    mf = [build_fields(_mol("c1ccccc1Br", 99, hydrogens=False), fs, "bare")]
    assert "hydrogens" in _codes(check_molecules(model, mf), "warn")
    model_h, fs_h = _fitted(hydrogens=False)
    mf = [build_fields(_mol("c1ccccc1Br", 99, hydrogens=True), fs_h, "withH")]
    assert "hydrogens" in _codes(check_molecules(model_h, mf), "warn")


def test_a_molecule_outside_the_frame_is_a_warning():
    model, fs = _fitted()
    m = _mol("c1ccccc1Br", 99)
    conf = m.GetConformer()
    for i in range(m.GetNumAtoms()):
        p = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (p.x + 3 * FRAME_MARGIN + 20.0, p.y, p.z))
    mf = [build_fields(m, fs, "far")]
    findings = check_molecules(model, mf)
    assert "frame" in _codes(findings, "warn")
    assert "frame" not in _codes(check_molecules(model, [build_fields(_mol("c1ccccc1Br", 99), fs, "near")]))


def test_the_command_line_refuses_and_allow_mismatch_goes_on(tmp_path, capsys):
    from cmfqsar.cli import main
    model, fs = _fitted(charge_model="file")
    path = str(tmp_path / "m.cmf")
    save_model(path, model)
    sdf = str(tmp_path / "new.sdf")
    w = Chem.SDWriter(sdf)
    m = _mol("c1ccccc1I", 98, charges=False)
    m.SetProp("_Name", "iodobenzene")
    w.write(m)
    w.close()
    with pytest.raises(SystemExit, match="refused"):
        main(["predict", path, sdf])
    main(["predict", path, sdf, "--allow-mismatch"])
    out = capsys.readouterr().out
    assert "[refuse]" in out and "iodobenzene" in out


def test_a_descriptor_block_survives_the_file_and_an_old_file_refits_it(tmp_path):
    fs = FieldSet.create(physchem=["steric"], alpha=0.6, descriptors=["calc:mw"], descriptor_h=0.5)
    mols = [_mol(s, 21 + i) for i, s in enumerate(SMILES)]
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    table = AtomTable.from_molecules(mf)
    y = np.array([float(m.GetNumAtoms()) for m in mols])
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.01})
    model.fit(table, y)
    before = model.predict(table)
    assert model.desc_train_ is not None
    path = str(tmp_path / "d.cmf")
    save_model(path, model)
    back, _ = load_model(path)
    np.testing.assert_allclose(back.predict(table), before)
    # a version-1 file carried no descriptor statistics: they are fitted again on the stored atoms
    st = model.state()
    st["version"] = 1
    for k in ("provenance", "desc_scaler", "desc_train", "desc_scale"):
        del st[k]
    old = CMFModel.from_state(st)
    np.testing.assert_allclose(old.predict(table), before)


# --------------------------------------------------------------------------- #
# Inside the box, but turned the wrong way
# --------------------------------------------------------------------------- #

def _turned(mol, axis=2, degrees=90.0):
    """The same molecule rotated about its own centroid: the right place, the wrong way."""
    m = Chem.Mol(mol)
    conf = m.GetConformer()
    xyz = np.array(conf.GetPositions())
    c = xyz.mean(axis=0)
    t = np.radians(degrees)
    R = {0: np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]]),
         1: np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]]),
         2: np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])}[axis]
    for i, p in enumerate((xyz - c) @ R.T + c):
        conf.SetAtomPosition(i, [float(v) for v in p])
    return m


def _aligned_set(n=14, seed=3):
    """A congeneric series in one frame: benzenes superimposed on their ring."""
    from rdkit.Chem import rdMolAlign

    subs = ["O", "N", "C", "Cl", "F", "Br", "CC", "OC", "NC", "C=O", "S", "I", "CCC", "C#N"]
    fs = FieldSet.create(alpha=0.6, charge_model="gasteiger")
    mols = []
    ref = None
    for i, s in enumerate(subs[:n]):
        m = _mol(f"c1ccccc1{s}", seed + i, charges=False)
        if ref is None:
            ref = m
        else:
            rdMolAlign.AlignMol(m, ref, atomMap=[(j, j) for j in range(6)])
        mols.append(m)
    return fs, mols


def _fitted_aligned(tmp_path, n=14):
    fs, mols = _aligned_set(n=n)
    mf = [build_fields(m, fs, f"m{i}") for i, m in enumerate(mols)]
    y = np.linspace(4.0, 7.0, len(mf))
    model = CMFModel(fieldset=fs, method="krr", normalize=True, params={"lam": 0.1})
    model.fit(AtomTable.from_molecules(mf), y)
    path = str(tmp_path / "aligned.cmf")
    save_model(path, model)                 # the provenance is written on the way out
    return load_model(path)[0], fs, mols


def test_a_molecule_of_the_series_raises_no_complaint(tmp_path):
    model, fs, mols = _fitted_aligned(tmp_path)
    mf = [build_fields(m, fs, f"ok{i}") for i, m in enumerate(mols)]
    assert "orientation" not in _codes(check_molecules(model, mf))


def test_a_molecule_turned_about_its_own_centre_is_caught(tmp_path):
    """The box test cannot see this: a rotation about the centroid moves no centroid."""
    model, fs, mols = _fitted_aligned(tmp_path)
    turned = [build_fields(_turned(m, axis=1), fs, f"turned{i}") for i, m in enumerate(mols)]
    findings = check_molecules(model, turned)
    assert "orientation" in _codes(findings, level="warn")
    assert "frame" not in _codes(findings)       # the centroids are exactly where they were
    msg = [f.message for f in findings if f.code == "orientation"][0]
    assert "turned the wrong way" in msg and "% of their atoms" in msg


def test_the_threshold_comes_from_the_training_set_itself(tmp_path):
    from cmfqsar.contract import _occupancy, _occupancy_threshold

    model, fs, mols = _fitted_aligned(tmp_path)
    thr = _occupancy_threshold(model)
    assert 0.5 <= thr <= 1.0
    # every training molecule passes its own set's rule, and a turned one does not
    good = [_occupancy(model, build_fields(m, fs, "x").coords) for m in mols]
    bad = [_occupancy(model, build_fields(_turned(m, axis=1), fs, "x").coords) for m in mols]
    assert float(np.mean(good)) > thr > float(np.mean(bad))


def test_a_model_with_no_training_coordinates_says_nothing(tmp_path):
    """The check needs the cloud; without it there is no verdict to give, not a false one."""
    model, fs, mols = _fitted_aligned(tmp_path)
    model.train = None
    mf = [build_fields(_turned(m), fs, f"t{i}") for i, m in enumerate(mols)]
    assert "orientation" not in _codes(check_molecules(model, mf))
