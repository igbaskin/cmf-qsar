"""Tests for the script that poses every ligand while it is held out (part of plan item D8).

The science of the posing is tested with the docking and the pose protocol. What is fixed here
is what the script promises about its files: a run with another orientation setting writes
another file, so the poses the repository carries cannot be overwritten by a run with different
settings, and the setting is stored with the poses it produced.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))

import make_docked_poses as MK  # noqa: E402


def test_the_file_name_carries_the_orientation_setting():
    assert MK.label("mcl1", "pocket", 180.0) == "mcl1_pocket_rot180"
    assert MK.label("mcl1", "pocket", 360.0) == "mcl1_pocket_rot360"


def test_the_first_files_keep_their_name():
    """They were made at 30 degrees, before the setting existed, and the statement says so."""
    assert MK.label("mcl1", "pocket", MK.HISTORICAL_ROTATION) == "mcl1_pocket"
    assert os.path.isfile(os.path.join(MK.OUT_DIR, "mcl1_pocket.sdf"))


def test_the_default_draws_any_orientation():
    assert MK.ROTATION >= 180.0, "the protocol destroys the orientation, so it must be findable"


def test_the_posing_passes_the_orientation_setting_to_the_docking(monkeypatch):
    """What the poser is given is what the script asked for, not the docking's own default."""
    from cmfqsar import docking
    from cmfqsar.fields import FieldSet
    from cmfqsar.poses import POSERS

    seen = {}

    def fake(mol, pocket, fieldset, **kw):
        seen.update(kw)
        return np.asarray(mol.coords, float), None

    monkeypatch.setattr(docking, "dock_coords", fake)
    fs = FieldSet.create(physchem=["steric"], alpha=0.5, charge_model="gasteiger",
                         param_scheme="rdkit")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    POSERS["pocket"]([mol], {"fieldset": fs, "pocket": None, "restarts": 3,
                             "max_rotation_deg": 180.0})
    assert seen.get("max_rotation_deg") == 180.0
    assert seen.get("restarts") == 3


def _probe():
    """One small molecule with coordinates, and a field set to build it in."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    from cmfqsar.fields import FieldSet

    fs = FieldSet.create(physchem=["steric"], alpha=0.5, charge_model="gasteiger",
                         param_scheme="rdkit")
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    return mol, fs


def test_the_confidence_search_is_told_when_the_orientation_is_unknown(monkeypatch):
    """`variance` recovers a destroyed pose too, so the setting must reach its restarts."""
    from cmfqsar import varalign
    from cmfqsar.poses import POSERS

    seen = {}
    monkeypatch.setattr(varalign, "variance_align",
                        lambda mols, model, **kw: seen.update(kw) or [])
    mol, fs = _probe()
    for rotation, expected in ((30.0, False), (180.0, True)):
        seen.clear()
        POSERS["variance"]([mol], {"fieldset": fs, "model": None, "restarts": 2,
                                   "max_rotation_deg": rotation})
        assert seen["any_orientation"] is expected


def test_the_joint_search_is_told_when_the_orientation_is_unknown(monkeypatch):
    """The pocket and the confidence together: the same restarts, the same setting."""
    from cmfqsar import pocketalign
    from cmfqsar.poses import POSERS

    seen = {}
    monkeypatch.setattr(pocketalign, "pocket_align",
                        lambda mols, model, pocket, **kw: seen.update(kw) or [])
    mol, fs = _probe()
    for rotation, expected in ((30.0, False), (180.0, True)):
        seen.clear()
        POSERS["pocket+variance"]([mol], {"fieldset": fs, "model": None, "pocket": None,
                                          "restarts": 2, "max_rotation_deg": rotation})
        assert seen["any_orientation"] is expected


def test_the_flexible_search_is_told_when_the_orientation_is_unknown(monkeypatch):
    """Freeing the torsions does not turn the molecule round, so it needs the setting as well."""
    from cmfqsar import flexalign
    from cmfqsar.poses import POSERS

    seen = {}
    monkeypatch.setattr(flexalign, "flexible_align",
                        lambda mols, rdmols, model, **kw: seen.update(kw) or [])
    mol, fs = _probe()
    for rotation, expected in ((30.0, False), (180.0, True)):
        seen.clear()
        POSERS["flex"]([mol], {"fieldset": fs, "model": None, "restarts": 2,
                               "max_rotation_deg": rotation})
        assert seen["any_orientation"] is expected


def test_the_record_stores_the_setting_the_poses_were_made_with():
    import json

    path = os.path.join(MK.OUT_DIR, "mcl1_pocket.json")
    with open(path) as fh:
        rec = json.load(fh)
    assert rec.get("max_rotation_deg", MK.HISTORICAL_ROTATION) == MK.HISTORICAL_ROTATION
    assert rec["restarts"] == 2
