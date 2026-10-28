"""Projects (plan item G2): pose snapshots and undo, the project file round trip, two models
compared on one set of molecules, and the application's own save, reopen and undo
(run with: python -m pytest tests/test_project.py)."""
import os
import sys

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cmfqsar.dataio import Dataset  # noqa: E402
from cmfqsar.fields import AtomTable, FieldSet, build_fields  # noqa: E402
from cmfqsar.model import CMFModel  # noqa: E402
from cmfqsar.project import (PoseHistory, Project, compare_models, is_project_file,  # noqa: E402
                             load_project, pose_displacement, restore_poses, save_project,
                             snapshot_poses)

SMILES = ["CCO", "CCN", "CCC", "CC(=O)O", "CCF", "CCCl", "CCBr", "CCI", "CCCO", "CCCN"]


def _mol(smiles, seed, name, y):
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    p = AllChem.ETKDGv3()
    p.randomSeed = seed
    AllChem.EmbedMolecule(m, p)
    m.SetProp("_Name", name)
    m.SetProp("pKi", f"{y:.2f}")
    return m


def _dataset():
    mols = [_mol(s, i, f"m{i}", 0.3 * i) for i, s in enumerate(SMILES)]
    return Dataset(mols=mols, names=[f"m{i}" for i in range(len(SMILES))],
                   activity=0.3 * np.arange(len(SMILES)),
                   subset=np.array(["train"] * 8 + ["test"] * 2, dtype=object),
                   activity_prop="pKi", properties=["pKi"])


def _shift(mols, dx):
    for m in mols:
        c = m.GetConformer()
        for a in range(m.GetNumAtoms()):
            x = c.GetAtomPosition(a)
            c.SetAtomPosition(a, [x.x + dx, x.y, x.z])


def _fit(ds, lam, idx=range(8)):
    fs = FieldSet.create(physchem=("electrostatic", "steric"), alpha=0.7)
    tab = AtomTable.from_molecules([build_fields(ds.mols[i], fs, ds.names[i]) for i in idx])
    model = CMFModel(fieldset=fs, method="krr", params={"lam": lam}, normalize=True)
    model.fit(tab, ds.activity[list(idx)], cv_folds=3)
    return model


def test_snapshot_restores_every_conformer_and_undo_is_last_in_first_out():
    ds = _dataset()
    m0 = ds.mols[0]
    AllChem.EmbedMultipleConfs(m0, numConfs=3, randomSeed=7)       # a molecule with an ensemble
    start = [c.GetPositions().copy() for c in m0.GetConformers()]
    h = PoseHistory()
    h.remember(ds.mols, "first")
    _shift(ds.mols, 1.0)
    h.remember(ds.mols, "second")
    _shift(ds.mols, 2.0)
    assert h.labels() == ["first", "second"] and h.last_label == "second"
    label, moved = h.undo(ds.mols)
    assert label == "second" and np.allclose(moved, 2.0, atol=1e-6)
    assert np.allclose(m0.GetConformer(0).GetPositions(), start[0] + [1.0, 0, 0])
    label, moved = h.undo(ds.mols)
    assert label == "first" and len(h) == 0
    assert m0.GetNumConformers() == 3
    for c, s in zip(m0.GetConformers(), start):
        assert np.allclose(c.GetPositions(), s)
    with pytest.raises(ValueError):
        h.undo(ds.mols)


def test_a_snapshot_taken_before_conformers_removes_them_again():
    ds = _dataset()
    snap = snapshot_poses(ds.mols, "single conformer")
    AllChem.EmbedMultipleConfs(ds.mols[1], numConfs=4, randomSeed=3)
    assert ds.mols[1].GetNumConformers() == 4
    restore_poses(ds.mols, snap)
    assert ds.mols[1].GetNumConformers() == 1
    assert np.allclose(pose_displacement(ds.mols, snap), 0.0)
    # a snapshot of another set is refused rather than applied to the wrong molecules
    with pytest.raises(ValueError):
        restore_poses(ds.mols[:3], snap)


def test_the_history_is_bounded():
    ds = _dataset()
    h = PoseHistory(limit=3)
    for k in range(5):
        h.remember(ds.mols, f"step {k}")
    assert h.labels() == ["step 2", "step 3", "step 4"]


def test_project_round_trip_keeps_molecules_properties_model_and_settings(tmp_path):
    ds = _dataset()
    h = PoseHistory()
    h.remember(ds.mols, "Align to m0")
    _shift(ds.mols, 1.5)
    model = _fit(ds, 0.1)
    proj = Project(dataset=ds, settings={"model_panel": {"ed_lams": "0.1, 1", "chk_cif": True}},
                   history=h.steps, model_state=model.state(), train_indices=list(range(8)),
                   test_indices=[8, 9], search_rows=[{"alpha": 0.7, "lam": 0.1, "rmse_cv": 0.4}],
                   validation={"lines": ["y-scrambling: fine"]}, log="built a model",
                   target=ds.activity)
    path = str(tmp_path / "session.cmfproj")
    save_project(path, proj)
    assert proj.saved and proj.program_version
    assert is_project_file(path)
    back = load_project(path)
    assert back.dataset.names == ds.names
    assert list(back.dataset.subset) == list(ds.subset)
    assert np.allclose(back.dataset.activity, ds.activity)
    assert back.dataset.mols[3].GetProp("pKi") == ds.mols[3].GetProp("pKi")
    assert back.dataset.activity_prop == "pKi"
    for a, b in zip(back.dataset.mols, ds.mols):
        assert np.allclose(a.GetConformer().GetPositions(), b.GetConformer().GetPositions())
    assert back.settings == proj.settings
    assert [s["label"] for s in back.history] == ["Align to m0"]
    assert back.train_indices == list(range(8)) and back.test_indices == [8, 9]
    assert back.search_rows == proj.search_rows and back.validation == proj.validation
    assert back.log == "built a model" and np.allclose(back.target, ds.activity)
    # the model predicts exactly as before
    tab = AtomTable.from_molecules([build_fields(m, model.fieldset, n) for m, n in zip(ds.mols, ds.names)])
    assert np.allclose(back.model.predict(tab), model.predict(tab))
    # and the history restores the poses from before the alignment step
    PoseHistory(back.history).undo(back.dataset.mols)
    assert np.allclose(back.dataset.mols[0].GetConformer().GetPositions(),
                       ds.mols[0].GetConformer().GetPositions() - [1.5, 0, 0])


def test_a_model_file_is_not_a_project_and_a_newer_project_is_refused(tmp_path):
    import joblib
    from cmfqsar.dataio import save_model
    ds = _dataset()
    model = _fit(ds, 1.0)
    mpath = str(tmp_path / "m.cmf")
    save_model(mpath, model)
    assert not is_project_file(mpath)
    with pytest.raises(ValueError):
        load_project(mpath)
    joblib.dump({"kind": "cmf-project", "version": 99, "dataset": {}}, str(tmp_path / "new.cmfproj"))
    with pytest.raises(ValueError, match="newer"):
        load_project(str(tmp_path / "new.cmfproj"))


def test_compare_models_scores_both_on_the_same_molecules(tmp_path):
    ds = _dataset()
    a, b = _fit(ds, 0.01), _fit(ds, 100.0)
    cmp = compare_models(a, b, ds, labels=("tight", "loose"))
    assert len(cmp.rows) == len(ds)
    assert cmp.summary["train"]["n"] == 8 and cmp.summary["test"]["n"] == 2
    assert cmp.summary["all"]["n"] == 10
    # the tightly fitted model is nearer the training activities than the heavily shrunk one
    assert cmp.summary["train"]["rmse_a"] < cmp.summary["train"]["rmse_b"]
    assert cmp.agreement["n_with_activity"] == 10
    assert 0 <= cmp.agreement["b_closer"] <= 10
    assert cmp.agreement["most_different"] in ds.names
    rows = [r for r in cmp.rows if r.name == cmp.agreement["most_different"]]
    assert abs(rows[0].difference) == pytest.approx(cmp.agreement["max_abs_difference"])
    lines = cmp.lines()
    assert lines[0].startswith("Two models on 10 molecules") and any("correlation" in l for l in lines)
    # the same model twice agrees with itself exactly
    same = compare_models(a, a, ds)
    assert same.agreement["max_abs_difference"] == pytest.approx(0.0, abs=1e-12)
    out = tmp_path / "cmp.csv"
    cmp.to_csv(str(out))
    text = out.read_text().splitlines()
    assert text[0].split(",")[3:5] == ["predicted_tight", "predicted_loose"] and len(text) == 11


# --------------------------------------------------------------------------- the application

def _wait(app, win, timeout=120):
    import time
    t0 = time.time()
    while win.worker is not None and win.worker.isRunning() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyqtgraph")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _window(app):
    from cmfqsar.gui.main_window import MainWindow
    win = MainWindow()
    win._auto = True
    return win


def test_the_application_undoes_a_step_and_discards_the_model(app):
    win = _window(app)
    ds = _dataset()
    win.ds = ds
    win._after_dataset_change()
    assert not win.act_undo.isEnabled()
    before = ds.mols[2].GetConformer().GetPositions().copy()
    win.model = _fit(ds, 1.0)
    win.model_train_idx = list(range(8))
    win._remember_poses("Test step")
    _shift(ds.mols, 3.0)
    win._after_poses_changed()
    assert win.act_undo.isEnabled() and "Test step" in win.act_undo.text()
    win.undo_alignment()
    assert np.allclose(ds.mols[2].GetConformer().GetPositions(), before)
    assert win.model is None and not win.act_undo.isEnabled()
    assert "Undid 'Test step'" in win.results_panel.log.toPlainText()


def test_the_application_saves_and_reopens_a_session(app, tmp_path):
    win = _window(app)
    ds = _dataset()
    win.ds = ds
    win._after_dataset_change()
    win.model_panel.ed_alphas.setText("0.7")
    win.model_panel.ed_lams.setText("0.1, 1")
    win.model_panel.spin_folds.setValue(4)
    win._remember_poses("Before the model")
    win.build_model()
    _wait(app, win)
    assert win.model is not None
    q2 = win.model.stats.q2
    win.model_panel.ed_lams.setText("0.5, 2, 7")        # a setting changed after the fit
    win.controls.spin_spacing.setValue(0.9)
    path = str(tmp_path / "session.cmfproj")
    win.project_path = path
    win.save_project_file()
    assert os.path.exists(path) and os.path.basename(path) in win.windowTitle()
    assert "Project saved" in win.results_panel.log.toPlainText()

    win2 = _window(app)
    win2.load_project_file(path)
    _wait(app, win2)                                     # reading the file
    _wait(app, win2)                                     # restoring the model's fields
    assert len(win2.ds) == 10 and win2.ds.names == ds.names
    assert win2.model is not None and win2.model.stats.q2 == pytest.approx(q2)
    assert win2.model_train_idx == list(range(8)) and win2.model_test_idx == [8, 9]
    assert win2.model_panel.ed_lams.text() == "0.5, 2, 7"
    assert win2.model_panel.spin_folds.value() == 4
    assert win2.controls.spin_spacing.value() == pytest.approx(0.9)
    assert len(win2.history) == 1 and "Before the model" in win2.act_undo.text()
    assert "q2 (4-fold CV)" in win2.results_panel.txt_stats.toPlainText()
    assert "log of the saved session" in win2.results_panel.log.toPlainText()
    # the project's model and the live one agree on every molecule
    cmp = compare_models(win.model, win2.model, win2.ds, labels=("live", "reopened"))
    assert cmp.agreement["max_abs_difference"] == pytest.approx(0.0, abs=1e-9)
    # undo after reopening restores the poses from before the model and drops the model
    win2.undo_alignment()
    assert win2.model is None and len(win2.history) == 0


def test_the_comparison_window_lists_every_molecule(app):
    from cmfqsar.gui.main_window import ComparisonWindow, CompareModelsDialog
    ds = _dataset()
    cmp = compare_models(_fit(ds, 0.1), _fit(ds, 10.0), ds, labels=("A", "B"))
    w = ComparisonWindow(cmp)
    assert w.table.rowCount() == 10 and w.table.columnCount() == 6
    names = {w.table.item(r, 0).text() for r in range(10)}
    assert names == set(ds.names)
    dlg = CompareModelsDialog(has_current=True)
    assert dlg.side(0)[0] == 0 and dlg.side(1)[0] == 1
    dlg2 = CompareModelsDialog(has_current=False)
    assert dlg2.side(0)[0] == 1
