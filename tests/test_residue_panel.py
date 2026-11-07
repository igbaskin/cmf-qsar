"""The by-residue view: what the table lists, and what the 3D view labels."""
import os
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cmfqsar.complementarity import InteractionProfile          # noqa: E402
from cmfqsar.gui.main_window import CONTACT_DISTANCE, MainWindow  # noqa: E402
from cmfqsar.gui.panels import ResultsPanel                     # noqa: E402

LABELS = ["electrostatic", "hydrophobic", "burial"]
RESIDUES = ["GLU81.A", "LEU83.A", "LYS200.A"]


@pytest.fixture(scope="module")
def app(qt_application):
    return qt_application


@pytest.fixture()
def panel(app):
    return ResultsPanel()


def _profile(name="lig_1"):
    """Two residues in contact, one far away, and one term below the display threshold."""
    terms = np.array([[0.004, -0.002, 0.007],
                      [0.0000001, 0.030, 0.019],
                      [0.001, 0.001, 0.001]])
    return InteractionProfile(name=name, residues=list(RESIDUES), labels=list(LABELS),
                              terms=terms, contacts=np.array([3.1, 2.8, 9.0]))


def _stub(panel, prof, conserved=None, cur=0):
    return SimpleNamespace(
        complementarity={"profiles": {cur: prof}, "residues": list(RESIDUES),
                         "conserved": conserved},
        results_panel=panel,
        dataset_panel=SimpleNamespace(current_index=lambda: cur))


def test_only_residues_in_contact_are_listed(panel):
    """Every Gaussian reaches everywhere, so a non-zero term is not a contact; distance is."""
    MainWindow._show_residue_profile(_stub(panel, _profile()), 0)
    t = panel.tbl_comp_res
    assert t.rowCount() == 2
    listed = {t.item(r, 0).text() for r in range(t.rowCount())}
    assert listed == {"GLU81.A", "LEU83.A"}
    assert "LYS200.A" not in listed
    assert f"within {CONTACT_DISTANCE:g}" in panel.lbl_comp_res.text()


def test_the_header_says_how_much_of_the_score_those_residues_carry(panel):
    MainWindow._show_residue_profile(_stub(panel, _profile()), 0)
    assert "% of the score" in panel.lbl_comp_res.text()


def test_the_strongest_residue_comes_first(panel):
    MainWindow._show_residue_profile(_stub(panel, _profile()), 0)
    assert panel.tbl_comp_res.item(0, 0).text() == "LEU83.A"     # 0.030 ranks above 0.007


def test_a_term_below_the_display_threshold_is_blank_not_zero(panel):
    MainWindow._show_residue_profile(_stub(panel, _profile()), 0)
    t = panel.tbl_comp_res
    row = [r for r in range(t.rowCount()) if t.item(r, 0).text() == "LEU83.A"][0]
    assert t.item(row, 1).text() == ""                            # 1e-7 electrostatic
    assert t.item(row, 2).text() == "+0.0300"


def test_the_cells_sort_as_numbers(panel):
    """Qt sorts table items with <, which on text would put +0.0070 above +0.0300."""
    MainWindow._show_residue_profile(_stub(panel, _profile()), 0)
    t = panel.tbl_comp_res
    a = t.item(0, 3)
    b = t.item(1, 3)
    assert (a < b) == (float(a.text()) < float(b.text()))


def test_the_conserved_column_is_filled_when_a_consensus_exists(panel):
    conserved = np.array([[1.0, 0.2, 1.0], [1.0, 1.0, 0.4], [0.0, 0.0, 0.0]])
    MainWindow._show_residue_profile(_stub(panel, _profile(), conserved), 0)
    t = panel.tbl_comp_res
    assert t.item(0, t.columnCount() - 1).text() == "100"


def test_without_a_profile_the_table_says_so(panel):
    stub = SimpleNamespace(complementarity={}, results_panel=panel,
                           dataset_panel=SimpleNamespace(current_index=lambda: 0))
    MainWindow._show_residue_profile(stub, 0)
    assert panel.tbl_comp_res.rowCount() == 0
    assert "by residue" in panel.lbl_comp_res.text()


def test_contributions_cover_both_spellings_of_a_residue(panel):
    """A single-chain site labels residues without the chain; the profile always carries it."""
    contrib = MainWindow._residue_contributions(_stub(panel, _profile()))
    assert contrib["GLU81.A"] == pytest.approx(contrib["GLU81"])
    assert "LYS200.A" not in contrib and "LYS200" not in contrib   # not in contact


def test_contributions_are_the_row_sums(panel):
    contrib = MainWindow._residue_contributions(_stub(panel, _profile()))
    assert contrib["LEU83.A"] == pytest.approx(0.0000001 + 0.030 + 0.019)
