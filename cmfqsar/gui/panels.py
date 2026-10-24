"""Dock panels of the CMF application: data set table, model settings, results, viewer controls."""
from __future__ import annotations

import os

from typing import Dict, List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCompleter, QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
                               QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton,
                               QRadioButton, QSlider, QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget,
                               QTextEdit, QVBoxLayout, QWidget, QSizePolicy)

from ..fields import PHYSCHEM_FIELDS, FIELD_LABELS, FieldSet


_CHECK_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check.svg").replace("\\", "/")
# explicit indicator look, independent of the (possibly dark) application palette
_SHOW_BOX_STYLE = (
    "QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #7a7a7a; border-radius: 3px; "
    "background: #ffffff; }"
    "QCheckBox::indicator:hover { border-color: #2d7be5; }"
    f"QCheckBox::indicator:checked {{ background: #2d7be5; border-color: #2d7be5; image: url({_CHECK_ICON}); }}"
)


def _group_colour(g: int) -> QColor:
    """A pale, repeating colour per chemotype, so the groups can be told apart at a glance."""
    hues = [205, 25, 120, 290, 55, 170, 330, 240, 85, 10]
    col = QColor()
    col.setHsv(hues[g % len(hues)], 72, 255 - 12 * ((g // len(hues)) % 4))
    return col


def _float_list(text: str, default: List[float]) -> List[float]:
    try:
        vals = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
        return vals or default
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Data set panel
# --------------------------------------------------------------------------- #

class DatasetPanel(QWidget):
    #: the last column holds the applicability-domain similarity, which is filled only when
    #: the domain was asked for; like the prediction columns it sorts and blanks sink
    COLS = ["Show", "Name", "Activity", "Set", "Pred (fit)", "Pred (CV)", "Pred (test)",
            "AD"]
    selection_changed = Signal(int)          # current row (molecule index) or -1
    show_changed = Signal()                  # "show" checkboxes toggled
    subset_changed = Signal()
    load_requested = Signal(str)             # "train" | "test"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        btns = QHBoxLayout()
        self.btn_load_train = QPushButton("Open training set…")
        self.btn_load_test = QPushButton("Add test set…")
        self.btn_load_decoys = QPushButton("Add decoys…")
        self.btn_load_decoys.setToolTip("Background molecules, presumed inactive, used only to evaluate "
                                        "a screening model (AUC and enrichment); never fitted on")
        btns.addWidget(self.btn_load_train)
        btns.addWidget(self.btn_load_test)
        btns.addWidget(self.btn_load_decoys)
        lay.addLayout(btns)
        self.btn_load_train.clicked.connect(lambda: self.load_requested.emit("train"))
        self.btn_load_test.clicked.connect(lambda: self.load_requested.emit("test"))
        self.btn_load_decoys.clicked.connect(lambda: self.load_requested.emit("decoy"))

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Activity:"))
        self.cmb_activity = QComboBox()
        self.cmb_activity.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row2.addWidget(self.cmb_activity)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.btn_split = QPushButton("Random split")
        self.spin_test_frac = QDoubleSpinBox()
        self.spin_test_frac.setRange(0.0, 0.9)
        self.spin_test_frac.setSingleStep(0.05)
        self.spin_test_frac.setValue(0.25)
        self.spin_test_frac.setPrefix("test ")
        self.btn_to_train = QPushButton("→ train")
        self.btn_to_test = QPushButton("→ test")
        row3.addWidget(self.btn_split)
        row3.addWidget(self.spin_test_frac)
        row3.addWidget(self.btn_to_train)
        row3.addWidget(self.btn_to_test)
        lay.addLayout(row3)

        row4 = QHBoxLayout()
        self.btn_show_all = QPushButton("Show all")
        self.btn_show_none = QPushButton("Show none")
        self.btn_show_sel = QPushButton("Show selected only")
        row4.addWidget(self.btn_show_all)
        row4.addWidget(self.btn_show_none)
        row4.addWidget(self.btn_show_sel)
        lay.addLayout(row4)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.Interactive)
        hh.resizeSection(1, 110)
        self.table.setSortingEnabled(False)       # the panel sorts itself, see _sort_by
        hh.setSectionsClickable(True)
        hh.sectionClicked.connect(self._sort_by)
        hh.setToolTip("Click a column to sort by it: again to reverse, a third time to go back "
                      "to the order of the file")
        lay.addWidget(self.table)
        self.lbl_info = QLabel("No data set loaded")
        lay.addWidget(self.lbl_info)

        self.table.currentCellChanged.connect(lambda r, c, pr, pc: self.selection_changed.emit(self.row_to_index(r)))
        self.table.itemChanged.connect(self._item_changed)
        self._show_boxes: Dict[int, QCheckBox] = {}
        self._group_labels = None
        self._group_names = None
        self._names: List[str] = []
        self._activity = np.zeros(0)
        self._subset: List[str] = []
        self._pred: Dict[int, Dict[int, float]] = {c: {} for c in range(4, len(self.COLS))}
        self._shown: List[bool] = []
        self._order: List[int] = []
        self._pos: Dict[int, int] = {}
        self._sort_col, self._sort_desc = -1, False
        self._note = ""
        self.btn_show_all.clicked.connect(lambda: self.set_all_shown(True))
        self.btn_show_none.clicked.connect(lambda: self.set_all_shown(False))
        self.btn_show_sel.clicked.connect(self.show_selected_only)
        self._block = False

    # The table may be sorted, so a row is not a molecule: `_order` holds the data-set index of
    # every row and `_pos` the inverse.  Everything the panel is asked about is in data-set
    # indices; only the rendering knows about rows.
    def row_to_index(self, row: int) -> int:
        return int(self._order[row]) if 0 <= row < len(self._order) else -1

    def index_to_row(self, index: int) -> int:
        return int(self._pos.get(int(index), -1))

    def populate(self, names, activity, subset) -> None:
        self.set_group_labels(None)              # a different series needs its own grouping
        self._note = ""
        self._names = [str(n) for n in names]
        self._activity = np.asarray(activity, float)
        self._subset = [str(s) for s in subset]
        self._pred = {c: {} for c in range(4, len(self.COLS))}
        self._shown = [i == 0 for i in range(len(self._names))]
        self._order = list(range(len(self._names)))
        self._pos = {i: i for i in self._order}
        self._sort_col, self._sort_desc = -1, False
        self.table.horizontalHeader().setSortIndicatorShown(False)
        self._render()
        self._update_info()

    def set_note(self, text: str) -> None:
        """A line beside the counts saying what these molecules now are."""
        self._note = str(text or "")
        self._update_info()

    def _update_info(self) -> None:
        n_dec = sum(1 for s in self._subset if s == "decoy")
        self.lbl_info.setText(
            f"{len(self._names)} molecules  (train {sum(1 for s in self._subset if s == 'train')}, "
            f"test {sum(1 for s in self._subset if s == 'test')}"
            + (f", decoys {n_dec}" if n_dec else "") + ")"
            + (f"   [{self._note}]" if getattr(self, "_note", "") else "")
            + ("" if self._sort_col < 0 else
               f"   sorted by {self._header_text(self._sort_col)}, "
               + self._sort_style(self._sort_col)[2 if self._sort_desc else 1]))

    def _header_text(self, col: int) -> str:
        item = self.table.horizontalHeaderItem(col)
        return item.text() if item is not None else str(col)

    # ----------------------------------------------------------------- rendering
    def _render(self) -> None:
        """Fill every row from the stored data, in the current order."""
        self._block = True
        self.table.setRowCount(0)
        self._show_boxes = {}
        self.table.setRowCount(len(self._order))
        for row, i in enumerate(self._order):
            self.table.setItem(row, 0, QTableWidgetItem(""))
            self._add_show_box(row, i)
            self.table.setItem(row, 1, QTableWidgetItem(self._names[i]))
            self.table.setItem(row, 2, QTableWidgetItem(self._activity_text(i)))
            self.table.setItem(row, 3, QTableWidgetItem(self._subset[i]))
            for c in range(4, len(self.COLS)):
                self.table.setItem(row, c, QTableWidgetItem(self._pred_text(c, i)))
            self._color_row(row, self._subset[i])
            self._set_group_cell(row, i)
        self._block = False

    def _activity_text(self, i: int) -> str:
        y = self._activity[i] if i < len(self._activity) else float("nan")
        return "" if not np.isfinite(y) else f"{y:.3f}"

    def _pred_text(self, col: int, i: int) -> str:
        v = self._pred.get(col, {}).get(i)
        return "" if v is None or not np.isfinite(v) else f"{v:.3f}"

    def _color_row(self, row: int, subset: str) -> None:
        col = {"train": QColor(225, 238, 255), "test": QColor(255, 232, 215)}.get(
            subset, QColor(228, 228, 228))        # decoys: neutral grey
        for c in range(min(self.table.columnCount(), len(self.COLS))):
            it = self.table.item(row, c)
            if it is not None:
                it.setBackground(col)
                it.setForeground(QColor(15, 15, 15))

    def update_subsets(self, subset) -> None:
        self._subset = [str(s) for s in subset]
        self._block = True
        for row, i in enumerate(self._order):
            self.table.item(row, 3).setText(self._subset[i])
            self._color_row(row, self._subset[i])
        self._block = False
        self._update_info()
        if self._sort_col == 3:
            self._apply_sort()

    def update_activity(self, activity) -> None:
        self._activity = np.asarray(activity, float)
        self._block = True
        for row, i in enumerate(self._order):
            self.table.item(row, 2).setText(self._activity_text(i))
        self._block = False
        if self._sort_col == 2:
            self._apply_sort()

    def set_predictions(self, col: int, values: Dict[int, float]) -> None:
        self._pred[col] = {int(k): float(v) for k, v in values.items()}
        self._block = True
        for row, i in enumerate(self._order):
            self.table.item(row, col).setText(self._pred_text(col, i))
        self._block = False
        if self._sort_col == col:
            self._apply_sort()

    def clear_predictions(self) -> None:
        self._pred = {c: {} for c in range(4, len(self.COLS))}
        self._block = True
        for row in range(self.table.rowCount()):
            for c in range(4, len(self.COLS)):
                self.table.item(row, c).setText("")
        self._block = False

    def set_group_labels(self, labels, names=None) -> None:
        """Show which chemotype each molecule belongs to, in a column added for the purpose.

        The column appears only once a grouping exists, and ``None`` takes it away again; the
        row colours keep meaning the subset, so the group is carried by its own cell.
        """
        if labels is None:
            self._group_labels = self._group_names = None
            self.table.setColumnCount(len(self.COLS))
            self.table.setHorizontalHeaderLabels(self.COLS)
            if getattr(self, "_sort_col", -1) >= len(self.COLS):
                self._sort_col = -1
            return
        self._group_labels = np.asarray(labels, dtype=int)
        self._group_names = list(names) if names is not None else None
        if self.table.columnCount() <= len(self.COLS):
            self.table.setColumnCount(len(self.COLS) + 1)
            self.table.setHorizontalHeaderLabels(self.COLS + ["Chemotype"])
        self._block = True
        for row, i in enumerate(self._order):
            self._set_group_cell(row, i)
        self._block = False

    def _set_group_cell(self, row: int, i: int) -> None:
        labels = getattr(self, "_group_labels", None)
        if labels is None or self.table.columnCount() <= len(self.COLS):
            return
        g = int(labels[i]) if i < len(labels) else -1
        it = QTableWidgetItem("" if g < 0 else str(g))
        it.setTextAlignment(Qt.AlignCenter)
        if g >= 0:
            it.setBackground(_group_colour(g))
            it.setForeground(QColor(15, 15, 15))
            names = getattr(self, "_group_names", None)
            it.setToolTip(str(names[g]) if names is not None and g < len(names)
                          else f"chemotype {g}")
        self.table.setItem(row, len(self.COLS), it)

    # ----------------------------------------------------------------- sorting
    #: how each column reads best on the first click, and what to call the two directions
    SORT_STYLE = {0: (True, "shown first", "hidden first"),
                  1: (False, "A to Z", "Z to A"),
                  2: (True, "highest first", "lowest first"),
                  3: (False, "train first", "decoys first")}
    SORT_NUMERIC = ("highest first", "lowest first")

    def _sort_value(self, col: int, i: int):
        """The value a column is ordered by, or None when the molecule has none."""
        if col == 0:
            return (bool(self._shown[i]), self._names[i].lower())
        if col == 1:
            return (self._names[i].lower(),)
        if col == 2:
            y = self._activity[i] if i < len(self._activity) else float("nan")
            return None if not np.isfinite(y) else (float(y),)
        if col == 3:
            return ({"train": 0, "test": 1, "decoy": 2}.get(self._subset[i], 3),
                    self._names[i].lower())
        if col >= len(self.COLS):
            labels = getattr(self, "_group_labels", None)
            if labels is None or i >= len(labels):
                return None
            return (int(labels[i]),)
        v = self._pred.get(col, {}).get(i)
        return None if v is None or not np.isfinite(v) else (float(v),)

    def _sort_style(self, col: int):
        if col >= len(self.COLS):                 # the chemotypes are numbered largest group first
            return (False, "largest group first", "smallest group first")
        return self.SORT_STYLE.get(col, (True,) + self.SORT_NUMERIC)

    def _sort_by(self, col: int) -> None:
        """A click on a header sorts by it; a second reverses; a third restores the file order."""
        if col == self._sort_col:
            if not self._sort_desc:
                self._sort_desc = True
            else:
                self._sort_col, self._sort_desc = -1, False
        else:
            self._sort_col, self._sort_desc = col, False
        self._apply_sort()

    def _apply_sort(self) -> None:
        keep = self.current_index()
        hh = self.table.horizontalHeader()
        if self._sort_col < 0:
            self._order = list(range(len(self._names)))
            hh.setSortIndicatorShown(False)
        else:
            col = self._sort_col
            first_desc, _, _ = self._sort_style(col)
            reverse = first_desc != self._sort_desc
            have = [i for i in range(len(self._names)) if self._sort_value(col, i) is not None]
            missing = [i for i in range(len(self._names)) if self._sort_value(col, i) is None]
            have.sort(key=lambda i: self._sort_value(col, i), reverse=reverse)
            self._order = have + missing          # a molecule with no value always sinks
            hh.setSortIndicatorShown(True)
            hh.setSortIndicator(col, Qt.DescendingOrder if reverse else Qt.AscendingOrder)
        self._pos = {int(i): row for row, i in enumerate(self._order)}
        self._render()
        self._update_info()
        if keep >= 0:
            self.select_index(keep)

    # -- "Show" column: real check-box widgets (item check states render poorly on coloured rows)
    def _add_show_box(self, row: int, index: int) -> None:
        box = QCheckBox()
        box.setChecked(bool(self._shown[index]))
        box.setToolTip("Show this molecule in the 3D view")
        box.setStyleSheet(_SHOW_BOX_STYLE)
        box.toggled.connect(lambda on, i=index: self._box_toggled(i, on))
        holder = QWidget()
        hl = QHBoxLayout(holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setAlignment(Qt.AlignCenter)
        hl.addWidget(box)
        holder.setAutoFillBackground(False)
        self.table.setCellWidget(row, 0, holder)
        self._show_boxes[index] = box

    def _box_toggled(self, index: int, on: bool) -> None:
        self._shown[index] = bool(on)
        if not self._block:
            self.show_changed.emit()

    def _set_box(self, index: int, on: bool) -> None:
        if not (0 <= index < len(self._shown)):
            return
        self._shown[index] = bool(on)
        box = self._show_boxes.get(index)
        if box is not None:
            box.blockSignals(True)
            box.setChecked(on)
            box.blockSignals(False)

    def shown_indices(self) -> List[int]:
        return [i for i, on in enumerate(self._shown) if on]

    def selected_indices(self) -> List[int]:
        return sorted({self.row_to_index(ix.row()) for ix in self.table.selectedIndexes()}
                      - {-1})

    def current_index(self) -> int:
        return self.row_to_index(self.table.currentRow())

    def set_all_shown(self, on: bool) -> None:
        self._block = True
        for i in range(len(self._shown)):
            self._set_box(i, on)
        self._block = False
        self.show_changed.emit()

    def show_selected_only(self) -> None:
        sel = set(self.selected_indices())
        self._block = True
        for i in range(len(self._shown)):
            self._set_box(i, i in sel)
        self._block = False
        self.show_changed.emit()

    def set_shown(self, indices, on=True):
        self._block = True
        for i in indices:
            self._set_box(int(i), on)
        self._block = False
        self.show_changed.emit()

    def select_index(self, i: int) -> None:
        row = self.index_to_row(i)
        if row >= 0:
            self.table.selectRow(row)
            self.table.setCurrentCell(row, 1)

    def _item_changed(self, item: QTableWidgetItem) -> None:
        if self._block:
            return
        if item.column() == 0:
            self.show_changed.emit()


# --------------------------------------------------------------------------- #
# Model settings panel
# --------------------------------------------------------------------------- #

class ModelPanel(QWidget):
    build_requested = Signal()
    nested_requested = Signal()
    predict_requested = Signal()
    cancel_requested = Signal()
    fields_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)

        # --- fields -------------------------------------------------------
        gb = QGroupBox("Molecular fields")
        fl = QVBoxLayout(gb)
        self.chk_fields: Dict[str, QCheckBox] = {}
        row = QHBoxLayout()
        for k, name in enumerate(PHYSCHEM_FIELDS):
            cb = QCheckBox(FIELD_LABELS[name])
            cb.setChecked(True)
            cb.toggled.connect(self.fields_changed)
            self.chk_fields[name] = cb
            row.addWidget(cb)
            if k == 2:
                fl.addLayout(row)
                row = QHBoxLayout()
        fl.addLayout(row)
        cif_row = QHBoxLayout()
        self.chk_cif = QCheckBox("Indicator fields (CIF)")
        self.chk_cif.toggled.connect(self.fields_changed)
        self.cmb_cif_scheme = QComboBox()
        self.cmb_cif_scheme.addItems(["sybyl", "element", "element+aromatic"])
        self.cmb_cif_scheme.currentIndexChanged.connect(self.fields_changed)
        self.spin_cif_min = QSpinBox()
        self.spin_cif_min.setRange(1, 999)
        self.spin_cif_min.setValue(5)
        self.spin_cif_min.setPrefix("min. mols ")
        self.spin_cif_min.valueChanged.connect(self.fields_changed)
        cif_row.addWidget(self.chk_cif)
        cif_row.addWidget(self.cmb_cif_scheme)
        cif_row.addWidget(self.spin_cif_min)
        fl.addLayout(cif_row)
        # per-atom property fields: one tick box per property the loaded file carries, so a
        # quantity the program cannot compute itself becomes a field like any other
        self.row_props = QWidget()
        self.lay_props = QHBoxLayout(self.row_props)
        self.lay_props.setContentsMargins(0, 0, 0, 0)
        self.lbl_props = QLabel("Per-atom property fields:")
        self.lay_props.addWidget(self.lbl_props)
        self.lay_props.addStretch(1)
        self.row_props.setVisible(False)
        self._prop_boxes: Dict[str, QCheckBox] = {}
        fl.addWidget(self.row_props)
        # GRID-like probe fields: what a probe would gain by sitting at each point
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Probes:"))
        self._probe_boxes: Dict[str, QCheckBox] = {}
        from ..probes import PROBES as _PROBES
        for name, pr in _PROBES.items():
            box = QCheckBox(name)
            box.setToolTip(
                f"<p>{pr.label}: a field of what that probe would gain by sitting at each point, "
                f"after Goodford's GRID.</p>"
                f"<p>Every atom carries several Gaussians fitted to that attraction, so the "
                f"kernel stays analytic and the co-field can still be drawn. The repulsive wall "
                f"is not in it -- no sum of Gaussians can carry one -- and is left to the steric "
                f"field, which already is a van der Waals envelope.</p>")
            box.toggled.connect(self.fields_changed)
            prow.addWidget(box)
            self._probe_boxes[name] = box
        self.spin_probe_terms = QSpinBox()
        self.spin_probe_terms.setRange(2, 6)
        self.spin_probe_terms.setValue(4)
        self.spin_probe_terms.setPrefix("terms ")
        self.spin_probe_terms.setToolTip(
            "<p>Gaussians per atom in a probe field. The kernel costs the square of this: 3 is "
            "the cheap setting, 5 the careful one.</p>")
        self.spin_probe_terms.valueChanged.connect(lambda _: self.fields_changed.emit())
        prow.addWidget(self.spin_probe_terms)
        prow.addStretch(1)
        fl.addLayout(prow)
        # descriptors the program computes itself, so that a file with no numeric tags still has
        # something to offer; all off by default, because a 3D model should stand on its geometry
        crow3 = QHBoxLayout()
        crow3.addWidget(QLabel("Computed:"))
        self._calc_boxes: Dict[str, QCheckBox] = {}
        from ..descriptors import COMPUTED as _COMPUTED
        for key, (what, _fn) in _COMPUTED.items():
            box = QCheckBox(key)
            box.setChecked(False)
            box.setToolTip(
                f"<p>Use the {what} as a descriptor: one number per molecule, computed by RDKit "
                f"and standardised on the training set, beside the fields.</p>"
                f"<p>Off by default. These are 2D quantities: they can lift a model that has a "
                f"size or lipophilicity trend in it, and they can also carry a model that then "
                f"owes nothing to its geometry, which the contribution table will show.</p>")
            box.toggled.connect(self.fields_changed)
            crow3.addWidget(box)
            self._calc_boxes[key] = box
        # the four above are shortcuts; any of RDKit's 200-odd descriptors can be named here
        self.ed_calc_more = QLineEdit()
        self.ed_calc_more.setPlaceholderText("more, by RDKit name: NumHAcceptors, FractionCSP3…")
        self.ed_calc_more.setToolTip(
            "<p>Any other RDKit descriptor, by its own name, comma separated — "
            "<i>NumHDonors</i>, <i>NumAromaticRings</i>, <i>FractionCSP3</i>, <i>BalabanJ</i>, "
            "<i>LabuteASA</i>, <i>qed</i>, and some two hundred more.</p>"
            "<p>A name that does not exist is reported with the nearest ones that do, when the "
            "model is built.</p>")
        self.ed_calc_more.editingFinished.connect(self.fields_changed)
        try:
            from ..descriptors import rdkit_descriptors
            comp = QCompleter(sorted(rdkit_descriptors()))
            comp.setCaseSensitivity(Qt.CaseInsensitive)
            comp.setFilterMode(Qt.MatchContains)
            self.ed_calc_more.setCompleter(comp)
        except Exception:                            # noqa: BLE001 - completion is a convenience
            pass
        crow3.addWidget(self.ed_calc_more, 1)
        fl.addLayout(crow3)
        # non-field descriptors: one number per molecule, added to the kernel as its own block
        self.row_desc = QWidget()
        self.lay_desc = QHBoxLayout(self.row_desc)
        self.lay_desc.setContentsMargins(0, 0, 0, 0)
        self.lbl_desc = QLabel("Descriptors (per molecule):")
        self.lay_desc.addWidget(self.lbl_desc)
        self.lay_desc.addStretch(1)
        self.spin_desc_h = QDoubleSpinBox()
        self.spin_desc_h.setRange(0.0, 100.0)
        self.spin_desc_h.setSingleStep(0.25)
        self.spin_desc_h.setDecimals(2)
        self.spin_desc_h.setValue(1.0)
        self.spin_desc_h.setPrefix("h = ")
        self.spin_desc_h.setToolTip(
            "<p>Mixing coefficient of the descriptor block, the counterpart of a field's h. The "
            "block is standardised on the training set and scaled to unit mean diagonal, so "
            "h = 1 puts it on the same footing as one field; 0 switches it off.</p>")
        self.lay_desc.addWidget(self.spin_desc_h)
        self.row_desc.setVisible(False)
        self._desc_boxes: Dict[str, QCheckBox] = {}
        fl.addWidget(self.row_desc)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Partial charges:"))
        self.cmb_charges = QComboBox()
        self.cmb_charges.addItems(["gasteiger", "mmff", "file"])
        self.cmb_charges.currentIndexChanged.connect(self.fields_changed)
        prow.addWidget(self.cmb_charges)
        prow.addWidget(QLabel("Increments:"))
        self.cmb_param = QComboBox()
        self.cmb_param.addItems(["rdkit", "tripos", "tripos-legacy"])
        self.cmb_param.setToolTip("Parameterisation of the hydrophobic and H-bond fields:\n"
                                  "rdkit - Wildman-Crippen logP contributions and built-in H-bond rules\n"
                                  "tripos - Sybyl-type increments of the original CMF scripts (2013 paper)\n"
                                  "tripos-legacy - as tripos, reproducing the H-count bug of the 2013 scripts")
        self.cmb_param.currentIndexChanged.connect(self.fields_changed)
        prow.addWidget(self.cmb_param)
        self.chk_normalize = QCheckBox("Normalise kernels")
        self.chk_normalize.setToolTip("Use cosine-normalised field kernels K/sqrt(Kii Kjj)")
        self.chk_scale = QCheckBox("Scale fields")
        self.chk_scale.setChecked(True)
        self.chk_scale.setToolTip("Rescale every field kernel to unit mean diagonal before mixing\n"
                                  "(uncheck to reproduce the raw-kernel convention of the papers)")
        # the two kernel conventions sit under the parameterisation rather than beside it,
        # so that the row does not dictate how wide the whole dock has to be

        prow2 = QHBoxLayout()
        prow2.addWidget(QLabel("Kernel conventions:"))
        prow2.addWidget(self.chk_normalize)
        prow2.addWidget(self.chk_scale)
        prow2.addStretch(1)
        fl.addLayout(prow2)
        fl.addLayout(prow)

        # --- conformations: the fourth dimension
        crow2 = QHBoxLayout()
        crow2.addWidget(QLabel("Conformations:"))
        self.cmb_conformers = QComboBox()
        self.cmb_conformers.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_conformers.setMinimumContentsLength(18)
        self.cmb_conformers.addItems(["single (one per molecule)",
                                      "ensemble (4D: mean field over conformations)",
                                      "ensemble, weights learned (multi-instance)"])
        self.cmb_conformers.setToolTip("\n".join([
            "Single: every molecule is one conformation, as in the papers. This is the default",
            "  and nothing about it changes.",
            "Ensemble: a molecule is the weighted mean of the fields of its conformations, which",
            "  is the occupancy averaging of 4D-QSAR. The kernel of two ensembles becomes the",
            "  weighted mean of the kernels of their conformation pairs, so co-fields and the",
            "  decomposition rules still hold, but the kernels cost k^2 times as much.",
            "Load a multi-conformer file, or generate conformations from Tools."]))
        self.cmb_conformers.currentIndexChanged.connect(self._conformers_changed)
        crow2.addWidget(self.cmb_conformers)
        self.cmb_conf_weight = QComboBox()
        self.cmb_conf_weight.addItems(["uniform weights", "Boltzmann weights"])
        self.cmb_conf_weight.setToolTip("Boltzmann weights need per-conformer energies, which the "
                                        "conformer generator stores;\nwithout them the weights "
                                        "stay uniform")
        self.cmb_conf_weight.currentIndexChanged.connect(self.fields_changed)
        crow2.addWidget(self.cmb_conf_weight)
        self.spin_conf_kt = QDoubleSpinBox()
        self.spin_conf_kt.setRange(0.05, 20.0)
        self.spin_conf_kt.setSingleStep(0.1)
        self.spin_conf_kt.setValue(0.6)
        self.spin_conf_kt.setPrefix("kT ")
        self.spin_conf_kt.setSuffix(" kcal")
        self.spin_conf_kt.valueChanged.connect(self.fields_changed)
        crow2.addWidget(self.spin_conf_kt)
        self.spin_conf_max = QSpinBox()
        self.spin_conf_max.setRange(0, 999)
        self.spin_conf_max.setValue(0)
        self.spin_conf_max.setPrefix("max ")
        self.spin_conf_max.setToolTip("Use at most this many of the heaviest conformations "
                                      "per molecule; 0 uses all of them")
        self.spin_conf_max.valueChanged.connect(self.fields_changed)
        crow2.addWidget(self.spin_conf_max)
        self.spin_mil_beta = QDoubleSpinBox()
        self.spin_mil_beta.setRange(0.0, 100.0)
        self.spin_mil_beta.setSingleStep(1.0)
        self.spin_mil_beta.setValue(3.0)
        self.spin_mil_beta.setPrefix("β ")
        self.spin_mil_beta.setToolTip("\n".join([
            "How sharply the model prefers the conformations it scores highest.",
            "Zero is the plain ensemble; a large value puts nearly all the weight on one",
            "conformation, which is the witness assumption of multi-instance learning."]))
        self.spin_mil_beta.valueChanged.connect(self.fields_changed)
        # the conformation settings do not fit one line without forcing the whole dock wide,
        # so the multi-instance half sits on its own
        crow2.addStretch(1)
        fl.addLayout(crow2)
        crow2b = QHBoxLayout()
        crow2b.addWidget(QLabel("Multi-instance:"))
        crow2b.addWidget(self.spin_mil_beta)
        self.chk_mil_hard = QCheckBox("one conformation")
        self.chk_mil_hard.setToolTip("Choose exactly one conformation per molecule instead of "
                                     "weighting them")
        self.chk_mil_hard.toggled.connect(self.fields_changed)
        crow2b.addWidget(self.chk_mil_hard)
        self.lbl_conformers = QLabel("")
        self.lbl_conformers.setStyleSheet("color: palette(mid)")
        crow2b.addWidget(self.lbl_conformers)
        crow2b.addStretch(1)
        fl.addLayout(crow2b)

        self.tbl_fields = QTableWidget(0, 5)
        self.tbl_fields.setHorizontalHeaderLabels(["Use", "Field", "alpha", "h (mix)", "scale"])
        self.tbl_fields.verticalHeader().setVisible(False)
        self.tbl_fields.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_fields.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_fields.setMinimumHeight(140)
        fl.addWidget(self.tbl_fields)
        outer.addWidget(gb)

        # --- learner / hyper-parameters ------------------------------------
        gb2 = QGroupBox("Learning method and hyper-parameters")
        form = QFormLayout(gb2)
        # a long label must not force the whole dock wide: let it sit above its editor
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["Regression", "Binary classification", "One-class (virtual screening)"])
        self.cmb_task.setToolTip("Binary classification separates active from inactive compounds. "
                                 "A one-class model is fitted to the actives alone and scores how well "
                                 "a molecule fits them (similarity-based virtual screening).")
        self.cmb_task.currentIndexChanged.connect(self._task_changed)
        form.addRow("Task:", self.cmb_task)
        crow = QHBoxLayout()
        self.cmb_class_src = QComboBox()
        self.cmb_class_src.addItems(["activity threshold", "SD property"])
        self.cmb_class_src.currentIndexChanged.connect(self._task_changed)
        crow.addWidget(self.cmb_class_src)
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(-1e6, 1e6)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(0.0)
        crow.addWidget(self.spin_threshold)
        self.chk_active_below = QCheckBox("active below")
        self.chk_active_below.setToolTip("Tick when a smaller value means more active (IC50 in nM, for example)")
        crow.addWidget(self.chk_active_below)
        self.cmb_class_prop = QComboBox()
        self.cmb_class_prop.setMinimumWidth(110)
        crow.addWidget(self.cmb_class_prop)
        self.row_classes = crow
        form.addRow("Classes:", crow)
        self.cmb_method = QComboBox()
        self.cmb_method.addItems(["Kernel ridge regression (KRR)", "Gaussian process (KRR + variance)",
                                  "epsilon-SVR", "nu-SVR"])
        form.addRow("Method:", self.cmb_method)
        self.cmb_kernel = QComboBox()
        self.cmb_kernel.addItems(["none: linear CMF kernel (overlap integral)",
                                  "Gaussian layer (RBF on the fields)",
                                  "arc-cosine layer", "two arc-cosine layers"])
        self.cmb_kernel.setToolTip("\n".join([
            "A nonlinear layer on top of the CMF kernel, which makes the model more",
            "expressive than the overlap integral alone:",
            "  Gaussian - a radial basis function of the distance between the fields,",
            "    exp(-gamma d^2) with d^2 = K(i,i) + K(j,j) - 2 K(i,j);",
            "  arc-cosine - the kernel of an infinitely wide network with one or two",
            "    hidden layers (Cho & Saul 2009).",
            "The model is then no longer linear in the molecular field, so co-fields and",
            "atom contributions are not available; the molecular similarity decomposition",
            "(rule 2) stays valid and the Shapley tab decomposes the prediction instead."]))
        self.cmb_kernel.currentIndexChanged.connect(self._kernel_changed)
        form.addRow("Nonlinear kernel:", self.cmb_kernel)
        self.ed_gammas = QLineEdit("0.25, 0.5, 1, 2, 4")
        self.ed_gammas.setToolTip("\n".join([
            "Bandwidth grid of the Gaussian layer, in units of the inverse median squared",
            "distance between the training molecules (1 = the median heuristic)"]))
        self.lbl_gammas = QLabel("γ grid (Gaussian):")
        form.addRow(self.lbl_gammas, self.ed_gammas)
        self.cmb_mix = QComboBox()
        self.cmb_mix.addItems(["fixed (from table, default 1)", "optimise by coordinate descent",
                               "kernel alignment, per field (MKL)",
                               "kernel alignment, joint (MKL)"])
        self.cmb_mix.setToolTip("\n".join([
            "How the mixing coefficients h_f of the field types are chosen; the values in use are",
            "shown in the h (mix) column of the field table above.",
            "Coordinate descent increases q2 but may cause model-selection bias.",
            "Multiple kernel learning sets them by centred alignment of the field kernels with",
            "the activities: per field (Cristianini et al. 2001) or jointly (Cortes et al. 2012,",
            "shrunk towards equal weights). The joint kernel stays linear, so co-fields remain",
            "valid; the weights are re-derived inside every cross-validation fold, so q2 is honest.",
            "It pays off when the fields differ in relevance, e.g. together with indicator fields."]))
        form.addRow("Mixing coeff. h:", self.cmb_mix)
        arow = QHBoxLayout()
        self.rad_alpha_grid = QRadioButton("grid search (shared)")
        self.rad_alpha_grid.setChecked(True)
        self.rad_alpha_table = QRadioButton("fixed from table")
        arow.addWidget(self.rad_alpha_grid)
        arow.addWidget(self.rad_alpha_table)
        form.addRow("Attenuation α:", arow)
        self.cmb_kernel_src = QComboBox()
        self.cmb_kernel_src.addItems(["Overlap integrals of the fields (needs one frame)",
                                      "Field autocorrelation (alignment-free, rotation-invariant)"])
        self.cmb_kernel_src.setToolTip("\n".join([
            "Where the kernel comes from.",
            "  Overlap integrals: the CMF kernel; co-fields and atom contributions are defined,",
            "    and the molecules must share a frame.",
            "  Field autocorrelation: the distance distribution of the field weights inside each",
            "    molecule, which no rigid motion changes. No alignment, no reference molecule;",
            "    the attenuation grid is not used and there are no co-fields."]))
        form.addRow("Kernel from:", self.cmb_kernel_src)
        self.spin_ac_sigma = QDoubleSpinBox()
        self.spin_ac_sigma.setRange(0.1, 3.0)
        self.spin_ac_sigma.setSingleStep(0.05)
        self.spin_ac_sigma.setDecimals(2)
        self.spin_ac_sigma.setValue(0.75)
        self.spin_ac_sigma.setSuffix(" A")
        self.spin_ac_sigma.setToolTip("Width along the distance axis of the autocorrelation, "
                                      "which also sets the spacing of its channels")
        self.lbl_ac_sigma = QLabel("Distance resolution σ:")
        form.addRow(self.lbl_ac_sigma, self.spin_ac_sigma)
        self.chk_ac_chirality = QCheckBox("Signed three-point channels (chirality)")
        self.chk_ac_chirality.setToolTip(
            "A distance distribution is the same for a molecule and its mirror image, so the "
            "plain autocorrelation cannot tell enantiomers apart. This adds the signed volume "
            "spanned by three field-weighted moments, which changes sign under reflection.")
        form.addRow("", self.chk_ac_chirality)
        self.ed_alphas = QLineEdit("0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0")
        form.addRow("α grid:", self.ed_alphas)
        self.ed_lams = QLineEdit("1e-4, 1e-3, 1e-2, 0.1, 1, 10")
        form.addRow("λ grid (KRR/GP):", self.ed_lams)
        self.ed_Cs = QLineEdit("0.1, 1, 10, 100")
        self.ed_Cs.setToolTip("C of the support-vector methods; for the Gaussian-process "
                              "classifier the same grid is the amplitude of the kernel, which "
                              "plays the same part and can also be read off the marginal "
                              "likelihood printed in the Statistics tab")
        self.ed_Cs.setToolTip("C for the support vector machines, and the amplitude of the "
                              "Gaussian-process classifier")
        form.addRow("C grid:", self.ed_Cs)
        self.ed_eps = QLineEdit("0.05, 0.1, 0.2")
        form.addRow("ε grid / ν grid:", self.ed_eps)
        self.ed_nus = QLineEdit("0.3, 0.5, 0.7")
        form.addRow("ν grid (ν-SVR):", self.ed_nus)
        self.spin_oc_param = QSpinBox()
        self.spin_oc_param.setRange(1, 200)
        self.spin_oc_param.setValue(5)
        self.lbl_oc_param = QLabel("Neighbours k:")
        form.addRow(self.lbl_oc_param, self.spin_oc_param)
        # the two fusion rules that count matches need a threshold, and on a field similarity
        # the literature's default of zero makes that count a constant: see the README
        self.spin_fusion_thr = QDoubleSpinBox()
        self.spin_fusion_thr.setRange(-1.0, 1.0)
        self.spin_fusion_thr.setDecimals(3)
        self.spin_fusion_thr.setSingleStep(0.05)
        self.spin_fusion_thr.setValue(0.0)
        self.spin_fusion_thr.setToolTip(
            "A similarity above this counts as a match. At 0 every active matches every molecule "
            "on a field kernel, the count is constant, and ANZ and MNZ become SUM exactly; take a "
            "value from inside the range the similarities actually span.")
        self.lbl_fusion_thr = QLabel("Match threshold:")
        form.addRow(self.lbl_fusion_thr, self.spin_fusion_thr)
        cvrow = QHBoxLayout()
        self.spin_folds = QSpinBox()
        self.spin_folds.setRange(2, 100)
        self.spin_folds.setValue(10)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(0)
        self.spin_seed.setPrefix("seed ")
        cvrow.addWidget(self.spin_folds)
        cvrow.addWidget(QLabel("-fold"))
        cvrow.addWidget(self.spin_seed)
        form.addRow("Cross-validation:", cvrow)
        nrow = QHBoxLayout()
        self.spin_outer = QSpinBox()
        self.spin_outer.setRange(2, 50)
        self.spin_outer.setValue(5)
        self.spin_outer.setPrefix("outer ")
        self.spin_repeats = QSpinBox()
        self.spin_repeats.setRange(1, 1000)
        self.spin_repeats.setValue(1)
        self.spin_repeats.setPrefix("repeats ")
        nrow.addWidget(self.spin_outer)
        nrow.addWidget(self.spin_repeats)
        form.addRow("External CV:", nrow)
        vrow = QHBoxLayout()
        self.spin_scramble = QSpinBox()
        self.spin_scramble.setRange(0, 1000)
        self.spin_scramble.setValue(0)
        self.spin_scramble.setToolTip("\n".join([
            "y-randomisation: refit the model on this many shuffles of the activities and",
            "report where the real q2 falls among them. 0 = off; 20 is enough to recognise a",
            "chance model, 100 gives a p-value worth quoting. Unless the box beside it is",
            "ticked the hyper-parameters are kept, so it asks whether the fit can be reached",
            "by chance, not whether the whole search can."]))
        vrow.addWidget(QLabel("shuffles"))
        vrow.addWidget(self.spin_scramble)
        self.chk_scramble_select = QCheckBox("redo the search")
        self.chk_scramble_select.setToolTip("\n".join([
            "Repeat the whole alpha/lambda search inside every shuffle, so that the test covers",
            "the model selection as well as the fit. This is the stricter and more honest",
            "question, and it raises the scrambled baseline noticeably, because a search over a",
            "grid can pick the luckiest point on noise. Costs one cross-validation per grid",
            "point per shuffle; the kernels are computed once."]))
        vrow.addWidget(self.chk_scramble_select)
        self.spin_boot = QSpinBox()
        self.spin_boot.setRange(0, 10000)
        self.spin_boot.setSingleStep(100)
        self.spin_boot.setValue(0)
        self.spin_boot.setToolTip("\n".join([
            "Bootstrap confidence intervals for q2 and R2pred, by resampling the (observed,",
            "predicted) pairs. 0 = off; 500 to 1000 is usual. It says how much of the",
            "statistic is the particular molecules it was computed over, which is what decides",
            "whether two settings differing by 0.02 differ at all."]))
        vrow.addWidget(QLabel(" bootstrap"))
        vrow.addWidget(self.spin_boot)
        self.chk_domain = QCheckBox("applicability domain")
        self.chk_domain.setToolTip("\n".join([
            "Measure how close every molecule outside the training set is to it, in the kernel",
            "the model itself uses, against a threshold taken from the training set's own",
            "spread. Molecules below it are listed in the log: their predictions are",
            "extrapolations, however confident they look."]))
        vrow.addWidget(self.chk_domain)
        vrow.addStretch(1)
        form.addRow("Validation:", vrow)
        outer.addWidget(gb2)

        # --- buttons ------------------------------------------------------
        brow = QHBoxLayout()
        self.btn_build = QPushButton("Build model")
        self.btn_build.setStyleSheet("font-weight: bold")
        self.btn_nested = QPushButton("External CV")
        self.btn_predict = QPushButton("Predict test set")
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        brow.addWidget(self.btn_build)
        brow.addWidget(self.btn_nested)
        brow.addWidget(self.btn_predict)
        brow.addWidget(self.btn_cancel)
        outer.addLayout(brow)
        self.btn_build.clicked.connect(self.build_requested)
        self.btn_nested.clicked.connect(self.nested_requested)
        self.btn_predict.clicked.connect(self.predict_requested)
        self.btn_cancel.clicked.connect(self.cancel_requested)
        outer.addStretch(1)
        self._conformers_changed()  # the weighting controls belong to the ensemble mode only
        # the one-class parameter box depends on which description is chosen
        self.cmb_method.currentIndexChanged.connect(self._task_changed)
        self.cmb_kernel_src.currentIndexChanged.connect(self._task_changed)
        self._task_changed()      # hide the class controls until a classification task is chosen
        self._kernel_changed()    # the bandwidth grid belongs to the Gaussian layer only

    # ---------------------------------------------------------------- API
    def conformer_mode(self) -> str:
        """The field-set mode; the learned variant is an ensemble whose weights are fitted."""
        return ["single", "ensemble", "ensemble"][self.cmb_conformers.currentIndex()]

    def learns_weights(self) -> bool:
        return self.cmb_conformers.currentIndex() == 2

    def mil_beta(self) -> float:
        return float(self.spin_mil_beta.value())

    def mil_selection(self) -> str:
        return "hard" if self.chk_mil_hard.isChecked() else "soft"

    def conformer_weighting(self) -> str:
        return ["uniform", "boltzmann"][self.cmb_conf_weight.currentIndex()]

    def _conformers_changed(self, *_):
        learned = self.learns_weights()
        on = self.conformer_mode() == "ensemble"
        for w in (self.cmb_conf_weight, self.spin_conf_kt):
            w.setVisible(on and not learned)     # the fit decides the weights in the learned mode
        self.spin_conf_max.setVisible(on)
        for w in (self.spin_mil_beta, self.chk_mil_hard):
            w.setVisible(learned)
        self.fields_changed.emit()

    def set_conformer_info(self, text: str) -> None:
        self.lbl_conformers.setText(text)

    def selected_physchem(self) -> List[str]:
        return [n for n, cb in self.chk_fields.items() if cb.isChecked()]

    REG_METHODS = ["Kernel ridge regression (KRR)", "Gaussian process (KRR + variance)",
                   "epsilon-SVR", "nu-SVR"]
    REG_KEYS = ["krr", "gp", "svr", "nusvr"]
    CLS_METHODS = ["Support vector classification (C-SVC)", "nu-SVC",
                   "Gaussian process classifier (Laplace)"]
    CLS_KEYS = ["svc", "nusvc", "gpc"]
    OC_METHODS = ["One-class SVM (1-SVM)",
                  "Distance to the centre of the actives",
                  "Distance to the k-th nearest active",
                  "One-class Gaussian process",
                  "Distance from the principal subspace (kernel PCA)"]
    OC_METHODS = OC_METHODS + [
        "Fusion: best-matching active (MAX)",
        "Fusion: worst-matching active (MIN)",
        "Fusion: sum of the similarities (SUM)",
        "Fusion: median of the similarities (MED)",
        "Fusion: sum over the actives that match (ANZ)",
        "Fusion: sum times the number that match (MNZ)",
        "Fusion: Euclidean norm of the row (EUC)",
        "Fusion: reciprocal rank fusion (RRF)"]
    OC_KEYS = ["ocsvm", "ocmean", "ocknn", "ocgp", "ockpca",
               "fuse-max", "fuse-min", "fuse-sum", "fuse-med",
               "fuse-anz", "fuse-mnz", "fuse-euc", "fuse-rrf"]

    def kernel_source(self) -> str:
        return ["fields", "autocorrelation"][self.cmb_kernel_src.currentIndex()]

    def ac_sigma(self) -> float:
        return float(self.spin_ac_sigma.value())

    def ac_chirality(self) -> bool:
        return bool(self.chk_ac_chirality.isChecked())

    def task_key(self) -> str:
        return ["regression", "classification", "oneclass"][self.cmb_task.currentIndex()]

    KERNEL_KEYS = ["linear", "gauss", "arccos1", "arccos2"]

    def kernel_layer(self) -> str:
        return self.KERNEL_KEYS[self.cmb_kernel.currentIndex()]

    def mkl_mode(self):
        """False, "align" or "alignf" - the multiple-kernel-learning variant chosen."""
        return [False, False, "align", "alignf"][self.cmb_mix.currentIndex()]

    def gammas(self) -> List[float]:
        return _float_list(self.ed_gammas.text(), [1.0])

    def _kernel_changed(self, *_):
        gauss = self.kernel_layer() == "gauss"
        self.ed_gammas.setVisible(gauss)
        self.lbl_gammas.setVisible(gauss)

    def _task_changed(self, *_):
        """Offer the methods and the class controls that belong to the selected task."""
        task = self.task_key()
        methods = {"regression": self.REG_METHODS, "classification": self.CLS_METHODS,
                   "oneclass": self.OC_METHODS}[task]
        cur = self.cmb_method.currentText()
        self.cmb_method.blockSignals(True)
        self.cmb_method.clear()
        self.cmb_method.addItems(methods)
        if cur in methods:
            self.cmb_method.setCurrentText(cur)
        self.cmb_method.blockSignals(False)
        classes = task in ("classification", "oneclass")
        by_prop = self.cmb_class_src.currentIndex() == 1
        for w in (self.cmb_class_src,):
            w.setEnabled(classes)
        self.spin_threshold.setVisible(classes and not by_prop)
        self.chk_active_below.setVisible(classes and not by_prop)
        self.cmb_class_prop.setVisible(classes and by_prop)
        self.cmb_class_src.setVisible(classes)
        # one-class has no cross-validated objective: alpha and nu are chosen directly
        oc = task == "oneclass"
        if oc and not self.chk_normalize.isChecked():
            # on raw kernels the one-class score is dominated by molecular size
            self.chk_normalize.setChecked(True)
        # a one-class model has no cross-validation objective, so the first value of the alpha
        # box is used directly as the level of generalisation rather than searched
        self.ed_alphas.setToolTip("One-class models use the first value as the level of generalisation "
                                  "(no cross-validated search is possible without negatives)" if oc else "")
        # the extra parameter of the one-class descriptions, which means something different
        # for each of them and nothing for two of them
        key = self.method_key() if oc else ""
        labels = {"ocknn": ("Neighbours k:", 1, 200, 5),
                  "ockpca": ("Subspace dimension:", 1, 200, 5)}
        show = key in labels
        self.lbl_oc_param.setVisible(show)
        self.spin_oc_param.setVisible(show)
        if show:
            text, lo, hi, default = labels[key]
            self.lbl_oc_param.setText(text)
            self.spin_oc_param.setRange(lo, hi)
            if self.spin_oc_param.value() not in range(lo, hi + 1):
                self.spin_oc_param.setValue(default)
        fusion_thr = key in ("fuse-anz", "fuse-mnz")
        self.lbl_fusion_thr.setVisible(fusion_thr)
        self.spin_fusion_thr.setVisible(fusion_thr)
        self.ed_lams.setToolTip("The one-class Gaussian process takes the first value as its "
                                "ridge on the constant target" if key == "ocgp" else "")
        # the autocorrelation kernel has its own width instead of the attenuation
        invariant = self.kernel_source() != "fields"
        self.lbl_ac_sigma.setVisible(invariant)
        self.spin_ac_sigma.setVisible(invariant)
        self.chk_ac_chirality.setVisible(invariant)
        self.ed_alphas.setEnabled(not invariant)
        self.cmb_mix.setEnabled(not oc)
        self.spin_outer.setEnabled(task == "regression")
        self.spin_repeats.setEnabled(task == "regression")
        self.btn_nested.setEnabled(task == "regression")
        self.fields_changed.emit()

    def set_atom_properties(self, props) -> None:
        """Offer one field per per-atom property the data set carries (none: hide the row)."""
        keep = {n for n, b in self._prop_boxes.items() if b.isChecked()}
        for b in self._prop_boxes.values():
            self.lay_props.removeWidget(b)
            b.setParent(None)
            b.deleteLater()
        self._prop_boxes = {}
        for i, name in enumerate(list(props)):
            box = QCheckBox(str(name))
            box.setToolTip(f"A field whose atomic weights are the values of {name!r} carried by "
                           f"the file, one per atom")
            box.setChecked(name in keep)
            box.toggled.connect(self.fields_changed)
            self.lay_props.insertWidget(i + 1, box)
            self._prop_boxes[str(name)] = box
        self.row_props.setVisible(bool(self._prop_boxes))

    def atom_properties(self) -> List[str]:
        return [n for n, b in self._prop_boxes.items() if b.isChecked()]

    def set_descriptor_properties(self, props) -> None:
        """Offer every numeric molecule property as a non-field descriptor (none: hide the row)."""
        keep = {n for n, b in self._desc_boxes.items() if b.isChecked()}
        for b in self._desc_boxes.values():
            self.lay_desc.removeWidget(b)
            b.setParent(None)
            b.deleteLater()
        self._desc_boxes = {}
        for i, name in enumerate(list(props)):
            box = QCheckBox(str(name))
            box.setToolTip(
                f"<p>Use {name!r} as a descriptor: one number per molecule, standardised on the "
                f"training set and added to the kernel beside the fields.</p>"
                f"<p>For what no field can represent - a measured property, a medium, a "
                f"temperature. Never something derived from the activity: it would dominate the "
                f"model and the geometry would look worthless.</p>")
            box.setChecked(name in keep)
            box.toggled.connect(self.fields_changed)
            self.lay_desc.insertWidget(i + 1, box)
            self._desc_boxes[str(name)] = box
        self.row_desc.setVisible(bool(self._desc_boxes))

    def probes(self) -> List[str]:
        return [n for n, b in self._probe_boxes.items() if b.isChecked()]

    def probe_terms(self) -> int:
        return int(self.spin_probe_terms.value())

    def descriptors(self) -> List[str]:
        """Both kinds: the properties the file carries and the ones the program computes."""
        out = [n for n, b in self._desc_boxes.items() if b.isChecked()]
        out += [f"calc:{n}" for n, b in self._calc_boxes.items() if b.isChecked()]
        for name in self.ed_calc_more.text().replace(";", ",").split(","):
            name = name.strip()
            if name and f"calc:{name}" not in out:
                out.append(f"calc:{name}")
        return out

    def descriptor_h(self) -> float:
        return float(self.spin_desc_h.value())

    def set_class_properties(self, props):
        cur = self.cmb_class_prop.currentText()
        self.cmb_class_prop.blockSignals(True)
        self.cmb_class_prop.clear()
        self.cmb_class_prop.addItems(list(props))
        if cur in props:
            self.cmb_class_prop.setCurrentText(cur)
        self.cmb_class_prop.blockSignals(False)

    def method_key(self) -> str:
        task = self.task_key()
        keys = {"regression": self.REG_KEYS, "classification": self.CLS_KEYS,
                "oneclass": self.OC_KEYS}[task]
        return keys[min(self.cmb_method.currentIndex(), len(keys) - 1)]

    def oneclass_params(self) -> dict:
        """The extra parameters of the one-class descriptions, by method."""
        key = self.method_key()
        return {"k": int(self.spin_oc_param.value()),
                "components": int(self.spin_oc_param.value()),
                "oc_lam": float(self.lams()[0]) if key == "ocgp" else 0.1,
                "fusion_threshold": float(self.spin_fusion_thr.value()),
                "rrf_k": 60.0}

    def alphas(self) -> List[float]:
        return _float_list(self.ed_alphas.text(), [1.0])

    def lams(self) -> List[float]:
        return _float_list(self.ed_lams.text(), [1.0])

    def Cs(self) -> List[float]:
        return _float_list(self.ed_Cs.text(), [1.0])

    def epsilons(self) -> List[float]:
        return _float_list(self.ed_eps.text(), [0.1])

    def nus(self) -> List[float]:
        return _float_list(self.ed_nus.text(), [0.5])

    def set_busy(self, busy: bool) -> None:
        for b in (self.btn_build, self.btn_nested, self.btn_predict):
            b.setEnabled(not busy)
        self.btn_cancel.setEnabled(busy)

    def fill_field_table(self, fieldset: FieldSet, scale: Optional[np.ndarray] = None) -> None:
        """Populate the field table from a FieldSet, keeping user edits for same-named fields."""
        old = self.table_values()
        self.tbl_fields.blockSignals(True)
        self.tbl_fields.setRowCount(0)
        self.tbl_fields.setRowCount(len(fieldset.specs))
        for i, s in enumerate(fieldset.specs):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            prev = old.get(s.name)
            chk.setCheckState(Qt.Checked if (prev["enabled"] if prev else s.enabled) else Qt.Unchecked)
            self.tbl_fields.setItem(i, 0, chk)
            nm = QTableWidgetItem(s.label)
            nm.setFlags(Qt.ItemIsEnabled)
            nm.setData(Qt.UserRole, s.name)
            self.tbl_fields.setItem(i, 1, nm)
            self.tbl_fields.setItem(i, 2, QTableWidgetItem(f"{(prev['alpha'] if prev else s.alpha):g}"))
            self.tbl_fields.setItem(i, 3, QTableWidgetItem(f"{(prev['h'] if prev else s.h):g}"))
            sc = QTableWidgetItem("" if scale is None else f"{scale[i]:.3g}")
            sc.setFlags(Qt.ItemIsEnabled)
            self.tbl_fields.setItem(i, 4, sc)
        self.tbl_fields.blockSignals(False)

    def show_fitted_values(self, fieldset: FieldSet, scale: Optional[np.ndarray]) -> None:
        self.tbl_fields.blockSignals(True)
        for i, s in enumerate(fieldset.specs):
            if i >= self.tbl_fields.rowCount():
                break
            self.tbl_fields.item(i, 2).setText(f"{s.alpha:g}")
            self.tbl_fields.item(i, 3).setText(f"{s.h:g}")
            scaled = scale is not None and i < len(np.atleast_1d(scale))
            self.tbl_fields.item(i, 4).setText(f"{np.atleast_1d(scale)[i]:.3g}" if scaled else "")
        self.tbl_fields.blockSignals(False)

    def table_values(self) -> Dict[str, dict]:
        out = {}
        for i in range(self.tbl_fields.rowCount()):
            name = self.tbl_fields.item(i, 1).data(Qt.UserRole)
            try:
                alpha = float(self.tbl_fields.item(i, 2).text())
            except (ValueError, AttributeError):
                alpha = 1.0
            try:
                h = float(self.tbl_fields.item(i, 3).text())
            except (ValueError, AttributeError):
                h = 1.0
            out[name] = {"enabled": self.tbl_fields.item(i, 0).checkState() == Qt.Checked, "alpha": alpha, "h": h}
        return out


# --------------------------------------------------------------------------- #
# Results panel (statistics, search table, scatter plot)
# --------------------------------------------------------------------------- #

class ResultsPanel(QWidget):
    point_clicked = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.tabs = QTabWidget()
        lay.addWidget(self.tabs)

        self.txt_stats = QTextEdit()
        self.txt_stats.setReadOnly(True)
        self.txt_stats.setFontFamily("Consolas")
        self.tabs.addTab(self.txt_stats, "Statistics")

        pg.setConfigOptions(antialias=True, background="w", foreground="k")
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Experimental")
        self.plot.setLabel("left", "Predicted")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.legend = self.plot.addLegend(offset=(10, 10))
        self.sc_cv = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(60, 120, 220, 180), pen=pg.mkPen("k", width=0.5),
                                        name="training (CV)", hoverable=True, tip=self._tip)
        self.sc_test = pg.ScatterPlotItem(size=10, brush=pg.mkBrush(230, 80, 60, 200), pen=pg.mkPen("k", width=0.5),
                                          symbol="s", name="test set", hoverable=True, tip=self._tip)
        self.sc_ex = pg.ScatterPlotItem(size=9, brush=pg.mkBrush(40, 170, 90, 180), pen=pg.mkPen("k", width=0.5),
                                        symbol="t", name="external CV", hoverable=True, tip=self._tip)
        self.line = pg.InfiniteLine(angle=45, pen=pg.mkPen((120, 120, 120), style=Qt.DashLine))
        self.plot.addItem(self.line)
        # classification: the class threshold on the activity and the decision boundary, whose
        # quadrants are the true/false positives and negatives
        self.vline = pg.InfiniteLine(angle=90, pen=pg.mkPen((120, 120, 120), style=Qt.DashLine))
        self.hline = pg.InfiniteLine(angle=0, pos=0.0, pen=pg.mkPen((120, 120, 120), style=Qt.DashLine))
        for ln in (self.vline, self.hline):
            self.plot.addItem(ln)
            ln.setVisible(False)
        for sc in (self.sc_cv, self.sc_test, self.sc_ex):
            self.plot.addItem(sc)
            sc.sigClicked.connect(self._clicked)
        self.tab_scatter = self.tabs.addTab(self.plot, "Scatter plot")

        self.roc = pg.PlotWidget()
        self.roc.setLabel("bottom", "false positive rate")
        self.roc.setLabel("left", "true positive rate")
        self.roc.showGrid(x=True, y=True, alpha=0.3)
        self.roc.addLegend(offset=(-10, 10))
        self.roc.setXRange(0, 1)
        self.roc.setYRange(0, 1)
        self.roc.plot([0, 1], [0, 1], pen=pg.mkPen((140, 140, 140), style=Qt.DashLine))
        self._roc_items = []
        self.tab_roc = self.tabs.addTab(self.roc, "ROC")
        self.tabs.setTabVisible(self.tab_roc, False)

        # --- scores (decision values): the actual output of a classification / screening model
        self.tab_scores_w = QWidget()
        sl = QVBoxLayout(self.tab_scores_w)
        sl.setContentsMargins(0, 0, 0, 0)
        self.score_hist = pg.PlotWidget()
        self.score_hist.setLabel("bottom", "decision value")
        self.score_hist.setLabel("left", "molecules")
        self.score_hist.showGrid(x=True, y=True, alpha=0.3)
        self.score_hist.addLegend(offset=(-10, 10))
        self.score_hist.setMaximumHeight(220)
        self._score_items = []
        sl.addWidget(self.score_hist)
        self.tbl_scores = QTableWidget(0, 5)
        self.tbl_scores.setHorizontalHeaderLabels(["Rank", "Molecule", "Score", "Set", "Class"])
        self.tbl_scores.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_scores.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl_scores.verticalHeader().setVisible(False)
        sl.addWidget(self.tbl_scores)
        self.tab_scores = self.tabs.addTab(self.tab_scores_w, "Scores")
        self.tabs.setTabVisible(self.tab_scores, False)

        self.enrich = pg.PlotWidget()
        self.enrich.setLabel("bottom", "fraction of the library screened")
        self.enrich.setLabel("left", "fraction of the actives found")
        self.enrich.showGrid(x=True, y=True, alpha=0.3)
        self.enrich.addLegend(offset=(-10, 10))
        self.enrich.setXRange(0, 1)
        self.enrich.setYRange(0, 1)
        self.enrich.plot([0, 1], [0, 1], pen=pg.mkPen((140, 140, 140), style=Qt.DashLine))
        self._enrich_items = []
        self.tab_enrich = self.tabs.addTab(self.enrich, "Enrichment")
        self.tabs.setTabVisible(self.tab_enrich, False)

        # --- chemical space map (GTM)
        self.map_plot = pg.PlotWidget()
        self.map_plot.setLabel("bottom", "latent coordinate 1")
        self.map_plot.setLabel("left", "latent coordinate 2")
        self.map_plot.setAspectLocked(True)
        self.map_plot.showGrid(x=True, y=True, alpha=0.2)
        self.map_img = pg.ImageItem()
        self.map_plot.addItem(self.map_img)
        self.map_pts = pg.ScatterPlotItem(size=9, pen=pg.mkPen("k", width=0.5), hoverable=True,
                                          tip=lambda x, y, data: str(data))
        self.map_plot.addItem(self.map_pts)
        self.map_pts.sigClicked.connect(self._clicked)
        self.map_bar = pg.ColorBarItem(interactive=False, width=12)
        map_w = QWidget()
        ml = QVBoxLayout(map_w)
        ml.setContentsMargins(0, 0, 0, 0)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Points:"))
        self.cmb_map_points = QComboBox()
        self.cmb_map_points.addItems(["coloured by value", "white with black outline", "hidden"])
        self.cmb_map_points.setToolTip("How the molecules are drawn over the landscape")
        self.cmb_map_points.currentIndexChanged.connect(self._draw_map_points)
        mrow.addWidget(self.cmb_map_points)
        mrow.addWidget(QLabel("size:"))
        self.spin_map_size = QSpinBox()
        self.spin_map_size.setRange(2, 30)
        self.spin_map_size.setValue(9)
        self.spin_map_size.valueChanged.connect(self._draw_map_points)
        mrow.addWidget(self.spin_map_size)
        mrow.addStretch(1)
        ml.addLayout(mrow)
        ml.addWidget(self.map_plot, 1)
        self._map_data = None
        self.tab_map = self.tabs.addTab(map_w, "Map")
        self.tabs.setTabVisible(self.tab_map, False)

        self.tbl_search = QTableWidget(0, 4)
        self.tbl_search.setHorizontalHeaderLabels(["alpha", "params", "RMSEcv", "q2"])
        self.tbl_search.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_search.verticalHeader().setVisible(False)
        self.tabs.addTab(self.tbl_search, "Hyper-parameter search")

        # --- contributions: decomposition rules 2 (training molecules) and 3 (atoms)
        contrib_w = QWidget()
        cl = QVBoxLayout(contrib_w)
        cl.setContentsMargins(4, 4, 4, 4)
        self.lbl_contrib = QLabel("Select a molecule in the data-set table to decompose its prediction.")
        self.lbl_contrib.setWordWrap(True)
        cl.addWidget(self.lbl_contrib)
        tables = QHBoxLayout()
        self.tbl_contrib = QTableWidget(0, 3)
        self.tbl_contrib.setHorizontalHeaderLabels(["Training molecule", "a_j", "a_j K(X, X_j)"])
        self.tbl_contrib.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_contrib.verticalHeader().setVisible(False)
        self.tbl_contrib.setToolTip("Decomposition rule 2: the prediction as a sum over training molecules")
        tables.addWidget(self.tbl_contrib, 1)
        self.tbl_atoms = QTableWidget(0, 4)
        self.tbl_atoms.setHorizontalHeaderLabels(["Atom", "Element", "Type", "Contribution"])
        self.tbl_atoms.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_atoms.verticalHeader().setVisible(False)
        self.tbl_atoms.setToolTip("Decomposition rule 3: the prediction as a sum over the atoms of this molecule")
        tables.addWidget(self.tbl_atoms, 1)
        cl.addLayout(tables)
        self.tabs.addTab(contrib_w, "Contributions")

        # --- Shapley decomposition: works for any kernel, including a nonlinear layer
        shap_w = QWidget()
        sl = QVBoxLayout(shap_w)
        sl.setContentsMargins(4, 4, 4, 4)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Players:"))
        self.cmb_shap_players = QComboBox()
        self.cmb_shap_players.addItems(["atoms", "substituents of the common scaffold", "field types",
                                        "atoms x field types (Shapley field)"])
        self.cmb_shap_players.setToolTip("\n".join([
            "What the prediction is divided between.",
            "Atoms: one value per atom, estimated by permutation sampling.",
            "Substituents: the common scaffold of the training set is held fixed and the",
            "  values explain the difference from it; few enough players to be exact.",
            "Field types: what each molecular field contributes, which replaces the overlap",
            "  integral of co-field and molecular field when the kernel is not linear.",
            "Atoms x field types: one value per atom per field, which the viewer smears over",
            "  the Gaussians of that field into a Shapley field, one map per field type."]))
        self.cmb_shap_players.currentIndexChanged.connect(self._shap_players_changed)
        srow.addWidget(self.cmb_shap_players)
        self.spin_shap_samples = QSpinBox()
        self.spin_shap_samples.setRange(20, 100000)
        self.spin_shap_samples.setValue(300)
        self.spin_shap_samples.setPrefix("samples ")
        self.spin_shap_samples.setToolTip("Permutations sampled when the players are too many to "
                                          "enumerate exactly; the table shows the standard error")
        srow.addWidget(self.spin_shap_samples)
        self.chk_shap_inter = QCheckBox("interaction indices")
        self.chk_shap_inter.setToolTip("Second-order Shapley interaction indices: the part of the "
                                       "prediction that two players produce only together.\n"
                                       "They vanish for a linear kernel without normalisation.")
        srow.addWidget(self.chk_shap_inter)
        self.btn_shapley = QPushButton("Compute")
        self.btn_shapley.setStyleSheet("font-weight: bold")
        srow.addWidget(self.btn_shapley)
        srow.addStretch(1)
        sl.addLayout(srow)
        self.lbl_shapley = QLabel("Select a molecule in the data-set table and press Compute.")
        self.lbl_shapley.setWordWrap(True)
        sl.addWidget(self.lbl_shapley)
        stables = QHBoxLayout()
        self.tbl_shapley = QTableWidget(0, 3)
        self.tbl_shapley.setHorizontalHeaderLabels(["Player", "Shapley value", "std. error"])
        self.tbl_shapley.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_shapley.verticalHeader().setVisible(False)
        stables.addWidget(self.tbl_shapley, 1)
        self.tbl_shap_inter = QTableWidget(0, 3)
        self.tbl_shap_inter.setHorizontalHeaderLabels(["Player", "Player", "Interaction"])
        self.tbl_shap_inter.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_shap_inter.verticalHeader().setVisible(False)
        stables.addWidget(self.tbl_shap_inter, 1)
        sl.addLayout(stables)
        self.tabs.addTab(shap_w, "Shapley")
        self._shap_players_changed()

        # --- complementarity with the protein pocket
        comp_w = QWidget()
        kl = QVBoxLayout(comp_w)
        kl.setContentsMargins(4, 4, 4, 4)
        krow = QHBoxLayout()
        self.btn_complement = QPushButton("Compute")
        self.btn_complement.setStyleSheet("font-weight: bold")
        krow.addWidget(self.btn_complement)
        self.chk_comp_surface = QCheckBox("surface scores")
        self.chk_comp_surface.setChecked(True)
        self.chk_comp_surface.setToolTip("\n".join([
            "Also evaluate the fields on the surface of every molecule, which gives the",
            "electrostatic complementarity score of the literature and the colours for the",
            "surface in the 3D view. Slower on a large series."]))
        krow.addWidget(self.chk_comp_surface)
        krow.addWidget(QLabel("statistic:"))
        self.cmb_comp_stat = QComboBox()
        self.cmb_comp_stat.addItems(["sign agreement", "cosine", "correlation"])
        self.cmb_comp_stat.setToolTip("\n".join([
            "How a co-field is compared with what the pocket offers:",
            "  sign agreement: do they want the same sign in the same place, weighted by",
            "    how strongly each wants it; this is how a co-field is read",
            "  cosine: also demands that the two maps have the same shape",
            "  correlation: removes the means first, so it asks only whether the variations track"]))
        krow.addWidget(self.cmb_comp_stat)
        krow.addStretch(1)
        kl.addLayout(krow)
        self.lbl_complement = QLabel("Load a protein structure, then press Compute.")
        self.lbl_complement.setWordWrap(True)
        kl.addWidget(self.lbl_complement)
        ktab = QHBoxLayout()
        self.tbl_comp_mol = QTableWidget(0, 3)
        self.tbl_comp_mol.setHorizontalHeaderLabels(["Term", "This molecule", "Series mean"])
        self.tbl_comp_mol.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_comp_mol.verticalHeader().setVisible(False)
        self.tbl_comp_mol.setToolTip("Complementarity of the selected molecule with the pocket, "
                                     "from exact overlap integrals; +1 is perfect, -1 is opposite")
        ktab.addWidget(self.tbl_comp_mol, 1)
        self.tbl_comp_series = QTableWidget(0, 2)
        self.tbl_comp_series.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.tbl_comp_series.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_comp_series.verticalHeader().setVisible(False)
        self.tbl_comp_series.setToolTip("How the scores relate to the activity across the series, "
                                        "and how far the co-fields agree with the pocket")
        ktab.addWidget(self.tbl_comp_series, 1)
        kl.addLayout(ktab)
        self.tabs.addTab(comp_w, "Complementarity")

        # ----- chemotypes: one held out at a time, against the model that ignores the poses
        chem_w = QWidget()
        cml = QVBoxLayout(chem_w)
        cml.setContentsMargins(4, 4, 4, 4)
        self.lbl_chemo = QLabel("Tools, Heterogeneous sets: find the chemotypes, then hold each "
                                "one out in turn.")
        self.lbl_chemo.setWordWrap(True)
        cml.addWidget(self.lbl_chemo)
        crow = QHBoxLayout()
        self.tbl_chemo = QTableWidget(0, 2)
        self.tbl_chemo.setHorizontalHeaderLabels(["Quantity", "Value"])
        self.tbl_chemo.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_chemo.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.tbl_chemo.verticalHeader().setVisible(False)
        self.tbl_chemo.setToolTip("\n".join([
            "Random folds predict a molecule from its own scaffold-mates and flatter a",
            "mixed series. The honest number is the grouped one, and it has to beat the",
            "alignment-free model, which uses no poses at all."]))
        crow.addWidget(self.tbl_chemo, 1)
        self.tbl_chemo_group = QTableWidget(0, 4)
        self.tbl_chemo_group.setHorizontalHeaderLabels(["Chemotype", "N", "RMSE", "Bias"])
        chh = self.tbl_chemo_group.horizontalHeader()
        chh.setSectionResizeMode(QHeaderView.ResizeToContents)
        chh.setSectionResizeMode(0, QHeaderView.Stretch)
        self.tbl_chemo_group.verticalHeader().setVisible(False)
        self.tbl_chemo_group.setToolTip("Every chemotype while it was the one held out; a large "
                                        "bias means the model cannot reach that scaffold at all")
        crow.addWidget(self.tbl_chemo_group, 2)
        cml.addLayout(crow)
        self.tabs.addTab(chem_w, "Chemotypes")

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Log")

    def complement_statistic(self) -> str:
        return ["sign", "cosine", "pearson"][self.cmb_comp_stat.currentIndex()]

    def set_complementarity(self, header: str, rows, series) -> None:
        """Fill the complementarity tab: per-molecule terms and series-level numbers."""
        self.lbl_complement.setText(header)
        self.tbl_comp_mol.setRowCount(0)
        self.tbl_comp_mol.setRowCount(len(rows))
        for r, (label, here, mean) in enumerate(rows):
            self.tbl_comp_mol.setItem(r, 0, QTableWidgetItem(str(label)))
            for c, v in ((1, here), (2, mean)):
                txt = "" if v is None else (f"{v:+.3f}" if isinstance(v, float) else str(v))
                self.tbl_comp_mol.setItem(r, c, QTableWidgetItem(txt))
        self.tbl_comp_series.setRowCount(0)
        self.tbl_comp_series.setRowCount(len(series))
        for r, (label, v) in enumerate(series):
            self.tbl_comp_series.setItem(r, 0, QTableWidgetItem(str(label)))
            self.tbl_comp_series.setItem(r, 1, QTableWidgetItem(
                v if isinstance(v, str) else f"{v:+.3f}"))

    def set_chemotypes(self, header: str, rows, per_group) -> None:
        """Fill the Chemotypes tab: the three q² values, then every held-out chemotype."""
        self.lbl_chemo.setText(header)
        self.tbl_chemo.setRowCount(0)
        self.tbl_chemo.setRowCount(len(rows))
        for r, (label, v) in enumerate(rows):
            self.tbl_chemo.setItem(r, 0, QTableWidgetItem(str(label)))
            if isinstance(v, str):
                txt = v
            elif v is None or not np.isfinite(v):
                txt = ""
            elif str(label).startswith("q²"):
                txt = f"{v:+.3f}"
            elif str(label).startswith("RMSE"):
                txt = f"{v:.3f}"
            else:
                txt = f"{v:g}"
            self.tbl_chemo.setItem(r, 1, QTableWidgetItem(txt))
        self.tbl_chemo_group.setRowCount(0)
        self.tbl_chemo_group.setRowCount(len(per_group))
        for r, (name, n, rmse, bias) in enumerate(per_group):
            self.tbl_chemo_group.setItem(r, 0, QTableWidgetItem(str(name)))
            it = QTableWidgetItem(str(int(n)))
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.tbl_chemo_group.setItem(r, 1, it)
            for c, v, fmt in ((2, rmse, "{:.3f}"), (3, bias, "{:+.3f}")):
                it = QTableWidgetItem("" if v is None or not np.isfinite(v) else fmt.format(v))
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.tbl_chemo_group.setItem(r, c, it)

    def _shap_players_changed(self, *_):
        """Sampling is used where the players are too many to enumerate."""
        self.spin_shap_samples.setVisible(self.cmb_shap_players.currentIndex() in (0, 3))

    def shapley_players(self) -> str:
        return ["atoms", "groups", "fields", "atomfields"][self.cmb_shap_players.currentIndex()]

    def set_shapley(self, res, header: str) -> None:
        """Fill the Shapley tab from a :class:`cmfqsar.shapley.ShapleyResult`."""
        self.lbl_shapley.setText(header)
        rows = res.table() if res is not None else []
        self.tbl_shapley.setRowCount(0)
        self.tbl_shapley.setRowCount(len(rows))
        for r, (label, v, se) in enumerate(rows):
            self.tbl_shapley.setItem(r, 0, QTableWidgetItem(str(label)))
            self.tbl_shapley.setItem(r, 1, QTableWidgetItem(f"{v:+.4f}"))
            self.tbl_shapley.setItem(r, 2, QTableWidgetItem("exact" if np.isnan(se) else f"{se:.4f}"))
        inter = res.top_interactions(25) if res is not None else []
        self.tbl_shap_inter.setRowCount(0)
        self.tbl_shap_inter.setRowCount(len(inter))
        for r, (a, b, v) in enumerate(inter):
            self.tbl_shap_inter.setItem(r, 0, QTableWidgetItem(str(a)))
            self.tbl_shap_inter.setItem(r, 1, QTableWidgetItem(str(b)))
            self.tbl_shap_inter.setItem(r, 2, QTableWidgetItem(f"{v:+.4f}"))

    @staticmethod
    def _tip(x, y, data):
        return f"{data}\nexp {x:.3f}, pred {y:.3f}"

    def _clicked(self, item, points):
        if len(points):
            d = points[0].data()
            try:
                self.point_clicked.emit(int(str(d).split("#")[-1]))
            except ValueError:
                pass

    def set_points(self, which: str, x, y, names, indices):
        sc = {"cv": self.sc_cv, "test": self.sc_test, "ex": self.sc_ex}[which]
        pts = [{"pos": (float(a), float(b)), "data": f"{n} #{i}"} for a, b, n, i in zip(x, y, names, indices)
               if np.isfinite(a) and np.isfinite(b)]
        sc.setData(pts)
        self._autorange()

    def clear_points(self, which=None):
        for k, sc in (("cv", self.sc_cv), ("test", self.sc_test), ("ex", self.sc_ex)):
            if which is None or which == k:
                sc.setData([])

    def _autorange(self):
        xs, ys = [], []
        for sc in (self.sc_cv, self.sc_test, self.sc_ex):
            d = sc.getData()
            if d[0] is not None and len(d[0]):
                xs.extend(d[0])
                ys.extend(d[1])
        if xs:
            lo = min(min(xs), min(ys))
            hi = max(max(xs), max(ys))
            pad = 0.05 * (hi - lo + 1e-9)
            self.plot.enableAutoRange(enable=False)
            self.plot.setRange(xRange=(lo - pad, hi + pad), yRange=(lo - pad, hi + pad), padding=0)

    def set_classification_mode(self, on: bool, activity_name: str = "activity",
                                threshold: Optional[float] = None, have_activity: bool = True) -> None:
        """Switch the plots between the regression and the classification layout.

        For a classifier the scatter plot only means something when the classes come from a
        threshold on a continuous activity: it then shows how the decision value tracks that
        activity, and the two dashed lines cut it into the true and false positives and negatives.
        Without such an activity the tab is hidden, because the ranking (Scores) and the ROC curve
        are what a classifier is judged by.
        """
        self.plot.setLabel("left", "decision value" if on else "Predicted")
        self.plot.setLabel("bottom", activity_name if on else "Experimental")
        self.line.setVisible(not on)
        self.vline.setVisible(bool(on and threshold is not None))
        self.hline.setVisible(on)
        if threshold is not None:
            self.vline.setPos(float(threshold))
        if on:
            self.plot.enableAutoRange(enable=True)
        if not on:
            self.tabs.setTabVisible(self.tab_roc, False)
        self.tabs.setTabVisible(self.tab_scatter, (not on) or have_activity)
        self._relabel_legend(on)

    def _relabel_legend(self, classification: bool) -> None:
        try:
            self.legend.clear()
        except Exception:                                    # noqa: BLE001
            return
        if classification:
            pairs = ((self.sc_cv, "active (circle = training, square = test)"),
                     (self.sc_test, "inactive"))
        else:
            pairs = ((self.sc_cv, "training (CV)"), (self.sc_test, "test set"),
                     (self.sc_ex, "external CV"))
        for item, label in pairs:
            self.legend.addItem(item, label)

    def set_class_points(self, x, y, classes, is_test, names, indices) -> None:
        """Scatter of decision value against activity, coloured by true class.

        Training molecules are drawn as circles and test molecules as squares.
        """
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        c = np.asarray(classes, float)
        ok = np.isfinite(x) & np.isfinite(y)
        for item, sel in ((self.sc_cv, ok & (c > 0)), (self.sc_test, ok & ~(c > 0))):
            pts = [{"pos": (float(x[i]), float(y[i])),
                    "symbol": "s" if is_test[i] else "o",
                    "data": f"{names[i]} #{indices[i]}"}
                   for i in np.where(sel)[0]]
            item.setData(pts)
        self.sc_ex.setData([])
        self._autorange()

    @staticmethod
    def _roc_curve(y_true, decision):
        """(false positive rate, true positive rate, AUC) of a binary decision function."""
        y = np.asarray(y_true, float)
        d = np.asarray(decision, float)
        ok = np.isfinite(y) & np.isfinite(d)
        y, d = y[ok], d[ok]
        pos, neg = y > 0, y < 0
        if not pos.any() or not neg.any():
            return None
        order = np.argsort(-d)
        y = y[order]
        tp = np.cumsum(y > 0) / max(1, int(pos.sum()))
        fp = np.cumsum(y < 0) / max(1, int(neg.sum()))
        tp = np.concatenate([[0.0], tp])
        fp = np.concatenate([[0.0], fp])
        return fp, tp, float(np.trapezoid(tp, fp)) if hasattr(np, "trapezoid") else float(np.trapz(tp, fp))

    def set_roc(self, curves) -> None:
        """curves: list of (label, true classes +-1, decision values).

        The tab is hidden when none of them can be drawn, so an empty plot is never shown.
        """
        for it in self._roc_items:
            self.roc.removeItem(it)
        self._roc_items = []
        colors = [(60, 120, 220), (230, 80, 60), (40, 170, 90)]
        for k, (label, y_true, decision) in enumerate(curves):
            cur = self._roc_curve(y_true, decision)
            if cur is None:
                continue
            fp, tp, auc = cur
            item = self.roc.plot(fp, tp, pen=pg.mkPen(colors[k % len(colors)], width=2),
                                 name=f"{label} (AUC {auc:.3f})")
            self._roc_items.append(item)
        self.tabs.setTabVisible(self.tab_roc, bool(self._roc_items))

    def set_scores(self, names, scores, classes=None, subsets=None, boundary: float = 0.0) -> None:
        """Ranked decision values: the output of a classification or one-class model.

        `classes` is +1/-1/NaN where known, `subsets` the train/test membership.
        """
        names = list(names)
        d = np.asarray(scores, dtype=float)
        ok = np.isfinite(d)
        self.tabs.setTabVisible(self.tab_scores, bool(ok.any()))
        for it in self._score_items:
            self.score_hist.removeItem(it)
        self._score_items = []
        if not ok.any():
            self.tbl_scores.setRowCount(0)
            return
        cls = None if classes is None else np.asarray(classes, dtype=float)
        lo, hi = float(np.min(d[ok])), float(np.max(d[ok]))
        if hi <= lo:
            hi = lo + 1.0
        edges = np.linspace(lo, hi, 26)
        if cls is not None and np.isfinite(cls).any() and (cls[np.isfinite(cls)] < 0).any():
            groups = [("active", d[ok & (cls > 0)], (60, 120, 220, 170)),
                      ("inactive", d[ok & (cls < 0)], (230, 80, 60, 170))]
        else:
            groups = [("inside the boundary", d[ok & (d >= boundary)], (40, 170, 90, 170)),
                      ("outside", d[ok & (d < boundary)], (230, 80, 60, 170))]
        width = (edges[1] - edges[0]) * 0.45
        for k, (label, vals, colour) in enumerate(groups):
            if len(vals) == 0:
                continue
            counts, _ = np.histogram(vals, bins=edges)
            centres = 0.5 * (edges[:-1] + edges[1:]) + (k - 0.5) * width
            item = pg.BarGraphItem(x=centres, height=counts, width=width,
                                   brush=pg.mkBrush(colour), pen=pg.mkPen(None), name=label)
            self.score_hist.addItem(item)
            self._score_items.append(item)
        line = pg.InfiniteLine(pos=boundary, angle=90,
                               pen=pg.mkPen((120, 120, 120), width=2, style=Qt.DashLine))
        self.score_hist.addItem(line)
        self._score_items.append(line)

        order = np.argsort(-np.where(ok, d, -np.inf))
        self.tbl_scores.setRowCount(0)
        self.tbl_scores.setRowCount(int(ok.sum()))
        for row, i in enumerate(order[:int(ok.sum())]):
            lab = ""
            if cls is not None and np.isfinite(cls[i]):
                lab = "active" if cls[i] > 0 else "inactive"
            for col, text in ((0, str(row + 1)), (1, str(names[i])), (2, f"{d[i]:.4f}"),
                              (3, "" if subsets is None else str(subsets[i])), (4, lab)):
                self.tbl_scores.setItem(row, col, QTableWidgetItem(text))

    def set_enrichment(self, y_true=None, decision=None, label: str = "screening") -> None:
        """Accumulation curve: actives found against the fraction of the library screened."""
        from ..model import enrichment_curve
        for it in self._enrich_items:
            self.enrich.removeItem(it)
        self._enrich_items = []
        curve = None if y_true is None else enrichment_curve(y_true, decision)
        self.tabs.setTabVisible(self.tab_enrich, curve is not None)
        if curve is None:
            return
        x, y = curve
        item = self.enrich.plot(x, y, pen=pg.mkPen((40, 170, 90), width=2), name=label)
        self._enrich_items.append(item)
        for f in (0.01, 0.05, 0.10):
            ln = pg.InfiniteLine(pos=f, angle=90, pen=pg.mkPen((170, 170, 170), style=Qt.DotLine))
            self.enrich.addItem(ln)
            self._enrich_items.append(ln)

    def set_map(self, image=None, coords=None, values=None, names=None, indices=None,
                title: str = "", cmap: str = "viridis") -> None:
        """Draw a GTM map: `image` is the landscape on the node grid, `coords` the molecules."""
        if image is None:
            self.tabs.setTabVisible(self.tab_map, False)
            return
        img = np.asarray(image, dtype=float)
        finite = np.isfinite(img)
        lo = float(np.nanmin(img)) if finite.any() else 0.0
        hi = float(np.nanmax(img)) if finite.any() else 1.0
        if hi <= lo:
            hi = lo + 1.0
        shown = np.where(finite, img, lo - (hi - lo))          # unsupported nodes: below the scale
        self.map_img.setImage(shown.T, levels=(lo - (hi - lo) * 1e-3, hi), autoLevels=False)
        self.map_img.setRect(pg.QtCore.QRectF(-1.0, -1.0, 2.0, 2.0))
        try:
            self.map_img.setColorMap(pg.colormap.get(cmap))
        except Exception:                                       # noqa: BLE001
            pass
        for side, text in (("bottom", "latent coordinate 1"),
                           ("left", "latent coordinate 2")):
            self.map_plot.getAxis(side).enableAutoSIPrefix(True)
            self.map_plot.setLabel(side, text)
        self._map_data = {"coords": None if coords is None else np.asarray(coords, dtype=float),
                          "values": None if values is None else np.asarray(values, dtype=float),
                          "names": names, "indices": indices, "cmap": cmap, "lo": lo, "hi": hi}
        self._draw_map_points()
        self.map_plot.setTitle(title)
        self.tabs.setTabVisible(self.tab_map, True)

    def set_scatter(self, coords, values=None, names=None, indices=None, title: str = "",
                    cmap: str = "viridis", xlabel: str = "", ylabel: str = "") -> None:
        """Molecules as points with nothing behind them: the kernel-PCA score plot.

        The landscape of a GTM fills the square it is drawn on, so the eye has somewhere to put
        every molecule.  A score plot has no such backdrop and must not pretend to one -- what
        it shows is the spacing of the points themselves, which is why the axes carry the share
        of the kernel variance each component explains and the aspect ratio stays locked.
        """
        coords = None if coords is None else np.asarray(coords, dtype=float)
        if coords is None or not len(coords):
            self.tabs.setTabVisible(self.tab_map, False)
            return
        self.map_img.clear()
        v = None if values is None else np.asarray(values, dtype=float)
        lo, hi = 0.0, 1.0
        if v is not None and np.isfinite(v).any():
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if hi <= lo:
            hi = lo + 1.0
        self._map_data = {"coords": coords, "values": v, "names": names, "indices": indices,
                          "cmap": cmap, "lo": lo, "hi": hi}
        self._draw_map_points()
        for side, text in (("bottom", xlabel or "component 1"),
                           ("left", ylabel or "component 2")):
            # the label already carries a figure in brackets, and pyqtgraph's own
            # "(x0.001)" beside it reads as if that figure had been scaled
            self.map_plot.getAxis(side).enableAutoSIPrefix(False)
            self.map_plot.setLabel(side, text)
        self.map_plot.setTitle(title)
        self.map_plot.enableAutoRange()
        self.tabs.setTabVisible(self.tab_map, True)

    def _draw_map_points(self, *_):
        """Draw the molecules over the landscape in the chosen style."""
        d = self._map_data
        if not d or d["coords"] is None or not len(d["coords"]):
            self.map_pts.setData([])
            return
        style = self.cmb_map_points.currentIndex()
        if style == 2:                                   # hidden: show the landscape alone
            self.map_pts.setData([])
            return
        c, v = d["coords"], d["values"]
        lo, hi = d["lo"], d["hi"]
        size = self.spin_map_size.value()
        pts = []
        for i in range(len(c)):
            if style == 1:                               # white marker with a black outline
                brush, pen = pg.mkBrush(255, 255, 255), pg.mkPen((0, 0, 0), width=1.4)
            elif v is None or not np.isfinite(v[i]):
                brush, pen = pg.mkBrush(170, 170, 170, 200), pg.mkPen("k", width=0.5)
            else:
                f = float(np.clip((v[i] - lo) / (hi - lo), 0, 1))
                brush = pg.mkBrush(pg.colormap.get(d["cmap"]).map(f, mode="qcolor"))
                pen = pg.mkPen((0, 0, 0), width=0.5)
            pts.append({"pos": (float(c[i, 0]), float(c[i, 1])), "brush": brush, "pen": pen,
                        "size": size,
                        "data": f"{d['names'][i]} #{d['indices'][i]}" if d["names"] is not None else ""})
        self.map_pts.setData(pts)

    def set_search_table(self, rows):
        self.tbl_search.setRowCount(0)
        self.tbl_search.setRowCount(len(rows))
        for i, r in enumerate(rows):
            params = {k: v for k, v in r.items() if k not in ("alpha", "rmse_cv", "q2", "h")}
            ptxt = ", ".join(f"{k}={v:g}" for k, v in params.items())
            if "h" in r:
                ptxt += "  h=" + ",".join(f"{v:g}" for v in r["h"].values())
            self.tbl_search.setItem(i, 0, QTableWidgetItem(f"{r['alpha']:g}"))
            self.tbl_search.setItem(i, 1, QTableWidgetItem(ptxt))
            self.tbl_search.setItem(i, 2, QTableWidgetItem(f"{r['rmse_cv']:.4f}"))
            self.tbl_search.setItem(i, 3, QTableWidgetItem(f"{r['q2']:.4f}"))

    def set_contributions(self, names, coef, contrib, header: str = "") -> None:
        """Decomposition rule 2: the contribution of every training molecule."""
        if header:
            self.lbl_contrib.setText(header)
        order = np.argsort(-np.abs(np.asarray(contrib)))
        self.tbl_contrib.setRowCount(0)
        self.tbl_contrib.setRowCount(len(order))
        for r, j in enumerate(order):
            self.tbl_contrib.setItem(r, 0, QTableWidgetItem(str(names[j])))
            self.tbl_contrib.setItem(r, 1, QTableWidgetItem(f"{coef[j]:.4g}"))
            self.tbl_contrib.setItem(r, 2, QTableWidgetItem(f"{contrib[j]:+.4f}"))

    def set_atom_contributions(self, symbols, types, contrib) -> None:
        """Decomposition rule 3: the contribution of every atom of the selected molecule."""
        c = np.asarray(contrib)
        order = np.argsort(-np.abs(c))
        self.tbl_atoms.setRowCount(0)
        self.tbl_atoms.setRowCount(len(order))
        for r, j in enumerate(order):
            self.tbl_atoms.setItem(r, 0, QTableWidgetItem(str(j + 1)))
            self.tbl_atoms.setItem(r, 1, QTableWidgetItem(str(symbols[j])))
            self.tbl_atoms.setItem(r, 2, QTableWidgetItem(str(types[j])))
            self.tbl_atoms.setItem(r, 3, QTableWidgetItem(f"{c[j]:+.4f}"))

    def clear_contributions(self, message: str = "") -> None:
        self.tbl_contrib.setRowCount(0)
        self.tbl_atoms.setRowCount(0)
        if message:
            self.lbl_contrib.setText(message)

    def append_log(self, text: str):
        self.log.append(text)


# --------------------------------------------------------------------------- #
# Viewer controls (toolbar above the 3D view)
# --------------------------------------------------------------------------- #

class ViewerControls(QWidget):
    changed = Signal()                 # any display option changed
    compute_requested = Signal()       # co-field computation requested
    endpoint_changed = Signal()        # which endpoint's co-field to show
    style_changed = Signal()
    reset_view = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Style:"))
        self.cmb_style = QComboBox()
        self.cmb_style.addItems(["ball & stick", "sticks", "lines", "space fill"])
        self.cmb_style.currentIndexChanged.connect(self.style_changed)
        row1.addWidget(self.cmb_style)
        self.chk_hydrogens = QCheckBox("H atoms")
        self.chk_hydrogens.setChecked(True)
        self.chk_hydrogens.toggled.connect(self.style_changed)
        row1.addWidget(self.chk_hydrogens)
        row1.addWidget(QLabel("Labels:"))
        self.cmb_labels = QComboBox()
        self.cmb_labels.addItems(["none", "element + number", "atom type", "partial charge"])
        self.cmb_labels.setToolTip("\n".join([
            "Write a label at every atom of the selected molecule:",
            "  element + number: C7, N12, as the contribution tables number them",
            "  atom type: the Sybyl code used by the indicator fields, C.ar, N.am",
            "  partial charge: the charge that enters the electrostatic field"]))
        self.cmb_labels.currentIndexChanged.connect(self.changed)
        row1.addWidget(self.cmb_labels)
        self.chk_dark = QCheckBox("dark background")
        self.chk_dark.setChecked(True)
        self.chk_dark.toggled.connect(self.changed)
        row1.addWidget(self.chk_dark)
        self.btn_reset = QPushButton("Reset view")
        self.btn_reset.clicked.connect(self.reset_view)
        row1.addWidget(self.btn_reset)
        row1.addStretch(1)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Field type:"))
        self.cmb_field = QComboBox()
        self.cmb_field.setMinimumWidth(160)
        self.cmb_field.setToolTip("Field type shown in the 3D view: used for the co-field C(r), "
                                  "the molecular field X(r) and their overlap")
        self.cmb_field.currentIndexChanged.connect(self.changed)
        row2.addWidget(self.cmb_field)
        # a model of two endpoints has a co-field per endpoint and one for their
        # difference; the row appears only once such a model exists
        self.lbl_endpoint = QLabel("Endpoint:")
        self.cmb_endpoint = QComboBox()
        self.cmb_endpoint.setMinimumWidth(170)
        self.cmb_endpoint.setToolTip(
            "<p>Which endpoint's co-field is shown. The difference is the "
            "<b>selectivity co-field</b>: where more of this field raises one endpoint "
            "more than the other, which is what a separation factor is made of.</p>")
        self.cmb_endpoint.currentIndexChanged.connect(self.endpoint_changed)
        row2.addWidget(self.lbl_endpoint)
        row2.addWidget(self.cmb_endpoint)
        self.lbl_endpoint.setVisible(False)
        self.cmb_endpoint.setVisible(False)
        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setRange(0.2, 2.0)
        self.spin_spacing.setSingleStep(0.1)
        self.spin_spacing.setValue(0.5)
        self.spin_spacing.setPrefix("grid ")
        self.spin_spacing.setSuffix(" Å")
        row2.addWidget(self.spin_spacing)
        self.spin_margin = QDoubleSpinBox()
        self.spin_margin.setRange(0.0, 10.0)
        self.spin_margin.setValue(4.0)
        self.spin_margin.setPrefix("margin ")
        self.spin_margin.setSuffix(" Å")
        row2.addWidget(self.spin_margin)
        self.btn_compute = QPushButton("Compute co-field")
        self.btn_compute.clicked.connect(self.compute_requested)
        row2.addWidget(self.btn_compute)
        row2.addStretch(1)
        lay.addLayout(row2)

        row3 = QHBoxLayout()
        self.chk_cofield = QCheckBox("co-field C(r)")
        self.chk_cofield.setChecked(True)
        self.cmb_cofield_style = QComboBox()
        self.cmb_cofield_style.addItems(["solid", "mesh"])
        self.cmb_cofield_style.setToolTip("Drawing style of the co-field isosurfaces")
        self.cmb_cofield_style.currentIndexChanged.connect(self.changed)
        self.chk_molfield = QCheckBox("molecular field X(r)")
        self.cmb_molfield_style = QComboBox()
        self.cmb_molfield_style.addItems(["auto", "mesh", "solid"])
        self.cmb_molfield_style.setToolTip(
            "Drawing style of the molecular-field isosurfaces.\n"
            "'auto' draws a mesh while a co-field is on screen, so that the two can be told\n"
            "apart, and a solid surface when the molecular field is the only one shown.")
        self.cmb_molfield_style.currentIndexChanged.connect(self.changed)
        self.chk_overlap = QCheckBox("overlap C·X")
        self.chk_shapfield = QCheckBox("Shapley field Φ(r)")
        self.chk_shapfield.setToolTip("\n".join([
            "The Shapley values of the selected field smeared over the Gaussians of that field.",
            "Its integral is the part of the prediction attributed to the field, as the integral",
            "of a charge density is the total charge. Compute it in the Results panel, Shapley",
            "tab, with the players set to 'atoms x field types'.",
            "Unlike a co-field the map belongs to one molecule, because a model with a nonlinear",
            "kernel has no co-field for it to belong to."]))
        self.cmb_shapfield_style = QComboBox()
        self.cmb_shapfield_style.addItems(["solid", "mesh"])
        self.cmb_shapfield_style.currentIndexChanged.connect(self.changed)
        self.chk_atoms = QCheckBox("atom contributions")
        self.chk_diff = QCheckBox("difference map")
        self.chk_diff.setToolTip(
            "The field of the selected molecule minus the field of a reference molecule "
            "(Tools, Field difference from a reference): negative where the selected molecule "
            "fails to reproduce what the reference presents, positive where it adds something "
            "of its own. The field type is the one in the 'Field type' box.")
        self.chk_diff.setEnabled(False)
        self.chk_pharmacophore = QCheckBox("pharmacophore")
        self.chk_pharmacophore.setToolTip(
            "The features extracted from the co-fields (Tools, Extract pharmacophore), drawn as "
            "translucent spheres coloured by type. Untick to take them off the view; they are "
            "kept until the model is rebuilt or the molecules move.")
        self.chk_pharmacophore.setEnabled(False)
        # the maps go on two rows of their own: on one line the labels were clipped to a few
        # letters and the tick boxes ran into each other
        row3.addWidget(QLabel("Maps:"))
        row3b = QHBoxLayout()
        row3b.addWidget(QLabel("    "))              # indent, so the two rows read as one group
        for cb in (self.chk_cofield, self.chk_molfield, self.chk_overlap):
            cb.toggled.connect(self.changed)
            row3.addWidget(cb)
            if cb is self.chk_cofield:
                row3.addWidget(self.cmb_cofield_style)
            if cb is self.chk_molfield:
                row3.addWidget(self.cmb_molfield_style)
        for cb in (self.chk_shapfield, self.chk_atoms, self.chk_diff, self.chk_pharmacophore):
            cb.toggled.connect(self.changed)
            row3b.addWidget(cb)
            if cb is self.chk_shapfield:
                row3b.addWidget(self.cmb_shapfield_style)
        row3.addStretch(1)
        lay.addLayout(row3)
        row3b.addStretch(1)
        lay.addLayout(row3b)

        row3c = QHBoxLayout()
        row3 = row3c                                 # the sliders now have a row to themselves
        row3.addWidget(QLabel("iso level:"))
        self.sld_level = QSlider(Qt.Horizontal)
        self.sld_level.setRange(1, 99)
        self.sld_level.setValue(50)
        self.sld_level.setMinimumWidth(110)
        self.sld_level.valueChanged.connect(self.changed)
        row3.addWidget(self.sld_level)
        self.lbl_level = QLabel("50 %")
        row3.addWidget(self.lbl_level)
        self.sld_level.valueChanged.connect(lambda v: self.lbl_level.setText(f"{v} %"))
        row3.addWidget(QLabel("opacity:"))
        self.sld_alpha = QSlider(Qt.Horizontal)
        self.sld_alpha.setRange(5, 100)
        self.sld_alpha.setValue(45)
        self.sld_alpha.setMaximumWidth(90)
        self.sld_alpha.valueChanged.connect(self.changed)
        row3.addWidget(self.sld_alpha)
        row3.addStretch(1)
        lay.addLayout(row3)

        # --- protein binding site (free-energy sets are already in the crystal frame)
        row4 = QHBoxLayout()
        self.chk_protein = QCheckBox("protein site")
        self.chk_protein.setToolTip("\n".join([
            "Show the residues of a loaded structure around the molecules on screen.",
            "The benchmark ligands are given in the frame of their reference structure, so",
            "nothing is superimposed; File -> Fetch protein reads the code from the data set.",
            "Realigning the molecules afterwards takes them out of that frame."]))
        self.chk_protein.toggled.connect(self.changed)
        row4.addWidget(self.chk_protein)
        self.cmb_protein_style = QComboBox()
        self.cmb_protein_style.addItems(["lines", "sticks"])
        self.cmb_protein_style.currentIndexChanged.connect(self.changed)
        row4.addWidget(self.cmb_protein_style)
        self.spin_site = QDoubleSpinBox()
        self.spin_site.setRange(2.0, 30.0)
        self.spin_site.setSingleStep(1.0)
        self.spin_site.setValue(6.0)
        self.spin_site.setPrefix("site ")
        self.spin_site.setSuffix(" Å")
        self.spin_site.setToolTip("Residues with any atom within this distance of a displayed "
                                  "molecule are drawn, as whole residues")
        self.spin_site.valueChanged.connect(self.changed)
        row4.addWidget(self.spin_site)
        self.chk_site_h = QCheckBox("H")
        self.chk_site_h.setToolTip("Show hydrogens of the structure when it has any")
        self.chk_site_h.toggled.connect(self.changed)
        row4.addWidget(self.chk_site_h)
        self.chk_waters = QCheckBox("waters")
        self.chk_waters.setToolTip("Keep crystallographic waters inside the site")
        self.chk_waters.toggled.connect(self.changed)
        row4.addWidget(self.chk_waters)
        self.chk_comp_map = QCheckBox("complementarity surface")
        self.chk_comp_map.setToolTip("\n".join([
            "Draw the surface of the selected molecule coloured by how well its fields",
            "complement the pocket: green where they do, red where they fight.",
            "Compute it first in the Results panel, Complementarity tab."]))
        row4.addStretch(1)
        lay.addLayout(row4)
        # what is drawn on top of the site goes on its own row, for the same reason as the maps
        row4b = QHBoxLayout()
        row4b.addWidget(QLabel("    "))
        row4 = row4b
        self.chk_comp_map.toggled.connect(self.changed)
        row4.addWidget(self.chk_comp_map)
        self.chk_site_labels = QCheckBox("residue labels")
        self.chk_site_labels.setToolTip("\n".join([
            "Write the name and number of every residue of the site at its alpha carbon,",
            "and list them in the legend below"]))
        self.chk_site_labels.toggled.connect(self.changed)
        row4.addWidget(self.chk_site_labels)
        self.lbl_protein = QLabel("no structure loaded")
        self.lbl_protein.setStyleSheet("color: palette(mid)")
        row4.addWidget(self.lbl_protein)
        row4.addStretch(1)
        lay.addLayout(row4)
        self.lbl_legend = QLabel("")
        self.lbl_legend.setWordWrap(True)
        self.lbl_legend.setStyleSheet("color: palette(text); font-size: 9pt")
        lay.addWidget(self.lbl_legend)

    def label_kind(self) -> str:
        return ["none", "index", "type", "charge"][self.cmb_labels.currentIndex()]

    def style_key(self) -> str:
        return ["ballstick", "sticks", "lines", "spacefill"][self.cmb_style.currentIndex()]

    def level_fraction(self) -> float:
        return self.sld_level.value() / 100.0

    def opacity(self) -> float:
        return self.sld_alpha.value() / 100.0

    def shapfield_style(self) -> str:
        return ["solid", "mesh"][self.cmb_shapfield_style.currentIndex()]

    def protein_style(self) -> str:
        return ["lines", "sticks"][self.cmb_protein_style.currentIndex()]

    def set_protein_label(self, text: str) -> None:
        self.lbl_protein.setText(text)

    def set_fields(self, names_labels):
        cur = self.cmb_field.currentData()
        self.cmb_field.blockSignals(True)
        self.cmb_field.clear()
        for name, label in names_labels:
            self.cmb_field.addItem(label, name)
        if cur is not None:
            ix = self.cmb_field.findData(cur)
            if ix >= 0:
                self.cmb_field.setCurrentIndex(ix)
        self.cmb_field.blockSignals(False)

    def current_field(self) -> Optional[str]:
        return self.cmb_field.currentData()

    def set_endpoints(self, items) -> None:
        """Offer one entry per endpoint of a two-endpoint model, plus their difference."""
        self.cmb_endpoint.blockSignals(True)
        self.cmb_endpoint.clear()
        for label, key in list(items):
            self.cmb_endpoint.addItem(str(label), key)
        self.cmb_endpoint.blockSignals(False)
        on = self.cmb_endpoint.count() > 0
        self.lbl_endpoint.setVisible(on)
        self.cmb_endpoint.setVisible(on)

    def current_endpoint(self):
        # count, not visibility: a populated combo whose dock is hidden still holds the
        # user's choice, and gating on isVisible() would silently ignore it
        return self.cmb_endpoint.currentData() if self.cmb_endpoint.count() else None
