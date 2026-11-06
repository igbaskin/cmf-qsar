"""A periodic structure as the sample, in the application (plan item M13).

Everything the program can do with crystals -- the periodic field autocorrelation, the periodic
overlap kernel with its co-field inside the cell, the split of a prediction over the sites --
lived on the command line and in the library only.  This window is that side of the program on
screen: open a folder of CIF files or one structure file, give the structures their measured
values, fit a model, and read it on the cell itself.

What is drawn, and why it is not the molecular view
---------------------------------------------------
A crystal is not a molecule with a box around it.  It has no bond table, it has no single
"molecule" to centre on, and half of what the kernel sees lies outside the cell, in the periodic
images.  So the view has three parts the molecular view does not:

* **the cell**, as the twelve edges of its box, with the three lattice vectors marked;
* **the images**, every copy of the cell's atoms that comes within the kernel's own cutoff, drawn
  as dots so that thousands of them cost nothing.  They are what makes the periodic kernel
  periodic, and seeing them is how one checks that the cutoff reaches as far as it should;
* **the reading**: the co-field inside the cell where the kernel has one, and one sphere per site
  coloured by what that site adds to the prediction.

Which reading applies depends on the kernel, and the window says so rather than offering both:

* the **periodic overlap kernel** compares two cells point by point, so it has a co-field --
  C(r) inside the cell, positive where more field raises the property -- and the prediction
  splits over the sites by decomposition rule 3;
* the **periodic field autocorrelation** is a distribution of distances and can never say
  *where*, so it has no co-field.  Its reading is the co-pattern over the channels (which pair
  of field types at which distance) and the per-site split of plan item M5, which is exact.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDockWidget,
                               QDoubleSpinBox, QFileDialog,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QMainWindow, QMessageBox, QProgressBar, QPushButton, QSpinBox,
                               QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                               QVBoxLayout, QWidget)

from ..bonds import (MANY_BONDS, MAX_POLYHEDRA, coordination_polyhedra, is_bonding,
                     periodic_bonds, polyhedra_mesh)
from ..cofields import Grid, cofield_volume, isosurface
from ..dataio import save_model, write_predictions_csv
from ..fields import FieldSet
from ..model import CMFModel, grid_search
from ..periodic import (crystal_fields, crystal_table, crystal_targets, elements_in,
                        inorganic_fieldset, overlap_cutoff, periodic_channel_labels,
                        read_crystal_set, top_channels)
from .copattern_panel import CoPatternChart
from .viewer3d import (Viewer3D, cofield_colors, color_name as _cname, element_color,
                       signed_spheres)
from .workers import TaskWorker

#: the file types the open dialog offers; a folder is opened with the other entry of the menu
STRUCTURE_FILTER = ("Periodic structures (*.cif *.vasp *.extxyz *.xyz *.json *.traj POSCAR "
                    "CONTCAR);;All files (*)")
#: the physico-chemical fields offered, with the label each gets in the panel
PHYSCHEM = [("electrostatic", "Electrostatic (charges)"), ("steric", "Steric (atomic volume)"),
            ("prop:electronegativity", "Electronegativity"), ("prop:mass", "Mass")]
#: how many points the co-field grid over one cell may have before the spacing is coarsened
MAX_GRID_POINTS = 400_000
#: how the atoms of a cell can be drawn.  The first two show the atoms alone, which is what a
#: structure file states; the last two add the bonds of :mod:`cmfqsar.bonds`, perceived from the
#: distances for drawing only, which an organic molecule on a surface needs to be readable
ATOM_STYLES = ("balls", "dots", "ball and stick", "sticks")
#: radius of a drawn atom in the two bonded styles, as a fraction of its covalent radius
BONDED_ATOM_SCALE = 0.26


# --------------------------------------------------------------------------- #
# What the window computes, as plain functions
# --------------------------------------------------------------------------- #
def cell_grid(crystal, spacing: float = 0.3, margin: float = 3.0,
              max_points: int = MAX_GRID_POINTS) -> Grid:
    """A rectangular grid over the box one cell spans, cut to the atoms where there is vacuum.

    The grid is axis-aligned, as every grid in this program is, so for a cell that is not
    rectangular it covers a little more than the cell.  That costs nothing to look at: the
    co-field of a periodic model is itself periodic, so the extra corners carry the field of the
    neighbouring cell and not noise.

    Where the structure is *not* periodic -- the vacuum above a layer, which is most of its cell
    -- the grid is cut to the atoms plus `margin`, because there is no field out there and the
    points would only be spent.  The spacing is then coarsened until the grid fits in
    `max_points`, so a large supercell is drawn more coarsely rather than not at all.
    """
    A = np.asarray(crystal.cell, float).reshape(3, 3)
    corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) @ A
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    pbc = tuple(bool(p) for p in getattr(crystal, "pbc", (True, True, True)))
    if not all(pbc):
        xyz = np.asarray(crystal.coords, float).reshape(-1, 3)
        # the open directions of the cell carry the vacuum; keep only what the atoms occupy
        open_axes = np.abs(A[[k for k, p in enumerate(pbc) if not p]]).sum(axis=0) > 1e-9
        if len(xyz):
            lo = np.where(open_axes, np.maximum(lo, xyz.min(axis=0) - margin), lo)
            hi = np.where(open_axes, np.minimum(hi, xyz.max(axis=0) + margin), hi)
    s = float(spacing)
    while True:
        shape = tuple(max(int(np.ceil((hi[k] - lo[k]) / s)) + 1, 2) for k in range(3))
        if int(np.prod(shape)) <= int(max_points):
            return Grid(origin=lo.astype(float), spacing=s, shape=shape)
        s *= 1.25


def site_contributions_of(model: CMFModel, crystal) -> Tuple[np.ndarray, str]:
    """One number per site of `crystal`, and one line saying what the numbers are.

    The two periodic kernels split a prediction in two different ways, and both are exact:
    the overlap kernel over the atoms of the cell (decomposition rule 3), the autocorrelation
    over the sites of the cell (plan item M5).  The sum of the numbers is the prediction less
    the bias either way.
    """
    if getattr(model, "pbc_overlap", False):
        values = np.asarray(model.atom_contributions(crystal), float)
        how = ("one contribution per atom of the cell (decomposition rule 3 of the overlap "
               "kernel); they sum to the prediction less the bias")
    else:
        from ..sites import site_contributions
        values = np.asarray(site_contributions(model, crystal), float)
        how = ("one contribution per site, from the split of the autocorrelation over the sites "
               "(plan item M5); they sum to the prediction less the bias")
    return values, how


def reading_rows(model: CMFModel, fieldset: FieldSet, n: int = 20) -> Tuple[List, List]:
    """The table under the view: the co-pattern of the model, or the sites of one cell.

    Returns ``(column titles, rows)``.  For the autocorrelation kernel the co-pattern over the
    channels is the model's own reading and does not depend on which structure is selected; the
    per-site split does, so both are offered and the window shows whichever the user asks for.
    """
    labels = periodic_channel_labels(fieldset, model._ac_spec())
    rows = [[lab, f"{w:+.4f}", f"{sd:.4f}", f"{w * sd:+.4f}"]
            for lab, w, sd in top_channels(model, labels, n=n)]
    return ["Channel", "Weight", "Spread over the training set", "Weight x spread"], rows


def site_rows(values: Sequence[float], crystal, n: int = 20) -> Tuple[List, List]:
    """The sites of one cell that move its prediction most, largest magnitude first."""
    v = np.asarray(values, float)
    order = np.argsort(-np.abs(v))[:int(n)]
    rows = [[str(int(k) + 1), crystal.symbols[int(k)], f"{v[int(k)]:+.4f}",
             "  ".join(f"{x:7.3f}" for x in crystal.coords[int(k)])] for k in order]
    return ["Site", "Element", "Contribution", "Position (A)"], rows


class ColorScale(QWidget):
    """The colour scale of the site contributions, with the value at each end written on it.

    A colouring with no scale beside it says only which site is red; with the scale it says how
    much, in the units of the activity, which is what makes it a reading rather than a picture.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.lab_lo = QLabel("")
        self.bar = QLabel("")
        self.bar.setMinimumWidth(110)
        self.bar.setFixedHeight(12)
        self.bar.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgb(51,102,242), "
            "stop:0.5 rgb(225,225,230), stop:1 rgb(242,51,51)); border: 1px solid #888;")
        self.lab_hi = QLabel("")
        lay.addWidget(self.lab_lo)
        lay.addWidget(self.bar)
        lay.addWidget(self.lab_hi)
        self.set_range(0.0)

    def set_range(self, vmax: float, unit: str = "") -> None:
        """Show ±`vmax`; a scale of zero means nothing is drawn and the labels say so."""
        self.vmax = float(vmax)
        if not np.isfinite(self.vmax) or self.vmax <= 0:
            self.lab_lo.setText("")
            self.lab_hi.setText("no contributions")
            self.bar.setVisible(False)
            return
        self.bar.setVisible(True)
        tail = f" {unit}" if unit else ""
        self.lab_lo.setText(f"{-self.vmax:+.3g}{tail}")
        self.lab_hi.setText(f"{self.vmax:+.3g}{tail}")


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
class CrystalWindow(QMainWindow):
    """A data set of periodic structures: open it, fit a model, read it on the cell."""

    def __init__(self, parent=None, background: bool = True):
        super().__init__(parent)
        #: run the long steps in a worker thread, as the application does.  A script or a test
        #: that drives this window without an event loop sets it to False and every step then
        #: finishes before the call returns
        self.background = bool(background)
        self.setWindowTitle("CMF 3D-QSAR  -  crystals and other periodic structures")
        self.resize(1400, 900)

        self.atoms: list = []
        self.names: List[str] = []
        self.y: np.ndarray = np.zeros(0)
        self.subset: List[str] = []
        self.source = ""
        self.target_name = ""
        self._target_path: Optional[str] = None     # the CSV the measured values came from
        self._target_column: Optional[str] = None
        self.model: Optional[CMFModel] = None
        self.model_fieldset: Optional[FieldSet] = None
        self.predictions: Dict[int, float] = {}
        self.worker: Optional[TaskWorker] = None
        self._display: Dict[Tuple[int, str], object] = {}   # (index, fieldset tag) -> CrystalFields
        self._volume = None            # (structure index, field name, volume, grid)
        self._contributions = None     # (structure index, values, how)
        self._bonded = None            # (structure index, bonds.DrawableStructure)
        self._polyhedra = None         # (structure index, [bonds.Polyhedron])

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.viewer = Viewer3D()
        self.controls = self._build_controls()
        self.legend = QLabel("Open a folder of CIF files, or one structure file, to begin.")
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet("color: #9aa3b0; padding: 2px 6px;")
        lay.addWidget(self.controls)
        lay.addWidget(self.viewer, 1)
        lay.addWidget(self.legend)
        self.setCentralWidget(central)

        self._build_table_dock()
        self._build_model_dock()
        self._build_results_dock()
        self._build_menu()

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.status = QLabel("Ready")
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.progress)
        self._kernel_changed()

    # ------------------------------------------------------------------ build
    def _build_controls(self) -> QWidget:
        w = QWidget()
        outer = QVBoxLayout(w)
        outer.setContentsMargins(6, 4, 6, 2)
        outer.setSpacing(2)

        row1 = QHBoxLayout()
        self.chk_cell = QCheckBox("Cell box")
        self.chk_cell.setChecked(True)
        self.chk_images = QCheckBox("Periodic images")
        self.chk_images.setChecked(True)
        self.chk_images.setToolTip("Every copy of the cell's atoms within the radius beside it, "
                                   "which is what the kernel sums over")
        self.spin_images = QDoubleSpinBox()
        self.spin_images.setRange(0.0, 30.0)
        self.spin_images.setSingleStep(0.5)
        self.spin_images.setValue(6.0)
        self.spin_images.setSuffix(" A")
        self.cmb_atoms = QComboBox()
        self.cmb_atoms.addItems(ATOM_STYLES)
        self.cmb_atoms.setToolTip("\n".join([
            "How the atoms of the cell are drawn.",
            "",
            "'balls' and 'dots' show the atoms alone, which is the honest picture of a file that",
            "states where the atoms are and nothing about which of them are bonded.",
            "",
            "'ball and stick' and 'sticks' add bonds perceived from the distances, over the",
            "periodic images, so a bond that crosses the cell boundary is drawn rather than",
            "lost. A metal atom gets no bonds, because a distance rule would give it about",
            "twelve: a slab stays spheres and the organic molecule on it gets its bonds, which",
            "is the picture an adsorbate needs. The bonds are for looking at only -- no bond",
            "order, no charge and no field is read from them.",
            "",
            "What a metal gets instead is 'Coordination polyhedra' beside this box."]))
        self.chk_polyhedra = QCheckBox("Coordination polyhedra")
        self.chk_polyhedra.setToolTip("\n".join([
            "The first coordination shell of every metal atom, drawn as a polyhedron.",
            "",
            "The shell is found without any chemistry: the nearest neighbour sets a distance and",
            "every neighbour within 1.3 times that distance is a vertex, with the non-metal",
            "neighbours preferred where the atom has any. Neighbours are taken over the periodic",
            "images, so a polyhedron at the edge of the cell is whole.",
            "",
            "This is how an inorganic structure is read: a chain of corner-sharing octahedra",
            "says more than any number of sticks would."]))
        self.chk_labels = QCheckBox("Site numbers")
        self.chk_dark = QCheckBox("Dark background")
        self.chk_dark.setChecked(True)
        btn_reset = QPushButton("Reset view")
        btn_reset.clicked.connect(self.viewer.reset_view)
        for widget in (self.chk_cell, self.chk_images, QLabel("to"), self.spin_images,
                       QLabel("  Atoms:"), self.cmb_atoms, self.chk_polyhedra, self.chk_labels,
                       self.chk_dark, btn_reset):
            row1.addWidget(widget)
        row1.addStretch(1)

        row2 = QHBoxLayout()
        self.chk_cofield = QCheckBox("Co-field in the cell")
        self.cmb_field = QComboBox()
        self.cmb_field.setMinimumWidth(150)
        self.spin_level = QSpinBox()
        self.spin_level.setRange(1, 95)
        self.spin_level.setValue(35)
        self.spin_level.setSuffix(" % iso")
        self.spin_opacity = QDoubleSpinBox()
        self.spin_opacity.setRange(0.05, 1.0)
        self.spin_opacity.setSingleStep(0.05)
        self.spin_opacity.setValue(0.45)
        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setRange(0.1, 1.0)
        self.spin_spacing.setSingleStep(0.05)
        self.spin_spacing.setValue(0.3)
        self.spin_spacing.setSuffix(" A grid")
        self.btn_cofield = QPushButton("Compute co-field")
        self.btn_cofield.clicked.connect(self.compute_cofield)
        self.chk_sites = QCheckBox("Site contributions")
        self.scale = ColorScale()
        for widget in (self.chk_cofield, self.cmb_field, self.spin_level, self.spin_opacity,
                       self.spin_spacing, self.btn_cofield, QLabel("   "), self.chk_sites,
                       self.scale):
            row2.addWidget(widget)
        row2.addStretch(1)
        outer.addLayout(row1)
        outer.addLayout(row2)
        for box in (self.chk_cell, self.chk_images, self.chk_labels, self.chk_dark,
                    self.chk_cofield, self.chk_sites, self.chk_polyhedra):
            box.toggled.connect(self.update_display)
        self.spin_images.valueChanged.connect(self.update_display)
        self.spin_level.valueChanged.connect(self.update_display)
        self.spin_opacity.valueChanged.connect(self.update_display)
        self.cmb_atoms.currentIndexChanged.connect(self.update_display)
        self.cmb_field.currentIndexChanged.connect(self.update_display)
        return w

    def _build_table_dock(self) -> None:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(4, 4, 4, 4)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Structure", "Measured", "Set", "Predicted"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.currentCellChanged.connect(lambda *_: self.update_display())
        self.note = QLabel("No structures loaded.")
        self.note.setWordWrap(True)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.note)
        d = QDockWidget("Structures", self)
        d.setObjectName("Structures")
        d.setWidget(panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, d)

    def _build_model_dock(self) -> None:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)

        fields = QGroupBox("Fields")
        ff = QVBoxLayout(fields)
        self.chk_physchem = {}
        for name, label in PHYSCHEM:
            box = QCheckBox(label)
            box.setChecked(name in ("electrostatic", "steric", "prop:electronegativity"))
            self.chk_physchem[name] = box
            ff.addWidget(box)
        form = QFormLayout()
        self.cmb_elements = QComboBox()
        self.cmb_elements.addItems(["one per element (auto)", "none"])
        self.spin_min_structures = QSpinBox()
        self.spin_min_structures.setRange(1, 999)
        self.spin_min_structures.setValue(3)
        self.cmb_charges = QComboBox()
        self.cmb_charges.addItems(["oxidation", "electronegativity", "file", "dft"])
        self.cmb_charges.setToolTip(
            "Where the weights of the electrostatic field come from. 'oxidation' guesses the "
            "oxidation states from the composition, which is what an ionic solid asks for and "
            "is slow for a large cell of many elements; 'electronegativity' is a cheap proxy "
            "that always works; 'file' and 'dft' read the charges a calculation wrote into the "
            "structure (Bader charges, for example)")
        self.cmb_pbc = QComboBox()
        self.cmb_pbc.addItems(["as the files say", "abc: a crystal", "ab: a layer",
                               "a: a chain"])
        self.cmb_pbc.setToolTip("Which lattice directions are periodic. A layer or a chain is "
                                "summed only along its own directions, so no pair is counted "
                                "across the vacuum, and the uniform background is taken per "
                                "unit area or repeat length instead of per volume")
        self.cmb_pbc.currentIndexChanged.connect(self._pbc_changed)
        form.addRow("Indicator fields", self.cmb_elements)
        form.addRow("In at least", self.spin_min_structures)
        form.addRow("Charges", self.cmb_charges)
        form.addRow("Periodic in", self.cmb_pbc)
        ff.addLayout(form)
        lay.addWidget(fields)

        kernel = QGroupBox("Kernel")
        kf = QFormLayout(kernel)
        self.cmb_kernel = QComboBox()
        self.cmb_kernel.addItems(["periodic field autocorrelation (any cells)",
                                  "periodic overlap (cells on one lattice)"])
        self.cmb_kernel.currentIndexChanged.connect(self._kernel_changed)
        self.spin_cutoff = QDoubleSpinBox()
        self.spin_cutoff.setRange(2.0, 30.0)
        self.spin_cutoff.setValue(10.0)
        self.spin_cutoff.setSuffix(" A")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.05, 2.0)
        self.spin_sigma.setSingleStep(0.05)
        self.spin_sigma.setValue(0.5)
        self.spin_sigma.setSuffix(" A")
        self.spin_taper = QDoubleSpinBox()
        self.spin_taper.setRange(0.0, 6.0)
        self.spin_taper.setValue(2.0)
        self.spin_taper.setSuffix(" A")
        self.chk_background = QCheckBox("Remove the uniform background")
        self.chk_background.setChecked(True)
        self.ed_alphas = QLineEdit("1")
        self.ed_lams = QLineEdit("1e-4,1e-3,1e-2,0.1,1")
        kf.addRow("Kernel", self.cmb_kernel)
        kf.addRow("Neighbours to", self.spin_cutoff)
        kf.addRow("Channel width", self.spin_sigma)
        kf.addRow("Taper", self.spin_taper)
        kf.addRow("", self.chk_background)
        kf.addRow("Attenuation a", self.ed_alphas)
        kf.addRow("Ridge l", self.ed_lams)
        lay.addWidget(kernel)

        fitting = QGroupBox("Fitting")
        gf = QFormLayout(fitting)
        self.spin_folds = QSpinBox()
        self.spin_folds.setRange(2, 50)
        self.spin_folds.setValue(5)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 10_000)
        self.spin_test = QDoubleSpinBox()
        self.spin_test.setRange(0.0, 0.9)
        self.spin_test.setSingleStep(0.05)
        self.spin_test.setValue(0.0)
        gf.addRow("Cross-validation folds", self.spin_folds)
        gf.addRow("Random seed", self.spin_seed)
        gf.addRow("Test fraction", self.spin_test)
        lay.addWidget(fitting)

        buttons = QHBoxLayout()
        self.btn_build = QPushButton("Build model")
        self.btn_build.clicked.connect(self.build_model)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.cancel_worker)
        self.btn_cancel.setEnabled(False)
        buttons.addWidget(self.btn_build)
        buttons.addWidget(self.btn_cancel)
        lay.addLayout(buttons)
        lay.addStretch(1)

        d = QDockWidget("Model", self)
        d.setObjectName("Model")
        d.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, d)

    def _build_results_dock(self) -> None:
        self.tabs = QTabWidget()
        self.text_stats = QTextBrowser()
        self.reading = QTableWidget(0, 4)
        self.reading.verticalHeader().setVisible(False)
        self.cmb_reading = QComboBox()
        self.cmb_reading.addItems(["the model: the co-pattern over the channels",
                                   "the selected structure: the sites"])
        self.cmb_reading.currentIndexChanged.connect(self.update_reading)
        wrap = QWidget()
        wl = QVBoxLayout(wrap)
        wl.setContentsMargins(4, 4, 4, 4)
        head = QHBoxLayout()
        head.addWidget(QLabel("Show"))
        head.addWidget(self.cmb_reading)
        head.addStretch(1)
        wl.addLayout(head)
        # the ranked channels on the left, the same numbers as a curve on the right: the table
        # says which channel, the chart says what the shape over distance is (plan item NT9)
        split = QHBoxLayout()
        split.addWidget(self.reading, 1)
        self.chart = CoPatternChart()
        split.addWidget(self.chart, 1)
        wl.addLayout(split, 1)
        self.text_log = QTextBrowser()
        self.tabs.addTab(self.text_stats, "Statistics")
        self.tabs.addTab(wrap, "Reading")
        self.tabs.addTab(self.text_log, "Log")
        d = QDockWidget("Results", self)
        d.setObjectName("Results")
        d.setWidget(self.tabs)
        d.setMinimumHeight(170)
        self.addDockWidget(Qt.BottomDockWidgetArea, d)

    def _build_menu(self) -> None:
        fm = self.menuBar().addMenu("&File")
        self._act(fm, "Open a folder of structures...", self.open_folder, "Ctrl+O")
        self._act(fm, "Open one structure file...", self.open_file)
        self._act(fm, "Read the measured values from a CSV...", self.import_targets)
        fm.addSeparator()
        self._act(fm, "Save model...", self.save_model_file)
        self._act(fm, "Export predictions (CSV)...", self.export_predictions)
        self._act(fm, "Save 3D view as image...", self.save_screenshot)
        fm.addSeparator()
        self._act(fm, "Close this window", self.close)
        vm = self.menuBar().addMenu("&View")
        for d in self.findChildren(QDockWidget):
            vm.addAction(d.toggleViewAction())
        self._act(vm, "Centre on the selected cell", self.center_current)

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        menu.addAction(a)
        return a

    # ------------------------------------------------------------------ helpers
    def log(self, text: str) -> None:
        self.text_log.append(time.strftime("%H:%M:%S  ") + text)
        self.status.setText(text.splitlines()[0][:140])

    def error(self, text: str) -> None:
        self.log("ERROR: " + text.splitlines()[0])
        if self.background:
            QMessageBox.critical(self, "Error", text)

    def _tell(self, title: str, text: str) -> None:
        """Say something the user has to act on: a dialog in the application, the log in a script.

        A modal dialog waits for a person, so a script or a test driving this window
        (``background=False``) must never raise one: it reads the log instead.
        """
        self.log(f"{title}: {text}")
        if self.background:
            QMessageBox.information(self, title, text)

    def run_task(self, fn, on_done, label="Working"):
        if not self.background:
            self.status.setText(label)
            on_done(fn(lambda msg, frac: self.status.setText(str(msg)[:140]), lambda: False))
            return
        if self.worker is not None and self.worker.isRunning():
            self._tell("Busy", "A computation is already running.")
            return
        self.worker = TaskWorker(fn, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(lambda res: self._task_finished(res, on_done))
        self.worker.failed.connect(self._task_failed)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.btn_build.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.status.setText(label)
        self.worker.start()

    def _on_progress(self, msg, frac):
        self.progress.setValue(int(frac * 100))
        self.status.setText(msg[:140])

    def _task_finished(self, res, on_done):
        self.progress.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        try:
            on_done(res)
        except Exception as exc:                   # noqa: BLE001
            import traceback
            self.error(f"{exc}\n\n{traceback.format_exc()}")

    def _task_failed(self, msg):
        self.progress.setVisible(False)
        self.btn_build.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        if msg == "Cancelled":
            self.log("Cancelled")
        else:
            self.error(msg)

    def cancel_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status.setText("Cancelling...")

    def _kernel_changed(self, *_):
        overlap = self.cmb_kernel.currentIndex() == 1
        for w in (self.spin_cutoff, self.spin_sigma, self.spin_taper, self.chk_background):
            w.setEnabled(not overlap)
        if overlap and self.ed_alphas.text().strip() == "1":
            self.ed_alphas.setText("0.5,1,2")
        self.chk_cofield.setEnabled(overlap)
        if not overlap:
            self.chk_cofield.setChecked(False)
        self.btn_cofield.setEnabled(overlap)

    @staticmethod
    def _floats(text: str, default: Sequence[float]) -> List[float]:
        try:
            vals = [float(x) for x in text.replace(";", ",").split(",") if x.strip()]
        except ValueError:
            return list(default)
        return vals or list(default)

    # ------------------------------------------------------------------ loading
    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open a folder of periodic structures")
        if path:
            self.load(path)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open a periodic structure file", "",
                                              STRUCTURE_FILTER)
        if path:
            self.load(path)

    def _pbc_text(self) -> Optional[str]:
        """The letters the periodic-directions box stands for, or None for 'as the files say'."""
        return [None, "abc", "ab", "a"][self.cmb_pbc.currentIndex()]

    def _pbc_changed(self, *_):
        """Re-read the set, because changing the periodic directions can move the atoms.

        A layer a file leaves split across the cell boundary is gathered about the middle of the
        cell, which only makes sense once the program knows which direction is the vacuum.
        """
        if self.source:
            self.load(self.source, target_file=self._target_path, target_column=self._target_column)

    def load(self, path: str, target_file: Optional[str] = None, target_key: Optional[str] = None,
             target_column: Optional[str] = None):
        """Read a folder of structure files, or one file, and show what came back."""
        pbc = self._pbc_text()

        def task(progress, cancel):
            progress(f"Reading {os.path.basename(path)}", 0.2)
            atoms, names = read_crystal_set(path, pbc=pbc)
            progress(f"{len(atoms)} structures", 0.9)
            y = crystal_targets(atoms, names, key=target_key, csv=target_file,
                                column=target_column)
            return atoms, names, y

        def done(out):
            self.atoms, self.names, self.y = out[0], list(out[1]), np.asarray(out[2], float)
            self.source = path
            self._target_path, self._target_column = target_file, target_column
            self.target_name = target_file or target_key or ""
            self.subset = ["train"] * len(self.names)
            self.model = self.model_fieldset = None
            self.predictions.clear()
            self._display.clear()
            self._volume = self._contributions = None
            self._bonded = self._polyhedra = None
            self.cmb_field.clear()
            self.reading.setRowCount(0)
            self.text_stats.clear()
            self.populate()
            self.log(f"Read {len(self.atoms)} periodic structures from {path}"
                     + (f"; measured values from {self.target_name}" if self.target_name else
                        "; no measured values yet (File -> Read the measured values from a CSV)"))
            if self.names:
                self.table.setCurrentCell(0, 0)
                self.center_current()
            self.update_display()

        self.run_task(task, done, "Reading structures")

    def import_targets(self, path: str = "", column: Optional[str] = None):
        """Read the measured values from a CSV whose first column holds the structure names.

        A table with several numeric columns -- which is what a prepared data set usually is --
        asks which column is the property; a plain ``name,value`` file asks nothing.
        """
        if not self.names:
            self._tell("No structures", "Open a set of structures first.")
            return
        if not path:
            path, _ = QFileDialog.getOpenFileName(self, "Measured values (CSV)", "",
                                                  "CSV (*.csv *.txt);;All files (*)")
        if not path:
            return
        from ..dataio import activity_columns
        columns = activity_columns(path)
        if column is None and len(columns) > 1:
            if not self.background:
                column = columns[0]                # a script takes the first numeric column
            else:
                from PySide6.QtWidgets import QInputDialog
                column, ok = QInputDialog.getItem(
                    self, "Which column", "The column that holds the measured property:",
                    columns, 0, False)
                if not ok:
                    return
        self.y = crystal_targets(self.atoms, self.names, csv=path, column=column)
        self._target_path, self._target_column = path, column
        self.target_name = os.path.basename(path) + (f" [{column}]" if column else "")
        got = int(np.isfinite(self.y).sum())
        self.populate()
        self.log(f"Read {got} measured values of {len(self.names)} structures from "
                 f"{self.target_name}"
                 + ("" if got else "; no name in the file matches a structure name"))

    def populate(self) -> None:
        if len(self.subset) != len(self.names):
            self.subset = ["train"] * len(self.names)
        if len(self.y) != len(self.names):
            y = np.full(len(self.names), np.nan)
            y[:min(len(self.y), len(y))] = self.y[:min(len(self.y), len(y))]
            self.y = y
        self.table.setRowCount(len(self.names))
        for i, name in enumerate(self.names):
            self.table.setItem(i, 0, QTableWidgetItem(name))
            y = self.y[i] if i < len(self.y) else np.nan
            self.table.setItem(i, 1, QTableWidgetItem("" if not np.isfinite(y) else f"{y:.4g}"))
            self.table.setItem(i, 2, QTableWidgetItem(self.subset[i]))
            p = self.predictions.get(i)
            self.table.setItem(i, 3, QTableWidgetItem("" if p is None else f"{p:.4g}"))
        self.table.resizeColumnsToContents()
        n_y = int(np.isfinite(self.y).sum()) if len(self.y) else 0
        atoms = [len(a) for a in self.atoms]
        self.note.setText(
            f"{len(self.names)} structures, {n_y} with a measured value, "
            f"{min(atoms) if atoms else 0} to {max(atoms) if atoms else 0} atoms per cell"
            + (f"; values from {self.target_name}" if self.target_name else ""))

    def current_index(self) -> int:
        row = self.table.currentRow()
        return int(row) if 0 <= row < len(self.names) else -1

    # ------------------------------------------------------------------ fitting
    def _fieldset(self, indices: Sequence[int]) -> FieldSet:
        physchem = [name for name, _ in PHYSCHEM if self.chk_physchem[name].isChecked()]
        if self.cmb_elements.currentIndex() == 0:
            elements = elements_in([self.atoms[i] for i in indices],
                                   self.spin_min_structures.value())
        else:
            elements = []
        return inorganic_fieldset(elements, physchem=physchem,
                                  charge_model=self.cmb_charges.currentText(), alpha=1.0)

    def build_model(self):
        if not self.names:
            self._tell("No structures", "Open a set of structures first.")
            return
        ok = np.isfinite(self.y)
        if int(ok.sum()) < 4:
            self._tell(
                "No measured values",
                "At least four structures need a measured value. Read them from a CSV "
                "(File -> Read the measured values from a CSV), with one line 'name,value' per "
                "structure; the names are the ones in the table.")
            return
        idx = np.flatnonzero(ok)
        rng = np.random.default_rng(self.spin_seed.value())
        te = np.array([], int)
        if self.spin_test.value() > 0:
            perm = rng.permutation(idx)
            n_te = int(round(self.spin_test.value() * len(idx)))
            te, idx = np.sort(perm[:n_te]), np.sort(perm[n_te:])
        fs = self._fieldset(idx)
        if not fs.specs:
            self._tell("No fields", "Select at least one field.")
            return
        overlap = self.cmb_kernel.currentIndex() == 1
        alphas = self._floats(self.ed_alphas.text(), [1.0])
        lams = self._floats(self.ed_lams.text(), [1e-3])
        folds, seed = self.spin_folds.value(), self.spin_seed.value()
        cutoff, sigma, taper = (self.spin_cutoff.value(), self.spin_sigma.value(),
                                self.spin_taper.value())
        background = self.chk_background.isChecked()
        atoms, names, y = list(self.atoms), list(self.names), self.y.copy()

        def task(progress, cancel):
            progress("Computing the fields of every cell", 0.05)
            table = crystal_table(atoms, fs, names)
            if overlap:
                model = CMFModel(fieldset=fs, method="krr", normalize=True, pbc_overlap=True)
            else:
                model = CMFModel(fieldset=fs, method="krr", normalize=True,
                                 kernel_source="periodic", ac_sigma=sigma, ac_d_max=cutoff,
                                 pbc_taper=taper, pbc_background=background)
            train = table.subset([int(i) for i in idx])
            res = grid_search(train, y[idx], model, alphas, lams=lams, folds=folds, seed=seed,
                              progress=lambda m, f: progress(m, 0.1 + 0.75 * f), cancel=cancel)
            progress("Final fit and cross-validation", 0.9)
            model.fit(train, y[idx], cv_folds=folds, cv_seed=seed, reuse_kernels=True)
            fitted = np.asarray(model.stats.extra["y_fit"], float)
            test_pred = None
            if len(te):
                progress("Predicting the test set", 0.95)
                test = table.subset([int(i) for i in te])
                model.evaluate_test(test, y[te])
                test_pred = np.asarray(model.stats.extra.get("y_test_pred"), float)
            return {"model": model, "fieldset": fs, "res": res, "train": list(map(int, idx)),
                    "test": list(map(int, te)), "fitted": fitted, "test_pred": test_pred,
                    "overlap": overlap}

        self.run_task(task, self._model_built, "Building the model")

    def _model_built(self, out):
        self.model, self.model_fieldset = out["model"], out["fieldset"]
        self._display.clear()
        self._volume = self._contributions = None
        self._bonded = self._polyhedra = None
        res = out["res"]
        self.subset = ["train" if i in set(out["train"]) else
                       "test" if i in set(out["test"]) else "unused"
                       for i in range(len(self.names))]
        self.predictions = {i: float(v) for i, v in zip(out["train"], out["fitted"])}
        if out["test_pred"] is not None:
            self.predictions.update({i: float(v) for i, v in zip(out["test"], out["test_pred"])})
        self.populate()
        self.cmb_field.clear()
        self.cmb_field.addItems([s.label for s in self.model.fieldset.specs if s.enabled])
        self.cmb_reading.model().item(0).setEnabled(not out["overlap"])
        if out["overlap"]:
            self.cmb_reading.setCurrentIndex(1)
        kernel = ("the periodic overlap kernel, which compares two cells point by point"
                  if out["overlap"] else
                  "the periodic field autocorrelation, which is invariant to the choice of "
                  "cell, origin and orientation")
        self.text_stats.setPlainText(
            f"{len(out['train'])} training cells, {len(out['test'])} test cells; {kernel}.\n"
            f"Best: {res.best_params}, attenuation {res.best_alpha:g}\n\n"
            + self.model.stats.summary())
        self.log(f"Model built: {res.best_params}, RMSEcv = {res.best_rmse:.4g}, "
                 f"q2 = {res.best_q2:.3f}")
        self.spin_images.setValue(round(self._kernel_cutoff(), 1))
        self.update_reading()
        self.update_display()

    def _kernel_cutoff(self) -> float:
        """How far the fitted kernel reaches, which is how far the images are worth drawing."""
        if self.model is None:
            return float(self.spin_images.value())
        if getattr(self.model, "pbc_overlap", False):
            return overlap_cutoff([s for s in self.model.fieldset.specs if s.enabled])
        return float(self.model._ac_spec().r_max)

    # ------------------------------------------------------------------ reading
    def update_reading(self, *_):
        self.reading.setRowCount(0)
        if self.model is None or not self.model.is_linear:
            self.chart.clear("Build a model on the periodic field autocorrelation to see its "
                             "co-pattern.")
            return
        cur = self.current_index()
        channels = (self.cmb_reading.currentIndex() == 0
                    and not getattr(self.model, "pbc_overlap", False))
        if channels:
            self.chart.set_model(self.model, self.model.fieldset)
        else:
            self.chart.clear(
                "The chart is the co-pattern of the model: choose 'the model: the co-pattern "
                "over the channels' above."
                if not getattr(self.model, "pbc_overlap", False) else
                "The periodic overlap kernel has a co-field, which is the map in the 3D view; "
                "the chart is for the kernel that has none.")
        try:
            if channels:
                cols, rows = reading_rows(self.model, self.model.fieldset)
            elif cur < 0:
                return
            else:
                crystal = self._crystal(cur, self.model.fieldset)
                values, _how = self._site_values(cur)
                cols, rows = site_rows(values, crystal)
        except (ValueError, RuntimeError) as exc:
            self.log(f"the reading is not available: {exc}")
            return
        self.reading.setColumnCount(len(cols))
        self.reading.setHorizontalHeaderLabels(cols)
        self.reading.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, text in enumerate(row):
                self.reading.setItem(r, c, QTableWidgetItem(text))
        self.reading.resizeColumnsToContents()

    def _site_values(self, index: int) -> Tuple[np.ndarray, str]:
        if self._contributions is not None and self._contributions[0] == index:
            return self._contributions[1], self._contributions[2]
        crystal = self._crystal(index, self.model.fieldset)
        values, how = site_contributions_of(self.model, crystal)
        self._contributions = (index, values, how)
        return values, how

    def compute_cofield(self):
        """Evaluate the co-field of the fitted model inside the selected cell."""
        cur = self.current_index()
        if self.model is None or cur < 0:
            self._tell("No model", "Build a model on the overlap kernel first, then select "
                       "a structure.")
            return
        if not getattr(self.model, "pbc_overlap", False):
            self._tell(
                "No co-field",
                "The periodic field autocorrelation is a distribution of distances, so it has "
                "no co-field: it cannot say where in the cell anything is. Its reading is the "
                "co-pattern over the channels and the per-site split, both in the Reading tab. "
                "Build the model on the periodic overlap kernel for a co-field.")
            return
        names = [s.name for s in self.model.fieldset.specs if s.enabled]
        labels = [s.label for s in self.model.fieldset.specs if s.enabled]
        if self.cmb_field.currentIndex() < 0:
            return
        f = self.cmb_field.currentIndex()
        field_name = names[f]
        crystal = self._crystal(cur, self.model.fieldset)
        grid = cell_grid(crystal, spacing=self.spin_spacing.value())
        model = self.model

        def task(progress, cancel):
            progress(f"The {labels[f]} co-field on {int(np.prod(grid.shape))} points", 0.1)
            t0 = time.time()
            volume = cofield_volume(model, f, grid, include_scale=True)
            return volume, time.time() - t0

        def done(out):
            volume, seconds = out
            self._volume = (cur, field_name, labels[f], volume, grid)
            self.chk_cofield.setChecked(True)
            self.log(f"Co-field {labels[f]} in the cell of {self.names[cur]}: "
                     f"{grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} points at "
                     f"{grid.spacing:.2f} A, {seconds:.1f} s")
            self.update_display()

        self.run_task(task, done, "Computing the co-field")

    # ------------------------------------------------------------------ drawing
    def _crystal(self, index: int, fieldset: Optional[FieldSet] = None):
        """The structure at `index` with its field weights, kept so it is built once."""
        fs = fieldset if fieldset is not None else FieldSet(specs=[])
        tag = "|".join(s.name for s in fs.specs) or "(none)"
        key = (int(index), tag)
        if key not in self._display:
            self._display[key] = crystal_fields(self.atoms[index], fs, self.names[index])
        return self._display[key]

    def center_current(self):
        cur = self.current_index()
        if cur < 0:
            return
        crystal = self._crystal(cur)
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)],
                           float) @ np.asarray(crystal.cell, float)
        self.viewer.center_on(corners)

    def update_display(self, *_):
        v = self.viewer
        for g in ("cell", "atoms", "images", "cofield", "sites", "labels", "bonds",
                  "polyhedra"):
            v.clear_group(g)
        v.set_dark(self.chk_dark.isChecked())
        cur = self.current_index()
        if cur < 0:
            self.legend.setText("Open a folder of CIF files, or one structure file, to begin.")
            return
        crystal = self._crystal(cur, self.model.fieldset if self.model is not None else None)
        lines = []
        if self.chk_cell.isChecked():
            v.add_cell(crystal.cell, "cell")
            a, b, c = np.linalg.norm(np.asarray(crystal.cell, float), axis=1)
            lines.append(f"{crystal.name}: {crystal.n_atoms} atoms, cell {a:.2f} x {b:.2f} x "
                         f"{c:.2f} A, periodic in "
                         f"{crystal.n_periodic} direction{'s' if crystal.n_periodic != 1 else ''} "
                         f"(a red, b green, c blue)")
        lines += self._draw_atoms(cur, crystal)
        lines += self._draw_polyhedra(cur, crystal)
        if self.chk_images.isChecked() and self.spin_images.value() > 0:
            from ..periodic import cell_image_indices
            coords, index = cell_image_indices(crystal, float(self.spin_images.value()))
            n_cell = crystal.n_atoms
            image_coords, image_index = coords[n_cell:], index[n_cell:]
            v.add_points(image_coords, [crystal.symbols[i] for i in image_index], "images",
                         size=5.0, alpha=0.40)
            reach = ("the kernel's own cutoff" if self.model is not None
                     and abs(self.spin_images.value() - self._kernel_cutoff()) < 0.05
                     else "the radius on the control strip")
            lines.append(f"{len(image_coords)} image atoms within "
                         f"{self.spin_images.value():g} A of the cell ({reach})")
        if self.chk_labels.isChecked():
            v.add_labels(crystal.coords,
                         [f"{i + 1} {s}" for i, s in enumerate(crystal.symbols)], "labels",
                         size=8)
        lines += self._draw_cofield(cur)
        lines += self._draw_sites(cur, crystal)
        self.legend.setText("   |   ".join(lines))

    def _drawable(self, index: int, crystal):
        """The cell with its perceived bonds, kept so one cell is perceived once."""
        if self._bonded is not None and self._bonded[0] == index:
            return self._bonded[1]
        drawable = periodic_bonds(crystal)
        self._bonded = (int(index), drawable)
        return drawable

    def _draw_atoms(self, cur: int, crystal) -> List[str]:
        """Draw the atoms of the cell in the chosen representation, and say what was drawn.

        The two bonded styles draw the bonds of :mod:`cmfqsar.bonds` as well, and with them the
        image atoms a bond ends on, so a bond that crosses the cell boundary is a bond on screen
        and not a stick into nothing.  Atoms that have no bonds -- every metal atom, and a lone
        ion -- are drawn as spheres in both styles, so a slab does not disappear when the sticks
        are the only thing asked for.
        """
        v = self.viewer
        style = self.cmb_atoms.currentText()
        if style == "balls":
            v.add_atoms(crystal.coords, crystal.symbols, "atoms", radii=crystal.radii)
            return []
        if style == "dots":
            v.add_points(crystal.coords, crystal.symbols, "atoms", size=9.0, alpha=1.0)
            return []
        d = self._drawable(cur, crystal)
        if not len(d.bonds):
            # nothing to stick together: fall back to the spheres rather than an empty view
            v.add_atoms(crystal.coords, crystal.symbols, "atoms", radii=crystal.radii)
            return [f"bonds: {d.describe()}; the atoms are drawn as balls instead"]
        many = len(d.bonds) > MANY_BONDS
        if style == "ball and stick":
            v.add_atoms(d.coords, d.symbols, "atoms", radii=d.radii, scale=BONDED_ATOM_SCALE)
        else:
            bonded = set(np.unique(d.bonds).tolist())
            lone = np.array([i for i in range(len(d.coords)) if i not in bonded], dtype=int)
            if len(lone):
                v.add_atoms(d.coords[lone], [d.symbols[int(i)] for i in lone], "atoms",
                            radii=d.radii[lone], scale=BONDED_ATOM_SCALE)
        v.add_structure(d.coords, d.bonds, d.symbols, "bonds",
                        style="lines" if many else "sticks",
                        carbon_color=element_color("C"), radius=0.12, width=2.0)
        text = f"{style}: {d.describe()}"
        if d.n_images_drawn:
            text += f"; {d.n_images_drawn} image atoms drawn with them"
        if many:
            text += (f"; more than {MANY_BONDS} sticks, so the bonds are drawn as lines")
        text += ". The bonds are perceived from the distances for drawing only"
        return [text]

    def _draw_polyhedra(self, cur: int, crystal) -> List[str]:
        if not self.chk_polyhedra.isChecked():
            return []
        if self._polyhedra is not None and self._polyhedra[0] == cur:
            polys = self._polyhedra[1]
        else:
            polys = coordination_polyhedra(crystal)
            self._polyhedra = (int(cur), polys)
        if not polys:
            return ["coordination polyhedra: no atom of this cell has a shell of three or more "
                    "neighbours that gets none of the bonds"]
        mesh = polyhedra_mesh(polys)
        if mesh is None:
            return []
        self.viewer.add_polyhedra(mesh[0], mesh[1], "polyhedra",
                                  alpha=float(self.spin_opacity.value()))
        kinds: Dict[str, int] = {}
        for p in polys:
            kinds[p.label()] = kinds.get(p.label(), 0) + 1
        shown = sorted(kinds.items(), key=lambda kv: -kv[1])[:4]
        flat = sum(1 for p in polys if p.planar)
        text = ("coordination polyhedra: "
                + ", ".join(f"{v} x {k}" for k, v in shown)
                + (" ..." if len(kinds) > len(shown) else ""))
        if flat:
            text += f"; {flat} of them flat, drawn as polygons"
        # a cell can hold more metal atoms than polyhedra, and the reason is worth saying: on a
        # disordered structure a site whose nearest neighbour is much closer than the next has a
        # shell of one atom, which is not a shape
        metals = sum(1 for s in crystal.symbols if not is_bonding(s))
        if len(polys) >= MAX_POLYHEDRA:
            text += f"; the first {MAX_POLYHEDRA} of {metals} metal atoms"
        elif metals > len(polys):
            text += (f"; {metals - len(polys)} of the {metals} metal atoms have a shell of fewer "
                     f"than three neighbours, which is not a shape")
        return [text]

    def _draw_cofield(self, cur: int) -> List[str]:
        if not self.chk_cofield.isChecked():
            return []
        if self._volume is None:
            return ["co-field: click 'Compute co-field'"]
        index, field_name, label, volume, grid = self._volume
        if index != cur:
            return [f"co-field: computed for {self.names[index]}; compute it again for this cell"]
        amax = float(np.abs(volume).max())
        if amax <= 0:
            return [f"the {label} co-field is zero everywhere in this cell"]
        frac = self.spin_level.value() / 100.0
        pos_c, neg_c = cofield_colors(field_name)
        for sign, col in ((1, pos_c), (-1, neg_c)):
            s = isosurface(volume, sign * frac * amax, grid)
            if s is not None:
                self.viewer.add_isosurface(s[0], s[1], col, "cofield",
                                           alpha=float(self.spin_opacity.value()))
        return [f"co-field {label} inside the cell: {_cname(pos_c)} where more field raises "
                f"the property, {_cname(neg_c)} where it lowers it, "
                f"iso = +/-{frac * amax:.3g}"]

    def _draw_sites(self, cur: int, crystal) -> List[str]:
        if not self.chk_sites.isChecked():
            self.scale.set_range(0.0)
            return []
        if self.model is None or not self.model.is_linear:
            self.scale.set_range(0.0)
            return ["site contributions: build a linear model first"]
        try:
            values, how = self._site_values(cur)
        except (ValueError, RuntimeError) as exc:
            self.scale.set_range(0.0)
            return [f"site contributions: {exc}"]
        radii, colors, vmax = signed_spheres(values)
        self.scale.set_range(vmax)
        if vmax <= 0:
            return ["every site contributes nothing to this prediction"]
        self.viewer.add_spheres(crystal.coords, radii, colors, "sites", alpha=0.5)
        total = float(np.sum(values)) + float(self.model.bias)
        text = (f"sites: red adds to the prediction, blue subtracts, largest "
                f"+/-{vmax:.3g}; {how}")
        predicted = self.predictions.get(cur)
        if predicted is None:
            text += f"; with the bias they sum to {total:.4g}"
        elif abs(predicted - total) <= max(1e-6, 1e-3 * abs(predicted)):
            text += f"; they sum to the prediction, {total:.4g}"
        else:
            # the periodic overlap integral runs over the cell of one structure and the images of
            # the other, so the two orders agree only on one lattice; the split is one order
            text += (f"; they sum to {total:.4g} against the prediction {predicted:.4g}: the two "
                     f"orders of the periodic integral disagree by {abs(predicted - total):.2g}, "
                     f"as they do when the cells do not stand exactly on one lattice "
                     f"(periodic.overlap_asymmetry measures it)")
        return [text]

    # ------------------------------------------------------------------ output
    def save_model_file(self):
        if self.model is None:
            self._tell("No model", "Build a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model", "", "CMF model (*.cmf)")
        if not path:
            return
        save_model(path, self.model, extra={"dataset": self.source, "task": "regression",
                                            "activity": self.target_name})
        self.log(f"Model saved to {path}; `cmf predict {os.path.basename(path)} <structures>` "
                 f"applies it to new cells")

    def export_predictions(self):
        if not self.predictions:
            self._tell("No predictions", "Build a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export predictions", "", "CSV (*.csv)")
        if not path:
            return
        pred = np.array([self.predictions.get(i, np.nan) for i in range(len(self.names))])
        write_predictions_csv(path, self.names, self.y, pred, self.subset)
        self.log(f"Predictions written to {path}")

    def save_screenshot(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save 3D view", "", "PNG image (*.png)")
        if path:
            self.viewer.screenshot(path)
            self.log(f"3D view written to {path}")
