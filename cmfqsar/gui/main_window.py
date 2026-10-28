"""Main window of the CMF 3D-QSAR desktop application."""
from __future__ import annotations

import os
import re
import time
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QFormLayout,
                               QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QCheckBox,
                               QSpinBox, QDoubleSpinBox, QTextBrowser, QVBoxLayout, QHBoxLayout,
                               QWidget, QComboBox, QLineEdit, QScrollArea, QPushButton)
from rdkit import Chem

from .. import __version__
from ..align import align_dataset
from ..cofields import (Grid, cofield_variance, cofield_volume, export_cube, isosurface,
                        make_grid, mask_noise, molfield_volume, noise_level, overlap_volume)
from ..conformers import CONFORMER_ALIGNMENTS
from ..contract import check_molecules, summary as model_summary
from ..dataio import Dataset, load_dataset, load_model, numeric_properties, save_model, write_predictions_csv, write_sdf
from ..fields import AtomTable, FieldSet, MoleculeFields, build_fields, cif_types_in
from ..model import DEFAULT_COMPONENTS, CMFModel, grid_search, nested_cv
from ..project import (PROJECT_EXTENSION, PoseHistory, Project, compare_models, is_project_file,
                       load_project, save_project)
from ..protein import (binding_site, contact_distance, fetch_pdb, find_pocket,
                       read_structure, residue_anchors)
from .panels import (DatasetPanel, ModelPanel, ResultsPanel, ViewerControls, apply_field_table,
                     apply_settings, capture_settings)
from .viewer3d import Viewer3D, cofield_colors
from .workers import TaskWorker

ABOUT_HTML = f"""
<h2>CMF 3D-QSAR {__version__}</h2>
<p>Desktop implementation of the <b>Continuous Molecular Fields</b> (CMF) approach to 3D-QSAR.</p>
<p><b>Method.</b> Every molecular field of type <i>f</i> is approximated as a sum of atom-centred
Gaussians &rho;<sub>f,l</sub>(r) = w<sub>f,l</sub> exp(&minus;&alpha;<sub>f</sub>/2 |r &minus; r<sub>l</sub>|&sup2;)
(the steric field uses widths scaled by the van der Waals radius, &alpha;/R<sub>vdW</sub>&sup2;).
The kernel of two molecules is the overlap integral of their fields, summed over all atom pairs and over the
field types with mixing coefficients h<sub>f</sub>:
K(M<sub>i</sub>,M<sub>j</sub>) = &Sigma;<sub>f</sub> h<sub>f</sub> &Sigma;<sub>l,m</sub> w<sub>f,il</sub> w<sub>f,jm</sub>
(&pi;/&alpha;<sub>f</sub>)<sup>3/2</sup> exp(&minus;&alpha;<sub>f</sub>/4 |r<sub>il</sub> &minus; r<sub>jm</sub>|&sup2;).
Kernel ridge regression with a bias term (or SVR / a Gaussian-process view of KRR) yields the dual coefficients
a<sub>j</sub>; hyper-parameters &alpha; and &lambda; are selected by minimising RMSE in k-fold cross-validation.</p>
<p><b>Co-fields.</b> The field of regression coefficients (molecular co-field)
C<sub>f</sub>(r) = &Sigma;<sub>j</sub> a<sub>j</sub> X<sub>f,j</sub>(r) is a linear combination of the training-set fields.
The activity is the overlap integral y = &Sigma;<sub>f</sub> h<sub>f</sub> &int; C<sub>f</sub>(r) X<sub>f</sub>(r) dr + b.
Positive zones of a co-field mark positions where a larger field value increases the activity; negative zones where
it decreases it.  Continuous indicator fields (one indicator field per atom type) make the co-fields directly
interpretable as preferable / undesirable positions of atoms of each type.</p>
<p><b>Decomposition rules.</b> (1) co-fields decompose into training-set fields; (2) a prediction decomposes
into contributions a<sub>j</sub> K(X, X<sub>j</sub>) of the training molecules ("Contributions" tab);
(3) a prediction decomposes into atom contributions (spheres in the 3D view).</p>
<p><b>References</b><br>
Baskin I.I., Zhokhova N.I. J. Comput.-Aided Mol. Des. 27 (2013) 427-442.<br>
Sitnikov G.V., Zhokhova N.I., Ustynyuk Y.A., Varnek A., Baskin I.I. J. Comput.-Aided Mol. Des. 29 (2015) 233-247.<br>
Baskin I.I., Zhokhova N.I. Future Med. Chem. 11 (2019) 2701-2713.</p>
<p><b>Field parameterisation in this program.</b> electrostatic: Gasteiger / MMFF94 / file charges;
steric: Tripos vdW parameters; hydrophobic: Wildman-Crippen atomic logP contributions; H-bond acceptor / donor:
approximate Abraham-type increments (donor weights on polar hydrogens); CIF: Sybyl-like atom types assigned
by the program (or read from MOL2).</p>
"""


class AlignDialog(QDialog):
    def __init__(self, names: List[str], current: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Align data set")
        form = QFormLayout(self)
        self.cmb_ref = QComboBox()
        self.cmb_ref.addItems(names)
        self.cmb_ref.setCurrentIndex(max(0, current))
        form.addRow("Reference molecule:", self.cmb_ref)
        self.chk_mcs = QCheckBox("Superimpose on maximum common substructure")
        self.chk_mcs.setChecked(True)
        form.addRow(self.chk_mcs)
        self.chk_field = QCheckBox("Refine by maximising field similarity (CMF kernel)")
        form.addRow(self.chk_field)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


class VarianceAlignDialog(QDialog):
    """Options for aligning molecules by the posterior variance of their prediction."""

    def __init__(self, n_test: int, n_sel: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Align by prediction confidence")
        lay = QVBoxLayout(self)
        info = QLabel(
            "Each molecule is rotated and translated so that the posterior variance of its predicted "
            "activity is as small as possible, i.e. it is moved into the applicability domain of the "
            "model (Baskin & Zhokhova, Future Med. Chem. 2019). No activity value of the molecule is used."
            "\n\nThe training molecules define the frame and are never moved: applied to them the "
            "criterion is degenerate, because superimposing everything minimises every variance. "
            "After alignment the variance of a molecule no longer measures its applicability domain.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        info.setStyleSheet("color: palette(text)")
        lay.addWidget(info)
        form = QFormLayout()
        self.cmb_which = QComboBox()
        self.cmb_which.addItems([f"test set ({n_test} molecules)", f"selected molecules ({n_sel})"])
        form.addRow("Align:", self.cmb_which)
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(0.0, 50.0)
        self.spin_max.setSingleStep(0.5)
        self.spin_max.setValue(2.0)
        self.spin_max.setSuffix(" A")
        self.spin_max.setSpecialValueText("no limit")
        self.spin_max.setToolTip("Maximum displacement from the current pose; keeps a molecule near "
                                 "its maximum-common-substructure alignment (0 = no limit)")
        form.addRow("Max. displacement:", self.spin_max)
        self.spin_restarts = QSpinBox()
        self.spin_restarts.setRange(0, 20)
        self.spin_restarts.setValue(2)
        form.addRow("Random restarts:", self.spin_restarts)
        self.chk_flexible = QCheckBox("turn the rotatable bonds as well")
        self.chk_flexible.setToolTip(
            "<p>Search the conformation together with the pose, which is the 2019 proposal in "
            "full. The force field acts as a prior: the conformation may rise a little above "
            "the one it started in and is penalised beyond that, so the search cannot fold the "
            "molecule into whatever shape the training set finds familiar.</p>"
            "<p>The displacement limit above then applies to the rigid motion only &mdash; a "
            "turning bond is not a translation, and charging it as one would suppress the very "
            "thing this adds.</p>")
        self.chk_flexible.toggled.connect(self._flex_toggled)
        form.addRow("", self.chk_flexible)
        self.spin_conformers = QSpinBox()
        self.spin_conformers.setRange(0, 50)
        self.spin_conformers.setValue(5)
        self.spin_conformers.setSpecialValueText("the current one only")
        self.spin_conformers.setToolTip(
            "Starting conformations from the generator, besides the one the molecule is in. A "
            "torsion search is local, so what it can reach is decided by where it starts; on "
            "the FEP benchmark these are what remove the conformational part of the error.")
        form.addRow("Starting conformations:", self.spin_conformers)
        self.spin_slack = QDoubleSpinBox()
        self.spin_slack.setRange(0.0, 100.0)
        self.spin_slack.setSingleStep(1.0)
        self.spin_slack.setValue(10.0)
        self.spin_slack.setSuffix(" kcal/mol")
        self.spin_slack.setToolTip("How far the conformation may rise above the one it started "
                                   "in before the strain prior penalises it")
        form.addRow("Strain allowance:", self.spin_slack)
        lay.addLayout(form)
        self._flex_toggled(False)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def max_rmsd(self):
        v = self.spin_max.value()
        return None if v <= 0 else float(v)

    def _flex_toggled(self, on: bool) -> None:
        for w in (self.spin_conformers, self.spin_slack):
            w.setEnabled(bool(on))

    def flexible(self) -> bool:
        return bool(self.chk_flexible.isChecked())

    def conformers(self) -> int:
        return int(self.spin_conformers.value()) if self.flexible() else 0

    def energy_slack(self) -> float:
        return float(self.spin_slack.value())


class TrainAlignDialog(QDialog):
    """Options for the self-consistent (cross-validated) alignment of the training set."""

    def __init__(self, n_train: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Align training set by prediction confidence")
        lay = QVBoxLayout(self)
        info = QLabel(
            f"The {n_train} training molecules are aligned against each other: in every fold the "
            "held-out molecules are aligned to a model fitted to the remaining ones, so no molecule "
            "is ever aligned against a model that contains it. Repeating over the folds gives a "
            "self-consistent alignment."
            "\n\nMeasured on four data sets, the gain is NOT reliable: one set improved, one degraded, "
            "two were neutral, and rounds after the first consistently made the external prediction "
            "worse. Hence one round and a displacement limit by default."
            "\n\nq2 of the realigned set is not a valid criterion: it is computed on the geometry that "
            "was just changed, and in the tests q2 and the external R2pred repeatedly moved in "
            "opposite directions (thrombin: q2 0.674 to 0.706 while R2pred fell 0.428 to 0.192). "
            "Judge the result on a test set that was not realigned.")
        info.setWordWrap(True)
        info.setMaximumWidth(600)
        lay.addWidget(info)
        form = QFormLayout()
        self.spin_rounds = QSpinBox()
        self.spin_rounds.setRange(1, 20)
        self.spin_rounds.setValue(1)
        form.addRow("Rounds:", self.spin_rounds)
        self.spin_folds = QSpinBox()
        self.spin_folds.setRange(2, 50)
        self.spin_folds.setValue(5)
        form.addRow("Folds:", self.spin_folds)
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setRange(0.0, 50.0)
        self.spin_max.setSingleStep(0.5)
        self.spin_max.setValue(1.0)
        self.spin_max.setSuffix(" A")
        self.spin_max.setSpecialValueText("no limit")
        form.addRow("Max. displacement:", self.spin_max)
        self.spin_restarts = QSpinBox()
        self.spin_restarts.setRange(0, 10)
        self.spin_restarts.setValue(0)
        self.spin_restarts.setToolTip("0 keeps every molecule in the basin of its current pose")
        form.addRow("Random restarts:", self.spin_restarts)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def max_rmsd(self):
        v = self.spin_max.value()
        return None if v <= 0 else float(v)


class GTMDialog(QDialog):
    """Options for the chemical space map."""

    def __init__(self, n_mol: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Chemical space map")
        lay = QVBoxLayout(self)
        info = QLabel(
            "The molecules are described by their CMF kernels, whose leading functional principal "
            "components are mapped onto a two-dimensional grid by Generative Topographic Mapping. "
            "Each node then carries the responsibility-weighted mean of the activity (the activity "
            "landscape), the fraction of actives, or simply the amount of data supporting it."
            "\n\nThe kernel is meaningful only between molecules aligned in a common frame. For a map "
            "spanning several series, tick the small-attenuation option: as alpha tends to zero a "
            "continuous indicator field tends to the count of atoms of that type, so the kernel "
            "becomes an alignment-free fragment-descriptor kernel.")
        info.setWordWrap(True)
        info.setMaximumWidth(580)
        lay.addWidget(info)
        form = QFormLayout()
        self.cmb_which = QComboBox()
        self.cmb_which.addItems(["training molecules", f"all molecules ({n_mol})"])
        form.addRow("Map:", self.cmb_which)
        self.cmb_projection = QComboBox()
        self.cmb_projection.addItems(["GTM landscape", "kernel PCA score plot"])
        self.cmb_projection.setToolTip(
            "The GTM spreads the set over a square grid, which makes a landscape possible but\n"
            "distorts distances; the score plot is the linear projection the GTM is fitted on,\n"
            "where distances are the kernel's own and the axes carry their share of its variance.")
        form.addRow("Projection:", self.cmb_projection)
        self.cmb_colour = QComboBox()
        self.cmb_colour.addItems(["activity landscape", "class landscape", "data density"])
        form.addRow("Landscape:", self.cmb_colour)
        self.spin_grid = QSpinBox()
        self.spin_grid.setRange(5, 60)
        self.spin_grid.setValue(20)
        form.addRow("Map nodes (per side):", self.spin_grid)
        self.spin_rbf = QSpinBox()
        self.spin_rbf.setRange(2, 20)
        self.spin_rbf.setValue(5)
        self.spin_rbf.setToolTip("Radial basis functions per side: fewer means a smoother map")
        form.addRow("Basis functions:", self.spin_rbf)
        self.spin_comp = QSpinBox()
        self.spin_comp.setRange(2, 200)
        self.spin_comp.setValue(15)
        self.spin_comp.setToolTip("Functional principal components of the kernel kept before mapping")
        form.addRow("Components:", self.spin_comp)
        prow = QHBoxLayout()
        self.spin_pc_x = QSpinBox()
        self.spin_pc_x.setRange(1, 200)
        self.spin_pc_x.setValue(1)
        self.spin_pc_y = QSpinBox()
        self.spin_pc_y.setRange(1, 200)
        self.spin_pc_y.setValue(2)
        for w in (self.spin_pc_x, self.spin_pc_y):
            w.setToolTip("Which two components the score plot draws")
            w.setEnabled(False)
        prow.addWidget(self.spin_pc_x)
        prow.addWidget(self.spin_pc_y)
        prow.addStretch(1)
        form.addRow("Components plotted:", prow)

        def _projection_changed(i):
            score = i == 1
            for w in (self.spin_pc_x, self.spin_pc_y):
                w.setEnabled(score)
            for w in (self.spin_grid, self.spin_rbf):
                w.setEnabled(not score)       # the grid and its bases belong to the GTM alone

        self.cmb_projection.currentIndexChanged.connect(_projection_changed)
        row = QHBoxLayout()
        self.chk_small_alpha = QCheckBox("alignment-free (small attenuation)")
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.001, 5.0)
        self.spin_alpha.setDecimals(3)
        self.spin_alpha.setValue(0.02)
        self.spin_alpha.setEnabled(False)
        self.chk_small_alpha.toggled.connect(self.spin_alpha.setEnabled)
        row.addWidget(self.chk_small_alpha)
        row.addWidget(self.spin_alpha)
        form.addRow("Attenuation:", row)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


def _learn_conformer_weights(mols, names, train_idx, test_idx, y_all, fieldset, beta, selection,
                             lams, normalize, progress=None):
    """Fit multi-instance weights for the conformations and store them on the molecules.

    The weights are written to the molecules as a property, and the field set is switched to read
    them, so from here on everything behaves as an ordinary ensemble model: the same fit, the same
    co-fields, the same decomposition. Test molecules are weighted by the rule the model was
    fitted with, which uses their conformation scores and never their activity.
    """
    from ..mil import flatten_conformers, learn_weights, predict_bags, softmax_weights, argmax_weights
    single = fieldset.copy()
    single.conformers = "single"
    flat_tr, bags_tr = flatten_conformers([mols[i] for i in train_idx], single,
                                          [names[i] for i in train_idx],
                                          max_conformers=fieldset.max_conformers)
    probe = CMFModel(fieldset=single, method="krr", normalize=normalize)
    K_conf = probe.joint_kernel(probe.prepare_training_kernels(AtomTable.from_molecules(flat_tr)))
    y = np.asarray(y_all, float)[train_idx]
    lam = float(sorted(lams)[len(lams) // 2]) if lams else 1.0
    fit = learn_weights(K_conf, bags_tr, y, beta=float(beta), lam=lam, selection=selection,
                        init="first")
    _store_weights(mols, train_idx, bags_tr, fit.weights, fieldset)
    if test_idx:
        flat_te, bags_te = flatten_conformers([mols[i] for i in test_idx], single,
                                              [names[i] for i in test_idx],
                                              max_conformers=fieldset.max_conformers)
        K_te = probe.joint_kernel(probe.test_kernels(AtomTable.from_molecules(flat_te)))
        scores = (K_te @ fit.W().T) @ fit.coef + fit.bias
        w_te = (argmax_weights(scores, bags_te) if selection == "hard"
                else softmax_weights(scores, bags_te, float(beta)))
        _store_weights(mols, test_idx, bags_te, w_te, fieldset)
    fieldset.conformer_weighting = "given"
    return fit


def _store_weights(mols, idx, bags, weights, fieldset):
    prop = getattr(fieldset, "weight_prop", "conformer_weights")
    for b, i in enumerate(idx):
        w = np.asarray(weights, float)[bags == b]
        if len(w) == mols[i].GetNumConformers():
            mols[i].SetProp(prop, ",".join(f"{v:.6f}" for v in w))


class ChemotypeDialog(QDialog):
    """Options for grouping a heterogeneous series by scaffold."""

    def __init__(self, n_mol: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Find the chemotypes")
        lay = QVBoxLayout(self)
        info = QLabel(
            f"The {n_mol} molecules are grouped by their Bemis-Murcko framework: strip the side "
            "chains, keep the ring systems and what links them, and group what is left.\n\n"
            "Everything else in this submenu works from that grouping: aligning inside a group, "
            "placing the groups in one frame, and holding a whole group out of the validation.")
        info.setWordWrap(True)
        info.setMaximumWidth(540)
        lay.addWidget(info)
        form = QFormLayout()
        self.chk_generic = QCheckBox("ignore elements and bond orders")
        self.chk_generic.setToolTip("The generic framework merges scaffolds that differ only in "
                                    "their heteroatoms, which usually gives fewer, larger groups")
        form.addRow("Framework:", self.chk_generic)
        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 50)
        self.spin_min.setValue(3)
        self.spin_min.setToolTip("Scaffolds with fewer molecules than this are pooled into one "
                                 "group, so that singletons do not turn the validation into "
                                 "leave-one-molecule-out")
        form.addRow("Smallest group:", self.spin_min)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


class PlaceChemotypesDialog(QDialog):
    """Options for moving each chemotype, as a rigid body, into a common frame."""

    def __init__(self, names, has_protein: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Place the chemotypes")
        lay = QVBoxLayout(self)
        info = QLabel(
            "Each chemotype is moved as a whole, so the alignment made inside it survives. Only "
            "the relation between chemotypes changes.\n\n"
            "By field similarity, one chemotype stays still and the others are superimposed on it "
            "by maximising the CMF kernel between their reference molecules, which needs no common "
            "substructure. By docking, each chemotype's reference is placed in the binding site "
            "instead, which needs a structure but no reference chemotype.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        lay.addWidget(info)
        form = QFormLayout()
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["field similarity to a reference chemotype",
                                "docking into the binding site"])
        if not has_protein:
            self.cmb_mode.model().item(1).setEnabled(False)
            self.cmb_mode.setToolTip("Load a protein structure to enable docking")
        form.addRow("Place by:", self.cmb_mode)
        self.cmb_ref = QComboBox()
        for g, nm in enumerate(names):
            self.cmb_ref.addItem(f"{g}: {nm[:56]}")
        form.addRow("Reference chemotype:", self.cmb_ref)
        self.spin_restarts = QSpinBox()
        self.spin_restarts.setRange(0, 20)
        self.spin_restarts.setValue(2)
        form.addRow("Restarts:", self.spin_restarts)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def mode(self) -> str:
        return ["field", "dock"][self.cmb_mode.currentIndex()]


class ChemotypeValidationDialog(QDialog):
    """Options for leave-one-chemotype-out validation and the alignment-free baseline."""

    def __init__(self, n_groups: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Leave one chemotype out")
        lay = QVBoxLayout(self)
        info = QLabel(
            f"Each of the {n_groups} chemotypes is held out in turn, so a molecule is never "
            "predicted from its own scaffold-mates. Random folds overstate the quality of a mixed series badly, "
            "and both numbers are reported side by side.\n\n"
            "The comparison that matters is the last one: a continuous indicator field tends to "
            "a count of atoms of its type as its attenuation vanishes, and the baseline is that "
            "limit, so it ignores the poses entirely. If the aligned model cannot beat it across "
            "chemotypes, the alignment is not paying for itself.\n\n"
            "The attenuation and regularisation grids come from the Model panel.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        lay.addWidget(info)
        form = QFormLayout()
        self.spin_folds = QSpinBox()
        self.spin_folds.setRange(0, 50)
        self.spin_folds.setValue(0)
        self.spin_folds.setSpecialValueText("one per chemotype")
        self.spin_folds.setToolTip("Bundle the chemotypes into this many folds instead of holding "
                                   "out one at a time, which is worth doing when there are many "
                                   "small groups")
        form.addRow("Folds:", self.spin_folds)
        self.spin_alpha = QDoubleSpinBox()
        self.spin_alpha.setRange(0.0, 1.0)
        self.spin_alpha.setDecimals(5)
        self.spin_alpha.setSingleStep(1e-4)
        self.spin_alpha.setValue(0.0)
        self.spin_alpha.setSpecialValueText("the limit (atom counts)")
        self.spin_alpha.setToolTip("Attenuation of the alignment-free indicator fields. At zero "
                                   "the baseline is the limit they tend to, taken from the atom "
                                   "counts directly: exactly pose-free, and much faster. With an "
                                   "attenuation it is the Gaussian kernel at that value, which "
                                   "has to be far smaller than a modelling one, since the "
                                   "dependence on the poses dies only linearly in it: at 0.02 a "
                                   "rigid motion of every molecule still moves the kernel by "
                                   "0.7, at 0.0002 by 0.013.")
        form.addRow("Baseline α:", self.spin_alpha)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


class TopomerDialog(QDialog):
    """Options for replacing the molecules by their substituents, posed by rule."""

    def __init__(self, n_mol: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Topomer substituents")
        lay = QVBoxLayout(self)
        info = QLabel(
            f"Each of the {n_mol} molecules is cut at the bonds holding its substituents to a "
            f"common core, and every substituent is built into a pose that depends on its "
            f"topology alone: canonical numbering, one embedding, every rotatable torsion at the "
            f"angle below, the attachment atom at the origin and its open valence along one "
            f"axis. The same substituent then always lands in the same place, so the fields can "
            f"be compared with no alignment at all (Cramer's topomer CoMFA, 1997-2003).\n\n"
            f"The molecules on screen are replaced by those substituents, each attachment point "
            f"in its own region of space, because that is the frame a topomer model lives in: "
            f"its co-fields and contributions belong there and not around the original ligands. "
            f"Names, activities and the train/test split are kept; reload the file to get the "
            f"molecules back.")
        info.setWordWrap(True)
        info.setMaximumWidth(580)
        lay.addWidget(info)
        form = QFormLayout()
        self.cmb_core = QComboBox()
        self.cmb_core.addItems(["framework of the largest chemotype",
                                "maximum common substructure of the series", "this SMILES",
                                "this SMARTS"])
        self.cmb_core.setToolTip(
            "<p>Which scaffold the substituents are cut from.</p>"
            "<p>The framework of the largest chemotype is the scaffold of one series, and "
            "is the default: the maximum common substructure of a set that mixes chemotypes "
            "is a fragment fused into every ring system, and on the GPB set it is CCO, which "
            "cuts nothing off any molecule.</p>"
            "<p>Use the common substructure for a set you know is congeneric, and type the "
            "core yourself when you know it.</p>")
        form.addRow("Core:", self.cmb_core)
        self.ed_core = QLineEdit("")
        self.ed_core.setPlaceholderText("c1ccccc1C(=O)N")
        self.ed_core.setEnabled(False)
        self.cmb_core.currentIndexChanged.connect(
            lambda i: self.ed_core.setEnabled(i > 1))
        form.addRow("", self.ed_core)
        self.spin_torsion = QDoubleSpinBox()
        self.spin_torsion.setRange(-180.0, 180.0)
        self.spin_torsion.setSingleStep(10.0)
        self.spin_torsion.setDecimals(1)
        self.spin_torsion.setValue(180.0)
        self.spin_torsion.setSuffix(" °")
        self.spin_torsion.setToolTip("Every rotatable torsion of every substituent is set to "
                                     "this; 180° is the anti convention of the original method")
        form.addRow("Torsions:", self.spin_torsion)
        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setRange(10.0, 500.0)
        self.spin_spacing.setSingleStep(10.0)
        self.spin_spacing.setDecimals(0)
        self.spin_spacing.setValue(100.0)
        self.spin_spacing.setSuffix(" Å")
        self.spin_spacing.setToolTip(
            "How far apart the attachment sites are placed. Far enough and the kernel is exactly "
            "the sum over the positions, which is the point of the construction; closer makes "
            "the series easier to look at, but lets the sites see one another when the "
            "attenuation is very small: at 30 Å two of them still overlap at alpha = 0.01.")
        form.addRow("Sites apart:", self.spin_spacing)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def core(self):
        """(mol, description) of the core to cut from, or (None, 'mcs')."""
        from rdkit import Chem
        text = self.ed_core.text().strip()
        mode = self.cmb_core.currentIndex()
        if mode == 2 and text:
            return Chem.MolFromSmiles(text), text
        if mode == 3 and text:
            return Chem.MolFromSmarts(text), text
        return None, ("the framework of the largest chemotype" if mode == 0
                      else "the maximum common substructure")


class FieldDifferenceDialog(QDialog):
    """Choose the molecule a candidate is meant to imitate."""

    def __init__(self, names, current: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Field difference from a reference")
        lay = QVBoxLayout(self)
        info = QLabel(
            "The difference between the fields of the molecule selected in the data-set table "
            "and those of a reference: negative where the selected molecule fails to reproduce "
            "what the reference presents, positive where it adds something of its own.\n\n"
            "This is the readout for scaffold hopping and peptidomimetics, where the question "
            "is not how a candidate scores but what is wrong with it. Both molecules must be in "
            "the same frame; the map of two molecules that were never aligned measures the "
            "alignment.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        lay.addWidget(info)
        form = QFormLayout()
        self.cmb_ref = QComboBox()
        for i, n in enumerate(names):
            self.cmb_ref.addItem(f"{i + 1}. {n}")
        if 0 <= current < len(names):
            self.cmb_ref.setCurrentIndex(current)
        form.addRow("Reference molecule:", self.cmb_ref)
        self.spin_level = QDoubleSpinBox()
        self.spin_level.setRange(0.1, 0.95)
        self.spin_level.setSingleStep(0.05)
        self.spin_level.setDecimals(2)
        self.spin_level.setValue(0.5)
        self.spin_level.setToolTip("Contour bounding a lobe of the difference map, as a fraction "
                                   "of its largest value, for the list of where they differ")
        form.addRow("Contour level:", self.spin_level)
        self.chk_draw = QCheckBox("show the map in the 3D view")
        self.chk_draw.setChecked(True)
        form.addRow("", self.chk_draw)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


class PharmacophoreDialog(QDialog):
    """Options for discretising the co-fields into typed features."""

    def __init__(self, n_fields: int, has_protein: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extract a pharmacophore")
        lay = QVBoxLayout(self)
        info = QLabel(
            "Every lobe of a co-field becomes one feature: its centre is the field-weighted "
            "centroid of the lobe, its tolerance radius the radius of a sphere of the same "
            "volume, and its weight the integral of the co-field over it.\n\n"
            "The sign says what the feature asks for. A positive electrostatic lobe wants a "
            "positive partial charge and a negative one wants a negative charge; a negative "
            "lobe of any other field is a warning, and a negative steric lobe is the model's "
            "own exclusion volume.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        lay.addWidget(info)
        form = QFormLayout()
        self.spin_level = QDoubleSpinBox()
        self.spin_level.setRange(0.1, 0.95)
        self.spin_level.setSingleStep(0.05)
        self.spin_level.setDecimals(2)
        self.spin_level.setValue(0.5)
        self.spin_level.setToolTip("A feature is a region where the co-field exceeds this "
                                   "fraction of its largest value: higher gives tighter and "
                                   "fewer features")
        form.addRow("Contour level:", self.spin_level)
        self.spin_max = QSpinBox()
        self.spin_max.setRange(1, 60)
        self.spin_max.setValue(10)
        form.addRow("Features to keep:", self.spin_max)
        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setRange(0.2, 2.0)
        self.spin_spacing.setSingleStep(0.1)
        self.spin_spacing.setValue(0.6)
        self.spin_spacing.setSuffix(" A")
        form.addRow("Grid spacing:", self.spin_spacing)
        self.chk_pocket = QCheckBox("add exclusion volumes where the receptor is")
        self.chk_pocket.setEnabled(has_protein)
        self.chk_pocket.setChecked(has_protein)
        if not has_protein:
            self.chk_pocket.setToolTip("Load a protein structure to enable this")
        form.addRow("Pocket:", self.chk_pocket)
        self.chk_draw = QCheckBox("show the features in the 3D view")
        self.chk_draw.setChecked(True)
        form.addRow("", self.chk_draw)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


class DecoyAlignDialog(QDialog):
    """Options for giving every background molecule its best pose against the training set."""

    def __init__(self, n_decoys: int, n_train: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Align decoys")
        lay = QVBoxLayout(self)
        info = QLabel(
            f"The {n_decoys} background molecules are placed in the frame of the {n_train} "
            f"training molecules by maximising their field similarity to them, so that no decoy "
            f"is rejected merely for being posed badly and whatever enrichment survives is not "
            f"an artefact of sloppy placement.\n\n"
            f"The alignment is rigid: it turns and moves a conformation but does not change it. "
            f"Generating conformations lets a decoy be judged on the best one it can adopt "
            f"rather than the one its file happened to carry; the pose from the file is always "
            f"among the candidates, and the cost is multiplied by their number.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        lay.addWidget(info)
        form = QFormLayout()
        self.spin_restarts = QSpinBox()
        self.spin_restarts.setRange(0, 50)
        self.spin_restarts.setValue(6)
        self.spin_restarts.setToolTip("A molecule of another chemotype has no reason to start in "
                                      "the basin of its best pose, so the search is repeated "
                                      "from this many random orientations")
        form.addRow("Random starting orientations:", self.spin_restarts)
        self.spin_conformers = QSpinBox()
        self.spin_conformers.setRange(1, 50)
        self.spin_conformers.setValue(1)
        self.spin_conformers.setSpecialValueText("keep the pose in the file")
        form.addRow("Conformations per decoy:", self.spin_conformers)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


class DockDialog(QDialog):
    """Options for placing molecules, or their conformations, inside the pocket."""

    def __init__(self, n_mol: int, n_conf: int, pocket_name: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Dock into the pocket")
        lay = QVBoxLayout(self)
        info = QLabel(
            f"Every conformation of every molecule is placed in {pocket_name} by optimising its "
            "six rigid-body parameters against a score made of field complementarity, burial, a "
            "clash term and a tether. The internal geometry is not touched, so only the placement "
            "changes.\n\n"
            f"{n_mol} molecules, {n_conf} conformations in all. It takes a few seconds per "
            "conformation.\n\n"
            "This moves the molecules, so a model built on their present poses is discarded. "
            "Measured against the alternatives, docking is a worse alignment than superposition "
            "on a common substructure, so treat the result as a hypothesis to inspect rather "
            "than as an improvement.")
        info.setWordWrap(True)
        info.setMaximumWidth(580)
        lay.addWidget(info)
        form = QFormLayout()
        self.spin_restarts = QSpinBox()
        self.spin_restarts.setRange(0, 20)
        self.spin_restarts.setValue(1)
        self.spin_restarts.setToolTip("Bounded random restarts per conformation; 0 optimises from "
                                      "where it stands")
        form.addRow("Restarts:", self.spin_restarts)
        self.spin_tether = QDoubleSpinBox()
        self.spin_tether.setRange(0.0, 30.0)
        self.spin_tether.setSingleStep(0.5)
        self.spin_tether.setValue(3.0)
        self.spin_tether.setSuffix(" Å")
        self.spin_tether.setToolTip("How far the centroid may travel before it is penalised; "
                                    "widen it when the molecules do not start in the site")
        form.addRow("Tether radius:", self.spin_tether)
        self.spin_keep = QSpinBox()
        self.spin_keep.setRange(0, 999)
        self.spin_keep.setValue(0)
        self.spin_keep.setSpecialValueText("keep all")
        self.spin_keep.setToolTip("Keep only this many best-scoring poses of each molecule")
        form.addRow("Keep per molecule:", self.spin_keep)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)


class ConformerDialog(QDialog):
    """Options for generating a conformational ensemble: the fourth dimension."""

    def __init__(self, n_mol: int, n_have: int, parent=None, names=None):
        super().__init__(parent)
        self.setWindowTitle("Generate conformations")
        lay = QVBoxLayout(self)
        info = QLabel(
            "Each molecule is given a set of conformations, which the ensemble mode of the model "
            "averages over: a molecule is then the weighted mean of the fields of its "
            "conformations, which is the fourth dimension of 4D-QSAR.\n\n"
            "The pose a molecule has now is kept as its first conformation, so an alignment made "
            "earlier survives. Where the other conformations go is the choice below: the kernel "
            "compares molecules with each other, so a conformation superimposed on its own "
            "molecule alone is still in an arbitrary orientation with respect to the rest of the "
            "series. The kernels cost about the square of the number of conformations, so start "
            "small.\n\n"
            f"{n_have} of {n_mol} molecules already carry more than one conformation.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        lay.addWidget(info)
        form = QFormLayout()
        self.spin_n = QSpinBox()
        self.spin_n.setRange(2, 200)
        self.spin_n.setValue(10)
        form.addRow("Conformations per molecule:", self.spin_n)
        self.spin_prune = QDoubleSpinBox()
        self.spin_prune.setRange(0.0, 5.0)
        self.spin_prune.setSingleStep(0.1)
        self.spin_prune.setValue(0.5)
        self.spin_prune.setSuffix(" Å")
        self.spin_prune.setToolTip("Discard a conformation within this heavy-atom RMSD of a "
                                   "better one; 0 keeps them all")
        form.addRow("Prune below:", self.spin_prune)
        self.spin_window = QDoubleSpinBox()
        self.spin_window.setRange(0.0, 20.0)
        self.spin_window.setSingleStep(0.25)
        self.spin_window.setValue(0.0)
        self.spin_window.setSuffix(" Å")
        self.spin_window.setSpecialValueText("no window")
        self.spin_window.setToolTip("\n".join([
            "Keep only the conformations that stand within this heavy-atom RMSD of the pose the",
            "molecule arrived in, measured where they are rather than after a best fit.",
            "The ensemble then covers the flexibility around an aligned pose instead of the",
            "whole conformational surface, which is what a series aligned by its author needs."]))
        form.addRow("Keep within:", self.spin_window)
        self.cmb_align = QComboBox()
        for label in ("on the molecule's own pose (all heavy atoms)",
                      "on the molecule's own pose (common scaffold only)",
                      "on a reference molecule of the series (common substructure)",
                      "on a reference molecule, refined on the field similarity"):
            self.cmb_align.addItem(label)
        self.cmb_align.setToolTip(
            "The first keeps every conformation close to its own molecule but leaves it in an "
            "arbitrary orientation with respect to the others. The second holds the scaffold "
            "exactly where the alignment of the series put it and lets only the flexible part "
            "move. The last two put every conformation in the frame of one reference molecule, "
            "the field variant by maximising the CMF kernel with it.")
        form.addRow("Place the conformations:", self.cmb_align)
        self.cmb_ref = QComboBox()
        self.cmb_ref.addItem("largest molecule", -1)
        for i, nm in enumerate(names or []):
            self.cmb_ref.addItem(f"{i + 1}. {nm}", i)
        self.cmb_ref.setToolTip("The molecule whose pose defines the frame of the series; its "
                                "common substructure with the others is what the conformations "
                                "are superimposed on")
        form.addRow("Reference molecule:", self.cmb_ref)
        self.chk_optimise = QCheckBox("optimise with MMFF94s and store the energies")
        self.chk_optimise.setChecked(True)
        self.chk_optimise.setToolTip("The energies are what Boltzmann weighting needs")
        form.addRow("", self.chk_optimise)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(0)
        form.addRow("Seed:", self.spin_seed)
        self.chk_switch = QCheckBox("switch the model to the ensemble mode afterwards")
        self.chk_switch.setChecked(True)
        form.addRow("", self.chk_switch)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def alignment(self) -> str:
        """The conformer placement chosen, as :data:`cmfqsar.conformers.CONFORMER_ALIGNMENTS`."""
        return CONFORMER_ALIGNMENTS[self.cmb_align.currentIndex()]

    def reference_index(self):
        i = self.cmb_ref.currentData()
        return None if i is None or int(i) < 0 else int(i)


class UnalignDialog(QDialog):
    """Options for destroying the alignment: give every molecule a random pose."""

    def __init__(self, n_all: int, n_train: int, n_test: int, n_sel: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unalign (randomise poses)")
        lay = QVBoxLayout(self)
        info = QLabel(
            "Every selected molecule is rotated about its own centroid and translated at random. "
            "The internal geometry is untouched, only the mutual arrangement is destroyed."
            "\n\nThis is the control experiment for any alignment method: scramble the poses, realign, "
            "and measure how much of the original arrangement comes back. The poses before "
            "randomisation are remembered, so a later alignment reports its recovery against them."
            "\n\nMoving training molecules invalidates the current model, which refers to their old poses.")
        info.setWordWrap(True)
        info.setMaximumWidth(560)
        lay.addWidget(info)
        form = QFormLayout()
        self.cmb_which = QComboBox()
        self.cmb_which.addItems([f"test set ({n_test})", f"selected ({n_sel})",
                                 f"training set ({n_train})", f"all molecules ({n_all})"])
        form.addRow("Randomise:", self.cmb_which)
        self.cmb_rot = QComboBox()
        self.cmb_rot.addItems(["random orientation (alignment fully destroyed)", "bounded rotation"])
        form.addRow("Rotation:", self.cmb_rot)
        self.spin_angle = QDoubleSpinBox()
        self.spin_angle.setRange(0.0, 180.0)
        self.spin_angle.setValue(40.0)
        self.spin_angle.setSuffix(" deg")
        self.spin_angle.setEnabled(False)
        self.cmb_rot.currentIndexChanged.connect(lambda i: self.spin_angle.setEnabled(i == 1))
        form.addRow("Max. angle:", self.spin_angle)
        self.spin_shift = QDoubleSpinBox()
        self.spin_shift.setRange(0.0, 50.0)
        self.spin_shift.setSingleStep(0.5)
        self.spin_shift.setValue(2.0)
        self.spin_shift.setSuffix(" A")
        form.addRow("Max. displacement:", self.spin_shift)
        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        form.addRow("Random seed:", self.spin_seed)
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def max_rotation(self):
        return None if self.cmb_rot.currentIndex() == 0 else float(self.spin_angle.value())


def _run_validation(model, mf, tr, te, y_all, names, task_name, n_scramble, n_boot, want_domain,
                    folds, seed, progress, cancel, scramble_select=False, alphas=None,
                    lams=None):
    """y-randomisation, bootstrap intervals and the applicability domain, if they were asked for.

    Runs inside the build worker, because all three are loops over fits and the window must stay
    responsive. Returns the lines to print under the statistics and the per-molecule domain
    result, or None when nothing was requested.
    """
    if not (n_scramble or n_boot or want_domain):
        return None
    from ..validation import bootstrap_ci, domain, scramble_kernel
    lines, outside = [], []
    regression = task_name == "regression"
    y = np.asarray(y_all, float)
    if n_scramble and regression:
        progress("y-randomisation", 0.95)
        K = model.joint_kernel(model.K_fields_train) if model.uses_fields else \
            model.training_kernel(model.train)
        if scramble_select and alphas:
            from ..validation import scramble_selection
            # walking the alpha grid moves the field set off the alpha the search chose, so put
            # it back before anything downstream uses the model again
            chosen = [s.alpha for s in model.fieldset.specs]
            grids = []
            for a in alphas:
                model.fieldset.set_alpha(a)
                grids.append(model.training_kernel(model.train))
            if model.method == "kpls":
                params = [{"components": c} for c in DEFAULT_COMPONENTS]
            else:
                params = [{"lam": float(l)} for l in (lams or [])] or [dict(model.params)]
            res = scramble_selection(grids, y[tr], method=model.method, param_grid=params,
                                     folds=folds, cv_seed=seed, runs=int(n_scramble), seed=seed,
                                     progress=lambda m_, f_: progress(m_, 0.95 + 0.02 * f_),
                                     cancel=cancel)
            for spec, a in zip(model.fieldset.specs, chosen):
                spec.alpha = float(a)
            model.training_kernel(model.train)
        else:
            res = scramble_kernel(K, y[tr], method=model.method, params=model.params, folds=folds,
                                  cv_seed=seed, runs=int(n_scramble), seed=seed,
                                  progress=lambda m_, f_: progress(m_, 0.95 + 0.02 * f_),
                                  cancel=cancel)
        lines.append(res.describe())
        note = res.caution()
        if note:
            lines.append("  " + note)
    elif n_scramble:
        lines.append("y-randomisation is implemented for regression models only; skipped.")
    if n_boot and regression:
        progress("Bootstrap", 0.97)
        st = model.stats
        if "y_cv" in st.extra:
            lines.append(bootstrap_ci(y[tr], np.asarray(st.extra["y_cv"], float),
                                      statistic="q2 (cross-validation)", runs=int(n_boot),
                                      seed=seed).describe())
        if te and "y_test_pred" in st.extra:
            lines.append(bootstrap_ci(y[te], np.asarray(st.extra["y_test_pred"], float),
                                      statistic="R2pred (test set, deviations from the training "
                                                "mean)", runs=int(n_boot), seed=seed,
                                      y_ref_mean=float(np.mean(y[tr]))).describe())
    elif n_boot:
        lines.append("Bootstrap intervals are implemented for regression models only; skipped.")
    dom_rows = None
    if want_domain and model.uses_fields:
        progress("Applicability domain", 0.99)
        scored = [i for i in range(len(mf)) if i not in set(tr)] or list(tr)
        table = AtomTable.from_molecules([mf[i] for i in scored])
        dom = domain(model.test_kernel(table), model.joint_self(table),
                     model.training_kernel(model.train))
        lines.append(dom.describe())
        outside = [(names[i], float(s)) for i, s in zip(scored, dom.mean_top)
                   if s < dom.threshold]
        dom_rows = {"index": scored, "similarity": [float(v) for v in dom.mean_top],
                    "threshold": float(dom.threshold),
                    "inside": [bool(v) for v in dom.inside]}
    elif want_domain:
        lines.append("The applicability domain needs the field kernel; it is not defined for the "
                     "autocorrelation kernel source.")
    return {"lines": lines, "outside": outside, "domain": dom_rows}


class OrcaDialog(QDialog):
    """Options for a quantum-chemical run: what ORCA computes, and for which molecules.

    The two kinds of result go to the two places the model keeps them: a per-atom quantity (a
    charge, a Fukui index) becomes the weights of a field, and a per-molecule quantity (the gap,
    the hardness) becomes a descriptor beside the fields.
    """

    LEVELS = [
        ("r2SCAN-3c TightSCF", "fast composite method, its own basis; the one to start with"),
        ("B3LYP D3BJ def2-SVP RIJCOSX def2/J TightSCF", "the usual hybrid, small basis"),
        ("B3LYP D3BJ def2-TZVP RIJCOSX def2/J TightSCF", "the same with a triple-zeta basis: slower"),
        ("PBE0 D3BJ def2-SVP RIJCOSX def2/J TightSCF", "a different hybrid"),
        ("wB97X-D3 def2-SVP RIJCOSX def2/J TightSCF", "range-separated, better for charge transfer"),
    ]

    def __init__(self, exe, groups, charges, mults, parent=None, workdir="orca_work",
                 has_mpi=False):
        super().__init__(parent)
        self.setWindowTitle("Quantum-chemical properties (ORCA)")
        self._groups = list(groups)
        self._charges, self._mults = list(charges), list(mults)
        lay = QVBoxLayout(self)
        info = QLabel(
            "ORCA computes two kinds of quantity, and they enter the model in two different "
            "places. <b>Per-atom</b> values &mdash; Mulliken, L&ouml;wdin, Hirshfeld and CHELPG "
            "charges, HOMO and LUMO populations, Fukui indices, the dual descriptor, local "
            "softness &mdash; appear as tick boxes in the <i>Per-atom property fields</i> row: "
            "each becomes a field whose atomic weights are those values. <b>Per-molecule</b> "
            "values &mdash; total energy, HOMO and LUMO energies, the gap, the dipole, the "
            "ionisation potential, electron affinity, hardness, softness, electrophilicity "
            "&mdash; appear in the <i>Descriptors</i> row: one number per molecule, standardised "
            "and added to the kernel beside the fields."
            "<p>The geometry each molecule has now is what ORCA is given, so an alignment "
            "survives. Every job is cached in the working directory: a run stopped half way "
            "continues where it left off rather than starting again.</p>")
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        info.setMaximumWidth(620)
        lay.addWidget(info)
        form = QFormLayout()

        row = QHBoxLayout()
        self.ed_exe = QLineEdit(exe or "")
        self.ed_exe.setPlaceholderText("path to orca.exe")
        self.ed_exe.textChanged.connect(self._refresh)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_exe)
        row.addWidget(self.ed_exe)
        row.addWidget(btn)
        w = QWidget()
        w.setLayout(row)
        form.addRow("ORCA program:", w)

        self.cmb_level = QComboBox()
        self.cmb_level.setEditable(True)
        for level, why in self.LEVELS:
            self.cmb_level.addItem(level)
            self.cmb_level.setItemData(self.cmb_level.count() - 1, why, Qt.ToolTipRole)
        self.cmb_level.setToolTip("The ORCA keyword line, which may be edited: anything ORCA "
                                  "accepts as a single-point method will do")
        form.addRow("Level of theory:", self.cmb_level)

        self.cmb_which = QComboBox()
        for label, idx in self._groups:
            self.cmb_which.addItem(f"{label} ({len(idx)})")
        self.cmb_which.currentIndexChanged.connect(self._refresh)
        form.addRow("Molecules:", self.cmb_which)

        self.chk_fukui = QCheckBox("Fukui indices and the global reactivity numbers")
        self.chk_fukui.setChecked(True)
        self.chk_fukui.setToolTip(
            "<p>The N+1 and N&minus;1 states as well, so three calculations per molecule instead "
            "of one. They are what the Fukui indices, the dual descriptor, the local softness, "
            "the ionisation potential, the electron affinity, the hardness and the "
            "electrophilicity are made of &mdash; the quantities a reactivity model wants.</p>")
        self.chk_fukui.toggled.connect(self._refresh)
        form.addRow("", self.chk_fukui)

        self.chk_opt = QCheckBox("optimise the geometry first")
        self.chk_opt.setChecked(False)
        self.chk_opt.toggled.connect(self._refresh)
        form.addRow("", self.chk_opt)

        self.spin_jobs = QSpinBox()
        self.spin_jobs.setRange(1, 64)
        self.spin_jobs.setValue(min(4, max(1, (os.cpu_count() or 4) // 2)))
        self.spin_jobs.valueChanged.connect(self._refresh)
        self.spin_jobs.setToolTip(
            "Molecules computed at the same time. This build of ORCA is "
            + ("parallel, so a job may use several cores as well (below)."
               if has_mpi else
               "the serial one: it has no MPI, so one job uses one core and the way to use the "
               "machine is to run several molecules at once."))
        form.addRow("Molecules at a time:", self.spin_jobs)

        self.spin_nprocs = QSpinBox()
        self.spin_nprocs.setRange(1, 64)
        self.spin_nprocs.setValue(1)
        self.spin_nprocs.setEnabled(has_mpi)
        self.spin_nprocs.setToolTip("Cores per job (%pal). Needs an ORCA built with MPI; the "
                                    "serial build fails outright when asked for more than one.")
        form.addRow("Cores per job:", self.spin_nprocs)

        self.spin_mem = QSpinBox()
        self.spin_mem.setRange(200, 64000)
        self.spin_mem.setSingleStep(500)
        self.spin_mem.setValue(2000)
        self.spin_mem.setSuffix(" MB")
        self.spin_mem.setToolTip("%maxcore: memory per core. Times the number of molecules at a "
                                 "time is what the run may ask of the machine.")
        form.addRow("Memory per core:", self.spin_mem)

        self.spin_timeout = QSpinBox()
        self.spin_timeout.setRange(0, 600)
        self.spin_timeout.setValue(30)
        self.spin_timeout.setSuffix(" min")
        self.spin_timeout.setSpecialValueText("no limit")
        self.spin_timeout.setToolTip("A job that runs longer than this is killed and its molecule "
                                     "reported as failed, so one pathological case cannot hold up "
                                     "the set")
        form.addRow("Give up on a job after:", self.spin_timeout)

        row = QHBoxLayout()
        self.ed_dir = QLineEdit(workdir)
        self.ed_dir.textChanged.connect(self._refresh)
        btn = QPushButton("Browse...")
        btn.clicked.connect(self._browse_dir)
        row.addWidget(self.ed_dir)
        row.addWidget(btn)
        w = QWidget()
        w.setLayout(row)
        w.setToolTip("Where the inputs, the outputs and the cache live. Keep it: it is both the "
                     "record of what was computed and what makes a second run cheap.")
        form.addRow("Working directory:", w)
        lay.addLayout(form)

        self.lbl_cost = QLabel()
        self.lbl_cost.setWordWrap(True)
        self.lbl_cost.setTextFormat(Qt.RichText)
        self.lbl_cost.setMaximumWidth(620)
        lay.addWidget(self.lbl_cost)
        self.lbl_warn = QLabel()
        self.lbl_warn.setWordWrap(True)
        self.lbl_warn.setTextFormat(Qt.RichText)
        self.lbl_warn.setMaximumWidth(620)
        self.lbl_warn.setStyleSheet("color: #a33;")
        lay.addWidget(self.lbl_warn)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("Run")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)
        self._refresh()

    # ---------------------------------------------------------------- the choices
    def indices(self) -> List[int]:
        return list(self._groups[max(0, self.cmb_which.currentIndex())][1])

    def level(self) -> str:
        return self.cmb_level.currentText().strip()

    def workdir(self) -> str:
        return self.ed_dir.text().strip() or "orca_work"

    def timeout(self):
        v = self.spin_timeout.value()
        return None if v <= 0 else float(v) * 60.0

    def _browse_exe(self):
        path, _ = QFileDialog.getOpenFileName(self, "ORCA program", self.ed_exe.text(),
                                              "Programs (*.exe);;All files (*)")
        if path:
            self.ed_exe.setText(path)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Working directory", self.ed_dir.text())
        if path:
            self.ed_dir.setText(path)

    # ------------------------------------------------- what the run will cost, and what it will move
    def _refresh(self, *_):
        from ..orca import estimate_seconds, measured_seconds_per_job
        idx = self.indices()
        fukui = self.chk_fukui.isChecked()
        per_job = measured_seconds_per_job(self.workdir(), fukui)
        source = "measured in this working directory" if per_job else "assumed for this machine"
        seconds = estimate_seconds(len(idx), fukui, self.spin_jobs.value(), per_job or 150.0)
        n_jobs = (3 if fukui else 1)
        if self.chk_opt.isChecked():
            # an optimisation is a series of single points; three is a modest guess at how many
            seconds += 3.0 * estimate_seconds(len(idx), False, self.spin_jobs.value(),
                                              per_job or 150.0)
            n_jobs += 1
        charges = [self._charges[i] for i in idx] or [0]
        mults = [self._mults[i] for i in idx] or [1]
        open_shell = sum(1 for m in mults if m != 1)
        chem = (f"all {len(idx)} neutral" if set(charges) == {0}
                else "formal charges from %s to %s" % (f"{min(charges):+d}".replace("+0", "0"),
                                                       f"{max(charges):+d}".replace("+0", "0")))
        shells = ("every one a closed-shell singlet" if not open_shell
                  else f"{open_shell} with an odd electron count, run as doublets")
        self.lbl_cost.setText(
            f"{len(idx)} molecules &times; {n_jobs} ORCA job(s) each: about "
            f"<b>{_hms(seconds)}</b> at {self.spin_jobs.value()} molecule(s) at a time "
            f"({_hms(per_job or 150.0)} a job, {source}). Molecules whose jobs are already in the "
            f"working directory are reused and cost nothing.<br>"
            f"Charge and multiplicity are read off the structure: {chem}, {shells}.")
        warn = []
        if self.chk_opt.isChecked():
            warn.append("<b>Optimising the geometry moves every atom.</b> The alignment the "
                        "fields are built on does not survive it, and neither does a model "
                        "trained on the present poses: align the set again afterwards. Leave "
                        "this off to compute the properties where the molecules stand.")
        if not self.ed_exe.text().strip():
            warn.append("<b>No ORCA program found.</b> Give the path to orca.exe above.")
        self.lbl_warn.setText("<br>".join(warn))


def _hms(seconds: float) -> str:
    """A duration a person can read; the dialogs quote their estimates in these."""
    s = int(round(max(0.0, float(seconds))))
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


#: how the task covariance may be chosen in the selectivity dialog
COVARIANCES = (("one correlation, cross-validated", "cv"),
               ("one correlation, by the evidence", "evidence"),
               ("free-form rank 1, by the evidence", "free1"),
               ("free-form rank 2, by the evidence", "free2"))


class SelectivityDialog(QDialog):
    """Refit the current model as several endpoints, so that selectivity becomes a co-field.

    A separation problem is two properties, and what matters is their difference.  Because a CMF
    model is linear in the molecular field, the difference of two such models is itself a model
    linear in the field -- so the difference has a co-field of its own, and it can be read, drawn
    and exported exactly like any other.

    Any number of endpoints can be fitted together.  With more than two, one shared correlation
    assumes that every pair is related to the same degree, which is rarely true, so the covariance
    can instead be a free-form low-rank matrix selected by the marginal likelihood.
    """

    def __init__(self, props, current, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model several endpoints (selectivity co-field)")
        lay = QVBoxLayout(self)
        info = QLabel(
            "Every endpoint is fitted on the <b>fields, attenuation and kernel of the model you "
            "have already built</b>; only the task covariance and the ridge are searched here. "
            "Each endpoint then has its own co-field, and so does the difference of the pair you "
            "name below &mdash; the <b>selectivity co-field</b>, whose positive zones are where "
            "more of a field raises the first endpoint more than the second."
            "<p>Molecules missing some of the measurements are kept: they still inform the "
            "endpoints they do carry, and through the covariance they inform the others. "
            "Cross-validation holds out whole molecules, never single measurements, because a "
            "molecule with one endpoint in the training set and another in the test set would "
            "have its answer handed over by the covariance.</p>"
            "<p>With three or more endpoints, a <b>free-form</b> covariance is usually better: "
            "one shared correlation forces every pair of endpoints to be related to the same "
            "degree. The free-form matrix is selected by the marginal likelihood, which needs one "
            "matrix factorisation per candidate instead of a full cross-validation, and its "
            "diagonal is free, so endpoints measured in different units get separate "
            "variances.</p>")
        info.setWordWrap(True)
        info.setTextFormat(Qt.RichText)
        info.setMaximumWidth(620)
        lay.addWidget(info)

        lay.addWidget(QLabel("Endpoints to fit together:"))
        holder = QWidget()
        inner = QVBoxLayout(holder)
        inner.setContentsMargins(4, 0, 4, 0)
        self.checks = []
        for name in props:
            c = QCheckBox(str(name))
            inner.addWidget(c)
            self.checks.append(c)
        area = QScrollArea()
        area.setWidget(holder)
        area.setWidgetResizable(True)
        area.setMaximumHeight(160)
        lay.addWidget(area)

        form = QFormLayout()
        self.cmb_a = QComboBox()
        self.cmb_b = QComboBox()
        for c in (self.cmb_a, self.cmb_b):
            c.addItems(list(props))
        if current in props:
            self.cmb_a.setCurrentText(current)
        if len(props) > 1:
            self.cmb_b.setCurrentIndex(1 if self.cmb_a.currentIndex() == 0 else 0)
        for c in self.checks:                       # the named pair starts out checked
            if c.text() in (self.cmb_a.currentText(), self.cmb_b.currentText()):
                c.setChecked(True)
        self.cmb_a.setToolTip("The endpoint whose increase the selectivity co-field shows as "
                              "positive, e.g. the affinity for americium")
        self.cmb_b.setToolTip("The endpoint subtracted, e.g. the affinity for europium")
        form.addRow("Difference, first endpoint:", self.cmb_a)
        form.addRow("Difference, second endpoint:", self.cmb_b)
        self.cmb_cov = QComboBox()
        for label, key in COVARIANCES:
            self.cmb_cov.addItem(label, key)
        self.cmb_cov.setToolTip("\n".join([
            "How the task covariance is chosen.",
            "One correlation: the matrix (1-rho) I + rho 11', one number for every pair.",
            "Free-form: L L' + diag(d), always positive definite, with a separate",
            "variance per endpoint; selected by the marginal likelihood."]))
        form.addRow("Task covariance:", self.cmb_cov)
        self.ed_rhos = QLineEdit("0, 0.2, 0.4, 0.6, 0.8, 0.9")
        self.ed_rhos.setToolTip("\n".join([
            "Task correlations to try; the cross-validation chooses one.",
            "0 fits the endpoints separately, sharing only the kernel.",
            "Towards 1 they share information with each other, and at 1 exactly the",
            "model could only predict a constant selectivity, so 1 is excluded."]))
        form.addRow("Task correlation:", self.ed_rhos)
        self.ed_lams = QLineEdit("0.001, 0.01, 0.1, 1, 10")
        form.addRow("Ridge (lambda):", self.ed_lams)
        self.spin_folds = QSpinBox()
        self.spin_folds.setRange(2, 50)
        self.spin_folds.setValue(10)
        form.addRow("Cross-validation folds:", self.spin_folds)
        lay.addLayout(form)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("Fit")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def endpoints(self):
        """Every endpoint to fit, with the named pair first and second."""
        a, b = self.cmb_a.currentText(), self.cmb_b.currentText()
        chosen = [c.text() for c in self.checks if c.isChecked()]
        out = [a] + ([b] if b != a else [])
        out += [n for n in chosen if n not in out]
        return tuple(out)

    def covariance(self):
        return self.cmb_cov.currentData() or "cv"

    def rhos(self):
        out = []
        for part in self.ed_rhos.text().replace(";", ",").split(","):
            part = part.strip()
            if part:
                try:
                    v = float(part)
                except ValueError:
                    continue
                if -0.99 < v < 1.0:
                    out.append(v)
        return tuple(out) or (0.0,)

    def lams(self):
        out = []
        for part in self.ed_lams.text().replace(";", ",").split(","):
            part = part.strip()
            if part:
                try:
                    out.append(float(part))
                except ValueError:
                    pass
        return tuple(out) or (1.0,)


class CompareModelsDialog(QDialog):
    """Which two models to score on the loaded molecules (plan item G2)."""
    SOURCES = ["the model of this session", "a model file (.cmf)", "a project file (.cmfproj)"]

    def __init__(self, has_current: bool, parent=None):
        from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
                                       QPushButton, QVBoxLayout)
        super().__init__(parent)
        self.setWindowTitle("Compare two models on these molecules")
        lay = QVBoxLayout(self)
        note = QLabel("Both models predict every loaded molecule as it is posed now, so a model "
                      "fitted on another alignment of the series is judged on this one. That is "
                      "the comparison to make after a realignment, and its caveat: a model built "
                      "in another frame will look wrong here because the frame moved, not "
                      "because the model is worse.")
        note.setWordWrap(True)
        lay.addWidget(note)
        grid = QGridLayout()
        for col, head in enumerate(["", "Source", "File", "", "Label"]):
            grid.addWidget(QLabel(head), 0, col)
        self._rows = []
        for k, letter in enumerate(("A", "B")):
            cmb = QComboBox()
            cmb.addItems(self.SOURCES)
            cmb.setCurrentIndex(0 if (k == 0 and has_current) else 1)
            ed = QLineEdit()
            ed.setPlaceholderText("model or project file")
            ed.setMinimumWidth(260)
            btn = QPushButton("Browse…")
            lab = QLineEdit(letter)
            lab.setMaximumWidth(120)
            grid.addWidget(QLabel(f"Model {letter}"), k + 1, 0)
            grid.addWidget(cmb, k + 1, 1)
            grid.addWidget(ed, k + 1, 2)
            grid.addWidget(btn, k + 1, 3)
            grid.addWidget(lab, k + 1, 4)
            btn.clicked.connect(lambda _=False, e=ed, c=cmb: self._browse(e, c))
            cmb.currentIndexChanged.connect(lambda ix, e=ed, b=btn: (e.setEnabled(ix > 0),
                                                                     b.setEnabled(ix > 0)))
            ed.setEnabled(cmb.currentIndex() > 0)
            btn.setEnabled(cmb.currentIndex() > 0)
            self._rows.append((cmb, ed, lab))
        lay.addLayout(grid)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _browse(self, ed, cmb):
        proj = cmb.currentIndex() == 2
        flt = (f"CMF project (*{PROJECT_EXTENSION});;All files (*)" if proj
               else "CMF model (*.cmf);;All files (*)")
        path, _ = QFileDialog.getOpenFileName(self, "Choose a model", "", flt)
        if path:
            ed.setText(path)

    def _accept(self):
        for k, (cmb, ed, _lab) in enumerate(self._rows):
            if cmb.currentIndex() > 0 and not os.path.exists(ed.text().strip()):
                QMessageBox.information(self, "File", f"Model {'AB'[k]}: choose an existing file.")
                return
        self.accept()

    def side(self, k: int):
        """(source index, file path, label) of model A (0) or B (1)."""
        cmb, ed, lab = self._rows[k]
        return cmb.currentIndex(), ed.text().strip(), lab.text().strip()


class ComparisonWindow(QDialog):
    """The two models' predictions side by side, with the summary and a CSV export."""

    def __init__(self, cmp, parent=None):
        from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QLabel, QPushButton,
                                       QTableWidget, QTableWidgetItem, QVBoxLayout)
        from .panels import _NumItem
        super().__init__(parent)
        self.cmp = cmp
        self.setWindowTitle(f"Two models: {cmp.label_a} against {cmp.label_b}")
        self.setModal(False)
        self.resize(760, 560)
        lay = QVBoxLayout(self)
        summary = QLabel("\n".join(cmp.lines()))
        summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(summary)
        heads = ["Molecule", "Subset", "Experimental", cmp.label_a, cmp.label_b, "B − A"]
        tbl = QTableWidget(len(cmp.rows), len(heads))
        tbl.setHorizontalHeaderLabels(heads)
        tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        for r, row in enumerate(cmp.rows):
            tbl.setItem(r, 0, QTableWidgetItem(row.name))
            tbl.setItem(r, 1, QTableWidgetItem(row.subset))
            for c, v in enumerate((row.y, row.pred_a, row.pred_b, row.difference), start=2):
                txt = "" if not np.isfinite(v) else (f"{v:+.3f}" if c == 5 else f"{v:.3f}")
                tbl.setItem(r, c, _NumItem(v if np.isfinite(v) else -np.inf, txt))
        tbl.setSortingEnabled(True)
        tbl.sortItems(5, Qt.DescendingOrder)
        tbl.resizeColumnsToContents()
        self.table = tbl
        lay.addWidget(tbl, 1)
        row = QHBoxLayout()
        row.addStretch(1)
        btn_csv = QPushButton("Save as CSV…")
        btn_csv.clicked.connect(self._save_csv)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        row.addWidget(btn_csv)
        row.addWidget(btn_close)
        lay.addLayout(row)

    def _save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save the comparison", "comparison.csv",
                                              "CSV (*.csv)")
        if path:
            self.cmp.to_csv(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"CMF 3D-QSAR  –  Continuous Molecular Fields  (v{__version__})")
        self.resize(1500, 950)

        self.ds: Optional[Dataset] = None
        self.molfields: List[MoleculeFields] = []
        self.fieldset: FieldSet = FieldSet.create()
        self.model: Optional[CMFModel] = None
        self.model_train_idx: List[int] = []
        #: a two-endpoint fit, when one has been made: its coefficients
        #: stand in for the model's own so that every co-field reader works
        self.multitask = None
        self.endpoint_label = ""
        self.model_test_idx: List[int] = []
        self.shapley_result = None          # (molecule index, ShapleyResult) of the last run
        self.validation = None          # y-randomisation / bootstrap / domain of the last fit
        self.shapley_fields = None          # (molecule index, (F, N) values) for the Shapley field
        self.pharmacophore = None           # features extracted from the co-field, drawn in 3D
        self.topomer_core = None            # set once the molecules are topomer substituents
        self.field_reference = None         # molecule a candidate is compared with
        self.field_diff_rows = None
        self.protein = None                 # cmfqsar.protein.Protein behind the molecules
        self.complementarity = None         # scores of the series against the pocket
        self.chemotypes = None              # (labels, scaffold names) of a heterogeneous series
        self.chemotype_result = None
        self._comp_surface_cache = {}
        self._site_cache = None             # (key, Protein) of the last binding site drawn
        self.search_rows = []
        self.grid: Optional[Grid] = None
        self.volumes: Dict[str, np.ndarray] = {}
        self.cofield_sd: Dict[str, np.ndarray] = {}      # standard deviation of each co-field
        self._mol_volume_cache: Dict[tuple, np.ndarray] = {}
        self.worker: Optional[TaskWorker] = None
        self._pose_reference: Optional[Dict[int, np.ndarray]] = None   # poses before "Unalign"
        self.history = PoseHistory()        # the poses before every step that moved molecules
        self.project_path: Optional[str] = None
        self.model_target = None
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(60)
        self._display_timer.timeout.connect(self.update_display)

        # central: viewer + controls
        central = QWidget()
        lay = QVBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        self.controls = ViewerControls()
        self.viewer = Viewer3D()
        lay.addWidget(self.controls)
        lay.addWidget(self.viewer, 1)
        self.setCentralWidget(central)

        # docks
        self.dataset_panel = DatasetPanel()
        self.model_panel = ModelPanel()
        self.results_panel = ResultsPanel()
        # side docks own the corners, so the Results dock sits only under the 3D view and the
        # horizontal separator is not constrained by the height of the side panels
        self.setCorner(Qt.BottomLeftCorner, Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, Qt.RightDockWidgetArea)
        self._dock(self.dataset_panel, "Data set", Qt.LeftDockWidgetArea)
        scroll = QScrollArea()
        scroll.setWidget(self.model_panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        # The panel is tall and wide; forcing the dock to its full natural width squeezed the
        # 3D view, and with horizontal scrolling off the right-hand columns of the field table
        # simply could not be reached. Ask for a workable width and let the dock scroll.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setMinimumWidth(min(self.model_panel.minimumSizeHint().width() + 20, 520))
        self._dock(scroll, "Model", Qt.RightDockWidgetArea)
        d = self._dock(self.results_panel, "Results", Qt.BottomDockWidgetArea)
        d.setMinimumHeight(160)
        self.resizeDocks([d], [280], Qt.Vertical)

        # status bar
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(300)
        self.progress.setVisible(False)
        self.status_label = QLabel("Ready")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.progress)

        self._build_menu()
        self._connect()
        self.refresh_fieldset()
        self.restore_saved_layout()

    # ------------------------------------------------------------------ setup
    def _dock(self, widget, title, area):
        d = QDockWidget(title, self)
        d.setWidget(widget)
        d.setObjectName(title)
        self.addDockWidget(area, d)
        return d

    def _build_menu(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        self._act(fm, "Open training set…", lambda: self.open_dataset("train"), "Ctrl+O")
        self._act(fm, "Add test set…", lambda: self.open_dataset("test"))
        self._act(fm, "Add decoys (background set)…", lambda: self.open_dataset("decoy"))
        self._act(fm, "Import activities from CSV…", self.import_activities)
        fm.addSeparator()
        self._act(fm, "Load protein structure (PDB/mmCIF)…", self.load_protein_file)
        self._act(fm, "Fetch protein from the PDB…", self.fetch_protein)
        self._act(fm, "Forget protein structure", self.clear_protein)
        fm.addSeparator()
        self._act(fm, "Open project…", self.open_project_file, "Ctrl+Shift+O")
        self._act(fm, "Save project", self.save_project_file, "Ctrl+Shift+S")
        self._act(fm, "Save project as…", lambda: self.save_project_file(save_as=True))
        fm.addSeparator()
        self._act(fm, "Load model…", self.load_model_file, "Ctrl+L")
        self._act(fm, "Save model…", self.save_model_file, "Ctrl+S")
        fm.addSeparator()
        self._act(fm, "Export predictions (CSV)…", self.export_predictions)
        self._act(fm, "Export aligned data set (SDF)…", self.export_sdf)
        self._act(fm, "Export current co-field (Gaussian cube)…", self.export_cube)
        self._act(fm, "Export current Shapley field (Gaussian cube)…", self.export_shapley_cube)
        self._act(fm, "Save 3D view as image…", self.save_screenshot)
        fm.addSeparator()
        self._act(fm, "Quit", self.close, "Ctrl+Q")
        vm = mb.addMenu("&View")
        for d in self.findChildren(QDockWidget):
            vm.addAction(d.toggleViewAction())
        vm.addSeparator()
        self.act_full_width = self._act(vm, "Results panel spans full width", self.toggle_results_span)
        self.act_full_width.setCheckable(True)
        self.act_full_width.setToolTip("Place the Results panel under the side panels too; the border between "
                                       "the side panels and Results then moves vertically")
        self._act(vm, "Restore default layout", self.restore_layout)
        self._default_state = self.saveState()
        tm = mb.addMenu("&Tools")
        self._act(tm, "Generate conformations (4D)…", self.generate_conformers_dialog)
        self._act(tm, "Dock into the pocket…", self.dock_dialog)
        self._act(tm, "Align data set to reference…", self.align_dataset_dialog)
        self._act(tm, "Align by prediction confidence (GP variance)…", self.variance_align_dialog)
        self._act(tm, "Align training set by prediction confidence (cross-validated)…",
                  self.train_align_dialog)
        tm.addSeparator()
        self._act(tm, "Replace molecules by their topomer substituents…", self.topomer_dialog)
        self._act(tm, "Field difference from a reference molecule…", self.field_difference_dialog)
        self._act(tm, "Extract pharmacophore from the co-field…", self.pharmacophore_dialog)
        self._act(tm, "Model two endpoints (selectivity co-field)…",
                  self.selectivity_dialog)
        self._act(tm, "Chemical space map…", self.gtm_map_dialog)
        tm.addSeparator()
        self._act(tm, "Align decoys to the training set…", self.align_decoys_dialog)
        self._act(tm, "Unalign (randomise poses)…", self.unalign_dialog)
        self.act_undo = self._act(tm, "Undo last alignment step", self.undo_alignment, "Ctrl+Z")
        self.act_undo.setEnabled(False)
        self._act(tm, "Compare two models on these molecules…", self.compare_models_dialog)
        self._act(tm, "Center view on current molecule", self.center_current)
        tm.addSeparator()
        self._act(tm, "Quantum-chemical properties with ORCA…", self.orca_dialog)
        self._act(tm, "Save the data set with its properties (SDF)…",
                  self.save_annotated_sdf)
        tm.addSeparator()
        hm = tm.addMenu("Heterogeneous sets (several chemotypes)")
        hm.setToolTip("A series of several scaffolds has no single common substructure, so the "
                      "usual recipe quietly stops meaning what it says")
        self._act(hm, "1. Find the chemotypes (Murcko scaffolds)…", self.chemotypes_dialog)
        self._act(hm, "2. Align within each chemotype", self.align_within_chemotypes)
        self._act(hm, "3. Place the chemotypes in one frame…", self.place_chemotypes_dialog)
        self._act(hm, "4. Leave one chemotype out, with the alignment-free baseline…",
                  self.chemotype_validation_dialog)
        hm = mb.addMenu("&Help")
        self._act(hm, "About CMF 3D-QSAR / method summary", self.show_about)

    def toggle_results_span(self, full_width: bool):
        """Choose whether the bottom (Results) dock spans the whole width or only the central area.

        Qt dock separators move only perpendicular to themselves, so this switch decides which
        borders exist: with full width, the border between the side panels and Results moves
        vertically; otherwise the side panels run the full height.
        """
        area = Qt.BottomDockWidgetArea if full_width else None
        self.setCorner(Qt.BottomLeftCorner, area or Qt.LeftDockWidgetArea)
        self.setCorner(Qt.BottomRightCorner, area or Qt.RightDockWidgetArea)

    def restore_layout(self):
        self.act_full_width.setChecked(False)
        self.toggle_results_span(False)
        self.restoreState(self._default_state)

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        a.triggered.connect(slot)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
        menu.addAction(a)
        return a

    def _connect(self):
        dp, mp, rp, vc = self.dataset_panel, self.model_panel, self.results_panel, self.controls
        dp.load_requested.connect(self.open_dataset)
        dp.selection_changed.connect(lambda i: self.schedule_display())
        dp.show_changed.connect(self.schedule_display)
        dp.btn_split.clicked.connect(self.random_split)
        dp.btn_to_train.clicked.connect(lambda: self.set_subset("train"))
        dp.btn_to_test.clicked.connect(lambda: self.set_subset("test"))
        dp.cmb_activity.currentIndexChanged.connect(self.activity_changed)
        mp.fields_changed.connect(self.refresh_fieldset)
        mp.build_requested.connect(self.build_model)
        mp.nested_requested.connect(self.run_nested_cv)
        mp.predict_requested.connect(self.predict_test)
        mp.cancel_requested.connect(self.cancel_worker)
        rp.point_clicked.connect(self.dataset_panel.select_index)
        vc.changed.connect(self.schedule_display)
        vc.style_changed.connect(self.schedule_display)
        vc.compute_requested.connect(self.compute_cofields)
        vc.cofield_noise_requested.connect(self.cofield_noise_toggled)
        vc.endpoint_changed.connect(self.endpoint_changed)
        vc.reset_view.connect(self.viewer.reset_view)
        rp.btn_shapley.clicked.connect(self.compute_shapley)
        rp.btn_complement.clicked.connect(self.compute_complementarity)

    # ------------------------------------------------------------------ helpers
    def log(self, text: str):
        self.results_panel.append_log(time.strftime("%H:%M:%S  ") + text)
        self.status_label.setText(text.splitlines()[0][:120])

    def error(self, text: str):
        self.log("ERROR: " + text.splitlines()[0])
        QMessageBox.critical(self, "Error", text)

    def run_task(self, fn, on_done, label="Working"):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.information(self, "Busy", "A computation is already running.")
            return
        self.worker = TaskWorker(fn, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(lambda res: self._task_finished(res, on_done))
        self.worker.failed.connect(self._task_failed)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.model_panel.set_busy(True)
        self.status_label.setText(label)
        self.worker.start()

    def _on_progress(self, msg, frac):
        self.progress.setValue(int(frac * 100))
        self.status_label.setText(msg[:120])

    def _task_finished(self, res, on_done):
        self.progress.setVisible(False)
        self.model_panel.set_busy(False)
        try:
            on_done(res)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.error(f"{exc}\n\n{traceback.format_exc()}")

    def _task_failed(self, msg):
        self.progress.setVisible(False)
        self.model_panel.set_busy(False)
        if msg == "Cancelled":
            self.log("Cancelled")
        else:
            self.error(msg)

    def cancel_worker(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.status_label.setText("Cancelling…")

    def schedule_display(self):
        self._display_timer.start()

    # ------------------------------------------------------------------ data set
    def open_dataset(self, kind: str = "train"):
        titles = {"train": "Open training set", "test": "Add test set",
                  "decoy": "Add decoys (background molecules, presumed inactive)"}
        path, _ = QFileDialog.getOpenFileName(
            self, titles.get(kind, "Open"), "",
            "Structure files (*.sdf *.sd *.mol *.mol2 *.smi *.csv *.tsv *.txt);;All files (*)")
        if not path:
            return
        self.load_dataset_file(path, kind)

    def load_dataset_file(self, path: str, kind: str = "train", activity: Optional[str] = None,
                          on_loaded=None):
        def task(progress, cancel):
            progress(f"Reading {os.path.basename(path)}", 0.2)
            return load_dataset(path, activity_prop=activity, add_hs=True, subset=kind)

        def done(ds: Dataset):
            props = numeric_properties(ds.mols)
            if activity is None and props and QApplication.instance() is not None and not getattr(self, "_auto", False):
                cur = props.index(ds.activity_prop) if ds.activity_prop in props else 0
                item, ok = QInputDialog.getItem(self, "Activity property", "SD property with the activity values:",
                                                props, cur, False)
                if ok and item:
                    ds.set_activity_from_property(item)
            if kind == "train" or self.ds is None:
                self.ds = ds
            elif kind == "decoy":
                ds.subset = np.array(["decoy"] * len(ds), dtype=object)
                self.ds.append(ds, subset="decoy")
            else:
                self.ds.append(ds, subset="test")
                if not self.ds.activity_prop and ds.activity_prop:
                    self.ds.activity_prop = ds.activity_prop
                if ds.activity_prop and self.ds.activity_prop == ds.activity_prop:
                    pass
            self._after_dataset_change()
            self.log(f"Loaded {len(ds)} molecules from {path} as {kind} set "
                     f"(activity: {ds.activity_prop or 'none'})")
            if kind == "train":
                self._autoload_pocket(path)
            self._update_conformer_info()
            if on_loaded:
                on_loaded()

        self.run_task(task, done, "Loading data set")

    def _after_dataset_change(self):
        ds = self.ds
        self.molfields = [build_fields(m, FieldSet(specs=[]), n) for m, n in zip(ds.mols, ds.names)]
        self.chemotypes = self.chemotype_result = None   # another series, another grouping
        self.clear_pharmacophore()                       # the features belonged to the old model
        self.dataset_panel.populate(ds.names, ds.activity, ds.subset)
        props = numeric_properties(ds.mols)
        cmb = self.dataset_panel.cmb_activity
        cmb.blockSignals(True)
        cmb.clear()
        cmb.addItems(props)
        if ds.activity_prop in props:
            cmb.setCurrentIndex(props.index(ds.activity_prop))
        cmb.blockSignals(False)
        self.model_panel.set_class_properties(list(ds.properties))
        self.model_panel.set_atom_properties(ds.atom_properties)
        from ..descriptors import available as _numeric_props
        self.model_panel.set_descriptor_properties(
            [p for p in _numeric_props(ds.mols) if p != ds.activity_prop])
        if np.isfinite(ds.activity).any():
            self.model_panel.spin_threshold.setValue(float(np.nanmedian(ds.activity)))
        self.model = None
        self.volumes.clear()
        self._mol_volume_cache.clear()
        self.grid = None
        self.results_panel.clear_points()
        self.refresh_fieldset()
        self.dataset_panel.select_index(0)
        all_coords = np.concatenate([m.coords for m in self.molfields])
        self.viewer.center_on(all_coords)
        self.schedule_display()

    def activity_changed(self, ix):
        if self.ds is None or ix < 0:
            return
        prop = self.dataset_panel.cmb_activity.currentText()
        self.ds.set_activity_from_property(prop)
        self.dataset_panel.update_activity(self.ds.activity)
        self.log(f"Activity property: {prop}")

    def import_activities(self):
        if self.ds is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import activities (name,value CSV)", "", "CSV (*.csv *.txt)")
        if not path:
            return
        from ..dataio import read_activities_csv
        vals = read_activities_csv(path)
        n = 0
        for i, nm in enumerate(self.ds.names):
            if nm in vals:
                self.ds.activity[i] = vals[nm]
                n += 1
        self.ds.activity_prop = os.path.basename(path)
        self.dataset_panel.update_activity(self.ds.activity)
        self.log(f"Imported {n} activity values from {path}")

    def random_split(self):
        if self.ds is None:
            return
        dec = [int(i) for i in self.ds.decoy_indices]
        self.ds.random_split(self.dataset_panel.spin_test_frac.value(), seed=self.model_panel.spin_seed.value())
        if dec:
            self.ds.subset[dec] = "decoy"          # the background is never trained or tested on
        self.dataset_panel.update_subsets(self.ds.subset)
        self.refresh_fieldset()

    def set_subset(self, which: str):
        if self.ds is None:
            return
        for i in self.dataset_panel.selected_indices():
            self.ds.subset[i] = which
        self.dataset_panel.update_subsets(self.ds.subset)
        self.refresh_fieldset()

    # ------------------------------------------------------------------ fields
    def refresh_fieldset(self):
        mp = self.model_panel
        cif_types = []
        if mp.chk_cif.isChecked() and self.ds is not None:
            tr = [self.ds.mols[i] for i in self.ds.train_indices]
            if tr:
                cif_types = cif_types_in(tr, mp.cmb_cif_scheme.currentText(), mp.spin_cif_min.value())
        fs = FieldSet.create(physchem=mp.selected_physchem(), cif_types=cif_types, alpha=1.0,
                             atom_properties=mp.atom_properties(),
                             charge_model=mp.cmb_charges.currentText(), cif_scheme=mp.cmb_cif_scheme.currentText(),
                             param_scheme=mp.cmb_param.currentText(),
                             conformers=mp.conformer_mode(),
                             conformer_weighting=mp.conformer_weighting(),
                             conformer_kt=mp.spin_conf_kt.value(),
                             max_conformers=mp.spin_conf_max.value(),
                             descriptors=mp.descriptors(),
                             descriptor_h=mp.descriptor_h(),
                             probes=mp.probes(), probe_terms=mp.probe_terms())
        vals = mp.table_values()
        for s in fs.specs:
            if s.name in vals:
                s.alpha, s.h, s.enabled = vals[s.name]["alpha"], vals[s.name]["h"], vals[s.name]["enabled"]
        mode_changed = getattr(self, "_shown_conformer_mode", None) != fs.conformers
        self.fieldset = fs
        mp.fill_field_table(fs)
        self.controls.set_fields([(s.name, s.label) for s in fs.specs])
        if mode_changed and self.ds is not None and self.molfields:
            # switching between one conformation and the ensemble changes what is on screen, so
            # the drawn molecules are rebuilt rather than left half in one mode and half in the other
            self._shown_conformer_mode = fs.conformers
            self.molfields = [build_fields(m, FieldSet(specs=[], conformers=fs.conformers), n)
                              for m, n in zip(self.ds.mols, self.ds.names)]
            self._mol_volume_cache.clear()
            self.schedule_display()
        else:
            self._shown_conformer_mode = fs.conformers

    def _fieldset_from_panel(self) -> FieldSet:
        self.refresh_fieldset()
        fs = self.fieldset.copy()
        fs.specs = [s for s in fs.specs if s.enabled]
        return fs

    # ------------------------------------------------------------------ modelling
    def _check_ready(self) -> bool:
        if self.ds is None:
            QMessageBox.information(self, "No data", "Load a training set first.")
            return False
        ok = np.isfinite(self.ds.activity)
        if sum(1 for i in self.ds.train_indices if ok[i]) < 4:
            QMessageBox.information(self, "No activities", "The training set needs at least 4 molecules with activity values.")
            return False
        return True

    def build_model(self):
        if not self._check_ready():
            return
        self.clear_pharmacophore()          # a new model means new co-fields and new features
        fs = self._fieldset_from_panel()
        if not fs.specs:
            QMessageBox.information(self, "No fields", "Select at least one molecular field.")
            return
        mp = self.model_panel
        ds = self.ds
        ok = np.isfinite(ds.activity)
        tr = [int(i) for i in ds.train_indices if ok[i]]
        te = [int(i) for i in ds.test_indices]
        dec = [int(i) for i in ds.decoy_indices]
        method = mp.method_key()
        alphas = mp.alphas() if mp.rad_alpha_grid.isChecked() else None
        lams, Cs, eps, nus = mp.lams(), mp.Cs(), mp.epsilons(), mp.nus()
        oc_params = mp.oneclass_params()
        kernel_src, ac_sigma = mp.kernel_source(), mp.ac_sigma()
        ac_chirality = mp.ac_chirality()
        folds, seed = mp.spin_folds.value(), mp.spin_seed.value()
        n_scramble, n_boot = mp.spin_scramble.value(), mp.spin_boot.value()
        want_domain = mp.chk_domain.isChecked()
        scramble_select = mp.chk_scramble_select.isChecked()
        optimize_h = mp.cmb_mix.currentIndex() == 1
        mkl = mp.mkl_mode()
        layer, gammas = mp.kernel_layer(), mp.gammas()
        normalize, scale = mp.chk_normalize.isChecked(), mp.chk_scale.isChecked()
        learn_weights_first = mp.learns_weights() and fs.is_ensemble
        mil_beta, mil_selection = mp.mil_beta(), mp.mil_selection()
        if layer != "linear":
            self.log(f"Nonlinear kernel layer '{layer}': the model is no longer linear in the "
                     "molecular field, so co-fields and atom contributions are not available for it; "
                     "the decomposition of a prediction into training-set similarities stays valid.")
        if learn_weights_first:
            self.log(f"Multi-instance mode: the weights of the conformations are fitted "
                     f"({mil_selection}, beta {mil_beta:g}) and then used as an ordinary ensemble, "
                     f"so co-fields and the decompositions stay available.")
        mols, names = ds.mols, ds.names
        task_name = mp.task_key()
        y_all = ds.activity.copy()
        if task_name in ("classification", "oneclass"):
            if mp.cmb_class_src.currentIndex() == 1:
                prop = mp.cmb_class_prop.currentText()
                if not prop:
                    QMessageBox.information(self, "No class property",
                                            "Choose the SD property that holds the class.")
                    return
                y_all = ds.classes_from_property(prop)
                src = f"property '{prop}'"
            else:
                thr = mp.spin_threshold.value()
                y_all = ds.classes_from_threshold(thr, active_above=not mp.chk_active_below.isChecked())
                rel = "<=" if mp.chk_active_below.isChecked() else ">="
                src = f"{ds.activity_prop} {rel} {thr:g}"
            tr = [i for i in tr if np.isfinite(y_all[i])]
            n_pos = int(np.sum(y_all[tr] > 0))
            if task_name == "classification" and (n_pos < 2 or len(tr) - n_pos < 2):
                QMessageBox.information(self, "Classes", "Both classes need at least two training "
                                                         "molecules; adjust the threshold.")
                return
            if task_name == "oneclass":
                tr = [i for i in tr if y_all[i] > 0]
                if len(tr) < 4:
                    QMessageBox.information(self, "Too few actives",
                                            "A one-class model needs at least four active molecules.")
                    return
            self.log(f"Classes from {src}: {n_pos} active, {len(tr) - n_pos if task_name == 'classification' else 0}"
                     f" inactive in the training set"
                     + (f"; one-class model fitted on the {len(tr)} actives" if task_name == "oneclass" else ""))

        def task(progress, cancel):
            if learn_weights_first:
                progress("Learning the conformation weights", 0.02)
                _learn_conformer_weights(mols, names, tr, te, y_all, fs, mil_beta, mil_selection,
                                         lams, normalize, progress)
            mf = []
            for k, (m, n) in enumerate(zip(mols, names)):
                mf.append(build_fields(m, fs, n))
                if k % 20 == 0:
                    progress(f"Computing molecular fields {k + 1}/{len(mols)}", 0.05 * k / len(mols))
            train = AtomTable.from_molecules([mf[i] for i in tr])
            model = CMFModel(fieldset=fs, task=task_name, method=method, normalize=normalize,
                             scale_fields=scale, kernel_layer=layer, kernel_source=kernel_src,
                             ac_sigma=ac_sigma, ac_chirality=ac_chirality)
            if task_name == "oneclass":
                # no cross-validation objective exists without negatives: alpha sets the level of
                # generalisation and nu the expected fraction of outliers, both chosen by the user
                if alphas:
                    model.fieldset.set_alpha(alphas[0])
                model.params["nu"] = nus[0]
                model.params.update(oc_params)
                progress("Fitting the one-class model", 0.9)
                model.fit(train, y_all[tr], cv_folds=0)
                res = None
            else:
                res = grid_search(train, y_all[tr], model, alphas, lams=lams, Cs=Cs, epsilons=eps,
                                  nus=nus, gammas=gammas, folds=folds, seed=seed,
                                  optimize_h=optimize_h, mkl=mkl,
                                  progress=lambda m_, f_: progress(m_, 0.05 + 0.85 * f_), cancel=cancel)
                progress("Final fit and cross-validation", 0.92)
                model.fit(train, y_all[tr], cv_folds=folds, cv_seed=seed, reuse_kernels=True)
            if te:
                progress("Predicting the test set", 0.97)
                test = AtomTable.from_molecules([mf[i] for i in te])
                model.evaluate_test(test, y_all[te], cancel=cancel)
            d_scores = None
            if dec and task_name in ("classification", "oneclass"):
                progress("Scoring the background set", 0.99)
                d_scores = model.predict(AtomTable.from_molecules([mf[i] for i in dec]))
                act = [i for i in te if y_all[i] > 0] or [i for i in tr if y_all[i] > 0]
                if act:
                    from ..model import screening_scores
                    y_scr = np.concatenate([np.ones(len(act)), -np.ones(len(dec))])
                    s_act = (model.stats.extra["y_test_pred"][[te.index(i) for i in act]]
                             if act and act[0] in te else
                             np.asarray(model.stats.extra.get("y_fit"))[[tr.index(i) for i in act]])
                    model.stats.screening = screening_scores(y_scr, np.concatenate([s_act, d_scores]))
                    model.stats.extra["screening_truth"] = y_scr
                    model.stats.extra["screening_scores"] = np.concatenate([s_act, d_scores])
            validation = _run_validation(model, mf, tr, te, y_all, names, task_name,
                                         n_scramble, n_boot, want_domain, folds, seed,
                                         progress, cancel, scramble_select=scramble_select,
                                         alphas=alphas, lams=lams)
            return {"model": model, "mf": mf, "res": res, "tr": tr, "te": te, "dec": dec,
                    "d_scores": d_scores, "target": y_all, "validation": validation}

        self.run_task(task, self._model_built, "Building model")

    def _model_built(self, out):
        self.model = out["model"]
        self.shapley_result = None
        self.shapley_fields = None
        self.molfields = out["mf"]
        self.model_train_idx, self.model_test_idx = out["tr"], out["te"]
        self.model_decoy_idx, self.model_decoy_scores = out.get("dec", []), out.get("d_scores")
        res = out["res"]
        self.model_target = out.get("target")
        self.validation = out.get("validation")
        self.search_rows = res.table if res is not None else []
        self.volumes.clear()
        self._mol_volume_cache.clear()
        self.grid = None
        self.fieldset = self.model.fieldset
        self.model_panel.show_fitted_values(self.model.fieldset, self.model.scale)
        self.controls.set_fields([(s.name, s.label) for s in self.model.fieldset.specs])
        self._show_results()
        if res is None:
            a = self.model.fieldset.specs[0].alpha if self.model.fieldset.specs else float("nan")
            self.log(f"One-class model built: alpha = {a:g}, nu = {self.model.params.get('nu', 0.5):g} "
                     f"(not selected by cross-validation: alpha sets the level of generalisation, "
                     f"nu the expected fraction of outliers)")
            self.schedule_display()
            return
        atxt = "table" if not np.isfinite(res.best_alpha) else f"{res.best_alpha:g}"
        if self.model.task == "classification":
            self.log(f"Classification model built: alpha = {atxt}, {res.best_params}, "
                     f"1-AUCcv = {res.best_rmse:.3f}, balanced accuracy = {res.best_q2:.3f}")
        else:
            self.log(f"Model built: alpha = {atxt}, {res.best_params}, "
                     f"RMSEcv = {res.best_rmse:.3f}, q2 = {res.best_q2:.3f}")
        if self.model_panel.cmb_mix.currentIndex() >= 1:
            how = ("coordinate descent" if self.model_panel.cmb_mix.currentIndex() == 1
                   else "kernel alignment")
            self.log(f"Mixing coefficients ({how}): "
                     + ", ".join(f"{k}={v:.3g}" for k, v in res.best_h.items()))
        self.schedule_display()

    def _show_results(self):
        m, ds = self.model, self.ds
        st = m.stats
        rp = self.results_panel
        tr, te = self.model_train_idx, self.model_test_idx
        kern = {"linear": "linear (overlap integral)", "gauss": "Gaussian layer",
                "arccos1": "arc-cosine layer", "arccos2": "two arc-cosine layers"}.get(
                    m.kernel_layer, str(m.kernel_layer))
        if not m.is_linear:
            kern += "; co-fields and atom contributions are not defined for it"
        if m.mkl:
            kern += f"; mixing coefficients by kernel alignment ({m.mkl})"
        txt = [st.summary(), "",
               f"Kernel: {kern}",
               f"Method: {m.method}   params: {m.params}   normalised kernels: {m.normalize}   field scaling: {m.scale_fields}",
               ("Kernel source: overlap integrals of the fields" if m.uses_fields else
                f"Kernel source: field autocorrelation"
                f"{', signed three-point channels' if m.ac_chirality else ''}"
                f", sigma {m.ac_sigma:g} A up to "
                f"{m.ac_d_max:g} A - invariant to rotation and translation, so no alignment is "
                f"assumed and no co-field exists"),
               "Fields: " + ", ".join(f"{s.label} (alpha={s.alpha:g}, h={s.h:g})" for s in m.fieldset.specs)]
        if m.method == "ranksvm":
            txt.insert(2, "Ranking SVM: fitted to the order of the activities and not to their "
                          "values, so the score has no units and the scatter below is a cloud that "
                          "rises rather than a line to lie on. Co-fields and contributions work as "
                          "usual, and say where more field promotes a molecule rather than what it "
                          "adds to the activity.")
        if m.method == "gpc":
            from ..gpc import brier_score
            try:
                txt.insert(2, f"Gaussian process: log marginal likelihood "
                              f"{m.log_marginal_likelihood:.2f}, Brier score of the fitted "
                              f"probabilities {brier_score(m.y, m.predict_proba(m.train)):.3f} "
                              f"(0.25 is a coin). The Scores tab carries p(active) from the latent mean.")
            except Exception as exc:                   # noqa: BLE001
                txt.insert(2, f"Gaussian process: probabilities unavailable ({exc})")
        if m.method == "kfda":
            from ..gpc import brier_score
            cal = getattr(m.learner_, "kfda_", None)
            try:
                txt.insert(2, "Kernel Fisher discriminant: the score is linear in the field, so "
                              "this classifier has a co-field (the Co-field tab works on it). "
                              + (f"Its {cal.describe()}, and the Brier score of the fitted "
                                 f"probabilities is "
                                 f"{brier_score(m.y, m.predict_proba(m.train)):.3f} "
                                 f"(0.25 is no better than chance)." if cal is not None else
                                 "Its threshold is the midpoint of the two class means."))
            except Exception as exc:                   # noqa: BLE001
                txt.insert(2, f"Kernel Fisher discriminant: probabilities unavailable ({exc})")
        val = getattr(self, "validation", None)
        if val and val.get("lines"):
            txt += ["", "Validation:"] + ["  " + line for line in val["lines"]]
        rp.txt_stats.setPlainText("\n".join(txt))
        if val:
            for line in val.get("lines", []):
                self.log(line)
            outside = val.get("outside") or []
            if outside:
                self.log(f"{len(outside)} molecules lie outside the applicability "
                         f"domain; their predictions are extrapolations:")
                for name, sim in sorted(outside, key=lambda kv: kv[1])[:10]:
                    self.results_panel.append_log(
                        f"    {name[:46]:46s} similarity {sim:.3f}")
        rp.set_search_table(self.search_rows)
        dp = self.dataset_panel
        dp.clear_predictions()
        dp.set_predictions(4, dict(zip(tr, st.extra.get("y_fit", []))))
        rp.clear_points()
        is_class = m.task in ("classification", "oneclass")
        x_tr = ds.activity[tr]
        x_te = ds.activity[te] if te else np.zeros(0)
        if "y_cv" in st.extra:
            dp.set_predictions(5, dict(zip(tr, st.extra["y_cv"])))
        if "y_test_pred" in st.extra and te:
            dp.set_predictions(6, dict(zip(te, st.extra["y_test_pred"])))
        val = getattr(self, "validation", None)
        dom = (val or {}).get("domain")
        if dom:
            dp.set_predictions(7, dict(zip(dom["index"], dom["similarity"])))
        if is_class:
            # one scatter of decision value against activity, coloured by the true class; it only
            # carries information when the classes came from a threshold on a continuous activity
            mp = self.model_panel
            by_threshold = mp.cmb_class_src.currentIndex() == 0
            thr = mp.spin_threshold.value() if by_threshold else None
            have_activity = bool(np.isfinite(ds.activity).any())
            idx = list(tr) + list(te)
            xs = np.concatenate([x_tr, x_te]) if te else np.asarray(x_tr)
            ys = np.concatenate([np.asarray(st.extra.get("y_cv", st.extra.get("y_fit", []))),
                                 np.asarray(st.extra.get("y_test_pred", []))]) if te                 else np.asarray(st.extra.get("y_cv", st.extra.get("y_fit", [])))
            target = getattr(self, "model_target", None)
            cls = np.asarray([target[i] if target is not None else np.nan for i in idx], dtype=float)
            is_test = [i in set(te) for i in idx]
            if len(xs) == len(ys) == len(idx):
                rp.set_class_points(xs, ys, cls, is_test, [ds.names[i] for i in idx], idx)
            rp.set_classification_mode(True, ds.activity_prop or "activity", threshold=thr,
                                       have_activity=have_activity)
        else:
            if "y_cv" in st.extra:
                rp.set_points("cv", x_tr, st.extra["y_cv"], [ds.names[i] for i in tr], tr)
            if "y_test_pred" in st.extra and te:
                rp.set_points("test", x_te, st.extra["y_test_pred"], [ds.names[i] for i in te], te)
            rp.set_classification_mode(False, ds.activity_prop or "activity")
        if is_class:
            target = getattr(self, "model_target", None)
            curves = []
            if target is not None:
                if "y_cv" in st.extra:
                    curves.append(("cross-validation", target[tr], np.asarray(st.extra["y_cv"])))
                elif "y_fit" in st.extra and m.task == "classification":
                    curves.append(("fit", target[tr], np.asarray(st.extra["y_fit"])))
                if "y_test_pred" in st.extra and te:
                    curves.append(("test set", target[te], np.asarray(st.extra["y_test_pred"])))
            # the screening comparison (actives against the background) is the curve that matters
            # for a one-class model, and the only one it can produce without a labelled test set
            if "screening_truth" in st.extra:
                curves.append(("actives vs decoys", st.extra["screening_truth"],
                               st.extra["screening_scores"]))
            rp.set_roc(curves)
            # the decision values of every scored molecule: the actual output of the model
            scores = np.full(len(ds), np.nan)
            for k, i in enumerate(tr):
                scores[i] = st.extra.get("y_cv", st.extra.get("y_fit", []))[k]
            for k, i in enumerate(te):
                if "y_test_pred" in st.extra:
                    scores[i] = st.extra["y_test_pred"][k]
            d_scores = getattr(self, "model_decoy_scores", None)
            for k, i in enumerate(getattr(self, "model_decoy_idx", [])):
                if d_scores is not None:
                    scores[i] = d_scores[k]
            if m.method == "gpc":
                # the classifier's own output is a probability; the latent value behind it is
                # what the ROC and the statistics above are computed from, as for any classifier
                from ..gpc import squash
                scores = squash(scores, np.zeros_like(scores),
                                str(m.params.get("likelihood", "probit")))
                rp.set_scores(ds.names, scores, target, ds.subset, boundary=0.5)
            elif getattr(m.learner_, "kfda_", None) is not None:
                # the discriminant score turned into a probability by the logistic map fitted
                # out of fold; the bias is removed first, because that map is on the raw score
                cal = m.learner_.kfda_
                rp.set_scores(ds.names, cal.probability(scores - float(m.bias)), target,
                              ds.subset, boundary=0.5)
            else:
                rp.set_scores(ds.names, scores, target, ds.subset)
            if "screening_truth" in st.extra:
                rp.set_enrichment(st.extra["screening_truth"], st.extra["screening_scores"],
                                  label="actives vs decoys")
            else:
                rp.set_enrichment(None)
        else:
            rp.set_scores(ds.names, np.full(len(ds), np.nan))
            rp.set_enrichment(None)
        rp.tabs.setCurrentIndex(0)

    def run_nested_cv(self):
        if not self._check_ready():
            return
        fs = self._fieldset_from_panel()
        mp = self.model_panel
        ds = self.ds
        ok = np.isfinite(ds.activity)
        tr = [int(i) for i in ds.train_indices if ok[i]]
        method = mp.method_key()
        alphas = mp.alphas() if mp.rad_alpha_grid.isChecked() else None
        lams, Cs, eps, nus = mp.lams(), mp.Cs(), mp.epsilons(), mp.nus()
        folds, seed = mp.spin_folds.value(), mp.spin_seed.value()
        outer, repeats = mp.spin_outer.value(), mp.spin_repeats.value()
        optimize_h = mp.cmb_mix.currentIndex() == 1
        mkl = mp.mkl_mode()
        layer, gammas = mp.kernel_layer(), mp.gammas()
        normalize, scale = mp.chk_normalize.isChecked(), mp.chk_scale.isChecked()
        mols, names, y = ds.mols, ds.names, ds.activity.copy()

        def task(progress, cancel):
            mf = [build_fields(mols[i], fs, names[i]) for i in tr]
            train = AtomTable.from_molecules(mf)
            template = CMFModel(fieldset=fs, method=method, normalize=normalize, scale_fields=scale,
                                kernel_layer=layer)
            return nested_cv(train, y[tr], template, alphas, lams=lams, Cs=Cs, epsilons=eps, nus=nus,
                             gammas=gammas, outer_folds=outer, inner_folds=folds, repeats=repeats,
                             seed=seed, optimize_h=optimize_h, mkl=mkl,
                             progress=progress, cancel=cancel)

        def done(ncv):
            msg = (f"External {outer}-fold CV ({repeats} repeat(s)): q2ex = {ncv.q2_ex:.3f}"
                   + (f" +/- {ncv.q2_ex_sd:.3f}" if np.isfinite(ncv.q2_ex_sd) else "")
                   + f", RMSEcvex = {ncv.rmse_ex:.3f}"
                   + (f" +/- {ncv.rmse_ex_sd:.3f}" if np.isfinite(ncv.rmse_ex_sd) else ""))
            self.log(msg)
            if self.model is not None:
                self.model.stats.q2_ex, self.model.stats.rmse_ex = ncv.q2_ex, ncv.rmse_ex
                self.model.stats.q2_ex_sd, self.model.stats.rmse_ex_sd = ncv.q2_ex_sd, ncv.rmse_ex_sd
                self._show_results()
            else:
                self.results_panel.txt_stats.setPlainText(msg)
            self.results_panel.set_points("ex", y[tr], ncv.y_pred, [names[i] for i in tr], tr)
            self.results_panel.tabs.setCurrentIndex(1)

        self.run_task(task, done, "External cross-validation")

    def predict_test(self):
        if self.model is None:
            QMessageBox.information(self, "No model", "Build or load a model first.")
            return
        ds = self.ds
        te = [int(i) for i in ds.test_indices]
        if not te:
            QMessageBox.information(self, "No test set", "Mark some molecules as test set (or add a test set file).")
            return
        model, fs = self.model, self.model.fieldset
        mols, names, y = ds.mols, ds.names, ds.activity.copy()

        def task(progress, cancel):
            mf = [build_fields(mols[i], fs, names[i]) for i in te]
            # were these molecules prepared the way the training set was? (cmfqsar/contract.py)
            findings = check_molecules(model, mf)
            if any(f.level == "refuse" for f in findings):
                return None, None, None, findings
            table = AtomTable.from_molecules(mf)
            progress("Predicting", 0.3)
            yp = model.evaluate_test(table, y[te], cancel=cancel)
            var = None
            if model.method in ("krr", "gp"):
                try:
                    _, var = model.predict_with_variance(table)
                except Exception:
                    var = None
            return mf, yp, var, findings

        def done(out):
            mf, yp, var, findings = out
            for f in findings:
                self.log(f"Contract: {f}")
            if mf is None:
                QMessageBox.warning(
                    self, "Prediction refused",
                    "\n\n".join(f.message for f in findings if f.level == "refuse")
                    + "\n\nThe molecules were not predicted. Prepare them as the training set "
                      "was, or build the model again with computed charges.")
                return
            for k, i in enumerate(te):
                self.molfields[i] = mf[k]
            self.model_test_idx = te
            self.model.stats.extra["y_test_pred"] = yp
            self._show_results()
            self.log(f"Predicted {len(te)} test molecules" +
                     (f"; R2pred = {self.model.stats.r2_pred:.3f}, RMSEpred = {self.model.stats.rmse_pred:.3f}"
                      if np.isfinite(self.model.stats.rmse_pred) else ""))
            if var is not None:
                self.log("GP predictive variances: " + ", ".join(f"{names[i]}={v:.3g}" for i, v in zip(te, var)))

        self.run_task(task, done, "Predicting")

    # ------------------------------------------------- heterogeneous series (several chemotypes)
    def _need_chemotypes(self) -> bool:
        if self.ds is None or not self.molfields:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return False
        if self.chemotypes is None:
            QMessageBox.information(self, "No chemotypes",
                                    "Find the chemotypes first: Tools, Heterogeneous sets, "
                                    "step 1. Everything else in this submenu works from that "
                                    "grouping.")
            return False
        return True

    def chemotypes_dialog(self):
        """Step 1: group the molecules by their Bemis-Murcko framework."""
        from ..chemotypes import chemotype_labels, grouped_folds, summarise_groups
        if self.ds is None or not self.ds.mols:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return
        dlg = ChemotypeDialog(len(self.ds.mols), self)
        if dlg.exec() != QDialog.Accepted:
            return
        generic, min_size = dlg.chk_generic.isChecked(), dlg.spin_min.value()
        labels, names = chemotype_labels(self.ds.mols, generic=generic, min_size=min_size)
        self.chemotypes = (labels, names)
        folds = grouped_folds(labels)
        self.log(summarise_groups(labels, names)
                 + f"; leave-one-chemotype-out would use {len(folds)} folds of "
                   f"{[len(f) for f in folds][:8]}{' …' if len(folds) > 8 else ''}")
        for g, nm in enumerate(names):
            n = int(np.sum(labels == g))
            if n:
                self.results_panel.append_log(f"    chemotype {g}: {n:4d} molecules  {nm[:70]}")
        self.dataset_panel.set_group_labels(labels, names)
        self.schedule_display()

    def align_within_chemotypes(self):
        """Step 2: superimpose the molecules of each chemotype on its largest member."""
        from ..chemotypes import align_within_groups
        if not self._need_chemotypes():
            return
        labels = self.chemotypes[0]
        if QMessageBox.question(
                self, "Align within chemotypes",
                "Every molecule is superimposed on the largest molecule of its own chemotype, "
                "where a common substructure exists and does the job well. The molecules move, "
                "so a model built on their present poses is discarded.\n\nContinue?") != \
                QMessageBox.Yes:
            return
        mols = self.ds.mols
        self._remember_poses("Align within chemotypes")

        def task(progress, cancel):
            return align_within_groups(mols, labels, progress=progress, cancel=cancel)

        def done(log):
            bad = [l for l in log if "FAILED" in l]
            self._after_poses_changed()
            self.log(f"Aligned inside {int(labels.max()) + 1} chemotypes: {len(log) - len(bad)} "
                     f"molecules placed, {len(bad)} failed")
            for line in bad[:5]:
                self.results_panel.append_log("    " + line)

        self.run_task(task, done, "Aligning within chemotypes")

    def place_chemotypes_dialog(self):
        """Step 3: move each chemotype, as a rigid body, into one frame."""
        from ..chemotypes import place_groups
        if not self._need_chemotypes():
            return
        labels, names = self.chemotypes
        dlg = PlaceChemotypesDialog(names, self.protein is not None, self)
        if dlg.exec() != QDialog.Accepted:
            return
        mode = dlg.mode()
        if mode == "dock" and self.protein is None:
            QMessageBox.information(self, "No structure", "Placing by docking needs a protein.")
            return
        ref_group, restarts = dlg.cmb_ref.currentIndex(), dlg.spin_restarts.value()
        fs = self._fieldset_from_panel()
        fs.conformers = "single"
        mols, protein = self.ds.mols, self.protein
        self._remember_poses("Place chemotypes in one frame")

        def task(progress, cancel):
            pocket = None
            if mode == "dock":
                from ..complementarity import pocket_fields
                progress("Parameterising the pocket", 0.05)
                pocket = pocket_fields(protein, fs)
            return place_groups(mols, labels, fs, mode=mode, reference_group=ref_group,
                                pocket=pocket, restarts=restarts, progress=progress, cancel=cancel)

        def done(log):
            self._after_poses_changed()
            self.log(f"Placed {int(labels.max()) + 1} chemotypes in one frame by "
                     f"{'docking' if mode == 'dock' else 'field similarity'}")
            for line in log:
                self.results_panel.append_log("    " + line[:140])

        self.run_task(task, done, "Placing chemotypes")

    def chemotype_validation_dialog(self):
        """Step 4: hold out a whole chemotype, and compare with a model that ignores the poses."""
        from ..chemotypes import evaluate
        if not self._need_chemotypes():
            return
        if self.ds is None or not np.isfinite(self.ds.activity).any():
            QMessageBox.information(self, "No activities", "The molecules need activities.")
            return
        labels, names = self.chemotypes
        dlg = ChemotypeValidationDialog(len(names), self)
        if dlg.exec() != QDialog.Accepted:
            return
        mp = self.model_panel
        fs = self._fieldset_from_panel()
        alphas = mp.alphas() or [1.0]
        lams = mp.lams() or [1.0]
        max_folds = dlg.spin_folds.value()
        base_alpha = dlg.spin_alpha.value() or None      # zero: the limit itself
        normalize = mp.chk_normalize.isChecked()
        ds = self.ds

        def task(progress, cancel):
            return evaluate(ds.mols, ds.names, ds.activity, labels, names, fs, alphas=alphas,
                            lams=lams, max_folds=max_folds, normalize=normalize,
                            baseline_alpha=base_alpha, progress=progress, cancel=cancel)

        def done(res):
            self.chemotype_result = res
            rows = [("q² with random folds", res.q2_random),
                    ("q² leave-one-chemotype-out", res.q2_grouped),
                    ("RMSE leave-one-chemotype-out", res.rmse_grouped),
                    ("q² alignment-free fragment model", res.q2_baseline),
                    ("q² alignment-free, geometry-aware", res.q2_invariant),
                    ("alpha", res.alpha), ("lambda", res.lam)]
            per = [(f"{g['name'][:48]}", g["n"], g["rmse"], g["bias"]) for g in res.per_group]
            self.results_panel.set_chemotypes(
                f"{len(res.per_group)} chemotypes held out in turn. Random folds overstate the quality of a mixed "
                f"series, because a molecule is then predicted from its own scaffold-mates. "
                f"Verdict: {res.verdict()}.", rows, per)
            for i in range(self.results_panel.tabs.count()):
                if self.results_panel.tabs.tabText(i) == "Chemotypes":
                    self.results_panel.tabs.setCurrentIndex(i)
            self.log(f"Leave-one-chemotype-out q² {res.q2_grouped:+.3f} against {res.q2_random:+.3f} "
                     f"with random folds and {res.q2_baseline:+.3f} for the alignment-free model: "
                     f"{res.verdict()}")

        self.run_task(task, done, "Leave one chemotype out")

    def _after_poses_changed(self) -> None:
        """Molecules have moved: rebuild what was drawn and drop what no longer applies."""
        self.molfields = [build_fields(m, FieldSet(specs=[], conformers=self.fieldset.conformers), n)
                          for m, n in zip(self.ds.mols, self.ds.names)]
        self._mol_volume_cache.clear()
        self.complementarity = None
        self._comp_surface_cache = {}
        self.shapley_result = self.shapley_fields = None
        self.clear_pharmacophore()
        self.field_reference = self.field_diff_rows = None
        self.controls.chk_diff.setChecked(False)
        self.controls.chk_diff.setEnabled(False)
        if self.model is not None:
            self.model = None
            self.results_panel.clear_contributions(
                "The model was discarded: the molecules it was fitted on have moved.")
        self.schedule_display()

    # ------------------------------------------------------------------ conformations (4D)
    def _conformer_counts(self) -> List[int]:
        return [m.GetNumConformers() for m in self.ds.mols] if self.ds else []

    def _update_conformer_info(self) -> None:
        """Say in the model panel how many conformations the data set carries."""
        counts = self._conformer_counts()
        if not counts:
            self.model_panel.set_conformer_info("")
            return
        many = sum(1 for c in counts if c > 1)
        if many == 0:
            self.model_panel.set_conformer_info("one conformation each")
        else:
            self.model_panel.set_conformer_info(
                f"{many}/{len(counts)} molecules with up to {max(counts)} conformations")

    def generate_conformers_dialog(self):
        if self.ds is None:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return
        counts = self._conformer_counts()
        dlg = ConformerDialog(len(counts), sum(1 for c in counts if c > 1), self,
                              names=list(self.ds.names))
        if dlg.exec() != QDialog.Accepted:
            return
        n, prune = dlg.spin_n.value(), dlg.spin_prune.value()
        window = dlg.spin_window.value()
        optimise, seed = dlg.chk_optimise.isChecked(), dlg.spin_seed.value()
        switch = dlg.chk_switch.isChecked()
        align_to, ref_index = dlg.alignment(), dlg.reference_index()
        fs = self._fieldset_from_panel() if align_to == "field" else None
        mols = list(self.ds.mols)
        self._remember_poses(f"Generate {n} conformations")

        def task(progress, cancel):
            from ..conformers import generate_for_dataset
            return generate_for_dataset(mols, n=n, seed=seed, prune_rmsd=prune,
                                        optimise=optimise, align_to=align_to,
                                        reference_index=ref_index, fieldset=fs,
                                        window_rmsd=window, progress=progress, cancel=cancel)

        def done(out):
            new_mols, failures = out
            self.ds.mols = new_mols
            counts2 = [m.GetNumConformers() for m in new_mols]
            self.molfields = [build_fields(m, FieldSet(specs=[]), nm)
                              for m, nm in zip(new_mols, self.ds.names)]
            if switch:
                self.model_panel.cmb_conformers.setCurrentIndex(1)
            self._update_conformer_info()
            where = {"self": "superimposed on each molecule's own pose",
                     "core": "superimposed on each molecule's own pose over the common scaffold",
                     "reference": "superimposed on the reference molecule of the series",
                     "field": "superimposed on the reference molecule and refined on the field"}
            self.log(f"Generated conformations ({where[align_to]}): {np.mean(counts2):.1f} per "
                     f"molecule on average, up to {max(counts2)}; the kernels of the ensemble "
                     f"mode will cost about "
                     f"{np.mean(counts2) ** 2:.0f} times the single-conformation ones"
                     + (f". {len(failures)} molecules failed and kept their pose" if failures else ""))
            for line in failures[:5]:
                self.results_panel.append_log("  " + line)
            self.schedule_display()

        self.run_task(task, done, "Generating conformations")

    def dock_dialog(self):
        """Place every conformation of every molecule in the loaded pocket."""
        from ..complementarity import pocket_fields
        from ..docking import dock_dataset, summarise
        if self.ds is None or not self.molfields:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return
        if self.protein is None:
            QMessageBox.information(self, "No structure",
                                    "Load a protein structure or fetch one by its PDB code "
                                    "first; the free-energy example sets bring their pocket "
                                    "with them.")
            return
        counts = self._conformer_counts()
        dlg = DockDialog(len(counts), int(sum(counts)), self.protein.name, self)
        if dlg.exec() != QDialog.Accepted:
            return
        restarts, tether = dlg.spin_restarts.value(), dlg.spin_tether.value()
        keep = dlg.spin_keep.value()
        self._remember_poses(f"Dock into {self.protein.name}")
        fs = self._fieldset_from_panel()
        fs.conformers = "single"          # the docking scores one placement at a time
        mols, protein = list(self.ds.mols), self.protein

        def task(progress, cancel):
            progress("Parameterising the pocket", 0.02)
            pocket = pocket_fields(protein, fs)
            out, res = dock_dataset(mols, pocket, fs, restarts=restarts, seed=0, keep_best=keep,
                                    progress=lambda m_, f_: progress(m_, 0.05 + 0.9 * f_),
                                    cancel=cancel, tether_radius=tether)
            return out, res

        def done(out):
            docked, res = out
            self.ds.mols = docked
            self.molfields = [build_fields(m, FieldSet(specs=[], conformers=self.fieldset.conformers), n)
                              for m, n in zip(docked, self.ds.names)]
            self._mol_volume_cache.clear()
            self.complementarity = None
            self._comp_surface_cache = {}
            self.shapley_result = self.shapley_fields = None
            info = summarise(res)
            if self.model is not None:
                self.model = None
                self.results_panel.clear_contributions(
                    "The model was discarded: docking moved the molecules it was fitted on.")
            self._update_conformer_info()
            if info:
                self.log(f"Docked {int(info['poses'])} poses into {protein.name}: score "
                         f"{info['score_before']:+.2f} -> {info['score_after']:+.2f}, clash "
                         f"{info['clash_before']:.1f} -> {info['clash_after']:.1f}, moved "
                         f"{info['moved']:.1f} Å on average. Docking is a worse alignment than "
                         f"superposition on a common substructure; inspect the poses before "
                         f"building on them.")
            self.schedule_display()

        self.run_task(task, done, "Docking")

    # ------------------------------------------------------------------ protein structures
    def _pdb_codes(self) -> List[str]:
        """PDB identifiers carried by the data set, as the benchmark collections store them."""
        if self.ds is None:
            return []
        codes = []
        for m in self.ds.mols:
            for prop in ("pdb", "PDB", "pdb_id", "pdbid"):
                if m is not None and m.HasProp(prop):
                    c = m.GetProp(prop).strip()
                    if c and c not in codes:
                        codes.append(c)
        return codes

    def _install_protein(self, prot, source: str) -> None:
        self.protein = prot
        self._site_cache = None
        self.complementarity = None
        self._comp_surface_cache = {}
        self.controls.set_protein_label(f"{prot.name}, {prot.n_atoms} atoms")
        self.controls.chk_protein.setChecked(True)
        msg = (f"Loaded {prot.name} from {source}: {prot.n_atoms} atoms, {prot.n_residues} residues")
        if self.molfields:
            pts = np.concatenate([m.coords for m in self.molfields])
            d = contact_distance(prot, pts)
            if not np.isfinite(d):
                pass
            elif d > 10.0:
                msg += (f". The nearest protein atom is {d:.1f} Å from any molecule, so the two are "
                        f"not in the same frame: the structure belongs to a different series, or the "
                        f"molecules were realigned and left the crystal frame.")
            else:
                msg += f". Closest approach to the molecules: {d:.2f} Å"
        self.log(msg)
        self.schedule_display()

    def _autoload_pocket(self, dataset_path: str) -> None:
        """Load a stored binding site that belongs to this data set, if there is one.

        The example collections keep trimmed pockets in examples/pockets, already in the frame of
        the ligands, so a free-energy set opens with its protein without any further step.
        """
        codes = self._pdb_codes()
        try:
            path = find_pocket(dataset_path, codes[0] if codes else None)
            if not path:
                return
            prot = read_structure(path, keep_waters=True)
        except Exception as exc:                       # noqa: BLE001 - a missing pocket is not an error
            self.log(f"Could not read the stored pocket: {exc}")
            return
        self._install_protein(prot, os.path.basename(path) + " (stored with the example)")

    def load_protein_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load protein structure", "",
                                              "Structures (*.pdb *.ent *.cif *.mmcif);;All files (*)")
        if not path:
            return
        try:
            prot = read_structure(path, keep_waters=True)
        except Exception as exc:                       # noqa: BLE001 - shown to the user
            self.error(f"Could not read {os.path.basename(path)}: {exc}")
            return
        self._install_protein(prot, os.path.basename(path))

    def fetch_protein(self):
        codes = self._pdb_codes()
        start = codes[0] if codes else ""
        hint = ("The data set names " + ", ".join(codes[:4]) + "." if codes else
                "This data set carries no PDB identifier.")
        code, ok = QInputDialog.getText(self, "Fetch protein from the PDB",
                                        f"{hint}\nIdentifier to download from RCSB:", text=start)
        if not ok or not code.strip():
            return
        code = code.strip()

        def task(progress, cancel):
            progress(f"Downloading {code.upper()} from RCSB", 0.3)
            path = fetch_pdb(code)
            progress("Reading the structure", 0.7)
            return read_structure(path, keep_waters=True), path

        def done(out):
            prot, path = out
            self._install_protein(prot, path)

        self.run_task(task, done, f"Fetching {code.upper()}")

    def clear_protein(self):
        self.protein = None
        self._site_cache = None
        self.complementarity = None
        self._comp_surface_cache = {}
        self.controls.set_protein_label("no structure loaded")
        self.controls.chk_protein.setChecked(False)
        self.viewer.clear_group("protein")
        self.log("Protein structure forgotten")
        self.schedule_display()

    def _binding_site(self, points: np.ndarray):
        """The site around `points`, recomputed only when the request changes."""
        vc = self.controls
        key = (id(self.protein), round(float(vc.spin_site.value()), 2), vc.chk_waters.isChecked(),
               vc.chk_site_h.isChecked(), len(points), float(points.sum()) if len(points) else 0.0)
        if self._site_cache is not None and self._site_cache[0] == key:
            return self._site_cache[1]
        site = binding_site(self.protein, points, cutoff=vc.spin_site.value(),
                            keep_waters=vc.chk_waters.isChecked())
        if not vc.chk_site_h.isChecked() and site.n_atoms:
            keep = [i for i, e in enumerate(site.elements) if e.upper() not in ("H", "D")]
            site = site.subset(keep)
        self._site_cache = (key, site)
        return site

    # ------------------------------------------------------------------ complementarity
    def compute_complementarity(self):
        """Score every molecule against the pocket, and the co-fields against what it offers."""
        from ..complementarity import (cofield_agreement, occupancy, pocket_fields, series_scores)
        if self.protein is None:
            QMessageBox.information(self, "No structure",
                                    "Load a protein structure or fetch one by its PDB code first; "
                                    "the free-energy example sets bring their pocket with them.")
            return
        if self.ds is None or not self.molfields:
            return
        rp = self.results_panel
        fs = self._display_fieldset().copy()
        want_surface = rp.chk_comp_surface.isChecked()
        want_residues = rp.chk_comp_residues.isChecked()
        stat = rp.complement_statistic()
        ds, model = self.ds, self.model
        idx = [i for i in range(len(ds)) if np.isfinite(ds.activity[i])] or list(range(len(ds)))
        names = [ds.names[i] for i in idx]
        mols_rd = [ds.mols[i] for i in idx]
        activity = ds.activity[idx]
        protein = self.protein
        grid = self.grid

        def task(progress, cancel):
            progress("Parameterising the pocket", 0.05)
            pocket = pocket_fields(protein, fs)
            mf = [build_fields(m, fs, n) for m, n in zip(mols_rd, names)]
            res, stats = series_scores(mf, pocket, fs, activity=activity, surface=want_surface,
                                       progress=lambda m_, f_: progress(m_, 0.1 + 0.8 * f_))
            agree = {}
            if model is not None and model.is_linear:
                progress("Co-fields against the pocket", 0.95)
                g = grid if grid is not None else make_grid(model.train.coords, margin=3.0,
                                                            spacing=0.6)
                w = occupancy(model.train.molecules, g).reshape(-1)
                agree = cofield_agreement(model, pocket, g, w, statistic=stat)
            profiles, order, conserved = [], [], None
            if want_residues:
                from ..complementarity import (consensus_profile, interaction_profile,
                                               pocket_residues)
                progress("Resolving the scores by residue", 0.97)
                labels, order = pocket_residues(protein, pocket)
                profiles = [interaction_profile(m, pocket, fs, labels, order) for m in mf]
                if profiles:
                    freq, _strength = consensus_profile(profiles)
                    conserved = freq.reshape(len(order), len(profiles[0].labels))
            return pocket, mf, res, stats, agree, profiles, order, conserved

        def done(out):
            pocket, mf, res, stats, agree, profiles, order, conserved = out
            self.complementarity = {"pocket": pocket, "fields": fs, "by_index": dict(zip(idx, res)),
                                    "mols": dict(zip(idx, mf)), "stats": stats, "agreement": agree,
                                    "surface": {}, "profiles": dict(zip(idx, profiles)),
                                    "residues": list(order), "conserved": conserved}
            self._comp_surface_cache = {}
            self._show_complementarity()
            if want_surface:
                self.controls.chk_comp_map.setChecked(True)
            n_clash = sum(1 for c in res if c.clashes)
            self.log(f"Complementarity with {protein.name}: mean score "
                     f"{np.nanmean([c.score for c in res]):+.3f} over {len(res)} molecules"
                     + (f", {n_clash} of them clashing with the structure" if n_clash else "")
                     + (f"; score against activity r = {stats['r_score']:+.3f}"
                        if "r_score" in stats and np.isfinite(stats["r_score"]) else ""))
            if profiles and conserved is not None:
                cols = list(profiles[0].labels)
                top = sorted(((float(conserved[i, k]), order[i], cols[k])
                              for i in range(len(order)) for k in range(len(cols))
                              if conserved[i, k] > 0.5), reverse=True)[:8]
                self.log("  conserved in more than half of the series: "
                         + (", ".join(f"{r} {c} ({f:.0%})" for f, r, c in top) or "none"))
            self.schedule_display()

        self.run_task(task, done, "Complementarity")

    def _show_complementarity(self):
        """Fill the complementarity tab for the molecule selected now."""
        comp = self.complementarity
        rp = self.results_panel
        if not comp:
            return
        cur = self.dataset_panel.current_index()
        here = comp["by_index"].get(cur)
        every = list(comp["by_index"].values())
        keys = list(every[0].terms) if every else []
        rows = []
        for k in keys:
            mean = float(np.nanmean([c.terms.get(k, np.nan) for c in every]))
            rows.append((k, here.terms.get(k) if here else None, mean))
        rows.append(("total score", here.score if here else None,
                     float(np.nanmean([c.score for c in every]))))
        rows.append(("burial", here.burial if here else None,
                     float(np.nanmean([c.burial for c in every]))))
        if here is not None and here.surface:
            for k, v in here.surface.items():
                mean = float(np.nanmean([c.surface.get(k, np.nan) for c in every if c.surface]))
                rows.append((f"surface {k}", v, mean))
        if here is not None:
            rows.append(("clashes with the structure", float(here.clashes),
                         float(np.mean([c.clashes for c in every]))))
        series = []
        for k, v in sorted(comp["stats"].items()):
            series.append((k.replace("r_", "activity vs ").replace("r2_", "activity vs r2 "), v))
        for k, v in comp["agreement"].items():
            series.append((f"co-field agreement, {k}", v))
        if not comp["agreement"]:
            series.append(("co-field agreement", "needs a linear model"))
        name = self.ds.names[cur] if cur >= 0 else "no molecule selected"
        head = (f"Fields of {len(every)} molecules against {comp['pocket'].name}, "
                f"{comp['pocket'].n_atoms} atoms. Each term is a cosine in [-1, 1] from exact "
                f"overlap integrals, with the electrostatic one signed so that positive means "
                f"complementary. Left column: {name}.")
        if here is not None and here.clashes:
            head += (f" This molecule clashes with the structure at {here.clashes} atom pairs, "
                     f"worst overlap {here.worst_clash:.2f} Å, so read its scores with that in mind.")
        rp.set_complementarity(head, rows, series)
        self._show_residue_profile(cur)

    def _show_residue_profile(self, cur: int) -> None:
        """The selected molecule's score, residue by residue, with what the series shares."""
        comp = self.complementarity
        rp = self.results_panel
        prof = (comp.get("profiles") or {}).get(cur)
        if prof is None:
            rp.set_residue_profile(
                "By residue: tick 'by residue' and press Compute, then select a molecule."
                if not (comp.get("profiles") or {}) else
                "By residue: this molecule was not scored (no activity).", [], [])
            return
        cols = list(prof.labels)
        conserved = comp.get("conserved")
        order = comp.get("residues") or prof.residues
        terms = np.asarray(prof.terms, float)
        # every Gaussian reaches everywhere, so a non-zero term is not a contact; the distance
        # from the molecule to the residue is, and it is what the command line reports too
        near = np.asarray(prof.contacts, float)
        rows = []
        for i, res in enumerate(prof.residues):
            if not (i < len(near) and np.isfinite(near[i]) and near[i] < CONTACT_DISTANCE):
                continue
            values = [terms[i, k] for k in range(len(cols))]
            share = None
            if conserved is not None and i < conserved.shape[0]:
                share = float(np.max(conserved[i]))
            rows.append((res, values, share))
        rows.sort(key=lambda r: -max(abs(v) for v in r[1]))
        kept = float(sum(sum(abs(v) for v in r[1]) for r in rows))
        whole = float(np.abs(terms).sum())
        rp.set_residue_profile(
            f"{prof.name}: {len(rows)} residues within {CONTACT_DISTANCE:g} Å, of {len(order)} in "
            f"the pocket, carrying {100 * kept / whole:.0f} % of the score; the rest is the tail "
            f"of the Gaussians over the residues behind them. Each column sums over all the "
            f"residues to the matching term above. 'conserved' is the largest share of the series "
            f"that shows any of this residue's interactions.",
            ["Residue"] + cols + ["conserved %"], rows)

    def _residue_contributions(self):
        """{residue: what it contributes to the selected molecule's score}, or {}."""
        comp = self.complementarity
        if not comp:
            return {}
        prof = (comp.get("profiles") or {}).get(self.dataset_panel.current_index())
        if prof is None:
            return {}
        near = np.asarray(prof.contacts, float)
        out = {}
        for i, (res, value) in enumerate(prof.per_residue().items()):
            if not (i < len(near) and np.isfinite(near[i]) and near[i] < CONTACT_DISTANCE):
                continue                  # labelling every residue of the pocket is not a picture
            # the profile always carries the chain, while the labels of a single-chain site drop
            # it; both spellings are offered, and they can only collide when there are two chains,
            # in which case the site keeps the chain and the stripped key is never consulted
            out[res] = value
            out.setdefault(res.split(".")[0], value)
        return out

    def _comp_surface(self, cur: int):
        """Per-vertex complementarity surface of one molecule, computed once."""
        from ..complementarity import surface_complementarity
        if cur in self._comp_surface_cache:
            return self._comp_surface_cache[cur]
        comp = self.complementarity
        mol = comp["mols"].get(cur)
        if mol is None:
            mol = build_fields(self.ds.mols[cur], comp["fields"], self.ds.names[cur])
        out = surface_complementarity(mol, comp["pocket"], comp["fields"], spacing=0.4)
        self._comp_surface_cache[cur] = out
        return out

    # ------------------------------------------------------------------ Shapley values
    def compute_shapley(self):
        """Decompose the prediction for the selected molecule by Shapley values.

        Unlike the co-field and the atom contributions this works for any kernel, so it is the
        interpretation available when a nonlinear layer is used.
        """
        from ..shapley import (atom_field_shapley, atom_shapley, common_core, field_shapley,
                               substituent_groups)
        if self.model is None:
            QMessageBox.information(self, "No model", "Build or load a model first.")
            return
        cur = self.dataset_panel.current_index()
        if cur < 0 or self.ds is None:
            QMessageBox.information(self, "No molecule",
                                    "Select a molecule in the data-set table first.")
            return
        rp = self.results_panel
        kind = rp.shapley_players()
        samples = rp.spin_shap_samples.value()
        want_inter = rp.chk_shap_inter.isChecked()
        model, ds = self.model, self.ds
        mol = self.molfields[cur]
        if mol.weights.shape[1] != model.fieldset.n_fields:
            mol = build_fields(ds.mols[cur], model.fieldset, ds.names[cur])
            self.molfields[cur] = mol
        rdmol = ds.mols[cur]
        refs = [ds.mols[i] for i in self.model_train_idx[:20] if i != cur] or [rdmol]
        name = ds.names[cur]

        def task(progress, cancel):
            if kind == "fields":
                return field_shapley(model, mol, interactions=want_inter,
                                     progress=progress, cancel=cancel), None
            if kind == "atomfields":
                return atom_field_shapley(model, mol, samples=samples, interactions=want_inter,
                                          progress=progress, cancel=cancel), None
            if kind == "groups":
                progress("Common substructure of the training set", 0.05)
                core = common_core(rdmol, refs)
                groups, labels = substituent_groups(rdmol, core)
                if not groups:
                    raise RuntimeError("this molecule is the common scaffold itself: there is "
                                       "nothing outside the core to decompose")
                res = atom_shapley(model, mol, groups=groups, core=core, labels=labels,
                                   samples=samples, interactions=want_inter,
                                   progress=progress, cancel=cancel)
                return res, len(core)
            return atom_shapley(model, mol, samples=samples, interactions=want_inter,
                                progress=progress, cancel=cancel), None

        def done(out):
            res, n_core = out
            self.shapley_result = (cur, res)
            self.shapley_fields = None
            if res.player_fields is not None:
                self.shapley_fields = (cur, res.field_atom_values(mol.n_atoms,
                                                                  model.fieldset.n_fields))
                self.controls.chk_shapfield.setChecked(True)
                self._mol_volume_cache.clear()
            what = {"atoms": "atoms", "groups": "substituents", "fields": "field types",
                    "atomfields": "atoms and field types"}[kind]
            how = ("all coalitions enumerated" if res.exact
                   else f"{samples} permutations sampled")
            base = ("the empty field" if n_core is None
                    else f"the common scaffold of {n_core} atoms")
            head = (f"{name}: the prediction {res.prediction:.3f} is {res.baseline:.3f} for "
                    f"{base} plus the {len(res.labels)} values below, which sum to "
                    f"{res.values.sum():+.3f} ({what}, {how}, {res.n_evaluations} evaluations). "
                    f"The Shapley value is the average contribution of a player over all orders "
                    f"in which the molecule can be assembled.")
            if not model.is_linear:
                head += (f" The '{model.kernel_layer}' kernel layer has no co-field, so this is "
                         f"the decomposition that replaces it.")
            if res.interactions is not None:
                head += (" Interaction indices are the part of the prediction that two players "
                         "produce only together; they vanish for a linear kernel.")
            rp.set_shapley(res, head)
            for i in range(rp.tabs.count()):
                if rp.tabs.tabText(i) == "Shapley":
                    rp.tabs.setCurrentIndex(i)
            self.log(f"Shapley decomposition of {name} over {what}: "
                     f"baseline {res.baseline:.3f}, prediction {res.prediction:.3f}, "
                     f"{res.n_evaluations} coalition evaluations")
            self.schedule_display()

        self.run_task(task, done, "Shapley values")

    # ------------------------------------------------------------------ co-fields
    def compute_cofields(self):
        if self.model is None:
            QMessageBox.information(self, "No model", "Build or load a model first.")
            return
        if not self.model.is_linear:
            QMessageBox.information(
                self, "Nonlinear kernel",
                f"This model uses the '{self.model.kernel_layer}' kernel layer, so it is not linear "
                "in the molecular field and no co-field exists: the regression coefficients of the "
                "field are what a co-field displays, and they are defined only for the plain "
                "overlap-integral kernel.\n\nRebuild the model with the linear kernel to see "
                "co-fields and atom contributions. Predictions of this model can still be "
                "decomposed into the contributions of the training molecules (Contributions tab).")
            return
        model = self.model
        spacing = self.controls.spin_spacing.value()
        margin = self.controls.spin_margin.value()
        names = [s.name for s in model.fieldset.specs]

        def task(progress, cancel):
            # extend the margin to ~2.5 sigma of the broadest Gaussian so surfaces are not cut by the box
            widths = []
            for s in model.fieldset.specs:
                widths.append(s.alpha if s.kind == "gauss" else s.alpha / float(model.train.radii.max()) ** 2)
            auto_margin = 2.5 / np.sqrt(max(min(widths), 1e-3))
            grid = make_grid(model.train.coords, margin=max(margin, auto_margin), spacing=spacing)
            vols = {}
            for k, name in enumerate(names):
                progress(f"Co-field {name} on {grid.shape[0]}x{grid.shape[1]}x{grid.shape[2]} grid", k / len(names))
                vols[name] = cofield_volume(model, k, grid)
                if cancel():
                    break
            return grid, vols

        def done(out):
            self.grid, self.volumes = out
            self.cofield_sd = {}                 # the error bars belong to the old grid
            self._mol_volume_cache.clear()
            self.log(f"Co-fields computed on a {'x'.join(map(str, self.grid.shape))} grid "
                     f"(spacing {self.grid.spacing:.2f} Å)")
            self.controls.chk_cofield.setChecked(True)
            if self.controls.chk_cofield_noise.isChecked():
                self.cofield_noise_toggled(True)
            self.schedule_display()

        self.run_task(task, done, "Computing co-fields")

    def cofield_noise_toggled(self, on: bool):
        """Compute the error bar of every co-field, once, when the user asks to hide the noise.

        It costs about what the co-fields themselves cost, so it is not computed until it is
        wanted; unticking the box keeps what was computed, so the two pictures can be compared
        by clicking back and forth.
        """
        if not on or self.grid is None or not self.volumes or self.cofield_sd:
            self.schedule_display()
            return
        model = self.model
        if model is None:
            return
        try:
            sigma, how = noise_level(model)
        except ValueError as exc:
            QMessageBox.information(self, "No error bar", str(exc))
            self.controls.chk_cofield_noise.setChecked(False)
            return
        grid, names = self.grid, [s.name for s in model.fieldset.specs]

        def task(progress, cancel):
            out = {}
            for k, name in enumerate(names):
                progress(f"Error bar of the {name} co-field", k / len(names))
                out[name] = np.sqrt(np.maximum(
                    cofield_variance(model, k, grid, sigma=sigma), 0.0))
                if cancel():
                    break
            return out

        def done(out):
            self.cofield_sd = out
            self.log(f"Co-field error bars from sigma = {sigma:.3f} ({how}); "
                     f"only what lies more than 2 standard deviations from zero is drawn")
            self.schedule_display()

        self.run_task(task, done, "Co-field error bars")

    def _mol_volume(self, mol_index: int, field_index: int, kind: str) -> Optional[np.ndarray]:
        key = (mol_index, field_index, kind)
        if key in self._mol_volume_cache:
            return self._mol_volume_cache[key]
        mol = self.molfields[mol_index]
        if mol.weights.shape[1] != self.model.fieldset.n_fields:
            mol = build_fields(self.ds.mols[mol_index], self.model.fieldset, self.ds.names[mol_index])
            self.molfields[mol_index] = mol
        if kind == "mol":
            v = molfield_volume(self.model, mol, field_index, self.grid)
        else:
            v = overlap_volume(self.model, mol, field_index, self.grid)
        self._mol_volume_cache[key] = v
        return v

    def _shapley_volume(self, cur: int, fs: FieldSet, f: int):
        """Shapley field of molecule `cur` for field `f`, on the co-field grid or a local one."""
        from ..cofields import shapley_volume
        values = self.shapley_fields[1]
        spacing = self.controls.spin_spacing.value()
        key = ("shapfield", cur, f, id(values), spacing, id(self.grid))
        if key in self._mol_volume_cache:
            return self._mol_volume_cache[key]
        mol = self.molfields[cur]
        spec = fs.specs[f]
        grid = self.grid
        if grid is None:
            width = spec.alpha if spec.kind == "gauss" else spec.alpha / float(mol.radii.max()) ** 2
            grid = make_grid(mol.coords, margin=max(3.0, 2.5 / np.sqrt(max(width, 1e-3))),
                             spacing=spacing)
        vol = shapley_volume(mol, spec, values[f], grid)
        self._mol_volume_cache[key] = (vol, grid)
        return vol, grid

    @staticmethod
    def _atom_labels(mol: MoleculeFields, kind: str) -> List[str]:
        """Per-atom label texts; the numbering is the one used by the contribution tables."""
        if kind == "type":
            return [str(t) for t in mol.atom_types]
        if kind == "charge":
            return [f"{q:+.2f}" for q in mol.charges]
        return [f"{s}{i + 1}" for i, s in enumerate(mol.symbols)]

    def _display_fieldset(self) -> FieldSet:
        """Field set used for drawing: the model's if a model exists, else the current panel settings."""
        if self.model is not None:
            return self.model.fieldset
        fs = self.fieldset.copy()
        fs.specs = [s for s in fs.specs if s.enabled]
        return fs

    @staticmethod
    def _fieldset_signature(fs: FieldSet) -> str:
        return fs.charge_model + "|" + fs.cif_scheme + "|" + ";".join(f"{s.name}:{s.kind}:{s.alpha:g}" for s in fs.specs)

    def _current_mol_volume(self, cur: int, fs: FieldSet, f: int):
        """Molecular field X_f of molecule `cur` on the co-field grid (if any) or on a local grid."""
        from ..cofields import field_volume
        sig = self._fieldset_signature(fs)
        spacing = self.controls.spin_spacing.value()
        key = ("molfield", cur, f, sig, spacing, id(self.grid))
        if key in self._mol_volume_cache:
            return self._mol_volume_cache[key]
        mol = self.molfields[cur]
        if mol.weights.shape[1] != fs.n_fields or getattr(mol, "_sig", None) not in (None, sig):
            mol = build_fields(self.ds.mols[cur], fs, self.ds.names[cur])
            mol._sig = sig
            self.molfields[cur] = mol
        grid = self.grid
        if grid is None:
            spec = fs.specs[f]
            width = spec.alpha if spec.kind == "gauss" else spec.alpha / float(mol.radii.max()) ** 2
            grid = make_grid(mol.coords, margin=max(3.0, 2.5 / np.sqrt(max(width, 1e-3))), spacing=spacing)
        vol = field_volume(mol, fs.specs[f], f, grid)
        self._mol_volume_cache[key] = (vol, grid)
        return vol, grid

    # ------------------------------------------------------------------ display
    def update_display(self):
        v = self.viewer
        for g in ("molecules", "current", "cofield", "molfield", "overlap", "contrib", "protein",
                  "shapfield", "labels", "sitelabels", "compsurf"):
            v.clear_group(g)
        v.set_dark(self.controls.chk_dark.isChecked())
        if self.ds is None or not self.molfields:
            return
        vc = self.controls
        style = vc.style_key()
        show_h = vc.chk_hydrogens.isChecked()
        shown = self.dataset_panel.shown_indices()
        cur = self.dataset_panel.current_index()
        legend = []

        # --- complementarity surface of the selected molecule
        if vc.chk_comp_map.isChecked():
            if not self.complementarity:
                legend.append("complementarity surface: compute it in the Complementarity tab")
            elif cur < 0:
                legend.append("complementarity surface: select a molecule")
            else:
                out = self._comp_surface(cur)
                if out is None:
                    legend.append("complementarity surface: no surface for this molecule")
                else:
                    sv, sf, per_vertex, corr = out
                    v.add_scalar_surface(sv, sf, per_vertex, "compsurf", alpha=vc.opacity() + 0.35)
                    txt = ", ".join(f"{k} {val:+.2f}" for k, val in corr.items()
                                    if np.isfinite(val))
                    legend.append(f"complementarity of {self.ds.names[cur]} with "
                                  f"{self.protein.name}: green complementary, red opposed; "
                                  f"over the surface {txt}")

        # --- protein binding site, drawn behind everything else
        if vc.chk_protein.isChecked():
            if self.protein is None:
                legend.append("protein: load a structure or fetch one by its PDB code (File menu)")
            else:
                idx = shown or ([cur] if cur >= 0 else [])
                pts = (np.concatenate([self.molfields[i].coords for i in idx]) if idx
                       else np.zeros((0, 3)))
                site = self._binding_site(pts)
                if site.n_atoms == 0:
                    d = contact_distance(self.protein, pts)
                    legend.append(f"protein: no residue within {vc.spin_site.value():g} Å of the "
                                  f"molecules shown (nearest atom {d:.1f} Å away)")
                else:
                    v.add_structure(site.coords, site.bonds, site.elements, "protein",
                                    style=vc.protein_style(), alpha=0.9)
                    anchors, res = residue_anchors(site)
                    txt = (f"{self.protein.name}: {len(res)} residues within "
                           f"{vc.spin_site.value():g} Å")
                    contrib = self._residue_contributions()
                    if contrib and vc.chk_site_contrib.isChecked():
                        pos, neg, lp, ln, drawn = [], [], [], [], []
                        for anchor, name in zip(anchors, res):
                            val = contrib.get(name)
                            if val is None or not np.isfinite(val) or val == 0.0:
                                continue
                            (pos if val > 0 else neg).append(anchor)
                            (lp if val > 0 else ln).append(f"{name} {val:+.3f}")
                            drawn.append((name, val))
                        if pos:
                            v.add_labels(np.asarray(pos), lp, "sitecontrib",
                                         color=(0.45, 0.95, 0.55), size=8, offset=0.15)
                        if neg:
                            v.add_labels(np.asarray(neg), ln, "sitecontrib",
                                         color=(1.00, 0.55, 0.55), size=8, offset=0.15)
                        # from what was drawn, not from the dictionary, which carries every
                        # residue under both spellings of its name
                        best = sorted(drawn, key=lambda kv: -abs(kv[1]))[:6]
                        txt += ("; contributions " + ", ".join(f"{k} {x:+.3f}" for k, x in best)
                                if best else "; no residue contributes")
                    elif vc.chk_site_labels.isChecked():
                        # the name sits on the alpha carbon, which is where a chemist looks for it
                        v.add_labels(anchors, res, "sitelabels", color=(0.80, 0.85, 0.95),
                                     size=8, offset=0.15)
                        txt += " = " + ", ".join(res[:40]) + (" …" if len(res) > 40 else "")
                    legend.append(txt)
        for i in shown:
            mol = self._display_mol(i, show_h)
            if i == cur:
                v.add_molecule(mol, "current", style=style)
            else:
                v.add_molecule(mol, "molecules", style=style, alpha=0.35 if len(shown) > 1 else 1.0)
        if cur >= 0 and cur not in shown:
            v.add_molecule(self._display_mol(cur, show_h), "current", style=style)

        # --- atom labels of the selected molecule
        if vc.label_kind() != "none" and cur >= 0:
            # numbered on the full molecule, so the labels match the contribution tables even
            # when hydrogens are hidden
            lmol = self.molfields[cur]
            texts = self._atom_labels(lmol, vc.label_kind())
            keep = [i for i in range(lmol.n_atoms) if show_h or lmol.symbols[i] != "H"]
            v.add_labels(lmol.coords[keep], [texts[i] for i in keep], "labels",
                         color=(1.0, 1.0, 0.85), size=8, offset=0.18)
            legend.append(f"atom labels of {self.ds.names[cur]}: "
                          + {"index": "element and number",
                             "type": "atom type",
                             "charge": "partial charge"}[vc.label_kind()])

        model = self.model
        fname = vc.current_field()
        frac = vc.level_fraction()
        op = vc.opacity()
        fs = self._display_fieldset()
        names = [s.name for s in fs.specs]
        f = names.index(fname) if fname in names else -1
        flabel = fs.specs[f].label if f >= 0 else (fname or "?")
        cur_name = self.ds.names[cur] if cur >= 0 else ""

        # --- co-field (needs a model and computed co-field grids)
        cofield_drawn = False
        if vc.chk_cofield.isChecked():
            if model is None:
                legend.append("co-field: build a model, then click 'Compute co-field'")
            elif fname not in self.volumes:
                legend.append("co-field: click 'Compute co-field'")
            else:
                pos_c, neg_c = cofield_colors(fname)
                vol = self.volumes[fname]
                hiding = vc.chk_cofield_noise.isChecked()
                if hiding and fname in self.cofield_sd:
                    # only what lies more than two standard deviations from zero, the rest
                    # being shape the noise in the activities could have produced on its own
                    sd = self.cofield_sd[fname]
                    kept = np.abs(vol) >= 2.0 * sd
                    drawn = np.abs(vol) >= 0.1 * float(np.abs(vol).max() or 1.0)
                    vol = mask_noise(vol, sd, 2.0)
                    share = float(np.mean(kept[drawn])) if drawn.any() else 0.0
                    legend.append(f"noise hidden: {share:.0%} of the {flabel} co-field stands "
                                  f"more than 2 standard deviations from zero")
                elif hiding:
                    legend.append("noise hidden: the error bars are still being computed")
                amax = float(np.abs(vol).max())
                if amax > 0:
                    mesh = vc.cmb_cofield_style.currentText() == "mesh"
                    for sign, col in ((1, pos_c), (-1, neg_c)):
                        s = isosurface(vol, sign * frac * amax, self.grid)
                        if s is not None:
                            v.add_isosurface(s[0], s[1], col, "cofield", alpha=0.75 if mesh else op,
                                             wireframe=mesh)
                            cofield_drawn = True
                    legend.append(f"co-field {flabel} ({'mesh' if mesh else 'solid'}): positive = {_cname(pos_c)}, "
                                  f"negative = {_cname(neg_c)}, iso = ±{frac * amax:.3g}")
                else:
                    legend.append(f"co-field {flabel} is zero everywhere")

        # --- molecular field of the current molecule (no model needed)
        if vc.chk_molfield.isChecked():
            if cur < 0:
                legend.append("molecular field: select a molecule in the data-set table")
            elif f < 0:
                legend.append("molecular field: choose a field type in the 'Field type' box")
            else:
                vol, grid = self._current_mol_volume(cur, fs, f)
                amax = float(np.abs(vol).max())
                if amax > 0:
                    pos_c, neg_c = cofield_colors(fname)
                    mesh = molfield_wireframe(vc.cmb_molfield_style.currentText(),
                                              cofield_drawn)
                    for sign, col in ((1, pos_c), (-1, neg_c)):
                        s = isosurface(vol, sign * frac * amax, grid)
                        if s is not None:
                            v.add_isosurface(s[0], s[1], col, "molfield", alpha=0.75 if mesh else op,
                                             wireframe=mesh)
                    style = "mesh" if mesh else "solid"
                    legend.append(f"molecular field {flabel} of {cur_name} ({style}): positive = {_cname(pos_c)}, "
                                  f"negative = {_cname(neg_c)}, iso = ±{frac * amax:.3g}")
                else:
                    legend.append(f"{cur_name} has no atoms contributing to the {flabel} field")

        # --- overlap of co-field and molecular field (needs model + co-field grid + molecule)
        if vc.chk_overlap.isChecked():
            if model is None or self.grid is None:
                legend.append("overlap: build a model and click 'Compute co-field'")
            elif cur < 0:
                legend.append("overlap: select a molecule in the data-set table")
            elif fname in [s.name for s in model.fieldset.specs]:
                fm = [s.name for s in model.fieldset.specs].index(fname)
                vol = self._mol_volume(cur, fm, "overlap")
                amax = float(np.abs(vol).max())
                dv = self.grid.spacing ** 3
                if amax > 0:
                    for sign, col in ((1, (0.2, 0.9, 0.3)), (-1, (0.95, 0.3, 0.1))):
                        s = isosurface(vol, sign * frac * amax, self.grid)
                        if s is not None:
                            v.add_isosurface(s[0], s[1], col, "overlap", alpha=op)
                legend.append(f"overlap C·X ({flabel}): positive = green, negative = red-orange; "
                              f"∫ = {vol.sum() * dv:+.3f} (contribution of this field)")

        # --- Shapley field: the attribution of one field type smeared over its own Gaussians
        if vc.chk_shapfield.isChecked():
            if self.shapley_fields is None or cur < 0 or self.shapley_fields[0] != cur:
                legend.append("Shapley field: compute it for the selected molecule in the Shapley "
                              "tab with the players set to 'atoms x field types'")
            elif f < 0:
                legend.append(f"Shapley field: the model has no {flabel} field")
            else:
                vol, sgrid = self._shapley_volume(cur, fs, f)
                amax = float(np.abs(vol).max())
                pos_c, neg_c = cofield_colors(fname)
                mesh = vc.shapfield_style() == "mesh"
                if amax > 0:
                    for sign, col in ((1, pos_c), (-1, neg_c)):
                        srf = isosurface(vol, sign * frac * amax, sgrid)
                        if srf is not None:
                            v.add_isosurface(srf[0], srf[1], col, "shapfield", alpha=op,
                                             wireframe=mesh, normals=srf[2])
                    legend.append(
                        f"Shapley field Φ ({flabel}, {'mesh' if mesh else 'solid'}): positive = "
                        f"{_cname(pos_c)}, negative = {_cname(neg_c)}; "
                        f"∫ = {vol.sum() * sgrid.spacing ** 3:+.3f} "
                        f"(what this field contributes to the prediction for {cur_name})")
                else:
                    legend.append(f"the {flabel} field contributes nothing to {cur_name}")
        shap = None
        if self.shapley_result is not None and self.shapley_result[0] == cur:
            res = self.shapley_result[1]
            if res.kind == "atoms" and cur >= 0:
                shap = res.atom_values(self.molfields[cur].n_atoms)
        if vc.chk_atoms.isChecked() and model is None:
            legend.append("atom contributions: build a model first")
        elif vc.chk_atoms.isChecked() and not model.is_linear and shap is None:
            legend.append(f"atom contributions are not defined for the '{model.kernel_layer}' "
                          "kernel layer; compute Shapley values in the Shapley tab instead")
        # the two decomposition rules are reported for the selected molecule whenever a model
        # exists; the 3D spheres are only the visual form of the atom contributions
        if model is None:
            self.results_panel.clear_contributions("Build or load a model to decompose a prediction.")
        elif cur < 0:
            self.results_panel.clear_contributions(
                "Select a molecule in the data-set table to decompose its prediction.")
        else:
            mol = self.molfields[cur]
            if mol.weights.shape[1] != model.fieldset.n_fields:
                mol = build_fields(self.ds.mols[cur], model.fieldset, self.ds.names[cur])
                self.molfields[cur] = mol
            kind = "decision value" if model.task in ("classification", "oneclass") else "predicted activity"
            try:
                tc = model.training_contributions(mol)
            except ValueError as exc:
                # a one-class score that is not a sum over the training molecules
                self.results_panel.clear_contributions(str(exc))
                self.results_panel.set_atom_contributions([], [], np.zeros(0))
                return
            try:
                # the overlap kernel and the rotation-invariant one both split a prediction
                # over the atoms; the surface and periodic kernels do not
                c = model.atom_contributions(mol) if model.is_linear else None
            except (ValueError, RuntimeError):
                c = None
            value = float((c.sum() if c is not None else tc.sum()) + model.bias)
            if c is None:
                self.results_panel.set_contributions(
                    model.train.names, model.coef, tc,
                    header=(f"{self.ds.names[cur]}: {kind} {value:.3f} = bias {model.bias:.3f} + the "
                            f"sum of the column below (decomposition rule 2 of Baskin & Zhokhova 2019). "
                            f"With the '{model.kernel_layer}' kernel layer the model is not linear in "
                            f"the molecular field, so there is no decomposition into atoms (rule 3) "
                            f"and no co-field. The Shapley tab decomposes the prediction over atoms "
                            f"or substituents for any kernel."))
                self.results_panel.set_atom_contributions([], [], np.zeros(0))
            else:
                self.results_panel.set_contributions(
                    model.train.names, model.coef, tc,
                    header=(f"{self.ds.names[cur]}: {kind} {value:.3f} = bias {model.bias:.3f} + the sums of "
                            f"either column below (decomposition rules 2 and 3 of Baskin & Zhokhova 2019)."))
                self.results_panel.set_atom_contributions(mol.symbols, mol.atom_types, c)
            draw = c if c is not None else shap
            if draw is not None and vc.chk_atoms.isChecked() and len(draw) == mol.n_atoms:
                amax = float(np.abs(draw).max())
                if amax > 0:
                    radii = 0.12 + 0.75 * np.sqrt(np.abs(draw) / amax)
                    cols = [(0.95, 0.2, 0.2) if x > 0 else (0.2, 0.4, 0.95) for x in draw]
                    v.add_spheres(mol.coords, radii, cols, "contrib", alpha=0.5)
                    if c is not None:
                        legend.append(f"atom contributions of {self.ds.names[cur]}: red = positive, "
                                      f"blue = negative; sum + b = {value:.3f} (b = {model.bias:.3f})")
                    else:
                        legend.append(f"Shapley values of {self.ds.names[cur]}: red = positive, "
                                      f"blue = negative; their sum is the prediction minus the "
                                      f"baseline")
        if vc.chk_diff.isChecked() and self.field_reference is not None and cur >= 0 and f >= 0                 and self.field_reference != cur:
            # what the selected molecule fails to reproduce, and what it adds, against a reference
            vol, dgrid = self._difference_volume(cur, fs, f)
            amax = float(np.abs(vol).max())
            if amax > 0:
                pos_c, neg_c = cofield_colors(fname)
                for sign, col in ((1, pos_c), (-1, neg_c)):
                    srf = isosurface(vol, sign * frac * amax, dgrid)
                    if srf is not None:
                        v.add_isosurface(srf[0], srf[1], col, "fielddiff", alpha=op)
                legend.append(f"difference {flabel}: {cur_name} minus "
                              f"{self.ds.names[self.field_reference]}; "
                              f"{_cname(pos_c)} = added, {_cname(neg_c)} = missing, "
                              f"iso = ±{frac * amax:.3g}")
            else:
                legend.append(f"difference {flabel}: the two molecules agree exactly")
        if self.pharmacophore and vc.chk_pharmacophore.isChecked():
            # the discretised co-field: one translucent sphere per feature, coloured by type
            from ..pharmacophore import colour_of
            feats = self.pharmacophore
            v.add_spheres(np.array([f.centre for f in feats]), [f.radius for f in feats],
                          [colour_of(f) for f in feats], "pharmacophore", alpha=0.35)
            kinds = sorted({f.kind for f in feats})
            legend.append(f"pharmacophore ({len(feats)} features): " + ", ".join(kinds))
        if self.complementarity:
            self._show_complementarity()          # the table follows the selected molecule
        vc.lbl_legend.setText("   |   ".join(legend))

    def _display_mol(self, i: int, show_h: bool) -> MoleculeFields:
        mol = self.molfields[i]
        if show_h:
            return mol
        keep = np.array([s != "H" for s in mol.symbols])
        idx_map = -np.ones(len(keep), dtype=int)
        idx_map[keep] = np.arange(keep.sum())
        bonds = np.array([[idx_map[a], idx_map[b]] for a, b in mol.bonds if keep[a] and keep[b]], dtype=int).reshape(-1, 2)
        return MoleculeFields(mol.name, mol.coords[keep], mol.weights[keep], mol.radii[keep],
                              [s for s, k in zip(mol.symbols, keep) if k], [t for t, k in zip(mol.atom_types, keep) if k],
                              bonds, mol.charges[keep])

    def center_current(self):
        cur = self.dataset_panel.current_index()
        if cur >= 0 and self.molfields:
            self.viewer.center_on(self.molfields[cur].coords)

    # ------------------------------------------------------------------ tools
    def variance_align_dialog(self):
        """Align molecules by minimising the posterior variance of the current model."""
        if self.model is None:
            QMessageBox.information(self, "No model", "Build or load a model first: the fitted model "
                                                      "defines the frame molecules are aligned to.")
            return
        if self.ds is None:
            return
        if not self.model.is_linear:
            QMessageBox.information(
                self, "Nonlinear kernel",
                f"The posterior variance is implemented for the linear CMF kernel; this model uses "
                f"the '{self.model.kernel_layer}' layer. Build a linear model (it can use the same "
                "fields and alpha) to align by prediction confidence.")
            return
        te = [int(i) for i in self.ds.test_indices]
        sel = [i for i in self.dataset_panel.selected_indices() if i not in set(self.model_train_idx)]
        dlg = VarianceAlignDialog(len(te), len(sel), self)
        if dlg.exec() != QDialog.Accepted:
            return
        idx = te if dlg.cmb_which.currentIndex() == 0 else sel
        if not idx:
            QMessageBox.information(self, "Nothing to align",
                                    "Select molecules outside the training set, or define a test set.")
            return
        max_rmsd, restarts = dlg.max_rmsd(), dlg.spin_restarts.value()
        flexible, conformers = dlg.flexible(), dlg.conformers()
        slack = dlg.energy_slack()
        self._remember_poses("Align by prediction confidence")
        model, fs = self.model, self.model.fieldset
        mols, names = self.ds.mols, self.ds.names
        if flexible:
            self.log(f"Aligning by confidence over rigid motion and the rotatable bonds"
                     + (f", from {conformers} generated conformations" if conformers else "")
                     + f"; strain allowance {slack:g} kcal/mol")

        def task(progress, cancel):
            mf = [build_fields(mols[i], fs, names[i]) for i in idx]
            report = lambda i, n: progress(f"Aligning {i}/{n}", i / n)   # noqa: E731
            if flexible:
                from ..flexalign import flexible_align
                res = flexible_align(mf, [mols[i] for i in idx], model, max_rmsd=max_rmsd,
                                     restarts=restarts, energy_slack=slack,
                                     conformers=conformers, progress=report, cancel=cancel)
            else:
                from ..varalign import variance_align
                res = variance_align(mf, model, max_rmsd=max_rmsd, restarts=restarts,
                                     progress=report, cancel=cancel)
            return mf, res

        def done(out):
            mf, res = out
            for k, i in enumerate(idx[:len(res)]):
                self.molfields[i] = mf[k]
                conf = self.ds.mols[i].GetConformer()
                for a, xyz in enumerate(mf[k].coords):
                    conf.SetAtomPosition(a, [float(v) for v in xyz])
            self._mol_volume_cache.clear()
            moved = np.array([r.rmsd for r in res])
            dv = np.array([r.variance_before - r.variance_after for r in res])
            if self._pose_reference:
                rec = [float(np.sqrt(((mf[k].coords - self._pose_reference[i]) ** 2).sum(1).mean()))
                       for k, i in enumerate(idx[:len(res)]) if i in self._pose_reference]
                if rec:
                    self.log(f"Recovery against the poses before randomisation: "
                             f"{np.mean(rec):.2f} A on average (max {np.max(rec):.2f}); "
                             f"{sum(1 for v in rec if v < 1.0)}/{len(rec)} within 1 A")
            self.log(f"Aligned {len(res)} molecules by prediction confidence: "
                     f"mean displacement {moved.mean():.2f} A (max {moved.max():.2f}), "
                     f"mean variance drop {dv.mean():.4f}"
                     + (f"; {sum(1 for r in res if r.limited)} at the displacement limit"
                        if any(r.limited for r in res) else ""))
            for r in res:
                self.results_panel.append_log(
                    f"    {r.name:26s} variance {r.variance_before:8.4f} -> {r.variance_after:8.4f}   "
                    f"moved {r.rmsd:5.2f} A   prediction {r.prediction_before:7.3f} -> {r.prediction_after:7.3f}")
            self.results_panel.append_log("    (the variance of an aligned molecule no longer measures "
                                          "its applicability domain)")
            self.results_panel.tabs.setCurrentIndex(4)
            self.schedule_display()

        self.run_task(task, done, "Aligning by prediction confidence")

    def clear_pharmacophore(self) -> None:
        """Take the features off the view and forget them: they belong to one fitted model."""
        self.pharmacophore = None
        box = self.controls.chk_pharmacophore
        box.blockSignals(True)
        box.setChecked(False)
        box.setEnabled(False)
        box.blockSignals(False)

    def topomer_dialog(self):
        """Replace the molecules by their substituents, each posed by the topomer rule."""
        if self.ds is None or len(self.ds.mols) < 2:
            QMessageBox.information(self, "No data", "Load a data set of at least two molecules.")
            return
        dlg = TopomerDialog(len(self.ds.mols), self)
        if dlg.exec() != QDialog.Accepted:
            return
        core, core_text = dlg.core()
        if core is None and dlg.cmb_core.currentIndex() > 1:
            QMessageBox.information(self, "Bad core", "That core could not be read as a molecule.")
            return
        core_mode = "mcs" if dlg.cmb_core.currentIndex() == 1 else "framework"
        torsion = float(dlg.spin_torsion.value())
        spacing = float(dlg.spin_spacing.value())
        ds, mols, names = self.ds, list(self.ds.mols), list(self.ds.names)

        def task(progress, cancel):
            from rdkit import Chem
            from ..topomer import (TopomerSpec, choose_core, cuttable_fraction, series_labels,
                                   topomer_molecule, unique_substituents)
            use = core
            if use is None:
                progress("Looking for the core", 0.05)
                use, how = choose_core(mols, mode=core_mode)
                if use is None:
                    raise ValueError("no scaffold found: give a core yourself")
                progress(f"Core: {how}", 0.08)
            labels = series_labels(mols, use)
            if not labels:
                raise ValueError("the core covers the whole series: nothing to cut off it")
            spec = TopomerSpec(torsion=torsion, spacing=spacing)
            cache, built, kept, lost = {}, [], [], []
            for i, mol in enumerate(mols):
                if cancel is not None and cancel():
                    break
                if i % 5 == 0:
                    progress(f"Posing substituents {i + 1}/{len(mols)}", 0.1 + 0.85 * i / len(mols))
                try:
                    one = topomer_molecule(mol, use, labels, spec, cache)
                except Exception as exc:                  # noqa: BLE001 - one molecule, not the set
                    one, reason = None, str(exc)
                else:
                    reason = "nothing outside the core"
                if one is None:
                    lost.append(f"{names[i]}: {reason}")
                    continue
                one.SetProp("_Name", names[i])
                built.append(one)
                kept.append(i)
            return {"mols": built, "keep": kept, "lost": lost, "core": use, "labels": labels,
                    "share": cuttable_fraction(mols, use),
                    "unique": len(unique_substituents(mols, use))}

        def done(out):
            from rdkit import Chem
            keep, built = out["keep"], out["mols"]
            if len(built) < 2:
                QMessageBox.information(
                    self, "Nothing to pose",
                    "Fewer than two molecules could be cut into substituents. "
                    + ("The core comes off cleanly in only %.0f%% of them: it is fused to the "
                       "rest or attached by something other than a single bond."
                       % (100 * out["share"])))
                return
            smi = Chem.MolToSmiles(out["core"])
            self.ds.mols = built
            self.ds.names = [self.ds.names[i] for i in keep]
            self.ds.activity = self.ds.activity[keep]
            self.ds.subset = self.ds.subset[keep]
            self.topomer_core = smi
            self._after_dataset_change()
            self.dataset_panel.set_note(f"topomer substituents of {smi[:40]}")
            # the sites stand far apart, so the whole series at once is a row of specks: start
            # on one molecule, with the camera on it
            self.dataset_panel.set_all_shown(False)
            self.dataset_panel.set_shown([0], True)
            self.dataset_panel.select_index(0)
            if self.molfields:
                # and the camera on the first attachment site, not on the row as a whole:
                # centred on all of it, a substituent is a speck
                xyz = self.molfields[0].coords
                first = xyz[xyz[:, 0] < 0.5 * spacing]
                self.viewer.center_on(first if len(first) else xyz)
            self.log(f"Topomer substituents of {smi}: {len(out['labels'])} attachment points, "
                     f"{out['unique']} distinct substituents, {len(built)}/{len(keep) + len(out['lost'])} "
                     f"molecules kept; the core comes off cleanly in {out['share']:.0%} of them")
            for line in out["lost"][:5]:
                self.results_panel.append_log("    " + line[:150])
            if out["share"] < 0.9:
                self.results_panel.append_log(
                    "    the core is fused to the rest of many molecules, so what is left of "
                    "them is not a substituent; choose a core joined by single acyclic bonds")
            self.results_panel.append_log(
                f"    the molecules shown are the posed substituents, one region of space per "
                f"attachment point and {spacing:.0f} A between them: a model built now is a "
                f"topomer model, and its co-field lives in that frame. Only the first molecule "
                f"is displayed, since the whole series at once is a row of specks. Reload the "
                f"file for the molecules themselves.")

        self.run_task(task, done, "Building the topomer substituents")

    def field_difference_dialog(self):
        """Compare the selected molecule's fields with those of a reference molecule."""
        cur = self.dataset_panel.current_index()
        if self.ds is None or cur < 0:
            QMessageBox.information(self, "No molecule",
                                    "Select the candidate in the data-set table first.")
            return
        dlg = FieldDifferenceDialog(list(self.ds.names), self.field_reference
                                    if self.field_reference is not None else 0, self)
        if dlg.exec() != QDialog.Accepted:
            return
        ref = dlg.cmb_ref.currentIndex()
        if ref == cur:
            QMessageBox.information(self, "Same molecule",
                                    "The candidate and the reference are the same molecule.")
            return
        self.field_reference = ref
        level, draw = dlg.spin_level.value(), dlg.chk_draw.isChecked()
        fs = self._fieldset_from_panel()
        mols, names = self.ds.mols, self.ds.names

        def task(progress, cancel):
            from ..cofields import make_grid
            from ..fielddiff import difference_features, mismatch, overall_similarity
            progress("Fields", 0.2)
            a = build_fields(mols[cur], fs, names[cur])
            b = build_fields(mols[ref], fs, names[ref])
            grid = self.grid or make_grid(np.vstack([a.coords, b.coords]),
                                          margin=self.controls.spin_margin.value(),
                                          spacing=self.controls.spin_spacing.value())
            progress("Difference map", 0.6)
            rows = mismatch(a, b, fs, grid=grid)
            feats = difference_features(a, b, fs, grid=grid, level=level, max_features=8)
            return rows, feats, overall_similarity(a, b, fs)

        def done(out):
            rows, feats, cos = out
            self.field_diff_rows = rows
            box = self.controls.chk_diff
            box.setEnabled(True)
            box.blockSignals(True)
            box.setChecked(draw)
            box.blockSignals(False)
            self.log(f"{names[cur]} against {names[ref]}: field cosine {cos:+.3f} over all fields")
            for r in rows:
                self.results_panel.append_log("    " + r.describe())
            for f in feats:
                self.results_panel.append_log("    " + f.describe())
            self.schedule_display()

        self.run_task(task, done, "Field difference")

    def _difference_volume(self, cur: int, fs: FieldSet, f: int):
        """X_f of the selected molecule minus X_f of the reference, on a shared grid."""
        from ..cofields import difference_volume, make_grid
        ref = self.field_reference
        sig = self._fieldset_signature(fs)
        spacing = self.controls.spin_spacing.value()
        key = ("diff", cur, ref, f, sig, spacing, id(self.grid))
        if key in self._mol_volume_cache:
            return self._mol_volume_cache[key]
        a = build_fields(self.ds.mols[cur], fs, self.ds.names[cur])
        b = build_fields(self.ds.mols[ref], fs, self.ds.names[ref])
        grid = self.grid or make_grid(np.vstack([a.coords, b.coords]),
                                      margin=self.controls.spin_margin.value(), spacing=spacing)
        out = (difference_volume(a, b, fs.specs[f], f, grid), grid)
        self._mol_volume_cache[key] = out
        return out

    def pharmacophore_dialog(self):
        """Discretise the co-fields of the fitted model into typed features with tolerances."""
        if self.model is None:
            QMessageBox.information(self, "No model", "Build or load a model first.")
            return
        try:
            self.model._require_linear("pharmacophore features")
            self.model._require_fields("pharmacophore features")
        except ValueError as exc:
            QMessageBox.information(self, "Not available", str(exc))
            return
        dlg = PharmacophoreDialog(len(self.model.fieldset.specs), self.protein is not None, self)
        if dlg.exec() != QDialog.Accepted:
            return
        level, keep = dlg.spin_level.value(), dlg.spin_max.value()
        spacing = dlg.spin_spacing.value()
        use_pocket = dlg.chk_pocket.isChecked() and self.protein is not None
        draw = dlg.chk_draw.isChecked()
        model, protein = self.model, self.protein

        def task(progress, cancel):
            from ..cofields import make_grid
            from ..pharmacophore import extract, pocket_exclusions
            grid = make_grid(model.train.coords, margin=3.0, spacing=spacing)
            feats = extract(model, grid=grid, level=level, max_features=keep, spacing=spacing,
                            progress=progress)
            if use_pocket:
                progress("Exclusion volumes from the receptor", 0.95)
                feats = feats + pocket_exclusions(protein, grid)
            return feats

        def done(feats):
            from ..pharmacophore import summary
            self.pharmacophore = feats or None
            box = self.controls.chk_pharmacophore
            box.setEnabled(bool(feats))
            box.blockSignals(True)
            box.setChecked(bool(feats) and draw)
            box.blockSignals(False)
            self.log(f"Pharmacophore from the co-fields (level {level:g}): {summary(feats)}")
            for f in feats:
                self.results_panel.append_log("    " + f.describe())
            self.schedule_display()
            if feats:
                self._offer_pharmacophore_save(feats)

        self.run_task(task, done, "Extracting the pharmacophore")

    def _offer_pharmacophore_save(self, feats):
        """Write the features out, as a table and as pseudo-atoms, if the user wants them."""
        if QMessageBox.question(
                self, "Save the pharmacophore",
                f"{len(feats)} features extracted. Save them as a CSV table and as pseudo-atoms "
                f"in a PDB file, which any viewer can draw beside the molecules?") != \
                QMessageBox.Yes:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save the pharmacophore", "pharmacophore.csv",
                                              "CSV (*.csv)")
        if not path:
            return
        from ..pharmacophore import write_csv, write_pdb
        write_csv(path, feats)
        pdb = os.path.splitext(path)[0] + ".pdb"
        write_pdb(pdb, feats)
        self.log(f"Pharmacophore written to {path} and {pdb}")

    def _show_score_plot(self, cmap, idx, act, colour_by, target, pc_x, pc_y) -> None:
        """Draw the functional principal components themselves, with no landscape behind them."""
        ds = self.ds
        kept = cmap.scores_.shape[1]
        if pc_x >= kept or pc_y >= kept:
            self.log(f"Chemical space map: only {kept} components have non-zero variance; "
                     f"plotting the first two instead.")
            pc_x, pc_y = 0, min(1, kept - 1)
        ratio = cmap.kpca.explained_variance_ratio
        if colour_by == 1 and target is not None:
            values = np.asarray([target[i] for i in idx], dtype=float)
            what, cm = "class", "inferno"
        elif colour_by == 1:
            values, what, cm = None, "no class set", "viridis"
        elif colour_by == 2:
            values, what, cm = None, "uncoloured", "viridis"
        else:
            values, what, cm = act, (ds.activity_prop or "activity"), "viridis"
        self.results_panel.set_scatter(
            cmap.scores(pc_x, pc_y), values, [ds.names[i] for i in idx], idx,
            title=f"kernel PCA score plot ({what})", cmap=cm,
            xlabel=f"PC{pc_x + 1} ({100 * ratio[pc_x]:.1f} % of the kernel variance)",
            ylabel=f"PC{pc_y + 1} ({100 * ratio[pc_y]:.1f} %)")
        self.results_panel.tabs.setCurrentIndex(self.results_panel.tab_map)
        self.log(f"Kernel PCA score plot: {len(idx)} molecules, PC{pc_x + 1} vs PC{pc_y + 1} "
                 f"({100 * (ratio[pc_x] + ratio[pc_y]):.0f} % of the kernel variance between them; "
                 f"{kept} components kept explain {100 * ratio.sum():.0f} %). "
                 f"The projection is linear, so the distances in the plot are the kernel's own.")

    def gtm_map_dialog(self):
        """Chemical space map and activity landscape from the CMF kernels (kernel PCA + GTM)."""
        if self.ds is None:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return
        dlg = GTMDialog(len(self.ds), self)
        if dlg.exec() != QDialog.Accepted:
            return
        fs = self.model.fieldset if self.model is not None else self._fieldset_from_panel()
        if not fs.specs:
            QMessageBox.information(self, "No fields", "Select at least one molecular field.")
            return
        if dlg.chk_small_alpha.isChecked():
            fs = fs.copy()
            fs.set_alpha(dlg.spin_alpha.value())
        ds = self.ds
        ok = np.isfinite(ds.activity)
        idx = ([int(i) for i in ds.train_indices if ok[i]] if dlg.cmb_which.currentIndex() == 0
               else list(range(len(ds))))
        if len(idx) < 8:
            QMessageBox.information(self, "Too few molecules", "A map needs at least eight molecules.")
            return
        grid, comps, rbf = dlg.spin_grid.value(), dlg.spin_comp.value(), dlg.spin_rbf.value()
        mols, names = ds.mols, ds.names
        colour_by = dlg.cmb_colour.currentIndex()
        target = getattr(self, "model_target", None)
        scores_only = dlg.cmb_projection.currentIndex() == 1
        pc_x, pc_y = dlg.spin_pc_x.value() - 1, dlg.spin_pc_y.value() - 1
        if scores_only and pc_x == pc_y:
            QMessageBox.information(self, "Same component twice",
                                    "Choose two different components to plot.")
            return
        if scores_only:
            comps = max(comps, pc_x + 1, pc_y + 1)

        def task(progress, cancel):
            from ..gtm import CMFMap
            mf = [build_fields(mols[i], fs, names[i]) for i in idx]
            cmap = CMFMap(n_components=comps, grid=grid, rbf=rbf, normalize=True)
            if scores_only:
                cmap.fit_pca(AtomTable.from_molecules(mf), fs, progress=progress)
            else:
                cmap.fit(AtomTable.from_molecules(mf), fs, progress=progress)
            return cmap, mf

        def done(out):
            cmap, mf = out
            self.gtm_map = cmap
            self.gtm_map_idx = idx
            act = ds.activity[idx]
            if scores_only:
                self._show_score_plot(cmap, idx, act, colour_by, target, pc_x, pc_y)
                return
            if colour_by == 0:                       # activity
                land = cmap.activity_landscape(act)
                values, title = act, f"activity landscape ({ds.activity_prop or 'activity'})"
                cm = "viridis"
            elif colour_by == 1 and target is not None:   # class
                land = cmap.class_landscape(np.asarray([target[i] for i in idx], dtype=float))
                values = np.asarray([target[i] for i in idx], dtype=float)
                title, cm = "class landscape (fraction of actives)", "inferno"
            else:                                    # data density = applicability domain
                land = cmap.density()
                values, title, cm = None, "data density (applicability domain)", "magma"
            self.results_panel.set_map(cmap.as_image(land), cmap.coords_, values,
                                       [ds.names[i] for i in idx], idx, title=title, cmap=cm)
            self.results_panel.tabs.setCurrentIndex(self.results_panel.tab_map)
            ll = cmap.gtm.log_likelihood_
            self.log(f"Chemical space map: {len(idx)} molecules, {grid}x{grid} nodes, "
                     f"{comps} functional principal components "
                     f"(first two explain {100 * cmap.kpca.explained_variance_ratio[:2].sum():.0f} % "
                     f"of the kernel variance); log-likelihood {ll[0]:.0f} -> {ll[-1]:.0f} "
                     f"in {len(ll)} EM iterations; "
                     f"{int(np.isfinite(land).sum())} of {grid * grid} nodes carry data.")

        self.run_task(task, done, "Building the map")

    def align_decoys_dialog(self):
        """Give every background molecule its best pose with respect to the training set."""
        if self.ds is None:
            return
        dec = [int(i) for i in self.ds.decoy_indices]
        if not dec:
            QMessageBox.information(self, "No decoys",
                                    "Load a background set first with 'Add decoys…'.")
            return
        ok = np.isfinite(self.ds.activity)
        tr = [int(i) for i in self.ds.train_indices if ok[i]]
        if len(tr) < 2:
            QMessageBox.information(self, "No training set", "Define a training set first.")
            return
        dlg = DecoyAlignDialog(len(dec), len(tr), self)
        if dlg.exec() != QDialog.Accepted:
            return
        n, n_conf = dlg.spin_restarts.value(), dlg.spin_conformers.value()
        self._remember_poses(f"Align {len(dec)} decoys to the training set")
        fs = self.model.fieldset if self.model is not None else self._fieldset_from_panel()
        mols, names = self.ds.mols, self.ds.names

        def task(progress, cancel):
            from ..decoys import align_decoys, align_decoys_flexible
            ref = AtomTable.from_molecules([build_fields(mols[i], fs, names[i]) for i in tr])

            def report(i, m):
                progress(f"Aligning decoy {i}/{m}", i / m)

            if n_conf > 1:
                mf, res = align_decoys_flexible([mols[i] for i in dec], [names[i] for i in dec],
                                                ref, fs, n_conformers=n_conf, restarts=n,
                                                progress=report, cancel=cancel)
            else:
                mf = [build_fields(mols[i], fs, names[i]) for i in dec]
                res = align_decoys(mf, ref, fs, restarts=n, progress=report, cancel=cancel)
            return mf, res

        def done(out):
            mf, res = out
            for k, i in enumerate(dec[:len(res)]):
                self.molfields[i] = mf[k]
                conf = self.ds.mols[i].GetConformer()
                for a, xyz in enumerate(mf[k].coords):
                    conf.SetAtomPosition(a, [float(v) for v in xyz])
            self._mol_volume_cache.clear()
            before = np.array([r.similarity_before for r in res])
            after = np.array([r.similarity_after for r in res])
            self.log(f"Aligned {len(res)} decoys: field similarity to the nearest training molecule "
                     f"{before.mean():.3f} -> {after.mean():.3f} on average (best {after.max():.3f}). "
                     f"Rebuild the model to score them in their new poses.")
            for r in res:
                self.results_panel.append_log(
                    f"    {r.name:28s} {r.similarity_before:.3f} -> {r.similarity_after:.3f} "
                    f"(closest: {r.best_match})")
            self.results_panel.tabs.setCurrentIndex(self.results_panel.tabs.count() - 1)
            self.schedule_display()

        self.run_task(task, done, "Aligning decoys")

    def orca_dialog(self):
        """Compute quantum-chemical properties with ORCA: per-atom into fields, per-molecule into descriptors."""
        if self.ds is None:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return
        from .. import orca as qm
        ds = self.ds
        try:
            exe = qm.find_orca()
        except Exception:        # noqa: BLE001  -- not installed is a message, not a crash
            exe = ""
        charges, mults = [], []
        for m in ds.mols:
            q = int(Chem.GetFormalCharge(m))
            # implicit hydrogens count too, so the parity is right whether or not they are drawn
            n_el = sum(a.GetAtomicNum() + a.GetTotalNumHs() for a in m.GetAtoms()) - q
            charges.append(q)
            mults.append(1 if n_el % 2 == 0 else 2)
        groups = [("all molecules", list(range(len(ds.mols)))),
                  ("only those without the properties", qm.missing_properties(ds.mols)),
                  ("selected", self.dataset_panel.selected_indices()),
                  ("training set", [int(i) for i in ds.train_indices])]
        groups = [(lab, idx) for lab, idx in groups if idx]
        base = os.path.dirname(os.path.abspath(ds.path)) if ds.path else os.getcwd()
        dlg = OrcaDialog(exe, groups, charges, mults, self,
                         workdir=os.path.join(base, "orca_work"),
                         has_mpi=bool(exe) and qm.orca_has_mpi(exe))
        for i, (label, _) in enumerate(groups):     # the molecules still to do are the usual wish
            if label.startswith("only those"):
                dlg.cmb_which.setCurrentIndex(i)
                break
        if dlg.exec() != QDialog.Accepted:
            return
        exe = dlg.ed_exe.text().strip()
        if not exe or not os.path.isfile(exe):
            self.error(f"No ORCA program at {exe or '(nothing given)'}. Install ORCA and give the "
                       f"path to orca.exe in the dialog.")
            return
        idx = dlg.indices()
        if not idx:
            return
        cfg = qm.Config(orca=exe, level=dlg.level(), opt=dlg.chk_opt.isChecked(),
                        fukui=dlg.chk_fukui.isChecked(), nprocs=dlg.spin_nprocs.value(),
                        maxcore=dlg.spin_mem.value(), timeout=dlg.timeout(),
                        workdir=dlg.workdir())
        jobs = dlg.spin_jobs.value()
        mols, names = list(ds.mols), list(ds.names)
        self.log(f"ORCA: {len(idx)} molecules at '{cfg.level}', "
                 f"{'optimised' if cfg.opt else 'input'} geometry, "
                 f"Fukui {'on' if cfg.fukui else 'off'}, {jobs} at a time, work in {cfg.workdir}")

        def task(progress, cancel):
            return qm.annotate_molecules(mols, names, cfg, jobs=jobs, indices=idx,
                                         progress=progress, cancel=cancel)

        def done(outcomes):
            good = [oc for oc in outcomes if oc.mol is not None]
            for oc in good:
                ds.mols[oc.index - 1] = oc.mol
            fresh = [oc for oc in good if not oc.reused]
            failed = [oc for oc in outcomes if oc.mol is None and oc.error != "cancelled"]
            stopped = [oc for oc in outcomes if oc.error == "cancelled"]
            from ..dataio import common_atom_properties
            from ..descriptors import available as numeric_desc
            ds.atom_properties = common_atom_properties(ds.mols)
            ds.properties = sorted({k for m in ds.mols for k in m.GetPropNames()})
            self.model_panel.set_class_properties(list(ds.properties))
            self.model_panel.set_atom_properties(ds.atom_properties)
            self.model_panel.set_descriptor_properties(
                [p for p in numeric_desc(ds.mols) if p != ds.activity_prop])
            props = numeric_properties(ds.mols)
            cmb = self.dataset_panel.cmb_activity
            cmb.blockSignals(True)
            cmb.clear()
            cmb.addItems(props)
            if ds.activity_prop in props:
                cmb.setCurrentIndex(props.index(ds.activity_prop))
            cmb.blockSignals(False)
            self.molfields = [build_fields(m, FieldSet(specs=[]), n)
                              for m, n in zip(ds.mols, ds.names)]
            self._mol_volume_cache.clear()
            self.refresh_fieldset()
            self.schedule_display()
            parts = [f"{len(fresh)} computed in {_hms(sum(oc.seconds for oc in fresh))} of job time"]
            if len(good) > len(fresh):
                parts.append(f"{len(good) - len(fresh)} reused from {cfg.workdir}")
            if failed:
                parts.append(f"{len(failed)} failed and were left as they were")
            if stopped:
                parts.append(f"{len(stopped)} not started (cancelled)")
            self.log("ORCA: " + ", ".join(parts))
            for oc in failed[:5]:
                self.results_panel.append_log(f"  {oc.name}: {oc.error}")
            if ds.atom_properties:
                self.log("Per-atom property fields now on offer: " + ", ".join(ds.atom_properties))
                self.log("Per-molecule quantities now on offer as descriptors: "
                         + ", ".join(p for p in (qm.MOL_PROPS_N + qm.MOL_PROPS_FUKUI)
                                     if p in ds.properties))
            if cfg.opt and good:
                self.log("The geometries were optimised, so the poses have moved: align the set "
                         "again before building a model.")
            if good and QMessageBox.question(
                    self, "Save the annotated data set?",
                    f"{len(good)} molecules now carry the quantum-chemical properties. Written to "
                    f"an SDF they keep them: opening that file later brings the fields and the "
                    f"descriptors back without running ORCA again.",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes) == QMessageBox.Yes:
                self.save_annotated_sdf()

        self.run_task(task, done, "Running ORCA")

    def save_annotated_sdf(self):
        """Write the molecules with every property they carry, atom property lists included."""
        if self.ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save annotated data set", "dataset_qm.sdf",
                                              "SDF (*.sdf)")
        if not path:
            return
        w = Chem.SDWriter(path)
        for m in self.ds.mols:
            w.write(m)
        w.close()
        self.log(f"Annotated data set written to {path}")

    def selectivity_dialog(self):
        """Refit the current model as two endpoints and offer their difference as a co-field."""
        if self.model is None or not self.model.uses_fields:
            QMessageBox.information(self, "No model",
                                    "Build a field model first: the two endpoints are fitted on "
                                    "its fields, attenuation and kernel.")
            return
        if not self.model.is_linear:
            QMessageBox.information(
                self, "Nonlinear kernel",
                f"This model uses the '{self.model.kernel_layer}' kernel layer, so it is not "
                "linear in the molecular field. A difference of two such models has no co-field; "
                "rebuild with the linear kernel.")
            return
        from ..descriptors import available as numeric_props
        from ..multitask import (Observations, cv_predict, evidence_rho, fit as mt_fit,
                                 fit_covariance, score_by_task, search)
        ds = self.ds
        props = [p for p in numeric_props(ds.mols)]
        if len(props) < 2:
            QMessageBox.information(
                self, "Only one endpoint",
                "Two numeric molecule properties are needed, one per endpoint. Import the second "
                "with File / Import activities from CSV, or add it to the file.\n\n"
                "For a separation factor the two endpoints are the two affinities, not the ratio: "
                "a model of the ratio is already the difference, but it cannot say which of the "
                "two affinities a substituent moved.")
            return
        dlg = SelectivityDialog(props, self.ds.activity_prop or props[0], self)
        if dlg.exec() != QDialog.Accepted:
            return
        names = dlg.endpoints()
        if len(names) < 2:
            QMessageBox.information(self, "Two endpoints at least",
                                    "Name two different properties for the difference, and tick "
                                    "any further endpoints to fit alongside them.")
            return
        name_a, name_b = names[0], names[1]
        idx = [int(i) for i in self.model_train_idx]
        Y = np.full((len(idx), len(names)), np.nan)
        for row, i in enumerate(idx):
            for col, prop in enumerate(names):
                m = ds.mols[i]
                if m is not None and m.HasProp(prop):
                    try:
                        Y[row, col] = float(m.GetProp(prop))
                    except ValueError:
                        pass
        obs = Observations.from_columns(Y, names=names)
        counts = obs.counts()
        if min(counts) < 5:
            thin = ", ".join(f"{n}: {c}" for n, c in zip(names, counts) if c < 5)
            QMessageBox.information(self, "Too few measurements",
                                    f"{thin}. Each endpoint needs at least five molecules.")
            return
        model = self.model
        K = model.joint_kernel(model.K_fields_train,
                               model.train.descriptors if model.desc_train_ is not None else None)
        rhos, lams, folds = dlg.rhos(), dlg.lams(), dlg.spin_folds.value()
        how = dlg.covariance()
        self.log("Endpoints on the current kernel: "
                 + ", ".join(f"{n} ({c} molecules)" for n, c in zip(names, counts))
                 + f"; task covariance by '{how}'")

        def task(progress, cancel):
            if how == "cv":
                progress("Searching the task correlation", 0.2)
                rho, lam, _score, _per = search(K, obs, rhos=rhos, lams=lams, folds=folds, seed=0)
                cov = None
            else:
                progress("Choosing the task covariance by the evidence", 0.2)
                if how == "evidence":
                    cov = evidence_rho(K, obs, lams=lams)
                else:
                    cov = fit_covariance(K, obs, rank=1 if how == "free1" else 2, lams=(0.1,))
                rho, lam = float("nan"), cov.lam
            if cancel():
                return None
            progress("Scoring every endpoint out of fold", 0.5)
            T = None if cov is None else cov.T
            pred = cv_predict(K, obs, rho=0.0 if cov is not None else rho, lam=lam,
                              folds=folds, seed=0, T=T)
            per = score_by_task(obs, pred)
            progress("Fitting every endpoint", 0.9)
            f = mt_fit(K, obs, rho=0.0 if cov is not None else rho, lam=lam, T=T)
            return rho, lam, per, f, cov

        def done(out):
            if out is None:
                return
            rho, lam, per, f, cov = out
            self.multitask = {"fit": f, "names": tuple(names), "pair": (0, 1),
                              "base_coef": np.array(model.coef, copy=True),
                              "base_bias": float(model.bias), "rho": rho, "lam": lam}
            if cov is None:
                self.log(f"{len(names)}-endpoint model: task correlation {rho:g}, "
                         f"lambda {lam:g}")
                if rho == 0:
                    self.log("    the correlation chosen is 0, so the endpoints were modelled "
                             "separately; the difference co-field is still exact")
            else:
                for line in cov.describe(names).splitlines():
                    self.log("    " + line)
            self.log(f"    mean q2 over the endpoints "
                     f"{float(np.nanmean([p['q2'] for p in per])):+.3f}")
            for p in per:
                self.log(f"    {p['task']}: n {p['n']}, q2 {p['q2']:+.3f}, "
                         f"RMSEcv {p['rmse_cv']:.3f}")
            items = [(str(n), ("task", t)) for t, n in enumerate(names)]
            items.append((f"{name_a} − {name_b} (selectivity)", ("difference", (0, 1))))
            self.controls.set_endpoints(items)
            self.controls.cmb_endpoint.setCurrentIndex(len(items) - 1)   # the selectivity
            self.endpoint_changed()

        self.run_task(task, done, f"Fitting {len(names)} endpoints")

    def endpoint_changed(self):
        """Point the co-field machinery at one endpoint, or at the difference of the two."""
        mt = getattr(self, "multitask", None)
        if not mt or self.model is None:
            return
        kind = self.controls.current_endpoint()
        f, names = mt["fit"], tuple(mt["names"])
        if kind is None:
            self.model.coef, self.model.bias = mt["base_coef"], mt["base_bias"]
            label = "the single-endpoint model"
        elif kind[0] == "task":
            t = int(kind[1])
            self.model.coef = f.task_coefficients(t)
            self.model.bias = float(f.bias[t])
            label = names[t] if t < len(names) else f"task{t}"
        else:
            pair = kind[1]
            i, j = (int(pair[0]), int(pair[1])) if isinstance(pair, (tuple, list)) else (0, 1)
            self.model.coef = f.selectivity_coefficients(i, j)
            self.model.bias = float(f.bias[i] - f.bias[j])
            label = f"{names[i]} - {names[j]} (selectivity)"
        self.endpoint_label = label
        if self.worker is not None and self.worker.isRunning():
            # the coefficients are already swapped, so the volumes on screen would be
            # the previous endpoint's; say so rather than clearing them and computing
            # nothing, which would leave an empty view and no explanation
            self.log(f"Co-field set to {label}; a computation is running, so press "
                     f"'Compute co-field' when it finishes")
            return
        self.volumes.clear()
        self._mol_volume_cache.clear()
        self.log(f"Co-field now shows: {label}")
        self.compute_cofields()
    def unalign_dialog(self):
        """Give the chosen molecules a random pose (control experiment for alignment methods)."""
        if self.ds is None:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return
        ds = self.ds
        te = [int(i) for i in ds.test_indices]
        tr = [int(i) for i in ds.train_indices]
        sel = self.dataset_panel.selected_indices()
        dlg = UnalignDialog(len(ds), len(tr), len(te), len(sel), self)
        if dlg.exec() != QDialog.Accepted:
            return
        idx = [te, sel, tr, list(range(len(ds)))][dlg.cmb_which.currentIndex()]
        if not idx:
            QMessageBox.information(self, "Nothing to randomise", "That group is empty.")
            return
        from ..align import unalign
        fs = self.model.fieldset if self.model is not None else self._display_fieldset()
        self._remember_poses(f"Unalign {len(idx)} molecules")
        if self._pose_reference is None:
            self._pose_reference = {}
        mf = []
        for i in idx:
            mol = self.molfields[i]
            if mol.weights.shape[1] != fs.n_fields:
                mol = build_fields(ds.mols[i], fs, ds.names[i])
                self.molfields[i] = mol
            self._pose_reference.setdefault(i, mol.coords.copy())
            mf.append(mol)
        moved = unalign(mf, max_rotation=dlg.max_rotation(), max_translation=dlg.spin_shift.value(),
                        seed=dlg.spin_seed.value())
        for i, mol in zip(idx, mf):
            conf = ds.mols[i].GetConformer()
            for a, xyz in enumerate(mol.coords):
                conf.SetAtomPosition(a, [float(v) for v in xyz])
        self._mol_volume_cache.clear()
        touched_training = bool(set(idx) & set(self.model_train_idx))
        if touched_training and self.model is not None:
            self.model = None
            self.volumes.clear()
            self.grid = None
            self.dataset_panel.clear_predictions()
            self.results_panel.clear_points()
        rot = "random orientation" if dlg.max_rotation() is None else f"up to {dlg.max_rotation():g} deg"
        self.log(f"Randomised the poses of {len(idx)} molecules ({rot}, up to "
                 f"{dlg.spin_shift.value():g} A): displacement {np.mean(moved):.2f} A on average "
                 f"(max {np.max(moved):.2f})."
                 + (" The model was discarded because training molecules were moved."
                    if touched_training else " The original poses are remembered for the recovery report."))
        self.schedule_display()

    def train_align_dialog(self):
        """Self-consistent alignment of the training set by the posterior-variance criterion."""
        if not self._check_ready():
            return
        fs = self._fieldset_from_panel() if self.model is None else self.model.fieldset.copy()
        if not fs.specs:
            QMessageBox.information(self, "No fields", "Select at least one molecular field.")
            return
        ds = self.ds
        ok = np.isfinite(ds.activity)
        tr = [int(i) for i in ds.train_indices if ok[i]]
        dlg = TrainAlignDialog(len(tr), self)
        if dlg.exec() != QDialog.Accepted:
            return
        rounds, folds = dlg.spin_rounds.value(), dlg.spin_folds.value()
        max_rmsd, restarts = dlg.max_rmsd(), dlg.spin_restarts.value()
        self._remember_poses("Align training set by prediction confidence")
        mp = self.model_panel
        params = dict(self.model.params) if self.model is not None else {"lam": 1.0}
        method = self.model.method if self.model is not None else mp.method_key()
        normalize = self.model.normalize if self.model is not None else mp.chk_normalize.isChecked()
        scale = self.model.scale_fields if self.model is not None else mp.chk_scale.isChecked()
        mols, names, y = ds.mols, ds.names, ds.activity.copy()

        def task(progress, cancel):
            from ..varalign import cv_align_training_set
            from ..model import CMFModel
            mf = [build_fields(mols[i], fs, names[i]) for i in tr]
            template = CMFModel(fieldset=fs, method=method, params=params,
                                normalize=normalize, scale_fields=scale)
            rows = cv_align_training_set(mf, y[tr], template, folds=folds, rounds=rounds,
                                         max_rmsd=max_rmsd, restarts=restarts,
                                         progress=progress, cancel=cancel)
            return mf, rows

        def done(out):
            mf, rows = out
            for k, i in enumerate(tr):
                self.molfields[i] = mf[k]
                conf = self.ds.mols[i].GetConformer()
                for a, xyz in enumerate(mf[k].coords):
                    conf.SetAtomPosition(a, [float(v) for v in xyz])
            self._mol_volume_cache.clear()
            self.volumes.clear()
            self.grid = None
            self.model = None                      # the old model refers to the old poses
            self.dataset_panel.clear_predictions()
            self.results_panel.clear_points()
            for r in rows:
                self.results_panel.append_log("    " + r.describe())
            last = rows[-1] if rows else None
            self.log(f"Training set realigned in {len(rows)} round(s)"
                     + (f": total drift {last.drift:.2f} A, mean mutual similarity {last.similarity:.4f}"
                        if last else "")
                     + ". Rebuild the model, and judge the result on a test set that was not realigned "
                       "(q2 is computed on the geometry that has just been changed).")
            self.results_panel.tabs.setCurrentIndex(4)
            self.schedule_display()

        self.run_task(task, done, "Aligning the training set")

    def align_dataset_dialog(self):
        if self.ds is None:
            return
        dlg = AlignDialog(self.ds.names, self.dataset_panel.current_index(), self)
        if dlg.exec() != QDialog.Accepted:
            return
        ref = dlg.cmb_ref.currentIndex()
        use_mcs, refine = dlg.chk_mcs.isChecked(), dlg.chk_field.isChecked()
        fs = self._fieldset_from_panel()
        mols = self.ds.mols
        self._remember_poses(f"Align to {self.ds.names[ref]}")

        def task(progress, cancel):
            return align_dataset(mols, ref, use_mcs=use_mcs, field_refine=refine, fieldset=fs,
                                 progress=lambda i, n: progress(f"Aligning {i}/{n}", i / n))

        def done(log):
            for i, line in enumerate(log):
                self.results_panel.append_log(f"  {self.ds.names[i]}: {line}")
            n_fail = sum(1 for l in log if "FAILED" in l)
            self.log(f"Alignment finished ({n_fail} failures). Rebuild the model to use the new coordinates.")
            self._after_dataset_change()
            self.results_panel.tabs.setCurrentIndex(4)

        self.run_task(task, done, "Aligning")

    # ------------------------------------------------------------------ projects (plan item G2)
    def _remember_poses(self, label: str) -> None:
        """Keep the poses before a step that moves molecules, so that the step can be undone."""
        if self.ds is None:
            return
        self.history.remember(self.ds.mols, label)
        self._refresh_undo_action()

    def _refresh_undo_action(self) -> None:
        lab = self.history.last_label
        self.act_undo.setEnabled(lab is not None)
        self.act_undo.setText("Undo last alignment step" + (f" ({lab})" if lab else ""))

    def undo_alignment(self):
        """Put the molecules back where they were before the last step that moved them."""
        if self.ds is None or not len(self.history):
            QMessageBox.information(self, "Nothing to undo",
                                    "No step has moved the molecules in this session.")
            return
        try:
            label, moved = self.history.undo(self.ds.mols)
        except ValueError as exc:
            self.error(f"Could not undo: {exc}")
            return
        self._pose_reference = None
        self._after_poses_changed()          # rebuilds the fields and drops the model
        self._update_conformer_info()
        self._refresh_undo_action()
        ok = np.isfinite(moved)
        n = int((moved[ok] > 1e-6).sum())
        msg = f"Undid '{label}': {n} molecules moved back"
        if ok.any():
            msg += f", {np.nanmean(moved[ok]):.2f} A on average (max {np.nanmax(moved[ok]):.2f})"
        msg += (f"; {len(self.history)} earlier step(s) can still be undone."
                if len(self.history) else ".")
        self.log(msg)

    def _session_settings(self) -> dict:
        return {"model_panel": capture_settings(self.model_panel),
                "dataset_panel": capture_settings(self.dataset_panel),
                "controls": capture_settings(self.controls),
                "results_full_width": bool(self.act_full_width.isChecked())}

    def _apply_session_settings(self, settings: dict) -> None:
        mp = self.model_panel
        for key, panel in (("model_panel", mp), ("dataset_panel", self.dataset_panel),
                           ("controls", self.controls)):
            apply_settings(panel, settings.get(key) or {})
        for fn in ("_task_changed", "_kernel_changed", "_conformers_changed"):
            f = getattr(mp, fn, None)
            if callable(f):
                try:
                    f()
                except Exception:  # noqa: BLE001 - a handler that needs a data set
                    pass
        self.refresh_fieldset()
        apply_field_table(mp, (settings.get("model_panel") or {}).get("__fields__", {}))
        self.refresh_fieldset()              # keeps the edits and rebuilds the field set
        full = bool(settings.get("results_full_width", False))
        self.act_full_width.setChecked(full)
        self.toggle_results_span(full)

    def _project_from_session(self) -> Project:
        return Project(dataset=self.ds, settings=self._session_settings(),
                       history=list(self.history.steps),
                       model_state=self.model.state() if self.model is not None else None,
                       train_indices=list(self.model_train_idx),
                       test_indices=list(self.model_test_idx),
                       decoy_indices=list(getattr(self, "model_decoy_idx", None) or []),
                       search_rows=list(self.search_rows), validation=self.validation,
                       log=self.results_panel.log.toPlainText(), protein=self.protein,
                       protein_path=str(getattr(self.protein, "path", "") or ""),
                       target=None if self.model_target is None
                       else np.asarray(self.model_target, float))

    def _set_project_path(self, path: Optional[str]) -> None:
        self.project_path = path
        base = f"CMF 3D-QSAR  –  Continuous Molecular Fields  (v{__version__})"
        self.setWindowTitle(f"{os.path.basename(path)}  –  {base}" if path else base)

    def save_project_file(self, save_as: bool = False):
        """Write the session: molecules and poses, pose history, settings, model, results."""
        if self.ds is None:
            QMessageBox.information(self, "Nothing to save",
                                    "Load a data set first; a project is a session.")
            return
        path = self.project_path
        if save_as or not path:
            start = path or "project" + PROJECT_EXTENSION
            path, _ = QFileDialog.getSaveFileName(self, "Save project", start,
                                                  f"CMF project (*{PROJECT_EXTENSION})")
            if not path:
                return
            if not path.lower().endswith(PROJECT_EXTENSION):
                path += PROJECT_EXTENSION
        try:
            save_project(path, self._project_from_session())
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.error(f"Could not save the project: {exc}")
            return
        self._set_project_path(path)
        what = [f"{len(self.ds)} molecules"]
        if len(self.history):
            what.append(f"{len(self.history)} pose snapshot(s)")
        what.append("the model" if self.model is not None else "no model")
        if self.protein is not None:
            what.append("the protein")
        self.log(f"Project saved to {path} ({', '.join(what)})")

    def open_project_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "",
                                              f"CMF project (*{PROJECT_EXTENSION});;All files (*)")
        if not path:
            return
        self.load_project_file(path)

    def load_project_file(self, path: str, on_loaded=None):
        def task(progress, cancel):
            progress(f"Reading {os.path.basename(path)}", 0.2)
            return load_project(path)

        self.run_task(task, lambda proj: self._apply_project(proj, path, on_loaded),
                      "Opening project")

    def _apply_project(self, proj: Project, path: str, on_loaded=None) -> None:
        self.ds = proj.dataset
        self.model_target = None
        self._after_dataset_change()
        self.history = PoseHistory(proj.history)
        self._refresh_undo_action()
        self._pose_reference = None
        try:
            self._apply_session_settings(proj.settings)
        except Exception as exc:  # noqa: BLE001 - a project from another version of the panels
            self.log(f"Some settings of the project could not be restored: {exc}")
        self._set_project_path(path)
        if proj.log:
            self.results_panel.append_log("---- log of the saved session ----")
            self.results_panel.append_log(proj.log)
            self.results_panel.append_log("---- end of the saved log ----")
        if proj.protein is not None:
            try:
                self._install_protein(proj.protein, "the project")
            except Exception as exc:  # noqa: BLE001
                self.log(f"The protein of the project could not be restored: {exc}")
        when = f", saved {proj.saved}" if proj.saved else ""
        self.log(f"Project opened from {path}: {len(self.ds)} molecules, "
                 f"{len(self.history)} pose snapshot(s){when}")
        self._update_conformer_info()
        try:
            model = proj.model
        except Exception as exc:  # noqa: BLE001
            self.error(f"The model in the project could not be read: {exc}")
            model = None
        if model is None:
            self.schedule_display()
            if on_loaded:
                on_loaded()
            return
        fs, mols, names = model.fieldset, self.ds.mols, self.ds.names

        def task(progress, cancel):
            mf = []
            for k, (m, n) in enumerate(zip(mols, names)):
                mf.append(build_fields(m, fs, n))
                if k % 20 == 0:
                    progress(f"Molecular fields of the model {k + 1}/{len(mols)}",
                             k / max(len(mols), 1))
            return mf

        def done(mf):
            self.model = model
            self.molfields = mf
            self.shapley_result = self.shapley_fields = None
            self.model_train_idx = list(proj.train_indices)
            self.model_test_idx = list(proj.test_indices)
            self.model_decoy_idx, self.model_decoy_scores = list(proj.decoy_indices), None
            self.model_target = proj.target
            self.search_rows = list(proj.search_rows)
            self.validation = proj.validation
            self.fieldset = model.fieldset
            self.volumes.clear()
            self._mol_volume_cache.clear()
            self.grid = None
            self.model_panel.show_fitted_values(model.fieldset, model.scale)
            self.controls.set_fields([(s.name, s.label) for s in model.fieldset.specs])
            self._show_results()
            self.log("    " + model_summary(model))
            self.schedule_display()
            if on_loaded:
                on_loaded()

        self.run_task(task, done, "Restoring the model")

    def _model_from_choice(self, source: int, path: str, label: str):
        """The model a side of the comparison names: this session's, a model file or a project."""
        if source == 0:
            if self.model is None:
                raise ValueError("there is no model in this session; build or load one, or "
                                 "choose a file for both sides")
            return self.model, (label or "this session")
        if not path or not os.path.exists(path):
            raise ValueError(f"no such file: {path or '(empty)'}")
        name = label or os.path.splitext(os.path.basename(path))[0]
        if source == 2 or is_project_file(path):
            model = load_project(path).model
            if model is None:
                raise ValueError(f"the project {os.path.basename(path)} holds no model")
            return model, name
        return load_model(path)[0], name

    def compare_models_dialog(self):
        """Two models, one set of molecules: the predictions side by side (plan item G2)."""
        if self.ds is None:
            QMessageBox.information(self, "No data", "Load a data set first.")
            return
        dlg = CompareModelsDialog(self.model is not None, self)
        if dlg.exec() != QDialog.Accepted:
            return
        try:
            model_a, label_a = self._model_from_choice(*dlg.side(0))
            model_b, label_b = self._model_from_choice(*dlg.side(1))
        except Exception as exc:  # noqa: BLE001 - shown to the user
            self.error(f"Could not set up the comparison: {exc}")
            return
        if model_a is model_b:
            QMessageBox.information(self, "Same model", "Choose two different models.")
            return
        ds = self.ds

        def task(progress, cancel):
            return compare_models(model_a, model_b, ds, labels=(label_a, label_b),
                                  progress=progress)

        def done(cmp):
            self.comparison = cmp
            for line in cmp.lines():
                self.results_panel.append_log(line)
            self.results_panel.tabs.setCurrentIndex(4)
            win = ComparisonWindow(cmp, self)
            self._comparison_window = win
            win.show()

        self.run_task(task, done, "Comparing two models")

    # ------------------------------------------------------------------ files
    def save_model_file(self):
        if self.model is None:
            QMessageBox.information(self, "No model", "Build a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save model", "model.cmf", "CMF model (*.cmf)")
        if not path:
            return
        extra = {"activity": self.ds.activity_prop if self.ds else "", "search_rows": self.search_rows}
        save_model(path, self.model, extra=extra)
        self.log(f"Model saved to {path}")

    def load_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load model", "", "CMF model (*.cmf);;All files (*)")
        if not path:
            return
        try:
            model, extra = load_model(path)
        except Exception as exc:  # noqa: BLE001
            self.error(f"Could not load model: {exc}")
            return
        # rebuild a data set from the training molecules stored in the model
        mols, names = [], []
        for mf in model.train.molecules:
            m = Chem.MolFromMolBlock(mf.molblock, removeHs=False) if mf.molblock else None
            if m is None:
                continue
            m.SetProp("_Name", mf.name)
            mols.append(m)
            names.append(mf.name)
        if len(mols) != model.train.n_mol:
            QMessageBox.warning(self, "Model", "The model does not contain displayable training structures.")
        ds = Dataset(mols=mols, names=names, activity=np.array(model.y, float),
                     subset=np.array(["train"] * len(mols), dtype=object), path=path,
                     activity_prop=extra.get("activity", "activity"))
        self.ds = ds
        self._after_dataset_change()
        self.model = model
        self.shapley_result = None
        self.shapley_fields = None
        self.molfields = list(model.train.molecules)
        self.model_train_idx = list(range(len(mols)))
        self.model_test_idx = []
        self.search_rows = extra.get("search_rows", [])
        self.fieldset = model.fieldset
        self.model_panel.fill_field_table(model.fieldset, model.scale)
        self.controls.set_fields([(s.name, s.label) for s in model.fieldset.specs])
        self._show_results()
        self.log(f"Model loaded from {path} ({model.train.n_mol} training molecules)")
        self.log("    " + model_summary(model))
        self.schedule_display()

    def export_predictions(self):
        if self.model is None or self.ds is None:
            QMessageBox.information(self, "No model", "Build a model first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export predictions", "predictions.csv", "CSV (*.csv)")
        if not path:
            return
        n = len(self.ds)
        yp = np.full(n, np.nan)
        ycv = np.full(n, np.nan)
        st = self.model.stats
        for k, i in enumerate(self.model_train_idx):
            yp[i] = st.extra.get("y_fit", [np.nan] * (k + 1))[k]
            if "y_cv" in st.extra:
                ycv[i] = st.extra["y_cv"][k]
        if "y_test_pred" in st.extra:
            for k, i in enumerate(self.model_test_idx):
                yp[i] = st.extra["y_test_pred"][k]
        write_predictions_csv(path, self.ds.names, self.ds.activity, yp, self.ds.subset, y_cv=ycv)
        self.log(f"Predictions exported to {path}")

    def export_sdf(self):
        if self.ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export data set", "dataset_aligned.sdf", "SDF (*.sdf)")
        if not path:
            return
        write_sdf(path, self.ds, activity_prop=self.ds.activity_prop or "activity")
        self.log(f"Data set exported to {path}")

    def export_cube(self):
        fname = self.controls.current_field()
        if self.model is None or fname not in self.volumes:
            QMessageBox.information(self, "No co-field", "Compute the co-fields first.")
            return
        tag = ""
        if self.multitask is not None and self.endpoint_label:
            # the file must say which endpoint it holds: a selectivity co-field and an
            # activity co-field of the same field type are otherwise indistinguishable
            tag = "_" + re.sub(r"[^A-Za-z0-9]+", "_", self.endpoint_label).strip("_")
        path, _ = QFileDialog.getSaveFileName(self, "Export co-field",
                                              f"cofield_{fname.replace(':', '_')}{tag}.cube",
                                              "Gaussian cube (*.cube)")
        if not path:
            return
        cur = self.dataset_panel.current_index()
        mol = self.molfields[cur] if cur >= 0 else None
        title = f"CMF co-field {fname}" + (f" [{self.endpoint_label}]" if tag else "")
        export_cube(path, self.grid, self.volumes[fname], mol, title=title)
        self.log(f"Co-field {fname}{tag} exported to {path}")

    def export_shapley_cube(self):
        """Write the Shapley field of the selected molecule, so it can be opened with the protein."""
        cur = self.dataset_panel.current_index()
        fname = self.controls.current_field()
        fs = self._display_fieldset()
        names = [s.name for s in fs.specs]
        if self.shapley_fields is None or cur < 0 or self.shapley_fields[0] != cur:
            QMessageBox.information(self, "No Shapley field",
                                    "Select a molecule and compute its decomposition in the "
                                    "Shapley tab with the players set to 'atoms x field types'.")
            return
        if fname not in names:
            QMessageBox.information(self, "No such field", f"The model has no {fname} field.")
            return
        f = names.index(fname)
        name = self.ds.names[cur]
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Shapley field",
            f"shapley_{name}_{fname.replace(':', '_')}.cube", "Gaussian cube (*.cube)")
        if not path:
            return
        vol, grid = self._shapley_volume(cur, fs, f)
        export_cube(path, grid, vol, self.molfields[cur],
                    title=f"CMF Shapley field {fname} of {name}")
        self.log(f"Shapley field ({fname}) of {name} exported to {path}; its integral is "
                 f"{vol.sum() * grid.spacing ** 3:+.3f}")

    def save_screenshot(self, path: Optional[str] = None):
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Save image", "cmf_view.png", "PNG image (*.png)")
        if not path:
            return
        self.viewer.screenshot(path)
        self.log(f"Image saved to {path}")

    def show_about(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("About CMF 3D-QSAR")
        dlg.resize(760, 640)
        lay = QVBoxLayout(dlg)
        tb = QTextBrowser()
        tb.setHtml(ABOUT_HTML)
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok)
        bb.accepted.connect(dlg.accept)
        lay.addWidget(bb)
        dlg.exec()

    def closeEvent(self, ev):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(5000)
        try:
            s = QSettings("cmfqsar", "CMF3DQSAR")
            s.setValue("geometry", self.saveGeometry())
            s.setValue("state", self.saveState())
            s.setValue("results_full_width", self.act_full_width.isChecked())
        except Exception:
            pass
        super().closeEvent(ev)

    def restore_saved_layout(self):
        """Restore the window geometry and dock arrangement of the previous session, if any."""
        try:
            s = QSettings("cmfqsar", "CMF3DQSAR")
            if s.value("geometry") is not None:
                self.restoreGeometry(s.value("geometry"))
            full = str(s.value("results_full_width", "false")).lower() == "true"
            self.act_full_width.setChecked(full)
            self.toggle_results_span(full)
            if s.value("state") is not None:
                self.restoreState(s.value("state"))
        except Exception:
            pass


def molfield_wireframe(choice: str, cofield_drawn: bool) -> bool:
    """Whether the molecular field is drawn as a mesh rather than a solid surface.

    The two maps share a colour scale -- positive and negative in the field's own two colours --
    so when both are on screen only the drawing style tells them apart, and the mesh has to be
    the molecular field: it is one connected envelope around the whole molecule, and drawn solid
    it would hide the molecule and the co-field lobes alike.  Alone on screen it has nothing to
    hide and nothing to be confused with, so it is drawn solid.  That is `auto`; a reader who
    wants the opposite picture -- lobes as a cage around a solid field -- names the style.
    """
    return choice == "mesh" or (choice == "auto" and bool(cofield_drawn))


#: a residue is listed as being in contact within this distance, as `complement --residues` has it
CONTACT_DISTANCE = 4.5


def _cname(c) -> str:
    r, g, b = c
    names = {(0.90, 0.15, 0.15): "red", (0.15, 0.30, 0.95): "blue", (0.15, 0.80, 0.20): "green",
             (0.95, 0.85, 0.10): "yellow", (1.00, 0.55, 0.05): "orange", (0.45, 0.80, 1.00): "light blue",
             (0.90, 0.20, 0.90): "magenta", (0.55, 0.05, 0.10): "dark red", (0.10, 0.85, 0.90): "cyan",
             (0.55, 0.20, 0.80): "purple"}
    return names.get((r, g, b), f"rgb({r:.1f},{g:.1f},{b:.1f})")
