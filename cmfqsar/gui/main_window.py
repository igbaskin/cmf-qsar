"""Main window of the CMF 3D-QSAR desktop application."""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QDockWidget, QFileDialog, QFormLayout,
                               QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QCheckBox,
                               QSpinBox, QDoubleSpinBox, QTextBrowser, QVBoxLayout, QHBoxLayout,
                               QWidget, QComboBox, QScrollArea)
from rdkit import Chem

from .. import __version__
from ..align import align_dataset
from ..cofields import (Grid, cofield_volume, export_cube, isosurface, make_grid, molfield_volume,
                        overlap_volume)
from ..conformers import CONFORMER_ALIGNMENTS
from ..dataio import Dataset, load_dataset, load_model, numeric_properties, save_model, write_predictions_csv, write_sdf
from ..fields import AtomTable, FieldSet, MoleculeFields, build_fields, cif_types_in
from ..model import CMFModel, grid_search, nested_cv
from ..protein import (binding_site, contact_distance, fetch_pdb, find_pocket,
                       read_structure, residue_anchors)
from .panels import DatasetPanel, ModelPanel, ResultsPanel, ViewerControls
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
        lay.addLayout(form)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def max_rmsd(self):
        v = self.spin_max.value()
        return None if v <= 0 else float(v)


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
        self.setWindowTitle("Chemical space map (GTM)")
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
            "predicted from its own scaffold-mates. Random folds flatter a mixed series badly, "
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
        self.model_test_idx: List[int] = []
        self.shapley_result = None          # (molecule index, ShapleyResult) of the last run
        self.shapley_fields = None          # (molecule index, (F, N) values) for the Shapley field
        self.protein = None                 # cmfqsar.protein.Protein behind the molecules
        self.complementarity = None         # scores of the series against the pocket
        self.chemotypes = None              # (labels, scaffold names) of a heterogeneous series
        self.chemotype_result = None
        self._comp_surface_cache = {}
        self._site_cache = None             # (key, Protein) of the last binding site drawn
        self.search_rows = []
        self.grid: Optional[Grid] = None
        self.volumes: Dict[str, np.ndarray] = {}
        self._mol_volume_cache: Dict[tuple, np.ndarray] = {}
        self.worker: Optional[TaskWorker] = None
        self._pose_reference: Optional[Dict[int, np.ndarray]] = None   # poses before "Unalign"
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
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(self.model_panel.minimumSizeHint().width() + 20)
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
        self._act(tm, "Chemical space map (GTM)…", self.gtm_map_dialog)
        tm.addSeparator()
        self._act(tm, "Align decoys to the training set…", self.align_decoys_dialog)
        self._act(tm, "Unalign (randomise poses)…", self.unalign_dialog)
        self._act(tm, "Center view on current molecule", self.center_current)
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
                             charge_model=mp.cmb_charges.currentText(), cif_scheme=mp.cmb_cif_scheme.currentText(),
                             param_scheme=mp.cmb_param.currentText(),
                             conformers=mp.conformer_mode(),
                             conformer_weighting=mp.conformer_weighting(),
                             conformer_kt=mp.spin_conf_kt.value(),
                             max_conformers=mp.spin_conf_max.value())
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
        folds, seed = mp.spin_folds.value(), mp.spin_seed.value()
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
                             scale_fields=scale, kernel_layer=layer)
            if task_name == "oneclass":
                # no cross-validation objective exists without negatives: alpha sets the level of
                # generalisation and nu the expected fraction of outliers, both chosen by the user
                if alphas:
                    model.fieldset.set_alpha(alphas[0])
                model.params["nu"] = nus[0]
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
            return {"model": model, "mf": mf, "res": res, "tr": tr, "te": te, "dec": dec,
                    "d_scores": d_scores, "target": y_all}

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
               "Fields: " + ", ".join(f"{s.label} (alpha={s.alpha:g}, h={s.h:g})" for s in m.fieldset.specs)]
        if m.method == "gpc":
            from ..gpc import brier_score
            try:
                txt.insert(2, f"Gaussian process: log marginal likelihood "
                              f"{m.log_marginal_likelihood:.2f}, Brier score of the fitted "
                              f"probabilities {brier_score(m.y, m.predict_proba(m.train)):.3f} "
                              f"(0.25 is a coin). The Scores tab carries p(active) from the latent mean.")
            except Exception as exc:                   # noqa: BLE001
                txt.insert(2, f"Gaussian process: probabilities unavailable ({exc})")
        rp.txt_stats.setPlainText("\n".join(txt))
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
            table = AtomTable.from_molecules(mf)
            progress("Predicting", 0.3)
            yp = model.evaluate_test(table, y[te], cancel=cancel)
            var = None
            if model.method in ("krr", "gp"):
                try:
                    _, var = model.predict_with_variance(table)
                except Exception:
                    var = None
            return mf, yp, var

        def done(out):
            mf, yp, var = out
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
                    ("alpha", res.alpha), ("lambda", res.lam)]
            per = [(f"{g['name'][:48]}", g["n"], g["rmse"], g["bias"]) for g in res.per_group]
            self.results_panel.set_chemotypes(
                f"{len(res.per_group)} chemotypes held out in turn. Random folds flatter a mixed "
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
            return pocket, mf, res, stats, agree

        def done(out):
            pocket, mf, res, stats, agree = out
            self.complementarity = {"pocket": pocket, "fields": fs, "by_index": dict(zip(idx, res)),
                                    "mols": dict(zip(idx, mf)), "stats": stats, "agreement": agree,
                                    "surface": {}}
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
            self._mol_volume_cache.clear()
            self.log(f"Co-fields computed on a {'x'.join(map(str, self.grid.shape))} grid "
                     f"(spacing {self.grid.spacing:.2f} Å)")
            self.controls.chk_cofield.setChecked(True)
            self.schedule_display()

        self.run_task(task, done, "Computing co-fields")

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
                    if vc.chk_site_labels.isChecked():
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
        if vc.chk_cofield.isChecked():
            if model is None:
                legend.append("co-field: build a model, then click 'Compute co-field'")
            elif fname not in self.volumes:
                legend.append("co-field: click 'Compute co-field'")
            else:
                pos_c, neg_c = cofield_colors(fname)
                vol = self.volumes[fname]
                amax = float(np.abs(vol).max())
                if amax > 0:
                    mesh = vc.cmb_cofield_style.currentText() == "mesh"
                    for sign, col in ((1, pos_c), (-1, neg_c)):
                        s = isosurface(vol, sign * frac * amax, self.grid)
                        if s is not None:
                            v.add_isosurface(s[0], s[1], col, "cofield", alpha=0.75 if mesh else op,
                                             wireframe=mesh)
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
                    mesh = vc.cmb_molfield_style.currentText() == "mesh"
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
            tc = model.training_contributions(mol)
            c = model.atom_contributions(mol) if model.is_linear else None
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
        model, fs = self.model, self.model.fieldset
        mols, names = self.ds.mols, self.ds.names

        def task(progress, cancel):
            from ..varalign import variance_align
            mf = [build_fields(mols[i], fs, names[i]) for i in idx]
            res = variance_align(mf, model, max_rmsd=max_rmsd, restarts=restarts,
                                 progress=lambda i, n: progress(f"Aligning {i}/{n}", i / n),
                                 cancel=cancel)
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

        def task(progress, cancel):
            from ..gtm import CMFMap
            mf = [build_fields(mols[i], fs, names[i]) for i in idx]
            cmap = CMFMap(n_components=comps, grid=grid, rbf=rbf, normalize=True)
            cmap.fit(AtomTable.from_molecules(mf), fs, progress=progress)
            return cmap, mf

        def done(out):
            cmap, mf = out
            self.gtm_map = cmap
            self.gtm_map_idx = idx
            act = ds.activity[idx]
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
        n, ok_btn = QInputDialog.getInt(
            self, "Align decoys",
            f"Place the {len(dec)} background molecules in the frame of the {len(tr)} training "
            f"molecules by maximising their field similarity to them, so that no decoy is "
            f"rejected merely for being posed badly."
            "\n\nRandom starting orientations per decoy:",
            6, 0, 50)
        if not ok_btn:
            return
        fs = self.model.fieldset if self.model is not None else self._fieldset_from_panel()
        mols, names = self.ds.mols, self.ds.names

        def task(progress, cancel):
            from ..decoys import align_decoys
            ref = AtomTable.from_molecules([build_fields(mols[i], fs, names[i]) for i in tr])
            mf = [build_fields(mols[i], fs, names[i]) for i in dec]
            res = align_decoys(mf, ref, fs, restarts=n,
                               progress=lambda i, m: progress(f"Aligning decoy {i}/{m}", i / m),
                               cancel=cancel)
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
        path, _ = QFileDialog.getSaveFileName(self, "Export co-field", f"cofield_{fname.replace(':', '_')}.cube",
                                              "Gaussian cube (*.cube)")
        if not path:
            return
        cur = self.dataset_panel.current_index()
        mol = self.molfields[cur] if cur >= 0 else None
        export_cube(path, self.grid, self.volumes[fname], mol, title=f"CMF co-field {fname}")
        self.log(f"Co-field {fname} exported to {path}")

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


def _cname(c) -> str:
    r, g, b = c
    names = {(0.90, 0.15, 0.15): "red", (0.15, 0.30, 0.95): "blue", (0.15, 0.80, 0.20): "green",
             (0.95, 0.85, 0.10): "yellow", (1.00, 0.55, 0.05): "orange", (0.45, 0.80, 1.00): "light blue",
             (0.90, 0.20, 0.90): "magenta", (0.55, 0.05, 0.10): "dark red", (0.10, 0.85, 0.90): "cyan",
             (0.55, 0.20, 0.80): "purple"}
    return names.get((r, g, b), f"rgb({r:.1f},{g:.1f},{b:.1f})")
