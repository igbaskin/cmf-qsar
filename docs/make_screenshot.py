#!/usr/bin/env python
"""
Take the screenshots of the application for the manual.

Three pictures, chosen with ``--what``:

* ``pocket``: the main window with a free-energy series and the binding site of its reference
  crystal structure.  The CDK8 set of ``examples/openff`` is loaded (its pocket comes with it from
  ``examples/pockets``), a model is built and its co-field computed, the complementarity of every
  ligand to the pocket is scored, and the window is captured with the most active ligand selected,
  the site drawn as sticks with its residue labels and contributions, and the co-field as
  isosurfaces.
* ``crystal``: the window for crystals with the thirteen solid electrolytes of
  ``examples/hops/disordered``, a periodic-autocorrelation model fitted to the number of lithium
  sites per cubic nanometre, and the largest cell, the garnet, on screen with its periodic images
  out to the kernel's cutoff and its sites coloured by their contribution to the prediction.
* ``slab``: the same window with the Fe(110) slab of ``examples/corrosion_fe110`` and a
  benzimidazole molecule adsorbed flat on it, the pose relaxed with the UMA potential, read as a
  layer periodic in two directions and drawn in ball and stick, so the molecule has its bonds
  and the iron atoms stay spheres.

Grabbing a window renders the OpenGL view without the text it draws over the scene (the residue
labels, the site numbers), so the view's own framebuffer is painted into the window image at the
place of the view.  The pictures are rendered by OpenGL on the screen, so they need a display and
are not reproduced byte for byte; they are therefore not part of ``docs/make_figures.py --check``.

Usage:  python docs/make_screenshot.py [--what all|pocket|crystal|slab] [--width 1700]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from cmfqsar.dataio import relative_path  # noqa: E402

FIGURES = os.path.join(HERE, "figures")
POCKET_SET = os.path.join(ROOT, "examples", "openff", "cdk8.sdf")
CRYSTAL_SET = os.path.join(ROOT, "examples", "hops", "disordered")
SLAB_FILE = os.path.join(ROOT, "examples", "corrosion_fe110", "157_benzimidazole_uma_work",
                         "Benzimidazole_0_flat.xyz")
OUTPUTS = {"pocket": "pocket_window.png", "crystal": "crystal_window.png",
           "slab": "slab_window.png"}


def capture(win, view, out: str) -> None:
    """The window as an image, with the OpenGL view's own framebuffer painted in its place."""
    from PySide6.QtCore import QPoint, QRect
    from PySide6.QtGui import QPainter
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    window = win.grab()
    frame = view.grabFramebuffer()
    ratio = window.width() / max(win.width(), 1)      # physical pixels per logical pixel
    origin = view.mapTo(win, QPoint(0, 0))
    size = view.size()
    target = QRect(int(round(origin.x() * ratio)), int(round(origin.y() * ratio)),
                   int(round(size.width() * ratio)), int(round(size.height() * ratio)))
    image = window.toImage()
    image.setDevicePixelRatio(1.0)                     # paint in physical pixels
    painter = QPainter(image)
    painter.drawImage(target, frame)
    painter.end()
    image.save(out)
    print(f"written {relative_path(out, ROOT)} ({image.width()} x {image.height()})")


def shoot_pocket(app, args, then):
    """The main window: the CDK8 series inside its crystal binding site."""
    import numpy as np
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication
    from cmfqsar.gui.main_window import MainWindow

    win = MainWindow()
    win.resize(args.width, args.height)
    win.show()
    win._auto = True                      # the activity is given: no property dialog

    def when_done(next_step):
        win.worker.finished_ok.connect(lambda _: QTimer.singleShot(300, next_step))

    def step_model():
        win.model_panel.ed_alphas.setText("1.0")
        win.model_panel.ed_lams.setText("0.1")
        win.build_model()
        when_done(step_cofield)

    def step_cofield():
        win.controls.spin_spacing.setValue(0.5)
        win.compute_cofields()
        when_done(step_complementarity)

    def step_complementarity():
        win.results_panel.chk_comp_surface.setChecked(True)
        win.results_panel.chk_comp_residues.setChecked(True)
        win.compute_complementarity()
        when_done(step_display)

    def step_display():
        c = win.controls
        c.chk_cofield.setChecked(True)
        c.chk_molfield.setChecked(False)
        c.chk_protein.setChecked(True)
        c.cmb_protein_style.setCurrentText("sticks")
        c.spin_site.setValue(5.0)
        # the residues labelled with what each contributes to the ligand's complementarity;
        # the complementarity surface itself would hide the ligand and the co-field
        c.chk_site_labels.setChecked(True)
        c.chk_site_contrib.setChecked(True)
        c.chk_comp_map.setChecked(False)
        tabs = win.results_panel.tabs
        for k in range(tabs.count()):
            if tabs.tabText(k).startswith("Complementarity"):
                tabs.setCurrentIndex(k)
        best = int(np.nanargmax(win.ds.activity))
        win.dataset_panel.select_index(best)
        win.update_display()
        win.center_current()
        win.viewer.view.setCameraPosition(distance=30, elevation=22, azimuth=-50)
        QApplication.processEvents()
        QTimer.singleShot(1500, step_capture)

    def step_capture():
        capture(win, win.viewer.view, os.path.join(FIGURES, OUTPUTS["pocket"]))
        win.close()
        QTimer.singleShot(300, then)

    win.load_dataset_file(POCKET_SET, "train", activity="pAffinity",
                          on_loaded=lambda: QTimer.singleShot(300, step_model))


def _crystal_window(args):
    """The window for crystals, with its tasks run in the foreground so that a script can
    drive it step by step."""
    from cmfqsar.gui.crystal_window import CrystalWindow
    win = CrystalWindow(background=False)
    win.resize(args.width, args.height)
    win.show()
    return win


def shoot_crystal(app, args, then):
    """The crystal window: the garnet of the solid-electrolyte set with a fitted model."""
    import numpy as np
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    win = _crystal_window(args)

    def step_all():
        win.load(relative_path(CRYSTAL_SET, os.getcwd()).replace(os.sep, "/"))
        # the target benchmarks/run_cell_view.py uses: a property of the cell that every one
        # of these conductors has and the fields can see, lithium sites per cubic nanometre
        y = []
        for a in win.atoms:
            occ = a.arrays.get("occupancy")
            li = [k for k, s in enumerate(a.get_chemical_symbols()) if s == "Li"]
            count = float(np.sum(occ[li])) if occ is not None else float(len(li))
            y.append(1000.0 * count / float(a.get_volume()))
        win.y = np.array(y, float)
        win.target_name = "Li sites per cubic nanometre"
        win.populate()
        win.cmb_kernel.setCurrentIndex(0)                 # the periodic autocorrelation
        win.cmb_charges.setCurrentText("electronegativity")
        win.spin_min_structures.setValue(3)
        win.spin_folds.setValue(3)
        win.ed_alphas.setText("1")
        win.ed_lams.setText("1e-3,1e-2,0.1")
        win.build_model()
        largest = int(np.argmax([len(a) for a in win.atoms]))
        win.table.setCurrentCell(largest, 0)
        win.cmb_atoms.setCurrentText("balls")
        win.chk_cell.setChecked(True)
        win.chk_images.setChecked(True)
        win.chk_labels.setChecked(False)
        win.chk_sites.setChecked(True)
        win.update_display()
        win.center_current()
        win.viewer.view.setCameraPosition(elevation=24, azimuth=-55)
        QApplication.processEvents()
        QTimer.singleShot(1500, step_capture)

    def step_capture():
        capture(win, win.viewer.view, os.path.join(FIGURES, OUTPUTS["crystal"]))
        win.close()
        QTimer.singleShot(300, then)

    QTimer.singleShot(300, step_all)


def shoot_slab(app, args, then):
    """The crystal window: the Fe(110) slab with a benzimidazole adsorbed on it."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    win = _crystal_window(args)

    def step_all():
        win.cmb_pbc.setCurrentIndex(2)                    # ab: a layer
        # the path as given is what the window's log and status line show
        win.load(relative_path(SLAB_FILE, os.getcwd()).replace(os.sep, "/"))
        win.table.setCurrentCell(0, 0)
        # bonds perceived from the distances: the molecule gets its sticks, the iron atoms,
        # which the perception leaves without bonds, stay spheres
        win.cmb_atoms.setCurrentText("ball and stick")
        win.chk_cell.setChecked(True)
        win.chk_images.setChecked(True)
        win.spin_images.setValue(5.0)
        win.chk_labels.setChecked(False)
        win.update_display()
        win.center_current()
        win.viewer.view.setCameraPosition(distance=26, elevation=26, azimuth=-62)
        QApplication.processEvents()
        QTimer.singleShot(1500, step_capture)

    def step_capture():
        capture(win, win.viewer.view, os.path.join(FIGURES, OUTPUTS["slab"]))
        win.close()
        QTimer.singleShot(300, then)

    QTimer.singleShot(300, step_all)


SHOTS = {"pocket": shoot_pocket, "crystal": shoot_crystal, "slab": shoot_slab}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--what", choices=["all"] + list(SHOTS), default="all")
    ap.add_argument("--width", type=int, default=1700)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args(argv)

    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("CMF 3D-QSAR")
    app.setStyle("Fusion")
    from cmfqsar.gui.panels import _SHOW_BOX_STYLE
    app.setStyleSheet(_SHOW_BOX_STYLE)

    wanted = list(SHOTS) if args.what == "all" else [args.what]

    def run(k: int):
        if k >= len(wanted):
            app.quit()
            return
        SHOTS[wanted[k]](app, args, lambda: run(k + 1))

    run(0)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
