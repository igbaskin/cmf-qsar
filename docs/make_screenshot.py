#!/usr/bin/env python
"""
Take the screenshot of the application for the manual: a free-energy series with the binding site
of its reference crystal structure on screen.

The application is started as `cmf-app` starts it, the CDK8 set of ``examples/openff`` is loaded
(its pocket comes with it from ``examples/pockets``), a model is built and its co-field computed,
the complementarity of every ligand to the pocket is scored, and the window is captured with the
most active ligand selected, the site drawn as sticks with its residue labels, the co-field as
isosurfaces and the complementarity surface on the ligand.  The picture is rendered by OpenGL on
the screen, so it needs a display and is not reproduced byte for byte; it is therefore not part of
``docs/make_figures.py --check``.

Usage:  python docs/make_screenshot.py [--out docs/figures/pocket_window.png] [--width 1700]
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DATASET = os.path.join(ROOT, "examples", "openff", "cdk8.sdf")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=os.path.join(HERE, "figures", "pocket_window.png"))
    ap.add_argument("--width", type=int, default=1700)
    ap.add_argument("--height", type=int, default=1000)
    args = ap.parse_args(argv)

    import numpy as np
    from PySide6.QtCore import QTimer
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
    from cmfqsar.gui.main_window import MainWindow

    win = MainWindow()
    win.resize(args.width, args.height)
    win.show()
    win._auto = True                      # the activity is given: no property dialog

    def when_done(then):
        """Run `then` once the task the window has just started reports success."""
        win.worker.finished_ok.connect(lambda _: QTimer.singleShot(300, then))

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
        from PySide6.QtCore import QPoint, QRect
        from PySide6.QtGui import QPainter
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        # grabbing the window renders the OpenGL view without the text it draws over the scene
        # (the residue labels), so the view's own framebuffer is painted into the window image
        # at the place of the view, scaled by the device pixel ratio of both
        window = win.grab()
        frame = win.viewer.view.grabFramebuffer()
        ratio = window.width() / max(win.width(), 1)      # physical pixels per logical pixel
        origin = win.viewer.view.mapTo(win, QPoint(0, 0))
        size = win.viewer.view.size()
        target = QRect(int(round(origin.x() * ratio)), int(round(origin.y() * ratio)),
                       int(round(size.width() * ratio)), int(round(size.height() * ratio)))
        print(f"window {win.width()}x{win.height()} logical, grab {window.width()}x"
              f"{window.height()}, view at {origin.x()},{origin.y()} size {size.width()}x"
              f"{size.height()}, framebuffer {frame.width()}x{frame.height()}, ratio {ratio:.3f}")
        image = window.toImage()
        image.setDevicePixelRatio(1.0)                     # paint in physical pixels
        painter = QPainter(image)
        painter.drawImage(target, frame)
        painter.end()
        image.save(args.out)
        print(f"written {os.path.relpath(args.out, ROOT)} ({image.width()} x {image.height()})")
        QTimer.singleShot(300, app.quit)

    win.load_dataset_file(DATASET, "train", activity="pAffinity",
                          on_loaded=lambda: QTimer.singleShot(300, step_model))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
