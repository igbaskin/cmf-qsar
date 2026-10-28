"""Application entry point."""
from __future__ import annotations

import argparse
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description="CMF 3D-QSAR desktop application")
    p.add_argument("dataset", nargs="?",
                   help="structure file to load as training set, or a project file (.cmfproj) "
                        "to reopen")
    p.add_argument("--activity", default=None, help="SD property with activity values")
    p.add_argument("--test-fraction", type=float, default=0.0)
    p.add_argument("--autotest", default=None, metavar="PNG",
                   help="(development) build a model, compute co-fields, save a screenshot and quit")
    args = p.parse_args(argv)

    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QTimer

    fmt = QSurfaceFormat()
    fmt.setSamples(4)
    QSurfaceFormat.setDefaultFormat(fmt)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("CMF 3D-QSAR")
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    # draw check boxes with an explicit indicator: on a dark palette the default unchecked box
    # is a filled square and is easily mistaken for a ticked one
    from .panels import _SHOW_BOX_STYLE
    app.setStyleSheet(_SHOW_BOX_STYLE)

    from .main_window import MainWindow
    win = MainWindow()
    win.show()

    if args.dataset:
        win._auto = args.activity is not None

        def after_load():
            if args.test_fraction > 0:
                win.dataset_panel.spin_test_frac.setValue(args.test_fraction)
                win.random_split()
            if args.autotest:
                _autotest(win, args.autotest)

        from ..project import PROJECT_EXTENSION, is_project_file
        if args.dataset.lower().endswith(PROJECT_EXTENSION) or is_project_file(args.dataset):
            win.load_project_file(args.dataset, on_loaded=after_load)
        else:
            win.load_dataset_file(args.dataset, "train", activity=args.activity, on_loaded=after_load)

    return app.exec()


def _autotest(win, png):
    """Drive the GUI through a complete workflow for smoke testing."""
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    def step1():
        win.model_panel.ed_alphas.setText("0.3, 0.6, 1.0")
        win.model_panel.ed_lams.setText("1e-2, 0.1, 1")
        win.model_panel.chk_cif.setChecked(True)
        win.build_model()
        win.worker.finished_ok.connect(lambda _: QTimer.singleShot(200, step2))

    def step2():
        win.controls.spin_spacing.setValue(0.6)
        win.compute_cofields()
        win.worker.finished_ok.connect(lambda _: QTimer.singleShot(200, step3))

    def step3():
        win.controls.chk_molfield.setChecked(True)
        win.controls.chk_atoms.setChecked(True)
        win.controls.chk_overlap.setChecked(False)
        win.dataset_panel.select_index(1)
        win.update_display()
        win.center_current()
        QApplication.processEvents()
        QTimer.singleShot(800, step4)

    def step4():
        win.save_screenshot(png)
        win.results_panel.tabs.setCurrentIndex(1)
        QApplication.processEvents()
        win.grab().save(png.replace(".png", "_window.png"))
        print("AUTOTEST OK")
        print(win.results_panel.txt_stats.toPlainText())
        QTimer.singleShot(300, QApplication.instance().quit)

    QTimer.singleShot(300, step1)


if __name__ == "__main__":
    sys.exit(main())
