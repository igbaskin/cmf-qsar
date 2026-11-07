"""The one Qt application the suite shares (plan item A2).

Every window, panel and chart of the suite belongs to a single application, created once by the
session fixture in `conftest.py` and held there for the whole run. Before that, each module
built its own with `QApplication.instance() or QApplication([])` in a module-scoped fixture, and
the end of a module could therefore delete the application other modules were still using.
"""
import os
import shutil
import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

HERE = os.path.dirname(os.path.abspath(__file__))


def test_the_suite_has_one_application(qt_application):
    """Asking twice gives the same object, and Qt knows it as the running application."""
    from PySide6.QtWidgets import QApplication

    assert qt_application is not None
    assert QApplication.instance() is qt_application


def test_the_application_is_not_owned_by_the_fixture_alone(qt_application):
    """What went wrong before: the only reference lived in a fixture that was torn down.

    `conftest` keeps the application in a module-level name, so no module teardown and no
    garbage collection between two modules can take it away. Whether it is *still* alive at the
    end of the run is checked by `pytest_sessionfinish` in that file, because the deletion used
    to happen after the last test.
    """
    import conftest

    assert conftest._APPLICATION is qt_application


def test_the_c_plus_plus_object_behind_it_exists(qt_application):
    """A deleted application leaves a Python wrapper that looks fine; shiboken6 can tell."""
    shiboken = pytest.importorskip("shiboken6")

    assert shiboken.Shiboken.isValid(qt_application)


def test_the_check_notices_a_deleted_object():
    """The check that guards the run, driven into failure: a check that never fails is not known
    to work. A plain QObject stands in for the application, since deleting the real one would
    take the process with it."""
    shiboken = pytest.importorskip("shiboken6")
    from PySide6.QtCore import QObject

    import conftest

    obj = QObject()
    assert conftest._application_is_alive(obj)
    shiboken.delete(obj)
    assert not conftest._application_is_alive(obj)


def _run_pytest_on(folder, *args):
    """Start pytest on a folder of its own and return the finished process."""
    return subprocess.run([sys.executable, "-m", "pytest", str(folder), "-q", *args],
                          capture_output=True, text=True)


@pytest.mark.parametrize("body, status, wanted", [
    ("    assert qt_application is not None\n", 0, "1 passed"),
    ("    assert qt_application is None, 'deliberate'\n", 1, "1 failed"),
])
def test_the_run_ends_with_the_status_pytest_arrived_at(tmp_path, body, status, wanted):
    """A run that has built a Qt application ends with `os._exit`, so what it passes on is
    checked in a run of its own: a copy of `conftest.py` beside one test, started as a
    subprocess. A failing run must still fail, and the summary must be in the output -- a run
    that ends before its report is written would be worse than one that crashes after it.
    """
    shutil.copyfile(os.path.join(HERE, "conftest.py"), str(tmp_path / "conftest.py"))
    with open(str(tmp_path / "test_one.py"), "w", encoding="utf-8") as fh:
        fh.write("def test_one(qt_application):\n" + body)

    done = _run_pytest_on(tmp_path)
    assert done.returncode == status, done.stdout + done.stderr
    assert wanted in done.stdout, done.stdout + done.stderr
