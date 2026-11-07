"""What the whole test suite shares: the one Qt application of the run.

Seven test modules drive windows, panels and charts, and each of them used to build its own
application with ``QApplication.instance() or QApplication([])``. Only the first of those calls
creates one and the rest are handed that same object, so there was never more than one
application -- but the fixture that created it was module-scoped, and a module-scoped fixture is
torn down when its module is finished. With the last Python reference gone, Qt deleted the
application under the modules that still needed it. Qt said so, once per run: "QObject: shared
QObject was deleted directly. The program is malformed and may crash." The process then crashed
as it shut down, after every test had passed -- a segmentation fault on the Ubuntu and macOS
runners, exit code 1 on the Windows ones -- which made five of the six continuous-integration
jobs red on a suite that reported 2,275 passed (plan item A2).

The application lives here instead. The session fixture below creates it at most once and keeps
it in a module-level name for the rest of the run, so no module teardown can take it away. It is
not created eagerly: a run that asks for no window asks for no application either.
"""
import os

import pytest

# the runners have no screen, and neither has a developer running the suite over a terminal
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: the one application of the run, held for as long as the process lives
_APPLICATION = None


@pytest.fixture(scope="session")
def qt_application():
    """The Qt application every window of the suite belongs to.

    Skips the test that asks for it when Qt is not installed, which is how the suite behaves
    without the ``gui`` extra.
    """
    global _APPLICATION
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    if _APPLICATION is None:
        _APPLICATION = QApplication.instance() or QApplication([])
    return _APPLICATION


def _application_is_alive(app) -> bool:
    """Whether the C++ object behind the Python wrapper still exists."""
    try:
        from shiboken6 import Shiboken
    except Exception:                                 # noqa: BLE001 - then we cannot tell
        return True
    return bool(Shiboken.isValid(app))


def pytest_sessionfinish(session, exitstatus):
    """Close what the windows left behind, and say so if the application was deleted under them.

    The application itself is left alive on purpose: the process is ending, the operating system
    reclaims it, and destroying it from Python while another Qt object is still wrapped is the
    fault this file exists to avoid.

    The check belongs here and not in a test, because the deletion it looks for happened after
    the last module that used a window was finished -- which is to say after every test had run.
    That is why the symptom was a process that crashed while shutting down on a suite whose every
    test had passed, and why nothing in the suite noticed.
    """
    app = _APPLICATION
    if app is None:
        return
    if not _application_is_alive(app):
        print("\nthe Qt application was deleted while the suite was still running: a fixture "
              "owned the only reference to it (tests/conftest.py, plan item A2)")
        session.exitstatus = 1
        return
    for widget in list(app.topLevelWidgets()):
        widget.close()
    app.processEvents()
