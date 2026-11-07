"""What the whole test suite shares: the one Qt application of the run, and how the run ends.

**One application.** Seven test modules drive windows, panels and charts, and each of them used
to build its own with ``QApplication.instance() or QApplication([])``. Only the first of those
calls creates one and the rest are handed that same object, so there was never more than one
application -- but the fixture that created it was module-scoped, and a module-scoped fixture is
torn down when its module is finished. With the last Python reference gone, Qt deleted the
application under the modules that still needed it, and said so once per run: "QObject: shared
QObject was deleted directly. The program is malformed and may crash." The application lives
here instead, in a session fixture that holds it in a module-level name for the whole run. It is
not created eagerly: a run that asks for no window asks for no application.

**How the run ends.** That alone did not make the runners green, and the check in
:func:`pytest_sessionfinish` is what said why: the application was still there after the last
test, so what crashes is Python's own shutdown afterwards. Taking PySide6 apart at that point is
not reliable on every machine. With one application owned here, five of the six suite jobs went
green and two still failed after a passing suite -- a segmentation fault on Ubuntu with Python
3.10, with "Fatal Python error: none_dealloc: deallocating None", and exit code 1 with no
message at all on Windows with Python 3.13 (plan item A2). None of it is a test, and no test can
catch it: it happens after the last one, and the summary line on those jobs read 2,279 passed.

So the process ends here, with the status pytest arrived at. A failing run still returns a
failing status, because that status is what is passed on; what is skipped is only what Python
does after the report is written. Set ``CMF_TESTS_FULL_SHUTDOWN=1`` to shut down normally
instead, which a tool that writes its output after the run needs -- coverage, for instance.
"""
import os
import sys

import pytest

# the runners have no screen, and neither has a developer running the suite over a terminal
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

#: the one application of the run, held for as long as the process lives
_APPLICATION = None

#: set it to let the interpreter shut down in the ordinary way, crash and all
FULL_SHUTDOWN = "CMF_TESTS_FULL_SHUTDOWN"

#: the status pytest arrived at, handed from the end of the session to the end of the run
_EXIT_STATUS = 0


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


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Check that the application survived the run, and close the windows it left behind.

    The check belongs here and not in a test, because the deletion it looks for happened after
    the last module that used a window was finished -- which is to say after every test had run.
    That is why the symptom was a process that crashed while shutting down on a suite whose every
    test had passed, and why nothing in the suite noticed.
    """
    global _EXIT_STATUS

    _EXIT_STATUS = int(exitstatus)
    app = _APPLICATION
    if app is None:
        return
    if not _application_is_alive(app):
        print("\nthe Qt application was deleted while the suite was still running: a fixture "
              "owned the only reference to it (tests/conftest.py, plan item A2)")
        _EXIT_STATUS = 1
        session.exitstatus = 1
        return
    for widget in list(app.topLevelWidgets()):
        widget.close()
    app.processEvents()


def pytest_unconfigure(config):
    """End the process with the status pytest arrived at, rather than shut Python down.

    This is the last hook of a run, so the report and the summary are written by the time it is
    called. It must not be done in :func:`pytest_sessionfinish`: the terminal reporter writes
    the summary in a hook *wrapper* around that hook, so a run that ended there printed its
    results and its reason for failing nowhere.
    """
    if _APPLICATION is None or os.environ.get(FULL_SHUTDOWN):
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_EXIT_STATUS)
