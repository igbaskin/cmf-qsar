"""Background execution of long computations with progress reporting and cancellation."""
from __future__ import annotations

import threading
import traceback
from typing import Callable, Optional

from PySide6.QtCore import QObject, QThread, Signal


class TaskWorker(QThread):
    """Run ``fn(progress, cancel)`` in a thread.

    ``progress(message: str, fraction: float)`` may be called from the task to
    report progress; ``cancel()`` returns True when cancellation was requested.
    """
    progress = Signal(str, float)
    finished_ok = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._fn = fn
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def run(self) -> None:
        try:
            result = self._fn(self._report, self.is_cancelled)
        except Exception as exc:  # noqa: BLE001
            from cmfqsar.kernels import Cancelled
            if isinstance(exc, Cancelled):
                self.failed.emit("Cancelled")
            else:
                self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")
            return
        self.finished_ok.emit(result)

    def _report(self, message: str, fraction: float) -> None:
        self.progress.emit(str(message), float(fraction))
