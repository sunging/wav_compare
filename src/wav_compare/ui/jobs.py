from __future__ import annotations

import traceback

from PySide6 import QtCore

from ..models import Cancellation, Cancelled


class Signals(QtCore.QObject):
    done = QtCore.Signal(int, str, object)
    progress = QtCore.Signal(int, float, str)


class Job(QtCore.QRunnable):
    def __init__(self, identity, kind, function, signals):
        super().__init__()
        self.identity, self.kind, self.function, self.signals = identity, kind, function, signals
        self.cancel = Cancellation()

    def run(self):
        try:
            self.cancel.check()
            value = self.function(
                self.cancel, lambda p, s: self.signals.progress.emit(self.identity, p, s)
            )
            self.signals.done.emit(self.identity, self.kind, (True, value))
        except Cancelled:
            self.signals.done.emit(self.identity, self.kind, (False, "Cancelled"))
        except Exception as error:
            traceback.print_exc()
            self.signals.done.emit(self.identity, self.kind, (False, str(error)))


class Jobs(QtCore.QObject):
    done = QtCore.Signal(str, object)
    progress = QtCore.Signal(float, str)
    idle = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.signals = Signals(self)
        self.signals.done.connect(self._done)
        self.signals.progress.connect(self._progress)
        self.sequence = 0
        self.jobs = {}
        self.latest = {}

    def submit(self, kind, function):
        old = self.latest.get(kind)
        if old in self.jobs:
            self.jobs[old].cancel.cancel()
        self.sequence += 1
        identity = self.sequence
        job = Job(identity, kind, function, self.signals)
        self.jobs[identity] = job
        self.latest[kind] = identity
        self.pool.start(job)

    def cancel_all(self):
        for job in self.jobs.values():
            job.cancel.cancel()
        self.latest.clear()

    def cancel_kind(self, kind):
        identity = self.latest.pop(kind, None)
        if identity in self.jobs:
            self.jobs[identity].cancel.cancel()

    def _progress(self, identity, value, message):
        if identity in self.latest.values():
            self.progress.emit(value, message)

    def _done(self, identity, kind, result):
        self.jobs.pop(identity, None)
        if self.latest.get(kind) == identity:
            self.done.emit(kind, result)
        if not self.jobs:
            self.idle.emit()
