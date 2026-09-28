from __future__ import annotations

import traceback

from PySide6 import QtCore

from ..models import Cancellation, Cancelled


class Signals(QtCore.QObject):
    done = QtCore.Signal(int, str, object)
    progress = QtCore.Signal(int, float, str)
    partial = QtCore.Signal(int, str, object)


class Job(QtCore.QRunnable):
    def __init__(self, identity, kind, function, signals, streaming=False):
        super().__init__()
        self.identity, self.kind, self.function, self.signals = identity, kind, function, signals
        self.streaming = streaming
        self.cancel = Cancellation()

    def run(self):
        try:
            self.cancel.check()
            arguments = [self.cancel, lambda p, s: self.signals.progress.emit(self.identity, p, s)]
            if self.streaming:
                # Intermediate values reach the GUI thread while the job keeps running.
                arguments.append(lambda v: self.signals.partial.emit(self.identity, self.kind, v))
            value = self.function(*arguments)
            self.signals.done.emit(self.identity, self.kind, (True, value))
        except Cancelled:
            self.signals.done.emit(self.identity, self.kind, (False, "Cancelled"))
        except Exception as error:
            traceback.print_exc()
            self.signals.done.emit(self.identity, self.kind, (False, str(error)))


class Jobs(QtCore.QObject):
    done = QtCore.Signal(str, object)
    progress = QtCore.Signal(float, str)
    partial = QtCore.Signal(str, object)
    idle = QtCore.Signal()
    activityChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QtCore.QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self.signals = Signals(self)
        self.signals.done.connect(self._done)
        self.signals.progress.connect(self._progress)
        self.signals.partial.connect(self._partial)
        self.sequence = 0
        self.jobs = {}
        self.latest = {}
        self.accepting = True

    def submit(self, kind, function, *, streaming=False):
        """Run function(cancel, progress[, emit]) on the pool; only the latest kind reports."""
        if not self.accepting:
            return
        old = self.latest.get(kind)
        if old in self.jobs:
            self.jobs[old].cancel.cancel()
        self.sequence += 1
        identity = self.sequence
        job = Job(identity, kind, function, self.signals, streaming)
        self.jobs[identity] = job
        self.latest[kind] = identity
        self.pool.start(job)
        self.activityChanged.emit()

    def cancel_all(self):
        for job in self.jobs.values():
            job.cancel.cancel()
        self.latest.clear()
        self.activityChanged.emit()

    def cancel_kind(self, kind):
        identity = self.latest.pop(kind, None)
        if identity in self.jobs:
            self.jobs[identity].cancel.cancel()
        self.activityChanged.emit()

    def _progress(self, identity, value, message):
        if identity in self.latest.values():
            self.progress.emit(value, message)

    def _partial(self, identity, kind, value):
        if self.latest.get(kind) == identity:
            self.partial.emit(kind, value)

    def _done(self, identity, kind, result):
        self.jobs.pop(identity, None)
        if self.latest.get(kind) == identity:
            self.latest.pop(kind)
            self.done.emit(kind, result)
        self.activityChanged.emit()
        if not self.jobs:
            self.idle.emit()
