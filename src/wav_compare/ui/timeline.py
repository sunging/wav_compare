"""Playback position and selection on a shared, seconds-based timeline."""

from PySide6 import QtCore, QtGui, QtWidgets


class PlaybackTimeline(QtWidgets.QWidget):
    seekRequested = QtCore.Signal(float)
    selectionPreview = QtCore.Signal(float, float)
    selectionCommitted = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.duration = 0.0
        self.step = 1.0
        self.position = 0.0
        self.selection = (0.0, 0.0)
        self.seek_bounds = (0.0, 0.0)
        self.interacting = False
        self._dragging = False
        self._edge = None
        self.select_all_text = "Select all"
        self.foreground = self.palette().color(QtGui.QPalette.WindowText)
        self.track_color = self.palette().color(QtGui.QPalette.Mid)
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(120, 32)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setEnabled(False)

    def set_duration(self, duration, rate):
        self.duration = max(0.0, duration)
        self.step = min(1 / rate, self.duration) if self.duration else 0.0
        self.position = 0.0
        self.selection = (0.0, self.duration)
        self.seek_bounds = (0.0, self.duration)
        self.interacting = False
        self.setEnabled(self.duration > 0)
        self.update()

    def set_position(self, seconds):
        self.position = max(0.0, min(seconds, self.duration))
        self.update()

    def set_selection(self, lo, hi):
        lo, hi = sorted((lo, hi))
        lo = max(0.0, min(lo, self.duration - self.step))
        hi = max(lo + self.step, min(hi, self.duration))
        self.selection = (lo, hi)
        self.update()

    def x_at(self, seconds):
        return 9 + (self.width() - 18) * seconds / self.duration if self.duration else 9

    def time_at(self, x):
        return max(0.0, min(1.0, (x - 9) / max(1, self.width() - 18))) * self.duration

    def _seek(self, seconds):
        lo, hi = self.seek_bounds
        self.seekRequested.emit(max(lo, min(hi, seconds)))

    def mousePressEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton or not self.isEnabled():
            return super().mousePressEvent(event)
        self.setFocus(QtCore.Qt.MouseFocusReason)
        self.interacting = True
        self._dragging = False
        self._press = event.position()
        distances = [abs(self.x_at(t) - self._press.x()) for t in self.selection]
        edge = min(range(2), key=lambda i: distances[i])
        self._edge = edge if distances[edge] <= 6 else None
        event.accept()

    def mouseMoveEvent(self, event):
        if not self.interacting:
            return super().mouseMoveEvent(event)
        if (event.position() - self._press).manhattanLength() >= (
            QtWidgets.QApplication.startDragDistance()
        ):
            self._dragging = True
        if self._dragging:
            self._preview(event.position().x())
        event.accept()

    def _preview(self, x):
        time = self.time_at(x)
        lo, hi = self.selection
        if self._edge == 0:
            lo = min(time, hi - self.step)
        elif self._edge == 1:
            hi = max(time, lo + self.step)
        else:
            lo, hi = sorted((self.time_at(self._press.x()), time))
        self.set_selection(lo, hi)
        self.selectionPreview.emit(*self.selection)

    def mouseReleaseEvent(self, event):
        if event.button() != QtCore.Qt.LeftButton or not self.interacting:
            return super().mouseReleaseEvent(event)
        if (event.position() - self._press).manhattanLength() >= (
            QtWidgets.QApplication.startDragDistance()
        ):
            self._dragging = True
        if self._dragging:
            self._preview(event.position().x())
        self.interacting = False
        if self._dragging:
            self.selectionCommitted.emit(*self.selection)
        else:
            self._seek(self.time_at(event.position().x()))
        event.accept()

    def select_all(self):
        self.set_selection(0, self.duration)
        self.selectionPreview.emit(*self.selection)
        self.selectionCommitted.emit(*self.selection)

    def contextMenuEvent(self, event):
        menu = QtWidgets.QMenu(self)
        menu.addAction(self.select_all_text, self.select_all)
        menu.exec(event.globalPos())

    def keyPressEvent(self, event):
        keys = {
            QtCore.Qt.Key_Left: self.position - 0.1,
            QtCore.Qt.Key_Right: self.position + 0.1,
            QtCore.Qt.Key_Home: self.seek_bounds[0],
            QtCore.Qt.Key_End: self.seek_bounds[1],
        }
        if event.key() in keys:
            self._seek(keys[event.key()])
            event.accept()
        else:
            super().keyPressEvent(event)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        fg = self.foreground
        accent = QtGui.QColor("#4298d8")
        if not self.isEnabled():
            painter.setOpacity(0.4)
        y = self.height() / 2
        painter.setPen(QtGui.QPen(self.track_color, 4))
        painter.drawLine(QtCore.QPointF(9, y), QtCore.QPointF(self.width() - 9, y))
        lo, hi = (self.x_at(t) for t in self.selection)
        fill = QtGui.QColor(accent)
        fill.setAlpha(80)
        painter.fillRect(QtCore.QRectF(lo, y - 8, hi - lo, 16), fill)
        painter.setPen(QtGui.QPen(accent, 3))
        for x in (lo, hi):
            painter.drawLine(QtCore.QPointF(x, y - 10), QtCore.QPointF(x, y + 10))
        x = self.x_at(self.position)
        painter.setPen(QtGui.QPen(fg, 2))
        painter.drawLine(QtCore.QPointF(x, y - 12), QtCore.QPointF(x, y + 12))
        painter.setBrush(fg)
        painter.drawEllipse(QtCore.QPointF(x, y), 3, 3)
        if self.hasFocus():
            painter.setPen(QtGui.QPen(accent, 1, QtCore.Qt.DotLine))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 3, 3)

    def apply_theme(self, dark):
        self.foreground = QtGui.QColor("#e2e9f3" if dark else "#1e293b")
        self.track_color = QtGui.QColor("#344156" if dark else "#d1dbe8")
        self.update()
