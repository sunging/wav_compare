"""Independent, non-intercepting plot probes with optional pinned readings."""

from __future__ import annotations

import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets


class Readout(QtWidgets.QLabel):
    """Wrap at the allocated width without inflating a scroll area's size hint."""

    def __init__(self):
        super().__init__()
        self.setWordWrap(True)
        self.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setFixedHeight(34)

    def setText(self, text):
        super().setText(text)
        # QLabel enables height-for-width again on setText. In a QScrollArea that
        # propagates the PlotWidget's preferred 480px height to the whole page.
        policy = self.sizePolicy()
        policy.setHeightForWidth(False)
        self.setSizePolicy(policy)
        self.fit_height()

    def clear(self):
        self.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit_height()

    def fit_height(self):
        width = max(1, self.contentsRect().width())
        bounds = self.fontMetrics().boundingRect(
            QtCore.QRect(0, 0, width, 10000),
            QtCore.Qt.TextWordWrap,
            self.text(),
        )
        # Pending reads clear the text briefly; keep the plot still while hovering.
        minimum = self.height() if width == getattr(self, "last_width", None) else 34
        self.last_width = width
        self.setFixedHeight(max(34, minimum, bounds.height() + 2))


class PlotIndicator(QtCore.QObject):
    requested = QtCore.Signal(int, object)
    invalidated = QtCore.Signal()

    def __init__(self, plot, translate, *, crosshair=False):
        super().__init__(plot)
        self.plot, self.translate = plot, translate
        self.revision = 0
        self.fixed = False
        self.fields = []
        self.inside = False
        self.panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(self.panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(plot, 1)
        footer = QtWidgets.QHBoxLayout()
        self.toggle = QtWidgets.QCheckBox()
        self.toggle.setChecked(True)
        footer.addWidget(self.toggle, 0, QtCore.Qt.AlignTop)
        self.readout = Readout()
        footer.addWidget(self.readout, 1)
        layout.addLayout(footer)
        self.lines = []
        for angle in (90, 0) if crosshair else (90,):
            line = pg.InfiniteLine(angle=angle, movable=False)
            line.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            line.setZValue(20)
            plot.addItem(line, ignoreBounds=True)
            line.hide()
            self.lines.append(line)
        self.toggle.toggled.connect(self.reset)
        plot.viewport().installEventFilter(self)
        plot.scene().sigMouseMoved.connect(self.track_inside)
        self.proxy = pg.SignalProxy(plot.scene().sigMouseMoved, rateLimit=15, slot=self.move)
        plot.scene().sigMouseClicked.connect(self.click)
        self.retranslate()

    def contains(self, position):
        return self.plot.getViewBox().sceneBoundingRect().contains(position)

    def track_inside(self, position):
        self.inside = self.contains(position)
        if not self.inside and not self.fixed:
            self.reset()

    def eventFilter(self, watched, event):
        if event.type() == QtCore.QEvent.Leave:
            self.inside = False
            if not self.fixed:
                self.reset()
        return False

    def reset(self, *_):
        self.revision += 1
        self.fixed = False
        self.fields = []
        self.invalidated.emit()
        for line in self.lines:
            line.hide()
        self.readout.clear()
        self.readout.setVisible(self.toggle.isChecked())

    def move(self, event):
        if self.inside and not self.fixed and not QtWidgets.QApplication.mouseButtons():
            self.query(event[0])

    def query(self, position):
        if not self.toggle.isChecked() or not self.contains(position):
            return
        self.revision += 1
        self.invalidated.emit()
        self.fields = []
        for line in self.lines:
            line.hide()
        self.render()
        self.requested.emit(self.revision, self.plot.getViewBox().mapSceneToView(position))

    def click(self, event):
        if (
            event.button() != QtCore.Qt.LeftButton
            or event.isAccepted()
            or event.double()
            or not self.toggle.isChecked()
            or not self.contains(event.scenePos())
        ):
            return
        was_fixed = self.fixed
        self.reset()
        self.query(event.scenePos())
        # Only pin positions for which a query was accepted (including pending I/O).
        self.fixed = not was_fixed and self.pending
        self.render()

    @property
    def pending(self):
        return getattr(self, "accepted_revision", None) == self.revision

    def accept(self, revision):
        self.accepted_revision = revision

    def finish(self, revision, x, fields, y=None):
        if revision != self.revision or not self.toggle.isChecked():
            return
        self.fields = fields
        self.lines[0].setPos(x)
        if y is not None and len(self.lines) > 1:
            self.lines[1].setPos(y)
        for line in self.lines:
            line.show()
        self.render()

    def render(self):
        parts = [f"{self.translate(key)}: {value}" for key, value in self.fields]
        if self.fixed:
            parts.insert(0, self.translate("Pinned"))
        self.readout.setText("   |   ".join(parts))

    def retranslate(self):
        self.toggle.setText(self.translate("Show indicator"))
        hint = self.translate("Hover for values; click to pin or unpin. Drag to pan.")
        self.toggle.setToolTip(hint)
        self.readout.setToolTip(hint)
        self.render()

    def apply_theme(self, dark):
        pen = pg.mkPen("#e5b4ff" if dark else "#7630a8", style=QtCore.Qt.DotLine)
        for line in self.lines:
            line.setPen(pen)
