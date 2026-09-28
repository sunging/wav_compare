"""Sample-exact selection state and a responsive, multi-format editor."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, localcontext

from PySide6 import QtCore, QtWidgets

FORMATS = {"seconds": "Seconds", "clock": "hh:mm:ss", "samples": "Samples"}
MODES = {
    "start_end": ("Start / end", ("Start", "End")),
    "start_length": ("Start / length", ("Start", "Length")),
    "length_end": ("Length / end", ("Length", "End")),
    "center_length": ("Center / length", ("Center", "Length")),
}


def clamp(value, low, high):
    return max(low, min(value, high))


@dataclass
class SampleSelection:
    rate: int = 1
    frames: int = 0
    start: int = 0
    end: int = 0

    @property
    def bounds(self):
        return self.start, self.end

    @property
    def seconds(self):
        return self.start / self.rate, self.end / self.rate

    def set_context(self, rate, frames):
        self.rate, self.frames = max(1, int(rate)), max(0, int(frames))
        self.set_bounds(0, self.frames)

    def set_bounds(self, start, end):
        if not self.frames:
            self.start = self.end = 0
            return
        lo, hi = sorted((int(start), int(end)))
        self.start = clamp(lo, 0, self.frames - 1)
        self.end = clamp(hi, self.start + 1, self.frames)

    def values(self, mode):
        values = {
            "Start": Decimal(self.start),
            "End": Decimal(self.end),
            "Length": Decimal(self.end - self.start),
            "Center": Decimal(self.start + self.end) / 2,
        }
        return tuple(values[name] for name in MODES[mode][1])

    def edit(self, mode, index, value):
        if not self.frames:
            return
        start, end = self.bounds
        length = end - start
        number = round(value)
        field = MODES[mode][1][index]
        if mode == "start_end":
            if field == "Start":
                start = clamp(number, 0, end - 1)
            else:
                end = clamp(number, start + 1, self.frames)
        elif mode == "start_length":
            if field == "Start":
                start = clamp(number, 0, self.frames - length)
                end = start + length
            else:
                end = start + clamp(number, 1, self.frames - start)
        elif mode == "length_end":
            if field == "End":
                end = clamp(number, length, self.frames)
                start = end - length
            else:
                start = end - clamp(number, 1, end)
        else:
            if field == "Center":
                center = Decimal(round(value * 2)) / 2
            else:
                center = Decimal(start + end) / 2
                length = clamp(number, 1, min(start + end, 2 * self.frames - start - end))
            # Round half up (not half-even) so odd lengths always shift the center by +0.5.
            start = math.floor(center - Decimal(length) / 2 + Decimal("0.5"))
            start = clamp(start, 0, self.frames - length)
            end = start + length
        self.start, self.end = start, end


def time_digits(rate):
    # Rounding error must be smaller than a quarter sample, including half-sample centers.
    return max(6, len(str(4 * rate)))


def format_value(samples, value_format, rate):
    if value_format == "samples":
        return format(samples, "f").rstrip("0").rstrip(".") if samples % 1 else str(int(samples))
    digits = time_digits(rate)
    with localcontext() as context:
        context.prec = 60
        seconds = samples / rate
        if value_format == "seconds":
            return f"{seconds:.{digits}f}"
        ticks = round(seconds * 10**digits)
        whole, fraction = divmod(ticks, 10**digits)
        hours, remaining = divmod(whole, 3600)
        minutes, seconds = divmod(remaining, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction:0{digits}d}"


def parse_value(text, value_format, rate, center=False):
    text = text.strip()
    with localcontext() as context:
        context.prec = 60
        if value_format == "clock":
            if not re.fullmatch(r"\d+:[0-5]\d:[0-5]\d(?:\.\d+)?", text):
                raise ValueError("Invalid selection value")
            hours, minutes, seconds = text.split(":")
            value = (Decimal(hours) * 3600 + Decimal(minutes) * 60 + Decimal(seconds)) * rate
        else:
            if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", text):
                raise ValueError("Invalid selection value")
            value = Decimal(text)
            if value_format == "samples":
                if value * (2 if center else 1) % 1:
                    raise ValueError("Invalid selection value")
            else:
                value *= rate
        return Decimal(round(value * 2)) / 2 if center else Decimal(round(value))


class SelectionField(QtWidgets.QAbstractSpinBox):
    edited = QtCore.Signal(object)
    invalid = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.samples = Decimal(0)
        self.rate, self.frames = 1, 0
        self.value_format = "seconds"
        self.center = False
        self.rendered = ""
        self.setKeyboardTracking(False)
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        self.editingFinished.connect(self.commit)

    def display(self, samples, value_format, rate, frames, center=False):
        self.samples, self.value_format = samples, value_format
        self.rate, self.frames, self.center = rate, frames, center
        self.rendered = format_value(samples, value_format, rate)
        self.lineEdit().setText(self.rendered)
        self.lineEdit().setCursorPosition(0)
        self.setEnabled(frames > 0)
        self.updateGeometry()

    def commit(self):
        if not self.isEnabled() or self.lineEdit().text() == self.rendered:
            return
        try:
            value = parse_value(self.lineEdit().text(), self.value_format, self.rate, self.center)
        except (ValueError, InvalidOperation, OverflowError):
            self.lineEdit().setText(self.rendered)
            self.invalid.emit()
            return
        self.edited.emit(value)

    def stepBy(self, steps):
        self.commit()
        self.edited.emit(self.samples + steps)

    def stepEnabled(self):
        if not self.isEnabled():
            return QtWidgets.QAbstractSpinBox.StepNone
        return QtWidgets.QAbstractSpinBox.StepUpEnabled | QtWidgets.QAbstractSpinBox.StepDownEnabled

    def sizeHint(self):
        longest = format_value(Decimal(self.frames) + Decimal("0.5"), self.value_format, self.rate)
        width = self.fontMetrics().horizontalAdvance(longest) + 34
        return QtCore.QSize(
            width, max(super().sizeHint().height(), self.fontMetrics().height() + 14)
        )

    def minimumSizeHint(self):
        return self.sizeHint()


class SelectionEditor(QtWidgets.QWidget):
    selectionEdited = QtCore.Signal(object, object)
    preferencesChanged = QtCore.Signal()
    message = QtCore.Signal(str)

    def __init__(self, translate, actions=(), parent=None):
        super().__init__(parent)
        self.tr_text = translate
        self.state = SampleSelection()
        self.value_format, self.mode = "seconds", "start_end"
        self.wrapped = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.reflow_timer = QtCore.QTimer(self)
        self.reflow_timer.setSingleShot(True)
        self.reflow_timer.timeout.connect(self.reflow)
        self.grid = QtWidgets.QGridLayout(self)
        self.grid.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(6)
        self.selectors = QtWidgets.QWidget()
        selectors = QtWidgets.QHBoxLayout(self.selectors)
        selectors.setContentsMargins(0, 0, 0, 0)
        selectors.setSpacing(4)
        self.format_combo, self.mode_combo = QtWidgets.QComboBox(), QtWidgets.QComboBox()
        for key, name in FORMATS.items():
            self.format_combo.addItem(name, key)
        for key, (name, _) in MODES.items():
            self.mode_combo.addItem(name, key)
        selectors.addWidget(self.format_combo)
        selectors.addWidget(self.mode_combo)
        self.values_panel = QtWidgets.QWidget()
        values = QtWidgets.QHBoxLayout(self.values_panel)
        values.setContentsMargins(0, 0, 0, 0)
        values.setSpacing(4)
        self.fields, self.labels = [], []
        for index in range(2):
            group = QtWidgets.QHBoxLayout()
            group.setSpacing(3)
            label, field = QtWidgets.QLabel(), SelectionField()
            label.setBuddy(field)
            group.addWidget(label)
            group.addWidget(field)
            values.addLayout(group)
            self.fields.append(field)
            self.labels.append(label)
            field.edited.connect(lambda value, i=index: self.edit(i, value))
            field.invalid.connect(lambda: self.message.emit("Invalid selection value; restored."))
        self.action_button = QtWidgets.QToolButton()
        self.action_button.setText("⋯")
        self.action_button.setFixedWidth(28)
        self.action_button.setStyleSheet(
            "QToolButton { padding: 2px; } QToolButton::menu-indicator { image: none; width: 0px; }"
        )
        self.action_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(self.action_button)
        menu.addActions(actions)
        self.action_button.setMenu(menu)
        values.addWidget(self.action_button)
        self.format_combo.currentIndexChanged.connect(self.display_changed)
        self.mode_combo.currentIndexChanged.connect(self.display_changed)
        self.retranslate()
        self.reflow()

    @property
    def bounds(self):
        return self.state.bounds

    @property
    def seconds(self):
        return self.state.seconds

    def set_context(self, rate, frames):
        self.state.set_context(rate, frames)
        self.render()

    def set_selection(self, start, end):
        self.state.set_bounds(start, end)
        self.render()

    def set_preferences(self, value_format, mode):
        with QtCore.QSignalBlocker(self.format_combo), QtCore.QSignalBlocker(self.mode_combo):
            self.format_combo.setCurrentIndex(max(0, self.format_combo.findData(value_format)))
            self.mode_combo.setCurrentIndex(max(0, self.mode_combo.findData(mode)))
        self.display_changed()

    def display_changed(self, *_):
        self.value_format = self.format_combo.currentData()
        self.mode = self.mode_combo.currentData()
        self.render()
        self.preferencesChanged.emit()

    def edit(self, index, value):
        before = self.bounds
        self.state.edit(self.mode, index, value)
        self.render()
        if self.bounds != before:
            self.selectionEdited.emit(*self.bounds)

    def render(self):
        for label, field, name, value in zip(
            self.labels, self.fields, MODES[self.mode][1], self.state.values(self.mode), strict=True
        ):
            label.setText(self.tr_text(name))
            field.setAccessibleName(self.tr_text(name))
            field.display(
                value,
                self.value_format,
                self.state.rate,
                self.state.frames,
                center=name == "Center",
            )
        self.reflow_timer.start(0)

    def retranslate(self):
        for combo, names in (
            (self.format_combo, FORMATS),
            (self.mode_combo, {key: name for key, (name, _) in MODES.items()}),
        ):
            with QtCore.QSignalBlocker(combo):
                for i in range(combo.count()):
                    combo.setItemText(i, self.tr_text(names[combo.itemData(i)]))
        for widget, text in (
            (self.format_combo, "Selection format"),
            (self.mode_combo, "Selection type"),
            (self.action_button, "Selection"),
        ):
            widget.setAccessibleName(self.tr_text(text))
            widget.setToolTip(self.tr_text(text))
        for field in self.fields:
            field.setToolTip(
                self.tr_text(
                    "Positions use the current analysis sample rate. End is exclusive. "
                    "Commit with Enter; arrows step one sample. Centers may use half samples."
                )
            )
        self.render()

    def reflow(self):
        full_width = self.selectors.sizeHint().width() + self.values_panel.sizeHint().width() + 6
        wrapped = self.width() < full_width
        if wrapped != self.wrapped:
            self.wrapped = wrapped
            self.grid.removeWidget(self.selectors)
            self.grid.removeWidget(self.values_panel)
            self.grid.addWidget(self.selectors, 0, 0, 1, 1, QtCore.Qt.AlignLeft)
            self.grid.addWidget(
                self.values_panel,
                1 if wrapped else 0,
                0 if wrapped else 1,
                1,
                1,
                QtCore.Qt.AlignLeft,
            )
            self.grid.setColumnStretch(1, 1 if wrapped else 0)
            self.grid.setColumnStretch(2, 0 if wrapped else 1)
        # Field hints change with format/language; recompute cells now instead of relying on
        # a posted layout request, which can leave a stale, clipped column in place.
        self.grid.invalidate()
        self.grid.activate()
        self.updateGeometry()

    def minimumSizeHint(self):
        return QtCore.QSize(
            max(
                self.selectors.minimumSizeHint().width(),
                self.values_panel.minimumSizeHint().width(),
            ),
            self.sizeHint().height(),
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.reflow()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() in (QtCore.QEvent.FontChange, QtCore.QEvent.StyleChange):
            self.reflow_timer.start(0)
