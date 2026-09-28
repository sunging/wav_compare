"""Grouped analysis/display parameters with dependency-aware enabling and validation."""

from __future__ import annotations

import json

from PySide6 import QtCore, QtWidgets

from ..cli import channel_pairs
from ..models import Options

NUMERIC_MODES = ("float64", "float32", "pcm16", "pcm24", "pcm32")
FFT_SIZES = ("256", "512", "1024", "2048", "4096", "8192", "16384")
INT_LIMIT = 2147483647


class ParametersPanel(QtWidgets.QWidget):
    """Owns the analysis controls; the window reads options and persists snapshots."""

    optionsChanged = QtCore.Signal()
    spectrumOptionsChanged = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.texts = []
        self.tr_text = str
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 6, 2)
        layout.setSpacing(6)

        self.strict = self._text(QtWidgets.QCheckBox(), "Strict comparison")
        self.align = self._text(QtWidgets.QCheckBox(), "Auto alignment")
        self.align.setChecked(True)
        self.manual = self._text(QtWidgets.QCheckBox(), "Manual offset")
        self.mix = self._text(QtWidgets.QCheckBox(), "Mix to mono")
        self.max_lag = QtWidgets.QDoubleSpinBox()
        self.max_lag.setRange(0, 300)
        self.max_lag.setValue(5)
        self.offset = QtWidgets.QSpinBox()
        self.offset.setRange(-INT_LIMIT, INT_LIMIT)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(NUMERIC_MODES)
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(12)
        self.threshold.setRange(0, 4294967295)
        self.segment = QtWidgets.QSpinBox()
        self.segment.setRange(1, INT_LIMIT)
        self.segment.setValue(1000)
        self.pairs = QtWidgets.QLineEdit()
        self.pairs.setPlaceholderText("1:1,2:2")
        self.pairs.setClearButtonEnabled(True)
        self.channel = QtWidgets.QComboBox()
        self.channel.addItem("1")
        self.fft = QtWidgets.QComboBox()
        self.fft.addItems(FFT_SIZES)
        self.fft.setCurrentText("2048")
        self.hop = QtWidgets.QSpinBox()
        self.hop.setRange(1, 16384)
        self.hop.setValue(512)

        alignment = self._group(layout, "Alignment options")
        alignment.addRow(self.strict)
        alignment.addRow(self.align)
        alignment.addRow(self._label("Max lag (s)"), self.max_lag)
        alignment.addRow(self.manual)
        alignment.addRow(self._label("Offset (samples)"), self.offset)
        comparison = self._group(layout, "Comparison options")
        for text, control in (
            ("Numeric mode", self.mode),
            ("Threshold", self.threshold),
            ("Segment (samples)", self.segment),
            ("Channel pairs", self.pairs),
        ):
            comparison.addRow(self._label(text), control)
        comparison.addRow(self.mix)
        display = self._group(layout, "Display options")
        for text, control in (
            ("Display channel", self.channel),
            ("FFT size", self.fft),
            ("Hop size", self.hop),
        ):
            display.addRow(self._label(text), control)
        layout.addStretch()

        for control in (self.strict, self.align, self.manual, self.mix):
            control.toggled.connect(self.update_state)
            control.toggled.connect(self.optionsChanged)
        for control in (self.max_lag, self.offset, self.threshold, self.segment):
            control.valueChanged.connect(self.optionsChanged)
        self.mode.currentIndexChanged.connect(self.optionsChanged)
        self.pairs.textChanged.connect(self.update_state)
        self.pairs.textChanged.connect(self.optionsChanged)
        self.fft.currentIndexChanged.connect(self.spectrumOptionsChanged)
        self.hop.valueChanged.connect(self.spectrumOptionsChanged)
        self.update_state()

    def _text(self, widget, text, method="setText"):
        self.texts.append((widget, text, method))
        return widget

    def _label(self, text):
        return self._text(QtWidgets.QLabel(), text)

    def _group(self, layout, title):
        box = self._text(QtWidgets.QGroupBox(), title, "setTitle")
        form = QtWidgets.QFormLayout(box)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setContentsMargins(6, 4, 6, 6)
        layout.addWidget(box)
        return form

    def retranslate(self, translate):
        self.tr_text = translate
        for widget, text, method in self.texts:
            getattr(widget, method)(translate(text))
        self.update_state()

    def pairs_error(self):
        try:
            pairs = channel_pairs(self.pairs.text())
        except ValueError as error:
            return str(error)
        if pairs and self.mix.isChecked():
            return "Mix and channel pairs are mutually exclusive"
        return None

    def update_state(self, *_):
        """Disable controls the current mode ignores and flag invalid channel pairs."""
        strict, manual = self.strict.isChecked(), self.manual.isChecked()
        self.align.setEnabled(not strict and not manual)
        self.manual.setEnabled(not strict)
        self.offset.setEnabled(not strict and manual)
        self.max_lag.setEnabled(not strict and not manual and self.align.isChecked())
        error = self.pairs_error()
        if self.pairs.property("invalid") != bool(error):
            self.pairs.setProperty("invalid", bool(error))
            self.pairs.style().unpolish(self.pairs)
            self.pairs.style().polish(self.pairs)
        self.pairs.setToolTip(
            self.tr_text(error) if error else self.tr_text("1-based A:B pairs, e.g. 1:1,2:2")
        )

    def options(self, region=None):
        options = Options(
            strict=self.strict.isChecked(),
            align=self.align.isChecked(),
            max_lag=self.max_lag.value(),
            offset=self.offset.value()
            if self.manual.isChecked() and not self.strict.isChecked()
            else None,
            threshold=self.threshold.value(),
            segment_size=self.segment.value(),
            mode=self.mode.currentText(),
            pairs=channel_pairs(self.pairs.text()),
            mix=self.mix.isChecked(),
            region=region,
        )
        options.validate()
        return options

    def restore(self, prefs):
        options = prefs.object("options")
        for name in ("strict", "align", "mix"):
            value = options.get(name, name == "align")
            getattr(self, name).setChecked(value if isinstance(value, bool) else name == "align")
        for name, key in (
            ("max_lag", "max_lag"),
            ("threshold", "threshold"),
            ("segment", "segment_size"),
        ):
            control = getattr(self, name)
            control.setValue(
                prefs.valid_number(
                    options.get(key),
                    control.value(),
                    control.minimum(),
                    control.maximum(),
                    isinstance(control, QtWidgets.QSpinBox),
                )
            )
        if options.get("mode") in NUMERIC_MODES:
            self.mode.setCurrentText(options["mode"])
        self.manual.setChecked(prefs.boolean("analysis/manual", options.get("offset") is not None))
        old_offset = prefs.valid_number(options.get("offset"), 0, -INT_LIMIT, INT_LIMIT, True)
        self.offset.setValue(prefs.number("analysis/offset", old_offset, -INT_LIMIT, INT_LIMIT))
        try:
            pairs = options.get("pairs", [])
            if not isinstance(pairs, list) or any(
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(type(n) is not int or n < 0 for n in pair)
                for pair in pairs
            ):
                pairs = []
            self.pairs.setText(",".join(f"{a + 1}:{b + 1}" for a, b in pairs))
            self.options()
        except (ValueError, TypeError):
            self.pairs.clear()
        self.fft.setCurrentText(prefs.choice("spectrum/fft", FFT_SIZES, "2048"))
        self.hop.setValue(prefs.number("spectrum/hop", 512, 1, 16384))
        self.update_state()

    def settings_values(self):
        values = {"spectrum/fft": self.fft.currentText(), "spectrum/hop": self.hop.value()}
        # Only a valid analysis snapshot replaces the stored one.
        try:
            values["options"] = json.dumps(self.options().report())
            values["analysis/manual"] = self.manual.isChecked()
            values["analysis/offset"] = self.offset.value()
        except ValueError:
            pass
        return values
