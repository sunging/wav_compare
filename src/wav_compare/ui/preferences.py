"""Validated application preferences; the analysis engine remains Qt independent."""

from __future__ import annotations

import json
import math

from PySide6 import QtWidgets


class Preferences:
    def __init__(self, settings):
        self.settings = settings

    def boolean(self, key, default=True):
        value = self.settings.value(key, default)
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "1"):
            return True
        if str(value).lower() in ("false", "0"):
            return False
        return default

    def choice(self, key, choices, default):
        value = self.settings.value(key, default)
        return value if value in choices else default

    def number(self, key, default, low, high, *, integer=True):
        return self.valid_number(self.settings.value(key, default), default, low, high, integer)

    @staticmethod
    def valid_number(value, default, low, high, integer=False):
        try:
            number = float(value)
            if not math.isfinite(number) or not low <= number <= high:
                return default
            if integer and not number.is_integer():
                return default
            return int(number) if integer else number
        except (TypeError, ValueError, OverflowError):
            return default

    def object(self, key):
        try:
            value = json.loads(self.settings.value(key, "{}"))
            return value if isinstance(value, dict) else {}
        except (ValueError, TypeError):
            return {}


class PreferencesDialog(QtWidgets.QDialog):
    """Edit a copy; the window applies it only after acceptance."""

    def __init__(self, window):
        super().__init__(window)
        tr = window.tr
        self.setWindowTitle(tr("Preferences…"))
        layout = QtWidgets.QFormLayout(self)
        self.language = QtWidgets.QComboBox()
        for text, data in (("中文", "zh"), ("English", "en")):
            self.language.addItem(text, data)
        self.language.setCurrentIndex(self.language.findData(window.language))
        self.theme = QtWidgets.QComboBox()
        for name in ("System", "Light", "Dark"):
            self.theme.addItem(tr(name), name.lower())
        self.theme.setCurrentIndex(self.theme.findData(window.theme_combo.currentData()))
        self.auto_compare = QtWidgets.QCheckBox(tr("Automatically compare changes"))
        self.auto_compare.setChecked(window.auto_compare_action.isChecked())
        self.startup = QtWidgets.QCheckBox(tr("Compare automatically on startup"))
        self.startup.setChecked(window.startup_compare)
        self.startup.setEnabled(self.auto_compare.isChecked())
        self.auto_compare.toggled.connect(self.startup.setEnabled)
        self.restore_paths = QtWidgets.QCheckBox(tr("Restore previous paths"))
        self.restore_paths.setChecked(window.restore_paths)
        layout.addRow(tr("Language"), self.language)
        layout.addRow(tr("Theme"), self.theme)
        for widget in (self.auto_compare, self.startup, self.restore_paths):
            layout.addRow(widget)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.button(QtWidgets.QDialogButtonBox.Ok).setText(tr("OK"))
        buttons.button(QtWidgets.QDialogButtonBox.Cancel).setText(tr("Cancel"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addRow(buttons)
