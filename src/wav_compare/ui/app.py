from __future__ import annotations

import json
import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import get_version
from ..batch import discover, run_batch
from ..cli import channel_pairs
from ..engine import compare
from ..models import Options
from ..reports import export_csv, export_json
from ..views import ByteCache, amplitude_ranges, spectra, waveform
from .i18n import translate
from .indicator import PlotIndicator
from .jobs import Jobs
from .player import Player
from .preferences import Preferences, PreferencesDialog
from .selection import FORMATS, MODES, SelectionEditor
from .timeline import PlaybackTimeline

COLORS = ("#52b9f5", "#ffb454", "#74dbb0")


class Window(QtWidgets.QMainWindow):
    def __init__(self, paths=(), settings=None):
        super().__init__()
        self.settings = settings or QtCore.QSettings("sunging", "wav-compare")
        self.preferences = Preferences(self.settings)
        default_language = (
            "zh" if QtCore.QLocale.system().language() == QtCore.QLocale.Chinese else "en"
        )
        self.language = self.preferences.choice("language", ("zh", "en"), default_language)
        self.bindings = []
        self.result = None
        self.spectral_ranges = None
        self.spectral_data = None
        self.rows = []
        self.detail_cache = OrderedDict()
        self.view_cache = ByteCache()
        self.closing = False
        self.restoring = True
        self.preferred_row = None
        self.batch_cancelled = False
        self.panel_widths = [240, 300]
        self.startup_compare = self.preferences.boolean("automation/startup")
        self.restore_paths = self.preferences.boolean("restore_paths")
        self.compare_timer = QtCore.QTimer(self)
        self.compare_timer.setSingleShot(True)
        self.compare_timer.setInterval(400)
        self.compare_timer.timeout.connect(self.start_compare)
        self.spectrum_timer = QtCore.QTimer(self)
        self.spectrum_timer.setSingleShot(True)
        self.spectrum_timer.setInterval(400)
        self.spectrum_timer.timeout.connect(self.update_spectrum)
        self.save_timer = QtCore.QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.setInterval(500)
        self.save_timer.timeout.connect(self.save_preferences)
        self.jobs = Jobs(self)
        self.jobs.done.connect(self.job_done)
        self.jobs.progress.connect(self.progress_changed)
        self.jobs.idle.connect(self.on_idle)
        self.player = Player(self)
        self.player.error.connect(self.message)
        self.player.position.connect(self.play_position)
        self.playback_position = 0.0
        self.setWindowTitle("WAV Compare")
        self.resize(1440, 940)
        self.setMinimumSize(1000, 680)
        self.setAcceptDrops(True)
        self._build()
        self._restore(paths)
        self._connect()
        self.retranslate()
        self.apply_theme()
        self.player.set_volume(self.volume.value() / 100)
        self.restoring = False
        self.refresh_actions()
        self.message(
            "Choose two files or folders to compare automatically."
            if self.auto_compare_action.isChecked()
            else "Drop two WAV files or folders here, then compare."
        )
        if self.auto_compare_action.isChecked() and self.startup_compare:
            self.schedule_compare()

    def tr(self, text):
        return translate(text, self.language)

    def bind(self, widget, text, method="setText"):
        self.bindings.append((widget, text, method))
        getattr(widget, method)(self.tr(text))
        return widget

    def button(self, text, callback, parent_layout=None):
        button = self.bind(QtWidgets.QPushButton(), text)
        button.clicked.connect(callback)
        if parent_layout is not None:
            parent_layout.addWidget(button)
        return button

    def label(self, text):
        return self.bind(QtWidgets.QLabel(), text)

    def action(self, text, callback, shortcut=None, checkable=False):
        action = self.bind(QtGui.QAction(self), text)
        action.setCheckable(checkable)
        action.triggered.connect(callback)
        if shortcut:
            action.setShortcut(QtGui.QKeySequence(shortcut))
        self.addAction(action)
        return action

    def action_button(self, action, layout, compact=None):
        button = QtWidgets.QToolButton()
        button.setDefaultAction(action)
        if compact:
            self.bind(action, compact, "setIconText")
            self.bind(
                button,
                next(text for widget, text, _ in self.bindings if widget is action),
                "setToolTip",
            )
        layout.addWidget(button)
        return button

    def menu(self, title, parent=None):
        return self.bind((parent or self.menuBar()).addMenu(title), title, "setTitle")

    def _build_actions(self):
        files = self.menu("File")
        self.open_actions = []
        for side, label in enumerate(("Reference A", "Candidate B")):
            menu = self.menu(label, files)
            actions = [
                self.action("File…", lambda _, n=side: self.browse(n, False)),
                self.action("Folder…", lambda _, n=side: self.browse(n, True)),
            ]
            menu.addActions(actions)
            self.open_actions.append(actions)
        self.swap_action = self.action("Swap A / B", self.swap)
        files.addAction(self.swap_action)
        files.addSeparator()
        self.export_actions = [
            self.action("Export JSON…", lambda: self.export("json")),
            self.action("Export CSV…", lambda: self.export("csv")),
        ]
        files.addActions(self.export_actions)
        files.addSeparator()
        files.addAction(self.action("Exit", self.close, "Ctrl+Q"))
        analysis = self.menu("Analysis")
        self.compare_action = self.action("Compare", self.start_compare, "Ctrl+Return")
        self.cancel_action = self.action("Cancel", self.cancel_tasks, "Escape")
        analysis.addActions([self.compare_action, self.cancel_action])
        analysis.addSeparator()
        self.selection_actions = [
            self.action("Analyze selection", lambda: self.analyze_region(True)),
            self.action("Full comparison", lambda: self.analyze_region(False)),
            self.action("Update spectrum", self.update_spectrum),
        ]
        analysis.addActions(self.selection_actions)
        analysis.addSeparator()
        analysis.addAction(self.action("Clear cache", self.clear_cache))
        view = self.menu("View")
        self.panel_actions = [
            self.action(
                "Files & results", lambda checked: self.toggle_panel(0, checked), checkable=True
            ),
            self.action(
                "Parameters and metrics",
                lambda checked: self.toggle_panel(1, checked),
                checkable=True,
            ),
        ]
        for action in self.panel_actions:
            action.setChecked(True)
        view.addActions(self.panel_actions)
        view.addAction(self.action("Reset layout", self.reset_layout))
        view.addSeparator()
        self.navigation_actions = [
            self.action("Reset zoom", self.reset_zoom, "Ctrl+0"),
            self.action("Largest difference", self.largest),
            self.action("Previous difference", lambda: self.next_difference(-1)),
            self.action("Next difference", lambda: self.next_difference(1)),
        ]
        view.addActions(self.navigation_actions)
        settings = self.menu("Settings")
        self.auto_compare_action = self.action(
            "Automatically compare changes", self.auto_compare_changed, checkable=True
        )
        self.auto_compare_action.setChecked(self.preferences.boolean("automation/enabled"))
        settings.addAction(self.auto_compare_action)
        settings.addAction(self.action("Preferences…", self.edit_preferences))
        self.menu("Help").addAction(self.action("About", self.about))

    def panel_header(self, layout, index):
        header = QtWidgets.QHBoxLayout()
        title = self.label(("Files & results", "Parameters")[index])
        title.setWordWrap(True)
        header.addWidget(title)
        header.addStretch()
        button = QtWidgets.QToolButton()
        button.setText("‹" if index == 0 else "›")
        button.setFixedWidth(24)
        self.bind(button, "Collapse panel", "setToolTip")
        self.bind(button, "Collapse panel", "setAccessibleName")
        button.clicked.connect(lambda: self.panel_actions[index].trigger())
        header.addWidget(button)
        layout.addLayout(header)

    def _build(self):
        self._build_actions()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        # These controls hold the applied state; the preferences dialog edits a copy.
        self.language_combo = QtWidgets.QComboBox(self)
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        self.language_combo.hide()
        self.theme_combo = QtWidgets.QComboBox(self)
        for key in ("System", "Light", "Dark"):
            self.theme_combo.addItem(key, key.lower())
        self.theme_combo.hide()
        inputs = QtWidgets.QHBoxLayout()
        self.paths = []
        for i, text in enumerate(("Reference A", "Candidate B")):
            inputs.addWidget(QtWidgets.QLabel("A" if i == 0 else "B"))
            edit = QtWidgets.QLineEdit()
            edit.setClearButtonEnabled(True)
            self.bind(edit, text, "setAccessibleName")
            self.bind(edit, text, "setPlaceholderText")
            edit.setMinimumWidth(80)
            self.paths.append(edit)
            inputs.addWidget(edit, 1)
            picker = QtWidgets.QToolButton()
            self.bind(picker, "Open…")
            menu = QtWidgets.QMenu(picker)
            menu.addActions(self.open_actions[i])
            picker.setMenu(menu)
            picker.setPopupMode(QtWidgets.QToolButton.InstantPopup)
            inputs.addWidget(picker)
        self.action_button(self.swap_action, inputs)
        self.compare_button = self.action_button(self.compare_action, inputs)
        self.compare_button.setObjectName("primary")
        layout.addLayout(inputs)
        toolbar = QtWidgets.QHBoxLayout()
        for action, compact in zip(self.panel_actions, ("Files", "Parameters"), strict=True):
            self.action_button(action, toolbar, compact)
        toolbar.addStretch()
        for action, compact in zip(
            self.navigation_actions, ("Reset zoom", "Largest", "Previous", "Next"), strict=True
        ):
            self.action_button(action, toolbar, compact)
        layout.addLayout(toolbar)
        self.splitter = QtWidgets.QSplitter()
        layout.addWidget(self.splitter, 1)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 4, 0)
        self.panel_header(ll, 0)
        self.filter = self.bind(QtWidgets.QLineEdit(), "Filter results…", "setPlaceholderText")
        ll.addWidget(self.filter)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        ll.addWidget(self.table, 1)
        self.splitter.addWidget(left)
        middle = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(middle)
        ml.setContentsMargins(2, 0, 2, 0)
        ml.setSpacing(4)
        self.tabs = QtWidgets.QTabWidget()
        ml.addWidget(self.tabs, 1)
        wave_page = QtWidgets.QWidget()
        wl = QtWidgets.QVBoxLayout(wave_page)
        wl.setContentsMargins(0, 0, 0, 0)
        pg.setConfigOptions(antialias=False, useOpenGL=False)
        self.wave_plot = pg.PlotWidget()
        self.diff_plot = pg.PlotWidget()
        self.segment_plot = pg.PlotWidget()
        self.indicators = {}
        self.wave_plot.addLegend(offset=(8, 8))
        self.segment_plot.addLegend(offset=(8, 8))
        for plot, weight in ((self.wave_plot, 3), (self.diff_plot, 2), (self.segment_plot, 2)):
            plot.setMinimumHeight(120)
            name = {self.wave_plot: "wave", self.diff_plot: "diff", self.segment_plot: "segment"}[
                plot
            ]
            wl.addWidget(self.add_indicator(name, plot), weight)
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setMouseEnabled(x=True, y=True)
            plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.wave_curves = [
            self.wave_plot.plot(pen=pg.mkPen(c, width=1), name=name)
            for c, name in zip(COLORS[:2], ("A", "B"), strict=True)
        ]
        self.diff_curve = self.diff_plot.plot(pen=pg.mkPen(COLORS[2]))
        self.segment_curves = [
            self.segment_plot.plot(pen=COLORS[k], name=name)
            for k, name in enumerate(("Max", "MAE"))
        ]
        self.diff_plot.setXLink(self.wave_plot)
        self.segment_plot.setXLink(self.wave_plot)
        self.region = pg.LinearRegionItem(
            (0, 1), brush=pg.mkBrush(90, 160, 220, 25), swapMode="block"
        )
        # Let drags inside the selection reach the ViewBox; only its edges resize it.
        self.region.setMovable(False)
        for line in self.region.lines:
            line.setMovable(True)
        self.bind(
            self.wave_plot,
            "Drag to pan; drag selection edges to resize the selection.",
            "setToolTip",
        )
        self.wave_plot.addItem(self.region, ignoreBounds=True)
        self.cursors = []
        for plot in (self.wave_plot, self.diff_plot):
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#c6a0f6", style=QtCore.Qt.DashLine))
            plot.addItem(line, ignoreBounds=True)
            self.cursors.append(line)
        wave_scroll = QtWidgets.QScrollArea()
        wave_scroll.setWidgetResizable(True)
        wave_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        wave_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        wave_scroll.setWidget(wave_page)
        self.tabs.addTab(wave_scroll, "Waveform")
        self.spectrum_plot = pg.PlotWidget()
        self.spectrum_plot.addLegend()
        self.spectrum_curves = [
            self.spectrum_plot.plot(pen=color, name=name)
            for color, name in zip(COLORS, ("A", "B", "B − A"), strict=True)
        ]
        self.tabs.addTab(self.add_indicator("spectrum", self.spectrum_plot), "Spectrum")
        self.spectrogram_plot = pg.PlotWidget()
        self.image = pg.ImageItem(axisOrder="row-major")
        self.image.setLookupTable(pg.colormap.get("viridis").getLookupTable())
        self.spectrogram_plot.addItem(self.image)
        for plot in (self.spectrum_plot, self.spectrogram_plot):
            plot.setMouseEnabled(x=True, y=True)
            plot.getViewBox().setMouseMode(pg.ViewBox.PanMode)
        self.tabs.addTab(self.add_indicator("spectrogram", self.spectrogram_plot), "Spectrogram")
        self.selection_editor = SelectionEditor(self.tr, self.selection_actions)
        ml.addWidget(self.selection_editor)
        self.splitter.addWidget(middle)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(4, 0, 0, 0)
        self.panel_header(rl, 1)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
        form.setRowWrapPolicy(QtWidgets.QFormLayout.WrapLongRows)
        form.setContentsMargins(2, 2, 6, 2)
        self.strict = self.bind(QtWidgets.QCheckBox(), "Strict comparison")
        self.align = self.bind(QtWidgets.QCheckBox(), "Auto alignment")
        self.align.setChecked(True)
        self.manual = self.bind(QtWidgets.QCheckBox(), "Manual offset")
        self.mix = self.bind(QtWidgets.QCheckBox(), "Mix to mono")
        for control in (self.strict, self.align, self.manual):
            form.addRow(control)
        self.max_lag = QtWidgets.QDoubleSpinBox()
        self.max_lag.setRange(0, 300)
        self.max_lag.setValue(5)
        self.offset = QtWidgets.QSpinBox()
        self.offset.setRange(-2147483647, 2147483647)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["float64", "float32", "pcm16", "pcm24", "pcm32"])
        self.threshold = QtWidgets.QDoubleSpinBox()
        self.threshold.setDecimals(12)
        self.threshold.setRange(0, 4294967295)
        self.segment = QtWidgets.QSpinBox()
        self.segment.setRange(1, 2147483647)
        self.segment.setValue(1000)
        self.pairs = QtWidgets.QLineEdit()
        self.pairs.setPlaceholderText("1:1,2:2")
        self.channel = QtWidgets.QComboBox()
        self.channel.addItem("1")
        self.fft = QtWidgets.QComboBox()
        self.fft.addItems(["256", "512", "1024", "2048", "4096", "8192", "16384"])
        self.fft.setCurrentText("2048")
        self.hop = QtWidgets.QSpinBox()
        self.hop.setRange(1, 16384)
        self.hop.setValue(512)
        for text, control in (
            ("Max lag (s)", self.max_lag),
            ("Offset (samples)", self.offset),
            ("Numeric mode", self.mode),
            ("Threshold", self.threshold),
            ("Segment (samples)", self.segment),
            ("Channel pairs", self.pairs),
        ):
            form.addRow(self.label(text), control)
        form.addRow(self.mix)
        for text, control in (
            ("Display channel", self.channel),
            ("FFT size", self.fft),
            ("Hop size", self.hop),
        ):
            form.addRow(self.label(text), control)
        scroll.setWidget(panel)
        rl.addWidget(scroll, 2)
        rl.addWidget(self.label("Results"))
        self.metrics = QtWidgets.QPlainTextEdit()
        self.metrics.setReadOnly(True)
        rl.addWidget(self.metrics, 2)
        self.splitter.addWidget(right)
        self.panels = [left, right]
        self.splitter.setSizes([240, 884, 300])
        for i in range(3):
            self.splitter.setCollapsible(i, False)
            self.splitter.setStretchFactor(i, 1 if i == 1 else 0)
        playback = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["A", "B", "B − A"])
        playback.addWidget(self.source)
        self.original = self.bind(QtWidgets.QCheckBox(), "Original input")
        self.loop = self.bind(QtWidgets.QCheckBox(), "Loop selection")
        playback.addWidget(self.original)
        self.button("Play", self.play, playback)
        self.button("Pause / resume", self.pause_playback, playback)
        self.button("Stop", self.player.stop, playback)
        playback.addWidget(self.loop)
        volume_group = QtWidgets.QWidget()
        volume_group.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        volume_layout = QtWidgets.QHBoxLayout(volume_group)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.setSpacing(6)
        volume_layout.addWidget(self.label("Volume"))
        self.volume = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(50)
        self.volume.setMaximumWidth(100)
        volume_layout.addWidget(self.volume)
        playback.addWidget(volume_group)
        layout.addLayout(playback)
        timeline_row = QtWidgets.QHBoxLayout()
        self.seek = PlaybackTimeline()
        self.bind(
            self.seek,
            "Click to seek; drag to select; drag edges to resize. Right-click to select all.",
            "setToolTip",
        )
        self.bind(self.seek, "Playback timeline", "setAccessibleName")
        self.time_label = QtWidgets.QLabel("00:00.000 / 00:00.000")
        timeline_row.addWidget(self.seek, 1)
        timeline_row.addWidget(self.time_label)
        layout.addLayout(timeline_row)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setMaximumWidth(180)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.cancel_button = QtWidgets.QToolButton()
        self.cancel_button.setDefaultAction(self.cancel_action)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(50)
        self.redraw_timer.timeout.connect(self.redraw)
        for shortcut, callback in (("Space", self.pause_playback),):
            action = QtGui.QShortcut(QtGui.QKeySequence(shortcut), self)
            action.activated.connect(callback)

    def _restore(self, paths):
        prefs = self.preferences
        for name, indicator in self.indicators.items():
            indicator.toggle.setChecked(prefs.boolean(f"indicators/{name}"))
        for i, edit in enumerate(self.paths):
            value = (
                str(paths[i])
                if len(paths) == 2
                else (self.settings.value(f"path{i}", "") if self.restore_paths else "")
            )
            edit.setText(value if isinstance(value, str) else "")
            edit.setToolTip(edit.text())
        self.language_combo.setCurrentIndex(self.language_combo.findData(self.language))
        theme = prefs.choice("theme", ("system", "light", "dark"), "system")
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(theme)))
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
        if options.get("mode") in ("float64", "float32", "pcm16", "pcm24", "pcm32"):
            self.mode.setCurrentText(options["mode"])
        self.manual.setChecked(prefs.boolean("analysis/manual", options.get("offset") is not None))
        old_offset = prefs.valid_number(options.get("offset"), 0, -2147483647, 2147483647, True)
        self.offset.setValue(prefs.number("analysis/offset", old_offset, -2147483647, 2147483647))
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
        self.fft.setCurrentText(
            prefs.choice(
                "spectrum/fft", [self.fft.itemText(i) for i in range(self.fft.count())], "2048"
            )
        )
        self.hop.setValue(prefs.number("spectrum/hop", 512, 1, 16384))
        self.volume.setValue(prefs.number("playback/volume", 50, 0, 100))
        self.source.setCurrentIndex(prefs.number("playback/source", 0, 0, 2))
        self.loop.setChecked(prefs.boolean("playback/loop", False))
        self.original.setChecked(prefs.boolean("playback/original", False))
        self.selection_editor.set_preferences(
            prefs.choice("selection/format", tuple(FORMATS), "seconds"),
            prefs.choice("selection/mode", tuple(MODES), "start_end"),
        )
        geometry = self.settings.value("geometry")
        split = self.settings.value("splitter")
        if isinstance(geometry, QtCore.QByteArray):
            self.restoreGeometry(geometry)
        # Legacy splitter sizes need the actual shown geometry and styled size hints.
        self.legacy_split_state = split if isinstance(split, QtCore.QByteArray) else None
        for index, default in enumerate((240, 300)):
            self.panel_widths[index] = prefs.number(
                f"layout/width{index}",
                default,
                80,
                10000,
            )
            visible = prefs.boolean(f"layout/visible{index}")
            self.panel_actions[index].setChecked(visible)
            self.panels[index].setVisible(visible)
        self.apply_panel_sizes()

    def _connect(self):
        self.jobs.activityChanged.connect(self.refresh_actions)
        self.player.stateChanged.connect(self.refresh_actions)
        for edit in self.paths:
            edit.textChanged.connect(self.inputs_changed)
            edit.textChanged.connect(edit.setToolTip)
            edit.returnPressed.connect(self.start_compare)
        for control in (self.strict, self.align, self.manual, self.mix):
            control.toggled.connect(self.inputs_changed)
        for control in (self.max_lag, self.offset, self.threshold, self.segment):
            control.valueChanged.connect(self.inputs_changed)
        self.mode.currentIndexChanged.connect(self.inputs_changed)
        self.pairs.textChanged.connect(self.inputs_changed)
        self.fft.currentIndexChanged.connect(self.spectrum_options_changed)
        self.hop.valueChanged.connect(self.spectrum_options_changed)
        self.splitter.splitterMoved.connect(self.panel_sizes_changed)
        self.filter.textChanged.connect(self.filter_rows)
        self.table.itemSelectionChanged.connect(self.select_row)
        self.channel.currentIndexChanged.connect(self.channel_changed)
        self.channel.currentIndexChanged.connect(self.show_metrics)
        self.channel.currentIndexChanged.connect(self.update_spectrum)
        self.language_combo.currentIndexChanged.connect(self.language_changed)
        self.theme_combo.currentIndexChanged.connect(self.apply_theme)
        self.wave_plot.sigXRangeChanged.connect(lambda: self.redraw_timer.start())
        self.region.sigRegionChanged.connect(self.region_changed)
        self.region.sigRegionChangeFinished.connect(self.selection_committed)
        self.selection_editor.selectionEdited.connect(self.selection_committed)
        self.selection_editor.preferencesChanged.connect(self.schedule_save)
        self.selection_editor.message.connect(self.message)
        self.tabs.currentChanged.connect(self.update_spectrum)
        self.source.currentIndexChanged.connect(self.update_spectrum)
        self.volume.valueChanged.connect(lambda n: self.player.set_volume(n / 100))
        self.seek.seekRequested.connect(self.seek_play)
        self.seek.selectionPreview.connect(self.selection_preview)
        self.seek.selectionCommitted.connect(self.selection_committed)
        self.loop.toggled.connect(self.selection_committed)
        self.source.currentIndexChanged.connect(self.playback_source_changed)
        self.original.toggled.connect(self.playback_source_changed)
        for control in (self.language_combo, self.theme_combo, self.source):
            control.currentIndexChanged.connect(self.schedule_save)
        for control in (self.loop, self.original):
            control.toggled.connect(self.schedule_save)
        self.volume.valueChanged.connect(self.schedule_save)
        for indicator in self.indicators.values():
            indicator.toggle.toggled.connect(self.schedule_save)

    def apply_panel_sizes(self):
        widths = [
            width if action.isChecked() else 0
            for width, action in zip(self.panel_widths, self.panel_actions, strict=True)
        ]
        self.splitter.setSizes(
            [widths[0], max(300, self.splitter.width() - sum(widths) - 8), widths[1]]
        )

    def toggle_panel(self, index, visible):
        self.centralWidget().layout().activate()
        if not visible:
            width = self.splitter.sizes()[index * 2]
            if width:
                self.panel_widths[index] = width
        self.panels[index].setVisible(visible)
        self.centralWidget().layout().activate()
        self.apply_panel_sizes()
        self.schedule_save()

    def panel_sizes_changed(self, *_):
        for i, action in enumerate(self.panel_actions):
            if action.isChecked():
                self.panel_widths[i] = self.splitter.sizes()[i * 2]
        self.schedule_save()

    def reset_layout(self):
        self.showNormal()
        self.resize(1440, 940)
        self.panel_widths = [240, 300]
        for action, panel in zip(self.panel_actions, self.panels, strict=True):
            action.setChecked(True)
            panel.show()
        self.apply_panel_sizes()
        self.schedule_save()

    def showEvent(self, event):
        super().showEvent(event)
        self.centralWidget().layout().activate()
        if self.legacy_split_state is not None:
            if self.splitter.restoreState(self.legacy_split_state):
                for index in range(2):
                    if not self.settings.contains(f"layout/width{index}"):
                        width = self.splitter.sizes()[index * 2]
                        if width > 0:
                            self.panel_widths[index] = width
            self.legacy_split_state = None
        self.apply_panel_sizes()

    def edit_preferences(self):
        dialog = PreferencesDialog(self)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self.startup_compare = dialog.startup.isChecked()
        self.restore_paths = dialog.restore_paths.isChecked()
        self.language_combo.setCurrentIndex(
            self.language_combo.findData(dialog.language.currentData())
        )
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(dialog.theme.currentData()))
        changed = self.auto_compare_action.isChecked() != dialog.auto_compare.isChecked()
        self.auto_compare_action.setChecked(dialog.auto_compare.isChecked())
        if changed:
            self.auto_compare_changed()
        self.schedule_save()

    def auto_compare_changed(self, *_):
        if self.auto_compare_action.isChecked():
            self.schedule_compare()
        else:
            self.compare_timer.stop()
        self.refresh_actions()
        self.schedule_save()

    def schedule_save(self, *_):
        if not self.restoring and not self.closing:
            self.save_timer.start()

    def save_preferences(self):
        values = {
            "language": self.language,
            "theme": self.theme_combo.currentData(),
            "geometry": self.saveGeometry(),
            "splitter": self.splitter.saveState(),
            "automation/enabled": self.auto_compare_action.isChecked(),
            "automation/startup": self.startup_compare,
            "restore_paths": self.restore_paths,
            "spectrum/fft": self.fft.currentText(),
            "spectrum/hop": self.hop.value(),
            "playback/volume": self.volume.value(),
            "playback/source": self.source.currentIndex(),
            "playback/loop": self.loop.isChecked(),
            "playback/original": self.original.isChecked(),
            "selection/format": self.selection_editor.value_format,
            "selection/mode": self.selection_editor.mode,
        }
        for i, action in enumerate(self.panel_actions):
            values[f"layout/visible{i}"] = action.isChecked()
            width = self.splitter.sizes()[i * 2] if action.isChecked() else self.panel_widths[i]
            values[f"layout/width{i}"] = width or self.panel_widths[i]
        for name, indicator in self.indicators.items():
            values[f"indicators/{name}"] = indicator.toggle.isChecked()
        for i, path in enumerate(self.paths):
            values[f"path{i}"] = path.text() if self.restore_paths else ""
        try:
            values["options"] = json.dumps(self.options().report())
            values["analysis/manual"] = self.manual.isChecked()
            values["analysis/offset"] = self.offset.value()
        except ValueError:
            pass
        for key, value in values.items():
            self.settings.setValue(key, value)
        self.settings.sync()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "save_timer"):
            self.schedule_save()

    def moveEvent(self, event):
        super().moveEvent(event)
        if hasattr(self, "save_timer"):
            self.schedule_save()

    def inputs_changed(self, *_):
        if self.restoring or self.closing:
            return
        self.invalidate()
        self.schedule_save()
        if self.auto_compare_action.isChecked():
            self.schedule_compare()

    def comparison_inputs(self):
        a, b = [edit.text().strip() for edit in self.paths]
        if not a or not b:
            raise ValueError("Choose two files or two folders")
        if not (
            (Path(a).is_file() and Path(b).is_file()) or (Path(a).is_dir() and Path(b).is_dir())
        ):
            raise ValueError("Choose two files or two folders")
        return a, b, self.options()

    def schedule_compare(self):
        self.compare_timer.stop()
        if self.closing:
            return
        try:
            self.comparison_inputs()
        except (ValueError, OSError) as error:
            self.message(str(error))
        else:
            self.message("Waiting for changes…")
            self.compare_timer.start()
        self.refresh_actions()

    def spectrum_options_changed(self, *_):
        self.spectrum_timer.stop()
        self.jobs.cancel_kind("spectrum")
        self.reset_indicators(("spectrum", "spectrogram"))
        self.spectral_data = None
        self.spectral_ranges = None
        for curve in self.spectrum_curves:
            curve.setData([], [])
        self.image.clear()
        if self.result and self.tabs.currentIndex() != 0 and not self.closing:
            self.spectrum_timer.start()
        self.schedule_save()

    def refresh_actions(self, *_):
        for action in self.selection_actions + self.navigation_actions:
            action.setEnabled(self.result is not None)
        for action in self.export_actions:
            action.setEnabled(bool(self.rows) or self.result is not None)
        self.cancel_action.setEnabled(
            bool(self.jobs.latest)
            or self.compare_timer.isActive()
            or self.spectrum_timer.isActive()
            or self.player.state != "stopped"
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

    def invalidate(self, *_):
        self.compare_timer.stop()
        self.spectrum_timer.stop()
        self.redraw_timer.stop()
        self.reset_indicators()
        self.spectral_data = None
        self.jobs.cancel_all()
        self.player.stop()
        self.result = None
        self.reset_playback()
        self.spectral_ranges = None
        self.rows = []
        self.table.setRowCount(0)
        self.detail_cache.clear()
        self.view_cache.clear()
        for curve in [
            *self.wave_curves,
            self.diff_curve,
            *self.segment_curves,
            *self.spectrum_curves,
        ]:
            curve.setData([], [])
        self.image.clear()
        self.metrics.clear()
        self.refresh_actions()
        self.message("Settings changed; compare again.")

    def browse(self, side, folder):
        old = self.paths[side].text()
        initial = old if Path(old).is_dir() else str(Path(old).parent)
        if folder:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Folder…"), initial)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, self.tr("File…"), initial, "WAV (*.wav *.WAV *.rf64 *.w64)"
            )
        if path:
            self.paths[side].setText(path)

    def swap(self):
        a, b = (edit.text() for edit in self.paths)
        self.set_paths((b, a))

    def set_paths(self, paths):
        for edit, path in zip(self.paths, paths, strict=False):
            with QtCore.QSignalBlocker(edit):
                edit.setText(str(path))
            edit.setToolTip(edit.text())
        self.inputs_changed()

    def start_compare(self):
        self.compare_timer.stop()
        if self.closing:
            return
        try:
            a, b, options = self.comparison_inputs()
            self.invalidate()
            self.batch_cancelled = False
            self.message("Loading…")
            if Path(a).is_file() and Path(b).is_file():
                self.jobs.submit(
                    "detail", lambda cancel, progress: compare(a, b, options, cancel, progress)
                )
            else:
                self.jobs.submit(
                    "batch",
                    lambda cancel, progress: run_batch(
                        discover(a, b, cancel), options, cancel, progress
                    ),
                )
        except Exception as error:
            self.message(str(error))
        self.refresh_actions()

    def cancel_tasks(self):
        self.compare_timer.stop()
        self.spectrum_timer.stop()
        self.redraw_timer.stop()
        self.batch_cancelled = True
        # Keep the batch's final partial report, but prevent queued view/detail work.
        for job in self.jobs.jobs.values():
            job.cancel.cancel()
        for kind in list(self.jobs.latest):
            if kind != "batch":
                self.jobs.cancel_kind(kind)
        self.player.stop()
        self.refresh_actions()
        self.message("Cancelled")

    def progress_changed(self, value, text):
        self.progress_bar.setValue(round(value * 1000))
        self.message(text)

    def job_done(self, kind, outcome):
        success, value = outcome
        if not success:
            self.message(value)
            return
        if kind == "batch":
            self.rows = value
            self.populate_rows()
            if not self.batch_cancelled:
                candidates = [
                    i
                    for i, row in enumerate(self.rows)
                    if row["status"] not in ("missing", "collision", "error", "cancelled")
                ]
                chosen = next(
                    (i for i in candidates if self.rows[i]["name"] == self.preferred_row),
                    next(iter(candidates), None),
                )
                if chosen is not None:
                    for i in range(self.table.rowCount()):
                        if self.table.item(i, 0).data(QtCore.Qt.UserRole) == chosen:
                            self.table.selectRow(i)
                            break
            else:
                self.message("Cancelled")
        elif kind == "detail":
            self.set_result(value)
            if not self.rows:
                self.rows = [
                    {
                        "name": Path(value.a_path).name,
                        "a_path": value.report["a"]["path"],
                        "b_path": value.report["b"]["path"],
                        **value.report,
                    }
                ]
                self.populate_rows()
        elif kind == "view":
            key, data = value
            self.view_cache.put(key, data)
            self.draw_data(data)
        elif kind == "spectrum":
            self.draw_spectrum(value)
        elif kind == "segments":
            x, maxes, means = value
            self.segment_curves[0].setData(x, maxes)
            self.segment_curves[1].setData(x, means)
        elif kind.startswith("indicator:"):
            revision, x, fields = value
            self.indicators[kind.split(":", 1)[1]].finish(revision, x, fields)
        elif kind == "navigate":
            if value is not None:
                self.zoom_to(value)

    def populate_rows(self):
        self.table.blockSignals(True)
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(self.rows))
        for i, row in enumerate(self.rows):
            maximum = max((m["max_abs"] for m in row.get("metrics", [])), default=None)
            for j, value in enumerate(
                (
                    row["name"],
                    self.tr(row["status"]),
                    f"{maximum:.6g}" if maximum is not None else "—",
                )
            ):
                item = QtWidgets.QTableWidgetItem(value)
                item.setData(QtCore.Qt.UserRole, i)
                item.setToolTip(row.get("error", row["name"]))
                self.table.setItem(i, j, item)
        self.table.setSortingEnabled(True)
        self.table.blockSignals(False)
        self.filter_rows()
        self.refresh_actions()
        self.message(f"{self.tr('Complete')} · {len(self.rows)}")

    def filter_rows(self):
        query = self.filter.text().casefold()
        for row in range(self.table.rowCount()):
            text = " ".join(self.table.item(row, col).text() for col in range(3))
            self.table.setRowHidden(row, query not in text.casefold())

    def select_row(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        row = self.rows[selected[0].data(QtCore.Qt.UserRole)]
        self.preferred_row = row["name"]
        self.spectrum_timer.stop()
        self.jobs.cancel_all()
        self.player.stop()
        self.result = None
        self.reset_playback()
        self.refresh_actions()
        self.view_cache.clear()
        for curve in [
            *self.wave_curves,
            self.diff_curve,
            *self.segment_curves,
            *self.spectrum_curves,
        ]:
            curve.setData([], [])
        self.image.clear()
        self.reset_indicators()
        self.spectral_data = None
        if row["status"] in ("missing", "collision"):
            self.metrics.setPlainText(json.dumps(row, ensure_ascii=False, indent=2))
            return
        try:
            options = self.options()
            a, b = str(Path(row["a_path"]).resolve()), str(Path(row["b_path"]).resolve())
            key = (
                a,
                b,
                Path(a).stat().st_mtime_ns,
                Path(b).stat().st_mtime_ns,
                Path(a).stat().st_size,
                Path(b).stat().st_size,
                options,
            )
            cached = self.detail_cache.get(key)
            self.jobs.cancel_all()
            self.player.stop()
            self.result = None
            self.reset_indicators()
            self.spectral_data = None
            if cached:
                self.set_result(cached)
            else:
                self.jobs.submit(
                    "detail", lambda cancel, progress: compare(a, b, options, cancel, progress)
                )
        except Exception as error:
            self.message(str(error))

    def set_result(self, result):
        self.player.stop()
        self.jobs.cancel_all()
        self.reset_indicators()
        self.spectral_data = None
        self.result = result
        self.refresh_actions()
        self.spectral_ranges = None
        a, b = result.report["a"], result.report["b"]
        key = (
            a["path"],
            b["path"],
            a["mtime_ns"],
            b["mtime_ns"],
            a["size"],
            b["size"],
            result.options,
        )
        self.detail_cache[key] = result
        self.detail_cache.move_to_end(key)
        while len(self.detail_cache) > 2:
            self.detail_cache.popitem(last=False)
        self.view_cache.clear()
        self.channel.blockSignals(True)
        self.channel.clear()
        self.channel.addItems(
            [f"{m['a_channel']} → {m['b_channel']}" for m in result.report["metrics"]]
        )
        self.channel.blockSignals(False)
        duration = result.frames / result.rate
        self.seek.set_duration(duration, result.rate)
        self.selection_editor.set_context(result.rate, result.frames)
        self.region.setBounds((0, result.frames / result.rate))
        self.region.setRegion((0, min(10.0, result.frames / result.rate)))
        self.region_changed()
        self.play_position(0.0)
        self.update_seek_bounds()
        self.show_metrics()
        self.reset_zoom()
        self.update_spectrum()
        self.message("Complete")

    def channel_changed(self):
        self.reset_indicators()
        self.spectral_data = None
        for kind in ("view", "spectrum", "segments", "navigate"):
            self.jobs.cancel_kind(kind)
        self.player.stop()
        self.reset_zoom()

    def show_metrics(self):
        if not self.result:
            return
        r = self.result.report
        ch = max(0, self.channel.currentIndex())
        m = r["metrics"][ch]
        text = (
            f"{self.tr(r['status'])}\n"
            f"{r['analysis_samplerate']:,} Hz · {r['frames_compared']:,} {self.tr('samples')}\n"
            f"{self.tr('Domain')}: {self.tr(r['comparison_domain'])}\n"
            f"{self.tr('Resampled B')}: {self.tr(str(r['resampled_b']))}\n"
            f"{self.tr('Alignment')}: {self.tr(r['alignment']['status'])}\n"
            f"{self.tr('Lag')}: {r['alignment']['lag_samples']} {self.tr('samples')}\n\n"
            + "\n".join(
                f"{self.tr(key)}: {value:.9g}"
                if isinstance(value, float)
                else f"{self.tr(key)}: {self.tr(str(value))}"
                for key, value in m.items()
            )
            + f"\n\n{self.tr('Excluded')}:\n"
            + "\n".join(f"{self.tr(k)}: {self.tr(str(v))}" for k, v in r["excluded"].items())
        )
        self.metrics.setPlainText(text)

    def reset_zoom(self):
        self.reset_indicators()
        if not self.result:
            return
        duration = self.result.frames / self.result.rate
        ranges = amplitude_ranges(self.result, max(0, self.channel.currentIndex()))
        for plot, (low, high) in zip(
            (self.wave_plot, self.diff_plot, self.segment_plot), ranges, strict=True
        ):
            if low == high:
                low, high = -1.0, 1.0
            span = high - low
            plot.setLimits(
                xMin=0,
                xMax=duration,
                minXRange=1 / self.result.rate,
                yMin=low - span * 0.25,
                yMax=high + span * 0.25,
            )
            # Fixed global bounds avoid viewport refreshes undoing a manual pan.
            plot.setYRange(low - span * 0.05, high + span * 0.05, padding=0)
        self.wave_plot.setXRange(0, duration, padding=0)
        self.reset_spectral_zoom()
        for line in self.cursors:
            line.setPos(0)
        self.redraw_timer.start()

    def redraw(self):
        if not self.result:
            return
        result = self.result
        lo, hi = self.wave_plot.viewRange()[0]
        channel = max(0, self.channel.currentIndex())
        pixels = max(100, self.wave_plot.width())
        key = (id(result), round(lo * result.rate), round(hi * result.rate), channel, pixels)
        cached = self.view_cache.get(key)
        if cached is not None:
            self.jobs.cancel_kind("view")
            self.draw_data(cached)
        else:
            self.jobs.submit(
                "view", lambda cancel, progress: (key, waveform(result, lo, hi, channel, pixels))
            )

        def segment_view(cancel, progress):
            data = result.segment_data()
            start = max(0, int(lo * result.rate / result.options.segment_size))
            stop = min(len(data), int(hi * result.rate / result.options.segment_size) + 1)
            if stop <= start:
                return np.array([]), np.array([]), np.array([])
            stride = max(1, int(np.ceil((stop - start) / pixels)))
            # Aggregate without dropping high-error segments.
            x, maxima, means = [], [], []
            for i in range(start, stop, stride):
                cancel.check()
                group = data[i : min(i + stride, stop), channel]
                x.append(i * result.options.segment_size / result.rate)
                maxima.append(group[:, 0].max())
                means.append(group[:, 1].sum() / max(1, group[:, 3].sum()))
            return np.array(x), np.array(maxima), np.array(means)

        self.jobs.submit("segments", segment_view)

    def draw_data(self, data):
        if not self.result:
            return
        times, a, b, difference = data
        self.wave_curves[0].setData(times, a)
        self.wave_curves[1].setData(times, b)
        self.diff_curve.setData(times, difference)

    def region_changed(self):
        if not self.result:
            return
        self.selection_preview(*self.region.getRegion())

    def selection_preview(self, lo, hi):
        if not self.result:
            return
        rate = self.result.rate
        self.selection_editor.set_selection(round(lo * rate), round(hi * rate))
        self.sync_selection()

    def sync_selection(self):
        lo, hi = self.selection_editor.seconds
        self.seek.set_selection(lo, hi)
        with QtCore.QSignalBlocker(self.region):
            self.region.setRegion((lo, hi))
        self.update_seek_bounds()

    def selection_committed(self, *_):
        if not self.result:
            return
        self.sync_selection()
        self.update_seek_bounds()
        if self.loop.isChecked():
            lo, hi = self.seek.seek_bounds
            position = self.playback_position
            if not lo <= position < hi:
                position = lo
            self.seek_play(position)
        elif self.player.state != "stopped" and self.player.loop:
            self.seek_play(self.playback_position)
        else:
            self.play_position(self.playback_position)

    def analyze_region(self, selected):
        if not self.result:
            return
        result = self.result
        # Convert display-relative selection to the full aligned timeline.
        origin = round(result.options.region[0] * result.rate) if result.options.region else 0
        start, end = self.selection_editor.bounds
        region = (
            ((origin + start) / result.rate, (origin + end) / result.rate) if selected else None
        )
        options = replace(result.options, region=region)
        a, b = result.report["a"]["path"], result.report["b"]["path"]
        self.jobs.cancel_all()
        self.player.stop()
        self.jobs.submit(
            "detail", lambda cancel, progress: compare(a, b, options, cancel, progress)
        )

    def update_spectrum(self, *_):
        self.spectrum_timer.stop()
        self.reset_indicators(("spectrum", "spectrogram"))
        self.spectral_data = None
        self.jobs.cancel_kind("spectrum")
        if not self.result or self.tabs.currentIndex() == 0:
            return
        result = self.result
        begin, end = self.selection_editor.seconds
        ch = max(0, self.channel.currentIndex())
        fft, hop = int(self.fft.currentText()), self.hop.value()
        bounds = (
            max(0, round(begin * result.rate)) / result.rate,
            min(result.frames, round(end * result.rate)) / result.rate,
        )
        self.message("Loading…")
        self.jobs.submit(
            "spectrum",
            lambda cancel, progress: (bounds, spectra(result, begin, end, ch, fft, hop, cancel)),
        )

    def draw_spectrum(self, value):
        (begin, end), (frequencies, psd, times, images) = value
        self.reset_indicators(("spectrum", "spectrogram"))
        self.spectral_data = (frequencies, psd, times, images)
        for curve, data in zip(self.spectrum_curves, psd, strict=True):
            curve.setData(frequencies, data)
        nyquist = self.result.rate / 2
        low, high = float(psd.min()), float(psd.max())
        margin = max(6.0, (high - low) * 0.1)
        self.spectrum_plot.setLimits(
            xMin=0, xMax=nyquist, yMin=low - 2 * margin, yMax=high + 2 * margin
        )
        self.spectrogram_plot.setLimits(
            xMin=begin, xMax=end, minXRange=1 / self.result.rate, yMin=0, yMax=nyquist
        )
        self.spectral_ranges = (
            ((0, nyquist), (low - margin, high + margin)),
            ((begin, end), (0, nyquist)),
        )
        source = self.source.currentIndex()
        self.image.setImage(images[source], levels=(-120, 0), autoLevels=False)
        # A single STFT column covers its selection, not a window starting at its center.
        left, right = (float(times[0]), float(times[-1])) if len(times) > 1 else (begin, end)
        self.image.setRect(QtCore.QRectF(left, 0, right - left, nyquist))
        self.reset_spectral_zoom()
        self.message("Complete")

    def reset_spectral_zoom(self):
        if self.spectral_ranges is None:
            return
        for plot, (x_range, y_range) in zip(
            (self.spectrum_plot, self.spectrogram_plot), self.spectral_ranges, strict=True
        ):
            plot.setRange(xRange=x_range, yRange=y_range, padding=0)

    def add_indicator(self, name, plot):
        indicator = PlotIndicator(plot, self.tr, crosshair=name == "spectrogram")
        self.indicators[name] = indicator
        indicator.requested.connect(lambda revision, point: self.probe(name, revision, point))
        indicator.invalidated.connect(lambda: self.jobs.cancel_kind(f"indicator:{name}"))
        return indicator.panel

    def reset_indicators(self, names=None):
        for name in self.indicators if names is None else names:
            self.indicators[name].reset()

    def probe(self, name, revision, point):
        if not self.result:
            return
        indicator = self.indicators[name]
        result, channel = self.result, max(0, self.channel.currentIndex())
        fields = [("Display channel", self.channel.currentText())]
        if name in ("wave", "diff", "segment"):
            if not 0 <= point.x() < result.frames / result.rate:
                return
            sample = int(point.x() * result.rate)
            indicator.accept(revision)

            def read(cancel, progress):
                cancel.check()
                if name == "segment":
                    width = result.options.segment_size
                    index = sample // width
                    maximum, total, _, count = result.segment_data()[index, channel]
                    start, stop = index * width, min(result.frames, (index + 1) * width)
                    details = [
                        ("Segment index (0-based)", str(index)),
                        ("Time range", f"[{start / result.rate:.6f}, {stop / result.rate:.6f}) s"),
                        ("Sample count", str(int(count))),
                        ("Max absolute difference", f"{maximum:.9g}"),
                        ("MAE", f"{total / max(1, count):.9g}"),
                    ]
                else:
                    a, b, difference = (float(v[0, channel]) for v in result.samples(sample, 1))
                    details = [
                        ("Time", f"{sample / result.rate:.6f} s"),
                        ("Sample index (0-based)", str(sample)),
                    ]
                    if name == "wave":
                        details.extend((("A", f"{a:.9g}"), ("B", f"{b:.9g}")))
                    details.append(("B − A", f"{difference:.9g}"))
                    if name == "diff":
                        details.append(("Absolute difference", f"{abs(difference):.9g}"))
                cancel.check()
                return revision, sample / result.rate, fields + details

            self.jobs.submit(f"indicator:{name}", read)
            return
        if self.spectral_data is None:
            return
        frequencies, psd, times, images = self.spectral_data
        if name == "spectrum":
            # Scene/view transforms can round an exact endpoint a few ULPs outward.
            tolerance = np.finfo(float).eps * max(1.0, abs(frequencies[-1])) * 8
            if not frequencies[0] - tolerance <= point.x() <= frequencies[-1] + tolerance:
                return
            index = int(np.abs(frequencies - point.x()).argmin())
            fields.append(("Frequency", f"{frequencies[index]:.9g} Hz"))
            fields.extend(
                (label, f"{psd[k, index]:.9g} dB/Hz") for k, label in enumerate(("A", "B", "B − A"))
            )
            indicator.accept(revision)
            indicator.finish(revision, float(frequencies[index]), fields)
        else:
            pixel = self.image.mapFromParent(point)
            rows, columns = images.shape[1:]
            if not (0 <= pixel.x() <= columns and 0 <= pixel.y() <= rows):
                return
            column, row = min(int(pixel.x()), columns - 1), min(int(pixel.y()), rows - 1)
            source = self.source.currentIndex()
            fields.extend(
                (
                    ("Source", self.source.currentText()),
                    ("Frame time", f"{times[column]:.6f} s"),
                    ("Frequency", f"{frequencies[row]:.9g} Hz"),
                    ("Amplitude", f"{images[source, row, column]:.9g} dB"),
                )
            )
            # Locate the displayed cell; its analysis coordinates are in the readout.
            center = self.image.mapToParent(QtCore.QPointF(column + 0.5, row + 0.5))
            indicator.accept(revision)
            indicator.finish(revision, center.x(), fields, center.y())

    def largest(self):
        if self.result:
            self.zoom_to(
                self.result.report["metrics"][max(0, self.channel.currentIndex())]["max_at_sample"]
            )

    def zoom_to(self, sample):
        if self.result:
            center = sample / self.result.rate
            self.wave_plot.setXRange(
                max(0, center - 0.1),
                min(self.result.frames / self.result.rate, center + 0.1),
                padding=0.1,
            )
            for line in self.cursors:
                line.setPos(center)

    def next_difference(self, direction):
        if not self.result:
            return
        result, channel = self.result, max(0, self.channel.currentIndex())
        current = round(self.cursors[0].value() * result.rate)

        def find(cancel, progress):
            data = result.segment_data()
            seg = current // result.options.segment_size
            iterator = range(seg, len(data)) if direction > 0 else range(seg, -1, -1)
            for index in iterator:
                cancel.check()
                if data[index, channel, 0] > result.options.threshold:
                    start = index * result.options.segment_size
                    end = min(result.frames, start + result.options.segment_size)
                    if direction > 0:
                        start = max(start, current + 1)
                        chunks = range(start, end, 65536)
                    else:
                        end = min(end, current)
                        chunks = reversed(range(start, end, 65536))
                    for chunk in chunks:
                        cancel.check()
                        d = result.samples(chunk, min(65536, end - chunk))[2][:, channel]
                        hits = np.flatnonzero(np.abs(d) > result.options.threshold)
                        if len(hits):
                            return chunk + int(hits[0 if direction > 0 else -1])
            return None

        self.jobs.submit("navigate", find)

    def reset_playback(self):
        self.selection_editor.set_context(1, 0)
        self.seek.set_duration(0, 1)
        self.playback_position = 0.0
        self.time_label.setText("00:00.000 / 00:00.000")
        for line in self.cursors:
            line.setPos(0)

    def playback_bounds(self):
        if not self.result:
            return 0.0, 0.0
        duration = self.result.frames / self.result.rate
        if self.original.isChecked() and self.source.currentIndex() in (0, 1):
            info = self.result.report["a" if self.source.currentIndex() == 0 else "b"]
            duration = min(duration, info["frames"] / info["samplerate"])
        lo, hi = self.selection_editor.seconds if self.loop.isChecked() else (0.0, duration)
        return lo, min(hi, duration)

    def update_seek_bounds(self):
        lo, hi = self.playback_bounds()
        self.seek.seek_bounds = (lo, max(lo, hi))

    def playback_source_changed(self, *_):
        self.player.stop()
        self.update_seek_bounds()
        if self.result:
            self.seek_play(self.playback_position)

    def start_playback(self, position, paused=False):
        lo, hi = self.playback_bounds()
        rate = self.result.rate
        if self.original.isChecked() and self.source.currentIndex() in (0, 1):
            rate = self.result.report["a" if self.source.currentIndex() == 0 else "b"]["samplerate"]
        if round(hi * rate) <= round(lo * rate):
            self.player.stop()
            self.message("No playable audio in selection")
            return
        # A loop's end is exclusive; seeking there starts the next iteration.
        at_end = round(position * rate) >= round(hi * rate)
        if self.loop.isChecked() and at_end:
            position = lo
        elif at_end:
            self.player.park(hi, paused=paused)
            return
        self.player.play(
            self.result,
            self.source.currentIndex(),
            max(0, self.channel.currentIndex()),
            max(lo, position),
            hi,
            self.loop.isChecked(),
            self.original.isChecked(),
            loop_begin=lo,
            paused=paused,
        )

    def play(self, *_):
        if not self.result:
            return
        lo, hi = self.playback_bounds()
        position = self.playback_position
        if not lo <= position < hi:
            position = lo
        self.start_playback(position)

    def pause_playback(self):
        if self.player.state == "paused" and self.player.sink is None:
            self.play()
        else:
            self.player.pause()

    def play_position(self, seconds):
        if self.result:
            self.playback_position = max(0.0, min(seconds, self.seek.duration))
            if self.seek.interacting:
                return
            self.seek.set_position(self.playback_position)

            def format_time(value):
                millis = round(value * 1000)
                minutes, millis = divmod(millis, 60000)
                return f"{minutes:02d}:{millis // 1000:02d}.{millis % 1000:03d}"

            self.time_label.setText(
                f"{format_time(self.playback_position)} / {format_time(self.seek.duration)}"
            )
            for line in self.cursors:
                line.setPos(self.playback_position)

    def seek_play(self, position):
        if not self.result:
            return
        lo, hi = self.playback_bounds()
        position = max(lo, min(position, hi))
        state = self.player.state
        self.play_position(position)
        if state != "stopped":
            self.start_playback(position, paused=state == "paused")

    def export(self, kind):
        rows = self.rows
        if self.result and len(rows) <= 1:
            rows = [{"name": Path(self.result.a_path).name, **self.result.report}]
        if not rows:
            self.message("No results to export")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Export JSON…" if kind == "json" else "Export CSV…"),
            f"comparison.{kind}",
            f"{kind.upper()} (*.{kind})",
        )
        if path:
            try:
                (export_json if kind == "json" else export_csv)(path, rows)
                self.message("Export complete")
            except Exception as error:
                self.message(str(error))

    def clear_cache(self):
        self.invalidate()
        self.message("Clear cache")

    def message(self, text):
        self.statusBar().showMessage(self.tr(text))

    def on_idle(self):
        if self.closing:
            self.close()

    def language_changed(self):
        self.language = self.language_combo.currentData()
        self.retranslate()

    def retranslate(self):
        self.selection_editor.retranslate()
        self.seek.select_all_text = self.tr("Select all")
        for indicator in self.indicators.values():
            indicator.retranslate()
        for widget, text, method in self.bindings:
            getattr(widget, method)(self.tr(text))
        for i, text in enumerate(("Waveform", "Spectrum", "Spectrogram")):
            self.tabs.setTabText(i, self.tr(text))
        for i, text in enumerate(("System", "Light", "Dark")):
            self.theme_combo.setItemText(i, self.tr(text))
        self.table.setHorizontalHeaderLabels(
            [self.tr(t) for t in ("Name", "Status", "Max difference")]
        )
        for plot, title in (
            (self.wave_plot, "A / B waveforms"),
            (self.diff_plot, "B − A difference"),
            (self.segment_plot, "Segment differences"),
        ):
            plot.setTitle(self.tr(title))
            plot.setLabel("bottom", self.tr("Time (s)"))
            plot.setLabel("left", self.tr("Amplitude"))
        self.spectrum_plot.setLabel("bottom", self.tr("Frequency (Hz)"))
        self.spectrum_plot.setLabel("left", self.tr("PSD (dB/Hz)"))
        self.spectrogram_plot.setLabel("bottom", self.tr("Time (s)"))
        self.spectrogram_plot.setLabel("left", self.tr("Frequency (Hz)"))
        if self.rows:
            self.populate_rows()
        self.show_metrics()

    def apply_theme(self):
        choice = self.theme_combo.currentData()
        dark = choice == "dark" or (
            choice == "system"
            and QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        )
        bg, panel, fg, border = (
            ("#151b27", "#202938", "#e2e9f3", "#344156")
            if dark
            else ("#f4f6fa", "#ffffff", "#1e293b", "#d1dbe8")
        )
        for indicator in self.indicators.values():
            indicator.apply_theme(dark)
        self.seek.apply_theme(dark)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {bg}; color: {fg}; font-size: 12px; }}
            QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTableWidget {{
                background: {panel}; border: 1px solid {border}; border-radius: 4px; padding: 5px; }}
            QPushButton, QToolButton {{ background: {panel}; border: 1px solid {border}; border-radius: 4px; padding: 4px 7px; }}
            QPushButton:hover, QToolButton:hover {{ border-color: #4298d8; }}
            QToolButton:checked {{ background: {border}; }}
            QToolButton#primary {{ background: #277ab8; color: white; font-weight: 600; }}
            QPushButton:disabled, QToolButton:disabled, QMenu::item:disabled {{ color: #7d8795; }}
            QMenu {{ background: {panel}; border: 1px solid {border}; }}
            QMenu::item {{ padding: 5px 22px; }}
            QMenu::item:selected, QMenuBar::item:selected {{ background: {border}; }}
            QHeaderView::section {{ background: {panel}; padding: 5px; border: none; }}
            QTabBar::tab {{ padding: 5px 12px; }}
            QTabBar::tab:selected {{ border-bottom: 2px solid #4298d8; }}
            QSplitter::handle {{ background: {border}; }}
        """)
        for plot in (
            self.wave_plot,
            self.diff_plot,
            self.segment_plot,
            self.spectrum_plot,
            self.spectrogram_plot,
        ):
            plot.setBackground(panel)
            for name in ("bottom", "left"):
                plot.getAxis(name).setPen(fg)
                plot.getAxis(name).setTextPen(fg)

    def about(self):
        QtWidgets.QMessageBox.about(
            self,
            self.tr("About"),
            f"WAV Compare {get_version()}\nMIT · sunging\nhttps://github.com/sunging/wav_compare",
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and all(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.set_paths(paths[:2])
        event.acceptProposedAction()

    def closeEvent(self, event):
        self.closing = True
        self.jobs.accepting = False
        self.compare_timer.stop()
        self.spectrum_timer.stop()
        self.redraw_timer.stop()
        self.save_timer.stop()
        self.reset_indicators()
        self.player.stop()
        if self.jobs.jobs or self.player.has_workers():
            self.closing = True
            self.jobs.cancel_all()
            if self.player.has_workers():
                QtCore.QTimer.singleShot(50, self.close)
            self.message("Closing after background tasks stop…")
            event.ignore()
            return
        self.save_preferences()
        self.result = None
        self.detail_cache.clear()
        self.view_cache.clear()
        event.accept()


def launch(paths=()):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("WAV Compare")
    app.setStyle("Fusion")
    window = Window(paths)
    window.show()
    return app.exec()
