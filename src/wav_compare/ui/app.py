from __future__ import annotations

import sys
from collections import OrderedDict
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from .. import get_version
from ..batch import discover, run_batch
from ..engine import compare
from ..reports import export_csv, export_json
from ..views import ByteCache, amplitude_ranges, difference_overview, spectra, waveform
from .i18n import translate
from .indicator import PlotIndicator
from .jobs import Jobs
from .parameters import ParametersPanel
from .player import Player
from .preferences import Preferences, PreferencesDialog
from .results import ANALYZABLE_EXCLUDED, MetricsView, ResultsTable
from .selection import FORMATS, MODES, SelectionEditor
from .timeline import PlaybackTimeline

COLORS = ("#52b9f5", "#ffb454", "#74dbb0")
NAVIGATION_COLOR = "#ff6b6b"
PARAMETER_CONTROLS = (
    *("strict", "align", "manual", "mix", "max_lag", "offset", "mode"),
    *("threshold", "segment", "pairs", "channel", "fft", "hop"),
)
THEMES = {
    # background, panel, foreground, border, muted
    False: ("#f4f6fa", "#ffffff", "#1e293b", "#d1dbe8", "#64748b"),
    True: ("#151b27", "#202938", "#e2e9f3", "#344156", "#94a3b8"),
}


def format_clock(seconds):
    millis = round(seconds * 1000)
    minutes, millis = divmod(millis, 60000)
    hours, minutes = divmod(minutes, 60)
    text = f"{minutes:02d}:{millis // 1000:02d}.{millis % 1000:03d}"
    return f"{hours}:{text}" if hours else text


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
        self.foreground = THEMES[False][2]
        self.result = None
        self.spectral_ranges = None
        self.spectral_data = None
        # Identifies the latest requested spectrum so tab/source switches do not recompute it.
        self.spectral_key = None
        self.navigation_sample = None
        self.detail_cache = OrderedDict()
        self.view_cache = ByteCache()
        self.closing = False
        self.restoring = True
        self.preferred_row = None
        self.batch_cancelled = False
        self.batch_mode = False
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

    @property
    def rows(self):
        return self.results.rows

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
            # Dropping onto one field replaces only that side.
            edit.installEventFilter(self)
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
        self.results = ResultsTable(self.tr)
        self.filter, self.table = self.results.filter, self.results.table
        ll.addWidget(self.results, 1)
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
        self.navigation_lines = []
        for plot in (self.wave_plot, self.diff_plot):
            line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#c6a0f6", style=QtCore.Qt.DashLine))
            plot.addItem(line, ignoreBounds=True)
            self.cursors.append(line)
            # Difference navigation has its own marker; playback never moves it.
            marker = pg.InfiniteLine(angle=90, pen=pg.mkPen(NAVIGATION_COLOR, width=1.5))
            marker.setAcceptedMouseButtons(QtCore.Qt.NoButton)
            marker.hide()
            plot.addItem(marker, ignoreBounds=True)
            self.navigation_lines.append(marker)
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
        self.parameters = ParametersPanel()
        # Keep the controls addressable on the window for automation and tests.
        for name in PARAMETER_CONTROLS:
            setattr(self, name, getattr(self.parameters, name))
        scroll.setWidget(self.parameters)
        # Parameters and results share the side panel; the boundary is user adjustable.
        self.side_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.side_splitter.setChildrenCollapsible(False)
        self.side_splitter.addWidget(scroll)
        report = QtWidgets.QWidget()
        report_layout = QtWidgets.QVBoxLayout(report)
        report_layout.setContentsMargins(0, 4, 0, 0)
        report_layout.addWidget(self.label("Results"))
        self.metrics = MetricsView(self.tr)
        report_layout.addWidget(self.metrics, 1)
        self.side_splitter.addWidget(report)
        self.side_splitter.setSizes([520, 360])
        rl.addWidget(self.side_splitter, 1)
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
        self.play_button = QtWidgets.QPushButton()
        self.play_button.setMinimumWidth(72)
        self.play_button.clicked.connect(self.toggle_playback)
        playback.addWidget(self.play_button)
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
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.cancel_button = QtWidgets.QToolButton()
        self.cancel_button.setDefaultAction(self.cancel_action)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(50)
        self.redraw_timer.timeout.connect(self.redraw)
        QtGui.QShortcut(QtGui.QKeySequence("Space"), self).activated.connect(self.toggle_playback)

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
        self.parameters.restore(prefs)
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
        self.parameters.optionsChanged.connect(self.inputs_changed)
        self.parameters.spectrumOptionsChanged.connect(self.spectrum_options_changed)
        self.splitter.splitterMoved.connect(self.panel_sizes_changed)
        self.results.rowSelected.connect(self.select_row)
        self.jobs.partial.connect(self.job_partial)
        self.jobs.activityChanged.connect(self.update_progress_visibility)
        self.player.stateChanged.connect(self.update_play_button)
        QtGui.QGuiApplication.styleHints().colorSchemeChanged.connect(self.apply_theme)
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
        self.source.currentIndexChanged.connect(self.show_spectrogram_source)
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
            **self.parameters.settings_values(),
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
        return self.parameters.options(region)

    def clear_plots(self):
        for curve in [
            *self.wave_curves,
            self.diff_curve,
            *self.segment_curves,
            *self.spectrum_curves,
        ]:
            curve.setData([], [])
        self.image.clear()

    def cancel_views(self):
        """Cancel all per-result work while a running batch keeps streaming rows."""
        for kind in list(self.jobs.latest):
            if kind != "batch":
                self.jobs.cancel_kind(kind)

    def invalidate(self, *_):
        self.compare_timer.stop()
        self.spectrum_timer.stop()
        self.redraw_timer.stop()
        self.reset_indicators()
        self.spectral_data = None
        self.spectral_key = None
        self.jobs.cancel_all()
        self.player.stop()
        self.result = None
        self.reset_playback()
        self.spectral_ranges = None
        self.results.clear()
        self.detail_cache.clear()
        self.view_cache.clear()
        self.clear_plots()
        self.metrics.clear_report()
        self.refresh_actions()
        self.message("Settings changed; compare again.")

    def browse(self, side, folder):
        old = self.paths[side].text()
        initial = old if Path(old).is_dir() else str(Path(old).parent)
        if folder:
            path = QtWidgets.QFileDialog.getExistingDirectory(self, self.tr("Folder…"), initial)
        else:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("File…"),
                initial,
                "WAV (*.wav *.WAV *.rf64 *.RF64 *.w64 *.W64);;* (*)",
            )
        if path:
            self.paths[side].setText(QtCore.QDir.toNativeSeparators(path))

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
            self.batch_mode = not (Path(a).is_file() and Path(b).is_file())
            self.message("Loading…")
            if not self.batch_mode:
                self.jobs.submit(
                    "detail", lambda cancel, progress: compare(a, b, options, cancel, progress)
                )
            else:
                self.jobs.submit(
                    "batch",
                    lambda cancel, progress, emit: run_batch(
                        discover(a, b, cancel), options, cancel, progress, emit
                    ),
                    streaming=True,
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
        self.cancel_views()
        self.player.stop()
        self.refresh_actions()
        self.message("Cancelled")

    def progress_changed(self, value, text):
        self.progress_bar.setValue(round(value * 1000))
        self.message(text)

    def update_progress_visibility(self):
        # Only analysis work reports progress; hide stale values once it stops.
        busy = any(kind in self.jobs.latest for kind in ("batch", "detail"))
        if not busy:
            self.progress_bar.setValue(0)
        self.progress_bar.setVisible(busy)

    def analyzable(self, index):
        return self.rows[index]["status"] not in ANALYZABLE_EXCLUDED

    def job_partial(self, kind, row):
        if kind != "batch":
            return
        self.results.append_row(row)
        self.refresh_actions()
        index = len(self.rows) - 1
        # Open the remembered path as soon as it arrives, or else the first analyzable row.
        wanted = self.preferred_row is None or row["name"] == self.preferred_row
        if (
            not self.batch_cancelled
            and self.results.selected_index() is None
            and wanted
            and self.analyzable(index)
        ):
            self.results.select_index(index)

    def job_done(self, kind, outcome):
        success, value = outcome
        if not success:
            self.message(value)
            return
        if kind == "batch":
            selected = self.results.selected_index()
            selected_name = self.rows[selected]["name"] if selected is not None else None
            self.results.set_rows(value)
            if selected_name is not None:
                # Streaming already opened this row; restore the highlight without reloading.
                index = next(i for i, row in enumerate(self.rows) if row["name"] == selected_name)
                with QtCore.QSignalBlocker(self.table):
                    self.results.select_index(index)
            elif not self.batch_cancelled:
                candidates = [i for i in range(len(self.rows)) if self.analyzable(i)]
                chosen = next(
                    (i for i in candidates if self.rows[i]["name"] == self.preferred_row),
                    next(iter(candidates), None),
                )
                if chosen is not None:
                    self.results.select_index(chosen)
            self.refresh_actions()
            if self.batch_cancelled:
                self.message("Cancelled")
            elif self.result is None and not self.jobs.latest:
                self.message(f"{self.tr('Complete')} · {len(self.rows)}")
        elif kind == "detail":
            self.set_result(value)
            if not self.rows:
                self.results.set_rows(
                    [
                        {
                            "name": Path(value.a_path).name,
                            "a_path": value.report["a"]["path"],
                            "b_path": value.report["b"]["path"],
                            **value.report,
                        }
                    ]
                )
                with QtCore.QSignalBlocker(self.table):
                    self.results.select_index(0)
                self.refresh_actions()
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
            if value is None:
                self.message("No further differences")
            else:
                self.zoom_to(value)

    @staticmethod
    def detail_key(a, b, options):
        """Cache key for a detail result; a file change on disk yields a new key."""
        sa, sb = Path(a).stat(), Path(b).stat()
        return (str(a), str(b), sa.st_mtime_ns, sb.st_mtime_ns, sa.st_size, sb.st_size, options)

    def select_row(self, index=None):
        index = self.results.selected_index() if index is None else index
        if index is None:
            return
        row = self.rows[index]
        self.preferred_row = row["name"]
        self.spectrum_timer.stop()
        self.cancel_views()
        self.player.stop()
        self.result = None
        self.reset_playback()
        self.refresh_actions()
        self.view_cache.clear()
        self.clear_plots()
        self.reset_indicators()
        self.spectral_data = None
        if row["status"] in ("missing", "collision"):
            self.metrics.show_row(row)
            return
        try:
            options = self.options()
            a, b = str(Path(row["a_path"]).resolve()), str(Path(row["b_path"]).resolve())
            cached = self.detail_cache.get(self.detail_key(a, b, options))
            if cached:
                self.set_result(cached)
            else:
                self.metrics.clear_report()
                self.message("Loading…")
                self.jobs.submit(
                    "detail", lambda cancel, progress: compare(a, b, options, cancel, progress)
                )
        except Exception as error:
            self.message(str(error))

    def set_result(self, result):
        self.player.stop()
        self.cancel_views()
        self.reset_indicators()
        self.spectral_data = None
        self.spectral_key = None
        self.navigation_sample = None
        for line in self.navigation_lines:
            line.hide()
        self.result = result
        self.refresh_actions()
        self.spectral_ranges = None
        a, b = result.report["a"], result.report["b"]
        key = (a["path"], b["path"], a["mtime_ns"], b["mtime_ns"], a["size"], b["size"])
        key = (*key, result.options)
        self.detail_cache[key] = result
        self.detail_cache.move_to_end(key)
        while len(self.detail_cache) > 2:
            self.detail_cache.popitem(last=False)
        self.view_cache.clear()
        # Keep the displayed channel when the new result has it (e.g. switching batch rows).
        previous = max(0, self.channel.currentIndex())
        with QtCore.QSignalBlocker(self.channel):
            self.channel.clear()
            self.channel.addItems(
                [f"{m['a_channel']} → {m['b_channel']}" for m in result.report["metrics"]]
            )
            self.channel.setCurrentIndex(previous if previous < self.channel.count() else 0)
        duration = result.frames / result.rate
        self.seek.set_duration(duration, result.rate)
        self.update_overview()
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

    def update_overview(self):
        if self.result:
            self.seek.set_overview(
                *difference_overview(self.result, max(0, self.channel.currentIndex()))
            )

    def channel_changed(self):
        self.reset_indicators()
        self.spectral_data = None
        self.navigation_sample = None
        for line in self.navigation_lines:
            line.hide()
        for kind in ("view", "spectrum", "segments", "navigate"):
            self.jobs.cancel_kind(kind)
        self.player.stop()
        self.update_overview()
        self.reset_zoom()

    def show_metrics(self):
        if self.result:
            self.metrics.show_report(self.result.report, max(0, self.channel.currentIndex()))

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
            line.setPos(self.playback_position)
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
            store, size = result.segment_data(), result.options.segment_size
            start = max(0, int(lo * result.rate / size))
            stop = min(len(store), int(hi * result.rate / size) + 1)
            if stop <= start:
                return np.array([]), np.array([]), np.array([])
            stride = max(1, int(np.ceil((stop - start) / pixels)))
            # Aggregate without dropping high-error segments; blocks are stride-aligned.
            x, maxima, means = [], [], []
            block = stride * max(1, 65536 // stride)
            with closing(store.blocks(start, stop, channel, block)) as blocks:
                for first, rows in blocks:
                    cancel.check()
                    edges = np.arange(0, len(rows), stride)
                    x.append((first + edges) * size / result.rate)
                    maxima.append(np.maximum.reduceat(rows[:, 0], edges))
                    counts = np.maximum(1, np.add.reduceat(rows[:, 3], edges))
                    means.append(np.add.reduceat(rows[:, 1], edges) / counts)
            return np.concatenate(x), np.concatenate(maxima), np.concatenate(means)

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
        self.cancel_views()
        self.player.stop()
        self.message("Loading…")
        self.jobs.submit(
            "detail", lambda cancel, progress: compare(a, b, options, cancel, progress)
        )

    def update_spectrum(self, *_):
        self.spectrum_timer.stop()
        if not self.result or self.tabs.currentIndex() == 0:
            self.jobs.cancel_kind("spectrum")
            return
        result = self.result
        begin, end = self.selection_editor.seconds
        ch = max(0, self.channel.currentIndex())
        fft, hop = int(self.fft.currentText()), self.hop.value()
        key = (id(result), self.selection_editor.bounds, ch, fft, hop)
        # Switching between spectral tabs reuses a finished or still-running identical request.
        if key == self.spectral_key and (
            self.spectral_data is not None or "spectrum" in self.jobs.latest
        ):
            return
        self.reset_indicators(("spectrum", "spectrogram"))
        self.spectral_data = None
        self.spectral_key = key
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

    def show_spectrogram_source(self, *_):
        """All sources were analyzed together; switching only swaps the displayed image."""
        self.reset_indicators(("spectrogram",))
        if self.spectral_data is not None:
            images = self.spectral_data[3]
            self.image.setImage(
                images[self.source.currentIndex()], levels=(-120, 0), autoLevels=False
            )

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
        if not self.result:
            return
        center = sample / self.result.rate
        self.wave_plot.setXRange(
            max(0, center - 0.1),
            min(self.result.frames / self.result.rate, center + 0.1),
            padding=0.1,
        )
        self.navigation_sample = int(sample)
        for line in self.navigation_lines:
            line.setPos(center)
            line.show()
        # A stopped, unlooped player starts from the difference so it can be heard directly.
        if self.player.state == "stopped" and not self.loop.isChecked():
            self.play_position(center)

    def next_difference(self, direction):
        if not self.result:
            return
        result, channel = self.result, max(0, self.channel.currentIndex())
        size, threshold = result.options.segment_size, result.options.threshold
        # Continue from the last visited difference, else from (and including) playback.
        if self.navigation_sample is None:
            current = first = round(self.playback_position * result.rate)
        else:
            current, first = self.navigation_sample, self.navigation_sample + 1

        def find(cancel, progress):
            def scan(start, end):
                chunks = range(start, end, 65536)
                for chunk in chunks if direction > 0 else reversed(chunks):
                    cancel.check()
                    d = result.samples(chunk, min(65536, end - chunk))[2][:, channel]
                    hits = np.flatnonzero(np.abs(d) > threshold)
                    if len(hits):
                        return chunk + int(hits[0 if direction > 0 else -1])
                return None

            store = result.segment_data()
            if direction > 0:
                blocks = store.blocks(first // size, len(store), channel)
            else:
                blocks = store.blocks(0, current // size + 1, channel, reverse=True)
            # Read segment maxima in blocks, then scan samples only inside candidate segments.
            with closing(blocks):
                for base, rows in blocks:
                    cancel.check()
                    hits = np.flatnonzero(rows[:, 0] > threshold) + base
                    for index in hits if direction > 0 else hits[::-1]:
                        start = int(index) * size
                        end = min(result.frames, start + size)
                        found = (
                            scan(max(start, first), end)
                            if direction > 0
                            else scan(start, min(end, current))
                        )
                        if found is not None:
                            return found
            return None

        self.jobs.submit("navigate", find)

    def reset_playback(self):
        self.selection_editor.set_context(1, 0)
        self.seek.set_duration(0, 1)
        self.playback_position = 0.0
        self.navigation_sample = None
        self.time_label.setText(f"{format_clock(0)} / {format_clock(0)}")
        for line in self.cursors:
            line.setPos(0)
        for line in self.navigation_lines:
            line.hide()

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

    def toggle_playback(self):
        """Play when stopped; otherwise pause or resume (Space and the play button)."""
        if self.player.state == "stopped":
            self.play()
        else:
            self.pause_playback()

    def update_play_button(self, *_):
        playing = self.player.state == "playing"
        self.play_button.setText(self.tr("Pause" if playing else "Play"))
        self.play_button.setIcon(
            self.style().standardIcon(
                QtWidgets.QStyle.SP_MediaPause if playing else QtWidgets.QStyle.SP_MediaPlay
            )
        )

    def play_position(self, seconds):
        if self.result:
            self.playback_position = max(0.0, min(seconds, self.seek.duration))
            if self.seek.interacting:
                return
            self.seek.set_position(self.playback_position)
            self.time_label.setText(
                f"{format_clock(self.playback_position)} / {format_clock(self.seek.duration)}"
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
        # Single-file export reflects the current (possibly selection) analysis.
        if self.result and not self.batch_mode:
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
        self.parameters.retranslate(self.tr)
        self.results.retranslate()
        self.metrics.retranslate()
        self.update_play_button()
        for i, text in enumerate(("Waveform", "Spectrum", "Spectrogram")):
            self.tabs.setTabText(i, self.tr(text))
        for i, text in enumerate(("System", "Light", "Dark")):
            self.theme_combo.setItemText(i, self.tr(text))
        for plot, title in (
            (self.wave_plot, "A / B waveforms"),
            (self.diff_plot, "B − A difference"),
            (self.segment_plot, "Segment differences"),
        ):
            plot.setTitle(self.tr(title), color=self.foreground)
            plot.setLabel("bottom", self.tr("Time (s)"))
            plot.setLabel("left", self.tr("Amplitude"))
        self.spectrum_plot.setLabel("bottom", self.tr("Frequency (Hz)"))
        self.spectrum_plot.setLabel("left", self.tr("PSD (dB/Hz)"))
        self.spectrogram_plot.setLabel("bottom", self.tr("Time (s)"))
        self.spectrogram_plot.setLabel("left", self.tr("Frequency (Hz)"))

    def apply_theme(self):
        choice = self.theme_combo.currentData()
        dark = choice == "dark" or (
            choice == "system"
            and QtGui.QGuiApplication.styleHints().colorScheme() == QtCore.Qt.ColorScheme.Dark
        )
        bg, panel, fg, border, muted = THEMES[dark]
        self.foreground = fg
        for indicator in self.indicators.values():
            indicator.apply_theme(dark)
        self.seek.apply_theme(dark)
        # A matching palette themes natively drawn parts: check marks, scroll bars, tooltips.
        palette = QtGui.QPalette()
        for role, color in (
            (QtGui.QPalette.Window, bg),
            (QtGui.QPalette.WindowText, fg),
            (QtGui.QPalette.Base, panel),
            (QtGui.QPalette.AlternateBase, bg),
            (QtGui.QPalette.Text, fg),
            (QtGui.QPalette.Button, panel),
            (QtGui.QPalette.ButtonText, fg),
            (QtGui.QPalette.ToolTipBase, panel),
            (QtGui.QPalette.ToolTipText, fg),
            (QtGui.QPalette.PlaceholderText, muted),
            (QtGui.QPalette.Mid, border),
            (QtGui.QPalette.Highlight, "#277ab8"),
            (QtGui.QPalette.HighlightedText, "#ffffff"),
        ):
            palette.setColor(role, QtGui.QColor(color))
        for role in (QtGui.QPalette.Text, QtGui.QPalette.ButtonText, QtGui.QPalette.WindowText):
            palette.setColor(QtGui.QPalette.Disabled, role, QtGui.QColor(muted))
        self.setPalette(palette)
        QtWidgets.QToolTip.setPalette(palette)
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {bg}; color: {fg}; font-size: 12px; }}
            QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTableWidget, QTreeWidget {{
                background: {panel}; border: 1px solid {border}; border-radius: 4px; padding: 5px; }}
            QTableWidget, QTreeWidget {{ alternate-background-color: {bg}; }}
            QLineEdit[invalid="true"] {{ border: 1px solid #e5534b; }}
            QAbstractSpinBox:disabled, QComboBox:disabled, QLineEdit:disabled,
            QCheckBox:disabled, QLabel:disabled {{ color: {muted}; }}
            QGroupBox {{ border: 1px solid {border}; border-radius: 6px; margin-top: 16px;
                padding-top: 4px; font-weight: 600; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 4px; }}
            QCheckBox::indicator {{ width: 12px; height: 12px; border-radius: 3px;
                border: 1px solid {muted}; background: {panel}; }}
            QCheckBox::indicator:checked {{ background: #277ab8; border-color: #277ab8; }}
            QCheckBox::indicator:disabled {{ border-color: {border}; }}
            QCheckBox::indicator:checked:disabled {{ background: {muted}; border-color: {muted}; }}
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
            legend = plot.getPlotItem().legend
            if legend is not None:
                legend.setLabelTextColor(fg)
        for plot in (self.wave_plot, self.diff_plot, self.segment_plot):
            plot.setTitle(plot.getPlotItem().titleLabel.text, color=fg)

    def about(self):
        QtWidgets.QMessageBox.about(
            self,
            self.tr("About"),
            f"WAV Compare {get_version()}\nMIT · sunging\nhttps://github.com/sunging/wav_compare",
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and all(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()

    @staticmethod
    def dropped_paths(mime):
        if not mime.hasUrls() or not all(url.isLocalFile() for url in mime.urls()):
            return []
        return [QtCore.QDir.toNativeSeparators(url.toLocalFile()) for url in mime.urls()]

    def drop_paths(self, paths, side=None):
        """Two paths set A/B; one path fills the target side, else the first empty side."""
        if len(paths) >= 2:
            self.set_paths(paths[:2])
            return
        values = [edit.text() for edit in self.paths]
        if side is None:
            side = 0 if not values[0].strip() else 1
        values[side] = paths[0]
        self.set_paths(values)

    def dropEvent(self, event):
        paths = self.dropped_paths(event.mimeData())
        if paths:
            self.drop_paths(paths)
            event.acceptProposedAction()

    def eventFilter(self, watched, event):
        if watched in getattr(self, "paths", ()):
            kind = event.type()
            if kind in (QtCore.QEvent.DragEnter, QtCore.QEvent.DragMove):
                if self.dropped_paths(event.mimeData()):
                    event.acceptProposedAction()
                    return True
            elif kind == QtCore.QEvent.Drop:
                paths = self.dropped_paths(event.mimeData())
                if paths:
                    self.drop_paths(paths, self.paths.index(watched))
                    event.acceptProposedAction()
                    return True
        return super().eventFilter(watched, event)

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
