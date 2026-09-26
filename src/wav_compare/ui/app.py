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

COLORS = ("#52b9f5", "#ffb454", "#74dbb0")


class Window(QtWidgets.QMainWindow):
    def __init__(self, paths=(), settings=None):
        super().__init__()
        self.settings = settings or QtCore.QSettings("sunging", "wav-compare")
        default_language = (
            "zh" if QtCore.QLocale.system().language() == QtCore.QLocale.Chinese else "en"
        )
        self.language = self.settings.value("language", default_language)
        self.bindings = []
        self.result = None
        self.spectral_ranges = None
        self.spectral_data = None
        self.rows = []
        self.detail_cache = OrderedDict()
        self.view_cache = ByteCache()
        self.closing = False
        self.jobs = Jobs(self)
        self.jobs.done.connect(self.job_done)
        self.jobs.progress.connect(self.progress_changed)
        self.jobs.idle.connect(self.on_idle)
        self.player = Player(self)
        self.player.error.connect(self.message)
        self.player.position.connect(self.play_position)
        self.setWindowTitle("WAV Compare")
        self.resize(1440, 940)
        self.setMinimumSize(1000, 680)
        self.setAcceptDrops(True)
        self._build()
        self._restore(paths)
        self._connect()
        self.retranslate()
        self.apply_theme()
        self.message("Drop two WAV files or folders here, then compare.")

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

    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(16, 12, 16, 12)
        heading = QtWidgets.QHBoxLayout()
        logo = QtWidgets.QLabel("WAV Compare")
        logo.setStyleSheet("font-size: 23px; font-weight: 700;")
        heading.addWidget(logo)
        heading.addWidget(self.label("Audio analysis workbench"))
        heading.addStretch()
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItem("中文", "zh")
        self.language_combo.addItem("English", "en")
        heading.addWidget(self.language_combo)
        self.theme_combo = QtWidgets.QComboBox()
        for key in ("System", "Light", "Dark"):
            self.theme_combo.addItem(key, key.lower())
        heading.addWidget(self.theme_combo)
        self.button("About", self.about, heading)
        layout.addLayout(heading)
        inputs = QtWidgets.QGridLayout()
        self.paths = []
        for i, text in enumerate(("Reference A", "Candidate B")):
            inputs.addWidget(self.label(text), i, 0)
            edit = QtWidgets.QLineEdit()
            edit.setClearButtonEnabled(True)
            self.paths.append(edit)
            inputs.addWidget(edit, i, 1)
            inputs.addWidget(self.button("File…", lambda _, n=i: self.browse(n, False)), i, 2)
            inputs.addWidget(self.button("Folder…", lambda _, n=i: self.browse(n, True)), i, 3)
        inputs.addWidget(self.button("Swap A / B", self.swap), 0, 4)
        self.compare_button = self.button("Compare", self.start_compare)
        self.compare_button.setObjectName("primary")
        inputs.addWidget(self.compare_button, 1, 4)
        layout.addLayout(inputs)
        self.splitter = QtWidgets.QSplitter()
        layout.addWidget(self.splitter, 1)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 8, 8, 0)
        ll.addWidget(self.label("Files & results"))
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
        self.button("Export JSON…", lambda: self.export("json"), ll)
        self.button("Export CSV…", lambda: self.export("csv"), ll)
        self.button("Clear cache", self.clear_cache, ll)
        self.splitter.addWidget(left)
        middle = QtWidgets.QWidget()
        ml = QtWidgets.QVBoxLayout(middle)
        ml.setContentsMargins(4, 8, 4, 0)
        nav = QtWidgets.QHBoxLayout()
        self.button("Reset zoom", self.reset_zoom, nav)
        self.button("Largest difference", self.largest, nav)
        self.button("Previous difference", lambda: self.next_difference(-1), nav)
        self.button("Next difference", lambda: self.next_difference(1), nav)
        ml.addLayout(nav)
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
        self.region = pg.LinearRegionItem((0, 1), brush=pg.mkBrush(90, 160, 220, 25))
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
        selection = QtWidgets.QHBoxLayout()
        self.begin, self.end = QtWidgets.QDoubleSpinBox(), QtWidgets.QDoubleSpinBox()
        for text, control in (("Selection start (s)", self.begin), ("Selection end (s)", self.end)):
            control.setRange(0, 10**8)
            control.setDecimals(6)
            selection.addWidget(self.label(text))
            selection.addWidget(control)
        ml.addLayout(selection)
        select_actions = QtWidgets.QHBoxLayout()
        self.button("Analyze selection", lambda: self.analyze_region(True), select_actions)
        self.button("Full comparison", lambda: self.analyze_region(False), select_actions)
        self.button("Update spectrum", self.update_spectrum, select_actions)
        ml.addLayout(select_actions)
        self.splitter.addWidget(middle)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(8, 8, 0, 0)
        rl.addWidget(self.label("Parameters and metrics"))
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        panel = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(panel)
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
        self.splitter.setSizes([270, 840, 300])
        for i in range(3):
            self.splitter.setCollapsible(i, False)
        playback = QtWidgets.QHBoxLayout()
        self.source = QtWidgets.QComboBox()
        self.source.addItems(["A", "B", "B − A"])
        playback.addWidget(self.source)
        self.original = self.bind(QtWidgets.QCheckBox(), "Original input")
        self.loop = self.bind(QtWidgets.QCheckBox(), "Loop selection")
        playback.addWidget(self.original)
        self.button("Play", self.play, playback)
        self.button("Pause / resume", self.player.pause, playback)
        self.button("Stop", self.player.stop, playback)
        playback.addWidget(self.loop)
        playback.addWidget(self.label("Volume"))
        self.volume = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(50)
        self.volume.setMaximumWidth(100)
        playback.addWidget(self.volume)
        self.seek = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.seek.setRange(0, 10000)
        playback.addWidget(self.seek, 1)
        layout.addLayout(playback)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setMaximumWidth(180)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.cancel_button = self.button("Cancel", self.cancel_tasks)
        self.statusBar().addPermanentWidget(self.cancel_button)
        self.redraw_timer = QtCore.QTimer(self)
        self.redraw_timer.setSingleShot(True)
        self.redraw_timer.setInterval(50)
        self.redraw_timer.timeout.connect(self.redraw)
        for shortcut, callback in (
            ("Ctrl+Return", self.start_compare),
            ("Escape", self.cancel_tasks),
            ("Space", self.player.pause),
            ("Ctrl+0", self.reset_zoom),
        ):
            action = QtGui.QShortcut(QtGui.QKeySequence(shortcut), self)
            action.activated.connect(callback)

    def _restore(self, paths):
        for name, indicator in self.indicators.items():
            indicator.toggle.setChecked(self.settings.value(f"indicators/{name}", True, type=bool))
        for i, edit in enumerate(self.paths):
            edit.setText(paths[i] if len(paths) == 2 else self.settings.value(f"path{i}", ""))
        self.language_combo.setCurrentIndex(self.language_combo.findData(self.language))
        theme = self.settings.value("theme", "system")
        self.theme_combo.setCurrentIndex(max(0, self.theme_combo.findData(theme)))
        try:
            options = json.loads(self.settings.value("options", "{}"))
            for name in ("strict", "align", "mix"):
                getattr(self, name).setChecked(options.get(name, name == "align"))
            for name, key in (
                ("max_lag", "max_lag"),
                ("threshold", "threshold"),
                ("segment", "segment_size"),
            ):
                getattr(self, name).setValue(options.get(key, getattr(self, name).value()))
            self.mode.setCurrentText(options.get("mode", "float64"))
            self.manual.setChecked(options.get("offset") is not None)
            self.offset.setValue(options.get("offset") or 0)
            self.pairs.setText(",".join(f"{a + 1}:{b + 1}" for a, b in options.get("pairs", [])))
        except (ValueError, TypeError):
            pass
        geometry = self.settings.value("geometry")
        split = self.settings.value("splitter")
        if geometry:
            self.restoreGeometry(geometry)
        if split:
            self.splitter.restoreState(split)

    def _connect(self):
        for edit in self.paths:
            edit.textChanged.connect(self.invalidate)
            edit.returnPressed.connect(self.start_compare)
        for control in (self.strict, self.align, self.manual, self.mix):
            control.toggled.connect(self.invalidate)
        for control in (self.max_lag, self.offset, self.threshold, self.segment):
            control.valueChanged.connect(self.invalidate)
        self.mode.currentIndexChanged.connect(self.invalidate)
        self.pairs.textChanged.connect(self.invalidate)
        self.filter.textChanged.connect(self.filter_rows)
        self.table.itemSelectionChanged.connect(self.select_row)
        self.channel.currentIndexChanged.connect(self.channel_changed)
        self.channel.currentIndexChanged.connect(self.show_metrics)
        self.channel.currentIndexChanged.connect(self.update_spectrum)
        self.language_combo.currentIndexChanged.connect(self.language_changed)
        self.theme_combo.currentIndexChanged.connect(self.apply_theme)
        self.wave_plot.sigXRangeChanged.connect(lambda: self.redraw_timer.start())
        self.region.sigRegionChangeFinished.connect(self.region_changed)
        self.begin.valueChanged.connect(self.spin_region)
        self.end.valueChanged.connect(self.spin_region)
        self.tabs.currentChanged.connect(self.update_spectrum)
        self.source.currentIndexChanged.connect(self.update_spectrum)
        self.volume.valueChanged.connect(lambda n: self.player.set_volume(n / 100))
        self.seek.sliderReleased.connect(self.seek_play)

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
        self.reset_indicators()
        self.spectral_data = None
        self.jobs.cancel_all()
        self.player.stop()
        self.result = None
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
        self.paths[0].setText(b)
        self.paths[1].setText(a)

    def start_compare(self):
        try:
            options = self.options()
            a, b = [edit.text().strip() for edit in self.paths]
            self.invalidate()
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

    def cancel_tasks(self):
        # Keep the batch's final partial report, but prevent queued view/detail work.
        for job in self.jobs.jobs.values():
            job.cancel.cancel()
        self.player.stop()
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
        self.jobs.cancel_all()
        self.player.stop()
        self.result = None
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
        self.jobs.cancel_all()
        self.reset_indicators()
        self.spectral_data = None
        self.result = result
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
        self.region.setBounds((0, result.frames / result.rate))
        self.region.setRegion((0, min(10.0, result.frames / result.rate)))
        self.region_changed()
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
        lo, hi = self.region.getRegion()
        for control, value in ((self.begin, lo), (self.end, hi)):
            blocker = QtCore.QSignalBlocker(control)
            control.setValue(value)
            del blocker

    def spin_region(self):
        if self.begin.value() < self.end.value():
            self.region.setRegion((self.begin.value(), self.end.value()))

    def analyze_region(self, selected):
        if not self.result:
            return
        result = self.result
        # Convert display-relative selection to the full aligned timeline.
        origin = result.options.region[0] if result.options.region else 0
        region = (origin + self.begin.value(), origin + self.end.value()) if selected else None
        options = replace(result.options, region=region)
        a, b = result.report["a"]["path"], result.report["b"]["path"]
        self.jobs.cancel_all()
        self.player.stop()
        self.jobs.submit(
            "detail", lambda cancel, progress: compare(a, b, options, cancel, progress)
        )

    def update_spectrum(self, *_):
        self.reset_indicators(("spectrum", "spectrogram"))
        self.spectral_data = None
        self.jobs.cancel_kind("spectrum")
        if not self.result or self.tabs.currentIndex() == 0:
            return
        result = self.result
        begin, end, ch = self.begin.value(), self.end.value(), max(0, self.channel.currentIndex())
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
            if not frequencies[0] <= point.x() <= frequencies[-1]:
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

    def play(self, begin=None):
        if self.result:
            start = self.begin.value() if begin is None or isinstance(begin, bool) else begin
            self.player.play(
                self.result,
                self.source.currentIndex(),
                max(0, self.channel.currentIndex()),
                start,
                self.end.value(),
                self.loop.isChecked(),
                self.original.isChecked(),
            )

    def play_position(self, seconds):
        if self.result:
            self.seek.setValue(round(seconds / (self.result.frames / self.result.rate) * 10000))
            for line in self.cursors:
                line.setPos(seconds)

    def seek_play(self):
        if self.result:
            position = self.seek.value() / 10000 * self.result.frames / self.result.rate
            if position < self.end.value():
                self.play(position)

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
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{ background: {bg}; color: {fg}; font-size: 12px; }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTableWidget {{
                background: {panel}; border: 1px solid {border}; border-radius: 4px; padding: 5px; }}
            QPushButton {{ background: {panel}; border: 1px solid {border}; border-radius: 5px; padding: 7px 10px; }}
            QPushButton:hover {{ border-color: #4298d8; }}
            QPushButton#primary {{ background: #277ab8; color: white; font-weight: 600; }}
            QHeaderView::section {{ background: {panel}; padding: 5px; border: none; }}
            QTabBar::tab {{ padding: 8px 16px; }}
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
        for edit, path in zip(self.paths, paths[:2], strict=False):
            edit.setText(path)
        event.acceptProposedAction()

    def closeEvent(self, event):
        self.reset_indicators()
        self.player.stop()
        if self.jobs.jobs:
            self.closing = True
            self.jobs.cancel_all()
            self.message("Closing after background tasks stop…")
            event.ignore()
            return
        self.settings.setValue("language", self.language)
        self.settings.setValue("theme", self.theme_combo.currentData())
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("splitter", self.splitter.saveState())
        for name, indicator in self.indicators.items():
            self.settings.setValue(f"indicators/{name}", indicator.toggle.isChecked())
        for i, path in enumerate(self.paths):
            self.settings.setValue(f"path{i}", path.text())
        try:
            self.settings.setValue("options", json.dumps(self.options().report()))
        except ValueError:
            pass
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
