import json
import threading
from pathlib import Path

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui import app as app_module
from wav_compare.ui.app import Window
from wav_compare.ui.preferences import PreferencesDialog


@pytest.fixture
def settings(tmp_path):
    return QtCore.QSettings(str(tmp_path / "preferences.ini"), QtCore.QSettings.IniFormat)


@pytest.fixture
def window(qtbot, settings):
    widget = Window(settings=settings)
    widget.show()
    yield widget
    widget.close()
    # Own teardown: pytest-qt's automatic deleteLater can precede queued worker completion.
    qtbot.waitUntil(lambda: not widget.jobs.jobs, timeout=10000)
    widget.deleteLater()


def wait_result(qtbot, window):
    qtbot.waitUntil(lambda: window.result is not None, timeout=10000)
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=10000)


def track_comparisons(monkeypatch):
    calls = []

    def tracked(a, b, options, *args, **kwargs):
        calls.append((str(a), str(b), options))
        return compare(a, b, options, *args, **kwargs)

    monkeypatch.setattr(app_module, "compare", tracked)
    return calls


def test_automatic_files_debounce_swap_and_manual(qtbot, window, audio, monkeypatch):
    calls = track_comparisons(monkeypatch)
    a = audio("a.wav", np.zeros(800))
    b = audio("b.wav", np.ones(800) * 0.01)
    window.set_paths((a, b))
    for threshold in (0.001, 0.002, 0.003):
        window.threshold.setValue(threshold)
        qtbot.wait(50)
    assert not calls
    wait_result(qtbot, window)
    assert len(calls) == 1 and calls[0][2].threshold == 0.003
    window.swap()
    wait_result(qtbot, window)
    assert len(calls) == 2 and tuple(map(Path, calls[-1][:2])) == (b, a)
    window.threshold.setValue(0.004)
    window.compare_action.trigger()
    wait_result(qtbot, window)
    qtbot.wait(500)
    assert len(calls) == 3
    assert window.paths[0].toolTip() == str(b)


def test_browse_and_drop_trigger_once(qtbot, window, audio, monkeypatch):
    calls = track_comparisons(monkeypatch)
    a = audio("a.wav", np.zeros(80))
    b = audio("b.wav", np.zeros(80))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getOpenFileName", lambda *args: (str(a), ""))
    window.browse(0, False)
    qtbot.wait(450)
    assert not calls
    window.browse(1, False)
    wait_result(qtbot, window)
    mime = QtCore.QMimeData()
    mime.setUrls([QtCore.QUrl.fromLocalFile(str(b)), QtCore.QUrl.fromLocalFile(str(a))])
    event = QtGui.QDropEvent(
        QtCore.QPointF(), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
    )
    window.dropEvent(event)
    wait_result(qtbot, window)
    assert event.isAccepted()
    assert len(calls) == 2 and tuple(map(Path, calls[-1][:2])) == (b, a)


@pytest.mark.parametrize("case", ["empty", "missing", "mixed", "pairs", "conflicting"])
def test_invalid_inputs_never_submit(window, audio, tmp_path, qtbot, monkeypatch, case):
    calls = track_comparisons(monkeypatch)
    path = audio("a.wav", np.zeros(80))
    values = {
        "empty": ("", ""),
        "missing": (path, tmp_path / "missing.wav"),
        "mixed": (path, tmp_path),
        "pairs": (path, path),
        "conflicting": (path, path),
    }
    window.set_paths(values[case])
    if case == "pairs":
        window.pairs.setText("1:")
    elif case == "conflicting":
        window.pairs.setText("1:1")
        window.mix.setChecked(True)
    window.start_compare()
    qtbot.wait(450)
    assert not calls and not window.jobs.jobs
    assert window.statusBar().currentMessage()


def test_disable_cancel_clear_and_close_stop_pending(qtbot, window, audio, monkeypatch):
    calls = track_comparisons(monkeypatch)
    path = audio("a.wav", np.zeros(80))
    window.set_paths((path, path))
    window.auto_compare_action.trigger()
    qtbot.wait(450)
    assert not calls
    window.threshold.setValue(0.1)
    assert not window.compare_timer.isActive()
    window.auto_compare_action.trigger()
    window.cancel_tasks()
    qtbot.wait(450)
    assert not calls
    window.threshold.setValue(0.2)
    window.clear_cache()
    qtbot.wait(450)
    assert not calls
    window.threshold.setValue(0.3)
    window.close()
    qtbot.wait(450)
    assert not calls


def test_old_running_result_and_progress_cannot_replace_new(qtbot, window, audio, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    calls = []
    path = audio("a.wav", np.zeros(80))

    def slow(a, b, options, cancel, progress):
        calls.append(options.threshold)
        if options.threshold == 0:
            entered.set()
            release.wait(5)
            progress(0.99, "obsolete progress")
        # Deliberately emulate a worker that ignores cancellation.
        return compare(a, b, options)

    monkeypatch.setattr(app_module, "compare", slow)
    try:
        window.set_paths((path, path))
        qtbot.waitUntil(entered.is_set, timeout=3000)
        window.threshold.setValue(0.1)
        qtbot.waitUntil(lambda: window.result is not None, timeout=5000)
        assert window.result.options.threshold == 0.1
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    assert window.result.options.threshold == 0.1
    assert window.statusBar().currentMessage() != "obsolete progress"
    assert calls == [0, 0.1]


def test_disable_does_not_cancel_running_comparison(qtbot, window, audio, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    path = audio("a.wav", np.zeros(80))

    def slow(a, b, options, cancel, progress):
        entered.set()
        release.wait(5)
        return compare(a, b, options, cancel, progress)

    monkeypatch.setattr(app_module, "compare", slow)
    window.set_paths((path, path))
    try:
        qtbot.waitUntil(entered.is_set, timeout=3000)
        window.auto_compare_action.trigger()
        assert all(not job.cancel.event.is_set() for job in window.jobs.jobs.values())
    finally:
        release.set()
    wait_result(qtbot, window)


def test_batch_auto_selects_and_preserves_selected_path(qtbot, window, audio, tmp_path):
    for folder in ("left", "right"):
        for name in ("a.wav", "b.wav"):
            audio(f"{folder}/{name}", np.zeros(80))
    window.set_paths((tmp_path / "left", tmp_path / "right"))
    wait_result(qtbot, window)
    assert len(window.rows) == 2
    index = next(i for i in range(2) if window.table.item(i, 0).text() == "b.wav")
    window.table.selectRow(index)
    wait_result(qtbot, window)
    window.threshold.setValue(0.1)
    wait_result(qtbot, window)
    assert window.preferred_row == "b.wav"
    assert window.result.report["a"]["path"].endswith("b.wav")
    assert len(window.rows) == 2


def test_cancel_batch_keeps_partial_report_without_detail(
    qtbot, window, audio, monkeypatch, tmp_path
):
    entered, release = threading.Event(), threading.Event()
    for folder in ("left", "right"):
        audio(f"{folder}/a.wav", np.zeros(80))

    def batch(rows, options, cancel, progress):
        entered.set()
        release.wait(5)
        assert cancel.event.is_set()
        return [{**row, "status": "cancelled"} for row in rows]

    monkeypatch.setattr(app_module, "run_batch", batch)
    calls = track_comparisons(monkeypatch)
    window.set_paths((tmp_path / "left", tmp_path / "right"))
    try:
        qtbot.waitUntil(entered.is_set, timeout=3000)
        window.cancel_tasks()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=3000)
    assert window.rows[0]["status"] == "cancelled"
    assert not calls and window.result is None
    assert not window.compare_timer.isActive()


@pytest.mark.parametrize(
    "startup, enabled, restore, expected",
    [
        (True, True, True, 1),
        (False, True, True, 0),
        (True, False, True, 0),
        (True, True, False, 0),
    ],
)
def test_startup_preferences(
    qtbot, settings, audio, monkeypatch, startup, enabled, restore, expected
):
    calls = track_comparisons(monkeypatch)
    path = audio("a.wav", np.zeros(80))
    for i in range(2):
        settings.setValue(f"path{i}", str(path))
    settings.setValue("automation/startup", startup)
    settings.setValue("automation/enabled", enabled)
    settings.setValue("restore_paths", restore)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    if expected:
        wait_result(qtbot, window)
    else:
        qtbot.wait(500)
    assert len(calls) == expected
    window.close()


def test_explicit_paths_override_history(qtbot, settings, audio):
    path = audio("explicit.wav", np.zeros(80))
    settings.setValue("restore_paths", False)
    settings.setValue("path0", "missing.wav")
    window = Window((path, path), settings)
    qtbot.addWidget(window)
    wait_result(qtbot, window)
    assert [edit.text() for edit in window.paths] == [str(path)] * 2
    window.close()


def test_spectrum_options_only_refresh_spectrum(qtbot, window, audio, monkeypatch):
    path = audio("a.wav", np.zeros(800))
    window.set_result(compare(path, path, Options(strict=True)))
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    calls = []
    original = app_module.spectra

    def tracked(*args):
        calls.append(args[4:6])
        return original(*args)

    monkeypatch.setattr(app_module, "spectra", tracked)
    comparisons = track_comparisons(monkeypatch)
    window.fft.setCurrentText("1024")
    window.hop.setValue(256)
    qtbot.wait(450)
    assert not calls
    window.tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: window.spectral_data is not None, timeout=5000)
    assert calls == [(1024, 256)]
    window.fft.setCurrentText("512")
    window.hop.setValue(128)
    window.hop.setValue(64)
    qtbot.waitUntil(lambda: window.spectral_data is not None, timeout=5000)
    assert calls[-1] == (512, 64) and len(calls) == 2
    window.region.setRegion((0.02, 0.08))
    window.selection_committed()
    qtbot.wait(450)
    assert len(calls) == 2 and not comparisons


def test_panels_expand_main_area_and_restore(qtbot, window, settings):
    qtbot.wait(50)
    window.splitter.setSizes([240, 800, 300])
    window.panel_sizes_changed()
    before = window.tabs.width()
    widths = window.splitter.sizes()
    for action in window.panel_actions:
        action.trigger()
    qtbot.wait(50)
    assert all(panel.isHidden() for panel in window.panels)
    assert window.tabs.width() > before + 400
    window.save_preferences()
    restored = Window(settings=settings)
    qtbot.addWidget(restored)
    restored.show()
    assert not any(action.isChecked() for action in restored.panel_actions)
    for action in restored.panel_actions:
        action.trigger()
    qtbot.wait(50)
    for actual, expected in zip(restored.splitter.sizes()[::2], widths[::2], strict=True):
        assert abs(actual - expected) < 20
    restored.threshold.setValue(0.125)
    restored.reset_layout()
    assert restored.threshold.value() == 0.125
    assert all(action.isChecked() for action in restored.panel_actions)
    restored.close()


def test_preferences_round_trip_and_valid_analysis_snapshot(qtbot, window, settings):
    window.language_combo.setCurrentIndex(window.language_combo.findData("zh"))
    window.theme_combo.setCurrentIndex(window.theme_combo.findData("dark"))
    window.strict.setChecked(True)
    window.manual.setChecked(True)
    window.offset.setValue(123)
    window.threshold.setValue(0.5)
    window.fft.setCurrentText("4096")
    window.hop.setValue(128)
    window.volume.setValue(23)
    window.source.setCurrentIndex(2)
    window.loop.setChecked(True)
    window.original.setChecked(True)
    window.indicators["wave"].toggle.setChecked(False)
    qtbot.waitUntil(lambda: settings.value("playback/volume") == 23, timeout=2000)
    assert json.loads(settings.value("options"))["offset"] is None
    window.pairs.setText("invalid")
    window.threshold.setValue(0.9)
    window.save_preferences()
    restored = Window(settings=settings)
    qtbot.addWidget(restored)
    assert restored.language == "zh" and restored.theme_combo.currentData() == "dark"
    assert restored.manual.isChecked() and restored.offset.value() == 123
    assert restored.threshold.value() == 0.5 and restored.pairs.text() == ""
    assert restored.fft.currentText() == "4096" and restored.hop.value() == 128
    assert restored.player.volume == 0.23
    assert restored.source.currentIndex() == 2
    assert restored.loop.isChecked() and restored.original.isChecked()
    assert not restored.indicators["wave"].toggle.isChecked()
    assert restored.result is None
    restored.close()


def test_old_and_corrupt_settings_restore_independently(qtbot, settings):
    settings.setValue(
        "options",
        json.dumps(
            {
                "threshold": "bad",
                "segment_size": 23,
                "max_lag": -1,
                "mode": "unknown",
                "offset": 42,
                "pairs": [[0, 1]],
                "strict": True,
            }
        ),
    )
    settings.setValue("language", "unknown")
    settings.setValue("theme", "unknown")
    settings.setValue("spectrum/hop", 999999)
    settings.setValue("spectrum/fft", "bad")
    settings.setValue("playback/volume", "nan")
    settings.setValue("geometry", "bad")
    settings.setValue("splitter", "bad")
    settings.setValue("layout/visible0", "bad")
    window = Window(settings=settings)
    qtbot.addWidget(window)
    assert window.language in ("zh", "en")
    assert window.theme_combo.currentData() == "system"
    assert window.threshold.value() == 0 and window.segment.value() == 23
    assert window.max_lag.value() == 5 and window.mode.currentText() == "float64"
    assert window.manual.isChecked() and window.offset.value() == 42
    assert window.pairs.text() == "1:2"
    assert window.hop.value() == 512 and window.fft.currentText() == "2048"
    assert window.volume.value() == 50 and window.panel_actions[0].isChecked()
    window.close()


def test_preferences_dialog_cancel_and_apply(window, qtbot, monkeypatch):
    def reject(dialog):
        dialog.language.setCurrentIndex(1 if window.language == "zh" else 0)
        dialog.auto_compare.setChecked(False)
        return QtWidgets.QDialog.Rejected

    monkeypatch.setattr(PreferencesDialog, "exec", reject)
    old_language = window.language
    window.edit_preferences()
    assert window.language == old_language and window.auto_compare_action.isChecked()

    def accept(dialog):
        dialog.language.setCurrentIndex(dialog.language.findData("en"))
        dialog.theme.setCurrentIndex(dialog.theme.findData("dark"))
        dialog.auto_compare.setChecked(False)
        assert not dialog.startup.isEnabled()
        dialog.startup.setChecked(False)
        dialog.restore_paths.setChecked(False)
        return QtWidgets.QDialog.Accepted

    monkeypatch.setattr(PreferencesDialog, "exec", accept)
    window.edit_preferences()
    assert window.language == "en" and window.theme_combo.currentData() == "dark"
    assert not window.auto_compare_action.isChecked()
    assert not window.startup_compare and not window.restore_paths


def test_legacy_splitter_widths_are_preserved(window, qtbot, settings):
    window.splitter.setSizes([280, 700, 320])
    expected = window.splitter.sizes()[::2]
    settings.setValue("geometry", window.saveGeometry())
    settings.setValue("splitter", window.splitter.saveState())
    settings.remove("layout")
    restored = Window(settings=settings)
    qtbot.addWidget(restored)
    # The offscreen screen is smaller than this window; restoreGeometry may clamp it.
    restored.resize(window.size())
    restored.show()
    qtbot.wait(50)
    assert restored.splitter.sizes()[::2] == pytest.approx(expected, abs=20)
    restored.close()


def test_failed_comparison_does_not_retry(qtbot, window, audio, monkeypatch):
    path = audio("a.wav", np.zeros(80))
    calls = []

    def fail(*args):
        calls.append(1)
        raise ValueError("Unreadable audio")

    monkeypatch.setattr(app_module, "compare", fail)
    window.set_paths((path, path))
    qtbot.waitUntil(lambda: bool(calls), timeout=3000)
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=3000)
    qtbot.wait(500)
    assert calls == [1] and window.result is None
    assert window.statusBar().currentMessage() == "Unreadable audio"
    assert not window.cancel_action.isEnabled()


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("size", [(1000, 680), (1440, 940)])
def test_compact_layout_fits_window(window, qtbot, language, size):
    window.language_combo.setCurrentIndex(window.language_combo.findData(language))
    window.resize(*size)
    qtbot.wait(50)
    assert window.width() == size[0] and window.height() == size[1]
    assert window.centralWidget().minimumSizeHint().width() <= size[0]
    assert window.begin.width() == 140
    assert window.end.width() == 140
    assert window.tabs.width() > 350 and window.tabs.height() > 400
    for edit in window.paths:
        assert edit.width() >= 80
    assert window.compare_button.defaultAction() is window.compare_action
    assert window.cancel_button.defaultAction() is window.cancel_action
