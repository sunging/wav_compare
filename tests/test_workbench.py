"""Results/metrics panels, parameter dependencies, streaming batches and bounded navigation."""

import json
import threading

import numpy as np
import pytest
from PySide6 import QtCore, QtGui, QtWidgets

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui import app as app_module
from wav_compare.ui.app import Window
from wav_compare.ui.jobs import Jobs
from wav_compare.ui.parameters import ParametersPanel
from wav_compare.ui.results import MetricsView, ResultsTable
from wav_compare.views import difference_overview


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "workbench.ini"), QtCore.QSettings.IniFormat)
    widget = Window(settings=settings)
    widget.show()
    yield widget
    widget.close()
    qtbot.waitUntil(lambda: not widget.jobs.jobs, timeout=10000)
    widget.deleteLater()


def metric(max_abs, snr=None, status="finite"):
    return {"max_abs": max_abs, "rmse": max_abs / 2, "snr_db": snr, "snr_status": status}


def test_results_table_sorts_numbers_and_keeps_missing_values_last(qtbot):
    table = ResultsTable(str)
    qtbot.addWidget(table)
    rows = [
        {"name": "a", "status": "different", "metrics": [metric(0.5, 3.0)], "alignment": {}},
        {"name": "b", "status": "missing"},
        {"name": "c", "status": "different", "metrics": [metric(1e-05, 90.0)], "alignment": {}},
        {"name": "d", "status": "within_threshold", "metrics": [metric(0, None, "infinite")]},
    ]
    table.set_rows(rows)
    table.table.sortItems(2, QtCore.Qt.AscendingOrder)
    names = [table.table.item(i, 0).text() for i in range(4)]
    assert names == ["d", "c", "a", "b"]
    assert table.table.item(0, 5).text() == "∞"
    table.table.sortItems(5, QtCore.Qt.AscendingOrder)
    assert [table.table.item(i, 0).text() for i in range(4)] == ["a", "c", "d", "b"]
    assert table.table.item(3, 1).foreground().color() == QtGui.QColor("#e5534b")
    table.filter.setText("c")
    assert [table.table.isRowHidden(i) for i in range(4)] == [True, False, True, True]


def test_metrics_view_groups_undefined_values_and_copies(qtbot, audio):
    view = MetricsView(str)
    qtbot.addWidget(view)
    path = audio("zero.wav", np.zeros(100))
    report = compare(path, path, Options(strict=True)).report
    view.show_report(report, 0)
    groups = [view.topLevelItem(i).text(0) for i in range(view.topLevelItemCount())]
    assert groups == ["Summary", "Channel metrics", "Excluded"]
    channel = {
        view.topLevelItem(1).child(j).text(0): view.topLevelItem(1).child(j).text(1)
        for j in range(view.topLevelItem(1).childCount())
    }
    assert channel["Correlation"] == "—" and channel["SNR status"] == "undefined"
    view.copy()
    text = QtWidgets.QApplication.clipboard().text()
    assert "[Summary]" in text and "Max absolute difference\t0" in text


def test_parameters_disable_ignored_controls_and_flag_invalid_pairs(qtbot):
    panel = ParametersPanel()
    qtbot.addWidget(panel)
    assert panel.align.isEnabled() and panel.max_lag.isEnabled()
    assert not panel.offset.isEnabled()
    panel.manual.setChecked(True)
    assert panel.offset.isEnabled() and not panel.max_lag.isEnabled()
    panel.strict.setChecked(True)
    assert not any(c.isEnabled() for c in (panel.align, panel.max_lag, panel.manual, panel.offset))
    panel.pairs.setText("1:")
    assert panel.pairs.property("invalid") is True
    assert "1-based" in panel.pairs.toolTip()
    panel.pairs.setText("1:2")
    assert panel.pairs.property("invalid") is False
    panel.mix.setChecked(True)
    assert panel.pairs.property("invalid") is True
    with pytest.raises(ValueError):
        panel.options()


def test_streaming_jobs_only_deliver_latest_partials(qtbot):
    jobs = Jobs()
    partials, release = [], threading.Event()
    jobs.partial.connect(lambda kind, value: partials.append(value))

    def old(cancel, progress, emit):
        release.wait(5)
        emit("old")

    jobs.submit("batch", old, streaming=True)
    jobs.submit("batch", lambda cancel, progress, emit: emit("new"), streaming=True)
    release.set()
    qtbot.waitUntil(lambda: not jobs.jobs, timeout=5000)
    assert partials == ["new"]


def test_batch_rows_stream_before_completion_and_export_batch(
    qtbot, window, audio, tmp_path, monkeypatch
):
    for folder in ("left", "right"):
        audio(f"{folder}/one.wav", np.zeros(80))
    entered, release = threading.Event(), threading.Event()
    original = app_module.run_batch

    def batch(rows, options, cancel, progress, on_result):
        def streamed(row):
            on_result(row)
            entered.set()
            release.wait(5)

        return original(rows, options, cancel, progress, streamed)

    monkeypatch.setattr(app_module, "run_batch", batch)
    window.set_paths((tmp_path / "left", tmp_path / "right"))
    try:
        qtbot.waitUntil(entered.is_set, timeout=5000)
        qtbot.waitUntil(lambda: len(window.rows) == 1, timeout=5000)
        assert "batch" in window.jobs.latest and window.progress_bar.isVisible()
    finally:
        release.set()
    qtbot.waitUntil(lambda: window.result is not None and not window.jobs.jobs, timeout=10000)
    assert len(window.rows) == 1 and window.table.selectedItems()
    assert not window.progress_bar.isVisible()
    target = tmp_path / "batch.json"
    monkeypatch.setattr(QtWidgets.QFileDialog, "getSaveFileName", lambda *args: (str(target), ""))
    window.export("json")
    exported = json.loads(target.read_text(encoding="utf-8"))["results"]
    # A one-row folder batch still exports the batch row, including its pairing details.
    assert exported[0]["name"] == "one.wav" and exported[0]["a_candidates"]


def test_navigation_reads_segment_statistics_in_blocks(qtbot, window, audio, monkeypatch):
    a, b = np.zeros(200000), np.zeros(200000)
    b[199990] = 0.5
    result = compare(audio("a.wav", a), audio("b.wav", b), Options(strict=True, segment_size=1))
    window.set_result(result)
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    loads = []
    original = np.load

    def counted(*args, **kwargs):
        loads.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted)
    window.next_difference(1)
    qtbot.waitUntil(lambda: window.navigation_sample == 199990, timeout=10000)
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    assert sum(str(path).endswith("segments.npy") for path in loads) <= 2


def test_spectral_tab_and_source_switches_reuse_analysis(qtbot, window, audio, monkeypatch):
    path = audio("tone.wav", np.sin(np.arange(8000) * 0.1))
    window.set_result(compare(path, path, Options(strict=True)))
    window.tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: window.spectral_data is not None, timeout=5000)
    calls, original = [], app_module.spectra

    def tracked(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(app_module, "spectra", tracked)
    window.tabs.setCurrentIndex(2)
    window.tabs.setCurrentIndex(1)
    window.source.setCurrentIndex(2)
    qtbot.wait(100)
    assert not calls and window.spectral_data is not None
    window.selection_preview(0.1, 0.2)
    window.update_spectrum()
    qtbot.waitUntil(lambda: bool(calls) and not window.jobs.jobs, timeout=5000)


def test_drop_on_one_field_sets_only_that_side(qtbot, window, audio, tmp_path):
    a, b, c = (audio(name, np.zeros(80)) for name in ("a.wav", "b.wav", "c.wav"))
    window.auto_compare_action.setChecked(False)
    window.set_paths((a, b))

    def drop(target, *paths):
        mime = QtCore.QMimeData()
        mime.setUrls([QtCore.QUrl.fromLocalFile(str(path)) for path in paths])
        event = QtGui.QDropEvent(
            QtCore.QPointF(), QtCore.Qt.CopyAction, mime, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier
        )
        if target is None:
            window.dropEvent(event)
        else:
            window.eventFilter(target, event)
        assert event.isAccepted()

    drop(window.paths[0], c)
    assert [edit.text() for edit in window.paths] == [str(c), str(b)]
    window.set_paths(("", ""))
    drop(None, a)
    assert [edit.text() for edit in window.paths] == [str(a), ""]
    drop(None, b)
    assert [edit.text() for edit in window.paths] == [str(a), str(b)]


def test_difference_overview_and_timeline_strip(qtbot, window, audio):
    a, b = np.zeros(8000), np.zeros(8000)
    b[4000] = 0.25
    window.set_result(compare(audio("a.wav", a), audio("b.wav", b), Options(strict=True)))
    width, values = difference_overview(window.result, 0)
    assert values.max() == 1 and np.count_nonzero(values) == 1
    assert int(np.argmax(values)) == 4000 // width
    assert window.seek.overview is not None
    window.channel.setCurrentIndex(0)
    same = compare(audio("c.wav", a), audio("d.wav", a), Options(strict=True))
    assert not difference_overview(same, 0)[1].any()
