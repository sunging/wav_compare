import threading

import numpy as np
import pytest
from PySide6 import QtCore, QtWidgets

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui.app import Window


@pytest.fixture
def window(qtbot, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "indicators.ini"), QtCore.QSettings.IniFormat)
    view = Window(settings=settings)
    view.language_combo.setCurrentIndex(view.language_combo.findData("en"))
    view.show()
    yield view
    view.jobs.cancel_all()
    qtbot.waitUntil(lambda: not view.jobs.jobs, timeout=10000)
    view.close()


def load(window, qtbot, audio, *, frames=2050):
    a = np.column_stack((np.arange(frames) / frames, np.full(frames, 0.75)))
    b = a * 0.5
    window.set_result(
        compare(audio("a.wav", a), audio("b.wav", b), Options(strict=True, segment_size=1024))
    )
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    return a, b


def query(window, qtbot, name, x, y=0):
    probe = window.indicators[name]
    scene = probe.plot.getViewBox().mapViewToScene(QtCore.QPointF(x, y))
    probe.query(scene)
    qtbot.waitUntil(lambda: bool(probe.fields), timeout=5000)
    return dict(probe.fields)


@pytest.mark.parametrize("name", ["wave", "diff"])
def test_samples_are_exact_even_for_envelope_views(window, qtbot, audio, name):
    a, b = load(window, qtbot, audio, frames=16000)
    fields = query(window, qtbot, name, 4321.25 / 8000)
    assert fields["Sample index (0-based)"] == "4321"
    assert fields["Time"] == "0.540125 s"
    assert float(fields["B − A"]) == pytest.approx(b[4321, 0] - a[4321, 0])
    if name == "wave":
        assert float(fields["A"]) == pytest.approx(a[4321, 0])
        assert float(fields["B"]) == pytest.approx(b[4321, 0])
    else:
        assert float(fields["Absolute difference"]) == pytest.approx(abs(b[4321, 0] - a[4321, 0]))
    window.channel.setCurrentIndex(1)
    assert not window.indicators[name].fields
    fields = query(window, qtbot, name, 4321.25 / 8000)
    assert fields["Display channel"] == "2 → 2"
    assert float(fields["B − A"]) == -0.375


@pytest.mark.parametrize("sample", [1023, 1024, 2049])
def test_original_segments_and_short_tail(window, qtbot, audio, sample):
    a, b = load(window, qtbot, audio)
    fields = query(window, qtbot, "segment", (sample + 0.25) / 8000, 0.2)
    start = sample // 1024 * 1024
    stop = min(len(a), start + 1024)
    differences = np.abs(b[start:stop, 0] - a[start:stop, 0])
    assert fields["Segment index (0-based)"] == str(sample // 1024)
    assert fields["Time range"] == f"[{start / 8000:.6f}, {stop / 8000:.6f}) s"
    assert int(fields["Sample count"]) == stop - start
    assert float(fields["Max absolute difference"]) == pytest.approx(differences.max())
    assert float(fields["MAE"]) == pytest.approx(differences.mean())


def spectral_fixture(window, qtbot, columns=3):
    window.tabs.setCurrentIndex(2)
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    frequencies = np.array([0, 1000, 2000, 3000, 4000])
    psd = np.arange(15).reshape(3, 5) - 160.0
    times = np.array([0.125]) if columns == 1 else np.array([0.04, 0.12, 0.21])
    images = np.arange(3 * 5 * columns).reshape(3, 5, columns) - 200.0
    window.draw_spectrum(((0, 0.25), (frequencies, psd, times, images)))
    return frequencies, psd, times, images


def test_frequency_nearest_bin_and_endpoints(window, qtbot, audio):
    load(window, qtbot, audio)
    window.tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: window.spectral_data is not None, timeout=5000)
    frequencies, psd, _, _ = window.spectral_data
    for index in (0, 2, len(frequencies) - 1):
        fields = query(window, qtbot, "spectrum", frequencies[index], -100)
        assert float(fields["Frequency"].split()[0]) == pytest.approx(frequencies[index])
        for k, name in enumerate(("A", "B", "B − A")):
            assert float(fields[name].split()[0]) == pytest.approx(psd[k, index])
    x = frequencies[2] + 0.2 * (frequencies[3] - frequencies[2])
    fields = query(window, qtbot, "spectrum", x, -100)
    assert float(fields["Frequency"].split()[0]) == pytest.approx(frequencies[2])


@pytest.mark.parametrize("columns", [1, 3])
@pytest.mark.parametrize("source", [0, 1, 2])
def test_spectrogram_uses_displayed_cell_and_unclipped_value(window, qtbot, audio, columns, source):
    load(window, qtbot, audio)
    window.source.setCurrentIndex(source)
    frequencies, _, times, images = spectral_fixture(window, qtbot, columns)
    column, row = columns - 1, 4
    point = window.image.mapToParent(QtCore.QPointF(column + 0.5, row + 0.5))
    fields = query(window, qtbot, "spectrogram", point.x(), point.y())
    assert fields["Frame time"] == f"{times[column]:.6f} s"
    assert fields["Frequency"] == "4000 Hz"
    assert fields["Amplitude"] == f"{images[source, row, column]:.9g} dB"
    assert fields["Source"] == ("A", "B", "B − A")[source]
    probe = window.indicators["spectrogram"]
    assert probe.lines[0].value() == pytest.approx(point.x())
    assert probe.lines[1].value() == pytest.approx(point.y())
    # Editing the selection must not relabel the already rendered STFT.
    window.region.setRegion((0.05, 0.1))
    assert query(window, qtbot, "spectrogram", point.x(), point.y()) == fields
    window.source.setCurrentIndex((source + 1) % 3)
    assert not probe.fields
    assert window.spectral_data is None


@pytest.mark.parametrize(
    "name,tab", [("wave", 0), ("diff", 0), ("segment", 0), ("spectrum", 1), ("spectrogram", 2)]
)
def test_real_click_pin_leave_unpin_and_independence(window, qtbot, audio, name, tab):
    load(window, qtbot, audio)
    window.tabs.setCurrentIndex(tab)
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    probe = window.indicators[name]
    plot = probe.plot
    center = plot.mapFromScene(plot.getViewBox().sceneBoundingRect().center())
    qtbot.mouseMove(plot.viewport(), center)
    qtbot.waitUntil(lambda: bool(probe.fields), timeout=5000)
    assert not probe.fixed
    qtbot.mouseClick(plot.viewport(), QtCore.Qt.LeftButton, pos=center)
    qtbot.waitUntil(lambda: probe.fixed and bool(probe.fields), timeout=5000)
    text, position = probe.readout.text(), probe.lines[0].value()
    assert "Pinned" in text
    assert not any(other.fixed for key, other in window.indicators.items() if key != name)
    if tab == 0:
        qtbot.wait(100)
        assert window.tabs.widget(0).verticalScrollBar().maximum() == 0
        assert not probe.panel.hasHeightForWidth()
    qtbot.mouseMove(plot.viewport(), center + QtCore.QPoint(25, 0))
    window.play_position(0.2)
    window.zoom_to(1000)
    QtWidgets.QApplication.sendEvent(plot.viewport(), QtCore.QEvent(QtCore.QEvent.Leave))
    qtbot.wait(100)
    assert probe.readout.text() == text
    assert probe.lines[0].value() == position
    # Language/theme changes reformat the same pinned values.
    window.language_combo.setCurrentIndex(window.language_combo.findData("zh"))
    window.theme_combo.setCurrentIndex(window.theme_combo.findData("dark"))
    assert "已固定" in probe.readout.text()
    center = plot.mapFromScene(plot.getViewBox().sceneBoundingRect().center())
    qtbot.mouseClick(plot.viewport(), QtCore.Qt.LeftButton, pos=center)
    assert not probe.fixed
    QtWidgets.QApplication.sendEvent(plot.viewport(), QtCore.QEvent(QtCore.QEvent.Leave))
    assert not probe.readout.text()
    assert not probe.lines[0].isVisible()


def test_disabled_persistence_and_default_enabled(window, qtbot, audio):
    load(window, qtbot, audio)
    assert all(p.toggle.isChecked() for p in window.indicators.values())
    query(window, qtbot, "wave", 0.1)
    window.indicators["wave"].toggle.setChecked(False)
    assert not window.indicators["wave"].readout.isVisible()
    assert not window.indicators["wave"].lines[0].isVisible()
    query(window, qtbot, "diff", 0.15)
    window.close()
    restored = Window(settings=window.settings)
    qtbot.addWidget(restored)
    assert not restored.indicators["wave"].toggle.isChecked()
    assert all(
        restored.indicators[name].toggle.isChecked()
        for name in ("diff", "segment", "spectrum", "spectrogram")
    )
    assert not any(p.fixed for p in restored.indicators.values())
    restored.close()


@pytest.mark.parametrize("change", ["toggle", "channel", "result", "reset", "invalidate"])
def test_late_sample_read_cannot_restore_stale_values(window, qtbot, audio, monkeypatch, change):
    load(window, qtbot, audio)
    probe = window.indicators["wave"]
    result = window.result
    started, release = threading.Event(), threading.Event()
    original = result.samples

    def slow(*args):
        started.set()
        release.wait(5)
        return original(*args)

    monkeypatch.setattr(result, "samples", slow)
    try:
        probe.query(probe.plot.getViewBox().mapViewToScene(QtCore.QPointF(0.1, 0)))
        qtbot.waitUntil(started.is_set, timeout=5000)
        if change == "toggle":
            probe.toggle.setChecked(False)
            probe.toggle.setChecked(True)
        elif change == "channel":
            window.channel.setCurrentIndex(1)
        elif change == "result":
            window.set_result(result)
        elif change == "reset":
            window.reset_zoom()
        else:
            window.invalidate()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    assert not probe.fields
    assert not probe.lines[0].isVisible()


def test_empty_and_outside_data_do_not_pin(window, qtbot, audio):
    probe = window.indicators["wave"]
    center = probe.plot.mapFromScene(probe.plot.getViewBox().sceneBoundingRect().center())
    qtbot.mouseClick(probe.plot.viewport(), QtCore.Qt.LeftButton, pos=center)
    assert not probe.fixed
    load(window, qtbot, audio)
    query(window, qtbot, "wave", 0.1)
    probe.query(probe.plot.getViewBox().mapViewToScene(QtCore.QPointF(-1, 0)))
    probe.track_inside(QtCore.QPointF(-1, -1))
    assert not probe.fields


@pytest.mark.parametrize("language,theme", [("en", "light"), ("zh", "dark")])
def test_minimum_window_keeps_plots_and_wrapped_readings_usable(
    window, qtbot, audio, language, theme
):
    load(window, qtbot, audio)
    window.language_combo.setCurrentIndex(window.language_combo.findData(language))
    window.theme_combo.setCurrentIndex(window.theme_combo.findData(theme))
    window.resize(1000, 680)
    for name in ("wave", "diff", "segment"):
        y = np.mean(window.indicators[name].plot.viewRange()[1])
        query(window, qtbot, name, 0.12, y)
        window.indicators[name].fixed = True
        window.indicators[name].render()
    qtbot.wait(100)
    for name in ("wave", "diff", "segment"):
        probe = window.indicators[name]
        assert probe.plot.getViewBox().height() >= 40
        assert probe.readout.wordWrap()
        assert probe.readout.height() >= probe.readout.heightForWidth(probe.readout.width())
        assert probe.readout.textInteractionFlags() & QtCore.Qt.TextSelectableByMouse
    assert window.width() == 1000 and window.height() == 680
    assert window.tabs.widget(0).verticalScrollBar().maximum() > 0
    window.resize(1440, 940)
    qtbot.wait(100)
    assert window.tabs.widget(0).verticalScrollBar().maximum() < 100
    assert all(
        window.indicators[name].readout.height() < 100 for name in ("wave", "diff", "segment")
    )
