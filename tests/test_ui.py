import time

import numpy as np
import pytest
from PySide6 import QtCore

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui.app import Window
from wav_compare.ui.jobs import Jobs


def drag(qtbot, plot, start, end):
    viewport = plot.viewport()
    qtbot.mouseMove(viewport, start)
    qtbot.wait(30)
    qtbot.mousePress(viewport, QtCore.Qt.LeftButton, pos=start)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        qtbot.mouseMove(viewport, start + (end - start) * fraction)
        qtbot.wait(30)
    qtbot.mouseRelease(viewport, QtCore.Qt.LeftButton, pos=end)


@pytest.mark.parametrize("plot_name", ["wave_plot", "diff_plot", "segment_plot"])
def test_plot_drag_and_limits(qtbot, audio, tmp_path, plot_name):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    window.show()
    values = np.sin(np.arange(16000) * 0.05)
    result = compare(audio("a.wav", values), audio("b.wav", values * 0.5), Options(strict=True))
    window.set_result(result)
    qtbot.waitUntil(lambda: window.wave_curves[0].xData is not None, timeout=5000)
    plot = getattr(window, plot_name)
    box = plot.getViewBox()
    window.wave_plot.setXRange(0.5, 1.5, padding=0)
    plot.setYRange(0.1, 0.3, padding=0)
    qtbot.wait(100)
    before = np.array(box.viewRange())
    selection = window.region.getRegion()
    start = plot.mapFromScene(box.sceneBoundingRect().center())
    drag(qtbot, plot, start, start + QtCore.QPoint(25, 20))
    assert not any(indicator.fixed for indicator in window.indicators.values())
    after = np.array(box.viewRange())
    assert not np.allclose(before[0], after[0]), "Time must pan inside the selection"
    assert not np.allclose(before[1], after[1]), "Amplitude must pan inside the selection"
    assert window.region.getRegion() == selection
    qtbot.wait(150)
    np.testing.assert_allclose(box.viewRange()[1], after[1])
    # The amplitude axis itself must also support vertical-only dragging.
    axis = plot.getAxis("left")
    start = plot.mapFromScene(axis.mapToScene(axis.boundingRect().center()))
    drag(qtbot, plot, start, start + QtCore.QPoint(0, -10))
    assert not np.allclose(box.viewRange()[1], after[1])
    np.testing.assert_allclose(box.viewRange()[0], after[0])
    for direction in (-1, 1):
        box.translateBy(x=direction * 1e6, y=direction * 1e6)
        for visible, limits in zip(
            box.viewRange(),
            (box.state["limits"]["xLimits"], box.state["limits"]["yLimits"]),
            strict=True,
        ):
            assert visible[0] >= limits[0] - 1e-8
            assert visible[1] <= limits[1] + 1e-8
        for linked_plot in (window.wave_plot, window.diff_plot, window.segment_plot):
            lo, hi = linked_plot.viewRange()[0]
            assert lo >= -1e-8 and hi <= 2 + 1e-8
    if plot_name == "wave_plot":
        window.reset_zoom()
        window.region.setRegion((0.5, 1.5))
        qtbot.wait(100)
        start = plot.mapFromScene(box.mapViewToScene(QtCore.QPointF(0.5, 0)))
        end = plot.mapFromScene(box.mapViewToScene(QtCore.QPointF(0.7, 0)))
        drag(qtbot, plot, start, end)
        assert not window.indicators["wave"].fixed
        assert window.region.getRegion()[0] > 0.6
        assert window.begin.value() == pytest.approx(window.region.getRegion()[0], abs=1e-6)
    window.jobs.cancel_all()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    window.close()


def test_amplitude_limits_follow_channel_and_reset_silence(qtbot, audio, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    values = np.column_stack((np.linspace(-0.1, 0.1, 8000), np.linspace(-4, 4, 8000)))
    path = audio("float.wav", values)
    window.set_result(compare(path, path, Options(strict=True)))
    assert window.wave_plot.getViewBox().state["limits"]["yLimits"] == pytest.approx([-0.15, 0.15])
    window.channel.setCurrentIndex(1)
    assert window.wave_plot.getViewBox().state["limits"]["yLimits"] == pytest.approx([-6, 6])
    silent = audio("silent.wav", np.zeros(80))
    window.set_result(compare(silent, silent, Options(strict=True)))
    window.reset_zoom()
    for plot in (window.wave_plot, window.diff_plot, window.segment_plot):
        x, y = plot.viewRange()
        assert np.isfinite(y).all() and y[0] < 0 < y[1]
        assert x[0] >= -1e-8 and x[1] <= 0.01 + 1e-8
    window.jobs.cancel_all()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    window.close()


@pytest.mark.parametrize("rate", [8000, 48000])
@pytest.mark.parametrize("plot_name, tab", [("spectrum_plot", 1), ("spectrogram_plot", 2)])
def test_spectral_drag_limits_and_reset(qtbot, audio, tmp_path, rate, plot_name, tab):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    window.show()
    path = audio("a.wav", np.sin(np.arange(rate) * 0.1), rate=rate)
    window.set_result(compare(path, path, Options(strict=True)))
    window.region.setRegion((0.25, 0.75))
    window.region_changed()
    window.tabs.setCurrentIndex(tab)
    qtbot.waitUntil(lambda: window.spectral_ranges is not None, timeout=5000)
    plot = getattr(window, plot_name)
    box = plot.getViewBox()
    limits = box.state["limits"]
    if tab == 1:
        assert limits["xLimits"] == [0, rate / 2]
        assert limits["yLimits"][0] < -200 < limits["yLimits"][1]
    else:
        assert limits["xLimits"] == [0.25, 0.75]
        assert limits["yLimits"] == [0, rate / 2]
    initial = np.array(box.viewRange())
    middle = initial.mean(axis=1)
    half_width = np.diff(initial, axis=1).ravel() / 4
    box.setRange(
        xRange=(middle[0] - half_width[0], middle[0] + half_width[0]),
        yRange=(middle[1] - half_width[1], middle[1] + half_width[1]),
        padding=0,
    )
    qtbot.wait(100)
    before = np.array(box.viewRange())
    start = plot.mapFromScene(box.sceneBoundingRect().center())
    drag(qtbot, plot, start, start + QtCore.QPoint(20, 20))
    assert not any(indicator.fixed for indicator in window.indicators.values())
    assert not np.allclose(before[0], box.viewRange()[0])
    assert not np.allclose(before[1], box.viewRange()[1])
    for direction in (-1, 1):
        box.translateBy(x=direction * 1e6, y=direction * 1e6)
        for axis, key in enumerate(("xLimits", "yLimits")):
            assert box.viewRange()[axis][0] >= limits[key][0] - 1e-8
            assert box.viewRange()[axis][1] <= limits[key][1] + 1e-8
    box.setRange(xRange=(-1e9, 1e9), yRange=(-1e9, 1e9), padding=0)
    np.testing.assert_allclose(box.viewRange(), [limits["xLimits"], limits["yLimits"]])
    window.reset_zoom()
    np.testing.assert_allclose(box.viewRange(), initial)
    window.jobs.cancel_all()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    window.close()


def test_spectral_bounds_follow_completed_selection_and_sample_rate(qtbot, audio, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    for rate, frames in ((48000, 4800), (8000, 80)):
        path = audio(f"silent-{rate}.wav", np.zeros(frames), rate=rate)
        window.set_result(compare(path, path, Options(strict=True)))
        window.tabs.setCurrentIndex(2)
        # Editing a selection does not recompute until Update spectrum is clicked.
        # A pending result must keep the bounds of the selection it actually analyzed.
        window.region.setRegion((0, frames / rate / 2))
        window.region_changed()
        qtbot.waitUntil(lambda: window.spectral_ranges is not None, timeout=5000)
        assert window.spectrogram_plot.getViewBox().state["limits"]["xLimits"] == [0, frames / rate]
        assert window.spectrum_plot.getViewBox().state["limits"]["xLimits"] == [0, rate / 2]
        assert window.spectrogram_plot.getViewBox().state["limits"]["yLimits"] == [0, rate / 2]
        low, high = window.spectrum_plot.viewRange()[1]
        assert np.isfinite([low, high]).all() and low < -200 < high
        if frames == 80:
            rectangle = window.image.mapRectToParent(window.image.boundingRect())
            assert rectangle.left() == pytest.approx(0)
            assert rectangle.right() == pytest.approx(frames / rate)
        qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    window.close()


def test_window_zero_diff_theme_locale_and_invalidation(qtbot, audio, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    window.show()
    path = audio("long directory/声音测试.wav", np.zeros((8000, 2)))
    result = compare(path, path, Options(strict=True))
    window.set_result(result)
    window.reset_zoom()
    window.language_combo.setCurrentIndex(window.language_combo.findData("zh"))
    assert window.compare_button.text() == "开始比较"
    window.theme_combo.setCurrentIndex(window.theme_combo.findData("dark"))
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=10000)
    window.threshold.setValue(0.001)
    assert window.result is None
    assert not window.rows
    window.close()


def test_stale_jobs_never_replace_latest(qtbot):
    jobs = Jobs()
    delivered = []
    jobs.done.connect(lambda kind, result: delivered.append(result[1]))

    def slow(cancel, progress):
        time.sleep(0.1)
        return "old"

    jobs.submit("detail", slow)
    jobs.submit("detail", lambda *_: "new")
    qtbot.waitUntil(lambda: not jobs.jobs, timeout=5000)
    assert delivered == ["new"]


def test_close_cancels_workers(qtbot, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    window.show()

    def work(cancel, progress):
        while True:
            cancel.check()
            time.sleep(0.01)

    window.jobs.submit("detail", work)
    window.close()
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)


def test_missing_audio_device(qtbot, audio, monkeypatch):
    from PySide6 import QtMultimedia

    from wav_compare.ui.player import Player

    path = audio("a.wav", np.zeros(1000))
    result = compare(path, path, Options(strict=True))
    monkeypatch.setattr(
        QtMultimedia.QMediaDevices, "defaultAudioOutput", lambda: QtMultimedia.QAudioDevice()
    )
    player = Player()
    errors = []
    player.error.connect(errors.append)
    player.play(result, 0, 0, 0, 0.1)
    assert errors == ["No audio output device"]


def test_difference_in_current_segment(qtbot, audio, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    qtbot.addWidget(window)
    a, b = np.zeros(5000), np.zeros(5000)
    b[50], b[100] = 0.5, -0.5
    result = compare(audio("a.wav", a), audio("b.wav", b), Options(strict=True))
    window.set_result(result)
    window.next_difference(1)
    qtbot.waitUntil(lambda: abs(window.cursors[0].value() - 50 / 8000) < 1e-8, timeout=5000)
    window.next_difference(1)
    qtbot.waitUntil(lambda: abs(window.cursors[0].value() - 100 / 8000) < 1e-8, timeout=5000)
    window.next_difference(-1)
    qtbot.waitUntil(lambda: abs(window.cursors[0].value() - 50 / 8000) < 1e-8, timeout=5000)
    window.jobs.cancel_all()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    window.close()
