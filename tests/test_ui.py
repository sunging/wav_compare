import time

import numpy as np
from PySide6 import QtCore

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui.app import Window
from wav_compare.ui.jobs import Jobs


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
