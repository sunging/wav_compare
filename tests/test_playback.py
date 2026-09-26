"""Timeline gestures and transport state without requiring a physical audio device."""

import queue
import threading

import numpy as np
import pytest
from PySide6 import QtCore, QtMultimedia, QtWidgets

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui.app import Window
from wav_compare.ui.player import Player
from wav_compare.ui.timeline import PlaybackTimeline


@pytest.fixture
def fake_audio(monkeypatch):
    fmt = QtMultimedia.QAudioFormat()
    fmt.setSampleRate(8000)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QtMultimedia.QAudioFormat.Float)

    class Device:
        def isNull(self):
            return False

        def preferredFormat(self):
            return fmt

    class Sink:
        instances = []

        def __init__(self, *_):
            self.instances.append(self)
            self.elapsed = 0
            self.audio_state = QtMultimedia.QAudio.ActiveState
            self.writes = []
            self.suspended = False
            self.stopped = False

        def setVolume(self, value):
            self.volume = value

        def setBufferSize(self, size):
            pass

        def start(self):
            return self

        def write(self, value):
            assert not self.suspended
            self.writes.append(value)
            return len(value)

        def bytesFree(self):
            return 100000

        def processedUSecs(self):
            return self.elapsed

        def state(self):
            return self.audio_state

        def suspend(self):
            self.suspended = True

        def resume(self):
            self.suspended = False

        def reset(self):
            self.stopped = True

        def deleteLater(self):
            pass

    monkeypatch.setattr(QtMultimedia.QMediaDevices, "defaultAudioOutput", Device)
    monkeypatch.setattr(QtMultimedia, "QAudioSink", Sink)
    return Sink, fmt


@pytest.fixture
def window(qtbot, audio, tmp_path, fake_audio):
    settings = QtCore.QSettings(str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat)
    widget = Window(settings=settings)
    path = audio("tone.wav", np.sin(np.arange(16000) * 0.1) * 0.1)
    widget.set_result(compare(path, path, Options(strict=True)))
    widget.show()
    qtbot.wait(30)
    yield widget
    widget.player.stop()
    if widget.player.producer:
        widget.player.producer.join(timeout=2)
    widget.jobs.cancel_all()
    qtbot.waitUntil(lambda: not widget.jobs.jobs, timeout=5000)
    widget.close()


def point(bar, time):
    return QtCore.QPoint(round(bar.x_at(time)), bar.height() // 2)


def drag(qtbot, bar, start, end):
    qtbot.mousePress(bar, QtCore.Qt.LeftButton, pos=start)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        qtbot.mouseMove(bar, start + (end - start) * fraction)
    qtbot.mouseRelease(bar, QtCore.Qt.LeftButton, pos=end)


@pytest.mark.parametrize("start,end", [(0.4, 1.5), (1.5, 0.4)])
def test_drag_selection_sync_and_no_analysis(window, qtbot, monkeypatch, start, end):
    bar = window.seek
    calls = []
    monkeypatch.setattr(window.jobs, "submit", lambda *args: calls.append(args))
    drag(qtbot, bar, point(bar, start), point(bar, end))
    assert bar.selection == pytest.approx((0.4, 1.5), abs=0.01)
    assert window.region.getRegion() == bar.selection
    assert (window.begin.value(), window.end.value()) == pytest.approx(bar.selection, abs=1e-6)
    assert window.player.state == "stopped"
    assert calls == []
    window.region.setRegion((0.6, 1.7))
    assert bar.selection == pytest.approx((0.6, 1.7))
    window.begin.setValue(0.8)
    assert bar.selection == pytest.approx((0.8, 1.7))
    window.end.setValue(0.1)
    assert bar.selection[1] - bar.selection[0] == pytest.approx(1 / 8000)


def test_edges_cannot_cross_and_outside_drag_clamps(window, qtbot):
    bar = window.seek
    window.selection_preview(0.5, 1.5)
    drag(qtbot, bar, point(bar, 0.5), point(bar, 1.8))
    assert bar.selection == pytest.approx((1.5 - 1 / 8000, 1.5))
    window.selection_preview(0.5, 1.5)
    drag(qtbot, bar, point(bar, 1.5), point(bar, 0.1))
    assert bar.selection == pytest.approx((0.5, 0.5 + 1 / 8000))
    window.selection_preview(0.5, 1.5)
    drag(qtbot, bar, point(bar, 1), QtCore.QPoint(bar.width() + 100, 16))
    assert bar.selection == pytest.approx((1, 2), abs=0.01)


def test_select_all_context_menu(window, qtbot):
    window.selection_preview(0.5, 1.5)

    def activate():
        menu = QtWidgets.QApplication.activePopupWidget()
        assert isinstance(menu, QtWidgets.QMenu)
        assert menu.actions()[0].text() == window.tr("Select all")
        menu.actions()[0].trigger()
        menu.close()

    from PySide6 import QtGui

    pos = point(window.seek, 1)
    QtCore.QTimer.singleShot(10, activate)
    event = QtGui.QContextMenuEvent(
        QtGui.QContextMenuEvent.Mouse, pos, window.seek.mapToGlobal(pos)
    )
    QtWidgets.QApplication.sendEvent(window.seek, event)
    assert window.region.getRegion() == (0, 2)


@pytest.mark.parametrize("state", ["stopped", "playing", "paused"])
def test_click_preserves_state_and_cancels_old_stream(window, qtbot, state):
    if state != "stopped":
        window.play()
        window.player.timer.stop()
        if state == "paused":
            window.player.pause()
    old_cancel, old_sink = window.player.cancel, window.player.sink
    qtbot.mouseClick(window.seek, QtCore.Qt.LeftButton, pos=point(window.seek, 1.6))
    assert window.playback_position == pytest.approx(1.6, abs=0.01)
    assert window.player.state == state
    assert window.cursors[0].value() == window.playback_position
    if state != "stopped":
        assert old_cancel.is_set() and old_sink.stopped
        assert window.player.origin == window.playback_position
        if state == "paused":
            window.player._pump()
            assert window.player.sink.suspended
            assert window.player.sink.writes == []
    else:
        assert window.player.sink is None
        window.play()
        assert window.player.origin == pytest.approx(1.6, abs=0.01)
    window.player.stop()
    position = window.playback_position
    window.play()
    assert window.player.origin == position


def test_loop_bounds_and_live_selection_changes(window, qtbot):
    window.selection_preview(0.5, 1.5)
    window.seek_play(1.8)
    window.play()
    assert window.player.end == 2
    window.loop.setChecked(True)
    assert window.player.origin == 0.5
    assert window.player.end == 1.5
    window.seek_play(1)
    assert window.player.loop_begin == 0.5
    window.player.pause()
    window.selection_preview(0.1, 0.4)
    window.selection_committed()
    assert window.player.state == "paused"
    assert window.player.origin == 0.1
    window.loop.setChecked(False)
    assert window.player.state == "paused"
    assert window.player.end == 2
    qtbot.keyClick(window.seek, QtCore.Qt.Key_End)
    assert window.playback_position == 2
    assert window.player.state == "paused"
    window.play()
    assert window.player.origin == 0


def test_keyboard_and_position_callback_during_drag(window, qtbot):
    bar = window.seek
    window.seek_play(0.5)
    qtbot.keyClick(bar, QtCore.Qt.Key_Right)
    assert window.playback_position == pytest.approx(0.6)
    qtbot.keyClick(bar, QtCore.Qt.Key_Left)
    assert window.playback_position == pytest.approx(0.5)
    qtbot.mousePress(bar, QtCore.Qt.LeftButton, pos=point(bar, 0.7))
    window.play_position(1.2)
    assert bar.position == 0.5
    qtbot.mouseMove(bar, point(bar, 1.6))
    qtbot.mouseRelease(bar, QtCore.Qt.LeftButton, pos=point(bar, 1.6))
    assert bar.position == 1.2
    window.loop.setChecked(True)
    qtbot.keyClick(bar, QtCore.Qt.Key_Home)
    assert window.playback_position == bar.selection[0]
    qtbot.keyClick(bar, QtCore.Qt.Key_End)
    assert window.playback_position == bar.selection[1]


def test_reset_result_and_tiny_audio(window, qtbot, audio):
    window.seek_play(1)
    window.play()
    path = audio("short.wav", [0.1])
    window.set_result(compare(path, path, Options(strict=True)))
    assert window.player.state == "stopped"
    assert window.playback_position == 0
    assert window.seek.selection == (0, 1 / 8000)
    window.selection_preview(0, 0)
    assert window.seek.selection == (0, 1 / 8000)
    window.invalidate()
    assert not window.seek.isEnabled()
    assert window.seek.position == 0


def test_original_input_shorter_than_aligned_timeline(window, audio):
    a = audio("long.wav", np.zeros(16000))
    b = audio("shorter.wav", np.zeros(8000))
    window.set_result(compare(a, b, Options(strict=True)))
    # Exercise the defensive source boundary independently of alignment geometry.
    window.result.report["b"]["frames"] = 2000
    window.source.setCurrentIndex(1)
    window.original.setChecked(True)
    window.seek_play(0.8)
    assert window.playback_position == 0.25
    window.play()
    assert window.player.end == 0.25
    window.player.stop()
    window.selection_preview(0.5, 0.9)
    window.loop.setChecked(True)
    window.play()
    assert window.player.state == "stopped"
    assert window.statusBar().currentMessage() == window.tr("No playable audio in selection")


def test_producer_repeats_whole_selection_after_partial_first_pass(fake_audio):
    _, fmt = fake_audio
    cancel = threading.Event()
    output = queue.Queue()

    class Result:
        rate = 8000
        options = Options(strict=True)
        calls = []

        def samples(self, start, count):
            self.calls.append((start, count))
            if len(self.calls) == 3:
                cancel.set()
            values = np.zeros((count, 1))
            return values, values, values

    result = Result()
    player = Player()
    player._produce(result, 0, 0, 0.15, 0.2, True, False, fmt, output, cancel, 0.1)
    assert result.calls == [(1200, 400), (800, 800), (800, 800)]


def test_loop_position_and_natural_completion(window):
    window.selection_preview(0.5, 1.5)
    window.loop.setChecked(True)
    window.seek_play(1.0)
    window.play()
    player = window.player
    player.timer.stop()
    player.cancel.set()
    player.producer.join(timeout=2)
    player.queue = queue.Queue()
    for elapsed, expected in [(0.25, 1.25), (0.5, 0.5), (1.75, 0.75)]:
        player.queue.put(b"pcm")
        player.sink.elapsed = elapsed * 1e6
        player._pump()
        assert window.playback_position == expected
    player.stop()
    window.loop.setChecked(False)
    window.seek_play(1.0)
    window.play()
    player.timer.stop()
    player.cancel.set()
    player.producer.join(timeout=2)
    player.queue = queue.Queue()
    player.queue.put(None)
    player.sink.audio_state = QtMultimedia.QAudio.IdleState
    player._pump()
    assert player.state == "stopped"
    assert window.playback_position == 2
    window.play()
    assert player.origin == 0


def test_repeated_seeks_and_state_notifications(window):
    states = []
    window.player.stateChanged.connect(states.append)
    window.play()
    previous = []
    for seconds in (0.2, 0.7, 0.4, 1.2):
        previous.append((window.player.cancel, window.player.producer, window.player.sink))
        window.seek_play(seconds)
    for cancel, producer, sink in previous:
        producer.join(timeout=2)
        assert cancel.is_set() and sink.stopped and not producer.is_alive()
    assert window.player.origin == 1.2
    window.player.pause()
    window.player.pause()
    window.player.stop()
    assert states[-3:] == ["paused", "playing", "stopped"]


def test_timeline_disabled_and_minimum_span(qtbot):
    bar = PlaybackTimeline()
    qtbot.addWidget(bar)
    assert not bar.isEnabled()
    bar.set_duration(0.001, 48000)
    bar.set_selection(0.001, 0.001)
    assert bar.selection == pytest.approx((0.001 - 1 / 48000, 0.001))


def test_close_waits_for_cancelled_audio_workers(window, qtbot):
    finished = threading.Event()
    worker = threading.Thread(target=lambda: finished.wait(timeout=2))
    window.player.workers.append(worker)
    worker.start()
    try:
        window.close()
        assert window.isVisible()
        assert window.closing
    finally:
        finished.set()
        worker.join(timeout=2)
    qtbot.waitUntil(lambda: not window.isVisible(), timeout=5000)
