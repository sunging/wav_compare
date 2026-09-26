"""Bounded PCM producer. Disk I/O stays on a worker, Qt audio stays on the GUI thread."""

from __future__ import annotations

import queue
import threading

import numpy as np
import soxr
from PySide6 import QtCore, QtMultimedia

from ..audio import read_range


class Player(QtCore.QObject):
    error = QtCore.Signal(str)
    position = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.sink = None
        self.producer = None
        self.cancel = threading.Event()
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(15)
        self.timer.timeout.connect(self._pump)
        self.pending = b""
        self.paused = False
        self.volume = 0.5

    def play(self, result, source, channel, begin, end, loop=False, original=False):
        self.stop()
        device = QtMultimedia.QMediaDevices.defaultAudioOutput()
        if device.isNull():
            self.error.emit("No audio output device")
            return
        fmt = device.preferredFormat()
        if fmt.sampleFormat() not in (
            QtMultimedia.QAudioFormat.Float,
            QtMultimedia.QAudioFormat.Int16,
            QtMultimedia.QAudioFormat.Int32,
        ):
            self.error.emit("Unsupported output device format")
            return
        self.sink = QtMultimedia.QAudioSink(device, fmt, self)
        self.sink.setVolume(self.volume)
        self.sink.setBufferSize(max(4096, fmt.bytesForDuration(200000)))
        self.output = self.sink.start()
        if self.output is None:
            self.error.emit("Cannot open audio output device")
            self.stop()
            return
        self.queue = queue.Queue(maxsize=8)
        self.cancel = threading.Event()
        self.origin, self.end = begin, end
        self.loop = loop
        self.finished = False
        self.paused = False
        self.pending = b""
        args = result, source, channel, begin, end, loop, original, fmt, self.queue, self.cancel
        self.producer = threading.Thread(target=self._produce, args=args, daemon=True)
        self.producer.start()
        self.timer.start()

    def _produce(self, result, source, channel, begin, end, loop, original, fmt, output, cancel):
        def put(value):
            while not cancel.is_set():
                try:
                    output.put(value, timeout=0.05)
                    return True
                except queue.Full:
                    pass
            return False

        try:
            while not cancel.is_set():
                rate = result.rate
                if original and source in (0, 1):
                    info = result.report["a" if source == 0 else "b"]
                    rate = info["samplerate"]
                lo, hi = round(begin * rate), round(end * rate)
                if hi <= lo:
                    break
                converter = soxr.ResampleStream(rate, fmt.sampleRate(), 1, dtype="float32")
                for start in range(lo, hi, 8192):
                    if cancel.is_set():
                        return
                    count = min(8192, hi - start)
                    if original and source in (0, 1):
                        info = result.report["a" if source == 0 else "b"]
                        data = read_range(
                            info["path"], start, min(count, max(0, info["frames"] - start))
                        )
                        value = (
                            data.mean(axis=1)
                            if result.options.mix
                            else data[:, result.channels[source][channel]]
                        )
                    else:
                        value = result.samples(start, count)[source][:, channel]
                        if result.options.mode.startswith("pcm"):
                            value = value / 2 ** (int(result.options.mode[3:]) - 1)
                    value = converter.resample_chunk(
                        value.astype("float32"), last=start + count == hi
                    )
                    value = np.repeat(np.clip(value, -1, 1)[:, None], fmt.channelCount(), axis=1)
                    if fmt.sampleFormat() == QtMultimedia.QAudioFormat.Int16:
                        value = np.clip(np.rint(value * 32768), -32768, 32767).astype("int16")
                    elif fmt.sampleFormat() == QtMultimedia.QAudioFormat.Int32:
                        value = np.clip(
                            np.rint(value.astype("float64") * 2147483648), -2147483648, 2147483647
                        ).astype("int32")
                    else:
                        value = value.astype("float32")
                    if not put(value.tobytes()):
                        return
                if not loop:
                    break
            put(None)
        except Exception as error:
            put(str(error))

    def _pump(self):
        if not self.sink or self.paused:
            return
        try:
            if not self.pending and not self.finished:
                value = self.queue.get_nowait()
                if value is None:
                    self.finished = True
                elif isinstance(value, str):
                    self.error.emit(value)
                    self.stop()
                    return
                else:
                    self.pending = value
            if self.pending:
                size = min(len(self.pending), self.sink.bytesFree())
                if size:
                    written = self.output.write(self.pending[:size])
                    if written < 0:
                        raise OSError("Audio output write failed")
                    self.pending = self.pending[written:]
            elapsed = self.sink.processedUSecs() / 1e6
            duration = max(0.001, self.end - self.origin)
            self.position.emit(
                self.origin + (elapsed % duration if self.loop else min(elapsed, duration))
            )
            if (
                self.finished
                and not self.pending
                and self.sink.state() == QtMultimedia.QAudio.IdleState
            ):
                self.stop()
        except queue.Empty:
            pass
        except Exception as error:
            self.error.emit(str(error))
            self.stop()

    def pause(self):
        if self.sink:
            self.paused = not self.paused
            self.sink.suspend() if self.paused else self.sink.resume()

    def set_volume(self, value):
        self.volume = value
        if self.sink:
            self.sink.setVolume(value)

    def stop(self):
        self.cancel.set()
        self.timer.stop()
        if self.sink:
            self.sink.stop()
            self.sink.deleteLater()
            self.sink = None
        # Producer owns result until it observes cancellation; never delete its cache early.
        self.pending = b""
