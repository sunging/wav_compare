"""Bounded viewport and spectral calculations, independent of widgets."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
from scipy import signal

from .engine import Comparison
from .models import Cancellation


class ByteCache:
    def __init__(self, budget=256 * 1024**2):
        self.budget, self.size = budget, 0
        self.items = OrderedDict()

    def get(self, key):
        item = self.items.get(key)
        if item is not None:
            self.items.move_to_end(key)
            return item[0]

    def put(self, key, value):
        size = sum(v.nbytes for v in value if isinstance(v, np.ndarray))
        if key in self.items:
            self.size -= self.items.pop(key)[1]
        if size > self.budget:
            return
        while self.size + size > self.budget and self.items:
            self.size -= self.items.popitem(last=False)[1][1]
        self.items[key] = value, size
        self.size += size

    def clear(self):
        self.items.clear()
        self.size = 0


def waveform(result: Comparison, begin: float, end: float, channel: int, pixels: int):
    start = max(0, min(result.frames - 1, int(begin * result.rate)))
    stop = min(result.frames, max(start + 1, int(end * result.rate)))
    pixels = max(100, min(pixels, 4096))
    if stop - start <= pixels * 2 or not result.peaks:
        a, b, d = result.samples(start, min(stop - start, pixels * 2))
        return (
            np.arange(start, start + len(a), dtype=np.float64) / result.rate,
            a[:, channel],
            b[:, channel],
            d[:, channel],
        )
    if (stop - start) / pixels < result.peaks[0][0]:
        # Refine below the coarsest cached bin without loading an entire view.
        width = max(1, int(np.ceil((stop - start) / pixels)))
        count = int(np.ceil((stop - start) / width))
        data = np.empty((count, 3, 2))
        bins_per_read = max(1, 65536 // width)
        for first in range(0, count, bins_per_read):
            lo = start + first * width
            values = result.samples(lo, min(bins_per_read * width, stop - lo))
            edges = np.arange(0, len(values[0]), width)
            for k, value in enumerate(values):
                data[first : first + len(edges), k, 0] = np.minimum.reduceat(
                    value[:, channel], edges
                )
                data[first : first + len(edges), k, 1] = np.maximum.reduceat(
                    value[:, channel], edges
                )
        times = np.minimum(start + np.arange(count) * width + width / 2, stop - 1) / result.rate
        return np.repeat(times, 2), *(data[:, k].reshape(-1) for k in range(3))
    chosen = result.peaks[0]
    for peak in result.peaks:
        if peak[0] <= (stop - start) / pixels:
            chosen = peak
    width, path, size = chosen
    lo, hi = start // width, min(size, (stop + width - 1) // width)
    array = np.load(path, mmap_mode="r")
    data = np.array(array[lo:hi, channel])
    # Also cap the finest level when the viewport falls between raw and envelope.
    stride = max(1, int(np.ceil(len(data) / pixels)))
    if stride > 1:
        edges = np.arange(0, len(data), stride)
        data = np.stack(
            [np.minimum.reduceat(data[..., 0], edges), np.maximum.reduceat(data[..., 1], edges)],
            axis=-1,
        )
    times = (
        np.minimum(
            np.arange(len(data)) * width * stride + lo * width + width / 2, result.frames - 1
        )
        / result.rate
    )
    return np.repeat(times, 2), *(data[:, k].reshape(-1) for k in range(3))


def spectra(
    result: Comparison,
    begin: float,
    end: float,
    channel: int,
    fft: int,
    hop: int,
    cancel: Cancellation,
):
    if fft < 32 or fft > 16384 or not 1 <= hop <= fft:
        raise ValueError("FFT must be 32..16384 and hop must be 1..FFT")
    lo, hi = max(0, round(begin * result.rate)), min(result.frames, round(end * result.rate))
    if hi - lo < 2:
        raise ValueError("Select at least two samples")
    nfft = min(fft, hi - lo)
    hop = min(hop, nfft)
    window = signal.windows.hann(nfft, sym=False)
    scale = result.rate * np.sum(window**2)
    total = np.zeros((3, nfft // 2 + 1))
    count = 0
    # Stream Welch windows with exact global hop; bounded windows per batch.
    for first in range(lo, hi - nfft + 1, hop * 64):
        cancel.check()
        last = min(hi, first + hop * 63 + nfft)
        values = result.samples(first, last - first)
        for k, value in enumerate(values):
            windows = np.lib.stride_tricks.sliding_window_view(value[:, channel], nfft)[::hop]
            windows = windows - windows.mean(axis=1, keepdims=True)
            power = np.abs(np.fft.rfft(windows * window, axis=1)) ** 2 / scale
            power[:, 1 : (-1 if nfft % 2 == 0 else None)] *= 2
            total[k] += power.sum(axis=0)
            if k == 0:
                count += len(windows)
    frequencies = np.fft.rfftfreq(nfft, 1 / result.rate)
    psd = 10 * np.log10(np.maximum(total / max(1, count), 1e-20))
    # Time-sample the STFT grid for an overview, never allocate an entire long STFT.
    n_windows = (hi - lo - nfft) // hop + 1
    indices = np.unique(np.linspace(0, n_windows - 1, min(n_windows, 512)).astype(int))
    image = np.empty((3, len(frequencies), len(indices)))
    for j, index in enumerate(indices):
        cancel.check()
        values = result.samples(lo + int(index) * hop, nfft)
        for k, value in enumerate(values):
            amplitude = np.abs(np.fft.rfft(value[:, channel] * window)) / max(window.sum(), 1)
            amplitude[1 : (-1 if nfft % 2 == 0 else None)] *= 2
            image[k, :, j] = 20 * np.log10(np.maximum(amplitude, 1e-10))
    times = (lo + indices * hop + nfft / 2) / result.rate
    return frequencies, psd, times, image
