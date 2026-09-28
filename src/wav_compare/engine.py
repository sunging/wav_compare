"""Streaming comparisons. No Qt imports and no full-file in-memory arrays."""

from __future__ import annotations

import math
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio import (
    AudioInfo,
    ensure_space,
    estimate_delay,
    numeric,
    read_range,
    resample_file,
    select_channels,
)
from .models import Cancellation, Options, Progress, no_progress

PEAK_BIN = 1024


class SegmentStore:
    """Read slices without leaking mmap lifetimes into callers on Windows."""

    def __init__(self, owner):
        self.owner = owner
        self.shape = (
            math.ceil(owner.frames / owner.options.segment_size),
            len(owner.channels[0]),
            4,
        )

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        data = np.load(self.owner.segments_path, mmap_mode="r")
        try:
            return np.array(data[index])
        finally:
            data._mmap.close()

    def blocks(self, start, stop, channel, size=65536, reverse=False):
        """Yield (first, rows) slices of one channel, opening the file once per scan."""
        start, stop = max(0, start), min(len(self), stop)
        if stop <= start:
            return
        data = np.load(self.owner.segments_path, mmap_mode="r")
        try:
            firsts = range(start, stop, size)
            for first in reversed(firsts) if reverse else firsts:
                yield first, np.array(data[first : min(first + size, stop), channel])
        finally:
            data._mmap.close()


@dataclass
class Comparison:
    report: dict
    options: Options
    a_path: str
    b_path: str
    starts: tuple[int, int]
    frames: int
    rate: int
    channels: tuple[list[int], list[int]]
    temporary: tempfile.TemporaryDirectory = field(repr=False)
    segments_path: str | None = None
    peaks: list[tuple[int, str, int]] = field(default_factory=list)

    def samples(self, start: int, count: int):
        start = max(0, min(start, self.frames))
        count = max(0, min(count, self.frames - start))
        a = read_range(self.a_path, self.starts[0] + start, count)
        b = read_range(self.b_path, self.starts[1] + start, count)
        a = numeric(select_channels(a, self.channels[0], self.options.mix), self.options.mode)
        b = numeric(select_channels(b, self.channels[1], self.options.mix), self.options.mode)
        return a, b, b.astype(np.float64) - a.astype(np.float64)

    def segment_data(self):
        if not self.segments_path:
            return np.empty((0, len(self.channels[0]), 4))
        return SegmentStore(self)


def _bins(start: int, count: int, width: int):
    first = width - start % width
    edges = np.r_[0, np.arange(first, count, width)]
    indices = (start + edges) // width
    counts = np.diff(np.r_[edges, count])
    return edges, indices, counts


def compare(
    a_path: str | Path,
    b_path: str | Path,
    options: Options | None = None,
    cancel: Cancellation | None = None,
    progress: Progress = no_progress,
    details: bool = True,
) -> Comparison:
    options, cancel = options or Options(), cancel or Cancellation()
    options.validate()
    cancel.check()
    a_info, b_info = AudioInfo.read(a_path), AudioInfo.read(b_path)
    if options.strict and a_info.samplerate != b_info.samplerate:
        raise ValueError("Strict comparison requires matching sample rates")
    if options.pairs:
        if any(
            a < 0 or b < 0 or a >= a_info.channels or b >= b_info.channels for a, b in options.pairs
        ):
            raise ValueError("Channel index is out of range")
        ca, cb = [p[0] for p in options.pairs], [p[1] for p in options.pairs]
    else:
        ca = cb = list(range(1 if options.mix else min(a_info.channels, b_info.channels)))
    # A late mmap release on Windows must not turn result disposal into an error.
    temporary = tempfile.TemporaryDirectory(prefix="wav-compare-", ignore_cleanup_errors=True)
    directory = Path(temporary.name)
    try:
        rate = a_info.samplerate
        resampled = rate != b_info.samplerate
        b_prepared = (
            resample_file(
                b_info,
                rate,
                directory,
                cancel,
                lambda p, s: progress(p * 0.3, s),
                options.block_size,
            )
            if resampled
            else b_info.path
        )
        b_frames = sf.info(b_prepared).frames
        if options.strict or (not options.align and options.offset is None):
            alignment = {"status": "disabled", "lag_samples": 0}
        elif options.offset is not None:
            alignment = {"status": "manual", "lag_samples": options.offset}
        else:
            progress(0.3, "Estimating fixed delay")
            alignment = estimate_delay(
                a_info.path, b_prepared, rate, options.max_lag, ca[0], cb[0], cancel
            )
        lag = alignment["lag_samples"]
        sa, sb = max(0, -lag), max(0, lag)
        frames = min(a_info.frames - sa, b_frames - sb)
        if frames <= 0:
            raise ValueError("No overlapping audio after alignment")
        overlap = frames
        if options.region:
            begin = round(options.region[0] * rate)
            end = min(frames, round(options.region[1] * rate))
            if begin >= end:
                raise ValueError("Selected region does not overlap the audio")
            sa, sb, frames = sa + begin, sb + begin, end - begin
        nchan = len(ca)
        nseg = math.ceil(frames / options.segment_size)
        npeak = math.ceil(frames / PEAK_BIN)
        segments = peaks = None
        if details:
            ensure_space(directory, nseg * nchan * 4 * 8 + npeak * nchan * 3 * 2 * 8 * 2)
            segments = np.lib.format.open_memmap(
                directory / "segments.npy", mode="w+", dtype="float64", shape=(nseg, nchan, 4)
            )
            segments[:] = 0
            peaks = np.lib.format.open_memmap(
                directory / "peaks0.npy", mode="w+", dtype="float64", shape=(npeak, nchan, 3, 2)
            )
            peaks[..., 0], peaks[..., 1] = np.inf, -np.inf
        sums = np.zeros((nchan, 8), dtype=np.float64)
        maximum, maximum_at = np.zeros(nchan), np.zeros(nchan, dtype=np.int64)
        exceeded = np.zeros(nchan, dtype=np.int64)
        with sf.SoundFile(a_info.path) as fa, sf.SoundFile(b_prepared) as fb:
            fa.seek(sa)
            fb.seek(sb)
            for start in range(0, frames, options.block_size):
                cancel.check()
                count = min(options.block_size, frames - start)
                aa = fa.read(count, dtype="float64", always_2d=True)
                bb = fb.read(count, dtype="float64", always_2d=True)
                if len(aa) != count or len(bb) != count:
                    raise OSError("Input changed while reading")
                if not (np.isfinite(aa).all() and np.isfinite(bb).all()):
                    raise ValueError("Non-finite audio samples")
                aa = numeric(select_channels(aa, ca, options.mix), options.mode)
                bb = numeric(select_channels(bb, cb, options.mix), options.mode)
                # int64 subtraction preserves PCM32's entire difference range.
                dd = (
                    (bb - aa).astype(np.float64)
                    if options.mode.startswith("pcm")
                    else bb.astype(np.float64) - aa.astype(np.float64)
                )
                aa, bb = aa.astype(np.float64), bb.astype(np.float64)
                absolute = np.abs(dd)
                local_at = np.argmax(absolute, axis=0)
                local_max = absolute[local_at, np.arange(nchan)]
                changed = local_max > maximum
                maximum_at[changed] = start + local_at[changed]
                maximum = np.maximum(maximum, local_max)
                exceeded += np.count_nonzero(absolute > options.threshold, axis=0)
                sums += np.stack(
                    [
                        absolute.sum(axis=0),
                        (dd * dd).sum(axis=0),
                        aa.sum(axis=0),
                        bb.sum(axis=0),
                        (aa * aa).sum(axis=0),
                        (bb * bb).sum(axis=0),
                        (aa * bb).sum(axis=0),
                        dd.sum(axis=0),
                    ],
                    axis=1,
                )
                if details:
                    edges, indices, counts = _bins(start, count, options.segment_size)
                    segments[indices, :, 0] = np.maximum(
                        segments[indices, :, 0], np.maximum.reduceat(absolute, edges)
                    )
                    segments[indices, :, 1] += np.add.reduceat(absolute, edges)
                    segments[indices, :, 2] += np.add.reduceat(dd * dd, edges)
                    segments[indices, :, 3] += counts[:, None]
                    edges, indices, _ = _bins(start, count, PEAK_BIN)
                    for k, values in enumerate((aa, bb, dd)):
                        peaks[indices, :, k, 0] = np.minimum(
                            peaks[indices, :, k, 0], np.minimum.reduceat(values, edges)
                        )
                        peaks[indices, :, k, 1] = np.maximum(
                            peaks[indices, :, k, 1], np.maximum.reduceat(values, edges)
                        )
                progress(0.4 + 0.55 * (start + count) / frames, "Comparing audio")
        # Do not associate a result with file fingerprints that changed during work.
        if AudioInfo.read(a_info.path) != a_info or AudioInfo.read(b_info.path) != b_info:
            raise OSError("Input changed during comparison; please compare again")
        if not np.isfinite(sums).all():
            raise ValueError("Audio magnitude exceeds supported finite statistics range")
        metrics = []
        for c in range(nchan):
            abs_sum, square, sx, sy, xx, yy, xy, _ = sums[c]
            variance = max(0.0, xx - sx * sx / frames) * max(0.0, yy - sy * sy / frames)
            correlation = (
                float(np.clip((xy - sx * sy / frames) / math.sqrt(variance), -1, 1))
                if variance > 0
                else None
            )
            snr = 10 * math.log10(xx / square) if xx > 0 and square > 0 else None
            metrics.append(
                {
                    "a_channel": "mix" if options.mix else ca[c] + 1,
                    "b_channel": "mix" if options.mix else cb[c] + 1,
                    "max_abs": float(maximum[c]),
                    "max_at_sample": int(maximum_at[c]),
                    "mae": float(abs_sum / frames),
                    "rmse": math.sqrt(square / frames),
                    "correlation": correlation,
                    "snr_db": snr,
                    "snr_status": "infinite"
                    if square == 0 and xx > 0
                    else "undefined"
                    if xx == 0
                    else "finite",
                    "above_threshold": int(exceeded[c]),
                    "above_threshold_ratio": float(exceeded[c] / frames),
                }
            )
        missing_channels = (
            not options.pairs and not options.mix and a_info.channels != b_info.channels
        )
        length_mismatch = (a_info.frames - sa) != (b_frames - sb)
        status = (
            "different"
            if exceeded.any() or missing_channels or length_mismatch
            else "within_threshold"
        )
        # An uncertain delay is explicit; it is not silently advertised as aligned.
        report = {
            "a": asdict(a_info),
            "b": asdict(b_info),
            "options": options.report(),
            "status": status,
            "comparison_domain": "processed" if resampled or lag else "unshifted",
            "resampled_b": resampled,
            "analysis_samplerate": rate,
            "alignment": alignment,
            "frames_compared": frames,
            "a_start_sample": sa,
            "b_start_sample": sb,
            "overlap_frames": overlap,
            "excluded": {
                "a_before": sa,
                "b_before": sb,
                "a_after": a_info.frames - sa - frames,
                "b_after": b_frames - sb - frames,
                "missing_channels": missing_channels,
            },
            "metrics": metrics,
        }
        result = Comparison(
            report, options, a_info.path, b_prepared, (sa, sb), frames, rate, (ca, cb), temporary
        )
        if details:
            segments.flush()
            peaks.flush()
            result.segments_path = str(directory / "segments.npy")
            result.peaks = [(PEAK_BIN, str(directory / "peaks0.npy"), npeak)]
            previous, width, level = peaks, PEAK_BIN, 0
            while len(previous) > 1024:
                cancel.check()
                level += 1
                path = directory / f"peaks{level}.npy"
                size = math.ceil(len(previous) / 4)
                nxt = np.lib.format.open_memmap(
                    path, mode="w+", dtype="float64", shape=(size, nchan, 3, 2)
                )
                for i in range(0, size, 16384):
                    cancel.check()
                    block = previous[i * 4 : min((i + 16384) * 4, len(previous))]
                    edges = np.arange(0, len(block), 4)
                    nxt[i : i + len(edges), ..., 0] = np.minimum.reduceat(block[..., 0], edges)
                    nxt[i : i + len(edges), ..., 1] = np.maximum.reduceat(block[..., 1], edges)
                nxt.flush()
                width *= 4
                result.peaks.append((width, str(path), size))
                previous = nxt
            del previous, peaks, segments
        progress(1.0, "Complete")
        return result
    except BaseException:
        # Release mmap handles before TemporaryDirectory removes files on Windows.
        for name in ("segments", "peaks", "previous", "nxt"):
            value = locals().get(name)
            if isinstance(value, np.memmap):
                value._mmap.close()
        temporary.cleanup()
        raise
