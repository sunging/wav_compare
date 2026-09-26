"""Bounded audio I/O, streaming resampling and conservative delay estimation."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
from scipy import signal

from .models import Cancellation, Progress, no_progress


@dataclass(frozen=True)
class AudioInfo:
    path: str
    frames: int
    samplerate: int
    channels: int
    subtype: str
    size: int
    mtime_ns: int

    @classmethod
    def read(cls, path: str | Path):
        path = Path(path).resolve()
        info = sf.info(path)
        stat = path.stat()
        if info.format not in ("WAV", "WAVEX", "RF64", "W64"):
            raise ValueError(f"Not a WAV-family file: {path}")
        if info.frames == 0:
            raise ValueError(f"Empty audio: {path}")
        return cls(
            str(path),
            info.frames,
            info.samplerate,
            info.channels,
            info.subtype,
            stat.st_size,
            stat.st_mtime_ns,
        )


def ensure_space(directory: Path, required: int):
    if shutil.disk_usage(directory).free < required + 64 * 1024**2:
        raise OSError(f"Insufficient cache disk space (need {required / 1024**2:.0f} MiB)")


def read_range(path: str | Path, start: int, frames: int) -> np.ndarray:
    with sf.SoundFile(path) as stream:
        stream.seek(max(0, int(start)))
        data = stream.read(max(0, int(frames)), dtype="float64", always_2d=True)
    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite audio samples: {path}")
    return data


def resample_file(
    info: AudioInfo,
    rate: int,
    directory: Path,
    cancel: Cancellation,
    progress: Progress = no_progress,
    block_size: int = 65536,
) -> str:
    target = directory / "resampled.w64"
    ensure_space(directory, math.ceil(info.frames * rate / info.samplerate) * info.channels * 8)
    converter = soxr.ResampleStream(
        info.samplerate, rate, info.channels, dtype="float64", quality="HQ"
    )
    with (
        sf.SoundFile(info.path) as source,
        sf.SoundFile(
            target, "w", samplerate=rate, channels=info.channels, format="W64", subtype="DOUBLE"
        ) as destination,
    ):
        while source.tell() < len(source):
            cancel.check()
            block = source.read(block_size, dtype="float64", always_2d=True)
            if not np.all(np.isfinite(block)):
                raise ValueError("Non-finite audio samples")
            output = converter.resample_chunk(block, last=source.tell() == len(source))
            destination.write(output)
            progress(source.tell() / len(source), "Resampling B")
    return str(target)


def select_channels(data: np.ndarray, indices: list[int], mix: bool) -> np.ndarray:
    return data.mean(axis=1, keepdims=True) if mix else data[:, indices]


def numeric(data: np.ndarray, mode: str) -> np.ndarray:
    if mode.startswith("pcm"):
        scale = 2 ** (int(mode[3:]) - 1)
        return np.clip(np.rint(data * scale), -scale, scale - 1).astype(np.int64)
    return data.astype(mode, copy=False)


def estimate_delay(
    a: str, b: str, rate: int, max_lag: float, channel_a: int, channel_b: int, cancel: Cancellation
) -> dict:
    """Use up to three distributed windows; positive lag means B is delayed.

    Coarse correlation uses <=8 kHz samples; a short full-rate window refines
    the integer lag. Ambiguous periodic peaks and inconsistent windows fail closed.
    """
    frames = min(sf.info(a).frames, sf.info(b).frames)
    window = min(frames, round(rate * max(10, 3 * max_lag + 1)))
    if window < 32:
        return {"status": "unreliable", "lag_samples": 0, "reason": "too_short"}
    # Bound every FFT even for high-rate/long inputs.
    step = max(1, math.ceil(rate / 8000), math.ceil(window / 262144))
    limit = min(round(max_lag * rate / step), (window // step) // 3)
    starts = sorted({0, max(0, (frames - window) // 2), max(0, frames - window)})
    candidates = []
    scores = []
    for start in starts:
        cancel.check()

        def coarse_channel(path, channel, position=start):
            pieces = []
            converter = soxr.ResampleStream(rate, rate / step, 1, dtype="float64")
            with sf.SoundFile(path) as stream:
                stream.seek(position)
                remaining = window
                while remaining:
                    cancel.check()
                    data = stream.read(min(65536, remaining), dtype="float64", always_2d=True)
                    if not len(data) or not np.isfinite(data).all():
                        raise ValueError("Invalid or changed alignment input")
                    remaining -= len(data)
                    pieces.append(
                        converter.resample_chunk(data[:, channel].copy(), last=remaining == 0)
                    )
            return np.concatenate(pieces)

        x = coarse_channel(a, channel_a)
        y = coarse_channel(b, channel_b)
        x -= x.mean()
        y -= y.mean()
        energy = float(np.linalg.norm(x) * np.linalg.norm(y))
        if energy < len(x) * 1e-12:
            continue
        corr = signal.correlate(y, x, method="fft") / energy
        middle = len(x) - 1
        crop = corr[middle - limit : middle + limit + 1]
        peak = int(np.argmax(crop))
        score = float(crop[peak])
        # Ignore the central lobe, but reject other similarly strong peaks.
        others = crop.copy()
        radius = max(2, round(rate / step * 0.002))
        others[max(0, peak - radius) : peak + radius + 1] = -np.inf
        if score < 0.35 or (np.max(others, initial=-np.inf) > score * 0.92):
            continue
        coarse = (peak - limit) * step
        # Read only a short full-rate refinement window, even for minute-scale lags.
        base_a, base_b = max(0, -coarse), max(0, coarse)
        refine_start = max(step, min(window // 2, window - max(base_a, base_b) - rate * 2 - step))
        count = min(rate * 2, window - refine_start - max(base_a, base_b) - step)
        if count <= 16:
            continue
        raw_a = read_range(a, start + refine_start + base_a, count)[:, channel_a].copy()
        raw_b = read_range(b, start + refine_start + base_b - step, count + step * 2)[
            :, channel_b
        ].copy()
        best, best_score = coarse, -np.inf
        for lag in range(coarse - step, coarse + step + 1):
            cancel.check()
            if abs(lag) > round(max_lag * rate):
                continue
            sb = lag - coarse + step
            n = min(len(raw_a), len(raw_b) - sb)
            if n <= 16:
                continue
            xa, yb = raw_a[:n], raw_b[sb : sb + n]
            xa, yb = xa - xa.mean(), yb - yb.mean()
            den = np.linalg.norm(xa) * np.linalg.norm(yb)
            value = float(np.dot(xa, yb) / den) if den > 1e-12 else -np.inf
            if value > best_score:
                best, best_score = lag, value
        if best_score >= 0.35:
            candidates.append(best)
            scores.append(best_score)
    if not candidates or (len(starts) > 1 and len(candidates) < 2):
        return {
            "status": "unreliable",
            "lag_samples": 0,
            "reason": "silence_or_ambiguous_correlation",
            "candidates": candidates,
        }
    if max(candidates) - min(candidates) > 2:
        return {
            "status": "unreliable",
            "lag_samples": 0,
            "reason": "inconsistent_windows",
            "candidates": candidates,
        }
    return {
        "status": "estimated",
        "lag_samples": int(np.median(candidates)),
        "confidence": float(min(scores)),
        "candidates": candidates,
    }
