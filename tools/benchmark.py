"""Repeatable synthetic benchmark; output goes to ignored .artifacts only."""

import argparse
import json
import os
import platform
import shutil
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import psutil
import soundfile as sf
from PySide6 import QtWidgets

from wav_compare.batch import discover, run_batch
from wav_compare.engine import compare
from wav_compare.models import Cancellation, Cancelled, Options
from wav_compare.ui.app import Window
from wav_compare.views import waveform


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--pairs", type=int, default=1000)
    args = parser.parse_args()
    root = Path(".artifacts/benchmark")
    root.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory():
        while not stop.wait(0.025):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    rate = 96000
    data = np.random.default_rng(111).normal(0, 0.15, (rate, 2))
    a, b = root / "a.wav", root / "b.wav"
    with sf.SoundFile(
        a, "w", samplerate=rate, channels=2, subtype="PCM_16", format="RF64"
    ) as output:
        for _ in range(args.seconds):
            output.write(data)
    shutil.copyfile(a, b)
    start = time.perf_counter()
    result = compare(a, b)
    elapsed = time.perf_counter() - start
    print(f"Long comparison: {elapsed:.3f} s", flush=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = Window()
    window.set_result(result)
    window.show()
    window.jobs.cancel_all()
    latencies = []
    for i in range(40):
        start = time.perf_counter()
        value = waveform(result, (i % 2) * args.seconds / 4, (i % 2 + 1) * args.seconds / 4, 0, 800)
        window.draw_data(value)
        app.processEvents()
        latencies.append((time.perf_counter() - start) * 1000)
    # Cancel a running comparison after at least one progress notification.
    cancel = Cancellation()
    cancelled_at = [None]

    def progress(*_):
        if cancelled_at[0] is None:
            cancelled_at[0] = time.perf_counter()
            cancel.cancel()

    try:
        compare(a, b, Options(strict=True), cancel, progress, details=False)
    except Cancelled:
        cancel_ms = (time.perf_counter() - cancelled_at[0]) * 1000
    for folder in ("left", "right"):
        (root / folder).mkdir(exist_ok=True)
    for i in range(args.pairs):
        for folder in ("left", "right"):
            sf.write(root / folder / f"{i:05d}.wav", data[:8000], 8000, subtype="PCM_16")
    start = time.perf_counter()
    rows = run_batch(discover(str(root / "left"), str(root / "right")), Options(), Cancellation())
    batch_seconds = time.perf_counter() - start
    print(f"Batch: {batch_seconds:.3f} s", flush=True)
    # Compare the original script's per-segment loop against vectorized reductions.
    diff = np.random.default_rng(22).normal(size=96000 * 120).astype("float32")
    start = time.perf_counter()
    reference = [
        (float(np.max(np.abs(diff[i : i + 1000]))), float(np.mean(np.abs(diff[i : i + 1000]))))
        for i in range(0, len(diff), 1000)
    ]
    original_seconds = time.perf_counter() - start
    start = time.perf_counter()
    absolute = np.abs(diff)
    edges = np.arange(0, len(diff), 1000)
    fast_max = np.maximum.reduceat(absolute, edges)
    fast_mean = np.add.reduceat(absolute.astype("float64"), edges) / np.diff(
        np.r_[edges, len(diff)]
    )
    optimized_seconds = time.perf_counter() - start
    np.testing.assert_allclose(np.column_stack((fast_max, fast_mean)), reference, rtol=1e-6)
    stop.set()
    monitor.join()
    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "memory_gib": psutil.virtual_memory().total / 1024**3,
        "duration_seconds": args.seconds,
        "sample_rate": rate,
        "channels": 2,
        "comparison_seconds": elapsed,
        "peak_rss_mib": peak[0] / 1024**2,
        "viewport_p95_ms": float(np.percentile(latencies[5:], 95)),
        "cancel_ms": cancel_ms,
        "pairs": len(rows),
        "batch_seconds": batch_seconds,
        "original_segment_loop_seconds": original_seconds,
        "vector_segment_seconds": optimized_seconds,
        "all_pairs_within_threshold": all(r["status"] == "within_threshold" for r in rows),
    }
    (root / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    window.jobs.cancel_all()
    while window.jobs.jobs:
        app.processEvents()
        time.sleep(0.01)
    window.close()


if __name__ == "__main__":
    main()
