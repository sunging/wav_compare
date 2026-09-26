# Performance measurements

Measured on Windows 11 build 26200, Intel Core i7-13700H, 31.73 GiB RAM, CPython 3.12.13. The full synthetic workload includes preprocessing, statistics, disk-backed segments and waveform envelopes.

| Workload | Measurement | Target |
| --- | ---: | ---: |
| Two hours, 96 kHz, stereo RF64/PCM16 | 101.46 s | Report time |
| Peak RSS including GUI/batch stages | 337.24 MiB | ≤1024 MiB |
| Viewport update p95, warm filesystem | 24.36 ms | ≤100 ms |
| Cancel after progress callback | 0.92 ms | ≤1000 ms |
| 1000 one-second stereo pairs, 8 kHz | 32.08 s | Report time |
| Original segment loop, 120 s difference array | 90.31 ms | Baseline |
| Vectorized reductions on the same array | 62.38 ms | 1.45× faster |

These are the final verification run. An earlier implementation run measured 64.73 s / 359.62 MiB / 12.55 ms and 24.92 s for the batch; the final run includes the expanded fixed-delay search and was concurrent with development checks. The timings are not controlled A/B comparisons between those revisions. Both runs met memory, viewport and cancellation targets.

Run `uv run python tools/benchmark.py`. Generated WAVs and `results.json` are placed in ignored `.artifacts/benchmark`, requiring about 6 GiB of input storage. Different-rate B adds a double-precision resampling cache. Disk space is checked first. Very small segment sizes substantially increase disk-backed statistics storage.

Viewport timing includes cached envelope reads, plot data updates and Qt event processing with the offscreen platform. It is not a hardware frame-rate measurement. Cancellation is measured after progress, not during a stalled OS read. The original baseline measures the script's segment algorithm only; it does not establish a whole-application speed ratio. Overflow-prone original integer calculations are not used as a correctness oracle.

Storage, channel count, preprocessing and selection size affect runtime. RSS is sampled every 25 ms. CI checks numerical correctness, cache eviction, bounded plotting and spike preservation rather than enforcing unstable timing thresholds on shared machines.

Windows desktop testing used a Realtek output device, streamed over six seconds of generated audio and observed no Qt/player errors. CI does not certify audio hardware or audible quality on other machines.
