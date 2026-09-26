# WAV Compare

[![Checks](https://github.com/sunging/wav_compare/actions/workflows/checks.yml/badge.svg)](https://github.com/sunging/wav_compare/actions/workflows/checks.yml)
[中文说明](README.zh-CN.md) · [Comparison semantics](docs/comparison.md) · [Releasing](docs/releasing.md)

A desktop workbench and headless CLI for comparing WAV audio, from **8 kHz** speech to high-rate multichannel recordings. Built with PySide6, pyqtgraph, NumPy, SoundFile, SciPy and SoXR.

![Workbench showing synthetic audio](docs/images/workbench.png)

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```sh
uv tool install --python 3.14 git+https://github.com/sunging/wav_compare.git
wav-compare
```

Python 3.11 or newer is required; the development environment defaults to Python 3.14. uv can download Python automatically. If the command is not on your PATH, run `uv tool update-shell` and reopen your terminal. The first formal release is **0.1.0**, prepared in an open release PR. Until that PR is merged, main reports the unreleased bootstrap version **0.0.0**.

After a version has actually been released, append its tag to the Git URL, e.g. `git+https://github.com/sunging/wav_compare.git@v0.1.0`. No PyPI publication or standalone installer is required.

Windows, macOS and Linux are covered by CI. Linux needs a graphical session and Qt system libraries (Debian/Ubuntu: `libegl1 libopengl0 libxkbcommon0 libpulse0`, plus your desktop's Qt/XCB dependencies). Playback needs a working audio device. CLI comparison never opens a window or audio device.

## Workbench

- Open or drop two files or folders to compare automatically. Path and analysis-option changes are debounced for 400 ms; only the latest inputs are analyzed. Recursive folder matching uses case-insensitive relative paths and reports missing files and collisions. The first analyzable result opens automatically; recomputing a batch restores the previously selected relative path when possible.
- Collapse **Files & results** and **Parameters and metrics** independently using their header arrows, the persistent top toggles or the **View** menu. Plots expand into the freed space; reopening a panel restores its width. **View → Reset layout** restores the default layout without changing analysis options.
- Inspect A/B waveforms, B − A differences, segment statistics, Welch spectra and STFT spectrograms.
- Each plot has an independent **Show indicator** toggle, enabled by default and saved locally. Hover for values; click to pin or unpin, and select the readout text to copy it. Pins are independent of other plots and playback. Segment readings use original segment statistics; spectra show PSD in dB/Hz, and spectrograms show the analyzed frame's actual amplitude in dB without color-range clipping. Sample and segment indexes are zero-based.
- Drag inside waveform plots to pan, or drag an axis to pan that axis only. Drag selection edges to resize the selection. Panning is bounded by the audio duration and current channel's global amplitude range with a small margin; Reset zoom restores the full view.
- Spectrum and spectrogram frequencies stay within 0–Nyquist. Spectrum power is bounded by the analyzed values with a margin; spectrogram time stays within the analyzed selection. Reset zoom also restores these plots.
- Default preprocessing resamples **B to A's rate**, then estimates one fixed delay. Every transformation is reported. Strict mode disables both.
- Compare all channels by index, enter explicit **1-based** pairs (`1:1,2:2`), or deliberately mix each input to mono.
- Select a region, find the largest difference, step through differing regions, and listen to A, B or their difference. Playback volume never changes metrics.
- FFT/hop changes update only the spectrum, with computation deferred until a spectral tab is visible. Selection edits do not rerun analysis: use the **Selection** menu below the plots or the **Analysis** menu to analyze the selection, compare the full files or refresh the spectrum.
- **Settings → Preferences** offers English/Chinese, system/light/dark themes, automatic comparison, startup comparison and path restoration. All three behavior switches default to enabled. Existing local QSettings are preserved and extended to remember panel visibility/widths, geometry, analysis options, FFT/hop and playback preferences. Results, playback position and temporary selections are not persisted. Disabling path restoration clears saved paths; explicit launch paths take precedence.
- Manual comparison remains available. Cancel stops pending automatic comparison and current tasks without retrying until another input change or manual comparison. Disabling automatic comparison leaves running work alone. Exports live in **File**, and **Clear cache** in **Analysis**.
- Export versioned JSON or CSV. Single-file export reflects the current selection analysis; directory export contains the complete batch.

The playback timeline supports click-to-seek, dragging in either direction to select, dragging selection edges to resize, and **Select all** in its context menu. Selection stays synchronized with the waveform and time inputs. Seeking preserves playing, paused or stopped state; Stop keeps the position. Without looping, playback continues to the audio end; with looping, it stays inside the selection. Seeking into a loop plays its remainder first, then repeats the entire selection. Playing again at the end restarts from the beginning. Changing the source stops playback. Original-input playback uses each source's original seconds and is limited to that file's end; processed playback follows the aligned timeline.

Keyboard: **Ctrl+Enter** compare, **Esc** cancel, **Ctrl+0** reset zoom, **Space** pause/resume, **Ctrl+Q** exit. With the timeline focused, **Left/Right** seek by 0.1 seconds and **Home/End** seek to the effective playback bounds.

## Command line

```sh
wav-compare gui reference.wav candidate.wav
wav-compare compare reference.wav candidate.wav --json report.json --csv report.csv
wav-compare compare before/ after/ --strict --threshold 0.0001 --json batch.json
wav-compare compare a.wav b.wav --channels 1:2,2:1 --no-align
wav-compare compare a.wav b.wav --offset 160 --region 2 8
wav-compare --version
```

JSON goes to stdout, errors to stderr. Positive `--offset` means B is delayed, measured in A-rate samples. `--region START END` selects seconds on the aligned overlap timeline. Numeric modes are `float64` (default normalized samples), `float32`, `pcm16`, `pcm24`, `pcm32`; thresholds use the chosen mode's units. See `wav-compare compare --help`.

Exit codes: **0** within threshold without missing channels/tails; **1** differences or missing files; **2** invalid input, collision, empty batch or processing errors; **130** cancellation. Batch failures do not discard successful rows.

## Performance and development

Block reads, streaming resampling, multilevel waveform envelopes and disk-backed statistics avoid retaining full recordings in RAM. The GUI uses at most two workers and a 256 MiB viewport cache. Selecting a directory result does not rerun the batch. See [measured performance](docs/performance.md).

```sh
git clone https://github.com/sunging/wav_compare.git
cd wav_compare
uv sync --locked
uv run wav-compare
uv run ruff check .
uv run pytest -q
uv build
uv run python tools/check_version.py
```

Tests synthesize their own audio. The package separates `audio`, `engine`, `batch`, `views`, `reports`, `cli` and `ui`. Core/CLI imports do not initialize Qt. `tools/benchmark.py` writes ignored synthetic data to `.artifacts`; the default workload needs about 6 GiB of disk. Resampling B adds roughly `frames × channels × 8` bytes of disk cache.

## Versions and provenance

Conventional Commits feed release-please, which maintains the version and [CHANGELOG](CHANGELOG.md) in a release PR. Merging that PR is the release decision; ordinary pushes do not automatically merge it. See [release maintenance](docs/releasing.md).

Based on [sunging/py_script's wav_compare_qt.py](https://github.com/sunging/py_script/blob/1780b081541e424d73ae2b0e24edebfc7f7ab4af/wav_compare_qt.py), commit `1780b081541e424d73ae2b0e24edebfc7f7ab4af`. The independent implementation fixes integer overflow, first-channel-only comparisons, inconsistent sample-rate handling and synchronous directory recomputation. [MIT](LICENSE), with the source owner's authorization. Dependencies retain their respective licenses.
