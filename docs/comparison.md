# Comparison semantics

## Timeline

A is the reference. A stateful SoXR HQ stream resamples B to A's rate when necessary, flushing once at EOF. Automatic alignment uses up to three distributed windows, band-limited coarse correlation and full-rate refinement. Search defaults to ±5 seconds. Positive lag means B is late: skip B's prefix. Negative lag skips A's prefix. No gain or polarity correction is applied.

Correlation below 0.35, competing peaks above 92% of the main peak, silence, or estimates differing by more than two samples are rejected. Short inputs may have one window; otherwise at least two windows must agree. Alignment uses the first selected channel pair (original channel 1 in mix mode). A rejected estimate uses zero lag and is reported as `unreliable`, with a reason. Metrics then describe that unshifted overlap, not successful alignment.

Strict mode requires equal rates and applies no lag. Manual offset overrides automatic alignment outside strict mode. Clock-drift compensation, time stretching and content matching are not performed.

Durations use integer frames, each containing one sample per channel. Regions use `[start, end)` seconds on the aligned overlap, rounded to frames. Reports retain original metadata, transformations, starts and excluded prefixes/tails. Unmatched tails and missing default channel pairs count as differences. Explicit pairs and mixing intentionally choose a subset or derived signal.

## Numeric modes

Normalized float64 is default. Float32 mode quantizes input before double-precision subtraction. PCM16/24/32 round and clamp to signed integer codes, then subtract in int64, preserving PCM32's ±4294967295 difference range. PCM modes quantize samples rather than reinterpret file bytes. I/O and statistics retain double precision.

For `d = B - A`, metrics include maximum absolute difference and first index, MAE, RMSE, Pearson correlation, SNR `10 log10(sum(A²)/sum(d²))`, and count/fraction of samples with `abs(d) > threshold`. Threshold defaults to zero in the chosen numeric units. Constants have undefined correlation. Zero reference energy has undefined SNR; zero error with nonzero reference energy has infinite SNR. JSON uses `null` with `snr_status` instead of nonstandard Infinity/NaN.

Empty, unreadable, changed-during-read and non-finite inputs fail explicitly. Batch failures retain successful rows. Passing thresholds does not establish byte identity or perceptual equivalence, especially after preprocessing or quantization.

## Spectra and playback

Welch spectra average Hann-window periodograms over the selected interval (default FFT 2048/hop 512). One-sided PSD is in numeric-unit²/Hz, shown in dB/Hz. Spectrograms show Hann-window amplitude in dB relative to one numeric unit, sampled at up to 512 positions on the STFT grid. Long-region overviews can omit transients; select a shorter interval for detail. Nyquist follows the actual rate: 8 kHz audio ends at 4 kHz.

Playback is prepared on a background thread and delivered to Qt in bounded chunks. PCM values are normalized; device rate/format conversion and output clipping do not modify analysis. A selected channel is duplicated across output channels. Volume starts at 50%; difference amplification is not automatic. Original-input playback uses original source seconds, processed playback uses the aligned timeline.

## Reports

JSON schema 1 includes application version, UTC export time and `results`. Successful rows contain `a`, `b`, `options`, `status`, `comparison_domain`, `resampled_b`, `analysis_samplerate`, `alignment`, `frames_compared`, start samples, `excluded` and `metrics`. Folder rows include relative names and pairing paths; failures include status and available error details.

CSV uses UTF-8 with BOM and one row per paired channel. Its `metadata_json` column preserves the complete row, including parameters. Cancellation retains completed rows and marks remaining matched rows cancelled. Missing files, channel coverage and unmatched tails are never silently counted as equality.
