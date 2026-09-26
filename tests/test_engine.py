import json

import numpy as np
import pytest
import soundfile as sf
import soxr

from wav_compare.audio import AudioInfo, estimate_delay, resample_file
from wav_compare.engine import compare
from wav_compare.models import Cancellation, Cancelled, Options
from wav_compare.reports import document, exit_code, export_csv, export_json
from wav_compare.views import ByteCache, spectra, waveform


@pytest.mark.parametrize(
    "rate", [8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000, 192000]
)
def test_rates(audio, rate):
    data = np.random.default_rng(1).normal(0, 0.1, (1537, 2))
    a, b = audio("a.wav", data, rate), audio("b.wav", data, rate)
    result = compare(a, b, Options(strict=True, block_size=173, segment_size=99))
    assert result.report["status"] == "within_threshold"
    assert result.report["analysis_samplerate"] == rate
    assert all(m["max_abs"] == 0 for m in result.report["metrics"])
    assert result.segment_data()[:, 0, 3].sum() == len(data)
    json.dumps(document([result.report]), allow_nan=False)


@pytest.mark.parametrize("bits", [16, 24, 32])
def test_pcm_extremes_no_overflow(audio, bits):
    scale = 2 ** (bits - 1)
    a = audio("a.wav", [-1.0, (scale - 1) / scale], subtype=f"PCM_{bits}")
    b = audio("b.wav", [(scale - 1) / scale, -1.0], subtype=f"PCM_{bits}")
    result = compare(a, b, Options(strict=True, mode=f"pcm{bits}", segment_size=1))
    assert result.report["metrics"][0]["max_abs"] == 2 * scale - 1
    np.testing.assert_array_equal(result.samples(0, 2)[2][:, 0], [2 * scale - 1, -(2 * scale - 1)])


def test_streamed_statistics_and_segments(audio):
    rng = np.random.default_rng(17)
    a, b = rng.normal(0, 0.1, (19057, 3)), rng.normal(0, 0.1, (19057, 3))
    result = compare(
        audio("a.wav", a),
        audio("b.wav", b),
        Options(strict=True, segment_size=777, block_size=1531, threshold=0.05),
    )
    d = b - a
    for ch, metric in enumerate(result.report["metrics"]):
        assert metric["mae"] == pytest.approx(np.abs(d[:, ch]).mean())
        assert metric["rmse"] == pytest.approx(np.sqrt(np.mean(d[:, ch] ** 2)))
        assert metric["correlation"] == pytest.approx(np.corrcoef(a[:, ch], b[:, ch])[0, 1])
        assert metric["above_threshold"] == np.sum(np.abs(d[:, ch]) > 0.05)
    segments = result.segment_data()
    for i in range(len(segments)):
        block = d[i * 777 : (i + 1) * 777]
        np.testing.assert_allclose(segments[i, :, 0], np.max(np.abs(block), axis=0))
        np.testing.assert_allclose(
            segments[i, :, 1] / segments[i, :, 3], np.mean(np.abs(block), axis=0)
        )


@pytest.mark.parametrize(
    "source,target",
    [
        (8000, 16000),
        (16000, 8000),
        (8000, 48000),
        (48000, 8000),
        (44100, 48000),
        (48000, 44100),
        (48000, 96000),
        (96000, 48000),
    ],
)
def test_resampling_blocks_and_tail(audio, tmp_path, source, target):
    data = np.random.default_rng(3).normal(0, 0.1, (source + 137, 2))
    path = audio("input.wav", data, source)
    output = resample_file(AudioInfo.read(path), target, tmp_path, Cancellation(), block_size=777)
    actual, rate = sf.read(output, always_2d=True)
    expected = soxr.resample(data, source, target, quality="HQ")
    assert rate == target
    np.testing.assert_allclose(actual, expected, atol=2e-7)


@pytest.mark.parametrize("lag", [-711, 0, 527])
def test_delay_direction(audio, lag):
    x = np.random.default_rng(4).normal(0, 0.1, 80000)
    if lag >= 0:
        a, b = x, np.r_[np.zeros(lag), x]
    else:
        a, b = np.r_[np.zeros(-lag), x], x
    result = compare(audio("a.wav", a), audio("b.wav", b), Options(max_lag=1))
    assert result.report["alignment"]["lag_samples"] == lag
    assert result.report["metrics"][0]["max_abs"] == 0


@pytest.mark.parametrize("silence", [True, False])
def test_ambiguous_alignment(audio, silence):
    data = np.zeros(80000) if silence else 0.1 * np.sin(2 * np.pi * 440 * np.arange(80000) / 8000)
    a, b = audio("a.wav", data), audio("b.wav", data)
    result = estimate_delay(str(a), str(b), 8000, 1.0, 0, 0, Cancellation())
    assert result["status"] == "unreliable"


def test_channel_pairs_mix_lengths_and_region(audio):
    a = np.column_stack((np.arange(50) / 100, -np.arange(50) / 100))
    pa, pb = audio("a.wav", a), audio("b.wav", a[:, ::-1])
    result = compare(pa, pb, Options(strict=True, pairs=((0, 1), (1, 0))))
    assert result.report["status"] == "within_threshold"
    result = compare(pa, pb, Options(strict=True, mix=True))
    assert result.report["metrics"][0]["max_abs"] == 0
    result = compare(pa, audio("c.wav", a[:40, 0]), Options(strict=True))
    assert result.report["excluded"]["missing_channels"]
    assert result.report["excluded"]["a_after"] == 10
    result = compare(pa, pa, Options(strict=True, region=(0.001, 0.003)))
    assert result.frames == 16
    assert result.starts == (8, 8)


def test_bad_inputs(audio, tmp_path):
    normal = audio("normal.wav", np.zeros(100))
    for bad in [audio("empty.wav", []), audio("nan.wav", [0, np.nan, 0])]:
        with pytest.raises(ValueError):
            compare(normal, bad, Options(strict=True))
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"bad")
    with pytest.raises(sf.LibsndfileError):
        compare(normal, bad)
    with pytest.raises(ValueError, match="sample rates"):
        compare(normal, audio("other.wav", np.zeros(100), 16000), Options(strict=True))
    with pytest.raises(ValueError, match="No overlapping"):
        compare(normal, normal, Options(offset=200))
    with pytest.raises(ValueError):
        compare(normal, normal, Options(pairs=((2, 0),)))


def test_cancel(audio):
    path = audio("a.wav", np.ones(5000))
    cancel = Cancellation()

    def progress(*_):
        cancel.cancel()

    with pytest.raises(Cancelled):
        compare(path, path, Options(strict=True, block_size=100), cancel, progress)


def test_envelope_preserves_impulse_and_spectral_peak(audio):
    rate = 8000
    data = 0.1 * np.sin(2 * np.pi * 1000 * np.arange(80000) / rate)
    data[44001] = 0.9
    path = audio("a.wav", data)
    result = compare(path, path, Options(strict=True))
    _, a, _, d = waveform(result, 0, 10, 0, 100)
    assert a.max() == 0.9
    assert np.max(np.abs(d)) == 0
    f, psd, t, image = spectra(result, 0, 1, 0, 2048, 512, Cancellation())
    assert f[np.argmax(psd[0])] == 1000
    assert f[-1] == 4000
    assert image.shape[0] == 3
    assert len(t) <= 512


def test_cache_budget():
    cache = ByteCache(128)
    cache.put("a", (np.zeros(10),))
    cache.put("b", (np.ones(10),))
    assert cache.get("a") is None
    assert cache.size <= 128


def test_reports(audio, tmp_path):
    path = audio("a.wav", np.zeros(100))
    result = compare(path, path, Options(strict=True))
    export_json(tmp_path / "report.json", [result.report])
    export_csv(tmp_path / "report.csv", [result.report])
    doc = json.loads((tmp_path / "report.json").read_text())
    assert doc["schema_version"] == 1
    assert exit_code([result.report]) == 0
    assert exit_code([{"status": "missing"}]) == 1
    assert exit_code([{"status": "error"}]) == 2
    assert exit_code([{"status": "cancelled"}]) == 130


def test_full_five_second_search(audio):
    data = np.random.default_rng(19).normal(0, 0.1, 8000 * 18)
    a = audio("a.wav", data)
    b = audio("b.wav", np.r_[np.zeros(8000 * 4), data])
    result = compare(a, b, Options(max_lag=5))
    assert result.report["alignment"]["lag_samples"] == 32000
    assert result.report["metrics"][0]["max_abs"] == 0


def test_unusual_rate_is_not_rejected(audio):
    path = audio("a.wav", np.ones(200), 12345)
    assert compare(path, path, Options(strict=True)).rate == 12345


def test_fine_view_has_pixel_resolution(audio):
    data = np.zeros(16000)
    data[8001] = 0.7
    path = audio("a.wav", data)
    result = compare(path, path, Options(strict=True))
    x, a, _, _ = waveform(result, 0, 2, 0, 1000)
    assert len(x) >= 1000
    assert a.max() == 0.7


def test_disk_space_failure(audio, monkeypatch):
    from collections import namedtuple

    from wav_compare import audio as audio_module

    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(audio_module.shutil, "disk_usage", lambda _: usage(100, 99, 1))
    a = audio("a.wav", np.ones(100), 8000)
    b = audio("b.wav", np.ones(100), 16000)
    with pytest.raises(OSError, match="cache disk"):
        compare(a, b)
