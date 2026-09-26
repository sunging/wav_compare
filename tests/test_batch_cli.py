import json
import subprocess
import sys

import numpy as np

from wav_compare.batch import discover, run_batch
from wav_compare.models import Cancellation, Options


def test_batch_partial_failures_and_missing(audio):
    audio("a/sub/One.wav", np.zeros(30))
    audio("b/sub/one.WAV", np.zeros(30))
    a = audio("a/missing.wav", np.zeros(30)).parent
    b = audio("b/bad.wav", np.zeros(30)).parent
    (a / "bad.wav").write_bytes(b"broken")
    rows = run_batch(discover(str(a), str(b)), Options(strict=True), Cancellation())
    assert {r["status"] for r in rows} == {"within_threshold", "missing", "error"}


def test_cancellation_keeps_rows(audio):
    a, b = audio("a/one.wav", [0.0, 0.0]).parent, audio("b/one.wav", [0.0, 0.0]).parent
    rows = discover(str(a), str(b))
    cancel = Cancellation()
    cancel.cancel()
    assert run_batch(rows, Options(), cancel)[0]["status"] == "cancelled"


def test_headless_cli(audio, tmp_path):
    a, b = audio("a.wav", [0.0, 0.1]), audio("b.wav", [0.0, 0.2])
    report = tmp_path / "out.json"
    run = subprocess.run(
        [
            sys.executable,
            "-m",
            "wav_compare",
            "compare",
            str(a),
            str(b),
            "--strict",
            "--json",
            str(report),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 1, run.stderr
    assert json.loads(report.read_text())["results"][0]["status"] == "different"
    script = "from wav_compare.cli import parser; import sys; parser(); assert 'PySide6' not in sys.modules"
    subprocess.run([sys.executable, "-c", script], check=True)
