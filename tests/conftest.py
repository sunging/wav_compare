import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.*=false")
if sys.platform == "win32":
    # Offscreen Qt on Windows has no font directory; layout tests need real font metrics,
    # as fontconfig already provides on Linux and macOS.
    os.environ.setdefault(
        "QT_QPA_FONTDIR", os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts")
    )

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def audio(tmp_path):
    def write(name, data, rate=8000, subtype="DOUBLE"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.asarray(data), rate, subtype=subtype)
        return path

    return write
