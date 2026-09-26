import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.*=false")

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
