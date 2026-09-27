"""Visible synthetic desktop/Qt audio smoke test; optional screenshot artifact."""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6 import QtCore, QtMultimedia, QtWidgets

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui.app import Window
from wav_compare.ui.selection import FORMATS, MODES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=8)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--screenshot", default=".artifacts/workbench.png")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=940)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--theme", choices=("light", "dark"), default="dark")
    parser.add_argument("--collapse-panels", action="store_true")
    parser.add_argument("--selection-format", choices=FORMATS, default="seconds")
    parser.add_argument("--selection-mode", choices=MODES, default="start_end")
    args = parser.parse_args()
    root = Path(".artifacts/demo")
    root.mkdir(parents=True, exist_ok=True)
    t = np.arange(80000) / 8000
    a = 0.12 * np.sin(2 * np.pi * 220 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    a += 0.04 * np.random.default_rng(8).normal(size=len(t))
    b = a.copy()
    b[25000:26000] += 0.025 * np.sin(2 * np.pi * 1400 * t[:1000])
    sf.write(root / "reference.wav", np.column_stack((a, a * 0.8)), 8000, subtype="PCM_24")
    sf.write(root / "candidate.wav", np.column_stack((b, b * 0.8)), 8000, subtype="PCM_24")
    app = QtWidgets.QApplication([])
    app.setStyle("Fusion")
    settings = QtCore.QSettings(str(root / "settings.ini"), QtCore.QSettings.IniFormat)
    settings.clear()
    settings.setValue("automation/startup", False)
    window = Window((str(root / "reference.wav"), str(root / "candidate.wav")), settings)
    window.resize(args.width, args.height)
    window.language_combo.setCurrentIndex(window.language_combo.findData(args.language))
    window.theme_combo.setCurrentIndex(window.theme_combo.findData(args.theme))
    with QtCore.QSignalBlocker(window.strict):
        window.strict.setChecked(True)
    result = compare(root / "reference.wav", root / "candidate.wav", Options(strict=True))
    window.set_result(result)
    window.selection_editor.set_preferences(args.selection_format, args.selection_mode)
    window.rows = [
        {
            "name": "demo / reference ↔ candidate",
            "a_path": result.a_path,
            "b_path": result.b_path,
            **result.report,
        }
    ]
    window.populate_rows()
    window.show()
    if args.collapse_panels:
        for action in window.panel_actions:
            action.trigger()
    errors, positions = [], []
    window.player.error.connect(errors.append)
    window.player.position.connect(positions.append)
    devices = [d.description() for d in QtMultimedia.QMediaDevices.audioOutputs()]
    print(json.dumps({"audio_devices": devices}, ensure_ascii=False), flush=True)
    if args.play:
        QtCore.QTimer.singleShot(1000, window.play)

    def finish():
        target = Path(args.screenshot)
        target.parent.mkdir(parents=True, exist_ok=True)
        window.grab().save(str(target))
        print(
            json.dumps(
                {
                    "errors": errors,
                    "played_until": max(positions, default=0),
                    "screenshot": str(target.resolve()),
                    "window_size": [window.width(), window.height()],
                    "minimum_size_hint": [
                        window.minimumSizeHint().width(),
                        window.minimumSizeHint().height(),
                    ],
                    "plot_size": [window.tabs.width(), window.tabs.height()],
                    "selection_wrapped": window.selection_editor.wrapped,
                    "selection_samples": window.selection_editor.bounds,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        window.close()

    QtCore.QTimer.singleShot(args.seconds * 1000, finish)
    app.exec()


if __name__ == "__main__":
    main()
