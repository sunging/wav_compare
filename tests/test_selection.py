from decimal import Decimal

import numpy as np
import pytest
from PySide6 import QtCore

from wav_compare.engine import compare
from wav_compare.models import Options
from wav_compare.ui.app import Window
from wav_compare.ui.selection import (
    FORMATS,
    MODES,
    SampleSelection,
    SelectionEditor,
    format_value,
    parse_value,
)


@pytest.mark.parametrize("rate", [8000, 44100, 48000, 192000])
@pytest.mark.parametrize("value_format", FORMATS)
def test_formats_round_trip_samples_and_half_samples(rate, value_format):
    for sample in (
        Decimal(0),
        Decimal(1),
        Decimal("0.5"),
        Decimal("999.5"),
        Decimal(2**35) + Decimal("0.5"),
        Decimal(3600 * 49 * rate + 1),
    ):
        text = format_value(sample, value_format, rate)
        assert parse_value(text, value_format, rate, center=True) == sample
    if value_format == "clock":
        assert format_value(Decimal(3600 * 49 * rate), value_format, rate).startswith("49:00:00.")


@pytest.mark.parametrize(
    "text,value_format,center",
    [
        ("", "seconds", False),
        ("NaN", "seconds", False),
        ("1e999", "seconds", False),
        ("-1", "seconds", False),
        ("00:60:00", "clock", False),
        ("00:00:60", "clock", False),
        ("00:01:", "clock", False),
        ("3.5", "samples", False),
        ("3.25", "samples", True),
        ("1:2", "seconds", False),
    ],
)
def test_invalid_values(text, value_format, center):
    with pytest.raises(ValueError):
        parse_value(text, value_format, 44100, center)


@pytest.mark.parametrize(
    "mode,index,value,expected",
    [
        ("start_end", 0, 50, (39, 40)),
        ("start_end", 1, 10, (20, 21)),
        ("start_end", 0, -10, (0, 40)),
        ("start_end", 1, 200, (20, 100)),
        ("start_length", 0, 200, (80, 100)),
        ("start_length", 0, -20, (0, 20)),
        ("start_length", 1, 200, (20, 100)),
        ("start_length", 1, 0, (20, 21)),
        ("length_end", 1, 0, (0, 20)),
        ("length_end", 1, 200, (80, 100)),
        ("length_end", 0, 200, (0, 40)),
        ("length_end", 0, 0, (39, 40)),
        ("center_length", 0, 0, (0, 20)),
        ("center_length", 0, 100, (80, 100)),
        ("center_length", 1, 100, (0, 60)),
        ("center_length", 1, 19, (20, 39)),
        ("center_length", 1, 21, (20, 41)),
        ("center_length", 0, "30.5", (20, 40)),
    ],
)
def test_edit_constraints(mode, index, value, expected):
    state = SampleSelection(8000, 100, 20, 40)
    state.edit(mode, index, Decimal(value))
    assert state.bounds == expected


@pytest.mark.parametrize("mode", MODES)
def test_one_sample_and_large_sample_positions(mode):
    state = SampleSelection(192000, 1, 0, 1)
    for i in range(2):
        for value in (Decimal(0), Decimal("0.5"), Decimal(100)):
            state.edit(mode, i, value)
            assert state.bounds == (0, 1)
    state.set_context(192000, 2**35)
    state.set_bounds(2**34, 2**34 + 123)
    assert state.values("start_length") == (Decimal(2**34), Decimal(123))
    assert state.values("center_length")[0] == Decimal(2**34) + Decimal("61.5")
    for i, value in enumerate(state.values(mode)):
        state.edit(mode, i, value)
    assert state.bounds == (2**34, 2**34 + 123)


@pytest.fixture
def editor(qtbot):
    editor = SelectionEditor(lambda text: text)
    qtbot.addWidget(editor)
    editor.set_context(44100, 441000)
    editor.set_selection(441, 88201)
    editor.resize(1000, 80)
    editor.show()
    return editor


def type_value(qtbot, field, text):
    field.setFocus()
    field.lineEdit().selectAll()
    qtbot.keyClicks(field.lineEdit(), text)


def test_switches_preserve_selection_without_edit_signals(editor, qtbot):
    edited = []
    editor.selectionEdited.connect(lambda *args: edited.append(args))
    for _ in range(3):
        for mode in MODES:
            for value_format in FORMATS:
                editor.set_preferences(value_format, mode)
                qtbot.wait(1)
                assert editor.bounds == (441, 88201)
                for field in editor.fields:
                    field.commit()
    assert not edited


def test_text_commit_enter_blur_invalid_restore_and_steps(editor, qtbot):
    editor.set_preferences("samples", "start_end")
    field = editor.fields[0]
    type_value(qtbot, field, "1000")
    assert editor.bounds[0] == 441
    qtbot.keyClick(field, QtCore.Qt.Key_Return)
    assert editor.bounds[0] == 1000
    type_value(qtbot, field, "2000")
    editor.fields[1].setFocus()
    qtbot.wait(10)
    assert editor.bounds[0] == 2000
    messages = []
    editor.message.connect(messages.append)
    type_value(qtbot, field, "invalid")
    qtbot.keyClick(field, QtCore.Qt.Key_Return)
    assert editor.bounds[0] == 2000 and field.lineEdit().text() == "2000"
    assert messages == ["Invalid selection value; restored."]
    qtbot.keyClick(field, QtCore.Qt.Key_Up)
    assert editor.bounds[0] == 2001
    qtbot.keyClick(field, QtCore.Qt.Key_Down)
    assert editor.bounds[0] == 2000
    editor.set_preferences("clock", "start_end")
    type_value(qtbot, field, "00:00:01.000000")
    qtbot.keyClick(field, QtCore.Qt.Key_Return)
    assert editor.bounds[0] == 44100


def test_widget_half_center_and_signals_above_32_bits(editor, qtbot):
    editor.set_context(192000, 2**35)
    editor.set_selection(2**34, 2**34 + 3)
    editor.set_preferences("samples", "center_length")
    assert editor.fields[0].lineEdit().text() == f"{2**34 + 1}.5"
    emitted = []
    editor.selectionEdited.connect(lambda *args: emitted.append(args))
    type_value(qtbot, editor.fields[0], f"{2**34 + 10}.5")
    qtbot.keyClick(editor.fields[0], QtCore.Qt.Key_Return)
    assert editor.bounds == (2**34 + 9, 2**34 + 12)
    assert emitted == [editor.bounds]
    editor.set_context(1, 0)
    assert not any(field.isEnabled() for field in editor.fields)
    assert editor.format_combo.isEnabled() and editor.mode_combo.isEnabled()


@pytest.fixture
def window(qtbot, audio, tmp_path):
    settings = QtCore.QSettings(str(tmp_path / "selection.ini"), QtCore.QSettings.IniFormat)
    window = Window(settings=settings)
    path = audio("tone.wav", np.sin(np.arange(4410) * 0.1), rate=44100)
    window.set_result(compare(path, path, Options(strict=True)))
    window.show()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    yield window
    window.close()
    qtbot.waitUntil(lambda: not window.jobs.jobs, timeout=5000)
    window.deleteLater()


def test_drag_snap_sync_and_no_computation_on_format_change(window, qtbot, monkeypatch):
    window.selection_preview(100.4 / 44100, 200.6 / 44100)
    assert window.selection_editor.bounds == (100, 201)
    assert window.seek.selection == window.region.getRegion() == (100 / 44100, 201 / 44100)
    window.region.setRegion((123.7 / 44100, 309.1 / 44100))
    assert window.selection_editor.bounds == (124, 309)
    calls = []
    monkeypatch.setattr(window.jobs, "submit", lambda *args: calls.append(args))
    for mode in MODES:
        for value_format in FORMATS:
            window.selection_editor.set_preferences(value_format, mode)
    assert window.selection_editor.bounds == (124, 309)
    assert not calls
    window.selection_editor.set_preferences("samples", "start_length")
    type_value(qtbot, window.selection_editor.fields[0], "200")
    qtbot.keyClick(window.selection_editor.fields[0], QtCore.Qt.Key_Return)
    assert window.selection_editor.bounds == (200, 385)
    assert window.seek.selection == (200 / 44100, 385 / 44100)
    assert not calls


def test_nested_region_uses_sample_origin(window, qtbot, audio):
    path = audio("region.wav", np.zeros(4410), rate=44100)
    result = compare(path, path, Options(strict=True, region=(0.010001, 0.080001)))
    window.set_result(result)
    window.selection_preview(100 / 44100, 501 / 44100)
    window.selection_editor.set_preferences("samples", "center_length")
    window.analyze_region(True)
    qtbot.waitUntil(lambda: window.result is not result, timeout=5000)
    assert window.result.frames == 401
    assert window.result.options.region == (541 / 44100, 942 / 44100)
    result = window.result
    window.selection_preview(1 / 44100, 3 / 44100)
    window.analyze_region(True)
    qtbot.waitUntil(lambda: window.result is not result, timeout=5000)
    assert window.result.frames == 2
    assert window.result.options.region == (542 / 44100, 544 / 44100)


def test_resampled_input_uses_analysis_rate(window, qtbot, audio):
    a = audio("a8k.wav", np.zeros(800), rate=8000)
    b = audio("b48k.wav", np.zeros(4800), rate=48000)
    window.set_result(compare(a, b, Options(align=False)))
    assert window.result.rate == window.selection_editor.state.rate == 8000
    window.selection_preview(0.0125, 0.025)
    window.selection_editor.set_preferences("samples", "start_length")
    assert [field.lineEdit().text() for field in window.selection_editor.fields] == ["100", "100"]
    assert window.result.report["resampled_b"]


def test_spectrum_uses_boundaries_in_center_length_sample_display(window, qtbot):
    window.selection_preview(101 / 44100, 402 / 44100)
    window.selection_editor.set_preferences("samples", "center_length")
    window.tabs.setCurrentIndex(1)
    qtbot.waitUntil(lambda: window.spectral_ranges is not None, timeout=5000)
    assert window.spectrogram_plot.getViewBox().state["limits"]["xLimits"] == [
        101 / 44100,
        402 / 44100,
    ]


def test_format_preferences_saved_without_selection(window, qtbot):
    window.selection_editor.set_preferences("samples", "length_end")
    window.selection_preview(0.01, 0.02)
    window.save_preferences()
    restored = Window(settings=window.settings)
    qtbot.addWidget(restored)
    assert (restored.selection_editor.value_format, restored.selection_editor.mode) == (
        "samples",
        "length_end",
    )
    assert restored.selection_editor.bounds == (0, 0)
    restored.close()
    window.settings.setValue("selection/format", ["invalid", "type"])
    window.settings.setValue("selection/mode", "unknown")
    restored = Window(settings=window.settings)
    qtbot.addWidget(restored)
    assert (restored.selection_editor.value_format, restored.selection_editor.mode) == (
        "seconds",
        "start_end",
    )
    restored.close()


@pytest.mark.parametrize("language", ["en", "zh"])
@pytest.mark.parametrize("size", [(1000, 680), (1440, 940)])
def test_layout_wraps_without_clipping(window, qtbot, language, size):
    window.language_combo.setCurrentIndex(window.language_combo.findData(language))
    window.resize(*size)
    editor = window.selection_editor
    # Include large counts / >24h text without creating a multi-day audio fixture.
    editor.set_context(192000, 2**35)
    editor.set_selection(2**35 - 3, 2**35)
    for collapsed in (False, True):
        if collapsed:
            for action in window.panel_actions:
                action.trigger()
        for value_format in FORMATS:
            for mode in MODES:
                editor.set_preferences(value_format, mode)
                qtbot.wait(10)
                assert window.width() == size[0]
                assert editor.values_panel.width() >= editor.values_panel.minimumSizeHint().width()
                for field in editor.fields:
                    assert field.width() >= field.minimumSizeHint().width()
                    assert field.lineEdit().fontMetrics().horizontalAdvance(field.rendered) <= (
                        field.lineEdit().width() - 2
                    )
                    assert editor.rect().contains(field.mapTo(editor, field.rect().bottomRight()))
                if editor.wrapped:
                    assert editor.values_panel.y() > editor.selectors.y()
                else:
                    assert editor.values_panel.x() > editor.selectors.x()
                assert all(
                    label.width() >= label.minimumSizeHint().width() for label in editor.labels
                )
    assert not editor.wrapped
