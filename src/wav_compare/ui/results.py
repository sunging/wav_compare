"""Batch result table with numeric sorting, and a structured, copyable metrics view."""

from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets

SORT_ROLE = QtCore.Qt.UserRole + 1
COLUMNS = ("Name", "Status", "Max difference", "Lag", "RMSE", "SNR (dB)")
STATUS_COLORS = {
    "different": "#e8912d",
    "within_threshold": "#2ea66f",
    "error": "#e5534b",
    "collision": "#e5534b",
    "missing": "#e5534b",
    "cancelled": "#8b949e",
}
ANALYZABLE_EXCLUDED = ("missing", "collision", "error", "cancelled")
# Readable English names for stable report keys; other languages translate the keys directly.
LABELS = {
    "a_channel": "A channel",
    "b_channel": "B channel",
    "max_abs": "Max absolute difference",
    "max_at_sample": "Largest difference at sample",
    "mae": "MAE",
    "rmse": "RMSE",
    "correlation": "Correlation",
    "snr_db": "SNR (dB)",
    "snr_status": "SNR status",
    "above_threshold": "Samples above threshold",
    "above_threshold_ratio": "Ratio above threshold",
    "a_before": "A before start",
    "b_before": "B before start",
    "a_after": "A after end",
    "b_after": "B after end",
    "missing_channels": "Unpaired channels",
    "a_path": "A path",
    "b_path": "B path",
    "a_candidates": "A candidates",
    "b_candidates": "B candidates",
}


def format_number(value):
    return "—" if value is None else f"{value:.6g}"


def row_values(row):
    """Sortable per-row summary: worst channel for errors, weakest channel for SNR."""
    metrics = row.get("metrics") or []
    maximum = max((m["max_abs"] for m in metrics), default=None)
    rmse = max((m["rmse"] for m in metrics), default=None)
    snrs = [m for m in metrics if m.get("snr_status") != "undefined"]
    if not snrs:
        snr = None
    elif all(m["snr_status"] == "infinite" for m in snrs):
        snr = float("inf")
    else:
        snr = min(m["snr_db"] for m in snrs if m["snr_db"] is not None)
    lag = (row.get("alignment") or {}).get("lag_samples") if metrics else None
    return maximum, lag, rmse, snr


class NumericItem(QtWidgets.QTableWidgetItem):
    """Sort by the stored number; rows without a value always sort last in ascending order."""

    def __lt__(self, other):
        mine, theirs = self.data(SORT_ROLE), other.data(SORT_ROLE)
        if mine is None or theirs is None:
            return theirs is None and mine is not None
        return mine < theirs


class ResultsTable(QtWidgets.QWidget):
    rowSelected = QtCore.Signal(int)

    def __init__(self, translate, parent=None):
        super().__init__(parent)
        self.translate = translate
        self.rows = []
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.filter = QtWidgets.QLineEdit()
        self.filter.setClearButtonEnabled(True)
        layout.addWidget(self.filter)
        self.table = QtWidgets.QTableWidget(0, len(COLUMNS))
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table, 1)
        self.filter.textChanged.connect(self.filter_rows)
        self.table.itemSelectionChanged.connect(self._selection_changed)

    def set_rows(self, rows):
        self.rows = list(rows)
        with QtCore.QSignalBlocker(self.table):
            self.table.setSortingEnabled(False)
            self.table.setRowCount(0)
            for index, row in enumerate(self.rows):
                self._insert(index, row)
            self.table.setSortingEnabled(True)
        self.filter_rows()

    def append_row(self, row):
        """Add one streamed row without disturbing the current selection or sort order."""
        self.rows.append(row)
        with QtCore.QSignalBlocker(self.table):
            self.table.setSortingEnabled(False)
            self._insert(len(self.rows) - 1, row)
            self.table.setSortingEnabled(True)
        self.filter_rows()

    def _insert(self, index, row):
        position = self.table.rowCount()
        self.table.insertRow(position)
        maximum, lag, rmse, snr = row_values(row)
        status = row["status"]
        snr_text = "∞" if snr == float("inf") else format_number(snr)
        cells = (
            (row["name"], None),
            (self.translate(status), None),
            (format_number(maximum), maximum),
            ("—" if lag is None else str(lag), lag),
            (format_number(rmse), rmse),
            (snr_text, snr),
        )
        tooltip = row.get("error") or "\n".join(
            str(path) for path in (row["name"], row.get("a_path"), row.get("b_path")) if path
        )
        for column, (text, number) in enumerate(cells):
            item = NumericItem(text) if column >= 2 else QtWidgets.QTableWidgetItem(text)
            item.setData(QtCore.Qt.UserRole, index)
            item.setData(SORT_ROLE, number)
            item.setToolTip(tooltip)
            if column >= 2:
                item.setTextAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            if column == 1 and status in STATUS_COLORS:
                item.setForeground(QtGui.QColor(STATUS_COLORS[status]))
            self.table.setItem(position, column, item)

    def filter_rows(self, *_):
        query = self.filter.text().casefold()
        for row in range(self.table.rowCount()):
            text = " ".join(
                self.table.item(row, column).text() for column in range(self.table.columnCount())
            )
            self.table.setRowHidden(row, query not in text.casefold())

    def select_index(self, index):
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).data(QtCore.Qt.UserRole) == index:
                self.table.selectRow(row)
                return True
        return False

    def selected_index(self):
        selected = self.table.selectedItems()
        return selected[0].data(QtCore.Qt.UserRole) if selected else None

    def _selection_changed(self):
        index = self.selected_index()
        if index is not None:
            self.rowSelected.emit(index)

    def clear(self):
        self.rows = []
        self.table.setRowCount(0)

    def retranslate(self):
        self.filter.setPlaceholderText(self.translate("Filter results…"))
        self.table.setHorizontalHeaderLabels([self.translate(text) for text in COLUMNS])
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 1)
            item.setText(self.translate(self.rows[item.data(QtCore.Qt.UserRole)]["status"]))


class MetricsView(QtWidgets.QTreeWidget):
    """Grouped two-column report view; the last content re-renders after language changes."""

    def __init__(self, translate, parent=None):
        super().__init__(parent)
        self.translate = translate
        self.content = None
        self.setColumnCount(2)
        self.setRootIsDecorated(True)
        self.setUniformRowHeights(True)
        self.setAlternatingRowColors(True)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.header().setStretchLastSection(True)
        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)
        copy = QtGui.QShortcut(QtGui.QKeySequence.Copy, self)
        copy.setContext(QtCore.Qt.WidgetShortcut)
        copy.activated.connect(lambda: self.copy(selected=True))
        self.retranslate()

    def value(self, value):
        if value is None:
            return "—"
        if isinstance(value, bool):
            return self.translate("Yes" if value else "No")
        if isinstance(value, float):
            return f"{value:.9g}"
        if isinstance(value, int):
            return f"{value:,}"
        if isinstance(value, (list, tuple)):
            return ", ".join(map(str, value)) or "—"
        return self.translate(str(value))

    def _group(self, title, pairs):
        group = QtWidgets.QTreeWidgetItem([self.translate(title), ""])
        font = group.font(0)
        font.setBold(True)
        group.setFont(0, font)
        group.setFirstColumnSpanned(True)
        self.addTopLevelItem(group)
        for key, value in pairs:
            name = self.translate(key)
            item = QtWidgets.QTreeWidgetItem([LABELS.get(key, key) if name == key else name, value])
            item.setTextAlignment(1, QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            item.setToolTip(1, value)
            group.addChild(item)
        group.setExpanded(True)

    def show_report(self, report, channel):
        self.content = ("report", report, channel)
        self.clear()
        alignment = report["alignment"]
        summary = [
            ("Status", self.value(report["status"])),
            ("Sample rate", f"{report['analysis_samplerate']:,} Hz"),
            ("Samples compared", self.value(report["frames_compared"])),
            ("Domain", self.value(report["comparison_domain"])),
            ("Resampled B", self.value(report["resampled_b"])),
            ("Alignment", self.value(alignment["status"])),
            ("Lag", f"{alignment['lag_samples']:,} {self.translate('samples')}"),
        ]
        if "confidence" in alignment:
            summary.append(("Alignment confidence", self.value(alignment["confidence"])))
        if "reason" in alignment:
            summary.append(("Alignment reason", self.value(alignment["reason"])))
        summary += [
            ("A start sample", self.value(report["a_start_sample"])),
            ("B start sample", self.value(report["b_start_sample"])),
        ]
        self._group("Summary", summary)
        metric = report["metrics"][channel]
        values = []
        for key, value in metric.items():
            text = self.value(value)
            if key == "snr_db" and value is None and metric.get("snr_status") == "infinite":
                text = "∞"
            values.append((key, text))
        self._group("Channel metrics", values)
        self._group(
            "Excluded", [(key, self.value(value)) for key, value in report["excluded"].items()]
        )
        self.resizeColumnToContents(0)

    def show_row(self, row):
        """Rows without a comparison (missing, collision) list their pairing details."""
        self.content = ("row", row)
        self.clear()
        pairs = [("Name", row["name"]), ("Status", self.value(row["status"]))]
        for key, label in (("a_path", "a_path"), ("b_path", "b_path"), ("error", "Error")):
            if row.get(key):
                pairs.append((label, str(row[key])))
        for key in ("a_candidates", "b_candidates"):
            if row.get(key):
                pairs.append((key, "\n".join(row[key])))
        self._group("Summary", pairs)
        self.resizeColumnToContents(0)

    def clear_report(self):
        self.content = None
        self.clear()

    def retranslate(self):
        self.setHeaderLabels([self.translate("Item"), self.translate("Value")])
        if self.content and self.content[0] == "report":
            self.show_report(*self.content[1:])
        elif self.content:
            self.show_row(self.content[1])

    def text(self, selected=False):
        lines = []
        for i in range(self.topLevelItemCount()):
            group = self.topLevelItem(i)
            children = [group.child(j) for j in range(group.childCount())]
            if selected:
                children = [child for child in children if child.isSelected()]
            elif children:
                lines.append(f"[{group.text(0)}]")
            lines.extend(f"{child.text(0)}\t{child.text(1)}" for child in children)
        return "\n".join(lines)

    def copy(self, selected=False):
        text = self.text(selected)
        if text:
            QtWidgets.QApplication.clipboard().setText(text)

    def _menu(self, position):
        menu = QtWidgets.QMenu(self)
        copy = menu.addAction(self.translate("Copy"), lambda: self.copy(selected=True))
        copy.setEnabled(bool(self.selectedItems()))
        menu.addAction(self.translate("Copy all"), self.copy)
        menu.exec(self.viewport().mapToGlobal(position))
