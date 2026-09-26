"""Versioned, standards-compliant reports shared by GUI and CLI."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from . import get_version


def document(rows: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "application_version": get_version(),
        "created_utc": datetime.now(UTC).isoformat(),
        "results": rows,
    }


def export_json(path: str | Path, rows: list[dict]):
    Path(path).write_text(
        json.dumps(document(rows), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )


def export_csv(path: str | Path, rows: list[dict]):
    fields = [
        "name",
        "status",
        "a_path",
        "b_path",
        "a_channel",
        "b_channel",
        "analysis_samplerate",
        "frames_compared",
        "max_abs",
        "mae",
        "rmse",
        "correlation",
        "snr_db",
        "above_threshold_ratio",
        "error",
        "metadata_json",
    ]
    with Path(path).open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for metric in row.get("metrics", [{}]):
                flat = {key: row.get(key, "") for key in fields}
                flat.update({key: value for key, value in metric.items() if key in fields})
                flat["metadata_json"] = json.dumps(row, ensure_ascii=False, allow_nan=False)
                writer.writerow(flat)


def exit_code(rows: list[dict]) -> int:
    statuses = {row["status"] for row in rows}
    if "cancelled" in statuses:
        return 130
    if not rows or statuses & {"error", "collision", "pending"}:
        return 2
    if statuses & {"different", "missing"}:
        return 1
    return 0
