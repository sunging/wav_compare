"""Deterministic folder matching, including collisions and missing files."""

from __future__ import annotations

import os
from pathlib import Path

from .engine import compare
from .models import Cancellation, Cancelled, Options, no_progress


def discover(a: str, b: str, cancel: Cancellation | None = None) -> list[dict]:
    cancel = cancel or Cancellation()
    pa, pb = Path(a), Path(b)
    if pa.is_file() and pb.is_file():
        return [{"name": pa.name, "a_path": str(pa), "b_path": str(pb), "status": "pending"}]
    if not pa.is_dir() or not pb.is_dir():
        raise ValueError("Choose two files or two folders")

    def index(root):
        mapping = {}

        def failed(error):
            raise error

        for parent, dirs, files in os.walk(root, followlinks=False, onerror=failed):
            cancel.check()
            dirs[:] = sorted(d for d in dirs if not Path(parent, d).is_symlink())
            for name in sorted(files):
                if Path(name).suffix.lower() not in (".wav", ".rf64", ".w64"):
                    continue
                path = Path(parent, name)
                key = path.relative_to(root).as_posix().casefold()
                mapping.setdefault(key, []).append(str(path))
        return mapping

    left, right = index(pa), index(pb)
    rows = []
    for key in sorted(left.keys() | right.keys()):
        la, lb = left.get(key, []), right.get(key, [])
        status = (
            "collision"
            if len(la) > 1 or len(lb) > 1
            else "missing"
            if not la or not lb
            else "pending"
        )
        rows.append(
            {
                "name": key,
                "a_path": la[0] if la else None,
                "b_path": lb[0] if lb else None,
                "status": status,
                "a_candidates": la,
                "b_candidates": lb,
            }
        )
    return rows


def run_batch(
    rows: list[dict],
    options: Options,
    cancel: Cancellation,
    progress=no_progress,
    on_result=lambda row: None,
) -> list[dict]:
    output = []
    for i, row in enumerate(rows):
        result_row = dict(row)
        if row["status"] not in ("missing", "collision"):
            if cancel.event.is_set():
                result_row["status"] = "cancelled"
            else:
                try:
                    result = compare(
                        row["a_path"],
                        row["b_path"],
                        options,
                        cancel,
                        lambda p, s, index=i: progress((index + p) / max(1, len(rows)), s),
                        details=False,
                    )
                    result_row.update(result.report)
                    result.temporary.cleanup()
                except Cancelled:
                    result_row["status"] = "cancelled"
                except Exception as error:
                    result_row.update(status="error", error=str(error))
        output.append(result_row)
        on_result(result_row)
    return output
