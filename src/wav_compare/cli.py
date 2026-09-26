"""CLI parsing intentionally does not import Qt or initialize audio devices."""

from __future__ import annotations

import argparse
import signal
import sys

from . import get_version
from .models import Cancellation, Options


def parser():
    root = argparse.ArgumentParser(prog="wav-compare")
    root.add_argument("--version", action="version", version=f"wav-compare {get_version()}")
    commands = root.add_subparsers(dest="command")
    gui = commands.add_parser("gui", help="Open the desktop workbench")
    gui.add_argument("paths", nargs="*")
    command = commands.add_parser("compare", help="Compare two WAV files or folders")
    command.add_argument("a")
    command.add_argument("b")
    command.add_argument("--strict", action="store_true")
    command.add_argument("--no-align", action="store_true")
    command.add_argument("--max-lag", type=float, default=5.0)
    command.add_argument(
        "--offset", type=int, help="Manual lag in A-rate samples; positive means B lags"
    )
    command.add_argument("--threshold", type=float, default=0.0)
    command.add_argument("--segment-size", type=int, default=1000)
    command.add_argument(
        "--mode", choices=["float64", "float32", "pcm16", "pcm24", "pcm32"], default="float64"
    )
    command.add_argument("--channels", help="1-based pairs, e.g. 1:1,2:2")
    command.add_argument("--mix", action="store_true")
    command.add_argument("--region", nargs=2, type=float, metavar=("START", "END"))
    command.add_argument("--json", dest="json_path")
    command.add_argument("--csv", dest="csv_path")
    return root


def channel_pairs(text: str):
    if not text.strip():
        return ()
    pairs = tuple(tuple(int(v) - 1 for v in item.split(":")) for item in text.split(","))
    if any(len(p) != 2 or min(p) < 0 for p in pairs):
        raise ValueError("Use 1-based channel pairs, e.g. 1:1,2:2")
    return pairs


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in (None, "gui"):
        from .ui.app import launch

        paths = getattr(args, "paths", [])
        if len(paths) not in (0, 2):
            parser().error("GUI accepts either zero or two paths")
        return launch(paths)
    import json

    from .batch import discover, run_batch
    from .reports import document, exit_code, export_csv, export_json

    cancel = Cancellation()
    original = signal.signal(signal.SIGINT, lambda *_: cancel.cancel())
    try:
        options = Options(
            strict=args.strict,
            align=not args.no_align,
            max_lag=args.max_lag,
            offset=args.offset,
            threshold=args.threshold,
            segment_size=args.segment_size,
            mode=args.mode,
            pairs=channel_pairs(args.channels or ""),
            mix=args.mix,
            region=tuple(args.region) if args.region else None,
        )
        options.validate()
        rows = run_batch(discover(args.a, args.b, cancel), options, cancel)
        if args.json_path:
            export_json(args.json_path, rows)
        if args.csv_path:
            export_csv(args.csv_path, rows)
        print(json.dumps(document(rows), ensure_ascii=False, allow_nan=False))
        return exit_code(rows)
    except Exception as error:
        from .models import Cancelled

        print(str(error), file=sys.stderr)
        return 130 if isinstance(error, Cancelled) else 2
    finally:
        signal.signal(signal.SIGINT, original)
