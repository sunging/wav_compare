# Contributing

Use Python 3.14 and `uv sync --locked`. Run `uv run ruff check .`, `uv run ruff format --check .`, `uv run pytest -q` and `uv build` before a PR. Keep analysis independent of Qt; GUI and CLI share semantics.

Use Conventional Commit PR titles; see [releasing](docs/releasing.md). Tests synthesize audio rather than committing recordings. Preserve cancellation and never access widgets from worker threads. Breaking report changes need a schema-version update.

`uv run python tools/smoke_desktop.py` opens a synthetic desktop example and saves a screenshot under ignored `.artifacts`. Add `--play` only when short generated playback is desired. The tool exits after eight seconds.
