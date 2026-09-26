"""WAV Compare: shared audio analysis engine and desktop workbench."""

from importlib.metadata import version


def get_version() -> str:
    return version("wav-compare")
