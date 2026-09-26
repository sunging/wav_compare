"""Qt-independent options, cancellation and portable report types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from threading import Event


class Cancelled(Exception):
    pass


class Cancellation:
    def __init__(self):
        self.event = Event()

    def cancel(self):
        self.event.set()

    def check(self):
        if self.event.is_set():
            raise Cancelled("Cancelled")


Progress = Callable[[float, str], None]


def no_progress(value: float, message: str):
    pass


@dataclass(frozen=True)
class Options:
    strict: bool = False
    align: bool = True
    max_lag: float = 5.0
    offset: int | None = None
    threshold: float = 0.0
    segment_size: int = 1000
    mode: str = "float64"
    pairs: tuple[tuple[int, int], ...] = ()
    mix: bool = False
    region: tuple[float, float] | None = None
    block_size: int = 65536

    def validate(self):
        import math

        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("Threshold must be finite and nonnegative")
        if not math.isfinite(self.max_lag) or not 0 <= self.max_lag <= 300:
            raise ValueError("Maximum lag must be between 0 and 300 seconds")
        if self.segment_size < 1 or self.block_size < 1:
            raise ValueError("Segment and block sizes must be positive")
        if self.mode not in ("float64", "float32", "pcm16", "pcm24", "pcm32"):
            raise ValueError("Unsupported numeric mode")
        if self.mix and self.pairs:
            raise ValueError("Mix and channel pairs are mutually exclusive")
        if self.strict and self.offset not in (None, 0):
            raise ValueError("Strict comparison does not allow a manual offset")
        if self.region and not (
            all(math.isfinite(v) for v in self.region) and 0 <= self.region[0] < self.region[1]
        ):
            raise ValueError("Region must satisfy 0 <= start < end")

    def report(self):
        return asdict(self)
