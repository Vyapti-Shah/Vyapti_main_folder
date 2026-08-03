"""Structured representation of a single frame-to-frame camera motion."""

from dataclasses import dataclass


@dataclass
class Motion:
    dx: float
    dy: float
    angle: float = 0.0

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.dx, self.dy, self.angle)
