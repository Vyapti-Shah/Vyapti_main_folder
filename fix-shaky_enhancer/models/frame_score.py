"""Data model representing a single frame's sharpness score."""

from dataclasses import dataclass


@dataclass
class FrameScore:
    """Represents the sharpness score of a single frame."""

    frame_number: int
    sharpness: float
