"""Data models for the application."""

from dataclasses import dataclass


@dataclass
class FrameScore:
    """Represents the sharpness score of a single frame."""

    frame_number: int
    sharpness: float


@dataclass
class Scene:
    """Represents a detected scene as a frame range."""

    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        """Returns the number of frames spanned by this scene."""
        return self.end_frame - self.start_frame
