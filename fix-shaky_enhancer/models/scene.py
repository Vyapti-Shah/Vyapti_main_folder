"""Data model representing a detected video scene."""

from dataclasses import dataclass


@dataclass
class Scene:
    """Represents a detected scene as a frame range."""

    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        """Returns the number of frames spanned by this scene."""
        return self.end_frame - self.start_frame
