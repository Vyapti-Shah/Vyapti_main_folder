"""Repository layer encapsulating all direct video I/O via OpenCV."""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoRepository:
    """Encapsulates all direct interaction with cv2.VideoCapture."""

    def __init__(self, video_path: str) -> None:
        self._video_path = video_path
        self._capture = cv2.VideoCapture(video_path)

        if not self._capture.isOpened():
            raise IOError(f"Unable to open video file: {video_path}")

    def get_total_frames(self) -> int:
        """Returns the total number of frames in the video."""
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def set_frame_position(self, frame_number: int) -> None:
        """Moves the read pointer to the given frame index."""
        self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Reads the next frame from the video."""
        return self._capture.read()

    def read_frame_range(self, start_frame: int, end_frame: int) -> list[np.ndarray]:
        """Reads and returns all frames within [start_frame, end_frame)."""
        self.set_frame_position(start_frame)
        frames: list[np.ndarray] = []

        for _ in range(start_frame, end_frame):
            ret, frame = self.read_frame()
            if not ret:
                break
            frames.append(frame)

        return frames

    def release(self) -> None:
        """Releases the underlying video capture resource."""
        self._capture.release()

    def __enter__(self) -> "VideoRepository":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
