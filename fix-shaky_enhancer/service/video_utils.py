"""
Context-manager wrappers around cv2.VideoCapture / cv2.VideoWriter so
resources are always released automatically, even on error, replacing
manual cap.release() / out.release() calls scattered through the code.
"""

from pathlib import Path

import cv2


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap: cv2.VideoCapture | None = None

    def __enter__(self) -> "VideoReader":
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {self.path}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.cap is not None:
            self.cap.release()

    @property
    def frame_count(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        return self.cap.read()

    def reset(self) -> None:
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


class VideoWriter:
    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self.path = path
        self.fps = fps
        self.size = size
        self.writer: cv2.VideoWriter | None = None

    def __enter__(self) -> "VideoWriter":
        ext = Path(self.path).suffix.lower()
        fourcc_str = "mp4v" if ext == ".mp4" else "MJPG"
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, self.size)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.writer is not None:
            self.writer.release()

    def write(self, frame) -> None:
        self.writer.write(frame)
