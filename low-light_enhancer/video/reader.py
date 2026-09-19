"""Reads frames out of a video file onto disk."""

import logging
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)


class VideoReader:
    """Extracts all frames from a video into a target directory as PNG files."""

    def extract_frames(self, video_path: str, output_dir: str) -> None:
        cap = cv2.VideoCapture(video_path)
        idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imwrite(str(Path(output_dir) / f"{idx:06d}.png"), frame)
                idx += 1
        finally:
            cap.release()
        logger.info("Extracted %d frames from %s", idx, video_path)
