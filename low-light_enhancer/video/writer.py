"""Compiles a sequence of frame images back into a video file."""

import logging
from pathlib import Path

import cv2

from config import DEFAULT_FPS

logger = logging.getLogger(__name__)


class VideoWriter:
    """Reassembles frames from a directory into an output video, preserving FPS."""

    def compile_video(self, frames_dir: str, output_path: str, original_video_path: str) -> None:
        cap = cv2.VideoCapture(original_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = DEFAULT_FPS
        cap.release()

        frame_files = sorted(Path(frames_dir).glob("*.png"))
        if not frame_files:
            logger.warning("No frames found in %s; nothing to compile.", frames_dir)
            return

        first_frame = cv2.imread(str(frame_files[0]))
        height, width, _ = first_frame.shape

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        try:
            for frame_file in frame_files:
                out.write(cv2.imread(str(frame_file)))
        finally:
            out.release()

        logger.info("Compiled %d frames into %s", len(frame_files), output_path)
