"""Concrete pipeline stages, each independently testable via execute()."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import cv2

from video_enhancer.config import WATERMARK_TEXT
from video_enhancer.processing.frame_processor import FrameProcessor
from video_enhancer.utils.progress import ProgressTracker
from video_enhancer.video.reader import VideoReader
from video_enhancer.video.writer import VideoWriter

logger = logging.getLogger(__name__)


class PipelineStage(ABC):
    """Base class for a single, independently testable pipeline step."""

    @abstractmethod
    def execute(self, *args, **kwargs):
        raise NotImplementedError


class ExtractStage(PipelineStage):
    """Extracts frames from the video."""

    def __init__(self, video_reader: VideoReader) -> None:
        self._video_reader = video_reader

    def execute(self, input_video_path: str, output_dir: str) -> None:
        logger.info("[Stage 1] Extracting frames...")
        self._video_reader.extract_frames(input_video_path, output_dir)


class EnhancementStage(PipelineStage):
    """Applies low-light enhancement + watermark, frame by frame."""

    def __init__(self, frame_processor: FrameProcessor) -> None:
        self._frame_processor = frame_processor

    def execute(self, input_dir: str, output_dir: str) -> None:
        frame_files = sorted(Path(input_dir).glob("*.png"))
        logger.info(
            "[Stage 2] Applying Low Light Enhancement on %d frames...", len(frame_files)
        )

        for i, frame_path in enumerate(ProgressTracker(frame_files, description="Enhancing frames")):
            img = cv2.imread(str(frame_path))

            img = self._frame_processor.enhance_frame(img)

            if WATERMARK_TEXT:
                cv2.putText(
                    img, WATERMARK_TEXT, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
                )

            cv2.imwrite(str(Path(output_dir) / f"{i:06d}.png"), img)


class CompileStage(PipelineStage):
    """Reassembles enhanced frames into the final output video."""

    def __init__(self, video_writer: VideoWriter) -> None:
        self._video_writer = video_writer

    def execute(self, frames_dir: str, output_video_path: str, original_video_path: str) -> None:
        logger.info("[Stage 3] Recompiling enhanced video...")
        self._video_writer.compile_video(frames_dir, output_video_path, original_video_path)
