"""Orchestrates the full forensic video enhancement pipeline."""

import logging
import tempfile
from pathlib import Path

from video_enhancer.pipeline.stages import CompileStage, EnhancementStage, ExtractStage
from video_enhancer.processing.frame_processor import FrameProcessor
from video_enhancer.video.reader import VideoReader
from video_enhancer.video.writer import VideoWriter

logger = logging.getLogger(__name__)


class PipelineController:
    """Wires together stages via dependency injection and runs them in sequence."""

    def __init__(
        self,
        frame_processor: FrameProcessor = None,
        video_reader: VideoReader = None,
        video_writer: VideoWriter = None,
    ) -> None:
        self._video_reader = video_reader or VideoReader()
        self._video_writer = video_writer or VideoWriter()
        self._frame_processor = frame_processor or FrameProcessor()

        self._extract_stage = ExtractStage(self._video_reader)
        self._enhancement_stage = EnhancementStage(self._frame_processor)
        self._compile_stage = CompileStage(self._video_writer)

    def process_video(
        self,
        input_video_path: str,
        output_video_path: str,
    ) -> None:
        logger.info("--- Starting Processing for: %s ---", input_video_path)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            extracted_frames_dir = tmp_dir_path / "extracted"
            extracted_frames_dir.mkdir()

            self._extract_stage.execute(input_video_path, str(extracted_frames_dir))

            final_frames_dir = tmp_dir_path / "final"
            final_frames_dir.mkdir()

            self._enhancement_stage.execute(
                str(extracted_frames_dir), str(final_frames_dir)
            )

            self._compile_stage.execute(
                str(final_frames_dir), output_video_path, input_video_path
            )

        logger.info("Enhancement complete! Output saved to: %s", output_video_path)
