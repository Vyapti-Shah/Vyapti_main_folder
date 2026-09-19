"""Service responsible for detecting scenes within a video."""

import logging
import sys

try:
    from scenedetect import ContentDetector, detect
except ImportError:
    print("Error: PySceneDetect is not installed. Please run: pip install scenedetect")
    sys.exit(1)

import config as settings
from models import Scene
from repository.video_repository import VideoRepository

logger = logging.getLogger(__name__)


class SceneService:
    """Handles scene detection for a video."""

    def detect_scenes(
        self,
        video_path: str,
        threshold: float = settings.SCENE_THRESHOLD,
    ) -> list[Scene]:
        """Detects scenes in a video using PySceneDetect."""
        logger.info("Detecting scenes (this may take a moment)...")
        scene_list = detect(video_path, ContentDetector(threshold=threshold))
        logger.info("Detected %d scenes in video.", len(scene_list))

        return [
            Scene(start_frame=start.frame_num, end_frame=end.frame_num)
            for start, end in scene_list
        ]

    def get_total_frames(self, repository: VideoRepository) -> int:
        """Returns the total number of frames available in a video."""
        return repository.get_total_frames()
