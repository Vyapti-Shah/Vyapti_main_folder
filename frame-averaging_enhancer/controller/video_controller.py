"""Controller that orchestrates the full video enhancement pipeline."""

import logging

import config as settings
from models import Scene
from repository.video_repository import VideoRepository
from service.alignment_service import AlignmentService
from service.graph_service import GraphService
from service.scene_service import SceneService
from service.sharpness_service import SharpnessService
from service.thumbnail_service import ThumbnailService
from utils import (
    CSV_FILENAME_TEMPLATE,
    GRAPH_FILENAME_TEMPLATE,
    THUMBNAIL_FILENAME_TEMPLATE,
    build_path,
    create_directory,
)

logger = logging.getLogger(__name__)


class VideoController:
    """Orchestrates the full video thumbnail enhancement pipeline."""

    def __init__(self) -> None:
        self._scene_service = SceneService()
        self._sharpness_service = SharpnessService()
        self._alignment_service = AlignmentService()
        self._graph_service = GraphService()
        self._thumbnail_service = ThumbnailService()

    def run(
        self,
        video_path: str,
        output_dir: str = settings.DEFAULT_OUTPUT_DIR,
        window_size: int = settings.DEFAULT_WINDOW_SIZE,
    ) -> None:
        """Runs the end-to-end pipeline for a given video."""
        graphs_dir, thumbs_dir = self._prepare_output_dirs(output_dir)

        with VideoRepository(video_path) as repository:
            scenes = self._get_scenes(video_path, repository)

            for scene_index, scene in enumerate(scenes, start=1):
                self._process_scene(
                    repository, scene, scene_index, graphs_dir, thumbs_dir, window_size
                )

        logger.info("Pipeline Complete! Check the '%s' folder.", output_dir)

    def _prepare_output_dirs(self, output_dir: str) -> tuple[str, str]:
        """Creates and returns the graphs/thumbnails output directories."""
        graphs_dir = build_path(output_dir, settings.GRAPHS_SUBDIR)
        thumbs_dir = build_path(output_dir, settings.THUMBNAILS_SUBDIR)
        create_directory(graphs_dir)
        create_directory(thumbs_dir)
        return graphs_dir, thumbs_dir

    def _get_scenes(self, video_path: str, repository: VideoRepository) -> list[Scene]:
        """Detects scenes, falling back to a single whole-video scene."""
        scenes = self._scene_service.detect_scenes(video_path)
        if not scenes:
            logger.info("No scenes detected. Treating the entire video as one scene.")
            total_frames = self._scene_service.get_total_frames(repository)
            scenes = [Scene(start_frame=0, end_frame=total_frames)]
        return scenes

    def _process_scene(
        self,
        repository: VideoRepository,
        scene: Scene,
        scene_index: int,
        graphs_dir: str,
        thumbs_dir: str,
        window_size: int,
    ) -> None:
        """Runs the full pipeline for a single scene."""
        logger.info(
            "Processing Scene %d (Frames %d-%d)",
            scene_index,
            scene.start_frame,
            scene.end_frame,
        )

        best_score, all_scores = self._sharpness_service.find_best_frame(repository, scene)
        if not all_scores:
            return

        logger.info(
            "Scene %d Best Frame: %d (Score: %.2f)",
            scene_index,
            best_score.frame_number,
            best_score.sharpness,
        )

        self._save_graph_and_csv(all_scores, best_score, scene_index, graphs_dir)
        self._save_enhanced_thumbnail(
            repository, scene, best_score, scene_index, thumbs_dir, window_size
        )

    def _save_graph_and_csv(
        self,
        all_scores,
        best_score,
        scene_index: int,
        graphs_dir: str,
    ) -> None:
        """Saves the sharpness graph and CSV export for a scene."""
        graph_path = build_path(graphs_dir, GRAPH_FILENAME_TEMPLATE.format(index=scene_index))
        self._graph_service.save_graph(all_scores, best_score, graph_path, scene_index)

        csv_path = build_path(graphs_dir, CSV_FILENAME_TEMPLATE.format(index=scene_index))
        self._graph_service.save_csv(all_scores, csv_path)

    def _save_enhanced_thumbnail(
        self,
        repository: VideoRepository,
        scene: Scene,
        best_score,
        scene_index: int,
        thumbs_dir: str,
        window_size: int,
    ) -> None:
        """Extracts a frame window, aligns/averages it, and saves the thumbnail."""
        start_window, end_window = self._alignment_service.extract_window(
            best_score.frame_number, scene.start_frame, scene.end_frame, window_size
        )

        window_frames = repository.read_frame_range(start_window, end_window)
        if not window_frames:
            return

        reference_index = best_score.frame_number - start_window

        logger.info(
            "Aligning and averaging %d frames around frame %d...",
            len(window_frames),
            best_score.frame_number,
        )
        enhanced_thumb, aligned_count = self._alignment_service.align_and_average(
            window_frames, reference_index
        )

        thumb_path = build_path(
            thumbs_dir, THUMBNAIL_FILENAME_TEMPLATE.format(index=scene_index)
        )
        self._thumbnail_service.save_thumbnail(thumb_path, enhanced_thumb)
        logger.info("%d frames averaged for scene %d thumbnail.", aligned_count, scene_index)
