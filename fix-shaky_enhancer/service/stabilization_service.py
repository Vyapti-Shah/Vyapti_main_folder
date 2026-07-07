"""
Orchestrates the full stabilization pipeline:

    _initialize -> _estimate_camera_motion -> _smooth_trajectory
                -> _apply_warp -> _finalize

Each step stays small and single-purpose so it's easy to test and modify.
"""

import numpy as np

from service.settings import settings
from service.raft_model import RAFTModel
from service.motion_service import MotionService
from service.raft_service import RaftService
from service.smoothing_service import MovingAverageSmoother
from service.warp_service import WarpService
from service.logger import get_logger
from service.video_utils import VideoReader, VideoWriter

logger = get_logger(__name__)


class StabilizationService:
    def __init__(self, radius: int = settings.DEFAULT_RADIUS):
        self.radius = radius
        self.raft_service = RaftService(RAFTModel(settings.LOCAL_MODEL_PATH))
        self.motion_service = MotionService()
        self.smoother = MovingAverageSmoother()
        self.warp_service = WarpService()

    def process_video(self, input_path: str, output_path: str) -> None:
        import time
        start_time = time.time()
        logger.info("Starting stabilization for: %s", input_path)

        self._initialize()
        transforms, valid_frames, video_meta = self._estimate_camera_motion(input_path)
        transforms_smooth = self._smooth_trajectory(transforms)
        self._apply_warp(input_path, output_path, transforms_smooth, valid_frames, video_meta)
        
        processing_time = time.time() - start_time
        self._finalize(output_path, transforms, transforms_smooth, valid_frames, processing_time)

    # -- pipeline steps ----------------------------------------------------

    def _initialize(self) -> None:
        self.raft_service.load_model()

    def _estimate_camera_motion(self, input_path: str):
        with VideoReader(input_path) as reader:
            n_frames = reader.frame_count
            video_meta = (reader.fps, reader.width, reader.height)

            ret, prev = reader.read()
            if not ret:
                raise IOError(f"Empty or unreadable video: {input_path}")

            transforms = np.zeros((n_frames - 1, 3), np.float32)
            valid_frames = 0

            logger.info("Step 1/3: Estimating camera motion for %d frames...", n_frames)
            for i in range(n_frames - 1):
                ret, curr = reader.read()
                if not ret:
                    break

                u, v = self.raft_service.estimate_motion(prev, curr)
                motion = self.motion_service.extract_camera_motion(u, v)
                transforms[i] = motion.as_tuple()

                prev = curr
                valid_frames += 1

                if (i + 1) % settings.PROGRESS_LOG_INTERVAL == 0:
                    logger.info("  -> Processed %d/%d frames", i + 1, n_frames)

        return transforms, valid_frames, video_meta

    def _smooth_trajectory(self, transforms: np.ndarray) -> np.ndarray:
        logger.info("Step 2/3: Smoothing trajectory...")
        trajectory = np.cumsum(transforms, axis=0)
        smoothed_trajectory = self.smoother.smooth(trajectory, self.radius)
        return smoothed_trajectory - trajectory

    def _apply_warp(
        self,
        input_path: str,
        output_path: str,
        transforms_smooth: np.ndarray,
        valid_frames: int,
        video_meta: tuple[float, int, int],
    ) -> None:
        fps, width, height = video_meta

        logger.info("Step 3/3: Warping frames...")

        with VideoReader(input_path) as reader, VideoWriter(output_path, fps, (width, height)) as writer:
            ret, frame = reader.read()
            if ret:
                writer.write(self.warp_service.fix_first_frame(frame))

            for i in range(valid_frames):
                ret, frame = reader.read()
                if not ret:
                    break

                dx, dy = transforms_smooth[i, 0], transforms_smooth[i, 1]
                writer.write(self.warp_service.stabilize_frame(frame, dx, dy))

                if (i + 1) % settings.PROGRESS_LOG_INTERVAL == 0:
                    logger.info("  -> Warped %d/%d frames", i + 1, valid_frames)

    def _finalize(self, output_path: str, transforms: np.ndarray, transforms_smooth: np.ndarray, valid_frames: int, processing_time: float) -> None:
        logger.info("Stabilization complete! Saved to %s", output_path)
        logger.info("=== Stabilization Metrics ===")
        logger.info("Total Processing Time: %.2f seconds", processing_time)
        logger.info("Total Frames Processed: %d", valid_frames)
        
        if valid_frames > 0:
            max_raw_dx = float(np.max(np.abs(transforms[:, 0])))
            max_raw_dy = float(np.max(np.abs(transforms[:, 1])))
            avg_corr_dx = float(np.mean(np.abs(transforms_smooth[:, 0])))
            avg_corr_dy = float(np.mean(np.abs(transforms_smooth[:, 1])))
            max_corr_dx = float(np.max(np.abs(transforms_smooth[:, 0])))
            max_corr_dy = float(np.max(np.abs(transforms_smooth[:, 1])))

            logger.info("Maximum Raw Shake (Frame-to-Frame): dx=%.2f, dy=%.2f", max_raw_dx, max_raw_dy)
            logger.info("Average Correction Applied: dx=%.2f, dy=%.2f", avg_corr_dx, avg_corr_dy)
            logger.info("Maximum Correction Applied: dx=%.2f, dy=%.2f", max_corr_dx, max_corr_dy)
