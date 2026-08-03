"""Service responsible for aligning and averaging a window of frames."""

import logging

import cv2
import numpy as np

import config as settings
from utils import clip_to_uint8, convert_gray

logger = logging.getLogger(__name__)


class AlignmentService:
    """Aligns a set of frames to a reference frame and averages them."""

    def align_and_average(
        self,
        frames: list[np.ndarray],
        reference_index: int,
    ) -> tuple[np.ndarray, int]:
        """Align frames to the reference frame using ECC and return the
        averaged image along with the number of successfully aligned frames.
        """
        reference_frame = frames[reference_index]
        ref_gray = convert_gray(reference_frame)

        aligned_frames = [reference_frame.astype(np.float32)]

        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            settings.ECC_MAX_ITERATIONS,
            settings.ECC_EPSILON,
        )

        for index, frame in enumerate(frames):
            if index == reference_index:
                continue

            aligned_frame = self._align_frame(frame, ref_gray, reference_frame, criteria)
            if aligned_frame is not None:
                aligned_frames.append(aligned_frame)

        return self._average_frames(aligned_frames, reference_frame.shape), len(aligned_frames)

    def _align_frame(
        self,
        frame: np.ndarray,
        ref_gray: np.ndarray,
        reference_frame: np.ndarray,
        criteria: tuple,
    ):
        """Aligns a single frame to the reference frame. Returns None on failure."""
        frame_gray = convert_gray(frame)
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            _, warp_matrix = cv2.findTransformECC(
                ref_gray, frame_gray, warp_matrix, cv2.MOTION_AFFINE, criteria, None, 5
            )
            aligned = cv2.warpAffine(
                frame,
                warp_matrix,
                (reference_frame.shape[1], reference_frame.shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned.astype(np.float32)
        except cv2.error:
            logger.warning("ECC alignment failed for a frame; skipping.")
            return None

    def _average_frames(self, frames: list[np.ndarray], shape: tuple) -> np.ndarray:
        """Averages a list of frames and clips the result to a valid image."""
        avg_frame = np.zeros(shape, dtype=np.float32)
        for frame in frames:
            avg_frame += frame
        avg_frame /= len(frames)

        return clip_to_uint8(avg_frame)

    def extract_window(
        self,
        best_frame_number: int,
        scene_start: int,
        scene_end: int,
        window_size: int,
    ) -> tuple[int, int]:
        """Computes the [start, end) frame range centered on the best frame."""
        half_window = window_size // 2
        start_window = max(scene_start, best_frame_number - half_window)
        end_window = min(scene_end, best_frame_number + half_window + 1)
        return start_window, end_window
