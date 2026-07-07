"""Service responsible for scoring and comparing frame sharpness."""

import logging

import cv2
import numpy as np

from models.frame_score import FrameScore
from models.scene import Scene
from service.video_repository import VideoRepository
from service.image_utils import convert_gray

logger = logging.getLogger(__name__)


class SharpnessService:
    """Computes and compares frame sharpness using Variance of Laplacian."""

    def variance_of_laplacian(self, image: np.ndarray) -> float:
        """Computes the sharpness of an image using Variance of Laplacian."""
        gray = convert_gray(image)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def find_best_frame(
        self,
        repository: VideoRepository,
        scene: Scene,
    ) -> tuple[FrameScore, list[FrameScore]]:
        """Finds the sharpest frame within a scene.

        Returns the best FrameScore along with the scores for every
        frame examined in the scene.
        """
        repository.set_frame_position(scene.start_frame)

        all_scores: list[FrameScore] = []
        best_score = FrameScore(frame_number=scene.start_frame, sharpness=-1.0)

        for frame_number in range(scene.start_frame, scene.end_frame):
            ret, frame = repository.read_frame()
            if not ret:
                break

            sharpness = self.variance_of_laplacian(frame)
            score = FrameScore(frame_number=frame_number, sharpness=sharpness)
            all_scores.append(score)

            if sharpness > best_score.sharpness:
                best_score = score

        return best_score, all_scores
