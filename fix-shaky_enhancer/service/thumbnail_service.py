"""Service responsible for writing enhanced thumbnail images to disk."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ThumbnailService:
    """Handles writing enhanced thumbnail images to disk."""

    def save_thumbnail(self, output_path: str, image: np.ndarray) -> None:
        """Saves an image to disk as a thumbnail."""
        cv2.imwrite(output_path, image)
        logger.info("Saved enhanced thumbnail to: %s", output_path)
