"""Composes the individual algorithmic enhancement steps into one frame pipeline.

The internal call order focuses only on low-light enhancement:
  1. CLAHE
  2. global contrast boost
"""

import logging

import numpy as np

from video_enhancer.processing.contrast_enhancer import ContrastEnhancer

logger = logging.getLogger(__name__)


class FrameProcessor:
    """Runs the low-light enhancement chain on a single frame."""

    def __init__(
        self,
        contrast_enhancer: ContrastEnhancer = None,
    ) -> None:
        self._contrast_enhancer = contrast_enhancer or ContrastEnhancer()

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = self._contrast_enhancer.apply_clahe(frame)
        frame = self._contrast_enhancer.boost_contrast(frame)

        return frame
