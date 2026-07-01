"""CLAHE-based local contrast plus a global linear contrast/brightness boost."""

import cv2
import numpy as np

from video_enhancer.config import EnhancementConfig


class ContrastEnhancer:
    """Applies Contrast Limited Adaptive Histogram Equalization and a global boost."""

    def __init__(
        self,
        clip_limit: float = EnhancementConfig.CLAHE_CLIP,
        grid_size: tuple = EnhancementConfig.CLAHE_GRID,
        alpha: float = EnhancementConfig.CONTRAST_ALPHA,
        beta: float = EnhancementConfig.CONTRAST_BETA,
    ) -> None:
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        self._alpha = alpha
        self._beta = beta

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        cl = self._clahe.apply(l_channel)
        limg = cv2.merge((cl, a_channel, b_channel))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def boost_contrast(self, frame: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(frame, alpha=self._alpha, beta=self._beta)
