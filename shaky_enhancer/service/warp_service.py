"""Applies the smoothed translation correction to each frame."""

import numpy as np
import cv2

from config.settings import settings
from utils.image_utils import fix_border


class WarpService:
    def __init__(self, crop_ratio: float = settings.BORDER_CROP_RATIO):
        self.crop_ratio = crop_ratio

    def stabilize_frame(self, frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
        h, w = frame.shape[:2]

        transform = np.zeros((2, 3), np.float32)
        transform[0, 0] = 1.0
        transform[0, 2] = dx
        transform[1, 1] = 1.0
        transform[1, 2] = dy

        warped = cv2.warpAffine(frame, transform, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return fix_border(warped, self.crop_ratio)

    def fix_first_frame(self, frame: np.ndarray) -> np.ndarray:
        """The very first frame has no prior motion to correct — just border-fix it."""
        return fix_border(frame, self.crop_ratio)
