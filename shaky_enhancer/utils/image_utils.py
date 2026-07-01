"""Frame-level image operations shared across services."""

import cv2
import numpy as np


def resize_pair(
    frame1: np.ndarray,
    frame2: np.ndarray,
    max_width: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Downscale a pair of frames to max_width (if wider) for faster flow
    inference. Returns the (possibly unchanged) frames plus the scale
    factor applied, so callers can rescale flow vectors back up.
    """
    h, w = frame1.shape[:2]
    if w <= max_width:
        return frame1, frame2, 1.0

    scale = max_width / w
    new_w, new_h = max_width, int(h * scale)
    frame1_small = cv2.resize(frame1, (new_w, new_h))
    frame2_small = cv2.resize(frame2, (new_w, new_h))
    return frame1_small, frame2_small, scale


def fix_border(frame: np.ndarray, crop_ratio: float) -> np.ndarray:
    """
    Apply a very slight zoom-in so the edges shifted by stabilization
    warping never expose black borders. BORDER_REPLICATE smears edge
    pixels outward as a final safety net.
    """
    h, w = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 0, crop_ratio)
    return cv2.warpAffine(frame, rotation_matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)
