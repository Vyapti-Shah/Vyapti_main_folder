"""Extracts the dominant background (camera) motion from dense optical flow."""

import numpy as np

from model.motion import Motion


class MotionService:
    @staticmethod
    def extract_camera_motion(u: np.ndarray, v: np.ndarray) -> Motion:
        """
        The median of the dense flow field naturally filters out moving
        foreground objects (cars, people) and isolates the dominant
        background motion caused by camera shake.
        """
        dx = float(np.median(u))
        dy = float(np.median(v))
        return Motion(dx=dx, dy=dy, angle=0.0)
