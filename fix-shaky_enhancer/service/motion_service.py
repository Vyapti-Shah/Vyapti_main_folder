"""Extracts the dominant background (camera) motion from dense optical flow."""

import numpy as np

from service.motion import Motion


class MotionService:
    @staticmethod
    def extract_camera_motion(u: np.ndarray, v: np.ndarray) -> Motion:
        """
        The median of the dense flow field isolates the dominant background motion.
        We mask out the center region to ensure foreground subjects do not skew the median.
        """
        h, w = u.shape
        edge_y = int(0.25 * h)
        edge_x = int(0.25 * w)

        mask = np.ones((h, w), dtype=bool)
        mask[edge_y:-edge_y, edge_x:-edge_x] = False

        dx = float(np.median(u[mask]))
        dy = float(np.median(v[mask]))
        return Motion(dx=dx, dy=dy, angle=0.0)
