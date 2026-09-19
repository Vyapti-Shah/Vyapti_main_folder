"""
Contract for trajectory-smoothing strategies.

Today only a moving-average smoother is implemented, but this interface
lets you drop in a Kalman filter or spline-based smoother later without
touching StabilizationService.
"""

from abc import ABC, abstractmethod

import numpy as np


class TrajectorySmoother(ABC):
    """Base class for camera-trajectory smoothing algorithms."""

    @abstractmethod
    def smooth(self, trajectory: np.ndarray, radius: int) -> np.ndarray:
        """
        Args:
            trajectory: Array of shape [N, 3] — cumulative (dx, dy, angle).
            radius: Strength/window of the smoothing.

        Returns:
            A smoothed copy of the same shape.
        """
