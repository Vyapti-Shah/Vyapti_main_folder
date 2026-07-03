"""
Smooths the camera's cumulative trajectory so only high-frequency shake
is removed while intentional camera movement (panning, walking) is kept.
"""

import numpy as np

from service.smoother import TrajectorySmoother


class MovingAverageSmoother(TrajectorySmoother):
    def smooth(self, trajectory: np.ndarray, radius: int) -> np.ndarray:
        if radius < 0:
            # -1 or any negative value completely locks the camera (tripod effect).
            # We return a zero trajectory, so smoothed_trajectory - trajectory = -trajectory,
            # which cancels out all camera motion.
            return np.zeros_like(trajectory)

        window_size = 2 * radius + 1
        kernel = np.ones(window_size) / window_size

        smoothed = np.copy(trajectory)
        for axis in range(trajectory.shape[1]):
            curr_smoothed = trajectory[:, axis]
            # Apply moving average 3 times to approximate a Gaussian filter
            # for much stronger, cinematic stabilization.
            for _ in range(3):
                padded = np.pad(curr_smoothed, (radius, radius), mode="edge")
                curr_smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius]
            smoothed[:, axis] = curr_smoothed

        return smoothed
