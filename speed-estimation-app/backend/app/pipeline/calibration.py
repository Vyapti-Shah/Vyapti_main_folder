"""Ground-plane calibration via homography.

Maps image pixel coordinates (on the flat surface) to real-world
metric coordinates, and reports a reprojection RMSE so the user can
see how trustworthy the calibration itself is.
"""
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Calibration:
    H: np.ndarray  # 3x3 homography, image px -> world m
    rmse_px: float  # reprojection error of the calibration points themselves

    def image_to_world(self, pts_px: np.ndarray) -> np.ndarray:
        """pts_px: (N,2) array of (x,y) image points -> (N,2) world (X,Y) meters."""
        pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
        world = cv2.perspectiveTransform(pts, self.H)
        return world.reshape(-1, 2)


def compute_homography(
    image_points: List[Tuple[float, float]],
    world_points: List[Tuple[float, float]],
) -> Calibration:
    if len(image_points) != len(world_points):
        raise ValueError("image_points and world_points must have the same length")
    if len(image_points) < 4:
        raise ValueError("Need at least 4 point correspondences for a homography")

    src = np.array(image_points, dtype=np.float64)
    dst = np.array(world_points, dtype=np.float64)

    if len(image_points) == 4:
        H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
        H = H.astype(np.float64)
    else:
        H, _ = cv2.findHomography(src, dst, method=0)

    # Reprojection error in *image* space: map world back with H^-1 and compare to src.
    # This tells the user how self-consistent their clicked points are.
    H_inv = np.linalg.inv(H)
    reprojected = cv2.perspectiveTransform(
        dst.reshape(-1, 1, 2).astype(np.float64), H_inv
    ).reshape(-1, 2)
    errs = np.linalg.norm(reprojected - src, axis=1)
    rmse_px = float(np.sqrt(np.mean(errs**2)))

    return Calibration(H=H, rmse_px=rmse_px)
