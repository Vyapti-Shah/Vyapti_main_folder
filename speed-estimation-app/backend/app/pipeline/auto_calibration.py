"""Automatic pixel->meter calibration using known average real-world sizes
of common object classes — no user-provided reference points needed.

This is the standard fallback technique for uncalibrated cameras: instead
of a homography built from measured ground-plane points, we use the
object's own apparent size in the frame (its bbox width or height) plus a
population average for that class's real-world size to derive a *local*
scale (meters per pixel) at the object's current position. Because that
scale is recomputed every frame, it naturally adapts as the object moves
closer to or further from the camera (apparent size grows/shrinks).

This is inherently less accurate than a proper ground-plane calibration:
real objects vary around the class average (a "car" can be a hatchback or
an SUV), and the technique assumes the camera is roughly side-on to the
object's reference dimension. Both effects are propagated into the speed
uncertainty rather than hidden.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

# class_name -> (dimension to use, mean real-world size in meters, std in meters)
# dimension: "width" = bbox x2-x1, "height" = bbox y2-y1
KNOWN_SIZES = {
    "person":      ("height", 1.70, 0.10),
    "bicycle":     ("width",  1.70, 0.15),   # length, viewed side-on
    "motorcycle":  ("width",  2.00, 0.20),
    "car":         ("width",  4.50, 0.45),   # typical sedan/hatchback length, side-on
    "bus":         ("width",  11.0, 1.50),
    "truck":       ("width",  7.50, 2.00),
    "train":       ("width",  20.0, 5.00),
}

# Used for any detected class not in KNOWN_SIZES. Wide std reflects the
# much weaker prior — the pipeline still runs, but flags low confidence.
FALLBACK_SIZE = ("width", 1.50, 0.60)


@dataclass
class ScaleEstimate:
    meters_per_pixel: float
    relative_std: float  # fractional uncertainty from class-size variance (e.g. 0.10 = 10%)
    known_class: bool


def estimate_scale(class_name: str, bbox_xyxy: np.ndarray) -> Optional[ScaleEstimate]:
    dim, mean_m, std_m = KNOWN_SIZES.get(class_name.lower(), FALLBACK_SIZE)
    known = class_name.lower() in KNOWN_SIZES

    x1, y1, x2, y2 = bbox_xyxy
    pixel_dim = (x2 - x1) if dim == "width" else (y2 - y1)
    if pixel_dim <= 1:
        return None

    meters_per_pixel = mean_m / pixel_dim
    relative_std = std_m / mean_m
    return ScaleEstimate(meters_per_pixel=meters_per_pixel, relative_std=relative_std, known_class=known)
