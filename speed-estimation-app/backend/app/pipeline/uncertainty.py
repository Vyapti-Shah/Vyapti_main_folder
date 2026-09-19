"""Monte Carlo uncertainty propagation for a track's mean speed.

Sources of error modeled:
  - detection/bbox jitter (per-frame ground-point pixel noise)
  - calibration error (homography reprojection RMSE, applied as noise
    on the *world* coordinates after transform)
  - timestamp/FPS uncertainty (per-frame dt noise)

We resample the whole trajectory N times, recompute total distance /
elapsed time each time, and report the mean, std and 95% CI of the
resulting speed distribution.
"""
from dataclasses import dataclass

import numpy as np

from app.config import MC_SAMPLES
from app.pipeline.calibration import Calibration


@dataclass
class UncertaintyResult:
    mean_speed_kmh: float
    std_kmh: float
    ci95_low_kmh: float
    ci95_high_kmh: float
    n_samples: int


def estimate_uncertainty(
    ground_px: np.ndarray,  # (N,2) raw pixel ground points, in original (unsmoothed) order
    time_s: np.ndarray,     # (N,)
    calib: Calibration,
    bbox_jitter_px: float,
    calib_jitter_px: float,
    timestamp_jitter_s: float,
    n_samples: int = MC_SAMPLES,
) -> UncertaintyResult:
    rng = np.random.default_rng(42)
    n = len(ground_px)
    if n < 2:
        return UncertaintyResult(0.0, 0.0, 0.0, 0.0, 0)

    # Approximate the effect of calibration-point error by converting the
    # homography's pixel-space reprojection RMSE into an equivalent extra
    # pixel-noise term (added in quadrature to detection jitter).
    combined_px_std = float(np.sqrt(bbox_jitter_px**2 + calib_jitter_px**2))

    speeds = np.empty(n_samples)
    for s in range(n_samples):
        px_noisy = ground_px + rng.normal(0, combined_px_std, size=ground_px.shape)
        t_noisy = time_s + rng.normal(0, timestamp_jitter_s, size=time_s.shape)
        t_noisy = np.sort(t_noisy)  # keep temporal order sane

        world_noisy = calib.image_to_world(px_noisy)
        d = float(np.linalg.norm(world_noisy[-1] - world_noisy[0]))
        dt = max(float(t_noisy[-1] - t_noisy[0]), 1e-6)
        speeds[s] = d / dt * 3.6

    mean = float(np.mean(speeds))
    std = float(np.std(speeds))
    lo, hi = np.percentile(speeds, [2.5, 97.5])
    return UncertaintyResult(
        mean_speed_kmh=mean,
        std_kmh=std,
        ci95_low_kmh=float(lo),
        ci95_high_kmh=float(hi),
        n_samples=n_samples,
    )


def estimate_uncertainty_auto(
    point_estimate_kmh: float,
    mean_relative_size_std: float,
    bbox_jitter_px: float,
    typical_pixel_dim: float,
    timestamp_jitter_s: float,
    typical_dt_s: float,
    n_samples: int = MC_SAMPLES,
) -> UncertaintyResult:
    """Uncertainty for automatic (no-calibration-point) mode.

    Dominant term is almost always the class-size variance (a real car's
    length isn't exactly the population average) — this is typically much
    larger than the manual-calibration error, and that's reflected here
    rather than hidden. Detection-pixel jitter and timestamp jitter are
    folded in the same way as the manual path.
    """
    rng = np.random.default_rng(7)

    # relative error on the scale itself (meters/pixel) from class-size variance
    scale_rel_err = rng.normal(0, mean_relative_size_std, size=n_samples)
    # relative error on the apparent pixel size from detection jitter (affects the
    # scale estimate too, since scale = known_size / pixel_dim)
    pixel_dim_rel_err = rng.normal(0, bbox_jitter_px / max(typical_pixel_dim, 1e-6), size=n_samples)
    # timestamp jitter as a relative error on dt
    dt_rel_err = rng.normal(0, timestamp_jitter_s / max(typical_dt_s, 1e-6), size=n_samples)

    # speed = distance/time, distance ~ scale * pixels, scale ~ known_size/pixel_dim
    # so speed's relative error combines all three roughly independently (first order)
    combined_rel_err = scale_rel_err - pixel_dim_rel_err - dt_rel_err
    speeds = point_estimate_kmh * (1.0 + combined_rel_err)
    speeds = np.clip(speeds, 0, None)

    mean = float(np.mean(speeds))
    std = float(np.std(speeds))
    lo, hi = np.percentile(speeds, [2.5, 97.5])
    return UncertaintyResult(
        mean_speed_kmh=mean,
        std_kmh=std,
        ci95_low_kmh=float(lo),
        ci95_high_kmh=float(hi),
        n_samples=n_samples,
    )
