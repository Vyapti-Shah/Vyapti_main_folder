"""Windowed speed calculation from a track's world-coordinate trajectory."""
from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import savgol_filter

from app.config import SPEED_WINDOW_FRAMES
from app.pipeline.calibration import Calibration
from app.pipeline.tracker import TrackData
from app.pipeline.auto_calibration import estimate_scale


@dataclass
class SpeedSeries:
    time_s: np.ndarray
    speed_kmh: np.ndarray
    world_xy: np.ndarray  # (N,2)
    cum_distance_m: np.ndarray = None  # only populated in auto mode


def compute_speed_series(track: TrackData, calib: Calibration) -> SpeedSeries:
    obs = track.observations
    px = np.array([o.ground_px for o in obs])
    t = np.array([o.time_s for o in obs])
    world = calib.image_to_world(px)

    # Smooth the trajectory itself first (reduces detection jitter) — a
    # Savitzky-Golay filter preserves the underlying motion trend better
    # than a plain moving average.
    n = len(world)
    if n >= 7:
        wlen = min(11, n - (1 - n % 2))  # odd window <= n
        if wlen % 2 == 0:
            wlen -= 1
        wlen = max(5, wlen)
        try:
            world_smooth = np.stack(
                [savgol_filter(world[:, i], wlen, 2) for i in range(2)], axis=1
            )
        except Exception:
            world_smooth = world
    else:
        world_smooth = world

    win = min(SPEED_WINDOW_FRAMES, max(2, n - 1))
    speeds = []
    times = []
    for i in range(n - win):
        p0, p1 = world_smooth[i], world_smooth[i + win]
        t0, t1 = t[i], t[i + win]
        dt = t1 - t0
        if dt <= 0:
            continue
        d = float(np.linalg.norm(p1 - p0))
        v_ms = d / dt
        speeds.append(v_ms * 3.6)
        times.append((t0 + t1) / 2.0)

    if not speeds:
        # fall back to simple two-point estimate
        d = float(np.linalg.norm(world_smooth[-1] - world_smooth[0]))
        dt = max(t[-1] - t[0], 1e-6)
        speeds = [d / dt * 3.6]
        times = [t[0]]

    return SpeedSeries(time_s=np.array(times), speed_kmh=np.array(speeds), world_xy=world_smooth)


def compute_speed_series_auto(track: TrackData):
    """Automatic mode: no ground-plane calibration required. Uses each
    frame's known-class-size-derived local scale (meters/pixel) to convert
    the smoothed pixel path into a cumulative real-world distance, then
    applies the same sliding-window speed calc as the manual-calibration
    path. Returns (SpeedSeries, mean_relative_size_std, known_class)."""
    obs = track.observations
    px = np.array([o.ground_px for o in obs])
    bboxes = np.array([o.bbox for o in obs])
    t = np.array([o.time_s for o in obs])
    n = len(obs)

    if n >= 7:
        wlen = min(11, n - (1 - n % 2))
        if wlen % 2 == 0:
            wlen -= 1
        wlen = max(5, wlen)
        try:
            px_smooth = np.stack([savgol_filter(px[:, i], wlen, 2) for i in range(2)], axis=1)
            widths_smooth = savgol_filter(bboxes[:, 2] - bboxes[:, 0], wlen, 2)
            heights_smooth = savgol_filter(bboxes[:, 3] - bboxes[:, 1], wlen, 2)
        except Exception:
            px_smooth = px
            widths_smooth = bboxes[:, 2] - bboxes[:, 0]
            heights_smooth = bboxes[:, 3] - bboxes[:, 1]
    else:
        px_smooth = px
        widths_smooth = bboxes[:, 2] - bboxes[:, 0]
        heights_smooth = bboxes[:, 3] - bboxes[:, 1]

    scales = np.zeros(n)
    rel_stds = []
    known_flags = []
    for i in range(n):
        smoothed_bbox = np.array([
            bboxes[i, 0], bboxes[i, 1],
            bboxes[i, 0] + widths_smooth[i], bboxes[i, 1] + heights_smooth[i],
        ])
        est = estimate_scale(track.class_name, smoothed_bbox)
        if est is None:
            scales[i] = scales[i - 1] if i > 0 else 0.0
            continue
        scales[i] = est.meters_per_pixel
        rel_stds.append(est.relative_std)
        known_flags.append(est.known_class)

    mean_rel_std = float(np.mean(rel_stds)) if rel_stds else 0.3
    known_class = bool(np.mean(known_flags) > 0.5) if known_flags else False

    cumdist = np.zeros(n)
    for i in range(n - 1):
        pixel_disp = float(np.linalg.norm(px_smooth[i + 1] - px_smooth[i]))
        step_scale = (scales[i] + scales[i + 1]) / 2.0
        cumdist[i + 1] = cumdist[i] + pixel_disp * step_scale

    win = min(SPEED_WINDOW_FRAMES, max(2, n - 1))
    speeds, times = [], []
    for i in range(n - win):
        d = cumdist[i + win] - cumdist[i]
        dt = t[i + win] - t[i]
        if dt <= 0:
            continue
        speeds.append(d / dt * 3.6)
        times.append((t[i] + t[i + win]) / 2.0)

    if not speeds:
        d = cumdist[-1] - cumdist[0]
        dt = max(t[-1] - t[0], 1e-6)
        speeds = [d / dt * 3.6]
        times = [t[0]]

    ss = SpeedSeries(
        time_s=np.array(times), speed_kmh=np.array(speeds),
        world_xy=px_smooth, cum_distance_m=cumdist,
    )
    return ss, mean_rel_std, known_class


def reject_outliers_mad(speeds: np.ndarray, thresh: float = 3.5) -> np.ndarray:
    """Boolean mask of inlier samples using median-absolute-deviation."""
    if len(speeds) < 4:
        return np.ones_like(speeds, dtype=bool)
    median = np.median(speeds)
    mad = np.median(np.abs(speeds - median)) or 1e-6
    modified_z = 0.6745 * (speeds - median) / mad
    return np.abs(modified_z) < thresh
