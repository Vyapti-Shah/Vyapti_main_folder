"""Follow zoom: given a box the operator drag-selected on one frame, render a
NEW video that keeps the ORIGINAL unzoomed footage as the main view — with a
red rectangle showing exactly where the zoom window currently is — and burns a
small top-right picture-in-picture of the smoothly zoomed crop that follows the
object as SAM2 (sam2_tracker.py) tracks it through the WHOLE clip.

The main view is the original on purpose: it is the evidence, and an analyst
watching needs the context around the subject (who else is in frame, what the
subject is walking toward) that a tight crop throws away. The zoom is the
annotation, so it goes in the corner. Both halves are letterboxed rather than
stretched — see _fit().

Adapted from a standalone reference CLI tool the operator supplied (click-once
SAM2 tracking + Kalman/SMA-smoothed camera path + crop/PiP compositor); the
algorithm here is the same shape, rewritten as an on-demand appliance service
in this project's own conventions (see routers/follow_zoom.py for the job
lifecycle, mirroring routers/supervision.py).

Two smoothers are supported, selected via config.follow_zoom_smoother:
  - "kalman" (default): constant-velocity Kalman filter. During an occlusion
    gap (SAM2 returns no mask that frame) it keeps extrapolating the camera
    path from the last known velocity instead of holding position, which
    reads far more naturally on a moving subject.
  - "sma": a simple moving average, fully deterministic, no motion model.
`filterpy` (the reference code's Kalman dependency) is deliberately NOT added
to requirements.txt for this — a constant-velocity 2D Kalman filter is ~40
lines of plain numpy, and filterpy is a near-unmaintained package for
something this small and fully ownable.
"""
from __future__ import annotations

import logging
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from ..config import get_settings
from . import sam2_tracker

log = logging.getLogger("vision.follow_zoom")

BBox = tuple[float, float, float, float]


@dataclass
class FollowZoomRenderResult:
    output_path: str
    frame_count: int = 0
    fps: float = 0.0
    frames_tracked: int = 0
    frames_occluded: int = 0
    zoom_w: int = 0
    zoom_h: int = 0


# --------------------------------------------------------------------------- #
# Smoothing — gap-filled camera path
# --------------------------------------------------------------------------- #

class _SMASmoother:
    def __init__(self, window: int = 12) -> None:
        self.xs: deque[float] = deque(maxlen=window)
        self.ys: deque[float] = deque(maxlen=window)
        self._last: tuple[float, float] | None = None

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        self.xs.append(cx)
        self.ys.append(cy)
        self._last = (float(np.mean(self.xs)), float(np.mean(self.ys)))
        return self._last

    def predict(self) -> tuple[float, float]:
        return self._last if self._last is not None else (0.0, 0.0)


class _KalmanSmoother:
    """Constant-velocity Kalman filter over (cx, cy). State: [cx, cy, vx, vy].
    `predict()` without a matching `update()` (an occlusion gap) keeps
    extrapolating position from the last known velocity rather than holding
    still — this is the whole reason to prefer this over a moving average."""

    def __init__(self, process_noise: float = 1.0, measurement_noise: float = 8.0) -> None:
        self._x = np.zeros(4, dtype=np.float64)          # [cx, cy, vx, vy]
        self._p = np.eye(4) * 500.0
        self._f = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
        self._h = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        self._q = np.eye(4) * process_noise
        self._r = np.eye(2) * measurement_noise
        self._initialized = False

    def _predict_step(self) -> None:
        self._x = self._f @ self._x
        self._p = self._f @ self._p @ self._f.T + self._q

    def update(self, cx: float, cy: float) -> tuple[float, float]:
        if not self._initialized:
            self._x[:2] = (cx, cy)
            self._initialized = True
        else:
            self._predict_step()
            z = np.array([cx, cy], dtype=np.float64)
            y = z - self._h @ self._x
            sk = self._h @ self._p @ self._h.T + self._r
            k = self._p @ self._h.T @ np.linalg.inv(sk)
            self._x = self._x + k @ y
            self._p = (np.eye(4) - k @ self._h) @ self._p
        return float(self._x[0]), float(self._x[1])

    def predict(self) -> tuple[float, float]:
        if not self._initialized:
            return 0.0, 0.0
        self._predict_step()
        return float(self._x[0]), float(self._x[1])


def _smoother_factory():
    kind = get_settings().follow_zoom_smoother
    return (lambda: _SMASmoother()) if kind == "sma" else (lambda: _KalmanSmoother())


def _bbox_center(b: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = b
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _fill_gaps_with_prediction(
    trajectory: dict[int, BBox | None], total_frames: int,
    max_predict_frames: int | None = None,
) -> tuple[list[tuple[float, float]], set[int]]:
    """The per-frame camera path, with occlusion gaps filled.

    Extrapolation is BOUNDED. The Kalman smoother happily coasts on the last
    known velocity forever, so a long occlusion used to send the centre off the
    frame; the crop then clamped to an edge and stayed there, which looks like
    the zoom abandoning the object for the rest of the video. After
    `max_predict_frames` the path holds the last confident position instead —
    and stops advancing the filter, so a reacquisition is not fighting a state
    that has drifted hundreds of pixels away.
    """
    limit = (max_predict_frames if max_predict_frames is not None
             else get_settings().follow_zoom_max_predict_frames)
    smoother = _smoother_factory()()
    centers: list[tuple[float, float]] = []
    occluded: set[int] = set()
    gap = 0
    last_known: tuple[float, float] | None = None
    for i in range(total_frames):
        bbox = trajectory.get(i)
        if bbox is not None:
            gap = 0
            last_known = smoother.update(*_bbox_center(bbox))
            centers.append(last_known)
            continue
        occluded.add(i)
        gap += 1
        if gap <= limit:
            last_known = smoother.predict()
        # else: hold `last_known` — deliberately without calling predict(), so
        # the filter's state stops running away from where the object was.
        centers.append(last_known if last_known is not None else smoother.predict())
    return centers, occluded


def _compute_zoom_size(
    seed_box_px: BBox, *, padding_scale: float, min_w: int, min_h: int,
) -> tuple[int, int]:
    """The zoom window, taken from the box the operator actually dragged.

    It used to be derived from the LARGEST box SAM2 produced anywhere in the
    clip, scaled up by a padding factor and floored at 240x180 — three separate
    reasons for the rendered zoom to be wider than what was selected, and no
    relationship to the drag at all. Someone boxing a face got a zoom on the
    whole torso and wondered why.

    Now the drag IS the boundary. `padding_scale` defaults to 1.0 so this is a
    literal match; it stays as an escape hatch for anyone who wants breathing
    room without a code change. The floors are a degenerate-crop guard only
    (a stray click can produce a 2px box), not a framing decision.

    Fixed for the whole clip on purpose: a window that resized as the object
    moved toward the camera would change the PiP's magnification frame to
    frame, which is unwatchable. The window slides to follow, never resizes —
    see the clamp in _render_raw.
    """
    x1, y1, x2, y2 = seed_box_px
    w = max(0.0, x2 - x1) * padding_scale
    h = max(0.0, y2 - y1) * padding_scale
    return max(min_w, int(round(w))), max(min_h, int(round(h)))


# --------------------------------------------------------------------------- #
# Render — original main view + zoomed PiP composite
# --------------------------------------------------------------------------- #

def _fit(img, box_w: int, box_h: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Scale `img` to fit inside box_w x box_h WITHOUT distorting it, centred
    on black. Returns (canvas, (ox, oy, sw, sh)) so callers can map source
    coordinates onto the letterboxed placement.

    This exists because the two things being composited almost never share the
    output canvas's aspect ratio: the zoom window is sized from the tracked
    object (a 4:3 400x300 floor by default) while the canvas is 16:9. A plain
    `cv2.resize` to the box stretches whatever it is given, and a stretched
    face or number plate is exactly the thing this feature is supposed to make
    identifiable. Letterboxing costs some black margin and keeps the geometry
    true.
    """
    h, w = img.shape[:2]
    canvas = np.zeros((box_h, box_w, 3), dtype=np.uint8)
    if w <= 0 or h <= 0 or box_w <= 0 or box_h <= 0:
        return canvas, (0, 0, box_w, box_h)
    scale = min(box_w / w, box_h / h)
    sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (sw, sh), interpolation=interp)
    ox, oy = (box_w - sw) // 2, (box_h - sh) // 2
    canvas[oy:oy + sh, ox:ox + sw] = resized
    return canvas, (ox, oy, sw, sh)


def _pip_box(zoom_w: int, zoom_h: int, output_w: int, output_h: int,
             pip_scale: float) -> tuple[int, int]:
    """PiP dimensions, taken from the ZOOM WINDOW's aspect ratio rather than
    the source frame's.

    The PiP now carries the zoomed crop, so sizing it from the source frame
    would letterbox a 4:3 crop into a 16:9 slot and waste most of it. Deriving
    it from zoom_w:zoom_h means the whole zoom window fills the PiP. Constant
    for the clip (zoom_w/zoom_h are computed once), because a PiP that resized
    frame to frame would be unwatchable. Height is capped so a very tall zoom
    box cannot run down the entire output.
    """
    pip_w = max(1, int(output_w * pip_scale))
    pip_h = max(1, int(round(pip_w * zoom_h / max(1, zoom_w))))
    max_h = max(1, int(output_h * 0.55))
    if pip_h > max_h:
        pip_w = max(1, int(round(pip_w * max_h / pip_h)))
        pip_h = max_h
    return pip_w, pip_h


def _render_raw(
    video_path: str, raw_path: Path, centers: list[tuple[float, float]],
    zoom_w: int, zoom_h: int, *, output_w: int, output_h: int, pip_scale: float,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    show_window_box: bool = True,
) -> tuple[int, float]:
    from .render_control import check

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(centers)

    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"),
                              fps, (output_w, output_h))
    half_w, half_h = zoom_w / 2.0, zoom_h / 2.0
    pip_margin = 20
    pip_w, pip_h = _pip_box(zoom_w, zoom_h, output_w, output_h, pip_scale)
    frame_idx = 0
    try:
        while True:
            # Checked unconditionally, not through the progress hook below:
            # that one is guarded by a known frame count, so on a video whose
            # container reports none it never fires and cancel never landed.
            check(should_cancel)
            ret, frame = cap.read()
            if not ret:
                break
            orig_h, orig_w = frame.shape[:2]
            cx, cy = centers[frame_idx] if frame_idx < len(centers) else (
                centers[-1] if centers else (orig_w / 2.0, orig_h / 2.0))

            x1 = int(round(cx - half_w)); y1 = int(round(cy - half_h))
            x2 = int(round(cx + half_w)); y2 = int(round(cy + half_h))
            # Shift the window (not shrink it) at frame edges, so the crop
            # never resizes/distorts — it just slides to stay in bounds.
            if x1 < 0:
                x2 -= x1; x1 = 0
            if y1 < 0:
                y2 -= y1; y1 = 0
            if x2 > orig_w:
                x1 -= (x2 - orig_w); x2 = orig_w
            if y2 > orig_h:
                y1 -= (y2 - orig_h); y2 = orig_h
            x1 = max(0, x1); y1 = max(0, y1)

            crop = frame[y1:y2, x1:x2]

            # MAIN VIEW = the original, unzoomed frame, letterboxed so it keeps
            # its true proportions on the output canvas.
            main_view, (mox, moy, msw, msh) = _fit(frame, output_w, output_h)
            # The outline marking what the PiP is showing. Source pixel coords
            # have to be mapped through the letterbox placement — using them
            # raw would put the box in the wrong place on any clip whose aspect
            # ratio differs from the output canvas.
            if show_window_box:
                sx, sy = msw / max(1, orig_w), msh / max(1, orig_h)
                cv2.rectangle(
                    main_view,
                    (mox + int(round(x1 * sx)), moy + int(round(y1 * sy))),
                    (mox + int(round(x2 * sx)), moy + int(round(y2 * sy))),
                    (0, 0, 255), 2)

            # PiP = the ZOOMED crop, fitted whole and undistorted. A crop can
            # come back empty if the window fell entirely outside the frame;
            # showing the full frame there beats writing a black hole.
            pip_source = frame if crop.size == 0 else crop
            pip_view, _ = _fit(pip_source, pip_w, pip_h)
            cv2.rectangle(pip_view, (0, 0), (pip_w - 1, pip_h - 1), (255, 255, 255), 2)

            y_end = min(pip_margin + pip_h, output_h)
            x_start = max(0, output_w - pip_w - pip_margin)
            x_end = min(output_w, x_start + pip_w)
            main_view[pip_margin:y_end, x_start:x_end] = pip_view[:y_end - pip_margin, :x_end - x_start]

            writer.write(main_view)
            frame_idx += 1
            if progress and total:
                progress(frame_idx / total)
    finally:
        cap.release()
        writer.release()
    return frame_idx, fps


def _mux_with_audio(raw_path: Path, source_path: str, output_path: Path,
                    should_cancel: Callable[[], bool] | None = None) -> None:
    """cv2.VideoWriter's raw output has no audio track at all — unlike
    supervision_tracking.py's burned-in video (fine for a forensic boxes pass,
    a bad surprise for "watch my video zoomed in"), remux the ORIGINAL
    source's audio back in alongside the re-encoded, browser-playable video."""
    from .render_control import run_cancellable

    # A full H.264 re-encode of the finished video: minutes on a long clip, and
    # it reports nothing, so it runs killable rather than making a cancel wait.
    run_cancellable(
        ["ffmpeg", "-y", "-i", str(raw_path), "-i", source_path,
         "-map", "0:v", "-map", "1:a?",
         "-c:v", "libx264", "-c:a", "aac", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(output_path)],
        should_cancel,
    )


def render_follow_zoom(
    video_id: str, video_path: str, output_path: Path, *,
    seed_frame_idx: int, seed_box_px: BBox,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> FollowZoomRenderResult:
    from .render_control import check

    s = get_settings()

    raw = sam2_tracker.track_bidirectional(
        video_id, video_path, seed_frame_idx=seed_frame_idx, seed_box_px=seed_box_px,
        progress=(lambda f: progress and progress(f * 0.6)),
        should_cancel=should_cancel,
    )
    if not raw:
        raise RuntimeError("SAM2 lost the object on every frame — nothing to render")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (max(raw) + 1)
    cap.release()

    check(should_cancel)
    centers, occluded = _fill_gaps_with_prediction(
        raw, total_frames, s.follow_zoom_max_predict_frames)
    zoom_w, zoom_h = _compute_zoom_size(
        seed_box_px, padding_scale=s.follow_zoom_padding_scale,
        min_w=s.follow_zoom_min_zoom_w, min_h=s.follow_zoom_min_zoom_h)

    raw_path = output_path.with_suffix(".raw.mp4")
    try:
        frame_count, fps = _render_raw(
            video_path, raw_path, centers, zoom_w, zoom_h,
            output_w=s.follow_zoom_output_w, output_h=s.follow_zoom_output_h,
            pip_scale=s.follow_zoom_pip_scale,
            progress=(lambda f: progress and progress(0.6 + f * 0.3)),
            should_cancel=should_cancel,
            show_window_box=s.follow_zoom_show_window_box)
        if progress:
            progress(0.95)
        _mux_with_audio(raw_path, video_path, output_path, should_cancel)
    finally:
        raw_path.unlink(missing_ok=True)

    frames_tracked = sum(1 for v in raw.values() if v is not None)
    return FollowZoomRenderResult(
        output_path=str(output_path), frame_count=frame_count, fps=fps,
        frames_tracked=frames_tracked, frames_occluded=len(occluded),
        zoom_w=zoom_w, zoom_h=zoom_h,
    )
