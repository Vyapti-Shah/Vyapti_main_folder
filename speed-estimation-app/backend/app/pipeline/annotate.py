"""Renders the annotated output video: thin green box on the tracked
object, a white ground-contact cross, a "V <id>" label, and a clean black
rounded speed badge — matching a speed-camera-style readout rather than
plain debug text. If the job used manual ground-plane calibration, the
actual calibration quadrilateral is drawn too (there's nothing real to draw
in automatic mode, since it has no measured reference points)."""
import subprocess
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.pipeline.tracker import TrackData

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_GREEN = (70, 200, 90)    # BGR accent: boxes + calibration polygon
_WHITE = (255, 255, 255)
_BADGE_BG = (18, 18, 18)  # near-black badge background


def _text_with_outline(frame, text, org, font_scale, color, thickness):
    """White text with a black outline so it stays legible over any
    background (road, sky, headlights, ...)."""
    cv2.putText(frame, text, org, _FONT, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, org, _FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_cross(frame, center, size=7, color=_WHITE, thickness=2):
    x, y = int(center[0]), int(center[1])
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def _rounded_rect(frame, pt1, pt2, color, radius):
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if radius <= 0:
        cv2.rectangle(frame, pt1, pt2, color, -1)
        return
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(frame, (cx, cy), radius, color, -1)


def _draw_speed_badge(frame, x: int, y_top: int, speed_kmh: float) -> int:
    """Big bold '92 km/h' on a rounded black badge, top-left anchored at
    (x, y_top). Returns the badge's bottom y."""
    text = f"{speed_kmh:.0f} km/h"
    (tw, th), _ = cv2.getTextSize(text, _FONT, 1.0, 2)
    pad_x, pad_y = 12, 9
    x2 = x + tw + 2 * pad_x
    y2 = y_top + th + 2 * pad_y
    _rounded_rect(frame, (x, y_top), (x2, y2), _BADGE_BG, radius=10)
    cv2.putText(frame, text, (x + pad_x, y2 - pad_y), _FONT, 1.0, _WHITE, 2, cv2.LINE_AA)
    return y2


def render_annotated_video(
    video_path: str,
    output_path: str,
    tracks: Dict[int, TrackData],
    speed_lookup: Dict[int, callable],  # track_id -> f(time_s) -> (speed_kmh, unc_kmh) or None
    fps_override: float = None,
    calibration_polygon: Optional[List[Tuple[float, float]]] = None,
):
    cap = cv2.VideoCapture(video_path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write with OpenCV first (widely supported for *writing*, not for
    # browser *playback*) to a temp file, then transcode to H.264 below.
    raw_path = output_path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

    poly_pts = None
    if calibration_polygon and len(calibration_polygon) >= 3:
        poly_pts = np.array(calibration_polygon, dtype=np.int32).reshape(-1, 1, 2)

    obs_by_frame: Dict[int, list] = {}
    for tid, tdata in tracks.items():
        for o in tdata.observations:
            obs_by_frame.setdefault(o.frame_idx, []).append((tid, tdata.class_name, o))

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        time_s = frame_idx / fps

        if poly_pts is not None:
            cv2.polylines(frame, [poly_pts], isClosed=True, color=_GREEN, thickness=2, lineType=cv2.LINE_AA)

        for tid, cls_name, o in obs_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = [int(v) for v in o.bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), _GREEN, 2)
            _draw_cross(frame, o.ground_px)

            label_y = max(16, y1 - 6)
            _text_with_outline(frame, f"V {tid}", (x1, label_y), 0.6, _WHITE, 2)

            fn = speed_lookup.get(tid)
            res = fn(time_s) if fn is not None else None
            if res is not None:
                v, _unc = res
                _draw_speed_badge(frame, x1, label_y + 6, v)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    # Transcode to browser-compatible H.264 + AAC-less mp4 (yuv420p is
    # required for wide compatibility; libx264 is the actual H.264 encoder).
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", raw_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        # If ffmpeg isn't available for some reason, fall back to the raw
        # file so the pipeline doesn't hard-fail — but note it won't play
        # in most browsers.
        print(f"[annotate] ffmpeg transcode failed ({exc}); serving raw mp4v file instead")
        os.replace(raw_path, output_path)
        return
    finally:
        if os.path.exists(raw_path) and raw_path != output_path:
            try:
                os.remove(raw_path)
            except OSError:
                pass
