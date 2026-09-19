"""Renders the annotated output video: bounding box, track id, class,
and a live (windowed) speed readout with its uncertainty band."""
import subprocess
import os
from typing import Dict

import cv2
import numpy as np

from app.pipeline.tracker import TrackData


def _color_for_id(track_id: int):
    rng = np.random.default_rng(track_id * 7919 + 13)
    return tuple(int(c) for c in rng.integers(60, 255, size=3))


def render_annotated_video(
    video_path: str,
    output_path: str,
    tracks: Dict[int, TrackData],
    speed_lookup: Dict[int, callable],  # track_id -> f(time_s) -> (speed_kmh, unc_kmh) or None
    fps_override: float = None,
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

        for tid, cls_name, o in obs_by_frame.get(frame_idx, []):
            x1, y1, x2, y2 = [int(v) for v in o.bbox]
            color = _color_for_id(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(o.ground_px[0]), int(o.ground_px[1])), 4, color, -1)

            label = f"#{tid} {cls_name}"
            speed_txt = ""
            fn = speed_lookup.get(tid)
            if fn is not None:
                res = fn(time_s)
                if res is not None:
                    v, unc = res
                    speed_txt = f"{v:.1f} +/- {unc:.1f} km/h"

            y_text = max(20, y1 - 8)
            cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if speed_txt:
                cv2.putText(frame, speed_txt, (x1, y_text + 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.55, (255, 255, 255), 2)

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