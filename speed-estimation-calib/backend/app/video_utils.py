from __future__ import annotations

import cv2
import numpy as np


def probe(path, frame_out):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Could not open this file as a video.")
    ok, frame = cap.read()
    if not ok:
        raise ValueError("Could not read any frame from this video.")
    h, w = frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    assumed = not (1 <= fps <= 240)
    if assumed:
        fps = 30.0
    cv2.imwrite(str(frame_out), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {"width": w, "height": h, "fps": round(float(fps), 3), "frames": max(n, 0),
            "duration_s": round(n / fps, 2) if n > 0 else None, "fps_assumed": assumed}


def estimate_camera_motion(path, fps, n_frames, samples=10, span_s=8.0, thresh_pct=1.5):
    """Heuristic: median feature shift (static background dominates) between frame 0 and later frames."""
    unknown = {"moving": False, "known": False, "shift_pct": None}
    last = min(n_frames - 1, int(span_s * fps))
    if n_frames <= 0 or last < 5:
        return unknown
    cap = cv2.VideoCapture(str(path))

    def gray(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if not ok:
            return None
        g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (640, max(1, int(g.shape[0] * 640 / g.shape[1]))))

    g0 = gray(0)
    pts = None if g0 is None else cv2.goodFeaturesToTrack(g0, 400, 0.01, 8)
    if pts is None or len(pts) < 30:
        cap.release()
        return unknown
    shifts = []
    for i in np.linspace(last / samples, last, samples):
        g = gray(i)
        if g is None:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(g0, g, pts, None, winSize=(31, 31), maxLevel=4)
        m = st.ravel() == 1
        if m.sum() < 25:
            continue
        d = (p1 - pts).reshape(-1, 2)[m]
        shifts.append(float(np.linalg.norm(np.median(d, axis=0))))
    cap.release()
    if not shifts:
        return unknown
    pct = max(shifts) / 640.0 * 100
    return {"moving": pct > thresh_pct, "known": True, "shift_pct": round(pct, 2)}
