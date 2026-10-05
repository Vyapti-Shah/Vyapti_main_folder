"""Per-track speed maths (no model / video dependencies)."""
from __future__ import annotations

import math
from collections import deque

import numpy as np

from .geometry import segment_hit

# Physical plausibility caps (km/h); faster readings are treated as tracking glitches.
MAX_KMH = {"person": 45, "bicycle": 80, "motorcycle": 250, "car": 250, "bus": 160, "truck": 160, "train": 350}


class Track:
    def __init__(self, tid, cls, t):
        self.id, self.cls = tid, cls
        self.hist = deque()          # (t, X, Y) in metres
        self.speeds = []             # valid km/h samples
        self.prev = None             # (t, u, v) previous ground point in pixels
        self.cross = {}              # 'A'/'B' -> (t, X, Y)
        self.n, self.first_t, self.last_t, self.kmh = 0, t, t, None


def window_speed(hist, window_s, min_pts=4):
    """m/s from a least-squares line through the last `window_s` seconds of positions."""
    if len(hist) < min_pts:
        return None
    t_end = hist[-1][0]
    pts = [h for h in hist if t_end - h[0] <= window_s + 1e-9]
    if len(pts) < min_pts or pts[-1][0] - pts[0][0] < 0.5 * window_s:
        return None
    a = np.asarray(pts, float)
    t = a[:, 0] - a[:, 0].mean()
    den = float((t * t).sum())
    if den <= 0:
        return None
    vx = float((t * (a[:, 1] - a[:, 1].mean())).sum() / den)
    vy = float((t * (a[:, 2] - a[:, 2].mean())).sum() / den)
    return math.hypot(vx, vy)


def update_speed(tr, t, X, Y, window_s):
    tr.hist.append((t, X, Y))
    while tr.hist[0][0] < t - 2 * window_s:
        tr.hist.popleft()
    tr.kmh = None
    ms = window_speed(tr.hist, window_s)
    if ms is not None and ms * 3.6 <= MAX_KMH.get(tr.cls, 250):
        tr.kmh = ms * 3.6
        tr.speeds.append(tr.kmh)
    return tr.kmh


def check_gates(tr, t, gp, gates, proj):
    """Record first crossing of Line A / Line B with sub-frame time interpolation."""
    if tr.prev is None:
        return
    p0 = (tr.prev[1], tr.prev[2])
    for key in ("A", "B"):
        if key in tr.cross:
            continue
        s = segment_hit(p0, gp, gates[key][0], gates[key][1])
        if s is None:
            continue
        pc = (p0[0] + s * (gp[0] - p0[0]), p0[1] + s * (gp[1] - p0[1]))
        w = proj.world_point(*pc)
        if w is not None:
            tr.cross[key] = (tr.prev[0] + s * (t - tr.prev[0]), w[0], w[1])


def summarize(tracks, min_frames=5):
    rows = []
    for tr in sorted(tracks.values(), key=lambda x: x.id):
        if tr.n < min_frames:
            continue
        sp = np.asarray(tr.speeds)
        ok = len(sp) >= 5
        row = {"id": tr.id, "cls": tr.cls, "first_s": round(tr.first_t, 3), "last_s": round(tr.last_t, 3),
               "avg_kmh": round(float(sp.mean()), 1) if ok else None,
               "peak_kmh": round(float(np.percentile(sp, 95)), 1) if ok else None, "gate": None}
        if "A" in tr.cross and "B" in tr.cross:
            (ta, xa, ya), (tb, xb, yb) = tr.cross["A"], tr.cross["B"]
            dt, dist = abs(tb - ta), math.hypot(xa - xb, ya - yb)
            ms = dist / dt if dt >= 0.04 else None
            row["gate"] = {"order": "A→B" if ta <= tb else "B→A", "t_a": round(ta, 3), "t_b": round(tb, 3),
                           "time_s": round(dt, 3), "distance_m": round(dist, 2),
                           "ms": round(ms, 2) if ms else None, "kmh": round(ms * 3.6, 1) if ms else None}
        g = row["gate"]
        row["speed_kmh"] = g["kmh"] if g and g["kmh"] is not None else row["avg_kmh"]
        row["basis"] = "line crossing" if g and g["kmh"] is not None else "tracked average"
        rows.append(row)
    return rows
