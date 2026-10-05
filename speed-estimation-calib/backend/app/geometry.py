"""Calibration geometry: pixel -> ground-plane metres."""
from __future__ import annotations

import math

import cv2
import numpy as np

# Typical full heights (m), used only by Automatic calibration.
CLASS_HEIGHT_M = {"person": 1.70, "bicycle": 1.70, "motorcycle": 1.50, "car": 1.50,
                  "bus": 3.20, "truck": 3.20, "train": 3.80}
LEVELS = ["LOW", "MEDIUM", "HIGH"]


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def segment_hit(p0, p1, a, b):
    """Fraction t in [0,1] along p0->p1 where it meets segment a-b, else None."""
    p0, p1, a, b = (np.asarray(v, float) for v in (p0, p1, a, b))
    r, s = p1 - p0, b - a
    den = _cross(r, s)
    if abs(den) < 1e-12:
        return None
    qp = a - p0
    t, u = _cross(qp, s) / den, _cross(qp, r) / den
    return float(t) if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0 else None


class HomographyProjector:
    """Image -> ground plane (metres) through a 3x3 homography."""

    def __init__(self, H, ref_uv=None):
        self.H = np.asarray(H, float)
        self.sign = 1.0
        if ref_uv is not None:  # side of the horizon the calibrated area lives on
            self.sign = 1.0 if self.H[2] @ [ref_uv[0], ref_uv[1], 1.0] > 0 else -1.0

    def world_point(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        if p[2] * self.sign < 1e-9:  # at / above the horizon
            return None
        return float(p[0] / p[2]), float(p[1] / p[2])

    def detection(self, box, cls_name, W, H):
        x1, y1, x2, y2 = box
        if y2 >= H - 2:  # feet cut by the frame edge -> ground contact unknown
            return None
        return self.world_point((x1 + x2) / 2.0, y2)


class AutoProjector:
    """Pinhole + typical object height: Z = f*H_real/h_px (experimental)."""

    def __init__(self, W, hfov_deg):
        self.f = (W / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
        self.cx = W / 2.0

    def detection(self, box, cls_name, W, H):
        x1, y1, x2, y2 = box
        if y1 <= 1 or y2 >= H - 2 or (y2 - y1) < 12:
            return None
        Z = self.f * CLASS_HEIGHT_M.get(cls_name, 1.5) / (y2 - y1)
        return ((x1 + x2) / 2.0 - self.cx) * Z / self.f, Z


def _line(v, name):
    try:
        a = np.asarray(v, float)
    except Exception:
        a = None
    if a is None or a.shape != (2, 2) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be two points [[x1,y1],[x2,y2]].")
    return a


def _convex(q) -> bool:
    s = [_cross(q[(i + 1) % 4] - q[i], q[(i + 2) % 4] - q[(i + 1) % 4]) for i in range(4)]
    return all(x > 0 for x in s) or all(x < 0 for x in s)


def _pos(cal, key):
    v = cal.get(key)
    if v is None or not float(v) > 0:
        raise ValueError(f"'{key}' must be a number greater than 0.")
    return float(v)


def _cm_per_px(proj, u, v):
    a, b = proj.world_point(u, v), proj.world_point(u + 1, v)
    return None if a is None or b is None else math.hypot(a[0] - b[0], a[1] - b[1]) * 100


def build_projector(cal: dict, W: int, H: int):
    """Return (projector, info). Raises ValueError with a user-readable message."""
    mode = cal["mode"]
    info = {"mode": mode, "checks": [], "warnings": [], "accuracy": "HIGH", "gates": None}

    if mode == "known_distance":
        la, lb = _line(cal.get("line_a"), "Line A"), _line(cal.get("line_b"), "Line B")
        wa, wb, d = _pos(cal, "len_a_m"), _pos(cal, "len_b_m"), _pos(cal, "sep_m")
        src = np.float32([la[0], la[1], lb[1], lb[0]])  # P1, P2, P3, P4
        if not _convex(src.astype(float)):
            raise ValueError("Line A and Line B must not cross and must form a four-sided area on the road.")
        dst = np.float32([[-wa / 2, 0], [wa / 2, 0], [wb / 2, d], [-wb / 2, d]])
        proj = HomographyProjector(cv2.getPerspectiveTransform(src, dst), src.mean(0))
        info["gates"] = {"A": la.tolist(), "B": lb.tolist()}
        info["distance_m"] = d
        info["checks"] = [f"Line A (P1 → P2): {wa:g} m", f"Line B (P4 → P3): {wb:g} m",
                          f"Distance between the lines (P1 → P4): {d:g} m"]
        sa = _cm_per_px(proj, *la.mean(0))
        sb = _cm_per_px(proj, *lb.mean(0))
        if sa and sb:
            info["checks"].append(f"Scale: {sa:.1f} cm/px at Line A, {sb:.1f} cm/px at Line B")
            if max(sa, sb) / min(sa, sb) > 15:
                info["warnings"].append("Extreme perspective between the lines – check the entered distances.")
                info["accuracy"] = "MEDIUM"
        if cv2.contourArea(src) / float(W * H) < 0.04:
            info["warnings"].append("The calibrated area is small. Speed far outside it is extrapolated – "
                                    "draw the lines further apart for better accuracy.")
            info["accuracy"] = "MEDIUM"
        return proj, info

    hfov = cal.get("hfov_deg")
    if mode == "camera_params":
        h = _pos(cal, "cam_height_m")
        tilt = float(cal.get("tilt_deg") or 0)
        if not 1 <= tilt <= 90:
            raise ValueError("Camera tilt must be between 1° and 90° below the horizon.")
        if hfov:
            if not 5 <= float(hfov) <= 170:
                raise ValueError("Horizontal field of view must be between 5° and 170°.")
            f = (W / 2.0) / math.tan(math.radians(float(hfov)) / 2.0)
        else:
            f = _pos(cal, "focal_mm") * W / _pos(cal, "sensor_width_mm")
        th, c = math.radians(tilt), (W / 2.0, H / 2.0)
        Kinv = np.array([[1 / f, 0, -c[0] / f], [0, 1 / f, -c[1] / f], [0, 0, 1.0]])
        G = np.array([[h, 0, 0], [0, -h * math.sin(th), h * math.cos(th)], [0, math.cos(th), math.sin(th)]])
        info["checks"] = [f"Camera height: {h:g} m", f"Tilt: {tilt:g}° below horizontal",
                          f"Focal length: {f:.0f} px (horizontal FOV {math.degrees(2 * math.atan(W / (2 * f))):.1f}°)"]
        v_h = c[1] - f * math.tan(th)
        if v_h > 0:
            info["warnings"].append(f"The horizon is inside the frame (row {v_h:.0f}); objects above it are ignored "
                                    "and distant objects are less accurate.")
        info["accuracy"] = "MEDIUM"
        return HomographyProjector(G @ Kinv), info

    if mode == "automatic":
        hf = float(hfov or 70)
        if not 5 <= hf <= 170:
            raise ValueError("Assumed horizontal field of view must be between 5° and 170°.")
        info["checks"] = [f"Scale from typical object heights, assumed horizontal FOV {hf:g}°"]
        info["warnings"].append("Experimental: scale comes from average object sizes, so errors of ±20% or more are normal.")
        info["accuracy"] = "LOW"
        return AutoProjector(W, hf), info

    raise ValueError("Unknown calibration mode.")
