"""Texture projection: visible vertices take their colour from the source pixels;
hidden vertices copy the colour at the same image position (mirrored fill)."""
import cv2
import numpy as np
from scipy.spatial import cKDTree


def colorize(mesh, bgr, mask, f, cx, cy):
    H, W = bgr.shape[:2]
    v, n = mesh.vertices, mesh.vertex_normals
    z = np.clip(v[:, 2], 1e-3, None)
    u, w = f * v[:, 0] / z + cx, f * v[:, 1] / z + cy
    facing = (n * -(v / np.linalg.norm(v, axis=1, keepdims=True))).sum(1) > 0.05
    gs = 4
    gw, gh = W // gs + 1, H // gs + 1
    ix, iy = np.clip((u / gs).astype(int), 0, gw - 1), np.clip((w / gs).astype(int), 0, gh - 1)
    zb = np.full((gh, gw), np.inf)
    np.minimum.at(zb, (iy[facing], ix[facing]), z[facing])
    vis = facing & (z <= zb[iy, ix] + 0.05)
    ui, wi = np.clip(np.round(u).astype(int), 0, W - 1), np.clip(np.round(w).astype(int), 0, H - 1)
    inside = (u >= 0) & (u < W) & (w >= 0) & (w < H)
    er = cv2.erode(mask, np.ones((7, 7), np.uint8))
    vis &= inside & (er[wi, ui] > 0)
    rgb = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (0, 0), 1.5)
    col = np.full((len(v), 3), 128, np.uint8)
    if vis.sum() > 50:
        col[vis] = rgb[wi[vis], ui[vis]]
        _, nn = cKDTree(np.c_[u[vis], w[vis]]).query(np.c_[u[~vis], w[~vis]])
        col[~vis] = col[vis][nn]
    return np.c_[col, np.full(len(v), 255, np.uint8)]
