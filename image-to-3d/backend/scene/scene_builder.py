"""Compose per-object GLBs into one scene. Layout is heuristic: x from image position,
depth from the object's bottom edge, size from a per-class nominal real-world size."""
import numpy as np
import trimesh
from trimesh.transformations import scale_matrix, translation_matrix as T

NOMINAL = {"person": 1.7, "bicycle": 1.7, "car": 4.5, "motorcycle": 2.1, "bus": 11.0, "truck": 7.0,
           "train": 20.0, "airplane": 30.0, "boat": 6.0, "chair": 0.9, "couch": 2.0, "bed": 2.0,
           "dining table": 1.6, "potted plant": 0.6, "tv": 1.0, "laptop": 0.35, "bench": 1.5,
           "backpack": 0.5, "suitcase": 0.7, "bottle": 0.25, "cup": 0.1, "vase": 0.3, "toilet": 0.75,
           "refrigerator": 1.8, "umbrella": 1.0, "handbag": 0.35, "dog": 0.7, "cat": 0.45,
           "horse": 2.2, "entire image": 1.5}


def build_scene(parts, W, H, out_path):
    sc = trimesh.Scene()
    for p in parts:
        meshes = trimesh.load(str(p["file"]), force="scene").dump(concatenate=False)
        b = np.array([m.bounds for m in meshes])
        lo, hi = b[:, 0].min(0), b[:, 1].max(0)
        c = (lo + hi) / 2
        s = 1.0 if p["metric"] else NOMINAL.get(p["label"], 0.6) / max((hi - lo).max(), 1e-6)
        x0, y0, x1, y1 = p["bbox"]
        pos = [((x0 + x1) / 2 / W - 0.5) * 8.0, 0.0, (y1 / H - 0.5) * 6.0]
        M = T(pos) @ scale_matrix(s) @ T([-c[0], -lo[1], -c[2]])
        for i, m in enumerate(meshes):
            m = m.copy()
            m.apply_transform(M)
            sc.add_geometry(m, node_name=f"{p['id']}_{i}", geom_name=f"{p['id']}_{i}")
    sc.export(str(out_path))
