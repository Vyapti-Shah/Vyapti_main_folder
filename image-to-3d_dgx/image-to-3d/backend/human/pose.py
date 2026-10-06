import numpy as np


def _iou(a, b):
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def attach_keypoints(objs, pose_result):
    """Match COCO-17 keypoints to each detected person by bbox IoU."""
    if pose_result.keypoints is None or pose_result.boxes is None or len(pose_result.boxes) == 0:
        return
    xy = pose_result.keypoints.xy.cpu().numpy()
    cf = pose_result.keypoints.conf
    cf = cf.cpu().numpy() if cf is not None else np.ones(xy.shape[:2])
    bx = pose_result.boxes.xyxy.cpu().numpy()
    used = set()
    for o in objs:
        if o["label"] != "person":
            continue
        best, bi = 0.0, -1
        for j in range(len(bx)):
            if j not in used:
                v = _iou(o["bbox"], bx[j])
                if v > best:
                    best, bi = v, j
        if best > 0.4:
            used.add(bi)
            o["keypoints"], o["kp_conf"] = xy[bi].tolist(), cf[bi].tolist()
