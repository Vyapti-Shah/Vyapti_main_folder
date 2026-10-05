"""Detection + instance segmentation (YOLOv8-seg) and person keypoints (YOLOv8-pose)."""
import cv2
import numpy as np
from ultralytics import YOLO

from config import DATA
from human.pose import attach_keypoints

_seg = _pose = None


def _models():
    global _seg, _pose
    if _seg is None:
        _seg, _pose = YOLO("yolov8x-seg.pt"), YOLO("yolov8x-pose.pt")
    return _seg, _pose


def detect(bgr, image_id, conf=0.35):
    seg, pose = _models()
    H, W = bgr.shape[:2]
    (DATA / "masks" / image_id).mkdir(parents=True, exist_ok=True)
    (DATA / "crops" / image_id).mkdir(parents=True, exist_ok=True)
    r = seg(bgr, conf=conf, verbose=False)[0]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    objs = []
    if r.masks is not None:
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy()
        for i, poly in enumerate(r.masks.xy):
            if len(poly) < 3:
                continue
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [poly.astype(np.int32)], 255)
            oid = f"obj{i}"
            cv2.imwrite(str(DATA / "masks" / image_id / f"{oid}.png"), m)
            x0, y0, x1, y1 = [int(v) for v in boxes[i]]
            pw, ph = int((x1 - x0) * .06), int((y1 - y0) * .06)
            a, b, c, d = max(0, x0 - pw), max(0, y0 - ph), min(W, x1 + pw), min(H, y1 + ph)
            rgba = np.dstack([rgb, m])[b:d, a:c]
            cv2.imwrite(str(DATA / "crops" / image_id / f"{oid}.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            objs.append({"id": oid, "label": r.names[int(clss[i])], "conf": float(confs[i]),
                         "bbox": [x0, y0, x1, y1],
                         "crop_url": f"/files/crops/{image_id}/{oid}.png"})
    attach_keypoints(objs, pose(bgr, verbose=False)[0])
    # fallback for anything outside the 80 COCO classes
    cv2.imwrite(str(DATA / "masks" / image_id / "whole.png"), np.full((H, W), 255, np.uint8))
    cv2.imwrite(str(DATA / "crops" / image_id / "whole.png"), bgr)
    objs.append({"id": "whole", "label": "entire image", "conf": 1.0, "bbox": [0, 0, W, H],
                 "crop_url": f"/files/crops/{image_id}/whole.png"})
    return {"image_id": image_id, "width": W, "height": H, "objects": objs}
