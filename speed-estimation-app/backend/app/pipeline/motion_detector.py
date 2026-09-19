"""Class-agnostic moving-object detection via background subtraction.

Doesn't know or care what the object *is* — only whether it moved during a
warm-up window. Produces bounding boxes on a single anchor frame, which are
then handed to SAM2 as prompts (motion_detector finds *what*, SAM2 tracks
*where it goes*).
"""
from typing import List, Tuple

import cv2
import numpy as np


def detect_moving_object_boxes(
    video_path: str,
    warmup_frames: int = 90,
    min_area_px: int = 600,
    max_boxes: int = 30,
) -> Tuple[int, List[np.ndarray]]:
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or warmup_frames
    anchor_idx = min(warmup_frames - 1, max(n_frames - 1, 0))

    bg = cv2.createBackgroundSubtractorMOG2(history=warmup_frames, varThreshold=25, detectShadows=True)
    fg_mask = None
    for _ in range(anchor_idx + 1):
        ok, frame = cap.read()
        if not ok:
            break
        fg_mask = bg.apply(frame)
    cap.release()

    if fg_mask is None:
        return anchor_idx, []

    # MOG2 marks shadows as 127 — drop them, keep hard foreground (255) only
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_px:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append(np.array([x, y, x + w, y + h], dtype=np.float64))

    boxes.sort(key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))
    boxes = boxes[:max_boxes]
    return anchor_idx, boxes