"""Class-agnostic moving-object detection via background subtraction.

Doesn't know or care what the object *is* — only whether it moved.

Used two ways:
  1. One-shot warm-up seed (`detect_moving_object_boxes`): finds whatever is
     already moving at the end of a short warm-up window. This is the first
     generation of objects, handed to SAM2 as initial box prompts.
  2. Continuous rescanning (`LiveMotionScanner`): the same background model
     keeps running for the rest of the video so objects that enter the
     frame — or start moving — *after* the warm-up window are still caught.
     A single anchor-frame snapshot alone only sees the first few seconds,
     which is why videos with objects appearing partway through used to
     lose them entirely.
"""
from typing import List, Tuple

import cv2
import numpy as np


def _boxes_from_fg_mask(fg_mask: np.ndarray, min_area_px: int, max_boxes: int) -> List[np.ndarray]:
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
    return boxes[:max_boxes]


def _box_iou(a: np.ndarray, b: np.ndarray) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


class LiveMotionScanner:
    """Wraps a single running MOG2 model so it keeps learning/adapting frame
    by frame for the whole video, and can be asked at any later point
    "what's moving here that isn't already being tracked?".

    Must be fed every frame in order (no gaps) to keep the background model
    coherent — skipping frames would make it mistake real motion for noise.
    """

    def __init__(self, warmup_frames: int = 90):
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=warmup_frames, varThreshold=25, detectShadows=True
        )
        self._last_fg_mask = None

    def feed(self, frame_bgr: np.ndarray) -> None:
        self._last_fg_mask = self._bg.apply(frame_bgr)

    def candidate_boxes(self, min_area_px: int, max_boxes: int) -> List[np.ndarray]:
        if self._last_fg_mask is None:
            return []
        return _boxes_from_fg_mask(self._last_fg_mask, min_area_px, max_boxes)

    def new_boxes(
        self,
        existing_boxes: List[np.ndarray],
        min_area_px: int,
        max_boxes: int,
        iou_thresh: float = 0.2,
    ) -> List[np.ndarray]:
        """Candidate boxes that don't already overlap a currently-tracked
        object — i.e. genuinely new moving objects, not a re-detection of
        one SAM2 is already following."""
        candidates = self.candidate_boxes(min_area_px, max_boxes)
        fresh = []
        for box in candidates:
            if any(_box_iou(box, existing) > iou_thresh for existing in existing_boxes):
                continue
            fresh.append(box)
        return fresh


def detect_moving_object_boxes(
    video_path: str,
    warmup_frames: int = 90,
    min_area_px: int = 600,
    max_boxes: int = 30,
) -> Tuple[int, List[np.ndarray], LiveMotionScanner]:
    """Warms up a background model over the first `warmup_frames` frames and
    returns the boxes already moving at that point, plus the still-running
    scanner so the caller can keep feeding it later frames and ask for *new*
    moving objects as the video goes on.
    """
    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or warmup_frames
    anchor_idx = min(warmup_frames - 1, max(n_frames - 1, 0))

    scanner = LiveMotionScanner(warmup_frames=warmup_frames)
    for _ in range(anchor_idx + 1):
        ok, frame = cap.read()
        if not ok:
            break
        scanner.feed(frame)
    cap.release()

    boxes = scanner.candidate_boxes(min_area_px, max_boxes)
    return anchor_idx, boxes, scanner
