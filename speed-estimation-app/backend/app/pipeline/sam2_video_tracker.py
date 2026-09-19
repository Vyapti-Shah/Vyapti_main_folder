"""SAM2-only detection + tracking.

Pipeline: motion_detector finds where things moved (class-agnostic) on one
anchor frame -> SAM2's video predictor is prompted with those boxes and
propagates memory-based masks forward AND backward through the whole video
-> each frame's mask gives a precise object boundary, from which we take
the bottom-of-mask point as the ground-contact point (better than a bbox
bottom-center for irregular/rotated objects).

Runs on CPU or GPU — CPU is just much slower, since SAM2 does real
per-frame segmentation rather than a lightweight bbox regression.
"""
import os
import shutil
import tempfile
from typing import Dict, Optional

import cv2
import numpy as np

from app.config import SAM2_CHECKPOINT, SAM2_MODEL_CFG
from app.pipeline.motion_detector import detect_moving_object_boxes
from app.pipeline.tracker import TrackData, FrameObs

_predictor = None


def _lazy_predictor():
    global _predictor
    if _predictor is not None:
        return _predictor
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _predictor = build_sam2_video_predictor(SAM2_MODEL_CFG, SAM2_CHECKPOINT, device=device)
    return _predictor


def _extract_frames(video_path: str, out_dir: str) -> int:
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        idx += 1
    cap.release()
    return idx


def _mask_to_bbox_and_ground(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    y_max = ys.max()
    xs_at_bottom = xs[ys == y_max]
    x_ground = float(np.mean(xs_at_bottom))
    return (
        np.array([x1, y1, x2, y2], dtype=np.float64),
        np.array([x_ground, float(y_max)]),
    )


def _record_frame(tracks: Dict[int, TrackData], frame_idx: int, obj_ids, mask_logits, fps: float):
    for i, obj_id in enumerate(obj_ids):
        mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
        result = _mask_to_bbox_and_ground(mask)
        if result is None:
            continue
        bbox, ground_px = result
        tracks[obj_id].observations.append(
            FrameObs(
                frame_idx=frame_idx,
                time_s=frame_idx / fps,
                ground_px=ground_px,
                bbox=bbox,
                # SAM2 doesn't expose a scalar detection confidence like
                # YOLO's objectness score; mask presence is binary here.
                conf=1.0,
            )
        )


def run_sam2_tracking(
    video_path: str,
    fps_override: Optional[float] = None,
    progress_cb=None,
    warmup_frames: int = 90,
    min_area_px: int = 600,
) -> Dict[int, TrackData]:
    cap = cv2.VideoCapture(video_path)
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    anchor_idx, boxes = detect_moving_object_boxes(
        video_path, warmup_frames=warmup_frames, min_area_px=min_area_px
    )
    if not boxes:
        return {}

    predictor = _lazy_predictor()
    frames_dir = tempfile.mkdtemp(prefix="sam2_frames_")
    try:
        n_frames = _extract_frames(video_path, frames_dir)
        if n_frames == 0:
            return {}
        anchor_idx = min(anchor_idx, n_frames - 1)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        tracks: Dict[int, TrackData] = {
            obj_id: TrackData(track_id=obj_id, class_name="object") for obj_id in range(len(boxes))
        }

        with torch.inference_mode(), torch.autocast(device, dtype=autocast_dtype):
            state = predictor.init_state(video_path=frames_dir)

            for obj_id, box in enumerate(boxes):
                predictor.add_new_points_or_box(
                    inference_state=state, frame_idx=anchor_idx, obj_id=obj_id, box=box,
                )

            done_steps = 0
            for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, start_frame_idx=anchor_idx):
                _record_frame(tracks, frame_idx, obj_ids, mask_logits, fps)
                done_steps += 1
                if progress_cb:
                    progress_cb(min(0.9, done_steps / max(n_frames, 1)))

            if anchor_idx > 0:
                for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
                    state, start_frame_idx=anchor_idx, reverse=True
                ):
                    if frame_idx == anchor_idx:
                        continue  # already recorded in the forward pass
                    _record_frame(tracks, frame_idx, obj_ids, mask_logits, fps)
                    done_steps += 1
                    if progress_cb:
                        progress_cb(min(0.9, done_steps / max(n_frames, 1)))

        return {tid: t for tid, t in tracks.items() if t.observations}
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)