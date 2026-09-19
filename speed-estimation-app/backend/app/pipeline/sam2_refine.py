"""Optional ground-contact refinement using SAM2 (facebook/sam2-hiera-small-hf).

This is OFF by default (USE_SAM2=0) because it roughly doubles processing
time. When enabled, for each YOLO box we prompt SAM2 with that box and take
the lowest point of the resulting mask that lies within the box's horizontal
span as the ground-contact point, instead of the raw bbox bottom-center.
This mainly helps with tall or partially occluded objects where the bbox
bottom edge is a poor proxy for where the object actually touches the ground.

If the model can't be loaded (no internet / no weights cached), we log a
warning and the caller falls back to the plain bbox bottom-center.
"""
from typing import Optional

import numpy as np

from app.config import SAM2_MODEL_ID

_model = None
_processor = None
_load_failed = False


def _lazy_load():
    global _model, _processor, _load_failed
    if _model is not None or _load_failed:
        return
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
        _model = Sam2Model.from_pretrained(SAM2_MODEL_ID).to(device)
        _model.eval()
    except Exception as exc:  # noqa: BLE001
        print(f"[sam2_refine] could not load SAM2 ({SAM2_MODEL_ID}): {exc}. "
              f"Falling back to bbox bottom-center for ground point.")
        _load_failed = True


def refine_ground_point(frame_bgr: np.ndarray, box_xyxy: np.ndarray) -> Optional[np.ndarray]:
    """Returns a refined (x, y) ground-contact point, or None if unavailable."""
    _lazy_load()
    if _model is None:
        return None

    try:
        import torch
        import cv2

        device = next(_model.parameters()).device
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        box = [[float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])]]

        inputs = _processor(images=image_rgb, input_boxes=[box], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = _model(**inputs, multimask_output=False)

        masks = _processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
        )[0]
        mask = masks[0, 0].numpy().astype(bool)

        ys, xs = np.where(mask)
        if len(ys) == 0:
            return None
        y_max = ys.max()
        xs_at_bottom = xs[ys == y_max]
        x_ground = float(np.mean(xs_at_bottom))
        return np.array([x_ground, float(y_max)])
    except Exception as exc:  # noqa: BLE001
        print(f"[sam2_refine] inference failed: {exc}")
        return None
