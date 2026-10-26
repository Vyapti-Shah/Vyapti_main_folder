"""POST /detect/video: frame extraction (OpenCV), per-frame inference, mean-probability aggregation.

This is frame-sampled image classification, not temporal analysis: frames are scored
independently by the same image detector and averaged. It does not model motion, optical
flow or any frame-to-frame relationship, so it will not catch manipulations that only show
up across time.
"""
import logging
import os
import tempfile
import time

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from .common import decide, evaluation_accuracy, threshold_info
from .model import MODEL_ID, detector

log = logging.getLogger("video")
router = APIRouter()

ALLOWED_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_BYTES = int(float(os.getenv("MAX_VIDEO_MB", "200")) * 1024 * 1024)
SAMPLE_INTERVAL_SECONDS = float(os.getenv("VIDEO_SAMPLE_INTERVAL_SECONDS", "1.0"))
MAX_DURATION_SECONDS = float(os.getenv("VIDEO_MAX_DURATION_SECONDS", "120"))
MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "60"))


class VideoError(Exception):
    """A video-specific problem that maps to a 4xx response."""

    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message


def extract_frames(path: str) -> list:
    """Sample up to MAX_FRAMES frames, at most one per SAMPLE_INTERVAL_SECONDS, within the
    first MAX_DURATION_SECONDS of the video. Returns a list of PIL RGB images."""
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            raise VideoError(400, "File is not a readable video.")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps != fps:
            fps = 25.0  # fall back to a sane default; some containers misreport fps

        frame_step = max(1, round(fps * SAMPLE_INTERVAL_SECONDS))
        max_frame_index = round(fps * MAX_DURATION_SECONDS)

        frames = []
        idx = 0
        while len(frames) < MAX_FRAMES and idx <= max_frame_index:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(rgb))
            idx += frame_step

        if not frames:
            raise VideoError(400, "No decodable frames were found in the video.")
        return frames
    finally:
        cap.release()


def aggregate(frame_probabilities: list[float]) -> float:
    """Mean-probability baseline: the video-level fake probability is the arithmetic mean of
    each sampled frame's independent fake probability. No temporal modelling is applied."""
    return sum(frame_probabilities) / len(frame_probabilities)


@router.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    if detector.state != "ready":
        raise HTTPException(503, f"Model not ready (state: {detector.state}).")
    name = (file.filename or "").lower()
    ext = os.path.splitext(name)[1]
    if name and ext not in ALLOWED_EXT:
        raise HTTPException(415, f"Unsupported file extension {ext!r}; use mp4, mov, avi, mkv or webm.")

    tmp_path = None
    try:
        size = 0
        with tempfile.NamedTemporaryFile(suffix=ext or ".mp4", delete=False) as tmp:
            tmp_path = tmp.name
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(413, f"File too large (max {MAX_BYTES // (1024 * 1024)} MB).")
                tmp.write(chunk)
        if size == 0:
            raise HTTPException(400, "Empty file.")

        try:
            frames = extract_frames(tmp_path)
        except VideoError as e:
            raise HTTPException(e.status_code, e.message)

        t0 = time.perf_counter()
        per_frame = []
        try:
            for frame in frames:
                p = detector.predict(frame)
                if not 0.0 <= p <= 1.0 or p != p:
                    raise ValueError(f"invalid probability {p}")
                per_frame.append(p)
        except Exception:
            log.exception("Frame inference failed")
            raise HTTPException(500, "Inference failed; no prediction available.")
        ms = round((time.perf_counter() - t0) * 1000, 1)

        fake_p = aggregate(per_frame)
        prediction, confidence = decide(fake_p)
        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {"fake": fake_p, "real": 1 - fake_p},
            "aggregation": {
                "method": "mean_probability",
                "description": (
                    "Frames are sampled and classified independently by the image detector; the "
                    "video-level probability is their arithmetic mean. This is not temporal "
                    "deepfake analysis and does not look for inter-frame inconsistencies."
                ),
            },
            "frames_analyzed": len(per_frame),
            "frame_scores": per_frame,
            "sampling": {
                "interval_seconds": SAMPLE_INTERVAL_SECONDS,
                "max_duration_seconds": MAX_DURATION_SECONDS,
                "max_frames": MAX_FRAMES,
            },
            "thresholds": threshold_info(),
            "evaluation": evaluation_accuracy(),
            "model": {"id": MODEL_ID, "revision": detector.info.get("revision")},
            "inference_ms": ms,
            "video": {"bytes": size},
            "note": (
                "Estimates whether sampled frames look AI-generated, averaged across the video. "
                "It is not a guarantee, and it does not perform temporal or motion-based deepfake detection."
            ),
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
