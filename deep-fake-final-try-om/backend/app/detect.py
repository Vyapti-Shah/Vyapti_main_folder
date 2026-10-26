"""POST /detect/image: validation, inference, decision rules."""
import io
import logging
import os
import time

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from .common import decide, evaluation_accuracy, threshold_info
from .model import MODEL_ID, detector

log = logging.getLogger("detect")
router = APIRouter()

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BYTES = int(float(os.getenv("MAX_IMAGE_MB", "20")) * 1024 * 1024)
MAX_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", str(100_000_000)))


def decode_image(data: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(data)) as probe:
            fmt = probe.format
            if fmt not in ALLOWED_FORMATS:
                raise HTTPException(415, f"Unsupported image format {fmt!r}; use JPG, PNG or WEBP.")
            if probe.width * probe.height > MAX_PIXELS:
                raise HTTPException(413, f"Image dimensions too large (max {MAX_PIXELS} pixels).")
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as e:
        raise HTTPException(400, f"File is not a readable image: {e}") from e


@router.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    if detector.state != "ready":
        raise HTTPException(503, f"Model not ready (state: {detector.state}).")
    name = (file.filename or "").lower()
    if name and os.path.splitext(name)[1] not in ALLOWED_EXT:
        raise HTTPException(415, "Unsupported file extension; use .jpg, .jpeg, .png or .webp.")
    data = await file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(400, "Empty file.")

    image = decode_image(data)
    t0 = time.perf_counter()
    try:
        fake_p = detector.predict(image)
        if not 0.0 <= fake_p <= 1.0 or fake_p != fake_p:
            raise ValueError(f"invalid probability {fake_p}")
    except Exception:
        log.exception("Inference failed")
        raise HTTPException(500, "Inference failed; no prediction available.")
    ms = round((time.perf_counter() - t0) * 1000, 1)

    prediction, confidence = decide(fake_p)
    return {
        "prediction": prediction,
        "confidence": confidence,  # probability of the predicted class; null when INCONCLUSIVE
        "probabilities": {"fake": fake_p, "real": 1 - fake_p},
        "thresholds": threshold_info(),
        "evaluation": evaluation_accuracy(),  # null unless a measured eval result file exists
        "model": {"id": MODEL_ID, "revision": detector.info.get("revision")},
        "inference_ms": ms,
        "image": {"width": image.width, "height": image.height, "bytes": len(data)},
        "note": "Estimates whether the image is AI-generated; not a guarantee, and not a detector of all manipulation types.",
    }
