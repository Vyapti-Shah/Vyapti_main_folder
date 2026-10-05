from __future__ import annotations

import json
import os
import re
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from . import processor, video_utils
from .geometry import LEVELS, build_projector

DATA = Path(os.getenv("DATA_DIR", "/data"))
for sub in ("videos", "jobs", "images"):
    (DATA / sub).mkdir(parents=True, exist_ok=True)

JOBS: dict[str, dict] = {}
EXEC = ThreadPoolExecutor(max_workers=1)   # jobs run one at a time, in order
ID_RE = re.compile(r"[0-9a-f]{12}")


@asynccontextmanager
async def lifespan(_app):
    threading.Thread(target=lambda: processor.get_model(), daemon=True).start()  # warm the model
    yield


app = FastAPI(title="Speed Estimation API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- schemas
class Calibration(BaseModel):
    mode: Literal["known_distance", "camera_params", "automatic"]
    line_a: Optional[list[list[float]]] = None     # [P1, P2]
    line_b: Optional[list[list[float]]] = None     # [P4, P3]
    len_a_m: Optional[float] = None                # |P1P2|
    len_b_m: Optional[float] = None                # |P4P3|
    sep_m: Optional[float] = None                  # P1 -> P4 along the road
    cam_height_m: Optional[float] = None
    tilt_deg: Optional[float] = None
    hfov_deg: Optional[float] = None
    focal_mm: Optional[float] = None
    sensor_width_mm: Optional[float] = None


class Settings(BaseModel):
    conf: float = Field(0.3, ge=0.05, le=0.9)
    imgsz: int = Field(640, ge=320, le=1920)
    stride: int = Field(1, ge=1, le=5)
    window_s: float = Field(1.0, ge=0.4, le=3.0)
    classes: list[int] = Field(default_factory=lambda: list(processor.COCO))

    @field_validator("classes")
    @classmethod
    def _classes(cls, v):
        if not v or any(c not in processor.COCO for c in v):
            raise ValueError("classes must be a non-empty subset of the supported COCO ids")
        return v


class CalibrateRequest(BaseModel):
    video_id: str
    calibration: Calibration


class JobRequest(CalibrateRequest):
    settings: Settings = Settings()


# ---------------------------------------------------------------- helpers
def _video(vid: str):
    if not ID_RE.fullmatch(vid):
        raise HTTPException(404, "Unknown video")
    d = DATA / "videos" / vid
    if not (d / "meta.json").exists():
        raise HTTPException(404, "Unknown video – please upload it again.")
    return d, json.loads((d / "meta.json").read_text())


def _build(cal: Calibration, meta: dict):
    try:
        return build_projector(cal.model_dump(), meta["width"], meta["height"])
    except ValueError as e:
        raise HTTPException(400, str(e))


def _summary(info: dict, meta: dict):
    warnings, acc = list(info["warnings"]), info["accuracy"]
    cam = meta.get("camera", {})

    def lower(a, b):
        return LEVELS[min(LEVELS.index(a), LEVELS.index(b))]

    if cam.get("moving"):
        warnings.append("⚠️ Camera appears to be moving. Speed estimates may be less accurate.")
        acc = lower(acc, "LOW")
    if meta.get("fps_assumed"):
        warnings.append("No frame rate found in the file; 30 FPS assumed. Speeds scale directly with this value.")
        acc = lower(acc, "LOW")
    return {"mode": info["mode"], "checks": info["checks"], "warnings": warnings, "accuracy": acc,
            "fps": meta["fps"], "camera": "Moving" if cam.get("moving") else "Fixed",
            "camera_known": bool(cam.get("known")), "distance_m": info.get("distance_m")}


# ---------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    return {"ok": True, "model": processor.MODEL_NAME, "model_loaded": processor._model is not None}


@app.post("/api/videos")
def upload_video(file: UploadFile = File(...)):
    vid = uuid.uuid4().hex[:12]
    d = DATA / "videos" / vid
    d.mkdir(parents=True)
    dest = d / ("input" + (Path(file.filename or "").suffix.lower() or ".mp4"))
    with dest.open("wb") as fh:
        shutil.copyfileobj(file.file, fh, 1 << 20)
    try:
        info = video_utils.probe(dest, d / "frame0.jpg")
    except ValueError as e:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, str(e))
    info["camera"] = video_utils.estimate_camera_motion(dest, info["fps"], info["frames"])
    info["file"] = dest.name
    (d / "meta.json").write_text(json.dumps(info))
    return {"video_id": vid, "frame_url": f"/files/videos/{vid}/frame0.jpg", **info}


@app.post("/api/images")
def upload_image(file: UploadFile = File(...), conf: float = 0.3):
    iid = uuid.uuid4().hex[:12]
    d = DATA / "images" / iid
    d.mkdir(parents=True)
    src = d / ("input" + (Path(file.filename or "").suffix.lower() or ".jpg"))
    with src.open("wb") as fh:
        shutil.copyfileobj(file.file, fh, 1 << 20)
    if cv2.imread(str(src)) is None:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "Could not read this file as an image.")
    dets = processor.detect_image(src, d / "annotated.jpg", list(processor.COCO), conf, 960)
    return {"image_id": iid, "annotated_url": f"/files/images/{iid}/annotated.jpg", "detections": dets}


@app.post("/api/calibrate")
def calibrate(req: CalibrateRequest):
    _, meta = _video(req.video_id)
    _, info = _build(req.calibration, meta)
    return {"ok": True, **_summary(info, meta)}


@app.post("/api/jobs")
def create_job(req: JobRequest):
    vdir, meta = _video(req.video_id)
    proj, info = _build(req.calibration, meta)
    jid = uuid.uuid4().hex[:12]
    out = DATA / "jobs" / jid
    out.mkdir(parents=True)
    job = {"id": jid, "status": "queued", "progress": 0.0, "time_s": 0.0, "live": [], "error": None, "result": None}
    JOBS[jid] = job
    EXEC.submit(processor.run_job, job, vdir / meta["file"], meta, proj, info.get("gates"),
                req.settings.model_dump(), out, _summary(info, meta))
    return {"job_id": jid}


@app.get("/api/jobs/{jid}")
def get_job(jid: str):
    job = JOBS.get(jid)
    if job is None:
        raise HTTPException(404, "Unknown job (the server may have restarted).")
    return job
