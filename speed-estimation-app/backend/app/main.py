import shutil
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.config import UPLOAD_DIR, OUTPUT_DIR
from app.jobs import start_job, JOBS, REPORTS
from app.schemas import CalibrationRequest, JobStatus, Report

app = FastAPI(title="Forensic Speed Estimation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

VIDEOS: dict[str, Path] = {}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".mp4"
    dest = UPLOAD_DIR / f"{video_id}{ext}"
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    VIDEOS[video_id] = dest

    cap = cv2.VideoCapture(str(dest))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "video_id": video_id,
        "fps": fps,
        "n_frames": n_frames,
        "width": width,
        "height": height,
    }


@app.get("/api/frame/{video_id}")
def get_first_frame(video_id: str):
    if video_id not in VIDEOS:
        raise HTTPException(404, "unknown video_id")
    cap = cv2.VideoCapture(str(VIDEOS[video_id]))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(500, "could not read first frame")
    out_path = OUTPUT_DIR / f"{video_id}_frame0.jpg"
    cv2.imwrite(str(out_path), frame)
    return FileResponse(out_path, media_type="image/jpeg")


@app.post("/api/process", response_model=JobStatus)
def process_video(req: CalibrationRequest):
    if req.video_id not in VIDEOS:
        raise HTTPException(404, "unknown video_id")
    job_id = start_job(VIDEOS[req.video_id], req)
    return JOBS[job_id]


@app.get("/api/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job_id")
    return JOBS[job_id]


@app.get("/api/result/{job_id}", response_model=Report)
def get_result(job_id: str):
    if job_id not in REPORTS:
        raise HTTPException(404, "report not ready")
    return REPORTS[job_id]


@app.get("/api/video/{job_id}")
def get_video(job_id: str):
    path = OUTPUT_DIR / f"{job_id}_annotated.mp4"
    if not path.exists():
        raise HTTPException(404, "video not ready")
    return FileResponse(path, media_type="video/mp4", filename="annotated.mp4")


@app.get("/api/chart/{job_id}/{name}")
def get_chart(job_id: str, name: str):
    path = OUTPUT_DIR / job_id / name
    if not path.exists():
        raise HTTPException(404, "chart not found")
    return FileResponse(path, media_type="image/png")


@app.get("/api/charts/{job_id}")
def list_charts(job_id: str):
    d = OUTPUT_DIR / job_id
    if not d.exists():
        return JSONResponse([])
    return JSONResponse(sorted(p.name for p in d.glob("*.png")))
