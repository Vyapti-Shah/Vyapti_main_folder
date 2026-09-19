"""The Track & Zoom API.

Mirrors the endpoint shape the feature has in the parent application, minus the
auth and the indexing pipeline it is embedded in:

    POST   /api/clips                          upload a source video
    GET    /api/clips                          list them
    DELETE /api/clips/{clip_id}                remove clip + its renders
    GET    /api/clips/{clip_id}/stream         the original
    POST   /api/clips/{clip_id}/render         start a render (box + timestamp)
    GET    /api/clips/{clip_id}/render/status  poll it
    POST   /api/clips/{clip_id}/render/cancel  stop it mid-way
    GET    /api/clips/{clip_id}/renders        the saved history, newest first
    DELETE /api/clips/{clip_id}/renders/{id}   delete one, row and file
    GET    /api/clips/{clip_id}/renders/{id}/stream   play one
"""
from __future__ import annotations

import logging
import shutil
import time
import uuid
from pathlib import Path

import cv2
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ..config import get_settings
from ..models import Clip, Render
from ..services.render_control import RenderCanceled
from ..store import get_store

log = logging.getLogger("track_and_zoom.api")

router = APIRouter(prefix="/api", tags=["track-and-zoom"])

_ALLOWED = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

# clip_id -> {status, progress, error, render_id, cancel_requested, updated_at}.
# In-memory like the parent's: a render is a one-shot foreground-ish operation,
# and a job surviving a restart it can no longer be running is not useful.
_jobs: dict[str, dict] = {}


class RenderRequest(BaseModel):
    box_norm: list[float]     # [x1, y1, x2, y2] in 0..1, as dragged on the frame
    time_s: float = 0.0       # the paused instant the box was drawn at
    label: str = ""


def _probe(path: Path) -> tuple[int, int, float, int]:
    cap = cv2.VideoCapture(str(path))
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return w, h, fps, n


# --------------------------------------------------------------------------- #
# Clips
# --------------------------------------------------------------------------- #
@router.post("/clips")
async def upload_clip(file: UploadFile = File(...), name: str = Form("")) -> dict:
    s = get_settings()
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED:
        raise HTTPException(400, f"unsupported type '{ext}'; allowed: {sorted(_ALLOWED)}")
    clip_id = uuid.uuid4().hex[:12]
    dest = s.upload_dir / f"{clip_id}{ext}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    w, h, fps, n = _probe(dest)
    if not w or not h:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "could not read that video — is it a valid file?")
    clip = Clip(clip_id=clip_id, filename=dest.name,
                display_name=name.strip() or (file.filename or dest.name),
                width=w, height=h, fps=fps, frame_count=n,
                duration=(n / fps) if fps else 0.0)
    get_store().put_clip(clip)
    return clip.model_dump(mode="json")


@router.get("/clips")
def list_clips() -> list[dict]:
    return [c.model_dump(mode="json") for c in get_store().list_clips()]


@router.delete("/clips/{clip_id}")
def delete_clip(clip_id: str) -> dict:
    s, store = get_settings(), get_store()
    clip = store.get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    for r in store.list_renders(clip_id):
        (s.follow_zoom_dir / r.filename).unlink(missing_ok=True)
    (s.upload_dir / clip.filename).unlink(missing_ok=True)
    store.delete_clip(clip_id)
    return {"deleted": clip_id}


@router.get("/clips/{clip_id}/stream")
def stream_clip(clip_id: str):
    s = get_settings()
    clip = get_store().get_clip(clip_id)
    if clip is None:
        raise HTTPException(404, "clip not found")
    path = s.upload_dir / clip.filename
    if not path.exists():
        raise HTTPException(404, "source file missing")
    return FileResponse(path)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _run_render(clip_id: str, box_norm: list[float], time_s: float,
                render_id: str, label: str) -> None:
    from ..services import follow_zoom

    s, store = get_settings(), get_store()
    job = _jobs[clip_id]
    target = s.follow_zoom_dir / f"{clip_id}_{render_id}.mp4"
    try:
        clip = store.get_clip(clip_id)
        if clip is None:
            job.update(status="failed", error="clip no longer exists")
            return
        source = s.upload_dir / clip.filename
        if not source.exists():
            job.update(status="failed", error="source file missing")
            return

        seed_frame_idx = max(0, round(time_s * clip.fps)) if clip.fps else 0
        x1, y1, x2, y2 = box_norm
        seed_box_px = (x1 * clip.width, y1 * clip.height,
                       x2 * clip.width, y2 * clip.height)

        def _progress(frac: float) -> None:
            job["progress"] = round(min(1.0, max(0.0, frac)), 3)
            job["updated_at"] = time.time()

        # Passed INTO the service, not smuggled through _progress: the tracking
        # and encode passes spend minutes in stretches that report nothing, and
        # a cancel arriving in one of those has to land anyway.
        def _cancelled() -> bool:
            return bool(job.get("cancel_requested"))

        result = follow_zoom.render_follow_zoom(
            clip_id, str(source), target,
            seed_frame_idx=seed_frame_idx, seed_box_px=seed_box_px,
            progress=_progress, should_cancel=_cancelled)

        store.put_render(Render(
            render_id=render_id, clip_id=clip_id, filename=target.name, label=label,
            seed_frame_idx=seed_frame_idx, seed_box_norm=box_norm,
            frame_count=result.frame_count, fps=result.fps,
            frames_tracked=result.frames_tracked,
            frames_occluded=result.frames_occluded,
            zoom_w=result.zoom_w, zoom_h=result.zoom_h))
        job.update(status="done", progress=1.0, error=None)
    except RenderCanceled:
        # A stopped render leaves a truncated file; drop it so the history never
        # lists a half-written video as if it were usable.
        target.unlink(missing_ok=True)
        job.update(status="canceled", error=None)
    except Exception as exc:      # noqa: BLE001 — a background task's exception is
        # otherwise swallowed; the poller is the only way this reaches anyone.
        log.exception("render failed for %s", clip_id)
        target.unlink(missing_ok=True)
        job.update(status="failed", error=str(exc))
    finally:
        job["updated_at"] = time.time()


@router.post("/clips/{clip_id}/render")
def start_render(clip_id: str, req: RenderRequest, background: BackgroundTasks) -> dict:
    from ..services import sam2_tracker

    if get_store().get_clip(clip_id) is None:
        raise HTTPException(404, "clip not found")
    if len(req.box_norm) != 4:
        raise HTTPException(400, "box_norm must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = req.box_norm
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise HTTPException(400, "box_norm must be a non-degenerate box in 0..1")
    if not sam2_tracker.available():
        raise HTTPException(503, "SAM2 is not installed in this environment")

    existing = _jobs.get(clip_id)
    if existing and existing["status"] == "running":
        return existing              # already in flight; don't start a second

    render_id = uuid.uuid4().hex[:10]
    _jobs[clip_id] = {"status": "running", "progress": 0.0, "error": None,
                      "render_id": render_id, "cancel_requested": False,
                      "updated_at": time.time()}
    background.add_task(_run_render, clip_id, req.box_norm, req.time_s,
                        render_id, req.label)
    return _jobs[clip_id]


@router.get("/clips/{clip_id}/render/status")
def render_status(clip_id: str) -> dict:
    job = _jobs.get(clip_id)
    if job is None:
        raise HTTPException(404, "no render has been started for this clip")
    return job


@router.post("/clips/{clip_id}/render/cancel")
def cancel_render(clip_id: str) -> dict:
    """Cooperative stop. The flag is read inside every frame loop and by the
    killable ffmpeg steps, so it lands within a frame rather than at the end of
    whatever pass happens to be running."""
    job = _jobs.get(clip_id)
    if job is None:
        raise HTTPException(404, "no render has been started for this clip")
    if job["status"] != "running":
        return job
    job["cancel_requested"] = True
    job["updated_at"] = time.time()
    return job


# --------------------------------------------------------------------------- #
# Saved renders
# --------------------------------------------------------------------------- #
@router.get("/clips/{clip_id}/renders")
def list_renders(clip_id: str) -> list[dict]:
    if get_store().get_clip(clip_id) is None:
        raise HTTPException(404, "clip not found")
    return [r.model_dump(mode="json") for r in get_store().list_renders(clip_id)]


@router.delete("/clips/{clip_id}/renders/{render_id}")
def delete_render(clip_id: str, render_id: str) -> dict:
    removed = get_store().delete_render(clip_id, render_id)
    if removed is None:
        raise HTTPException(404, "no such render")
    (get_settings().follow_zoom_dir / removed.filename).unlink(missing_ok=True)
    return {"deleted": render_id}


@router.get("/clips/{clip_id}/renders/{render_id}/stream")
def stream_render(clip_id: str, render_id: str):
    s = get_settings()
    render = get_store().get_render(clip_id, render_id)
    if render is None:
        raise HTTPException(404, "no such render")
    path = s.follow_zoom_dir / render.filename
    if not path.exists():
        raise HTTPException(404, "render file missing")
    return FileResponse(path)
