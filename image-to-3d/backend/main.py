import json
import re
import threading
import traceback
import uuid

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from config import DATA
from detection.detector import detect
from export.glb_export import glb_to_obj_zip
from human.body_reconstruction import reconstruct_person
from object3d.client import generate_object, worker_up
from scene.scene_builder import build_scene

app = FastAPI(title="Image to 3D Studio")
app.mount("/files", StaticFiles(directory=str(DATA)), name="files")
JOBS, GPU_LOCK = {}, threading.Lock()
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ID = re.compile(r"^[A-Za-z0-9_]+$")


class GenReq(BaseModel):
    image_id: str
    object_ids: list[str]
    engine: str = "trellis"        # trellis | hunyuan
    person_mode: str = "smplx"     # smplx | engine
    seed: int = 1


@app.get("/api/health")
def health():
    return {"smplx": config.smplx_available(), "device": DEV,
            "trellis": worker_up("trellis"), "hunyuan": worker_up("hunyuan")}


@app.post("/api/detect")
def api_detect(file: UploadFile = File(...)):
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Not a valid image")
    if max(img.shape[:2]) > 2048:
        s = 2048 / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    image_id = uuid.uuid4().hex[:12]
    cv2.imwrite(str(DATA / "uploads" / f"{image_id}.png"), img)
    meta = detect(img, image_id)
    (DATA / "uploads" / f"{image_id}.json").write_text(json.dumps(meta))
    meta["image_url"] = f"/files/uploads/{image_id}.png"
    return meta


def run_job(job_id, req: GenReq):
    job = JOBS[job_id]
    log = job["log"].append
    try:
        meta = json.loads((DATA / "uploads" / f"{req.image_id}.json").read_text())
        objs = {o["id"]: o for o in meta["objects"]}
        out = DATA / "outputs" / job_id
        out.mkdir(parents=True, exist_ok=True)
        bgr = cv2.imread(str(DATA / "uploads" / f"{req.image_id}.png"))
        parts = []
        with GPU_LOCK:
            job["status"] = "running"
            for n, oid in enumerate(req.object_ids, 1):
                o, glb = objs[oid], out / f"{oid}.glb"
                log(f"[{n}/{len(req.object_ids)}] {o['label']} ({oid})")
                done = metric = False
                if o["label"] == "person" and req.person_mode == "smplx":
                    if not config.smplx_available():
                        log("  SMPL-X weights not found in models/smpl/smplx -> using image-to-3D engine")
                    elif "keypoints" not in o:
                        log("  no pose keypoints found -> using image-to-3D engine")
                    else:
                        try:
                            mask = cv2.imread(str(DATA / "masks" / req.image_id / f"{oid}.png"), 0)
                            reconstruct_person(bgr, mask, o["keypoints"], o["kp_conf"], o["bbox"], glb, DEV)
                            done = metric = True
                            log("  pose-fitted SMPL-X body + texture projection")
                        except Exception as e:
                            log(f"  SMPL-X failed ({e}) -> using image-to-3D engine")
                if not done:
                    log(f"  generating with {req.engine} (can take minutes on first run)")
                    generate_object(req.engine, DATA / "crops" / req.image_id / f"{oid}.png", glb, req.seed)
                parts.append({"id": oid, "label": o["label"], "bbox": o["bbox"], "file": glb, "metric": metric})
            log("composing scene")
            build_scene(parts, meta["width"], meta["height"], out / "scene.glb")
        job.update(status="done", scene_url=f"/files/outputs/{job_id}/scene.glb",
                   parts=[{"id": p["id"], "label": p["label"], "url": f"/files/outputs/{job_id}/{p['id']}.glb"} for p in parts])
    except Exception as e:
        traceback.print_exc()
        job.update(status="error", error=str(e))
        log(f"ERROR: {e}")


@app.post("/api/generate")
def api_generate(req: GenReq):
    if not ID.match(req.image_id) or not all(ID.match(i) for i in req.object_ids) or not req.object_ids:
        raise HTTPException(400, "bad request")
    if req.engine not in config.WORKERS:
        raise HTTPException(400, "engine must be trellis or hunyuan")
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "queued", "log": []}
    threading.Thread(target=run_job, args=(job_id, req), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[job_id]


@app.get("/api/jobs/{job_id}/download")
def api_download(job_id: str, fmt: str = "glb", part: str = "scene"):
    if not ID.match(job_id) or not ID.match(part):
        raise HTTPException(400, "bad id")
    p = DATA / "outputs" / job_id / f"{part}.glb"
    if not p.exists():
        raise HTTPException(404, "not found")
    if fmt == "glb":
        return FileResponse(p, media_type="model/gltf-binary", filename=f"{part}.glb")
    z = p.with_suffix(".obj.zip")
    if not z.exists():
        glb_to_obj_zip(p, z)
    return FileResponse(z, filename=f"{part}_obj.zip")
