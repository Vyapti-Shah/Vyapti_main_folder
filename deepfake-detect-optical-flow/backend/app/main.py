import os, time, uuid, glob, tempfile, shutil, cv2, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from . import pipeline, model as M

OUT_DIR = os.getenv("OUT_DIR", "/app/outputs")
os.makedirs(OUT_DIR, exist_ok=True)
app = FastAPI(title="Deepfake Detection (Optical-Flow Clip Selection)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
RESULTS = {}


def _cleanup(max_age=24 * 3600):
    for f in glob.glob(os.path.join(OUT_DIR, "*")):
        try:
            if time.time() - os.path.getmtime(f) > max_age: os.unlink(f)
        except OSError: pass


@app.on_event("startup")
def warm():
    M.get_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": M.MODEL_LOADED, "model_path": M.MODEL_PATH}


@app.post("/detect/image")
def detect_image(file: UploadFile = File(...)):
    _cleanup()
    img = cv2.imdecode(np.frombuffer(file.file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None: raise HTTPException(400, "Invalid image")
    rid = uuid.uuid4().hex[:12]
    try: res = pipeline.analyze_image(img, os.path.join(OUT_DIR, f"{rid}.jpg"))
    except ValueError as e: raise HTTPException(422, str(e))
    res["id"] = rid
    res["annotated_url"] = f"/outputs/{rid}.jpg" if res.pop("annotated", False) else None
    RESULTS[rid] = res
    return res


@app.post("/detect/video")
def detect_video(file: UploadFile = File(...)):
    _cleanup()
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp); path = tmp.name
    rid = uuid.uuid4().hex[:12]
    try: res = pipeline.analyze_video(path, os.path.join(OUT_DIR, f"{rid}.mp4"))
    except ValueError as e: raise HTTPException(422, str(e))
    finally: os.unlink(path)
    res["id"] = rid
    res["annotated_url"] = f"/outputs/{rid}.mp4" if res.pop("annotated", False) else None
    RESULTS[rid] = res
    return res


@app.get("/analysis/{rid}")
def analysis(rid: str):
    if rid not in RESULTS: raise HTTPException(404, "Not found")
    return RESULTS[rid]