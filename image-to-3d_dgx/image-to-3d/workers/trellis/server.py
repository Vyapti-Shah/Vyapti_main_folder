import gc, io, os, tempfile, threading
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.background import BackgroundTask

app, LOCK, PIPE = FastAPI(), threading.Lock(), None
KEEP = os.getenv("KEEP_LOADED", "0") == "1"


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(file: UploadFile = File(...), seed: int = Form(1)):
    global PIPE
    img = Image.open(io.BytesIO(file.file.read()))
    with LOCK:
        from trellis.pipelines import TrellisImageTo3DPipeline
        from trellis.utils import postprocessing_utils
        if PIPE is None:
            PIPE = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
            PIPE.cuda()
        out = PIPE.run(img, seed=seed)  # RGBA input -> alpha used as mask; RGB -> rembg
        glb = postprocessing_utils.to_glb(out["gaussian"][0], out["mesh"][0], simplify=0.95, texture_size=1024)
        tmp = tempfile.NamedTemporaryFile(suffix=".glb", delete=False)
        glb.export(tmp.name)
        if not KEEP:
            PIPE = None
            gc.collect()
            torch.cuda.empty_cache()
    return FileResponse(tmp.name, media_type="model/gltf-binary", background=BackgroundTask(os.unlink, tmp.name))
