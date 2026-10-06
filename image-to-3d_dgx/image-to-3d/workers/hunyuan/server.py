import gc, io, os, shutil, sys, tempfile, threading
ROOT = "/opt/Hunyuan3D-2.1"
sys.path[:0] = [f"{ROOT}/hy3dshape", f"{ROOT}/hy3dpaint", ROOT]
os.chdir(ROOT)
import torch
import trimesh
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from starlette.background import BackgroundTask

try:  # torchvision compatibility shim used by the official demo
    from hy3dpaint.utils.torchvision_fix import apply_fix
    apply_fix()
except Exception:
    pass

app, LOCK = FastAPI(), threading.Lock()


def _free():
    gc.collect()
    torch.cuda.empty_cache()


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(file: UploadFile = File(...), seed: int = Form(1)):
    work = tempfile.mkdtemp()
    img = Image.open(io.BytesIO(file.file.read()))
    with LOCK:
        from hy3dshape.rembg import BackgroundRemover
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        from hy3dpaint.textureGenPipeline import Hunyuan3DPaintConfig, Hunyuan3DPaintPipeline
        if img.mode != "RGBA":
            img = BackgroundRemover()(img.convert("RGB"))
        in_png = f"{work}/in.png"
        img.save(in_png)
        shape = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")
        mesh = shape(image=img)[0]
        raw = f"{work}/shape.obj"
        mesh.export(raw)
        del shape, mesh
        _free()
        paint = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))
        out_obj = f"{work}/textured.obj"
        res = paint(mesh_path=raw, image_path=in_png, output_mesh_path=out_obj, save_glb=False)
        del paint
        _free()
    glb_path = f"{work}/textured.glb"
    if not os.path.exists(glb_path):
        trimesh.load(res if isinstance(res, str) and os.path.exists(res) else out_obj, force="mesh").export(glb_path)
    return FileResponse(glb_path, media_type="model/gltf-binary", background=BackgroundTask(shutil.rmtree, work, True))
