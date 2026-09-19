"""Standalone Track & Zoom — FastAPI entrypoint.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Drag a box around an object on a paused frame; SAM2 follows it through the
whole clip and a new video is rendered showing the original footage with a
top-right picture-in-picture of the zoomed crop tracking the object.
"""
from __future__ import annotations

import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from . import weights                     # noqa: E402
from .config import get_settings           # noqa: E402
from .routers import renders               # noqa: E402

log = logging.getLogger("track_and_zoom")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Make sure the checkpoint is present before anyone tries to render.

    Best-effort by design: a failed download must not stop the API coming up,
    or you cannot even reach /api/health to find out what went wrong. The
    render that needs it reports the real error.
    """
    s = get_settings()
    if not weights.ensure(s.sam2_weights_dir, s.sam2_model,
                          allow_download=s.sam2_auto_download):
        log.warning("SAM2 checkpoint not available — uploads and playback work, "
                    "but renders will fail until it is fetched")
    yield


app = FastAPI(title="Track & Zoom", version="1.0.0", lifespan=lifespan)

# Wide open: this app has no auth and is meant to be run locally while working
# on the feature. Do not expose it to an untrusted network.
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

app.include_router(renders.router)


@app.get("/api/health")
def health() -> dict:
    from .services import sam2_tracker

    s = get_settings()
    ckpt = weights.checkpoint_path(s.sam2_weights_dir, s.sam2_model)
    return {
        "status": "ok",
        "sam2": {
            # `available()` is the cheap "is the package importable" check. A
            # missing checkpoint only surfaces when a render actually loads the
            # model, so the path is reported separately rather than implied.
            "importable": sam2_tracker.available(),
            "model": s.sam2_model,
            "weights_path": str(ckpt),
            "weights_present": weights.is_present(s.sam2_weights_dir, s.sam2_model),
            "auto_download": s.sam2_auto_download,
            "device": s.sam2_device,
        },
        "zoom": {
            "padding_scale": s.follow_zoom_padding_scale,
            "pip_scale": s.follow_zoom_pip_scale,
            "max_predict_frames": s.follow_zoom_max_predict_frames,
            "show_window_box": s.follow_zoom_show_window_box,
            "output": f"{s.follow_zoom_output_w}x{s.follow_zoom_output_h}",
        },
        "data_dir": str(s.data_dir),
    }
