import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .detect import router as detect_router
from .model import detector
from .video import router as video_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(_: FastAPI):
    detector.load_async()
    yield


app = FastAPI(title="Deepfake Detection API", lifespan=lifespan)

# Default covers the frontend container's published port and the Vite dev server, so
# "docker compose up" and "npm run dev" both work without extra setup. Override with a
# comma-separated list for any other deployment; "*" disables credentials automatically
# per the CORS spec, which is fine here since no cookies/auth are used.
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3100,http://127.0.0.1:3100,http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(detect_router)
app.include_router(video_router)


@app.get("/health")
def health():
    """Liveness: always 200 while the process is up; reports model state."""
    return {"status": "ok", "model_state": detector.state, "model_error": detector.error}


@app.get("/ready")
def ready():
    """Readiness: 200 only when the model is loaded and validated."""
    if detector.state != "ready":
        return JSONResponse({"status": detector.state, "error": detector.error}, status_code=503)
    return {"status": "ready"}


@app.get("/model-info")
def model_info():
    if detector.state != "ready":
        return JSONResponse(
            {"state": detector.state, "error": detector.error, "model_id": detector.info.get("model_id")},
            status_code=503,
        )
    return {"state": "ready", "load_seconds": detector.load_seconds, **detector.info}
