from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .models import HealthResponse
from .services.detection_service import DeepFakeService
from .services.video_processor import VideoProcessor
from .routers import detection


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up: loading DeepFake detection model...")
    app.state.detection_service = DeepFakeService(
        model_path=settings.model_path,
        fake_class_index=settings.fake_class_index,
        face_margin=settings.face_margin,
        general_detector_model=settings.general_detector_model,
    )
    app.state.video_processor = VideoProcessor(
        detection_service=app.state.detection_service,
        frame_sample_fps=settings.frame_sample_fps,
        fake_frame_threshold=settings.fake_frame_threshold,
    )
    print("Model loaded. Ready to serve requests.")
    yield
    print("Shutting down.")


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detection.router)


@app.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", app_name=settings.app_name)
