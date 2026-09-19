import uuid
import shutil
from pathlib import Path
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from fastapi.responses import FileResponse

from ..config import settings
from ..models import DetectionResult

router = APIRouter(prefix="/api", tags=["detection"])


def get_file_extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def get_file_type(extension: str) -> str:
    if extension in settings.allowed_image_formats:
        return "image"
    elif extension in settings.allowed_video_formats:
        return "video"
    return "unknown"


@router.post("/detect", response_model=DetectionResult)
async def detect(request: Request, file: UploadFile = File(...)):
    """Upload an image or video and run deepfake detection on it."""
    extension = get_file_extension(file.filename)
    file_type = get_file_type(extension)

    if file_type == "unknown":
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {extension}. "
            f"Allowed: {settings.allowed_image_formats + settings.allowed_video_formats}",
        )

    file_id = str(uuid.uuid4())
    upload_path = Path(settings.uploads_dir) / f"{file_id}.{extension}"
    output_ext = extension if file_type == "image" else "mp4"
    output_path = Path(settings.outputs_dir) / f"{file_id}_result.{output_ext}"

    upload_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save uploaded file
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    detection_service = request.app.state.detection_service
    frames_analyzed = None

    try:
        if file_type == "image":
            is_fake, confidence = detection_service.predict(str(upload_path))
            detection_service.add_watermark(
                image_path=str(upload_path),
                output_path=str(output_path),
                is_real=not is_fake,
            )
        else:
            video_processor = request.app.state.video_processor
            is_fake, confidence, frames_analyzed = video_processor.process_video(
                video_path=str(upload_path),
                output_path=str(output_path),
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    return DetectionResult(
        file_id=file_id,
        file_name=file.filename,
        file_type=file_type,
        is_fake=is_fake,
        confidence=confidence,
        output_path=f"/api/result/{file_id}_result.{output_ext}",
        created_at=datetime.utcnow(),
        frames_analyzed=frames_analyzed,
    )


@router.get("/result/{filename}")
async def get_result(filename: str):
    """Serve a processed output file."""
    file_path = Path(settings.outputs_dir) / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
