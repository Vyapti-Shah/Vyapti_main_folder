from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DetectionResult(BaseModel):
    file_id: str
    file_name: str
    file_type: str  # "image" or "video"
    is_fake: bool
    confidence: float
    output_path: str
    created_at: datetime
    frames_analyzed: Optional[int] = None  # For videos


class DetectionRequest(BaseModel):
    file_name: str
    file_type: str


class HealthResponse(BaseModel):
    status: str
    app_name: str
