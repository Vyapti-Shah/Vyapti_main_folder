from typing import List, Optional
from pydantic import BaseModel, Field


class Point2D(BaseModel):
    x: float
    y: float


class CalibrationRequest(BaseModel):
    video_id: str
    # "auto" (default): no user input needed, scale derived automatically
    # from known average real-world sizes of the detected object classes.
    # "manual": use image_points/world_points for a proper ground-plane
    # homography — more accurate when you have a real reference.
    calibration_mode: str = "auto"
    image_points: Optional[List[Point2D]] = None
    world_points: Optional[List[Point2D]] = None
    # optional overrides
    fps_override: Optional[float] = None
    bbox_jitter_px: Optional[float] = None
    calib_jitter_px: Optional[float] = None
    timestamp_jitter_s: Optional[float] = None
    use_sam2: Optional[bool] = None
    target_classes: Optional[List[str]] = None  # e.g. ["car", "person"]
    known_speeds_kmh: Optional[dict] = None  # {track_id_str: actual_speed} for evaluation


class JobStatus(BaseModel):
    job_id: str
    status: str  # queued | running | done | error
    progress: float = 0.0
    message: str = ""


class TrackSpeedResult(BaseModel):
    track_id: int
    class_name: str
    start_time_s: float
    end_time_s: float
    distance_m: float
    elapsed_s: float
    mean_speed_kmh: float
    speed_std_kmh: float
    ci95_low_kmh: float
    ci95_high_kmh: float
    n_samples: int
    tracking_confidence: float
    speed_series_kmh: List[float]
    time_series_s: List[float]
    known_class_size: Optional[bool] = None  # auto mode only: was the class size assumption a known prior?


class Report(BaseModel):
    job_id: str
    video_id: str
    fps: float
    n_frames: int
    calibration_mode: str = "auto"
    calibration_rmse_px: Optional[float] = None
    tracks: List[TrackSpeedResult]
    evaluation: Optional[dict] = None
