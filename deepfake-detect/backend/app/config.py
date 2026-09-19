from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DeepFake Detection API"
    debug: bool = True
    model_path: str = "/app/models/efficientnet_b0_ffpp_c23.pth"
    fake_class_index: int = 1
    uploads_dir: str = "/app/data/uploads"
    outputs_dir: str = "/app/data/outputs"
    max_file_size: int = 500 * 1024 * 1024  # 500MB
    allowed_image_formats: list = ["jpg", "jpeg", "png", "gif", "bmp"]
    allowed_video_formats: list = ["mp4", "avi", "mov", "mkv", "webm"]
    frame_sample_fps: float = 10.0  # Sample ~10 frames/sec for video analysis
    face_margin: float = 0.3  # Expand MTCNN face boxes by 30% before classifying
    fake_frame_threshold: float = 0.6  # Flag video as fake if >=60% of frames are fake
    general_detector_model: str = "Organika/sdxl-detector"  # Whole-image real-vs-AI classifier

    class Config:
        env_file = ".env"


settings = Settings()
