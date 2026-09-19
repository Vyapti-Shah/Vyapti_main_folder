import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SAM2 video predictor — used for ALL detection+tracking now (no YOLO).
# Checkpoint is downloaded at image build time (see Dockerfile).
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/app/checkpoints/sam2.1_hiera_small.pt")
SAM2_MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")

# Motion detector: seeds candidate object boxes for SAM2 by finding regions
# that actually moved during a warm-up window, class-agnostic.
MOTION_WARMUP_FRAMES = int(os.environ.get("MOTION_WARMUP_FRAMES", "90"))
MOTION_MIN_AREA_PX = int(os.environ.get("MOTION_MIN_AREA_PX", "600"))
MOTION_MAX_OBJECTS = int(os.environ.get("MOTION_MAX_OBJECTS", "30"))

# Sliding window (frames) for windowed speed estimation
SPEED_WINDOW_FRAMES = int(os.environ.get("SPEED_WINDOW_FRAMES", "15"))

# Monte Carlo simulations for uncertainty propagation
MC_SAMPLES = int(os.environ.get("MC_SAMPLES", "5000"))

# Default assumed measurement noise (used unless the user overrides them)
DEFAULT_BBOX_JITTER_PX = float(os.environ.get("DEFAULT_BBOX_JITTER_PX", "2.5"))
DEFAULT_CALIB_JITTER_PX = float(os.environ.get("DEFAULT_CALIB_JITTER_PX", "3.0"))
DEFAULT_TIMESTAMP_JITTER_S = float(os.environ.get("DEFAULT_TIMESTAMP_JITTER_S", "0.003"))

# Objects whose estimated speed stays below this are treated as stationary
# and excluded from both the report and the annotated video.
STATIONARY_SPEED_THRESHOLD_KMH = float(os.environ.get("STATIONARY_SPEED_THRESHOLD_KMH", "3.0"))
# It must also have moved at least this fraction of its own bounding-box
# diagonal over the whole track (scale-independent second check).
MIN_RELATIVE_MOTION = float(os.environ.get("MIN_RELATIVE_MOTION", "0.4"))