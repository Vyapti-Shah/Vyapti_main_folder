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
# How often (in frames) to rescan for newly-appearing moving objects during
# the forward pass. The initial warm-up only sees what's already moving in
# its first few seconds; without periodic rescanning, anything that enters
# the frame (or starts moving) later in the video is silently never tracked.
MOTION_RESCAN_INTERVAL_FRAMES = int(os.environ.get("MOTION_RESCAN_INTERVAL_FRAMES", "45"))
# A rescan candidate box is treated as "already tracked" (and skipped) if it
# overlaps an existing track's latest known box by more than this IoU.
MOTION_NEW_OBJECT_IOU_THRESH = float(os.environ.get("MOTION_NEW_OBJECT_IOU_THRESH", "0.2"))

# Speed thresholds (km/h) for the color-coded speed badges: below SLOW is
# shown green, SLOW..FAST amber, at/above FAST red.
SPEED_BAND_SLOW_KMH = float(os.environ.get("SPEED_BAND_SLOW_KMH", "20"))
SPEED_BAND_FAST_KMH = float(os.environ.get("SPEED_BAND_FAST_KMH", "60"))

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