"""
Centralized configuration for the shaky stabilization pipeline.
No magic numbers should exist anywhere else in the codebase — everything
tunable lives here.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # RAFT is slow on CPU at full resolution. Downscaling the frames before
    # inference speeds things up ~10x while still capturing the dominant
    # background camera motion.
    MAX_RAFT_WIDTH: int = 480

    # RAFT requires input dimensions divisible by this value.
    FLOW_PADDING_MULTIPLE: int = 8

    # Slight zoom-in applied to every output frame so the frame edges
    # (which shift due to stabilization) never show black borders.
    # Set to 1.0 to disable zoom completely as requested.
    BORDER_CROP_RATIO: float = 1.0

    # Size of the moving-average window used to smooth the camera trajectory.
    DEFAULT_RADIUS: int = 120

    # Local RAFT checkpoint (must sit in the project root, or pass an
    # absolute path via CLI/config override).
    LOCAL_MODEL_PATH: str = "raft_large_C_T_SKHT_V2-ff5fadd5.pth"

    # How often (in frames) to emit progress logs.
    PROGRESS_LOG_INTERVAL: int = 50


settings = Settings()
