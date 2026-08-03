"""Centralized configuration for the forensic video enhancement pipeline."""

from enum import Enum


class Device(Enum):
    CPU = "cpu"
    CUDA = "cuda"


DEFAULT_FPS = 30.0
WATERMARK_TEXT = "ENHANCED (LOW LIGHT)"


class EnhancementConfig:
    """All tunable parameters for the algorithmic enhancement stages."""

    # CLAHE
    CLAHE_CLIP: float = 1.2
    CLAHE_GRID: tuple = (8, 8)

    # Global contrast boost
    CONTRAST_ALPHA: float = 1.15
    CONTRAST_BETA: float = 15.0
