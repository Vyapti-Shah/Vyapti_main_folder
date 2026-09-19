"""Shared per-frame / per-track data structures. Detection+tracking itself
now happens in sam2_video_tracker.py (SAM2-only pipeline)."""
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class FrameObs:
    frame_idx: int
    time_s: float
    ground_px: np.ndarray  # (2,) bottom-of-mask ground-contact point in image coords
    bbox: np.ndarray  # (4,) x1,y1,x2,y2
    conf: float


@dataclass
class TrackData:
    track_id: int
    class_name: str
    observations: List[FrameObs] = field(default_factory=list)