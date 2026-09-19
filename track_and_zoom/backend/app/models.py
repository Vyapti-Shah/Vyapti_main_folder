"""The two shapes this app persists."""
from __future__ import annotations

import time

from pydantic import BaseModel, Field


class Clip(BaseModel):
    """One uploaded source video."""
    clip_id: str
    filename: str                       # under data/uploads
    display_name: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration: float = 0.0
    created_at: float = Field(default_factory=time.time)


class Render(BaseModel):
    """One finished track-and-zoom render.

    `render_id` is in the filename, so a second render never overwrites the
    first — the whole history is kept and playable.
    """
    render_id: str
    clip_id: str
    filename: str                       # under data/renders
    label: str = ""
    seed_frame_idx: int = 0
    seed_box_norm: list[float] = Field(default_factory=list)
    frame_count: int = 0
    fps: float = 0.0
    frames_tracked: int = 0
    frames_occluded: int = 0
    zoom_w: int = 0
    zoom_h: int = 0
    created_at: float = Field(default_factory=time.time)
