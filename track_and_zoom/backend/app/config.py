"""Settings for the standalone Track & Zoom app.

Only the twelve settings the extracted services actually read, so the copied
`services/follow_zoom.py`, `services/sam2_tracker.py` and
`services/render_control.py` run here UNMODIFIED — they import
`..config.get_settings()` and nothing else from the parent application. Keeping
them byte-identical is the point: this app is meant to be where the feature is
worked on, and a divergent copy would be worse than no copy at all.

Every value is overridable with a TZ_-prefixed environment variable.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TZ_", extra="ignore")

    # --- storage ---
    data_dir: Path = Path("data")

    # --- SAM2 (the tracker) ---
    # Weights are looked up as <sam2_weights_dir>/<sam2_model>.pt, inside this
    # app's own folder. Deliberately NOT pointed at any sibling project: this
    # app has to run from a fresh clone of just this directory.
    sam2_weights_dir: Path = Path("models/sam2")
    # Fetch the checkpoint on first boot if it is not there. Set false for an
    # air-gapped machine and place the .pt file yourself.
    sam2_auto_download: bool = True
    sam2_model: str = "sam2.1_hiera_base_plus"
    sam2_device: str = "cuda"                    # falls back to cpu automatically

    # --- follow zoom (the compositor) ---
    follow_zoom_smoother: str = "kalman"         # kalman | sma
    # Frames the camera path may coast on the last known velocity during an
    # occlusion before it holds position. Unbounded coasting sends the crop off
    # the frame and it never comes back.
    follow_zoom_max_predict_frames: int = 15
    # Multiplier on the operator's drag box, applied around its centre.
    # 1.15 keeps a thin margin OUTSIDE the selection — about 7% of the box's
    # width added to each side — so the subject is not jammed against the edge
    # of the zoomed panel with no visible surroundings. 1.0 would be a literal
    # match to the drag; anything much above ~1.3 starts undoing the zoom.
    follow_zoom_padding_scale: float = 1.15
    # Degenerate-crop guard only — a stray click can produce a 2px box.
    follow_zoom_min_zoom_w: int = 32
    follow_zoom_min_zoom_h: int = 32
    follow_zoom_output_w: int = 1920
    follow_zoom_output_h: int = 1080
    follow_zoom_pip_scale: float = 0.32          # PiP width as a fraction of output
    follow_zoom_show_window_box: bool = True     # red outline marking the zoom region

    # --- this app's own dirs (not read by the copied services) ---
    uploads_subdir: str = "uploads"
    renders_subdir: str = "renders"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / self.uploads_subdir

    @property
    def follow_zoom_dir(self) -> Path:
        return self.data_dir / self.renders_subdir

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.upload_dir, self.follow_zoom_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
