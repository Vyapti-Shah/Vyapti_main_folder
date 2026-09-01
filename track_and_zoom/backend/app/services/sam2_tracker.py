"""Class-agnostic single-object video tracking via Meta's SAM2 video predictor.

This is the only tracker in this app's stack that is NOT restricted to a fixed
class vocabulary — dense_tracking.py and supervision_tracking.py both run YOLO
first, so they only ever follow person/vehicle detections. SAM2 instead
segments whatever region the operator drew a box around on frame 0..N and
propagates that exact mask forward AND backward through the rest of the video,
which is what services/follow_zoom.py needs to track an arbitrary object
(a bag, a jacket, anything) for its "drag-select, get a zoomed follow video"
feature.

Lazy-load idiom copied from dense_tracking.py::_load()/available() — cache the
loaded predictor OR a `_failed` sentinel, never raise from here, degrade to
"tracking unavailable" so a missing/unstaged checkpoint doesn't take the whole
app down.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Callable

from ..config import get_settings

log = logging.getLogger("vision.sam2_tracker")

_lock = threading.Lock()
_predictor = None
_failed = False
_last_load_error: str | None = None

BBox = tuple[float, float, float, float]

# sam2_model setting -> the config path bundled inside the installed `sam2`
# package (same names SAM2's own image_predictor_example.ipynb references).
_CONFIGS = {
    "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}


def available() -> bool:
    try:
        import sam2  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def last_load_error() -> str | None:
    """Why _load() last failed (checkpoint missing, unknown model name, an
    exception from build_sam2_video_predictor itself, ...) — None if it never
    ran or last succeeded. available() only checks the package imports, so a
    caller needing to explain a 503/failed job to the operator should surface
    this instead of guessing a single fixed reason."""
    return _last_load_error


def _is_oom(exc: Exception) -> bool:
    """String-based, not isinstance-based: SAM2's hydra-driven model
    instantiation wraps the underlying torch.cuda.OutOfMemoryError inside its
    own generic RuntimeError (confirmed live — the real exception surfaces as
    "Error in call to target '...PositionEmbeddingSine': RuntimeError('CUDA
    error: out of memory...')"), so catching only the torch OOM class misses
    it. Same idiom as services/embeddings.py's _is_oom for the same reason."""
    msg = str(exc)
    return "out of memory" in msg.lower() or "OutOfMemoryError" in msg


def _load():
    global _predictor, _failed, _last_load_error
    if _predictor is not None or _failed:
        return _predictor
    with _lock:
        if _predictor is not None or _failed:
            return _predictor
        try:
            import torch
            from sam2.build_sam import build_sam2_video_predictor

            s = get_settings()
            ckpt = s.sam2_weights_dir / f"{s.sam2_model}.pt"
            cfg = _CONFIGS.get(s.sam2_model)
            if cfg is None:
                _last_load_error = f"unknown sam2_model {s.sam2_model!r}"
                log.warning("%s; follow-zoom disabled", _last_load_error)
                _failed = True
                return None
            if not ckpt.exists():
                _last_load_error = f"checkpoint not staged at {ckpt}"
                log.warning("SAM2 %s; follow-zoom disabled", _last_load_error)
                _failed = True
                return None
            device = "cuda" if (s.sam2_device == "cuda" and torch.cuda.is_available()) else "cpu"
            if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

            try:
                _predictor = build_sam2_video_predictor(cfg, str(ckpt), device=device)
            except Exception as exc:  # noqa: BLE001 — see _is_oom's docstring
                # This box's GPU is shared with vLLM + the rest of this app's
                # own warmed models (confirmed live: a 16GB card can sit at
                # essentially 0 free MiB with everything loaded) — headroom
                # genuinely fluctuates below what SAM2's image encoder needs,
                # with nothing here to reclaim. Same "clear cache, then fall
                # back to CPU" tier as services/embeddings.py's OOM retry.
                # Broad except (not RuntimeError): confirmed live that SAM2's
                # hydra-driven instantiate() wraps the underlying CUDA OOM in
                # hydra.errors.InstantiationException, which subclasses plain
                # Exception, not RuntimeError — a narrower catch here would
                # let every real OOM slip past this retry entirely, exactly
                # as it did before this was widened.
                if device != "cuda" or not _is_oom(exc):
                    raise
                log.warning("SAM2 GPU load hit CUDA OOM, clearing cache and retrying on CPU (%s)", exc)
                torch.cuda.empty_cache()
                # build_sam2_video_predictor's device= argument only takes
                # effect via a `.to(device)` AFTER the whole model tree is
                # built — but PositionEmbeddingSine.__init__ unconditionally
                # runs a CUDA "warmup cache" step whenever
                # torch.cuda.is_available() is true (confirmed live in
                # sam2/modeling/position_encoding.py), entirely ignoring
                # `device`. On a genuinely full GPU that warmup itself OOMs
                # before construction ever reaches the CPU move, so
                # device="cpu" alone does NOT avoid touching CUDA. Hiding
                # availability for just this one build call is the only way
                # to actually skip that branch. Narrow, synchronous window
                # (this whole block already holds `_lock`) — the only cost is
                # any OTHER thread that happens to check
                # torch.cuda.is_available() during this one build also sees
                # False for a few seconds, which is a false "no GPU" read
                # elsewhere, never a false "GPU is free" one, so nothing
                # can be corrupted by it, only briefly slowed.
                orig_is_available = torch.cuda.is_available
                torch.cuda.is_available = lambda: False
                try:
                    _predictor = build_sam2_video_predictor(cfg, str(ckpt), device="cpu")
                finally:
                    torch.cuda.is_available = orig_is_available
        except Exception as exc:  # noqa: BLE001
            _last_load_error = str(exc)
            log.warning("SAM2 unavailable (%s); follow-zoom disabled", exc)
            _failed = True
    return _predictor


def _extract_frames(video_path: str, dest_dir: Path,
                    should_cancel=None) -> int:
    """SAM2's video predictor expects a directory of JPEG frames, one file per
    source frame, named so lexical order == frame order. No frame-dump helper
    exists elsewhere in this repo to reuse (media.py/scenes.py only pull
    sparse keyframes), so this is its own small ffmpeg shell-out."""
    from .render_control import run_cancellable

    dest_dir.mkdir(parents=True, exist_ok=True)
    # Decoding a whole video to JPEGs is minutes of work that reports nothing,
    # so it runs killable rather than blocking a cancel until it finishes.
    run_cancellable(
        ["ffmpeg", "-y", "-i", video_path, "-q:v", "2",
         str(dest_dir / "%06d.jpg")],
        should_cancel,
    )
    return len(list(dest_dir.glob("*.jpg")))


def _mask_to_bbox(mask) -> BBox | None:
    import numpy as np
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def track_bidirectional(
    video_id: str,
    video_path: str,
    *,
    seed_frame_idx: int,
    seed_box_px: BBox,
    progress: Callable[[float], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[int, BBox | None]:
    """Seed a box on `seed_frame_idx` and propagate it through the WHOLE
    video, both forward and backward from the seed — the object may have been
    on screen well before the operator paused to draw the box.

    Returns {frame_idx: bbox|None}, bbox in native pixel coordinates, None on
    frames where the mask came back empty (the object is fully occluded or
    off-screen that frame).
    """
    import numpy as np

    from .render_control import check

    check(should_cancel)
    predictor = _load()
    if predictor is None:
        raise RuntimeError(f"SAM2 tracker unavailable ({_last_load_error or 'unknown reason'})")

    s = get_settings()
    frames_dir = s.data_dir / "tmp_follow_zoom" / video_id
    try:
        n_frames = _extract_frames(video_path, frames_dir, should_cancel)
        if n_frames == 0:
            raise RuntimeError("no frames extracted from source video")
        check(should_cancel)

        # offload_*_to_cpu unconditionally: SAM2's own escape hatch for videos
        # too long to hold the per-frame memory bank entirely on the GPU. The
        # only downside is speed, never correctness, so this isn't config-gated.
        state = predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        # init_state above is inside SAM2 and cannot be interrupted part-way;
        # checking on both sides of it is the most that can be done, so a cancel
        # arriving during it takes effect as soon as it returns.
        check(should_cancel)
        predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=seed_frame_idx,
            obj_id=1,
            box=np.array(seed_box_px, dtype=np.float32),
        )

        results: dict[int, BBox | None] = {}

        def _consume(gen, weight: float, base: float) -> None:
            for frame_idx, obj_ids, mask_logits in gen:
                # Before the `obj_ids` skip, so a run of empty frames cannot
                # sail past a cancel.
                check(should_cancel)
                if not obj_ids:
                    continue
                mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
                results.setdefault(frame_idx, _mask_to_bbox(mask))
                if progress:
                    progress(base + weight * (frame_idx / max(1, n_frames - 1)))

        _consume(predictor.propagate_in_video(state, start_frame_idx=seed_frame_idx),
                 0.5, 0.0)
        _consume(predictor.propagate_in_video(state, start_frame_idx=seed_frame_idx, reverse=True),
                 0.5, 0.5)

        return results
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)
