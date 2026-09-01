"""Fetch the SAM2 checkpoint this app needs, into its own models directory.

This exists so the app depends on nothing outside its own folder. It used to
mount the sibling video-indexer's staged checkpoint, which made "run this on
its own" untrue — a fresh clone of just this directory would have started and
then failed at the first render with a missing-weights error.

Downloads are idempotent and resumable-ish: the file is written to a .part and
moved into place only on success, so an interrupted download can never leave a
truncated checkpoint that torch would fail to load in a much more confusing way.
Never raises — a missing checkpoint is reported through /api/health and by the
render that needs it, not by refusing to boot.
"""
from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

log = logging.getLogger("track_and_zoom.weights")

BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
KNOWN = {
    "sam2.1_hiera_tiny": "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small": "sam2.1_hiera_small.pt",
    "sam2.1_hiera_base_plus": "sam2.1_hiera_base_plus.pt",
    "sam2.1_hiera_large": "sam2.1_hiera_large.pt",
}
# A truncated download still produces a file. The smallest real checkpoint is
# ~150MB, so anything under this is a failed transfer, not a model.
_MIN_PLAUSIBLE_BYTES = 50 * 1024 * 1024


def checkpoint_path(weights_dir: Path, model: str) -> Path:
    return weights_dir / f"{model}.pt"


def is_present(weights_dir: Path, model: str) -> bool:
    p = checkpoint_path(weights_dir, model)
    return p.exists() and p.stat().st_size >= _MIN_PLAUSIBLE_BYTES


def ensure(weights_dir: Path, model: str, *, allow_download: bool = True) -> bool:
    """Make sure <weights_dir>/<model>.pt exists. Returns whether it does."""
    dest = checkpoint_path(weights_dir, model)
    if is_present(weights_dir, model):
        return True
    if dest.exists():
        log.warning("removing a truncated checkpoint at %s (%d bytes)",
                    dest, dest.stat().st_size)
        dest.unlink(missing_ok=True)
    if model not in KNOWN:
        log.error("unknown SAM2 model %r; expected one of %s", model, sorted(KNOWN))
        return False
    if not allow_download:
        log.warning("checkpoint missing at %s and downloads are disabled", dest)
        return False

    url = f"{BASE_URL}/{KNOWN[model]}"
    weights_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".pt.part")
    log.info("downloading SAM2 checkpoint %s -> %s (this is a few hundred MB, once)",
             url, dest)
    try:
        with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as out:
            shutil.copyfileobj(r, out, length=1024 * 1024)
        if part.stat().st_size < _MIN_PLAUSIBLE_BYTES:
            raise OSError(f"download too small ({part.stat().st_size} bytes)")
        part.replace(dest)      # atomic: only a complete file appears at `dest`
        log.info("SAM2 checkpoint ready (%.0f MB)", dest.stat().st_size / 1e6)
        return True
    except Exception as exc:      # noqa: BLE001 — report, never block startup
        log.error("could not download the SAM2 checkpoint (%s). Fetch it manually: "
                  "curl -L -o %s %s", exc, dest, url)
        part.unlink(missing_ok=True)
        return False
