"""A small JSON-file store.

The parent app's store is a large threaded, debounced, whole-document affair
because it holds an entire indexing pipeline's output. This app holds two
tables, so it is a dict written through on every change — simpler to read, and
nothing here is hot enough to need the machinery.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .config import get_settings
from .models import Clip, Render


class Store:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._clips: dict[str, dict] = {}
        self._renders: dict[str, list[dict]] = {}   # clip_id -> renders, oldest first
        self._load()

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:      # missing or corrupt = empty, never fatal on boot
            return
        self._clips = raw.get("clips", {})
        self._renders = raw.get("renders", {})

    def _save(self) -> None:
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(
            {"clips": self._clips, "renders": self._renders}, indent=2), encoding="utf-8")
        tmp.replace(self._path)      # atomic: a crash mid-write cannot truncate the store

    # ---- clips ----
    def put_clip(self, clip: Clip) -> None:
        with self._lock:
            self._clips[clip.clip_id] = clip.model_dump(mode="json")
            self._save()

    def get_clip(self, clip_id: str) -> Clip | None:
        with self._lock:
            raw = self._clips.get(clip_id)
        return Clip.model_validate(raw) if raw else None

    def list_clips(self) -> list[Clip]:
        with self._lock:
            rows = list(self._clips.values())
        rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        return [Clip.model_validate(r) for r in rows]

    def delete_clip(self, clip_id: str) -> None:
        with self._lock:
            self._clips.pop(clip_id, None)
            self._renders.pop(clip_id, None)
            self._save()

    # ---- renders ----
    def put_render(self, render: Render) -> None:
        with self._lock:
            rows = self._renders.setdefault(render.clip_id, [])
            rows[:] = [r for r in rows if r.get("render_id") != render.render_id]
            rows.append(render.model_dump(mode="json"))
            self._save()

    def list_renders(self, clip_id: str) -> list[Render]:
        """Newest first — the order the UI lists them in."""
        with self._lock:
            rows = list(self._renders.get(clip_id, []))
        rows.sort(key=lambda r: r.get("created_at") or 0, reverse=True)
        return [Render.model_validate(r) for r in rows]

    def get_render(self, clip_id: str, render_id: str) -> Render | None:
        return next((r for r in self.list_renders(clip_id) if r.render_id == render_id), None)

    def latest_render(self, clip_id: str) -> Render | None:
        rows = self.list_renders(clip_id)
        return rows[0] if rows else None

    def delete_render(self, clip_id: str, render_id: str) -> Render | None:
        """Drop the row and hand it back so the caller can unlink its file."""
        with self._lock:
            rows = self._renders.get(clip_id, [])
            match = next((r for r in rows if r.get("render_id") == render_id), None)
            if match is None:
                return None
            rows[:] = [r for r in rows if r.get("render_id") != render_id]
            self._save()
        return Render.model_validate(match)


_store: Store | None = None


def get_store() -> Store:
    global _store
    if _store is None:
        _store = Store(get_settings().data_dir / "store.json")
    return _store
