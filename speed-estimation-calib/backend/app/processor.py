"""Detection + tracking + speed pipeline."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
from pathlib import Path

import cv2

from .speed import Track, check_gates, summarize, update_speed

MODEL_NAME = os.getenv("YOLO_MODEL", "yolo11s.pt")
TRACKER_CFG = str(Path(__file__).resolve().parent.parent / "bytetrack.yaml")
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 6: "train", 7: "truck"}
COLORS = {"person": (80, 200, 60), "bicycle": (200, 160, 40), "car": (40, 160, 245), "motorcycle": (60, 90, 240),
          "bus": (200, 90, 200), "truck": (60, 200, 230), "train": (170, 170, 170)}

INFER_LOCK = threading.Lock()   # one model, one inference at a time
_model = None


def get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO  # lazy: keeps the maths importable without torch
        _model = YOLO(MODEL_NAME)
    return _model


class YoloTracker:
    """YOLO detections + ByteTrack IDs. step(frame) -> [(xyxy, track_id, class_id, conf)]."""

    def __init__(self, conf, imgsz, classes, iou=0.5):
        self.kw = dict(classes=classes, conf=conf, iou=iou, imgsz=imgsz, verbose=False)

    def reset(self):
        get_model().predictor = None  # forces a fresh ByteTrack state

    def step(self, frame):
        b = get_model().track(frame, persist=True, tracker=TRACKER_CFG, **self.kw)[0].boxes
        if b is None or b.id is None or len(b) == 0:
            return []
        return list(zip(b.xyxy.cpu().numpy().tolist(), b.id.int().cpu().tolist(),
                        b.cls.int().cpu().tolist(), b.conf.cpu().tolist()))


# ---------------------------------------------------------------- drawing
def _text(img, text, x, y, color, sc, bg=True):
    th = max(1, round(sc * 2))
    (w, h), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    y = max(y, h + bl + 2)
    if bg:
        cv2.rectangle(img, (x, y - h - bl - 2), (x + w + 6, y + 2), color, -1)
    cv2.putText(img, text, (x + 3, y - bl // 2), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), th, cv2.LINE_AA)


def _draw_box(img, box, label, color, sc):
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, max(2, round(sc * 3)))
    _text(img, label, x1, y1 - 4, color, sc)


def _draw_gates(img, gates, sc):
    for key, col in (("A", (94, 197, 34)), ("B", (11, 158, 245))):
        a, b = gates[key]
        cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), col, max(2, round(sc * 3)), cv2.LINE_AA)
        _text(img, f"LINE {key}", int(a[0]), int(a[1]) - 6, col, sc)


def _transcode(raw: Path, dst: Path):
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr[-300:])
    raw.unlink(missing_ok=True)


# ---------------------------------------------------------------- video job
def process_video(job, video_path, meta, proj, gates, st, out: Path, tracker, transcode=True):
    W, H, fps, total = meta["width"], meta["height"], meta["fps"], meta["frames"]
    stride, window = st["stride"], st["window_s"]
    sc = max(0.5, W / 1600.0)
    cap = cv2.VideoCapture(str(video_path))
    raw = out / "raw.mp4"
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps / stride, (W, H))
    if not wr.isOpened():
        raise RuntimeError("Could not open the video writer.")
    tracker.reset()
    tracks, idx = {}, -1
    with open(out / "trajectories.csv", "w", newline="") as fh:
        tw = csv.writer(fh)
        tw.writerow(["frame", "time_s", "id", "class", "ground_x_px", "ground_y_px", "world_x_m", "world_y_m", "speed_kmh"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride:
                continue
            if frame.shape[1] != W or frame.shape[0] != H:
                frame = cv2.resize(frame, (W, H))
            t = idx / fps
            live = []
            if gates:
                _draw_gates(frame, gates, sc)
            for box, tid, cid, _cf in tracker.step(frame):
                name = COCO.get(cid)
                if name is None:
                    continue
                tr = tracks.get(tid) or tracks.setdefault(tid, Track(tid, name, t))
                tr.n, tr.last_t = tr.n + 1, t
                gp = ((box[0] + box[2]) / 2.0, box[3])
                w = proj.detection(box, tr.cls, W, H)
                if w is None:
                    tr.kmh, tr.prev = None, None
                else:
                    update_speed(tr, t, w[0], w[1], window)
                    if gates:
                        check_gates(tr, t, gp, gates, proj)
                    tr.prev = (t, gp[0], gp[1])
                k = tr.kmh
                tw.writerow([idx, round(t, 3), tid, tr.cls, round(gp[0], 1), round(gp[1], 1),
                             "" if w is None else round(w[0], 3), "" if w is None else round(w[1], 3),
                             "" if k is None else round(k, 2)])
                live.append({"id": tid, "cls": tr.cls, "kmh": None if k is None else round(k, 1)})
                _draw_box(frame, box, f"{tr.cls} #{tid}  " + ("..." if k is None else f"{k:.1f} km/h"),
                          COLORS[tr.cls], sc)
                if w is not None and not hasattr(proj, "f"):
                    cv2.circle(frame, (int(gp[0]), int(gp[1])), max(3, round(sc * 4)), (255, 255, 255), -1)
            wr.write(frame)
            job["live"], job["time_s"] = live, round(t, 2)
            job["progress"] = min(1.0, (idx + 1) / total) if total else 0.0
    cap.release()
    wr.release()
    job["status"], job["progress"] = "encoding", 1.0
    if transcode:
        _transcode(raw, out / "annotated.mp4")
    rows = summarize(tracks)
    with open(out / "tracks.csv", "w", newline="") as fh:
        tw = csv.writer(fh)
        tw.writerow(["id", "class", "first_s", "last_s", "speed_kmh", "basis", "avg_kmh", "peak_kmh",
                     "gate_order", "gate_time_s", "gate_distance_m", "gate_kmh"])
        for r in rows:
            g = r["gate"] or {}
            tw.writerow([r["id"], r["cls"], r["first_s"], r["last_s"], r["speed_kmh"], r["basis"], r["avg_kmh"],
                         r["peak_kmh"], g.get("order", ""), g.get("time_s", ""), g.get("distance_m", ""), g.get("kmh", "")])
    return rows


def run_job(job, video_path, meta, proj, gates, st, out: Path, summary):
    try:
        with INFER_LOCK:
            job["status"] = "running"
            tracker = YoloTracker(st["conf"], st["imgsz"], st["classes"])
            rows = process_video(job, video_path, meta, proj, gates, st, out, tracker)
        base = f"/files/jobs/{job['id']}"
        job["result"] = {"tracks": rows, "calibration": summary, "video_url": f"{base}/annotated.mp4",
                         "tracks_csv": f"{base}/tracks.csv", "trajectories_csv": f"{base}/trajectories.csv"}
        (out / "results.json").write_text(json.dumps(job["result"]))
        job["status"] = "done"
    except Exception as e:  # surfaced to the UI
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- image
def detect_image(src: Path, dst: Path, classes, conf, imgsz):
    with INFER_LOCK:
        m = get_model()
        m.predictor = None
        b = m.predict(str(src), classes=classes, conf=conf, imgsz=imgsz, verbose=False)[0].boxes
        boxes = [] if b is None else list(zip(b.xyxy.cpu().numpy().tolist(), b.cls.int().cpu().tolist(), b.conf.cpu().tolist()))
    img = cv2.imread(str(src))
    sc = max(0.5, img.shape[1] / 1600.0)
    out = []
    for box, cid, cf in boxes:
        name = COCO.get(cid, str(cid))
        _draw_box(img, box, f"{name} {cf:.0%}", COLORS.get(name, (200, 200, 200)), sc)
        out.append({"cls": name, "conf": round(cf, 3), "box": [round(v, 1) for v in box]})
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out
