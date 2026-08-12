"""
find_target.py
================
Merged version of find_person.py + find_object.py.

For EVERY image in reference/, this script auto-detects whether it's a
PERSON (a face is detected in it) or an OBJECT (no face detected), and runs
the matching approach for that reference:

    PERSON reference -> InsightFace identity match (face) + YOLOv8 person
                         detection/tracking, exactly like find_person.py.
    OBJECT reference  -> YOLOv8 general-object candidate boxes + ORB
                         keypoint/homography instance matching + tracking,
                         exactly like find_object.py.

Each reference gets its own output folder named after the reference image's
filename (no extension), containing every matching clip across ALL input
videos, numbered in processing order, plus a "<reference_name>_merged.mp4"
sitting outside that folder.

Folder layout expected next to this script:

    reference/   -> put ONE clear photo per target (person OR object) here.
                     e.g. reference/kid.png (person), reference/bag.jpg
                     (object). EVERY image found is treated as a separate
                     target and auto-classified as person/object.
    inputs/      -> put video file(s) to scan here (.mp4, .mov, .mkv, .avi)
    outputs/     -> auto-created.
                       outputs/<reference_name>/clip_01.mp4, clip_02.mp4, ...
                       outputs/<reference_name>_merged.mp4  <- outside the
                                                                 per-reference
                                                                 folder

Usage:
    python find_target.py

Dependencies:
    pip install insightface onnxruntime opencv-python numpy ultralytics
    FFmpeg must be installed and available on PATH.
"""

import os
import sys
import glob
import subprocess
import tempfile
import numpy as np

# ---------------------------------------------------------------------------
# Model storage location — set this BEFORE importing insightface/cv2 so the
# library picks it up internally.
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ROOT = r"D:\insightface_models"
os.environ.setdefault("INSIGHTFACE_HOME", DEFAULT_MODEL_ROOT)

import cv2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
REFERENCE_DIR = "reference"
INPUT_DIR = "inputs"
OUTPUT_DIR = "outputs"

SAMPLE_FPS = 2                  # frames analyzed per second of video
MAX_ANALYSIS_WIDTH = 1280       # downscale frames wider than this before detection

# --- Person path (face identity + person detection) ---
FACE_SIM_THRESHOLD = 0.50       # cosine similarity to confirm identity via face
PERSON_MODEL_NAME = "yolov8n.pt"
PERSON_CONF_THRESHOLD = 0.40

# --- Object path (ORB instance matching) ---
OBJECT_MODEL_NAME = "yolov8n.pt"
OBJECT_CONF_THRESHOLD = 0.30
TARGET_CLASS_NAMES = None       # e.g. {"backpack", "handbag"} to restrict candidates

ORB_N_FEATURES = 500
ORB_RATIO_TEST = 0.75
MIN_GOOD_MATCHES = 12
MIN_INLIERS = 8
HIST_PREFILTER_THRESHOLD = 0.25

# --- Shared tracking / re-identification (person path) ---
IOU_MATCH_THRESHOLD = 0.30
APPEARANCE_SIM_THRESHOLD = 0.70
TRACK_ACTIVE_GAP_SECONDS = 2.0
TRACK_REID_GAP_SECONDS = 15.0

# --- Object-path tracking (deliberately much tighter than person tracking).
# Generic color-histogram appearance matching is too loose to trust an
# object track for long — it will happily "reconnect" to any similarly
# colored object anywhere in frame. If we let that continuity run for
# TRACK_REID_GAP_SECONDS like the person path does, a track that was
# confirmed once near the start can keep drifting onto unrelated boxes for
# the rest of the video, making hit_timestamps run all the way to the end
# instead of stopping shortly after the real object leaves frame.
# Keeping these short means an object track "dies" quickly once the object
# is no longer actually detected/matched, so padding correctly ends
# ~PADDING_SECONDS after the object's real last appearance.
OBJECT_IOU_MATCH_THRESHOLD = 0.30
OBJECT_APPEARANCE_SIM_THRESHOLD = 0.85   # stricter than person's 0.70
OBJECT_TRACK_ACTIVE_GAP_SECONDS = 1.0    # vs 2.0 for person
OBJECT_TRACK_REID_GAP_SECONDS = 3.0      # vs 15.0 for person

# How close two separate hit windows must be (seconds) to merge into one clip
MAX_GAP_SECONDS = 1.5

# Extra context to include before/after every appearance window, e.g. for an
# investigator who wants to see what happened right before/after the person
# showed up. If the person appears at 10s and again at 30s, each becomes its
# own clip: (10 - PADDING_SECONDS) -> (10 + PADDING_SECONDS), and likewise
# for 30s — unless the padded windows overlap, in which case they merge into
# one continuous clip. Set to 0 to disable and get exact hit-only clips.
PADDING_SECONDS = 15.0

VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def log(msg):
    print(f"[find_target] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Reference discovery
# ---------------------------------------------------------------------------
def find_reference_images(reference_dir):
    """Return a sorted list of every image path found in reference_dir."""
    candidates = []
    for ext in IMAGE_EXTENSIONS:
        candidates.extend(glob.glob(os.path.join(reference_dir, f"*{ext}")))
        candidates.extend(glob.glob(os.path.join(reference_dir, f"*{ext.upper()}")))

    if not candidates:
        log(f"ERROR: No reference image found in '{reference_dir}/'. "
            f"Add one clear photo per target (person or object, jpg/png).")
        sys.exit(1)

    return sorted(set(candidates))


# ---------------------------------------------------------------------------
# PERSON path — InsightFace identity (from find_person.py)
# ---------------------------------------------------------------------------
def load_face_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_s",
        root=os.environ["INSIGHTFACE_HOME"],
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_reference_face_embedding(app, ref_path):
    """Try to extract a face embedding from ref_path. Returns None if no
    face is found (i.e. this reference is NOT a person)."""
    img = cv2.imread(ref_path)
    if img is None:
        log(f"ERROR: Could not read reference image '{ref_path}'.")
        return None

    faces = app.get(img)
    if not faces:
        return None

    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    return faces[0].normed_embedding  # already L2-normalized, 512-d


def cosine_similarity(a, b):
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8))


def load_person_detector():
    from ultralytics import YOLO
    return YOLO(PERSON_MODEL_NAME)


def detect_people(model, frame):
    results = model.predict(
        frame, classes=[0], conf=PERSON_CONF_THRESHOLD,
        verbose=False, device="cpu",
    )
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def face_center_in_box(face_bbox, person_box):
    fx = (face_bbox[0] + face_bbox[2]) / 2
    fy = (face_bbox[1] + face_bbox[3]) / 2
    x1, y1, x2, y2 = person_box
    return x1 <= fx <= x2 and y1 <= fy <= y2


def scan_video_for_person(face_app, person_model, video_path, target_embedding):
    """Person-path scan (identical approach to find_person.py)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"  WARNING: Could not open video '{video_path}', skipping.")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(round(fps / SAMPLE_FPS)))

    all_tracks = []
    active_tracks = []
    next_track_id = 0

    frame_idx = 0
    checked = 0
    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue

            frame = _shrink_if_needed(frame)
            ts = frame_idx / fps
            checked += 1

            person_boxes = detect_people(person_model, frame)
            faces = face_app.get(frame)

            matched_person_indices = set()
            for face in faces:
                sim = cosine_similarity(face.normed_embedding, target_embedding)
                if sim > FACE_SIM_THRESHOLD:
                    for i, pbox in enumerate(person_boxes):
                        if face_center_in_box(face.bbox, pbox):
                            matched_person_indices.add(i)
                            break

            active_tracks = [
                t for t in active_tracks
                if ts - t.last_ts <= TRACK_REID_GAP_SECONDS
            ]

            assigned_track_ids_this_frame = set()

            for i, pbox in enumerate(person_boxes):
                embedding = compute_appearance_embedding(frame, pbox)
                best_track, best_score = None, 0.0

                for t in active_tracks:
                    if t.track_id in assigned_track_ids_this_frame:
                        continue
                    gap = ts - t.last_ts
                    score = 0.0
                    if gap <= TRACK_ACTIVE_GAP_SECONDS:
                        iou_score = iou(t.last_box, pbox)
                        if iou_score >= IOU_MATCH_THRESHOLD:
                            score = max(score, iou_score)
                    appearance_score = appearance_similarity(t.last_embedding, embedding)
                    if appearance_score >= APPEARANCE_SIM_THRESHOLD:
                        score = max(score, appearance_score)
                    if score > best_score:
                        best_track, best_score = t, score

                if best_track is not None:
                    best_track.update(pbox, embedding, ts)
                    assigned_track_ids_this_frame.add(best_track.track_id)
                    if i in matched_person_indices:
                        best_track.is_target = True
                else:
                    new_track = Track(next_track_id, pbox, embedding, ts)
                    if i in matched_person_indices:
                        new_track.is_target = True
                    all_tracks.append(new_track)
                    active_tracks.append(new_track)
                    assigned_track_ids_this_frame.add(next_track_id)
                    next_track_id += 1

        frame_idx += 1

    cap.release()
    return _finalize_hits(all_tracks, checked, duration)


# ---------------------------------------------------------------------------
# OBJECT path — ORB instance matching (from find_object.py)
# ---------------------------------------------------------------------------
class ReferenceObject:
    def __init__(self, keypoints, descriptors, histogram):
        self.keypoints = keypoints
        self.descriptors = descriptors
        self.histogram = histogram


def get_reference_object(ref_path, orb):
    """Build a ReferenceObject from ref_path. Returns None if the image has
    too few distinctive features to match reliably."""
    img = cv2.imread(ref_path)
    if img is None:
        log(f"ERROR: Could not read reference image '{ref_path}'.")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is None or len(keypoints) < 8:
        log(f"ERROR: Reference image has too few distinctive features to match "
            f"reliably ({0 if keypoints is None else len(keypoints)} found). "
            f"Use a sharper, well-lit, closely-cropped photo of the object.")
        return None

    histogram = compute_appearance_embedding_full(img)
    log(f"Reference object registered ({len(keypoints)} keypoints).")
    return ReferenceObject(keypoints, descriptors, histogram)


def load_object_detector():
    from ultralytics import YOLO
    return YOLO(OBJECT_MODEL_NAME)


def detect_objects(model, frame):
    kwargs = dict(conf=OBJECT_CONF_THRESHOLD, verbose=False, device="cpu")
    if TARGET_CLASS_NAMES:
        name_to_id = {v: k for k, v in model.names.items()}
        class_ids = [name_to_id[n] for n in TARGET_CLASS_NAMES if n in name_to_id]
        if class_ids:
            kwargs["classes"] = class_ids
    results = model.predict(frame, **kwargs)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
    return boxes


def match_object_instance(orb, matcher, reference, crop_gray):
    if crop_gray is None or crop_gray.size == 0:
        return False

    kp2, des2 = orb.detectAndCompute(crop_gray, None)
    if des2 is None or len(kp2) < 8:
        return False

    matches = matcher.knnMatch(reference.descriptors, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ORB_RATIO_TEST * n.distance:
                good.append(m)

    if len(good) < MIN_GOOD_MATCHES:
        return False

    src_pts = np.float32([reference.keypoints[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if mask is None:
        return False

    inliers = int(mask.sum())
    return inliers >= MIN_INLIERS


def scan_video_for_object(object_model, orb, matcher, reference, video_path):
    """Object-path scan (identical approach to find_object.py)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"  WARNING: Could not open video '{video_path}', skipping.")
        return [], 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    frame_step = max(1, int(round(fps / SAMPLE_FPS)))

    all_tracks = []
    active_tracks = []
    next_track_id = 0

    frame_idx = 0
    checked = 0
    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            ret, frame = cap.retrieve()
            if not ret:
                frame_idx += 1
                continue

            frame = _shrink_if_needed(frame)
            ts = frame_idx / fps
            checked += 1

            candidate_boxes = detect_objects(object_model, frame)

            matched_indices = set()
            for i, box in enumerate(candidate_boxes):
                x1, y1, x2, y2 = box
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(frame.shape[1], x2), min(frame.shape[0], y2)
                crop = frame[y1c:y2c, x1c:x2c]
                if crop.size == 0:
                    continue

                crop_hist = compute_appearance_embedding_full(crop)
                if appearance_similarity(reference.histogram, crop_hist) < HIST_PREFILTER_THRESHOLD:
                    continue

                crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if match_object_instance(orb, matcher, reference, crop_gray):
                    matched_indices.add(i)

            active_tracks = [
                t for t in active_tracks
                if ts - t.last_ts <= OBJECT_TRACK_REID_GAP_SECONDS
            ]

            assigned_track_ids_this_frame = set()

            for i, box in enumerate(candidate_boxes):
                embedding = compute_appearance_embedding(frame, box)
                best_track, best_score = None, 0.0

                for t in active_tracks:
                    if t.track_id in assigned_track_ids_this_frame:
                        continue
                    gap = ts - t.last_ts
                    score = 0.0
                    if gap <= OBJECT_TRACK_ACTIVE_GAP_SECONDS:
                        iou_score = iou(t.last_box, box)
                        if iou_score >= OBJECT_IOU_MATCH_THRESHOLD:
                            score = max(score, iou_score)
                    appearance_score = appearance_similarity(t.last_embedding, embedding)
                    if appearance_score >= OBJECT_APPEARANCE_SIM_THRESHOLD:
                        score = max(score, appearance_score)
                    if score > best_score:
                        best_track, best_score = t, score

                if best_track is not None:
                    best_track.update(box, embedding, ts)
                    assigned_track_ids_this_frame.add(best_track.track_id)
                    if i in matched_indices:
                        best_track.is_target = True
                else:
                    new_track = Track(next_track_id, box, embedding, ts)
                    if i in matched_indices:
                        new_track.is_target = True
                    all_tracks.append(new_track)
                    active_tracks.append(new_track)
                    assigned_track_ids_this_frame.add(next_track_id)
                    next_track_id += 1

        frame_idx += 1

    cap.release()
    return _finalize_hits(all_tracks, checked, duration)


# ---------------------------------------------------------------------------
# Shared: Track class, appearance embedding, IoU, hit finalization
# ---------------------------------------------------------------------------
class Track:
    __slots__ = ("track_id", "last_box", "last_embedding", "last_ts",
                 "timestamps", "is_target")

    def __init__(self, track_id, box, embedding, ts):
        self.track_id = track_id
        self.last_box = box
        self.last_embedding = embedding
        self.last_ts = ts
        self.timestamps = [ts]
        self.is_target = False

    def update(self, box, embedding, ts):
        self.last_box = box
        self.last_embedding = embedding
        self.last_ts = ts
        self.timestamps.append(ts)


def compute_appearance_embedding_full(image_bgr):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def compute_appearance_embedding(frame, box):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return compute_appearance_embedding_full(crop)


def appearance_similarity(hist_a, hist_b):
    if hist_a is None or hist_b is None:
        return 0.0
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _finalize_hits(all_tracks, checked, duration):
    hit_timestamps = []
    for t in all_tracks:
        if t.is_target:
            hit_timestamps.extend(t.timestamps)
    hit_timestamps = sorted(set(round(t, 2) for t in hit_timestamps))

    target_tracks = sum(1 for t in all_tracks if t.is_target)
    log(f"  Sampled {checked} frames, {len(all_tracks)} track(s) formed, "
        f"{target_tracks} confirmed as target, {len(hit_timestamps)} hit instant(s).")
    return hit_timestamps, duration


def _shrink_if_needed(frame):
    h, w = frame.shape[:2]
    if w > MAX_ANALYSIS_WIDTH:
        scale = MAX_ANALYSIS_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


# ---------------------------------------------------------------------------
# Interval merging
# ---------------------------------------------------------------------------
def get_extraction_clips(hit_timestamps, sample_step_seconds=1.0 / SAMPLE_FPS,
                          max_gap_seconds=MAX_GAP_SECONDS, video_length=0,
                          padding_seconds=PADDING_SECONDS):
    """
    Turn hit timestamps into merged [start, end] windows, each padded with
    `padding_seconds` of extra context before and after the actual
    appearance. e.g. person appears at 10s and 30s, padding_seconds=15 ->
    clip 1 = [0s (clamped), 25s], clip 2 = [15s, 45s]. Windows that overlap
    once padded (or are within max_gap_seconds of each other) are merged
    into a single clip instead of being extracted twice.
    """
    if not hit_timestamps:
        return []

    half_step = sample_step_seconds / 2.0

    # Step 1: raw per-instant windows (unpadded), same as before.
    raw_intervals = []
    for t in sorted(hit_timestamps):
        start = max(0, t - half_step)
        end = min(video_length, t + half_step) if video_length else t + half_step
        raw_intervals.append([start, end])

    # Step 2: merge raw windows that are close together into one appearance
    # window BEFORE padding, so one continuous appearance doesn't get
    # padded/split into several overlapping clips.
    merged_raw = [raw_intervals[0]]
    for current in raw_intervals[1:]:
        previous = merged_raw[-1]
        if current[0] <= previous[1] + max_gap_seconds:
            previous[1] = max(previous[1], current[1])
        else:
            merged_raw.append(current)

    # Step 3: pad each appearance window with extra context, clamped to the
    # video bounds.
    padded = []
    for start, end in merged_raw:
        p_start = max(0, start - padding_seconds)
        p_end = min(video_length, end + padding_seconds) if video_length else end + padding_seconds
        padded.append([p_start, p_end])

    # Step 4: padded windows may now overlap each other (e.g. two
    # appearances 20s apart with 15s padding each) — merge those into one
    # continuous clip so the same footage isn't extracted twice.
    final = [padded[0]]
    for current in padded[1:]:
        previous = final[-1]
        if current[0] <= previous[1]:
            previous[1] = max(previous[1], current[1])
        else:
            final.append(current)

    return final


# ---------------------------------------------------------------------------
# FFmpeg extraction + merging
# ---------------------------------------------------------------------------
def seconds_to_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_clip(video_path, start, end, out_path):
    cmd = [
        "ffmpeg", "-y",
        "-ss", seconds_to_timestamp(start),
        "-to", seconds_to_timestamp(end),
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        log(f"  WARNING: FFmpeg stream-copy failed for {out_path}. Retrying with re-encode...")
        fallback_cmd = [
            "ffmpeg", "-y",
            "-ss", seconds_to_timestamp(start),
            "-to", seconds_to_timestamp(end),
            "-i", video_path,
            out_path,
        ]
        subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def merge_clips(clip_paths, merged_out_path):
    if not clip_paths:
        return

    list_fd, list_path = tempfile.mkstemp(suffix=".txt")
    try:
        with os.fdopen(list_fd, "w", encoding="utf-8") as f:
            for p in clip_paths:
                escaped = os.path.abspath(p).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            merged_out_path,
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            log(f"  WARNING: Fast merge failed for {merged_out_path}. Retrying with re-encode...")
            fallback_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", list_path,
                merged_out_path,
            ]
            subprocess.run(fallback_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        os.remove(list_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("Loading InsightFace (buffalo_s, CPU)...")
    face_app = load_face_app()

    log("Loading YOLOv8-nano detector (shared for person + object paths)...")
    person_model = load_person_detector()
    object_model = person_model  # same weights file; reused for both paths

    orb = cv2.ORB_create(nfeatures=ORB_N_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    reference_paths = find_reference_images(REFERENCE_DIR)
    log(f"Found {len(reference_paths)} reference image(s): "
        f"{', '.join(os.path.basename(p) for p in reference_paths)}")

    video_paths = []
    for ext in VIDEO_EXTENSIONS:
        video_paths.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext}")))
        video_paths.extend(glob.glob(os.path.join(INPUT_DIR, f"*{ext.upper()}")))
    video_paths = sorted(set(video_paths))

    if not video_paths:
        log(f"ERROR: No video files found in '{INPUT_DIR}/'.")
        sys.exit(1)

    log(f"Found {len(video_paths)} video(s) to process.\n")

    # Outer loop: each reference image gets its own output folder, named
    # after the reference image file (without extension). It is first
    # auto-classified as PERSON (face detected in the reference photo) or
    # OBJECT (no face detected), then scanned with the matching approach
    # across every input video. Clips are numbered in processing order.
    for ref_path in reference_paths:
        ref_name = os.path.splitext(os.path.basename(ref_path))[0]
        log(f"\n=== Reference: '{ref_name}' ({ref_path}) ===")

        # --- Auto-detect: person (face found) vs object (no face) ---
        target_embedding = get_reference_face_embedding(face_app, ref_path)

        if target_embedding is not None:
            mode = "person"
            log(f"  Detected a face in '{ref_name}' -> treating as PERSON.")
            reference = target_embedding
        else:
            mode = "object"
            log(f"  No face detected in '{ref_name}' -> treating as OBJECT.")
            reference = get_reference_object(ref_path, orb)
            if reference is None:
                log(f"  Skipping reference '{ref_name}' (not usable as a face "
                    f"or an object reference).\n")
                continue

        ref_out_dir = os.path.join(OUTPUT_DIR, ref_name)
        os.makedirs(ref_out_dir, exist_ok=True)

        clip_paths = []
        clip_counter = 1

        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            log(f"  Processing video '{video_name}' for '{ref_name}' ({mode})...")

            if mode == "person":
                hit_timestamps, duration = scan_video_for_person(
                    face_app, person_model, video_path, reference
                )
            else:
                hit_timestamps, duration = scan_video_for_object(
                    object_model, orb, matcher, reference, video_path
                )

            if not hit_timestamps:
                log(f"    No appearances of '{ref_name}' found in '{video_name}'.")
                continue

            clips = get_extraction_clips(hit_timestamps, video_length=duration)
            log(f"    Merged into {len(clips)} clip(s).")

            for start, end in clips:
                out_path = os.path.join(ref_out_dir, f"clip_{clip_counter:02d}.mp4")
                log(f"    Extracting clip {clip_counter} ({video_name}): "
                    f"{start:.2f}s -> {end:.2f}s")
                extract_clip(video_path, start, end, out_path)
                clip_paths.append(out_path)
                clip_counter += 1

        if not clip_paths:
            log(f"  No appearances of '{ref_name}' found in any video.\n")
            continue

        log(f"  Done. {len(clip_paths)} clip(s) saved to '{ref_out_dir}/'.")

        merged_path = os.path.join(OUTPUT_DIR, f"{ref_name}_merged.mp4")
        log(f"  Merging {len(clip_paths)} clip(s) into '{merged_path}'...")
        merge_clips(clip_paths, merged_path)
        log(f"  Merged video saved to '{merged_path}'.\n")

    log("All references processed.")


if __name__ == "__main__":
    main()