# Person Finder — Face + Motion Tracking Clip Extractor

Finds every appearance of a target person in one or more videos — using the
person's face to confirm identity, and body detection + tracking to follow
them through stretches where their face isn't visible — and exports the
matching footage as clips, plus one merged video per source video.

## Stack
- **Face identity confirmation:** InsightFace `buffalo_s` (ONNX Runtime, CPU)
- **Person detection:** YOLOv8-nano (Ultralytics, CPU)
- **Tracking / re-identification:** bounding-box overlap (motion continuity)
  + HSV color-histogram appearance matching (reconnects a track across brief
  occlusions/cuts)
- **Video reading:** OpenCV
- **Clip extraction & merging:** FFmpeg stream-copy (`-c copy`, no re-encoding)

## Folder layout
```
person_finder/
├── find_person.py
├── requirements.txt
├── reference/   <- put ONE clear photo of the target person here
├── inputs/      <- put video files here (.mp4, .mov, .mkv, .avi, .m4v, .webm)
└── outputs/
      <video_name>/clip_01.mp4, clip_02.mp4, ...   <- individual clips
      <video_name>_merged.mp4                       <- all clips joined, one level up
```

## Setup
```bash
pip install -r requirements.txt
```
FFmpeg must also be installed and on your PATH (`ffmpeg -version` to check).

First run will download two small model files automatically:
- InsightFace `buffalo_s` → wherever `INSIGHTFACE_HOME` points (defaults to
  `D:\insightface_models` in the script — edit `DEFAULT_MODEL_ROOT` at the
  top of `find_person.py` if you want it elsewhere)
- `yolov8n.pt` (~6MB) → the folder you run the script from

## Usage
1. Drop a clear, front-facing photo of the target into `reference/`.
2. Drop the video(s) you want to search into `inputs/`.
3. Run:
   ```bash
   python find_person.py
   ```
4. Check `outputs/<video_name>/clip_01.mp4`, `clip_02.mp4`, ... for individual
   appearances, and `outputs/<video_name>_merged.mp4` for everything joined
   into one file.

## How it works
1. **Register identity** — the reference photo is passed through InsightFace
   once to get a 512-d face embedding.
2. **Sample, detect, track** — the video is sampled (default 2 frames/sec).
   Each sampled frame:
   - YOLOv8-nano finds every person's bounding box.
   - InsightFace finds every face and checks it against the reference
     embedding (cosine similarity threshold 0.5).
   - Each detected person box is linked to an existing "track" using box
     overlap (if seen very recently) or appearance-histogram similarity (if
     the track had a short gap, e.g. from occlusion or a camera cut) — this
     is what lets the same person be followed even when their face isn't
     visible.
   - **A track is only ever confirmed as the target because its owner's face
     matched the reference at some point** — appearance similarity is only
     used to reconnect a *broken* track back to itself, never to assign
     identity from scratch. This keeps the strict "only this person, not
     others" guarantee even though the pipeline now tracks bodies.
3. **Collect hits** — every timestamp belonging to a confirmed target track
   becomes a "hit," including the parts of the track where the face wasn't
   visible.
4. **Merge intervals** — consecutive/near-consecutive hits are merged into
   continuous windows (no padding is added — only actual presence).
5. **Extract & merge** — FFmpeg slices each window with `-c copy` into
   `outputs/<video_name>/clip_NN.mp4`, then concatenates all of that video's
   clips into `outputs/<video_name>_merged.mp4`.

## Tuning knobs (top of `find_person.py`)
| Variable | Default | Effect |
|---|---|---|
| `SAMPLE_FPS` | 2 | Frames analyzed per second of footage |
| `FACE_SIM_THRESHOLD` | 0.50 | Cosine similarity cutoff to confirm identity via face |
| `PERSON_CONF_THRESHOLD` | 0.40 | Min YOLO confidence to count as a person |
| `IOU_MATCH_THRESHOLD` | 0.30 | Min box overlap to continue a track frame-to-frame |
| `APPEARANCE_SIM_THRESHOLD` | 0.70 | Min histogram similarity to reconnect a broken track |
| `TRACK_ACTIVE_GAP_SECONDS` | 2.0 | How recently a track must've been seen for IOU matching |
| `TRACK_REID_GAP_SECONDS` | 15.0 | Max gap before a track is no longer eligible for re-id at all |
| `MAX_GAP_SECONDS` | 1.5 | How close two hit windows must be to merge into one clip |
| `MAX_ANALYSIS_WIDTH` | 1280 | Frames wider than this are downscaled before detection |

**If the target is being lost across cuts/occlusion too often:** lower
`APPEARANCE_SIM_THRESHOLD` slightly (e.g. 0.6) or raise `TRACK_REID_GAP_SECONDS`.

**If two different people (e.g. similar clothing) are getting merged into
one track:** raise `APPEARANCE_SIM_THRESHOLD` (e.g. 0.8) and/or lower
`TRACK_REID_GAP_SECONDS` so re-identification is stricter.

**If it's too slow on your hardware:** lower `SAMPLE_FPS` back to 1, or drop
`MAX_ANALYSIS_WIDTH` to 960/720.

## Notes
- This uses a lightweight color-histogram descriptor for body re-identification
  rather than a full deep ReID network, to keep CPU/hardware requirements low.
  It's good at reconnecting a track through brief gaps but isn't a substitute
  for a full person-ReID model in scenes with many similarly-dressed people —
  identity is always ultimately anchored by the face match, never appearance
  alone.
- If a video has no confirmed hits, it's skipped — no output folder or merged
  file is created for it.
- `-c copy` cuts on the nearest keyframe, so clip boundaries can be off by up
  to ~1-2s; irrelevant for most investigative use but worth knowing.

---

## Matching a specific OBJECT instead of a person

Use **`find_object.py`** (same folder, same `reference/` / `inputs/` /
`outputs/` layout) if you need to find a *specific object instance* — e.g.
this exact bag, not just "a bag" — rather than a person.

- **Detection:** YOLOv8-nano's general 80-class COCO detector proposes
  candidate object boxes per frame (covers bags, suitcases, bottles,
  laptops, phones, cars, knives, books, etc.). If your object isn't in one
  of those 80 classes, it won't be proposed as a candidate at all.
- **Identity confirmation:** ORB keypoint/descriptor matching + a ratio test
  + RANSAC homography verification against the reference photo — this
  checks the object's actual distinctive local features (texture, edges,
  print/pattern), not just its color, so it's much stricter than "looks
  similar." No extra install needed — ORB is built into `opencv-python`.
- **Tracking/re-ID and merged output** work identically to `find_person.py`.

Tuning knobs specific to `find_object.py`:
| Variable | Default | Effect |
|---|---|---|
| `MIN_GOOD_MATCHES` | 12 | Min ORB ratio-test matches before attempting verification |
| `MIN_INLIERS` | 8 | Min RANSAC-verified inliers to confirm the same instance |
| `TARGET_CLASS_NAMES` | `None` | Restrict candidates to specific COCO classes (e.g. `{"backpack"}`) for speed/precision |

**Reference photo matters a lot here** — crop it tightly to the object,
shoot it well-lit and in reasonably sharp focus, and avoid objects with very
plain, texture-less surfaces (a flat single-color surface has few keypoints
to match on, so ORB will struggle). Highly repetitive patterns (checkerboards,
uniform grids) can also confuse keypoint matching since every region looks
similar — a distinctive logo, label, texture, or asymmetric shape works best.