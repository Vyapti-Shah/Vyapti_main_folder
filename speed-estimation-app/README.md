# Forensic Speed Estimation

Video → object detection (YOLO) → tracking (ByteTrack, optional SAM2 mask
refinement) → ground-plane homography → windowed speed → Monte Carlo
uncertainty → annotated video + report + charts.

## Run it

CPU (works everywhere, slower):
```
docker compose up --build
```

GPU (NVIDIA, needs the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host):
```
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Then open **http://localhost:5173**. The backend API is on **http://localhost:8000/api** (also reverse-proxied under `/api` by the frontend's nginx).

## How to use it

**Just upload a video.** Processing starts automatically — no calibration
points, class selection, or any other setup required. Moving objects are
detected, tracked, and their speed calculated using each object's known
average real-world size for its class (car, person, bus, etc.) as the
pixel→meter scale, recomputed every frame so it adapts as the object moves
closer to or further from the camera.

Once done you get: the annotated video (boxes + live speed ± uncertainty),
a per-object table (distance, duration, mean speed, 95% CI, tracking
confidence), and speed-vs-time charts per object.

### Trade-off you should know about

Automatic mode has no real-world reference, so the scale is a *population
average* for each object's class (e.g. "cars are ~4.5 m long") rather than
a measurement of the actual object or scene. That's the dominant source of
uncertainty in automatic mode, and it's sized into the reported 95% CI
rather than hidden — expect wider intervals than a properly calibrated
setup would give you. If you need tighter, defensible numbers (e.g. for
evidentiary use), the API still accepts a manual ground-plane calibration:
`POST /api/process` with `"calibration_mode": "manual"` plus 4+
`image_points`/`world_points` pairs measured from the actual scene, which
uses a homography instead of the class-size assumption and is
substantially more accurate. The current frontend only exposes the
automatic, zero-input flow; wire up a calibration UI against that same
endpoint if you want to expose manual mode too.

You can also pass `known_speeds_kmh` (`{"track_id": actual_speed_kmh}`) in
the `/api/process` body to unlock the accuracy-evaluation panel
(MAE/RMSE/MAPE/bias/CI coverage) against ground truth, and `target_classes`
to restrict detection to specific COCO classes — both optional, the
frontend doesn't ask for either.

## Why the results are defensible, not just numbers

- **Automatic mode** uses per-frame, per-object known-class-size scaling
  (not a single flat meters-per-pixel guess for the whole frame), so
  perspective is still accounted for as the object moves through the scene.
  **Manual mode** (available via the API) uses a proper ground-plane
  homography, whose reprojection RMSE is reported so you know how tight the
  calibration itself is (`calibration_rmse_px` in the report).
- **Windowed speed estimation** (default 15-frame sliding window) plus
  Savitzky–Golay trajectory smoothing, instead of a raw two-frame diff, so
  detection jitter doesn't dominate.
- **MAD-based outlier rejection** on the windowed speed series before it's
  summarized or charted (a single bad detection won't produce a 200 km/h
  spike in the report).
- **Monte Carlo uncertainty propagation** (5,000 samples by default) over
  detection-pixel jitter, calibration error, and timestamp/FPS jitter,
  reported as mean ± std and a 95% CI rather than a bare point estimate.
- **Evaluation harness** (MAE/RMSE/MAPE/bias/CI coverage, actual-vs-estimated
  scatter) built in whenever you supply known speeds, so you can validate the
  pipeline on a controlled test set before trusting it on real evidence.

## Tuning accuracy

Environment variables on the `backend` service (see `docker-compose.yml`):

| Variable | Default | Effect |
|---|---|---|
| `YOLO_MODEL` | `yolo11n.pt` | Swap for `yolo11s.pt` / `yolo11m.pt` for better detection accuracy at the cost of speed |
| `TRACKER_CFG` | `bytetrack.yaml` | Ultralytics tracker config |
| `USE_SAM2` | `0` | Enable SAM2 (`facebook/sam2-hiera-small-hf`) ground-contact refinement — GPU strongly recommended, roughly doubles processing time |
| `SPEED_WINDOW_FRAMES` | `15` | Larger = smoother but less time-resolved speed |
| `MC_SAMPLES` | `5000` | Monte Carlo samples for the uncertainty interval |
| `DEFAULT_BBOX_JITTER_PX`, `DEFAULT_CALIB_JITTER_PX`, `DEFAULT_TIMESTAMP_JITTER_S` | `2.5`, `3.0`, `0.003` | Assumed measurement noise if you don't override them per-request |

Per-request overrides (`bbox_jitter_px`, `calib_jitter_px`, `timestamp_jitter_s`,
`use_sam2`) can also be sent in the `/api/process` body from the frontend's
options panel or directly via the API.

## Project layout

```
backend/            FastAPI service
  app/main.py        API routes
  app/jobs.py         orchestrates the full pipeline per request
  app/pipeline/
    tracker.py         YOLO + ByteTrack detection/tracking
    sam2_refine.py      optional SAM2 ground-point refinement
    calibration.py      homography (image px -> world m)
    speed.py             windowed speed + smoothing + outlier rejection
    uncertainty.py       Monte Carlo uncertainty propagation
    annotate.py           renders the annotated output video
    report.py              charts + evaluation metrics
frontend/            React (Vite) UI, served by nginx which also proxies /api
docker-compose.yml    CPU setup
docker-compose.gpu.yml  GPU override (CUDA base image + SAM2 on)
```

## Notes / limitations

- Calibration assumes a **planar** ground surface, consistent with your
  requirement — it will silently give wrong distances if the tracked object
  leaves that plane (e.g. a ramp, stairs).
- The default YOLO model detects COCO classes only (people/vehicles/etc.);
  for domain-specific objects you'd need a fine-tuned `.pt` and can just
  swap `YOLO_MODEL`.
- SAM2 mask refinement is optional and off by default specifically because
  its accuracy gain is marginal for ordinary vehicle/person bottom-center
  points and only clearly helps with tall/occluded objects — turn it on via
  `USE_SAM2=1` (GPU) if your footage has that problem.
- This is a measurement tool, not a certification — always validate against
  a known-speed test set (`known_speeds_kmh`) for your specific camera
  placement before treating outputs as courtroom-ready evidence.
