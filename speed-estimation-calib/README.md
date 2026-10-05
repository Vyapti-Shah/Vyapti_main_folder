# Speed Estimation

Pretrained YOLO detector → ByteTrack → camera calibration (pixel → metres) → speed = distance / time.
No neural network predicts speed; the AI only answers *where* each object is.

## Run
```bash
docker compose build
docker compose up
# open http://localhost:8080
```
First build downloads the YOLO weights (default `yolo11s.pt`). Other size: `YOLO_MODEL=yolo11m.pt docker compose build`.

## Flow
1. Upload a video (an image gives detections only – speed needs time).
2. Pick a calibration mode, press **Calibrate**, review the summary, press **Start Detection**.
3. Watch live speeds, then get the annotated video, a per-object table and CSV files.

## Calibration modes
| Mode | Use when | Accuracy |
|---|---|---|
| **Known-distance calibration** (recommended) | Fixed/CCTV camera, you know distances on the road | High |
| Known camera parameters | You know camera height, tilt and field of view / focal length | Medium |
| Automatic calibration (experimental) | Nothing is known; scale comes from typical object heights | Low |

Mode 1 geometry: Line A = P1→P2, Line B = P4→P3. You enter |P1P2|, |P4P3| and the distance between the two
lines **measured along the road** (P1→P4). Lines are placed symmetrically about the road axis in the world
frame (exact when both lines have the same length), and a homography maps image → metres.

## How speed is computed
* Position = bottom-centre of the box (ground contact) → metres via the homography.
* Speed = least-squares slope of position vs. time over a sliding window (default 1 s) – the noise-robust
  form of `distance / time`; time = frame index / FPS.
* Mode 1 also logs when each object crosses Line A and Line B (sub-frame interpolation) and reports
  gate speed = real distance between the two crossing points / time between crossings.

## Limits (be honest with your numbers)
* Needs a **fixed camera** and objects moving on a **flat ground plane**. A moving-camera warning is shown, not compensated.
* Detections cut by the bottom frame edge are ignored (ground contact unknown).
* Accuracy drops far from the calibrated area and near the horizon.
