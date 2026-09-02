# DeepFake / AI-Generated Media Detector

A full-stack web app that detects whether an uploaded **image or video** is real or AI-generated, and returns the media watermarked accordingly:
- **REAL** (green text) if the content is classified as authentic
- **AI Generated** (red text) if classified as fake

## How it works

### Architecture

Two containers, started together by `docker compose up --build`:

- **frontend** (React + Vite, served by nginx on port 3043) - upload UI
- **backend** (FastAPI on port 8043, bound to `127.0.0.1` only - not reachable from your LAN) - does all the detection

The frontend's nginx proxies `/api/*` to the backend over Docker's internal network, so the browser only ever talks to port 3043.

### Request flow

1. **Upload**: the browser sends the file to `POST /api/detect` (multipart), saved to `data/uploads/`.
2. **Routing** (`routers/detection.py`): the file extension decides the image vs. video path.
3. **Result**: the watermarked output is saved to `data/outputs/` and served back via `GET /api/result/{filename}`; the frontend renders it with a REAL/AI Generated badge.

### Detection pipeline

For every frame (a still image, or each sampled video frame), `DeepFakeService.predict()` runs an ensemble of three signals and combines them:

1. **Face-swap detector**: MTCNN finds the largest face in the frame, crops it with a `FACE_MARGIN` (default 30%) margin - so blending-edge artifacts at the crop boundary are included - and classifies that crop with EfficientNet-B0 (`efficientnet_b0_ffpp_c23.pth`, trained on FaceForensics++ face-swap forgeries). Falls back to classifying the whole frame if no face is found.
2. **General AI-image detector**: a Swin Transformer (`Organika/sdxl-detector`, pulled from Hugging Face and baked into the Docker image at build time) runs on the *whole* frame, catching fully-synthetic diffusion-generated images that a face-swap detector was never trained to recognize.
3. **Forensic heuristics** (`forensic_heuristics.py`): missing camera EXIF metadata and unnaturally smooth texture each nudge the general detector's score by a small, bounded amount - enough to tip an already-borderline case, never enough to flip a confident verdict on their own.

The two models' fake-probabilities are **averaged** to get the final score (not "either model triggers fake") - averaging proved far more robust than OR-logic, which let one noisy/borderline model override a confident, correct call from the other.

### Image vs. video

- **Image**: the ensemble above runs once; the original image gets a REAL/AI Generated text watermark burned into the top-left corner.
- **Video** (`video_processor.py`): OpenCV samples frames at a target rate (`FRAME_SAMPLE_FPS`, default 10 fps) converted to a frame-interval based on the source video's actual fps - not every frame is classified. Each sampled frame runs through the same ensemble; its verdict is carried forward to the frames in between so the watermark stays continuous instead of flickering. The whole video is flagged fake if the fraction of fake frames reaches `FAKE_FRAME_THRESHOLD` (default 60%). Output is re-encoded via an `ffmpeg` subprocess to H.264/yuv420p - OpenCV's own `VideoWriter` produces a codec (MPEG-4 Part 2) that no browser can play.

### Performance

Everything runs on CPU by default (no GPU configured). Images are near-instant (sub-second); video is the expensive path since two deep models run per sampled frame - a 10s/720p clip takes roughly 60-90 seconds on a modern multi-core machine. A GPU would speed this up substantially if available.

## Quick Start

```bash
docker compose up --build
```

Then open **http://localhost:3043** in your browser, upload an image or video, and view the watermarked result.

- Frontend: http://localhost:3043
- Backend API: http://localhost:8043 (docs at http://localhost:8043/docs) — bound to `127.0.0.1` only, not reachable from your LAN. The frontend talks to it over Docker's internal network regardless.

## Layout

```
├── docker-compose.yml        Orchestrates backend + frontend containers
├── run.sh                    Local dev runner (backend --reload + vite dev)
├── models/                   EfficientNet-B0 model weights (.pth)
├── data/
│   ├── uploads/               Uploaded source images/videos
│   └── outputs/                Watermarked results
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app/
│       ├── main.py             FastAPI app entrypoint, lifespan, /api/health
│       ├── config.py            Settings (env-var driven)
│       ├── models.py            Pydantic response/request models
│       ├── routers/
│       │   └── detection.py      /api/detect, /api/result/{filename}
│       └── services/
│           ├── detection_service.py     Ensemble orchestration: MTCNN crop + EfficientNet-B0 + general detector + watermarking
│           ├── general_ai_detector.py    Swin Transformer whole-image real-vs-AI classifier
│           ├── forensic_heuristics.py    EXIF + texture-smoothness signals
│           └── video_processor.py        fps-based frame sampling, per-frame classification, temporal aggregation
└── frontend/
    ├── Dockerfile, nginx.conf
    └── src/
        ├── main.tsx             React entrypoint
        ├── App.tsx               Upload UI, progress, result display
        ├── api.ts                 Backend API client
        └── styles.css
```

## Local Development (without Docker)

```bash
# Backend deps
cd backend && pip install -r requirements.txt && cd ..

# Frontend deps
cd frontend && npm install && cd ..

# Run both (backend on :8000, frontend on :5173)
./run.sh
```

## API

### `POST /api/detect`
Upload a file (`multipart/form-data`, field name `file`). Supports images (jpg, png, gif, bmp) and videos (mp4, avi, mov, mkv, webm).

Response:
```json
{
  "file_id": "uuid",
  "file_name": "example.mp4",
  "file_type": "video",
  "is_fake": false,
  "confidence": 0.94,
  "output_path": "/api/result/uuid_result.mp4",
  "created_at": "2026-09-02T12:00:00Z",
  "frames_analyzed": 42
}
```

### `GET /api/result/{filename}`
Serves the watermarked output file.

## Configuration

Environment variables (set in `docker-compose.yml`):

| Variable | Default | Description |
|---|---|---|
| `MODEL_PATH` | `/app/models/efficientnet_b0_ffpp_c23.pth` | Path to model weights |
| `FAKE_CLASS_INDEX` | `1` | Model output index representing "Fake" |
| `FRAME_SAMPLE_FPS` | `10` | Target frames/sec to sample and classify in videos |
| `FACE_MARGIN` | `0.3` | Margin (fraction of box size) added around each detected face before classifying |
| `FAKE_FRAME_THRESHOLD` | `0.6` | Fraction of analyzed video frames that must be flagged fake to flag the whole video |
| `GENERAL_DETECTOR_MODEL` | `Organika/sdxl-detector` | Hugging Face model id for the whole-image real-vs-AI classifier |

## Models

Two models are used together (see [Detection pipeline](#detection-pipeline) above):

1. **`models/efficientnet_b0_ffpp_c23.pth`** - EfficientNet-B0 trained on faces (FaceForensics++ c23). Catches face-swap/reenactment forgery. To use your own EfficientNet-B0 binary classifier, replace the file and update `MODEL_PATH` if the filename changes. Other options:
   - [Xicor9/efficientnet-b0-ffpp-c23](https://huggingface.co/Xicor9/efficientnet-b0-ffpp-c23/tree/main)
   - [divyanshu-chauhan-7786/deepfake_image](https://huggingface.co/divyanshu-chauhan-7786/deepfake_image)
2. **`Organika/sdxl-detector`** - a Swin Transformer classifying whole images as real vs. AI-generated, fetched from Hugging Face and baked into the backend image at build time (`ENV HF_HUB_OFFLINE=1` after that, so it never hits the network at runtime). Catches fully-synthetic diffusion-generated images the face-swap model alone would miss. Swap it via `GENERAL_DETECTOR_MODEL` for a different Hugging Face `image-classification` model with an `{"artificial", "human"}` label set - update the pre-download step in `backend/Dockerfile` too if you do.

The FF++ model alone confidently misclassified fully-AI-generated (non-face-swap) test images as real - it was trained for a different task (face-swap detection) than general AI-image detection. The ensemble above was built specifically to close that gap.
