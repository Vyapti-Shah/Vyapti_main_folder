# Deepfake Detection – Optical-Flow Clip Selection

    docker compose build
    docker compose up

| Service  | URL                          | Host port |
|----------|------------------------------|-----------|
| Frontend (dashboard) | http://localhost:3000 | 3000 |
| Backend (FastAPI)    | http://localhost:8000 | 8000 |
| Swagger docs         | http://localhost:8000/docs | 8000 |

API: POST /detect/image, POST /detect/video, GET /analysis/{id}, GET /health

Pipeline: 5 FPS sampling -> face detect -> IoU track -> quality check (blur/size/brightness)
-> Farneback flow on face crop (3x3 regions) -> dedupe + peak + normal-motion selection
-> temporal clips -> EfficientNet-B0 + BiLSTM -> quality-weighted aggregation -> dashboard.

## IMPORTANT: trained weights
No weights ship with this project. Put your checkpoint at `backend/models/deepfake_model.pt`
(a state_dict of `app.model.DeepfakeNet`). Without it the app runs in DEMO MODE and reports UNVERIFIED.
