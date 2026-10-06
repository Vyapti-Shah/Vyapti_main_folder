# Image → 3D Studio (ARM64 / Blackwell, e.g. DGX Spark)
Requirements: NVIDIA GPU, NVIDIA Container Toolkit, Docker Compose v2.
Base images: nvcr.io/nvidia/pytorch:25.09-py3 (multi-arch; change the tag if it can't be pulled).

1. (Optional, pose-accurate people) register at https://smpl-x.is.tue.mpg.de and place
   `SMPLX_NEUTRAL.npz` at `models/smpl/smplx/SMPLX_NEUTRAL.npz`.
2. Build in stages:
   docker compose build backend frontend
   docker compose build hunyuan
   docker compose up backend frontend hunyuan
3. Open http://localhost:8080

TRELLIS is opt-in and experimental on ARM/Blackwell (spconv/kaolin build from source):
   docker compose --profile trellis build trellis
   docker compose --profile trellis up

Weights download on first use into `models/hf`.
