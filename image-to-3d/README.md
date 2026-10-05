# Image → 3D Studio
Requirements: NVIDIA GPU (~24 GB VRAM recommended), NVIDIA Container Toolkit, Docker Compose v2.

1. (Optional, for pose-accurate people) register at https://smpl-x.is.tue.mpg.de, download SMPL-X and place
   `SMPLX_NEUTRAL.npz` at `models/smpl/smplx/SMPLX_NEUTRAL.npz`.
2. `docker compose build` (long: compiles CUDA extensions) then `docker compose up`
3. Open http://localhost:8080

First generation per engine downloads weights into `models/hf` (many GB).
Set `KEEP_LOADED: "1"` for a worker in docker-compose.yml to keep it in VRAM between jobs.
