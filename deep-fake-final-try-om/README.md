# Deepfake Detection Studio

An image and video AI-generated-content detector built on
[`buildborderless/CommunityForensics-DeepfakeDet-ViT`](https://huggingface.co/buildborderless/CommunityForensics-DeepfakeDet-ViT)
(ViT-Small, single-logit sigmoid output), with independent image and video workflows and a
React frontend. The entire app runs with one command via Docker Compose — no local Python,
Node, or ML dependencies needed.

**Important scope note:** the model detects *AI-generated images* (images produced end-to-end
by a generative model). It is **not** a face-swap or partial-manipulation detector. See
[`docs/evaluation-and-limitations.md`](docs/evaluation-and-limitations.md) for the full
limitations list, including the lack of a calibration claim.

## Prerequisites

- Docker and Docker Compose v2 (`docker compose`, not the old standalone `docker-compose`).
  Tested with Docker 29.2.1 / Compose v5.0.2.
- ~2 GB free disk for the model cache (Docker volume) plus image layers.
- Internet access on first run to download the model from Hugging Face (not needed again
  afterwards — see [Model caching](#model-caching)).
- No local Python, Node, PyTorch, FastAPI, or npm install required.

## Start

```bash
docker compose up --build
```

- Frontend: http://localhost:3100
- Backend API: http://localhost:8000
- Interactive API docs (Swagger UI): http://localhost:8000/docs

First run downloads the ~85 MB model into a persistent Docker volume (`model-cache`); this
takes well under a minute on a normal connection and the backend reports itself not-ready
until it finishes (see [Health checks](#health-checks--startup-order)). Subsequent restarts
reuse the cached model and the backend becomes ready in under a second — verified in testing
(13.9s on a cold cache vs 0.7s on a warm one).

Run detached with `docker compose up --build -d` and follow logs with `docker compose logs -f`.

## Shutdown

```bash
docker compose down          # stop and remove containers, keep the model cache volume
docker compose down -v       # also delete the model cache (next start re-downloads the model)
```

## Project layout

```
backend/    FastAPI app (image/video detection, model loading, evaluation script, tests)
frontend/   React + Vite UI, served by nginx, built into the production image
docs/       evaluation methodology and limitations
docker-compose.yml, .env.example
```

## Environment variables

Copy `.env.example` to `.env` to override any of these; all have working defaults.

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_ID` | `buildborderless/CommunityForensics-DeepfakeDet-ViT` | Hugging Face model to load |
| `MODEL_REVISION` | `main` | Pin to a commit hash for reproducible builds |
| `TORCH_NUM_THREADS` | `0` (let torch pick) | Cap CPU threads used for inference |
| `BACKEND_PORT` | `8000` | Host port for the API |
| `FRONTEND_PORT` | `3100` | Host port for the UI (not `3000` — see [Troubleshooting](#troubleshooting)) |
| `ALLOWED_ORIGINS` | frontend + Vite dev server origins | Comma-separated CORS allow-list for the backend |
| `MAX_IMAGE_MB` | `20` | Max accepted image upload size |
| `MAX_VIDEO_MB` | `200` | Max accepted video upload size |
| `VIDEO_SAMPLE_INTERVAL_SECONDS` | `1.0` | Seconds between sampled video frames |
| `VIDEO_MAX_DURATION_SECONDS` | `120` | Only sample within this many seconds of the video |
| `VIDEO_MAX_FRAMES` | `60` | Hard cap on frames analyzed per video |
| `FAKE_THRESHOLD` / `REAL_THRESHOLD` | not set (see below) | Override the decision thresholds |

Decision thresholds (`FAKE_THRESHOLD`/`REAL_THRESHOLD`) are **not** wired into
`docker-compose.yml` by default — see
[`backend/app/thresholds.json`](backend/app/thresholds.json) and
[`docs/evaluation-and-limitations.md`](docs/evaluation-and-limitations.md) for why the
checked-in defaults aren't a bare 0.5 cut-off, and how to run a real evaluation that updates
that file properly. To override via env instead, add the variables under `backend.environment`
in `docker-compose.yml`.

## Health checks & startup order

- `GET /health` — liveness; always 200 while the process is up, reports `model_state`
  (`loading` / `ready` / `failed`) and `model_error`.
- `GET /ready` — readiness; 200 only once the model has loaded *and* passed startup
  validation (correct `num_labels`, `num_attention_heads`, preprocessing config, and a
  smoke-test inference). Returns 503 (with the error) while loading or if loading failed.
  This is what Docker's `healthcheck:` polls.
- The frontend container starts immediately rather than waiting on the backend's health
  check, so the UI is reachable right away; if you try to analyze a file before the model has
  finished loading, the API returns a 503 and the UI shows it as a clear error with a
  "Try again" button — deliberately, so a slow first-run download doesn't leave you looking at
  a blank page.

## Model caching

The Hugging Face cache lives in the `model-cache` named volume, mounted at `/models`
(`HF_HOME`) in the backend container. It survives `docker compose down` and container
restarts/recreates; only `docker compose down -v` or manually removing the volume forces a
re-download. Verified in testing: `down -v` → `up` re-downloaded and validated the model in
13.9s; a normal `down` → `up` reused the cache and was ready in 0.7s.

## Tests

Run the backend test suite inside the backend container (no local Python needed):

```bash
docker compose run --rm --no-deps -v $PWD/backend:/srv backend \
  sh -c "pip install -q -r requirements-dev.txt && python -m pytest -q"
```

This includes mocked unit tests (thresholds, decode/validation, video aggregation,
evaluation-metrics lookup) plus two integration tests that run against the real loaded model
and auto-skip if it isn't available.

**Actually run for this change** (reported here as run, not assumed):
- `python -m pytest -q` → **65 passed**, including the 2 real-model integration tests.

## Manual verification performed

Everything below was run against this codebase, not assumed:

- **Clean build**: removed all project containers, images, and the `model-cache` volume
  (`docker compose down -v --rmi local`), then `docker compose up --build` — both containers
  built and started; both reported `healthy` in `docker compose ps`.
- **Model download on empty cache**: with the volume wiped, the model downloaded and passed
  startup validation in 13.9s; `/health` moved `loading` → `ready`.
- **`/docs`**: `GET /docs` → 200, `GET /openapi.json` → 200.
- **Frontend**: `GET /` on the frontend container → 200; proxied `GET /api/health` through
  nginx → 200 with the backend's JSON.
- **Image inference**: a real JPEG through `POST /detect/image` → 200 with
  prediction/probabilities/thresholds/inference time, both via curl and via a real browser
  (Playwright) clicking through the actual UI — REAL, confidence 99.5%, zero console errors.
- **Video inference**: a generated MP4 through `POST /detect/video` → 200, correct
  frame-count/aggregation, both via curl and the real browser UI.
- **Invalid inputs**: empty file → 400, wrong extension → 415, corrupted bytes with an
  image/video content-type and extension → 400, missing `file` field → 422 — for both the
  image and video endpoints.
- **Temp-file cleanup**: counted files under `/tmp` in the backend container before and after
  a video upload — unchanged, confirming the temp video file is always removed.
- **Backend failure**: stopped the backend container; the frontend kept serving (200), and the
  proxied `/api/health` returned a clear 502 rather than hanging. Restarted the backend and
  confirmed it came back `healthy`.
- **Model-loading failure**: set `MODEL_ID` to a nonexistent repo; `/health` reported
  `model_state: "failed"` with the underlying Hugging Face error text, `/ready`,
  `/model-info`, and `/detect/image` all returned 503 with that same reported state — no
  silent hang, no fabricated success. Restored the working `MODEL_ID` afterwards and confirmed
  recovery.
- **CORS**: the frontend's configured origin is in `ALLOWED_ORIGINS` by default; the Vite dev
  server origin (`:5173`) is included too so `npm run dev` against the containerized backend
  also works without extra config.

## Troubleshooting

- **"port is already allocated"** — something else on your machine is already using port
  3000/8000/etc. This project defaults the frontend to **3100** specifically to avoid clashing
  with other local projects on 3000; if 8000 or 3100 is still taken, set `BACKEND_PORT` /
  `FRONTEND_PORT` in `.env` to free ports and re-run `docker compose up --build`.
- **Backend stuck at `model_state: "loading"` for a long time** — this is normal on the very
  first run (model download); check `docker compose logs backend` for download/HTTP progress.
  If it never finishes, verify the container has internet access and that `MODEL_ID` /
  `MODEL_REVISION` are correct.
- **`model_state: "failed"`** — check `model_error` from `GET /health` (or the logs); it
  contains the underlying error (bad model id, network failure, a config that fails the
  startup validation checks). Fix the cause (e.g. `MODEL_ID`/`MODEL_REVISION` in `.env`) and
  run `docker compose up -d backend` to retry.
- **Frontend loads but every analysis fails with a network error** — check
  `docker compose ps` for the backend's health status, and confirm `ALLOWED_ORIGINS` includes
  whatever origin you're loading the frontend from if you're not using the default port.
- **Stale model after an upstream fix** — the model repo has previously shipped a broken
  config that the backend's startup validation explicitly checks for and rejects with a clear
  error; if you suspect a stale/broken cached copy, run `docker compose down -v` to force a
  clean re-download.
- **Full reset**: `docker compose down -v --rmi local` removes containers, the model cache,
  and the locally built images, so the next `docker compose up --build` is a genuinely clean
  build (this is exactly what was run to verify this release — see above).

## Known limitations

See [`docs/evaluation-and-limitations.md`](docs/evaluation-and-limitations.md) for the full,
current list — summarized:

- Detects AI-generated images, not face-swaps or other partial manipulations.
- No calibration study has been performed; probabilities are not claimed to be calibrated
  confidence values.
- Decision thresholds are documented placeholders until a real evaluation is run with
  `backend/scripts/evaluate.py` against a labelled dataset.
- Video analysis is independent per-frame image classification averaged together — not
  temporal/motion-based deepfake analysis.
- Evaluation accuracy shown in the UI/API is only ever a real, measured number from a run of
  `scripts/evaluate.py`; it is `null` until one has been run, never fabricated.
