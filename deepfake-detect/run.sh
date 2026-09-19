#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT_DIR/data/uploads" "$ROOT_DIR/data/outputs"

cleanup() {
  kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT

echo "Starting backend (FastAPI --reload) on :8000"
(
  cd "$ROOT_DIR/backend"
  export MODEL_PATH="$ROOT_DIR/models/efficientnet_b0_ffpp_c23.pth"
  export UPLOADS_DIR="$ROOT_DIR/data/uploads"
  export OUTPUTS_DIR="$ROOT_DIR/data/outputs"
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
) &

echo "Starting frontend (vite dev) on :5173"
(
  cd "$ROOT_DIR/frontend"
  VITE_API_PROXY_TARGET="http://localhost:8000" npm run dev
) &

wait
