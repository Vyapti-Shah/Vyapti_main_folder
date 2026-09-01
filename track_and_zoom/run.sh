#!/usr/bin/env bash
# Run Track & Zoom without Docker — two processes, for iterating on the code.
#
#   ./run.sh            start backend (18100) + frontend dev server (5173)
#   ./run.sh backend    backend only
#   ./run.sh frontend   frontend only
#
# Docker is the other option and needs no local Python/Node:
#   docker compose up -d --build   &&   open http://localhost:13082
set -euo pipefail
cd "$(dirname "$0")"

BACKEND_PORT="${TZ_BACKEND_PORT:-18100}"
# Everything stays inside this folder — no sibling project is referenced.
export TZ_SAM2_WEIGHTS_DIR="${TZ_SAM2_WEIGHTS_DIR:-$(pwd)/models/sam2}"
export TZ_DATA_DIR="${TZ_DATA_DIR:-$(pwd)/data}"

start_backend() {
  echo "backend  -> http://localhost:${BACKEND_PORT}   (weights: ${TZ_SAM2_WEIGHTS_DIR})"
  cd backend
  [ -d .venv ] || python3 -m venv .venv
  ./.venv/bin/pip install -q --upgrade pip
  ./.venv/bin/pip install -q -r requirements.txt
  # SAM2 separately — see the note in requirements.txt for why it is not listed
  # there. Skipped when already present so repeat runs stay fast.
  ./.venv/bin/python -c "import sam2" 2>/dev/null || \
    SAM2_BUILD_CUDA=0 ./.venv/bin/pip install -q \
      "git+https://github.com/facebookresearch/sam2.git@main"
  exec ./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${BACKEND_PORT}" --reload
}

start_frontend() {
  echo "frontend -> http://localhost:5173  (proxies /api to :${BACKEND_PORT})"
  cd frontend
  [ -d node_modules ] || npm install
  TZ_API="http://localhost:${BACKEND_PORT}" exec npm run dev
}

case "${1:-both}" in
  backend)  start_backend ;;
  frontend) start_frontend ;;
  both)
    ( start_backend ) & BACK=$!
    trap 'kill $BACK 2>/dev/null || true' EXIT INT TERM
    sleep 3
    start_frontend
    ;;
  *) echo "usage: $0 [backend|frontend|both]" >&2; exit 2 ;;
esac
