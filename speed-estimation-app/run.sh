#!/usr/bin/env bash
# Starts the app, preferring GPU acceleration (SAM2 tracking is much faster
# on GPU) and automatically falling back to the CPU setup when no NVIDIA GPU
# is present or the NVIDIA Container Toolkit isn't installed for Docker.
#
# Usage: ./run.sh [docker compose args...]   e.g. ./run.sh -d   or   ./run.sh down
set -euo pipefail
cd "$(dirname "$0")"

gpu_usable=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  if docker info 2>/dev/null | grep -qi "nvidia"; then
    gpu_usable=true
  else
    echo "NVIDIA GPU detected, but Docker isn't set up with the NVIDIA Container Toolkit -- falling back to CPU." >&2
    echo "Install it to use GPU acceleration: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
  fi
fi

args=("$@")
if [ ${#args[@]} -eq 0 ]; then
  args=(up --build)
fi

if [ "$gpu_usable" = true ]; then
  echo "GPU detected -- starting with GPU acceleration (SAM2 on CUDA)." >&2
  exec docker compose -f docker-compose.yml -f docker-compose.gpu.yml "${args[@]}"
else
  echo "No usable GPU -- starting on CPU (slower, especially the SAM2 tracking step)." >&2
  exec docker compose -f docker-compose.yml "${args[@]}"
fi
