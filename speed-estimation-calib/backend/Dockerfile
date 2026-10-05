FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 YOLO_CONFIG_DIR=/app/.ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
# CPU wheels keep the image small. For NVIDIA GPUs replace the index URL with the CUDA one.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt
ARG YOLO_MODEL=yolo11s.pt
ENV YOLO_MODEL=${YOLO_MODEL}
RUN python -c "from ultralytics import YOLO; YOLO('${YOLO_MODEL}')"
COPY bytetrack.yaml .
COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
