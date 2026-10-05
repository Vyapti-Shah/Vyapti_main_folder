FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from ultralytics import YOLO; YOLO('yolov8x-seg.pt'); YOLO('yolov8x-pose.pt')"
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
