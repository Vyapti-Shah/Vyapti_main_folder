FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /srv

# shared libs opencv-python-headless needs for video decoding, even without GUI support
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch wheels (no CUDA libraries) keep the image small
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
