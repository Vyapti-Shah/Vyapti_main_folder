FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TF_CPP_MIN_LOG_LEVEL=2
# Pretrained deepfake detector (SCLBD/DeepfakeBench release v1.0.1, EfficientNet-B4, CC BY-NC 4.0)
ARG MODEL_URL="https://github.com/SCLBD/DeepfakeBench/releases/download/v1.0.1/effnb4_best.pth"
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip install --no-deps retina-face==0.0.17
# Pre-download RetinaFace weights at build time (non-fatal: fetched on first use otherwise)
RUN python -c "from retinaface import RetinaFace; RetinaFace.build_model(); print('RetinaFace weights cached')" \
    || echo "WARNING: RetinaFace weights not pre-downloaded"
# ffmpeg: writes the browser-playable annotated (watermarked) video
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* && mkdir -p /app/outputs /app/models
COPY app ./app
COPY models /app/models
# Download the pretrained deepfake detector (build fails loudly if the download fails; set MODEL_URL= to skip)
RUN if [ -n "$MODEL_URL" ]; then \
      python -c "import urllib.request,os,sys; p='/app/models/effnb4_best.pth'; urllib.request.urlretrieve(sys.argv[1], p); s=os.path.getsize(p); print('downloaded', s, 'bytes'); assert s>50_000_000, 'download too small'" "$MODEL_URL"; \
    fi
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]