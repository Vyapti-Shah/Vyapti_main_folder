FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=120 \
    torch==2.4.1 torchvision==0.19.1
RUN sed -i '/^sam2/d' requirements.txt && \
    pip install --no-cache-dir --default-timeout=120 -r requirements.txt

# SAM2 in editable mode so Hydra can find its yaml configs
RUN git clone --depth 1 https://github.com/facebookresearch/sam2.git /opt/sam2 && \
    SAM2_BUILD_CUDA=0 pip install --no-cache-dir -e /opt/sam2 --no-deps --no-build-isolation

# SAM2 checkpoint
RUN mkdir -p checkpoints && \
    curl -fL --retry 3 -o checkpoints/sam2.1_hiera_small.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt && \
    test $(stat -c%s checkpoints/sam2.1_hiera_small.pt) -gt 100000000

COPY app ./app

RUN mkdir -p /data/uploads /data/outputs

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]