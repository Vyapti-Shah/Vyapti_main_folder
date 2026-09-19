FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ffmpeg is not optional here: the tracker dumps frames with it and both render
# steps re-encode with it. exiftool/etc from the parent image are not needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev ffmpeg git libgl1 "libglib2.0-0*" \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch FIRST, from the CUDA index — installing it via requirements.txt would
# resolve the CPU wheel and the tracker would silently run on CPU.
RUN pip install --upgrade pip && \
    pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
# Torch/torchvision are already satisfied by the CUDA install above, so this
# only adds the small deps.
RUN pip install -r requirements.txt

# SAM2 last, and on its own line. Two reasons it cannot live in
# requirements.txt: a `sam2 @ git+...` direct reference makes pip look for
# `sam2` on PyPI and fail with "No matching distribution found", and the build
# needs SAM2_BUILD_CUDA=0 — compiling its CUDA extension at image-build time
# requires a GPU, which `docker build` does not have. The runtime path used
# here does not need that extension.
RUN SAM2_BUILD_CUDA=0 pip install "git+https://github.com/facebookresearch/sam2.git@main"

COPY app ./app
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
