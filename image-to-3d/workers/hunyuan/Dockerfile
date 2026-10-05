FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" CUDA_HOME=/usr/local/cuda
RUN apt-get update && apt-get install -y --no-install-recommends python3.10 python3.10-dev python3-pip git wget ninja-build build-essential libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libegl1 \
 && ln -sf /usr/bin/python3.10 /usr/bin/python && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /opt/Hunyuan3D-2.1
WORKDIR /opt/Hunyuan3D-2.1
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart pybind11
RUN cd hy3dpaint/custom_rasterizer && pip install --no-cache-dir --no-build-isolation -e .
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
COPY server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9002"]
