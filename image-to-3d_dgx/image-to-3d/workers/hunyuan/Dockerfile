FROM nvcr.io/nvidia/pytorch:25.09-py3
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="12.1"
RUN apt-get update && apt-get install -y --no-install-recommends git wget ninja-build libgl1 libglib2.0-0 libegl1 && rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /opt/Hunyuan3D-2.1
WORKDIR /opt/Hunyuan3D-2.1
RUN sed -i -E '/^(torch|torchvision|torchaudio|bpy)([=<>~ ].*)?$/d' requirements.txt \
 && pip install --no-cache-dir -r requirements.txt fastapi uvicorn python-multipart pybind11
RUN cd hy3dpaint/custom_rasterizer && pip install --no-cache-dir --no-build-isolation -e .
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
COPY server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9002"]
