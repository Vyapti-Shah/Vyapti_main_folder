FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" \
    ATTN_BACKEND=xformers SPCONV_ALGO=native PYTHONPATH=/opt/TRELLIS
RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build build-essential libgl1 libglib2.0-0 libjpeg-dev && rm -rf /var/lib/apt/lists/*
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /opt/TRELLIS
# Same dependency set as TRELLIS' setup.sh (which needs a visible GPU, so it can't run at build time)
RUN pip install --no-cache-dir pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers fastapi uvicorn python-multipart \
 && pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git \
 && pip install --no-cache-dir xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121 \
 && pip install --no-cache-dir kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html \
 && pip install --no-cache-dir spconv-cu120
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast \
 && git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && pip install --no-cache-dir --no-build-isolation /tmp/diffoctreerast \
 && git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && pip install --no-cache-dir --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/
WORKDIR /opt/TRELLIS
COPY server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9001"]
