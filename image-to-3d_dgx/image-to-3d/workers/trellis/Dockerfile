FROM nvcr.io/nvidia/pytorch:25.09-py3
ENV DEBIAN_FRONTEND=noninteractive TORCH_CUDA_ARCH_LIST="12.1" ATTN_BACKEND=sdpa SPCONV_ALGO=native PYTHONPATH=/opt/TRELLIS
RUN apt-get update && apt-get install -y --no-install-recommends git ninja-build libgl1 libglib2.0-0 libjpeg-dev && rm -rf /var/lib/apt/lists/*
RUN git clone --recurse-submodules https://github.com/microsoft/TRELLIS.git /opt/TRELLIS
RUN pip install --no-cache-dir pillow imageio imageio-ffmpeg tqdm easydict scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers fastapi uvicorn python-multipart \
 && pip install --no-cache-dir git+https://github.com/EasternJournalist/utils3d.git
RUN pip install --no-cache-dir --no-build-isolation git+https://github.com/NVIDIAGameWorks/kaolin.git \
 && pip install --no-cache-dir --no-build-isolation git+https://github.com/FindDefinition/cumm.git git+https://github.com/traveller59/spconv.git
RUN git clone https://github.com/NVlabs/nvdiffrast.git /tmp/nvdiffrast && pip install --no-cache-dir --no-build-isolation /tmp/nvdiffrast \
 && git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/diffoctreerast && pip install --no-cache-dir --no-build-isolation /tmp/diffoctreerast \
 && git clone https://github.com/autonomousvision/mip-splatting.git /tmp/mip-splatting && pip install --no-cache-dir --no-build-isolation /tmp/mip-splatting/submodules/diff-gaussian-rasterization/
WORKDIR /opt/TRELLIS
COPY server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "9001"]
