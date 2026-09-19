FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the general AI-image detector so it's baked into the image
# instead of being fetched from Hugging Face on every container start.
RUN python3 -c "\
from transformers import AutoImageProcessor, AutoModelForImageClassification; \
AutoImageProcessor.from_pretrained('Organika/sdxl-detector'); \
AutoModelForImageClassification.from_pretrained('Organika/sdxl-detector')"

# Weights are now baked into the image cache - never hit the network for them again.
ENV HF_HUB_OFFLINE=1

COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
