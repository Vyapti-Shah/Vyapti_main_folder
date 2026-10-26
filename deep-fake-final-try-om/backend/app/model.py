"""Loads and validates the CommunityForensics ViT detector (CPU, single-logit sigmoid)."""
import logging
import os
import threading
import time

import torch
import transformers
from packaging.version import Version
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

log = logging.getLogger("model")

MODEL_ID = os.getenv("MODEL_ID", "buildborderless/CommunityForensics-DeepfakeDet-ViT")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
MIN_TRANSFORMERS = Version("5.4.0")


class ModelValidationError(RuntimeError):
    pass


class Detector:
    def __init__(self):
        self.state = "loading"  # loading | ready | failed
        self.error: str | None = None
        self.info: dict = {}
        self.load_seconds: float | None = None
        self.model = None
        self.processor = None

    def load_async(self):
        threading.Thread(target=self._load, name="model-loader", daemon=True).start()

    def _load(self):
        t0 = time.time()
        try:
            threads = int(os.getenv("TORCH_NUM_THREADS", "0"))
            if threads > 0:
                torch.set_num_threads(threads)
            if Version(transformers.__version__) < MIN_TRANSFORMERS:
                raise ModelValidationError(
                    f"transformers {transformers.__version__} < {MIN_TRANSFORMERS} (required for correct preprocessing)"
                )
            log.info("Loading %s@%s", MODEL_ID, MODEL_REVISION)
            model = ViTForImageClassification.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            processor = ViTImageProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
            model.eval()
            self._validate(model, processor)
            self.model, self.processor = model, processor
            self.info = self._describe(model, processor)
            self.load_seconds = round(time.time() - t0, 2)
            self.state = "ready"
            log.info("Model ready in %.1fs", self.load_seconds)
        except Exception as e:  # keep the API alive and report the reason
            self.error = f"{type(e).__name__}: {e}"
            self.state = "failed"
            log.exception("Model loading failed")

    def predict(self, image: Image.Image) -> float:
        """Return (logit-derived) P(fake) for an RGB image: sigmoid of the single logit."""
        with torch.inference_mode():
            inputs = self.processor(image, return_tensors="pt")
            logit = self.model(**inputs).logits[0, 0]
            return float(torch.sigmoid(logit))

    @staticmethod
    def _validate(model, processor):
        cfg = model.config
        checks = {
            "num_labels == 1": cfg.num_labels == 1,
            "num_attention_heads == 6": cfg.num_attention_heads == 6,
            "intermediate_size == 1536": cfg.intermediate_size == 1536,
            "image_size == 384": cfg.image_size == 384,
            "processor shortest_edge == 440": getattr(processor, "size", {}).get("shortest_edge") == 440,
            "processor center-crop 384": bool(processor.do_center_crop) and processor.crop_size.get("height") == 384,
        }
        bad = [k for k, ok in checks.items() if not ok]
        if bad:
            raise ModelValidationError(
                "Model config looks like the pre-July-2026 broken copy; failed: " + ", ".join(bad)
                + ". Clear the model-cache volume (docker compose down -v) and retry."
            )
        # smoke test: non-square input exercises resize + center-crop, output must be one logit
        with torch.inference_mode():
            inputs = processor(Image.new("RGB", (640, 480), (128, 128, 128)), return_tensors="pt")
            if tuple(inputs["pixel_values"].shape) != (1, 3, 384, 384):
                raise ModelValidationError(f"unexpected pixel_values shape {tuple(inputs['pixel_values'].shape)}")
            logits = model(**inputs).logits
            if tuple(logits.shape) != (1, 1) or not torch.isfinite(logits).all():
                raise ModelValidationError(f"unexpected logits {tuple(logits.shape)} / non-finite values")

    @staticmethod
    def _describe(model, processor):
        c = model.config
        return {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "architecture": "ViT-Small (patch16, 384px)",
            "num_labels": c.num_labels,
            "num_attention_heads": c.num_attention_heads,
            "num_hidden_layers": c.num_hidden_layers,
            "hidden_size": c.hidden_size,
            "image_size": c.image_size,
            "parameters": sum(p.numel() for p in model.parameters()),
            "output": "single logit -> sigmoid = P(AI-generated / fake)",
            "preprocessing": "resize shortest edge 440 -> center crop 384 -> CLIP normalize",
            "device": "cpu",
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "scope_note": (
                "Trained to detect AI-generated images. It is not a general detector of every kind "
                "of face-swap or video manipulation, and its probability is not a guarantee of correctness."
            ),
        }


detector = Detector()
