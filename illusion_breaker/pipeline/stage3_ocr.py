import logging
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import numpy as np
import cv2
import os

logger = logging.getLogger(__name__)


class GotOcrReader:
    def __init__(self, config: dict):
        model_name = config["ocr"]["model_name"]
        configured_device = config["ocr"].get("device", "auto")

        if configured_device == "auto":
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = configured_device

        logger.info(f"Stage3 OCR using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
            use_safetensors=True,
        ).eval().to(self.device)

    def _prepare_for_ocr(self, mask_img: np.ndarray, target_height: int = 300) -> np.ndarray:
        """The incoming crop is a hard binary silhouette (pure 0/255,
        jagged edges from thresholding+morphology upstream), often quite
        small in pixel terms. OCR models like GOT-OCR2 are trained mostly
        on real rendered/scanned glyphs that are comfortably large and
        anti-aliased. At small native resolution, thin strokes (e.g. the
        stem of a "T" or "I") can look ambiguous or even get read as two
        separate strokes once blurred. Upscaling FIRST (while edges are
        still crisp) and then applying only a light blur on top of the
        larger image keeps strokes visually single and well-separated
        from their neighbors, instead of smearing them at a small scale."""
        h, w = mask_img.shape[:2]
        if h > 0 and h < target_height:
            scale = target_height / h
            mask_img = cv2.resize(
                mask_img, (max(1, int(w * scale)), target_height),
                interpolation=cv2.INTER_CUBIC
            )

        padded = cv2.copyMakeBorder(
            mask_img, 20, 20, 20, 20,
            borderType=cv2.BORDER_CONSTANT, value=0
        )
        smoothed = cv2.GaussianBlur(padded, (0, 0), sigmaX=1.2)
        return smoothed

    def process(self, mask_img: np.ndarray) -> str:
        """
        mask_img: clean binary mask from Stage 1/2 (isolated characters
        on a plain background). Plain 'ocr' mode is correct here — no
        layout/table/formatting to preserve.
        """
        mask_img = self._prepare_for_ocr(mask_img)
        tmp_path = f"_tmp_ocr_{os.getpid()}_{id(mask_img)}.png"
        original_tensor_cuda = torch.Tensor.cuda
        original_tensor_half = torch.Tensor.half
        original_tensor_to = torch.Tensor.to
        original_module_cuda = torch.nn.Module.cuda

        def patched_tensor_to(self, *args, **kwargs):
            args = tuple(
                "cpu" if isinstance(a, str) and "cuda" in a else a
                for a in args
            )
            if "device" in kwargs and isinstance(kwargs["device"], str) and "cuda" in kwargs["device"]:
                kwargs["device"] = "cpu"
            return original_tensor_to(self, *args, **kwargs)

        try:
            Image.fromarray(mask_img).convert("RGB").save(tmp_path)

            if self.device == "cpu":
                # GOT-OCR2's chat() / forward code hardcodes .cuda(), .half(),
                # and/or .to('cuda') internally regardless of the model's
                # actual device. On CPU-only machines these crash, so
                # neutralize them for the duration of this call only.
                # NOTE: this monkey-patches torch globally (not thread-locally).
                # This is only safe because run_batch.py enforces num_workers=1
                # when running on CPU (see orchestrator.py's startup check) —
                # do not raise num_workers above 1 for a CPU device without
                # reintroducing a lock around this block.
                torch.Tensor.cuda = lambda self, *args, **kwargs: self
                torch.Tensor.half = lambda self, *args, **kwargs: self.float()
                torch.Tensor.to = patched_tensor_to
                torch.nn.Module.cuda = lambda self, *args, **kwargs: self

            text = self.model.chat(self.tokenizer, tmp_path, ocr_type="ocr")
            return text.strip()
        except Exception:
            logger.exception("Stage3 OCR failed")
            raise
        finally:
            torch.Tensor.cuda = original_tensor_cuda
            torch.Tensor.half = original_tensor_half
            torch.Tensor.to = original_tensor_to
            torch.nn.Module.cuda = original_module_cuda
            if os.path.exists(tmp_path):
                os.remove(tmp_path)