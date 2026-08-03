import cv2
import logging
import os
from .stage1_preprocess import StripePreprocessor
from .stage2_digit_classifier import DigitShapeClassifier
from .stage3_ocr import GotOcrReader

logger = logging.getLogger(__name__)


class CamoTextPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = config["paths"]["output_dir"]

        self._check_worker_safety(config)

        self.stage1 = StripePreprocessor(config)
        self.stage2 = DigitShapeClassifier(
            enable_ring_rules=config.get("classifier", {}).get("enable_ring_rules", True)
        )
        self.stage3 = GotOcrReader(config)

    @staticmethod
    def _check_worker_safety(config: dict):
        """stage3_ocr.py's CPU fallback works by monkey-patching torch
        globally (torch.Tensor.cuda/half/to, torch.nn.Module.cuda) for the
        duration of each OCR call, then reverting it. That patch is NOT
        thread-safe: with more than one worker thread, one thread can
        revert the patch while another thread's model call is still
        mid-inference, causing intermittent "Half/float mismatch" or
        "Torch not compiled with CUDA" errors. Rather than re-adding a
        lock, we fail fast at startup instead, so this footgun can't be
        reintroduced by silently bumping batch.num_workers in a config file."""
        num_workers = config.get("batch", {}).get("num_workers", 1)
        configured_device = config.get("ocr", {}).get("device", "auto")
        device = configured_device
        if configured_device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if num_workers > 1 and device == "cpu":
            raise ValueError(
                f"config has batch.num_workers={num_workers} with OCR running on CPU. "
                "stage3_ocr.py's CPU fallback monkey-patches torch globally and is only "
                "safe with a single worker. Set batch.num_workers: 1 for CPU runs, or "
                "run on a CUDA device if you need concurrency."
            )

    def _ocr_crop(self, crop) -> str:
        """Runs OCR on a small cropped region (used only for character
        groups the shape classifier can't handle via its ring rules)."""
        return self.stage3.process(crop)

    def run(self, image_path: str) -> dict:
        result = {"image_path": image_path, "status": "failed"}
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")

            revealed = self.stage1.process(img)

            out_img_path = os.path.join(self.output_dir, f"{base_name}_text.png")
            cv2.imwrite(out_img_path, revealed)

            try:
                detected_text = self.stage2.classify(revealed, ocr_fallback=self._ocr_crop)
            except Exception:
                logger.exception(f"Shape classification failed for {image_path}")
                detected_text = None

            result.update({
                "status": "success",
                "output_image_path": out_img_path,
                "detected_text": detected_text,
            })

        except Exception as e:
            logger.exception(f"Pipeline failed for {image_path}")
            result["error"] = str(e)
        return result