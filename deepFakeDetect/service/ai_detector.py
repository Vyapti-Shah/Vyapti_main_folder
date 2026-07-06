import sys

import cv2
import numpy as np
import torch
from PIL import Image


class AIDetectorService:
    """Manages the Deepfake Detection Model and runs inference."""

    def __init__(
        self,
        model_name: str = "prithivMLmods/Deep-Fake-Detector-Model",
        device: str = 'cuda'
    ):
        """
        Initializes the AI Detector Service.

        Args:
            model_name (str): The Hugging Face model ID.
            device (str): The device to run inference on ('cuda' or 'cpu').
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[+] Initializing AI Detector on {self.device.upper()}...")
        print(f"    Loading model: {model_name}")

        try:
            from transformers import pipeline
            device_id = 0 if self.device == 'cuda' else -1
            self.classifier = pipeline(
                "image-classification",
                model=model_name,
                device=device_id
            )
        except ImportError:
            print("  -> ERROR: 'transformers' or 'torch' is not installed.")
            print("     Please run: pip install transformers torch pillow scipy")
            sys.exit(1)
        except Exception as e:
            print(f"  -> ERROR: Failed to load model. {e}")
            sys.exit(1)

    def analyze_frame(self, img_bgr: np.ndarray) -> str:
        """
        Analyzes a single BGR frame and returns the predicted label.

        Args:
            img_bgr (np.ndarray): The input image in BGR format (OpenCV default).

        Returns:
            str: "AI GENERATED" or "REAL".
        """
        # Convert OpenCV BGR image to PIL RGB Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Run inference
        results = self.classifier(pil_img)
        
        # We take the top result.
        top_result = results[0]
        label = top_result['label'].upper()
        
        # Standardize labels
        if "FAKE" in label or "AI" in label or "SPOOF" in label:
            return "AI GENERATED"
        else:
            return "REAL"
