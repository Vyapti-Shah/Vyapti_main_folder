import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification


class GeneralAIImageDetector:
    """
    Whole-image real-vs-AI-generated classifier (Swin Transformer, trained on
    real photos vs. diffusion/GAN-generated images). Complements the
    FaceForensics++ face-swap model: that model looks for face-swap blending
    artifacts, this one looks for the holistic artifacts of fully synthetic
    generation (unnatural smoothness, lighting, texture) - the failure mode a
    face-swap detector was never trained for.
    """

    def __init__(
        self,
        model_name: str = "Organika/sdxl-detector",
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading general AI-image detector ({model_name})...")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.artificial_index = next(
            int(k) for k, v in self.model.config.id2label.items() if v == "artificial"
        )
        print("General AI-image detector loaded.")

    def predict(self, img: Image.Image) -> tuple[bool, float]:
        """Returns (is_artificial, confidence) for the given full image."""
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)[0]
        artificial_prob = probabilities[self.artificial_index].item()
        is_artificial = artificial_prob >= 0.5
        confidence = artificial_prob if is_artificial else (1 - artificial_prob)
        return is_artificial, confidence
