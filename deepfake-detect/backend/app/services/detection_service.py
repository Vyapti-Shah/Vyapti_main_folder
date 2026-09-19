import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from facenet_pytorch import MTCNN

from . import forensic_heuristics
from .general_ai_detector import GeneralAIImageDetector

# Bounded nudge applied to the general detector's fake-probability when a weak
# forensic heuristic fires. Small enough that it can never flip a confident
# verdict on its own - it can only tip an already-borderline (near 50/50) case.
HEURISTIC_NUDGE = 0.05


class DeepFakeService:
    """
    Ensemble deepfake/AI-image detection:
    1. MTCNN detects the largest face; EfficientNet-B0 (trained on
       FaceForensics++ face-swap forgeries) classifies the margin-expanded
       crop - catches face-swap/reenactment manipulation.
    2. A general Swin-based real-vs-AI-generated classifier runs on the full
       frame - catches fully synthetic (diffusion/GAN-generated) images that
       a face-swap detector was never trained to recognize.
    3. Weak forensic heuristics (missing camera EXIF, unnaturally smooth
       texture) nudge the general detector's score, but only enough to tip an
       already-borderline case - they don't independently decide the verdict.
    The final fake-probability is the average of the two models' scores, and
    the frame is flagged fake if that average is >= 0.5. Averaging (rather
    than flagging fake if either model alone crosses 0.5) avoids letting one
    noisy/borderline signal override a confident, correct call from the
    other model.
    """

    def __init__(
        self,
        model_path: str,
        fake_class_index: int = 1,
        face_margin: float = 0.3,
        general_detector_model: str = "Organika/sdxl-detector",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fake_class_index = fake_class_index
        self.face_margin = face_margin
        print(f"Using device: {self.device}")

        self.face_detector = MTCNN(keep_all=True, device=self.device)
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.general_detector = GeneralAIImageDetector(
            model_name=general_detector_model, device=self.device
        )

    def _load_model(self, model_path: str) -> nn.Module:
        print("Loading EfficientNet-B0 model...")
        try:
            model = models.efficientnet_b0(weights=None)
        except TypeError:
            model = models.efficientnet_b0(pretrained=False)

        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            raise

        model.to(self.device)
        model.eval()
        return model

    def _largest_face_box(
        self, img: Image.Image
    ) -> tuple[int, int, int, int] | None:
        """
        Detect faces and return the largest one (by area), expanded by
        face_margin and clamped to image bounds. Only the most prominent face
        is used so incidental background faces (crowd, bystanders) - typically
        tiny and low quality - can't override the verdict on the main subject.
        """
        boxes, _ = self.face_detector.detect(img)
        if boxes is None:
            return None

        width, height = img.size
        x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        mx = (x2 - x1) * self.face_margin
        my = (y2 - y1) * self.face_margin
        return (
            max(0, int(x1 - mx)),
            max(0, int(y1 - my)),
            min(width, int(x2 + mx)),
            min(height, int(y2 + my)),
        )

    def _classify(self, img: Image.Image) -> tuple[bool, float]:
        """Classify a single (already-cropped) image with EfficientNet-B0."""
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return predicted_class == self.fake_class_index, confidence

    def predict(self, image_path: str) -> tuple[bool, float]:
        """
        Ensemble verdict for one frame/image: face-swap model + general
        AI-image model + weak forensic heuristics. See class docstring.
        """
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        box = self._largest_face_box(img)
        face_img = img.crop(box) if box is not None else img
        if box is None:
            print("No face detected; classifying full frame.")

        is_fake_face, conf_face = self._classify(face_img)
        p_fake_face = conf_face if is_fake_face else (1 - conf_face)

        is_artificial, conf_general = self.general_detector.predict(img)
        p_fake_general = conf_general if is_artificial else (1 - conf_general)

        if forensic_heuristics.has_camera_exif(image_path):
            p_fake_general -= HEURISTIC_NUDGE
        else:
            p_fake_general += HEURISTIC_NUDGE
        if forensic_heuristics.is_unnaturally_smooth(face_img):
            p_fake_general += HEURISTIC_NUDGE
        p_fake_general = min(1.0, max(0.0, p_fake_general))

        p_fake_combined = (p_fake_face + p_fake_general) / 2
        is_fake = p_fake_combined >= 0.5
        confidence = p_fake_combined if is_fake else (1 - p_fake_combined)

        print(
            f"p_fake_face={p_fake_face:.3f} p_fake_general={p_fake_general:.3f} "
            f"combined={p_fake_combined:.3f} -> is_fake={is_fake}"
        )

        return is_fake, confidence

    def add_watermark(
        self, image_path: str, output_path: str, is_real: bool
    ) -> None:
        """Adds a text watermark to the top-left corner of the image."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image with OpenCV: {image_path}")
            raise ValueError(f"Could not read image: {image_path}")

        text = "REAL" if is_real else "AI Generated"
        color = (0, 255, 0) if is_real else (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = 10, 30

        cv2.rectangle(
            img,
            (x, y - text_height - 5),
            (x + text_width, y + baseline),
            (0, 0, 0),
            cv2.FILLED,
        )

        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, img)
        print(f"Watermarked image saved to: {output_path}")
