import sys
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


class DeepFakeService:
    """
    Service class to handle the loading of the EfficientNet-B0 model,
    preprocessing of images, prediction, and watermarking.
    """
    def __init__(self, model_path: str, fake_class_index: int = 1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fake_class_index = fake_class_index
        print(f"Using device: {self.device}")
        self.model = self._load_model(model_path)

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
            # Handle possible nested state_dict
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            model.load_state_dict(state_dict, strict=False)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")
            sys.exit(1)

        model.to(self.device)
        model.eval()
        return model

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            sys.exit(1)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        img_tensor = transform(img).unsqueeze(0)
        return img_tensor

    def predict(self, image_path: str) -> bool:
        """
        Returns True if the image is predicted as Fake/AI Generated,
        False otherwise.
        """
        img_tensor = self._preprocess_image(image_path).to(self.device)

        print("Classifying image...")
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()

        print(f"Output Probabilities: {probabilities[0].cpu().numpy()}")
        return predicted_class == self.fake_class_index

    def add_watermark(self, image_path: str, output_path: str, is_real: bool):
        """
        Adds a text watermark to the top-left corner of the image.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error reading image with OpenCV: {image_path}")
            return

        text = "REAL" if is_real else "AI Generated"
        color = (0, 255, 0) if is_real else (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        x, y = 10, 30

        # Draw black background rectangle for text visibility
        cv2.rectangle(
            img,
            (x, y - text_height - 5),
            (x + text_width, y + baseline),
            (0, 0, 0),
            cv2.FILLED
        )

        # Draw text
        cv2.putText(
            img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA
        )

        cv2.imwrite(output_path, img)
        print(f"Watermarked image saved to: {output_path}")
