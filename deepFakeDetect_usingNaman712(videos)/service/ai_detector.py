import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image


class AIDetectorService:
    """Manages the Deepfake Detection Model and runs inference."""

    def __init__(
        self,
        model_name: str = "resnext-lstm-untrained",
        device: str = 'cuda',
        threshold: float = 0.70
    ):
        """
        Initializes the Hugging Face pipeline or custom un-trained PyTorch architecture.
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"[+] Initializing AI Detector on {self.device.upper()}...")
        self.model_name = model_name
        self.threshold = threshold
        self.is_custom_model = False
        self.custom_model = None

        if model_name == "mkfanet-untrained":
            from .models.mkfanet import MkfaNet
            self.custom_model = MkfaNet(num_classes=2)
            weights_file = "mkfanet_trained_weights.pth"
            if torch.cuda.is_available():
                self.custom_model = self.custom_model.cuda()
            if os.path.exists(weights_file):
                print(f"[*] Loading trained weights from {weights_file}...")
                self.custom_model.load_state_dict(torch.load(weights_file, map_location=self.device))
            else:
                print(f"[*] Loading native untrained MkfaNet Architecture...")
            self.custom_model.eval()
            self.is_custom_model = True
            return
            
        if model_name == "resnext-lstm-untrained":
            from .models.resnext_lstm import ResNeXtLSTMDetector
            self.custom_model = ResNeXtLSTMDetector(num_classes=2)
            weights_file = "resnext-lstm_trained_weights.pth"
            if torch.cuda.is_available():
                self.custom_model = self.custom_model.cuda()
            if os.path.exists(weights_file):
                print(f"[*] Loading trained weights from {weights_file}...")
                self.custom_model.load_state_dict(torch.load(weights_file, map_location=self.device))
            else:
                print(f"[*] Loading native untrained ResNeXt50+LSTM Architecture...")
            self.custom_model.eval()
            self.is_custom_model = True
            return

        print(f"[*] Loading Hugging Face Pipeline: {model_name}...")
        try:
            from transformers import pipeline
            self.classifier = pipeline("image-classification", model=model_name)
        except Exception as e:
            print(f"  -> ERROR: Failed to load model. {e}")
            sys.exit(1)

    def analyze_frame(self, img_bgr: np.ndarray) -> tuple[str, float, dict]:
        """
        Analyzes a single BGR frame and returns the predicted label, confidence, and all scores.

        Args:
            img_bgr (np.ndarray): The input image in BGR format (OpenCV default).

        Returns:
            tuple: (Final Label, Confidence Score, Dictionary of all class scores)
        """
        # Convert OpenCV BGR image to PIL RGB Image
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # Run inference
        if self.is_custom_model:
            # We must process the single frame and pass it to the PyTorch model
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Convert to Tensor and add Batch dimension
            input_tensor = transform(pil_img).unsqueeze(0).to(self.device)
            
            # If it's the LSTM model, add a sequence dimension (Batch, Seq=1, C, H, W)
            if self.model_name == "resnext-lstm-untrained":
                input_tensor = input_tensor.unsqueeze(1)
                
            with torch.no_grad():
                logits = self.custom_model(input_tensor)
                probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
                
            # Naman712 mapping: class 0 = FAKE, class 1 = REAL
            fake_prob = probs[0]
            real_prob = probs[1]
            
            confidence = fake_prob if fake_prob > real_prob else real_prob
            label = "AI GENERATED" if fake_prob > real_prob else "REAL"
            
            confidence_matrix = {"REAL": real_prob, "FAKE": fake_prob}
            final_label = label
            
            # Apply thresholding
            if final_label == "AI GENERATED" and confidence < self.threshold:
                final_label = "REAL"
                
            return final_label, float(confidence), {k: float(v) for k, v in confidence_matrix.items()}
            
        results = self.classifier(pil_img)
        
        # Results is usually a list of dicts: [{'label': 'Fake', 'score': 0.98}, ...]
        top_result = results[0]
        label = top_result['label'].upper()
        confidence = top_result['score']
        
        # Format confidence matrix
        confidence_matrix = {res['label']: res['score'] for res in results}
        
        # Standardize labels
        if "FAKE" in label or "AI" in label or "SPOOF" in label or "REALISM" not in label:
            final_label = "AI GENERATED" if "REAL" not in label else "REAL"
        else:
            final_label = "REAL"
            
        # Hard override for specific models like Siglip which output different formats
        if "FAKE" in label:
            final_label = "AI GENERATED"
        elif "REAL" in label:
            final_label = "REAL"
            
        # --- NO-TRAINING FALSE POSITIVE FIX (THRESHOLDING) ---
        # If the model thinks it's AI, but its confidence is lower than our strict threshold (e.g. 70%),
        # we overrule the model and declare it REAL.
        # This prevents the 50-60% false-positives the user was experiencing.
        if final_label == "AI GENERATED" and confidence < self.threshold:
            print(f"  [!] Overruled: Model said AI but confidence ({confidence*100:.1f}%) was below {self.threshold*100}% threshold. Setting to REAL.")
            final_label = "REAL"
            
        return final_label, confidence, confidence_matrix

    def analyze_sequence(self, frames_bgr: list[np.ndarray]) -> tuple[str, float, dict]:
        """
        Analyzes a sequence of BGR frames (e.g., 20 frames for LSTM) and returns a single final label.
        """
        if not self.is_custom_model or self.model_name != "resnext-lstm-untrained":
            # Fallback to analyzing just the middle frame if it's not a sequence model
            mid_idx = len(frames_bgr) // 2
            return self.analyze_frame(frames_bgr[mid_idx])

        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensor_list = []
        for img_bgr in frames_bgr:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            tensor_list.append(transform(pil_img))
            
        # Stack into (Seq, C, H, W)
        seq_tensor = torch.stack(tensor_list)
        # Add batch dimension: (Batch=1, Seq=N, C, H, W)
        input_tensor = seq_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.custom_model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            
        # Naman712 mapping: class 0 = FAKE, class 1 = REAL
        fake_prob = float(probs[0])
        real_prob = float(probs[1])
        
        confidence = fake_prob if fake_prob > real_prob else real_prob
        label = "AI GENERATED" if fake_prob > real_prob else "REAL"
        
        confidence_matrix = {"REAL": real_prob, "FAKE": fake_prob}
        final_label = label
        
        # Apply thresholding
        if final_label == "AI GENERATED" and confidence < self.threshold:
            print(f"  [!] Sequence Overruled: Model said AI but confidence ({confidence*100:.1f}%) was below {self.threshold*100}% threshold. Setting to REAL.")
            final_label = "REAL"
            
        return final_label, confidence, confidence_matrix
