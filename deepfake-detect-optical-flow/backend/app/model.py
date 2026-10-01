"""
Spatial deepfake detector: EfficientNet-B4 from SCLBD/DeepfakeBench (pretrained weights, release v1.0.1:
effnb4_best.pth), applied per face frame. Frame scores are aggregated per clip (flow-selected frames).
DeepfakeBench weights are CC BY-NC 4.0 (non-commercial use only).
"""
import os, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/effnb4_best.pth")
RES = 256                                   # DeepfakeBench input resolution
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class _Backbone(nn.Module):                 # mirrors DeepfakeBench training/networks/efficientnetb4.py
    def __init__(self):
        super().__init__()
        self.efficientnet = EfficientNet.from_name("efficientnet-b4")
        self.efficientnet._fc = nn.Identity()
        self.last_layer = nn.Linear(1792, 2)

    def forward(self, x):
        f = self.efficientnet.extract_features(x)
        return self.last_layer(F.adaptive_avg_pool2d(f, (1, 1)).flatten(1))


class DeepfakeNet(nn.Module):               # state-dict keys: backbone.efficientnet.*, backbone.last_layer.*
    def __init__(self):
        super().__init__()
        self.backbone = _Backbone()

    def forward(self, x):                   # x: N,3,256,256 -> P(fake) per frame (softmax class 1 = fake)
        return torch.softmax(self.backbone(x), dim=1)[:, 1]


_model, MODEL_LOADED = None, False


def get_model():
    global _model, MODEL_LOADED
    if _model is None:
        m = DeepfakeNet().eval()
        if os.path.exists(MODEL_PATH):
            try:
                sd = torch.load(MODEL_PATH, map_location="cpu")
                sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
                sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
                res = m.load_state_dict(sd, strict=True)
                MODEL_LOADED = True
                print(f"[model] loaded DeepfakeBench EfficientNet-B4 weights from {MODEL_PATH} ({res})")
            except Exception as e:
                print(f"[model] FAILED to load {MODEL_PATH}: {str(e)[:600]} -> DEMO MODE")
        else:
            print(f"[model] {MODEL_PATH} not found -> DEMO MODE (untrained)")
        _model = m
    return _model


def preprocess(bgr):
    x = cv2.resize(bgr, (RES, RES), interpolation=cv2.INTER_CUBIC)[:, :, ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(((x - MEAN) / STD).transpose(2, 0, 1).copy())


@torch.no_grad()
def predict_clip(crops):
    """crops: list of BGR face crops -> (clip P(fake), [P(fake) per frame]). Clip score = mean of frame scores."""
    m = get_model()
    p = m(torch.stack([preprocess(c) for c in crops])).tolist()
    return float(np.mean(p)), p