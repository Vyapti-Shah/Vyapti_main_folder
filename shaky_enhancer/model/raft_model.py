"""
RAFT optical-flow model.

This class ONLY knows how to load weights and run inference on tensors.
It has no knowledge of video files, frame resizing, or numpy arrays —
that preprocessing belongs to service/raft_service.py. This keeps the
model swappable (see interfaces/optical_flow.py).
"""

from pathlib import Path

import torch
from torchvision.models.optical_flow import raft_large

from interfaces.optical_flow import OpticalFlowEstimator
from utils.logger import get_logger

logger = get_logger(__name__)


class RAFTModel(OpticalFlowEstimator):
    def __init__(self, weights_path: str, device: str | None = None):
        self.weights_path = weights_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None

    def load(self) -> None:
        logger.info("Using device: %s", self.device)

        weights_file = Path(self.weights_path)
        if not weights_file.exists():
            raise FileNotFoundError(
                f"RAFT weights not found at '{weights_file}'. "
                "Download/place the .pth checkpoint in the project root."
            )

        model = raft_large(weights=None, progress=False).to(self.device)
        state_dict = torch.load(weights_file, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        self._model = model.eval()

        logger.info("RAFT model loaded from %s", weights_file)

    def estimate(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("RAFTModel.load() must be called before estimate().")

        with torch.no_grad():
            flows = self._model(img1, img2)

        # RAFT returns a list of flow refinements; the last one is the
        # most accurate. Index [0] drops the batch dimension.
        return flows[-1][0]
