"""
Orchestrates RAFT-based flow estimation: resize -> tensor conversion ->
padding -> inference -> unpad -> rescale. The model itself (model/raft_model.py)
stays free of this preprocessing so it can be swapped out.
"""

import numpy as np

from config.settings import settings
from interfaces.optical_flow import OpticalFlowEstimator
from utils.image_utils import resize_pair
from utils.logger import get_logger
from utils.tensor_utils import frames_to_tensors, pad_to_multiple, unpad_flow

logger = get_logger(__name__)


class RaftService:
    def __init__(self, model: OpticalFlowEstimator):
        self.model = model
        self.device = getattr(model, "device", "cpu")

    def load_model(self) -> None:
        logger.info("Loading RAFT model...")
        self.model.load()

    def estimate_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns dense (u, v) flow arrays at the ORIGINAL frame resolution,
        even though inference internally runs on a downscaled copy.
        """
        frame1_small, frame2_small, scale = resize_pair(frame1, frame2, settings.MAX_RAFT_WIDTH)
        new_h, new_w = frame1_small.shape[:2]

        img1, img2 = frames_to_tensors(frame1_small, frame2_small, self.device)
        img1, img2 = pad_to_multiple(img1, img2, settings.FLOW_PADDING_MULTIPLE)

        flow = self.model.estimate(img1, img2)
        flow = unpad_flow(flow, new_h, new_w).cpu().numpy()

        u = flow[0] * (1.0 / scale)
        v = flow[1] * (1.0 / scale)
        return u, v
